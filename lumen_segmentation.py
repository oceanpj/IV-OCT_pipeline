"""
lumen_segmentation.py
---------------------
Segments the lumen in intravascular OCT images.

OCT image characteristics:
  - DICOM shape : (N_frames, 512, 512, 3)  — RGB, uint8
  - Catheter    : sits at the image centre, appears as a bright ring then dark core
  - Lumen       : dark region immediately outside the catheter
  - Vessel wall : bright ring surrounding the lumen
  - The wall is DISCONTINUOUS (shadows, plaques) but the lumen itself is smooth
  - The lumen may be off-centre relative to the image (catheter not always centred
    in the vessel)

Strategy
--------
  1. Grayscale + mild denoise per frame
  2. Polar transform from image centre (catheter centre)
  3. For each angle: scan radially outward past the catheter dead-zone and find
     the FIRST strong dark→bright transition — that is the lumen inner wall
  4. Circular (wrap-around) cubic interpolation bridges gaps in the boundary
  5. Gaussian smoothing gives a smooth, closed-ish contour
  6. Fill polar interior → back to Cartesian + morphological closing for holes
  7. Debug PNGs for a handful of frames + full DICOM mask volume saved out

Output
------
  lumen_output/
    lumen_debug_XXXX.png   — 5-panel visual per debug frame
    lumen_coverage.png     — area coverage per frame
    lumen_mask.dcm         — binary mask volume  (1 = lumen, 0 = outside)
"""

import os
import datetime
import numpy as np
import cv2
import pydicom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates, gaussian_filter1d
from scipy.interpolate import interp1d
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import generate_uid

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_DIR   = "/Users/oceanpunsalan/Data/Intravascular/20_raw"
OUTPUT_DIR = "lumen_output"

# Pick which case to test on (first one found if None)
CASE_ID = None   # e.g. "B36087B8-BD7C-4853-8156-99A2733E125C"

# ── Algorithm parameters ─────────────────────────────────────────────────────
THETA_STEPS      = 720    # angular samples around the circle
CATHETER_SKIP    = 45     # ignore everything closer than this (px) — catheter dead-zone
MAX_SEARCH_R     = 230    # don't look for wall beyond this radius (px)
PROFILE_SMOOTH_K = 11     # box-filter width to denoise radial profile before gradient
GRAD_THRESHOLD   = 0.25   # fraction of profile's local max to call a real edge
BOUNDARY_SMOOTH  = 9.0    # Gaussian sigma (in angle-index units) for final smoothing
MAX_GAP_DEG      = 90     # biggest gap (degrees) we'll interpolate across
CLOSE_KERNEL     = 17     # morphological closing kernel size to patch holes in mask

DEBUG_FRAMES = (10, 50, 100, 150)   # which frames to save debug figures for


# ══════════════════════════════════════════════════════════════════════════════
#  I/O helpers
# ══════════════════════════════════════════════════════════════════════════════

def find_dcm(case_id=None):
    """Return path to the round_color DICOM for one case."""
    entries = sorted(os.listdir(DATA_DIR))
    cases   = [e for e in entries if os.path.isdir(os.path.join(DATA_DIR, e))]
    if not cases:
        raise RuntimeError(f"No case folders found in {DATA_DIR}")
    if case_id is None:
        case_id = cases[0]
    dcm = os.path.join(DATA_DIR, case_id, "round_color", f"{case_id}.dcm")
    if not os.path.exists(dcm):
        raise FileNotFoundError(f"DICOM not found: {dcm}")
    return dcm, case_id


def load_frames(dcm_path):
    """
    Load pixel data from a multi-frame DICOM.
    Returns (N, H, W) uint8 grayscale array.
    """
    px = pydicom.dcmread(dcm_path).pixel_array          # (N, H, W, 3)
    print(f"  raw pixel array : {px.shape}  {px.dtype}")
    gray = np.stack(
        [cv2.cvtColor(px[i], cv2.COLOR_RGB2GRAY) for i in range(px.shape[0])],
        axis=0
    )
    return gray, px   # also keep RGB for overlays


# ══════════════════════════════════════════════════════════════════════════════
#  Polar ↔ Cartesian  (vectorised, no loops)
# ══════════════════════════════════════════════════════════════════════════════

def to_polar(image, center, r_max):
    """
    Remap a (H, W) image into polar coordinates (r_max, THETA_STEPS).
    center = (cy, cx) in pixel units.
    """
    rs     = np.arange(r_max, dtype=np.float32)
    thetas = np.linspace(0, 2 * np.pi, THETA_STEPS, endpoint=False).astype(np.float32)
    r_grid, t_grid = np.meshgrid(rs, thetas, indexing='ij')       # (r_max, T)
    xs = center[1] + r_grid * np.cos(t_grid)
    ys = center[0] + r_grid * np.sin(t_grid)
    return map_coordinates(image.astype(np.float32),
                           [ys, xs], order=1, mode='constant', cval=0)


def to_cartesian(polar, output_shape, center):
    """Map a (r_max, THETA_STEPS) polar array back to (H, W) Cartesian."""
    h, w = output_shape
    r_max = polar.shape[0]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    dy = ys - center[0];  dx = xs - center[1]
    r     = np.sqrt(dy**2 + dx**2).clip(0, r_max - 1)
    theta = np.arctan2(dy, dx) % (2 * np.pi)
    t_idx = theta / (2 * np.pi) * THETA_STEPS
    return map_coordinates(polar.astype(np.float32),
                           [r, t_idx], order=1, mode='constant', cval=0)


# ══════════════════════════════════════════════════════════════════════════════
#  Lumen boundary detection in polar space
# ══════════════════════════════════════════════════════════════════════════════

def detect_boundary(polar):
    """
    For every angle column find the radius of the lumen wall
    (= first strong dark→bright edge going outward from the catheter).

    Returns (THETA_STEPS,) float array; NaN where no edge found.
    """
    r_max, n_ang = polar.shape
    boundary = np.full(n_ang, np.nan)

    r0 = CATHETER_SKIP
    r1 = min(MAX_SEARCH_R, r_max - 1)

    for col in range(n_ang):
        profile = polar[r0:r1, col].astype(np.float32)
        n = len(profile)
        if n < 8:
            continue

        # box-smooth to kill speckle
        k = min(PROFILE_SMOOTH_K, n)
        if k % 2 == 0:
            k -= 1
        sm = np.convolve(profile, np.ones(k) / k, mode='same')

        # first-derivative: positive peak = rising brightness = wall inner face
        grad = np.gradient(sm)

        # adaptive threshold: fraction of the brightest gradient in this profile
        peak = grad.max()
        if peak <= 0:
            continue
        thresh = GRAD_THRESHOLD * peak

        hits = np.where(grad >= thresh)[0]
        if hits.size == 0:
            continue

        # take the innermost hit — closest wall to the lumen
        boundary[col] = float(r0 + hits[0])

    return boundary


# ══════════════════════════════════════════════════════════════════════════════
#  Gap-filling + smoothing  (handles circular wrap-around)
# ══════════════════════════════════════════════════════════════════════════════

def smooth_boundary(boundary):
    """
    1. Circular cubic interpolation across gaps ≤ MAX_GAP_DEG degrees.
    2. Gaussian smooth for overall roundness.

    Returns (THETA_STEPS,) float; NaN only where gaps were too large to bridge.
    """
    n          = len(boundary)
    max_gap_px = int(round(MAX_GAP_DEG / 360.0 * n))

    valid = np.where(~np.isnan(boundary))[0]
    if len(valid) < 6:
        return boundary.copy()

    # extend indices/values on both sides so interp1d sees a continuous signal
    ext_idx = np.concatenate([valid - n, valid, valid + n])
    ext_val = np.tile(boundary[valid], 3)
    order   = np.argsort(ext_idx)
    ext_idx = ext_idx[order];  ext_val = ext_val[order]

    f     = interp1d(ext_idx, ext_val, kind='cubic',
                     bounds_error=False, fill_value='extrapolate')
    filled = f(np.arange(n))

    # re-NaN gaps that were too wide
    for i in range(n):
        if np.isnan(boundary[i]):
            circ_dist = min(np.min(np.abs(valid - i)),
                            n - np.max(np.abs(valid - i)) if len(valid) else n)
            if circ_dist > max_gap_px:
                filled[i] = np.nan

    # Gaussian smooth (mode='wrap' respects the circular nature)
    ok = ~np.isnan(filled)
    if ok.sum() > 20:
        tmp = filled.copy()
        tmp[ok] = gaussian_filter1d(filled[ok], BOUNDARY_SMOOTH, mode='wrap')
        return tmp

    return filled


# ══════════════════════════════════════════════════════════════════════════════
#  Mask assembly
# ══════════════════════════════════════════════════════════════════════════════

def make_mask(boundary_smooth, polar_shape, image_shape, center):
    """
    Fill polar interior → Cartesian → morphological closing.
    Returns (H, W) uint8 binary mask.
    """
    r_max = polar_shape[0]
    polar_mask = np.zeros((r_max, THETA_STEPS), dtype=np.float32)

    for col in range(THETA_STEPS):
        if not np.isnan(boundary_smooth[col]):
            r = int(np.clip(np.round(boundary_smooth[col]), 0, r_max - 1))
            polar_mask[:r, col] = 1.0

    cart = to_cartesian(polar_mask, image_shape, center)
    mask = (cart > 0.4).astype(np.uint8)

    # close holes from NaN gaps / interpolation artefacts
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KERNEL, CLOSE_KERNEL))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    return mask


# ══════════════════════════════════════════════════════════════════════════════
#  Per-frame pipeline
# ══════════════════════════════════════════════════════════════════════════════

def process_frame(gray):
    h, w   = gray.shape
    center = (h / 2.0, w / 2.0)
    r_max  = int(min(center[0], center[1]))

    polar            = to_polar(gray, center, r_max)
    boundary_raw     = detect_boundary(polar)
    boundary_smooth  = smooth_boundary(boundary_raw)
    mask             = make_mask(boundary_smooth, polar.shape, (h, w), center)

    return mask, boundary_raw, boundary_smooth, polar, center


# ══════════════════════════════════════════════════════════════════════════════
#  Debug figure
# ══════════════════════════════════════════════════════════════════════════════

def save_debug(fidx, rgb, gray, mask, b_raw, b_smooth, polar, out_dir):
    fig, axes = plt.subplots(1, 5, figsize=(30, 6))
    fig.suptitle(f"Frame {fidx} — lumen segmentation", fontsize=12)

    axes[0].imshow(rgb);  axes[0].set_title("OCT (colour)")

    axes[1].imshow(polar, aspect='auto', cmap='gray', origin='lower')
    axes[1].set_title("Polar space");  axes[1].set_xlabel("angle");  axes[1].set_ylabel("r (px)")

    axes[2].imshow(polar, aspect='auto', cmap='gray', origin='lower')
    t = np.arange(THETA_STEPS)
    ok_r = ~np.isnan(b_raw);    ok_s = ~np.isnan(b_smooth)
    axes[2].scatter(t[ok_r], b_raw[ok_r],    c='red',  s=1, label='raw')
    axes[2].plot   (t[ok_s], b_smooth[ok_s], c='cyan', lw=1, label='smooth')
    axes[2].legend(fontsize=7);  axes[2].set_title("Boundary (polar)")

    axes[3].imshow(mask, cmap='gray');  axes[3].set_title("Lumen mask")

    axes[4].imshow(rgb)
    axes[4].imshow(np.ma.masked_where(mask == 0, mask),
                   cmap='cool', alpha=0.5)
    axes[4].set_title("Overlay")

    for ax in axes: ax.axis('off')
    axes[1].axis('on');  axes[2].axis('on')

    plt.tight_layout()
    path = os.path.join(out_dir, f"lumen_debug_{fidx:04d}.png")
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close('all')
    print(f"  saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  DICOM save
# ══════════════════════════════════════════════════════════════════════════════

def save_dicom(volume, out_path):
    """Save (N, H, W) uint8 mask as multi-frame DICOM."""
    n, h, w = volume.shape
    now     = datetime.datetime.now()

    fm = FileMetaDataset()
    fm.MediaStorageSOPClassUID    = '1.2.840.10008.5.1.4.1.1.2'
    fm.MediaStorageSOPInstanceUID = generate_uid()
    fm.TransferSyntaxUID          = '1.2.840.10008.1.2.1'

    ds = Dataset()
    ds.file_meta = fm;  ds.is_implicit_VR = False;  ds.is_little_endian = True
    ds.SOPClassUID       = fm.MediaStorageSOPClassUID
    ds.SOPInstanceUID    = fm.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID  = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.InstanceNumber    = '1'
    ds.StudyDate         = now.strftime('%Y%m%d')
    ds.StudyTime         = now.strftime('%H%M%S')
    ds.Modality          = 'OT'
    ds.PatientName       = 'Anonymous';  ds.PatientID        = 'ANON001'
    ds.PatientBirthDate  = '';           ds.PatientSex        = ''
    ds.Rows                      = h;    ds.Columns           = w
    ds.NumberOfFrames            = n
    ds.SamplesPerPixel           = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.BitsAllocated = 8;  ds.BitsStored = 8;  ds.HighBit = 7
    ds.PixelRepresentation       = 0
    ds.PixelData = (volume * 255).astype(np.uint8).tobytes()

    pydicom.dcmwrite(out_path, ds, write_like_original=False)
    print(f"  DICOM saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dcm_path, case_id = find_dcm(CASE_ID)
    print(f"Case   : {case_id}")
    print(f"DICOM  : {dcm_path}")

    gray_vol, rgb_vol = load_frames(dcm_path)
    n, h, w = gray_vol.shape
    print(f"Frames : {n}   Size : {h}×{w}\n")

    masks = np.zeros((n, h, w), dtype=np.uint8)

    for i in range(n):
        mask, b_raw, b_smooth, polar, center = process_frame(gray_vol[i])
        masks[i] = mask

        n_det  = int((~np.isnan(b_raw)).sum())
        n_fill = int((~np.isnan(b_smooth)).sum())

        if i % 50 == 0:
            print(f"  frame {i:3d}/{n}  detected {n_det}/{THETA_STEPS} angles  "
                  f"→ filled {n_fill}/{THETA_STEPS} "
                  f"({100*n_fill/THETA_STEPS:.0f}%)")

        if i in DEBUG_FRAMES:
            save_debug(i, rgb_vol[i], gray_vol[i],
                       mask, b_raw, b_smooth, polar, OUTPUT_DIR)

    # coverage plot
    cov = masks.sum(axis=(1, 2)) / (h * w) * 100
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(cov, lw=0.9)
    ax.set(xlabel="Frame", ylabel="Lumen area (% of image)",
           title="Lumen mask coverage per frame")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "lumen_coverage.png"), dpi=120)
    plt.close('all')

    # DICOM
    save_dicom(masks, os.path.join(OUTPUT_DIR, "lumen_mask.dcm"))
    print("\nAll done.")


if __name__ == "__main__":
    main()
