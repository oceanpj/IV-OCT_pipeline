"""
oct_pipeline.py
---------------
Polar-space vessel wall segmentation for intravascular OCT.

Pipeline per slice:
  1. Load NIfTI annotation → fill wedge in polar space  (outer boundary)
  2. Load DICOM frame → convert to polar (same coordinate space)
  3. For each angle: find inner edge of bright vessel wall via gradient peak
  4. Final mask = region between detected inner wall and annotation outer boundary
  5. Convert back to Cartesian → save as multi-frame DICOM
"""

import os, sys, datetime, uuid
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import pydicom, cv2
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import generate_uid
from scipy.ndimage import map_coordinates

# ── Config ────────────────────────────────────────────────────────────────────
NIFTI_PATH = (
    "/Users/oceanpunsalan/Library/Mobile Documents/"
    "com~apple~CloudDocs/Data/Intravascular/IntraVascular/"
    "0B360D4D-3B16-4DCC-AD86-32361D1B47A9.nii"
)
DCM_PATH = (
    "/Users/oceanpunsalan/Library/Mobile Documents/"
    "com~apple~CloudDocs/Data/Intravascular/IntraVascular/"
    "0B360D4D-3B16-4DCC-AD86-32361D1B47A9/round_color/"
    "0B360D4D-3B16-4DCC-AD86-32361D1B47A9.dcm"
)

CENTER_RADIUS             = 40     # catheter radius to skip (pixels)
THETA_STEPS               = 720    # angular resolution
WALL_THRESHOLD_FRACTION   = 0.35   # brightness fraction of peak to call vessel wall
SMOOTH_KERNEL             = 7      # smoothing window for intensity profile
EXPLICIT_VR_LITTLE_ENDIAN = '1.2.840.10008.1.2.1'
OUTPUT_DIR                = "pipeline_output"

try:
    generate_uid()
except Exception:
    def generate_uid():
        return '2.25.' + str(uuid.uuid4().int)


# ══════════════════════════════════════════════════════════════════════════════
#  Coordinate transforms
# ══════════════════════════════════════════════════════════════════════════════

def cartesian_to_polar(image, center=None, r_max=None, theta_steps=THETA_STEPS):
    h, w = image.shape
    if center is None:
        center = (h / 2.0, w / 2.0)
    if r_max is None:
        r_max = int(min(center[0], center[1], h - center[0], w - center[1]))
    rs     = np.arange(r_max)
    thetas = np.linspace(0, 2 * np.pi, theta_steps, endpoint=False)
    r_grid, t_grid = np.meshgrid(rs, thetas, indexing='ij')
    xs = center[1] + r_grid * np.cos(t_grid)
    ys = center[0] + r_grid * np.sin(t_grid)
    return map_coordinates(image, [ys, xs], order=0, mode='constant', cval=0)


def polar_to_cartesian(polar, output_shape, center=None):
    h, w = output_shape
    if center is None:
        center = (h / 2.0, w / 2.0)
    r_max, theta_steps = polar.shape
    ys, xs = np.mgrid[0:h, 0:w]
    dy = ys - center[0];  dx = xs - center[1]
    r     = np.sqrt(dy**2 + dx**2).clip(0, r_max - 1)
    theta = np.arctan2(dy, dx) % (2 * np.pi)
    t_idx = theta / (2 * np.pi) * theta_steps
    return map_coordinates(polar, [r, t_idx], order=0, mode='constant', cval=0)


def reflect_antidiag(arr):
    return arr[::-1, ::-1].T


# ══════════════════════════════════════════════════════════════════════════════
#  Annotation fill (polar space)
# ══════════════════════════════════════════════════════════════════════════════

def fill_annotations_polar(polar):
    """Fill thin arc annotations radially inward to produce a solid wedge."""
    filled = np.zeros_like(polar)
    labels = np.unique(polar)
    labels = labels[labels != 0]
    label_radii = {}
    for label in labels:
        rows, _ = np.where(polar == label)
        label_radii[label] = np.median(rows) if len(rows) > 0 else 0
    for label in sorted(labels, key=lambda l: label_radii[l]):
        mask = (polar == label)
        for col in range(polar.shape[1]):
            hits = np.where(mask[:, col])[0]
            if hits.size == 0:
                continue
            filled[0: hits[-1] + 1, col] = label
    return filled


# ══════════════════════════════════════════════════════════════════════════════
#  Polar-space vessel wall detection  ← new core method
# ══════════════════════════════════════════════════════════════════════════════

def find_vessel_wall_polar(dcm_polar, filled_ann_polar):
    """
    For each angle column, detect the inner edge of the bright vessel wall
    using the gradient of the intensity profile.

    Strategy:
      - Work in the region [CENTER_RADIUS, annotation_outer_radius] per column
      - Smooth the intensity profile to suppress noise
      - Find the steepest positive gradient → inner edge of vessel wall ring
      - Mask = True from that inner edge out to the annotation boundary

    Returns
    -------
    wall_mask : (r_max, theta_steps) bool array
    """
    r_max, theta_steps = dcm_polar.shape
    wall_mask = np.zeros((r_max, theta_steps), dtype=bool)

    for col in range(theta_steps):
        # Outer boundary from annotation
        ann_rows = np.where(filled_ann_polar[:, col] > 0)[0]
        if ann_rows.size == 0:
            continue
        r_outer = int(ann_rows[-1])

        if r_outer <= CENTER_RADIUS + 4:
            continue

        # Intensity profile from just past catheter to annotation boundary
        profile = dcm_polar[CENTER_RADIUS:r_outer + 1, col].astype(np.float32)
        n = len(profile)
        if n < 4:
            continue

        # Smooth to reduce salt-and-pepper noise
        k = min(SMOOTH_KERNEL, n)
        if k % 2 == 0:
            k -= 1
        profile_smooth = np.convolve(profile, np.ones(k) / k, mode='same')

        # Gradient of smoothed profile → peaks = edges of bright ring
        grad = np.gradient(profile_smooth)

        # The innermost strong positive gradient = rising edge of vessel wall
        peak_val = grad.max()
        if peak_val <= 0:
            # No clear rising edge — fall back to peak brightness location
            r_inner = CENTER_RADIUS + int(np.argmax(profile_smooth))
        else:
            # Threshold: look for first point at ≥50% of peak gradient
            candidates = np.where(grad >= 0.5 * peak_val)[0]
            r_inner = CENTER_RADIUS + int(candidates[0])

        # Clamp
        r_inner = max(CENTER_RADIUS, min(r_inner, r_outer))

        wall_mask[r_inner:r_outer + 1, col] = True

    return wall_mask


# ══════════════════════════════════════════════════════════════════════════════
#  Per-slice pipeline
# ══════════════════════════════════════════════════════════════════════════════

def process_slice(slice_idx, nifti_data, dcm_data):
    """
    Returns
    -------
    filled_ann  : (H, W) float  — filled annotation wedge (outer boundary)
    wall_mask   : (H, W) bool   — detected vessel wall region
    final_out   : (H, W) float  — final mask (inner wall → annotation boundary)
    dcm_img     : (H, W, 3)     — raw DICOM colour frame
    """
    # ── 1. Raw annotation slice, no pre-transform (exact same as main.py) ───────
    ann_sl    = nifti_data[..., slice_idx]
    ann_shape = ann_sl.shape

    # ── 2. Annotation → polar → fill → back to Cartesian (exact main.py) ────────
    ann_polar    = cartesian_to_polar(ann_sl)
    filled_polar = fill_annotations_polar(ann_polar)
    r_max        = filled_polar.shape[0]
    filled_cart  = polar_to_cartesian(filled_polar.astype(np.float32), ann_shape)

    # ── 3. Apply exact main.py orientation chain to filled annotation ────────────
    def orient(arr):
        reflected = np.flipud(np.fliplr(arr)).T   # anti-diagonal
        out       = np.flipud(reflected)           # flip about x
        return np.fliplr(out)                      # flip about y

    filled_ann = orient(filled_cart)

    # ── 4. Load DICOM, resize to annotation shape ─────────────────────────────
    dcm_img  = dcm_data[slice_idx + 1].astype(np.uint8)
    dcm_gray = cv2.cvtColor(dcm_img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    if dcm_gray.shape != ann_shape:
        dcm_gray = cv2.resize(dcm_gray, (ann_shape[1], ann_shape[0]),
                              interpolation=cv2.INTER_LINEAR)

    # ── 5. DCM → polar (same geometry as annotation) ──────────────────────────
    dcm_polar = cartesian_to_polar(dcm_gray, r_max=r_max, theta_steps=THETA_STEPS)

    # ── 6. Detect vessel wall in polar space ──────────────────────────────────
    wall_polar = find_vessel_wall_polar(dcm_polar, filled_polar)

    # ── 7. Wall mask → Cartesian → same orientation chain ────────────────────
    wall_cart = polar_to_cartesian(wall_polar.astype(np.float32), ann_shape)
    wall_mask = orient(wall_cart) > 0.5

    # ── 8. Final output ───────────────────────────────────────────────────────
    final_out = wall_mask.astype(np.float32)

    return filled_ann, wall_mask, final_out, dcm_img


# ══════════════════════════════════════════════════════════════════════════════
#  Demo mode — single slice debug figure
# ══════════════════════════════════════════════════════════════════════════════

def demo(slices=(118, 128, 132)):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading NIfTI ...")
    nifti_data = nib.load(NIFTI_PATH).get_fdata()
    print(f"  shape={nifti_data.shape}  labels={np.unique(nifti_data)}")

    print("Loading DICOM ...")
    dcm_data = pydicom.dcmread(DCM_PATH).pixel_array

    for slice_idx in slices:
        print(f"Processing slice {slice_idx} ...")
        filled_ann, wall_mask, final_out, dcm_img = process_slice(
            slice_idx, nifti_data, dcm_data)

        real_frame = slice_idx + 1
        fig, axes = plt.subplots(1, 5, figsize=(30, 6))

        axes[0].imshow(dcm_img)
        axes[0].set_title(f'DICOM frame {real_frame}')

        axes[1].imshow(filled_ann, cmap='nipy_spectral')
        axes[1].set_title('Annotation (outer boundary)')

        axes[2].imshow(wall_mask, cmap='gray')
        axes[2].set_title('Wall mask (raw)')

        axes[3].imshow(dcm_img)
        axes[3].imshow(wall_mask, cmap='Reds', alpha=0.5)
        axes[3].set_title('Wall mask overlay')

        axes[4].imshow(dcm_img)
        axes[4].imshow(final_out, cmap='Reds', alpha=0.5)
        axes[4].set_title('FINAL mask overlay')

        for ax in axes:
            ax.axis('off')
        plt.tight_layout()
        out_png = os.path.join(OUTPUT_DIR, f"pipeline_slice_{slice_idx:04d}.png")
        plt.savefig(out_png, dpi=120)
        plt.show()
        print(f"Saved → {os.path.abspath(out_png)}")


# ══════════════════════════════════════════════════════════════════════════════
#  Volume mode — all slices → DICOM
# ══════════════════════════════════════════════════════════════════════════════

def run_volume():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading NIfTI ...")
    nifti_data = nib.load(NIFTI_PATH).get_fdata()
    n_slices   = nifti_data.shape[-1]

    print("Loading DICOM ...")
    dcm_data = pydicom.dcmread(DCM_PATH).pixel_array

    processed = []
    for z in range(n_slices):
        _, _, final_out, _ = process_slice(z, nifti_data, dcm_data)
        processed.append(final_out)
        if z % 50 == 0:
            print(f"  slice {z:4d}/{n_slices}")

    print("Saving DICOM ...")
    now     = datetime.datetime.now()
    volume  = np.stack(processed, axis=0)
    vol_f   = volume.astype(np.float32)
    lo, hi  = vol_f.min(), vol_f.max()
    vol_u16 = ((vol_f - lo) / (hi - lo + 1e-8) * 65535).astype(np.uint16)

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID    = '1.2.840.10008.5.1.4.1.1.2'
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID          = EXPLICIT_VR_LITTLE_ENDIAN

    ds = Dataset()
    ds.file_meta        = file_meta
    ds.is_implicit_VR   = False
    ds.is_little_endian = True
    ds.SOPClassUID      = '1.2.840.10008.5.1.4.1.1.2'
    ds.SOPInstanceUID   = file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID  = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.InstanceNumber   = '1'
    ds.StudyDate = now.strftime('%Y%m%d')
    ds.StudyTime = now.strftime('%H%M%S')
    ds.Modality  = 'OT'
    ds.PatientName      = 'Anonymous'
    ds.PatientID        = 'ANON001'
    ds.PatientBirthDate = ''
    ds.PatientSex       = ''
    ds.Rows                      = vol_u16.shape[1]
    ds.Columns                   = vol_u16.shape[2]
    ds.NumberOfFrames            = vol_u16.shape[0]
    ds.SamplesPerPixel           = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.BitsAllocated             = 16
    ds.BitsStored                = 16
    ds.HighBit                   = 15
    ds.PixelRepresentation       = 0
    ds.PixelData                 = vol_u16.tobytes()

    out_path = os.path.join(OUTPUT_DIR, "output_vessel_wall.dcm")
    pydicom.dcmwrite(out_path, ds, write_like_original=False)
    print(f"Saved → {os.path.abspath(out_path)}")


# ── Entry point ───────────────────────────────────────────────────────────────
# ┌─────────────────────────────────────────────────────────────────────────────
# │  HIT RUN HERE — just change the settings below
# │
# │  MODE   : "demo"   → save a 4-panel debug PNG for each slice in SLICES
# │           "volume" → process every slice, save output_vessel_wall.dcm
# │
# │  SLICES : which slices to preview in demo mode (ignored for volume)
# └─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    MODE   = "demo"           # "demo" or "volume"
    SLICES = (118, 128, 132)  # slices for demo mode only

    if MODE == "demo":
        demo(SLICES)
    else:
        run_volume()
