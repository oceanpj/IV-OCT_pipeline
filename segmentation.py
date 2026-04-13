
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import torch
import cv2
import os
import sys
import pydicom

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sam2_repo'))
from pathlib import Path

# ── SAM 2 imports (requires: pip install -e sam2/) ────────────────────────────
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


CHECKPOINT  = "checkpoints/sam2.1_hiera_large.pt"
MODEL_CFG   = "configs/sam2.1/sam2.1_hiera_l.yaml"   # inside the sam2 repo
DEVICE      = "cuda" if torch.cuda.is_available() else \
              ("mps"  if torch.backends.mps.is_available() else "cpu")

DCM_PATH = (
    "/Users/oceanpunsalan/Library/Mobile Documents/"
    "com~apple~CloudDocs/Data/Intravascular/IntraVascular/"
    "0B360D4D-3B16-4DCC-AD86-32361D1B47A9/round_color/"
    "0B360D4D-3B16-4DCC-AD86-32361D1B47A9.dcm"
)
OUTPUT_DIR  = "seg_output"




def load_predictor():
    """Build and return a SAM2ImagePredictor on the best available device."""
    print(f"Loading SAM 2 on {DEVICE} ...")
    sam2 = build_sam2(MODEL_CFG, CHECKPOINT, device=DEVICE,
                      apply_postprocessing=False)
    return SAM2ImagePredictor(sam2)


def slice_to_rgb(arr2d):
    """
    Convert a 2-D float/int array (one OCT slice) to uint8 RGB.
    SAM 2 expects an (H, W, 3) uint8 image.
    """
    lo, hi = arr2d.min(), arr2d.max()
    norm = ((arr2d - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)
    return np.stack([norm, norm, norm], axis=-1)   # grayscale → RGB


def mask_to_contour(mask):
    """
    Convert a boolean mask to a list of (x, y) contour points.
    Returns the longest contour (the main boundary).
    """
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.array([])
    # return the longest contour — usually the vessel/lumen wall
    return max(contours, key=cv2.contourArea).squeeze()


def segment_with_point(predictor, image_rgb, point_xy, label=1,
                       multimask=False):
    """
    Segment using a single (x, y) point prompt.

    Args:
        predictor   : SAM2ImagePredictor
        image_rgb   : (H, W, 3) uint8
        point_xy    : (x, y) tuple — click on or near the boundary
        label       : 1 = foreground, 0 = background
        multimask   : if True returns 3 candidate masks (best, medium, worst)

    Returns:
        masks  : (N, H, W) bool array
        scores : (N,) confidence scores
    """
    predictor.set_image(image_rgb)
    points = np.array([[point_xy[0], point_xy[1]]], dtype=np.float32)
    labels = np.array([label], dtype=np.int32)
    masks, scores, _ = predictor.predict(
        point_coords=points,
        point_labels=labels,
        multimask_output=multimask,
    )
    return masks, scores


def segment_with_box(predictor, image_rgb, box_xyxy):
    """
    Segment using a bounding box prompt — often more reliable than a point
    for circular OCT cross-sections.

    Args:
        box_xyxy : (x1, y1, x2, y2) covering the region of interest
    """
    predictor.set_image(image_rgb)
    box = np.array(box_xyxy, dtype=np.float32)
    masks, scores, _ = predictor.predict(
        box=box,
        multimask_output=False,
    )
    return masks, scores



def demo_single_slice(slice_idx=128, point_xy=None, box_xyxy=None):
    """
    Run SAM 2 on one slice of the NIfTI and save a debug PNG.
    Provide either point_xy=(x,y) or box_xyxy=(x1,y1,x2,y2).
    If neither is given, uses the image centre as a point prompt.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dcm = pydicom.dcmread(DCM_PATH)
    data = dcm.pixel_array  # (n_slices, H, W, 3)
    sl = data[slice_idx]  # (H, W, 3)
    img = sl.astype(np.uint8)
    h, w = img.shape[:2]

    predictor = load_predictor()

    if box_xyxy is not None:
        masks, scores = segment_with_box(predictor, img, box_xyxy)
    else:
        pt = point_xy if point_xy is not None else (w // 2, h // 2)
        masks, scores = segment_with_point(predictor, img, pt,
                                           multimask=True)

    best_mask = masks[np.argmax(scores)]
    contour   = mask_to_contour(best_mask)

    # ── visualise ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img, cmap='gray')
    axes[0].set_title(f'Slice {slice_idx} (original)')

    axes[1].imshow(img, cmap='gray')
    axes[1].imshow(best_mask, alpha=0.4, cmap='Reds')
    axes[1].set_title(f'SAM 2 mask  (score={scores.max():.3f})')

    axes[2].imshow(img, cmap='gray')
    if contour.ndim == 2 and len(contour) > 0:
        axes[2].plot(contour[:, 0], contour[:, 1], 'r-', linewidth=1)
    axes[2].set_title('Boundary contour')

    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"seg_slice_{slice_idx:04d}.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Saved → {os.path.abspath(out)}")
    return best_mask, contour



def segment_volume_auto(center_fraction=0.5):
    """
    Run SAM 2 on every slice using a centre-point prompt.
    Saves all masks as a NIfTI file (same shape as input).

    center_fraction : where to place the point prompt (0.5 = image centre).
                      For intravascular OCT the catheter is at the centre,
                      so the lumen/wall is slightly outward — adjust if needed.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    img_nib = nib.load(NIFTI_PATH)
    data    = img_nib.get_fdata()
    n_slices = data.shape[-1]

    predictor  = load_predictor()
    mask_vol   = np.zeros(data.shape, dtype=np.uint8)

    for z in range(n_slices):
        sl  = data[..., z]
        img = slice_to_rgb(sl)
        h, w = img.shape[:2]

        # place prompt just inside the expected vessel wall
        pt = (int(w * center_fraction), int(h * center_fraction))

        masks, scores = segment_with_point(predictor, img, pt,
                                           multimask=True)
        best = masks[np.argmax(scores)]
        mask_vol[..., z] = best.astype(np.uint8)

        if z % 50 == 0:
            print(f"  slice {z:4d}/{n_slices}  score={scores.max():.3f}")

    # save as NIfTI
    out_nib = nib.Nifti1Image(mask_vol, img_nib.affine, img_nib.header)
    out_path = os.path.join(OUTPUT_DIR, "sam2_masks.nii")
    nib.save(out_nib, out_path)
    print(f"\nSaved volume → {os.path.abspath(out_path)}")
    return mask_vol


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="SAM 2 boundary segmentation for OCT / vessel images"
    )
    parser.add_argument("--mode", choices=["demo", "volume"],
                        default="demo",
                        help="demo = single slice, volume = all slices")
    parser.add_argument("--slice", type=int, default=128,
                        help="slice index for demo mode")
    parser.add_argument("--point", type=int, nargs=2, metavar=("X", "Y"),
                        help="point prompt (x y) for demo mode")
    parser.add_argument("--box", type=int, nargs=4,
                        metavar=("X1", "Y1", "X2", "Y2"),
                        help="box prompt for demo mode")
    args = parser.parse_args()

    if args.mode == "demo":
        demo_single_slice(
            slice_idx=args.slice,
            point_xy=tuple(args.point) if args.point else None,
            box_xyxy=args.box,
        )
    else:
        segment_volume_auto()
