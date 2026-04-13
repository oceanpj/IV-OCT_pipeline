"""
create_catheter_mask.py
-----------------------
Creates a mask around the catheter (center circle) in OCT DICOM images.

The catheter appears as a dark circular region in the center of each frame.
This script detects the catheter boundary and creates a binary mask.
"""

import os
import numpy as np
import pydicom
import cv2
from scipy.ndimage import map_coordinates

# ── Config ──────────────────────────────────────────────────────────────────
CASE_ID = "0B360D4D-3B16-4DCC-AD86-32361D1B47A9"
DATA_DIR = "/Users/oceanpunsalan/Data/Intravascular/20_raw"
ANALYSIS_DIR = "/Users/oceanpunsalan/Data/Intravascular/Analysis"

# Catheter detection parameters
CENTER_RADIUS_GUESS = 40      # Initial guess for catheter radius (pixels)
FIXED_CATHER_RADIUS = 37      # Fixed radius for catheter mask (pixels)
THRESHOLD_METHOD = 'otsu'     # 'otsu' or 'manual'
MANUAL_THRESHOLD = 50         # Used if THRESHOLD_METHOD is 'manual'


def load_dicom(case_id):
    """Load the round_color DICOM for a case."""
    dcm_path = os.path.join(DATA_DIR, case_id, 'round_color', f'{case_id}.dcm')
    if not os.path.exists(dcm_path):
        raise FileNotFoundError(f"DICOM not found: {dcm_path}")

    ds = pydicom.dcmread(dcm_path)
    data = ds.pixel_array

    # Handle different shapes
    if len(data.shape) == 4:
        # (n_frames, H, W, 3) - RGB
        # Convert to grayscale
        n_frames, h, w, c = data.shape
        gray = np.zeros((n_frames, h, w), dtype=np.uint8)
        for i in range(n_frames):
            gray[i] = cv2.cvtColor(data[i], cv2.COLOR_RGB2GRAY)
        data = gray
    elif len(data.shape) == 3 and data.shape[-1] == 3:
        # (H, W, 3) - single RGB frame
        data = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
        data = data[np.newaxis, ...]  # Add frame dimension

    print(f"Loaded DICOM: shape={data.shape}, dtype={data.dtype}")
    return data, ds


def detect_catheter_radius(frame, center):
    """
    Detect the catheter radius in a single frame.

    Strategy:
    1. Look at radial intensity profiles from center
    2. Find the edge where intensity jumps (catheter wall)
    """
    h, w = frame.shape
    cy, cx = center

    # Sample radial profiles at multiple angles
    angles = np.linspace(0, 2*np.pi, 72, endpoint=False)
    radii = []

    for angle in angles:
        # Sample along this angle from center outward
        for r in range(5, 100):
            y = int(cy + r * np.sin(angle))
            x = int(cx + r * np.cos(angle))

            if y < 0 or y >= h or x < 0 or x >= w:
                break

            # Check if we've hit vessel wall (bright)
            if r > 20 and frame[y, x] > 30:  # Threshold for vessel wall
                radii.append(r)
                break

    if len(radii) < 10:
        # Fall back to default if detection failed
        return CENTER_RADIUS_GUESS

    # Use median to be robust to outliers
    radius = int(np.median(radii))
    print(f"  Detected catheter radius: {radius}px (from {len(radii)} samples)")
    return radius


def create_catheter_mask(frame_shape, center, radius):
    """Create a binary mask for the catheter region."""
    h, w = frame_shape
    mask = np.zeros((h, w), dtype=np.uint8)

    cy, cx = center
    y, x = np.ogrid[:h, :w]

    # Create circular mask
    dist_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)

    # Mask = 1 inside catheter (radius + small buffer)
    buffer = 2
    mask[dist_from_center <= (radius + buffer)] = 1

    return mask


def process_volume(dicom_data):
    """Process all frames in the DICOM volume."""
    n_frames = dicom_data.shape[0]
    h, w = dicom_data.shape[1:3]

    # Center is typically at image center
    center = (h // 2, w // 2)
    print(f"Image center: ({center[1]}, {center[0]})")

    # Use fixed radius for catheter mask
    radius = FIXED_CATHER_RADIUS
    print(f"Using fixed catheter radius: {radius}px")

    # Create masks for all frames
    masks = np.zeros((n_frames, h, w), dtype=np.uint8)

    for i in range(n_frames):
        masks[i] = create_catheter_mask((h, w), center, radius)

        if i % 50 == 0:
            print(f"  Processed frame {i}/{n_frames}")

    print(f"Created catheter mask volume: {masks.shape}")
    return masks


def save_dicom_mask(mask_volume, reference_ds, output_path):
    """Save mask as DICOM using reference for metadata."""
    import datetime

    now = datetime.datetime.now()

    # Create new dataset
    file_meta = pydicom.Dataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    ds = pydicom.Dataset()
    ds.file_meta = file_meta
    ds.is_implicit_VR = False
    ds.is_little_endian = True

    # Copy relevant metadata from reference
    ds.SOPClassUID = pydicom.uid.CTImageStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = getattr(reference_ds, 'StudyInstanceUID', pydicom.uid.generate_uid())
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.PatientName = getattr(reference_ds, 'PatientName', 'Anonymous')
    ds.PatientID = getattr(reference_ds, 'PatientID', 'ANON001')
    ds.StudyDate = getattr(reference_ds, 'StudyDate', now.strftime('%Y%m%d'))
    ds.StudyTime = getattr(reference_ds, 'StudyTime', now.strftime('%H%M%S'))
    ds.Modality = 'OT'
    ds.InstanceNumber = '1'

    ds.Rows = mask_volume.shape[1]
    ds.Columns = mask_volume.shape[2]
    ds.NumberOfFrames = mask_volume.shape[0]
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = mask_volume.astype(np.uint8).tobytes()

    pydicom.dcmwrite(output_path, ds, write_like_original=False)
    print(f"Saved catheter mask DICOM: {output_path}")


def main():
    print(f"Creating catheter mask for case: {CASE_ID}")
    print("=" * 60)

    # Load DICOM
    dicom_data, ds = load_dicom(CASE_ID)

    # Process to create mask
    mask_volume = process_volume(dicom_data)

    # Create output directory
    output_dir = os.path.join(ANALYSIS_DIR, f"{CASE_ID}_cath_mask_test")
    os.makedirs(output_dir, exist_ok=True)

    # Save mask
    output_path = os.path.join(output_dir, "cath_mask.dcm")
    save_dicom_mask(mask_volume, ds, output_path)

    print("=" * 60)
    print(f"Complete! Output: {output_path}")


if __name__ == '__main__':
    main()
