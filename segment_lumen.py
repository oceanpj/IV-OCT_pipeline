"""
segment_lumen.py
----------------
Segments the lumen (vessel cavity) in OCT images.

The lumen appears as a dark region surrounded by a bright vessel wall.
The wall is often discontinuous (gaps due to plaque, shadowing, etc.).
Strategy:
    1. Convert to polar coordinates from center
    2. Detect the outer wall boundary using intensity gradient
    3. Interpolate across gaps to create smooth boundary
    4. Fill interior to create lumen mask
    5. Convert back to Cartesian
"""

import os
import numpy as np
import pydicom
import cv2
from scipy.ndimage import map_coordinates, gaussian_filter1d
from scipy.interpolate import interp1d

# ── Config ──────────────────────────────────────────────────────────────────
CASE_ID = "0B360D4D-3B16-4DCC-AD86-32361D1B47A9"
DATA_DIR = "/Users/oceanpunsalan/Data/Intravascular/20_raw"
ANALYSIS_DIR = "/Users/oceanpunsalan/Data/Intravascular/Analysis"

# Lumen detection parameters
THETA_STEPS = 360               # Angular resolution for polar transform
MIN_WALL_RADIUS = 30            # Minimum radius to start searching (avoid catheter)
MAX_WALL_RADIUS = 200           # Maximum radius to search
GRADIENT_THRESHOLD = 15         # Minimum intensity jump to detect wall
GAUSSIAN_SIGMA = 2.0            # Smoothing for radial profile
WALL_SMOOTHING_WINDOW = 15      # For smoothing detected wall boundary
GAP_FILL_THRESHOLD = 30         # Max gap size to interpolate (in degrees)


def load_dicom(case_id):
    """Load the round_color DICOM for a case."""
    dcm_path = os.path.join(DATA_DIR, case_id, 'round_color', f'{case_id}.dcm')
    if not os.path.exists(dcm_path):
        raise FileNotFoundError(f"DICOM not found: {dcm_path}")

    ds = pydicom.dcmread(dcm_path)
    data = ds.pixel_array

    # Handle different shapes
    if len(data.shape) == 4:
        n_frames, h, w, c = data.shape
        gray = np.zeros((n_frames, h, w), dtype=np.uint8)
        for i in range(n_frames):
            gray[i] = cv2.cvtColor(data[i], cv2.COLOR_RGB2GRAY)
        data = gray
    elif len(data.shape) == 3 and data.shape[-1] == 3:
        data = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
        data = data[np.newaxis, ...]

    print(f"Loaded DICOM: shape={data.shape}, dtype={data.dtype}")
    return data, ds


def cartesian_to_polar(image, center, r_max, theta_steps=THETA_STEPS):
    """Convert Cartesian image to polar coordinates."""
    h, w = image.shape
    cy, cx = center

    rs = np.arange(r_max)
    thetas = np.linspace(0, 2*np.pi, theta_steps, endpoint=False)

    polar = np.zeros((r_max, theta_steps), dtype=image.dtype)

    for i, theta in enumerate(thetas):
        for r in range(r_max):
            y = int(cy + r * np.sin(theta))
            x = int(cx + r * np.cos(theta))
            if 0 <= y < h and 0 <= x < w:
                polar[r, i] = image[y, x]

    return polar


def polar_to_cartesian(polar_mask, output_shape, center):
    """Convert polar mask back to Cartesian."""
    h, w = output_shape
    cy, cx = center
    r_max, theta_steps = polar.shape

    cartesian = np.zeros((h, w), dtype=np.uint8)

    for i in range(theta_steps):
        theta = 2 * np.pi * i / theta_steps
        for r in range(r_max):
            if polar_mask[r, i]:
                y = int(cy + r * np.sin(theta))
                x = int(cx + r * np.cos(theta))
                if 0 <= y < h and 0 <= x < w:
                    cartesian[y, x] = 1

    return cartesian


def detect_wall_in_polar(polar_frame):
    """
    Detect vessel wall boundary in polar coordinates.

    Returns:
        wall_radii: Array of radius values for each angle (NaN where no wall detected)
    """
    r_max, n_angles = polar_frame.shape
    wall_radii = np.full(n_angles, np.nan)

    for angle_idx in range(n_angles):
        profile = polar_frame[:, angle_idx].astype(np.float32)

        # Skip if too short
        if len(profile) < MIN_WALL_RADIUS + 10:
            continue

        # Focus on region outside catheter
        search_region = profile[MIN_WALL_RADIUS:min(MAX_WALL_RADIUS, r_max)]

        if len(search_region) < 10:
            continue

        # Smooth profile to reduce noise
        smoothed = gaussian_filter1d(search_region, GAUSSIAN_SIGMA)

        # Find gradient (edge detection)
        gradient = np.gradient(smoothed)

        # Look for strong positive gradient (dark to bright = lumen to wall)
        peak_idx = np.argmax(gradient)
        peak_val = gradient[peak_idx]

        if peak_val > GRADIENT_THRESHOLD:
            wall_radii[angle_idx] = MIN_WALL_RADIUS + peak_idx

    return wall_radii


def interpolate_gaps(wall_radii, max_gap=GAP_FILL_THRESHOLD):
    """
    Interpolate across gaps in the wall boundary.

    Args:
        wall_radii: Array with NaN for missing detections
        max_gap: Maximum gap size (in indices) to interpolate

    Returns:
        Smoothed wall radii with gaps filled
    """
    n = len(wall_radii)
    result = wall_radii.copy()

    # Find valid (non-NaN) points
    valid_idx = np.where(~np.isnan(wall_radii))[0]

    if len(valid_idx) < 10:
        # Too few detections, return as-is
        return result

    # Close the circle by wrapping indices
    valid_idx_extended = np.concatenate([
        valid_idx - n,
        valid_idx,
        valid_idx + n
    ])
    valid_radii_extended = np.tile(wall_radii[valid_idx], 3)

    # Sort by index
    sort_order = np.argsort(valid_idx_extended)
    valid_idx_sorted = valid_idx_extended[sort_order]
    valid_radii_sorted = valid_radii_extended[sort_order]

    # Interpolate for all angles
    all_angles = np.arange(n)

    # Use cubic interpolation for smoothness
    f = interp1d(valid_idx_sorted, valid_radii_sorted, kind='cubic',
                 bounds_error=False, fill_value='extrapolate')

    interpolated = f(all_angles)

    # Only keep interpolated values for small gaps
    # Mark large gaps as invalid
    for i in range(n):
        if np.isnan(wall_radii[i]):
            # Find distance to nearest valid point
            dist_to_valid = min(
                np.min(np.abs(valid_idx - i)),
                np.min(np.abs(valid_idx - (i - n))),
                np.min(np.abs(valid_idx - (i + n)))
            )
            if dist_to_valid > max_gap:
                interpolated[i] = np.nan

    # Smooth the result
    valid_for_smoothing = ~np.isnan(interpolated)
    if np.sum(valid_for_smoothing) > 20:
        smoothed = interpolated.copy()
        smoothed_valid = gaussian_filter1d(
            interpolated[valid_for_smoothing],
            WALL_SMOOTHING_WINDOW / 2.355  # Convert FWHM to sigma
        )
        smoothed[valid_for_smoothing] = smoothed_valid
        return smoothed

    return interpolated


def create_lumen_mask_polar(wall_radii, r_max):
    """Create filled lumen mask in polar coordinates."""
    n_angles = len(wall_radii)
    mask = np.zeros((r_max, n_angles), dtype=np.uint8)

    for angle_idx in range(n_angles):
        if not np.isnan(wall_radii[angle_idx]):
            radius = int(wall_radii[angle_idx])
            mask[:min(radius, r_max), angle_idx] = 1

    return mask


def process_frame(frame, center):
    """Process a single frame to create lumen mask."""
    h, w = frame.shape

    # Convert to polar
    r_max = min(h, w) // 2
    polar = cartesian_to_polar(frame, center, r_max)

    # Detect wall
    wall_radii = detect_wall_in_polar(polar)

    # Count detections
    n_detected = np.sum(~np.isnan(wall_radii))

    # Interpolate gaps
    wall_radii_smooth = interpolate_gaps(wall_radii)
    n_filled = np.sum(~np.isnan(wall_radii_smooth))

    # Create mask
    lumen_mask_polar = create_lumen_mask_polar(wall_radii_smooth, r_max)

    # Convert back to Cartesian
    lumen_mask = polar_to_cartesian(lumen_mask_polar, (h, w), center)

    return lumen_mask, wall_radii_smooth, n_detected, n_filled


def save_dicom_mask(mask_volume, reference_ds, output_path):
    """Save mask as DICOM using reference for metadata."""
    import datetime

    now = datetime.datetime.now()

    file_meta = pydicom.Dataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    ds = pydicom.Dataset()
    ds.file_meta = file_meta
    ds.is_implicit_VR = False
    ds.is_little_endian = True

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


def main():
    print(f"Segmenting lumen for case: {CASE_ID}")
    print("=" * 60)

    # Load DICOM
    dicom_data, ds = load_dicom(CASE_ID)
    n_frames = dicom_data.shape[0]
    h, w = dicom_data.shape[1:3]

    # Center is typically at image center
    center = (h // 2, w // 2)
    print(f"Image center: ({center[1]}, {center[0]})")
    print()

    # Process each frame
    lumen_masks = np.zeros_like(dicom_data, dtype=np.uint8)

    for i in range(n_frames):
        mask, wall_radii, n_detected, n_filled = process_frame(dicom_data[i], center)
        lumen_masks[i] = mask

        if i % 25 == 0:
            print(f"Frame {i:3d}: detected {n_detected:3.0f} angles, filled to {n_filled:3.0f} angles")

    print()
    print(f"Created lumen mask volume: {lumen_masks.shape}")

    # Create output directory
    output_dir = os.path.join(ANALYSIS_DIR, f"{CASE_ID}_lumen_test")
    os.makedirs(output_dir, exist_ok=True)

    # Save mask
    output_path = os.path.join(output_dir, "lumen_mask.dcm")
    save_dicom_mask(lumen_masks, ds, output_path)

    print("=" * 60)
    print(f"Complete! Output: {output_path}")


if __name__ == '__main__':
    main()
