"""
full_res_warp.py
----------------
Warps envelope.bin from polar to Cartesian at full radial resolution.

The envelope is (250, 800, 1024) in polar — 1024 depth samples.
Standard export at 512×512 downsamples radially by 4×.
Here we warp to 2048×2048 so polar depth maps 1:1 to Cartesian radius:
    radial pixels = 2048 / 2 = 1024  ==  polar depth samples

Output: full_res_output/full_res.dcm
"""

import os
import numpy as np
import pydicom
import datetime
import uuid
from scipy.ndimage import map_coordinates
from extract_data import read_envelope, read_hardware_inf

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = r"/Users/oceanpunsalan/Library/Mobile Documents/com~apple~CloudDocs/Data/Intravascular/IntraVascular/B36087B8-BD7C-4853-8156-99A2733E125C/"   # ← change this
OUTPUT_DIR = "full_res_output"
EXPORT_DIM = 2048    # 2048/2 = 1024 radial pixels = 1:1 with polar depth
DTYPE      = np.uint8

# ── Polar → Cartesian ─────────────────────────────────────────────────────────

def polar_to_cartesian(bscan_polar, output_size):
    """
    Warp a single polar frame (depth × alines) to a square Cartesian image.

    Parameters
    ----------
    bscan_polar : (depth, alines) uint8
    output_size : int — side length of square output

    Returns
    -------
    (output_size, output_size) uint8
    """
    depth, alines = bscan_polar.shape

    x = np.linspace(-1, 1, output_size)
    y = np.linspace(-1, 1, output_size)
    X, Y = np.meshgrid(x, y)

    r     = np.sqrt(X**2 + Y**2)                        # [0, ~1.41]
    theta = np.arctan2(Y, X)                             # [-π, π]

    r_idx     = r     * (depth  - 1)                    # [0, depth-1]
    theta_idx = ((theta + np.pi) / (2 * np.pi)) * (alines - 1)  # [0, alines-1]

    # Clip to valid range
    r_idx     = np.clip(r_idx,     0, depth  - 1)
    theta_idx = np.clip(theta_idx, 0, alines - 1)

    cart = map_coordinates(
        bscan_polar,
        [r_idx, theta_idx],
        order=1,          # bilinear interpolation
        mode='constant',
        cval=0
    )
    return cart.astype(np.uint8)


# ── DICOM writer ──────────────────────────────────────────────────────────────

def save_dicom(volume, path):
    """
    Save a (n_frames, H, W) uint8 volume as a multi-frame DICOM.
    """
    now = datetime.datetime.now()

    file_meta = pydicom.Dataset()
    file_meta.MediaStorageSOPClassUID    = pydicom.uid.CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID          = pydicom.uid.ExplicitVRLittleEndian

    ds = pydicom.Dataset()
    ds.file_meta        = file_meta
    ds.is_implicit_VR   = False
    ds.is_little_endian = True

    ds.SOPClassUID      = pydicom.uid.CTImageStorage
    ds.SOPInstanceUID   = file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID  = pydicom.uid.generate_uid()
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()

    ds.PatientName      = 'Anonymous'
    ds.PatientID        = 'ANON001'
    ds.PatientBirthDate = ''
    ds.PatientSex       = ''
    ds.StudyDate        = now.strftime('%Y%m%d')
    ds.StudyTime        = now.strftime('%H%M%S')
    ds.Modality         = 'OT'
    ds.InstanceNumber   = '1'

    ds.NumberOfFrames            = volume.shape[0]
    ds.Rows                      = volume.shape[1]
    ds.Columns                   = volume.shape[2]
    ds.SamplesPerPixel           = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.BitsAllocated             = 8
    ds.BitsStored                = 8
    ds.HighBit                   = 7
    ds.PixelRepresentation       = 0
    ds.PixelData                 = volume.tobytes()

    pydicom.dcmwrite(path, ds, write_like_original=False)
    print(f"Saved → {os.path.abspath(path)}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading envelope.bin ...")
    envelope, hdr = read_envelope(os.path.join(DATA_DIR, 'envelope.bin'))
    n_frames, n_alines, depth = envelope.shape
    print(f"  polar shape: {envelope.shape}")
    print(f"  export dim:  {EXPORT_DIM}×{EXPORT_DIM}")
    print(f"  radial px:   {EXPORT_DIM // 2}  (polar depth: {depth})  ratio: {depth / (EXPORT_DIM // 2):.2f}×")

    print(f"\nWarping {n_frames} frames to {EXPORT_DIM}×{EXPORT_DIM} ...")
    volume = np.zeros((n_frames, EXPORT_DIM, EXPORT_DIM), dtype=DTYPE)

    for i in range(n_frames):
        # envelope[i] is (alines, depth) — transpose to (depth, alines) for warp
        frame_polar = envelope[i].T   # (1024, 800)
        volume[i]   = polar_to_cartesian(frame_polar, EXPORT_DIM)

        if i % 50 == 0:
            print(f"  frame {i:3d}/{n_frames}")

    print("\nSaving DICOM ...")
    out_path = os.path.join(OUTPUT_DIR, "full_res.dcm")
    save_dicom(volume, out_path)
    print("Done.")
