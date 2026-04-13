import nibabel as nib
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import generate_uid
import uuid
import datetime
import os

EXPLICIT_VR_LITTLE_ENDIAN = '1.2.840.10008.1.2.1'

try:
    generate_uid()
except Exception:
    def generate_uid():
        return '2.25.' + str(uuid.uuid4().int)

# ─── Load NIfTI ───────────────────────────────────────────────────────────────
img = nib.load(
    "/Users/oceanpunsalan/Library/Mobile Documents/"
    "com~apple~CloudDocs/Data/Intravascular/IntraVascular/"
    "0B360D4D-3B16-4DCC-AD86-32361D1B47A9.nii"
)
data = img.get_fdata()
print("Original shape:", data.shape)
print("Unique labels: ", np.unique(data))


# ─── Cartesian → Polar ────────────────────────────────────────────────────────
def cartesian_to_polar(image, center=None, r_max=None, theta_steps=720):
    h, w = image.shape
    if center is None:
        center = (h / 2.0, w / 2.0)
    if r_max is None:
        r_max = int(min(center[0], center[1], h - center[0], w - center[1]))

    rs = np.arange(r_max)
    thetas = np.linspace(0, 2 * np.pi, theta_steps, endpoint=False)
    r_grid, t_grid = np.meshgrid(rs, thetas, indexing='ij')

    xs = center[1] + r_grid * np.cos(t_grid)
    ys = center[0] + r_grid * np.sin(t_grid)

    polar = map_coordinates(image, [ys, xs], order=0, mode='constant', cval=0)
    return polar


# ─── Fill arcs in polar space ────────────────────────────────────────────────
def fill_annotations_polar(polar):
    """
    For each angle column, sort labels by their outermost radius.
    Fill from r=0 to each label's arc, with outer labels overwriting inner ones.
    This turns thin arc annotations into filled wedge regions.

    If you have multiple concentric arcs (e.g. label 1 inner, label 2 outer),
    this fills:
      - label 1 from r=0 to its arc
      - label 2 from r=0 to its arc (overwriting label 1 in that region)
    giving you a filled annular structure.
    """
    filled = np.zeros_like(polar)

    labels = np.unique(polar)
    labels = labels[labels != 0]

    # Sort labels by their median radius so inner labels are drawn first
    # and outer labels overwrite them — producing correct layered fills
    label_radii = {}
    for label in labels:
        rows, _ = np.where(polar == label)
        label_radii[label] = np.median(rows) if len(rows) > 0 else 0
    labels_sorted = sorted(labels, key=lambda l: label_radii[l])

    for label in labels_sorted:
        mask = (polar == label)
        for col in range(polar.shape[1]):
            col_hits = np.where(mask[:, col])[0]
            if col_hits.size == 0:
                continue
            # Fill from r=0 all the way out to the outermost arc pixel
            r_outer = col_hits[-1]
            filled[0: r_outer + 1, col] = label

    return filled


# ─── Polar → Cartesian ────────────────────────────────────────────────────────
def polar_to_cartesian(polar, output_shape, center=None):
    h, w = output_shape
    if center is None:
        center = (h / 2.0, w / 2.0)

    r_max, theta_steps = polar.shape

    ys, xs = np.mgrid[0:h, 0:w]
    dy = ys - center[0]
    dx = xs - center[1]

    r = np.sqrt(dy ** 2 + dx ** 2).clip(0, r_max - 1)
    theta = np.arctan2(dy, dx) % (2 * np.pi)
    t_idx = theta / (2 * np.pi) * theta_steps

    cart = map_coordinates(polar, [r, t_idx], order=0, mode='constant', cval=0)

    return cart


n_slices = data.shape[-1]
processed = []
def reflect_antidiag(arr):
    return arr[::-1, ::-1].T

for z in range(n_slices):
    sl = data[..., z]

    polar        = cartesian_to_polar(sl)
    filled_polar = fill_annotations_polar(polar)
    filled_cart  = polar_to_cartesian(filled_polar, sl.shape)

    reflected = np.flipud(np.fliplr(filled_cart)).T  # anti-diagonal
    out = np.flipud(reflected)  # then flip about x axis
    finalout = np.fliplr(out)
    processed.append(finalout)

    if z % 50 == 0:
        print(f"  slice {z:4d}/{n_slices}  |  labels in polar: {np.unique(polar)}  |  labels in output: {np.unique(out)}")

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(sl,           cmap='gray'); axes[0].set_title('Original')
        axes[1].imshow(filled_cart,  cmap='gray'); axes[1].set_title('Before reflection')
        axes[2].imshow(out,          cmap='gray'); axes[2].set_title('After reflection')
        axes[3].imshow(sl.T,         cmap='gray'); axes[3].set_title('sl.T for reference')
        for ax in axes: ax.axis('off')
        plt.tight_layout()
        plt.savefig(f"debug_slice_{z:04d}.png", dpi=100)
        plt.close('all')

print("All slices processed.")

# ─── Save as single multi-frame DICOM ────────────────────────────────────────
now = datetime.datetime.now()
volume = np.stack(processed, axis=0)  # (n_slices, H, W)

vol_f = volume.astype(np.float32)
lo, hi = vol_f.min(), vol_f.max()
vol_u16 = ((vol_f - lo) / (hi - lo + 1e-8) * 65535).astype(np.uint16)

file_meta = FileMetaDataset()
file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
file_meta.MediaStorageSOPInstanceUID = generate_uid()
file_meta.TransferSyntaxUID = EXPLICIT_VR_LITTLE_ENDIAN

ds = Dataset()
ds.file_meta = file_meta
ds.is_implicit_VR = False
ds.is_little_endian = True

ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
ds.StudyInstanceUID = generate_uid()
ds.SeriesInstanceUID = generate_uid()
ds.InstanceNumber = '1'

ds.StudyDate = now.strftime('%Y%m%d')
ds.StudyTime = now.strftime('%H%M%S')
ds.Modality = 'OT'
ds.PatientName = 'Anonymous'
ds.PatientID = 'ANON001'
ds.PatientBirthDate = ''
ds.PatientSex = ''

ds.Rows = vol_u16.shape[1]
ds.Columns = vol_u16.shape[2]
ds.NumberOfFrames = vol_u16.shape[0]
ds.SamplesPerPixel = 1
ds.PhotometricInterpretation = 'MONOCHROME2'
ds.BitsAllocated = 16
ds.BitsStored = 16
ds.HighBit = 15
ds.PixelRepresentation = 0
ds.PixelData = vol_u16.tobytes()

out_path = "output.dcm"
pydicom.dcmwrite(out_path, ds, write_like_original=False)
print("Saved to:", os.path.abspath(out_path))