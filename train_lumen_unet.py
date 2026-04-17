"""
train_lumen_unet.py
-------------------
Trains a 2D U-Net to segment the OCT lumen from a single annotated volume,
then runs inference on a new volume and saves the predicted mask as .h5.

Data
----
  Training DICOM  : (250, 512, 512, 3) uint8 RGB
  Training mask   : mask.h5  keys '1' and '2' — both (250, 512, 512) int8 binary
                    (two annotators; we union them for a robust training signal)
  Test DICOM      : same shape, no mask

Output
------
  checkpoints/unet_lumen.pth          — saved model weights
  <test_case_dir>/mask.h5             — predicted mask in identical format to input
  training_curves.png                 — loss / Dice per epoch

Usage
-----
  python train_lumen_unet.py          # train then infer
  python train_lumen_unet.py --infer  # infer only (loads saved weights)
"""

import os
import sys
import argparse
import numpy as np
import h5py
import pydicom
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import random

# ══════════════════════════════════════════════════════════════════════════════
#  Paths  —  folders are auto-scanned for volumes
# ══════════════════════════════════════════════════════════════════════════════

TRAIN_DIR = r"Y:\Eye\Ocean\Intravascular\IV-OCT_pipeline\Lumen_Training"
TEST_DIR  = r"Y:\Eye\Ocean\Intravascular\IV-OCT_pipeline\Lumen_Testing"

# Expected format in TRAIN_DIR : <name>.dcm  +  <name>_mask.h5
# Expected format in TEST_DIR  : <name>.dcm  (mask will be written as <name>_mask.h5)

CKPT_DIR   = "checkpoints"
CKPT_PATH  = os.path.join(CKPT_DIR, "unet_lumen.pth")

# ══════════════════════════════════════════════════════════════════════════════
#  Hyperparameters
# ══════════════════════════════════════════════════════════════════════════════

IMG_SIZE      = 512       # input resolution (no resize needed, already 512)
BATCH_SIZE    = 4         # reduce to 2 if GPU OOM
NUM_EPOCHS    = 60
LR            = 3e-4
VAL_SPLIT     = 0.2       # 20% of frames held out for validation
EARLY_STOP    = 12        # stop if val Dice doesn't improve for this many epochs
MASK_KEY      = '1'       # only key present in mask.h5
# annotated-frame detection: any frame whose mask has ≥ this many positive
# pixels is considered annotated and used for training.  Frames that are
# entirely zero get skipped automatically.
MIN_MASK_PIXELS = 1
DEVICE        = (
    "mps"  if torch.backends.mps.is_available() else   # Apple Silicon
    "cuda" if torch.cuda.is_available()         else
    "cpu"
)

# ══════════════════════════════════════════════════════════════════════════════
#  Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_dicom_gray(path):
    """Return (N, H, W) float32 grayscale volume in [0, 1]."""
    px = pydicom.dcmread(path).pixel_array                # (N, H, W, 3) uint8
    if px.ndim == 4 and px.shape[-1] == 3:
        gray = np.stack(
            [cv2.cvtColor(px[i], cv2.COLOR_RGB2GRAY) for i in range(px.shape[0])],
            axis=0
        )
    else:
        gray = px
    return gray.astype(np.float32) / 255.0


def load_mask(path, key=MASK_KEY):
    """Return (N, H, W) float32 binary mask."""
    with h5py.File(path, 'r') as f:
        return (f[key][()] > 0).astype(np.float32)


def discover_training_pairs(train_dir):
    """
    Return list of (dcm_path, mask_path) tuples for every <name>.dcm in
    train_dir that has a matching <name>_mask.h5 sibling.
    """
    pairs = []
    for fname in sorted(os.listdir(train_dir)):
        if not fname.lower().endswith('.dcm'):
            continue
        stem     = os.path.splitext(fname)[0]
        dcm_p    = os.path.join(train_dir, fname)
        mask_p   = os.path.join(train_dir, stem + '_mask.h5')
        if os.path.exists(mask_p):
            pairs.append((dcm_p, mask_p))
        else:
            print(f"  WARN: no mask found for {fname} — skipping")
    return pairs


def build_training_set(train_dir):
    """
    Load every <name>.dcm + <name>_mask.h5 pair from train_dir.
    Keep ONLY frames that are annotated (mask has >= MIN_MASK_PIXELS positives).

    Returns
    -------
    images : (K, H, W) float32
    masks  : (K, H, W) float32
    where K = total annotated frames across all training volumes.
    """
    pairs = discover_training_pairs(train_dir)
    if not pairs:
        raise RuntimeError(f"No (.dcm, _mask.h5) pairs found in {train_dir}")

    all_imgs, all_masks = [], []
    for dcm_p, mask_p in pairs:
        name = os.path.basename(dcm_p)
        print(f"  loading {name}")
        imgs = load_dicom_gray(dcm_p)                     # (N, H, W)
        mks  = load_mask(mask_p)                          # (M, H, W)

        # If the mask covers fewer frames than the DICOM, pad with zeros so
        # shapes align (unannotated frames will then be filtered out below).
        if mks.shape[0] < imgs.shape[0]:
            pad = np.zeros((imgs.shape[0] - mks.shape[0],
                            mks.shape[1], mks.shape[2]), dtype=mks.dtype)
            mks = np.concatenate([mks, pad], axis=0)
        elif mks.shape[0] > imgs.shape[0]:
            mks = mks[:imgs.shape[0]]

        # Keep only frames that actually have an annotation
        has_ann = mks.reshape(mks.shape[0], -1).sum(axis=1) >= MIN_MASK_PIXELS
        n_ann   = int(has_ann.sum())
        print(f"    frames={imgs.shape[0]}  annotated={n_ann}  "
              f"({100*n_ann/imgs.shape[0]:.0f}%)")

        all_imgs.append(imgs[has_ann])
        all_masks.append(mks[has_ann])

    images = np.concatenate(all_imgs,  axis=0)
    masks  = np.concatenate(all_masks, axis=0)
    return images, masks


def discover_test_volumes(test_dir):
    """Return list of absolute paths to every .dcm in test_dir."""
    return [os.path.join(test_dir, f) for f in sorted(os.listdir(test_dir))
            if f.lower().endswith('.dcm')]


# ══════════════════════════════════════════════════════════════════════════════
#  Augmentation
# ══════════════════════════════════════════════════════════════════════════════

def augment(image, mask):
    """
    image, mask : torch.Tensor (1, H, W) float32
    Returns augmented pair with same shapes.
    """
    # Random horizontal flip
    if random.random() > 0.5:
        image = TF.hflip(image)
        mask  = TF.hflip(mask)

    # Random vertical flip
    if random.random() > 0.5:
        image = TF.vflip(image)
        mask  = TF.vflip(mask)

    # Random rotation ±20°
    if random.random() > 0.3:
        angle = random.uniform(-20, 20)
        image = TF.rotate(image, angle)
        mask  = TF.rotate(mask,  angle)

    # Random brightness / contrast  (image only)
    if random.random() > 0.4:
        image = TF.adjust_brightness(image, random.uniform(0.7, 1.3))
    if random.random() > 0.4:
        image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))

    # Gaussian noise  (image only)
    if random.random() > 0.4:
        noise = torch.randn_like(image) * random.uniform(0.01, 0.04)
        image = (image + noise).clamp(0, 1)

    return image, mask


# ══════════════════════════════════════════════════════════════════════════════
#  Dataset
# ══════════════════════════════════════════════════════════════════════════════

class LumenDataset(Dataset):
    def __init__(self, images, masks, augment_fn=None):
        """
        images : (N, H, W) float32 [0,1]
        masks  : (N, H, W) float32 binary
        """
        self.images     = images
        self.masks      = masks
        self.augment_fn = augment_fn

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img  = torch.from_numpy(self.images[idx]).unsqueeze(0)  # (1, H, W)
        mask = torch.from_numpy(self.masks[idx]).unsqueeze(0)   # (1, H, W)
        if self.augment_fn is not None:
            img, mask = self.augment_fn(img, mask)
        return img, mask


# ══════════════════════════════════════════════════════════════════════════════
#  U-Net
# ══════════════════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    """Two conv-BN-ReLU layers."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """
    Standard 4-level U-Net.
    in_ch=1 (grayscale), out_ch=1 (binary lumen mask).
    Features: 32-64-128-256-512
    """
    def __init__(self, in_ch=1, out_ch=1, features=(32, 64, 128, 256, 512)):
        super().__init__()
        self.encoders  = nn.ModuleList()
        self.pools     = nn.ModuleList()
        self.decoders  = nn.ModuleList()
        self.upconvs   = nn.ModuleList()

        # Encoder
        ch = in_ch
        for f in features[:-1]:
            self.encoders.append(ConvBlock(ch, f))
            self.pools.append(nn.MaxPool2d(2))
            ch = f

        # Bottleneck
        self.bottleneck = ConvBlock(ch, features[-1])

        # Decoder
        for f in reversed(features[:-1]):
            self.upconvs.append(nn.ConvTranspose2d(f * 2, f, 2, stride=2))
            self.decoders.append(ConvBlock(f * 2, f))

        self.final = nn.Conv2d(features[0], out_ch, 1)

    def forward(self, x):
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for upconv, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            # handle odd input sizes
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        return self.final(x)   # raw logits


# ══════════════════════════════════════════════════════════════════════════════
#  Loss — Dice + BCE combined
# ══════════════════════════════════════════════════════════════════════════════

def dice_loss(pred, target, smooth=1.0):
    pred   = torch.sigmoid(pred)
    flat_p = pred.view(-1)
    flat_t = target.view(-1)
    inter  = (flat_p * flat_t).sum()
    return 1 - (2 * inter + smooth) / (flat_p.sum() + flat_t.sum() + smooth)


def combined_loss(pred, target):
    return dice_loss(pred, target) + F.binary_cross_entropy_with_logits(pred, target)


def dice_score(pred_logits, target):
    pred = (torch.sigmoid(pred_logits) > 0.5).float()
    flat_p = pred.view(-1);  flat_t = target.view(-1)
    inter  = (flat_p * flat_t).sum()
    return ((2 * inter + 1) / (flat_p.sum() + flat_t.sum() + 1)).item()


# ══════════════════════════════════════════════════════════════════════════════
#  Training
# ══════════════════════════════════════════════════════════════════════════════

def train(model, train_loader, val_loader):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )

    best_val_dice  = 0.0
    patience_count = 0
    history        = {'train_loss': [], 'val_loss': [], 'val_dice': []}

    os.makedirs(CKPT_DIR, exist_ok=True)

    print(f"Device : {DEVICE}")
    print(f"Train  : {len(train_loader.dataset)} slices  |  "
          f"Val : {len(val_loader.dataset)} slices")
    print(f"Epochs : {NUM_EPOCHS}  |  Batch : {BATCH_SIZE}  |  LR : {LR}\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        # ── train ──────────────────────────────────────────────────────────
        model.train()
        train_losses = []
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            preds = model(imgs)
            loss  = combined_loss(preds, masks)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # ── validate ────────────────────────────────────────────────────────
        model.eval()
        val_losses, val_dices = [], []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                preds = model(imgs)
                val_losses.append(combined_loss(preds, masks).item())
                val_dices.append(dice_score(preds, masks))

        t_loss = np.mean(train_losses)
        v_loss = np.mean(val_losses)
        v_dice = np.mean(val_dices)
        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['val_dice'].append(v_dice)

        scheduler.step()

        print(f"Epoch {epoch:3d}/{NUM_EPOCHS}  "
              f"train_loss={t_loss:.4f}  val_loss={v_loss:.4f}  val_Dice={v_dice:.4f}")

        # ── checkpoint ──────────────────────────────────────────────────────
        if v_dice > best_val_dice:
            best_val_dice = v_dice
            patience_count = 0
            torch.save(model.state_dict(), CKPT_PATH)
            print(f"  ✓ saved best model  (Dice={best_val_dice:.4f})")
        else:
            patience_count += 1
            if patience_count >= EARLY_STOP:
                print(f"\nEarly stopping — no improvement for {EARLY_STOP} epochs.")
                break

    print(f"\nBest val Dice : {best_val_dice:.4f}")
    save_curves(history)
    return history


def save_curves(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history['train_loss'], label='train')
    ax1.plot(history['val_loss'],   label='val')
    ax1.set(title='Loss', xlabel='Epoch');  ax1.legend();  ax1.grid(alpha=0.3)
    ax2.plot(history['val_dice'], color='green')
    ax2.set(title='Val Dice', xlabel='Epoch');  ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=120)
    plt.close()
    print("Training curves → training_curves.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Inference
# ══════════════════════════════════════════════════════════════════════════════

def infer_volume(model, dcm_path):
    """
    Run inference on every frame of dcm_path and save
        <same_folder>/<dcm_stem>_mask.h5
    with key '1' matching the training mask.h5 format.
    """
    model.eval()

    stem         = os.path.splitext(os.path.basename(dcm_path))[0]
    out_h5_path  = os.path.join(os.path.dirname(dcm_path), f"{stem}_mask.h5")

    print(f"\n── {os.path.basename(dcm_path)} ──")
    images = load_dicom_gray(dcm_path)                # (N, H, W) float32
    n = images.shape[0]
    pred = np.zeros(images.shape, dtype=np.int8)

    with torch.no_grad():
        for i in range(n):
            img = torch.from_numpy(images[i]).unsqueeze(0).unsqueeze(0).to(DEVICE)
            logit = model(img)
            prob  = torch.sigmoid(logit).squeeze().cpu().numpy()
            pred[i] = (prob > 0.5).astype(np.int8)

            if i % 50 == 0:
                print(f"  frame {i:3d}/{n}  lumen px = {pred[i].sum()}")

    with h5py.File(out_h5_path, 'w') as f:
        f.create_dataset('1', data=pred, dtype=np.int8, compression='gzip')

    print(f"  saved → {out_h5_path}")
    print(f"  positive voxels : {pred.sum()} / {pred.size} "
          f"({100*pred.mean():.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main(infer_only=False):
    model = UNet(in_ch=1, out_ch=1)

    if not infer_only:
        print("=" * 60)
        print("LOADING TRAINING DATA")
        print(f"  dir: {TRAIN_DIR}")
        print("=" * 60)
        images, masks = build_training_set(TRAIN_DIR)
        print(f"\nTotal annotated slices : {len(images)}   "
              f"shape = {images.shape}")
        print(f"Positive pixel fraction : {masks.mean()*100:.2f}%\n")

        idx = np.arange(len(images))
        train_idx, val_idx = train_test_split(idx, test_size=VAL_SPLIT,
                                              random_state=42, shuffle=True)

        train_ds = LumenDataset(images[train_idx], masks[train_idx], augment_fn=augment)
        val_ds   = LumenDataset(images[val_idx],   masks[val_idx],   augment_fn=None)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                  shuffle=True,  num_workers=0, pin_memory=False)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=0, pin_memory=False)

        print("=" * 60)
        print("TRAINING")
        print("=" * 60)
        train(model, train_loader, val_loader)

    # Load best weights
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"No checkpoint found at {CKPT_PATH}. "
                                f"Run without --infer first to train.")
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    model.to(DEVICE)
    print(f"\nLoaded weights from {CKPT_PATH}")

    print("=" * 60)
    print("INFERENCE")
    print(f"  dir: {TEST_DIR}")
    print("=" * 60)
    test_volumes = discover_test_volumes(TEST_DIR)
    if not test_volumes:
        print(f"(no .dcm files found in {TEST_DIR})")
        return
    for dcm_path in test_volumes:
        infer_volume(model, dcm_path)
    print("\nAll done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer", action="store_true",
                        help="Skip training, run inference only (needs saved checkpoint)")
    args = parser.parse_args()
    main(infer_only=args.infer)
