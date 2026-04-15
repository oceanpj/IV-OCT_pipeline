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
#  Paths
# ══════════════════════════════════════════════════════════════════════════════

TRAIN_DCM  = ("/Users/oceanpunsalan/Data/Intravascular/Analysis/"
              "0B360D4D-3B16-4DCC-AD86-32361D1B47A9/"
              "0B360D4D-3B16-4DCC-AD86-32361D1B47A9.dcm")
TRAIN_MASK = ("/Users/oceanpunsalan/Data/Intravascular/Analysis/"
              "0B360D4D-3B16-4DCC-AD86-32361D1B47A9/mask.h5")
TEST_DCM   = ("/Users/oceanpunsalan/Data/Intravascular/Analysis/"
              "137D983E-2D45-4C63-AA65-5DC75F6860B5/"
              "137D983E-2D45-4C63-AA65-5DC75F6860B5.dcm")
TEST_OUT   = ("/Users/oceanpunsalan/Data/Intravascular/Analysis/"
              "137D983E-2D45-4C63-AA65-5DC75F6860B5/mask.h5")
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
MAX_FRAMES    = 156       # only use frames 0–155 (set to None to use all)
DEVICE        = (
    "mps"  if torch.backends.mps.is_available() else   # Apple Silicon
    "cuda" if torch.cuda.is_available()         else
    "cpu"
)

# ══════════════════════════════════════════════════════════════════════════════
#  Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_dicom_gray(path, max_frames=MAX_FRAMES):
    """Return (N, H, W) float32 in [0, 1]. Caps at max_frames if set."""
    px = pydicom.dcmread(path).pixel_array          # (N, H, W, 3) uint8
    if max_frames is not None:
        px = px[:max_frames]
    gray = np.stack(
        [cv2.cvtColor(px[i], cv2.COLOR_RGB2GRAY) for i in range(px.shape[0])],
        axis=0
    ).astype(np.float32) / 255.0
    return gray


def load_mask(path, key=MASK_KEY, max_frames=MAX_FRAMES):
    """Return (N, H, W) float32 binary mask. Caps at max_frames if set."""
    with h5py.File(path, 'r') as f:
        m = (f[key][()] > 0).astype(np.float32)
    if max_frames is not None:
        m = m[:max_frames]
    return m


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

def infer(model, dcm_path, out_h5_path):
    """Run inference on a full DICOM volume and save mask.h5."""
    model.eval()
    model.to(DEVICE)

    print(f"\nInference on : {dcm_path}")
    images = load_dicom_gray(dcm_path)         # (N, H, W) float32
    n = images.shape[0]
    pred_volume = np.zeros(images.shape, dtype=np.int8)

    with torch.no_grad():
        for i in range(n):
            img = torch.from_numpy(images[i]).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
            img = img.to(DEVICE)
            logit = model(img)                          # (1,1,H,W)
            prob  = torch.sigmoid(logit).squeeze().cpu().numpy()
            pred_volume[i] = (prob > 0.5).astype(np.int8)

            if i % 50 == 0:
                print(f"  frame {i:3d}/{n}  lumen px = {pred_volume[i].sum()}")

    # Save in identical format to the training mask.h5 (key '1' only)
    os.makedirs(os.path.dirname(out_h5_path), exist_ok=True)
    with h5py.File(out_h5_path, 'w') as f:
        f.create_dataset('1', data=pred_volume, dtype=np.int8, compression='gzip')

    print(f"Mask saved → {out_h5_path}")
    print(f"  Volume shape : {pred_volume.shape}")
    print(f"  Positive voxels : {pred_volume.sum()} / {pred_volume.size} "
          f"({100*pred_volume.mean():.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main(infer_only=False):
    model = UNet(in_ch=1, out_ch=1)

    if not infer_only:
        print("=" * 60)
        print("LOADING TRAINING DATA")
        print("=" * 60)
        images = load_dicom_gray(TRAIN_DCM)
        masks  = load_mask(TRAIN_MASK, key=MASK_KEY)
        print(f"Images : {images.shape}   range [{images.min():.2f}, {images.max():.2f}]")
        print(f"Masks  : {masks.shape}    positives = {masks.mean()*100:.1f}%\n")

        # Train / val split (stratified by frame index for even coverage)
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
    print(f"\nLoaded weights from {CKPT_PATH}")

    print("=" * 60)
    print("INFERENCE")
    print("=" * 60)
    infer(model, TEST_DCM, TEST_OUT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer", action="store_true",
                        help="Skip training, run inference only (needs saved checkpoint)")
    args = parser.parse_args()
    main(infer_only=args.infer)
