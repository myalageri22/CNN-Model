#Install & imports
!pip install -q "monai[all]" nibabel

import os
import json
import math
from glob import glob
import matplotlib.pyplot as plt
import torch
import numpy as np

from torch.cuda.amp import autocast, GradScaler

from sklearn.model_selection import train_test_split

import monai
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityd,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandGaussianNoised,
    ResizeWithPadOrCropd,
    EnsureTyped,
    Activations,
    AsDiscrete,
    Resized,
    ScaleIntensityRanged,
    ScaleIntensityRangePercentilesd,
)
from monai.data import PersistentDataset, DataLoader, decollate_batch
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.data.utils import pad_list_data_collate

mri_intensity = [
    ScaleIntensityRangePercentilesd(
        keys=["image"], lower=1, upper=99, b_min=0.0, b_max=1.0, clip=True
    ),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
]

#Mount Drive (if needed)
from google.colab import drive
drive.mount('/content/drive')
# datapath = "/content/drive/MyDrive/orig"
DATADIR = '/content/drive/MyDrive/orig'
WORKDIR = "/content/drive/MyDrive/CNNProjectWorkflow"      # where we store caches, splits, checkpoints, plots
print(WORKDIR)

SPLIT_JSON = os.path.join(WORKDIR, "split.json")   # we’ll save train/val file lists here
CACHE_DIR  = os.path.join(WORKDIR, "cache")        # PersistentDataset cache
CKPT_BEST  = os.path.join(WORKDIR, "unet3d_best.ckpt")
CKPT_LAST  = os.path.join(WORKDIR, "unet3d_last.ckpt")

os.makedirs(CACHE_DIR, exist_ok=True)

#DIAGNOSTICS (prints you should verify)
print("DATADIR exists? ", os.path.isdir(DATADIR), "->", DATADIR)
print("WORKDIR exists? ", os.path.isdir(WORKDIR), "->", WORKDIR)
print("CACHE_DIR exists?", os.path.isdir(CACHE_DIR), "->", CACHE_DIR)
print("SPLIT_JSON path ->", SPLIT_JSON)
try:
    _test_path = os.path.join(WORKDIR, "_write_test.txt")
    with open(_test_path, "w") as f:
        f.write("ok")
    print("Write test OK:", _test_path)
    os.remove(_test_path)
except Exception as e:
    print("Write test FAILED:", repr(e))

#File discovery
if not os.path.exists(SPLIT_JSON):
    from glob import glob
    images = sorted(glob(os.path.join(DATADIR, "*orig.nii.gz")))
    labels = sorted(glob(os.path.join(DATADIR, "*orig_seg.nii.gz")))
    assert len(images) == len(labels) and len(images) > 0, "No matching image/label pairs found."

    data = [{"image": i, "label": l} for i, l in zip(images, labels)]
    from sklearn.model_selection import train_test_split
    train_files, val_files = train_test_split(data, test_size=0.2, random_state=42)

    # Make SURE the parent folder exists (belt-and-suspenders)
    os.makedirs(os.path.dirname(SPLIT_JSON), exist_ok=True)

    with open(SPLIT_JSON, "w") as f:
        json.dump({"train": train_files, "val": val_files}, f, indent=2)
    print("Created split.json at:", SPLIT_JSON)
else:
    with open(SPLIT_JSON, "r") as f:
        splits = json.load(f)
    train_files, val_files = splits["train"], splits["val"]
    print("Loaded split.json from:", SPLIT_JSON)

print(f"Train cases: {len(train_files)} | Val cases: {len(val_files)}")



#Create list of dicts for MONAI
#data_dicts = [{"image": img, "label": seg} for img, seg in zip(images, labels)]

# Train/Val split
#train_files, val_files = train_test_split(data_dicts, test_size=0.2, random_state=42)
# After loading or creating split.json:
with open(SPLIT_JSON) as f:
    splits = json.load(f)
train_files, val_files = splits["train"], splits["val"]
# REMOVE the second construction of data_dicts and the second train_test_split.


#Hyperparameters
is_3d = True
pixdim = (1.2, 1.2, 1.2)
# ROI (must be divisible by 16 because UNet downsamples 4 times -> 2^4=16)
roi_size = (64, 160, 160) #64,  192, 160  # (D, H, W) - all divisible by 16
num_samples = 4 #46           # number of patches sampled per volume per iteration
batch_size = 1 #2
val_bs = 1
max_epochs = 10
val_every = 1
start_epoch = 0
best_dice = -1.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

train_transforms = Compose([
    LoadImaged(keys=["image","label"]),
    EnsureChannelFirstd(keys=["image","label"]),
    Orientationd(keys=["image","label"], axcodes="RAS"),
    Spacingd(keys=["image","label"], pixdim=pixdim, mode=("bilinear","nearest")),
    # MRI users: comment the CT window below
    ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=200, b_min=0.0, b_max=1.0, clip=True),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),

    RandCropByPosNegLabeld(
        keys=["image","label"], label_key="label",
        spatial_size=roi_size, pos=4, neg=1, num_samples=num_samples,
        image_key="image", image_threshold=0.0, allow_smaller=True
    ),
    RandFlipd(keys=["image","label"], prob=0.5, spatial_axis=0),
    RandRotate90d(keys=["image","label"], prob=0.5, spatial_axes=(1,2)),
    RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.01),
    EnsureTyped(keys=["image","label"]),
])

val_transforms = Compose([
    LoadImaged(keys=["image","label"]),
    EnsureChannelFirstd(keys=["image","label"]),
    Orientationd(keys=["image","label"], axcodes="RAS"),
    Spacingd(keys=["image","label"], pixdim=pixdim, mode=("bilinear","nearest")),
    ScaleIntensityRangePercentilesd(keys=["image"], lower=1, upper=99, b_min=0.0, b_max=1.0, clip=True),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    EnsureTyped(keys=["image","label"]),
])

#Datasets & Loaders (CacheDataset for speed)
os.makedirs(CACHE_DIR, exist_ok=True)
train_ds = PersistentDataset(data=train_files, transform=train_transforms, cache_dir=CACHE_DIR)
val_ds   = PersistentDataset(data=val_files,   transform=val_transforms,   cache_dir=CACHE_DIR)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2,
                          collate_fn=pad_list_data_collate, pin_memory=torch.cuda.is_available())
val_loader   = DataLoader(val_ds,   batch_size=val_bs,      shuffle=False, num_workers=1,
                          collate_fn=pad_list_data_collate, pin_memory=torch.cuda.is_available())
# Quick sanity check: print one batch shape
batch = next(iter(train_loader))
print("Sanity check batch shapes (image, label):", batch["image"].shape, batch["label"].shape)
# Expect: (B, C, D, H, W) e.g. (2,1,64,192,192)

#Model, loss, optimizer, metric
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16,32,64,128,256),
    strides=(2,2,2,2),
    num_res_units=2,
).to(device)

loss_dice = DiceLoss(sigmoid=True)
loss_bce  = torch.nn.BCEWithLogitsLoss()
def seg_loss(logits, target):
    return 0.7 * loss_dice(logits, target) + 0.3 * loss_bce(logits, target)

#loss_function = DiceLoss(sigmoid=True) #+ torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay = 1e-5)
dice_metric = DiceMetric(include_background=False, reduction="mean")
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5, verbose=True)



# Post-processing for validation
post_pred  = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
post_label = Compose([AsDiscrete(threshold=0.5)])
scaler = GradScaler(enabled=torch.cuda.is_available())  # mixed precision


#Training loop with tracking and plotting
#max_epochs = 75
#val_interval = 1

if os.path.exists(CKPT_LAST):
    ckpt = torch.load(CKPT_LAST, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    start_epoch = ckpt["epoch"] + 1
    best_dice   = ckpt.get("best_dice", best_dice)
    print(f"Resumed from epoch {start_epoch}, best_dice={best_dice:.4f}")

#train_losses = []
#val_dice_scores = []
train_losses, val_dices, lrs = [], [], []

patience = 15     # Early stopping patience (epochs without improvement)
wait     = 0

#save_dir = "/content/models"
#os.makedirs(save_dir, exist_ok=True)
#model_path = os.path.join(save_dir, "unet3d_best.pth")

#best_val = -1.0

for epoch in range(start_epoch, max_epochs):
    print(f"\n--- Epoch {epoch+1}/{max_epochs} ---")
    model.train()
    running = 0.0
    steps = 0
    #epoch_loss = 0.0
    #num_steps = 0

    for batch_data in train_loader:
        x = batch_data["image"].to(device)   # [B,1,D,H,W]
        y = batch_data["label"].to(device).float()  # float for DiceLoss

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=torch.cuda.is_available()):
            logits = model(x)            # raw logits
            loss = seg_loss(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        #outputs = model(inputs)                    # raw logits [B,1,D,H,W]
        #loss = loss_function(outputs, labels)
        #loss.backward()
        #optimizer.step()
        running += loss.item()
        steps   += 1

        #epoch_loss += loss.item()
        #num_steps += 1
        with torch.no_grad():
              x = batch["image"]              # before .to(device) is fine
              x_min, x_max = x.min().item(), x.max().item()
              x_mean, x_std = x.mean().item(), x.std().item()
              print(f"[INTENSITY] min={x_min:.3f} max={x_max:.3f} mean={x_mean:.3f} std={x_std:.3f}")


    avg_loss = running / max(1, steps)
    train_losses.append(avg_loss)
    lrs.append(optimizer.param_groups[0]["lr"])
    print("debug single loss:", float(seg_loss(logits[:1], y[:1]).item()))
    print(f"Train loss: {avg_loss:.4f}")

    # --- Validation ---
    if (epoch + 1) % val_every == 0:
        model.eval()
        dice_metric.reset()
        with torch.no_grad(), autocast(enabled=torch.cuda.is_available()):
            for vb in val_loader:
                vx = vb["image"].to(device)
                vy = vb["label"].to(device).float()

                vlogits = model(vx)   # raw logits
                # decollate batch into list of single items
                #val_logits_list = decollate_batch(val_logits)
                #val_labels_list = decollate_batch(val_labels)
                pr_list = [post_pred(p)  for p in decollate_batch(vlogits)]
                gt_list = [post_label(g) for g in decollate_batch(vy)]

                #val_preds = [post_pred(x) for x in val_logits_list]     # list of binary preds
                #val_gts   = [post_label(y) for y in val_labels_list]    # list of binary gts

                # update metric with lists
                dice_metric(y_pred=pr_list, y=gt_list)

            val_dice = dice_metric.aggregate().item()
            dice_metric.reset()
            val_dices.append(val_dice)
            print(f"Validation Dice: {val_dice:.4f}")
            # Step LR scheduler on metric
        scheduler.step(val_dice)

        # Save "last" checkpoint every epoch
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_dice": best_dice,
            },
            CKPT_LAST,
        )

        # Save "best" checkpoint on improvement
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_dice": best_dice,
                },
                CKPT_BEST,
            )
            print("  Saved new BEST checkpoint.")

        # Early stopping
        if val_dice >= max(val_dices[:-1], default=-1):  # improved
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(" Early stopping triggered.")
                break

# 6) Curves

plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.plot(train_losses); plt.title("Train loss"); plt.xlabel("epoch"); plt.ylabel("loss")
plt.subplot(1,3,2); plt.plot(val_dices);    plt.title("Val Dice");  plt.xlabel("epoch"); plt.ylabel("dice")
plt.subplot(1,3,3); plt.plot(lrs);          plt.title("LR");        plt.xlabel("epoch"); plt.ylabel("lr")
plt.tight_layout(); plt.show()

print(f"Best Val Dice = {best_dice:.4f}")
