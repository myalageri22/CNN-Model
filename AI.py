#3D Binary U-Net CNN
!pip install -q "monai[all]" nibabel

 #Imports and Preliminary
import os
from glob import glob
import matplotlib.pyplot as plt
import torch
import numpy as np

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
)
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.data.utils import pad_list_data_collate

from google.colab import drive
drive.mount('/content/drive')
datapath = '/content/drive/MyDrive/orig'

images = sorted(glob(os.path.join(datapath, "*orig.nii.gz")))
labels = sorted(glob(os.path.join(datapath, "*orig_seg.nii.gz")))
print("Found images:", len(images), "labels:", len(labels))
assert len(images) == len(labels) and len(images) > 0, "Mismatch or no files found."

data_dicts = [{"image": img, "label": seg} for img, seg in zip(images, labels)]

train_files, val_files = train_test_split(data_dicts, test_size=0.2, random_state=42)

is_3d = True
pixdim = (1.2, 1.2, 1.2)
# ROI (must be divisible by 16 because UNet downsamples 4 times -> 2^4=16)
roi_size = (64, 192, 160)  # (D, H, W) - all divisible by 16
num_samples = 2            # number of patches sampled per volume per iteration
batch_size = 2
val_batch_size = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
 #Prepping Dataset
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),               # shape -> (C, D, H, W)
    Orientationd(keys=["image", "label"], axcodes="RAS"),      # consistent orientation
    Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear","nearest")),
    ScaleIntensityd(keys=["image"]),                           # robust intensity scaling
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    # sample positive/negative patches of fixed size -> ensures consistent shapes
    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=roi_size,
        pos=2,
        neg=1,
        num_samples=num_samples,
        image_key="image",
        image_threshold=0.0,
        allow_smaller= True
    ),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(1, 2)),
    RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.01),
    EnsureTyped(keys=["image", "label"]),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear","nearest")),
    ScaleIntensityd(keys=["image"]),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    # deterministic center pad/crop to roi_size so shape matches training patches
    ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=roi_size),
    EnsureTyped(keys=["image", "label"]),
])
 #Loading Dataset
train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0)
val_ds   = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=pad_list_data_collate)
val_loader   = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False, num_workers=1, collate_fn=pad_list_data_collate)

batch = next(iter(train_loader))
print("Sanity check batch shapes (image, label):", batch["image"].shape, batch["label"].shape)


#Model: 
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16,32,64,128,256),
    strides=(2,2,2,2),
    num_res_units=2,
).to(device)

loss_function = DiceLoss(sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
dice_metric = DiceMetric(include_background=False, reduction="mean")

post_pred  = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
post_label = Compose([AsDiscrete(threshold=0.5)])

# === Training loop with tracking and plotting ===
max_epochs = 30
val_interval = 1

train_losses = []
val_dice_scores = []

save_dir = "/content/models"
os.makedirs(save_dir, exist_ok=True)
model_path = os.path.join(save_dir, "unet3d_best.pth")

best_val = -1.0

for epoch in range(max_epochs):
    print(f"\n--- Epoch {epoch+1}/{max_epochs} ---")
    model.train()
    epoch_loss = 0.0
    num_steps = 0

    for batch_data in train_loader:
        inputs = batch_data["image"].to(device)   # [B,1,D,H,W]
        labels = batch_data["label"].to(device).float()  # float for DiceLoss

        optimizer.zero_grad()
        outputs = model(inputs)                    # raw logits [B,1,D,H,W]
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_steps += 1

    avg_loss = epoch_loss / max(1, num_steps)
    train_losses.append(avg_loss)
    print(f"Train loss: {avg_loss:.4f}")

    # Validation
    if (epoch + 1) % val_interval == 0:
        model.eval()
        dice_metric.reset()
        with torch.no_grad():
            for val_batch in val_loader:
                val_inputs = val_batch["image"].to(device)
                val_labels = val_batch["label"].to(device).float()

                val_logits = model(val_inputs)   # raw logits
                # decollate batch into list of single items
                val_logits_list = decollate_batch(val_logits)
                val_labels_list = decollate_batch(val_labels)

                val_preds = [post_pred(x) for x in val_logits_list]     # list of binary preds
                val_gts   = [post_label(y) for y in val_labels_list]    # list of binary gts

                # update metric with lists
                dice_metric(y_pred=val_preds, y=val_gts)

            val_metric = dice_metric.aggregate().item()
            dice_metric.reset()
            val_dice_scores.append(val_metric)
            print(f"Validation Dice: {val_metric:.4f}")

            # save model if improved
            if val_metric > best_val:
                best_val = val_metric
                torch.save(model.state_dict(), model_path)
                print("Saved best model.")

#Plot training curves
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(train_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(val_dice_scores, label="Val Dice")
plt.xlabel("Epoch")
plt.ylabel("Dice")
plt.legend()
plt.show()

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print("Loaded best model.")
else:
  print("error")
