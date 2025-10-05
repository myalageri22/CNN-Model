# Fine-tune from BEST checkpoint with improved loss & sampling

import os, json, math
import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn.functional as F
from glob import glob
import matplotlib.pyplot as plt
!pip install -q --upgrade "torch==2.3.1+cu121" "torchvision==0.18.1+cu121" -f https://download.pytorch.org/whl/torch_stable.html
!pip install -q "monai[all]" nibabel
import monai
print(torch.__version__, monai.__version__)


from torch.cuda.amp import autocast, GradScaler

from monai.inferers import sliding_window_inference
from tqdm import tqdm

from sklearn.model_selection import train_test_split

from monai.data import PersistentDataset, DataLoader, decollate_batch
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.data.utils import pad_list_data_collate

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, NormalizeIntensityd,
    RandCropByPosNegLabeld, RandFlipd, RandRotate90d,
    ResizeWithPadOrCropd, Resized, EnsureTyped, Activations, AsDiscrete,
    Rand3DElasticd, RandAdjustContrastd
)
from monai.data import PersistentDataset, DataLoader, decollate_batch
from monai.networks.nets import UNet
from monai.losses import DiceLoss, TverskyLoss
from monai.metrics import DiceMetric
from monai.data.utils import pad_list_data_collate
from sklearn.model_selection import train_test_split

# --- reuse your paths (EDIT if needed) ---
# Assumes these exist from your earlier code:
DATADIR = "/content/drive/MyDrive/orig"
WORKDIR = "/content/drive/MyDrive/CNNProjectWorkFlow"
SPLIT_JSON = os.path.join(WORKDIR, "split.json")
CKPT_BEST = os.path.join(WORKDIR, "unet3d_best.ckpt")
CKPT_LAST = os.path.join(WORKDIR, "unet3d_last.ckpt")

# NEW cache dir for fine-tuning (so updated transforms don’t conflict with old cache)
CACHE_DIR_FT = os.path.join(WORKDIR, "cache_ft")
os.makedirs(CACHE_DIR_FT, exist_ok=True)

# --- load the saved split (don’t reshuffle) ---
#with open(SPLIT_JSON, "r") as f:
#splits = json.load(f)
#train_files, val_files = splits["train"], splits["val"]
#print(f"[FT] Train cases: {len(train_files)} | Val cases: {len(val_files)}")
# --- sanity check the paths
print("DATADIR:", DATADIR, "exists?", os.path.isdir(DATADIR))
print("WORKDIR:", WORKDIR, "exists?", os.path.isdir(WORKDIR))

# ------------- HELPER --------------
def remove_small_objects_mask(mask, min_size=64):
    # mask: numpy array 0/1
    lab, n = ndi.label(mask)
    sizes = ndi.sum(mask, lab, range(1, n+1))
    out = np.zeros_like(mask)
    for i, s in enumerate(sizes, start=1):
        if s >= min_size:
            out[lab == i] = 1
    return out


# show what’s inside WORKDIR (helps catch /WorkFlow vs /Workflow mistakes)
if os.path.isdir(WORKDIR):
    print("WORKDIR contents:", os.listdir(WORKDIR))

# --- build or load split.json robustly
os.makedirs(WORKDIR, exist_ok=True)  # ensure exists before writing
SPLIT_JSON = os.path.join(WORKDIR, "split.json")

if not os.path.exists(SPLIT_JSON):
    # Try to find any existing split.json elsewhere (maybe different folder spelling)
    candidates = glob("/content/drive/MyDrive/**/split.json", recursive=True)
    if candidates:
        print("Found existing split.json candidates:", candidates)
        # pick the most recent one
        candidates.sort(key=os.path.getmtime, reverse=True)
        src = candidates[0]
        print("Copying most recent split.json from:", src)
        import shutil
        shutil.copy(src, SPLIT_JSON)

if not os.path.exists(SPLIT_JSON):
    # No split found anywhere → build from DATADIR now
    images = sorted(glob(os.path.join(DATADIR, "*orig.nii.gz")))
    labels = sorted(glob(os.path.join(DATADIR, "*orig_seg.nii.gz")))
    print(f"Discovered: {len(images)} images, {len(labels)} labels in {DATADIR}")
    assert len(images) == len(labels) and len(images) > 0, \
        "No matching image/label pairs found in DATADIR. Check patterns and DATADIR path."

    data = [{"image": i, "label": l} for i, l in zip(images, labels)]
    train_files, val_files = train_test_split(data, test_size=0.2, random_state=42)

    with open(SPLIT_JSON, "w") as f:
        json.dump({"train": train_files, "val": val_files}, f, indent=2)
    print("Created split.json at:", SPLIT_JSON)
else:
    print("Loading split from:", SPLIT_JSON)
    with open(SPLIT_JSON, "r") as f:
        splits = json.load(f)
    train_files, val_files = splits["train"], splits["val"]

print(f"[FT] Train cases: {len(train_files)} | Val cases: {len(val_files)}")


# Hyperparameters for fine-tuning
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pixdim = (1.2, 1.2, 1.2)
roi_size = (64, 192, 160)          # divisible by 16
batch_size = 2
val_bs = 1
max_epochs = 50                       # fine-tune for 50 more
val_every = 1
best_dice = -1.0

pos_weight = None
try:
    import nibabel as nib
    pos_vox, neg_vox = 0, 0
    for item in train_files:
        lab = nib.load(item["label"]).get_fdata()
        pos_vox += (lab > 0.5).sum()
        neg_vox += (lab <= 0.5).sum()
    pos_weight_val = float(max(1.0, neg_vox / (pos_vox + 1e-9)))
    pos_weight_val = min(pos_weight_val, 200.0)
    pos_weight = torch.tensor([pos_weight_val], device=device)
    print(f"[pos_weight] pos_vox={int(pos_vox)} neg_vox={int(neg_vox)} pos_weight={pos_weight_val:.3f}")
except Exception as e:
    pos_weight = None
    print("[pos_weight] Could not compute pos_weight; continuing without it. Error:", e)

# Increase positive sampling & number of patches (helps sparse vessels)
pos_patches = 4
neg_patches = 1
num_samples = 6

# Toggle extra augmentations after model is already learning
USE_ELASTIC = True
USE_CONTRAST = True

from monai.transforms import Rand3DElasticd, RandAdjustContrastd

aug_list = []
if USE_ELASTIC:
    aug_list.append(
        Rand3DElasticd(
            keys=["image", "label"],
            prob=0.20,
            sigma_range=(5, 8),          # smoothing of the displacement field (voxels)
            magnitude_range=(50, 100),    # deformation intensity (larger -> stronger)
            mode=("bilinear", "nearest"), # image / label interpolation
            padding_mode="zeros",         # how to pad outside area
            # NOTE: no 'as_tensor_output' here
        )
    )
if USE_CONTRAST:
    aug_list.append(
        RandAdjustContrastd(keys=["image"], prob=0.30, gamma=(0.7, 1.5))
    )


# Transforms (FT)
train_transforms_ft = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear","nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=200, b_min=0.0, b_max=1.0, clip=True),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),

        # More positive-biased patch sampling + more patches per volume
        RandCropByPosNegLabeld(
            keys=["image","label"], label_key="label",
            spatial_size=roi_size, pos=pos_patches, neg=neg_patches,
            num_samples=num_samples, image_key="image",
            image_threshold=0.0, allow_smaller=True
        ),

        # Optional extra augmentations for generalization
        *aug_list,

        # Guarantee exact shape for batching & UNet skip-concats
        Resized(keys=["image","label"], spatial_size=roi_size, mode=("trilinear","nearest")),
        EnsureTyped(keys=["image","label"]),
    ]
)

val_transforms_ft = Compose(
    [
        LoadImaged(keys=["image","label"]),
        EnsureChannelFirstd(keys=["image","label"]),
        Orientationd(keys=["image","label"], axcodes="RAS"),
        Spacingd(keys=["image","label"], pixdim=pixdim, mode=("bilinear","nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=200, b_min=0.0, b_max=1.0, clip=True),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ResizeWithPadOrCropd(keys=["image","label"], spatial_size=roi_size),
        EnsureTyped(keys=["image","label"]),
    ]
)

# Datasets & Loaders (Persistent cache for FT)
"""
train_ds_ft = PersistentDataset(data=train_files, transform=train_transforms_ft, cache_dir=CACHE_DIR_FT)
val_ds_ft   = PersistentDataset(data=val_files,   transform=val_transforms_ft,   cache_dir=CACHE_DIR_FT)

train_loader = DataLoader(
    train_ds_ft, batch_size=batch_size, shuffle=True, num_workers=2,
    collate_fn=pad_list_data_collate, pin_memory=torch.cuda.is_available()
)
val_loader   = DataLoader(
    val_ds_ft, batch_size=val_bs, shuffle=False, num_workers=1,
    collate_fn=pad_list_data_collate, pin_memory=torch.cuda.is_available()
)
"""
# --- Datasets & Loaders (use RAM cache to avoid torch.save path) ---
from monai.data import CacheDataset

# If you earlier created a disk cache, it's safe to ignore/remove it now.
# import shutil; shutil.rmtree(CACHE_DIR_FT, ignore_errors=True)

train_ds_ft = CacheDataset(
    data=train_files, transform=train_transforms_ft, cache_rate=1.0  # cache everything in RAM
)
val_ds_ft = CacheDataset(
    data=val_files, transform=val_transforms_ft, cache_rate=1.0
)

train_loader = DataLoader(
    train_ds_ft, batch_size=batch_size, shuffle=True, num_workers=2,
    collate_fn=pad_list_data_collate, pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_ds_ft, batch_size=val_bs, shuffle=False, num_workers=1,
    collate_fn=pad_list_data_collate, pin_memory=torch.cuda.is_available()
)

# Sanity check after swap:
b = next(iter(train_loader))
print("[FT] Batch shapes:", b["image"].shape, b["label"].shape)

# Sanity check
b = next(iter(train_loader))
print("[FT] Batch shapes:", b["image"].shape, b["label"].shape)

# Model, Loss, Optimizer, Scheduler, Metric
model = UNet(
    spatial_dims=3, in_channels=1, out_channels=1,
    channels=(16,32,64,128,256), strides=(2,2,2,2), num_res_units=2
).to(device)

# Switch to Tversky to emphasize recall (thin vessels); keep a BCE component for stability
loss_tversky = TverskyLoss(alpha=0.3, beta=0.7, sigmoid=True)  # beta>alpha => penalize FN more
if pos_weight is not None:
    bce_logits = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
else:
    bce_logits = torch.nn.BCEWithLogitsLoss()

def seg_loss(logits, target):
    return 0.7 * loss_tversky(logits, target) + 0.3 * bce_logits(logits, target)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)  # lower LR for fine-tuning
scaler = GradScaler()
torch.backends.cudnn.benchmark = True

accum_steps = 1 # increase to 2-4 for effective larger batch
clip_grad_norm = 1.0  # gradient clipping


scheduler = None
USE_ONECYCLE = True

dice_metric = DiceMetric(include_background=False, reduction="mean")
post_pred   = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
post_label  = Compose([AsDiscrete(threshold=0.5)])

# Resume from BEST checkpoint
assert os.path.exists(CKPT_BEST), f"Best checkpoint not found at {CKPT_BEST}"
ckpt = torch.load(CKPT_BEST, map_location=device)
model.load_state_dict(ckpt["model"])
best_dice = ckpt.get("best_dice", best_dice)
start_epoch = ckpt.get("epoch", -1) + 1
print(f"[FT] Resumed from BEST: epoch={start_epoch}, best_dice={best_dice:.4f}")

# Fine-tuning loop
train_losses, val_dices, lrs = [], [], []

if USE_ONECYCLE:
    steps_per_epoch = len(train_loader)
    max_lr = 5e-5
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=steps_per_epoch * max_epochs,
        anneal_strategy="cos",
        pct_start=0.1
    )
else:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5, verbose=True
    )
for epoch in range(start_epoch, start_epoch + max_epochs):
    model.train()
    running, steps = 0.0, 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} train", leave=False)
    for i, batch in enumerate(pbar):
        x = batch["image"].to(device)
        y = batch["label"].to(device).float()

        with autocast():
            logits = model(x)
            loss = seg_loss(logits, y) / accum_steps

        scaler.scale(loss).backward()
        running += loss.item() * accum_steps
        steps += 1

        if (i + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if USE_ONECYCLE:
                scheduler.step()  # step per optimizer update

        pbar.set_postfix({"loss": f"{running/max(1,steps):.4f}", 
                          "lr": optimizer.param_groups[0]["lr"]})

    avg_loss = running / max(1, steps)
    train_losses.append(avg_loss)
    lrs.append(optimizer.param_groups[0]["lr"])
    print(f"Epoch {epoch} | train loss: {avg_loss:.4f} | lr: {lrs[-1]:.2e}")

    # ----------------- VALIDATION -----------------
    model.eval(); dice_metric.reset()
    with torch.no_grad():
        for vb in tqdm(val_loader, desc=f"Epoch {epoch} val", leave=False):
            vx = vb["image"].to(device)
            vy = vb["label"].to(device).float()

            # Sliding window inference
            vlogits = sliding_window_inference(
                inputs=vx,
                roi_size=roi_size,
                sw_batch_size=1,
                predictor=model,
                overlap=0.5
            )

            pr_list = [post_pred(p) for p in decollate_batch(vlogits)]
            gt_list = [post_label(g) for g in decollate_batch(vy)]
            
            # Optional: remove small connected components
            # pr_list = [remove_small_objects_mask(p.cpu().numpy(), min_size=64) for p in pr_list]

            dice_metric(y_pred=pr_list, y=gt_list)

        val_dice = dice_metric.aggregate().item()
        dice_metric.reset()
        val_dices.append(val_dice)
        print(f"  val dice: {val_dice:.4f}")

    if not USE_ONECYCLE:
        scheduler.step(val_dice)

    # Save LAST every epoch
    torch.save(
        {"epoch": epoch, "model": model.state_dict(),
         "optimizer": optimizer.state_dict(), 
         "scheduler": scheduler.state_dict() if scheduler is not None else None,
         "best_dice": best_dice},
        CKPT_LAST,
    )

    # Save BEST on improvement
    if val_dice > best_dice:
        best_dice = val_dice
        torch.save(
            {"epoch": epoch, "model": model.state_dict(),
             "optimizer": optimizer.state_dict(), 
             "scheduler": scheduler.state_dict() if scheduler is not None else None,
             "best_dice": best_dice},
            CKPT_BEST,
        )
        print(" new BEST saved")

# --- Curves ---
plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.plot(train_losses); plt.title("Train loss"); plt.xlabel("epoch"); plt.ylabel("loss")
plt.subplot(1,3,2); plt.plot(val_dices);    plt.title("Val Dice");  plt.xlabel("epoch"); plt.ylabel("dice")
plt.subplot(1,3,3); plt.plot(lrs);          plt.title("LR");        plt.xlabel("epoch"); plt.ylabel("lr")
plt.tight_layout(); plt.show()

print(f"[FT] Best Val Dice (this run): {best_dice:.4f}")
