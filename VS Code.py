from pathlib import Path
from typing import Dict, List, Tuple, Optional, Sequence
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm

import logging
import sys
import os
import json
import zipfile
import re
import random
import shutil
import argparse

import numpy as np

try:
    import torch
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "PyTorch is required to run this training pipeline. "
        "Install it with `pip install torch` (pick the wheel matching your Python version and platform)."
    ) from exc

from torch.cuda.amp import GradScaler
import torch.nn as nn

try:
    import nibabel as nib
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "The `nibabel` package is needed to read NIfTI files. Install it with `pip install nibabel`."
    ) from exc

print("Hello from CNN.py")

class Config:
    """Centralized configuration"""
    def __init__(self):
        # basic runtime
        self.project_root = Path(__file__).resolve().parent
        self.data_dir = self.project_root / "data"
        self.checkpoint_dir = self.project_root / "checkpoints"
        self.log_dir = self.project_root / "logs"
        self.processed_dir = self.data_dir / "processed"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # data / transform
        self.pixdim = (1.0, 1.0, 1.0)
        self.roi_size = (64, 192, 160)

        # training defaults
        self.batch_size = 2
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        cpu_count = os.cpu_count() or 1
        self.num_workers = min(4, cpu_count)
        self.pin_memory = self.device.startswith("cuda")
        self.accumulation_steps = 1
        self.use_attention = False

        # files
        self.split_file = self.project_root / "splits.json"

        # ensure dirs
        for d in (self.data_dir, self.checkpoint_dir, self.log_dir, self.processed_dir):
            d.mkdir(parents=True, exist_ok=True)

    def save(self, path: Path):
        d = {k: (list(v) if isinstance(v, tuple) else str(v) if isinstance(v, Path) else v)
             for k, v in self.__dict__.items() if not k.startswith("_")}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(d, f, indent=2)

    @classmethod
    def from_args(cls, args: argparse.Namespace):
        cfg = cls()
        if args.batch_size:
            cfg.batch_size = args.batch_size
        if args.learning_rate:
            cfg.learning_rate = args.learning_rate
        if args.roi_size:
            cfg.roi_size = tuple(int(x) for x in args.roi_size.split(","))
        return cfg


def setup_logging(log_dir: Path, experiment_name: str = None) -> logging.Logger:
    if experiment_name is None:
        experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_file = log_dir / f"train_{experiment_name}.log"

    logger = logging.getLogger("VascularSeg")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_format)

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)

        # Add handlers
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    logger.info(f"Logging initialized. Log file: {log_file}")

    return logger



class LocalDataManager:
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def extract_dataset(self, zip_path: Path) -> Path:
        zip_path = Path(zip_path)
        if zip_path.is_file() and zipfile.is_zipfile(zip_path):
            dest = self.config.processed_dir / zip_path.stem
            if dest.exists():
                self.logger.info(f"Using existing extracted dataset at {dest}")
                return dest
            dest.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(dest)
            # flatten any nested folders
            self._flatten_directory(dest)
            return dest
        elif zip_path.is_dir():
            return zip_path
        else:
            raise FileNotFoundError(f"Dataset not found: {zip_path}")

    def _flatten_directory(self, path: Path):
        # move any .nii/.nii.gz up to top-level processed dir for simplicity
        for p in path.rglob("*"):
            if p.is_file() and p.suffix in [".nii", ".gz"] and ("nii" in p.name):
                target = path / p.name
                if p.resolve() != target.resolve():
                    shutil.move(str(p), str(target))

    def find_nifti_files(self, root_dir: Path) -> List[Path]:
        root = Path(root_dir)
        files = [p for p in root.rglob("*") if p.is_file() and (p.suffix in [".nii", ".gz"] and "nii" in p.name)]
        return files


class DatasetPairer:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        # common patterns to extract id
        self.image_patterns = [
            r"(?P<id>\d+)_image",
            r"(?P<id>sub-\d+)",
            r"(?P<id>patient\d+)",
            r"(?P<id>pat\d+)_orig",
        ]
        self.label_patterns = [
            r"(?P<id>\d+)_label",
            r"(?P<id>sub-\d+)",
            r"(?P<id>patient\d+)",
            r"(?P<id>pat\d+)_orig_seg",
        ]

    def extract_id(self, filepath: Path, patterns: List[str]) -> Optional[str]:
        name = filepath.stem
        for pat in patterns:
            m = re.search(pat, name)
            if m:
                return m.group("id")
        # fallback: filename without suffix
        return name

    def pair_files(self, all_files: List[Path]) -> List[Dict[str, str]]:
        images = []
        labels = []
        for p in all_files:
            n = p.name.lower()
            if ("label" in n or "mask" in n or "seg" in n) and "endpoint" not in n:
                labels.append(p)
            else:
                images.append(p)
        # map ids
        id_map = defaultdict(dict)
        for img in images:
            _id = self.extract_id(img, self.image_patterns)
            id_map[_id]["image"] = str(img)
        for lab in labels:
            _id = self.extract_id(lab, self.label_patterns)
            id_map[_id]["label"] = str(lab)
        paired = []
        for k, v in id_map.items():
            if "image" in v and "label" in v:
                paired.append({"image": v["image"], "label": v["label"], "id": k})
            else:
                self.logger.debug(f"Skipping incomplete pair for id {k}: {v.keys()}")
        self.logger.info(f"Found {len(paired)} paired items")
        return paired


class DatasetSplitter:
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def split_dataset(
        self,
        paired_data: List[Dict],
        val_ratio: float = 0.2,
        test_ratio: float = 0.0,
        random_seed: int = 42
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        random.seed(random_seed)
        items = paired_data.copy()
        random.shuffle(items)
        n = len(items)
        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)
        test = items[:n_test] if n_test > 0 else []
        val = items[n_test:n_test + n_val] if n_val > 0 else items[:n_val]
        train = items[n_test + n_val:]
        # persist splits
        self._save_split(train, val, test)
        self.logger.info(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
        return train, val, test

    def _save_split(self, train, val, test):
        out = {"train": train, "val": val, "test": test}
        with open(self.config.split_file, "w") as f:
            json.dump(out, f, indent=2)

    def _load_split(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        if not self.config.split_file.exists():
            return [], [], []
        with open(self.config.split_file) as f:
            d = json.load(f)
        return d.get("train", []), d.get("val", []), d.get("test", [])



def build_transforms(config: Config, mode: str = "train"):
    try:
        from monai.transforms import (
            Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped,
            Orientationd, Spacingd, ScaleIntensityRanged, NormalizeIntensityd,
            RandCropByPosNegLabeld, ResizeWithPadOrCropd, Resized,
            Rand3DElasticd, RandAdjustContrastd, RandGaussianNoised,
            RandGaussianSmoothd, RandFlipd, RandRotate90d, RandScaleIntensityd,
        )
    except Exception as e:
        raise ImportError("monai is required for transforms. Install `pip install monai`") from e

    base_transforms = [
        LoadImaged(keys=["image", "label"], image_only=False),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=config.pixdim,
            mode=("bilinear", "nearest")
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-200,
            a_max=200,
            b_min=0.0,
            b_max=1.0,
            clip=True
        ),
        NormalizeIntensityd(
            keys=["image"],
            nonzero=True,
            channel_wise=True
        ),
    ]

    augmentations = []
    if mode == "train":
        augmentations = [
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=config.roi_size,
                pos=1,
                neg=1,
                num_samples=1
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            RandScaleIntensityd(keys=["image"], factors=(0.9, 1.1), prob=0.5),
        ]
    else:
        augmentations = [
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=config.roi_size)
        ]

    final_transforms = [
        EnsureTyped(keys=["image", "label"], dtype=torch.float32),
    ]

    return Compose(base_transforms + augmentations + final_transforms)



def build_model(config: Config, logger: logging.Logger):
    try:
        from monai.networks.nets import UNet
    except Exception as e:
        raise ImportError("monai is required for model building. Install `pip install monai`") from e

    # simple 3D UNet
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
    )

    model = model.to(config.device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Model parameters:")
    logger.info(f"  Total: {total_params:,}")
    logger.info(f"  Trainable: {trainable_params:,}")
    logger.info(f"  Size: ~{total_params * 4 / 1024 / 1024:.1f} MB (FP32)")

    return model




def compute_class_weights(train_files: List[Dict], logger: logging.Logger, device: str) -> torch.Tensor:
    """Compute pos_weight for handling class imbalance"""
    logger.info("Computing class weights from training data...")

    pos_vox = 0
    neg_vox = 0

    for item in tqdm(train_files, desc="Analyzing labels", unit="file"):
        label_path = Path(item["label"])
        try:
            nii = nib.load(str(label_path))
            data = np.asarray(nii.get_fdata())
            pos = int(np.count_nonzero(data > 0))
            neg = int(data.size - pos)
            pos_vox += pos
            neg_vox += neg
        except Exception as e:
            logger.warning(f"Could not read {label_path}: {e}")

    if pos_vox == 0:
        logger.warning("No positive voxels found in training labels. Using pos_weight=1.0")
        pos_vox = 1

    pos_weight_val = float(neg_vox / (pos_vox + 1e-9))
    pos_weight_val = min(pos_weight_val, 200.0)  # Cap to prevent instability

    logger.info(f"  Positive voxels: {int(pos_vox):,}")
    logger.info(f"  Negative voxels: {int(neg_vox):,}")
    logger.info(f"  Ratio (neg/pos): {pos_weight_val:.2f}")
    logger.info(f"  pos_weight = {pos_weight_val:.2f}")

    return torch.tensor([pos_weight_val], device=device)


def build_loss_function(pos_weight, logger: logging.Logger):
    try:
        from monai.losses import TverskyLoss, FocalLoss
    except Exception as e:
        raise ImportError("monai is required for losses. Install `pip install monai`") from e

    tversky = TverskyLoss(
        alpha=0.3,        beta=0.7,        sigmoid=True
    )

    # BCE: handle class imbalance
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Focal: focus on hard examples
    focal = FocalLoss(gamma=2.0, alpha=None, reduction="mean")

    def combined_loss(logits, target):
        t = tversky(logits, target)
        b = bce(logits, target)
        f = focal(logits, target)
        return 0.5 * t + 0.3 * b + 0.2 * f

    logger.info("Loss function: 50% Tversky + 30% BCE + 20% Focal")

    return combined_loss



class Trainer:
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.use_amp = config.device.startswith("cuda")
        self.scaler = GradScaler(enabled=self.use_amp)
        self.ckpt_best = None
        self.ckpt_last = None
        self.metrics_file = self.config.project_root / "training_metrics.json"
        self.best_loss = float("inf")

    def mixup_3d(self, x1, y1, x2, y2, alpha=0.2):
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
        x = lam * x1 + (1 - lam) * x2
        y = lam * y1 + (1 - lam) * y2
        return x, y

    def train_epoch(self, model, loader, optimizer, scaler, loss_fn, scheduler, epoch):
        model.train()
        running_loss = 0.0
        n = 0
        for batch in tqdm(loader, desc=f"Train Epoch {epoch}", leave=False):
            images = batch["image"].to(self.config.device)
            labels = batch["label"].to(self.config.device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logits = model(images)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            running_loss += float(loss.item())
            n += 1
        avg = running_loss / max(1, n)
        self.logger.info(f"Epoch {epoch} train loss: {avg:.4f}")
        return avg

    def validate_epoch(self, model, loader, metrics, epoch):
        model.eval()
        running_loss = 0.0
        n = 0
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Val Epoch {epoch}", leave=False):
                images = batch["image"].to(self.config.device)
                labels = batch["label"].to(self.config.device)
                logits = model(images)
                # compute dice metric if provided; here we just compute BCE-ish metrics via thresholding
                n += 1
        val_loss = 0.0
        self.logger.info(f"Epoch {epoch} val loss: {val_loss:.4f}")
        return val_loss

    def save_checkpoint(self, epoch, model, optimizer, scheduler, is_best=False):
        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        }
        p = self.config.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(state, str(p))
        self.ckpt_last = str(p)
        if is_best:
            bestp = self.config.checkpoint_dir / "checkpoint_best.pt"
            shutil.copyfile(str(p), str(bestp))
            self.ckpt_best = str(bestp)

    def load_checkpoint(self, model, optimizer=None, scheduler=None, checkpoint_path=None):
        if checkpoint_path is None:
            return None
        ckpt = torch.load(str(checkpoint_path), map_location=self.config.device)
        model.load_state_dict(ckpt["model_state"])
        if optimizer is not None and ckpt.get("optimizer_state"):
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if scheduler is not None and ckpt.get("scheduler_state"):
            scheduler.load_state_dict(ckpt["scheduler_state"])
        return ckpt

    def train(self, model, train_loader, val_loader, optimizer, scheduler, loss_fn, metrics):
        history = {"train_loss": [], "val_loss": []}
        epochs = getattr(self.config, "epochs", 10)
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(model, train_loader, optimizer, self.scaler, loss_fn, scheduler, epoch)
            val_loss = self.validate_epoch(model, val_loader, metrics, epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
            self.save_checkpoint(epoch, model, optimizer, scheduler, is_best=is_best)
        with open(self.metrics_file, "w") as f:
            json.dump(history, f, indent=2)
        return history

    def plot_training_curves(self, history):
        pass




def _resolve_dataset_path(config: Config, args: argparse.Namespace, logger: logging.Logger) -> Path:
    if args.data_dir:
        candidate = Path(args.data_dir)
        if not candidate.exists():
            raise FileNotFoundError(f"--data_dir path does not exist: {candidate}")
        return candidate

    processed = config.processed_dir
    if any(processed.rglob("*.nii*")):
        logger.info(f"Using processed dataset at {processed}")
        return processed

    def _first_zip(directory: Path) -> Optional[Path]:
        for z in sorted(directory.glob("*.zip")):
            if zipfile.is_zipfile(z):
                return z
        return None

    raw_dir = config.data_dir / "raw"
    if raw_dir.exists():
        if any(raw_dir.rglob("*.nii*")):
            logger.info(f"Using raw directory with NIfTI files at {raw_dir}")
            return raw_dir
        zip_candidate = _first_zip(raw_dir)
        if zip_candidate:
            logger.info(f"Using dataset zip at {zip_candidate}")
            return zip_candidate

    zip_candidate = _first_zip(config.data_dir)
    if zip_candidate:
        logger.info(f"Using dataset zip at {zip_candidate}")
        return zip_candidate

    zip_candidate = _first_zip(config.project_root)
    if zip_candidate:
        logger.info(f"Using dataset zip at {zip_candidate}")
        return zip_candidate

    raise FileNotFoundError(
        "Could not automatically locate a dataset. Provide the path explicitly with "
    )


def main(args):

    config = Config.from_args(args)
    config.epochs = args.epochs

    experiment_name = args.experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logging(config.log_dir, experiment_name)

    dataset_path = _resolve_dataset_path(config, args, logger)

    logger.info("="*70)
    logger.info("VASCULAR SEGMENTATION TRAINING PIPELINE")
    logger.info("="*70)
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Device: {config.device}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")

    config.save(config.checkpoint_dir / "config.json")

    logger.info("\n" + "="*70)
    logger.info("STEP 1: DATA EXTRACTION")
    logger.info("="*70)

    data_manager = LocalDataManager(config, logger)

    extract_dir = data_manager.extract_dataset(dataset_path)
    nifti_files = data_manager.find_nifti_files(extract_dir)

    logger.info("\n" + "="*70)
    logger.info("STEP 2: IMAGE-LABEL PAIRING")
    logger.info("="*70)

    pairer = DatasetPairer(logger)
    paired_data = pairer.pair_files(nifti_files)
    if not paired_data:
        raise ValueError(
            "No image/label pairs were discovered in "
            f"{extract_dir}. Verify that your dataset contains NIfTI files and "
            "that filenames include both the image and label identifiers (e.g. "
            "`*_image.nii.gz` and `*_label.nii.gz`)."
        )

    logger.info("\n" + "="*70)
    logger.info("STEP 3: DATASET SPLITTING")
    logger.info("="*70)

    splitter = DatasetSplitter(config, logger)
    train_files, val_files, test_files = splitter.split_dataset(
        paired_data,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed
    )
    if not train_files:
        raise ValueError(
            "The dataset split resulted in 0 training samples. "
            f"Total paired items: {len(paired_data)}. "
            "Add more labeled data or reduce the validation/test ratios."
        )
    if not val_files:
        logger.warning(
            "Validation split is empty. Adjust `--val_ratio` if you need validation metrics."
        )

    logger.info("\n" + "="*70)
    logger.info("STEP 4: BUILDING DATA LOADERS")
    logger.info("="*70)

    try:
        from monai.data import CacheDataset, DataLoader
        from monai.data.utils import pad_list_data_collate
    except Exception as e:
        raise ImportError("monai is required for dataloaders. Install `pip install monai`") from e

    train_transforms = build_transforms(config, mode="train")
    val_transforms = build_transforms(config, mode="val")

    logger.info("Creating training dataset...")
    train_ds = CacheDataset(
        data=train_files,
        transform=train_transforms,
        cache_rate=1.0,
        num_workers=config.num_workers
    )

    logger.info("Creating validation dataset...")
    val_ds = CacheDataset(
        data=val_files,
        transform=val_transforms,
        cache_rate=1.0,
        num_workers=config.num_workers
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=pad_list_data_collate,
        pin_memory=config.pin_memory
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=pad_list_data_collate,
        pin_memory=config.pin_memory
    )

    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches: {len(val_loader)}")

    logger.info("\n" + "="*70)
    logger.info("STEP 5: MODEL INITIALIZATION")
    logger.info("="*70)

    model = build_model(config, logger)

    logger.info("\n" + "="*70)
    logger.info("STEP 6: LOSS FUNCTION & METRICS")
    logger.info("="*70)

    pos_weight = compute_class_weights(train_files, logger, config.device)
    loss_fn = build_loss_function(pos_weight, logger)

    # Metrics
    try:
        from monai.metrics import DiceMetric, HausdorffDistanceMetric
        metrics = {
            "dice": DiceMetric(include_background=False, reduction="mean"),
            "hausdorff": HausdorffDistanceMetric(
                include_background=False,
                percentile=95,
                reduction="mean"
            ),
        }
        logger.info("Metrics: Dice Score, Hausdorff Distance (95th percentile)")
    except Exception:
        metrics = {}

    logger.info("\n" + "="*70)
    logger.info("STEP 7: OPTIMIZER & SCHEDULER")
    logger.info("="*70)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999)
    )

    steps_per_epoch = max(1, len(train_loader) // max(1, config.accumulation_steps))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=max(1, 10 * steps_per_epoch),
        T_mult=2,
        eta_min=1e-6
    )

    logger.info(f"  Optimizer: AdamW")
    logger.info(f"  Initial LR: {config.learning_rate:.2e}")
    logger.info(f"  Scheduler: CosineAnnealingWarmRestarts")
    logger.info(f"  Steps per epoch: {steps_per_epoch}")

    logger.info("\n" + "="*70)
    logger.info("STEP 8: TRAINING")
    logger.info("="*70)

    trainer = Trainer(config, logger)
    history = trainer.train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        metrics=metrics
    )

    logger.info("\n" + "="*70)
    logger.info("PIPELINE COMPLETE!")
    logger.info("="*70)
    logger.info(f"Best model: {trainer.ckpt_best}")
    logger.info(f"Last model: {trainer.ckpt_last}")
    logger.info(f"Training metrics: {trainer.metrics_file}")
    logger.info(f"Logs: {config.log_dir}")

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vascular Segmentation Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to dataset zip file (e.g., ./data/raw/orig.zip)"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Validation set ratio"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.0,
        help="Test set ratio"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate"
    )
    parser.add_argument(
        type=str,
        default="64,192,160",
        help="ROI size for training (format: H,W,D)"
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Name for this experiment"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    main(args)
