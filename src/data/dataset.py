

import random
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_aug() -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomCrop(224, 224),
        # Основная магия CNNDetect — blur и JPEG мешают учить артефакты домена
        A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.0, 3.0), p=0.5),
        A.ImageCompression(quality_lower=30, quality_upper=100, p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_aug() -> A.Compose:
    return A.Compose([
        A.CenterCrop(224, 224),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class DeepfakeDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        val_split: float = 0.2,
        seed: int = 42,
    ):
        assert split in ("train", "val")

        self.data_dir = Path(data_dir)
        self.split    = split
        self.aug      = get_train_aug() if split == "train" else get_val_aug()

        self.samples = []

        for path in sorted((self.data_dir / "real").glob("*.png")):
            self.samples.append((path, 0))

        for path in sorted((self.data_dir / "fake").glob("*.png")):
            self.samples.append((path, 1))

        if not self.samples:
            raise RuntimeError(f"Изображения не найдены в {self.data_dir}.")

        random.seed(seed)
        random.shuffle(self.samples)

        n_val = int(len(self.samples) * val_split)
        self.samples = self.samples[:n_val] if split == "val" else self.samples[n_val:]

        labels = [s[1] for s in self.samples]
        print(f"[{split.upper()}] Загружено: {len(self.samples)} "
              f"(real: {labels.count(0)}, fake: {labels.count(1)})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]

        img = cv2.imread(str(path))
        if img is None:
            img = np.zeros((256, 256, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ресайз до 256 перед crop — как в оригинальном CNNDetect
        img = cv2.resize(img, (256, 256))

        return self.aug(image=img)["image"], label


def get_dataloaders(
    data_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 0,
    val_split: float = 0.2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = DeepfakeDataset(data_dir, "train", val_split, seed)
    val_ds   = DeepfakeDataset(data_dir, "val",   val_split, seed)

    labels         = [s[1] for s in train_ds.samples]
    n_real, n_fake = labels.count(0), labels.count(1)
    total          = len(labels)
    w_real         = total / (2 * n_real) if n_real else 1.0
    w_fake         = total / (2 * n_fake) if n_fake else 1.0
    weights        = [w_real if l == 0 else w_fake for l in labels]

    sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader
