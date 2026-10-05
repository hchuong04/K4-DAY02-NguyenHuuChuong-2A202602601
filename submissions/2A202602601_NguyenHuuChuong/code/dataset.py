"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Quy tắc chia dữ liệu bắt buộc (S1-S6) nằm ở README.md, mục 2.1.
Giao diện giữ nguyên:
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as T

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
TOTAL_EXPECTED_IMAGES = 17509
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(labels_dir: str | Path, fold: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1).

    Mỗi file có cột `Filename, Label, Species`. Trả về ba DataFrame.
    KHÔNG sửa, lọc hay chia lại dữ liệu.
    """
    labels_dir = Path(labels_dir)
    train_path = labels_dir / f"train_subset{fold}.csv"
    val_path = labels_dir / f"val_subset{fold}.csv"
    test_path = labels_dir / f"test_subset{fold}.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {val_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {test_path}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        required_cols = {"Filename", "Label"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"{name}_df thiếu các cột bắt buộc: {required_cols - set(df.columns)}")

    return train_df, val_df, test_df


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path | None = None) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1). In ra và trả về dict số liệu.

    Kiểm tra:
      1. Số ảnh mỗi tập và số ảnh mỗi lớp trong từng tập (kỳ vọng xấp xỉ 60/20/20)
      2. Giao của từng cặp tập theo Filename phải RỖNG (train∩val, train∩test, val∩test)
      3. Hợp ba tập phải bằng đúng 17.509 ảnh
      4. Mọi Filename đều tồn tại trong `images_dir` (nếu images_dir được truyền và tồn tại)
    """
    n_train = len(train_df)
    n_val = len(val_df)
    n_test = len(test_df)
    n_total = n_train + n_val + n_test

    # 1. Kiểm tra tập hợp tên file
    train_files = set(train_df["Filename"])
    val_files = set(val_df["Filename"])
    test_files = set(test_df["Filename"])

    train_val_overlap = train_files & val_files
    train_test_overlap = train_files & test_files
    val_test_overlap = val_files & test_files

    assert len(train_val_overlap) == 0, f"LỖI S1-S6: train ∩ val có {len(train_val_overlap)} ảnh trùng!"
    assert len(train_test_overlap) == 0, f"LỖI S1-S6: train ∩ test có {len(train_test_overlap)} ảnh trùng!"
    assert len(val_test_overlap) == 0, f"LỖI S1-S6: val ∩ test có {len(val_test_overlap)} ảnh trùng!"

    union_files = train_files | val_files | test_files
    assert len(union_files) == TOTAL_EXPECTED_IMAGES, (
        f"LỖI S1-S6: Hợp ba tập có {len(union_files)} ảnh, kỳ vọng đúng {TOTAL_EXPECTED_IMAGES}!"
    )
    assert n_total == TOTAL_EXPECTED_IMAGES, (
        f"LỖI S1-S6: Tổng số ảnh là {n_total}, kỳ vọng đúng {TOTAL_EXPECTED_IMAGES}!"
    )

    # Đếm số lượng theo lớp
    train_counts = train_df["Label"].value_counts().sort_index().to_dict()
    val_counts = val_df["Label"].value_counts().sort_index().to_dict()
    test_counts = test_df["Label"].value_counts().sort_index().to_dict()

    # 4. Kiểm tra tồn tại file ảnh trên đĩa nếu images_dir được cung cấp
    missing_files = []
    if images_dir is not None:
        img_p = Path(images_dir)
        if img_p.exists():
            for f in union_files:
                if not (img_p / f).exists():
                    missing_files.append(f)
                    if len(missing_files) > 10:
                        break
            assert len(missing_files) == 0, f"LỖI: Có ảnh trong CSV không tìm thấy trên đĩa: {missing_files[:5]}..."

    stats = {
        "n": {"train": n_train, "val": n_val, "test": n_test, "total": n_total},
        "ratio": {
            "train": n_train / n_total,
            "val": n_val / n_total,
            "test": n_test / n_total,
        },
        "per_class": {
            "train": train_counts,
            "val": val_counts,
            "test": test_counts,
        },
        "overlap": {
            "train_val": len(train_val_overlap),
            "train_test": len(train_test_overlap),
            "val_test": len(val_test_overlap),
        },
        "all_files_exist": len(missing_files) == 0,
    }
    return stats


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    """Tạo torchvision transform.

    Train:
      - basic: RandomResizedCrop(img_size) + RandomHorizontalFlip() + ToTensor() + Normalize()
      - color: basic + ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
      - randaug: RandomResizedCrop(img_size) + RandomHorizontalFlip() + RandAugment() + ToTensor() + Normalize()
      - trivial: RandomResizedCrop(img_size) + RandomHorizontalFlip() + TrivialAugmentWide() + ToTensor() + Normalize()
    Val/test:
      - Resize 256 -> CenterCrop(img_size) (hoặc Resize(img_size) nếu img_size == 256) + ToTensor() + Normalize().
      KHÔNG dùng augmentation ngẫu nhiên khi val/test.
    """
    normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if train:
        crop_transform = T.RandomResizedCrop(img_size, scale=(0.8, 1.0))
        hflip_transform = T.RandomHorizontalFlip(p=0.5)

        if aug == "basic":
            return T.Compose([
                crop_transform,
                hflip_transform,
                T.ToTensor(),
                normalize,
            ])
        elif aug == "color":
            return T.Compose([
                crop_transform,
                hflip_transform,
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                T.ToTensor(),
                normalize,
            ])
        elif aug == "randaug":
            return T.Compose([
                crop_transform,
                hflip_transform,
                T.RandAugment(num_ops=2, magnitude=9),
                T.ToTensor(),
                normalize,
            ])
        elif aug == "trivial":
            return T.Compose([
                crop_transform,
                hflip_transform,
                T.TrivialAugmentWide(),
                T.ToTensor(),
                normalize,
            ])
        elif aug == "hvflip":
            # Thử nghiệm lật cả 2 chiều (ảnh từ trên xuống trong nông nghiệp)
            return T.Compose([
                crop_transform,
                hflip_transform,
                T.RandomVerticalFlip(p=0.5),
                T.ToTensor(),
                normalize,
            ])
        else:
            raise ValueError(f"Không hỗ trợ augmentation: {aug}")
    else:
        if img_size == 256:
            resize_transform = T.Resize((256, 256))
            crop_transform = None
        else:
            resize_transform = T.Resize(256)
            crop_transform = T.CenterCrop(img_size)

        transforms_list = [resize_transform]
        if crop_transform is not None:
            transforms_list.append(crop_transform)
        transforms_list.extend([T.ToTensor(), normalize])

        return T.Compose(transforms_list)


class DeepWeedsDataset(Dataset):
    """Dataset đọc ảnh từ `images_dir` theo DataFrame (Filename, Label).

    __getitem__(i) trả về (image_tensor, label: int, filename: str).
    """

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.filenames = self.df["Filename"].tolist()
        self.labels = self.df["Label"].to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int, str]:
        fname = self.filenames[i]
        path = self.images_dir / fname
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                if self.transform is not None:
                    img = self.transform(img)
                return img, int(self.labels[i]), fname
        except Exception as e:
            raise IOError(f"Lỗi đọc ảnh {path}: {e}")


def seed_worker(worker_id):
    """Cố định seed cho worker của DataLoader nhằm đảm bảo tính tái lập."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2) -> DataLoader:
    """Tạo DataLoader theo cấu hình:

    - train=True: shuffle=True (nếu không có sampler).
    - sampler="balanced": dùng WeightedRandomSampler với trọng số 1 / N_c.
    - train=False: không shuffle, giữ nguyên thứ tự df.
    """
    dataset = DeepWeedsDataset(df, images_dir=images_dir, transform=transform)

    if train:
        if sampler == "balanced":
            class_counts = df["Label"].value_counts().to_dict()
            weights = [1.0 / class_counts[label] for label in df["Label"]]
            weights = torch.as_tensor(weights, dtype=torch.double)
            random_sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            return DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=random_sampler,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                drop_last=len(dataset) > batch_size,
                worker_init_fn=seed_worker,
            )
        else:
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                drop_last=len(dataset) > batch_size,
                worker_init_fn=seed_worker,
            )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
