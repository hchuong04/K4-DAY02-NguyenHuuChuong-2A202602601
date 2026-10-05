"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

Dùng MỘT hàm `run(cfg)` cho mọi cấu hình (RUBRIC mục H):
đổi thí nghiệm chỉ bằng cách đổi `Config`.

Chạy một thí nghiệm từ dòng lệnh:
    python train.py --set exp_id=B01 backbone=resnet50 seed=0
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, fields
import json
import math
import os
from pathlib import Path
import random
import sys

# Cấu hình UTF-8 cho console Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Tự động tìm đường dẫn chứa eval.py và các module
CURRENT_DIR = Path(__file__).resolve().parent
candidate_paths = [
    CURRENT_DIR,
    CURRENT_DIR.parent,
    CURRENT_DIR.parent.parent,
    CURRENT_DIR.parent.parent.parent,
    Path.cwd(),
]
for p in candidate_paths:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

import eval as ev  # noqa: E402
import dataset  # noqa: E402
import model as model_lib  # noqa: E402
import losses  # noqa: E402


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug ...
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"             # config.json, history.csv, checkpoint, logit của từng lần chạy
    pred_dir: str = "predictions"     # file dự đoán đúng định dạng eval.py (nộp cùng bài)
    curves_dir: str = "curves"        # ảnh biểu đồ training
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    """Cố định mọi nguồn ngẫu nhiên."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(model: nn.Module, cfg: Config):
    """AdamW với 3 nhóm tham số (xem model_lib.param_groups)."""
    groups = model_lib.param_groups(
        model,
        lr_backbone=cfg.lr_backbone,
        lr_head=cfg.lr_head,
        weight_decay=cfg.weight_decay,
    )
    return torch.optim.AdamW(groups)


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính rồi cosine về ~0 (slide trang 55)."""
    total_steps = max(1, cfg.epochs * steps_per_epoch)
    warmup_steps = int(cfg.warmup_epochs * steps_per_epoch)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EMA:
    """Trung bình động trọng số: W_ema <- d * W_ema + (1 - d) * W  (slide trang 56)."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.backup = {}

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def apply_shadow(self, model: nn.Module) -> None:
        """Lưu trọng số hiện tại vào backup và gán trọng số EMA vào model để eval."""
        with torch.no_grad():
            self.backup = {}
            for name, param in model.named_parameters():
                if name in self.shadow:
                    self.backup[name] = param.clone().detach()
                    param.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        """Khôi phục lại trọng số huấn luyện ban đầu từ backup."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.backup:
                    param.copy_(self.backup[name])
            self.backup = {}


def train_one_epoch(model: nn.Module, loader, criterion, optimizer, scheduler, scaler,
                    cfg: Config, device: torch.device, ema: EMA | None = None) -> dict:
    """Một epoch huấn luyện."""
    model.train()

    # Nếu backbone bị đóng băng, luôn giữ BatchNorm trong eval mode (GUIDE 3.2)
    if getattr(model, "_backbone_frozen", False):
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
                m.eval()

    total_loss = 0.0
    total_samples = 0
    use_amp = cfg.amp and device.type == "cuda"
    try:
        autocast_ctx = torch.amp.autocast("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp)

    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = images.size(0)

        optimizer.zero_grad(set_to_none=True)

        if cfg.mix in ("mixup", "cutmix"):
            images_mix, mix_targets = losses.mix_batch(
                images, targets, alpha=cfg.mix_alpha, mode=cfg.mix
            )
            with autocast_ctx:
                outputs = model(images_mix)
                loss = losses.mixed_loss(criterion, outputs, mix_targets)
        else:
            with autocast_ctx:
                outputs = model(images)
                loss = criterion(outputs, targets)

        if use_amp:
            scaler.scale(loss).backward()
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()
            # Chỉ cập nhật scheduler nếu optimizer.step() không bị bỏ qua do inf/nan gradient
            if scale_before <= scale_after:
                scheduler.step()
        else:
            loss.backward()
            optimizer.step()
            scheduler.step()

        if ema is not None:
            ema.update(model)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    avg_loss = total_loss / max(1, total_samples)
    current_lr = optimizer.param_groups[0]["lr"]
    return {"train_loss": avg_loss, "lr": current_lr}


def evaluate(model: nn.Module, loader, criterion, device: torch.device):
    """Chạy model trên một loader ở chế độ eval, KHÔNG tính gradient.

    Trả về (filenames, y_true, logits[N, 9], loss).
    """
    model.eval()
    all_filenames = []
    all_y = []
    all_logits = []
    total_loss = 0.0
    total_samples = 0

    with torch.inference_mode():
        for images, targets, filenames in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            batch_size = images.size(0)

            outputs = model(images)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_filenames.extend(filenames)
            all_y.append(targets.cpu().numpy())
            all_logits.append(outputs.cpu().numpy())

    avg_loss = total_loss / max(1, total_samples)
    y_true = np.concatenate(all_y, axis=0) if all_y else np.array([], dtype=int)
    logits = np.concatenate(all_logits, axis=0) if all_logits else np.empty((0, ev.NUM_CLASSES))

    return all_filenames, y_true, logits, avg_loss


def softmax_np(z: np.ndarray) -> np.ndarray:
    """Softmax an toàn trên numpy."""
    z_max = np.max(z, axis=-1, keepdims=True)
    e = np.exp(z - z_max)
    return e / np.sum(e, axis=-1, keepdims=True)


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    """Vẽ đường cong training của một thí nghiệm -> curves/<exp_id>_<mota>.png."""
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_macro_f1 = [h["val_macro_f1"] for h in history]
    val_top1 = [h["val_top1"] for h in history]
    lrs = [h["lr"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # 1. Loss curves
    axes[0].plot(epochs, train_loss, "o-", label="Train Loss", color="tab:blue")
    axes[0].plot(epochs, val_loss, "s--", label="Val Loss", color="tab:orange")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss (Train vs Val)")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    # 2. Metric curves
    axes[1].plot(epochs, val_macro_f1, "o-", label="Val Macro-F1", color="tab:green")
    axes[1].plot(epochs, val_top1, "^--", label="Val Top-1 Acc", color="tab:red")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Validation Metrics")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    # 3. Learning Rate schedule
    axes[2].plot(epochs, lrs, "d-", label="LR", color="tab:purple")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("Learning Rate Schedule")
    axes[2].grid(True, linestyle=":", alpha=0.6)
    axes[2].legend()

    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close(fig)


def run(cfg: Config) -> dict:
    """Huấn luyện một cấu hình và lưu mọi sản phẩm cần thiết.

    Quy tắc S1-S6:
      - Val dùng để chọn checkpoint có Macro-F1 cao nhất.
      - Test CHỈ chạy một lần duy nhất khi cfg.save_test_predictions = True.
    """
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    r_dir = run_dir(cfg)
    r_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg.pred_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.curves_dir).mkdir(parents=True, exist_ok=True)

    # 1. Ghi config.json
    with open(r_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    # 2. Đọc split & kiểm tra tính hợp lệ
    train_df, val_df, test_df = dataset.load_split(cfg.labels_dir, fold=cfg.fold)
    dataset.check_split(train_df, val_df, test_df, cfg.images_dir)

    # 3. Tạo DataLoaders
    train_tf = dataset.build_transforms(train=True, img_size=cfg.img_size, aug=cfg.aug)
    val_tf = dataset.build_transforms(train=False, img_size=cfg.img_size)

    train_loader = dataset.make_loader(
        train_df, cfg.images_dir, train_tf, cfg.batch_size, train=True,
        sampler=cfg.sampler, num_workers=cfg.num_workers
    )
    val_loader = dataset.make_loader(
        val_df, cfg.images_dir, val_tf, cfg.batch_size, train=False,
        num_workers=cfg.num_workers
    )

    # 4. Khởi tạo model, loss, optimizer, scheduler, scaler
    model = model_lib.build_model(
        name=cfg.backbone,
        pretrained=True,
        num_classes=ev.NUM_CLASSES,
        drop_rate=cfg.drop_rate,
        init=cfg.init,
    ).to(device)

    num_params = model_lib.count_params(model)
    gmacs = model_lib.count_gmacs(model, img_size=cfg.img_size)

    # Tính trọng số lớp nếu có
    weight_tensor = None
    if cfg.loss == "ce_weighted" or cfg.class_weight_beta is not None:
        beta_val = cfg.class_weight_beta if cfg.class_weight_beta is not None else 0.0
        train_counts = train_df["Label"].value_counts().to_dict()
        weight_tensor = losses.class_weights(train_counts, beta=beta_val).to(device)

    criterion = losses.build_criterion(
        cfg.loss,
        smoothing=cfg.label_smoothing,
        gamma=cfg.focal_gamma,
        weight=weight_tensor,
    )

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")
    ema = EMA(model, decay=cfg.ema_decay) if cfg.ema_decay is not None else None

    # 5. Vòng lặp huấn luyện các epoch
    history = []
    best_macro_f1 = -1.0
    best_epoch = 1
    best_val_logits = None
    best_val_y = None
    best_val_filenames = None
    epoch_times = []

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_res = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema
        )
        t_epoch = time.time() - t0
        epoch_times.append(t_epoch)

        # Đánh giá trên VAL
        if ema is not None:
            ema.apply_shadow(model)

        val_fns, val_y, val_logits, val_loss = evaluate(model, val_loader, criterion, device)

        if ema is not None:
            ema.restore(model)

        val_probs = softmax_np(val_logits)
        val_ypred = val_probs.argmax(axis=1)
        val_metrics = ev.compute_metrics(val_y, val_ypred, val_probs)
        val_f1 = float(val_metrics["macro_f1"])
        val_top1 = float(val_metrics["top1"])

        history.append({
            "epoch": epoch,
            "train_loss": train_res["train_loss"],
            "val_loss": val_loss,
            "val_macro_f1": val_f1,
            "val_top1": val_top1,
            "lr": train_res["lr"],
            "time_sec": t_epoch,
        })

        # Chọn checkpoint theo MACRO-F1 VAL cao nhất (hòa thì epoch sớm hơn)
        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            best_epoch = epoch
            best_val_logits = val_logits
            best_val_y = val_y
            best_val_filenames = val_fns
            torch.save(model.state_dict(), r_dir / "best_model.pt")

    # 6. Ghi log lịch sử và vẽ biểu đồ training
    history_df = pd.DataFrame(history)
    history_df.to_csv(r_dir / "history.csv", index=False)

    curve_path = Path(cfg.curves_dir) / f"{cfg.exp_id}_{cfg.backbone}.png"
    plot_curves(history, curve_path, f"Exp {cfg.exp_id} - {cfg.backbone} (Seed {cfg.seed})")

    # Lưu val predictions của checkpoint tốt nhất
    if best_val_logits is not None:
        best_val_probs = softmax_np(best_val_logits)
        ev.save_predictions(
            pred_path(cfg, "val"),
            best_val_filenames,
            best_val_y,
            best_val_probs,
        )

    # 7. Đánh giá TEST nếu được yêu cầu (chỉ dùng ở Bước 4)
    test_metrics = None
    if cfg.save_test_predictions:
        # Tải lại trọng số tốt nhất
        model.load_state_dict(torch.load(r_dir / "best_model.pt", map_location=device))
        test_tf = dataset.build_transforms(train=False, img_size=cfg.img_size)
        test_loader = dataset.make_loader(
            test_df, cfg.images_dir, test_tf, cfg.batch_size, train=False,
            num_workers=cfg.num_workers
        )
        test_fns, test_y, test_logits, _ = evaluate(model, test_loader, criterion, device)
        test_probs = softmax_np(test_logits)
        ev.save_predictions(
            pred_path(cfg, "test"),
            test_fns,
            test_y,
            test_probs,
        )
        test_metrics = ev.compute_metrics(test_y, test_probs.argmax(axis=1), test_probs)

    summary = {
        "exp_id": cfg.exp_id,
        "backbone": cfg.backbone,
        "seed": cfg.seed,
        "params_m": num_params,
        "gmacs": gmacs,
        "best_epoch": best_epoch,
        "val_macro_f1": best_macro_f1,
        "avg_train_time_per_epoch_s": round(float(np.mean(epoch_times)), 2),
        "test_metrics": test_metrics,
    }
    return summary


def parse_overrides(pairs: list[str]) -> dict:
    """Biến ['seed=1', 'loss=focal', 'ema_decay=none'] thành dict, ép kiểu theo field của Config."""
    cfg_fields = {f.name: f.type for f in fields(Config)}
    overrides: dict[str, Any] = {}

    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Tham số không hợp lệ (thiếu '='): {pair}")
        key, val = pair.split("=", 1)
        key = key.strip()
        val = val.strip()

        if key not in cfg_fields:
            raise KeyError(f"Trường '{key}' không tồn tại trong Config. Các trường hợp lệ: {list(cfg_fields.keys())}")

        # Ép kiểu giá trị
        if val.lower() == "none":
            overrides[key] = None
        elif val.lower() in ("true", "1"):
            overrides[key] = True
        elif val.lower() in ("false", "0"):
            overrides[key] = False
        else:
            target_type = cfg_fields[key]
            # Thử parse int -> float -> str
            try:
                overrides[key] = int(val)
            except ValueError:
                try:
                    overrides[key] = float(val)
                except ValueError:
                    overrides[key] = val

    return overrides


def main() -> None:
    """Điểm vào dòng lệnh: `python train.py --set exp_id=B01 backbone=resnet50 seed=0`."""
    parser = argparse.ArgumentParser(description="Chạy huấn luyện thí nghiệm DeepWeeds.")
    parser.add_argument("--set", nargs="*", default=[], help="Danh sách ghi đè Config dạng key=val")
    args = parser.parse_args()

    overrides = parse_overrides(args.set)
    cfg = Config(**overrides)
    print(f"Bắt đầu huấn luyện: exp_id={cfg.exp_id}, backbone={cfg.backbone}, seed={cfg.seed}...")
    res = run(cfg)
    print("Huấn luyện hoàn tất! Kết quả tóm tắt:")
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
