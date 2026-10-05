"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

Liên hệ slide Day 2:
  - TTA (trang 62-66, 75)
  - ensemble/EMA/soup (trang 67)
  - độ phân giải kiểm tra (trang 68)
  - temperature scaling (trang 69)
  - gộp BatchNorm (trang 71)

Giao diện giữ nguyên:
    predict_logits(model, loader, device, view=None) -> (filenames, y_true, logits[N, 9])
    aggregate_views(list_of_logits, space)           -> probs[N, 9]
    fit_temperature(val_logits, val_labels)          -> float T
    apply_temperature(logits, T)                     -> probs
    ensemble_probs(list_of_probs)                    -> probs
    fuse_conv_bn(model)                              -> model (BN đã gộp vào conv)
"""
from __future__ import annotations

import copy
from typing import Callable, List, Tuple

import numpy as np
from scipy.optimize import minimize_scalar
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 9


def softmax_np(z: np.ndarray) -> np.ndarray:
    """Softmax an toàn số học trên numpy."""
    z_max = np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z - z_max)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


def predict_logits(model: nn.Module, loader, device: str | torch.device,
                   view: Callable[[torch.Tensor], torch.Tensor] | None = None
                   ) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Chạy model trên loader ở chế độ eval, không tính gradient.

    `view` là hàm biến đổi batch ảnh trước khi đưa vào model (ví dụ lật ngang), hoặc None.
    Trả về (filenames, y_true, logits[N, 9]).
    """
    model.eval()
    all_filenames = []
    all_y = []
    all_logits = []

    with torch.inference_mode():
        for batch in loader:
            images, targets, filenames = batch[0], batch[1], batch[2]
            images = images.to(device, non_blocking=True)
            if view is not None:
                images = view(images)

            outputs = model(images)
            all_filenames.extend(filenames)
            all_y.append(targets.cpu().numpy())
            all_logits.append(outputs.cpu().numpy())

    y_true = np.concatenate(all_y, axis=0) if all_y else np.array([])
    logits = np.concatenate(all_logits, axis=0) if all_logits else np.empty((0, NUM_CLASSES))
    return all_filenames, y_true, logits


def view_identity(x: torch.Tensor) -> torch.Tensor:
    """Giữ nguyên ảnh."""
    return x


def view_hflip(x: torch.Tensor) -> torch.Tensor:
    """Lật ngang batch (N, C, H, W) (slide trang 75)."""
    return torch.flip(x, dims=[-1])


def views_multicrop(x: torch.Tensor, crop: int = 224) -> List[torch.Tensor]:
    """5 crop (4 góc + chính giữa) kích thước `crop` x `crop`."""
    _, _, h, w = x.shape
    if h < crop or w < crop:
        # Nếu ảnh nhỏ hơn crop thì resize trước
        x = F.interpolate(x, size=(max(h, crop), max(w, crop)), mode="bilinear", align_corners=False)
        _, _, h, w = x.shape

    top_left = x[:, :, :crop, :crop]
    top_right = x[:, :, :crop, w - crop:]
    bottom_left = x[:, :, h - crop:, :crop]
    bottom_right = x[:, :, h - crop:, w - crop:]

    cy = (h - crop) // 2
    cx = (w - crop) // 2
    center = x[:, :, cy:cy + crop, cx:cx + crop]

    return [center, top_left, top_right, bottom_left, bottom_right]


def views_multiscale(x: torch.Tensor, sizes: List[int]) -> List[torch.Tensor]:
    """Resize batch về từng kích thước trong `sizes`."""
    return [F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False) for s in sizes]


def aggregate_views(logits_per_view: List[np.ndarray], space: str = "prob") -> np.ndarray:
    """Gộp K lượt chạy của TTA thành một dự đoán (slide trang 62).

    - space="prob":  trung bình softmax của từng view
    - space="logit": trung bình logit rồi softmax
    """
    if not logits_per_view:
        raise ValueError("Danh sách logits rỗng!")

    if space == "prob":
        probs = [softmax_np(z) for z in logits_per_view]
        mean_prob = np.mean(probs, axis=0)
        # Chuẩn hoá lại tổng bằng 1 tránh sai số số học
        return mean_prob / np.sum(mean_prob, axis=-1, keepdims=True)

    elif space == "logit":
        mean_logits = np.mean(logits_per_view, axis=0)
        return softmax_np(mean_logits)

    else:
        raise ValueError(f"Không hỗ trợ không gian gộp: {space}. Chọn 'prob' hoặc 'logit'.")


def ensemble_probs(list_of_probs: List[np.ndarray]) -> np.ndarray:
    """Trung bình xác suất của nhiều mô hình (khác backbone hoặc khác seed).

    Chi phí suy luận = tổng số mô hình.
    """
    if not list_of_probs:
        raise ValueError("Danh sách xác suất rỗng!")
    mean_prob = np.mean(list_of_probs, axis=0)
    return mean_prob / np.sum(mean_prob, axis=-1, keepdims=True)


def fit_temperature(val_logits: np.ndarray | torch.Tensor, val_labels: np.ndarray | torch.Tensor) -> float:
    """Tìm nhiệt độ T > 0 cực tiểu Cross-Entropy (Negative Log Likelihood) trên tập VAL:

    p = softmax(logit / T)  (Guo et al. 2017, slide trang 69).
    Chỉ khớp T trên VAL, TUYỆT ĐỐI không dùng TEST.
    """
    if isinstance(val_logits, torch.Tensor):
        val_logits = val_logits.cpu().numpy()
    if isinstance(val_labels, torch.Tensor):
        val_labels = val_labels.cpu().numpy()

    y_one_hot = np.zeros_like(val_logits)
    for i, label in enumerate(val_labels):
        y_one_hot[i, int(label)] = 1.0

    def nll_obj(T_val: float) -> float:
        scaled_logits = val_logits / max(T_val, 1e-4)
        z_max = np.max(scaled_logits, axis=-1, keepdims=True)
        log_sum_exp = z_max + np.log(np.sum(np.exp(scaled_logits - z_max), axis=-1, keepdims=True))
        log_probs = scaled_logits - log_sum_exp
        nll = -np.sum(y_one_hot * log_probs) / len(val_labels)
        return float(nll)

    res = minimize_scalar(nll_obj, bounds=(0.05, 10.0), method="bounded")
    optimal_T = round(float(res.x), 4)
    return max(optimal_T, 0.05)


def apply_temperature(logits: np.ndarray | torch.Tensor, T: float) -> np.ndarray:
    """Trả về softmax(logits / T)."""
    if isinstance(logits, torch.Tensor):
        logits = logits.cpu().numpy()
    T = max(float(T), 1e-4)
    return softmax_np(logits / T)


def fuse_conv_bn(model: nn.Module) -> nn.Module:
    """Gộp BatchNorm vào tích chập liền trước (slide trang 71, 75).

    w' = gamma * w / sqrt(var + eps)
    b' = beta + gamma * (b - mean) / sqrt(var + eps)
    """
    model_copy = copy.deepcopy(model).eval()

    try:
        from torch.nn.utils.fusion import fuse_conv_bn_eval
    except ImportError:
        fuse_conv_bn_eval = None

    # Hàm tự gộp thủ công theo công thức slide
    def fuse_modules(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
        with torch.no_grad():
            gamma = bn.weight
            beta = bn.bias
            mean = bn.running_mean
            var = bn.running_var
            eps = bn.eps

            std = torch.sqrt(var + eps)
            w = conv.weight
            b = conv.bias if conv.bias is not None else torch.zeros(conv.out_channels, device=w.device)

            scale = gamma / std
            w_fused = w * scale.view(-1, 1, 1, 1)
            b_fused = beta + scale * (b - mean)

            fused_conv = nn.Conv2d(
                conv.in_channels,
                conv.out_channels,
                conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                dilation=conv.dilation,
                groups=conv.groups,
                bias=True,
                padding_mode=conv.padding_mode,
            ).to(w.device)

            fused_conv.weight.copy_(w_fused)
            fused_conv.bias.copy_(b_fused)
            return fused_conv

    # Duyệt qua các submodule và thay thế conv+bn
    fused_count = 0
    for name, module in list(model_copy.named_children()):
        if isinstance(module, nn.Sequential):
            i = 0
            while i < len(module) - 1:
                sub1 = module[i]
                sub2 = module[i + 1]
                if isinstance(sub1, nn.Conv2d) and isinstance(sub2, nn.BatchNorm2d):
                    if fuse_conv_bn_eval is not None:
                        module[i] = fuse_conv_bn_eval(sub1, sub2)
                    else:
                        module[i] = fuse_modules(sub1, sub2)
                    module[i + 1] = nn.Identity()
                    fused_count += 1
                    i += 2
                else:
                    i += 1
        elif len(list(module.children())) > 0:
            # Đệ quy xuống các tầng con
            setattr(model_copy, name, fuse_conv_bn(module))

    model_copy._is_fused_bn = True
    return model_copy
