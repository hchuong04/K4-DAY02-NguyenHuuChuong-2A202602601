"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

Liên hệ slide Day 2:
  - label smoothing (trang 56)
  - focal loss (trang 57)
  - Mixup/CutMix (trang 48)

Giao diện giữ nguyên:
    build_criterion(kind, **kw)                 -> callable(logits, target) -> loss scalar
    class_weights(counts, beta)                 -> tensor trọng số lớp
    mix_batch(x, y, alpha, mode)                -> (x_mixed, (y_a, y_b, lam))
    mixed_loss(criterion, logits, targets)      -> loss scalar
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 9


def build_criterion(kind: str = "ce", **kw):
    """Trả về hàm loss theo `kind`: "ce", "ls", "focal", "ce_weighted"."""
    if kind == "ce":
        return nn.CrossEntropyLoss()
    elif kind == "ls":
        smoothing = kw.get("smoothing", kw.get("label_smoothing", 0.1))
        return LabelSmoothingCE(smoothing=smoothing)
    elif kind == "focal":
        gamma = kw.get("gamma", kw.get("focal_gamma", 2.0))
        alpha = kw.get("alpha", None)
        return FocalLoss(gamma=gamma, alpha=alpha)
    elif kind == "ce_weighted":
        weight = kw.get("weight", None)
        return nn.CrossEntropyLoss(weight=weight)
    else:
        raise ValueError(f"Loại criterion không hỗ trợ: {kind}")


class LabelSmoothingCE(nn.Module):
    """Cross-entropy với label smoothing:
    q'(k) = (1 - eps) * 1[k == y] + eps / K  (slide trang 56).
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = float(smoothing)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=self.smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits, targets)


class FocalLoss(nn.Module):
    """Focal loss nhiều lớp (Lin et al., slide trang 57):
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t).

    Khi gamma = 0 và alpha = None: tương đương chính xác CrossEntropyLoss thông thường.
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | list[float] | None = None,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.as_tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (N, C), targets: (N,)
        log_p = F.log_softmax(logits, dim=-1)
        p = torch.exp(log_p)

        # Lấy log p_t và p_t tương ứng nhãn thật
        log_pt = log_p.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        pt = p.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

        modulating_factor = (1.0 - pt) ** self.gamma
        loss = -modulating_factor * log_pt

        if self.alpha is not None:
            at = self.alpha.to(logits.device).gather(dim=-1, index=targets)
            loss = at * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def class_weights(counts: dict | list | np.ndarray, beta: float = 0.0) -> torch.Tensor:
    """Tính trọng số theo lớp từ số ảnh mỗi lớp trong tập TRAIN:

    - beta = 0: trọng số nghịch đảo tần suất (1 / n_c), chuẩn hoá trung bình bằng 1
    - beta > 0: class-balanced theo số mẫu hiệu dụng (Cui et al., slide trang 57)
                w_c = (1 - beta) / (1 - beta^n_c), chuẩn hoá tổng bằng NUM_CLASSES
    """
    if isinstance(counts, dict):
        n_c = np.array([counts.get(i, 1) for i in range(NUM_CLASSES)], dtype=np.float32)
    else:
        n_c = np.array(counts, dtype=np.float32)

    n_c = np.maximum(n_c, 1.0)  # tránh chia 0

    if beta <= 0.0:
        raw_weights = 1.0 / n_c
        weights = raw_weights / raw_weights.mean()
    else:
        # Effective number of samples: E_n = (1 - beta^n) / (1 - beta)
        effective_num = (1.0 - np.power(beta, n_c)) / (1.0 - beta)
        raw_weights = 1.0 / effective_num
        weights = (raw_weights / raw_weights.sum()) * NUM_CLASSES

    return torch.tensor(weights, dtype=torch.float32)


def mix_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0, mode: str = "cutmix"):
    """Trộn một batch ảnh và nhãn.

    - mode="mixup": x_mix = lam * x + (1 - lam) * x[perm]
    - mode="cutmix": cắt một hộp chữ nhật từ x[perm] dán vào x, điều chỉnh lam theo diện tích thực tế.
    Trả về (x_mixed, (y_a, y_b, lam)).
    """
    if alpha <= 0.0:
        return x, (y, y, 1.0)

    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    perm = torch.randperm(batch_size, device=x.device)

    y_a = y
    y_b = y[perm]

    if mode == "mixup":
        x_mixed = lam * x + (1.0 - lam) * x[perm]
        return x_mixed, (y_a, y_b, lam)

    elif mode == "cutmix":
        _, _, h, w = x.shape
        # Tính kích thước hộp theo diện tích (1 - lam)
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)

        # Toạ độ tâm ngẫu nhiên
        cx = np.random.randint(w)
        cy = np.random.randint(h)

        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        x_mixed = x.clone()
        x_mixed[:, :, bby1:bby2, bbx1:bbx2] = x[perm, :, bby1:bby2, bbx1:bbx2]

        # Điều chỉnh lại lam theo diện tích hộp thực tế
        actual_lam = 1.0 - float((bbx2 - bbx1) * (bby2 - bby1)) / float(w * h)
        return x_mixed, (y_a, y_b, actual_lam)

    else:
        raise ValueError(f"Không hỗ trợ chế độ trộn: {mode}")


def mixed_loss(criterion, logits: torch.Tensor, targets: tuple[torch.Tensor, torch.Tensor, float]) -> torch.Tensor:
    """Loss kết hợp cho batch đã trộn:
    lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b).
    """
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
