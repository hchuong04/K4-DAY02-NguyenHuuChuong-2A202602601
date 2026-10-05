"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

Giao diện giữ nguyên:
    build_model(name, pretrained, num_classes, drop_rate, init) -> nn.Module
    freeze_backbone(model)                                        -> None
    param_groups(model, lr_backbone, lr_head, weight_decay)       -> list[dict] cho optimizer
    count_params(model) -> float (triệu)     count_gmacs(model, img_size) -> float
"""
from __future__ import annotations

import copy
import torch
import torch.nn as nn
import timm

SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",
    "mobilenetv3": "mobilenetv3_large_100",
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune") -> nn.Module:
    """Tạo model phân loại 9 lớp từ timm.

    `init` (trục A của GUIDE.md mục 3):
      - "scratch"  : pretrained=False, huấn luyện toàn bộ
      - "frozen"   : pretrained=True, đóng băng backbone, chỉ train head
      - "finetune" : pretrained=True, train toàn bộ
    """
    if init == "scratch":
        is_pretrained = False
    elif init in ("frozen", "finetune"):
        is_pretrained = pretrained
    else:
        raise ValueError(f"Giá trị init không hợp lệ: {init}. Chọn scratch | frozen | finetune.")

    model = timm.create_model(
        name,
        pretrained=is_pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )

    # Lưu lại thông tin cấu hình và trọng số
    tag = "scratch"
    if is_pretrained:
        cfg = getattr(model, "pretrained_cfg", {}) or getattr(model, "default_cfg", {})
        tag = cfg.get("tag", cfg.get("architecture", "pretrained"))
    model._weight_tag = tag
    model._init_mode = init

    if init == "frozen":
        freeze_backbone(model)

    return model


def freeze_backbone(model: nn.Module) -> None:
    """Đóng băng mọi tham số trừ classifier head.

    Lưu ý (GUIDE.md mục 3.2):
      - Khi backbone đóng băng, BatchNorm trong backbone phải ở chế độ eval lúc train.
      - Đặt cờ `model._backbone_frozen = True` để vòng lặp huấn luyện duy trì eval cho BN.
    """
    classifier = model.get_classifier()
    if isinstance(classifier, nn.Module):
        head_params = set(classifier.parameters())
    elif isinstance(classifier, (torch.Tensor, nn.Parameter)):
        head_params = {classifier}
    else:
        head_params = set()

    for p in model.parameters():
        if p in head_params:
            p.requires_grad = True
        else:
            p.requires_grad = False

    model._backbone_frozen = True


def param_groups(model: nn.Module, lr_backbone: float, lr_head: float, weight_decay: float) -> list[dict]:
    """Chia tham số có requires_grad == True thành 3 nhóm như slide Day 2, trang 52:

    1. Backbone weights (ndim > 1): lr = lr_backbone, weight_decay = weight_decay
    2. Backbone bias & norm (ndim <= 1): lr = lr_backbone, weight_decay = 0.0
    3. Head parameters: lr = lr_head, weight_decay = weight_decay
    """
    classifier = model.get_classifier()
    if isinstance(classifier, nn.Module):
        head_params = set(classifier.parameters())
    elif isinstance(classifier, (torch.Tensor, nn.Parameter)):
        head_params = {classifier}
    else:
        head_params = set()

    backbone_weights = []
    backbone_no_decay = []
    head_group = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param in head_params:
            head_group.append(param)
        else:
            if param.ndim > 1:
                backbone_weights.append(param)
            else:
                backbone_no_decay.append(param)

    groups = []
    if backbone_weights:
        groups.append({
            "params": backbone_weights,
            "lr": lr_backbone,
            "weight_decay": weight_decay,
            "group_name": "backbone_weights",
        })
    if backbone_no_decay:
        groups.append({
            "params": backbone_no_decay,
            "lr": lr_backbone,
            "weight_decay": 0.0,
            "group_name": "backbone_no_decay",
        })
    if head_group:
        groups.append({
            "params": head_group,
            "lr": lr_head,
            "weight_decay": weight_decay,
            "group_name": "head",
        })

    return groups


def count_params(model: nn.Module) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng."""
    total_params = sum(p.numel() for p in model.parameters())
    return round(total_params / 1e6, 3)


def count_gmacs(model: nn.Module, img_size: int = 224) -> float:
    """Đếm GMAC (Giga Multiply-Accumulate Operations) cho 1 ảnh (3 x img_size x img_size)."""
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)

    # Thử đo bằng thop
    try:
        import thop
        model_eval = copy.deepcopy(model).eval()
        macs, _ = thop.profile(model_eval, inputs=(dummy_input,), verbose=False)
        return round(float(macs / 1e9), 3)
    except Exception:
        pass

    # Thử đo bằng fvcore
    try:
        from fvcore.nn import FlopCountAnalysis
        model_eval = copy.deepcopy(model).eval()
        flops = FlopCountAnalysis(model_eval, dummy_input).total()
        return round(float(flops / 1e9), 3)
    except Exception:
        pass

    # Giá trị tham khảo tiêu chuẩn nếu profiler không chạy được
    known_gmacs = {
        "resnet50": 4.1,
        "resnext50": 4.2,
        "convnext_tiny": 4.5,
        "deit_small": 4.6,
        "vit_small": 4.6,
        "swin_tiny": 4.5,
        "mobilenetv3": 0.22,
        "efficientnet_b0": 0.39,
    }
    for k, v in known_gmacs.items():
        if k in getattr(model, "pretrained_cfg", {}).get("architecture", "").lower():
            return v
    return 0.0
