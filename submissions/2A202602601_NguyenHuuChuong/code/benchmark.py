"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

Quy tắc đo:
  - warmup: bỏ >= 10 lần chạy đầu
  - đồng bộ GPU: torch.cuda.synchronize() trước và sau đoạn cần đo
  - >= 50 lần đo, báo cáo p50, p95, p99 (không chỉ trung bình)
  - ghi rõ GPU, dtype (FP32/AMP/FP16), batch, độ phân giải, có/không gộp BN, phiên bản torch

Giao diện giữ nguyên:
    bench(fn, warmup, iters, sync) -> dict
    latency_report(model, batch_size, img_size, dtype, device, warmup, iters) -> dict
    tta_latency(model, k_views, **kw) -> dict
"""
from __future__ import annotations

import copy
import time
from typing import Callable

import numpy as np
import torch
import torch.nn as nn


def bench(fn: Callable[[], None], warmup: int = 10, iters: int = 100, sync: Callable[[], None] | None = None) -> dict:
    """Đo thời gian thực thi hàm `fn()` theo mili-giây với warmup và đồng bộ GPU.

    Trả về dict: {"p50": ..., "p95": ..., "p99": ..., "mean": ..., "n": iters}.
    """
    # 1. Warmup
    for _ in range(warmup):
        fn()
    if sync is not None:
        sync()

    # 2. Vòng lặp đo độ trễ
    times = []
    for _ in range(iters):
        if sync is not None:
            sync()
        t0 = time.perf_counter()

        fn()

        if sync is not None:
            sync()
        t1 = time.perf_counter()

        times.append((t1 - t0) * 1000.0)  # chuyển sang ms

    times_arr = np.array(times, dtype=np.float64)
    p50 = float(np.percentile(times_arr, 50))
    p95 = float(np.percentile(times_arr, 95))
    p99 = float(np.percentile(times_arr, 99))
    mean_val = float(np.mean(times_arr))

    return {
        "p50": round(p50, 3),
        "p95": round(p95, 3),
        "p99": round(p99, 3),
        "mean": round(mean_val, 3),
        "n": iters,
    }


def latency_report(model: nn.Module, batch_size: int = 1, img_size: int = 224,
                   dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100) -> dict:
    """Đo độ trễ forward của `model` với tensor giả lập (batch_size, 3, img_size, img_size).

    Trả về dict ghi vào sheet Latency của results.xlsx.
    """
    dev = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    model_eval = copy.deepcopy(model).eval().to(dev)

    if dtype == "fp16" and dev.type == "cuda":
        model_eval = model_eval.half()

    sync_fn = torch.cuda.synchronize if dev.type == "cuda" else None

    # Tạo dummy input
    dummy_input = torch.randn(batch_size, 3, img_size, img_size, device=dev)
    if dtype == "fp16" and dev.type == "cuda":
        dummy_input = dummy_input.half()

    if dtype == "amp" and dev.type == "cuda":
        def forward_fn():
            with torch.inference_mode(), torch.cuda.amp.autocast():
                _ = model_eval(dummy_input)
    else:
        def forward_fn():
            with torch.inference_mode():
                _ = model_eval(dummy_input)

    stats = bench(forward_fn, warmup=warmup, iters=iters, sync=sync_fn)

    gpu_name = torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU"
    throughput = (batch_size / (stats["p50"] / 1000.0)) if stats["p50"] > 0 else 0.0

    return {
        "gpu": gpu_name,
        "dtype": dtype,
        "batch": batch_size,
        "img_size": img_size,
        "p50": stats["p50"],
        "p95": stats["p95"],
        "p99": stats["p99"],
        "images_per_s": round(throughput, 1),
        "torch": torch.__version__,
    }


def tta_latency(model: nn.Module, k_views: int = 2, **kw) -> dict:
    """Đo độ trễ của TTA K views: xấp xỉ K lần một lượt chạy (slide trang 63)."""
    single = latency_report(model, **kw)
    approx_k = round(single["p50"] * k_views, 3)
    return {
        "k_views": k_views,
        "p50_1view": single["p50"],
        "p50_kviews": approx_k,
        "single_report": single,
    }
