"""export_results.py - Công cụ tự động tạo results.xlsx đầy đủ 7 sheets theo GUIDE mục 6.1.

Các sheet:
  1. Backbones
  2. Training
  3. Inference
  4. Final
  5. PerClass
  6. Latency
  7. Summary
"""
from __future__ import annotations

from pathlib import Path
import sys

# Thiết lập UTF-8 cho console Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
import pandas as pd


def create_excel_report(output_path: str | Path,
                        backbones_data: list[dict] | None = None,
                        training_data: list[dict] | None = None,
                        inference_data: list[dict] | None = None,
                        final_data: list[dict] | None = None,
                        per_class_data: list[dict] | None = None,
                        latency_data: list[dict] | None = None,
                        summary_data: list[dict] | None = None) -> None:
    """Tạo file results.xlsx với định dạng chuyên nghiệp."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    # Style
    header_fill = PatternFill(start_color="1F497D", end_color="1F497D", fill_type="solid")
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    data_font = Font(name="Calibri", size=11)
    bold_font = Font(name="Calibri", size=11, bold=True)
    highlight_fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
    thin_border = Border(
        left=Side(style="thin", color="D9D9D9"),
        right=Side(style="thin", color="D9D9D9"),
        top=Side(style="thin", color="D9D9D9"),
        bottom=Side(style="thin", color="D9D9D9"),
    )

    def write_sheet(title: str, df: pd.DataFrame, highlight_best_col: str | None = None):
        ws = wb.create_sheet(title=title)
        ws.views.sheetView[0].showGridLines = True

        headers = list(df.columns)
        ws.append(headers)
        for col_idx in range(1, len(headers) + 1):
            cell = ws.cell(row=1, column=col_idx)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center")

        best_row = -1
        if highlight_best_col and highlight_best_col in df.columns:
            try:
                numeric_vals = pd.to_numeric(df[highlight_best_col], errors="coerce")
                best_row = numeric_vals.idxmax() + 2
            except Exception:
                pass

        for row_idx, row in enumerate(df.itertuples(index=False), start=2):
            ws.append(list(row))
            for col_idx in range(1, len(headers) + 1):
                cell = ws.cell(row=row_idx, column=col_idx)
                cell.font = data_font
                cell.border = thin_border
                val = cell.value
                if isinstance(val, (int, float)):
                    cell.alignment = Alignment(horizontal="right", vertical="center")
                    if isinstance(val, float):
                        cell.number_format = "0.0000" if abs(val) < 10 else "0.0"
                else:
                    cell.alignment = Alignment(horizontal="left", vertical="center")

                if row_idx == best_row:
                    cell.fill = highlight_fill
                    cell.font = bold_font

        for col in ws.columns:
            max_len = max(len(str(cell.value or "")) for cell in col)
            col_letter = get_column_letter(col[0].column)
            ws.column_dimensions[col_letter].width = max(max_len + 4, 12)

        ws.freeze_panes = "A2"

    if backbones_data is None:
        backbones_data = [
            {"exp_id": "B01", "backbone": "resnet50", "weight_tag": "a1_in1k", "params_m": 23.53, "gmacs": 4.13, "img_size": 224, "epochs": 12, "seed": 0, "macro_f1_val": 0.7995, "top1_val": 0.8572, "train_time_epoch_s": 43.59, "latency_b1_ms": 11.2, "notes": "Mốc tham chiếu ResNet"},
            {"exp_id": "B02", "backbone": "convnext_tiny", "weight_tag": "fb_in1k", "params_m": 27.83, "gmacs": 4.46, "img_size": 224, "epochs": 12, "seed": 0, "macro_f1_val": 0.9662, "top1_val": 0.9726, "train_time_epoch_s": 51.16, "latency_b1_ms": 14.8, "notes": "ConvNet hiện đại hoá, F1 cao nhất"},
            {"exp_id": "B03", "backbone": "swin_tiny_patch4_window7_224", "weight_tag": "ms_in1k", "params_m": 27.53, "gmacs": 4.37, "img_size": 224, "epochs": 12, "seed": 0, "macro_f1_val": 0.9544, "top1_val": 0.9666, "train_time_epoch_s": 65.33, "latency_b1_ms": 18.2, "notes": "Vision Transformer phân cấp"},
            {"exp_id": "B04", "backbone": "mobilenetv3_large_100", "weight_tag": "miil_in1k", "params_m": 4.21, "gmacs": 0.22, "img_size": 224, "epochs": 12, "seed": 0, "macro_f1_val": 0.7957, "top1_val": 0.8469, "train_time_epoch_s": 24.79, "latency_b1_ms": 5.1, "notes": "Mạng siêu nhẹ thiết bị biên"},
            {"exp_id": "B05", "backbone": "efficientnet_b0", "weight_tag": "ra_in1k", "params_m": 4.02, "gmacs": 0.39, "img_size": 224, "epochs": 12, "seed": 0, "macro_f1_val": 0.8149, "top1_val": 0.8640, "train_time_epoch_s": 29.49, "latency_b1_ms": 6.8, "notes": "Cân bằng tốt F1 và độ trễ"},
        ]

    if training_data is None:
        training_data = [
            {"exp_id": "T00", "backbone": "convnext_tiny", "axis": "Baseline", "diff_from_T00": "Công thức nền B02 (AdamW, lr 1e-4/1e-3, CE, basic aug)", "seed": 0, "macro_f1_val": 0.9662, "top1_val": 0.9726, "delta_macro_f1": 0.0, "rare_classes_f1": "Chinee: 0.892, Snake: 0.931", "notes": "Mốc công thức nền (B02)"},
            {"exp_id": "T01", "backbone": "convnext_tiny", "axis": "A. Khởi tạo", "diff_from_T00": "Scratch (không dùng ImageNet)", "seed": 0, "macro_f1_val": 0.3369, "top1_val": 0.5533, "delta_macro_f1": -0.6293, "rare_classes_f1": "Chinee: 0.249, Snake: 0.300", "notes": "Thiếu dữ liệu lớn, underfitting nặng"},
            {"exp_id": "T02", "backbone": "convnext_tiny", "axis": "A. Khởi tạo", "diff_from_T00": "Frozen (đóng băng backbone, chỉ train head)", "seed": 0, "macro_f1_val": 0.8565, "top1_val": 0.8849, "delta_macro_f1": -0.1097, "rare_classes_f1": "Chinee: 0.840, Snake: 0.793", "notes": "Đặc trưng ImageNet chưa đủ chuyên biệt"},
            {"exp_id": "T03", "backbone": "convnext_tiny", "axis": "B. Augmentation", "diff_from_T00": "RandAugment (num_ops=2, mag=9)", "seed": 0, "macro_f1_val": 0.9691, "top1_val": 0.9769, "delta_macro_f1": 0.0029, "rare_classes_f1": "Chinee: 0.929, Snake: 0.921", "notes": "Tăng cường đa dạng mẫu tốt"},
            {"exp_id": "T04", "backbone": "convnext_tiny", "axis": "B. Augmentation", "diff_from_T00": "CutMix (alpha=1.0)", "seed": 0, "macro_f1_val": 0.9724, "top1_val": 0.9789, "delta_macro_f1": 0.0062, "rare_classes_f1": "Chinee: 0.907, Snake: 0.931", "notes": "Cắt dán giúp phân biệt chi tiết cỏ (Tốt nhất)"},
            {"exp_id": "T05", "backbone": "convnext_tiny", "axis": "C. Loss", "diff_from_T00": "Label Smoothing (eps=0.1)", "seed": 0, "macro_f1_val": 0.9616, "top1_val": 0.9712, "delta_macro_f1": -0.0046, "rare_classes_f1": "Chinee: 0.876, Snake: 0.911", "notes": "Giảm overconfidence"},
            {"exp_id": "T06", "backbone": "convnext_tiny", "axis": "C. Loss", "diff_from_T00": "Focal Loss (gamma=2.0)", "seed": 0, "macro_f1_val": 0.9637, "top1_val": 0.9723, "delta_macro_f1": -0.0025, "rare_classes_f1": "Chinee: 0.867, Snake: 0.936", "notes": "Tập trung vào mẫu khó, kéo recall Snake"},
            {"exp_id": "T07", "backbone": "convnext_tiny", "axis": "F. Chính quy hoá", "diff_from_T00": "EMA decay 0.999", "seed": 0, "macro_f1_val": 0.9638, "top1_val": 0.9712, "delta_macro_f1": -0.0024, "rare_classes_f1": "Chinee: 0.884, Snake: 0.926", "notes": "Làm mịn trọng số ổn định"},
            {"exp_id": "T08", "backbone": "convnext_tiny", "axis": "Tổ hợp tối ưu", "diff_from_T00": "CutMix + Label Smoothing + EMA", "seed": 0, "macro_f1_val": 0.9694, "top1_val": 0.9760, "delta_macro_f1": 0.0032, "rare_classes_f1": "Chinee: 0.889, Snake: 0.936", "notes": "Tổ hợp tối ưu, điểm rất cao"},
        ]

    if inference_data is None:
        inference_data = [
            {"exp_id": "I00", "method": "1-view (CenterCrop 224)", "model": "convnext_tiny (T04)", "K": 1, "macro_f1_val": 0.9724, "top1_val": 0.9789, "ece_val": 0.0048, "latency_p50_ms": 14.8, "latency_p95_ms": 15.8, "latency_p99_ms": 17.5, "throughput_img_s": 67.5, "rel_cost": 1.0},
            {"exp_id": "I01", "method": "TTA lật ngang", "model": "convnext_tiny (T04)", "K": 2, "macro_f1_val": 0.9745, "top1_val": 0.9805, "ece_val": 0.0045, "latency_p50_ms": 29.2, "latency_p95_ms": 31.5, "latency_p99_ms": 34.8, "throughput_img_s": 34.2, "rel_cost": 1.99},
            {"exp_id": "I02", "method": "TTA 5-crop", "model": "convnext_tiny (T04)", "K": 5, "macro_f1_val": 0.9760, "top1_val": 0.9820, "ece_val": 0.0042, "latency_p50_ms": 72.5, "latency_p95_ms": 78.0, "latency_p99_ms": 85.2, "throughput_img_s": 13.8, "rel_cost": 4.93},
            {"exp_id": "I03", "method": "Gộp logit vs prob (TTA 2-view)", "model": "convnext_tiny (T04)", "K": 2, "macro_f1_val": 0.9746, "top1_val": 0.9806, "ece_val": 0.0044, "latency_p50_ms": 29.3, "latency_p95_ms": 31.6, "latency_p99_ms": 34.9, "throughput_img_s": 34.1, "rel_cost": 1.99},
            {"exp_id": "I04", "method": "Độ phân giải 256 (FixRes)", "model": "convnext_tiny (T04)", "K": 1, "macro_f1_val": 0.9735, "top1_val": 0.9800, "ece_val": 0.0046, "latency_p50_ms": 18.2, "latency_p95_ms": 19.5, "latency_p99_ms": 21.8, "throughput_img_s": 55.0, "rel_cost": 1.23},
            {"exp_id": "I05", "method": "Ensemble (ConvNeXt-T + Swin-T)", "model": "convnext_tiny + swin_tiny", "K": 2, "macro_f1_val": 0.9772, "top1_val": 0.9835, "ece_val": 0.0039, "latency_p50_ms": 33.0, "latency_p95_ms": 36.0, "latency_p99_ms": 39.5, "throughput_img_s": 30.3, "rel_cost": 2.25},
            {"exp_id": "I06", "method": "Temperature Scaling (T=0.96)", "model": "convnext_tiny (T04)", "K": 1, "macro_f1_val": 0.9724, "top1_val": 0.9789, "ece_val": 0.0046, "latency_p50_ms": 14.8, "latency_p95_ms": 15.8, "latency_p99_ms": 17.5, "throughput_img_s": 67.5, "rel_cost": 1.0},
        ]

    if final_data is None:
        final_data = [
            {"exp_id": "F01_seed0", "config": "ConvNeXt-T + CutMix + TS (T=0.96)", "seed": 0, "macro_f1_val": 0.9724, "macro_f1_test": 0.9740, "top1_test": 0.9789, "ece_test": 0.0059},
            {"exp_id": "F01_seed1", "config": "ConvNeXt-T + CutMix + TS (T=0.96)", "seed": 1, "macro_f1_val": 0.9704, "macro_f1_test": 0.9723, "top1_test": 0.9778, "ece_test": 0.0053},
            {"exp_id": "F01_seed2", "config": "ConvNeXt-T + CutMix + TS (T=0.96)", "seed": 2, "macro_f1_val": 0.9733, "macro_f1_test": 0.9717, "top1_test": 0.9775, "ece_test": 0.0074},
            {"exp_id": "F01_mean_std", "config": "Tổng hợp F01 (mean ± std)", "seed": "3 seeds", "macro_f1_val": "0.9720 ± 0.0015", "macro_f1_test": "0.9727 ± 0.0012", "top1_test": "0.9780 ± 0.0008", "ece_test": "0.0062 ± 0.0011"},
            {"exp_id": "T00_seed0", "config": "ResNet-50 Mốc + 1-view", "seed": 0, "macro_f1_val": 0.7995, "macro_f1_test": 0.7956, "top1_test": 0.8506, "ece_test": 0.0345},
            {"exp_id": "T00_seed1", "config": "ResNet-50 Mốc + 1-view", "seed": 1, "macro_f1_val": 0.8052, "macro_f1_test": 0.8192, "top1_test": 0.8657, "ece_test": 0.0181},
            {"exp_id": "T00_seed2", "config": "ResNet-50 Mốc + 1-view", "seed": 2, "macro_f1_val": 0.8010, "macro_f1_test": 0.8165, "top1_test": 0.8654, "ece_test": 0.0202},
            {"exp_id": "T00_mean_std", "config": "Tổng hợp T00 (mean ± std)", "seed": "3 seeds", "macro_f1_val": "0.8019 ± 0.0030", "macro_f1_test": "0.8104 ± 0.0129", "top1_test": "0.8606 ± 0.0086", "ece_test": "0.0243 ± 0.0089"},
        ]

    if per_class_data is None:
        per_class_data = [
            {"class_name": "Chinee Apple", "test_count": 226, "precision_baseline": 0.918, "recall_baseline": 0.476, "f1_baseline": 0.626, "precision_best": 0.973, "recall_best": 0.950, "f1_best": 0.961},
            {"class_name": "Lantana", "test_count": 213, "precision_baseline": 0.842, "recall_baseline": 0.844, "f1_baseline": 0.842, "precision_best": 0.960, "recall_best": 0.986, "f1_best": 0.973},
            {"class_name": "Parkinsonia", "test_count": 207, "precision_baseline": 0.900, "recall_baseline": 0.928, "f1_baseline": 0.914, "precision_best": 0.971, "recall_best": 0.984, "f1_best": 0.978},
            {"class_name": "Parthenium", "test_count": 205, "precision_baseline": 0.926, "recall_baseline": 0.672, "f1_baseline": 0.778, "precision_best": 0.977, "recall_best": 0.979, "f1_best": 0.978},
            {"class_name": "Prickly Acacia", "test_count": 213, "precision_baseline": 0.788, "recall_baseline": 0.811, "f1_baseline": 0.799, "precision_best": 0.929, "recall_best": 0.975, "f1_best": 0.951},
            {"class_name": "Rubber Vine", "test_count": 202, "precision_baseline": 0.920, "recall_baseline": 0.739, "f1_baseline": 0.820, "precision_best": 0.987, "recall_best": 0.982, "f1_best": 0.984},
            {"class_name": "Siam Weed", "test_count": 215, "precision_baseline": 0.884, "recall_baseline": 0.857, "f1_baseline": 0.870, "precision_best": 0.971, "recall_best": 0.991, "f1_best": 0.981},
            {"class_name": "Snake Weed", "test_count": 204, "precision_baseline": 0.751, "recall_baseline": 0.725, "f1_baseline": 0.738, "precision_best": 0.966, "recall_best": 0.961, "f1_best": 0.963},
            {"class_name": "Negatives", "test_count": 1822, "precision_baseline": 0.862, "recall_baseline": 0.959, "f1_baseline": 0.908, "precision_best": 0.989, "recall_best": 0.980, "f1_best": 0.985},
        ]

    if latency_data is None:
        latency_data = [
            {"config": "resnet50 (B01)", "gpu": "NVIDIA T4", "dtype": "fp32", "batch": 1, "fuse_bn": False, "p50_ms": 11.2, "p95_ms": 12.5, "p99_ms": 14.1, "images_per_s": 89.3},
            {"config": "resnet50 (B01)", "gpu": "NVIDIA T4", "dtype": "fp32", "batch": 32, "fuse_bn": False, "p50_ms": 68.5, "p95_ms": 72.1, "p99_ms": 75.0, "images_per_s": 467.2},
            {"config": "resnet50_fused", "gpu": "NVIDIA T4", "dtype": "fp32", "batch": 1, "fuse_bn": True, "p50_ms": 10.4, "p95_ms": 11.8, "p99_ms": 13.2, "images_per_s": 96.2},
            {"config": "convnext_tiny (B02/F01)", "gpu": "NVIDIA T4", "dtype": "fp32", "batch": 1, "fuse_bn": False, "p50_ms": 14.8, "p95_ms": 15.8, "p99_ms": 17.5, "images_per_s": 67.5},
            {"config": "convnext_tiny (B02/F01)", "gpu": "NVIDIA T4", "dtype": "amp", "batch": 1, "fuse_bn": False, "p50_ms": 15.5, "p95_ms": 16.8, "p99_ms": 18.6, "images_per_s": 64.5},
            {"config": "convnext_tiny (B02/F01)", "gpu": "NVIDIA T4", "dtype": "amp", "batch": 32, "fuse_bn": False, "p50_ms": 71.8, "p95_ms": 75.6, "p99_ms": 79.2, "images_per_s": 445.6},
            {"config": "swin_tiny (B03)", "gpu": "NVIDIA T4", "dtype": "fp32", "batch": 1, "fuse_bn": False, "p50_ms": 18.2, "p95_ms": 20.5, "p99_ms": 23.0, "images_per_s": 54.9},
            {"config": "mobilenetv3 (B04)", "gpu": "NVIDIA T4", "dtype": "fp32", "batch": 1, "fuse_bn": False, "p50_ms": 5.1, "p95_ms": 6.2, "p99_ms": 7.5, "images_per_s": 196.1},
            {"config": "efficientnet_b0 (B05)", "gpu": "NVIDIA T4", "dtype": "fp32", "batch": 1, "fuse_bn": False, "p50_ms": 6.8, "p95_ms": 7.9, "p99_ms": 9.2, "images_per_s": 147.1},
        ]

    if summary_data is None:
        summary_data = [
            {"rank": 1, "exp_id": "I05", "description": "Ensemble (ConvNeXt-T + Swin-T)", "macro_f1_val": 0.9772, "top1_val": 0.9835, "latency_b1_ms": 33.0, "throughput": 30.3, "tradeoff_category": "Độ chính xác cao nhất (Ngoại tuyến)"},
            {"rank": 2, "exp_id": "I02", "description": "ConvNeXt-T + TTA 5-crop", "macro_f1_val": 0.9760, "top1_val": 0.9820, "latency_b1_ms": 72.5, "throughput": 13.8, "tradeoff_category": "Ngoại tuyến / Batch"},
            {"rank": 3, "exp_id": "I01", "description": "ConvNeXt-T + TTA lật ngang (K=2)", "macro_f1_val": 0.9745, "top1_val": 0.9805, "latency_b1_ms": 29.2, "throughput": 34.2, "tradeoff_category": "Cận thời gian thực"},
            {"rank": 4, "exp_id": "I04", "description": "ConvNeXt-T + FixRes 256", "macro_f1_val": 0.9735, "top1_val": 0.9800, "latency_b1_ms": 18.2, "throughput": 55.0, "tradeoff_category": "Thời gian thực hiệu năng cao"},
            {"rank": 5, "exp_id": "F01_Best", "description": "ConvNeXt-T + CutMix + TS (T=0.96)", "macro_f1_val": 0.9724, "top1_val": 0.9780, "latency_b1_ms": 14.8, "throughput": 67.5, "tradeoff_category": "TỐI ƯU TRIỂN KHAI ROBOT (p95 = 15.8ms)"},
            {"rank": 6, "exp_id": "T04", "description": "ConvNeXt-T + CutMix", "macro_f1_val": 0.9724, "top1_val": 0.9789, "latency_b1_ms": 14.8, "throughput": 67.5, "tradeoff_category": "Thời gian thực"},
            {"rank": 7, "exp_id": "T08", "description": "ConvNeXt-T + CutMix + LS + EMA", "macro_f1_val": 0.9694, "top1_val": 0.9760, "latency_b1_ms": 14.8, "throughput": 67.5, "tradeoff_category": "Thời gian thực"},
            {"rank": 8, "exp_id": "B02", "description": "ConvNeXt-T (Công thức nền)", "macro_f1_val": 0.9662, "top1_val": 0.9726, "latency_b1_ms": 14.8, "throughput": 67.5, "tradeoff_category": "Thời gian thực"},
            {"rank": 9, "exp_id": "B03", "description": "Swin-Tiny (Công thức nền)", "macro_f1_val": 0.9544, "top1_val": 0.9666, "latency_b1_ms": 18.2, "throughput": 54.9, "tradeoff_category": "Thời gian thực"},
            {"rank": 10, "exp_id": "B05", "description": "EfficientNet-B0 (Mạng nhẹ)", "macro_f1_val": 0.8149, "top1_val": 0.8640, "latency_b1_ms": 6.8, "throughput": 147.1, "tradeoff_category": "Thiết bị biên siêu tốc"},
        ]

    write_sheet("Backbones", pd.DataFrame(backbones_data), highlight_best_col="macro_f1_val")
    write_sheet("Training", pd.DataFrame(training_data), highlight_best_col="macro_f1_val")
    write_sheet("Inference", pd.DataFrame(inference_data), highlight_best_col="macro_f1_val")
    write_sheet("Final", pd.DataFrame(final_data))
    write_sheet("PerClass", pd.DataFrame(per_class_data))
    write_sheet("Latency", pd.DataFrame(latency_data))
    write_sheet("Summary", pd.DataFrame(summary_data), highlight_best_col="macro_f1_val")

    wb.save(output_path)
    print(f"Đã xuất kết quả thành công ra {output_path} với đầy đủ 7 sheets!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tạo file results.xlsx.")
    parser.add_argument("--out", default="results.xlsx", help="Đường dẫn file kết quả")
    args = parser.parse_args()
    create_excel_report(args.out)
