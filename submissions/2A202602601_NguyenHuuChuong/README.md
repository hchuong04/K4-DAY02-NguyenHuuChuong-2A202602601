# Hướng Dẫn Tái Lập Bài Làm Lab Day 2 — DeepWeeds

- **Sinh viên:** Nguyễn Hữu Chương
- **MSSV:** 2A202602601
- **Lớp / Khóa:** Track 4 — Deep Learning Advance (Day 2: Backbone, Công thức huấn luyện & Suy luận)
- **Link Notebook Kaggle:** https://www.kaggle.com/code/huuchuong/track4-day2-deeplearning-advance

---

## 1. Môi trường & Phiên bản thư viện

Mã nguồn được phát triển và kiểm thử đồng bộ trên cả máy cục bộ (Windows, Python 3.11) và môi trường GPU Google Colab / Kaggle (Linux, Ubuntu 22.04, GPU NVIDIA T4 16GB):

- **Python:** `3.10.x` / `3.11.x`
- **PyTorch:** `2.3.1` (hoặc mới hơn)
- **Torchvision:** `0.18.1`
- **Timm:** `1.0.30`
- **Pandas:** `2.2.x`
- **NumPy:** `1.26.x`
- **OpenPyXL:** `3.1.x`
- **Thop:** `0.1.1` (đo GMAC / FLOPs)

---

## 2. Cấu trúc thư mục bài nộp

```
submissions/2A202602601_NguyenHuuChuong/
├── README.md               # File này: hướng dẫn chạy, tái lập và cấu hình
├── results.xlsx            # Bảng tổng hợp đầy đủ 7 sheets theo GUIDE mục 6.1
├── report.md               # Báo cáo kết luận khoa học chi tiết (4–8 trang)
├── curves/                 # Biểu đồ training cho từng thí nghiệm (B01-B05, T00-T08, F01)
├── predictions/            # File dự đoán test và val cho F01 và T00 (3 seed)
└── code/
    ├── dataset.py          # Xử lý data, split check, transforms, dataset, dataloader
    ├── model.py            # timm backbone, freeze, param_groups, FLOPs/GMAC
    ├── losses.py           # LabelSmoothingCE, FocalLoss, ClassWeights, Mixup/CutMix
    ├── train.py            # Config, Optimizer, Scheduler, EMA, train_one_epoch, run(cfg)
    ├── inference.py        # TTA, Aggregation, Temperature Scaling, Ensemble, Fuse BN
    ├── benchmark.py        # Đo latency p50/p95/p99 với warmup và GPU synchronize
    ├── export_results.py   # Script tự động tạo results.xlsx
    ├── test_pipeline.py    # Unit test toàn diện các module
    └── lab_day2.ipynb      # Notebook hoàn chỉnh chạy trên Kaggle / Colab
```

---

## 3. Thứ tự các bước chạy lại toàn bộ thí nghiệm

### Chạy trên Google Colab / Kaggle (Khuyên Dùng)
1. Tải file `code/lab_day2.ipynb` lên Google Colab hoặc Kaggle Notebooks.
2. Bật GPU:
   - Trên Colab: **Runtime** $\to$ **Change runtime type** $\to$ Chọn **T4 GPU**.
   - Trên Kaggle: **Settings** $\to$ **Accelerator** $\to$ Chọn **GPU T4 x2**, bật **Internet: On**.
3. Chạy lần lượt các ô từ trên xuống dưới (hoặc bấm **Run all**):
   - Ô 0–1: Tự động cài đặt thư viện và tải dữ liệu từ Zenodo & GitHub.
   - Ô 2: Khám phá dữ liệu EDA, kiểm tra quy tắc S1–S6.
   - Ô 3: Checklist pipeline (loss ban đầu, overfit 1 batch nhỏ).
   - Ô 4: Huấn luyện so sánh $\ge 5$ Backbone (`B01` - `B05`).
   - Ô 5: Khảo sát bóc tách công thức huấn luyện (`T01` - `T08`).
   - Ô 6: Thử nghiệm suy luận và đo độ trễ chuẩn xác (`I00` - `I07`).
   - Ô 7–8: Vòng chung kết 3 seed (`F01_seed0..2` vs `T00_seed0..2`) và chạy `eval.py score`, `eval.py grade`.
   - Ô 9: Xuất file `results.xlsx`.

## 4. Hạt giống ngẫu nhiên (Seeds) đã sử dụng
- Huấn luyện sàng lọc backbone và ablation: Cố định `seed = 0`.
- Vòng chung kết (`F01` và `T00` mốc): Huấn luyện trên **3 seed độc lập**: `seed = 0`, `seed = 1`, `seed = 2`.
- Tính chỉ số tổng hợp: Mean $\pm$ Std mẫu (`ddof=1`) qua 3 seeds.
