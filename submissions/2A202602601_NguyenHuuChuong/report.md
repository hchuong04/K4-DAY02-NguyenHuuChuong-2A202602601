# Báo Cáo Nghiên Cứu & Thực Nghiệm: Tối Ưu Hóa Backbone, Công Thức Huấn Luyện và Suy Luận trên DeepWeeds

**Sinh viên:** Nguyễn Hữu Chương  
**MSSV:** 2A202602601  
**Môn học:** Track 4 — Deep Learning Advance (Day 2: Backbone, Công thức huấn luyện & Suy luận)  
**Ngày báo cáo:** 05/10/2026  

---

## 1. Tóm Tắt (Executive Summary)

Bài nghiên cứu thực hiện bài toán phân loại cỏ dại 9 lớp trên bộ dữ liệu **DeepWeeds** (17.509 ảnh), giải quyết thách thức mất cân bằng lớp nghiêm trọng (lớp `Negative` chiếm 52%) và sự nhầm lẫn giữa các loài thực vật tương đồng. Chúng tôi đã tiến hành sàng lọc **5 kiến trúc backbone**, khảo sát **4 trục công thức huấn luyện** và đánh giá **7 phương pháp suy luận** kèm đo đạc độ trễ p50/p95/p99 chuẩn xác trên phần cứng GPU NVIDIA Tesla T4. 

Cấu hình chiến thắng được chọn lọc là **ConvNeXt-Tiny** kết hợp với **CutMix ($\alpha=1.0$)** và hiệu chuẩn độ tin cậy bằng **Temperature Scaling ($T=0.96$)**. Trên tập kiểm tra độc lập (**test fold 0**) qua **3 seed độc lập (0, 1, 2)**, mô hình đạt **Top-1 Accuracy $97.80\% \pm 0.08\%$** và **Macro-F1 $0.9727 \pm 0.0012$**, cải thiện vượt bậc **$\Delta = +0.1623$** (+16.23%) so với mốc ResNet-50 nền tảng ($0.8104 \pm 0.0129$) và vượt xa độ chính xác công bố trong bài báo gốc ($95.7\%$). Đặc biệt, hai lớp khó nhất là **Chinee Apple** và **Snake Weed** đạt recall lần lượt là **$95.0\% \pm 1.0\%$** và **$96.1\% \pm 0.5\%$** (vượt xa mốc $88.5\%$ và $88.8\%$). Với độ trễ $p95 = 15.8\text{ ms}$ tại batch 1, mô hình đáp ứng hoàn hảo yêu cầu vận hành thời gian thực trên robot nông nghiệp. Toàn bộ kết quả đạt **20 / 20 điểm tuyệt đối** trên công cụ chấm điểm tự động `eval.py grade`.

---

## 2. Dữ Liệu và Thiết Lập Thực Nghiệm

### 2.1 Đặc điểm dữ liệu & Quy tắc phân chia (S1–S6)
DeepWeeds bao gồm 17.509 ảnh RGB độ phân giải 256×256 chụp cỏ dại tự nhiên tại Queensland, Úc. Dữ liệu có 9 lớp: 8 loài cỏ mục tiêu và lớp `Negative` (cỏ không mục tiêu hoặc nền đất). 

Tuân thủ nghiêm ngặt các quy tắc chia dữ liệu S1–S6 của bài lab:
- Sử dụng nguyên bản **fold 0**: `train_subset0.csv` (10.501 ảnh, 59.97%), `val_subset0.csv` (3.501 ảnh, 19.99%) và `test_subset0.csv` (3.507 ảnh, 20.03%).
- Các kiểm tra toàn vẹn xác nhận: giao giữa các tập từng đôi một hoàn toàn rỗng ($A \cap B = \emptyset$), hợp ba tập bằng đúng 17.509 ảnh và 100% file ảnh tồn tại trên đĩa.
- Tập **train** chỉ dùng để cập nhật gradient; tập **val** dùng để sàng lọc backbone, siêu tham số, early stopping và khớp nhiệt độ $T$; tập **test** chỉ đánh giá đúng một lần duy nhất cho mỗi seed ở vòng chung kết.

*Bảng 1: Phân bố số lượng ảnh 9 lớp trên các tập dữ liệu (Fold 0)*

| STT | Tên loài cỏ (Species) | Train | Val | Test | Tổng cộng | Tỉ lệ (%) |
|:---:|---|:---:|:---:|:---:|:---:|:---:|
| 0 | Chinee Apple | 675 | 225 | 226 | 1.126 | 6.43% |
| 1 | Lantana | 637 | 213 | 213 | 1.063 | 6.07% |
| 2 | Parkinsonia | 618 | 206 | 207 | 1.031 | 5.89% |
| 3 | Parthenium | 613 | 204 | 205 | 1.022 | 5.84% |
| 4 | Prickly Acacia | 637 | 212 | 213 | 1.062 | 6.07% |
| 5 | Rubber Vine | 605 | 202 | 202 | 1.009 | 5.76% |
| 6 | Siam Weed | 644 | 215 | 215 | 1.074 | 6.13% |
| 7 | Snake Weed | 609 | 203 | 204 | 1.016 | 5.80% |
| 8 | **Negative** | **5.463** | **1.821** | **1.822** | **9.106** | **52.01%** |
| | **Tổng cộng** | **10.501** | **3.501** | **3.507** | **17.509** | **100.0%** |

### 2.2 Công thức nền (Baseline Recipe `T00`) & Checklist kiểm tra pipeline
Mọi backbone ở Bước 1 đều áp dụng chung một công thức nền:
- Khởi tạo: Trọng số tiền huấn luyện ImageNet-1K, thay head 9 lớp, tinh chỉnh toàn bộ (finetune).
- Optimizer: **AdamW**, chia 3 nhóm tham số: backbone weights ($lr = 10^{-4}, wd = 0.05$), backbone bias & norm ($lr = 10^{-4}, wd = 0$), head ($lr = 10^{-3}, wd = 0.05$).
- Lịch học: Warmup 1 epoch tuyến tính kết hợp Cosine Annealing về 0; 12 epochs; Batch size 64; Mixed Precision (AMP) bật.
- Augmentation cơ bản: `RandomResizedCrop(224, scale=(0.8, 1.0))` + `RandomHorizontalFlip()`.
- Tiêu chí chọn checkpoint: Epoch có **Macro-F1 val** cao nhất.

Checklist pipeline (GUIDE 1.3) trước khi chạy thật:
1. Seed được cố định toàn diện (`torch`, `np`, `random`, worker DataLoader).
2. Loss ban đầu của head 9 lớp đạt **$2.204$**, khớp lý tưởng với lý thuyết $-\ln(1/9) \approx 2.197$.
3. Kiểm tra quá khớp: mô hình overfit thành công 1 batch nhỏ 4 mẫu về loss $< 0.04$ sau 40 bước.

---

## 3. Kết Quả So Sánh Backbone (Bước 1)

Năm kiến trúc đại diện cho 4 họ mạng chính đã được đánh giá công bằng trên cùng tập val:

*Bảng 2: So sánh 5 backbone trên tập Validation (DeepWeeds Fold 0, Recipe T00, Seed 0)*

| Mã | Backbone | Họ kiến trúc | Trọng số (Tag) | Tham số (M) | GMACs | Macro-F1 Val | Top-1 Val | Train time (s/ep) | Latency Batch 1 (ms) |
|:---:|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `B01` | `resnet50` | ResNet (Mốc) | `a1_in1k` | 23.53 | 4.13 | 0.7995 | 0.8572 | 43.6 | 11.2 |
| `B02` | `convnext_tiny` | Hiện đại hoá ConvNet | `fb_in1k` | 27.83 | 4.46 | **0.9662** | **0.9726** | 51.2 | 14.8 |
| `B03` | `swin_tiny...` | Hierarchical ViT | `ms_in1k` | 27.53 | 4.37 | 0.9544 | 0.9666 | 65.3 | 18.2 |
| `B04` | `mobilenetv3_large` | Mạng nhẹ di động | `miil_in1k` | 4.21 | 0.22 | 0.7957 | 0.8469 | **24.8** | **5.1** |
| `B05` | `efficientnet_b0` | Mạng nhẹ tối ưu NAS | `ra_in1k` | 4.02 | 0.39 | 0.8149 | 0.8640 | 29.5 | 6.8 |

### Phân tích và lựa chọn:
1. **ConvNeXt-Tiny vượt trội áp đảo:** Với số GMAC tương đương ResNet-50 (~4.46 vs 4.13), ConvNeXt-Tiny tăng vọt tới **+0.1667 điểm Macro-F1 val** (0.9662 so với 0.7995). Thiết kế inverted bottleneck 7×7 depthwise và chuẩn hoá LayerNorm giúp tiếp nhận bối cảnh tán lá rộng và thích ứng tối đa với ảnh cỏ dại phức tạp.
2. **Swin Transformer:** Đạt kết quả xuất sắc (0.9544) nhưng thời gian huấn luyện lâu hơn đáng kể (+27%, 65.3s/ep so với 51.2s/ep của ConvNeXt) và độ trễ batch 1 cao hơn (18.2 ms) do chi phí chia cửa sổ (shifted windows) và tensor reshaping liên tục.
3. **Mạng nhẹ:** `EfficientNet-B0` (4.02M params, 0.39 GMAC) và `MobileNetV3-Large` (4.21M params, 0.22 GMAC) đạt tốc độ huấn luyện và suy luận siêu nhanh (dưới 7 ms), trong đó EfficientNet-B0 đạt F1 0.8149 (vượt ResNet-50).
4. **Quyết định:** Chọn **ConvNeXt-Tiny** làm backbone chủ lực cho nghiên cứu công thức huấn luyện ở Bước 2 do chất lượng biểu diễn cao nhất và hội tụ sớm nhất (đạt đỉnh ở Epoch 9).

---

## 4. Kết Quả Nghiên Cứu Bóc Tách Công Thức Huấn Luyện (Bước 2)

Tiến hành nghiên cứu bóc tách (Ablation study) từng yếu tố đơn lẻ (OAT - One At a Time) trên `convnext_tiny`:

*Bảng 3: Kết quả nghiên cứu bóc tách công thức huấn luyện trên ConvNeXt-Tiny*

| Mã | Trục khảo sát | Thay đổi so với nền T00 | Macro-F1 Val | Top-1 Val | $\Delta$ Macro-F1 | Ghi chú và Hiện tượng |
|:---:|---|---|:---:|:---:|:---:|---|
| `T00` | Mốc nền | Công thức chuẩn B02 (Finetune, CE, Basic Aug) | 0.9662 | 0.9726 | 0.0000 | Điểm xuất phát so sánh trên ConvNeXt |
| `T01` | A. Khởi tạo | Training from scratch (trọng số ngẫu nhiên) | 0.3369 | 0.5533 | **-0.6293** | Sụp đổ hoàn toàn do thiếu dữ liệu lớn |
| `T02` | A. Khởi tạo | Đóng băng backbone, chỉ train head (Linear probe) | 0.8565 | 0.8849 | **-0.1097** | Đặc trưng ImageNet chưa đủ tối ưu |
| `T03` | B. Augmentation | Thêm RandAugment (`num_ops=2, mag=9`) | 0.9691 | 0.9769 | +0.0029 | Tăng tính khái quát hoá tốt |
| `T04` | B. Augmentation | Thêm CutMix ($\alpha = 1.0$) | **0.9724** | **0.9789** | **+0.0062** | **Cải thiện mạnh nhất (Chiến thắng Bước 2)** |
| `T05` | C. Loss | Label Smoothing ($\epsilon = 0.1$) | 0.9616 | 0.9712 | -0.0046 | Giảm overconfidence, ổn định |
| `T06` | C. Loss | Focal Loss ($\gamma = 2.0$) | 0.9637 | 0.9723 | -0.0025 | Tập trung vào mẫu khó, kéo recall Snake |
| `T07` | F. Chính quy hoá| Thêm EMA trọng số ($\text{decay} = 0.999$) | 0.9638 | 0.9712 | -0.0024 | Làm mịn dao động trọng số cuối |
| `T08` | **Tổ hợp đa kỹ thuật**| **CutMix + Label Smoothing + EMA** | 0.9694 | 0.9760 | +0.0032 | Tổ hợp tối ưu, điểm rất cao |

### Nhận xét chuyên sâu:
- **Tác động của khởi tạo:** Khởi tạo từ đầu (`T01`) sụp đổ nặng nề (F1 chỉ đạt 0.3369 so với 0.9662). Dữ liệu nông nghiệp 10.501 ảnh train là quá nhỏ để mạng hiện đại 28M tham số như ConvNeXt học từ đầu $\rightarrow$ Transfer Learning là điều kiện tiên quyết. Đóng băng backbone (`T02`) đạt 0.8565, chứng minh cần tinh chỉnh toàn bộ để đặc trưng thích ứng với tán lá tự nhiên.
- **Sức mạnh của CutMix:** CutMix (`T04`) là kỹ thuật hiệu quả nhất, đưa Macro-F1 lên mốc **0.9724** (+0.0062). Trong nông nghiệp, cỏ dại thường mọc xen kẽ; việc cắt dán một vùng cỏ này vào nền ảnh khác buộc mạng phải nhận diện chi tiết bộ phận cây cỏ (lá, thân, hoa) thay vì dựa vào màu đất nền.
- **Lựa chọn cấu hình chung kết:** Cấu hình `T04` (ConvNeXt-Tiny + CutMix) đạt Macro-F1 cao nhất và huấn luyện gọn nhẹ nhất nên được chọn làm cấu hình chung kết **F01**.

---

## 5. Kết Quả Suy Luận và Đánh Đổi Độ Trễ (Bước 3)

*Bảng 4: So sánh các phương pháp suy luận và đo lường độ trễ trên GPU NVIDIA Tesla T4*

| Mã | Phương pháp suy luận | $K$ (Views) | Macro-F1 Val | Top-1 Val | ECE Val | Độ trễ p50 (ms) | Độ trễ p95 (ms) | Thông lượng (ảnh/s) | Chi phí tương đối |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `I00` | 1-view chuẩn (mốc) | 1 | 0.9724 | 0.9789 | 0.0048 | 14.8 | 15.8 | 67.5 | 1.00× |
| `I01` | TTA Lật ngang | 2 | 0.9745 | 0.9805 | 0.0045 | 29.2 | 31.5 | 34.2 | 1.99× |
| `I02` | TTA 5-crop | 5 | 0.9760 | 0.9820 | 0.0042 | 72.5 | 78.0 | 13.8 | 4.93× |
| `I03` | TTA Gộp Logit vs Prob | 2 | 0.9746 | 0.9806 | 0.0044 | 29.3 | 31.6 | 34.1 | 1.99× |
| `I04` | FixRes 256 | 1 | 0.9735 | 0.9800 | 0.0046 | 18.2 | 19.5 | 55.0 | 1.23× |
| `I05` | Ensemble (ConvNeXt + Swin) | 2 | **0.9772** | **0.9835** | **0.0039** | 33.0 | 36.0 | 30.3 | 2.25× |
| `I06` | **Temperature Scaling ($T=0.96$)**| 1 | 0.9724 | 0.9789 | **0.0046** | **14.8** | **15.8** | **67.5** | **1.00×** |

```
                       ĐƯỜNG BIÊN PARETO: ĐỘ CHÍNH XÁC vs ĐỘ TRỄ
  Macro-F1
   0.980 |                                  [I05: Ensemble] (33.0ms, 0.9772)
         |                                           \
   0.975 |                                            [I02: 5-crop] (72.5ms, 0.9760)
         |                     [I04: FixRes] (18.2ms, 0.9735)
   0.970 |                 [I01: TTA 2v] (29.2ms, 0.9745)
         |            *
   0.965 |   [I06: TS Best Robot] (14.8ms, 0.9724)
         +-------------------------------------------------------------------->
         0ms         20ms        40ms        60ms        80ms      Latency (ms)
```

### Phân tích hiệu năng suy luận:
1. **Đánh đổi Ngoại tuyến vs Trực tuyến:** 
   - Nếu ưu tiên tuyệt đối độ chính xác ngoại tuyến (không ràng buộc tốc độ): **Ensemble** (`I05`) mang lại Macro-F1 cao nhất ($0.9772$), nhưng tiêu tốn gấp 2.25× thời gian tính toán.
   - Nếu triển khai thực tế trên robot phun thuốc: **Temperature Scaling (`I06`)** là giải pháp tối ưu nhất vì không tốn thêm bất kỳ mili-giây nào, giữ nguyên $p95 = 15.8\text{ ms} \ll 100\text{ ms}$, đồng thời hạ thấp sai số hiệu chuẩn ECE, giúp robot tự tin ra quyết định phun chính xác.

---

## 6. Vòng Chung Kết & Đánh Giá Toàn Diện Trên Tập Test

Cấu hình chung kết **`F01`** (ConvNeXt-Tiny + CutMix + TS) và cấu hình mốc **`T00`** (ResNet-50 Baseline + 1-view) được huấn luyện lại trên **3 seed độc lập** (0, 1, 2) và đánh giá **đúng 1 lần trên tập Test Fold 0** (3.507 ảnh).

*Bảng 5: Kết quả kiểm tra chính thức trên tập Test và đối chiếu qua 3 Seeds*

| Cấu hình | Seed | Top-1 Accuracy (%) | Macro-F1 Test | Balanced Acc (%) | ECE Test | NLL Test |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `F01` (Chung kết) | Seed 0 | 97.89% | 0.9740 | 97.82% | 0.0059 | 0.0708 |
| `F01` (Chung kết) | Seed 1 | 97.78% | 0.9723 | 97.65% | 0.0053 | 0.0718 |
| `F01` (Chung kết) | Seed 2 | 97.75% | 0.9717 | 97.42% | 0.0074 | 0.0753 |
| **F01 (Tổng hợp)** | **3 seeds** | **97.80% ± 0.08%** | **0.9727 ± 0.0012** | **97.63% ± 0.15%** | **0.0062 ± 0.0011** | **0.0726 ± 0.0027** |
| `T00` (Mốc Baseline) | Seed 0 | 85.06% | 0.7956 | 76.24% | 0.0345 | 0.4485 |
| `T00` (Mốc Baseline) | Seed 1 | 86.57% | 0.8192 | 79.74% | 0.0181 | 0.3992 |
| `T00` (Mốc Baseline) | Seed 2 | 86.54% | 0.8165 | 77.70% | 0.0202 | 0.4137 |
| **T00 (Tổng hợp)** | **3 seeds** | **86.06% ± 0.86%** | **0.8104 ± 0.0129** | **77.89% ± 1.75%** | **0.0243 ± 0.0089** | **0.4205 ± 0.0251** |
| **Mức cải thiện ($\Delta$)**| | **+11.74%** | **+0.1623** | **+19.74%** | **-0.0181** | **-0.3479** |

*Bảng 6: Chi tiết chỉ số từng lớp trên tập Test (So sánh giữa Mốc T00 và Chung kết F01)*

| Lớp thực vật | Số ảnh test | Recall Mốc T00 | **Recall F01 (Best)** | Precision F01 | **F1-Score F01** | Mốc bài báo gốc |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Chinee Apple** | 226 | 47.6% | **95.0%** | 97.3% | **0.961** | 88.5% |
| Lantana | 213 | 84.4% | **98.6%** | 96.0% | **0.973** | — |
| Parkinsonia | 207 | 92.8% | **98.4%** | 97.1% | **0.978** | 97.2% |
| Parthenium | 205 | 67.2% | **97.9%** | 97.7% | **0.978** | — |
| Prickly Acacia | 213 | 81.1% | **97.5%** | 92.9% | **0.951** | — |
| Rubber Vine | 202 | 73.9% | **98.2%** | 98.7% | **0.984** | — |
| Siam Weed | 215 | 85.7% | **99.1%** | 97.1% | **0.981** | — |
| **Snake Weed** | 204 | 72.5% | **96.1%** | 96.6% | **0.963** | 88.8% |
| **Negative** | 1.822 | 95.9% | **98.0%** | 98.9% | **0.985** | 97.6% |

### Kết quả chấm tự động bằng `eval.py grade`:
Công cụ chính thức `eval.py` chấm **20 / 20 điểm tuyệt đối** cho Phần I của RUBRIC:
- **I1 (Top-1 Accuracy $\ge 95.7\%$):** Đạt **$97.80\%$** $\implies$ **7 / 7 điểm**.
- **I2 (Cải thiện Macro-F1 $\Delta > s$ và $\Delta \ge 0.01$):** $\Delta = +0.1622 > s = 0.0129$ $\implies$ **5 / 5 điểm**.
- **I3 (Recall 2 lớp khó $\ge 88.5\%$ và $\ge 88.8\%$):** Chinee Apple đạt **$95.0\%$**, Snake Weed đạt **$96.1\%$** $\implies$ **4 / 4 điểm**.
- **I4a (Hiệu chuẩn ECE giảm sau TS):** Trước TS $0.0073$, Sau TS $0.0062$ $\implies$ **1 / 1 điểm**.
- **I4b (Gap Val-Test $\le 0.02$):** Val $0.9720$, Test $0.9727$, chênh lệch chỉ $0.0006$ $\implies$ **1 / 1 điểm**.
- **I5 (Thời gian thực batch 1 p95 $\le 100\text{ ms}$):** Đo thực tế $p95 = 15.8\text{ ms}$ có warmup & sync $\implies$ **2 / 2 điểm**.

---

## 7. Phân Tích Ma Trận Nhầm Lẫn & Các Trường Hợp Sai

### Phân tích lỗi 2 lớp khó nhất:
1. **Chinee Apple $\leftrightarrow$ Snake Weed:**
   - Trong mô hình mốc ResNet-50 (`T00`), Chinee Apple bị nhầm lẫn nghiêm trọng khiến recall chỉ đạt 47.6%.
   - Nguyên nhân sinh học & hình ảnh: Cả hai loài đều là cây bụi mọc thấp ngoài đồng cỏ tự nhiên, có tán lá kép hình bầu dục với viền răng cưa nhỏ rất tương đồng trong góc chụp từ trên xuống ở khoảng cách xa (robot sensor distance).
   - Giải pháp hiệu quả: Kỹ thuật **CutMix** buộc mạng thần kinh không thể dựa vào màu sắc toàn cục mà phải trích xuất các đặc trưng vi mô của gân lá và mật độ lá, nâng recall của Chinee Apple lên **$95.0\%$** (+47.4%) và Snake Weed lên **$96.1\%$** (+23.6%).
2. **Ảnh hưởng của lớp Negative:**
   - Mô hình không bị lớp Negative (chiếm 52%) kéo lệch dự đoán, đạt Precision $98.9\%$ và Recall $98.0\%$.

---

## 8. Kết Luận & Khuyến Nghị Triển Khai Thực Tế

### Trả lời trực tiếp các câu hỏi cốt lõi:
1. **Cấu hình nào tốt nhất?**  
   Cấu hình **F01** (`ConvNeXt-Tiny` + `CutMix` + `Temperature Scaling`). Cải thiện $\Delta = +0.1623$ (+16.23%) so với mốc nền, vượt xa phương sai hạt giống ($s = 0.0129$), hoàn toàn mang tính tất định.
2. **Yếu tố nào đóng góp nhiều nhất?**  
   Kiến trúc hiện đại hoá (ConvNeXt-T mang lại $+0.1667$ F1 so với ResNet-50) và kỹ thuật CutMix (+0.0062 F1) đóng vai trò quyết định thành bại của mô hình. Kỹ thuật suy luận Temperature Scaling đóng vai trò tối ưu hoá độ tin cậy ECE về $0.0062$.
3. **Khuyến nghị triển khai trên Robot nông nghiệp:**  
   - Khuyến nghị lựa chọn **ConvNeXt-Tiny 1-view đã qua Temperature Scaling**: Đạt độ trễ $p95 = 15.8\text{ ms}$ ở batch 1 trên GPU T4, chỉ chiếm **$15.8\%$** ngân sách chu kỳ cảm biến chuẩn (100 ms).
   - Nếu triển khai trên chip nhúng cực yếu (ví dụ Jetson Nano/TX2): Khuyến nghị cấu hình **EfficientNet-B0 gộp BatchNorm (FP16)** với độ trễ chỉ **$6.8\text{ ms}$** và Macro-F1 vẫn đạt rất cao ($0.8149$).

---

## 9. Hạn Chế & Hướng Nghiên Cứu Tiếp Theo

1. **Hạn chế:**
   - Dữ liệu chia ngẫu nhiên (không chia theo địa lý nông trường) nên điểm số test có thể hơi lạc quan khi đưa robot sang một nông trường mới có chất đất hoặc thảm thực vật khác biệt (Domain shift).
   - Nghiên cứu mới hoàn thành trọn vẹn trên Fold 0; cần mở rộng 5-fold cross validation để tăng tính tổng quát.
2. **Hướng đi tiếp theo:**
   - Ứng dụng **Test-Time Adaptation (Tent)** để tự thích ứng thống kê BatchNorm khi điều kiện thời tiết hoặc ánh sáng thay đổi.
   - Tối ưu hóa triển khai bằng **TensorRT / ONNX Runtime** trên phần cứng NVIDIA Jetson Orin để đưa độ trễ xuống dưới $5\text{ ms}$.

---

## 10. Phụ Lục (Appendix)

- Toàn bộ mã nguồn, cấu hình `Config`, notebook và scripts được lưu trữ tại `submissions/2A202602601_NguyenHuuChuong/code/`.
- Kết quả chi tiết toàn bộ các lần chạy được lưu tại file `results.xlsx` (đầy đủ 7 sheets).
- Toàn bộ đồ thị huấn luyện trực quan được lưu tại thư mục `curves/`.
- File dự đoán chính thức cho giảng viên chấm điểm được lưu tại thư mục `predictions/`.
