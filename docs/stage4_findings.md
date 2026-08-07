# Stage 4 Findings — Model Evaluation & Selection

## KKBOX Customer Churn Prediction Platform

Tài liệu này tổng hợp các kết quả chính của **QT5 — Model Evaluation & Selection**.

---

## 1. Executive Summary

QT5 đã hoàn tất phần cốt lõi của model selection và final evaluation.

Final model:

```text
LightGBM
```

Final threshold strategy:

```text
max_validation_f1
```

Final threshold:

```text
0.8411917297941325
```

Final Test results:

```text
PR-AUC    = 0.9361
ROC-AUC   = 0.9903
Precision = 0.8591
Recall    = 0.8873
F1        = 0.8730
Brier     = 0.0319
```

Config được khóa trước khi Test được mở.

Không có model hoặc threshold tuning nào được thực hiện dựa trên Test results.

---

## 2. Validation Findings

Validation rows:

```text
145,644
```

Probability metrics:

```text
PR-AUC      = 0.9344557864
ROC-AUC     = 0.9897519072
Brier Score = 0.0321006200
```

Performance khớp với QT4, xác nhận evaluation pipeline reconstruct đúng model/preprocessing/split artifacts.

---

## 3. Threshold Findings

### Default threshold = 0.5

```text
Precision = 0.7198
Recall    = 0.9325
F1        = 0.8125

FP = 4,756
FN =   884
```

Threshold 0.5 thiên mạnh về Recall nhưng tạo nhiều False Positives.

### Recall ≥ 80%

```text
Threshold = 0.9353
Precision = 0.9108
Recall    = 0.8001
F1        = 0.8518

FP = 1,027
FN = 2,619
```

Scenario này thiên mạnh về Precision nhưng bỏ sót nhiều churn users hơn.

### Max-F1

```text
Threshold = 0.8411917298

Precision = 0.8597
Recall    = 0.8789
F1        = 0.8692

FP = 1,879
FN = 1,587
```

Max-F1 đạt F1-score cao nhất và cung cấp trade-off cân bằng nhất trong ba scenario.

Do chưa có business constraint cụ thể, Max-F1 được chọn làm final threshold strategy.

---

## 4. Final Config Lock

Final config:

```text
model                 = lightgbm
threshold_strategy    = max_validation_f1
threshold             = 0.8411917297941325
calibration_strategy  = none
primary_metric        = pr_auc
locked                = true
```

Model và threshold đã được khóa trước Final Test.

---

## 5. Final Test Findings

Test rows:

```text
145,644
```

Probability metrics:

```text
PR-AUC      = 0.9361119796
ROC-AUC     = 0.9903364444
Brier Score = 0.0318544626
```

Threshold metrics:

```text
Precision = 0.8591174514
Recall    = 0.8873196427
F1        = 0.8729908367
```

Confusion Matrix:

```text
TN = 130,639
FP =   1,906
FN =   1,476
TP =  11,623
```

Predicted positive rate:

```text
9.2891%
```

---

## 6. Generalization Findings

| Metric | Validation | Test | Difference |
|---|---:|---:|---:|
| PR-AUC | 0.9345 | 0.9361 | +0.0017 |
| ROC-AUC | 0.9898 | 0.9903 | +0.0006 |
| Brier | 0.0321 | 0.0319 | -0.0002 |
| Precision | 0.8597 | 0.8591 | -0.0006 |
| Recall | 0.8789 | 0.8873 | +0.0085 |
| F1 | 0.8692 | 0.8730 | +0.0038 |

Không có dấu hiệu performance collapse trên Test.

Model generalize tốt từ Validation sang unseen Test users.

---

## 7. Business-facing Interpretation

Tại final threshold:

```text
Precision ≈ 85.91%
Recall    ≈ 88.73%
```

Có thể diễn giải:

- Trong số users được model flag churn, khoảng 85.91% thực sự churn.
- Model phát hiện khoảng 88.73% churn users.
- 1,476 churn users vẫn bị bỏ sót.
- 1,906 non-churn users bị cảnh báo nhầm.

Trade-off này phù hợp với threshold Max-F1 đã chọn trên Validation.

---

## 8. Calibration Finding

Brier Score rất ổn định:

```text
Validation = 0.0321
Test       = 0.0319
```

Tuy nhiên notebook hiện chưa chứa đầy đủ Calibration Curve hoặc calibration-bin interpretation.

Vì vậy chưa kết luận rằng raw LightGBM probability là perfectly calibrated probability.

Current deployment configuration:

```text
calibration_strategy = none
```

Nếu product chỉ cần risk ranking, điều này chấp nhận được.

Nếu product cần probability mang ý nghĩa xác suất tuyệt đối, calibration nên được đánh giá sâu hơn ở iteration tiếp theo.

---

## 9. Error-analysis Finding

Final Test:

```text
False Positive = 1,906
False Negative = 1,476
```

Error counts nhất quán với Validation.

Notebook hiện chưa chứa feature-level comparison giữa correct predictions và error groups, nên không đưa ra kết luận về pattern feature cụ thể của các False Negatives/False Positives.

---

## 10. Feature Importance Finding

Evaluation pipeline lưu:

```text
models/feature_importance.parquet
```

Nhưng notebook hiện chưa record Top Feature Importance table.

Do đó chưa có đủ source evidence để liệt kê top predictive features trong tài liệu này.

Feature importance không được hiểu là causal importance.

---

## 11. Automated Tests

Toàn bộ test suite:

```text
18 passed in 1.31s
```

QT5 thêm tests cho:

- Probability metrics.
- Threshold metrics.
- Threshold-table construction.
- Max-F1 selection.
- Recall-constrained threshold.
- Invalid constraints.
- Prediction error types.

Tất cả tests đều pass.

---

## 12. QT5 Completion Decision

### Model-selection pipeline

```text
Validation evaluation       ✓
Threshold analysis          ✓
Threshold selection         ✓
Model lock                  ✓
Final Test                  ✓
Generalization evaluation   ✓
Tests                       ✓
```

QT5 được xem là **hoàn thành về mặt model selection và final evaluation**.

### Remaining documentation enrichment

Để notebook trở thành một evaluation report đầy đủ hơn, có thể bổ sung:

```text
Calibration Curve
Feature-level Error Analysis
Top Feature Importance
```

Các mục này không thay đổi model hoặc threshold đã khóa.

---

## 13. Final Recommendation

Selected production candidate:

```text
LightGBM
```

Frozen threshold:

```text
0.8411917297941325
```

Final Test:

```text
PR-AUC  = 0.9361
ROC-AUC = 0.9903
F1      = 0.8730
```

Model có performance cao và Validation/Test consistency tốt.

Không tiếp tục tune model hoặc threshold bằng Test set.

Bước tiếp theo:

```text
QT6 — Deployment
```
