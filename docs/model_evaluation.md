# Model Evaluation & Selection

## KKBOX Customer Churn Prediction Platform

Tài liệu này mô tả **QT5 — Model Evaluation & Selection** của dự án KKBOX Customer Churn Prediction.

Nguồn kết quả chính:

```text
notebooks/04_model_evaluation.ipynb
models/evaluation_results.json
models/final_model_config.json
```

Production evaluation pipeline:

```text
src/kkbox_churn_prediction/modeling/evaluation.py
scripts/evaluate_model.py
tests/test_evaluation.py
```

---

## 1. Mục tiêu QT5

QT5 tiếp nhận các model artifacts từ QT4 và thực hiện:

1. Reconstruct đúng Validation/Test split đã khóa ở QT4.
2. Đánh giá probability-ranking performance trên Validation.
3. Phân tích Precision-Recall và ROC.
4. So sánh classification thresholds.
5. Chọn threshold chỉ bằng Validation.
6. Khóa model + threshold trước khi mở Test.
7. Thực hiện Final Test đúng một lần.
8. So sánh Validation và Test để kiểm tra generalization.
9. Lưu prediction/evaluation artifacts cho deployment.

QT5 không train lại model.

---

## 2. Model Candidate từ QT4

Primary candidate:

```text
LightGBM
```

QT4 Validation performance:

```text
PR-AUC  ≈ 0.9345
ROC-AUC ≈ 0.9898
```

Secondary candidate:

```text
Random Forest
```

LightGBM được đưa vào QT5 vì có Validation PR-AUC và ROC-AUC cao nhất trong QT4.

---

## 3. Fixed Validation Split

QT5 không tạo split mới.

Split assignment đã được lưu từ QT4:

```text
data/processed/model_splits.parquet
```

Validation rows:

```text
145,644
```

Sau khi áp dụng preprocessor đã fit ở QT4:

```text
Processed Validation shape = (145644, 88)
```

Điều này đảm bảo QT5 sử dụng đúng dữ liệu đã được tách trước đó và tránh split leakage.

---

## 4. Validation Probability Metrics

LightGBM trên Validation:

| Metric | Giá trị |
|---|---:|
| PR-AUC | **0.9344557864** |
| ROC-AUC | **0.9897519072** |
| Brier Score | **0.0321006200** |

PR-AUC và ROC-AUC khớp với QT4, xác nhận rằng:

- Fixed split được reconstruct đúng.
- Saved preprocessor hoạt động đúng.
- Saved LightGBM model được load đúng.
- Evaluation pipeline nhất quán với training pipeline.

PR-AUC tiếp tục là primary metric do target bị class imbalance.

---

## 5. Threshold Analysis

Ba threshold strategies được phân tích trên Validation.

| Strategy | Threshold | Precision | Recall | F1 | FP | FN | Predicted Positive Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Default 0.5 | 0.5000 | 0.7198 | **0.9325** | 0.8125 | 4,756 | **884** | 0.1165 |
| **Max F1** | **0.8412** | **0.8597** | **0.8789** | **0.8692** | 1,879 | 1,587 | 0.0920 |
| Recall ≥ 80% | 0.9353 | **0.9108** | 0.8001 | 0.8518 | **1,027** | 2,619 | 0.0790 |

---

## 6. Threshold Interpretation

### Threshold = 0.5

Ưu điểm:

- Recall rất cao: **0.9325**.
- Chỉ bỏ sót 884 churn users trên Validation.

Nhược điểm:

- Precision chỉ **0.7198**.
- False Positive lên tới **4,756** users.
- Predicted positive rate **11.65%**, cao hơn actual churn prevalence.

Threshold này thiên mạnh về Recall.

### Recall ≥ 80%

Ưu điểm:

- Precision cao nhất: **0.9108**.
- False Positive thấp nhất: **1,027**.

Nhược điểm:

- Recall chỉ khoảng **0.8001**.
- False Negative tăng lên **2,619**.

Threshold này thiên mạnh về Precision.

### Max-F1

Kết quả:

```text
Threshold = 0.8411917297941325
Precision = 0.8596923536
Recall    = 0.8788549618
F1        = 0.8691680507
```

Confusion Matrix:

```text
TN = 130,665
FP =   1,879
FN =   1,587
TP =  11,513
```

Đây là threshold có F1-score cao nhất và tạo trade-off cân bằng giữa Precision và Recall.

Vì project chưa có một business constraint chính thức về minimum Recall, intervention capacity hoặc FP/FN cost, Max-F1 được chọn làm threshold strategy cuối.

---

## 7. Final Model Configuration

Final configuration được khóa trước khi Test được mở:

```json
{
  "model": "lightgbm",
  "threshold_strategy": "max_validation_f1",
  "threshold": 0.8411917297941325,
  "calibration_strategy": "none",
  "primary_metric": "pr_auc",
  "locked": true
}
```

Điều này đảm bảo:

- Model không được thay đổi sau khi xem Test.
- Threshold không được điều chỉnh dựa trên Test.
- Test giữ vai trò unbiased final evaluation set.

---

## 8. Final Test Evaluation

Sau khi configuration đã được khóa, Test set được mở đúng một lần.

Test rows:

```text
145,644
```

Final Test probability metrics:

| Metric | Test |
|---|---:|
| PR-AUC | **0.9361119796** |
| ROC-AUC | **0.9903364444** |
| Brier Score | **0.0318544626** |

Final Test threshold metrics:

| Metric | Test |
|---|---:|
| Threshold | 0.8411917298 |
| Precision | **0.8591174514** |
| Recall | **0.8873196427** |
| F1 | **0.8729908367** |
| Predicted Positive Rate | 0.0928908846 |

Final Test Confusion Matrix:

```text
TN = 130,639
FP =   1,906
FN =   1,476
TP =  11,623
```

---

## 9. Validation vs Test

| Metric | Validation | Test | Test - Validation |
|---|---:|---:|---:|
| PR-AUC | 0.9345 | 0.9361 | +0.0017 |
| ROC-AUC | 0.9898 | 0.9903 | +0.0006 |
| Brier Score | 0.0321 | 0.0319 | -0.0002 |
| Precision | 0.8597 | 0.8591 | -0.0006 |
| Recall | 0.8789 | 0.8873 | +0.0085 |
| F1 | 0.8692 | 0.8730 | +0.0038 |

Không có performance degradation đáng kể khi chuyển từ Validation sang Test.

Test performance thậm chí cao hơn nhẹ ở:

- PR-AUC.
- ROC-AUC.
- Recall.
- F1.
- Brier Score thấp hơn nhẹ.

Điều này cho thấy selected LightGBM model generalize tốt trên unseen Test users.

---

## 10. Final Classification Interpretation

Với threshold đã khóa:

```text
0.8411917297941325
```

model đạt trên Test:

```text
Precision ≈ 85.91%
Recall    ≈ 88.73%
F1        ≈ 87.30%
```

Diễn giải:

- Khoảng 85.91% users được model cảnh báo churn thực sự churn.
- Model phát hiện được khoảng 88.73% số churn users.
- 1,476 churn users bị bỏ sót.
- 1,906 non-churn users bị cảnh báo nhầm.

Actual Test churn users:

```text
TP + FN = 11,623 + 1,476 = 13,099
```

Predicted churn users:

```text
TP + FP = 11,623 + 1,906 = 13,529
```

Predicted positive rate:

```text
≈ 9.29%
```

gần với actual churn prevalence khoảng 9%.

Đây là một sanity check tốt, nhưng không phải tiêu chí dùng để chọn threshold.

---

## 11. Calibration

Brier Score:

```text
Validation = 0.0321006200
Test       = 0.0318544626
```

Evaluation configuration hiện lưu:

```text
calibration_strategy = none
```

Notebook hiện ghi nhận Brier Score nhưng chưa document đầy đủ Calibration Curve hoặc calibration-bin analysis.

Vì LightGBM sử dụng imbalance weighting trong training, raw `predict_proba()` nên được xem thận trọng nếu deployment muốn diễn giải trực tiếp thành xác suất churn thực tế.

Nếu sản phẩm chỉ sử dụng output để ranking/risk scoring, current configuration vẫn phù hợp.

Nếu sản phẩm cần diễn giải ví dụ:

```text
churn_probability = 0.80
```

thành “80% khả năng churn”, calibration nên được đánh giá kỹ hơn ở iteration sau.

---

## 12. Error Analysis

Threshold đã khóa cho phép phân biệt:

```text
true_positive
true_negative
false_positive
false_negative
```

Validation:

```text
FP = 1,879
FN = 1,587
```

Test:

```text
FP = 1,906
FN = 1,476
```

Final Test không cho thấy sự gia tăng bất thường về error counts so với Validation.

Notebook hiện chưa document đầy đủ feature-level error analysis để xác định các pattern cụ thể giữa:

```text
True Positive vs False Negative
True Negative vs False Positive
```

Do đó không đưa ra kết luận feature-level nào trong tài liệu này.

---

## 13. Feature Importance

Evaluation pipeline đã hỗ trợ lưu LightGBM gain importance tại:

```text
models/feature_importance.parquet
```

Tuy nhiên notebook được sử dụng để viết tài liệu này chưa chứa bảng Top Feature Importance thực tế.

Vì vậy tài liệu không suy đoán top features.

Feature importance cần được diễn giải như predictive importance, không phải causal effect.

---

## 14. Automated Tests

QT5 evaluation tests được thêm vào:

```text
tests/test_evaluation.py
```

Toàn bộ project test suite:

```text
18 passed in 1.31s
```

Các QT5 tests xác nhận:

- Probability metrics hoạt động với perfect ranking.
- Threshold metrics tính đúng TP/TN/FP/FN.
- Threshold table được tạo hợp lệ.
- Max-F1 selection hợp lệ.
- Minimum Recall strategy hoạt động đúng.
- Invalid Recall constraint raise error.
- Error type assignment hoạt động đúng.

Các tests từ QT3/QT4 vẫn tiếp tục pass.

---

## 15. Test-set Protection

QT5 sử dụng quy trình:

```text
Validation evaluation
        ↓
Threshold analysis
        ↓
Final config lock
        ↓
Test opened once
        ↓
Final evaluation
```

Config trước Final Test:

```text
locked = true
```

Sau khi Final Test được thực hiện, Test results không được dùng để:

- Train lại model.
- Thay hyperparameters.
- Thay preprocessing.
- Thay threshold.
- Chọn model khác.

Final Test được xem là kết quả cuối cùng của model-development cycle.

---

## 16. Saved QT5 Artifacts

Các artifact chính:

```text
models/
├── evaluation_results.json
├── final_model_config.json
├── threshold_analysis.parquet
└── feature_importance.parquet
```

Predictions:

```text
data/processed/
├── validation_predictions.parquet
└── test_predictions.parquet
```

Notebook:

```text
notebooks/04_model_evaluation.ipynb
```

---

## 17. Final Model Decision

Final selected model:

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

Primary ranking metric:

```text
PR-AUC
```

Final Test:

```text
PR-AUC    = 0.9361
ROC-AUC   = 0.9903
Precision = 0.8591
Recall    = 0.8873
F1        = 0.8730
```

Model selection and threshold selection are therefore considered complete.

---

## 18. QT5 Completion Status

### Core evaluation pipeline

```text
Validation evaluation        ✓
PR-AUC / ROC-AUC             ✓
Threshold analysis           ✓
Threshold selection          ✓
Model configuration lock     ✓
Final Test                   ✓
Validation/Test comparison   ✓
Automated tests              ✓
```

### Documentation gaps

The current notebook does not yet fully record:

```text
Calibration Curve interpretation
Feature-level error analysis
Top feature-importance ranking
```

These are analysis/documentation gaps rather than blockers for the locked final model.

They should be completed before presenting the notebook as a fully documented model-evaluation report.

---

## 19. Handoff to QT6

QT6 should consume only frozen artifacts:

```text
preprocessor.joblib
        ↓
lightgbm.joblib
        ↓
threshold = 0.8411917297941325
        ↓
prediction / risk output
```

QT6 will focus on:

- Inference pipeline.
- Model artifact loading.
- FastAPI.
- Input validation.
- Prediction endpoint.
- Health endpoint.
- Docker.
- API tests.
- Deployment packaging.
