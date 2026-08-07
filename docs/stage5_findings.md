# Stage 5 Findings — Deployment

## KKBOX Customer Churn Prediction Platform

This document summarizes the main findings from **QT6 — Deployment**.

---

## 1. Final Deployment Candidate

```text
Model: LightGBM
Threshold: 0.8411917297941325
Threshold strategy: max_validation_f1
Locked: true
```

QT6 did not retrain the model or change the threshold.

---

## 2. Final Model Performance

Final Test performance inherited from QT5:

```text
PR-AUC    = 0.9361
ROC-AUC   = 0.9903
Precision = 0.8591
Recall    = 0.8873
F1        = 0.8730
Brier     = 0.0319
```

---

## 3. API Findings

The model was successfully exposed through FastAPI.

Implemented endpoints:

```text
GET  /health
GET  /ready
GET  /metadata
POST /predict
POST /predict/batch
```

The frozen artifacts are loaded once during application startup.

---

## 4. Feature Contract Findings

The production inference API expects exactly:

```text
40 engineered features
```

Feature names and order are recovered from:

```text
preprocessor.feature_names_in_
```

The service rejects missing, unexpected, and invalid feature values rather than silently altering the request.

---

## 5. Automated Test Findings

```text
33 passed
```

Tests cover data invariants, modeling, evaluation, inference behavior, feature-contract validation, API endpoints, and batch inference.

---

## 6. Local API Findings

Verified:

```text
/health   → 200 OK
/ready    → 200 OK
/metadata → 200 OK
```

Metadata:

```text
model = lightgbm
threshold = 0.8411917297941325
threshold_strategy = max_validation_f1
calibration_strategy = none
primary_metric = pr_auc
raw_feature_count = 40
locked = true
```

---

## 7. Real Sample Prediction Finding

Validation sample:

```text
msno = MNRUD2pAtKpbaPsD2bJqhwKQsIt06ZkosKVWXFZI2TQ=
actual is_churn = 0
```

API prediction:

```text
churn_score = 0.0008858738399855358
prediction = 0
```

Offline result:

```text
churn_probability ≈ 0.000886
prediction = 0
error_type = true_negative
```

The deployed API reproduces the offline result.

---

## 8. Docker Findings

```text
Image: kkbox-churn-api:1.0.0
Build: successful
Container startup: successful
Container status: healthy
```

The container successfully loads:

```text
preprocessor.joblib
lightgbm.joblib
final_model_config.json
```

---

## 9. Docker Endpoint Findings

```text
GET /health          → 200 OK
GET /ready           → 200 OK
GET /metadata        → 200 OK
POST /predict        → successful
POST /predict/batch  → successful
```

---

## 10. Local vs Docker Consistency

The same feature vector was evaluated through:

```text
Offline model
Local FastAPI
Docker FastAPI
```

All three produced:

```text
churn_score = 0.0008858738399855358
prediction = 0
```

Result:

```text
Offline = Local API = Docker API
```

This confirms end-to-end inference consistency for the verified sample.

---

## 11. Batch Inference Finding

Batch inference was verified using two records with identical features.

Both produced the same:

```text
churn_score
prediction
threshold
```

while preserving independent `msno` values.

---

## 12. Reliability Finding

The Docker health check reports:

```text
healthy
```

The service separately exposes liveness and readiness endpoints, providing a good foundation for later production monitoring.

---

## 13. Docker Image Finding

Current Docker image size:

```text
approximately 3.79 GB
```

The image is fully functional but can be reduced in a later optimization pass using production-only dependencies and multi-stage builds.

---

## 14. Calibration Limitation

Current configuration:

```text
calibration_strategy = none
```

The API therefore uses the name `churn_score` rather than claiming a perfectly calibrated real-world churn probability.

---

## 15. Feature Pipeline Limitation

The current API starts from:

```text
40 engineered features
```

rather than raw member, transaction, and user-log events.

A future production architecture could add an online/offline feature pipeline or feature store.

---

## 16. Security Limitation

Current scope is suitable for:

```text
local development
portfolio demonstration
controlled internal environments
```

Public deployment would additionally require authentication, authorization, TLS, rate limiting, and secret management.

---

## 17. QT6 Final Status

```text
FastAPI application              ✓
Frozen model loading             ✓
Locked threshold                 ✓
Feature contract                 ✓
Health/readiness                 ✓
Metadata                         ✓
Single inference                 ✓
Batch inference                  ✓
33 tests                         ✓
Real sample validation           ✓
Offline/local consistency        ✓
Docker build                     ✓
Docker startup                   ✓
Docker health                    ✓
Docker inference                 ✓
Local/Docker consistency         ✓
```

QT6 is complete.

---

## 18. Next Stage

```text
QT7 — Monitoring & Maintenance
```

Recommended QT7 scope:

```text
request count
latency
API error rate
health status
prediction distribution
feature missing rate
feature drift
prediction drift
data drift
model performance
model versions
retraining triggers
alerts
```
