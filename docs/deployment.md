# Deployment

## KKBOX Customer Churn Prediction Platform

Tài liệu này mô tả **QT6 — Deployment** của dự án KKBOX Customer Churn Prediction.

Mục tiêu của QT6 là đưa model đã được lựa chọn và khóa ở QT5 vào một inference service có thể chạy độc lập bằng FastAPI và Docker.

---

## 1. Deployment Objective

QT6 không train lại model và không thay đổi threshold.

Frozen model configuration:

```text
Model: LightGBM
Threshold strategy: max_validation_f1
Threshold: 0.8411917297941325
Primary metric: PR-AUC
Calibration strategy: none
Locked: true
```

Final Test performance từ QT5:

```text
PR-AUC    = 0.9361
ROC-AUC   = 0.9903
Precision = 0.8591
Recall    = 0.8873
F1        = 0.8730
```

---

## 2. Deployment Architecture

```text
Client
  ↓
HTTP / JSON
  ↓
FastAPI
  ↓
Pydantic validation
  ↓
40 engineered features
  ↓
Frozen preprocessor
  ↓
Frozen LightGBM model
  ↓
churn_score
  ↓
threshold = 0.8411917297941325
  ↓
prediction = 0 / 1
  ↓
JSON response
```

Current API operates on the 40 engineered features created in QT3. It does not rebuild features directly from raw KKBOX transaction/log/member events at request time.

---

## 3. Frozen Artifacts

QT6 uses only frozen artifacts produced by QT4 and QT5:

```text
models/
├── preprocessor.joblib
├── lightgbm.joblib
└── final_model_config.json
```

The service verifies:

```text
locked = true
```

before allowing inference.

---

## 4. API Package Structure

```text
src/
└── kkbox_churn_prediction/
    └── api/
        ├── __init__.py
        ├── schemas.py
        ├── service.py
        └── main.py
```

### `schemas.py`
Defines request/response contracts using Pydantic.

### `service.py`
Responsible for loading artifacts, validating the 40-feature contract, restoring feature order, applying preprocessing, running LightGBM inference, and applying the frozen threshold.

### `main.py`
Contains FastAPI startup and endpoints. The model is loaded once during application startup.

---

## 5. Feature Contract

The deployed model expects exactly:

```text
40 engineered features
```

Feature names and order are recovered from:

```python
preprocessor.feature_names_in_
```

The service rejects:

```text
missing features
unexpected features
invalid numeric values
```

This prevents silent feature mismatch and reduces training-serving skew.

---

## 6. API Endpoints

### `GET /health`

Response:

```json
{"status":"ok"}
```

### `GET /ready`

Response:

```json
{"status":"ready"}
```

### `GET /metadata`

Verified response:

```json
{
  "model": "lightgbm",
  "threshold": 0.8411917297941325,
  "threshold_strategy": "max_validation_f1",
  "calibration_strategy": "none",
  "primary_metric": "pr_auc",
  "raw_feature_count": 40,
  "locked": true
}
```

### `POST /predict`

Request:

```json
{
  "msno": "user_id",
  "features": {
    "...": "40 engineered features"
  }
}
```

Response:

```json
{
  "msno": "user_id",
  "model": "lightgbm",
  "churn_score": 0.91,
  "threshold": 0.8411917297941325,
  "prediction": 1,
  "threshold_strategy": "max_validation_f1"
}
```

Prediction rule:

```text
prediction = 1 if churn_score >= 0.8411917297941325
prediction = 0 otherwise
```

Because the current config uses `calibration_strategy = none`, the API exposes the model output as `churn_score` rather than claiming it is a perfectly calibrated real-world probability.

### `POST /predict/batch`

Performs vectorized batch inference over multiple feature records.

---

## 7. Error Handling

```text
400  Feature contract / feature-value errors
422  Pydantic request validation errors
503  Model service not ready
500  Unexpected inference failure
```

---

## 8. Local Development

Install dependencies:

```bash
uv sync
```

Run tests:

```bash
uv run pytest -v
```

Verified result:

```text
33 passed
```

Run API:

```bash
uv run uvicorn \
  kkbox_churn_prediction.api.main:app \
  --host 127.0.0.1 \
  --port 8000
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

---

## 9. Local Smoke Tests

```bash
curl -i http://127.0.0.1:8000/health
curl -i http://127.0.0.1:8000/ready
curl -s http://127.0.0.1:8000/metadata
```

Verified:

```text
/health   → 200 OK
/ready    → 200 OK
/metadata → correct frozen model configuration
```

---

## 10. Real Validation Sample

Sample user:

```text
MNRUD2pAtKpbaPsD2bJqhwKQsIt06ZkosKVWXFZI2TQ=
```

Actual target:

```text
is_churn = 0
```

Local API result:

```text
churn_score = 0.0008858738399855358
prediction = 0
```

Offline validation result:

```text
churn_probability ≈ 0.000886
prediction = 0
error_type = true_negative
```

The API prediction matches the offline model prediction.

---

## 11. Automated Tests

Current suite:

```text
33 passed
```

QT6 tests cover:

```text
health endpoint
readiness endpoint
metadata endpoint
valid prediction
missing features
unexpected features
invalid numeric values
batch prediction
locked configuration
40-feature contract
deterministic inference
score range
threshold behavior
```

---

## 12. Docker Deployment

Build:

```bash
docker build \
  -t kkbox-churn-api:1.0.0 \
  .
```

Run:

```bash
docker run \
  --rm \
  -p 8000:8000 \
  --name kkbox-churn-api \
  kkbox-churn-api:1.0.0
```

Verified image:

```text
kkbox-churn-api:1.0.0
```

Verified startup:

```text
Model ready: lightgbm
threshold = 0.841192
Application startup complete
```

---

## 13. Docker Health Check

Verified container status:

```text
healthy
```

Verified endpoints:

```text
GET /health     → 200 OK
GET /ready      → 200 OK
GET /metadata   → 200 OK
```

---

## 14. Docker Prediction Verification

The same `sample_request.json` used locally was sent to the Dockerized API.

Docker result:

```text
churn_score = 0.0008858738399855358
prediction = 0
threshold = 0.8411917297941325
```

Batch prediction also succeeded.

---

## 15. End-to-End Consistency

Verified deployment property:

```text
Offline model
      =
Local FastAPI
      =
Docker FastAPI
```

For the verified Validation sample:

```text
score = 0.0008858738399855358
prediction = 0
```

This is the main deployment-validation result of QT6.

---

## 16. Docker Image Size

Current image size is approximately:

```text
3.79 GB
```

This is functional but larger than necessary for an inference service.

Future optimization can include:

```text
production-only dependency groups
separating training and runtime dependencies
multi-stage Docker builds
removing notebook/development packages from runtime
```

This is an optimization opportunity, not a blocker for QT6 completion.

---

## 17. Known Limitations

### Engineered-feature input
The API expects the final 40 engineered features and does not build them directly from raw member, transaction, and user-log events.

### Calibration
Current configuration uses:

```text
calibration_strategy = none
```

Therefore `churn_score` should primarily be treated as a model risk score.

### Security
The current API is suitable for local development, portfolio demonstration, and controlled environments. Public Internet deployment would require authentication, authorization, TLS, rate limiting, and secret management.

### Monitoring
QT6 does not yet implement drift, latency, error-rate, and model-performance monitoring. These belong to QT7.

---

## 18. QT6 Completion

```text
API implementation              ✓
Model artifact loading          ✓
Locked config validation        ✓
40-feature contract             ✓
Health endpoint                 ✓
Readiness endpoint              ✓
Metadata endpoint               ✓
Single prediction               ✓
Batch prediction                ✓
Automated tests                 ✓
Local smoke test                ✓
Real Validation sample          ✓
Offline/API consistency         ✓
Docker image build              ✓
Docker startup                  ✓
Docker health check             ✓
Docker prediction               ✓
Docker batch prediction         ✓
Local/Docker consistency        ✓
```

QT6 is complete.

---

## 19. Handoff to QT7

Next stage:

```text
QT7 — Monitoring & Maintenance
```

Recommended scope:

```text
API availability
latency
request volume
error rate
prediction distribution
feature missing rate
feature drift
data drift
model performance
model versioning
retraining triggers
alerts
```
