# KKBOX Customer Churn Prediction

End-to-end machine learning project for predicting customer churn on the KKBOX subscription platform.

The project covers the full ML lifecycle:

```text
Problem Definition
      ↓
Data Understanding
      ↓
Feature Engineering
      ↓
Model Development
      ↓
Model Evaluation
      ↓
Deployment
      ↓
Monitoring & Maintenance
```

Current status:

```text
QT1 — Problem Definition               ✓
QT2 — Data Understanding               ✓
QT3 — Feature Engineering              ✓
QT4 — Model Development & Training     ✓
QT5 — Model Evaluation & Selection     ✓
QT6 — Deployment                       ✓
QT7 — Monitoring & Maintenance         next
```

---

## 1. Problem

KKBOX is a subscription-based music streaming platform. The objective is to predict whether a user will churn.

```text
Target: is_churn
Prediction unit: 1 msno = 1 user = 1 prediction
Primary metric: PR-AUC
```

PR-AUC is used as the primary metric because the target is imbalanced.

---

## 2. Dataset

Raw datasets:

```text
data/raw/
├── members_v3.csv
├── train_v2.csv
├── transactions_v2.csv
└── user_logs_v2.csv
```

Target dataset:

```text
970,960 users
Non-churn ≈ 91.01%
Churn     ≈ 8.99%
```

---

## 3. Feature Engineering

Final processed dataset:

```text
970,960 rows
42 columns
```

Structure:

```text
40 model features
+ msno
+ is_churn
```

Feature groups:

```text
Member features       6
Transaction features 15
User-log features    19
```

Leakage prevention cutoff:

```text
2017-04-01
```

---

## 4. Model Development

Split:

```text
Train      70% = 679,672
Validation 15% = 145,644
Test       15% = 145,644
```

Models evaluated:

```text
Dummy Classifier
Logistic Regression
Random Forest
LightGBM
```

Selected model:

```text
LightGBM
```

Validation:

```text
PR-AUC  = 0.9345
ROC-AUC = 0.9898
```

---

## 5. Final Model

```text
Model: LightGBM
Threshold strategy: max_validation_f1
Threshold: 0.8411917297941325
Locked: true
```

---

## 6. Final Test Results

```text
PR-AUC      = 0.9361
ROC-AUC     = 0.9903
Brier Score = 0.0319
Precision   = 0.8591
Recall      = 0.8873
F1          = 0.8730
```

Confusion matrix:

```text
TN = 130,639
FP =   1,906
FN =   1,476
TP =  11,623
```

---

## 7. Project Structure

```text
kkbox-churn-prediction/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── docs/
│   ├── problem_definition.md
│   ├── data_dictionary.md
│   ├── stage1_findings.md
│   ├── feature_dictionary.md
│   ├── stage2_findings.md
│   ├── model_training.md
│   ├── stage3_findings.md
│   ├── model_evaluation.md
│   ├── stage4_findings.md
│   ├── deployment.md
│   └── stage5_findings.md
├── models/
│   ├── preprocessor.joblib
│   ├── lightgbm.joblib
│   └── final_model_config.json
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
├── scripts/
│   ├── profile_data.py
│   ├── build_features.py
│   ├── train_models.py
│   ├── evaluate_model.py
│   └── create_sample_request.py
├── src/
│   └── kkbox_churn_prediction/
│       ├── config.py
│       ├── data/
│       ├── features/
│       ├── modeling/
│       └── api/
│           ├── __init__.py
│           ├── schemas.py
│           ├── service.py
│           └── main.py
├── tests/
│   ├── test_processed_dataset.py
│   ├── test_modeling.py
│   ├── test_evaluation.py
│   ├── test_inference.py
│   └── test_api.py
├── Dockerfile
├── .dockerignore
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## 8. Main Stack

```text
Python
uv
Pandas
NumPy
scikit-learn
LightGBM
FastAPI
Uvicorn
Pydantic
Pytest
Docker
```

---

## 9. Install

```bash
uv sync
```

---

## 10. Run Tests

```bash
uv run pytest -v
```

Verified result:

```text
33 passed
```

---

## 11. Run API Locally

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

## 12. API Endpoints

```text
GET  /health
GET  /ready
GET  /metadata
POST /predict
POST /predict/batch
```

### Health

```bash
curl http://127.0.0.1:8000/health
```

### Ready

```bash
curl http://127.0.0.1:8000/ready
```

### Metadata

```bash
curl http://127.0.0.1:8000/metadata
```

Verified metadata includes:

```text
model = lightgbm
threshold = 0.8411917297941325
raw_feature_count = 40
locked = true
```

---

## 13. Prediction API

The API expects the same 40 engineered features used during training.

```bash
curl -s \
  -X POST \
  http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  --data @sample_request.json
```

Verified sample response:

```text
churn_score = 0.0008858738399855358
prediction = 0
threshold = 0.8411917297941325
```

---

## 14. Batch Prediction

```bash
curl -s \
  -X POST \
  http://127.0.0.1:8000/predict/batch \
  -H "Content-Type: application/json" \
  --data @sample_batch_request.json
```

---

## 15. Docker

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

Check container:

```bash
docker ps
```

Verified status:

```text
healthy
```

---

## 16. Deployment Consistency

Verified end-to-end:

```text
Offline model
      =
Local FastAPI
      =
Docker FastAPI
```

for the tested Validation sample.

---

## 17. Current Limitations

### Engineered-feature input
The API currently expects 40 engineered features rather than raw member, transaction, and user-log events.

### Calibration
Current config:

```text
calibration_strategy = none
```

The API exposes the output as `churn_score`.

### Docker size
Current image size is approximately:

```text
3.79 GB
```

Future work can reduce runtime dependencies and use a smaller production build.

### Monitoring
Monitoring and maintenance are planned for QT7.

---

## 18. Next Stage

```text
QT7 — Monitoring & Maintenance
```

Planned scope:

```text
API availability
latency
error rate
prediction drift
feature drift
data quality
model performance
model versioning
retraining triggers
alerts
```
