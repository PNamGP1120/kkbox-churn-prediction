from datetime import date
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

TRAIN_PATH = RAW_DIR / "train_v2.csv"
MEMBERS_PATH = RAW_DIR / "members_v3.csv"
TRANSACTIONS_PATH = RAW_DIR / "transactions_v2.csv"
USER_LOGS_PATH = RAW_DIR / "user_logs_v2.csv"

MEMBER_FEATURES_PATH = INTERIM_DIR / "members_features.parquet"
TRANSACTION_FEATURES_PATH = INTERIM_DIR / "transactions_features.parquet"
USER_LOG_FEATURES_PATH = INTERIM_DIR / "user_logs_features.parquet"

TRAIN_FEATURES_PATH = PROCESSED_DIR / "train_features.parquet"


CUTOFF_DATE = date(2017, 4, 1)

MIN_VALID_AGE = 1
MAX_VALID_AGE = 100

MAX_DAILY_LISTENING_SECS = 86_400

MODELS_DIR = PROJECT_ROOT / "models"

MODEL_SPLITS_PATH = PROCESSED_DIR / "model_splits.parquet"

PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"

FEATURE_NAMES_PATH = MODELS_DIR / "preprocessed_feature_names.json"

DUMMY_MODEL_PATH = MODELS_DIR / "dummy_classifier.joblib"

LOGISTIC_MODEL_PATH = MODELS_DIR / "logistic_regression.joblib"

RANDOM_FOREST_MODEL_PATH = MODELS_DIR / "random_forest.joblib"

LIGHTGBM_MODEL_PATH = MODELS_DIR / "lightgbm.joblib"

TRAINING_RESULTS_PATH = MODELS_DIR / "validation_results.json"

TRAINING_METADATA_PATH = MODELS_DIR / "training_metadata.json"

RANDOM_STATE = 42

EVALUATION_RESULTS_PATH = (
    MODELS_DIR
    / "evaluation_results.json"
)

THRESHOLD_ANALYSIS_PATH = (
    MODELS_DIR
    / "threshold_analysis.parquet"
)

FINAL_MODEL_CONFIG_PATH = (
    MODELS_DIR
    / "final_model_config.json"
)

VALIDATION_PREDICTIONS_PATH = (
    PROCESSED_DIR
    / "validation_predictions.parquet"
)

TEST_PREDICTIONS_PATH = (
    PROCESSED_DIR
    / "test_predictions.parquet"
)

FEATURE_IMPORTANCE_PATH = (
    MODELS_DIR
    / "feature_importance.parquet"
)