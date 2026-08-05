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