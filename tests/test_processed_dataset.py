import duckdb

from kkbox_churn_prediction.config import (
    TRAIN_FEATURES_PATH,
    TRANSACTION_FEATURES_PATH,
    USER_LOG_FEATURES_PATH,
)


def test_processed_dataset_exists() -> None:
    assert TRAIN_FEATURES_PATH.exists()


def test_one_row_per_user() -> None:
    result = duckdb.sql(
        f"""
        SELECT
            COUNT(*) AS rows,
            COUNT(DISTINCT msno) AS users
        FROM read_parquet(
            '{TRAIN_FEATURES_PATH.as_posix()}'
        )
        """
    ).df().iloc[0]

    assert result["rows"] == result["users"]


def test_expected_row_count() -> None:
    count = duckdb.sql(
        f"""
        SELECT COUNT(*) AS rows
        FROM read_parquet(
            '{TRAIN_FEATURES_PATH.as_posix()}'
        )
        """
    ).df().iloc[0]["rows"]

    assert count == 970_960


def test_target_has_no_missing() -> None:
    missing = duckdb.sql(
        f"""
        SELECT COUNT(*) AS n
        FROM read_parquet(
            '{TRAIN_FEATURES_PATH.as_posix()}'
        )
        WHERE is_churn IS NULL
        """
    ).df().iloc[0]["n"]

    assert missing == 0


def test_target_is_binary() -> None:
    invalid = duckdb.sql(
        f"""
        SELECT COUNT(*) AS n
        FROM read_parquet(
            '{TRAIN_FEATURES_PATH.as_posix()}'
        )
        WHERE is_churn NOT IN (0, 1)
        """
    ).df().iloc[0]["n"]

    assert invalid == 0


def test_transaction_dates_before_cutoff() -> None:
    invalid = duckdb.sql(
        f"""
        SELECT COUNT(*) AS n
        FROM read_parquet(
            '{TRANSACTION_FEATURES_PATH.as_posix()}'
        )
        WHERE last_transaction_date >= DATE '2017-04-01'
        """
    ).df().iloc[0]["n"]

    assert invalid == 0


def test_log_dates_before_cutoff() -> None:
    invalid = duckdb.sql(
        f"""
        SELECT COUNT(*) AS n
        FROM read_parquet(
            '{USER_LOG_FEATURES_PATH.as_posix()}'
        )
        WHERE last_active_date >= DATE '2017-04-01'
        """
    ).df().iloc[0]["n"]

    assert invalid == 0