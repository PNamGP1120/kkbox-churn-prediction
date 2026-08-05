from pathlib import Path

import duckdb


TRAIN_COLUMNS = {
    "msno",
    "is_churn",
}

MEMBER_COLUMNS = {
    "msno",
    "city",
    "bd",
    "gender",
    "registered_via",
    "registration_init_time",
}

TRANSACTION_COLUMNS = {
    "msno",
    "payment_method_id",
    "payment_plan_days",
    "plan_list_price",
    "actual_amount_paid",
    "is_auto_renew",
    "transaction_date",
    "membership_expire_date",
    "is_cancel",
}

USER_LOG_COLUMNS = {
    "msno",
    "date",
    "num_25",
    "num_50",
    "num_75",
    "num_985",
    "num_100",
    "num_unq",
    "total_secs",
}


def get_columns(path: Path) -> set[str]:
    connection = duckdb.connect()

    result = connection.sql(
        f"""
        DESCRIBE
        SELECT *
        FROM read_csv_auto('{path.as_posix()}')
        """
    ).df()

    connection.close()

    return set(result["column_name"])


def validate_required_columns(
    path: Path,
    required_columns: set[str],
) -> None:
    columns = get_columns(path)

    missing = required_columns - columns

    if missing:
        raise ValueError(
            f"{path.name} missing columns: {sorted(missing)}"
        )


def validate_train(path: Path) -> None:
    validate_required_columns(path, TRAIN_COLUMNS)

    connection = duckdb.connect()

    result = connection.sql(
        f"""
        SELECT
            COUNT(*) AS rows,
            COUNT(DISTINCT msno) AS users,
            SUM(
                CASE WHEN msno IS NULL
                THEN 1 ELSE 0 END
            ) AS missing_msno,
            SUM(
                CASE WHEN is_churn NOT IN (0, 1)
                THEN 1 ELSE 0 END
            ) AS invalid_target
        FROM read_csv_auto('{path.as_posix()}')
        """
    ).df().iloc[0]

    connection.close()

    if result["rows"] != result["users"]:
        raise ValueError("train_v2.csv contains duplicate users")

    if result["missing_msno"] != 0:
        raise ValueError("train_v2.csv contains missing msno")

    if result["invalid_target"] != 0:
        raise ValueError("is_churn must contain only 0 or 1")


def validate_all(
    train_path: Path,
    members_path: Path,
    transactions_path: Path,
    logs_path: Path,
) -> None:
    validate_train(train_path)

    validate_required_columns(
        members_path,
        MEMBER_COLUMNS,
    )

    validate_required_columns(
        transactions_path,
        TRANSACTION_COLUMNS,
    )

    validate_required_columns(
        logs_path,
        USER_LOG_COLUMNS,
    )

def validate_processed_dataset(
    path: Path,
    expected_rows: int,
) -> None:
    connection = duckdb.connect()

    result = connection.sql(
        f"""
        SELECT
            COUNT(*) AS rows,
            COUNT(DISTINCT msno)
                AS users,

            SUM(
                CASE
                    WHEN msno IS NULL
                    THEN 1 ELSE 0
                END
            ) AS missing_msno,

            SUM(
                CASE
                    WHEN is_churn IS NULL
                    THEN 1 ELSE 0
                END
            ) AS missing_target,

            SUM(
                CASE
                    WHEN is_churn NOT IN (0, 1)
                    THEN 1 ELSE 0
                END
            ) AS invalid_target

        FROM read_parquet(
            '{path.as_posix()}'
        )
        """
    ).df().iloc[0]

    connection.close()

    if result["rows"] != expected_rows:
        raise ValueError(
            "Final row count does not match train"
        )

    if result["rows"] != result["users"]:
        raise ValueError(
            "Final dataset is not one row per user"
        )

    if result["missing_msno"] != 0:
        raise ValueError(
            "Final dataset contains missing msno"
        )

    if result["missing_target"] != 0:
        raise ValueError(
            "Final dataset contains missing target"
        )

    if result["invalid_target"] != 0:
        raise ValueError(
            "Final dataset contains invalid target"
        )