from datetime import date
from pathlib import Path

import duckdb


def build_transaction_features(
    input_path: Path,
    output_path: Path,
    cutoff_date: date,
) -> None:
    cutoff = cutoff_date.isoformat()

    connection = duckdb.connect()

    connection.execute(
        f"""
        COPY (
            WITH transactions AS (
                SELECT
                    msno,

                    payment_method_id,

                    CASE
                        WHEN payment_plan_days > 0
                        THEN payment_plan_days
                        ELSE NULL
                    END AS payment_plan_days,

                    plan_list_price,
                    actual_amount_paid,

                    plan_list_price
                    - actual_amount_paid
                    AS discount_amount,

                    is_auto_renew,
                    is_cancel,

                    CAST(
                        TRY_STRPTIME(
                            CAST(
                                transaction_date
                                AS VARCHAR
                            ),
                            '%Y%m%d'
                        )
                        AS DATE
                    ) AS transaction_date

                FROM read_csv_auto(
                    '{input_path.as_posix()}'
                )
            ),

            valid_transactions AS (
                SELECT *
                FROM transactions
                WHERE transaction_date < DATE '{cutoff}'
            )

            SELECT
                msno,

                COUNT(*) AS transaction_count,

                SUM(actual_amount_paid)
                    AS total_paid,

                AVG(actual_amount_paid)
                    AS avg_paid,

                MIN(actual_amount_paid)
                    AS min_paid,

                MAX(actual_amount_paid)
                    AS max_paid,

                AVG(plan_list_price)
                    AS avg_plan_price,

                AVG(payment_plan_days)
                    AS avg_plan_days,

                AVG(discount_amount)
                    AS avg_discount,

                AVG(is_auto_renew)
                    AS auto_renew_rate,

                ARG_MAX(
                    is_auto_renew,
                    transaction_date
                ) AS last_auto_renew,

                SUM(is_cancel)
                    AS cancel_count,

                AVG(is_cancel)
                    AS cancel_rate,

                ARG_MAX(
                    is_cancel,
                    transaction_date
                ) AS last_is_cancel,

                MAX(transaction_date)
                    AS last_transaction_date,

                DATE_DIFF(
                    'day',
                    MAX(transaction_date),
                    DATE '{cutoff}'
                ) AS days_since_last_transaction

            FROM valid_transactions

            GROUP BY msno
        )
        TO '{output_path.as_posix()}'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        )
        """
    )

    connection.close()