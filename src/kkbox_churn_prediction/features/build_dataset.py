from pathlib import Path

import duckdb


def build_training_dataset(
    train_path: Path,
    members_path: Path,
    transactions_path: Path,
    logs_path: Path,
    output_path: Path,
) -> None:
    connection = duckdb.connect()

    connection.execute(
        f"""
        COPY (
            SELECT
                t.msno,
                t.is_churn,

                CASE
                    WHEN m.msno IS NULL
                    THEN 0 ELSE 1
                END AS has_member_data,

                CASE
                    WHEN tx.msno IS NULL
                    THEN 0 ELSE 1
                END AS has_transaction_data,

                CASE
                    WHEN l.msno IS NULL
                    THEN 0 ELSE 1
                END AS has_log_data,

                m.city,
                m.age,
                m.gender,
                m.registered_via,
                m.account_age_days,

                COALESCE(
                    tx.transaction_count,
                    0
                ) AS transaction_count,

                tx.total_paid,
                tx.avg_paid,
                tx.min_paid,
                tx.max_paid,
                tx.avg_plan_price,
                tx.avg_plan_days,
                tx.avg_discount,
                tx.auto_renew_rate,
                tx.last_auto_renew,

                COALESCE(
                    tx.cancel_count,
                    0
                ) AS cancel_count,

                tx.cancel_rate,
                tx.last_is_cancel,
                tx.days_since_last_transaction,

                COALESCE(
                    l.active_days,
                    0
                ) AS active_days,

                COALESCE(
                    l.total_listening_secs,
                    0
                ) AS total_listening_secs,

                l.avg_listening_secs_per_day,
                l.median_listening_secs_per_day,
                l.avg_unique_songs,

                COALESCE(
                    l.total_unique_song_events,
                    0
                ) AS total_unique_song_events,

                COALESCE(
                    l.total_num_25,
                    0
                ) AS total_num_25,

                COALESCE(
                    l.total_num_50,
                    0
                ) AS total_num_50,

                COALESCE(
                    l.total_num_75,
                    0
                ) AS total_num_75,

                COALESCE(
                    l.total_num_985,
                    0
                ) AS total_num_985,

                COALESCE(
                    l.total_num_100,
                    0
                ) AS total_num_100,

                l.completion_rate,
                l.days_since_last_activity,

                COALESCE(
                    l.active_days_7d,
                    0
                ) AS active_days_7d,

                COALESCE(
                    l.active_days_30d,
                    0
                ) AS active_days_30d,

                COALESCE(
                    l.listening_secs_7d,
                    0
                ) AS listening_secs_7d,

                COALESCE(
                    l.listening_secs_30d,
                    0
                ) AS listening_secs_30d,

                COALESCE(
                    l.extreme_listening_day_count,
                    0
                ) AS extreme_listening_day_count

            FROM read_csv_auto(
                '{train_path.as_posix()}'
            ) AS t

            LEFT JOIN read_parquet(
                '{members_path.as_posix()}'
            ) AS m
                ON t.msno = m.msno

            LEFT JOIN read_parquet(
                '{transactions_path.as_posix()}'
            ) AS tx
                ON t.msno = tx.msno

            LEFT JOIN read_parquet(
                '{logs_path.as_posix()}'
            ) AS l
                ON t.msno = l.msno
        )
        TO '{output_path.as_posix()}'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        )
        """
    )

    connection.close()