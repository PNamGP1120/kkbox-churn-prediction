from datetime import date
from pathlib import Path

import duckdb


def build_user_log_features(
    input_path: Path,
    output_path: Path,
    cutoff_date: date,
) -> None:
    cutoff = cutoff_date.isoformat()

    connection = duckdb.connect()

    connection.execute(
        f"""
        COPY (
            WITH logs AS (
                SELECT
                    msno,

                    CAST(
                        TRY_STRPTIME(
                            CAST(date AS VARCHAR),
                            '%Y%m%d'
                        )
                        AS DATE
                    ) AS activity_date,

                    num_25,
                    num_50,
                    num_75,
                    num_985,
                    num_100,
                    num_unq,

                    total_secs AS total_secs_raw,

                    CASE
                        WHEN total_secs < 0
                        THEN NULL

                        WHEN total_secs > 86400
                        THEN 86400

                        ELSE total_secs
                    END AS total_secs_clean,

                    CASE
                        WHEN total_secs > 86400
                        THEN 1
                        ELSE 0
                    END AS is_extreme_listening_day

                FROM read_csv_auto(
                    '{input_path.as_posix()}'
                )
            ),

            valid_logs AS (
                SELECT *
                FROM logs
                WHERE activity_date < DATE '{cutoff}'
            )

            SELECT
                msno,

                COUNT(DISTINCT activity_date)
                    AS active_days,

                SUM(total_secs_clean)
                    AS total_listening_secs,

                AVG(total_secs_clean)
                    AS avg_listening_secs_per_day,

                MEDIAN(total_secs_clean)
                    AS median_listening_secs_per_day,

                AVG(num_unq)
                    AS avg_unique_songs,

                SUM(num_unq)
                    AS total_unique_song_events,

                SUM(num_25)
                    AS total_num_25,

                SUM(num_50)
                    AS total_num_50,

                SUM(num_75)
                    AS total_num_75,

                SUM(num_985)
                    AS total_num_985,

                SUM(num_100)
                    AS total_num_100,

                CASE
                    WHEN (
                        SUM(num_25)
                        + SUM(num_50)
                        + SUM(num_75)
                        + SUM(num_985)
                        + SUM(num_100)
                    ) > 0

                    THEN
                        SUM(num_100) * 1.0
                        /
                        (
                            SUM(num_25)
                            + SUM(num_50)
                            + SUM(num_75)
                            + SUM(num_985)
                            + SUM(num_100)
                        )

                    ELSE NULL
                END AS completion_rate,

                MAX(activity_date)
                    AS last_active_date,

                DATE_DIFF(
                    'day',
                    MAX(activity_date),
                    DATE '{cutoff}'
                ) AS days_since_last_activity,

                SUM(
                    CASE
                        WHEN activity_date
                             >= DATE '{cutoff}'
                                - INTERVAL 7 DAY
                        THEN 1
                        ELSE 0
                    END
                ) AS active_days_7d,

                SUM(
                    CASE
                        WHEN activity_date
                             >= DATE '{cutoff}'
                                - INTERVAL 30 DAY
                        THEN 1
                        ELSE 0
                    END
                ) AS active_days_30d,

                SUM(
                    CASE
                        WHEN activity_date
                             >= DATE '{cutoff}'
                                - INTERVAL 7 DAY
                        THEN total_secs_clean
                        ELSE 0
                    END
                ) AS listening_secs_7d,

                SUM(
                    CASE
                        WHEN activity_date
                             >= DATE '{cutoff}'
                                - INTERVAL 30 DAY
                        THEN total_secs_clean
                        ELSE 0
                    END
                ) AS listening_secs_30d,

                SUM(is_extreme_listening_day)
                    AS extreme_listening_day_count

            FROM valid_logs

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