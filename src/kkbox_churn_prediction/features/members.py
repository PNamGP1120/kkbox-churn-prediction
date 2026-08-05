from datetime import date
from pathlib import Path

import duckdb


def build_member_features(
    input_path: Path,
    output_path: Path,
    cutoff_date: date,
) -> None:
    cutoff = cutoff_date.isoformat()

    connection = duckdb.connect()

    connection.execute(
        f"""
        COPY (
            WITH members AS (
                SELECT
                    msno,

                    city,

                    CASE
                        WHEN bd BETWEEN 1 AND 100
                        THEN bd
                        ELSE NULL
                    END AS age,

                    COALESCE(
                        CAST(gender AS VARCHAR),
                        'unknown'
                    ) AS gender,

                    registered_via,

                    CAST(
                        TRY_STRPTIME(
                            CAST(
                                registration_init_time
                                AS VARCHAR
                            ),
                            '%Y%m%d'
                        )
                        AS DATE
                    ) AS registration_date

                FROM read_csv_auto(
                    '{input_path.as_posix()}'
                )
            )

            SELECT
                msno,
                city,
                age,
                gender,
                registered_via,

                CASE
                    WHEN registration_date < DATE '{cutoff}'
                    THEN DATE_DIFF(
                        'day',
                        registration_date,
                        DATE '{cutoff}'
                    )
                    ELSE NULL
                END AS account_age_days

            FROM members
        )
        TO '{output_path.as_posix()}'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        )
        """
    )

    connection.close()