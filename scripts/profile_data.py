from pathlib import Path

import duckdb


DATA_DIR = Path("data/raw")

DATASETS = {
    "train": DATA_DIR / "train_v2.csv",
    "members": DATA_DIR / "members_v3.csv",
    "transactions": DATA_DIR / "transactions_v2.csv",
    "user_logs": DATA_DIR / "user_logs_v2.csv",
}


def profile_dataset(name: str, path: Path) -> None:
    print("=" * 70)
    print(f"Dataset: {name}")
    print(f"Path: {path}")

    result = duckdb.sql(
        f"""
        SELECT
            COUNT(*) AS row_count,
            COUNT(DISTINCT msno) AS unique_users
        FROM read_csv_auto('{path}')
        """
    ).fetchone()

    row_count, unique_users = result

    print(f"Rows: {row_count:,}")
    print(f"Unique users: {unique_users:,}")
    print()


def main() -> None:
    for name, path in DATASETS.items():
        profile_dataset(name, path)


if __name__ == "__main__":
    main()