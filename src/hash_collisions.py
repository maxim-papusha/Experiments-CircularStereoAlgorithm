from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "chembl_smg.db"
CSV_PATH = ROOT / "data" / "chembl_smg_hash_collision_groups.csv"
TOP_GROUPS = 20

SUMMARY_SQL = """
WITH hash_groups AS (
    SELECT smg_hash, COUNT(*) AS group_size
    FROM smg_records
    GROUP BY smg_hash
),
collision_groups AS (
    SELECT smg_hash, group_size
    FROM hash_groups
    WHERE group_size > 1
)
SELECT
    (SELECT COUNT(*) FROM smg_records) AS total_molecules,
    (SELECT COUNT(*) FROM hash_groups) AS unique_hashes,
    (SELECT COUNT(*) FROM collision_groups) AS collision_hashes,
    COALESCE(
        (SELECT SUM(group_size) FROM collision_groups),
        0
    ) AS affected_molecules,
    COALESCE(
        (SELECT MAX(group_size) FROM collision_groups),
        0
    ) AS largest_group
"""

DISTRIBUTION_SQL = """
WITH collision_groups AS (
    SELECT COUNT(*) AS group_size
    FROM smg_records
    GROUP BY smg_hash
    HAVING COUNT(*) > 1
)
SELECT
    group_size,
    COUNT(*) AS n_groups,
    SUM(group_size) AS molecules
FROM collision_groups
GROUP BY group_size
ORDER BY group_size
"""

TOP_GROUPS_SQL = """
WITH collision_groups AS (
    SELECT smg_hash, COUNT(*) AS group_size
    FROM smg_records
    GROUP BY smg_hash
    HAVING COUNT(*) > 1
)
SELECT smg_hash, group_size
FROM collision_groups
ORDER BY group_size DESC, smg_hash
LIMIT ?
"""

COLLISION_ROWS_SQL = """
WITH collision_groups AS (
    SELECT smg_hash, COUNT(*) AS group_size
    FROM smg_records
    GROUP BY smg_hash
    HAVING COUNT(*) > 1
)
SELECT
    collision_groups.smg_hash,
    collision_groups.group_size,
    smg_records.chembl_id,
    smg_records.smiles,
    smg_records.inchi
FROM collision_groups
JOIN smg_records USING (smg_hash)
ORDER BY
    collision_groups.group_size DESC,
    collision_groups.smg_hash,
    smg_records.chembl_id
"""


def fetch_summary(
    connection: sqlite3.Connection,
) -> tuple[int, int, int, int, int]:
    row = connection.execute(SUMMARY_SQL).fetchone()
    if row is None:
        raise RuntimeError("Could not compute collision summary.")
    total, unique_hashes, collision_hashes, affected, largest_group = row
    return (
        int(total),
        int(unique_hashes),
        int(collision_hashes),
        int(affected),
        int(largest_group),
    )


def fetch_distribution(
    connection: sqlite3.Connection,
) -> list[tuple[int, int, int]]:
    return [
        (int(group_size), int(n_groups), int(molecules))
        for group_size, n_groups, molecules in connection.execute(
            DISTRIBUTION_SQL
        )
    ]


def fetch_top_groups(
    connection: sqlite3.Connection,
) -> list[tuple[int, int]]:
    return [
        (int(smg_hash), int(group_size))
        for smg_hash, group_size in connection.execute(
            TOP_GROUPS_SQL, (TOP_GROUPS,)
        )
    ]


def write_collision_csv(
    connection: sqlite3.Connection,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["collision_group", "chembl_id", "smiles", "inchi"])

        current_hash: int | None = None
        current_group = 0
        for (
            smg_hash,
            _group_size,
            chembl_id,
            smiles,
            inchi,
        ) in connection.execute(COLLISION_ROWS_SQL):
            if smg_hash != current_hash:
                current_hash = int(smg_hash)
                current_group += 1
            writer.writerow([current_group, chembl_id, smiles, inchi])


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    with sqlite3.connect(DB_PATH) as connection:
        total, unique_hashes, collision_hashes, affected, largest_group = (
            fetch_summary(connection)
        )
        distribution = fetch_distribution(connection)
        top_groups = fetch_top_groups(connection)
        write_collision_csv(connection, CSV_PATH)

    percent = (100.0 * affected / total) if total else 0.0

    print(f"Database:               {DB_PATH}")
    print(f"Total molecules:        {total:,}")
    print(f"Unique SMG hashes:      {unique_hashes:,}")
    print(f"Collision hash values:  {collision_hashes:,}")
    print(f"Affected molecules:     {affected:,}")
    print(f"Affected fraction:      {percent:.4f}%")
    print(f"Largest collision size: {largest_group:,}")

    print("\nCollision group size distribution:")
    if distribution:
        for group_size, n_groups, molecules in distribution:
            print(
                "  "
                f"size={group_size:<3} "
                f"groups={n_groups:>8,} "
                f"molecules={molecules:>8,}"
            )
    else:
        print("  none")

    print(f"\nTop {TOP_GROUPS} collision hashes:")
    if top_groups:
        for smg_hash, group_size in top_groups:
            print(f"  hash={smg_hash}  size={group_size}")
    else:
        print("  none")

    print(f"\nWrote collision CSV:     {CSV_PATH}")


if __name__ == "__main__":
    main()
