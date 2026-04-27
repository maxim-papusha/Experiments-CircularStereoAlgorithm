"""Analyze color (ECFP hash) consistency of per-atom subgraphs in subgraph_smg.db.

For every (radius, color), deserializes all subgraph JSONs and checks how many
are structurally distinct.  Results go into ``color_analysis``.

Resumable: already-stored rows are skipped.
"""

from __future__ import annotations

import multiprocessing as mp
import sqlite3
from collections import defaultdict
from contextlib import closing
from pathlib import Path

from stereomolgraph import StereoMolGraph
from stereomolgraph.experimental import JSONHandler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "subgraph_smg.db"

RADII = (2, 3, 4)
SAMPLE_SIZE: int | None = (
    None  # None = use all entries (no false-negative risk)
)
WORKERS = max(1, (mp.cpu_count() or 1) - 1)
INSERT_BATCH = 500


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — module-level so they are picklable under "spawn"
# ─────────────────────────────────────────────────────────────────────────────


def _deserialize(s: str) -> StereoMolGraph:
    smg = JSONHandler.json_deserialize(s)
    if not isinstance(smg, StereoMolGraph):
        raise TypeError(f"Expected StereoMolGraph, got {type(smg).__name__}")
    smg.freeze()
    return smg


def _distinct(graphs: list) -> list:
    """Return one representative per structurally distinct graph."""
    hash_groups: dict[int, list] = defaultdict(list)
    for g in graphs:
        hash_groups[hash(g)].append(g)
    result: list = []
    for group in hash_groups.values():
        rep = group[0]
        if not any(rep == existing for existing in result):
            result.append(rep)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Worker function
# ─────────────────────────────────────────────────────────────────────────────


def analyze_color(
    task: tuple[str, int, int, int],
) -> tuple[int, int, int, int, int, int]:
    """
    (db_path, radius, color, n_entries) ->
        (radius, color, n_entries, n_loaded, n_distinct, is_collision)
    """
    db_path, radius, color, n_entries = task
    with closing(sqlite3.connect(db_path)) as con:
        if SAMPLE_SIZE is None:
            jsons = [
                r[0]
                for r in con.execute(
                    "SELECT subgraph_json FROM atom_fingerprints "
                    "WHERE radius = ? AND color = ?",
                    (radius, color),
                )
            ]
        else:
            jsons = [
                r[0]
                for r in con.execute(
                    "SELECT subgraph_json FROM atom_fingerprints "
                    "WHERE radius = ? AND color = ? LIMIT ?",
                    (radius, color, SAMPLE_SIZE),
                )
            ]
    if len(jsons) <= 1:
        return (radius, color, n_entries, len(jsons), 1, 0)
    smgs = [_deserialize(s) for s in jsons]
    n_distinct = len(_distinct(smgs))
    return (
        radius,
        color,
        n_entries,
        len(jsons),
        n_distinct,
        int(n_distinct > 1),
    )


# ─────────────────────────────────────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────────────────────────────────────


def init_analysis_tables(connection: sqlite3.Connection) -> None:
    connection.executescript("""
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = NORMAL;
        CREATE TABLE IF NOT EXISTS color_analysis (
            radius       INTEGER NOT NULL,
            color        INTEGER NOT NULL,
            n_entries    INTEGER NOT NULL,
            n_sampled    INTEGER NOT NULL,
            n_distinct   INTEGER NOT NULL,
            is_collision INTEGER NOT NULL,
            PRIMARY KEY (radius, color)
        );
        CREATE INDEX IF NOT EXISTS idx_color_analysis_radius_collision
        ON color_analysis (radius, is_collision);
    """)
    connection.commit()


def _done_colors(
    connection: sqlite3.Connection, table: str, radius: int
) -> set[int]:
    return {
        r[0]
        for r in connection.execute(
            f"SELECT color FROM {table} WHERE radius = ?", (radius,)
        )
    }


def _load_color_counts(
    connection: sqlite3.Connection, radius: int
) -> list[tuple[int, int]]:
    """Return (color, cnt) for all colors at *radius*.

    Uses the covering index (radius, color) — no table row fetches.
    """
    return connection.execute(
        "SELECT color, COUNT(*) FROM atom_fingerprints "
        "WHERE radius = ? GROUP BY color",
        (radius,),
    ).fetchall()


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: color consistency
# ─────────────────────────────────────────────────────────────────────────────


def run_analyze_phase(connection: sqlite3.Connection, workers: int) -> None:
    print("\nColor consistency analysis")
    print("=" * 60)

    for radius in RADII:
        # Covering-index GROUP BY scan — reads only (radius, color), no JSON.
        print(f"\n  r={radius}: running GROUP BY …", end="", flush=True)
        color_counts = _load_color_counts(connection, radius)
        done = _done_colors(connection, "color_analysis", radius)
        n_todo = len(color_counts) - len(done)
        print(
            f"\r  r={radius}: {len(color_counts):,} distinct colors | "
            f"{len(done):,} already done | {n_todo:,} to analyze"
        )

        if n_todo == 0:
            continue

        # Fast-path: a color with cnt==1 has exactly one subgraph — no JSON read
        # needed.  Colors with cnt>1 require JSON deserialization & comparison.
        singles: list[tuple] = []
        multi_colors: list[tuple[int, int]] = []
        for color, cnt in color_counts:
            if color in done:
                continue
            if cnt == 1:
                singles.append((radius, color, 1, 1, 1, 0))
            else:
                multi_colors.append((color, cnt))

        n_singles = len(singles)
        n_multi = len(multi_colors)
        print(
            f"  r={radius}: {n_singles:,} singles (no JSON needed) | "
            f"{n_multi:,} multi-entry colors (workers fetch JSON)"
        )

        if singles:
            for i in range(0, len(singles), INSERT_BATCH):
                connection.executemany(
                    "INSERT OR IGNORE INTO color_analysis VALUES (?, ?, ?, ?, ?, ?)",
                    singles[i : i + INSERT_BATCH],
                )
            connection.commit()

        if not multi_colors:
            print(f"  r={radius}: 0 collisions")
            continue

        # Workers open their own DB connections — no JSON held in main process.
        # Tasks contain only lightweight integer metadata.
        tasks: list[tuple] = [
            (str(DB_PATH), radius, color, cnt) for color, cnt in multi_colors
        ]

        batch: list[tuple] = []
        n_consistent = n_collision = 0

        with mp.get_context("spawn").Pool(processes=workers) as pool:
            with tqdm(
                total=len(tasks),
                desc=f"  r={radius} analyze",
                unit="color",
                dynamic_ncols=True,
            ) as pbar:
                for result in pool.imap_unordered(
                    analyze_color, tasks, chunksize=64
                ):
                    _, _, _, _, _, is_coll = result
                    if is_coll:
                        n_collision += 1
                    else:
                        n_consistent += 1
                    batch.append(result)
                    pbar.update()
                    pbar.set_postfix(
                        consistent=f"{n_consistent:,}",
                        collisions=f"{n_collision:,}",
                        refresh=False,
                    )
                    if len(batch) >= INSERT_BATCH:
                        connection.executemany(
                            "INSERT OR IGNORE INTO color_analysis "
                            "VALUES (?, ?, ?, ?, ?, ?)",
                            batch,
                        )
                        connection.commit()
                        batch.clear()

        if batch:
            connection.executemany(
                "INSERT OR IGNORE INTO color_analysis VALUES (?, ?, ?, ?, ?, ?)",
                batch,
            )
            connection.commit()

        total_in_db = connection.execute(
            "SELECT COUNT(*) FROM color_analysis WHERE radius = ?", (radius,)
        ).fetchone()[0]
        coll_in_db = connection.execute(
            "SELECT COUNT(*) FROM color_analysis WHERE radius = ? AND is_collision = 1",
            (radius,),
        ).fetchone()[0]
        print(
            f"  r={radius} stored: {total_in_db:,} total | "
            f"{coll_in_db:,} collisions ({100.0 * coll_in_db / total_in_db:.3f}%)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────


def print_summary(connection: sqlite3.Connection) -> None:
    print("\n\nSummary")
    print("=" * 70)

    n_mol = connection.execute("SELECT COUNT(*) FROM molecules").fetchone()[0]
    n_ok = connection.execute(
        "SELECT COUNT(*) FROM molecules WHERE error IS NULL"
    ).fetchone()[0]
    print(f"\nMolecules:   {n_mol:,}  (ok={n_ok:,}, errors={n_mol - n_ok:,})")

    n_atoms = connection.execute(
        "SELECT COUNT(*) FROM atom_fingerprints WHERE radius = ?", (RADII[0],)
    ).fetchone()[0]
    print(f"Non-H atoms: {n_atoms:,}")

    print(
        f"\n{'r':>2}  {'colors':>10}  {'consistent':>12}  "
        f"{'collisions':>12}  {'coll%':>7}"
    )
    print("-" * 50)
    for radius in RADII:
        total = connection.execute(
            "SELECT COUNT(*) FROM color_analysis WHERE radius = ?", (radius,)
        ).fetchone()[0]
        coll = connection.execute(
            "SELECT COUNT(*) FROM color_analysis "
            "WHERE radius = ? AND is_collision = 1",
            (radius,),
        ).fetchone()[0]
        pct = 100.0 * coll / total if total else 0.0
        print(
            f"{radius:>2}  {total:>10,}  {total - coll:>12,}  "
            f"{coll:>12,}  {pct:>6.3f}%"
        )

    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"DB not found: {DB_PATH}")

    print(f"DB:          {DB_PATH}")
    print(f"Radii:       {RADII}")
    print(
        f"Sample size: {SAMPLE_SIZE if SAMPLE_SIZE is not None else 'all (no limit)'}"
    )
    print(f"Workers:     {WORKERS}")

    with closing(sqlite3.connect(DB_PATH)) as con:
        init_analysis_tables(con)
        run_analyze_phase(con, WORKERS)
        print_summary(con)


if __name__ == "__main__":
    mp.freeze_support()
    main()
