from __future__ import annotations

import multiprocessing as mp
import sqlite3
from contextlib import closing
from pathlib import Path

from stereomolgraph import StereoMolGraph
from stereomolgraph.algorithms.circular import circular_stereo_generator
from stereomolgraph.experimental import JSONHandler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
INPUT_DB_PATH = ROOT / "data" / "chembl_smg.db"
OUTPUT_DB_PATH = ROOT / "data" / "subgraph_smg.db"
LIMIT: int | None = None
WORKERS = max(1, (mp.cpu_count() or 1) - 1)
MOLECULE_BATCH_SIZE = 200
RADII = (2, 3, 4)
MAX_RADIUS = max(RADII)
HYDROGEN_ATOM_TYPE = 1

MOLECULE_SQL = """
INSERT OR REPLACE INTO molecules (chembl_id, smiles, inchi, error)
VALUES (?, ?, ?, ?)
"""

FINGERPRINT_SQL = """
INSERT OR REPLACE INTO atom_fingerprints (
    chembl_id,
    atom_id,
    radius,
    color,
    subgraph_json
)
VALUES (?, ?, ?, ?, ?)
"""


def count_rows(
    connection: sqlite3.Connection,
    limit: int | None = None,
) -> int:
    if limit is None:
        row = connection.execute("SELECT COUNT(*) FROM smg_records").fetchone()
    else:
        row = connection.execute(
            "SELECT COUNT(*) FROM (SELECT 1 FROM smg_records LIMIT ?)",
            (limit,),
        ).fetchone()
    if row is None:
        raise RuntimeError("Could not count source molecules.")
    return int(row[0])


def iter_row_batches(
    connection: sqlite3.Connection,
    limit: int | None = None,
    batch_size: int = MOLECULE_BATCH_SIZE,
):
    query = (
        "SELECT chembl_id, smiles, inchi, smg_json "
        "FROM smg_records ORDER BY chembl_id"
    )
    params: tuple[int, ...] = ()
    if limit is not None:
        query += " LIMIT ?"
        params = (limit,)

    cursor = connection.execute(query, params)
    try:
        while rows := cursor.fetchmany(batch_size):
            yield rows
    finally:
        cursor.close()


def atoms_within_radius(smg, center: int, radius: int) -> set[int]:
    visited = {center}
    frontier = [center]
    for _ in range(radius):
        next_frontier = []
        for node in frontier:
            for neighbor in smg.bonded_to(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.append(neighbor)
        frontier = next_frontier
    return visited


def process_molecule(
    row: tuple[str, str, str, str],
) -> tuple[
    str,
    str,
    str,
    list[tuple[int, int, int, str]] | None,
    str | None,
]:
    chembl_id, smiles, inchi, smg_json = row
    try:
        smg = JSONHandler.json_deserialize(smg_json)
        if not isinstance(smg, StereoMolGraph):
            raise TypeError("Expected StereoMolGraph JSON.")

        gen = circular_stereo_generator(smg)
        colors_at_radius: dict[int, tuple[int, ...]] = {}
        for radius in range(MAX_RADIUS + 1):
            colors = next(gen)
            if radius in RADII:
                colors_at_radius[radius] = tuple(
                    int(color) for color in colors
                )

        fingerprints: list[tuple[int, int, int, str]] = []
        atoms = list(smg.atoms)
        atom_types = list(smg.atom_types)

        # Skip hydrogen centers. Subgraphs may still contain H atoms.
        for index, atom_id in enumerate(atoms):
            if atom_types[index] == HYDROGEN_ATOM_TYPE:
                continue

            for radius in RADII:
                atom_ids = atoms_within_radius(smg, atom_id, radius)
                subgraph = smg.subgraph(atom_ids)
                subgraph.freeze()
                fingerprints.append(
                    (
                        int(atom_id),
                        radius,
                        int(colors_at_radius[radius][index]),
                        JSONHandler.json_serialize(subgraph),
                    )
                )

        return chembl_id, smiles, inchi, fingerprints, None
    except Exception as exc:
        return chembl_id, smiles, inchi, None, str(exc)


def init_db(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = NORMAL;
        CREATE TABLE IF NOT EXISTS molecules (
            chembl_id TEXT PRIMARY KEY,
            smiles TEXT NOT NULL,
            inchi TEXT NOT NULL,
            error TEXT
        );
        CREATE TABLE IF NOT EXISTS atom_fingerprints (
            chembl_id TEXT NOT NULL,
            atom_id INTEGER NOT NULL,
            radius INTEGER NOT NULL,
            color INTEGER NOT NULL,
            subgraph_json TEXT NOT NULL,
            PRIMARY KEY (chembl_id, atom_id, radius),
            FOREIGN KEY (chembl_id) REFERENCES molecules(chembl_id)
        );
        CREATE INDEX IF NOT EXISTS idx_atom_fingerprints_radius_color
        ON atom_fingerprints (radius, color);
        CREATE INDEX IF NOT EXISTS idx_atom_fingerprints_chembl
        ON atom_fingerprints (chembl_id);
        """
    )
    connection.commit()


def flush_batches(
    connection: sqlite3.Connection,
    molecule_batch: list[tuple[str, str, str, str | None]],
    fingerprint_batch: list[tuple[str, int, int, int, str]],
) -> None:
    if not molecule_batch:
        return

    chembl_ids = [
        chembl_id for chembl_id, _smiles, _inchi, _error in molecule_batch
    ]
    placeholders = ",".join("?" * len(chembl_ids))
    connection.execute(
        f"DELETE FROM atom_fingerprints WHERE chembl_id IN ({placeholders})",
        chembl_ids,
    )
    connection.executemany(MOLECULE_SQL, molecule_batch)
    if fingerprint_batch:
        connection.executemany(FINGERPRINT_SQL, fingerprint_batch)
    connection.commit()


def build_db(
    input_db_path: Path,
    output_db_path: Path,
    limit: int | None = None,
    workers: int = WORKERS,
) -> tuple[int, int, int, int]:
    output_db_path.parent.mkdir(parents=True, exist_ok=True)

    with closing(sqlite3.connect(input_db_path)) as input_connection:
        total = count_rows(input_connection, limit=limit)

    with (
        closing(sqlite3.connect(output_db_path)) as output_connection,
        mp.get_context("spawn").Pool(processes=workers) as pool,
        tqdm(
            total=total,
            desc="Building subgraph DB",
            unit="mol",
        ) as progress,
    ):
        init_db(output_connection)

        processed = succeeded = failed = fingerprint_rows = 0
        molecule_batch: list[tuple[str, str, str, str | None]] = []
        fingerprint_batch: list[tuple[str, int, int, int, str]] = []

        def update_progress_postfix() -> None:
            progress.set_postfix(
                succeeded=succeeded,
                failed=failed,
                fp_rows=fingerprint_rows,
                refresh=False,
            )

        update_progress_postfix()

        with closing(sqlite3.connect(input_db_path)) as input_connection:
            for row_batch in iter_row_batches(input_connection, limit=limit):
                for (
                    chembl_id,
                    smiles,
                    inchi,
                    fingerprints,
                    error,
                ) in pool.imap_unordered(
                    process_molecule,
                    row_batch,
                    chunksize=20,
                ):
                    processed += 1
                    molecule_batch.append((chembl_id, smiles, inchi, error))
                    if error is None and fingerprints is not None:
                        succeeded += 1
                        fingerprint_rows += len(fingerprints)
                        fingerprint_batch.extend(
                            (
                                chembl_id,
                                atom_id,
                                radius,
                                color,
                                subgraph_json,
                            )
                            for (
                                atom_id,
                                radius,
                                color,
                                subgraph_json,
                            ) in fingerprints
                        )
                    else:
                        failed += 1

                    if len(molecule_batch) >= MOLECULE_BATCH_SIZE:
                        flush_batches(
                            output_connection,
                            molecule_batch,
                            fingerprint_batch,
                        )
                        molecule_batch.clear()
                        fingerprint_batch.clear()

                    progress.update()
                    if processed % MOLECULE_BATCH_SIZE == 0:
                        update_progress_postfix()

        flush_batches(output_connection, molecule_batch, fingerprint_batch)
        update_progress_postfix()

    return total, succeeded, failed, fingerprint_rows


def main() -> None:
    if not INPUT_DB_PATH.exists():
        raise FileNotFoundError(f"Input DB not found: {INPUT_DB_PATH}")

    total, succeeded, failed, fingerprint_rows = build_db(
        INPUT_DB_PATH,
        OUTPUT_DB_PATH,
        limit=LIMIT,
        workers=WORKERS,
    )
    print(f"Input DB:         {INPUT_DB_PATH}")
    print(f"Output DB:        {OUTPUT_DB_PATH}")
    print(f"Source molecules: {total:,}")
    print(f"Successful:       {succeeded:,}")
    print(f"Failed:           {failed:,}")
    print(f"Fingerprint rows: {fingerprint_rows:,}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
