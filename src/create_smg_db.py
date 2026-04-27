from __future__ import annotations

import csv
import multiprocessing as mp
import sqlite3
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool
from pathlib import Path

from rdkit import Chem, rdBase
from stereomolgraph.algorithms.circular import color_refine_hash_smg
from stereomolgraph.experimental._json import JSONHandler
from stereomolgraph.rdmol2graph import RDMol2StereoMolGraph
from tqdm import tqdm

rdBase.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "chembl_small.csv"
OUTPUT_DB_PATH = ROOT / "data" / "chembl_smg.db"
REQUIRED_COLUMNS = {"ChEMBL ID", "Smiles", "Inchi"}
LIMIT: int | None = None
WORKERS = max(1, (mp.cpu_count() or 1) - 1)
BATCH_SIZE = 1_000
LOOKUP_BATCH_SIZE = 1_000
POSTFIX_UPDATE_EVERY = 250
PROGRESS_INTERVAL = 0.5
MOLECULE_TIMEOUT_SECONDS = 5 * 60.0
EXCLUDED_ISOTOPES: set[int] | None = None
CTX = mp.get_context("spawn")

CONVERTER = RDMol2StereoMolGraph(
    resonance=True,
    stereo_complete=True,
    lone_pair_stereo=True,
)


@dataclass(frozen=True, slots=True)
class RecordResult:
    status: str
    record: tuple[str, str, str, int, str] | None = None


def iter_rows(input_path: Path, limit: int | None = None):
    with input_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise KeyError(f"Missing required columns: {sorted(missing)}")

        for index, row in enumerate(reader, start=1):
            chembl_id = (row.get("ChEMBL ID") or "").strip().strip('"')
            smiles = (row.get("Smiles") or "").strip().strip('"')
            inchi = (row.get("Inchi") or "").strip().strip('"')
            if chembl_id and smiles and inchi:
                yield chembl_id, smiles, inchi
            if limit is not None and index >= limit:
                break


def count_rows(input_path: Path, limit: int | None = None) -> int:
    return sum(1 for _ in iter_rows(input_path, limit=limit))


def iter_row_batches(input_path: Path, limit: int | None = None):
    batch: list[tuple[str, str, str]] = []
    for row in iter_rows(input_path, limit=limit):
        batch.append(row)
        if len(batch) >= LOOKUP_BATCH_SIZE:
            yield batch
            batch = []

    if batch:
        yield batch


def split_missing_rows(
    connection: sqlite3.Connection,
    rows: list[tuple[str, str, str]],
) -> tuple[list[tuple[str, str, str]], int]:
    if not rows:
        return [], 0

    chembl_ids = tuple(dict.fromkeys(chembl_id for chembl_id, _, _ in rows))
    placeholders = ",".join("?" * len(chembl_ids))
    blocked_ids = {
        chembl_id
        for (chembl_id,) in connection.execute(
            "SELECT chembl_id FROM smg_records "
            f"WHERE chembl_id IN ({placeholders})",
            chembl_ids,
        )
    }

    missing_rows: list[tuple[str, str, str]] = []
    skipped = 0
    for row in rows:
        chembl_id = row[0]
        if chembl_id in blocked_ids:
            skipped += 1
            continue
        missing_rows.append(row)
        blocked_ids.add(chembl_id)

    return missing_rows, skipped


def make_record(row: tuple[str, str, str]) -> RecordResult:
    chembl_id, smiles, inchi = row
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
    except Exception:
        return RecordResult("skipped")

    if mol is None:
        return RecordResult("skipped")

    isotopes = {atom.GetIsotope() for atom in mol.GetAtoms()} - {0}
    if isotopes and (
        EXCLUDED_ISOTOPES is None or isotopes & EXCLUDED_ISOTOPES
    ):
        return RecordResult("skipped")

    if any(
        info.specified != Chem.StereoSpecified.Specified
        for info in Chem.FindPotentialStereo(mol)
    ):
        return RecordResult("skipped")

    try:
        smg = CONVERTER(Chem.AddHs(mol)).freeze()
        smg_hash = int(color_refine_hash_smg(smg))
        smg_json = JSONHandler.json_serialize(smg)
        return RecordResult(
            "ok",
            (chembl_id, smiles, inchi, smg_hash, smg_json),
        )
    except Exception:
        return RecordResult("skipped")


def _send_record(row: tuple[str, str, str], connection) -> None:
    try:
        connection.send(make_record(row))
    except Exception:
        connection.send(RecordResult("skipped"))
    finally:
        connection.close()


def build_record(row: tuple[str, str, str]) -> RecordResult:
    parent_conn, child_conn = CTX.Pipe(duplex=False)
    process = CTX.Process(target=_send_record, args=(row, child_conn))
    process.start()
    child_conn.close()
    try:
        process.join(MOLECULE_TIMEOUT_SECONDS)
        if process.is_alive():
            process.terminate()
            process.join()
            return RecordResult("timed_out")
        return (
            parent_conn.recv()
            if parent_conn.poll()
            else RecordResult("skipped")
        )
    finally:
        parent_conn.close()


def init_db(connection: sqlite3.Connection) -> None:
    existing_columns = {
        row[1] for row in connection.execute("PRAGMA table_info(smg_records)")
    }
    required_columns = {"chembl_id", "smiles", "inchi", "smg_hash", "smg_json"}
    if existing_columns and not required_columns.issubset(existing_columns):
        raise RuntimeError(
            "Existing smg_records table is incompatible. "
            "Expected columns: chembl_id, smiles, inchi, smg_hash, smg_json."
        )

    connection.executescript(
        """
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = NORMAL;
        CREATE TABLE IF NOT EXISTS smg_records (
            chembl_id TEXT PRIMARY KEY,
            smiles TEXT NOT NULL,
            inchi TEXT NOT NULL,
            smg_hash INTEGER NOT NULL,
            smg_json TEXT NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_smg_records_inchi
        ON smg_records (inchi);
        CREATE INDEX IF NOT EXISTS idx_smg_records_hash
        ON smg_records (smg_hash);
        """
    )
    connection.commit()


def flush_batch(
    connection: sqlite3.Connection,
    batch: list[tuple[str, str, str, int, str]],
) -> tuple[int, int]:
    if not batch:
        return 0, 0

    before_changes = connection.total_changes
    connection.executemany(
        "INSERT OR IGNORE INTO smg_records "
        "(chembl_id, smiles, inchi, smg_hash, smg_json) "
        "VALUES (?, ?, ?, ?, ?)",
        batch,
    )
    connection.commit()
    inserted = connection.total_changes - before_changes
    duplicates = len(batch) - inserted
    return inserted, duplicates


def consume_results(
    connection: sqlite3.Connection,
    input_path: Path,
    total: int,
    limit: int | None,
    workers: int,
) -> tuple[int, int, int, int, int]:
    processed = inserted = duplicates = skipped = timed_out = 0
    existing = 0
    batch: list[tuple[str, str, str, int, str]] = []
    with (
        ThreadPool(processes=workers) as pool,
        tqdm(
            total=total,
            desc="Building SMG DB",
            unit="mol",
            mininterval=PROGRESS_INTERVAL,
        ) as progress,
    ):

        def update_progress_postfix() -> None:
            progress.set_postfix(
                inserted=inserted,
                duplicates=duplicates,
                existing=existing,
                skipped=skipped,
                timed_out=timed_out,
                refresh=False,
            )

        update_progress_postfix()

        for rows in iter_row_batches(input_path, limit=limit):
            missing_rows, skipped_existing = split_missing_rows(
                connection, rows
            )
            if skipped_existing:
                existing += skipped_existing
                progress.update(skipped_existing)
                update_progress_postfix()

            if not missing_rows:
                continue

            for result in pool.imap_unordered(
                build_record,
                missing_rows,
                chunksize=1,
            ):
                processed += 1

                if result.status == "ok" and result.record is not None:
                    batch.append(result.record)
                    if len(batch) >= BATCH_SIZE:
                        batch_inserted, batch_duplicates = flush_batch(
                            connection, batch
                        )
                        inserted += batch_inserted
                        duplicates += batch_duplicates
                        batch.clear()
                        update_progress_postfix()
                elif result.status == "timed_out":
                    timed_out += 1
                else:
                    skipped += 1

                progress.update()
                if progress.n % POSTFIX_UPDATE_EVERY == 0:
                    update_progress_postfix()

        batch_inserted, batch_duplicates = flush_batch(connection, batch)
        inserted += batch_inserted
        duplicates += batch_duplicates
        update_progress_postfix()

    return processed, inserted, duplicates, skipped, timed_out


def build_db(
    input_path: Path,
    output_path: Path,
    limit: int | None = None,
    workers: int = WORKERS,
) -> tuple[int, int, int, int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(output_path) as connection:
        init_db(connection)
        total = count_rows(input_path, limit=limit)
        return consume_results(
            connection,
            input_path,
            total,
            limit,
            workers,
        )


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_PATH}")

    processed, inserted, duplicates, skipped, timed_out = build_db(
        INPUT_PATH,
        OUTPUT_DB_PATH,
        limit=LIMIT,
        workers=WORKERS,
    )
    print(f"Input file: {INPUT_PATH}")
    print(f"Output DB:  {OUTPUT_DB_PATH}")
    print(f"Processed:  {processed}")
    print(f"Inserted:   {inserted}")
    print(f"Duplicates: {duplicates}")
    print(f"Skipped:    {skipped}")
    print(f"Timed out:  {timed_out}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
