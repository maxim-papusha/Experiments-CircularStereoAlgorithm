from __future__ import annotations

import hashlib
import multiprocessing as mp
import random
import sqlite3
from contextlib import closing
from copy import deepcopy
from pathlib import Path

from stereomolgraph import StereoMolGraph
from stereomolgraph.algorithms.circular import color_refine_hash_smg
from stereomolgraph.experimental import JSONHandler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "chembl_smg.db"
REORDERINGS = 30
RANDOM_SEED = 0
WORKERS = max(1, (mp.cpu_count() or 1) - 1)
BATCH_SIZE = 1_000
CHUNK_SIZE = 10
CTX = mp.get_context("spawn")

COUNT_SQL = "SELECT COUNT(*) FROM smg_records"
ROWS_SQL = """
SELECT chembl_id, smiles, smg_hash, smg_json
FROM smg_records
ORDER BY chembl_id
"""


def count_rows(connection: sqlite3.Connection) -> int:
    row = connection.execute(COUNT_SQL).fetchone()
    if row is None:
        raise RuntimeError("Could not count smg_records rows.")
    return int(row[0])


def iter_row_batches(
    connection: sqlite3.Connection,
    batch_size: int = BATCH_SIZE,
):
    cursor = connection.execute(ROWS_SQL)
    try:
        while rows := cursor.fetchmany(batch_size):
            yield [
                (
                    str(chembl_id),
                    str(smiles),
                    int(smg_hash),
                    str(smg_json),
                )
                for chembl_id, smiles, smg_hash, smg_json in rows
            ]
    finally:
        cursor.close()


def build_hash(graph: StereoMolGraph) -> int:
    return int(color_refine_hash_smg(graph.copy(frozen=True)))


def rebuild_graph(
    graph: StereoMolGraph,
    atom_order: list[int],
    bond_order: list[tuple[int, ...]],
) -> StereoMolGraph:
    other_graph = StereoMolGraph()

    for atom in atom_order:
        atom_attrs = dict(graph.get_atom_attributes(atom))
        atom_type = atom_attrs.pop("atom_type")
        other_graph.add_atom(atom, atom_type, **atom_attrs)

    for atom1, atom2 in bond_order:
        bond_attrs = dict(graph.get_bond_attributes(atom1, atom2))
        other_graph.add_bond(atom1, atom2, **bond_attrs)

    for atom_stereo in graph.atom_stereo.values():
        other_graph.set_atom_stereo(deepcopy(atom_stereo))

    for bond_stereo in graph.bond_stereo.values():
        other_graph.set_bond_stereo(deepcopy(bond_stereo))

    return other_graph


def check_row(row: tuple[str, str, int, str]) -> dict[str, object]:
    chembl_id, smiles, stored_hash, smg_json = row

    try:
        graph = JSONHandler.json_deserialize(smg_json)
    except Exception:
        return {
            "chembl_id": chembl_id,
            "smiles": smiles,
            "status": "parse_failed",
        }

    if not isinstance(graph, StereoMolGraph):
        return {
            "chembl_id": chembl_id,
            "smiles": smiles,
            "status": "parse_failed",
        }

    current_hash = build_hash(graph)
    reordered_hashes = {stored_hash, current_hash}
    changed_hashes: set[int] = set()
    atoms = sorted(int(atom) for atom in graph.atoms)
    bonds = sorted(tuple(sorted(bond)) for bond in graph.bonds)
    atom_count = len(atoms)

    if current_hash != stored_hash:
        changed_hashes.add(current_hash)

    if atom_count > 1:
        digest = hashlib.blake2b(
            f"{RANDOM_SEED}\t{chembl_id}\t{stored_hash}".encode("utf-8"),
            digest_size=8,
        ).digest()
        rng = random.Random(
            int.from_bytes(digest, byteorder="big", signed=False)
        )
        seen_orders = {tuple(atoms)}

        for _ in range(max(REORDERINGS * 10, 20)):
            atom_order = atoms.copy()
            bond_order = bonds.copy()
            rng.shuffle(atom_order)
            rng.shuffle(bond_order)
            order_tuple = tuple(atom_order)
            if order_tuple in seen_orders:
                continue
            seen_orders.add(order_tuple)

            reordered_hash = build_hash(
                rebuild_graph(graph, atom_order, bond_order)
            )
            reordered_hashes.add(reordered_hash)
            if reordered_hash != stored_hash:
                changed_hashes.add(reordered_hash)
            if len(seen_orders) - 1 >= REORDERINGS:
                break

    return {
        "chembl_id": chembl_id,
        "smiles": smiles,
        "status": "changed" if changed_hashes else "stable",
        "stored_hash": stored_hash,
        "current_hash": current_hash,
        "unique_hashes": sorted(reordered_hashes),
        "changed_hashes": sorted(changed_hashes),
    }


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    total = stable = changed = parse_failed = 0
    changed_rows: list[dict[str, object]] = []

    with closing(sqlite3.connect(DB_PATH)) as connection:
        total_rows = count_rows(connection)

        with (
            CTX.Pool(processes=WORKERS) as pool,
            tqdm(
                total=total_rows,
                desc="Checking ChEMBL hashes",
                unit="row",
            ) as progress,
        ):
            for batch in iter_row_batches(connection):
                for result in pool.imap_unordered(
                    check_row,
                    batch,
                    chunksize=CHUNK_SIZE,
                ):
                    total += 1
                    status = result["status"]

                    if status == "stable":
                        stable += 1
                    elif status == "changed":
                        changed += 1
                        changed_rows.append(result)
                    else:
                        parse_failed += 1

                    progress.update()
                    progress.set_postfix(
                        changed=changed,
                        parse_failed=parse_failed,
                        refresh=False,
                    )

    print(f"Input DB:        {DB_PATH}")
    print(f"Rows checked:    {total}")
    print(f"Stable hashes:   {stable}")
    print(f"Changed hashes:  {changed}")
    print(f"Parse failures:  {parse_failed}")

    if changed_rows:
        print("\nRows with changed smg_hash under atom reordering:")
        for row in sorted(
            changed_rows,
            key=lambda row: str(row["chembl_id"]),
        ):
            print(
                f"{row['chembl_id']}: "
                f"stored={row['stored_hash']} "
                f"current={row['current_hash']}"
            )
            print(f"  changed_hashes={row['changed_hashes']}")
            print(f"  smiles={row['smiles']}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
