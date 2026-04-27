"""Visualization helpers for Morgan fingerprint exploration."""

from __future__ import annotations

import itertools
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from typing import Any, DefaultDict, Dict, FrozenSet, List, Set, Tuple

import rdkit
from rdkit import Chem  # type: ignore[import]
from rdkit.Chem import (
    Draw,  # type: ignore[import]
    rdFingerprintGenerator,  # type: ignore[import]
)
from rdkit.Chem.Draw import rdMolDraw2D  # type: ignore[import]
from stereomolgraph import StereoMolGraph
from stereomolgraph.algorithms.circular import circular_stereo_generator

DEFAULT_DISTINCT_COLORS: Sequence[Tuple[float, float, float]] = (
    (0.121, 0.466, 0.705),  # Blue
    (0.682, 0.780, 0.909),  # Light Blue
    (1.000, 0.498, 0.054),  # Orange
    (1.000, 0.733, 0.470),  # Light Orange
    (0.172, 0.627, 0.172),  # Green
    (0.596, 0.874, 0.541),  # Light Green
    (0.839, 0.152, 0.156),  # Red
    (1.000, 0.596, 0.588),  # Light Red
    (0.580, 0.403, 0.741),  # Purple
    (0.772, 0.690, 0.835),  # Light Purple
    (0.549, 0.337, 0.294),  # Brown
    (0.768, 0.611, 0.580),  # Light Brown
    (0.890, 0.466, 0.760),  # Pink
    (0.968, 0.713, 0.823),  # Light Pink
    (0.498, 0.498, 0.498),  # Gray
    (0.780, 0.780, 0.780),  # Light Gray
    (0.737, 0.741, 0.133),  # Olive
    (0.858, 0.858, 0.552),  # Light Olive
    (0.090, 0.745, 0.811),  # Cyan
    (0.619, 0.854, 0.898),  # Light Cyan
)


def ecfp(
    mol: Chem.Mol,
    radius: int,
    include_chirality: bool = True,
    includeRingMembership: bool = False,
    useBondTypes: bool = True,
    onlyNonzeroInvariants: bool = False,
) -> Dict[int, FrozenSet[int]]:
    """Return a per-atom hash summarising the Morgan fingerprint environment.

    Parameters
    ----------
    mol
        Molecule to fingerprint.
    radius
        Morgan fingerprint radius.
    include_chirality
        Whether to encode chirality information.
    includeRingMembership
        Whether to encode ring membership invariants.
    useBondTypes
        Whether to encode bond types in the fingerprint.
    onlyNonzeroInvariants
        Limit invariants to atoms that contribute non-zero features.
    """
    generator: Any = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        includeChirality=include_chirality,
        includeRingMembership=includeRingMembership,
        useBondTypes=useBondTypes,
        onlyNonzeroInvariants=onlyNonzeroInvariants,
    )
    aux_output = rdFingerprintGenerator.AdditionalOutput()
    aux_output.AllocateBitInfoMap()
    # generator.GetFingerprint(mol, additionalOutput=aux_output)
    generator.GetCountFingerprint(mol, additionalOutput=aux_output)
    bitinfo: dict[int, Any] = aux_output.GetBitInfoMap()

    atom_bits: Dict[int, Set[int]] = {}
    for bit, contributors in bitinfo.items():
        for atom, _ in contributors:
            atom_bits.setdefault(atom, set()).add(bit)

    atom_hashes: Dict[int, FrozenSet[int]] = {}
    num_atoms = mol.GetNumAtoms()
    for atom_idx in range(num_atoms):
        bits = atom_bits.get(atom_idx, set())
        atom_hashes[atom_idx] = frozenset(bits)

    return atom_hashes


def smg_fp(
    mol: Chem.Mol,
    radius: int,
) -> Dict[int, FrozenSet[int]]:
    ret = {}
    mol = Chem.AddHs(mol)
    smg = StereoMolGraph.from_rdmol(mol, stereo_complete=True)

    gen = circular_stereo_generator(smg)
    fp = next(itertools.islice(gen, radius, radius + 1))
    for i, (atom_fp, atom) in enumerate(zip(fp, smg.atoms)):
        if smg.atom_types[i] != 1:
            ret[atom] = frozenset([atom_fp])
    return ret


def visualize_molecules_with_atom_labels(
    smiles_list: Iterable[str],
    radii: Sequence[int] = (1, 2, 3),
    *,
    fingerprint_generator: Callable[..., Dict[int, Any]] = ecfp,
    show_atom_indices: bool = True,
    show_atom_bits: bool = True,
    distinct_colors: Sequence[Tuple[float, float, float]] | None = None,
    **kwargs: Any,
) -> str:
    """Render molecules with per-atom Morgan fingerprint hashes highlighted."""
    palette = tuple(distinct_colors or DEFAULT_DISTINCT_COLORS)

    mols: List[Chem.Mol] = []
    legends: List[str] = []
    highlight_atom_lists: List[List[int]] = []
    highlight_atom_colors: List[Dict[int, Tuple[float, float, float]]] = []

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        mol.UpdatePropertyCache(strict=False)
        # mol = Chem.AddHs(mol)
        rdkit.Chem.GetSymmSSSR(mol)
        if mol is None:  # type: ignore[unreachable]
            raise ValueError(f"Failed to parse SMILES string: {smiles}")
        num_atoms = mol.GetNumAtoms()

        for radius in radii:
            atom_hashes = fingerprint_generator(mol, radius=radius, **kwargs)
            unique_hashes = sorted(set(atom_hashes.values()))

            # Cycle through the palette if there are more unique hashes than colors
            colors = list(
                itertools.islice(itertools.cycle(palette), len(unique_hashes))
            )
            hash_to_color: Dict[Any, Tuple[float, float, float]] = dict(
                zip(unique_hashes, colors)
            )

            highlight_colors = {
                idx: hash_to_color[h] for idx, h in atom_hashes.items()
            }
            highlight_atoms = list(highlight_colors.keys())

            if show_atom_bits:
                hash_to_atoms: DefaultDict[int, List[int]] = defaultdict(list)
                for atom_idx, hash_val in atom_hashes.items():
                    hash_to_atoms[hash_val].append(atom_idx)
                legend_parts = [
                    f"{sorted(hash_val)}: {','.join(str(atom) for atom in sorted(atom_ids))}"
                    for hash_val, atom_ids in sorted(hash_to_atoms.items())
                ]
                legend = "\n ".join(legend_parts)
            else:
                legend = ""

            mols.append(mol)
            legends.append(legend)
            highlight_atom_lists.append(highlight_atoms)
            highlight_atom_colors.append(highlight_colors)

    draw_options = rdMolDraw2D.MolDrawOptions()
    draw_options.addStereoAnnotation = True
    if show_atom_indices:
        draw_options.addAtomIndices = True

    if show_atom_bits:
        draw_options.legendFraction = 0.25

    svg = Draw.MolsToGridImage(
        mols,
        molsPerRow=len(radii),
        legends=legends,
        highlightAtomLists=highlight_atom_lists,
        highlightAtomColors=highlight_atom_colors,
        highlightBondLists=[[] for _ in mols],
        useSVG=True,
        drawOptions=draw_options,
    )
    return svg
