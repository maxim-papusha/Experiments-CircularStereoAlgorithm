# %%
import re
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from stereomolgraph import StereoMolGraph
from stereomolgraph.algorithms.circular import circular_stereo_generator
from stereomolgraph.coords import Geometry


def resolve_existing(*relative_paths: tuple[str, ...]) -> Path:
    search_roots = (Path.cwd().resolve(), *Path.cwd().resolve().parents)
    for root in search_roots:
        for parts in relative_paths:
            candidate = root.joinpath(*parts)
            if candidate.exists():
                return candidate
    raise FileNotFoundError(
        "Could not resolve any of: "
        + ", ".join(
            str(Path.cwd().resolve().joinpath(*parts))
            for parts in relative_paths
        )
    )


stereomolgraph_src = resolve_existing(
    ("packages", "StereoMolGraph", "src"),
    ("StereoMolGraph", "src"),
)
if str(stereomolgraph_src) not in sys.path:
    sys.path.insert(0, str(stereomolgraph_src))


data_root = resolve_existing(
    ("packages", "SI_StereoFingerprint", "data"),
    ("SI_StereoFingerprint", "data"),
    ("data",),
)
figure_dir = resolve_existing(
    ("packages", "SI_StereoFingerprint", "figures"),
    ("SI_StereoFingerprint", "figures"),
    ("figures",),
)


def half_violin_density(values, y_grid, width: float = 0.35):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.zeros_like(y_grid)

    if (
        gaussian_kde is not None
        and values.size > 1
        and float(np.ptp(values)) > 0.0
    ):
        density = gaussian_kde(values, bw_method=0.3)(y_grid)
    else:
        bins = min(40, max(5, values.size * 2))
        hist, edges = np.histogram(
            values,
            bins=bins,
            range=(float(y_grid.min()), float(y_grid.max())),
            density=True,
        )
        centers = 0.5 * (edges[:-1] + edges[1:])
        density = np.interp(y_grid, centers, hist, left=0.0, right=0.0)
        density = np.convolve(
            density,
            np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0,
            mode="same",
        )

    max_density = float(np.max(density)) if density.size else 0.0
    if max_density == 0.0:
        return np.zeros_like(y_grid)
    return density / max_density * width


# %%
data_dir = data_root / "MeCN_NNhN_Me"
xyz_paths = sorted(p for p in data_dir.glob("*.xyz"))

file_smgs: dict[str, StereoMolGraph] = {}
for p in xyz_paths:
    geoms = Geometry.from_xyz_file_multi(p)
    file_smgs[p.name] = StereoMolGraph.from_geometry(geoms[0])


def get_base(fname: str) -> str:
    m = re.match(r"(.+?)_(\d+)\.xyz$", fname)
    return m.group(1) if m else fname.replace(".xyz", "")


families: dict[str, list[str]] = {}
for fname in file_smgs:
    families.setdefault(get_base(fname), []).append(fname)
families = {b: sorted(fs) for b, fs in families.items() if len(fs) >= 2}

print(f"Using data directory: {data_dir}")
print(f"Loaded {len(file_smgs)} files")
print(f"Stereoisomer families (>=2 members): {len(families)}")
for base, fs in families.items():
    print(f"  {base}: {len(fs)} isomers")

# %%
radii = [2, 3, 4, 5]
max_r = max(radii)

# Compute count fingerprints for each file at each radius
fps: dict[int, dict[str, Counter]] = {r: {} for r in radii}

for fname, smg in file_smgs.items():
    gen = circular_stereo_generator(smg)
    for r_idx, fp in enumerate(gen):
        if r_idx > max_r:
            break
        if r_idx in fps:
            fps[r_idx][fname] = Counter(fp.tolist())


def weighted_tanimoto(a: Counter, b: Counter) -> float:
    keys = set(a) | set(b)
    inter = sum(min(a[k], b[k]) for k in keys)
    union = sum(max(a[k], b[k]) for k in keys)
    return inter / union if union else 1.0


# All stereoisomer pairs
pairs = []
for base, fs in families.items():
    pairs.extend(combinations(fs, 2))

# Compute Jaccard distance = 1 - weighted_tanimoto for each pair at each radius
distances: dict[int, list[float]] = {}
for r in radii:
    distances[r] = [
        1.0 - weighted_tanimoto(fps[r][f1], fps[r][f2]) for f1, f2 in pairs
    ]

print(f"Total stereoisomer pairs: {len(pairs)}")
for r in radii:
    d = distances[r]
    print(
        f"  Radius {r}: mean={np.mean(d):.4f}, min={np.min(d):.4f}, max={np.max(d):.4f}"
    )

# %%
# Publication styling
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "figure.dpi": 300,
    }
)

fig, ax = plt.subplots(figsize=(3.5, 3.0))

data = [distances[r] for r in radii]
palette = ["#3A76AF", "#E07B39", "#59A257", "#C44E52"]
y_grid = np.linspace(-0.02, 1.02, 200)
rng = np.random.default_rng(42)

for r, d, col in zip(radii, data, palette):
    d_arr = np.asarray(d, dtype=float)

    density = half_violin_density(d_arr, y_grid)
    ax.fill_betweenx(y_grid, r, r + density, alpha=0.45, color=col, lw=0)
    ax.plot(r + density, y_grid, color=col, lw=0.8, alpha=0.7)

    jitter = rng.uniform(-0.22, -0.06, len(d_arr))
    ax.scatter(
        r + jitter,
        d_arr,
        s=8,
        alpha=0.55,
        color=col,
        edgecolors="white",
        linewidths=0.3,
        zorder=3,
    )

ax.set_xticks(radii)
ax.set_xlabel("Fingerprint radius", fontsize=10)
ax.set_ylabel("Jaccard distance", fontsize=10)
ax.set_ylim(-0.05, 1.05)
ax.set_xlim(1.3, 5.7)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.grid(True, lw=0.4, alpha=0.4, color="0.7")
ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig(figure_dir / "jaccard_violin_NNhN_Me.pdf", bbox_inches="tight")
fig.savefig(
    figure_dir / "jaccard_violin_NNhN_Me.png", bbox_inches="tight", dpi=300
)
plt.show()

# %%
data_dir2 = data_root / "MeCN_NNpyN_Me"

xyz_paths2 = sorted(p for p in data_dir2.glob("MeCN_*.xyz"))

file_smgs2: dict[str, StereoMolGraph] = {}
for p in xyz_paths2:
    geoms = Geometry.from_xyz_file_multi(p)
    file_smgs2[p.name] = StereoMolGraph.from_geometry(geoms[0])

families2: dict[str, list[str]] = {}
for fname in file_smgs2:
    families2.setdefault(get_base(fname), []).append(fname)
families2 = {b: sorted(fs) for b, fs in families2.items() if len(fs) >= 2}

fps2: dict[int, dict[str, Counter]] = {r: {} for r in radii}
for fname, smg in file_smgs2.items():
    gen = circular_stereo_generator(smg)
    for r_idx, fp in enumerate(gen):
        if r_idx > max_r:
            break
        if r_idx in fps2:
            fps2[r_idx][fname] = Counter(fp.tolist())

pairs2 = []
for base, fs in families2.items():
    pairs2.extend(combinations(fs, 2))

distances2: dict[int, list[float]] = {}
for r in radii:
    distances2[r] = [
        1.0 - weighted_tanimoto(fps2[r][f1], fps2[r][f2]) for f1, f2 in pairs2
    ]

print(f"Using data directory: {data_dir2}")
print(f"Loaded {len(file_smgs2)} files")
print(f"Stereoisomer families (>=2 members): {len(families2)}")
for base, fs in families2.items():
    print(f"  {base}: {len(fs)} isomers")
print(f"\nTotal stereoisomer pairs: {len(pairs2)}")
for r in radii:
    d = distances2[r]
    print(
        f"  Radius {r}: mean={np.mean(d):.4f}, min={np.min(d):.4f}, max={np.max(d):.4f}"
    )

# %%
fig, ax = plt.subplots(figsize=(3.5, 3.0))

data2 = [distances2[r] for r in radii]
palette = ["#3A76AF", "#E07B39", "#59A257", "#C44E52"]

for i, (r, d, col) in enumerate(zip(radii, data2, palette)):
    d_arr = np.array(d)

    # KDE half-violin on the right
    kde = gaussian_kde(d_arr, bw_method=0.3)
    y_grid = np.linspace(-0.02, 1.02, 200)
    density = kde(y_grid)
    density = density / density.max() * 0.35
    ax.fill_betweenx(y_grid, r, r + density, alpha=0.45, color=col, lw=0)
    ax.plot(r + density, y_grid, color=col, lw=0.8, alpha=0.7)

    # Jittered strip on the left
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.22, -0.06, len(d_arr))
    ax.scatter(
        r + jitter,
        d_arr,
        s=8,
        alpha=0.55,
        color=col,
        edgecolors="white",
        linewidths=0.3,
        zorder=3,
    )

ax.set_xticks(radii)
ax.set_xlabel("Fingerprint radius", fontsize=10)
ax.set_ylabel("Jaccard distance", fontsize=10)
ax.set_ylim(-0.05, 1.05)
ax.set_xlim(1.3, 5.7)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.grid(True, lw=0.4, alpha=0.4, color="0.7")
ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig("../figures/jaccard_violin_NNpyN_Me.pdf", bbox_inches="tight")
fig.savefig(
    "../figures/jaccard_violin_NNpyN_Me.png", bbox_inches="tight", dpi=300
)
plt.show()
