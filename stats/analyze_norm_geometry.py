"""Measure norm/centroid association and dot-product anisotropy.

Writes one row per vector set: original raw vectors, raw directions with
globally shuffled lengths, lengths shuffled within centroid-alignment
bins, and unit vectors (the cosine reference).
Run from any directory:

    python stats/analyze_norm_geometry.py --seed 1024
    python stats/analyze_norm_geometry.py --files embeddings/my_vectors.npz

The scale-independent dot anisotropy index is the mean off-diagonal dot
product divided by the mean self dot product. For unit vectors it reduces
to mean pairwise cosine similarity. It describes global alignment, not
nearest-neighbor hubness or the Codenames score.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FILES = [
    ROOT / "embeddings/polish_embeddings_unnormalize.npz",
    ROOT / "embeddings/random_10000_unnormalized.npz",
]
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "norm_geometry_results.csv"


def load_vectors(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        vectors = archive["embeddings"].astype(np.float64)
    if vectors.ndim != 2 or len(vectors) < 2:
        raise ValueError(f"{path}: expected at least two vector rows")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(vectors).all() or np.any(norms == 0):
        raise ValueError(f"{path}: vectors must be finite and nonzero")
    return vectors


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def norm_variants(
    vectors: np.ndarray, seed: int, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return unit vectors, norms, alignment, full shuffle, within-bin shuffle.

    Alignment is cosine to the centroid of unit vectors. The last variant
    preserves the norm/alignment relationship approximately, while changing
    which individual word receives each length within an alignment bin.
    """
    norms = np.linalg.norm(vectors, axis=1)
    unit = vectors / norms[:, None]
    centroid = unit.mean(axis=0)
    centroid_length = np.linalg.norm(centroid)
    if centroid_length == 0:
        raise ValueError("Centroid of unit vectors has zero length")
    alignment = unit @ (centroid / centroid_length)
    rng = np.random.default_rng(seed)
    fully_shuffled = rng.permutation(norms)
    within_bin = norms.copy()
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    edges = np.quantile(alignment, np.arange(1, n_bins) / n_bins)
    bin_ids = np.searchsorted(edges, alignment, side="right")
    for bin_id in range(n_bins):
        ids = np.flatnonzero(bin_ids == bin_id)
        within_bin[ids] = rng.permutation(norms[ids])
    return unit, norms, alignment, fully_shuffled, within_bin


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    # Unit-normalized vectors have (up to floating-point noise) identical
    # norms, so a correlation with their lengths is undefined.
    if x.std() <= 1e-12 * max(1.0, abs(float(x.mean()))):
        return None
    x = x - x.mean()
    y = y - y.mean()
    scale = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / scale) if scale else None


def summarize(
    file_label: str, variant: str, vectors: np.ndarray, alignment: np.ndarray,
    seed: int, bins: int,
) -> dict:
    n, dim = vectors.shape
    norms = np.linalg.norm(vectors, axis=1)
    mean_self_dot = float(np.mean(norms ** 2))
    centroid = vectors.mean(axis=0)
    # (sum_{i != j} v_i . v_j) / (n * (n - 1)).
    mean_pair_dot = float(
        (n * np.dot(centroid, centroid) - mean_self_dot) / (n - 1)
    )
    return {
        "file": file_label,
        "variant": variant,
        "n_words": n,
        "dimension": dim,
        "seed": seed,
        "centroid_bins": bins,
        "mean_norm": float(norms.mean()),
        "std_norm": float(norms.std()),
        "norm_cv": float(norms.std() / norms.mean()),
        "norm_centroid_alignment_pearson": pearson(norms, alignment),
        "centroid_norm": float(np.linalg.norm(centroid)),
        "mean_pair_dot": mean_pair_dot,
        "mean_self_dot": mean_self_dot,
        "dot_anisotropy": mean_pair_dot / mean_self_dot,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", type=Path, nargs="+", default=DEFAULT_FILES)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--bins", type=int, default=10,
                        help="Number of equal-frequency centroid-alignment bins")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.bins < 2:
        parser.error("--bins must be at least 2")

    rows = []
    for path in args.files:
        raw = load_vectors(path)
        unit, norms, alignment, shuffled, within_bin = norm_variants(
            raw, args.seed, args.bins
        )
        label = display_path(path)
        for variant, vectors in (
            ("raw", raw),
            ("shuffled_norms", unit * shuffled[:, None]),
            ("shuffled_within_centroid_bin", unit * within_bin[:, None]),
            ("unit_cosine_reference", unit),
        ):
            rows.append(summarize(
                label, variant, vectors, alignment, args.seed, args.bins
            ))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {args.output}")
    for row in rows:
        correlation = row["norm_centroid_alignment_pearson"]
        correlation_text = f"{correlation:.3f}" if correlation is not None else "n/a"
        print(f"{row['file']} {row['variant']}: "
              f"norm/alignment r={correlation_text}, "
              f"dot anisotropy={row['dot_anisotropy']:.6f}")


if __name__ == "__main__":
    main()
