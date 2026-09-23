"""
test_representativeness.py
----------------------------
Tests whether the small "builtin" word set (the most common Polish words)
is a representative random sample of the larger vocabulary, or whether it
occupies a distinct, more compact region of the embedding space -- e.g.
because basic, everyday nouns tend to be semantically closer to each other
than a random selection of words would be.

Method (bootstrap / permutation test):
  1. Compute two statistics for the real small-vocabulary embeddings:
       - compactness: the norm of the (unit-)normalized centroid of the
         set. For unit vectors this is a monotonic, cheap proxy for the
         mean pairwise cosine similarity within the set -- the more the
         vectors point in similar directions, the larger this norm (it
         approaches 1 for a tightly clustered set, and approaches 0 for
         vectors spread out in all directions).
       - centroid_shift: the distance between the small set's centroid
         and the centroid of the full large vocabulary.
  2. Draw n_bootstrap random subsamples of the same size from the large
     vocabulary and compute the same two statistics for each -- this is
     the null distribution: "what these statistics would look like if the
     small set really were just a random sample".
  3. Report where the real small-set statistics fall within that null
     distribution, as an empirical one-sided p-value (probability of a
     random subsample being at least as extreme as the real one).

A small p-value for either statistic supports the idea that the small
word set is NOT a representative random sample -- it is more compact
and/or more shifted from the center of the large vocabulary than random
chance would produce.

Note: this script lives in stats/. Its default embeddings/... and output
paths are resolved relative to the script's own location (not the current
working directory), so it works whether you run it as
`python stats/test_representativeness.py` from the project root or as
`python test_representativeness.py` from inside stats/.

Requirements:
    pip install numpy tqdm

Usage:
    python stats/test_representativeness.py
    python stats/test_representativeness.py --bootstrap 5000 --seed 42
    python stats/test_representativeness.py --small embeddings/builtin_embeddings.npz --large embeddings/polish_embeddings.npz
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"

DEFAULT_SMALL = str(EMBEDDINGS_DIR / "builtin_embeddings.npz")
DEFAULT_LARGE = str(EMBEDDINGS_DIR / "polish_embeddings.npz")
DEFAULT_OUTPUT = str(SCRIPT_DIR / "representativeness_results.csv")
DEFAULT_BOOTSTRAP = 2000


def load_normalized(path: str) -> np.ndarray:
    """Loads an embeddings file and returns unit-normalized vectors,
    regardless of whether the file itself was already normalized -- the
    compactness statistic below is only meaningful on unit vectors."""
    data = np.load(path, allow_pickle=True)
    embeddings = data["embeddings"].astype(np.float64)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


def compactness(vectors: np.ndarray) -> float:
    """Norm of the centroid of a set of unit vectors -- higher means the
    vectors point in more similar directions, i.e. a more tightly
    clustered set (see module docstring for the exact relationship to
    mean pairwise cosine similarity)."""
    centroid = vectors.mean(axis=0)
    return float(np.linalg.norm(centroid))


def centroid_shift(vectors: np.ndarray, reference_centroid: np.ndarray) -> float:
    centroid = vectors.mean(axis=0)
    return float(np.linalg.norm(centroid - reference_centroid))


def bootstrap_null(large_vectors: np.ndarray, sample_size: int, n_bootstrap: int,
                    rng: np.random.Generator):
    n_large = large_vectors.shape[0]
    reference_centroid = large_vectors.mean(axis=0)

    compactness_null = np.empty(n_bootstrap)
    shift_null = np.empty(n_bootstrap)

    for i in tqdm(range(n_bootstrap), desc="Bootstrap", leave=False):
        idx = rng.choice(n_large, size=sample_size, replace=False)
        sample = large_vectors[idx]
        compactness_null[i] = compactness(sample)
        shift_null[i] = centroid_shift(sample, reference_centroid)

    return compactness_null, shift_null, reference_centroid


def empirical_p_value(observed: float, null_distribution: np.ndarray) -> float:
    """One-sided p-value: fraction of random subsamples at least as
    extreme (>=) as the observed statistic. One-sided because the
    hypothesis being tested is directional: MORE compact / MORE shifted
    than a random subsample, not merely 'different'."""
    return float(np.mean(null_distribution >= observed))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap test for whether the small (builtin) word set "
                     "is a representative random sample of the large vocabulary.",
    )
    parser.add_argument("--small", default=DEFAULT_SMALL, help="Small embeddings file (.npz)")
    parser.add_argument("--large", default=DEFAULT_LARGE, help="Large embeddings file (.npz)")
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP,
                         help=f"Number of random subsamples to draw, default: {DEFAULT_BOOTSTRAP}")
    parser.add_argument("--seed", type=int, default=None, help="Random seed, for reproducible output")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                         help=f"Output CSV path, default: {DEFAULT_OUTPUT}")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    small_vectors = load_normalized(args.small)
    large_vectors = load_normalized(args.large)
    sample_size = small_vectors.shape[0]

    compactness_null, shift_null, reference_centroid = bootstrap_null(
        large_vectors, sample_size, args.bootstrap, rng,
    )

    observed_compactness = compactness(small_vectors)
    observed_shift = centroid_shift(small_vectors, reference_centroid)

    p_compactness = empirical_p_value(observed_compactness, compactness_null)
    p_shift = empirical_p_value(observed_shift, shift_null)

    result = {
        "small_file": args.small,
        "large_file": args.large,
        "sample_size": sample_size,
        "n_bootstrap": args.bootstrap,
        "observed_compactness": observed_compactness,
        "null_compactness_mean": float(compactness_null.mean()),
        "null_compactness_std": float(compactness_null.std()),
        "p_value_compactness": p_compactness,
        "observed_centroid_shift": observed_shift,
        "null_shift_mean": float(shift_null.mean()),
        "null_shift_std": float(shift_null.std()),
        "p_value_shift": p_shift,
    }

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(result.keys()))
        writer.writeheader()
        writer.writerow(result)

    print(f"\u2713 Saved results to '{args.output}'")
    print(f"Compactness   : observed={observed_compactness:.4f}, "
          f"random-subsample mean={compactness_null.mean():.4f} "
          f"(\u00b1{compactness_null.std():.4f}), p={p_compactness:.4f}")
    print(f"Centroid shift: observed={observed_shift:.4f}, "
          f"random-subsample mean={shift_null.mean():.4f} "
          f"(\u00b1{shift_null.std():.4f}), p={p_shift:.4f}")


if __name__ == "__main__":
    main()