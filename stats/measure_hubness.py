"""
measure_hubness.py
---------------------
Measures "hubness" -- the tendency of a few vectors to appear
disproportionately often as a "nearest neighbor" of many other vectors, a
well-documented artifact of high-dimensional nearest-neighbor search
(Radovanovic et al., 2010) that is also known to affect NLP word
embeddings specifically (Dinu & Baroni, 2014).

For a given embeddings file, this computes, for each word i, its
k-occurrence count N_k(i): how many times word i appears among the k
nearest neighbors of some OTHER word j (i != j), across the whole
vocabulary. If hubness is present, this distribution is strongly
right-skewed -- a small number of "hub" words appear as a neighbor of a
disproportionate share of the vocabulary, while most words almost never
do (compare to a uniform N_k, which would happen if every word were
equally likely to be anyone's neighbor).

This is computed for two similarity measures on the SAME underlying
(unnormalized) vectors, to isolate how much EXTRA hubness the metric
itself introduces on top of whatever hubness the data already has:
  - cosine: vectors normalized to unit length before comparing (direction
    only)
  - dot: raw, unnormalized dot product (direction AND magnitude)

The hypothesis this tests: because dot(a,b) = ||a|| ||b|| cos(theta), a
word with an unusually large vector norm gets an artificially inflated
similarity to almost every other word, regardless of direction -- which
should make the dot-product hubness distribution MORE skewed than the
pure-cosine one, especially on real word embeddings (where vector norm
is known to correlate with word frequency/genericness, and where
compare_vector_norms.py already showed a much larger relative spread of
norms than in random embeddings) and much less so on random embeddings
(where norm is independent of everything).

Skewness of the N_k distribution is the main summary statistic (see
compute_centroid_stats.py / earlier chat discussion for the
population-skewness formula used here).

Note: this script lives in stats/. Its default embeddings/... and output
paths are resolved relative to the script's own location, so it works
whether you run it as `python stats/measure_hubness.py` from the project
root or as `python measure_hubness.py` from inside stats/.

Performance note: computes a full (n_words x n_words) similarity matrix
in memory (float32) for each metric -- fine up to a several thousand
words on a typical machine, but memory and compute both grow as O(n^2).
Use --max-words to randomly subsample a large vocabulary if needed.

Requirements:
    pip install numpy

Usage:
    python stats/measure_hubness.py
    python stats/measure_hubness.py --k 5 10 20
    python stats/measure_hubness.py --files embeddings/polish_embeddings_unnormalize.npz --max-words 5000
"""
import argparse
import csv
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"

# Unnormalized files, since the whole point is comparing cosine vs dot on
# the same raw vectors -- normalizing happens inside the script itself.
DEFAULT_FILES = [
    str(EMBEDDINGS_DIR / "builtin_embeddings_unnormalize.npz"),
    str(EMBEDDINGS_DIR / "polish_embeddings_unnormalize.npz"),
    str(EMBEDDINGS_DIR / "random_524_unnormalized.npz"),
    str(EMBEDDINGS_DIR / "random_10000_unnormalized.npz"),
]
DEFAULT_OUTPUT = str(SCRIPT_DIR / "hubness_results.csv")
DEFAULT_K_VALUES = [10]


def display_path(path: str) -> str:
    """Returns a path relative to the project root when possible, so the
    CSV stays portable across machines instead of recording an absolute,
    user-specific path."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return path


def load_raw(path: str, max_words: int | None, rng: np.random.Generator) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    embeddings = data["embeddings"].astype(np.float32)
    if max_words is not None and embeddings.shape[0] > max_words:
        idx = rng.choice(embeddings.shape[0], size=max_words, replace=False)
        embeddings = embeddings[idx]
    return embeddings


def cosine_similarity_matrix(raw: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    unit = raw / norms
    return unit @ unit.T


def dot_similarity_matrix(raw: np.ndarray) -> np.ndarray:
    return raw @ raw.T


def top_k_neighbor_counts(similarity: np.ndarray, k: int) -> np.ndarray:
    """similarity: (n, n) matrix, higher = more similar. Returns N_k, an
    (n,) array where N_k[i] = how many times word i is among the k
    nearest neighbors of some other word j."""
    n = similarity.shape[0]
    sim = similarity.copy()
    np.fill_diagonal(sim, -np.inf)  # a word is never its own neighbor

    # Unordered top-k per row is enough -- we only need set membership,
    # not the exact rank within the top-k.
    neighbor_idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]  # (n, k)

    counts = np.zeros(n, dtype=np.int64)
    np.add.at(counts, neighbor_idx.ravel(), 1)
    return counts


def skewness(x: np.ndarray) -> float:
    """Population skewness (third standardized moment)."""
    x = x.astype(np.float64)
    std = x.std()
    if std == 0:
        return 0.0
    return float(np.mean((x - x.mean()) ** 3) / std ** 3)


def analyze_file(path: str, k_values: list, max_words: int | None,
                  rng: np.random.Generator) -> list:
    raw = load_raw(path, max_words, rng)
    n = raw.shape[0]

    cosine_sim = cosine_similarity_matrix(raw)
    dot_sim = dot_similarity_matrix(raw)

    rows = []
    for k in k_values:
        for metric_name, sim in (("cosine", cosine_sim), ("dot", dot_sim)):
            counts = top_k_neighbor_counts(sim, k)
            rows.append({
                "file": display_path(path),
                "n_words": n,
                "k": k,
                "metric": metric_name,
                "mean_Nk": float(counts.mean()),
                "std_Nk": float(counts.std()),
                "max_Nk": int(counts.max()),
                "skewness_Nk": skewness(counts),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measures hubness (skewness of the k-occurrence distribution) "
                     "for cosine vs raw dot-product similarity on embedding files.",
    )
    parser.add_argument("--files", nargs="+", default=DEFAULT_FILES,
                         help="Unnormalized embedding .npz files to process")
    parser.add_argument("--k", type=int, nargs="+", default=DEFAULT_K_VALUES,
                         help="k value(s) for the k-nearest-neighbor occurrence counts")
    parser.add_argument("--max-words", type=int, default=None,
                         help="Randomly subsample to at most this many words per file "
                              "(the full n x n similarity matrix grows as O(n^2))")
    parser.add_argument("--seed", type=int, default=None,
                         help="Random seed, for reproducible subsampling")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                         help=f"Output CSV path, default: {DEFAULT_OUTPUT}")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    all_rows = []
    for path in args.files:
        all_rows.extend(analyze_file(path, args.k, args.max_words, rng))

    fieldnames = ["file", "n_words", "k", "metric", "mean_Nk", "std_Nk", "max_Nk", "skewness_Nk"]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\u2713 Saved results to '{args.output}'\n")
    for row in all_rows:
        print(f"{row['file']:<50} k={row['k']:>3} {row['metric']:<7} "
              f"mean_Nk={row['mean_Nk']:.2f} std_Nk={row['std_Nk']:.2f} "
              f"max_Nk={row['max_Nk']:>4} skewness={row['skewness_Nk']:.3f}")


if __name__ == "__main__":
    main()