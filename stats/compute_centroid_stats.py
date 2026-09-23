"""
compute_centroid_stats.py
---------------------------
Computes, for each embeddings file, the mean and standard deviation of the
Euclidean distance and cosine similarity of all vectors to their centroid
(the mean vector). The mean shows how "spread out" the embeddings are on
average; the standard deviation shows how uniform that spread is (a tight
shell around the centroid vs. a mix of near-centroid and outlier words).
Used to compare the real word embeddings against the random baseline
embeddings.

Results are written to a CSV file rather than printed to the console.

Requirements:
    pip install numpy

Note: this script lives in stats/. Its default embeddings/... and output
paths are resolved relative to the script's own location (not the current
working directory), so it works whether you run it as
`python stats/compute_centroid_stats.py` from the project root or as
`python compute_centroid_stats.py` from inside stats/.

Usage:
    python stats/compute_centroid_stats.py
    python stats/compute_centroid_stats.py --output my_results.csv
    python stats/compute_centroid_stats.py --files a.npz b.npz c.npz
"""
import argparse
import csv
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"

DEFAULT_FILES = [
    str(EMBEDDINGS_DIR / "polish_embeddings.npz"),
    str(EMBEDDINGS_DIR / "builtin_embeddings.npz"),
    str(EMBEDDINGS_DIR / "polish_embeddings_unnormalize.npz"),
    str(EMBEDDINGS_DIR / "builtin_embeddings_unnormalize.npz"),
    str(EMBEDDINGS_DIR / "random_10000.npz"),
    str(EMBEDDINGS_DIR / "random_524.npz"),
    str(EMBEDDINGS_DIR / "random_10000_unnormalized.npz"),
    str(EMBEDDINGS_DIR / "random_524_unnormalized.npz"),
]
DEFAULT_OUTPUT = str(SCRIPT_DIR / "centroid_stats.csv")


class CentroidStats:
    """Loads an embeddings .npz file and computes centroid-distance stats."""

    def __init__(self, data_path: str):
        self.data_path = data_path
        data = np.load(data_path, allow_pickle=True)
        self.embeddings = data["embeddings"]
        self.centroid = self.embeddings.mean(axis=0)

    def euclidean_distances(self) -> np.ndarray:
        return np.linalg.norm(self.embeddings - self.centroid, axis=1)

    def cosine_similarities(self) -> np.ndarray:
        centroid_norm = self.centroid / np.linalg.norm(self.centroid)
        return self.embeddings @ centroid_norm

    def mean_euclidean_distance(self) -> float:
        return float(self.euclidean_distances().mean())

    def std_euclidean_distance(self) -> float:
        return float(self.euclidean_distances().std())

    def mean_cosine_similarity(self) -> float:
        return float(self.cosine_similarities().mean())

    def std_cosine_similarity(self) -> float:
        return float(self.cosine_similarities().std())


def compute_all(file_paths: list) -> list:
    results = []
    for path in file_paths:
        stats = CentroidStats(path)
        results.append({
            "file": path,
            "n_words": stats.embeddings.shape[0],
            "mean_euclidean_distance": stats.mean_euclidean_distance(),
            "std_euclidean_distance": stats.std_euclidean_distance(),
            "mean_cosine_similarity": stats.mean_cosine_similarity(),
            "std_cosine_similarity": stats.std_cosine_similarity(),
        })
    return results


def save_results(results: list, output_path: str) -> None:
    fieldnames = [
        "file", "n_words",
        "mean_euclidean_distance", "std_euclidean_distance",
        "mean_cosine_similarity", "std_cosine_similarity",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Computes mean distance-to-centroid stats for embedding files "
                     "and saves them to a CSV file.",
    )
    parser.add_argument("--files", nargs="+", default=DEFAULT_FILES,
                         help="Embedding .npz files to process")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                         help=f"Output CSV path, default: {DEFAULT_OUTPUT}")
    args = parser.parse_args()

    results = compute_all(args.files)
    save_results(results, args.output)
    print(f"\u2713 Saved results for {len(results)} files to '{args.output}'")


if __name__ == "__main__":
    main()