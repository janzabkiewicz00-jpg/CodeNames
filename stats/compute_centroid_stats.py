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

Usage:
    python stats/compute_centroid_stats.py
    python stats/compute_centroid_stats.py --output stats/my_results.csv
    python stats/compute_centroid_stats.py --files polish_embeddings.npz builtin_embeddings.npz
"""
import argparse
import csv
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "centroid_stats.csv"

DEFAULT_FILES = [
    "polish_embeddings.npz",
    "builtin_embeddings.npz",
    "polish_embeddings_unnormalize.npz",
    "builtin_embeddings_unnormalize.npz",
    "random_10000.npz",
    "random_524.npz",
    "random_10000_unnormalized.npz",
    "random_524_unnormalized.npz",
]


def resolve_embedding_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == EMBEDDINGS_DIR.name:
        return PROJECT_ROOT / path
    return EMBEDDINGS_DIR / path


class CentroidStats:
    """Loads an embeddings .npz file and computes centroid-distance stats."""

    def __init__(self, data_path: str | Path):
        self.data_path = resolve_embedding_path(data_path)
        data = np.load(self.data_path, allow_pickle=True)
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
        resolved_path = resolve_embedding_path(path)
        stats = CentroidStats(resolved_path)
        results.append({
            "file": str(resolved_path.relative_to(PROJECT_ROOT)),
            "n_words": stats.embeddings.shape[0],
            "mean_euclidean_distance": stats.mean_euclidean_distance(),
            "std_euclidean_distance": stats.std_euclidean_distance(),
            "mean_cosine_similarity": stats.mean_cosine_similarity(),
            "std_cosine_similarity": stats.std_cosine_similarity(),
        })
    return results


def save_results(results: list, output_path: str | Path) -> None:
    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

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
                        help="Embedding .npz files to process. Relative paths are resolved inside embeddings/")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"Output CSV path, default: {DEFAULT_OUTPUT}")
    args = parser.parse_args()

    results = compute_all(args.files)
    save_results(results, args.output)
    print(f"✓ Saved results for {len(results)} files to '{args.output}'")


if __name__ == "__main__":
    main()