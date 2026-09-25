"""Paired clue-score and top-k coverage experiment for vector norms.

Uses identical sampled boards and target labels for all four variants:
cosine, original dot product, globally shuffled norms, and norms shuffled
only among words in the same centroid-alignment quantile bin. The number
of distinct top-k sets is counted across *all* candidate clues, ignoring
the order within each set. This measures which possible k-word groups
can precede the first wrong board word.

    python stats/compare_norm_rankings.py --trials 1000 --seed 1024

Output is a summary CSV; paired score differences and their standard
errors use per-trial differences, not separate estimates of two means.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from analyze_norm_geometry import (
    DEFAULT_FILES, display_path, load_vectors, norm_variants,
)

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "norm_ranking_results.csv"
BOARD_SIZE = 25
CORRECT_WORDS = 8
MAX_K = 5


def trial_stats(order: np.ndarray, correct: np.ndarray) -> tuple[int, np.ndarray]:
    """Best score and distinct unordered top-1,...,top-5 sets."""
    ranked_correct = correct[order]
    best_score = int(np.argmax(~ranked_correct, axis=1).max())

    # A 25-bit mask uniquely identifies an unordered subset of the board.
    bits = np.left_shift(np.uint32(1), order[:, :MAX_K].astype(np.uint32))
    prefix_masks = np.bitwise_or.accumulate(bits, axis=1)
    unique_counts = np.array(
        [np.unique(prefix_masks[:, k]).size for k in range(MAX_K)],
        dtype=np.int32,
    )
    return best_score, unique_counts


def analyze_file(
    path: Path, trials: int, seed: int, bins: int,
) -> list[dict]:
    raw = load_vectors(path)
    n = len(raw)
    if n < BOARD_SIZE:
        raise ValueError(f"{path}: need at least {BOARD_SIZE} words to draw a board")
    unit, norms, _, shuffled, within_bin = norm_variants(raw, seed, bins)
    # A candidate clue's own norm is constant across board words, so it
    # cancels from its ranking. Multiply the cosine matrix by board norms.
    unit = unit.astype(np.float32)
    weights = {
        "cosine": np.ones(n, dtype=np.float32),
        "dot_original": norms.astype(np.float32),
        "dot_shuffled": shuffled.astype(np.float32),
        "dot_within_centroid_bin": within_bin.astype(np.float32),
    }
    scores = {name: np.empty(trials, dtype=np.int16) for name in weights}
    top_counts = {
        name: np.empty((trials, MAX_K), dtype=np.int32) for name in weights
    }
    # Separate board generator: changing the shuffle implementation does
    # not change which boards and target labels are sampled.
    rng = np.random.default_rng(seed + 1)
    for t in range(trials):
        board = rng.choice(n, size=BOARD_SIZE, replace=False)
        correct = np.zeros(BOARD_SIZE, dtype=bool)
        correct[rng.choice(BOARD_SIZE, size=CORRECT_WORDS, replace=False)] = True
        cosine_matrix = unit @ unit[board].T
        for name, word_norms in weights.items():
            order = np.argsort(
                -(cosine_matrix * word_norms[board]), axis=1
            )
            scores[name][t], top_counts[name][t] = trial_stats(order, correct)

    result = []
    for name in weights:
        arr = scores[name].astype(np.float64)
        difference = arr - scores["cosine"]
        row = {
            "file": display_path(path),
            "variant": name,
            "n_words": n,
            "n_trials": trials,
            "seed": seed,
            "centroid_bins": bins,
            "mean_score": float(arr.mean()),
            "std_score": float(arr.std()),
            "mean_difference_vs_cosine": float(difference.mean()),
            "se_difference_vs_cosine": float(
                difference.std(ddof=1) / np.sqrt(trials)
            ) if trials > 1 else float("nan"),
        }
        row.update({
            f"mean_distinct_top_{k}_sets": float(top_counts[name][:, k - 1].mean())
            for k in range(1, MAX_K + 1)
        })
        result.append(row)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", type=Path, nargs="+", default=DEFAULT_FILES)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--bins", type=int, default=10,
                        help="Number of equal-frequency centroid-alignment bins")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.trials <= 0:
        parser.error("--trials must be positive")
    if args.bins < 2:
        parser.error("--bins must be at least 2")

    rows = []
    for path in args.files:
        rows.extend(analyze_file(path, args.trials, args.seed, args.bins))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {args.output}")
    for row in rows:
        print(f"{row['file']} {row['variant']}: mean score="
              f"{row['mean_score']:.3f}, distinct top-4="
              f"{row['mean_distinct_top_4_sets']:.1f}")


if __name__ == "__main__":
    main()
