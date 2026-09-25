"""
hub_effect_analysis.py
-------------------------
Directly tests a specific mechanism proposed for why "hubness" helps the
dot-product metric on real embeddings: does a hub word landing among the
CORRECT board words make it easier to find a good clue (because many
candidate clue words are all pulled toward that one hub), while a hub
landing among the WRONG board words hurts almost every candidate at once
(because the hub gets ranked artificially high for everyone, regardless
of direction)?

For a chosen embeddings file and metric ("cosine" = true cosine
similarity on unit vectors, "dot" = raw dot product on unnormalized
vectors -- see measure_hubness.py / compare_vector_norms.py for the
background), this script:

  1. Runs --trials games ONCE per metric, recording each trial's
     best-clue score and which global word indices ended up correct vs
     wrong. This is independent of any hub definition.
  2. For each --hub-fraction threshold in a sweep (e.g. 0.1%, 1%, 5%,
     10% of the vocabulary), retrospectively re-categorizes those SAME
     trials by whether a hub word (top hub_fraction by k-occurrence
     count N_k) ended up among the correct words, the wrong words, both,
     or neither, and reports the mean/std score per category.

Categorizing retrospectively instead of re-running the simulation for
every threshold means the (expensive) trial loop happens only once per
metric, not once per (metric, hub_fraction) combination -- and it means
every threshold in the sweep is evaluated on the exact same set of
games, which is the cleaner comparison.

Note: this script lives in stats/ and imports directly from
compare_metrics_experiment.py (best_clue_score) and measure_hubness.py
(the similarity-matrix / N_k helpers), so both must be present in the
same folder. It resolves its own default embeddings/... and output
paths relative to its own location, so it works whether you run it as
`python stats/hub_effect_analysis.py` from the project root or as
`python hub_effect_analysis.py` from inside stats/.

Performance note: like measure_hubness.py, this builds a full
(n_words x n_words) similarity matrix once per metric to compute N_k --
grows as O(n^2). Use --max-words to subsample a large vocabulary if
needed. The per-trial simulation loop itself is cheap by comparison, and
now only runs once regardless of how many hub-fraction thresholds you
sweep over.

Requirements:
    pip install numpy tqdm

Usage:
    python stats/hub_effect_analysis.py
    python stats/hub_effect_analysis.py --file embeddings/builtin_embeddings_unnormalize.npz --trials 10000
    python stats/hub_effect_analysis.py --hub-fraction 0.001 0.005 0.01 0.02 0.05 0.1 --k 10
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from tqdm import tqdm

from compare_metrics_experiment import best_clue_score
from measure_hubness import cosine_similarity_matrix, dot_similarity_matrix, top_k_neighbor_counts

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"

DEFAULT_FILE = str(EMBEDDINGS_DIR / "polish_embeddings_unnormalize.npz")
DEFAULT_OUTPUT = str(SCRIPT_DIR / "hub_effect_results.csv")

BOARD_SIZE = 25
CORRECT_WORDS = 8

DEFAULT_K = 10
DEFAULT_HUB_FRACTIONS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
DEFAULT_TRIALS = 5000

CATEGORIES = ("hub_correct_only", "hub_wrong_only", "hub_both", "no_hub")


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
    embeddings = data["embeddings"].astype(np.float64)
    if max_words is not None and embeddings.shape[0] > max_words:
        idx = rng.choice(embeddings.shape[0], size=max_words, replace=False)
        embeddings = embeddings[idx]
    return embeddings


def hub_words_from_counts(counts: np.ndarray, hub_fraction: float) -> np.ndarray:
    """Returns the word indices among the top hub_fraction by k-occurrence
    count N_k -- i.e. the most disproportionately-frequent "neighbors" in
    the vocabulary, for one threshold."""
    n = counts.shape[0]
    n_hubs = max(1, round(n * hub_fraction))
    return np.argpartition(-counts, kth=n_hubs - 1)[:n_hubs]


def run_trials(candidates: np.ndarray, n_trials: int, rng: np.random.Generator,
                progress_bar: tqdm | None = None):
    """Runs n_trials games ONCE (no hub categorization yet). Returns
    scores (n_trials,), correct_boards (n_trials, CORRECT_WORDS) and
    wrong_boards (n_trials, BOARD_SIZE-CORRECT_WORDS) of global word
    indices, so any hub definition can be applied afterwards."""
    n = candidates.shape[0]
    scores = np.empty(n_trials, dtype=int)
    correct_boards = np.empty((n_trials, CORRECT_WORDS), dtype=int)
    wrong_boards = np.empty((n_trials, BOARD_SIZE - CORRECT_WORDS), dtype=int)

    for t in range(n_trials):
        board_indices = rng.choice(n, size=BOARD_SIZE, replace=False)
        correct_local = rng.choice(BOARD_SIZE, size=CORRECT_WORDS, replace=False)
        correct_mask = np.zeros(BOARD_SIZE, dtype=bool)
        correct_mask[correct_local] = True

        scores[t] = best_clue_score(candidates, board_indices, correct_mask, "cosine")
        correct_boards[t] = board_indices[correct_mask]
        wrong_boards[t] = board_indices[~correct_mask]

        if progress_bar is not None:
            progress_bar.update(1)

    return scores, correct_boards, wrong_boards


def categorize_trials(correct_boards: np.ndarray, wrong_boards: np.ndarray,
                       hub_words: np.ndarray) -> np.ndarray:
    """Vectorized categorization of every trial at once, for one hub
    definition. Returns an array of category labels, one per trial."""
    hub_in_correct = np.isin(correct_boards, hub_words).any(axis=1)
    hub_in_wrong = np.isin(wrong_boards, hub_words).any(axis=1)

    categories = np.full(correct_boards.shape[0], "no_hub", dtype=object)
    categories[hub_in_correct & ~hub_in_wrong] = "hub_correct_only"
    categories[~hub_in_correct & hub_in_wrong] = "hub_wrong_only"
    categories[hub_in_correct & hub_in_wrong] = "hub_both"
    return categories


def summarize(metric: str, file_label: str, hub_fraction: float, n_hubs: int,
              scores: np.ndarray, categories: np.ndarray) -> list:
    rows = []
    for category in CATEGORIES:
        arr = scores[categories == category]
        rows.append({
            "file": file_label,
            "metric": metric,
            "hub_fraction": hub_fraction,
            "n_hubs": n_hubs,
            "category": category,
            "n_trials": int(arr.size),
            "mean_score": float(arr.mean()) if arr.size else float("nan"),
            "std_score": float(arr.std()) if arr.size else float("nan"),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyzes how a hub word landing among correct vs wrong board "
                     "words affects the best-clue score, for cosine and dot product, "
                     "swept across several hub-fraction thresholds.",
    )
    parser.add_argument("--file", default=DEFAULT_FILE, help="Unnormalized embeddings .npz file")
    parser.add_argument("--k", type=int, default=DEFAULT_K,
                         help=f"k for the k-nearest-neighbor occurrence counts, default: {DEFAULT_K}")
    parser.add_argument("--hub-fraction", type=float, nargs="+", default=DEFAULT_HUB_FRACTIONS,
                         help="Top fraction(s) of words (by N_k) counted as hubs -- pass several "
                              f"to sweep, default: {DEFAULT_HUB_FRACTIONS}")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS,
                         help=f"Number of simulated games per metric, default: {DEFAULT_TRIALS} "
                              "(run once and reused across every hub_fraction)")
    parser.add_argument("--max-words", type=int, default=None,
                         help="Randomly subsample to at most this many words "
                              "(the N_k computation grows as O(n^2))")
    parser.add_argument("--seed", type=int, default=None, help="Random seed, for reproducible output")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                         help=f"Output CSV path, default: {DEFAULT_OUTPUT}")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    raw = load_raw(args.file, args.max_words, rng)
    n = raw.shape[0]
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    normalized = raw / norms

    cosine_sim = cosine_similarity_matrix(raw)
    dot_sim = dot_similarity_matrix(raw)
    cosine_counts = top_k_neighbor_counts(cosine_sim, args.k)
    dot_counts = top_k_neighbor_counts(dot_sim, args.k)

    file_label = display_path(args.file)

    # Trials run exactly once per metric, regardless of how many
    # hub_fraction thresholds are swept over afterwards.
    with tqdm(total=2 * args.trials, desc="Simulating", unit="trial") as bar:
        cosine_scores, cosine_correct, cosine_wrong = run_trials(normalized, args.trials, rng, progress_bar=bar)
        dot_scores, dot_correct, dot_wrong = run_trials(raw, args.trials, rng, progress_bar=bar)

    all_rows = []
    for hub_fraction in args.hub_fraction:
        cosine_hubs = hub_words_from_counts(cosine_counts, hub_fraction)
        dot_hubs = hub_words_from_counts(dot_counts, hub_fraction)

        cosine_categories = categorize_trials(cosine_correct, cosine_wrong, cosine_hubs)
        dot_categories = categorize_trials(dot_correct, dot_wrong, dot_hubs)

        all_rows += summarize("cosine", file_label, hub_fraction, len(cosine_hubs),
                               cosine_scores, cosine_categories)
        all_rows += summarize("dot", file_label, hub_fraction, len(dot_hubs),
                               dot_scores, dot_categories)

    fieldnames = ["file", "metric", "hub_fraction", "n_hubs", "category",
                  "n_trials", "mean_score", "std_score"]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n{n} words\n\u2713 Saved results to '{args.output}'\n")
    for row in all_rows:
        print(f"frac={row['hub_fraction']:<6} n_hubs={row['n_hubs']:>4}  "
              f"{row['metric']:<7} {row['category']:<18} n={row['n_trials']:>5}  "
              f"mean={row['mean_score']:.3f}  std={row['std_score']:.3f}")


if __name__ == "__main__":
    main()