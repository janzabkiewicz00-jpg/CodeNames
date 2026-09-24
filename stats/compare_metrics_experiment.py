"""
compare_metrics_experiment.py
--------------------------------
Standalone experiment isolating the effect of vocabulary size (n) and
embedding dimensionality (d) on how well Euclidean distance vs cosine
similarity perform as a Codenames clue-selection strategy, using purely
random (synthetic) vectors instead of real word embeddings.

This exists to test hypotheses about why Euclidean distance underperformed
cosine similarity in the main real-embeddings experiment
(codenames_simulation.py / run_all_configs.py), by varying n and d
directly, without the confound of real semantic structure.

For a fair, apples-to-apples comparison, two separate vector sets are
generated for each (n, d) configuration:
  - a "raw" (unnormalized) set, drawn from a standard multivariate normal,
    used for the Euclidean metric
  - a unit-normalized copy of that same set, used for the cosine metric,
    so cosine similarity here is TRUE cosine similarity (a dot product of
    unit vectors), not conflated with vector length.

Using the SAME normalized set for both metrics would make Euclidean
distance and cosine similarity produce IDENTICAL rankings, since for unit
vectors ||a-b||^2 = 2 - 2*cos(a,b) -- a monotonic transform of each other.
Keeping a separate raw set for Euclidean is what makes this comparison
meaningful at all, instead of testing the same ranking twice.

For each configuration and metric, --trials independent games are played:
a 25-word board is drawn from the n-word vocabulary (8 of them "correct"),
the entire vocabulary is searched for the single best clue word, and the
number of correct words it "covers" before the first wrong word is
recorded -- same rule as codenames_simulation.py's clue_score(). The
average score across all trials is reported.

Performance note: this is a single-process, fully vectorized (numpy)
implementation, not multiprocessed like codenames_simulation.py. It's
fast for exploratory sweeps (moderate n, d, a few thousand trials); for
very large n * d * trials combinations it may be worth adding
multiprocessing later.

Requirements:
    pip install numpy tqdm

Usage:
    python stats/compare_metrics_experiment.py
    python stats/compare_metrics_experiment.py --n 10000 --d 384 --trials 1000
    python stats/compare_metrics_experiment.py --n 1000 5000 10000 --d 10 50 300 --trials 1000
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = str(SCRIPT_DIR / "metric_comparison_results.csv")

BOARD_SIZE = 25
CORRECT_WORDS = 8


def make_vector_sets(n: int, d: int, rng: np.random.Generator):
    """Draws n random vectors in d dimensions from a standard normal
    distribution. Returns (raw, normalized) -- normalized is the same
    set, unit-scaled per row."""
    raw = rng.standard_normal((n, d))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    normalized = raw / norms
    return raw, normalized


def squared_euclidean_matrix(candidates: np.ndarray, board: np.ndarray) -> np.ndarray:
    """(n_candidates, n_board) matrix of squared Euclidean distances,
    computed via ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b, so the heavy
    lifting is a single matrix multiply instead of an explicit
    broadcasted subtraction (much less memory for large n)."""
    candidates_sq = np.sum(candidates ** 2, axis=1)[:, None]
    board_sq = np.sum(board ** 2, axis=1)[None, :]
    cross = candidates @ board.T
    sq_dists = candidates_sq + board_sq - 2 * cross
    np.maximum(sq_dists, 0, out=sq_dists)  # guard against tiny negative floats
    return sq_dists


def best_clue_score(candidates: np.ndarray, board_indices: np.ndarray,
                     correct_mask: np.ndarray, metric: str) -> int:
    """Vectorized equivalent of codenames_simulation.py's best_clue(): for
    every candidate word at once, ranks the board words by similarity and
    finds where the first wrong word appears in that ranking, then
    returns the best (maximum) score across all candidates."""
    board_vectors = candidates[board_indices]

    if metric == "cosine":
        sims = candidates @ board_vectors.T  # higher = more similar
    elif metric == "euclidean":
        sims = -squared_euclidean_matrix(candidates, board_vectors)  # higher = closer
    else:
        raise ValueError(f"Unknown metric: {metric}")

    order = np.argsort(-sims, axis=1)  # descending similarity, per candidate row
    ranked_correct = correct_mask[order]  # same shape, True where a correct word landed
    first_wrong = np.argmax(~ranked_correct, axis=1)  # rank of the first wrong word
    return int(first_wrong.max())


def run_paired_trials(raw: np.ndarray, normalized: np.ndarray, n_trials: int,
                       rng: np.random.Generator, progress_bar: tqdm | None = None) -> dict:
    """Runs n_trials games, evaluating BOTH metrics on the SAME board each
    trial (same board_indices / correct_mask), instead of drawing an
    independent board per metric. This pairs the comparison -- each trial
    is literally the same game scored two ways -- which removes the extra
    noise that comparing two separately-sampled sets of boards would add,
    giving a more precise estimate of the difference between metrics."""
    n = raw.shape[0]
    scores = {"cosine": np.empty(n_trials, dtype=int), "euclidean": np.empty(n_trials, dtype=int)}

    for t in range(n_trials):
        board_indices = rng.choice(n, size=BOARD_SIZE, replace=False)
        correct_local = rng.choice(BOARD_SIZE, size=CORRECT_WORDS, replace=False)
        correct_mask = np.zeros(BOARD_SIZE, dtype=bool)
        correct_mask[correct_local] = True

        scores["cosine"][t] = best_clue_score(normalized, board_indices, correct_mask, "cosine")
        scores["euclidean"][t] = best_clue_score(raw, board_indices, correct_mask, "euclidean")

        if progress_bar is not None:
            progress_bar.update(2)

    return scores


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compares Euclidean vs true-cosine clue-selection performance "
                     "on random vectors, sweeping vocabulary size (n) and dimensionality (d).",
    )
    parser.add_argument("--n", type=int, nargs="+", default=[10000],
                         help="Vocabulary size(s) to test")
    parser.add_argument("--d", type=int, nargs="+", default=[384],
                         help="Embedding dimensionality/ies to test")
    parser.add_argument("--trials", type=int, default=1000,
                         help="Number of simulated games per configuration (default: 1000)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed, for reproducible output")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                         help=f"Output CSV path (rows are appended), default: {DEFAULT_OUTPUT}")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    configs = [(n, d) for n in args.n for d in args.d]
    total_trials = len(configs) * 2 * args.trials  # x2 for cosine + euclidean

    results = []
    with tqdm(total=total_trials, desc="Overall progress", unit="trial") as bar:
        for n, d in configs:
            raw, normalized = make_vector_sets(n, d, rng)
            scores = run_paired_trials(raw, normalized, args.trials, rng, progress_bar=bar)

            for metric in ("cosine", "euclidean"):
                results.append({
                    "n": n,
                    "d": d,
                    "metric": metric,
                    "n_trials": args.trials,
                    "mean_score": float(scores[metric].mean()),
                    "std_score": float(scores[metric].std()),
                })

            # Paired difference (per-trial cosine score minus euclidean score),
            # more precise than comparing the two rows above independently
            # since it's computed from the SAME trials.
            diff = scores["cosine"].astype(int) - scores["euclidean"].astype(int)
            results.append({
                "n": n,
                "d": d,
                "metric": "cosine_minus_euclidean",
                "n_trials": args.trials,
                "mean_score": float(diff.mean()),
                "std_score": float(diff.std()),
            })

    output_path = Path(args.output)
    file_exists = output_path.exists()
    fieldnames = ["n", "d", "metric", "n_trials", "mean_score", "std_score"]
    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved {len(results)} rows to '{output_path}'\n")
    for row in results:
        print(f"n={row['n']:>6} d={row['d']:>4} {row['metric']:<9} "
              f"mean={row['mean_score']:.3f} std={row['std_score']:.3f}")


if __name__ == "__main__":
    main()