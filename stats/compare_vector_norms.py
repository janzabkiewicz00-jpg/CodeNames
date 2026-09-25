"""
compare_vector_norms.py
--------------------------
Tests a precondition of a hypothesis for why the "dot product" metric
(raw, un-normalized similarity -- see codenames_simulation.py's "cosine"
convention on _unnormalize.npz files) tends to outperform true cosine
similarity on real word embeddings, but performs about the same as
cosine on random embeddings.

Background / a mathematical correction to the naive hypothesis:

    dot(a, b) = ||a|| * ||b|| * cos(theta)

A candidate clue word's OWN norm ||a|| is a positive constant across all
25 board words it's compared to within a single trial, so it does NOT
change their relative ranking -- it cancels out of clue_score() entirely
(scaling every value in a row by the same positive constant preserves
argsort order). So a candidate's own vector length cannot, by itself,
affect the game score.

What CAN matter is variation in ||b|| across the 25 different BOARD
words for a fixed candidate: if board words' own vector lengths vary a
lot, the dot-product ranking is perturbed away from the pure-cosine
(pure-direction) ranking by each word's individual length -- an "extra
variable" that true cosine similarity, by construction, ignores.

This script checks only the NECESSARY precondition for that mechanism:
whether real embeddings' vector lengths are in fact more spread out
(relative to their own scale) than random embeddings' vector lengths. It
reports, for each file, the mean and standard deviation of vector norms,
and the coefficient of variation (CV = std/mean) -- a scale-independent
measure of "relative spread" that is comparable across files with very
different typical vector lengths (e.g. real vectors ~3, random vectors
~19.5).

Important caveat: a larger relative spread in real embeddings is
necessary for the "extra variable" hypothesis, but NOT sufficient proof
by itself. Board words' correct/wrong assignment is independent of word
identity each trial, so length variation that is uncorrelated with any
task-relevant property should, on average, just ADD NOISE to the
ranking -- exactly what unnormalized Euclidean showed against normalized
cosine in compare_metrics_experiment.py (extra length variation there
measurably HURT the mean score, not helped it). So if real vector
lengths do turn out more spread out, that only confirms the mechanism
has "more raw material" to work with -- it doesn't yet explain a net
*positive* effect on the game score. A follow-up causal test (e.g.
shuffling norms among real words while keeping directions fixed, then
re-running the simulation) would be needed to confirm the direction of
the effect.

Note: this script lives in stats/. Its default embeddings/... and output
paths are resolved relative to the script's own location, so it works
whether you run it as `python stats/compare_vector_norms.py` from the
project root or as `python compare_vector_norms.py` from inside stats/.

Requirements:
    pip install numpy

Usage:
    python stats/compare_vector_norms.py
    python stats/compare_vector_norms.py --files a.npz b.npz
"""
import argparse
import csv
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"

# Unnormalized files first (the ones actually relevant to the dot-product
# hypothesis); normalized files are included too, as a sanity check --
# their CV should come out essentially 0, since every vector has norm 1.
DEFAULT_FILES = [
    str(EMBEDDINGS_DIR / "builtin_embeddings_unnormalize.npz"),
    str(EMBEDDINGS_DIR / "polish_embeddings_unnormalize.npz"),
    str(EMBEDDINGS_DIR / "random_524_unnormalized.npz"),
    str(EMBEDDINGS_DIR / "random_10000_unnormalized.npz"),
    str(EMBEDDINGS_DIR / "builtin_embeddings.npz"),
    str(EMBEDDINGS_DIR / "polish_embeddings.npz"),
    str(EMBEDDINGS_DIR / "random_524.npz"),
    str(EMBEDDINGS_DIR / "random_10000.npz"),
]
DEFAULT_OUTPUT = str(SCRIPT_DIR / "vector_norm_stats.csv")


def display_path(path: str) -> str:
    """Returns a path relative to the project root when possible, so the
    CSV stays portable across machines instead of recording an absolute,
    user-specific path."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return path


def norm_stats(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    embeddings = data["embeddings"]
    norms = np.linalg.norm(embeddings, axis=1)

    mean_norm = float(norms.mean())
    std_norm = float(norms.std())
    cv = std_norm / mean_norm if mean_norm != 0 else float("nan")

    return {
        "file": display_path(path),
        "n_words": embeddings.shape[0],
        "mean_norm": mean_norm,
        "std_norm": std_norm,
        "coefficient_of_variation": cv,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compares how spread out (relative to scale) vector norms "
                     "are across embedding files -- real vs random.",
    )
    parser.add_argument("--files", nargs="+", default=DEFAULT_FILES,
                         help="Embedding .npz files to process")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                         help=f"Output CSV path, default: {DEFAULT_OUTPUT}")
    args = parser.parse_args()

    results = [norm_stats(path) for path in args.files]

    fieldnames = ["file", "n_words", "mean_norm", "std_norm", "coefficient_of_variation"]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\u2713 Saved results to '{args.output}'\n")
    for row in results:
        print(f"{row['file']:<50} n={row['n_words']:>6}  "
              f"mean_norm={row['mean_norm']:.4f}  std_norm={row['std_norm']:.4f}  "
              f"CV={row['coefficient_of_variation']:.4f}")


if __name__ == "__main__":
    main()