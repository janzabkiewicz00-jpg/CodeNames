"""
run_dimension_sweep.py
-------------------------
Dedicated runner for a specific sweep of compare_metrics_experiment.py:
vocabulary size fixed at n=10000, embedding dimensionality d swept over
2, 3, 4, 5, 10, 20, 50, 100, 300, 500, with 10,000 trials per d value.

This imports and reuses compare_metrics_experiment.py's functions
directly (same directory) rather than shelling out to it, so there's no
subprocess overhead across the 10 configurations.

Note: n=10000 with 10,000 trials per d value is a genuinely heavy
computation (roughly proportional to n * d * trials for the matrix
multiplies involved) -- expect this to take a while for the larger d
values (300, 500). There's a single shared progress bar across the whole
sweep so you can see where it's at.

Requirements:
    pip install numpy tqdm

Usage:
    python stats/run_dimension_sweep.py
    python stats/run_dimension_sweep.py --seed 42
"""
import argparse
import csv
from pathlib import Path

from tqdm import tqdm

from compare_metrics_experiment import make_vector_sets, run_paired_trials
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = str(SCRIPT_DIR / "dimension_sweep_results.csv")

N = 10000
D_VALUES = [2, 3, 4, 5, 10, 20, 50, 100, 300, 500]
N_TRIALS = 10000


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Runs the metric comparison for n={N}, d in {D_VALUES}, "
                     f"trials={N_TRIALS}, and saves the results to a CSV.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed, for reproducible output")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                         help=f"Output CSV path, default: {DEFAULT_OUTPUT}")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    total_trials = len(D_VALUES) * 2 * N_TRIALS  # x2 for cosine + euclidean
    results = []

    with tqdm(total=total_trials, desc="Dimension sweep", unit="trial") as bar:
        for d in D_VALUES:
            raw, normalized = make_vector_sets(N, d, rng)
            scores = run_paired_trials(raw, normalized, N_TRIALS, rng, progress_bar=bar)

            for metric in ("cosine", "euclidean"):
                results.append({
                    "n": N,
                    "d": d,
                    "metric": metric,
                    "n_trials": N_TRIALS,
                    "mean_score": float(scores[metric].mean()),
                    "std_score": float(scores[metric].std()),
                })

            diff = scores["cosine"].astype(int) - scores["euclidean"].astype(int)
            results.append({
                "n": N,
                "d": d,
                "metric": "cosine_minus_euclidean",
                "n_trials": N_TRIALS,
                "mean_score": float(diff.mean()),
                "std_score": float(diff.std()),
            })

    output_path = Path(args.output)
    fieldnames = ["n", "d", "metric", "n_trials", "mean_score", "std_score"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved {len(results)} rows to '{output_path}'\n")
    for row in results:
        print(f"d={row['d']:>4} {row['metric']:<24} "
              f"mean={row['mean_score']:.3f} std={row['std_score']:.3f}")


if __name__ == "__main__":
    main()