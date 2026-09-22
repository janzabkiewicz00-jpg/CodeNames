"""
run_all_configs.py
--------------------
Runs codenames_simulation.py's simulation for every configuration we care
about, and collects all results into codenames_results.csv:

    normalized ∈ {True, False}
    embeddings ∈ {real, random}
    metric     ∈ {cosine, euclidean}
    (play_set, general_set) ∈ {(small, small), (large, large), (small, large)}

That's 2 x 2 x 2 x 3 = 24 runs in total. All of them share a single, continuously
advancing progress bar (total = trials-per-run x number of not-yet-cached
configs), instead of a new bar per configuration -- and any configuration
already present in the CSV is skipped entirely.

Requirements:
    pip install numpy tqdm

Usage:
    python run_all_configs.py
    python run_all_configs.py --trials 5000
"""
import argparse

from tqdm import tqdm

from codenames_simulation import (
    RESULTS_CSV,
    load_cached_result,
    run_simulation,
    save_result,
)

NORMALIZATION_OPTIONS = [True, False]
EMBEDDINGS_OPTIONS = ["real", "random"]
METRICS = ["cosine", "euclidean"]
SET_COMBOS = [("small", "small"), ("large", "large"), ("small", "large")]


def all_configs(n_trials: int):
    for normalized in NORMALIZATION_OPTIONS:
        for embeddings in EMBEDDINGS_OPTIONS:
            for metric in METRICS:
                for play_set, general_set in SET_COMBOS:
                    yield {
                        "play_set": play_set,
                        "general_set": general_set,
                        "metric": metric,
                        "normalized": normalized,
                        "embeddings": embeddings,
                        "n_trials": n_trials,
                    }


def config_label(config: dict) -> str:
    normalization = "normalized" if config["normalized"] else "unnormalized"
    return (
        f"{normalization}/{config['embeddings']}/{config['metric']}/"
        f"play={config['play_set']},general={config['general_set']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Runs the Codenames simulation for every combination of "
                    "normalization mode, embeddings, metric and word-set pairing, "
                    "saving all results to a CSV.",
    )
    parser.add_argument("--trials", type=int, default=10000,
                        help="Number of simulated games per configuration")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of worker processes per run (default: os.cpu_count())")
    args = parser.parse_args()

    configs = list(all_configs(args.trials))

    # Resolve cached configs up front so the progress bar's total only
    # covers the trials that actually still need to run.
    to_run = []
    for config in configs:
        cached = load_cached_result(config)
        if cached is not None:
            tqdm.write(f"[skip, cached] {config_label(config)}: {cached}")
        else:
            to_run.append(config)

    if not to_run:
        print(f"\nNothing to do, all configurations were already cached in '{RESULTS_CSV}'.")
        return

    total_trials = args.trials * len(to_run)
    with tqdm(total=total_trials, desc="Overall progress", unit="trial") as bar:
        for config in to_run:
            label = config_label(config)
            tqdm.write(f"[running] {label}")
            histogram = run_simulation(
                config["play_set"], config["general_set"], config["metric"],
                config["normalized"], config["embeddings"], config["n_trials"],
                args.workers, progress_bar=bar,
            )
            save_result(config, histogram)
            tqdm.write(f"[done] {label}: {histogram}")

    print(f"\nAll configurations processed. Results in '{RESULTS_CSV}'.")


if __name__ == "__main__":
    main()