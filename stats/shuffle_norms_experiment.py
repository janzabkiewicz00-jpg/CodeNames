"""
shuffle_norms_experiment.py
------------------------------
Control experiment testing WHY the dot-product metric outperforms true
cosine similarity on real embeddings -- specifically, whether it's the
SPREAD of vector norms itself that matters, or WHICH words have which
norms (i.e. a real correlation between a word's norm and its role in
the semantic space, such as hub words tending to have large norms).

Method: take real (unnormalized) embeddings, keep every word's
DIRECTION (unit vector) exactly as it is, but randomly SHUFFLE the
vector norms among words. This produces a vector set whose norm
distribution (mean, std, CV) is IDENTICAL to the original -- so
compare_vector_norms.py would report the exact same numbers -- but the
pairing between a specific word and its norm is now random and
uncorrelated with anything semantic.

Then the same paired best-clue simulation is run for FIVE variants on the
exact same set of trials, so they're all directly comparable:
  - cosine             : true cosine similarity (unaffected by norm at
                          all -- included as the reference baseline)
  - dot_original        : raw dot product on the real, unshuffled vectors
  - dot_shuffled         : raw dot product on direction-preserved,
                          norm-shuffled vectors
  - euclidean_original   : Euclidean distance on the real, unshuffled
                          vectors
  - euclidean_shuffled   : Euclidean distance on direction-preserved,
                          norm-shuffled vectors

The Euclidean pair is included because the sign of the norm term flips
between the two metrics: dot(a,b) = ||a|| ||b|| cos(theta) means a large
||b|| INCREASES a board word's similarity, while Euclidean's
d^2(a,b) = ||a||^2 + ||b||^2 - 2 a.b means a large ||b|| INCREASES a
board word's distance, i.e. DECREASES its apparent closeness. If large
vector norm really does correlate with being a directional "hub" (see
the cosine-vs-dot hubness results in measure_hubness.py), dot product
would amplify that hub signal, while Euclidean would actively suppress
it -- so the shuffle experiment may reveal the opposite pattern for
Euclidean compared to dot.

How to read the result:
  - If dot_shuffled's mean score drops back down close to cosine's
    (losing most of dot_original's advantage), the advantage comes from
    WHICH words have large/small norms -- a real correlation with
    semantic structure (e.g. hub words specifically tending to have
    large norms) -- since shuffling destroys that correlation while
    leaving the raw amount of norm variance completely unchanged.
  - If dot_shuffled keeps most of dot_original's advantage over cosine,
    the norm SPREAD alone (regardless of which word has which norm) is
    sufficient to explain the effect.

Note: this script lives in stats/ and imports best_clue_score directly
from compare_metrics_experiment.py, so that file must be present in the
same folder. It resolves its own default embeddings/... and output
paths relative to its own location, so it works whether you run it as
`python stats/shuffle_norms_experiment.py` from the project root or as
`python shuffle_norms_experiment.py` from inside stats/.

Requirements:
    pip install numpy tqdm

Usage:
    python stats/shuffle_norms_experiment.py
    python stats/shuffle_norms_experiment.py --file embeddings/builtin_embeddings_unnormalize.npz --trials 10000
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from tqdm import tqdm

from compare_metrics_experiment import best_clue_score

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"

DEFAULT_FILE = str(EMBEDDINGS_DIR / "polish_embeddings_unnormalize.npz")
DEFAULT_OUTPUT = str(SCRIPT_DIR / "shuffle_norms_results.csv")
DEFAULT_TRIALS = 5000

BOARD_SIZE = 25
CORRECT_WORDS = 8

VARIANTS = ("cosine", "dot_original", "dot_shuffled", "euclidean_original", "euclidean_shuffled")


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


def shuffle_norms(raw: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Keeps each word's direction (unit vector) fixed but randomly
    permutes the vector norms among words -- the norm distribution stays
    identical, only the pairing between a specific word and its norm is
    randomized."""
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    unit = raw / norms
    shuffled_norms = rng.permutation(norms)
    return unit * shuffled_norms


def run_paired_trials(candidate_sets: dict, n_trials: int, rng: np.random.Generator,
                       progress_bar: tqdm | None = None) -> dict:
    """candidate_sets: {variant_name: (vectors_array, metric_string)}, all
    vector arrays with the same number of rows. Draws ONE board per trial
    and scores it under every variant, so all variants are compared on
    the exact same games. metric_string is "cosine" (a plain dot product
    of whatever vectors are given -- true cosine only if they're already
    unit-normalized) or "euclidean", matching best_clue_score's own
    convention from compare_metrics_experiment.py."""
    n = next(iter(candidate_sets.values()))[0].shape[0]
    scores = {name: np.empty(n_trials, dtype=int) for name in candidate_sets}

    for t in range(n_trials):
        board_indices = rng.choice(n, size=BOARD_SIZE, replace=False)
        correct_local = rng.choice(BOARD_SIZE, size=CORRECT_WORDS, replace=False)
        correct_mask = np.zeros(BOARD_SIZE, dtype=bool)
        correct_mask[correct_local] = True

        for name, (candidates, metric) in candidate_sets.items():
            scores[name][t] = best_clue_score(candidates, board_indices, correct_mask, metric)

        if progress_bar is not None:
            progress_bar.update(1)

    return scores


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Control experiment: shuffles vector norms among real words "
                     "(keeping directions fixed) to test whether dot product's "
                     "advantage over cosine comes from norm SPREAD alone or from "
                     "WHICH words have large/small norms.",
    )
    parser.add_argument("--file", default=DEFAULT_FILE, help="Unnormalized embeddings .npz file")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS,
                         help=f"Number of simulated games, default: {DEFAULT_TRIALS}")
    parser.add_argument("--max-words", type=int, default=None,
                         help="Randomly subsample to at most this many words")
    parser.add_argument("--seed", type=int, default=None, help="Random seed, for reproducible output")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                         help=f"Output CSV path, default: {DEFAULT_OUTPUT}")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    raw = load_raw(args.file, args.max_words, rng)
    n = raw.shape[0]
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    normalized = raw / norms
    shuffled = shuffle_norms(raw, rng)

    # Sanity check: shuffling must preserve the norm distribution exactly
    # (same multiset of norms, just reassigned to different words).
    assert np.allclose(np.sort(np.linalg.norm(raw, axis=1)),
                        np.sort(np.linalg.norm(shuffled, axis=1)))

    candidate_sets = {
        "cosine": (normalized, "cosine"),
        "dot_original": (raw, "cosine"),
        "dot_shuffled": (shuffled, "cosine"),
        "euclidean_original": (raw, "euclidean"),
        "euclidean_shuffled": (shuffled, "euclidean"),
    }

    with tqdm(total=args.trials, desc="Simulating", unit="trial") as bar:
        scores = run_paired_trials(candidate_sets, args.trials, rng, progress_bar=bar)

    file_label = display_path(args.file)
    rows = []
    for variant in VARIANTS:
        arr = scores[variant]
        rows.append({
            "file": file_label,
            "variant": variant,
            "n_trials": args.trials,
            "mean_score": float(arr.mean()),
            "std_score": float(arr.std()),
        })

    # Paired differences against the cosine baseline (same trials, so
    # this is the precise same-games comparison, not two separate means).
    for variant in ("dot_original", "dot_shuffled", "euclidean_original", "euclidean_shuffled"):
        diff = scores[variant].astype(int) - scores["cosine"].astype(int)
        rows.append({
            "file": file_label,
            "variant": f"{variant}_minus_cosine",
            "n_trials": args.trials,
            "mean_score": float(diff.mean()),
            "std_score": float(diff.std()),
        })

    # Paired original-vs-shuffled differences within each metric family --
    # isolates exactly how much the real norm<->word pairing (as opposed
    # to norm spread alone) contributes, for dot and for Euclidean.
    for family in ("dot", "euclidean"):
        diff = scores[f"{family}_original"].astype(int) - scores[f"{family}_shuffled"].astype(int)
        rows.append({
            "file": file_label,
            "variant": f"{family}_original_minus_{family}_shuffled",
            "n_trials": args.trials,
            "mean_score": float(diff.mean()),
            "std_score": float(diff.std()),
        })

    fieldnames = ["file", "variant", "n_trials", "mean_score", "std_score"]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{n} words\n\u2713 Saved results to '{args.output}'\n")
    for row in rows:
        print(f"{row['variant']:<24} mean={row['mean_score']:.3f}  std={row['std_score']:.3f}")


if __name__ == "__main__":
    main()