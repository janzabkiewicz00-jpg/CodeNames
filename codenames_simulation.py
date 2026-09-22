"""
codenames_simulation.py
-------------------------
Simulates a single-team round of Codenames using word embeddings as the
"spymaster": for a randomly drawn board of 25 words (8 of them "correct"),
it looks for the single clue-word whose embedding is closest to as many
correct words as possible, before being closer to any wrong word. The
result of one trial is how many correct words (0-8) that best clue can
"cover" in a row.

The simulation is repeated --trials times (default 10,000) and the counts
are collected into a histogram of length 9 (index i = number of trials
where the best possible clue covered exactly i correct words).

Results are cached in a CSV file, keyed by the run's configuration
(word sets, metric, normalization, real vs. random embeddings, number of
trials). Running the same configuration again does not recompute anything;
it just prints the cached histogram.

Requirements:
    pip install numpy tqdm

Usage:
    python codenames_simulation.py
    python codenames_simulation.py --play-set large --general-set large --metric cosine --embeddings real
    python codenames_simulation.py --embeddings real --no-normalize --trials 5000
"""
import argparse
import csv
import os
import random
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

RESULTS_CSV = "codenames_results.csv"
BOARD_SIZE = 25
CORRECT_WORDS = 8
COUNT_FIELDNAMES = [f"count_{i}" for i in range(CORRECT_WORDS + 1)]
RESULTS_FIELDNAMES = [
    "play_set", "general_set", "metric", "normalized", "embeddings", "n_trials",
] + COUNT_FIELDNAMES

# Filenames produced by build_embeddings.py / generate_random_embeddings.py,
# expected to live in the EMBEDDINGS_DIR subfolder.
EMBEDDINGS_DIR = "embeddings"


def _emb_path(filename: str) -> str:
    return os.path.join(EMBEDDINGS_DIR, filename)


EMBEDDING_FILES = {
    ("real", True): {"small": _emb_path("builtin_embeddings.npz"), "large": _emb_path("polish_embeddings.npz")},
    ("real", False): {"small": _emb_path("builtin_embeddings_unnormalize.npz"), "large": _emb_path("polish_embeddings_unnormalize.npz")},
    ("random", True): {"small": _emb_path("random_524.npz"), "large": _emb_path("random_10000.npz")},
    ("random", False): {"small": _emb_path("random_524_unnormalized.npz"), "large": _emb_path("random_10000_unnormalized.npz")},
}


class CodenamesSimulator:
    """One playable round: a board of words plus a clue vocabulary to
    search for the best possible clue word."""

    def __init__(self, board_words, board_vectors, clue_words, clue_vectors, metric):
        self.board_words = board_words
        self.board_vectors = board_vectors
        self.clue_words = clue_words
        self.clue_vectors = clue_vectors
        self.metric = metric

        self.board = []
        self.correct = set()
        self.wrong = set()

    def similarity(self, clue_word: str, board_word: str) -> float:
        clue_vec = self.clue_vectors[clue_word]
        board_vec = self.board_vectors[board_word]
        if self.metric == "cosine":
            return float(np.dot(clue_vec, board_vec))
        elif self.metric == "euclidean":
            return -float(np.linalg.norm(clue_vec - board_vec))
        raise ValueError(f"Unknown metric: {self.metric}")

    def draw_board(self) -> None:
        """Randomly draws a 25-word board, 8 of which are 'correct'."""
        self.board = random.sample(self.board_words, BOARD_SIZE)
        self.correct = set(random.sample(self.board, CORRECT_WORDS))
        self.wrong = set(self.board) - self.correct

    def clue_score(self, clue_word: str) -> int:
        """Number of correct board words that are all closer to clue_word
        than any wrong board word is (0 if the nearest board word is
        already wrong)."""
        ranked = sorted(
            (self.similarity(clue_word, board_word), board_word) for board_word in self.board
        )
        ranked.reverse()
        for rank, (_, board_word) in enumerate(ranked):
            if board_word in self.wrong:
                return rank
        return len(self.board)

    def best_clue(self):
        """Searches the whole clue vocabulary for the clue word with the
        highest score. Returns (best_word, best_score)."""
        best_word, best_score = None, -1
        for clue_word in self.clue_words:
            score = self.clue_score(clue_word)
            if score > best_score:
                best_word, best_score = clue_word, score
        return best_word, best_score


# ── Multiprocessing worker ────────────────────────────────────────────────

_game: CodenamesSimulator | None = None  # per-worker-process global


def _load_word_vectors(path: str):
    data = np.load(path, allow_pickle=True)
    words = data["words"].tolist()
    embeddings = data["embeddings"]
    return words, {word: embeddings[i] for i, word in enumerate(words)}


def _init_worker(play_set: str, general_set: str, normalized: bool, metric: str, embeddings: str) -> None:
    global _game

    files = EMBEDDING_FILES[(embeddings, normalized)]
    small_words, small_vectors = _load_word_vectors(files["small"])
    large_words, large_vectors = _load_word_vectors(files["large"])

    sets = {"small": (small_words, small_vectors), "large": (large_words, large_vectors)}
    board_words, board_vectors = sets[play_set]
    clue_words, clue_vectors = sets[general_set]

    _game = CodenamesSimulator(board_words, board_vectors, clue_words, clue_vectors, metric)


def _run_single_trial(_) -> int:
    global _game
    _game.draw_board()
    _, score = _game.best_clue()
    return score


def run_simulation(play_set: str, general_set: str, metric: str, normalized: bool,
                    embeddings: str, n_trials: int, n_workers: int | None = None,
                    progress_bar: tqdm | None = None) -> list:
    """Runs n_trials independent games in parallel and returns a length-9
    histogram of best-clue scores. If progress_bar is given (an existing
    tqdm instance), its counter is advanced instead of creating a new bar
    -- useful for sharing one continuously-moving bar across several calls."""
    own_bar = progress_bar is None
    if own_bar:
        progress_bar = tqdm(total=n_trials, desc="Trials", leave=False)

    histogram = [0] * (CORRECT_WORDS + 1)
    try:
        with Pool(
            processes=n_workers,
            initializer=_init_worker,
            initargs=(play_set, general_set, normalized, metric, embeddings),
        ) as pool:
            for score in pool.imap_unordered(_run_single_trial, range(n_trials)):
                histogram[score] += 1
                progress_bar.update(1)
    finally:
        if own_bar:
            progress_bar.close()

    return histogram


# ── Result caching ─────────────────────────────────────────────────────────

def _config_key(config: dict) -> tuple:
    return (
        config["play_set"], config["general_set"], config["metric"],
        str(config["normalized"]), config["embeddings"], str(config["n_trials"]),
    )


def _ensure_compatible_csv() -> None:
    """If codenames_results.csv exists but was written by an older version
    of this script (different columns, e.g. a single 'counts' column
    instead of count_0..count_8), move it aside instead of crashing or
    silently corrupting it with rows in the new format."""
    if not os.path.exists(RESULTS_CSV):
        return
    with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
        header = next(csv.reader(f), None)
    if header != RESULTS_FIELDNAMES:
        backup_path = RESULTS_CSV + ".bak"
        os.replace(RESULTS_CSV, backup_path)
        print(f"Note: '{RESULTS_CSV}' had an old/incompatible format; "
              f"moved it to '{backup_path}' and starting a fresh results file.")


def load_cached_result(config: dict) -> list | None:
    _ensure_compatible_csv()
    if not os.path.exists(RESULTS_CSV):
        return None
    with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row_key = (
                row["play_set"], row["general_set"], row["metric"],
                row["normalized"], row["embeddings"], row["n_trials"],
            )
            if row_key == _config_key(config):
                return [int(row[col]) for col in COUNT_FIELDNAMES]
    return None


def save_result(config: dict, histogram: list) -> None:
    _ensure_compatible_csv()
    file_exists = os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        row = {
            "play_set": config["play_set"],
            "general_set": config["general_set"],
            "metric": config["metric"],
            "normalized": config["normalized"],
            "embeddings": config["embeddings"],
            "n_trials": config["n_trials"],
        }
        row.update({col: count for col, count in zip(COUNT_FIELDNAMES, histogram)})
        writer.writerow(row)


# ── CLI ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulates Codenames rounds to measure how many words a "
                     "single embedding-based clue can cover, and caches results in a CSV.",
    )
    parser.add_argument("--play-set", choices=["small", "large"], default="small",
                         help="Word set the 25-word board is drawn from: "
                              "small = 524 words, large = 10,000 words")
    parser.add_argument("--general-set", choices=["small", "large"], default="small",
                         help="Word set the clue word is searched over")
    parser.add_argument("--metric", choices=["cosine", "euclidean"], default="euclidean",
                         help="Similarity metric used to compare clue and board words")
    parser.add_argument("--embeddings", choices=["real", "random"], default="random",
                         help="Use the real (SUBTLEX-based) embeddings or the random baseline")
    parser.add_argument("--no-normalize", action="store_true",
                         help="Use the unnormalized embedding files instead of the normalized ones")
    parser.add_argument("--trials", type=int, default=10000, help="Number of simulated games")
    parser.add_argument("--workers", type=int, default=None,
                         help="Number of worker processes (default: os.cpu_count())")
    args = parser.parse_args()

    config = {
        "play_set": args.play_set,
        "general_set": args.general_set,
        "metric": args.metric,
        "normalized": not args.no_normalize,
        "embeddings": args.embeddings,
        "n_trials": args.trials,
    }

    cached = load_cached_result(config)
    if cached is not None:
        print(f"Cached result found in '{RESULTS_CSV}' for this configuration:")
        print(cached)
        return

    histogram = run_simulation(
        args.play_set, args.general_set, args.metric, config["normalized"],
        args.embeddings, args.trials, args.workers,
    )
    save_result(config, histogram)
    print(f"Result (index = number of correct words the best clue covered), saved to '{RESULTS_CSV}':")
    print(histogram)


if __name__ == "__main__":
    main()