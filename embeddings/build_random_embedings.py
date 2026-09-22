"""
generate_random_embeddings.py
------------------------------
Generates random embedding vectors, used as a baseline/control to compare
against the real (SUBTLEX-based) embeddings.

Running this script with no arguments produces four files, matching the
word counts of build_embeddings.py's outputs:
    random_524_unnormalized.npz
    random_524.npz               (L2-normalized)
    random_10000_unnormalized.npz
    random_10000.npz             (L2-normalized)

Requirements:
    pip install numpy

Usage:
    python generate_random_embeddings.py
    python generate_random_embeddings.py --dim 768 --seed 42
"""
import argparse

import numpy as np

DEFAULT_DIM = 384
WORD_COUNTS = (524, 10000)


def generate_random_embeddings(
    n_words: int,
    dim: int = DEFAULT_DIM,
    output_path: str | None = None,
    normalize: bool = False,
    rng: np.random.Generator | None = None,
) -> None:
    """Generates n_words random embedding vectors and saves them to .npz.

    Words are just placeholder indices ("1", "2", ...) since these
    embeddings only serve as a random baseline, not real word vectors.
    """
    if rng is None:
        rng = np.random.default_rng()

    words = [str(i) for i in range(1, n_words + 1)]
    embeddings = rng.standard_normal((n_words, dim)).astype(np.float32)

    if normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

    if output_path is None:
        output_path = f"random_{n_words}"

    np.savez_compressed(
        output_path,
        words=np.array(words, dtype=object),
        embeddings=embeddings,
    )
    print(f"\u2713 Saved '{output_path}.npz' \u2014 {n_words} words, dim={dim}, "
          f"normalized={normalize}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generates random embedding baselines for comparison "
                     "against the real Polish word embeddings.",
    )
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM,
                         help=f"Embedding vector dimension, default: {DEFAULT_DIM}")
    parser.add_argument("--seed", type=int, default=None,
                         help="Random seed, for reproducible output")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    for n_words in WORD_COUNTS:
        generate_random_embeddings(
            n_words, dim=args.dim, output_path=f"random_{n_words}_unnormalized",
            normalize=False, rng=rng,
        )
        generate_random_embeddings(
            n_words, dim=args.dim, output_path=f"random_{n_words}",
            normalize=True, rng=rng,
        )


if __name__ == "__main__":
    main()