# Codenames Spymaster Simulation

A Python project that simulates a simplified version of the board game Codenames / Tajniacy using word embeddings.

The project tests an embedding-based clue selection strategy. For a randomly generated board, the program searches for a clue word such that as many of the nearest board words as possible are correct target words before the first wrong word appears.

## Project overview

In each simulation trial:

1. A board of 25 words is randomly sampled.
2. 8 words are selected as correct target words.
3. The remaining 17 words are treated as wrong words.
4. The program searches through a vocabulary of possible clue words.
5. For each clue word, board words are ranked by similarity to that clue.
6. The clue receives a score equal to the number of correct words that appear before the first wrong word.
7. The best clue score for the trial is saved.

The simulation repeats this process many times and stores the aggregated results in a CSV file.

## Strategy definition

For a clue word c, all board words are sorted from most similar to least similar.

Example ranking:

1. correct
2. correct
3. wrong
4. correct
5. ...

In this example, the clue score is 2, because the first wrong word appears in position 3.

The algorithm evaluates all available clue words and keeps the best score found in a given trial.

## Features

- Simplified Codenames board simulation.
- Word embedding based clue selection.
- Real embeddings and random baseline embeddings.
- Cosine similarity and Euclidean distance.
- Normalized and unnormalized embedding files.
- Small and large vocabulary configurations.
- Multiprocessing support.
- CSV result caching.
- Batch execution of multiple experiment configurations.
- Basic centroid statistics for embedding files.

## Project structure

PythonProject/
├── embeddings/
│   ├── builtin_embeddings.npz
│   ├── polish_embeddings.npz
│   ├── builtin_embeddings_unnormalize.npz
│   ├── polish_embeddings_unnormalize.npz
│   ├── random_524.npz
│   ├── random_10000.npz
│   ├── random_524_unnormalized.npz
│   └── random_10000_unnormalized.npz
│
├── stats/
│   ├── compute_centroid_stats.py
│   └── centroid_stats.csv
│
├── codenames_simulation.py
├── run_all_configs.py
├── codenames_results.csv
├── RESULTS.md
└── README.md

## Files

### codenames_simulation.py

Main simulation script.

Responsible for:

- loading selected embedding files,
- drawing random boards,
- assigning correct and wrong words,
- computing clue scores,
- selecting the best clue score for each trial,
- running multiple simulation trials,
- saving results to codenames_results.csv,
- reading cached results when the same configuration was already computed.

### run_all_configs.py

Batch runner for predefined simulation configurations.

It runs combinations of:

- normalized in {True, False}
- embeddings in {real, random}
- metric in {cosine, euclidean}
- (play_set, general_set) in {(small, small), (large, large), (small, large)}

This gives 24 configurations in total:

2 normalization modes × 2 embedding types × 2 metrics × 3 set combinations

The script uses the caching mechanism from codenames_simulation.py, so already computed configurations are skipped.

### stats/compute_centroid_stats.py

Script for computing centroid-based statistics for embedding files.

It calculates:

- mean Euclidean distance from the centroid,
- standard deviation of Euclidean distance from the centroid,
- mean cosine similarity to the centroid,
- standard deviation of cosine similarity to the centroid.

The output is saved to:

stats/centroid_stats.csv

### codenames_results.csv

CSV file containing simulation results.

Each row corresponds to one simulation configuration.

The file is used both as an output file and as a cache.

### stats/centroid_stats.csv

CSV file containing centroid statistics for embedding files.

### RESULTS.md

File intended for experiment results, analysis and conclusions.

This file is currently in progress.

### embeddings/

Directory containing .npz embedding files used by the simulation.

Expected files:

- builtin_embeddings.npz
- polish_embeddings.npz
- builtin_embeddings_unnormalize.npz
- polish_embeddings_unnormalize.npz
- random_524.npz
- random_10000.npz
- random_524_unnormalized.npz
- random_10000_unnormalized.npz

Each .npz file is expected to contain:

- words
- embeddings

where words is an array of words and embeddings is the corresponding matrix of word vectors.

## Embedding configurations

The simulation supports two vocabulary sizes:

| Name | Description |
|---|---|
| small | smaller word set |
| large | larger word set |

The simulation supports two embedding sources:

| Name | Description |
|---|---|
| real | real word embeddings |
| random | randomly generated baseline embeddings |

The simulation supports two normalization modes:

| Mode | Description |
|---|---|
| normalized | uses normalized embedding files |
| unnormalized | uses unnormalized embedding files |

## Similarity metrics

Two similarity modes are available:

| Metric | Description |
|---|---|
| cosine | ranks words using cosine similarity |
| euclidean | ranks words using Euclidean distance, where smaller distance means higher similarity |

## Installation

Create a virtual environment:

python -m venv .venv

Activate it on Windows:

.venv\Scripts\activate

Activate it on Linux/macOS:

source .venv/bin/activate

Install required packages:

pip install numpy tqdm

## Usage

Run the default simulation:

python codenames_simulation.py

Run one simulation with selected parameters:

python codenames_simulation.py --embeddings real --metric cosine --play-set small --general-set large --trials 5000

Run all predefined configurations:

python run_all_configs.py

Compute centroid statistics:

python stats/compute_centroid_stats.py

## Command-line arguments

### codenames_simulation.py

| Argument | Values | Description |
|---|---|---|
| --embeddings | real, random | Selects embedding source |
| --metric | cosine, euclidean | Selects similarity metric |
| --play-set | small, large | Selects word set used to draw the board |
| --general-set | small, large | Selects word set used as clue candidates |
| --trials | integer | Number of simulated games |
| --workers | integer | Number of worker processes |
| --no-normalize | flag | Uses unnormalized embedding files |

### run_all_configs.py

| Argument | Values | Description |
|---|---|---|
| --trials | integer | Number of simulated games per configuration |
| --workers | integer | Number of worker processes |

### stats/compute_centroid_stats.py

| Argument | Values | Description |
|---|---|---|
| --files | list of .npz files | Embedding files to process |
| --output | CSV path | Output path for centroid statistics |

## Output format

Simulation results are saved in:

codenames_results.csv

Columns:

play_set,general_set,metric,normalized,embeddings,n_trials,count_0,count_1,count_2,count_3,count_4,count_5,count_6,count_7,count_8

Column meanings:

| Column | Description |
|---|---|
| play_set | word set used to draw the board |
| general_set | word set used as possible clue words |
| metric | similarity metric |
| normalized | whether normalized embeddings were used |
| embeddings | embedding type: real or random |
| n_trials | number of simulated games |
| count_0 ... count_8 | number of trials where the best clue scored 0 ... 8 |

## Result caching

Before running a simulation, the program checks whether the same configuration is already present in codenames_results.csv.

If the result exists, it is loaded from the CSV file instead of being recomputed.

A configuration is identified by:

- play_set
- general_set
- metric
- normalized
- embeddings
- n_trials

## Results and analysis

Detailed experiment results and conclusions are intended to be described in RESULTS.md.

This file is currently in progress.

The planned analysis includes:

- comparison of real and random embeddings,
- comparison of cosine similarity and Euclidean distance,
- comparison of normalized and unnormalized embeddings,
- comparison of small and large vocabulary configurations,
- interpretation of the simulation result distributions.

## Requirements

Main dependencies:

- numpy
- tqdm

Python version:

Python 3.14+

## Notes

This project implements a simplified simulation of Codenames.

It does not include:

- opponent words,
- assassin word,
- multi-turn gameplay,
- human clue interpretation,
- validation of clue legality.

The simulation focuses on ranking board words by vector similarity to candidate clue words.