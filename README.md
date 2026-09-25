# Codenames Spymaster Simulation

A Python experiment about choosing a clue in a simplified game of **Codenames (Tajniacy)**. It uses Polish word embeddings to rank words on a board and measures how many target words can be reached before the first non-target. Additional experiments investigate metrics, vector norms, vocabulary size, dimensionality, and hubness.

## Simulation

Each trial samples 25 distinct board words, marks 8 as targets, and treats the remaining 17 as non-targets. For every candidate clue, the program ranks board words by vector similarity. The clue scores the number of consecutive targets before the first non-target. The trial records the **best score across all candidate clues**. For example, `target, target, non-target, target` scores **2**. Repeated trials produce a histogram of scores 0â8.

This measures the best available *single* clue under the vector-ranking rule. It does not simulate other teams, an assassin, multiple turns, or human interpretation. The code does not enforce clue legality or exclude a clue that is itself on the board. The resulting score is therefore optimistic for this specific task.

| Setting | Choices |
| --- | --- |
| Board (`--play-set`) and clue (`--general-set`) vocabularies | `small`: 524 built-in Polish words; `large`: 10,000 nouns drawn from SUBTLEX-PL |
| Embeddings | `real`: 384-dimensional vectors from `paraphrase-multilingual-MiniLM-L12-v2`; `random`: standard-normal vectors with placeholder labels |
| Vector lengths | Unit-normalized files by default; raw model vectors or unscaled random vectors with `--no-normalize` |
| Ranking | `euclidean`: increasing Euclidean distance; `cosine`: decreasing dot product |

**Metric naming:** `--metric cosine` calculates a plain dot product. This equals cosine similarity for normalized vectors; with `--no-normalize`, it is a raw, norm-sensitive dot product. Euclidean distance and cosine produce the same ranking on unit vectors (except possible numerical ties), since `||a-b||Â˛ = 2 - 2(aÂˇb)`.

The current random embedding builder samples the normalized and unnormalized files **independently**, so they are not two versions of the same random vector set. Comparisons between those files include random-sample variation. The paired experiments under `stats/` generate matched vectors where required.

## Quick start

Use Python **3.10+** from the repository root. The eight required embedding files are already committed, so the basic simulation does not need the original dataset or model download.

```bash
python -m venv .venv
source .venv/bin/activate             # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install numpy tqdm
python codenames_simulation.py --embeddings real --metric cosine --play-set small --general-set large --trials 100 --workers 2
```

The command prints nine counts: index `i` is the number of trials whose best clue scored `i`. Without arguments, the script runs 10,000 trials with a small board and clue vocabulary, random normalized vectors, Euclidean distance, and the multiprocessing pool's default worker count. Large clue vocabularies take longer to search.

```bash
python run_all_configs.py --trials 100 --workers 2
```

The batch runner evaluates 24 configurations: two normalization modes Ă two embedding sources Ă two metrics Ă the vocabulary pairs `(small, small)`, `(large, large)`, and `(small, large)` (board, clue). Its default is **10,000 trials per configuration**; it accepts `--trials` and `--workers`.

| Single-run option | Default | Values |
| --- | --- | --- |
| `--play-set` | `small` | `small`, `large` |
| `--general-set` | `small` | `small`, `large` |
| `--embeddings` | `random` | `real`, `random` |
| `--metric` | `euclidean` | `cosine`, `euclidean` |
| `--no-normalize` | off | Use the unnormalized embedding files |
| `--trials` | `10000` | Number of boards |
| `--workers` | Pool default | Number of processes |

Run the main scripts from the repository root: their embedding paths and results CSV are relative to the current working directory.

## Files and experiments

| Script | Purpose | Committed output |
| --- | --- | --- |
| [`codenames_simulation.py`](codenames_simulation.py) | Single-configuration histogram | [`codenames_results.csv`](codenames_results.csv) |
| [`run_all_configs.py`](run_all_configs.py) | Predefined 24-configuration batch | [`codenames_results.csv`](codenames_results.csv) |
| [`stats/compute_centroid_stats.py`](stats/compute_centroid_stats.py) | Distances and similarities to the embedding centroid | [`stats/centroid_stats.csv`](stats/centroid_stats.csv) |
| [`stats/test_representatives.py`](stats/test_representatives.py) | Compare the small set's compactness and centroid shift against random subsets of the large set | [`stats/representativeness_results.csv`](stats/representativeness_results.csv) |
| [`stats/compare_metrics_experiment.py`](stats/compare_metrics_experiment.py) | Paired synthetic-data trials: true cosine on unit vectors versus Euclidean on raw vectors, across vocabulary sizes and dimensions | [`stats/metric_comparison_results.csv`](stats/metric_comparison_results.csv) |
| [`stats/run_dimension_sweep.py`](stats/run_dimension_sweep.py) | Fixed 10,000-word synthetic vocabulary; 10 dimensions and 10,000 trials per dimension | [`stats/dimension_sweep_results.csv`](stats/dimension_sweep_results.csv) |
| [`stats/compare_vector_norms.py`](stats/compare_vector_norms.py) | Mean, standard deviation, and coefficient of variation of vector norms | [`stats/vector_norm_stats.csv`](stats/vector_norm_stats.csv) |
| [`stats/measure_hubness.py`](stats/measure_hubness.py) | Counts of appearances among other words' top-`k` neighbors under cosine and raw dot product | [`stats/hubness_results.csv`](stats/hubness_results.csv) |
| [`stats/hub_effect_analysis.py`](stats/hub_effect_analysis.py) | Scores grouped by whether hub words land among targets or non-targets | [`stats/hub_effect_results.csv`](stats/hub_effect_results.csv) |
| [`stats/shuffle_norms_experiment.py`](stats/shuffle_norms_experiment.py) | Paired scores before and after permuting vector lengths among word directions | [`stats/shuffle_norms_results.csv`](stats/shuffle_norms_results.csv) |

Examples (run from the repository root):

```bash
python stats/test_representatives.py --bootstrap 2000 --seed 42
python stats/compare_metrics_experiment.py --n 1000 --d 50 384 --trials 100 --seed 42
python stats/measure_hubness.py --max-words 1000 --seed 42
python stats/hub_effect_analysis.py --max-words 1000 --trials 100 --seed 42
python stats/shuffle_norms_experiment.py --max-words 1000 --trials 100 --seed 42
```

Use `--output` to avoid changing committed results: most analysis scripts overwrite their default CSV; `compare_metrics_experiment.py` appends rows. The dimension sweep has its dimensions and trial count fixed in the source and is computationally expensive. The hubness analyses form full vocabulary-by-vocabulary similarity matrices with quadratic memory use; `--max-words` subsamples the vocabulary.

The committed main CSV has 24 runs of 10,000 trials. For the real, large-board/large-clue configuration, the recorded mean best-clue scores are **3.970** (normalized dot product), **3.982** (normalized Euclidean), **4.394** (raw dot product), and **3.395** (raw Euclidean). These are descriptive results for the simplified score above. [`results_summary.xlsx`](results_summary.xlsx) is a separately maintained workbook with charts, not an automatically generated script output. [`RESULTS.md`](RESULTS.md) is currently empty.

### CSV cache

The main scripts reuse a row in `codenames_results.csv` when board vocabulary, clue vocabulary, metric, normalization flag, embedding source, and trial count all match. A row contains those settings and `count_0` through `count_8`. **Embedding contents, code version, and random seed are not part of the cache key.** Move or delete an old row when rerunning after data or algorithm changes. Main simulation trials have no seed, and multiprocessing prevents straightforward exact replay. If the existing CSV has an incompatible header, the program moves it to `codenames_results.csv.bak`.

## Rebuild embeddings (optional)

To recreate the real embeddings, install `sentence-transformers`, obtain [SUBTLEX-PL from OSF](https://osf.io/5a76z/), and put its tab-separated frequency file at `embeddings/subtlex-pl.csv` (or specify `--subtlex`). The builder reads columns `all.pos` and `spelling`, selecting `.subst.` entries. The sentence-transformer model may download on first use.

```bash
python -m pip install sentence-transformers
cd embeddings
python Build_embeddings.py
python build_random_embedings.py --seed 42
cd ..
```

The capitalization and spelling above match the **actual filenames**. Both builders write to the current directory; run them inside `embeddings/` to produce the paths expected by the simulator. The real builder also supports `--top`, `--words-file`, and `--output`; altering the word count or filename requires adjusting the hard-coded map in `codenames_simulation.py`. The random builder supports `--dim` and `--seed`. Each `.npz` file contains `words` and `embeddings` arrays; the loader uses `allow_pickle=True`, so load only trusted files.