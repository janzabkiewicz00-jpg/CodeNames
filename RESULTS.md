# Results and analysis

This report uses the committed [simulation results](codenames_results.csv), the two sheets of [results_summary.xlsx](results_summary.xlsx), and the relevant files in [stats/](stats/). Each of the 24 main configurations contains **10,000 independently sampled boards**. The reported score is the largest number of target words ranked before the first non-target word by any candidate clue. There are 25 board words, including 8 targets. Tables below report the mean best-clue score and, where stated, its standard deviation across boards.

These scores describe the repository's simplified ranking experiment. In particular, a clue may also be a word on the board, and clue legality and human guessing are not modeled. The comparisons below should be read in that context.

## Metric labels and normalization

The simulator ranks words using a dot product when the option is named `cosine`. With unit-normalized vectors, the dot product is cosine similarity. With unnormalized vectors, it is a **raw dot product**, so this report calls that configuration *dot product*. Otherwise, the two `cosine` configurations would be wrongly interpreted as the same metric.

For unit vectors, Euclidean distance and cosine similarity induce the same ranking:

```text
||a - b||² = 2 - 2(a · b).
```

Accordingly, normalized Euclidean distance adds no distinct ranking strategy. Its separately simulated rows have slightly different sample means because the runs draw different random boards. For example, in the large/large real case the recorded means are **3.9696** for cosine and **3.9815** for normalized Euclidean; in the random case they are **5.9489** and **5.9408**. The underlying ranking rule is mathematically equivalent on unit vectors. The rest of this report uses normalized cosine, raw dot product, and unnormalized Euclidean distance.

## Real embeddings versus random vectors

Random vectors yield higher mean scores than real embeddings in every corresponding main configuration. The table shows the three metrics for each board/clue vocabulary pairing. Each number comes from a separate 10,000-trial run; `random − real` is the difference in means.

| Board / clues | Metric | Real mean | Random mean | Random − real |
| --- | --- | ---: | ---: | ---: |
| small / small | cosine | 3.2051 | 4.4425 | 1.2374 |
| small / small | dot product | 3.6612 | 4.4337 | 0.7725 |
| small / small | Euclidean | 2.6666 | 3.9690 | 1.3024 |
| large / large | cosine | 3.9696 | 5.9489 | 1.9793 |
| large / large | dot product | 4.3943 | 5.9243 | 1.5300 |
| large / large | Euclidean | 3.3953 | 5.2386 | 1.8433 |
| small / large | cosine | 3.8745 | 5.9349 | 2.0604 |
| small / large | dot product | 4.5749 | 5.9223 | 1.3474 |
| small / large | Euclidean | 3.0765 | 5.2326 | 2.1561 |

The real embedding space is strongly **anisotropic**: many vectors point in broadly similar directions. In [centroid_stats.csv](stats/centroid_stats.csv), the average cosine similarity to the centroid for normalized large real embeddings is **0.6874**, compared with **0.0096** for normalized large random vectors. This large difference supports a geometric explanation for why the random baseline gives the clue search more distinct directions. It does not, by itself, establish a complete causal explanation for the score gap. The random vectors carry no word meaning; a higher score here means a higher score under the simulated ranking rule, not a better clue for people.

## Vocabulary size and the board-word distribution

**Small/small is the lowest-scoring pairing in every corresponding main comparison.** With only 524 possible clues rather than 10,000, the search has far fewer opportunities to find a high-scoring clue. The large/large and small/large pairings are much closer. Their differences in means (small/large minus large/large) are:

| Metric | Real | Random |
| --- | ---: | ---: |
| Cosine | −0.0951 | −0.0140 |
| Dot product | +0.1806 | −0.0020 |
| Euclidean | −0.3188 | −0.0060 |

The random small and large files are generated **independently**, rather than one being a literal subset of the other. Both use the same standard-normal sampling rule, however: drawing 25 board vectors from either random vocabulary gives broadly similar geometry when both searches have 10,000 candidate clues. For real embeddings, the built-in list of 524 common words is hand-selected and differs from the 10,000 nouns selected from SUBTLEX-PL, so the board-word distribution can change in a systematic way. The direction of that change depends on the metric, as the table shows.

The [representativeness experiment](stats/representativeness_results.csv) compares the 524-word set against **2,000 random subsamples of 524 words** from the large vocabulary. Its centroid is **0.08481** from the large-vocabulary centroid, whereas a random subsample's average distance is **0.03076** (standard deviation **0.00265**). None of the 2,000 subsamples had a shift at least this large; the reported empirical one-sided p-value is `0/2000`, not proof of a literal probability of zero. This supports the conclusion that the small real vocabulary is not a typical random sample of the large one.

The same test does **not** find the small set more compact: its centroid norm is **0.64597**, below the random-subsample mean of **0.68787**. The compactness test was one-sided for *greater* compactness and reports p = **1.0**. Thus the evidence here is for a **shifted centroid**, not for unusually tight clustering. The remaining metric comparisons focus on **large/large** to keep the board and clue vocabularies fixed.

## Large/large metric comparison

The second sheet of the workbook presents these six configurations, relabeling unnormalized `cosine` as *dot product*. The standard deviation measures variation in the **best-clue score across sampled boards**, not uncertainty in the estimated mean.

| Embeddings | Metric | Mean score | Score standard deviation |
| --- | --- | ---: | ---: |
| Real | Dot product | 4.3943 | 1.1089 |
| Real | Cosine | 3.9696 | 1.0057 |
| Real | Unnormalized Euclidean | 3.3953 | 1.0264 |
| Random | Cosine | 5.9489 | 0.6295 |
| Random | Dot product | 5.9243 | 0.6334 |
| Random | Unnormalized Euclidean | 5.2386 | 0.9639 |

For real embeddings, the observed order is **dot product > cosine > unnormalized Euclidean**. For random vectors, **cosine and dot product are close** (a difference of 0.0246 score points across separate simulations), and both exceed unnormalized Euclidean. This closeness is empirical, not an exact mathematical equality: the random normalized and raw files are independent draws.

### Why unnormalized Euclidean distance performs worse here

For a fixed candidate clue `a` and board word `b`, ordering by squared Euclidean distance is equivalent to ordering by

```text
||a - b||² = ||a||² + ||b||² - 2(a · b).
```

The `||a||²` term is constant across the 25 board words and does not change their order. The board-word term `||b||²` does change from word to word. In the random baseline, vector lengths are sampled without regard to which board words are targets; their variation perturbs the ranking without adding information about the assigned labels. In large/large, random cosine scores have standard deviation **0.6295**, versus **0.9639** for unnormalized Euclidean scores, alongside a fall in the mean from **5.9489** to **5.2386**. This is consistent with additional variation from the sampled board vectors and their lengths. Standard deviations alone cannot isolate length as the sole cause.

The paired synthetic-vector experiment in [metric_comparison_results.csv](stats/metric_comparison_results.csv) evaluates the same boards with cosine on normalized vectors and Euclidean distance on the corresponding raw vectors. At `n = 10,000`, `d = 384`, its mean **cosine minus Euclidean** score is **+0.722** over 1,000 paired trials. The [dimension sweep](stats/dimension_sweep_results.csv) shows that this is dimension-dependent: the paired difference is **−0.708** at `d = 2`, crosses zero between `d = 4` and `d = 5`, and reaches **+0.687** at `d = 300`. Thus “Euclidean is worst” describes these high-dimensional runs, not a universal rule.

Real large/large scores vary more across boards than random ones: standard deviations are **1.0057 versus 0.6295** for cosine and **1.1089 versus 0.6334** for dot product. Their nonuniform, anisotropic geometry is a plausible contributor: a board sampled from real embeddings can encounter quite different local arrangements, while the random vectors come from a more uniform sampling rule. The centroid statistics establish the strong geometric difference; they do not prove that it alone causes the observed score variance. For unnormalized Euclidean the standard deviations are **1.0264 versus 0.9639**, a smaller gap.
