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

## Why real dot product outperforms cosine in this experiment

For a fixed clue vector `a` and board word vector `b`, the dot-product score is `||a|| ||b|| cos(a,b)`. The clue's own norm `||a||` multiplies every board-word score by the same positive number, so it **cannot change their ordering** for that clue. The difference from cosine comes from the norms of the *board words*. A shuffled-norm experiment is therefore a way to test whether **which word has which norm** matters.

### Norms are tied to direction

The new [norm geometry analysis](stats/norm_geometry_results.csv) defines a word's centroid alignment as the cosine of its unit vector with the centroid of all unit vectors. For the 10,000 real words, Pearson's correlation between **raw norm** and **centroid alignment** is **−0.878**. For the random vectors it is approximately **0.0005**. Real words pointing closer to the dominant direction tend to have shorter vectors; words pointing farther from it tend to have longer ones. The relative standard deviation of norms (CV) is also **0.250** for real versus **0.036** for random embeddings.

To measure alignment while retaining the effect of lengths, the same script computes a dimensionless *dot alignment index*:

```text
A_dot = mean(v_i · v_j for i != j) / mean(v_i · v_i).
```

For unit vectors this becomes the mean pairwise cosine; a value near zero means there is little average shared dot product between different vectors. It is a measure of **global alignment**, not a complete measure of all ranking behavior.

| 10,000-word set | Norm/centroid correlation | Mean dot between different words | Mean self dot | Dot alignment index |
| --- | ---: | ---: | ---: | ---: |
| Real, original norms | −0.878 | 7.114 | 17.785 | **0.400** |
| Real, norms shuffled globally | −0.001 | 7.909 | 17.785 | **0.445** |
| Real, norms shuffled within centroid-alignment bins | −0.866 | 7.124 | 17.785 | **0.401** |
| Real, unit vectors (cosine reference) | undefined | 0.473 | 1.000 | **0.473** |
| Random, raw vectors | ≈ 0 | ≈ 0 | 383.636 | **≈ 0** |

The original assignment of norms **reduces the real vectors' shared dot-product component** relative to both the unit-vector reference and a global norm shuffle. The index remains **0.400**, so the real vectors are still strongly aligned under dot product. Rescaling does not change their directions or the mean pairwise cosine. Nor does a lower dot alignment index imply that all forms of anisotropy or hubness decrease.

### Paired simulation and available top-word sets

The existing [norm-shuffle experiment](stats/shuffle_norms_results.csv) found mean scores of **3.9826** for cosine, **4.3946** for original dot product, and **2.8898** for dot product after globally shuffling real norms, on the same 5,000 boards. The shuffle keeps the full distribution of norm lengths and every direction but breaks their association. Thus **variation in lengths alone does not explain the improvement**.

The new [paired ranking experiment](stats/norm_ranking_results.csv) repeats the comparison on **1,000 shared boards per embedding source**, with seed `1024`. It also shuffles norms only among words in the same one of ten centroid-alignment bins. This approximately preserves the norm/alignment relationship while changing the assignment among similarly positioned words. For every board and metric, it counts how many *distinct unordered groups* can occupy the first `k` ranks as the clue ranges over all 10,000 candidates.

| Real-vector ranking | Mean best score | Difference from cosine (paired SE) | Distinct top-4 sets | Distinct top-5 sets |
| --- | ---: | ---: | ---: | ---: |
| Cosine | 3.958 | reference | 626.7 | 811.9 |
| Original dot product | **4.420** | **+0.462 (0.050)** | **1,066.5** | **1,666.5** |
| Dot product, globally shuffled norms | 2.940 | −1.018 (0.042) | 203.9 | 290.7 |
| Dot product, shuffled within alignment bins | 4.363 | +0.405 (0.049) | 1,036.7 | 1,626.5 |

A fixed group of four board words has only `C(8,4) / C(25,4) ≈ 0.55%` chance of containing four targets under the random target assignment. Access to more different top-four groups gives the clue search more chances to find an all-target group. The groups overlap, so their count is **not** a count of independent opportunities and does not determine the score by itself.

One plausible mechanism is that short, centrally aligned words are less likely to rank highly for *many* clues under dot product, while a longer peripheral word can rank highly when a clue points toward it. A central **wrong** word is then less apt to block all candidate clues, although a central **correct** word is downweighted as well. The paired results show the **net** change: original norms make substantially more four- and five-word groups attainable. With norms assigned to unrelated directions, the common component grows and attainable groups collapse. Shuffling within alignment bins retains most of the advantage. These interventions support the norm/direction association as an important factor; they do not prove it is the sole factor.

### Why random dot product and cosine are close

For the same *raw* random file, the paired experiment normalizes those vectors for cosine. This avoids the separate-random-files issue in the main 24-run comparison. Random norms have much smaller relative spread and essentially no association with centroid alignment:

| Random-vector ranking | Mean best score | Difference from cosine (paired SE) | Distinct top-4 sets |
| --- | ---: | ---: | ---: |
| Cosine | 5.955 | reference | 6,716.8 |
| Original dot product | 5.960 | +0.005 (0.009) | 6,697.4 |
| Dot product, globally shuffled norms | 5.966 | +0.011 (0.009) | 6,696.8 |
| Dot product, shuffled within alignment bins | 5.953 | −0.002 (0.009) | 6,697.0 |

These small paired score differences are consistent with no meaningful improvement from the norm assignments in this random dataset.

### Hubness is a separate, mixed effect

The [hubness measurements](stats/hubness_results.csv) show that dot product makes some real words extraordinarily frequent among other words' ten nearest neighbors: the skewness of the neighbor-occurrence counts is **39.95** for raw dot product versus **4.10** for cosine on the real 10,000-word set. Thus the lower global dot alignment index does **not** mean the dot-product neighborhood structure is uniformly less concentrated.

In the [hub-on-board analysis](stats/hub_effect_results.csv), a top-1%-by-dot-hubness word appearing **only among targets** accompanies a mean score of **5.07** (294 boards), while one appearing **only among non-targets** accompanies a mean score of **3.75** (719 boards). These groups differ in their boards and cannot establish a causal net benefit from hubs. Hubs can help or hurt depending on their randomly assigned label. Notably, the paired ranking experiment finds **fewer** distinct top-one winners under original dot product (22.45 on average versus 25.00 for cosine), alongside **more** distinct top-four and top-five groups. The availability of multiword prefixes is more directly connected to this experiment's score than the number of possible single-word winners.
