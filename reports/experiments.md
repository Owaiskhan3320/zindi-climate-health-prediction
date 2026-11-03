# Experiments

This summary uses saved experiment results. The repository implements the selected
RF/HGB and six-month CHIRPS pipeline. Earlier training scripts and rejected-model
checkpoints remain in the separate experiment archive.

## How scores were compared

The metric is `0.60 * F1 + 0.40 * ROC-AUC`. F1 uses the fixed threshold of 0.5.
Each assignment gives every training row an out-of-fold probability. Its score
uses all those predictions together, then the assignment scores are averaged.

The main comparisons below use the same four corrected five-fold assignments:
17011, 29017, 51031 and 73013. Location aliases sharing coordinates are connected
before splitting, producing 38 training groups. Settings and weights for these
controlled experiments were fixed before their scores were inspected.

The assignments still reuse the same labels. Earlier model and feature choices,
including the July 2017 boundary, can make validation optimistic.

## Final candidates

The baseline is 60% Random Forest and 40% HistGradientBoosting on 23 age, gender,
zone and date features. Each rainfall pair uses the same settings with the stated
rainfall columns added. The outer blend is 80% baseline and 20% rainfall pair.

| Candidate | Mean geographic CV | Separate weather-area stress score |
| --- | ---: | ---: |
| RF/HGB baseline | 0.821459 | 0.823066 |
| Six-month rainfall pair alone | 0.820619 | Not recorded in this comparison |
| 80% baseline + 20% six-month rainfall pair | 0.821338 | 0.823455 |
| 80% baseline + 20% six- and twelve-month rainfall pair | 0.821678 | 0.823387 |

The weather stress assignment also holds out locations sharing the same complete
rainfall history. It is a different diagnostic, not a fifth ordinary repeat.

The six-month blend scored slightly below the baseline across the four assignments.
The six-plus-twelve-month blend had a small positive change. Neither showed strong
evidence of a meaningful generalization gain. Paired geographic bootstrap intervals
from the two later assignments included zero.

## CatBoost and XGBoost

This comparison uses the same 23-column one-hot matrix as the baseline.
CatBoost's native categorical representation was not used in this run.

- XGBoost: 500 trees, depth 3, learning rate 0.03, minimum child weight 20,
  row and column sampling 0.85, L2 regularization 10.
- CatBoost: 600 iterations, depth 4, learning rate 0.03, L2 regularization 8.

| Candidate | Mean over the same four assignments |
| --- | ---: |
| RF/HGB baseline | 0.821459 |
| XGBoost alone | 0.815151 |
| CatBoost alone | 0.812754 |
| 80% baseline + 20% XGBoost | 0.820813 |
| 80% baseline + 20% CatBoost | 0.819922 |
| 60% baseline + 20% XGBoost + 20% CatBoost | 0.818854 |

The XGBoost blend was the closest alternative, but its gains did not repeat
consistently. None passed the predeclared checks for replacing the baseline.
This concerns these settings and features; it does not establish that these
model families cannot work on the task.

## Stronger regularization and TabPFN

The stronger RF/HGB pair increased RF minimum leaf size from 30 to 45. HGB minimum
leaf size increased from 35 to 55 and L2 regularization from 1 to 5. Other settings
were unchanged.

TabPFN used the original v2 classifier checkpoint, four internal estimators and
seed 42 on CPU. It used eight columns covering the same underlying information,
with fixed categorical codes. Its representation differs from the trees' 23
one-hot and interaction columns.

| Candidate | Mean over the same four assignments |
| --- | ---: |
| Stronger RF/HGB | 0.819683 |
| 80% baseline + 20% stronger RF/HGB | 0.821244 |
| TabPFN v2 alone | 0.817996 |
| 80% baseline + 20% TabPFN v2 | 0.821216 |

None improved the overall result enough to replace the existing candidates.
TabPFN did better on the later-period subset but worse overall. No new
era-dependent blend was fitted after observing that result.

That separate experiment used Prior Labs' public weights and retained the
attribution "Built with PriorLabs-TabPFN". TabPFN is not needed by this repository.

## Earlier experiments

Earlier work included inner `GridSearchCV`, logistic models with smooth age/time
features, LightGBM, ExtraTrees and a two-hidden-layer PyTorch MLP. They were not
all evaluated on the four assignments above, so their scores are kept separate.

The older location-name folds had an issue: two names shared a coordinate but
were placed in different folds, affecting 418 records. Later validation connected
these aliases. The older results should not be described as fully holding out
physical locations.

| Experiment | Observation |
| --- | --- |
| Smooth logistic regression | A small initial blend gain did not hold consistently on corrected folds. |
| LightGBM and ExtraTrees | Tested settings did not justify replacing RF/HGB. |
| PyTorch MLP | Below the baseline on both older assignments. Not included in the selected ensemble. |
| Forward-time validation | Much weaker results that varied between time windows. This measures a different task from the challenge's mixed-year test records. |

The nested logistic follow-up on corrected groups changed the baseline by about
+0.000082 on one assignment and -0.000258 on the other. Its tuning ran inside each
outer training fold. The final pipeline does not run a new hyperparameter or
threshold search.

## Climate features tried

| Source | Features explored | Included in the selected solution? |
| --- | --- | --- |
| CHIRPS v3 | Six- and twelve-month rainfall totals | Six-month total only |
| WorldClim | Static climate normals, including BIO variables | No |
| NASA POWER | Humidity, dewpoint, soil wetness, wind, solar and recent changes | No |
| TerraClimate | Drought and water-balance indicators | No |
| MODIS land-surface temperature | Recent and rolling temperature summaries | No |
| Supplied climate_features.csv | Weather, vegetation and terrain features | No |

Adding more climate columns did not produce a consistent reason to replace the
baseline. The selected solution keeps the six-month CHIRPS total in a smaller-weight
branch. Earlier external-source comparisons used older folds and are not directly
comparable to the corrected four-assignment results.

## Leaderboard results

Reported public scores:

| Submission | Public score |
| --- | ---: |
| RF/HGB baseline | 0.840931879 |
| 80/20 six-plus-twelve-month CHIRPS blend | 0.841361384 |
| 80/20 six-month CHIRPS blend | 0.842894491 |

The shared account leaderboard showed private score **0.821546589** and rank
**109**, marked provisional. These values have not been refreshed here. The
platform displays the best private score of two selected submissions, so that
row alone does not identify the private result for each CSV.

The six-month blend's public gain influenced the choice to keep it as a submission
candidate. Its CV result does not prove it would improve a new private set.

## Evidence and reproduction

The selected pipeline has been refitted and compared with the saved six-month
submission. All binary predictions matched, and probabilities matched to
floating-point precision. The four geographic assignments and their baseline
and final-blend scores were also reproduced.

See [verification.md](verification.md) and [cv_reproduction.csv](cv_reproduction.csv).
The other experiments are summarized from saved results; they were not all rerun
during repository preparation.

Aggregate evidence is in [results/](results/). `source_manifest.json` records the
original relative filenames and hashes. That archive is not required to train
or predict with the selected pipeline.

## Lessons

- Holding out locations correctly mattered more than adding another model.
- A different model is useful only if its blend improves validation.
- Small public-score changes can depend on a few predictions.
- The best recorded public candidate was not the strongest average CV candidate.
- Reproducing the final artifact was a separate task from selecting the model.
