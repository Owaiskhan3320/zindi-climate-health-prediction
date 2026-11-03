# Pipeline preparation

Prepared on 3 November 2026 from the existing frozen CHIRPS experiment and its
independent reproduction audit. The original files remain unchanged.

## Original files

Paths below are relative to the original experiment folder:

- `frozen_chirps_review/frozen_chirps_review.py`
- `frozen_chirps_review/frozen_chirps_review.ipynb`
- `chirps_extended_rainfall_experiment.py`
- `chirps_compliance_audit_20261018/final_rf_hgb_pipeline.py`
- `chirps_compliance_audit_20261018/audit_chirps_pipeline.py`
- `chirps_compliance_audit_20261018/requirements_audit.txt`

Reference submission: `submission_frozen_chirps_6m_80_20.csv` in the original
`frozen_chirps_review` folder. Its recorded SHA-256 is:

```text
e2d733f792118e18bb9f1e26800aba9e4b4c6a94f1df558e3fbfb85764e5befd
```

## Preserved choices

- 23 baseline columns, in the original order, and one additional rainfall column.
- Fixed one-hot encoding; no fitted imputation, scaling or target encoding.
- Age boundaries: `[-1, 0, 1, 5, 17, 49, 64, 111]`.
- Period boundary: `2017-07-01`.
- Rainfall: sum of six complete months ending before the event month.
- RF: 300 trees, `max_features="sqrt"`, minimum leaf size 30, no class weights.
- HGB: learning rate 0.03, 100 iterations, seven leaf nodes, minimum leaf size 35,
  L2 regularization 1, early stopping disabled.
- Model seed 42, four computation threads, default values for unspecified parameters.
- Within each pair: 60% RF and 40% HGB.
- Final blend: 80% baseline and 20% rainfall pair.
- Submission labels use a fixed 0.5 threshold; probabilities are not calibrated or rounded.
- All training rows are passed to the final fits. RF uses its normal bootstrap.

## Changes for the repository

The final recipe is split into small modules. Paths are command-line arguments.
No script injects a personal package directory into `sys.path`.

The four models are fitted directly. The original script reused saved baseline
submission probabilities after comparing them with a fresh refit; the cleaned
version predicts from its saved baseline models instead.

Training depends only on Train.csv and the climate cache. Prediction additionally
needs Test.csv and SampleSubmission.csv. There is no dependency on old OOF files,
terrain diagnostics, prior submission CSVs or the 12-month rainfall experiment.

Training saves the feature recipe, input hashes and installed dependency versions
alongside the model bundle. Prediction checks the recipe, environment and cache
before loading inputs. It aligns output rows to sample IDs and checks the CSV
after writing it.

Validation rebuilds connected location/coordinate groups and uses the recorded
five-fold seeds 17011, 29017, 51031 and 73013. It does not tune settings. Earlier
weather-history stress tests remain in the original audit, outside this small
pipeline.

## Notebook and reproduction check

The explanatory notebook is now in `notebooks/solution.ipynb`, with aggregate EDA,
synthetic examples of the feature matrix, fixed geographic validation and full-data
training. It uses the same modules as the command-line pipeline.

The first fresh-environment run found a coordinate-rounding mismatch introduced
during the refactor. `DataFrame.round` and Python `round` differ for some half-way
values in the supplied coordinates. The lookup now uses Python `round`, matching
the original cache builder. The original data and cache were not changed.

The original FTP download path also failed. The downloader now uses the provider's
HTTPS directory for the same CHIRPS v3 monthly Africa files. One month's 55 points
were downloaded and compared with the original cache. This is a spot check, not
a redownload of the full climate dataset.

The notebook and full-data fits were then rerun in a fresh environment with the
recorded model-library versions. See `verification.md` for the results.

## Still to do

- Confirm the current leaderboard status before public release.
- Create the GitHub repository and push after the final file review.

The README, data instructions and dependency notes are complete. The saved
experiment comparisons are summarized in [experiments.md](experiments.md),
with aggregate score files and source hashes in `results/`.

Repeated earlier experiments used the same labels. The date boundary was informed
by exploration. Correct geographic grouping does not remove that model-selection
uncertainty, and the retrospective climate features are not a real-time forecast.
