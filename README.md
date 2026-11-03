# Climate Risk and Health Prediction

My solution for the [Zindi Climate Risk and Health Prediction Challenge](https://zindi.world/competitions/climate-risk-health-prediction-challenge).
The task was to predict whether a recorded death belonged to the supplied
climate-sensitive category. There are 3,146 training records and 1,030 test records.

Start with [the solution notebook](notebooks/solution.ipynb) for the EDA, features and model.

## Results

| Result | Score or rank |
| --- | ---: |
| Public score for the six-month rainfall submission | 0.842894491 |
| Private score shown on the account leaderboard | 0.821546589 |
| Private rank in the shared leaderboard result | 109, provisional |

These values are from the reported competition result. The rank has not been
refreshed here. The private leaderboard displays the best of two selected
submissions; the account result alone does not identify which CSV earned that score.

The metric is **60% F1 + 40% ROC-AUC**. These scores are not accuracy.
Binary predictions use the required threshold of 0.5.

## Approach

The baseline uses age, age groups, gender, zone, year, cyclic month features and
age-group interactions with a July 2017 period indicator. These make 23 columns.
A second branch adds rainfall over the six complete months before the record's month.

Each branch fits one Random Forest and one HistGradientBoosting model:

```text
baseline = 0.60 * RF_base + 0.40 * HGB_base
rainfall = 0.60 * RF_rain + 0.40 * HGB_rain
prediction = 0.80 * baseline + 0.20 * rainfall
```

All four models are fitted on the full training set for the final predictions.
CHIRPS v3 rainfall is the only external data used by this solution.

## Validation

Five-fold validation keeps each location together. Different location names
sharing coordinates are also kept in the same group. The fixed recipe was checked
on four fold assignments, using 38 connected geographic groups.

| Model | Mean geographic CV score |
| --- | ---: |
| RF/HGB baseline | 0.821459 |
| Final six-month rainfall blend | 0.821338 |

The rainfall blend improved the reported public score, but it did not consistently
improve geographic validation. [The experiment summary](reports/experiments.md)
records the other models and features tried.

## Run

Checked on Windows with Python 3.12.14. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

Follow [the data instructions](data/README.md), then run:

```powershell
.\.venv\Scripts\python.exe -m src.prepare_climate --data-dir data
.\.venv\Scripts\python.exe -m src.train --data-dir data --model-dir models
.\.venv\Scripts\python.exe -m src.predict --data-dir data --model-dir models --output outputs/submission.csv
```

Alternatively, select `.venv` as the notebook's Python environment and run
`notebooks/solution.ipynb` from top to bottom. The notebook also runs validation.

To run validation separately:

```powershell
.\.venv\Scripts\python.exe -m src.validate --data-dir data --output-dir outputs/validation
```

`--data-dir` can point to another folder. Prepare the complete Train/Test rainfall
cache before training; prediction checks that the cache has not changed.

## Reproduction

The cleaned code reproduced all 1,030 original binary predictions. The largest
probability difference was 4.44e-16. The original geographic fold assignments and
baseline/final-blend CV scores also matched.

[Verification report](reports/verification.md) · [Dependency details](reports/dependencies.md)

## Files

| Location | Purpose |
| --- | --- |
| `notebooks/solution.ipynb` | Explanation, EDA and executed results |
| `src/features.py`, `src/models.py` | Features, model settings and blend |
| `src/prepare_climate.py` | Rainfall download and cache preparation |
| `src/train.py`, `src/predict.py` | Full-data fitting and prediction |
| `src/validate.py`, `src/verify.py` | Geographic CV and reproduction checks |
| `reports/` | Experiment results, verification and dependency notes |
| `data/README.md` | Input files and source attribution |

## Limits

The July 2017 boundary came from training-data exploration. Repeated experiments
reused the same labels, so validation does not remove all model-selection bias.
CHIRPS is retrospective; this project does not demonstrate real-time forecasting
or establish that a weather exposure caused an individual death.
