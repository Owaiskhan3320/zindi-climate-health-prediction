# Climate Risk and Health Prediction

My six-month rainfall solution for the [Zindi Climate Risk and Health Prediction Challenge](https://zindi.world/competitions/climate-risk-health-prediction-challenge).

Start with [the solution notebook](notebooks/solution.ipynb) for the data analysis, features and models.

## The problem

Changes in rainfall and temperature can affect people's health. This competition
explores whether information about a person, their location and the surrounding
climate can help identify patterns in past death records.

Each row in the dataset represents a recorded death. It includes details such as
the person's age, gender, location and date of death.

The training data also contains a label called `is_climate_sensitive`:

- **1:** The death belongs to the competition's climate-sensitive category.
- **0:** The death does not belong to that category.

Here, climate-sensitive refers to health outcomes that may be influenced by
climate conditions. The labels are supplied by the competition.

My task was to learn from **3,146 labelled records** and predict the missing labels
for **1,030 test records**. For each test record, the model produces a probability
and a yes-or-no prediction.

The aim is to help understand patterns in climate-related health risks. These
predictions alone cannot establish that weather caused a particular death.

## Results

| Result | Score or rank |
| --- | ---: |
| Public score | 0.842894491 |
| Private score | 0.821546589 |
| Private rank | 109, provisional |

These are the reported results for my six-month rainfall submission. The rank
remains marked provisional while the competition results are under review.

The competition score combines **60% F1 and 40% ROC-AUC**. It is not an accuracy
percentage. Probabilities of 0.5 or higher receive label 1; lower probabilities
receive label 0, following the required threshold.

## My solution

My highest-scoring submission combined Random Forest and HistGradientBoosting
with six months of rainfall data from [CHIRPS v3](https://www.chc.ucsb.edu/data/chirps3).
CHIRPS is the only external dataset used in this solution.

For each record, I calculated the total rainfall at that location during the six
complete months before its month. For example, a death recorded in July uses
rainfall from January through June.

I trained two pairs of models:

- A pair using age, age groups, gender, zone and date-based features.
- A second pair using the same features plus the six-month rainfall total.

The date-based features include year, month and an indicator for records from
July 2017 onward. I also combined that indicator with the age groups. This gives
the baseline 23 features and the rainfall models 24 features.

Coordinates are used to find rainfall and create validation groups. They are not
direct model inputs.

Each pair contains one Random Forest and one HistGradientBoosting model. I
combined their predicted probabilities using these fixed weights:

```text
baseline = 0.60 * RF_base + 0.40 * HGB_base
rainfall = 0.60 * RF_rain + 0.40 * HGB_rain
prediction = 0.80 * baseline + 0.20 * rainfall
```

All four models were trained on the full training dataset before making the
final test predictions.

## How I checked the models

I kept records from the same location together during validation. Different
location names with matching coordinates were also grouped together. This checks
how the models perform on locations left out of training.

I used five-fold validation and repeated it with four different group assignments,
using 38 geographic groups.

| Model | Average validation score |
| --- | ---: |
| RF/HGB without rainfall | 0.821459 |
| Final six-month rainfall blend | 0.821338 |

The rainfall blend achieved my best public score, but it did not consistently
improve local validation. The leaderboard improvement alone does not prove that
rainfall will help on every new dataset. [The experiment summary](reports/experiments.md)
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
baseline/final-blend validation scores also matched.

[Verification report](reports/verification.md) · [Dependency details](reports/dependencies.md)

## Files

| Location | Purpose |
| --- | --- |
| `notebooks/solution.ipynb` | Explanation, data analysis and executed results |
| `src/features.py`, `src/models.py` | Features, model settings and blend |
| `src/prepare_climate.py` | Rainfall download and cache preparation |
| `src/train.py`, `src/predict.py` | Full-data fitting and prediction |
| `src/validate.py`, `src/verify.py` | Geographic validation and reproduction checks |
| `reports/` | Experiment results, verification and dependency notes |
| `data/README.md` | Input files and source attribution |

## Limits

The July 2017 feature came from exploring the training data. Repeated experiments
also used the same training labels, so some model-selection bias may remain.
The rainfall data is historical, and this project does not demonstrate real-time
forecasting.
