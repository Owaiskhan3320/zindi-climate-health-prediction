# Reproduction check

Checked on 3 November 2026 in a fresh Python 3.12.14 environment.
System site packages were disabled. The notebook dependencies and the original
model-library versions were installed from package requirements. `pip check`
reported no broken requirements.

## Results

| Check | Result |
| --- | --- |
| Notebook code cells executed | 10/10, no errors |
| Full-data training rows | 3,146 |
| Test predictions | 1,030 |
| Baseline / rainfall features | 23 / 24 |
| Largest probability difference from original | 4.44e-16 |
| Changed binary predictions | 0 |
| Submission IDs, row order and 0.5 threshold | Passed |
| Reproduction checks | 18/18 passed |
| Additional verification checks | 18/18 passed |
| Protected original files | 8, unchanged |

The CSV is numerically equivalent, not byte-for-byte identical: the serializer
preserves 17 significant digits and tiny floating-point differences remain.
The reference file SHA-256 is `e2d733f792118e18bb9f1e26800aba9e4b4c6a94f1df558e3fbfb85764e5befd`.

## Validation

All four original geographic fold assignments were reconstructed exactly. The
baseline and final-blend scores matched the saved audit results exactly.
See `cv_reproduction.csv` for the eight comparisons.

Mean score across four geographic assignments:

- Baseline: 0.821458842
- Rainfall pair alone: 0.820619499
- Final 80/20 blend: 0.821338156

Every training row received an out-of-fold prediction, with its connected
geographic group held out. These are the original assignments, not newly
independent validation data.

## Additional checks

The pipeline was fitted twice successfully. Repeated predictions differed by
at most 3.33e-16. Reordering Test rows and realigning by ID differed by
at most 2.22e-16.

Missing ages, unknown categories, non-finite probabilities, incorrect binary
labels, wrong output order and wrong feature order were rejected. Changing IDs
or target labels did not change the feature matrices. Rainfall totals were
independently reconstructed for all Train and Test rows.

The executed notebook contains aggregate summaries and synthetic feature examples.
No original record IDs or personal absolute paths were found in it. Data, model
artifacts and predictions remain in the ignored directories.

## Fixes found during verification

The first run exposed a rounding mismatch in the refactor: NumPy/pandas rounding
picked different keys from Python rounding for a few coordinates. The code now
matches the original cache builder's Python rounding. No original cache was edited.

The old FTP path failed. The downloader now uses the provider's
[HTTPS directory](https://data.chc.ucsb.edu/products/CHIRPS/v3.0/monthly/africa/tifs/).
One monthly raster (2006-10) was downloaded using the cleaned code.
All 55 points matched the saved cache within
1.42e-14 mm.

## Scope

This confirms reproduction of the original six-month submission using the saved
climate cache. It does not redownload all historical rasters, prove zero
model-selection overfitting or obtain a new leaderboard score. Training and
prediction do not read the reference submission; only verification uses it.

## Repeat the comparison

After running the notebook or the training/prediction scripts, put the original
CSV at `data/reference_submission.csv` and run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m src.verify --reference data/reference_submission.csv
```

Detailed local reports are in `outputs/reproduction_check.json`,
`outputs/final_verification.json` and `outputs/notebook_execution.json`.
