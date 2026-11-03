# Data setup

Run these commands from the repository root, not this folder.

## 1. Competition files

Open the [Zindi data page](https://zindi.world/competitions/climate-risk-health-prediction-challenge/data),
sign in if needed, and download:

| File | Used for |
| --- | --- |
| `Train.csv` | Features and the `is_climate_sensitive` target |
| `Test.csv` | Records to predict |
| `SampleSubmission.csv` | Output column names, IDs and row order |

Place the files here:

```text
data/
├── README.md
├── Train.csv
├── Test.csv
└── SampleSubmission.csv
```

Keep the original names and contents. The scripts expect the supplied columns,
including age, gender, zone, deathdate, coordinates and location.

The selected pipeline does not need `climate_features.csv`. Its environmental
features were used in earlier experiments described in `reports/experiments.md`.

## 2. Rainfall

The external source is **CHIRPS v3 monthly rainfall for Africa**, from the Climate
Hazards Center at UC Santa Barbara.

- [Dataset description](https://www.chc.ucsb.edu/data/chirps3)
- [Monthly raster directory](https://data.chc.ucsb.edu/products/CHIRPS/v3.0/monthly/africa/tifs/)
- [Dataset DOI](https://doi.org/10.15780/G2JQ0P)

If you already have the original `chirps_v3_monthly_point_cache.csv`, copy it into
`data/` without editing it. Otherwise, this command downloads the required monthly
rasters and samples the supplied coordinates locally:

```powershell
.\.venv\Scripts\python.exe -m src.prepare_climate --data-dir data
```

The cache has four columns:

```text
month,latitude,longitude,rainfall_mm
```

Coordinates use Python rounding to seven decimal places, matching the original
cache builder. Raster values are sampled at the nearest cell centre. Record IDs
and labels are not sent to the provider.

A fresh download covers 188 required months for the supplied files and can take
time. The original cache has 194 months because it also supported an earlier
12-month experiment. Both cover the selected six-month windows. Existing samples
are reused, and the cache is saved after each downloaded month so the command can
resume after an interruption. Rasters are read in memory rather than retained.

The feature sums six complete calendar months before each event month.
For 15 August 2020, it uses February through July 2020. The event month is excluded.
Amounts are in millimetres.

CHIRPS is retrospective. Excluding later observation months does not prove that
every final raster was available on the historical event date.

## 3. Generated files

| File | Purpose |
| --- | --- |
| `chirps_v3_monthly_point_cache.csv` | Lookup used by training and prediction |
| `chirps_features_train.csv`, `chirps_features_test.csv` | Inspectable six-month feature tables |
| `climate_manifest.json` | Cache hash, required coverage and source information |
| `chirps_downloads.jsonl` | URL and raster hash for each newly downloaded month, if any |

Training and prediction rebuild the feature from the cache; the two feature
tables are not required inputs. Prepare the complete Train/Test cache before
training. If it changes afterward, prepare the final cache and train again.

The checked reproduction used the original cache. One month was freshly downloaded
and its 55 points matched the cache; the full 188-month download was not repeated.

## 4. Optional reference submission

To compare with the original six-month submission, save a copy as
`data/reference_submission.csv`, then run after training and prediction:

```powershell
.\.venv\Scripts\python.exe -m src.verify --reference data/reference_submission.csv
```

This file is only used for verification. Training and prediction do not read it.
The notebook skips that comparison when the reference file is absent.

## Attribution and local files

Credit the competition dataset to its providers through the Zindi challenge page,
and rainfall data to the Climate Hazards Center through the links above.
Follow each source's licence terms when sharing its data; this repository does
not assign a new licence to those datasets.

Input data, reference predictions and generated files here are excluded by
`.gitignore`. Only this README belongs in the source repository.
