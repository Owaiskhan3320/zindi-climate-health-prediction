# Dependencies

The notebook and pipeline were checked on Windows with Python 3.12.14 in a new
virtual environment. System site packages were disabled. The installed packages
passed `pip check`.

## Which file to install

| File | Use |
| --- | --- |
| `requirements.txt` | Run the rainfall, training, prediction and validation scripts |
| `requirements-notebook.txt` | Add notebook, plotting and execution packages |
| `requirements-lock.txt` | Install all package versions used in the checked environment |

For reproduction, use the lock file from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

For only the command-line pipeline, install `requirements.txt` instead.
`requirements-notebook.txt` includes that file and adds the four notebook packages
below. The lock file also pins indirect dependencies.

## Model and data packages

| Package | Version | Used for |
| --- | --- | --- |
| numpy | 2.3.5 | Numeric arrays and probability calculations |
| pandas | 3.0.1 | CSV files, dates and feature tables |
| scikit-learn | 1.9.1 | Models, grouped folds and metrics |
| scipy | 1.18.1 | Numerical dependency used by scikit-learn |
| joblib | 1.6.0 | Saving and loading fitted models |
| threadpoolctl | 3.7.0 | Limiting numerical-library threads |
| Pillow | 12.3.0 | Reading CHIRPS GeoTIFF rasters |

## Notebook packages

| Package | Version | Used for |
| --- | --- | --- |
| matplotlib | 3.11.2 | EDA charts |
| nbformat | 5.11.1 | Notebook file format |
| nbclient | 0.11.0 | Executing the notebook during verification |
| ipykernel | 7.4.0 | Notebook Python kernel |

Choose the project's `.venv` interpreter in your notebook editor. Open the
notebook from the repository root or `notebooks/` folder.

Prediction checks the six model-runtime versions saved when fitting: numpy,
pandas, scikit-learn, scipy, joblib and threadpoolctl. Keep the recorded environment
when loading an existing artifact. Only load model files you created or trust;
joblib uses pickle internally.

CatBoost, XGBoost, LightGBM, PyTorch and TabPFN were used in separate experiments.
They are not dependencies of the selected four-model pipeline.

The checked environment is Windows-specific evidence. Path arguments are portable,
but other operating systems and package versions have not been independently
verified for identical predictions.
