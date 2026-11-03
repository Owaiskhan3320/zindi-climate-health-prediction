"""The four fitted models and their fixed probability weights."""

import hashlib
from importlib.metadata import version

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from .features import BASE_COLUMNS, RAIN_COLUMN, require

RF_PARAMS = dict(
    n_estimators=300, max_features="sqrt", min_samples_leaf=30,
    class_weight=None, random_state=42, n_jobs=4,
)
HGB_PARAMS = dict(
    learning_rate=0.03, max_iter=100, max_leaf_nodes=7,
    min_samples_leaf=35, l2_regularization=1.0,
    early_stopping=False, random_state=42,
)
WEIGHTS = {"rf": 0.6, "hgb": 0.4, "baseline": 0.8, "rain6": 0.2}
THRESHOLD = 0.5
FEATURE_COLUMNS = {"baseline": BASE_COLUMNS, "rain6": BASE_COLUMNS + [RAIN_COLUMN]}


def software_versions():
    packages = ["numpy", "pandas", "scikit-learn", "scipy", "joblib", "threadpoolctl"]
    return {name: version(name) for name in packages}


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recipe():
    return dict(rf_parameters=RF_PARAMS, hgb_parameters=HGB_PARAMS,
                weights=WEIGHTS, threshold=THRESHOLD, features=FEATURE_COLUMNS)


def fit_models(matrices, target):
    models = {}
    for branch, matrix in matrices.items():
        require(list(matrix.columns) == FEATURE_COLUMNS[branch], "Training schema changed.")
        models[branch] = {
            "rf": RandomForestClassifier(**RF_PARAMS).fit(matrix, target),
            "hgb": HistGradientBoostingClassifier(**HGB_PARAMS).fit(matrix, target),
        }
    return models


def predict_probabilities(models, matrices):
    components = {}
    for branch in ["baseline", "rain6"]:
        matrix = matrices[branch]
        require(list(matrix.columns) == FEATURE_COLUMNS[branch], "Prediction schema changed.")
        for family in ["rf", "hgb"]:
            model = models[branch][family]
            require(list(model.feature_names_in_) == list(matrix.columns),
                    "Model and input feature order differ.")
            require(np.array_equal(model.classes_, [0, 1]), "Unexpected model class order.")
            components[f"{branch}_{family}"] = model.predict_proba(matrix)[:, 1]
        components[branch] = (
            WEIGHTS["rf"] * components[f"{branch}_rf"]
            + WEIGHTS["hgb"] * components[f"{branch}_hgb"]
        )
    components["final"] = (
        WEIGHTS["baseline"] * components["baseline"]
        + WEIGHTS["rain6"] * components["rain6"]
    )
    for probabilities in components.values():
        require(np.isfinite(probabilities).all()
                and ((probabilities >= 0) & (probabilities <= 1)).all(),
                "Invalid model probabilities.")
    return components
