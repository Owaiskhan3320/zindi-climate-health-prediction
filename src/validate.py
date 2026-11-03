"""Evaluate the fixed recipe on the four recorded geographic fold assignments."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from .features import (
    CACHE_NAME, build_matrices, check_coordinates, load_rainfall_cache,
    load_rows, require, training_target,
)
from .models import THRESHOLD, fit_models, predict_probabilities

FOLD_SEEDS = [17011, 29017, 51031, 73013]


def geographic_groups(rows):
    """Connect location aliases at identical supplied coordinates."""
    require("location" in rows and rows.location.notna().all(), "Missing location names.")
    check_coordinates(rows)
    parent = {name: name for name in rows.location.unique()}

    def root(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    # Use the original coordinates, matching the audited grouping exactly.
    for _, block in rows.groupby(["latitude", "longitude"]):
        names = block.location.unique()
        for name in names[1:]:
            parent[root(name)] = root(names[0])
    roots = [root(name) for name in rows.location]
    numbering = {name: number for number, name in enumerate(sorted(set(roots)))}
    return np.array([numbering[name] for name in roots])


def metric_values(target, probabilities):
    require(np.isfinite(probabilities).all(), "Missing out-of-fold predictions.")
    f1 = f1_score(target, probabilities >= THRESHOLD, zero_division=0)
    auc = roc_auc_score(target, probabilities)
    return dict(f1=float(f1), auc=float(auc), score=float(0.6 * f1 + 0.4 * auc))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/validation"))
    args = parser.parse_args()
    rows = load_rows(args.data_dir / "Train.csv")
    target = training_target(rows)
    groups = geographic_groups(rows)
    matrices = build_matrices(rows, load_rainfall_cache(args.data_dir / CACHE_NAME))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assignments, score_rows = [], []
    for seed in FOLD_SEEDS:
        folds = np.full(len(rows), -1, dtype=int)
        probabilities = {name: np.full(len(rows), np.nan) for name in ["baseline", "rain6", "final"]}
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (train_indices, valid_indices) in enumerate(splitter.split(rows, target, groups)):
            require(not set(groups[train_indices]) & set(groups[valid_indices]),
                    "Training and validation geographic groups overlap.")
            train_matrix = {name: matrix.iloc[train_indices] for name, matrix in matrices.items()}
            valid_matrix = {name: matrix.iloc[valid_indices] for name, matrix in matrices.items()}
            with threadpool_limits(limits=4):
                models = fit_models(train_matrix, target[train_indices])
                predictions = predict_probabilities(models, valid_matrix)
            for name in probabilities:
                probabilities[name][valid_indices] = predictions[name]
            folds[valid_indices] = fold
            print(f"Seed {seed}: fold {fold + 1}/5 complete.", flush=True)

        require((folds >= 0).all(), "A row did not receive a validation fold.")
        assignment = pd.DataFrame({
            "ID": rows.ID, "seed": seed, "fold": folds,
            "geographic_group": groups, "target": target, **probabilities,
        })
        assignments.append(assignment)
        for name, values in probabilities.items():
            score_rows.append(dict(seed=seed, model=name, **metric_values(target, values)))
        # Save completed assignments so a partially completed run is still inspectable.
        pd.concat(assignments, ignore_index=True).to_csv(
            args.output_dir / "oof_predictions.csv", index=False
        )
        pd.DataFrame(score_rows).to_csv(args.output_dir / "scores.csv", index=False)

    scores = pd.DataFrame(score_rows)
    summary = scores.groupby("model", as_index=False)[["f1", "auc", "score"]].mean()
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))
    print("These assignments reuse the training labels; they are not an untouched test set.")


if __name__ == "__main__":
    main()
