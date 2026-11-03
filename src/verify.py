"""Check a generated submission against a supplied reference CSV."""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features import (
    CACHE_NAME, build_matrices, load_rainfall_cache, load_rows,
    require, training_target,
)
from .models import FEATURE_COLUMNS, file_hash, recipe, software_versions
from .predict import check_submission
from .train import MODEL_NAME
from .validate import geographic_groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--submission", type=Path, default=Path("outputs/submission.csv"))
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=Path("outputs/reproduction_check.json"))
    args = parser.parse_args()

    train = load_rows(args.data_dir / "Train.csv")
    test = load_rows(args.data_dir / "Test.csv")
    sample = pd.read_csv(args.data_dir / "SampleSubmission.csv")
    actual = pd.read_csv(args.submission, float_precision="round_trip")
    reference = pd.read_csv(args.reference, float_precision="round_trip")
    check_submission(actual, sample)
    check_submission(reference, sample)
    require(set(actual.ID) == set(test.ID), "Submission does not contain all Test IDs.")
    reference = reference.set_index("ID").loc[actual.ID]
    difference = np.abs(actual.TargetRAUC.to_numpy() - reference.TargetRAUC.to_numpy())
    binary_changes = int(np.count_nonzero(actual.TargetF1.to_numpy() != reference.TargetF1.to_numpy()))

    cache_path = args.data_dir / CACHE_NAME
    cache = load_rainfall_cache(cache_path)
    train_matrices = build_matrices(train, cache)
    test_matrices = build_matrices(test, cache)
    bundle = joblib.load(args.model_dir / MODEL_NAME)
    metadata = bundle["metadata"]
    changed = train.copy()
    changed["is_climate_sensitive"] = 1 - training_target(train)
    changed["ID"] = [f"changed_{i}" for i in range(len(changed))]
    changed_matrices = build_matrices(changed, cache)
    groups = geographic_groups(train)

    # Independently reconstruct the rainfall total and its month membership.
    lookup = cache.set_index(["month", "latitude", "longitude"])["rainfall_mm"]
    rainfall_errors = []
    before_event = []
    for rows, matrices in [(train, train_matrices), (test, test_matrices)]:
        independent = []
        for row in rows.itertuples(index=False):
            event = pd.Timestamp(row.deathdate)
            months = pd.period_range(end=event.to_period("M") - 1, periods=6, freq="M")
            before_event.append(months[-1].end_time < event)
            independent.append(sum(float(lookup.loc[(str(month), round(row.latitude, 7),
                                                     round(row.longitude, 7))]) for month in months))
        rainfall_errors.append(float(np.max(np.abs(
            matrices["rain6"].iloc[:, -1].to_numpy() - np.asarray(independent)
        ))))

    checks = {
        "submission_format_ids_order_and_threshold": True,
        "probabilities_match_reference_within_1e_10": bool(difference.max() < 1e-10),
        "binary_predictions_match_reference": binary_changes == 0,
        "train_test_features_have_expected_order": all(
            list(train_matrices[name]) == list(test_matrices[name]) == columns
            for name, columns in FEATURE_COLUMNS.items()
        ),
        "features_do_not_use_labels_or_ids": all(
            train_matrices[name].equals(changed_matrices[name]) for name in train_matrices
        ),
        "rainfall_windows_end_before_events": bool(all(before_event)),
        "independent_rainfall_reconstruction_matches": max(rainfall_errors) < 1e-8,
        "training_manifest_records_all_rows": metadata["training_rows"] == len(train),
        "training_input_hash_matches": metadata["train_sha256"] == file_hash(args.data_dir / "Train.csv"),
        "rainfall_cache_hash_matches": metadata["cache_sha256"] == file_hash(cache_path),
        "saved_recipe_matches_code": metadata["recipe"] == recipe(),
        "saved_environment_matches_current": metadata["software"] == software_versions(),
        "all_four_model_schemas_match": all(
            list(bundle["models"][branch][family].feature_names_in_) == FEATURE_COLUMNS[branch]
            for branch in FEATURE_COLUMNS for family in ["rf", "hgb"]
        ),
        "rf_bootstraps_use_full_training_pool": all(
            all(len(indices) == len(train) and indices.min() >= 0 and indices.max() < len(train)
                for indices in bundle["models"][branch]["rf"].estimators_samples_)
            for branch in FEATURE_COLUMNS
        ),
        "hgb_uses_100_iterations_without_early_stopping": all(
            bundle["models"][branch]["hgb"].n_iter_ == 100
            and bundle["models"][branch]["hgb"].early_stopping is False
            for branch in FEATURE_COLUMNS
        ),
        "python_is_in_virtual_environment": sys.prefix != sys.base_prefix,
    }
    oof_path = args.submission.parent / "validation" / "oof_predictions.csv"
    if oof_path.exists():
        oof = pd.read_csv(oof_path)
        aligned_groups = pd.Series(groups, index=train.ID)
        isolated = True
        complete = True
        for _, assignment in oof.groupby("seed"):
            complete &= (len(assignment) == len(train) and assignment.ID.is_unique
                         and set(assignment.ID) == set(train.ID)
                         and assignment.fold.nunique() == 5
                         and np.isfinite(assignment[["baseline", "rain6", "final"]]).all().all())
            actual_groups = assignment.ID.map(aligned_groups)
            isolated &= pd.DataFrame({"group": actual_groups, "fold": assignment.fold}).groupby("group").fold.nunique().max() == 1
        checks["all_four_cv_assignments_complete"] = bool(complete and oof.seed.nunique() == 4)
        checks["geographic_groups_are_isolated_in_cv"] = bool(isolated)

    report = dict(
        status="passed" if all(checks.values()) else "failed", checks=checks,
        training_rows=len(train), test_rows=len(test), geographic_groups=len(np.unique(groups)),
        baseline_features=len(FEATURE_COLUMNS["baseline"]), rainfall_features=len(FEATURE_COLUMNS["rain6"]),
        max_absolute_probability_difference=float(difference.max()), binary_prediction_differences=binary_changes,
        probability_min=float(actual.TargetRAUC.min()), probability_max=float(actual.TargetRAUC.max()),
        positive_predictions=int(actual.TargetF1.sum()),
        reference_sha256=file_hash(args.reference), submission_sha256=file_hash(args.submission),
        software=software_versions(), python_version=sys.version.split()[0],
        limitation="Uses the supplied rainfall cache. Does not independently download all source rasters or rule out model-selection overfitting.",
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Checks passed: {sum(checks.values())}/{len(checks)}")
    print(f"Maximum probability difference: {difference.max():.3g}")
    print(f"Binary prediction differences: {binary_changes}")
    print(f"Report: {args.report}")
    require(all(checks.values()), "Verification failed; inspect the saved report.")


if __name__ == "__main__":
    main()
