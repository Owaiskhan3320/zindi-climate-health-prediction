"""Load this project's models and write the fixed six-month CHIRPS submission."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .features import CACHE_NAME, build_matrices, check_ids, load_rainfall_cache, load_rows, require
from .models import THRESHOLD, file_hash, predict_probabilities, recipe, software_versions
from .train import MODEL_NAME

OUTPUT_COLUMNS = ["ID", "TargetF1", "TargetRAUC"]


def check_submission(submission, sample):
    require(list(submission.columns) == OUTPUT_COLUMNS == list(sample.columns),
            "Unexpected submission columns.")
    check_ids(submission, "Submission")
    require(submission.ID.tolist() == sample.ID.tolist(), "Submission ID order differs.")
    probabilities = submission.TargetRAUC.to_numpy(dtype=float)
    require(np.isfinite(probabilities).all()
            and ((probabilities >= 0) & (probabilities <= 1)).all(),
            "Submission probabilities must be finite and in [0, 1].")
    require(pd.api.types.is_integer_dtype(submission.TargetF1), "TargetF1 must be integer.")
    require(np.array_equal(submission.TargetF1, (probabilities >= THRESHOLD).astype(int)),
            "TargetF1 must equal TargetRAUC >= 0.5.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--output", type=Path, default=Path("outputs/submission.csv"))
    args = parser.parse_args()

    # joblib uses pickle. Load only the artifact created by this project's training script.
    bundle = joblib.load(args.model_dir / MODEL_NAME)
    metadata = bundle["metadata"]
    require(metadata["recipe"] == recipe(), "Saved model recipe differs from this code.")
    require(metadata["software"] == software_versions(),
            "Use the training environment versions recorded in training_manifest.json.")
    cache_path = args.data_dir / CACHE_NAME
    require(file_hash(cache_path) == metadata["cache_sha256"],
            "Rainfall cache changed since training. Prepare the complete cache before training.")
    test = load_rows(args.data_dir / "Test.csv")
    sample = pd.read_csv(args.data_dir / "SampleSubmission.csv")
    check_ids(sample, "SampleSubmission")
    require(set(test.ID) == set(sample.ID), "Test and sample ID sets differ.")
    matrices = build_matrices(test, load_rainfall_cache(cache_path))
    with threadpool_limits(limits=4):
        probabilities = predict_probabilities(bundle["models"], matrices)["final"]
    submission = sample.copy()
    submission["TargetRAUC"] = submission.ID.map(pd.Series(probabilities, index=test.ID))
    submission["TargetF1"] = (submission.TargetRAUC >= THRESHOLD).astype(int)
    check_submission(submission, sample)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False, float_format="%.17g")
    saved = pd.read_csv(args.output, float_precision="round_trip")
    check_submission(saved, sample)
    require(np.array_equal(saved.TargetRAUC, submission.TargetRAUC),
            "CSV serialization changed probabilities.")
    print(f"Saved {len(submission)} predictions to {args.output}")


if __name__ == "__main__":
    main()
