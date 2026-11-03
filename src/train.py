"""Train all four models using every eligible row in Train.csv."""

import argparse
import json
from pathlib import Path

import joblib
from threadpoolctl import threadpool_limits

from .features import CACHE_NAME, build_matrices, load_rainfall_cache, load_rows, training_target
from .models import file_hash, fit_models, recipe, software_versions

MODEL_NAME = "rf_hgb_chirps6.joblib"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    args = parser.parse_args()

    train_path = args.data_dir / "Train.csv"
    cache_path = args.data_dir / CACHE_NAME
    rows = load_rows(train_path)
    target = training_target(rows)
    matrices = build_matrices(rows, load_rainfall_cache(cache_path))
    with threadpool_limits(limits=4):
        models = fit_models(matrices, target)
    metadata = dict(
        training_rows=len(rows), recipe=recipe(), software=software_versions(),
        train_sha256=file_hash(train_path), cache_sha256=file_hash(cache_path),
    )
    args.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": models, "metadata": metadata}, args.model_dir / MODEL_NAME, compress=3)
    (args.model_dir / "training_manifest.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Trained all four models on {len(rows)} rows.")
    print(f"Saved {args.model_dir / MODEL_NAME}")


if __name__ == "__main__":
    main()
