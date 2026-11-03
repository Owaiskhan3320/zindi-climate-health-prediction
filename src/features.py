"""Build the same ordered features for training, validation and prediction."""

from pathlib import Path

import numpy as np
import pandas as pd

AGE_BINS = [-1, 0, 1, 5, 17, 49, 64, 111]
AGE_LABELS = [
    "Age 0", "Age 1", "Ages 2-5", "Ages 6-17",
    "Ages 18-49", "Ages 50-64", "Age 65+",
]
AGE_COLUMNS = ["age_group_" + label for label in sorted(AGE_LABELS)]
BASE_COLUMNS = (
    ["age"] + AGE_COLUMNS
    + ["gender_Female", "gender_Male", "zone_Peri_urban", "zone_Rural"]
    + ["year", "month_sin", "month_cos", "post_2017_07"]
    + [column + "_after_mid_2017" for column in AGE_COLUMNS]
)
RAIN_COLUMN = "chirps_rain_sum_6m"
CACHE_NAME = "chirps_v3_monthly_point_cache.csv"
CACHE_COLUMNS = ["month", "latitude", "longitude", "rainfall_mm"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_ids(rows, name):
    require("ID" in rows, f"{name} has no ID column.")
    require(rows.ID.notna().all() and rows.ID.is_unique,
            f"{name} IDs must be unique and non-missing.")


def load_rows(path):
    rows = pd.read_csv(Path(path))
    check_ids(rows, Path(path).name)
    require("deathdate" in rows, "Missing deathdate column.")
    rows["deathdate"] = pd.to_datetime(rows["deathdate"], errors="raise")
    require(rows.deathdate.notna().all(), "Missing deathdate values.")
    return rows


def training_target(rows):
    require("is_climate_sensitive" in rows, "Missing training target.")
    target = rows["is_climate_sensitive"]
    require(target.notna().all() and set(target.unique()) == {0, 1},
            "Training target must contain both binary classes and no missing values.")
    return target.to_numpy(dtype=int)


def base_features(rows):
    required = ["age", "gender", "zone", "deathdate"]
    require(rows.columns.is_unique, "Duplicate input column names.")
    require(all(column in rows for column in required), "Missing baseline input columns.")
    require(not rows[required].isna().any().any(), "Missing baseline input values.")
    age = pd.to_numeric(rows.age, errors="raise").astype(float)
    require(np.isfinite(age).all() and age.between(0, 111).all(),
            "Age must be finite and between 0 and 111.")
    require(set(rows.gender) <= {"Female", "Male"}, "Unknown gender category.")
    require(set(rows.zone) <= {"Peri_urban", "Rural"}, "Unknown zone category.")
    dates = pd.to_datetime(rows.deathdate, errors="raise")
    groups = pd.cut(age, AGE_BINS, labels=AGE_LABELS, include_lowest=True, right=True)
    require(groups.notna().all(), "An age did not map to an age group.")

    matrix = pd.DataFrame({"age": age}, index=rows.index)
    for label in sorted(AGE_LABELS):
        matrix["age_group_" + label] = (groups == label).astype(float)
    for name, levels in [("gender", ["Female", "Male"]),
                         ("zone", ["Peri_urban", "Rural"])]:
        for level in levels:
            matrix[f"{name}_{level}"] = (rows[name] == level).astype(float)

    post = (dates >= pd.Timestamp("2017-07-01")).astype(float)
    matrix["year"] = dates.dt.year.astype(float)
    matrix["month_sin"] = np.sin(2 * np.pi * dates.dt.month / 12)
    matrix["month_cos"] = np.cos(2 * np.pi * dates.dt.month / 12)
    matrix["post_2017_07"] = post
    for column in AGE_COLUMNS:
        matrix[column + "_after_mid_2017"] = matrix[column] * post

    require(list(matrix.columns) == BASE_COLUMNS, "Unexpected baseline feature order.")
    require(np.isfinite(matrix.to_numpy()).all(), "Non-finite baseline features.")
    return matrix


def check_coordinates(rows):
    require(all(column in rows for column in ["latitude", "longitude"]),
            "Missing latitude or longitude.")
    coordinates = rows[["latitude", "longitude"]].astype(float)
    require(np.isfinite(coordinates.to_numpy()).all(), "Invalid coordinates.")
    require(coordinates.latitude.between(-90, 90).all()
            and coordinates.longitude.between(-180, 180).all(), "Coordinates out of range.")
    # Python round matches the original cache builder at half-way decimal values.
    # DataFrame.round uses NumPy rounding and can choose a different cache key.
    return coordinates.map(lambda value: round(float(value), 7))


def load_rainfall_cache(path):
    cache = pd.read_csv(path, dtype={"month": str})
    require(list(cache.columns) == CACHE_COLUMNS, "Unexpected rainfall cache columns.")
    cache[["latitude", "longitude"]] = check_coordinates(cache)
    months = pd.PeriodIndex(cache.month, freq="M")
    require(not months.isna().any(), "Missing month in rainfall cache.")
    cache["month"] = months.astype(str)
    cache["rainfall_mm"] = pd.to_numeric(cache.rainfall_mm, errors="raise")
    require(not cache.duplicated(["month", "latitude", "longitude"]).any(),
            "Duplicate rainfall cache keys.")
    require(np.isfinite(cache.rainfall_mm).all() and (cache.rainfall_mm >= 0).all(),
            "Rainfall must be finite and non-negative.")
    return cache


def rainfall_features(rows, cache):
    coordinates = check_coordinates(rows)
    dates = pd.to_datetime(rows.deathdate, errors="raise")
    require(dates.notna().all(), "Missing rainfall lookup date.")
    lookup = {
        (month, float(lat), float(lon)): float(rain)
        for month, lat, lon, rain in cache[CACHE_COLUMNS].itertuples(index=False, name=None)
    }
    totals = []
    for date, (lat, lon) in zip(dates, coordinates.itertuples(index=False, name=None)):
        last_month = date.to_period("M") - 1
        months = pd.period_range(end=last_month, periods=6, freq="M")
        try:
            # Use the same chronological Python sum as the frozen submission script.
            total = sum(lookup[(str(month), lat, lon)] for month in months)
        except KeyError as error:
            raise ValueError(
                f"Rainfall cache is incomplete at {error.args[0]}. Run src.prepare_climate."
            ) from error
        totals.append(total)
    matrix = pd.DataFrame({RAIN_COLUMN: totals}, index=rows.index)
    require(np.isfinite(matrix.to_numpy()).all() and (matrix.to_numpy() >= 0).all(),
            "Invalid rainfall feature.")
    return matrix


def build_matrices(rows, cache):
    baseline = base_features(rows)
    rain = rainfall_features(rows, cache)
    return {
        "baseline": baseline,
        "rain6": pd.concat([baseline, rain], axis=1),
    }
