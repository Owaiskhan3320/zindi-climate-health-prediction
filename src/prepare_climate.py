"""Reuse or download CHIRPS v3 monthly point samples and create rainfall features."""

import argparse
import hashlib
import json
import time
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from PIL import Image

from .features import (
    CACHE_COLUMNS, CACHE_NAME, check_coordinates, load_rainfall_cache,
    load_rows, rainfall_features, require,
)
from .models import file_hash

SOURCE_URL = (
    "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/"
    "monthly/africa/tifs/chirps-v3.0.{month}.tif"
)


def sample_month(month, coordinates):
    url = SOURCE_URL.format(month=month.replace("-", "."))
    raw = None
    for attempt in range(3):
        try:
            with urlopen(url, timeout=30) as response:
                raw = response.read()
            break
        except OSError:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)

    with Image.open(BytesIO(raw)) as raster:
        scale = raster.tag_v2.get(33550)
        tiepoint = raster.tag_v2.get(33922)
        require(scale is not None and tiepoint is not None, "Missing GeoTIFF georeferencing.")
        dx, dy = float(scale[0]), float(scale[1])
        x0, y0 = float(tiepoint[3]), float(tiepoint[4])
        values = np.asarray(raster, dtype=np.float32)

    samples = []
    for latitude, longitude in coordinates:
        column = int(np.rint((longitude - x0) / dx - 0.5))
        row = int(np.rint((y0 - latitude) / dy - 0.5))
        require(0 <= row < values.shape[0] and 0 <= column < values.shape[1],
                f"Coordinate outside CHIRPS raster: {(latitude, longitude)}")
        rain = float(values[row, column])
        require(np.isfinite(rain) and rain >= 0, "Invalid sampled rainfall.")
        samples.append([month, latitude, longitude, rain])
    evidence = dict(month=month, url=url, raster_sha256=hashlib.sha256(raw).hexdigest())
    return pd.DataFrame(samples, columns=CACHE_COLUMNS), evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    frames = {name: load_rows(args.data_dir / f"{name}.csv") for name in ["Train", "Test"]}
    coordinates = sorted({
        pair for frame in frames.values()
        for pair in check_coordinates(frame).itertuples(index=False, name=None)
    })
    first = min(frame.deathdate.min() for frame in frames.values()).to_period("M") - 6
    last = max(frame.deathdate.max() for frame in frames.values()).to_period("M") - 1
    months = [str(month) for month in pd.period_range(first, last, freq="M")]

    cache_path = args.data_dir / CACHE_NAME
    cache = (load_rainfall_cache(cache_path) if cache_path.exists()
             else pd.DataFrame(columns=CACHE_COLUMNS))
    present = set(cache[["month", "latitude", "longitude"]].itertuples(index=False, name=None))
    missing = [month for month in months
               if any((month, lat, lon) not in present for lat, lon in coordinates)]
    print(f"Coordinates: {len(coordinates)}. Monthly rasters to download: {len(missing)}.", flush=True)
    log_path = args.data_dir / "chirps_downloads.jsonl"
    for count, month in enumerate(missing, start=1):
        sampled, evidence = sample_month(month, coordinates)
        cache = pd.concat([cache, sampled], ignore_index=True)
        cache = cache.drop_duplicates(["month", "latitude", "longitude"], keep="last")
        cache = cache.sort_values(["month", "latitude", "longitude"])
        temporary = cache_path.with_suffix(".tmp")
        cache.to_csv(temporary, index=False, float_format="%.17g")
        temporary.replace(cache_path)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(evidence) + "\n")
        print(f"Downloaded {month} ({count}/{len(missing)}).", flush=True)

    cache = load_rainfall_cache(cache_path)
    for name, rows in frames.items():
        rain = rainfall_features(rows, cache)
        rain.insert(0, "ID", rows.ID)
        rain.to_csv(args.data_dir / f"chirps_features_{name.lower()}.csv", index=False,
                    float_format="%.17g")
    manifest = dict(
        source="Climate Hazards Center CHIRPS v3 monthly Africa",
        source_page="https://www.chc.ucsb.edu/data/chirps3",
        citation="https://doi.org/10.15780/G2JQ0P",
        cache_sha256=file_hash(cache_path), cache_rows=len(cache),
        downloaded_months_this_run=missing,
        months_needed=[str(first), str(last)], coordinates_needed=len(coordinates),
        sampling="Nearest raster cell center; coordinates rounded to 7 decimal places",
        window="6 complete calendar months strictly before the event month",
        note="Existing samples are reused; reuse alone does not independently verify their source.",
    )
    (args.data_dir / "climate_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("Rainfall features are ready. Training rebuilds them from this same cache.")


if __name__ == "__main__":
    main()
