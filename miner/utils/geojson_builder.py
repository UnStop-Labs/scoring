"""
Build a GeoJSON FeatureCollection grid over a field.

Given:
  - centroid (lat, lon)
  - area_rai (1 Rai = 1600 m²)
  - grid_resolution_m  (cell size in metres, default 20)

Returns a GeoJSON FeatureCollection where each Feature is one grid cell
with geometry (Polygon) and spectral / moisture properties.
"""

import math
from typing import Any, Dict, List

import numpy as np

from miner.utils.ndvi import classify_moisture, compute_all_indices, per_cell_confidence

METERS_PER_DEGREE_LAT = 111_320.0


def _meters_to_deg_lat(meters: float) -> float:
    return meters / METERS_PER_DEGREE_LAT


def _meters_to_deg_lon(meters: float, latitude: float) -> float:
    return meters / (METERS_PER_DEGREE_LAT * math.cos(math.radians(latitude)))


def build_geojson_grid(
    latitude: float,
    longitude: float,
    area_rai: float,
    grid_resolution_m: int,
    B03: np.ndarray,
    B04: np.ndarray,
    B08: np.ndarray,
    B11: np.ndarray,
    cloud_cover: float = 0.0,
    et_mm: float = 5.0,
    data_source: str = "Sentinel-2",
) -> Dict[str, Any]:
    """
    Create a GeoJSON FeatureCollection grid aligned to the field centroid.

    Band arrays (B03, B04, B08, B11) are assumed to cover the full field
    extent and will be resampled to match the requested grid resolution.
    """
    area_m2 = area_rai * 1_600.0
    field_side_m = math.sqrt(area_m2)

    n_cols = max(1, int(round(field_side_m / grid_resolution_m)))
    n_rows = max(1, int(round(field_side_m / grid_resolution_m)))

    cell_w_deg = _meters_to_deg_lon(grid_resolution_m, latitude)
    cell_h_deg = _meters_to_deg_lat(grid_resolution_m)

    total_w_deg = n_cols * cell_w_deg
    total_h_deg = n_rows * cell_h_deg

    origin_lon = longitude - total_w_deg / 2.0
    origin_lat = latitude + total_h_deg / 2.0

    # Resize band arrays to match grid dimensions (n_rows × n_cols)
    B03_r = _resize_band(B03, n_rows, n_cols)
    B04_r = _resize_band(B04, n_rows, n_cols)
    B08_r = _resize_band(B08, n_rows, n_cols)
    B11_r = _resize_band(B11, n_rows, n_cols)

    ndvi_grid, ndwi_grid, mi_grid = compute_all_indices(B03_r, B04_r, B08_r, B11_r)

    cloud_fraction = cloud_cover / 100.0
    cloud_mask = np.random.random((n_rows, n_cols)) < cloud_fraction

    features: List[Dict] = []
    cell_id = 0

    for row in range(n_rows):
        for col in range(n_cols):
            min_lon = origin_lon + col * cell_w_deg
            max_lon = min_lon + cell_w_deg
            max_lat = origin_lat - row * cell_h_deg
            min_lat = max_lat - cell_h_deg

            ndvi_val = float(ndvi_grid[row, col])
            ndwi_val = float(ndwi_grid[row, col])
            mi_val = float(mi_grid[row, col])
            is_cloud = bool(cloud_mask[row, col])

            moisture_class = classify_moisture(mi_val)
            cell_conf = per_cell_confidence(ndvi_val, ndwi_val, is_cloud, mi_val)

            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [min_lon, min_lat],
                        [max_lon, min_lat],
                        [max_lon, max_lat],
                        [min_lon, max_lat],
                        [min_lon, min_lat],
                    ]],
                },
                "properties": {
                    "cell_id": cell_id,
                    "moisture_class": moisture_class,
                    "moisture_index": round(mi_val, 4),
                    "ndvi": round(ndvi_val, 4),
                    "ndwi": round(ndwi_val, 4),
                    "evapotranspiration_mm": round(et_mm + float(np.random.normal(0, 0.5)), 2),
                    "cloud_masked": is_cloud,
                    "data_source": data_source,
                    "cell_confidence": round(cell_conf, 4),
                },
            }
            features.append(feature)
            cell_id += 1

    return {"type": "FeatureCollection", "features": features}


def _resize_band(arr: np.ndarray, target_rows: int, target_cols: int) -> np.ndarray:
    """Nearest-neighbour resize of a 2-D array."""
    src_rows, src_cols = arr.shape
    row_idx = np.linspace(0, src_rows - 1, target_rows, dtype=int)
    col_idx = np.linspace(0, src_cols - 1, target_cols, dtype=int)
    return arr[np.ix_(row_idx, col_idx)]
