"""
Satellite data fetching via Microsoft Planetary Computer STAC API.

Fetches Sentinel-2 L2A imagery for a given bounding box and date window,
returning raw band arrays (B03 Green, B04 Red, B08 NIR, B11 SWIR) clipped
to the requested area.  Falls back to synthetic data if the API is
unreachable or no scenes are found.
"""

import asyncio
import math
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
# Bounding-box helpers
# ---------------------------------------------------------------------------

METERS_PER_DEGREE_LAT = 111_320.0


def field_bbox(
    latitude: float, longitude: float, area_rai: float, padding_m: float = 100.0
) -> Tuple[float, float, float, float]:
    """Return (min_lon, min_lat, max_lon, max_lat) for a square field."""
    area_m2 = area_rai * 1_600.0
    half_side_m = math.sqrt(area_m2) / 2.0 + padding_m

    lat_deg = half_side_m / METERS_PER_DEGREE_LAT
    lon_deg = half_side_m / (METERS_PER_DEGREE_LAT * math.cos(math.radians(latitude)))

    return (
        longitude - lon_deg,
        latitude - lat_deg,
        longitude + lon_deg,
        latitude + lat_deg,
    )


# ---------------------------------------------------------------------------
# Planetary Computer / STAC fetch
# ---------------------------------------------------------------------------

async def fetch_sentinel2_bands(
    latitude: float,
    longitude: float,
    area_rai: float,
    query_date: str,
    window_days: int = 14,
) -> Dict[str, Any]:
    """
    Search Planetary Computer for a Sentinel-2 L2A scene within *window_days*
    of *query_date*, clip to the field bounding box, and return per-band arrays.

    Returns a dict with keys:
        B03, B04, B08, B11  – 2-D float32 arrays (0-1 reflectance)
        scene_id            – str, the STAC item id used
        scene_date          – str ISO date of the scene
        cloud_cover         – float 0-100
        source              – "Sentinel-2" | "Landsat-9" | "synthetic"
    """
    try:
        return await asyncio.wait_for(
            _planetary_computer_fetch(latitude, longitude, area_rai, query_date, window_days),
            timeout=30.0,
        )
    except Exception as exc:
        logger.warning(f"Satellite fetch failed ({exc}); falling back to synthetic data")
        return _synthetic_bands(latitude, longitude, area_rai, query_date)


async def _planetary_computer_fetch(
    latitude: float,
    longitude: float,
    area_rai: float,
    query_date: str,
    window_days: int,
) -> Dict[str, Any]:
    import pystac_client
    import planetary_computer
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.transform import array_bounds
    import httpx

    bbox = field_bbox(latitude, longitude, area_rai)
    qdate = datetime.fromisoformat(query_date[:10])
    start = (qdate - timedelta(days=window_days)).strftime("%Y-%m-%d")
    end = (qdate + timedelta(days=window_days)).strftime("%Y-%m-%d")

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=list(bbox),
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": 30}},
        sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        max_items=5,
    )

    items = list(search.items())
    if not items:
        # Try Landsat as fallback
        search_ls = catalog.search(
            collections=["landsat-c2-l2"],
            bbox=list(bbox),
            datetime=f"{start}/{end}",
            query={"eo:cloud_cover": {"lt": 40}},
            max_items=5,
        )
        items = list(search_ls.items())
        source = "Landsat-9"
        band_map = {"B03": "green", "B04": "red", "B08": "nir08", "B11": "swir16"}
    else:
        source = "Sentinel-2"
        band_map = {"B03": "B03", "B04": "B04", "B08": "B08", "B11": "B11"}

    if not items:
        raise RuntimeError("No satellite scenes found")

    item = items[0]
    scene_id = item.id
    scene_date = item.datetime.strftime("%Y-%m-%d") if item.datetime else query_date
    cloud_cover = item.properties.get("eo:cloud_cover", 0.0)

    bands: Dict[str, np.ndarray] = {}
    for out_name, asset_key in band_map.items():
        href = item.assets[asset_key].href
        async with httpx.AsyncClient() as client:
            resp = await client.get(href, follow_redirects=True)
            resp.raise_for_status()
            import io
            with rasterio.open(io.BytesIO(resp.content)) as src:
                win = from_bounds(*bbox, transform=src.transform)
                arr = src.read(1, window=win).astype(np.float32)
                # Sentinel-2 reflectance is stored as uint16 scaled by 10000
                if source == "Sentinel-2":
                    arr = np.clip(arr / 10_000.0, 0.0, 1.0)
                else:
                    arr = np.clip(arr / 65_535.0, 0.0, 1.0)
                # Down-sample to keep arrays manageable (max 200x200)
                arr = _resize_if_needed(arr, max_side=200)
                bands[out_name] = arr

    return {
        **bands,
        "scene_id": scene_id,
        "scene_date": scene_date,
        "cloud_cover": cloud_cover,
        "source": source,
    }


def _resize_if_needed(arr: np.ndarray, max_side: int = 200) -> np.ndarray:
    h, w = arr.shape
    if h <= max_side and w <= max_side:
        return arr
    scale = min(max_side / h, max_side / w)
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
    # Simple block-average downsampling
    bh, bw = max(1, h // new_h), max(1, w // new_w)
    out = arr[: new_h * bh, : new_w * bw].reshape(new_h, bh, new_w, bw).mean(axis=(1, 3))
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Synthetic fallback — deterministic from coords + date
# ---------------------------------------------------------------------------

def _synthetic_bands(
    latitude: float,
    longitude: float,
    area_rai: float,
    query_date: str,
) -> Dict[str, Any]:
    """
    Generate plausible synthetic band values for development / testing.
    Values are seeded from coordinates and date so they are reproducible.
    """
    seed = int(abs(latitude * 1000) + abs(longitude * 1000) + _date_seed(query_date))
    rng = np.random.default_rng(seed)

    size = max(10, int(math.sqrt(area_rai * 1_600) / 20))  # grid cells per side

    # Rough Thai agricultural baseline: moderate vegetation, mid moisture
    B03 = rng.uniform(0.04, 0.10, (size, size)).astype(np.float32)   # green
    B04 = rng.uniform(0.05, 0.12, (size, size)).astype(np.float32)   # red
    B08 = rng.uniform(0.25, 0.55, (size, size)).astype(np.float32)   # NIR
    B11 = rng.uniform(0.08, 0.30, (size, size)).astype(np.float32)   # SWIR

    # Introduce a spatial gradient to simulate realistic heterogeneity
    gradient = np.linspace(0.8, 1.2, size)
    B08 *= gradient[:, None]
    B11 *= gradient[None, :]

    return {
        "B03": B03,
        "B04": B04,
        "B08": B08,
        "B11": B11,
        "scene_id": f"SYNTHETIC_{seed}",
        "scene_date": query_date[:10],
        "cloud_cover": 0.0,
        "source": "synthetic",
    }


def _date_seed(query_date: str) -> int:
    try:
        d = datetime.fromisoformat(query_date[:10])
        return d.toordinal()
    except Exception:
        return 0
