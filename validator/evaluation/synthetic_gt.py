"""
Synthetic Ground Truth (SGT) generator for the irrigation subnet.

Instead of relying on cross-miner consensus (which can be gamed by colluding
miners), every challenge carries a *deterministic* ground truth derived from:

    seed = SHA256(nonce + lat + lon + date)

A lightweight physics-based model then simulates:
  - NDVI seasonal curve (sinusoidal, latitude-tuned)
  - Soil moisture from rainfall-season proxy
  - Per-cell noise injected deterministically from seed

This mirrors TurboVision's SAM3 Pseudo Ground Truth approach: a reference
model the validator controls and miners cannot influence.

GT schema per cell
------------------
{
  "cell_id":        int,
  "ndvi":           float,    # [-1, 1]
  "ndwi":           float,    # [-1, 1]
  "moisture_index": float,    # [0, 1]
  "moisture_class": str,      # CRITICAL_DRY | DRY | OPTIMAL | WET
  "cloud_masked":   bool
}
"""

from __future__ import annotations

import hashlib
import math
import struct
from datetime import datetime
from typing import Any


# ---------------------------------------------------------------------------
# Moisture class thresholds  (moisture_index → class)
# ---------------------------------------------------------------------------

MOISTURE_THRESHOLDS = [
    (0.20, "CRITICAL_DRY"),
    (0.40, "DRY"),
    (0.70, "OPTIMAL"),
    (1.01, "WET"),
]

CLASS_ORDINAL = {"CRITICAL_DRY": 0, "DRY": 1, "OPTIMAL": 2, "WET": 3}


def moisture_class_from_index(mi: float) -> str:
    for threshold, cls in MOISTURE_THRESHOLDS:
        if mi < threshold:
            return cls
    return "WET"


# ---------------------------------------------------------------------------
# Deterministic PRNG seeded from challenge metadata
# ---------------------------------------------------------------------------

class _DeterministicRNG:
    """Tiny LCG seeded from a SHA-256 hash — reproducible across validators."""

    def __init__(self, seed_bytes: bytes) -> None:
        self._state = int.from_bytes(seed_bytes[:8], "big")
        self._a = 6364136223846793005
        self._c = 1442695040888963407
        self._m = 2 ** 64

    def next_float(self) -> float:
        self._state = (self._a * self._state + self._c) % self._m
        return self._state / self._m

    def next_range(self, lo: float, hi: float) -> float:
        return lo + self.next_float() * (hi - lo)


def _make_seed(nonce: str, latitude: float, longitude: float, query_date: str) -> bytes:
    raw = f"{nonce}:{latitude:.6f}:{longitude:.6f}:{query_date[:10]}"
    return hashlib.sha256(raw.encode()).digest()


# ---------------------------------------------------------------------------
# Physics-based vegetation / moisture model
# ---------------------------------------------------------------------------

def _ndvi_baseline(latitude: float, day_of_year: int) -> float:
    """
    Sinusoidal seasonal NDVI.
    - Tropical latitudes (|lat| < 15°): stable ~0.55
    - Temperate latitudes: peaks in summer, dips in winter
    """
    phase_offset = math.pi if latitude < 0 else 0.0   # flip seasons in S hemisphere
    seasonal_amplitude = max(0.0, (abs(latitude) - 15.0) / 60.0) * 0.30
    seasonal = seasonal_amplitude * math.cos(2 * math.pi * day_of_year / 365.0 + phase_offset)
    base = 0.55 - abs(latitude) / 180.0 * 0.15        # tropics greener
    return float(max(-0.20, min(0.90, base + seasonal)))


def _moisture_baseline(latitude: float, day_of_year: int) -> float:
    """
    Rough soil-moisture proxy.
    Thai agricultural zones peak May-Oct (wet season).
    Northern latitudes have different seasonality.
    """
    # Thai wet season: May (120) → October (300)
    if 10 < abs(latitude) < 25:
        wet_peak   = 210   # mid-August
        amplitude  = 0.30
    else:
        wet_peak   = 180
        amplitude  = 0.20

    seasonal = amplitude * math.cos(2 * math.pi * (day_of_year - wet_peak) / 365.0)
    base = 0.45  # neutral moisture
    return float(max(0.05, min(0.95, base - seasonal)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_synthetic_gt(
    nonce: str,
    latitude: float,
    longitude: float,
    query_date: str,
    n_cells: int,
    cloud_fraction: float = 0.05,
) -> list[dict[str, Any]]:
    """
    Generate a deterministic list of synthetic GT cells for one challenge.

    Parameters
    ----------
    nonce          : challenge nonce (ensures uniqueness per challenge)
    latitude       : field centre latitude
    longitude      : field centre longitude
    query_date     : ISO-8601 date string
    n_cells        : number of grid cells (must match miner's grid)
    cloud_fraction : fraction of cells to mark as cloud-masked

    Returns
    -------
    List of cell dicts with keys: cell_id, ndvi, ndwi, moisture_index,
    moisture_class, cloud_masked
    """
    seed = _make_seed(nonce, latitude, longitude, query_date)
    rng  = _DeterministicRNG(seed)

    try:
        doy = datetime.fromisoformat(query_date[:10]).timetuple().tm_yday
    except Exception:
        doy = 180  # default to mid-year

    ndvi_base     = _ndvi_baseline(latitude, doy)
    moisture_base = _moisture_baseline(latitude, doy)

    cells = []
    for cell_id in range(n_cells):
        # Per-cell noise (spatial heterogeneity)
        ndvi_noise     = rng.next_range(-0.12, 0.12)
        moisture_noise = rng.next_range(-0.15, 0.15)

        ndvi     = float(max(-1.0, min(1.0, ndvi_base + ndvi_noise)))
        ndwi     = float(max(-1.0, min(1.0, -ndvi * 0.6 + rng.next_range(-0.10, 0.10))))
        mi       = float(max(0.0,  min(1.0, moisture_base + moisture_noise)))
        cls      = moisture_class_from_index(mi)
        clouded  = rng.next_float() < cloud_fraction

        cells.append({
            "cell_id":        cell_id,
            "ndvi":           round(ndvi, 4),
            "ndwi":           round(ndwi, 4),
            "moisture_index": round(mi,   4),
            "moisture_class": cls,
            "cloud_masked":   clouded,
        })

    return cells


def cells_from_geojson(geojson: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract flat cell dicts from a miner's GeoJSON FeatureCollection."""
    cells = []
    for i, feat in enumerate(geojson.get("features", [])):
        props = feat.get("properties", {})
        cells.append({
            "cell_id":        props.get("cell_id", i),
            "ndvi":           float(props.get("ndvi", 0.0)),
            "ndwi":           float(props.get("ndwi", 0.0)),
            "moisture_index": float(props.get("moisture_index", 0.0)),
            "moisture_class": props.get("moisture_class", "DRY"),
            "cloud_masked":   bool(props.get("cloud_masked", False)),
        })
    return cells
