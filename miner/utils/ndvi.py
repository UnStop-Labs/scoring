"""
Spectral index computation from satellite bands.

NDVI  = (B08 - B04) / (B08 + B04)          vegetation health
NDWI  = (B03 - B08) / (B03 + B08)          open water (McFeeters)
NDMI  = (B08 - B11) / (B08 + B11)          moisture / SWIR water index

moisture_index is derived from NDMI, normalised to [0, 1].
moisture_class is assigned from four thresholds:
    < 0.20  → CRITICAL_DRY
    0.20–0.40 → DRY
    0.40–0.70 → OPTIMAL
    >= 0.70   → WET
"""

from typing import Tuple
import numpy as np


_EPS = 1e-8


def compute_ndvi(B08: np.ndarray, B04: np.ndarray) -> np.ndarray:
    num = B08 - B04
    den = B08 + B04 + _EPS
    return np.clip(num / den, -1.0, 1.0).astype(np.float32)


def compute_ndwi(B03: np.ndarray, B08: np.ndarray) -> np.ndarray:
    """McFeeters open-water NDWI."""
    num = B03 - B08
    den = B03 + B08 + _EPS
    return np.clip(num / den, -1.0, 1.0).astype(np.float32)


def compute_ndmi(B08: np.ndarray, B11: np.ndarray) -> np.ndarray:
    """Normalised Difference Moisture Index — soil / canopy water content."""
    num = B08 - B11
    den = B08 + B11 + _EPS
    return np.clip(num / den, -1.0, 1.0).astype(np.float32)


def moisture_index_from_ndmi(ndmi: np.ndarray) -> np.ndarray:
    """Map NDMI [-1, 1] to moisture_index [0, 1]."""
    return np.clip((ndmi + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)


def classify_moisture(moisture_index: float) -> str:
    if moisture_index < 0.20:
        return "CRITICAL_DRY"
    elif moisture_index < 0.40:
        return "DRY"
    elif moisture_index < 0.70:
        return "OPTIMAL"
    else:
        return "WET"


def compute_all_indices(
    B03: np.ndarray,
    B04: np.ndarray,
    B08: np.ndarray,
    B11: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (ndvi, ndwi, moisture_index) arrays of identical shape."""
    ndvi = compute_ndvi(B08, B04)
    ndwi = compute_ndwi(B03, B08)
    ndmi = compute_ndmi(B08, B11)
    mi = moisture_index_from_ndmi(ndmi)
    return ndvi, ndwi, mi


def per_cell_confidence(
    ndvi: float,
    ndwi: float,
    cloud_masked: bool,
    moisture_index: float,
) -> float:
    """Heuristic per-cell confidence based on index plausibility."""
    if cloud_masked:
        return 0.45

    conf = 0.90
    # Penalty for physically implausible NDVI
    if not -1.0 <= ndvi <= 1.0:
        conf -= 0.30
    # Slightly lower confidence at extremes (very dry or very wet)
    if moisture_index < 0.10 or moisture_index > 0.90:
        conf -= 0.10
    return float(np.clip(conf, 0.0, 1.0))
