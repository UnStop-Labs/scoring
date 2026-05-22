"""
Spectral validity and false-alarm-rate pillar metrics.

SPECTRAL_VALIDITY
-----------------
Checks that every cell's spectral indices obey known physical constraints:
  • NDVI ∈ [-1, 1]
  • NDWI ∈ [-1, 1]
  • moisture_index ∈ [0, 1]
  • Cross-band consistency: high NDWI must correlate with high moisture
  • Cloud fraction < 30 % (above → confidence penalty)

FALSE_ALARM_RATE
----------------
Counts cells where the miner predicts WET/OPTIMAL but synthetic GT says
CRITICAL_DRY/DRY — i.e., spurious irrigation detections.

    FAPK = false_alarms / km²
    score = max(0, 1 - FAPK / FA_CEILING)

Mirrors TurboVision's compare_false_positive: max(0, 1 - ffpi / 10).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from validator.evaluation.metrics.registry import (
    ElementType, PillarName, register_metric,
)
from validator.evaluation.synthetic_gt import CLASS_ORDINAL


# ---------------------------------------------------------------------------
# Pillar: SPECTRAL_VALIDITY  (enhanced version of original spectral_score)
# ---------------------------------------------------------------------------

_PHYSICS_RULES = [
    # (field, lo, hi, penalty_weight)
    ("ndvi",           -1.0, 1.0, 1.00),
    ("ndwi",           -1.0, 1.0, 0.60),
    ("moisture_index",  0.0, 1.0, 0.60),
]


def _cross_band_penalty(props: dict[str, Any]) -> float:
    """
    Physical cross-band consistency checks.
    High NDWI means water is present → moisture_index should also be high.
    """
    ndwi = props.get("ndwi", 0.0)
    mi   = props.get("moisture_index", 0.0)
    ndvi = props.get("ndvi", 0.0)

    penalty = 0.0

    # Water body without moisture signal
    if ndwi > 0.30 and mi < 0.20:
        penalty += 0.40

    # Very high NDVI (dense canopy) paired with very low moisture is implausible
    if ndvi > 0.70 and mi < 0.15:
        penalty += 0.20

    return penalty


@register_metric((ElementType.IRRIGATION_DETECTION, PillarName.SPECTRAL_VALIDITY))
@register_metric((ElementType.CHANGE_DETECTION,     PillarName.SPECTRAL_VALIDITY))
@register_metric((ElementType.MOISTURE_ESTIMATION,  PillarName.SPECTRAL_VALIDITY))
def compute_spectral_validity(
    gt_cells:   list[dict[str, Any]],   # unused — spectral validity is self-contained
    pred_cells: list[dict[str, Any]],
    **kwargs,
) -> float:
    if not pred_cells:
        return 0.0

    total_penalty = 0.0
    cloud_count   = 0
    n             = len(pred_cells)

    for cell in pred_cells:
        if cell.get("cloud_masked"):
            cloud_count += 1
            continue

        for field, lo, hi, weight in _PHYSICS_RULES:
            val = cell.get(field, 0.0)
            if not (lo <= val <= hi):
                total_penalty += weight

        total_penalty += _cross_band_penalty(cell)

    cloud_fraction = cloud_count / n
    cloud_penalty  = max(0.0, (cloud_fraction - 0.30) * 1.5)

    penalty_rate = (total_penalty / max(n - cloud_count, 1)) + cloud_penalty
    return float(np.clip(1.0 - penalty_rate, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Pillar: FALSE_ALARM_RATE
# ---------------------------------------------------------------------------

# Classes considered "positive" (miner claims irrigation / wetness)
_POSITIVE_CLASSES = {"WET", "OPTIMAL"}
# Classes the GT considers "should be dry"
_DRY_GT_CLASSES   = {"CRITICAL_DRY", "DRY"}

# False alarms per km² that drives score to 0
FA_CEILING_PER_KM2 = 3.0


@register_metric((ElementType.IRRIGATION_DETECTION, PillarName.FALSE_ALARM_RATE))
@register_metric((ElementType.CHANGE_DETECTION,     PillarName.FALSE_ALARM_RATE))
def compute_false_alarm_rate(
    gt_cells:   list[dict[str, Any]],
    pred_cells: list[dict[str, Any]],
    area_km2:   float = 1.0,
    **kwargs,
) -> float:
    """
    Penalise miners that predict irrigation (WET / OPTIMAL) on cells the
    synthetic GT marks as dry (CRITICAL_DRY / DRY).

    score = max(0, 1 - false_alarms_per_km² / FA_CEILING)

    Mirrors TurboVision compare_false_positive: max(0, 1 - ffpi / 10).
    """
    gt_by_id   = {c["cell_id"]: c for c in gt_cells   if not c.get("cloud_masked")}
    pred_by_id = {c["cell_id"]: c for c in pred_cells if not c.get("cloud_masked")}

    false_alarms = 0
    for cid, pred_cell in pred_by_id.items():
        gt_cell = gt_by_id.get(cid)
        if gt_cell is None:
            continue
        if (
            pred_cell["moisture_class"] in _POSITIVE_CLASSES
            and gt_cell["moisture_class"] in _DRY_GT_CLASSES
        ):
            false_alarms += 1

    fapk = false_alarms / max(area_km2, 0.01)    # false alarms per km²
    return float(max(0.0, 1.0 - fapk / FA_CEILING_PER_KM2))
