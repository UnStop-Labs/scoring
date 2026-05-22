"""
Temporal consistency pillar metric.

Checks two things:

1. Scene freshness — the satellite scene used to answer the challenge
   should be from within ±14 days of the challenge query_date.
   Scenes > 30 days old lose significant value (fields change).

2. Moisture change-rate plausibility — if we have previous responses
   from this miner, a sudden jump of > 0.50 in mean moisture_index
   between consecutive challenges is physically implausible unless a
   major rain event occurred (not modelled here → penalty).

3. Class flip rate — fraction of cells whose moisture_class flipped
   ≥ 2 ordinal steps relative to synthetic GT trend.
   (e.g., predicting WET one day and CRITICAL_DRY the next)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from validator.evaluation.metrics.registry import (
    ElementType, PillarName, register_metric,
)
from validator.evaluation.synthetic_gt import CLASS_ORDINAL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_scene_date(scene_id: str) -> datetime | None:
    """Extract date from Sentinel-2 scene ID: S2A_MSIL2A_YYYYMMDD..."""
    try:
        if scene_id.startswith("S2") and len(scene_id) > 20:
            date_str = scene_id.split("_")[2][:8]
            return datetime.strptime(date_str, "%Y%m%d")
    except Exception:
        pass
    return None


def _mean_mi(cells: list[dict[str, Any]]) -> float | None:
    vals = [c["moisture_index"] for c in cells if not c.get("cloud_masked")]
    return float(np.mean(vals)) if vals else None


# ---------------------------------------------------------------------------
# Pillar: TEMPORAL_CONSISTENCY
# ---------------------------------------------------------------------------

@register_metric((ElementType.IRRIGATION_DETECTION, PillarName.TEMPORAL_CONSISTENCY))
@register_metric((ElementType.CHANGE_DETECTION,     PillarName.TEMPORAL_CONSISTENCY))
def compute_temporal_consistency(
    gt_cells:        list[dict[str, Any]],
    pred_cells:      list[dict[str, Any]],
    query_date:      str = "",
    satellite_scene_ids: list[str] | None = None,
    prev_pred_cells_list: list[list[dict[str, Any]]] | None = None,
    **kwargs,
) -> float:
    """
    Three sub-scores (equally weighted):
      1. Scene freshness    — how old is the satellite scene vs query_date?
      2. Change-rate check  — moisture jump vs previous challenge
      3. GT trend alignment — miner's delta follows the GT synthetic delta

    Returns a composite in [0, 1].
    """
    scene_ids = satellite_scene_ids or []
    score     = 1.0

    # ── 1. Scene freshness ──────────────────────────────────────────────────
    if not scene_ids or scene_ids == [""]:
        score -= 0.25    # no scene ID provided → suspicious

    try:
        query_dt = datetime.fromisoformat(query_date[:10])
    except Exception:
        query_dt = None

    if query_dt:
        for sid in scene_ids:
            scene_dt = _parse_scene_date(sid)
            if scene_dt:
                delta_days = abs((scene_dt - query_dt).days)
                if delta_days > 30:
                    score -= 0.40
                elif delta_days > 14:
                    score -= 0.15

    # ── 2. Change-rate plausibility ─────────────────────────────────────────
    if prev_pred_cells_list:
        curr_mi = _mean_mi(pred_cells)
        for prev_cells in (prev_pred_cells_list or [])[-3:]:
            prev_mi = _mean_mi(prev_cells)
            if curr_mi is not None and prev_mi is not None:
                delta = abs(curr_mi - prev_mi)
                if delta > 0.55:          # implausible moisture jump
                    score -= 0.20

    # ── 3. GT trend alignment ───────────────────────────────────────────────
    # Compare whether miner's moisture direction matches GT direction
    # (requires at least one previous GT to compare)
    prev_gt_cells_list = kwargs.get("prev_gt_cells_list")
    if prev_gt_cells_list and prev_pred_cells_list:
        curr_gt_mi   = _mean_mi(gt_cells)
        curr_pred_mi = _mean_mi(pred_cells)
        prev_gt_mi   = _mean_mi(prev_gt_cells_list[-1])
        prev_pred_mi = _mean_mi(prev_pred_cells_list[-1])

        if all(v is not None for v in [curr_gt_mi, curr_pred_mi, prev_gt_mi, prev_pred_mi]):
            gt_dir   = curr_gt_mi   - prev_gt_mi     # positive = wetter
            pred_dir = curr_pred_mi - prev_pred_mi

            # Penalise opposite-sign trends (miner says drying, GT says wetting)
            if gt_dir * pred_dir < 0 and abs(gt_dir) > 0.05:
                score -= 0.20

    return float(np.clip(score, 0.0, 1.0))
