"""
Confidence calibration pillar metric (real ECE proxy).

A well-calibrated miner that reports confidence = 0.80 should be correct
on 80 % of cells.  This metric computes an Expected Calibration Error (ECE)
style score using the per-cell confidence field (if present) or falls back
to the global confidence_overall scalar.

ECE formula
-----------
Bin predictions into M = 10 equal-width confidence buckets [0,0.1), [0.1,0.2) …
For each non-empty bin b:
    accuracy_b = fraction of cells in b correctly classified
    conf_b     = mean confidence in b
    ECE += |b| / N * |accuracy_b − conf_b|

score = max(0, 1 − 1.5 * ECE)

The 1.5 multiplier applies a harsher penalty for miscalibration than for
a small accuracy loss — matching the TurboVision philosophy that
over-confident wrong predictions are worse than honest uncertain ones.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from validator.evaluation.metrics.registry import (
    ElementType, PillarName, register_metric,
)


# ---------------------------------------------------------------------------
# ECE computation
# ---------------------------------------------------------------------------

N_BINS = 10


def _compute_ece(
    confidences: list[float],
    corrects:    list[bool],
) -> float:
    if not confidences:
        return 0.5     # no data → neutral

    conf_arr = np.array(confidences, dtype=np.float64)
    corr_arr = np.array(corrects,    dtype=np.float64)
    n        = len(conf_arr)

    ece = 0.0
    edges = np.linspace(0.0, 1.0, N_BINS + 1)

    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf_arr >= lo) & (conf_arr < hi)
        if not np.any(mask):
            continue
        b_conf = conf_arr[mask].mean()
        b_acc  = corr_arr[mask].mean()
        ece   += mask.sum() / n * abs(b_conf - b_acc)

    return float(ece)


# ---------------------------------------------------------------------------
# Pillar: CONFIDENCE_CALIBRATION
# ---------------------------------------------------------------------------

@register_metric((ElementType.IRRIGATION_DETECTION, PillarName.CONFIDENCE_CALIBRATION))
@register_metric((ElementType.MOISTURE_ESTIMATION,  PillarName.CONFIDENCE_CALIBRATION))
def compute_confidence_calibration(
    gt_cells:           list[dict[str, Any]],
    pred_cells:         list[dict[str, Any]],
    confidence_overall: float = 0.5,
    **kwargs,
) -> float:
    """
    Prefer per-cell calibration (ECE) when cell_confidence is available.
    Fall back to global scalar comparison otherwise.

    Hard penalty: reported confidence > 0.85 AND actual accuracy < 0.45 → 0.
    """
    gt_by_id = {c["cell_id"]: c for c in gt_cells if not c.get("cloud_masked")}

    confidences: list[float] = []
    corrects:    list[bool]  = []

    for cell in pred_cells:
        if cell.get("cloud_masked"):
            continue
        conf = cell.get("cell_confidence", None)
        if conf is None:
            continue
        gt = gt_by_id.get(cell["cell_id"])
        if gt is None:
            continue
        confidences.append(float(np.clip(conf, 0.0, 1.0)))
        corrects.append(cell["moisture_class"] == gt["moisture_class"])

    if not confidences:
        # Fall back: compare global confidence_overall vs spatial F1 proxy
        reported = float(np.clip(confidence_overall, 0.0, 1.0))
        # Rough actual accuracy: mean of correct class predictions
        correct_count = sum(
            1 for p in pred_cells
            if not p.get("cloud_masked")
            and gt_by_id.get(p["cell_id"], {}).get("moisture_class") == p["moisture_class"]
        )
        n_valid = sum(1 for p in pred_cells if not p.get("cloud_masked"))
        actual  = correct_count / max(n_valid, 1)

        # Hard penalty
        if reported > 0.85 and actual < 0.45:
            return 0.0

        calib_error = abs(reported - actual)
        return float(np.clip(1.0 - 1.5 * calib_error, 0.0, 1.0))

    # Per-cell ECE
    actual_accuracy = float(np.mean(corrects))
    reported        = float(np.clip(confidence_overall, 0.0, 1.0))

    if reported > 0.85 and actual_accuracy < 0.45:
        return 0.0

    ece = _compute_ece(confidences, corrects)
    return float(np.clip(1.0 - 1.5 * ece, 0.0, 1.0))
