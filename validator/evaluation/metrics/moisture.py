"""
Moisture-class and moisture-index pillar metrics.

MOISTURE_CLASS_F1
-----------------
Mirrors TurboVision's _auc_f1() / _hungarian_f1() pattern.

Since irrigation cells are on a predefined deterministic grid, cell matching
is done by cell_id (O(1)). For mismatched counts we fall back to nearest-cell
matching using a cost matrix + scipy.optimize.linear_sum_assignment
(the Hungarian algorithm), exactly as TurboVision does for bounding boxes.

Two strictness levels are averaged to form an AUC-style score:
  • strict : exact class match only
  • lenient: ±1 ordinal step counts as correct
             (e.g. predicting DRY for a CRITICAL_DRY cell is half-wrong)

MOISTURE_INDEX_MAE
------------------
Mean Absolute Error of the numeric moisture_index field.
Normalised so MAE ≤ 0.10 → score 1.0, MAE ≥ 0.40 → score 0.0.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from validator.evaluation.metrics.registry import (
    ElementType, PillarName, register_metric,
)
from validator.evaluation.synthetic_gt import CLASS_ORDINAL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _class_ordinal(cls: str) -> int:
    return CLASS_ORDINAL.get(cls, 1)   # unknown → DRY


def _strict_match(gt_cls: str, pred_cls: str) -> float:
    return 1.0 if gt_cls == pred_cls else 0.0


def _lenient_match(gt_cls: str, pred_cls: str) -> float:
    """1 ordinal step off → 0.5, 2 steps → 0.0."""
    delta = abs(_class_ordinal(gt_cls) - _class_ordinal(pred_cls))
    return max(0.0, 1.0 - delta * 0.5)


def _f1_from_matches(matches: list[float], n_gt: int, n_pred: int) -> float:
    if n_gt == 0 and n_pred == 0:
        return 1.0
    if n_gt == 0 or n_pred == 0:
        return 0.0
    tp = sum(matches)
    precision = tp / n_pred
    recall    = tp / n_gt
    denom = precision + recall
    return float(2 * precision * recall / denom) if denom > 0 else 0.0


def _align_cells(
    gt_cells:   list[dict[str, Any]],
    pred_cells: list[dict[str, Any]],
) -> list[tuple[dict, dict]]:
    """
    Return a list of (gt_cell, pred_cell) pairs.

    Fast path: cell_id sets overlap ≥ 80 % → match by cell_id directly.
    Slow path: Hungarian matching on |cell_id difference| cost matrix
               (gracefully handles shifted or truncated grids).
    """
    gt_by_id   = {c["cell_id"]: c for c in gt_cells   if not c.get("cloud_masked")}
    pred_by_id = {c["cell_id"]: c for c in pred_cells if not c.get("cloud_masked")}

    shared = set(gt_by_id) & set(pred_by_id)

    # Fast path
    if len(shared) >= 0.80 * max(len(gt_by_id), 1):
        return [(gt_by_id[cid], pred_by_id[cid]) for cid in shared]

    # Slow path — Hungarian on cell_id distance
    gt_list   = list(gt_by_id.values())
    pred_list = list(pred_by_id.values())
    if not gt_list or not pred_list:
        return []

    cost = np.abs(
        np.array([c["cell_id"] for c in gt_list  ])[:, None] -
        np.array([c["cell_id"] for c in pred_list])[None, :]
    ).astype(np.float32)

    row_idx, col_idx = linear_sum_assignment(cost)

    MAX_ID_GAP = 5   # don't pair cells that are too far apart in ID space
    pairs = [
        (gt_list[r], pred_list[c])
        for r, c in zip(row_idx, col_idx)
        if cost[r, c] <= MAX_ID_GAP
    ]
    return pairs


# ---------------------------------------------------------------------------
# Pillar: MOISTURE_CLASS_F1
# ---------------------------------------------------------------------------

def _moisture_class_f1_at(
    pairs: list[tuple[dict, dict]],
    n_gt:  int,
    n_pred: int,
    strictness: str,   # "strict" | "lenient"
) -> float:
    match_fn = _strict_match if strictness == "strict" else _lenient_match
    matches  = [match_fn(g["moisture_class"], p["moisture_class"]) for g, p in pairs]
    return _f1_from_matches(matches, n_gt, n_pred)


@register_metric((ElementType.IRRIGATION_DETECTION, PillarName.MOISTURE_CLASS_F1))
@register_metric((ElementType.CHANGE_DETECTION,     PillarName.MOISTURE_CLASS_F1))
def compute_moisture_class_f1(
    gt_cells:   list[dict[str, Any]],
    pred_cells: list[dict[str, Any]],
    **kwargs,
) -> float:
    """
    AUC-F1 averaged over two strictness levels (strict + lenient).

    Analogous to TurboVision's _auc_f1([0.3, 0.5]) for bounding boxes.
    """
    pairs  = _align_cells(gt_cells, pred_cells)
    n_gt   = sum(1 for c in gt_cells   if not c.get("cloud_masked"))
    n_pred = sum(1 for c in pred_cells if not c.get("cloud_masked"))

    if not pairs:
        return 0.0

    strict_f1  = _moisture_class_f1_at(pairs, n_gt, n_pred, "strict")
    lenient_f1 = _moisture_class_f1_at(pairs, n_gt, n_pred, "lenient")

    # Weighted: stricter level counts more (mirrors higher IoU threshold weighting)
    return float(0.60 * strict_f1 + 0.40 * lenient_f1)


# ---------------------------------------------------------------------------
# Pillar: MOISTURE_INDEX_MAE
# ---------------------------------------------------------------------------

@register_metric((ElementType.IRRIGATION_DETECTION, PillarName.MOISTURE_INDEX_MAE))
@register_metric((ElementType.MOISTURE_ESTIMATION,  PillarName.MOISTURE_INDEX_MAE))
def compute_moisture_index_mae(
    gt_cells:   list[dict[str, Any]],
    pred_cells: list[dict[str, Any]],
    **kwargs,
) -> float:
    """
    Normalised MAE of the numeric moisture_index field.

    score = max(0, 1 - MAE / MAE_CEILING)
    MAE_CEILING = 0.35 (35 % moisture-index error → score 0)
    """
    MAE_CEILING = 0.35
    pairs = _align_cells(gt_cells, pred_cells)
    if not pairs:
        return 0.0

    errors = [
        abs(g["moisture_index"] - p["moisture_index"])
        for g, p in pairs
    ]
    mae = float(np.mean(errors))
    return float(max(0.0, 1.0 - mae / MAE_CEILING))
