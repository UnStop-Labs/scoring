"""
METRIC_REGISTRY — pluggable pillar system, mirroring TurboVision's
scorevision/vlm_pipeline/non_vlm_scoring/objects.py pattern.

Each metric is registered under a (ElementType, PillarName) key.
The scoring engine iterates the element's declared pillars, looks up
the function here, calls it, and produces a weighted mean.

Usage
-----
    @register_metric((ElementType.IRRIGATION_DETECTION, PillarName.MOISTURE_CLASS_F1))
    def my_metric(gt_cells, pred_cells, **kwargs) -> float:
        ...
"""

from __future__ import annotations

from enum import Enum
from typing import Callable

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ElementType(str, Enum):
    """Task categories — one element per evaluation unit."""
    IRRIGATION_DETECTION = "irrigation_detection"
    MOISTURE_ESTIMATION  = "moisture_estimation"
    CHANGE_DETECTION     = "change_detection"


class PillarName(str, Enum):
    """Scoring dimensions — each maps to one metric function."""
    MOISTURE_CLASS_F1   = "moisture_class_f1"    # correct WET/DRY/OPTIMAL/CRITICAL_DRY label
    MOISTURE_INDEX_MAE  = "moisture_index_mae"   # numeric soil-moisture accuracy
    SPECTRAL_VALIDITY   = "spectral_validity"    # physics-bound NDVI / NDWI checks
    TEMPORAL_CONSISTENCY = "temporal_consistency" # scene freshness + change-rate plausibility
    CONFIDENCE_CALIBRATION = "confidence_calibration"  # ECE — self-reported vs actual
    FALSE_ALARM_RATE    = "false_alarm_rate"     # spurious WET predictions in DRY areas


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

METRIC_REGISTRY: dict[tuple[ElementType, PillarName], Callable] = {}


def register_metric(key: tuple[ElementType, PillarName]) -> Callable:
    """Decorator that registers a scoring function in METRIC_REGISTRY."""
    def decorator(fn: Callable) -> Callable:
        METRIC_REGISTRY[key] = fn
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Element configuration
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field as dc_field


@dataclass
class ElementConfig:
    """
    Defines which pillars are active for one task type, their weights,
    and the baseline_theta gate.

    Mirrors TurboVision's manifest Element / MetricConfig.
    """
    element_type:    ElementType
    pillars:         dict[PillarName, float]   # pillar → weight (must sum to 1)
    baseline_theta:  float = 0.30              # score below this → 0 after gating

    def weight_score(self, raw_score: float) -> float:
        """
        Apply baseline gating.
        Scores ≤ baseline_theta map to 0; scores above are linearly
        re-mapped to (0, 1] so 1.0 still means perfect.

        Mirrors TurboVision Element.weight_score().
        """
        if raw_score <= self.baseline_theta:
            return 0.0
        return (raw_score - self.baseline_theta) / (1.0 - self.baseline_theta)


# Default element configurations ----------------------------------------

IRRIGATION_DETECTION_ELEMENT = ElementConfig(
    element_type=ElementType.IRRIGATION_DETECTION,
    baseline_theta=0.30,
    pillars={
        PillarName.MOISTURE_CLASS_F1:      0.35,
        PillarName.MOISTURE_INDEX_MAE:     0.20,
        PillarName.SPECTRAL_VALIDITY:      0.15,
        PillarName.FALSE_ALARM_RATE:       0.15,
        PillarName.TEMPORAL_CONSISTENCY:   0.10,
        PillarName.CONFIDENCE_CALIBRATION: 0.05,
    },
)

CHANGE_DETECTION_ELEMENT = ElementConfig(
    element_type=ElementType.CHANGE_DETECTION,
    baseline_theta=0.25,
    pillars={
        PillarName.TEMPORAL_CONSISTENCY:   0.40,
        PillarName.MOISTURE_CLASS_F1:      0.30,
        PillarName.FALSE_ALARM_RATE:       0.20,
        PillarName.SPECTRAL_VALIDITY:      0.10,
    },
)
