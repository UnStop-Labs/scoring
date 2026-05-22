"""
Metrics package — import side-effects register all pillar functions.

Order matters: registry must be imported before the metric modules
so that @register_metric has METRIC_REGISTRY available.
"""

from validator.evaluation.metrics.registry import (   # noqa: F401
    METRIC_REGISTRY,
    ElementConfig,
    ElementType,
    PillarName,
    register_metric,
    IRRIGATION_DETECTION_ELEMENT,
    CHANGE_DETECTION_ELEMENT,
)

# Trigger registration of all metric functions
import validator.evaluation.metrics.moisture     # noqa: F401
import validator.evaluation.metrics.spectral     # noqa: F401
import validator.evaluation.metrics.temporal     # noqa: F401
import validator.evaluation.metrics.calibration  # noqa: F401
