"""
Five-dimension scoring for FieldAnalysisResponse objects.

Dimensions and weights (from the design doc):
  Spatial Accuracy     35 %  – cross-miner consensus
  Spectral Validity    20 %  – physics-bound NDVI / NDWI checks
  Temporal Consistency 20 %  – scene freshness + change-rate plausibility
  Confidence Calibration 15 % – self-reported vs actual accuracy (ECE proxy)
  Response Latency     10 %  – penalise < 2 s and > 90 s

Final score uses an exponential moving average (alpha=0.3) over the last
20 epochs stored in the database.
"""

import math
import statistics
from collections import Counter
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from validator.challenge.challenge_types import FieldAnalysisResponse, ScoringResult
from validator.evaluation.canary_fields import CANARY_FIELDS, canary_passes


# ---------------------------------------------------------------------------
# Latency score (pure function, design-doc table)
# ---------------------------------------------------------------------------

def latency_score(processing_time_s: float) -> float:
    if processing_time_s < 2.0:
        return 0.0          # suspiciously fast
    if processing_time_s <= 8.0:
        return 1.0
    if processing_time_s <= 30.0:
        return 0.7
    if processing_time_s <= 90.0:
        return 0.4
    return 0.0              # timeout


# ---------------------------------------------------------------------------
# Spectral validity score
# ---------------------------------------------------------------------------

def spectral_score(response: FieldAnalysisResponse) -> float:
    features = response.geojson_grid.get("features", [])
    if not features:
        return 0.0

    penalties = 0.0
    n = len(features)
    cloud_count = 0

    for feat in features:
        props = feat.get("properties", {})
        ndvi = props.get("ndvi", 0.0)
        ndwi = props.get("ndwi", 0.0)
        mi = props.get("moisture_index", 0.0)

        # Physics bounds
        if not -1.0 <= ndvi <= 1.0:
            penalties += 1.0
        if not -1.0 <= ndwi <= 1.0:
            penalties += 0.5
        if not 0.0 <= mi <= 1.0:
            penalties += 0.5

        # Cross-band consistency: cannot have very high NDWI and very low moisture
        if ndwi > 0.3 and mi < 0.2:
            penalties += 0.3

        if props.get("cloud_masked", False):
            cloud_count += 1

    cloud_fraction = cloud_count / n
    # More than 30 % cloud-masked degrades confidence
    cloud_penalty = max(0.0, (cloud_fraction - 0.30) * 2.0)

    penalty_rate = (penalties / n) + cloud_penalty
    return float(np.clip(1.0 - penalty_rate, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Spatial accuracy score (cross-miner consensus)
# ---------------------------------------------------------------------------

def spatial_score(
    response: FieldAnalysisResponse,
    all_responses: List[FieldAnalysisResponse],
    is_canary: bool = False,
    canary_expected_class: Optional[str] = None,
) -> float:
    if is_canary and canary_expected_class:
        return _canary_spatial_score(response, canary_expected_class)

    return _consensus_spatial_score(response, all_responses)


def _canary_spatial_score(response: FieldAnalysisResponse, expected_class: str) -> float:
    features = response.geojson_grid.get("features", [])
    if not features:
        return 0.0

    classes = [
        f["properties"].get("moisture_class", "CRITICAL_DRY")
        for f in features
        if not f["properties"].get("cloud_masked", False)
    ]
    if not classes:
        return 0.5

    majority = Counter(classes).most_common(1)[0][0]
    return 1.0 if canary_passes(expected_class, majority) else 0.0


def _consensus_spatial_score(
    response: FieldAnalysisResponse,
    all_responses: List[FieldAnalysisResponse],
) -> float:
    if len(all_responses) < 2:
        return 0.7          # no consensus possible — neutral score

    n_cells = len(response.geojson_grid.get("features", []))
    if n_cells == 0:
        return 0.0

    # Build per-cell consensus from all miners
    cell_votes: Dict[int, Counter] = {}
    for resp in all_responses:
        for feat in resp.geojson_grid.get("features", []):
            props = feat.get("properties", {})
            cid = props.get("cell_id", -1)
            cls = props.get("moisture_class", "")
            if cid >= 0 and cls:
                cell_votes.setdefault(cid, Counter())[cls] += 1

    matches = 0
    total = 0
    for feat in response.geojson_grid.get("features", []):
        props = feat.get("properties", {})
        if props.get("cloud_masked", False):
            continue
        cid = props.get("cell_id", -1)
        cls = props.get("moisture_class", "")
        if cid in cell_votes and cls:
            consensus = cell_votes[cid].most_common(1)[0][0]
            if cls == consensus:
                matches += 1
            total += 1

    return float(matches / max(total, 1))


# ---------------------------------------------------------------------------
# Temporal consistency score
# ---------------------------------------------------------------------------

def temporal_score(
    response: FieldAnalysisResponse,
    query_date: str,
    prev_responses: Optional[List[FieldAnalysisResponse]] = None,
) -> float:
    score = 1.0

    # Check scene date freshness
    scene_ids = response.satellite_scene_ids
    if not scene_ids or scene_ids == [""]:
        score -= 0.30     # no scene ID — suspicious

    # If scene_id encodes a date (Sentinel-2 format: S2A_MSIL2A_YYYYMMDD...)
    # verify it falls within ±14 days of query_date
    for sid in scene_ids:
        if sid.startswith("S2") and len(sid) > 20:
            try:
                date_str = sid.split("_")[2][:8]   # YYYYMMDD
                scene_dt = datetime.strptime(date_str, "%Y%m%d")
                query_dt = datetime.fromisoformat(query_date[:10])
                delta_days = abs((scene_dt - query_dt).days)
                if delta_days > 14:
                    score -= 0.40
            except Exception:
                pass

    # If we have previous responses for this miner, check moisture change rate
    if prev_responses:
        curr_mi_values = _mean_moisture_indices(response)
        for prev in prev_responses[-3:]:
            prev_mi_values = _mean_moisture_indices(prev)
            if curr_mi_values is not None and prev_mi_values is not None:
                delta = abs(curr_mi_values - prev_mi_values)
                # Jump > 0.60 in moisture without ERA5 rain event is implausible
                if delta > 0.60:
                    score -= 0.20

    return float(np.clip(score, 0.0, 1.0))


def _mean_moisture_indices(response: FieldAnalysisResponse) -> Optional[float]:
    values = [
        f["properties"].get("moisture_index", None)
        for f in response.geojson_grid.get("features", [])
        if not f["properties"].get("cloud_masked", False)
    ]
    valid = [v for v in values if v is not None]
    return float(np.mean(valid)) if valid else None


# ---------------------------------------------------------------------------
# Confidence calibration score (ECE proxy)
# ---------------------------------------------------------------------------

def confidence_calibration_score(
    response: FieldAnalysisResponse,
    spatial_accuracy: float,
) -> float:
    reported = float(response.confidence_overall)
    # Clamp both to [0,1]
    reported = max(0.0, min(1.0, reported))
    actual = max(0.0, min(1.0, spatial_accuracy))

    calibration_error = abs(reported - actual)

    # If miner always reports high confidence but is wrong, penalise hard
    if reported > 0.85 and actual < 0.50:
        return 0.0

    return float(np.clip(1.0 - calibration_error * 1.5, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Plagiarism detection
# ---------------------------------------------------------------------------

def compute_response_vector(response: FieldAnalysisResponse) -> Optional[np.ndarray]:
    """Flatten moisture_index values into a vector for cosine similarity."""
    values = [
        f["properties"].get("moisture_index", 0.0)
        for f in response.geojson_grid.get("features", [])
    ]
    if not values:
        return None
    return np.array(values, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def detect_plagiarism(responses: List[FieldAnalysisResponse]) -> Dict[str, bool]:
    """Return {miner_hotkey: is_plagiarist} for responses with similarity > 0.99."""
    vectors = {r.miner_hotkey: compute_response_vector(r) for r in responses}
    plagiarists: Dict[str, bool] = {r.miner_hotkey: False for r in responses}

    keys = list(vectors.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            va, vb = vectors[keys[i]], vectors[keys[j]]
            if va is None or vb is None or len(va) != len(vb):
                continue
            sim = cosine_similarity(va, vb)
            if sim > 0.99:
                plagiarists[keys[i]] = True
                plagiarists[keys[j]] = True
                logger.warning(
                    f"Plagiarism detected: {keys[i][:8]} ↔ {keys[j][:8]} "
                    f"(similarity={sim:.4f})"
                )
    return plagiarists


# ---------------------------------------------------------------------------
# Full scoring pipeline for a batch of responses to one challenge
# ---------------------------------------------------------------------------

def score_responses(
    responses: List[FieldAnalysisResponse],
    query_date: str,
    is_canary: bool = False,
    canary_expected_class: Optional[str] = None,
    prev_responses_by_hotkey: Optional[Dict[str, List[FieldAnalysisResponse]]] = None,
) -> Dict[str, ScoringResult]:
    """
    Score every response in the batch and return {miner_hotkey: ScoringResult}.

    Applies EMA smoothing and plagiarism penalties.
    """
    if not responses:
        return {}

    plagiarists = detect_plagiarism(responses)
    results: Dict[str, ScoringResult] = {}

    for resp in responses:
        proc_s = resp.processing_time_ms / 1000.0

        lat_score = latency_score(proc_s)
        spec_score = spectral_score(resp)
        spat_score = spatial_score(resp, responses, is_canary, canary_expected_class)
        prev = (prev_responses_by_hotkey or {}).get(resp.miner_hotkey)
        temp_score = temporal_score(resp, query_date, prev)
        conf_score = confidence_calibration_score(resp, spat_score)

        sr = ScoringResult(
            spatial_score=spat_score,
            spectral_score=spec_score,
            temporal_score=temp_score,
            confidence_score=conf_score,
            latency_score=lat_score,
        )
        sr.compute_final()

        # Plagiarism: cap to single-miner weight (hard penalty)
        if plagiarists.get(resp.miner_hotkey, False):
            sr.final_score *= 0.20
            sr.feedback = "Plagiarism detected — score capped"

        results[resp.miner_hotkey] = sr

    return results
