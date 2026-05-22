"""
AgriScore v2 — TurboVision-inspired multi-pillar scoring engine.

Pipeline (mirrors scorevision/utils/evaluate.py → post_vlm_ranking):

  1.  Generate synthetic GT from challenge nonce + coordinates
  2.  Extract miner cells from GeoJSON response
  3.  Validate cell count (reject wild mismatches)
  4.  Run METRIC_REGISTRY pillars → weighted mean (acc)
  5.  Processing-rate gate  (latency too fast/slow → score = 0)
  6.  Baseline gating       (acc < theta → score = 0)
  7.  Plagiarism penalty    (cosine similarity > 0.99 → score × 0.20)
  8.  Return ScoringResultV2

Key differences from scoring v1
---------------------------------
v1  Cross-miner consensus for spatial accuracy (gameable by colluding miners)
v2  Synthetic deterministic GT → pillar metrics → baseline gate → rate gate

Anti-copy (tiebreak)
--------------------
When two miners produce nearly identical responses, the one with the earlier
on-chain commit block wins.  `pick_winner_with_tiebreak()` implements this.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger

from validator.challenge.challenge_types import FieldAnalysisChallenge, FieldAnalysisResponse
from validator.evaluation.metrics import (
    METRIC_REGISTRY,
    ElementConfig,
    ElementType,
    PillarName,
    IRRIGATION_DETECTION_ELEMENT,
)
from validator.evaluation.synthetic_gt import generate_synthetic_gt, cells_from_geojson


# ---------------------------------------------------------------------------
# Processing-rate gate  (analog of TurboVision RTF gate)
# ---------------------------------------------------------------------------

# How fast a response must arrive to be considered real inference
MIN_PROCESSING_S = 2.0     # below this → pre-computed / cheating
# Maximum tolerable latency
MAX_PROCESSING_S = 90.0    # above this → timeout


def processing_rate_pass(processing_time_ms: int) -> tuple[bool, str]:
    """
    Binary gate — mirrors TurboVision RTF > 1.0 → score = 0.

    Returns (passed, reason).
    """
    s = processing_time_ms / 1000.0
    if s < MIN_PROCESSING_S:
        return False, f"Response too fast ({s:.2f}s < {MIN_PROCESSING_S}s) — pre-computed?"
    if s > MAX_PROCESSING_S:
        return False, f"Response timeout ({s:.2f}s > {MAX_PROCESSING_S}s)"
    return True, ""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PillarResult:
    score:          float
    weight:         float
    weighted_score: float
    details:        dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoringResultV2:
    """Full scoring result for one (miner, challenge) pair."""

    # Per-pillar breakdown
    pillar_scores:     dict[str, PillarResult] = field(default_factory=dict)

    # Weighted mean of pillar scores (before gating)
    acc: float = 0.0

    # Final score after baseline gating + rate gate (this goes to Bittensor)
    score: float = 0.0

    # Gate outcomes
    rate_pass:      bool  = True
    rate_reason:    str   = ""
    baseline_theta: float = 0.0

    # Plagiarism
    plagiarism_detected: bool = False

    # Human-readable summary
    feedback: str = ""

    def summary(self) -> str:
        lines = [
            f"acc={self.acc:.3f}  score={self.score:.3f}  "
            f"rate_pass={self.rate_pass}  plagiarism={self.plagiarism_detected}",
        ]
        for name, pr in self.pillar_scores.items():
            lines.append(f"  {name:<28} {pr.score:.3f}  (w={pr.weight:.2f})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core scoring function
# ---------------------------------------------------------------------------

def score_response(
    challenge:     FieldAnalysisChallenge,
    response:      FieldAnalysisResponse,
    element:       ElementConfig | None = None,
    prev_pred_cells_list: list[list[dict]] | None = None,
    prev_gt_cells_list:   list[list[dict]] | None = None,
    area_km2:      float = 1.0,
) -> ScoringResultV2:
    """
    Score one (challenge, response) pair.

    Parameters
    ----------
    challenge             : the original challenge sent to the miner
    response              : the miner's reply
    element               : element configuration (defaults to IRRIGATION_DETECTION)
    prev_pred_cells_list  : last N cell lists from this miner (for temporal)
    prev_gt_cells_list    : last N synthetic GT cell lists (for temporal)
    area_km2              : tile area in km² (for false-alarm rate normalisation)
    """
    if element is None:
        element = IRRIGATION_DETECTION_ELEMENT

    result = ScoringResultV2(baseline_theta=element.baseline_theta)

    # ── 1. Processing-rate gate ──────────────────────────────────────────────
    rate_pass, rate_reason = processing_rate_pass(response.processing_time_ms)
    result.rate_pass   = rate_pass
    result.rate_reason = rate_reason

    if not rate_pass:
        result.score    = 0.0
        result.feedback = rate_reason
        logger.warning(f"Rate gate failed for {response.miner_hotkey[:8]}: {rate_reason}")
        return result

    # ── 2. Determine grid size from response ────────────────────────────────
    features = response.geojson_grid.get("features", [])
    n_cells  = len(features)
    if n_cells == 0:
        result.feedback = "Empty GeoJSON response"
        return result

    # ── 3. Generate synthetic GT ─────────────────────────────────────────────
    gt_cells = generate_synthetic_gt(
        nonce      = challenge.nonce,
        latitude   = challenge.latitude,
        longitude  = challenge.longitude,
        query_date = challenge.query_date,
        n_cells    = n_cells,
    )

    # ── 4. Extract miner cells ───────────────────────────────────────────────
    pred_cells = cells_from_geojson(response.geojson_grid)

    # Cell count sanity check (allow ≤ 10 % deviation)
    if abs(len(pred_cells) - len(gt_cells)) > max(2, 0.10 * len(gt_cells)):
        result.feedback = (
            f"Cell count mismatch: expected ~{len(gt_cells)}, got {len(pred_cells)}"
        )
        result.score = 0.0
        return result

    # ── 5. Common kwargs passed to all metric functions ──────────────────────
    shared_kwargs = dict(
        query_date           = challenge.query_date,
        satellite_scene_ids  = response.satellite_scene_ids,
        confidence_overall   = response.confidence_overall,
        prev_pred_cells_list = prev_pred_cells_list,
        prev_gt_cells_list   = prev_gt_cells_list,
        area_km2             = area_km2,
    )

    # ── 6. Run pillar metrics ────────────────────────────────────────────────
    total_weighted = 0.0
    total_weight   = 0.0

    for pillar_name, pillar_weight in element.pillars.items():
        key    = (element.element_type, pillar_name)
        metric = METRIC_REGISTRY.get(key)

        if metric is None:
            logger.debug(f"No metric registered for {key} — skipping")
            continue

        try:
            raw_score = metric(
                gt_cells=gt_cells,
                pred_cells=pred_cells,
                **shared_kwargs,
            )
        except Exception as exc:
            logger.error(f"Pillar {pillar_name} raised: {exc}", exc_info=True)
            raw_score = 0.0

        raw_score    = float(np.clip(raw_score, 0.0, 1.0))
        weighted     = raw_score * pillar_weight
        total_weighted += weighted
        total_weight   += pillar_weight

        result.pillar_scores[pillar_name.value] = PillarResult(
            score=raw_score,
            weight=pillar_weight,
            weighted_score=weighted,
        )

    # Weighted mean accuracy
    result.acc = total_weighted / max(total_weight, 1e-9)

    # ── 7. Baseline gating ───────────────────────────────────────────────────
    #  Mirrors TurboVision element.weight_score():
    #  score < theta → 0 ; above theta → linearly re-mapped to (0, 1]
    result.score = element.weight_score(result.acc)

    return result


# ---------------------------------------------------------------------------
# Batch scoring (one challenge → N miner responses)
# ---------------------------------------------------------------------------

def score_responses_v2(
    challenge:   FieldAnalysisChallenge,
    responses:   list[FieldAnalysisResponse],
    element:     ElementConfig | None = None,
    prev_cells_by_hotkey: dict[str, list[list[dict]]] | None = None,
    area_km2:    float = 1.0,
) -> dict[str, ScoringResultV2]:
    """
    Score all responses for one challenge.  Applies plagiarism penalty.

    Returns {miner_hotkey: ScoringResultV2}.
    """
    if not responses:
        return {}

    # Plagiarism detection (unchanged from v1)
    from validator.evaluation.calculate_score import detect_plagiarism
    plagiarists = detect_plagiarism(responses)

    results: dict[str, ScoringResultV2] = {}

    for resp in responses:
        prev = (prev_cells_by_hotkey or {}).get(resp.miner_hotkey)

        res = score_response(
            challenge  = challenge,
            response   = resp,
            element    = element,
            prev_pred_cells_list = prev,
            area_km2   = area_km2,
        )

        # Plagiarism cap (same as v1: score × 0.20)
        if plagiarists.get(resp.miner_hotkey, False):
            res.score                *= 0.20
            res.plagiarism_detected   = True
            res.feedback              = "Plagiarism detected — score capped at 20 %"
            logger.warning(f"Plagiarism: {resp.miner_hotkey[:8]} capped")

        results[resp.miner_hotkey] = res
        logger.info(
            f"[{resp.miner_hotkey[:8]}] acc={res.acc:.3f} "
            f"score={res.score:.3f} rate_pass={res.rate_pass}"
        )

    return results


# ---------------------------------------------------------------------------
# Anti-copy tiebreak  (mirrors TurboVision scoring.py pick_winner_with_tiebreak)
# ---------------------------------------------------------------------------

def are_responses_similar(
    cells_a: list[dict],
    cells_b: list[dict],
    delta_mi: float = 0.03,
) -> bool:
    """
    Two responses are 'similar' if their per-cell moisture_index values
    differ by < delta_mi on average.  This catches exact model copies.
    """
    ids_a = {c["cell_id"]: c["moisture_index"] for c in cells_a}
    ids_b = {c["cell_id"]: c["moisture_index"] for c in cells_b}
    shared = set(ids_a) & set(ids_b)
    if not shared:
        return False
    diffs = [abs(ids_a[cid] - ids_b[cid]) for cid in shared]
    return float(np.mean(diffs)) < delta_mi


def pick_winner_with_tiebreak(
    scores_by_hotkey: dict[str, float],
    commit_block_by_hotkey: dict[str, int],
    cells_by_hotkey:  dict[str, list[dict]] | None = None,
) -> str | None:
    """
    Select the best-scoring miner.  If two miners are suspiciously close
    (copy of each other), the one who committed to Bittensor *earlier* wins.

    Mirrors TurboVision winner.py / scoring.py tiebreak logic.

    Parameters
    ----------
    scores_by_hotkey        : {hotkey: final_score}
    commit_block_by_hotkey  : {hotkey: on-chain commit block number}
    cells_by_hotkey         : {hotkey: pred_cells} — optional, enables similarity check

    Returns
    -------
    Winning hotkey or None if scores_by_hotkey is empty.
    """
    if not scores_by_hotkey:
        return None

    ranked = sorted(scores_by_hotkey, key=lambda hk: -scores_by_hotkey[hk])
    winner = ranked[0]

    if cells_by_hotkey is None:
        return winner

    # Check each runner-up for suspicious similarity with the current winner
    for challenger in ranked[1:]:
        if scores_by_hotkey[challenger] < scores_by_hotkey[winner] - 0.10:
            break    # gap too large — stop comparing

        if are_responses_similar(
            cells_by_hotkey.get(winner, []),
            cells_by_hotkey.get(challenger, []),
        ):
            # Identical performance → original model (earlier commit) wins
            winner_block     = commit_block_by_hotkey.get(winner,     10**18)
            challenger_block = commit_block_by_hotkey.get(challenger, 10**18)
            if challenger_block < winner_block:
                logger.warning(
                    f"Tiebreak: {challenger[:8]} (block {challenger_block}) "
                    f"beats {winner[:8]} (block {winner_block}) — earlier commit"
                )
                winner = challenger

    return winner
