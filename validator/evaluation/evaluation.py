"""
IrrigationValidator — orchestrates scoring for a completed challenge round.
"""

from typing import Any, Dict, List, Optional

from loguru import logger

from validator.challenge.challenge_types import FieldAnalysisResponse, ScoringResult
from validator.evaluation.calculate_score import score_responses
from validator.evaluation.canary_fields import CANARY_FIELDS


class IrrigationValidator:
    def __init__(self, validator_hotkey: str):
        self.validator_hotkey = validator_hotkey
        # Cache of recent responses per miner for temporal consistency checks
        self._response_history: Dict[str, List[FieldAnalysisResponse]] = {}

    def score_challenge_round(
        self,
        responses: List[FieldAnalysisResponse],
        query_date: str,
        latitude: float,
        longitude: float,
        is_canary: bool = False,
    ) -> Dict[str, ScoringResult]:
        """
        Score all miner responses for a single challenge.

        Returns {miner_hotkey: ScoringResult}.
        """
        if not responses:
            return {}

        canary_expected_class: Optional[str] = None
        if is_canary:
            canary_expected_class = self._find_canary_expected_class(latitude, longitude)

        results = score_responses(
            responses=responses,
            query_date=query_date,
            is_canary=is_canary,
            canary_expected_class=canary_expected_class,
            prev_responses_by_hotkey=self._response_history,
        )

        # Update history (keep last 20 per miner for EMA / temporal checks)
        for resp in responses:
            hist = self._response_history.setdefault(resp.miner_hotkey, [])
            hist.append(resp)
            if len(hist) > 20:
                hist.pop(0)

        for hotkey, sr in results.items():
            logger.info(
                f"  {hotkey[:10]}…: spatial={sr.spatial_score:.2f} "
                f"spectral={sr.spectral_score:.2f} temporal={sr.temporal_score:.2f} "
                f"conf={sr.confidence_score:.2f} latency={sr.latency_score:.2f} "
                f"→ final={sr.final_score:.4f}"
            )

        return results

    def _find_canary_expected_class(self, lat: float, lon: float) -> Optional[str]:
        best = None
        best_dist = float("inf")
        for c in CANARY_FIELDS:
            dist = ((c["latitude"] - lat) ** 2 + (c["longitude"] - lon) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best = c.get("expected_moisture_class")
        return best
