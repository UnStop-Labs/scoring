"""
Evaluation loop — runs as an async task inside the evaluation subprocess.

Periodically queries the DB for completed (but un-evaluated) challenge rounds,
runs IrrigationValidator.score_challenge_round(), and persists the scores.
"""

import asyncio
import time
from typing import Optional

from loguru import logger

from validator.challenge.challenge_types import FieldAnalysisResponse
from validator.db.operations import DatabaseManager
from validator.evaluation.evaluation import IrrigationValidator
from validator.evaluation.calculate_score import score_responses


async def run_evaluation_loop(
    db_path: str,
    validator_hotkey: str,
    sleep_interval: int = 60,
) -> None:
    logger.info("Evaluation loop starting")
    db = DatabaseManager(db_path)
    validator = IrrigationValidator(validator_hotkey=validator_hotkey)
    iteration = 0

    while True:
        try:
            iteration += 1
            challenges = db.get_pending_challenges_for_eval(delay_minutes=5)

            if not challenges:
                logger.debug(f"[eval #{iteration}] No pending challenges — sleeping {sleep_interval}s")
                await asyncio.sleep(sleep_interval)
                continue

            for ch in challenges:
                challenge_id = ch["challenge_id"]
                logger.info(f"[eval #{iteration}] Evaluating challenge {challenge_id}")

                try:
                    responses = db.get_pending_responses(challenge_id)
                    if not responses:
                        logger.warning(f"No responses for {challenge_id}")
                        continue

                    results = validator.score_challenge_round(
                        responses=responses,
                        query_date=ch["query_date"],
                        latitude=ch["latitude"],
                        longitude=ch["longitude"],
                        is_canary=bool(ch.get("is_canary", 0)),
                    )

                    for resp in responses:
                        sr = results.get(resp.miner_hotkey)
                        if sr is None:
                            continue

                        db.mark_response_evaluated(resp.response_id, sr.final_score)
                        db.store_response_score(
                            response_id=resp.response_id,
                            challenge_id=challenge_id,
                            miner_hotkey=resp.miner_hotkey,
                            node_id=resp.node_id or 0,
                            spatial_score=sr.spatial_score,
                            spectral_score=sr.spectral_score,
                            temporal_score=sr.temporal_score,
                            confidence_score=sr.confidence_score,
                            latency_score=sr.latency_score,
                            final_score=sr.final_score,
                            processing_time=resp.processing_time_ms / 1000.0,
                        )

                    logger.info(f"Scored {len(results)} responses for challenge {challenge_id}")

                except Exception as exc:
                    logger.error(f"Error evaluating challenge {challenge_id}: {exc}", exc_info=True)

        except Exception as exc:
            logger.error(f"Evaluation loop error: {exc}", exc_info=True)

        await asyncio.sleep(sleep_interval)
