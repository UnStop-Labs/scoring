"""
Send a FieldAnalysisChallenge to a single miner node using the fiber protocol.
"""

import json
import time
from datetime import datetime, timezone
from typing import Optional

import httpx
from fiber import Keypair
from fiber.logging_utils import get_logger
from fiber.validator import client as validator_client

from validator.challenge.challenge_types import FieldAnalysisChallenge, FieldAnalysisResponse
from validator.config import CHALLENGE_TIMEOUT
from validator.utils.async_utils import AsyncBarrier

logger = get_logger(__name__)


async def send_challenge(
    challenge: FieldAnalysisChallenge,
    server_address: str,
    hotkey: str,
    keypair: Keypair,
    node_id: int,
    barrier: AsyncBarrier,
    db_manager,
    client: httpx.AsyncClient,
    timeout: float = CHALLENGE_TIMEOUT.total_seconds(),
) -> Optional[httpx.Response]:
    endpoint = "/irrigation/challenge"
    payload = challenge.to_dict()

    logger.info(
        f"Sending challenge {challenge.challenge_id} to node {node_id} "
        f"({challenge.latitude}, {challenge.longitude})"
    )

    remaining_barriers = 2
    response = None

    try:
        if db_manager:
            db_manager.store_challenge(
                challenge_id=challenge.challenge_id,
                latitude=challenge.latitude,
                longitude=challenge.longitude,
                area_rai=challenge.area_rai,
                query_date=challenge.query_date,
                is_canary=challenge.is_canary,
            )
            db_manager.assign_challenge(challenge.challenge_id, hotkey, node_id)

        await barrier.wait()
        remaining_barriers -= 1

        if db_manager:
            db_manager.mark_challenge_sent(challenge.challenge_id, hotkey)

        sent_time = datetime.now(timezone.utc)

        try:
            response = await validator_client.make_non_streamed_post(
                httpx_client=client,
                server_address=server_address,
                validator_ss58_address=keypair.ss58_address,
                miner_ss58_address=hotkey,
                keypair=keypair,
                endpoint=endpoint,
                payload=payload,
                timeout=timeout,
            )
            response.raise_for_status()
        except Exception as exc:
            logger.warning(f"Challenge send error node {node_id}: {exc}")
            response = httpx.Response(status_code=200, json={
                "challenge_id": challenge.challenge_id,
                "miner_hotkey": hotkey,
                "geojson_grid": {"type": "FeatureCollection", "features": []},
                "processing_time_ms": int(CHALLENGE_TIMEOUT.total_seconds() * 1000),
                "satellite_scene_ids": [],
                "model_version": "unknown",
                "confidence_overall": 0.0,
                "signature": "",
            })

        received_time = datetime.now(timezone.utc)
        processing_time_s = (received_time - sent_time).total_seconds()

        await barrier.wait()
        remaining_barriers -= 1

        try:
            resp_data = response.json()
            if not resp_data:
                resp_data = {
                    "challenge_id": challenge.challenge_id,
                    "geojson_grid": {"type": "FeatureCollection", "features": []},
                    "processing_time_ms": int(processing_time_s * 1000),
                    "satellite_scene_ids": [],
                    "model_version": "unknown",
                    "confidence_overall": 0.0,
                    "signature": "",
                }

            far = FieldAnalysisResponse(
                challenge_id=challenge.challenge_id,
                miner_hotkey=hotkey,
                geojson_grid=resp_data.get("geojson_grid", {"type": "FeatureCollection", "features": []}),
                processing_time_ms=resp_data.get("processing_time_ms", int(processing_time_s * 1000)),
                satellite_scene_ids=resp_data.get("satellite_scene_ids", []),
                model_version=resp_data.get("model_version", "unknown"),
                confidence_overall=float(resp_data.get("confidence_overall", 0.0)),
                signature=resp_data.get("signature", ""),
                node_id=node_id,
                received_at=sent_time,
            )

            if db_manager:
                db_manager.store_response(
                    challenge_id=challenge.challenge_id,
                    miner_hotkey=hotkey,
                    response=far,
                    node_id=node_id,
                    processing_time=processing_time_s,
                    received_at=sent_time,
                    completed_at=received_time,
                )

            logger.info(
                f"Stored response for challenge {challenge.challenge_id} "
                f"node {node_id} ({processing_time_s:.1f}s)"
            )

        except Exception as exc:
            logger.error(f"Error parsing response from node {node_id}: {exc}", exc_info=True)
            raise

        return response

    except Exception as exc:
        # Drain remaining barriers so other tasks aren't stuck
        for _ in range(remaining_barriers):
            try:
                await barrier.wait()
            except Exception:
                pass
        logger.error(f"send_challenge failed for node {node_id}: {exc}", exc_info=True)
        return None
