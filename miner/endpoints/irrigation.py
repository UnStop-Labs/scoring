"""
Miner endpoint for FieldAnalysisChallenge.

POST /irrigation/challenge
  Receives a challenge with field coordinates + metadata.
  Fetches Sentinel-2 imagery, computes spectral indices, builds a GeoJSON grid,
  and returns a FieldAnalysisResponse.
"""

import hashlib
import json
import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from loguru import logger

from miner.core.models.config import Config
from miner.dependencies import blacklist_low_stake, get_config, verify_request
from miner.utils.geojson_builder import build_geojson_grid
from miner.utils.satellite import fetch_sentinel2_bands
from miner.utils.shared import miner_lock

MODEL_VERSION = "irrigation-miner-v1.0.0"

router = APIRouter()


async def _process_challenge(challenge_data: dict, config: Config) -> dict:
    challenge_id = challenge_data.get("challenge_id", "")
    latitude = float(challenge_data["latitude"])
    longitude = float(challenge_data["longitude"])
    area_rai = float(challenge_data.get("area_rai", 10.0))
    query_date = challenge_data.get("query_date", "")
    grid_resolution_m = int(challenge_data.get("grid_resolution_m", 20))
    nonce = challenge_data.get("nonce", "")
    timestamp_utc = int(challenge_data.get("timestamp_utc", 0))
    miner_hotkey = config.keypair.ss58_address

    # Reject stale challenges (> 90 seconds old)
    now_ts = int(time.time())
    if timestamp_utc and now_ts - timestamp_utc > 90:
        raise HTTPException(status_code=400, detail="Challenge timestamp expired")

    start_ms = time.time()

    # Fetch satellite bands
    bands = await fetch_sentinel2_bands(
        latitude=latitude,
        longitude=longitude,
        area_rai=area_rai,
        query_date=query_date,
    )

    # Build GeoJSON grid
    geojson_grid = build_geojson_grid(
        latitude=latitude,
        longitude=longitude,
        area_rai=area_rai,
        grid_resolution_m=grid_resolution_m,
        B03=bands["B03"],
        B04=bands["B04"],
        B08=bands["B08"],
        B11=bands["B11"],
        cloud_cover=bands.get("cloud_cover", 0.0),
        data_source=bands.get("source", "Sentinel-2"),
    )

    processing_time_ms = int((time.time() - start_ms) * 1000)

    # Compute overall confidence as mean of per-cell confidence
    cell_confs = [
        f["properties"]["cell_confidence"]
        for f in geojson_grid["features"]
        if not f["properties"]["cloud_masked"]
    ]
    confidence_overall = round(sum(cell_confs) / max(len(cell_confs), 1), 4)

    # Sign: challenge_id + nonce + grid hash
    grid_hash = hashlib.sha256(
        json.dumps(geojson_grid, sort_keys=True).encode()
    ).hexdigest()
    signature_payload = f"{challenge_id}:{nonce}:{grid_hash}"
    signature = config.keypair.sign(signature_payload.encode()).hex()

    return {
        "challenge_id": challenge_id,
        "miner_hotkey": miner_hotkey,
        "geojson_grid": geojson_grid,
        "processing_time_ms": processing_time_ms,
        "satellite_scene_ids": [bands.get("scene_id", "")],
        "model_version": MODEL_VERSION,
        "confidence_overall": confidence_overall,
        "signature": signature,
    }


async def process_challenge(
    request: Request,
    config: Config = Depends(get_config),
):
    logger.info("Acquiring miner lock for irrigation challenge…")
    async with miner_lock:
        try:
            challenge_data = await request.json()
            logger.info(
                f"Received challenge {challenge_data.get('challenge_id')} "
                f"for ({challenge_data.get('latitude')}, {challenge_data.get('longitude')})"
            )
            result = await _process_challenge(challenge_data, config)
            logger.info(
                f"Challenge {challenge_data.get('challenge_id')} completed "
                f"in {result['processing_time_ms']} ms"
            )
            return result
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(f"Error processing irrigation challenge: {exc}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))


router.add_api_route(
    "/challenge",
    process_challenge,
    tags=["irrigation"],
    dependencies=[Depends(blacklist_low_stake), Depends(verify_request)],
    methods=["POST"],
)
