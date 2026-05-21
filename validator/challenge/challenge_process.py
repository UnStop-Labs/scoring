"""
Challenge sender loop — runs as a separate multiprocessing.Process.
Periodically picks Thai agricultural field coordinates, injects canary
fields at ~12 % rate, and broadcasts challenges to all active miners.
"""

import asyncio
import os
import random
import time
import uuid
from datetime import datetime, timezone, timedelta
from multiprocessing import Process

import httpx
from fiber.chain.interface import get_substrate
from fiber.chain.chain_utils import load_hotkey_keypair
from loguru import logger

from validator.challenge.challenge_types import FieldAnalysisChallenge
from validator.challenge.send_challenge import send_challenge
from validator.config import (
    CHALLENGE_INTERVAL,
    CHALLENGE_TIMEOUT,
    MAX_MINERS,
    NETUID,
    SUBTENSOR_ADDRESS,
    SUBTENSOR_NETWORK,
    WALLET_NAME,
    HOTKEY_NAME,
)
from validator.db.operations import DatabaseManager
from validator.evaluation.canary_fields import CANARY_FIELDS, REAL_FIELD_POOL
from validator.utils.async_utils import AsyncBarrier


def start_challenge_sender():
    asyncio.run(_challenge_loop())


async def _challenge_loop():
    from validator.main import get_active_nodes, construct_server_address

    wallet_name = os.getenv("WALLET_NAME", WALLET_NAME)
    hotkey_name = os.getenv("HOTKEY_NAME", HOTKEY_NAME)
    keypair = load_hotkey_keypair(wallet_name, hotkey_name)

    db_manager = DatabaseManager(os.environ["DB_PATH"])
    substrate = get_substrate(
        subtensor_network=SUBTENSOR_NETWORK,
        subtensor_address=SUBTENSOR_ADDRESS,
    )
    limits = httpx.Limits(max_connections=256, max_keepalive_connections=64)
    timeout = httpx.Timeout(
        connect=5.0,
        read=CHALLENGE_TIMEOUT.total_seconds(),
        write=10.0,
        pool=10,
    )

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        while True:
            try:
                nodes = get_active_nodes()
                if not nodes:
                    logger.warning("No active nodes — sleeping")
                    await asyncio.sleep(CHALLENGE_INTERVAL.total_seconds())
                    continue

                selected = random.sample(nodes, min(MAX_MINERS, len(nodes)))

                # Decide whether this round uses a canary field (~12 %)
                use_canary = random.random() < 0.12
                if use_canary:
                    field = random.choice(CANARY_FIELDS)
                    is_canary = True
                    logger.info(f"Injecting canary field: {field['name']}")
                else:
                    field = random.choice(REAL_FIELD_POOL)
                    is_canary = False

                query_date = (datetime.utcnow() - timedelta(days=random.randint(0, 10))).strftime("%Y-%m-%dT00:00:00")
                nonce = uuid.uuid4().hex + uuid.uuid4().hex  # 256-bit hex

                challenge = FieldAnalysisChallenge(
                    challenge_id=FieldAnalysisChallenge.make_id(
                        field["latitude"], field["longitude"], query_date, nonce
                    ),
                    latitude=field["latitude"],
                    longitude=field["longitude"],
                    area_rai=field.get("area_rai", 10.0),
                    query_date=query_date,
                    grid_resolution_m=20,
                    crop_hint=field.get("crop_hint"),
                    nonce=nonce,
                    validator_hotkey=keypair.ss58_address,
                    timestamp_utc=int(time.time()),
                    is_canary=is_canary,
                )

                barrier = AsyncBarrier(parties=len(selected))
                tasks = [
                    asyncio.create_task(
                        send_challenge(
                            challenge=challenge,
                            server_address=construct_server_address(node),
                            hotkey=node.hotkey,
                            keypair=keypair,
                            node_id=node.node_id,
                            barrier=barrier,
                            db_manager=db_manager,
                            client=client,
                        )
                    )
                    for node in selected
                ]

                await asyncio.gather(*tasks, return_exceptions=True)
                logger.info(f"Challenge {challenge.challenge_id} sent to {len(selected)} miners")

            except Exception as exc:
                logger.error(f"Challenge loop error: {exc}", exc_info=True)

            await asyncio.sleep(CHALLENGE_INTERVAL.total_seconds())
