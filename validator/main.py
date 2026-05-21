"""
Validator main entry point.

Spawns three long-running processes/tasks:
  1. challenge_proc  – sends field challenges to miners (multiprocessing.Process)
  2. evaluation_proc – evaluates miner responses and stores scores
  3. weights_task    – periodically sets on-chain weights (asyncio task)
"""

import asyncio
import os
import sys
from multiprocessing import Process
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fiber.chain import fetch_nodes
from fiber.chain.chain_utils import load_coldkeypub_keypair, load_hotkey_keypair
from fiber.chain.interface import get_substrate
from fiber.chain.models import Node
from loguru import logger

from validator.challenge.challenge_process import start_challenge_sender
from validator.config import (
    DB_PATH,
    HOTKEY_NAME,
    MAX_MINERS,
    NETUID,
    SUBTENSOR_ADDRESS,
    SUBTENSOR_NETWORK,
    WALLET_NAME,
    WEIGHTS_INTERVAL,
)
from validator.db.operations import DatabaseManager
from validator.db.schema import init_db
from validator.evaluation.evaluation_process import start_evaluation
from validator.evaluation.set_weights import set_weights


# ---------------------------------------------------------------------------
# Node helpers (used by challenge_process via import)
# ---------------------------------------------------------------------------

def get_active_nodes() -> list:
    try:
        substrate = get_substrate(
            subtensor_network=SUBTENSOR_NETWORK,
            subtensor_address=SUBTENSOR_ADDRESS,
        )
        nodes = fetch_nodes.get_nodes_for_netuid(substrate, NETUID)
        MAX_STAKE = 999
        active = [n for n in nodes if n.stake < MAX_STAKE]
        logger.info(f"Active nodes: {len(active)}/{len(nodes)}")
        return active
    except Exception as exc:
        logger.error(f"get_active_nodes failed: {exc}")
        return []


def construct_server_address(node: Node) -> str:
    if node.ip == "0.0.0.1":
        return f"http://127.0.0.1:{node.port}"
    return f"http://{node.ip}:{node.port}"


# ---------------------------------------------------------------------------
# Weight-setting loop
# ---------------------------------------------------------------------------

async def weights_update_loop(db_manager: DatabaseManager) -> None:
    logger.info("Weight update loop starting")
    failures = 0
    while True:
        try:
            await set_weights(db_manager)
            failures = 0
            await asyncio.sleep(WEIGHTS_INTERVAL.total_seconds())
        except Exception as exc:
            failures += 1
            logger.error(f"Weight update error #{failures}: {exc}")
            wait = WEIGHTS_INTERVAL.total_seconds() * (2 if failures >= 3 else 1)
            await asyncio.sleep(wait)
            if failures >= 3:
                failures = 0


# ---------------------------------------------------------------------------
# Periodic cleanup
# ---------------------------------------------------------------------------

async def periodic_cleanup(db_manager: DatabaseManager, interval_hours: int = 24) -> None:
    while True:
        try:
            db_manager.cleanup_old_data(days=7)
            logger.info("Database cleanup done")
        except Exception as exc:
            logger.error(f"Cleanup error: {exc}")
        await asyncio.sleep(interval_hours * 3600)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    load_dotenv()

    hotkey = load_hotkey_keypair(WALLET_NAME, HOTKEY_NAME)
    logger.info(f"Validator hotkey: {hotkey.ss58_address}")

    init_db(str(DB_PATH))
    db_manager = DatabaseManager(DB_PATH)

    # Pass env vars to sub-processes
    os.environ["DB_PATH"] = str(DB_PATH)
    os.environ["VALIDATOR_HOTKEY"] = hotkey.ss58_address

    evaluation_proc = Process(target=start_evaluation)
    evaluation_proc.start()
    logger.info(f"Evaluation subprocess PID {evaluation_proc.pid}")

    challenge_proc = Process(target=start_challenge_sender)
    challenge_proc.start()
    logger.info(f"Challenge subprocess PID {challenge_proc.pid}")

    weights_task = asyncio.create_task(weights_update_loop(db_manager))
    cleanup_task = asyncio.create_task(periodic_cleanup(db_manager))

    try:
        iteration = 0
        while True:
            iteration += 1
            logger.debug(f"Main loop iteration {iteration}")

            # Restart any crashed background tasks
            if weights_task.done():
                logger.warning("Restarting weights task")
                weights_task = asyncio.create_task(weights_update_loop(db_manager))
            if cleanup_task.done():
                logger.warning("Restarting cleanup task")
                cleanup_task = asyncio.create_task(periodic_cleanup(db_manager))

            await asyncio.sleep(60)

    except KeyboardInterrupt:
        logger.info("Shutting down…")
    finally:
        weights_task.cancel()
        cleanup_task.cancel()
        await asyncio.gather(weights_task, cleanup_task, return_exceptions=True)

        for proc in (evaluation_proc, challenge_proc):
            if proc.is_alive():
                proc.terminate()
                proc.join()

        db_manager.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
