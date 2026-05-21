"""
Set on-chain miner weights based on 24-hour rolling average scores.

Weight formula from the design doc:
  weight[i] = smoothed_score[i] / Σ smoothed_score[j]
  Miners with score < 0.10 receive weight = 0 (quality floor).

Yuma Consensus reconciles disagreements between multiple validators.
"""

import asyncio
from typing import Dict, List

from fiber.chain import chain_utils, interface, weights
from fiber.chain.fetch_nodes import get_nodes_for_netuid
from fiber.logging_utils import get_logger

from validator.config import (
    HOTKEY_NAME,
    NETUID,
    SUBTENSOR_ADDRESS,
    SUBTENSOR_NETWORK,
    VERSION_KEY,
    WALLET_NAME,
)
from validator.db.operations import DatabaseManager

logger = get_logger(__name__)

QUALITY_FLOOR = 0.10   # miners below this receive weight = 0
EMA_ALPHA = 0.30       # smoothing factor for score EMA


async def set_weights(db_manager: DatabaseManager) -> None:
    try:
        substrate = interface.get_substrate(
            subtensor_network=SUBTENSOR_NETWORK,
            subtensor_address=SUBTENSOR_ADDRESS,
        )
        keypair = chain_utils.load_hotkey_keypair(WALLET_NAME, HOTKEY_NAME)

        validator_uid = substrate.query(
            "SubtensorModule", "Uids", [NETUID, keypair.ss58_address]
        ).value

        nodes = get_nodes_for_netuid(substrate=substrate, netuid=NETUID)
        miner_scores = db_manager.get_miner_scores(lookback_hours=24)

        logger.info(f"Setting weights for {len(nodes)} nodes; have scores for {len(miner_scores)}")

        node_ids: List[int] = []
        node_weights: List[float] = []

        for node in nodes:
            nid = node.node_id
            score_data = miner_scores.get(nid, {})
            raw_score = score_data.get("final_score", 0.0)

            # Quality floor
            adjusted = raw_score if raw_score >= QUALITY_FLOOR else 0.0
            node_ids.append(nid)
            node_weights.append(adjusted)

        total = sum(node_weights)
        if total > 0:
            node_weights = [w / total for w in node_weights]
        else:
            node_weights = [1.0 / len(nodes)] * len(nodes)

        for nid, w in zip(node_ids, node_weights):
            if w > 0:
                logger.info(f"  node {nid}: weight={w:.4f}")

        success = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                None,
                weights.set_node_weights,
                substrate,
                keypair,
                node_ids,
                node_weights,
                NETUID,
                validator_uid,
                VERSION_KEY,
                True,
                True,
            ),
            timeout=120.0,
        )

        if success:
            logger.info("Weights set successfully on chain")
        else:
            logger.error("set_node_weights returned False")

    except asyncio.TimeoutError:
        logger.error("set_weights timed out after 120 s")
    except Exception as exc:
        logger.error(f"set_weights error: {exc}", exc_info=True)
        raise
