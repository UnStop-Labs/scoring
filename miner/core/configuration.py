import os
from functools import lru_cache

import httpx
from dotenv import load_dotenv
from fiber.chain import chain_utils, interface
from fiber.chain.metagraph import Metagraph
from fiber.miner.security.nonce_management import NonceManager
from miner.core.models.config import Config

load_dotenv()


@lru_cache
def factory_config() -> Config:
    nonce_manager = NonceManager()

    wallet_name = os.getenv("WALLET_NAME", "default")
    hotkey_name = os.getenv("HOTKEY_NAME", "default")
    netuid = os.getenv("NETUID")
    subtensor_network = os.getenv("SUBTENSOR_NETWORK")
    subtensor_address = os.getenv("SUBTENSOR_ADDRESS")
    load_old_nodes = bool(os.getenv("LOAD_OLD_NODES", True))
    min_stake_threshold = float(os.getenv("MIN_STAKE_THRESHOLD", 1000))
    refresh_nodes = os.getenv("REFRESH_NODES", "true").lower() == "true"

    assert netuid is not None, "Must set NETUID env var!"

    if refresh_nodes:
        substrate = interface.get_substrate(subtensor_network, subtensor_address)
        metagraph = Metagraph(substrate=substrate, netuid=netuid, load_old_nodes=load_old_nodes)
    else:
        metagraph = Metagraph(substrate=None, netuid=netuid, load_old_nodes=load_old_nodes)

    keypair = chain_utils.load_hotkey_keypair(wallet_name, hotkey_name)

    return Config(
        nonce_manager=nonce_manager,
        keypair=keypair,
        metagraph=metagraph,
        min_stake_threshold=min_stake_threshold,
        httpx_client=httpx.AsyncClient(),
    )
