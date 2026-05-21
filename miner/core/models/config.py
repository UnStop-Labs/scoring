from pydantic import BaseModel, Field
from typing import Any
import httpx
from substrateinterface import Keypair
from fiber.chain.metagraph import Metagraph
from fiber.miner.security.nonce_management import NonceManager


class Config(BaseModel):
    device: str = Field(default="cpu")
    keypair: Keypair
    metagraph: Metagraph
    min_stake_threshold: float = Field(default=1000.0)
    httpx_client: httpx.AsyncClient
    nonce_manager: Any

    class Config:
        arbitrary_types_allowed = True
