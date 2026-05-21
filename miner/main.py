from fastapi import FastAPI
from fiber.logging_utils import get_logger

from miner.core.models.config import Config
from miner.core.configuration import factory_config
from miner.dependencies import get_config
from miner.endpoints.irrigation import router as irrigation_router
from miner.endpoints.availability import router as availability_router

logger = get_logger(__name__)

app = FastAPI(title="Irrigation Subnet Miner")

app.dependency_overrides[Config] = get_config

app.include_router(irrigation_router, prefix="/irrigation", tags=["irrigation"])
app.include_router(availability_router, tags=["availability"])


if __name__ == "__main__":
    import uvicorn
    import os

    uvicorn.run(
        "miner.main:app",
        host="0.0.0.0",
        port=int(os.getenv("MINER_PORT", "8001")),
        reload=False,
    )
