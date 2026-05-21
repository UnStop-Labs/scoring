from fastapi import APIRouter
from pydantic import BaseModel
from loguru import logger
from miner.utils.shared import miner_lock


class AvailabilityResponse(BaseModel):
    available: bool


router = APIRouter()


@router.get("/availability", response_model=AvailabilityResponse)
async def check_availability():
    is_available = not miner_lock.locked()
    logger.info(f"Availability check: {is_available}")
    return AvailabilityResponse(available=is_available)
