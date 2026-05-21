import asyncio
import os

from loguru import logger
from validator.evaluation.evaluation_loop import run_evaluation_loop


def start_evaluation():
    logger.info("[EVAL_PROC] Starting evaluation subprocess…")
    try:
        db_path = os.getenv("DB_PATH")
        hotkey = os.getenv("VALIDATOR_HOTKEY")
        if not db_path or not hotkey:
            raise ValueError(f"Missing DB_PATH={db_path} or VALIDATOR_HOTKEY={hotkey}")
        asyncio.run(run_evaluation_loop(db_path=db_path, validator_hotkey=hotkey, sleep_interval=60))
    except Exception as exc:
        logger.exception(f"[EVAL_PROC] Crashed: {exc}")
