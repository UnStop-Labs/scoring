import os
from datetime import timedelta
from pathlib import Path

from dotenv import load_dotenv

validator_dir = Path(__file__).parent
load_dotenv(validator_dir / ".env")

# Network
NETUID = int(os.getenv("NETUID", "1"))
SUBTENSOR_NETWORK = os.getenv("SUBTENSOR_NETWORK", "test")
SUBTENSOR_ADDRESS = os.getenv("SUBTENSOR_ADDRESS", "127.0.0.1:9944")

# Wallet
WALLET_NAME = os.getenv("WALLET_NAME", "default")
HOTKEY_NAME = os.getenv("HOTKEY_NAME", "default")

# Miner selection
MIN_MINERS = 1
MAX_MINERS = 60
MIN_STAKE_THRESHOLD = float(os.getenv("MIN_STAKE_THRESHOLD", "2"))

# Timing
CHALLENGE_INTERVAL = timedelta(minutes=int(os.getenv("CHALLENGE_INTERVAL_MINUTES", "12")))
CHALLENGE_TIMEOUT = timedelta(seconds=90)
WEIGHTS_INTERVAL = timedelta(minutes=30)
VALIDATION_DELAY = timedelta(minutes=5)

# Scoring
SCORE_THRESHOLD = 0.10          # quality floor
VERSION_KEY = int(os.getenv("VERSION_KEY", "1000"))

# Paths
DB_PATH = Path(os.getenv("DB_PATH", "validator.db"))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
