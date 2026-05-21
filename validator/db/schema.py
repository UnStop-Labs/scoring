import sqlite3
from pathlib import Path
from typing import List


SCHEMA_VERSION = 1


def get_schema() -> List[str]:
    return [
        """
        CREATE TABLE IF NOT EXISTS challenges (
            challenge_id TEXT PRIMARY KEY,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            area_rai REAL NOT NULL,
            query_date TEXT NOT NULL,
            is_canary INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS challenge_assignments (
            assignment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            challenge_id TEXT NOT NULL,
            miner_hotkey TEXT NOT NULL,
            node_id INTEGER NOT NULL,
            assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            sent_at TIMESTAMP,
            status TEXT DEFAULT 'assigned',
            FOREIGN KEY (challenge_id) REFERENCES challenges(challenge_id),
            UNIQUE(challenge_id, miner_hotkey)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS responses (
            response_id INTEGER PRIMARY KEY AUTOINCREMENT,
            challenge_id TEXT NOT NULL,
            miner_hotkey TEXT NOT NULL,
            node_id INTEGER,
            processing_time REAL,
            geojson_grid TEXT,
            satellite_scene_ids TEXT,
            model_version TEXT,
            confidence_overall REAL,
            signature TEXT,
            received_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            evaluated INTEGER DEFAULT 0,
            score REAL,
            evaluated_at TIMESTAMP,
            FOREIGN KEY (challenge_id) REFERENCES challenges(challenge_id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS response_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            response_id INTEGER NOT NULL,
            challenge_id TEXT NOT NULL,
            miner_hotkey TEXT NOT NULL,
            node_id INTEGER NOT NULL,
            spatial_score REAL,
            spectral_score REAL,
            temporal_score REAL,
            confidence_score REAL,
            latency_score REAL,
            final_score REAL NOT NULL,
            processing_time REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (response_id) REFERENCES responses(response_id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS availability_checks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id INTEGER NOT NULL,
            hotkey TEXT NOT NULL,
            checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_available INTEGER NOT NULL,
            response_time_ms REAL,
            error TEXT
        )
        """,
    ]


def check_initialized(db_path: str) -> bool:
    if not Path(db_path).exists():
        return False
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {r[0] for r in cursor.fetchall()}
        conn.close()
        required = {"challenges", "challenge_assignments", "responses", "response_scores"}
        return required.issubset(tables)
    except Exception:
        return False


def init_db(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    if not check_initialized(db_path):
        for stmt in get_schema():
            conn.execute(stmt)
        conn.commit()
    return conn
