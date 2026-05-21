import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from fiber.logging_utils import get_logger

from validator.challenge.challenge_types import FieldAnalysisResponse
from validator.db.schema import check_initialized, init_db

logger = get_logger(__name__)


class DatabaseManager:
    def __init__(self, db_path):
        self.db_path = Path(db_path)
        if not check_initialized(str(self.db_path)):
            logger.info(f"Initialising database at {self.db_path}")
            init_db(str(self.db_path))
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

    def _get(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def close(self):
        if self.conn:
            self.conn.close()

    # ------------------------------------------------------------------
    # Challenges
    # ------------------------------------------------------------------

    def store_challenge(
        self,
        challenge_id: str,
        latitude: float,
        longitude: float,
        area_rai: float,
        query_date: str,
        is_canary: bool = False,
    ) -> None:
        conn = self._get()
        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO challenges
                    (challenge_id, latitude, longitude, area_rai, query_date, is_canary)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (challenge_id, latitude, longitude, area_rai, query_date, int(is_canary)),
            )
            conn.commit()
        finally:
            conn.close()

    def assign_challenge(self, challenge_id: str, miner_hotkey: str, node_id: int) -> None:
        conn = self._get()
        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO challenge_assignments
                    (challenge_id, miner_hotkey, node_id, status)
                VALUES (?, ?, ?, 'assigned')
                """,
                (challenge_id, miner_hotkey, node_id),
            )
            conn.commit()
        finally:
            conn.close()

    def mark_challenge_sent(self, challenge_id: str, miner_hotkey: str) -> None:
        conn = self._get()
        try:
            conn.execute(
                """
                UPDATE challenge_assignments
                SET status = 'sent', sent_at = CURRENT_TIMESTAMP
                WHERE challenge_id = ? AND miner_hotkey = ?
                """,
                (challenge_id, miner_hotkey),
            )
            conn.commit()
        finally:
            conn.close()

    def get_challenge_sent_at(self, challenge_id: str, miner_hotkey: str) -> Optional[str]:
        conn = self._get()
        try:
            row = conn.execute(
                "SELECT sent_at FROM challenge_assignments WHERE challenge_id=? AND miner_hotkey=?",
                (challenge_id, miner_hotkey),
            ).fetchone()
            return row["sent_at"] if row else None
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Responses
    # ------------------------------------------------------------------

    def store_response(
        self,
        challenge_id: str,
        miner_hotkey: str,
        response: FieldAnalysisResponse,
        node_id: int,
        processing_time: float,
        received_at: datetime,
        completed_at: datetime,
    ) -> int:
        conn = self._get()
        try:
            cursor = conn.execute(
                """
                INSERT INTO responses
                    (challenge_id, miner_hotkey, node_id, processing_time,
                     geojson_grid, satellite_scene_ids, model_version,
                     confidence_overall, signature, received_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    challenge_id,
                    miner_hotkey,
                    node_id,
                    processing_time,
                    json.dumps(response.geojson_grid),
                    json.dumps(response.satellite_scene_ids),
                    response.model_version,
                    response.confidence_overall,
                    response.signature,
                    received_at.isoformat() if received_at else None,
                    completed_at.isoformat() if completed_at else None,
                ),
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def get_pending_responses(self, challenge_id: str) -> List[FieldAnalysisResponse]:
        conn = self._get()
        try:
            rows = conn.execute(
                """
                SELECT r.*, c.latitude, c.longitude, c.query_date, c.is_canary
                FROM responses r
                JOIN challenges c ON r.challenge_id = c.challenge_id
                WHERE r.challenge_id = ? AND r.evaluated = 0
                """,
                (challenge_id,),
            ).fetchall()

            results = []
            for row in rows:
                grid = json.loads(row["geojson_grid"] or "{}")
                scene_ids = json.loads(row["satellite_scene_ids"] or "[]")
                far = FieldAnalysisResponse(
                    challenge_id=row["challenge_id"],
                    miner_hotkey=row["miner_hotkey"],
                    geojson_grid=grid,
                    processing_time_ms=int((row["processing_time"] or 0) * 1000),
                    satellite_scene_ids=scene_ids,
                    model_version=row["model_version"] or "unknown",
                    confidence_overall=float(row["confidence_overall"] or 0.0),
                    signature=row["signature"] or "",
                    node_id=row["node_id"],
                    response_id=row["response_id"],
                )
                results.append(far)
            return results
        finally:
            conn.close()

    def mark_response_evaluated(self, response_id: int, score: float) -> None:
        conn = self._get()
        try:
            conn.execute(
                """
                UPDATE responses
                SET evaluated = 1, score = ?, evaluated_at = CURRENT_TIMESTAMP
                WHERE response_id = ?
                """,
                (score, response_id),
            )
            conn.commit()
        finally:
            conn.close()

    def mark_responses_failed(self, challenge_id: str) -> None:
        conn = self._get()
        try:
            conn.execute(
                "UPDATE responses SET evaluated = 1, score = 0 WHERE challenge_id = ? AND evaluated = 0",
                (challenge_id,),
            )
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Scores
    # ------------------------------------------------------------------

    def store_response_score(
        self,
        response_id: int,
        challenge_id: str,
        miner_hotkey: str,
        node_id: int,
        spatial_score: float,
        spectral_score: float,
        temporal_score: float,
        confidence_score: float,
        latency_score: float,
        final_score: float,
        processing_time: float,
    ) -> None:
        conn = self._get()
        try:
            conn.execute(
                """
                INSERT INTO response_scores
                    (response_id, challenge_id, miner_hotkey, node_id,
                     spatial_score, spectral_score, temporal_score,
                     confidence_score, latency_score, final_score, processing_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    response_id, challenge_id, miner_hotkey, node_id,
                    spatial_score, spectral_score, temporal_score,
                    confidence_score, latency_score, final_score, processing_time,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_miner_scores(self, lookback_hours: int = 24) -> Dict[int, Dict[str, float]]:
        """Return {node_id: {final_score, count}} averaged over last N hours."""
        conn = self._get()
        try:
            rows = conn.execute(
                """
                SELECT node_id,
                       AVG(final_score) as avg_score,
                       COUNT(*) as cnt
                FROM response_scores
                WHERE created_at >= datetime('now', ? || ' hours')
                GROUP BY node_id
                """,
                (f"-{lookback_hours}",),
            ).fetchall()
            return {
                row["node_id"]: {
                    "final_score": float(row["avg_score"] or 0.0),
                    "count": int(row["cnt"]),
                }
                for row in rows
            }
        finally:
            conn.close()

    def get_pending_challenges_for_eval(self, delay_minutes: int = 5) -> List[Dict]:
        conn = self._get()
        try:
            rows = conn.execute(
                """
                SELECT DISTINCT c.challenge_id, c.latitude, c.longitude,
                       c.query_date, c.is_canary,
                       COUNT(r.response_id) as pending_count
                FROM responses r
                JOIN challenges c ON r.challenge_id = c.challenge_id
                WHERE r.evaluated = 0
                  AND datetime(r.received_at) <= datetime('now', ? || ' minutes')
                GROUP BY c.challenge_id
                LIMIT 1
                """,
                (f"-{delay_minutes}",),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def cleanup_old_data(self, days: int = 7) -> None:
        conn = self._get()
        try:
            conn.execute(
                "DELETE FROM response_scores WHERE created_at < datetime('now', ? || ' days')",
                (f"-{days}",),
            )
            conn.execute(
                "DELETE FROM responses WHERE received_at < datetime('now', ? || ' days')",
                (f"-{days}",),
            )
            conn.commit()
        finally:
            conn.close()
