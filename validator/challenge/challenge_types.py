"""
Synapse definitions for the Field Irrigation Intelligence Network.

FieldAnalysisChallenge  – validator → miner
FieldAnalysisResponse   – miner → validator
"""

import hashlib
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Challenge (validator → miner)
# ---------------------------------------------------------------------------

@dataclass
class FieldAnalysisChallenge:
    challenge_id: str
    latitude: float
    longitude: float
    area_rai: float
    query_date: str                   # ISO 8601
    grid_resolution_m: int = 20
    crop_hint: Optional[str] = None
    nonce: str = field(default_factory=lambda: secrets.token_hex(32))
    validator_hotkey: str = ""
    timestamp_utc: int = field(default_factory=lambda: int(time.time()))
    is_canary: bool = False           # internal flag, not sent to miner

    def to_dict(self) -> Dict[str, Any]:
        return {
            "challenge_id": self.challenge_id,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "area_rai": self.area_rai,
            "query_date": self.query_date,
            "grid_resolution_m": self.grid_resolution_m,
            "crop_hint": self.crop_hint,
            "nonce": self.nonce,
            "validator_hotkey": self.validator_hotkey,
            "timestamp_utc": self.timestamp_utc,
        }

    @staticmethod
    def make_id(latitude: float, longitude: float, query_date: str, nonce: str) -> str:
        raw = f"{latitude}:{longitude}:{query_date}:{nonce}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Response (miner → validator)
# ---------------------------------------------------------------------------

@dataclass
class FieldAnalysisResponse:
    challenge_id: str
    miner_hotkey: str
    geojson_grid: Dict[str, Any]
    processing_time_ms: int
    satellite_scene_ids: List[str]
    model_version: str
    confidence_overall: float
    signature: str = ""
    node_id: Optional[int] = None
    response_id: Optional[int] = None
    received_at: Optional[datetime] = None
    score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "challenge_id": self.challenge_id,
            "miner_hotkey": self.miner_hotkey,
            "geojson_grid": self.geojson_grid,
            "processing_time_ms": self.processing_time_ms,
            "satellite_scene_ids": self.satellite_scene_ids,
            "model_version": self.model_version,
            "confidence_overall": self.confidence_overall,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FieldAnalysisResponse":
        received_at = data.get("received_at")
        if received_at and isinstance(received_at, str):
            received_at = datetime.fromisoformat(received_at)
        return cls(
            challenge_id=data["challenge_id"],
            miner_hotkey=data.get("miner_hotkey", ""),
            geojson_grid=data.get("geojson_grid", {"type": "FeatureCollection", "features": []}),
            processing_time_ms=int(data.get("processing_time_ms", 0)),
            satellite_scene_ids=data.get("satellite_scene_ids", []),
            model_version=data.get("model_version", "unknown"),
            confidence_overall=float(data.get("confidence_overall", 0.0)),
            signature=data.get("signature", ""),
            node_id=data.get("node_id"),
            response_id=data.get("response_id"),
            received_at=received_at,
            score=data.get("score"),
        )


# ---------------------------------------------------------------------------
# Scoring result
# ---------------------------------------------------------------------------

@dataclass
class ScoringResult:
    spatial_score: float = 0.0
    spectral_score: float = 0.0
    temporal_score: float = 0.0
    confidence_score: float = 0.0
    latency_score: float = 0.0
    final_score: float = 0.0
    feedback: str = ""

    WEIGHTS = dict(spatial=0.35, spectral=0.20, temporal=0.20, confidence=0.15, latency=0.10)

    def compute_final(self) -> float:
        w = self.WEIGHTS
        self.final_score = (
            w["spatial"] * self.spatial_score
            + w["spectral"] * self.spectral_score
            + w["temporal"] * self.temporal_score
            + w["confidence"] * self.confidence_score
            + w["latency"] * self.latency_score
        )
        return self.final_score
