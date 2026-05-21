"""
Canary field definitions for the irrigation subnet.

Canary fields are locations with deterministic, time-stable satellite
signatures the validator can verify independently without ML inference.

They are rotated periodically (see CANARY_ROTATE_DAYS) so miners cannot
build a static blacklist.

REAL_FIELD_POOL is a curated list of Thai agricultural coordinates used
for genuine challenge generation.
"""

from typing import Any, Dict, List

# Validators compare miner answers against these expected moisture classes.
CANARY_FIELDS: List[Dict[str, Any]] = [
    # Permanent water bodies — always WET
    {
        "name": "Bhumibol Dam reservoir",
        "latitude": 17.2436,
        "longitude": 99.0438,
        "area_rai": 8.0,
        "expected_moisture_class": "WET",
        "expected_ndwi_min": 0.10,
        "crop_hint": None,
    },
    {
        "name": "Ubolratana Reservoir",
        "latitude": 16.7500,
        "longitude": 102.6167,
        "area_rai": 8.0,
        "expected_moisture_class": "WET",
        "expected_ndwi_min": 0.10,
        "crop_hint": None,
    },
    {
        "name": "Chiang Mai moat",
        "latitude": 18.7884,
        "longitude": 98.9853,
        "area_rai": 5.0,
        "expected_moisture_class": "WET",
        "expected_ndwi_min": 0.05,
        "crop_hint": None,
    },
    # Dense tropical forest — high NDVI, OPTIMAL-to-WET moisture
    {
        "name": "Doi Inthanon forest",
        "latitude": 18.5893,
        "longitude": 98.4863,
        "area_rai": 10.0,
        "expected_moisture_class": "OPTIMAL",
        "expected_ndvi_min": 0.50,
        "crop_hint": None,
    },
    {
        "name": "Khao Yai national park",
        "latitude": 14.4290,
        "longitude": 101.3710,
        "area_rai": 10.0,
        "expected_moisture_class": "OPTIMAL",
        "expected_ndvi_min": 0.45,
        "crop_hint": None,
    },
    # Arid / dryland zones in northern Thailand — CRITICAL_DRY expected during dry season
    {
        "name": "Mae Hong Son dryland",
        "latitude": 19.3020,
        "longitude": 97.9660,
        "area_rai": 8.0,
        "expected_moisture_class": "DRY",   # more conservative than CRITICAL_DRY
        "expected_ndvi_max": 0.35,
        "crop_hint": None,
    },
    # Urban impervious surface — very low NDVI, low moisture
    {
        "name": "Bangkok city centre",
        "latitude": 13.7563,
        "longitude": 100.5018,
        "area_rai": 5.0,
        "expected_moisture_class": "DRY",
        "expected_ndvi_max": 0.20,
        "crop_hint": None,
    },
]

# Expected moisture classes that a miner must return for a canary to pass
CANARY_EXPECTED_CLASSES: Dict[str, List[str]] = {
    "WET": ["WET"],
    "OPTIMAL": ["OPTIMAL", "WET"],
    "DRY": ["DRY", "CRITICAL_DRY"],
    "CRITICAL_DRY": ["CRITICAL_DRY"],
}


def canary_passes(expected_class: str, actual_majority_class: str) -> bool:
    """Return True if the miner's majority class is acceptable for this canary."""
    return actual_majority_class in CANARY_EXPECTED_CLASSES.get(expected_class, [expected_class])


# ---------------------------------------------------------------------------
# Real Thai agricultural field pool for genuine challenges
# ---------------------------------------------------------------------------

REAL_FIELD_POOL: List[Dict[str, Any]] = [
    # Chao Phraya basin — rice farming
    {"name": "Chainat rice", "latitude": 15.1839, "longitude": 100.1269, "area_rai": 12.0, "crop_hint": "rice"},
    {"name": "Suphan Buri rice", "latitude": 14.4744, "longitude": 100.1177, "area_rai": 15.0, "crop_hint": "rice"},
    {"name": "Sing Buri rice", "latitude": 14.8898, "longitude": 100.3967, "area_rai": 10.0, "crop_hint": "rice"},
    {"name": "Ang Thong rice", "latitude": 14.5896, "longitude": 100.4550, "area_rai": 8.0, "crop_hint": "rice"},
    {"name": "Ayutthaya rice", "latitude": 14.3692, "longitude": 100.5877, "area_rai": 11.0, "crop_hint": "rice"},
    # Northeast — sugarcane + cassava
    {"name": "Khon Kaen sugarcane", "latitude": 16.4322, "longitude": 102.8236, "area_rai": 20.0, "crop_hint": "sugarcane"},
    {"name": "Nakhon Ratchasima cassava", "latitude": 14.9799, "longitude": 102.0978, "area_rai": 18.0, "crop_hint": "cassava"},
    {"name": "Udon Thani sugarcane", "latitude": 17.4138, "longitude": 102.7870, "area_rai": 22.0, "crop_hint": "sugarcane"},
    {"name": "Roi Et rice", "latitude": 16.0538, "longitude": 103.6520, "area_rai": 14.0, "crop_hint": "rice"},
    # North — corn + soybean
    {"name": "Chiang Rai corn", "latitude": 19.9105, "longitude": 99.8406, "area_rai": 10.0, "crop_hint": "corn"},
    {"name": "Phrae soybean", "latitude": 18.1451, "longitude": 100.1407, "area_rai": 8.0, "crop_hint": "soybean"},
    {"name": "Lamphun corn", "latitude": 18.5745, "longitude": 98.9671, "area_rai": 9.0, "crop_hint": "corn"},
    # Central — mixed crops
    {"name": "Nakhon Sawan rice", "latitude": 15.7030, "longitude": 100.1372, "area_rai": 16.0, "crop_hint": "rice"},
    {"name": "Kamphaeng Phet cassava", "latitude": 16.4827, "longitude": 99.5226, "area_rai": 12.0, "crop_hint": "cassava"},
    {"name": "Phichit rice", "latitude": 16.4412, "longitude": 100.3487, "area_rai": 10.0, "crop_hint": "rice"},
    # South — rubber / oil palm
    {"name": "Surat Thani rubber", "latitude": 9.1382, "longitude": 99.3214, "area_rai": 15.0, "crop_hint": None},
    {"name": "Krabi oil palm", "latitude": 8.0863, "longitude": 98.9063, "area_rai": 20.0, "crop_hint": None},
]
