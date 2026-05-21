# Field Irrigation Intelligence Network — Bittensor Subnet

A decentralised subnet that crowdsources geospatial soil-moisture analysis
of agricultural fields using satellite imagery (Sentinel-2 / Landsat) and
the TAO incentive model.

## Architecture

```
Validator ──challenges──► Miners (N)
   ▲                          │
   └──── scored responses ────┘
         (5-dimension score → on-chain weights)
```

### Scoring dimensions (design-doc weights)
| Dimension | Weight | Method |
|---|---|---|
| Spatial Accuracy | 35 % | Cross-miner consensus + canary field checks |
| Spectral Validity | 20 % | Physics-bound NDVI/NDWI range checks |
| Temporal Consistency | 20 % | Scene freshness + change-rate plausibility |
| Confidence Calibration | 15 % | ECE proxy — self-reported vs actual accuracy |
| Response Latency | 10 % | 2–8 s ideal; <2 s flagged; >90 s rejected |

### Moisture classes
| Class | Soil moisture | Action |
|---|---|---|
| `CRITICAL_DRY` (red) | < 20 % field capacity | Irrigate within 24 h |
| `DRY` (orange) | 20–40 % | Irrigate within 2–3 days |
| `OPTIMAL` (green) | 40–70 % | Monitor |
| `WET` (blue) | > 70 % | No irrigation needed |

## Quick start

```bash
# 1. Clone and copy env
cp env.example .env        # edit NETUID, wallet names, etc.

# 2. Install miner deps
pip install -e ".[miner]"

# 3. Run miner
uvicorn miner.main:app --host 0.0.0.0 --port 8001

# 4. Install validator deps (separate venv recommended)
pip install -e ".[validator]"

# 5. Run validator
python -m validator.main
```

### Docker

```bash
docker compose up --build
```

## Satellite data

Miners fetch Sentinel-2 L2A imagery (10 m resolution) via the
[Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/) STAC API
— no API key required for anonymous access. Landsat-9 is used as fallback.

When the API is unavailable (e.g. development mode) the miner automatically
falls back to deterministic synthetic data seeded from the field coordinates.

## Anti-cheat mechanisms

- **Nonce** — fresh 256-bit random per challenge prevents pre-computation
- **Timing analysis** — responses < 2 s are flagged (below satellite API round-trip)
- **Scene ID audit** — validator cross-checks returned scene IDs against STAC
- **Canary fields** — ~12 % of challenges use known locations (water bodies, forests)
- **Plagiarism detection** — cosine similarity > 0.99 across miners triggers penalty
- **Confidence calibration** — over-confident miners are penalised via ECE metric

## Project structure

```
irrigation-subnet/
├── miner/
│   ├── main.py                  FastAPI application
│   ├── endpoints/
│   │   ├── irrigation.py        POST /irrigation/challenge
│   │   └── availability.py      GET  /availability
│   └── utils/
│       ├── satellite.py         Planetary Computer STAC fetch
│       ├── ndvi.py              NDVI / NDWI / NDMI computation
│       └── geojson_builder.py   GeoJSON grid construction
└── validator/
    ├── main.py                  Main loop (processes + tasks)
    ├── config.py
    ├── challenge/
    │   ├── challenge_types.py   FieldAnalysisChallenge / Response
    │   ├── challenge_process.py Challenge sender loop
    │   └── send_challenge.py    Per-miner send via fiber
    ├── db/
    │   ├── schema.py
    │   └── operations.py
    └── evaluation/
        ├── canary_fields.py     Known calibration coordinates
        ├── calculate_score.py   5-dimension scoring engine
        ├── evaluation.py        IrrigationValidator
        ├── evaluation_loop.py   Async evaluation loop
        └── set_weights.py       On-chain weight setter
```
