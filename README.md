# AgriScore — Field Irrigation Intelligence Network

A **Bittensor subnet** that crowdsources geospatial soil-moisture analysis of agricultural fields using multi-spectral satellite imagery (Sentinel-2 / SAR).  
Miners compete to produce the most accurate irrigation predictions; validators score them using a deterministic, multi-pillar evaluation engine and submit weights to the Bittensor chain.

---

## Architecture

```
Validator ──challenge──► Miners (N)
    ▲                        │
    │   scored responses ◄───┘
    │
    └──► Bittensor weights (TAO rewards)
```

| Component | Role |
|---|---|
| **Validator** | Generates challenges, holds synthetic ground truth, runs the scoring engine, sets on-chain weights |
| **Miner** | Fetches Sentinel-2 imagery via Planetary Computer STAC, runs soil-moisture model, returns a GeoJSON grid |
| **Bittensor** | Distributes TAO rewards proportional to validator-assigned weights |

---

## Moisture Classes

| Class | Soil moisture | Farmer action |
|---|---|---|
| `CRITICAL_DRY` 🔴 | < 20 % field capacity | Irrigate within 24 h |
| `DRY` 🟠 | 20–40 % | Irrigate within 2–3 days |
| `OPTIMAL` 🟢 | 40–70 % | Monitor |
| `WET` 🔵 | > 70 % | No irrigation needed |

---

## Scoring System (v2)

The scoring engine is inspired by the [TurboVision subnet 44](https://github.com/score-technologies/turbovision) architecture — same **METRIC\_REGISTRY** pattern, same **baseline gating**, same **tiebreak logic** — adapted from sports video intelligence to agricultural satellite intelligence.

### Pipeline

```
Miner Response
      │
      ▼
[1] Processing-Rate Gate   (hard gate: 2 s – 90 s window)
      │
      ▼
[2] Synthetic GT Generation  (SHA-256 seed from nonce + lat/lon + date)
      │
      ▼
[3] Six Pillar Metrics  (weighted mean → acc)
      │
      ▼
[4] Baseline Gate  (acc ≤ θ=0.30 → score = 0)
      │
      ▼
[5] Plagiarism Penalty  (cosine similarity > 0.99 → score × 0.20)
      │
      ▼
[6] Bittensor Weight  +  Anti-Copy Tiebreak
```

### Scoring Pillars

| Pillar | Weight | Method | Score = 0 when |
|---|---|---|---|
| **Moisture Class F1** | 35 % | Hungarian cell-matching + AUC-F1 at two strictness levels (strict + lenient ordinal) | All class predictions wrong |
| **Moisture Index MAE** | 20 % | Mean absolute error of numeric `moisture_index`, normalised at 0.35 | MAE ≥ 0.35 |
| **Spectral Validity** | 15 % | Physics-bound checks on NDVI, NDWI, moisture\_index + cross-band consistency | All cells out-of-range |
| **False Alarm Rate** | 15 % | Spurious WET/OPTIMAL predictions on GT-DRY cells, per km² | ≥ 3 false alarms / km² |
| **Temporal Consistency** | 10 % | Scene freshness (±14 days), moisture change-rate plausibility, trend direction alignment | All checks fail |
| **Confidence Calibration** | 5 % | Expected Calibration Error (ECE) over 10 confidence bins | Confidence > 85 % but accuracy < 45 % |

#### Baseline gating

Any miner scoring at or below the baseline threshold earns **zero rewards** — preventing lazy submissions from collecting TAO.

```
acc ≤ θ   →  score = 0.0
acc > θ   →  score = (acc − θ) / (1 − θ)     # re-mapped to (0, 1]
```

`θ = 0.30` for `IRRIGATION_DETECTION` (a naive "always OPTIMAL" model scores ~0.28).

#### Processing-rate gate

Mirrors TurboVision's RTF (Real-Time Factor) gate — responses outside the acceptable latency window score zero:

| Processing time | Outcome |
|---|---|
| < 2 s | **Blocked** — below satellite-API round-trip (pre-computed) |
| 2 – 90 s | **Pass** |
| > 90 s | **Blocked** — timeout |

#### Anti-copy tiebreak

Two independent mechanisms stop model copying:

1. **Cosine similarity** — if two miners' `moisture_index` vectors have similarity > 0.99, both are capped at 20 % of their score.  
2. **Commit-block tiebreak** — when responses are near-identical (mean absolute difference < 0.03), the miner with the **earlier on-chain commit block** wins, making copying economically pointless.

---

## Synthetic Ground Truth

Instead of relying on cross-miner consensus (gameable by colluding miners), every challenge carries a **deterministic ground truth** the validator controls:

```python
seed  = SHA256(nonce + latitude + longitude + date)
ndvi  = seasonal_baseline(lat, day_of_year) + deterministic_noise(seed, cell_id)
mi    = moisture_baseline(lat, day_of_year) + deterministic_noise(seed, cell_id)
class = moisture_class_from_index(mi)
```

Two validators seeding with the same nonce produce **identical GT** — enabling multi-validator cross-checking without coordination.

---

## Anti-Cheat Mechanisms

| Mechanism | Guards against |
|---|---|
| **Nonce** (256-bit random per challenge) | Pre-computation of answers |
| **Processing-rate gate** (< 2 s blocked) | Sub-API-round-trip responses |
| **Scene ID audit** | Using stale or fabricated satellite scenes |
| **Canary fields** (~12 % of challenges) | Models that ignore actual satellite data |
| **Cosine similarity** (> 0.99 capped) | Copy-pasting another miner's answer |
| **Commit-block tiebreak** | Copying another miner's model weights |

---

## Project Structure

```
irrigation-subnet/
├── slides.html                      ⭐ Interactive scoring system presentation
│
├── miner/
│   ├── main.py                      FastAPI application
│   ├── endpoints/
│   │   ├── irrigation.py            POST /irrigation/challenge
│   │   └── availability.py          GET  /availability
│   └── utils/
│       ├── satellite.py             Planetary Computer STAC fetch
│       ├── ndvi.py                  NDVI / NDWI / NDMI computation
│       └── geojson_builder.py       GeoJSON grid construction
│
└── validator/
    ├── main.py
    ├── config.py
    ├── challenge/
    │   ├── challenge_types.py        FieldAnalysisChallenge / Response / ScoringResult
    │   ├── challenge_process.py      Challenge sender loop
    │   └── send_challenge.py         Per-miner send via fiber
    ├── db/
    │   ├── schema.py
    │   └── operations.py
    └── evaluation/
        ├── synthetic_gt.py           ⭐ Deterministic ground-truth generator
        ├── scoring_v2.py             ⭐ Main scoring pipeline (v2)
        ├── metrics/
        │   ├── registry.py           ⭐ METRIC_REGISTRY + ElementConfig
        │   ├── moisture.py           ⭐ Moisture Class F1 + MAE (Hungarian matching)
        │   ├── spectral.py           ⭐ Spectral validity + False alarm rate
        │   ├── temporal.py           ⭐ Temporal consistency
        │   └── calibration.py        ⭐ ECE confidence calibration
        ├── calculate_score.py        Legacy v1 scorer (plagiarism detection)
        ├── canary_fields.py          Known-answer calibration locations (Thai agriculture)
        ├── evaluation.py             IrrigationValidator
        ├── evaluation_loop.py        Async evaluation loop
        └── set_weights.py            On-chain weight setter
```

`⭐` = files added / significantly upgraded in v2

---

## Quick Start

```bash
# 1. Clone and configure
git clone https://github.com/UnStop-Labs/scoring.git
cd scoring
cp env.example .env          # edit NETUID, wallet names, etc.

# 2. Install miner dependencies
pip install -e ".[miner]"

# 3. Run miner
uvicorn miner.main:app --host 0.0.0.0 --port 8001

# 4. Install validator dependencies (separate venv recommended)
pip install -e ".[validator]"

# 5. Run validator
python -m validator.main
```

### Docker

```bash
docker compose up --build
```

---

## Writing a Miner

Your miner must expose `POST /irrigation/challenge` and return a GeoJSON `FeatureCollection`.  
Each feature's `properties` must include:

| Field | Type | Description |
|---|---|---|
| `cell_id` | int | Grid cell index (matches challenge grid) |
| `moisture_class` | str | One of `CRITICAL_DRY`, `DRY`, `OPTIMAL`, `WET` |
| `moisture_index` | float | Soil moisture in [0, 1] |
| `ndvi` | float | Normalised Difference Vegetation Index in [-1, 1] |
| `ndwi` | float | Normalised Difference Water Index in [-1, 1] |
| `cloud_masked` | bool | True if cell is obscured by cloud |
| `cell_confidence` | float | Per-cell confidence in [0, 1] *(optional but rewarded)* |

### Satellite data

Miners fetch **Sentinel-2 L2A imagery** (10 m resolution) via the  
[Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/) STAC API — no API key required.  
Landsat-9 is used as fallback. Development mode falls back to deterministic synthetic data seeded from field coordinates.

---

## Scoring API (v2)

```python
from validator.evaluation.scoring_v2 import score_response, score_responses_v2
from validator.evaluation.metrics import IRRIGATION_DETECTION_ELEMENT

# Score a single response
result = score_response(
    challenge = my_challenge,
    response  = miner_response,
    element   = IRRIGATION_DETECTION_ELEMENT,
)

print(result.summary())
# acc=0.82  score=0.74  rate_pass=True  plagiarism=False
#   moisture_class_f1      0.85  (w=0.35)
#   moisture_index_mae     0.78  (w=0.20)
#   spectral_validity      0.94  (w=0.15)
#   false_alarm_rate       0.91  (w=0.15)
#   temporal_consistency   0.60  (w=0.10)
#   confidence_calibration 0.72  (w=0.05)

# Score all responses for one challenge (batch + plagiarism detection)
results = score_responses_v2(
    challenge = my_challenge,
    responses = [resp_a, resp_b, resp_c],
)
```

### Adding a new metric

```python
# Add to any file in validator/evaluation/metrics/
from validator.evaluation.metrics.registry import register_metric, ElementType, PillarName

@register_metric((ElementType.IRRIGATION_DETECTION, PillarName.MY_NEW_PILLAR))
def compute_my_metric(gt_cells, pred_cells, **kwargs) -> float:
    # Return a float in [0, 1]
    ...
```

No changes to the scoring engine required — the registry picks it up automatically on import.

---

## Slides

An interactive 18-slide presentation of the full scoring system is at [`slides.html`](slides.html).  
Open in any browser and navigate with **← →** arrow keys or click.

---

## Canary Fields

~12 % of challenges use known reference locations the validator can score independently, without needing a miner reference:

| Location | Expected class | Purpose |
|---|---|---|
| Bhumibol Dam reservoir | `WET` | Permanent water body |
| Doi Inthanon forest | `OPTIMAL` | Dense tropical canopy |
| Mae Hong Son dryland | `DRY` | Arid northern zone |
| Bangkok city centre | `DRY` | Urban impervious surface |

---

## v1 → v2 Migration

| Concern | v1 | v2 |
|---|---|---|
| Ground truth | Cross-miner majority vote | Deterministic synthetic GT |
| Spatial accuracy | % agreement with peer miners | Hungarian-matched AUC-F1 vs GT |
| Architecture | Hardcoded 5-score function | METRIC\_REGISTRY — pluggable pillars |
| Minimum quality bar | None | Baseline gate θ = 0.30 |
| Latency | Soft bands (0.7 / 0.4 / 0.0) | Hard gate [2 s, 90 s] |
| Anti-copy | Cosine similarity cap | Cosine sim + commit-block tiebreak |
| Backwards compatible | — | ✓ v1 miners still work |

---

## License

MIT
