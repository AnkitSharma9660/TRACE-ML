# TRACE - AI-Powered Network Intrusion Detection System

A production-ready **Intrusion Detection System (IDS)** that combines a **Variational Autoencoder (VAE)** for anomaly detection with a **Mixture of Experts (MoE)** ensemble for attack classification, wrapped in a **FastAPI** backend with **LLM-powered alert summarization** via Google Gemini.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
  - [High-Level Design](#high-level-design)
  - [Component Diagram](#component-diagram)
- [Core Components](#core-components)
  - [1. VAE Anomaly Detector (`VAEScorer`)](#1-vae-anomaly-detector-vaescorer)
  - [2. Mixture of Experts Classifier (`MoEPredictor`)](#2-mixture-of-experts-classifier-moepredictor)
  - [3. Threat Pipeline (`ThreatPipeline`)](#3-threat-pipeline-threatpipeline)
  - [4. LLM Alert Summarizer](#4-llm-alert-summarizer)
- [API Documentation](#api-documentation)
  - [Request/Response Schemas](#requestresponse-schemas)
  - [Endpoints](#endpoints)
- [Project Structure](#project-structure)
- [Model Artifacts](#model-artifacts)
- [Feature Sets](#feature-sets)
- [Setup & Deployment](#setup--deployment)
  - [Local Development](#local-development)
  - [Docker Deployment](#docker-deployment)
- [Configuration](#configuration)
- [Frontend Integration Guide](#frontend-integration-guide)
- [Tech Stack](#tech-stack)
- [Data Sources](#data-sources)

---

## Overview

TRACE is a two-stage ML pipeline designed for real-time network threat detection:

1. **Anomaly Detection** - A Variational Autoencoder (VAE) scores each network flow. Flows exceeding a learned reconstruction-error threshold are flagged as anomalous.
2. **Attack Classification** - Flagged flows are passed to a Mixture of Experts (MoE) ensemble of 8 classifiers that predict the specific attack type and confidence.
3. **LLM Summarization** - An optional Gemini-powered LLM generates human-readable security alerts with severity ratings, evidence, and recommended actions.

---

## System Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FastAPI Server (main.py)                     │
│                                                                      │
│  POST /analyze-batch   POST /predict   POST /attack-type             │
│  POST /anomaly-flag    POST /analyze   POST /summarize               │
│  GET  /health          GET  /random-analyze                          │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Threat Pipeline (compute.py)                     │
│                                                                      │
│  ┌──────────────────────┐         ┌──────────────────────────────┐  │
│  │   VAEScorer          │         │      MoEPredictor            │  │
│  │  (Anomaly Detection) │────────▶│   (Attack Classification)    │  │
│  │                      │  flags  │                              │  │
│  │  - 77 features       │         │  - XGBoost                   │  │
│  │  - Reconstruction    │         │  - LightGBM                  │  │
│  │    error scoring     │         │  - Random Forest             │  │
│  │  - Top contributors  │         │  - CatBoost                  │  │
│  │  - Configurable      │         │  - Logistic Regression       │  │
│  │    threshold         │         │  - SVM                       │  │
│  └──────────────────────┘         │  - Small MLP (PyTorch)       │  │
│                                   │  - Deep MLP (PyTorch)        │  │
│                                   │  - Gating Network            │  │
│                                   └──────────────────────────────┘  │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  LLM Summarizer (llm.py)                             │
│                                                                      │
│  Google Gemini 1.5 Pro → Structured JSON alert with:                 │
│  severity, summary, evidence, attack assessment,                     │
│  recommended actions, investigative questions                        │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Diagram

```
Raw Network Flow Features (dict)
         │
         ▼
┌─────────────────┐
│ Feature Mapping │──▶ 77-dim VAE feature vector
│ & Validation    │──▶ 46-dim MoE feature vector
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────┐
│  VAEScorer      │     │  Feature Scaler  │
│  (VAE Model)    │◀───▶│  (StandardScaler)│
│  input: 77 dims │     │  (benign-only)   │
│  output: score, │     └──────────────────┘
│  flag, top-k    │
└────────┬────────┘
         │ is_anomaly == True?
         ├────── Yes ──────▶ ┌──────────────────┐
         │                   │  MoEPredictor    │
         │                   │  8 Experts +     │
         │                   │  Gating Network  │
         │                   │  output: label,  │
         │                   │  confidence      │
         │                   └────────┬─────────┘
         │                            │
         ▼                            ▼
┌─────────────────────────────────────────────────┐
│              Combined Result JSON                │
│  {is_anomaly, score, threshold,                  │
│   attack_type, attack_confidence,                │
│   top_contributors}                              │
└─────────────────────────────────────────────────┘
```

---

## Core Components

### 1. VAE Anomaly Detector (`VAEScorer`)

**File:** `compute.py` | **Model:** `app/model/model.py`

A **Variational Autoencoder** trained exclusively on benign network traffic. Anomalous flows produce high reconstruction errors.

| Parameter | Value |
|-----------|-------|
| Input dimension | 77 features |
| Hidden dimension | 256 |
| Latent dimension | 16 |
| Top-K contributors | 5 |
| Threshold source | `artifacts/threshold.json` (pre-computed) |

**Architecture:**
```
Encoder: Linear(77→256) → LeakyReLU → Linear(256→256) → LeakyReLU
         ├──→ Linear(256→16) [mu]
         └──→ Linear(256→16) [logvar] (clamped to [-10, 10])
         
Reparameterization: z = μ + σ × ε, ε ~ N(0,1)

Decoder: Linear(16→256) → LeakyReLU → Linear(256→256) → LeakyReLU → Linear(256→77)
```

**Loss Function:**
```python
Total Loss = MSE_Reconstruction + KL_Divergence
KL = -0.5 × Σ(1 + logvar - μ² - exp(logvar))
```

**Key Methods:**
- `score_raw(X_raw_np)` — Accepts raw numpy arrays, applies scaler, returns scores
- `score_scaled(X_scaled_np)` — Accepts pre-scaled data, returns scores + flags + top-K contributors

---

### 2. Mixture of Experts Classifier (`MoEPredictor`)

**File:** `compute.py`

An ensemble of **8 expert classifiers** whose predictions are dynamically weighted by a learned gating network.

| Expert | Type | Artifact |
|--------|------|----------|
| XGBoost | Gradient Boosting | `xgb.json` |
| LightGBM | Gradient Boosting | `lgb.pkl` |
| Random Forest | Ensemble Trees | `rf.pkl` |
| CatBoost | Gradient Boosting | `cat.cbm` |
| Logistic Regression | Linear | `lr.pkl` |
| SVM | Kernel Method | `svm.pkl` |
| Small MLP | PyTorch NN (46→64→32→3) | `small_mlp.pt` |
| Deep MLP | PyTorch NN (46→128→64→32→3) | `deep_mlp.pt` |
| **Gating Network** | PyTorch NN (46→64→32→8) | `gating.pt` |

**Input:** 46 features (subset of the 77 VAE features)
**Output:** 3-class softmax probabilities → attack label + confidence

**Gating Mechanism:**
```python
final_probs = Σ(gate_i × expert_i_proba) for i in 8 experts
```

The gating network learns to trust different experts for different input patterns, producing a weighted ensemble that adapts per-sample.

---

### 3. Threat Pipeline (`ThreatPipeline`)

**File:** `compute.py`

Orchestrates the two-stage detection flow:

```python
def predict_batch(rows):
    # Stage 1: VAE anomaly scoring (all rows)
    vae_out = vae.score_raw(X_vae_raw)
    
    # Stage 2: MoE classification (flagged rows only)
    if flags.any():
        labels, probs = moe.predict_batch(flagged_rows)
    
    # Combine results
    return results  # list of dicts
```

**Validation:** The `validate_rows()` function checks for missing features in input data and reports them in the `input_issues` field.

---

### 4. LLM Alert Summarizer

**File:** `llm.py`

Uses **Google Gemini 1.5 Pro** via LangChain to generate structured, human-readable security alerts.

**System Prompt Role:** Cybersecurity SOC Analyst Assistant

**Output Schema:**
```json
{
  "severity": "low|medium|high|critical",
  "summary": "string",
  "what_is_wrong": ["string"],
  "evidence": [{"feature": "string", "signal": "string", "note": "string"}],
  "attack_assessment": {
    "predicted_type": "string",
    "confidence": 0.0,
    "interpretation": "string"
  },
  "recommended_actions": {
    "immediate": ["string"],
    "short_term": ["string"],
    "long_term": ["string"]
  },
  "questions_to_ask": ["string"],
  "limitations": ["string"]
}
```

**Temperature:** 0.2 (deterministic, analytical responses)

---

## API Documentation

**Base URL:** `http://localhost:8000`
**CORS Origins:** `http://localhost:3000`, `http://127.0.0.1:3000`

### Request/Response Schemas

#### `RowIn` - Single Row Request
```json
{
  "row": {
    "Dst Port": 443,
    "Flow Duration": 12345,
    "Tot Fwd Pkts": 10,
    "...": "..."
  }
}
```

#### `BatchIn` - Batch Request
```json
{
  "rows": [
    {"Dst Port": 443, "Flow Duration": 12345, "...": "..."},
    {"Dst Port": 22, "Flow Duration": 54321, "...": "..."}
  ]
}
```

### Endpoints

#### `GET /health`
Health check endpoint.

**Response:**
```json
{"status": "ok"}
```

---

#### `GET /random-analyze`
Loads a random row from the CSV dataset, runs the full pipeline, and returns model + LLM payload. Useful for testing/demo purposes.

**Response:**
```json
{
  "model_output": { ... full per-row result ... },
  "llm_payload": { ... curated subset for LLM ... }
}
```

---

#### `POST /predict`
Full analysis for a single row. Returns all important fields in a flattened structure.

**Request:** `RowIn`

**Response:**
```json
{
  "is_anomaly": true,
  "score": 45.67,
  "threshold": 30.0,
  "attack_type": "DDoS",
  "attack_confidence": 0.92,
  "top_contributors": [
    {"feature": "Flow Duration", "sq_error": 12.3},
    {"feature": "Tot Fwd Pkts", "sq_error": 8.1},
    "..."
  ],
  "input_issues": {"vae_missing": {}, "moe_missing": {}}
}
```

**Use case:** Primary "Predict" button - when you need everything.

---

#### `POST /anomaly-flag`
Lightweight endpoint returning only anomaly detection results.

**Request:** `RowIn`

**Response:**
```json
{
  "is_anomaly": true,
  "score": 45.67,
  "threshold": 30.0
}
```

**Use case:** Quick anomaly check without attack classification overhead.

---

#### `POST /attack-type`
Returns only the attack classification for a single row.

**Request:** `RowIn`

**Response:**
```json
{
  "attack_type": "DDoS",
  "attack_confidence": 0.92
}
```

**Use case:** Display attack type for already-detected anomalies.

---

#### `POST /analyze`
Full model output + curated LLM payload in a single response.

**Request:** `RowIn`

**Response:**
```json
{
  "model_output": {
    "is_anomaly": true,
    "score": 45.67,
    "threshold": 30.0,
    "attack_type": "DDoS",
    "attack_confidence": 0.92,
    "top_contributors": [...],
    "input_issues": {}
  },
  "llm_payload": {
    "is_anomaly": true,
    "score": 45.67,
    "threshold": 30.0,
    "attack_type": "DDoS",
    "attack_confidence": 0.92,
    "top_contributors": [...],
    "input_issues": {}
  }
}
```

**Use case:** Send `llm_payload` to `/summarize` for human-readable alert generation.

---

#### `POST /analyze-batch`
Batch analysis of multiple rows. Returns full results for each row.

**Request:** `BatchIn`

**Response:**
```json
{
  "results": [
    {
      "is_anomaly": true,
      "score": 45.67,
      "threshold": 30.0,
      "attack_type": "DDoS",
      "attack_confidence": 0.92,
      "top_contributors": [...]
    },
    "..."
  ],
  "input_issues": {"vae_missing": {}, "moe_missing": {}}
}
```

**Use case:** Bulk table scanning, precomputing UI badges.

---

#### `POST /summarize`
Generates a human-readable security alert summary using Google Gemini LLM.

**Request:**
```json
{
  "llm_payload": { ... from /analyze ... },
  "model_output": { ... from /analyze ... }
}
```

**Response:** See [Output Schema](#4-llm-alert-summarizer) above.

**Use case:** Generate explanatory text for dashboard alerts.

---

## Project Structure

```
TRACE/
├── main.py                  # FastAPI application, all API endpoints
├── compute.py               # Core ML pipeline (VAE + MoE + ThreatPipeline)
├── llm.py                   # LLM summarizer (Gemini via LangChain)
├── requirements.txt         # Python dependencies
├── Dockerfile               # Container deployment config
├── .example.env             # Environment variable template
│
├── app/
│   └── model/
│       └── model.py         # VAE model definition + loss function
│
├── artifacts/               # Trained model artifacts (loaded at startup)
│   ├── model.pth            # VAE checkpoint
│   ├── threshold.json       # Anomaly detection threshold
│   ├── scaler_train_benign.pkl  # StandardScaler (benign-only training)
│   ├── features_name.json   # 77 VAE feature names
│   └── moeModels/           # Mixture of Experts artifacts
│       ├── scaler.pkl       # MoE feature scaler
│       ├── label_encoder.pkl  # Label encoder for attack classes
│       ├── xgb.json         # XGBoost expert
│       ├── lgb.pkl          # LightGBM expert
│       ├── rf.pkl           # Random Forest expert
│       ├── cat.cbm          # CatBoost expert
│       ├── lr.pkl           # Logistic Regression expert
│       ├── svm.pkl           # SVM expert
│       ├── small_mlp.pt     # Small MLP expert (PyTorch)
│       ├── deep_mlp.pt      # Deep MLP expert (PyTorch)
│       └── gating.pt        # Gating network (PyTorch)
│
└── solarmainframe/          # (external) CSV dataset directory
    └── ids-intrusion-csv/
        └── versions/1/*.csv # CICIDS-format CSV files
```

---

## Model Artifacts

All artifacts in `artifacts/` are **pre-trained** and loaded at server startup. No training occurs in production.

| Artifact | Purpose | Format |
|----------|---------|--------|
| `model.pth` | VAE weights | PyTorch state_dict |
| `threshold.json` | Anomaly threshold | `{"threshold": float}` |
| `scaler_train_benign.pkl` | VAE feature scaler | scikit-learn StandardScaler |
| `features_name.json` | 77 VAE feature names | `{"feature_names": [...]}` |
| `moeModels/scaler.pkl` | MoE feature scaler | scikit-learn StandardScaler |
| `moeModels/label_encoder.pkl` | Attack class encoder | scikit-learn LabelEncoder |
| `moeModels/xgb.json` | XGBoost model | XGBoost JSON format |
| `moeModels/lgb.pkl` | LightGBM model | joblib pickle |
| `moeModels/rf.pkl` | Random Forest model | joblib pickle |
| `moeModels/cat.cbm` | CatBoost model | CatBoost binary |
| `moeModels/lr.pkl` | Logistic Regression | joblib pickle |
| `moeModels/svm.pkl` | SVM model | joblib pickle |
| `moeModels/small_mlp.pt` | Small MLP | PyTorch state_dict |
| `moeModels/deep_mlp.pt` | Deep MLP | PyTorch state_dict |
| `moeModels/gating.pt` | Gating network | PyTorch state_dict |

---

## Feature Sets

### VAE Features (77)
Full CICIDS2017-style network flow features including:
- Flow-level: Duration, packet counts, byte rates, IAT statistics
- Fwd/Bwd direction: Packet lengths, flags, bulk rates, window sizes
- Protocol flags: FIN, RST, PSH, ACK, URG, CWE counts
- Activity-based: Active/Idle time statistics

### MoE Features (46)
A curated subset of the VAE features optimized for attack classification:

```
Dst Port, Flow Duration, Tot Fwd Pkts, Tot Bwd Pkts,
TotLen Fwd Pkts, Fwd Pkt Len Max, Fwd Pkt Len Min, Fwd Pkt Len Mean,
Bwd Pkt Len Max, Bwd Pkt Len Min, Bwd Pkt Len Mean,
Flow Byts/s, Flow Pkts/s, Flow IAT Mean, Flow IAT Std,
Flow IAT Max, Bwd IAT Tot, Bwd IAT Mean, Bwd IAT Std, Bwd IAT Min,
Fwd PSH Flags, Bwd PSH Flags, Fwd URG Flags, Bwd URG Flags,
Pkt Len Var, FIN Flag Cnt, RST Flag Cnt, PSH Flag Cnt,
ACK Flag Cnt, URG Flag Cnt, CWE Flag Count, Down/Up Ratio,
Fwd Byts/b Avg, Fwd Pkts/b Avg, Fwd Blk Rate Avg,
Bwd Byts/b Avg, Bwd Pkts/b Avg, Bwd Blk Rate Avg,
Init Fwd Win Byts, Init Bwd Win Byts, Fwd Act Data Pkts,
Fwd Seg Size Min, Active Mean, Active Std, Active Max, Idle Min
```

---

## Setup & Deployment

### Local Development

**Prerequisites:**
- Python 3.10+
- Google Gemini API key

```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment variables
cp .example.env .env
# Edit .env and add your GOOGLE_API_KEY

# 4. Ensure model artifacts are in artifacts/ directory

# 5. Run the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Server will be available at `http://localhost:8000`.
Interactive API docs at `http://localhost:8000/docs` (Swagger UI).

### Docker Deployment

```bash
# 1. Build image
docker build -t trace-ids .

# 2. Run container (mount artifacts if external)
docker run -d \
  --name trace-server \
  -p 8000:8000 \
  -e GOOGLE_API_KEY=your_key_here \
  -v $(pwd)/artifacts:/app/artifacts \
  trace-ids

# 3. Verify
curl http://localhost:8000/health
```

**Production considerations:**
- Use a reverse proxy (nginx, traefik) for TLS termination
- Set `--workers` flag for uvicorn (e.g., `uvicorn main:app --workers 4`)
- Consider GPU support for PyTorch inference (`--gpus all`)
- Mount artifacts as a volume for persistence across deployments

---

## Configuration

### Environment Variables (`.env`)

| Variable | Description | Required |
|----------|-------------|----------|
| `GOOGLE_API_KEY` | Google Gemini API key for LLM summarization | Yes (for `/summarize`) |

### Hardcoded Configuration

| Setting | Value | Location |
|---------|-------|----------|
| `CSV_FOLDER` | Path to CICIDS CSV files | `main.py` (line 10) |
| `CORS Origins` | `localhost:3000`, `127.0.0.1:3000` | `main.py` |
| `VAE input_dim` | 77 | `compute.py` |
| `VAE hidden_dim` | 256 | `compute.py` |
| `VAE latent_dim` | 16 | `compute.py` |
| `VAE topk` | 5 | `compute.py` |
| `LLM model` | `gemini-1.5-pro` | `llm.py` |
| `LLM temperature` | 0.2 | `llm.py` |

---

## Frontend Integration Guide

### Typical Workflow

```
1. User views table of network flows
   └── Call POST /analyze-batch to score all rows

2. User clicks on a specific row
   └── Call POST /analyze to get full results + llm_payload

3. Display model results in UI
   ├── Anomaly badge (is_anomaly)
   ├── Score bar (score vs threshold)
   ├── Attack type badge
   └── Top contributors tooltip

4. Generate human-readable explanation
   └── POST /summarize with llm_payload + model_output
       └── Display summary in alert panel
```

### Quick Anomaly Check
```javascript
// Lightweight check before full analysis
const flag = await fetch('/anomaly-flag', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({row: flowData})
});
const {is_anomaly, score, threshold} = await flag.json();
```

### Full Analysis + LLM Summary
```javascript
// Get full results
const analyze = await fetch('/analyze', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({row: flowData})
});
const {model_output, llm_payload} = await analyze.json();

// Generate summary
const summary = await fetch('/summarize', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    llm_payload,
    model_output
  })
});
const alert = await summary.json();
// alert.severity, alert.summary, alert.what_is_wrong, ...
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **API Framework** | FastAPI + Uvicorn |
| **Anomaly Detection** | PyTorch (Variational Autoencoder) |
| **Attack Classification** | XGBoost, LightGBM, CatBoost, Random Forest, Logistic Regression, SVM, PyTorch MLPs |
| **Ensemble** | Custom Mixture of Experts with Gating Network |
| **LLM Integration** | LangChain + Google Gemini 1.5 Pro |
| **Data Processing** | NumPy, Pandas, scikit-learn |
| **Serialization** | joblib, PyTorch state_dict, JSON |
| **Deployment** | Docker (python:3.10-slim) |
| **Validation** | Pydantic |

---

## Data Sources

The system is designed to work with **CICIDS2017**-style network flow data. The `/random-analyze` endpoint reads from CSV files in the configured `CSV_FOLDER` path (currently pointing to `solarmainframe/ids-intrusion-csv/versions/1`).

Expected CSV format: CICIDS2017 feature columns with optional `Label` column (dropped during random sampling).

---

## License

Proprietary - All rights reserved.
