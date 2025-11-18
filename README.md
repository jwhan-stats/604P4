# 604P4
STATS604 Project 4 - Load Prediction

## Project Overview

This project predicts electricity load using XGBoost models. It combines three different prediction models:
1. **XGBoost Regressor** - Hourly MW predictions (24h ahead)
2. **XGBoost Classifier (peak_hour)** - Daily peak hour predictions
3. **XGBoost Classifier (peak_days)** - Peak day identification

---

## Quick Start

```bash
# Train all models
make

# Make predictions for tomorrow
make predictions

# Make predictions for specific date
make predictions DATE=2025-11-29

# Clean all intermediate files
make clean

# See all available commands
make help
```

---

## Project Structure

```
604P4/
├── Makefile                    # Build automation and workflow orchestration
├── Dockerfile                  # Docker image configuration
├── requirements.txt            # Python dependencies
│
├── codes/                      # Source code
│   ├── feature_engineering.ipynb         # [1] Data preprocessing and feature creation
│   ├── create_november_data.py          # [2] Extract November data for test features
│   ├── model_xgboost_24h.ipynb          # [3] Train XGBoost Regressor (MW predictions)
│   ├── model_xgboost_peak_hour.ipynb    # [4] Train XGBoost Classifier (peak hour)
│   ├── model_xgboost_peak_days.ipynb    # [5] Train XGBoost Classifier (peak days)
│   │
│   ├── fetch_latest_data.py             # Download latest update data from PJM API Portal
│   ├── update_features.py               # Create test features from raw data
│   ├── prepare_test_data.py             # Prepare test data for predictions
│   ├── make_predictions.py              # Load models and make predictions
│   ├── show_predictions.py              # Display predictions in required format
│   └── predict.py                       # Prediction workflow orchestrator
│
├── dataset/                    # Data storage
│   ├── raw/                    # Raw PJM data (hrl_load_metered_*.csv)
│   ├── preprocessed/           # Processed features (*.parquet)
│   └── test/                   # Test data for predictions
│
├── models/                     # Trained models (*.pkl)
└── results/                    # Prediction outputs (*.csv)
```

---

## File Relationships

### 1. Training Pipeline (`make` or `make train`)

```
feature_engineering.ipynb
    ↓ (creates preprocessed features)
dataset/preprocessed/full_year_features.parquet
    ↓ (extracts November data)
create_november_data.py
    ↓ (creates November CSV)
dataset/test/hrl_load_metered_november.csv
    ↓ (reads for training)
model_xgboost_24h.ipynb
    ↓ (saves models)
models/xgbreg_*.pkl

model_xgboost_peak_hour.ipynb
    ↓ (saves models)
models/xgb_*.pkl

model_xgboost_peak_days.ipynb
    ↓ (saves models)
models/xgb_peak_day_*.pkl
```

**Workflow:**
1. `feature_engineering.ipynb` - Reads raw data, creates features, saves to `dataset/preprocessed/`
2. `create_november_data.py` - Extracts November 2021-2024 data for test feature creation
3. `model_xgboost_24h.ipynb` - Trains XGBoost Regressor for MW predictions with business day aware masking
4. `model_xgboost_peak_hour.ipynb` - Trains XGBoost Classifier for peak hour predictions
5. `model_xgboost_peak_days.ipynb` - Trains XGBoost Classifier for peak day predictions
6. All models saved to `models/` directory

---

### 2. Prediction Pipeline (`make predictions`)

```
fetch_latest_data.py
    ↓ (updates)
update_features.py
    ↓ (updates)
dataset/raw/*.csv, dataset/test/test_*.parquet
    ↓ (reads)
prepare_test_data.py
    ↓ (creates test.csv)
dataset/test/test.csv
    ↓ (loads test data + models)
make_predictions.py ← models/*.pkl
    ↓ (saves predictions)
results/hybrid_predictions.csv
    ↓ (displays)
show_predictions.py
```

**Workflow:**
1. `fetch_latest_data.py` - Downloads latest update data from PJM API Portal (automatic)
2. `update_features.py` - Update test features
3. `prepare_test_data.py` - Prepares test data for target date with all lag features
4. `make_predictions.py` - Loads trained models and makes predictions
5. `show_predictions.py` - Displays predictions in required format

**Note:** Latest data is automatically fetched using PJM API call before each prediction

---

### 3. Key Dependencies

| File | Reads From | Writes To | Purpose |
|------|-----------|-----------|---------|
| `feature_engineering.ipynb` | `dataset/raw/` | `dataset/preprocessed/` | Create training features |
| `create_november_data.py` | `dataset/preprocessed/full_year_features.parquet` | `dataset/test/hrl_load_metered_november.csv` | Extract November data for test features |
| `model_xgboost_24h.ipynb` | `dataset/preprocessed/` | `models/xgbreg_*.pkl` | Train MW prediction model |
| `model_xgboost_peak_hour.ipynb` | `dataset/preprocessed/` | `models/xgb_*.pkl` | Train peak hour model |
| `model_xgboost_peak_days.ipynb` | `dataset/preprocessed/` | `models/xgb_peak_day_*.pkl` | Train peak day model |
| `update_features.py` | `dataset/test/hrl_load_metered_november.csv` | `dataset/test/test_features.parquet` | Create test features |
| `prepare_test_data.py` | `dataset/test/test_features.parquet` | `dataset/test/test.csv` | Prepare prediction data |
| `make_predictions.py` | `dataset/test/test.csv`<br>`models/*.pkl` | `results/hybrid_predictions.csv` | Generate predictions |
| `show_predictions.py` | `results/hybrid_predictions.csv` | stdout | Display results |

---

## Makefile Targets

| Target | Description | Workflow |
|--------|-------------|----------|
| `make` | Full analysis pipeline | `preprocess` → `train` (5 steps: feature engineering → November data → 3 model notebooks) |
| `make preprocess` | Preprocess data | Runs `update_features.py` to create test features |
| `make train` | Train all models | Runs 5 steps: feature engineering → November data → 3 model notebooks |
| `make update-pjm` | Extract PJM data | Runs `fetch_latest_data.py` to fetch latest data|
| `make predictions` | Make predictions for tomorrow | `prepare_test_data.py` → `make_predictions.py` → `show_predictions.py` |
| `make predictions DATE=...` | Make predictions for specific date | Same as above with custom date |
| `make clean` | Remove intermediate files | Deletes `dataset/preprocessed/`, `dataset/test/`, `models/`, `results/` (preserves Docker image) |
| `make clean-docker` | Remove Docker image only | Removes Docker image |
| `make cleanall` | Remove everything | `clean` + `clean-docker` |
| `make rawdata` | Re-download raw data | Deletes and re-downloads all files from zip folder and PJM API portal |
| `make build` | Build Docker image | Creates Docker image with trained models |
| `make run-prediction` | Run prediction via Docker | Runs prediction inside Docker container |
| `make help` | Show available commands | Displays all available targets |

---

## Features

### XGBoost Regressor (MW Predictions)
- **Annual lags**: 1yr, 2yr, 3yr, 4yr (8760h, 17520h, 26280h, 35040h)
- **Weekly lags**: 2d, 3d, 4d, 5d, 6d, 7d, 8d, 9d (48h, 72h, 96h, 120h, 144h, 168h, 192h, 216h)
- **Indicators**: is_blackfriday, is_thanksgiving
- **Business day aware masking**: Simulates Mon-Thu data updates for realistic training

### XGBoost Classifier (Peak Hour)
- Temporal features: day_of_week, month, day_of_month
- Black Friday features: is_blackfriday, is_thanksgiving, days_from_bf
- Daily peak hour lags: 2d-9d
- BF-based peak hour lags: 1yr-4yr

### XGBoost Classifier (Peak Days)
- Predicted peak loads: BF-8 to BF+1 (4-year average)
- BF day index and temporal features
- Actual peak load lags: 1d-9d
- BF-based peak load lags: 1yr-2yr

---

## Docker Usage

```bash
# Build Docker image (includes all code, data, and dependencies)
make build

# Run training in Docker (full pipeline)
docker run --platform linux/amd64 jwhan1223/bf-prediction make

# Run prediction in Docker
make run-prediction DATE=2025-11-29

# Or use Docker directly for predictions
docker run --platform linux/amd64 jwhan1223/bf-prediction make predictions

# Run specific commands
docker run --platform linux/amd64 jwhan1223/bf-prediction make train
docker run --platform linux/amd64 jwhan1223/bf-prediction make preprocess
```

**Note:** Docker image includes Jupyter, all notebooks, and dataset files for full training capability.

---

## Requirements

- Python 3.8+
- Jupyter notebook
- See `requirements.txt` for Python dependencies

```bash
pip install -r requirements.txt
```

---

## Data Source

Raw data from PJM Data Miner: https://dataminer2.pjm.com/list

**Automatic download (recommended):**
```bash
make rawdata
```
This will automatically download all raw data files from the PJM API Portal.

**Manual download (alternative):**
- Download hourly load metered data: `hrl_load_metered_YYYY.csv`
- Place in `dataset/raw/` directory
