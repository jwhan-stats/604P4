.PHONY: all prediction predictions run-prediction build clean clean-docker cleanall rawdata preprocess train help test

# Docker image name
IMAGE_NAME = jwhan1223/bf-prediction
PLATFORM = linux/amd64

# Date for prediction (default: tomorrow)
DATE ?= $(shell \
  TZ="America/New_York" date -v+1d +%Y-%m-%d 2>/dev/null || \
  TZ="America/New_York" date -d tomorrow +%Y-%m-%d 2>/dev/null)
  
# Directories
RAW_DATA_DIR = dataset/raw
PREPROCESSED_DIR = dataset/preprocessed
TEST_DATA_DIR = dataset/test
MODELS_DIR = models
RESULTS_DIR = results
CODES_DIR = codes
PJM_API_KEY = 9300a1cb4fa6425caefcdab8ddbd9644
OSF_ZIP_URL = https://files.osf.io/v1/resources/Py3u6/providers/osfstorage/?zip=


# Default target: run full analysis pipeline
all: preprocess train
	@echo "========================================="
	@echo "Full analysis pipeline completed"
	@echo "========================================="
	@echo ""
	@echo "Models trained and saved to: $(MODELS_DIR)/"
	@echo "Run predictions with: make predictions"
	@echo ""

# Preprocess data (prepare features)
preprocess:
	@echo "========================================="
	@echo "Running data preprocessing..."
	@echo "========================================="
	@cd $(CODES_DIR) && python3 update_features.py 2>/dev/null || true
	@echo "✓ Preprocessing complete"

# Train all models by running Jupyter notebooks
train:
	@echo "========================================="
	@echo "Training models..."
	@echo "========================================="
	@echo "Step 1/5: Feature engineering..."
	@cd $(CODES_DIR) && jupyter nbconvert --to notebook --execute --inplace feature_engineering.ipynb
	@echo ""
	@echo "Step 2/5: Creating November data for test features..."
	@cd $(CODES_DIR) && python3 create_november_data.py
	@echo ""
	@echo "Step 3/5: Training XGBoost Regressor (MW predictions)..."
	@cd $(CODES_DIR) && jupyter nbconvert --to notebook --execute --inplace model_xgboost_24h.ipynb
	@echo ""
	@echo "Step 4/5: Training XGBoost Classifier (peak hour)..."
	@cd $(CODES_DIR) && jupyter nbconvert --to notebook --execute --inplace model_xgboost_peak_hour.ipynb
	@echo ""
	@echo "Step 5/5: Training XGBoost Classifier (peak days)..."
	@cd $(CODES_DIR) && jupyter nbconvert --to notebook --execute --inplace model_xgboost_peak_days.ipynb
	@echo ""
	@echo "✓ All models trained and saved to $(MODELS_DIR)/"

# Update PJM data with API key
update-pjm:
	@cd $(CODES_DIR) && \
		PJM_API_KEY=$(PJM_API_KEY) \
		python3 fetch_latest_data.py >/dev/null 2>&1 || true

# Run prediction for specified date (default: tomorrow)
prediction: update-pjm
	@cd $(CODES_DIR) && python3 update_features.py >/dev/null 2>&1 || true
	@cd $(CODES_DIR) && python3 prepare_test_data.py $(DATE) >/dev/null 2>&1
	@cd $(CODES_DIR) && python3 make_predictions.py >/dev/null 2>&1
	@cd $(CODES_DIR) && python3 show_predictions.py

# Alias for prediction
predictions: prediction

# Run prediction via Docker from host
run-prediction: build
	docker run --platform $(PLATFORM) $(IMAGE_NAME) make prediction DATE=$(DATE)

# Build Docker image
build:
	@echo "Building Docker image..."
	@echo "ℹ️  This image includes Jupyter, all notebooks, and datasets for full training capability"
	@echo "⚠️  Image will be larger (~1-2GB) due to training dependencies"
	docker build --platform $(PLATFORM) -t $(IMAGE_NAME) .
	@echo "✓ Build complete"

# Clean intermediate files and results (keep code and raw data)
clean:
	@echo "Cleaning intermediate files and results..."
	@echo "Removing preprocessed data..."
	@rm -rf $(PREPROCESSED_DIR)/*
	@echo "Removing test data..."
	@rm -rf $(TEST_DATA_DIR)/*
	@echo "Removing models..."
	@rm -rf $(MODELS_DIR)/*
	@echo "Removing results..."
	@rm -rf $(RESULTS_DIR)/*
	@echo "Removing Python cache..."
	@find $(CODES_DIR) -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find $(CODES_DIR) -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✓ Clean complete (raw data and Docker image preserved)"

# Clean Docker image
clean-docker:
	@echo "Removing Docker image..."
	@docker rmi $(IMAGE_NAME) 2>/dev/null || echo "Docker image not found or already removed"
	@echo "✓ Docker image removed"

# Clean everything including Docker image
cleanall: clean clean-docker

# Delete raw data and re-download
rawdata:
	@echo "========================================="
	@echo "Deleting raw data..."
	@echo "========================================="
	@rm -rf $(RAW_DATA_DIR)/*.csv
	@mkdir -p $(RAW_DATA_DIR)
	@echo "✓ Raw data deleted"
	@echo ""
	@echo "Downloading OSF ZIP archive..."
	@TMP_ZIP=$$(mktemp /tmp/pjm_osf_XXXXXX.zip); \
	echo "  URL: $(OSF_ZIP_URL)"; \
	curl -s -L -o $$TMP_ZIP "$(OSF_ZIP_URL)"; \
	echo "Unpacking OSF ZIP into $(RAW_DATA_DIR)..."; \
	unzip -oq $$TMP_ZIP -d $(RAW_DATA_DIR); \
	rm -f $$TMP_ZIP
	@echo "OSF raw data download and extract complete."
	@echo "    Files placed in: $(RAW_DATA_DIR)/"
	@echo ""
	@echo "Running update-pjm (PJM API refresh)..."
	@$(MAKE) update-pjm
	@echo ""
	@echo "Raw data refresh complete."
	@echo "After this, run: make preprocess"

# Rebuild from scratch
rebuild: clean build

# Test prediction (for development)
test:
	@echo "Testing prediction with date: $(DATE)"
	@make prediction DATE=$(DATE)

# Help target
help:
	@echo "Available targets:"
	@echo "  make              - Run full analysis pipeline (preprocess + train all models)"
	@echo "  make predictions  - Run prediction for tomorrow (or DATE=YYYY-MM-DD)"
	@echo "  make clean        - Remove intermediate files, results, and models"
	@echo "                      (preserves code, raw data, and Docker image)"
	@echo "  make rawdata      - Delete raw data (manual re-download required)"
	@echo ""
	@echo "Additional targets:"
	@echo "  make preprocess   - Run update_features.py to create test features"
	@echo "  make train        - Train all models (feature engineering + November setup + 3 XGBoost models)"
	@echo "  make build        - Build Docker image"
	@echo "  make run-prediction DATE=... - Run prediction via Docker"
	@echo "  make clean-docker - Remove Docker image only"
	@echo "  make cleanall     - Remove everything including Docker image"
	@echo "  make test         - Test prediction with specified date"
	@echo "  make rebuild      - Clean and rebuild Docker image"
	@echo ""
	@echo "Examples:"
	@echo "  make predictions DATE=2025-11-29"
	@echo "  make run-prediction DATE=2025-11-29"
