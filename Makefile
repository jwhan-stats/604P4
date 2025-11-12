.PHONY: predictions run-predictions build clean

# Docker image name
IMAGE_NAME = jwhan1223/bf-prediction
PLATFORM = linux/amd64

# Run predictions inside container (called from within Docker)
predictions:
	@echo "Generating 2025 Black Friday predictions..."
	python3 predict_2025_baseline.py
	@echo ""
	@echo "Displaying predictions..."
	python3 show_predictions.py

# Run predictions via Docker from host
run-predictions: build
	docker run --platform $(PLATFORM) $(IMAGE_NAME) make predictions

# Build Docker image
build:
	docker build --platform $(PLATFORM) -t $(IMAGE_NAME) .

# Clean up Docker image
clean:
	docker rmi $(IMAGE_NAME) || true

# Rebuild from scratch
rebuild: clean build
