# Use Python base image for AMD64 platform
FROM --platform=linux/amd64 python:3.11-slim

# Install make
RUN apt-get update && apt-get install -y make && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the dataset directory
COPY dataset ./dataset

# Copy both prediction scripts
COPY predict_2025_baseline.py .
COPY show_predictions.py .

# Copy Makefile for in-container commands
COPY Makefile .

# Make the scripts executable
RUN chmod +x predict_2025_baseline.py show_predictions.py

# Default to bash for interactive mode
# Users can override with commands like: docker run image make predictions
CMD ["/bin/bash"]
