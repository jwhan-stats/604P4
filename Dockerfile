# Use Python base image
FROM python:3.11-slim

# Install make
RUN apt-get update && apt-get install -y make && rm -rf /var/lib/apt/lists/*

# Install curl and unzip for Makefile rawdata target
RUN apt-get update && apt-get install -y curl unzip && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all dataset files (for both training and prediction)
COPY dataset ./dataset

# Copy trained models directory (if exists)
COPY models ./models

# Copy all code files (Python scripts and Jupyter notebooks)
COPY codes ./codes

# Copy Makefile for in-container commands
COPY Makefile .

# Make the scripts executable
RUN chmod +x codes/*.py

# Default to bash for interactive mode
# Users can override with commands like: docker run image make train
CMD ["/bin/bash"]
