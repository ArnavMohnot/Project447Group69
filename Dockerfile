FROM python:3.12-slim

# Install system utilities
RUN apt-get update && apt-get install -y zip && rm -rf /var/lib/apt/lists/*

WORKDIR /job

# Copy requirements and install BEFORE copying the rest of the code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copying rest of source
COPY . .
