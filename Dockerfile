# Base image
FROM tensorflow/tensorflow:2.13.0-gpu

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git python3-pip libglib2.0-0 libsm6 libxrender1 libxext6

# Copy files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Default command
CMD ["python", "main.py", "--mode", "train", "--config", "configs/train_config.yaml"]