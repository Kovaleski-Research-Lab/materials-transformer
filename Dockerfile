# Use the NVIDIA CUDA base image directly
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python and its package manager
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install uv system-wide
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy the configuration and the lock file
# The lock file now contains the information about the CUDA-enabled torch
COPY pyproject.toml uv.lock ./

# Install dependencies using the lock file.
# uv will read the index configuration from pyproject.toml automatically.
RUN uv pip sync uv.lock --system --no-cache --require-hashes --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128

# Copy your project source code
COPY src/ ./src/

ENTRYPOINT ["python3", "src/materials_transformer/main.py"]