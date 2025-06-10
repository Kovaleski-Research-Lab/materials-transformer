# Stage 1: The Poetry Builder
# Use a standard, lightweight Python image. Its only job is to install
# our Python dependencies using Poetry's lock file.
FROM python:3.10-slim as builder

# Install poetry
RUN pip install --no-cache-dir poetry

WORKDIR /app

COPY pyproject.toml poetry.lock ./

# Install packages, but without the development ones.
# The --no-root flag skips installing our own project code, as we'll copy that later.
# We're only interested in the `site-packages` directory this creates.
RUN poetry install --no-interaction --no-ansi --only main --no-root

# ---

# Stage 2: The Final Conda-based Production Image
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tip

ENV PATH /opt/conda/bin:$PATH

RUN conda create -n app-env python=3.10 cudatoolkit=12.1 -c nvidia -c conda-forge

# --- THE HYBRID STEP ---
COPY --from=builder /app/.venv/lib/python3.11/site-packages /opt/conda/envs/app-env/lib/python3.11/site-packages

WORKDIR /app
COPY src/ ./src/

ENV PATH /opt/conda/envs/app-env/bin:$PATH

ENTRYPOINT ["python", "src/materials_transformer/main.py"]