# Refactored Dockerfile based on user provided reference
# Using CUDA 12.3 and Ubuntu 22.04 for stability with typical ML workflows
FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

LABEL maintainer="Nuk"

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# 1. Install System Dependencies (merged from reference + required)
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    wget \
    git \
    pkg-config \
    sudo \
    ssh \
    libssl-dev \
    pbzip2 \
    pv \
    bzip2 \
    unzip \
    build-essential \
    ffmpeg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python 3
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    python3-wheel \
    python3-venv \
    && ln -s /usr/bin/python3 /usr/local/bin/python \
    && rm -rf /var/lib/apt/lists/*

# 3. Create Virtual Env (As referenced)
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# 4. Install CMake (System provided is usually 3.22 on 22.04, which is > 3.12 required)
RUN apt-get update && apt-get install -y cmake

# 5. Install TensorRT
# Using the method from the snippet (Tarball) can be fragile with URLs.
# We will use the layout from the snippet but with a robust installation method (pip + apt or pip + libs)
# OR we try to mimic the tarball extraction if the URL is valid.
# Given I cannot verify the URL '10.14.1.48' without fail (it requires me to guess availability), 
# I will use the standard NVIDIA Repo installation for TensorRT which is safer for a general setup,
# BUT I will structure it to match the snippet's intent (separate install step).

# Note: Configuring TRT repo for 22.04
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    tensorrt \
    && rm -rf /var/lib/apt/lists/*

# Install TensorRT Python bindings via pip (matching the system version if possible, or usually just `tensorrt`)
RUN pip install --upgrade pip wheel setuptools
RUN pip install tensorrt

# 6. Install Project Dependencies
RUN pip install \
    av \
    onnx \
    onnxconverter-common \
    numpy

WORKDIR /workspace

# 7. Copy local source
COPY . /workspace/trt-sr/

# Ensure clean build by removing any potential pre-existing binaries and build artifacts
# This prevents "GLIBC version not found" errors if the repo contained binaries built on a newer OS
RUN rm -rf /workspace/trt-sr/bin/* /workspace/trt-sr/cpp/superres/build

WORKDIR /workspace/trt-sr/cpp/superres

# Fix: Set explicit CUDA Architectures to avoid "native" error during docker build
RUN sed -i 's/set(CMAKE_CUDA_ARCHITECTURES "native")/set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89")/' CMakeLists.txt

RUN mkdir build && cd build && \
    cmake .. && \
    make -j$(nproc)

# 8. Entrypoint Script
RUN printf '#!/bin/bash\n\
    if [ "$#" -ne 3 ]; then\n\
    echo "Usage: $0 <input_video> <onnx_model> <output_video>"\n\
    exit 1\n\
    fi\n\
    \n\
    INPUT_VIDEO=$1\n\
    ONNX_MODEL=$2\n\
    OUTPUT_VIDEO=$3\n\
    \n\
    APP_BIN="/workspace/trt-sr/bin/real-esrgan-app"\n\
    PLAN_FILE="/workspace/model.plan"\n\
    \n\
    echo "Running TRT-SR Pipeline..."\n\
    python /workspace/trt-sr/scripts/run_pipeline.py \\\n\
    --input "$INPUT_VIDEO" \\\n\
    --output "$OUTPUT_VIDEO" \\\n\
    --mode 3 \\\n\
    --onnx-file "$ONNX_MODEL" \\\n\
    --plan-file "$PLAN_FILE" \\\n\
    --app-bin "$APP_BIN"\n\
    ' > /workspace/entrypoint.sh && chmod +x /workspace/entrypoint.sh

WORKDIR /workspace
ENTRYPOINT ["/workspace/entrypoint.sh"]
