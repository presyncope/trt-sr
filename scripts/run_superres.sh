#!/bin/bash

# Script to build and run the superres pipeline inside the trt-build docker image

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <input_video_path> <onnx_model_path> <output_video_path> [prescale]"
    echo "Example: $0 ./sample.mp4 ./models/realesrgan.onnx ./output.mp4 0.5"
    exit 1
fi

INPUT_HOST_PATH=$(realpath "$1")
ONNX_HOST_PATH=$(realpath "$2")
OUTPUT_HOST_PATH=$(realpath "$3")
PRESCALE=${4:-1.0}

INPUT_DIR=$(dirname "$INPUT_HOST_PATH")
INPUT_FILE=$(basename "$INPUT_HOST_PATH")

ONNX_DIR=$(dirname "$ONNX_HOST_PATH")
ONNX_FILE=$(basename "$ONNX_HOST_PATH")

OUTPUT_DIR=$(dirname "$OUTPUT_HOST_PATH")
OUTPUT_FILE=$(basename "$OUTPUT_HOST_PATH")

# Create output dir if not exists
mkdir -p "$OUTPUT_DIR"

# Root of the repo
REPO_ROOT=$(git rev-parse --show-toplevel)

# Docker run command
# 1. Mount repo to /workspace/src
# 2. Mount input/output/onnx
# 3. Build project
# 4. Run python script
docker run --gpus all --rm \
    -u $(id -u):$(id -g) \
    -v "$REPO_ROOT":/workspace/src \
    -v "$INPUT_DIR":/mnt/input \
    -v "$ONNX_DIR":/mnt/onnx \
    -v "$OUTPUT_DIR":/mnt/output \
    trt-build \
    /bin/bash -c "
    set -e
    
    echo 'Building C++ project...'
    cd /workspace/src/cpp
    cmake -B build -S .
    cmake --build build -j$(nproc)
    
    echo 'Running Superres Pipeline...'
    export PYTHONPATH=/workspace/src
    python3 /workspace/src/scripts/superres.py \
        --input '/mnt/input/$INPUT_FILE' \
        --output '/mnt/output/$OUTPUT_FILE' \
        --onnx-file '/mnt/onnx/$ONNX_FILE' \
        --plan-file '/mnt/output/$ONNX_FILE.plan' \
        --app-bin '/workspace/src/cpp/build/superres/superres_app' \
        --mode 3 \
        --prescale $PRESCALE
    "
