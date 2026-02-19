#!/bin/bash

# 스크립트의 디렉토리 위치 파악
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

LOCAL_BIN="$PROJECT_ROOT/bin_rt"
IMAGE_NAME="trt-devel:latest"

# 호스트 파일 설정
HOST_INPUT="/mnt/d/raw_videos/motorcycle.yuv"
PLAN_NAME="x4v3.plan"
OUTPUT_NAME="output.yuv"

# 디렉토리 확인
if [ ! -d "$LOCAL_BIN" ]; then
    echo "Error: 'bin' directory not found at $LOCAL_BIN"
    exit 1
fi

if [ ! -f "$HOST_INPUT" ]; then
    echo "Error: Input file not found at $HOST_INPUT"
    exit 1
fi

# Docker 내부 경로 설정
DOCKER_BIN_DIR="/app/bin"
DOCKER_INPUT_DIR="/input" # 단순히 /input 매핑

INPUT_FILENAME=$(basename "$HOST_INPUT")

# Docker 내부 경로
DOCKER_INPUT_PATH="$DOCKER_INPUT_DIR/$INPUT_FILENAME"
DOCKER_PLAN_PATH="$DOCKER_BIN_DIR/$PLAN_NAME"
DOCKER_OUTPUT_PATH="$DOCKER_BIN_DIR/$OUTPUT_NAME" # Output도 bin 폴더(매핑된 로컬 bin)에 생성

# Docker 실행 명령어
CMD="/app/bin/superres_app \
  --mode 2 \
  --plan-file $DOCKER_PLAN_PATH \
  --input-yuv-file $DOCKER_INPUT_PATH \
  --output-yuv-file $DOCKER_OUTPUT_PATH \
  --width 1920 \
  --height 1080 \
  --input-format i420 \
  --input-full-range 1 \
  --output-width 1920 \
  --output-height 1080 \
  --output-format i420 \
  --output-full-range 1 \
  --prescale 0.75 \
  --overlap 4 \
  --batches 16"

echo "Starting SuperRes in Docker..."
echo "Local Bin:   $LOCAL_BIN (Plan/Output)"
echo "Host Input:  $HOST_INPUT"
echo "--------------------------------"
echo "Docker Input:  $DOCKER_INPUT_PATH"
echo "Docker Plan:   $DOCKER_PLAN_PATH"
echo "Docker Output: $DOCKER_OUTPUT_PATH"
echo "--------------------------------"

# Docker 실행
# LOCAL_BIN -> /app/bin (Plan 읽기 및 Output 쓰기)
# HOST_INPUT dir -> /input (Input 읽기)
docker run --rm -it \
    --gpus all \
    -v "$LOCAL_BIN":$DOCKER_BIN_DIR \
    -v "$(dirname "$HOST_INPUT")":$DOCKER_INPUT_DIR:ro \
    -e LD_LIBRARY_PATH=$DOCKER_BIN_DIR:$LD_LIBRARY_PATH \
    "$IMAGE_NAME" \
    $CMD
