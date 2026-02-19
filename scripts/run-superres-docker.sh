#!/bin/bash

set -e

# 기본값 설정
DOCKER_IMAGE="trt-lean:latest"
CONFIG_FILE="superres-config.yaml"

# 도움말 출력 함수
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --image IMAGE                Docker image name (default: trt-lean:latest)"
    echo "  -c, --config PATH            Path to YAML config file (default: superres-config.yaml)"
    echo "  -h, --help                   Print this help message"
    exit 1
}

# 인자 파싱
while [[ $# -gt 0 ]]; do
    case "$1" in
        --image)
            DOCKER_IMAGE="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            ;;
    esac
done

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found."
    exit 1
fi

# 절대 경로 변환 함수
to_abs_path() {
    # $1: relative path
    # $2: base directory (optional, default: cwd)
    
    local path="$1"
    local base="$2"
    
    if [[ "$path" == /* ]]; then
        echo "$path"
    else
        if [ -n "$base" ]; then
            echo "$(cd "$base" && cd "$(dirname "$path")" && pwd)/$(basename "$path")"
        else
            echo "$(cd "$(dirname "$path")" && pwd)/$(basename "$path")"
        fi
    fi
}

CONFIG_PATH_ABS=$(to_abs_path "$CONFIG_FILE")
CONFIG_DIR=$(dirname "$CONFIG_PATH_ABS")
SCIPRT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Python을 사용하여 YAML 파싱 및 절대 경로 추출 (host 기준)
# YAML 내의 상대 경로는 YAML 파일 위치 기준임.
eval $(python3 -c "
import yaml
import os
import sys

config_path = '$CONFIG_PATH_ABS'
config_dir = '$CONFIG_DIR'

def to_abs(path, base_dir):
    if not path: return ''
    if os.path.isabs(path): return path
    return os.path.normpath(os.path.join(base_dir, path))

try:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f) or {}

    input_path = to_abs(config.get('input', ''), config_dir)
    output_path = to_abs(config.get('output', ''), config_dir)
    plan_path = to_abs(config.get('plan_file', ''), config_dir)
    app_dir = to_abs(config.get('app_dir', ''), config_dir)

    print(f'INPUT_ABS=\"{input_path}\"')
    print(f'OUTPUT_ABS=\"{output_path}\"')
    print(f'PLAN_ABS=\"{plan_path}\"')
    print(f'APP_DIR_ABS=\"{app_dir}\"')

except Exception as e:
    print(f'Error parsing config: {e}', file=sys.stderr)
    sys.exit(1)
")

# 경로 확인
if [ ! -f "$INPUT_ABS" ]; then
    echo "Error: Input file '$INPUT_ABS' not found."
    exit 1
fi

if [ ! -f "$PLAN_ABS" ]; then
    echo "Error: Plan file '$PLAN_ABS' not found."
    exit 1
fi

if [ ! -d "$APP_DIR_ABS" ]; then
    echo "Error: App directory '$APP_DIR_ABS' not found."
    exit 1
fi

# Docker Mount Points
# Container structure:
# /input -> dirname(INPUT_ABS)
# /output -> dirname(OUTPUT_ABS)
# /plan -> dirname(PLAN_ABS)
# /app_bin -> APP_DIR_ABS
# /scripts -> SCRIPT_DIR
# /config -> CONFIG_DIR

INPUT_DIR=$(dirname "$INPUT_ABS")
INPUT_FILE=$(basename "$INPUT_ABS")
OUTPUT_DIR=$(dirname "$OUTPUT_ABS")
mkdir -p "$OUTPUT_DIR"
OUTPUT_FILE=$(basename "$OUTPUT_ABS")
PLAN_DIR=$(dirname "$PLAN_ABS")
PLAN_FILE=$(basename "$PLAN_ABS")
CONFIG_FILE_NAME=$(basename "$CONFIG_PATH_ABS")

# Docker args
DOCKER_ARGS=(
    run --rm
    --gpus all
    -v "$INPUT_DIR":/input:ro
    -v "$OUTPUT_DIR":/output:rw
    -v "$PLAN_DIR":/plan:ro
    -v "$APP_DIR_ABS":/app_bin:ro
    -v "$SCIPRT_DIR":/scripts:ro
    -v "$CONFIG_DIR":/config:ro
    -u "$(id -u):$(id -g)"
    "$DOCKER_IMAGE"
)

# Command to run inside docker
# We invoke superres-rtonly.py but we need to override paths to point to mapped locations
# We can pass them as arguments to override config values.
# Config file itself is passed as /config/filename
CMD="python3 /scripts/superres-rtonly.py \
--config /config/$CONFIG_FILE_NAME \
--input /input/$INPUT_FILE \
--output /output/$OUTPUT_FILE \
--plan-file /plan/$PLAN_FILE \
--app-dir /app_bin"

echo "=========================================="
echo "Running SuperRes in Docker"
echo "  Image:      $DOCKER_IMAGE"
echo "  Config:     $CONFIG_PATH_ABS"
echo "  Input:      $INPUT_ABS"
echo "  Output:     $OUTPUT_ABS"
echo "  Plan:       $PLAN_ABS"
echo "  App Dir:    $APP_DIR_ABS"
echo "=========================================="

echo "Executing: docker ${DOCKER_ARGS[*]} $CMD"
docker "${DOCKER_ARGS[@]}" $CMD
