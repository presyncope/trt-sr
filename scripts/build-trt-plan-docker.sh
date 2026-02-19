#!/bin/bash

# 에러 발생 시 중단
set -e

# 기본값 설정
DOCKER_IMAGE="trt-devel:latest"
APP_DIR="../bin"
INPUT_ONNX=""
OUTPUT_PLAN=""
WORKSPACE_SIZE="128M"
MIN_BATCH=1
MAX_BATCH=32
OPT_BATCH=16
STRONGLY_TYPED=true
VERSION_COMPATIBLE=true
EXCLUDE_LEAN_RUNTIME=true

# 도움말 출력 함수
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --image IMAGE                Docker image name (default: trt-devel:latest)"
    echo "  --appdir PATH                Path to superres_app directory (default: ../bin)"
    echo "  -i, --input PATH             Input ONNX file path (required)"
    echo "  -o, --output PATH            Output Plan file path (required)"
    echo "  --workspace SIZE             Workspace size (e.g., 1024, 128K, 128M, 1G) (default: 128M)"
    echo "  --min-batch INT              Minimum batch size (default: 1)"
    echo "  --max-batch INT              Maximum batch size (default: 32)"
    echo "  --opt-batch INT              Optimal batch size (default: 16)"
    echo "  --strongly-typed BOOL        Enable strongly typed network (default: true)"
    echo "  --version-compatible BOOL    Enable version compatibility (default: true)"
    echo "  --exclude-lean-runtime BOOL  Exclude lean runtime from plan (default: true)"
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
        --appdir)
            APP_DIR="$2"
            shift 2
            ;;
        -i|--input)
            INPUT_ONNX="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_PLAN="$2"
            shift 2
            ;;
        --workspace)
            WORKSPACE_SIZE="$2"
            shift 2
            ;;
        --min-batch)
            MIN_BATCH="$2"
            shift 2
            ;;
        --max-batch)
            MAX_BATCH="$2"
            shift 2
            ;;
        --opt-batch)
            OPT_BATCH="$2"
            shift 2
            ;;
        --strongly-typed)
            STRONGLY_TYPED="$2"
            shift 2
            ;;
        --version-compatible)
            VERSION_COMPATIBLE="$2"
            shift 2
            ;;
        --exclude-lean-runtime)
            EXCLUDE_LEAN_RUNTIME="$2"
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

# 필수 인자 확인
if [ -z "$INPUT_ONNX" ] || [ -z "$OUTPUT_PLAN" ]; then
    echo "Error: Input ONNX file and Output Plan file are required."
    usage
fi

# 경로 절대 경로로 변환
to_abs_path() {
    echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
}

# APP_DIR은 디렉토리이므로 dirname 처리가 필요할 수도 있고 아닐 수도 있음.
# 사용자가 --appdir ./bin 처럼 입력했을 때를 대비.
if [ -d "$APP_DIR" ]; then
    APP_DIR_ABS="$(cd "$APP_DIR" && pwd)"
else
    # 디렉토리가 존재하지 않으면 에러
    echo "Error: App directory '$APP_DIR' does not exist."
    exit 1
fi

INPUT_ONNX_ABS=$(to_abs_path "$INPUT_ONNX")
OUTPUT_PLAN_DIR=$(dirname "$OUTPUT_PLAN")
mkdir -p "$OUTPUT_PLAN_DIR" # 출력 디렉터리 생성
OUTPUT_PLAN_ABS="$(cd "$OUTPUT_PLAN_DIR" && pwd)/$(basename "$OUTPUT_PLAN")"

# Workspace Size 변환 함수
convert_size() {
    local size=$1
    local unit="${size: -1}"
    local number="${size%?}"

    case "$unit" in
        G|g) echo $(($number * 1024 * 1024 * 1024)) ;;
        M|m) echo $(($number * 1024 * 1024)) ;;
        K|k) echo $(($number * 1024)) ;;
        *)   echo $size ;; # 단위가 없거나 숫자인 경우 그대로 반환
    esac
}

WORKSPACE_BYTES=$(convert_size "$WORKSPACE_SIZE")

# Docker 명령어 구성
# -u 옵션은 파일 권한 문제를 피하기 위해 사용
# APP_DIR -> /app (ro)
# INPUT_DIR -> /input (ro)
# OUTPUT_DIR -> /output (rw)

INPUT_DIR=$(dirname "$INPUT_ONNX_ABS")
INPUT_FILE=$(basename "$INPUT_ONNX_ABS")
OUTPUT_DIR=$(dirname "$OUTPUT_PLAN_ABS")
OUTPUT_FILE=$(basename "$OUTPUT_PLAN_ABS")

# Docker 명령어 배열로 구성 (안전한 실행을 위해)
DOCKER_ARGS=(
    run --rm
    --gpus all
    -v "$APP_DIR_ABS":/app:ro
    -v "$INPUT_DIR":/input:ro
    -v "$OUTPUT_DIR":/output:rw
    -u "$(id -u):$(id -g)"
    "$DOCKER_IMAGE"
)

# superres_app 명령어 구성
APP_CMD="LD_LIBRARY_PATH=/app:\$LD_LIBRARY_PATH /app/superres_app --mode 1 \
--onnx-file /input/$INPUT_FILE \
--plan-file /output/$OUTPUT_FILE \
--workspace $WORKSPACE_BYTES \
--min-batch $MIN_BATCH \
--max-batch $MAX_BATCH \
--opt-batch $OPT_BATCH "

if [ "$STRONGLY_TYPED" = "true" ] || [ "$STRONGLY_TYPED" = "True" ]; then
    APP_CMD+="--strongly-typed "
fi

if [ "$VERSION_COMPATIBLE" = "true" ] || [ "$VERSION_COMPATIBLE" = "True" ]; then
    APP_CMD+="--version-compatible "
fi

if [ "$EXCLUDE_LEAN_RUNTIME" = "true" ] || [ "$EXCLUDE_LEAN_RUNTIME" = "True" ]; then
    APP_CMD+="--exclude-lean-runtime "
fi

echo "=========================================="
echo "Building TensorRT Plan in Docker"
echo "  Image:        $DOCKER_IMAGE"
echo "  App Dir:      $APP_DIR_ABS"
echo "  Input ONNX:   $INPUT_ONNX_ABS"
echo "  Output Plan:  $OUTPUT_PLAN_ABS"
echo "  Workspace:    $WORKSPACE_BYTES bytes"
echo "=========================================="

echo "Executing: docker ${DOCKER_ARGS[*]} bash -c \"$APP_CMD\""
docker "${DOCKER_ARGS[@]}" bash -c "$APP_CMD"

echo "Build finished."
