#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# SCRIPT_DIR is .../scripts
# PROJECT_ROOT is .../trt-sr
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 기본값 설정
BUILD_TYPE="Release"
DO_CLEAN=false
NO_WARNINGS=false
DOCKER_IMAGE="trt-devel:latest"
INSTALL_PREFIX="bin"
SOURCE_DIR="."
RUNTIME_MODE=false

# 인자 파싱
while [[ $# -gt 0 ]]; do
    case "$1" in
        --release)
            BUILD_TYPE="Release"
            shift
            ;;
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --clean)
            DO_CLEAN=true
            shift
            ;;
        --no-warnings)
            NO_WARNINGS=true
            shift
            ;;
        --runtime)
            RUNTIME_MODE=true
            shift
            ;;
        --image)
            if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
                DOCKER_IMAGE="$2"
                shift 2
            else
                echo "Error: Argument for --image is missing" >&2
                exit 1
            fi
            ;;
        --prefix)
            if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
                INSTALL_PREFIX="$2"
                shift 2
            else
                echo "Error: Argument for --prefix is missing" >&2
                exit 1
            fi
            ;;
        --source)
            if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
                SOURCE_DIR="$2"
                shift 2
            else
                echo "Error: Argument for --source is missing" >&2
                exit 1
            fi
            ;;
        *)
            echo "알 수 없는 인자입니다: $1"
            echo "사용법: $0 [--release|--debug] [--clean] [--no-warnings] [--runtime] [--image <image_name>] [--prefix <path>] [--source <dir>]"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Docker Build Configuration:"
echo "  Project Root: $PROJECT_ROOT"
echo "  Image:        $DOCKER_IMAGE"
echo "  Type:         $BUILD_TYPE"
echo "  Source:       $SOURCE_DIR"
echo "  Prefix:       $INSTALL_PREFIX"
echo "  Runtime Mode: $RUNTIME_MODE"
echo "  Clean:        $DO_CLEAN"
echo "=========================================="

# Clean 옵션 처리 (호스트에서 수행)
if [ "$DO_CLEAN" = true ]; then
    echo "빌드 디렉터리 정리 중..."
    rm -rf "$SCRIPT_DIR/build"
fi

# Docker 명령어 구성
# -u $(id -u):$(id -g): 생성된 파일의 소유권이 현재 사용자와 일치하도록 설정
# -v "$PROJECT_ROOT":/workspace: 프로젝트 루트를 컨테이너의 /workspace에 마운트
# -w /workspace/scripts: 작업 디렉터리 설정 (스크립트 위치)

DOCKER_CMD="docker run --rm -v $PROJECT_ROOT:/workspace -w /workspace/scripts -u $(id -u):$(id -g) $DOCKER_IMAGE"

# Prefix 경로 처리 (절대 경로가 아니면 /workspace 기준 상대 경로로 변환)
if [[ "$INSTALL_PREFIX" != /* ]]; then
    DOCKER_INSTALL_PREFIX="/workspace/$INSTALL_PREFIX"
else
    DOCKER_INSTALL_PREFIX="$INSTALL_PREFIX"
fi

# 컨테이너 내부에서 실행할 스크립트 구성
# SOURCE_DIR: /workspace/cpp (scripts 기준 ../cpp)
# OUTPUT_DIR: $DOCKER_INSTALL_PREFIX (RUNTIME 및 LIBRARY 모두 설정)

BUILD_CMD="TRT_ROOT=\$(ls -d /TensorRT* 2>/dev/null | head -n 1) && "
BUILD_CMD+="if [ -z \"\$TRT_ROOT\" ]; then echo \"WARNING: TensorRT root not found in /\"; else echo \"Found TensorRT at: \$TRT_ROOT\"; fi && "
BUILD_CMD+="mkdir -p build && "
BUILD_CMD+="cmake -S $SOURCE_DIR -B build "
BUILD_CMD+="-DCMAKE_BUILD_TYPE=$BUILD_TYPE "
BUILD_CMD+="-DTENSORRT_INCLUDE_DIR=\$TRT_ROOT/include "
BUILD_CMD+="-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=$DOCKER_INSTALL_PREFIX "
BUILD_CMD+="-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$DOCKER_INSTALL_PREFIX "
BUILD_CMD+="-DCMAKE_EXPORT_COMPILE_COMMANDS=ON "

if [ "$RUNTIME_MODE" = true ]; then
    BUILD_CMD+="-DUSE_TRT_LEAN=ON -DUSE_ONNX_PARSER=OFF "
fi

if [ "$NO_WARNINGS" = true ]; then
    BUILD_CMD+="-DCMAKE_CXX_FLAGS=-w -DCMAKE_C_FLAGS=-w "
fi

BUILD_CMD+="&& cmake --build build --parallel"

echo "Executing build in Docker..."
# bash -c를 사용하여 명령어 문자열 실행
$DOCKER_CMD bash -c "$BUILD_CMD"

echo "Docker build process finished. Artifacts are in: $DOCKER_INSTALL_PREFIX"
