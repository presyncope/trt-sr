#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 기본값 설정
BUILD_TYPE="Release"
DO_CLEAN=false
NO_WARNINGS=false
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
            echo "사용법: $0 [--release|--debug] [--clean] [--no-warnings] [--runtime] [--prefix <path>] [--source <dir>]"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Local Build Configuration:"
echo "  Project Root: $PROJECT_ROOT"
echo "  Type:         $BUILD_TYPE"
echo "  Source:       $SOURCE_DIR"
echo "  Prefix:       $INSTALL_PREFIX"
echo "  Runtime Mode: $RUNTIME_MODE"
echo "  Clean:        $DO_CLEAN"
echo "=========================================="

BUILD_DIR="$SCRIPT_DIR/build"

# Clean 옵션 처리
if [ "$DO_CLEAN" = true ]; then
    echo "빌드 디렉터리 정리 중..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"

# Prefix 경로 처리 (절대 경로 변환)
if [[ "$INSTALL_PREFIX" != /* ]]; then
    # 상대 경로인 경우 PROJECT_ROOT 기준
    ABS_INSTALL_PREFIX="$PROJECT_ROOT/$INSTALL_PREFIX"
else
    ABS_INSTALL_PREFIX="$INSTALL_PREFIX"
fi

# Build Command 구성
CMAKE_CMD="cmake -S $SOURCE_DIR -B $BUILD_DIR "
CMAKE_CMD+="-DCMAKE_BUILD_TYPE=$BUILD_TYPE "
CMAKE_CMD+="-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=\"$ABS_INSTALL_PREFIX\" "
CMAKE_CMD+="-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=\"$ABS_INSTALL_PREFIX\" "
CMAKE_CMD+="-DCMAKE_EXPORT_COMPILE_COMMANDS=ON "

if [ "$RUNTIME_MODE" = true ]; then
    CMAKE_CMD+="-DUSE_TRT_LEAN=ON -DUSE_ONNX_PARSER=OFF "
fi

if [ "$NO_WARNINGS" = true ]; then
    CMAKE_CMD+="-DCMAKE_CXX_FLAGS=-w -DCMAKE_C_FLAGS=-w "
fi

echo "Configuring cmake..."
echo "$CMAKE_CMD"
eval $CMAKE_CMD

echo "Building..."
cmake --build "$BUILD_DIR" --parallel

echo "Build process finished. Artifacts are in: $ABS_INSTALL_PREFIX"
