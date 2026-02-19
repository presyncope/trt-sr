#!/bin/bash

# 에러 발생 시 중단
set -e

# 경로 설정
SOURCE_DIR="../../cpp"
BUILD_DIR="./build"
# CMAKE_RUNTIME_OUTPUT_DIRECTORY는 절대 경로를 사용하는 것이 안전합니다.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="$PROJECT_ROOT/bin"

# 기본값 설정
BUILD_TYPE="Release"
DO_CLEAN=false
NO_WARNINGS=false

# 인자 파싱
for arg in "$@"
do
    case $arg in
        --release)
            BUILD_TYPE="Release"
            ;;
        --debug)
            BUILD_TYPE="Debug"
            ;;
        --clean)
            DO_CLEAN=true
            ;;
        --no-warnings)
            NO_WARNINGS=true
            ;;
        *)
            echo "알 수 없는 인자입니다: $arg"
            echo "사용법: $0 [--release|--debug] [--clean] [--no-warnings]"
            exit 1
            ;;
    esac
done

# Clean 옵션 처리
if [ "$DO_CLEAN" = true ]; then
    echo "빌드 디렉토리($BUILD_DIR)를 삭제하고 다시 빌드합니다..."
    rm -rf "$BUILD_DIR"
fi

echo "=========================================="
echo "Build Configuration:"
echo "  Project: superres"
echo "  Type:    $BUILD_TYPE"
echo "  Source:  $SOURCE_DIR"
echo "  Build:   $BUILD_DIR"
echo "  Bin:     $BIN_DIR"
if [ "$NO_WARNINGS" = true ]; then
    echo "  Warnings: Suppressed"
fi
echo "=========================================="

# 빌드 디렉토리 생성 (cmake가 자동으로 처리하지만 명시적으로 확인)
mkdir -p "$BUILD_DIR"

# CMake Arguments 구성
CMAKE_ARGS=(
    "-S" "$SOURCE_DIR"
    "-B" "$BUILD_DIR"
    "-DCMAKE_BUILD_TYPE=$BUILD_TYPE"
    "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=$BIN_DIR"
    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
)

if [ "$NO_WARNINGS" = true ]; then
    CMAKE_ARGS+=("-DCMAKE_CXX_FLAGS=-w" "-DCMAKE_C_FLAGS=-w")
fi

# CMake Configure
cmake "${CMAKE_ARGS[@]}"

# Build
# --parallel: 병렬 빌드 사용
echo "Building..."
cmake --build "$BUILD_DIR" --parallel

echo "Build process finished."
