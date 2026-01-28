#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DOCKER_IMAGE="trt-build"

echo "Building inside ${DOCKER_IMAGE}..."

docker run --rm \
    --entrypoint /bin/bash \
    -v "$SCRIPT_DIR:/workspace" \
    -w /workspace \
    ${DOCKER_IMAGE} \
    -c "mkdir -p build && cd build && cmake .. && make -j\$(nproc)"

echo "Build complete."
