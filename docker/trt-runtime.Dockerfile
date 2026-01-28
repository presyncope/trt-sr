#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

ARG CUDA_VERSION=12.8.0

# Use RUNTIME image (significantly smaller, contains only libs)
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04
LABEL maintainer="NVIDIA CORPORATION"

ENV TRT_VERSION=10.14.1.48
SHELL ["/bin/bash", "-c"]

# Setup user account
ARG uid=1000
ARG gid=1000
RUN groupadd -r -f -g ${gid} trtuser && useradd -o -r -l -u ${uid} -g ${gid} -ms /bin/bash trtuser
RUN usermod -aG sudo trtuser
RUN echo 'trtuser:nvidia' | chpasswd
# Create workspace
RUN mkdir -p /workspace && chown trtuser /workspace

ENV DEBIAN_FRONTEND=noninteractive

# Install minimal runtime dependencies
# wget/tar/ca-certificates: to download TRT
# NOTE: We remove wget after use to keep image minimal if desired, 
# but keeping it is often useful. To strictly minimize, we can remove.
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    tar \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install TensorRT Runtime Libraries ONLY
RUN wget -q https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.14.1/tars/TensorRT-10.14.1.48.Linux.x86_64-gnu.cuda-12.9.tar.gz \
    && tar -xf TensorRT-10.14.1.48.Linux.x86_64-gnu.cuda-12.9.tar.gz -C /usr/local \
    && mv /usr/local/TensorRT-10.14.1.48 /usr/local/tensorrt \
    && rm TensorRT-10.14.1.48.Linux.x86_64-gnu.cuda-12.9.tar.gz \
    # Remove things not needed for runtime (headers, samples, python, etc)
    && rm -rf /usr/local/tensorrt/include \
    && rm -rf /usr/local/tensorrt/samples \
    && rm -rf /usr/local/tensorrt/python \
    && rm -rf /usr/local/tensorrt/data \
    && rm -rf /usr/local/tensorrt/doc \
    && rm -rf /usr/local/tensorrt/onnx_graphsurgeon

# Configure Dynamic Linker
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/tensorrt/lib"
# For runtime, we do NOT need TENSORRT_INCLUDE_DIR or related CMake envs.

WORKDIR /workspace
USER trtuser
CMD ["/bin/bash"]
