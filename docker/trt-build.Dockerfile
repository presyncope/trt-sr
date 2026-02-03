#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

ARG CUDA_VERSION=12.8.0

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04
LABEL maintainer="NVIDIA CORPORATION"

ENV TRT_VERSION=10.14.1.48
SHELL ["/bin/bash", "-c"]

# Setup user account
ARG uid=1000
ARG gid=1000
RUN groupadd -r -f -g ${gid} trtuser && useradd -o -r -l -u ${uid} -g ${gid} -ms /bin/bash trtuser
RUN usermod -aG sudo trtuser
RUN echo 'trtuser:nvidia' | chpasswd
RUN mkdir -p /workspace && chown trtuser /workspace

ENV DEBIAN_FRONTEND=noninteractive

# Update CUDA signing key (if needed, usually handled by base image but good practice from ref)
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub

# Install required libraries for C++ build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    pkg-config \
    libssl-dev \
    ca-certificates \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install CMake (Newer version often needed for newer CUDA/TRT, using apt cmake might be old but 24.04 usually has decent one. 
# The reference installed 3.27.9. Ubuntu 24.04 has 3.28+. So apt install cmake should be fine.
# We stick to apt cmake for simplicity unless specific version needed.)

# Install TensorRT C++ Libraries
# Using the URL from the reference logic for CUDA 12.x
# URL: https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.14.1/tars/TensorRT-10.14.1.48.Linux.x86_64-gnu.cuda-12.9.tar.gz
RUN wget -q https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.14.1/tars/TensorRT-10.14.1.48.Linux.x86_64-gnu.cuda-12.9.tar.gz \
    && tar -xf TusfeensorRT-10.14.1.48.Linux.x86_64-gnu.cuda-12.9.tar.gz -C /usr/local \
    && mv /usr/local/TensorRT-10.14.1.48 /usr/local/tensorrt \
    && rm TensorRT-10.14.1.48.Linux.x86_64-gnu.cuda-12.9.tar.gz

# Configure Dynamic Linker
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/tensorrt/lib"
ENV TENSORRT_INCLUDE_DIR="/usr/local/tensorrt/include"
ENV TRT_LIBPATH="/usr/local/tensorrt/lib"
ENV TRT_OSSPATH="/usr/local/tensorrt"

# Ensure CMake can find TensorRT via standard paths if find_package usage relies on it
# or manually via environment variables in CMakeLists.
# We also link include directory for easier access
# RUN ln -s /usr/local/tensorrt/include/* /usr/include/ && \
#     ln -s /usr/local/tensorrt/lib/*.so* /usr/lib/

WORKDIR /workspace
USER trtuser
CMD ["/bin/bash"]
