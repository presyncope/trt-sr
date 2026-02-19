ARG CUDA_VERSION=13.1.0

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TRT_VERSION=10.15.1

# 1. 시스템 패키지 및 TensorRT Lean 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnvinfer-lean10=${TRT_VERSION}* \
    libnvinfer-vc-plugin10=${TRT_VERSION}* \
    && rm -rf /var/lib/apt/lists/* \
    && apt-mark hold libnvinfer-lean10 libnvinfer-vc-plugin10

# 1.1 Install ffmpeg and Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# 1.2 Set up Python environment and install av
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install --no-cache-dir av pyyaml numpy

# 2. 프로덕션용 Non-root 유저 생성 (공식 Dockerfile 참고)
ARG uid=1000
ARG gid=1000
RUN groupadd -r -f -g ${gid} trtuser && \
    useradd -o -r -l -u ${uid} -g ${gid} -ms /bin/bash trtuser

# 3. 작업 디렉토리 설정 및 권한 부여
RUN mkdir -p /workspace && chown trtuser:trtuser /workspace
WORKDIR /workspace

# 4. 환경 변수 설정
ENV TRT_LIBPATH=/usr/lib/x86_64-linux-gnu/
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${TRT_LIBPATH}"

# 5. 유저 전환
USER trtuser

CMD ["/bin/bash", "-c", "echo 'TensorRT Lean Environment Ready' && dpkg -l | grep nvinfer"]