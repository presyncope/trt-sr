cmake_minimum_required(VERSION 3.12)
project(real-esrgan-build LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86)

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/build" CACHE PATH "default install path" FORCE)
endif()

# Suppress error about FindCUDA being removed in newer CMake versions
if(POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
endif()
find_package(CUDA REQUIRED)

# Find TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS /usr/include /usr/local/include /usr/local/tensorrt/include
)
find_library(NVINFER_LIB nvinfer
    HINTS /usr/lib /usr/local/lib /usr/local/tensorrt/lib
)
find_library(NVONNXPARSER_LIB nvonnxparser
    HINTS /usr/lib /usr/local/lib /usr/local/tensorrt/lib
)

if (NOT NVINFER_LIB OR NOT NVONNXPARSER_LIB OR NOT TENSORRT_INCLUDE_DIR)
    message(FATAL_ERROR "Could not find TensorRT. Please set TENSORRT_INCLUDE_DIR and library paths.")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
include_directories(SYSTEM ${TENSORRT_INCLUDE_DIR})

add_executable(real-esrgan-app
    src/main.cpp
    src/real-esrgan.cpp
    src/postprocess.cu
    src/preprocess.cu
    src/resize.cu
)

target_link_libraries(real-esrgan-app
    ${NVINFER_LIB}
    ${NVONNXPARSER_LIB}
    ${CUDA_LIBRARIES}
)
