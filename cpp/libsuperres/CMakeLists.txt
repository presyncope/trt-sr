cmake_minimum_required(VERSION 3.18)
project(superres-build LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CheckLanguage)
check_language(CUDA)

option(ENABLE_SUPERRES "Enable Super Resolution feature with CUDA" ON)

if(CMAKE_CUDA_COMPILER AND ENABLE_SUPERRES)
    message(STATUS "CUDA Compiler found: ${CMAKE_CUDA_COMPILER}. Enabling SuperRes.")
    enable_language(CUDA)
    
    set(CMAKE_CUDA_ARCHITECTURES "native") # 75 80 86 89

    find_package(CUDAToolkit REQUIRED)
    message(STATUS "Found CUDA Toolkit Version: ${CUDAToolkit_VERSION}")
    message(STATUS "CUDA Toolkit Root: ${CUDAToolkit_LIBRARY_ROOT}")

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

    add_library(superres SHARED
        src/superres.cpp
        src/postprocess.cu
        src/preprocess.cu
        src/resize.cu
    )

    target_compile_options(superres PRIVATE -Wno-deprecated-declarations)

    target_include_directories(superres
        PUBLIC 
            ${CMAKE_CURRENT_SOURCE_DIR}/api
        PRIVATE 
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${TENSORRT_INCLUDE_DIR}
    )

    target_link_libraries(superres
        ${NVINFER_LIB}
        ${NVONNXPARSER_LIB}
        CUDA::cudart
    )
else()
    message(STATUS "CUDA Compiler not found. Disabling SuperRes.")
endif()