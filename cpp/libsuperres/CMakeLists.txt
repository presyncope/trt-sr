project(superres-build LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CheckLanguage)
check_language(CUDA)

option(ENABLE_SUPERRES "Enable Super Resolution feature with CUDA" ON)

# 1. Check CUDA Compiler existence
if(NOT CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA Compiler not found. Disabling SuperRes.")
    set(SUPERRES_FOUND OFF CACHE INTERNAL "SuperRes library is built and available")
    return()
endif()

# 2. Check User Option
if(NOT ENABLE_SUPERRES)
    message(STATUS "SuperRes disabled by user.")
    set(SUPERRES_FOUND OFF CACHE INTERNAL "SuperRes library is built and available")
    return()
endif()

message(STATUS "CUDA Compiler found: ${CMAKE_CUDA_COMPILER}. Enabling SuperRes.")
enable_language(CUDA)

set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)

find_package(CUDAToolkit REQUIRED)
message(STATUS "Found CUDA Toolkit Version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Toolkit Root: ${CUDAToolkit_LIBRARY_ROOT}")

# 3. Find TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS /usr/include /usr/local/include /usr/local/tensorrt/include
)
find_library(NVINFER_LIB nvinfer
    HINTS /usr/lib /usr/local/lib /usr/local/tensorrt/lib
)
find_library(NVONNXPARSER_LIB nvonnxparser
    HINTS /usr/lib /usr/local/lib /usr/local/tensorrt/lib
)

# 4. Validate TensorRT and Define Target
if (NOT NVINFER_LIB OR NOT TENSORRT_INCLUDE_DIR)
    message(STATUS "Could not find TensorRT (nvinfer or headers). SuperRes build will be skipped (ENABLE_SUPERRES remains ON).")
    set(SUPERRES_FOUND OFF CACHE INTERNAL "SuperRes library is built and available")
    return()
endif()

message(STATUS "TensorRT Found:")
message(STATUS "  Include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "  Lib nvinfer: ${NVINFER_LIB}")

if (NVONNXPARSER_LIB)
    message(STATUS "  Lib nvonnxparser: ${NVONNXPARSER_LIB} (Found)")
else()
    message(STATUS "  Lib nvonnxparser: NOT FOUND. sr_build will be disabled.")
endif()

add_library(superres SHARED
    src/superres.cpp
    src/postprocess.cu
    src/preprocess.cu
    src/resize.cu
    src/common.cpp
)

target_compile_options(superres PRIVATE -Wno-deprecated-declarations)

target_include_directories(superres
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/api
    PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${TENSORRT_INCLUDE_DIR}
)

if (NVONNXPARSER_LIB)
    target_compile_definitions(superres PRIVATE ENABLE_NVONNXPARSER=1)
    target_link_libraries(superres
        ${NVINFER_LIB}
        ${NVONNXPARSER_LIB}
        CUDA::cudart
    )
else()
    target_compile_definitions(superres PRIVATE ENABLE_NVONNXPARSER=0)
    target_link_libraries(superres
        ${NVINFER_LIB}
        CUDA::cudart
    )
endif()

# 다른 프로젝트에서 참조할 수 있도록 상태 변수 설정
set(SUPERRES_FOUND ON CACHE INTERNAL "SuperRes library is built and available")

# C++ 코드에서도 USE_SUPERRES 매크로를 사용할 수 있도록 정의
target_compile_definitions(superres PUBLIC USE_SUPERRES)

# # Install ONNX models
# install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/models/
#     DESTINATION lib/onnx
#     COMPONENT hero
#     FILES_MATCHING PATTERN "*.onnx"
# )