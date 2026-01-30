cmake_minimum_required(VERSION 3.18)
project(superres_app LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)

add_executable(superres_app main.cpp)

target_link_libraries(superres_app PRIVATE superres)
