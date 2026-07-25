# 2D SLAM demo over the generated arael C++ interface.
#   cmake -S . -B build && cmake --build build && ./build/slam2d_simple_demo
cmake_minimum_required(VERSION 3.16)
project(slam2d_simple_demo CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../model/cxx arael_gen)

add_executable(slam2d_simple_demo main.cpp)
target_link_libraries(slam2d_simple_demo PRIVATE arael::slam2d_simple)
