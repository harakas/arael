# M3500 pose-graph demo over the generated arael C++ interface.
#   cmake -S . -B build && cmake --build build && ./build/m3500_demo
cmake_minimum_required(VERSION 3.16)
project(m3500_demo CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(model/cxx arael_gen)

add_executable(m3500_demo main.cpp)
target_link_libraries(m3500_demo PRIVATE arael::m3500_demo)
target_compile_definitions(m3500_demo PRIVATE
  M3500_DEFAULT_DATASET="${CMAKE_CURRENT_SOURCE_DIR}/../../benchmarks/pgo/datasets/input_M3500_g2o.g2o")
