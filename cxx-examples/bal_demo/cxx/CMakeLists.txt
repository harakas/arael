# BAL bundle adjustment over the generated arael C++ interface, CPU or GPU.
#   cmake -S . -B build [-DARAEL_GPU=cuda] && cmake --build build && ./build/bal_demo
cmake_minimum_required(VERSION 3.16)
project(bal_demo CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../model/cxx arael_gen)

add_executable(bal_demo main.cpp)
target_link_libraries(bal_demo PRIVATE arael::bal_demo)
target_compile_definitions(bal_demo PRIVATE
  BAL_DEFAULT_DATASET="${CMAKE_CURRENT_SOURCE_DIR}/../../../benchmarks/bal/datasets/problem-49-7776-pre.txt")
