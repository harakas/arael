# Visual-inertial SLAM demo over the generated arael C++ interface.
#   cmake -S . -B build && cmake --build build && ./build/slam_demo_gm
cmake_minimum_required(VERSION 3.16)
project(slam_demo_gm CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(model/cxx arael_gen)

add_executable(slam_demo_gm main.cpp)
target_link_libraries(slam_demo_gm PRIVATE arael::slam_demo_gm)
