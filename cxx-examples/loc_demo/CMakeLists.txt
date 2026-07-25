# Localization demo over the generated arael C++ interface.
#   cmake -S . -B build && cmake --build build && ./build/loc_demo
cmake_minimum_required(VERSION 3.16)
project(loc_demo CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(model/cxx arael_gen)

add_executable(loc_demo main.cpp)
target_link_libraries(loc_demo PRIVATE arael::loc_demo)
