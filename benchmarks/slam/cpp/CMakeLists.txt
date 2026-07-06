cmake_minimum_required(VERSION 3.16)
project(slam_bench_cpp CXX)
set(CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Ceres REQUIRED)
add_executable(ceres_slam ceres_slam.cpp)
target_link_libraries(ceres_slam Ceres::ceres)

# SymForce runner: requires a built symforce source checkout, pointed to
# by -DSYMFORCE_DIR=... (or env SYMFORCE_DIR). The generated factor
# headers come from ../symforce_gen.py (committed under symforce_gen/).
if(NOT SYMFORCE_DIR AND DEFINED ENV{SYMFORCE_DIR})
    set(SYMFORCE_DIR $ENV{SYMFORCE_DIR})
endif()
if(SYMFORCE_DIR)
    file(GLOB SYMFORCE_BUILD_TEMP ${SYMFORCE_DIR}/build/temp.*)
    find_library(SYMFORCE_OPT symforce_opt PATHS ${SYMFORCE_BUILD_TEMP} NO_DEFAULT_PATH)
    find_library(SYMFORCE_GEN symforce_gen PATHS ${SYMFORCE_BUILD_TEMP} NO_DEFAULT_PATH)
    find_library(SYMFORCE_CHOLESKY symforce_cholesky PATHS ${SYMFORCE_BUILD_TEMP} NO_DEFAULT_PATH)
    add_executable(symforce_slam symforce_slam.cpp)
    target_include_directories(symforce_slam PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${SYMFORCE_DIR}
        ${SYMFORCE_DIR}/gen/cpp
        ${SYMFORCE_BUILD_TEMP}/lcmtypes/cpp
        ${SYMFORCE_DIR}/third_party/skymarshal/include
        ${SYMFORCE_DIR}/third_party/eigen_lcm/lcmtypes/eigen_lcm_lcm/cpp
        ${SYMFORCE_BUILD_TEMP}/_deps/metis-src/include
        ${SYMFORCE_BUILD_TEMP}/_deps/spdlog-src/include
        ${SYMFORCE_BUILD_TEMP}/_deps/fmtlib-src/include
        /usr/include/eigen3)
    target_link_libraries(symforce_slam ${SYMFORCE_OPT} ${SYMFORCE_GEN} ${SYMFORCE_CHOLESKY})
    set_target_properties(symforce_slam PROPERTIES
        BUILD_RPATH ${SYMFORCE_BUILD_TEMP})
endif()
