cmake_minimum_required(VERSION 3.16)
project(pgo_bench_cpp CXX)
set(CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Ceres REQUIRED)
add_executable(ceres_bench ceres_bench.cpp)
target_link_libraries(ceres_bench Ceres::ceres)

# Ubuntu's g2o package ships no cmake config; link the libraries directly.
find_package(Eigen3 REQUIRED)
find_library(G2O_CORE g2o_core REQUIRED)
find_library(G2O_STUFF g2o_stuff REQUIRED)
find_library(G2O_TYPES_SLAM2D g2o_types_slam2d REQUIRED)
find_library(G2O_SOLVER_CHOLMOD g2o_solver_cholmod REQUIRED)
find_library(CHOLMOD cholmod REQUIRED)
add_executable(g2o_bench g2o_bench.cpp)
target_include_directories(g2o_bench PRIVATE /usr/include/suitesparse)
target_link_libraries(g2o_bench
    ${G2O_CORE} ${G2O_STUFF} ${G2O_TYPES_SLAM2D} ${G2O_SOLVER_CHOLMOD}
    ${CHOLMOD} Eigen3::Eigen)

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
    add_executable(symforce_bench symforce_bench.cpp)
    target_include_directories(symforce_bench PRIVATE
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
    target_link_libraries(symforce_bench ${SYMFORCE_OPT} ${SYMFORCE_GEN} ${SYMFORCE_CHOLESKY})
    set_target_properties(symforce_bench PROPERTIES
        BUILD_RPATH ${SYMFORCE_BUILD_TEMP})
endif()
