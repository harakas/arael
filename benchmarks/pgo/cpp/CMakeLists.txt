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
