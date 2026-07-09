cmake_minimum_required(VERSION 3.16)
project(loc_bench_cpp CXX)
set(CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Ceres REQUIRED)
add_executable(ceres_loc ceres_loc.cpp)
target_link_libraries(ceres_loc Ceres::ceres)

# g2o runner: custom pose vertex + drift/tilt/bearing/odometry edges with
# analytic Jacobians.
find_package(Eigen3 REQUIRED)
find_library(G2O_CORE g2o_core REQUIRED)
find_library(G2O_STUFF g2o_stuff REQUIRED)
find_library(G2O_SOLVER_CHOLMOD g2o_solver_cholmod REQUIRED)
find_library(CHOLMOD cholmod REQUIRED)
add_executable(g2o_loc g2o_loc.cpp)
target_include_directories(g2o_loc PRIVATE /usr/include/suitesparse)
target_link_libraries(g2o_loc
    ${G2O_CORE} ${G2O_STUFF} ${G2O_SOLVER_CHOLMOD} ${CHOLMOD} Eigen3::Eigen)

# GTSAM runner: custom NoiseModelFactorN factors with analytic Jacobians.
# The Ubuntu libgtsam-dev CMake config is broken (its exports reference a
# non-shipped libCppUnitLite.a), so link libgtsam directly.
find_library(GTSAM_LIB gtsam)
find_library(TBB_LIB tbb)
if(GTSAM_LIB)
    add_executable(gtsam_loc gtsam_loc.cpp)
    target_link_libraries(gtsam_loc ${GTSAM_LIB} ${TBB_LIB} Eigen3::Eigen)
else()
    message(STATUS "GTSAM not found; skipping gtsam_loc (apt install libgtsam-dev)")
endif()

# SymForce loc runner is added here when it lands.
