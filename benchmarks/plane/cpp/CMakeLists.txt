cmake_minimum_required(VERSION 3.16)
project(plane_bench_cpp CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

find_package(Eigen3 REQUIRED)
find_library(G2O_CORE g2o_core REQUIRED)
find_library(G2O_STUFF g2o_stuff REQUIRED)
find_library(G2O_SOLVER_EIGEN g2o_solver_eigen REQUIRED)
find_library(G2O_TYPES_SLAM3D g2o_types_slam3d REQUIRED)
find_library(G2O_TYPES_SLAM3D_ADDONS g2o_types_slam3d_addons REQUIRED)

add_executable(g2o_plane g2o_plane.cpp)
target_link_libraries(g2o_plane
    ${G2O_CORE} ${G2O_STUFF} ${G2O_SOLVER_EIGEN}
    ${G2O_TYPES_SLAM3D} ${G2O_TYPES_SLAM3D_ADDONS} Eigen3::Eigen)

find_package(Ceres REQUIRED)
add_executable(ceres_plane ceres_plane.cpp)
target_link_libraries(ceres_plane Ceres::ceres Eigen3::Eigen)

# GTSAM runner: custom NoiseModelFactorN factors with analytic Jacobians.
# The Ubuntu libgtsam-dev CMake config is broken (its exports reference a
# non-shipped libCppUnitLite.a), so link libgtsam directly instead of
# find_package(GTSAM).
find_library(GTSAM_LIB gtsam)
find_library(TBB_LIB tbb)
if(GTSAM_LIB)
    add_executable(gtsam_plane gtsam_plane.cpp)
    target_link_libraries(gtsam_plane ${GTSAM_LIB} ${TBB_LIB} Eigen3::Eigen)
else()
    message(STATUS "GTSAM not found; skipping gtsam_plane (apt install libgtsam-dev)")
endif()

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
    add_executable(symforce_plane symforce_plane.cpp)
    target_include_directories(symforce_plane PRIVATE
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
    target_link_libraries(symforce_plane ${SYMFORCE_OPT} ${SYMFORCE_GEN} ${SYMFORCE_CHOLESKY})
    set_target_properties(symforce_plane PROPERTIES
        BUILD_RPATH ${SYMFORCE_BUILD_TEMP})
else()
    message(STATUS "SYMFORCE_DIR not set; skipping symforce_plane")
endif()
