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
