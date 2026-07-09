cmake_minimum_required(VERSION 3.16)
project(loc_bench_cpp CXX)
set(CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Ceres REQUIRED)
add_executable(ceres_loc ceres_loc.cpp)
target_link_libraries(ceres_loc Ceres::ceres)

# g2o / GTSAM / SymForce loc runners are added here as they land (mirroring
# benchmarks/slam/cpp/CMakeLists.txt).
