# A consumer of the generated interface, exactly as a user would write
# it: add_subdirectory the generated cxx/ tree and link the target.
# MODEL_CXX_DIR is passed by the driving test.
cmake_minimum_required(VERSION 3.16)
project(arael_consumer CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(${MODEL_CXX_DIR} arael_gen)

add_executable(smoke main.cpp)
target_link_libraries(smoke PRIVATE arael::cxx_fit)

# A shared library linking the target, and a program using it. The
# staticlib's symbols must stay inside the library; the driving test reads
# its exports.
add_library(consumer_shared SHARED shared.cpp)
target_link_libraries(consumer_shared PRIVATE arael::cxx_fit)
add_executable(smoke_shared smoke_shared.cpp)
target_link_libraries(smoke_shared PRIVATE consumer_shared)
