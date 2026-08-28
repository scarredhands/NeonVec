cmake_minimum_required(VERSION 3.10)
project(NeonVec VERSION 1.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Maximize optimization and target the M1 CPU architecture for NEON SIMD
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mcpu=apple-m1 -Wall -Wextra")

# Include header files
include_directories(include)

# Gather source files
set(SOURCES
    src/VectorStorage.cpp
    src/NSWGraph.cpp
    src/main.cpp
)

# Create the executable
add_executable(NeonVec ${SOURCES})