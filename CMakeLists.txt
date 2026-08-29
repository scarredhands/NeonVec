cmake_minimum_required(VERSION 3.14...3.27)
project(NeonVec VERSION 1.0)

# Generate map for VS Code to remove red squiggles
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(include)

# ==========================================
# Pybind11 Python Module Configuration
# ==========================================

# --- Anaconda Python Fix ---
# Force Pybind11 to use modern Python3 detection to find Anaconda headers correctly
set(PYBIND11_FINDPYTHON ON)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# 1. Automatically fetch pybind11 directly from GitHub
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# 2. Build the Python module 
pybind11_add_module(neonvec 
    src/bindings.cpp 
    src/CoreEngine.cpp 
    src/HNSWGraph.cpp 
    src/VectorStorage.cpp
)

# 3. Apply Apple M1 NEON hardware optimizations
target_compile_options(neonvec PRIVATE -O3 -mcpu=apple-m1 -Wall -Wextra)