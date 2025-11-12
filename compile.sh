#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Starting C++ Compilation for student_agent_module ---"

# 1. Clean up previous build artifacts if the build directory exists
if [ -d "build" ]; then
    echo "Removing previous build directory..."
    rm -rf build
fi

# 2. Create and enter the build directory
echo "Creating and entering build directory..."
mkdir build
cd build

# 3. Configure the build using cmake
# We pass the pybind11 directory path as recommended in the README,
# and specify C/C++ compilers (as often required in tournament environments).
echo "Configuring CMake..."
cmake .. \
 -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir) \
 -DCMAKE_C_COMPILER=gcc \
 -DCMAKE_CXX_COMPILER=g++

# 4. Compile the module
echo "Compiling with make..."
make

# 5. Return to the root directory
cd ..

echo "--- Compilation successful! Module is located in ./build ---"

# The Python wrapper (student_agent_cpp.py) should now be able to import the module.