#!/usr/bin/env bash
set -eux

# Set CUDA version based on nvcc output
export CUDAVER=$(nvcc --version | sed -n 's/^.*release \([0-9]\+\.[0-9]\+\).*$/\1/p' | tr -d '.')

# Determine the platform name based on the current architecture
ARCH=$(uname -m)
if [ "$ARCH" == "x86_64" ]; then
    export PLAT_NAME="manylinux2014_x86_64"
elif [ "$ARCH" == "aarch64" ]; then
    export PLAT_NAME="manylinux2014_aarch64"
else
    echo "Unsupported architecture: $ARCH"
    exit 1
fi

# # Install necessary packages
# apt update -y

# Clean up and prepare the build directory
rm -rf /tmpbuild
mkdir -p /tmpbuild

# Install build dependencies with caching
mkdir -p /docker_build_cache/.pip
pip3 install --cache-dir /docker_build_cache/.pip ninja cmake wheel

# Ensure the target directory exists before copying
mkdir -p /lmdeploy

# Copy source files to the build directory
cp -r /ant_lmdeploy/* /lmdeploy/

# Build the project
cd /lmdeploy
rm -rf /lmdeploy/lib
mkdir -p build && cd build && rm -rf *

# Use generate.sh to set up the build environment with external cache directory
bash ../generate.sh

ninja -j$(nproc) && ninja install || { echo "Build failed"; exit 1; }

cd ..
rm -rf build

# Update version information if LMDEPLOY_VERSION is set
if [ -n "$LMDEPLOY_VERSION" ]; then
    sed -i "s/__version__ = '.*'/__version__ = '$LMDEPLOY_VERSION'/" /lmdeploy/lmdeploy/version.py
fi

# Build the wheel with the determined platform name
python setup.py bdist_wheel --cuda=${CUDAVER} --plat-name $PLAT_NAME -d /tmpbuild/

# Process the built wheel to include CUDA version information
for whl in /tmpbuild/*.whl; do
    base_name=$(basename "$whl" .whl)
    
    # Extract version number and add CUDA information
    version=$(echo "$base_name" | sed -n 's/.*-\([0-9.]*\)-cp.*/\1/p')
    new_version="${version}+cu${CUDAVER}"
    
    # Construct the new file name
    new_base_name=$(echo "$base_name" | sed "s/${version}/${new_version}/")
    
    mv "$whl" "/tmpbuild/${new_base_name}.whl"

    # Check WRITE_WHL environment variable to determine if the wheel should be copied to /lmdeploy_build
    if [ "$WRITE_WHL" == "true" ]; then
        if [ ! -d "/lmdeploy_build" ]; then
            mkdir -p /lmdeploy_build
        fi
        cp "/tmpbuild/${new_base_name}.whl" "/lmdeploy_build/${new_base_name}.whl"
    fi
done
