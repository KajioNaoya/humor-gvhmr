# HuMoR Docker Environment for CPU
# Based on Ubuntu 20.04 with Python 3.8 and CPU-only PyTorch 1.6.0

FROM ubuntu:20.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Set working directory
WORKDIR /workspace

# Install system dependencies (Ubuntu 20.04 comes with Python 3.8 by default)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    git \
    wget \
    curl \
    unzip \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libosmesa6 \
    freeglut3-dev \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# OpenPose dependencies (OPTIONAL - uncomment to build OpenPose)
# ============================================================================
# Note: Building OpenPose takes 30-60+ minutes and requires significant resources
# Uncomment the following sections if you want to build OpenPose from source
# ============================================================================

# Install OpenPose build dependencies
# RUN apt-get update && apt-get install -y \
#     cmake \
#     libopencv-dev \
#     libatlas-base-dev \
#     libprotobuf-dev \
#     protobuf-compiler \
#     libgoogle-glog-dev \
#     libboost-all-dev \
#     libhdf5-dev \
#     libopenblas-dev \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

# Clone and build OpenPose (CPU-only version)
# Note: Models are downloaded separately to avoid build-time network issues
# RUN cd /workspace && \
#     git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git external/openpose && \
#     cd external/openpose && \
#     git submodule update --init --recursive && \
#     mkdir build && cd build && \
#     cmake .. \
#         -DGPU_MODE=CPU_ONLY \
#         -DBUILD_PYTHON=ON \
#         -DDOWNLOAD_BODY_25_MODEL=OFF \
#         -DDOWNLOAD_HAND_MODEL=OFF \
#         -DDOWNLOAD_FACE_MODEL=OFF && \
#     make -j$(nproc)

# Copy pre-downloaded OpenPose BODY_25 model files from host
# Place the following files in ./body_25/ on your host machine:
#   - pose_iter_584000.caffemodel
#   - pose_deploy.prototxt
# COPY ./body_25/ /workspace/external/openpose/models/pose/body_25/

# Add OpenPose Python bindings to PYTHONPATH
# ENV PYTHONPATH="${PYTHONPATH}:/workspace/external/openpose/build/python"

# ============================================================================
# End of OpenPose dependencies
# ============================================================================

# Set python3 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Install PyTorch 1.6.0 CPU version and torchvision 0.7.0
RUN pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Copy requirements file
COPY requirements.txt /workspace/requirements.txt

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables for rendering
ENV PYOPENGL_PLATFORM=osmesa
ENV MUJOCO_GL=osmesa

# Copy the project code
COPY . /workspace/humor

# Set the working directory to the project
WORKDIR /workspace/humor

# Create necessary directories for models and checkpoints
RUN mkdir -p /workspace/humor/body_models/smplh && \
    mkdir -p /workspace/humor/body_models/vposer_v1_0 && \
    mkdir -p /workspace/humor/checkpoints && \
    mkdir -p /workspace/humor/out && \
    mkdir -p /workspace/humor/external

# Set Python path
ENV PYTHONPATH=/workspace/humor:$PYTHONPATH

# Default command
CMD ["/bin/bash"]

