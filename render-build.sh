#!/usr/bin/env bash
# render-build.sh — Render build script
# Installs system-level OpenGL/EGL libraries required by MediaPipe,
# then installs Python dependencies.

set -o errexit

echo ">>> Installing system dependencies (OpenGL/EGL for MediaPipe)..."
apt-get update -qq && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libgles2-mesa \
    libegl1-mesa \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

echo ">>> Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ">>> Build complete!"
