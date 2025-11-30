#!/bin/bash
# Quick test script for Jetson Nano ALPR
# Run this on the Jetson after Docker build completes

set -e

echo "======================================"
echo "Quick Test - Jetson ALPR TensorRT"
echo "======================================"
echo ""

# Test 1: Verify Docker image exists
echo "[1/3] Checking Docker image..."
if sudo docker images | grep -q "jetson-alpr-tensorrt"; then
    echo "✓ Docker image found"
else
    echo "✗ Docker image not found!"
    exit 1
fi
echo ""

# Test 2: Test with single sample image
echo "[2/3] Testing with sample image..."
sudo docker run --runtime nvidia --rm \
  -v /home/samples:/app/samples:ro \
  jetson-alpr-tensorrt:latest \
  python3 /app/jetson_alpr_tensorrt.py /app/samples/0_OUT.jpg
echo ""

# Test 3: Run benchmark
echo "[3/3] Running performance benchmark..."
sudo docker run --runtime nvidia --rm \
  -v /home/samples:/app/samples:ro \
  jetson-alpr-tensorrt:latest \
  python3 /app/benchmark_comparison.py
echo ""

echo "======================================"
echo "Quick test complete!"
echo "======================================"
