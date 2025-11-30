# Jetson Nano ALPR Optimization Summary

## Objective
Achieve **10-15 FPS** license plate detection and OCR on Jetson Nano

## Current Architecture Analysis

### Original Implementation (jetson-alpr repository)
- **Detection**: YOLOv9-Tiny (7.5MB ONNX model, 384x384 input)
- **OCR**: MobileViT v2 (4.8MB ONNX model, 140x70 input)
- **Runtime**: ONNX Runtime 1.10 with TensorrtExecutionProvider
- **Performance**: ~10 FPS (~100ms per frame)
- **Limitations**: ONNX Runtime overhead, suboptimal for Jetson

### Jetson Nano Specifications
- **GPU**: 128-core Maxwell GPU @ 921MHz
- **RAM**: 2GB (shared with GPU)
- **CPU**: Quad-core ARM A57 @ 1.43GHz
- **JetPack**: R32.7.6 (L4T)
- **CUDA**: 10.2
- **TensorRT**: 8.2.1.8

## Optimization Strategy

### 1. Direct TensorRT Engine Conversion
**What**: Convert ONNX models to TensorRT `.engine` files

**Why**:
- Eliminates ONNX Runtime overhead
- Hardware-optimized execution graphs
- Kernel fusion and layer optimization
- Direct CUDA memory management

**Expected Speedup**: 1.3-1.7x (30-40% faster)

**Implementation**:
- FP16 precision (2x throughput vs FP32)
- Dynamic tensor memory allocation
- Layer fusion and dead code elimination
- Custom PyCUDA bindings

### 2. Optimized Inference Pipeline
**Changes**:
- Direct TensorRT Runtime (no ONNX Runtime)
- PyCUDA for zero-copy memory transfer
- Async execution with CUDA streams
- Minimal preprocessing overhead

**Code**: `jetson_alpr_tensorrt.py`

### 3. Docker-Based Deployment
**Benefits**:
- Consistent environment across Jetsons
- Pre-built TensorRT engines (no runtime conversion)
- Easy multi-device deployment
- Isolated dependencies

## Files Created

### Core Implementation
1. **`convert_to_tensorrt.py`** (5.7KB)
   - Converts ONNX → TensorRT .engine
   - FP16 optimization
   - Batch size = 1 (Jetson Nano)
   - Workspace memory: 256MB

2. **`jetson_alpr_tensorrt.py`** (13KB)
   - Pure TensorRT inference
   - PyCUDA memory management
   - Optimized preprocessing
   - Target: 10-15 FPS

3. **`benchmark_comparison.py`** (7.7KB)
   - Compare ONNX vs TensorRT
   - Multiple image testing
   - Performance statistics

### Deployment
4. **`Dockerfile.tensorrt`** (1.2KB)
   - Based on dustynv/l4t-ml:r32.7.1
   - Converts models during build
   - PyCUDA integration

5. **`docker-compose.tensorrt.yml`** (697B)
   - Service configuration
   - Volume mounts for samples
   - Runtime: nvidia

6. **`DEPLOYMENT_README.md`** (6KB)
   - Complete deployment guide
   - Troubleshooting tips
   - Multi-Jetson deployment

## Expected Performance Improvements

| Metric | ONNX Runtime | TensorRT .engine | Improvement |
|--------|--------------|------------------|-------------|
| Detector inference | ~60ms | ~35-40ms | 1.5-1.7x |
| OCR inference | ~25ms | ~15-20ms | 1.3-1.5x |
| **Total pipeline** | **~100ms (10 FPS)** | **~65-75ms (13-15 FPS)** | **1.3-1.5x** |

## Memory Optimization

### TensorRT Engine Sizes
- Detector: ~8-10 MB (vs 7.5MB ONNX)
- OCR: ~5-6 MB (vs 4.8MB ONNX)
- **Total**: ~15 MB (similar to ONNX)

### Runtime Memory
- GPU Memory: ~400MB (detector) + ~150MB (OCR) = ~550MB
- RAM: ~300MB Python + models
- **Total**: ~850MB (leaves 1.1GB free on 2GB Nano)

## Deployment Process

### One-Time Setup (First Jetson)
```bash
# 1. Upload files
scp files... john@10.8.1.2:/home/john/jetson-alpr/

# 2. Build Docker image (10-15 min)
sudo docker build -f Dockerfile.tensorrt -t jetson-alpr-tensorrt:latest .

# 3. Run benchmark
sudo docker run --runtime nvidia --rm \
  -v /home/samples:/app/samples:ro \
  jetson-alpr-tensorrt:latest \
  python3 /app/benchmark_comparison.py
```

### Multi-Jetson Deployment
```bash
# Save image
sudo docker save jetson-alpr-tensorrt:latest | gzip > image.tar.gz

# Transfer and load on each Jetson
scp image.tar.gz john@<jetson-ip>:/tmp/
ssh john@<jetson-ip> 'sudo docker load < /tmp/image.tar.gz'
```

## Alternative Optimizations (If Needed)

If 10-15 FPS is not achieved, additional optimizations:

### 1. Reduce Detector Resolution
```python
# 384x384 → 320x320 (1.4x faster, slight accuracy loss)
self.input_size = (320, 320)
```

### 2. INT8 Quantization
- Requires calibration dataset
- 2-4x speedup vs FP16
- More complex conversion

### 3. Frame Skipping
```python
# Process every 2nd frame for video
if frame_count % 2 == 0:
    results = alpr.predict(frame)
```

### 4. ROI Optimization
- Only run OCR on high-confidence detections
- Adjust confidence threshold: 0.4 → 0.5

## Verification Checklist

- [ ] Docker image builds successfully
- [ ] TensorRT engines created (detector_fp16.engine, ocr_fp16.engine)
- [ ] Benchmark shows 10-15 FPS on sample images
- [ ] Detection accuracy maintained (>80%)
- [ ] OCR accuracy maintained (>95%)
- [ ] Memory usage < 1.5GB
- [ ] GPU utilization 70-90%

## Known Issues & Solutions

### Issue: "Illegal instruction" error
**Solution**: Always use Docker with `--runtime nvidia`

### Issue: TensorRT conversion fails
**Solution**: Increase swap space to 4GB

### Issue: GPU not detected
**Solution**: Check `sudo tegrastats` shows GPU usage

## Production Recommendations

1. **Monitoring**
   - Use `tegrastats` to monitor GPU/RAM
   - Set up watchdog for automatic restart
   - Log detection/OCR confidences

2. **Scaling**
   - Load balance across multiple Jetsons
   - Use message queue (Redis/RabbitMQ)
   - Implement result caching

3. **Maintenance**
   - Weekly log rotation
   - Monthly accuracy audits
   - Quarterly model updates

## References

- Original repo: https://github.com/Ionut13gmail/jetson-alpr
- TensorRT docs: https://docs.nvidia.com/deeplearning/tensorrt/
- Jetson performance: https://developer.nvidia.com/embedded/jetson-nano
