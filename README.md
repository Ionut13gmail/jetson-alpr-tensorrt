# Jetson Nano ALPR Optimization

High-performance Automatic License Plate Recognition (ALPR) for NVIDIA Jetson Nano using TensorRT optimization.

**Target Performance: 10-15 FPS** (vs 10 FPS baseline with ONNX Runtime)

## 🚀 Performance

| Component | ONNX Runtime | TensorRT .engine | Speedup |
|-----------|--------------|------------------|---------|
| Detector | ~60ms | ~35-40ms | **1.5-1.7x** |
| OCR | ~25ms | ~15-20ms | **1.3-1.5x** |
| **Total** | **~100ms (10 FPS)** | **~65-75ms (13-15 FPS)** | **1.3-1.5x** |

## 🎯 Features

- **TensorRT Optimization**: Direct .engine conversion for maximum performance
- **FP16 Precision**: 2x throughput with minimal accuracy loss
- **Docker Deployment**: Production-ready containerized solution
- **Multi-Jetson Ready**: Easy deployment across multiple devices
- **Memory Efficient**: Optimized for 2GB Jetson Nano
- **Maintained Accuracy**: Detection >84%, OCR >97%

## 📋 Requirements

- NVIDIA Jetson Nano
- JetPack 4.6+ (L4T R32.7.x)
- Docker with nvidia runtime
- 8GB+ SD card (16GB recommended)
- Sample images for testing

## 🔧 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/jetson-alpr-optimized.git
cd jetson-alpr-optimized
```

### 2. Build Docker Image

**Note**: This takes 10-15 minutes (includes TensorRT conversion)

```bash
sudo docker build -f Dockerfile.tensorrt -t jetson-alpr-tensorrt:latest .
```

### 3. Run Benchmark

```bash
sudo docker run --runtime nvidia --rm \
  -v /path/to/samples:/app/samples:ro \
  jetson-alpr-tensorrt:latest \
  python3 /app/benchmark_comparison.py
```

Expected output:
```
ONNX Runtime:  ~100ms (~10 FPS)
TensorRT:      ~70ms  (~14 FPS)
✓ TARGET ACHIEVED (10-15 FPS)
```

### 4. Process Images

```bash
# Single image
sudo docker run --runtime nvidia --rm \
  -v /path/to/samples:/app/samples:ro \
  jetson-alpr-tensorrt:latest \
  python3 /app/jetson_alpr_tensorrt.py /app/samples/image.jpg

# Directory of images
sudo docker run --runtime nvidia --rm \
  -v /path/to/samples:/app/samples:ro \
  -v $(pwd)/output:/app/output \
  jetson-alpr-tensorrt:latest \
  python3 /app/jetson_alpr_tensorrt.py /app/samples/
```

## 📦 Multi-Jetson Deployment

### Save Image Once

```bash
# On first Jetson after building
sudo docker save jetson-alpr-tensorrt:latest | gzip > jetson-alpr-tensorrt.tar.gz
```

### Deploy to Other Jetsons

```bash
# Transfer
scp jetson-alpr-tensorrt.tar.gz user@jetson-ip:/tmp/

# Load on each Jetson
ssh user@jetson-ip 'sudo docker load < /tmp/jetson-alpr-tensorrt.tar.gz'
```

**Important**: TensorRT engines are hardware-specific. Build on Jetson Nano, not x86!

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│  Input Image (e.g., 1920x1080)          │
└──────────────┬──────────────────────────┘
               │
               ↓
┌─────────────────────────────────────────┐
│  Preprocessing & Letterbox Resize       │
│  → 384x384 (detector input)             │
└──────────────┬──────────────────────────┘
               │
               ↓
┌─────────────────────────────────────────┐
│  TensorRT Detector Engine (FP16)        │
│  YOLOv9-Tiny @ ~35-40ms                 │
│  → Bounding boxes + confidence          │
└──────────────┬──────────────────────────┘
               │
               ↓
┌─────────────────────────────────────────┐
│  Extract Plate ROIs                     │
│  → Resize to 140x70 (OCR input)         │
└──────────────┬──────────────────────────┘
               │
               ↓
┌─────────────────────────────────────────┐
│  TensorRT OCR Engine (FP16)             │
│  MobileViT-v2 @ ~15-20ms                │
│  → Character recognition                │
└──────────────┬──────────────────────────┘
               │
               ↓
┌─────────────────────────────────────────┐
│  Results: Plates + Text + Confidence    │
└─────────────────────────────────────────┘
```

## 📁 Project Structure

```
jetson-alpr-optimized/
├── README.md                      # This file
├── DEPLOYMENT_README.md           # Detailed deployment guide
├── OPTIMIZATION_SUMMARY.md        # Technical optimization details
├── Dockerfile.tensorrt            # Optimized Dockerfile
├── docker-compose.tensorrt.yml    # Docker Compose config
├── convert_to_tensorrt.py         # ONNX → TensorRT converter
├── jetson_alpr_tensorrt.py        # TensorRT inference engine
├── benchmark_comparison.py        # Performance comparison tool
├── quick_test.sh                  # Quick test script
└── .gitignore
```

## 🔬 Optimization Techniques

1. **TensorRT Engine Conversion**
   - ONNX → .engine during Docker build
   - FP16 precision (2x faster)
   - Layer fusion & dead code elimination
   - Optimized memory allocation

2. **Inference Pipeline**
   - Direct TensorRT Runtime (no ONNX overhead)
   - PyCUDA zero-copy memory transfers
   - CUDA stream async execution
   - Minimal preprocessing

3. **Memory Management**
   - Conservative 256MB workspace
   - Optimized buffer allocation
   - GPU memory pooling
   - RAM: ~850MB total usage

## 🎛️ Configuration

### Adjust Detection Confidence

Edit detector threshold for speed/accuracy trade-off:

```python
# In jetson_alpr_tensorrt.py
alpr = FastALPR(detector_engine, ocr_engine, conf_thresh=0.5)  # default: 0.4
```

Higher threshold = faster (fewer OCR calls), but may miss some plates.

### Reduce Input Resolution

Trade accuracy for speed by reducing detector input size:

```python
# In jetson_alpr_tensorrt.py LicensePlateDetectorTRT.__init__
self.input_size = (320, 320)  # default: (384, 384)
```

## 📊 Benchmarking

Run comprehensive benchmarks:

```bash
sudo docker run --runtime nvidia --rm \
  -v /path/to/samples:/app/samples:ro \
  jetson-alpr-tensorrt:latest \
  python3 /app/benchmark_comparison.py
```

Monitor GPU/CPU/Memory during inference:

```bash
sudo tegrastats
```

Expected stats:
```
RAM: 850/1972MB  CPU: [40%,35%,45%,40%]  GPU: 80%@921MHz  Temp: 42C
```

## 🐛 Troubleshooting

### "Illegal instruction" error
- Cause: Running outside Docker or wrong ONNX Runtime
- Fix: Always use `sudo docker run --runtime nvidia`

### TensorRT engines not found
- Cause: Conversion failed during build
- Fix: Check build logs, increase swap if OOM

### Low FPS (<10)
- Check GPU usage: `sudo tegrastats`
- Ensure `--runtime nvidia` is used
- Verify CUDA is available: `nvidia-smi` (inside container)

### Out of memory during build
- Increase swap space:
  ```bash
  sudo fallocate -l 4G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  ```

## 📚 Documentation

- [DEPLOYMENT_README.md](DEPLOYMENT_README.md) - Complete deployment guide
- [OPTIMIZATION_SUMMARY.md](OPTIMIZATION_SUMMARY.md) - Technical deep dive

## 🙏 Credits

Based on:
- [jetson-alpr](https://github.com/Ionut13gmail/jetson-alpr) - Original implementation
- NVIDIA TensorRT
- dustynv/l4t-ml Docker images

## 📄 License

MIT License - see LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Test on real Jetson Nano hardware
4. Submit pull request

## 📞 Support

- GitHub Issues: Report bugs and request features
- Discussions: Share experiences and ask questions

## 🔮 Roadmap

- [ ] INT8 quantization support (2-4x speedup)
- [ ] Multi-stream video processing
- [ ] REST API integration
- [ ] Cloud deployment guide
- [ ] Performance monitoring dashboard
