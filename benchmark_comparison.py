#!/usr/bin/env python3
"""
Benchmark comparison: ONNX Runtime vs TensorRT .engine
Run this inside the Docker container to compare performance
"""
import os
import sys
import time
import glob
import numpy as np
import cv2


def benchmark_onnx(image_paths, num_runs=10):
    """Benchmark ONNX Runtime implementation"""
    print("\n" + "="*70)
    print("BENCHMARKING: ONNX Runtime (Original)")
    print("="*70)

    try:
        import onnxruntime as ort
        sys.path.insert(0, '/app')
        from jetson_alpr import JetsonALPR

        # Check providers
        providers = ort.get_available_providers()
        print(f"Available providers: {providers}")

        # Initialize
        use_gpu = 'TensorrtExecutionProvider' in providers or 'CUDAExecutionProvider' in providers
        alpr = JetsonALPR(use_gpu=use_gpu, conf_thresh=0.4)

        results = run_benchmark(alpr, image_paths, num_runs, "ONNX")
        return results

    except Exception as e:
        print(f"ERROR: ONNX benchmark failed: {e}")
        return None


def benchmark_tensorrt(image_paths, num_runs=10):
    """Benchmark TensorRT .engine implementation"""
    print("\n" + "="*70)
    print("BENCHMARKING: TensorRT .engine (Optimized)")
    print("="*70)

    try:
        sys.path.insert(0, '/app')
        from jetson_alpr_tensorrt import FastALPR

        # Check engines exist
        detector_engine = "/app/models/detector_fp16.engine"
        ocr_engine = "/app/models/ocr_fp16.engine"

        if not os.path.exists(detector_engine) or not os.path.exists(ocr_engine):
            print("ERROR: TensorRT engines not found!")
            print(f"  Detector: {os.path.exists(detector_engine)}")
            print(f"  OCR: {os.path.exists(ocr_engine)}")
            return None

        # Initialize
        alpr = FastALPR(detector_engine, ocr_engine, conf_thresh=0.4)

        results = run_benchmark(alpr, image_paths, num_runs, "TensorRT")
        return results

    except Exception as e:
        print(f"ERROR: TensorRT benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_benchmark(alpr, image_paths, num_runs, label):
    """Run benchmark on a set of images"""
    all_times = []
    total_detections = 0

    print(f"\nTesting {len(image_paths)} images, {num_runs} runs each")
    print()

    # Warmup
    print("Warming up...")
    test_img = cv2.imread(image_paths[0])
    for _ in range(3):
        alpr.predict(test_img)

    # Benchmark each image
    for img_path in image_paths[:5]:  # Limit to first 5 images
        img = cv2.imread(img_path)
        if img is None:
            continue

        print(f"Testing: {os.path.basename(img_path)} ({img.shape[1]}x{img.shape[0]})")

        times = []
        results = None

        for _ in range(num_runs):
            start = time.time()
            results = alpr.predict(img)
            elapsed = time.time() - start
            times.append(elapsed)

        all_times.extend(times)

        # Show detections from last run
        if results:
            total_detections += len(results)
            for r in results:
                text = r.get('text', '')
                det_conf = r.get('det_confidence', 0.0)
                ocr_conf = r.get('ocr_confidence', 0.0)
                print(f"  → '{text}' (det: {det_conf:.2f}, ocr: {ocr_conf:.2f})")

        avg = np.mean(times) * 1000
        print(f"  Avg: {avg:.1f} ms ({1000/avg:.1f} FPS)")

    # Overall statistics
    times_array = np.array(all_times)
    avg_time = np.mean(times_array)
    min_time = np.min(times_array)
    max_time = np.max(times_array)
    p50_time = np.percentile(times_array, 50)
    p95_time = np.percentile(times_array, 95)
    std_time = np.std(times_array)

    print(f"\n{'='*70}")
    print(f"{label} - OVERALL RESULTS")
    print(f"{'='*70}")
    print(f"Total runs:      {len(times_array)}")
    print(f"Total plates:    {total_detections}")
    print(f"Average:         {avg_time*1000:6.1f} ms  ({1.0/avg_time:5.1f} FPS)")
    print(f"Median (p50):    {p50_time*1000:6.1f} ms  ({1.0/p50_time:5.1f} FPS)")
    print(f"95th percentile: {p95_time*1000:6.1f} ms  ({1.0/p95_time:5.1f} FPS)")
    print(f"Min:             {min_time*1000:6.1f} ms  ({1.0/min_time:5.1f} FPS)")
    print(f"Max:             {max_time*1000:6.1f} ms  ({1.0/max_time:5.1f} FPS)")
    print(f"Std Dev:         {std_time*1000:6.1f} ms")
    print(f"{'='*70}\n")

    return {
        'label': label,
        'avg_ms': avg_time * 1000,
        'avg_fps': 1.0 / avg_time,
        'median_ms': p50_time * 1000,
        'median_fps': 1.0 / p50_time,
        'min_ms': min_time * 1000,
        'max_ms': max_time * 1000,
        'std_ms': std_time * 1000,
        'total_detections': total_detections
    }


def print_comparison(onnx_results, trt_results):
    """Print comparison table"""
    if not onnx_results or not trt_results:
        return

    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)
    print()

    speedup = onnx_results['avg_ms'] / trt_results['avg_ms']
    fps_improvement = ((trt_results['avg_fps'] - onnx_results['avg_fps']) / onnx_results['avg_fps']) * 100

    print(f"{'Metric':<20} {'ONNX Runtime':<20} {'TensorRT':< 20} {'Improvement':<15}")
    print("-" * 70)
    print(f"{'Average Time':<20} {onnx_results['avg_ms']:>6.1f} ms         {trt_results['avg_ms']:>6.1f} ms         {speedup:.2f}x faster")
    print(f"{'Average FPS':<20} {onnx_results['avg_fps']:>6.1f} FPS        {trt_results['avg_fps']:>6.1f} FPS        +{fps_improvement:.1f}%")
    print(f"{'Median Time':<20} {onnx_results['median_ms']:>6.1f} ms         {trt_results['median_ms']:>6.1f} ms")
    print(f"{'Min Time':<20} {onnx_results['min_ms']:>6.1f} ms         {trt_results['min_ms']:>6.1f} ms")
    print(f"{'Max Time':<20} {onnx_results['max_ms']:>6.1f} ms         {trt_results['max_ms']:>6.1f} ms")
    print(f"{'Std Dev':<20} {onnx_results['std_ms']:>6.1f} ms         {trt_results['std_ms']:>6.1f} ms")
    print("-" * 70)
    print()

    if trt_results['avg_fps'] >= 10:
        status = "✓ TARGET ACHIEVED (10-15 FPS)"
    else:
        status = "⚠ Below target (need optimization)"

    print(f"Status: {status}")
    print("="*70)
    print()


def main():
    """Main benchmark comparison"""
    print("="*70)
    print("ALPR Performance Comparison: ONNX vs TensorRT")
    print("="*70)

    # Find sample images
    sample_dirs = ['/home/samples', '/app/samples', './samples']
    image_paths = []

    for d in sample_dirs:
        if os.path.isdir(d):
            image_paths = glob.glob(os.path.join(d, '*.jpg'))
            image_paths.extend(glob.glob(os.path.join(d, '*.png')))
            if image_paths:
                print(f"Found {len(image_paths)} images in {d}")
                break

    if not image_paths:
        print("ERROR: No sample images found!")
        print("Checked directories:", sample_dirs)
        return 1

    # Limit to reasonable number
    image_paths = sorted(image_paths)[:10]
    num_runs = 10

    # Run benchmarks
    onnx_results = benchmark_onnx(image_paths, num_runs)
    trt_results = benchmark_tensorrt(image_paths, num_runs)

    # Compare
    if onnx_results and trt_results:
        print_comparison(onnx_results, trt_results)
    elif trt_results:
        print("\n✓ TensorRT benchmark completed")
        print(f"  Average: {trt_results['avg_fps']:.1f} FPS")
    elif onnx_results:
        print("\n✓ ONNX Runtime benchmark completed")
        print(f"  Average: {onnx_results['avg_fps']:.1f} FPS")
    else:
        print("\n✗ Both benchmarks failed!")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
