#!/usr/bin/env python3
"""
Optimized License Plate Recognition using TensorRT .engine files
Target: 10-15 FPS on Jetson Nano
"""
import os
import sys
import time
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# OCR vocabulary
OCR_VOCABULARY = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
MAX_PLATE_SLOTS = 7


class TensorRTInference:
    """Base class for TensorRT inference"""

    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        self._load_engine()
        self._allocate_buffers()

    def _load_engine(self):
        """Load TensorRT engine from file"""
        print(f"Loading TensorRT engine: {self.engine_path}")

        runtime = trt.Runtime(TRT_LOGGER)

        with open(self.engine_path, 'rb') as f:
            engine_data = f.read()

        self.engine = runtime.deserialize_cuda_engine(engine_data)

        if self.engine is None:
            raise RuntimeError(f"Failed to load engine from {self.engine_path}")

        self.context = self.engine.create_execution_context()
        print(f"Engine loaded successfully!")

    def _allocate_buffers(self):
        """Allocate buffers for input and output"""
        self.inputs = []
        self.outputs = []
        self.bindings = []

        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            shape = self.engine.get_binding_shape(i)
            is_input = self.engine.binding_is_input(i)

            # Calculate size
            size = trt.volume(shape) * self.engine.max_batch_size
            if size < 0:
                # Dynamic shape - use default
                size = abs(trt.volume(shape))

            # Allocate memory
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))

            if is_input:
                self.inputs.append({
                    'name': name,
                    'host': host_mem,
                    'device': device_mem,
                    'shape': shape,
                    'dtype': dtype
                })
                print(f"  Input: {name}, shape={shape}, dtype={dtype}")
            else:
                self.outputs.append({
                    'name': name,
                    'host': host_mem,
                    'device': device_mem,
                    'shape': shape,
                    'dtype': dtype
                })
                print(f"  Output: {name}, shape={shape}, dtype={dtype}")

    def infer(self, input_data):
        """Run inference"""
        # Copy input to device
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)

        # Run inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        # Copy output to host
        output_data = []
        for output in self.outputs:
            cuda.memcpy_dtoh_async(output['host'], output['device'], self.stream)
            output_data.append(output['host'].copy())

        # Synchronize
        self.stream.synchronize()

        return output_data


class LicensePlateDetectorTRT:
    """License plate detector using TensorRT"""

    def __init__(self, engine_path, conf_thresh=0.4, input_size=(384, 384)):
        self.conf_thresh = conf_thresh
        self.input_size = input_size
        self.trt_infer = TensorRTInference(engine_path)

    def letterbox_resize(self, image):
        """Resize image with letterbox padding"""
        h, w = image.shape[:2]
        target_h, target_w = self.input_size

        # Calculate scale
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create padded image
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)

        # Calculate padding offsets
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2

        # Place resized image
        padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized

        return padded, scale, pad_x, pad_y

    def preprocess(self, image):
        """Preprocess image for detector"""
        padded, scale, pad_x, pad_y = self.letterbox_resize(image)

        # Convert BGR to RGB and normalize
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0

        # NCHW format
        input_tensor = np.transpose(normalized, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)

        return input_tensor, scale, pad_x, pad_y

    def postprocess(self, outputs, orig_shape, scale, pad_x, pad_y):
        """Postprocess detector outputs"""
        detections = []
        h, w = orig_shape[:2]

        # Handle different output formats
        if len(outputs) >= 4:
            # Format: num_dets, boxes, scores, labels
            num_dets = int(outputs[0][0]) if outputs[0].size > 0 else 0
            boxes = outputs[1].reshape(-1, 4) if len(outputs[1].shape) > 1 else outputs[1]
            scores = outputs[2].flatten() if len(outputs[2].shape) > 0 else outputs[2]

            for i in range(min(num_dets, len(boxes))):
                conf = float(scores[i])
                if conf < self.conf_thresh:
                    continue

                # Get box coordinates
                x1, y1, x2, y2 = boxes[i]

                # Convert from padded coordinates to original
                x1 = (x1 - pad_x) / scale
                y1 = (y1 - pad_y) / scale
                x2 = (x2 - pad_x) / scale
                y2 = (y2 - pad_y) / scale

                # Clip to image bounds
                x1 = max(0, min(w, x1))
                y1 = max(0, min(h, y1))
                x2 = max(0, min(w, x2))
                y2 = max(0, min(h, y2))

                if x2 > x1 and y2 > y1:
                    detections.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': conf
                    })

        return detections

    def detect(self, image):
        """Detect license plates in image"""
        # Preprocess
        input_tensor, scale, pad_x, pad_y = self.preprocess(image)

        # Inference
        outputs = self.trt_infer.infer(input_tensor)

        # Postprocess
        detections = self.postprocess(outputs, image.shape, scale, pad_x, pad_y)

        return detections


class LicensePlateOCRTRT:
    """License plate OCR using TensorRT"""

    def __init__(self, engine_path, input_size=(140, 70)):
        self.input_size = input_size  # (width, height)
        self.trt_infer = TensorRTInference(engine_path)

    def preprocess(self, plate_image):
        """Preprocess plate image for OCR"""
        # Convert to grayscale if needed
        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image

        # Resize to model input size
        resized = cv2.resize(gray, self.input_size, interpolation=cv2.INTER_LINEAR)

        # Normalize to [-1, 1]
        normalized = (resized.astype(np.float32) / 127.5) - 1.0

        # Add batch and channel dimensions: (1, 1, H, W)
        input_tensor = np.expand_dims(normalized, axis=(0, 1)).astype(np.float32)

        return input_tensor

    def decode_prediction(self, prediction):
        """Decode OCR prediction to text"""
        # prediction shape: (1, seq_len, vocab_size)
        pred = prediction.reshape(-1, len(OCR_VOCABULARY))

        # Get most likely character at each position
        indices = np.argmax(pred, axis=1)

        # Decode to text
        text = ""
        for idx in indices:
            if idx < len(OCR_VOCABULARY):
                char = OCR_VOCABULARY[idx]
                if char != '_':  # Skip padding character
                    text += char

        return text.strip()

    def recognize(self, plate_image):
        """Recognize text from plate image"""
        if plate_image is None or plate_image.size == 0:
            return "", 0.0

        # Preprocess
        input_tensor = self.preprocess(plate_image)

        # Inference
        outputs = self.trt_infer.infer(input_tensor)

        # Decode
        text = self.decode_prediction(outputs[0])

        # Calculate confidence (simplified)
        confidence = 0.95 if text else 0.0

        return text, confidence


class FastALPR:
    """Fast ALPR system using TensorRT"""

    def __init__(self, detector_engine, ocr_engine, conf_thresh=0.4):
        print("Initializing Fast ALPR with TensorRT...")
        self.detector = LicensePlateDetectorTRT(detector_engine, conf_thresh)
        self.ocr = LicensePlateOCRTRT(ocr_engine)
        print("Fast ALPR initialized!")

    def predict(self, image):
        """Detect and recognize license plates"""
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                return []

        # Detect plates
        detections = self.detector.detect(image)

        # Recognize text for each detection
        results = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            plate_roi = image[y1:y2, x1:x2]

            if plate_roi.size == 0:
                continue

            # OCR
            text, ocr_conf = self.ocr.recognize(plate_roi)

            results.append({
                'bbox': det['bbox'],
                'text': text,
                'det_confidence': det['confidence'],
                'ocr_confidence': ocr_conf
            })

        return results


def benchmark(alpr, image_path, num_runs=20):
    """Benchmark ALPR performance"""
    print(f"\n{'='*70}")
    print("BENCHMARKING")
    print(f"{'='*70}")

    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not read image: {image_path}")
        return None

    print(f"Image: {image_path}")
    print(f"Resolution: {image.shape[1]}x{image.shape[0]}")
    print(f"Runs: {num_runs}")
    print()

    # Warmup
    print("Warming up (3 runs)...")
    for _ in range(3):
        alpr.predict(image)

    # Benchmark
    print(f"Running benchmark ({num_runs} iterations)...")
    times = []

    for i in range(num_runs):
        start = time.time()
        results = alpr.predict(image)
        elapsed = time.time() - start
        times.append(elapsed)

        if i == 0:
            # Show results from first run
            print(f"\nDetection results:")
            for r in results:
                print(f"  Plate: '{r['text']}' (det: {r['det_confidence']:.2f}, ocr: {r['ocr_confidence']:.2f})")

    # Statistics
    times = np.array(times)
    avg_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)
    std_time = np.std(times)
    fps = 1.0 / avg_time

    print(f"\n{'='*70}")
    print("BENCHMARK RESULTS")
    print(f"{'='*70}")
    print(f"Average:  {avg_time*1000:6.1f} ms  ({fps:5.1f} FPS)")
    print(f"Min:      {min_time*1000:6.1f} ms  ({1.0/min_time:5.1f} FPS)")
    print(f"Max:      {max_time*1000:6.1f} ms  ({1.0/max_time:5.1f} FPS)")
    print(f"Std Dev:  {std_time*1000:6.1f} ms")
    print(f"{'='*70}\n")

    return results


def main():
    print("=" * 70)
    print("Fast ALPR for Jetson Nano - TensorRT Optimized")
    print("=" * 70)
    print()

    # Model paths
    detector_engine = "/app/models/detector_fp16.engine"
    ocr_engine = "/app/models/ocr_fp16.engine"

    # Check if engines exist
    if not os.path.exists(detector_engine):
        print(f"ERROR: Detector engine not found: {detector_engine}")
        print("Please run convert_to_tensorrt.py first!")
        return 1

    if not os.path.exists(ocr_engine):
        print(f"ERROR: OCR engine not found: {ocr_engine}")
        print("Please run convert_to_tensorrt.py first!")
        return 1

    # Initialize ALPR
    alpr = FastALPR(detector_engine, ocr_engine, conf_thresh=0.4)

    # Test with sample image
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    else:
        # Default test images
        test_images = [
            "/app/assets/test_image.png",
            "/home/samples/0_OUT.jpg",
            "test.jpg"
        ]
        test_image = None
        for img in test_images:
            if os.path.exists(img):
                test_image = img
                break

    if test_image and os.path.exists(test_image):
        results = benchmark(alpr, test_image, num_runs=20)
    else:
        print("No test image found. Usage:")
        print(f"  python3 {sys.argv[0]} <image_path>")

    return 0


if __name__ == '__main__':
    sys.exit(main())
