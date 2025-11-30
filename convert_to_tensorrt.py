#!/usr/bin/env python3
"""
Convert ONNX models to TensorRT .engine files for maximum performance on Jetson Nano
This script should be run inside the Docker container or on the Jetson directly
"""
import os
import sys
import tensorrt as trt
import numpy as np

# Logger for TensorRT
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def build_engine(onnx_file_path, engine_file_path, fp16_mode=True, max_batch_size=1):
    """
    Convert ONNX model to TensorRT engine

    Args:
        onnx_file_path: Path to ONNX model
        engine_file_path: Path to save TensorRT engine
        fp16_mode: Use FP16 precision (recommended for Jetson Nano)
        max_batch_size: Maximum batch size
    """
    print(f"Converting {onnx_file_path} to TensorRT engine...")
    print(f"FP16 mode: {fp16_mode}, Max batch size: {max_batch_size}")

    # Create builder
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX model
    print("Parsing ONNX model...")
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    print(f"Network inputs: {network.num_inputs}")
    print(f"Network outputs: {network.num_outputs}")

    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f"  Input {i}: {inp.name}, shape={inp.shape}, dtype={inp.dtype}")

    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f"  Output {i}: {out.name}, shape={out.shape}, dtype={out.dtype}")

    # Build engine configuration
    config = builder.create_builder_config()

    # Set memory pool limit (important for Jetson Nano with limited memory)
    # Jetson Nano has 2GB RAM, be conservative
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)  # 256MB

    # Enable FP16 mode if supported and requested
    if fp16_mode and builder.platform_has_fast_fp16:
        print("Enabling FP16 mode for faster inference")
        config.set_flag(trt.BuilderFlag.FP16)

    # Enable TF32 (Tensor Float 32) if available
    if builder.platform_has_tf32:
        print("Enabling TF32 mode")
        config.set_flag(trt.BuilderFlag.TF32)

    # Build engine
    print("Building TensorRT engine... (this may take several minutes)")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("ERROR: Failed to build engine")
        return None

    # Save engine to file
    print(f"Saving engine to {engine_file_path}")
    with open(engine_file_path, 'wb') as f:
        f.write(serialized_engine)

    # Get file size
    size_mb = os.path.getsize(engine_file_path) / (1024 * 1024)
    print(f"Engine saved successfully! Size: {size_mb:.2f} MB")

    return engine_file_path


def verify_engine(engine_file_path):
    """Verify that the engine can be loaded"""
    print(f"\nVerifying engine: {engine_file_path}")

    runtime = trt.Runtime(TRT_LOGGER)

    with open(engine_file_path, 'rb') as f:
        engine_data = f.read()

    engine = runtime.deserialize_cuda_engine(engine_data)

    if engine is None:
        print("ERROR: Failed to load engine")
        return False

    print(f"Engine loaded successfully!")
    print(f"  Bindings: {engine.num_bindings}")

    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        shape = engine.get_binding_shape(i)
        dtype = engine.get_binding_dtype(i)
        is_input = engine.binding_is_input(i)
        print(f"  [{i}] {name}: shape={shape}, dtype={dtype}, is_input={is_input}")

    return True


def main():
    """Convert models from ONNX to TensorRT"""
    models_dir = "/app/models"

    # Models to convert
    models = [
        {
            'name': 'detector',
            'onnx': os.path.join(models_dir, 'detector_opset15.onnx'),
            'engine': os.path.join(models_dir, 'detector_fp16.engine'),
            'fp16': True
        },
        {
            'name': 'ocr',
            'onnx': os.path.join(models_dir, 'ocr_opset15.onnx'),
            'engine': os.path.join(models_dir, 'ocr_fp16.engine'),
            'fp16': True
        }
    ]

    print("=" * 70)
    print("TensorRT Model Conversion for Jetson Nano ALPR")
    print("=" * 70)
    print()

    # Check TensorRT version
    print(f"TensorRT version: {trt.__version__}")
    print()

    success_count = 0

    for model_info in models:
        print("-" * 70)
        print(f"Processing {model_info['name']} model")
        print("-" * 70)

        if not os.path.exists(model_info['onnx']):
            print(f"ERROR: ONNX file not found: {model_info['onnx']}")
            continue

        # Convert
        engine_path = build_engine(
            model_info['onnx'],
            model_info['engine'],
            fp16_mode=model_info['fp16']
        )

        if engine_path:
            # Verify
            if verify_engine(engine_path):
                success_count += 1
                print(f"✓ {model_info['name']} conversion successful!")
            else:
                print(f"✗ {model_info['name']} verification failed!")
        else:
            print(f"✗ {model_info['name']} conversion failed!")

        print()

    print("=" * 70)
    print(f"Conversion complete: {success_count}/{len(models)} models converted successfully")
    print("=" * 70)

    return success_count == len(models)


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
