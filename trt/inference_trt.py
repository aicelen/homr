import sys
import numpy as np

import tensorrt as trt
from cuda.bindings import runtime as cudart

def _check(err):
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {err}")

def build_engine(onnx_path: str, engine_path: str) -> bytes:
    """Build a strongly typed TensorRT engine from an ONNX file and save it."""
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx_path):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError(f"Failed to parse {onnx_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GiB

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build failed")
    with open(engine_path, "wb") as f:
        f.write(serialized)
    return bytes(serialized)

def run_inference(engine_bytes: bytes, input_array: np.ndarray) -> np.ndarray:
    """Deserialize the engine and run a single inference on the supplied input."""
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    context = engine.create_execution_context()

    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    context.set_input_shape(input_name, input_array.shape)

    output_shape = tuple(context.get_tensor_shape(output_name))
    host_input = np.ascontiguousarray(input_array, dtype=np.float32)
    host_output = np.empty(output_shape, dtype=np.float32)

    err, d_input = cudart.cudaMalloc(host_input.nbytes); _check(err)
    err, d_output = cudart.cudaMalloc(host_output.nbytes); _check(err)
    err, stream = cudart.cudaStreamCreate(); _check(err)

    _check(cudart.cudaMemcpyAsync(
        d_input, host_input.ctypes.data, host_input.nbytes,
        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream,
    ))
    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))
    context.execute_async_v3(stream)
    _check(cudart.cudaMemcpyAsync(
        host_output.ctypes.data, d_output, host_output.nbytes,
        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream,
    ))
    _check(cudart.cudaStreamSynchronize(stream))

    cudart.cudaFree(d_input)
    cudart.cudaFree(d_output)
    cudart.cudaStreamDestroy(stream)
    return host_output

if __name__ == "__main__":
    onnx_path = sys.argv[1] if len(sys.argv) > 1 else "resnet50-v1-12.onnx"
    engine_path = "model.engine"
    engine_bytes = build_engine(onnx_path, engine_path)

    # Replace this with a preprocessed image batch in your own application.
    dummy_input = np.random.rand(1, 3, 224, 224).astype(np.float32)
    output = run_inference(engine_bytes, dummy_input)
    print("Output shape:", output.shape)
    print("Top-1 class index:", int(np.argmax(output[0])))