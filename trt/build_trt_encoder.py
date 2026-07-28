import tensorrt as trt
from homr.transformer.configs import default_config
from cuda.bindings import runtime as cudart
import numpy as np
from time import perf_counter


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine_from_onnx(onnx_file_path, engine_file_path=None,
                            fp16_mode=False, max_workspace_size=1 << 30):
    """
    Parses an ONNX model and builds a serialized TensorRT engine.
    """
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Read and parse the ONNX file
    with open(onnx_file_path, "rb") as model_file:
        if not parser.parse(model_file.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    # Builder configuration
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)

    if fp16_mode:
        config.set_flag(trt.BuilderFlag.FP16)

    input_tensor = network.get_input(0)
    
    profile = builder.create_optimization_profile()

    profile.set_shape(input_tensor.name,
                        min=(1, 1,256,1280),
                        opt=(8, 1,256,1280),
                        max=(16, 1,256,1280))
    config.add_optimization_profile(profile)

    # Build the serialized engine
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("ERROR: Failed to build the engine.")
        return None

    if engine_file_path:
        with open(engine_file_path, "wb") as f:
            f.write(serialized_engine)
        print(f"Engine saved to {engine_file_path}")

    return serialized_engine


def load_engine(engine_file_path):
    """Loads a serialized TensorRT engine from disk."""
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_file_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine

def run_inference(path: str, input_array: np.ndarray) -> np.ndarray:
    """Deserialize the engine and run a single inference on the supplied input."""
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(path, "rb") as f:
        engine_data = f.read()

    engine = runtime.deserialize_cuda_engine(engine_data)
    context = engine.create_execution_context()

    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)

    context.set_input_shape(input_name, input_array.shape)

    # Resolve dynamic output dimensions after setting the input shape.
    output_shape = tuple(context.get_tensor_shape(output_name))
    if any(dim < 0 for dim in output_shape):
        raise RuntimeError(f"Output shape is still dynamic: {output_shape}")

    host_input = np.ascontiguousarray(input_array, dtype=np.float16)
    host_output = np.empty(output_shape, dtype=np.float16)
    
    err, d_input = cudart.cudaMalloc(host_input.nbytes)
    err, d_output = cudart.cudaMalloc(host_output.nbytes)
    err, stream = cudart.cudaStreamCreate()

    err, = cudart.cudaMemcpyAsync(
        d_input, host_input.ctypes.data, host_input.nbytes,
        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream,
    )

    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))
    context.execute_async_v3(stream)

    err, = cudart.cudaMemcpyAsync(
        host_output.ctypes.data, d_output, host_output.nbytes,
        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream,
    )

    err, = cudart.cudaStreamSynchronize(stream)

    err, = cudart.cudaFree(d_input)
    err, = cudart.cudaFree(d_output)
    err, = cudart.cudaStreamDestroy(stream)
    return host_output

def create():
    onnx_path = default_config.filepaths.encoder_path_fp16
    engine_path = "encoder.trt"

    # Build (or rebuild) the engine
    build_engine_from_onnx(onnx_path, engine_path)

def run():
    dummy_input_16 = np.random.rand(16, 1, 256, 1280).astype(np.float16)
    for i in range(8):
        t0  = perf_counter()
        output = run_inference("encoder.trt", dummy_input_16)
        print(perf_counter() - t0)


if __name__ == "__main__":
    create()
