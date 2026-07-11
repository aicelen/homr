import tensorrt as trt
from homr.transformer.configs import default_config

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

    # Optional: set an optimization profile if your model has dynamic input shapes
    input_tensor = network.get_input(0)
    if -1 in input_tensor.shape:  # dynamic shape present
        profile = builder.create_optimization_profile()
        # Example: adjust min/opt/max shapes to your model's actual input name/shape
        profile.set_shape(input_tensor.name,
                           min=(1, 3, 256, 1280),
                           opt=(1, 3, 256, 1280),
                           max=(1, 3, 256, 1280))
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


if __name__ == "__main__":
    onnx_path = default_config.filepaths.encoder_path_fp16
    engine_path = "encoder.trt"

    # Build (or rebuild) the engine
    build_engine_from_onnx(onnx_path, engine_path)
