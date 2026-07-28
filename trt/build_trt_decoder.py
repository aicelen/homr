"""Build and run the TensorRT decoder engine.

The decoder is exported as a one-token step with a dynamic KV cache.  This
module deliberately keeps the cache outside of the engine: callers pass the
32 cache tensors to :func:`run` and receive the 32 updated tensors back.
"""

from collections.abc import Mapping
from time import perf_counter

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart
from homr.transformer.configs import default_config


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
BATCH_SIZE = 16
NUM_CACHE_TENSORS = 32
MAX_SEQ_LEN = 608
MAX_CONTEXT_LEN = 1280

TOKEN_INPUT_NAMES = (
    "rhythms",
    "pitchs",
    "lifts",
    "articulations",
    "slurs",
)
CACHE_INPUT_NAMES = tuple(f"cache_in{i}" for i in range(NUM_CACHE_TENSORS))
INPUT_NAMES = TOKEN_INPUT_NAMES + ("context", "cache_len") + CACHE_INPUT_NAMES


def _check(err: object) -> None:
    if isinstance(err, tuple):
        err = err[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {err}")


def _is_cross_attention_cache(name: str) -> bool:
    cache_index = int(name.removeprefix("cache_in"))
    return cache_index % 4 in (2, 3)


def _rename_dynamic_cache_dimensions(network: trt.INetworkDefinition) -> None:
    """Avoid TensorRT treating every cache length as one shared dimension."""
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        try:
            if tensor.name in CACHE_INPUT_NAMES:
                tensor.set_dimension_name(2, f"{tensor.name}_seq_len")
            elif tensor.name == "context":
                tensor.set_dimension_name(1, "context_len")
        except AttributeError as exc:
            raise RuntimeError(
                "TensorRT Python bindings do not support renaming dynamic "
                "dimensions; rebuild the ONNX with unique cache axis names."
            ) from exc


def _set_decoder_profile(
    profile: trt.IOptimizationProfile, network: trt.INetworkDefinition
) -> None:
    """Set profiles for every decoder input, including all cache tensors."""
    network_inputs = {
        network.get_input(i).name: network.get_input(i) for i in range(network.num_inputs)
    }
    missing = set(INPUT_NAMES) - network_inputs.keys()
    if missing:
        raise RuntimeError(f"ONNX decoder is missing inputs: {sorted(missing)}")

    for name in TOKEN_INPUT_NAMES:
        profile.set_shape(name, (BATCH_SIZE, 1), (BATCH_SIZE, 1), (BATCH_SIZE, 1))

    # The first decoder step receives the complete encoder context.  Later
    # steps receive one context vector, so both shapes must be supported.
    profile.set_shape(
        "context",
        (BATCH_SIZE, 1, 512),
        (BATCH_SIZE, MAX_CONTEXT_LEN, 512),
        (BATCH_SIZE, MAX_CONTEXT_LEN, 512),
    )
    profile.set_shape("cache_len", (1,), (1,), (1,))

    for name in CACHE_INPUT_NAMES:
        max_cache_len = (
            MAX_CONTEXT_LEN + MAX_SEQ_LEN if _is_cross_attention_cache(name) else MAX_SEQ_LEN
        )
        profile.set_shape(
            name,
            (BATCH_SIZE, 8, 0, 64),
            (BATCH_SIZE, 8, 1, 64),
            (BATCH_SIZE, 8, max_cache_len, 64),
        )


def build_engine_from_onnx(
    onnx_file_path: str,
    engine_file_path: str | None = None,
    fp16_mode: bool = False,
    max_workspace_size: int = 1 << 30,
) -> bytes | None:
    """Parse the decoder ONNX model and optionally save a serialized engine."""
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_file_path, "rb") as model_file:
        if not parser.parse(model_file.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    _rename_dynamic_cache_dimensions(network)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)
    if fp16_mode:
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    _set_decoder_profile(profile, network)
    config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("ERROR: Failed to build the engine.")
        return None

    if engine_file_path:
        with open(engine_file_path, "wb") as engine_file:
            engine_file.write(serialized_engine)
        print(f"Engine saved to {engine_file_path}")

    return bytes(serialized_engine)


def load_engine(engine_file_path: str) -> trt.ICudaEngine:
    """Load a serialized TensorRT decoder engine."""
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_file_path, "rb") as engine_file:
        engine = runtime.deserialize_cuda_engine(engine_file.read())
    if engine is None:
        raise RuntimeError(f"Could not deserialize TensorRT engine: {engine_file_path}")
    return engine


def run(
    inputs: Mapping[str, np.ndarray], engine_path: str = "decoder.trt"
) -> dict[str, np.ndarray]:
    """Run exactly one decoder step and return every engine output.

    ``inputs`` must contain the five token tensors, ``context``, ``cache_len``,
    and ``cache_in0`` through ``cache_in31``.  The returned dictionary is keyed
    by the ONNX output names, including all ``out_*`` and ``cache_out*``
    tensors (and ``attention`` when present in the model).
    """
    engine = load_engine(engine_path)
    context = engine.create_execution_context()

    engine_input_names = [
        engine.get_tensor_name(i)
        for i in range(engine.num_io_tensors)
        if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT
    ]
    missing_engine_inputs = set(INPUT_NAMES) - set(engine_input_names)
    if missing_engine_inputs:
        raise RuntimeError(f"TensorRT engine is missing inputs: {sorted(missing_engine_inputs)}")
    missing = set(INPUT_NAMES) - set(inputs)
    unexpected = set(inputs) - set(INPUT_NAMES)
    if missing or unexpected:
        raise ValueError(
            f"Invalid decoder inputs; missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )

    device_buffers: dict[str, object] = {}
    host_outputs: dict[str, np.ndarray] = {}
    stream = None
    try:
        err, stream = cudart.cudaStreamCreate()
        _check(err)
        host_inputs: dict[str, np.ndarray] = {}
        for name in engine_input_names:
            value = np.asarray(inputs[name])
            if name in TOKEN_INPUT_NAMES or name == "cache_len":
                expected_shape = (BATCH_SIZE, 1) if name in TOKEN_INPUT_NAMES else (1,)
                if value.shape != expected_shape:
                    raise ValueError(f"{name} must have shape {expected_shape}, got {value.shape}")
            elif name in CACHE_INPUT_NAMES and (
                value.ndim != 4
                or value.shape[0] != BATCH_SIZE
                or value.shape[1] != 8
                or value.shape[3] != 64
            ):
                raise ValueError(
                    f"{name} must have shape ({BATCH_SIZE}, 8, seq_len, 64), got {value.shape}"
                )
            elif name in CACHE_INPUT_NAMES:
                max_cache_len = (
                    MAX_CONTEXT_LEN + MAX_SEQ_LEN
                    if _is_cross_attention_cache(name)
                    else MAX_SEQ_LEN
                )
                if value.shape[2] > max_cache_len:
                    raise ValueError(
                        f"{name} seq_len must be <= {max_cache_len}, got {value.shape[2]}"
                    )
            elif name == "context" and (
                value.ndim != 3 or value.shape[0] != BATCH_SIZE or value.shape[2] != 512
            ):
                raise ValueError(
                    f"context must have shape ({BATCH_SIZE}, cache_exists, 512), got {value.shape}"
                )

            dtype = trt.nptype(engine.get_tensor_dtype(name))
            host_input = np.ascontiguousarray(value, dtype=dtype)
            host_inputs[name] = host_input
            if not context.set_input_shape(name, host_input.shape):
                raise RuntimeError(
                    f"TensorRT rejected input shape for {name}: {host_input.shape}"
                )
            err, device_buffer = cudart.cudaMalloc(max(1, host_input.nbytes))
            _check(err)
            device_buffers[name] = device_buffer

        for name, value in host_inputs.items():
            if value.nbytes:
                _check(
                    cudart.cudaMemcpyAsync(
                        device_buffers[name],
                        value.ctypes.data,
                        value.nbytes,
                        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                        stream,
                    )
                )
            context.set_tensor_address(name, int(device_buffers[name]))

        output_names = [
            engine.get_tensor_name(i)
            for i in range(engine.num_io_tensors)
            if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT
        ]
        for name in output_names:
            shape = tuple(context.get_tensor_shape(name))
            if any(dim < 0 for dim in shape):
                raise RuntimeError(f"Output shape for {name} is still dynamic: {shape}")
            output = np.empty(shape, dtype=trt.nptype(engine.get_tensor_dtype(name)))
            host_outputs[name] = output
            err, device_buffer = cudart.cudaMalloc(max(1, output.nbytes))
            _check(err)
            device_buffers[name] = device_buffer
            context.set_tensor_address(name, int(device_buffer))

        if not context.execute_async_v3(stream):
            raise RuntimeError("TensorRT decoder execution failed")
        for name, output in host_outputs.items():
            if output.nbytes:
                _check(
                    cudart.cudaMemcpyAsync(
                        output.ctypes.data,
                        device_buffers[name],
                        output.nbytes,
                        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                        stream,
                    )
                )
        _check(cudart.cudaStreamSynchronize(stream))
        return host_outputs
    finally:
        for buffer in device_buffers.values():
            cudart.cudaFree(buffer)
        if stream is not None:
            cudart.cudaStreamDestroy(stream)


def run_inference(
    inputs: Mapping[str, np.ndarray], engine_path: str = "decoder.trt"
) -> dict[str, np.ndarray]:
    """Backward-compatible descriptive alias for :func:`run`."""
    return run(inputs, engine_path)


def create() -> None:
    """Build the default fp32 decoder engine."""
    build_engine_from_onnx(default_config.filepaths.decoder_path_fp16, "decoder.trt")


def _dummy_inputs() -> dict[str, np.ndarray]:
    inputs = {name: np.zeros((BATCH_SIZE, 1), dtype=np.int64) for name in TOKEN_INPUT_NAMES}
    inputs["context"] = np.random.rand(BATCH_SIZE, 1280, 512).astype(np.float32)
    inputs["cache_len"] = np.zeros((1,), dtype=np.int64)
    for name in CACHE_INPUT_NAMES:
        inputs[name] = np.zeros((BATCH_SIZE, 8, 0, 64), dtype=np.float32)
    return inputs

def run_test():
    dummy_inputs = _dummy_inputs()
    for _ in range(8):
        start = perf_counter()
        outputs = run(dummy_inputs)
        print(perf_counter() - start)


if __name__ == "__main__":
    create()
