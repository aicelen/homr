"""Build and run the TensorRT decoder engine.

The decoder is exported as a one-token step with a dynamic KV cache.  This
module deliberately keeps the cache outside of the engine: callers pass the
32 cache tensors to :func:`run` and receive the 32 updated tensors back.
"""

from collections.abc import Mapping
from time import perf_counter

import numpy as np
import onnx
import tensorrt as trt
from cuda.bindings import runtime as cudart
from homr.transformer.configs import default_config


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
BATCH_SIZE = 32
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


def _set_static_dim(value_info: onnx.ValueInfoProto, axis: int, value: int) -> None:
    dim = value_info.type.tensor_type.shape.dim[axis]
    dim.ClearField("dim_param")
    dim.dim_value = value


def _clear_dim_name(value_info: onnx.ValueInfoProto, axis: int) -> None:
    value_info.type.tensor_type.shape.dim[axis].ClearField("dim_param")


def _decoder_onnx_for_tensorrt(onnx_file_path: str) -> bytes:
    """Return decoder ONNX bytes with non-varying input axes made static.

    The exported decoder keeps batch-size, token-length, and ``cache_len`` axes
    symbolic even though this TensorRT runner only supports batch 16 one-token
    steps with a one-element ``cache_len`` tensor. TensorRT can fail shape
    analysis when those symbols do not drive any real dynamic shape relation, so
    only the context/cache sequence axes are left dynamic. The fp16 ONNX also
    contains inferred intermediate symbolic shapes from previous passes; those
    annotations are not needed for parsing and can trigger TensorRT symbol-tie
    assertions, so they are removed.
    """
    model = onnx.load(onnx_file_path)
    del model.graph.value_info[:]
    expected_inputs = set(INPUT_NAMES)
    for graph_input in model.graph.input:
        expected_inputs.discard(graph_input.name)
        if graph_input.name in TOKEN_INPUT_NAMES:
            _set_static_dim(graph_input, 0, BATCH_SIZE)
            _set_static_dim(graph_input, 1, 1)
        elif graph_input.name == "context":
            _set_static_dim(graph_input, 0, BATCH_SIZE)
            _clear_dim_name(graph_input, 1)
        elif graph_input.name == "cache_len":
            if len(graph_input.type.tensor_type.shape.dim) != 1:
                raise RuntimeError(
                    "cache_len must be rank 1, got "
                    f"rank {len(graph_input.type.tensor_type.shape.dim)}"
                )
            _set_static_dim(graph_input, 0, 1)
        elif graph_input.name in CACHE_INPUT_NAMES:
            _set_static_dim(graph_input, 0, BATCH_SIZE)
            _clear_dim_name(graph_input, 2)
    if expected_inputs:
        raise RuntimeError(f"ONNX decoder is missing inputs: {sorted(expected_inputs)}")

    for graph_output in model.graph.output:
        if graph_output.name.startswith("out_"):
            _set_static_dim(graph_output, 0, BATCH_SIZE)
            _set_static_dim(graph_output, 1, 1)
        elif graph_output.name.startswith("cache_out"):
            _set_static_dim(graph_output, 0, BATCH_SIZE)
            _clear_dim_name(graph_output, 2)
    return model.SerializeToString()


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
    max_workspace_size: int = 1 << 32,
) -> bytes | None:
    """Parse the decoder ONNX model and optionally save a serialized engine."""
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    if not parser.parse(_decoder_onnx_for_tensorrt(onnx_file_path)):
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


class DecoderSession:
    """Holds engine state + GPU-resident KV cache across decode steps."""

    def __init__(self, engine_path: str = "decoder.trt"):
        self.engine = load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        err, self.stream = cudart.cudaStreamCreate()
        _check(err)

        # name -> (device_ptr, shape, dtype)  -- lives entirely on GPU
        self.cache: dict[str, tuple[int, tuple[int, ...], np.dtype]] = {}
        self._init_empty_cache()

    def _init_empty_cache(self) -> None:
        for name in CACHE_INPUT_NAMES:
            shape = (BATCH_SIZE, 8, 0, 64)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            err, ptr = cudart.cudaMalloc(1)  # zero-length placeholder
            _check(err)
            self.cache[name] = (ptr, shape, dtype)

    def step(
        self,
        token_inputs: Mapping[str, np.ndarray],  # rhythms/pitchs/.../slurs
        context_arr: np.ndarray,
        cache_len: np.ndarray,
    ) -> dict[str, np.ndarray]:
        device_buffers: dict[str, int] = {}
        host_outputs: dict[str, np.ndarray] = {}
        new_cache_ptrs: dict[str, tuple[int, tuple[int, ...], np.dtype]] = {}

        try:
            # --- small inputs: host -> device as usual ---
            for name, value in token_inputs.items():
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                host_arr = np.ascontiguousarray(value, dtype=dtype)
                self.context.set_input_shape(name, host_arr.shape)
                err, ptr = cudart.cudaMalloc(max(1, host_arr.nbytes))
                _check(err)
                device_buffers[name] = ptr
                if host_arr.nbytes:
                    _check(cudart.cudaMemcpyAsync(
                        ptr, host_arr.ctypes.data, host_arr.nbytes,
                        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream))
                self.context.set_tensor_address(name, ptr)

            for name, value in (("context", context_arr), ("cache_len", cache_len)):
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                host_arr = np.ascontiguousarray(value, dtype=dtype)
                self.context.set_input_shape(name, host_arr.shape)
                err, ptr = cudart.cudaMalloc(max(1, host_arr.nbytes))
                _check(err)
                device_buffers[name] = ptr
                _check(cudart.cudaMemcpyAsync(
                    ptr, host_arr.ctypes.data, host_arr.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream))
                self.context.set_tensor_address(name, ptr)

            # --- cache inputs: reuse the device pointer from last step's output ---
            for name in CACHE_INPUT_NAMES:
                ptr, shape, _ = self.cache[name]
                self.context.set_input_shape(name, shape)
                self.context.set_tensor_address(name, ptr)

            # --- allocate outputs ---
            for i in range(self.engine.num_io_tensors):
                out_name = self.engine.get_tensor_name(i)
                if self.engine.get_tensor_mode(out_name) != trt.TensorIOMode.OUTPUT:
                    continue
                shape = tuple(self.context.get_tensor_shape(out_name))
                dtype = trt.nptype(self.engine.get_tensor_dtype(out_name))
                nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
                err, ptr = cudart.cudaMalloc(max(1, nbytes))
                _check(err)

                if out_name.startswith("cache_out"):
                    # stays on GPU — becomes next step's cache_in, no D2H copy
                    new_cache_ptrs[out_name] = (ptr, shape, dtype)
                else:
                    # small logits etc — worth bringing back to host
                    device_buffers[out_name] = ptr
                    host_outputs[out_name] = np.empty(shape, dtype=dtype)
                self.context.set_tensor_address(out_name, ptr)

            if not self.context.execute_async_v3(self.stream):
                raise RuntimeError("TensorRT decoder execution failed")

            for name, arr in host_outputs.items():
                if arr.nbytes:
                    _check(cudart.cudaMemcpyAsync(
                        arr.ctypes.data, device_buffers[name], arr.nbytes,
                        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream))
            _check(cudart.cudaStreamSynchronize(self.stream))

        finally:
            # free everything EXCEPT the cache buffers we're carrying forward
            for name, ptr in device_buffers.items():
                cudart.cudaFree(ptr)

        # swap in the new cache, freeing the old cache buffers now that
        # the step that consumed them has finished executing
        for name, (old_ptr, _, _) in self.cache.items():
            cudart.cudaFree(old_ptr)
        cache_out_map = {f"cache_in{i}": f"cache_out{i}" for i in range(NUM_CACHE_TENSORS)}
        for cache_in_name, cache_out_name in cache_out_map.items():
            self.cache[cache_in_name] = new_cache_ptrs[cache_out_name]

        return host_outputs

    def close(self) -> None:
        for ptr, _, _ in self.cache.values():
            cudart.cudaFree(ptr)
        cudart.cudaStreamDestroy(self.stream)


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
