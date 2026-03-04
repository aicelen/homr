import os
from time import perf_counter

import numpy as np

from homr.simple_logging import eprint
from homr.transformer.configs import Config
from homr.transformer.vocabulary import EncodedSymbol, nonote
from homr.type_definitions import NDArray


SELF_CACHE_TENSOR_INDICES = (
    0,
    1,
    4,
    5,
    8,
    9,
    12,
    13,
    16,
    17,
    20,
    21,
    24,
    25,
    28,
    29,
)


class LiteRTDecoder:
    def __init__(self, config: Config) -> None:
        try:
            from ai_edge_litert import interpreter  # noqa: F401
        except ImportError as ex:
            raise RuntimeError("Failed to import ai_edge_litert interpreter") from ex

        model_path = config.filepaths.transformer_litert_tflite
        if not os.path.exists(model_path):
            raise FileNotFoundError("Failed to find LiteRT model " + model_path)

        # Use default delegates (XNNPACK on CPU) for faster inference.
        self.interpreter = interpreter.Interpreter(model_path=model_path, num_threads=8)
        self.interpreter.allocate_tensors()
        signatures = self.interpreter.get_signature_list()
        if "prefill" not in signatures or "decode" not in signatures:
            raise RuntimeError(
                "LiteRT model is missing expected signatures. Found " + str(signatures.keys())
            )
        self.prefill = self.interpreter.get_signature_runner("prefill")
        self.decode = self.interpreter.get_signature_runner("decode")
        decode_inputs = set(signatures["decode"]["inputs"])
        self.use_context_self_cache_format = (
            "context" in decode_inputs and "self_cache0" in decode_inputs
        )
        self.self_cache_names = [f"self_cache{i}" for i in range(len(SELF_CACHE_TENSOR_INDICES))]
        self.self_cache_len = config.max_seq_len - 1

        self.max_seq_len = config.max_seq_len
        self.eos_token = config.eos_token
        self.pad_token = config.pad_token
        self.inv_rhythm_vocab = {v: k for k, v in config.rhythm_vocab.items()}
        self.inv_pitch_vocab = {v: k for k, v in config.pitch_vocab.items()}
        self.inv_lift_vocab = {v: k for k, v in config.lift_vocab.items()}
        self.inv_articulation_vocab = {v: k for k, v in config.articulation_vocab.items()}
        self.inv_position_vocab = {v: k for k, v in config.position_vocab.items()}

        self.self_cache_indices = {
            *SELF_CACHE_TENSOR_INDICES,
        }

    def _ordered_outputs(self, outputs: dict[str, NDArray]) -> list[NDArray]:
        output_tuples = [
            (int(key.split("_")[1]), value) for key, value in outputs.items() if key.startswith("output_")
        ]
        if not output_tuples:
            raise RuntimeError("Failed to decode LiteRT outputs: " + str(outputs.keys()))
        ordered: list[NDArray | None] = [None] * (max(index for index, _ in output_tuples) + 1)
        for index, value in output_tuples:
            ordered[index] = value
        if any(v is None for v in ordered):
            raise RuntimeError("Failed to decode LiteRT outputs: " + str(outputs.keys()))
        return [v for v in ordered if v is not None]

    def _ensure_float32(self, array: NDArray) -> NDArray:
        out = np.asarray(array)
        if out.dtype != np.float32:
            out = out.astype(np.float32, copy=False)
        return out

    def _ensure_int64(self, array: NDArray) -> NDArray:
        out = np.asarray(array)
        if out.dtype != np.int64:
            out = out.astype(np.int64, copy=False)
        return out

    def _to_token(self, array: NDArray) -> int:
        return int(np.asarray(array).item())

    def _to_symbol(
        self,
        rhythm: NDArray,
        pitch: NDArray,
        lift: NDArray,
        position: NDArray,
        articulation: NDArray,
    ) -> EncodedSymbol:
        rhythm_id = self._to_token(rhythm)
        pitch_id = self._to_token(pitch)
        lift_id = self._to_token(lift)
        position_id = self._to_token(position)
        articulation_id = self._to_token(articulation)
        return EncodedSymbol(
            rhythm=self.inv_rhythm_vocab.get(rhythm_id, nonote),
            pitch=self.inv_pitch_vocab.get(pitch_id, nonote),
            lift=self.inv_lift_vocab.get(lift_id, nonote),
            articulation=self.inv_articulation_vocab.get(articulation_id, nonote),
            position=self.inv_position_vocab.get(position_id, nonote),
        )

    def _compact_caches(self, caches: list[NDArray]) -> list[NDArray]:
        # Workaround: exported decode signature expects one-step self cache input.
        # Keep only the latest token in self-cache tensors and pass cross-cache as-is.
        compacted: list[NDArray] = []
        for index, cache in enumerate(caches):
            if index in self.self_cache_indices and cache.shape[2] > 1:
                compacted.append(cache[:, :, -1:, :])
            else:
                compacted.append(cache)
        return compacted

    def _generate_legacy(self, x: NDArray) -> list[EncodedSymbol]:
        prefill_outputs = self._ordered_outputs(self.prefill(image=self._ensure_float32(x)))
        rhythm = prefill_outputs[0]
        pitch = prefill_outputs[1]
        lift = prefill_outputs[2]
        position = prefill_outputs[3]
        articulation = prefill_outputs[4]
        caches = self._compact_caches([self._ensure_float32(cache) for cache in prefill_outputs[6:]])

        symbols: list[EncodedSymbol] = []
        for step in range(self.max_seq_len):
            rhythm_id = self._to_token(rhythm)
            if rhythm_id in (self.eos_token, self.pad_token):
                break
            symbols.append(self._to_symbol(rhythm, pitch, lift, position, articulation))
            if step + 1 >= self.max_seq_len:
                break

            decode_inputs: dict[str, NDArray] = {
                "rhythms": self._ensure_int64(rhythm),
                "pitchs": self._ensure_int64(pitch),
                "lifts": self._ensure_int64(lift),
                "articulations": self._ensure_int64(articulation),
                "cache_len": np.array([step + 1], dtype=np.int64),
            }
            for cache_index, cache in enumerate(caches):
                decode_inputs[f"cache{cache_index}"] = cache
            decode_outputs = self._ordered_outputs(self.decode(**decode_inputs))
            rhythm = decode_outputs[0]
            pitch = decode_outputs[1]
            lift = decode_outputs[2]
            position = decode_outputs[3]
            articulation = decode_outputs[4]
            caches = self._compact_caches([self._ensure_float32(cache) for cache in decode_outputs[6:]])

        return symbols

    def _generate_context_self_cache(self, x: NDArray) -> list[EncodedSymbol]:
        prefill_outputs = self._ordered_outputs(self.prefill(image=self._ensure_float32(x)))
        rhythm = prefill_outputs[0]
        pitch = prefill_outputs[1]
        lift = prefill_outputs[2]
        position = prefill_outputs[3]
        articulation = prefill_outputs[4]
        context = self._ensure_float32(prefill_outputs[6])
        self_caches = [self._ensure_float32(cache) for cache in prefill_outputs[7:]]
        if len(self_caches) != len(self.self_cache_names):
            raise RuntimeError(
                f"Expected {len(self.self_cache_names)} self-cache tensors, got {len(self_caches)}"
            )

        symbols: list[EncodedSymbol] = []
        for step in range(self.max_seq_len):
            rhythm_id = self._to_token(rhythm)
            if rhythm_id in (self.eos_token, self.pad_token):
                break
            symbols.append(self._to_symbol(rhythm, pitch, lift, position, articulation))
            if step + 1 >= self.max_seq_len:
                break

            cache_len = step + 1
            decode_inputs: dict[str, NDArray] = {
                "rhythms": self._ensure_int64(rhythm),
                "pitchs": self._ensure_int64(pitch),
                "lifts": self._ensure_int64(lift),
                "articulations": self._ensure_int64(articulation),
                "context": context,
                "cache_len": np.array([cache_len], dtype=np.int64),
            }
            for name, cache in zip(self.self_cache_names, self_caches, strict=True):
                decode_inputs[name] = cache

            decode_outputs = self._ordered_outputs(self.decode(**decode_inputs))
            rhythm = decode_outputs[0]
            pitch = decode_outputs[1]
            lift = decode_outputs[2]
            position = decode_outputs[3]
            articulation = decode_outputs[4]
            if cache_len < self.self_cache_len:
                for cache, cache_update in zip(self_caches, decode_outputs[6:], strict=True):
                    cache[:, :, cache_len : cache_len + 1, :] = self._ensure_float32(cache_update)

        return symbols

    def generate(self, x: NDArray) -> list[EncodedSymbol]:
        if self.use_context_self_cache_format:
            return self._generate_context_self_cache(x)
        return self._generate_legacy(x)


class Staff2Score:
    """
    Inference class for Tromr. Use predict() for prediction.
    Prefers LiteRT backend when a converted TFLite model is available,
    and falls back to ONNX backend otherwise.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.litert: LiteRTDecoder | None = None
        self.encoder = None
        self.decoder = None

        if os.path.exists(self.config.filepaths.transformer_litert_tflite):
            try:
                self.litert = LiteRTDecoder(self.config)
                eprint(
                    "Using LiteRT transformer backend:",
                    self.config.filepaths.transformer_litert_tflite,
                )
            except Exception as ex:
                eprint(ex)
                eprint("Failed to initialize LiteRT backend, falling back to ONNX")

        if self.litert is None:
            from homr.transformer.decoder_inference import get_decoder
            from homr.transformer.encoder_inference import Encoder

            self.encoder = Encoder(self.config)
            self.decoder = get_decoder(self.config)
            eprint("Using ONNX transformer backend")

        if not os.path.exists(self.config.filepaths.rhythmtokenizer):
            raise RuntimeError(
                "Failed to find tokenizer config" + self.config.filepaths.rhythmtokenizer
            )  # noqa: E501

    def predict(self, image: NDArray) -> list[EncodedSymbol]:
        """
        Inference an image (NDArray) using Tromr.
        """
        x = _transform(image=image)
        t0 = perf_counter()

        if self.litert is not None:
            out = self.litert.generate(x)
            eprint(f"Inference Time Tromr LiteRT: {perf_counter()-t0}")
            return out

        assert self.encoder is not None
        assert self.decoder is not None

        # Create special tokens
        start_token = np.array([[1]], dtype=np.int64)
        nonote_token = np.array([[0]], dtype=np.int64)

        # Generate context with encoder
        context = self.encoder.generate(x)

        # Make a prediction using decoder
        out = self.decoder.generate(
            start_token,
            nonote_token,
            seq_len=self.config.max_seq_len,
            eos_token=self.config.eos_token,
            context=context,
        )

        eprint(f"Inference Time Tromr ONNX: {perf_counter()-t0}")
        return out


class ConvertToArray:
    def __init__(self) -> None:
        self.mean = np.array([0.7931]).reshape(1, 1, 1)
        self.std = np.array([0.1738]).reshape(1, 1, 1)

    def normalize(self, array: NDArray) -> NDArray:
        return (array - self.mean) / self.std

    def __call__(self, image: NDArray) -> NDArray:
        arr = np.asarray(image)
        if arr.ndim == 3:
            if arr.shape[2] == 4:
                arr = 255 - arr[:, :, 3]
            else:
                arr = np.mean(arr[:, :, :3], axis=2)
        arr = arr.astype(np.float32, copy=False) / 255.0
        arr = arr[np.newaxis, np.newaxis, :, :]
        return self.normalize(arr).astype(np.float32, copy=False)


_transform = ConvertToArray()


def test_transformer_on_image(path_to_img: str) -> None:
    """
    Tests the transformer on an image and prints the results.
    Args:
        path_to_img(str): Path to the image to test
    """
    from PIL import Image

    model = Staff2Score(Config())
    image = Image.open(path_to_img)
    out = model.predict(np.array(image))
    eprint(out)


if __name__ == "__main__":
    import sys

    test_transformer_on_image(sys.argv[1])
