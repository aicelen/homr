from typing import Any

import numpy as np

from homr.transformer.configs import Config
from homr.transformer.vocabulary import EncodedSymbol
from homr.type_definitions import NDArray
from trt.build_trt_decoder import BATCH_SIZE, CACHE_INPUT_NAMES, run


class ScoreDecoder:
    def __init__(
        self,
        engine_path: str,
        config: Config,
        fp16: bool = False,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.config = config
        self.engine_path = engine_path
        self.max_seq_len = config.max_seq_len
        self.eos_token = config.eos_token

        self.inv_rhythm_vocab = {v: k for k, v in config.rhythm_vocab.items()}
        self.inv_pitch_vocab = {v: k for k, v in config.pitch_vocab.items()}
        self.inv_lift_vocab = {v: k for k, v in config.lift_vocab.items()}
        self.inv_articulation_vocab = {v: k for k, v in config.articulation_vocab.items()}
        self.inv_slur_vocab = {v: k for k, v in config.slur_vocab.items()}
        self.inv_position_vocab = {v: k for k, v in config.position_vocab.items()}

        self.fp16 = fp16
        self.use_gpu = True
        self.output_names = [
            "out_rhythms",
            "out_pitchs",
            "out_lifts",
            "out_positions",
            "out_articulations",
            "out_slurs",
            "attention",
        ]

    def generate(
        self,
        start_tokens: NDArray,
        nonote_tokens: NDArray,
        **kwargs: Any,
    ) -> list[EncodedSymbol]:
        if len(start_tokens.shape) == 1:
            start_tokens = start_tokens[None, :]
        if len(nonote_tokens.shape) == 1:
            nonote_tokens = nonote_tokens[None, :]

        out_rhythm = _to_trt_batch(start_tokens, "start_tokens")
        nonote_tokens = _to_trt_batch(nonote_tokens, "nonote_tokens")
        out_pitch = nonote_tokens
        out_lift = nonote_tokens
        out_articulations = nonote_tokens
        out_slurs = nonote_tokens

        cache = self.init_cache()
        context = _to_trt_batch(kwargs["context"], "context")
        context_reduced = context[:, :1]

        symbols: list[EncodedSymbol] = []

        for step in range(self.max_seq_len):
            step_context = context if step == 0 else context_reduced
            inputs: dict[str, NDArray] = {
                "rhythms": out_rhythm[:, -1:],
                "pitchs": out_pitch[:, -1:],
                "lifts": out_lift[:, -1:],
                "articulations": out_articulations[:, -1:],
                "slurs": out_slurs[:, -1:],
                "context": step_context,
                "cache_len": np.array([step], dtype=np.int64),
            }
            inputs.update({name: cache[name] for name in CACHE_INPUT_NAMES})

            outputs = run(inputs, self.engine_path)
            cache = {
                input_name: outputs[f"cache_out{i}"]
                for i, input_name in enumerate(CACHE_INPUT_NAMES)
            }

            rhythmsp = outputs["out_rhythms"]
            pitchsp = outputs["out_pitchs"]
            liftsp = outputs["out_lifts"]
            positionsp = outputs["out_positions"]
            articulationsp = outputs["out_articulations"]
            slursp = outputs["out_slurs"]
            attention = outputs.get("attention", np.empty((0,), dtype=np.float32))

            rhythm_sample = _sample_last_token(rhythmsp)
            pitch_sample = _sample_last_token(pitchsp)
            lift_sample = _sample_last_token(liftsp)
            articulation_sample = _sample_last_token(articulationsp)
            slur_sample = _sample_last_token(slursp)
            position_sample = _sample_last_token(positionsp)

            lift_token = detokenize(lift_sample[:1], self.inv_lift_vocab)
            pitch_token = detokenize(pitch_sample[:1], self.inv_pitch_vocab)
            rhythm_token = detokenize(rhythm_sample[:1], self.inv_rhythm_vocab)
            articulation_token = detokenize(
                articulation_sample[:1], self.inv_articulation_vocab
            )
            slur_token = detokenize(slur_sample[:1], self.inv_slur_vocab)
            position_token = detokenize(position_sample[:1], self.inv_position_vocab)

            if rhythm_sample[0][0] == self.eos_token:
                break

            symbol = EncodedSymbol(
                rhythm=rhythm_token[0],
                pitch=pitch_token[0],
                lift=lift_token[0],
                articulation=articulation_token[0],
                slur=slur_token[0],
                position=position_token[0],
                coordinates=attention,
            )
            symbols.append(symbol)

            out_lift = np.concatenate((out_lift, lift_sample), axis=-1)
            out_pitch = np.concatenate((out_pitch, pitch_sample), axis=-1)
            out_rhythm = np.concatenate((out_rhythm, rhythm_sample), axis=-1)
            out_articulations = np.concatenate((out_articulations, articulation_sample), axis=-1)
            out_slurs = np.concatenate((out_slurs, slur_sample), axis=-1)

        return symbols

    def init_cache(self, cache_len: int = 0) -> dict[str, NDArray]:
        cache = {}
        heads = self.config.decoder_heads
        head_dim = self.config.decoder_dim // heads
        dtype = np.float16 if self.fp16 else np.float32
        for name in CACHE_INPUT_NAMES:
            cache[name] = np.zeros((BATCH_SIZE, heads, cache_len, head_dim), dtype=dtype)
        return cache


def _sample_last_token(logits: NDArray) -> NDArray:
    return np.argmax(logits[:, -1, :], axis=-1).astype(np.int64)[:, None]


def _to_trt_batch(value: NDArray, name: str) -> NDArray:
    if value.shape[0] == BATCH_SIZE:
        return value
    if value.shape[0] == 1:
        return np.repeat(value, BATCH_SIZE, axis=0)
    raise ValueError(f"{name} must have batch size 1 or {BATCH_SIZE}, got {value.shape[0]}")


def detokenize(tokens: NDArray, vocab: dict[int, str]) -> list[str]:
    toks = [vocab[tok.item()] for tok in tokens]
    toks = [t for t in toks if t not in ("[BOS]", "[EOS]", "[PAD]")]
    return toks


def get_decoder(config: Config, engine_path: str = "decoder.trt") -> ScoreDecoder:
    """
    Returns Tromr's TensorRT Decoder.
    """
    return ScoreDecoder(engine_path, config=config)
