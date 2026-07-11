from typing import Any

import numpy as np
import onnxruntime as ort

from homr.simple_logging import eprint
from homr.transformer.configs import Config, BATCH_SIZE
from homr.transformer.vocabulary import EncodedSymbol
from homr.type_definitions import NDArray

class ScoreDecoder:
    def __init__(
        self,
        transformer: ort.InferenceSession,
        fp16: bool,
        use_gpu: bool,
        config: Config,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.config = config
        self.net = transformer
        self.io_binding = self.net.io_binding()
        self.max_seq_len = config.max_seq_len
        self.eos_token = config.eos_token

        self.inv_rhythm_vocab = {v: k for k, v in config.rhythm_vocab.items()}
        self.inv_pitch_vocab = {v: k for k, v in config.pitch_vocab.items()}
        self.inv_lift_vocab = {v: k for k, v in config.lift_vocab.items()}
        self.inv_articulation_vocab = {v: k for k, v in config.articulation_vocab.items()}
        self.inv_slur_vocab = {v: k for k, v in config.slur_vocab.items()}
        self.inv_position_vocab = {v: k for k, v in config.position_vocab.items()}

        self.fp16 = fp16
        self.use_gpu = use_gpu
        self.device_id = 0
        self.output_names = [
            "out_rhythms",
            "out_pitchs",
            "out_lifts",
            "out_positions",
            "out_articulations",
            "out_slurs",
        ]

    def generate(
            self,
            start_tokens: NDArray,
            nonote_tokens: NDArray,
            **kwargs: Any,
        ) -> list[EncodedSymbol]:
            num_dims = len(start_tokens.shape)

            if num_dims == 1:
                start_tokens = start_tokens[None, :]

            out_rhythm = start_tokens.copy()
            out_pitch = nonote_tokens.copy()
            out_lift = nonote_tokens.copy()
            out_articulations = nonote_tokens.copy()
            out_slurs = nonote_tokens.copy()
            
            # Initialize cache as numpy arrays so they can be dynamically sliced
            cache_ort, kv_input_names, kv_output_names = self.init_cache()
            cache_active = [c.numpy() for c in cache_ort]
            
            output_names = self.output_names + kv_output_names
            
            context_active = kwargs["context"]
            context_reduced_active = kwargs["context"][:, :1]

            symbols: list[list[EncodedSymbol]] = [[] for _ in range(BATCH_SIZE)]
            
            # Track which batch indices are still generating
            active_indices = list(range(BATCH_SIZE))

            # Initial inputs
            x_lift_active = out_lift[:, -1:]
            x_pitch_active = out_pitch[:, -1:]
            x_rhythm_active = out_rhythm[:, -1:]
            x_articulations_active = out_articulations[:, -1:]
            x_slurs_active = out_slurs[:, -1:]

            device_type = "cuda" if self.use_gpu else "cpu"

            for step in range(self.max_seq_len):
                if not active_indices:
                    break
                    
                cur_batch_size = len(active_indices)
                current_context = context_active if step == 0 else context_reduced_active

                # Bind Standard Inputs
                self.io_binding.bind_cpu_input("rhythms", x_rhythm_active)
                self.io_binding.bind_cpu_input("pitchs", x_pitch_active)
                self.io_binding.bind_cpu_input("lifts", x_lift_active)
                self.io_binding.bind_cpu_input("articulations", x_articulations_active)
                self.io_binding.bind_cpu_input("slurs", x_slurs_active)
                self.io_binding.bind_cpu_input("context", current_context)
                self.io_binding.bind_cpu_input("cache_len", np.full(cur_batch_size, step, dtype=np.int64))
                
                # Re-bind Cache (moves sliced numpy arrays back to OrtValues on the correct device)
                for name, cache_arr in zip(kv_input_names, cache_active, strict=True):
                    ort_val = ort.OrtValue.ortvalue_from_numpy(cache_arr, device_type, self.device_id)
                    self.io_binding.bind_ortvalue_input(name, ort_val)

                # Bind Outputs
                for name in output_names:
                    self.io_binding.bind_output(name, device_type, self.device_id)

                # Run inference
                self.net.run_with_iobinding(iobinding=self.io_binding)

                # Get outputs
                outputs = self.io_binding.get_outputs()
                
                # The new cache for the currently active sequences
                new_cache_active = [out.numpy() for out in outputs[6:]]

                # Greedy decoding
                rhythmsp = outputs[0].numpy()
                pitchsp = outputs[1].numpy()
                liftsp = outputs[2].numpy()
                positionsp = outputs[3].numpy()
                articulationsp = outputs[4].numpy()
                slursp = outputs[5].numpy()

                rhythm_samples       = rhythmsp[:, -1, :].argmax(axis=-1).reshape(-1, 1)
                pitch_samples        = pitchsp[:, -1, :].argmax(axis=-1).reshape(-1, 1)
                lift_samples         = liftsp[:, -1, :].argmax(axis=-1).reshape(-1, 1)
                articulation_samples = articulationsp[:, -1, :].argmax(axis=-1).reshape(-1, 1)
                slur_samples         = slursp[:, -1, :].argmax(axis=-1).reshape(-1, 1)
                position_samples     = positionsp[:, -1, :].argmax(axis=-1).reshape(-1, 1)

                # Determine which of the *current* active sequences are surviving this step
                keep_mask = (rhythm_samples[:, 0] != self.eos_token)

                # Process symbols mapped back to their original batch index
                for i, orig_idx in enumerate(active_indices):
                    if not keep_mask[i]:
                        continue  # Sequence generated EOS on this exact step

                    lift_token = detokenize(lift_samples[i:i+1], self.inv_lift_vocab)[0]
                    pitch_token = detokenize(pitch_samples[i:i+1], self.inv_pitch_vocab)[0]
                    rhythm_token = detokenize(rhythm_samples[i:i+1], self.inv_rhythm_vocab)[0]
                    articulation_token = detokenize(articulation_samples[i:i+1], self.inv_articulation_vocab)[0]
                    slur_token = detokenize(slur_samples[i:i+1], self.inv_slur_vocab)[0]
                    position_token = detokenize(position_samples[i:i+1], self.inv_position_vocab)[0]

                    symbol = EncodedSymbol(
                        rhythm=rhythm_token,
                        pitch=pitch_token,
                        lift=lift_token,
                        articulation=articulation_token,
                        slur=slur_token,
                        position=position_token,
                        coordinates=None,
                    )
                    symbols[orig_idx].append(symbol)

                # Break early if all remaining sequences just hit EOS
                if not keep_mask.any():
                    break

                # ---------------------------------------------------------
                # RESIZE THE MODEL INPUTS FOR THE NEXT STEP
                # ---------------------------------------------------------
                
                # Update the tracker with only the indices that survived
                active_indices = [orig for i, orig in enumerate(active_indices) if keep_mask[i]]
                
                # Slice all arrays to reduce the batch dimension (removes finished paths)
                x_rhythm_active = rhythm_samples[keep_mask]
                x_pitch_active = pitch_samples[keep_mask]
                x_lift_active = lift_samples[keep_mask]
                x_articulations_active = articulation_samples[keep_mask]
                x_slurs_active = slur_samples[keep_mask]
                
                context_active = context_active[keep_mask]
                context_reduced_active = context_reduced_active[keep_mask]
                
                # Slice the KV cache along the batch dimension
                cache_active = [c[keep_mask] for c in new_cache_active]

            return symbols


    def init_cache(self, cache_len: int = 0) -> tuple[list[NDArray], list[str], list[str]]:
        cache = []
        input_names = []
        output_names = []
        heads = self.config.decoder_heads
        head_dim = self.config.decoder_dim // heads
        for i in range(self.config.decoder_depth * 4):
            if self.fp16:  # the cache needs to be fp16 as well
                cache.append(
                    ort.OrtValue.ortvalue_from_numpy(
                        np.zeros((BATCH_SIZE, heads, cache_len, head_dim), dtype=np.float16),
                        "cuda" if self.use_gpu else "cpu",
                        self.device_id,
                    )
                )
            else:
                cache.append(
                    ort.OrtValue.ortvalue_from_numpy(
                        np.zeros((BATCH_SIZE, heads, cache_len, head_dim), dtype=np.float32),
                        "cuda" if self.use_gpu else "cpu",
                        self.device_id,
                    )
                )
            input_names.append(f"cache_in{i}")
            output_names.append(f"cache_out{i}")
        return cache, input_names, output_names


def detokenize(tokens: NDArray, vocab: dict[int, str]) -> list[list[str]]:
    result = []
    for row in tokens:
        toks = [vocab[tok.item()] for tok in row]
        toks = [t for t in toks if t not in ("[BOS]", "[EOS]", "[PAD]")]
        result.append(toks[0])
    return result


def get_decoder(config: Config) -> ScoreDecoder:
    """
    Returns Tromr's Decoder
    """
    use_gpu = False
    if config.use_gpu_inference:
        try:
            onnx_transformer = ort.InferenceSession(
                config.filepaths.decoder_path_fp16, providers=["CUDAExecutionProvider"]
            )
            fp16 = True
            # Sometimes Ort falls automatically back to the CPU EP
            # if so we get an error due to the device selection in init_cache()
            if "CUDAExecutionProvider" in onnx_transformer.get_providers():
                use_gpu = True
            else:
                eprint(
                    "Onnxruntime is not using GPU and therefore falling back to CPU. This is slow."
                )

        except Exception as ex:
            eprint(ex)
            eprint("Going on without GPU support")
            onnx_transformer = ort.InferenceSession(config.filepaths.decoder_path_fp16)
            fp16 = True

    else:
        onnx_transformer = ort.InferenceSession(config.filepaths.decoder_path)
        fp16 = False

    return ScoreDecoder(onnx_transformer, fp16, use_gpu, config=config)
