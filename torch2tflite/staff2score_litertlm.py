"""Convert TrOMR (staff2score) PyTorch weights to LiteRT-LM (.litertlm).

This exports two LiteRT signatures:
- `prefill`: image -> first-step token ids + KV cache
- `decode`: one decode step -> next token ids + updated KV cache

The exported graph is CPU-focused and avoids Python generation loops.
"""

from __future__ import annotations

import argparse
import inspect
import os
import tempfile
import textwrap
from pathlib import Path
from typing import Any

import torch

# Ensure project root is importable when executed as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from homr.transformer.configs import Config
from training.architecture.transformer.staff2score import load_model_weights
from training.architecture.transformer.tromr_arch import TrOMR


def _patch_torch_accelerator_for_ai_edge_torch() -> None:
    """Work around torch.accelerator backend registration issues on CPU setups."""
    if hasattr(torch, "accelerator") and hasattr(torch.accelerator, "current_device_index"):
        torch.accelerator.current_device_index = lambda: 0  # type: ignore[assignment]


def _patch_x_transformers_attend_for_export() -> None:
    """
    Patch x_transformers Attend.forward to avoid data-dependent Python branching
    that breaks torch.export with symbolic masks.
    """
    import x_transformers.attend as attend_mod

    if getattr(attend_mod.Attend.forward, "_homr_export_patched", False):
        return

    source = inspect.getsource(attend_mod.Attend.forward)
    source = source.replace(
        "if exists(row_is_entirely_masked) and row_is_entirely_masked.any():",
        "if exists(row_is_entirely_masked):",
    )
    namespace: dict[str, Any] = {}
    exec(  # noqa: S102
        textwrap.dedent(source),
        attend_mod.__dict__,
        namespace,
    )
    patched_forward = namespace["forward"]
    patched_forward._homr_export_patched = True  # type: ignore[attr-defined]
    attend_mod.Attend.forward = patched_forward


def _zero_cache(device: torch.device) -> list[torch.Tensor]:
    return [torch.zeros((1, 8, 0, 64), dtype=torch.float32, device=device) for _ in range(32)]


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


def _token_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits[:, -1, :], dim=-1, keepdim=True).to(torch.int64)


class Staff2ScorePrefill(torch.nn.Module):
    def __init__(self, model: TrOMR, config: Config) -> None:
        super().__init__()
        self.encoder = model.encoder
        self.decoder = model.decoder.net
        self.self_cache_len = config.max_seq_len - 1
        self.register_buffer(
            "start_rhythm",
            torch.tensor([[config.bos_token]], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "start_nonote",
            torch.tensor([[config.nonote_token]], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "cache_len_zero",
            torch.zeros((1,), dtype=torch.long),
            persistent=False,
        )
        self.eos_token = config.eos_token

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        context = self.encoder(image)
        out = self.decoder(
            rhythms=self.start_rhythm,
            pitchs=self.start_nonote,
            lifts=self.start_nonote,
            articulations=self.start_nonote,
            context=context,
            cache_len=self.cache_len_zero,
            mask=None,
            cache=_zero_cache(image.device),
            return_center_of_attention=False,
        )
        out_rhythm, out_pitch, out_lift, out_pos, out_art, _x, _attn, cache_out = out
        rhythm = _token_from_logits(out_rhythm)
        pitch = _token_from_logits(out_pitch)
        lift = _token_from_logits(out_lift)
        position = _token_from_logits(out_pos)
        articulation = _token_from_logits(out_art)
        eos = (rhythm == self.eos_token).to(torch.int64)
        self_cache_buffers: list[torch.Tensor] = []
        for index in SELF_CACHE_TENSOR_INDICES:
            full_cache = torch.zeros(
                (1, 8, self.self_cache_len, 64),
                dtype=torch.float32,
                device=image.device,
            )
            full_cache[:, :, 0:1, :] = cache_out[index][:, :, -1:, :]
            self_cache_buffers.append(full_cache)
        return rhythm, pitch, lift, position, articulation, eos, context, *self_cache_buffers


class Staff2ScoreDecode(torch.nn.Module):
    def __init__(self, model: TrOMR, config: Config) -> None:
        super().__init__()
        self.decoder = model.decoder.net
        self.self_cache_len = config.max_seq_len - 1
        self.register_buffer(
            "cross_cache_empty",
            torch.zeros((1, 8, 0, 64), dtype=torch.float32),
            persistent=False,
        )
        self.eos_token = config.eos_token

    def forward(
        self,
        rhythms: torch.Tensor,
        pitchs: torch.Tensor,
        lifts: torch.Tensor,
        articulations: torch.Tensor,
        context: torch.Tensor,
        cache_len: torch.Tensor,
        self_cache0: torch.Tensor,
        self_cache1: torch.Tensor,
        self_cache2: torch.Tensor,
        self_cache3: torch.Tensor,
        self_cache4: torch.Tensor,
        self_cache5: torch.Tensor,
        self_cache6: torch.Tensor,
        self_cache7: torch.Tensor,
        self_cache8: torch.Tensor,
        self_cache9: torch.Tensor,
        self_cache10: torch.Tensor,
        self_cache11: torch.Tensor,
        self_cache12: torch.Tensor,
        self_cache13: torch.Tensor,
        self_cache14: torch.Tensor,
        self_cache15: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        self_cache_in = [
            self_cache0,
            self_cache1,
            self_cache2,
            self_cache3,
            self_cache4,
            self_cache5,
            self_cache6,
            self_cache7,
            self_cache8,
            self_cache9,
            self_cache10,
            self_cache11,
            self_cache12,
            self_cache13,
            self_cache14,
            self_cache15,
        ]
        cache_in: list[torch.Tensor] = []
        self_cache_i = 0
        for index in range(32):
            if index in SELF_CACHE_TENSOR_INDICES:
                cache_in.append(self_cache_in[self_cache_i])
                self_cache_i += 1
            else:
                cache_in.append(self.cross_cache_empty)

        kv_mask = torch.arange(self.self_cache_len, device=cache_len.device).unsqueeze(0)
        kv_mask = kv_mask < cache_len.unsqueeze(1)
        self_attn_kv_mask = torch.cat(
            [kv_mask, torch.ones((1, 1), dtype=torch.bool, device=cache_len.device)], dim=1
        )

        out = self.decoder(
            rhythms=rhythms,
            pitchs=pitchs,
            lifts=lifts,
            articulations=articulations,
            context=context,
            cache_len=cache_len,
            mask=None,
            cache=cache_in,
            self_attn_kv_mask=self_attn_kv_mask,
            return_center_of_attention=False,
        )
        out_rhythm, out_pitch, out_lift, out_pos, out_art, _x, _attn, cache_out = out
        rhythm = _token_from_logits(out_rhythm)
        pitch = _token_from_logits(out_pitch)
        lift = _token_from_logits(out_lift)
        position = _token_from_logits(out_pos)
        articulation = _token_from_logits(out_art)
        eos = (rhythm == self.eos_token).to(torch.int64)
        self_cache_updates = [cache_out[index][:, :, -1:, :] for index in SELF_CACHE_TENSOR_INDICES]
        return rhythm, pitch, lift, position, articulation, eos, *self_cache_updates


def _quant_config_from_name(name: str) -> Any | None:
    if name == "none":
        return None
    if name == "dynamic_int8":
        from ai_edge_torch.generative.quantize import quant_recipes

        return quant_recipes.full_dynamic_recipe()
    raise ValueError(f"Unsupported quantization: {name}")


def _load_trained_model(config: Config, checkpoint_path: str | None) -> TrOMR:
    if checkpoint_path:
        config.filepaths.checkpoint = checkpoint_path
    model = TrOMR(config)
    model.eval_mode()
    model.load_state_dict(load_model_weights(config.filepaths.checkpoint), strict=False)
    model.to(torch.device("cpu"))
    # Flash attention uses data-dependent control flow that breaks export
    # when cache validity masks are dynamic.
    for module in model.modules():
        if hasattr(module, "flash"):
            setattr(module, "flash", False)
    return model


def _decode_sample_kwargs_from_prefill(
    prefill_out: tuple[torch.Tensor, ...],
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {
        "rhythms": prefill_out[0],
        "pitchs": prefill_out[1],
        "lifts": prefill_out[2],
        "articulations": prefill_out[4],
        "context": prefill_out[6],
        "cache_len": torch.ones((1,), dtype=torch.long),
    }
    for i, cache in enumerate(prefill_out[7:]):
        out[f"self_cache{i}"] = cache
    return out


def convert_staff2score_to_litertlm(
    output_dir: str,
    output_prefix: str,
    checkpoint_path: str | None,
    rhythm_tokenizer_path: str,
    quantize: str,
) -> tuple[str, str]:
    _patch_torch_accelerator_for_ai_edge_torch()
    _patch_x_transformers_attend_for_export()

    import ai_edge_torch
    from ai_edge_torch.generative.utilities import litertlm_builder

    config = Config()
    model = _load_trained_model(config, checkpoint_path)

    prefill_module = Staff2ScorePrefill(model, config).eval()
    decode_module = Staff2ScoreDecode(model, config).eval()

    sample_image = torch.zeros((1, 1, config.max_height, config.max_width), dtype=torch.float32)
    with torch.no_grad():
        prefill_sample_out = prefill_module(sample_image)
    decode_kwargs = _decode_sample_kwargs_from_prefill(prefill_sample_out)

    converter = ai_edge_torch.signature(
        "prefill",
        prefill_module,
        sample_kwargs={"image": sample_image},
    )
    converter.signature(
        "decode",
        decode_module,
        sample_kwargs=decode_kwargs,
    )

    edge_model = converter.convert(
        strict_export=False,
        quant_config=_quant_config_from_name(quantize),
    )

    os.makedirs(output_dir, exist_ok=True)
    tflite_path = os.path.join(output_dir, f"{output_prefix}.tflite")
    edge_model.export(tflite_path)

    with tempfile.TemporaryDirectory() as workdir:
        litertlm_builder.build_litertlm(
            tflite_model_path=tflite_path,
            workdir=workdir,
            output_path=output_dir,
            context_length=config.max_seq_len,
            hf_tokenizer_model_path=rhythm_tokenizer_path,
            start_token_id=config.bos_token,
            stop_token_ids=[config.eos_token],
        )

    litertlm_path = os.path.join(output_dir, f"{output_prefix}.litertlm")
    return tflite_path, litertlm_path


def parse_args() -> argparse.Namespace:
    config = Config()
    parser = argparse.ArgumentParser(
        description="Convert staff2score PyTorch model to LiteRT-LM (.litertlm).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path("torch2tflite") / "out"),
        help="Directory for generated files.",
    )
    parser.add_argument(
        "--output-prefix",
        default="staff2score_cpu",
        help="Base filename prefix for output artifacts.",
    )
    parser.add_argument(
        "--checkpoint",
        default=config.filepaths.checkpoint,
        help="Path to PyTorch checkpoint (.pth or .safetensors).",
    )
    parser.add_argument(
        "--rhythm-tokenizer",
        default=config.filepaths.rhythmtokenizer,
        help="Path to rhythm tokenizer JSON (used as HF tokenizer payload in .litertlm).",
    )
    parser.add_argument(
        "--quantize",
        choices=["none", "dynamic_int8"],
        default="dynamic_int8",
        help="Post-training quantization mode for TFLite export.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tflite_path, litertlm_path = convert_staff2score_to_litertlm(
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        checkpoint_path=args.checkpoint,
        rhythm_tokenizer_path=args.rhythm_tokenizer,
        quantize=args.quantize,
    )
    print(f"Generated TFLite:   {tflite_path}")  # noqa: T201
    print(f"Generated LiteRTLM: {litertlm_path}")  # noqa: T201


if __name__ == "__main__":
    main()
