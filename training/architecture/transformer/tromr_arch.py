from typing import Any

import torch
from torch import nn
from x_transformers.x_transformers import LayerNorm as XTransformerLayerNorm

from homr.transformer.configs import Config
from homr.transformer.vocabulary import EncodedSymbol
from training.architecture.transformer.decoder import get_decoder
from training.architecture.transformer.encoder import get_encoder


def _cast_input_to_fp32(_module: nn.Module, args: tuple[Any, ...]) -> tuple[Any, ...] | None:
    if not args or not isinstance(args[0], torch.Tensor):
        return None
    x = args[0]
    if x.is_floating_point() and x.dtype != torch.float32:
        return (x.to(torch.float32), *args[1:])
    return None


def _cast_output_to_autocast_dtype(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> Any:
    if isinstance(output, torch.Tensor) and torch.is_autocast_enabled(output.device.type):
        return output.to(torch.get_autocast_dtype(output.device.type))
    return output


class TrOMR(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.encoder = get_encoder(config)
        self.decoder = get_decoder(config)
        self.config = config

    def eval_mode(self) -> None:
        self.decoder.eval()
        self.encoder.eval()

    def compile_modules(self, backend: str = "inductor") -> None:
        """Compile encoder and repeated decoder blocks separately in-place."""
        self.encoder.compile(backend=backend)
        for _norm, block, _residual in self.decoder.net.attn_layers.layers:
            block.compile(backend=backend)

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        """Enable or disable gradient checkpointing on encoder and decoder."""
        self.encoder.set_grad_checkpointing(enable)
        self.decoder.net.attn_layers.grad_checkpointing = enable

    def cast_layernorms_to_fp32(self) -> None:
        """
        Keep LayerNorm inputs/ops in fp32 under bfloat16 autocast and cast outputs back.
        Avoids mixed-dtype broadcast errors in torch_tpu's MLIR lowering of LayerNorm.
        """
        inner_lns: set[nn.Module] = {
            mod.ln for mod in self.modules() if isinstance(mod, XTransformerLayerNorm)
        }
        for mod in self.modules():
            if mod in inner_lns:
                continue
            if isinstance(mod, (nn.LayerNorm, XTransformerLayerNorm)):
                mod.register_forward_pre_hook(_cast_input_to_fp32)
                mod.register_forward_hook(_cast_output_to_autocast_dtype)

    def forward(
        self,
        inputs: torch.Tensor,
        rhythms: torch.Tensor,
        pitchs: torch.Tensor,
        lifts: torch.Tensor,
        articulations: torch.Tensor,
        slurs: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor,
        sampling_prob: float | torch.Tensor = 1.0,
        **kwargs: Any,
    ) -> Any:
        context = self.encoder(inputs)
        loss = self.decoder(
            rhythms=rhythms,
            pitchs=pitchs,
            lifts=lifts,
            articulations=articulations,
            slurs=slurs,
            positions=positions,
            context=context,
            mask=mask,
            sampling_prob=sampling_prob,
            **kwargs,
        )
        return loss

    @torch.no_grad()
    def generate(self, x: torch.Tensor) -> list[EncodedSymbol]:
        start_token = torch.tensor([[1]], dtype=torch.long, device=x.device)
        nonote_token = torch.tensor([[0]], dtype=torch.long, device=x.device)

        context = self.encoder(x)
        out = self.decoder.generate(start_token, nonote_token, context=context)

        return out

    def freeze_decoder(self) -> None:
        """Freeze all decoder parameters to prevent updates during training."""
        for param in self.decoder.parameters():
            param.requires_grad = False

    def freeze_encoder(self) -> None:
        """Freeze all encoder parameters to prevent updates during training."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def freeze_backbone(self) -> None:
        """Freeze only the encoder backbone."""
        if hasattr(self.encoder, "freeze_backbone"):
            self.encoder.freeze_backbone()

    def unfreeze_backbone(self) -> None:
        """Unfreeze the encoder backbone."""
        if hasattr(self.encoder, "unfreeze_backbone"):
            self.encoder.unfreeze_backbone()

    def unfreeze_lift_decoder(self) -> None:
        for param in self.decoder.net.lift_emb.parameters():
            param.requires_grad = True
        for param in self.decoder.net.to_logits_lift.parameters():
            param.requires_grad = True


def load_model(config: Config) -> TrOMR:
    """Load model from checkpoint."""
    model = TrOMR(config)
    checkpoint_path = config.filepaths.checkpoint
    if checkpoint_path.endswith(".safetensors"):
        import safetensors  # noqa: PLC0415

        tensors = {}
        with safetensors.safe_open(checkpoint_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        model.load_state_dict(tensors, strict=False)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=device, weights_only=True), strict=False
        )
    if torch.cuda.is_available():
        model.to(torch.device("cuda"))
    return model
