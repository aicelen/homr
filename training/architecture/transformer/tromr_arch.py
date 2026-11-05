from typing import Any

import torch
from torch import nn

from homr.transformer.configs import Config
from homr.transformer.vocabulary import EncodedSymbol
from training.architecture.transformer.decoder import get_decoder
from training.architecture.transformer.encoder import get_encoder
from homr.simple_logging import eprint
class TrOMR(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.encoder = get_encoder(config)
        self.decoder = get_decoder(config)
        self.config = config
        self.pretrained_decoder_params: set[str] = set()

    def eval_mode(self) -> None:
        self.decoder.eval()
        self.encoder.eval()

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
        sampling_prob: float = 1.0,
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


    def _set_pretrained_decoder_grad(self, requires_grad: bool) -> None:
        params = dict(self.named_parameters())
        for name in self.pretrained_decoder_params:
            params[name].requires_grad = requires_grad

    def freeze_pretrained_decoder(self) -> None:
        """Freeze all pertrained decoder parameters."""
        self._set_pretrained_decoder_grad(False)


    def unfreeze_pretrained_decoder(self) -> None:
        """Unfreeze all pertrained decoder parameters."""
        self._set_pretrained_decoder_grad(True)

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
        with safetensors.safe_open(checkpoint_path, framework="pt", device=0) as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        model.load_state_dict(tensors, strict=False)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=device, weights_only=True), strict=False
        )
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return model


def remap_pretrained_decoder(state_dict_weights: dict, model: nn.Module) -> dict:
    PREFIX = "net.attn_layers.layers."

    # (Attention -> cross-attention -> feed forward) x 8
    # weights do not contain cross attention
    layer_types = model.decoder.net.attn_layers.layer_types
    attn_idx = [i for i, t in enumerate(layer_types) if t == "a"]
    ff_idx = [i for i, t in enumerate(layer_types) if t == "f"]

    new_sd = {}
    for k, v in state_dict_weights.items():
        if k.startswith(PREFIX):
            idx, rest = k[len(PREFIX):].split(".", 1)
            i = int(idx)
            # even: attention, odd: feedforward
            layer_index = attn_idx[i // 2] if i % 2 == 0 else ff_idx[i // 2]
            k = f"{PREFIX}{layer_index}.{rest}"

        k = "decoder." + k

        new_sd[k] = v
    return new_sd

def load_model_from_pretrained_decoder(config: Config) -> TrOMR:
    model = TrOMR(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = torch.load(config.filepaths.pretrained_decoder, map_location=device, weights_only=True)

    state_dict = remap_pretrained_decoder(weights, model)
    result = model.load_state_dict(state_dict, strict=False)

    eprint("Unexpected:", result.unexpected_keys)
    missing = set(result.missing_keys)
    model.pretrained_decoder_params = {
        name
        for name, _ in model.named_parameters()
        if name.startswith("decoder.") and name not in missing
    }

    model.to(device)
    return model

if __name__ == "__main__":
    config = Config()
    load_model_from_pretrained_decoder(config)
