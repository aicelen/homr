import os
from collections import defaultdict
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import torch
from torch import Tensor
from transformers import Trainer

from training.transformer.distribute import Distribute
from training.transformer.training_vocabulary import map_rhythm_symbol_with_position

LOSS_COMPONENTS = [
    "loss_rhythm",
    "loss_pitch",
    "loss_lift",
    "loss_consist",
    "loss_position",
    "loss_articulations",
    "loss_slurs",
]


class HomrTrainer(Trainer):
    """
    Custom Trainer for HOMR transformer.
    Tracks per-branch losses, per-branch token accuracy,
    and overall token-weighted (micro) accuracy.
    """

    def __init__(
        self, *args: Any, distribute: Distribute, tpu_bf16: bool = False, **kwargs: Any
    ) -> None:
        """
        tpu_bf16: Runs the forward pass under bf16 autocast on TPUs. Used instead of
        TrainingArguments.bf16, because transformers 4.x rejects bf16=True when neither
        CUDA nor torch_xla is present.
        """
        super().__init__(*args, **kwargs)

        self._distribute = distribute
        self._tpu_bf16 = tpu_bf16

        # Training loss accumulators, kept on the device to avoid a sync per step
        self._loss_sum: dict[str, Tensor | float] = defaultdict(float)
        self._loss_count: dict[str, int] = defaultdict(int)

        # Evaluation accumulators, kept on the device to avoid a sync per step
        self._eval_loss_sum: dict[str, Tensor | float] = defaultdict(float)
        self._eval_loss_count: dict[str, int] = defaultdict(int)
        self._acc_correct: dict[str, Tensor | float] = defaultdict(float)
        self._acc_total: dict[str, Tensor | float] = defaultdict(float)

        self._rhythm_map = map_rhythm_symbol_with_position()

    def _autocast(self) -> AbstractContextManager[Any]:
        if self._tpu_bf16:
            return torch.autocast(device_type="tpu", dtype=torch.bfloat16)
        return nullcontext()

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if model.training:
            # Calculate sampling probability
            config = getattr(model, "config", None)
            if config:
                start_p = config.scheduled_sampling_start_prob
                end_p = config.scheduled_sampling_end_prob
                decay = config.scheduled_sampling_decay_steps
                step = self.state.global_step
                sampling_prob = max(end_p, start_p - (start_p - end_p) * (step / decay))
                if sampling_prob < 1.0:
                    # A tensor, so torch.compile doesn't recompile for every new value
                    inputs["sampling_prob"] = torch.tensor(
                        sampling_prob, device=inputs["rhythms"].device
                    )

        with self._autocast():
            outputs = model(**inputs)
        loss = outputs["loss"]

        if model.training:
            for component in LOSS_COMPONENTS:
                self._loss_sum[component] += outputs[component].detach()
                self._loss_count[component] += 1

        if return_outputs:
            return loss, outputs
        return loss

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:

        inputs = self._prepare_inputs(inputs)

        with torch.no_grad(), self._autocast():
            outputs = model(**inputs)

        # Accumulate per-branch eval losses
        for k in LOSS_COMPONENTS:
            self._eval_loss_sum[k] += outputs[k].detach().float()
            self._eval_loss_count[k] += 1

        # Accumulate per-branch accuracy stats
        self._accumulate_accuracy(outputs, inputs)

        if not self._distribute.enabled and not self._distribute.tpu:
            loss = outputs["loss"].mean().detach().cpu()
        else:
            # when using distributed training, loss on each device should be gathered,
            # which can only run on the device. On TPU, moving it to the CPU would sync
            # every step.
            loss = outputs["loss"].mean().detach()
        return loss, None, None

    def _accumulate_accuracy(
        self,
        outputs: dict[str, Any],
        inputs: dict[str, torch.Tensor],
    ) -> None:
        rhythm, pitch, lift, position, articulations, slurs = outputs["logits"]
        # Pull binary mask from inputs and align with y_out (shift by 1)
        eval_mask = inputs["mask"][:, 1:]

        rhythms_output = inputs["rhythms"][:, 1:]
        _rhythm_map_on_device = self._rhythm_map.to(rhythms_output.device)
        rhythms_output_clamped = rhythms_output.clamp(min=0, max=_rhythm_map_on_device.shape[0] - 1)
        is_positional = _rhythm_map_on_device[rhythms_output_clamped]

        branches = [
            ("rhythm", rhythm, inputs["rhythms"]),
            ("pitch", pitch, inputs["pitchs"]),
            ("lift", lift, inputs["lifts"]),
            ("position", position, inputs["positions"]),
            ("articulations", articulations, inputs["articulations"]),
            ("slurs", slurs, inputs["slurs"]),
        ]

        for name, logits, input_labels in branches:
            preds = logits.argmax(dim=-1)
            labels = input_labels[:, 1:]  # next-token alignment

            min_len = min(preds.shape[1], labels.shape[1], eval_mask.shape[1])
            preds = preds[:, :min_len]
            labels = labels[:, :min_len]
            current_eval_mask = eval_mask[:, :min_len]

            # Combine explicit ignore_index with the binary mask.
            # Branches without tokens add zero, evaluation_loop skips them.
            mask = (labels != -100) & current_eval_mask

            self._acc_correct[name] += ((preds == labels) & mask).sum()
            self._acc_total[name] += mask.sum()

            if name == "pitch" and os.environ.get("DEBUG_PITCH_RHYTHM_ACCURACY", "0") == "1":
                current_is_pos = is_positional[:, :min_len]
                is_correct = preds == labels
                is_pos = mask & current_is_pos
                is_nonpos = mask & (~current_is_pos)
                self._acc_correct["pitch_posRhy"] += (is_correct & is_pos).sum()
                self._acc_total["pitch_posRhy"] += is_pos.sum()
                self._acc_correct["pitch_nonposRhy"] += (is_correct & is_nonpos).sum()
                self._acc_total["pitch_nonposRhy"] += is_nonpos.sum()

    def _to_floats(self, d: dict[str, Tensor | float]) -> dict[str, float]:
        """Moves the accumulated values to the CPU, one sync per evaluation."""
        result = {k: float(v) for k, v in d.items()}
        self._distribute.all_reduce_sum(result)
        return result

    def evaluation_loop(self, *args, metric_key_prefix="eval", **kwargs):  # type: ignore
        # Reset eval accumulators
        self._eval_loss_sum.clear()
        self._eval_loss_count.clear()
        self._acc_correct.clear()
        self._acc_total.clear()

        output = super().evaluation_loop(
            *args,
            metric_key_prefix=metric_key_prefix,
            **kwargs,
        )

        eval_loss_sum = self._to_floats(self._eval_loss_sum)
        eval_loss_count = self._to_floats(dict(self._eval_loss_count))
        acc_correct = self._to_floats(self._acc_correct)
        acc_total = self._to_floats(self._acc_total)

        # Per-component eval losses
        for k in LOSS_COMPONENTS:
            if eval_loss_count.get(k, 0) > 0:
                output.metrics[f"{metric_key_prefix}_{k}"] = eval_loss_sum[k] / eval_loss_count[k]

        # Per-branch accuracies
        for k in acc_total:
            if acc_total[k] > 0:
                output.metrics[f"{metric_key_prefix}_{k}_accuracy"] = acc_correct[k] / acc_total[k]

        # Overall token-weighted (micro) accuracy
        total_correct = sum(acc_correct.values())
        total_tokens = sum(acc_total.values())
        if total_tokens > 0:
            output.metrics[f"{metric_key_prefix}_accuracy"] = total_correct / total_tokens

        return output

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        if self._loss_count and "loss" in logs:
            for component in LOSS_COMPONENTS:
                if self._loss_count[component] > 0:
                    loss_sum = float(self._loss_sum[component])
                    logs[component] = loss_sum / self._loss_count[component]
            self._loss_sum.clear()
            self._loss_count.clear()

        super().log(logs, start_time)
