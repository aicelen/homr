import os
import shutil
import sys
from typing import Any

import torch
import torch._dynamo
from transformers import (
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from homr.simple_logging import eprint
from homr.transformer.configs import Config
from training.architecture.transformer.tromr_arch import TrOMR, load_model
from training.omr_datasets.convert_grandstaff import (
    convert_grandstaff,
    grandstaff_train_index,
)
from training.omr_datasets.convert_lieder import convert_lieder, lieder_train_index
from training.omr_datasets.convert_pdmx import convert_pdmx, pdmx_train_index
from training.omr_datasets.filter_from_index import (
    read_invalid_tokens,
    without_invalid_tokens,
    write_invalid_tokens,
)
from training.run_id import get_run_id
from training.transformer.data_loader import (
    collate_trimmed,
    filter_valid_samples,
    label_names,
    load_dataset,
)
from training.transformer.distribute import Distribute
from training.transformer.metrics import HomrTrainer
from training.transformer.mix_datasets import mix_training_sets

torch._dynamo.config.suppress_errors = True
# Batches are padded to multiples of SEQ_LEN_MULTIPLE tokens. Compile each length statically:
# Inductor fails on a dynamic sequence length and then falls back to eager mode.
# Up to 5 lengths, each for training, the no-grad pass and evaluation.
torch._dynamo.config.automatic_dynamic_shapes = False
torch._dynamo.config.recompile_limit = 64

TPU_TARGET_GLOBAL_BATCH_SIZE = 64


class FreezeCallback(TrainerCallback):
    """
    Callback to freeze the backbone for a set number of epochs.
    Standard practice is ~2 epochs.
    """

    def __init__(self, epochs_to_freeze: int = 2):
        self.epochs_to_freeze = epochs_to_freeze
        self._backbone_frozen = False

    def on_train_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        model = kwargs.get("model")
        if model and hasattr(model, "freeze_backbone"):
            eprint(f"Freezing backbone for the first {self.epochs_to_freeze} epochs")
            model.freeze_backbone()
            self._backbone_frozen = True

    def on_epoch_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        model = kwargs.get("model")
        if model and self._backbone_frozen and state.epoch and state.epoch >= self.epochs_to_freeze:
            eprint(f"Unfreezing backbone at epoch {state.epoch}")
            model.unfreeze_backbone()
            self._backbone_frozen = False


def load_training_index(file_path: str) -> list[str]:
    with open(file_path) as f:
        return f.readlines()


def check_data_source(all_file_paths: list[str]) -> bool:
    result = True
    for file_paths in all_file_paths:
        paths = file_paths.strip().split(",")
        for path in paths:
            if path == "nosymbols":
                continue
            if not os.path.exists(path):
                eprint(f"Index {file_paths} does not exist due to {path}")
                result = False
    return result


def load_and_mix_training_sets(
    index_paths: list[str],
    weights: list[float],
    number_of_files: int,
    invalid_tokens: set[str] | None = None,
) -> list[str]:
    if len(index_paths) != len(weights):
        eprint("Error: Number of index paths and weights do not match")
        sys.exit(1)
    data_sources = [
        without_invalid_tokens(load_training_index(index), invalid_tokens or set())
        for index in index_paths
    ]
    if not all(check_data_source(data) for data in data_sources):
        eprint("Error in datasets found")
        sys.exit(1)
    eprint(
        "Total number of training files to choose from", sum([len(data) for data in data_sources])
    )
    return mix_training_sets(data_sources, weights, number_of_files)


script_location = os.path.dirname(os.path.realpath(__file__))

git_root = os.path.join(script_location, "..", "..")


def _check_datasets_are_present(selected_datasets: list[str], invalid_tokens_file: str) -> None:
    for dataset in selected_datasets:
        if dataset == grandstaff_train_index and not os.path.exists(grandstaff_train_index):
            convert_grandstaff()

        if dataset == lieder_train_index and not os.path.exists(lieder_train_index):
            convert_lieder()

        if dataset == pdmx_train_index and not os.path.exists(pdmx_train_index):
            convert_pdmx()
    write_invalid_tokens(invalid_tokens_file)


def compute_training_schedule(
    tpu: bool,
    world_size: int,
    cpu_count: int | None = None,
    batch_size_override: int | None = None,
    grad_accum_override: int | None = None,
    workers_override: int | None = None,
) -> tuple[int, int, int, int]:
    """
    Returns (per_device_train_batch_size, per_device_eval_batch_size,
    gradient_accumulation_steps, dataloader_num_workers).
    On TPU, keeps global batch size at 64 across v5e-1 (16*4*1) and v5e-8 (8*1*8)
    so optimizer step counts and the 45k-step scheduled-sampling decay match.
    """
    cpus = cpu_count if cpu_count is not None else (os.cpu_count() or 16)
    if tpu:
        default_batch = min(16, max(1, TPU_TARGET_GLOBAL_BATCH_SIZE // world_size))
        batch_size = batch_size_override if batch_size_override is not None else default_batch
        default_grad_accum = max(1, TPU_TARGET_GLOBAL_BATCH_SIZE // (batch_size * world_size))
        grad_accum = grad_accum_override if grad_accum_override is not None else default_grad_accum
        eval_batch_size = 16
        max_workers_per_rank = 20 if world_size == 1 else 12
        default_workers = min(max_workers_per_rank, max(2, cpus // world_size - 2))
    else:
        batch_size = batch_size_override if batch_size_override is not None else 16
        grad_accum = (
            grad_accum_override if grad_accum_override is not None else max(1, 2 // world_size)
        )
        eval_batch_size = 32
        default_workers = min(12, max(2, cpus // world_size))
    workers = workers_override if workers_override is not None else default_workers
    return batch_size, eval_batch_size, grad_accum, workers


def _build_model(config: Config, fine_tune: bool) -> TrOMR:
    if fine_tune:
        eprint("Fine tuning model from", config.filepaths.checkpoint)
        model = load_model(config)
        model.freeze_encoder()
        model.freeze_decoder()
        model.unfreeze_lift_decoder()
        return model
    return TrOMR(config)


def train_transformer(
    fp32: bool = False,
    resume: str = "",
    smoke_test: bool = False,
    fine_tune: bool = False,
    batch_size_override: int | None = None,
    grad_accum_override: int | None = None,
    workers_override: int | None = None,
) -> None:
    distribute = Distribute()
    tpu = distribute.tpu
    world_size = distribute.get_world_size()
    tpu_bf16 = tpu and not fp32
    if tpu:
        torch.set_float32_matmul_precision("highest" if fp32 else "medium")
        eprint(f"Training on TPU with {world_size} device(s)")

    number_of_epochs = 35
    if smoke_test:
        number_of_epochs = 10
    elif fine_tune:
        number_of_epochs = 15
    resume_from_checkpoint = None

    checkpoint_folder = "current_training"
    checkpoint_dir = os.path.join(git_root, checkpoint_folder)
    if resume:
        resume_from_checkpoint = os.path.join(checkpoint_dir, resume)
    elif os.path.exists(checkpoint_dir) and distribute.is_rank0():
        shutil.rmtree(checkpoint_dir)

    # PrIMuS isn't used: it has no beams and its Hugging Face copy is CC BY-NC (non-commercial)
    # MuseTrainer isn't used: 1k samples, it adds little next to PDMX and has no license file
    dataset_index = [
        lieder_train_index,
        grandstaff_train_index,
        pdmx_train_index,
    ]
    dataset_weights = [1.0, 1.0, 1.0]
    invalid_tokens_file = os.path.join(checkpoint_dir, "invalid_tokens.txt")
    filtered_samples_file = os.path.join(checkpoint_dir, "filtered_samples.txt")
    number_of_files = -1

    if distribute.is_rank0():
        os.makedirs(checkpoint_dir, exist_ok=True)
        _check_datasets_are_present(dataset_index, invalid_tokens_file)
        mixed_index = load_and_mix_training_sets(
            dataset_index,
            dataset_weights,
            number_of_files,
            read_invalid_tokens(invalid_tokens_file),
        )
        filtered_samples = filter_valid_samples(mixed_index)
        with open(filtered_samples_file, "w") as f:
            f.writelines(filtered_samples)
    distribute.barrier()

    with open(filtered_samples_file) as f:
        filtered_samples = f.readlines()

    config = Config()
    datasets = load_dataset(
        filtered_samples,
        config,
        val_split=0.1,
        validation_cache_dir=os.path.join(checkpoint_dir, "validation_cache"),
        pre_filtered=True,
    )

    compile_threshold = 50000
    compile_model = (
        number_of_files < 0 or number_of_files * number_of_epochs >= compile_threshold
    )  # Compiling needs time, but pays off for large datasets
    if compile_model and distribute.is_rank0():
        eprint("Compiling model")

    run_id = get_run_id()

    batch_size, eval_batch_size, grad_accum, num_workers = compute_training_schedule(
        tpu=tpu,
        world_size=world_size,
        batch_size_override=batch_size_override,
        grad_accum_override=grad_accum_override,
        workers_override=workers_override,
    )
    val_len = len(datasets["validation"])
    if tpu and val_len > 0 and val_len < eval_batch_size * world_size:
        eval_batch_size = max(1, val_len // world_size)

    if distribute.is_rank0():
        eprint(
            f"Batch schedule: per_device={batch_size}, grad_accum={grad_accum}, "
            f"world_size={world_size}, global_batch={batch_size * grad_accum * world_size}, "
            f"workers_per_rank={num_workers}"
        )

    accelerator_config = {"use_configured_state": True} if (tpu and distribute.enabled) else None
    ddp_find_unused = (not fine_tune) if distribute.enabled else None

    train_args = TrainingArguments(
        checkpoint_folder,
        torch_compile=False,  # compiled via model.compile_modules()
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-5 if fine_tune else 1e-4,
        optim="adamw_torch" if tpu else "adamw_torch_fused",  # TPU has no CUDA fused AdamW
        gradient_accumulation_steps=grad_accum,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=number_of_epochs,
        weight_decay=0.05,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        report_to=["tensorboard"],
        logging_dir=os.path.join("logs", f"run{run_id}"),
        label_names=label_names,
        # On TPU, TrainingArguments(bf16=True) rejects torch_tpu, so HomrTrainer
        # enables torch.autocast("tpu", dtype=torch.bfloat16) via tpu_bf16 instead
        bf16=not fp32 and not tpu,
        logging_nan_inf_filter=False,  # the filter syncs with the accelerator every step
        dataloader_pin_memory=not tpu,
        # TPU compiles a new graph for every batch shape, a smaller last batch would add more
        dataloader_drop_last=tpu,
        dataloader_num_workers=num_workers,
        accelerator_config=accelerator_config,
        ddp_find_unused_parameters=ddp_find_unused,
        ddp_timeout=10800,
    )

    if distribute.enabled:
        # Rank 0 caches pretrained backbone weights first to avoid concurrent Hub downloads
        model: TrOMR | None = None
        if distribute.is_rank0():
            model = _build_model(config, fine_tune)
        distribute.barrier()
        if model is None:
            model = _build_model(config, fine_tune)
    else:
        model = _build_model(config, fine_tune)

    model.set_grad_checkpointing(True)
    if tpu_bf16:
        model.cast_layernorms_to_fp32()

    if compile_model:
        model.compile_modules(backend="tpu" if tpu else "inductor")

    model_name = "pytorch_model"

    model_destination = os.path.join(
        git_root, "training", "architecture", "transformer", f"{model_name}_{run_id}.pth"
    )

    if os.path.exists(model_destination):
        if distribute.is_rank0():
            eprint("Model already exists", model_destination)
        distribute.destroy()
        return

    try:
        callbacks: list[TrainerCallback] = [EarlyStoppingCallback(early_stopping_patience=5)]
        if not fine_tune:
            callbacks.append(FreezeCallback(epochs_to_freeze=2))

        trainer = HomrTrainer(
            model,
            train_args,
            train_dataset=datasets["train"],
            data_collator=collate_trimmed,
            eval_dataset=datasets["validation"],
            callbacks=callbacks,
            distribute=distribute,
            tpu_bf16=tpu_bf16,
        )

        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    except KeyboardInterrupt:
        eprint("Interrupted")
    if distribute.is_rank0():
        # Moved to the CPU, so that the weights also load without torch_tpu
        state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save(state_dict, model_destination)
        eprint(f"Saved model to {model_destination}")
    distribute.barrier()
    distribute.destroy()


if __name__ == "__main__":
    if "--fine" in sys.argv:
        train_transformer(fp32=False, fine_tune=True)
    elif len(sys.argv) > 1:
        raise ValueError("Unknown argument")
    else:
        train_transformer(smoke_test=True)
