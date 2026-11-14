# S101: assert is intentional for distributed init preconditions
# flake8: noqa: S101

import os
from datetime import timedelta

import torch

try:
    import torch_tpu  # noqa: F401
except ImportError:
    pass


def use_tpu() -> bool:
    """
    True on a TPU with torch_tpu installed, e.g. a Colab TPU v5e-1 runtime.
    """
    if not hasattr(torch, "accelerator"):
        return False
    acc = torch.accelerator.current_accelerator()
    return acc is not None and acc.type == "tpu" and torch.accelerator.is_available()


def _configure_transformers_for_tpu() -> None:
    """
    Bridges torch_tpu into transformers 4.x / accelerate 1.x, whose device and
    DynamoBackend lists do not yet include PrivateUse1 ("tpu").
    """
    import accelerate.utils.dataclasses as acc_dataclasses  # noqa: PLC0415

    os.environ.setdefault("ACCELERATE_TORCH_DEVICE", "tpu:0")
    if "TPU" not in acc_dataclasses.DynamoBackend.__members__:
        from enum import Enum  # noqa: PLC0415

        members = {m.name: m.value for m in acc_dataclasses.DynamoBackend}
        members["TPU"] = "TPU"
        acc_dataclasses.DynamoBackend = Enum(  # type: ignore[misc]
            "DynamoBackend", members, type=acc_dataclasses.BaseEnum
        )


class Distribute:
    def __init__(self) -> None:
        self.tpu = use_tpu()
        if self.tpu:
            _configure_transformers_for_tpu()

        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size <= 1:
            self.enabled = False
            self.rank = 0
            self.world_size = 1
            self._device = (
                torch.device("tpu:0")
                if self.tpu
                else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            )
            return

        assert torch.distributed.is_available()
        assert not torch.distributed.is_initialized()

        if self.tpu:
            # Each torch_tpu worker process is bound to its own TPU chip (tpu:0)
            self._device = torch.device("tpu:0")
            torch.distributed.init_process_group(backend="tpu_dist", timeout=timedelta(hours=3))
        else:
            assert torch.cuda.is_available()
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            torch.cuda.set_device(local_rank)
            self._device = torch.device(f"cuda:{local_rank}")
            torch.distributed.init_process_group(
                backend="nccl", device_id=self._device, timeout=timedelta(hours=3)
            )
        self.enabled = True
        self.rank = torch.distributed.get_rank()
        self.world_size = world_size

    def barrier(self) -> None:
        if not self.enabled:
            return
        if self.tpu:
            torch.distributed.barrier()
            return
        torch.distributed.barrier(device_ids=[self._device.index])

    def all_reduce_sum(self, d: dict[str, float]) -> None:
        """Sums the values of d over all processes, in place."""
        if not self.enabled or not d:
            return

        keys = sorted(d.keys())
        dtype = torch.float32 if self.tpu else torch.float64
        tensor = torch.tensor([float(d[k]) for k in keys], dtype=dtype, device=self._device)
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        for k, v in zip(keys, tensor.tolist(), strict=True):
            d[k] = v

    def destroy(self) -> None:
        if not self.enabled:
            return
        torch.distributed.destroy_process_group()

    def is_rank0(self) -> bool:
        if not self.enabled:
            return True
        return self.rank == 0

    def get_world_size(self) -> int:
        if not self.enabled:
            return 1
        return self.world_size
