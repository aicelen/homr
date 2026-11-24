# S101: assert is intentional for distributed init preconditions
# flake8: noqa: S101

import os
from datetime import timedelta

import torch


def use_tpu() -> bool:
    """
    True on a TPU with torch_tpu installed, e.g. a TPU v5e-1 or v5e-8 runtime.
    """
    try:
        import torch_tpu  # noqa: F401, PLC0415
    except ImportError:
        pass

    if not hasattr(torch, "accelerator"):
        return False
    acc = torch.accelerator.current_accelerator()
    return acc is not None and acc.type == "tpu" and torch.accelerator.is_available()


def configure_accelerate_for_tpu_ddp(
    rank: int,
    local_rank: int,
    world_size: int,
    backend: str = "tpu_dist",
    device: torch.device | None = None,
) -> None:
    """
    accelerate 1.x PartialState._prepare_backend() does not yet recognize PrivateUse1 ("tpu"),
    falling back to DistributedType.NO (world_size=1) and CUDA-style DDP device_ids=[local_rank].
    Pre-configure PartialState for use_configured_state=True so dataloaders are sharded and
    DistributedDataParallel uses device_ids=None.
    """
    from accelerate.state import (  # noqa: PLC0415
        AcceleratorState,
        DistributedType,
        PartialState,
    )

    os.environ.setdefault("ACCELERATE_BYPASS_DEVICE_MAP", "true")
    AcceleratorState._reset_state(reset_partial_state=True)
    PartialState._shared_state.update(
        {
            "_cpu": False,
            "backend": backend,
            "debug": False,
            "device": device if device is not None else torch.device("tpu:0"),
            "distributed_type": DistributedType.MULTI_GPU,
            "fork_launched": False,
            "local_process_index": local_rank,
            "num_processes": world_size,
            "process_index": rank,
        }
    )


class Distribute:
    def __init__(self) -> None:
        self.tpu = use_tpu()
        if self.tpu:
            os.environ.setdefault("ACCELERATE_TORCH_DEVICE", "tpu:0")
            # Prevent spawned DataLoader workers from autoloading torch_tpu on `import torch`,
            # which otherwise fails in worker torch.manual_seed() because the main process
            # already holds the TPU PJRT lock.
            os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

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

        # Prevent spawned DataLoader workers across multiple ranks from oversubscribing CPU threads
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")

        assert torch.distributed.is_available()
        assert not torch.distributed.is_initialized()

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if self.tpu:
            if "TORCH_TPU_TOPOLOGY" not in os.environ:
                raise RuntimeError(
                    "TORCH_TPU_TOPOLOGY is not set for multi-chip TPU training. "
                    "Launch via `python3 -m training.train transformer` or run "
                    "`eval $(python3 -m torch.tpu.distributed | sed 's/^/export /')` "
                    "before `torchrun`."
                )
            # Each torch_tpu worker process is bound to its own TPU chip (tpu:0)
            self._device = torch.device("tpu:0")
            torch.distributed.init_process_group(backend="tpu_dist", timeout=timedelta(hours=3))
            self.rank = torch.distributed.get_rank()
            configure_accelerate_for_tpu_ddp(
                rank=self.rank,
                local_rank=local_rank,
                world_size=world_size,
                backend="tpu_dist",
                device=self._device,
            )
        else:
            assert torch.cuda.is_available()
            torch.cuda.set_device(local_rank)
            self._device = torch.device(f"cuda:{local_rank}")
            torch.distributed.init_process_group(
                backend="nccl", device_id=self._device, timeout=timedelta(hours=3)
            )
            self.rank = torch.distributed.get_rank()
        self.enabled = True
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
