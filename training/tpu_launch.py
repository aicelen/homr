import importlib.util
import json
import os
import subprocess
import sys

from homr.simple_logging import eprint

_TPU_ENV_KEYS = (
    "TORCH_TPU_TOPOLOGY",
    "TORCH_TPU_SLICEBUILDER_ADDRESSES",
    "MASTER_ADDR",
    "MASTER_PORT",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "NODE_RANK",
    "NNODES",
)


def query_tpu_launch_env() -> dict[str, str]:
    """
    Runs torch.tpu.distributed.set_tpu_launch_env() in a short-lived child process
    so the parent launcher never imports torch or locks the TPU PJRT client.
    """
    if importlib.util.find_spec("torch_tpu") is None:
        return {}

    script = (
        "import json, os, torch\n"
        "import torch_tpu\n"
        "from torch.tpu.distributed import set_tpu_launch_env\n"
        "set_tpu_launch_env()\n"
        f"keys = {_TPU_ENV_KEYS!r}\n"
        "print(json.dumps({k: os.environ[k] for k in keys if k in os.environ}))\n"
    )
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return {}
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        return {}
    try:
        data = json.loads(lines[-1])
    except json.JSONDecodeError:
        return {}
    return {str(k): str(v) for k, v in data.items()}


def maybe_launch_tpu_distributed(module: str, args: list[str]) -> None:
    """
    On a multi-chip TPU host (such as v5e-8) invoked directly without torchrun,
    discovers the TPU slice topology and re-execs under torch.distributed.run.
    No-op on CPU/GPU, on single-chip TPUs (v5e-1), or inside existing torchrun workers.
    """
    if "LOCAL_RANK" in os.environ or os.environ.get("HOMR_NO_AUTO_TORCHRUN") == "1":
        return

    tpu_env = query_tpu_launch_env()
    world_size = int(tpu_env.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return

    nnodes = int(tpu_env.get("NNODES", "1"))
    nproc_per_node = int(tpu_env.get("LOCAL_WORLD_SIZE", str(max(1, world_size // max(1, nnodes)))))

    for key, value in tpu_env.items():
        if key not in ("WORLD_SIZE", "LOCAL_WORLD_SIZE", "RANK", "LOCAL_RANK"):
            os.environ.setdefault(key, value)
    os.environ.pop("WORLD_SIZE", None)
    os.environ.pop("RANK", None)

    cmd = [sys.executable, "-m", "torch.distributed.run"]
    if nnodes <= 1:
        cmd.extend(["--standalone", "--nnodes=1", f"--nproc_per_node={nproc_per_node}"])
    else:
        node_rank = tpu_env.get("NODE_RANK", "0")
        master_addr = tpu_env.get("MASTER_ADDR", "127.0.0.1")
        master_port = tpu_env.get("MASTER_PORT", "29500")
        cmd.extend(
            [
                f"--nnodes={nnodes}",
                f"--node_rank={node_rank}",
                f"--nproc_per_node={nproc_per_node}",
                f"--master_addr={master_addr}",
                f"--master_port={master_port}",
            ]
        )
    cmd.extend(["-m", module, *args])

    topology = tpu_env.get("TORCH_TPU_TOPOLOGY", "unknown")
    eprint(
        f"Detected {world_size} TPU chips (topology={topology}); "
        f"launching {nproc_per_node} local worker processes via torchrun"
    )
    os.execv(sys.executable, cmd)  # noqa: S606
