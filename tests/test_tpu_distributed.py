import os
import tempfile
import unittest
from typing import Any
from unittest.mock import patch

import timm
import torch
from accelerate.state import AcceleratorState
from transformers import TrainerState, TrainingArguments

from homr.transformer.configs import Config
from training.architecture.transformer.tromr_arch import TrOMR
from training.tpu_launch import maybe_launch_tpu_distributed
from training.transformer.distribute import Distribute, configure_accelerate_for_tpu_ddp
from training.transformer.metrics import HomrTrainer
from training.transformer.train import FreezeCallback, compute_training_schedule


def _create_tromr_offline() -> TrOMR:
    orig_create_model = timm.create_model

    def _no_download(*args: Any, **kwargs: Any) -> Any:
        kwargs["pretrained"] = False
        return orig_create_model(*args, **kwargs)

    with patch("timm.create_model", side_effect=_no_download):
        return TrOMR(Config())


class TestTpuDistributed(unittest.TestCase):
    def tearDown(self) -> None:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        AcceleratorState._reset_state(reset_partial_state=True)

    def test_compute_training_schedule_keeps_global_batch_64_on_tpu(self) -> None:
        # v5e-1: 16 * 4 * 1 = 64
        b1, eval1, ga1, w1 = compute_training_schedule(tpu=True, world_size=1, cpu_count=44)
        self.assertEqual((b1, eval1, ga1, w1), (16, 16, 4, 20))
        self.assertEqual(b1 * ga1 * 1, 64)

        # v5e-8: 8 * 1 * 8 = 64
        b8, eval8, ga8, w8 = compute_training_schedule(tpu=True, world_size=8, cpu_count=224)
        self.assertEqual((b8, eval8, ga8, w8), (8, 16, 1, 12))
        self.assertEqual(b8 * ga8 * 8, 64)

        # Overrides are respected
        bo, _, gao, wo = compute_training_schedule(
            tpu=True,
            world_size=8,
            cpu_count=224,
            batch_size_override=16,
            grad_accum_override=2,
            workers_override=6,
        )
        self.assertEqual((bo, gao, wo), (16, 2, 6))

    def test_maybe_launch_tpu_distributed_v5e8_execs_torchrun(self) -> None:
        fake_tpu_env = {
            "TORCH_TPU_TOPOLOGY": "2,4,1",
            "TORCH_TPU_SLICEBUILDER_ADDRESSES": "localhost:8471,localhost:8472",
            "WORLD_SIZE": "8",
            "LOCAL_WORLD_SIZE": "8",
            "NNODES": "1",
            "NODE_RANK": "0",
        }
        clean_env = {"PATH": os.environ.get("PATH", "")}
        with (
            patch.dict(os.environ, clean_env, clear=True),
            patch("training.tpu_launch.query_tpu_launch_env", return_value=fake_tpu_env),
            patch("os.execv") as mock_execv,
        ):
            maybe_launch_tpu_distributed("training.train", ["transformer", "--smoke-test"])
            mock_execv.assert_called_once()
            _, cmd = mock_execv.call_args[0]
            self.assertIn("--standalone", cmd)
            self.assertIn("--nproc_per_node=8", cmd)
            self.assertEqual(cmd[-3:], ["-m", "training.train", "transformer", "--smoke-test"][-3:])
            self.assertEqual(os.environ["TORCH_TPU_TOPOLOGY"], "2,4,1")
            self.assertNotIn("WORLD_SIZE", os.environ)

    def test_maybe_launch_tpu_distributed_noop_on_v5e1_or_worker(self) -> None:
        with (
            patch.dict(os.environ, {"LOCAL_RANK": "3"}, clear=True),
            patch("training.tpu_launch.query_tpu_launch_env") as mock_query,
            patch("os.execv") as mock_execv,
        ):
            maybe_launch_tpu_distributed("training.train", ["transformer"])
            mock_query.assert_not_called()
            mock_execv.assert_not_called()

        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "training.tpu_launch.query_tpu_launch_env",
                return_value={"TORCH_TPU_TOPOLOGY": "1,1,1", "WORLD_SIZE": "1"},
            ),
            patch("os.execv") as mock_execv,
        ):
            maybe_launch_tpu_distributed("training.train", ["transformer"])
            mock_execv.assert_not_called()

    def test_configure_accelerate_for_tpu_ddp_with_8_ranks(self) -> None:
        import torch.testing._internal.distributed.fake_pg  # noqa: F401, PLC0415

        env = {"RANK": "5", "LOCAL_RANK": "5", "WORLD_SIZE": "8"}
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, env, clear=False):
            torch.distributed.init_process_group(backend="fake", rank=5, world_size=8)
            configure_accelerate_for_tpu_ddp(
                rank=5,
                local_rank=5,
                world_size=8,
                backend="fake",
                device=torch.device("cpu"),
            )
            args = TrainingArguments(
                tmpdir,
                accelerator_config={"use_configured_state": True},
                per_device_train_batch_size=8,
                per_device_eval_batch_size=16,
                ddp_find_unused_parameters=True,
                report_to=[],
            )
            self.assertEqual(args.world_size, 8)
            self.assertEqual(args.process_index, 5)
            self.assertEqual(args.local_rank, 5)
            self.assertFalse(args.should_save)
            self.assertEqual(str(args.parallel_mode), "ParallelMode.DISTRIBUTED")

    def test_pruned_encoder_ddp_scheduled_sampling_and_freeze_callback(self) -> None:
        store = torch.distributed.HashStore()
        torch.distributed.init_process_group(backend="gloo", store=store, rank=0, world_size=1)

        model = _create_tromr_offline()
        model.set_grad_checkpointing(True)

        # Verify stage 3 and head norm were pruned so no dead trainable params remain
        param_names = {name for name, _ in model.named_parameters()}
        self.assertFalse(any(name.startswith("encoder.model.stages.3.") for name in param_names))
        self.assertFalse(any(name.startswith("encoder.model.head.") for name in param_names))

        env = {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1"}
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, env, clear=False):
            configure_accelerate_for_tpu_ddp(
                rank=0,
                local_rank=0,
                world_size=1,
                backend="gloo",
                device=torch.device("cpu"),
            )
            distribute = Distribute.__new__(Distribute)
            distribute.tpu = False
            distribute.enabled = True
            distribute.rank = 0
            distribute.world_size = 1
            distribute._device = torch.device("cpu")

            args = TrainingArguments(
                tmpdir,
                accelerator_config={"use_configured_state": True},
                per_device_train_batch_size=2,
                ddp_find_unused_parameters=True,
                use_cpu=False,
                report_to=[],
            )
            trainer = HomrTrainer(model=model, args=args, distribute=distribute, tpu_bf16=False)
            ddp_model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=None, output_device=None, find_unused_parameters=True
            )
            ddp_model.train()

            # Simulate step > 0 so scheduled sampling must activate even through DDP wrapper
            trainer.state = TrainerState()
            trainer.state.global_step = 1000

            freeze_cb = FreezeCallback(epochs_to_freeze=2)
            freeze_cb.on_train_begin(args, trainer.state, trainer.control, model=model)

            inputs: dict[str, Any] = {
                "inputs": torch.randn(2, 1, 256, 1280),
                "rhythms": torch.zeros(2, 16, dtype=torch.long),
                "pitchs": torch.zeros(2, 16, dtype=torch.long),
                "lifts": torch.zeros(2, 16, dtype=torch.long),
                "articulations": torch.zeros(2, 16, dtype=torch.long),
                "slurs": torch.zeros(2, 16, dtype=torch.long),
                "positions": torch.zeros(2, 16, dtype=torch.long),
                "mask": torch.ones(2, 16, dtype=torch.bool),
            }
            loss = trainer.compute_loss(ddp_model, inputs)
            self.assertIn("sampling_prob", inputs)
            if not isinstance(loss, torch.Tensor):
                self.fail("Expected tensor loss")
            loss.backward()
            ddp_model.zero_grad()

            # Unfreeze at epoch 2 and verify all parameters receive gradients
            trainer.state.epoch = 2.0
            freeze_cb.on_epoch_begin(args, trainer.state, trainer.control, model=model)
            inputs.pop("sampling_prob", None)
            loss2 = trainer.compute_loss(ddp_model, inputs)
            if not isinstance(loss2, torch.Tensor):
                self.fail("Expected tensor loss")
            loss2.backward()
            unused = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
            self.assertEqual(unused, [])


if __name__ == "__main__":
    unittest.main()
