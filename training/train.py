import argparse
import sys

from homr.simple_logging import eprint
from training.tpu_launch import maybe_launch_tpu_distributed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("model_name", type=str, help="The name of the model to train")
    parser.add_argument("--batch-size", type=int, default=None, help="Per-device train batch size")
    parser.add_argument("--grad-accum", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--workers", type=int, default=None, help="DataLoader workers per process")
    parser.add_argument("--resume", type=str, default="", help="Checkpoint subfolder to resume")
    parser.add_argument("--fp32", action="store_true", help="Disable bf16 mixed precision")
    parser.add_argument("--smoke-test", action="store_true", help="Run 10-epoch smoke test")
    parser.add_argument("--fine", action="store_true", help="Fine-tune lift decoder only")
    args = parser.parse_args()

    model_type = args.model_name

    if model_type == "segnet":
        from training import download  # noqa: PLC0415
        from training.segmentation import train  # noqa: PLC0415

        download.download_deep_scores()
        train.train_segnet()
    elif model_type == "transformer":
        maybe_launch_tpu_distributed("training.train", sys.argv[1:])
        from training.transformer import train as transformer  # noqa: PLC0415

        transformer.train_transformer(
            fp32=args.fp32,
            resume=args.resume,
            smoke_test=args.smoke_test,
            fine_tune=args.fine,
            batch_size_override=args.batch_size,
            grad_accum_override=args.grad_accum,
            workers_override=args.workers,
        )
    else:
        eprint("Unknown model: " + model_type)
        sys.exit(1)


if __name__ == "__main__":
    main()
