from datasets import load_dataset
from validation.tools import _REPO_ROOT
from pathlib import Path
import subprocess
import os

script_location = os.path.dirname(os.path.realpath(__file__))
git_root = Path(script_location).parent.parent.absolute()
dataset_root = os.path.join(git_root, "datasets")
lieder = os.path.join(dataset_root, "Lieder-main")
lieder_train_index = os.path.join(lieder, "index.txt")


def _run_homr_on_dir(image_dir: Path) -> None:
    """Run homr on every image in image_dir, writing .musicxml alongside each."""
    result = subprocess.run(  # noqa: S603
        ["poetry", "run", "python3", "-m", "homr.main", str(image_dir), "--no-title", "--datagen"],  # noqa: S607
        cwd=str(_REPO_ROOT),
        capture_output=True,  # if True does not show log from homr
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"homr exited with code {result.returncode}\n{stderr[:1000]}")



def main():
    """
    1. Run homr 
    2. Gather data and save it to sql database
    3. 
    """
    ds = load_dataset("parquet", data_files="datasets/sq_native/*.parquet", split="train")

    # access an example

    example = ds[0]
    image_imslp = example["image_imslp"]
    musicxml = example["musicxml"]
    filename = example["filename"]

