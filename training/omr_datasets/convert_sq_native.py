from datasets import load_dataset
from validation.tools import _REPO_ROOT
from pathlib import Path
import subprocess
import os
from homr.sql_database import datagen_db, org_images_path

def run_homr(image_path: Path, img_index: int) -> None:
    """Run homr on every image in image_dir, writing .musicxml alongside each."""
    result = subprocess.run(  # noqa: S603
        ["poetry", "run", "python3", "-m", "homr.main", image_path, "--no-title", "--datagen", str(img_index)],  # noqa: S607
        cwd=str(_REPO_ROOT),
        capture_output=False,  # if True does not show log from homr
        check=False,
    )
    if result.returncode != 0:
        # capture_output=False leaves stdout/stderr as None (output went straight to the console)
        stderr = (result.stderr or b"").decode("utf-8", errors="replace")
        raise RuntimeError(f"homr exited with code {result.returncode}\n{stderr[:1000]}")



def main(number_of_images: int, clean=False):
    """
    1. Run homr 
    2. Gather data and save it to sql database
    3. 
    """
    ds = load_dataset("parquet", data_files="datasets/sq_native/*.parquet", split="train")

    # access an example
    for i in range(min(number_of_images, len(ds))):
        # Get data
        row = ds[i]
        image_imslp = row["image_imslp"]
        filename = row["filename"]
        musicxml = row["musicxml"]

        path = os.path.join(org_images_path, f"page_hf_{i+1}.png")

        # Save image
        image_imslp.save(path)

        index = datagen_db.add_page(path, musicxml, filename)
        run_homr(path, index)

if __name__ == "__main__":
    main(5, clean=True)
