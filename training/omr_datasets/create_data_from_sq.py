import os
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_dataset

from homr.sql_database import datagen_db, org_images_path
from homr.simple_logging import eprint
from validation.tools import _REPO_ROOT


def run_homr(image_path: Path, img_index: int) -> None:
    """Run homr on every image in image_dir, writing .musicxml alongside each."""
    result = subprocess.run(  # noqa: S603
        [
            "poetry",
            "run",
            "python3",
            "-m",
            "homr.main",
            image_path,
            "--no-title",
            "--datagen",
            str(img_index),
        ],  # noqa: S607
        cwd=str(_REPO_ROOT),
        capture_output=True,  # if True does not show log from homr
        check=False,
    )
    if result.returncode != 0:
        # capture_output=False leaves stdout/stderr as None (output went straight to the console)
        stderr = (result.stderr or b"").decode("utf-8", errors="replace")
        raise RuntimeError(f"homr exited with code {result.returncode}\n{stderr[:1000]}")


def main(number_of_images: int, max_workers=6):
    """
    1. Run homr
    2. Gather data and save it to sql database
    3.
    """
    ds = load_dataset("parquet", data_files="datasets/sq_native/*.parquet", split="train")

    jobs = []
    for i in range(min(number_of_images, len(ds))):
        # Get data
        row = ds[i]
        image_imslp = row["image_imslp"]
        filename = row["filename"]
        musicxml = row["musicxml"]

        # Save image and create sql entry
        path = os.path.join(org_images_path, f"page_hf_{i + 1}.png")
        image_imslp.save(path)
        index = datagen_db.add_page(path, musicxml, filename)

        # Add to job for multithreading
        jobs.append((path, index))

    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_job = {
            pool.submit(run_homr, path, index): (path, index)
            for path, index in jobs
        }
        for future in as_completed(future_to_job):
            path, index = future_to_job[future]
            done += 1
            try:
                future.result()
                eprint(f"Completed {done}/{len(jobs)} jobs")
            except Exception as e:
                eprint(f"FAILED job {index} ({path}): {e}")
            


if __name__ == "__main__":
    main(20)
