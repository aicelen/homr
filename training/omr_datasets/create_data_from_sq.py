import os
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv

from music21 import converter
import xml.etree.ElementTree as ET

from homr.sql_database import datagen_db, org_images_path, dataset_root
from homr.simple_logging import eprint
from validation.tools import _REPO_ROOT
from training.omr_datasets.convert_pdmx import _read_mxl
string_quartets_root = os.path.join(dataset_root, "sq-git-repo")
string_quartets_index = os.path.join(string_quartets_root, "data", "scores.tsv")
string_quartets_scores = os.path.join(string_quartets_root, "scores")

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
        capture_output=False,  # if True does not show log from homr
        check=False,
    )
    if result.returncode != 0:
        # capture_output=False leaves stdout/stderr as None (output went straight to the console)
        stderr = (result.stderr or b"").decode("utf-8", errors="replace")
        raise RuntimeError(f"homr exited with code {result.returncode}\n{stderr[:1000]}")


def run_segnet_server() -> subprocess.Popen:
    """Run homr on every image in image_dir, writing .musicxml alongside each."""
    process = subprocess.Popen(  # noqa: S603
        [
            "poetry",
            "run",
            "python3",
            "-m",
            "homr.segmentation.segnet_server",
        ],  # noqa: S607
        cwd=str(_REPO_ROOT),
    )
    return process


def main(number_of_images: int, max_workers=4):
    """
    1. Run homr
    2. Gather data and save it to sql database
    3.
    """
    server = run_segnet_server()
    try:
        with open(string_quartets_index, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            rows = list(reader)
            jobs = []

            for i in range(min(number_of_images, len(rows))):
                # Get data
                entry = rows[i]
                id = entry["id"] # e.g. 7313978
                path = entry["path"] # e.g. Andrée,_Elfrida/String_Quartet_in_A_major 
                filename = entry["name"] # e.g. String Quartet in A major

                # build paths
                pdf_path = os.path.join(string_quartets_scores, path, f"sq{id}.pdf")
                mxl_path = os.path.join(string_quartets_scores, path, f"sq{id}.mxl")

                musicxml = _read_mxl(mxl_path)
                index = datagen_db.add_page(pdf_path, musicxml, filename)

                # Add to job for multithreading
                jobs.append((pdf_path, index))

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
    finally:
        server.terminate()
        server.wait(timeout=10)
            

if __name__ == "__main__":
    main(number_of_images=2, max_workers=2)
