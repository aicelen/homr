import csv
import multiprocessing
import os
import random
import sys
from itertools import zip_longest
from pathlib import Path

from homr.download_utils import download_file, untar_file
from homr.simple_logging import eprint

script_location = os.path.dirname(os.path.realpath(__file__))
git_root = Path(script_location).parent.parent.absolute()
dataset_root = os.path.join(git_root, "datasets")
pdmx_root = os.path.join(dataset_root, "pdmx")
pdmx_csv = os.path.join(pdmx_root, "PDMX.csv")
pdmx_mxl_root = os.path.join(pdmx_root, "mxl")
pdmx_train_index = os.path.join(pdmx_root, "index.txt")
pdmx_pretrain_index = os.path.join(pdmx_root, "index_pretraining.txt")
flat_pdmx = os.path.join(pdmx_root, "flat")
flat_pretraining_pdmx = os.path.join(pdmx_root, "flat_pretraining")

rendered_scores = os.path.join(pdmx_root, "rendered_scores")
rendered_scores_pretraining = os.path.join(pdmx_root, "rendered_scores_pretraining")

from training.omr_datasets.convert_lieder import (
    convert_file_only_token_pdmx,
    convert_token_and_image_pdmx,
    convert_musicxml_only_token_pdmx,
    create_musicxml_and_svg_files_from_mxl,
)

# Complexity 3 often has octave shifts (unsupported) and long sequences (see max_seq_len),
# so we skip attempting those files rather than waste time converting ones we'd filter out later.
_MAX_COMPLEXITY = 2
_MAX_TRACKS = 2
_TARGET_FILES = 10000  # ~50K images
_TARGET_FILES_PRETRAINING = 100000 # ~500k staffs

def _load_filtered_paths(pretraining: bool = False) -> list[Path]:
    if pretraining:
        start_idx = _TARGET_FILES
        stop_idx = _TARGET_FILES_PRETRAINING + _TARGET_FILES
    else:
        start_idx = 0
        stop_idx = _TARGET_FILES

    buckets: dict[tuple[int, int], list[Path]] = {}
    with open(pdmx_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["subset:no_license_conflict"] != "True":
                continue
            if row["subset:all_valid"] != "True":
                continue
            try:
                n_tracks = int(row["n_tracks"])
                if n_tracks > _MAX_TRACKS:
                    continue
            except (ValueError, KeyError):
                continue
            try:
                complexity = int(row["complexity"])
                if complexity > _MAX_COMPLEXITY:
                    continue
            except (ValueError, KeyError):
                continue
            rel_mxl = row["mxl"].lstrip("./")
            buckets.setdefault((n_tracks, complexity), []).append(Path(pdmx_root) / rel_mxl)

    rng = random.Random(42)
    for paths in buckets.values():
        rng.shuffle(paths)

    for key in sorted(buckets):
        n_tracks, complexity = key
        eprint(f"  tracks={n_tracks} complexity={complexity}: {len(buckets[key]):,} files")

    sentinel = object()
    result: list[Path] = []
    for group in zip_longest(*[buckets[k] for k in sorted(buckets)], fillvalue=sentinel):
        for item in group:
            if item is not sentinel:
                result.append(item)  # type: ignore[arg-type]
            if len(result) >= stop_idx:
                return result[start_idx:]
    return result[start_idx:]


def convert_pdmx(only_recreate_token_files: bool = False, from_xml: bool = False) -> None:
    os.makedirs(pdmx_root, exist_ok=True)
    os.makedirs(flat_pdmx, exist_ok=True)
    os.makedirs(flat_pretraining_pdmx, exist_ok=True)
    os.makedirs(rendered_scores, exist_ok=True)
    os.makedirs(rendered_scores_pretraining, exist_ok=True)
    if not os.path.exists(pdmx_csv):
        eprint("Downloading PDMX.csv (~214 MB)")
        download_file(
            "https://zenodo.org/api/records/15571083/files/PDMX.csv/content",
            pdmx_csv,
        )

    if not os.path.exists(pdmx_mxl_root):
        eprint("Downloading PDMX mxl.tar.gz (~1.8 GB)")
        mxl_archive = os.path.join(pdmx_root, "mxl.tar.gz")
        download_file(
            "https://zenodo.org/api/records/15571083/files/mxl.tar.gz/content",
            mxl_archive,
        )
        eprint("Extracting mxl.tar.gz")
        untar_file(mxl_archive, pdmx_root)

    eprint("Reading CSV and applying filters")
    mxl_paths = _load_filtered_paths(from_xml)
    eprint(f"{len(mxl_paths)} files pass pre-filters (c<={_MAX_COMPLEXITY}, tracks<={_MAX_TRACKS})")

    if from_mxl:
        create_musicxml_and_svg_files_from_mxl(mxl_paths, rendered_scores_pretraining, include_svg=False)
        music_xml_files = list(Path(rendered_scores_pretraining).rglob("*.musicxml"))
    else:
        create_musicxml_and_svg_files_from_mxl(mxl_paths, rendered_scores)
        music_xml_files = list(Path(rendered_scores).rglob("*.musicxml"))

    with open(pdmx_pretrain_index if from_xml else pdmx_train_index, "w", buffering=1 << 20) as f:
        file_number = 0
        skipped_files = 0
        with multiprocessing.Pool(processes=8, maxtasksperchild=100) as p:
            for result in p.imap_unordered(
                (
                convert_file_only_token_pdmx if only_recreate_token_files
                    else (
                        convert_musicxml_only_token_pdmx if from_xml
                        else convert_token_and_image_pdmx
                    )
                ),
                music_xml_files,
            ):
                if len(result) > 0:
                    for line in result:
                        f.write(line)
                    f.flush()
                else:
                    skipped_files += 1
                file_number += 1
                if file_number % 10 == 0:
                    eprint(
                        f"Processed {file_number}/{len(music_xml_files)} files,",
                        f"skipped {skipped_files} files",
                    )
    eprint("Done indexing")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    only_recreate_token_files = False
    if "--from-mxl" in sys.argv:
        from_mxl = True
    if "--only-tokens" in sys.argv:
        only_recreate_token_files = True
    convert_pdmx(only_recreate_token_files, from_mxl)
