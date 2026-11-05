import glob
import os

from homr.simple_logging import eprint
from training.transformer.training_vocabulary import (
    VocabularyStats,
    check_token_lines,
    read_tokens,
)


def find_tokens() -> tuple[VocabularyStats, set[str]]:
    stats = VocabularyStats()
    errors: set[str] = set()
    exclude = "validation"
    exclude_pre = "flat_pretraining"
    files = [
        f
        for f in glob.glob(os.path.join("datasets", "**", "*.tokens"), recursive=True)
        if exclude not in f.split(os.sep) and exclude_pre not in f.split(os.sep)
    ]
    for i, file in enumerate(files):
        try:
            tokens = read_tokens(file)
            stats.add_lines(tokens)
            check_token_lines(tokens)
            if i % 1000 == 0:
                eprint(i, len(errors))
        except Exception as e:
            eprint("======", file, "======")
            eprint(e)
            errors.add(file)
    return stats, errors


def remove_entry_from_train_index(file_path: str, files: set[str]) -> None:
    if not files:
        return

    normalized = {os.path.normpath(f) for f in files}
    temp_filename = file_path + ".tmp"

    with open(file_path, "r") as src, open(temp_filename, "w") as dst:
        for line in src:
            parts = line.strip().split(",")
            if any(os.path.normpath(p) in normalized for p in parts):
                continue  # drop entries that reference a bad tokens file
            dst.write(line)

    os.replace(temp_filename, file_path)
    print("Finished overwriting index")


def remove_tokens(indexes: list[str]) -> None:
    stats, errors = find_tokens()
    for index in indexes:
        remove_entry_from_train_index(index, errors)
    for file in errors:
        try:
            os.remove(file)
            eprint(f"removing {file}")
        except FileNotFoundError:
            pass

if __name__ == "__main__":
    from training.omr_datasets.convert_grandstaff import grandstaff_train_index
    from training.omr_datasets.convert_lieder import lieder_train_index
    from training.omr_datasets.convert_musetrainer import musetrainer_train_index
    from training.omr_datasets.convert_pdmx import pdmx_train_index
    from training.omr_datasets.convert_primus import primus_train_index

    remove_tokens(
        [
            lieder_train_index,
            musetrainer_train_index,
            pdmx_train_index,
            grandstaff_train_index,
            primus_train_index,
        ],
    )
