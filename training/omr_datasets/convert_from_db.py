import os
import sys
import xml.etree.ElementTree as ET
from collections import deque

from homr.circle_of_fifths import strip_naturals
from homr.transformer.vocabulary import EncodedSymbol
from training.omr_datasets.music_xml_parser import music_xml_string_to_tokens
from training.transformer.training_vocabulary import (
    check_token_lines,
    token_lines_to_str,
)
from homr.sql_database import datagen_db, datagen_train_index, Page

def get_part_ids_in_order(root: ET.Element) -> list:
    """Return part IDs in <part-list> order (top-to-bottom voice order)."""
    part_list = root.find("part-list")
    return [sp.get("id") for sp in part_list.findall("score-part")]


def get_system_measure_counts(part: ET.Element) -> list:
    """Return a list of measure counts per system for a single part."""
    counts = []
    current = 0

    for measure in part.findall("measure"):
        print_el = measure.find("print")
        is_new_system = print_el is not None and print_el.get("new-system") == "yes"

        if is_new_system and current > 0:
            counts.append(current)
            current = 0

        current += 1

    if current > 0:
        counts.append(current)

    return counts


def count_measures_in_xml(musicxml: str) -> list:
    """Return a flat list of measure counts per system, e.g. [5, 5, 7, 8]."""
    root = ET.fromstring(musicxml)

    part_ids = get_part_ids_in_order(root)
    parts = {p.get("id"): p for p in root.findall("part")}

    per_voice_counts = [get_system_measure_counts(parts[pid]) for pid in part_ids]

    if any(counts != per_voice_counts[0] for counts in per_voice_counts):
        print(
            f"WARNING: voices disagree on measures per system: {per_voice_counts}", file=sys.stderr
        )

    return per_voice_counts[0] if per_voice_counts else []


def match_xml_and_staffs(page_data: Page) -> None:
    """
    This matches the staff and the musicxml file by counting measures
    """
    ground_truth_tokens = music_xml_string_to_tokens(page_data.musicxml)
    number_measures = count_measures_in_xml(page_data.musicxml)
    number_of_systems = len(number_measures)
    expected_staffs = len(ground_truth_tokens) * number_of_systems
    if expected_staffs != len(page_data.staffs):
        raise ValueError(
            f"Cannot match musicxml and staffs of page"
            f" {os.path.basename(os.path.dirname(page_data.staffs[0]))}:"
            f" expected {expected_staffs} staffs"
            f" ({len(ground_truth_tokens)} voices x {number_of_systems} systems)"
            f" but found {len(page_data.staffs)} staffs"
        )

    for voice_index, voice in enumerate(ground_truth_tokens):
        voice = deque(voice)
        for i, number_measure in enumerate(number_measures):
            measures = [voice.popleft() for _ in range(number_measure)]
            flat = [symbol for measure in measures for symbol in measure]

            # Musicxml only shows the clef, key-sign and time-sign once at the very beginning. Therefore we need
            # to append it.
            if i == 0:
                # First iteration we set our standard header
                standard_header = get_header(flat)

            cur_header = get_header(flat)

            # Combine header from last staff with the current ones
            merged_header = [
                a if a is not None else b
                for a, b in zip(cur_header, standard_header)
                if a is not None or b is not None
            ]

            # we need to strip the current header
            flat = merged_header + flat[len([x for x in cur_header if x is not None]) :]

            flat = strip_naturals(flat)
            check_token_lines(flat)
            tokens_str = token_lines_to_str(flat)
            # Staff images are numbered voice-major (see parse_staffs in
            # homr/staff_parsing.py): first all systems of voice 0, then all
            # systems of voice 1 and so on. Therefore we have to offset each
            # system index by all the staffs of the previous voices.
            staff_index = voice_index * number_of_systems + i
            save_tokens(tokens_str, page_data.staffs[staff_index])

def get_header(
    flat: list[EncodedSymbol],
) -> list[EncodedSymbol | None, EncodedSymbol | None, EncodedSymbol | None]:
    clef, key, time = None, None, None
    for sym in flat[:3]:
        if sym.rhythm.startswith("clef"):
            clef = sym
        elif sym.rhythm.startswith("key"):
            key = sym
        elif sym.rhythm.startswith("time"):
            time = sym
    return [clef, key, time]


def save_tokens(tokens_ground_truth: str, staff_path: str):
    base_path, _ = os.path.splitext(staff_path)
    token_path = base_path + ".tokens"
    with open(token_path, "w") as f:
        f.write(tokens_ground_truth)

    with open(datagen_train_index, "a") as f:
        f.write(f"{staff_path},{token_path}\n")


def convert_from_db():
    for i in range(datagen_db.get_page_count()):
        page_data = datagen_db.get_data_samples(i+1)
        match_xml_and_staffs(page_data)


if __name__ == "__main__":
    convert_from_db()
