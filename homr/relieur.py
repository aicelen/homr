"""
This code is based on relieur: https://github.com/papoteur-mga/relieur/tree/master
which is licensed under Apache-2.0.
We thank Papoteur (https://github.com/papoteur-mga) and contributors for their work!
"""

import os
from glob import glob
from pathlib import Path

import musicxml.xmlelement.xmlelement as mxl  # type: ignore
from musicxml.parser.parser import parse_musicxml  # type: ignore

from homr.simple_logging import eprint


def get_children_of_type(element, type_):
    return [c for c in element.get_children() if isinstance(c, type_)]


def clef_attributes(clef) -> dict[str]:  # type ignore
    # take mxl.XMLClef
    attributes = {}
    for a in get_children_of_type(clef, mxl.XMLSign):
        attributes["Sign"] = a.value_
    for a in get_children_of_type(clef, mxl.XMLSign):
        attributes["Line"] = a.value_
    for a in get_children_of_type(clef, mxl.XMLClefOctaveChange):
        attributes["ClefOctaveChange"] = a.value_
    return attributes


def time_attributes(time) -> dict[str]:  # type ignore
    # take mxl.XMLTime
    attributes = {}
    for a in get_children_of_type(time, mxl.XMLBeats):
        attributes["Beats"] = a.value_
    for a in get_children_of_type(time, mxl.XMLBeatType):
        attributes["BeatType"] = a.value_
    for a in get_children_of_type(time, mxl.XMLInterchangeable):
        attributes["Interchangeable"] = a.value_
    for a in get_children_of_type(time, mxl.XMLSenzaMisura):
        attributes["SenzaMisura"] = a.value_
    return attributes


def key_attributes(key) -> dict[str]:  # type ignore
    # take mxl.XMLKey
    attributes = {}
    for a in get_children_of_type(key, mxl.XMLFifths):
        attributes["Fifths"] = a.value_
    for a in get_children_of_type(key, mxl.XMLKeyAlter):
        attributes["KeyAlter"] = a.value_
    for a in get_children_of_type(key, mxl.XMLMode):
        attributes["Mode"] = a.value_
    return attributes


def process_concat(
    concat: list[str],
    debug=False,
) -> tuple:
    sorted_list = get_file_list(concat, debug=debug)
    eprint(sorted_list)
    if not sorted_list:
        return None, 0, 0
    # Main file is the first of the list
    main_file = sorted_list[0]
    eprint(f"Starting with {main_file}")
    m = parse_musicxml(main_file)
    # look for the last key and last divisions
    last_parts_attributes = []
    part1 = None  # Initialize part1 to avoid UnboundLocalError
    for part1 in get_children_of_type(m, mxl.XMLPart):
        part_attributes = {}
        for measure in get_children_of_type(part1, mxl.XMLMeasure):
            for attrib in get_children_of_type(measure, mxl.XMLAttributes):
                for div in get_children_of_type(attrib, mxl.XMLDivisions):
                    part_attributes["Divisions"] = div.value_
                for key in get_children_of_type(attrib, mxl.XMLKey):
                    part_attributes["Key"] = key_attributes(key)
                for xtime in get_children_of_type(attrib, mxl.XMLTime):
                    part_attributes["Time"] = time_attributes(xtime)
                for clef in get_children_of_type(attrib, mxl.XMLClef):
                    part_attributes["Clef"] = clef_attributes(clef)
        last_parts_attributes.append(part_attributes)
    for f in sorted_list[1:]:
        # new file to add
        if debug:
            eprint(f"Processing {f}")
        b = parse_musicxml(f)
        ip = 0
        for part1 in get_children_of_type(m, mxl.XMLPart):
            # each part from the main score
            current_len = len(get_children_of_type(part1, mxl.XMLMeasure))
            if debug:
                eprint(f"Main part has {current_len} measures")
            ib = 0
            for part in get_children_of_type(b, mxl.XMLPart):
                if ib == ip:
                    # we add the part of the new file having the same order
                    # as the part from the main score, else we pass
                    for measure in get_children_of_type(part, mxl.XMLMeasure):
                        new_number = str(int(measure.number) + current_len)
                        if int(measure.number) == 1:
                            for attrib in get_children_of_type(measure, mxl.XMLAttributes):
                                for div in get_children_of_type(attrib, mxl.XMLDivisions):
                                    if last_parts_attributes[ib]["Divisions"] == div.value_:
                                        attrib.remove(div)
                                        if debug:
                                            eprint(
                                                f"Remove division at measure {new_number}, part {ib + 1}"  # noqa: E501
                                            )
                                for key in get_children_of_type(attrib, mxl.XMLKey):
                                    if last_parts_attributes[ib]["Key"] == key_attributes(key):
                                        if debug:
                                            eprint(
                                                f"Remove key at measure {new_number} part {ib + 1}"
                                            )
                                        attrib.remove(key)
                                for xtime in get_children_of_type(attrib, mxl.XMLTime):
                                    if last_parts_attributes[ib]["Time"] == time_attributes(xtime):
                                        attrib.remove(xtime)
                                        if debug:
                                            eprint(
                                                f"Remove time at measure {new_number} part {ib + 1}"
                                            )
                                for clef in get_children_of_type(attrib, mxl.XMLClef):
                                    if last_parts_attributes[ib]["Clef"] == clef_attributes(clef):
                                        attrib.remove(clef)
                                        if debug:
                                            eprint(
                                                f"Remove clef at measure {new_number} part {ib + 1}"
                                            )
                        measure.number = new_number
                        part1.add_child(measure)
                        if debug:
                            eprint(f"Added measure {new_number}, part {ib + 1}")
                    current_len = len(get_children_of_type(part1, mxl.XMLMeasure))
                ib += 1
            ip += 1
    return m, len(sorted_list), len(get_children_of_type(part1, mxl.XMLMeasure)) if part1 else 0


def get_file_list(
    concat: tuple[str],
    debug=False,
) -> list[str]:
    # get the list of files
    sorted_list = []

    for pattern in concat:
        if not Path(pattern).suffix:
            pattern += "*.musicxml"  # noqa: PLW2901

        matched_files = list(glob(pattern))

        if len(matched_files) == 0 and debug:
            eprint(f"No file found for {pattern}")

        for fichier in matched_files:
            if not os.path.exists(fichier):
                if debug:
                    eprint(f"The file {fichier} does not exist.")
                return None
            if os.path.isdir(fichier):
                if debug:
                    eprint(f"{fichier} is a directory.")
                return None

            sorted_list.append(fichier)
    if len(sorted_list) == 0:
        eprint(f"No files found for {concat}")
        return None
    return sorted(sorted_list)
