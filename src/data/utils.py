from collections import OrderedDict
from typing import Literal

from mir_eval.chord import pitch_class_to_semitone

from src.data.hchord import Hchord, _shorthands

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
ALL_TRIADS_INTERVALS = OrderedDict((triad, _shorthands[triad]) for triad in [
    "maj", "min", "dim", "aug", "sus4", "sus2"])
ALL_TETRADS_INTERVALS = OrderedDict((tetrad, _shorthands[tetrad]) for tetrad in [
    "7", "maj7", "min7", "dim7", "hdim7", "minmaj7", "aug7", "7sus4", "maj6", "min6", "maj", "min", "dim", "aug", "sus4", "sus2"])
def get_chord_quality(chord: Hchord, vocab: Literal["tetrads", "triads"]) -> str:
    """Get the chord quality string based on the specified vocabulary.

    Args:
        chord (Hchord): The chord to analyze.
        vocab (str): The vocabulary type.

    Returns:
        str: The chord quality string.
    """
    if chord.root is None:
        return "N"
    if vocab == "triads":
        for shorthand, intervals in ALL_TRIADS_INTERVALS.items():
            if set(intervals).issubset(chord.intervals):
                return shorthand

    elif vocab == "tetrads":
        for shorthand, intervals in ALL_TETRADS_INTERVALS.items():
            if set(intervals).issubset(chord.intervals):
                return shorthand

    else:
        raise ValueError(f"Unknown vocabulary: {vocab}")

    return "N"

def categorize_hchord(chord: Hchord, vocab: Literal["tetrads", "triads"]) -> int:
    """Categorize an Hchord into a predefined class index.

    Args:
        chord (Hchord): The chord to categorize.

    Returns:
        int: The category index of the chord.
    """

    quality = get_chord_quality(chord, vocab)
    if quality == "N":
        return -1
    else:
        root = pitch_class_to_semitone(chord.root)
        if vocab == "triads":
            quality_list = list(ALL_TRIADS_INTERVALS.keys())
        elif vocab == "tetrads":
            quality_list = list(ALL_TETRADS_INTERVALS.keys())
        else:
            raise ValueError(f"Unknown vocabulary: {vocab}")
        quality_index = quality_list.index(quality)
        category_index = root + quality_index * 12
        return category_index

def convert_index_to_hchord(index: int, vocab: Literal["tetrads", "triads"]) -> Hchord:
    """Convert a category index back to an Hchord.

    Args:
        index (int): The category index to convert.

    Returns:
        Hchord: The corresponding Hchord object.
    """
    if index == -1:
        return Hchord("N")
    
    if vocab == "triads":
        quality_list = list(ALL_TRIADS_INTERVALS.keys())
    elif vocab == "tetrads":
        quality_list = list(ALL_TETRADS_INTERVALS.keys())
    else:
        raise ValueError(f"Unknown vocabulary: {vocab}")
    
    root_index = index % 12
    quality_index = index // 12
    
    root_pitch_class = PITCH_CLASSES[root_index]
    quality = quality_list[quality_index]

    return Hchord(root=root_pitch_class, shorthand=quality)
