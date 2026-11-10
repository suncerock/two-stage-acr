from collections import OrderedDict
from typing import Literal

from mir_eval.chord import pitch_class_to_semitone

from src.data.hchord import Hchord, _shorthands

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
ALL_TRIADS_INTERVALS = OrderedDict((triad, _shorthands[triad]) for triad in [
    "maj", "min", "dim", "aug", "sus4", "sus2"])
ALL_TETRADS_INTERVALS = OrderedDict((tetrad, _shorthands[tetrad]) for tetrad in [
    "7", "maj7", "min7", "dim7", "hdim7", "minmaj7", "aug7", "7sus4", "maj6", "min6", "maj", "min", "dim", "aug", "sus4", "sus2"])


def num_harte_classes(vocab: Literal["tetrads", "triads"], include_no_chord: bool = False) -> int:
    """Return the number of Harte classes, optionally including an explicit N class (last index)."""
    if vocab == "triads":
        num_quality_classes = len(ALL_TRIADS_INTERVALS)
    elif vocab == "tetrads":
        num_quality_classes = len(ALL_TETRADS_INTERVALS)
    else:
        raise ValueError(f"Unknown vocabulary: {vocab}")
    return num_quality_classes * 12 + int(include_no_chord)


def get_chord_quality(chord: Hchord, vocab: Literal["tetrads", "triads"]) -> str:
    """Get the chord quality string based on the specified vocabulary.

    Args:
        chord (Hchord): The chord to analyze.
        vocab (str): The vocabulary type.

    Returns:
        str: The chord quality string, "N" for no-chord, or "X" if the quality is not in the vocabulary.
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

    return "X"


def categorize_hchord(
    chord: Hchord,
    vocab: Literal["tetrads", "triads"],
    include_no_chord: bool = False,
) -> int:
    """Categorize an Hchord into a predefined class index.

    Args:
        chord (Hchord): The chord to categorize.
        vocab (str): The vocabulary type.
        include_no_chord (bool): If True, N maps to the last class index
            (``num_harte_classes(vocab, True) - 1``); otherwise to -1 (ignored).

    Returns:
        int: The category index of the chord (-1 for ignored chords).
    """
    if vocab == "triads":
        quality_list = list(ALL_TRIADS_INTERVALS.keys())
    elif vocab == "tetrads":
        quality_list = list(ALL_TETRADS_INTERVALS.keys())
    else:
        raise ValueError(f"Unknown vocabulary: {vocab}")

    quality = get_chord_quality(chord, vocab)
    if quality == "X":
        return -1
    if quality == "N":
        return len(quality_list) * 12 if include_no_chord else -1

    root = pitch_class_to_semitone(chord.root)
    return root + quality_list.index(quality) * 12


def convert_index_to_hchord(
    index: int,
    vocab: Literal["tetrads", "triads"],
    include_no_chord: bool = False,
) -> Hchord:
    """Convert a category index back to an Hchord.

    Args:
        index (int): The category index to convert.
        vocab (str): The vocabulary type.
        include_no_chord (bool): Whether the last index denotes N.

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

    if include_no_chord and index == len(quality_list) * 12:
        return Hchord("N")

    root_index = index % 12
    quality_index = index // 12

    root_pitch_class = PITCH_CLASSES[root_index]
    quality = quality_list[quality_index]

    return Hchord(root=root_pitch_class, shorthand=quality)
