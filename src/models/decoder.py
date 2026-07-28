"""RBP to Hchord string decoder.

Converts Root-Bitmap-Bass (RBP) model outputs into :class:`~src.data.hchord.Hchord`
and then exports them as strings that can be used for evaluation with ``mir_eval``.
"""

from typing import List, Dict

import numpy as np

from src.data.hchord import Hchord
from src.data.utils import ALL_TRIADS_INTERVALS, ALL_TETRADS_INTERVALS

INTERVAL_MAP: Dict[int, str] = {
    0:  "1",
    1:  "b2",
    2:  "2",
    3:  "b3",
    4:  "3",
    5:  "4",
    6:  "b5",
    7:  "5",
    8:  "b6",
    9:  "6",
    10: "b7",
    11: "7",
}

# Root pitch classes
PITCH_CLASSES: List[str] = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def convert_harte_index_to_hchord(index: int, harte_vocab: str) -> Hchord:
    """Convert categorized Harte index to harte chord symbol based on the specified vocabulary.
    This assumes enharmonic equivalence because index does not include enharmonic distinctions.

    Parameters:
    ----------
    index: int
        The categorized Harte index to convert.
    harte_vocab: str
        The harte vocabulary used (e.g., "triads", "tetrads").

    Returns:
    -------
    str
        The corresponding harte chord symbol.
    """
    assert index >= 0, "Index must be non-negative."

    if harte_vocab == "triads":
        quality_list = list(ALL_TRIADS_INTERVALS.keys())
    elif harte_vocab == "tetrads":
        quality_list = list(ALL_TETRADS_INTERVALS.keys())
    else:
        raise ValueError(f"Unsupported harte vocabulary: {harte_vocab}")

    root_index = index % 12
    quality_index = index // 12

    root = PITCH_CLASSES[root_index]
    quality = quality_list[quality_index]
    harte_chord = f"{root}:{quality}"
    return Hchord(harte_chord)

# ---------------------------------------------------------------------------
# Frame-level decoder
# ---------------------------------------------------------------------------

def decode_rbp_to_hchord_str(root_idx: int, pitch_bitmap: np.ndarray, bass_idx: int) -> str:
    """Decode a single RBP frame into an HChord string.

    Parameters
    ----------
    root_idx : int
        Predicted root pitch class in ``[0, 11]`` (0 = C).
    pitch_bitmap : array-like of shape ``(12,)``
        Absolute predicted binary pitch-class activity bitmap (values 0 or 1).
    bass_idx : int
        Absolute predicted bass pitch class in ``[0, 11]`` (0 = C).

    Returns
    -------
    str
        String exported from Hchord
    """
    pitch_bitmap = np.asarray(pitch_bitmap, dtype=bool)

    if not pitch_bitmap.any():
        return "N"

    root_name = PITCH_CLASSES[root_idx]

    # Build interval list: semitone offset of each active pitch class relative
    # to root, mapped to Harte interval names.
    intervals: List[str] = []
    for pc in range(12):
        if pitch_bitmap[pc]:
            offset = (pc - root_idx) % 12
            intervals.append(INTERVAL_MAP[offset])

    # Bass annotation: only add explicit bass when bass ≠ root
    bass_offset = (bass_idx - root_idx) % 12
    bass_interval = INTERVAL_MAP[bass_offset] if bass_offset != 0 else None

    return Hchord(root=root_name, intervals=",".join(intervals), bass=bass_interval).export_string()


# ---------------------------------------------------------------------------
# Batched decoder operating on raw model output tensors
# ---------------------------------------------------------------------------

def decode_rbp_sequence(
    root_logits: np.ndarray,
    pitch_logits: np.ndarray,
    bass_logits: np.ndarray,
    pitch_threshold: float = 0.4,
) -> List[str]:
    """Decode a batch of RBP model outputs into sequences of :class:`Hchord`.

    Parameters
    ----------
    root_logits : Tensor of shape ``(12, T)``
        Raw root logits.
    pitch_logits : Tensor of shape ``(T, 12)``
        Raw pitch-class logits; sigmoid + threshold applied to binarise.
    bass_logits : Tensor of shape ``(12, T)``
        Raw bass logits.
    pitch_threshold : float
        Sigmoid threshold for binarising pitch logits (default ``0.4``).

    Returns
    -------
    list of strings
        Shape ``(T, )`` — one string per time frame, exported from Hchord.
    """
    root_pred = root_logits.argmax(axis=0)
    bass_pred = bass_logits.argmax(axis=0)
    pitch_pred = (pitch_logits > pitch_threshold).astype(int)

    seq: List[str] = []
    for t in range(len(root_pred)):
        chord = decode_rbp_to_hchord_str(
            root_idx=int(root_pred[t]),
            pitch_bitmap=pitch_pred[t],
            bass_idx=int(bass_pred[t]),
        )
        seq.append(chord)

    return seq
