from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data

from src.data.hchord import Hchord
from src.data.utils import categorize_hchord

read_label_csv_kwargs = {
    "swd": {"sep": ";", "header": 0},
    "bpsd": {"sep": ";", "header": 0},
    "bsqd": {"sep": ";", "header": None, "index_col": False, "names": ['start', 'end', 'shorthand', 'extended', 'majmin_inv', 'majmin']},
    "isophonics": {"sep": r"\s+", "header": None, "names": ['start', 'end', 'shorthand']},
    "billboard": {"sep": r"\s+", "header": None, "names": ['start', 'end', 'shorthand']},
    "dcml": {"sep": "\t", "header": 0},
}

label_entry = {
    "swd": "extended",
    "bpsd": "extended",
    "bsqd": "extended",
    "isophonics": "shorthand",
    "billboard": "shorthand",
    "dcml": "extended",
}


@dataclass(frozen=True)
class ChordLabelConfig:
    """Configuration flags describing which chord-label representations to load."""

    require_harte: bool = False
    harte_vocab: str = "tetrads"
    # Add an explicit N class: index 12 for root/bass, last index for the Harte chord head.
    include_no_chord: bool = False

    require_rbp: bool = False
    rbp_enforce_root: bool = False
    rbp_enforce_bass: bool = False
    rbp_include_extended: bool = False

    @classmethod
    def rbp_only(cls, include_no_chord: bool = False) -> "ChordLabelConfig":
        return cls(require_rbp=True, include_no_chord=include_no_chord)

    @classmethod
    def mcfee_structured(cls, harte_vocab: str = "tetrads", include_no_chord: bool = False) -> "ChordLabelConfig":
        return cls(
            require_harte=True,
            harte_vocab=harte_vocab,
            require_rbp=True,
            include_no_chord=include_no_chord,
        )

    @classmethod
    def poltronieri_structured(cls, include_no_chord: bool = False) -> "ChordLabelConfig":
        return cls.rbp_only(include_no_chord=include_no_chord)

class ChordSingleTrackDataset(Data.Dataset):
    def __init__(
        self,
        feature_fps: float,
        label_path: str,
        dataset: str,
        label_config: Optional[ChordLabelConfig] = None,

        audio_feature_path: Optional[str] = None,
        mpe_input_feature_path: Optional[str] = None,
        mpe_output_feature_path: Optional[str] = None,

        seg_length: float = 10.0,
        seg_shift_length: float = 2.0,

        use_mmap: bool = False,
        test: bool = False,
    ) -> None:
        """
        Dataset for a single track, concatenated later into a single dataset.

        Supported input combinations
        ----------------------------
        - ``audio_feature_path`` only        - audio feature as sole input.
        - ``mpe_output_feature_path`` only   - precomputed MPE tensor as sole input.
        - ``mpe_input_feature_path`` only    - high-fps audio feature for MPE processing.

        Parameters
        ----------
        audio_feature_path : Optional[str]
            Path to the audio input feature (.pt), aligned to feature_fps.
        mpe_input_feature_path : Optional[str]
            Path to the high-res MPE input feature (.pt), uses mpe_input_feature_fps.
        feature_fps : float
            Frame rate shared by all feature tensors.
        label_path : str
            Path to the chord label file.
        dataset : str
            Dataset key (controls CSV parsing and label field).
        label_config : Optional[ChordLabelConfig]
            Which label representations to build; defaults to empty config.
        mpe_output_feature_path : Optional[str]
            Path to the precomputed MPE tensor (.pt), already normalized by sigmoid
            and resampled to feature_fps.
        seg_length : float
            Segment length in seconds (0 or negative → whole track).
        seg_shift_length : float
            Stride between consecutive segments in seconds.
        use_mmap : bool
            If True, load feature tensors using memory mapping to reduce memory usage.
        """
        super().__init__()

        # ------------------------------------------------------------------ #
        # Validate input combination
        # ------------------------------------------------------------------ #
        if audio_feature_path is None and mpe_input_feature_path is None and mpe_output_feature_path is None:
            raise ValueError(
                "At least one of audio_feature_path, mpe_input_feature_path, or mpe_output_feature_path must be provided."
            )

        self.audio_feature_path = audio_feature_path
        self.mpe_input_feature_path = mpe_input_feature_path
        self.mpe_output_feature_path = mpe_output_feature_path
        self.feature_fps = feature_fps
        self.label_path = label_path

        self.dataset = dataset
        # Derive track key from whichever path is available
        key_path = audio_feature_path or mpe_input_feature_path or mpe_output_feature_path
        self.key = key_path.replace(".pt", "").split("/")[-1]

        self.seg_length = seg_length
        self.seg_shift_length = seg_shift_length

        self.label_config = label_config or ChordLabelConfig()
        self.require_harte = self.label_config.require_harte
        self.require_rbp = self.label_config.require_rbp
        self.include_no_chord = self.label_config.include_no_chord
        self.test = test

        self.feature_seg_frames = int(feature_fps * seg_length) if seg_length > 0.0 else -1
        self.feature_seg_shift_frames = int(feature_fps * seg_shift_length)

        # ------------------------------------------------------------------ #
        # Load the label file
        # ------------------------------------------------------------------ #
        self.label_df: pd.DataFrame = pd.read_csv(label_path, **read_label_csv_kwargs[dataset])
        self.label_df = self.label_df.astype({"start": "float32", "end": "float32"})

        # ------------------------------------------------------------------ #
        # Load feature tensors
        # ------------------------------------------------------------------ #
        self.audio_feature: Optional[torch.Tensor] = None
        if audio_feature_path is not None:
            self.audio_feature = torch.load(audio_feature_path, map_location="cpu", mmap=use_mmap, weights_only=True)

        self.mpe_input_feature: Optional[torch.Tensor] = None
        if mpe_input_feature_path is not None:
            self.mpe_input_feature = torch.load(mpe_input_feature_path, map_location="cpu", mmap=use_mmap, weights_only=True)

        self.mpe_output_feature: Optional[torch.Tensor] = None
        if mpe_output_feature_path is not None:
            self.mpe_output_feature = torch.load(mpe_output_feature_path, map_location="cpu", mmap=use_mmap, weights_only=True).float()

        # The length of the sequence is dictated by the ACR features
        if self.mpe_output_feature is not None:
            self.acr_frame_count = self.mpe_output_feature.shape[-1]
        elif self.audio_feature is not None:
            self.acr_frame_count = self.audio_feature.shape[-1]
        elif self.mpe_input_feature is not None:
            self.acr_frame_count = self.mpe_input_feature.shape[-1]
        else:
            raise ValueError("At least one feature tensor must be provided to determine sequence length.")

        if seg_length > 0.0:
            self.total_segments = (self.acr_frame_count - self.feature_seg_frames) // self.feature_seg_shift_frames
        else:
            self.total_segments = 1

        self.labels = self.load_labels()
        self.valid_segment_indices: List[int] = self._compute_valid_indices()
        self.length = len(self.valid_segment_indices)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.seg_length > 0:
            segment_idx = self.valid_segment_indices[index]
            start_frame = segment_idx * self.feature_seg_shift_frames
            frame = np.arange(start_frame, start_frame + self.feature_seg_frames)
        else:
            frame = np.arange(self.acr_frame_count)

        start_idx = int(frame[0])
        end_idx = int(frame[-1]) + 1

        data = dict(key=self.key, dataset=self.dataset)

        if self.audio_feature is not None:
            data["x_audio"] = self.audio_feature[..., start_idx:end_idx]

        if self.mpe_input_feature is not None:
            data["x_mpe_input"] = self.mpe_input_feature[..., start_idx:end_idx]

        # MPE feature – precomputed, frozen MPE activations
        if self.mpe_output_feature is not None:
            mpe_segment = self.mpe_output_feature[..., start_idx:end_idx]
            data["x_mpe_output"] = mpe_segment.unsqueeze(0).contiguous()

        for label_name, label_values in self.labels.items():
            if isinstance(label_values, list):
                # Python list (e.g. y_hchord_str)
                data[label_name] = [label_values[i] for i in frame]
            else:
                data[label_name] = label_values[frame]

        return data

    def _compute_valid_indices(self):
        valid_frames = self._build_valid_frame_mask()

        if self.seg_length <= 0:
            return [0] if valid_frames.any() else []

        indices: List[int] = []
        for segment_idx in range(self.total_segments):
            start = segment_idx * self.feature_seg_shift_frames
            end = start + self.feature_seg_frames

            if valid_frames[start:end].any():
                indices.append(segment_idx)
        return indices

    def _build_valid_frame_mask(self) -> np.ndarray:
        time_frame = np.arange(self.acr_frame_count) / self.feature_fps
        valid_mask = np.zeros(self.acr_frame_count, dtype=bool)

        for _, row in self.label_df.iterrows():
            if row[label_entry[self.dataset]] == "X":
                continue
            chord = Hchord(row[label_entry[self.dataset]])
            if chord.root is None and not self.include_no_chord:
                continue
            frame_mask = (
                (time_frame - 0.5 / self.feature_fps > row["start"]) &
                (time_frame + 0.5 / self.feature_fps < row["end"])
            )
            valid_mask |= frame_mask

        return valid_mask

    def load_labels(self):
        frame = np.arange(self.acr_frame_count)
        time_frame = frame / self.feature_fps

        labels = dict()

        if self.include_no_chord:
            # Regions before the first and after the last annotation are no-chord
            first_start = float(self.label_df["start"].min()) if not self.label_df.empty else 0.0
            last_end = float(self.label_df["end"].max()) if not self.label_df.empty else 0.0
            track_end = self.acr_frame_count / self.feature_fps
            boundary_rows = []
            if first_start > 0.0:
                boundary_rows.append({"start": 0.0, "end": first_start, label_entry[self.dataset]: "N"})
            if last_end < track_end:
                boundary_rows.append({"start": last_end, "end": track_end, label_entry[self.dataset]: "N"})
            if boundary_rows:
                self.label_df = pd.concat([self.label_df, pd.DataFrame(boundary_rows)], ignore_index=True)
                self.label_df = self.label_df.sort_values(["start", "end"], kind="stable").reset_index(drop=True)

        if self.require_harte:
            chord_labels = -1 * np.ones(len(time_frame), dtype=np.int64)
            for _, row in self.label_df.iterrows():
                if row[label_entry[self.dataset]] == "X":
                    continue

                chord = Hchord(row[label_entry[self.dataset]])

                chord_labels[(time_frame - 0.5 / self.feature_fps > row["start"]) & (time_frame + 0.5 / self.feature_fps < row["end"])] = categorize_hchord(
                    chord,
                    vocab=self.label_config.harte_vocab,
                    include_no_chord=self.include_no_chord,
                )
                
            labels["y_chord"] = chord_labels
        
        if self.require_rbp:
            # Frames without a valid label (X, or N when disabled) keep -1 everywhere
            root_labels = -1 * np.ones(len(time_frame), dtype=np.int64)
            pitch_labels = -1 * np.ones((len(time_frame), 12), dtype=np.float32)
            bass_labels = -1 * np.ones(len(time_frame), dtype=np.int64)

            for _, row in self.label_df.iterrows():
                if row[label_entry[self.dataset]] == "X":
                    continue

                chord = Hchord(row[label_entry[self.dataset]])
                if chord.root is None and not self.include_no_chord:
                    continue

                root, bitmap, bass = chord.encode(
                    absolute=True,
                    enforce_root=self.label_config.rbp_enforce_root,
                    enforce_bass=self.label_config.rbp_enforce_bass,
                    include_extended=self.label_config.rbp_include_extended,
                    include_no_chord=self.include_no_chord,
                )

                root_labels[(time_frame - 0.5 / self.feature_fps > row["start"]) & (time_frame + 0.5 / self.feature_fps < row["end"])] = root
                pitch_labels[(time_frame - 0.5 / self.feature_fps > row["start"]) & (time_frame + 0.5 / self.feature_fps < row["end"]), :] = bitmap
                bass_labels[(time_frame - 0.5 / self.feature_fps > row["start"]) & (time_frame + 0.5 / self.feature_fps < row["end"])] = bass

            labels["y_root"] = root_labels
            labels["y_pitch"] = pitch_labels
            labels["y_bass"] = bass_labels

        if self.test:
            # Unannotated frames are "X" and ignored by evaluation
            chord_str_labels: List[str] = ["X"] * len(time_frame)
            for _, row in self.label_df.iterrows():
                if row[label_entry[self.dataset]] == "X":
                    continue
                chord_str = Hchord(row[label_entry[self.dataset]]).export_string()
                frame_mask = (
                    (time_frame - 0.5 / self.feature_fps > row["start"]) &
                    (time_frame + 0.5 / self.feature_fps < row["end"])
                )
                for i in np.where(frame_mask)[0]:
                    chord_str_labels[i] = chord_str
            labels["y_hchord_str"] = chord_str_labels

        return labels
