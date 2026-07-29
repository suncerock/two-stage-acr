#!/usr/bin/env python3
"""Run two-stage chord inference on a single audio file.

The script supports the four inference checkpoints that remain in the compact
repo:
CRNN or Conformer × Classical or Pop.

The runtime pipeline is fixed to:
1. Load mono audio at 22050 Hz.
2. Compute a 6-harmonic HCQT with 512-sample hop.
3. Run the shared MPE front-end on the HCQT.
4. Downsample the MPE output to the requested chord fps.
5. Run the chord model and save both soft outputs and decoded segments.
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch.nn as nn
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.data.dataset import ChordLabelConfig
from src.models.decoder import convert_harte_index_to_hchord, decode_rbp_sequence
from src.models.backbone.conformer import Conformer
from src.models.backbone.crnn import CRNN
from src.models.frontend.classic_pitch import HCQTEncoder
from src.models.model import ACRModel, BackboneStageConfig, MPEStageConfig


HCQT_SR = 22050
HCQT_HOP_LENGTH = 512
HCQT_MIN_NOTE = 24
HCQT_BINS_PER_OCTAVE = 36
HCQT_N_BINS = 216
HCQT_HARMONICS = [0.5, 1, 2, 3, 4, 5]
HCQT_SOURCE_FPS = HCQT_SR / HCQT_HOP_LENGTH
DEFAULT_MPE_CKPT = _REPO_ROOT / "ckpt" / "pretrained_mpe.ckpt"
FPS = 10.0
DECODE_PITCH_THRESHOLD = 0.4

def _normalize_audio(audio: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(audio)))
    if peak == 0.0:
        raise ValueError("Audio signal is silent; cannot compute HCQT for inference.")
    return audio / peak

def _audio_path_to_hcqt(audio_path: Path) -> torch.Tensor:
    import librosa

    from scripts.compute_audio_hcqt import audio_to_hcqt

    audio, _ = librosa.load(str(audio_path), sr=HCQT_SR, mono=True)
    audio = _normalize_audio(audio)
    hcqt = audio_to_hcqt(
        audio,
        sr_audio=HCQT_SR,
        min_note=HCQT_MIN_NOTE,
        bins_per_octave=HCQT_BINS_PER_OCTAVE,
        n_bins=HCQT_N_BINS,
        harmonics=HCQT_HARMONICS,
        hop_length=HCQT_HOP_LENGTH,
    )
    return torch.from_numpy(hcqt).float().unsqueeze(0)


def _downsample_mpe_output(
    mpe_output: torch.Tensor,
    source_fps: float,
    target_fps: float,
) -> torch.Tensor:
    target_frames = round(mpe_output.shape[-1] * target_fps / source_fps)
    if target_frames <= 0:
        raise ValueError("Downsampled MPE output would have zero frames.")
    return torch.nn.functional.interpolate(
        mpe_output,
        size=(mpe_output.shape[2], target_frames),
        mode="bilinear",
        align_corners=False,
    )

def _resolve_output_paths(output_path: Path) -> tuple[Path, Path]:
    base_path = output_path if output_path.suffix == "" else output_path.with_suffix("")
    base_path.parent.mkdir(parents=True, exist_ok=True)
    return base_path.with_suffix(".npz"), base_path.with_suffix(".lab")

def _build_model(
    backbone_type: str,
    chord_ckpt_path: Path,
    mpe_ckpt_path: Path,
) -> ACRModel:
    class DummyLoss(nn.Module):
        def setup_loss(self, *args, **kwargs):
            pass

    if backbone_type == "crnn":
        backbone = CRNN(input_channels=1, freq_dim=72, num_channels=32, num_rnn_layers=4, dropout=0.2)
        label_config = ChordLabelConfig.mcfee_structured(harte_vocab="tetrads")
        pitch_informed_rbp = False
        rbp_informed_chord = True
    elif backbone_type == "conformer":
        backbone = Conformer(freq_dim=72, hidden_size=192, num_hidden_layers=4, intermediate_size=768)
        label_config = ChordLabelConfig.poltronieri_structured()
        pitch_informed_rbp = True
        rbp_informed_chord = False
    else:
        raise ValueError(f"Unsupported backbone type: {backbone_type!r}")

    model = ACRModel(
        feature=nn.Identity(),  # dummy module - not used in inference
        backbone=backbone,
        loss_fn=DummyLoss(),  # dummy module - not used in inference
        optim_cfg={"optimizer": {}, "scheduler": {}},
        mpe_model=HCQTEncoder(),
        mpe_stage=MPEStageConfig(freeze=True, checkpoint_path=str(mpe_ckpt_path)),
        backbone_stage=BackboneStageConfig(freeze=True, checkpoint_path=str(chord_ckpt_path)),
        model_input_key="x_mpe_input",
        label_config=label_config,
        pitch_informed_rbp=pitch_informed_rbp,
        rbp_informed_chord=rbp_informed_chord,
    )
    model.eval()
    return model

def _group_frames_to_segments(
    chord_strings: List[str],
    fps: float,
) -> Tuple[np.ndarray, List[str]]:
    """Group consecutive identical frame labels into timed (start, end) segments."""
    assert len(chord_strings) > 0, "Input chord_strings must not be empty."

    intervals: List[List[float]] = []
    labels: List[str] = []
    start = 0
    current = chord_strings[0]
    T = len(chord_strings)

    for t in range(1, T):
        if chord_strings[t] != current:
            intervals.append([start / fps, t / fps])
            labels.append(current)
            start = t
            current = chord_strings[t]

    intervals.append([start / fps, T / fps])
    labels.append(current)
    return np.array(intervals, dtype=np.float64), labels

def run_inference(
    model: ACRModel,
    audio_path: Path,
    output_path: Path,
    save_soft_output: bool = False,
) -> None:
    if model.mpe_model is None:
        raise ValueError("This inference script requires a two-stage model with an MPE front-end.")

    hcqt = _audio_path_to_hcqt(audio_path)
    hcqt = torch.log1p(hcqt * 1000)

    with torch.no_grad():
        mpe_output = model._run_mpe_stage(hcqt)
        chord_input = _downsample_mpe_output(mpe_output, source_fps=HCQT_SOURCE_FPS, target_fps=FPS)
        embedding = model.backbone(chord_input)
        predictions = model._forward_heads(embedding)
        predictions = {k: v.squeeze(0).float().cpu().numpy() for k, v in predictions.items()}

    if "pitch" in predictions:
        predictions["pc"] = predictions.pop("pitch").T

    npz_path, lab_path = _resolve_output_paths(output_path)
    if save_soft_output:
        np.savez_compressed(npz_path, **predictions)

    if "chord" in predictions:
        vocab = model.label_config.harte_vocab
        chord_indices = np.argmax(predictions["chord"], axis=0)    # (T,)
        frame_chord_strings = [
            convert_harte_index_to_hchord(int(idx), harte_vocab=vocab).export_string()
            for idx in chord_indices
        ]
    
    elif "root" in predictions and "pc" in predictions and "bass" in predictions:
        frame_chord_strings = decode_rbp_sequence(
            root_logits=predictions["root"],
            pitch_logits=1 / (1 + np.exp(-predictions["pc"].T)),  # sigmoid + transpose to (T, 12)
            bass_logits=predictions["bass"],
            pitch_threshold=DECODE_PITCH_THRESHOLD,
        )
    
    else:
        raise ValueError("Predictions must contain either 'chord' or all of 'root', 'pc', and 'bass'.")
    
    intervals, labels = _group_frames_to_segments(frame_chord_strings, FPS)
    with open(lab_path, "w") as f:
        for (start, end), label in zip(intervals, labels):
            f.write(f"{start:.6f}\t{end:.6f}\t{label}\n")

    print(f"Audio path    : {audio_path}")
    if save_soft_output:
        print(f"NPZ output   : {npz_path}")
    print(f"LAB output    : {lab_path}")
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run two-stage chord inference on one audio file."
    )
    parser.add_argument(
        "--audio-path", required=True, type=Path,
        help="Path to the input audio file.",
    )
    parser.add_argument(
        "--output-path", required=True, type=Path,
        help="Output file stem; the script writes both .npz and .lab files using this path.",
    )
    parser.add_argument(
        "--backbone-type",
        choices=["crnn", "conformer"],
        default="crnn",
        help="Chord-model backbone family to use (default: crnn).",
    )
    parser.add_argument(
        "--model-type",
        choices=["classical", "pop"],
        default=None,
        help="Checkpoint to use (classical or pop).",
    )
    parser.add_argument(
        "--mpe-ckpt-path",
        type=Path,
        default=DEFAULT_MPE_CKPT,
        help="Path to the shared MPE frontend checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--soft-output",
        action="store_true",
        help="Save .npz soft outputs in addition to .lab decoded segments.",
    )
    args = parser.parse_args()

    chord_ckpt_path = Path(f"ckpt/{args.backbone_type}_{args.model_type}.ckpt")
    if not chord_ckpt_path.exists():
        raise FileNotFoundError(f"Chord checkpoint not found: {chord_ckpt_path}. ")
    if not args.mpe_ckpt_path.exists():
        raise FileNotFoundError(f"MPE checkpoint not found: {args.mpe_ckpt_path}")

    print(f"Backbone     : {args.backbone_type}")
    print(f"Trained on   : {args.model_type}")
    print(f"MPE ckpt     : {args.mpe_ckpt_path}")
    print(f"Chord ckpt   : {chord_ckpt_path}")

    model = _build_model(
        backbone_type=args.backbone_type,
        chord_ckpt_path=chord_ckpt_path,
        mpe_ckpt_path=args.mpe_ckpt_path,
    )

    run_inference(
        model=model,
        audio_path=args.audio_path,
        output_path=args.output_path,
        save_soft_output=args.soft_output,
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
