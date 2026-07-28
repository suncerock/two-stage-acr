"""Utility for computing HCQT features from audio files."""
import argparse
from pathlib import Path
from typing import List
from tqdm import tqdm

import librosa
import numpy as np
import torch


def audio_to_hcqt(audio, sr_audio, min_note=21, bins_per_octave=36, n_bins=252, harmonics=[0.5, 1, 2, 3, 4, 5], hop_length=512):
    """
        Compute Harmonic CQT

        Parameters
        ----------
        audio_v : np.array
            mono audio signal as np.array
        sr_audio : int
            sampling rate of the audio signal
        param_hcqt : dict
            configuration parameters for the Harmonic CQT

        Returns

        -------
        data_3m : np.array
            3D array of Harmonic CQT
        time_sec_v : np.array
            corresponding time [in sec] of analysis windows
        frequency_hz_v : np.array
            corresponding frequency [in Hz] of 1st harmonic bin
    """
    fmin = float(librosa.midi_to_hz(min_note))
    tuning_est = librosa.estimate_tuning(y=audio)
    fmin_tuned = fmin * 2**(tuning_est / bins_per_octave)

    hcqt_list = []
    min_time_frames = float('inf')
    for h in harmonics:
        A_m = np.abs(librosa.cqt(y=audio, sr=sr_audio,
                                fmin=h*fmin_tuned,
                                hop_length=hop_length,
                                bins_per_octave=bins_per_octave,
                                n_bins=n_bins,
                                tuning=0.0,
                                ))
        hcqt_list.append(A_m)
        min_time_frames = min(min_time_frames, A_m.shape[1])
    
    # Trim all HCQT arrays to the smallest frame length
    hcqt_list = [A[:, :min_time_frames] for A in hcqt_list]
    data_3m = np.stack(hcqt_list, axis=0)
    return data_3m

def compute_features(
    wav_dir: Path,
    tgt_dir_base: Path,
    sr: int = 22050,
    hop_length: float = 0.1,
    bins_per_octave: int = 36,
    n_bins: int = 252,
    harmonics: List[float] = [0.5, 1, 2, 3, 4, 5],
    min_note: float = 21,
    wav_ext: str = "wav",
    force_recompute: bool = False,
) -> None:
    """Process all .{wav_ext} files in ``wav_dir`` and save HCQT tensors under ``tgt_dir_base``."""

    hop_frames = int(sr * hop_length)
    fps = int(1 / hop_length)
    n_harmonics = len(harmonics)
    tgt_dir = tgt_dir_base / f"hcqt_fps{fps}_n{n_bins}_b{bins_per_octave}_min{min_note}_h{n_harmonics}"
    tgt_dir.mkdir(parents=True, exist_ok=True)
    wav_paths = sorted(wav_dir.glob(f"*.{wav_ext}"))

    for wav_path in tqdm(wav_paths, desc="hcqt", unit="track"):
        tgt_path = tgt_dir / wav_path.with_suffix(".pt").name

        if tgt_path.exists() and not force_recompute:
            continue

        audio, sr = librosa.load(wav_path, sr=sr, mono=True)
        audio = audio / np.max(np.abs(audio))  # Normalize audio
        feature = audio_to_hcqt(
            audio, sr, min_note, bins_per_octave, n_bins,
            harmonics, hop_frames
        )

        feature = torch.from_numpy(feature).float()
        torch.save(feature, tgt_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HCQT feature extraction")

    parser.add_argument("--wav-dir", type=Path, required=True,
                        help="Directory containing .wav files")
    parser.add_argument("--tgt-dir-base", type=Path, required=True,
                        help="Base directory for output tensors")

    parser.add_argument("--sr", type=int, default=22050,
                        help="Sample rate")
    parser.add_argument("--hop-length", type=float, default=0.1,
                        help="Hop length in seconds")
    parser.add_argument("--bins-per-octave", type=int, default=36,
                        help="Bins per octave")
    parser.add_argument("--n-bins", type=int, default=252,
                        help="Number of bins")
    parser.add_argument("--harmonics", type=float, nargs="+",
                        default=[1],
                        help="List of harmonic factors (e.g., 0.5 for subharmonic, 1 for fundamental, 2 for octave, etc.)")
    parser.add_argument("--min-note", type=int, default=24,
                        help="MIDI number of the minimum note (e.g., 24 for C1)")
    parser.add_argument("--wav-ext", type=str, default="wav",
                        help="Audio file extension to look for (default: wav)")
    parser.add_argument("--force-recompute", action=argparse.BooleanOptionalAction, default=False,
                        help="Overwrite existing tensors")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compute_features(
        wav_dir=args.wav_dir,
        tgt_dir_base=args.tgt_dir_base,
        sr=args.sr,
        hop_length=args.hop_length,
        bins_per_octave=args.bins_per_octave,
        n_bins=args.n_bins,
        harmonics=args.harmonics,
        min_note=args.min_note,
        wav_ext=args.wav_ext,
        force_recompute=args.force_recompute,
    )


if __name__ == "__main__":
    main()
