import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch.utils.data as Data
from lightning import LightningDataModule

from .dataset import ChordLabelConfig, ChordSingleTrackDataset


# ---------------------------------------------------------------------------
# Label-file extension per dataset
# ---------------------------------------------------------------------------

ext: Dict[str, str] = {
    "swd": "csv",
    "bpsd": "csv",
    "bsqd": "csv",
    "isophonics": "lab",
    "billboard": "chords",
    "dcml": "tsv",
}


# ---------------------------------------------------------------------------
# Data module
# ---------------------------------------------------------------------------

class ACRDatamodule(LightningDataModule):
    """LightningDataModule driven by an explicit JSON split configuration.

    Parameters
    ----------
    datasets : Mapping[str, Mapping]
        Per-dataset configuration keyed by dataset name.  Each value is a
        mapping with the following keys (all paths are strings)::

            label_folder              (required)
            audio_feature_folder      (optional, default None)
            mpe_input_feature_folder  (optional, default None)
            mpe_output_feature_folder (optional, default None)

    split_config : str | Path | dict
        Either a path to a JSON file **or** an already-parsed dict with the
        structure::

            {
              "train": {"dataset_name": ["stem1", "stem2", ...], ...},
              "val":   {...},
              "test":  {...}
            }

        A dataset may appear in any subset of splits.  Datasets absent from
        a split key (or mapped to an empty list) are skipped for that split.
        Relative paths are resolved first against the current working
        directory; when running under Hydra the original working directory
        is tried as a fallback.
    """

    def __init__(
        self,
        datasets: Union[Dict[str, Dict[str, Optional[str]]], Mapping],
        split_config: Union[str, Path, Dict],
        feature_fps: int = 10,
        label_config: Union["ChordLabelConfig", Mapping[str, object], None] = None,
        seg_length: float = 10.0,
        seg_shift_length: float = 1.0,
        use_mmap: bool = False,
        batch_size: int = 32,
        num_workers: int = 8,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.feature_fps = feature_fps
        self.seg_length = seg_length
        self.seg_shift_length = seg_shift_length
        self.label_config = coerce_label_config(label_config)
        self.use_mmap = use_mmap
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Resolve split config
        if isinstance(split_config, (str, Path)):
            split_config = _load_split_config(split_config)

        train_splits: Dict[str, List[str]] = split_config.get("train", {})
        val_splits:   Dict[str, List[str]] = split_config.get("val",   {})
        test_splits:  Dict[str, List[str]] = split_config.get("test",  {})

        # ------------------------------------------------------------------
        # Train + val datasets
        # ------------------------------------------------------------------
        train_ds_list: List[ChordSingleTrackDataset] = []
        val_ds_list:   List[ChordSingleTrackDataset] = []

        for ds_name, ds_cfg in datasets.items():
            train_list = train_splits.get(ds_name) or []
            val_list   = val_splits.get(ds_name)   or []

            label_folder      = ds_cfg["label_folder"]
            audio_folder      = ds_cfg.get("audio_feature_folder")
            mpe_input_folder  = ds_cfg.get("mpe_input_feature_folder")
            mpe_output_folder = ds_cfg.get("mpe_output_feature_folder")

            train_ds_list.extend(
                self._build_track_datasets(
                    train_list, ds_name, label_folder,
                    audio_folder, mpe_input_folder,
                    mpe_output_folder, test=False,
                )
            )
            val_ds_list.extend(
                self._build_track_datasets(
                    val_list, ds_name, label_folder,
                    audio_folder, mpe_input_folder,
                    mpe_output_folder, test=False,
                )
            )

        self.train_dataset = Data.ConcatDataset(train_ds_list)
        self.val_dataset   = Data.ConcatDataset(val_ds_list)

        # ------------------------------------------------------------------
        # Test datasets — one ConcatDataset per dataset that has test tracks
        # ------------------------------------------------------------------
        self.test_datasets: List[Data.ConcatDataset] = []

        for ds_name, ds_cfg in datasets.items():
            test_list = test_splits.get(ds_name) or []
            if not test_list:
                continue

            label_folder      = ds_cfg["label_folder"]
            audio_folder      = ds_cfg.get("audio_feature_folder")
            mpe_input_folder  = ds_cfg.get("mpe_input_feature_folder")
            mpe_output_folder = ds_cfg.get("mpe_output_feature_folder")

            ds = self._build_track_datasets(
                test_list, ds_name, label_folder,
                audio_folder, mpe_input_folder,
                mpe_output_folder, test=True,
            )
            self.test_datasets.append(Data.ConcatDataset(ds))

    # ------------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------------

    def train_dataloader(self):
        return Data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return Data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        def collate_fn(batch: List[Dict[str, Any]]):
            collated: Dict[str, Any] = {}
            for key in batch[0].keys():
                if isinstance(batch[0][key], list) and isinstance(batch[0][key][0], str):
                    collated[key] = [b[key] for b in batch]
                else:
                    collated[key] = Data.default_collate([b[key] for b in batch])
            return collated

        return [
            Data.DataLoader(
                ds,
                batch_size=1,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=collate_fn,
            )
            for ds in self.test_datasets
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_track_datasets(
        self,
        track_list: List[str],
        dataset_name: str,
        label_folder: str,
        audio_feature_folder: Optional[str] = None,
        mpe_input_feature_folder: Optional[str] = None,
        mpe_output_feature_folder: Optional[str] = None,
        test: bool = False,
    ) -> List[ChordSingleTrackDataset]:
        """Build one ChordSingleTrackDataset per track stem."""
        result = []
        for track in track_list:
            def _pt(folder, _track=track):
                return os.path.join(folder, f"{_track}.pt") if folder is not None else None

            result.append(
                ChordSingleTrackDataset(
                    feature_fps=self.feature_fps,
                    label_path=os.path.join(
                        label_folder, f"{track}.{ext[dataset_name]}"
                    ),
                    dataset=dataset_name,
                    label_config=self.label_config,
                    audio_feature_path=_pt(audio_feature_folder),
                    mpe_input_feature_path=_pt(mpe_input_feature_folder),
                    mpe_output_feature_path=_pt(mpe_output_feature_folder),
                    seg_length=self.seg_length if not test else -1,
                    seg_shift_length=self.seg_shift_length,
                    use_mmap=self.use_mmap,
                    test=test,
                )
            )
        return result

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _load_split_config(path: Union[str, Path]) -> Dict:
    """Load a JSON split config, resolving relative paths intelligently.

    Resolution order:
    1. Absolute paths are used as-is.
    2. Relative paths are tried against CWD first.
    3. When running under Hydra, the original working directory is tried
       as a fallback if CWD did not find the file.
    """
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        try:
            import hydra.utils
            candidate = Path(hydra.utils.get_original_cwd()) / path
            if candidate.exists():
                path = candidate
        except Exception:
            pass  # not under Hydra; let open() raise FileNotFoundError
    with open(path) as f:
        return json.load(f)


def coerce_label_config(
    config: Union["ChordLabelConfig", Mapping[str, object], None],
) -> "ChordLabelConfig":
    if config is None:
        return ChordLabelConfig()
    if isinstance(config, ChordLabelConfig):
        return config
    if isinstance(config, Mapping):
        return ChordLabelConfig(**dict(config))
    raise TypeError(
        "label_config must be a ChordLabelConfig, a mapping, or None."
    )
