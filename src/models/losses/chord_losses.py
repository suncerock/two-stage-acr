from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class ChordLoss(nn.Module):
    def __init__(self, ce_label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.ce_label_smoothing = ce_label_smoothing
        self.label_config = None
        self.chord_loss = None
        self.root_loss = None
        self.bass_loss = None
        self.pitch_loss = None

    def setup_loss(self, label_config) -> None:
        self.label_config = label_config
        if self.label_config.require_harte:
            self.chord_loss = nn.CrossEntropyLoss(
                ignore_index=-1,
                label_smoothing=self.ce_label_smoothing,
            )
        if self.label_config.require_rbp:
            self.root_loss = nn.CrossEntropyLoss(
                ignore_index=-1,
                label_smoothing=self.ce_label_smoothing,
            )
            self.bass_loss = nn.CrossEntropyLoss(
                ignore_index=-1,
                label_smoothing=self.ce_label_smoothing,
            )
            self.pitch_loss = nn.BCEWithLogitsLoss()

    def forward(self, output: Dict, target: Dict):
        if self.label_config is None:
            raise RuntimeError("ChordLoss.setup_loss must be called before forward")

        targets = {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in target.items()
        }

        losses: Dict[str, torch.Tensor] = {}

        if self.label_config.require_rbp:
            mask = (targets["pitch"] != -1).any(dim=-1)  # exclude frames without valid labels
            if mask.any():
                losses["pitch"] = self.pitch_loss(
                    output["pitch"][mask], targets["pitch"][mask].float()
                )
            losses["root"] = self.root_loss(output["root"], targets["root"])
            losses["bass"] = self.bass_loss(output["bass"], targets["bass"])

        if self.label_config.require_harte:
            losses["chord"] = self.chord_loss(output["chord"], targets["chord"])

        for key, loss in losses.items():
            if torch.isnan(loss):
                losses[key] = torch.tensor(0.0, device=loss.device)

        if not losses:
            raise RuntimeError("ChordLoss received no active loss terms")

        loss = sum(losses.values()) / len(losses)
        return dict(total=loss, **losses)
