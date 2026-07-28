from typing import Optional

import torch
import torch.nn as nn


class HCQTShiftAndNormalize(nn.Module):
    def __init__(self, bins_per_octave=36, num_octaves=6, normalize: bool = True, base_shift: Optional[int] = 0, min_shift: Optional[int] = 0, max_shift: Optional[int] = 11, db_instead_of_log_compression: bool = False):
        super().__init__()

        self.bins_per_octave = bins_per_octave
        self.bins_per_semitone = bins_per_octave // 12
        self.num_octaves = num_octaves
        self.normalize = normalize
        self.db_instead_of_log_compression = db_instead_of_log_compression

        self.base_shift = base_shift
        self.min_shift = min_shift
        self.max_shift = max_shift

    def forward(self, batch, shift, input_key, label_config):
        batch_new, _ = self._shift_inputs(batch, shift, input_key, label_config)
        return batch_new

    def _shift_inputs(self, batch, shift, input_key, label_config):
        x = batch[input_key]
        batch_size = x.shape[0]
        device = x.device
        shift_tensor = self._sample_shift(batch_size, device, shift, self.min_shift, self.max_shift)

        with torch.no_grad():
            cqt_list = []
            batch_indices = torch.arange(batch_size, device=device)[:, None]
            freq_indices = torch.arange(self.num_octaves * self.bins_per_octave, device=device)

            for h in range(x.shape[1]):
                s = x[:, h]
                s = s[batch_indices, self.bins_per_semitone * (self.base_shift + shift_tensor) + freq_indices]
                s = self.normalize_cqt(s)
                cqt_list.append(s.unsqueeze(dim=1))

            s = torch.cat(cqt_list, dim=1)
            batch_new = {input_key: s}

            if label_config.require_harte and "y_chord" in batch:
                y_root = torch.where(batch["y_chord"] != -1, (batch["y_chord"] % 12 - shift_tensor) % 12, -1)
                y_quality = torch.where(batch["y_chord"] != -1, batch["y_chord"] // 12, -1)
                y_chord = torch.where(batch["y_chord"] != -1, y_root + y_quality * 12, -1)
                batch_new["y_chord"] = y_chord
            if label_config.require_rbp and "y_root" in batch and "y_pitch" in batch and "y_bass" in batch:
                y_root = torch.where(batch["y_root"] != -1, (batch["y_root"] - shift_tensor) % 12, -1)
                batch_new["y_root"] = y_root

                y_bass = torch.where(batch["y_bass"] != -1, (batch["y_bass"] - shift_tensor) % 12, -1)

                idx = torch.arange(12, device=batch["y_pitch"].device).view(1, 1, 12)
                shifted_idx = (idx + shift_tensor.unsqueeze(-1)) % 12
                y_pitch = torch.gather(batch["y_pitch"], dim=2, index=shifted_idx.repeat(1, batch["y_pitch"].shape[1], 1))

                batch_new["y_pitch"] = y_pitch
                batch_new["y_bass"] = y_bass

        for key in batch:
            if key not in batch_new:
                batch_new[key] = batch[key]

        return batch_new, shift_tensor

    @staticmethod
    def _sample_shift(batch_size: int, device: torch.device, enable_shift: bool, min_shift: int, max_shift: int) -> torch.Tensor:
        if enable_shift:
            return torch.randint(low=min_shift, high=max_shift + 1, size=(batch_size, 1), device=device)
        return torch.zeros((batch_size, 1), dtype=torch.int64, device=device)
    
    def normalize_cqt(self, cqt: torch.Tensor):
        if self.db_instead_of_log_compression:
            cqt = torch.log(cqt + 1e-7)
        else:
            cqt = torch.log1p(cqt * 1000)
        if self.normalize:
            cqt = (cqt - torch.mean(cqt, dim=(1, 2), keepdim=True)) / (cqt.var(dim=(1, 2), keepdim=True) + 1e-6).sqrt()
        return cqt

class PitchReprShift(HCQTShiftAndNormalize):
    def __init__(
        self,
        bins_per_octave: int = 12,
        num_octaves: int = 6,

        base_shift: Optional[int] = 0,
        min_shift: Optional[int] = 0,
        max_shift: Optional[int] = 11
    ):
        super().__init__(bins_per_octave=bins_per_octave, num_octaves=num_octaves, normalize=False, base_shift=base_shift, min_shift=min_shift, max_shift=max_shift)

    def forward(self, batch, shift, input_key, label_config):
        batch_new, _ = self._shift_inputs(batch, shift, input_key, label_config)
        return batch_new

    def normalize_cqt(self, cqt):
        return cqt  # Pitch features are already processed, so we skip this step.
