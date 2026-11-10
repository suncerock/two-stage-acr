from typing import Dict, List
import mir_eval

import numpy as np
import torch
from torchmetrics.functional.classification import binary_average_precision
from torchmetrics import Metric


class RBPEval(Metric):
    def __init__(self, pitch_threshold: float = 0.4):
        """Frame-level evaluation metrics for structured chord labels (root, bass, pitch class).
        """
        super().__init__()

        # Count either frames or samples
        self.add_state("count", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

        self.pitch_threshold = pitch_threshold
        self.add_state("pitch_tp", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("pitch_fp", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("pitch_fn", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("pitch_ap_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.add_state("root_correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("bass_correct", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, output: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]):
        mask = target["root"] != -1
        num_valid = mask.sum()
        if num_valid == 0:
            return

        # either count frames or samples
        self.count += num_valid

  
        y_root = target["root"][mask]
        y_hat_root = output["root"].argmax(dim=1)[mask]
        correct = (y_hat_root == y_root).float().sum()
        self.root_correct += correct

        y_bass = target["bass"][mask]
        y_hat_bass = output["bass"].argmax(dim=1)[mask]
        correct = (y_hat_bass == y_bass).float().sum()
        self.bass_correct += correct

        y_pitch = target["pitch"][mask].long()
        y_hat_pitch = (torch.sigmoid(output["pitch"][mask]) > self.pitch_threshold).long()

        tp = (y_hat_pitch & y_pitch).sum().float()
        fp = ((y_hat_pitch == 1) & (y_pitch == 0)).sum().float()
        fn = ((y_hat_pitch == 0) & (y_pitch == 1)).sum().float()

        self.pitch_tp += tp
        self.pitch_fp += fp
        self.pitch_fn += fn

        ap = binary_average_precision(preds=output["pitch"][mask], target=y_pitch)
        self.pitch_ap_sum += ap * num_valid

    def compute(self):
        metrics = {}
        count = self.count.clamp_min(1)

        metrics["root_acc"] = self.root_correct / count
        metrics["bass_acc"] = self.bass_correct / count

        precision = self.pitch_tp / (self.pitch_tp + self.pitch_fp + 1e-8)
        recall = self.pitch_tp / (self.pitch_tp + self.pitch_fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        metrics.update({
            "pitch_f1": f1,
            "pitch_ap": self.pitch_ap_sum / count,
        })

        return metrics


class ChordAccuracy(Metric):
    """Fast frame-level chord accuracy, ignoring invalid frames (label == -1).

    Suitable for per-step logging during training/validation without the
    overhead of mir_eval string conversion.
    """

    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("total",   default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Parameters
        ----------
        logits : Tensor shape (N, C, T)
        targets : Tensor shape (N, T), with -1 for invalid frames
        """
        valid = targets != -1
        preds = logits.argmax(dim=1)          # (N, T)
        self.correct += (preds[valid] == targets[valid]).sum()
        self.total   += valid.sum()

    def compute(self) -> torch.Tensor:
        return self.correct.float() / self.total.clamp_min(1).float()


class ChordEval(Metric):
    def __init__(self, include_no_chord: bool = False):
        super().__init__()
        self.include_no_chord = include_no_chord

        for rules in ["root", "thirds", "triads", "sevenths", "tetrads", "majmin", "mirex"]:
            self.add_state(rules, default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state(f"count_{rules}", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def _accumulate_strings(self, pred_str: List[str], true_str: List[str]) -> None:
        """Accumulate mir_eval scores for one pair of chord-string sequences."""

        def _accum(scores: np.ndarray, score_state, count_state):
            valid = scores[scores != -1]
            score_state += float(valid.sum())
            count_state += int(len(valid))

        _accum(mir_eval.chord.root(true_str, pred_str),      self.root,      self.count_root)
        _accum(mir_eval.chord.thirds(true_str, pred_str),    self.thirds,    self.count_thirds)
        _accum(mir_eval.chord.triads(true_str, pred_str),    self.triads,    self.count_triads)
        _accum(mir_eval.chord.sevenths(true_str, pred_str),  self.sevenths,  self.count_sevenths)
        _accum(mir_eval.chord.tetrads(true_str, pred_str),   self.tetrads,   self.count_tetrads)
        _accum(mir_eval.chord.majmin(true_str, pred_str),    self.majmin,    self.count_majmin)
        _accum(mir_eval.chord.mirex(true_str, pred_str),     self.mirex,     self.count_mirex)

    def update(self, y_hat: List[str], y: List[str]) -> None:
        """Accumulate mir_eval scores from pre-decoded Harte chord string sequences.

        Parameters
        ----------
        y_hat : list of str
            Predicted chord strings for a single track/segment.
        y : list of str
            Ground-truth chord strings for a single track/segment.
            Frames where the ground-truth is ``'X'`` are ignored, as are ``'N'``
            frames unless ``include_no_chord`` is set.
        """
        ignored = ('X',) if self.include_no_chord else ('N', 'X')
        y_hat_masked = [p for p, t in zip(y_hat, y) if t not in ignored]
        y_masked = [t for t in y if t not in ignored]

        if y_masked:
            self._accumulate_strings(y_hat_masked, y_masked)

    def compute(self):
        def _div(score_state, count_state):
            return score_state / count_state.clamp_min(1).float()

        return {
            "root":     _div(self.root,      self.count_root),
            "thirds":   _div(self.thirds,    self.count_thirds),
            "triads":   _div(self.triads,    self.count_triads),
            "sevenths": _div(self.sevenths,  self.count_sevenths),
            "tetrads":  _div(self.tetrads,   self.count_tetrads),
            "majmin":   _div(self.majmin,    self.count_majmin),
            "mirex":    _div(self.mirex,     self.count_mirex),
        }
