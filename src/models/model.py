import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from lightning import LightningModule

from src.data.dataset import ChordLabelConfig
from src.data.datamodule import coerce_label_config
from src.data.utils import ALL_TETRADS_INTERVALS, ALL_TRIADS_INTERVALS, num_harte_classes
from src.models.eval import ChordEval, ChordAccuracy, RBPEval
from src.models.decoder import decode_rbp_sequence, convert_harte_index_to_hchord


HARTE_CLASS_FACTORS = {
    "triads": len(ALL_TRIADS_INTERVALS),
    "tetrads": len(ALL_TETRADS_INTERVALS),
}

class RBPInformedChordHead(nn.Module):
    """Root-Bass-Pitch-informed chord head used by the CRNN backbone.

    Architecture
    ------------
    1. RBP predictions are produced directly from the backbone embedding
       (N, T, C) via linear layers.
    2. They are concatenated back to the embedding and fed
       through a linear layer to obtain the chord predictions.
    
    Parameters
    ----------
    feature_dim : int
        Dimensionality of the backbone output embedding.
    """
    def __init__(self, feature_dim: int, num_classes: int, include_no_chord: bool = False) -> None:
        super().__init__()
        self.num_classes = num_classes
        num_rb_classes = 12 + int(include_no_chord)
        self.root_head = nn.Linear(feature_dim, num_rb_classes)
        self.bass_head = nn.Linear(feature_dim, num_rb_classes)
        self.pitch_head = nn.Linear(feature_dim, 12)
        self.chord_head = nn.Linear(feature_dim + num_rb_classes + num_rb_classes + 12, num_classes)

    def forward(self, embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        embedding : (N, T, C)

        Returns
        -------
        chord_logits : (N, num_classes, T) - permuted to match ACRModel convention
        """
        root_logits = self.root_head(embedding)  # (N, T, 12(+1))
        bass_logits = self.bass_head(embedding)  # (N, T, 12(+1))
        pitch_logits = self.pitch_head(embedding)  # (N, T, 12)

        fused = torch.cat([embedding, root_logits, bass_logits, pitch_logits], dim=-1)  # (N, T, C+12+12+12)
        chord_logits = self.chord_head(fused).permute(0, 2, 1)  # (N, num_classes, T)
        root_logits = root_logits.permute(0, 2, 1)   # (N, 12(+1), T)
        bass_logits = bass_logits.permute(0, 2, 1)   # (N, 12(+1), T)

        return dict(chord=chord_logits, root=root_logits, bass=bass_logits, pitch=pitch_logits)

class PitchInformedRBPHead(nn.Module):
    """Pitch-informed Root-Bitmap-Bass head used by the Conformer backbone.

    Architecture
    ------------
    1. A linear ``pitch`` head produces 12-dim pitch-class logits directly from
       the backbone embedding  (N, T, 12).
    2. Root and bass predictions are each conditioned on both the embedding *and*
       the pitch logits via a small fusion MLP, so they can leverage the
       pitch-class information without being constrained to it.

    Parameters
    ----------
    feature_dim : int
        Dimensionality of the backbone output embedding.
    fusion_dropout : float
        Dropout probability inside the fusion MLP.
    """

    def __init__(self, feature_dim: int, fusion_dropout: float = 0.1, include_no_chord: bool = False) -> None:
        super().__init__()

        # Primary pitch-class prediction (12 binary logits)
        self.pitch = nn.Linear(feature_dim, 12)

        fused_dim = feature_dim + 12  # embedding ‖ pitch logits

        self.root_fusion = nn.Sequential(
            nn.Linear(fused_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
        )
        self.root = nn.Linear(feature_dim, 12 + int(include_no_chord))

        self.bass_fusion = nn.Sequential(
            nn.Linear(fused_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
        )
        self.bass = nn.Linear(feature_dim, 12 + int(include_no_chord))

    def forward(self, embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        embedding : (N, T, C)

        Returns
        -------
        pitch_logits : (N, T, 12)
        root_logits  : (N, 12, T)   - permuted to match ACRModel convention
        bass_logits  : (N, 12, T)   - permuted to match ACRModel convention
        """
        pitch_logits = self.pitch(embedding)                            # (N, T, 12)

        fused = torch.cat([embedding, pitch_logits], dim=-1)            # (N, T, C+12)
        root_logits = self.root(self.root_fusion(fused)).permute(0, 2, 1)  # (N, 12(+1), T)
        bass_logits = self.bass(self.bass_fusion(fused)).permute(0, 2, 1)  # (N, 12(+1), T)

        return dict(pitch=pitch_logits, root=root_logits, bass=bass_logits)


@dataclass(frozen=True)
class MPEStageConfig:
    freeze: bool = True
    checkpoint_path: Optional[str] = None


def coerce_mpe_stage_config(
    config: Union[MPEStageConfig, Mapping[str, object], None]
) -> MPEStageConfig:
    if config is None:
        return MPEStageConfig()
    if isinstance(config, MPEStageConfig):
        return config
    if isinstance(config, Mapping):
        return MPEStageConfig(**dict(config))
    raise TypeError("mpe_stage must be a MPEStageConfig, mapping, or None.")


@dataclass(frozen=True)
class BackboneStageConfig:
    """Configuration for loading and optionally freezing a pretrained backbone.

    Intended for the scenario where the backbone (e.g. score-to-chord CRNN) was
    pretrained independently and is now combined with a pretrained MPE model.

    freeze : bool
        If True, the backbone and classification heads are frozen (no gradient
        updates). Useful for the "both frozen (test only)" regime.
    checkpoint_path : Optional[str]
        Path to a Lightning checkpoint (.ckpt) or a raw state-dict (.pt/.pth)
        from which backbone weights (and optionally head weights) are loaded.
        The loader tries ``state_dict`` key first (Lightning format), then falls
        back to treating the file as a raw state-dict.
    """
    freeze: bool = False
    checkpoint_path: Optional[str] = None


def coerce_backbone_stage_config(
    config: Union[BackboneStageConfig, Mapping[str, object], None]
) -> BackboneStageConfig:
    if config is None:
        return BackboneStageConfig()
    if isinstance(config, BackboneStageConfig):
        return config
    if isinstance(config, Mapping):
        return BackboneStageConfig(**dict(config))
    raise TypeError("backbone_stage must be a BackboneStageConfig, mapping, or None.")


class ACRModel(LightningModule):
    def __init__(
        self,
        feature: nn.Module,
        backbone: nn.Module,
        loss_fn: nn.Module,
        optim_cfg: Dict,
        mpe_model: Optional[nn.Module] = None,
        mpe_stage: Union[MPEStageConfig, Mapping[str, object], None] = None,
        backbone_stage: Union[BackboneStageConfig, Mapping[str, object], None] = None,
        model_input_key: str = "x",
        label_config: Union[ChordLabelConfig, Mapping[str, object], None] = None,
        
        pitch_informed_rbp: bool = False,
        rbp_informed_chord: bool = False,

        compile: bool = True,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=("feature", "backbone", "loss_fn", "mpe_model"))

        self.feature = feature
        self.backbone = backbone
        self.backbone_stage_cfg = coerce_backbone_stage_config(backbone_stage)
        self.loss_fn = loss_fn

        self.label_config = coerce_label_config(label_config)
        self.pitch_informed_rbp = pitch_informed_rbp
        self.rbp_informed_chord = rbp_informed_chord

        self.mpe_model = mpe_model
        self.mpe_stage_cfg = coerce_mpe_stage_config(mpe_stage)
        
        self.model_input_key = model_input_key

        self.loss_fn.setup_loss(self.label_config)

        if self.mpe_model is not None:
            self._initialize_mpe_stage()

        self._initialize_backbone_stage()

        feature_dim = getattr(self.backbone, "output_dim")
        self._initialize_heads(feature_dim)

    def model_step(self, batch, shift=True):
        batch, chord_input = self._prepare_chord_inputs(batch, shift=shift)

        embedding = self.backbone(chord_input)  # (N, T, C)
        outputs = self._forward_heads(embedding)
        targets = self._gather_targets(batch)

        loss_dict = self.loss_fn(outputs, targets)

        return loss_dict, outputs, targets

    def training_step(self, batch, batch_idx):
        loss_dict, outputs, targets = self.model_step(batch, shift=True)
        loss = loss_dict.pop("total")

        self.log("loss/train", loss)
        for k, v in loss_dict.items():
            self.log(f"loss/train/{k}", v)
        
        if self.train_chord_acc is not None:
            self.train_chord_acc.update(outputs["chord"], targets["chord"])
        if self.train_structured_eval is not None:
            self.train_structured_eval.update(outputs, targets)
        return loss

    def validation_step(self, batch, batch_idx):
        loss_dict, outputs, targets = self.model_step(batch, shift=False)
        loss = loss_dict.pop("total")

        self.log("loss/val", loss)
        for k, v in loss_dict.items():
            self.log(f"loss/val/{k}", v)

        if self.val_chord_acc is not None:
            self.val_chord_acc.update(outputs["chord"], targets["chord"])

        if self.val_structured_eval is not None:
            self.val_structured_eval.update(outputs, targets)

        return loss

    def predict_step(self, inputs):
        if isinstance(inputs, torch.Tensor):
            if inputs.ndim == 3:  # add batch size = 1
                inputs = inputs.unsqueeze(0)
            batch = {"x": inputs}
        elif isinstance(inputs, Mapping):
            batch = dict(inputs)
        else:
            raise TypeError("predict_step expects a tensor or mapping of tensors.")
        batch, chord_input = self._prepare_chord_inputs(batch, shift=False)
        embedding = self.backbone(chord_input)  # N, T, C

        return self._forward_heads(embedding)

    def configure_optimizers(self):
        self.optimizer_cfg = self.hparams.optim_cfg["optimizer"]
        self.scheduler_cfg = self.hparams.optim_cfg["scheduler"]
        trainable = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(trainable, **self.optimizer_cfg)

    def train(self, mode: bool = True):
        """Override to keep frozen sub-modules permanently in eval mode.

        PyTorch/Lightning call ``self.train()`` at the start of every training
        epoch, which would flip BatchNorm/Dropout in frozen stages back to
        train mode.  We immediately counter that for any stage marked frozen.
        """
        super().train(mode)
        if self.mpe_model is not None and self.mpe_stage_cfg.freeze:
            self.mpe_model.eval()
        if self.backbone_stage_cfg.freeze:
            self.backbone.eval()
            self.heads.eval()
        return self

    def on_train_epoch_start(self):
        if self.train_chord_acc is not None:
            self.train_chord_acc.reset()
        if self.train_structured_eval is not None:
            self.train_structured_eval.reset()

    def on_train_epoch_end(self):
        if self.train_chord_acc is not None:
            self.log("train/chord_acc", self.train_chord_acc.compute())
        if self.train_structured_eval is not None:
            for k, v in self.train_structured_eval.compute().items():
                self.log(f"train/{k}", float(v))

    def on_validation_epoch_start(self):
        if self.val_chord_acc is not None:
            self.val_chord_acc.reset()
        if self.val_structured_eval is not None:
            self.val_structured_eval.reset()

    def on_validation_epoch_end(self):
        if self.val_chord_acc is not None:
            self.log("val/chord_acc", self.val_chord_acc.compute())
        if self.val_structured_eval is not None:
            for k, v in self.val_structured_eval.compute().items():
                self.log(f"val/{k}", float(v))

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss_dict, outputs, targets = self.model_step(batch, shift=False)
        loss = loss_dict.pop("total")

        if self.test_chord_acc is not None:
            self.test_chord_acc.update(outputs["chord"], targets["chord"])

        if self.test_structured_eval is not None:
            self.test_structured_eval.update(outputs, targets)

        if self.test_chord_eval is not None:
            # Harte head: compare predicted chord index → string vs ground-truth string
            assert len(targets["hchord_str"]) == 1, "test_chord_eval must be done per sample (batch size = 1)"
            assert len(outputs["chord"]) == 1, "test_chord_eval must be done per sample (batch size = 1)"

            target_strings: List[str] = targets["hchord_str"][0]  # (T,) for batch-size-1 test
            pred_indices = outputs["chord"].argmax(dim=1)[0].cpu().int().numpy()  # (T,)
            pred_strings: List[str] = [
                convert_harte_index_to_hchord(
                    i,
                    self.label_config.harte_vocab,
                    include_no_chord=self.label_config.include_no_chord,
                ).export_string()
                for i in pred_indices
            ]
            self.test_chord_eval.update(pred_strings, target_strings)

        if self.test_rbp_chord_eval is not None:
            assert len(targets["hchord_str"]) == 1, "test_rbp_chord_eval must be done per sample (batch size = 1)"
            assert len(outputs["root"]) == 1, "test_rbp_chord_eval must be done per sample (batch size = 1)"

            # RBP head: compare decoded RBP output → string vs ground-truth string
            target_strings = targets["hchord_str"][0]  # (T,) for batch-size-1 test
            pred_strings = decode_rbp_sequence(
                root_logits=outputs["root"][0].cpu().float().numpy(),
                pitch_logits=torch.sigmoid(outputs["pitch"][0]).cpu().float().numpy(),
                bass_logits=outputs["bass"][0].cpu().float().numpy(),
                include_no_chord=self.label_config.include_no_chord,
            )
            self.test_rbp_chord_eval.update(pred_strings, target_strings)

        return loss

    def on_test_epoch_start(self):
        if self.test_chord_acc is not None:
            self.test_chord_acc.reset()
        if self.test_structured_eval is not None:
            self.test_structured_eval.reset()
        if self.test_chord_eval is not None:
            self.test_chord_eval.reset()
        if self.test_rbp_chord_eval is not None:
            self.test_rbp_chord_eval.reset()

    def on_test_epoch_end(self):
        if self.test_chord_acc is not None:
            self.log("test/chord_acc", self.test_chord_acc.compute())
        if self.test_structured_eval is not None:
            for k, v in self.test_structured_eval.compute().items():
                self.log(f"test/{k}", float(v))
        if self.test_chord_eval is not None:
            for k, v in self.test_chord_eval.compute().items():
                self.log(f"test/{k}", float(v))
        if self.test_rbp_chord_eval is not None:
            for k, v in self.test_rbp_chord_eval.compute().items():
                self.log(f"test/rbp_{k}", float(v))

    def setup(self, stage: str):
        if self.hparams.compile and stage == "fit":
            self.feature = torch.compile(self.feature)
            self.backbone = torch.compile(self.backbone)
            self.loss_fn = torch.compile(self.loss_fn)

    def _initialize_heads(self, feature_dim: int) -> None:
        # Metrics – only created when the relevant heads are active
        self.train_chord_acc: Optional[ChordAccuracy] = None
        self.val_chord_acc: Optional[ChordAccuracy] = None
        self.test_chord_acc: Optional[ChordAccuracy] = None

        self.train_structured_eval: Optional[RBPEval] = None
        self.val_structured_eval: Optional[RBPEval] = None
        self.test_structured_eval: Optional[RBPEval] = None

        self.test_chord_eval: Optional[ChordEval] = None
        self.test_rbp_chord_eval: Optional[ChordEval] = None

        if self.pitch_informed_rbp:
            assert self.label_config.require_rbp
            self.heads = PitchInformedRBPHead(feature_dim, include_no_chord=self.label_config.include_no_chord)
        elif self.rbp_informed_chord:
            assert self.label_config.require_harte and self.label_config.require_rbp
            num_classes = num_harte_classes(self.label_config.harte_vocab, self.label_config.include_no_chord)
            self.heads = RBPInformedChordHead(
                feature_dim,
                num_classes=num_classes,
                include_no_chord=self.label_config.include_no_chord,
            )
        else:
            raise ValueError("Model must enable either pitch_informed_rbp or rbp_informed_chord.")

        if self.label_config.require_harte:
            self.train_chord_acc = ChordAccuracy()
            self.val_chord_acc = ChordAccuracy()
            self.test_chord_acc = ChordAccuracy()
            self.test_chord_eval = ChordEval(include_no_chord=self.label_config.include_no_chord)

        if self.label_config.require_rbp:
            self.train_structured_eval  = RBPEval()
            self.val_structured_eval  = RBPEval()
            self.test_structured_eval = RBPEval()
            self.test_rbp_chord_eval = ChordEval(include_no_chord=self.label_config.include_no_chord)

        # Apply any backbone-stage heads weights that were stashed before heads existed
        if getattr(self, "_pending_heads_sd", None):
            # Heads trained with a different class count (e.g. without N) cannot be reused
            head_shapes = {k: v.shape for k, v in self.heads.state_dict().items()}
            skipped = [k for k, v in self._pending_heads_sd.items() if k in head_shapes and v.shape != head_shapes[k]]
            if skipped:
                print(f"Skipping heads weights with mismatched shapes (re-initialized): {skipped}")
            pending = {k: v for k, v in self._pending_heads_sd.items() if k not in skipped}
            missing, unexpected = self.heads.load_state_dict(pending, strict=False)
            if missing or unexpected:
                print(
                    f"Heads checkpoint load mismatch. "
                    f"Missing: {missing}, unexpected: {unexpected}."
                )
            self._pending_heads_sd = None

        # Freeze heads if the backbone stage is frozen
        if self.backbone_stage_cfg.freeze:
            self.heads.requires_grad_(False)

    def _prepare_chord_inputs(self, batch: Dict[str, torch.Tensor], shift: bool) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        batch = self.feature(batch, shift=shift, input_key=self.model_input_key, label_config=self.label_config)
        if self.model_input_key not in batch:
            raise KeyError(f"Batch is missing '{self.model_input_key}' required for chord input.")

        chord_input = batch[self.model_input_key]
    
        # Apply MPE stage if configured
        if self.mpe_model is not None:
            chord_input = self._run_mpe_stage(chord_input)
            batch["mpe_output"] = chord_input

        return batch, chord_input

    def _initialize_mpe_stage(self) -> None:
        if self.mpe_stage_cfg.checkpoint_path:
            state_dict = torch.load(self.mpe_stage_cfg.checkpoint_path, map_location="cpu")
            missing, unexpected = self.mpe_model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                msg = f"MPE checkpoint load mismatch. Missing keys: {missing}, unexpected keys: {unexpected}."
                if self._trainer is not None:
                    self.print(msg)
                else:
                    print(msg)
        if self.mpe_stage_cfg.freeze:
            self.mpe_model.requires_grad_(False)

    def _initialize_backbone_stage(self) -> None:
        """Load pretrained backbone weights and optionally freeze backbone + heads.

        The checkpoint may be:
        - A Lightning ``.ckpt`` file (contains ``"state_dict"`` key with
          ``"backbone.*"`` and ``"heads.*"`` prefixes).
        - A raw state-dict ``.pt``/``.pth`` file with matching key names.
        """
        self._pending_heads_sd: Optional[Dict] = None  # applied after heads are built

        if self.backbone_stage_cfg.checkpoint_path:
            state_dict = torch.load(self.backbone_stage_cfg.checkpoint_path, map_location="cpu")
            state_dict = state_dict.get("state_dict", state_dict)  # unwrap Lightning checkpoint if needed

            backbone_sd = {
                k.removeprefix("backbone."): v
                for k, v in state_dict.items()
                if k.startswith("backbone.")
            }
            heads_sd = {
                k.removeprefix("heads."): v
                for k, v in state_dict.items()
                if k.startswith("heads.")
            }

            missing, unexpected = self.backbone.load_state_dict(backbone_sd, strict=False)
            if missing or unexpected:
                print(
                    f"Backbone checkpoint load mismatch. "
                    f"Missing: {missing}, unexpected: {unexpected}."
                )

            # Heads are not built yet; stash for _initialize_heads to pick up
            self._pending_heads_sd = heads_sd

        if self.backbone_stage_cfg.freeze:
            self.backbone.requires_grad_(False)
            # heads.requires_grad_(False) is deferred to _initialize_heads

    def _run_mpe_stage(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.mpe_stage_cfg.freeze:
            with torch.no_grad():
                mpe_output = self.mpe_model(inputs).detach()
        else:
            mpe_output = self.mpe_model(inputs)

        return torch.sigmoid(mpe_output)  # normalize MPE output to [0, 1]

    def _forward_heads(self, embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.heads(embedding)

    def _gather_targets(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        targets: Dict[str, torch.Tensor] = {}
        if "y_chord" in batch:
            targets["chord"] = batch["y_chord"].long()
        if "y_root" in batch:
            targets["root"] = batch["y_root"].long()
        if "y_bass" in batch:
            targets["bass"] = batch["y_bass"].long()
        if "y_pitch" in batch:
            targets["pitch"] = batch["y_pitch"].float()
        if "y_hchord_str" in batch:
            targets["hchord_str"] = batch["y_hchord_str"]
        return targets

    # cosine annealing learning rate decay with warm-up
    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_closure
    ):
        base_lr = self.optimizer_cfg["lr"]
        warmup_epochs = self.scheduler_cfg.get("warmup_epochs", 5)
        T_max = self.scheduler_cfg.get("T_max", 55)
        eta_min = self.scheduler_cfg.get("eta_min", 0.0)

        # Compute learning rate
        if self.trainer.current_epoch < warmup_epochs:
            # Linear warmup from 0 → base_lr
            lr = base_lr * (self.trainer.current_epoch + 1) / warmup_epochs
        else:
            # Cosine annealing decay
            lr = eta_min + (base_lr - eta_min) * 0.5 * (1 + math.cos(math.pi * (self.trainer.current_epoch - warmup_epochs) / (T_max - warmup_epochs)))

        # Set learning rate
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Take optimization step
        optimizer.step(closure=optimizer_closure)
