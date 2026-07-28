#!/usr/bin/env python3
"""Minimal fine-tuning entry point for the compact repo."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import hydra
import lightning as L
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.utils import (  # noqa: E402
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if cfg.get("seed") is not None:
        L.seed_everything(int(cfg.seed), workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train", True):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics
    metric_dict = {**train_metrics, **test_metrics}
    return metric_dict, object_dict


def load_config(config_path: Path) -> DictConfig:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return OmegaConf.load(config_path)


def main() -> Optional[float]:
    parser = argparse.ArgumentParser(description="Run the minimal fine-tuning entry point.")
    parser.add_argument(
        "--config",
        type=Path,
        default=_REPO_ROOT / "configs" / "finetune_classical.yaml",
        help="Path to the training config file.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    extras(cfg)

    metric_dict, _ = train(cfg)
    return get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()