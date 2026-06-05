"""
Unified training script.

Handles three modes via the same core loop:
  1. Single training run       — call train_on_the_fly() directly
  2. Optuna hyperparameter search — call run_hparam_search()
  3. CLI                       — python train.py [--mode train|search]

All runs write to:
  <log_dir>/
  └── <run_id>_<run_name>/          (plain run)
  └── <study_name>/
      ├── study.db
      ├── best_trial.json
      └── trial_NNNN/               (one per Optuna trial)
          ├── config.json
          ├── metrics.jsonl
          ├── final_eval.json
          └── checkpoints/
              └── best_model.pt

  CLI usage :
  # plain training run
python train.py --mode train --datasets mcfw-gaze GGTG

# hyperparameter search
python train.py --mode search --n_trials 50 --n_epochs_per_trial 10 --study_name my_study
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import optuna
import torch
from accelerate import Accelerator
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from tqdm import tqdm

from kaamba.net.models.kaamba import build_gaze_predictor
from kaamba.utils.dataloader_config_builder import (
    DataloaderConfigBuilder,
    InterleavedLoader,
)
from kaamba.utils.loss_functions import gmm_nll
from kaamba.utils.memory_monitor import MemoryMonitor, memory_tracker

import os

os.environ["NCCL_TIMEOUT"] = "60"
# ---------------------------------------------------------------------------
# Encoder presets
# ---------------------------------------------------------------------------

ENCODER_CONFIGS = {
    "vit_base": {"encoder_type": "vit", "model_name": "google/vit-base-patch16-224"},
    "vit_large": {"encoder_type": "vit", "model_name": "google/vit-large-patch16-224"},
    "resnet": {"encoder_type": "resnet"},
    "siglip": {
        "encoder_type": "siglip",
        "model_name": "google/siglip-base-patch16-224",
    },
}

# Maps the short encoder_type names used in Optuna suggestions to their default
# HuggingFace model names.  ResNet is absent — it loads ImageNet weights via
# torchvision and does not accept a model_name argument.
_ENCODER_MODEL_NAMES = {
    "vit": "google/vit-base-patch16-224",
    "siglip": "google/siglip-base-patch16-224",
}
# ---------------------------------------------------------------------------
# ExperimentConfig — single source of truth for every run
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    run_name: str
    run_id: str = field(
        default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    # Model
    model_config: Dict[str, Any] = field(default_factory=dict)

    # Data
    dataset_names: List[str] = field(default_factory=list)
    split_strategy: str = "participant"
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    context_len: int = 3200
    stride: int = 1
    sampling_step: int = 1
    max_image_size: int = 224
    exclude_participants: List = field(default_factory=list)
    exclude_stimuli: List = field(default_factory=list)
    exclude_trials: List = field(default_factory=list)

    # Training
    batch_size: int = 128
    num_workers: int = 1
    num_epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    patience: int = 5

    # Optuna (None when not inside a study)
    trial_number: Optional[int] = None
    study_name: Optional[str] = None

    log_dir: str = "logs/runs"
    resume_from: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path):
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "ExperimentConfig":
        return cls(**json.loads(path.read_text()))


# ---------------------------------------------------------------------------
# ExperimentTracker
# ---------------------------------------------------------------------------


class ExperimentTracker:
    """
    Writes config, streams metrics to .jsonl, manages checkpoints.
    Aware of both Accelerate (only main process writes) and
    Optuna (reports to pruner, raises TrialPruned when needed).
    """

    def __init__(
        self,
        config: ExperimentConfig,
        accelerator: Accelerator,
        trial: Optional[optuna.Trial] = None,
        use_wandb: bool = False,
    ):
        self.config = config
        self.accelerator = accelerator
        self.trial = trial
        self.use_wandb = use_wandb
        self.best_val_loss = float("inf")
        self.start_time = time.time()

        # Directory layout
        base = Path(config.log_dir)
        if trial is not None:
            self.run_dir = base / f"trial_{trial.number:04d}"
        else:
            self.run_dir = base / f"{config.run_id}_{config.run_name}"

        self.ckpt_dir = self.run_dir / "checkpoints"
        self.metrics_path = self.run_dir / "metrics.jsonl"

        if self.accelerator.is_main_process:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.ckpt_dir.mkdir(exist_ok=True)
            config.save(self.run_dir / "config.json")
            if use_wandb:
                self._init_wandb()

        self.accelerator.print(f"[tracker] {self.run_dir}")

    # ── Metrics ──────────────────────────────────────────────────────────

    def log_epoch(self, epoch: int, val_loss: float, **metrics):
        """
        Log one epoch. Also handles Optuna pruning — may raise TrialPruned.
        All GPU processes participate in the pruning broadcast to avoid hangs.
        """
        row = {
            "epoch": epoch,
            "val_loss": val_loss,
            "timestamp": time.time(),
            **metrics,
        }

        if self.accelerator.is_main_process:
            with self.metrics_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            if self.use_wandb:
                import wandb

                wandb.log(row, step=epoch)

        # Optuna: report + pruning decision broadcast
        if self.trial is not None:
            should_prune = torch.zeros(1, device=self.accelerator.device)

            if self.accelerator.is_main_process:
                self.trial.report(val_loss, epoch)
                if self.trial.should_prune():
                    should_prune[0] = 1.0

            if (
                self.accelerator.num_processes > 1
                and torch.distributed.is_initialized()
            ):
                torch.distributed.broadcast(should_prune, src=0)

            if should_prune.item() == 1.0:
                raise optuna.TrialPruned(
                    f"Pruned at epoch {epoch} (val_loss={val_loss:.4f})"
                )

    def log_final_eval(self, metrics: Dict[str, Any]):
        if not self.accelerator.is_main_process:
            return
        out = {
            "timestamp": time.time(),
            "total_time_s": time.time() - self.start_time,
            **metrics,
        }
        (self.run_dir / "final_eval.json").write_text(json.dumps(out, indent=2))
        if self.use_wandb:
            import wandb

            wandb.summary.update(metrics)
        self.accelerator.print(f"[tracker] final eval: {out}")

    def save_loader_configs(self, loader_configs: dict):
        """
        Persist loader configs to disk. Call once after dataloaders are built.

        Args:
            loader_configs: dict returned by DataloaderConfigBuilder.create_loaders()
                            Either {"train": DataloaderConfig, "val": ..., "test": ...}
                            or     {"dataset_a": {"train": ..., ...}, "dataset_b": ...}
                            for multi-dataset runs.
        """
        if not self.accelerator.is_main_process:
            return

        # Normalise both single and multi-dataset shapes to plain dicts
        def _serialise(cfg) -> dict:
            if hasattr(cfg, "to_dict"):  # DataloaderConfig dataclass
                return cfg.to_dict()
            elif isinstance(cfg, dict):
                return {k: _serialise(v) for k, v in cfg.items()}
            return cfg  # already a plain value

        (self.run_dir / "loader_configs.json").write_text(
            json.dumps(_serialise(loader_configs), indent=2)
        )
        self.accelerator.print("[tracker] loader configs saved")

    # ── Checkpoints ──────────────────────────────────────────────────────

    def save_checkpoint(
        self,
        model,
        optimizer,
        scheduler,
        epoch: int,
        val_loss: float,
        save_every_epoch: bool = False,
    ):
        if not self.accelerator.is_main_process:
            return
        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
            "config": self.config.to_dict(),
        }
        if save_every_epoch:
            torch.save(payload, self.ckpt_dir / f"checkpoint_epoch_{epoch:04d}.pt")
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(payload, self.ckpt_dir / "best_model.pt")
            self.accelerator.print(f"[tracker] ✓ best model  val_loss={val_loss:.4f}")

    def finish(self):
        elapsed = time.time() - self.start_time
        self.accelerator.print(
            f"[tracker] done in {elapsed / 60:.1f} min → {self.run_dir}"
        )
        if self.use_wandb and self.accelerator.is_main_process:
            import wandb

            wandb.finish()

    def load_metrics(self):
        import pandas as pd

        rows = [json.loads(line) for line in self.metrics_path.read_text().splitlines()]
        return pd.DataFrame(rows)

    def _init_wandb(self):
        import wandb

        wandb.init(
            project="gaze-mamba",
            group=self.config.study_name or self.config.run_name,
            name=f"trial_{self.trial.number}" if self.trial else self.config.run_name,
            id=self.config.run_id,
            config=self.config.to_dict(),
            resume="allow",
            dir=str(self.run_dir),
        )


# ---------------------------------------------------------------------------
# TrainingMonitor
# ---------------------------------------------------------------------------


class TrainingMonitor:
    def __init__(self, patience=5, min_delta=1e-4, max_grad_norm=10.0):
        self.patience = patience
        self.min_delta = min_delta
        self.max_grad_norm = max_grad_norm
        self.best_loss = float("inf")
        self.patience_counter = 0

    def check_epoch_loss(self, loss) -> Tuple[bool, str]:
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.patience_counter = 0
            return False, f"✓ Loss improved to {loss:.6f}"
        self.patience_counter += 1
        if self.patience_counter >= self.patience:
            return True, f"✗ No improvement for {self.patience} epochs"
        return (
            False,
            f"  Patience {self.patience_counter}/{self.patience} (best {self.best_loss:.6f})",
        )

    def check_loss_validity(self, loss) -> Tuple[bool, str]:
        if loss != loss or not (-1e10 < loss < 1e10):
            return True, f"✗ Invalid loss: {loss}"
        return False, ""

    def check_gradient_norm(self, model) -> Tuple[bool, str]:
        norm = (
            sum(
                p.grad.data.norm(2).item() ** 2
                for p in model.parameters()
                if p.grad is not None
            )
            ** 0.5
        )
        if norm > self.max_grad_norm:
            return True, f"✗ Gradient norm {norm:.4f} > {self.max_grad_norm}"
        return False, ""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate(model, val_loader, accelerator) -> float:
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for batch in tqdm(
            val_loader, desc="Val", disable=not accelerator.is_main_process
        ):
            images = batch["image"].to(accelerator.device)
            inputs = batch["input_seq"].to(accelerator.device)
            targets = batch["target_seq"].to(accelerator.device).permute(0, 2, 1)
            pi, mu, log_sx, log_sy, rho_raw = model(images, inputs)
            total_loss += gmm_nll(pi, mu, log_sx, log_sy, rho_raw, targets).item()
            n += 1
    avg = total_loss / n if n > 0 else 0.0
    avg = (
        accelerator.gather(torch.tensor([avg], device=accelerator.device)).mean().item()
    )
    model.train()
    return avg


# ---------------------------------------------------------------------------
# Core training loop  (shared by plain runs AND Optuna trials)
# ---------------------------------------------------------------------------


def train_on_the_fly(
    # Model
    model_config: dict,
    # Data
    dataset_name: str | List[str],
    root: str,
    split_strategy: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    # Loader
    batch_size: int,
    num_workers: int,
    context_len: int,
    stride: int,
    sampling_step: int,
    max_image_size: int,
    # Training
    num_epochs: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    patience: int,
    exclude_participants: Optional[List] = None,
    exclude_stimuli: Optional[List] = None,
    exclude_trials: Optional[List] = None,
    # Tracking
    log_dir: str = "logs/runs",
    run_name: Optional[str] = None,
    resume_from: Optional[str] = None,
    use_wandb: bool = False,
    save_every_epoch: bool = False,
    # Optuna (set by run_trial — do not pass manually)
    trial: Optional[optuna.Trial] = None,
    # Accelerate
    accelerator: Optional[Accelerator] = None,
) -> Tuple[torch.nn.Module, float]:
    """
    Core training loop. Returns (model, best_val_loss).
    Raises optuna.TrialPruned if inside an Optuna study and pruned.
    """
    _owns_accelerator = accelerator is None
    if _owns_accelerator:
        accelerator = Accelerator()

    try:
        datasets = [dataset_name] if isinstance(dataset_name, str) else dataset_name

        # ── Config object ─────────────────────────────────────────────────
        config = ExperimentConfig(
            run_name=run_name or "_".join(datasets),
            model_config=model_config,
            dataset_names=datasets,
            split_strategy=split_strategy,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            context_len=context_len,
            stride=stride,
            sampling_step=sampling_step,
            max_image_size=max_image_size,
            exclude_participants=exclude_participants or [],
            exclude_stimuli=exclude_stimuli or [],
            exclude_trials=exclude_trials or [],
            batch_size=batch_size,
            num_workers=num_workers,
            num_epochs=num_epochs,
            lr=lr,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            patience=patience,
            log_dir=log_dir,
            resume_from=resume_from,
            trial_number=trial.number if trial else None,
            study_name=trial.study.study_name if trial else None,
        )

        tracker = ExperimentTracker(
            config, accelerator, trial=trial, use_wandb=use_wandb
        )
        monitor = MemoryMonitor(log_dir=log_dir)

        accelerator.print("=" * 70)
        accelerator.print(f"RUN  {config.run_name}  [{config.run_id}]")
        accelerator.print(f"datasets={datasets}  strategy={split_strategy}")
        accelerator.print(f"model={model_config}")
        accelerator.print("=" * 70)

        # ── Dataloaders ───────────────────────────────────────────────────
        builder = DataloaderConfigBuilder(
            datasets=datasets,
            root=root,
            context_len=context_len,
            stride=stride,
            sampling_step=sampling_step,
            max_image_size=max_image_size,
        )
        with memory_tracker("DataLoader creation"):
            train_loader, val_loader, test_loader, loader_configs = (
                builder.create_loaders(
                    split_strategy=split_strategy,
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                    test_ratio=test_ratio,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    exclude_participants=exclude_participants,
                    exclude_stimuli=exclude_stimuli,
                    exclude_trials=exclude_trials,
                )
            )
        tracker.save_loader_configs(loader_configs)
        # ── Model ─────────────────────────────────────────────────────────
        model = build_gaze_predictor(**model_config)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs
        )
        training_monitor = TrainingMonitor(patience=patience)

        model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

        def _prepare_loader(loader, accelerator):
            if isinstance(loader, InterleavedLoader):
                loader.loaders = [accelerator.prepare(load) for load in loader.loaders]
                return loader
            return accelerator.prepare(loader)

        train_loader = _prepare_loader(train_loader, accelerator)
        if not val_loader == 0.0:
            val_loader = _prepare_loader(val_loader, accelerator)
        if not test_ratio == 0.0:
            test_loader = _prepare_loader(test_loader, accelerator)

        start_epoch = 0
        if resume_from:
            start_epoch = _load_checkpoint(
                resume_from, model, optimizer, scheduler, accelerator
            )

        model.train()
        accelerator.print("\nTRAINING\n" + "=" * 70)

        # ── Epoch loop ────────────────────────────────────────────────────
        final_epoch = start_epoch
        for epoch in range(start_epoch, num_epochs):
            final_epoch = epoch
            epoch_start = time.time()
            total_loss, nb = 0.0, 0

            for batch_idx, batch in enumerate(
                tqdm(
                    train_loader,
                    desc=f"Epoch {epoch + 1}",
                    disable=not accelerator.is_main_process,
                )
            ):
                images = batch["image"].to(accelerator.device)
                inputs = batch["input_seq"].to(accelerator.device)
                targets = batch["target_seq"].to(accelerator.device).permute(0, 2, 1)
                try:
                    optimizer.zero_grad()
                    pi, mu, log_sx, log_sy, rho_raw = model(images, inputs)
                    loss = gmm_nll(pi, mu, log_sx, log_sy, rho_raw, targets)

                    if batch_idx == 0 and accelerator.is_main_process:
                        accelerator.print(
                            f"  [dbg] mu {mu.min():.3f}/{mu.mean():.3f}/{mu.max():.3f}"
                            f"  log_sx {log_sx.mean():.3f}  rho {rho_raw.max():.3f}"
                            f"  target {targets.min():.3f}/{targets.max():.3f}"
                            f"  loss {loss.item():.4f}"
                        )

                    accelerator.backward(loss)
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                    optimizer.step()
                except torch.cuda.OutOfMemoryError:
                    # Synchronize both ranks before raising — prevents NCCL desync
                    if torch.distributed.is_initialized():
                        torch.distributed.barrier()
                    raise

                total_loss += loss.item()
                nb += 1

                if (batch_idx + 1) % 50 == 0 and accelerator.is_main_process:
                    monitor.log_memory(batch_idx + 1, phase="training")

            avg_train = total_loss / nb if nb else 0.0
            avg_val = validate(model, val_loader, accelerator)
            epoch_t = time.time() - epoch_start
            scheduler.step()

            accelerator.print(
                f"Epoch {epoch + 1:3d} | train {avg_train:.4f} | val {avg_val:.4f}"
                f" | lr {scheduler.get_last_lr()[0]:.2e} | {epoch_t:.1f}s"
            )

            # ── Log metrics (may raise TrialPruned) ───────────────────────
            try:
                tracker.log_epoch(
                    epoch=epoch + 1,
                    val_loss=avg_val,
                    train_loss=avg_train,
                    lr=scheduler.get_last_lr()[0],
                    epoch_time_s=epoch_t,
                )
            except optuna.TrialPruned:
                accelerator.print(f"[optuna] pruned at epoch {epoch + 1}")
                tracker.finish()
                raise

            # ── Checkpoint ────────────────────────────────────────────────
            accelerator.wait_for_everyone()
            tracker.save_checkpoint(
                accelerator.unwrap_model(model),
                optimizer,
                scheduler,
                epoch=epoch + 1,
                val_loss=avg_val,
                save_every_epoch=save_every_epoch,
            )

            # ── Early stopping ────────────────────────────────────────────
            should_stop = False
            for fn, args in [
                (training_monitor.check_loss_validity, (avg_val,)),
                (training_monitor.check_gradient_norm, (model,)),
                (training_monitor.check_epoch_loss, (avg_val,)),
            ]:
                flag, msg = fn(*args)
                if msg:
                    accelerator.print(f"  {msg}")
                if flag:
                    should_stop = True
                    break

            if should_stop:
                accelerator.print("⚠️  Early stop")
                break

        # ── Final test eval ───────────────────────────────────────────────
        if not test_loader:
            test_loader = val_loader
        test_loss = validate(model, test_loader, accelerator)
        tracker.log_final_eval(
            {
                "test_nll": test_loss,
                "best_val_loss": tracker.best_val_loss,
                "epochs_trained": final_epoch + 1,
            }
        )

        if accelerator.is_main_process:
            monitor.save_log()
            accelerator.print(
                f"Peak RAM {monitor.peak_ram:.2f} GB | "
                f"Peak VRAM {monitor.peak_vram:.2f} GB"
            )

        tracker.finish()
        return accelerator.unwrap_model(model), tracker.best_val_loss

    finally:
        if _owns_accelerator:
            accelerator.end_training()


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def _load_checkpoint(path, model, optimizer, scheduler, accelerator) -> int:
    p = Path(path)
    if not p.exists():
        accelerator.print(f"[ckpt] {p} not found, starting fresh")
        return 0
    ckpt = torch.load(p, map_location=accelerator.device)
    accelerator.unwrap_model(model).load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    accelerator.print(
        f"[ckpt] resumed epoch {ckpt['epoch']} "
        f"val={ckpt.get('val_loss', float('nan')):.4f}"
    )
    return ckpt["epoch"]


# ---------------------------------------------------------------------------
# Optuna study
# ---------------------------------------------------------------------------


def run_hparam_search(
    # Fixed data config (not searched)
    dataset_name: str | List[str],
    root: str,
    split_strategy: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    context_len: int,
    sampling_step: int,
    max_image_size: int,
    num_workers: int,
    stride: int,
    exclude_participants: Optional[List] = None,
    exclude_stimuli: Optional[List] = None,
    exclude_trials: Optional[List] = None,
    # Search budget
    n_trials: int = 50,
    n_epochs_per_trial: int = 10,
    max_batches_per_epoch: Optional[int] = 200,
    # Study persistence
    study_name: str = "gaze_mamba_search",
    log_dir: str = "logs/runs",
    storage: Optional[str] = None,  # e.g. "sqlite:///study.db"
    use_wandb: bool = False,
):
    accelerator = Accelerator()
    study_dir = Path(log_dir) / study_name
    if accelerator.is_main_process:
        study_dir.mkdir(parents=True, exist_ok=True)
    # accelerator.wait_for_everyone()

    if storage is None:
        storage = f"sqlite:///{study_dir}/study.db"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )

    print(f"[optuna] study '{study_name}'  ({len(study.trials)} trials so far)")
    print(f"[optuna] storage: {storage}")

    def objective(trial: optuna.Trial) -> float:
        # ── Sample hyperparameters ────────────────────────────────────────
        model_config = {
            "encoder_type": trial.suggest_categorical(
                "encoder_type", ["vit", "resnet", "siglip"]
            ),
            "d_model": trial.suggest_categorical("d_model", [128, 256, 512]),
            "n_layers": trial.suggest_int("n_layers", 4, 12),
            "n_mix": trial.suggest_int("n_mix", 3, 8),
            "image_embed_dim": trial.suggest_categorical("image_embed_dim", [256, 512]),
            "conditioning_mode": trial.suggest_categorical(
                "conditioning_mode", ["initial_state", "every_step"]
            ),
            "freeze_encoder": True,
        }

        # ResNet has no HuggingFace model_name — only add it for vit / siglip
        model_name = _ENCODER_MODEL_NAMES.get(model_config["encoder_type"])
        if model_name is not None:
            model_config["model_name"] = model_name

        lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)
        grad_clip = trial.suggest_float("grad_clip", 0.1, 2.0)
        batch_size = trial.suggest_categorical("batch_size", [256, 512])

        try:
            _, best_val = train_on_the_fly(
                model_config=model_config,
                dataset_name=dataset_name,
                root=root,
                split_strategy=split_strategy,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                exclude_participants=exclude_participants,
                exclude_stimuli=exclude_stimuli,
                exclude_trials=exclude_trials,
                context_len=context_len,
                stride=stride,
                sampling_step=sampling_step,
                max_image_size=max_image_size,
                num_workers=num_workers,
                batch_size=batch_size,
                num_epochs=n_epochs_per_trial,
                lr=lr,
                weight_decay=weight_decay,
                grad_clip=grad_clip,
                patience=n_epochs_per_trial,  # no early stop inside trial
                log_dir=str(study_dir),
                run_name=f"trial_{trial.number:04d}",
                save_every_epoch=False,
                use_wandb=use_wandb,
                trial=trial,
                accelerator=accelerator,
            )
            return best_val
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            raise optuna.TrialPruned("OOM")

        finally:
            gc.collect()
            torch.cuda.empty_cache()

    study.optimize(
        objective,
        n_trials=n_trials,
        catch=(RuntimeError, ValueError),
    )

    # ── Save best trial ───────────────────────────────────────────────────
    best = study.best_trial
    print(f"\n[optuna] best trial #{best.number}  val_loss={best.value:.4f}")
    (study_dir / "best_trial.json").write_text(
        json.dumps(
            {"number": best.number, "value": best.value, "params": best.params},
            indent=2,
        )
    )

    try:
        importances = optuna.importance.get_param_importances(study)
        print("\nParameter importances:")
        for k, v in sorted(importances.items(), key=lambda x: -x[1]):
            print(f"  {k:<28} {'█' * int(v * 40)} {v:.3f}")
    except Exception:
        pass

    return study


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_config(path: str) -> Tuple[dict, dict]:
    """
    Load a JSON config file and return ``(cli_overrides, model_config)``.

    Auto-detects one of three formats produced by this codebase:

    ``config.json``     — ExperimentConfig saved by ExperimentConfig.save().
                          Has ``model_config`` and ``dataset_names`` keys.
    ``best_trial.json`` — Optuna best-trial summary from run_hparam_search().
                          Has ``number``, ``value``, ``params`` keys.
    plain JSON          — User-written file whose keys already match CLI arg
                          names.  An optional ``model_config`` key is extracted
                          and returned separately.

    The first return value uses CLI argument names (e.g. ``datasets`` not
    ``dataset_names``) so it can be passed directly to
    ``ArgumentParser.set_defaults(**cli_overrides)``.
    CLI flags always override file values — the file only sets defaults.
    """
    data: dict = json.loads(Path(path).read_text())

    # ── Optuna best_trial.json ────────────────────────────────────────────
    if {"number", "value", "params"} <= data.keys():
        params = dict(data["params"])
        _MODEL_KEYS = {
            "encoder_type",
            "d_model",
            "n_layers",
            "n_mix",
            "image_embed_dim",
            "conditioning_mode",
            "freeze_encoder",
        }
        model_cfg = {k: v for k, v in params.items() if k in _MODEL_KEYS}
        cli = {k: v for k, v in params.items() if k not in _MODEL_KEYS}
        return cli, model_cfg

    # ── ExperimentConfig config.json ──────────────────────────────────────
    if "model_config" in data or "dataset_names" in data:
        model_cfg = data.pop("model_config", {})
        _SKIP = {
            "run_name",
            "run_id",
            "trial_number",
            "split_strategy",
            "train_ratio",
            "val_ratio",
            "test_ratio",
            "stride",
        }
        _KEY_MAP = {"dataset_names": "datasets"}
        cli = {_KEY_MAP.get(k, k): v for k, v in data.items() if k not in _SKIP}
        return cli, model_cfg

    # ── Plain JSON (keys already match CLI arg names) ─────────────────────
    model_cfg = data.pop("model_config", {})
    return data, model_cfg


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Kaamba gaze predictor — training & hyperparameter search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Config file (--config):
          Pass any of the JSON files produced by this script to reproduce or continue
          a run without retyping every argument.  CLI flags always win over the file.

          config.json       saved per run by ExperimentConfig  (recommended)
          best_trial.json   Optuna best-trial summary
          plain JSON        hand-written file with CLI-style keys

          Examples:
            python train.py --config logs/runs/trial_0019/config.json
            python train.py --config logs/study/best_trial.json --num_epochs 200
            python train.py --config my_config.json --datasets mcfw-gaze GGTG
        """,
    )
    p.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="JSON config file.  Keys override defaults; CLI flags override file.",
    )
    p.add_argument("--mode", choices=["train", "search"], default="train")

    # ── shared data args ──────────────────────────────────────────────────
    p.add_argument("--datasets", nargs="+", default=["mcfw-gaze"])
    p.add_argument("--root", default="/home/janhof/thesis/data/")
    p.add_argument("--log_dir", default="/home/janhof/thesis/logs/runs")
    p.add_argument("--context_len", type=int, default=200)
    p.add_argument("--sampling_step", type=int, default=1)
    p.add_argument("--max_image_size", type=int, default=224)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument(
        "--stride",
        type=int,
        choices=[1, 8],
        default=1,
        help="1 for mcfw gaze 8 for GGTG",
    )
    p.add_argument(
        "--exclude_participants",
        nargs="*",
        default=None,
        help="Participant IDs to exclude.  Overrides the hardcoded list.",
    )
    p.add_argument(
        "--exclude_stimuli",
        nargs="*",
        default=None,
        help="Stimulus IDs to exclude.",
    )
    p.add_argument(
        "--exclude_trials",
        nargs="*",
        default=None,
        help="Trial IDs to exclude.",
    )
    p.add_argument(
        "--split_strategy",
        type=str,
        choices=["participant", "stimulus", "random"],
        default="stimulus",
    )

    # ── train-only ────────────────────────────────────────────────────────
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=0.00014293225713757568)
    p.add_argument("--weight_decay", type=float, default=1.49611258828362e-06)
    p.add_argument("--grad_clip", type=float, default=0.8121195670502164)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--resume_from", default=None)
    p.add_argument("--use_wandb", action="store_true")

    # ── search-only ───────────────────────────────────────────────────────
    p.add_argument("--n_trials", type=int, default=50)
    p.add_argument("--n_epochs_per_trial", type=int, default=8)
    p.add_argument("--max_batches_per_epoch", type=int, default=512)
    p.add_argument(
        "--study_name", default="gaze_mamba_search_only_mfcw_gaze_context_200_siglip"
    )
    p.add_argument("--storage", default=None)

    return p


# Hardcoded fallback exclude lists (used when --exclude_* is not set via CLI
# or config file).
_DEFAULT_EXCLUDE_PARTICIPANTS_TRAIN = ["P01", "P02"]
_DEFAULT_EXCLUDE_STIMULI_TRAIN = []
_DEFAULT_EXCLUDE_TRIALS_TRAIN = ["", "4", "5"]
_DEFAULT_EXCLUDE_PARTICIPANTS_SEARCH = [
    "P01",
    "P02",
    "P03",
    "P04",
    "P05",
    "P06",
    "P07",
    "P08",
    "001",
    "002",
    "015",
    "014",
    "013",
    "012",
]
_DEFAULT_EXCLUDE_STIMULI_SEARCH = ["22", "23"]
_DEFAULT_EXCLUDE_TRIALS_SEARCH = ["", "3", "2", "4", "5"]


def main():
    import os

    print(os.environ["CUDA_VISIBLE_DEVICES"])
    p = _build_parser()

    # ── Two-pass parse: peek at --config → set file defaults → full parse ─
    pre, _ = p.parse_known_args()
    model_config_override: dict = {}
    if pre.config:
        print(f"[config] loading {pre.config}")
        cli_defaults, model_config_override = _load_config(pre.config)
        valid_dests = {a.dest for a in p._actions}
        applied = {k: v for k, v in cli_defaults.items() if k in valid_dests}
        p.set_defaults(**applied)
        print(
            f"[config] applied {len(applied)} defaults from file"
            + (
                f", model_config keys: {list(model_config_override)}"
                if model_config_override
                else ""
            )
        )

    args = p.parse_args()

    if args.mode == "train":
        # Model config: hardcoded base, updated by any config-file override
        model_config = {
            **ENCODER_CONFIGS["siglip"],
            "d_model": 128,
            "n_layers": 6,
            "n_mix": 3,
            "image_embed_dim": 256,
            "conditioning_mode": "initial_state",
            "freeze_encoder": True,
        }
        model_config.update(model_config_override)

        accelerator = Accelerator()
        train_on_the_fly(
            model_config=model_config,
            dataset_name=args.datasets,
            root=args.root,
            split_strategy="stimuli",
            context_len=args.context_len,
            sampling_step=args.sampling_step,
            max_image_size=args.max_image_size,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            patience=args.patience,
            log_dir=args.log_dir,
            resume_from=args.resume_from,
            use_wandb=args.use_wandb,
            accelerator=accelerator,
            exclude_participants=(
                args.exclude_participants
                if args.exclude_participants is not None
                else _DEFAULT_EXCLUDE_PARTICIPANTS_TRAIN
            ),
            exclude_stimuli=(
                args.exclude_stimuli
                if args.exclude_stimuli is not None
                else _DEFAULT_EXCLUDE_STIMULI_TRAIN
            ),
            exclude_trials=(
                args.exclude_trials
                if args.exclude_trials is not None
                else _DEFAULT_EXCLUDE_TRIALS_TRAIN
            ),
        )
        accelerator.end_training()

    elif args.mode == "search":
        run_hparam_search(
            dataset_name=args.datasets,
            root=args.root,
            context_len=args.context_len,
            sampling_step=args.sampling_step,
            max_image_size=args.max_image_size,
            num_workers=args.num_workers,
            n_trials=args.n_trials,
            n_epochs_per_trial=args.n_epochs_per_trial,
            max_batches_per_epoch=args.max_batches_per_epoch,
            study_name=args.study_name,
            log_dir=args.log_dir,
            storage=args.storage,
            use_wandb=args.use_wandb,
            exclude_participants=(
                args.exclude_participants
                if args.exclude_participants is not None
                else _DEFAULT_EXCLUDE_PARTICIPANTS_SEARCH
            ),
            exclude_stimuli=(
                args.exclude_stimuli
                if args.exclude_stimuli is not None
                else _DEFAULT_EXCLUDE_STIMULI_SEARCH
            ),
            exclude_trials=(
                args.exclude_trials
                if args.exclude_trials is not None
                else _DEFAULT_EXCLUDE_TRIALS_SEARCH
            ),
            split_strategy=args.split_strategy,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            stride=args.stride,
        )


if __name__ == "__main__":
    main()
