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
    context_len: int = 32
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

            if self.accelerator.num_processes > 1:
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
    split_strategy: str = "participant",
    train_ratio: float = 0.80,
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    exclude_participants: Optional[List] = None,
    exclude_stimuli: Optional[List] = None,
    exclude_trials: Optional[List] = None,
    # Loader
    batch_size: int = 256,
    num_workers: int = 1,
    context_len: int = 32,
    stride: int = 1,
    sampling_step: int = 1,
    max_image_size: int = 224,
    # Training
    num_epochs: int = 100,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    grad_clip: float = 1.0,
    patience: int = 5,
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
    split_strategy: str = "participant",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    exclude_participants: Optional[List] = None,
    exclude_stimuli: Optional[List] = None,
    exclude_trials: Optional[List] = None,
    context_len: int = 128,
    sampling_step: int = 1,
    max_image_size: int = 224,
    num_workers: int = 1,
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
    study_dir = Path(log_dir) / study_name
    study_dir.mkdir(parents=True, exist_ok=True)

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
            "model_name": "google/vit-base-patch16-224",  # fixed for vit
            "d_model": trial.suggest_categorical("d_model", [128, 256, 512, 1024]),
            "n_layers": trial.suggest_int("n_layers", 4, 12),
            "n_mix": trial.suggest_int("n_mix", 3, 8),
            "image_embed_dim": trial.suggest_categorical("image_embed_dim", [256, 512]),
            "conditioning_mode": trial.suggest_categorical(
                "conditioning_mode", ["initial_state", "every_step"]
            ),
            "freeze_encoder": True,
        }
        model_config["model_name"] = ENCODER_CONFIGS[model_config["encoder_type"]][
            "model_name"
        ]
        lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)
        grad_clip = trial.suggest_float("grad_clip", 0.1, 2.0)
        batch_size = trial.suggest_categorical("batch_size", [256])
        stride = 32  # trial.suggest_int("stride", 1, 32)

        accelerator = Accelerator()
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
            raise optuna.TrialPruned("OOM")
        finally:
            gc.collect()
            torch.cuda.empty_cache()
            accelerator.end_training()

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


def _build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "search"], default="train")

    # shared data args
    p.add_argument("--datasets", nargs="+", default=["mcfw-gaze", "GGTG"])
    p.add_argument("--root", default="/home/janhof/thesis/data/")
    p.add_argument("--log_dir", default="/home/janhof/thesis/logs/runs")
    p.add_argument("--context_len", type=int, default=32)
    p.add_argument("--sampling_step", type=int, default=1)
    p.add_argument("--max_image_size", type=int, default=224)
    p.add_argument("--num_workers", type=int, default=1)

    # train-only
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=0.00047323940014353053)
    p.add_argument("--weight_decay", type=float, default=1.1930365027846787e-07)
    p.add_argument("--grad_clip", type=float, default=1.6233822288702624)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--resume_from", default=None)
    p.add_argument("--use_wandb", action="store_true")

    # search-only
    p.add_argument("--n_trials", type=int, default=50)
    p.add_argument("--n_epochs_per_trial", type=int, default=10)
    p.add_argument("--max_batches_per_epoch", type=int, default=128)
    p.add_argument("--study_name", default="gaze_mamba_search")
    p.add_argument("--storage", default=None)

    return p


def main():
    args = _build_parser().parse_args()

    if args.mode == "train":
        model_config = {
            **ENCODER_CONFIGS["siglip"],
            "d_model": 128,
            "n_layers": 4,
            "n_mix": 4,
            "image_embed_dim": 512,
            "conditioning_mode": "initial_state",
            "freeze_encoder": True,
        }
        accelerator = Accelerator()
        train_on_the_fly(
            model_config=model_config,
            dataset_name=args.datasets,
            root=args.root,
            split_strategy="random",
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
            exclude_trials=["", "4", "5"],
            exclude_participants=["P01", "P02", "P03", "P04", "P06", "P07", "P08"],
            exclude_stimuli=None,
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
            exclude_trials=["", "2", "4", "5"],
            exclude_participants=[
                "P01",
                "P02",
                "P03",
                "P04",
                "P05",
                "P06",
                "P07",
                "P08",
            ],
            exclude_stimuli=None,
        )


if __name__ == "__main__":
    main()
