"""
Training script using DataloaderConfigBuilder for clean train/val/test splits.
"""

from typing import Optional, List
from pathlib import Path

import torch
from accelerate import Accelerator
from kaamba.net.models.kaamba import build_gaze_predictor
from kaamba.utils.dataloader_config_builder import DataloaderConfigBuilder
from kaamba.utils.loss_functions import gmm_nll
from kaamba.utils.memory_monitor import MemoryMonitor, memory_tracker
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Encoder presets
# ---------------------------------------------------------------------------

ENCODER_CONFIGS = {
    "vit_base": {"encoder_type": "vit", "model_name": "google/vit-base-patch16-224"},
    "vit_large": {"encoder_type": "vit", "model_name": "google/vit-large-patch16-224"},
    "resnet": {"encoder_type": "resnet"},
}


# ---------------------------------------------------------------------------
# TrainingMonitor — unchanged
# ---------------------------------------------------------------------------


class TrainingMonitor:
    def __init__(self, patience=5, min_delta=1e-4, max_grad_norm=10.0):
        self.patience = patience
        self.min_delta = min_delta
        self.max_grad_norm = max_grad_norm
        self.best_loss = float("inf")
        self.patience_counter = 0

    def check_epoch_loss(self, current_loss):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.patience_counter = 0
            return False, f"✓ Loss improved to {current_loss:.6f}"
        self.patience_counter += 1
        if self.patience_counter >= self.patience:
            return True, f"✗ No improvement for {self.patience} epochs"
        return (
            False,
            f"  Patience: {self.patience_counter}/{self.patience} (best: {self.best_loss:.6f})",
        )

    def check_loss_validity(self, loss):
        if loss != loss or not (-1e10 < loss < 1e10):
            return True, f"✗ Invalid loss: {loss}"
        return False, ""

    def check_gradient_norm(self, model):
        total_norm = (
            sum(
                p.grad.data.norm(2).item() ** 2
                for p in model.parameters()
                if p.grad is not None
            )
            ** 0.5
        )
        if total_norm > self.max_grad_norm:
            return True, f"✗ Gradient norm too large: {total_norm:.4f}"
        return False, ""


# ---------------------------------------------------------------------------
# Validation — updated to use gmm_nll
# ---------------------------------------------------------------------------


def validate(model, val_loader, accelerator):
    model.eval()
    total_loss, num_batches = 0.0, 0

    with torch.no_grad():
        for batch in tqdm(
            val_loader, desc="Validation", disable=not accelerator.is_main_process
        ):
            images = batch["image"].to(accelerator.device)
            inputs = batch["input_seq"].to(accelerator.device)
            targets = batch["target_seq"].to(accelerator.device).permute(0, 2, 1)

            pi, mu, log_sx, log_sy, rho_raw = model(images, inputs)
            loss = gmm_nll(pi, mu, log_sx, log_sy, rho_raw, targets)
            total_loss += loss.item()
            num_batches += 1

    avg = total_loss / num_batches if num_batches > 0 else 0.0
    avg = (
        accelerator.gather(torch.tensor([avg], device=accelerator.device)).mean().item()
    )

    model.train()
    return avg


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train_on_the_fly(
    # Model
    model_config: dict,
    # Dataset
    dataset_name: str | List[str],
    root: str,
    split_strategy: str = "participant",  # "participant"|"stimulus"|"trial"|"random"
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    exclude_participants: Optional[List] = None,
    exclude_stimuli: Optional[List] = None,
    exclude_trials: Optional[List] = None,
    # Loader
    batch_size: int = 64,
    num_workers: int = 4,
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
    # Misc
    log_dir: str = "logs",
    resume_from: Optional[str] = None,
    accelerator: Optional[Accelerator] = None,
):
    _owns_accelerator = accelerator is None
    if _owns_accelerator:
        accelerator = Accelerator()

    try:
        accelerator.print("=" * 70)
        accelerator.print("TAMBA TRAINING")
        accelerator.print("=" * 70)
        accelerator.print(f"Dataset:          {dataset_name}")
        accelerator.print(
            f"Split strategy:   {split_strategy}  "
            f"({train_ratio:.0%} / {val_ratio:.0%} / {test_ratio:.0%})"
        )
        accelerator.print(f"Model config:     {model_config}")

        monitor = MemoryMonitor(log_dir=log_dir)

        # ── 1. Build loaders via DataloaderConfigBuilder ──────────────────
        accelerator.print("\nBuilding dataloaders...")
        datasets = [dataset_name] if isinstance(dataset_name, str) else dataset_name

        builder = DataloaderConfigBuilder(
            datasets=datasets,  # ← passes list directly
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

        accelerator.print(f"Train config: {loader_configs['train']}")
        accelerator.print(f"Val config:   {loader_configs['val']}")
        accelerator.print(f"Test config:  {loader_configs['test']}")

        # ── 2. Model ──────────────────────────────────────────────────────
        accelerator.print("\nInitializing model...")
        model = build_gaze_predictor(**model_config)
        accelerator.print(
            f"Parameters: {sum(p.numel() for p in model.parameters()):,} total, "
            f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable"
        )

        # ── 3. Optimizer & scheduler ──────────────────────────────────────
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs
        )
        training_monitor = TrainingMonitor(patience=patience)

        model, optimizer, train_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, scheduler
        )
        # val/test loaders don't need gradient tracking — prepare separately
        val_loader, test_loader = accelerator.prepare(val_loader, test_loader)

        # ── 4. Optional resume ────────────────────────────────────────────
        start_epoch = 0
        if resume_from:
            start_epoch = load_checkpoint(
                resume_from, model, optimizer, scheduler, accelerator
            )

        model.train()

        # ── 5. Training loop ──────────────────────────────────────────────
        accelerator.print("\n" + "=" * 70)
        accelerator.print("TRAINING")
        accelerator.print("=" * 70)

        for epoch in range(start_epoch, num_epochs):
            accelerator.print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
            total_loss, num_batches = 0.0, 0

            for batch_idx, batch in enumerate(
                tqdm(
                    train_loader,
                    desc=f"Epoch {epoch + 1}",
                    disable=not accelerator.is_main_process,
                )
            ):
                images = batch["image"]
                inputs = batch["input_seq"]
                targets = batch["target_seq"].permute(0, 2, 1)  # (B,T,2)

                optimizer.zero_grad()
                pi, mu, log_sx, log_sy, rho_raw = model(images, inputs)
                loss = gmm_nll(pi, mu, log_sx, log_sy, rho_raw, targets)

                # Debug print on first batch of each epoch
                if batch_idx == 0 and accelerator.is_main_process:
                    accelerator.print(
                        f"\n  [debug] mu    {mu.min():.3f} / {mu.mean():.3f} / {mu.max():.3f}"
                        f"\n  [debug] log_sx {log_sx.min():.3f} / {log_sx.mean():.3f} / {log_sx.max():.3f}"
                        f"\n  [debug] rho   {rho_raw.min():.3f} / {rho_raw.max():.3f}"
                        f"\n  [debug] target {targets.min():.3f} / {targets.max():.3f}"
                        f"\n  [debug] loss  {loss.item():.4f}"
                    )

                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                if (batch_idx + 1) % 10 == 0 and accelerator.is_main_process:
                    monitor.log_memory(batch_idx + 1, phase="training")
                    accelerator.print(
                        f"  Batch {batch_idx + 1:5d} | "
                        f"loss {loss.item():.4f} | avg {total_loss / num_batches:.4f}"
                    )

            avg_train_loss = total_loss / num_batches if num_batches > 0 else 0.0
            scheduler.step()

            # ── Validation ────────────────────────────────────────────────
            avg_val_loss = validate(model, val_loader, accelerator)

            accelerator.print(
                f"\nEpoch {epoch + 1} | train {avg_train_loss:.4f} "
                f"| val {avg_val_loss:.4f} "
                f"| lr {scheduler.get_last_lr()[0]:.2e}"
            )

            # ── Stopping checks ───────────────────────────────────────────
            should_stop, stop_reason = False, ""
            for check_fn, args in [
                (training_monitor.check_loss_validity, (avg_val_loss,)),
                (training_monitor.check_gradient_norm, (model,)),
                (training_monitor.check_epoch_loss, (avg_val_loss,)),
            ]:
                flag, msg = check_fn(*args)
                accelerator.print(f"  {msg}")
                if flag:
                    should_stop, stop_reason = True, msg
                    break

            # ── Checkpoint ────────────────────────────────────────────────
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                ckpt = {
                    "epoch": epoch + 1,
                    "model_state_dict": accelerator.unwrap_model(model).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "model_config": model_config,
                    "loader_configs": {
                        k: v.to_dict() for k, v in loader_configs.items()
                    },
                }
                ckpt_path = Path(log_dir) / f"checkpoint_epoch_{epoch + 1}.pt"
                torch.save(ckpt, ckpt_path)
                accelerator.print(f"  Checkpoint: {ckpt_path}")

            if should_stop:
                accelerator.print(f"\n⚠️  STOPPED — {stop_reason}")
                break

        if accelerator.is_main_process:
            monitor.save_log()
            accelerator.print(
                f"\nPeak RAM: {monitor.peak_ram:.2f} GB | "
                f"Peak VRAM: {monitor.peak_vram:.2f} GB"
            )

        accelerator.print("\nTRAINING COMPLETE")
        return accelerator.unwrap_model(model)

    finally:
        if _owns_accelerator:
            accelerator.end_training()


# ---------------------------------------------------------------------------
# Checkpoint loading — unchanged
# ---------------------------------------------------------------------------


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, accelerator):
    p = Path(checkpoint_path)
    if not p.exists():
        accelerator.print(f"[checkpoint] Not found at {p}, starting fresh.")
        return 0
    accelerator.print(f"[checkpoint] Resuming from {p}")
    ckpt = torch.load(p, map_location=accelerator.device)
    accelerator.unwrap_model(model).load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    accelerator.print(
        f"[checkpoint] Epoch {ckpt['epoch']} | "
        f"train {ckpt.get('train_loss', float('nan')):.4f} | "
        f"val {ckpt.get('val_loss', float('nan')):.4f}"
    )
    return ckpt["epoch"]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    model_config = {
        **ENCODER_CONFIGS["vit_base"],
        "d_model": 256,
        "n_layers": 8,
        "n_mix": 5,
        "image_embed_dim": 512,
        "conditioning_mode": "initial_state",
        "freeze_encoder": True,
    }

    train_config = {
        "dataset_name": ["mcfw-gaze", "GGTG"],  # ← now a list
        "root": "/home/janhof/thesis/data/",
        "split_strategy": "participant",
        "train_ratio": 0.70,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "exclude_participants": None,
        "exclude_stimuli": None,
        "exclude_trials": None,
        "batch_size": 128,
        "num_workers": 1,
        "context_len": 32,
        "stride": 1,
        "sampling_step": 1,
        "max_image_size": 224,
        "num_epochs": 100,
        "lr": 1.76e-05,
        "weight_decay": 6.58e-06,
        "grad_clip": 0.69,
        "patience": 5,
        "log_dir": "/home/janhof/thesis/logs",
        "resume_from": None,
    }

    accelerator = Accelerator()
    train_on_the_fly(model_config=model_config, accelerator=accelerator, **train_config)
    accelerator.end_training()


if __name__ == "__main__":
    main()
