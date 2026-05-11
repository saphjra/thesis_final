"""
Training script using on-the-fly sequence generation with Accelerate multi-GPU support.
"""

from typing import Optional
from pathlib import Path

import torch
from accelerate import Accelerator
from kaamba.net.models.test_mamba import GazePredictor
from kaamba.utils.loss_functions import gaussian_nll
from kaamba.utils.on_the_fly_dataset import create_on_the_fly_loader
from kaamba.utils.memory_monitor import MemoryMonitor, memory_tracker

from tqdm import tqdm


def train_on_the_fly(
    root: Optional[str] = None,
    dataset_name: Optional[str] = None,
    metadata_path: Optional[str] = None,
    batch_size: int = 512,
    num_workers: int = 4,
    num_epochs: int = 3,
    context_len: int = 32,
    stride: int = 1,
    sampling_ratio: Optional[int] = 1,
    log_dir: str = "logs",
    max_image_size: int = 224,
    image_folder_path: str = None,
    subset: Optional[dict] = None,
    resume_from: Optional[str] = None,  # path to checkpoint
):
    # --- Accelerator replaces manual device management ---
    accelerator = Accelerator()
    accelerator.print(f"Total GPUs: {accelerator.num_processes}")
    print(
        f"This process is rank {accelerator.process_index} on device {accelerator.device}"
    )
    try:
        device = accelerator.device

        accelerator.print("=" * 70)
        accelerator.print("TRAINING WITH ON-THE-FLY SEQUENCE GENERATION (multi-GPU)")
        accelerator.print("=" * 70)
        accelerator.print(f"Dataset:        {metadata_path}")
        accelerator.print(f"Device:         {device}")
        accelerator.print(f"Num processes:  {accelerator.num_processes}")
        accelerator.print(f"Batch size:     {batch_size} (per GPU)")
        accelerator.print(f"Workers:        {num_workers}")
        accelerator.print(f"Context length: {context_len}")
        accelerator.print(f"Stride:         {stride}")

        # Memory monitor — only meaningful on main process
        monitor = MemoryMonitor(log_dir=log_dir)

        # --- Data loader ---
        accelerator.print("Creating data loader...")
        with memory_tracker("DataLoader Creation"):
            loader = create_on_the_fly_loader(
                batch_size=batch_size,
                num_workers=num_workers,
                context_len=context_len,
                stride=stride,
                dataset_type="standard",
                max_image_size=max_image_size,
                image_folder_path=image_folder_path,
                root=root,
                dataset_name=dataset_name,
                subset=subset,
                sampling_ratio=sampling_ratio,
            )

        # --- Model ---
        accelerator.print("Initializing model...")

        model = GazePredictor()
        accelerator.print(
            f"Model parameters: {sum(p.numel() for p in model.parameters()):,}"
        )

        # --- Optimizer & scheduler ---
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-4,
            weight_decay=1e-5,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs
        )

        # --- Prepare everything for distributed training ---
        # accelerator.prepare() handles:
        #   - Moving model to the correct GPU
        #   - Wrapping model with DDP
        #   - Sharding the dataloader across GPUs
        model, optimizer, loader, scheduler = accelerator.prepare(
            model, optimizer, loader, scheduler
        )
        start_epoch = 0
        if resume_from is not None:
            start_epoch = load_checkpoint(
                resume_from, model, optimizer, scheduler, accelerator
            )

        # training loop — start from loaded epoch

        model.train()

        # --- Training loop ---
        accelerator.print("\n" + "=" * 70)
        accelerator.print("TRAINING")
        accelerator.print("=" * 70)

        for epoch in range(start_epoch, num_epochs):
            accelerator.print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")

            total_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch + 1}")):
                # Data is already on the correct device after accelerator.prepare()
                images = batch["image"]
                inputs = batch["input_seq"]
                targets = batch["target_seq"]

                optimizer.zero_grad()

                # Forward
                mu, sigma = model(images, inputs)
                loss = gaussian_nll(mu, sigma, targets)

                # Backward — accelerator handles gradient sync across GPUs
                accelerator.backward(loss)

                # Gradient clipping must happen AFTER accelerator.backward()
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                if (batch_idx + 1) % 100 == 0:
                    avg_loss = total_loss / num_batches
                    if accelerator.is_main_process:
                        monitor.log_memory(batch_idx + 1, phase="training")
                    accelerator.print(
                        f"  Batch {batch_idx + 1:5d} | "
                        f"Loss: {loss.item():.4f} | "
                        f"Avg: {avg_loss:.4f}"
                        f"  sigma mean: {sigma.mean().item():.4f} "
                        f"min: {sigma.min().item():.4f} "
                        f"max: {sigma.max().item():.4f}"
                    )

            avg_epoch_loss = total_loss / num_batches if num_batches > 0 else 0.0
            scheduler.step()

            accelerator.print(f"\nEpoch {epoch + 1} Summary:")
            accelerator.print(f"  Total batches: {num_batches}")
            accelerator.print(f"  Avg loss:      {avg_epoch_loss:.4f}")
            accelerator.print(f"  Learning rate: {scheduler.get_last_lr()[0]:.2e}")

            # --- Checkpoint: only save on the main process ---
            # wait_for_everyone() ensures all GPUs have finished the epoch first
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                checkpoint_path = Path(log_dir) / f"checkpoint_epoch_{epoch + 1}.pt"
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": unwrapped_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "loss": avg_epoch_loss,
                    },
                    checkpoint_path,
                )
                accelerator.print(f"  Checkpoint saved: {checkpoint_path}")

        # --- Final log (main process only) ---
        if accelerator.is_main_process:
            monitor.save_log()

        accelerator.print("\n" + "=" * 70)
        accelerator.print("TRAINING COMPLETE")
        accelerator.print("=" * 70)

        if accelerator.is_main_process:
            accelerator.print(f"Peak RAM:  {monitor.peak_ram:.2f} GB")
            accelerator.print(f"Peak VRAM: {monitor.peak_vram:.2f} GB")

        return accelerator.unwrap_model(model)
    finally:
        # Always clean up, even if training crashes or is interrupted
        accelerator.end_training()


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    accelerator: Accelerator,
) -> int:
    """
    Load a checkpoint and restore model, optimizer, and scheduler state.
    Returns the epoch to resume from.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        accelerator.print(
            f"[checkpoint] No checkpoint found at {checkpoint_path}, starting fresh."
        )
        return 0

    accelerator.print(f"[checkpoint] Resuming from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=accelerator.device)

    # Unwrap model before loading state dict
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.load_state_dict(ckpt["model_state_dict"])

    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    # gracefully handle checkpoints saved before scheduler state was added
    if "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    else:
        accelerator.print(
            "[checkpoint] No scheduler state found, scheduler starts fresh."
        )

    epoch = ckpt["epoch"]
    loss = ckpt.get("loss", float("nan"))
    accelerator.print(f"[checkpoint] Resumed from epoch {epoch}, loss={loss:.4f}")

    return epoch  # training loop should start from this epoch


def main():
    # multiprocessing.set_start_method("spawn", force=True)
    accelerator_print = print  # plain print is fine here — runs before Accelerator init

    accelerator_print("\n" + "=" * 70)
    accelerator_print("STARTING TRAINING")
    accelerator_print("=" * 70)

    config_mcfw = {
        "dataset_name": "mcfw-gaze",
        # "subset": {"subject_id": ["001"]},
        "batch_size": 1024,  # per-GPU batch size
        "num_workers": 2,
        "num_epochs": 100,
        "context_len": 32,
        "stride": 30,
        "sampling_ratio": 1,
        "log_dir": "/home/janhof/thesis/logs",
        "max_image_size": 224,
        "root": "/home/janhof/thesis/data/",
        "resume_from": "/home/janhof/thesis/logs/checkpoint_epoch_30.pt",  # None to start fresh
    }
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"

    train_on_the_fly(**config_mcfw)


if __name__ == "__main__":
    main()
