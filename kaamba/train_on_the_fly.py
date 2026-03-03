"""
Training script using on-the-fly sequence generation.
Demonstrates the best practice for memory-efficient training.
"""

import torch
from torch.utils.data import DataLoader
from kaamba.net.models.tamba import GazePredictor
from kaamba.utils.loss_functions import gaussian_nll
from kaamba.utils.on_the_fly_dataset import create_on_the_fly_loader
from kaamba.utils.memory_monitor import MemoryMonitor, memory_tracker
from pathlib import Path
from kaamba_dataset.constants import ROOT_DIR

def train_on_the_fly(
    metadata_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    num_epochs: int = 3,
    context_len: int = 32,
    stride: int = 1,
    device: str = "cuda",
    log_dir: str = "logs",
    max_image_size: int = 224,
):
    """
    Train model using on-the-fly sequence generation.

    This approach:
    1. Stores raw gaze data once (no pre-computed sequences)
    2. Generates sequences during data loading
    3. Enables flexible sequence length experimentation
    4. Minimal disk space, minimal preprocessing time

    Args:
        metadata_path: Path to metadata file (parquet or jsonl)
        batch_size: Batch size for training
        num_workers: Number of parallel data loading workers
        num_epochs: Number of training epochs
        context_len: Sequence context length
        stride: Step between sequences
        device: 'cuda' or 'cpu'
        log_dir: Directory to save logs
        :param max_image_size:
    """

    if not torch.cuda.is_available():
        device = "cpu"
        print("⚠️  CUDA not available, using CPU")

    print("=" * 70)
    print("TRAINING WITH ON-THE-FLY SEQUENCE GENERATION")
    print("=" * 70)
    print(f"Dataset: {metadata_path}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Workers: {num_workers}")
    print(f"Context length: {context_len}")
    print(f"Stride: {stride}")
    print()

    # Create memory monitor
    monitor = MemoryMonitor(log_dir=log_dir)

    # Create data loader with on-the-fly generation
    print("Creating data loader...")
    with memory_tracker("DataLoader Creation"):
        loader = create_on_the_fly_loader(
            metadata_path=metadata_path,
            batch_size=batch_size,
            num_workers=num_workers,
            context_len=context_len,
            stride=stride,
            dataset_type="standard",
            max_image_size = max_image_size,
        )

    # Initialize model
    print("Initializing model...")
    model = GazePredictor().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=1e-5,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )

    model.train()

    # Training loop
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")

        total_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(loader):
            # Move to device
            with memory_tracker("Data Transfer to Device"):
                images = batch["image"].to(device)
                inputs = batch["input_seq"].to(device)
                targets = batch["target_seq"].to(device)

            # Forward pass
            optimizer.zero_grad()

            with memory_tracker("Forward Pass"):
                mu, sigma = model(images, inputs)
                loss = gaussian_nll(mu, sigma, targets)

            # Backward pass
            with memory_tracker("Backward Pass"):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Log memory every 100 batches
            if (batch_idx + 1) % 100 == 0:
                avg_loss = total_loss / num_batches
                monitor.log_memory(batch_idx + 1, phase="training")
                print(
                    f"  Batch {batch_idx + 1:5d} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Avg: {avg_loss:.4f}"
                )

        avg_epoch_loss = total_loss / num_batches if num_batches > 0 else 0
        scheduler.step()

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Total batches: {num_batches}")
        print(f"  Avg loss: {avg_epoch_loss:.4f}")
        print(f"  Learning rate: {scheduler.get_last_lr()[0]:.2e}")

        # Save checkpoint
        checkpoint_path = Path(log_dir) / f"checkpoint_epoch_{epoch+1}.pt"
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_epoch_loss,
            },
            checkpoint_path
        )
        print(f"  Checkpoint saved: {checkpoint_path}")

    # Save final log
    monitor.save_log()

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Peak RAM: {monitor.peak_ram:.2f} GB")
    print(f"Peak VRAM: {monitor.peak_vram:.2f} GB")

    return model


if __name__ == "__main__":
    # Show comparison
    # Train with on-the-fly generation
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    import os
    print(os.path.exists(ROOT_DIR/"kaamba_dataset/stimuli/Goettingen/metadata_Goettingen.jsonl"))

    model = train_on_the_fly(
        metadata_path=ROOT_DIR / "kaamba_dataset/stimuli/Goettingen/metadata_Goettingen.jsonl",
        batch_size=128,
        num_workers=2,
        num_epochs=3,
        context_len=64,
        stride=10,
        log_dir="logs/on_the_fly",
        max_image_size=224
    )

