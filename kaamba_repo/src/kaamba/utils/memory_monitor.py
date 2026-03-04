"""
Memory monitoring and profiling utilities for large dataset training.
"""

import torch
import psutil
import gc
from typing import Dict, Optional, Callable
from contextlib import contextmanager
import time
from pathlib import Path
import json


class MemoryMonitor:
    """Monitor RAM and GPU memory during training"""

    def __init__(self, log_dir: Optional[str] = None):
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

        self.memory_log = []
        self.peak_ram = 0
        self.peak_vram = 0

    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory usage"""
        # RAM
        process = psutil.Process()
        ram_info = process.memory_info()
        ram_used = ram_info.rss / (1024**3)  # GB

        # GPU
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / (1e9)
            gpu_reserved = torch.cuda.memory_reserved() / (1e9)
        else:
            gpu_allocated = 0
            gpu_reserved = 0

        return {
            "ram_gb": ram_used,
            "gpu_allocated_gb": gpu_allocated,
            "gpu_reserved_gb": gpu_reserved,
            "timestamp": time.time(),
        }

    def log_memory(self, step: int, phase: str = "training", verbose=False) -> None:
        """Log current memory usage"""
        stats = self.get_memory_stats()
        stats["step"] = step
        stats["phase"] = phase

        self.memory_log.append(stats)

        # Update peaks
        self.peak_ram = max(self.peak_ram, stats["ram_gb"])
        self.peak_vram = max(self.peak_vram, stats["gpu_allocated_gb"])

        # Print
        if verbose:
            print(
                f"[{phase}] Step {step:5d} | "
                f"RAM: {stats['ram_gb']:6.2f}GB (peak: {self.peak_ram:6.2f}GB) | "
                f"VRAM: {stats['gpu_allocated_gb']:6.2f}GB (peak: {self.peak_vram:6.2f}GB)"
            )

    def save_log(self, filename: str = "memory_log.json") -> None:
        """Save memory log to file"""
        if not self.log_dir:
            return

        log_file = self.log_dir / filename
        with open(log_file, "w") as f:
            json.dump(self.memory_log, f, indent=2)

        print(f"Memory log saved to {log_file}")


@contextmanager
def memory_tracker(name: str = "operation", verbose: bool = False):
    """Context manager to track memory of a block"""
    torch.cuda.empty_cache()
    gc.collect()

    start_stats = {
        "ram": psutil.Process().memory_info().rss / (1024**3),
        "gpu": torch.cuda.memory_allocated() / (1e9) if torch.cuda.is_available() else 0,
    }

    start_time = time.time()

    try:
        yield
    finally:
        elapsed = time.time() - start_time
        end_stats = {
            "ram": psutil.Process().memory_info().rss / (1024**3),
            "gpu": torch.cuda.memory_allocated() / (1e9) if torch.cuda.is_available() else 0,
        }

        ram_delta = end_stats["ram"] - start_stats["ram"]
        gpu_delta = end_stats["gpu"] - start_stats["gpu"]

        if verbose:
            print(
                f"\n{name}:\n"
                f"  Time: {elapsed:.2f}s\n"
                f"  RAM: {start_stats['ram']:.2f}GB → {end_stats['ram']:.2f}GB "
                f"({ram_delta:+.2f}GB)\n"
                f"  GPU: {start_stats['gpu']:.2f}GB → {end_stats['gpu']:.2f}GB "
                f"({gpu_delta:+.2f}GB)\n"
            )


def estimate_batch_memory(
    batch_size: int,
    context_len: int = 32,
    num_features: int = 2,  # pixel_x, pixel_y
    dtype_bytes: int = 4,  # float32
) -> Dict[str, float]:
    """Estimate memory usage per batch"""

    # Single sequence memory
    seq_bytes = context_len * num_features * dtype_bytes

    # Batch memory (input + target + model overhead ~2x)
    batch_bytes = batch_size * seq_bytes * 2 * 2.5

    return {
        "sequence_mb": seq_bytes / (1024**2),
        "batch_mb": batch_bytes / (1024**2),
        "batch_size": batch_size,
        "context_len": context_len,
    }


def get_optimal_config(
    available_ram_gb: float = None,
    available_vram_gb: float = None,
    context_len: int = 32,
) -> Dict:
    """Get optimal batch size and num_workers based on available memory"""

    if available_ram_gb is None:
        available_ram_gb = psutil.virtual_memory().available / (1024**3)

    if available_vram_gb is None and torch.cuda.is_available():
        available_vram_gb = (
            torch.cuda.get_device_properties(0).total_memory / (1e9)
            * 0.8  # Use 80% to be safe
        )

    # Conservative memory allocation
    target_batch_memory = 0.5 if available_vram_gb else 2.0  # GB

    # Estimate batch size
    seq_bytes = context_len * 2 * 4  # 2 features, float32
    batch_bytes_per_sample = seq_bytes * 2 * 2.5  # input + target + overhead
    batch_size = int((target_batch_memory * 1024**3) / batch_bytes_per_sample)
    batch_size = max(4, min(batch_size, 256))

    # Estimate num_workers
    if available_ram_gb > 32:
        num_workers = 8
    elif available_ram_gb > 16:
        num_workers = 4
    else:
        num_workers = 2

    return {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": available_vram_gb is not None,
        "prefetch_factor": 2,
        "persistent_workers": num_workers > 0,
        "available_ram_gb": available_ram_gb,
        "available_vram_gb": available_vram_gb,
    }

def get_summary() -> str:

    # Example usage
    print("=" * 60)
    print("Memory Configuration Recommendation")
    print("=" * 60)

    config = get_optimal_config()
    print(f"Recommended configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("Memory Usage Estimates")
    print("=" * 60)

    for batch_size in [32, 64, 128, 256]:
        stats = estimate_batch_memory(batch_size=batch_size)
        print(f"Batch size {batch_size:3d}: {stats['batch_mb']:.1f} MB")

    print("\nExample usage in training:")
    print("""
       monitor = MemoryMonitor(log_dir="logs")

       for epoch in range(num_epochs):
           for step, batch in enumerate(loader):
               with memory_tracker("forward_pass"):
                   loss = model(batch)

               if step % 100 == 0:
                   monitor.log_memory(step, "training")

           monitor.save_log()
       """)

if __name__ == "__main__":
    get_summary()
