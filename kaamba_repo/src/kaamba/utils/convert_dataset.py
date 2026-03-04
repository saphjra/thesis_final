"""
Dataset format conversion guide.
Converts JSONL metadata to optimized formats for memory efficiency.
"""

import polars as pl
from pathlib import Path

import time
from datasets import DatasetBuilder, DatasetInfo, Features, Array2D, Sequence, Value
from pathlib import Path
import os


def convert_jsonl_to_parquet(
    input_path: str,
    output_path: str | Path | None = None,
    compression: str = "gzip",
    batch_size: int = 10000,
) -> None:
    """
    Convert JSONL metadata to Parquet format.

    Benefits of Parquet:
    - 50-80% smaller file size than JSONL
    - Columnar format enables selective column reading
    - Better compression algorithms
    - Faster read/write performance

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output Parquet file
        compression: 'snappy' (fast), 'gzip' (good compression), 'brotli' (best)
        batch_size: Number of rows per batch
    """
    if output_path is None:
        output_path = Path(input_path).with_suffix(".parquet")
    print(f"Converting {input_path} to Parquet...")
    start = time.time()

    # Use Polars lazy evaluation for memory efficiency
    data = pl.scan_ndjson(input_path)
    data.sink_parquet(output_path, compression=compression)

    elapsed = time.time() - start
    print(f"✓ Conversion complete in {elapsed:.2f}s")
    print(f"  Output: {output_path}")

    # Show file size comparison
    input_size = Path(input_path).stat().st_size / (1024**3)
    output_size = Path(output_path).stat().st_size / (1024**3)
    ratio = (1 - output_size/input_size) * 100

    print(f"  Size: {input_size:.2f}GB → {output_size:.2f}GB ({ratio:.1f}% reduction)")


def shard_large_dataset(
    input_path: str,
    output_dir: str,
    num_shards: int = 10,
    compression: str = "snappy",
) -> None:
    """
    Split large dataset into sharded Parquet files.

    Benefits of sharding:
    - Enables parallel loading with multiple workers
    - Reduces peak memory usage
    - Better for distributed training
    - Each shard can be processed independently

    Args:
        input_path: Path to input file (JSONL or Parquet)
        output_dir: Directory to save shards
        num_shards: Number of shards to create
        compression: Compression algorithm
    """
    print(f"Sharding dataset into {num_shards} files...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data lazily
    if input_path.endswith(".parquet"):
        data = pl.scan_parquet(input_path)
    else:
        data = pl.scan_ndjson(input_path)

    total_rows = data.select(pl.len()).collect().item()
    rows_per_shard = (total_rows + num_shards - 1) // num_shards

    print(f"Total rows: {total_rows}")
    print(f"Rows per shard: {rows_per_shard}")

    collected = data.collect()

    for shard_idx in range(num_shards):
        start_idx = shard_idx * rows_per_shard
        end_idx = min((shard_idx + 1) * rows_per_shard, total_rows)

        shard = collected[start_idx:end_idx]
        output_file = output_path / f"shard-{shard_idx:05d}-of-{num_shards:05d}.parquet"

        shard.write_parquet(str(output_file), compression=compression)
        print(f"  ✓ {output_file.name}: {len(shard)} rows")

    print(f"✓ Sharding complete: {output_path}")


def create_hf_dataset_config(
    input_path: str,
    output_dir: str,
    split_name: str = "train",
) -> None:
    """
    Create HuggingFace Datasets configuration.
    Enables easy loading with: load_dataset(output_dir)

    Args:
        input_path: Path to sharded dataset directory
        output_dir: Directory to save dataset config
        split_name: Name of split (train, test, etc.)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create dataset.py script
    dataset_script = ''''''


class CustomGazeDataset(DatasetBuilder):
    VERSION = "1.0.0"

    BUILDER_CONFIGS = []

    def _info(self):
        return DatasetInfo(
            features=Features({
                "input_seq": Sequence(Array2D((None, 2), dtype="float32")),
                "target_seq": Sequence(Array2D((None, 2), dtype="float32")),
                "participant_id": Value("string"),
                "stimulus_id": Value("string"),
            })
        )

    def _split_generators(self, dl_manager):
        # Auto-detect shard files
        data_dir = os.path.dirname(self.config.data_files) if self.config.data_files else "."
        shard_files = sorted(Path(data_dir).glob("shard-*.parquet"))
        return [
            {
                "name": "train",
                "gen_kwargs": {"shard_files": shard_files},
            }
        ]

    def _generate_examples(self, shard_files):
        for shard_idx, shard_file in enumerate(shard_files):
            data = pl.read_parquet(str(shard_file))
            for idx, row in enumerate(data.iter_rows(named=True)):
                yield f"{shard_idx}_{idx}", row


    # Create README
readme = f'''# Eyetracking Dataset

This is a sharded eyetracking dataset optimized for streaming.

## Format   
- Format: Parquet (columnar, compressed)
- Shards: Multiple files for parallel loading
- Compression: Snappy

## Usage

### With HuggingFace Datasets
```python
out_dir = "kaamba_dataset/sharded"
from datasets import load_dataset
dataset = load_dataset("output_dir", split="train", streaming=True)
```

### With Polars
```python
import polars as pl
data = pl.scan_parquet("output_dir/shard-*.parquet")
```

### With PyTorch
```python
from torch.utils.data import DataLoader
from datasets import load_dataset

dataset = load_dataset("output_dir", streaming=True)
loader = DataLoader(dataset, batch_size=32, num_workers=4)
```

## Memory Usage
- Lazy loading enabled for minimal RAM usage
- Streaming mode loads only required batches
- Each worker processes one shard independently

## Optimization Tips
1. Use `num_workers > 0` in DataLoader for parallel loading
2. Set `pin_memory=True` for GPU training
3. Adjust batch_size based on GPU memory
4. Monitor disk I/O to ensure sufficient bandwidth


    config_file = output_path / "README.md"
    with open(config_file, "w") as f:
        f.write(readme)

    print(f"✓ Configuration created: config_file")
'''

def estimate_memory_usage(
    input_path: str,
    context_len: int = 32,
    batch_size: int = 32,
) -> dict:
    """
    Estimate memory usage for different configurations.

    Args:
        input_path: Path to dataset
        context_len: Sequence length
        batch_size: Batch size

    Returns:
        Dict with memory estimates
    """
    # Load sample to estimate
    if input_path.endswith(".parquet"):
        data = pl.read_parquet(input_path, n_rows=100)
    else:
        data = pl.read_ndjson(input_path, n_rows=100)

    # Estimate memory per sample
    # Each sequence: 2 floats × context_len × 4 bytes (float32)
    bytes_per_seq = context_len * 2 * 4

    # Number of sequences per row (depends on gaze data length)
    # Assume average 1000 samples per row
    seqs_per_row = 1000 // context_len
    bytes_per_row = bytes_per_seq * seqs_per_row * 2  # input + target

    total_rows = data.shape[0] * 100  # Extrapolate
    total_bytes = total_rows * bytes_per_row

    # Batch memory
    batch_memory_bytes = batch_size * bytes_per_seq * 2

    return {
        "total_dataset_gb": total_bytes / (1024**3),
        "batch_memory_mb": batch_memory_bytes / (1024**2),
        "context_len": context_len,
        "batch_size": batch_size,
    }


if __name__ == "__main__":
    # Example: Convert and shard your dataset
    from constants import STIMULUS_FOLDER
    input_file = str(STIMULUS_FOLDER / "Goettingen/metadata_Goettingen.jsonl")

    # Step 1: Convert JSONL to Parquet
    print("=" * 60)
    print("STEP 1: Convert JSONL to Parquet")
    print("=" * 60)
    convert_jsonl_to_parquet(
        input_file,
        input_file.replace(".jsonl", ".parquet")
    )

    # Step 2: Shard the dataset
    print("\n" + "=" * 60)
    print("STEP 2: Shard dataset for parallel loading")
    print("=" * 60)
    shard_large_dataset(
        input_file.replace(".jsonl", ".parquet"),
        str(STIMULUS_FOLDER / "Goettingen/sharded"),
        num_shards=10,
    )

    # Step 3: Estimate memory
    print("\n" + "=" * 60)
    print("STEP 3: Memory usage estimates")
    print("=" * 60)
    estimates = estimate_memory_usage(input_file)
    print(f"Total dataset: {estimates['total_dataset_gb']:.2f} GB")
    print(f"Per-batch memory: {estimates['batch_memory_mb']:.2f} MB")

