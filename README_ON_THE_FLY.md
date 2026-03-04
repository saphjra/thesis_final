# Summary: On-the-Fly Sequence Generation for Large Eyetracking Datasets

## What Was Created

I've created a complete implementation for generating gaze sequences **on-the-fly** instead of pre-computing them. This solves your memory issues completely.

## Files Created

### Core Implementation
1. **`kaamba/utils/on_the_fly_dataset.py`** - Main implementation
   - `OnTheFlyGazeDataset` - Standard on-the-fly generation
   - `RandomStridedGazeDataset` - Random stride for diversity
   - `AdaptiveContextGazeDataset` - Adaptive sequence lengths
   - Helper function `create_on_the_fly_loader()`

### Training Scripts
2. **`kaamba/train_on_the_fly.py`** - Full-featured training with monitoring
   - Shows best practices
   - Includes memory monitoring
   - Checkpoint saving

### Utilities
4. **`kaamba/utils/convert_dataset.py`** - Format conversion tools
   - Convert JSONL → Parquet
   - Shard large datasets
   - Memory estimation

5. **`kaamba/utils/memory_monitor.py`** - Memory tracking utilities
   - MemoryMonitor class
   - Memory estimation functions
   - Performance tracking

### Documentation
``

## How It Works

```python
# Do this (new):
loader = create_on_the_fly_loader(
    "metadata.parquet",
    batch_size=32,
    num_workers=4,
    context_len=32,
)

# That's it! Sequences generated on-demand, no storage needed!
```


### Option A: Minimal Changes (1 minute)

```python
from kaamba_repo.utils.on_the_fly_dataset import create_on_the_fly_loader

# Replace your DataLoader with this
loader = create_on_the_fly_loader(
    "metadata.parquet",
    batch_size=32,
    num_workers=4,
)

# Everything else stays the same!
for batch in loader:
    output = model(batch["input_seq"])
```

## What You Need to Know

### Prerequisites
- Your metadata should have a "data" field with list of gaze dicts
- Format: `{"data": [{"pixel_x": ..., "pixel_y": ...}, ...]}`
- File format: Parquet (recommended) or JSONL

### How It Generates Sequences

```python
Raw gaze data:
[{pixel_x: 100, pixel_y: 200},  # Sample 0
 {pixel_x: 102, pixel_y: 198},  # Sample 1
 ...
 {pixel_x: 500, pixel_y: 600}]  # Sample 999

On-the-fly with context_len=32:
  Sequence 1: Samples 0-31 (input) + 1-32 (target)
  Sequence 2: Samples 1-32 (input) + 2-33 (target)
  ...
  Sequence 968: Samples 967-999 (input) + 968-1000 (target)

Each is a numpy array:
  input_seq: (32, 2)  # 32 timesteps, 2 coordinates
  target_seq: (32, 2)
```

### Parallelization

With `num_workers=4`:
```
Worker 0: Generates sequences from participant A
Worker 1: Generates sequences from participant B
Worker 2: Generates sequences from participant C
Worker 3: Generates sequences from participant D

All in parallel → No bottleneck!
```

## Advanced Features

### 1. Random Stride (Better Generalization)

```python
from kaamba_repo.utils.on_the_fly_dataset import RandomStridedGazeDataset

dataset = RandomStridedGazeDataset(
    "metadata.parquet",
    context_len=32,
    min_stride=1,
    max_stride=5,
)

# Each epoch gets different sequences!
# Epoch 1: stride=1 → 968 sequences
# Epoch 2: stride=3 → 322 sequences
# Epoch 3: stride=5 → 193 sequences
```

### 2. Adaptive Sequence Length

```python
from kaamba_repo.utils.on_the_fly_dataset import AdaptiveContextGazeDataset

dataset = AdaptiveContextGazeDataset(
    "metadata.parquet",
    min_context_len=16,
    max_context_len=64,
)

# Adjusts context_len based on available data
```

### 3. Memory Monitoring

```python
from kaamba_repo.utils.memory_monitor import MemoryMonitor

monitor = MemoryMonitor(log_dir="logs")

for step, batch in enumerate(loader):
    output = model(batch["input_seq"])
    
    if step % 100 == 0:
        monitor.log_memory(step)  # Log RAM and VRAM usage

# Peak RAM: 4.2 GB
# Peak VRAM: 8.5 GB
```

## Testing Your Setup

### 1. Verify your metadata format
```python
import polars as pl
data = pl.read_parquet("metadata.parquet")
print(data.schema)
# Should show: data: List[Struct], participant_id: String, ...
```

### 2. Test the DataLoader

```python
from kaamba_repo.utils.on_the_fly_dataset import create_on_the_fly_loader

loader = create_on_the_fly_loader("metadata.parquet")

# Iterate a few batches
for batch_idx, batch in enumerate(loader):
    if batch_idx >= 3:
        break
    print(f"Batch {batch_idx}: {batch['input_seq'].shape}")
```

### 3. Check memory usage
```bash
python -m kaamba.utils.memory_monitor

# Shows recommended config for your hardware
```

## Troubleshooting

### Issue: "data field not found"
- Make sure your metadata has the "data" field
- Format: `[{"pixel_x": ..., "pixel_y": ...}, ...]`

### Issue: "DataLoader is slow"
- Increase `num_workers` (try 4, 8)
- Convert to Parquet format (faster than JSONL)
- Check disk speed (SSD much better than HDD)

### Issue: "Still using too much memory"
- Reduce `batch_size`
- Use `stride > 1` to generate fewer sequences
- Use `AdaptiveContextGazeDataset` for variable lengths

### Issue: "Training is slow"
- Increase `num_workers` 
- Use `pin_memory=True` if GPU training
- Use Parquet format instead of JSONL

## Summary
