"""
On-the-fly sequence generation for eyetracking data.
Sequences are created during iteration, not during preprocessing.

Key advantages:
- No extra storage needed for pre-computed sequences
- Flexible sequence length and stride at inference time
- Easier to experiment with different context lengths
- Minimal preprocessing time
"""

import torch
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import polars as pl
from pathlib import Path
from typing import Iterator, Dict, Optional, Tuple
import random
from constants import STIMULUS_FOLDER, SCREEN_RESOLUTION
from torchvision.transforms import v2
from torchvision.io import decode_image

class MyCustomTransform(v2.Pad):
    def __init__(self, *args, **kwargs):
        super().__init__(padding = 0, *args, **kwargs)

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be padded.

        Returns:
            PIL Image or Tensor: Padded image.

        """
        print(
            f"I'm transforming an image of shape {img.shape} "

        )
        pad_vals = [0, 0, img.shape[2] - img.shape[2], img.shape[2] - img.shape[1]]
        return v2.functional.pad(img, pad_vals, self.fill, self.padding_mode)



class OnTheFlyGazeDataset(IterableDataset):
    """
    Dataset that generates gaze sequences on-the-fly.

    Instead of pre-computing all sequences, this generates them during iteration.
    Raw gaze data is stored once, sequences are created as needed.

    Example:
        Raw data (1000 gaze points) → generates ~967 sequences with context_len=32
        Memory: Only stores 1000 points once, not ~967k sequences
    """



    def __init__(
        self,
        metadata_path: str,
        context_len: int = 32,
        stride: int = 1,
        lazy: bool = True,
        max_image_size: int = 512,
    ):
        """
        Args:
            metadata_path: Path to metadata file (parquet or jsonl)
                          Should contain 'data' field with list of gaze dicts
            context_len: Length of input sequence
            stride: Step between sequences (1 = all sequences, 2 = skip every other)
            lazy: Use Polars lazy evaluation (recommended for large datasets)
        """
        self.metadata_path = Path(metadata_path)
        self.datacollection_name = self.metadata_path.stem.split("_")[-1]
        print(self.datacollection_name) # Extract datacollection name from filename
        self.context_len = context_len
        self.stride = stride
        self.lazy = lazy
        self.max_image_size = max_image_size

        # Load metadata lazily
        if self.metadata_path.suffix == ".parquet":
            self.data = pl.scan_parquet(str(metadata_path)) if lazy else pl.read_parquet(str(metadata_path))
        elif self.metadata_path.suffix in [".jsonl", ".ndjson"]:
            self.data = pl.scan_ndjson(str(metadata_path)) if lazy else pl.read_ndjson(str(metadata_path))
        else:
            raise ValueError(f"Unsupported format: {self.metadata_path.suffix}")

    def __iter__(self) -> Iterator[Dict]:
        """
        Iterate over the dataset, yielding one sequence at a time.
        Sequences are generated on-the-fly from raw gaze data.
        """
        # Collect data if lazy
        data = self.data.collect() if self.lazy else self.data

        # Iterate through each recording session
        for row in data.iter_rows(named=True):
            gaze_data = row.get("data", [])

            # Skip if not enough data to create sequences
            if not gaze_data or len(gaze_data) < self.context_len + 1:
                continue

            # Generate sequences from this recording
            yield from self._generate_sequences(row, gaze_data)

    def _image_transform(self, image_path: Path, max_size=512) -> torch.Tensor:
        if self.max_image_size is not None:
            max_size = self.max_image_size

        screen_resolution = SCREEN_RESOLUTION[self.datacollection_name]  # SCREEN_RESOLUTION[self.datacollection_name] # for example (1920, 1080)  # Example screen resolution, adjust as needed
        image = decode_image(str(image_path))
        padding_val = [0, 0, screen_resolution[0] - image.shape[2], screen_resolution[1] - image.shape[1]]
        self.scaling_factor = max_size / screen_resolution[0]
        transform = v2.Compose([
            v2.Pad(padding=padding_val, padding_mode="edge"),
            v2.Resize(size=None, max_size=max_size),
            # ToDo just for testing purposes has to be adapted to the actual image size and model requirements max size
            MyCustomTransform(padding_mode="edge"),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # get normalized values for RGB channels (assuming image is in RGB format
        ])

        return transform(image)


    def _text_transform(self, text: str) -> torch.Tensor:
        # Placeholder for text transformation (e.g., tokenization)
        # For now, we return an empty tensor since we don't have text data
        return torch.empty(0)


    def _generate_sequences(
        self,
        row: Dict,
        gaze_data: list,
    ) -> Iterator[Dict]:
        """
        Generate all sequences from a single recording session.

        Args:
            row: Metadata row (contains participant_id, stimulus_id, etc.)
            gaze_data: List of gaze samples [{'pixel_x': ..., 'pixel_y': ...}, ...]

        Yields:
            Dict with input_seq, target_seq, stimulus image and metadata
        """
        transformed_image = self._image_transform(STIMULUS_FOLDER/ self.datacollection_name / row.get("file_name"))
        # Generate sequences with specified stride
        for i in range(0, len(gaze_data) - self.context_len, self.stride):
            # Extract input and target sequences
            input_gaze = gaze_data[i:i + self.context_len]
            target_gaze = gaze_data[i + 1:i + self.context_len + 1]

            # Convert to numpy arrays
            input_seq = np.array(
                [[g["pixel_x"] * self.scaling_factor  if g["pixel_x"] is not None else 0, g["pixel_y"] * self.scaling_factor  if g["pixel_y"] is not None else 0] for g in input_gaze],
                dtype=np.float32
            )
            target_seq = np.array(
                [[g["pixel_x"] * self.scaling_factor  if g["pixel_x"] is not None else 0, g["pixel_y"] * self.scaling_factor  if g["pixel_y"] is not None else 0] for g in target_gaze],
                dtype=np.float32
            )

            yield {
                "input_seq": torch.from_numpy(input_seq),
                "target_seq": torch.from_numpy(target_seq),
                #"participant_id": row.get("participant_id"),
                #"stimulus_id": row.get("stimulus_id"),
                "image": transformed_image,
            }



def create_on_the_fly_loader(
    metadata_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    context_len: int = 32,
    stride: int = 1,
    dataset_type: str = "standard",
    max_image_size: int = 224,
) -> DataLoader:
    """
    Create a DataLoader with on-the-fly sequence generation.

    Args:
        metadata_path: Path to metadata
        batch_size: Batch size
        num_workers: Number of workers for parallel loading
        context_len: Sequence length
        stride: Step between sequences
        dataset_type: "standard", "random_stride", or "adaptive"

    Returns:
        DataLoader ready for training
    """

    if dataset_type == "standard":
        dataset = OnTheFlyGazeDataset(
            metadata_path,
            context_len=context_len,
            stride=stride,
            max_image_size=max_image_size,
        )
    elif dataset_type == "random_stride":
        dataset = RandomStridedGazeDataset(
            metadata_path,
            context_len=context_len,
        )
    elif dataset_type == "adaptive":
        dataset = AdaptiveContextGazeDataset(
            metadata_path,
            min_context_len=max(16, context_len // 2),
            max_context_len=context_len * 2,
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
    )


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("On-the-Fly Sequence Generation Example")
    print("=" * 60)
    path = STIMULUS_FOLDER / "Goettingen" / "metadata_Goettingen.jsonl"  # Adjust path as needed
    import os
    print(os.path.exists(path))  # Check if file exists
    # Create loader
    loader = create_on_the_fly_loader(
        path,  # or "metadata.jsonl"
        batch_size=32,
        num_workers=4,
        context_len=32,
        stride=100,
        dataset_type="standard",
    )

    # Iterate - sequences generated on-demand
    print("\nIterating over batches (sequences generated on-the-fly):")
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= 5:  # Show first 5 batches
            break

        print(f"\nBatch {batch_idx}:")
        print(f"  Input shape: {batch['input_seq']}")
        print(f"  Target shape: {batch['target_seq'].shape}")
        print(f"  img shape: {batch['image'].shape}")

        # This is what you'd do in training:
        # output = model(batch['input_seq'])
        # loss = criterion(output, batch['target_seq'])

