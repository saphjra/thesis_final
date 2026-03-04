import torch
import numpy as np
import datasets
from datasets import Dataset, load_dataset, load_from_disk
from torchvision.transforms import v2
import polars as pl
from pathlib import Path
from typing import Optional

from kaamba_repo.kaamba_dataset.constants import ROOT_DIR


# MEMORY-OPTIMIZED EYETRACKING DATASET HANDLING
# Uses lazy evaluation and streaming to handle large datasets


class StreamingGazeDataset(torch.utils.data.IterableDataset):
    """Memory-efficient streaming dataset for eyetracking data.

    Key features:
    - Loads data on-the-fly instead of pre-loading
    - Uses Polars lazy evaluation for large CSV/Parquet files
    - Generates sequences at load time, not preprocessing time
    - Compatible with DataLoader with num_workers > 0
    """

    def __init__(
        self,
        metadata_path: str,
        context_len: int = 32,
        stride: int = 1,
        lazy: bool = True,
    ):
        self.metadata_path = Path(metadata_path)
        self.context_len = context_len
        self.stride = stride
        self.lazy = lazy

        # Load with Polars lazy evaluation for memory efficiency
        if self.metadata_path.suffix == ".parquet":
            self.data = pl.scan_parquet(str(metadata_path)) if lazy else pl.read_parquet(str(metadata_path))
        elif self.metadata_path.suffix in [".jsonl", ".ndjson"]:
            self.data = pl.scan_ndjson(str(metadata_path)) if lazy else pl.read_ndjson(str(metadata_path))
        else:
            raise ValueError(f"Unsupported format: {self.metadata_path.suffix}")

    def __iter__(self):
        """Iterate over sequences without loading entire dataset"""
        data = self.data.collect() if self.lazy else self.data

        for row in data.iter_rows(named=True):
            gaze_data = row.get("data", [])

            if not gaze_data or len(gaze_data) < self.context_len + 1:
                continue

            # Generate sequences on-the-fly
            for i in range(0, len(gaze_data) - self.context_len, self.stride):
                input_seq = np.array([
                    [g["pixel_x"], g["pixel_y"]]
                    for g in gaze_data[i:i + self.context_len]
                ], dtype=np.float32)

                target_seq = np.array([
                    [g["pixel_x"], g["pixel_y"]]
                    for g in gaze_data[i + 1:i + self.context_len + 1]
                ], dtype=np.float32)

                yield {
                    "input_seq": torch.from_numpy(input_seq),
                    "target_seq": torch.from_numpy(target_seq),
                    "participant_id": row.get("participant_id"),
                    "stimulus_id": row.get("stimulus_id"),
                }


class GazeDataset(Dataset):
    def __init__(self, samples, context_len=50):
        self.samples = []
        self.context_len = context_len

        for sample in samples:
            image = sample["image_tensor"]     # preloaded image
            gaze = sample["gaze_tensor"]       # shape [T, 3]

            for i in range(len(gaze) - context_len - 1):
                input_seq = gaze[i:i+context_len]
                target_seq = gaze[i+1:i+context_len+1]

                self.samples.append((image, input_seq, target_seq))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image, x, y = self.samples[idx]
        return image, x, y

def transform_gaze(example, context_len = 32):

    gaze = example["data"]
    inputs, targets = [], []

    for i in range(len(gaze) - context_len - 1):
        inputs.append([tuple(di.values()) for di in gaze[i:i+context_len]])
        targets.append([tuple(di.values()) for di in gaze[i+1:i+context_len+1]])

    return {
        "input_seq": inputs,
        "target_seq": targets,
    }

def transform_image(example):
    transforms = v2.Compose([
        v2.RandomResizedCrop(size=(224, 224), antialias=True),
        v2.PILToTensor(),
        # v2.RandomHorizontalFlip(p=0.5),
        # v2.ToDtype(torch.float32, scale=True),
        #v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return {
        "transformed_image": transforms(example["image"]) #[example["file_name"]] * len(inputs)
    }


def preprocess_data(dataset_path = str(ROOT_DIR) + "/data/MultiplEYE_DE_DE_Goettingen_1_2026/stimuli_MultiplEYE_DE_DE_Goettingen_1_2026/stimuli_images_de_de_1"):
    dataset = load_dataset(dataset_path)
    dataset = dataset.map(transform_image)

    print("Windowing complete")
    dataset = dataset.remove_columns(["participant_id", "text", "data", "file_name"])
    dataset.save_to_disk(ROOT_DIR / f"kaamba_dataset")

    print("Data preprocessing complete and saved to disk.")


if __name__     == "__main__":
    datasets.logging.set_verbosity_info()
    preprocess_data()