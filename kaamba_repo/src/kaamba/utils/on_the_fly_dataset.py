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
from typing import Iterator, Dict, Optional, List
import os
import hashlib
import pickle
import json
import pymovements as pm
from tqdm import tqdm

from torchvision.transforms import v2
from torchvision.io import decode_image


class MyCustomTransform(v2.Pad):
    def __init__(self, *args, **kwargs):
        super().__init__(padding=0, *args, **kwargs)

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be padded.

        Returns:
            PIL Image or Tensor: Padded image.

        """
        # print(f"I'm transforming an image of shape {img.shape} ")
        pad_vals = [0, 0, img.shape[2] - img.shape[2], img.shape[2] - img.shape[1]]
        return v2.functional.pad(img, pad_vals, self.fill, self.padding_mode)


class PymovementsOnTheFlyGazeDataset(IterableDataset):
    """
    Dataset that generates gaze sequences on-the-fly using pymovements datasets.

    Loads stimuli and gazeframes from pymovements, generates sequences during iteration.
    """

    def __init__(
        self,
        dataset_name: str,
        context_len: int = 32,
        max_image_size: int = 512,
        sampling_step: Optional[int] = 1,
        stride: Optional[int] = 32,
        root: Optional[str] = None,
        subset: Optional[dict] = None,
        cache_dir: Optional[str] = "/home/janhof/thesis/data/seq_cache",
        stimulus: Optional[List[str]] = None,
        fill_strategy: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            dataset_name: Name of the pymovements dataset (e.g., 'GGTG', 'MultiplEYE_DE_DE_Goettingen_1_2026')
            context_len: Length of input sequence
            stride: step between sequences
            sampling_step: Step between timesteps, e.g. a way to downsample the raw data
            max_image_size: Max size for image resizing
            root: Root directory for dataset
            subset: Subset to load (e.g., {"subject_id": ["P01"]})

        creates a gazeframe with the following schema:
            subject_id: int
            stimulus: str
            pixel: list[float] (x, y)
            also additionally the follwoing inforamtion is kept:
            timestamp: int
            dataset dependent metadata (e.g. round_id, session_id for GazeBase)

        """
        self.dataset_name = dataset_name
        if root is None:
            self.root = f"data/{dataset_name}"
        else:
            self.root = root
        assert os.path.exists(self.root), f"Path {self.root} does not exist"

        self.context_len = context_len
        self.stride = stride
        self.max_image_size = max_image_size
        self.sampling_step = sampling_step
        self.all_sequences = []
        self.stimuli = stimulus

        _cache_key = hashlib.md5(
            json.dumps(
                {
                    "dataset": dataset_name,
                    "subset": str(sorted(subset.items()) if subset else None),
                    "context_len": context_len,
                    "stride": stride,
                    "sampling_step": sampling_step,
                    "max_image_size": max_image_size,
                    "stimuli": sorted(self.stimuli) if self.stimuli else None,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()[:16]

        _cache_path = Path(cache_dir) / f"{dataset_name}_{_cache_key}.pkl"
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

        if _cache_path.exists():
            print(f"[cache] loading sequences from {_cache_path}")
            with open(_cache_path, "rb") as f:
                self.all_sequences = pickle.load(f)
            print(f"[cache] loaded {len(self.all_sequences)} sequences")
        else:
            # Load pymovements dataset
            dataset_paths = pm.DatasetPaths(root=root)

            self.pm_dataset = pm.Dataset(dataset_name, path=dataset_paths)
            try:
                self.pm_dataset.load(subset=subset)
            except pl.exceptions.NoDataError:
                return

            # add subject id and stimulus to the gaze dataframe for further processing
            # Dataset specific preprocesing
            if dataset_name == "GazeBase":
                # x and y are already in dva have to be back transformed :
                self.pm_dataset.deg2pix()
                self.gazeframes = pl.concat(
                    [
                        gaze.samples.select(["pixel", "time"]).with_columns(
                            # Normalise in-place using this gaze object's screen resolution.
                            # Values outside [0,1] (off-screen gaze) are kept as-is —
                            # they are valid signal (e.g. -0.02 = just off the left edge).
                            pl.concat_list(
                                [
                                    pl.col("pixel")
                                    .list.get(0)
                                    .fill_null(strategy="forward")
                                    / gaze.experiment.screen.width_px,
                                    # Todo have to make this preprocessing transparent
                                    pl.col("pixel")
                                    .list.get(1)
                                    .fill_null(strategy="forward")
                                    / gaze.experiment.screen.height_px,
                                ]
                            ).alias("pixel"),
                            pl.lit(gaze.metadata["subject_id"]).alias("subject_id"),
                            pl.lit(
                                f"R{gaze.metadata['round_id']}S{gaze.metadata['session_id']}"
                            ).alias("stimulus"),
                        )
                        for gaze in self.pm_dataset.gaze
                    ]
                )
                self.gazeframes.head()

            elif dataset_name == "mcfw-gaze":
                self.gazeframes = pl.concat(
                    [
                        gaze.samples.select(["pixel", "time"]).with_columns(
                            pl.concat_list(
                                [
                                    pl.col("pixel").list.get(0),
                                    pl.col("pixel").list.get(1),
                                ]
                            ).alias("pixel"),
                            pl.lit(gaze.metadata["subject_id"])
                            .cast(pl.Utf8)
                            .alias("subject_id"),
                            pl.lit(gaze.metadata["stimulus"])
                            .cast(pl.Utf8)
                            .alias("stimulus"),
                            # pl.lit(gaze.metadata["trial_id"]).alias("trial_id"),
                        )
                        # Todo implement it with the actual yaml from pm, should be smth like:
                        for gaze in self.pm_dataset.gaze
                    ]
                )
            elif dataset_name == "GGTG":
                # split data by stimulus and normalize pixel values
                self.pm_dataset.split_gaze_data(by="stimulus")
                self.gazeframes = pl.concat(
                    [
                        gaze.samples.select(["pixel", "time"]).with_columns(
                            # Normalise in-place using this gaze object's screen resolution.
                            # Values outside [0,1] (off-screen gaze) are kept as-is —
                            # they are valid signal (e.g. -0.02 = just off the left edge).
                            pl.concat_list(
                                [
                                    pl.col("pixel")
                                    .list.get(0)
                                    .fill_null(strategy="forward")
                                    / gaze.experiment.screen.width_px,  # Todo have to make this preprocessing transparent
                                    pl.col("pixel")
                                    .list.get(1)
                                    .fill_null(strategy="forward")
                                    / gaze.experiment.screen.height_px,
                                ]
                            ).alias("pixel"),
                            pl.lit(gaze.metadata["subject_id"])
                            .cast(pl.Utf8)
                            .alias("subject_id"),
                            pl.lit(gaze.metadata["stimulus"])
                            .cast(pl.Utf8)
                            .alias("stimulus"),
                        )
                        for gaze in self.pm_dataset.gaze
                        if self.stimuli is None
                        or gaze.metadata["stimulus"] in self.stimuli
                    ]
                )

            # Concatenate all gaze frames into a single polars dataframe
            else:
                self.gazeframes = pl.concat(
                    [
                        gaze.samples.select(["pixel", "time"]).with_columns(
                            # Normalise in-place using this gaze object's screen resolution.
                            # Values outside [0,1] (off-screen gaze) are kept as-is —
                            # they are valid signal (e.g. -0.02 = just off the left edge).
                            pl.concat_list(
                                [
                                    pl.col("pixel")
                                    .list.get(0)
                                    .fill_null(strategy="forward")
                                    / gaze.experiment.screen.width_px,
                                    pl.col("pixel")
                                    .list.get(1)
                                    .fill_null(strategy="forward")
                                    / gaze.experiment.screen.height_px,
                                ]
                            ).alias("pixel"),
                            pl.lit(gaze.metadata["subject_id"])
                            .cast(pl.Utf8)
                            .alias("subject_id"),
                            pl.lit(gaze.metadata["stimulus"])
                            .cast(pl.Utf8)
                            .alias("stimulus"),
                        )
                        for gaze in self.pm_dataset.gaze
                    ]
                )
            # Convert stimuli to polars
            grouped = self.gazeframes.group_by(["subject_id", "stimulus"])
            self.groups = [
                (key, group) for key, group in grouped
            ]  # materialise once at init
            image_stimuli = self.pm_dataset.fileinfo["ImageStimulus"]

            image_cache = {}
            for (subject_id, stimulus), group in tqdm(
                self.groups, desc="Pre-computing sequences"
            ):
                self.scaling_factor = 1
                gaze_data = group.select("pixel")

                if len(gaze_data) < self.context_len + 1:
                    continue

                stimulus_row = image_stimuli.filter(pl.col("stimulus") == stimulus)
                if stimulus_row.is_empty():
                    continue

                image_path = Path(
                    self.pm_dataset.paths.stimuli / stimulus_row["filepath"][0]
                )
                if image_path not in image_cache:
                    image_cache[image_path] = self._image_transform(image_path)
                transformed_image = image_cache[image_path]

                seqs = list(self._generate_sequences(gaze_data, transformed_image))
                self.all_sequences.extend(seqs)

            print(
                f"Pre-computed {len(self.all_sequences)} sequences, "
                f"{len(image_cache)} unique images cached"
            )

            with open(_cache_path, "wb") as f:
                pickle.dump(self.all_sequences, f)
            print(f"[cache] saved to {_cache_path}")

            print(
                f"Pre-computed {len(self.all_sequences)} sequences, "
                f"{len(image_cache)} unique images cached"
            )

    def __iter__(self) -> Iterator[Dict]:
        """
        Iterate over pre-computed sequences with rank and worker splitting.
        All Polars operations happen at init time in the main process,
        so workers only ever see plain Python lists — no deadlocks.
        """
        sequences = self.all_sequences

        # Split across GPUs (ranks)
        num_replicas = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))
        sequences = sequences[rank::num_replicas]

        # Split across dataloader workers within this rank
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            sequences = sequences[worker_info.id :: worker_info.num_workers]

        yield from iter(sequences)

    def __len__(self) -> int:
        return len(self.all_sequences)

    def _image_transform_coordiantes_preserving(
        self, image_path: Path, screen_width_px: int, screen_height_px: int
    ) -> torch.Tensor:
        """this function could be used in an architecture where the model needs to preserve
        the original coordinates of the gaze data, for example if the model uses a spatial attention mechanism or a
        SSM with a mechanism resembling cross attention implements that
        directly attends to pixel locations in the image. In this case, we need to ensure that the image is padded to
        the original screen resolution, so that the gaze coordinates still correspond to the correct locations
        in the image. The padding is done using edge values, which means that the original image content is preserved
        and not distorted by resizing. This way, the model can learn to attend to the correct regions of the image based
        on the gaze data, without any misalignment caused by resizing."""
        image = decode_image(str(image_path), mode="RGB")
        assert image.shape == (3, screen_height_px, screen_width_px)

        padding_val = [
            0,
            0,
            screen_width_px - image.shape[2],
            screen_height_px - image.shape[1],
        ]
        transform = v2.Compose(
            [
                v2.Pad(padding=padding_val, padding_mode="edge"),
                v2.Resize(size=None, max_size=self.max_image_size),
                MyCustomTransform(padding_mode="edge"),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        return transform(image)

    def _image_transform(self, image_path: Path) -> torch.Tensor:
        """this function is used in the version where a global image embedding is extracted and fed into the model,
        for example in a ViT-based architecture. In this case, we can simply resize the image to the desired max size,
        without worrying about preserving the original coordinates of the gaze data.
        The resizing is done while maintaining the aspect ratio, so that the image content is not distorted.
        This way, the model can learn to extract relevant features from the image based on the gaze data,
        without any misalignment caused by resizing."""

        image = decode_image(str(image_path), mode="RGB")

        transform = v2.Compose(
            [
                v2.Resize(size=None, max_size=self.max_image_size),
                v2.ToDtype(torch.float32, scale=True),
                MyCustomTransform(padding_mode="edge"),
                v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        return transform(image)

    def _generate_sequences(
        self, gaze_data: pl.DataFrame, transformed_image: torch.Tensor
    ) -> Iterator[Dict]:
        # 1. Clean nulls
        gaze_data_cleaned = gaze_data.with_columns(
            pl.col("pixel")
            .fill_null(
                pl.lit([0.0, 0.0])
            )  # Todo have to make this preprocessing transparent (first fill gaze data with polars fill null forward (to get less 0.0 data and then fill the rest with 0.0, i think other wise model is often predicting 0.0
            .list.eval(pl.element().fill_null(0.0))
            .alias("pixel")
        )

        # 2. Downsample AFTER cleaning (was applied to wrong variable before)
        if self.sampling_step > 1:
            gaze_data_cleaned = gaze_data_cleaned.gather_every(self.sampling_step)

        for i in range(0, len(gaze_data_cleaned) - self.context_len, self.stride):
            input_gaze = gaze_data_cleaned[i : i + self.context_len]
            target_gaze = gaze_data_cleaned[i + 1 : i + self.context_len + 1]

            # Build (T, 2) arrays then transpose to (2, T) — no squeeze needed
            input_seq = np.array(
                [
                    [
                        row["pixel"][0] * self.scaling_factor,
                        row["pixel"][1] * self.scaling_factor,
                    ]
                    for row in input_gaze.iter_rows(named=True)
                ],
                dtype=np.float32,
            ).T  # (2, T)

            target_seq = np.array(
                [
                    [
                        row["pixel"][0] * self.scaling_factor,
                        row["pixel"][1] * self.scaling_factor,
                    ]
                    for row in target_gaze.iter_rows(named=True)
                ],
                dtype=np.float32,
            ).T  # (2, T)

            yield {
                "input_seq": torch.from_numpy(input_seq),  # (2, T)
                "target_seq": torch.from_numpy(target_seq),  # (2, T)
                "image": transformed_image,
            }


def create_on_the_fly_loader(
    dataset_name: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    context_len: int = 32,
    stride: int = 1,
    dataset_type: str = "standard",
    max_image_size: int = 224,
    root: Optional[str] = None,
    subset: Optional[Dict] = None,
    sampling_step=1,
    persistent_workers=False,
    prefetch_factor=None,
) -> DataLoader:
    """
    Create a DataLoader with on-the-fly sequence generation.

    Args:
        metadata_path: Path to metadata (for original dataset)
        dataset_name: Name of pymovements dataset (for pymovements dataset)
        batch_size: Batch size
        num_workers: Number of workers for parallel loading
        context_len: Sequence length
        stride: Step between sequences
        dataset_type: "standard", "random_stride", or "adaptive"
        max_image_size: Max size for image resizing
        image_folder_path: Folder for images (original)
        root: Root for pymovements
        subset: Subset for pymovements

    Returns:
        DataLoader ready for training
        :param sampling_step:
    """

    if dataset_name is not None:
        stimuli = None
        if dataset_name == "GGTG":
            try:
                stimuli = subset.get("stimulus", None)
                subset = {"subject_id": subset["subject_id"]}

            except (TypeError, AttributeError, KeyError):
                stimuli = subset.get("stimulus", None)
                subset = None

                print(
                    f"subset loading impaired, loading {dataset_name} by: subset {subset} and stimulus {stimuli} "
                )

        # Guard: if the subset contains no subjects there is nothing to load
        if subset is not None:
            _subjects = subset.get("subject_id", None)
            if isinstance(_subjects, list) and len(_subjects) == 0:
                return None

        # Use pymovements dataset
        if dataset_type == "standard":
            dataset = PymovementsOnTheFlyGazeDataset(
                dataset_name,
                context_len=context_len,
                stride=stride,
                max_image_size=max_image_size,
                root=root,
                subset=subset,
                stimulus=stimuli,
            )
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

    else:
        raise ValueError("dataset_name must be provided")

    # prefetch_factor can only be used with num_workers > 0
    dataloader_kwargs = {
        "batch_size": batch_size,
        "num_workers": 0,
        "pin_memory": False,
        "prefetch_factor": None,
        "persistent_workers": None,
    }

    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = 2
        dataloader_kwargs["num_workers"] = num_workers
        dataloader_kwargs["pin_memory"] = True
        dataloader_kwargs["persistent_workers"] = True
    # print(f"Raw gaze rows: {len(dataset.gazeframes)}")
    # print(dataset.gazeframes.head())

    return DataLoader(dataset, **dataloader_kwargs)


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("On-the-Fly Sequence Generation Example")
    print("=" * 60)

    config_mcfw = {
        "dataset_name": "mcfw-gaze",
        "context_len": 32,
        "batch_size": 128,
        "sampling_step": 1,
        "stride": 100,
        "max_image_size": 224,
        "root": "/home/janhof/thesis/data/",
    }
    config_GGTG = {
        "dataset_name": "GGTG",
        "context_len": 32,
        "batch_size": 128,
        "sampling_step": 100,
        "stride": 100,
        "max_image_size": 224,
        "root": "/home/janhof/thesis/data/",
    }

    config_Gazebase = {
        "dataset_name": "GazeBase",
        "context_len": 32,
        "batch_size": 128,
        "sampling_step": 100,
        "stride": 100,
        "max_image_size": 224,
        "root": "/home/janhof/thesis/data/",
        "subset": {
            "subject_id": [288],
            "round_id": [1],
            "task_name": ["TEX"],
        },
    }

    val_loader = create_on_the_fly_loader(**config_Gazebase)

    # Iterate - sequences generated on-demand
    print("\nIterating over batches (sequences generated on-the-fly):")
    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= 5:  # Show first 5 batches
            break

        print(f"\nBatch {batch_idx}:")
        print(f"  Input shape: {batch['input_seq'].shape}")
        print(f"  Target shape: {batch['target_seq'].shape}")
        print(f"  img shape: {batch['image'].shape}")

        # This is what you'd do in training:
        # output = model(batch['input_seq'])
        # loss = criterion(output, batch['target_seq'])
    test_loader = create_on_the_fly_loader(**config_GGTG)
