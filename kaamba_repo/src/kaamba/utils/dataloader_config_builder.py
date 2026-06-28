# File: kaamba/utils/dataloader_config_builder.py

"""
Automatic DataLoader Configuration Builder

This utility creates train/val/test dataloaders with flexible splitting strategies.
Supports splitting by: stimulus, participant (subject_id), trial, or random.

Usage:
    builder = DataloaderConfigBuilder(
        datasets=["mcfw-gaze"],
        root="/path/to/data/",
        context_len=100,
        stride=4,
        sampling_step=11,
        max_image_size=224,
    )

    train_loader, val_loader, test_loader, configs = builder.create_loaders(
        split_strategy="participant",  # or "stimulus", "trial", "random"
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        batch_size=512,
        num_workers=2,
    )
"""

from typing import Optional, List, Dict, Tuple, Literal, Any
from dataclasses import dataclass, asdict
import numpy as np
import pymovements as pm
from torch.utils.data import DataLoader
from kaamba.utils.on_the_fly_dataset import create_on_the_fly_loader
# Add to dataloader_config_builder.py or a new utils/combined_loader.py


class InterleavedLoader:
    """
    Interleaves batches from multiple DataLoaders in round-robin order.

    Each __iter__ cycle exhausts the longest loader, cycling shorter
    ones so every batch always contains real data.

    Works as a drop-in replacement for a single DataLoader in the
    training loop — supports len() and iteration.

    Example:
        loader_a = DataLoader(dataset_a, batch_size=32)  # 100 batches
        loader_b = DataLoader(dataset_b, batch_size=32)  #  60 batches
        combined = InterleavedLoader([loader_a, loader_b])
        # → 200 batches total, alternating A, B, A, B, ...
        # loader_b cycles back to start after its 60th batch
    """

    def __init__(self, loaders: list, lengths: list[int] | None = None):
        assert len(loaders) > 0
        self.loaders = loaders
        self._lengths = lengths  # batch counts, not sequence counts

    def _get_length(self, loader) -> int:
        try:
            return len(loader)
        except TypeError:
            return sum(1 for _ in loader)

    def __len__(self) -> int:
        if self._lengths is not None:
            return sum(self._lengths)
        return sum(self._get_length(loader) for loader in self.loaders)

    def __iter__(self):
        lengths = self._lengths or [self._get_length(loader) for loader in self.loaders]
        max_len = max(lengths)
        iters = [iter(loader) for loader in self.loaders]
        for _ in range(max_len):
            for i, it in enumerate(iters):
                try:
                    yield next(it)
                except StopIteration:
                    iters[i] = iter(self.loaders[i])
                    yield next(iters[i])

    def to(self, device):
        """No-op: tensors are moved in the training loop."""
        return self


@dataclass
class DataloaderConfig:
    """Configuration for a single loader"""

    dataset_name: str
    subset: Dict
    batch_size: int
    num_workers: int
    context_len: int
    stride: int
    sampling_step: int
    max_image_size: int
    purpose: str  # "train", "val", or "test"

    def to_dict(self) -> dict:
        """Convert to dict for passing to create_on_the_fly_loader"""
        return asdict(self)


class DataloaderConfigBuilder:
    """
    Builds train/val/test dataloaders with flexible splitting strategies.

    Supports splitting by:
    - "participant": Split by subject_id
    - "stimulus": Split by stimulus
    - "trial": Split by trial_id
    - "random": Random split (respects dataset constraints)
    """

    def __init__(
        self,
        datasets: List[str],
        root: str,
        context_len: int = 100,
        stride: int = 4,
        sampling_step: int = 11,
        max_image_size: int = 224,
    ) -> None:
        """
        Initialize builder with common parameters.

        Args:
            datasets: List of dataset names (e.g., ["mcfw-gaze", "GGTG"])
            root: Root directory for data
            context_len: Sequence length (consistent across all loaders)
            stride: Step between sequences (consistent across all loaders)
            sampling_step: Downsampling step (consistent across all loaders)
            max_image_size: Image size (consistent across all loaders)
        """
        self.datasets = datasets
        self.root = root
        self.context_len = context_len
        self.stride = stride
        self.sampling_step = sampling_step
        self.max_image_size = max_image_size

        # Cache dataset info to avoid re-scanning
        self.dataset_info = {}
        for dataset_name in datasets:
            self.dataset_info[dataset_name] = self._scan_dataset(dataset_name)

    def _scan_dataset(self, dataset_name: str) -> Dict:
        """Scan dataset to get available subjects, stimuli, trial_ids"""
        dataset_paths = pm.DatasetPaths(root=self.root)
        pm_dataset = pm.Dataset(dataset_name, path=dataset_paths)
        pm_dataset.scan()

        fileinfo = pm_dataset.fileinfo["gaze"]

        subjects = sorted(fileinfo["subject_id"].unique().to_list())
        stimuli = sorted(
            pm_dataset.fileinfo["ImageStimulus"]["stimulus"].unique().to_list()
        )

        try:
            trial_ids = sorted(fileinfo["trial_id"].unique().to_list())
        except Exception:
            trial_ids = []

        print(f"\n📊 Dataset: {dataset_name}")
        print(f"   Subjects:  {len(subjects)} — {subjects}")
        print(f"   Stimuli:   {len(stimuli)} — {stimuli}")
        if trial_ids:
            print(f"   Trials:    {len(trial_ids)} — {trial_ids}")

        return {
            "dataset_name": dataset_name,
            "subjects": subjects,
            "stimuli": stimuli,
            "trial_ids": trial_ids,
        }

    # Updated DataloaderConfigBuilder with exclusion support

    def create_loaders(
        self,
        split_strategy: Literal[
            "participant", "stimulus", "trial", "random"
        ] = "participant",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        batch_size: int = 512,
        num_workers: int = 2,
        seed: int = 42,
        exclude_participants: Optional[List] = None,
        exclude_stimuli: Optional[List] = None,
        exclude_trials: Optional[List] = None,
    ) -> (
        tuple[DataLoader, DataLoader, DataLoader, dict[str, DataloaderConfig]]
        | tuple[InterleavedLoader, InterleavedLoader, InterleavedLoader, dict[Any, Any]]
    ):
        """
        Create train/val/test dataloaders with specified split strategy.

        Args:
            split_strategy: How to split data
                - "participant": Split by subject_id
                - "stimulus": Split by stimulus
                - "trial": Split by trial_id
                - "random": Random split
            train_ratio: Fraction for training (default 0.7)
            val_ratio: Fraction for validation (default 0.15)
            test_ratio: Fraction for testing (default 0.15)
            batch_size: Batch size for all loaders
            num_workers: Workers for all loaders
            seed: Random seed for reproducibility
            exclude_participants: Participants to completely exclude (e.g., ["P01", "P02"])
            exclude_stimuli: Stimuli to completely exclude (e.g., ["1", "2"])
            exclude_trials: Trial IDs to completely exclude (e.g., ["1", "2"])

        Returns:
            train_loader, val_loader, test_loader, configs_dict

        Example:
            train_loader, val_loader, test_loader, configs = builder.create_loaders(
                split_strategy="participant",
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15,
                exclude_participants=["P01", "P02"],  # Exclude these participants
                exclude_stimuli=["60", "61"],         # Exclude these stimuli
            )
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
            f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
        )

        np.random.seed(seed)
        if len(self.datasets) == 1:
            return self._create_single_dataset_loaders(
                split_strategy,
                train_ratio,
                val_ratio,
                test_ratio,
                batch_size,
                num_workers,
                seed,
                exclude_participants,
                exclude_stimuli,
                exclude_trials,
            )
        else:
            return self._create_multi_dataset_loaders(
                split_strategy,
                train_ratio,
                val_ratio,
                test_ratio,
                batch_size,
                num_workers,
                seed,
                exclude_participants,
                exclude_stimuli,
                exclude_trials,
            )
        if len(self.datasets) > 1:
            raise NotImplementedError(
                "Multi-dataset splitting not yet implemented. Use one dataset at a time."
            )

    def _create_multi_dataset_loaders(
        self,
        split_strategy,
        train_ratio,
        val_ratio,
        test_ratio,
        batch_size,
        num_workers,
        seed,
        exclude_participants,
        exclude_stimuli,
        exclude_trials,
    ):
        """
        Split each dataset independently, then combine with CombinedDataLoader.
        Each dataset gets its own builder so splits are fully independent.
        """

        all_train_loaders = []
        all_val_loaders = []
        all_test_loaders = []
        all_configs = {}

        for dataset_name in self.datasets:
            print(f"\n{'=' * 60}")
            print(f"Processing dataset: {dataset_name}")
            print(f"{'=' * 60}")
            if dataset_name == "GGTG":
                sampling_step = 8
            else:
                sampling_step = self.sampling_step
            # Build a single-dataset builder for each dataset
            single_builder = DataloaderConfigBuilder(
                datasets=[dataset_name],
                root=self.root,
                context_len=self.context_len,
                stride=self.stride,
                sampling_step=sampling_step,
                max_image_size=self.max_image_size,
            )

            train_loader, val_loader, test_loader, configs = (
                single_builder.create_loaders(
                    split_strategy=split_strategy,
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                    test_ratio=test_ratio,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    seed=seed,
                    exclude_participants=exclude_participants,
                    exclude_stimuli=exclude_stimuli,
                    exclude_trials=exclude_trials,
                )
            )

            all_train_loaders.append(train_loader)
            all_val_loaders.append(val_loader)
            all_test_loaders.append(test_loader)
            all_configs[dataset_name] = configs

        # Combine loaders by interleaving batches
        combined_train = InterleavedLoader(all_train_loaders)
        combined_val = InterleavedLoader(all_val_loaders)
        combined_test = InterleavedLoader(all_test_loaders)

        return combined_train, combined_val, combined_test, all_configs

    def _create_single_dataset_loaders(
        self,
        split_strategy,
        train_ratio,
        val_ratio,
        test_ratio,
        batch_size,
        num_workers,
        seed,
        exclude_participants,
        exclude_stimuli,
        exclude_trials,
    ):

        dataset_name = self.datasets[0]
        info = self.dataset_info[dataset_name].copy()  # Make a copy to modify

        # Apply exclusions
        print("\n🚫 Applying exclusions...")

        exclude_participants = exclude_participants or []
        exclude_stimuli = exclude_stimuli or []
        exclude_trials = exclude_trials or []

        if exclude_participants:
            original_count = len(info["subjects"])
            info["subjects"] = [
                s for s in info["subjects"] if s not in exclude_participants
            ]
            excluded_count = original_count - len(info["subjects"])
            print(f"   Excluded {excluded_count} participants: {exclude_participants}")
            print(
                f"   Remaining participants ({len(info['subjects'])}): {info['subjects']}"
            )

        if exclude_stimuli:
            original_count = len(info["stimuli"])
            info["stimuli"] = [s for s in info["stimuli"] if s not in exclude_stimuli]
            excluded_count = original_count - len(info["stimuli"])
            print(f"   Excluded {excluded_count} stimuli: {exclude_stimuli}")
            print(f"   Remaining stimuli ({len(info['stimuli'])}): {info['stimuli']}")

        if exclude_trials:
            if not info["trial_ids"]:
                print("   ⚠️  Warning: No trial_ids to exclude!")
            else:
                original_count = len(info["trial_ids"])
                info["trial_ids"] = [
                    t for t in info["trial_ids"] if t not in exclude_trials
                ]
                excluded_count = original_count - len(info["trial_ids"])
                print(f"   Excluded {excluded_count} trials: {exclude_trials}")
                print(
                    f"   Remaining trials ({len(info['trial_ids'])}): {info['trial_ids']}"
                )

        print(f"\n🔀 Creating splits using strategy: {split_strategy}")
        print(
            f"   Train: {train_ratio * 100:.0f}% | Val: {val_ratio * 100:.0f}% | Test: {test_ratio * 100:.0f}%"
        )

        # Get split entities from filtered info
        train_subset, val_subset, test_subset = self._split_dataset(
            info, split_strategy, train_ratio, val_ratio, test_ratio
        )

        print(f"\n✓ Train subset: {train_subset}")
        print(f"✓ Val subset:   {val_subset}")
        print(f"✓ Test subset:  {test_subset}")

        # Create configs
        configs = {
            "train": DataloaderConfig(
                dataset_name=dataset_name,
                subset=train_subset,
                batch_size=batch_size,
                num_workers=num_workers,
                context_len=self.context_len,
                stride=self.stride,
                sampling_step=self.sampling_step,
                max_image_size=self.max_image_size,
                purpose="train",
            ),
            "val": DataloaderConfig(
                dataset_name=dataset_name,
                subset=val_subset,
                batch_size=batch_size,
                num_workers=num_workers,
                context_len=self.context_len,
                stride=self.stride,
                sampling_step=self.sampling_step,
                max_image_size=self.max_image_size,
                purpose="val",
            ),
            "test": DataloaderConfig(
                dataset_name=dataset_name,
                subset=test_subset,
                batch_size=batch_size,
                num_workers=num_workers,
                context_len=self.context_len,
                stride=self.stride,
                sampling_step=self.sampling_step,
                max_image_size=self.max_image_size,
                purpose="test",
            ),
        }

        # Create loaders
        print("\n📥 Creating dataloaders...")
        train_loader = create_on_the_fly_loader(
            dataset_name=dataset_name,
            root=self.root,
            subset=train_subset,
            batch_size=batch_size,
            num_workers=num_workers,
            context_len=self.context_len,
            stride=self.stride,
            sampling_step=self.sampling_step,
            max_image_size=self.max_image_size,
        )

        val_loader = create_on_the_fly_loader(
            dataset_name=dataset_name,
            root=self.root,
            subset=val_subset,
            batch_size=batch_size,
            num_workers=0,  # No shuffling for val
            context_len=self.context_len,
            stride=self.stride,
            sampling_step=self.sampling_step,
            max_image_size=self.max_image_size,
        )

        test_loader = create_on_the_fly_loader(
            dataset_name=dataset_name,
            root=self.root,
            subset=test_subset,
            batch_size=batch_size,
            num_workers=0,  # No shuffling for test
            context_len=self.context_len,
            stride=self.stride,
            sampling_step=self.sampling_step,
            max_image_size=self.max_image_size,
        )

        print("\n✅ Dataloaders created successfully!")

        return train_loader, val_loader, test_loader, configs

    def _split_dataset(
        self,
        info: Dict,
        strategy: str,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
    ) -> Tuple[Dict, Dict, Dict]:
        """Split dataset according to strategy and explicitly include stimuli/trials"""

        if strategy == "participant":
            return self._split_by_participant(info, train_ratio, val_ratio, test_ratio)
        elif strategy == "stimulus":
            return self._split_by_stimulus(info, train_ratio, val_ratio, test_ratio)
        elif strategy == "trial":
            return self._split_by_trial(info, train_ratio, val_ratio, test_ratio)
        elif strategy == "random":
            return self._split_random(info, train_ratio, val_ratio, test_ratio)
        else:
            raise ValueError(f"Unknown split strategy: {strategy}")

    def _split_by_participant(
        self, info: Dict, train_ratio: float, val_ratio: float, test_ratio: float
    ) -> Tuple[Dict, Dict, Dict]:
        """Split by subject_id, explicitly include all stimuli and trials"""
        subjects = info["subjects"]
        n_train = max(1, int(len(subjects) * train_ratio))
        n_val = max(1, int(len(subjects) * val_ratio))

        train_subjects = subjects[:n_train]
        val_subjects = subjects[n_train : n_train + n_val]
        test_subjects = subjects[n_train + n_val :]

        # Build subsets with explicit stimulus and trial inclusion
        def _make_subset(participant_ids):
            subset = {"subject_id": participant_ids}
            # Explicitly include all remaining stimuli
            if info["stimuli"]:
                subset["stimulus"] = info["stimuli"]
            # Explicitly include all remaining trials
            if info["trial_ids"]:
                subset["trial_id"] = info["trial_ids"]
            return subset

        return (
            _make_subset(train_subjects),
            _make_subset(val_subjects),
            _make_subset(test_subjects),
        )

    def _split_by_stimulus(
        self, info: Dict, train_ratio: float, val_ratio: float, test_ratio: float
    ) -> Tuple[Dict, Dict, Dict]:
        """Split by stimulus, explicitly include all participants and trials"""

        stimuli = info["stimuli"]
        n_train = max(1, int(len(stimuli) * train_ratio))
        n_val = max(1, int(len(stimuli) * val_ratio))

        train_stimuli = stimuli[:n_train]
        val_stimuli = stimuli[n_train : n_train + n_val]
        test_stimuli = stimuli[n_train + n_val :]

        # Build subsets with explicit participant and trial inclusion
        def _make_subset(stimulus_ids):
            subset = {"stimulus": stimulus_ids}
            # Explicitly include all remaining participants
            if info["subjects"]:
                subset["subject_id"] = info["subjects"]
            # Explicitly include all remaining trials
            if info["trial_ids"]:
                subset["trial_id"] = info["trial_ids"]
            return subset

        return (
            _make_subset(train_stimuli),
            _make_subset(val_stimuli),
            _make_subset(test_stimuli),
        )

    def _split_by_trial(
        self, info: Dict, train_ratio: float, val_ratio: float, test_ratio: float
    ) -> Tuple[Dict, Dict, Dict]:
        """Split by trial_id, explicitly include all participants and stimuli"""
        if not info["trial_ids"]:
            raise ValueError("Dataset has no trial_ids!")

        trials = info["trial_ids"]
        n_train = max(1, int(len(trials) * train_ratio))
        n_val = max(1, int(len(trials) * val_ratio))

        train_trials = trials[:n_train]
        val_trials = trials[n_train : n_train + n_val]
        test_trials = trials[n_train + n_val :]

        # Build subsets with explicit participant and stimulus inclusion
        def _make_subset(trial_ids):
            subset = {"trial_id": trial_ids}
            # Explicitly include all remaining participants
            if info["subjects"]:
                subset["subject_id"] = info["subjects"]
            # Explicitly include all remaining stimuli
            if info["stimuli"]:
                subset["stimulus"] = info["stimuli"]
            return subset

        return (
            _make_subset(train_trials),
            _make_subset(val_trials),
            _make_subset(test_trials),
        )

    def _split_random(
        self, info: Dict, train_ratio: float, val_ratio: float, test_ratio: float
    ) -> Tuple[Dict, Dict, Dict]:
        """Random split by subject_id with shuffling, explicitly include all stimuli and trials"""
        subjects = info["subjects"].copy()
        np.random.shuffle(subjects)

        n_train = max(1, int(len(subjects) * train_ratio))
        n_val = max(1, int(len(subjects) * val_ratio))

        train_subjects = sorted(subjects[:n_train])
        val_subjects = sorted(subjects[n_train : n_train + n_val])
        test_subjects = sorted(subjects[n_train + n_val :])

        # Build subsets with explicit stimulus and trial inclusion
        def _make_subset(participant_ids):
            subset = {"subject_id": participant_ids}
            # Explicitly include all remaining stimuli
            if info["stimuli"]:
                subset["stimulus"] = info["stimuli"]
            # Explicitly include all remaining trials
            if info["trial_ids"]:
                subset["trial_id"] = info["trial_ids"]
            return subset

        return (
            _make_subset(train_subjects),
            _make_subset(val_subjects),
            _make_subset(test_subjects),
        )

    def get_configs_dict(
        self,
        train_config: DataloaderConfig,
        val_config: DataloaderConfig,
        test_config: DataloaderConfig,
    ) -> Dict:
        """Convert configs to dict format for saving/logging"""
        return {
            "train": asdict(train_config),
            "val": asdict(val_config),
            "test": asdict(test_config),
        }


# Example usage
if __name__ == "__main__":
    # Create builder
    builder = DataloaderConfigBuilder(
        datasets=["GGTG", "mcfw-gaze"],
        root="/home/janhof/thesis/data",
        context_len=100,
        stride=4,
        sampling_step=11,
        max_image_size=224,
    )

    # Create loaders with participant-based split
    train_loader, val_loader, test_loader, configs = builder.create_loaders(
        split_strategy="participant",  # Options: "participant", "stimulus", "trial", "random"
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        batch_size=512,
        num_workers=2,
        exclude_participants=["001", "002"],  # Exclude these participants entirely
        exclude_stimuli=["60", "61"],
        exclude_trials=["", "5", "4"],
    )

    print("\n" + "=" * 70)
    print("Example batch from train loader:")
    print("=" * 70)
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= 1:
            break
        print(f"Input shape:  {batch['input_seq'].shape}")
        print(f"Target shape: {batch['target_seq'].shape}")
        print(f"Image shape:  {batch['image'].shape}")

    # You can also access configs

    print(configs)
