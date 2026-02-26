import torch
import numpy as np
import datasets
from datasets import Dataset, load_dataset, load_from_disk

from torchvision.transforms import v2

from kaamba.kaamba_dataset.constants import ROOT_DIR


# https://huggingface.co/docs/datasets/v1.1.1/processing.html#saving-a-processed-dataset-on-disk-and-reload-it
# simulate one image per sample and one gaze sequence



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



def transform_data(example, context_len = 32):
    transforms = v2.Compose([
        # v2.RandomResizedCrop(size=(224, 224), antialias=True),
        v2.PILToTensor(),
        # v2.RandomHorizontalFlip(p=0.5),
        # v2.ToDtype(torch.float32, scale=True),
        # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    gaze = example["data"]
    inputs, targets = [], []

    for i in range(len(gaze) - context_len - 1):
        inputs.append(gaze[i:i+context_len])
        targets.append(gaze[i+1:i+context_len+1])

    return {
        "input_seq": inputs,
        "target_seq": targets,
        "image_path": transforms(example["image"]) #[example["file_name"]] * len(inputs)
    }


def preprocess_data(dataset_path = str(ROOT_DIR) + "/data/MultiplEYE_DE_DE_Goettingen_1_2026/stimuli_MultiplEYE_DE_DE_Goettingen_1_2026/stimuli_images_de_de_1"):
    dataset = load_dataset(dataset_path, streaming=True)
    dataset = dataset.map(transform_data)
    print("Windowing complete")
    for key in dataset.keys():

        num_shards = dataset[key].num_shards
        print(num_shards)
        if num_shards == 1:
            dataset[key].to_parquet(ROOT_DIR / "kaamba_dataset/data.parquet")
            print(f"Data preprocessing {key} complete and saved to disk.")
        else:
            for index in range(num_shards):
                shard = dataset[key].shard(index, num_shards)
                shard.to_parquet(ROOT_DIR /"kaamba_dataset/data-{index:05d}.parquet")

    print("Data preprocessing complete and saved to disk.")


if __name__     == "__main__":
    datasets.logging.set_verbosity_info()
    preprocess_data()