import torch
import numpy as np
import datasets
from datasets import Dataset, load_dataset, load_from_disk
import torch.nn as nn
from transformers import ViTModel
from torch.utils.data import DataLoader


# https://huggingface.co/docs/datasets/v1.1.1/processing.html#saving-a-processed-dataset-on-disk-and-reload-it
# simulate one image per sample and one gaze sequence

def create_windows(example, context_len = 32):
    gaze = example["data"]
    inputs, targets = [], []

    for i in range(len(gaze) - context_len - 1):
        inputs.append(gaze[i:i+context_len])
        targets.append(gaze[i+1:i+context_len+1])

    return {
        "input_seq": inputs,
        "target_seq": targets,
        "image_path": example["image"] #[example["file_name"]] * len(inputs)
    }



class GazePredictor(nn.Module):
    def __init__(self, d_model=256, n_heads=4, n_layers=3):
        super().__init__()

        # Vision encoder
        self.vision = ViTModel.from_pretrained("google/vit-base-patch16-224")
        for p in self.vision.parameters():
            p.requires_grad = False  # freeze encoder

        self.img_proj = nn.Linear(768, d_model)

        self.gaze_proj = nn.Linear(3, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True
        )

        self.decoder = nn.TransformerDecoder(decoder_layer, n_layers)

        # Gaussian output: μx, μy, logσx, logσy
        self.output_head = nn.Linear(d_model, 4)

    def forward(self, image, gaze_seq):
        with torch.no_grad():
            img_feat = self.vision(pixel_values=image).last_hidden_state[:, 0]

        img_feat = self.img_proj(img_feat).unsqueeze(1)

        gaze_emb = self.gaze_proj(gaze_seq)

        T = gaze_emb.size(1)
        mask = torch.triu(
            torch.ones(T, T, device=gaze_emb.device),
            diagonal=1
        ).bool()

        decoded = self.decoder(
            tgt=gaze_emb,
            memory=img_feat,
            tgt_mask=mask
        )

        params = self.output_head(decoded)
        mu = params[..., :2]
        log_sigma = params[..., 2:]
        sigma = torch.exp(log_sigma)

        return mu, sigma

def gaussian_nll(mu, sigma, target):
    dist = torch.distributions.Normal(mu, sigma)
    return -dist.log_prob(target[..., :2]).sum(-1).mean()


def preprocess_data(dataset_path = "data/MultiplEYE_DE_DE_Goettingen_1_2026/stimuli_MultiplEYE_DE_DE_Goettingen_1_2026/stimuli_images_de_de_1", output_path = "data/processed_gaze_dataset"):
    dataset = load_dataset(dataset_path)
    dataset = dataset.map(create_windows)
    print("Windowing complete. Flattening dataset...")
    dataset = dataset.flatten()
    print("Flattening complete. Preprocessing dataset...")
    dataset.save_to_disk(output_path)
    print("Data preprocessing complete and saved to disk.")


def train(dataset_path = "data/processed_gaze_dataset"):
    try:
        dataset = load_from_disk(dataset_path)
    except FileNotFoundError:
        print("Processed dataset not found. Preprocessing raw data...")
        preprocess_data()
        dataset = load_from_disk(dataset_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GazePredictor().to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4
    )

    loader = DataLoader(dataset["train"], batch_size=8, shuffle=True)

    model.train()

    for epoch in range(3):
        total_loss = 0

        for batch in loader:
            images = batch["image"].to(device)
            inputs = batch["input_seq"].to(device)
            targets = batch["target_seq"].to(device)

            optimizer.zero_grad()

            mu, sigma = model(images, inputs)
            loss = gaussian_nll(mu, sigma, targets)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch} Loss: {total_loss/len(loader):.4f}")

if __name__     == "__main__":
    datasets.logging.set_verbosity_info()
    train()