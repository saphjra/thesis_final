from datasets import load_from_disk
from torch.utils.data import DataLoader
from net.models.tamba import GazePredictor
from utils.loss_functions import gaussian_nll
from utils.data_process import preprocess_data
import torch

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

    loader = DataLoader(dataset["train"], batch_size=1)

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
