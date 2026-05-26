import torch
import torch.nn as nn
from transformers import ViTModel


class GazePredictor(nn.Module):
    def __init__(self, d_model=224, n_heads=4, n_layers=3):
        super().__init__()

        # Vision encoder
        self.vision = ViTModel.from_pretrained("google/vit-base-patch16-224")
        for p in self.vision.parameters():
            p.requires_grad = False  # freeze encoder

        self.img_proj = nn.Linear(768, d_model)

        self.gaze_proj = nn.Linear(2, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, batch_first=True
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
        mask = torch.triu(torch.ones(T, T, device=gaze_emb.device), diagonal=1).bool()

        decoded = self.decoder(tgt=gaze_emb, memory=img_feat, tgt_mask=mask)

        params = self.output_head(decoded)
        mu = params[..., :2]
        log_sigma = params[..., 2:]
        sigma = torch.exp(log_sigma)

        return mu, sigma
