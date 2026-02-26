import torch
import torch.nn as nn
from transformers import ViTModel


class GazePredictor(nn.Module):
    def __init__(self, d_model=512, n_layers=4, n_heads=8):
        super().__init__()

        self.vision = ViTModel.from_pretrained("google/vit-base-patch16-224")
        for param in self.vision.parameters():
            param.requires_grad = False

        self.gaze_proj = nn.Linear(3, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True
        )

        self.decoder = nn.TransformerDecoder(decoder_layer, n_layers)

        self.output_head = nn.Linear(d_model, 2)  # predict x,y

    def forward(self, image, gaze_seq):
        # image encoding
        img_feat = self.vision(pixel_values=image).last_hidden_state

        # gaze embedding
        gaze_emb = self.gaze_proj(gaze_seq)

        # autoregressive mask
        T = gaze_emb.size(1)
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()

        decoded = self.decoder(
            tgt=gaze_emb,
            memory=img_feat,
            tgt_mask=mask
        )

        return self.output_head(decoded)