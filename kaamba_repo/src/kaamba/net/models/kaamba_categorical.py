"""
Categorical Gaze Predictor: Mamba2-based gaze predictor with discrete output.

Architecture mirrors kaamba.py but replaces the Gaussian (mu, sigma) regression
head with independent categorical distributions over discretised x and y axes.
The model predicts the next fixation bin as two independent softmax distributions,
one per axis.

Architecture:
    - Swappable image encoder (ViT or ResNet)
    - Mamba2 sequence backbone
    - Image conditioning either as initial hidden state or fused at every step
    - Dual categorical output head: n_bins logits for x, n_bins logits for y

Training:
    Use cross-entropy on the quantised gaze targets returned by
    ``GazeCategoricalPredictor.quantise(gaze_seq, n_bins)``.
    The model outputs raw logits — apply softmax only for inference / visualisation.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from enum import Enum
from mamba_ssm import Mamba2
from transformers import ViTModel


class ImageEncoderType(str, Enum):
    VIT = "vit"
    RESNET = "resnet"
    SIGLIP = "siglip"


class ConditioningMode(str, Enum):
    INITIAL_STATE = "initial_state"
    EVERY_STEP = "every_step"


# ---------------------------------------------------------------------------
# Image encoders
# ---------------------------------------------------------------------------


class ViTImageEncoder(nn.Module):
    def __init__(
        self,
        model_name="google/vit-base-patch16-224",
        out_dim=256,
        freeze=True,
        verbose=True,
    ):
        super().__init__()
        self.vit = ViTModel.from_pretrained(model_name)
        if freeze:
            for p in self.vit.parameters():
                p.requires_grad = False
        vit_hidden = self.vit.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(vit_hidden, out_dim),
            nn.LayerNorm(out_dim),
        )
        if verbose:
            print(
                f"[ViTImageEncoder] {model_name} ({'frozen' if freeze else 'trainable'}), "
                f"proj {vit_hidden}→{out_dim}"
            )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        cls = self.vit(pixel_values=images).last_hidden_state[:, 0]
        return self.proj(cls)


class ResNetImageEncoder(nn.Module):
    def __init__(self, model_name=None, out_dim=256, freeze=True, verbose=True):
        super().__init__()
        import torchvision.models as models

        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.proj = nn.Sequential(
            nn.Linear(2048, out_dim),
            nn.LayerNorm(out_dim),
        )
        if verbose:
            print(
                f"[ResNetImageEncoder] ResNet50 ({'frozen' if freeze else 'trainable'}), "
                f"proj 2048→{out_dim}"
            )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.proj(self.backbone(images).flatten(1))


class SigLIPImageEncoder(nn.Module):
    """
    SigLIP visual encoder (Google's improved CLIP alternative).

    Uses the vision tower only; the text encoder is discarded.
    Recommended model: 'google/siglip-base-patch16-224'
    Larger option:     'google/siglip-large-patch16-384'

    SigLIP is trained on image-text pairs with sigmoid loss rather than
    softmax (CLIP), giving better calibration on fine-grained tasks.
    The CLS token captures both scene-level and text-region semantics,
    which is well-suited for mixed-stimulus gaze prediction.

    Preprocessing note:
        SigLIP expects images normalised with its own mean/std (not ImageNet).
        Pass raw [0,1] tensors and let the processor handle normalisation,
        OR pre-normalise with:
            mean = [0.5, 0.5, 0.5]
            std  = [0.5, 0.5, 0.5]
        If you pre-normalise in your dataset, set use_processor=False.
    """

    DEFAULT_MODEL = "google/siglip-base-patch16-224"

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        out_dim: int = 256,
        freeze: bool = True,
        use_processor: bool = False,
        verbose: bool = True,
    ):
        super().__init__()
        from transformers import SiglipVisionModel

        self.vision_model = SiglipVisionModel.from_pretrained(model_name)

        if freeze:
            for p in self.vision_model.parameters():
                p.requires_grad = False

            # Optionally unfreeze the final transformer block for fine-tuning
        # (set freeze=True then call encoder.unfreeze_top_k(1) after init)

        hidden = self.vision_model.config.hidden_size  # 768 for base, 1024 for large
        self.proj = nn.Sequential(
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim),
        )

        self.use_processor = use_processor
        if use_processor:
            from transformers import SiglipProcessor

            self.processor = SiglipProcessor.from_pretrained(model_name)

        if verbose:
            n_total = sum(p.numel() for p in self.vision_model.parameters())
            print(
                f"[SigLIPImageEncoder] {model_name} ({'frozen' if freeze else 'trainable'}), "
                f"hidden={hidden}, proj→{out_dim}, total={n_total:,} params"
            )

    def unfreeze_top_k(self, k: int = 1) -> None:
        """Unfreeze the last k transformer encoder layers for fine-tuning."""
        layers = self.vision_model.vision_model.encoder.layers
        for layer in layers[-k:]:
            for p in layer.parameters():
                p.requires_grad = True
        print(f"[SigLIPImageEncoder] Unfroze top {k} encoder layer(s).")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, H, W) float tensor.
                    If use_processor=False, normalise with mean/std=[0.5,0.5,0.5].
        Returns:
            (B, out_dim) image embeddings.
        """
        if self.use_processor:
            # CPU-side preprocessing — avoid in training hot-paths
            inputs = self.processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(next(self.parameters()).device)
        else:
            pixel_values = images

        outputs = self.vision_model(pixel_values=pixel_values)
        # SigLIP pooled_output is the mean-pooled patch tokens (no CLS token in SigLIP)
        pooled = outputs.pooler_output  # (B, hidden)
        return self.proj(pooled)


# ---------------------------------------------------------------------------
# Encoder factory
# ---------------------------------------------------------------------------


def build_image_encoder(
    encoder_type: str,
    out_dim: int,
    freeze: bool = True,
    verbose: bool = True,
    **kwargs,
) -> nn.Module:
    etype = ImageEncoderType(encoder_type)
    if etype == ImageEncoderType.VIT:
        return ViTImageEncoder(
            out_dim=out_dim, freeze=freeze, verbose=verbose, **kwargs
        )
    if etype == ImageEncoderType.RESNET:
        return ResNetImageEncoder(
            out_dim=out_dim, freeze=freeze, verbose=verbose, **kwargs
        )
    if etype == ImageEncoderType.SIGLIP:
        return SigLIPImageEncoder(
            out_dim=out_dim, freeze=freeze, verbose=verbose, **kwargs
        )
    raise ValueError(f"Unknown encoder type: {encoder_type}")


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class GazeCategoricalPredictor(nn.Module):
    """
    Mamba2-based gaze sequence predictor with categorical (discrete) output.

    Instead of predicting a Gaussian (mu, sigma) over continuous coordinates,
    the model predicts the next gaze location as two independent categorical
    distributions — one for the x-axis and one for the y-axis — over a uniform
    grid of ``n_bins`` equally-spaced bins.

    Args:
        d_model:            Mamba2 model dimension.
        n_layers:           Number of Mamba2 layers.
        n_bins:             Number of discrete bins per axis (e.g. 64, 128, 256).
                            Larger values give finer spatial resolution but need
                            more training data to learn well.
        image_encoder_type: ``'vit'`` or ``'resnet'``.
        image_embed_dim:    Dimension of the projected image embedding.
        conditioning_mode:  ``'initial_state'`` or ``'every_step'``.
        freeze_encoder:     Whether to freeze the image encoder weights.
        verbose:            Print model info on initialisation.
        encoder_kwargs:     Extra kwargs forwarded to the image encoder
                            (e.g. ``model_name`` for ViT).

    Input shapes:
        images:     (B, 3, H, W)
        gaze_seq:   (B, 2, T)   — (x, y) coordinates over T timesteps

    Output shapes:
        logits_x:  (B, n_bins, T)  — raw logits for the x-bin distribution
        logits_y:  (B, n_bins, T)  — raw logits for the y-bin distribution

    Training example::

        model = GazeCategoricalPredictor(n_bins=64)

        # Convert continuous gaze to integer bin indices for CE loss
        # gaze_seq_shifted is the *next* step targets, shape (B, 2, T)
        x_labels, y_labels = GazeCategoricalPredictor.quantise(
            gaze_seq_shifted, n_bins=64
        )  # each (B, T), dtype=torch.long

        logits_x, logits_y = model(images, gaze_seq)

        # nn.CrossEntropyLoss expects (B, C, T) logits and (B, T) targets
        loss = F.cross_entropy(logits_x, x_labels) + F.cross_entropy(logits_y, y_labels)

    Inference example::

        logits_x, logits_y = model(images, gaze_seq)
        pred_x, pred_y = GazeCategoricalPredictor.decode(logits_x, logits_y, n_bins=64)
        # pred_x, pred_y: (B, T) continuous coordinates in [0, 1]
    """

    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 4,
        n_bins: int = 64,
        image_encoder_type: ImageEncoderType | str = ImageEncoderType.VIT,
        image_embed_dim: int = 256,
        conditioning_mode: ConditioningMode | str = ConditioningMode.INITIAL_STATE,
        freeze_encoder: bool = True,
        verbose: bool = True,
        **encoder_kwargs,
    ):
        super().__init__()

        self.conditioning_mode = ConditioningMode(conditioning_mode)
        self.d_model = d_model
        self.image_embed_dim = image_embed_dim
        self.n_bins = n_bins

        # --- Image encoder ---
        self.image_encoder = build_image_encoder(
            encoder_type=ImageEncoderType(image_encoder_type),
            out_dim=image_embed_dim,
            freeze=freeze_encoder,
            verbose=verbose,
            **encoder_kwargs,
        )

        # --- Gaze input projection ---
        gaze_input_dim = 2
        if self.conditioning_mode == ConditioningMode.EVERY_STEP:
            self.input_proj = nn.Sequential(
                nn.Linear(gaze_input_dim + image_embed_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
            )
        else:
            self.input_proj = nn.Sequential(
                nn.Linear(gaze_input_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
            )

        # --- initial_state mode: project image embedding → d_model ---
        if self.conditioning_mode == ConditioningMode.INITIAL_STATE:
            self.image_to_state = nn.Sequential(
                nn.Linear(image_embed_dim, d_model),
                nn.LayerNorm(d_model),
                nn.Tanh(),
            )

        # --- Mamba2 layers ---
        self.layers = nn.ModuleList([Mamba2(d_model=d_model) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

        # --- Categorical output heads ---
        # Two separate heads so each axis can learn independent bin distributions.
        self.x_head = nn.Linear(d_model, n_bins)
        self.y_head = nn.Linear(d_model, n_bins)

        if verbose:
            mode_str = self.conditioning_mode.value
            total = sum(p.numel() for p in self.parameters())
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(
                f"[GazeCategoricalPredictor] conditioning={mode_str} | "
                f"d_model={d_model} | layers={n_layers} | n_bins={n_bins}"
            )
            print(
                f"[GazeCategoricalPredictor] params: {total:,} total, {trainable:,} trainable"
            )

    # ------------------------------------------------------------------
    # Forward internals (mirrors kaamba.py)
    # ------------------------------------------------------------------

    def _encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(images)  # (B, image_embed_dim)

    def _prepare_input(
        self,
        gaze_seq: torch.Tensor,
        image_embed: torch.Tensor,
    ) -> torch.Tensor:
        if self.conditioning_mode == ConditioningMode.EVERY_STEP:
            gaze = gaze_seq.permute(0, 2, 1)  # (B, T, 2)
            B, T, _ = gaze.shape
            img = image_embed.unsqueeze(1).expand(B, T, self.image_embed_dim)
            x = torch.cat([gaze, img], dim=-1)  # (B, T, 2 + image_embed_dim)
            return self.input_proj(x)  # (B, T, d_model)
        else:
            gaze = gaze_seq.permute(0, 2, 1)  # (B, 2, T) -> (B, T, 2)
            return self.input_proj(gaze)  # (B, T, d_model)

    def _apply_initial_state(
        self,
        x: torch.Tensor,
        image_embed: torch.Tensor,
    ) -> torch.Tensor:
        state_token = self.image_to_state(image_embed).unsqueeze(1)  # (B, 1, d_model)
        x = torch.cat([state_token, x], dim=1)  # (B, T+1, d_model)
        for layer in self.layers:
            x = layer(x)
        return x[:, 1:, :]  # strip image token → (B, T, d_model)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        images: torch.Tensor,
        gaze_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images:   (B, 3, H, W)
            gaze_seq: (B, 2, T)  — continuous gaze coordinates (input context)

        Returns:
            logits_x: (B, n_bins, T)  raw logits for the x-axis bin
            logits_y: (B, n_bins, T)  raw logits for the y-axis bin
        """
        image_embed = self._encode_image(images)  # (B, image_embed_dim)
        x = self._prepare_input(gaze_seq, image_embed)  # (B, T, d_model)

        if self.conditioning_mode == ConditioningMode.INITIAL_STATE:
            x = self._apply_initial_state(x, image_embed)
        else:
            for layer in self.layers:
                x = layer(x)

        x = self.norm(x)  # (B, T, d_model)

        logits_x = self.x_head(x).permute(0, 2, 1)  # (B, n_bins, T)
        logits_y = self.y_head(x).permute(0, 2, 1)  # (B, n_bins, T)

        return logits_x, logits_y

    # ------------------------------------------------------------------
    # Static helpers for quantisation / decoding
    # ------------------------------------------------------------------

    @staticmethod
    def quantise(
        gaze_seq: torch.Tensor,
        n_bins: int,
        coord_min: float = 0.0,
        coord_max: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert continuous gaze coordinates to integer bin indices.

        Args:
            gaze_seq:  (B, 2, T) or (B, T, 2) — continuous gaze coordinates.
                       Assumes coords in [coord_min, coord_max].
            n_bins:    Number of bins per axis.
            coord_min: Lower bound of the coordinate range (default 0.0).
            coord_max: Upper bound of the coordinate range (default 1.0).

        Returns:
            x_labels: (B, T) LongTensor of x-bin indices in [0, n_bins - 1]
            y_labels: (B, T) LongTensor of y-bin indices in [0, n_bins - 1]
        """
        # Accept both (B, 2, T) and (B, T, 2)
        if gaze_seq.shape[1] == 2 and gaze_seq.ndim == 3:
            gaze_seq = gaze_seq.permute(0, 2, 1)  # → (B, T, 2)

        coords = (gaze_seq - coord_min) / (coord_max - coord_min)  # [0, 1]
        indices = (coords * n_bins).long().clamp(0, n_bins - 1)  # [0, n_bins-1]

        x_labels = indices[..., 0]  # (B, T)
        y_labels = indices[..., 1]  # (B, T)
        return x_labels, y_labels

    @staticmethod
    def decode(
        logits_x: torch.Tensor,
        logits_y: torch.Tensor,
        n_bins: int,
        coord_min: float = 0.0,
        coord_max: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert categorical logits to continuous coordinate predictions via argmax.

        Args:
            logits_x: (B, n_bins, T)
            logits_y: (B, n_bins, T)
            n_bins:   Number of bins used during training.
            coord_min, coord_max: Coordinate range to map bins back into.

        Returns:
            pred_x: (B, T) continuous x coordinates
            pred_y: (B, T) continuous y coordinates
        """
        bin_x = logits_x.argmax(dim=1).float()  # (B, T)
        bin_y = logits_y.argmax(dim=1).float()  # (B, T)

        # Map bin centre back to coordinate space
        scale = (coord_max - coord_min) / n_bins
        pred_x = coord_min + (bin_x + 0.5) * scale
        pred_y = coord_min + (bin_y + 0.5) * scale
        return pred_x, pred_y


# ---------------------------------------------------------------------------
# Convenience constructor
# ---------------------------------------------------------------------------


def build_categorical_gaze_predictor(
    conditioning_mode: str = "initial_state",
    encoder_type: str = "siglip",
    d_model: int = 128,
    n_layers: int = 4,
    n_bins: int = 64,
    image_embed_dim: int = 256,
    freeze_encoder: bool = True,
    verbose: bool = True,
    **encoder_kwargs,
) -> GazeCategoricalPredictor:
    """
    Convenience factory — mirrors ``build_gaze_predictor`` from kaamba.py.

    Args:
        n_bins: Spatial resolution of the categorical distribution per axis.
                64 → 1/64 of the normalised range per bin (~1.6 % of the screen).
                128 → finer; needs more data.

    Example::

        model = build_categorical_gaze_predictor(
            conditioning_mode="every_step",
            encoder_type="resnet",
            d_model=256,
            n_layers=6,
            n_bins=128,
        )
    """
    return GazeCategoricalPredictor(
        d_model=d_model,
        n_layers=n_layers,
        n_bins=n_bins,
        image_encoder_type=encoder_type,
        image_embed_dim=image_embed_dim,
        conditioning_mode=conditioning_mode,
        freeze_encoder=freeze_encoder,
        verbose=verbose,
        **encoder_kwargs,
    )


def main():
    build_categorical_gaze_predictor()


if __name__ == "__main__":
    main()
