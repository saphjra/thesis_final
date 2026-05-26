"""
Tamba: Mamba2-based gaze predictor with modular image conditioning.

Architecture:
    - Swappable image encoder (ViT or ResNet)
    - Mamba2 sequence backbone
    - Image conditioning either as initial hidden state or fused at every step
    - Gaussian (mu, sigma) output head for NLL training
"""

from __future__ import annotations

from enum import Enum
from typing import Tuple

import torch
import torch.nn as nn
from mamba_ssm import Mamba2
from transformers import ViTModel

# ---------------------------------------------------------------------------
# Enums for config
# ---------------------------------------------------------------------------


class ImageEncoderType(str, Enum):
    VIT = "vit"
    RESNET = "resnet"


class ConditioningMode(str, Enum):
    INITIAL_STATE = "initial_state"  # image → first hidden state
    EVERY_STEP = "every_step"  # image embedding concatenated to every input step


# ---------------------------------------------------------------------------
# Image Encoders
# ---------------------------------------------------------------------------


class ViTImageEncoder(nn.Module):
    """
    Wraps HuggingFace ViT, projects CLS token to `out_dim`.
    """

    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        out_dim: int = 256,
        freeze: bool = True,
        verbose: bool = True,
    ):
        super().__init__()
        self.vit = ViTModel.from_pretrained(model_name)

        if freeze:
            for param in self.vit.parameters():
                param.requires_grad = False

        vit_hidden = self.vit.config.hidden_size  # 768 for base
        self.proj = nn.Sequential(
            nn.Linear(vit_hidden, out_dim),
            nn.LayerNorm(out_dim),
        )

        if verbose:
            frozen = "frozen" if freeze else "trainable"
            print(
                f"[ViTImageEncoder] {model_name} ({frozen}), proj {vit_hidden}→{out_dim}"
            )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, H, W)
        Returns:
            embedding: (B, out_dim)
        """
        outputs = self.vit(pixel_values=images)
        cls_token = outputs.last_hidden_state[:, 0]  # (B, vit_hidden)
        return self.proj(cls_token)  # (B, out_dim)


class ResNetImageEncoder(nn.Module):
    """
    ResNet50 backbone (torchvision), global average pool → projection.
    """

    def __init__(
        self,
        out_dim: int = 256,
        freeze: bool = True,
        verbose: bool = True,
    ):
        super().__init__()
        import torchvision.models as models

        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # Strip the classification head
        self.backbone = nn.Sequential(
            *list(backbone.children())[:-1]
        )  # → (B, 2048, 1, 1)
        resnet_dim = 2048

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.proj = nn.Sequential(
            nn.Linear(resnet_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

        if verbose:
            frozen = "frozen" if freeze else "trainable"
            print(
                f"[ResNetImageEncoder] ResNet50 ({frozen}), proj {resnet_dim}→{out_dim}"
            )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, H, W)
        Returns:
            embedding: (B, out_dim)
        """
        features = self.backbone(images)  # (B, 2048, 1, 1)
        features = features.flatten(1)  # (B, 2048)
        return self.proj(features)  # (B, out_dim)


def build_image_encoder(
    encoder_type: ImageEncoderType,
    out_dim: int,
    freeze: bool = True,
    verbose: bool = True,
    **kwargs,
) -> nn.Module:
    if encoder_type == ImageEncoderType.VIT:
        return ViTImageEncoder(
            out_dim=out_dim, freeze=freeze, verbose=verbose, **kwargs
        )
    elif encoder_type == ImageEncoderType.RESNET:
        return ResNetImageEncoder(
            out_dim=out_dim, freeze=freeze, verbose=verbose, **kwargs
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


# ---------------------------------------------------------------------------
# Gaze input projection
# ---------------------------------------------------------------------------


class GazeInputProjection(nn.Module):
    """
    Projects raw gaze coordinates (x, y) to model dimension.
    Input shape: (B, 2, T)  →  (B, T, d_model)
    """

    def __init__(self, gaze_dim: int = 2, d_model: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(gaze_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, T) → (B, T, 2)

        x = x.permute(0, 2, 1)

        return self.proj(x)  # (B, T, d_model)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class GazePredictor(nn.Module):
    """
    Mamba2-based gaze sequence predictor with modular image conditioning.

    Args:
        d_model:            Mamba2 model dimension
        n_layers:           Number of Mamba2 layers
        image_encoder_type: 'vit' or 'resnet'
        image_embed_dim:    Dimension of projected image embedding
        conditioning_mode:  'initial_state' or 'every_step'
        freeze_encoder:     Whether to freeze the image encoder weights
        verbose:            Print model info on init
        encoder_kwargs:     Extra kwargs forwarded to the image encoder
                            (e.g. model_name for ViT)

    Input shapes:
        images:     (B, 3, H, W)
        gaze_seq:   (B, 2, T)   — (x, y) coordinates over T timesteps

    Output shapes:
        mu:         (B, 2, T)
        sigma:      (B, 2, T)   — strictly positive (softplus activated)
    """

    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 4,
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

        # --- Image encoder ---
        self.image_encoder = build_image_encoder(
            encoder_type=ImageEncoderType(image_encoder_type),
            out_dim=image_embed_dim,
            freeze=freeze_encoder,
            verbose=verbose,
            **encoder_kwargs,
        )

        # --- Gaze input projection ---
        # In every_step mode, gaze (2) + image embedding are concatenated before projection
        gaze_input_dim = 2
        if self.conditioning_mode == ConditioningMode.EVERY_STEP:
            # project (gaze || image_embed) → d_model
            self.input_proj = nn.Sequential(
                nn.Linear(gaze_input_dim + image_embed_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
            )
        else:
            self.input_proj = GazeInputProjection(
                gaze_dim=gaze_input_dim, d_model=d_model
            )

        # --- initial_state mode: project image embedding → d_model ---
        if self.conditioning_mode == ConditioningMode.INITIAL_STATE:
            self.image_to_state = nn.Sequential(
                nn.Linear(image_embed_dim, d_model),
                nn.LayerNorm(d_model),
                nn.Tanh(),  # bound the initial state
            )

        # --- Mamba2 layers ---
        self.layers = nn.ModuleList([Mamba2(d_model=d_model) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

        # --- Output head: predict mu and log_sigma ---
        self.output_head = nn.Linear(d_model, 4)  # 4 = (mu_x, mu_y, sigma_x, sigma_y)

        if verbose:
            mode_str = self.conditioning_mode.value
            total = sum(p.numel() for p in self.parameters())
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(
                f"[GazePredictor] conditioning={mode_str} | "
                f"d_model={d_model} | layers={n_layers}"
            )
            print(f"[GazePredictor] params: {total:,} total, {trainable:,} trainable")

    def _encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Returns image embedding (B, image_embed_dim)."""
        return self.image_encoder(images)

    def _prepare_input(
        self,
        gaze_seq: torch.Tensor,
        image_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Prepares the sequence input to Mamba2.

        initial_state mode: gaze_seq projected normally → (B, T, d_model)
        every_step mode:    image_embed broadcast + concat → (B, T, d_model)
        """
        if self.conditioning_mode == ConditioningMode.EVERY_STEP:
            # gaze_seq: (B, 2, T) → (B, T, 2)
            gaze = gaze_seq.permute(0, 2, 1)
            B, T, _ = gaze.shape
            # broadcast image embedding across time: (B, image_embed_dim) → (B, T, image_embed_dim)
            img = image_embed.unsqueeze(1).expand(B, T, self.image_embed_dim)
            # concatenate and project
            x = torch.cat([gaze, img], dim=-1)  # (B, T, 2 + image_embed_dim)
            return self.input_proj(x)  # (B, T, d_model)
        else:
            return self.input_proj(gaze_seq)  # (B, T, d_model)

    def _apply_initial_state(
        self,
        x: torch.Tensor,
        image_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Injects image embedding as an additional token prepended to the sequence.
        The token is removed after processing so output length matches input length.

        x:            (B, T, d_model)
        image_embed:  (B, image_embed_dim)
        returns:      (B, T, d_model)
        """
        state_token = self.image_to_state(image_embed)  # (B, d_model)
        state_token = state_token.unsqueeze(1)  # (B, 1, d_model)
        x = torch.cat([state_token, x], dim=1)  # (B, T+1, d_model)

        for layer in self.layers:
            x = layer(x)

        return x[:, 1:, :]  # strip the image token → (B, T, d_model)

    def forward(
        self,
        images: torch.Tensor,
        gaze_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images:   (B, 3, H, W)
            gaze_seq: (B, 2, T)

        Returns:
            mu:    (B, 2, T)
            sigma: (B, 2, T)  strictly positive
        """
        # 1. Encode image
        image_embed = self._encode_image(images)  # (B, image_embed_dim)

        # 2. Prepare sequence input
        x = self._prepare_input(gaze_seq, image_embed)  # (B, T, d_model)

        # 3. Run through Mamba2 layers
        if self.conditioning_mode == ConditioningMode.INITIAL_STATE:
            x = self._apply_initial_state(x, image_embed)  # handles layers internally
        else:
            for layer in self.layers:
                x = layer(x)

        x = self.norm(x)  # (B, T, d_model)

        # 4. Output head
        out = self.output_head(x)  # (B, T, 4)
        out = out.permute(0, 2, 1)  # (B, 4, T)

        mu = out[:, :2, :]  # (B, 2, T)
        sigma = (
            torch.nn.functional.softplus(out[:, 2:, :]) + 1e-6
        )  # (B, 2, T), strictly > 0 #ToDo its not actually taking the predicted sigma values check !!!

        return mu, sigma


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------


def build_gaze_predictor(
    conditioning_mode: str = "initial_state",
    encoder_type: str = "vit",
    d_model: int = 128,
    n_layers: int = 4,
    image_embed_dim: int = 256,
    freeze_encoder: bool = True,
    verbose: bool = True,
    **encoder_kwargs,
) -> GazePredictor:
    """
    Convenience factory so callers don't need to import the enums.

    Example:
        model = build_gaze_predictor(
            conditioning_mode="every_step",
            encoder_type="resnet",
            d_model=256,
            n_layers=6,
        )
    """
    return GazePredictor(
        d_model=d_model,
        n_layers=n_layers,
        image_encoder_type=encoder_type,
        image_embed_dim=image_embed_dim,
        conditioning_mode=conditioning_mode,
        freeze_encoder=freeze_encoder,
        verbose=verbose,
        **encoder_kwargs,
    )
