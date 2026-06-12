"""
kaamba: Mamba2-based gaze predictor with modular image conditioning.

Architecture:
    - Swappable image encoder (ViT, ResNet, or SigLIP)
    - Mamba2 sequence backbone
    - Image conditioning either as initial hidden state or fused at every step
    - Gaussian mixture model (GMM) output head for NLL training

Notes:
    - SigLIP requires: pip install transformers>=4.39
    - For INITIAL_STATE mode, image signal decays over long sequences;
      EVERY_STEP conditioning is recommended for sequences > ~30 fixations.
    - Clamp log_sx / log_sy before exp() in your loss to avoid NaN:
        log_sx = log_sx.clamp(-6, 6)
"""

from __future__ import annotations

from enum import Enum
from typing import Tuple

import torch
import torch.nn as nn

from mamba_ssm import Mamba2
from transformers import ViTModel


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ImageEncoderType(str, Enum):
    VIT = "vit"
    RESNET = "resnet"
    SIGLIP = "siglip"


class ConditioningMode(str, Enum):
    INITIAL_STATE = "initial_state"
    EVERY_STEP = "every_step"
    NONE = "none"


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
# GMM output head
# ---------------------------------------------------------------------------


class GazeMDNHead(nn.Module):
    """
    Projects d_model → raw GMM parameters.

    No activations are applied here; caller must apply softmax / exp / tanh
    before computing the NLL.

    Important: clamp log_sx and log_sy to e.g. (-6, 6) before calling exp()
    in your loss function to prevent NaN during early training.
    """

    def __init__(self, d_model: int, n_mix: int = 5):
        super().__init__()
        self.K = n_mix
        self.proj = nn.Linear(d_model, n_mix * 6)

    def forward(self, h: torch.Tensor) -> Tuple:
        B, T, _ = h.shape
        out = self.proj(h).reshape(B, T, self.K, 6)
        pi_logits = out[..., 0]  # (B, T, K)   — softmax to get mixture weights
        mu = out[..., 1:3]  # (B, T, K, 2) — gaze (x, y)
        log_sx = out[..., 3]  # (B, T, K)   — clamp then exp for sigma_x
        log_sy = out[..., 4]  # (B, T, K)   — clamp then exp for sigma_y
        rho_raw = out[..., 5]  # (B, T, K)   — tanh for correlation in (-1, 1)
        return pi_logits, mu, log_sx, log_sy, rho_raw


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class GazePredictor(nn.Module):
    """
    Mamba2-based autoregressive gaze predictor with image conditioning.

    Input:
        images:   (B, 3, H, W)
        gaze_seq: (B, 2, T)   — shifted input (positions 0..T-1)

    Output: raw GMM parameters for positions 1..T
        pi_logits : (B, T, K)
        mu        : (B, T, K, 2)
        log_sx    : (B, T, K)
        log_sy    : (B, T, K)
        rho_raw   : (B, T, K)

    Conditioning modes:
        INITIAL_STATE — image embedding prepended as a leading token, then
                        stripped. Simple, but the image signal decays over
                        long sequences due to SSM recurrence.
        EVERY_STEP    — image embedding concatenated to gaze input at every
                        time step. Stronger for long or image-dense stimuli.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 4,
        image_encoder_type: str = "siglip",
        image_embed_dim: int = 256,
        conditioning_mode: str = "initial_state",
        n_mix: int = 5,
        freeze_encoder: bool = True,
        verbose: bool = True,
        **encoder_kwargs,
    ):
        super().__init__()
        self.conditioning_mode = ConditioningMode(conditioning_mode)
        self.d_model = d_model
        self.image_embed_dim = image_embed_dim

        if self.conditioning_mode != ConditioningMode.NONE:
            self.image_encoder = build_image_encoder(
                image_encoder_type,
                image_embed_dim,
                freeze_encoder,
                verbose,
                **encoder_kwargs,
            )
        else:
            self.image_encoder = None
            if verbose:
                print("[GazePredictor] image conditioning disabled (ablation)")

        input_dim = 2 + (
            image_embed_dim
            if self.conditioning_mode == ConditioningMode.EVERY_STEP
            else 0
        )
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        if self.conditioning_mode == ConditioningMode.INITIAL_STATE:
            # GELU preserves more encoder signal than Tanh (no hard cap at ±1)
            self.image_to_state = nn.Sequential(
                nn.Linear(image_embed_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
            )

        self.layers = nn.ModuleList([Mamba2(d_model=d_model) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = GazeMDNHead(d_model, n_mix)

        if verbose:
            total = sum(p.numel() for p in self.parameters())
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(
                f"[GazePredictor] encoder={image_encoder_type} mode={conditioning_mode} "
                f"d_model={d_model} layers={n_layers} K={n_mix}"
            )
            print(f"[GazePredictor] {total:,} total params, {trainable:,} trainable")

    def _run_layers(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def _prepare_input(
        self,
        gaze_seq: torch.Tensor,
        image_embed: torch.Tensor,
    ) -> torch.Tensor:
        gaze = gaze_seq.permute(0, 2, 1)  # (B, T, 2)
        if self.conditioning_mode == ConditioningMode.EVERY_STEP:
            B, T, _ = gaze.shape
            img = image_embed.unsqueeze(1).expand(B, T, self.image_embed_dim)
            gaze = torch.cat([gaze, img], dim=-1)  # (B, T, 2+E)
        return self.input_proj(gaze)  # (B, T, D)

    def forward(
        self,
        images: torch.Tensor,
        gaze_seq: torch.Tensor,
    ) -> Tuple:
        if self.conditioning_mode == ConditioningMode.NONE:
            x = self.input_proj(gaze_seq.permute(0, 2, 1))  # (B, T, D)
            x = self._run_layers(x)
            return self.head(self.norm(x))

        image_embed = self.image_encoder(images)  # (B, E)
        x = self._prepare_input(gaze_seq, image_embed)  # (B, T, D)

        if self.conditioning_mode == ConditioningMode.INITIAL_STATE:
            state = self.image_to_state(image_embed).unsqueeze(1)  # (B, 1, D)
            x = self._run_layers(torch.cat([state, x], dim=1))
            x = x[:, 1:, :]  # strip image token
        else:
            x = self._run_layers(x)

        return self.head(self.norm(x))  # 5-tuple of raw params


class CrossAttentionConditioner(nn.Module):
    """
    Single cross-attention block: gaze hidden states attend over image patches.

    Q = gaze sequence  (B, T, d_model)
    K = V = patches    (B, P, patch_dim)
    """

    def __init__(
        self, d_model: int, patch_dim: int, n_heads: int = 4, dropout: float = 0.0
    ):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(patch_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            kdim=patch_dim,
            vdim=patch_dim,
            batch_first=True,
            dropout=dropout,
        )
        self.norm_out = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        # x:       (B, T, d_model)
        # patches: (B, P, patch_dim)
        residual = x
        x = self.norm_q(x)
        kv = self.norm_kv(patches)
        x, _ = self.cross_attn(query=x, key=kv, value=kv)
        x = residual + x  # residual around attention
        x = x + self.ff(self.norm_out(x))  # residual around FF
        return x


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------


def build_gaze_predictor(
    conditioning_mode: str = "initial_state",
    encoder_type: str = "siglip",
    d_model: int = 128,
    n_layers: int = 4,
    image_embed_dim: int = 256,
    freeze_encoder: bool = True,
    verbose: bool = True,
    **encoder_kwargs,
) -> GazePredictor:
    """
    Convenience factory so callers don't need to import the enums.

    Examples:
        # Default: SigLIP + initial state conditioning
        model = build_gaze_predictor()

        # SigLIP with every-step conditioning (stronger for long sequences)
        model = build_gaze_predictor(
            conditioning_mode="every_step",
            encoder_type="siglip",
            d_model=256,
            n_layers=6,
        )

        # Larger SigLIP variant
        model = build_gaze_predictor(
            encoder_type="siglip",
            model_name="google/siglip-large-patch16-384",
            image_embed_dim=512,
        )

        # Fine-tune top encoder layer after building
        model = build_gaze_predictor(encoder_type="siglip", freeze_encoder=True)
        model.image_encoder.unfreeze_top_k(1)
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
