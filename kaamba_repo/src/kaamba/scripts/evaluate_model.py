"""
evaluate_model.py

Loads a pymovements dataset, iterates over gaze objects per stimulus,
generates matching synthetic sequences from a *sequence generator*, and
computes all GazeEvaluator metrics — both per-stimulus and aggregated.

Three built-in generators
──────────────────────────
  GMMModelGenerator          load a trained GMM  checkpoint (kaamba.py)
  CategoricalModelGenerator  load a trained categorical checkpoint
  SyntheticGenerator         generate step-function synthetic gaze

Usage
─────
  # GMM model
  python evaluate_model.py model \
      --checkpoint /path/best_model.pt --dataset mcfw-gaze --root /data

  # Categorical model
  python evaluate_model.py categorical \
      --checkpoint /path/best_cat.pt   --dataset mcfw-gaze --root /data

  # Synthetic baseline
  python evaluate_model.py synthetic \
      --dataset mcfw-gaze --root /data --noise 15

  # Side-by-side comparison (model vs synthetic, or two checkpoints)
  python evaluate_model.py compare \
      --checkpoint /path/model.pt --also_synthetic \
      --dataset mcfw-gaze --root /data

Output (each generator writes to its own sub-directory):
    <out_dir>/<generator_name>/
    ├── per_stimulus/<stimulus>.json
    ├── aggregate.json
    └── eval_report.txt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import polars as pl
import pymovements as pm
import torch
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from kaamba.net.models.kaamba import build_gaze_predictor
from kaamba.net.models.kaamba_categorical import (
    build_categorical_gaze_predictor,
)


# ---------------------------------------------------------------------------
# Sequence generator abstraction
# ---------------------------------------------------------------------------


class SequenceGenerator(ABC):
    """
    Abstract interface for anything that produces ``(N, gen_len, 2)``
    normalised gaze sequences.  The evaluation loop calls ``generate()``
    once per stimulus and treats the return value as the "fake" pool.
    """

    name: str = "unnamed"

    @abstractmethod
    def generate(
        self,
        img_path: Optional[Path],
        n: int,
        gen_len: int,
        seed_len: int,
        experiment,  # pm.gaze.Experiment — carries screen / sr info
        device: str = "cpu",
    ) -> np.ndarray:
        """Return (N, gen_len, 2) float32 array of normalised gaze in [0,1]."""


def generate_sequences(
    model,
    images,  # torch.Tensor (N, 3, H, W)
    seed_len: int = 10,
    gen_len: int = 200,
    temperature: float = 1.0,
    device: str = "cuda",
) -> np.ndarray:
    """
    Autoregressively sample from a trained GazePredictor.
    Returns (N, gen_len, 2) numpy array in normalised [0,1].
    """
    import torch

    model.eval()
    N = images.shape[0]
    images = images.to(device)
    generated = torch.full((N, 2, seed_len), 0.5, device=device)

    with torch.no_grad():
        for _ in range(gen_len - seed_len):
            pi, mu, log_sx, log_sy, rho_raw = model(images, generated)
            pi_t = torch.softmax(pi[:, -1, :], dim=-1)  # (N, K)
            mu_t = mu[:, -1, :, :]  # (N, K, 2)
            sx_t = log_sx[:, -1, :].exp().clamp(1e-4) * temperature
            sy_t = log_sy[:, -1, :].exp().clamp(1e-4) * temperature
            rho_t = torch.tanh(rho_raw[:, -1, :]) * 0.99

            k_idx = torch.multinomial(pi_t, 1).squeeze(-1)  # (N,)
            mu_k = mu_t[torch.arange(N), k_idx]  # (N, 2)
            sx_k = sx_t[torch.arange(N), k_idx]
            sy_k = sy_t[torch.arange(N), k_idx]
            rho_k = rho_t[torch.arange(N), k_idx]

            z1 = torch.randn(N, device=device)
            z2 = torch.randn(N, device=device)
            x_t = mu_k[:, 0] + sx_k * z1
            y_t = mu_k[:, 1] + sy_k * (rho_k * z1 + (1 - rho_k**2).sqrt() * z2)

            new_pt = torch.stack([x_t, y_t], dim=1).unsqueeze(-1)
            generated = torch.cat([generated, new_pt], dim=-1)

    return generated.permute(0, 2, 1).cpu().numpy()  # (N, T, 2)


# ── GMM model (kaamba.py) ────────────────────────────────────────────────────


class GMMModelGenerator(SequenceGenerator):
    """Load a trained GMM checkpoint and generate via bivariate Gaussian sampling."""

    def __init__(
        self,
        checkpoint_path: str,
        temperature: float = 1.0,
        device: str = "cpu",
        label: Optional[str] = None,
    ):
        self.device = device
        self.temperature = temperature
        ckpt = torch.load(checkpoint_path, map_location=device)
        model_config = ckpt["config"]["model_config"]
        self.model = build_gaze_predictor(**model_config, verbose=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model = self.model.to(device).eval()
        run_name = Path(checkpoint_path).parent.parent.name
        self.name = label or f"gmm_{run_name}"
        print(
            f"[{self.name}] loaded GMM model "
            f"({sum(p.numel() for p in self.model.parameters()):,} params)"
        )

    def generate(self, img_path, n, gen_len, seed_len, experiment, device=None):
        device = device or self.device
        img_tensor = _load_image_tensor(img_path, device)  # (1, 3, 224, 224)
        imgs_batch = img_tensor.expand(n, -1, -1, -1)
        with torch.no_grad():
            return generate_sequences(
                model=self.model,
                images=imgs_batch,
                seed_len=seed_len,
                gen_len=gen_len,
                temperature=self.temperature,
                device=device,
            )  # (N, gen_len, 2)


# ── Categorical model (kaamba_categorical.py) ────────────────────────────────


class CategoricalModelGenerator(SequenceGenerator):
    """Load a trained categorical checkpoint and generate via multinomial sampling."""

    def __init__(
        self,
        checkpoint_path: str,
        temperature: float = 1.0,
        device: str = "cpu",
        label: Optional[str] = None,
    ):
        self.device = device
        self.temperature = temperature
        ckpt = torch.load(checkpoint_path, map_location=device)
        model_config = ckpt["config"]["model_config"]
        self.model = build_categorical_gaze_predictor(**model_config, verbose=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model = self.model.to(device).eval()
        self.n_bins = self.model.n_bins
        run_name = Path(checkpoint_path).parent.parent.name
        self.name = label or f"categorical_{run_name}"
        print(
            f"[{self.name}] loaded categorical model "
            f"(n_bins={self.n_bins}, "
            f"{sum(p.numel() for p in self.model.parameters()):,} params)"
        )

    def generate(self, img_path, n, gen_len, seed_len, experiment, device=None):
        device = device or self.device
        n_bins = self.n_bins
        temp = self.temperature
        img_t = _load_image_tensor(img_path, device).expand(n, -1, -1, -1)
        seq = torch.full((n, 2, seed_len), 0.5, device=device)

        with torch.no_grad():
            for _ in range(gen_len - seed_len):
                lx, ly = self.model(img_t, seq)  # each (N, n_bins, T)
                # last-step logits → (N, n_bins)
                px = (lx[:, :, -1] / temp).softmax(dim=1)
                py = (ly[:, :, -1] / temp).softmax(dim=1)
                xb = torch.multinomial(px, 1).squeeze(1).float()
                yb = torch.multinomial(py, 1).squeeze(1).float()
                xc = (xb + 0.5) / n_bins
                yc = (yb + 0.5) / n_bins
                nxt = torch.stack([xc, yc], dim=1).unsqueeze(2)  # (N,2,1)
                seq = torch.cat([seq, nxt], dim=2)

        return seq.permute(0, 2, 1).cpu().numpy()  # (N, T, 2)


# ── Synthetic step-function baseline ─────────────────────────────────────────


class SyntheticGenerator(SequenceGenerator):
    """
    Generate synthetic gaze using pm.synthetic.step_function.
    Screen dimensions and sampling rate are taken from the experiment object
    passed at generation time (extracted from the real dataset).
    """

    def __init__(
        self,
        fix_dur_mean_ms: float = 250.0,
        fix_dur_std_ms: float = 80.0,
        sac_dur_mean_ms: float = 40.0,
        sac_dur_std_ms: float = 15.0,
        noise: float = 0.5,
        values_spread: float = 0.7,
        start_value: Optional[tuple] = None,
        values_center: Optional[tuple] = None,
        seed: int = 42,
        label: Optional[str] = None,
    ):
        self.fix_mean = fix_dur_mean_ms
        self.fix_std = fix_dur_std_ms
        self.sac_mean = sac_dur_mean_ms
        self.sac_std = sac_dur_std_ms
        self.noise = noise
        self.spread = values_spread
        self.start = start_value
        self.center = values_center
        self.seed = seed
        self.name = label or "synthetic"

    def generate(self, img_path, n, gen_len, seed_len, experiment, device=None):
        screen = experiment.screen
        sr = float(experiment.sampling_rate)
        sw = int(screen.width_px)
        sh = int(screen.height_px)

        cx, cy = self.center or (sw / 2, sh / 2)
        start = self.start or (sw / 2, sh / 2)
        half_w = sw / 2 * self.spread
        half_h = sh / 2 * self.spread

        rng = np.random.default_rng(self.seed)
        seqs = []

        def _ms2s(ms):
            return max(1, int(round(ms * sr / 1000)))

        for _ in range(n):
            steps, values, cursor = [], [], 0
            while cursor < gen_len:
                fd = max(
                    _ms2s(self.fix_mean * 0.3),
                    int(rng.normal(_ms2s(self.fix_mean), _ms2s(self.fix_std))),
                )
                x = float(np.clip(rng.uniform(cx - half_w, cx + half_w), 0, sw))
                y = float(np.clip(rng.uniform(cy - half_h, cy + half_h), 0, sh))
                steps.append(cursor)
                values.append((x, y))
                cursor += fd
                if cursor >= gen_len:
                    break
                sd = max(1, int(rng.normal(_ms2s(self.sac_mean), _ms2s(self.sac_std))))
                nx = float(np.clip(rng.uniform(cx - half_w, cx + half_w), 0, sw))
                ny = float(np.clip(rng.uniform(cy - half_h, cy + half_h), 0, sh))
                steps.append(cursor)
                values.append(((x + nx) / 2, (y + ny) / 2))
                cursor += sd

            pairs = [(s, v) for s, v in zip(steps, values) if s < gen_len]
            if not pairs:
                pairs = [(0, (sw / 2, sh / 2))]
            s_list, v_list = zip(*pairs)

            pos = pm.synthetic.step_function(
                length=gen_len,
                steps=list(s_list),
                values=list(v_list),
                start_value=start,
                noise=self.noise,
            )  # (gen_len, 2) in pixels
            norm = np.stack([pos[:, 0] / sw, pos[:, 1] / sh], axis=1)
            seqs.append(norm.astype(np.float32))

        return np.stack(seqs)  # (N, gen_len, 2)


# ── Image loading helper ─────────────────────────────────────────────────────


def _load_image_tensor(img_path: Path, device: str) -> torch.Tensor:
    """Load and resize an image to (1, 3, 224, 224)."""
    import torchvision.transforms.functional as TF
    from PIL import Image

    img = Image.open(img_path).convert("RGB")
    return TF.to_tensor(TF.resize(img, [224, 224])).unsqueeze(0).to(device)


# ---------------------------------------------------------------------------
# pymovements-native preprocessing helpers
# ---------------------------------------------------------------------------

_EMPTY_FIX = pl.DataFrame(
    schema={
        "name": pl.Utf8,
        "onset": pl.Int64,
        "offset": pl.Int64,
        "duration": pl.Int64,
        "cx_deg": pl.Float64,
        "cy_deg": pl.Float64,
    }
)
_EMPTY_SAC = pl.DataFrame(
    schema={
        "name": pl.Utf8,
        "onset": pl.Int64,
        "offset": pl.Int64,
        "duration": pl.Int64,
        "amplitude_deg": pl.Float64,
        "peak_vel_deg_s": pl.Float64,
        "angle_rad": pl.Float64,
    }
)


def _fix_df_from_events(ev_frame: pl.DataFrame, pos_arr: np.ndarray) -> pl.DataFrame:
    """
    Filter fixation events from a preprocessed events DataFrame and append
    centroid columns ``cx_deg`` / ``cy_deg`` (mean deg position per fixation).

    ``pos_arr`` must be (T, 2) in degrees of visual angle — taken from
    ``gaze.samples["position"]`` after ``pix2deg()``.
    """
    if ev_frame is None or len(ev_frame) == 0:
        return _EMPTY_FIX
    fix = ev_frame.filter(pl.col("name") == "fixation")
    if len(fix) == 0:
        return _EMPTY_FIX
    cx_list, cy_list = [], []
    for row in fix.iter_rows(named=True):
        seg = pos_arr[row["onset"] : row["offset"]]
        if len(seg) == 0:
            cx_list.append(float("nan"))
            cy_list.append(float("nan"))
        else:
            cx_list.append(float(seg[:, 0].mean()))
            cy_list.append(float(seg[:, 1].mean()))
    result = fix.with_columns(
        [
            pl.Series("cx_deg", cx_list, dtype=pl.Float64),
            pl.Series("cy_deg", cy_list, dtype=pl.Float64),
        ]
    )
    # Keep only the columns downstream code expects (drops dispersion etc.)
    keep = ["name", "onset", "offset", "duration", "cx_deg", "cy_deg"]
    return result.select([c for c in keep if c in result.columns])


def _sac_df_from_events(ev_frame: pl.DataFrame, pos_arr: np.ndarray) -> pl.DataFrame:
    """
    Filter saccade events, append saccade direction ``angle_rad``, and rename
    ``amplitude`` → ``amplitude_deg``  /  ``peak_velocity`` → ``peak_vel_deg_s``
    to match the ``evaluate_stimulus()`` column expectations.

    ``amplitude`` and ``peak_velocity`` must already be present — they are
    added by ``compute_event_properties(["amplitude", "peak_velocity", ...])``.
    """
    if ev_frame is None or len(ev_frame) == 0:
        return _EMPTY_SAC
    sac = ev_frame.filter(pl.col("name") == "saccade")
    if len(sac) == 0:
        return _EMPTY_SAC
    angle_list = []
    for row in sac.iter_rows(named=True):
        seg = pos_arr[row["onset"] : row["offset"]]
        if len(seg) > 1:
            angle_list.append(
                float(np.arctan2(seg[-1, 1] - seg[0, 1], seg[-1, 0] - seg[0, 0]))
            )
        else:
            angle_list.append(float("nan"))
    result = sac.with_columns(pl.Series("angle_rad", angle_list, dtype=pl.Float64))
    rename = {}
    if "amplitude" in result.columns:
        rename["amplitude"] = "amplitude_deg"
    if "peak_velocity" in result.columns:
        rename["peak_velocity"] = "peak_vel_deg_s"
    if rename:
        result = result.rename(rename)
    keep = [
        "name",
        "onset",
        "offset",
        "duration",
        "amplitude_deg",
        "peak_vel_deg_s",
        "angle_rad",
    ]
    return result.select([c for c in keep if c in result.columns])


# ---------------------------------------------------------------------------
# Per-stimulus evaluation
# ---------------------------------------------------------------------------


def evaluate_stimulus(
    real_seqs: np.ndarray,  # (N_real, T, 2) normalised
    fake_seqs: np.ndarray,  # (N_fake, T, 2) normalised
    real_fix_df: pl.DataFrame,  # enriched fixation frame (real)
    real_sac_df: pl.DataFrame,  # enriched saccade frame  (real)
    fake_fix_df: pl.DataFrame,  # enriched fixation frame (fake)
    fake_sac_df: pl.DataFrame,  # enriched saccade frame  (fake)
) -> Dict:
    """Run all metrics for one stimulus, using pre-extracted event frames."""
    from scipy import stats
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    def _ks(a, b):
        if len(a) < 2 or len(b) < 2:
            return {
                "ks_stat": float("nan"),
                "p_value": float("nan"),
                "n_real": len(a),
                "n_fake": len(b),
            }
        s, p = stats.ks_2samp(a, b)
        return {
            "ks_stat": float(s),
            "p_value": float(p),
            "n_real": len(a),
            "n_fake": len(b),
        }

    # ── Fixation duration ─────────────────────────────────────────────────
    r_dur = (
        real_fix_df["duration"].to_numpy().astype(float)
        if len(real_fix_df)
        else np.array([])
    )
    f_dur = (
        fake_fix_df["duration"].to_numpy().astype(float)
        if len(fake_fix_df)
        else np.array([])
    )
    fix_dur = _ks(r_dur, f_dur)
    fix_dur.update(
        {
            "real_mean": float(r_dur.mean()) if len(r_dur) else float("nan"),
            "fake_mean": float(f_dur.mean()) if len(f_dur) else float("nan"),
            "real_std": float(r_dur.std()) if len(r_dur) else float("nan"),
            "fake_std": float(f_dur.std()) if len(f_dur) else float("nan"),
        }
    )

    # ── Saccade amplitude ─────────────────────────────────────────────────
    r_amp = (
        real_sac_df["amplitude_deg"].to_numpy() if len(real_sac_df) else np.array([])
    )
    f_amp = (
        fake_sac_df["amplitude_deg"].to_numpy() if len(fake_sac_df) else np.array([])
    )
    sac_amp = _ks(r_amp, f_amp)
    sac_amp.update(
        {
            "real_mean_deg": float(r_amp.mean()) if len(r_amp) else float("nan"),
            "fake_mean_deg": float(f_amp.mean()) if len(f_amp) else float("nan"),
        }
    )

    # ── Main sequence ─────────────────────────────────────────────────────
    def _ms_r(sac_df):
        if len(sac_df) < 5:
            return float("nan")
        amp = sac_df["amplitude_deg"].to_numpy()
        pv = sac_df["peak_vel_deg_s"].to_numpy()
        m = (amp > 0.1) & (pv > 1.0)
        if m.sum() < 5:
            return float("nan")
        r, _ = stats.pearsonr(amp[m], pv[m])
        return float(r)

    fake_r = _ms_r(fake_sac_df)
    main_seq = {
        "real_r": _ms_r(real_sac_df),
        "fake_r": fake_r,
        "pass": fake_r > 0.9 if not np.isnan(fake_r) else False,
    }

    # ── ISI ───────────────────────────────────────────────────────────────
    isi = {
        "real_mean": float(r_dur.mean()) if len(r_dur) else float("nan"),
        "fake_mean": float(f_dur.mean()) if len(f_dur) else float("nan"),
        "real_var": float(r_dur.var()) if len(r_dur) else float("nan"),
        "fake_var": float(f_dur.var()) if len(f_dur) else float("nan"),
        "mean_err": float(abs(r_dur.mean() - f_dur.mean()))
        if (len(r_dur) and len(f_dur))
        else float("nan"),
    }

    # ── Fixation density (KL) ─────────────────────────────────────────────
    def _density(fix_df, grid=32):
        if len(fix_df) == 0:
            return None
        cx = fix_df["cx_deg"].to_numpy()
        cy = fix_df["cy_deg"].to_numpy()

        # Drop NaN centroids (from fixations with empty position slices)
        valid = ~(np.isnan(cx) | np.isnan(cy))
        cx, cy = cx[valid], cy[valid]

        if len(cx) < 3:  # too few fixations to build a meaningful map
            return None

        r = max(abs(cx).max(), abs(cy).max(), 1e-9)
        xi = ((cx / r * 0.5 + 0.5) * (grid - 1)).astype(int).clip(0, grid - 1)
        yi = ((cy / r * 0.5 + 0.5) * (grid - 1)).astype(int).clip(0, grid - 1)
        h, _, _ = np.histogram2d(xi, yi, bins=grid, range=[[0, grid], [0, grid]])
        h += 1e-8
        return h / h.sum()

    rd, fd = _density(real_fix_df), _density(fake_fix_df)
    if rd is not None and fd is not None:
        kl = float(stats.entropy(rd.ravel(), fd.ravel()))
        density = {
            "kl_divergence": kl,
            "real_n_fixations": len(real_fix_df),
            "fake_n_fixations": len(fake_fix_df),
        }
    else:
        density = {
            "kl_divergence": float("nan"),
            "real_n_fixations": len(real_fix_df),
            "fake_n_fixations": len(fake_fix_df),
        }

    # ── Saccade direction ─────────────────────────────────────────────────
    def _dir_hist(sac_df, n=8):
        if len(sac_df) == 0 or "angle_rad" not in sac_df.columns:
            return None
        ang = sac_df["angle_rad"].drop_nulls().to_numpy()
        ang = ang[~np.isnan(ang)]
        if len(ang) < 3:
            return None
        h, _ = np.histogram(ang, bins=n, range=(-np.pi, np.pi))
        return h.astype(float)

    rh, fh = _dir_hist(real_sac_df), _dir_hist(fake_sac_df)
    if rh is not None and fh is not None:
        rhn = rh / rh.sum()
        fhn = fh / fh.sum()
        dir_kl = float(stats.entropy(rhn + 1e-8, fhn + 1e-8))
        direction = {"kl_divergence": dir_kl, **_ks(rh, fh)}
    else:
        direction = {"kl_divergence": float("nan"), "note": "insufficient saccades"}

    # ── Classifier AUC ────────────────────────────────────────────────────
    def _feats(seqs):
        dx = np.diff(seqs, axis=1)
        speed = np.linalg.norm(dx, axis=-1)
        return np.concatenate(
            [
                seqs[:, :, 0].mean(1, keepdims=True),
                seqs[:, :, 1].mean(1, keepdims=True),
                seqs[:, :, 0].std(1, keepdims=True),
                seqs[:, :, 1].std(1, keepdims=True),
                speed.mean(1, keepdims=True),
                speed.std(1, keepdims=True),
                speed.max(1, keepdims=True),
            ],
            axis=1,
        )

    nr = min(len(real_seqs), 500)
    nf = min(len(fake_seqs), 500)
    if nr >= 5 and nf >= 5:
        X = np.concatenate([_feats(real_seqs[:nr]), _feats(fake_seqs[:nf])])
        y = np.array([1] * nr + [0] * nf)
        # Drop any rows that still contain NaN or Inf after feature extraction
        valid = np.isfinite(X).all(axis=1)
        X, y = X[valid], y[valid]
        if len(np.unique(y)) < 2 or len(y) < 10:
            clf_result = {
                "auc": float("nan"),
                "pass": False,
                "note": "too few valid rows after NaN removal",
            }
        else:
            X = StandardScaler().fit_transform(X)
            clf = LogisticRegression(max_iter=500, random_state=42)
            clf.fit(X, y)
            auc = float(roc_auc_score(y, clf.predict_proba(X)[:, 1]))
            clf_result = {"auc": auc, "pass": abs(auc - 0.5) < 0.1}
    else:
        clf_result = {
            "auc": float("nan"),
            "pass": False,
            "note": "too few sequences for classifier",
        }

    return {
        "fixation_duration": fix_dur,
        "saccade_amplitude": sac_amp,
        "main_sequence": main_seq,
        "intersaccadic_interval": isi,
        "fixation_density_map": density,
        "saccade_direction": direction,
        "classifier_auc": clf_result,
        "n_real_seqs": len(real_seqs),
        "n_fake_seqs": len(fake_seqs),
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def run_evaluation(
    generator: SequenceGenerator,
    dataset_name: str,
    root: str,
    out_dir: str,
    subset: Optional[Dict] = None,
    n_generate: int = 50,
    seed_len: int = 10,
    gen_len: int = 128,
    vel_threshold: float = 30.0,  # deg/s IVT threshold
    min_fix_duration: int = 100,  # samples
    dispersion_threshold: float = 1.0,  # deg, for I-DT fixation detection
    vel_method: str = "fivepoint",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict:
    """
    Main evaluation loop.

    Accepts any SequenceGenerator (GMMModelGenerator, CategoricalModelGenerator,
    or SyntheticGenerator) and writes results to out_dir / generator.name /.

    For each (subject, stimulus) pair in the dataset:
      1. Load real gaze sequences
      2. Call generator.generate() for the matching stimulus image
      3. Run all  metrics
      4. Save per-stimulus JSON + aggregate report
    """
    out_dir = Path(out_dir) / generator.name
    stim_dir = out_dir / "per_stimulus"
    stim_dir.mkdir(parents=True, exist_ok=True)

    # ── Load dataset ──────────────────────────────────────────────────────
    print(f"\n[eval] Generator : {generator.name}")
    print(f"[eval] Loading dataset: {dataset_name}")
    dataset_paths = pm.DatasetPaths(root=root)
    dataset = pm.Dataset(dataset_name, path=dataset_paths)
    dataset.scan()
    dataset.load(subset=subset)
    if dataset_name == "GGTG":
        dataset.split_gaze_data(by="stimulus")

    print(f"[eval] Loaded {len(dataset.gaze)} gaze files")

    # ── Screen info from first gaze object ───────────────────────────────
    first_gaze = dataset.gaze[0]
    screen = first_gaze.experiment.screen
    sr = first_gaze.experiment.sampling_rate
    scr_w_px = screen.width_px
    scr_h_px = screen.height_px

    # ── Preprocess entire dataset with pymovements built-ins ──────────────
    # This runs once on the stored polars frames — no cloning, no manual
    # numpy velocity computation.  After these calls every gaze object in
    # dataset.gaze has  position  and  velocity  columns, and every entry in
    # dataset.events has  fixation  and  saccade  rows with amplitude /
    # dispersion / peak_velocity properties already populated.
    print(
        "[eval] Preprocessing: pix2deg → pos2vel → ID-T → microsaccades → "
        "event properties …"
    )

    dataset.pix2deg()
    dataset.pos2vel(method=vel_method)
    dataset.save_preprocessed()

    dataset.detect_events(
        "idt",
        dispersion_threshold=dispersion_threshold,
        clear=True,
        minimum_duration=min_fix_duration,
    )
    dataset.detect_events("fill")

    dataset.compute_event_properties(["amplitude", "dispersion", "peak_velocity"])
    dataset.save_events()
    print(f"[eval] Preprocessing complete — {len(dataset.gaze)} recordings ready")

    # ── Group gaze objects + event frames by stimulus ─────────────────────
    from collections import defaultdict

    by_stimulus = defaultdict(list)
    for gaze, ev_frame in zip(dataset.gaze, dataset.events):
        stim = gaze.metadata.get("stimulus", "unknown")
        by_stimulus[stim].append((gaze, ev_frame))

    print(f"[eval] {len(by_stimulus)} unique stimuli\n")

    # ── Build stimulus name → image path lookup ───────────────────────────
    # dataset.stimuli is broken for some dataset configs — use fileinfo instead,
    # matching the workaround used during training:
    #   self.stimuli = self.pm_dataset.fileinfo["ImageStimulus"]
    #
    # fileinfo["ImageStimulus"] is a polars DataFrame with columns:
    #   "stimulus"  — name matching gaze.metadata["stimulus"]
    #   "filepath"  — path to the image file
    stim_fileinfo = dataset.fileinfo["ImageStimulus"]
    stim_images: Dict[str, Path] = {}
    for row in stim_fileinfo.iter_rows(named=True):
        name = row["stimulus"]
        img_path = Path(row["filepath"])
        if not img_path.is_absolute():
            img_path = f"{root}/{dataset_name}/stimuli" / img_path
        stim_images[name] = img_path
    print(f"[eval] {len(stim_images)} stimulus images in fileinfo")
    valid_stimuli = set(stim_images.keys())
    by_stimulus = {s: g for s, g in by_stimulus.items() if s in valid_stimuli}
    print(f"[eval] {len(by_stimulus)} stimuli with matching images")

    # ── Per-stimulus loop ─────────────────────────────────────────────────
    all_results = {}
    _plot_cache = {}  # stores raw arrays for post-hoc plotting
    timing_total = 0.0

    for stim_name, gaze_list in tqdm(by_stimulus.items(), desc="Stimuli"):
        t0 = time.time()

        # ── Collect real sequences ────────────────────────────────────────
        real_norm_seqs = []  # (T, 2) normalised, one per recording
        all_real_fix = []  # fixation DataFrames
        all_real_sac = []  # saccade DataFrames

        for gaze, ev_frame in gaze_list:
            try:
                # position and pixel columns are already populated by the
                # dataset-level preprocessing above — no clone, no copy.
                px_raw = np.stack(gaze.samples["pixel"].to_numpy())  # (T,2) px
                pos_arr = np.stack(gaze.samples["position"].to_numpy())  # (T,2) deg
                norm_arr = np.column_stack(
                    [px_raw[:, 0] / scr_w_px, px_raw[:, 1] / scr_h_px]
                )
                fix_df = _fix_df_from_events(ev_frame.frame, pos_arr)
                sac_df = _sac_df_from_events(ev_frame.frame, pos_arr)
            except Exception as e:
                print(
                    f"  [warn] {stim_name} / {gaze.metadata.get('subject_id')} "
                    f"event extraction failed: {e}"
                )
                continue

            T_avail = len(norm_arr)
            if T_avail < seed_len + gen_len:
                tqdm.write(
                    f"  [dbg] {stim_name}/{gaze.metadata.get('subject_id')}: "
                    f"only {T_avail} samples, need {seed_len + gen_len} — skipping"
                )
                continue

            step = max(1, gen_len // 2)  # 50 % overlap — richer real pool
            for start in range(0, T_avail - gen_len + 1, step):
                real_norm_seqs.append(norm_arr[start : start + gen_len])

            all_real_fix.append(fix_df)
            all_real_sac.append(sac_df)

        if len(real_norm_seqs) == 0:
            print(f"  [skip] {stim_name}: no valid real sequences")
            continue

        real_arr = np.stack(real_norm_seqs)  # (N_r, T, 2)
        real_fix_df = pl.concat(all_real_fix) if all_real_fix else pl.DataFrame()
        real_sac_df = pl.concat(all_real_sac) if all_real_sac else pl.DataFrame()

        # ── Load stimulus image via fileinfo path ────────────────────────
        # gaze.metadata["stimulus"] is the key — same as used during training
        img_path = stim_images.get(stim_name)
        if img_path is None:
            print(f"  [skip] {stim_name}: not in fileinfo ImageStimulus")
            continue
        if not img_path.exists():
            print(f"  [skip] {stim_name}: image file not found at {img_path}")
            continue

        # ── Generate sequences via generator ──────────────────────────────
        fake_norm = generator.generate(
            img_path=img_path,
            n=n_generate,
            gen_len=gen_len,
            seed_len=seed_len,
            experiment=first_gaze.experiment,
            device=device,
        )  # (N, gen_len, 2)

        # ── Extract events from fake sequences ────────────────────────────
        all_fake_fix = []
        all_fake_sac = []

        for seq in fake_norm:  # (gen_len, 2) normalised
            px_vals = seq * np.array([scr_w_px, scr_h_px], dtype=float)
            g_fake = pm.Gaze(
                pl.DataFrame(
                    {
                        "x_pix": px_vals[:, 0],
                        "y_pix": px_vals[:, 1],
                    }
                ),
                pixel_columns=["x_pix", "y_pix"],
                experiment=first_gaze.experiment,
            )
            try:
                g_fake.pix2deg()
                g_fake.pos2vel(method=vel_method)
                g_fake.detect("idt", clear=True, minimum_duration=100)
                g_fake.detect("microsaccades", minimum_duration=3)
                g_fake.compute_event_properties(
                    ["amplitude", "dispersion", "peak_velocity"]
                )
                pos_f = np.stack(g_fake.samples["position"].to_numpy())
                ev_df = g_fake.events.frame
                all_fake_fix.append(_fix_df_from_events(ev_df, pos_f))
                all_fake_sac.append(_sac_df_from_events(ev_df, pos_f))
            except Exception as e:
                tqdm.write(f"  [warn] fake event detection failed: {e}")

        fake_fix_df = pl.concat(all_fake_fix) if all_fake_fix else pl.DataFrame()
        fake_sac_df = pl.concat(all_fake_sac) if all_fake_sac else pl.DataFrame()

        # ── Run metrics ───────────────────────────────────────────────────
        metrics = evaluate_stimulus(
            real_seqs=real_arr,
            fake_seqs=fake_norm,
            real_fix_df=real_fix_df,
            real_sac_df=real_sac_df,
            fake_fix_df=fake_fix_df,
            fake_sac_df=fake_sac_df,
        )
        metrics["stimulus"] = stim_name
        metrics["time_s"] = time.time() - t0
        timing_total += metrics["time_s"]

        # Save per-stimulus JSON
        safe_name = stim_name.replace("/", "_").replace(" ", "_")
        (stim_dir / f"{safe_name}.json").write_text(json.dumps(metrics, indent=2))
        all_results[stim_name] = metrics

        # Cache raw arrays and event frames for plotting
        _plot_cache[stim_name] = {
            "real_seqs": real_arr,
            "fake_seqs": fake_norm,
            "real_fix_df": real_fix_df,
            "real_sac_df": real_sac_df,
            "fake_fix_df": fake_fix_df,
            "fake_sac_df": fake_sac_df,
            "img_path": img_path,
        }

        tqdm.write(
            f"  {stim_name:40s} | "
            f"fix_KS={metrics['fixation_duration']['ks_stat']:.3f} "
            f"sac_KS={metrics['saccade_amplitude']['ks_stat']:.3f} "
            f"AUC={metrics['classifier_auc']['auc']:.3f} "
            f"({metrics['time_s']:.1f}s)"
        )

    if not all_results:
        print("[eval] No results — check dataset loading and subset filter.")
        return {}

    # ── Aggregate across stimuli ──────────────────────────────────────────
    aggregate = _aggregate_results(all_results)
    (out_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2))

    # ── Human-readable report ─────────────────────────────────────────────
    report = _build_report(all_results, aggregate, timing_total)
    report_path = out_dir / "eval_report.txt"
    report_path.write_text(report)
    print(report)
    print(f"\n[eval] Results saved to {out_dir}")

    # ── Comparison plot (best vs worst matching stimulus) ─────────────────
    if len(all_results) >= 2:
        plot_path = out_dir / "comparison_best_worst.png"
        plot_best_worst_comparison(
            all_results=all_results,
            plot_cache=_plot_cache,
            out_path=plot_path,
            score_metric=("classifier_auc", "auc"),  # primary ranking metric
        )

    return all_results


# ---------------------------------------------------------------------------
# Aggregation + report
# ---------------------------------------------------------------------------


def _aggregate_results(all_results: Dict) -> Dict:
    """Average scalar metrics across all stimuli."""
    agg: Dict[str, Dict[str, list]] = {}

    for stim_metrics in all_results.values():
        for section, sub in stim_metrics.items():
            if not isinstance(sub, dict):
                continue
            if section not in agg:
                agg[section] = {}
            for k, v in sub.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    agg[section].setdefault(k, []).append(v)

    # Compute mean ± std, ignoring NaN
    out = {}
    for section, metrics in agg.items():
        out[section] = {}
        for k, vals in metrics.items():
            arr = np.array([v for v in vals if not np.isnan(v)])
            out[section][k] = {
                "mean": float(arr.mean()) if len(arr) else float("nan"),
                "std": float(arr.std()) if len(arr) else float("nan"),
                "n": int(len(arr)),
            }
    return out


def _build_report(all_results: Dict, aggregate: Dict, total_time: float) -> str:
    lines = [
        "=" * 70,
        "GAZE EVALUATION REPORT",
        f"  {len(all_results)} stimuli evaluated in {total_time:.1f}s",
        "=" * 70,
        "",
        "AGGREGATE METRICS (mean ± std across stimuli)",
        "-" * 70,
    ]

    key_metrics = [
        ("fixation_duration", "ks_stat", "Fixation duration KS stat    "),
        ("fixation_duration", "p_value", "Fixation duration KS p-value "),
        ("saccade_amplitude", "ks_stat", "Saccade amplitude KS stat    "),
        ("saccade_amplitude", "real_mean_deg", "Saccade amplitude real (deg) "),
        ("saccade_amplitude", "fake_mean_deg", "Saccade amplitude fake (deg) "),
        ("main_sequence", "real_r", "Main sequence r (real)       "),
        ("main_sequence", "fake_r", "Main sequence r (fake)       "),
        ("intersaccadic_interval", "mean_err", "ISI mean error (samples)     "),
        ("fixation_density_map", "kl_divergence", "Fixation density KL div      "),
        ("saccade_direction", "kl_divergence", "Saccade direction KL div     "),
        ("classifier_auc", "auc", "Classifier AUC               "),
    ]

    for section, key, label in key_metrics:
        val = aggregate.get(section, {}).get(key, {})
        if val:
            lines.append(
                f"  {label}  {val['mean']:7.4f} ± {val['std']:.4f}  (n={val['n']})"
            )

    lines += [
        "",
        "PER-STIMULUS SUMMARY",
        "-" * 70,
        f"  {'Stimulus':<40} {'fix_KS':>8} {'sac_KS':>8} {'AUC':>8}",
    ]

    for stim, m in sorted(all_results.items()):
        fks = m["fixation_duration"]["ks_stat"]
        asks = m["saccade_amplitude"]["ks_stat"]
        auc = m["classifier_auc"]["auc"]
        lines.append(f"  {stim:<40} {fks:>8.4f} {asks:>8.4f} {auc:>8.4f}")

    lines += ["", "PASS / FAIL (aggregate means)", "-" * 70]

    checks = {
        "Fixation KS p > 0.05": aggregate.get("fixation_duration", {})
        .get("p_value", {})
        .get("mean", 0)
        > 0.05,
        "Saccade  KS p > 0.05": aggregate.get("saccade_amplitude", {})
        .get("p_value", {})
        .get("mean", 0)
        > 0.05,
        "Main seq fake_r > 0.9": aggregate.get("main_sequence", {})
        .get("fake_r", {})
        .get("mean", 0)
        > 0.9,
        "Classifier AUC ≈ 0.5": abs(
            aggregate.get("classifier_auc", {}).get("auc", {}).get("mean", 1) - 0.5
        )
        < 0.1,
    }
    for name, passed in checks.items():
        lines.append(f"  {'✓' if passed else '✗'} {name}")

    lines.append("=" * 70)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Comparison plot: best vs worst matching stimulus
# ---------------------------------------------------------------------------


def plot_best_worst_comparison(
    all_results: Dict,
    plot_cache: Dict,
    out_path: "str | Path",
    score_metric: tuple = (
        "fixation_duration",
        "ks_stat",
    ),  # ("classifier_auc", "auc"),
    n_scanpaths: int = 8,
    density_grid: int = 32,
    density_sigma: float = 1.5,
) -> None:
    """
    Produce a 2-row × 5-column figure comparing the best and worst
    matching stimuli. Saves to out_path as a PNG.

    Layout (each row = one stimulus):
        col 0 : stimulus image
        col 1 : scanpath overlay  (real teal, fake purple)
        col 2 : fixation duration KDE  (real vs fake)
        col 3 : saccade amplitude KDE  (real vs fake)
        col 4 : fixation density difference map (real − fake)

    Ranking: stimuli are ranked by score_metric. The metric whose value
    is closest to its ideal (AUC → 0.5, KS p → 1.0) is the "best".
    For AUC the distance is |AUC - 0.5|; for KS p the distance is (1 - p).
    Lower distance = better match.

    Args:
        all_results:  dict returned by run_evaluation
        plot_cache:   dict of {stim_name: {real_seqs, fake_seqs, ...}}
        out_path:     save path for the PNG
        score_metric: (section, key) tuple to rank stimuli by
        n_scanpaths:  how many scanpath traces to draw per condition
        density_grid: grid resolution for density maps
        density_sigma: Gaussian smoothing sigma for density maps
    """
    from scipy import stats
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from PIL import Image

    section, key = score_metric

    # ── Rank stimuli ──────────────────────────────────────────────────────
    def _score(m):
        val = m.get(section, {}).get(key, float("nan"))
        if np.isnan(val):
            return float("inf")
        if section == "classifier_auc" and key == "auc":
            return abs(val - 0.5)  # ideal = 0.5
        return 1.0 - val  # for p-values: ideal = 1.0

    ranked = sorted(
        [(stim, _score(m)) for stim, m in all_results.items() if stim in plot_cache],
        key=lambda x: x[1],
    )

    if len(ranked) < 2:
        print("[plot] need at least 2 stimuli to compare — skipping")
        return

    best_name, best_score = ranked[0]
    worst_name, worst_score = ranked[-1]

    # ── Helpers ───────────────────────────────────────────────────────────
    C_REAL = "#1D9E75"  # Teal
    C_FAKE = "#7F77DD"  # purple
    ALPHA_T = 0.35  # trace alpha

    def _density_map(seqs, fix_df, grid, sigma):
        """Fixation-centroid density on a grid — falls back to all points."""
        h = np.zeros((grid, grid))
        if fix_df is not None and len(fix_df) > 0 and "cx_deg" in fix_df.columns:
            cx = fix_df["cx_deg"].to_numpy()
            cy = fix_df["cy_deg"].to_numpy()
            # Drop NaN centroids (from fixations with empty position slices)
            valid = ~(np.isnan(cx) | np.isnan(cy))
            cx, cy = cx[valid], cy[valid]

            if len(cx) < 3:  # too few fixations to build a meaningful map
                return None
            # normalise to [0,1] for gridding
            xrange = cx.max() - cx.min() or 1.0
            yrange = cy.max() - cy.min() or 1.0
            xi = ((cx - cx.min()) / xrange * (grid - 1)).astype(int).clip(0, grid - 1)
            yi = ((cy - cy.min()) / yrange * (grid - 1)).astype(int).clip(0, grid - 1)
        else:
            # fallback: all gaze points
            pts = np.clip(seqs.reshape(-1, 2), 0, 1 - 1e-9)
            xi = (pts[:, 0] * (grid - 1)).astype(int).clip(0, grid - 1)
            yi = (pts[:, 1] * (grid - 1)).astype(int).clip(0, grid - 1)
        for x, y in zip(xi, yi):
            h[y, x] += 1
        return gaussian_filter(h.astype(float) + 1e-8, sigma=sigma)

    def _kde_plot(ax, real_data, fake_data, xlabel, unit=""):
        real_data = real_data[~np.isnan(real_data)]
        fake_data = fake_data[~np.isnan(fake_data)]
        if len(real_data) < 3 or len(fake_data) < 3:
            ax.text(
                0.5,
                0.5,
                "insufficient data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
                color="#888780",
            )
            ax.set_xlabel(xlabel + unit, fontsize=8)
            return
        xmin = min(np.percentile(real_data, 1), np.percentile(fake_data, 1))
        xmax = max(np.percentile(real_data, 99), np.percentile(fake_data, 99))
        xs = np.linspace(xmin, xmax, 300)
        kde_r = stats.gaussian_kde(real_data, bw_method=0.3)(xs)
        kde_f = stats.gaussian_kde(fake_data, bw_method=0.3)(xs)
        ax.fill_between(xs, kde_r, alpha=0.25, color=C_REAL)
        ax.fill_between(xs, kde_f, alpha=0.25, color=C_FAKE)
        ax.plot(xs, kde_r, color=C_REAL, lw=1.5, label="real")
        ax.plot(xs, kde_f, color=C_FAKE, lw=1.5, label="fake")
        ks, p = stats.ks_2samp(real_data, fake_data)
        ax.set_title(f"KS={ks:.3f}  p={p:.3f}", fontsize=8, pad=3)
        ax.set_xlabel(xlabel + unit, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_yticks([])
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)

    def _draw_row(axes_row, cache, stim_name, metrics, row_label, score):
        real_seqs = cache["real_seqs"]  # (N, T, 2)
        fake_seqs = cache["fake_seqs"]  # (N, T, 2)
        real_fix_df = cache["real_fix_df"]
        fake_fix_df = cache["fake_fix_df"]
        real_sac_df = cache["real_sac_df"]
        fake_sac_df = cache["fake_sac_df"]
        img_path = cache["img_path"]

        ax_img, ax_scan, ax_fix, ax_sac, ax_dens = axes_row

        # ── col 0: stimulus image ─────────────────────────────────────────
        try:
            img = Image.open(img_path).convert("RGB")
            ax_img.imshow(img, aspect="auto")
        except Exception:
            ax_img.text(
                0.5,
                0.5,
                "image\nnot found",
                ha="center",
                va="center",
                transform=ax_img.transAxes,
                fontsize=8,
                color="#888780",
            )
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        ax_img.set_title(
            f"{row_label}\n{stim_name[:28]}",
            fontsize=8,
            loc="left",
            pad=4,
            color="#444441",
        )
        # Score badge
        auc = metrics.get("classifier_auc", {}).get("auc", float("nan"))
        badge_color = "#1D9E75" if score < 0.15 else "#E8593C"
        ax_img.text(
            0.97,
            0.03,
            f"AUC {auc:.3f}",
            transform=ax_img.transAxes,
            fontsize=7,
            ha="right",
            va="bottom",
            color="white",
            bbox=dict(boxstyle="round,pad=0.3", fc=badge_color, ec="none", alpha=0.85),
        )

        # ── col 1: scanpath overlay ───────────────────────────────────────
        n_r = min(n_scanpaths, len(real_seqs))
        n_f = min(n_scanpaths, len(fake_seqs))
        for seq in real_seqs[:n_r]:
            ax_scan.plot(seq[:, 0], seq[:, 1], lw=0.7, alpha=ALPHA_T, color=C_REAL)
        for seq in fake_seqs[:n_f]:
            ax_scan.plot(seq[:, 0], seq[:, 1], lw=0.7, alpha=ALPHA_T, color=C_FAKE)
        # Fixation circles
        if (
            real_fix_df is not None
            and len(real_fix_df) > 0
            and "cx_deg" in real_fix_df.columns
        ):
            cx = real_fix_df["cx_deg"].to_numpy()
            cy = real_fix_df["cy_deg"].to_numpy()
            dur = real_fix_df["duration"].to_numpy().astype(float)
            # normalise cx/cy to [0,1] for display
            cx_n = (cx - np.nanmin(cx)) / (np.nanmax(cx) - np.nanmin(cx) + 1e-9)
            cy_n = (cy - np.nanmin(cy)) / (np.nanmax(cy) - np.nanmin(cy) + 1e-9)
            sizes = np.clip(dur / dur.max() * 80, 5, 80)
            ax_scan.scatter(
                cx_n, cy_n, s=sizes, color=C_REAL, alpha=0.4, linewidths=0, zorder=3
            )
        if (
            fake_fix_df is not None
            and len(fake_fix_df) > 0
            and "cx_deg" in fake_fix_df.columns
        ):
            cx = fake_fix_df["cx_deg"].to_numpy()
            cy = fake_fix_df["cy_deg"].to_numpy()
            dur = fake_fix_df["duration"].to_numpy().astype(float)
            cx_n = (cx - np.nanmin(cx)) / (np.nanmax(cx) - np.nanmin(cx) + 1e-9)
            cy_n = (cy - np.nanmin(cy)) / (np.nanmax(cy) - np.nanmin(cy) + 1e-9)
            sizes = np.clip(dur / dur.max() * 80, 5, 80)
            ax_scan.scatter(
                cx_n, cy_n, s=sizes, color=C_FAKE, alpha=0.4, linewidths=0, zorder=3
            )
        ax_scan.set_xlim(0, 1)
        ax_scan.set_ylim(0, 1)
        ax_scan.set_aspect("equal")
        ax_scan.set_xticks([])
        ax_scan.set_yticks([])
        ax_scan.set_title("Scanpaths", fontsize=8, pad=3)
        for spine in ax_scan.spines.values():
            spine.set_linewidth(0.4)
            spine.set_color("#D3D1C7")

        # ── col 2: fixation duration KDE ──────────────────────────────────
        r_dur = (
            real_fix_df["duration"].to_numpy().astype(float)
            if real_fix_df is not None and len(real_fix_df) > 0
            else np.array([])
        )
        f_dur = (
            fake_fix_df["duration"].to_numpy().astype(float)
            if fake_fix_df is not None and len(fake_fix_df) > 0
            else np.array([])
        )
        _kde_plot(ax_fix, r_dur, f_dur, "Fixation duration", " (samples)")

        # ── col 3: saccade amplitude KDE ──────────────────────────────────
        r_amp = (
            real_sac_df["amplitude_deg"].to_numpy()
            if real_sac_df is not None
            and len(real_sac_df) > 0
            and "amplitude_deg" in real_sac_df.columns
            else np.array([])
        )
        f_amp = (
            fake_sac_df["amplitude_deg"].to_numpy()
            if fake_sac_df is not None
            and len(fake_sac_df) > 0
            and "amplitude_deg" in fake_sac_df.columns
            else np.array([])
        )
        _kde_plot(ax_sac, r_amp, f_amp, "Saccade amplitude", " (deg)")

        # ── col 4: fixation density difference map ────────────────────────
        d_real = _density_map(real_seqs, real_fix_df, density_grid, density_sigma)
        d_fake = _density_map(fake_seqs, fake_fix_df, density_grid, density_sigma)
        diff = d_real / d_real.sum() - d_fake / d_fake.sum()
        vmax = np.abs(diff).max()
        im = ax_dens.imshow(
            diff,
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            origin="lower",
            aspect="auto",
        )
        ax_dens.set_xticks([])
        ax_dens.set_yticks([])
        ax_dens.set_title("Density: real − fake", fontsize=8, pad=3)
        # Small colourbar
        cbar = plt.colorbar(im, ax=ax_dens, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=6)
        kl = metrics.get("fixation_density_map", {}).get("kl_divergence", float("nan"))
        ax_dens.set_xlabel(f"KL div = {kl:.3f}", fontsize=7)

    # ── Build figure ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 9), facecolor="white")
    gs = gridspec.GridSpec(
        2,
        5,
        figure=fig,
        hspace=0.45,
        wspace=0.25,
        left=0.03,
        right=0.97,
        top=0.92,
        bottom=0.06,
    )
    axes_best = [fig.add_subplot(gs[0, c]) for c in range(5)]
    axes_worst = [fig.add_subplot(gs[1, c]) for c in range(5)]

    _draw_row(
        axes_best,
        plot_cache[best_name],
        best_name,
        all_results[best_name],
        "Best match",
        best_score,
    )
    _draw_row(
        axes_worst,
        plot_cache[worst_name],
        worst_name,
        all_results[worst_name],
        "Worst match",
        worst_score,
    )

    # ── Column headers ────────────────────────────────────────────────────
    col_headers = [
        "Stimulus",
        "Scanpaths\n(real / fake)",
        "Fixation duration",
        "Saccade amplitude",
        "Fixation density\ndifference",
    ]
    for ax, hdr in zip(axes_best, col_headers):
        ax.annotate(
            hdr,
            xy=(0.5, 1.0),
            xycoords="axes fraction",
            xytext=(0, 28),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight=500,
            color="#444441",
        )

    # ── Legend ────────────────────────────────────────────────────────────
    import matplotlib.patches as mpatches

    legend_handles = [
        mpatches.Patch(color=C_REAL, alpha=0.7, label="Real data"),
        mpatches.Patch(color=C_FAKE, alpha=0.7, label="Generated (fake)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=9,
        frameon=True,
        framealpha=0.9,
        edgecolor="#D3D1C7",
        bbox_to_anchor=(0.97, 0.99),
    )

    # ── Figure title ──────────────────────────────────────────────────────
    section_label = section.replace("_", " ")
    fig.suptitle(
        f"Gaze generation evaluation — best vs worst matching stimulus\n"
        f"Ranked by {section_label} {key}  "
        f"(best: {best_score:.3f}  worst: {worst_score:.3f})",
        fontsize=10,
        y=0.99,
        va="top",
        color="#2C2C2A",
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[plot] saved → {out_path}")


# ---------------------------------------------------------------------------
# Multi-generator evaluation (comparison mode)
# ---------------------------------------------------------------------------


def _save_comparison_table(all_gen_results: Dict[str, Dict], out_path: Path) -> None:
    """Write a side-by-side mean±std comparison table across all generators."""
    key_metrics = [
        ("fixation_duration", "ks_stat", "Fix duration KS "),
        ("fixation_duration", "p_value", "Fix duration p   "),
        ("saccade_amplitude", "ks_stat", "Sac amplitude KS "),
        ("main_sequence", "fake_r", "Main sequence r  "),
        ("fixation_density_map", "kl_divergence", "Fix density KL   "),
        ("saccade_direction", "kl_divergence", "Sac direction KL "),
        ("classifier_auc", "auc", "Classifier AUC   "),
    ]

    gen_names = list(all_gen_results.keys())
    col_w = max(20, max(len(n) for n in gen_names) + 4)
    header = f"  {'Metric':<22}" + "".join(f"  {n:^{col_w}}" for n in gen_names)
    sep = "-" * len(header)

    lines = [
        "=" * len(header),
        "GENERATOR COMPARISON",
        "=" * len(header),
        "",
        header,
        sep,
    ]

    for section, key, label in key_metrics:
        row = f"  {label:<22}"
        for gen_name in gen_names:
            per_stim = all_gen_results[gen_name]
            vals = [
                m[section][key]
                for m in per_stim.values()
                if isinstance(m, dict)
                and section in m
                and isinstance(m[section], dict)
                and isinstance(m[section].get(key), (int, float))
            ]
            arr = np.array([v for v in vals if not np.isnan(v)])
            cell = f"{arr.mean():.4f} ±{arr.std():.4f}" if len(arr) else "n/a"
            row += f"  {cell:^{col_w}}"
        lines.append(row)

    lines += ["", "=" * len(header)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"[compare] table → {out_path}")


def run_multi_evaluation(
    generators: List[SequenceGenerator],
    dataset_name: str,
    root: str,
    out_dir: str,
    **kwargs,
) -> Dict[str, Dict]:
    """
    Run run_evaluation for every generator in ``generators`` and write a
    cross-generator comparison table to ``out_dir/comparison.txt``.

    Returns
    -------
    dict mapping generator.name → all_results dict from run_evaluation
    """
    all_gen_results: Dict[str, Dict] = {}
    for gen in generators:
        results = run_evaluation(gen, dataset_name, root, out_dir, **kwargs)
        all_gen_results[gen.name] = results

    _save_comparison_table(all_gen_results, Path(out_dir) / "comparison.txt")
    return all_gen_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_config(path: str) -> dict:
    """
    Load a JSON config file and return a flat dict of CLI arg overrides.

    Keys should match CLI argument names, e.g.::

        {
            "dataset":       "mcfw-gaze",
            "root":          "/data",
            "out_dir":       "eval_results",
            "checkpoint":    "/logs/runs/trial_0019/checkpoints/best_model.pt",
            "n_generate":    100,
            "gen_len":       256,
            "vel_threshold": 30.0
        }

    Unlike the training scripts there is no ``model_config`` to extract —
    model weights and architecture are read directly from the checkpoint.
    CLI flags always override file values; the file only sets defaults.
    """
    return json.loads(Path(path).read_text())


def _add_common_args(sp) -> None:
    """Attach dataset / generation / event-detection flags to a subparser."""
    sp.add_argument(
        "--dataset", required=True, help="pymovements dataset name, e.g. mcfw-gaze"
    )
    sp.add_argument("--root", required=True, help="Root directory for pymovements data")
    sp.add_argument(
        "--out_dir",
        default="eval_results",
        help="Output root directory (generator sub-dir added automatically)",
    )
    # Subset
    sp.add_argument("--subjects", nargs="*", default=None)
    sp.add_argument("--stimuli", nargs="*", default=None)
    sp.add_argument("--trial_ids", nargs="*", default=None)
    # Generation
    sp.add_argument(
        "--n_generate", type=int, default=50, help="Sequences to generate per stimulus"
    )
    sp.add_argument("--seed_len", type=int, default=10)
    sp.add_argument("--gen_len", type=int, default=128)
    # Event detection
    sp.add_argument(
        "--vel_threshold",
        type=float,
        default=30.0,
        help="IVT velocity threshold in deg/s",
    )
    sp.add_argument(
        "--dispersion_threshold",
        type=float,
        default=1.0,
        help="IDT velocity threshold in deg/s",
    )
    sp.add_argument(
        "--min_fix_dur",
        type=int,
        default=10,
        help="Minimum fixation duration in samples",
    )
    sp.add_argument(
        "--vel_method",
        default="fivepoint",
        choices=["fivepoint", "preceding", "smooth"],
    )
    sp.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")


def _build_parser() -> tuple:
    """Build the argument parser. Returns ``(parser, subparsers_dict)``."""
    p = argparse.ArgumentParser(
        description="Evaluate gaze generators against real pymovements data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Config file (--config):
  Pass a JSON file with default values for any CLI argument so you don't
  have to retype dataset paths, generation settings, or checkpoint locations.
  CLI flags always override file values.

  Useful keys:
    dataset, root, out_dir, checkpoint, n_generate, gen_len, seed_len,
    vel_threshold, min_fix_dur, vel_method, device, temperature, label

  Example config (eval_config.json):
    {
        "dataset":    "mcfw-gaze",
        "root":       "/data",
        "checkpoint": "/logs/runs/trial_0019/checkpoints/best_model.pt",
        "n_generate": 100,
        "gen_len":    256
    }

  Usage:
    python evaluate_model.py model --config eval_config.json
    python evaluate_model.py model --config eval_config.json --n_generate 200
    python evaluate_model.py compare --config eval_config.json --also_synthetic
""",
    )
    p.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="JSON config file.  Keys set CLI defaults; explicit flags override.",
    )
    sub = p.add_subparsers(dest="mode", required=True)

    # ── model ──────────────────────────────────────────────────────────────
    model_p = sub.add_parser("model", help="Evaluate a trained GMM checkpoint")
    model_p.add_argument(
        "--checkpoint",
        default=None,
        help="Path to best_model.pt (required unless set via --config)",
    )
    model_p.add_argument("--temperature", type=float, default=1.0)
    model_p.add_argument(
        "--label", default=None, help="Override generator name in output paths"
    )
    _add_common_args(model_p)

    # ── categorical ────────────────────────────────────────────────────────
    cat_p = sub.add_parser(
        "categorical", help="Evaluate a trained categorical checkpoint"
    )
    cat_p.add_argument(
        "--checkpoint",
        default=None,
        help="Path to best_model.pt (required unless set via --config)",
    )
    cat_p.add_argument("--temperature", type=float, default=1.0)
    cat_p.add_argument("--label", default=None)
    _add_common_args(cat_p)

    # ── synthetic ──────────────────────────────────────────────────────────
    syn_p = sub.add_parser(
        "synthetic", help="Synthetic step-function baseline (no model needed)"
    )
    syn_p.add_argument("--noise", type=float, default=5)
    syn_p.add_argument("--values_spread", type=float, default=0.7)
    syn_p.add_argument("--fix_dur_mean", type=float, default=250.0)
    syn_p.add_argument("--fix_dur_std", type=float, default=80.0)
    syn_p.add_argument("--sac_dur_mean", type=float, default=40.0)
    syn_p.add_argument("--sac_dur_std", type=float, default=15.0)
    syn_p.add_argument("--seed", type=int, default=42)
    syn_p.add_argument("--label", default=None)
    _add_common_args(syn_p)

    # ── compare ────────────────────────────────────────────────────────────
    cmp_p = sub.add_parser(
        "compare", help="Side-by-side comparison of multiple generators"
    )
    cmp_p.add_argument(
        "--checkpoint", default=None, help="GMM checkpoint path (optional)"
    )
    cmp_p.add_argument(
        "--cat_checkpoint", default=None, help="Categorical checkpoint path (optional)"
    )
    cmp_p.add_argument(
        "--also_synthetic",
        action="store_true",
        help="Also include the synthetic step-function baseline",
    )
    cmp_p.add_argument("--temperature", type=float, default=1.0)
    _add_common_args(cmp_p)

    return p, {
        "model": model_p,
        "categorical": cat_p,
        "synthetic": syn_p,
        "compare": cmp_p,
    }


def _build_subset(args) -> Optional[Dict]:
    subset: Dict = {}
    if getattr(args, "subjects", None):
        subset["subject_id"] = args.subjects
    if getattr(args, "stimuli", None):
        subset["stimulus_id"] = args.stimuli
    if getattr(args, "trial_ids", None):
        subset["trial_id"] = args.trial_ids
    return subset or None


def test():
    """Quick smoke-test with the synthetic model generator."""
    syn = SyntheticGenerator()

    run_evaluation(
        generator=syn,
        dataset_name="GGTG",
        root=r"C:\Users\saphi\PycharmProjects\thesis\data",
        out_dir=r"C:\Users\saphi\PycharmProjects\thesis\eval_results",
        # subset={"subject_id": ["P01"]},
        n_generate=20,
        seed_len=32,
        gen_len=2000,  # for each subject's recording of  stimulus, extracts normalized (x,y) gaze coordinates as sliding windows of length gen_len.
        dispersion_threshold=1.0,
        min_fix_duration=90,
        vel_method="fivepoint",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


def main():
    p, subparsers = _build_parser()

    # ── Config file: scan sys.argv directly for --config ──────────────────
    # We scan rather than use parse_known_args() to avoid conflicts with
    # required-seeming args in subparsers (e.g. --checkpoint which is now
    # optional at the parser level and validated manually below).
    _argv = sys.argv[1:]
    cfg: dict = {}
    for i, arg in enumerate(_argv):
        if arg == "--config" and i + 1 < len(_argv):
            cfg_path = _argv[i + 1]
            cfg = _load_config(cfg_path)
            print(f"[config] loading {cfg_path}")
            break

    if cfg:
        # Identify the subcommand so we only apply relevant defaults —
        # e.g. don't set synthetic-only keys on the model subparser.
        _modes = {"model", "categorical", "synthetic", "compare"}
        mode_from_argv = next((a for a in _argv if a in _modes), None)
        targets = (
            [subparsers[mode_from_argv]]
            if mode_from_argv in subparsers
            else list(subparsers.values())
        )
        total_applied = 0
        for sp in targets:
            valid = {a.dest for a in sp._actions}
            applied = {k: v for k, v in cfg.items() if k in valid}
            if applied:
                sp.set_defaults(**applied)
                total_applied += len(applied)
        print(f"[config] applied {total_applied} defaults from file")

    args = p.parse_args()

    # ── Manual validation for args that the config file may satisfy ───────
    if args.mode in ("model", "categorical") and not args.checkpoint:
        p.error(
            f"{args.mode} mode requires --checkpoint (or set 'checkpoint' in --config)"
        )

    subset = _build_subset(args)

    common = dict(
        dataset_name=args.dataset,
        root=args.root,
        out_dir=args.out_dir,
        subset=subset,
        n_generate=args.n_generate,
        seed_len=args.seed_len,
        gen_len=args.gen_len,
        vel_threshold=args.vel_threshold,
        min_fix_duration=args.min_fix_dur,
        vel_method=args.vel_method,
        device=args.device,
    )

    if args.mode == "model":
        gen = GMMModelGenerator(
            args.checkpoint, args.temperature, args.device, args.label
        )
        run_evaluation(gen, **common)

    elif args.mode == "categorical":
        gen = CategoricalModelGenerator(
            args.checkpoint, args.temperature, args.device, args.label
        )
        run_evaluation(gen, **common)

    elif args.mode == "synthetic":
        gen = SyntheticGenerator(
            fix_dur_mean_ms=args.fix_dur_mean,
            fix_dur_std_ms=args.fix_dur_std,
            sac_dur_mean_ms=args.sac_dur_mean,
            sac_dur_std_ms=args.sac_dur_std,
            noise=args.noise,
            values_spread=args.values_spread,
            seed=args.seed,
            label=args.label,
        )
        run_evaluation(gen, **common)

    elif args.mode == "compare":
        generators: List[SequenceGenerator] = []
        if args.checkpoint:
            generators.append(
                GMMModelGenerator(args.checkpoint, args.temperature, args.device)
            )
        if args.cat_checkpoint:
            generators.append(
                CategoricalModelGenerator(
                    args.cat_checkpoint, args.temperature, args.device
                )
            )
        if args.also_synthetic:
            generators.append(SyntheticGenerator())
        if not generators:
            raise ValueError(
                "compare mode needs at least one of --checkpoint, "
                "--cat_checkpoint, or --also_synthetic"
            )
        run_multi_evaluation(generators, **common)


if __name__ == "__main__":
    test()
