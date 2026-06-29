"""
evaluate_model.py

Loads a pymovements dataset, iterates over gaze objects per stimulus,
generates matching synthetic sequences from a *sequence generator*, and
computes all GazeEvaluator metrics — both per-stimulus and aggregated.

Built-in generators
───────────────────
  GMMModelGenerator              load a trained GMM checkpoint (kaamba.py)
  SyntheticGenerator             step-function synthetic gaze baseline
  TrainingDistributionGenerator  i.i.d. samples from training-data distribution

Usage
─────
  # GMM model only
  python evaluate_model.py model \
      --checkpoint /path/best_model.pt --dataset mcfw-gaze --root /data

  # Synthetic baseline only
  python evaluate_model.py synthetic \
      --dataset mcfw-gaze --root /data

  # Empirical (training-distribution) baseline only
  python evaluate_model.py empirical \
      --dataset mcfw-gaze --root /data \
      --train_stimuli 1 2 3

  # Model + one or both baselines as extra conditions
  python evaluate_model.py model \
      --checkpoint /path/best_model.pt \
      --also_synthetic --also_empirical \
      --train_stimuli 1 2 3 \
      --dataset mcfw-gaze --root /data

  # Side-by-side: model vs synthetic vs empirical (shorthand)
  python evaluate_model.py compare \
      --checkpoint /path/model.pt --baseline both \
      --train_stimuli 1 2 3 \
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
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import polars as pl
import pymovements as pm
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
# NOTE: build_gaze_predictor is imported lazily inside GMMModelGenerator.__init__
# to avoid loading mamba_ssm/triton (which require a CUDA driver) in non-model modes.
from kaamba.utils.baselines import (
    SequenceGenerator,
    SyntheticGenerator,
    TrainingDistributionGenerator,
)
from kaamba.utils.eval_plots import (
    plot_aggregate_metrics,
    plot_best_worst_comparison,
    plot_per_stimulus_metrics,
    plot_scanpath_overview,
)
from kaamba.utils.eval_report import (
    aggregate_results,
    build_eval_report,
    save_comparison_table,
)

# from kaamba.utils.gaze_eval import generate_sequences
from kaamba.utils.gaze_preprocessing import GazePreprocessor


# ---------------------------------------------------------------------------
# Model generators
# ---------------------------------------------------------------------------

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
        from kaamba.net.models.kaamba import (
            build_gaze_predictor,
        )  # lazy: needs CUDA/triton

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

    def _generate_sequences(
        self,
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

    def generate(self, img_path, n, gen_len, seed_len, experiment, device=None):
        device = device or self.device
        img_tensor = _load_image_tensor(img_path, device)  # (1, 3, 224, 224)
        imgs_batch = img_tensor.expand(n, -1, -1, -1)
        with torch.no_grad():
            return self._generate_sequences(
                model=self.model,
                images=imgs_batch,
                seed_len=seed_len,
                gen_len=gen_len,
                temperature=self.temperature,
                device=device,
            )  # (N, gen_len, 2)


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


def _fix_df_from_events(
    ev_frame: pl.DataFrame, pos_arr: np.ndarray, time_arr: np.ndarray
) -> pl.DataFrame:
    """
    Filter fixation events from a preprocessed events DataFrame and append
    centroid columns ``cx_deg`` / ``cy_deg`` (mean deg position per fixation).

    ``pos_arr`` must be (T, 2) in degrees of visual angle — taken from
    ``gaze.samples["position"]`` after ``pix2deg()``.
    ``time_arr`` must be (T,) with the same time unit as event onset/offset.
    """
    if ev_frame is None or len(ev_frame) == 0:
        return _EMPTY_FIX
    fix = ev_frame.filter(pl.col("name") == "fixation")
    if len(fix) == 0:
        return _EMPTY_FIX
    cx_list, cy_list = [], []
    for row in fix.iter_rows(named=True):
        i0 = int(np.searchsorted(time_arr, row["onset"]))
        i1 = int(np.searchsorted(time_arr, row["offset"], side="right"))
        seg = pos_arr[i0:i1]
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


def _sac_df_from_events(
    ev_frame: pl.DataFrame, pos_arr: np.ndarray, time_arr: np.ndarray
) -> pl.DataFrame:
    """
    Filter saccade events, append saccade direction ``angle_rad``, and rename
    ``amplitude`` → ``amplitude_deg``  /  ``peak_velocity`` → ``peak_vel_deg_s``
    to match the ``evaluate_stimulus()`` column expectations.

    ``amplitude`` and ``peak_velocity`` must already be present — they are
    added by ``compute_event_properties(["amplitude", "peak_velocity", ...])``.
    ``time_arr`` must be (T,) with the same time unit as event onset/offset.
    """
    if ev_frame is None or len(ev_frame) == 0:
        return _EMPTY_SAC
    sac = ev_frame.filter(pl.col("name") == "saccade")
    if len(sac) == 0:
        return _EMPTY_SAC
    angle_list = []
    for row in sac.iter_rows(named=True):
        i0 = int(np.searchsorted(time_arr, row["onset"]))
        i1 = int(np.searchsorted(time_arr, row["offset"], side="right"))
        seg = pos_arr[i0:i1]
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


def run_evaluation(
    generator: SequenceGenerator,
    dataset_name: str,
    root: str,
    out_dir: str,
    subset: Optional[Dict] = None,
    n_generate: int = 50,
    seed_len: int = 10,
    gen_len: int = 128,
    vel_threshold: float = 30.0,
    min_fix_duration: int = 100,
    min_sac_duration: int = 30,
    dispersion_threshold: float = 1.0,
    vel_method: str = "fivepoint",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    extra_generators: Optional[List[SequenceGenerator]] = None,
    per_stimulus_plots: bool = False,
    scanpath_overview: bool = False,
    rng_seed: int = 42,
) -> Dict:
    """
    Main evaluation loop.

    Accepts a primary SequenceGenerator (the model) and an optional list of
    ``extra_generators`` (e.g. a SyntheticGenerator used as a baseline).

    For each stimulus:
      1. Load real gaze sequences and run event detection.
      2. Generate sequences from ``generator`` (primary / model).
      3. Optionally generate from each ``extra_generators`` entry.
      4. Compute all metrics (primary generator vs real).
      5. Save per-stimulus JSON + aggregate report.
      6. Produce per-stimulus distribution plots and a best/worst figure.

    The condition-keyed plot cache written to disk has the structure::

        {stim_name: {
            "img_path": Path,
            "real":          {"seqs", "fix_df", "sac_df"},
            generator.name:  {"seqs", "fix_df", "sac_df"},
            extra.name:      {"seqs", "fix_df", "sac_df"},   # one per extra_generator
        }}
    """
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)

    out_dir = Path(out_dir) / generator.name
    stim_dir = out_dir / "per_stimulus"
    stim_dir.mkdir(parents=True, exist_ok=True)

    # ── Load dataset ──────────────────────────────────────────────────────
    print(f"\n[eval] Generator : {generator.name}")
    print(f"[eval] Loading dataset: {dataset_name}")
    dataset_paths = pm.DatasetPaths(root=root)
    dataset = pm.Dataset(dataset_name, path=dataset_paths)
    dataset.scan()

    if dataset_name == "GGTG":
        # split_gaze_data must run before stimulus-based filtering is possible
        # participant = subset.get("participant", None) if subset else None
        # dataset.load(subset={"subject_id": participant}) if participant else dataset.load()
        dataset.load()  # todo adapt to load only certain participants

        dataset.split_gaze_data(by="stimulus")
        if subset and "stimulus" in subset:
            keep = set(subset["stimulus"])
            dataset.gaze = [
                g for g in dataset.gaze if g.metadata.get("stimulus") in keep
            ]
    else:
        dataset.load(subset=subset)

    print(f"[eval] Loaded {len(dataset.gaze)} gaze files")

    # ── Screen info from first gaze object ───────────────────────────────
    first_gaze = dataset.gaze[0]
    screen = first_gaze.experiment.screen
    scr_w_px = screen.width_px
    scr_h_px = screen.height_px
    sampling_rate = first_gaze.experiment.sampling_rate

    # ── Preprocess entire dataset with pymovements built-ins ──────────────
    print(
        "[eval] Preprocessing: pix2deg → pos2vel → IDT → microsaccades → "
        "event properties …"
    )
    preprocessor = GazePreprocessor(
        vel_threshold=vel_threshold,
        dispersion_threshold=dispersion_threshold,
        min_fix_duration=min_fix_duration,
        min_sac_duration=min_sac_duration,
        vel_method=vel_method,
    )
    preprocessor.apply_dataset(dataset, dataset_name)
    # dataset.save_preprocessed()
    # dataset.save_events()
    print(f"[eval] Preprocessing complete — {len(dataset.gaze)} recordings ready")

    # ── Group gaze objects + event frames by stimulus ─────────────────────
    from collections import defaultdict

    by_stimulus = defaultdict(list)
    for gaze, ev_frame in zip(dataset.gaze, dataset.events):
        stim = gaze.metadata.get("stimulus", "unknown")
        by_stimulus[stim].append((gaze, ev_frame))

    print(f"[eval] {len(by_stimulus)} unique stimuli\n")

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
    extra_all_results: Dict[str, Dict] = {
        eg.name: {} for eg in (extra_generators or [])
    }
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
                time_arr = gaze.samples["time"].to_numpy()  # (T,) timestamps
                norm_arr = np.column_stack(
                    [px_raw[:, 0] / scr_w_px, px_raw[:, 1] / scr_h_px]
                )
                fix_df = _fix_df_from_events(ev_frame.frame, pos_arr, time_arr)
                sac_df = _sac_df_from_events(ev_frame.frame, pos_arr, time_arr)
            except Exception as e:
                print(
                    f"  [warn] {stim_name} / {gaze.metadata.get('subject_id')} "
                    f"event extraction failed: {e}"
                )
                continue

            # whole recording per subject — pad to longest later
            real_norm_seqs.append(norm_arr)

            # sliding-window alternative (uncomment to use instead):
            # step = gen_len  # no overlap
            # # step = max(1, gen_len // 2)  # 50 % overlap — richer real pool
            # for start in range(0, len(norm_arr) - gen_len + 1, step):
            #     real_norm_seqs.append(norm_arr[start : start + gen_len])

            all_real_fix.append(fix_df)
            all_real_sac.append(sac_df)

        if len(real_norm_seqs) == 0:
            print(f"  [skip] {stim_name}: no valid real sequences")
            continue

        # pad recordings to the longest one with NaN; the classifier in
        # evaluate_stimulus drops NaN rows via the isfinite() mask
        max_len = max(len(s) for s in real_norm_seqs)
        real_arr = np.full((len(real_norm_seqs), max_len, 2), np.nan, dtype=np.float32)
        for i, s in enumerate(real_norm_seqs):
            real_arr[i, : len(s)] = s

        # sliding-window alternative (uncomment to use instead):
        # real_arr = np.stack(real_norm_seqs)  # requires all seqs same length
        real_fix_df = pl.concat(all_real_fix) if all_real_fix else pl.DataFrame()
        real_sac_df = pl.concat(all_real_sac) if all_real_sac else pl.DataFrame()

        # ── Load stimulus image via fileinfo path ────────────────────────
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
            px_vals = seq * np.array([scr_w_px, scr_h_px], dtype=float)  # denormalize
            g_fake = pm.Gaze(
                pl.DataFrame(
                    {
                        "x_pix": px_vals[:, 0],
                        "y_pix": px_vals[:, 1],
                        "time": [(x + 1) * 8 for x in range(len(px_vals))]
                        if sampling_rate == 120
                        else [x + 1 for x in range(len(px_vals))],
                    }
                ),
                pixel_columns=["x_pix", "y_pix"],
                experiment=first_gaze.experiment,
            )
            try:
                preprocessor.apply_gaze(g_fake)
                pos_f = np.stack(g_fake.samples["position"].to_numpy())
                time_f = g_fake.samples["time"].to_numpy()
                fixations = g_fake.events.fixations
                saccades = g_fake.events.saccades
                all_fake_fix.append(_fix_df_from_events(fixations, pos_f, time_f))
                all_fake_sac.append(_sac_df_from_events(saccades, pos_f, time_f))
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

        # ── Generate from extra generators (e.g. synthetic baseline) ────────
        extra_cond_data: Dict[str, dict] = {}
        for extra_gen in extra_generators or []:
            try:
                extra_norm = extra_gen.generate(
                    img_path=img_path,
                    n=n_generate,
                    gen_len=gen_len,
                    seed_len=seed_len,
                    experiment=first_gaze.experiment,
                    device=device,
                )
                ex_fix_list, ex_sac_list = [], []
                for seq in extra_norm:
                    px_vals = seq * np.array([scr_w_px, scr_h_px], dtype=float)
                    g_ex = pm.Gaze(
                        pl.DataFrame(
                            {
                                "x_pix": px_vals[:, 0],
                                "y_pix": px_vals[:, 1],
                                "time": [(x + 1) * 8 for x in range(len(px_vals))]
                                if sampling_rate == 120
                                else [x + 1 for x in range(len(px_vals))],
                            }
                        ),
                        pixel_columns=["x_pix", "y_pix"],
                        experiment=first_gaze.experiment,
                    )
                    try:
                        preprocessor.apply_gaze(g_ex)
                        pos_ex = np.stack(g_ex.samples["position"].to_numpy())
                        time_ex = g_ex.samples["time"].to_numpy()
                        ex_fix_list.append(
                            _fix_df_from_events(g_ex.events.fixations, pos_ex, time_ex)
                        )
                        ex_sac_list.append(
                            _sac_df_from_events(g_ex.events.saccades, pos_ex, time_ex)
                        )
                    except Exception as e:
                        tqdm.write(
                            f"  [warn] extra gen '{extra_gen.name}' event detection failed: {e}"
                        )
                extra_fix_df = pl.concat(ex_fix_list) if ex_fix_list else pl.DataFrame()
                extra_sac_df = pl.concat(ex_sac_list) if ex_sac_list else pl.DataFrame()
                extra_cond_data[extra_gen.name] = {
                    "seqs": extra_norm,
                    "fix_df": extra_fix_df,
                    "sac_df": extra_sac_df,
                }
                ex_metrics = evaluate_stimulus(
                    real_seqs=real_arr,
                    fake_seqs=extra_norm,
                    real_fix_df=real_fix_df,
                    real_sac_df=real_sac_df,
                    fake_fix_df=extra_fix_df,
                    fake_sac_df=extra_sac_df,
                )
                ex_metrics["stimulus"] = stim_name
                extra_all_results[extra_gen.name][stim_name] = ex_metrics
            except Exception as e:
                tqdm.write(
                    f"  [warn] extra generator '{extra_gen.name}' failed for {stim_name}: {e}"
                )

        # ── Build condition-keyed plot cache ──────────────────────────────
        _plot_cache[stim_name] = {
            "img_path": img_path,
            "real": {
                "seqs": real_arr,
                "fix_df": real_fix_df,
                "sac_df": real_sac_df,
            },
            generator.name: {
                "seqs": fake_norm,
                "fix_df": fake_fix_df,
                "sac_df": fake_sac_df,
            },
            **extra_cond_data,
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
    aggregate = aggregate_results(all_results)
    (out_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2))

    # ── Human-readable report ─────────────────────────────────────────────
    report = build_eval_report(all_results, aggregate, timing_total)
    report_path = out_dir / "eval_report.txt"
    report_path.write_text(report)
    print(report)
    print(f"\n[eval] Results saved to {out_dir}")

    # ── Comparison table (primary + all extra generators) ─────────────────
    if extra_generators:
        combined = {generator.name: all_results}
        for eg in extra_generators:
            if extra_all_results.get(eg.name):
                combined[eg.name] = extra_all_results[eg.name]
        baseline_names = "_vs_".join(eg.name for eg in extra_generators)
        comparison_path = out_dir / f"comparison_vs_{baseline_names}.txt"
        save_comparison_table(combined, comparison_path)
        print(f"[eval] Comparison table → {comparison_path}")

    # ── Aggregate metric plots (always generated) ─────────────────────────
    plot_aggregate_metrics(
        plot_cache=_plot_cache,
        out_dir=out_dir / "aggregate_plots",
        sr=sampling_rate,
    )

    # ── Best vs worst comparison (always generated) ───────────────────────
    if len(all_results) >= 2:
        plot_best_worst_comparison(
            all_results=all_results,
            plot_cache=_plot_cache,
            out_path=out_dir / "comparison_best_worst.png",
            score_metric=("classifier_auc", "auc"),
            primary_condition=generator.name,
        )

    # ── Per-stimulus metric plots (optional) ──────────────────────────────
    if per_stimulus_plots:
        plot_per_stimulus_metrics(
            plot_cache=_plot_cache,
            out_dir=out_dir / "per_stimulus_plots",
            sr=sampling_rate,
        )

    # ── Scanpath overview (optional) ──────────────────────────────────────
    if scanpath_overview:
        plot_scanpath_overview(
            plot_cache=_plot_cache,
            out_path=out_dir / "scanpath_overview.png",
            n_cols=4,
            n_scanpaths=6,
        )

    return all_results


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

    gen_names = "_vs_".join(g.name for g in generators)
    comparison_path = Path(out_dir) / f"comparison_{gen_names}.txt"
    save_comparison_table(all_gen_results, comparison_path)
    print(f"[eval] Comparison table → {comparison_path}")
    return all_gen_results


def _load_config(path: str) -> dict:
    """
    Load a JSON config file and return a flat dict of CLI arg overrides.

    Keys should match CLI argument names, e.g.::

        {
            "dataset":       "mcfw-gaze",
            "root":          "/data",
            "out_dir":       "/path/to/eval_results",  # default: <root>/../eval_results
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
    """Attach dataset / generation / event-detection flags dataset name (default: mcfw-gaze)")"""
    # Dataset

    sp.add_argument(
        "--root",
        default=None,
        help="Path to root directory, where pymovement data is located (or set via --config)",
    )

    sp.add_argument("--dataset", default="mcfw-gaze", help="pymovements movements data")
    sp.add_argument(
        "--out_dir",
        default=None,
        help="Output root (default: <root>/../eval_results, i.e. sibling of the data folder). "
        "A generator sub-dir is added automatically.",
    )
    # Subset filters
    sp.add_argument(
        "--subjects", nargs="*", default=None, help="Limit to specific subject IDs"
    )
    sp.add_argument(
        "--stimuli",
        nargs="*",
        default=[
            "88",
            "89",
            "90",
            "91",
            "92",
            "93",
            "94",
            "95",
            "96",
            "97",
            "98",
            "99",
        ],
        help="Limit to specific stimulus IDs",
    )
    sp.add_argument(
        "--trial_ids", nargs="*", default=None, help="Limit to specific trial IDs"
    )
    # Generation
    sp.add_argument(
        "--n_generate",
        type=int,
        default=45,
        help="Sequences to generate per stimulus (default: 45)",
    )
    sp.add_argument("--seed_len", type=int, default=10)
    sp.add_argument("--gen_len", type=int, default=580)
    # Event detection
    sp.add_argument(
        "--vel_threshold",
        type=float,
        default=30.0,
        help="IVT velocity threshold in deg/s (default: 30)",
    )
    sp.add_argument(
        "--dispersion_threshold",
        type=float,
        default=1.0,
        help="IDT dispersion threshold in deg (default: 1.0)",
    )
    sp.add_argument(
        "--min_fix_dur",
        type=int,
        default=96,
        help="Minimum fixation duration in samples (default: 96)",
    )
    sp.add_argument(
        "--min_sac_dur",
        type=int,
        default=16,
        help="Minimum saccade duration in samples (default: 16)",
    )
    sp.add_argument(
        "--vel_method",
        default="fivepoint",
        choices=["fivepoint", "preceding", "smooth"],
    )
    sp.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Optional plots
    sp.add_argument(
        "--per_stimulus_plots",
        action="store_true",
        help="Also generate individual metric plots per stimulus "
        "(saved under per_stimulus_plots/)",
    )
    sp.add_argument(
        "--scanpath_overview",
        default=True,
        help="Also generate the tiled scanpath overview figure",
    )
    sp.add_argument(
        "--rng_seed",
        type=int,
        default=42,
        help="Global RNG seed for reproducible generation (default: 42)",
    )


def _build_parser() -> tuple:
    """Build the argument parser.  Returns ``(parser, subparsers_dict)``."""
    p = argparse.ArgumentParser(
        description="Evaluate gaze generators against real pymovements data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes
─────
  model      Evaluate a trained GMM checkpoint.
             Add --also_synthetic and/or --also_empirical for extra baselines.
  synthetic  Evaluate the step-function baseline only (no model needed).
  empirical  Evaluate the training-distribution baseline only (no model needed).
             Pass --train_stimuli to specify which recordings form the training set.
  compare    GMM checkpoint + one or both baselines.
             --baseline synthetic|empirical|both  (default: synthetic)

Config file (--config)
──────────────────────
  JSON file of default values for any CLI argument.  Explicit flags override.

  Useful keys: dataset, root, out_dir, checkpoint, n_generate, gen_len,
               seed_len, vel_threshold, min_fix_dur, device, temperature

  Example:
    {
        "dataset":    "mcfw-gaze",
        "root":       "/data",
        "checkpoint": "/logs/runs/best_model.pt",
        "n_generate": 100,
        "gen_len":    256
    }

  Usage:
    python evaluate_model.py model     --config eval_config.json
    python evaluate_model.py model     --config eval_config.json --also_synthetic
    python evaluate_model.py model     --config eval_config.json --also_empirical --train_stimuli P01 P02
    python evaluate_model.py synthetic --config eval_config.json
    python evaluate_model.py empirical --config eval_config.json --train_stimuli P01 P02
    python evaluate_model.py compare   --config eval_config.json --baseline both
""",
    )
    p.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="JSON config file — keys set CLI defaults",
    )
    sub = p.add_subparsers(dest="mode", required=True)

    # ── model ──────────────────────────────────────────────────────────────
    model_p = sub.add_parser("model", help="Evaluate a trained GMM checkpoint")
    model_p.add_argument(
        "--checkpoint", default=None, help="Path to best_model.pt (or set via --config)"
    )
    model_p.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)",
    )
    model_p.add_argument(
        "--label", default=None, help="Override the generator name used in output paths"
    )
    model_p.add_argument(
        "--also_synthetic",
        action="store_true",
        help="Also run the step-function synthetic baseline as an extra condition",
    )
    model_p.add_argument(
        "--also_empirical",
        action="store_true",
        help="Also run the training-distribution empirical baseline as an extra condition",
    )
    model_p.add_argument(
        "--train_stimuli",
        nargs="*",
        default=None,
        help="Stimulus IDs constituting the training set (for --also_empirical). "
        "None → all stimuli.",
    )
    _add_common_args(model_p)

    # ── synthetic ──────────────────────────────────────────────────────────
    syn_p = sub.add_parser(
        "synthetic", help="Synthetic step-function baseline (no model needed)"
    )
    syn_p.add_argument(
        "--label", default=None, help="Override the generator name used in output paths"
    )
    _add_common_args(syn_p)

    # ── empirical ──────────────────────────────────────────────────────────
    emp_p = sub.add_parser(
        "empirical",
        help="Training-distribution baseline — samples i.i.d. from the observed "
        "coordinate distribution of the training subset",
    )
    emp_p.add_argument(
        "--train_stimuli",
        nargs="*",
        default=None,
        help="Stimulus IDs constituting the training set. None → all stimuli.",
    )
    emp_p.add_argument(
        "--label", default=None, help="Override the generator name used in output paths"
    )
    _add_common_args(emp_p)

    # ── compare ────────────────────────────────────────────────────────────
    cmp_p = sub.add_parser(
        "compare",
        help="GMM checkpoint + one or both baselines as extra conditions",
    )
    cmp_p.add_argument(
        "--checkpoint", default=None, help="Path to best_model.pt (or set via --config)"
    )
    cmp_p.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Sampling temperature (default: 0.5)",
    )
    cmp_p.add_argument(
        "--baseline",
        choices=["synthetic", "empirical", "both"],
        default="synthetic",
        help="Which baseline(s) to run alongside the model (default: synthetic)",
    )
    cmp_p.add_argument(
        "--train_stimuli",
        nargs="*",
        default=None,
        help="Stimulus IDs for the empirical baseline training set. None → all stimuli.",
    )
    _add_common_args(cmp_p)

    return p, {
        "model": model_p,
        "synthetic": syn_p,
        "empirical": emp_p,
        "compare": cmp_p,
    }


def _build_subset(args) -> Optional[Dict]:
    subset: Dict = {}
    if getattr(args, "subjects", None):
        subset["subject_id"] = args.subjects
    if getattr(args, "stimuli", None):
        subset["stimulus"] = args.stimuli
    if getattr(args, "trial_ids", None):
        subset["trial_id"] = args.trial_ids
    return subset or None


def test():
    """Quick smoke-test: synthetic baseline + empirical baseline side-by-side."""
    from kaamba.utils.baselines import test_baselines

    GGTG_TRAIN_STIM = [
        "blackout-neg.difficulty",
        "blackout-neg.interest",
        "blackout-neg.naturalness",
        "blackout-neg.question",
        "blackout-neg.text.0",
        "blackout-neg.text.1",
        "blackout-neg.text.2",
        "blackout-neg.text.3",
        "blackout-pos.difficulty",
        "blackout-pos.interest",
        "blackout-pos.naturalness",
        "blackout-pos.question",
        "blackout-pos.text.0",
        "blackout-pos.text.1",
        "blackout-pos.text.2",
        "blackout-pos.text.3",
        "blackout-pos.text.4",
        "blackout-zero.difficulty",
        "blackout-zero.interest",
        "blackout-zero.naturalness",
        "blackout-zero.question",
        "blackout-zero.text.0",
        "blackout-zero.text.1",
        "blackout-zero.text.2",
        "blackout-zero.text.3",
        "blackout-zero.text.4",
        "breakfast-neg.difficulty",
        "breakfast-neg.interest",
        "breakfast-neg.naturalness",
        "breakfast-neg.question",
        "breakfast-neg.text.0",
        "breakfast-neg.text.1",
        "breakfast-neg.text.2",
        "breakfast-neg.text.3",
        "breakfast-pos.difficulty",
        "breakfast-pos.interest",
        "breakfast-pos.naturalness",
        "breakfast-pos.question",
        "breakfast-pos.text.0",
        "breakfast-pos.text.1",
        "breakfast-pos.text.2",
        "breakfast-pos.text.3",
        "breakfast-zero.difficulty",
        "breakfast-zero.interest",
        "breakfast-zero.naturalness",
        "breakfast-zero.question",
        "breakfast-zero.text.0",
        "breakfast-zero.text.1",
        "breakfast-zero.text.2",
        "breakfast-zero.text.3",
        "delayed-neg.difficulty",
        "delayed-neg.interest",
        "delayed-neg.naturalness",
        "delayed-neg.question",
        "delayed-neg.text.0",
        "delayed-neg.text.1",
        "delayed-neg.text.2",
        "delayed-neg.text.3",
        "delayed-neg.text.4",
        "delayed-pos.difficulty",
        "delayed-pos.interest",
        "delayed-pos.naturalness",
        "delayed-pos.question",
        "delayed-pos.text.0",
        "delayed-pos.text.1",
        "delayed-pos.text.2",
        "delayed-pos.text.3",
        "delayed-zero.difficulty",
        "delayed-zero.interest",
        "delayed-zero.naturalness",
        "delayed-zero.question",
        "delayed-zero.text.0",
        "delayed-zero.text.1",
        "delayed-zero.text.2",
        "delayed-zero.text.3",
        "delayed-zero.text.4",
        "goldfish-neg.difficulty",
        "goldfish-neg.interest",
        "goldfish-neg.naturalness",
        "goldfish-neg.question",
        "goldfish-neg.text.0",
        "goldfish-neg.text.1",
        "goldfish-neg.text.2",
        "goldfish-neg.text.3",
        "goldfish-pos.difficulty",
        "goldfish-pos.interest",
        "goldfish-pos.naturalness",
        "goldfish-pos.question",
        "goldfish-pos.text.0",
        "goldfish-pos.text.1",
        "goldfish-pos.text.2",
        "goldfish-pos.text.3",
        "goldfish-pos.text.4",
        "goldfish-zero.difficulty",
        "goldfish-zero.interest",
        "goldfish-zero.naturalness",
        "goldfish-zero.question",
        "goldfish-zero.text.0",
        "goldfish-zero.text.1",
        "goldfish-zero.text.2",
        "goldfish-zero.text.3",
        "goldfish-zero.text.4",
        "practice.difficulty",
        "practice.interest",
        "practice.naturalness",
        "practice.question",
        "practice.text.0",
        "practice.text.1",
        "prize-neg.difficulty",
        "prize-neg.interest",
    ]
    GGTG_EVAL_STIM = [
        "prize-zero.text.4",
        "voicemail-neg.difficulty",
        "voicemail-neg.interest",
        "voicemail-neg.naturalness",
        "voicemail-neg.question",
        "voicemail-neg.text.0",
        "voicemail-neg.text.1",
        "voicemail-neg.text.2",
        "voicemail-neg.text.3",
        "voicemail-pos.difficulty",
        "voicemail-pos.interest",
        "voicemail-pos.naturalness",
        "voicemail-pos.question",
        "voicemail-pos.text.0",
        "voicemail-pos.text.1",
        "voicemail-pos.text.2",
        "voicemail-pos.text.3",
        "voicemail-zero.difficulty",
        "voicemail-zero.interest",
        "voicemail-zero.naturalness",
        "voicemail-zero.question",
        "voicemail-zero.text.0",
        "voicemail-zero.text.1",
        "voicemail-zero.text.2",
        "voicemail-zero.text.3",
    ]

    # Run the lightweight baseline unit test first
    test_baselines(
        dataset_name="GGTG",
        root=r"/home/janhof/thesis/data",
        train_subset={"stimulus": GGTG_TRAIN_STIM},
        eval_subset={"stimulus": GGTG_EVAL_STIM},
        n_generate=5,
        gen_len=2000,
    )

    # Full evaluation with both baselines
    _ROOT = r"/home/janhof/thesis/data"
    _COMMON = dict(
        dataset_name="GGTG",
        root=_ROOT,
        out_dir=str(Path(_ROOT).parent / "eval_results"),
        subset={"stimulus": GGTG_EVAL_STIM},
        n_generate=20,
        seed_len=32,
        gen_len=2000,
        dispersion_threshold=1.0,
        min_fix_duration=98,
        min_sac_duration=18,
        vel_method="fivepoint",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    syn = SyntheticGenerator()
    emp = TrainingDistributionGenerator(
        dataset_name="GGTG",
        root=_ROOT,
        train_subset={"stimulus": GGTG_TRAIN_STIM},
    )
    run_multi_evaluation(generators=[syn, emp], **_COMMON)


def main():
    p, subparsers = _build_parser()

    # ── Config file: scan sys.argv directly for --config ──────────────────
    #  scan rather than use parse_known_args() to avoid conflicts with
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
        # Apply config defaults only to the active subparser so that keys
        # that don't exist on a given subparser are silently ignored.
        mode_from_argv = next(
            (a for a in _argv if a in {"model", "synthetic", "empirical", "compare"}),
            None,
        )
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

    # ── Validate checkpoint where required ────────────────────────────────
    if args.mode in ("model", "compare") and not getattr(args, "checkpoint", None):
        p.error(
            f"'{args.mode}' mode requires --checkpoint "
            "(or set 'checkpoint' in --config)"
        )

    subset = _build_subset(args)

    out_dir = args.out_dir or str(Path(args.root).parent / "eval_results")

    common = dict(
        dataset_name=args.dataset,
        root=args.root,
        out_dir=out_dir,
        subset=subset,
        n_generate=args.n_generate,
        seed_len=args.seed_len,
        gen_len=args.gen_len,
        vel_threshold=args.vel_threshold,
        dispersion_threshold=args.dispersion_threshold,
        min_fix_duration=args.min_fix_dur,
        min_sac_duration=args.min_sac_dur,
        vel_method=args.vel_method,
        device=args.device,
        per_stimulus_plots=args.per_stimulus_plots,
        scanpath_overview=args.scanpath_overview,
        rng_seed=args.rng_seed,
    )

    def _make_empirical(label=None):
        train_stimuli = getattr(args, "train_stimuli", None)
        train_subset = {"stimulus": train_stimuli} if train_stimuli else None
        return TrainingDistributionGenerator(
            dataset_name=args.dataset,
            root=args.root,
            train_subset=train_subset,
            label=label,
        )

    if args.mode == "model":
        gen = GMMModelGenerator(
            args.checkpoint, args.temperature, args.device, args.label
        )
        extras: List[SequenceGenerator] = []
        if args.also_synthetic:
            extras.append(SyntheticGenerator())
        if args.also_empirical:
            extras.append(_make_empirical())
        run_evaluation(gen, extra_generators=extras or None, **common)

    elif args.mode == "synthetic":
        gen = SyntheticGenerator(label=getattr(args, "label", None))
        run_evaluation(gen, **common)

    elif args.mode == "empirical":
        gen = _make_empirical(label=getattr(args, "label", None))
        run_evaluation(gen, **common)

    elif args.mode == "compare":
        gen = GMMModelGenerator(args.checkpoint, args.temperature, args.device)
        baseline = getattr(args, "baseline", "synthetic")
        extras = []
        if baseline in ("synthetic", "both"):
            extras.append(SyntheticGenerator())
        if baseline in ("empirical", "both"):
            extras.append(_make_empirical())
        run_evaluation(gen, extra_generators=extras, **common)


if __name__ == "__main__":
    main()
