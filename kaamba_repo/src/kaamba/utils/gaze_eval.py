"""
kaamba/utils/gaze_eval.py  — pymovements edition

Evaluation metrics for synthetic gaze trajectories using pymovements
for oculomotor event detection (IVT fixations, fill saccades).

Key difference from the manual version:
  - pymovements.events.ivt()  detects fixations via velocity threshold
  - pymovements.events.fill() labels remaining samples as saccades
  - Both return Events objects with onset/offset/duration columns
  - Velocities must be in degrees/s — pass screen_w_deg + screen_h_deg
    to handle the [0,1] → degrees conversion automatically

Usage:
    from kaamba.utils.gaze_eval import GazeEvaluator, generate_sequences

    evaluator = GazeEvaluator(
        sample_rate_hz = 500,
        screen_w_deg   = 36.0,   # horizontal extent of screen in degrees
        screen_h_deg   = 20.0,   # vertical   extent of screen in degrees
    )
    results = evaluator.evaluate(real_seqs, fake_seqs)
    evaluator.report(results)
    tracker.log_final_eval(evaluator.flatten(results))
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import polars as pl
import pymovements as pm
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------


def _norm_to_deg(
    xy: np.ndarray, screen_w_deg: float, screen_h_deg: float
) -> np.ndarray:
    """
    Convert normalised [0,1] gaze coordinates to visual degrees.
    xy: (..., 2)
    """
    out = xy.copy().astype(float)
    out[..., 0] *= screen_w_deg
    out[..., 1] *= screen_h_deg
    return out


def _compute_velocity_deg(xy_deg: np.ndarray, sr: float) -> np.ndarray:
    """
    xy_deg: (T, 2) in degrees
    Returns velocity in deg/s, shape (T, 2).
    First sample is duplicated to keep length T.
    """
    dx = np.diff(xy_deg, axis=0) * sr  # (T-1, 2)
    return np.concatenate([dx[:1], dx], axis=0)  # (T, 2)


# ---------------------------------------------------------------------------
# Event extraction via pymovements
# ---------------------------------------------------------------------------


def extract_events_pm(
    xy_norm: np.ndarray,
    sr: float,
    screen_w_deg: float,
    screen_h_deg: float,
    vel_threshold: float = 30.0,  # deg/s — standard IVT threshold
    min_fix_duration: int = 50,  # samples
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Run IVT fixation detection + fill saccades on a single (T, 2) trajectory.

    Args:
        xy_norm:          (T, 2) normalised [0,1] gaze coordinates
        sr:               sample rate in Hz
        screen_w_deg:     screen width  in visual degrees
        screen_h_deg:     screen height in visual degrees
        vel_threshold:    IVT velocity threshold in deg/s (default 30 deg/s)
        min_fix_duration: minimum fixation length in samples

    Returns:
        fix_df:  polars DataFrame with fixation events
                 columns: name, onset, offset, duration, cx_deg, cy_deg
        sac_df:  polars DataFrame with saccade events
                 columns: name, onset, offset, duration, amplitude_deg, peak_vel_deg_s
    """
    T = len(xy_norm)
    xy_deg = _norm_to_deg(xy_norm, screen_w_deg, screen_h_deg)  # (T, 2)
    vel_deg = _compute_velocity_deg(xy_deg, sr)  # (T, 2)
    timesteps = np.arange(T, dtype=int)

    # ── Fixation detection (IVT) ─────────────────────────────────────────
    fix_events = pm.events.ivt(
        velocities=vel_deg,
        timesteps=timesteps,
        velocity_threshold=vel_threshold,
        minimum_duration=min_fix_duration,
        name="fixation",
    )

    # ── Saccade detection (fill gaps between fixations) ──────────────────
    sac_events = pm.events.fill(
        events=fix_events,
        timesteps=timesteps,
        name="saccade",
    )

    # ── Enrich fixation frame with centroid position ──────────────────────
    fix_rows = []
    for row in fix_events.frame.iter_rows(named=True):
        seg = xy_deg[row["onset"] : row["offset"]]
        fix_rows.append(
            {
                **row,
                "cx_deg": float(seg[:, 0].mean()) if len(seg) else float("nan"),
                "cy_deg": float(seg[:, 1].mean()) if len(seg) else float("nan"),
            }
        )
    fix_df = (
        pl.DataFrame(fix_rows)
        if fix_rows
        else pl.DataFrame(
            schema={
                "name": pl.Utf8,
                "onset": pl.Int64,
                "offset": pl.Int64,
                "duration": pl.Int64,
                "cx_deg": pl.Float64,
                "cy_deg": pl.Float64,
            }
        )
    )

    # ── Enrich saccade frame with amplitude + peak velocity ───────────────
    sac_rows = []
    for row in sac_events.frame.iter_rows(named=True):
        seg_xy = xy_deg[row["onset"] : row["offset"]]
        seg_vel = vel_deg[row["onset"] : row["offset"]]
        amp = float(np.linalg.norm(seg_xy[-1] - seg_xy[0])) if len(seg_xy) > 1 else 0.0
        pv = float(np.linalg.norm(seg_vel, axis=1).max()) if len(seg_vel) > 0 else 0.0
        angle = (
            float(
                np.arctan2(seg_xy[-1, 1] - seg_xy[0, 1], seg_xy[-1, 0] - seg_xy[0, 0])
            )
            if len(seg_xy) > 1
            else float("nan")
        )
        sac_rows.append(
            {**row, "amplitude_deg": amp, "peak_vel_deg_s": pv, "angle_rad": angle}
        )
    sac_df = (
        pl.DataFrame(sac_rows)
        if sac_rows
        else pl.DataFrame(
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
    )

    return fix_df, sac_df


def _extract_all_events(
    seqs: np.ndarray,
    sr: float,
    screen_w_deg: float,
    screen_h_deg: float,
    vel_threshold: float,
    min_fix_dur: int,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Run extract_events_pm over all (N, T, 2) sequences and concatenate.
    Returns two DataFrames: all fixations, all saccades.
    """
    all_fix, all_sac = [], []
    for seq in seqs:
        f, s = extract_events_pm(
            seq, sr, screen_w_deg, screen_h_deg, vel_threshold, min_fix_dur
        )
        all_fix.append(f)
        all_sac.append(s)

    fix_df = pl.concat(all_fix) if all_fix else pl.DataFrame()
    sac_df = pl.concat(all_sac) if all_sac else pl.DataFrame()
    return fix_df, sac_df


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------


def _ks(a: np.ndarray, b: np.ndarray) -> Dict:
    if len(a) < 2 or len(b) < 2:
        return {
            "ks_stat": float("nan"),
            "p_value": float("nan"),
            "n_real": int(len(a)),
            "n_fake": int(len(b)),
        }
    stat, p = stats.ks_2samp(a, b)
    return {
        "ks_stat": float(stat),
        "p_value": float(p),
        "n_real": int(len(a)),
        "n_fake": int(len(b)),
    }


def metric_fixation_duration(real_fix: pl.DataFrame, fake_fix: pl.DataFrame) -> Dict:
    r = real_fix["duration"].to_numpy().astype(float) if len(real_fix) else np.array([])
    f = fake_fix["duration"].to_numpy().astype(float) if len(fake_fix) else np.array([])
    result = _ks(r, f)
    result.update(
        {
            "real_mean_samples": float(r.mean()) if len(r) else float("nan"),
            "fake_mean_samples": float(f.mean()) if len(f) else float("nan"),
            "real_std_samples": float(r.std()) if len(r) else float("nan"),
            "fake_std_samples": float(f.std()) if len(f) else float("nan"),
        }
    )
    return result


def metric_saccade_amplitude(real_sac: pl.DataFrame, fake_sac: pl.DataFrame) -> Dict:
    r = real_sac["amplitude_deg"].to_numpy() if len(real_sac) else np.array([])
    f = fake_sac["amplitude_deg"].to_numpy() if len(fake_sac) else np.array([])
    result = _ks(r, f)
    result.update(
        {
            "real_mean_deg": float(r.mean()) if len(r) else float("nan"),
            "fake_mean_deg": float(f.mean()) if len(f) else float("nan"),
            "real_std_deg": float(r.std()) if len(r) else float("nan"),
            "fake_std_deg": float(f.std()) if len(f) else float("nan"),
        }
    )
    return result


def metric_main_sequence(real_sac: pl.DataFrame, fake_sac: pl.DataFrame) -> Dict:
    """
    Pearson r between saccade amplitude and peak velocity.
    The main sequence (Bahill 1975) should yield r > 0.9.
    """

    def _r(sac_df):
        if len(sac_df) < 5:
            return float("nan")
        amp = sac_df["amplitude_deg"].to_numpy()
        pv = sac_df["peak_vel_deg_s"].to_numpy()
        mask = (amp > 0.1) & (pv > 1.0)
        if mask.sum() < 5:
            return float("nan")
        r, _ = stats.pearsonr(amp[mask], pv[mask])
        return float(r)

    fake_r = _r(fake_sac)
    return {
        "real_r": _r(real_sac),
        "fake_r": fake_r,
        "target": "> 0.9",
        "pass": fake_r > 0.9 if not np.isnan(fake_r) else False,
    }


def metric_intersaccadic_interval(
    real_fix: pl.DataFrame, fake_fix: pl.DataFrame
) -> Dict:
    """ISI ≈ fixation duration. Match mean and variance."""
    r = real_fix["duration"].to_numpy().astype(float) if len(real_fix) else np.array([])
    f = fake_fix["duration"].to_numpy().astype(float) if len(fake_fix) else np.array([])
    return {
        "real_mean_samples": float(r.mean()) if len(r) else float("nan"),
        "fake_mean_samples": float(f.mean()) if len(f) else float("nan"),
        "real_var_samples": float(r.var()) if len(r) else float("nan"),
        "fake_var_samples": float(f.var()) if len(f) else float("nan"),
        "mean_abs_err": float(abs(r.mean() - f.mean()))
        if (len(r) and len(f))
        else float("nan"),
        "var_ratio": float(f.var() / r.var())
        if (len(r) and r.var() > 0)
        else float("nan"),
    }


def metric_fixation_density(
    real_seqs: np.ndarray,
    fake_seqs: np.ndarray,
    real_fix: pl.DataFrame,
    fake_fix: pl.DataFrame,
    grid: int = 32,
) -> Dict:
    """
    Compare fixation centroid spatial distributions via KL divergence.
    Uses actual fixation centroids from pymovements rather than all gaze points.
    """

    def _density(fix_df, grid):
        if len(fix_df) == 0 or "cx_deg" not in fix_df.columns:
            return None
        # Normalise centroids to [0,1] for grid
        cx = fix_df["cx_deg"].to_numpy()
        cy = fix_df["cy_deg"].to_numpy()
        cx_n = (cx - cx.min()) / (cx.max() - cx.min() + 1e-9)
        cy_n = (cy - cy.min()) / (cy.max() - cy.min() + 1e-9)
        xi = (cx_n * (grid - 1)).astype(int).clip(0, grid - 1)
        yi = (cy_n * (grid - 1)).astype(int).clip(0, grid - 1)
        h, _, _ = np.histogram2d(xi, yi, bins=grid, range=[[0, grid], [0, grid]])
        h += 1e-8
        return h / h.sum()

    real_d = _density(real_fix, grid)
    fake_d = _density(fake_fix, grid)

    if real_d is None or fake_d is None:
        return {
            "kl_divergence": float("nan"),
            "note": "insufficient fixations for density map",
        }

    kl = float(stats.entropy(real_d.ravel(), fake_d.ravel()))
    return {
        "kl_divergence": kl,
        "real_n_fixations": int(len(real_fix)),
        "fake_n_fixations": int(len(fake_fix)),
        "note": "lower is better; 0 = identical spatial distributions",
    }


def metric_saccade_direction(
    real_sac: pl.DataFrame, fake_sac: pl.DataFrame, n_bins: int = 8
) -> Dict:
    """
    Compare saccade direction histograms.
    Angles come directly from the enriched saccade DataFrame.
    """

    def _hist(sac_df):
        if len(sac_df) == 0 or "angle_rad" not in sac_df.columns:
            return None
        angles = sac_df["angle_rad"].drop_nulls().to_numpy()
        angles = angles[~np.isnan(angles)]
        if len(angles) < 3:
            return None
        h, _ = np.histogram(angles, bins=n_bins, range=(-np.pi, np.pi))
        return h.astype(float)

    rh = _hist(real_sac)
    fh = _hist(fake_sac)

    if rh is None or fh is None:
        return {"note": "insufficient saccades for direction histogram"}

    rh_n = rh / rh.sum()
    fh_n = fh / fh.sum()
    kl = float(stats.entropy(rh_n + 1e-8, fh_n + 1e-8))
    ks = _ks(rh, fh)
    return {"kl_divergence": kl, **ks}


def metric_classifier_auc(
    real_seqs: np.ndarray,
    fake_seqs: np.ndarray,
    max_samples: int = 2000,
) -> Dict:
    """
    Train a logistic regression to distinguish real vs fake.
    AUC ≈ 0.5 → indistinguishable (good); AUC ≈ 1.0 → easily separable (bad).
    """

    def _features(seqs: np.ndarray) -> np.ndarray:
        dx = np.diff(seqs, axis=1)
        speed = np.linalg.norm(dx, axis=-1)
        return np.concatenate(
            [
                seqs[:, :, 0].mean(axis=1, keepdims=True),
                seqs[:, :, 1].mean(axis=1, keepdims=True),
                seqs[:, :, 0].std(axis=1, keepdims=True),
                seqs[:, :, 1].std(axis=1, keepdims=True),
                speed.mean(axis=1, keepdims=True),
                speed.std(axis=1, keepdims=True),
                speed.max(axis=1, keepdims=True),
                np.percentile(speed, 90, axis=1, keepdims=True),
            ],
            axis=1,
        )

    n_r = min(len(real_seqs), max_samples // 2)
    n_f = min(len(fake_seqs), max_samples // 2)

    X = np.concatenate([_features(real_seqs[:n_r]), _features(fake_seqs[:n_f])])
    y = np.array([1] * n_r + [0] * n_f)

    X = StandardScaler().fit_transform(X)
    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(X, y)
    auc = float(roc_auc_score(y, clf.predict_proba(X)[:, 1]))

    return {
        "auc": auc,
        "target": "≈ 0.5",
        "pass": abs(auc - 0.5) < 0.1,
        "note": "AUC=0.5 indistinguishable (good); AUC=1.0 easily separable (bad)",
    }


# ---------------------------------------------------------------------------
# GazeEvaluator — main interface
# ---------------------------------------------------------------------------


@dataclass
class EvalResults:
    fixation_duration: Dict
    saccade_amplitude: Dict
    main_sequence: Dict
    intersaccadic_interval: Dict
    fixation_density_map: Dict
    saccade_direction: Dict
    classifier_auc: Dict

    def to_dict(self) -> dict:
        return asdict(self)


class GazeEvaluator:
    """
    Evaluate synthetic gaze trajectories against real ones using pymovements
    for event detection.

    Args:
        sample_rate_hz:   recording sample rate (Hz)
        screen_w_deg:     screen width  in visual degrees (needed for deg/s conversion)
        screen_h_deg:     screen height in visual degrees
        vel_threshold:    IVT velocity threshold in deg/s (default 30 — standard value)
        min_fix_duration: minimum fixation duration in samples

    Screen size → degrees conversion:
        degrees = 2 * atan(screen_size_m / (2 * viewing_distance_m)) * 180/pi
        Typical monitor at 60cm: ~36° wide, ~20° tall
        If unknown, use width_px/40 as a rough approximation.
    """

    def __init__(
        self,
        sample_rate_hz: float = 500.0,
        screen_w_deg: float = 36.0,
        screen_h_deg: float = 20.0,
        vel_threshold: float = 30.0,
        min_fix_duration: int = 50,
    ):
        self.sr = sample_rate_hz
        self.screen_w_deg = screen_w_deg
        self.screen_h_deg = screen_h_deg
        self.vel_threshold = vel_threshold
        self.min_fix_duration = min_fix_duration

    def evaluate(
        self,
        real: np.ndarray,
        fake: np.ndarray,
        density_grid: int = 32,
        classifier_samples: int = 2000,
    ) -> EvalResults:
        """
        Args:
            real: (N, T, 2) real gaze sequences, normalised [0,1]
            fake: (N, T, 2) generated gaze sequences, normalised [0,1]
        """
        assert real.ndim == 3 and real.shape[2] == 2, (
            f"real must be (N,T,2), got {real.shape}"
        )
        assert fake.ndim == 3 and fake.shape[2] == 2, (
            f"fake must be (N,T,2), got {fake.shape}"
        )

        print(
            f"[eval] detecting events in {len(real)} real + {len(fake)} fake sequences..."
        )

        real_fix, real_sac = _extract_all_events(
            real,
            self.sr,
            self.screen_w_deg,
            self.screen_h_deg,
            self.vel_threshold,
            self.min_fix_duration,
        )
        fake_fix, fake_sac = _extract_all_events(
            fake,
            self.sr,
            self.screen_w_deg,
            self.screen_h_deg,
            self.vel_threshold,
            self.min_fix_duration,
        )

        print(f"[eval] real: {len(real_fix)} fixations, {len(real_sac)} saccades")
        print(f"[eval] fake: {len(fake_fix)} fixations, {len(fake_sac)} saccades")

        return EvalResults(
            fixation_duration=metric_fixation_duration(real_fix, fake_fix),
            saccade_amplitude=metric_saccade_amplitude(real_sac, fake_sac),
            main_sequence=metric_main_sequence(real_sac, fake_sac),
            intersaccadic_interval=metric_intersaccadic_interval(real_fix, fake_fix),
            fixation_density_map=metric_fixation_density(
                real, fake, real_fix, fake_fix, density_grid
            ),
            saccade_direction=metric_saccade_direction(real_sac, fake_sac),
            classifier_auc=metric_classifier_auc(real, fake, classifier_samples),
        )

    def report(self, results: EvalResults):
        d = results.to_dict()
        print("\n" + "=" * 65)
        print("GAZE EVALUATION REPORT  (pymovements IVT)")
        print(
            f"  vel_threshold={self.vel_threshold} deg/s  "
            f"sr={self.sr} Hz  "
            f"screen={self.screen_w_deg}×{self.screen_h_deg}°"
        )
        print("=" * 65)

        sections = {
            "Fixation duration (KS test)": "fixation_duration",
            "Saccade amplitude (KS test)": "saccade_amplitude",
            "Main sequence (amp ~ peak vel)": "main_sequence",
            "Intersaccadic interval": "intersaccadic_interval",
            "Fixation density map (KL div)": "fixation_density_map",
            "Saccade direction histogram (KL div)": "saccade_direction",
            "Classifier AUC  (↓ better)": "classifier_auc",
        }
        for label, key in sections.items():
            print(f"\n  {label}")
            for k, v in d[key].items():
                if isinstance(v, float):
                    print(f"    {k:<30} {v:.4f}")
                else:
                    print(f"    {k:<30} {v}")

        print("\n  PASS / FAIL")
        checks = {
            "main_sequence r > 0.9": results.main_sequence.get("pass", False),
            "classifier AUC ≈ 0.5": results.classifier_auc.get("pass", False),
        }
        for kp, pv in [
            ("fixation_duration", "fixation_dur KS p > 0.05"),
            ("saccade_amplitude", "saccade_amp  KS p > 0.05"),
        ]:
            p = results.to_dict()[kp].get("p_value", 0.0)
            checks[pv] = (p > 0.05) if not np.isnan(p) else False
        for name, passed in checks.items():
            print(f"    {'✓' if passed else '✗'} {name}")
        print("=" * 65 + "\n")

    def save(self, results: EvalResults, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(results.to_dict(), indent=2))
        print(f"[eval] saved → {path}")

    def flatten(self, results: EvalResults) -> Dict:
        """Flat dict for ExperimentTracker / W&B."""
        out = {}
        for section, sub in results.to_dict().items():
            if isinstance(sub, dict):
                for k, v in sub.items():
                    if isinstance(v, (int, float, bool)):
                        out[f"{section}/{k}"] = v
        return out


# ---------------------------------------------------------------------------
# Autoregressive generation helper
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)
    sr, N, T = 500, 100, 300

    def _make(n, t, sr, noise=0.003):
        seqs = []
        for _ in range(n):
            xy = np.zeros((t, 2))
            pos = np.array([10.0, 8.0])
            i = 0
            while i < t:
                fl = min(int(np.random.uniform(0.1, 0.4) * sr), t - i)
                xy[i : i + fl] = pos + np.random.randn(fl, 2) * noise
                i += fl
                if i >= t:
                    break
                sl = min(int(np.random.uniform(0.02, 0.06) * sr), t - i)
                tgt = pos + np.random.randn(2) * 3
                for j in range(sl):
                    xy[i + j] = pos + (tgt - pos) * j / sl
                pos = tgt
                i += sl
            seqs.append(np.clip(xy / 36.0, 0, 1))  # rough deg→norm
        return np.array(seqs)

    real = _make(N, T, sr, noise=0.003)
    fake = _make(N, T, sr, noise=0.006)  # slightly noisier

    evaluator = GazeEvaluator(sample_rate_hz=sr, screen_w_deg=36.0, screen_h_deg=20.0)
    results = evaluator.evaluate(real, fake)
    evaluator.report(results)
    evaluator.save(results, "/tmp/gaze_eval_pm_test.json")
