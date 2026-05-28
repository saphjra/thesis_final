"""
evaluate_model.py

Loads a pymovements dataset, iterates over gaze objects per stimulus,
generates matching synthetic sequences from a trained model, and computes
all GazeEvaluator metrics — both per-stimulus and aggregated.

Usage:
    python evaluate_model.py \
        --checkpoint  /path/to/best_model.pt \
        --dataset     mcfw-gaze \
        --root        /home/janhof/thesis/data \
        --out_dir     /home/janhof/thesis/eval \
        --split       test          # participant split to use
        --n_generate  50            # synthetic sequences per stimulus
        --seed_len    10
        --gen_len     128

Output:
    <out_dir>/
    ├── per_stimulus/
    │   ├── <stimulus_name>.json     ← full metrics for each stimulus
    │   └── ...
    ├── aggregate.json               ← metrics averaged over all stimuli
    └── eval_report.txt              ← human-readable summary
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import polars as pl
import pymovements as pm
import torch
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project imports  (adjust paths to your package layout if needed)
# ---------------------------------------------------------------------------
from kaamba.net.models.kaamba import build_gaze_predictor
from kaamba.utils.gaze_eval import GazeEvaluator, generate_sequences


# ---------------------------------------------------------------------------
# Helpers: event extraction via pymovements
# ---------------------------------------------------------------------------
def _gaze_obj_to_arrays(
    gaze,
    vel_method: str = "fivepoint",
    vel_threshold: float = 30.0,  # deg/s — rescaled below if needed
    min_fix_dur: int = 10,
    screen_w_deg: float = 36.0,  # used for threshold rescaling only
    screen_h_deg: float = 20.0,
):
    px_arr = np.stack(gaze.samples["pixel"].to_numpy())  # (T, 2)

    # Detect whether coordinates are already normalised
    is_normalised = px_arr.max() <= 2.0  # >2.0 = definitely pixel space
    screen = gaze.experiment.screen
    if is_normalised:
        # Already in [0,1] — convert to degrees for IVT
        # so velocity units are consistent with the threshold

        pos_arr = np.stack(
            [
                px_arr[:, 0] * screen_w_deg,
                px_arr[:, 1] * screen_h_deg,
            ],
            axis=1,
        )
        norm_arr = px_arr  # already normalised
    else:
        # Raw pixels — use pymovements pix2deg
        gaze.pix2deg()
        pos_arr = np.stack(gaze.samples["position"].to_numpy())
        norm_arr = np.stack(
            [
                px_arr[:, 0] / screen.width_px,
                px_arr[:, 1] / screen.height_px,
            ],
            axis=1,
        )

    # Compute velocity in deg/s from pos_arr (now always in degrees)
    sr = gaze.experiment.sampling_rate
    vel_arr = np.diff(pos_arr, axis=0) * sr  # (T-1, 2) deg/s
    vel_arr = np.concatenate([vel_arr[:1], vel_arr])  # (T, 2) pad first
    vel_arr = np.nan_to_num(vel_arr, nan=0.0)

    T = len(pos_arr)
    timesteps = np.arange(T, dtype=int)

    fix_events = pm.events.ivt(
        vel_arr,
        timesteps=timesteps,
        velocity_threshold=vel_threshold,  # deg/s — now valid
        minimum_duration=min_fix_dur,
    )
    sac_events = pm.events.fill(fix_events, timesteps=timesteps, name="saccade")

    return pos_arr, vel_arr, norm_arr, fix_events, sac_events


def _enrich_saccades(sac_events, pos_arr, vel_arr):
    rows = []
    for row in sac_events.frame.iter_rows(named=True):
        seg_pos = pos_arr[row["onset"] : row["offset"]]
        seg_vel = vel_arr[row["onset"] : row["offset"]]
        if len(seg_pos) == 0:  # skip degenerate events
            continue
        amp = (
            float(np.linalg.norm(seg_pos[-1] - seg_pos[0])) if len(seg_pos) > 1 else 0.0
        )
        pv = float(np.linalg.norm(seg_vel, axis=1).max()) if len(seg_vel) > 0 else 0.0
        angle = (
            float(
                np.arctan2(
                    seg_pos[-1, 1] - seg_pos[0, 1],
                    seg_pos[-1, 0] - seg_pos[0, 0],
                )
            )
            if len(seg_pos) > 1
            else float("nan")
        )
        rows.append(
            {**row, "amplitude_deg": amp, "peak_vel_deg_s": pv, "angle_rad": angle}
        )
    return (
        pl.DataFrame(rows)
        if rows
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


def _enrich_fixations(fix_events, pos_arr):
    rows = []
    for row in fix_events.frame.iter_rows(named=True):
        seg = pos_arr[row["onset"] : row["offset"]]
        if len(seg) == 0:  # skip degenerate zero-length events
            continue
        rows.append(
            {
                **row,
                "cx_deg": float(seg[:, 0].mean()),
                "cy_deg": float(seg[:, 1].mean()),
            }
        )
    return (
        pl.DataFrame(rows)
        if rows
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
    evaluator: GazeEvaluator,
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
    checkpoint_path: str,
    dataset_name: str,
    root: str,
    out_dir: str,
    subset: Optional[Dict] = None,
    n_generate: int = 50,
    seed_len: int = 10,
    gen_len: int = 128,
    temperature: float = 1.0,
    vel_threshold: float = 30.0,  # deg/s IVT threshold
    min_fix_duration: int = 10,  # samples
    vel_method: str = "fivepoint",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Main evaluation loop.

    For each (subject, stimulus) pair in the dataset:
      1. Load real gaze sequences
      2. Load the matching stimulus image
      3. Generate synthetic sequences from the model
      4. Run all GazeEvaluator metrics
      5. Save per-stimulus JSON + aggregate report
    """
    out_dir = Path(out_dir)
    stim_dir = out_dir / "per_stimulus"
    stim_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────
    print(f"[eval] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    model_config = ckpt["config"]["model_config"]
    model = build_gaze_predictor(**model_config, verbose=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    print(
        f"[eval] Model loaded  ({sum(p.numel() for p in model.parameters()):,} params)"
    )

    # ── Load dataset ──────────────────────────────────────────────────────
    print(f"\n[eval] Loading dataset: {dataset_name}")
    dataset_paths = pm.DatasetPaths(root=root)
    dataset = pm.Dataset(dataset_name, path=dataset_paths)
    dataset.scan()
    dataset.load(subset=subset)

    print(f"[eval] Loaded {len(dataset.gaze)} gaze files")

    # ── Screen info from first gaze object ───────────────────────────────
    first_gaze = dataset.gaze[0]
    screen = first_gaze.experiment.screen
    sr = first_gaze.experiment.sampling_rate
    scr_w_px = screen.width_px
    scr_h_px = screen.height_px

    # Degree extent for GazeEvaluator (needs origin set for dva)
    try:
        scr_w_deg = screen.x_max_dva - screen.x_min_dva
        scr_h_deg = screen.y_max_dva - screen.y_min_dva
    except TypeError:
        # origin not set — fall back to rough estimate
        scr_w_deg = 2 * np.degrees(
            np.arctan(screen.width_cm / (2 * screen.distance_cm))
        )
        scr_h_deg = 2 * np.degrees(
            np.arctan(screen.height_cm / (2 * screen.distance_cm))
        )

    print(
        f"[eval] Screen: {scr_w_px}×{scr_h_px}px  "
        f"{scr_w_deg:.1f}×{scr_h_deg:.1f}°  sr={sr}Hz"
    )

    evaluator = GazeEvaluator(
        sample_rate_hz=sr,
        screen_w_deg=scr_w_deg,
        screen_h_deg=scr_h_deg,
        vel_threshold=vel_threshold,
        min_fix_duration=min_fix_duration,
    )

    # ── Group gaze objects by stimulus ───────────────────────────────────
    # Each gaze object corresponds to one (subject, stimulus) recording.
    # We group by stimulus so we can aggregate across subjects per image.
    from collections import defaultdict

    by_stimulus = defaultdict(list)
    for gaze in dataset.gaze:
        stim = gaze.metadata.get("stimulus", "unknown")
        by_stimulus[stim].append(gaze)

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
        all_real_fix = []  # enriched fixation DataFrames
        all_real_sac = []  # enriched saccade DataFrames

        for gaze in gaze_list:
            # Clone so pix2deg/pos2vel don't mutate the stored object
            g = gaze.clone()
            try:
                pos_arr, vel_arr, norm_arr, fix_ev, sac_ev = _gaze_obj_to_arrays(
                    g,
                    vel_method=vel_method,
                    vel_threshold=vel_threshold,
                    min_fix_dur=min_fix_duration,
                    screen_h_deg=scr_h_deg,
                    screen_w_deg=scr_w_deg,
                )
            except Exception as e:
                print(
                    f"  [warn] {stim_name} / {gaze.metadata.get('subject_id')} "
                    f"event detection failed: {e}"
                )
                continue

            T_avail = len(norm_arr)
            min_len = seed_len + gen_len
            if T_avail < min_len:
                tqdm.write(
                    f"  [dbg] {stim_name}/{gaze.metadata.get('subject_id')}: "
                    f"only {T_avail} samples, need {min_len} — skipping"
                )
                continue

            step = 1  # can be changed to step = max(1, gen_len // 2) if  with 50% stride for more samples per recording
            for start in range(0, T_avail - gen_len + 1, step):
                real_norm_seqs.append(norm_arr[start : start + gen_len])

            all_real_fix.append(_enrich_fixations(fix_ev, pos_arr))
            all_real_sac.append(_enrich_saccades(sac_ev, pos_arr, vel_arr))

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

        import torchvision.transforms.functional as TF
        from PIL import Image

        img = Image.open(img_path).convert("RGB")
        img_tensor = (
            TF.to_tensor(TF.resize(img, [224, 224])).unsqueeze(0).to(device)
        )  # (1, 3, 224, 224)

        # ── Generate synthetic sequences ──────────────────────────────────
        # Repeat the image n_generate times
        imgs_batch = img_tensor.expand(n_generate, -1, -1, -1)  # (N, 3, 224, 224)

        with torch.no_grad():
            fake_norm = generate_sequences(
                model=model,
                images=imgs_batch,
                seed_len=seed_len,
                gen_len=gen_len,
                temperature=temperature,
                device=device,
            )  # (N, gen_len, 2)

        # ── Extract events from fake sequences ────────────────────────────
        all_fake_fix = []
        all_fake_sac = []

        for seq in fake_norm:
            # Convert normalised → pixel → degrees for event detection
            px_seq = seq * np.array([scr_w_px, scr_h_px])  # (T, 2) px
            exp_obj = first_gaze.experiment

            g_fake = pm.gaze.from_numpy(pixel=px_seq.T, experiment=exp_obj)
            g_fake.pix2deg()
            g_fake.pos2vel(method=vel_method)

            pos_f = np.stack(g_fake.samples["position"].to_numpy())
            vel_f = np.nan_to_num(
                np.stack(g_fake.samples["velocity"].to_numpy()), nan=0.0
            )
            ts = np.arange(gen_len, dtype=int)

            fix_f = pm.events.ivt(
                vel_f,
                timesteps=ts,
                velocity_threshold=vel_threshold,
                minimum_duration=min_fix_duration,
            )
            sac_f = pm.events.fill(fix_f, timesteps=ts, name="saccade")

            all_fake_fix.append(_enrich_fixations(fix_f, pos_f))
            all_fake_sac.append(_enrich_saccades(sac_f, pos_f, vel_f))

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
            evaluator=evaluator,
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
# CLI
# ---------------------------------------------------------------------------


def _parse():
    p = argparse.ArgumentParser(description="Evaluate GazePredictor against real data")
    p.add_argument(
        "--checkpoint", required=True, help="Path to best_model.pt checkpoint"
    )
    p.add_argument(
        "--dataset", required=True, help="pymovements dataset name, e.g. mcfw-gaze"
    )
    p.add_argument("--root", required=True, help="Root directory for pymovements data")
    p.add_argument(
        "--out_dir", default="eval_results", help="Output directory for results"
    )

    # Subset filter (mirrors DataloaderConfigBuilder)
    p.add_argument("--subjects", nargs="*", default=None)
    p.add_argument("--stimuli", nargs="*", default=None)
    p.add_argument("--trial_ids", nargs="*", default=None)

    # Generation
    p.add_argument(
        "--n_generate",
        type=int,
        default=50,
        help="Synthetic sequences to generate per stimulus",
    )
    p.add_argument("--seed_len", type=int, default=10)
    p.add_argument("--gen_len", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.0)

    # Event detection
    p.add_argument(
        "--vel_threshold",
        type=float,
        default=30.0,
        help="IVT velocity threshold in deg/s",
    )
    p.add_argument(
        "--min_fix_dur",
        type=int,
        default=10,
        help="Minimum fixation duration in samples",
    )
    p.add_argument(
        "--vel_method",
        default="fivepoint",
        choices=["fivepoint", "preceding", "smooth"],
        help="Velocity computation method for pymovements",
    )

    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def test():

    run_evaluation(
        checkpoint_path="/home/janhof/thesis/logs/runs/gaze_mamba_search/trial_0019/checkpoints/best_model.pt",
        dataset_name="mcfw-gaze",
        root="/home/janhof/thesis/data",
        out_dir="/home/janhof/thesis/eval_results",
        subset={"subject_id": ["001", "002"]},
        n_generate=20,
        seed_len=32,
        gen_len=200,
        temperature=1,
        vel_threshold=30,
        min_fix_duration=10,
        vel_method="fivepoint",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


if __name__ == "__main__":
    test()
