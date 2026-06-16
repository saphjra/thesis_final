"""
dataset_stats.py

Descriptive statistics and visualisations for the gaze datasets used in
training. Event detection uses the exact same IDT + fill pipeline as
evaluate_model.py, so all numbers are directly comparable to evaluation
results and can be cited used side-by-side.

Statistics produced
───────────────────
Dataset level
  overview       : n_participants, n_stimuli, n_recordings, sampling rate,
                   screen resolution, total samples, valid sample rate
  data volume    : recording durations, estimated training sequences
  fixations      : total count, per-recording mean/std, duration mean/std/
                   median/IQR in milliseconds
  saccades       : total count, amplitude mean/std/median in degrees, peak
                   velocity, main-sequence Pearson r, direction entropy
  spatial        : fixation-density entropy (uniformity of spatial coverage)

Stimulus level
  n_participants, n_fixations, fixation duration mean/std,
  n_saccades, saccade amplitude mean/std, spatial entropy

Plots
─────
  overview.png              recording counts + duration distribution
  fixation_duration.png     KDE with physiological reference band
  saccade_amplitude.png     KDE
  main_sequence.png         scatter + OLS line + Pearson r annotation
  saccade_direction.png     polar rose chart
  spatial_coverage.png      Gaussian-smoothed 2-D fixation heatmap
  per_stimulus_density.png  grid of per-stimulus fixation density maps
  comparison.png            across-dataset summary (only when > 1 dataset)

Usage
─────
  python dataset_stats.py \\
      --datasets mcfw-gaze GGTG \\
      --root     /home/janhof/thesis/data \\
      --out_dir  /home/janhof/thesis/dataset_stats \\
      --context_len 32 \\
      --vel_threshold 30 \\
      --min_fix_dur 10
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import polars as pl
import pymovements as pm
from scipy import stats
from tqdm import tqdm

from kaamba.utils.gaze_preprocessing import GazePreprocessor
from kaamba.utils.stats_plots import (
    PHYSIO_FIX_MIN_MS,
    PHYSIO_FIX_MAX_MS,
    PHYSIO_SAC_MIN_DEG,
    PHYSIO_SAC_MAX_DEG,
    _plot_overview,
    _plot_fixation_duration,
    _plot_saccade_amplitude,
    _plot_main_sequence,
    _plot_saccade_direction,
    _plot_spatial_coverage,
    _plot_per_stimulus_density,
    _plot_comparison,
)
from kaamba.utils.stats_report import build_stats_report

MCFW_STIMULUS = [
    "20",
    "21",
    "24",
    "25",
    "26",
    "27",
    "28",
    "29",
    "30",
    "31",
    "32",
    "33",
    "34",
    "35",
    "36",
    "37",
    "38",
    "39",
    "40",
    "41",
    "42",
    "43",
    "44",
    "45",
    "46",
    "47",
    "48",
    "49",
    "50",
    "51",
    "52",
    "53",
    "54",
    "55",
    "56",
    "57",
    "58",
    "59",
    "60",
    "61",
    "62",
    "63",
    "64",
    "65",
    "66",
    "67",
    "68",
    "69",
    "70",
    "71",
    "72",
    "73",
    "74",
    "75",
    "76",
    "77",
    "78",
    "79",
    "80",
    "81",
    "82",
    "83",
    "84",
    "85",
    "86",
    "87",
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
]

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
    Filter fixation events and append centroid columns cx_deg / cy_deg
    (mean position in degrees of visual angle per fixation).
    pos_arr: (T, 2) from gaze.samples["position"] after pix2deg().
    time_arr: (T,) time values matching the rows of pos_arr (same unit as onset/offset).
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
    keep = ["name", "onset", "offset", "duration", "cx_deg", "cy_deg"]
    return result.select([c for c in keep if c in result.columns])


def _sac_df_from_events(
    ev_frame: pl.DataFrame, pos_arr: np.ndarray, time_arr: np.ndarray
) -> pl.DataFrame:
    """
    Filter saccade events, append angle_rad (direction), and rename
    amplitude → amplitude_deg  /  peak_velocity → peak_vel_deg_s.
    amplitude and peak_velocity are pre-computed by compute_event_properties.
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


# ---------------------------------------------------------------------------
# Core statistics computation
# ---------------------------------------------------------------------------


def _safe(arr: np.ndarray, fn, fallback=float("nan")):
    arr = arr[np.isfinite(arr)]
    return float(fn(arr)) if len(arr) >= 1 else fallback


def _entropy_bits(h: np.ndarray) -> float:
    """Shannon entropy of a normalised histogram in bits."""
    h = h[h > 0]
    return float(-np.sum(h * np.log2(h))) if len(h) > 0 else 0.0


def _density_entropy(cx: np.ndarray, cy: np.ndarray, grid: int = 32) -> float:
    """Spatial entropy of fixation centroids on a grid (bits)."""
    if len(cx) < 3:
        return float("nan")
    cx = cx[np.isfinite(cx)]
    cy = cy[np.isfinite(cy)]
    if len(cx) < 3:
        return float("nan")
    xr = cx.max() - cx.min() or 1.0
    yr = cy.max() - cy.min() or 1.0
    xi = ((cx - cx.min()) / xr * (grid - 1)).astype(int).clip(0, grid - 1)
    yi = ((cy - cy.min()) / yr * (grid - 1)).astype(int).clip(0, grid - 1)
    h, _, _ = np.histogram2d(xi, yi, bins=grid, range=[[0, grid], [0, grid]])
    h = h / h.sum()
    return _entropy_bits(h.ravel())


def _estimate_training_sequences(
    recording_lengths: List[int],
    context_len: int,
    stride: int = 1,
) -> int:
    """How many (input, target) windows fit across all recordings."""
    return sum(max(0, (n - context_len) // stride) for n in recording_lengths)


# ---------------------------------------------------------------------------
# Per-dataset computation
# ---------------------------------------------------------------------------


def compute_dataset_statistics(
    dataset_name: str,
    root: str,
    out_dir: Path,
    context_len: int = 32,
    stride: int = 1,
    vel_threshold: float = 30.0,
    dispersion_threshold: float = 1,
    min_fix_dur: int = 100,
    min_sac_dur: int = 30,
    vel_method: str = None,
    subset: Optional[dict] = None,
) -> Dict:
    """
    Load one dataset, run event detection on every recording, and return
    a nested dict of statistics.  Also saves per-stimulus JSON files.
    """
    print(f"\n{'=' * 65}")
    print(f"  Dataset: {dataset_name}")
    print(f"{'=' * 65}")

    dataset_paths = pm.DatasetPaths(root=root)
    dataset = pm.Dataset(dataset_name, path=dataset_paths)
    dataset.scan()
    if dataset_name == "mcfw-gaze":  # only include data concerning images
        default_subset = {"trial_id": ["1", "2", "3"], "stimulus": MCFW_STIMULUS}
        if subset is not None:
            default_subset.update(subset)
        subset = default_subset
    print(subset)
    dataset.load(subset=subset)
    if dataset_name == "GGTG":
        dataset.split_gaze_data("stimulus")

    if not dataset.gaze:
        print("  [warn] No gaze objects loaded — skipping")
        return {}

    # ── Screen / experiment metadata from first recording ─────────────────
    first_gaze = dataset.gaze[0]
    screen = first_gaze.experiment.screen
    sr = first_gaze.experiment.sampling_rate
    scr_w_px = screen.width_px
    scr_h_px = screen.height_px

    try:
        scr_w_deg = screen.x_max_dva - screen.x_min_dva
        scr_h_deg = screen.y_max_dva - screen.y_min_dva
    except TypeError:
        scr_w_deg = 2 * np.degrees(
            np.arctan(screen.width_cm / (2 * screen.distance_cm))
        )
        scr_h_deg = 2 * np.degrees(
            np.arctan(screen.height_cm / (2 * screen.distance_cm))
        )

    print(
        f"  Screen : {scr_w_px}×{scr_h_px} px  "
        f"{scr_w_deg:.1f}×{scr_h_deg:.1f}°  sr={sr} Hz"
    )
    print(f"  Recordings: {len(dataset.gaze)}")

    # ── Preprocess entire dataset with pymovements built-ins ──────────────
    print(
        "  Preprocessing: pix2deg → pos2vel → IDT → microsaccades → event properties …"
    )
    preprocessor = GazePreprocessor(
        vel_threshold=vel_threshold,
        dispersion_threshold=dispersion_threshold,
        min_fix_duration=min_fix_dur,
        min_sac_duration=min_sac_dur,
        vel_method=vel_method,
    )
    preprocessor.apply_dataset(dataset, dataset_name)
    print("  Preprocessing complete")

    # ── Iterate over recordings (position / events already populated) ─────
    participants = set()
    stimuli = set()

    recording_durations_s = []
    recording_lengths = []  # samples
    valid_rates = []  # fraction of finite normalised positions

    all_fix_df = []
    all_sac_df = []
    all_norm_pts = []  # (T, 2) normalised arrays for spatial coverage

    by_stimulus: Dict[str, List[Dict]] = defaultdict(list)

    for gaze, ev_frame in tqdm(
        zip(dataset.gaze, dataset.events),
        total=len(dataset.gaze),
        desc="  Processing recordings",
    ):
        subject_id = gaze.metadata.get("subject_id", "?")
        stimulus = gaze.metadata.get("stimulus", "?")
        participants.add(subject_id)
        stimuli.add(stimulus)

        try:
            px_raw = np.stack(gaze.samples["pixel"].to_numpy())  # (T,2) px
            pos_arr = np.stack(gaze.samples["position"].to_numpy())  # (T,2) deg
            time_arr = gaze.samples["time"].to_numpy()  # (T,) timestamps
            norm_arr = np.column_stack(
                [px_raw[:, 0] / scr_w_px, px_raw[:, 1] / scr_h_px]
            )
        except Exception as e:
            tqdm.write(f"    [warn] {subject_id}/{stimulus}: {e}")
            continue

        valid_mask = np.all(np.isfinite(norm_arr), axis=1)
        valid_rates.append(float(valid_mask.mean()))

        T = len(norm_arr)
        recording_lengths.append(T)
        recording_durations_s.append(T / sr)

        fix_df = _fix_df_from_events(ev_frame.frame, pos_arr, time_arr)
        sac_df = _sac_df_from_events(ev_frame.frame, pos_arr, time_arr)
        all_fix_df.append(fix_df)
        all_sac_df.append(sac_df)
        all_norm_pts.append(norm_arr)

        by_stimulus[stimulus].append(
            {
                "subject_id": subject_id,
                "fix_df": fix_df,
                "sac_df": sac_df,
                "norm_arr": norm_arr,
                "n_samples": T,
            }
        )

    if not all_fix_df:
        print("  [warn] No valid recordings found")
        return {}

    fix_all = pl.concat(all_fix_df)
    sac_all = pl.concat(all_sac_df)

    # ── Fixation statistics ────────────────────────────────────────────────
    fix_dur_samples = fix_all["duration"].to_numpy().astype(float)
    fix_dur_ms = fix_dur_samples
    n_fix_per_rec = np.array([len(f) for f in all_fix_df], dtype=float)

    fix_stats = {
        "total_fixations": int(len(fix_all)),
        "mean_per_recording": float(_safe(n_fix_per_rec, np.mean)),
        "std_per_recording": float(_safe(n_fix_per_rec, np.std)),
        "duration_mean_ms": float(_safe(fix_dur_ms, np.mean)),
        "duration_std_ms": float(_safe(fix_dur_ms, np.std)),
        "duration_median_ms": float(_safe(fix_dur_ms, np.median)),
        "duration_p25_ms": float(_safe(fix_dur_ms, lambda x: np.percentile(x, 25))),
        "duration_p75_ms": float(_safe(fix_dur_ms, lambda x: np.percentile(x, 75))),
        "pct_within_physio_range": float(
            np.mean(
                (fix_dur_ms >= PHYSIO_FIX_MIN_MS) & (fix_dur_ms <= PHYSIO_FIX_MAX_MS)
            )
            * 100
            if len(fix_dur_ms) > 0
            else float("nan")
        ),
    }

    # ── Saccade statistics ─────────────────────────────────────────────────
    sac_amp = sac_all["amplitude_deg"].to_numpy()
    sac_pv = sac_all["peak_vel_deg_s"].to_numpy()
    sac_ang = (
        sac_all["angle_rad"].drop_nulls().to_numpy()
        if "angle_rad" in sac_all.columns
        else np.array([])
    )
    sac_ang = sac_ang[np.isfinite(sac_ang)]
    n_sac_per_rec = np.array([len(s) for s in all_sac_df], dtype=float)

    # Main sequence
    mask = (sac_amp > 0.1) & (sac_pv > 1.0)
    ms_r = float("nan")
    if mask.sum() >= 5:
        ms_r, _ = stats.pearsonr(sac_amp[mask], sac_pv[mask])
        ms_r = float(ms_r)

    # Saccade direction entropy
    dir_entropy = float("nan")
    if len(sac_ang) >= 3:
        h, _ = np.histogram(sac_ang, bins=16, range=(-np.pi, np.pi))
        h_n = (h + 1e-8) / (h + 1e-8).sum()
        dir_entropy = _entropy_bits(h_n)

    sac_stats = {
        "total_saccades": int(len(sac_all)),
        "mean_per_recording": float(_safe(n_sac_per_rec, np.mean)),
        "std_per_recording": float(_safe(n_sac_per_rec, np.std)),
        "amplitude_mean_deg": float(_safe(sac_amp, np.mean)),
        "amplitude_std_deg": float(_safe(sac_amp, np.std)),
        "amplitude_median_deg": float(_safe(sac_amp, np.median)),
        "amplitude_p25_deg": float(_safe(sac_amp, lambda x: np.percentile(x, 25))),
        "amplitude_p75_deg": float(_safe(sac_amp, lambda x: np.percentile(x, 75))),
        "peak_velocity_mean_deg_s": float(_safe(sac_pv, np.mean)),
        "peak_velocity_std_deg_s": float(_safe(sac_pv, np.std)),
        "main_sequence_r": ms_r,
        "direction_entropy_bits": dir_entropy,
        "pct_within_physio_range": float(
            np.mean((sac_amp >= PHYSIO_SAC_MIN_DEG) & (sac_amp <= PHYSIO_SAC_MAX_DEG))
            * 100
            if len(sac_amp) > 0
            else float("nan")
        ),
    }

    # ── Spatial statistics ─────────────────────────────────────────────────
    cx_all = (
        fix_all["cx_deg"].to_numpy() if "cx_deg" in fix_all.columns else np.array([])
    )
    cy_all = (
        fix_all["cy_deg"].to_numpy() if "cy_deg" in fix_all.columns else np.array([])
    )

    spatial_stats = {
        "fixation_density_entropy_bits": _density_entropy(cx_all, cy_all),
        "mean_cx_deg": float(_safe(cx_all, np.mean)),
        "mean_cy_deg": float(_safe(cy_all, np.mean)),
        "std_cx_deg": float(_safe(cx_all, np.std)),
        "std_cy_deg": float(_safe(cy_all, np.std)),
    }

    # ── Data volume ────────────────────────────────────────────────────────
    dur_arr = np.array(recording_durations_s)
    total_seq = _estimate_training_sequences(recording_lengths, context_len, stride)

    volume_stats = {
        "total_gaze_samples": int(sum(recording_lengths)),
        "valid_sample_rate_pct": float(np.mean(valid_rates) * 100),
        "total_recording_duration_s": float(dur_arr.sum()),
        "mean_recording_duration_s": float(_safe(dur_arr, np.mean)),
        "std_recording_duration_s": float(_safe(dur_arr, np.std)),
        "min_recording_duration_s": float(_safe(dur_arr, np.min)),
        "max_recording_duration_s": float(_safe(dur_arr, np.max)),
        "estimated_training_sequences": int(total_seq),
        "context_len_used": context_len,
        "stride_used": stride,
    }

    # ── Per-stimulus statistics ───────────────────────────────────────────
    stim_dir = out_dir / "per_stimulus"
    stim_dir.mkdir(parents=True, exist_ok=True)

    per_stimulus = {}
    for stim, recordings in by_stimulus.items():
        s_fix = (
            pl.concat([r["fix_df"] for r in recordings if len(r["fix_df"]) > 0])
            if any(len(r["fix_df"]) > 0 for r in recordings)
            else pl.DataFrame()
        )
        s_sac = (
            pl.concat([r["sac_df"] for r in recordings if len(r["sac_df"]) > 0])
            if any(len(r["sac_df"]) > 0 for r in recordings)
            else pl.DataFrame()
        )

        s_dur_ms = (
            s_fix["duration"].to_numpy().astype(float) / sr * 1000
            if len(s_fix) > 0
            else np.array([])
        )
        s_amp = s_sac["amplitude_deg"].to_numpy() if len(s_sac) > 0 else np.array([])

        cx = (
            s_fix["cx_deg"].to_numpy()
            if ("cx_deg" in s_fix.columns and len(s_fix) > 0)
            else np.array([])
        )
        cy = (
            s_fix["cy_deg"].to_numpy()
            if ("cy_deg" in s_fix.columns and len(s_fix) > 0)
            else np.array([])
        )

        entry = {
            "n_participants": len({r["subject_id"] for r in recordings}),
            "n_recordings": len(recordings),
            "n_fixations": int(len(s_fix)),
            "fixation_duration_mean_ms": float(_safe(s_dur_ms, np.mean)),
            "fixation_duration_std_ms": float(_safe(s_dur_ms, np.std)),
            "n_saccades": int(len(s_sac)),
            "saccade_amplitude_mean_deg": float(_safe(s_amp, np.mean)),
            "saccade_amplitude_std_deg": float(_safe(s_amp, np.std)),
            "spatial_entropy_bits": _density_entropy(cx, cy),
        }
        per_stimulus[stim] = entry

        safe = stim.replace("/", "_").replace(" ", "_")
        (stim_dir / f"{safe}.json").write_text(json.dumps(entry, indent=2))

    # ── Assemble full stats dict ──────────────────────────────────────────
    dataset_stats = {
        "dataset_name": dataset_name,
        "overview": {
            "n_participants": len(participants),
            "n_stimuli": len(stimuli),
            "n_recordings": len(dataset.gaze),
            "n_valid_recordings": len(all_fix_df),
            "sampling_rate_hz": float(sr),
            "screen_width_px": int(scr_w_px),
            "screen_height_px": int(scr_h_px),
            "screen_w_deg": float(scr_w_deg),
            "screen_h_deg": float(scr_h_deg),
        },
        "data_volume": volume_stats,
        "fixations": fix_stats,
        "saccades": sac_stats,
        "spatial": spatial_stats,
        "per_stimulus": per_stimulus,
    }

    # ── Save JSON + report ────────────────────────────────────────────────
    (out_dir / "dataset_stats.json").write_text(json.dumps(dataset_stats, indent=2))
    report = build_stats_report(dataset_stats)
    (out_dir / "dataset_report.txt").write_text(report)
    print(report)

    # ── Generate plots ────────────────────────────────────────────────────
    raw_data = {
        "fix_dur_ms": fix_dur_ms,
        "sac_amp": sac_amp,
        "sac_pv": sac_pv,
        "sac_ang": sac_ang,
        "cx_all": cx_all,
        "cy_all": cy_all,
        "dur_arr": dur_arr,
        "scr_w_deg": scr_w_deg,
        "scr_h_deg": scr_h_deg,
        "by_stimulus": by_stimulus,
        "sr": sr,
    }

    _plot_overview(dataset_stats, raw_data, out_dir / "overview.png")
    _plot_fixation_duration(fix_dur_ms, out_dir / "fixation_duration.png")
    _plot_saccade_amplitude(sac_amp, out_dir / "saccade_amplitude.png")
    _plot_main_sequence(sac_amp, sac_pv, ms_r, out_dir / "main_sequence.png")
    _plot_saccade_direction(sac_ang, out_dir / "saccade_direction.png")
    _plot_spatial_coverage(
        cx_all, cy_all, scr_w_deg, scr_h_deg, out_dir / "spatial_coverage.png"
    )
    _plot_per_stimulus_density(
        by_stimulus, sr, scr_w_deg, scr_h_deg, out_dir / "per_stimulus_density.png"
    )

    return dataset_stats


# ---------------------------------------------------------------------------
# Entry point  (plots and report live in kaamba.utils.stats_plots / stats_report)
# ---------------------------------------------------------------------------


def run_dataset_stats(
    datasets: List[str],
    root: str,
    out_dir: str,
    context_len: int = 32,
    stride: int = 1,
    vel_threshold: float = 30.0,
    min_fix_dur: int = 10,
    min_sac_dur: int = 10,
    vel_method: str = None,
    dispersion_threshold: float = 1.0,
    subset: Optional[dict] = None,
) -> Dict[str, Dict]:
    base_dir = Path(out_dir)
    all_stats = {}

    for dataset_name in datasets:
        ds_dir = base_dir / dataset_name
        ds_dir.mkdir(parents=True, exist_ok=True)

        s = compute_dataset_statistics(
            dataset_name=dataset_name,
            root=root,
            out_dir=ds_dir,
            context_len=context_len,
            stride=stride,
            vel_threshold=vel_threshold,
            dispersion_threshold=dispersion_threshold,
            min_sac_dur=min_sac_dur,
            min_fix_dur=min_fix_dur,
            vel_method=vel_method,
            subset=subset,
        )
        if s:
            all_stats[dataset_name] = s

    if len(all_stats) > 1:
        _plot_comparison(all_stats, base_dir / "comparison.png")
        # Save combined JSON
        (base_dir / "all_datasets.json").write_text(json.dumps(all_stats, indent=2))
        print(f"\n[stats] Comparison plot → {base_dir / 'comparison.png'}")

    print(f"\n[stats] All output in {base_dir}")
    return all_stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse():
    p = argparse.ArgumentParser(
        description="Descriptive statistics and plots for gaze training datasets"
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["GGTG"],
        help="pymovements dataset names, e.g. mcfw-gaze GGTG",
    )
    p.add_argument(
        "--root",
        default=r"C:\Users\saphi\PycharmProjects\thesis\data",
        help="Root directory for pymovements data",
    )
    p.add_argument(
        "--out_dir",
        default="outputs/dataset_stats",
        help="Output directory",
    )
    p.add_argument(
        "--context_len",
        type=int,
        default=3200,
        help="Context length used to estimate training sequences",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=8,
        help="Stride used to estimate training sequences",
    )
    p.add_argument(
        "--vel_threshold",
        type=float,
        default=30,
        help="IDT threshold 30 for ggtg and 0.003 for mfcw",
    )
    p.add_argument(
        "--dispersion_threshold",
        type=float,
        default=1.0,
        help="IDT dispresion threshold in deg/visual angle 1 for ggtg and 0.0001 for mfcw",
    )
    p.add_argument(
        "--min_fix_dur",
        type=int,
        default=98,
        help="Minimum fixation duration in samples",
    )
    p.add_argument(
        "--min_sac_dur",
        type=int,
        default=18,
        help="Minimum saccade duration in samples",
    )
    p.add_argument(
        "--vel_method",
        type=str,
        default="fivepoint",
        choices=["fivepoint", "preceding", "savitzky_golay", "neighbors"],
        help="method to compute velocities",
    )
    p.add_argument(
        "--subjects", nargs="*", default=["P01"], help="Limit to specific subject IDs"
    )
    return p.parse_args()


def main():
    args = _parse()
    subset = {"subject_id": args.subjects} if args.subjects else None
    print(subset)
    run_dataset_stats(
        datasets=args.datasets,
        root=args.root,
        out_dir=args.out_dir,
        context_len=args.context_len,
        stride=args.stride,
        vel_threshold=args.vel_threshold,
        dispersion_threshold=args.dispersion_threshold,
        min_fix_dur=args.min_fix_dur,
        min_sac_dur=args.min_sac_dur,
        vel_method=args.vel_method,
        subset=subset,
    )


if __name__ == "__main__":
    main()
