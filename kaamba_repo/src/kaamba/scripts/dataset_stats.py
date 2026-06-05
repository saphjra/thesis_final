"""
dataset_stats.py

Descriptive statistics and visualisations for the gaze datasets used in
training. Event detection uses the exact same IDT + fill pipeline as
evaluate_model.py, so all numbers are directly comparable to evaluation
results and can be cited side-by-side in a thesis.

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
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from kaamba.utils.gaze_preprocessing import GazePreprocessor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── shared style ──────────────────────────────────────────────────────────────
C1 = "#1D9E75"  # teal  — matches evaluate_model.py real-data colour
C2 = "#7F77DD"  # purple
C_GRID = "#E8E6DE"
C_TEXT = "#2C2C2A"
PHYSIO_FIX_MIN_MS = 20.0  # minimum plausible fixation duration
PHYSIO_FIX_MAX_MS = 800.0  # maximum plausible fixation duration
PHYSIO_SAC_MIN_DEG = 30
PHYSIO_SAC_MAX_DEG = 500.0
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

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#BBBBBB",
        "axes.linewidth": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


# ---------------------------------------------------------------------------
# pymovements-native preprocessing helpers
# (mirrors evaluate_model.py — dataset.pix2deg / pos2vel / detect_events /
#  compute_event_properties are called at dataset level before the loop)
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
        vel_method="fivepoint",
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
        },
        "data_volume": volume_stats,
        "fixations": fix_stats,
        "saccades": sac_stats,
        "spatial": spatial_stats,
        "per_stimulus": per_stimulus,
    }

    # ── Save JSON + report ────────────────────────────────────────────────
    (out_dir / "dataset_stats.json").write_text(json.dumps(dataset_stats, indent=2))
    report = _build_report(dataset_stats)
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
# Plots
# ---------------------------------------------------------------------------


def _plot_overview(ds: Dict, raw: Dict, out_path: Path):
    """4-panel overview: recording counts, duration dist, fixation counts, sequence estimate."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)

    ov = ds["overview"]
    dv = ds["data_volume"]
    # fx = ds["fixations"]

    # col 0: high-level numbers as a simple text block
    ax = axes[0]
    ax.axis("off")
    lines = [
        f"Participants :  {ov['n_participants']}",
        f"Stimuli      :  {ov['n_stimuli']}",
        f"Recordings   :  {ov['n_valid_recordings']} / {ov['n_recordings']}",
        f"Sampling rate:  {ov['sampling_rate_hz']:.0f} Hz",
        f"Screen       :  {ov['screen_width_px']}×{ov['screen_height_px']} px",
        f"             :  {ov['screen_w_deg']:.1f}×{ov['screen_h_deg']:.1f}°",
        f"Valid samples:  {dv['valid_sample_rate_pct']:.1f}%",
        f"Total samples:  {dv['total_gaze_samples']:,}",
        f"Train seqs   :  {dv['estimated_training_sequences']:,}",
        f"  (ctx={dv['context_len_used']}, stride={dv['stride_used']})",
    ]
    ax.text(
        0.05,
        0.95,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        family="monospace",
        color=C_TEXT,
        linespacing=1.7,
    )
    ax.set_title("Dataset overview", fontsize=9, loc="left", pad=6)

    # col 1: recording duration histogram
    ax = axes[1]
    dur = raw["dur_arr"]
    ax.hist(
        dur,
        bins=min(40, len(dur) // 2 + 1),
        color=C1,
        alpha=0.8,
        edgecolor="white",
        lw=0.4,
    )
    ax.set_xlabel("Recording duration (s)")
    ax.set_ylabel("Count")
    ax.set_title("Recording durations", fontsize=9, loc="left")

    # col 2: fixation duration distribution
    ax = axes[2]
    fd = raw["fix_dur_ms"]
    fd = fd[(fd >= 0) & (fd < 2000)]
    if len(fd) >= 3:
        xs = np.linspace(0, 2000, 400)
        kde = stats.gaussian_kde(fd, bw_method=0.2)(xs)
        ax.fill_between(xs, kde, alpha=0.3, color=C1)
        ax.plot(xs, kde, color=C1, lw=1.5)
    ax.axvspan(
        PHYSIO_FIX_MIN_MS,
        PHYSIO_FIX_MAX_MS,
        color="#CCCCCC",
        alpha=0.25,
        label="100–800 ms band",
    )
    ax.set_xlabel("Fixation duration (ms)")
    ax.set_title("Fixation durations", fontsize=9, loc="left")
    ax.set_yticks([])
    ax.legend(fontsize=7, frameon=False)

    # col 3: per-stimulus recordings bar (top 20 stimuli by count)
    ax = axes[3]
    by_stim = ds["per_stimulus"]
    counts = sorted(
        [(s, d["n_recordings"]) for s, d in by_stim.items()], key=lambda x: -x[1]
    )[:20]
    if counts:
        names, vals = zip(*counts)
        y_pos = np.arange(len(names))
        ax.barh(y_pos, vals, color=C1, alpha=0.8, height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([n[:22] for n in names], fontsize=6)
        ax.invert_yaxis()
        ax.set_xlabel("Recordings")
        ax.set_title("Recordings per stimulus\n(top 20)", fontsize=9, loc="left")
    else:
        ax.axis("off")

    fig.suptitle(
        f"Dataset: {ds['dataset_name']}",
        fontsize=10,
        y=1.01,
        color=C_TEXT,
        fontweight=500,
    )
    _save(fig, out_path)


def _plot_fixation_duration(fix_dur_ms: np.ndarray, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

    fd = fix_dur_ms[(fix_dur_ms >= 0) & (fix_dur_ms < 2000)]
    if len(fd) >= 3:
        xs = np.linspace(0, 1500, 500)
        kde = stats.gaussian_kde(fd, bw_method=0.15)(xs)
        ax.fill_between(xs, kde, alpha=0.25, color=C1)
        ax.plot(xs, kde, color=C1, lw=2, label=f"n = {len(fd):,}")
        # Reference lines
        ax.axvline(
            np.median(fd),
            color=C1,
            lw=1.2,
            ls="--",
            label=f"median = {np.median(fd):.0f} ms",
        )
        ax.axvline(
            np.mean(fd),
            color="#888880",
            lw=1.0,
            ls=":",
            label=f"mean   = {np.mean(fd):.0f} ms",
        )
    ax.axvspan(
        PHYSIO_FIX_MIN_MS,
        PHYSIO_FIX_MAX_MS,
        color="#CCCCCC",
        alpha=0.3,
        zorder=0,
        label="Plausible range",
    )
    ax.set_xlabel("Fixation duration (ms)")
    ax.set_ylabel("Density")
    ax.set_title("Fixation duration distribution", loc="left")
    ax.legend(fontsize=8, frameon=False)
    ax.set_yticks([])
    _save(fig, out_path)


def _plot_saccade_amplitude(sac_amp: np.ndarray, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

    sa = sac_amp[(sac_amp >= 0) & (sac_amp < 30)]
    if len(sa) >= 3:
        xs = np.linspace(0, 25, 400)
        kde = stats.gaussian_kde(sa, bw_method=0.2)(xs)
        ax.fill_between(xs, kde, alpha=0.25, color=C1)
        ax.plot(xs, kde, color=C1, lw=2, label=f"n = {len(sa):,}")
        ax.axvline(
            np.median(sa),
            color=C1,
            lw=1.2,
            ls="--",
            label=f"median = {np.median(sa):.1f}°",
        )
        ax.axvline(
            np.mean(sa),
            color="#888880",
            lw=1.0,
            ls=":",
            label=f"mean   = {np.mean(sa):.1f}°",
        )
    ax.axvspan(
        PHYSIO_SAC_MIN_DEG,
        PHYSIO_SAC_MAX_DEG,
        color="#CCCCCC",
        alpha=0.3,
        zorder=0,
        label="Plausible range",
    )
    ax.set_xlabel("Saccade amplitude (°)")
    ax.set_ylabel("Density")
    ax.set_title("Saccade amplitude distribution", loc="left")
    ax.legend(fontsize=8, frameon=False)
    ax.set_yticks([])
    _save(fig, out_path)


def _plot_main_sequence(
    sac_amp: np.ndarray, sac_pv: np.ndarray, ms_r: float, out_path: Path
):
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)

    mask = (sac_amp > 0.1) & (sac_pv > 1.0) & np.isfinite(sac_amp) & np.isfinite(sac_pv)
    amp, pv = sac_amp[mask], sac_pv[mask]

    # Subsample for display if very large
    if len(amp) > 5000:
        idx = np.random.choice(len(amp), 5000, replace=False)
        amp_s, pv_s = amp[idx], pv[idx]
    else:
        amp_s, pv_s = amp, pv

    ax.scatter(amp_s, pv_s, s=4, alpha=0.25, color=C1, linewidths=0, rasterized=True)

    if len(amp) >= 5:
        # OLS regression in log-space (standard for main sequence)
        log_a, log_v = np.log(amp + 1e-6), np.log(pv + 1e-6)
        slope, intercept, *_ = stats.linregress(log_a, log_v)
        xs = np.linspace(amp.min(), amp.max(), 200)
        ax.plot(
            xs,
            np.exp(intercept + slope * np.log(xs + 1e-6)),
            color=C2,
            lw=2,
            label=f"OLS fit  r = {ms_r:.3f}",
        )

    ax.set_xlabel("Saccade amplitude (°)")
    ax.set_ylabel("Peak velocity (°/s)")
    ax.set_title("Main sequence", loc="left")
    if not np.isnan(ms_r):
        verdict = "✓ main sequence holds" if ms_r > 0.9 else "✗ r < 0.9"
        ax.text(
            0.97,
            0.05,
            verdict,
            transform=ax.transAxes,
            ha="right",
            fontsize=8,
            color=C1 if ms_r > 0.9 else "#E8593C",
        )
    ax.legend(fontsize=8, frameon=False)
    _save(fig, out_path)


def _plot_saccade_direction(sac_ang: np.ndarray, out_path: Path):
    """Polar rose chart of saccade directions."""
    fig = plt.figure(figsize=(5, 5), constrained_layout=True)
    ax = fig.add_subplot(111, polar=True)

    n_bins = 16
    ang = sac_ang[np.isfinite(sac_ang)]
    if len(ang) >= 3:
        h, edges = np.histogram(ang, bins=n_bins, range=(-np.pi, np.pi))
        theta = (edges[:-1] + edges[1:]) / 2
        width = 2 * np.pi / n_bins
        ax.bar(
            theta,
            h / h.sum(),
            width=width,
            color=C1,
            alpha=0.75,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_title("Saccade direction distribution", fontsize=9, pad=18)
    ax.set_yticks([])
    _save(fig, out_path)


def _plot_spatial_coverage(
    cx: np.ndarray, cy: np.ndarray, scr_w_deg: float, scr_h_deg: float, out_path: Path
):
    """Gaussian-smoothed fixation centroid density heatmap."""
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)

    cx = cx[np.isfinite(cx)]
    cy = cy[np.isfinite(cy)]
    if len(cx) >= 3 and len(cy) == len(cx):
        grid = 64
        xi = (
            ((cx - cx.min()) / (cx.max() - cx.min() + 1e-9) * (grid - 1))
            .astype(int)
            .clip(0, grid - 1)
        )
        yi = (
            ((cy - cy.min()) / (cy.max() - cy.min() + 1e-9) * (grid - 1))
            .astype(int)
            .clip(0, grid - 1)
        )
        h = np.zeros((grid, grid))
        for x, y in zip(xi, yi):
            h[y, x] += 1
        h = gaussian_filter(h.astype(float) + 1e-8, sigma=2.0)
        im = ax.imshow(
            h / h.sum(),
            cmap="YlOrRd",
            origin="lower",
            aspect="auto",
            extent=[cx.min(), cx.max(), cy.min(), cy.max()],
        )
        cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Fixation density", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        # Centre cross
        ax.axhline(np.mean(cy), color="white", lw=0.8, ls="--", alpha=0.7)
        ax.axvline(np.mean(cx), color="white", lw=0.8, ls="--", alpha=0.7)
    else:
        ax.text(
            0.5,
            0.5,
            "insufficient data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.set_xlabel("Horizontal position (°)")
    ax.set_ylabel("Vertical position (°)")
    ax.set_title("Spatial fixation density (all stimuli)", loc="left")
    _save(fig, out_path)


def _plot_per_stimulus_density(
    by_stimulus: Dict,
    sr: float,
    scr_w_deg: float,
    scr_h_deg: float,
    out_path: Path,
    max_stimuli: int = 20,
    grid_cols: int = 5,
):
    """Grid of fixation density maps, one panel per stimulus."""
    # Rank by number of recordings, take top N
    ranked = sorted(
        by_stimulus.items(), key=lambda kv: -sum(len(r["fix_df"]) for r in kv[1])
    )
    ranked = ranked[:max_stimuli]

    n = len(ranked)
    if n == 0:
        return
    nc = min(grid_cols, n)
    nr = (n + nc - 1) // nc

    fig, axes = plt.subplots(
        nr, nc, figsize=(nc * 2.8, nr * 2.6), constrained_layout=True
    )
    if nr == 1 and nc == 1:
        axes = np.array([[axes]])
    elif nr == 1 or nc == 1:
        axes = np.array(axes).reshape(nr, nc)

    for idx, (stim, recordings) in enumerate(ranked):
        ax = axes[idx // nc][idx % nc]

        s_fix = (
            pl.concat([r["fix_df"] for r in recordings if len(r["fix_df"]) > 0])
            if any(len(r["fix_df"]) > 0 for r in recordings)
            else pl.DataFrame()
        )

        if len(s_fix) > 0 and "cx_deg" in s_fix.columns:
            cx = s_fix["cx_deg"].to_numpy()
            cy = s_fix["cy_deg"].to_numpy()
            cx = cx[np.isfinite(cx)]
            cy = cy[np.isfinite(cy)]

            if len(cx) >= 3:
                g = 32
                cx_n = (
                    ((cx - cx.min()) / (cx.max() - cx.min() + 1e-9) * (g - 1))
                    .astype(int)
                    .clip(0, g - 1)
                )
                cy_n = (
                    ((cy - cy.min()) / (cy.max() - cy.min() + 1e-9) * (g - 1))
                    .astype(int)
                    .clip(0, g - 1)
                )
                h = np.zeros((g, g))
                for x, y in zip(cx_n, cy_n):
                    h[y, x] += 1
                h = gaussian_filter(h.astype(float) + 1e-8, sigma=1.5)
                ax.imshow(h / h.sum(), cmap="YlOrRd", origin="lower", aspect="auto")

        n_rec = len(recordings)
        n_fix = len(s_fix)
        ax.set_title(f"{stim[:20]}\n{n_rec} rec · {n_fix} fix", fontsize=6.5, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused axes
    for idx in range(len(ranked), nr * nc):
        axes[idx // nc][idx % nc].axis("off")

    fig.suptitle("Per-stimulus fixation density maps", fontsize=9, y=1.01)
    _save(fig, out_path)


def _save(fig: plt.Figure, path: Path, dpi: int = 150):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [plot] {path.name}")


# ---------------------------------------------------------------------------
# Human-readable report
# ---------------------------------------------------------------------------


def _build_report(s: Dict) -> str:
    ov = s["overview"]
    dv = s["data_volume"]
    fx = s["fixations"]
    sa = s["saccades"]
    sp = s["spatial"]

    lines = [
        "=" * 65,
        f"DATASET DESCRIPTIVE STATISTICS — {s['dataset_name']}",
        "=" * 65,
        "",
        "OVERVIEW",
        "-" * 65,
        f"  Participants          : {ov['n_participants']}",
        f"  Stimuli               : {ov['n_stimuli']}",
        f"  Recordings (valid)    : {ov['n_valid_recordings']} / {ov['n_recordings']}",
        f"  Sampling rate         : {ov['sampling_rate_hz']:.0f} Hz",
        f"  Screen resolution     : {ov['screen_width_px']}×{ov['screen_height_px']} px",
        f"  Screen extent         : {ov['screen_w_deg']:.1f}×{ov['screen_h_deg']:.1f}°",
        "",
        "DATA VOLUME",
        "-" * 65,
        f"  Total gaze samples    : {dv['total_gaze_samples']:,}",
        f"  Valid sample rate     : {dv['valid_sample_rate_pct']:.1f}%",
        f"  Total recording time  : {dv['total_recording_duration_s']:.1f} s  "
        f"({dv['total_recording_duration_s'] / 60:.1f} min)",
        f"  Rec. duration mean±std: {dv['mean_recording_duration_s']:.1f} ± "
        f"{dv['std_recording_duration_s']:.1f} s",
        f"  Rec. duration range   : {dv['min_recording_duration_s']:.1f} – "
        f"{dv['max_recording_duration_s']:.1f} s",
        f"  Est. training seqs    : {dv['estimated_training_sequences']:,}  "
        f"(ctx={dv['context_len_used']}, stride={dv['stride_used']})",
        "",
        "FIXATIONS",
        "-" * 65,
        f"  Total                 : {fx['total_fixations']:,}",
        f"  Per recording mean±std: {fx['mean_per_recording']:.1f} ± {fx['std_per_recording']:.1f}",
        f"  Duration mean±std     : {fx['duration_mean_ms']:.1f} ± {fx['duration_std_ms']:.1f} ms",
        f"  Duration median [IQR] : {fx['duration_median_ms']:.1f}  "
        f"[{fx['duration_p25_ms']:.1f} – {fx['duration_p75_ms']:.1f}] ms",
        f"  In 100–800 ms range   : {fx['pct_within_physio_range']:.1f}%",
        "",
        "SACCADES",
        "-" * 65,
        f"  Total                 : {sa['total_saccades']:,}",
        f"  Per recording mean±std: {sa['mean_per_recording']:.1f} ± {sa['std_per_recording']:.1f}",
        f"  Amplitude mean±std    : {sa['amplitude_mean_deg']:.2f} ± {sa['amplitude_std_deg']:.2f}°",
        f"  Amplitude median [IQR]: {sa['amplitude_median_deg']:.2f}  "
        f"[{sa['amplitude_p25_deg']:.2f} – {sa['amplitude_p75_deg']:.2f}]°",
        f"  Peak velocity mean±std: {sa['peak_velocity_mean_deg_s']:.1f} ± "
        f"{sa['peak_velocity_std_deg_s']:.1f} °/s",
        f"  Main sequence r       : {sa['main_sequence_r']:.4f}  "
        f"({'✓ > 0.9' if sa['main_sequence_r'] > 0.9 else '✗ < 0.9'})",
        f"  Direction entropy     : {sa['direction_entropy_bits']:.3f} bits  "
        f"(max = {np.log2(16):.2f} bits for 16 bins)",
        f"  In 0.5–20° range      : {sa['pct_within_physio_range']:.1f}%",
        "",
        "SPATIAL COVERAGE",
        "-" * 65,
        f"  Fixation density entropy: {sp['fixation_density_entropy_bits']:.3f} bits",
        f"  Mean fixation (cx, cy)  : ({sp['mean_cx_deg']:.2f}°, {sp['mean_cy_deg']:.2f}°)",
        f"  Std  fixation (cx, cy)  : ({sp['std_cx_deg']:.2f}°, {sp['std_cy_deg']:.2f}°)",
        "",
        "PER-STIMULUS SUMMARY (top 10 by fixation count)",
        "-" * 65,
        f"  {'Stimulus':<28} {'Recs':>5} {'Fix':>6} {'Dur(ms)':>9} {'Amp(°)':>8}",
    ]

    top_stim = sorted(s["per_stimulus"].items(), key=lambda kv: -kv[1]["n_fixations"])[
        :10
    ]
    for stim, d in top_stim:
        lines.append(
            f"  {stim[:28]:<28} {d['n_recordings']:>5} {d['n_fixations']:>6} "
            f"{d['fixation_duration_mean_ms']:>9.1f} {d['saccade_amplitude_mean_deg']:>8.2f}"
        )

    lines += ["", "=" * 65]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cross-dataset comparison plot
# ---------------------------------------------------------------------------


def _plot_comparison(all_stats: Dict[str, Dict], out_path: Path):
    """Side-by-side bar charts comparing key metrics across datasets."""
    datasets = list(all_stats.keys())
    n = len(datasets)
    if n < 2:
        return

    colours = [C1, C2, "#E8593C", "#F5A623"][:n]

    metrics = [
        ("fixations", "duration_mean_ms", "Fix. duration\n(ms)", None),
        ("fixations", "pct_within_range", "Fix. in 50–800 ms\n(%)", None),
        ("saccades", "amplitude_mean_deg", "Saccade amplitude\n(°)", None),
        ("saccades", "main_sequence_r", "Main sequence r", 0.9),
        ("saccades", "direction_entropy_bits", "Direction entropy\n(bits)", None),
        ("spatial", "fixation_density_entropy_bits", "Spatial entropy\n(bits)", None),
    ]

    fig, axes = plt.subplots(
        1, len(metrics), figsize=(len(metrics) * 2.8, 5), constrained_layout=True
    )

    for ax, (section, key, label, threshold) in zip(axes, metrics):
        vals = [all_stats[d].get(section, {}).get(key, float("nan")) for d in datasets]
        x = np.arange(n)
        bars = ax.bar(x, vals, color=colours, alpha=0.85, width=0.6, edgecolor="white")
        if threshold is not None:
            ax.axhline(
                threshold,
                color="#E8593C",
                lw=1.2,
                ls="--",
                alpha=0.8,
                label=f"threshold = {threshold}",
            )
            ax.legend(fontsize=7, frameon=False)
        ax.set_xticks(x)
        ax.set_xticklabels([d[:12] for d in datasets], fontsize=7, rotation=15)
        ax.set_ylabel(label, fontsize=8)
        ax.set_title(label, fontsize=8, loc="left")
        # Value labels on bars
        for bar, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01 * bar.get_height(),
                    f"{v:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    fig.suptitle("Dataset comparison", fontsize=10, y=1.02)
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Entry point
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
        default=r"C:\Users\saphi\PycharmProjects/thesis/logs/dataset_stats",
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
        subset=subset,
    )


if __name__ == "__main__":
    main()
