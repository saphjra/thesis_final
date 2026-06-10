"""
stats_report.py

Human-readable report builder for dataset_stats.py.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def build_stats_report(s: Dict) -> str:
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
