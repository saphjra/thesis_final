"""
stats_plots.py

Plot helpersfor dataset_stats.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats
from scipy.ndimage import gaussian_filter

# ── shared style ──────────────────────────────────────────────────────────────
C1 = "#1D9E75"  # teal  — matches evaluate_model.py real-data colour
C2 = "#7F77DD"  # purple
C_GRID = "#E8E6DE"
C_TEXT = "#2C2C2A"
PHYSIO_FIX_MIN_MS = 20.0  # minimum plausible fixation duration
PHYSIO_FIX_MAX_MS = 800.0  # maximum plausible fixation duration
PHYSIO_SAC_MIN_DEG = 30
PHYSIO_SAC_MAX_DEG = 500.0


def _save(fig: plt.Figure, path: Path, dpi: int = 150):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [plot] {path.name}")


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
