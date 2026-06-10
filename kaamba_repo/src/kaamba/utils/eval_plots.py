"""
eval_plots.py

Plot helpers for evaluate_model.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import stats
from scipy.ndimage import gaussian_filter

C_REAL = "#1D9E75"  # Teal
C_FAKE = "#7F77DD"  # purple


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
