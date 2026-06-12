"""
eval_plots.py

Plot helpers for evaluate_model.py.

Plot-cache format (condition-keyed)
─────────────────────────────────────
plot_cache = {
    stim_name: {
        "img_path": Path | None,
        "real":        {"seqs": np.ndarray (N,T,2), "fix_df": pl.DataFrame, "sac_df": pl.DataFrame},
        "<gen_name>":  {"seqs": ..., "fix_df": ..., "sac_df": ...},
        ...
    },
    ...
}

Public API
──────────
Individual metric figures (one figure, all conditions overlaid):
    plot_fixation_duration(conditions_data, out_path, sr, title)
    plot_saccade_amplitude(conditions_data, out_path, title)
    plot_main_sequence(conditions_data, out_path, title)
    plot_saccade_direction(conditions_data, out_path, title)
    plot_fixation_density(conditions_data, out_path, title)

Wrappers that call all five:
    plot_aggregate_metrics(plot_cache, out_dir, sr)   -- pools all stimuli
    plot_per_stimulus_metrics(plot_cache, out_dir, sr) -- one set per stimulus

Other figures:
    plot_best_worst_comparison(all_results, plot_cache, out_path, ...)
    plot_scanpath_overview(plot_cache, out_path, ...)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from PIL import Image
from scipy import stats
from scipy.ndimage import gaussian_filter

# ── Colour palette ────────────────────────────────────────────────────────────
C_REAL = "#1D9E75"  # teal   — real data (always first)
C_FAKE = "#7F77DD"  # purple — primary generator / model
C_SYNTH = "#D97B45"  # orange — synthetic / additional generators

_FALLBACK_COLORS = [C_FAKE, C_SYNTH, "#E8593C", "#4A7FAA", "#919191"]

# Physiological reference bands
PHYSIO_FIX_MIN_MS = 30.0
PHYSIO_FIX_MAX_MS = 800.0
PHYSIO_SAC_MIN_DEG = 0.0
PHYSIO_SAC_MAX_DEG = 20.0


# ── Low-level helpers ─────────────────────────────────────────────────────────


def _condition_color(name: str, non_real_index: int) -> str:
    if name == "real":
        return C_REAL
    return _FALLBACK_COLORS[non_real_index % len(_FALLBACK_COLORS)]


def _get_array(df, col: str) -> np.ndarray:
    """Extract a finite-only float column from a polars DataFrame (or None)."""
    try:
        if df is None or len(df) == 0 or col not in df.columns:
            return np.array([])
        arr = df[col].to_numpy().astype(float)
        return arr[np.isfinite(arr)]
    except Exception:
        return np.array([])


def _sorted_conditions(conditions_data: Dict) -> List[str]:
    """Return condition names with 'real' first, rest alphabetically."""
    others = sorted(k for k in conditions_data if k != "real")
    return (["real"] if "real" in conditions_data else []) + others


def _color_map(conditions_data: Dict) -> Dict[str, str]:
    order = _sorted_conditions(conditions_data)
    idx = 0
    cmap: Dict[str, str] = {}
    for name in order:
        cmap[name] = _condition_color(name, idx)
        if name != "real":
            idx += 1
    return cmap


def _legend_handles(colors: Dict[str, str]) -> List:
    return [mpatches.Patch(facecolor=c, label=n) for n, c in colors.items()]


def _safe_fig_save(fig, out_path: Path, dpi: int = 150) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[plot] → {out_path}")


def _ks_annotation(
    ax, ref: np.ndarray, other: np.ndarray, color: str, y_frac: float
) -> None:
    ref = ref[np.isfinite(ref)]
    other = other[np.isfinite(other)]
    if len(ref) < 3 or len(other) < 3:
        return
    ks, p = stats.ks_2samp(ref, other)
    ax.annotate(
        f"KS={ks:.3f} p={p:.3f}",
        xy=(0.98, y_frac),
        xycoords="axes fraction",
        fontsize=7,
        ha="right",
        va="top",
        color=color,
    )


# ── Conditions-data builder ───────────────────────────────────────────────────


def build_conditions_data(
    plot_cache: Dict,
    stimuli: Optional[Sequence[str]] = None,
) -> Dict[str, Dict]:
    """
    Aggregate fix_df / sac_df / seqs across the requested stimuli for every
    condition present in plot_cache.

    Returns
    -------
    {
        condition_name: {
            "fix_df": pl.DataFrame,
            "sac_df": pl.DataFrame,
            "seqs":   np.ndarray (N, T, 2),
        },
        ...
    }
    """
    if stimuli is None:
        stimuli = list(plot_cache.keys())

    accum: Dict[str, Dict[str, list]] = {}

    for stim in stimuli:
        cache = plot_cache.get(stim, {})
        for cname, cdata in cache.items():
            if cname == "img_path" or not isinstance(cdata, dict):
                continue
            if cname not in accum:
                accum[cname] = {"fix_dfs": [], "sac_dfs": [], "seqs": []}
            fix_df = cdata.get("fix_df")
            sac_df = cdata.get("sac_df")
            seqs = cdata.get("seqs")
            if fix_df is not None and len(fix_df) > 0:
                accum[cname]["fix_dfs"].append(fix_df)
            if sac_df is not None and len(sac_df) > 0:
                accum[cname]["sac_dfs"].append(sac_df)
            if seqs is not None and len(seqs) > 0:
                accum[cname]["seqs"].append(seqs)

    result: Dict[str, Dict] = {}
    for cname, data in accum.items():
        result[cname] = {
            "fix_df": pl.concat(data["fix_dfs"]) if data["fix_dfs"] else pl.DataFrame(),
            "sac_df": pl.concat(data["sac_dfs"]) if data["sac_dfs"] else pl.DataFrame(),
            "seqs": np.concatenate(data["seqs"])
            if data["seqs"]
            else np.zeros((0, 1, 2)),
        }
    return result


# ── Individual metric figures ─────────────────────────────────────────────────


def plot_fixation_duration(
    conditions_data: Dict,
    out_path: "str | Path",
    sr: float = 1000.0,
    title: str = "Fixation duration",
) -> None:
    """
    KDE overlay of fixation duration for all conditions.

    Parameters
    ----------
    conditions_data : output of build_conditions_data()
    out_path        : save path
    sr              : sampling rate in Hz (converts samples → ms)
    title           : figure title
    """
    colors = _color_map(conditions_data)
    order = _sorted_conditions(conditions_data)

    fig, ax = plt.subplots(figsize=(7, 4), facecolor="white")

    all_ms: List[np.ndarray] = []
    for cname in order:
        dur = _get_array(conditions_data[cname].get("fix_df"), "duration")
        dur_ms = dur * 1000.0 / sr
        all_ms.append(dur_ms)

    valid = [d for d in all_ms if len(d) >= 3]
    if not valid:
        ax.text(
            0.5,
            0.5,
            "insufficient data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            color="#888",
        )
        _safe_fig_save(fig, Path(out_path))
        return

    xmin = np.percentile(np.concatenate(valid), 1)
    xmax = np.percentile(np.concatenate(valid), 99)
    xs = np.linspace(xmin, xmax, 400)

    ax.axvspan(
        PHYSIO_FIX_MIN_MS,
        PHYSIO_FIX_MAX_MS,
        alpha=0.07,
        color="#888888",
    )  # label="physiological range")

    real_ms = all_ms[order.index("real")] if "real" in order else np.array([])
    ks_row = 0
    for cname, dur_ms in zip(order, all_ms):
        if len(dur_ms) < 3:
            continue
        kde = stats.gaussian_kde(dur_ms, bw_method=0.3)(xs)
        ax.fill_between(xs, kde, alpha=0.18, color=colors[cname])
        ax.plot(
            xs, kde, color=colors[cname], lw=1.8, label=f"{cname}  (n={len(dur_ms):,})"
        )
        if cname != "real" and len(real_ms) >= 3:
            _ks_annotation(ax, real_ms, dur_ms, colors[cname], 0.97 - ks_row * 0.10)
            ks_row += 1

    ax.set_xlabel("Fixation duration (ms)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.tick_params(labelsize=8)
    ax.set_yticks([])
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    _safe_fig_save(fig, Path(out_path))


def plot_saccade_amplitude(
    conditions_data: Dict,
    out_path: "str | Path",
    title: str = "Saccade amplitude",
) -> None:
    """KDE overlay of saccade amplitude for all conditions."""
    colors = _color_map(conditions_data)
    order = _sorted_conditions(conditions_data)

    fig, ax = plt.subplots(figsize=(7, 4), facecolor="white")

    all_amp: List[np.ndarray] = []
    for cname in order:
        all_amp.append(
            _get_array(conditions_data[cname].get("sac_df"), "amplitude_deg")
        )

    valid = [d for d in all_amp if len(d) >= 3]
    if not valid:
        ax.text(
            0.5,
            0.5,
            "insufficient data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            color="#888",
        )
        _safe_fig_save(fig, Path(out_path))
        return

    xmin = np.percentile(np.concatenate(valid), 1)
    xmax = np.percentile(np.concatenate(valid), 99)
    xs = np.linspace(xmin, xmax, 400)

    ax.axvspan(
        PHYSIO_SAC_MIN_DEG, PHYSIO_SAC_MAX_DEG, alpha=0.07, color="#888888"
    )  # , label="physiological range")

    real_amp = all_amp[order.index("real")] if "real" in order else np.array([])
    ks_row = 0
    for cname, amp in zip(order, all_amp):
        if len(amp) < 3:
            continue
        kde = stats.gaussian_kde(amp, bw_method=0.3)(xs)
        ax.fill_between(xs, kde, alpha=0.18, color=colors[cname])
        ax.plot(
            xs, kde, color=colors[cname], lw=1.8, label=f"{cname}  (n={len(amp):,})"
        )
        if cname != "real" and len(real_amp) >= 3:
            _ks_annotation(ax, real_amp, amp, colors[cname], 0.97 - ks_row * 0.10)
            ks_row += 1

    ax.set_xlabel("Saccade amplitude (deg)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.tick_params(labelsize=8)
    ax.set_yticks([])
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    _safe_fig_save(fig, Path(out_path))


def plot_main_sequence(
    conditions_data: Dict,
    out_path: "str | Path",
    title: str = "Main sequence",
) -> None:
    """Amplitude vs peak-velocity scatter with OLS line per condition."""
    colors = _color_map(conditions_data)
    order = _sorted_conditions(conditions_data)

    fig, ax = plt.subplots(figsize=(7, 5), facecolor="white")

    any_data = False
    for cname in order:
        sac_df = conditions_data[cname].get("sac_df")
        amp = _get_array(sac_df, "amplitude_deg")
        pv = _get_array(sac_df, "peak_vel_deg_s")
        n = min(len(amp), len(pv))
        if n < 5:
            continue
        amp, pv = amp[:n], pv[:n]
        mask = (amp > 0.1) & (pv > 1.0)
        if mask.sum() < 5:
            continue
        any_data = True
        ax.scatter(
            amp[mask], pv[mask], s=5, alpha=0.3, color=colors[cname], linewidths=0
        )
        slope, intercept, r, *_ = stats.linregress(amp[mask], pv[mask])
        xs_fit = np.linspace(amp[mask].min(), amp[mask].max(), 200)
        ax.plot(
            xs_fit,
            slope * xs_fit + intercept,
            color=colors[cname],
            lw=2.0,
            label=f"{cname}  r={r:.3f}  (n={mask.sum():,})",
        )

    if not any_data:
        ax.text(
            0.5,
            0.5,
            "insufficient data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            color="#888",
        )

    ax.set_xlabel("Amplitude (deg)", fontsize=10)
    ax.set_ylabel("Peak velocity (deg/s)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.tick_params(labelsize=8)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    _safe_fig_save(fig, Path(out_path))


def plot_saccade_direction(
    conditions_data: Dict,
    out_path: "str | Path",
    n_bins: int = 16,
    title: str = "Saccade direction distribution",
) -> None:
    """
    Polar rose chart of saccade directions — one subplot per condition.

    Each wedge's radius encodes the proportion of saccades in that angular bin.
    Degree labels follow the screen convention: 0° = right, 90° = up,
    180° = left, 270° = down.
    """
    colors = _color_map(conditions_data)
    order = _sorted_conditions(conditions_data)

    # Collect data first so we can skip conditions with too few saccades
    cond_data: List[tuple] = []
    for cname in order:
        ang = _get_array(conditions_data[cname].get("sac_df"), "angle_rad")
        if len(ang) < 3:
            continue
        cond_data.append((cname, ang))

    n_cond = len(cond_data)
    if n_cond == 0:
        fig, ax = plt.subplots(figsize=(5, 5), facecolor="white")
        ax.text(
            0.5,
            0.5,
            "insufficient data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            color="#888",
        )
        fig.suptitle(title, fontsize=11, fontweight="bold")
        _safe_fig_save(fig, Path(out_path))
        return

    # Layout: up to 3 columns
    n_cols = min(n_cond, 3)
    n_rows = (n_cond + n_cols - 1) // n_cols
    fig_w = 4.5 * n_cols
    fig_h = 4.5 * n_rows + 0.5  # room for suptitle

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)

    bin_w = 2 * np.pi / n_bins
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    # Bin centres in matplotlib polar convention (0 = right, CCW positive)
    centres = 0.5 * (edges[:-1] + edges[1:])

    # Degree labels at cardinal + inter-cardinal angles (screen convention)
    label_angles_deg = np.arange(0, 360, 45)
    label_angles_rad = np.deg2rad(label_angles_deg)

    for idx, (cname, ang) in enumerate(cond_data):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection="polar")

        h, _ = np.histogram(ang, bins=edges)
        h_n = h / h.sum() if h.sum() > 0 else h.astype(float)

        bars = ax.bar(
            centres,
            h_n,
            width=bin_w * 0.95,
            bottom=0.0,
            color=colors[cname],
            alpha=0.85,
            linewidth=0.4,
            edgecolor="white",
        )

        # Polar axes cosmetics
        ax.set_theta_zero_location("E")  # 0° to the right
        ax.set_theta_direction(1)  # counter-clockwise (standard math)
        ax.set_xticks(label_angles_rad)
        ax.set_xticklabels([f"{int(d)}°" for d in label_angles_deg], fontsize=8)
        ax.yaxis.set_visible(False)  # hide radial tick labels
        ax.set_facecolor("white")
        ax.spines["polar"].set_color("#cccccc")
        ax.spines["polar"].set_linewidth(0.8)
        ax.grid(color="#dddddd", linewidth=0.5)

        ax.set_title(
            f"{cname}  (n={len(ang):,})",
            fontsize=9,
            pad=12,
            color=colors[cname],
            fontweight="bold",
        )

    fig.tight_layout()
    _safe_fig_save(fig, Path(out_path))


def plot_fixation_density(
    conditions_data: Dict,
    out_path: "str | Path",
    grid: int = 48,
    sigma: float = 1.5,
    title: str = "Fixation density",
) -> None:
    """
    One Gaussian-smoothed 2-D fixation-density heatmap per condition,
    shown side by side (used for per-stimulus views).
    """
    colors = _color_map(conditions_data)
    order = _sorted_conditions(conditions_data)
    n_cond = len(order)

    fig, axes = plt.subplots(
        1, n_cond, figsize=(4 * n_cond, 4), facecolor="white", squeeze=False
    )

    for ax, cname in zip(axes[0], order):
        cx = _get_array(conditions_data[cname].get("fix_df"), "cx_deg")
        cy = _get_array(conditions_data[cname].get("fix_df"), "cy_deg")
        valid = np.isfinite(cx) & np.isfinite(cy)
        cx, cy = cx[valid], cy[valid]

        if len(cx) >= 3:
            xr = cx.max() - cx.min() or 1.0
            yr = cy.max() - cy.min() or 1.0
            xi = ((cx - cx.min()) / xr * (grid - 1)).astype(int).clip(0, grid - 1)
            yi = ((cy - cy.min()) / yr * (grid - 1)).astype(int).clip(0, grid - 1)
            h = np.zeros((grid, grid))
            for x, y in zip(xi, yi):
                h[y, x] += 1
            h = gaussian_filter(h.astype(float) + 1e-8, sigma=sigma)
            ax.imshow(
                h,
                origin="lower",
                aspect="auto",
                cmap="YlOrRd",
                interpolation="bilinear",
            )
        else:
            ax.text(
                0.5,
                0.5,
                "no fixations",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=9,
                color="#888",
            )
            ax.set_facecolor("#F5F5F3")

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"{cname}  (n={len(cx):,} fixations)",
            fontsize=9,
            pad=4,
            color=colors[cname],
            fontweight="bold",
        )

    fig.suptitle(title, fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    _safe_fig_save(fig, Path(out_path))


def plot_fixation_density_grid(
    plot_cache: Dict,
    out_dir: "str | Path",
    grid: int = 32,
    sigma: float = 1.5,
    grid_cols: int = 5,
    max_stimuli: int = 40,
) -> None:
    """
    Tiled fixation-density overview — one panel per stimulus, one figure per
    condition.  Mirrors the style of ``_plot_per_stimulus_density`` from
    stats_plots.py: YlOrRd heatmap, stimulus name + recording/fixation counts
    as panel title, stimuli ranked by descending fixation count.

    Saves one PNG per condition as ``<out_dir>/fixation_density_<cname>.png``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect all condition names (skip "img_path")
    all_conditions: set = set()
    for stim_data in plot_cache.values():
        all_conditions.update(k for k in stim_data if k != "img_path")

    colors = {}
    non_real = sorted(c for c in all_conditions if c != "real")
    for i, c in enumerate(["real"] + non_real):
        colors[c] = _condition_color(c, i)

    # Rank stimuli by total fixation count in the "real" condition
    def _n_fix(stim_data: dict, cname: str) -> int:
        fd = stim_data.get(cname, {}).get("fix_df")
        return len(fd) if fd is not None else 0

    ranked_stimuli = sorted(
        plot_cache.keys(),
        key=lambda s: -_n_fix(plot_cache[s], "real"),
    )[:max_stimuli]

    n = len(ranked_stimuli)
    if n == 0:
        return

    nc = min(grid_cols, n)
    nr = (n + nc - 1) // nc

    def _build_heatmap(fix_df, seqs) -> np.ndarray:
        """Return a (grid × grid) smoothed density array."""
        h = np.zeros((grid, grid))
        cx = _get_array(fix_df, "cx_deg")
        cy = _get_array(fix_df, "cy_deg")
        valid = np.isfinite(cx) & np.isfinite(cy)
        cx, cy = cx[valid], cy[valid]

        if len(cx) >= 3:
            xr = cx.max() - cx.min() or 1.0
            yr = cy.max() - cy.min() or 1.0
            xi = ((cx - cx.min()) / xr * (grid - 1)).astype(int).clip(0, grid - 1)
            yi = ((cy - cy.min()) / yr * (grid - 1)).astype(int).clip(0, grid - 1)
        elif seqs is not None and len(seqs) > 0:
            # Fall back to raw normalised sequence points
            pts = np.clip(seqs.reshape(-1, 2), 0, 1 - 1e-9)
            xi = (pts[:, 0] * (grid - 1)).astype(int).clip(0, grid - 1)
            yi = (pts[:, 1] * (grid - 1)).astype(int).clip(0, grid - 1)
        else:
            return None

        for x, y in zip(xi, yi):
            h[y, x] += 1
        return gaussian_filter(h.astype(float) + 1e-8, sigma=sigma)

    for cname in ["real"] + non_real:
        if cname not in all_conditions:
            continue

        fig, axes = plt.subplots(
            nr,
            nc,
            figsize=(nc * 2.8, nr * 2.6),
            constrained_layout=True,
            facecolor="white",
        )
        # Normalise axes to 2-D array
        if nr == 1 and nc == 1:
            axes = np.array([[axes]])
        elif nr == 1 or nc == 1:
            axes = np.array(axes).reshape(nr, nc)

        for idx, stim in enumerate(ranked_stimuli):
            ax = axes[idx // nc][idx % nc]
            sd = plot_cache[stim]
            cnd = sd.get(cname, {})
            fix_df = cnd.get("fix_df")
            seqs = cnd.get("seqs")

            hmap = _build_heatmap(fix_df, seqs)
            if hmap is not None:
                ax.imshow(
                    hmap / hmap.sum(),
                    cmap="YlOrRd",
                    origin="lower",
                    aspect="auto",
                    interpolation="bilinear",
                )
            else:
                ax.set_facecolor("#F5F5F3")
                ax.text(
                    0.5,
                    0.5,
                    "no data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=7,
                    color="#aaa",
                )

            n_fix = len(fix_df) if fix_df is not None else 0
            # Count "recordings" as number of sequences (each is one trial)
            n_rec = len(seqs) if seqs is not None else 0
            short_stim = stim[:20]
            ax.set_title(
                f"{short_stim}\n{n_rec} rec · {n_fix} fix",
                fontsize=6.5,
                pad=3,
            )
            ax.set_xticks([])
            ax.set_yticks([])

        # Hide unused panels
        for idx in range(len(ranked_stimuli), nr * nc):
            axes[idx // nc][idx % nc].axis("off")

        fig.suptitle(
            f"Per-stimulus fixation density maps — {cname}",
            fontsize=9,
            y=1.01,
        )
        safe_cname = cname.replace("/", "_").replace(" ", "_")
        _safe_fig_save(fig, out_dir / f"fixation_density_{safe_cname}.png")


# ── Aggregate and per-stimulus wrappers ──────────────────────────────────────

_METRIC_FNAMES = [
    "fixation_duration.png",
    "saccade_amplitude.png",
    "main_sequence.png",
    "saccade_direction.png",
    "fixation_density.png",
]


def _plot_all_metrics(
    conditions_data: Dict,
    out_dir: Path,
    sr: float,
    title_suffix: str = "",
    include_density: bool = True,
) -> None:
    """Call the individual metric plot functions into out_dir."""
    sfx = f" — {title_suffix}" if title_suffix else ""
    plot_fixation_duration(
        conditions_data,
        out_dir / "fixation_duration.png",
        sr=sr,
        title=f"Fixation duration{sfx}",
    )
    plot_saccade_amplitude(
        conditions_data,
        out_dir / "saccade_amplitude.png",
        title=f"Saccade amplitude{sfx}",
    )
    plot_main_sequence(
        conditions_data, out_dir / "main_sequence.png", title=f"Main sequence{sfx}"
    )
    plot_saccade_direction(
        conditions_data,
        out_dir / "saccade_direction.png",
        title=f"Saccade direction{sfx}",
    )
    if include_density:
        plot_fixation_density(
            conditions_data,
            out_dir / "fixation_density.png",
            title=f"Fixation density{sfx}",
        )


def plot_aggregate_metrics(
    plot_cache: Dict,
    out_dir: "str | Path",
    sr: float = 1000.0,
) -> None:
    """
    Pool all stimuli and generate metric figures saved in ``out_dir``:
        fixation_duration.png  saccade_amplitude.png  main_sequence.png
        saccade_direction.png
        fixation_density_<condition>.png  (one tiled grid per condition)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    conditions_data = build_conditions_data(plot_cache)
    # All metrics except density (density uses the tiled-grid form below)
    _plot_all_metrics(
        conditions_data, out_dir, sr, title_suffix="all stimuli", include_density=False
    )
    # Tiled per-stimulus density grid, one figure per condition
    plot_fixation_density_grid(plot_cache, out_dir)


def plot_per_stimulus_metrics(
    plot_cache: Dict,
    out_dir: "str | Path",
    sr: float = 1000.0,
    stimuli: Optional[Sequence[str]] = None,
) -> None:
    """
    Generate five individual metric figures **per stimulus** saved under
    ``out_dir/<stimulus_name>/``.

    Parameters
    ----------
    plot_cache : condition-keyed cache (see module docstring)
    out_dir    : parent output directory; one sub-directory per stimulus
    sr         : sampling rate in Hz
    stimuli    : subset of stimulus names; default all keys in plot_cache
    """
    out_dir = Path(out_dir)
    if stimuli is None:
        stimuli = list(plot_cache.keys())

    for stim in stimuli:
        if stim not in plot_cache:
            continue
        conditions_data = build_conditions_data(plot_cache, stimuli=[stim])
        safe = stim.replace("/", "_").replace(" ", "_")
        stim_dir = out_dir / safe
        stim_dir.mkdir(parents=True, exist_ok=True)
        _plot_all_metrics(conditions_data, stim_dir, sr, title_suffix=stim)


# ── Best / worst comparison ───────────────────────────────────────────────────


def plot_best_worst_comparison(
    all_results: Dict,
    plot_cache: Dict,
    out_path: "str | Path",
    score_metric: tuple = ("fixation_duration", "ks_stat"),
    n_scanpaths: int = 8,
    density_grid: int = 32,
    density_sigma: float = 1.5,
    primary_condition: Optional[str] = None,
) -> None:
    """
    2-row × 5-column figure comparing the best and worst matching stimuli.

    Accepts the condition-keyed plot_cache format.
    ``primary_condition`` names the generated condition to compare against real;
    auto-detected if None.
    """
    section, key = score_metric

    def _score(m):
        val = m.get(section, {}).get(key, float("nan"))
        if np.isnan(val):
            return float("inf")
        if section == "classifier_auc" and key == "auc":
            return abs(val - 0.5)
        return 1.0 - val

    ranked = sorted(
        [(s, _score(m)) for s, m in all_results.items() if s in plot_cache],
        key=lambda x: x[1],
    )
    if len(ranked) < 2:
        print("[plot] need ≥ 2 stimuli — skipping best/worst comparison")
        return

    best_name, best_score = ranked[0]
    worst_name, worst_score = ranked[-1]

    # Detect primary condition
    sample = plot_cache[best_name]
    if primary_condition is None:
        non_real = [k for k in sample if k not in ("img_path", "real")]
        primary_condition = non_real[0] if non_real else "generated"

    def _flat(cache: dict) -> dict:
        real = cache.get("real", {})
        fake = cache.get(primary_condition, {})
        return {
            "real_seqs": real.get("seqs", np.zeros((0, 1, 2))),
            "fake_seqs": fake.get("seqs", np.zeros((0, 1, 2))),
            "real_fix_df": real.get("fix_df"),
            "fake_fix_df": fake.get("fix_df"),
            "real_sac_df": real.get("sac_df"),
            "fake_sac_df": fake.get("sac_df"),
            "img_path": cache.get("img_path"),
        }

    flat_best = _flat(plot_cache[best_name])
    flat_worst = _flat(plot_cache[worst_name])

    ALPHA_T = 0.35

    def _dm(seqs, fix_df, g, sigma):
        h = np.zeros((g, g))
        cx = _get_array(fix_df, "cx_deg")
        cy = _get_array(fix_df, "cy_deg")
        if len(cx) >= 3:
            xr = cx.max() - cx.min() or 1.0
            yr = cy.max() - cy.min() or 1.0
            xi = ((cx - cx.min()) / xr * (g - 1)).astype(int).clip(0, g - 1)
            yi = ((cy - cy.min()) / yr * (g - 1)).astype(int).clip(0, g - 1)
        else:
            pts = np.clip(seqs.reshape(-1, 2), 0, 1 - 1e-9)
            xi = (pts[:, 0] * (g - 1)).astype(int).clip(0, g - 1)
            yi = (pts[:, 1] * (g - 1)).astype(int).clip(0, g - 1)
        for x, y in zip(xi, yi):
            h[y, x] += 1
        return gaussian_filter(h.astype(float) + 1e-8, sigma=sigma)

    def _kde_panel(ax, r_data, f_data, xlabel, unit=""):
        r_data = r_data[np.isfinite(r_data)]
        f_data = f_data[np.isfinite(f_data)]
        if len(r_data) < 3 or len(f_data) < 3:
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
        xmin = min(np.percentile(r_data, 1), np.percentile(f_data, 1))
        xmax = max(np.percentile(r_data, 99), np.percentile(f_data, 99))
        xs = np.linspace(xmin, xmax, 300)
        kde_r = stats.gaussian_kde(r_data, bw_method=0.3)(xs)
        kde_f = stats.gaussian_kde(f_data, bw_method=0.3)(xs)
        ax.fill_between(xs, kde_r, alpha=0.25, color=C_REAL)
        ax.fill_between(xs, kde_f, alpha=0.25, color=C_FAKE)
        ax.plot(xs, kde_r, color=C_REAL, lw=1.5, label="real")
        ax.plot(xs, kde_f, color=C_FAKE, lw=1.5, label=primary_condition)
        ks, p = stats.ks_2samp(r_data, f_data)
        ax.set_title(f"KS={ks:.3f}  p={p:.3f}", fontsize=8, pad=3)
        ax.set_xlabel(xlabel + unit, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_yticks([])
        for sp in ["top", "right", "left"]:
            ax.spines[sp].set_visible(False)

    def _draw_row(axes_row, flat, stim_name, metrics, row_label, score):
        ax_img, ax_scan, ax_fix, ax_sac, ax_dens = axes_row

        # col 0: image
        try:
            img = Image.open(flat["img_path"]).convert("RGB")
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
        auc = metrics.get("classifier_auc", {}).get("auc", float("nan"))
        badge = "#1D9E75" if score < 0.15 else "#E8593C"
        ax_img.text(
            0.97,
            0.03,
            f"AUC {auc:.3f}",
            transform=ax_img.transAxes,
            fontsize=7,
            ha="right",
            va="bottom",
            color="white",
            bbox=dict(boxstyle="round,pad=0.3", fc=badge, ec="none", alpha=0.85),
        )

        # col 1: scanpaths
        # norm_y = pixel_y / screen_height  →  0 at top of screen, 1 at bottom.
        # matplotlib y-axis has 0 at *bottom*, so we must flip: plot_y = 1 - norm_y.
        for seq in flat["real_seqs"][:n_scanpaths]:
            ax_scan.plot(seq[:, 0], 1 - seq[:, 1], lw=0.7, alpha=ALPHA_T, color=C_REAL)
        for seq in flat["fake_seqs"][:n_scanpaths]:
            ax_scan.plot(seq[:, 0], 1 - seq[:, 1], lw=0.7, alpha=ALPHA_T, color=C_FAKE)
        for fdf, color in [
            (flat["real_fix_df"], C_REAL),
            (flat["fake_fix_df"], C_FAKE),
        ]:
            cx = _get_array(fdf, "cx_deg")
            cy = _get_array(fdf, "cy_deg")
            dur = _get_array(fdf, "duration")
            n = min(len(cx), len(cy), len(dur))
            if n > 0:
                rng = lambda a: np.nanmax(a) - np.nanmin(a) + 1e-9
                cx_n = (cx[:n] - np.nanmin(cx[:n])) / rng(cx[:n])
                # cy_deg in visual-angle space: smaller values = higher on screen
                # → same inversion needed so fixation dots align with scanpath lines
                cy_n = 1 - (cy[:n] - np.nanmin(cy[:n])) / rng(cy[:n])
                sizes = np.clip(dur[:n] / dur[:n].max() * 80, 5, 80)
                ax_scan.scatter(
                    cx_n, cy_n, s=sizes, color=color, alpha=0.4, linewidths=0, zorder=3
                )
        ax_scan.set_xlim(0, 1)
        ax_scan.set_ylim(0, 1)
        ax_scan.set_aspect("equal")
        ax_scan.set_xticks([])
        ax_scan.set_yticks([])
        ax_scan.set_title("Scanpaths", fontsize=8, pad=3)
        for sp in ax_scan.spines.values():
            sp.set_linewidth(0.4)
            sp.set_color("#D3D1C7")

        # col 2 & 3: KDE
        _kde_panel(
            ax_fix,
            _get_array(flat["real_fix_df"], "duration"),
            _get_array(flat["fake_fix_df"], "duration"),
            "Fix. duration",
            " (samples)",
        )
        _kde_panel(
            ax_sac,
            _get_array(flat["real_sac_df"], "amplitude_deg"),
            _get_array(flat["fake_sac_df"], "amplitude_deg"),
            "Sac. amplitude",
            " (deg)",
        )

        # col 4: density difference
        d_r = _dm(flat["real_seqs"], flat["real_fix_df"], density_grid, density_sigma)
        d_f = _dm(flat["fake_seqs"], flat["fake_fix_df"], density_grid, density_sigma)
        diff = d_r / d_r.sum() - d_f / d_f.sum()
        vmax = np.abs(diff).max()
        im = ax_dens.imshow(
            diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower", aspect="auto"
        )
        ax_dens.set_xticks([])
        ax_dens.set_yticks([])
        ax_dens.set_title("Density: real − generated", fontsize=8, pad=3)
        cbar = plt.colorbar(im, ax=ax_dens, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=6)
        kl = metrics.get("fixation_density_map", {}).get("kl_divergence", float("nan"))
        ax_dens.set_xlabel(f"KL div = {kl:.3f}", fontsize=7, labelpad=6)

    fig = plt.figure(figsize=(18, 11), facecolor="white")
    gs = gridspec.GridSpec(
        2,
        5,
        figure=fig,
        hspace=0.65,
        wspace=0.28,
        left=0.03,
        right=0.97,
        top=0.88,
        bottom=0.08,
    )
    axes_best = [fig.add_subplot(gs[0, c]) for c in range(5)]
    axes_worst = [fig.add_subplot(gs[1, c]) for c in range(5)]

    _draw_row(
        axes_best,
        flat_best,
        best_name,
        all_results[best_name],
        "Best match",
        best_score,
    )
    _draw_row(
        axes_worst,
        flat_worst,
        worst_name,
        all_results[worst_name],
        "Worst match",
        worst_score,
    )

    col_headers = [
        "Stimulus",
        "Scanpaths\n(real / generated)",
        "Fixation duration",
        "Saccade amplitude",
        "Fixation density\ndifference",
    ]
    for ax, hdr in zip(axes_best, col_headers):
        ax.annotate(
            hdr,
            xy=(0.5, 1.0),
            xycoords="axes fraction",
            xytext=(0, 36),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight=500,
            color="#444441",
        )

    fig.legend(
        handles=[
            mpatches.Patch(color=C_REAL, alpha=0.7, label="Real data"),
            mpatches.Patch(
                color=C_FAKE, alpha=0.7, label=f"Generated ({primary_condition})"
            ),
        ],
        loc="upper right",
        fontsize=9,
        frameon=True,
        framealpha=0.9,
        edgecolor="#D3D1C7",
        bbox_to_anchor=(0.97, 0.97),
    )

    section_label = section.replace("_", " ")
    fig.suptitle(
        f"Gaze evaluation — best vs worst stimulus\n"
        f"Ranked by {section_label} {key}  "
        f"(best: {best_score:.3f}  worst: {worst_score:.3f})",
        fontsize=10,
        y=0.97,
        va="top",
        color="#2C2C2A",
    )

    _safe_fig_save(fig, Path(out_path))


# ── Scanpath overview ─────────────────────────────────────────────────────────


def plot_scanpath_overview(
    plot_cache: Dict,
    out_path: "str | Path",
    stimuli: Optional[Sequence[str]] = None,
    n_cols: int = 4,
    n_scanpaths: int = 6,
    show_image: bool = True,
    title: str = "Scanpath overview — real vs generated",
    condition_pair: tuple = ("real", None),
) -> None:
    """
    Tiled overview — one panel per stimulus with real and generated scanpaths
    overlaid on the stimulus image.  ``condition_pair`` selects which two
    conditions to draw; the second entry is auto-detected if None.
    """
    if stimuli is None:
        stimuli = list(plot_cache.keys())
    stimuli = [s for s in stimuli if s in plot_cache]
    if not stimuli:
        print("[plot] no stimuli in cache — skipping overview")
        return

    ref_cond, gen_cond = condition_pair
    if gen_cond is None:
        non_real = [k for k in plot_cache[stimuli[0]] if k not in ("img_path", "real")]
        gen_cond = non_real[0] if non_real else None

    n = len(stimuli)
    n_rows = math.ceil(n / n_cols)
    fig_w = n_cols * 3.2 + 0.4
    fig_h = n_rows * 2.6 + 0.9

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), facecolor="white")
    axes_flat: List = (
        np.array(axes).flatten().tolist() if n_rows > 1 or n_cols > 1 else [axes]
    )
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    for ax, stim_name in zip(axes_flat, stimuli):
        cache = plot_cache[stim_name]
        real_seqs = cache.get(ref_cond, {}).get("seqs", np.zeros((0, 1, 2)))
        fake_seqs = (
            cache.get(gen_cond, {}).get("seqs", np.zeros((0, 1, 2)))
            if gen_cond
            else np.zeros((0, 1, 2))
        )
        img_path = cache.get("img_path")

        if show_image and img_path is not None:
            try:
                img = Image.open(img_path).convert("RGB")
                ax.imshow(img, extent=[0, 1, 0, 1], aspect="auto", zorder=0, alpha=0.55)
            except Exception:
                pass

        ax.set_facecolor("#F5F5F3")
        # norm_y = pixel_y / screen_height  →  0 at top of screen, 1 at bottom.
        # matplotlib y=0 is at *bottom*, so flip: plot_y = 1 - norm_y.
        for seq in real_seqs[:n_scanpaths]:
            ax.plot(seq[:, 0], 1 - seq[:, 1], lw=0.8, alpha=0.4, color=C_REAL, zorder=2)
        for seq in fake_seqs[:n_scanpaths]:
            ax.plot(seq[:, 0], 1 - seq[:, 1], lw=0.8, alpha=0.4, color=C_FAKE, zorder=2)
        for seqs, color in [
            (real_seqs[:n_scanpaths], C_REAL),
            (fake_seqs[:n_scanpaths], C_FAKE),
        ]:
            for s in seqs:
                ax.scatter(
                    s[0, 0], 1 - s[0, 1], s=12, color=color, zorder=3, linewidths=0
                )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        short = stim_name if len(stim_name) <= 22 else stim_name[:20] + "…"
        ax.set_title(short, fontsize=7.5, pad=3, color="#333331")
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color("#C8C6BC")

    plt.subplots_adjust(
        left=0.02, right=0.98, top=0.90, bottom=0.04, hspace=0.38, wspace=0.12
    )

    gen_label = gen_cond or "generated"
    fig.legend(
        handles=[
            mpatches.Patch(
                color=C_REAL, alpha=0.75, label=f"Real  (up to {n_scanpaths})"
            ),
            mpatches.Patch(
                color=C_FAKE, alpha=0.75, label=f"{gen_label}  (up to {n_scanpaths})"
            ),
        ],
        loc="upper right",
        fontsize=8,
        frameon=True,
        framealpha=0.9,
        edgecolor="#D3D1C7",
        bbox_to_anchor=(0.99, 0.995),
    )

    fig.suptitle(
        title, fontsize=10, y=0.995, va="top", color="#2C2C2A", fontweight="bold"
    )

    _safe_fig_save(fig, Path(out_path))
