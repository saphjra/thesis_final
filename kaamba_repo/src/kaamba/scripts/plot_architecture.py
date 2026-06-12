"""
plot_architecture.py

Abstract architecture diagram of the Kaamba gaze predictor:
  Image Encoder (SigLIP / ViT / ResNet) + Mamba2 SSM backbone + GMM head.

Generates two figures (one per conditioning mode) as PNG + PDF.

Usage
─────
  python plot_architecture.py --out_dir plots/architecture
  python plot_architecture.py --out_dir plots/architecture --mode every_step
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np


# ── colour palette ────────────────────────────────────────────────────────────
C_IMG = "#7B72CF"  # purple  — image branch
C_GAZE = "#1D9E75"  # teal    — gaze branch
C_MERGE = "#4A7FAA"  # slate   — conditioning / merge
C_SSM = "#D97B45"  # orange  — Mamba2 layers
C_NORM = "#919191"  # grey    — LayerNorm
C_HEAD_GMM = "#8B5BAD"  # violet  — GMM head
C_ARROW = "#AAAAAA"
C_ANNO = "#444444"
FONT = "DejaVu Sans"


# ── low-level drawing helpers ─────────────────────────────────────────────────


def _box(
    ax,
    cx: float,
    cy: float,
    w: float,
    h: float,
    color: str,
    title: str,
    subtitle: str = "",
    title_fs: float = 9.0,
    sub_fs: float = 7.5,
    alpha: float = 1.0,
):
    """Draw a rounded rectangle with an optional two-line label."""
    pad = 0.012
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2),
        w,
        h,
        boxstyle=f"round,pad={pad}",
        facecolor=color,
        edgecolor="none",
        alpha=alpha,
        zorder=3,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(patch)

    dy = 0.013 if subtitle else 0.0
    ax.text(
        cx,
        cy + dy,
        title,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=title_fs,
        fontweight="bold",
        color="white",
        zorder=4,
        clip_on=False,
    )
    if subtitle:
        ax.text(
            cx,
            cy - dy,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=sub_fs,
            color="white",
            alpha=0.88,
            zorder=4,
            clip_on=False,
        )


def _arrow(
    ax,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: str = C_ARROW,
    lw: float = 1.5,
    rad: float = 0.0,
):
    """Draw an arrowhead from (x0,y0) to (x1,y1) in axes-fraction coords."""
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            mutation_scale=10,
            connectionstyle=f"arc3,rad={rad}",
        ),
        zorder=2,
    )


def _text(
    ax,
    x: float,
    y: float,
    txt: str,
    fs: float = 7.5,
    color: str = C_ANNO,
    ha: str = "center",
    va: str = "center",
    style: str = "normal",
    alpha: float = 1.0,
):
    ax.text(
        x,
        y,
        txt,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=fs,
        color=color,
        style=style,
        alpha=alpha,
        zorder=5,
        clip_on=False,
    )


def _brace_right(
    ax,
    x: float,
    y_bot: float,
    y_top: float,
    label: str,
    color: str = C_SSM,
    fs: float = 8.5,
):
    """Draw a right-side vertical brace with a label."""
    ax.annotate(
        "",
        xy=(x, y_bot),
        xytext=(x, y_top),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-", color="#CCCCCC", lw=1.2),
        zorder=1,
    )
    ax.plot(
        [x - 0.006, x + 0.006],
        [(y_top + y_bot) / 2, (y_top + y_bot) / 2],
        transform=ax.transAxes,
        color="#CCCCCC",
        lw=1.2,
        zorder=1,
    )
    _text(ax, x + 0.038, (y_top + y_bot) / 2, label, fs=fs, color=color, ha="left")


# ── main diagram ──────────────────────────────────────────────────────────────


def _draw_head_gmm(
    ax, Y_LN: float, BH_S: float, BH: float, XC: float, BW_CENTER: float
) -> None:
    """Draw the GMM (Mixture Density Network) output head below LayerNorm."""
    Y_HEAD = Y_LN - BH_S / 2 - 0.065
    _box(
        ax,
        XC,
        Y_HEAD,
        BW_CENTER,
        BH,
        C_HEAD_GMM,
        "GMM Head  (MDN)",
        "Linear → K × 6   (π · μ · log σ · ρ)",
    )
    _arrow(ax, XC, Y_LN - BH_S / 2, XC, Y_HEAD + BH / 2, color=C_HEAD_GMM)

    # Five output tensors spread across the width — keep within axes bounds
    outputs = [
        ("π", "(B, T, K)"),
        ("μ", "(B, T, K, 2)"),
        ("log σx", "(B, T, K)"),
        ("log σy", "(B, T, K)"),
        ("ρ", "(B, T, K)"),
    ]
    n = len(outputs)
    box_w = 0.12
    spread = 0.29  # half-span; keeps edge boxes (0.5±0.29 ± 0.06) within [0.09, 0.91]
    xs_out = np.linspace(XC - spread, XC + spread, n)
    Y_OUT = Y_HEAD - BH / 2 - 0.065
    for xi, (lbl, shape) in zip(xs_out, outputs):
        _box(
            ax,
            xi,
            Y_OUT,
            box_w,
            BH_S,
            C_HEAD_GMM,
            lbl,
            shape,
            title_fs=8.5,
            sub_fs=6.8,
            alpha=0.75,
        )
        _arrow(ax, XC, Y_HEAD - BH / 2, xi, Y_OUT + BH_S / 2, color=C_HEAD_GMM, lw=1.2)

    _text(
        ax,
        XC,
        Y_OUT - BH_S / 2 - 0.022,
        "K = number of mixture components  ·  raw (pre-activation) parameters",
        fs=7.0,
        color="#AAAAAA",
    )


def make_diagram(out_dir: Path, mode: str = "initial_state") -> None:
    """
    Draw one architecture figure for the given conditioning mode.

    Parameters
    ----------
    out_dir : Path   Directory where PNG + PDF are saved.
    mode    : str    ``"initial_state"`` or ``"every_step"``.
    """
    fig, ax = plt.subplots(figsize=(8, 11))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ── x-centres for the three columns ───────────────────────────────────────
    XL = 0.25  # image column
    XR = 0.75  # gaze column
    XC = 0.50  # backbone / centre

    BW_SIDE = 0.34  # side-column box width
    BW_CENTER = 0.44  # centre-column box width
    BH = 0.062  # normal box height
    BH_S = 0.050  # slim box height
    BH_SSM = 0.056  # SSM layer box height

    # ── ① INPUT labels ────────────────────────────────────────────────────────
    Y_IN = 0.95
    _text(ax, XL, Y_IN + 0.018, "Image", fs=10, color=C_IMG, style="normal")
    _text(ax, XL, Y_IN, "(B, 3, H, W)", fs=8, color="#999999")
    _text(ax, XR, Y_IN + 0.018, "Gaze sequence", fs=10, color=C_GAZE)
    _text(ax, XR, Y_IN, "(B, 2, T)", fs=8, color="#999999")

    # ── ② IMAGE ENCODER ───────────────────────────────────────────────────────
    Y_ENC = 0.860
    _box(
        ax,
        XL,
        Y_ENC,
        BW_SIDE,
        BH,
        C_IMG,
        "Image Encoder",
        "SigLIP · ViT · ResNet  [frozen]",
    )
    _arrow(ax, XL, Y_IN - 0.006, XL, Y_ENC + BH / 2, color=C_IMG)
    # In initial_state mode the State Projection box sits immediately below the
    # encoder, so we push the shape annotation to the left margin to avoid
    # collision.  In every_step mode there is room to place it below as usual.
    if mode == "initial_state":
        _text(
            ax,
            XL - BW_SIDE / 2 - 0.03,
            Y_ENC,
            "(B, E)",
            fs=7.5,
            color="#AAAAAA",
            ha="right",
        )
    else:
        _text(ax, XL, Y_ENC - BH / 2 - 0.018, "(B, E)", fs=7.5, color="#AAAAAA")

    # ── ③ GAZE PROJECTION ─────────────────────────────────────────────────────
    Y_GPROJ = 0.860  # same height as image encoder
    _box(
        ax,
        XR,
        Y_GPROJ,
        BW_SIDE,
        BH,
        C_GAZE,
        "Gaze Projection",
        "Linear → LayerNorm → GELU",
    )
    _arrow(ax, XR, Y_IN - 0.006, XR, Y_GPROJ + BH / 2, color=C_GAZE)
    _text(ax, XR, Y_GPROJ - BH / 2 - 0.018, "(B, T, d)", fs=7.5, color="#AAAAAA")

    # ── ④ CONDITIONING  (mode-specific) ───────────────────────────────────────
    # initial_state: sproj annotation sits at Y_SPROJ − BH_S/2 − 0.017 ≈ 0.753
    # We need cond-box top (Y_COND + BH_S/2) at least 0.025 below that → Y_COND ≤ 0.703
    Y_COND = 0.700

    if mode == "initial_state":
        # image path: extra state-projection before merge
        Y_SPROJ = 0.795  # raised to leave clear gap above Y_COND
        _box(
            ax,
            XL,
            Y_SPROJ,
            BW_SIDE,
            BH_S,
            C_IMG,
            "State Projection",
            "Linear → LayerNorm → GELU",
        )
        _arrow(ax, XL, Y_ENC - BH / 2, XL, Y_SPROJ + BH_S / 2, color=C_IMG)
        _text(
            ax,
            XL,
            Y_SPROJ - BH_S / 2 - 0.017,
            "(B, 1, d)  → prepend",
            fs=7.5,
            color="#AAAAAA",
        )

        # conditioning box — top is Y_COND + BH_S/2 = 0.745; sproj bottom = 0.770 → gap 0.025
        _box(
            ax,
            XC,
            Y_COND,
            BW_CENTER,
            BH_S,
            C_MERGE,
            "Condition on Image",
            "prepend image token  →  (B, T+1, d)",
        )
        _arrow(
            ax,
            XL + BW_SIDE / 2,
            Y_SPROJ - BH_S / 2,
            XC - BW_CENTER / 2 + 0.03,
            Y_COND + BH_S / 2,
            color=C_IMG,
            rad=-0.18,
        )
        _arrow(
            ax,
            XR - BW_SIDE / 2,
            Y_GPROJ - BH / 2,
            XC + BW_CENTER / 2 - 0.03,
            Y_COND + BH_S / 2,
            color=C_GAZE,
            rad=0.18,
        )

    else:  # every_step
        Y_EXPAND = 0.790
        _text(
            ax,
            XL,
            Y_EXPAND,
            "expand  (B, E)  →  (B, T, E)",
            fs=8.0,
            color=C_IMG,
            style="italic",
        )
        _text(
            ax,
            XL,
            Y_EXPAND - 0.025,
            "broadcast over T timesteps",
            fs=7.5,
            color="#AAAAAA",
        )
        _arrow(ax, XL, Y_ENC - BH / 2, XL, Y_EXPAND + 0.018, color=C_IMG)

        _box(
            ax,
            XC,
            Y_COND,
            BW_CENTER,
            BH_S,
            C_MERGE,
            "Condition on Image",
            "concat image at every step  →  (B, T, 2+E)",
        )
        _arrow(
            ax,
            XL + 0.02,
            Y_EXPAND - 0.035,
            XC - BW_CENTER / 2 + 0.03,
            Y_COND + BH_S / 2,
            color=C_IMG,
            rad=-0.15,
        )
        _arrow(
            ax,
            XR - BW_SIDE / 2,
            Y_GPROJ - BH / 2,
            XC + BW_CENTER / 2 - 0.03,
            Y_COND + BH_S / 2,
            color=C_GAZE,
            rad=0.18,
        )

    # ── ⑤ MAMBA2 BACKBONE ────────────────────────────────────────────────────
    N_VISIBLE = 2  # layers drawn explicitly
    Y_SSM_TOP = 0.630
    DY_SSM = 0.075  # vertical gap between SSM boxes

    _arrow(ax, XC, Y_COND - BH_S / 2, XC, Y_SSM_TOP + BH_SSM / 2, color=C_MERGE)

    for i in range(N_VISIBLE):
        yc = Y_SSM_TOP - i * DY_SSM
        lightness = 1.0 - i * 0.07

        import colorsys

        r, g, b = (
            int(C_SSM[1:3], 16) / 255,
            int(C_SSM[3:5], 16) / 255,
            int(C_SSM[5:7], 16) / 255,
        )
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        r2, g2, b2 = colorsys.hsv_to_rgb(h, s, min(1.0, v * lightness))
        col = f"#{int(r2 * 255):02x}{int(g2 * 255):02x}{int(b2 * 255):02x}"

        (_box(ax, XC, yc, BW_CENTER, BH_SSM, col, f"Mamba2  —  layer {i + 1} "),)
        if i > 0:
            _arrow(
                ax,
                XC,
                Y_SSM_TOP - (i - 1) * DY_SSM - BH_SSM / 2,
                XC,
                yc + BH_SSM / 2,
                color=C_SSM,
            )

    # ellipsis
    Y_DOTS = Y_SSM_TOP - N_VISIBLE * DY_SSM + 0.012
    ax.text(
        XC,
        Y_DOTS,
        "⋮",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=20,
        color=C_SSM,
        zorder=4,
    )
    _arrow(
        ax,
        XC,
        Y_SSM_TOP - (N_VISIBLE - 1) * DY_SSM - BH_SSM / 2,
        XC,
        Y_DOTS - 0.02,
        color=C_SSM,
    )

    # final (Nth) layer
    Y_SSMLAST = Y_DOTS - 0.075
    _box(
        ax,
        XC,
        Y_SSMLAST,
        BW_CENTER,
        BH_SSM,
        C_SSM,
        "Mamba2  —  layer N",
        "Selective State Space Model  ·  linear complexity",
    )
    _arrow(ax, XC, Y_DOTS - 0.038, XC, Y_SSMLAST + BH_SSM / 2, color=C_SSM)

    # right-side "× N layers" brace
    _brace_right(
        ax,
        XC + BW_CENTER / 2 + 0.025,
        Y_SSMLAST - BH_SSM / 2,
        Y_SSM_TOP + BH_SSM / 2,
        "× N\nlayers",
        color=C_SSM,
        fs=8.5,
    )

    # ── ⑥ LAYER NORM ─────────────────────────────────────────────────────────
    Y_LN = Y_SSMLAST - BH_SSM / 2 - 0.060
    _box(ax, XC, Y_LN, 0.28, BH_S, C_NORM, "LayerNorm")
    _arrow(ax, XC, Y_SSMLAST - BH_SSM / 2, XC, Y_LN + BH_S / 2, color=C_NORM)

    # strip-token annotation (initial_state only)
    if mode == "initial_state":
        _text(
            ax,
            XC + BW_CENTER / 2 + 0.04,
            Y_LN,
            "strip image\ntoken first",
            fs=7.0,
            color="#AAAAAA",
            ha="left",
        )

    # ── ⑦ GMM OUTPUT HEAD ────────────────────────────────────────────────────
    _draw_head_gmm(ax, Y_LN, BH_S, BH, XC, BW_CENTER)

    # ── LEGEND ────────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(facecolor=C_IMG, label="Image path"),
        mpatches.Patch(facecolor=C_GAZE, label="Gaze path"),
        mpatches.Patch(facecolor=C_MERGE, label="Conditioning"),
        mpatches.Patch(facecolor=C_SSM, label="Mamba2 SSM backbone"),
        mpatches.Patch(facecolor=C_HEAD_GMM, label="GMM head"),
    ]
    # Place the legend below all diagram content so it never overlaps boxes.
    # bbox_to_anchor in axes-fraction coords: (0.5, -0.04) is centred just
    # beneath the axes.  bbox_inches="tight" in savefig will capture it.
    ax.legend(
        handles=legend_patches,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=3,
        frameon=True,
        framealpha=0.9,
        fontsize=7.5,
        title="Module type",
        title_fontsize=8,
        edgecolor="#DDDDDD",
    )

    # ── TITLE ─────────────────────────────────────────────────────────────────
    mode_str = "Initial-state" if mode == "initial_state" else "Every-step"
    ax.text(
        0.50,
        1.002,
        f"Kaamba — Gaze Predictor Architecture  ({mode_str} conditioning · GMM head)",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10.5,
        color=C_ANNO,
        fontweight="bold",
    )

    # ── SAVE ──────────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"architecture_{mode}_gmm"
    fig.savefig(
        out_dir / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white"
    )
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    print(f"  Saved {out_dir / stem}.{{png,pdf}}")
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Generate Kaamba architecture diagram")
    parser.add_argument(
        "--out_dir",
        default="plots/architecture",
        help="Output directory (default: plots/architecture)",
    )
    parser.add_argument(
        "--mode",
        choices=["initial_state", "every_step", "both"],
        default="both",
        help="Conditioning mode to draw (default: both)",
    )
    args = parser.parse_args()
    out = Path(args.out_dir)
    modes = ["initial_state", "every_step"] if args.mode == "both" else [args.mode]
    for m in modes:
        print(f"Drawing mode={m}  head=gmm …")
        make_diagram(out, m)
    print("Done.")


if __name__ == "__main__":
    main()
