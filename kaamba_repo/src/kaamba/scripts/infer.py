"""
infer.py — Single-image scanpath inference
==========================================

Load a trained model checkpoint and sample one or more scanpaths for a
given stimulus image.  The model type (GMM vs categorical) is detected
automatically from the checkpoint.

Usage
─────
  # One scanpath, show interactively
  python infer.py --checkpoint best_model.pt --image stimulus.jpg

  # 10 scanpaths, save everything, skip display window
  python infer.py --checkpoint best_model.pt --image stimulus.jpg \\
      --n 10 --save_json --save_npy --save_plot --no_show

  # Grid layout (one subplot per scanpath) — good for N ≤ 12
  python infer.py --checkpoint best_model.pt --image stimulus.jpg \\
      --n 6 --layout grid --save_plot --no_show

  # Overlay fixation circles (requires screen size in degrees)
  python infer.py --checkpoint best_model.pt --image stimulus.jpg \\
      --n 5 --show_fixations --screen_w_deg 36 --screen_h_deg 20 --sr 500

  # Load settings from a config file, override n on the CLI
  python infer.py --config infer_config.json --image stimulus.jpg --n 20

Config file (--config)
──────────────────────
  Plain JSON whose keys match CLI argument names:

      {
          "checkpoint":  "/path/to/best_model.pt",
          "n":           10,
          "gen_len":     256,
          "seed_len":    10,
          "temperature": 1.0,
          "start_x":     0.5,
          "start_y":     0.1,
          "device":      "cuda",
          "out_dir":     "infer_results"
      }

Output files (all written to --out_dir / <model-name>/)
────────────────────────────────────────────────────────
  scanpaths.json   N × T × 2 coordinates + metadata     (--save_json)
  scanpaths.npy    (N, T, 2) float32 numpy array        (--save_npy)
  scanpaths.png    scanpath overlay on stimulus image    (--save_plot)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Model loading (auto-detect GMM vs categorical)
# ---------------------------------------------------------------------------


def _load_model(
    checkpoint_path: str,
    device: str,
) -> Tuple[torch.nn.Module, str, dict]:
    """
    Load a model from a checkpoint.

    Returns
    -------
    model       : the model in eval mode, moved to ``device``
    model_type  : ``"gmm"`` or ``"categorical"``
    model_config: dict stored in the checkpoint
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config: dict = ckpt.get("config", {}).get("model_config", {})
    from kaamba.net.models.kaamba import build_gaze_predictor

    model = build_gaze_predictor(**model_config)
    model_type = "gmm"

    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    n_params = sum(p.numel() for p in model.parameters())
    run_name = Path(checkpoint_path).parent.parent.name
    print(f"[infer] model type  : {model_type}  ({n_params:,} params)")
    print(f"[infer] run name    : {run_name}")
    if model_type == "categorical":
        print(f"[infer] n_bins      : {model_config.get('n_bins', '?')}")
    return model, model_type, model_config


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------


def _load_image_tensor(img_path: Path, device: str) -> torch.Tensor:
    """Load and resize an image to (1, 3, 224, 224)."""
    import torchvision.transforms.functional as TF
    from PIL import Image

    img = Image.open(img_path).convert("RGB")
    return TF.to_tensor(TF.resize(img, [224, 224])).unsqueeze(0).to(device)


# ---------------------------------------------------------------------------
# Autoregressive generation (standalone, supports custom seed position)
# ---------------------------------------------------------------------------


def _generate_gmm(
    model,
    img_tensor: torch.Tensor,  # (1, 3, 224, 224)
    n: int,
    gen_len: int,
    seed_len: int,
    temperature: float,
    start_xy: Tuple[float, float],
    device: str,
) -> np.ndarray:
    """
    Autoregressively sample from a GMM model.
    Returns (n, gen_len, 2) float32 in normalised [0, 1].
    """
    sx, sy = start_xy
    images = img_tensor.expand(n, -1, -1, -1)  # (N, 3, 224, 224)
    seq = torch.zeros(n, 2, seed_len, device=device)
    seq[:, 0, :] = sx
    seq[:, 1, :] = sy

    with torch.no_grad():
        for _ in range(gen_len - seed_len):
            pi, mu, log_sx, log_sy, rho_raw = model(images, seq)

            # ── Last time-step distribution ───────────────────────────────
            pi_t = torch.softmax(pi[:, -1, :], dim=-1)  # (N, K)
            mu_t = mu[:, -1, :, :]  # (N, K, 2)
            sx_t = log_sx[:, -1, :].exp().clamp(1e-4) * temperature
            sy_t = log_sy[:, -1, :].exp().clamp(1e-4) * temperature
            rho_t = torch.tanh(rho_raw[:, -1, :]) * 0.99  # (N, K)

            # ── Sample mixture component + draw from chosen Gaussian ──────
            k_idx = torch.multinomial(pi_t, 1).squeeze(-1)  # (N,)
            idx = torch.arange(n, device=device)
            mu_k = mu_t[idx, k_idx]  # (N, 2)
            sx_k = sx_t[idx, k_idx]
            sy_k = sy_t[idx, k_idx]
            rho_k = rho_t[idx, k_idx]

            z1 = torch.randn(n, device=device)
            z2 = torch.randn(n, device=device)
            x_t = mu_k[:, 0] + sx_k * z1
            y_t = mu_k[:, 1] + sy_k * (rho_k * z1 + (1 - rho_k**2).sqrt() * z2)

            seq = torch.cat([seq, torch.stack([x_t, y_t], 1).unsqueeze(2)], dim=2)

    return seq.permute(0, 2, 1).cpu().numpy().astype(np.float32)  # (N, T, 2)


def generate(
    checkpoint_path: str,
    image_path: str | Path,
    n: int = 1,
    gen_len: int = 128,
    seed_len: int = 10,
    temperature: float = 1.0,
    start_x: float = 0.5,
    start_y: float = 0.5,
    device: Optional[str] = None,
    rng_seed: int = 42,
) -> Tuple[np.ndarray, str]:
    """
    High-level generation API — usable from notebooks or other scripts.

    Parameters
    ----------
    checkpoint_path : path to ``best_model.pt``
    image_path      : path to the stimulus image
    n               : number of scanpaths to generate
    gen_len         : total sequence length in samples (seed + generated)
    seed_len        : burn-in length (samples held fixed at ``start_x/y``)
    temperature     : sampling temperature  (< 1 sharper, > 1 more diverse)
    start_x/y       : seed position in normalised [0, 1]  (default: centre)
    device          : ``"cuda"`` / ``"cpu"`` / ``None`` (auto-detect)
    rng_seed        : RNG seed for reproducibility

    Returns
    -------
    seqs       : (N, gen_len, 2) float32 normalised [0, 1]
    model_name : inferred from the checkpoint directory name
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)

    model, model_type, model_config = _load_model(str(checkpoint_path), device)
    img_t = _load_image_tensor(Path(image_path), device)  # (1, 3, 224, 224)

    start_xy = (start_x, start_y)

    if model_type == "gmm":
        seqs = _generate_gmm(
            model, img_t, n, gen_len, seed_len, temperature, start_xy, device
        )

    run_name = Path(checkpoint_path).parent.parent.name
    model_name = f"{model_type}_{run_name}"
    return seqs, model_name


# ---------------------------------------------------------------------------
# Optional fixation detection for visualisation
# ---------------------------------------------------------------------------


def _detect_fixations(
    seqs: np.ndarray,
    sr: float,
    screen_w_deg: float,
    screen_h_deg: float,
    vel_threshold: float = 30.0,
    min_fix_dur: int = 50,
):
    """
    Run IVT fixation detection on generated sequences.

    Returns a list of dicts, one per sequence, each containing:
      ``cx_norm``, ``cy_norm`` (centroid in [0,1]),
      ``duration`` (in samples).

    Skips gracefully if ``pymovements`` is not available.
    """
    import pymovements as pm

    fixations_per_seq = []

    for seq in seqs:
        xy_deg = seq.copy().astype(float)
        xy_deg[:, 0] *= screen_w_deg
        xy_deg[:, 1] *= screen_h_deg

        dx = np.diff(xy_deg, axis=0) * sr
        vel = np.concatenate([dx[:1], dx], axis=0)
        T = len(seq)

        try:
            evs = pm.events.ivt(
                velocities=vel,
                timesteps=np.arange(T, dtype=int),
                velocity_threshold=vel_threshold,
                minimum_duration=min_fix_dur,
                name="fixation",
            )
            rows = []
            for row in evs.frame.iter_rows(named=True):
                seg = seq[row["onset"] : row["offset"]]
                if len(seg) == 0:
                    continue
                rows.append(
                    {
                        "cx_norm": float(seg[:, 0].mean()),
                        "cy_norm": float(seg[:, 1].mean()),
                        "duration": int(row["duration"]),
                    }
                )
            fixations_per_seq.append(rows)
        except Exception:
            fixations_per_seq.append([])

    return fixations_per_seq


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def _draw_overlay(
    ax,
    seqs: np.ndarray,  # (N, T, 2) normalised
    image_path: Optional[Path],
    fixations_per_seq: Optional[list] = None,
    max_display: int = 20,
):
    """
    Draw all scanpaths on a single axes with the stimulus as background.

    • N = 1  → line coloured by time (plasma: light = early, dark = late)
    • N > 1  → each scanpath a different colour (tab10 / rainbow)
    Fixation circles are overlaid if ``fixations_per_seq`` is provided.
    """
    import matplotlib.pyplot as plt
    import matplotlib.collections as mc
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    n_show = min(len(seqs), max_display)
    seqs_s = seqs[:n_show]
    T = seqs_s.shape[1]

    # ── Background image ──────────────────────────────────────────────────
    if image_path is not None and Path(image_path).exists():
        from PIL import Image as _PIL

        img = np.array(_PIL.open(image_path).convert("RGB"))
        # extent=[left, right, bottom, top]; invert_yaxis makes y=0 the visual top
        ax.imshow(img, extent=[0, 1, 1, 0], aspect="auto", zorder=0)
    else:
        ax.set_facecolor("#1A1A2E")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.invert_yaxis()  # y=0 (screen top) → visual top
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Colour scheme ─────────────────────────────────────────────────────
    use_time_colour = n_show == 1

    if use_time_colour:
        cmap = plt.get_cmap("plasma")
        palette = None
    else:
        base = plt.get_cmap("tab10" if n_show <= 10 else "rainbow")
        palette = [base(i / max(n_show - 1, 1)) for i in range(n_show)]
        cmap = None

    # ── Draw traces ───────────────────────────────────────────────────────
    for idx, seq in enumerate(seqs_s):
        x, y = seq[:, 0], seq[:, 1]

        if use_time_colour:
            pts = np.array([x, y]).T.reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            norm = Normalize(vmin=0, vmax=T - 1)
            lc = mc.LineCollection(
                segs, cmap=cmap, norm=norm, linewidth=2.0, alpha=0.92, zorder=3
            )
            lc.set_array(np.arange(T - 1))
            ax.add_collection(lc)
            colour = cmap(1.0)
        else:
            colour = palette[idx]
            # thin white halo for legibility on busy images
            ax.plot(
                x,
                y,
                lw=2.8,
                color="white",
                alpha=0.30,
                solid_capstyle="round",
                zorder=2,
            )
            ax.plot(
                x, y, lw=1.6, color=colour, alpha=0.85, solid_capstyle="round", zorder=3
            )

        # ── Start (●) and end (▼) markers ────────────────────────────────
        for mx, my, marker in [(x[0], y[0], "o"), (x[-1], y[-1], "v")]:
            ax.plot(
                mx,
                my,
                marker,
                ms=7,
                color=colour,
                markeredgecolor="white",
                markeredgewidth=1.2,
                zorder=6,
            )

        # ── Fixation circles ──────────────────────────────────────────────
        if fixations_per_seq is not None and idx < len(fixations_per_seq):
            for fix in fixations_per_seq[idx]:
                r = np.clip(
                    fix["duration"] / 500.0, 0.005, 0.04
                )  # radius in data units
                circle = plt.Circle(
                    (fix["cx_norm"], fix["cy_norm"]),
                    r,
                    linewidth=1.5,
                    edgecolor=colour,
                    facecolor="none",
                    alpha=0.7,
                    zorder=5,
                )
                ax.add_patch(circle)

    # ── Time colourbar (single scanpath only) ─────────────────────────────
    if use_time_colour:
        sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=T - 1))
        sm.set_array([])
        cb = ax.figure.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
        cb.set_label("Time step", fontsize=8)
        cb.ax.tick_params(labelsize=7)

    if n_show < len(seqs):
        ax.text(
            0.5,
            -0.03,
            f"showing {n_show} of {len(seqs)}",
            transform=ax.transAxes,
            ha="center",
            fontsize=7,
            color="#888",
        )


def _draw_grid(
    fig,
    seqs: np.ndarray,  # (N, T, 2) normalised
    image_path: Optional[Path],
    fixations_per_seq: Optional[list] = None,
    max_cols: int = 4,
):
    """
    Draw one subplot per scanpath (time-coloured), up to len(seqs) panels.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    N = len(seqs)
    ncols = min(N, max_cols)
    nrows = (N + ncols - 1) // ncols
    gs = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.08, wspace=0.08)

    # pre-load image once
    img_arr = None
    if image_path is not None and Path(image_path).exists():
        from PIL import Image as _PIL

        img_arr = np.array(_PIL.open(image_path).convert("RGB"))

    for i in range(N):
        row, col = divmod(i, ncols)
        ax = fig.add_subplot(gs[row, col])
        if img_arr is not None:
            ax.imshow(img_arr, extent=[0, 1, 1, 0], aspect="auto", zorder=0)
        else:
            ax.set_facecolor("#1A1A2E")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.axis("off")

        # time-coloured single scanpath
        import matplotlib.collections as mc
        from matplotlib.colors import Normalize

        cmap = plt.get_cmap("plasma")
        seq = seqs[i]
        x, y = seq[:, 0], seq[:, 1]
        T = len(seq)
        pts = np.array([x, y]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        norm = Normalize(vmin=0, vmax=T - 1)
        lc = mc.LineCollection(
            segs, cmap=cmap, norm=norm, linewidth=1.5, alpha=0.90, zorder=3
        )
        lc.set_array(np.arange(T - 1))
        ax.add_collection(lc)

        # start/end markers
        ax.plot(
            x[0],
            y[0],
            "o",
            ms=5,
            color=cmap(0.0),
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=5,
        )
        ax.plot(
            x[-1],
            y[-1],
            "v",
            ms=5,
            color=cmap(1.0),
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=5,
        )

        # fixation circles
        if fixations_per_seq is not None and i < len(fixations_per_seq):
            for fix in fixations_per_seq[i]:
                r = np.clip(fix["duration"] / 500.0, 0.005, 0.04)
                circle = plt.Circle(
                    (fix["cx_norm"], fix["cy_norm"]),
                    r,
                    linewidth=1.2,
                    edgecolor="white",
                    facecolor="none",
                    alpha=0.6,
                    zorder=5,
                )
                ax.add_patch(circle)

        ax.set_title(f"#{i + 1}", fontsize=7, pad=2, color="#555")

    # hide unused subplots
    for i in range(N, nrows * ncols):
        row, col = divmod(i, ncols)
        fig.add_subplot(gs[row, col]).axis("off")


def visualize(
    seqs: np.ndarray,
    image_path: Optional[str | Path],
    out_path: Optional[str | Path] = None,
    show: bool = True,
    title: str = "Generated scanpaths",
    layout: str = "overlay",  # "overlay" or "grid"
    fixations_per_seq: Optional[list] = None,
    max_display: int = 20,
) -> None:
    """
    Visualise generated scanpaths overlaid on the stimulus image.

    Parameters
    ----------
    seqs             : (N, T, 2) normalised [0, 1]
    image_path       : path to stimulus image (or None for dark background)
    out_path         : where to save the PNG (or None to skip saving)
    show             : open an interactive matplotlib window
    title            : figure title
    layout           : ``"overlay"`` (all on one axes) or
                       ``"grid"``    (one subplot per scanpath)
    fixations_per_seq: list of fixation dicts from ``_detect_fixations``
    """
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N = len(seqs)

    if layout == "grid":
        ncols = min(N, 4)
        nrows = (N + ncols - 1) // ncols
        fig = plt.figure(figsize=(4 * ncols, 4 * nrows), facecolor="white")
        _draw_grid(
            fig,
            seqs,
            Path(image_path) if image_path else None,
            fixations_per_seq=fixations_per_seq,
        )
    else:  # overlay
        fig, ax = plt.subplots(figsize=(9, 7.5), facecolor="white")
        _draw_overlay(
            ax,
            seqs,
            Path(image_path) if image_path else None,
            fixations_per_seq=fixations_per_seq,
            max_display=max_display,
        )

    fig.suptitle(title, fontsize=10, y=0.995, va="top", color="#333")
    plt.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"[infer] plot  → {out_path}")

    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------


def save_json(seqs: np.ndarray, out_path: Path, meta: dict) -> None:
    """
    Save scanpaths as JSON.

    Schema::

        {
            "meta": { ... },
            "coordinate_system": "normalised_0_1",
            "shape": [N, T, 2],
            "scanpaths": [ [[x0,y0],[x1,y1],...], ... ]   // N × T × 2
        }
    """
    data = {
        "meta": meta,
        "coordinate_system": "normalised_0_1",
        "shape": list(seqs.shape),
        "scanpaths": seqs.tolist(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2))
    print(f"[infer] JSON  → {out_path}")


def save_npy(seqs: np.ndarray, out_path: Path) -> None:
    """Save (N, T, 2) float32 array as .npy."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), seqs)
    print(f"[infer] .npy  → {out_path}")


# ---------------------------------------------------------------------------
# Config file
# ---------------------------------------------------------------------------


def _load_config(path: str) -> dict:
    """Load a flat JSON config file whose keys match CLI argument names."""
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sample scanpath(s) from a trained model for a given stimulus image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Config file (--config):
  Store common settings in a JSON file so you don't retype them each run.
  CLI flags always override file values.

  Example (infer_config.json):
    {
        "checkpoint":  "/logs/runs/trial_0019/checkpoints/best_model.pt",
        "n":           10,
        "gen_len":     256,
        "temperature": 0.9,
        "out_dir":     "infer_results"
    }

  Usage:
    python infer.py --config infer_config.json --image stimulus.jpg
    python infer.py --config infer_config.json --image stimulus.jpg --n 20
""",
    )

    # ── Core ──────────────────────────────────────────────────────────────
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Path to best_model.pt (required unless set via --config)",
    )
    p.add_argument(
        "--image",
        required=True,
        metavar="PATH",
        help="Stimulus image to condition generation on",
    )
    p.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="JSON config file — keys set defaults, CLI flags override",
    )

    # ── Generation ────────────────────────────────────────────────────────
    p.add_argument(
        "--n", type=int, default=1, help="Number of scanpaths to generate  (default: 1)"
    )
    p.add_argument(
        "--gen_len",
        type=int,
        default=128,
        help="Total sequence length in samples  (default: 128)",
    )
    p.add_argument(
        "--seed_len",
        type=int,
        default=10,
        help="Seed / burn-in length in samples  (default: 10)",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature  (< 1 sharper, > 1 more diverse)",
    )
    p.add_argument(
        "--start_x",
        type=float,
        default=0.5,
        metavar="[0-1]",
        help="Seed x-position in normalised coordinates  (default: 0.5)",
    )
    p.add_argument(
        "--start_y",
        type=float,
        default=0.5,
        metavar="[0-1]",
        help="Seed y-position in normalised coordinates  (default: 0.5)",
    )
    p.add_argument(
        "--rng_seed",
        type=int,
        default=42,
        help="RNG seed for reproducibility  (default: 42)",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # ── Fixation overlay (optional) ───────────────────────────────────────
    p.add_argument(
        "--show_fixations",
        action="store_true",
        help="Overlay IVT fixation circles on the scanpath plot",
    )
    p.add_argument(
        "--sr",
        type=float,
        default=500.0,
        help="Sampling rate in Hz (needed for fixation detection, default: 500)",
    )
    p.add_argument(
        "--screen_w_deg",
        type=float,
        default=36.0,
        help="Screen width  in visual degrees (default: 36)",
    )
    p.add_argument(
        "--screen_h_deg",
        type=float,
        default=20.0,
        help="Screen height in visual degrees (default: 20)",
    )
    p.add_argument(
        "--vel_threshold",
        type=float,
        default=30.0,
        help="IVT velocity threshold in deg/s (default: 30)",
    )
    p.add_argument(
        "--min_fix_dur",
        type=int,
        default=50,
        help="Minimum fixation duration in samples (default: 50)",
    )

    # ── Output ────────────────────────────────────────────────────────────
    p.add_argument(
        "--out_dir",
        default="infer_results",
        help="Output root directory  (default: infer_results/)",
    )
    p.add_argument(
        "--label",
        default=None,
        help="Custom output-file prefix (default: derived from checkpoint)",
    )
    p.add_argument(
        "--layout",
        default="overlay",
        choices=["overlay", "grid"],
        help="Plot layout: 'overlay' (all in one) or 'grid' (one per scanpath)",
    )
    p.add_argument("--save_json", action="store_true", help="Save scanpaths as JSON")
    p.add_argument(
        "--save_npy",
        action="store_true",
        help="Save scanpaths as a (N, T, 2) .npy file",
    )
    p.add_argument(
        "--save_plot",
        action="store_true",
        help="Save the scanpath visualisation as PNG",
    )
    p.add_argument(
        "--no_show",
        action="store_true",
        help="Do not open an interactive matplotlib window",
    )

    return p


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    p = _build_parser()

    # ── Two-pass: scan sys.argv for --config, apply defaults, full parse ──
    _argv = sys.argv[1:]
    for i, arg in enumerate(_argv):
        if arg == "--config" and i + 1 < len(_argv):
            cfg = _load_config(_argv[i + 1])
            print(f"[config] loading {_argv[i + 1]}")
            valid = {a.dest for a in p._actions}
            applied = {k: v for k, v in cfg.items() if k in valid}
            p.set_defaults(**applied)
            print(f"[config] applied {len(applied)} defaults from file")
            break

    args = p.parse_args()

    # ── Validate ──────────────────────────────────────────────────────────
    if not args.checkpoint:
        p.error("--checkpoint is required (or set 'checkpoint' in --config)")
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        p.error(f"Checkpoint not found: {ckpt_path}")
    img_path = Path(args.image)
    if not img_path.exists():
        p.error(f"Image not found: {img_path}")
    if args.layout == "grid" and args.n > 24:
        print(
            f"[warn] --layout grid with n={args.n} will produce a very large figure. "
            "Consider reducing --n or switching to --layout overlay."
        )

    # ── Setup ─────────────────────────────────────────────────────────────
    np.random.seed(args.rng_seed)
    torch.manual_seed(args.rng_seed)

    # ── Load model ────────────────────────────────────────────────────────
    print(f"\n[infer] checkpoint  : {ckpt_path}")
    model, model_type, model_config = _load_model(str(ckpt_path), args.device)

    # ── Generate ──────────────────────────────────────────────────────────
    print(f"[infer] image       : {img_path}")
    print(
        f"[infer] generating  : {args.n} scanpath(s)"
        f"  gen_len={args.gen_len}  seed_len={args.seed_len}"
        f"  T={args.temperature}"
        f"  start=({args.start_x:.2f}, {args.start_y:.2f})"
    )

    img_t = _load_image_tensor(img_path, args.device)
    start_xy = (args.start_x, args.start_y)

    if model_type == "gmm":
        seqs = _generate_gmm(
            model,
            img_t,
            args.n,
            args.gen_len,
            args.seed_len,
            args.temperature,
            start_xy,
            args.device,
        )

    print(
        f"[infer] done        : shape={seqs.shape}"
        f"  x=[{seqs[..., 0].min():.3f}, {seqs[..., 0].max():.3f}]"
        f"  y=[{seqs[..., 1].min():.3f}, {seqs[..., 1].max():.3f}]"
    )

    # ── Optional fixation detection ───────────────────────────────────────
    fixations_per_seq = None
    if args.show_fixations:
        print("[infer] detecting fixations for plot overlay …")
        try:
            fixations_per_seq = _detect_fixations(
                seqs,
                args.sr,
                args.screen_w_deg,
                args.screen_h_deg,
                args.vel_threshold,
                args.min_fix_dur,
            )
            total_fix = sum(len(f) for f in fixations_per_seq)
            print(f"[infer] fixations   : {total_fix} across {len(seqs)} scanpath(s)")
        except ImportError:
            print("[warn] pymovements not available — skipping fixation overlay")

    # ── Determine output stem ─────────────────────────────────────────────
    run_name = ckpt_path.parent.parent.name
    stem = args.label or f"{model_type}_{run_name}"
    out_dir = Path(args.out_dir) / stem

    # ── Save outputs ──────────────────────────────────────────────────────
    if args.save_json:
        save_json(
            seqs,
            out_dir / "scanpaths.json",
            meta={
                "checkpoint": str(ckpt_path),
                "image": str(img_path),
                "model_type": model_type,
                "n": args.n,
                "gen_len": args.gen_len,
                "seed_len": args.seed_len,
                "temperature": args.temperature,
                "start_x": args.start_x,
                "start_y": args.start_y,
                "rng_seed": args.rng_seed,
            },
        )

    if args.save_npy:
        save_npy(seqs, out_dir / "scanpaths.npy")

    show = not args.no_show
    if args.save_plot or show:
        title = (
            f"{stem}  ·  {args.n} scanpath{'s' if args.n != 1 else ''}"
            f"  ·  T={args.temperature}"
            + (
                f"  ·  start=({args.start_x:.2f}, {args.start_y:.2f})"
                if (args.start_x, args.start_y) != (0.5, 0.5)
                else ""
            )
        )
        visualize(
            seqs=seqs,
            image_path=img_path,
            out_path=out_dir / "scanpaths.png" if args.save_plot else None,
            show=show,
            title=title,
            layout=args.layout,
            fixations_per_seq=fixations_per_seq,
        )

    print(f"\n[infer] {args.n} scanpath(s), length {args.gen_len} — complete")
    if args.save_json or args.save_npy or args.save_plot:
        print(f"[infer] output dir  : {out_dir}")


if __name__ == "__main__":
    main()
