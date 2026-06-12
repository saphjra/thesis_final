"""
synthetic_baseline.py

Generate synthetic gaze data via pm.synthetic.step_function() using the
physical properties (sampling rate, screen dimensions, average recording
length) extracted from a real pymovements dataset.

The generated pm.Gaze objects can be fed directly into evaluate_model.py
as a baseline comparison against the trained model.

Fixation structure
──────────────────
  Each "event" alternates between a fixation interval (gaze stays near a
  position) and a saccade interval (brief transition to the next position).
  Step onsets are drawn from normal distributions scaled by the sampling rate:

    fixation duration ~ N(fix_dur_mean_ms, fix_dur_std_ms)  [ms → samples]
    saccade  duration ~ N(sac_dur_mean_ms, sac_dur_std_ms)  [ms → samples]

  Fixation positions are drawn uniformly inside the screen, centred on
  ``values_center`` (default: screen centre) with adjustable spread.

Usage
─────
  from kaamba.scripts.synthetic_baseline import generate_synthetic_gaze

  gaze_objects = generate_synthetic_gaze(
      pm_dataset,
      n_recordings   = 10,
      start_value    = (640, 512),   # pixels; None → screen centre
      values_center  = (640, 512),   # pixels; None → screen centre
      values_spread  = 0.6,          # fraction of screen half-width/height
      noise          = 15.0,         # px, step_function Gaussian noise
      fix_dur_mean_ms = 250.0,
      fix_dur_std_ms  = 80.0,
      sac_dur_mean_ms = 40.0,
      sac_dur_std_ms  = 15.0,
  )

  # feed into evaluate_model.py:
  for g in gaze_objects:
      sequences = extract_sequences_from_gaze(g, ...)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pymovements as pm


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _dataset_properties(pm_dataset: pm.Dataset) -> dict:
    """
    Extract sampling rate, screen dimensions, and average recording length
    from the first valid gaze object in a loaded pymovements dataset.

    Returns
    -------
    dict with keys:
        sr             – sampling rate (Hz)
        screen_w_px    – screen width  (px)
        screen_h_px    – screen height (px)
        screen_w_cm    – screen width  (cm)   (may be None)
        screen_h_cm    – screen height (cm)   (may be None)
        distance_cm    – viewing distance (cm) (may be None)
        origin         – coordinate origin
        avg_length     – mean number of samples across all recordings (int)
    """

    if not pm_dataset.gaze:
        raise RuntimeError(
            "Dataset has no loaded gaze data — call dataset.load() first."
        )

    exp = pm_dataset.gaze[0].experiment
    screen = exp.screen

    lengths = [len(g.samples) for g in pm_dataset.gaze]
    avg_len = int(round(np.mean(lengths)))

    return dict(
        sr=float(exp.sampling_rate),
        screen_w_px=int(screen.width_px),
        screen_h_px=int(screen.height_px),
        screen_w_cm=getattr(screen, "width_cm", None),
        screen_h_cm=getattr(screen, "height_cm", None),
        distance_cm=getattr(exp, "distance_cm", None),
        origin=getattr(screen, "origin", "upper left"),
        avg_length=avg_len,
    )


def _ms_to_samples(duration_ms: float, sr: float) -> int:
    """Convert a duration in milliseconds to an integer sample count."""
    return max(1, int(round(duration_ms * sr / 1000.0)))


def _build_steps_and_values(
    length: int,
    sr: float,
    screen_w_px: int,
    screen_h_px: int,
    values_center: Tuple[float, float],
    values_spread: float,
    fix_dur_mean_ms: float,
    fix_dur_std_ms: float,
    sac_dur_mean_ms: float,
    sac_dur_std_ms: float,
    rng: np.random.Generator,
) -> Tuple[List[int], List[Tuple[float, float]]]:
    """
    Build the ``steps`` and ``values`` lists for pm.synthetic.step_function.

    Alternates fixation / saccade events.  Fixation positions are drawn
    from a uniform distribution centred on ``values_center``, bounded to
    the screen.  Saccade positions are the midpoint between the surrounding
    fixations (simulates the eye sweeping through intermediate space).
    """
    cx, cy = values_center
    half_w = screen_w_px / 2 * values_spread
    half_h = screen_h_px / 2 * values_spread

    steps: List[int] = []
    values: List[Tuple[float, float]] = []

    # Generate events until we fill ``length`` samples
    cursor = 0
    prev_fix: Optional[Tuple[float, float]] = None

    while cursor < length:
        # ── fixation ──────────────────────────────────────────────────────
        fix_dur = max(
            _ms_to_samples(fix_dur_mean_ms * 0.3, sr),
            int(
                rng.normal(
                    _ms_to_samples(fix_dur_mean_ms, sr),
                    _ms_to_samples(fix_dur_std_ms, sr),
                )
            ),
        )
        x = float(np.clip(rng.uniform(cx - half_w, cx + half_w), 0, screen_w_px))
        y = float(np.clip(rng.uniform(cy - half_h, cy + half_h), 0, screen_h_px))
        fix_pos = (x, y)

        steps.append(cursor)
        values.append(fix_pos)
        cursor += fix_dur

        if cursor >= length:
            break

        # ── saccade (midpoint between current and next fixation) ──────────
        sac_dur = max(
            1,
            int(
                rng.normal(
                    _ms_to_samples(sac_dur_mean_ms, sr),
                    _ms_to_samples(sac_dur_std_ms, sr),
                )
            ),
        )
        # Draw the *next* fixation position to interpolate through
        nx = float(np.clip(rng.uniform(cx - half_w, cx + half_w), 0, screen_w_px))
        ny = float(np.clip(rng.uniform(cy - half_h, cy + half_h), 0, screen_h_px))
        mid_pos = ((fix_pos[0] + nx) / 2, (fix_pos[1] + ny) / 2)

        steps.append(cursor)
        values.append(mid_pos)
        cursor += sac_dur

    # Trim to valid range
    pairs = [(s, v) for s, v in zip(steps, values) if s < length]
    if not pairs:
        pairs = [(0, values_center)]
    steps_out, values_out = zip(*pairs)
    return list(steps_out), list(values_out)


# ─────────────────────────────────────────────────────────────────────────────
# Main public function
# ─────────────────────────────────────────────────────────────────────────────


def generate_synthetic_gaze_random(
    root: str,
    dataset_name: str,
    n_recordings: int = 1,
    start_value: Optional[Tuple[float, float]] = None,
    values_center: Optional[Tuple[float, float]] = None,
    values_spread: float = 0.7,
    noise: float = 15.0,
    fix_dur_mean_ms: float = 250.0,
    fix_dur_std_ms: float = 80.0,
    sac_dur_mean_ms: float = 40.0,
    sac_dur_std_ms: float = 15.0,
    seed: int = 42,
    verbose: bool = True,
    subset: Optional[dict] = None,
) -> List[pm.Gaze]:
    """
    Generate synthetic gaze recordings using pm.synthetic.step_function,
    parameterised by the properties of ``pm_dataset``.

    Parameters
    ----------
    pm_dataset      : Loaded pm.Dataset (call .load() before passing).
    n_recordings    : Number of pm.Gaze objects to generate.
    start_value     : Starting (x, y) in pixels.  None → screen centre.
    values_center   : Centre of the random fixation-position distribution
                      in pixels.  None → screen centre.
    values_spread   : Spread as a fraction of the half-screen size
                      (0 = all fixations at centre; 1 = full screen).
    noise           : Gaussian noise std passed to step_function (pixels).
    fix_dur_mean_ms : Mean fixation duration (ms).
    fix_dur_std_ms  : Std  fixation duration (ms).
    sac_dur_mean_ms : Mean saccade  duration (ms).
    sac_dur_std_ms  : Std  saccade  duration (ms).
    seed            : RNG seed for reproducibility.
    verbose         : Print extracted dataset properties.

    Returns
    -------
    List of pm.Gaze objects, each of length ≈ dataset average recording length.
    These can be passed directly to evaluate_model.py
    """

    dataset_paths = pm.DatasetPaths(root=root)

    pm_dataset = pm.Dataset(dataset_name, path=dataset_paths)

    pm_dataset.scan()
    pm_dataset.load(subset=subset)

    props = _dataset_properties(pm_dataset)
    sr = props["sr"]
    sw = props["screen_w_px"]
    sh = props["screen_h_px"]
    avg_len = props["avg_length"]

    if verbose:
        print("─" * 55)
        print("Synthetic gaze generator — dataset properties")
        print(f"  Sampling rate   : {sr:.0f} Hz")
        print(f"  Screen          : {sw} × {sh} px")
        print(
            f"  Avg. rec. length: {avg_len:,} samples  "
            f"({avg_len / sr:.1f} s @ {sr:.0f} Hz)"
        )
        print(f"  n_recordings    : {n_recordings}")
        print(
            f"  fix_dur         : {fix_dur_mean_ms:.0f} ± {fix_dur_std_ms:.0f} ms  "
            f"= {_ms_to_samples(fix_dur_mean_ms, sr)} ± "
            f"{_ms_to_samples(fix_dur_std_ms, sr)} samples"
        )
        print(f"  sac_dur         : {sac_dur_mean_ms:.0f} ± {sac_dur_std_ms:.0f} ms")
        print(f"  noise           : {noise:.1f} px")
        print("─" * 55)

    # Defaults
    if start_value is None:
        start_value = (sw / 2, sh / 2)
    if values_center is None:
        values_center = (sw / 2, sh / 2)

    # Build the experiment object for the generated gaze
    exp_kwargs = dict(
        screen_width_px=sw,
        screen_height_px=sh,
        sampling_rate=sr,
        origin=props["origin"],
    )
    if props["screen_w_cm"] is not None:
        exp_kwargs["screen_width_cm"] = props["screen_w_cm"]
    if props["screen_h_cm"] is not None:
        exp_kwargs["screen_height_cm"] = props["screen_h_cm"]
    if props["distance_cm"] is not None:
        exp_kwargs["distance_cm"] = props["distance_cm"]

    experiment = pm.gaze.Experiment(**exp_kwargs)

    rng = np.random.default_rng(seed)
    results: List[pm.Gaze] = []

    for i in range(n_recordings):
        steps, values = _build_steps_and_values(
            length=avg_len,
            sr=sr,
            screen_w_px=sw,
            screen_h_px=sh,
            values_center=values_center,
            values_spread=values_spread,
            fix_dur_mean_ms=fix_dur_mean_ms,
            fix_dur_std_ms=fix_dur_std_ms,
            sac_dur_mean_ms=sac_dur_mean_ms,
            sac_dur_std_ms=sac_dur_std_ms,
            rng=rng,
        )

        positions = pm.synthetic.step_function(
            length=avg_len,
            steps=steps,
            values=values,
            start_value=start_value,
            noise=noise,
        )

        import polars as pl

        df = pl.DataFrame(
            {
                "time": np.arange(avg_len, dtype=float),
                "x_pix": positions[:, 0],
                "y_pix": positions[:, 1],
            }
        )

        gaze_obj = pm.Gaze(
            df,
            pixel_columns=["x_pix", "y_pix"],
            experiment=experiment,
        )
        results.append(gaze_obj)

        if verbose:
            n_fix = sum(
                1 for j in range(len(steps)) if j % 2 == 0
            )  # every other step is a fixation
            print(
                f"  [{i + 1:>3}/{n_recordings}] length={avg_len}  "
                f"steps={len(steps)}  ~{n_fix} fixations"
            )
    print(results)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Quick demo / CLI
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic gaze baselines from a pm.Dataset"
    )
    parser.add_argument("--dataset", default="GGTG")
    parser.add_argument("--root", default=r"C:\Users\saphi\PycharmProjects\thesis\data")
    parser.add_argument(
        "--subset",
        type=dict,
        default={"subject_id": ["P01"]},  # "trial_id": ["1", ",2", "3"]},
    )
    parser.add_argument("--n_recordings", type=int, default=5)
    parser.add_argument("--noise", type=float, default=15.0)
    parser.add_argument("--values_spread", type=float, default=0.7)
    parser.add_argument("--fix_dur_mean", type=float, default=250.0)
    parser.add_argument("--fix_dur_std", type=float, default=80.0)
    parser.add_argument("--sac_dur_mean", type=float, default=40.0)
    parser.add_argument("--sac_dur_std", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out_dir", default=None, help="If set, save traceplot PNGs here"
    )
    args = parser.parse_args()

    print(f"Loading {args.dataset} …")

    gazes = generate_synthetic_gaze_random(
        root=args.root,
        dataset_name=args.dataset,
        n_recordings=args.n_recordings,
        noise=args.noise,
        values_spread=args.values_spread,
        fix_dur_mean_ms=args.fix_dur_mean,
        fix_dur_std_ms=args.fix_dur_std,
        sac_dur_mean_ms=args.sac_dur_mean,
        sac_dur_std_ms=args.sac_dur_std,
        seed=args.seed,
        verbose=True,
        subset=args.subset,
    )

    if args.out_dir:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for i, g in enumerate(gazes):
            fig = pm.plotting.traceplot(g)
            fig.savefig(out / f"synthetic_{i:03d}.png", dpi=120, bbox_inches="tight")
            plt.close(fig)
        print(f"Traceplots saved to {out}")

    print(f"\nDone — generated {len(gazes)} synthetic pm.Gaze objects.")
    print("Pass these to evaluate_model.py for baseline comparison.")


if __name__ == "__main__":
    main()
