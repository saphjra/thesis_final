"""
baselines.py

Sequence-generator baselines for gaze evaluation.

Two baselines
─────────────
  SyntheticGenerator
      Generates gaze via pm.synthetic.step_function — random fixation/saccade
      structure with Gaussian noise.  Screen dimensions and sampling rate are
      taken from the experiment object passed at generation time.

  TrainingDistributionGenerator
      Samples normalised (x, y) gaze coordinates i.i.d. from the empirical
      distribution observed in the *training* subset of a pymovements dataset
      (mcfw-gaze or GGTG).  Each coordinate is drawn proportionally to its
      frequency in the training data (sampling with replacement from the pool
      of all observed coordinates).

Both implement the SequenceGenerator ABC defined in evaluate_model.py so they
can be passed directly to run_evaluation() / run_multi_evaluation().

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pymovements as pm
import torch
from abc import ABC, abstractmethod



class SequenceGenerator(ABC):
    """Abstract interface: produce (N, gen_len, 2) normalised gaze arrays."""

    name: str = "unnamed"

    @abstractmethod
    def generate(
        self,
        img_path: Optional[Path],
        n: int,
        gen_len: int,
        seed_len: int,
        experiment,  # pm.gaze.Experiment
        device: str = "cpu",
    ) -> np.ndarray:
        """Return (N, gen_len, 2) float32 array of normalised gaze in [0, 1]."""


class SyntheticGenerator(SequenceGenerator):
    """
    Generate synthetic gaze using pm.synthetic.step_function.
    Screen dimensions and sampling rate are taken from the experiment object
    passed at generation time (extracted from the real dataset).
    """

    def __init__(
        self,
        fix_dur_mean_ms: float = 250.0,
        fix_dur_std_ms: float = 80.0,
        sac_dur_mean_ms: float = 40.0,
        sac_dur_std_ms: float = 15.0,
        noise: float = 0.5,
        values_spread: float = 0.7,
        start_value: Optional[tuple] = None,
        values_center: Optional[tuple] = None,
        seed: int = 42,
        label: Optional[str] = None,
    ):
        self.fix_mean = fix_dur_mean_ms
        self.fix_std = fix_dur_std_ms
        self.sac_mean = sac_dur_mean_ms
        self.sac_std = sac_dur_std_ms
        self.noise = noise
        self.spread = values_spread
        self.start = start_value
        self.center = values_center
        self.seed = seed
        self.name = label or "synthetic"

    def generate(self, img_path, n, gen_len, seed_len, experiment, device=None):
        screen = experiment.screen
        sr = float(experiment.sampling_rate)
        sw = int(screen.width_px)
        sh = int(screen.height_px)

        cx, cy = self.center or (sw / 2, sh / 2)
        start = self.start or (sw / 2, sh / 2)
        half_w = sw / 2 * self.spread
        half_h = sh / 2 * self.spread

        rng = np.random.default_rng(self.seed)
        seqs = []

        def _ms2s(ms):
            return max(1, int(round(ms * sr / 1000)))

        for _ in range(n):
            steps, values, cursor = [], [], 0
            while cursor < gen_len:
                fd = max(
                    _ms2s(self.fix_mean * 0.3),
                    int(rng.normal(_ms2s(self.fix_mean), _ms2s(self.fix_std))),
                )
                x = float(np.clip(rng.uniform(cx - half_w, cx + half_w), 0, sw))
                y = float(np.clip(rng.uniform(cy - half_h, cy + half_h), 0, sh))
                steps.append(cursor)
                values.append((x, y))
                cursor += fd
                if cursor >= gen_len:
                    break
                sd = max(1, int(rng.normal(_ms2s(self.sac_mean), _ms2s(self.sac_std))))
                nx = float(np.clip(rng.uniform(cx - half_w, cx + half_w), 0, sw))
                ny = float(np.clip(rng.uniform(cy - half_h, cy + half_h), 0, sh))
                steps.append(cursor)
                values.append(((x + nx) / 2, (y + ny) / 2))
                cursor += sd

            pairs = [(s, v) for s, v in zip(steps, values) if s < gen_len]
            if not pairs:
                pairs = [(0, (sw / 2, sh / 2))]
            s_list, v_list = zip(*pairs)

            pos = pm.synthetic.step_function(
                length=gen_len,
                steps=list(s_list),
                values=list(v_list),
                start_value=start,
                noise=self.noise,
            )  # (gen_len, 2) in pixels
            norm = np.stack([pos[:, 0] / sw, pos[:, 1] / sh], axis=1)
            seqs.append(norm.astype(np.float32))

        return np.stack(seqs)  # (N, gen_len, 2)


# ---------------------------------------------------------------------------
# Baseline 2 – empirical training-distribution sampler
# ---------------------------------------------------------------------------


class TrainingDistributionGenerator(SequenceGenerator):
    """
    Sample normalised (x, y) gaze coordinates i.i.d. from the empirical
    distribution of the training subset of a pymovements dataset.

    At first generate() call (or at __init__ if ``eager=True``) the training
    subset is loaded, all pixel coordinates are normalised to [0, 1] and
    stored as a flat pool.  Each generated sequence draws ``gen_len`` points
    with replacement from this pool, which is equivalent to sampling according
    to the observed coordinate frequency.

    Parameters
    ----------
    dataset_name  : "mcfw-gaze" or "GGTG"
    root          : Root directory for pymovements data.
    train_subset  : Dict passed to dataset.load(subset=…) to select the
                    training recordings.  E.g. {"subject_id": ["P01", "P02"]}.
                    None → load all available recordings.
    seed          : RNG seed for reproducibility.
    eager         : If True, build the coordinate pool at __init__ time.
    label         : Override the generator name used in output paths.
    """

    def __init__(
        self,
        dataset_name: str,
        root: str,
        train_subset: Optional[Dict] = None,
        seed: int = 42,
        eager: bool = False,
        label: Optional[str] = None,
    ):
        self.dataset_name = dataset_name
        self.root = root
        self.train_subset = train_subset
        self.seed = seed
        self.name = label or f"empirical_{dataset_name}"
        self._pool: Optional[np.ndarray] = None  # (M, 2) float32

        if eager:
            self._build_pool()

    # ------------------------------------------------------------------
    def _build_pool(self) -> None:
        """Load training data and collect all normalised (x, y) points."""
        print(
            f"[{self.name}] Loading training distribution from "
            f"{self.dataset_name} (subset={self.train_subset}) …"
        )
        dataset_paths = pm.DatasetPaths(root=self.root)
        ds = pm.Dataset(self.dataset_name, path=dataset_paths)
        ds.scan()

        if self.dataset_name == "GGTG":
            # split_gaze_data must run before stimulus-based filtering is possible
            ds.load()
            ds.split_gaze_data(by="stimulus")
            if self.train_subset and "stimulus" in self.train_subset:
                stim_val = self.train_subset["stimulus"]
                if stim_val and isinstance(stim_val[0], list):
                    stim_val = stim_val[0]
                keep = set(stim_val)
                ds.gaze = [g for g in ds.gaze if g.metadata.get("stimulus") in keep]
        else:
            ds.load(subset=self.train_subset)

        first_gaze = ds.gaze[0]
        screen = first_gaze.experiment.screen
        sw = int(screen.width_px)
        sh = int(screen.height_px)

        coords: list[np.ndarray] = []
        for gaze in ds.gaze:
            try:
                px = np.stack(gaze.samples["pixel"].to_numpy())  # (T, 2)
                norm = np.column_stack([px[:, 0] / sw, px[:, 1] / sh]).astype(
                    np.float32
                )
                # drop blinks / missing samples (NaN) and out-of-screen points
                valid = np.isfinite(norm).all(axis=1)
                norm = np.clip(norm[valid], 0.0, 1.0)
                if len(norm):
                    coords.append(norm)
            except Exception as e:
                print(f"  [warn] skipping gaze object: {e}")

        if not coords:
            raise RuntimeError(
                f"[{self.name}] No valid pixel data found in training subset."
            )

        self._pool = np.concatenate(coords, axis=0)  # (M, 2)
        print(
            f"[{self.name}] Training pool built: {len(self._pool):,} coordinate samples"
        )

    # ------------------------------------------------------------------
    def generate_fully_random(self, img_path, n, gen_len, seed_len, experiment, device=None):
        if self._pool is None:
            self._build_pool()

        rng = np.random.default_rng(self.seed)
        idx = rng.integers(0, len(self._pool), size=(n, gen_len))
        return self._pool[idx].astype(np.float32)  # (N, gen_len, 2)

    def generate(self, img_path, n, gen_len, seed_len, experiment, device=None):
        """naive fixation generation"""
        if self._pool is None:
            self._build_pool()

        # probability of holding the current sample for one more step
        # = 1 - 1/mean_hold, so expected hold length = mean_hold samples
        mean_hold_ms = 220.0
        sampling_hz = getattr(experiment, "sampling_rate", 1000)
        mean_hold = mean_hold_ms * sampling_hz / 1000.0  # in samples
        p_hold = 1.0 - 1.0 / mean_hold

        rng = np.random.default_rng(self.seed)
        pool_len = len(self._pool)

        out = np.empty((n, gen_len, 2), dtype=np.float32)

        for i in range(n):
            # draw a starting point
            current = self._pool[rng.integers(0, pool_len)]
            for t in range(gen_len):
                out[i, t] = current
                if rng.random() >= p_hold:  # release: jump to new point
                    current = self._pool[rng.integers(0, pool_len)]

        return out
# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------


def test_baselines(
    dataset_name: str = "GGTG",
    root: str = r"C:\Users\saphi\PycharmProjects\thesis\data",
    train_subset: Optional[Dict] = None,
    eval_subset: Optional[Dict] = None,
    n_generate: int = 5,
    gen_len: int = 200,
    seed_len: int = 10,
):
    """
    Quick smoke-test: load a tiny slice of the dataset, build both generators,
    call generate() once each, and print shape + value-range checks.

    Parameters
    ----------
    dataset_name  : pymovements dataset to use.
    root          : Data root directory.
    train_subset  : Passed to TrainingDistributionGenerator.
    eval_subset   : Used to load dataset for the experiment object.
    n_generate    : Number of sequences to generate.
    gen_len       : Sequence length.
    seed_len      : Seed length (unused by baselines but part of the API).
    """
    print("=" * 60)
    print(f"Baseline smoke-test  |  dataset={dataset_name}")
    print("=" * 60)

    # ── Load a small slice to get a pm.Experiment object ─────────────
    dataset_paths = pm.DatasetPaths(root=root)
    ds = pm.Dataset(dataset_name, path=dataset_paths)
    ds.scan()

    if dataset_name == "GGTG":
        ds.load()
        ds.split_gaze_data(by="stimulus")
    else:
        ds.load(subset=eval_subset)
    experiment = ds.gaze[0].experiment
    print(f"Experiment loaded  screen={experiment.screen.width_px}x{experiment.screen.height_px}px  sr={experiment.sampling_rate}Hz")

    # ── Baseline 1: SyntheticGenerator ───────────────────────────────
    print("\n── SyntheticGenerator ──")
    syn = SyntheticGenerator(seed=0)
    syn_out = syn.generate(
        img_path=None,
        n=n_generate,
        gen_len=gen_len,
        seed_len=seed_len,
        experiment=experiment,
    )
    assert syn_out.shape == (n_generate, gen_len, 2), f"Bad shape: {syn_out.shape}"
    assert np.isfinite(syn_out).all(), "Contains non-finite values"
    print(f"  output shape : {syn_out.shape}")
    print(f"  x range      : [{syn_out[..., 0].min():.3f}, {syn_out[..., 0].max():.3f}]")
    print(f"  y range      : [{syn_out[..., 1].min():.3f}, {syn_out[..., 1].max():.3f}]")
    print("  PASS")

    # ── Baseline 2: TrainingDistributionGenerator ────────────────────
    print("\n── TrainingDistributionGenerator ──")
    emp = TrainingDistributionGenerator(
        dataset_name=dataset_name,
        root=root,
        train_subset=train_subset,
        seed=0,
        eager=True,
    )
    emp_out = emp.generate(
        img_path=None,
        n=n_generate,
        gen_len=gen_len,
        seed_len=seed_len,
        experiment=experiment,
    )
    assert emp_out.shape == (n_generate, gen_len, 2), f"Bad shape: {emp_out.shape}"
    assert np.isfinite(emp_out).all(), "Contains non-finite values"
    assert emp_out.min() >= 0.0 and emp_out.max() <= 1.0, "Values outside [0, 1]"
    print(f"  output shape : {emp_out.shape}")
    print(f"  x range      : [{emp_out[..., 0].min():.3f}, {emp_out[..., 0].max():.3f}]")
    print(f"  y range      : [{emp_out[..., 1].min():.3f}, {emp_out[..., 1].max():.3f}]")
    print(f"  pool size    : {len(emp._pool):,}")
    print("  PASS")

    print("\n" + "=" * 60)
    print("All baseline smoke-tests passed.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    GGTG_TRAIN_STIM = [
        "blackout-neg.difficulty",
        "blackout-neg.interest",
        "blackout-neg.naturalness",
        "blackout-neg.question",
        "blackout-neg.text.0",
        "blackout-neg.text.1",
        "blackout-neg.text.2",
        "blackout-neg.text.3",
        "blackout-pos.difficulty",
        "blackout-pos.interest",
        "blackout-pos.naturalness",
        "blackout-pos.question",
        "blackout-pos.text.0",
        "blackout-pos.text.1",
        "blackout-pos.text.2",
        "blackout-pos.text.3",
        "blackout-pos.text.4",
        "blackout-zero.difficulty",
        "blackout-zero.interest",
        "blackout-zero.naturalness",
        "blackout-zero.question",
        "blackout-zero.text.0",
        "blackout-zero.text.1",
        "blackout-zero.text.2",
        "blackout-zero.text.3",
        "blackout-zero.text.4",
        "breakfast-neg.difficulty",
        "breakfast-neg.interest",
        "breakfast-neg.naturalness",
        "breakfast-neg.question",
        "breakfast-neg.text.0",
        "breakfast-neg.text.1",
        "breakfast-neg.text.2",
        "breakfast-neg.text.3",
        "breakfast-pos.difficulty",
        "breakfast-pos.interest",
        "breakfast-pos.naturalness",
        "breakfast-pos.question",
        "breakfast-pos.text.0",
        "breakfast-pos.text.1",
        "breakfast-pos.text.2",
        "breakfast-pos.text.3",
        "breakfast-zero.difficulty",
        "breakfast-zero.interest",
        "breakfast-zero.naturalness",
        "breakfast-zero.question",
        "breakfast-zero.text.0",
        "breakfast-zero.text.1",
        "breakfast-zero.text.2",
        "breakfast-zero.text.3",
        "delayed-neg.difficulty",
        "delayed-neg.interest",
        "delayed-neg.naturalness",
        "delayed-neg.question",
        "delayed-neg.text.0",
        "delayed-neg.text.1",
        "delayed-neg.text.2",
        "delayed-neg.text.3",
        "delayed-neg.text.4",
        "delayed-pos.difficulty",
        "delayed-pos.interest",
        "delayed-pos.naturalness",
        "delayed-pos.question",
        "delayed-pos.text.0",
        "delayed-pos.text.1",
        "delayed-pos.text.2",
        "delayed-pos.text.3",
        "delayed-zero.difficulty",
        "delayed-zero.interest",
        "delayed-zero.naturalness",
        "delayed-zero.question",
        "delayed-zero.text.0",
        "delayed-zero.text.1",
        "delayed-zero.text.2",
        "delayed-zero.text.3",
        "delayed-zero.text.4",
        "goldfish-neg.difficulty",
        "goldfish-neg.interest",
        "goldfish-neg.naturalness",
        "goldfish-neg.question",
        "goldfish-neg.text.0",
        "goldfish-neg.text.1",
        "goldfish-neg.text.2",
        "goldfish-neg.text.3",
        "goldfish-pos.difficulty",
        "goldfish-pos.interest",
        "goldfish-pos.naturalness",
        "goldfish-pos.question",
        "goldfish-pos.text.0",
        "goldfish-pos.text.1",
        "goldfish-pos.text.2",
        "goldfish-pos.text.3",
        "goldfish-pos.text.4",
        "goldfish-zero.difficulty",
        "goldfish-zero.interest",
        "goldfish-zero.naturalness",
        "goldfish-zero.question",
        "goldfish-zero.text.0",
        "goldfish-zero.text.1",
        "goldfish-zero.text.2",
        "goldfish-zero.text.3",
        "goldfish-zero.text.4",
        "practice.difficulty",
        "practice.interest",
        "practice.naturalness",
        "practice.question",
        "practice.text.0",
        "practice.text.1",
        "prize-neg.difficulty",
        "prize-neg.interest"
      ]
    GGTG_EVAL_STIM= [
        "prize-zero.text.4",
        "voicemail-neg.difficulty",
        "voicemail-neg.interest",
        "voicemail-neg.naturalness",
        "voicemail-neg.question",
        "voicemail-neg.text.0",
        "voicemail-neg.text.1",
        "voicemail-neg.text.2",
        "voicemail-neg.text.3",
        "voicemail-pos.difficulty",
        "voicemail-pos.interest",
        "voicemail-pos.naturalness",
        "voicemail-pos.question",
        "voicemail-pos.text.0",
        "voicemail-pos.text.1",
        "voicemail-pos.text.2",
        "voicemail-pos.text.3",
        "voicemail-zero.difficulty",
        "voicemail-zero.interest",
        "voicemail-zero.naturalness",
        "voicemail-zero.question",
        "voicemail-zero.text.0",
        "voicemail-zero.text.1",
        "voicemail-zero.text.2",
        "voicemail-zero.text.3"
      ],

    parser = argparse.ArgumentParser(
        description="test both gaze baselines against a pymovements dataset"
    )
    parser.add_argument("--dataset", default="GGTG")
    parser.add_argument(
        "--root", default=r"C:\Users\saphi\PycharmProjects\thesis\data"
    )
    parser.add_argument(
        "--train_stimuli",
        nargs="*",
        default=GGTG_TRAIN_STIM,
        help="Stimulus IDs to use as training data for TrainingDistributionGenerator. "
             "None → use all stimuli.",
    )
    parser.add_argument(
        "--eval_stimuli",
        nargs="*",
        default=GGTG_EVAL_STIM,
        help="Stimulus IDs to load for the experiment object. None → all.",
    )
    parser.add_argument("--n_generate", type=int, default=5)
    parser.add_argument("--gen_len", type=int, default=400)
    args = parser.parse_args()

    train_subset = (
        {"stimulus": args.train_stimuli} if args.train_stimuli else None
    )
    eval_subset = (
        {"stimulus": args.eval_stimuli} if args.eval_stimuli else None
    )

    test_baselines(
        dataset_name=args.dataset,
        root=args.root,
        train_subset=train_subset,
        eval_subset=eval_subset,
        n_generate=args.n_generate,
        gen_len=args.gen_len,
    )


if __name__ == "__main__":
    main()
