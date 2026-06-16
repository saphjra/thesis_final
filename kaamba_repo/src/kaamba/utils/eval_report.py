"""
eval_report.py

Aggregation and report helpers for evaluate_model.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np


def aggregate_results(all_results: Dict) -> Dict:
    """Average scalar metrics across all stimuli."""
    agg: Dict[str, Dict[str, list]] = {}

    for stim_metrics in all_results.values():
        for section, sub in stim_metrics.items():
            if not isinstance(sub, dict):
                continue
            if section not in agg:
                agg[section] = {}
            for k, v in sub.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    agg[section].setdefault(k, []).append(v)

    # Compute mean ± std, ignoring NaN
    out = {}
    for section, metrics in agg.items():
        out[section] = {}
        for k, vals in metrics.items():
            arr = np.array([v for v in vals if not np.isnan(v)])
            out[section][k] = {
                "mean": float(arr.mean()) if len(arr) else float("nan"),
                "std": float(arr.std()) if len(arr) else float("nan"),
                "n": int(len(arr)),
            }
    return out


def _agg(aggregate: Dict, section: str, key: str) -> str:
    """Format an aggregate entry as 'mean ± std' or 'n/a'."""
    v = aggregate.get(section, {}).get(key, {})
    if not v or np.isnan(v.get("mean", float("nan"))):
        return "n/a"
    return f"{v['mean']:.4f} ± {v['std']:.4f}"


def build_eval_report(all_results: Dict, aggregate: Dict, total_time: float) -> str:
    W = 72
    lines = [
        "=" * W,
        "GAZE EVALUATION REPORT",
        f"  {len(all_results)} stimuli evaluated in {total_time:.1f}s",
        "=" * W,
    ]

    # ── Fixation duration ─────────────────────────────────────────────────
    lines += [
        "",
        "FIXATION DURATION",
        "-" * W,
        f"  {'':42} {'real':>12}  {'fake':>12}",
        f"  {'Mean (samples)':42} {_agg(aggregate, 'fixation_duration', 'real_mean'):>12}  "
        f"{_agg(aggregate, 'fixation_duration', 'fake_mean'):>12}",
        f"  {'Std  (samples)':42} {_agg(aggregate, 'fixation_duration', 'real_std'):>12}  "
        f"{_agg(aggregate, 'fixation_duration', 'fake_std'):>12}",
        f"  {'KS statistic':42} {_agg(aggregate, 'fixation_duration', 'ks_stat'):>12}",
        f"  {'KS p-value':42} {_agg(aggregate, 'fixation_duration', 'p_value'):>12}",
        f"  {'N fixations — real':42} {_agg(aggregate, 'fixation_density_map', 'real_n_fixations'):>12}",
        f"  {'N fixations — fake':42} {_agg(aggregate, 'fixation_density_map', 'fake_n_fixations'):>12}",
    ]

    # ── Saccade amplitude ─────────────────────────────────────────────────
    lines += [
        "",
        "SACCADE AMPLITUDE",
        "-" * W,
        f"  {'':42} {'real':>12}  {'fake':>12}",
        f"  {'Mean amplitude (deg)':42} {_agg(aggregate, 'saccade_amplitude', 'real_mean_deg'):>12}  "
        f"{_agg(aggregate, 'saccade_amplitude', 'fake_mean_deg'):>12}",
        f"  {'KS statistic':42} {_agg(aggregate, 'saccade_amplitude', 'ks_stat'):>12}",
        f"  {'KS p-value':42} {_agg(aggregate, 'saccade_amplitude', 'p_value'):>12}",
    ]

    # ── Main sequence ─────────────────────────────────────────────────────
    lines += [
        "",
        "MAIN SEQUENCE (amplitude–peak-velocity correlation)",
        "-" * W,
        f"  {'Pearson r — real':42} {_agg(aggregate, 'main_sequence', 'real_r'):>12}",
        f"  {'Pearson r — fake':42} {_agg(aggregate, 'main_sequence', 'fake_r'):>12}",
    ]

    # ── Intersaccadic interval ────────────────────────────────────────────
    lines += [
        "",
        "INTERSACCADIC INTERVAL",
        "-" * W,
        f"  {'':42} {'real':>12}  {'fake':>12}",
        f"  {'Mean (samples)':42} {_agg(aggregate, 'intersaccadic_interval', 'real_mean'):>12}  "
        f"{_agg(aggregate, 'intersaccadic_interval', 'fake_mean'):>12}",
        f"  {'Variance (samples²)':42} {_agg(aggregate, 'intersaccadic_interval', 'real_var'):>12}  "
        f"{_agg(aggregate, 'intersaccadic_interval', 'fake_var'):>12}",
        f"  {'|Δ mean| (samples)':42} {_agg(aggregate, 'intersaccadic_interval', 'mean_err'):>12}",
    ]

    # ── Spatial metrics ───────────────────────────────────────────────────
    lines += [
        "",
        "SPATIAL METRICS",
        "-" * W,
        f"  {'Fixation density KL divergence':42} {_agg(aggregate, 'fixation_density_map', 'kl_divergence'):>12}",
        f"  {'Saccade direction KL divergence':42} {_agg(aggregate, 'saccade_direction', 'kl_divergence'):>12}",
    ]

    # ── Discriminability ──────────────────────────────────────────────────
    lines += [
        "",
        "DISCRIMINABILITY",
        "-" * W,
        f"  {'Classifier AUC':42} {_agg(aggregate, 'classifier_auc', 'auc'):>12}",
    ]

    # ── Per-stimulus summary ──────────────────────────────────────────────
    lines += [
        "",
        "PER-STIMULUS SUMMARY",
        "-" * W,
        f"  {'Stimulus':<40} {'fix_mean_r':>10} {'fix_mean_f':>10} "
        f"{'sac_amp_r':>9} {'sac_amp_f':>9} {'fix_KS':>7} {'sac_KS':>7} {'AUC':>7}",
    ]

    def _v(m, section, key):
        try:
            val = m[section][key]
            return f"{val:.3f}" if not np.isnan(val) else "  nan"
        except (KeyError, TypeError):
            return "  n/a"

    for stim, m in sorted(all_results.items()):
        lines.append(
            f"  {stim:<40}"
            f" {_v(m, 'fixation_duration', 'real_mean'):>10}"
            f" {_v(m, 'fixation_duration', 'fake_mean'):>10}"
            f" {_v(m, 'saccade_amplitude', 'real_mean_deg'):>9}"
            f" {_v(m, 'saccade_amplitude', 'fake_mean_deg'):>9}"
            f" {_v(m, 'fixation_duration', 'ks_stat'):>7}"
            f" {_v(m, 'saccade_amplitude', 'ks_stat'):>7}"
            f" {_v(m, 'classifier_auc', 'auc'):>7}"
        )

    # ── Pass / fail checks ────────────────────────────────────────────────
    lines += ["", "PASS / FAIL (aggregate means)", "-" * W]

    checks = {
        "Fixation KS p > 0.05": aggregate.get("fixation_duration", {})
        .get("p_value", {})
        .get("mean", 0)
        > 0.05,
        "Saccade  KS p > 0.05": aggregate.get("saccade_amplitude", {})
        .get("p_value", {})
        .get("mean", 0)
        > 0.05,
        "Main seq fake_r > 0.9": aggregate.get("main_sequence", {})
        .get("fake_r", {})
        .get("mean", 0)
        > 0.9,
        "Classifier AUC ≈ 0.5 (|AUC−0.5| < 0.1)": abs(
            aggregate.get("classifier_auc", {}).get("auc", {}).get("mean", 1) - 0.5
        )
        < 0.1,
    }
    for name, passed in checks.items():
        lines.append(f"  {'PASS' if passed else 'FAIL'}  {name}")

    lines.append("=" * W)
    return "\n".join(lines)


def save_comparison_table(all_gen_results: Dict[str, Dict], out_path: Path) -> None:
    """Write a side-by-side mean±std comparison table across all generators."""
    key_metrics = [
        # section, key, label, note
        # ── Fixation duration ──────────────────────────────────────────
        ("fixation_duration", "real_mean", "Fix dur   real mean (samp)", ""),
        ("fixation_duration", "real_std", "Fix dur   real std  (samp)", ""),
        ("fixation_duration", "fake_mean", "Fix dur   fake mean (samp)", ""),
        ("fixation_duration", "fake_std", "Fix dur   fake std  (samp)", ""),
        ("fixation_duration", "ks_stat", "Fix dur   KS stat         ", ""),
        ("fixation_duration", "p_value", "Fix dur   KS p-value      ", ""),
        # ── Fixation count ─────────────────────────────────────────────
        ("fixation_density_map", "real_n_fixations", "N fixations  real         ", ""),
        ("fixation_density_map", "fake_n_fixations", "N fixations  fake         ", ""),
        # ── Saccade amplitude ──────────────────────────────────────────
        ("saccade_amplitude", "real_mean_deg", "Sac amp   real mean (deg) ", ""),
        ("saccade_amplitude", "fake_mean_deg", "Sac amp   fake mean (deg) ", ""),
        ("saccade_amplitude", "ks_stat", "Sac amp   KS stat         ", ""),
        ("saccade_amplitude", "p_value", "Sac amp   KS p-value      ", ""),
        # ── Main sequence ──────────────────────────────────────────────
        ("main_sequence", "real_r", "Main seq  Pearson r real   ", ""),
        ("main_sequence", "fake_r", "Main seq  Pearson r fake   ", ""),
        # ── Intersaccadic interval ─────────────────────────────────────
        ("intersaccadic_interval", "real_mean", "ISI       real mean (samp)", ""),
        ("intersaccadic_interval", "fake_mean", "ISI       fake mean (samp)", ""),
        ("intersaccadic_interval", "real_var", "ISI       real var        ", ""),
        ("intersaccadic_interval", "fake_var", "ISI       fake var        ", ""),
        ("intersaccadic_interval", "mean_err", "ISI       |Δ mean|        ", ""),
        # ── Spatial ───────────────────────────────────────────────────
        ("fixation_density_map", "kl_divergence", "Fix density  KL div       ", ""),
        ("saccade_direction", "kl_divergence", "Sac direction KL div      ", ""),
        # ── Discriminability ──────────────────────────────────────────
        ("classifier_auc", "auc", "Classifier AUC            ", "0.5 = ideal"),
    ]

    gen_names = list(all_gen_results.keys())
    col_w = max(22, max(len(n) for n in gen_names) + 4)
    label_w = 28
    header = f"  {'Metric':<{label_w}}" + "".join(f"  {n:^{col_w}}" for n in gen_names)
    sep = "-" * len(header)

    lines = [
        "=" * len(header),
        "GENERATOR COMPARISON  (mean ± std across stimuli)",
        "=" * len(header),
        "",
        header,
        sep,
    ]

    prev_section = None
    for section, key, label, note in key_metrics:
        # blank line between sections for readability
        if prev_section and section != prev_section:
            lines.append("")
        prev_section = section

        row = f"  {label:<{label_w}}"
        for gen_name in gen_names:
            per_stim = all_gen_results[gen_name]
            vals = [
                m[section][key]
                for m in per_stim.values()
                if isinstance(m, dict)
                and section in m
                and isinstance(m[section], dict)
                and isinstance(m[section].get(key), (int, float))
            ]
            arr = np.array([v for v in vals if not np.isnan(v)])
            if len(arr):
                cell = f"{arr.mean():.4f} ±{arr.std():.4f}"
                if note:
                    cell += f"  [{note}]"
            else:
                cell = "n/a"
            row += f"  {cell:^{col_w}}"
        lines.append(row)

    lines += ["", "=" * len(header)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"[compare] table → {out_path}")
