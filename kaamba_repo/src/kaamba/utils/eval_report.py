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


def build_eval_report(all_results: Dict, aggregate: Dict, total_time: float) -> str:
    lines = [
        "=" * 70,
        "GAZE EVALUATION REPORT",
        f"  {len(all_results)} stimuli evaluated in {total_time:.1f}s",
        "=" * 70,
        "",
        "AGGREGATE METRICS (mean ± std across stimuli)",
        "-" * 70,
    ]

    key_metrics = [
        ("fixation_duration", "ks_stat", "Fixation duration KS stat    "),
        ("fixation_duration", "p_value", "Fixation duration KS p-value "),
        ("saccade_amplitude", "ks_stat", "Saccade amplitude KS stat    "),
        ("saccade_amplitude", "real_mean_deg", "Saccade amplitude real (deg) "),
        ("saccade_amplitude", "fake_mean_deg", "Saccade amplitude fake (deg) "),
        ("main_sequence", "real_r", "Main sequence r (real)       "),
        ("main_sequence", "fake_r", "Main sequence r (fake)       "),
        ("intersaccadic_interval", "mean_err", "ISI mean error (samples)     "),
        ("fixation_density_map", "kl_divergence", "Fixation density KL div      "),
        ("saccade_direction", "kl_divergence", "Saccade direction KL div     "),
        ("classifier_auc", "auc", "Classifier AUC               "),
    ]

    for section, key, label in key_metrics:
        val = aggregate.get(section, {}).get(key, {})
        if val:
            lines.append(
                f"  {label}  {val['mean']:7.4f} ± {val['std']:.4f}  (n={val['n']})"
            )

    lines += [
        "",
        "PER-STIMULUS SUMMARY",
        "-" * 70,
        f"  {'Stimulus':<40} {'fix_KS':>8} {'sac_KS':>8} {'AUC':>8}",
    ]

    for stim, m in sorted(all_results.items()):
        fks = m["fixation_duration"]["ks_stat"]
        asks = m["saccade_amplitude"]["ks_stat"]
        auc = m["classifier_auc"]["auc"]
        lines.append(f"  {stim:<40} {fks:>8.4f} {asks:>8.4f} {auc:>8.4f}")

    lines += ["", "PASS / FAIL (aggregate means)", "-" * 70]

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
        "Classifier AUC ≈ 0.5": abs(
            aggregate.get("classifier_auc", {}).get("auc", {}).get("mean", 1) - 0.5
        )
        < 0.1,
    }
    for name, passed in checks.items():
        lines.append(f"  {'✓' if passed else '✗'} {name}")

    lines.append("=" * 70)
    return "\n".join(lines)


def save_comparison_table(all_gen_results: Dict[str, Dict], out_path: Path) -> None:
    """Write a side-by-side mean±std comparison table across all generators."""
    key_metrics = [
        ("fixation_duration", "ks_stat", "Fix duration KS "),
        ("fixation_duration", "p_value", "Fix duration p   "),
        ("saccade_amplitude", "ks_stat", "Sac amplitude KS "),
        ("main_sequence", "fake_r", "Main sequence r  "),
        ("fixation_density_map", "kl_divergence", "Fix density KL   "),
        ("saccade_direction", "kl_divergence", "Sac direction KL "),
        ("classifier_auc", "auc", "Classifier AUC   "),
    ]

    gen_names = list(all_gen_results.keys())
    col_w = max(20, max(len(n) for n in gen_names) + 4)
    header = f"  {'Metric':<22}" + "".join(f"  {n:^{col_w}}" for n in gen_names)
    sep = "-" * len(header)

    lines = [
        "=" * len(header),
        "GENERATOR COMPARISON",
        "=" * len(header),
        "",
        header,
        sep,
    ]

    for section, key, label in key_metrics:
        row = f"  {label:<22}"
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
            cell = f"{arr.mean():.4f} ±{arr.std():.4f}" if len(arr) else "n/a"
            row += f"  {cell:^{col_w}}"
        lines.append(row)

    lines += ["", "=" * len(header)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"[compare] table → {out_path}")
