#!/usr/bin/env python3
"""
Aggregates per-replicate JSON results from `sim_results/` into summary CSV and Markdown tables.
"""

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def mean_std(vals):
    if not vals:
        return 0.0, 0.0
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, 0.0
    v = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
    return m, math.sqrt(v)


def main():
    base_dir = Path(__file__).resolve().parent.parent
    sim_dir = base_dir / "sim_results"
    if not sim_dir.exists():
        print(f"No sim_results directory found at {sim_dir}")
        sys.exit(1)

    files = sorted(sim_dir.glob("*.json"))
    print(f"Found {len(files)} completed simulation replicate files.")

    grouped = defaultdict(lambda: defaultdict(list))
    scen_meta = {}

    for f in files:
        try:
            data = json.loads(f.read_text())
        except Exception as e:
            print(f"Warning: failed to parse {f}: {e}")
            continue
        scen_id = data["scenario_id"]
        scen_meta[scen_id] = {
            "regime": data.get("regime", ""),
            "description": data.get("description", ""),
            "N": data.get("N", 0),
            "L": data.get("L", 0),
        }
        for method, mdata in data.get("methods", {}).items():
            grouped[scen_id][method].append(mdata)

    rows = []
    method_order = [
        "Bushwhacker",
        "Bushwhacker_Binary",
        "IQTREE2",
        "IQTREE2_Polytomy",
        "RAxML_NG",
        "FastTree2",
        "BioNJ",
    ]

    for scen_id in sorted(grouped.keys()):
        meta = scen_meta[scen_id]
        for method in method_order:
            reps = grouped[scen_id].get(method, [])
            if not reps:
                continue
            n_reps = len(reps)
            prec_m, prec_s = mean_std([r["precision"] for r in reps])
            rec_m, rec_s = mean_std([r["recall"] for r in reps])
            rec1_m, rec1_s = mean_std([r["recall_ge1"] for r in reps])
            rec2_m, rec2_s = mean_std([r["recall_ge2"] for r in reps])
            fp_m, fp_s = mean_std([r["fp"] for r in reps])
            fn_m, fn_s = mean_std([r["fn"] for r in reps])
            rf_m, rf_s = mean_std([r["fp"] + r["fn"] for r in reps])
            tp_m, _ = mean_std([r["tp"] for r in reps])
            est_m, _ = mean_std([r["n_est_splits"] for r in reps])
            true_m, _ = mean_std([r["n_true_splits"] for r in reps])
            f1_m, f1_s = mean_std([r["f1"] for r in reps])
            nrf_m, nrf_s = mean_std([r["nrf"] for r in reps])
            rt_m, rt_s = mean_std([r["runtime_s"] for r in reps])
            shunts_m, _ = mean_std([r.get("certified_shunts", 0) for r in reps])

            rows.append(
                {
                    "scenario_id": scen_id,
                    "regime": meta["regime"],
                    "N": meta["N"],
                    "L": meta["L"],
                    "method": method,
                    "n_reps": n_reps,
                    "true_splits": round(true_m, 1),
                    "est_splits": round(est_m, 1),
                    "tp_mean": round(tp_m, 1),
                    "fp_mean": round(fp_m, 2),
                    "fn_mean": round(fn_m, 2),
                    "rf_mean": round(rf_m, 2),
                    "rf_sd": round(rf_s, 2),
                    "precision_mean": round(prec_m, 4),
                    "precision_sd": round(prec_s, 4),
                    "recall_all_mean": round(rec_m, 4),
                    "recall_ge1_mean": round(rec1_m, 4),
                    "recall_ge2_mean": round(rec2_m, 4),
                    "f1_mean": round(f1_m, 4),
                    "nrf_mean": round(nrf_m, 4),
                    "nrf_sd": round(nrf_s, 4),
                    "runtime_s_mean": round(rt_m, 4),
                    "shunts_mean": round(shunts_m, 2),
                }
            )

    csv_path = base_dir / "sim_summary.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote summary CSV to {csv_path}")

    # Print formatted table
    print(
        f"{'Scenario':<26} | {'Method':<18} | {'Reps':>4} | {'Splits':>11} | {'RF':>7} | {'nRF':>6} | {'FP':>6} | {'FN':>6} | {'Prec':>6} | {'Rec(all)':>8} | {'Rec(>=2)':>8} | {'Time(s)':>8}"
    )
    print("-" * 144)
    for r in rows:
        sp_str = f"{r['est_splits']:.0f}/{r['true_splits']:.0f}"
        print(
            f"{r['scenario_id']:<26} | {r['method']:<18} | {r['n_reps']:>4d} | {sp_str:>11} | {r['rf_mean']:>7.1f} | {r['nrf_mean']:>6.3f} | {r['fp_mean']:>6.1f} | {r['fn_mean']:>6.1f} | {r['precision_mean']:>6.3f} | {r['recall_all_mean']:>8.3f} | {r['recall_ge2_mean']:>8.3f} | {r['runtime_s_mean']:>8.3f}"
        )


if __name__ == "__main__":
    main()
