#!/usr/bin/env python3
"""
Aggregator for the Abadi et al. (2019) & Gerth (2019) 7,200-Dataset Benchmark.
"""


import sys as _sys
from pathlib import Path as _Path
if str(_Path(__file__).resolve().parent) not in _sys.path:
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
from _env_paths import REPO_ROOT, DATA_DIR, SPECIMENS_DIR, ALIGNMENTS_DIR, SINGLE_CELL_DIR, RESULTS_DIR, WORK_ROOT, EXTERNAL_DIR, resolve_bin, ASTRAL_JAR

import glob
import json
import math
import statistics
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


def summarize_sim_group(records, key_prefix="sim_bic", methods=None):
    if methods is None:
        methods = [
            "bw_mdl",
            "bw_bin",
            "iqtree_bic",
            "iqtree_gtr",
            "iqtree_poly",
            "iqtree_jc",
            "fasttree",
            "bionj",
        ]
    out = {}
    for m in methods:
        rf_bin, nrf_bin, rf_ge1, nrf_ge1, fp, fn, prec, rec, rec_ge2, n_est, n_true, n_act = (
            [], [], [], [], [], [], [], [], [], [], [], []
        )
        for r in records:
            block = r.get(key_prefix) if key_prefix == "sim_bic" else r
            if not block:
                continue
            d = block.get(m) if key_prefix == "sim_bic" else block.get(f"{key_prefix}_{m}")
            if not d:
                continue
            rf_bin.append(d["rf_bin"])
            nrf_bin.append(d["nrf_bin"])
            rf_ge1.append(d["rf_ge1"])
            nrf_ge1.append(d["nrf_ge1"])
            fp.append(d["fp"])
            fn.append(d["fn"])
            prec.append(d["precision"])
            rec.append(d["recall"])
            rec_ge2.append(d["recall_ge2"])
            n_est.append(d["n_est"])
            n_true.append(d["n_true"])
            n_act.append(d["n_active_ge1"])
        if not rf_bin:
            continue
        out[m] = {
            "count": len(rf_bin),
            "n_true": mean_std(n_true)[0],
            "n_active_ge1": mean_std(n_act)[0],
            "n_est": mean_std(n_est)[0],
            "rf_bin": mean_std(rf_bin)[0],
            "nrf_bin": mean_std(nrf_bin)[0],
            "rf_ge1": mean_std(rf_ge1)[0],
            "nrf_ge1": mean_std(nrf_ge1)[0],
            "fp": mean_std(fp)[0],
            "fn": mean_std(fn)[0],
            "precision": mean_std(prec)[0],
            "recall": mean_std(rec)[0],
            "recall_ge2": mean_std(rec_ge2)[0],
        }
    return out


def main():
    res_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else EXTERNAL_DIR / "abadi_gerth" / "results"
    files = sorted(res_dir.glob("ds_*.json"))
    records = []
    for f in files:
        try:
            records.append(json.loads(f.read_text()))
        except Exception:
            pass

    print(f"Loaded {len(records)} completed dataset results from {res_dir}")
    if not records:
        return

    # 1. Empirical Re-Analysis (Gerth 2019)
    emp_recs = [r for r in records if "emp_bic_vs_gtr" in r and r["emp_bic_vs_gtr"] is not None]
    print(f"\n=== 1. GERTH (2019) EMPIRICAL RE-ANALYSIS (N_datasets = {len(emp_recs)}) ===")
    if emp_recs:
        total_agree = sum(r.get("emp_n_agree_bic_gtr", 0) for r in emp_recs)
        total_conflict = sum(r.get("emp_n_conflict_bic_gtr", 0) for r in emp_recs)
        total_bw_agree = sum(r.get("emp_bw_recovers_agree", 0) for r in emp_recs)
        total_bw_contracts_conflict = sum(r.get("emp_bw_contracts_conflict", 0) for r in emp_recs)
        total_conflict_sub1 = sum(r.get("emp_conflict_sub1_mut", 0) for r in emp_recs)

        ident_gtr = sum(1 for r in emp_recs if r["emp_bic_vs_gtr"]["rf_bin"] == 0) / len(emp_recs)
        ident_jc = sum(1 for r in emp_recs if r.get("emp_bic_vs_jc") and r["emp_bic_vs_jc"]["rf_bin"] == 0) / len(emp_recs)
        bw_zero_fp_vs_bic = sum(1 for r in emp_recs if r.get("emp_bic_vs_bw_mdl") and r["emp_bic_vs_bw_mdl"]["fp"] == 0) / len(emp_recs)

        print(f"  Empirical Topologies Identical (BIC vs GTR+I+G4): {ident_gtr*100:.1f}%")
        print(f"  Empirical Topologies Identical (BIC vs JC):       {ident_jc*100:.1f}%")
        print(f"  Bushwhacker (MDL) Zero-Conflict / 100% Split-Compatible with BIC: {bw_zero_fp_vs_bic*100:.1f}%")
        if total_conflict > 0:
            print(f"  Total Splits Agreed by BIC & GTR+I+G4:   {total_agree}")
            print(f"  Total Discordant Splits (BIC != GTR+I+G4): {total_conflict}")
            print(f"    -> Discordant splits with <1 expected mutation (l*L < 1): {total_conflict_sub1}/{total_conflict} ({total_conflict_sub1*100/total_conflict:.1f}%)")
            print(f"    -> Discordant splits automatically contracted by Bushwhacker (MDL): {total_bw_contracts_conflict}/{total_conflict} ({total_bw_contracts_conflict*100/total_conflict:.1f}%)")
        print(f"  Mean Runtime per Empirical Dataset:")
        print(f"    Bushwhacker (MDL):      {mean_std([r['emp_bw_time_s'] for r in emp_recs])[0]*1000:.2f} ms")
        print(f"    IQ-TREE 2 (GTR+I+G4):   {mean_std([r['emp_gtr_time_s'] for r in emp_recs])[0]*1000:.2f} ms")
        print(f"    IQ-TREE 2 (BIC MFP):    {mean_std([r['emp_bic_time_s'] for r in emp_recs])[0]*1000:.2f} ms")

        print("\n  --- Stratified by BIC Best-Fit Model Class (Gerth Fig. 2a/2c) ---")
        by_base = defaultdict(list)
        for r in emp_recs:
            by_base[r.get("best_bic_base", "Other")].append(r)
        for base in ["JC", "F81", "K80", "HKY", "SYM", "GTR"]:
            sub = by_base.get(base, [])
            if not sub:
                continue
            id_rate = sum(1 for r in sub if r["emp_bic_vs_gtr"]["rf_bin"] == 0) / len(sub)
            mean_nrf_gtr = mean_std([r["emp_bic_vs_gtr"]["nrf_bin"] for r in sub])[0]
            mean_fp_gtr = mean_std([r["emp_bic_vs_gtr"]["fp"] for r in sub])[0]
            mean_fp_bw = mean_std([r["emp_bic_vs_bw_mdl"]["fp"] for r in sub if r.get("emp_bic_vs_bw_mdl")])[0]
            mean_NxL = mean_std([r["NxL"] for r in sub])[0]
            print(
                f"    {base:4s} (n={len(sub):4d}, mean NxL={mean_NxL:8.0f}): "
                f"GTR==BIC={id_rate*100:5.1f}%, nRF(GTR,BIC)={mean_nrf_gtr:.4f}, "
                f"FP_vs_BIC: GTR={mean_fp_gtr:.2f} vs BW_MDL={mean_fp_bw:.2f}"
            )

    # 2. Ground-Truth Simulation on the Empirical BIC Trees & Models (`sim_bic`)
    sim_recs = [r for r in records if "sim_bic" in r and r["sim_bic"]]
    print(f"\n=== 2. MATCHED GROUND-TRUTH SIMULATION (`sim_bic`, N_datasets = {len(sim_recs)}) ===")
    if sim_recs:
        overall = summarize_sim_group(sim_recs, "sim_bic")
        print(
            f"  {'Method':15s} | {'Res/True':11s} | {'RF_bin':7s} | {'nRF_bin':7s} | {'RF_ge1':7s} | {'nRF_ge1':7s} | {'FP':6s} | {'Prec%':6s} | {'Rec>=2':6s}"
        )
        print("  " + "-" * 88)
        for m, st in overall.items():
            print(
                f"  {m:15s} | {st['n_est']:5.1f}/{st['n_true']:<5.1f} | {st['rf_bin']:7.2f} | {st['nrf_bin']:7.4f} | "
                f"{st['rf_ge1']:7.2f} | {st['nrf_ge1']:7.4f} | {st['fp']:6.2f} | {st['precision']*100:5.1f}% | {st['recall_ge2']*100:5.1f}%"
            )

        # Stratified by Stemminess (Proportion of internal branches with < 1 expected mutation)
        print("\n  --- `sim_bic` Stratified by Low Stemminess (Fraction of True Branches with l*L < 1.0) ---")
        bins_stem = [
            ("High Stemminess (0% short branches)", lambda r: r.get("emp_prop_sub1_mut", 0) == 0.0),
            ("Moderate Stemminess (0-25% short)", lambda r: 0.0 < r.get("emp_prop_sub1_mut", 0) <= 0.25),
            ("Low Stemminess (25-50% short)", lambda r: 0.25 < r.get("emp_prop_sub1_mut", 0) <= 0.50),
            ("Very Low Stemminess (>50% short)", lambda r: r.get("emp_prop_sub1_mut", 0) > 0.50),
        ]
        for label, pred in bins_stem:
            sub = [r for r in sim_recs if pred(r)]
            if not sub:
                continue
            st_sub = summarize_sim_group(sub, "sim_bic")
            print(f"  * {label} (n={len(sub)}):")
            for m in ["bw_mdl", "bw_bin", "iqtree_bic", "iqtree_gtr", "iqtree_poly", "fasttree"]:
                if m in st_sub:
                    s = st_sub[m]
                    print(
                        f"      {m:13s}: RF_bin={s['rf_bin']:6.2f} (nRF={s['nrf_bin']:.4f}), "
                        f"Realized_RF_ge1={s['rf_ge1']:6.2f} (nRF_ge1={s['nrf_ge1']:.4f}), "
                        f"FP={s['fp']:5.2f}, Prec={s['precision']*100:5.1f}%"
                    )

    # 3. Abadi et al. (2019) Complex Simulated Datasets (c0, c1, c2, c3)
    for c_set in ("c0", "c1", "c2", "c3"):
        c_recs = [r for r in records if f"{c_set}_bw_mdl" in r and r[f"{c_set}_bw_mdl"] is not None]
        if not c_recs:
            continue
        print(f"\n=== 3. ABADI ET AL. (2019) `{c_set.upper()}` COMPLEX HETEROGENEOUS BENCHMARK (N_datasets = {len(c_recs)}) ===")
        c_sum = summarize_sim_group(
            c_recs,
            key_prefix=c_set,
            methods=["bw_mdl", "bw_bin", "iqtree_gtr", "iqtree_poly", "fasttree"],
        )
        print(
            f"  {'Method':15s} | {'Res/True':11s} | {'RF_bin':7s} | {'nRF_bin':7s} | {'RF_ge1':7s} | {'nRF_ge1':7s} | {'FP':6s} | {'Prec%':6s} | {'Rec>=2':6s}"
        )
        print("  " + "-" * 88)
        for m, st in c_sum.items():
            print(
                f"  {m:15s} | {st['n_est']:5.1f}/{st['n_true']:<5.1f} | {st['rf_bin']:7.2f} | {st['nrf_bin']:7.4f} | "
                f"{st['rf_ge1']:7.2f} | {st['nrf_ge1']:7.4f} | {st['fp']:6.2f} | {st['precision']*100:5.1f}% | {st['recall_ge2']*100:5.1f}%"
            )


if __name__ == "__main__":
    main()
