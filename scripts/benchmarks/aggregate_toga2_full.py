#!/usr/bin/env python3
"""
Aggregate all 19,953 TOGA2 Bushwhacker JSONL summaries on silverback into
a comprehensive genome-wide statistics report + ranked catalogs of:
1. Gene tree vs. Species tree concordance & MDL calibration across the mammalian proteome
2. Top certified INTERVAL shunts (ectopic gene conversion / exon projection chimeras)
3. Top certified SPARSE shunts (adaptive functional convergence across mammals)
4. De novo species-tree shunts (`sp_top_shunts`: shunts certified when topology is fixed to speciesTree.nh)
5. Matched comparison of Reference-Projected (.codonified.fa) vs. WALRUS De Novo Codon MSAs (.msa.fa.gz)
"""


import sys as _sys
from pathlib import Path as _Path
if str(_Path(__file__).resolve().parent) not in _sys.path:
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
from _env_paths import REPO_ROOT, DATA_DIR, SPECIMENS_DIR, ALIGNMENTS_DIR, SINGLE_CELL_DIR, RESULTS_DIR, WORK_ROOT, EXTERNAL_DIR, resolve_bin, ASTRAL_JAR

import os
import glob
import json
import statistics as st
from collections import Counter

SUMMARIES_GLOB = str(WORK_ROOT / "toga2_nt" / "summaries" / "worker_*.jsonl")
MASTER_JSONL = str(WORK_ROOT / "toga2_nt" / "toga2_bushwhacker_master_19953.jsonl")
REPORT_JSON = str(RESULTS_DIR / "mammalian_toga2" / "toga2_full_proteome_report.json")


def pct(arr, q):
    if not arr:
        return 0.0
    s = sorted(arr)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


def main():
    records = []
    errors = []
    seen_genes = set()

    files = sorted(glob.glob(SUMMARIES_GLOB))
    with open(MASTER_JSONL, "w") as mout:
        for fpath in files:
            with open(fpath, "r") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    g = r.get("gene")
                    if not g or g in seen_genes:
                        continue
                    seen_genes.add(g)
                    mout.write(line + "\n")
                    if r.get("status") == "ok":
                        records.append(r)
                    else:
                        errors.append(r)

    n_ok = len(records)
    print(f"Total unique genes aggregated: {len(seen_genes)} (OK: {n_ok}, Error/Skipped: {len(errors)})")
    if n_ok == 0:
        return

    n_taxa = [r["n_taxa"] for r in records if r.get("n_taxa")]
    seq_len = [r["seq_len"] for r in records if r.get("seq_len")]
    runtimes = [r["runtime_sec"] for r in records if r.get("runtime_sec") is not None]

    # 1. Species tree concordance & MDL calibration (genes with >= 10 taxa)
    valid_sp = [r for r in records if r.get("sp_splits_count", 0) >= 5 and r.get("bin_splits_count", 0) >= 5]
    sp_sup_frac = [r["sp_supported_frac"] for r in valid_sp if r.get("sp_supported_frac") is not None]
    poly_frac = [r["mdl_polytomy_contracted_frac"] for r in valid_sp]
    bin_conf_rate = [r["bin_sp_conflict_rate"] for r in valid_sp]
    mdl_conf_rate = [r["mdl_sp_conflict_rate"] for r in valid_sp]
    false_elim = [r["false_conflict_eliminated_frac"] for r in valid_sp]
    d_mdl = [r["delta_mdl_gene_vs_sp_nats"] for r in valid_sp if r.get("delta_mdl_gene_vs_sp_nats") is not None]

    tot_sp_splits = sum(r["sp_splits_count"] for r in valid_sp)
    tot_sp_sup = sum(r["sp_supported_splits_count"] or 0 for r in valid_sp)
    tot_bin_splits = sum(r["bin_splits_count"] for r in valid_sp)
    tot_mdl_splits = sum(r["mdl_splits_count"] for r in valid_sp)
    tot_bin_conf = sum(r["bin_sp_conflict"] for r in valid_sp)
    tot_mdl_conf = sum(r["mdl_sp_conflict"] for r in valid_sp)
    tot_bin_shared = sum(r["bin_sp_shared"] for r in valid_sp)
    tot_mdl_shared = sum(r["mdl_sp_shared"] for r in valid_sp)

    # 2. Shunt statistics across the proteome
    genes_with_shunts = [r for r in records if r.get("n_shunts", 0) > 0]
    genes_with_int = [r for r in records if r.get("n_interval_shunts", 0) > 0]
    genes_with_sparse = [r for r in records if r.get("n_sparse_shunts", 0) > 0]
    tot_int_events = sum(r.get("n_interval_shunts", 0) for r in records)
    tot_sparse_events = sum(r.get("n_sparse_shunts", 0) for r in records)
    tot_sp_tree_shunts = sum(r.get("sp_shunts_count") or 0 for r in records)

    all_interval = []
    all_sparse = []
    all_sp_tree_shunts = []

    for r in records:
        for s in r.get("top_shunts", []):
            item = {
                "gene": r["gene"],
                "gene_symbol": r["gene_symbol"],
                "n_taxa": r["n_taxa"],
                "seq_len": r["seq_len"],
                **s,
            }
            if s.get("mode") == "INTERVAL":
                all_interval.append(item)
            elif s.get("mode") == "SPARSE":
                all_sparse.append(item)
        for s in r.get("sp_top_shunts", []):
            all_sp_tree_shunts.append({
                "gene": r["gene"],
                "gene_symbol": r["gene_symbol"],
                "n_taxa": r["n_taxa"],
                "seq_len": r["seq_len"],
                **s,
            })

    all_interval.sort(key=lambda x: x.get("net_mdl_gain_nats", 0.0), reverse=True)
    all_sparse.sort(key=lambda x: x.get("net_mdl_gain_nats", 0.0), reverse=True)
    all_sp_tree_shunts.sort(key=lambda x: x.get("net_mdl_gain_nats", 0.0), reverse=True)

    # 3. WALRUS de novo MSA vs TOGA2 reference-projected comparison
    w_recs = [
        r for r in records
        if r.get("walrus_comparison") and r["walrus_comparison"].get("resolved_splits", 0) >= 3
    ]
    w_exact_n = [r for r in w_recs if r["walrus_comparison"]["n_taxa"] == r["n_taxa"]]

    walrus_summary = None
    if w_recs:
        c_conf_rates = [r["mdl_sp_conflict_rate"] for r in w_recs]
        w_conf_rates = [r["walrus_comparison"]["sp_conflict_rate"] for r in w_recs]
        c_tot_conf = sum(r["mdl_sp_conflict"] for r in w_recs)
        w_tot_conf = sum(r["walrus_comparison"]["sp_conflict_splits"] for r in w_recs)
        c_tot_shared = sum(r["mdl_sp_shared"] for r in w_recs)
        w_tot_shared = sum(r["walrus_comparison"]["sp_shared_splits"] for r in w_recs)
        c_tot_int = sum(r["n_interval_shunts"] for r in w_recs)
        w_tot_int = sum(r["walrus_comparison"]["n_interval_shunts"] for r in w_recs)
        c_tot_sp = sum(r["n_sparse_shunts"] for r in w_recs)
        w_tot_sp = sum(r["walrus_comparison"]["n_sparse_shunts"] for r in w_recs)
        walrus_summary = {
            "n_compared_genes": len(w_recs),
            "n_exact_taxa_matched": len(w_exact_n),
            "codonified_mean_conflict_rate": round(st.mean(c_conf_rates), 4),
            "walrus_mean_conflict_rate": round(st.mean(w_conf_rates), 4),
            "codonified_median_conflict_rate": round(st.median(c_conf_rates), 4),
            "walrus_median_conflict_rate": round(st.median(w_conf_rates), 4),
            "codonified_total_conflict_splits": c_tot_conf,
            "walrus_total_conflict_splits": w_tot_conf,
            "conflict_splits_reduction_pct": round(100.0 * (1.0 - w_tot_conf / max(1, c_tot_conf)), 2),
            "codonified_total_shared_sp_splits": c_tot_shared,
            "walrus_total_shared_sp_splits": w_tot_shared,
            "codonified_interval_shunts": c_tot_int,
            "walrus_interval_shunts": w_tot_int,
            "codonified_sparse_shunts": c_tot_sp,
            "walrus_sparse_shunts": w_tot_sp,
            "exact_n_mean_mdl_delta_nats": (
                round(st.mean(r["tree_MDL_nats"] - r["walrus_comparison"]["tree_MDL_nats"] for r in w_exact_n), 2)
                if w_exact_n else None
            ),
        }

    report = {
        "total_genes_in_manifest": 19953,
        "total_genes_processed": len(seen_genes),
        "total_genes_valid": n_ok,
        "total_errors": len(errors),
        "alignment_dimensions": {
            "median_n_taxa": st.median(n_taxa),
            "mean_n_taxa": round(st.mean(n_taxa), 1),
            "p10_n_taxa": pct(n_taxa, 0.10),
            "p90_n_taxa": pct(n_taxa, 0.90),
            "max_n_taxa": max(n_taxa),
            "median_seq_len_bp": st.median(seq_len),
            "mean_seq_len_bp": round(st.mean(seq_len), 1),
            "p10_seq_len_bp": pct(seq_len, 0.10),
            "p90_seq_len_bp": pct(seq_len, 0.90),
            "max_seq_len_bp": max(seq_len),
        },
        "compute_performance": {
            "median_runtime_sec": round(st.median(runtimes), 3),
            "mean_runtime_sec": round(st.mean(runtimes), 3),
            "p90_runtime_sec": round(pct(runtimes, 0.90), 3),
            "total_cpu_hours_denovo": round(sum(runtimes) / 3600.0, 2),
        },
        "species_tree_concordance": {
            "n_evaluated_genes": len(valid_sp),
            "total_species_tree_splits_tested": tot_sp_splits,
            "total_species_tree_splits_mdl_supported": tot_sp_sup,
            "overall_sp_split_support_pct": round(100.0 * tot_sp_sup / max(1, tot_sp_splits), 2),
            "median_sp_split_support_pct": round(100.0 * st.median(sp_sup_frac), 2),
            "mean_sp_split_support_pct": round(100.0 * st.mean(sp_sup_frac), 2),
            "total_binary_gene_tree_splits": tot_bin_splits,
            "total_mdl_calibrated_gene_tree_splits": tot_mdl_splits,
            "overall_polytomy_contraction_pct": round(100.0 * (1.0 - tot_mdl_splits / max(1, tot_bin_splits)), 2),
            "median_polytomy_contraction_pct": round(100.0 * st.median(poly_frac), 2),
            "total_binary_conflicting_splits": tot_bin_conf,
            "total_mdl_conflicting_splits": tot_mdl_conf,
            "total_false_conflicts_eliminated": tot_bin_conf - tot_mdl_conf,
            "overall_false_conflict_eliminated_pct": round(100.0 * (1.0 - tot_mdl_conf / max(1, tot_bin_conf)), 2),
            "median_false_conflict_eliminated_pct": round(100.0 * st.median(false_elim), 2),
            "mean_binary_conflict_rate_pct": round(100.0 * st.mean(bin_conf_rate), 2),
            "mean_mdl_conflict_rate_pct": round(100.0 * st.mean(mdl_conf_rate), 2),
            "total_binary_shared_sp_splits": tot_bin_shared,
            "total_mdl_shared_sp_splits": tot_mdl_shared,
            "retained_true_sp_splits_pct": round(100.0 * tot_mdl_shared / max(1, tot_bin_shared), 2),
            "median_delta_mdl_sp_minus_gene_nats": round(st.median(d_mdl), 2),
            "genes_where_species_tree_compresses_better_or_equal": sum(1 for x in d_mdl if x <= 0),
            "genes_where_species_tree_compresses_better_or_equal_pct": round(
                100.0 * sum(1 for x in d_mdl if x <= 0) / max(1, len(d_mdl)), 2
            ),
        },
        "reticulate_shunt_atlas": {
            "genes_with_any_shunt": len(genes_with_shunts),
            "genes_with_any_shunt_pct": round(100.0 * len(genes_with_shunts) / n_ok, 2),
            "genes_with_interval_shunt": len(genes_with_int),
            "genes_with_interval_shunt_pct": round(100.0 * len(genes_with_int) / n_ok, 2),
            "total_interval_shunts": tot_int_events,
            "genes_with_sparse_shunt": len(genes_with_sparse),
            "genes_with_sparse_shunt_pct": round(100.0 * len(genes_with_sparse) / n_ok, 2),
            "total_sparse_shunts": tot_sparse_events,
            "total_shunts_on_fixed_species_tree": tot_sp_tree_shunts,
            "top_50_interval_shunts": all_interval[:50],
            "top_50_sparse_shunts": all_sparse[:50],
            "top_50_species_tree_shunts": all_sp_tree_shunts[:50],
        },
        "walrus_vs_codonified_comparison": walrus_summary,
    }

    with open(REPORT_JSON, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({k: v for k, v in report.items() if k != "reticulate_shunt_atlas"}, indent=2))
    print("\n--- Shunt Atlas Summary ---")
    for k, v in report["reticulate_shunt_atlas"].items():
        if not isinstance(v, list):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
