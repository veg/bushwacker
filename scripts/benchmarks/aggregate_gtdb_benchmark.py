#!/usr/bin/env python3
"""
Aggregate GTDB R95 242-Marker Benchmark Results (ar122 + bac120)
================================================================
Aggregates all per-marker JSON reports produced by `run_gtdb_benchmark.py`
across Domain Archaea (122 markers) and Domain Bacteria (120 markers),
computing:
  1. Domain-wide gene-tree accuracy, precision, false-positive suppression,
     RF / nRF / soft-RF, and runtime speedup vs. FastTree 2.
  2. Taxonomic clade monophyly and Four-Gamete compatibility across all 5
     GTDB ranks (Phylum, Class, Order, Family, Genus).
  3. Complexity Hypothesis stratification across functional categories
     (Ribosomal & Translational, Transcription & DNA Replication,
     Aminoacyl-tRNA & tRNA Mod, Metabolic & Cellular Operational).
  4. Certified Horizontal Gene Transfer (INTERVAL) and Parallel/Convergent
     (SPARSE) shunt network statistics stratified by taxonomic bridge rank,
     plus the top certified inter-phylum / inter-class / inter-order HGT highways.
"""


import sys as _sys
from pathlib import Path as _Path
if str(_Path(__file__).resolve().parent) not in _sys.path:
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
from _env_paths import REPO_ROOT, DATA_DIR, SPECIMENS_DIR, ALIGNMENTS_DIR, SINGLE_CELL_DIR, RESULTS_DIR, WORK_ROOT, EXTERNAL_DIR, resolve_bin, ASTRAL_JAR

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

GTDB_DIR = EXTERNAL_DIR / "gtdb_r95"
RES_DIR = GTDB_DIR / "results"


def mean_sd(vals):
    if not vals:
        return 0.0, 0.0
    a = np.array(vals, dtype=float)
    return float(np.mean(a)), float(np.std(a))


def main():
    files = sorted(RES_DIR.glob("*.json"))
    records = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                records.append(json.load(f))
        except Exception as e:
            print(f"Warning: could not read {fp}: {e}")

    ar_recs = [r for r in records if r["domain"] == "ar122"]
    bac_recs = [r for r in records if r["domain"] == "bac120"]

    print("=" * 95)
    print(f"GTDB R95 MICROBIAL TREE OF LIFE BENCHMARK SUMMARY (Total={len(records)}: ar122={len(ar_recs)}, bac120={len(bac_recs)})")
    print("=" * 95)

    for dom_label, subset in [
        ("ARCHAEA (ar122, up to 1,672 species across 19 phyla)", ar_recs),
        ("BACTERIA (bac120, up to 1,000 species across 111 phyla)", bac_recs),
        ("ALL PROKARYOTIC MARKERS COMBINED (242 universal marker genes)", records),
    ]:
        if not subset:
            continue
        n = len(subset)
        print(f"\n--- {dom_label} (n = {n} markers) ---")
        mean_n = np.mean([r["n_taxa"] for r in subset])
        mean_l_raw = np.mean([r["l_raw_bp"] for r in subset])
        mean_l_trim = np.mean([r["l_trim_bp"] for r in subset])
        mean_rho = np.mean([r["rho_compression"] for r in subset])
        mean_bw_t = np.mean([r["bw_runtime_sec"] for r in subset])
        mean_ft_t = np.mean([r["ft_wall_sec"] for r in subset])
        tot_bw_t = np.sum([r["bw_runtime_sec"] for r in subset])
        tot_ft_t = np.sum([r["ft_wall_sec"] for r in subset])
        print(
            f"  Mean Taxa N = {mean_n:.1f} | Mean Codon MSA L_raw = {mean_l_raw:.1f} bp, L_trimmed = {mean_l_trim:.1f} bp | "
            f"Mean Compression rho = {100*mean_rho:.2f}%"
        )
        print(
            f"  Runtime: Bushwhacker mean = {mean_bw_t:.2f} s (total = {tot_bw_t:.1f} s) vs. "
            f"FastTree 2 mean = {mean_ft_t:.2f} s (total = {tot_ft_t:.1f} s) -> {tot_ft_t/max(1e-6, tot_bw_t):.2f}x speedup"
        )

        print(f"  {'Method':<22} | {'Res%':>6} | {'TP':>6} | {'FP':>6} | {'Prec%':>6} | {'Rec%':>6} | {'RF':>7} | {'nRF':>6} | {'RF_sp':>7} | {'nRF_sp':>6}")
        print("  " + "-" * 91)
        for m_key, m_name in [
            ("bw_mdl", "Bushwhacker (MDL)"),
            ("bw_bin", "Bushwhacker (Binary)"),
            ("fasttree", "FastTree 2 (GTR+G)"),
        ]:
            res_rate = 100.0 * np.mean([r[m_key]["resolution_rate"] for r in subset])
            tp = np.mean([r[m_key]["tp"] for r in subset])
            fp = np.mean([r[m_key]["fp"] for r in subset])
            prec = 100.0 * np.mean([r[m_key]["precision"] for r in subset])
            rec = 100.0 * np.mean([r[m_key]["recall"] for r in subset])
            rf = np.mean([r[m_key]["rf"] for r in subset])
            nrf = np.mean([r[m_key]["nrf"] for r in subset])
            rf_sp = np.mean([r[m_key]["rf_sp"] for r in subset])
            nrf_sp = np.mean([r[m_key]["nrf_sp"] for r in subset])
            print(
                f"  {m_name:<22} | {res_rate:5.1f}% | {tp:6.1f} | {fp:6.1f} | {prec:5.1f}% | {rec:5.1f}% | "
                f"{rf:7.1f} | {nrf:6.4f} | {rf_sp:7.1f} | {nrf_sp:6.4f}"
            )

        # Taxonomic Rank Monophyly & Compatibility
        print("\n  Taxonomic Clade Monophyly (Strict Recovery % vs. Four-Gamete Non-Conflict Compatibility %):")
        print(f"  {'Rank':<10} | {'Clades/Gene':>11} | {'BW(MDL) Rec%':>12} | {'BW(MDL) Compat%':>15} | {'BW(Bin) Rec%':>12} | {'FastTree Rec%':>13} | {'FastTree Compat%':>16}")
        print("  " + "-" * 102)
        for rk in ["Phylum", "Class", "Order", "Family", "Genus"]:
            tot_c = np.mean([r["bw_mdl"]["rank_stats"][rk]["total"] for r in subset])
            bw_rec = 100.0 * np.sum([r["bw_mdl"]["rank_stats"][rk]["recovered"] for r in subset]) / max(1, np.sum([r["bw_mdl"]["rank_stats"][rk]["total"] for r in subset]))
            bw_cmp = 100.0 * np.sum([r["bw_mdl"]["rank_stats"][rk]["compatible"] for r in subset]) / max(1, np.sum([r["bw_mdl"]["rank_stats"][rk]["total"] for r in subset]))
            bb_rec = 100.0 * np.sum([r["bw_bin"]["rank_stats"][rk]["recovered"] for r in subset]) / max(1, np.sum([r["bw_bin"]["rank_stats"][rk]["total"] for r in subset]))
            ft_rec = 100.0 * np.sum([r["fasttree"]["rank_stats"][rk]["recovered"] for r in subset]) / max(1, np.sum([r["fasttree"]["rank_stats"][rk]["total"] for r in subset]))
            ft_cmp = 100.0 * np.sum([r["fasttree"]["rank_stats"][rk]["compatible"] for r in subset]) / max(1, np.sum([r["fasttree"]["rank_stats"][rk]["total"] for r in subset]))
            print(
                f"  {rk:<10} | {tot_c:11.1f} | {bw_rec:11.2f}% | {bw_cmp:14.2f}% | {bb_rec:11.2f}% | {ft_rec:12.2f}% | {ft_cmp:15.2f}%"
            )

    # Functional Class Stratification (Complexity Hypothesis)
    print("\n" + "=" * 95)
    print("FUNCTIONAL CLASS STRATIFICATION & THE COMPLEXITY HYPOTHESIS (Jain, Rivera, & Lake 1999)")
    print("=" * 95)
    by_func = defaultdict(list)
    for r in records:
        by_func[r["func_class"]].append(r)

    print(
        f"{'Functional Category':<34} | {'Genes':>5} | {'Mean L':>6} | {'Rho%':>6} | {'BW Prec%':>8} | "
        f"{'FT Prec%':>8} | {'Shunts/Gene':>11} | {'INTERVAL/G':>10} | {'SPARSE/G':>8}"
    )
    print("-" * 116)
    for fc in [
        "Ribosomal & Translational",
        "Transcription & DNA Replication",
        "Aminoacyl-tRNA & tRNA Mod",
        "Metabolic & Cellular Operational",
    ]:
        sub = by_func.get(fc, [])
        if not sub:
            continue
        ml = np.mean([r["l_trim_bp"] for r in sub])
        rho = 100.0 * np.mean([r["rho_compression"] for r in sub])
        bwp = 100.0 * np.mean([r["bw_mdl"]["precision"] for r in sub])
        ftp = 100.0 * np.mean([r["fasttree"]["precision"] for r in sub])
        sh_g = np.mean([r["n_shunts"] for r in sub])
        int_g = np.mean([r["n_interval_shunts"] for r in sub])
        sp_g = np.mean([r["n_sparse_shunts"] for r in sub])
        print(
            f"{fc:<34} | {len(sub):5d} | {ml:6.1f} | {rho:5.1f}% | {bwp:7.2f}% | "
            f"{ftp:7.2f}% | {sh_g:11.2f} | {int_g:10.2f} | {sp_g:8.2f}"
        )

    # Reticulate Shunt & HGT Network Summary
    print("\n" + "=" * 95)
    print("MDL-CERTIFIED RETICULATE SHUNT & HORIZONTAL GENE TRANSFER ATLAS")
    print("=" * 95)
    all_shunts = []
    for r in records:
        for s in r["shunts"]:
            s_copy = dict(s)
            s_copy["domain"] = r["domain"]
            s_copy["marker_id"] = r["marker_id"]
            s_copy["gene_name"] = r["name"]
            s_copy["func_class"] = r["func_class"]
            all_shunts.append(s_copy)

    print(f"Total MDL-Certified Shunts across {len(records)} markers: {len(all_shunts)}")
    mode_counts = Counter(s["mode"] for s in all_shunts)
    bridge_counts = Counter(s["bridge_level"] for s in all_shunts)
    bridge_by_mode = defaultdict(Counter)
    for s in all_shunts:
        bridge_by_mode[s["mode"]][s["bridge_level"]] += 1

    print(f"  By Mode: INTERVAL (contiguous HGT/recombination) = {mode_counts['INTERVAL']}, SPARSE (dispersed convergence) = {mode_counts['SPARSE']}")
    print("  By Taxonomic Bridge Level:")
    for bl in ["Inter-Phylum", "Inter-Class", "Inter-Order", "Inter-Family", "Inter-Genus", "Intra-Genus", "Unknown"]:
        if bridge_counts[bl] > 0:
            print(
                f"    {bl:<15}: {bridge_counts[bl]:5d} total "
                f"(INTERVAL = {bridge_by_mode['INTERVAL'][bl]:4d}, SPARSE = {bridge_by_mode['SPARSE'][bl]:4d})"
            )

    # Top 20 highest-gain INTERVAL HGT shunts
    interval_shunts = sorted(
        [s for s in all_shunts if s["mode"] == "INTERVAL"],
        key=lambda x: x["net_mdl_gain_nats"],
        reverse=True,
    )
    print("\nTop 20 Highest-MDL-Gain Certified INTERVAL (Horizontal Gene Transfer) Shunts:")
    for i, s in enumerate(interval_shunts[:20], 1):
        print(
            f"  {i:2d}. [{s['domain']} {s['marker_id']} {s['gene_name']} ({s['func_class']})]: "
            f"+{s['net_mdl_gain_nats']:.1f} nats (m={s['shared_mutations']} sites, span={s['interval_span']}, P={s['posterior_prob']:.4f}) | "
            f"{s['bridge_level']}: {s['u_summary']} <-> {s['v_summary']}"
        )

    # Top 15 highest-gain SPARSE shunts
    sparse_shunts = sorted(
        [s for s in all_shunts if s["mode"] == "SPARSE"],
        key=lambda x: x["net_mdl_gain_nats"],
        reverse=True,
    )
    print("\nTop 15 Highest-MDL-Gain Certified SPARSE (Parallel/Convergent) Shunts:")
    for i, s in enumerate(sparse_shunts[:15], 1):
        print(
            f"  {i:2d}. [{s['domain']} {s['marker_id']} {s['gene_name']} ({s['func_class']})]: "
            f"+{s['net_mdl_gain_nats']:.1f} nats (m={s['shared_mutations']} sites, P={s['posterior_prob']:.4f}) | "
            f"{s['bridge_level']}: {s['u_summary']} <-> {s['v_summary']}"
        )


if __name__ == "__main__":
    main()
