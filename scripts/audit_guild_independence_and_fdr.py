#!/usr/bin/env python3
"""Audit TOGA2 guild family independence, cochlear DNA vs. AA reconciliation,
and compute study-wide Benjamini-Hochberg FDR q-values directly from executed
benchmark JSON ledgers.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env_paths import RESULTS_DIR


def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0.0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))


def poisson_upper_tail(k_obs: int, lam: float) -> float:
    """Exact P(X >= k_obs) for X ~ Poisson(lam)."""
    if k_obs <= 0:
        return 1.0
    cdf_below = sum(poisson_pmf(k, lam) for k in range(k_obs))
    return max(0.0, min(1.0, 1.0 - cdf_below))


def poisson_two_sided(k_obs: int, lam: float) -> float:
    """Exact two-sided Poisson test summing all outcomes with P(X=k) <= P(X=k_obs)."""
    p_obs = poisson_pmf(k_obs, lam)
    tol = p_obs * (1.0 + 1e-12)
    upper_bound = max(k_obs + 100, int(lam + 20.0 * math.sqrt(max(lam, 1.0)) + 50))
    total = 0.0
    for k in range(upper_bound):
        pk = poisson_pmf(k, lam)
        if pk <= tol:
            total += pk
    return max(0.0, min(1.0, total))


def gene_family_root(symbol: str) -> str:
    """Conservative paralog / multi-gene family grouping root for independence audit."""
    s = symbol.upper().strip()
    if re.match(r"^OR\d+", s):
        return "OR_olfactory_receptor_family"
    if s.startswith("PCDH"):
        return "PCDH_protocadherin_family"
    if re.match(r"^ZNF\d+|^ZSCAN\d+", s):
        return "ZNF_ZSCAN_zinc_finger_family"
    if re.match(r"^SSX\d+", s):
        return "SSX_cancer_testis_family"
    if s.startswith("MAGE"):
        return "MAGE_family"
    if s.startswith("HLA-"):
        return "HLA_MHC_family"
    if s.startswith("SPATA31"):
        return "SPATA31_family"
    return s


def bh_fdr(items: list[dict]) -> list[dict]:
    """Attach Benjamini-Hochberg step-up adjusted q_value to test records."""
    m = len(items)
    indexed = sorted(enumerate(items), key=lambda x: x[1]["p_value"])
    q_sorted = [1.0] * m
    running_min = 1.0
    for rank_0 in range(m - 1, -1, -1):
        rank = rank_0 + 1
        p = indexed[rank_0][1]["p_value"]
        running_min = min(running_min, min(1.0, m * p / rank))
        q_sorted[rank_0] = running_min
    out = [dict(rec) for rec in items]
    for rank_0, (orig_idx, _) in enumerate(indexed):
        out[orig_idx]["bh_rank"] = rank_0 + 1
        out[orig_idx]["bh_total_tests"] = m
        out[orig_idx]["q_value_bh"] = q_sorted[rank_0]
    return out


def main() -> None:
    protein_atlas = json.loads((RESULTS_DIR / "mammalian_toga2" / "toga2_protein_atlas.json").read_text())
    deep_atlas = json.loads((RESULTS_DIR / "mammalian_toga2" / "toga2_deep_mining_atlas.json").read_text())
    cochlear_aa = json.loads((RESULTS_DIR / "mammalian_toga2" / "toga2_cochlear_25genes.json").read_text())
    pillar3_bcr = json.loads((RESULTS_DIR / "single_cell" / "pillar3_bcr_cellular.json").read_text())
    cancer = json.loads((RESULTS_DIR / "single_cell" / "cancer_singlecell_summary.json").read_text())

    # 1. TOGA2 Ecological Guild Paralog-Family & Lineage-Pair Independence Audit
    guild_audit = {}
    for gname, gdata in protein_atlas["ecological_guild_enrichment"].items():
        obs = int(gdata["observed_inter_order_sparse_shunts"])
        exp = float(gdata["expected_random_inter_order"])
        top_hits = gdata.get("top_genes", [])
        distinct_symbols = sorted({h["symbol"] for h in top_hits})
        distinct_families = sorted({gene_family_root(h["symbol"]) for h in top_hits})
        distinct_taxon_family_pairs = sorted({
            tuple(sorted((h.get("u_family", ""), h.get("v_family", ""))))
            for h in top_hits
        })
        p_raw_upper = poisson_upper_tail(obs, exp)
        p_raw_two = poisson_two_sided(obs, exp)

        # Conservative deduplicated test treating multi-copy gene families as a single locus
        dedup_k = len(distinct_families)
        p_dedup_upper = poisson_upper_tail(dedup_k, exp)
        p_dedup_two = poisson_two_sided(dedup_k, exp)

        guild_audit[gname] = {
            "observed_inter_order_sparse_shunts": obs,
            "expected_random_inter_order": round(exp, 4),
            "fold_enrichment_raw": round(obs / exp, 3) if exp > 0 else None,
            "poisson_p_raw_upper": p_raw_upper,
            "poisson_p_raw_two_sided": p_raw_two,
            "top_hits_enumerated": len(top_hits),
            "distinct_symbols_in_top_hits": len(distinct_symbols),
            "distinct_gene_families_in_top_hits": len(distinct_families),
            "distinct_gene_family_list": distinct_families,
            "distinct_mammal_family_pairs_in_top_hits": len(distinct_taxon_family_pairs),
            "conservative_family_dedup_fold": round(dedup_k / exp, 3) if exp > 0 else None,
            "poisson_p_family_dedup_upper": p_dedup_upper,
            "poisson_p_family_dedup_two_sided": p_dedup_two,
            "top_hit_symbols": [h["symbol"] for h in top_hits],
        }

    # 2. 25-Gene Cochlear Mechanotransduction DNA vs. AA Reconciliation
    dna_coch = {x["gene_symbol"]: x for x in deep_atlas["cochlear_auditory_complex_25_genes"]}
    aa_coch = {x["symbol"]: x for x in cochlear_aa}
    reconciled_genes = []
    for sym in sorted(set(dna_coch) | set(aa_coch)):
        d = dna_coch.get(sym, {})
        a = aa_coch.get(sym, {})
        reconciled_genes.append({
            "symbol": sym,
            "dna_n_sparse": d.get("n_sparse_shunts", 0),
            "dna_n_interval": d.get("n_interval_shunts", 0),
            "dna_max_gain_nats": d.get("max_shunt_gain_nats", 0.0),
            "aa_sp_n_sparse": a.get("n_sp_sparse", 0),
            "aa_sp_n_interval": a.get("n_sp_interval", 0),
            "aa_sp_max_gain_nats": a.get("max_sp_shunt_gain_nats", 0.0),
            "aa_free_shunts": a.get("n_free_shunts", 0),
        })

    dna_free_set = {x["gene_symbol"] for x in deep_atlas["cochlear_auditory_complex_25_genes"] if (x["n_sparse_shunts"] + x["n_interval_shunts"]) > 0}
    dna_sp_set = {x["gene_symbol"] for x in deep_atlas["cochlear_auditory_complex_25_genes"] if x["sp_shunts_count"] > 0}
    aa_sp_set = {x["symbol"] for x in cochlear_aa if x["n_sp_shunts"] > 0}
    cochlear_summary = {
        "n_genes_total": len(cochlear_aa),
        "dna_free_tree_shunted_genes_count": len(dna_free_set),
        "dna_species_tree_shunted_genes_count": len(dna_sp_set),
        "dna_free_or_species_union_count": len(dna_free_set | dna_sp_set),
        "aa_species_tree_shunted_genes_count": len(aa_sp_set),
        "aa_or_dna_species_tree_union_count": len(aa_sp_set | dna_sp_set),
        "aa_species_tree_interval_genes_count": sum(1 for x in cochlear_aa if x["n_sp_interval"] > 0),
        "aa_species_tree_sparse_genes_count": sum(1 for x in cochlear_aa if x["n_sp_sparse"] > 0),
        "aa_free_tree_shunted_genes_count": sum(1 for x in cochlear_aa if x["n_free_shunts"] > 0),
        "loxhd1_aa_sp": {
            "n_interval": aa_coch["LOXHD1"]["n_sp_interval"],
            "n_sparse": aa_coch["LOXHD1"]["n_sp_sparse"],
            "max_gain_nats": aa_coch["LOXHD1"]["max_sp_shunt_gain_nats"],
        },
        "ush2a_aa_sp": {
            "n_interval": aa_coch["USH2A"]["n_sp_interval"],
            "n_sparse": aa_coch["USH2A"]["n_sp_sparse"],
            "max_gain_nats": aa_coch["USH2A"]["max_sp_shunt_gain_nats"],
        },
        "tecta_aa_sp": {
            "n_interval": aa_coch["TECTA"]["n_sp_interval"],
            "n_sparse": aa_coch["TECTA"]["n_sp_sparse"],
            "max_gain_nats": aa_coch["TECTA"]["max_sp_shunt_gain_nats"],
        },
        "adgrv1_aa_sp": {
            "n_interval": aa_coch["ADGRV1"]["n_sp_interval"],
            "n_sparse": aa_coch["ADGRV1"]["n_sp_sparse"],
            "max_gain_nats": aa_coch["ADGRV1"]["max_sp_shunt_gain_nats"],
        },
        "genes": reconciled_genes,
    }

    toga_audit_out = {
        "audit": "TOGA2_882_guild_family_independence_and_cochlear_reconciliation",
        "guild_enrichment_audit": guild_audit,
        "cochlear_25_gene_reconciliation": cochlear_summary,
    }
    (RESULTS_DIR / "mammalian_toga2" / "toga2_guild_independence_audit.json").write_text(
        json.dumps(toga_audit_out, indent=2)
    )

    # 3. Study-Wide Benjamini-Hochberg Multiple-Testing FDR Ledger
    disc = pillar3_bcr["part3b_empirical_gcreplay"]["zero_leakage_biological_discovery"]
    ab = disc["in_vivo_clone_abundance"]
    aff = disc["experimental_dms_binding_affinity"]
    par = pillar3_bcr["part3b_empirical_gcreplay"]["cross_mouse_replay_shunts"]["parallel_star_hub_replay"]
    aml = cancer["aml_singlecell_tapestri"]
    rc = aml["raw_singlecell_compass_zygosity"]
    wp = aml["within_patient_123_clone_expansion"]
    sig = aml["signaling_vs_founder_subclonal_enrichment"]
    dlp = cancer["dlp_singlecell_cna"]

    raw_tests = [
        {"test_id": "bcr_gc_paired_hub_abundance", "domain": "Single-cell BCR (147 GCs)", "unit_n": ab["hierarchical_gc_paired_diff_hub_minus_nonhub"]["n"], "two_sided": True, "p_value": ab["hierarchical_gc_paired_diff_hub_minus_nonhub"]["wilcoxon_p_two_sided"]},
        {"test_id": "bcr_gc_spearman_degree_abundance", "domain": "Single-cell BCR (147 GCs)", "unit_n": ab["hierarchical_gc_spearman_rho"]["n"], "two_sided": True, "p_value": ab["hierarchical_gc_spearman_rho"]["wilcoxon_p_two_sided"]},
        {"test_id": "bcr_mouse_paired_hub_abundance", "domain": "Single-cell BCR (mice)", "unit_n": ab["hierarchical_mouse_paired_diff_hub_minus_nonhub"]["n"], "two_sided": True, "p_value": ab["hierarchical_mouse_paired_diff_hub_minus_nonhub"]["wilcoxon_p_two_sided"]},
        {"test_id": "bcr_mouse_spearman_degree_abundance", "domain": "Single-cell BCR (mice)", "unit_n": ab["hierarchical_mouse_spearman_rho"]["n"], "two_sided": True, "p_value": ab["hierarchical_mouse_spearman_rho"]["wilcoxon_p_two_sided"]},
        {"test_id": "bcr_gc_spearman_degree_dms_affinity", "domain": "Single-cell BCR DMS", "unit_n": aff["hierarchical_gc_spearman_rho"]["n"], "two_sided": True, "p_value": aff["hierarchical_gc_spearman_rho"]["wilcoxon_p_two_sided"]},
        {"test_id": "bcr_mouse_spearman_degree_dms_affinity", "domain": "Single-cell BCR DMS", "unit_n": aff["hierarchical_mouse_spearman_rho"]["n"], "two_sided": True, "p_value": aff["hierarchical_mouse_spearman_rho"]["wilcoxon_p_two_sided"]},
        {"test_id": "bcr_hub_vs_nonhub_binary_dms_affinity", "domain": "Single-cell BCR DMS", "unit_n": disc["n_evaluated_unique_genotypes"], "two_sided": True, "p_value": aff["mwu_pvalue_hub_vs_nonhub"]},
        {"test_id": "bcr_parallel_star_m5_cdr_enrichment", "domain": "Cross-mouse replay (>=5 mice)", "unit_n": par["min_mice_5"]["n_parallel_aa_variants"] + par["min_mice_5"]["n_other_aa_variants"], "two_sided": True, "p_value": par["min_mice_5"]["cdr_enrichment_fisher_p_two_sided"]},
        {"test_id": "bcr_parallel_star_m5_dms_affinity", "domain": "Cross-mouse replay (>=5 mice)", "unit_n": par["min_mice_5"]["n_parallel_aa_variants"] + par["min_mice_5"]["n_other_aa_variants"], "two_sided": True, "p_value": par["min_mice_5"]["mwu_pvalue_delta_bind"]},
        {"test_id": "bcr_parallel_star_m3_cdr_enrichment", "domain": "Cross-mouse replay (>=3 mice)", "unit_n": par["min_mice_3"]["n_parallel_aa_variants"] + par["min_mice_3"]["n_other_aa_variants"], "two_sided": True, "p_value": par["min_mice_3"]["cdr_enrichment_fisher_p_two_sided"]},
        {"test_id": "bcr_parallel_star_m3_dms_affinity", "domain": "Cross-mouse replay (>=3 mice)", "unit_n": par["min_mice_3"]["n_parallel_aa_variants"] + par["min_mice_3"]["n_other_aa_variants"], "two_sided": True, "p_value": par["min_mice_3"]["mwu_pvalue_delta_bind"]},
        {"test_id": "aml_compass_per_run_hub_diff_cells", "domain": "AML scDNA COMPASS (168 runs)", "unit_n": rc["hierarchical_per_run_diff_cells"]["n"], "two_sided": True, "p_value": rc["hierarchical_per_run_diff_cells"]["wilcoxon_p_two_sided"]},
        {"test_id": "aml_compass_per_patient_hub_diff_cells", "domain": "AML scDNA COMPASS (129 pts)", "unit_n": rc["hierarchical_per_patient_diff_cells"]["n"], "two_sided": True, "p_value": rc["hierarchical_per_patient_diff_cells"]["wilcoxon_p_two_sided"]},
        {"test_id": "aml_morita_direct_cell_degree_null", "domain": "AML Morita (123 pts)", "unit_n": aml["n_curated_nonroot_clones"], "two_sided": True, "p_value": wp["k_out_vs_direct_cells_p_value"]},
        {"test_id": "aml_morita_per_patient_paired_lineage", "domain": "AML Morita (123 pts)", "unit_n": wp["per_patient_paired_n_patients"], "two_sided": True, "p_value": wp["per_patient_paired_wilcoxon_lineage_cells_p_two_sided"]},
        {"test_id": "aml_morita_per_patient_lineage_spearman", "domain": "AML Morita (123 pts)", "unit_n": wp["per_patient_paired_n_patients"], "two_sided": True, "p_value": wp["per_patient_spearman_rho_wilcoxon_p_two_sided"]},
        {"test_id": "aml_morita_signaling_vs_founder_subclonal", "domain": "AML Morita (123 pts)", "unit_n": sig["signaling_subclonal_hits"] + sig["signaling_trunk_hits"] + sig["founder_subclonal_hits"] + sig["founder_trunk_hits"], "two_sided": True, "p_value": sig["fisher_p_value_two_sided"]},
        {"test_id": "dlp_ov2295_aneuploidy_vs_progression", "domain": "DLP+ CNA (OV2295)", "unit_n": dlp["OV2295"]["n_cells"], "two_sided": True, "p_value": dlp["OV2295"]["aneuploidy_vs_progression_p_value"]},
        {"test_id": "dlp_sa609_aneuploidy_vs_progression", "domain": "DLP+ CNA (SA609)", "unit_n": dlp["SA609"]["n_cells"], "two_sided": True, "p_value": dlp["SA609"]["aneuploidy_vs_progression_p_value"]},
        {"test_id": "dlp_sa501_aneuploidy_vs_progression", "domain": "DLP+ CNA (SA501)", "unit_n": dlp["SA501"]["n_cells"], "two_sided": True, "p_value": dlp["SA501"]["aneuploidy_vs_progression_p_value"]},
        {"test_id": "dlp_sa535_aneuploidy_vs_progression", "domain": "DLP+ CNA (SA535)", "unit_n": dlp["SA535"]["n_cells"], "two_sided": True, "p_value": dlp["SA535"]["aneuploidy_vs_progression_p_value"]},
        {"test_id": "toga2_myrmecophagous_family_dedup", "domain": "TOGA2 882-mammal AA atlas", "unit_n": guild_audit["myrmecophagous_insectivorous_armor"]["distinct_gene_families_in_top_hits"], "two_sided": True, "p_value": guild_audit["myrmecophagous_insectivorous_armor"]["poisson_p_family_dedup_two_sided"]},
        {"test_id": "toga2_subterranean_blind_family_dedup", "domain": "TOGA2 882-mammal AA atlas", "unit_n": guild_audit["subterranean_blind"]["distinct_gene_families_in_top_hits"], "two_sided": True, "p_value": guild_audit["subterranean_blind"]["poisson_p_family_dedup_two_sided"]},
    ]

    adjusted = bh_fdr(raw_tests)
    ledger = {
        "method": f"Benjamini-Hochberg step-up FDR across all {len(adjusted)} primary two-sided hypothesis tests",
        "alpha_fdr": 0.05,
        "n_tests": len(adjusted),
        "n_significant_fdr_05": sum(1 for r in adjusted if r["q_value_bh"] < 0.05),
        "tests": adjusted,
    }
    (RESULTS_DIR / "statistical_fdr_ledger.json").write_text(json.dumps(ledger, indent=2))

    print("Saved toga2_guild_independence_audit.json and statistical_fdr_ledger.json")
    for r in adjusted:
        print(f"  [{r['bh_rank']:2d}] {r['test_id']:42s} p={r['p_value']:.3e}  q_BH={r['q_value_bh']:.3e}")


if __name__ == "__main__":
    main()
