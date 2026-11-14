#!/usr/bin/env python3
"""
Empirical Phylogenomic Benchmark on Zhou, Shen, Hittinger, & Rokas (2018, MBE):
"Evaluating Fast Maximum Likelihood-Based Phylogenetic Programs Using Empirical Phylogenomic Data Sets"
(Figshare Project 22040).

Evaluates Bushwhacker (MDL) and Bushwhacker (Binary) against:
  - 10 independent RAxML runs per gene (5 MP + 5 Random starting trees)
  - 10 independent IQ-TREE runs per gene
  - 10 independent PhyML runs per gene
  - FastTree 2
  - Best_observed ML tree (across all 40 ML searches per gene)
  - Coalescent (ASTRAL) and Concatenation Reference Species Trees

Across all 6 empirical nucleotide phylogenomic datasets (4,159 genes total):
  - SongD1  (Mammals, Song et al. 2012 PNAS, 37 taxa, 424 genes)
  - XiD4    (Seed Plants, Xi et al. 2014 PNAS, 46 taxa, 310 genes)
  - PrumD6  (Birds, Prum et al. 2015 Nature, 200 taxa, 259 genes)
  - WickD3a (Land Plants 1st+2nd codon pos, Wickett et al. 2014 PNAS, 103 taxa, 844 genes)
  - WickD3b (Land Plants all codon pos, Wickett et al. 2014 PNAS, 103 taxa, 844 genes)
  - MisoD2a (Insects 1st+2nd codon pos, Misof et al. 2014 Science, 144 taxa, 1,478 genes)
"""


import sys as _sys
from pathlib import Path as _Path
if str(_Path(__file__).resolve().parent) not in _sys.path:
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
from _env_paths import REPO_ROOT, DATA_DIR, SPECIMENS_DIR, ALIGNMENTS_DIR, SINGLE_CELL_DIR, RESULTS_DIR, WORK_ROOT, EXTERNAL_DIR, resolve_bin, ASTRAL_JAR

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import StringIO
from pathlib import Path
from Bio import Phylo


DATASETS = ["SongD1", "XiD4", "PrumD6", "WickD3a", "WickD3b", "MisoD2a"]


def extract_Induced_splits(newick_str_or_path, is_path=True, allowed_leaves=None, L=None):
    """
    Extract unrooted non-trivial splits induced on `allowed_leaves` (if given),
    along with branch lengths.
    """
    try:
        txt = Path(newick_str_or_path).read_text().strip() if is_path else newick_str_or_path.strip()
        if not txt:
            return None
        tree = Phylo.read(StringIO(txt), "newick")
    except Exception:
        return None

    terminals = [t.name for t in tree.get_terminals()]
    all_leaves = frozenset(terminals) if allowed_leaves is None else (frozenset(terminals) & allowed_leaves)
    if len(all_leaves) < 4:
        return None

    clade_leaves = {}
    for clade in tree.find_clades(order="postorder"):
        if clade.is_terminal():
            clade_leaves[id(clade)] = frozenset([clade.name]) & all_leaves
        else:
            s = frozenset().union(*(clade_leaves[id(c)] for c in clade.clades))
            clade_leaves[id(clade)] = s

    splits = {}
    for clade in tree.find_clades(order="postorder"):
        if clade == tree.root or clade.is_terminal():
            continue
        s = clade_leaves[id(clade)]
        comp = all_leaves - s
        if len(s) <= 1 or len(comp) <= 1:
            continue
        canon = s if len(s) < len(comp) or (len(s) == len(comp) and min(s) < min(comp)) else comp
        bl = float(clade.branch_length) if clade.branch_length is not None else 0.0
        splits[canon] = splits.get(canon, 0.0) + bl

    return {
        "splits": splits,
        "leaves": all_leaves,
        "n_leaves": len(all_leaves),
    }


def eval_splits(ref_splits_dict, est_splits_dict, n_leaves, L=None):
    if ref_splits_dict is None or est_splits_dict is None:
        return None
    ref_s = set(ref_splits_dict.keys())
    est_s = set(est_splits_dict.keys())
    n_true = len(ref_s)
    n_est = len(est_s)
    if n_true == 0:
        return None

    tp = len(ref_s & est_s)
    fp = len(est_s - ref_s)
    fn = len(ref_s - est_s)
    rf_bin = fp + fn
    nrf_bin = rf_bin / (2.0 * n_true)
    prec = tp / n_est if n_est > 0 else 1.0
    rec = tp / n_true

    res = {
        "n_true": n_true,
        "n_est": n_est,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "rf_bin": rf_bin,
        "nrf_bin": nrf_bin,
        "precision": prec,
        "recall": rec,
    }

    if L is not None and L > 0:
        active_ref = {s for s, bl in ref_splits_dict.items() if bl * L >= 1.0}
        tp_ge1 = len(active_ref & est_s)
        fp_ge1 = len(est_s - active_ref)
        fn_ge1 = len(active_ref - est_s)
        rf_ge1 = fp_ge1 + fn_ge1
        denom = len(active_ref) + max(len(active_ref), n_est)
        res["n_active_ge1"] = len(active_ref)
        res["rf_ge1"] = rf_ge1
        res["nrf_ge1"] = rf_ge1 / denom if denom > 0 else 0.0

    return res


def process_gene(task):
    ds, aln_path_str, base_dir_str, force_recompute = task
    base_dir = Path(base_dir_str)
    rokas_dir = base_dir / "rokas2018"
    aln_path = Path(aln_path_str)
    gene = re.sub(r"(\.nt)?\.(aln|fasta)$", "", aln_path.name)

    out_dir = rokas_dir / "bw_results_v2" / ds
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{gene}.json"
    raw_bw_json = out_dir / f"{gene}.bw_raw.json"

    if out_json.exists() and not force_recompute:
        try:
            return json.loads(out_json.read_text())
        except Exception:
            pass

    if raw_bw_json.exists():
        try:
            bw_data = json.loads(raw_bw_json.read_text())
            bw_time = bw_data.get("runtime_sec", 0.0)
        except Exception:
            bw_data = None
    else:
        bw_data = None

    if bw_data is None:
        bw_bin = str(base_dir / "target" / "release" / "bushwhacker")
        t0 = time.time()
        proc = subprocess.run(
            [bw_bin, str(aln_path), "-j", str(raw_bw_json)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=120,
        )
        bw_time = time.time() - t0
        if proc.returncode != 0 or not raw_bw_json.exists():
            return None
        try:
            bw_data = json.loads(raw_bw_json.read_text())
        except Exception:
            return None

    N = int(bw_data.get("n_taxa", 0))
    L = int(bw_data.get("seq_len", 0))
    null_cost = float(bw_data.get("star_tree_MDL_nats", 0.0))
    mdl_cost = float(bw_data.get("network_MDL_nats", 0.0))
    raw_cost = float(bw_data.get("raw_Description_Length_nats", 0.0))
    comp_ratio = (1.0 - mdl_cost / null_cost) if null_cost > 0 else 0.0

    bw_mdl_newick = bw_data.get("newick", "")
    bw_bin_newick = bw_data.get("binary_newick", "")
    bw_mdl_info = extract_Induced_splits(bw_mdl_newick, is_path=False, L=L)
    bw_bin_info = extract_Induced_splits(bw_bin_newick, is_path=False, L=L)
    if bw_mdl_info is None or bw_bin_info is None:
        return None

    gene_leaves = bw_bin_info["leaves"]
    trees_dir = rokas_dir / "single-gene_trees" / ds

    best_obs_info = extract_Induced_splits(trees_dir / "Best_observed" / f"{gene}.best.tre", is_path=True, allowed_leaves=gene_leaves, L=L)
    ft_info = extract_Induced_splits(trees_dir / "FastTree" / f"{gene}.fasttree.tre", is_path=True, allowed_leaves=gene_leaves, L=L)
    iq_best_info = extract_Induced_splits(trees_dir / "IQ-TREE-10" / f"{gene}.best.treefile", is_path=True, allowed_leaves=gene_leaves, L=L)
    iq_rep0_info = extract_Induced_splits(trees_dir / "IQ-TREE-10" / f"{gene}.rep0.treefile", is_path=True, allowed_leaves=gene_leaves, L=L)
    rx_best_info = extract_Induced_splits(trees_dir / "RAxML-10" / f"RAxML_bestTree.{gene}_best", is_path=True, allowed_leaves=gene_leaves, L=L)
    rx_rep0_info = extract_Induced_splits(trees_dir / "RAxML-10" / f"RAxML_bestTree.{gene}_mp.RUN.0", is_path=True, allowed_leaves=gene_leaves, L=L)
    ph_best_info = extract_Induced_splits(trees_dir / "PhyML-10" / f"{gene}.best_phyml_tree", is_path=True, allowed_leaves=gene_leaves, L=L)
    ph_rep0_info = extract_Induced_splits(trees_dir / "PhyML-10" / f"{gene}.mp_rep0_phyml_tree", is_path=True, allowed_leaves=gene_leaves, L=L)

    if best_obs_info is None:
        best_obs_info = iq_rep0_info or rx_rep0_info
    if best_obs_info is None:
        return None

    # Load all 10 IQ-TREE replicates + 10 RAxML replicates to measure Intra-Gene ML Run Stability
    ml_rep_sets = []
    for r in range(10):
        p_iq = trees_dir / "IQ-TREE-10" / f"{gene}.rep{r}.treefile"
        if p_iq.exists():
            inf = extract_Induced_splits(p_iq, is_path=True, allowed_leaves=gene_leaves)
            if inf:
                ml_rep_sets.append(set(inf["splits"].keys()))
    for r in range(5):
        for kind in ("mp", "rt"):
            p_rx = trees_dir / "RAxML-10" / f"RAxML_bestTree.{gene}_{kind}.RUN.{r}"
            if p_rx.exists():
                inf = extract_Induced_splits(p_rx, is_path=True, allowed_leaves=gene_leaves)
                if inf:
                    ml_rep_sets.append(set(inf["splits"].keys()))

    best_splits_set = set(best_obs_info["splits"].keys())
    bw_mdl_set = set(bw_mdl_info["splits"].keys())
    if len(ml_rep_sets) >= 2:
        stable_splits = set.intersection(*ml_rep_sets)
        unstable_in_best = best_splits_set - stable_splits
    else:
        stable_splits = best_splits_set
        unstable_in_best = set()

    n_unstable = len(unstable_in_best)
    n_unstable_sub1 = sum(1 for s in unstable_in_best if best_obs_info["splits"].get(s, 0.0) * L < 1.0)
    n_bw_contracts_unstable = len(unstable_in_best - bw_mdl_set)
    n_bw_recovers_stable = len(stable_splits & bw_mdl_set)

    # Species tree comparisons (ASTRAL coalescent & RAxML concatenation species trees induced on `gene_leaves`)
    sp_astral_path = rokas_dir / "coalescent_species_trees" / ds / f"{ds}.Best_observed.ASTRAL.tre"
    sp_concat_path = rokas_dir / "concatenation_species_trees" / ds / f"{ds}.RAxML.concatenation.tre"
    sp_astral = extract_Induced_splits(sp_astral_path, is_path=True, allowed_leaves=gene_leaves) if sp_astral_path.exists() else None
    sp_concat = extract_Induced_splits(sp_concat_path, is_path=True, allowed_leaves=gene_leaves) if sp_concat_path.exists() else None

    res = {
        "dataset": ds,
        "gene": gene,
        "N": N,
        "L": L,
        "raw_cost_nats": raw_cost,
        "null_cost_nats": null_cost,
        "mdl_cost_nats": mdl_cost,
        "comp_ratio": comp_ratio,
        "bw_time_s": bw_time,
        "bw_mdl_newick": bw_mdl_newick,
        "bw_bin_newick": bw_bin_newick,
        "n_true_splits": len(best_splits_set),
        "n_bw_mdl_splits": len(bw_mdl_set),
        "n_stable_ml_splits": len(stable_splits),
        "n_unstable_ml_splits": n_unstable,
        "n_unstable_sub1_mut": n_unstable_sub1,
        "n_bw_contracts_unstable": n_bw_contracts_unstable,
        "n_bw_recovers_stable": n_bw_recovers_stable,
        "vs_best_ml": {
            "bw_mdl": eval_splits(best_obs_info["splits"], bw_mdl_info["splits"], N, L),
            "bw_bin": eval_splits(best_obs_info["splits"], bw_bin_info["splits"], N, L),
            "iqtree_rep0": eval_splits(best_obs_info["splits"], iq_rep0_info["splits"] if iq_rep0_info else None, N, L),
            "raxml_rep0": eval_splits(best_obs_info["splits"], rx_rep0_info["splits"] if rx_rep0_info else None, N, L),
            "phyml_rep0": eval_splits(best_obs_info["splits"], ph_rep0_info["splits"] if ph_rep0_info else None, N, L),
            "fasttree": eval_splits(best_obs_info["splits"], ft_info["splits"] if ft_info else None, N, L),
        },
    }

    if sp_astral is not None:
        res["vs_astral_species"] = {
            "bw_mdl": eval_splits(sp_astral["splits"], bw_mdl_info["splits"], N),
            "bw_bin": eval_splits(sp_astral["splits"], bw_bin_info["splits"], N),
            "best_ml": eval_splits(sp_astral["splits"], best_obs_info["splits"], N),
            "iqtree_rep0": eval_splits(sp_astral["splits"], iq_rep0_info["splits"] if iq_rep0_info else None, N),
            "raxml_rep0": eval_splits(sp_astral["splits"], rx_rep0_info["splits"] if rx_rep0_info else None, N),
            "phyml_rep0": eval_splits(sp_astral["splits"], ph_rep0_info["splits"] if ph_rep0_info else None, N),
            "fasttree": eval_splits(sp_astral["splits"], ft_info["splits"] if ft_info else None, N),
        }

    out_json.write_text(json.dumps(res))
    return res


def summarize_rokas(records):
    print(f"\n==========================================================================================")
    print(f"ZHOU, SHEN, HITTINGER, & ROKAS (2018, MBE) EMPIRICAL PHYLOGENOMIC BENCHMARK ({len(records)} genes)")
    print(f"==========================================================================================")

    by_ds = defaultdict(list)
    for r in records:
        by_ds[r["dataset"]].append(r)

    for ds in DATASETS + ["ALL_COMBINED"]:
        sub = records if ds == "ALL_COMBINED" else by_ds.get(ds, [])
        if not sub:
            continue
        tot_splits = sum(r["n_true_splits"] for r in sub)
        tot_stable = sum(r["n_stable_ml_splits"] for r in sub)
        tot_unstable = sum(r["n_unstable_ml_splits"] for r in sub)
        tot_unstable_sub1 = sum(r["n_unstable_sub1_mut"] for r in sub)
        tot_bw_contracts = sum(r["n_bw_contracts_unstable"] for r in sub)
        mean_comp = sum(r["comp_ratio"] for r in sub) / len(sub)
        mean_bw_ms = sum(r["bw_time_s"] for r in sub) * 1000.0 / len(sub)

        print(f"\n--- Dataset: {ds} (n_genes = {len(sub)}, mean N = {sum(r['N'] for r in sub)/len(sub):.1f}, mean L = {sum(r['L'] for r in sub)/len(sub):.0f} bp) ---")
        print(f"  Mean Bushwhacker Compression Ratio rho(A): {mean_comp*100:.2f}% | Mean Runtime: {mean_bw_ms:.1f} ms/gene")
        if tot_unstable > 0:
            print(
                f"  Multi-Run ML Stability (across 10 RAxML + 10 IQ-TREE runs per gene): "
                f"{tot_stable}/{tot_splits} stable ({tot_stable*100/tot_splits:.1f}%), "
                f"{tot_unstable}/{tot_splits} seed-unstable ({tot_unstable*100/tot_splits:.1f}%)"
            )
            print(
                f"    -> Unstable ML splits with <1 expected mutation (l*L < 1): "
                f"{tot_unstable_sub1}/{tot_unstable} ({tot_unstable_sub1*100/tot_unstable:.1f}%)"
            )
            print(
                f"    -> Unstable ML splits automatically contracted by Bushwhacker (MDL): "
                f"{tot_bw_contracts}/{tot_unstable} ({tot_bw_contracts*100/tot_unstable:.1f}%)"
            )

        # Against ASTRAL Species Tree
        sp_recs = [r["vs_astral_species"] for r in sub if "vs_astral_species" in r]
        if sp_recs:
            print(f"  [Concordance vs. ASTRAL Coalescent Species Tree]")
            print(f"    {'Method':14s} | {'Res/SpSplits':13s} | {'RF_sp':7s} | {'nRF_sp':7s} | {'Discordant FP':13s} | {'Concordance Prec%':17s}")
            for m in ["bw_mdl", "bw_bin", "best_ml", "iqtree_rep0", "raxml_rep0", "phyml_rep0", "fasttree"]:
                vals = [x[m] for x in sp_recs if x.get(m)]
                if not vals:
                    continue
                n_est = sum(v["n_est"] for v in vals) / len(vals)
                n_tr = sum(v["n_true"] for v in vals) / len(vals)
                rf = sum(v["rf_bin"] for v in vals) / len(vals)
                nrf = sum(v["nrf_bin"] for v in vals) / len(vals)
                fp = sum(v["fp"] for v in vals) / len(vals)
                prec = sum(v["precision"] for v in vals) / len(vals)
                print(f"    {m:14s} | {n_est:5.1f}/{n_tr:<6.1f}  | {rf:7.2f} | {nrf:7.4f} | {fp:13.2f} | {prec*100:16.1f}%")

        # Against Best_observed ML Tree (Realized RF_ge1 & Precision)
        ml_recs = [r["vs_best_ml"] for r in sub if "vs_best_ml" in r]
        if ml_recs:
            print(f"  [Comparison vs. Best-of-40 ML Tree (Realized RF_ge1 & Split Precision)]")
            for m in ["bw_mdl", "bw_bin", "iqtree_rep0", "raxml_rep0", "phyml_rep0", "fasttree"]:
                vals = [x[m] for x in ml_recs if x.get(m)]
                if not vals:
                    continue
                rf_b = sum(v["rf_bin"] for v in vals) / len(vals)
                rf_g1 = sum(v.get("rf_ge1", 0) for v in vals) / len(vals)
                fp = sum(v["fp"] for v in vals) / len(vals)
                prec = sum(v["precision"] for v in vals) / len(vals)
                print(f"    {m:14s}: RF_bin={rf_b:6.2f}, Realized_RF_ge1={rf_g1:6.2f}, FP={fp:5.2f}, Precision={prec*100:5.1f}%")

        # Run ASTRAL 5.7.8 on Bushwhacker (MDL) and Bushwhacker (Binary) single-gene trees for this dataset!
        if ds != "ALL_COMBINED":
            rokas_dir = EXTERNAL_DIR / "rokas2018"
            astral_jar = Path(resolve_bin("ASTRAL_JAR", EXTERNAL_DIR / "Astral" / "astral.5.7.8.jar", "astral.jar"))
            sp_dir = rokas_dir / "coalescent_species_trees" / ds
            ref_astral_path = sp_dir / f"{ds}.Best_observed.ASTRAL.tre"
            ref_concat_path = rokas_dir / "concatenation_species_trees" / ds / f"{ds}.RAxML.concatenation.tre"
            if astral_jar.exists() and ref_astral_path.exists():
                bw_sp_dir = rokas_dir / "bw_species_trees" / ds
                bw_sp_dir.mkdir(parents=True, exist_ok=True)
                mdl_gt_file = bw_sp_dir / f"{ds}.bw_mdl.genes.tre"
                bin_gt_file = bw_sp_dir / f"{ds}.bw_bin.genes.tre"
                mdl_sp_file = bw_sp_dir / f"{ds}.bw_mdl.ASTRAL.tre"
                bin_sp_file = bw_sp_dir / f"{ds}.bw_bin.ASTRAL.tre"

                # Write gene trees with clean topology (strip branch lengths/supports for clean ASTRAL quartet parsing)
                def clean_nwk_for_astral(nwk_s):
                    return re.sub(r"\)[\d\.eE+-]+:", "):", nwk_s.strip())

                mdl_gt_file.write_text("\n".join(clean_nwk_for_astral(r["bw_mdl_newick"]) for r in sub if r.get("bw_mdl_newick")) + "\n")
                bin_gt_file.write_text("\n".join(clean_nwk_for_astral(r["bw_bin_newick"]) for r in sub if r.get("bw_bin_newick")) + "\n")

                subprocess.run(["java", "-jar", str(astral_jar), "-i", str(mdl_gt_file), "-o", str(mdl_sp_file)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.run(["java", "-jar", str(astral_jar), "-i", str(bin_gt_file), "-o", str(bin_sp_file)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                ref_ast = extract_Induced_splits(ref_astral_path, is_path=True)
                ref_cat = extract_Induced_splits(ref_concat_path, is_path=True) if ref_concat_path.exists() else None
                if ref_ast:
                    print(f"  [Coalescent Species Tree Reconstruction via ASTRAL (nRF vs. Best_observed.ASTRAL / RAxML.Concat)]")
                    candidates = [
                        ("BW_MDL+ASTRAL", mdl_sp_file),
                        ("BW_Bin+ASTRAL", bin_sp_file),
                        ("FastTree+ASTRAL", sp_dir / f"{ds}.FastTree.ASTRAL.tre"),
                        ("IQ-TREE+ASTRAL", sp_dir / f"{ds}.IQ-TREE.ASTRAL.tre"),
                        ("PhyML+ASTRAL", sp_dir / f"{ds}.PhyML.ASTRAL.tre"),
                        ("RAxML+ASTRAL", sp_dir / f"{ds}.RAxML.ASTRAL.tre"),
                    ]
                    for label, p_cand in candidates:
                        if p_cand.exists():
                            c_inf = extract_Induced_splits(p_cand, is_path=True, allowed_leaves=ref_ast["leaves"])
                            if c_inf:
                                ev_ast = eval_splits(ref_ast["splits"], c_inf["splits"], ref_ast["n_leaves"])
                                ev_cat = eval_splits(ref_cat["splits"], c_inf["splits"], ref_cat["n_leaves"]) if ref_cat else None
                                cat_str = f", RF_concat={ev_cat['rf_bin']:2d} (nRF={ev_cat['nrf_bin']:.4f})" if ev_cat else ""
                                print(f"    {label:16s}: RF_astral={ev_ast['rf_bin']:2d} (nRF={ev_ast['nrf_bin']:.4f}){cat_str}")

    # Stratify all genes by Bushwhacker Compression Ratio rho(A) quartiles!
    if len(records) >= 20:
        print(f"\n--- Empirical Genes Stratified by Bushwhacker Compression Ratio rho(A) ---")
        sorted_recs = sorted(records, key=lambda r: r["comp_ratio"])
        q = len(sorted_recs) // 4
        quartiles = [
            ("Q1: Least Compressible (Lowest rho)", sorted_recs[:q]),
            ("Q2: Low-Moderate Compressibility", sorted_recs[q : 2 * q]),
            ("Q3: Moderate-High Compressibility", sorted_recs[2 * q : 3 * q]),
            ("Q4: Most Compressible (Highest rho)", sorted_recs[3 * q :]),
        ]
        for label, qsub in quartiles:
            r_min, r_max = qsub[0]["comp_ratio"] * 100, qsub[-1]["comp_ratio"] * 100
            res_rate = sum(r["n_bw_mdl_splits"] for r in qsub) * 100.0 / max(1, sum(r["n_true_splits"] for r in qsub))
            unstab_rate = sum(r["n_unstable_ml_splits"] for r in qsub) * 100.0 / max(1, sum(r["n_true_splits"] for r in qsub))
            sp_vals_bw = [r["vs_astral_species"]["bw_mdl"] for r in qsub if "vs_astral_species" in r and r["vs_astral_species"].get("bw_mdl")]
            sp_vals_ml = [r["vs_astral_species"]["best_ml"] for r in qsub if "vs_astral_species" in r and r["vs_astral_species"].get("best_ml")]
            bw_prec = sum(v["precision"] for v in sp_vals_bw) * 100 / len(sp_vals_bw) if sp_vals_bw else 0.0
            ml_prec = sum(v["precision"] for v in sp_vals_ml) * 100 / len(sp_vals_ml) if sp_vals_ml else 0.0
            bw_fp = sum(v["fp"] for v in sp_vals_bw) / len(sp_vals_bw) if sp_vals_bw else 0.0
            ml_fp = sum(v["fp"] for v in sp_vals_ml) / len(sp_vals_ml) if sp_vals_ml else 0.0
            bw_nrf = sum(v["nrf_bin"] for v in sp_vals_bw) / len(sp_vals_bw) if sp_vals_bw else 0.0
            ml_nrf = sum(v["nrf_bin"] for v in sp_vals_ml) / len(sp_vals_ml) if sp_vals_ml else 0.0
            print(
                f"  * {label} (rho={r_min:.1f}%..{r_max:.1f}%, n={len(qsub)}): "
                f"BW Resolved={res_rate:.1f}%, ML Unstable Splits={unstab_rate:.1f}% | "
                f"Species Concordance Prec: BW_MDL={bw_prec:.1f}% (FP={bw_fp:.1f}) vs Best_ML={ml_prec:.1f}% (FP={ml_fp:.1f}) "
                f"(nRF_sp: BW_MDL={bw_nrf:.4f} vs Best_ML={ml_nrf:.4f})"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default=str(EXTERNAL_DIR))
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--datasets", nargs="*", default=DATASETS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    rokas_dir = base_dir / "rokas2018"

    tasks = []
    for ds in args.datasets:
        aln_dir = rokas_dir / "single-gene_alignments" / ds
        if not aln_dir.exists():
            continue
        files = sorted(list(aln_dir.glob("*.aln")) + list(aln_dir.glob("*.fasta")))
        for aln_file in files:
            tasks.append((ds, str(aln_file), str(base_dir), args.force))

    print(f"Running Bushwhacker across {len(tasks)} empirical phylogenomic genes ({args.datasets}) with {args.workers} workers...")
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_gene, t) for t in tasks]
        for idx, fut in enumerate(as_completed(futs), 1):
            try:
                r = fut.result()
                if r is not None:
                    records.append(r)
            except Exception as e:
                pass
            if idx % 500 == 0 or idx == len(tasks):
                print(f"  Processed {idx}/{len(tasks)} genes ({len(records)} valid)...", flush=True)

    summarize_rokas(records)


if __name__ == "__main__":
    main()
