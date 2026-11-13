#!/usr/bin/env python3
"""
BARS Revision Controls for Pillar 3 (Cellular BCR Germinal Centers):
  1. GCtree (DeWitt et al. 2018; PHYLIP dnapars V=200 thorough + branching-process MLE)
     across all 50 simulated GCs in BOTH zero-leakage (abundance=1) and oracle-abundance modes.
  2. Neutral Negative Control (50 simulated GCs with beta_bind = 0.0, no affinity selection)
     evaluated blindly by Bushwhacker_MDL.
  3. Within-GC Topological-Depth & Hamming-Depth Adjusted Controls on all 147 empirical
     GCReplay GCs (28 mice):
       - Fixed-depth stratification (d=1, d=2, d=3, d>=4) for k_out vs abundance and delta_bind.
       - Within-GC, exact-depth-matched paired differences (hub minus non-hub at same (GC, d)).
       - Two-way fixed-effects OLS (y ~ is_hub + depth + GC_FE) with mouse-clustered (n=28)
         sandwich standard errors & p-values for both log1p(abundance) and DMS delta_bind.

Updates `results/single_cell/pillar3_bcr_cellular.json` cleanly in place.
"""

import json
import math
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
from _env_paths import RESULTS_DIR, BUSHWHACKER_BIN

# Ensure dnapars and scratch/bcr_gcreplay are active when running from clean env or scratch
SCRATCH_DIR = REPO_ROOT / "scratch"
if (SCRATCH_DIR / "dnapars").exists():
    os.environ["PATH"] = f"{SCRATCH_DIR}:{os.environ.get('PATH', '')}"
if not os.environ.get("BCR_DATA_DIR") and (SCRATCH_DIR / "bcr_gcreplay").exists():
    os.environ["BCR_DATA_DIR"] = str(SCRATCH_DIR / "bcr_gcreplay")

import run_pillar3_bcr_cellular as p3
import gctree.branching_processes as bp
import gctree.phylip_parse as pp


DNAPARS_BIN = str(SCRATCH_DIR / "dnapars")
OUT_JSON = RESULTS_DIR / "single_cell" / "pillar3_bcr_cellular.json"


def run_gctree_on_sim_geno(geno_seqs: dict, geno_counts: dict, true_geno_nwk: str, mode: str) -> dict:
    """Run PHYLIP dnapars (V=200, J=3) + GCtree CollapsedForest MLE + filter_trees."""
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="gctree_sim_") as tmpd:
        td = Path(tmpd)
        phy = td / "seq.phy"
        abund = td / "abund.csv"
        short_to_orig = {"GL_NAIVE": "naive"}
        rows = [("GL_NAIVE", geno_seqs["naive"], 0)]
        for i, (k, sq) in enumerate(sorted(geno_seqs.items())):
            if k == "naive":
                continue
            sid = f"S{i:04d}"
            short_to_orig[sid] = k
            ab = 1 if mode == "zero_leakage" else int(geno_counts.get(sq, 1))
            rows.append((sid, sq, ab))
        seqlen = len(rows[0][1])
        with open(phy, "w") as f:
            f.write(f" {len(rows)} {seqlen}\n")
            for sid, sq, _ in rows:
                f.write(f"{sid:<10}{sq}\n")
        with open(abund, "w") as f:
            for sid, _, ab in rows:
                f.write(f"{sid},{ab}\n")

        cfg = "\n".join([str(phy), "V", "200", "J", "1", "3", "O", "1", "4", "5", ".", "Y"]) + "\n"
        proc = subprocess.run(
            [DNAPARS_BIN], input=cfg, cwd=td, capture_output=True, text=True, timeout=60
        )
        if proc.returncode != 0 or not (td / "outfile").exists():
            return {"status": "error", "sec": time.perf_counter() - t0}
        forest = bp.CollapsedForest(pp.parse_outfile(str(td / "outfile"), str(abund), "GL_NAIVE"))
        if forest.n_trees > 1:
            forest.mle(marginal=True)
        ctrees, _, _ = forest.filter_trees(verbose=False)
        ct = ctrees[0].tree.copy()
        for nd in list(ct.traverse()):
            nm = (nd.name or "").strip()
            if not nd.is_leaf():
                if nm in short_to_orig:
                    nd.add_child(name=short_to_orig[nm], dist=0.0)
                    nd.name = ""
            else:
                if nm in short_to_orig:
                    nd.name = short_to_orig[nm]
        nwk = ct.write(format=9)
        elapsed = time.perf_counter() - t0
        rf = p3.compare_trees_rf(nwk, true_geno_nwk)
        rf["sec"] = elapsed
        rf["n_parsimony_trees"] = forest.n_trees
        return rf


def _worker_sim_controls(args):
    rep_idx, naive_nt, codon_meta, dms_map, fmut, fsub = args
    # 1. Exact matching selection GC (seed=20261110 + rep_idx*7919) for GCtree head-to-head
    sim_sel = p3.simulate_single_germinal_center(
        rep_idx, naive_nt, codon_meta, dms_map, fmut, fsub, n_sample=90, beta_bind=1.65
    )
    gct_zero = None
    gct_oracle = None
    if sim_sel["n_unique_genotypes"] >= 5:
        gct_zero = run_gctree_on_sim_geno(sim_sel["geno_seqs"], sim_sel["geno_counts"], sim_sel["true_geno_nwk"], "zero_leakage")
        gct_oracle = run_gctree_on_sim_geno(sim_sel["geno_seqs"], sim_sel["geno_counts"], sim_sel["true_geno_nwk"], "oracle_abundance")

    # 2. Neutral control GC (beta_bind = 0.0)
    sim_neu = p3.simulate_single_germinal_center(
        rep_idx, naive_nt, codon_meta, dms_map, fmut, fsub, n_sample=90, beta_bind=0.0
    )
    neu_records = []
    neu_geno_eval = None
    if sim_neu["n_unique_genotypes"] >= 5:
        with tempfile.TemporaryDirectory(prefix="bw_neu_") as tmpd:
            td = Path(tmpd)
            fa = td / "geno.fa"
            js = td / "bw.json"
            nwk_p = td / "bw.nwk"
            p3.write_fasta(sim_neu["geno_seqs"], fa)
            t0 = time.perf_counter()
            subprocess.run(
                [p3.BUSHWHACKER_BIN, str(fa), "-o", str(nwk_p), "-j", str(js), "-T", "1"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
            )
            bw_sec = time.perf_counter() - t0
            bw_nwk = nwk_p.read_text().strip()
        neu_geno_eval = p3.compare_trees_rf(bw_nwk, sim_neu["true_geno_nwk"])
        neu_geno_eval["sec"] = bw_sec
        bw_burst, _ = p3.root_tree_at_outgroup_and_analyze_bursts(bw_nwk, "naive", sim_neu["geno_seqs"])
        for gnm, sq in sim_neu["geno_seqs"].items():
            if gnm == "naive":
                continue
            has_stop, db, _ = p3.score_sequence_dms(sq, codon_meta, dms_map)
            ham = sum(1 for a, b in zip(sq, naive_nt) if a != b)
            neu_records.append({
                "rep_idx": rep_idx,
                "k_out": bw_burst.get(gnm, 0),
                "abundance": int(sim_neu["geno_counts"].get(sq, 1)),
                "delta_bind": None if has_stop else db,
                "hamming": ham,
            })

    return {
        "rep_idx": rep_idx,
        "gctree_zero": gct_zero,
        "gctree_oracle": gct_oracle,
        "neu_geno_eval": neu_geno_eval,
        "neu_records": neu_records,
    }


def extract_depth_aware_geno_metrics(nwk_str: str, naive_nt: str, seq_map: dict):
    """
    Root Bushwhacker tree at naive outgroup (without using abundance) and return for each
    non-naive genotype leaf:
      - k_out: inferred mutant daughter lineages
      - topo_depth: mutational step depth from naive along the rooted Bushwhacker tree
      - hamming: exact nucleotide Hamming distance from naive_nt
    """
    t = p3.parse_newick_tree(nwk_str)
    adj = defaultdict(list)
    node_names = {}
    counter = [0]

    def _build(nd) -> int:
        nid = counter[0]
        counter[0] += 1
        node_names[nid] = nd.name if not nd.children else ""
        for ch in nd.children:
            cid = _build(ch)
            adj[nid].append(cid)
            adj[cid].append(nid)
        return nid

    _build(t)
    og_ids = [nid for nid, nm in node_names.items() if nm == "naive"]
    if not og_ids or not adj[og_ids[0]]:
        return {}
    og_id = og_ids[0]
    root_internal = adj[og_id][0]

    children_map = defaultdict(list)

    def _orient(u: int, p: int):
        for v in adj[u]:
            if v == p or v == og_id:
                continue
            children_map[u].append(v)
            _orient(v, u)

    _orient(root_internal, og_id)

    def _clean(u: int) -> int:
        new_ch = [_clean(v) for v in children_map[u]]
        children_map[u] = new_ch
        if len(new_ch) == 1 and not node_names[u]:
            return new_ch[0]
        return u

    root_internal = _clean(root_internal)

    # Fitch reconstruction
    postorder = []

    def _post(u: int):
        for v in children_map[u]:
            _post(v)
        postorder.append(u)

    _post(root_internal)
    seq_len = len(naive_nt)
    base_to_bit = {"A": 1, "C": 2, "G": 4, "T": 8}
    bit_to_base = {1: "A", 2: "C", 4: "G", 8: "T"}
    fitch_masks = {}
    for u in postorder:
        nm = node_names[u]
        if not children_map[u]:
            sq = seq_map.get(nm, naive_nt)
            fitch_masks[u] = [base_to_bit.get(ch, 15) for ch in sq]
        else:
            ch_lists = [fitch_masks[v] for v in children_map[u]]
            masks_u = [0] * seq_len
            for pos in range(seq_len):
                inter = 15
                union = 0
                for cl in ch_lists:
                    b = cl[pos]
                    inter &= b
                    union |= b
                masks_u[pos] = inter if inter != 0 else union
            fitch_masks[u] = masks_u

    recon_seq = {}
    topo_depth = {}

    def _topdown(u: int, parent_sq: str, cur_d: int):
        if not children_map[u] and node_names[u] in seq_map:
            sq_u = seq_map[node_names[u]]
            recon_seq[u] = sq_u
            topo_depth[u] = cur_d + (1 if sq_u != parent_sq else 0)
            return
        masks_u = fitch_masks[u]
        chars = []
        for pos in range(seq_len):
            pb = base_to_bit.get(parent_sq[pos], 1)
            m = masks_u[pos]
            if m & pb:
                chars.append(parent_sq[pos])
            else:
                low_bit = m & -m
                chars.append(bit_to_base.get(low_bit, "A"))
        sq_u = "".join(chars)
        recon_seq[u] = sq_u
        d_u = cur_d + (1 if sq_u != parent_sq else 0)
        topo_depth[u] = d_u
        for v in children_map[u]:
            _topdown(v, sq_u, d_u)

    _topdown(root_internal, naive_nt, 0)

    out = {}
    for u in postorder:
        if children_map[u]:
            sq_u = recon_seq[u]
            zero_leaves = [node_names[v] for v in children_map[u] if not children_map[v] and recon_seq[v] == sq_u]
            mutant_branches = sum(1 for v in children_map[u] if not (not children_map[v] and recon_seq[v] == sq_u))
            for lnm in zero_leaves:
                if lnm and lnm != "naive":
                    out[lnm] = {
                        "k_out": mutant_branches,
                        "topo_depth": max(1, topo_depth[u]),
                        "hamming": sum(1 for a, b in zip(seq_map[lnm], naive_nt) if a != b),
                    }
        else:
            lnm = node_names[u]
            if lnm and lnm != "naive" and lnm not in out:
                out[lnm] = {
                    "k_out": 0,
                    "topo_depth": max(1, topo_depth[u]),
                    "hamming": sum(1 for a, b in zip(seq_map[lnm], naive_nt) if a != b),
                }
    return out


def _worker_emp_depth(task):
    if len(task["geno_seq_map"]) < 5:
        return []
    with tempfile.TemporaryDirectory(prefix="bw_emp_d_") as tmpd:
        td = Path(tmpd)
        fa = td / "geno.fa"
        nwk_p = td / "bw.nwk"
        p3.write_fasta(task["geno_seq_map"], fa)
        subprocess.run(
            [p3.BUSHWHACKER_BIN, str(fa), "-o", str(nwk_p), "-T", "1"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
        )
        bw_nwk = nwk_p.read_text().strip()
    dmetrics = extract_depth_aware_geno_metrics(bw_nwk, task["geno_seq_map"]["naive"], task["geno_seq_map"])
    recs = []
    for gnm, md in task["geno_meta"].items():
        dm = dmetrics.get(gnm)
        if dm is None:
            continue
        recs.append({
            "uid": task["uid"],
            "mouse": task["mouse"],
            "k_out": dm["k_out"],
            "topo_depth": dm["topo_depth"],
            "hamming": dm["hamming"],
            "abundance": md["abundance"],
            "delta_bind": md["delta_bind"],
        })
    return recs


def fixed_effects_cluster_ols(y, x_vars, cluster_fe, se_clusters):
    """
    Fit y ~ x_vars + FE(cluster_fe) via Frisch-Waugh-Lovell within-FE demeaning,
    and compute Liang-Zeger / CR1 cluster-robust standard errors clustered by se_clusters.
    """
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(x_vars, dtype=np.float64)
    n, p = X.shape
    fe_groups = defaultdict(list)
    for idx, g in enumerate(cluster_fe):
        fe_groups[g].append(idx)

    # Demean within FE groups (also drop singleton FE groups which have 0 within-group degrees of freedom)
    keep = np.ones(n, dtype=bool)
    y_dm = y.copy()
    X_dm = X.copy()
    for g, idxs in fe_groups.items():
        if len(idxs) < 2:
            keep[idxs] = False
            continue
        y_dm[idxs] -= np.mean(y[idxs])
        X_dm[idxs, :] -= np.mean(X[idxs, :], axis=0, keepdims=True)

    y_k = y_dm[keep]
    X_k = X_dm[keep]
    se_cl_k = [se_clusters[i] for i in range(n) if keep[i]]
    n_k = len(y_k)
    if n_k <= p + 2:
        return {}

    XtX_inv = np.linalg.pinv(X_k.T @ X_k)
    beta = XtX_inv @ (X_k.T @ y_k)
    resid = y_k - X_k @ beta

    cl_map = defaultdict(list)
    for i, c in enumerate(se_cl_k):
        cl_map[c].append(i)
    G = len(cl_map)
    meat = np.zeros((p, p), dtype=np.float64)
    for idxs in cl_map.values():
        score = X_k[idxs, :].T @ resid[idxs]
        meat += np.outer(score, score)
    # Small-sample CR1 adjustment
    adj = (G / max(1, G - 1)) * ((n_k - 1) / max(1, n_k - p - len({cluster_fe[i] for i in range(n) if keep[i]})))
    cov = adj * (XtX_inv @ meat @ XtX_inv)
    se = np.sqrt(np.maximum(np.diag(cov), 1e-18))
    t_stats = beta / se
    p_vals = [float(2.0 * stats.t.sf(abs(ts), df=max(1, G - 1))) for ts in t_stats]
    return {
        "n_obs": int(n_k),
        "n_gc_clusters": int(len({cluster_fe[i] for i in range(n) if keep[i]})),
        "n_se_clusters": int(G),
        "beta": [round(float(b), 5) for b in beta],
        "se_clustered": [round(float(s), 5) for s in se],
        "t_stat": [round(float(t), 4) for t in t_stats],
        "p_value_two_sided": p_vals,
    }


def main():
    t0 = time.perf_counter()
    print("[BCR BARS Controls] Loading CGG germline & DMS...", flush=True)
    naive_nt, codon_meta, dms_map, fmut, fsub = p3.load_cgg_dms_and_shm()

    # 1. Run 50 simulated GCs: GCtree (zero_leakage + oracle) + Neutral beta_bind=0.0
    n_sims = 50
    n_workers = min(max(os.cpu_count() or 8, 1), 16)
    print(f"[1/2] Running GCtree + Neutral beta_bind=0.0 across {n_sims} GCs ({n_workers} workers)...", flush=True)
    args = [(r, naive_nt, codon_meta, dms_map, fmut, fsub) for r in range(n_sims)]
    sim_out = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = [pool.submit(_worker_sim_controls, a) for a in args]
        for fut in as_completed(futs):
            sim_out.append(fut.result())
    sim_out.sort(key=lambda x: x["rep_idx"])

    gct_zero_list = [{"GCtree_ZeroLeakage": r["gctree_zero"]} for r in sim_out if r["gctree_zero"] and r["gctree_zero"].get("status") != "error"]
    gct_oracle_list = [{"GCtree_OracleAbundance": r["gctree_oracle"]} for r in sim_out if r["gctree_oracle"] and r["gctree_oracle"].get("status") != "error"]
    gct_zero_sum = p3.aggregate_method_evals(gct_zero_list).get("GCtree_ZeroLeakage", {})
    gct_oracle_sum = p3.aggregate_method_evals(gct_oracle_list).get("GCtree_OracleAbundance", {})

    # Neutral beta_bind=0.0 analysis
    neu_recs = [rec for r in sim_out for rec in r["neu_records"]]
    neu_evals = [{"Bushwhacker_MDL_Neutral": r["neu_geno_eval"]} for r in sim_out if r["neu_geno_eval"]]
    neu_tree_sum = p3.aggregate_method_evals(neu_evals).get("Bushwhacker_MDL_Neutral", {})

    neu_hub_ab = [r["abundance"] for r in neu_recs if r["k_out"] >= 2]
    neu_nonhub_ab = [r["abundance"] for r in neu_recs if r["k_out"] < 2]
    _, neu_p_ab_mwu = p3.mann_whitney_u_two_sided(neu_hub_ab, neu_nonhub_ab)
    neu_rho_ab, neu_p_ab_rho = p3.spearman_corr([r["k_out"] for r in neu_recs], [r["abundance"] for r in neu_recs])

    neu_dms_recs = [r for r in neu_recs if r["delta_bind"] is not None]
    neu_hub_db = [r["delta_bind"] for r in neu_dms_recs if r["k_out"] >= 2]
    neu_nonhub_db = [r["delta_bind"] for r in neu_dms_recs if r["k_out"] < 2]
    _, neu_p_db_mwu = p3.mann_whitney_u_two_sided(neu_hub_db, neu_nonhub_db)
    neu_rho_db, neu_p_db_rho = p3.spearman_corr([r["k_out"] for r in neu_dms_recs], [r["delta_bind"] for r in neu_dms_recs])

    # Within-GC paired diff for neutral sims (k_out>=2 minus k_out<2 per rep)
    neu_rep_ab_diffs, neu_rep_db_diffs = [], []
    for r in sim_out:
        recs = r["neu_records"]
        ha = [x["abundance"] for x in recs if x["k_out"] >= 2]
        nha = [x["abundance"] for x in recs if x["k_out"] < 2]
        if ha and nha:
            neu_rep_ab_diffs.append(sum(ha) / len(ha) - sum(nha) / len(nha))
        hd = [x["delta_bind"] for x in recs if x["k_out"] >= 2 and x["delta_bind"] is not None]
        nhd = [x["delta_bind"] for x in recs if x["k_out"] < 2 and x["delta_bind"] is not None]
        if hd and nhd:
            neu_rep_db_diffs.append(sum(hd) / len(hd) - sum(nhd) / len(nhd))

    # Neutral OLS controlling for depth (Hamming from naive) + rep_idx FE
    neu_ab_ols = fixed_effects_cluster_ols(
        [math.log1p(r["abundance"]) for r in neu_recs],
        [[1.0 if r["k_out"] >= 2 else 0.0, float(r["hamming"])] for r in neu_recs],
        [r["rep_idx"] for r in neu_recs],
        [r["rep_idx"] for r in neu_recs],
    )
    neu_db_ols = fixed_effects_cluster_ols(
        [r["delta_bind"] for r in neu_dms_recs],
        [[1.0 if r["k_out"] >= 2 else 0.0, float(r["hamming"])] for r in neu_dms_recs],
        [r["rep_idx"] for r in neu_dms_recs],
        [r["rep_idx"] for r in neu_dms_recs],
    )

    print(
        f"  GCtree ZeroLeakage Prec={gct_zero_sum.get('mean_precision', 0):.4f}, Rec={gct_zero_sum.get('mean_recall', 0):.4f}, F1={gct_zero_sum.get('mean_f1', 0):.4f}; "
        f"Oracle Prec={gct_oracle_sum.get('mean_precision', 0):.4f}, F1={gct_oracle_sum.get('mean_f1', 0):.4f}",
        flush=True,
    )
    print(
        f"  Neutral beta_bind=0: abundance hub/nonhub={sum(neu_hub_ab)/max(1,len(neu_hub_ab)):.2f}/{sum(neu_nonhub_ab)/max(1,len(neu_nonhub_ab)):.2f} "
        f"(MWU p={neu_p_ab_mwu:.2e}), delta_bind diff={sum(neu_hub_db)/max(1,len(neu_hub_db)) - sum(neu_nonhub_db)/max(1,len(neu_nonhub_db)):+.4f} (MWU p={neu_p_db_mwu:.3f})",
        flush=True,
    )

    # 2. Depth-adjusted empirical GCReplay controls (147 GCs, 28 mice)
    print("[2/2] Running depth-adjusted empirical GCReplay controls across 147 GCs...", flush=True)
    gc_tasks = p3.load_empirical_gcreplay_datasets(naive_nt)
    emp_recs = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for res in pool.map(_worker_emp_depth, gc_tasks, chunksize=2):
            emp_recs.extend(res)

    # Fixed topological-depth stratification (d=1, d=2, d=3, d>=4)
    depth_strata = {}
    for lbl, pred in [
        ("depth=1", lambda d: d == 1),
        ("depth=2", lambda d: d == 2),
        ("depth=3", lambda d: d == 3),
        ("depth>=4", lambda d: d >= 4),
    ]:
        sub = [r for r in emp_recs if pred(r["topo_depth"])]
        hub = [r for r in sub if r["k_out"] >= 2]
        nonhub = [r for r in sub if r["k_out"] < 2]
        hub_a = [r["abundance"] for r in hub]
        non_a = [r["abundance"] for r in nonhub]
        _, p_a = p3.mann_whitney_u_two_sided(hub_a, non_a) if (hub_a and non_a) else (0.0, 1.0)
        hub_d = [r["delta_bind"] for r in hub if r["delta_bind"] is not None]
        non_d = [r["delta_bind"] for r in nonhub if r["delta_bind"] is not None]
        _, p_d = p3.mann_whitney_u_two_sided(hub_d, non_d) if (hub_d and non_d) else (0.0, 1.0)
        depth_strata[lbl] = {
            "n_genotypes": len(sub),
            "n_hubs_k_ge2": len(hub),
            "n_nonhubs_k_lt2": len(nonhub),
            "hub_mean_cells": round(sum(hub_a) / max(1, len(hub_a)), 4) if hub_a else None,
            "nonhub_mean_cells": round(sum(non_a) / max(1, len(non_a)), 4) if non_a else None,
            "cell_fold_hub_over_nonhub": round((sum(hub_a) / max(1, len(hub_a))) / max(1e-9, sum(non_a) / max(1, len(non_a))), 3) if (hub_a and non_a) else None,
            "mwu_p_abundance": p_a,
            "hub_mean_delta_bind": round(sum(hub_d) / max(1, len(hub_d)), 4) if hub_d else None,
            "nonhub_mean_delta_bind": round(sum(non_d) / max(1, len(non_d)), 4) if non_d else None,
            "delta_bind_diff_hub_minus_nonhub": round((sum(hub_d) / max(1, len(hub_d))) - (sum(non_d) / max(1, len(non_d))), 4) if (hub_d and non_d) else None,
            "mwu_p_delta_bind": p_d,
        }

    # Exact (GC, topo_depth) matched paired difference aggregated per GC -> mouse
    gc_depth_ab = defaultdict(list)
    gc_depth_db = defaultdict(list)
    gc_to_mouse = {r["uid"]: r["mouse"] for r in emp_recs}
    for r in emp_recs:
        gc_depth_ab[(r["uid"], r["topo_depth"])].append((r["k_out"], r["abundance"]))
        if r["delta_bind"] is not None:
            gc_depth_db[(r["uid"], r["topo_depth"])].append((r["k_out"], r["delta_bind"]))

    matched_gc_ab_diff, matched_gc_ab_mouse = [], []
    matched_gc_db_diff, matched_gc_db_mouse = [], []
    by_gc_ab = defaultdict(list)
    for (uid, _d), pairs in gc_depth_ab.items():
        ha = [v for k, v in pairs if k >= 2]
        nha = [v for k, v in pairs if k < 2]
        if ha and nha:
            by_gc_ab[uid].append(sum(ha) / len(ha) - sum(nha) / len(nha))
    for uid, diffs in sorted(by_gc_ab.items()):
        matched_gc_ab_diff.append(sum(diffs) / len(diffs))
        matched_gc_ab_mouse.append(gc_to_mouse[uid])

    by_gc_db = defaultdict(list)
    for (uid, _d), pairs in gc_depth_db.items():
        hd = [v for k, v in pairs if k >= 2]
        nhd = [v for k, v in pairs if k < 2]
        if hd and nhd:
            by_gc_db[uid].append(sum(hd) / len(hd) - sum(nhd) / len(nhd))
    for uid, diffs in sorted(by_gc_db.items()):
        matched_gc_db_diff.append(sum(diffs) / len(diffs))
        matched_gc_db_mouse.append(gc_to_mouse[uid])

    # Aggregate matched-depth diffs to mouse level (N<=28 mice)
    m_ab_by_m = defaultdict(list)
    for diff, ms in zip(matched_gc_ab_diff, matched_gc_ab_mouse):
        m_ab_by_m[ms].append(diff)
    matched_mouse_ab = [sum(v) / len(v) for v in m_ab_by_m.values()]

    m_db_by_m = defaultdict(list)
    for diff, ms in zip(matched_gc_db_diff, matched_gc_db_mouse):
        m_db_by_m[ms].append(diff)
    matched_mouse_db = [sum(v) / len(v) for v in m_db_by_m.values()]

    # Two-way FE OLS: y ~ is_hub + topo_depth + hamming + GC_FE (with mouse-clustered SEs)
    emp_dms_recs = [r for r in emp_recs if r["delta_bind"] is not None]
    fe_ab_topo = fixed_effects_cluster_ols(
        [math.log1p(r["abundance"]) for r in emp_recs],
        [[1.0 if r["k_out"] >= 2 else 0.0, float(r["topo_depth"]), float(r["hamming"])] for r in emp_recs],
        [r["uid"] for r in emp_recs],
        [r["mouse"] for r in emp_recs],
    )
    fe_db_topo = fixed_effects_cluster_ols(
        [r["delta_bind"] for r in emp_dms_recs],
        [[1.0 if r["k_out"] >= 2 else 0.0, float(r["topo_depth"]), float(r["hamming"])] for r in emp_dms_recs],
        [r["uid"] for r in emp_dms_recs],
        [r["mouse"] for r in emp_dms_recs],
    )

    controls_payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gctree_simulated_50_gcs_unique_genotype": {
            "settings": "PHYLIP dnapars (V=200 saved trees, J=3 random jumbles, Outgroup=GL_NAIVE) + gctree CollapsedForest branching-process MLE + filter_trees",
            "zero_leakage_abundance_1": gct_zero_sum,
            "oracle_true_abundance": gct_oracle_sum,
        },
        "neutral_simulated_50_gcs_beta_bind_0": {
            "beta_bind": 0.0,
            "n_genotypes": len(neu_recs),
            "tree_recovery_bushwhacker_mdl": neu_tree_sum,
            "hub_mean_cells": round(sum(neu_hub_ab) / max(1, len(neu_hub_ab)), 4),
            "nonhub_mean_cells": round(sum(neu_nonhub_ab) / max(1, len(neu_nonhub_ab)), 4),
            "cell_fold_hub_over_nonhub": round((sum(neu_hub_ab) / max(1, len(neu_hub_ab))) / max(1e-9, sum(neu_nonhub_ab) / max(1, len(neu_nonhub_ab))), 4),
            "mwu_p_abundance": neu_p_ab_mwu,
            "spearman_rho_kout_abundance": round(neu_rho_ab, 4),
            "spearman_p_kout_abundance": neu_p_ab_rho,
            "hub_mean_delta_bind": round(sum(neu_hub_db) / max(1, len(neu_hub_db)), 4) if neu_hub_db else 0.0,
            "nonhub_mean_delta_bind": round(sum(neu_nonhub_db) / max(1, len(neu_nonhub_db)), 4) if neu_nonhub_db else 0.0,
            "delta_bind_diff": round((sum(neu_hub_db) / max(1, len(neu_hub_db))) - (sum(neu_nonhub_db) / max(1, len(neu_nonhub_db))), 4) if (neu_hub_db and neu_nonhub_db) else 0.0,
            "mwu_p_delta_bind": neu_p_db_mwu,
            "spearman_rho_kout_delta_bind": round(neu_rho_db, 4),
            "spearman_p_kout_delta_bind": neu_p_db_rho,
            "rep_paired_abundance_diff": p3.paired_wilcoxon_and_cluster_ci(neu_rep_ab_diffs),
            "rep_paired_delta_bind_diff": p3.paired_wilcoxon_and_cluster_ci(neu_rep_db_diffs),
            "fe_ols_log1p_abundance_vs_ishub_hamming": neu_ab_ols,
            "fe_ols_delta_bind_vs_ishub_hamming": neu_db_ols,
        },
        "empirical_gcreplay_depth_adjusted_controls": {
            "n_evaluated_genotypes": len(emp_recs),
            "fixed_topo_depth_strata": depth_strata,
            "exact_gc_and_depth_matched_paired_abundance_gc": p3.paired_wilcoxon_and_cluster_ci(matched_gc_ab_diff, matched_gc_ab_mouse),
            "exact_gc_and_depth_matched_paired_abundance_mouse": p3.paired_wilcoxon_and_cluster_ci(matched_mouse_ab),
            "exact_gc_and_depth_matched_paired_delta_bind_gc": p3.paired_wilcoxon_and_cluster_ci(matched_gc_db_diff, matched_gc_db_mouse),
            "exact_gc_and_depth_matched_paired_delta_bind_mouse": p3.paired_wilcoxon_and_cluster_ci(matched_mouse_db),
            "fe_ols_log1p_abundance_vs_ishub_topodepth_hamming": fe_ab_topo,
            "fe_ols_delta_bind_vs_ishub_topodepth_hamming": fe_db_topo,
        },
    }

    # Merge cleanly into existing pillar3_bcr_cellular.json
    if OUT_JSON.exists():
        existing = json.loads(OUT_JSON.read_text())
    else:
        existing = {}
    existing["bars_revision_controls"] = controls_payload
    if "part3a_simulations" in existing and "unique_genotype_benchmark" in existing["part3a_simulations"]:
        existing["part3a_simulations"]["unique_genotype_benchmark"]["GCtree_ZeroLeakage"] = gct_zero_sum
        existing["part3a_simulations"]["unique_genotype_benchmark"]["GCtree_OracleAbundance"] = gct_oracle_sum
    OUT_JSON.write_text(json.dumps(existing, indent=2))
    print(f"[BCR BARS Controls COMPLETE] Updated {OUT_JSON} in {time.perf_counter()-t0:.2f}s", flush=True)
    print("  FE OLS log1p(abundance) [is_hub, topo_depth, hamming]:", fe_ab_topo, flush=True)
    print("  FE OLS delta_bind      [is_hub, topo_depth, hamming]:", fe_db_topo, flush=True)
    print("  Matched (GC,depth) mouse abundance diff:", controls_payload["empirical_gcreplay_depth_adjusted_controls"]["exact_gc_and_depth_matched_paired_abundance_mouse"], flush=True)
    print("  Matched (GC,depth) mouse delta_bind diff:", controls_payload["empirical_gcreplay_depth_adjusted_controls"]["exact_gc_and_depth_matched_paired_delta_bind_mouse"], flush=True)


if __name__ == "__main__":
    main()
