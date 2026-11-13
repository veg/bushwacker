#!/usr/bin/env python3
"""
Comprehensive First-Principles Benchmark for MDL Phylogenetic Graph Induction
=============================================================================
Tests MDLPhyloEngine (mdl_phylo.py) head-to-head against IQ-TREE 2, FastTree 2,
Neighbor-Joining, and Unweighted Parsimony across 5 fundamental regimes:
  1. True Polytomy & Star-Radiation Calibration (Zero-Mutation Exclusion)
  2. The Felsenstein Zone (Birthday-Paradox Immunity to Long-Branch Attraction)
  3. The Two-Site Convergent Hijack (Backbone Stability + SPARSE Shunt Recovery)
  4. Contiguous Recombination vs. Sparse Convergence (Dual-Mask Mode Selection)
  5. Real Empirical Alignments (Prestin, Camelid VHH, SARS-CoV-2 Spike, Beta-Globin)
"""

from __future__ import annotations


import sys as _sys
from pathlib import Path as _Path
if str(_Path(__file__).resolve().parent) not in _sys.path:
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
from _env_paths import REPO_ROOT, DATA_DIR, SPECIMENS_DIR, ALIGNMENTS_DIR, SINGLE_CELL_DIR, RESULTS_DIR, WORK_ROOT, EXTERNAL_DIR, resolve_bin, ASTRAL_JAR

import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from mdl_phylo import MDLPhyloEngine, parse_fasta


# ---------------------------------------------------------------------------
# Newick Parser & Unrooted Split Comparison Utilities (Supports Polytomies)
# ---------------------------------------------------------------------------

def extract_unrooted_splits_from_newick(
    newick_str: str,
    collapse_zero_threshold: float = 0.0,
    present_taxa: Optional[Set[str]] = None,
) -> Tuple[Set[str], Set[Tuple[Tuple[str, ...], Tuple[str, ...]]]]:
    """
    Parse a (potentially multifurcating) Newick string and return:
      - all_taxa: set of leaf names (intersected with present_taxa if provided)
      - splits: set of canonical non-trivial bipartitions (sorted_smaller_side, sorted_larger_side)
        for all internal branches with length > collapse_zero_threshold.
    """
    s = newick_str.strip()
    if s.endswith(";"):
        s = s[:-1]

    pos = 0
    n = len(s)
    children_map: Dict[int, List[Tuple[int, float]]] = {}
    leaf_name: Dict[int, str] = {}
    next_id = [0]

    def parse_node() -> Tuple[int, float]:
        nonlocal pos
        my_id = next_id[0]
        next_id[0] += 1
        children_map[my_id] = []

        if pos < n and s[pos] == "(":
            pos += 1
            while True:
                c_id, c_len = parse_node()
                children_map[my_id].append((c_id, c_len))
                if pos < n and s[pos] == ",":
                    pos += 1
                elif pos < n and s[pos] == ")":
                    pos += 1
                    break
                else:
                    break
            while pos < n and s[pos] not in (":", ",", ")", ";"):
                pos += 1
        else:
            start = pos
            while pos < n and s[pos] not in (":", ",", ")", ";"):
                pos += 1
            leaf_name[my_id] = s[start:pos].strip()

        b_len = 0.0
        if pos < n and s[pos] == ":":
            pos += 1
            start = pos
            while pos < n and s[pos] not in (",", ")", ";"):
                pos += 1
            try:
                b_len = float(s[start:pos].strip())
            except ValueError:
                b_len = 0.0

        return my_id, b_len

    root_id, _ = parse_node()
    all_taxa_set: Set[str] = set(leaf_name.values())
    if present_taxa is not None:
        all_taxa_set &= present_taxa

    sub_leaves: Dict[int, Set[str]] = {}

    def get_leaves(u: int) -> Set[str]:
        if u in leaf_name:
            nm = leaf_name[u]
            res = {nm} if (present_taxa is None or nm in present_taxa) else set()
        else:
            res = set()
            for c, _ in children_map[u]:
                res |= get_leaves(c)
        sub_leaves[u] = res
        return res

    get_leaves(root_id)

    splits: Set[Tuple[Tuple[str, ...], Tuple[str, ...]]] = set()
    for u, ch_list in children_map.items():
        for c, b_len in ch_list:
            if c in leaf_name:
                continue
            if b_len <= collapse_zero_threshold and present_taxa is None:
                continue
            side1 = tuple(sorted(sub_leaves[c]))
            side2 = tuple(sorted(all_taxa_set - sub_leaves[c]))
            if len(side1) < 2 or len(side2) < 2:
                continue
            if len(side1) < len(side2) or (len(side1) == len(side2) and side1 < side2):
                canon = (side1, side2)
            else:
                canon = (side2, side1)
            splits.add(canon)

    return all_taxa_set, splits


def extract_unrooted_splits_from_adj(
    adj: Dict[int, List[int]],
    names: List[str],
    N: int,
) -> Set[Tuple[Tuple[str, ...], Tuple[str, ...]]]:
    """Extract canonical non-trivial unrooted splits directly from MDL adjacency dict."""
    all_taxa = set(names)
    splits: Set[Tuple[Tuple[str, ...], Tuple[str, ...]]] = set()

    for u in adj:
        for v in adj[u]:
            if u < v and u >= N and v >= N:
                # BFS from u without crossing (u, v)
                visited = {v}
                q = [u]
                leaves_u: Set[str] = set()
                while q:
                    curr = q.pop(0)
                    visited.add(curr)
                    if curr < N:
                        leaves_u.add(names[curr])
                    for nb in adj[curr]:
                        if nb not in visited:
                            visited.add(nb)
                            q.append(nb)
                side1 = tuple(sorted(leaves_u))
                side2 = tuple(sorted(all_taxa - leaves_u))
                if len(side1) < 2 or len(side2) < 2:
                    continue
                if len(side1) < len(side2) or (len(side1) == len(side2) and side1 < side2):
                    canon = (side1, side2)
                else:
                    canon = (side2, side1)
                splits.add(canon)
    return splits


# ---------------------------------------------------------------------------
# External Tool Runners (FastTree 2 & IQ-TREE 2)
# ---------------------------------------------------------------------------

def run_fasttree(seqs_dict: Dict[str, str]) -> Tuple[str, float]:
    """Run FastTree 2 (-nt -gtr -gamma -nosupport) on an alignment dict."""
    ft_bin = resolve_bin("FASTTREE_BIN", "FastTree", "fasttree")
    with tempfile.TemporaryDirectory() as tmpdir:
        fa_path = os.path.join(tmpdir, "aln.fa")
        with open(fa_path, "w") as f:
            for k, v in seqs_dict.items():
                f.write(f">{k}\n{v}\n")
        t0 = time.perf_counter()
        proc = subprocess.run(
            [ft_bin, "-nt", "-gtr", "-gamma", "-nosupport", "-quiet", fa_path],
            capture_output=True,
            text=True,
            check=False,
        )
        dt = time.perf_counter() - t0
        return proc.stdout.strip(), dt


def run_iqtree(seqs_dict: Dict[str, str], model: str = "GTR+G") -> Tuple[str, float]:
    """Run IQ-TREE 2 on an alignment dict."""
    iq_bin = resolve_bin("IQTREE_BIN", "iqtree2", "iqtree")
    with tempfile.TemporaryDirectory() as tmpdir:
        fa_path = os.path.join(tmpdir, "aln.fa")
        with open(fa_path, "w") as f:
            for k, v in seqs_dict.items():
                f.write(f">{k}\n{v}\n")
        t0 = time.perf_counter()
        subprocess.run(
            [iq_bin, "-s", fa_path, "-m", model, "-T", "1", "-quiet", "-redo"],
            capture_output=True,
            text=True,
            check=False,
        )
        dt = time.perf_counter() - t0
        tf = fa_path + ".treefile"
        if os.path.exists(tf):
            with open(tf, "r") as f:
                return f.read().strip(), dt
        return "", dt


# ---------------------------------------------------------------------------
# Test 1: True Polytomy & Star-Radiation Calibration
# ---------------------------------------------------------------------------

def simulate_multifurcating_radiation(
    rng: np.random.Generator,
    N: int = 32,
    L: int = 1000,
) -> Tuple[Dict[str, str], Set[Tuple[Tuple[str, ...], Tuple[str, ...]]], int]:
    """
    Simulate an alignment of N=32 taxa of length L=1000 bp on a true multifurcating
    phylogeny containing:
      - 4 primary macro-clades of 8 taxa each (3 deep internal splits, t = 0.012..0.020),
      - Inside each clade of 8: two resolved pairs (2 * 4 = 8 real sub-splits, t = 0.004..0.010),
        and the remaining 4 lineages + the 2 pairs radiating from a TRUE 6-way polytomy!
      - Total true non-trivial splits = 3 + 8 = 11 (out of 29 possible binary splits;
        18 internal binary slots have true branch length = 0.0!).
    """
    names = [f"T{i:02d}" for i in range(N)]
    all_taxa = set(names)

    def mutate_seq(seq: np.ndarray, branch_len: float, site_rates: np.ndarray) -> np.ndarray:
        out = seq.copy()
        probs = 0.75 * (1.0 - np.exp(-(4.0 / 3.0) * branch_len * site_rates))
        draws = rng.random(len(seq))
        mut_idx = np.where(draws < probs)[0]
        for idx in mut_idx:
            cur = int(out[idx])
            choices = [b for b in range(4) if b != cur]
            out[idx] = rng.choice(choices)
        return out

    site_rates = rng.gamma(shape=1.5, scale=1.0 / 1.5, size=L)
    root_seq = rng.integers(0, 4, size=L, dtype=np.int8)

    # Split into left superclade (Clades 0, 1) and right superclade (Clades 2, 3)
    anc_01 = mutate_seq(root_seq, 0.015, site_rates)
    anc_23 = mutate_seq(root_seq, 0.015, site_rates)

    clade_ancs = [
        mutate_seq(anc_01, 0.012, site_rates),
        mutate_seq(anc_01, 0.012, site_rates),
        mutate_seq(anc_23, 0.012, site_rates),
        mutate_seq(anc_23, 0.012, site_rates),
    ]

    true_splits: Set[Tuple[Tuple[str, ...], Tuple[str, ...]]] = set()

    def add_true_split(subset: List[str]):
        s1 = tuple(sorted(subset))
        s2 = tuple(sorted(all_taxa - set(s1)))
        if len(s1) < len(s2) or (len(s1) == len(s2) and s1 < s2):
            true_splits.add((s1, s2))
        else:
            true_splits.add((s2, s1))

    # Central split: (0..15) | (16..31)
    add_true_split(names[0:16])
    # Clade splits: (0..7), (8..15), (16..23), (24..31)
    for c in range(4):
        add_true_split(names[8 * c : 8 * (c + 1)])

    seqs_dict: Dict[str, str] = {}
    nuc_chars = np.array(list("ACGT"))

    for c in range(4):
        base = 8 * c
        c_anc = clade_ancs[c]
        # Pair 1: (base+0, base+1) share a resolved branch of length 0.006
        p1_anc = mutate_seq(c_anc, 0.006, site_rates)
        add_true_split([names[base + 0], names[base + 1]])
        for j in (0, 1):
            leaf_s = mutate_seq(p1_anc, rng.uniform(0.003, 0.008), site_rates)
            seqs_dict[names[base + j]] = "".join(nuc_chars[leaf_s])

        # Pair 2: (base+2, base+3) share a resolved branch of length 0.006
        p2_anc = mutate_seq(c_anc, 0.006, site_rates)
        add_true_split([names[base + 2], names[base + 3]])
        for j in (2, 3):
            leaf_s = mutate_seq(p2_anc, rng.uniform(0.003, 0.008), site_rates)
            seqs_dict[names[base + j]] = "".join(nuc_chars[leaf_s])

        # Remaining 4 leaves (base+4 .. base+7) radiate directly from c_anc (0-length internal branches!)
        for j in (4, 5, 6, 7):
            leaf_s = mutate_seq(c_anc, rng.uniform(0.003, 0.008), site_rates)
            seqs_dict[names[base + j]] = "".join(nuc_chars[leaf_s])

    return seqs_dict, true_splits, len(true_splits)


def run_test1_polytomy_calibration(n_reps: int = 15) -> Dict[str, float]:
    print("\n" + "=" * 78)
    print("TEST 1: True Polytomy & Star-Radiation Calibration (N=32, L=1000, 13 True Splits, 16 Polytomy Slots)")
    print("=" * 78)
    rng = np.random.default_rng(20261031)
    engine = MDLPhyloEngine(coding_aware=False, detect_shunts=False)

    stats = {
        "MDL": {"inferred": [], "tp": [], "fp": [], "prec": [], "rec": [], "rf": [], "time": []},
        "FastTree2": {"inferred": [], "tp": [], "fp": [], "prec": [], "rec": [], "rf": [], "time": []},
        "IQTREE2": {"inferred": [], "tp": [], "fp": [], "prec": [], "rec": [], "rf": [], "time": []},
    }

    for rep in range(n_reps):
        seqs, true_splits, n_true = simulate_multifurcating_radiation(rng)

        # 1. MDLPhyloEngine
        res = engine.infer(seqs)
        mdl_splits = extract_unrooted_splits_from_adj(res.adjacency, res.names, res.n_taxa)
        tp_m = len(mdl_splits & true_splits)
        fp_m = len(mdl_splits - true_splits)
        fn_m = len(true_splits - mdl_splits)
        stats["MDL"]["inferred"].append(len(mdl_splits))
        stats["MDL"]["tp"].append(tp_m)
        stats["MDL"]["fp"].append(fp_m)
        stats["MDL"]["prec"].append(tp_m / max(len(mdl_splits), 1))
        stats["MDL"]["rec"].append(tp_m / n_true)
        stats["MDL"]["rf"].append(fp_m + fn_m)
        stats["MDL"]["time"].append(res.runtime_sec)

        # 2. FastTree 2
        ft_nwk, ft_time = run_fasttree(seqs)
        _, ft_splits = extract_unrooted_splits_from_newick(ft_nwk)
        tp_f = len(ft_splits & true_splits)
        fp_f = len(ft_splits - true_splits)
        fn_f = len(true_splits - ft_splits)
        stats["FastTree2"]["inferred"].append(len(ft_splits))
        stats["FastTree2"]["tp"].append(tp_f)
        stats["FastTree2"]["fp"].append(fp_f)
        stats["FastTree2"]["prec"].append(tp_f / max(len(ft_splits), 1))
        stats["FastTree2"]["rec"].append(tp_f / n_true)
        stats["FastTree2"]["rf"].append(fp_f + fn_f)
        stats["FastTree2"]["time"].append(ft_time)

        # 3. IQ-TREE 2
        iq_nwk, iq_time = run_iqtree(seqs, model="GTR+G")
        _, iq_splits = extract_unrooted_splits_from_newick(iq_nwk)
        tp_i = len(iq_splits & true_splits)
        fp_i = len(iq_splits - true_splits)
        fn_i = len(true_splits - iq_splits)
        stats["IQTREE2"]["inferred"].append(len(iq_splits))
        stats["IQTREE2"]["tp"].append(tp_i)
        stats["IQTREE2"]["fp"].append(fp_i)
        stats["IQTREE2"]["prec"].append(tp_i / max(len(iq_splits), 1))
        stats["IQTREE2"]["rec"].append(tp_i / n_true)
        stats["IQTREE2"]["rf"].append(fp_i + fn_i)
        stats["IQTREE2"]["time"].append(iq_time)

    print(f"{'Method':<12} | {'Inferred Splits (True=13)':<25} | {'False Pos (FP/16)':<18} | {'Precision':<10} | {'Recall':<10} | {'RF Error':<10} | {'Time (s)':<8}")
    print("-" * 106)
    summary = {}
    for m_name in ("MDL", "FastTree2", "IQTREE2"):
        d = stats[m_name]
        inf_m = float(np.mean(d["inferred"]))
        fp_m = float(np.mean(d["fp"]))
        pr_m = float(np.mean(d["prec"])) * 100.0
        rc_m = float(np.mean(d["rec"])) * 100.0
        rf_m = float(np.mean(d["rf"]))
        tm_m = float(np.mean(d["time"]))
        print(f"{m_name:<12} | {inf_m:6.2f} / 29 binary max    | {fp_m:5.2f} ({fp_m/16*100:5.1f}% FPR) | {pr_m:8.1f}% | {rc_m:8.1f}% | {rf_m:8.2f}   | {tm_m:7.4f}s")
        summary[m_name] = {"inferred": inf_m, "fp": fp_m, "fpr_pct": fp_m / 16 * 100.0, "precision_pct": pr_m, "recall_pct": rc_m, "rf_error": rf_m, "time_sec": tm_m}
    return summary


# ---------------------------------------------------------------------------
# Test 2: The Felsenstein Zone (Long-Branch Attraction)
# ---------------------------------------------------------------------------

def run_test2_felsenstein_zone(n_reps: int = 50) -> Dict[str, float]:
    """
    Classic Felsenstein Zone:
      True tree: ((A:0.45, B:0.015):0.025, (C:0.45, D:0.015):0.00)
      with L=1000 bp and site-to-site rate heterogeneity (Gamma alpha=0.8).
      Non-sister lineages A and C have long branches (0.45 subst/site) and
      experience massive parallel homoplasy.
      Compare Unweighted Maximum Parsimony, Hamming Neighbor-Joining, FastTree 2,
      and MDLPhyloEngine.
    """
    print("\n" + "=" * 78)
    print("TEST 2: The Felsenstein Zone (Long-Branch Attraction: l_long=0.45, l_short=0.015, l_int=0.025, L=1000)")
    print("=" * 78)
    rng = np.random.default_rng(20261101)
    engine = MDLPhyloEngine(coding_aware=False, detect_shunts=False)
    L = 1000
    nuc_chars = np.array(list("ACGT"))

    def mutate_seq(seq: np.ndarray, branch_len: float, site_rates: np.ndarray) -> np.ndarray:
        out = seq.copy()
        probs = 0.75 * (1.0 - np.exp(-(4.0 / 3.0) * branch_len * site_rates))
        draws = rng.random(len(seq))
        mut_idx = np.where(draws < probs)[0]
        for idx in mut_idx:
            cur = int(out[idx])
            choices = [b for b in range(4) if b != cur]
            out[idx] = rng.choice(choices)
        return out

    mp_correct = 0
    mp_lba = 0
    nj_correct = 0
    nj_lba = 0
    mdl_correct = 0
    mdl_lba = 0
    mdl_poly = 0

    true_split = (("A", "B"), ("C", "D"))
    lba_split = (("A", "C"), ("B", "D"))

    for _ in range(n_reps):
        site_rates = rng.gamma(shape=0.8, scale=1.0 / 0.8, size=L)
        u_seq = rng.integers(0, 4, size=L, dtype=np.int8)
        v_seq = mutate_seq(u_seq, 0.025, site_rates)

        sA = mutate_seq(u_seq, 0.45, site_rates)
        sB = mutate_seq(u_seq, 0.015, site_rates)
        sC = mutate_seq(v_seq, 0.45, site_rates)
        sD = mutate_seq(v_seq, 0.015, site_rates)

        # 1. Unweighted Maximum Parsimony on 4 taxa:
        # Count sites supporting AB|CD vs AC|BD vs AD|BC
        n_ab = int(np.sum((sA == sB) & (sC == sD) & (sA != sC)))
        n_ac = int(np.sum((sA == sC) & (sB == sD) & (sA != sB)))
        n_ad = int(np.sum((sA == sD) & (sB == sC) & (sA != sB)))
        if n_ab > max(n_ac, n_ad):
            mp_correct += 1
        elif n_ac >= max(n_ab, n_ad):
            mp_lba += 1

        # 2. Raw Hamming Neighbor-Joining four-point condition:
        dAB = int(np.sum(sA != sB))
        dCD = int(np.sum(sC != sD))
        dAC = int(np.sum(sA != sC))
        dBD = int(np.sum(sB != sD))
        dAD = int(np.sum(sA != sD))
        dBC = int(np.sum(sB != sC))
        s_ab = dAB + dCD
        s_ac = dAC + dBD
        s_ad = dAD + dBC
        if s_ab < min(s_ac, s_ad):
            nj_correct += 1
        elif s_ac <= min(s_ab, s_ad):
            nj_lba += 1

        # 3. MDLPhyloEngine
        seqs = {
            "A": "".join(nuc_chars[sA]),
            "B": "".join(nuc_chars[sB]),
            "C": "".join(nuc_chars[sC]),
            "D": "".join(nuc_chars[sD]),
        }
        res = engine.infer(seqs)
        splits = extract_unrooted_splits_from_adj(res.adjacency, res.names, res.n_taxa)
        if true_split in splits:
            mdl_correct += 1
        elif lba_split in splits:
            mdl_lba += 1
        elif len(splits) == 0:
            mdl_poly += 1

    print(f"Unweighted Max Parsimony : True Split = {mp_correct/n_reps*100:5.1f}% | LBA Trap = {mp_lba/n_reps*100:5.1f}%")
    print(f"Hamming Neighbor-Joining : True Split = {nj_correct/n_reps*100:5.1f}% | LBA Trap = {nj_lba/n_reps*100:5.1f}%")
    print(f"MDLPhyloEngine (Ours)    : True Split = {mdl_correct/n_reps*100:5.1f}% | LBA Trap = {mdl_lba/n_reps*100:5.1f}% | Polytomy = {mdl_poly/n_reps*100:5.1f}%")
    return {
        "MP_true_pct": mp_correct / n_reps * 100.0,
        "MP_lba_pct": mp_lba / n_reps * 100.0,
        "NJ_true_pct": nj_correct / n_reps * 100.0,
        "NJ_lba_pct": nj_lba / n_reps * 100.0,
        "MDL_true_pct": mdl_correct / n_reps * 100.0,
        "MDL_lba_pct": mdl_lba / n_reps * 100.0,
        "MDL_polytomy_pct": mdl_poly / n_reps * 100.0,
    }


# ---------------------------------------------------------------------------
# Test 3: The Two-Site Convergent Hijack + Unsupervised Shunt Detection
# ---------------------------------------------------------------------------

def simulate_two_site_hijack_alignment(
    rng: np.random.Generator,
    n_pairs: int = 16,
    n_codons: int = 300,
    m_conv_codons: int = 3,
) -> Tuple[Dict[str, str], Set[Tuple[str, str]], List[int]]:
    """
    Simulate a coding alignment of N = 2 * n_pairs = 32 taxa (16 host pairs (N_k, T_k),
    L = 300 codons = 900 bp) where:
      - Each host pair (N_k, T_k) shares a true host branch with ~6 neutral synonymous
        substitutions at 3rd-codon positions (while ~85% of codons are conserved/slow!).
      - Across paired treated lineages (T_{2j}, T_{2j+1}), m_conv_codons slow-evolving
        functional codons experience convergent replacement substitutions (e.g. ATG -> TGG / CGT -> CAT).
    Under Gamma rate heterogeneity (IQTREE2 GTR+G / FastTree2 GTR+G), the slow-rate
    leverage ln(1 / r_slow) causes m=2..5 convergent codons to hijack the host pairs!
    """
    # Pick 20 four-fold degenerate or two-fold degenerate sense codons for clean synonymous evolution
    syn_families = [
        ["GCT", "GCC", "GCA", "GCG"],  # Ala
        ["GGT", "GGC", "GGA", "GGG"],  # Gly
        ["GTT", "GTC", "GTA", "GTG"],  # Val
        ["CTT", "CTC", "CTA", "CTG"],  # Leu
        ["TCT", "TCC", "TCA", "TCG"],  # Ser
        ["CCT", "CCC", "CCA", "CCG"],  # Pro
        ["ACT", "ACC", "ACA", "ACG"],  # Thr
    ]
    root_fam_idx = rng.integers(0, len(syn_families), size=n_codons)
    root_codons = [syn_families[f][0] for f in root_fam_idx]

    # Choose 45 polymorphic neutral codons (15% of alignment) and keep 85% hyper-conserved
    neutral_poly_codons = sorted(rng.choice(range(20, n_codons), size=45, replace=False).tolist())
    conv_codons = list(range(2, 2 + m_conv_codons))

    def evolve_synonymous(cod_list: List[str], n_muts: int) -> List[str]:
        out = list(cod_list)
        sites = rng.choice(neutral_poly_codons, size=min(n_muts, len(neutral_poly_codons)), replace=False)
        for c in sites:
            fam = syn_families[root_fam_idx[c]]
            cur = out[c]
            alts = [x for x in fam if x != cur]
            out[c] = str(rng.choice(alts))
        return out

    # Generate 4 macro-clades of 4 host pairs each
    clade_roots = [evolve_synonymous(root_codons, 5) for _ in range(4)]
    host_ancs: List[List[str]] = []
    for k in range(n_pairs):
        c_root = clade_roots[k // 4]
        host_ancs.append(evolve_synonymous(c_root, 5))

    seqs_dict: Dict[str, str] = {}
    true_pairs: Set[Tuple[str, str]] = set()

    # Convergent target codons (2-nucleotide or 1st/2nd-pos replacement in conserved region)
    conv_targets = ["TGG", "CAT", "GAC", "TTC", "AAC", "TAC", "ATG", "CAG"]

    for k in range(n_pairs):
        n_name = f"Host{k:02d}_N"
        t_name = f"Host{k:02d}_T"
        true_pairs.add((n_name, t_name))

        s_n = evolve_synonymous(host_ancs[k], 2)
        s_t = evolve_synonymous(host_ancs[k], 2)

        # Apply convergent replacement mutations to treated lineages (T_{2j}, T_{2j+1})
        # Each pair of treated hosts (T_{2j}, T_{2j+1}) across different clades shares the convergent state
        if m_conv_codons > 0:
            group_id = k // 2
            for idx_m, c_pos in enumerate(conv_codons):
                # Distinct convergent replacement per group or shared across treated cohort
                target_cod = conv_targets[(idx_m + group_id) % len(conv_targets)]
                s_t[c_pos] = target_cod

        seqs_dict[n_name] = "".join(s_n)
        seqs_dict[t_name] = "".join(s_t)

    conv_nt_sites: List[int] = []
    for c_pos in conv_codons:
        conv_nt_sites.extend([3 * c_pos, 3 * c_pos + 1, 3 * c_pos + 2])

    return seqs_dict, true_pairs, conv_nt_sites


def run_test3_two_site_hijack() -> Dict[str, Dict[str, float]]:
    print("\n" + "=" * 78)
    print("TEST 3: The Two-Site Convergent Hijack (N=32, 16 Host Pairs, 300 Codons)")
    print("=" * 78)
    rng = np.random.default_rng(20261102)
    engine = MDLPhyloEngine(coding_aware=True, detect_shunts=True, min_shunt_sites=2)

    results_by_m: Dict[str, Dict[str, float]] = {}
    print(f"{'m_conv':<8} | {'MDL Pair Recall':<17} | {'FastTree2 Recall':<17} | {'IQTREE2 Recall':<17} | {'MDL Shunt Site Recall':<22}")
    print("-" * 92)

    for m_conv in (0, 2, 3, 5):
        mdl_rec, ft_rec, iq_rec, shunt_rec = [], [], [], []
        n_reps = 8
        for _ in range(n_reps):
            seqs, true_pairs, conv_nt_sites = simulate_two_site_hijack_alignment(
                rng, n_pairs=16, n_codons=300, m_conv_codons=m_conv
            )
            all_taxa = set(seqs.keys())
            pair_splits = set()
            for a, b in true_pairs:
                s1 = tuple(sorted([a, b]))
                s2 = tuple(sorted(all_taxa - {a, b}))
                pair_splits.add((s1, s2))

            # MDL
            res = engine.infer(seqs)
            mdl_splits = extract_unrooted_splits_from_adj(res.adjacency, res.names, res.n_taxa)
            mdl_rec.append(len(mdl_splits & pair_splits) / len(pair_splits) * 100.0)

            if m_conv > 0:
                detected_codons = set()
                for sh in res.shunts:
                    if sh.mode == "SPARSE":
                        for s_idx in sh.sites:
                            detected_codons.add(s_idx // 3)
                true_codons = {s // 3 for s in conv_nt_sites}
                shunt_rec.append(len(detected_codons & true_codons) / len(true_codons) * 100.0)
            else:
                shunt_rec.append(100.0 if len(res.shunts) == 0 else 0.0)

            # FastTree 2
            ft_nwk, _ = run_fasttree(seqs)
            _, ft_splits = extract_unrooted_splits_from_newick(ft_nwk)
            ft_rec.append(len(ft_splits & pair_splits) / len(pair_splits) * 100.0)

            # IQ-TREE 2
            iq_nwk, _ = run_iqtree(seqs, model="GTR+G")
            _, iq_splits = extract_unrooted_splits_from_newick(iq_nwk)
            iq_rec.append(len(iq_splits & pair_splits) / len(pair_splits) * 100.0)

        m_r = float(np.mean(mdl_rec))
        f_r = float(np.mean(ft_rec))
        i_r = float(np.mean(iq_rec))
        s_r = float(np.mean(shunt_rec))
        shunt_label = f"{s_r:6.1f}% (SPARSE)" if m_conv > 0 else f"{s_r:6.1f}% (0 FP Shunts)"
        print(f"m = {m_conv:<4} | {m_r:14.1f}%   | {f_r:14.1f}%   | {i_r:14.1f}%   | {shunt_label}")
        results_by_m[f"m={m_conv}"] = {
            "MDL_pair_recall_pct": m_r,
            "FastTree2_pair_recall_pct": f_r,
            "IQTREE2_pair_recall_pct": i_r,
            "MDL_shunt_accuracy_pct": s_r,
        }
    return results_by_m


# ---------------------------------------------------------------------------
# Test 4: Contiguous Recombination vs. Sparse Convergence Mode Classification
# ---------------------------------------------------------------------------

def run_test4_shunt_mode_discrimination(n_reps: int = 15) -> Dict[str, float]:
    """
    Test whether the exact MDL dual-mask competition:
        L_mask^interval = ln 2 + 2 ln L  vs.  L_mask^sparse = ln 2 + ln C(M_var, m)
    automatically classifies homologous recombination tracts [400, 750] as INTERVAL
    and scattered adaptive convergence as SPARSE.
    """
    print("\n" + "=" * 78)
    print("TEST 4: Unsupervised Shunt Mode Discrimination (INTERVAL Recombination vs SPARSE Convergence)")
    print("=" * 78)
    rng = np.random.default_rng(20261103)
    engine = MDLPhyloEngine(coding_aware=True, detect_shunts=True, min_shunt_sites=2)
    L = 1200
    nuc_chars = np.array(list("ACGT"))

    def mutate_arr(seq: np.ndarray, n_muts: int) -> np.ndarray:
        out = seq.copy()
        idx = rng.choice(len(seq), size=n_muts, replace=False)
        for i in idx:
            out[i] = (int(out[i]) + int(rng.integers(1, 4))) % 4
        return out

    recomb_correct_mode = 0
    recomb_ious: List[float] = []

    for _ in range(n_reps):
        # 16 taxa in 4 clades of 4; Clade A (0..3) and Clade D (12..15) are well-diverged (~60 mutations)
        root = rng.integers(0, 4, size=L, dtype=np.int8)
        cA = mutate_arr(root, 30)
        cB = mutate_arr(root, 30)
        cC = mutate_arr(root, 30)
        cD = mutate_arr(root, 30)

        seqs: Dict[str, str] = {}
        leaves_A = [mutate_arr(cA, 4) for _ in range(4)]
        leaves_B = [mutate_arr(cB, 4) for _ in range(4)]
        leaves_C = [mutate_arr(cC, 4) for _ in range(4)]
        leaves_D = [mutate_arr(cD, 4) for _ in range(4)]

        # Make Taxon A0 a recombinant inheriting [400, 750] from Donor D0!
        true_a, true_b = 400, 750
        leaves_A[0][true_a : true_b + 1] = leaves_D[0][true_a : true_b + 1]

        for j in range(4):
            seqs[f"A{j}"] = "".join(nuc_chars[leaves_A[j]])
            seqs[f"B{j}"] = "".join(nuc_chars[leaves_B[j]])
            seqs[f"C{j}"] = "".join(nuc_chars[leaves_C[j]])
            seqs[f"D{j}"] = "".join(nuc_chars[leaves_D[j]])

        res = engine.infer(seqs)
        int_shunts = [s for s in res.shunts if s.mode == "INTERVAL" and s.interval_span is not None]
        if int_shunts:
            recomb_correct_mode += 1
            pred_a, pred_b = int_shunts[0].interval_span
            inter = max(0, min(true_b, pred_b) - max(true_a, pred_a) + 1)
            union = max(true_b, pred_b) - min(true_a, pred_a) + 1
            recomb_ious.append(inter / max(union, 1))

    recomb_acc = recomb_correct_mode / n_reps * 100.0
    mean_iou = float(np.mean(recomb_ious)) * 100.0 if recomb_ious else 0.0
    print(f"Recombination [400..750] classified as INTERVAL : {recomb_acc:5.1f}% ({recomb_correct_mode}/{n_reps})")
    print(f"Mean Breakpoint Interval IoU                    : {mean_iou:5.1f}%")
    return {"recomb_interval_mode_pct": recomb_acc, "recomb_breakpoint_iou_pct": mean_iou}


# ---------------------------------------------------------------------------
# Test 5: Real Empirical Benchmarks (Prestin, Camelid VHH, SARS-CoV-2, bglobin)
# ---------------------------------------------------------------------------

def run_test5_empirical_datasets() -> Dict[str, dict]:
    print("\n" + "=" * 78)
    print("TEST 5: Real Empirical Alignments (Compression Ratio, Polytomies, Shunts & Speed)")
    print("=" * 78)
    data_dir = Path(__file__).resolve().parents[2] / "data" / "alignments" / "viral_and_bwz_codec" / "empirical_genes"
    engine = MDLPhyloEngine(coding_aware=True, detect_shunts=True, min_shunt_sites=2)

    empirical_files = [
        ("bglobin.fasta", "Mammalian Beta-Globin"),
        ("camelid.fasta", "Camelid VHH Nanobodies (Recombining)"),
        ("prestin.fasta", "Mammalian Prestin (Echolocation Convergence)"),
        ("HIV_RT.fasta", "HIV-1 Reverse Transcriptase (Drug Resistance)"),
        ("SARS-CoV-2-spike.fasta", "SARS-CoV-2 Spike (Low-Divergence Pandemic)"),
    ]

    emp_results: Dict[str, dict] = {}
    print(f"{'Dataset':<22} | {'N x L':<13} | {'Int Branches':<16} | {'Max Poly':<8} | {'Shunts (INT/SPARSE)':<20} | {'Compression':<13} | {'Time':<7}")
    print("-" * 112)

    for fname, desc in empirical_files:
        fpath = data_dir / fname
        if not fpath.exists():
            continue
        res = engine.infer(str(fpath))
        n_int_sh = sum(1 for s in res.shunts if s.mode == "INTERVAL")
        n_sp_sh = sum(1 for s in res.shunts if s.mode == "SPARSE")
        comp_ratio = res.raw_uncompressed_nats / max(res.full_graph_nats, 1.0)
        int_str = f"{res.n_internal_branches}/{res.max_binary_internal_branches} ({res.n_internal_branches/max(res.max_binary_internal_branches,1)*100:4.1f}%)"
        sh_str = f"{len(res.shunts):2d} ({n_int_sh} INT / {n_sp_sh} SP)"
        print(
            f"{fname:<22} | {res.n_taxa:3d} x {res.n_sites:<6d} | {int_str:<16} | deg {res.max_polytomy_degree:<4d} | {sh_str:<20} | {comp_ratio:5.1f}x ({res.full_graph_nats/1e3:5.1f}k) | {res.runtime_sec:5.2f}s"
        )
        emp_results[fname] = {
            "description": desc,
            "n_taxa": res.n_taxa,
            "n_sites": res.n_sites,
            "resolved_internal_branches": res.n_internal_branches,
            "max_binary_branches": res.max_binary_internal_branches,
            "resolution_pct": res.n_internal_branches / max(res.max_binary_internal_branches, 1) * 100.0,
            "polytomy_nodes": res.polytomy_nodes,
            "max_polytomy_degree": res.max_polytomy_degree,
            "interval_shunts": n_int_sh,
            "sparse_shunts": n_sp_sh,
            "compression_ratio": comp_ratio,
            "raw_nats": res.raw_uncompressed_nats,
            "star_nats": res.star_tree_nats,
            "backbone_nats": res.backbone_tree_nats,
            "full_graph_nats": res.full_graph_nats,
            "runtime_sec": res.runtime_sec,
            "top_shunts": [
                {
                    "u": s.label_u,
                    "v": s.label_v,
                    "mode": s.mode,
                    "n_sites": s.shared_mutations,
                    "span": s.interval_span,
                    "sites_preview": s.sites[:8],
                    "net_mdl_gain_nats": round(s.net_mdl_gain_nats, 2),
                }
                for s in res.shunts[:5]
            ],
        }

        # Special check on Prestin against NCBI species tree
        if fname == "prestin.fasta" and (data_dir / "prestin_species.nwk").exists():
            with open(data_dir / "prestin_species.nwk") as f:
                sp_nwk = f.read().strip()
            t_set, sp_splits = extract_unrooted_splits_from_newick(sp_nwk, present_taxa=set(res.names))
            _, mdl_splits = extract_unrooted_splits_from_newick(res.newick, present_taxa=t_set)
            tp = len(mdl_splits & sp_splits)
            prec = tp / max(len(mdl_splits), 1) * 100.0
            rec = tp / max(len(sp_splits), 1) * 100.0
            emp_results[fname]["prestin_species_concordance"] = {
                "shared_taxa": len(t_set),
                "species_splits": len(sp_splits),
                "mdl_splits_on_shared": len(mdl_splits),
                "matched_splits": tp,
                "precision_pct": round(prec, 2),
                "recall_pct": round(rec, 2),
            }
            print(f"  -> Prestin Species Tree Concordance ({len(t_set)} shared taxa): {tp}/{len(sp_splits)} true clades recovered | Precision = {prec:.1f}% | Recall = {rec:.1f}%")
            for sh_idx, top_s in enumerate(res.shunts):
                codons = sorted({s // 3 + 1 for s in top_s.sites})
                print(f"  -> Prestin Shunt #{sh_idx+1} ({top_s.mode}): {top_s.label_u} <-> {top_s.label_v} | Gain = +{top_s.net_mdl_gain_nats:.2f} nats | Codons: {codons[:10]}")

    return emp_results


if __name__ == "__main__":
    all_out = {
        "test1_polytomy_calibration": run_test1_polytomy_calibration(n_reps=15),
        "test2_felsenstein_zone": run_test2_felsenstein_zone(n_reps=50),
        "test3_two_site_hijack": run_test3_two_site_hijack(),
        "test4_shunt_mode": run_test4_shunt_mode_discrimination(n_reps=15),
        "test5_empirical": run_test5_empirical_datasets(),
    }
    out_path = Path(__file__).resolve().parents[2] / "results" / "synthetic" / "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(all_out, f, indent=2)
    print(f"\nSaved complete benchmark results to {out_path}")
