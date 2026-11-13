#!/usr/bin/env python3
"""
RAxML-NG & IQ-TREE Simulation Benchmark Worker for Bushwhacker (Silverback Cluster).

Executes a single (scenario_idx, replicate_idx) task from the 18-scenario benchmark matrix:
  - Generates ground-truth tree and alignment via IQ-TREE 2 AliSim (`iqtree2 --alisim`)
  - Runs:
      1. Bushwhacker (MDL-pruned tree & reticulate shunt detector)
      2. FastTree 2 (-nt -gtr -gamma)
      3. IQ-TREE 2 (-m GTR+F+G4) + BioNJ starting tree
      4. IQ-TREE 2 --polytomy (-m GTR+F+G4 --polytomy, zero-branch-length collapse)
      5. RAxML-NG (--model GTR+F+G4 --tree pars{1}, for N <= 512)
  - Computes exact unrooted bipartition metrics:
      * Total true splits, resolved splits
      * True Positives (TP), False Positives (FP), False Negatives (FN)
      * Split Precision (1 - FDR = TP / (TP + FP))
      * Overall Split Recall (TP / True_Splits)
      * Active-Branch Recall for expected substitutions l*L >= 1.0 and l*L >= 2.0
      * Normalized Robinson-Foulds distance nRF = (FP + FN) / (2 * True_Splits)
      * Split F1 score = 2*TP / (2*TP + FP + FN)
      * Certified shunts (false positive check on clonal trees; recall on HGT/convergent regimes)
"""


import sys as _sys
from pathlib import Path as _Path
if str(_Path(__file__).resolve().parent) not in _sys.path:
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
from _env_paths import REPO_ROOT, DATA_DIR, SPECIMENS_DIR, ALIGNMENTS_DIR, SINGLE_CELL_DIR, RESULTS_DIR, WORK_ROOT, EXTERNAL_DIR, resolve_bin, ASTRAL_JAR

import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from Bio import Phylo


SCENARIOS = [
    # Regime A: Taxon Scaling on Yule-Harding Birth-Death Trees (GTR+F+G4, L=1000)
    {
        "id": "S01_yule_N64_L1000",
        "regime": "A_Taxon_Scaling",
        "desc": "Yule-Harding N=64, L=1000, GTR+F+G4 (alpha=0.5)",
        "tree_spec": "RANDOM{yh,64}",
        "rlen": ("0.001", "0.02", "0.10"),
        "N": 64,
        "L": 1000,
        "model": "GTR{1.5/3.0/0.8/1.2/4.0}+F{0.30/0.20/0.20/0.30}+G4{0.5}",
        "run_raxml": True,
    },
    {
        "id": "S02_yule_N256_L1000",
        "regime": "A_Taxon_Scaling",
        "desc": "Yule-Harding N=256, L=1000, GTR+F+G4 (alpha=0.5)",
        "tree_spec": "RANDOM{yh,256}",
        "rlen": ("0.001", "0.02", "0.10"),
        "N": 256,
        "L": 1000,
        "model": "GTR{1.5/3.0/0.8/1.2/4.0}+F{0.30/0.20/0.20/0.30}+G4{0.5}",
        "run_raxml": True,
    },
    {
        "id": "S03_yule_N512_L1000",
        "regime": "A_Taxon_Scaling",
        "desc": "Yule-Harding N=512, L=1000, GTR+F+G4 (alpha=0.5)",
        "tree_spec": "RANDOM{yh,512}",
        "rlen": ("0.001", "0.02", "0.10"),
        "N": 512,
        "L": 1000,
        "model": "GTR{1.5/3.0/0.8/1.2/4.0}+F{0.30/0.20/0.20/0.30}+G4{0.5}",
        "run_raxml": False,
    },
    {
        "id": "S04_yule_N1024_L1000",
        "regime": "A_Taxon_Scaling",
        "desc": "Yule-Harding N=1024, L=1000, GTR+F+G4 (alpha=0.5)",
        "tree_spec": "RANDOM{yh,1024}",
        "rlen": ("0.001", "0.02", "0.10"),
        "N": 1024,
        "L": 1000,
        "model": "GTR{1.5/3.0/0.8/1.2/4.0}+F{0.30/0.20/0.20/0.30}+G4{0.5}",
        "run_raxml": False,
    },
    # Regime B: Tree Topology Extremes (Caterpillar/Ladder, Balanced, Uniform, Star Null)
    {
        "id": "S05_caterpillar_N256",
        "regime": "B_Topology_Extremes",
        "desc": "Caterpillar (Pectinate Ladder) N=256, L=1000, GTR+F+G4",
        "tree_spec": "RANDOM{cat,256}",
        "rlen": ("0.002", "0.015", "0.06"),
        "N": 256,
        "L": 1000,
        "model": "GTR{1.5/3.0/0.8/1.2/4.0}+F{0.30/0.20/0.20/0.30}+G4{0.5}",
        "run_raxml": True,
    },
    {
        "id": "S06_balanced_N256",
        "regime": "B_Topology_Extremes",
        "desc": "Balanced Binary N=256, L=1000, GTR+F+G4",
        "tree_spec": "RANDOM{bal,256}",
        "rlen": ("0.002", "0.02", "0.08"),
        "N": 256,
        "L": 1000,
        "model": "GTR{1.5/3.0/0.8/1.2/4.0}+F{0.30/0.20/0.20/0.30}+G4{0.5}",
        "run_raxml": True,
    },
    {
        "id": "S07_uniform_N256",
        "regime": "B_Topology_Extremes",
        "desc": "Uniform Random Unrooted N=256, L=1000, GTR+F+G4",
        "tree_spec": "RANDOM{u,256}",
        "rlen": ("0.001", "0.02", "0.10"),
        "N": 256,
        "L": 1000,
        "model": "GTR{1.5/3.0/0.8/1.2/4.0}+F{0.30/0.20/0.20/0.30}+G4{0.5}",
        "run_raxml": True,
    },
    {
        "id": "S08_star_null_N128",
        "regime": "B_Topology_Extremes",
        "desc": "Pure Star Radiation (0 true splits) N=128, L=1000, GTR+F+G4",
        "tree_spec": "STAR_CUSTOM",
        "rlen": ("0.01", "0.05", "0.10"),
        "N": 128,
        "L": 1000,
        "model": "GTR{1.5/3.0/0.8/1.2/4.0}+F{0.30/0.20/0.20/0.30}+G4{0.5}",
        "run_raxml": True,
    },
    # Regime C: Alignment Length Scaling (L = 250, 500, 2000, 4000 at N=200)
    {
        "id": "S09_len_N200_L250",
        "regime": "C_Sequence_Length",
        "desc": "Short Alignment N=200, L=250 bp, GTR+F+G4",
        "tree_spec": "RANDOM{yh,200}",
        "rlen": ("0.001", "0.02", "0.10"),
        "N": 200,
        "L": 250,
        "model": "GTR{1.5/3.0/0.8/1.2/4.0}+F{0.30/0.20/0.20/0.30}+G4{0.5}",
        "run_raxml": True,
    },
    {
        "id": "S10_len_N200_L500",
        "regime": "C_Sequence_Length",
        "desc": "Medium-Short Alignment N=200, L=500 bp, GTR+F+G4",
        "tree_spec": "RANDOM{yh,200}",
        "rlen": ("0.001", "0.02", "0.10"),
        "N": 200,
        "L": 500,
        "model": "GTR{1.5/3.0/0.8/1.2/4.0}+F{0.30/0.20/0.20/0.30}+G4{0.5}",
        "run_raxml": True,
    },
    {
        "id": "S11_len_N200_L2000",
        "regime": "C_Sequence_Length",
        "desc": "Long Alignment N=200, L=2000 bp, GTR+F+G4",
        "tree_spec": "RANDOM{yh,200}",
        "rlen": ("0.001", "0.02", "0.10"),
        "N": 200,
        "L": 2000,
        "model": "GTR{1.5/3.0/0.8/1.2/4.0}+F{0.30/0.20/0.20/0.30}+G4{0.5}",
        "run_raxml": True,
    },
    {
        "id": "S12_len_N200_L4000",
        "regime": "C_Sequence_Length",
        "desc": "High-Info Alignment N=200, L=4000 bp, GTR+F+G4",
        "tree_spec": "RANDOM{yh,200}",
        "rlen": ("0.001", "0.02", "0.10"),
        "N": 200,
        "L": 4000,
        "model": "GTR{1.5/3.0/0.8/1.2/4.0}+F{0.30/0.20/0.20/0.30}+G4{0.5}",
        "run_raxml": True,
    },
    # Regime D: Evolutionary Divergence Depth & Rate Heterogeneity Extremes
    {
        "id": "S13_lowdiv_outbreak_N200",
        "regime": "D_Divergence_RateHetero",
        "desc": "Low-Divergence Outbreak Radiation (mean b=0.003) N=200, L=1000",
        "tree_spec": "RANDOM{yh,200}",
        "rlen": ("0.0002", "0.003", "0.015"),
        "N": 200,
        "L": 1000,
        "model": "GTR{1.5/3.0/0.8/1.2/4.0}+F{0.30/0.20/0.20/0.30}+G4{0.5}",
        "run_raxml": True,
    },
    {
        "id": "S14_deep_saturated_N200",
        "regime": "D_Divergence_RateHetero",
        "desc": "Deep Divergence + Extreme Rate Hetero (mean b=0.06, alpha=0.25) N=200, L=1000",
        "tree_spec": "RANDOM{yh,200}",
        "rlen": ("0.002", "0.06", "0.25"),
        "N": 200,
        "L": 1000,
        "model": "GTR{1.5/3.0/0.8/1.2/4.0}+F{0.30/0.20/0.20/0.30}+G4{0.25}",
        "run_raxml": True,
    },
    # Regime E: Empirical Tree + Real Missing-Data Mimicking
    {
        "id": "S15_empirical_InfluenzaA",
        "regime": "E_Empirical_Mimic",
        "desc": "Empirical Influenza A H3N2 Ladder Tree (N=349, L=987, GTR+F+G4)",
        "tree_spec": "EMPIRICAL_FLU",
        "rlen": ("0.0", "0.0", "0.0"),
        "N": 349,
        "L": 987,
        "model": "GTR{1.5/3.0/0.8/1.2/4.0}+F{0.30/0.20/0.20/0.30}+G4{0.5}",
        "run_raxml": True,
    },
    {
        "id": "S16_empirical_HIV_RT_gaps",
        "regime": "E_Empirical_Mimic",
        "desc": "Empirical HIV-1 RT Patient-Pair Tree + Real Fragmentary Gaps (N=476, L=1005)",
        "tree_spec": "EMPIRICAL_HIV_RT",
        "rlen": ("0.0", "0.0", "0.0"),
        "N": 476,
        "L": 1005,
        "model": "GTR{1.5/3.0/0.8/1.2/4.0}+F{0.35/0.18/0.22/0.25}+G4{0.5}",
        "run_raxml": True,
    },
    # Regime F: Reticulate Evolution (Interval Recombination / HGT & Sparse Convergence)
    {
        "id": "S17_hgt_interval_N200",
        "regime": "F_Reticulate_Shunts",
        "desc": "Interval Recombination / HGT (150 bp tract transfer) N=200, L=1000",
        "tree_spec": "HGT_INTERVAL",
        "rlen": ("0.002", "0.02", "0.08"),
        "N": 200,
        "L": 1000,
        "model": "GTR{1.5/3.0/0.8/1.2/4.0}+F{0.30/0.20/0.20/0.30}+G4{0.5}",
        "run_raxml": True,
    },
    {
        "id": "S18_convergent_sparse_N200",
        "regime": "F_Reticulate_Shunts",
        "desc": "Sparse Convergent Selection (m=8 parallel sites across clades) N=200, L=1000",
        "tree_spec": "CONVERGENT_SPARSE",
        "rlen": ("0.002", "0.02", "0.08"),
        "N": 200,
        "L": 1000,
        "model": "GTR{1.5/3.0/0.8/1.2/4.0}+F{0.30/0.20/0.20/0.30}+G4{0.5}",
        "run_raxml": True,
    },
    # Extended Regime C Gene-Length Ramp (L = 100, 1000, 8000, 16000 at N=200)
    {
        "id": "S08b_len_N200_L100",
        "regime": "C_Sequence_Length",
        "desc": "Ultra-Short Alignment N=200, L=100 bp, GTR+F+G4",
        "tree_spec": "RANDOM{yh,200}",
        "rlen": ("0.001", "0.02", "0.10"),
        "N": 200,
        "L": 100,
        "model": "GTR{1.5/3.0/0.8/1.2/4.0}+F{0.30/0.20/0.20/0.30}+G4{0.5}",
        "run_raxml": True,
    },
    {
        "id": "S10b_len_N200_L1000",
        "regime": "C_Sequence_Length",
        "desc": "Baseline Alignment N=200, L=1000 bp, GTR+F+G4",
        "tree_spec": "RANDOM{yh,200}",
        "rlen": ("0.001", "0.02", "0.10"),
        "N": 200,
        "L": 1000,
        "model": "GTR{1.5/3.0/0.8/1.2/4.0}+F{0.30/0.20/0.20/0.30}+G4{0.5}",
        "run_raxml": True,
    },
    {
        "id": "S12b_len_N200_L8000",
        "regime": "C_Sequence_Length",
        "desc": "Very Long Alignment N=200, L=8000 bp, GTR+F+G4",
        "tree_spec": "RANDOM{yh,200}",
        "rlen": ("0.001", "0.02", "0.10"),
        "N": 200,
        "L": 8000,
        "model": "GTR{1.5/3.0/0.8/1.2/4.0}+F{0.30/0.20/0.20/0.30}+G4{0.5}",
        "run_raxml": True,
    },
    {
        "id": "S12c_len_N200_L16000",
        "regime": "C_Sequence_Length",
        "desc": "Ultra-Long Alignment N=200, L=16000 bp, GTR+F+G4",
        "tree_spec": "RANDOM{yh,200}",
        "rlen": ("0.001", "0.02", "0.10"),
        "N": 200,
        "L": 16000,
        "model": "GTR{1.5/3.0/0.8/1.2/4.0}+F{0.30/0.20/0.20/0.30}+G4{0.5}",
        "run_raxml": True,
    },
]


def popcount(x: int) -> int:
    return bin(x).count("1")


def extract_unrooted_splits(nwk_path: str, leaf_to_id: dict = None):
    """
    Extracts canonical unrooted non-trivial splits as integer bitmasks in O(N) time.
    Returns (leaf_to_id, splits_dict) where splits_dict maps canonical_bitmask -> branch_length.
    """
    tree = Phylo.read(nwk_path, "newick")
    terminals = [t.name for t in tree.get_terminals()]
    if leaf_to_id is None:
        sorted_leaves = sorted(terminals)
        leaf_to_id = {name: idx for idx, name in enumerate(sorted_leaves)}
    n = len(leaf_to_id)
    all_mask = (1 << n) - 1
    ref_mask = 1  # bit 0

    clade_mask = {}
    splits = {}
    for clade in tree.find_clades(order="postorder"):
        if clade.is_terminal():
            m = 1 << leaf_to_id[clade.name]
            clade_mask[id(clade)] = m
        else:
            m = 0
            for child in clade.clades:
                m |= clade_mask[id(child)]
            clade_mask[id(clade)] = m
            if clade is not tree.root:
                canon = (all_mask ^ m) if (m & ref_mask) else m
                k = popcount(canon)
                if 2 <= k <= n - 2:
                    bl = float(clade.branch_length) if clade.branch_length is not None else 0.0
                    splits[canon] = splits.get(canon, 0.0) + max(0.0, bl)
    return leaf_to_id, splits


def parse_fasta_dict(fa_path: str) -> dict:
    seqs = {}
    curr = None
    buf = []
    with open(fa_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if curr is not None:
                    seqs[curr] = "".join(buf)
                curr = line[1:].split()[0]
                buf = []
            else:
                buf.append(line)
    if curr is not None:
        seqs[curr] = "".join(buf)
    return seqs


def write_fasta_dict(seqs: dict, fa_path: str):
    with open(fa_path, "w") as f:
        for name, seq in seqs.items():
            f.write(f">{name}\n{seq}\n")


def evaluate_tree(est_nwk: str, true_splits: dict, leaf_to_id: dict, seq_len: int, runtime_s: float) -> dict:
    _, est_splits = extract_unrooted_splits(est_nwk, leaf_to_id)
    true_keys = set(true_splits.keys())
    est_keys = set(est_splits.keys())

    tp = len(true_keys & est_keys)
    fp = len(est_keys - true_keys)
    fn = len(true_keys - est_keys)

    n_true = len(true_keys)
    n_est = len(est_keys)

    precision = tp / n_est if n_est > 0 else (1.0 if n_true == 0 else 0.0)
    recall = tp / n_true if n_true > 0 else 1.0
    f1 = (2.0 * tp) / (2.0 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.0
    nrf = (fp + fn) / (2.0 * n_true) if n_true > 0 else float(fp)

    true_ge1 = {k for k, d in true_splits.items() if d * seq_len >= 1.0}
    true_ge2 = {k for k, d in true_splits.items() if d * seq_len >= 2.0}
    rec_ge1 = len(true_ge1 & est_keys) / len(true_ge1) if true_ge1 else 1.0
    rec_ge2 = len(true_ge2 & est_keys) / len(true_ge2) if true_ge2 else 1.0

    return {
        "n_true_splits": n_true,
        "n_true_ge1": len(true_ge1),
        "n_true_ge2": len(true_ge2),
        "n_est_splits": n_est,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 5),
        "recall": round(recall, 5),
        "recall_ge1": round(rec_ge1, 5),
        "recall_ge2": round(rec_ge2, 5),
        "f1": round(f1, 5),
        "nrf": round(nrf, 5),
        "runtime_s": round(runtime_s, 4),
    }


def run_task(scenario_idx: int, rep_idx: int, base_dir: Path, out_dir: Path, bw_only: bool = False):
    scen = SCENARIOS[scenario_idx]
    scen_id = scen["id"]
    seed = 10000 + scenario_idx * 100 + rep_idx
    rng = random.Random(seed)

    # Locate binaries
    bw_bin = str(base_dir / "repo" / "target" / "release" / "bushwhacker")
    if not os.path.exists(bw_bin):
        bw_bin = str(base_dir / "target" / "release" / "bushwhacker")

    iqtree_bin = "/opt/ohpc/pub/apps/iqtree/2.4.0/bin/iqtree2"
    if not os.path.exists(iqtree_bin):
        iqtree_bin = resolve_bin("IQTREE_BIN", "iqtree2", "iqtree")

    raxml_bin = "/opt/ohpc/pub/apps/raxml-ng/1.2.2/bin/raxml-ng"
    if not os.path.exists(raxml_bin):
        raxml_bin = shutil.which("raxml-ng") or "/opt/homebrew/bin/raxml-ng"

    fasttree_bin = "/usr/local/bin/FastTree"
    if not os.path.exists(fasttree_bin):
        fasttree_bin = shutil.which("fasttree") or "/usr/local/bin/fasttree"

    emp_dir = base_dir / "empirical"

    with tempfile.TemporaryDirectory(prefix=f"bw_{scen_id}_r{rep_idx}_") as tmp:
        tmp_path = Path(tmp)
        sim_prefix = tmp_path / "sim"
        true_tree_path = tmp_path / "true.treefile"
        fa_path = tmp_path / "sim.fa"

        t_spec = scen["tree_spec"]
        L = scen["L"]
        N = scen["N"]
        model = scen["model"]

        if t_spec == "STAR_CUSTOM":
            # Generate a 128-taxon star radiation (internal branch lengths = 0.0)
            # Represent as a binary tree with 0.0 internal branches and Exp(0.05) terminal branches
            leaves = [f"T{i+1}:{rng.expovariate(1.0/0.05):.6f}" for i in range(N)]
            curr = leaves
            while len(curr) > 2:
                nxt = []
                for i in range(0, len(curr), 2):
                    if i + 1 < len(curr):
                        nxt.append(f"({curr[i]},{curr[i+1]}):0.000000")
                    else:
                        nxt.append(curr[i])
                curr = nxt
            star_nwk = f"({curr[0]},{curr[1]});\n"
            true_tree_path.write_text(star_nwk)
            subprocess.run(
                [
                    iqtree_bin,
                    "--alisim",
                    str(sim_prefix),
                    "-t",
                    str(true_tree_path),
                    "--length",
                    str(L),
                    "-m",
                    model,
                    "-af",
                    "fasta",
                    "-seed",
                    str(seed),
                    "-quiet",
                ],
                check=True,
            )
        elif t_spec == "EMPIRICAL_FLU":
            shutil.copy(emp_dir / "InfluenzaA_iqtree.treefile", true_tree_path)
            subprocess.run(
                [
                    iqtree_bin,
                    "--alisim",
                    str(sim_prefix),
                    "-t",
                    str(true_tree_path),
                    "--length",
                    str(L),
                    "-m",
                    model,
                    "-af",
                    "fasta",
                    "-seed",
                    str(seed),
                    "-quiet",
                ],
                check=True,
            )
        elif t_spec == "EMPIRICAL_HIV_RT":
            shutil.copy(emp_dir / "HIV_RT_tree.nwk", true_tree_path)
            subprocess.run(
                [
                    iqtree_bin,
                    "--alisim",
                    str(sim_prefix),
                    "-t",
                    str(true_tree_path),
                    "--length",
                    str(L),
                    "-m",
                    model,
                    "-af",
                    "fasta",
                    "-seed",
                    str(seed),
                    "-quiet",
                ],
                check=True,
            )
            # Overlay authentic missing-data / fragmentary masks from empirical HIV_RT.fasta
            emp_seqs = parse_fasta_dict(str(emp_dir / "HIV_RT.fasta"))
            sim_seqs = parse_fasta_dict(str(fa_path))
            masked_seqs = {}
            for name, s_seq in sim_seqs.items():
                if name in emp_seqs and len(emp_seqs[name]) == len(s_seq):
                    e_seq = emp_seqs[name]
                    masked_seqs[name] = "".join(
                        "-" if e_ch in ("-", "?", "N", "n") else s_ch
                        for s_ch, e_ch in zip(s_seq, e_seq)
                    )
                else:
                    masked_seqs[name] = s_seq
            write_fasta_dict(masked_seqs, str(fa_path))
        elif t_spec in ("HGT_INTERVAL", "CONVERGENT_SPARSE"):
            rmin, rmean, rmax = scen["rlen"]
            subprocess.run(
                [
                    iqtree_bin,
                    "--alisim",
                    str(sim_prefix),
                    "-t",
                    f"RANDOM{{yh,{N}}}",
                    "-rlen",
                    rmin,
                    rmean,
                    rmax,
                    "--length",
                    str(L),
                    "-m",
                    model,
                    "-af",
                    "fasta",
                    "-seed",
                    str(seed),
                    "-quiet",
                ],
                check=True,
            )
            shutil.copy(str(sim_prefix) + ".treefile", true_tree_path)
            seqs = parse_fasta_dict(str(fa_path))
            tree = Phylo.read(str(true_tree_path), "newick")
            # Find disjoint small clades (size 2..4) separated in the tree
            small_clades = []
            for clade in tree.find_clades(order="postorder"):
                if not clade.is_terminal():
                    tips = [t.name for t in clade.get_terminals()]
                    if 2 <= len(tips) <= 4:
                        small_clades.append(tips)
            donor_tips = small_clades[0]
            recip_tips = next(
                (c for c in reversed(small_clades) if set(c).isdisjoint(donor_tips)),
                small_clades[-1],
            )
            if t_spec == "HGT_INTERVAL":
                # Transfer 150 bp recombinant block [300:450] from donor clade to recipient clade
                d_rep = seqs[donor_tips[0]]
                # Ensure the transferred block has strong divergence signature (at least 10 diffs)
                block = list(d_rep[300:450])
                comp = {"A": "G", "G": "A", "C": "T", "T": "C"}
                for pos in range(0, 150, 12):
                    block[pos] = comp.get(block[pos], "A")
                block_str = "".join(block)
                for t_name in donor_tips + recip_tips:
                    s_list = list(seqs[t_name])
                    s_list[300:450] = list(block_str)
                    seqs[t_name] = "".join(s_list)
            else:
                # Sparse convergent selection: 8 parallel adaptive mutations shared by two distant cherries
                conv_sites = [60, 135, 240, 390, 510, 660, 780, 900]
                comp = {"A": "G", "G": "A", "C": "T", "T": "C"}
                for s_idx in conv_sites:
                    # Pick a derived base distinct from the majority base at s_idx
                    counts = {"A": 0, "C": 0, "G": 0, "T": 0}
                    for sq in seqs.values():
                        if sq[s_idx] in counts:
                            counts[sq[s_idx]] += 1
                    maj_base = max(counts, key=counts.get)
                    mut_base = comp[maj_base]
                    for t_name in donor_tips + recip_tips:
                        s_list = list(seqs[t_name])
                        s_list[s_idx] = mut_base
                        seqs[t_name] = "".join(s_list)
            write_fasta_dict(seqs, str(fa_path))
        else:
            rmin, rmean, rmax = scen["rlen"]
            subprocess.run(
                [
                    iqtree_bin,
                    "--alisim",
                    str(sim_prefix),
                    "-t",
                    t_spec,
                    "-rlen",
                    rmin,
                    rmean,
                    rmax,
                    "--length",
                    str(L),
                    "-m",
                    model,
                    "-af",
                    "fasta",
                    "-seed",
                    str(seed),
                    "-quiet",
                ],
                check=True,
            )
            shutil.copy(str(sim_prefix) + ".treefile", true_tree_path)

        leaf_to_id, true_splits = extract_unrooted_splits(str(true_tree_path))
        if t_spec == "STAR_CUSTOM":
            true_splits = {}

        results = {
            "scenario_id": scen_id,
            "regime": scen["regime"],
            "description": scen["desc"],
            "replicate": rep_idx,
            "seed": seed,
            "N": N,
            "L": L,
            "methods": {},
        }

        # 1. Bushwhacker
        bw_nwk = tmp_path / "bw.nwk"
        bw_json = tmp_path / "bw.json"
        t0 = time.perf_counter()
        subprocess.run(
            [bw_bin, str(fa_path), "-o", str(bw_nwk), "-j", str(bw_json)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        t_bw = time.perf_counter() - t0
        bw_meta = json.loads(bw_json.read_text())
        m_bw = evaluate_tree(str(bw_nwk), true_splits, leaf_to_id, L, bw_meta.get("runtime_sec", t_bw))
        m_bw["certified_shunts"] = bw_meta.get("certified_shunts", 0)
        m_bw["shunt_modes"] = [s.get("mode") for s in bw_meta.get("shunts", [])]

        m_bw_bin = None
        if "binary_newick" in bw_meta and bw_meta["binary_newick"]:
            bw_bin_nwk = tmp_path / "bw_bin.nwk"
            bw_bin_nwk.write_text(bw_meta["binary_newick"] + "\n")
            m_bw_bin = evaluate_tree(str(bw_bin_nwk), true_splits, leaf_to_id, L, bw_meta.get("runtime_sec", t_bw))

        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{scen_id}_rep{rep_idx:02d}.json"
        if bw_only:
            if out_file.exists():
                existing = json.loads(out_file.read_text())
                existing["methods"]["Bushwhacker"] = m_bw
                if m_bw_bin is not None:
                    existing["methods"]["Bushwhacker_Binary"] = m_bw_bin
                out_file.write_text(json.dumps(existing, indent=2) + "\n")
                print(f"Updated Bushwhacker for {scen_id} rep {rep_idx}: saved to {out_file}")
            return

        results["methods"]["Bushwhacker"] = m_bw
        if m_bw_bin is not None:
            results["methods"]["Bushwhacker_Binary"] = m_bw_bin

        # 2. FastTree 2
        ft_nwk = tmp_path / "ft.nwk"
        t0 = time.perf_counter()
        with open(ft_nwk, "w") as out_f, open(fa_path, "r") as in_f:
            subprocess.run(
                [fasttree_bin, "-nt", "-gtr", "-gamma"],
                stdin=in_f,
                stdout=out_f,
                stderr=subprocess.DEVNULL,
                check=True,
            )
        t_ft = time.perf_counter() - t0
        results["methods"]["FastTree2"] = evaluate_tree(str(ft_nwk), true_splits, leaf_to_id, L, t_ft)

        # 3. IQ-TREE 2 (Standard GTR+F+G4) + BioNJ
        iq_prefix = tmp_path / "iq"
        iq_extra = ["-fast"] if (N >= 512 or L >= 8000) else []
        t0 = time.perf_counter()
        subprocess.run(
            [
                iqtree_bin,
                "-s",
                str(fa_path),
                "-m",
                "GTR+F+G4",
                "-pre",
                str(iq_prefix),
                "-nt",
                "4",
                "-seed",
                str(seed),
                "-quiet",
            ]
            + iq_extra,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        t_iq = time.perf_counter() - t0
        results["methods"]["IQTREE2"] = evaluate_tree(
            str(iq_prefix) + ".treefile", true_splits, leaf_to_id, L, t_iq
        )
        if os.path.exists(str(iq_prefix) + ".bionj"):
            results["methods"]["BioNJ"] = evaluate_tree(
                str(iq_prefix) + ".bionj", true_splits, leaf_to_id, L, 0.0
            )

        # 4. IQ-TREE 2 (--polytomy)
        iq_poly_prefix = tmp_path / "iq_poly"
        t0 = time.perf_counter()
        subprocess.run(
            [
                iqtree_bin,
                "-s",
                str(fa_path),
                "-m",
                "GTR+F+G4",
                "--polytomy",
                "-pre",
                str(iq_poly_prefix),
                "-nt",
                "4",
                "-seed",
                str(seed),
                "-quiet",
            ]
            + iq_extra,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        t_iq_poly = time.perf_counter() - t0
        results["methods"]["IQTREE2_Polytomy"] = evaluate_tree(
            str(iq_poly_prefix) + ".treefile", true_splits, leaf_to_id, L, t_iq_poly
        )

        # 5. RAxML-NG (for N <= 350 to keep per-task runtime fast)
        if scen.get("run_raxml", False) and N <= 350 and os.path.exists(raxml_bin):
            rx_prefix = tmp_path / "rx"
            t0 = time.perf_counter()
            subprocess.run(
                [
                    raxml_bin,
                    "--msa",
                    str(fa_path),
                    "--model",
                    "GTR+F+G4",
                    "--tree",
                    "pars{1}",
                    "--threads",
                    "4",
                    "--seed",
                    str(seed),
                    "--prefix",
                    str(rx_prefix),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            t_rx = time.perf_counter() - t0
            rx_tree = str(rx_prefix) + ".raxml.bestTree"
            if os.path.exists(rx_tree):
                results["methods"]["RAxML_NG"] = evaluate_tree(
                    rx_tree, true_splits, leaf_to_id, L, t_rx
                )

        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{scen_id}_rep{rep_idx:02d}.json"
        out_file.write_text(json.dumps(results, indent=2) + "\n")
        print(f"Completed {scen_id} rep {rep_idx}: saved to {out_file}")


def main():
    args = [a for a in sys.argv[1:] if a != "--bw-only"]
    bw_only = "--bw-only" in sys.argv[1:]
    if len(args) < 1:
        print("Usage: run_silverback_sim_worker.py [--bw-only] <task_id_0_to_179 | scenario_idx> [rep_idx]")
        sys.exit(1)

    base_dir = Path(__file__).resolve().parent.parent
    out_dir = base_dir / "sim_results"

    if len(args) == 2 and args[1].isdigit():
        scen_idx = int(args[0])
        rep_idx = int(args[1])
    else:
        task_id = int(args[0])
        n_reps = 10
        scen_idx = task_id // n_reps
        rep_idx = task_id % n_reps

    run_task(scen_idx, rep_idx, base_dir, out_dir, bw_only=bw_only)


if __name__ == "__main__":
    main()
