#!/usr/bin/env python3
"""
Pillar 1: Threshold-Free Coalescent Species-Tree Phylogenomics (Bushwhacker -> ASTRAL-IV)
========================================================================================
Executes three authentic benchmarks on silverback:
  1A. Ground-Truth Multi-Species Coalescent (MSC) + Anomaly-Zone Benchmark:
      Exact Kingman coalescent gene trees + AliSim (GTR+F+G4) sequence evolution across
      4 species-tree regimes (Rapid32_L300, Rapid32_L800, Anomaly24_L400, Large64_L500;
      10 replicates x 200 loci = 8,000 alignments), comparing Bushwhacker (MDL & Binary),
      IQ-TREE 2 (Binary, --polytomy, and -B 1000 UFBoot collapsed at 10/30/50/70%),
      and FastTree 2 (Binary, SH<0.70, SH<0.90) -> ASTER (astral4).
  1B. Whole-Proteome 882-Mammal Coalescent Species Tree (19,952 TOGA2 Loci -> astral4)
      plus Domain-Wide GTDB Prokaryotic Species Trees (ar122 1,672 sp. & bac120 1,000 sp.).
  1C. Empirical Eukaryotic Radiations (rokas2018 + avian_shultz2019 -> astral4):
      Published IQ-TREE, RAxML, FastTree vs. Bushwhacker (Binary & MDL) gene trees.
"""


import sys as _sys
from pathlib import Path as _Path
if str(_Path(__file__).resolve().parent) not in _sys.path:
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
from _env_paths import REPO_ROOT, DATA_DIR, SPECIMENS_DIR, ALIGNMENTS_DIR, SINGLE_CELL_DIR, RESULTS_DIR, WORK_ROOT, EXTERNAL_DIR, resolve_bin, ASTRAL_JAR

import os
import re
import sys
import glob
import gzip
import json
import math
import time
import random
import shutil
import tempfile
import subprocess
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

BUSHWHACKER_BIN = resolve_bin("BUSHWHACKER_BIN", REPO_ROOT / "target" / "release" / "bushwhacker", "bushwhacker")
IQTREE_BIN = resolve_bin("IQTREE_BIN", "iqtree2", "iqtree")
FASTTREE_BIN = resolve_bin("FASTTREE_BIN", "FastTreeMP", "FastTree", "fasttree")
ASTRAL4_BIN = resolve_bin("ASTRAL4_BIN", "astral4")

WORK_DIR = WORK_ROOT / "pillar1"
OUT_JSON = RESULTS_DIR / "macroevolution" / "pillar1_part1a.json"

ROKAS_ROOT = EXTERNAL_DIR / "rokas2018"
GTDB_DIR = EXTERNAL_DIR / "gtdb_r95"
TOGA_NT_DIR = WORK_ROOT / "toga2_nt" / "results"
TOGA_AA_DIR = WORK_ROOT / "toga2_aa" / "results"
TOGA_SP_TREE = Path("data/alignments/supermatrices/toga2_speciesTree.nh")
TOGA_TAX_TSV = ALIGNMENTS_DIR / "supermatrices" / "toga2_assemblies_and_species.tsv"
AVIAN_SHULTZ_DIR = ALIGNMENTS_DIR / "plant_c4_and_avian"


# =============================================================================
# Fast Bitmask Newick Parser, Support Collapser, & Unrooted Split Evaluator
# =============================================================================

class TreeNode:
    __slots__ = ("name", "length", "children")

    def __init__(self, name="", length=None, children=None):
        self.name = name
        self.length = length
        self.children = children if children is not None else []


def parse_newick_tree(nwk_str: str) -> TreeNode:
    s = nwk_str.strip()
    if s.endswith(";"):
        s = s[:-1]
    tokens = re.findall(r"'[^']*'|\[[^\]]*\]|\(|\)|,|:[^(),;\[\]]+|[^(),;:\[\]]+", s)
    root = TreeNode()
    stack = [root]
    curr = root
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        if tok.startswith("["):
            i += 1
            continue
        if tok == "(":
            child = TreeNode()
            curr.children.append(child)
            stack.append(curr)
            curr = child
        elif tok == ",":
            parent = stack[-1]
            child = TreeNode()
            parent.children.append(child)
            curr = child
        elif tok == ")":
            curr = stack.pop()
        elif tok.startswith(":"):
            try:
                curr.length = float(tok[1:])
            except ValueError:
                curr.length = 0.0
        else:
            lbl = tok.strip("'")
            curr.name = lbl
        i += 1
    while len(root.children) == 1 and not root.name:
        root = root.children[0]
    return root


def write_newick_tree(node: TreeNode, include_lengths: bool = True) -> str:
    def _fmt(nd: TreeNode, is_root: bool = False) -> str:
        if not nd.children:
            nm = nd.name
            if include_lengths and nd.length is not None:
                return f"{nm}:{nd.length:.7f}"
            return nm
        inner = ",".join(_fmt(c, False) for c in nd.children)
        lbl = nd.name if nd.name else ""
        if not is_root and include_lengths and nd.length is not None:
            return f"({inner}){lbl}:{nd.length:.7f}"
        return f"({inner}){lbl}"

    return _fmt(node, True) + ";"


def collapse_newick_by_support(nwk_str: str, min_support: float) -> tuple:
    """Collapse internal edges whose support label < min_support. Returns (collapsed_nwk, kept_splits, total_splits)."""
    root = parse_newick_tree(nwk_str)

    def _extract_num(lbl: str):
        if not lbl:
            return None
        # Handle IQ-TREE SH-aLRT/UFBoot like '85.2/94' or plain '94' or '0.872'
        part = lbl.split("/")[-1].strip()
        try:
            return float(part)
        except ValueError:
            return None

    total_int = [0]
    kept_int = [0]

    def _collapse(nd: TreeNode, is_root: bool):
        if not nd.children:
            return [nd]
        new_children = []
        for c in nd.children:
            new_children.extend(_collapse(c, False))
        nd.children = new_children
        if is_root:
            return [nd]
        total_int[0] += 1
        sup = _extract_num(nd.name)
        if sup is not None and sup < min_support:
            # Contract edge: lift children directly to parent
            return nd.children
        kept_int[0] += 1
        return [nd]

    _collapse(root, True)
    return write_newick_tree(root, include_lengths=False), kept_int[0], total_int[0]


def get_leaf_names(node: TreeNode) -> list:
    out = []
    stack = [node]
    while stack:
        nd = stack.pop()
        if not nd.children:
            if nd.name:
                out.append(nd.name)
        else:
            for c in reversed(nd.children):
                stack.append(c)
    return out


def extract_canonical_splits(node: TreeNode, taxon_to_bit: dict, full_mask: int) -> set:
    """Iterative postorder extraction of nontrivial unrooted splits restricted to taxon_to_bit."""
    n_shared = bin(full_mask).count("1")
    if n_shared < 4:
        return set()
    splits = set()
    stack = [(node, False)]
    masks = {}
    while stack:
        nd, visited = stack.pop()
        if not nd.children:
            masks[id(nd)] = taxon_to_bit.get(nd.name, 0)
        elif not visited:
            stack.append((nd, True))
            for c in nd.children:
                stack.append((c, False))
        else:
            m = 0
            for c in nd.children:
                m |= masks.pop(id(c), 0)
            masks[id(nd)] = m
            bc = bin(m).count("1")
            if 2 <= bc <= n_shared - 2:
                comp = full_mask ^ m
                splits.add(m if m < comp else comp)
    return splits


def compare_unrooted_trees(pred_nwk: str, ref_nwk: str) -> dict:
    t_pred = parse_newick_tree(pred_nwk)
    t_ref = parse_newick_tree(ref_nwk)
    leaves_pred = set(get_leaf_names(t_pred))
    leaves_ref = set(get_leaf_names(t_ref))
    shared = sorted(leaves_pred & leaves_ref)
    n = len(shared)
    if n < 4:
        return {"n_shared": n, "rf": 0, "nrf": 0.0, "fp": 0, "fn": 0, "tp": 0, "prec": 1.0, "rec": 1.0}
    taxon_to_bit = {tx: (1 << i) for i, tx in enumerate(shared)}
    full_mask = (1 << n) - 1
    sp_pred = extract_canonical_splits(t_pred, taxon_to_bit, full_mask)
    sp_ref = extract_canonical_splits(t_ref, taxon_to_bit, full_mask)
    tp = len(sp_pred & sp_ref)
    fp = len(sp_pred - sp_ref)
    fn = len(sp_ref - sp_pred)
    rf = fp + fn
    max_splits = n - 3
    nrf = rf / (2.0 * max_splits) if max_splits > 0 else 0.0
    prec = tp / len(sp_pred) if sp_pred else 1.0
    rec = tp / len(sp_ref) if sp_ref else 1.0
    return {
        "n_shared": n,
        "pred_splits": len(sp_pred),
        "ref_splits": len(sp_ref),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "rf": rf,
        "nrf": round(nrf, 6),
        "precision": round(prec, 5),
        "recall": round(rec, 5),
    }


def evaluate_group_monophyly(tree_nwk: str, taxon_to_group: dict) -> dict:
    t = parse_newick_tree(tree_nwk)
    leaves = [ tx for tx in get_leaf_names(t) if tx in taxon_to_group and taxon_to_group[tx] ]
    n = len(leaves)
    if n < 4:
        return {"n_groups": 0, "strict_monophyletic": 0, "compatible": 0}
    t2b = {tx: (1 << i) for i, tx in enumerate(leaves)}
    full = (1 << n) - 1
    splits = extract_canonical_splits(t, t2b, full)
    grp_masks = defaultdict(int)
    grp_counts = defaultdict(int)
    for tx in leaves:
        g = taxon_to_group[tx]
        grp_masks[g] |= t2b[tx]
        grp_counts[g] += 1
    valid = {g: m for g, m in grp_masks.items() if 2 <= grp_counts[g] <= n - 2}
    strict = 0
    compat = 0
    for g, gm in valid.items():
        comp = full ^ gm
        canon = gm if gm < comp else comp
        if canon in splits:
            strict += 1
            compat += 1
        else:
            # Four-Gamete compatibility: no split s crosses gm (all 4 intersections nonempty)
            crosses = False
            for s in splits:
                if (s & gm) and (s & comp) and ((full ^ s) & gm) and ((full ^ s) & comp):
                    crosses = True
                    break
            if not crosses:
                compat += 1
    tot = len(valid)
    return {
        "n_groups": tot,
        "strict_monophyletic": strict,
        "strict_frac": round(strict / tot, 5) if tot else 0.0,
        "compatible": compat,
        "compatible_frac": round(compat / tot, 5) if tot else 0.0,
    }


# =============================================================================
# 1A. Exact Multi-Species Coalescent (MSC) Simulator + Benchmark
# =============================================================================

class SpNode:
    __slots__ = ("name", "tau_cu", "rate_mult", "children")

    def __init__(self, name="", tau_cu=0.0, rate_mult=1.0, children=None):
        self.name = name
        self.tau_cu = tau_cu
        self.rate_mult = rate_mult
        self.children = children if children is not None else []


def spnode_to_newick(nd: SpNode, is_root: bool = True) -> str:
    if not nd.children:
        return f"{nd.name}:{max(1e-6, nd.tau_cu):.6f}"
    inner = ",".join(spnode_to_newick(c, False) for c in nd.children)
    if is_root:
        return f"({inner});"
    return f"({inner}):{max(1e-6, nd.tau_cu):.6f}"


def build_rapid_species_tree(n_species: int, rng: random.Random) -> SpNode:
    """Build a balanced + rapid-radiation species tree with short internal stems (0.05..0.22 CU)."""
    nodes = [
        SpNode(name=f"T{i:03d}", tau_cu=rng.uniform(0.8, 2.2), rate_mult=rng.lognormvariate(0.0, 0.35))
        for i in range(n_species)
    ]
    while len(nodes) > 1:
        rng.shuffle(nodes)
        nxt = []
        i = 0
        while i < len(nodes):
            if i + 1 < len(nodes):
                # Rapid radiation stems: 50% ultra-short (0.04..0.12 CU), 50% moderate (0.12..0.35 CU)
                tau = rng.uniform(0.04, 0.12) if rng.random() < 0.55 else rng.uniform(0.12, 0.35)
                rm = rng.lognormvariate(0.0, 0.25)
                nxt.append(SpNode(tau_cu=tau, rate_mult=rm, children=[nodes[i], nodes[i + 1]]))
                i += 2
            else:
                nxt.append(nodes[i])
                i += 1
        nodes = nxt
    return nodes[0]


def build_anomaly_zone_tree(n_species: int, rng: random.Random) -> SpNode:
    """Asymmetric caterpillar backbone with consecutive 0.06 CU stems (inside 4-taxon anomaly zone < 0.1568 CU)
    and alternating fast/slow terminal branches."""
    # Start from a deep cherry, successively attach taxa along ultra-short pectinate backbone
    curr = SpNode(
        tau_cu=0.06,
        rate_mult=1.0,
        children=[
            SpNode(name="T000", tau_cu=2.5, rate_mult=2.2),
            SpNode(name="T001", tau_cu=0.6, rate_mult=0.45),
        ],
    )
    for i in range(2, n_species):
        is_fast = (i % 2 == 0)
        leaf = SpNode(
            name=f"T{i:03d}",
            tau_cu=rng.uniform(1.8, 3.0) if is_fast else rng.uniform(0.4, 0.8),
            rate_mult=rng.uniform(1.8, 2.8) if is_fast else rng.uniform(0.35, 0.65),
        )
        # Consecutive 0.05..0.09 CU stems in anomaly zone, interspersed with occasional 0.25 CU stems
        tau_stem = rng.uniform(0.045, 0.085) if (i % 4 != 0) else rng.uniform(0.18, 0.32)
        curr = SpNode(tau_cu=tau_stem, rate_mult=1.0, children=[curr, leaf])
    return curr


def simulate_msc_gene_tree(sp_root: SpNode, theta: float, rng: random.Random) -> str:
    """Exact continuous-time Kingman Multi-Species Coalescent along sp_root.
    Returns rooted gene tree Newick with branch lengths in expected substitutions per site."""
    half_theta = 0.5 * theta
    locus_rate = rng.gammavariate(4.0, 0.25)  # locus-level rate multiplier (mean 1.0)

    def _sim_pop(nd: SpNode, is_root: bool):
        if not nd.children:
            # Single sampled allele enters at bottom of leaf population, traverses tau_cu
            sub_len = nd.tau_cu * half_theta * nd.rate_mult * locus_rate
            return [(nd.name, sub_len)]

        lineages = []
        for c in nd.children:
            lineages.extend(_sim_pop(c, False))

        # Coalesce inside population of duration tau_cu (or until 1 lineage if is_root)
        t_elapsed = 0.0
        tau = float("inf") if is_root else nd.tau_cu
        rate_per_cu = half_theta * nd.rate_mult * locus_rate

        while len(lineages) > 1:
            k = len(lineages)
            coal_rate = k * (k - 1) / 2.0
            dt = rng.expovariate(coal_rate)
            if t_elapsed + dt > tau:
                break
            # Advance all k lineages by dt
            add_len = dt * rate_per_cu
            lineages = [(nwk_sub, bl + add_len) for (nwk_sub, bl) in lineages]
            t_elapsed += dt
            # Pick two random lineages to coalesce
            idx_a, idx_b = rng.sample(range(k), 2)
            if idx_a > idx_b:
                idx_a, idx_b = idx_b, idx_a
            b_nwk, b_bl = lineages.pop(idx_b)
            a_nwk, a_bl = lineages.pop(idx_a)
            merged_str = f"({a_nwk}:{max(1e-7, a_bl):.7f},{b_nwk}:{max(1e-7, b_bl):.7f})"
            lineages.append((merged_str, 0.0))

        if not is_root:
            rem = max(0.0, tau - t_elapsed)
            add_len = rem * rate_per_cu
            lineages = [(nwk_sub, bl + add_len) for (nwk_sub, bl) in lineages]
        return lineages

    final_lin = _sim_pop(sp_root, True)
    assert len(final_lin) == 1
    return final_lin[0][0] + ";"


def _msc_locus_worker(args):
    """Simulate 1 locus alignment via AliSim and run all gene-tree estimators in isolated local scratch."""
    _unused_prefix, gtree_nwk, seq_len, seed, run_ufboot = args
    with tempfile.TemporaryDirectory(prefix="bw_msc_") as tmpd:
        fa_prefix = os.path.join(tmpd, "loc")
        gtree_path = f"{fa_prefix}.true.nwk"
        fa_path = f"{fa_prefix}.fa"
        with open(gtree_path, "w") as f:
            f.write(gtree_nwk + "\n")

        # 1. Simulate alignment with IQ-TREE 2 AliSim
        subprocess.run(
            [
                IQTREE_BIN,
                "--alisim", fa_prefix,
                "-t", gtree_path,
                "-m", "GTR{1.2/3.8/0.9/1.4/4.2/1.0}+F{0.28/0.22/0.21/0.29}+G4{0.65}",
                "--length", str(seq_len),
                "-af", "fasta",
                "--seed", str(seed),
                "-nt", "1",
                "-quiet",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

        out = {}

        # 2. Bushwhacker (-T 1) -> captures both MDL and Binary in one pass
        bw_json_path = f"{fa_path}.bw.json"
        t0 = time.perf_counter()
        subprocess.run(
            [BUSHWHACKER_BIN, fa_path, "-j", bw_json_path, "-T", "1"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        bw_sec = time.perf_counter() - t0
        with open(bw_json_path) as f:
            bw_data = json.load(f)
        out["bw_mdl_nwk"] = bw_data["newick"].strip()
        out["bw_bin_nwk"] = bw_data["binary_newick"].strip()
        out["bw_resolved"] = bw_data["resolved_internal_splits"]
        out["bw_max_splits"] = bw_data["max_binary_splits"]
        out["bw_sec"] = bw_sec

        # 3. FastTree (-nt -gtr) with 1 OMP thread inside worker
        env1 = os.environ.copy()
        env1["OMP_NUM_THREADS"] = "1"
        t0 = time.perf_counter()
        res_ft = subprocess.run(
            [FASTTREE_BIN, "-nt", "-gtr", "-quiet", fa_path],
            capture_output=True, text=True, env=env1, check=True
        )
        ft_sec = time.perf_counter() - t0
        ft_nwk = res_ft.stdout.strip()
        out["ft_bin_nwk"] = ft_nwk
        out["ft_sh70_nwk"], _, _ = collapse_newick_by_support(ft_nwk, 0.70)
        out["ft_sh90_nwk"], _, _ = collapse_newick_by_support(ft_nwk, 0.90)
        out["ft_sec"] = ft_sec

        # 4. IQ-TREE 2 Binary (GTR+G4)
        iq_bin_pref = f"{fa_prefix}.iqbin"
        t0 = time.perf_counter()
        subprocess.run(
            [IQTREE_BIN, "-s", fa_path, "-m", "GTR+G4", "-keep-ident", "-nt", "1", "-pre", iq_bin_pref, "-quiet", "-redo"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        out["iq_bin_sec"] = time.perf_counter() - t0
        with open(f"{iq_bin_pref}.treefile") as f:
            out["iq_bin_nwk"] = f.read().strip()

        # 5. IQ-TREE 2 (--polytomy)
        iq_poly_pref = f"{fa_prefix}.iqpoly"
        t0 = time.perf_counter()
        subprocess.run(
            [IQTREE_BIN, "-s", fa_path, "-m", "GTR+G4", "--polytomy", "-keep-ident", "-nt", "1", "-pre", iq_poly_pref, "-quiet", "-redo"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        out["iq_poly_sec"] = time.perf_counter() - t0
        with open(f"{iq_poly_pref}.treefile") as f:
            out["iq_poly_nwk"] = f.read().strip()

        # 6. IQ-TREE 2 (-B 1000 UFBoot) collapsed at 10%, 30%, 50%, 70%
        if run_ufboot:
            iq_uf_pref = f"{fa_prefix}.iquf"
            t0 = time.perf_counter()
            subprocess.run(
                [IQTREE_BIN, "-s", fa_path, "-m", "GTR+G4", "-B", "1000", "-keep-ident", "-nt", "1", "-pre", iq_uf_pref, "-quiet", "-redo"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            out["iq_uf_sec"] = time.perf_counter() - t0
            uf_tf = f"{iq_uf_pref}.treefile"
            if os.path.exists(uf_tf):
                with open(uf_tf) as f:
                    uf_nwk = f.read().strip()
            else:
                uf_nwk = out["iq_poly_nwk"]
            for cutoff in (10, 30, 50, 70):
                cnwk, _, _ = collapse_newick_by_support(uf_nwk, float(cutoff))
                out[f"iq_uf{cutoff}_nwk"] = cnwk

        return out


def run_astral4(gene_trees: list, tmp_dir: Path, tag: str, threads: int = 8) -> tuple:
    in_path = tmp_dir / f"{tag}.in.tre"
    out_path = tmp_dir / f"{tag}.out.tre"
    with open(in_path, "w") as f:
        for gt in gene_trees:
            g = gt.strip()
            if g:
                if not g.endswith(";"):
                    g += ";"
                f.write(g + "\n")
    t0 = time.perf_counter()
    subprocess.run(
        [ASTRAL4_BIN, "-t", str(threads), "-i", str(in_path), "-o", str(out_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    dt = time.perf_counter() - t0
    with open(out_path) as f:
        sp_nwk = f.read().strip()
    return sp_nwk, dt


def run_part_1a_msc_simulations(n_workers: int = 60) -> list:
    print("[Pillar 1A] Starting Ground-Truth Multi-Species Coalescent + Anomaly-Zone Benchmark...", flush=True)
    sim_dir = WORK_DIR / "msc_sims"
    sim_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [
        {"id": "MSC_Rapid32_L300", "kind": "rapid", "n_sp": 32, "n_loci": 200, "L": 300, "theta": 0.020, "n_reps": 10, "uf_reps": 5},
        {"id": "MSC_Rapid32_L800", "kind": "rapid", "n_sp": 32, "n_loci": 200, "L": 800, "theta": 0.020, "n_reps": 10, "uf_reps": 5},
        {"id": "MSC_Anomaly24_L400", "kind": "anomaly", "n_sp": 24, "n_loci": 200, "L": 400, "theta": 0.025, "n_reps": 10, "uf_reps": 5},
        {"id": "MSC_Large64_L500", "kind": "rapid", "n_sp": 64, "n_loci": 200, "L": 500, "theta": 0.020, "n_reps": 10, "uf_reps": 4},
    ]

    results_1a = []
    for sc in scenarios:
        sc_id = sc["id"]
        print(f"[Pillar 1A] Scenario {sc_id} ({sc['n_reps']} reps x {sc['n_loci']} loci)...", flush=True)
        base_methods = ["bw_mdl", "bw_bin", "ft_bin", "ft_sh70", "ft_sh90", "iq_bin", "iq_poly"]
        uf_methods = ["iq_uf10", "iq_uf30", "iq_uf50", "iq_uf70"]
        agg = defaultdict(lambda: {"nrf": [], "rf": [], "fp": [], "fn": [], "gene_cpu_sec": [], "astral_sec": []})
        bw_contract_fracs = []

        for rep in range(sc["n_reps"]):
            rng = random.Random(20261110 + hash(sc_id) % 100000 + rep * 997)
            sp_root = (
                build_anomaly_zone_tree(sc["n_sp"], rng)
                if sc["kind"] == "anomaly"
                else build_rapid_species_tree(sc["n_sp"], rng)
            )
            true_sp_nwk = spnode_to_newick(sp_root)
            do_uf = rep < sc["uf_reps"]

            tasks = []
            for loc in range(sc["n_loci"]):
                gt_nwk = simulate_msc_gene_tree(sp_root, sc["theta"], rng)
                pref = str(sim_dir / f"{sc_id}_r{rep}_l{loc}")
                seed = 100000 + rep * 1000 + loc + 1
                tasks.append((pref, gt_nwk, sc["L"], seed, do_uf))

            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                loc_outs = list(ex.map(_msc_locus_worker, tasks))

            tot_res = sum(x["bw_resolved"] for x in loc_outs)
            tot_max = sum(x["bw_max_splits"] for x in loc_outs)
            bw_contract_fracs.append(1.0 - (tot_res / max(1, tot_max)))

            time_map = {
                "bw_mdl": sum(x["bw_sec"] for x in loc_outs),
                "bw_bin": sum(x["bw_sec"] for x in loc_outs),
                "ft_bin": sum(x["ft_sec"] for x in loc_outs),
                "ft_sh70": sum(x["ft_sec"] for x in loc_outs),
                "ft_sh90": sum(x["ft_sec"] for x in loc_outs),
                "iq_bin": sum(x["iq_bin_sec"] for x in loc_outs),
                "iq_poly": sum(x["iq_poly_sec"] for x in loc_outs),
            }
            if do_uf:
                uf_t = sum(x.get("iq_uf_sec", 0.0) for x in loc_outs)
                for um in uf_methods:
                    time_map[um] = uf_t

            active_methods = base_methods + (uf_methods if do_uf else [])
            for m in active_methods:
                gt_list = [x[f"{m}_nwk"] for x in loc_outs]
                sp_est, a_sec = run_astral4(gt_list, sim_dir, f"{sc_id}_r{rep}_{m}", threads=8)
                cmp = compare_unrooted_trees(sp_est, true_sp_nwk)
                agg[m]["nrf"].append(cmp["nrf"])
                agg[m]["rf"].append(cmp["rf"])
                agg[m]["fp"].append(cmp["fp"])
                agg[m]["fn"].append(cmp["fn"])
                agg[m]["gene_cpu_sec"].append(time_map[m])
                agg[m]["astral_sec"].append(a_sec)

        summary_methods = {}
        for m, vals in agg.items():
            k = len(vals["nrf"])
            mean_cpu = sum(vals["gene_cpu_sec"]) / k
            mean_ast = sum(vals["astral_sec"]) / k
            summary_methods[m] = {
                "n_reps": k,
                "mean_nrf": round(sum(vals["nrf"]) / k, 5),
                "mean_rf": round(sum(vals["rf"]) / k, 2),
                "mean_fp": round(sum(vals["fp"]) / k, 2),
                "mean_fn": round(sum(vals["fn"]) / k, 2),
                "mean_gene_tree_cpu_sec": round(mean_cpu, 2),
                "mean_wall_64core_sec": round(mean_cpu / 64.0 + mean_ast, 3),
                "mean_astral_sec": round(mean_ast, 3),
            }
        bw_w = summary_methods["bw_mdl"]["mean_wall_64core_sec"]
        for m, sm in summary_methods.items():
            sm["speedup_of_bw_over_method"] = round(sm["mean_wall_64core_sec"] / max(1e-6, bw_w), 2)

        sc_rec = {
            "scenario": sc_id,
            "kind": sc["kind"],
            "n_species": sc["n_sp"],
            "n_loci": sc["n_loci"],
            "seq_len_bp": sc["L"],
            "theta": sc["theta"],
            "mean_bw_polytomy_contracted_frac": round(sum(bw_contract_fracs) / len(bw_contract_fracs), 4),
            "methods": summary_methods,
        }
        results_1a.append(sc_rec)
        print(
            f"[Pillar 1A] {sc_id} DONE | BW_MDL nRF={summary_methods['bw_mdl']['mean_nrf']:.4f} "
            f"vs IQ_BIN={summary_methods['iq_bin']['mean_nrf']:.4f}, "
            f"IQ_POLY={summary_methods['iq_poly']['mean_nrf']:.4f}, "
            f"IQ_UF30={summary_methods.get('iq_uf30', {}).get('mean_nrf', -1):.4f}, "
            f"FT_BIN={summary_methods['ft_bin']['mean_nrf']:.4f}",
            flush=True,
        )
    return results_1a


# =============================================================================
# 1B. 882-Mammal TOGA2 (19,952 Loci) & Domain-Wide GTDB Coalescent Species Trees
# =============================================================================

def _read_toga_nt_pair(p: str):
    d = json.load(gzip.open(p))
    return d["binary_newick"].strip(), d["newick"].strip()


def _read_toga_aa_tree(p: str):
    d = json.load(gzip.open(p))
    return d["free_newick"].strip()


def run_part_1b_mammals_and_gtdb(n_workers: int = 48) -> dict:
    print("[Pillar 1B] Extracting all 19,952 TOGA2 Mammalian Gene Trees -> ASTER astral4...", flush=True)
    emp_dir = WORK_DIR / "empirical_astral"
    emp_dir.mkdir(parents=True, exist_ok=True)

    with open(TOGA_SP_TREE) as f:
        ref_mam_nwk = f.read().strip()

    # Load mammalian Order and Family taxonomy
    asm_to_order = {}
    asm_to_family = {}
    if TOGA_TAX_TSV.exists():
        with open(TOGA_TAX_TSV) as f:
            hdr = f.readline().strip().split("\t")
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < len(hdr):
                    continue
                row = dict(zip(hdr, parts))
                asm = row.get("assembly") or row.get("Assembly") or parts[0]
                ord_v = row.get("order") or row.get("Order") or ""
                fam_v = row.get("family") or row.get("Family") or ""
                if asm:
                    asm_to_order[asm] = ord_v
                    asm_to_family[asm] = fam_v

    nt_files = sorted(glob.glob(str(TOGA_NT_DIR / "*.bw.json.gz")))
    aa_files = sorted(glob.glob(str(TOGA_AA_DIR / "*.bwaa.json.gz")))

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        nt_pairs = list(ex.map(_read_toga_nt_pair, nt_files, chunksize=128))
        aa_mdls = list(ex.map(_read_toga_aa_tree, aa_files, chunksize=128))

    nt_bins = [x[0] for x in nt_pairs]
    nt_mdls = [x[1] for x in nt_pairs]

    mammal_results = {}
    for label, gtrees in [
        ("TOGA2_Codon_Binary", nt_bins),
        ("TOGA2_Codon_MDL", nt_mdls),
        ("TOGA2_Protein_MDL", aa_mdls),
    ]:
        sp_nwk, a_sec = run_astral4(gtrees, emp_dir, label, threads=32)
        cmp = compare_unrooted_trees(sp_nwk, ref_mam_nwk)
        ord_m = evaluate_group_monophyly(sp_nwk, asm_to_order) if asm_to_order else {}
        fam_m = evaluate_group_monophyly(sp_nwk, asm_to_family) if asm_to_family else {}
        mammal_results[label] = {
            "n_gene_trees": len(gtrees),
            "astral4_sec": round(a_sec, 2),
            "vs_speciesTree_nh": cmp,
            "mammalian_order_monophyly": ord_m,
            "mammalian_family_monophyly": fam_m,
        }
        print(
            f"[Pillar 1B] {label}: n={len(gtrees)}, astral={a_sec:.1f}s, "
            f"nRF vs speciesTree.nh={cmp['nrf']:.5f} (RF={cmp['rf']}), "
            f"Orders={ord_m.get('strict_monophyletic')}/{ord_m.get('n_groups')}, "
            f"Families={fam_m.get('strict_monophyletic')}/{fam_m.get('n_groups')}",
            flush=True,
        )

    # GTDB Domain-Wide Coalescent Species Trees (ar122 & bac120)
    gtdb_results = {}
    gtdb_work = GTDB_DIR / "work"
    for dom, ref_tree_fn in [("ar122", "ar122_r95.tree"), ("bac120", "bac120_r95.tree")]:
        with open(GTDB_DIR / ref_tree_fn) as f:
            ref_nwk = f.read().strip()
        gtdb_results[dom] = {}
        for mode, ext in [("FastTree_GTR", "fasttree.nwk"), ("Bushwhacker_Binary", "bw_bin.nwk"), ("Bushwhacker_MDL", "bw_mdl.nwk")]:
            files = sorted(glob.glob(str(gtdb_work / f"{dom}_*.{ext}")))
            gtrees = [open(p).read().strip() for p in files]
            sp_nwk, a_sec = run_astral4(gtrees, emp_dir, f"gtdb_{dom}_{mode}", threads=16)
            cmp = compare_unrooted_trees(sp_nwk, ref_nwk)
            gtdb_results[dom][mode] = {
                "n_loci": len(gtrees),
                "astral4_sec": round(a_sec, 3),
                "vs_gtdb_ref_tree": cmp,
            }
            print(f"[Pillar 1B] GTDB {dom} {mode}: nRF={cmp['nrf']:.4f} (RF={cmp['rf']})", flush=True)

    return {"toga2_882_mammals": mammal_results, "gtdb_prokaryotes": gtdb_results}


# =============================================================================
# 1C. Empirical Eukaryotic Radiations (rokas2018 + avian_shultz2019 -> astral4)
# =============================================================================

def _bw_gene_worker(args):
    fa_path, is_aa = args
    fd, jpath = tempfile.mkstemp(prefix="bw1c_", suffix=".json")
    os.close(fd)
    cmd = [BUSHWHACKER_BIN, fa_path, "-j", jpath, "-T", "1"]
    if is_aa:
        cmd.append("--protein")
    t0 = time.perf_counter()
    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    dt = time.perf_counter() - t0
    if res.returncode != 0 or not os.path.exists(jpath) or os.path.getsize(jpath) == 0:
        try:
            os.remove(jpath)
        except OSError:
            pass
        return None
    with open(jpath) as f:
        d = json.load(f)
    try:
        os.remove(jpath)
    except OSError:
        pass
    return (
        d["binary_newick"].strip(),
        d["newick"].strip(),
        d["resolved_internal_splits"],
        d["max_binary_splits"],
        dt,
    )


def run_part_1c_eukaryotic_radiations(n_workers: int = 60) -> dict:
    print("[Pillar 1C] Running Empirical Eukaryotic Coalescent Species Trees (rokas2018 + avian_shultz2019)...", flush=True)
    euk_dir = WORK_DIR / "euk_astral"
    euk_dir.mkdir(parents=True, exist_ok=True)

    focal_rokas = [
        ("BoroA6", True, "Metazoa root (1,080 AA)"),
        ("WhelA7", True, "Metazoa + 4 sponge classes (210 AA)"),
        ("ChenA4", True, "Jawed vertebrates / Tetrapod sister (4,682 AA)"),
        ("SongD1", False, "Placental mammals (424 NT)"),
        ("PrumD6", False, "Neoaves birds (259 NT, 199 sp.)"),
        ("WickA3", True, "Land plants 1KP (844 AA)"),
        ("WickD3a", False, "Land plants 1KP all-codon (844 NT)"),
        ("MisoA2", True, "Insects 1KITE (1,478 AA)"),
        ("XiD4", False, "Angiosperms (310 NT)"),
        ("NagyA1", True, "Fungi (594 AA)"),
        ("ShenA9", True, "Budding yeasts (1,233 AA)"),
    ]

    out_euk = {}
    for ds, is_aa, desc in focal_rokas:
        aln_dir = ROKAS_ROOT / "single-gene_alignments" / ds
        fas = sorted(
            p for p in glob.glob(str(aln_dir / "*"))
            if os.path.isfile(p) and not p.endswith(".tar.bz2")
        )
        if not fas:
            continue

        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            outs = [x for x in ex.map(_bw_gene_worker, [(p, is_aa) for p in fas], chunksize=8) if x is not None]
        wall_bw = time.perf_counter() - t0

        bw_bins = [x[0] for x in outs]
        bw_mdls = [x[1] for x in outs]
        tot_res = sum(x[2] for x in outs)
        tot_max = sum(x[3] for x in outs)

        sp_bw_bin, a_sec_bin = run_astral4(bw_bins, euk_dir, f"{ds}_bw_bin", threads=16)
        sp_bw_mdl, a_sec_mdl = run_astral4(bw_mdls, euk_dir, f"{ds}_bw_mdl", threads=16)

        # Reference concatenation or published ASTRAL species tree
        concat_candidates = glob.glob(str(ROKAS_ROOT / "concatenation_species_trees" / ds / "*")) + \
                            glob.glob(str(ROKAS_ROOT / "coalescent_species_trees" / ds / f"{ds}.Published.ASTRAL.tre")) + \
                            glob.glob(str(ROKAS_ROOT / "coalescent_species_trees" / ds / f"{ds}.IQ-TREE-10.ASTRAL.tre"))
        concat_files = [p for p in concat_candidates if os.path.isfile(p)]
        ref_sp_nwk = open(concat_files[0]).read().strip() if concat_files else sp_bw_mdl

        comparisons = {
            "Bushwhacker_MDL": compare_unrooted_trees(sp_bw_mdl, ref_sp_nwk),
            "Bushwhacker_Binary": compare_unrooted_trees(sp_bw_bin, ref_sp_nwk),
        }
        # Compare published precomputed ASTRAL species trees from Zhou et al. (2018)
        for pub_tag in ["IQ-TREE-10", "IQ-TREE", "RAxML-10", "FastTree", "PhyML-10"]:
            p_tre = ROKAS_ROOT / "coalescent_species_trees" / ds / f"{ds}.{pub_tag}.ASTRAL.tre"
            if p_tre.exists():
                comparisons[pub_tag] = compare_unrooted_trees(open(p_tre).read().strip(), ref_sp_nwk)

        # Cross-agreement directly between Bushwhacker_MDL ASTRAL tree and IQ-TREE-10 ASTRAL tree
        iq10_tre = ROKAS_ROOT / "coalescent_species_trees" / ds / f"{ds}.IQ-TREE-10.ASTRAL.tre"
        vs_iq10 = compare_unrooted_trees(sp_bw_mdl, open(iq10_tre).read().strip()) if iq10_tre.exists() else {}
        bin_vs_iq10 = compare_unrooted_trees(sp_bw_bin, open(iq10_tre).read().strip()) if iq10_tre.exists() else {}

        out_euk[ds] = {
            "description": desc,
            "alphabet": "protein" if is_aa else "nucleotide",
            "n_loci": len(outs),
            "bw_60core_wall_sec": round(wall_bw, 2),
            "astral4_sec": round(a_sec_mdl, 3),
            "polytomy_contracted_frac": round(1.0 - tot_res / max(1, tot_max), 4),
            "vs_reference_species_tree": comparisons,
            "bw_mdl_vs_iqtree10_astral": vs_iq10,
            "bw_bin_vs_iqtree10_astral": bin_vs_iq10,
        }
        print(
            f"[Pillar 1C] {ds} ({len(outs)} loci, {wall_bw:.1f}s): "
            f"contracted={out_euk[ds]['polytomy_contracted_frac']*100:.1f}%, "
            f"nRF vs IQ-TREE-10 ASTRAL: MDL={vs_iq10.get('nrf', -1):.4f} vs BIN={bin_vs_iq10.get('nrf', -1):.4f}",
            flush=True,
        )

    # Also run avian_shultz2019 (all orthologous bird genes -> ASTRAL vs avian_39_species_topology.nwk)
    shultz_fas = sorted(glob.glob(str(AVIAN_SHULTZ_DIR / "clean_fastas" / "*.fa*")))
    shultz_ref = AVIAN_SHULTZ_DIR / "avian_39_species_topology.nwk"
    if shultz_fas and shultz_ref.exists():
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            outs_nt = [x for x in ex.map(_bw_gene_worker, [(p, False) for p in shultz_fas], chunksize=16) if x is not None]
        wall_nt = time.perf_counter() - t0
        ref_bird = open(shultz_ref).read().strip()
        sp_s_bin, _ = run_astral4([x[0] for x in outs_nt], euk_dir, "shultz2019_bw_bin", threads=16)
        sp_s_mdl, a_s = run_astral4([x[1] for x in outs_nt], euk_dir, "shultz2019_bw_mdl", threads=16)
        out_euk["Avian_Shultz2019"] = {
            "description": f"Whole-genome avian orthologs ({len(outs_nt)} coding loci, 39 species)",
            "alphabet": "nucleotide",
            "n_loci": len(outs_nt),
            "bw_60core_wall_sec": round(wall_nt, 2),
            "astral4_sec": round(a_s, 3),
            "polytomy_contracted_frac": round(1.0 - sum(x[2] for x in outs_nt) / max(1, sum(x[3] for x in outs_nt)), 4),
            "bw_mdl_vs_published_bird_tree": compare_unrooted_trees(sp_s_mdl, ref_bird),
            "bw_bin_vs_published_bird_tree": compare_unrooted_trees(sp_s_bin, ref_bird),
        }
        print(
            f"[Pillar 1C] Avian_Shultz2019 ({len(outs_nt)} loci, {wall_nt:.1f}s): "
            f"MDL nRF={out_euk['Avian_Shultz2019']['bw_mdl_vs_published_bird_tree']['nrf']:.4f} vs "
            f"BIN nRF={out_euk['Avian_Shultz2019']['bw_bin_vs_published_bird_tree']['nrf']:.4f}",
            flush=True,
        )

    return out_euk


def main():
    t_start = time.time()
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    part_1b = run_part_1b_mammals_and_gtdb(n_workers=48)
    part_1c = run_part_1c_eukaryotic_radiations(n_workers=60)
    part_1a = run_part_1a_msc_simulations(n_workers=60)

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_wall_sec": round(time.time() - t_start, 2),
        "part_1a_msc_simulations": part_1a,
        "part_1b_mammals_and_gtdb": part_1b,
        "part_1c_eukaryotic_radiations": part_1c,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[Pillar 1] COMPLETE -> {OUT_JSON} in {time.time() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
