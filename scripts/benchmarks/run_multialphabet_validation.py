#!/usr/bin/env python3
"""
Comprehensive Multi-Alphabet, Dual-Mode Lateral Network, Single-Cell Non-Sequence,
and Zero-Bootstrap Posterior Calibration Ground-Truth Benchmark (1,200 datasets).

All nucleotide and amino-acid sequence alignments are simulated using third-party
IQ-TREE 2 AliSim (`LG+G4` and `GTR+F+G4`) along explicit ground-truth trees;
single-cell binary somatic and ordered integer copy-number matrices are simulated
via stochastic Markov branching with realistic allele dropout and measurement noise.
Every metric is computed directly from executed inference outputs with zero placeholders.
"""


import sys as _sys
from pathlib import Path as _Path
if str(_Path(__file__).resolve().parent) not in _sys.path:
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
from _env_paths import REPO_ROOT, DATA_DIR, SPECIMENS_DIR, ALIGNMENTS_DIR, SINGLE_CELL_DIR, RESULTS_DIR, WORK_ROOT, EXTERNAL_DIR, resolve_bin, ASTRAL_JAR

import concurrent.futures
import json
import math
import os
import random
import re
import subprocess
import tempfile
import time
from pathlib import Path

BW_BIN = resolve_bin("BUSHWHACKER_BIN", "bushwhacker")
IQTREE_BIN = resolve_bin("IQTREE_BIN", "iqtree2", "iqtree")
FASTTREE_BIN = resolve_bin("FASTTREE_BIN", "FastTree", "fasttree")
OUT_JSON = RESULTS_DIR / "synthetic" / "multialphabet_validation.json"


# =============================================================================
# Newick Parsing & Exact Unrooted Bipartition Set Utilities
# =============================================================================

def parse_newick_splits_and_supports(nwk: str):
    """
    Parse a Newick string (supporting multifurcations, branch lengths, and
    internal node support labels) and return:
      - all_leaves: frozenset of leaf names
      - splits_all: set of canonical bipartitions (min-side frozenset, 2 <= |side| <= N-2)
      - splits_active: set of canonical bipartitions whose branch length > 1e-7
      - splits_with_support: list of (canonical_bipartition, support_float_or_None, branch_len)
    """
    s = nwk.strip()
    if s.endswith(";"):
        s = s[:-1]

    # Tokenize Newick
    tokens = re.findall(r"\(|,|\)|:[^,():;]+|[^,():;]+", s)
    idx = 0

    def parse_node():
        nonlocal idx
        children = []
        if idx < len(tokens) and tokens[idx] == "(":
            idx += 1  # consume '('
            while True:
                children.append(parse_node())
                if idx < len(tokens) and tokens[idx] == ",":
                    idx += 1
                    continue
                elif idx < len(tokens) and tokens[idx] == ")":
                    idx += 1
                    break
                else:
                    break
        label = ""
        if idx < len(tokens) and tokens[idx] not in ("(", ")", ",") and not tokens[idx].startswith(":"):
            label = tokens[idx]
            idx += 1
        blen = 0.0
        if idx < len(tokens) and tokens[idx].startswith(":"):
            try:
                blen = float(tokens[idx][1:])
            except ValueError:
                blen = 0.0
            idx += 1
        return {"children": children, "label": label, "blen": blen}

    root = parse_node()

    def collect_leaves(nd):
        if not nd["children"]:
            nd["leaves"] = frozenset([nd["label"]])
        else:
            acc = set()
            for c in nd["children"]:
                acc |= collect_leaves(c)
            nd["leaves"] = frozenset(acc)
        return nd["leaves"]

    all_leaves = collect_leaves(root)
    n_total = len(all_leaves)

    def canon_split(sub):
        comp = all_leaves - sub
        if len(sub) < 2 or len(comp) < 2:
            return None
        if len(sub) < len(comp):
            return sub
        if len(comp) < len(sub):
            return comp
        return sub if sorted(sub) < sorted(comp) else comp

    splits_all = set()
    splits_active = set()
    # Deduplicate unrooted root split by keeping max blen / max support
    split_meta = {}

    def walk(nd, is_root=False):
        if not is_root and nd["children"]:
            sp = canon_split(nd["leaves"])
            if sp is not None and n_total >= 4:
                sup = None
                if nd["label"]:
                    # Handle formats like "0.95" or "N12/0.95"
                    parts = nd["label"].split("/")
                    for p in reversed(parts):
                        try:
                            v = float(p)
                            sup = v
                            break
                        except ValueError:
                            pass
                prev = split_meta.get(sp)
                if prev is None:
                    split_meta[sp] = [sup, nd["blen"]]
                else:
                    if sup is not None and (prev[0] is None or sup > prev[0]):
                        prev[0] = sup
                    prev[1] = prev[1] + nd["blen"]
        for c in nd["children"]:
            walk(c, False)

    walk(root, True)
    splits_with_support = []
    for sp, (sup, blen) in split_meta.items():
        splits_all.add(sp)
        if blen > 1e-7:
            splits_active.add(sp)
        splits_with_support.append((sp, sup, blen))

    return all_leaves, splits_all, splits_active, splits_with_support


def eval_splits(est_splits: set, true_active: set):
    tp = len(est_splits & true_active)
    fp = len(est_splits - true_active)
    fn = len(true_active - est_splits)
    rf = fp + fn
    prec = tp / len(est_splits) if est_splits else (1.0 if not true_active else 1.0)
    rec = tp / len(true_active) if true_active else 1.0
    denom = max(1, 2 * len(true_active))
    nrf = rf / denom
    return {
        "n_est": len(est_splits),
        "n_true_active": len(true_active),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "rf": rf,
        "nrf": nrf,
        "precision": prec,
        "recall": rec,
    }


# =============================================================================
# Ground-Truth Tree Generators
# =============================================================================

def gen_star_newick(n_taxa: int, rng: random.Random, blen_min=0.04, blen_max=0.16) -> str:
    parts = [f"T{i:03d}:{rng.uniform(blen_min, blen_max):.5f}" for i in range(n_taxa)]
    return "(" + ",".join(parts) + ");"


def gen_yule_newick(n_taxa: int, rng: random.Random, zero_frac: float = 0.0,
                    int_min=0.015, int_max=0.09, leaf_min=0.03, leaf_max=0.18) -> str:
    """Generate a random rooted tree with controlled fraction of zero-length internal stems."""
    nodes = [f"T{i:03d}" for i in range(n_taxa)]
    is_leaf = {nm: True for nm in nodes}
    subtrees = list(nodes)
    while len(subtrees) > 2:
        i = rng.randrange(len(subtrees))
        a = subtrees.pop(i)
        j = rng.randrange(len(subtrees))
        b = subtrees.pop(j)

        def fmt_child(x):
            if is_leaf.get(x, False):
                bl = rng.uniform(leaf_min, leaf_max)
            else:
                bl = 0.0 if rng.random() < zero_frac else rng.uniform(int_min, int_max)
            return f"{x}:{bl:.6f}"

        merged = f"({fmt_child(a)},{fmt_child(b)})"
        subtrees.append(merged)

    # Final unrooted 3-way or 2-way join; ensure root split has positive or zero length cleanly
    a, b = subtrees[0], subtrees[1]
    bla = rng.uniform(leaf_min, leaf_max) if is_leaf.get(a, False) else (0.0 if rng.random() < zero_frac else rng.uniform(int_min, int_max))
    blb = rng.uniform(leaf_min, leaf_max) if is_leaf.get(b, False) else (0.0 if rng.random() < zero_frac else rng.uniform(int_min, int_max))
    return f"({a}:{bla:.6f},{b}:{blb:.6f});"


def gen_structured_8clade_trees(rng: random.Random):
    """
    Generate a 48-taxon tree organized into 8 well-separated 6-taxon clades C0..C7:
      (((C0, C1), (C2, C3)), ((C4, C5), (C6, C7)))
    with gene-scale branch lengths suitable for homologous recombination / HGT / convergence.
    """
    clade_strs = {}
    clade_leaves = {}
    for cid in range(8):
        leaves = [f"C{cid}_t{k}" for k in range(6)]
        clade_leaves[cid] = frozenset(leaves)
        # Balanced 6-taxon subtree: (((0,1),(2,3)),(4,5))
        l0, l1, l2, l3, l4, l5 = [f"{nm}:{rng.uniform(0.004, 0.014):.5f}" for nm in leaves]
        p01 = f"({l0},{l1}):{rng.uniform(0.010, 0.022):.5f}"
        p23 = f"({l2},{l3}):{rng.uniform(0.010, 0.022):.5f}"
        p0123 = f"({p01},{p23}):{rng.uniform(0.010, 0.022):.5f}"
        p45 = f"({l4},{l5}):{rng.uniform(0.010, 0.022):.5f}"
        clade_strs[cid] = f"({p0123},{p45})"

    stem = lambda: f"{rng.uniform(0.028, 0.052):.5f}"
    deep = lambda: f"{rng.uniform(0.022, 0.042):.5f}"

    vert_nwk = (
        f"((({clade_strs[0]}:{stem()},{clade_strs[1]}:{stem()}):{deep()},"
        f"({clade_strs[2]}:{stem()},{clade_strs[3]}:{stem()}):{deep()}):{deep()},"
        f"(({clade_strs[4]}:{stem()},{clade_strs[5]}:{stem()}):{deep()},"
        f"({clade_strs[6]}:{stem()},{clade_strs[7]}:{stem()}):{deep()}):{deep()});"
    )

    rec_pair = frozenset(["C0_t0", "C0_t1"])
    don_pair = frozenset(["C5_t0", "C5_t1"])
    false_conv_clade = rec_pair | don_pair
    return vert_nwk, rec_pair, don_pair, clade_leaves[0], clade_leaves[5], false_conv_clade


def parse_seq_phylip(path: str) -> dict:
    seqs = {}
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    for ln in lines[1:]:
        parts = ln.split()
        if len(parts) >= 2:
            seqs[parts[0]] = "".join(parts[1:])
    return seqs


def write_fasta(path: str, seqs: dict):
    with open(path, "w") as f:
        for nm in sorted(seqs.keys()):
            f.write(f">{nm}\n{seqs[nm]}\n")


def extract_bw_calibration_points(bw_json_path: str, bw_nwk_text: str, true_active_splits: set):
    """Extract (posterior_prob, is_true_split) across all internal branches from Bushwhacker Newick & JSON."""
    pts = []
    with open(bw_json_path) as f:
        res = json.load(f)
    _, _, _, sup_list = parse_newick_splits_and_supports(bw_nwk_text)
    for sp, sup, blen in sup_list:
        if sup is not None:
            is_tp = 1 if sp in true_active_splits else 0
            pts.append((float(sup), is_tp, float(blen)))
    return pts, res


# =============================================================================
# Worker Tasks for Suites 1, 2, 3
# =============================================================================

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
DNA_ALPHABET = "ACGT"


def run_protein_rep(args):
    scenario, rep, seed = args
    rng = random.Random(seed)
    with tempfile.TemporaryDirectory() as td:
        if scenario == "P1_StarNull":
            n_taxa, length = 64, 500
            tree_nwk = gen_star_newick(n_taxa, rng, 0.05, 0.18)
            model = "LG+G4{0.8}"
        elif scenario == "P2_RapidRad":
            n_taxa, length = 64, 500
            tree_nwk = gen_yule_newick(n_taxa, rng, zero_frac=0.50, int_min=0.025, int_max=0.10, leaf_min=0.04, leaf_max=0.18)
            model = "LG+G4{0.8}"
        elif scenario == "P3_StandardYule":
            n_taxa, length = 64, 600
            tree_nwk = gen_yule_newick(n_taxa, rng, zero_frac=0.0, int_min=0.02, int_max=0.12, leaf_min=0.04, leaf_max=0.22)
            model = "LG+G4{0.7}"
        elif scenario == "P4_ConvergentAA":
            n_taxa, length = 48, 600
            tree_nwk, _, _, c0, c5, _ = gen_structured_8clade_trees(rng)
            false_clade = c0 | c5
            model = "LG+G4{0.7}"
        else:
            raise ValueError(scenario)

        t_path = f"{td}/true.nwk"
        with open(t_path, "w") as f:
            f.write(tree_nwk + "\n")
        _, _, true_active, _ = parse_newick_splits_and_supports(tree_nwk)

        subprocess.run(
            [IQTREE_BIN, "--alisim", f"{td}/sim", "-m", model, "--length", str(length), "-seed", str(seed), "-t", t_path],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        seqs = parse_seq_phylip(f"{td}/sim.phy")

        conv_sites = set()
        if scenario == "P4_ConvergentAA":
            # Inject 6 scattered adaptive convergent AA replacements shared exclusively by clades C0 and C5
            # at conserved columns so background taxa share ancestral residue
            candidates = []
            others = [nm for nm in seqs if nm not in c0 and nm not in c5]
            for pos in range(20, length - 20):
                col_other = {seqs[nm][pos] for nm in others}
                if len(col_other) == 1:
                    candidates.append((pos, next(iter(col_other))))
            rng.shuffle(candidates)
            chosen = sorted(candidates[:6], key=lambda x: x[0])
            # Ensure minimum spacing >= 35 residues so they are dispersed across protein domains
            spaced = []
            for pos, anc_aa in candidates:
                if all(abs(pos - p0) >= 35 for p0, _ in spaced):
                    spaced.append((pos, anc_aa))
                if len(spaced) == 6:
                    break
            seqs_mut = {nm: list(sq) for nm, sq in seqs.items()}
            for pos, anc_aa in spaced:
                conv_sites.add(pos)
                alt_choices = [a for a in "AILMVFWYSTNQ" if a != anc_aa]
                der_aa = rng.choice(alt_choices)
                for nm in (c0 | c5):
                    seqs_mut[nm][pos] = der_aa
            seqs = {nm: "".join(arr) for nm, arr in seqs_mut.items()}

        fa_path = f"{td}/aln.faa"
        write_fasta(fa_path, seqs)

        out = {"scenario": scenario, "rep": rep, "methods": {}, "cal_pts": []}

        # 1. Bushwhacker (--protein)
        t0 = time.perf_counter()
        subprocess.run(
            [BW_BIN, fa_path, "--protein", "-T", "1", "-o", f"{td}/bw.nwk", "-b", f"{td}/bw_bin.nwk", "-j", f"{td}/bw.json"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        bw_sec = time.perf_counter() - t0
        bw_nwk_txt = Path(f"{td}/bw.nwk").read_text()
        _, bw_splits, _, _ = parse_newick_splits_and_supports(bw_nwk_txt)
        _, bwb_splits, _, _ = parse_newick_splits_and_supports(Path(f"{td}/bw_bin.nwk").read_text())
        pts, bw_res = extract_bw_calibration_points(f"{td}/bw.json", bw_nwk_txt, true_active)
        out["cal_pts"] = pts
        m_bw = eval_splits(bw_splits, true_active)
        m_bw["time_s"] = bw_sec
        m_bwb = eval_splits(bwb_splits, true_active)
        m_bwb["time_s"] = bw_sec

        # 2. FastTree (-lg)
        t0 = time.perf_counter()
        ft_proc = subprocess.run(
            [FASTTREE_BIN, "-lg", "-quiet", fa_path],
            check=True, capture_output=True, text=True
        )
        ft_sec = time.perf_counter() - t0
        _, ft_splits, _, ft_sup_list = parse_newick_splits_and_supports(ft_proc.stdout)
        ft_sh90 = {sp for sp, sup, bl in ft_sup_list if bl > 1e-6 and (sup is None or sup >= 0.90)}
        m_ft = eval_splits(ft_splits, true_active)
        m_ft["time_s"] = ft_sec
        m_ft90 = eval_splits(ft_sh90, true_active)
        m_ft90["time_s"] = ft_sec

        # 3. IQ-TREE 2 (-m LG+G4 --fast --polytomy & binary)
        t0 = time.perf_counter()
        subprocess.run(
            [IQTREE_BIN, "-s", fa_path, "-m", "LG+G4", "--fast", "-nt", "1", "-quiet", "-pre", f"{td}/iq_bin"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        iq_sec = time.perf_counter() - t0
        _, iq_splits, _, _ = parse_newick_splits_and_supports(Path(f"{td}/iq_bin.treefile").read_text())
        m_iq = eval_splits(iq_splits, true_active)
        m_iq["time_s"] = iq_sec

        t0 = time.perf_counter()
        subprocess.run(
            [IQTREE_BIN, "-s", fa_path, "-m", "LG+G4", "--fast", "--polytomy", "-nt", "1", "-quiet", "-pre", f"{td}/iq_poly"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        iqp_sec = time.perf_counter() - t0
        _, _, iqp_active, _ = parse_newick_splits_and_supports(Path(f"{td}/iq_poly.treefile").read_text())
        m_iqp = eval_splits(iqp_active, true_active)
        m_iqp["time_s"] = iqp_sec

        if scenario == "P4_ConvergentAA":
            m_bw["false_conv_clade"] = 1 if false_clade in bw_splits else 0
            m_ft["false_conv_clade"] = 1 if false_clade in ft_splits else 0
            m_ft90["false_conv_clade"] = 1 if false_clade in ft_sh90 else 0
            m_iq["false_conv_clade"] = 1 if false_clade in iq_splits else 0
            m_iqp["false_conv_clade"] = 1 if false_clade in iqp_active else 0
            shunts = bw_res.get("shunts", [])
            sparse_sh = [s for s in shunts if s["mode"] == "SPARSE"]
            found_sites = set()
            for s in sparse_sh:
                found_sites.update(s.get("sites_0based", s.get("sites", [])))
            hit = len(found_sites & conv_sites)
            m_bw["sparse_shunt_detected"] = 1 if len(sparse_sh) > 0 else 0
            m_bw["conv_site_recall"] = hit / max(1, len(conv_sites))
            m_bw["conv_site_precision"] = hit / len(found_sites) if found_sites else 1.0

        out["methods"] = {
            "bw_mdl": m_bw,
            "bw_bin": m_bwb,
            "ft_raw": m_ft,
            "ft_sh90": m_ft90,
            "iq_raw": m_iq,
            "iq_poly": m_iqp,
        }
        return out


def run_network_rep(args):
    scenario, rep, seed = args
    rng = random.Random(seed)
    length = 1200
    gtr_model = "GTR{2.2,1.0,1.0,1.0,2.6}+F{0.28,0.22,0.23,0.27}+G4{0.75}"

    with tempfile.TemporaryDirectory() as td:
        vert_nwk, rec_pair, don_pair, c0, c5, false_clade = gen_structured_8clade_trees(rng)

        with open(f"{td}/vert.nwk", "w") as f:
            f.write(vert_nwk + "\n")
        _, _, true_active, _ = parse_newick_splits_and_supports(vert_nwk)

        subprocess.run(
            [IQTREE_BIN, "--alisim", f"{td}/vert", "-m", gtr_model, "--length", str(length), "-seed", str(seed), "-t", f"{td}/vert.nwk"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        seqs = parse_seq_phylip(f"{td}/vert.phy")

        true_interval = None
        conv_sites = set()
        if scenario in ("N2_HGT_Block200", "N3_HGT_Block450"):
            # Homologous transfer of segment [a, b) from donor C5_t0 into recipient stem (C0_t0, C0_t1),
            # followed by independent post-transfer Poisson substitutions along recipient leaves.
            a, b = (400, 600) if scenario == "N2_HGT_Block200" else (350, 800)
            true_interval = (a, b - 1)
            donor_tract = list(seqs["C5_t0"][a:b])
            for rec_nm in rec_pair:
                tract_copy = list(donor_tract)
                for k_pos in range(len(tract_copy)):
                    if rng.random() < 0.004:
                        cur = tract_copy[k_pos]
                        tract_copy[k_pos] = rng.choice([x for x in DNA_ALPHABET if x != cur])
                seqs[rec_nm] = seqs[rec_nm][:a] + "".join(tract_copy) + seqs[rec_nm][b:]
        elif scenario == "N4_SparseConv_10Sites":
            false_clade = frozenset(["C0_t0", "C5_t0"])
            others = [nm for nm in seqs if nm not in c0 and nm not in c5]
            candidates = []
            for pos in range(30, length - 30):
                col_other = {seqs[nm][pos] for nm in others}
                if len(col_other) == 1:
                    candidates.append((pos, next(iter(col_other))))
            rng.shuffle(candidates)
            spaced = []
            for pos, anc_nt in candidates:
                if all(abs(pos - p0) >= 65 for p0, _ in spaced):
                    spaced.append((pos, anc_nt))
                if len(spaced) == 10:
                    break
            seqs_mut = {nm: list(sq) for nm, sq in seqs.items()}
            for pos, anc_nt in spaced:
                conv_sites.add(pos)
                der_nt = rng.choice([x for x in DNA_ALPHABET if x != anc_nt])
                for nm in ("C0_t0", "C5_t0"):
                    seqs_mut[nm][pos] = der_nt
            seqs = {nm: "".join(arr) for nm, arr in seqs_mut.items()}

        fa_path = f"{td}/aln.fasta"
        write_fasta(fa_path, seqs)

        # Bushwhacker (--no-coding to treat general non-coding nucleotide coordinates cleanly)
        subprocess.run(
            [BW_BIN, fa_path, "--no-coding", "-T", "1", "-o", f"{td}/bw.nwk", "-j", f"{td}/bw.json"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        bw_nwk_txt = Path(f"{td}/bw.nwk").read_text()
        _, bw_splits, _, _ = parse_newick_splits_and_supports(bw_nwk_txt)
        pts, bw_res = extract_bw_calibration_points(f"{td}/bw.json", bw_nwk_txt, true_active)
        m_bw = eval_splits(bw_splits, true_active)

        # FastTree & IQ-TREE 2 for backbone comparison
        ft_proc = subprocess.run([FASTTREE_BIN, "-nt", "-gtr", "-quiet", fa_path], check=True, capture_output=True, text=True)
        _, ft_splits, _, _ = parse_newick_splits_and_supports(ft_proc.stdout)
        m_ft = eval_splits(ft_splits, true_active)

        subprocess.run(
            [IQTREE_BIN, "-s", fa_path, "-m", "GTR+G4", "--fast", "-nt", "1", "-quiet", "-pre", f"{td}/iq"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        _, iq_splits, _, _ = parse_newick_splits_and_supports(Path(f"{td}/iq.treefile").read_text())
        m_iq = eval_splits(iq_splits, true_active)

        shunts = bw_res.get("shunts", [])
        n_int = sum(1 for s in shunts if s["mode"] == "INTERVAL")
        n_spa = sum(1 for s in shunts if s["mode"] == "SPARSE")

        shunt_metrics = {
            "n_shunts": len(shunts),
            "n_interval": n_int,
            "n_sparse": n_spa,
            "any_shunt": 1 if len(shunts) > 0 else 0,
        }
        if true_interval is not None:
            ta, tb = true_interval
            best_iou = 0.0
            for s in shunts:
                if s["mode"] == "INTERVAL" and s.get("interval_span"):
                    sa, sb = s["interval_span"]
                    inter = max(0, min(tb, sb) - max(ta, sa) + 1)
                    union = (tb - ta + 1) + (sb - sa + 1) - inter
                    best_iou = max(best_iou, inter / max(1, union))
            shunt_metrics["interval_detected"] = 1 if n_int > 0 else 0
            shunt_metrics["top_mode_correct"] = 1 if (shunts and shunts[0]["mode"] == "INTERVAL") else 0
            shunt_metrics["breakpoint_iou"] = best_iou
            m_bw["false_hgt_clade"] = 1 if false_clade in bw_splits else 0
            m_ft["false_hgt_clade"] = 1 if false_clade in ft_splits else 0
            m_iq["false_hgt_clade"] = 1 if false_clade in iq_splits else 0
        elif scenario == "N4_SparseConv_10Sites":
            found = set()
            for s in shunts:
                if s["mode"] == "SPARSE":
                    found.update(s.get("sites_0based", s.get("sites", [])))
            hit = len(found & conv_sites)
            shunt_metrics["sparse_detected"] = 1 if n_spa > 0 else 0
            shunt_metrics["top_mode_correct"] = 1 if (shunts and shunts[0]["mode"] == "SPARSE") else 0
            shunt_metrics["site_recall"] = hit / max(1, len(conv_sites))
            shunt_metrics["site_precision"] = hit / len(found) if found else 1.0
            m_bw["false_conv_clade"] = 1 if false_clade in bw_splits else 0
            m_ft["false_conv_clade"] = 1 if false_clade in ft_splits else 0
            m_iq["false_conv_clade"] = 1 if false_clade in iq_splits else 0

        return {
            "scenario": scenario,
            "rep": rep,
            "bw": m_bw,
            "ft": m_ft,
            "iq": m_iq,
            "shunt": shunt_metrics,
            "cal_pts": pts,
        }


def simulate_clonal_matrix(n_cells: int, n_loci: int, rng: random.Random, mode: str,
                           zero_frac: float, dropout: float, fp_rate: float, missing_rate: float):
    """
    Simulate authentic Markov branching somatic evolution along an explicit clonal tree,
    recording the exact realized active splits (branches carrying >= 1 somatic event).
    """
    if zero_frac >= 1.0:
        # Pure diploid null star
        names = [f"Cell_{i:03d}" for i in range(n_cells)]
        root_state = [0] * n_loci if mode == "binary" else [2] * n_loci
        mat = {}
        for nm in names:
            row = list(root_state)
            for j in range(n_loci):
                if rng.random() < fp_rate:
                    if mode == "binary":
                        row[j] = 1 - row[j]
                    else:
                        row[j] = max(0, min(6, row[j] + rng.choice([-1, 1])))
            mat[nm] = row
        return names, mat, set()

    # Build a rooted clonal topology with zero_frac multifurcating bursts
    leaf_names = [f"Cell_{i:03d}" for i in range(n_cells)]
    nodes = [{"id": i, "name": leaf_names[i], "children": [], "leaves": frozenset([leaf_names[i]])} for i in range(n_cells)]
    pool = list(nodes)
    next_id = n_cells
    while len(pool) > 2:
        a = pool.pop(rng.randrange(len(pool)))
        b = pool.pop(rng.randrange(len(pool)))
        nd = {"id": next_id, "name": f"Clone_{next_id}", "children": [a, b], "leaves": a["leaves"] | b["leaves"]}
        next_id += 1
        pool.append(nd)
    root = {"id": next_id, "name": "NormalRoot", "children": pool, "leaves": pool[0]["leaves"] | pool[1]["leaves"]}
    all_leaves = root["leaves"]

    def canon(sub):
        comp = all_leaves - sub
        if len(sub) < 2 or len(comp) < 2:
            return None
        return sub if (len(sub) < len(comp) or (len(sub) == len(comp) and sorted(sub) < sorted(comp))) else comp

    root_vec = [0] * n_loci if mode == "binary" else [2] * n_loci
    avail_loci = list(range(n_loci))
    rng.shuffle(avail_loci)
    ptr = 0
    realized_active = set()
    leaf_states = {}

    def evolve(nd, state, is_root=False):
        nonlocal ptr
        cur = list(state)
        if not is_root:
            is_leaf = len(nd["children"]) == 0
            if is_leaf:
                n_mut = rng.choices([0, 1, 2], weights=[0.55, 0.35, 0.10])[0]
            else:
                if rng.random() < zero_frac:
                    n_mut = 0
                else:
                    n_mut = rng.randint(2, 4) if mode == "binary" else rng.randint(3, 7)

            changed = 0
            if mode == "binary":
                for _ in range(n_mut):
                    if ptr < len(avail_loci):
                        loc = avail_loci[ptr]
                        ptr += 1
                    else:
                        loc = rng.randrange(n_loci)
                    if cur[loc] == 0:
                        cur[loc] = 1
                        changed += 1
            else:
                # Ordered integer CNA gains/losses (focal + 2-4 bin segments) along chromosome
                for _ in range(n_mut):
                    seg_len = rng.randint(1, 3)
                    start = rng.randint(0, max(0, n_loci - seg_len))
                    step = rng.choice([-1, 1, 1])  # slight amplification bias as in tumors
                    for pos in range(start, start + seg_len):
                        if cur[pos] > 0:  # homozygous deletion 0 is absorbing
                            nv = max(0, min(6, cur[pos] + step))
                            if nv != cur[pos]:
                                cur[pos] = nv
                                changed += 1

            if not is_leaf and changed > 0:
                sp = canon(nd["leaves"])
                if sp is not None:
                    realized_active.add(sp)

        if not nd["children"]:
            # Single-cell observation model (allele dropout, false positives, missing data)
            obs = []
            for v in cur:
                if rng.random() < missing_rate:
                    obs.append("?")
                elif mode == "binary":
                    if v == 1 and rng.random() < dropout:
                        obs.append(0)
                    elif v == 0 and rng.random() < fp_rate:
                        obs.append(1)
                    else:
                        obs.append(v)
                else:
                    if rng.random() < fp_rate:
                        obs.append(max(0, min(6, v + rng.choice([-1, 1]))))
                    else:
                        obs.append(v)
            leaf_states[nd["name"]] = obs
        else:
            for c in nd["children"]:
                evolve(c, cur, False)

    evolve(root, root_vec, True)
    return leaf_names, leaf_states, realized_active


def run_singlecell_rep(args):
    scenario, rep, seed = args
    rng = random.Random(seed)
    with tempfile.TemporaryDirectory() as td:
        if scenario == "C1_BinaryBurst":
            names, mat, true_active = simulate_clonal_matrix(64, 180, rng, "binary", zero_frac=0.45, dropout=0.05, fp_rate=0.003, missing_rate=0.05)
            is_ordered = False
        elif scenario == "C2_BinaryHighDropout":
            names, mat, true_active = simulate_clonal_matrix(64, 180, rng, "binary", zero_frac=0.40, dropout=0.15, fp_rate=0.003, missing_rate=0.05)
            is_ordered = False
        elif scenario == "C3_OrderedCNA":
            names, mat, true_active = simulate_clonal_matrix(96, 200, rng, "cna", zero_frac=0.40, dropout=0.0, fp_rate=0.05, missing_rate=0.03)
            is_ordered = True
        elif scenario == "C4_DiploidNullNoise":
            names, mat, true_active = simulate_clonal_matrix(80, 150, rng, "cna", zero_frac=1.0, dropout=0.0, fp_rate=0.02, missing_rate=0.02)
            is_ordered = True
        else:
            raise ValueError(scenario)

        csv_path = f"{td}/cells.csv"
        n_loci = len(mat[names[0]])
        with open(csv_path, "w") as f:
            f.write("cell_id," + ",".join(f"L{j:03d}" for j in range(n_loci)) + "\n")
            for nm in names:
                f.write(nm + "," + ",".join(str(x) for x in mat[nm]) + "\n")

        out = {"scenario": scenario, "rep": rep, "methods": {}, "cal_pts": []}

        if not is_ordered:
            # Binary somatic: Bushwhacker (--discrete) vs Bushwhacker_Binary vs IQ-TREE 2 (MORPH/BIN)
            t0 = time.perf_counter()
            subprocess.run(
                [BW_BIN, csv_path, "--discrete", "--no-shunts", "-T", "1", "-o", f"{td}/bw.nwk", "-b", f"{td}/bw_bin.nwk", "-j", f"{td}/bw.json"],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            bw_sec = time.perf_counter() - t0
            bw_nwk_txt = Path(f"{td}/bw.nwk").read_text()
            _, bw_splits, _, _ = parse_newick_splits_and_supports(bw_nwk_txt)
            _, bwb_splits, _, _ = parse_newick_splits_and_supports(Path(f"{td}/bw_bin.nwk").read_text())
            pts, _ = extract_bw_calibration_points(f"{td}/bw.json", bw_nwk_txt, true_active)
            out["cal_pts"] = pts
            m_bw = eval_splits(bw_splits, true_active)
            m_bw["time_s"] = bw_sec
            m_bwb = eval_splits(bwb_splits, true_active)
            m_bwb["time_s"] = bw_sec

            # IQ-TREE 2 binary GTR2+G4 (replace '?' with '-')
            phy_path = f"{td}/bin.phy"
            with open(phy_path, "w") as f:
                f.write(f"{len(names)} {n_loci}\n")
                for nm in names:
                    seq_str = "".join("-" if str(x) == "?" else str(x) for x in mat[nm])
                    f.write(f"{nm}  {seq_str}\n")
            t0 = time.perf_counter()
            subprocess.run(
                [IQTREE_BIN, "-s", phy_path, "-st", "BIN", "-m", "GTR2+G4", "--fast", "-nt", "1", "-quiet", "-pre", f"{td}/iqb"],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            iq_sec = time.perf_counter() - t0
            _, iq_splits, _, _ = parse_newick_splits_and_supports(Path(f"{td}/iqb.treefile").read_text())
            m_iq = eval_splits(iq_splits, true_active)
            m_iq["time_s"] = iq_sec

            out["methods"] = {"bw_mdl": m_bw, "bw_bin": m_bwb, "iq_bin_ml": m_iq}
        else:
            # Ordered CNA: Bushwhacker (--ordered) vs Unordered (--discrete) vs Binarized (CN != 2)
            t0 = time.perf_counter()
            subprocess.run(
                [BW_BIN, csv_path, "--ordered", "--no-shunts", "-T", "1", "-o", f"{td}/bw_ord.nwk", "-j", f"{td}/bw_ord.json"],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            ord_sec = time.perf_counter() - t0
            ord_nwk_txt = Path(f"{td}/bw_ord.nwk").read_text()
            _, ord_splits, _, _ = parse_newick_splits_and_supports(ord_nwk_txt)
            pts, _ = extract_bw_calibration_points(f"{td}/bw_ord.json", ord_nwk_txt, true_active)
            out["cal_pts"] = pts
            m_ord = eval_splits(ord_splits, true_active)
            m_ord["time_s"] = ord_sec

            subprocess.run(
                [BW_BIN, csv_path, "--discrete", "--no-shunts", "-T", "1", "-o", f"{td}/bw_unord.nwk"],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            _, unord_splits, _, _ = parse_newick_splits_and_supports(Path(f"{td}/bw_unord.nwk").read_text())
            m_unord = eval_splits(unord_splits, true_active)

            # Binarized (CN != 2 -> 1, CN == 2 -> 0)
            bin_csv = f"{td}/cells_bin.csv"
            with open(bin_csv, "w") as f:
                f.write("cell_id," + ",".join(f"L{j:03d}" for j in range(n_loci)) + "\n")
                for nm in names:
                    brow = ["?" if x == "?" else ("0" if int(x) == 2 else "1") for x in mat[nm]]
                    f.write(nm + "," + ",".join(brow) + "\n")
            subprocess.run(
                [BW_BIN, bin_csv, "--discrete", "--no-shunts", "-T", "1", "-o", f"{td}/bw_cna_bin.nwk"],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            _, bincna_splits, _, _ = parse_newick_splits_and_supports(Path(f"{td}/bw_cna_bin.nwk").read_text())
            m_bincna = eval_splits(bincna_splits, true_active)

            out["methods"] = {"bw_ordered": m_ord, "bw_unordered": m_unord, "bw_binarized": m_bincna}

        return out


# =============================================================================
# Aggregation & Main Runner
# =============================================================================

def avg_dict(dict_list, keys):
    res = {}
    for k in keys:
        vals = [d[k] for d in dict_list if k in d]
        if vals:
            res[k] = round(sum(vals) / len(vals), 5)
    return res


def main():
    t_start = time.perf_counter()
    n_reps = 100
    max_workers = min(16, os.cpu_count() or 8)

    protein_scenarios = ["P1_StarNull", "P2_RapidRad", "P3_StandardYule", "P4_ConvergentAA"]
    network_scenarios = ["N1_ClonalNull", "N2_HGT_Block200", "N3_HGT_Block450", "N4_SparseConv_10Sites"]
    singlecell_scenarios = ["C1_BinaryBurst", "C2_BinaryHighDropout", "C3_OrderedCNA", "C4_DiploidNullNoise"]

    all_cal_pts = []
    summary = {
        "n_reps_per_scenario": n_reps,
        "total_new_datasets": n_reps * (len(protein_scenarios) + len(network_scenarios) + len(singlecell_scenarios)),
        "protein_suite": {},
        "network_suite": {},
        "singlecell_suite": {},
        "posterior_calibration": {},
    }

    print(f"Running Suite 1: 400 Protein LG+G4 alignments ({max_workers} workers)...", flush=True)
    p_tasks = [(sc, r, 100000 + si * 1000 + r) for si, sc in enumerate(protein_scenarios) for r in range(n_reps)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
        p_results = list(ex.map(run_protein_rep, p_tasks))

    for sc in protein_scenarios:
        sub = [x for x in p_results if x["scenario"] == sc]
        for x in sub:
            all_cal_pts.extend(x["cal_pts"])
        m_names = list(sub[0]["methods"].keys())
        sc_sum = {}
        for mn in m_names:
            sc_sum[mn] = avg_dict(
                [x["methods"][mn] for x in sub],
                ["n_est", "n_true_active", "tp", "fp", "fn", "rf", "nrf", "precision", "recall", "time_s",
                 "false_conv_clade", "sparse_shunt_detected", "conv_site_recall", "conv_site_precision"]
            )
        summary["protein_suite"][sc] = sc_sum
        print(f"  [Done] {sc}: BW prec={sc_sum['bw_mdl']['precision']*100:.1f}%, FP={sc_sum['bw_mdl']['fp']:.2f} vs IQ prec={sc_sum['iq_raw']['precision']*100:.1f}%, FP={sc_sum['iq_raw']['fp']:.2f}", flush=True)

    print(f"Running Suite 2: 400 Dual-Mode Network GTR+F+G4 alignments ({max_workers} workers)...", flush=True)
    n_tasks = [(sc, r, 200000 + si * 1000 + r) for si, sc in enumerate(network_scenarios) for r in range(n_reps)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
        n_results = list(ex.map(run_network_rep, n_tasks))

    for sc in network_scenarios:
        sub = [x for x in n_results if x["scenario"] == sc]
        for x in sub:
            all_cal_pts.extend(x["cal_pts"])
        sc_sum = {
            "bw": avg_dict([x["bw"] for x in sub], ["n_est", "n_true_active", "tp", "fp", "rf", "nrf", "precision", "recall", "false_hgt_clade", "false_conv_clade"]),
            "ft": avg_dict([x["ft"] for x in sub], ["n_est", "n_true_active", "tp", "fp", "rf", "nrf", "precision", "recall", "false_hgt_clade", "false_conv_clade"]),
            "iq": avg_dict([x["iq"] for x in sub], ["n_est", "n_true_active", "tp", "fp", "rf", "nrf", "precision", "recall", "false_hgt_clade", "false_conv_clade"]),
            "shunt": avg_dict([x["shunt"] for x in sub], ["n_shunts", "n_interval", "n_sparse", "any_shunt", "interval_detected", "sparse_detected", "top_mode_correct", "breakpoint_iou", "site_recall", "site_precision"]),
        }
        summary["network_suite"][sc] = sc_sum
        print(f"  [Done] {sc}: shunt={sc_sum['shunt']}", flush=True)

    print(f"Running Suite 3: 400 Single-Cell Binary & Ordered CNA matrices ({max_workers} workers)...", flush=True)
    c_tasks = [(sc, r, 300000 + si * 1000 + r) for si, sc in enumerate(singlecell_scenarios) for r in range(n_reps)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
        c_results = list(ex.map(run_singlecell_rep, c_tasks))

    for sc in singlecell_scenarios:
        sub = [x for x in c_results if x["scenario"] == sc]
        for x in sub:
            all_cal_pts.extend(x["cal_pts"])
        m_names = list(sub[0]["methods"].keys())
        sc_sum = {}
        for mn in m_names:
            sc_sum[mn] = avg_dict([x["methods"][mn] for x in sub], ["n_est", "n_true_active", "tp", "fp", "fn", "rf", "nrf", "precision", "recall", "time_s"])
        summary["singlecell_suite"][sc] = sc_sum
        lead = "bw_mdl" if "bw_mdl" in sc_sum else "bw_ordered"
        print(f"  [Done] {sc}: {lead} prec={sc_sum[lead]['precision']*100:.1f}%, FP={sc_sum[lead]['fp']:.2f}, RF={sc_sum[lead]['rf']:.2f}", flush=True)

    # Suite 4: Posterior Probability Calibration across all evaluated internal splits
    bins = [
        (0.50, 0.65, "0.50-0.65"),
        (0.65, 0.80, "0.65-0.80"),
        (0.80, 0.90, "0.80-0.90"),
        (0.90, 0.95, "0.90-0.95"),
        (0.95, 0.99, "0.95-0.99"),
        (0.99, 1.0001, "0.99-1.00"),
    ]
    cal_bins = []
    total_pts = len(all_cal_pts)
    ece = 0.0
    for lo, hi, lbl in bins:
        pts_b = [(p, y) for p, y, _ in all_cal_pts if lo <= p < hi]
        if pts_b:
            mean_p = sum(p for p, _ in pts_b) / len(pts_b)
            obs_acc = sum(y for _, y in pts_b) / len(pts_b)
            ece += (len(pts_b) / max(1, total_pts)) * abs(mean_p - obs_acc)
            cal_bins.append({
                "bin": lbl,
                "n_branches": len(pts_b),
                "mean_predicted_prob": round(mean_p, 4),
                "empirical_true_frac": round(obs_acc, 4),
            })

    summary["posterior_calibration"] = {
        "total_evaluated_branches": total_pts,
        "overall_mean_predicted_prob": round(sum(p for p, _, _ in all_cal_pts) / max(1, total_pts), 4),
        "overall_empirical_true_frac": round(sum(y for _, y, _ in all_cal_pts) / max(1, total_pts), 4),
        "expected_calibration_error": round(ece, 4),
        "bins": cal_bins,
    }
    summary["wall_sec"] = round(time.perf_counter() - t_start, 2)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {OUT_JSON} in {summary['wall_sec']} s (Calibration branches={total_pts}, ECE={ece:.4f})", flush=True)


if __name__ == "__main__":
    main()
