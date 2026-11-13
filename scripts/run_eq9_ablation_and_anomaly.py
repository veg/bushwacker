#!/usr/bin/env python3
"""
Four-Way Eq. (9) Ablation Study & 24-Species Anomaly-Zone SOTA Baselines.

Evaluates Bushwhacker in 4 continuous-time Eq. (9) configurations:
  1. default:                    Standard Eq. (9) + fixed Gamma(1,1) octiles
  2. no_mult_pen:                --no-mult-pen (ablates 0.5 ln(N-3) multiplicity term)
  3. empirical_alpha:            --empirical-alpha (uses Stage-1 alpha_hat octiles)
  4. no_mult_pen_empirical:      --no-mult-pen --empirical-alpha

Across:
  - AliSim star/radiation regimes (S08_star_null_N128, S08b_len_N200_L100, S10b_len_N200_L1000)
  - Protein & Clonal Null suites (P1-P4, N1) + empirical P_local reliability curves
  - Stratified 500-dataset Abadi-Gerth empirical-parameterized benchmark (sim_bic, c0, c1, c2)
  - 4 Tree-of-Life supermatrices (GTDB ar122, GTDB bac120, TOGA2 Mammal882 AA & NT)
  - 24-species MSC Anomaly Zone (10 reps x 200 loci -> ASTRAL-IV) + IQ-TREE UFBoot95/SH-aLRT80 baselines
"""

from __future__ import annotations

import concurrent.futures
import gzip
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

BUSHWHACKER_BIN = os.environ.get(
    "BUSHWHACKER_BIN",
    "/home/sergei/bushwhacker_bars_rev/bushwhacker_engine/target/release/bushwhacker",
)
IQTREE_BIN = os.environ.get("IQTREE_BIN", "/home/sergei/tools/iqtree-2.4.0-Linux-arm/bin/iqtree2")
FASTTREE_BIN = os.environ.get("FASTTREE_BIN", "/home/sergei/Development/FastTree/FastTreeMP")
ASTRAL4_BIN = os.environ.get("ASTRAL4_BIN", "/home/sergei/tree_pillars_2026/tools/ASTER/bin/astral4")
PHYML_BIN = os.environ.get("PHYML_BIN", "/home/sergei/bushwhacker_sims/phyml/src/phyml")

OUT_DIR = Path(os.environ.get("JOB_B_OUT_DIR", "/home/sergei/bushwhacker_bars_rev/job_b_ablation"))
OUT_ABLATION_JSON = OUT_DIR / "eq9_ablation_summary.json"
OUT_ANOMALY_JSON = OUT_DIR / "anomaly24_baselines.json"
MAX_WORKERS = int(os.environ.get("BW_WORKERS", "64"))

MODES = [
    ("default", []),
    ("no_mult_pen", ["--no-mult-pen"]),
    ("empirical_alpha", ["--empirical-alpha"]),
    ("no_mult_pen_empirical", ["--no-mult-pen", "--empirical-alpha"]),
]


# ---------------------------------------------------------------------------
# Fast Unrooted Bipartition & Support Helpers
# ---------------------------------------------------------------------------

class TreeNode:
    __slots__ = ("name", "length", "children")

    def __init__(self, name: str = "", length: float | None = None, children=None):
        self.name = name
        self.length = length
        self.children = children if children is not None else []


def parse_newick(nwk_str: str) -> TreeNode:
    s = nwk_str.strip()
    if s.endswith(";"):
        s = s[:-1]
    tokens = re.findall(r"'[^']*'|\[[^\]]*\]|\(|\)|,|:[^(),;\[\]]+|[^(),;:\[\]]+", s)
    root = TreeNode()
    stack = [root]
    curr = root
    for tok in tokens:
        if tok.startswith("["):
            continue
        if tok == "(":
            ch = TreeNode()
            curr.children.append(ch)
            stack.append(curr)
            curr = ch
        elif tok == ",":
            ch = TreeNode()
            stack[-1].children.append(ch)
            curr = ch
        elif tok == ")":
            curr = stack.pop()
        elif tok.startswith(":"):
            try:
                curr.length = float(tok[1:])
            except ValueError:
                curr.length = 0.0
        else:
            curr.name = tok.strip("'")
    while len(root.children) == 1 and not root.name:
        root = root.children[0]
    return root


def to_newick(nd: TreeNode, is_root: bool = True) -> str:
    if not nd.children:
        if nd.length is not None:
            return f"{nd.name}:{nd.length:.6f}"
        return nd.name
    inner = ",".join(to_newick(c, False) for c in nd.children)
    lbl = nd.name if nd.name else ""
    bl = f":{nd.length:.6f}" if (not is_root and nd.length is not None) else ""
    return f"({inner}){lbl}{bl};" if is_root else f"({inner}){lbl}{bl}"


def extract_splits_and_supports(nwk: str):
    root = parse_newick(nwk)
    leaves = []

    def gather(nd):
        if not nd.children:
            if nd.name:
                leaves.append(nd.name)
        else:
            for c in nd.children:
                gather(c)

    gather(root)
    n = len(leaves)
    if n < 4:
        return set(leaves), set(), {}
    t2b = {tx: (1 << i) for i, tx in enumerate(sorted(leaves))}
    full = (1 << n) - 1
    splits = set()
    meta = {}

    def walk(nd, is_rt: bool) -> int:
        if not nd.children:
            return t2b[nd.name] if nd.name in t2b else 0
        mask = 0
        for c in nd.children:
            mask |= walk(c, False)
        if not is_rt:
            k = bin(mask).count("1")
            if 2 <= k <= n - 2:
                comp = full ^ mask
                canon = mask if mask < comp else comp
                splits.add(canon)
                sup = None
                if nd.name:
                    for part in reversed(nd.name.split("/")):
                        try:
                            sup = float(part)
                            break
                        except ValueError:
                            pass
                bl = nd.length if nd.length is not None else 0.0
                if canon not in meta:
                    meta[canon] = [sup, bl]
                else:
                    if sup is not None and (meta[canon][0] is None or sup > meta[canon][0]):
                        meta[canon][0] = sup
                    meta[canon][1] += bl
        return mask

    walk(root, True)
    return set(leaves), splits, meta


def eval_vs_ref(est_nwk: str, ref_nwk: str, L: int | None = None):
    ref_leaves, ref_splits, ref_meta = extract_splits_and_supports(ref_nwk)
    est_leaves, est_splits, _ = extract_splits_and_supports(est_nwk)
    shared = ref_leaves & est_leaves
    if len(shared) < 4:
        return None
    if shared != ref_leaves or shared != est_leaves:
        # Restrict bitmasks to shared leaves via string-free re-parse on shared subset
        return eval_shared_subset(est_nwk, ref_nwk, shared, L)
    n_true = len(ref_splits)
    n_est = len(est_splits)
    tp = len(ref_splits & est_splits)
    fp = n_est - tp
    fn = n_true - tp
    rf_bin = fp + fn
    nrf_bin = rf_bin / (2.0 * n_true) if n_true > 0 else (0.0 if n_est == 0 else float(n_est))
    prec = tp / n_est if n_est > 0 else 1.0
    rec = tp / n_true if n_true > 0 else 1.0
    out = {
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
    if L is not None:
        act = {sp for sp, (_, bl) in ref_meta.items() if bl * L >= 1.0}
        tp1 = len(act & est_splits)
        fp1 = len(est_splits - act)
        fn1 = len(act - est_splits)
        rf1 = fp1 + fn1
        den1 = len(act) + max(len(act), n_est)
        out["nrf_ge1"] = rf1 / den1 if den1 > 0 else 0.0
        out["recall_ge1"] = tp1 / len(act) if act else 1.0
    return out


def prune_to_set(nd: TreeNode, keep: set[str]):
    if not nd.children:
        return TreeNode(nd.name, nd.length) if nd.name in keep else None
    kc = [prune_to_set(c, keep) for c in nd.children]
    kc = [c for c in kc if c is not None]
    if not kc:
        return None
    if len(kc) == 1:
        ch = kc[0]
        if nd.length is not None:
            ch.length = (ch.length or 0.0) + nd.length
        return ch
    return TreeNode(nd.name, nd.length, kc)


def eval_shared_subset(est_nwk: str, ref_nwk: str, shared: set[str], L: int | None = None):
    e_pr = prune_to_set(parse_newick(est_nwk), shared)
    r_pr = prune_to_set(parse_newick(ref_nwk), shared)
    return eval_vs_ref(to_newick(e_pr), to_newick(r_pr), L)


def cal_points_from_splits(est_nwk: str, ref_nwk: str):
    _, ref_splits, _ = extract_splits_and_supports(ref_nwk)
    _, _, est_meta = extract_splits_and_supports(est_nwk)
    pts = []
    for sp, (sup, _) in est_meta.items():
        if sup is not None:
            pts.append((sup, 1 if sp in ref_splits else 0))
    return pts


def summarize_cal_pts(pts):
    bins = [
        (0.50, 0.65, "0.50-0.65"),
        (0.65, 0.80, "0.65-0.80"),
        (0.80, 0.90, "0.80-0.90"),
        (0.90, 0.95, "0.90-0.95"),
        (0.95, 0.99, "0.95-0.99"),
        (0.99, 1.0001, "0.99-1.00"),
    ]
    n_tot = len(pts)
    ece = 0.0
    out_bins = []
    for lo, hi, lbl in bins:
        sub = [(p, y) for p, y in pts if lo <= p < hi]
        if sub:
            mp = sum(p for p, _ in sub) / len(sub)
            oa = sum(y for _, y in sub) / len(sub)
            ece += (len(sub) / max(1, n_tot)) * abs(mp - oa)
            out_bins.append({
                "bin": lbl,
                "n_branches": len(sub),
                "mean_predicted_prob": round(mp, 4),
                "empirical_true_frac": round(oa, 4),
            })
    return {
        "n_branches": n_tot,
        "overall_mean_predicted_prob": round(sum(p for p, _ in pts) / max(1, n_tot), 4),
        "overall_empirical_true_frac": round(sum(y for _, y in pts) / max(1, n_tot), 4),
        "expected_calibration_error": round(ece, 4),
        "bins": out_bins,
    }


def avg_metrics(items):
    items = [x for x in items if x is not None]
    if not items:
        return {}
    keys = sorted({k for d in items for k in d if isinstance(d[k], (int, float))})
    return {k: round(sum(d[k] for d in items if k in d) / len(items), 5) for k in keys} | {"n": len(items)}


# ---------------------------------------------------------------------------
# 1. AliSim S08 / S08b / S10b
# ---------------------------------------------------------------------------

def _run_alisim_task(args):
    sc_id, N, L, tree_spec, rep, seed = args
    rng = random.Random(seed)
    model = "GTR{1.5/3.0/0.8/1.2/4.0}+F{0.30/0.20/0.20/0.30}+G4{0.5}"
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        t_path = td_p / "true.nwk"
        if tree_spec == "STAR_CUSTOM":
            parts = [f"T{i:04d}:{rng.uniform(0.01, 0.10):.5f}" for i in range(N)]
            true_nwk = "(" + ",".join(parts) + ");"
            t_path.write_text(true_nwk + "\n")
            subprocess.run(
                [IQTREE_BIN, "--alisim", str(td_p / "sim"), "-t", str(t_path), "-m", model, "--length", str(L), "-af", "fasta", "-seed", str(seed), "-quiet"],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        else:
            subprocess.run(
                [IQTREE_BIN, "--alisim", str(td_p / "sim"), "-t", f"{tree_spec}", "-rlen", "0.001", "0.02", "0.10", "-m", model, "--length", str(L), "-af", "fasta", "-seed", str(seed), "-quiet"],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            true_nwk = (td_p / "sim.treefile").read_text().strip()
        fa_path = td_p / "sim.fa"
        res = {"scenario": sc_id, "rep": rep, "modes": {}}
        for mname, flags in MODES:
            out_nwk = td_p / f"{mname}.nwk"
            t0 = time.perf_counter()
            subprocess.run(
                [BUSHWHACKER_BIN, str(fa_path), "--no-shunts", "-T", "1", "-o", str(out_nwk)] + flags,
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            dt = time.perf_counter() - t0
            est_nwk = out_nwk.read_text().strip()
            ev = eval_vs_ref(est_nwk, true_nwk, L=L)
            ev["time_s"] = dt
            ev["cal_pts"] = cal_points_from_splits(est_nwk, true_nwk)
            res["modes"][mname] = ev
        return res


# ---------------------------------------------------------------------------
# 2. P1-P4 and N1 Ablation
# ---------------------------------------------------------------------------

def _run_pn_task(args):
    sys.path.insert(0, "/home/sergei/bushwhacker_bars_rev")
    import run_multialphabet_validation as rmv

    kind, scenario, rep, seed = args
    rng = random.Random(seed)
    with tempfile.TemporaryDirectory() as td:
        if kind == "protein":
            if scenario == "P1_StarNull":
                n_taxa, length, tree_nwk, model = 64, 500, rmv.gen_star_newick(64, rng, 0.05, 0.18), "LG+G4{0.8}"
            elif scenario == "P2_RapidRad":
                n_taxa, length, tree_nwk, model = 64, 500, rmv.gen_yule_newick(64, rng, zero_frac=0.50, int_min=0.025, int_max=0.10, leaf_min=0.04, leaf_max=0.18), "LG+G4{0.8}"
            elif scenario == "P3_StandardYule":
                n_taxa, length, tree_nwk, model = 64, 600, rmv.gen_yule_newick(64, rng, zero_frac=0.0, int_min=0.02, int_max=0.12, leaf_min=0.04, leaf_max=0.22), "LG+G4{0.7}"
            else:
                n_taxa, length, model = 48, 600, "LG+G4{0.7}"
                tree_nwk, _, _, c0, c5, _ = rmv.gen_structured_8clade_trees(rng)
            Path(f"{td}/true.nwk").write_text(tree_nwk + "\n")
            subprocess.run([IQTREE_BIN, "--alisim", f"{td}/sim", "-m", model, "--length", str(length), "-seed", str(seed), "-t", f"{td}/true.nwk"],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            seqs = rmv.parse_seq_phylip(f"{td}/sim.phy")
            if scenario == "P4_ConvergentAA":
                others = [nm for nm in seqs if nm not in c0 and nm not in c5]
                candidates = [(pos, next(iter({seqs[nm][pos] for nm in others}))) for pos in range(20, length - 20) if len({seqs[nm][pos] for nm in others}) == 1]
                rng.shuffle(candidates)
                spaced = []
                for pos, anc_aa in candidates:
                    if all(abs(pos - p0) >= 35 for p0, _ in spaced):
                        spaced.append((pos, anc_aa))
                    if len(spaced) == 6:
                        break
                seqs_mut = {nm: list(sq) for nm, sq in seqs.items()}
                for pos, anc_aa in spaced:
                    der_aa = rng.choice([a for a in "AILMVFWYSTNQ" if a != anc_aa])
                    for nm in (c0 | c5):
                        seqs_mut[nm][pos] = der_aa
                seqs = {nm: "".join(arr) for nm, arr in seqs_mut.items()}
            fa_path = f"{td}/aln.fa"
            rmv.write_fasta(fa_path, seqs)
            extra = ["--protein"]
        else:
            length = 1200
            tree_nwk, _, _, _, _, _ = rmv.gen_structured_8clade_trees(rng)
            Path(f"{td}/true.nwk").write_text(tree_nwk + "\n")
            subprocess.run([IQTREE_BIN, "--alisim", f"{td}/sim", "-m", "GTR{2.2,1.0,1.0,1.0,2.6}+F{0.28,0.22,0.23,0.27}+G4{0.75}", "--length", str(length), "-seed", str(seed), "-t", f"{td}/true.nwk"],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            seqs = rmv.parse_seq_phylip(f"{td}/sim.phy")
            fa_path = f"{td}/aln.fa"
            rmv.write_fasta(fa_path, seqs)
            extra = ["--no-coding"]

        res = {"scenario": scenario, "rep": rep, "modes": {}}
        for mname, flags in MODES:
            out_nwk = f"{td}/{mname}.nwk"
            out_j = f"{td}/{mname}.json"
            t0 = time.perf_counter()
            subprocess.run([BUSHWHACKER_BIN, fa_path, "-T", "1", "-o", out_nwk, "-j", out_j] + extra + flags,
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            dt = time.perf_counter() - t0
            est_nwk = Path(out_nwk).read_text().strip()
            jd = json.loads(Path(out_j).read_text())
            ev = eval_vs_ref(est_nwk, tree_nwk, L=length)
            ev["time_s"] = dt
            ev["n_shunts"] = len(jd.get("shunts", []))
            ev["cal_pts"] = cal_points_from_splits(est_nwk, tree_nwk)
            res["modes"][mname] = ev
        return res


# ---------------------------------------------------------------------------
# 3. Stratified 500-Dataset Abadi-Gerth Eq.(9) Ablation (sim_bic, c0, c1, c2)
# ---------------------------------------------------------------------------

def _phy_to_fa_and_clean(phy_path: Path, fa_out: Path, clean_out: Path):
    lines = [l.rstrip("\r\n") for l in phy_path.read_text(errors="ignore").splitlines() if l.strip()]
    hdr = lines[0].split()
    ntaxa, nchars = int(hdr[0]), int(hdr[1])
    seqs = []
    idx = 0
    for ln in lines[1:]:
        s = ln.strip()
        if not s:
            continue
        if len(seqs) < ntaxa:
            parts = s.split(None, 1)
            seqs.append([parts[0], re.sub(r"\s+", "", parts[1]) if len(parts) > 1 else ""])
        else:
            seqs[idx][1] += re.sub(r"\s+", "", s)
            idx = (idx + 1) % ntaxa
    with open(fa_out, "w") as f:
        for nm, sq in seqs:
            f.write(f">{nm}\n{sq}\n")
    with open(clean_out, "w") as f:
        f.write(f" {ntaxa} {nchars}\n")
        for nm, sq in seqs:
            f.write(f"{nm}  {sq}\n")
    return ntaxa, nchars


def _run_abadi_one(ds_id: int):
    ag_dir = Path("/home/sergei/bushwhacker_sims/abadi_gerth")
    ds_json = ag_dir / "results" / f"ds_{ds_id}.json"
    emp_phy = ag_dir / "empirical" / f"dataset_{ds_id}.phy"
    if not ds_json.exists() or not emp_phy.exists():
        return None
    meta = json.loads(ds_json.read_text())
    best_bic = meta.get("best_bic_model", "GTR+F+I+G4")
    out = {"ds_id": ds_id}
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        emp_fa = td_p / "emp.fa"
        emp_clean = td_p / "emp.phy"
        N, L = _phy_to_fa_and_clean(emp_phy, emp_fa, emp_clean)
        # Reconstruct PhyML JC+BioNJ tree -> c0/c1/c2 generating tree
        subprocess.run(
            [PHYML_BIN, "-i", str(emp_clean), "-m", "000000", "-f", "0.25,0.25,0.25,0.25",
             "-c", "1", "-o", "lr", "-d", "nt", "-n", "1", "-b", "0", "--no_memory_check", "--run_id", "JC"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=90,
        )
        ptf = Path(str(emp_clean) + "_phyml_tree_JC.txt")
        phyml_nwk = ptf.read_text().strip() if ptf.exists() else None

        # Matched sim_bic: fit best_bic model on empirical tree via IQ-TREE then AliSim
        try:
            subprocess.run(
                [IQTREE_BIN, "-s", str(emp_fa), "-m", best_bic, "--fast", "-nt", "1", "-pre", str(td_p / "bicfit"), "-quiet", "-redo"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120, check=True,
            )
            bic_tree = (td_p / "bicfit.treefile").read_text().strip()
            subprocess.run(
                [IQTREE_BIN, "--alisim", str(td_p / "sim_bic"), "-t", str(td_p / "bicfit.treefile"), "-m", best_bic,
                 "--length", str(L), "-af", "fasta", "-seed", str(100000 + ds_id), "-quiet"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120, check=True,
            )
            sim_fa = td_p / "sim_bic.fa"
            if sim_fa.exists():
                out["sim_bic"] = {}
                for mname, flags in MODES:
                    onwk = td_p / f"sb_{mname}.nwk"
                    subprocess.run([BUSHWHACKER_BIN, str(sim_fa), "--no-shunts", "-T", "1", "-o", str(onwk)] + flags,
                                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
                    out["sim_bic"][mname] = eval_vs_ref(onwk.read_text().strip(), bic_tree, L=L)
        except Exception:
            pass

        if phyml_nwk:
            for c_set in ("c0", "c1", "c2"):
                c_phy = ag_dir / c_set / f"dataset_{ds_id}.phy"
                if not c_phy.exists():
                    continue
                c_fa = td_p / f"{c_set}.fa"
                c_cl = td_p / f"{c_set}.phy"
                _, cL = _phy_to_fa_and_clean(c_phy, c_fa, c_cl)
                out[c_set] = {}
                for mname, flags in MODES:
                    onwk = td_p / f"{c_set}_{mname}.nwk"
                    subprocess.run([BUSHWHACKER_BIN, str(c_fa), "--no-shunts", "-T", "1", "-o", str(onwk)] + flags,
                                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
                    out[c_set][mname] = eval_vs_ref(onwk.read_text().strip(), phyml_nwk, L=cL)
    return out


# ---------------------------------------------------------------------------
# 4. Supermatrices (4 ToL benchmarks) Across All 4 Eq.(9) Modes
# ---------------------------------------------------------------------------

def load_gtdb_taxonomy(tsv_paths):
    tax = {}
    for p in tsv_paths:
        if not Path(p).exists():
            continue
        for line in Path(p).read_text().splitlines():
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            gid = parts[0].strip()
            ranks = {}
            for item in parts[1].split(";"):
                if "__" in item:
                    k, v = item.split("__", 1)
                    ranks[k.strip()] = v.strip()
            tax[gid] = ranks
    return tax


def eval_group_monophyly(est_nwk: str, taxon_to_group: dict):
    root = parse_newick(est_nwk)
    leaves = []

    def g(nd):
        if not nd.children:
            if nd.name and taxon_to_group.get(nd.name):
                leaves.append(nd.name)
        else:
            for c in nd.children:
                g(c)

    g(root)
    if len(leaves) < 4:
        return {}
    t2b = {tx: (1 << i) for i, tx in enumerate(leaves)}
    full = (1 << len(leaves)) - 1
    splits = set()

    def walk(nd, is_rt):
        if not nd.children:
            return t2b.get(nd.name, 0)
        m = 0
        for c in nd.children:
            m |= walk(c, False)
        if not is_rt:
            k = bin(m).count("1")
            if 2 <= k <= len(leaves) - 2:
                comp = full ^ m
                splits.add(m if m < comp else comp)
        return m

    walk(root, True)
    grp_masks = defaultdict(int)
    grp_counts = defaultdict(int)
    for tx in leaves:
        gr = taxon_to_group[tx]
        grp_masks[gr] |= t2b[tx]
        grp_counts[gr] += 1
    valid = {gr: m for gr, m in grp_masks.items() if 2 <= grp_counts[gr] <= len(leaves) - 2}
    strict = 0
    for gr, gm in valid.items():
        comp = full ^ gm
        canon = gm if gm < comp else comp
        if canon in splits:
            strict += 1
    return {"n_groups": len(valid), "strict_monophyletic": strict, "strict_frac": round(strict / max(1, len(valid)), 4)}


def run_supermatrix_ablations():
    ar_tax = load_gtdb_taxonomy(["/home/sergei/bushwhacker_sims/gtdb_r95/ar122_taxonomy_r95.tsv"])
    bac_tax = load_gtdb_taxonomy(["/home/sergei/bushwhacker_sims/gtdb_r95/bac120_taxonomy_r95.tsv"])
    specs = [
        ("GTDB_ar122_AA", "/home/sergei/tree_pillars_2026/work_pillar2/gtdb_supermatrix/ar122_supermatrix.faa",
         "/home/sergei/bushwhacker_sims/gtdb_r95/ar122_r95.tree", True, ar_tax),
        ("GTDB_bac120_AA", "/home/sergei/tree_pillars_2026/work_pillar2/gtdb_supermatrix/bac120_supermatrix_1000.faa",
         "/home/sergei/bushwhacker_sims/gtdb_r95/bac120_r95.tree", True, bac_tax),
        ("TOGA2_Mammal882_AA", "/home/sergei/tree_pillars_2026/work_pillar2/toga2_mammal882/toga2_mammal882_aa.faa",
         "/home/sergei/toga2_walrus/speciesTree.nh", True, None),
        ("TOGA2_Mammal882_NT", "/home/sergei/tree_pillars_2026/work_pillar2/toga2_mammal882/toga2_mammal882_nt.fna",
         "/home/sergei/toga2_walrus/speciesTree.nh", False, None),
    ]
    out = {}
    with tempfile.TemporaryDirectory() as td:
        for name, aln_p, ref_p, is_aa, tax in specs:
            ref_nwk = Path(ref_p).read_text().strip()
            out[name] = {}
            for mname, flags in MODES:
                onwk = f"{td}/{name}_{mname}.nwk"
                oj = f"{td}/{name}_{mname}.json"
                cmd = [BUSHWHACKER_BIN, aln_p, "-T", str(min(32, MAX_WORKERS)), "--no-shunts", "-o", onwk, "-j", oj] + (["--protein"] if is_aa else []) + flags
                t0 = time.perf_counter()
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                dt = time.perf_counter() - t0
                est_nwk = Path(onwk).read_text().strip()
                jd = json.loads(Path(oj).read_text())
                ev = eval_vs_ref(est_nwk, ref_nwk)
                ev["wall_sec"] = round(dt, 2)
                ev["resolved_frac"] = round(jd["resolved_internal_splits"] / max(1, jd["max_binary_splits"]), 4)
                if tax is not None:
                    ev["phylum_monophyly"] = eval_group_monophyly(est_nwk, {k: v.get("p", "") for k, v in tax.items()})
                    ev["class_monophyly"] = eval_group_monophyly(est_nwk, {k: v.get("c", "") for k, v in tax.items()})
                out[name][mname] = ev
                print(f"  [Supermatrix] {name} / {mname}: prec={ev['precision']:.4f} rec={ev['recall']:.4f} nRF={ev['nrf_bin']:.4f}", flush=True)
    return out


# ---------------------------------------------------------------------------
# 5. MSC_Anomaly24_L400: 4 Eq.(9) Modes + IQ-TREE UFBoot95 / SH-aLRT80 -> ASTRAL-IV
# ---------------------------------------------------------------------------

def collapse_by_pair_support(nwk: str, idx_from_end: int, thresh: float) -> str:
    """Collapse internal branches whose slash-separated support field (e.g. alrt/ufboot) < thresh."""
    nd = parse_newick(nwk)

    def _rec(u: TreeNode, is_rt: bool) -> list[TreeNode]:
        if not u.children:
            return [u]
        new_ch = []
        for c in u.children:
            new_ch.extend(_rec(c, False))
        u.children = new_ch
        if not is_rt and u.name:
            parts = u.name.split("/")
            try:
                val = float(parts[idx_from_end])
                if val < thresh:
                    return u.children
            except (ValueError, IndexError):
                pass
        return [u]

    _rec(nd, True)
    return to_newick(nd, True)


def _anomaly_locus_worker(args):
    gt_nwk, L, seed, do_uf = args
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        Path(td_p / "gt.nwk").write_text(gt_nwk + "\n")
        subprocess.run(
            [IQTREE_BIN, "--alisim", str(td_p / "sim"), "-t", str(td_p / "gt.nwk"),
             "-m", "GTR{1.8,3.2,0.9,1.1,3.5}+F{0.28,0.22,0.22,0.28}+G4{0.6}",
             "--length", str(L), "-af", "fasta", "-seed", str(seed), "-quiet"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        fa_p = td_p / "sim.fa"
        out = {}
        for mname, flags in MODES:
            onwk = td_p / f"{mname}.nwk"
            t0 = time.perf_counter()
            subprocess.run([BUSHWHACKER_BIN, str(fa_p), "--no-shunts", "-T", "1", "-o", str(onwk)] + flags,
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            out[mname] = (onwk.read_text().strip(), time.perf_counter() - t0)
        if do_uf:
            t0 = time.perf_counter()
            subprocess.run(
                [IQTREE_BIN, "-s", str(fa_p), "-m", "GTR+G4", "-B", "1000", "-alrt", "1000", "-T", "1",
                 "-pre", str(td_p / "iq"), "-quiet", "-redo"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            dt_uf = time.perf_counter() - t0
            tf = td_p / "iq.treefile"
            raw_iq = tf.read_text().strip() if tf.exists() else gt_nwk
            # IQ-TREE formats SH-aLRT/UFBoot as "alrt/ufboot"
            out["iq_uf95"] = (collapse_by_pair_support(raw_iq, -1, 95.0), dt_uf)
            out["iq_alrt80"] = (collapse_by_pair_support(raw_iq, -2, 80.0), dt_uf)
        return out


def run_anomaly24_ablations_and_baselines():
    sys.path.insert(0, "/home/sergei/tree_pillars_2026")
    import run_pillar1_astral_species_trees as p1

    sc_id = "MSC_Anomaly24_L400"
    n_sp, n_loci, L, theta, n_reps, uf_reps = 24, 200, 400, 0.025, 10, 5
    agg = defaultdict(lambda: {"nrf": [], "rf": [], "fp": [], "fn": [], "cpu_sec": [], "astral_sec": []})

    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        for rep in range(n_reps):
            rng = random.Random(20261110 + hash(sc_id) % 100000 + rep * 997)
            sp_root = p1.build_anomaly_zone_tree(n_sp, rng)
            true_sp_nwk = p1.spnode_to_newick(sp_root)
            do_uf = rep < uf_reps
            tasks = []
            for loc in range(n_loci):
                gt_nwk = p1.simulate_msc_gene_tree(sp_root, theta, rng)
                seed = 100000 + rep * 1000 + loc + 1
                tasks.append((gt_nwk, L, seed, do_uf))
            with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
                locs = list(ex.map(_anomaly_locus_worker, tasks, chunksize=4))
            active_keys = [m for m, _ in MODES] + (["iq_uf95", "iq_alrt80"] if do_uf else [])
            for mk in active_keys:
                in_tre = td_p / f"r{rep}_{mk}.in.tre"
                out_tre = td_p / f"r{rep}_{mk}.out.tre"
                in_tre.write_text("\n".join(x[mk][0].rstrip(";") + ";" for x in locs) + "\n")
                cpu_s = sum(x[mk][1] for x in locs)
                t0 = time.perf_counter()
                subprocess.run([ASTRAL4_BIN, "-t", "8", "-i", str(in_tre), "-o", str(out_tre)],
                               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                a_s = time.perf_counter() - t0
                ev = eval_vs_ref(out_tre.read_text().strip(), true_sp_nwk)
                agg[mk]["nrf"].append(ev["nrf_bin"])
                agg[mk]["rf"].append(ev["rf_bin"])
                agg[mk]["fp"].append(ev["fp"])
                agg[mk]["fn"].append(ev["fn"])
                agg[mk]["cpu_sec"].append(cpu_s)
                agg[mk]["astral_sec"].append(a_s)

    summary = {}
    for mk, vals in agg.items():
        k = len(vals["nrf"])
        m_cpu = sum(vals["cpu_sec"]) / k
        m_ast = sum(vals["astral_sec"]) / k
        summary[mk] = {
            "n_reps": k,
            "mean_nrf": round(sum(vals["nrf"]) / k, 5),
            "mean_rf": round(sum(vals["rf"]) / k, 2),
            "mean_fp": round(sum(vals["fp"]) / k, 2),
            "mean_fn": round(sum(vals["fn"]) / k, 2),
            "mean_gene_tree_cpu_sec": round(m_cpu, 2),
            "mean_wall_64core_sec": round(m_cpu / 64.0 + m_ast, 3),
            "mean_astral_sec": round(m_ast, 3),
        }
    return summary


# ---------------------------------------------------------------------------
# Main Driver
# ---------------------------------------------------------------------------

def main():
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {"modes": [m for m, _ in MODES]}

    print(f"[Job B2] 1/5 AliSim S08, S08b, S10b ablations ({MAX_WORKERS} workers)...", flush=True)
    ali_specs = [
        ("S08_star_null_N128", 128, 1000, "STAR_CUSTOM"),
        ("S08b_len_N200_L100", 200, 100, "RANDOM{yh,200}"),
        ("S10b_len_N200_L1000", 200, 1000, "RANDOM{yh,200}"),
    ]
    ali_tasks = [(sc, N, L, ts, r, 410000 + si * 100 + r) for si, (sc, N, L, ts) in enumerate(ali_specs) for r in range(20)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        ali_res = list(ex.map(_run_alisim_task, ali_tasks))

    summary["alisim_ablations"] = {}
    cal_by_mode = defaultdict(list)
    for sc, _, _, _ in ali_specs:
        sub = [x for x in ali_res if x["scenario"] == sc]
        summary["alisim_ablations"][sc] = {}
        for mname, _ in MODES:
            summary["alisim_ablations"][sc][mname] = avg_metrics([x["modes"][mname] for x in sub])
            for x in sub:
                cal_by_mode[mname].extend(x["modes"][mname]["cal_pts"])
    print("  [Done] AliSim ablations", flush=True)

    print("[Job B2] 2/5 P1-P4 & N1 ablations...", flush=True)
    pn_tasks = [("protein", sc, r, 510000 + si * 100 + r) for si, sc in enumerate(["P1_StarNull", "P2_RapidRad", "P3_StandardYule", "P4_ConvergentAA"]) for r in range(25)]
    pn_tasks += [("network", "N1_ClonalNull", r, 520000 + r) for r in range(25)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        pn_res = list(ex.map(_run_pn_task, pn_tasks))

    summary["protein_and_network_ablations"] = {}
    for sc in ["P1_StarNull", "P2_RapidRad", "P3_StandardYule", "P4_ConvergentAA", "N1_ClonalNull"]:
        sub = [x for x in pn_res if x["scenario"] == sc]
        summary["protein_and_network_ablations"][sc] = {}
        for mname, _ in MODES:
            summary["protein_and_network_ablations"][sc][mname] = avg_metrics([x["modes"][mname] for x in sub])
            for x in sub:
                cal_by_mode[mname].extend(x["modes"][mname]["cal_pts"])
    summary["posterior_calibration_by_mode"] = {mname: summarize_cal_pts(cal_by_mode[mname]) for mname, _ in MODES}
    print("  [Done] P1-P4 & N1 ablations", flush=True)

    print("[Job B2] 3/5 Stratified 500-dataset Abadi-Gerth Eq.(9) ablations...", flush=True)
    all_ag_ids = sorted(
        int(p.stem.split("_")[1])
        for p in Path("/home/sergei/bushwhacker_sims/abadi_gerth/results").glob("ds_*.json")
    )
    step = max(1, len(all_ag_ids) // 500)
    sampled_ids = all_ag_ids[::step][:500]
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        ag_res = [x for x in ex.map(_run_abadi_one, sampled_ids, chunksize=4) if x is not None]
    summary["abadi_gerth_500_ablations"] = {}
    for corpus in ("sim_bic", "c0", "c1", "c2"):
        summary["abadi_gerth_500_ablations"][corpus] = {}
        for mname, _ in MODES:
            items = [x[corpus][mname] for x in ag_res if corpus in x and mname in x[corpus]]
            summary["abadi_gerth_500_ablations"][corpus][mname] = avg_metrics(items)
    print("  [Done] Abadi-Gerth 500 ablations", flush=True)

    print("[Job B2] 4/5 Tree-of-Life supermatrix ablations...", flush=True)
    summary["supermatrix_ablations"] = run_supermatrix_ablations()

    print("[Job B2] 5/5 MSC_Anomaly24_L400 ablations + UFBoot95/SH-aLRT80 baselines...", flush=True)
    anom = run_anomaly24_ablations_and_baselines()
    summary["msc_anomaly24_ablations_and_baselines"] = anom
    OUT_ANOMALY_JSON.write_text(json.dumps(anom, indent=2))

    summary["wall_sec"] = round(time.perf_counter() - t0, 2)
    OUT_ABLATION_JSON.write_text(json.dumps(summary, indent=2))
    print(f"[Job B2] ALL DONE in {summary['wall_sec']} s -> {OUT_ABLATION_JSON}", flush=True)


if __name__ == "__main__":
    main()
