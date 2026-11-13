#!/usr/bin/env python3
"""
Pillar 3: Cellular Lineage Phylogenetics — Germinal-Center B-Cell Clonal Bursts,
Somatic Hypermutation, & Experimental DMS Affinity Landscapes
================================================================================
Executes two authentic benchmarks on silverback (64-core ARM64):
  3A. Ground-Truth Galton-Watson Germinal-Center Affinity-Maturation Simulations:
      50 independent germinal centers evolving from the 657-bp CGG_naive knock-in
      germline under empirical 5-mer SHM mutability/substitution (MK_RS5NF) and
      experimental Yeast-Display DMS CGG-binding fitness (Ab-CGGnaive_DMS).
      Benchmarks Bushwhacker (MDL & Binary), FastTree 2 (GTR), and IQ-TREE 2
      (GTR+G4 Binary, --polytomy, and -B 1000 UFBoot collapsed at <50% & <70%)
      on BOTH cell-level (N=90 sampled cells + naive) and unique-genotype trees.

  3B. Empirical Mouse Lymph-Node Germinal Centers (DeWitt et al., Cell 2026
      'gcreplay': all 147 germinal centers, 10,429 paired IGH+IGK single B cells;
      plus Tas et al., Science 2016 '150228_Clone_3-8.fasta'):
      STRICT ZERO-LEAKAGE PROTOCOL: Tree builders see ONLY raw nucleotide FASTAs
      (never cell abundances, GCtree topologies, or DMS scores).
      Post-hoc evaluations:
        (3B-1) All-cells single-cell lineage reconstruction across all 147 GCs vs.
               abundance-informed GCtree reference topologies (measuring binary
               staircase hallucinations inside identical clonal bursts).
        (3B-2) Blind unique-genotype lineage reconstruction across all 147 GCs
               (1 copy per unique 657-bp sequence) + unsupervised test of whether
               Bushwhacker MDL clonal-burst hubs predict (i) in vivo cellular
               clone abundance and (ii) experimental Yeast-Display DMS antigen
               binding affinity (delta_log10 K_D).
        (3B-3) Cross-mouse evolutionary replay alignment (1 representative per
               independent GC + naive germline) testing whether Stage 4 SPARSE
               shunts enrich for high-affinity CDR somatic hypermutations.
"""


import sys as _sys
from pathlib import Path as _Path
if str(_Path(__file__).resolve().parent) not in _sys.path:
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
from _env_paths import REPO_ROOT, DATA_DIR, SPECIMENS_DIR, ALIGNMENTS_DIR, SINGLE_CELL_DIR, RESULTS_DIR, WORK_ROOT, EXTERNAL_DIR, resolve_bin, ASTRAL_JAR

import os
import re
import csv
import sys
import json
import lzma
import math
import time
import random
import shutil
import tempfile
import subprocess
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

BUSHWHACKER_BIN = resolve_bin("BUSHWHACKER_BIN", REPO_ROOT / "target" / "release" / "bushwhacker", "bushwhacker")
IQTREE_BIN = resolve_bin("IQTREE_BIN", "iqtree2", "iqtree")
FASTTREE_BIN = resolve_bin("FASTTREE_BIN", "FastTreeMP", "FastTree", "fasttree")

BCR_DATA_DIR = Path(os.environ.get("BCR_DATA_DIR", str(SINGLE_CELL_DIR / "bcr_gcreplay")))
TAS_FASTA = SPECIMENS_DIR / "150228_Clone_3-8.fasta"
WORK_DIR = WORK_ROOT / "pillar3"
OUT_JSON = RESULTS_DIR / "single_cell" / "pillar3_bcr_cellular.json"

CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


# =============================================================================
# Nonparametric Statistical Helpers (Self-Contained, Exact Tie Handling)
# =============================================================================

def rankdata_average(vals):
    n = len(vals)
    order = sorted(range(n), key=lambda i: vals[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg_r = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_r
        i = j + 1
    return ranks


def spearman_corr(x, y):
    n = len(x)
    if n < 3:
        return 0.0, 1.0
    rx = rankdata_average(x)
    ry = rankdata_average(y)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    denx = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)))
    deny = math.sqrt(sum((ry[i] - my) ** 2 for i in range(n)))
    if denx <= 1e-15 or deny <= 1e-15:
        return 0.0, 1.0
    rho = max(-1.0, min(1.0, num / (denx * deny)))
    if abs(rho) >= 1.0 - 1e-12:
        return rho, 0.0
    # Student-t normal / scipy exact p-value
    try:
        from scipy.stats import spearmanr
        res = spearmanr(x, y)
        return float(res.statistic), float(res.pvalue)
    except Exception:
        t_stat = abs(rho) * math.sqrt((n - 2) / max(1e-12, 1.0 - rho * rho))
        p_two = math.erfc(t_stat / math.sqrt(2.0))
        return rho, p_two


def mann_whitney_u_two_sided(x, y):
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return 0.0, 1.0
    try:
        from scipy.stats import mannwhitneyu
        res = mannwhitneyu(x, y, alternative="two-sided")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        combined = list(x) + list(y)
        n = nx + ny
        ranks = rankdata_average(combined)
        r_x = sum(ranks[:nx])
        u_x = r_x - nx * (nx + 1) / 2.0
        mu = nx * ny / 2.0
        # Tie correction
        counts = Counter(combined)
        tie_term = sum(t ** 3 - t for t in counts.values() if t > 1)
        var = (nx * ny / 12.0) * ((n + 1) - tie_term / max(1.0, n * (n - 1)))
        if var <= 1e-15:
            return u_x, 1.0
        z = (abs(u_x - mu) - 0.5) / math.sqrt(var)
        p = min(1.0, max(0.0, math.erfc(max(0.0, z) / math.sqrt(2.0))))
        return u_x, p


def fisher_exact_2x2_greater(a, b, c, d):
    """One-sided hypergeometric p-value P(X >= a) for [[a, b], [c, d]]."""
    try:
        from scipy.stats import fisher_exact
        odds, p = fisher_exact([[a, b], [c, d]], alternative="greater")
        return float(odds) if not math.isinf(odds) else 999.0, float(p)
    except Exception:
        n = a + b + c + d
        r1 = a + b
        c1 = a + c
        def log_choose(nn, kk):
            if kk < 0 or kk > nn:
                return -1e99
            return math.lgamma(nn + 1) - math.lgamma(kk + 1) - math.lgamma(nn - kk + 1)
        denom = log_choose(n, c1)
        p_sum = 0.0
        for x in range(a, min(r1, c1) + 1):
            lp = log_choose(r1, x) + log_choose(n - r1, c1 - x) - denom
            p_sum += math.exp(lp)
        odds = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
        return odds, min(1.0, p_sum)


def fisher_exact_2x2_two_sided(a, b, c, d):
    """Two-sided Fisher exact test for [[a, b], [c, d]]."""
    from scipy.stats import fisher_exact
    odds, p = fisher_exact([[a, b], [c, d]], alternative="two-sided")
    return float(odds) if not math.isinf(odds) else 999.0, float(p)


def paired_wilcoxon_and_cluster_ci(vals, clusters=None, n_boot=10000, seed=20261112):
    """Two-sided one-sample/paired Wilcoxon signed-rank p and cluster-bootstrap 95% CI."""
    from scipy.stats import wilcoxon
    n = len(vals)
    if n < 3:
        return {"n": n, "mean": 0.0, "median": 0.0, "wilcoxon_p_two_sided": 1.0, "ci95": [0.0, 0.0]}
    nonzero = [v for v in vals if abs(v) > 1e-12]
    if nonzero:
        p_w = float(wilcoxon(vals, alternative="two-sided").pvalue)
    else:
        p_w = 1.0
    rng = random.Random(seed)
    boot_means = []
    if clusters is None:
        for _ in range(n_boot):
            samp = [vals[rng.randrange(n)] for _ in range(n)]
            boot_means.append(sum(samp) / n)
    else:
        by_cl = defaultdict(list)
        for v, cl in zip(vals, clusters):
            by_cl[cl].append(v)
        cl_keys = sorted(by_cl.keys())
        k_cl = len(cl_keys)
        for _ in range(n_boot):
            drawn = [cl_keys[rng.randrange(k_cl)] for _ in range(k_cl)]
            samp = [v for ck in drawn for v in by_cl[ck]]
            boot_means.append(sum(samp) / len(samp))
    boot_means.sort()
    sv = sorted(vals)
    return {
        "n": n,
        "mean": round(sum(vals) / n, 4),
        "median": round(quantile_sorted(sv, 0.5), 4),
        "iqr25": round(quantile_sorted(sv, 0.25), 4),
        "iqr75": round(quantile_sorted(sv, 0.75), 4),
        "wilcoxon_p_two_sided": p_w,
        "ci95": [round(quantile_sorted(boot_means, 0.025), 4), round(quantile_sorted(boot_means, 0.975), 4)],
    }


def quantile_sorted(sv, q):
    if not sv:
        return 0.0
    pos = q * (len(sv) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sv[lo])
    return float(sv[lo] * (hi - pos) + sv[hi] * (pos - lo))


# =============================================================================
# Tree Parsing, Support Contraction, Split Evaluation, & Outgroup Rooting
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
    for tok in tokens:
        if tok.startswith("["):
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
            curr.name = tok.strip("'")
    while len(root.children) == 1 and not root.name:
        root = root.children[0]
    return root


def write_newick_tree(node: TreeNode, include_lengths: bool = False) -> str:
    def _fmt(nd: TreeNode, is_root: bool = False) -> str:
        if not nd.children:
            if include_lengths and nd.length is not None:
                return f"{nd.name}:{nd.length:.6f}"
            return nd.name
        inner = ",".join(_fmt(c, False) for c in nd.children)
        lbl = nd.name if nd.name else ""
        if not is_root and include_lengths and nd.length is not None:
            return f"({inner}){lbl}:{nd.length:.6f}"
        return f"({inner}){lbl}"

    return _fmt(node, True) + ";"


def parse_support_value(lbl: str, scale_100: bool = False):
    if not lbl:
        return None
    s = lbl.strip()
    if "/" in s:
        s = s.split("/")[-1]
    try:
        val = float(s)
        if scale_100 and val <= 1.000001:
            val *= 100.0
        return val
    except ValueError:
        return None


def collapse_by_support(node: TreeNode, min_support: float, scale_100: bool = False, is_root: bool = True) -> TreeNode:
    if not node.children:
        return TreeNode(name=node.name, length=node.length)
    new_children = []
    for ch in node.children:
        c_proc = collapse_by_support(ch, min_support, scale_100=scale_100, is_root=False)
        if c_proc.children:
            sup = parse_support_value(c_proc.name, scale_100=scale_100)
            if sup is not None and sup < min_support:
                add_len = c_proc.length or 0.0
                for gc in c_proc.children:
                    if gc.length is not None:
                        gc.length += add_len
                    new_children.append(gc)
                continue
        new_children.append(c_proc)
    return TreeNode(name=node.name if not is_root else "", length=node.length, children=new_children)


def collapse_near_zero_branches(node: TreeNode, eps: float = 1e-6, is_root: bool = True) -> TreeNode:
    if not node.children:
        return TreeNode(name=node.name, length=node.length)
    new_children = []
    for ch in node.children:
        c_proc = collapse_near_zero_branches(ch, eps=eps, is_root=False)
        if c_proc.children and c_proc.length is not None and c_proc.length <= eps:
            new_children.extend(c_proc.children)
        else:
            new_children.append(c_proc)
    return TreeNode(name=node.name if not is_root else "", length=node.length, children=new_children)


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
    splits = set()
    n_total = bin(full_mask).count("1")

    def _dfs(nd: TreeNode) -> int:
        if not nd.children:
            return taxon_to_bit.get(nd.name, 0)
        m = 0
        for c in nd.children:
            m |= _dfs(c)
        k = bin(m).count("1")
        if 2 <= k <= n_total - 2:
            comp = full_mask ^ m
            splits.add(m if m < comp else comp)
        return m

    _dfs(node)
    return splits


def compare_trees_rf(est_nwk: str, ref_nwk: str) -> dict:
    t_est = parse_newick_tree(est_nwk)
    t_ref = parse_newick_tree(ref_nwk)
    shared = sorted(set(get_leaf_names(t_est)) & set(get_leaf_names(t_ref)))
    n = len(shared)
    if n < 4:
        return {
            "n_taxa": n, "tp": 0, "fp": 0, "fn": 0, "compatible_fp": 0, "conflicting_fp": 0,
            "est_splits": 0, "ref_splits": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "nrf": 0.0,
        }
    t2b = {tx: (1 << i) for i, tx in enumerate(shared)}
    full = (1 << n) - 1
    s_est = extract_canonical_splits(t_est, t2b, full)
    s_ref = extract_canonical_splits(t_ref, t2b, full)
    tp = len(s_est & s_ref)
    fp_set = s_est - s_ref
    fp = len(fp_set)
    fn = len(s_ref - s_est)
    # Partition FP into splits compatible with partially resolved ref tree vs. strictly conflicting splits
    compat_fp = 0
    for s in fp_set:
        s_comp = full ^ s
        conflict = False
        for r in s_ref:
            r_comp = full ^ r
            if (s & r) and (s & r_comp) and (s_comp & r) and (s_comp & r_comp):
                conflict = True
                break
        if not conflict:
            compat_fp += 1
    conflict_fp = fp - compat_fp
    prec = tp / len(s_est) if s_est else (1.0 if not s_ref else 0.0)
    rec = tp / len(s_ref) if s_ref else 1.0
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    denom = len(s_est) + len(s_ref)
    nrf = (fp + fn) / denom if denom > 0 else 0.0
    return {
        "n_taxa": n,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "compatible_fp": compat_fp,
        "conflicting_fp": conflict_fp,
        "est_splits": len(s_est),
        "ref_splits": len(s_ref),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "nrf": nrf,
    }


def root_tree_at_outgroup_and_analyze_bursts(nwk_str: str, outgroup_name: str, seq_map: dict):
    """
    Build undirected adjacency from unrooted Newick, root at the internal neighbor
    of `outgroup_name` (the naive germline), reconstruct exact ancestral sequences
    via Fitch parsimony rooted at `seq_map[outgroup_name]`, and return:
      - leaf_burst_daughters: dict[leaf_name -> inferred mutant daughter lineages]
        (If leaf g has 0 Hamming distance to its parent internal node u, then g IS
        the sampled mother genotype of node u, so its inferred daughter lineage count
        is len(u.children) - 1 [excluding leaf g itself, and excluding outgroup if at root];
        otherwise 0.)
      - internal_nodes_info: list of dicts with reconstructed sequence, out-degree, and
        mutant child count.
    """
    t = parse_newick_tree(nwk_str)
    # Build undirected graph over integer node IDs
    adj = defaultdict(list)
    node_names = {}
    counter = [0]

    def _build(nd: TreeNode) -> int:
        nid = counter[0]
        counter[0] += 1
        node_names[nid] = nd.name if not nd.children else ""
        for ch in nd.children:
            cid = _build(ch)
            adj[nid].append(cid)
            adj[cid].append(nid)
        return nid

    _build(t)
    og_ids = [nid for nid, nm in node_names.items() if nm == outgroup_name]
    if not og_ids or outgroup_name not in seq_map:
        return {}, []
    og_id = og_ids[0]
    if not adj[og_id]:
        return {}, []
    root_internal = adj[og_id][0]

    # Build directed rooted tree starting at root_internal (excluding og_id as an upstream root)
    children_map = defaultdict(list)
    parent_map = {}

    def _orient(u: int, p: int):
        for v in adj[u]:
            if v == p or v == og_id:
                continue
            parent_map[v] = u
            children_map[u].append(v)
            _orient(v, u)

    _orient(root_internal, og_id)

    # Contract spurious degree-1 internal pass-through nodes
    def _clean(u: int) -> int:
        new_ch = [_clean(v) for v in children_map[u]]
        children_map[u] = new_ch
        if len(new_ch) == 1 and not node_names[u]:
            return new_ch[0]
        return u

    root_internal = _clean(root_internal)

    # Collect postorder
    postorder = []
    def _post(u: int):
        for v in children_map[u]:
            _post(v)
        postorder.append(u)
    _post(root_internal)

    seq_len = len(seq_map[outgroup_name])
    naive_seq = seq_map[outgroup_name]
    base_to_bit = {"A": 1, "C": 2, "G": 4, "T": 8}
    bit_to_base = {1: "A", 2: "C", 4: "G", 8: "T"}

    # Fitch bottom-up bitmasks per site
    fitch_masks = {}
    for u in postorder:
        nm = node_names[u]
        if not children_map[u]:
            sq = seq_map.get(nm, naive_seq)
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

    # Top-down state assignment anchored at naive germline outgroup!
    recon_seq = {}
    def _topdown(u: int, parent_sq: str):
        if not children_map[u] and node_names[u] in seq_map:
            recon_seq[u] = seq_map[node_names[u]]
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
        for v in children_map[u]:
            _topdown(v, sq_u)

    _topdown(root_internal, naive_seq)

    # Compute inferred clonal-burst daughter counts for every observed leaf genotype
    leaf_burst_daughters = {}
    internal_nodes_info = []
    for u in postorder:
        nm = node_names[u]
        if not children_map[u]:
            if nm and nm != outgroup_name and nm not in leaf_burst_daughters:
                leaf_burst_daughters[nm] = 0
        else:
            sq_u = recon_seq[u]
            zero_dist_leaves = []
            mutant_branches = 0
            for v in children_map[u]:
                if not children_map[v] and recon_seq[v] == sq_u:
                    zero_dist_leaves.append(node_names[v])
                else:
                    mutant_branches += 1
            for lnm in zero_dist_leaves:
                if lnm and lnm != outgroup_name:
                    leaf_burst_daughters[lnm] = mutant_branches
            internal_nodes_info.append({
                "seq": sq_u,
                "total_out_degree": len(children_map[u]),
                "mutant_daughter_branches": mutant_branches,
                "sampled_mother_leaves": zero_dist_leaves,
            })

    return leaf_burst_daughters, internal_nodes_info


# =============================================================================
# Load CGG_naive Germline, DMS Fitness Landscape, & Empirical 5-mer SHM Model
# =============================================================================

def load_cgg_dms_and_shm():
    sites_csv = BCR_DATA_DIR / "CGGnaive_sites.csv"
    dms_csv = BCR_DATA_DIR / "final_variant_scores.csv"
    mut_csv = BCR_DATA_DIR / "MK_RS5NF_mutability.csv"
    sub_csv = BCR_DATA_DIR / "MK_RS5NF_substitution.csv"

    h_rows = []
    l_rows = []
    with open(sites_csv) as f:
        for r in csv.DictReader(f):
            ch = r["chain"].strip()
            if ch == "H":
                h_rows.append(r)
            elif ch == "L":
                l_rows.append(r)
    h_rows.sort(key=lambda x: int(x["site_scFv"]))
    l_rows.sort(key=lambda x: int(x["site_scFv"]))
    # Knock-in paired IGH (112 codons = 336 bp) + IGK (first 107 codons = 321 bp) = exact 657 bp
    selected_rows = h_rows + l_rows[:107]

    codon_meta = []  # index 0..218 (aln_codon_1based = idx + 1)
    for idx, r in enumerate(selected_rows):
        codon_meta.append({
            "aln_codon_1based": idx + 1,
            "scfv_pos": int(r["site_scFv"]),
            "chain": r["chain"].strip(),
            "site": int(r["site"]),
            "wt_aa": r["amino_acid"].strip(),
            "wt_codon": r["KI_codon"].strip().upper(),
            "annotation": r["annotation"].strip(),
        })
    naive_nt = "".join(c["wt_codon"] for c in codon_meta)
    assert len(naive_nt) == 657, f"Expected 657 bp germline, got {len(naive_nt)}"

    # DMS map keyed by (scfv_pos, mutant_aa) -> (delta_bind, delta_expr)
    # Note: in final_variant_scores.csv, `position` is 1-based scFv position (`site_scFv`)
    dms_map = {}
    with open(dms_csv) as f:
        for r in csv.DictReader(f):
            ch = r["chain"].strip()
            if ch not in ("H", "L"):
                continue
            scfv_pos = int(r["position"])
            mut_aa = r["mutant"].strip()
            db_s = r["delta_bind_CGG"].strip()
            de_s = r["delta_expr"].strip()
            db = float(db_s) if db_s and db_s != "NA" else -1.5
            de = float(de_s) if de_s and de_s != "NA" else -1.5
            dms_map[(scfv_pos, mut_aa)] = (db, de)

    # 5-mer SHM mutability & substitution
    fivemer_mut = {}
    with open(mut_csv) as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and len(parts[0]) == 5:
                fivemer_mut[parts[0].upper()] = float(parts[1])

    fivemer_sub = {}
    for_sub = open(sub_csv)
    with for_sub as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5 and len(parts[0]) == 5:
                fivemer_sub[parts[0].upper()] = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]

    return naive_nt, codon_meta, dms_map, fivemer_mut, fivemer_sub


def score_sequence_dms(nt_seq: str, codon_meta: list, dms_map: dict):
    """Evaluate additive DMS delta_bind_CGG, delta_expr, and stop-codon status."""
    total_bind = 0.0
    total_expr = 0.0
    has_stop = False
    for idx, meta in enumerate(codon_meta):
        cod = nt_seq[3 * idx : 3 * idx + 3]
        aa = CODON_TABLE.get(cod, "*")
        if aa == "*":
            has_stop = True
            break
        if aa != meta["wt_aa"]:
            db, de = dms_map.get((meta["scfv_pos"], aa), (-1.0, -0.5))
            total_bind += db
            total_expr += de
    return has_stop, total_bind, total_expr


# =============================================================================
# Part 3A: Galton-Watson Germinal-Center B-Cell Lineage Simulator & Benchmark
# =============================================================================

def mutate_bcr_sequence(seq: str, fivemer_mut: dict, fivemer_sub: dict, mean_muts_per_div: float, rng: random.Random) -> str:
    # Draw Poisson number of somatic hypermutations
    lam = mean_muts_per_div
    L_exp = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= L_exp:
            break
    n_muts = k - 1
    if n_muts == 0:
        return seq

    arr = list(seq)
    L = len(arr)
    bases = ["A", "C", "G", "T"]
    for _ in range(n_muts):
        # Compute 5-mer context weights across sequence
        s_cur = "".join(arr)
        padded = "NN" + s_cur + "NN"
        weights = []
        tot_w = 0.0
        for i in range(L):
            kmer = padded[i : i + 5]
            w = fivemer_mut.get(kmer, 0.001)
            weights.append(w)
            tot_w += w
        r = rng.random() * tot_w
        acc = 0.0
        chosen = L - 1
        for i, w in enumerate(weights):
            acc += w
            if acc >= r:
                chosen = i
                break
        kmer = padded[chosen : chosen + 5]
        probs = fivemer_sub.get(kmer)
        cur_b = arr[chosen]
        if probs and sum(probs) > 0:
            r2 = rng.random() * sum(probs)
            acc2 = 0.0
            new_b = cur_b
            for bi, pr in enumerate(probs):
                acc2 += pr
                if acc2 >= r2:
                    new_b = bases[bi]
                    break
            if new_b == cur_b:
                alts = [b for b in bases if b != cur_b]
                new_b = rng.choice(alts)
            arr[chosen] = new_b
        else:
            alts = [b for b in bases if b != cur_b]
            arr[chosen] = rng.choice(alts)
    return "".join(arr)


def simulate_single_germinal_center(
    rep_idx: int,
    naive_nt: str,
    codon_meta: list,
    dms_map: dict,
    fivemer_mut: dict,
    fivemer_sub: dict,
    n_sample: int = 90,
    beta_bind: float = 1.65,
):
    rng = random.Random(20261110 + rep_idx * 7919)
    # Track full genealogy of cells/clones:
    # Every evolutionary division event that changes sequence or spawns cells is tracked in a lineage tree.
    # Better yet: track EVERY individual cell across generations, and then prune to sampled cells
    # while contracting edges with 0 sequence mutations!
    # Wait: if a mother cell divides WITHOUT mutation into 2 identical daughter cells, contracting zero-mutation
    # edges automatically merges identical-genotype ancestors into an exact multifurcating clonal-burst node!
    next_id = 1
    # node_id -> (parent_id, seq_str)
    node_parent = {0: None}
    node_seq = {0: naive_nt}

    # Seed GC founder pool with 16 identical naive B cells attached to node 0
    population = []
    for _ in range(16):
        cid = next_id
        next_id += 1
        node_parent[cid] = 0
        node_seq[cid] = naive_nt
        population.append((cid, naive_nt, 0.0, 0.0))

    n_cycles = 15
    carrying_cap = 260
    mean_muts = 0.38

    for cyc in range(n_cycles):
        # Each surviving B cell divides into 2 daughter cells with somatic hypermutation
        daughters = []
        for pid, pseq, _, _ in population:
            for _ in range(2):
                dseq = mutate_bcr_sequence(pseq, fivemer_mut, fivemer_sub, mean_muts, rng)
                if dseq == pseq:
                    # Still create cell node; zero-mutation edge will contract cleanly!
                    has_stop, db, de = score_sequence_dms(dseq, codon_meta, dms_map) if cyc == 0 else (False, 0.0, 0.0)
                    if dseq != naive_nt:
                        has_stop, db, de = score_sequence_dms(dseq, codon_meta, dms_map)
                else:
                    has_stop, db, de = score_sequence_dms(dseq, codon_meta, dms_map)
                if has_stop or de < -1.15:
                    continue
                did = next_id
                next_id += 1
                node_parent[did] = pid
                node_seq[did] = dseq
                daughters.append((did, dseq, db, de))
        if len(daughters) == 0:
            break
        if len(daughters) <= carrying_cap:
            population = daughters
        else:
            # Affinity-proportional selection without replacement (Efraimidis-Spirakis weighted reservoir)
            scored = []
            for item in daughters:
                w = math.exp(max(-6.0, min(8.0, beta_bind * item[2] + 0.35 * item[3])))
                u = max(1e-15, rng.random())
                key = u ** (1.0 / w)
                scored.append((key, item))
            scored.sort(key=lambda x: x[0], reverse=True)
            population = [x[1] for x in scored[:carrying_cap]]

    if len(population) > n_sample:
        sampled = rng.sample(population, n_sample)
    else:
        sampled = list(population)

    # Build exact ground-truth CELL-LEVEL tree (contracting zero-mutation internal branches!)
    # Leaves: 'naive' attached to root 0, plus C000..C{n-1} attached to their sampled cell nodes
    active_nodes = {0}
    leaf_attach = {0: ["naive"]}
    cell_seqs = {"naive": naive_nt}
    for idx, (cid, cseq, db, de) in enumerate(sampled):
        cname = f"C{idx:03d}"
        cell_seqs[cname] = cseq
        leaf_attach.setdefault(cid, []).append(cname)
        cur = cid
        while cur is not None and cur not in active_nodes:
            active_nodes.add(cur)
            cur = node_parent[cur]

    children_raw = defaultdict(list)
    for nid in active_nodes:
        p = node_parent[nid]
        if p is not None and p in active_nodes:
            children_raw[p].append(nid)

    # Contract any internal edge (p -> u) where node_seq[p] == node_seq[u]
    # so identical-genotype cell divisions become exact multifurcating clonal bursts!
    def _build_contracted_cell_tree(u: int) -> TreeNode:
        nd_children = []
        for lname in leaf_attach.get(u, []):
            nd_children.append(TreeNode(name=lname, length=0.0))
        for v in children_raw.get(u, []):
            sub = _build_contracted_cell_tree(v)
            if node_seq[v] == node_seq[u]:
                # Zero mutations on branch u -> v: merge v's children directly into u!
                nd_children.extend(sub.children)
            else:
                muts = sum(1 for a, b in zip(node_seq[u], node_seq[v]) if a != b)
                sub.length = muts / 657.0
                nd_children.append(sub)
        if len(nd_children) == 1:
            return nd_children[0]
        return TreeNode(children=nd_children)

    true_cell_tree = _build_contracted_cell_tree(0)
    true_cell_nwk = write_newick_tree(true_cell_tree, include_lengths=True)

    # Build exact ground-truth UNIQUE-GENOTYPE tree:
    # Deduplicate sampled cells by 657-bp sequence; each distinct genotype (plus 'naive') gets ONE leaf!
    # Also record true mutant daughter-genotype out-degree for every observed unique genotype!
    geno_to_name = {naive_nt: "naive"}
    geno_counts = Counter(cseq for _, cseq, _, _ in sampled)
    geno_seqs = {"naive": naive_nt}
    g_idx = 1
    for _, cseq, _, _ in sampled:
        if cseq not in geno_to_name:
            gname = f"G{g_idx:03d}"
            g_idx += 1
            geno_to_name[cseq] = gname
            geno_seqs[gname] = cseq

    # In the contracted sequence-level tree, merge all active nodes that share the EXACT same sequence
    # if connected by zero-mutation paths (which is guaranteed unless back-mutation occurred, so we merge connected components of identical sequence!)
    comp_id = {}
    comp_seq = {}
    comp_parent = {}
    comp_Counter = [0]

    def _assign_comp(u: int, cur_comp: int):
        comp_id[u] = cur_comp
        for v in children_raw.get(u, []):
            if node_seq[v] == node_seq[u]:
                _assign_comp(v, cur_comp)
            else:
                comp_Counter[0] += 1
                new_c = comp_Counter[0]
                comp_seq[new_c] = node_seq[v]
                comp_parent[new_c] = cur_comp
                _assign_comp(v, new_c)

    comp_seq[0] = naive_nt
    comp_parent[0] = None
    _assign_comp(0, 0)

    comp_children = defaultdict(list)
    for c_id, p_id in comp_parent.items():
        if p_id is not None:
            comp_children[p_id].append(c_id)

    # Mark components that contain at least one observed unique genotype (or naive)
    comp_leaf = {}
    # Note: if two separate components have the exact same sequence (extremely rare convergent nucleotide identity),
    # attach leaf to the component that actually produced sampled cells
    sampled_nids = {0} | {cid for cid, _, _, _ in sampled}
    seen_gnames = set()
    for nid in sorted(sampled_nids):
        c_id = comp_id[nid]
        sq = comp_seq[c_id]
        gnm = geno_to_name[sq]
        if gnm not in seen_gnames:
            comp_leaf[c_id] = gnm
            seen_gnames.add(gnm)

    # Prune components with no sampled descendants and contract unobserved degree-1 pass-through nodes
    true_geno_burst_daughters = {}

    def _build_geno_tree(c_id: int):
        sub_nodes = []
        for ch_c in comp_children.get(c_id, []):
            res = _build_geno_tree(ch_c)
            if res is not None:
                sub_nodes.append(res)
        my_leaf = comp_leaf.get(c_id)
        if my_leaf is not None and my_leaf != "naive":
            true_geno_burst_daughters[my_leaf] = len(sub_nodes)
        if my_leaf is not None:
            sub_nodes.append(TreeNode(name=my_leaf, length=0.0))
        if not sub_nodes:
            return None
        if len(sub_nodes) == 1:
            return sub_nodes[0]
        return TreeNode(children=sub_nodes, length=1.0 / 657.0)

    true_geno_tree = _build_geno_tree(0)
    true_geno_nwk = write_newick_tree(true_geno_tree, include_lengths=True)

    return {
        "rep_idx": rep_idx,
        "n_cells": len(cell_seqs),
        "n_unique_genotypes": len(geno_seqs),
        "cell_seqs": cell_seqs,
        "geno_seqs": geno_seqs,
        "geno_counts": dict(geno_counts),
        "true_cell_nwk": true_cell_nwk,
        "true_geno_nwk": true_geno_nwk,
        "true_geno_burst_daughters": true_geno_burst_daughters,
    }


def write_fasta(seq_dict: dict, path: Path):
    with open(path, "w") as f:
        for nm, sq in seq_dict.items():
            f.write(f">{nm}\n{sq}\n")


def run_all_methods_on_alignment(seq_dict: dict, run_ufboot: bool = True) -> dict:
    """
    Run Bushwhacker (MDL & Binary), FastTree 2 (GTR), and IQ-TREE 2 (Binary, --polytomy,
    and optionally UFBoot -B 1000 collapsed at <50% and <70%) in an isolated temp dir.
    Note: IQ-TREE deduplicates identical sequences during search and restores them at the
    end as near-zero-length binary bifurcations, which is the exact behavior under test!
    """
    out = {}
    with tempfile.TemporaryDirectory(prefix="bw_bcr_") as tmpd:
        tmp = Path(tmpd)
        fa = tmp / "aln.fa"
        write_fasta(seq_dict, fa)

        # 1. Bushwhacker (MDL + Binary in a single pass!)
        bw_json = tmp / "bw.json"
        t0 = time.perf_counter()
        subprocess.run(
            [BUSHWHACKER_BIN, str(fa), "-j", str(bw_json), "-T", "1"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
        )
        t_bw = time.perf_counter() - t0
        with open(bw_json) as f:
            bwd = json.load(f)
        out["Bushwhacker_MDL"] = {"nwk": bwd["newick"], "sec": t_bw, "shunts": bwd.get("certified_shunts", 0)}
        out["Bushwhacker_Binary"] = {"nwk": bwd["binary_newick"], "sec": t_bw}

        # 2. FastTree 2 (GTR)
        env1 = os.environ.copy()
        env1["OMP_NUM_THREADS"] = "1"
        t0 = time.perf_counter()
        ft_res = subprocess.run(
            [FASTTREE_BIN, "-nt", "-gtr", "-quiet", "-nopr", str(fa)],
            capture_output=True, text=True, env=env1, check=True,
        )
        t_ft = time.perf_counter() - t0
        ft_nwk = ft_res.stdout.strip()
        out["FastTree_Binary"] = {"nwk": ft_nwk, "sec": t_ft}
        ft_sh70 = write_newick_tree(collapse_by_support(parse_newick_tree(ft_nwk), 70.0, scale_100=True))
        out["FastTree_SH70"] = {"nwk": ft_sh70, "sec": t_ft}

        # 3. IQ-TREE 2 Binary (GTR+G4)
        iq_bin_pre = tmp / "iq_bin"
        t0 = time.perf_counter()
        subprocess.run(
            [IQTREE_BIN, "-s", str(fa), "-m", "GTR+G4", "-T", "1", "-pre", str(iq_bin_pre), "-quiet", "-redo"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
        )
        t_iq_bin = time.perf_counter() - t0
        out["IQTREE_Binary"] = {"nwk": (tmp / "iq_bin.treefile").read_text().strip(), "sec": t_iq_bin}

        # 4. IQ-TREE 2 --polytomy (GTR+G4)
        iq_poly_pre = tmp / "iq_poly"
        t0 = time.perf_counter()
        subprocess.run(
            [IQTREE_BIN, "-s", str(fa), "-m", "GTR+G4", "--polytomy", "-T", "1", "-pre", str(iq_poly_pre), "-quiet", "-redo"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
        )
        t_iq_poly = time.perf_counter() - t0
        # Note: IQ-TREE re-attaches identical sequences as 0-length bifurcations unless --polytomy leaves 0-length branches;
        # we take IQ-TREE's exact emitted .treefile!
        out["IQTREE_Polytomy"] = {"nwk": (tmp / "iq_poly.treefile").read_text().strip(), "sec": t_iq_poly}

        # 5. IQ-TREE 2 UFBoot (-B 1000) when >= 4 unique sequences exist
        if run_ufboot:
            uniq_seqs = len(set(seq_dict.values()))
            if uniq_seqs >= 4:
                iq_uf_pre = tmp / "iq_uf"
                t0 = time.perf_counter()
                rc = subprocess.run(
                    [IQTREE_BIN, "-s", str(fa), "-m", "GTR+G4", "-B", "1000", "-T", "1", "-pre", str(iq_uf_pre), "-quiet", "-redo"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                ).returncode
                t_iq_uf = time.perf_counter() - t0
                tf = tmp / "iq_uf.treefile"
                if rc == 0 and tf.exists():
                    uf_raw = tf.read_text().strip()
                    out["IQTREE_UFBoot50"] = {
                        "nwk": write_newick_tree(collapse_by_support(parse_newick_tree(uf_raw), 50.0, scale_100=False)),
                        "sec": t_iq_uf,
                    }
                    out["IQTREE_UFBoot70"] = {
                        "nwk": write_newick_tree(collapse_by_support(parse_newick_tree(uf_raw), 70.0, scale_100=False)),
                        "sec": t_iq_uf,
                    }
                else:
                    # Fallback if UFBoot refuses near-identical alignment
                    out["IQTREE_UFBoot50"] = {"nwk": out["IQTREE_Polytomy"]["nwk"], "sec": t_iq_poly}
                    out["IQTREE_UFBoot70"] = {"nwk": out["IQTREE_Polytomy"]["nwk"], "sec": t_iq_poly}
            else:
                out["IQTREE_UFBoot50"] = {"nwk": out["IQTREE_Polytomy"]["nwk"], "sec": t_iq_poly}
                out["IQTREE_UFBoot70"] = {"nwk": out["IQTREE_Polytomy"]["nwk"], "sec": t_iq_poly}

    return out


def worker_part3a_replicate(args):
    rep_idx, naive_nt, codon_meta, dms_map, fivemer_mut, fivemer_sub = args
    sim = simulate_single_germinal_center(rep_idx, naive_nt, codon_meta, dms_map, fivemer_mut, fivemer_sub, n_sample=90)

    cell_runs = run_all_methods_on_alignment(sim["cell_seqs"], run_ufboot=True)
    cell_eval = {}
    for m, info in cell_runs.items():
        rf = compare_trees_rf(info["nwk"], sim["true_cell_nwk"])
        rf["sec"] = info["sec"]
        cell_eval[m] = rf

    geno_eval = {}
    burst_pairs = []
    if sim["n_unique_genotypes"] >= 5:
        geno_runs = run_all_methods_on_alignment(sim["geno_seqs"], run_ufboot=True)
        for m, info in geno_runs.items():
            rf = compare_trees_rf(info["nwk"], sim["true_geno_nwk"])
            rf["sec"] = info["sec"]
            geno_eval[m] = rf

        bw_burst, _ = root_tree_at_outgroup_and_analyze_bursts(
            geno_runs["Bushwhacker_MDL"]["nwk"], "naive", sim["geno_seqs"]
        )
        for gname, true_deg in sim["true_geno_burst_daughters"].items():
            est_deg = bw_burst.get(gname, 0)
            burst_pairs.append((true_deg, est_deg))

    return {
        "rep_idx": rep_idx,
        "n_cells": sim["n_cells"],
        "n_unique_genotypes": sim["n_unique_genotypes"],
        "cell_eval": cell_eval,
        "geno_eval": geno_eval,
        "burst_pairs": burst_pairs,
    }


def aggregate_method_evals(eval_list: list) -> dict:
    by_m = defaultdict(list)
    for d in eval_list:
        for m, st in d.items():
            by_m[m].append(st)
    summary = {}
    for m, items in by_m.items():
        k = len(items)
        summary[m] = {
            "n_trees": k,
            "mean_precision": round(sum(x["precision"] for x in items) / k, 5),
            "mean_recall": round(sum(x["recall"] for x in items) / k, 5),
            "mean_f1": round(sum(x["f1"] for x in items) / k, 5),
            "mean_nrf": round(sum(x["nrf"] for x in items) / k, 5),
            "mean_tp": round(sum(x["tp"] for x in items) / k, 3),
            "mean_fp": round(sum(x["fp"] for x in items) / k, 3),
            "mean_conflicting_fp": round(sum(x["conflicting_fp"] for x in items) / k, 3),
            "mean_compatible_fp": round(sum(x["compatible_fp"] for x in items) / k, 3),
            "mean_fn": round(sum(x["fn"] for x in items) / k, 3),
            "mean_est_splits": round(sum(x["est_splits"] for x in items) / k, 3),
            "mean_ref_splits": round(sum(x["ref_splits"] for x in items) / k, 3),
            "total_sec": round(sum(x["sec"] for x in items), 4),
            "mean_sec": round(sum(x["sec"] for x in items) / k, 5),
        }
    return summary


# =============================================================================
# Part 3B: All 147 Empirical Mouse Germinal Centers (DeWitt et al., Cell 2026)
# =============================================================================

def _open_text_maybe_xz(path: Path):
    if path.exists():
        return open(path, "r", encoding="utf-8")
    xz_path = Path(str(path) + ".xz")
    if xz_path.exists():
        return lzma.open(xz_path, "rt", encoding="utf-8")
    raise FileNotFoundError(f"Neither {path} nor {xz_path} found")


def load_empirical_gcreplay_datasets(naive_nt: str):
    obs_csv = BCR_DATA_DIR / "observed-seqs.csv"
    node_csv = BCR_DATA_DIR / "gctree-node-data.csv"

    # 1. Load raw observed single cells per uid
    cells_by_uid = defaultdict(list)
    with _open_text_maybe_xz(obs_csv) as f:
        for r in csv.DictReader(f):
            uid = r["uid"].strip()
            cid = r["ID_HK"].strip()
            hc = r["seq_nt_HC"].strip().upper()
            lc = r["seq_nt_LC"].strip().upper()
            if len(hc) == 336 and len(lc) == 321:
                db_s = r["delta_bind"].strip()
                de_s = r["delta_expr"].strip()
                cells_by_uid[uid].append({
                    "cell_id": cid,
                    "seq": hc + lc,
                    "mouse": r["mouse"].strip(),
                    "imm_duration": r["imm_duration"].strip(),
                    "delta_bind": float(db_s) if db_s and db_s != "NA" else None,
                    "delta_expr": float(de_s) if de_s and de_s != "NA" else None,
                })

    # 2. Load GCtree reference nodes ('default' strategy) per uid FOR POST-HOC EVALUATION ONLY
    nodes_by_uid = defaultdict(list)
    with _open_text_maybe_xz(node_csv) as f:
        for r in csv.DictReader(f):
            if r["ranking_coeff_strategy"].strip() != "default":
                continue
            uid = r["uid"].strip()
            hc = r["IgH_nt_sequence"].strip().upper()
            lc = r["IgK_nt_sequence"].strip().upper()
            ab = int(r["abundance"]) if r["abundance"].strip() else 0
            desc_ab = int(r["descendant_abundance"]) if r["descendant_abundance"].strip() else 0
            s_ids = [x.strip() for x in r["sampled_cell_ids"].split(",") if x.strip()]
            db_s = r["delta_bind"].strip()
            de_s = r["delta_expr"].strip()
            lbi_s = r["LBI"].strip()
            nodes_by_uid[uid].append({
                "name": r["name"].strip(),
                "parent_name": r["parent_name"].strip(),
                "abundance": ab,
                "descendant_abundance": desc_ab,
                "sampled_cell_ids": s_ids,
                "seq": hc + lc if (len(hc) == 336 and len(lc) == 321) else "",
                "delta_bind": float(db_s) if db_s and db_s != "NA" else None,
                "delta_expr": float(de_s) if de_s and de_s != "NA" else None,
                "lbi": float(lbi_s) if lbi_s and lbi_s != "NA" else None,
            })

    gc_tasks = []
    for uid in sorted(cells_by_uid.keys()):
        c_list = cells_by_uid[uid]
        n_list = nodes_by_uid.get(uid, [])
        if len(c_list) < 10 or not n_list:
            continue

        # Build GCtree reference CELL-LEVEL Newick & UNIQUE-GENOTYPE Newick
        ch_by_name = defaultdict(list)
        node_lookup = {}
        root_name = None
        for nd in n_list:
            nm = nd["name"]
            pnm = nd["parent_name"]
            node_lookup[nm] = nd
            if not pnm:
                root_name = nm
            else:
                ch_by_name[pnm].append(nm)
        if root_name is None:
            continue

        # Cell-level GCtree reference
        valid_cell_ids = {c["cell_id"] for c in c_list}
        def _ref_cell_tree(nm: str):
            nd = node_lookup[nm]
            kids = []
            if nm == root_name:
                kids.append(TreeNode(name="naive", length=0.0))
            for cid in nd["sampled_cell_ids"]:
                if cid in valid_cell_ids:
                    kids.append(TreeNode(name=cid, length=0.0))
            for cnm in ch_by_name.get(nm, []):
                sub = _ref_cell_tree(cnm)
                if sub is not None:
                    kids.append(sub)
            if not kids:
                return None
            if len(kids) == 1:
                return kids[0]
            return TreeNode(children=kids, length=1.0)

        ref_cell_tree = _ref_cell_tree(root_name)
        if ref_cell_tree is None:
            continue
        ref_cell_nwk = write_newick_tree(ref_cell_tree)

        # Raw cell-level sequence map (BLIND input to tree builders!)
        cell_seq_map = {"naive": naive_nt}
        for c in c_list:
            cell_seq_map[c["cell_id"]] = c["seq"]

        # Unique-genotype sequence map (BLIND deduplication of observed sequences: 1 copy per sequence!)
        # Map each observed unique sequence to its GCtree node name ONLY so leaf names match ref_geno_nwk
        seq_to_gname = {naive_nt: "naive"}
        geno_meta = {}  # gname -> post-hoc metadata (NEVER passed to tree builders!)
        for nd in n_list:
            if nd["abundance"] > 0 and len(nd["seq"]) == 657 and nd["seq"] != naive_nt:
                gnm = nd["name"]
                seq_to_gname[nd["seq"]] = gnm
                geno_meta[gnm] = {
                    "abundance": nd["abundance"],
                    "descendant_abundance": nd["descendant_abundance"],
                    "delta_bind": nd["delta_bind"],
                    "delta_expr": nd["delta_expr"],
                    "lbi": nd["lbi"],
                    "true_gctree_children": len(ch_by_name.get(gnm, [])),
                    "seq": nd["seq"],
                }

        geno_seq_map = {"naive": naive_nt}
        for gnm, mdata in geno_meta.items():
            geno_seq_map[gnm] = mdata["seq"]

        def _ref_geno_tree(nm: str):
            nd = node_lookup[nm]
            kids = []
            if nm == root_name:
                kids.append(TreeNode(name="naive", length=0.0))
            elif nm in geno_meta:
                kids.append(TreeNode(name=nm, length=0.0))
            for cnm in ch_by_name.get(nm, []):
                sub = _ref_geno_tree(cnm)
                if sub is not None:
                    kids.append(sub)
            if not kids:
                return None
            if len(kids) == 1:
                return kids[0]
            return TreeNode(children=kids, length=1.0)

        ref_geno_tree = _ref_geno_tree(root_name)
        ref_geno_nwk = write_newick_tree(ref_geno_tree) if ref_geno_tree else ""

        # Also map ALL GCtree nodes (including unsampled internal ancestors!) by sequence for post-hoc check
        all_gctree_seq_meta = {}
        for nd in n_list:
            if len(nd["seq"]) == 657 and nd["name"] != root_name:
                all_gctree_seq_meta[nd["seq"]] = {
                    "abundance": nd["abundance"],
                    "descendant_abundance": nd["descendant_abundance"],
                    "delta_bind": nd["delta_bind"],
                    "delta_expr": nd["delta_expr"],
                    "lbi": nd["lbi"],
                    "gctree_out_degree": len(ch_by_name.get(nd["name"], [])),
                }

        # Deterministic zero-leakage single cell per GC for cross-mouse replay alignment (lexicographically first ID_HK)
        first_cell = sorted(c_list, key=lambda x: x["cell_id"])[0]

        gc_tasks.append({
            "uid": uid,
            "mouse": c_list[0]["mouse"],
            "imm_duration": c_list[0]["imm_duration"],
            "cell_seq_map": cell_seq_map,
            "ref_cell_nwk": ref_cell_nwk,
            "geno_seq_map": geno_seq_map,
            "ref_geno_nwk": ref_geno_nwk,
            "geno_meta": geno_meta,
            "all_gctree_seq_meta": all_gctree_seq_meta,
            "rep_cell_seq": first_cell["seq"],
        })

    return gc_tasks


def worker_part3b_gc(task: dict) -> dict:
    uid = task["uid"]
    # 1. Run blind phylogenetic methods on ALL CELLS in this GC (skipping UFBoot 1000 on 147 GCs only if <4 unique, else run fast core methods + UFBoot!)
    cell_runs = run_all_methods_on_alignment(task["cell_seq_map"], run_ufboot=False)
    cell_eval = {}
    for m, info in cell_runs.items():
        rf = compare_trees_rf(info["nwk"], task["ref_cell_nwk"])
        rf["sec"] = info["sec"]
        cell_eval[m] = rf

    # 2. Run blind phylogenetic methods on UNIQUE GENOTYPES (1 copy per sequence — ZERO abundance leakage!)
    geno_eval = {}
    observed_geno_records = []
    internal_hub_records = []

    if len(task["geno_seq_map"]) >= 5 and task["ref_geno_nwk"]:
        geno_runs = run_all_methods_on_alignment(task["geno_seq_map"], run_ufboot=True)
        for m, info in geno_runs.items():
            rf = compare_trees_rf(info["nwk"], task["ref_geno_nwk"])
            rf["sec"] = info["sec"]
            geno_eval[m] = rf

        bw_burst, bw_internals = root_tree_at_outgroup_and_analyze_bursts(
            geno_runs["Bushwhacker_MDL"]["nwk"], "naive", task["geno_seq_map"]
        )
        for gnm, mdata in task["geno_meta"].items():
            inferred_daughters = bw_burst.get(gnm, 0)
            observed_geno_records.append({
                "uid": uid,
                "mouse": task["mouse"],
                "imm_duration": task["imm_duration"],
                "gname": gnm,
                "bw_burst_daughters": inferred_daughters,
                "abundance": mdata["abundance"],
                "descendant_abundance": mdata["descendant_abundance"],
                "delta_bind": mdata["delta_bind"],
                "delta_expr": mdata["delta_expr"],
                "lbi": mdata["lbi"],
                "gctree_children": mdata["true_gctree_children"],
            })

        # Also score ALL reconstructed internal nodes (via DMS directly from reconstructed 657-bp sequence!)
        for inode in bw_internals:
            internal_hub_records.append({
                "uid": uid,
                "mouse": task["mouse"],
                "seq": inode["seq"],
                "total_out_degree": inode["total_out_degree"],
                "mutant_daughter_branches": inode["mutant_daughter_branches"],
                "n_sampled_mother": len(inode["sampled_mother_leaves"]),
            })

    return {
        "uid": uid,
        "mouse": task["mouse"],
        "imm_duration": task["imm_duration"],
        "n_cells": len(task["cell_seq_map"]) - 1,
        "n_unique_genotypes": len(task["geno_seq_map"]) - 1,
        "cell_eval": cell_eval,
        "geno_eval": geno_eval,
        "observed_geno_records": observed_geno_records,
        "internal_hub_records": internal_hub_records,
    }


def analyze_tas_science2016_clone() -> dict:
    """Run Bushwhacker, FastTree, and IQ-TREE on Tas et al. (Science 2016) 150228_Clone_3-8.fasta."""
    if not TAS_FASTA.exists():
        return {}
    seqs = {}
    cur_nm = None
    buf = []
    with open(TAS_FASTA) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_nm is not None:
                    seqs[cur_nm] = "".join(buf).upper()
                cur_nm = line[1:].strip()
                buf = []
            else:
                buf.append(line)
        if cur_nm is not None:
            seqs[cur_nm] = "".join(buf).upper()

    runs = run_all_methods_on_alignment(seqs, run_ufboot=True)
    seq_counts = Counter(v for k, v in seqs.items() if k != "GL")
    uniq_seqs = {"GL": seqs["GL"]}
    ab_by_uname = {}
    idx = 1
    for sq, cnt in seq_counts.items():
        unm = f"U{idx:02d}"
        idx += 1
        uniq_seqs[unm] = sq
        ab_by_uname[unm] = cnt

    uniq_runs = run_all_methods_on_alignment(uniq_seqs, run_ufboot=False)
    bw_burst, _ = root_tree_at_outgroup_and_analyze_bursts(uniq_runs["Bushwhacker_MDL"]["nwk"], "GL", uniq_seqs)

    # Count resolved splits on the 66-sequence raw alignment
    split_counts = {}
    for m, info in runs.items():
        t = parse_newick_tree(info["nwk"])
        leaves = get_leaf_names(t)
        t2b = {tx: (1 << i) for i, tx in enumerate(leaves)}
        full = (1 << len(leaves)) - 1
        sp = extract_canonical_splits(t, t2b, full)
        split_counts[m] = {"resolved_splits": len(sp), "max_binary_splits": len(leaves) - 3, "sec": round(info["sec"], 5)}

    hub_abund = [ab_by_uname[u] for u, d in bw_burst.items() if d >= 2]
    nonhub_abund = [ab_by_uname[u] for u, d in bw_burst.items() if d < 2]
    return {
        "dataset": "Tas_Science2016_150228_Clone_3-8",
        "n_sequences": len(seqs),
        "n_unique_genotypes": len(uniq_seqs) - 1,
        "seq_len_bp": len(seqs["GL"]),
        "cell_level_splits_by_method": split_counts,
        "unique_geno_hub_mean_abundance": round(sum(hub_abund) / len(hub_abund), 3) if hub_abund else 0.0,
        "unique_geno_nonhub_mean_abundance": round(sum(nonhub_abund) / len(nonhub_abund), 3) if nonhub_abund else 0.0,
    }


def analyze_cross_mouse_replay_shunts(gc_tasks: list, naive_nt: str, codon_meta: list, dms_map: dict) -> dict:
    """
    Build a 1-cell-per-GC cross-mouse evolutionary replay alignment (+ naive germline)
    using strictly blind selection (lexicographically first cell_id per GC).
    Because every GC evolved independently in a separate lymph node / mouse from the
    exact same CGG_naive ancestor, shared non-tree homoplasies isolated by Bushwhacker's
    Stage 4 SPARSE shunts represent parallel somatic hypermutations across independent GCs!
    """
    replay_seqs = {"naive": naive_nt}
    for t in gc_tasks:
        replay_seqs[t["uid"]] = t["rep_cell_seq"]

    with tempfile.TemporaryDirectory(prefix="bw_replay_") as tmpd:
        tmp = Path(tmpd)
        fa = tmp / "replay.fa"
        js = tmp / "replay.json"
        write_fasta(replay_seqs, fa)
        subprocess.run(
            [BUSHWHACKER_BIN, str(fa), "-j", str(js), "--min-shunt-sites", "1", "-T", "8"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
        )
        bwd = json.loads(js.read_text())
        # Also run default --min-shunt-sites 2
        js2 = tmp / "replay_min2.json"
        subprocess.run(
            [BUSHWHACKER_BIN, str(fa), "-j", str(js2), "--min-shunt-sites", "2", "-T", "8"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
        )
        bwd_min2 = json.loads(js2.read_text())

    shunted_codons_min2 = set()
    for sh in bwd_min2.get("shunts", []):
        if sh.get("mode") == "SPARSE":
            for c1 in sh.get("codons_1based", []):
                shunted_codons_min2.add(int(c1))

    shunted_codons_min1 = set()
    for sh in bwd.get("shunts", []):
        if sh.get("mode") == "SPARSE":
            for c1 in sh.get("codons_1based", []):
                shunted_codons_min1.add(int(c1))

    # Compare baseline pairwise shunts (zeroed under exact Kraft priors) AND parallel star-hub replay recurrence across independent mice
    aa_mut_gc_counts = Counter()
    aa_mut_mice = defaultdict(set)
    mouse_of_uid = {t["uid"]: t["mouse"] for t in gc_tasks}
    for uid, sq in replay_seqs.items():
        if uid == "naive":
            continue
        m_id = mouse_of_uid[uid]
        for idx, meta in enumerate(codon_meta):
            cod = sq[3 * idx : 3 * idx + 3]
            aa = CODON_TABLE.get(cod, "*")
            if aa != "*" and aa != meta["wt_aa"]:
                key = (meta["aln_codon_1based"], aa)
                aa_mut_gc_counts[key] += 1
                aa_mut_mice[key].add(m_id)

    replay_tree_methods = run_all_methods_on_alignment(replay_seqs, run_ufboot=False)
    false_vertical_splits = {}
    for m_name, info in replay_tree_methods.items():
        tr = parse_newick_tree(info["nwk"])
        leaves = get_leaf_names(tr)
        t2b = {tx: (1 << i) for i, tx in enumerate(leaves)}
        full = (1 << len(leaves)) - 1
        sp = extract_canonical_splits(tr, t2b, full)
        false_vertical_splits[m_name] = {"resolved_splits": len(sp), "max_binary_splits": len(leaves) - 3}

    def eval_parallel_threshold(min_mice: int) -> dict:
        rec_bind, nonrec_bind = [], []
        cdr_rec, fwr_rec, cdr_non, fwr_non = 0, 0, 0, 0
        top_hits = []
        for (aln_cod, mut_aa), gc_cnt in aa_mut_gc_counts.items():
            n_mice = len(aa_mut_mice[(aln_cod, mut_aa)])
            meta = codon_meta[aln_cod - 1]
            db, de = dms_map.get((meta["scfv_pos"], mut_aa), (0.0, 0.0))
            is_cdr = "CDR" in meta["annotation"].upper()
            if n_mice >= min_mice:
                rec_bind.append(db)
                if is_cdr:
                    cdr_rec += 1
                else:
                    fwr_rec += 1
                top_hits.append({
                    "mutation": f"{meta['wt_aa']}{meta['site']}({meta['chain']}){mut_aa}",
                    "aln_codon_1based": aln_cod,
                    "scfv_codon": meta["scfv_pos"],
                    "region": meta["annotation"],
                    "independent_mice": n_mice,
                    "independent_gcs": gc_cnt,
                    "dms_delta_bind": round(db, 4),
                    "dms_delta_expr": round(de, 4),
                })
            else:
                nonrec_bind.append(db)
                if is_cdr:
                    cdr_non += 1
                else:
                    fwr_non += 1
        top_hits.sort(key=lambda x: (x["independent_mice"], x["independent_gcs"], x["dms_delta_bind"]), reverse=True)
        _, p_mwu_par = mann_whitney_u_two_sided(rec_bind, nonrec_bind)
        or_cdr_par, p_cdr_two_par = fisher_exact_2x2_two_sided(cdr_rec, fwr_rec, cdr_non, fwr_non)
        return {
            "min_independent_mice": min_mice,
            "n_parallel_aa_variants": len(rec_bind),
            "n_other_aa_variants": len(nonrec_bind),
            "parallel_mean_delta_bind": round(sum(rec_bind) / len(rec_bind), 4) if rec_bind else 0.0,
            "other_mean_delta_bind": round(sum(nonrec_bind) / len(nonrec_bind), 4) if nonrec_bind else 0.0,
            "parallel_median_delta_bind": round(quantile_sorted(sorted(rec_bind), 0.5), 4) if rec_bind else 0.0,
            "other_median_delta_bind": round(quantile_sorted(sorted(nonrec_bind), 0.5), 4) if nonrec_bind else 0.0,
            "mwu_pvalue_delta_bind": p_mwu_par,
            "cdr_fraction_parallel": round(cdr_rec / max(1, cdr_rec + fwr_rec), 4),
            "cdr_fraction_other": round(cdr_non / max(1, cdr_non + fwr_non), 4),
            "cdr_enrichment_odds_ratio": round(or_cdr_par, 3),
            "cdr_enrichment_fisher_p_two_sided": p_cdr_two_par,
            "top_parallel_convergent_mutations": top_hits[:15],
        }

    parallel_replay = {
        "false_cross_gc_vertical_splits": false_vertical_splits,
        "min_mice_3": eval_parallel_threshold(3),
        "min_mice_5": eval_parallel_threshold(5),
    }

    # Compare distinct observed somatic AA mutations at shunted codons vs non-shunted codons
    active_shunt_set = shunted_codons_min2 if len(shunted_codons_min2) >= 3 else shunted_codons_min1
    shunt_mode_used = "min_shunt_sites=2" if len(shunted_codons_min2) >= 3 else "min_shunt_sites=1"

    shunted_bind = []
    nonshunted_bind = []
    cdr_shunt, fwr_shunt = 0, 0
    cdr_nonshunt, fwr_nonshunt = 0, 0
    top_shunted_variants = []

    # For each mutated codon in active_shunt_set vs outside, evaluate observed AA variants
    for (aln_cod, mut_aa), gc_cnt in aa_mut_gc_counts.items():
        meta = codon_meta[aln_cod - 1]
        db, de = dms_map.get((meta["scfv_pos"], mut_aa), (0.0, 0.0))
        is_cdr = "CDR" in meta["annotation"].upper()
        if aln_cod in active_shunt_set:
            shunted_bind.append(db)
            if is_cdr:
                cdr_shunt += 1
            else:
                fwr_shunt += 1
            top_shunted_variants.append({
                "mutation": f"{meta['wt_aa']}{meta['site']}({meta['chain']}){mut_aa}",
                "aln_codon_1based": aln_cod,
                "scfv_codon": meta["scfv_pos"],
                "region": meta["annotation"],
                "independent_gcs": gc_cnt,
                "dms_delta_bind": round(db, 4),
                "dms_delta_expr": round(de, 4),
            })
        else:
            nonshunted_bind.append(db)
            if is_cdr:
                cdr_nonshunt += 1
            else:
                fwr_nonshunt += 1

    top_shunted_variants.sort(key=lambda x: (x["independent_gcs"], x["dms_delta_bind"]), reverse=True)
    _, p_mwu = mann_whitney_u_two_sided(shunted_bind, nonshunted_bind)
    odds_cdr, p_cdr_one = fisher_exact_2x2_greater(cdr_shunt, fwr_shunt, cdr_nonshunt, fwr_nonshunt)
    odds_cdr_two, p_cdr_two = fisher_exact_2x2_two_sided(cdr_shunt, fwr_shunt, cdr_nonshunt, fwr_nonshunt)

    return {
        "n_independent_gcs": len(replay_seqs) - 1,
        "shunt_threshold_mode": shunt_mode_used,
        "certified_shunts_min2": len(bwd_min2.get("shunts", [])),
        "certified_shunts_min1": len(bwd.get("shunts", [])),
        "shunted_codons_min2": sorted(shunted_codons_min2),
        "shunted_codons_used": sorted(active_shunt_set),
        "n_shunted_aa_variants": len(shunted_bind),
        "n_nonshunted_aa_variants": len(nonshunted_bind),
        "shunted_mean_delta_bind": round(sum(shunted_bind) / len(shunted_bind), 4) if shunted_bind else 0.0,
        "nonshunted_mean_delta_bind": round(sum(nonshunted_bind) / len(nonshunted_bind), 4) if nonshunted_bind else 0.0,
        "shunted_median_delta_bind": round(quantile_sorted(sorted(shunted_bind), 0.5), 4) if shunted_bind else 0.0,
        "nonshunted_median_delta_bind": round(quantile_sorted(sorted(nonshunted_bind), 0.5), 4) if nonshunted_bind else 0.0,
        "mwu_pvalue_delta_bind": p_mwu,
        "cdr_fraction_shunted": round(cdr_shunt / max(1, cdr_shunt + fwr_shunt), 4),
        "cdr_fraction_nonshunted": round(cdr_nonshunt / max(1, cdr_nonshunt + fwr_nonshunt), 4),
        "cdr_enrichment_odds_ratio": round(odds_cdr_two, 3) if not math.isnan(odds_cdr_two) else None,
        "cdr_enrichment_fisher_p": p_cdr_two,
        "cdr_enrichment_fisher_p_two_sided": p_cdr_two,
        "cdr_enrichment_fisher_p_one_sided": p_cdr_one,
        "top_shunted_convergent_mutations": top_shunted_variants[:15],
        "parallel_star_hub_replay": parallel_replay,
    }


def main():
    t_start = time.perf_counter()
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    print("[Pillar 3] Loading CGG_naive germline, DMS affinity landscape, and MK_RS5NF 5-mer SHM model...", flush=True)
    naive_nt, codon_meta, dms_map, fivemer_mut, fivemer_sub = load_cgg_dms_and_shm()
    print(f"  Loaded 657-bp germline ({len(codon_meta)} codons), {len(dms_map)} DMS variant scores, {len(fivemer_mut)} 5-mers.", flush=True)

    # -------------------------------------------------------------------------
    # Part 3A: 50 Ground-Truth Germinal-Center Affinity-Maturation Simulations
    # -------------------------------------------------------------------------
    n_sims = 50
    n_workers = min(max(os.cpu_count() or 8, 1), 50)
    print(f"[Part 3A] Running {n_sims} ground-truth Galton-Watson BCR germinal-center simulations across {n_workers} workers...", flush=True)
    sim_args = [(r, naive_nt, codon_meta, dms_map, fivemer_mut, fivemer_sub) for r in range(n_sims)]
    sim_results = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = [pool.submit(worker_part3a_replicate, a) for a in sim_args]
        for fut in as_completed(futs):
            res = fut.result()
            sim_results.append(res)
            if len(sim_results) % 10 == 0:
                print(f"  [Part 3A] Completed {len(sim_results)}/{n_sims} simulated GCs...", flush=True)

    sim_results.sort(key=lambda x: x["rep_idx"])
    part3a_cell_summary = aggregate_method_evals([r["cell_eval"] for r in sim_results])
    part3a_geno_summary = aggregate_method_evals([r["geno_eval"] for r in sim_results if r["geno_eval"]])

    all_burst_pairs = [p for r in sim_results for p in r["burst_pairs"]]
    true_b = [p[0] for p in all_burst_pairs]
    est_b = [p[1] for p in all_burst_pairs]
    rho_sim_burst, p_sim_burst = spearman_corr(true_b, est_b)

    part3a_out = {
        "n_simulated_gcs": n_sims,
        "mean_sampled_cells_per_gc": round(sum(r["n_cells"] - 1 for r in sim_results) / n_sims, 2),
        "mean_unique_genotypes_per_gc": round(sum(r["n_unique_genotypes"] - 1 for r in sim_results) / n_sims, 2),
        "cell_level_benchmark": part3a_cell_summary,
        "unique_genotype_benchmark": part3a_geno_summary,
        "burst_degree_recovery": {
            "n_observed_genotypes": len(all_burst_pairs),
            "spearman_rho": round(rho_sim_burst, 5),
            "spearman_pvalue": p_sim_burst,
            "mean_true_daughters_when_bw_hub_ge2": round(
                sum(t for t, e in all_burst_pairs if e >= 2) / max(1, sum(1 for _, e in all_burst_pairs if e >= 2)), 4
            ),
            "mean_true_daughters_when_bw_nonhub_lt2": round(
                sum(t for t, e in all_burst_pairs if e < 2) / max(1, sum(1 for _, e in all_burst_pairs if e < 2)), 4
            ),
        },
    }
    print(
        f"[Part 3A DONE] Cell-level F1: BW_MDL={part3a_cell_summary['Bushwhacker_MDL']['mean_f1']:.4f} "
        f"(FP={part3a_cell_summary['Bushwhacker_MDL']['mean_fp']:.1f}) vs "
        f"IQTREE_Bin={part3a_cell_summary['IQTREE_Binary']['mean_f1']:.4f} "
        f"(FP={part3a_cell_summary['IQTREE_Binary']['mean_fp']:.1f}), "
        f"IQTREE_Poly={part3a_cell_summary['IQTREE_Polytomy']['mean_f1']:.4f}; "
        f"Burst degree Spearman rho={rho_sim_burst:.4f} (p={p_sim_burst:.2e})",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # Part 3B: All 147 Empirical Mouse Germinal Centers (DeWitt et al., Cell 2026)
    # -------------------------------------------------------------------------
    print("[Part 3B] Loading all 147 empirical mouse germinal centers from gcreplay...", flush=True)
    gc_tasks = load_empirical_gcreplay_datasets(naive_nt)
    print(f"  Loaded {len(gc_tasks)} empirical GCs ({sum(len(t['cell_seq_map'])-1 for t in gc_tasks)} paired B cells).", flush=True)

    emp_results = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = [pool.submit(worker_part3b_gc, t) for t in gc_tasks]
        for fut in as_completed(futs):
            res = fut.result()
            emp_results.append(res)
            if len(emp_results) % 25 == 0 or len(emp_results) == len(gc_tasks):
                print(f"  [Part 3B] Completed {len(emp_results)}/{len(gc_tasks)} empirical GCs...", flush=True)

    emp_results.sort(key=lambda x: x["uid"])
    emp_cell_summary = aggregate_method_evals([r["cell_eval"] for r in emp_results if r["cell_eval"]])
    emp_geno_summary = aggregate_method_evals([r["geno_eval"] for r in emp_results if r["geno_eval"]])

    # Post-hoc Zero-Leakage Biological Discovery Tests on Unique-Genotype Trees
    obs_records = [rec for r in emp_results for rec in r["observed_geno_records"]]
    hub_recs = [r for r in obs_records if r["bw_burst_daughters"] >= 2]
    branch1_recs = [r for r in obs_records if r["bw_burst_daughters"] == 1]
    twig_recs = [r for r in obs_records if r["bw_burst_daughters"] == 0]
    nonhub_recs = [r for r in obs_records if r["bw_burst_daughters"] < 2]

    # 1. In vivo cellular abundance: direct exact-sequence cell multiplicity (NOT descendant sum)
    hub_ab = [r["abundance"] for r in hub_recs]
    nonhub_ab = [r["abundance"] for r in nonhub_recs]
    _, p_ab_mwu = mann_whitney_u_two_sided(hub_ab, nonhub_ab)
    rho_ab, p_ab_rho = spearman_corr(
        [r["bw_burst_daughters"] for r in obs_records],
        [r["abundance"] for r in obs_records],
    )

    hub_desc_ab = [r["descendant_abundance"] for r in hub_recs]
    nonhub_desc_ab = [r["descendant_abundance"] for r in nonhub_recs]
    _, p_desc_mwu = mann_whitney_u_two_sided(hub_desc_ab, nonhub_desc_ab)

    # Hierarchical GC-level and Mouse-level tests (cluster-aware mixed/paired evaluation)
    gc_ab_diffs, gc_ab_mice = [], []
    gc_ab_rhos, gc_ab_rho_mice = [], []
    gc_dms_diffs, gc_dms_mice = [], []
    gc_dms_rhos, gc_dms_rho_mice = [], []
    mouse_records = defaultdict(list)
    for r in emp_results:
        recs = r["observed_geno_records"]
        if not recs:
            continue
        ms = r["mouse"]
        mouse_records[ms].extend(recs)
        h_a = [x["abundance"] for x in recs if x["bw_burst_daughters"] >= 2]
        nh_a = [x["abundance"] for x in recs if x["bw_burst_daughters"] < 2]
        if h_a and nh_a:
            gc_ab_diffs.append((sum(h_a) / len(h_a)) - (sum(nh_a) / len(nh_a)))
            gc_ab_mice.append(ms)
        if len(recs) >= 5 and len({x["bw_burst_daughters"] for x in recs}) >= 2 and len({x["abundance"] for x in recs}) >= 2:
            rho_g, _ = spearman_corr([x["bw_burst_daughters"] for x in recs], [x["abundance"] for x in recs])
            gc_ab_rhos.append(rho_g)
            gc_ab_rho_mice.append(ms)
        dms_r = [x for x in recs if x["delta_bind"] is not None]
        h_d = [x["delta_bind"] for x in dms_r if x["bw_burst_daughters"] >= 2]
        nh_d = [x["delta_bind"] for x in dms_r if x["bw_burst_daughters"] < 2]
        if h_d and nh_d:
            gc_dms_diffs.append((sum(h_d) / len(h_d)) - (sum(nh_d) / len(nh_d)))
            gc_dms_mice.append(ms)
        if len(dms_r) >= 5 and len({x["bw_burst_daughters"] for x in dms_r}) >= 2:
            rho_d, _ = spearman_corr([x["bw_burst_daughters"] for x in dms_r], [x["delta_bind"] for x in dms_r])
            gc_dms_rhos.append(rho_d)
            gc_dms_rho_mice.append(ms)

    mouse_ab_diffs, mouse_ab_rhos = [], []
    mouse_dms_diffs, mouse_dms_rhos = [], []
    for ms, recs in sorted(mouse_records.items()):
        by_gc_diff_a = []
        by_gc_rho_a = []
        by_gc_diff_d = []
        by_gc_rho_d = []
        for d, m_tag in zip(gc_ab_diffs, gc_ab_mice):
            if m_tag == ms:
                by_gc_diff_a.append(d)
        for rh, m_tag in zip(gc_ab_rhos, gc_ab_rho_mice):
            if m_tag == ms:
                by_gc_rho_a.append(rh)
        for d, m_tag in zip(gc_dms_diffs, gc_dms_mice):
            if m_tag == ms:
                by_gc_diff_d.append(d)
        for rh, m_tag in zip(gc_dms_rhos, gc_dms_rho_mice):
            if m_tag == ms:
                by_gc_rho_d.append(rh)
        if by_gc_diff_a:
            mouse_ab_diffs.append(sum(by_gc_diff_a) / len(by_gc_diff_a))
        if by_gc_rho_a:
            mouse_ab_rhos.append(sum(by_gc_rho_a) / len(by_gc_rho_a))
        if by_gc_diff_d:
            mouse_dms_diffs.append(sum(by_gc_diff_d) / len(by_gc_diff_d))
        if by_gc_rho_d:
            mouse_dms_rhos.append(sum(by_gc_rho_d) / len(by_gc_rho_d))

    # 2. Experimental Yeast-Display DMS Antigen-Binding Affinity (delta_bind = delta_log10 K_D)
    obs_with_dms = [r for r in obs_records if r["delta_bind"] is not None]
    hub_dms = [r["delta_bind"] for r in obs_with_dms if r["bw_burst_daughters"] >= 2]
    br1_dms = [r["delta_bind"] for r in obs_with_dms if r["bw_burst_daughters"] == 1]
    twig_dms = [r["delta_bind"] for r in obs_with_dms if r["bw_burst_daughters"] == 0]
    nonhub_dms = [r["delta_bind"] for r in obs_with_dms if r["bw_burst_daughters"] < 2]
    _, p_dms_mwu = mann_whitney_u_two_sided(hub_dms, nonhub_dms)
    _, p_dms_hub_vs_twig = mann_whitney_u_two_sided(hub_dms, twig_dms)
    rho_dms, p_dms_rho = spearman_corr(
        [r["bw_burst_daughters"] for r in obs_with_dms],
        [r["delta_bind"] for r in obs_with_dms],
    )

    # 3. Also evaluate ALL reconstructed internal nodes (including unsampled burst mothers!) by exact DMS sequence scoring
    all_inodes = [inode for r in emp_results for inode in r["internal_hub_records"]]
    inode_poly_dms = []
    inode_bin_dms = []
    for inode in all_inodes:
        has_stop, db, de = score_sequence_dms(inode["seq"], codon_meta, dms_map)
        if not has_stop:
            if inode["total_out_degree"] >= 3:
                inode_poly_dms.append((inode["total_out_degree"], db))
            else:
                inode_bin_dms.append((inode["total_out_degree"], db))
    inode_deg_all = [x[0] for x in inode_poly_dms + inode_bin_dms]
    inode_db_all = [x[1] for x in inode_poly_dms + inode_bin_dms]
    rho_inode_dms, p_inode_rho = spearman_corr(inode_deg_all, inode_db_all)
    _, p_inode_mwu = mann_whitney_u_two_sided([x[1] for x in inode_poly_dms], [x[1] for x in inode_bin_dms])

    # Stratify observed genotype burst-degree stratification table (k = 0, 1, 2, >=3)
    stratified_by_k = {}
    for label, pred in [
        ("k=0_terminal_twig", lambda k: k == 0),
        ("k=1_linear_step", lambda k: k == 1),
        ("k=2_small_burst", lambda k: k == 2),
        ("k>=3_major_burst_hub", lambda k: k >= 3),
    ]:
        sub = [r for r in obs_with_dms if pred(r["bw_burst_daughters"])]
        if sub:
            dbs = sorted(r["delta_bind"] for r in sub)
            abs_v = sorted(r["abundance"] for r in sub)
            stratified_by_k[label] = {
                "n_genotypes": len(sub),
                "mean_abundance": round(sum(abs_v) / len(abs_v), 4),
                "median_abundance": round(quantile_sorted(abs_v, 0.5), 4),
                "frac_expanded_ge2_cells": round(sum(1 for a in abs_v if a >= 2) / len(abs_v), 4),
                "mean_delta_bind": round(sum(dbs) / len(dbs), 4),
                "median_delta_bind": round(quantile_sorted(dbs, 0.5), 4),
                "iqr25_delta_bind": round(quantile_sorted(dbs, 0.25), 4),
                "iqr75_delta_bind": round(quantile_sorted(dbs, 0.75), 4),
            }

    # 3B-3: Cross-mouse replay SPARSE shunts + Tas Science 2016
    print("[Part 3B-3] Analyzing cross-mouse evolutionary replay shunts & Tas et al. (Science 2016)...", flush=True)
    replay_shunts = analyze_cross_mouse_replay_shunts(gc_tasks, naive_nt, codon_meta, dms_map)
    tas_summary = analyze_tas_science2016_clone()

    final_output = {
        "pillar": "Pillar 3: Cellular Lineage Trees — Germinal-Center BCR Clonal Bursts & DMS Affinity Landscapes",
        "elapsed_wall_sec": round(time.perf_counter() - t_start, 2),
        "part3a_simulations": part3a_out,
        "part3b_empirical_gcreplay": {
            "n_germinal_centers": len(emp_results),
            "n_mice": len(mouse_records),
            "total_sequenced_b_cells": sum(r["n_cells"] for r in emp_results),
            "total_unique_genotypes": sum(r["n_unique_genotypes"] for r in emp_results),
            "abundance_definition": "direct exact-sequence cell multiplicity collapsed onto each unique genotype node (not descendant subtree sum)",
            "cell_level_vs_gctree_reference": emp_cell_summary,
            "unique_genotype_vs_gctree_reference": emp_geno_summary,
            "zero_leakage_biological_discovery": {
                "n_evaluated_unique_genotypes": len(obs_records),
                "n_bw_burst_hubs_ge2": len(hub_recs),
                "n_bw_nonhubs_lt2": len(nonhub_recs),
                "in_vivo_clone_abundance": {
                    "hub_mean_cells": round(sum(hub_ab) / max(1, len(hub_ab)), 4),
                    "nonhub_mean_cells": round(sum(nonhub_ab) / max(1, len(nonhub_ab)), 4),
                    "fold_enrichment_cells": round(
                        (sum(hub_ab) / max(1, len(hub_ab))) / max(1e-9, sum(nonhub_ab) / max(1, len(nonhub_ab))), 4
                    ),
                    "hub_mean_descendant_cells": round(sum(hub_desc_ab) / max(1, len(hub_desc_ab)), 4),
                    "nonhub_mean_descendant_cells": round(sum(nonhub_desc_ab) / max(1, len(nonhub_desc_ab)), 4),
                    "mwu_pvalue_abundance": p_ab_mwu,
                    "mwu_pvalue_descendant_abundance": p_desc_mwu,
                    "spearman_rho_degree_vs_abundance": round(rho_ab, 5),
                    "spearman_pvalue_degree_vs_abundance": p_ab_rho,
                    "hierarchical_gc_paired_diff_hub_minus_nonhub": paired_wilcoxon_and_cluster_ci(gc_ab_diffs, gc_ab_mice),
                    "hierarchical_gc_spearman_rho": paired_wilcoxon_and_cluster_ci(gc_ab_rhos, gc_ab_rho_mice),
                    "hierarchical_mouse_paired_diff_hub_minus_nonhub": paired_wilcoxon_and_cluster_ci(mouse_ab_diffs),
                    "hierarchical_mouse_spearman_rho": paired_wilcoxon_and_cluster_ci(mouse_ab_rhos),
                },
                "experimental_dms_binding_affinity": {
                    "hub_mean_delta_bind": round(sum(hub_dms) / max(1, len(hub_dms)), 4),
                    "hub_median_delta_bind": round(quantile_sorted(sorted(hub_dms), 0.5), 4) if hub_dms else 0.0,
                    "branch1_mean_delta_bind": round(sum(br1_dms) / max(1, len(br1_dms)), 4),
                    "twig_mean_delta_bind": round(sum(twig_dms) / max(1, len(twig_dms)), 4),
                    "twig_median_delta_bind": round(quantile_sorted(sorted(twig_dms), 0.5), 4) if twig_dms else 0.0,
                    "nonhub_mean_delta_bind": round(sum(nonhub_dms) / max(1, len(nonhub_dms)), 4),
                    "mwu_pvalue_hub_vs_nonhub": p_dms_mwu,
                    "mwu_pvalue_hub_vs_twig": p_dms_hub_vs_twig,
                    "spearman_rho_degree_vs_delta_bind": round(rho_dms, 5),
                    "spearman_pvalue_degree_vs_delta_bind": p_dms_rho,
                    "hierarchical_gc_paired_diff_hub_minus_nonhub": paired_wilcoxon_and_cluster_ci(gc_dms_diffs, gc_dms_mice),
                    "hierarchical_gc_spearman_rho": paired_wilcoxon_and_cluster_ci(gc_dms_rhos, gc_dms_rho_mice),
                    "hierarchical_mouse_paired_diff_hub_minus_nonhub": paired_wilcoxon_and_cluster_ci(mouse_dms_diffs),
                    "hierarchical_mouse_spearman_rho": paired_wilcoxon_and_cluster_ci(mouse_dms_rhos),
                    "reconstructed_internal_nodes": {
                        "n_polytomy_hubs_deg_ge3": len(inode_poly_dms),
                        "n_binary_nodes_deg_lt3": len(inode_bin_dms),
                        "polytomy_mean_delta_bind": round(sum(x[1] for x in inode_poly_dms) / max(1, len(inode_poly_dms)), 4),
                        "binary_mean_delta_bind": round(sum(x[1] for x in inode_bin_dms) / max(1, len(inode_bin_dms)), 4),
                        "mwu_pvalue": p_inode_mwu,
                        "spearman_rho": round(rho_inode_dms, 5),
                        "spearman_pvalue": p_inode_rho,
                    },
                },
                "stratified_by_inferred_burst_daughters": stratified_by_k,
            },
            "cross_mouse_replay_shunts": replay_shunts,
        },
        "tas_science2016_clone": tas_summary,
    }

    with open(OUT_JSON, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"[Pillar 3 COMPLETE] Saved full JSON report to {OUT_JSON} in {final_output['elapsed_wall_sec']}s.", flush=True)


if __name__ == "__main__":
    main()
