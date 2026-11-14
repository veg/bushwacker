#!/usr/bin/env python3
"""
Pillar 2: Biodiversity & Domain Mega-Phylogenies (N = 256 -> 32,768+ Taxa & GTDB Supermatrices)
===============================================================================================
Executes two complementary mega-phylogeny tree-building benchmarks on silverback:
  2A. Greengenes 16S rRNA Microbiome Progression (N = 256, 512, 1024, 2048, 4096, 8192,
      16384, 32768 OTUs; L = 1,348 bp):
      - Quantifies the physical information-resolution saturation curve (resolved splits vs.
        N - 3 and splits per alignment column) as taxon sampling scales 128x on a fixed marker.
      - Evaluates strict clade monophyly, Four-Gamete taxonomic compatibility, and rogue-taxon
        clade-shattering across 5 curated Greengenes ranks (Phylum, Class, Order, Family, Genus)
        for Bushwhacker (MDL), Bushwhacker (Binary), and 32-core FastTreeMP (GTR).
      - Benchmarks lossless .bwz microbiome archiving (bytes/OTU vs. gzip -9 and xz -6).
  2B. Domain-Wide Concatenated GTDB Prokaryotic Supermatrices:
      - Archaeal domain supermatrix (ar122: 1,672 species across 19 phyla, ~32,000+ AA)
      - Bacterial domain supermatrix (bac120: 1,000 species across all 111 phyla, ~38,000+ AA)
      Demonstrates that when alignment length L scales to proteome supermatrices, 32-core
      20-state Bushwhacker resolves deep prokaryotic phyla/classes/orders cleanly in seconds
      to minutes.
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
import time
import shutil
import subprocess
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

BUSHWHACKER_BIN = resolve_bin("BUSHWHACKER_BIN", REPO_ROOT / "target" / "release" / "bushwhacker", "bushwhacker")
FASTTREE_BIN = resolve_bin("FASTTREE_BIN", "FastTreeMP", "FastTree", "fasttree")
MAFFT_BIN = resolve_bin("MAFFT_BIN", "mafft")
if not os.path.exists(MAFFT_BIN):
    MAFFT_BIN = resolve_bin("MAFFT_BIN", "mafft")

GG_DIR = ALIGNMENTS_DIR / "microbial_and_16s"
GG_POW2_DIR = GG_DIR / "progression_pow2"
GG_RES_DIR = GG_DIR / "results_pow2"
GG_TAX_FILE = GG_DIR / "gg_13_8_otus" / "taxonomy" / "97_otu_taxonomy.txt"

GTDB_DIR = EXTERNAL_DIR / "gtdb_r95"
WORK_DIR = WORK_ROOT / "pillar2"
OUT_JSON = RESULTS_DIR / "microbial_viral" / "pillar2_megatrees.json"


# =============================================================================
# Fast Iterative Bitset Tree & Taxonomy Evaluators (Handles N = 65,536 in ms)
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


def get_leaf_names_and_hub_degrees(root: TreeNode) -> tuple:
    leaves = []
    hub_degrees = []
    stack = [(root, True)]
    while stack:
        nd, is_rt = stack.pop()
        if not nd.children:
            if nd.name:
                leaves.append(nd.name)
        else:
            deg = len(nd.children) + (0 if is_rt else 1)
            if deg > 3:
                hub_degrees.append(deg)
            for c in reversed(nd.children):
                stack.append((c, False))
    return leaves, hub_degrees


def extract_canonical_splits(root: TreeNode, taxon_to_bit: dict, full_mask: int) -> set:
    n_shared = bin(full_mask).count("1")
    if n_shared < 4:
        return set()
    splits = set()
    stack = [(root, False)]
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


def evaluate_5rank_taxonomy(root: TreeNode, tax_map: dict) -> dict:
    leaves, _ = get_leaf_names_and_hub_degrees(root)
    n_all = len(leaves)
    t2b_all = {tx: (1 << i) for i, tx in enumerate(leaves)}
    full_all = (1 << n_all) - 1
    splits_all = list(extract_canonical_splits(root, t2b_all, full_all))
    splits_set = set(splits_all)

    out_ranks = {}
    for rank in ("phylum", "class", "order", "family", "genus"):
        # Restrict mask universe to leaves annotated at this rank so unannotated leaves don't penalize
        ann_mask = 0
        grp_masks = defaultdict(int)
        grp_counts = defaultdict(int)
        n_ann = 0
        for tx in leaves:
            rk_val = tax_map.get(tx, {}).get(rank)
            if rk_val:
                b = t2b_all[tx]
                ann_mask |= b
                grp_masks[rk_val] |= b
                grp_counts[rk_val] += 1
                n_ann += 1
        if n_ann < 4:
            continue

        # Project tree splits onto annotated leaf universe
        proj_splits = set()
        for s in splits_all:
            ps = s & ann_mask
            bc = bin(ps).count("1")
            if 2 <= bc <= n_ann - 2:
                comp = ann_mask ^ ps
                proj_splits.add(ps if ps < comp else comp)
        proj_list = list(proj_splits)

        valid_grps = {g: m for g, m in grp_masks.items() if 2 <= grp_counts[g] <= n_ann - 2}
        strict = 0
        compat = 0
        total_crossing_splits = 0

        for g, gm in valid_grps.items():
            gcomp = ann_mask ^ gm
            canon = gm if gm < gcomp else gcomp
            if canon in proj_splits:
                strict += 1
                compat += 1
            else:
                cross_cnt = 0
                for s in proj_list:
                    if (s & gm) and (s & gcomp) and ((ann_mask ^ s) & gm) and ((ann_mask ^ s) & gcomp):
                        cross_cnt += 1
                if cross_cnt == 0:
                    compat += 1
                total_crossing_splits += cross_cnt

        ng = len(valid_grps)
        out_ranks[rank] = {
            "n_annotated_leaves": n_ann,
            "n_groups": ng,
            "strict_monophyletic": strict,
            "strict_pct": round(100.0 * strict / max(1, ng), 2),
            "compatible_uncontradicted": compat,
            "compatible_pct": round(100.0 * compat / max(1, ng), 2),
            "shattered_groups": ng - compat,
            "mean_contradicting_splits_per_group": round(total_crossing_splits / max(1, ng), 3),
        }
    return out_ranks


def compare_unrooted_trees(root_pred: TreeNode, root_ref: TreeNode) -> dict:
    lp, _ = get_leaf_names_and_hub_degrees(root_pred)
    lr, _ = get_leaf_names_and_hub_degrees(root_ref)
    shared = sorted(set(lp) & set(lr))
    n = len(shared)
    t2b = {tx: (1 << i) for i, tx in enumerate(shared)}
    full = (1 << n) - 1
    sp_p = extract_canonical_splits(root_pred, t2b, full)
    sp_r = extract_canonical_splits(root_ref, t2b, full)
    tp = len(sp_p & sp_r)
    fp = len(sp_p - sp_r)
    fn = len(sp_r - sp_p)
    rf = fp + fn
    max_sp = n - 3
    return {
        "n_shared": n,
        "pred_splits": len(sp_p),
        "ref_splits": len(sp_r),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "rf": rf,
        "nrf": round(rf / (2.0 * max_sp), 5),
        "precision_pct": round(100.0 * tp / max(1, len(sp_p)), 2),
        "recall_pct": round(100.0 * tp / max(1, len(sp_r)), 2),
    }


def load_greengenes_taxonomy(path: Path) -> dict:
    tax = {}
    prefix_map = {"p__": "phylum", "c__": "class", "o__": "order", "f__": "family", "g__": "genus"}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            oid = parts[0].strip()
            ranks = {}
            for item in parts[1].split(";"):
                it = item.strip()
                p = it[:3]
                if p in prefix_map and len(it) > 3:
                    val = it[3:].strip("[]")
                    if val:
                        ranks[prefix_map[p]] = val
            tax[oid] = ranks
    return tax


def load_gtdb_taxonomy(path: Path) -> dict:
    tax = {}
    prefix_map = {"p__": "phylum", "c__": "class", "o__": "order", "f__": "family", "g__": "genus"}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            gid = parts[0].strip()
            ranks = {}
            for item in parts[1].split(";"):
                it = item.strip()
                p = it[:3]
                if p in prefix_map and len(it) > 3:
                    ranks[prefix_map[p]] = it[3:]
            tax[gid] = ranks
    return tax


def parse_fasta_dict(fa_path: Path) -> dict:
    seqs = {}
    curr_id = None
    buf = []
    with open(fa_path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith(">"):
                if curr_id is not None:
                    seqs[curr_id] = "".join(buf)
                curr_id = s[1:].split()[0]
                buf = []
            else:
                buf.append(s)
        if curr_id is not None:
            seqs[curr_id] = "".join(buf)
    return seqs


def run_compression_bench(fa_path: Path, bwz_path: Path, tmp_dir: Path, tag: str) -> dict:
    raw_bytes = fa_path.stat().st_size
    gz_path = tmp_dir / f"{tag}.fa.gz"
    xz_path = tmp_dir / f"{tag}.fa.xz"

    if not bwz_path.exists() or bwz_path.stat().st_size == 0:
        subprocess.run(
            [BUSHWHACKER_BIN, str(fa_path), "-c", str(bwz_path), "-T", "32"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
    subprocess.run(["gzip", "-9", "-c", str(fa_path)], stdout=open(gz_path, "wb"), check=True)
    subprocess.run(["xz", "-6", "-T", "16", "-c", str(fa_path)], stdout=open(xz_path, "wb"), check=True)

    bwz_b = bwz_path.stat().st_size
    gz_b = gz_path.stat().st_size
    xz_b = xz_path.stat().st_size
    for p in (bwz_path, gz_path, xz_path):
        try:
            p.unlink()
        except OSError:
            pass
    return {
        "raw_bytes": raw_bytes,
        "bwz_bytes": bwz_b,
        "gzip9_bytes": gz_b,
        "xz6_bytes": xz_b,
        "bwz_vs_gzip9_ratio": round(gz_b / max(1, bwz_b), 3),
        "bwz_vs_xz6_ratio": round(xz_b / max(1, bwz_b), 3),
    }


# =============================================================================
# 2B. Domain-Wide Concatenated GTDB Supermatrices (ar122 & bac120)
# =============================================================================

def _align_bac_marker_worker(args):
    faa_path_str, allowed_set, tmp_dir_str = args
    faa_path = Path(faa_path_str)
    marker = faa_path.stem
    seqs = parse_fasta_dict(faa_path)
    sub = {k: v.upper().rstrip("*") for k, v in seqs.items() if k in allowed_set and len(v) >= 20}
    if len(sub) < 20:
        return marker, {}, 0
    in_faa = Path(tmp_dir_str) / f"{marker}.in.faa"
    out_faa = Path(tmp_dir_str) / f"{marker}.aln.faa"
    with open(in_faa, "w") as f:
        for k in sorted(sub.keys()):
            f.write(f">{k}\n{sub[k]}\n")
    subprocess.run(
        ["mafft", "--retree", "1", "--thread", "2", "--quiet", str(in_faa)],
        env={**os.environ, "PATH": os.environ.get("PATH", "")},
        stdout=open(out_faa, "w"),
        stderr=subprocess.DEVNULL,
        check=True,
    )
    aln = parse_fasta_dict(out_faa)
    n_present = len(aln)
    first_seq = next(iter(aln.values()))
    L_raw = len(first_seq)
    keep_cols = [
        c for c in range(L_raw)
        if sum(1 for s in aln.values() if s[c] == "-") / n_present <= 0.50
    ]
    trimmed = {k: "".join(s[c] for c in keep_cols) for k, s in aln.items()}
    in_faa.unlink(missing_ok=True)
    out_faa.unlink(missing_ok=True)
    return marker, trimmed, len(keep_cols)


def run_part_2b_gtdb_supermatrices() -> dict:
    print("[Pillar 2B] Building & Running Domain-Wide Concatenated GTDB Supermatrices...", flush=True)
    sm_dir = WORK_DIR / "gtdb_supermatrix"
    sm_dir.mkdir(parents=True, exist_ok=True)
    results_2b = {}

    # 1. Archaeal ar122 supermatrix (all 122 aligned markers in ar122_msa/*.faa)
    ar_tax = load_gtdb_taxonomy(GTDB_DIR / "ar122_taxonomy_r95.tsv")
    ar_ref_root = parse_newick_tree(open(GTDB_DIR / "ar122_r95.tree").read())
    ar_taxa, _ = get_leaf_names_and_hub_degrees(ar_ref_root)
    ar_taxa = sorted(set(ar_taxa))

    ar_faa_files = sorted(glob.glob(str(GTDB_DIR / "ar122_msa" / "*.faa")))
    ar_concat = {tx: [] for tx in ar_taxa}
    ar_total_cols = 0
    for fp in ar_faa_files:
        d = parse_fasta_dict(Path(fp))
        if not d:
            continue
        L_m = len(next(iter(d.values())))
        gap_str = "-" * L_m
        ar_total_cols += L_m
        for tx in ar_taxa:
            ar_concat[tx].append(d.get(tx, gap_str))

    ar_sm_path = sm_dir / "ar122_supermatrix.faa"
    with open(ar_sm_path, "w") as f:
        for tx in ar_taxa:
            f.write(f">{tx}\n{''.join(ar_concat[tx])}\n")

    print(f"[Pillar 2B] ar122 supermatrix: N={len(ar_taxa)}, L={ar_total_cols} AA -> running BW -T 32 & FastTreeMP...", flush=True)
    ar_json_path = sm_dir / "ar122_bw.json"
    t0 = time.perf_counter()
    subprocess.run(
        [BUSHWHACKER_BIN, str(ar_sm_path), "--protein", "-j", str(ar_json_path), "-T", "32"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
    )
    bw_ar_wall = time.perf_counter() - t0
    bw_ar_data = json.load(open(ar_json_path))

    env32 = {**os.environ, "OMP_NUM_THREADS": "32"}
    ft_ar_path = sm_dir / "ar122_ft.nwk"
    t0 = time.perf_counter()
    subprocess.run([FASTTREE_BIN, "-lg", "-quiet", str(ar_sm_path)], stdout=open(ft_ar_path, "w"), stderr=subprocess.DEVNULL, env=env32, check=True)
    ft_ar_wall = time.perf_counter() - t0

    root_ar_mdl = parse_newick_tree(bw_ar_data["newick"])
    root_ar_bin = parse_newick_tree(bw_ar_data["binary_newick"])
    root_ar_ft = parse_newick_tree(open(ft_ar_path).read())

    results_2b["ar122_archaeal_domain_supermatrix"] = {
        "n_taxa": len(ar_taxa),
        "n_markers": len(ar_faa_files),
        "seq_len_aa": ar_total_cols,
        "bushwhacker_t32_sec": round(bw_ar_data["runtime_sec"], 2),
        "bushwhacker_wall_sec": round(bw_ar_wall, 2),
        "fasttree_lg_t32_sec": round(ft_ar_wall, 2),
        "speedup_vs_fasttree": round(ft_ar_wall / max(0.01, bw_ar_data["runtime_sec"]), 2),
        "resolved_splits": bw_ar_data["resolved_internal_splits"],
        "max_binary_splits": bw_ar_data["max_binary_splits"],
        "resolved_pct": round(100.0 * bw_ar_data["resolved_internal_splits"] / max(1, bw_ar_data["max_binary_splits"]), 2),
        "vs_gtdb_ref_tree": {
            "Bushwhacker_MDL": compare_unrooted_trees(root_ar_mdl, ar_ref_root),
            "Bushwhacker_Binary": compare_unrooted_trees(root_ar_bin, ar_ref_root),
            "FastTree_LG": compare_unrooted_trees(root_ar_ft, ar_ref_root),
        },
        "taxonomy_5ranks": {
            "Bushwhacker_MDL": evaluate_5rank_taxonomy(root_ar_mdl, ar_tax),
            "FastTree_LG": evaluate_5rank_taxonomy(root_ar_ft, ar_tax),
        },
    }
    print(
        f"[Pillar 2B] ar122 DONE: BW={bw_ar_data['runtime_sec']:.1f}s vs FT={ft_ar_wall:.1f}s, "
        f"resolved={results_2b['ar122_archaeal_domain_supermatrix']['resolved_pct']}%, "
        f"nRF: BW_MDL={results_2b['ar122_archaeal_domain_supermatrix']['vs_gtdb_ref_tree']['Bushwhacker_MDL']['nrf']} "
        f"vs FT={results_2b['ar122_archaeal_domain_supermatrix']['vs_gtdb_ref_tree']['FastTree_LG']['nrf']}",
        flush=True,
    )

    # 2. Bacterial bac120 supermatrix (1,000 species across all 111 Bacterial phyla)
    bac_panel_file = GTDB_DIR / "bac120_panel_1000.txt"
    bac_taxa = sorted(line.strip() for line in open(bac_panel_file) if line.strip())
    bac_set = set(bac_taxa)
    bac_tax = load_gtdb_taxonomy(GTDB_DIR / "bac120_taxonomy_r95.tsv")
    bac_ref_root = parse_newick_tree(open(GTDB_DIR / "bac120_r95.tree").read())

    bac_raw_faas = sorted(glob.glob(str(GTDB_DIR / "bac120_marker_genes_reps_r95" / "faa" / "*.faa")))
    print(f"[Pillar 2B] Aligning {len(bac_raw_faas)} bac120 markers across {len(bac_taxa)} species (111 phyla)...", flush=True)
    with ProcessPoolExecutor(max_workers=28) as ex:
        aligned_markers = list(ex.map(_align_bac_marker_worker, [(p, bac_set, str(sm_dir)) for p in bac_raw_faas]))

    bac_concat = {tx: [] for tx in bac_taxa}
    bac_total_cols = 0
    used_markers = 0
    for marker, trimmed, L_m in aligned_markers:
        if L_m <= 0:
            continue
        used_markers += 1
        bac_total_cols += L_m
        gap_str = "-" * L_m
        for tx in bac_taxa:
            bac_concat[tx].append(trimmed.get(tx, gap_str))

    bac_sm_path = sm_dir / "bac120_supermatrix_1000.faa"
    with open(bac_sm_path, "w") as f:
        for tx in bac_taxa:
            f.write(f">{tx}\n{''.join(bac_concat[tx])}\n")

    print(f"[Pillar 2B] bac120 supermatrix: N={len(bac_taxa)}, L={bac_total_cols} AA -> running BW -T 32 & FastTreeMP...", flush=True)
    bac_json_path = sm_dir / "bac120_bw.json"
    t0 = time.perf_counter()
    subprocess.run(
        [BUSHWHACKER_BIN, str(bac_sm_path), "--protein", "-j", str(bac_json_path), "-T", "32"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
    )
    bw_bac_wall = time.perf_counter() - t0
    bw_bac_data = json.load(open(bac_json_path))

    ft_bac_path = sm_dir / "bac120_ft.nwk"
    t0 = time.perf_counter()
    subprocess.run([FASTTREE_BIN, "-lg", "-quiet", str(bac_sm_path)], stdout=open(ft_bac_path, "w"), stderr=subprocess.DEVNULL, env=env32, check=True)
    ft_bac_wall = time.perf_counter() - t0

    root_bac_mdl = parse_newick_tree(bw_bac_data["newick"])
    root_bac_bin = parse_newick_tree(bw_bac_data["binary_newick"])
    root_bac_ft = parse_newick_tree(open(ft_bac_path).read())

    results_2b["bac120_bacterial_domain_supermatrix"] = {
        "n_taxa": len(bac_taxa),
        "n_phyla": 111,
        "n_markers": used_markers,
        "seq_len_aa": bac_total_cols,
        "bushwhacker_t32_sec": round(bw_bac_data["runtime_sec"], 2),
        "bushwhacker_wall_sec": round(bw_bac_wall, 2),
        "fasttree_lg_t32_sec": round(ft_bac_wall, 2),
        "speedup_vs_fasttree": round(ft_bac_wall / max(0.01, bw_bac_data["runtime_sec"]), 2),
        "resolved_splits": bw_bac_data["resolved_internal_splits"],
        "max_binary_splits": bw_bac_data["max_binary_splits"],
        "resolved_pct": round(100.0 * bw_bac_data["resolved_internal_splits"] / max(1, bw_bac_data["max_binary_splits"]), 2),
        "vs_gtdb_ref_tree": {
            "Bushwhacker_MDL": compare_unrooted_trees(root_bac_mdl, bac_ref_root),
            "Bushwhacker_Binary": compare_unrooted_trees(root_bac_bin, bac_ref_root),
            "FastTree_LG": compare_unrooted_trees(root_bac_ft, bac_ref_root),
        },
        "taxonomy_5ranks": {
            "Bushwhacker_MDL": evaluate_5rank_taxonomy(root_bac_mdl, bac_tax),
            "FastTree_LG": evaluate_5rank_taxonomy(root_bac_ft, bac_tax),
        },
    }
    print(
        f"[Pillar 2B] bac120 DONE: BW={bw_bac_data['runtime_sec']:.1f}s vs FT={ft_bac_wall:.1f}s, "
        f"resolved={results_2b['bac120_bacterial_domain_supermatrix']['resolved_pct']}%, "
        f"nRF: BW_MDL={results_2b['bac120_bacterial_domain_supermatrix']['vs_gtdb_ref_tree']['Bushwhacker_MDL']['nrf']} "
        f"vs FT={results_2b['bac120_bacterial_domain_supermatrix']['vs_gtdb_ref_tree']['FastTree_LG']['nrf']}",
        flush=True,
    )
    return results_2b


# =============================================================================
# 2A. Greengenes 16S Microbiome Mega-Tree Progression (N = 256 -> 32,768)
# =============================================================================

def run_part_2a_16s_progression(part_2b_results: dict, t_global_start: float):
    print("[Pillar 2A] Loading Greengenes 97_otu_taxonomy.txt...", flush=True)
    gg_tax = load_greengenes_taxonomy(GG_TAX_FILE)

    ft_prev_times = {}
    pow2_sum_path = GG_RES_DIR / "scaling_pow2_summary.json"
    if pow2_sum_path.exists():
        for rec in json.load(open(pow2_sum_path)):
            ft_prev_times[int(rec["N"])] = rec["fasttree_time_sec"]

    comp_dir = WORK_DIR / "16s_comp"
    comp_dir.mkdir(parents=True, exist_ok=True)

    env32 = {**os.environ, "OMP_NUM_THREADS": "32"}
    progression_results = []

    for N in [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
        fa_path = GG_POW2_DIR / f"16S_N{N}.fa"
        if not fa_path.exists():
            continue
        print(f"[Pillar 2A] Running 16S_N{N}.fa on 32 threads...", flush=True)

        bw_json_path = comp_dir / f"bw_N{N}.json"
        bwz_path = comp_dir / f"16S_N{N}.bwz"
        t0 = time.perf_counter()
        subprocess.run(
            [BUSHWHACKER_BIN, str(fa_path), "-j", str(bw_json_path), "-c", str(bwz_path), "-T", "32"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        bw_wall = time.perf_counter() - t0
        bw_data = json.load(open(bw_json_path))

        # Fresh 32-core FastTreeMP -nt -gtr benchmark on the exact same node
        ft_nwk_path = comp_dir / f"ft32_N{N}.nwk"
        t0 = time.perf_counter()
        subprocess.run(
            [FASTTREE_BIN, "-nt", "-gtr", "-quiet", str(fa_path)],
            stdout=open(ft_nwk_path, "w"), stderr=subprocess.DEVNULL, env=env32, check=True
        )
        ft32_sec = time.perf_counter() - t0
        ft_nwk_str = open(ft_nwk_path).read().strip()

        root_mdl = parse_newick_tree(bw_data["newick"])
        root_bin = parse_newick_tree(bw_data["binary_newick"])
        root_ft = parse_newick_tree(ft_nwk_str)

        _, hubs_mdl = get_leaf_names_and_hub_degrees(root_mdl)
        comp_stats = run_compression_bench(fa_path, bwz_path, comp_dir, f"16S_N{N}")
        comp_stats["bwz_bytes_per_otu"] = round(comp_stats["bwz_bytes"] / N, 2)

        tax_mdl = evaluate_5rank_taxonomy(root_mdl, gg_tax)
        tax_bin = evaluate_5rank_taxonomy(root_bin, gg_tax)
        tax_ft = evaluate_5rank_taxonomy(root_ft, gg_tax)

        max_sp = bw_data["max_binary_splits"]
        res_sp = bw_data["resolved_internal_splits"]
        L = bw_data["seq_len"]

        step_rec = {
            "N": N,
            "seq_len_bp": L,
            "taxa_per_site_ratio": round(N / L, 3),
            "bushwhacker_t32_sec": round(bw_data["runtime_sec"], 3),
            "bushwhacker_wall_sec": round(bw_wall, 3),
            "fasttree_mp_t32_sec": round(ft32_sec, 3),
            "fasttree_1core_baseline_sec": round(ft_prev_times.get(N, 0.0), 2),
            "speedup_vs_fasttree_mp32": round(ft32_sec / max(1e-3, bw_data["runtime_sec"]), 2),
            "resolved_splits": res_sp,
            "max_binary_splits": max_sp,
            "resolved_pct": round(100.0 * res_sp / max(1, max_sp), 2),
            "hallucinated_zero_gain_pct": round(100.0 * (max_sp - res_sp) / max(1, max_sp), 2),
            "resolved_splits_per_alignment_column": round(res_sp / L, 3),
            "polytomy_hub_count": len(hubs_mdl),
            "max_hub_degree": max(hubs_mdl) if hubs_mdl else 3,
            "certified_shunts": bw_data["certified_shunts"],
            "star_MDL_nats": round(bw_data["star_tree_MDL_nats"], 1),
            "tree_MDL_nats": round(bw_data["tree_MDL_nats"], 1),
            "compression": comp_stats,
            "taxonomy_5ranks": {
                "Bushwhacker_MDL": tax_mdl,
                "Bushwhacker_Binary": tax_bin,
                "FastTreeMP_GTR": tax_ft,
            },
        }
        progression_results.append(step_rec)
        print(
            f"[Pillar 2A] N={N:5d} | BW={bw_data['runtime_sec']:6.2f}s vs FT32={ft32_sec:6.2f}s "
            f"({step_rec['speedup_vs_fasttree_mp32']:.1f}x) | resolved={res_sp}/{max_sp} ({step_rec['resolved_pct']:.1f}%) | "
            f"Genus compat: MDL={tax_mdl['genus']['compatible_pct']:.1f}% vs FT={tax_ft['genus']['compatible_pct']:.1f}% | "
            f".bwz={comp_stats['bwz_bytes_per_otu']:.1f} B/OTU ({comp_stats['bwz_vs_gzip9_ratio']:.2f}x < gzip)",
            flush=True,
        )

        # Save incremental checkpoint after every N
        payload = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_wall_sec": round(time.time() - t_global_start, 2),
            "part_2a_16s_megatree_progression": progression_results,
            "part_2b_gtdb_domain_supermatrices": part_2b_results,
        }
        with open(OUT_JSON, "w") as f:
            json.dump(payload, f, indent=2)


def main():
    t_start = time.time()
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    part_2b = run_part_2b_gtdb_supermatrices()
    run_part_2a_16s_progression(part_2b, t_start)
    print(f"[Pillar 2] COMPLETE -> {OUT_JSON} in {time.time() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
