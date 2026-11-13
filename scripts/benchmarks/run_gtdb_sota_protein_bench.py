#!/usr/bin/env python3
"""
Benchmark #2: Domain-Wide GTDB 20-State Protein Supermatrices vs. SOTA Phylogenetics
====================================================================================
Evaluates upgraded LG08-MDL 20-state protein Bushwhacker head-to-head against:
  - FastTreeMP (-lg, 64 threads)
  - IQ-TREE 2 (-m LG+G4 --fast, 64 threads)
  - IQ-TREE 2 (-m LG+G4 full NNI/hill-climbing ML search, 64 threads)
  - Legacy pre-upgrade Bushwhacker baseline (cached JSONs)

Datasets (cached concatenated domain-wide marker supermatrices):
  1. Archaeal Domain (GTDB ar122): 1,672 species across all Archaeal phyla x 32,675 AA
  2. Bacterial Domain (GTDB bac120): 1,000 species across all 111 Bacterial phyla x 43,733 AA

Metrics:
  - Normalized Robinson-Foulds (nRF), Precision (%), and Recall (%) vs. curated GTDB r95 reference tree
  - Pairwise nRF agreement between Bushwhacker, FastTreeMP, and IQ-TREE 2
  - 5-rank GTDB taxonomic monophyly & Four-Gamete compatibility (Phylum, Class, Order, Family, Genus)
  - Wall-clock runtime (64 threads on identical silverback compute node)
"""


import sys as _sys
from pathlib import Path as _Path
if str(_Path(__file__).resolve().parent) not in _sys.path:
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
from _env_paths import REPO_ROOT, DATA_DIR, SPECIMENS_DIR, ALIGNMENTS_DIR, SINGLE_CELL_DIR, RESULTS_DIR, WORK_ROOT, EXTERNAL_DIR, resolve_bin, ASTRAL_JAR

import os
import re
import json
import time
import subprocess
from pathlib import Path
from collections import defaultdict

BUSHWHACKER_BIN = resolve_bin("BUSHWHACKER_BIN", REPO_ROOT / "target" / "release" / "bushwhacker", "bushwhacker")
FASTTREE_BIN = resolve_bin("FASTTREE_BIN", "FastTreeMP", "FastTree", "fasttree")
IQTREE_BIN = resolve_bin("IQTREE_BIN", "iqtree2", "iqtree")

GTDB_DIR = EXTERNAL_DIR / "gtdb_r95"
SM_DIR = ALIGNMENTS_DIR / "supermatrices"
BENCH_DIR = WORK_ROOT / "pillar2" / "gtdb_sota_bench"
OUT_JSON = WORK_ROOT / "pillar2" / "gtdb_sota_protein_bench.json"

THREADS = int(os.environ.get("BENCH_THREADS", "64"))


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
        "false_positive_rate_pct": round(100.0 * fp / max(1, len(sp_p)), 2),
        "precision_pct": round(100.0 * tp / max(1, len(sp_p)), 2),
        "recall_pct": round(100.0 * tp / max(1, len(sp_r)), 2),
    }


def evaluate_5rank_taxonomy(root: TreeNode, tax_map: dict) -> dict:
    leaves, _ = get_leaf_names_and_hub_degrees(root)
    n_all = len(leaves)
    t2b_all = {tx: (1 << i) for i, tx in enumerate(leaves)}
    full_all = (1 << n_all) - 1
    splits_all = list(extract_canonical_splits(root, t2b_all, full_all))

    out_ranks = {}
    for rank in ("phylum", "class", "order", "family", "genus"):
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
            "n_groups": ng,
            "strict_monophyletic": strict,
            "strict_pct": round(100.0 * strict / max(1, ng), 2),
            "compatible_uncontradicted": compat,
            "compatible_pct": round(100.0 * compat / max(1, ng), 2),
            "mean_contradicting_splits": round(total_crossing_splits / max(1, ng), 3),
        }
    return out_ranks


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


def save_checkpoint(results: dict, t_start: float):
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "threads": THREADS,
        "elapsed_wall_sec": round(time.time() - t_start, 2),
        "datasets": results,
    }
    tmp = OUT_JSON.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(OUT_JSON)


def evaluate_method(name: str, root: TreeNode, ref_root: TreeNode, tax_map: dict, wall_sec: float, extra: dict = None) -> dict:
    vs_ref = compare_unrooted_trees(root, ref_root)
    tax_5 = evaluate_5rank_taxonomy(root, tax_map)
    rec = {
        "method": name,
        "wall_sec": round(wall_sec, 2),
        "vs_gtdb_ref": vs_ref,
        "taxonomy_5ranks": tax_5,
    }
    if extra:
        rec.update(extra)
    print(
        f"  [{name:26s}] wall={wall_sec:7.2f}s | nRF={vs_ref['nrf']:.4f} "
        f"(Prec={vs_ref['precision_pct']:5.1f}%, Rec={vs_ref['recall_pct']:5.1f}%) | "
        f"Phylum strict={tax_5['phylum']['strict_pct']:5.1f}% (compat={tax_5['phylum']['compatible_pct']:5.1f}%), "
        f"Order strict={tax_5['order']['strict_pct']:5.1f}%, Genus strict={tax_5['genus']['strict_pct']:5.1f}%",
        flush=True,
    )
    return rec


def main():
    t_start = time.time()
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    env_omp = {**os.environ, "OMP_NUM_THREADS": str(THREADS)}

    datasets_cfg = [
        {
            "key": "ar122_archaea_1672",
            "title": "GTDB r95 Archaeal Domain (1,672 species x 32,675 AA; 122 markers)",
            "faa": SM_DIR / "ar122_supermatrix.faa",
            "ref_tree": GTDB_DIR / "ar122_r95.tree",
            "tax_tsv": GTDB_DIR / "ar122_taxonomy_r95.tsv",
            "old_bw_json": SM_DIR / "ar122_bw.json",
            "cached_ft_nwk": SM_DIR / "ar122_ft.nwk",
        },
        {
            "key": "bac120_bacteria_1000",
            "title": "GTDB r95 Bacterial Domain (1,000 species across all 111 phyla x 43,733 AA; 120 markers)",
            "faa": SM_DIR / "bac120_supermatrix_1000.faa",
            "ref_tree": GTDB_DIR / "bac120_r95.tree",
            "tax_tsv": GTDB_DIR / "bac120_taxonomy_r95.tsv",
            "old_bw_json": SM_DIR / "bac120_bw.json",
            "cached_ft_nwk": SM_DIR / "bac120_ft.nwk",
        },
    ]

    results = {}
    loaded_refs = {}
    loaded_tax = {}
    parsed_trees = defaultdict(dict)

    # =========================================================================
    # PHASE 1: Upgraded Bushwhacker (LG08-MDL) + Cached Baselines (< 2 minutes)
    # =========================================================================
    print(f"=== PHASE 1: Running Upgraded 20-State Bushwhacker (-T {THREADS}) on GTDB Supermatrices ===", flush=True)
    for cfg in datasets_cfg:
        key = cfg["key"]
        print(f"\n>>> Dataset: {cfg['title']}", flush=True)
        ref_root = parse_newick_tree(cfg["ref_tree"].read_text())
        tax_map = load_gtdb_taxonomy(cfg["tax_tsv"])
        loaded_refs[key] = ref_root
        loaded_tax[key] = tax_map
        results[key] = {"title": cfg["title"], "methods": {}, "pairwise_nrf": {}}

        # Reference tree taxonomy upper-bound check
        parsed_trees[key]["GTDB_Ref"] = ref_root
        results[key]["gtdb_ref_taxonomy_5ranks"] = evaluate_5rank_taxonomy(ref_root, tax_map)

        # 0. Pre-upgrade legacy Bushwhacker baseline (if cached)
        if cfg["old_bw_json"].exists():
            old_bw = json.loads(cfg["old_bw_json"].read_text())
            r_old_bin = parse_newick_tree(old_bw["binary_newick"])
            results[key]["methods"]["Bushwhacker_Legacy_PreFix"] = evaluate_method(
                "Bushwhacker_Legacy_PreFix",
                r_old_bin,
                ref_root,
                tax_map,
                old_bw.get("runtime_sec", 0.0),
                {"note": "Pre-fix unweighted Fitch + N>512 pivot clustering"},
            )

        # 1. Upgraded Bushwhacker (--protein --no-shunts, pure species-tree supermatrix mode)
        bw_json = BENCH_DIR / f"{key}_bw_lg08.json"
        t0 = time.perf_counter()
        subprocess.run(
            [
                BUSHWHACKER_BIN,
                str(cfg["faa"]),
                "--protein",
                "--no-shunts",
                "-j",
                str(bw_json),
                "-T",
                str(THREADS),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        bw_wall = time.perf_counter() - t0
        bw_data = json.loads(bw_json.read_text())

        r_bw_mdl = parse_newick_tree(bw_data["newick"])
        r_bw_bin = parse_newick_tree(bw_data["binary_newick"])
        parsed_trees[key]["Bushwhacker_LG08_MDL"] = r_bw_mdl
        parsed_trees[key]["Bushwhacker_LG08_Binary"] = r_bw_bin

        results[key]["methods"]["Bushwhacker_LG08_MDL"] = evaluate_method(
            "Bushwhacker_LG08_MDL",
            r_bw_mdl,
            ref_root,
            tax_map,
            bw_wall,
            {
                "engine_sec": round(bw_data["runtime_sec"], 2),
                "resolved_splits": bw_data["resolved_internal_splits"],
                "max_binary_splits": bw_data["max_binary_splits"],
                "resolved_pct": round(
                    100.0 * bw_data["resolved_internal_splits"] / max(1, bw_data["max_binary_splits"]), 2
                ),
                "star_MDL_nats": round(bw_data["star_tree_MDL_nats"], 1),
                "tree_MDL_nats": round(bw_data["tree_MDL_nats"], 1),
            },
        )
        results[key]["methods"]["Bushwhacker_LG08_Binary"] = evaluate_method(
            "Bushwhacker_LG08_Binary",
            r_bw_bin,
            ref_root,
            tax_map,
            bw_wall,
            {"engine_sec": round(bw_data["runtime_sec"], 2)},
        )

        # Quick check against cached FastTree_LG right away before re-timing on 64 threads
        if cfg["cached_ft_nwk"].exists():
            r_ft_cached = parse_newick_tree(cfg["cached_ft_nwk"].read_text())
            parsed_trees[key]["FastTreeMP_LG"] = r_ft_cached
            results[key]["methods"]["FastTreeMP_LG"] = evaluate_method(
                "FastTreeMP_LG (cached)",
                r_ft_cached,
                ref_root,
                tax_map,
                0.0,
                {"note": "Cached tree; fresh 64-core timing runs in Phase 2"},
            )

        save_checkpoint(results, t_start)

    # =========================================================================
    # PHASE 2: Fresh 64-Core FastTreeMP (-lg) & IQ-TREE 2 (-m LG+G4) Timing
    # =========================================================================
    print(f"\n=== PHASE 2: Fresh {THREADS}-Core FastTreeMP (-lg) & IQ-TREE 2 (-m LG+G4) ===", flush=True)
    for cfg in datasets_cfg:
        key = cfg["key"]
        ref_root = loaded_refs[key]
        tax_map = loaded_tax[key]
        print(f"\n>>> SOTA ML Baselines for {cfg['title']}", flush=True)

        # 2A. Fresh FastTreeMP -lg on identical 64-core node
        ft_nwk = BENCH_DIR / f"{key}_fasttree_lg_t{THREADS}.nwk"
        t0 = time.perf_counter()
        subprocess.run(
            [FASTTREE_BIN, "-lg", "-quiet", str(cfg["faa"])],
            stdout=open(ft_nwk, "w"),
            stderr=subprocess.DEVNULL,
            env=env_omp,
            check=True,
        )
        ft_wall = time.perf_counter() - t0
        r_ft = parse_newick_tree(ft_nwk.read_text())
        parsed_trees[key]["FastTreeMP_LG"] = r_ft
        bw_sec = results[key]["methods"]["Bushwhacker_LG08_Binary"]["wall_sec"]
        results[key]["methods"]["FastTreeMP_LG"] = evaluate_method(
            "FastTreeMP_LG",
            r_ft,
            ref_root,
            tax_map,
            ft_wall,
            {"speedup_of_bushwhacker_x": round(ft_wall / max(0.01, bw_sec), 2)},
        )
        save_checkpoint(results, t_start)

        # 2B. IQ-TREE 2 (-m LG+G4 --fast -T 64)
        iq_fast_prefix = BENCH_DIR / f"{key}_iqtree_lg_g4_fast"
        t0 = time.perf_counter()
        subprocess.run(
            [
                IQTREE_BIN,
                "-s",
                str(cfg["faa"]),
                "-m",
                "LG+G4",
                "--fast",
                "-T",
                str(THREADS),
                "-pre",
                str(iq_fast_prefix),
                "-redo",
                "-quiet",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        iq_fast_wall = time.perf_counter() - t0
        iq_fast_treefile = Path(f"{iq_fast_prefix}.treefile")
        r_iq_fast = parse_newick_tree(iq_fast_treefile.read_text())
        parsed_trees[key]["IQTREE2_LG_G4_fast"] = r_iq_fast
        results[key]["methods"]["IQTREE2_LG_G4_fast"] = evaluate_method(
            "IQTREE2_LG_G4_fast",
            r_iq_fast,
            ref_root,
            tax_map,
            iq_fast_wall,
            {"speedup_of_bushwhacker_x": round(iq_fast_wall / max(0.01, bw_sec), 2)},
        )

        # Update pairwise nRF matrix across methods
        mnames = ["Bushwhacker_LG08_MDL", "Bushwhacker_LG08_Binary", "FastTreeMP_LG", "IQTREE2_LG_G4_fast"]
        pw = {}
        for i, m1 in enumerate(mnames):
            for m2 in mnames[i + 1 :]:
                if m1 in parsed_trees[key] and m2 in parsed_trees[key]:
                    pw[f"{m1}__vs__{m2}"] = compare_unrooted_trees(
                        parsed_trees[key][m1], parsed_trees[key][m2]
                    )["nrf"]
        results[key]["pairwise_nrf"] = pw
        save_checkpoint(results, t_start)

    # =========================================================================
    # PHASE 3: Standard Full IQ-TREE 2 (-m LG+G4 without --fast)
    # =========================================================================
    print(f"\n=== PHASE 3: Full IQ-TREE 2 (-m LG+G4) Hill-Climbing ML Search ===", flush=True)
    for cfg in datasets_cfg:
        key = cfg["key"]
        ref_root = loaded_refs[key]
        tax_map = loaded_tax[key]
        iq_full_prefix = BENCH_DIR / f"{key}_iqtree_lg_g4_full"
        print(f"\n>>> Running Full IQ-TREE 2 (-m LG+G4) on {key}...", flush=True)
        t0 = time.perf_counter()
        subprocess.run(
            [
                IQTREE_BIN,
                "-s",
                str(cfg["faa"]),
                "-m",
                "LG+G4",
                "-T",
                str(THREADS),
                "-pre",
                str(iq_full_prefix),
                "-redo",
                "-quiet",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        iq_full_wall = time.perf_counter() - t0
        iq_full_treefile = Path(f"{iq_full_prefix}.treefile")
        r_iq_full = parse_newick_tree(iq_full_treefile.read_text())
        parsed_trees[key]["IQTREE2_LG_G4_full"] = r_iq_full
        bw_sec = results[key]["methods"]["Bushwhacker_LG08_Binary"]["wall_sec"]
        results[key]["methods"]["IQTREE2_LG_G4_full"] = evaluate_method(
            "IQTREE2_LG_G4_full",
            r_iq_full,
            ref_root,
            tax_map,
            iq_full_wall,
            {"speedup_of_bushwhacker_x": round(iq_full_wall / max(0.01, bw_sec), 2)},
        )

        mnames = [
            "Bushwhacker_LG08_MDL",
            "Bushwhacker_LG08_Binary",
            "FastTreeMP_LG",
            "IQTREE2_LG_G4_fast",
            "IQTREE2_LG_G4_full",
        ]
        pw = {}
        for i, m1 in enumerate(mnames):
            for m2 in mnames[i + 1 :]:
                if m1 in parsed_trees[key] and m2 in parsed_trees[key]:
                    pw[f"{m1}__vs__{m2}"] = compare_unrooted_trees(
                        parsed_trees[key][m1], parsed_trees[key][m2]
                    )["nrf"]
        results[key]["pairwise_nrf"] = pw
        save_checkpoint(results, t_start)

    print(f"\n[DONE] GTDB SOTA Protein Benchmark saved to {OUT_JSON} in {time.time() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
