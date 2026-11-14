#!/usr/bin/env python3
"""
Fast Completion of Pillar 1B:
  1. GTDB Domain-Wide Coalescent Species Trees (ar122 1,672 Archaea & bac120 1,000 Bacteria)
     via ASTER astral4 (FastTree_GTR, Bushwhacker_Binary, Bushwhacker_MDL).
  2. 882-Mammal TOGA2 Coalescent Species Tree across High-Completeness Single-Copy Orthologs
     (n_taxa >= 750, sorted deterministically by coding length, top 2,500 loci) AND full 19,952
     Protein_MDL (which has 65% polytomies so X is compact!) via fast topology-only astral4 (-r 1 -s 1 -u 0).
"""


import sys as _sys
from pathlib import Path as _Path
if str(_Path(__file__).resolve().parent) not in _sys.path:
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
from _env_paths import REPO_ROOT, DATA_DIR, SPECIMENS_DIR, ALIGNMENTS_DIR, SINGLE_CELL_DIR, RESULTS_DIR, WORK_ROOT, EXTERNAL_DIR, resolve_bin, ASTRAL_JAR

import os
import re
import glob
import gzip
import json
import time
import subprocess
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

ASTRAL4_BIN = resolve_bin("ASTRAL4_BIN", "astral4")
WORK_DIR = WORK_ROOT / "pillar1_fast1b"
OUT_JSON = RESULTS_DIR / "macroevolution" / "pillar1_part1c.json"

GTDB_DIR = EXTERNAL_DIR / "gtdb_r95"
TOGA_NT_DIR = WORK_ROOT / "toga2_nt" / "results"
TOGA_AA_DIR = WORK_ROOT / "toga2_aa" / "results"
TOGA_SP_TREE = Path("data/alignments/supermatrices/toga2_speciesTree.nh")
TOGA_TAX_TSV = ALIGNMENTS_DIR / "supermatrices" / "toga2_assemblies_and_species.tsv"


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
    if len(root.children) == 1 and not root.name:
        return root.children[0]
    return root


def extract_unrooted_splits(root: TreeNode, leaf_to_idx: dict):
    n = len(leaf_to_idx)
    full = (1 << n) - 1
    splits = set()
    stack = [(root, False)]
    masks = {}
    while stack:
        nd, visited = stack.pop()
        if not nd.children:
            idx = leaf_to_idx.get(nd.name)
            masks[id(nd)] = (1 << idx) if idx is not None else 0
            continue
        if not visited:
            stack.append((nd, True))
            for c in nd.children:
                stack.append((c, False))
        else:
            m = 0
            for c in nd.children:
                m |= masks[id(c)]
            masks[id(nd)] = m
            if nd is not root:
                cnt = bin(m).count("1")
                if 2 <= cnt <= n - 2:
                    comp = full ^ m
                    splits.add(m if m < comp else comp)
    return splits


def collect_leaves(root: TreeNode):
    out = []
    stack = [root]
    while stack:
        nd = stack.pop()
        if not nd.children:
            if nd.name:
                out.append(nd.name)
        else:
            for c in nd.children:
                stack.append(c)
    return out


def compare_unrooted_trees(pred_nwk: str, ref_nwk: str) -> dict:
    t_pred = parse_newick_tree(pred_nwk)
    t_ref = parse_newick_tree(ref_nwk)
    shared = sorted(set(collect_leaves(t_pred)) & set(collect_leaves(t_ref)))
    n = len(shared)
    if n < 4:
        return {"n_shared": n, "nrf": 1.0}
    l2i = {nm: idx for idx, nm in enumerate(shared)}
    sp_pred = extract_unrooted_splits(t_pred, l2i)
    sp_ref = extract_unrooted_splits(t_ref, l2i)
    tp = len(sp_pred & sp_ref)
    fp = len(sp_pred - sp_ref)
    fn = len(sp_ref - sp_pred)
    rf = fp + fn
    denom = len(sp_pred) + len(sp_ref)
    nrf = rf / denom if denom > 0 else 0.0
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
        "nrf": round(nrf, 5),
        "precision": round(prec, 5),
        "recall": round(rec, 5),
    }


def evaluate_group_monophyly(tree_nwk: str, leaf_to_group: dict) -> dict:
    t = parse_newick_tree(tree_nwk)
    leaves = sorted(nm for nm in collect_leaves(t) if leaf_to_group.get(nm))
    n = len(leaves)
    if n < 4:
        return {"n_groups": 0}
    l2i = {nm: i for i, nm in enumerate(leaves)}
    full = (1 << n) - 1
    splits = extract_unrooted_splits(t, l2i)
    grp_masks = defaultdict(int)
    grp_counts = defaultdict(int)
    for nm, idx in l2i.items():
        g = leaf_to_group[nm]
        grp_masks[g] |= (1 << idx)
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


def run_astral4_fast(gene_trees: list, out_dir: Path, tag: str, threads: int = 64, fast_rounds: bool = False):
    in_path = out_dir / f"{tag}.in.tre"
    out_path = out_dir / f"{tag}.out.tre"
    with open(in_path, "w") as f:
        for gt in gene_trees:
            s = gt.strip()
            if s:
                if not s.endswith(";"):
                    s += ";"
                f.write(s + "\n")
    cmd = [ASTRAL4_BIN, "-t", str(threads)]
    if fast_rounds:
        cmd.extend(["-u", "0", "-r", "1", "-s", "1"])
    cmd.extend(["-o", str(out_path), str(in_path)])
    t0 = time.perf_counter()
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    dt = time.perf_counter() - t0
    with open(out_path) as f:
        sp = f.read().strip()
    return sp, dt


def _read_nt_meta(p: str):
    d = json.load(gzip.open(p))
    stem = Path(p).name.replace(".bw.json.gz", "")
    return (stem, d.get("n_taxa", 0), d.get("seq_len", 0), d["binary_newick"].strip(), d["newick"].strip())


def _read_aa_meta(p: str):
    d = json.load(gzip.open(p))
    stem = Path(p).name.replace(".bwaa.json.gz", "")
    return (stem, d["free_newick"].strip())


def main():
    t_start = time.time()
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    # 1. GTDB Domain-Wide Coalescent Species Trees (122 Archaea & 120 Bacteria)
    print("[Part 1B-Fast] Step 1: GTDB Domain Coalescent Species Trees...", flush=True)
    gtdb_results = {}
    gtdb_work = GTDB_DIR / "work"
    for dom, ref_tree_fn in [("ar122", "ar122_r95.tree"), ("bac120", "bac120_r95.tree")]:
        ref_nwk = (GTDB_DIR / ref_tree_fn).read_text().strip()
        gtdb_results[dom] = {}
        for mode, ext in [
            ("FastTree_GTR", "fasttree.nwk"),
            ("Bushwhacker_Binary", "bw_bin.nwk"),
            ("Bushwhacker_MDL", "bw_mdl.nwk"),
        ]:
            files = sorted(glob.glob(str(gtdb_work / f"{dom}_*.{ext}")))
            gtrees = [Path(p).read_text().strip() for p in files]
            sp_nwk, a_sec = run_astral4_fast(gtrees, WORK_DIR, f"gtdb_{dom}_{mode}", threads=32, fast_rounds=False)
            cmp = compare_unrooted_trees(sp_nwk, ref_nwk)
            gtdb_results[dom][mode] = {
                "n_loci": len(gtrees),
                "astral4_sec": round(a_sec, 3),
                "vs_gtdb_ref_tree": cmp,
            }
            print(f"  GTDB {dom} {mode}: {len(gtrees)} loci in {a_sec:.2f}s -> nRF={cmp['nrf']:.4f}", flush=True)

    # Save partial immediately
    with open(OUT_JSON, "w") as f:
        json.dump({"gtdb_prokaryotes": gtdb_results}, f, indent=2)

    # 2. 882-Mammal TOGA2 Coalescent Species Trees
    print("[Part 1B-Fast] Step 2: Loading TOGA2 882-Mammal Gene Trees...", flush=True)
    ref_mam_nwk = TOGA_SP_TREE.read_text().strip()
    asm_to_order = {}
    asm_to_family = {}
    if TOGA_TAX_TSV.exists():
        lines = TOGA_TAX_TSV.read_text().splitlines()
        hdr = lines[0].split("\t")
        for ln in lines[1:]:
            parts = ln.split("\t")
            if len(parts) < len(hdr):
                continue
            row = dict(zip(hdr, parts))
            asm = row.get("assembly") or row.get("Assembly") or parts[0]
            if asm:
                asm_to_order[asm] = row.get("order") or row.get("Order") or ""
                asm_to_family[asm] = row.get("family") or row.get("Family") or ""

    nt_files = sorted(glob.glob(str(TOGA_NT_DIR / "*.bw.json.gz")))
    aa_files = sorted(glob.glob(str(TOGA_AA_DIR / "*.bwaa.json.gz")))
    with ProcessPoolExecutor(max_workers=48) as ex:
        nt_recs = list(ex.map(_read_nt_meta, nt_files, chunksize=128))
        aa_recs = dict(ex.map(_read_aa_meta, aa_files, chunksize=128))

    # Select top 2,500 high-completeness single-copy orthologs (n_taxa >= 750, longest coding loci)
    complete_recs = [r for r in nt_recs if r[1] >= 750 and r[0] in aa_recs]
    complete_recs.sort(key=lambda r: (-r[2], r[0]))
    top2500 = complete_recs[:2500]
    print(f"  Selected {len(top2500)} high-completeness mammalian orthologs (n_taxa>=750, out of {len(complete_recs)} eligible).", flush=True)

    mammal_results = {}
    for label, gtrees in [
        ("TOGA2_Top2500_Protein_MDL", [aa_recs[r[0]] for r in top2500]),
        ("TOGA2_Top2500_Codon_MDL", [r[4] for r in top2500]),
        ("TOGA2_Top2500_Codon_Binary", [r[3] for r in top2500]),
        ("TOGA2_All19952_Protein_MDL", [aa_recs[k] for k in sorted(aa_recs.keys())]),
    ]:
        t0 = time.perf_counter()
        sp_nwk, a_sec = run_astral4_fast(gtrees, WORK_DIR, label, threads=64, fast_rounds=True)
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
            f"  {label}: n={len(gtrees)}, astral={a_sec:.1f}s, nRF={cmp['nrf']:.5f} (RF={cmp['rf']}), "
            f"Orders={ord_m.get('strict_monophyletic')}/{ord_m.get('n_groups')}, "
            f"Families={fam_m.get('strict_monophyletic')}/{fam_m.get('n_groups')}",
            flush=True,
        )
        with open(OUT_JSON, "w") as f:
            json.dump(
                {
                    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "total_wall_sec": round(time.time() - t_start, 2),
                    "gtdb_prokaryotes": gtdb_results,
                    "toga2_882_mammals": mammal_results,
                },
                f,
                indent=2,
            )

    print(f"[Part 1B-Fast] ALL DONE in {time.time() - t_start:.1f}s -> {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
