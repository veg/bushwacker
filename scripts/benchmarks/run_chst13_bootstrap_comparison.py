#!/usr/bin/env python3
"""Benchmark ML bootstrap, SH-aLRT, aBayes, IQ-TREE --polytomy, FastTree, and
standard nonparametric bootstrap baselines on mammalian CHST13 (ENSG00000180767)
and record structured results inside results/macroevolution/fig3_empirical_tanglegram_gene.json.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
import time
from pathlib import Path
from ete3 import Tree

from _env_paths import (
    BUSHWHACKER_BIN,
    FASTTREE_BIN,
    IQTREE_BIN,
    REPO_ROOT,
    RESULTS_DIR,
    SPECIMENS_DIR,
)


GENE_ID = "ENSG00000180767"


def canon_split(clade: frozenset[str], all_tips: frozenset[str]) -> frozenset[str]:
    comp = all_tips - clade
    if len(clade) < len(comp):
        return clade
    if len(comp) < len(clade):
        return comp
    return clade if sorted(clade)[0] < sorted(comp)[0] else comp


def extract_splits(tree: Tree, all_tips: frozenset[str]) -> dict[frozenset[str], dict]:
    out: dict[frozenset[str], dict] = {}
    for node in tree.traverse():
        if node.is_leaf() or node.is_root():
            continue
        lv = frozenset(node.get_leaf_names())
        if len(lv) <= 1 or len(lv) >= len(all_tips) - 1:
            continue
        key = canon_split(lv, all_tips)
        if key not in out:
            out[key] = {
                "raw_support": str(node.name).strip() if node.name is not None else "",
                "branch_length": float(node.dist),
                "clade": sorted(key),
            }
    return out


def parse_iqtree_times(iqtree_log_path: Path, wall_fallback: float) -> tuple[float, float]:
    if not iqtree_log_path.exists():
        return wall_fallback, wall_fallback
    txt = iqtree_log_path.read_text(encoding="utf-8", errors="replace")
    m_cpu = re.search(r"Total CPU time used:\s*([0-9.]+)\s*sec", txt)
    m_wall = re.search(r"Total wall-clock time used:\s*([0-9.]+)\s*sec", txt)
    cpu_s = float(m_cpu.group(1)) if m_cpu else wall_fallback
    wall_s = float(m_wall.group(1)) if m_wall else wall_fallback
    return cpu_s, wall_s


def eval_splits(cand: set[frozenset[str]], ref: set[frozenset[str]], runtime_sec: float) -> dict:
    tp = len(cand & ref)
    fp = len(cand - ref)
    fn = len(ref - cand)
    prec = tp / len(cand) if cand else 0.0
    rec = tp / len(ref) if ref else 0.0
    rf = fp + fn
    nrf = rf / (len(cand) + len(ref)) if (cand or ref) else 0.0
    return {
        "resolved_splits": len(cand),
        "max_binary_splits": len(ref),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "rf": rf,
        "nrf": round(nrf, 6),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "runtime_sec": round(runtime_sec, 4),
    }


def main() -> None:
    aln = SPECIMENS_DIR / "CHST13_ENSG00000180767_22mammals.fasta"
    bundle_path = RESULTS_DIR / "macroevolution" / "fig3_empirical_tanglegram_gene.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    gene_entry = bundle["genes"][GENE_ID]

    bw_stored = gene_entry["bw_json"]
    t_bw_stored = Tree(bw_stored["newick"], format=1)
    all_tips = frozenset(t_bw_stored.get_leaf_names())

    t_ref = Tree(bundle["reference_astral_newick"], format=1)
    t_ref.prune(sorted(all_tips))
    ref_splits = set(extract_splits(t_ref, all_tips).keys())

    t_ml_obs = Tree(gene_entry["ml_best_observed_newick"], format=1)
    ml_obs_splits = extract_splits(t_ml_obs, all_tips)
    bw_stored_splits = extract_splits(t_bw_stored, all_tips)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # 1. Live single-thread Bushwhacker check (kept separate from stored Fig 3A bw_json)
        print("[1/6] Running Bushwhacker -T 1 ...", flush=True)
        t0 = time.perf_counter()
        subprocess.run(
            [BUSHWHACKER_BIN, str(aln), "-T", "1", "-o", str(tmp / "bw.nwk"), "-j", str(tmp / "bw.json")],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        bw_live_wall = time.perf_counter() - t0
        bw_live = json.loads((tmp / "bw.json").read_text(encoding="utf-8"))
        t_bw_live = Tree(bw_live["newick"], format=1)
        bw_live_splits = extract_splits(t_bw_live, all_tips)

        # 2. Unbootstrapped IQ-TREE GTR+G (1 thread)
        print("[2/6] Running IQ-TREE 2 GTR+G raw ML (1 thread) ...", flush=True)
        t0 = time.perf_counter()
        subprocess.run(
            [IQTREE_BIN, "-s", str(aln), "-m", "GTR+G", "-nt", "1", "-seed", "42", "-pre", str(tmp / "raw")],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        raw_wall = time.perf_counter() - t0
        raw_cpu, raw_log_wall = parse_iqtree_times(tmp / "raw.iqtree", raw_wall)
        raw_nwk = (tmp / "raw.treefile").read_text(encoding="utf-8").strip()
        raw_splits = extract_splits(Tree(raw_nwk, format=1), all_tips)

        # 3. IQ-TREE GTR+G --polytomy (1 thread)
        print("[3/6] Running IQ-TREE 2 GTR+G --polytomy (1 thread) ...", flush=True)
        t0 = time.perf_counter()
        subprocess.run(
            [IQTREE_BIN, "-s", str(aln), "-m", "GTR+G", "--polytomy", "-nt", "1", "-seed", "42", "-pre", str(tmp / "poly")],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        poly_wall = time.perf_counter() - t0
        poly_cpu, poly_log_wall = parse_iqtree_times(tmp / "poly.iqtree", poly_wall)
        poly_nwk = (tmp / "poly.treefile").read_text(encoding="utf-8").strip()
        poly_splits = extract_splits(Tree(poly_nwk, format=1), all_tips)

        # 4. IQ-TREE GTR+G UFBoot(1000) + SH-aLRT(1000) + aBayes (1 thread)
        print("[4/6] Running IQ-TREE 2 GTR+G -B 1000 -alrt 1000 -abayes (1 thread) ...", flush=True)
        t0 = time.perf_counter()
        subprocess.run(
            [
                IQTREE_BIN, "-s", str(aln), "-m", "GTR+G",
                "-B", "1000", "-alrt", "1000", "-abayes",
                "-nt", "1", "-seed", "42", "-pre", str(tmp / "uf"),
            ],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        uf_wall = time.perf_counter() - t0
        uf_cpu, uf_log_wall = parse_iqtree_times(tmp / "uf.iqtree", uf_wall)
        uf_nwk = (tmp / "uf.treefile").read_text(encoding="utf-8").strip()
        uf_contree_nwk = (tmp / "uf.contree").read_text(encoding="utf-8").strip()
        uf_splits = extract_splits(Tree(uf_nwk, format=1), all_tips)

        # 5. Standard Felsenstein nonparametric bootstrap (100 replicates, 1 thread)
        print("[5/6] Running IQ-TREE 2 GTR+G -b 100 standard bootstrap (1 thread, ~5 min) ...", flush=True)
        t0 = time.perf_counter()
        subprocess.run(
            [IQTREE_BIN, "-s", str(aln), "-m", "GTR+G", "-b", "100", "-nt", "1", "-seed", "42", "-pre", str(tmp / "sbs")],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        sbs_wall = time.perf_counter() - t0
        sbs_cpu, sbs_log_wall = parse_iqtree_times(tmp / "sbs.iqtree", sbs_wall)
        sbs_nwk = (tmp / "sbs.treefile").read_text(encoding="utf-8").strip()
        sbs_contree_nwk = (tmp / "sbs.contree").read_text(encoding="utf-8").strip()
        sbs_splits = extract_splits(Tree(sbs_nwk, format=1), all_tips)

        # 6. FastTree -nt -gtr (single thread)
        print("[6/6] Running FastTree -nt -gtr ...", flush=True)
        t0 = time.perf_counter()
        ft_proc = subprocess.run(
            [FASTTREE_BIN, "-nt", "-gtr", str(aln)],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        ft_wall = time.perf_counter() - t0
        ft_nwk = ft_proc.stdout.strip()
        ft_splits = extract_splits(Tree(ft_nwk, format=1), all_tips)

    # Parse split-level support vector across the ML topology
    assert set(uf_splits.keys()) == set(ml_obs_splits.keys()), (
        "UFBoot ML bipartition set differs from ml_best_observed_newick"
    )
    split_records = []
    uf_by_key: dict[frozenset[str], dict] = {}
    sbs_by_key: dict[frozenset[str], float | None] = {}

    for k, meta in sbs_splits.items():
        try:
            sbs_by_key[k] = float(meta["raw_support"])
        except ValueError:
            sbs_by_key[k] = None

    for k, meta in sorted(uf_splits.items(), key=lambda kv: (kv[0] in ref_splits, kv[1]["branch_length"])):
        parts = meta["raw_support"].split("/")
        sh_alrt = float(parts[0])
        abayes = float(parts[1])
        ufboot = float(parts[2])
        rec = {
            "clade": meta["clade"],
            "clade_short": [c.split("_", 1)[1] if "_" in c else c for c in meta["clade"]],
            "in_ref": k in ref_splits,
            "in_bw_stored": k in bw_stored_splits,
            "in_polytomy": k in poly_splits,
            "branch_length": round(meta["branch_length"], 6),
            "sh_alrt": round(sh_alrt, 2),
            "abayes": round(abayes, 4),
            "ufboot": int(round(ufboot)),
            "std_bs100": int(round(sbs_by_key[k])) if sbs_by_key.get(k) is not None else None,
        }
        split_records.append(rec)
        uf_by_key[k] = rec

    def uf_filter(min_uf: float = 0.0, min_alrt: float = 0.0, min_abayes: float = 0.0) -> set[frozenset[str]]:
        return {
            k for k, r in uf_by_key.items()
            if r["ufboot"] >= min_uf and r["sh_alrt"] >= min_alrt and r["abayes"] >= min_abayes
        }

    def sbs_filter(min_bs: float) -> set[frozenset[str]]:
        kept = set()
        for k, meta in sbs_splits.items():
            try:
                val = float(meta["raw_support"])
            except ValueError:
                val = 0.0
            if val >= min_bs:
                kept.add(k)
        return kept

    def ft_filter(min_sh: float) -> set[frozenset[str]]:
        kept = set()
        for k, meta in ft_splits.items():
            try:
                val = float(meta["raw_support"])
            except ValueError:
                val = 0.0
            if val >= min_sh:
                kept.add(k)
        return kept

    bw_stored_rt = float(bw_stored["runtime_sec"])
    bw_live_rt = float(bw_live.get("runtime_sec", bw_live_wall))

    methods_summary = {
        "Raw_ML": eval_splits(set(ml_obs_splits.keys()), ref_splits, raw_cpu),
        "IQTREE_polytomy": eval_splits(set(poly_splits.keys()), ref_splits, poly_cpu),
        "UFBoot_10": eval_splits(uf_filter(min_uf=10), ref_splits, uf_cpu),
        "UFBoot_30": eval_splits(uf_filter(min_uf=30), ref_splits, uf_cpu),
        "UFBoot_50": eval_splits(uf_filter(min_uf=50), ref_splits, uf_cpu),
        "UFBoot_70": eval_splits(uf_filter(min_uf=70), ref_splits, uf_cpu),
        "UFBoot_95": eval_splits(uf_filter(min_uf=95), ref_splits, uf_cpu),
        "SH_aLRT_80": eval_splits(uf_filter(min_alrt=80), ref_splits, uf_cpu),
        "aBayes_75": eval_splits(uf_filter(min_abayes=0.75), ref_splits, uf_cpu),
        "aBayes_95": eval_splits(uf_filter(min_abayes=0.95), ref_splits, uf_cpu),
        "SHaLRT80_UFBoot95": eval_splits(uf_filter(min_uf=95, min_alrt=80), ref_splits, uf_cpu),
        "StdBS_50": eval_splits(sbs_filter(50), ref_splits, sbs_cpu),
        "StdBS_70": eval_splits(sbs_filter(70), ref_splits, sbs_cpu),
        "StdBS_95": eval_splits(sbs_filter(95), ref_splits, sbs_cpu),
        "FastTree_Raw": eval_splits(set(ft_splits.keys()), ref_splits, ft_wall),
        "FastTree_SH70": eval_splits(ft_filter(0.70), ref_splits, ft_wall),
        "FastTree_SH90": eval_splits(ft_filter(0.90), ref_splits, ft_wall),
        "Bushwhacker_MDL": eval_splits(set(bw_stored_splits.keys()), ref_splits, bw_stored_rt),
        "Bushwhacker_MDL_live_T1": eval_splits(set(bw_live_splits.keys()), ref_splits, bw_live_rt),
    }

    gene_entry["bootstrap_comparison"] = {
        "alignment": str(aln.relative_to(REPO_ROOT)),
        "iqtree_ufboot_alrt_abayes_newick": uf_nwk,
        "iqtree_ufboot_contree_newick": uf_contree_nwk,
        "iqtree_polytomy_newick": poly_nwk,
        "iqtree_raw_newick": raw_nwk,
        "iqtree_std_bs100_newick": sbs_nwk,
        "iqtree_std_bs100_contree_newick": sbs_contree_nwk,
        "fasttree_sh_newick": ft_nwk,
        "runtimes_1core_sec": {
            "bw_1core_sec": round(bw_stored_rt, 4),
            "bw_live_1core_sec": round(bw_live_rt, 4),
            "bw_live_wall_sec": round(bw_live_wall, 4),
            "ml_raw_1core_sec": round(raw_cpu, 4),
            "polytomy_1core_sec": round(poly_cpu, 4),
            "ufboot1000_1core_sec": round(uf_cpu, 4),
            "std_bs100_cpu_sec": round(sbs_cpu, 4),
            "std_bs100_wall_sec": round(sbs_log_wall, 4),
            "fasttree_1core_sec": round(ft_wall, 4),
        },
        "ml_splits": split_records,
        "methods_summary": methods_summary,
    }

    bundle_path.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    print(f"\n[SAVED] Updated {bundle_path} with bootstrap_comparison for {GENE_ID}\n")
    print("Runtimes (1 core):", json.dumps(gene_entry["bootstrap_comparison"]["runtimes_1core_sec"], indent=2))
    print("\nMethods Summary:")
    for name, m in methods_summary.items():
        print(
            f"  {name:24s} | res={m['resolved_splits']:2d} TP={m['tp']:2d} FP={m['fp']:2d} "
            f"RF={m['rf']:2d} prec={m['precision']*100:5.1f}% rec={m['recall']*100:5.1f}% "
            f"time={m['runtime_sec']:.4f}s"
        )


if __name__ == "__main__":
    main()
