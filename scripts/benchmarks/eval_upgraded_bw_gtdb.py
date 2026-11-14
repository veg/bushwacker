#!/usr/bin/env python3
"""
Standalone SOTA Evaluation of Upgraded Site-Tiled LG08+CAT8 Bushwhacker on GTDB Domains
=======================================================================================
Runs upgraded 20-state Bushwhacker (-T 32 --protein --no-shunts) on:
  1. GTDB r95 Archaeal Domain (1,672 species x 32,675 AA; 122 markers)
  2. GTDB r95 Bacterial Domain (1,000 species across all 111 phyla x 43,733 AA; 120 markers)
and compares head-to-head against GTDB r95 curated reference trees, pre-fix Bushwhacker,
FastTreeMP (-lg), and any completed IQ-TREE 2 (-m LG+G4) trees from Job 1367512.
"""


import sys as _sys
from pathlib import Path as _Path
if str(_Path(__file__).resolve().parent) not in _sys.path:
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
from _env_paths import REPO_ROOT, DATA_DIR, SPECIMENS_DIR, ALIGNMENTS_DIR, SINGLE_CELL_DIR, RESULTS_DIR, WORK_ROOT, EXTERNAL_DIR, resolve_bin, ASTRAL_JAR

import os
import json
import time
import subprocess
from pathlib import Path
from collections import defaultdict

from run_gtdb_sota_protein_bench import (
    BUSHWHACKER_BIN,
    GTDB_DIR,
    SM_DIR,
    BENCH_DIR,
    parse_newick_tree,
    load_gtdb_taxonomy,
    evaluate_5rank_taxonomy,
    evaluate_method,
    compare_unrooted_trees,
)

OUT_JSON = RESULTS_DIR / "microbial_viral" / "gtdb_sota_bw_upgraded.json"
THREADS = int(os.environ.get("BENCH_THREADS", "32"))


def main():
    t_start = time.time()
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

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

    # Load any fresh FastTreeMP / IQ-TREE 2 timings already recorded by Job 1367512 or pillar2_megatrees.json
    prev_bench_json = WORK_ROOT / "pillar2" / "gtdb_sota_protein_bench.json"
    if not prev_bench_json.exists():
        prev_bench_json = RESULTS_DIR / "gtdb_sota_protein_bench.json"
    prev_data = json.loads(prev_bench_json.read_text()) if prev_bench_json.exists() else {}
    p2_json = RESULTS_DIR / "pillar2_megatrees.json"
    p2_data = json.loads(p2_json.read_text()) if p2_json.exists() else {}

    results = {}
    for cfg in datasets_cfg:
        key = cfg["key"]
        print(f"\n>>> Evaluating Upgraded LG08+CAT8 Bushwhacker (-T {THREADS}) on {cfg['title']}", flush=True)
        ref_root = parse_newick_tree(cfg["ref_tree"].read_text())
        tax_map = load_gtdb_taxonomy(cfg["tax_tsv"])
        results[key] = {
            "title": cfg["title"],
            "methods": {},
            "pairwise_nrf": {},
            "gtdb_ref_taxonomy_5ranks": evaluate_5rank_taxonomy(ref_root, tax_map),
        }
        trees = {"GTDB_Ref": ref_root}

        if cfg["old_bw_json"].exists():
            old_bw = json.loads(cfg["old_bw_json"].read_text())
            r_old = parse_newick_tree(old_bw["binary_newick"])
            results[key]["methods"]["Bushwhacker_v0_Unweighted_Fitch_Ablation"] = evaluate_method(
                "Bushwhacker_v0_Unweighted_Fitch_Ablation",
                r_old,
                ref_root,
                tax_map,
                old_bw.get("runtime_sec", 0.0),
            )

        bw_json = BENCH_DIR / f"{key}_bw_lg08_cat8.json"
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
        trees["Bushwhacker_LG08_CAT8_MDL"] = r_bw_mdl
        trees["Bushwhacker_LG08_CAT8_Binary"] = r_bw_bin

        results[key]["methods"]["Bushwhacker_LG08_CAT8_MDL"] = evaluate_method(
            "Bushwhacker_LG08_CAT8_MDL",
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
            },
        )
        results[key]["methods"]["Bushwhacker_LG08_CAT8_Binary"] = evaluate_method(
            "Bushwhacker_LG08_CAT8_Binary",
            r_bw_bin,
            ref_root,
            tax_map,
            bw_wall,
            {"engine_sec": round(bw_data["runtime_sec"], 2)},
        )

        # Check fresh 32-core FastTreeMP if finished, else cached FastTreeMP
        ft_fresh = BENCH_DIR / f"{key}_fasttree_lg_t32.nwk"
        prev_ft_wall = (
            prev_data.get("datasets", {})
            .get(key, {})
            .get("methods", {})
            .get("FastTreeMP_LG", {})
            .get("wall_sec", 0.0)
        )
        if not prev_ft_wall:
            p2_part = p2_data.get("part_2b_gtdb_domain_supermatrices", {})
            p2_key = (
                "ar122_archaeal_domain_supermatrix"
                if "ar122" in key
                else "bac120_bacterial_domain_supermatrix"
            )
            prev_ft_wall = float(p2_part.get(p2_key, {}).get("fasttree_lg_t32_sec", 0.0))
        if ft_fresh.exists() and ft_fresh.stat().st_size > 100:
            r_ft = parse_newick_tree(ft_fresh.read_text())
            trees["FastTreeMP_LG"] = r_ft
            results[key]["methods"]["FastTreeMP_LG"] = evaluate_method(
                "FastTreeMP_LG (T=32)",
                r_ft,
                ref_root,
                tax_map,
                prev_ft_wall,
                {"speedup_of_bushwhacker_x": round(prev_ft_wall / max(0.01, bw_wall), 2) if prev_ft_wall > 0 else None},
            )
        elif cfg["cached_ft_nwk"].exists():
            r_ft = parse_newick_tree(cfg["cached_ft_nwk"].read_text())
            trees["FastTreeMP_LG"] = r_ft
            results[key]["methods"]["FastTreeMP_LG"] = evaluate_method(
                "FastTreeMP_LG (cached)",
                r_ft,
                ref_root,
                tax_map,
                prev_ft_wall,
            )

        # Check any completed IQ-TREE 2 trees
        for iq_tag, iq_label in [("fast", "IQTREE2_LG_G4_fast"), ("full", "IQTREE2_LG_G4_full")]:
            tf = BENCH_DIR / f"{key}_iqtree_lg_g4_{iq_tag}.treefile"
            if tf.exists() and tf.stat().st_size > 100:
                iq_wall = (
                    prev_data.get("datasets", {})
                    .get(key, {})
                    .get("methods", {})
                    .get(iq_label, {})
                    .get("wall_sec", 0.0)
                )
                r_iq = parse_newick_tree(tf.read_text())
                trees[iq_label] = r_iq
                results[key]["methods"][iq_label] = evaluate_method(
                    iq_label,
                    r_iq,
                    ref_root,
                    tax_map,
                    iq_wall,
                    {"speedup_of_bushwhacker_x": round(iq_wall / max(0.01, bw_wall), 2) if iq_wall > 0 else None},
                )

        mnames = [m for m in ["Bushwhacker_LG08_CAT8_MDL", "Bushwhacker_LG08_CAT8_Binary", "FastTreeMP_LG", "IQTREE2_LG_G4_fast", "IQTREE2_LG_G4_full"] if m in trees]
        pw = {}
        for i, m1 in enumerate(mnames):
            for m2 in mnames[i + 1 :]:
                pw[f"{m1}__vs__{m2}"] = compare_unrooted_trees(trees[m1], trees[m2])["nrf"]
        results[key]["pairwise_nrf"] = pw

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "threads": THREADS,
        "elapsed_wall_sec": round(time.time() - t_start, 2),
        "datasets": results,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"\n[DONE] Wrote {OUT_JSON} in {time.time() - t_start:.2f}s", flush=True)


if __name__ == "__main__":
    main()
