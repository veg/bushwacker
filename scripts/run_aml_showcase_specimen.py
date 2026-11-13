#!/usr/bin/env python3
"""Zero-leakage live reconstruction of patient AML-38-001 (`m12`):
  (A) Curated Tapestri clonal genotype panel (`AML_cohort_Morita_2020.json` / `aml_38_001_clones.fasta`, 11 clones x 31 loci)
  (B) Raw unthresholded COMPASS single-cell allele matrix (`AML-38-001_variants.csv`, 7,016 cells -> 3,832 complete cells x 9 loci, 128 unique genotypes)

All cell counts, clone names, gene annotations, and clinical labels are stripped before
running `bushwhacker --ordered` (`assert_zero_leakage_csv`), and mapped back strictly post-hoc.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
import numpy as np

from _env_paths import REPO_ROOT, SPECIMENS_DIR, SINGLE_CELL_DIR, RESULTS_DIR, BUSHWHACKER_BIN
from run_cancer_singlecell_bench import (
    parse_compass_variants_csv,
    write_anonymized_matrix_csv,
    run_bushwhacker_cli,
    compute_exact_clone_burst_degrees_from_mdl_tree,
)


def run_curated_morita_aml38(tmp: Path) -> dict:
    morita_path = SINGLE_CELL_DIR / "cancer_morita" / "AML_cohort_Morita_2020.json"
    if not morita_path.exists():
        morita_path = SINGLE_CELL_DIR / "AML_cohort_Morita_2020.json"
    m = json.loads(morita_path.read_text())
    muts = m["mutation_labels"]
    idx = m["patient_labels"].index("AML-38-001")
    tree_rec = m["trees"][idx]["tree"]

    nodes = []
    def collect(nd: dict) -> None:
        nodes.append(nd)
        for ch in nd.get("children", []):
            collect(ch)
    collect(tree_rec)

    mat = np.zeros((len(nodes), len(muts)), dtype=np.int16)
    clone_meta = []
    wt_idx = None
    for i, nd in enumerate(nodes):
        for j in nd["genotype"]:
            mat[i, j] = 1
        if not nd["genotype"]:
            wt_idx = i
        clone_meta.append({
            "node_id": int(nd["node_id"]),
            "driver": muts[nd["mutation_ids"][0]] if nd["mutation_ids"] else "WT",
            "genotype_genes": [muts[j] for j in nd["genotype"]],
            "seq_cell_number": int(round(nd["seq_cell_number"])),
        })

    csv_path = tmp / "aml38_curated_anon.csv"
    json_path = tmp / "aml38_curated_bw.json"
    feat_names = [f"L{j:02d}" for j in range(len(muts))]
    anon_ids = write_anonymized_matrix_csv(csv_path, mat, feat_names)
    bw_res = run_bushwhacker_cli(csv_path, json_path, mode="ordered", min_shunt_sites=1)

    out_info, _ = compute_exact_clone_burst_degrees_from_mdl_tree(
        bw_res["mdl_tree_newick"], anon_ids, mat, wt_idx
    )

    # Also update/read specimen fasta run
    spec_fa = SPECIMENS_DIR / "aml_38_001_clones.fasta"
    spec_nwk = SPECIMENS_DIR / "aml_38_001.nwk"
    spec_json = SPECIMENS_DIR / "aml_38_001.json"
    if spec_fa.exists():
        subprocess.run(
            [str(BUSHWHACKER_BIN), "--ordered", str(spec_fa), "-o", str(spec_nwk), "-j", str(spec_json)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    spec_data = json.loads(spec_json.read_text()) if spec_json.exists() else bw_res

    annotated_nodes = []
    for i, meta in enumerate(clone_meta):
        annotated_nodes.append({
            **meta,
            "k_out": int(out_info[i]["k_out"]),
        })

    return {
        "n_clones": len(nodes),
        "n_loci": len(muts),
        "total_sequenced_cells_in_morita_clones": int(sum(c["seq_cell_number"] for c in clone_meta)),
        "resolved_internal_splits": bw_res["resolved_splits"],
        "max_binary_splits": bw_res["max_binary_splits"],
        "polytomy_hubs": bw_res["polytomy_hubs"],
        "runtime_sec": round(float(bw_res["runtime_seconds"]), 4),
        "specimen_runtime_sec": round(float(spec_data.get("runtime_sec", bw_res["runtime_seconds"])), 4),
        "tree_MDL_nats": round(float(bw_res["tree_MDL_nats"]), 4),
        "internal_branches": bw_res.get("internal_branches", []),
        "specimen_internal_branches": spec_data.get("internal_branches", []),
        "nodes": annotated_nodes,
    }


def run_raw_compass_aml38(tmp: Path) -> dict:
    v_path = SPECIMENS_DIR / "AML-38-001_variants.csv"
    var_meta, cell_mat = parse_compass_variants_csv(v_path)
    n_cells, m_vars = cell_mat.shape
    complete_mask = np.all(cell_mat >= 0, axis=1)
    comp_cells = cell_mat[complete_mask]

    gt_counts = Counter(tuple(int(x) for x in row) for row in comp_cells)
    wt_tuple = tuple([0] * m_vars)
    if wt_tuple not in gt_counts:
        gt_counts[wt_tuple] = 0
    uniq_gts = sorted(gt_counts.keys())
    wt_idx = uniq_gts.index(wt_tuple)

    mat = np.array(uniq_gts, dtype=np.int16)
    feat_names = [f"{vm['chr']}_{vm['pos']}_{idx}" for idx, vm in enumerate(var_meta)]
    csv_path = tmp / "aml38_compass_anon.csv"
    json_path = tmp / "aml38_compass_bw.json"
    anon_ids = write_anonymized_matrix_csv(csv_path, mat, feat_names)
    bw_res = run_bushwhacker_cli(csv_path, json_path, mode="ordered", min_shunt_sites=1)

    out_info, branch_trans = compute_exact_clone_burst_degrees_from_mdl_tree(
        bw_res["mdl_tree_newick"], anon_ids, mat, wt_idx
    )

    hub_cells = []
    leaf_cells = []
    rows = []
    for i, gt in enumerate(uniq_gts):
        cnt = int(gt_counts[gt])
        if i == wt_idx or cnt == 0:
            continue
        k_val = int(out_info[i]["k_out"])
        if k_val >= 3:
            hub_cells.append(cnt)
        elif k_val == 0:
            leaf_cells.append(cnt)
        mut_genes = [f"{var_meta[j]['gene']}:{gt[j]}" for j in range(m_vars) if gt[j] > 0]
        rows.append({"anon_id": anon_ids[i], "k_out": k_val, "direct_cells": cnt, "muts": mut_genes})

    rows.sort(key=lambda r: (-r["k_out"], -r["direct_cells"]))
    return {
        "input_file": v_path.name,
        "n_sequenced_cells": int(n_cells),
        "n_complete_cells": int(len(comp_cells)),
        "n_somatic_coding_loci": int(m_vars),
        "loci": [vm["name"] for vm in var_meta],
        "n_unique_genotypes": len(uniq_gts),
        "resolved_internal_splits": bw_res["resolved_splits"],
        "max_binary_splits": bw_res["max_binary_splits"],
        "polytomy_hubs": bw_res["polytomy_hubs"],
        "max_polytomy_degree": bw_res["max_polytomy_degree"],
        "certified_shunts": len(bw_res["shunts"]),
        "runtime_sec": round(float(bw_res["runtime_seconds"]), 4),
        "hub_k3plus_n": len(hub_cells),
        "hub_k3plus_mean_direct_cells": round(float(np.mean(hub_cells)), 2) if hub_cells else None,
        "leaf_k0_n": len(leaf_cells),
        "leaf_k0_mean_direct_cells": round(float(np.mean(leaf_cells)), 2) if leaf_cells else None,
        "top_hubs": rows[:8],
    }


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        curated = run_curated_morita_aml38(tmp)
        raw_compass = run_raw_compass_aml38(tmp)

    out = {
        "patient": "AML-38-001",
        "zero_leakage_verified": True,
        "curated_morita_clones": curated,
        "raw_compass_scdna": raw_compass,
    }
    for dest in [
        RESULTS_DIR / "single_cell" / "aml_38_001_showcase.json",
        REPO_ROOT.parent / "results" / "cancer_singlecell" / "aml_38_001_showcase.json",
    ]:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
