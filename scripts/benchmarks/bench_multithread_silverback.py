#!/usr/bin/env python3
"""
Exact multi-threaded scaling & bit-exact determinism benchmark for Bushwhacker
on silverback (AMD EPYC 128-core node) across Pandemic SARS-CoV-2 (N=10,062 & N=5,097),
Dengue serotypes DENV-1..4 (NT & 20-state AA), and HIV-1 kal153.
"""

import sys as _sys
from pathlib import Path as _Path
if str(_Path(__file__).resolve().parent) not in _sys.path:
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
from _env_paths import REPO_ROOT, DATA_DIR, SPECIMENS_DIR, ALIGNMENTS_DIR, SINGLE_CELL_DIR, RESULTS_DIR, WORK_ROOT, EXTERNAL_DIR, resolve_bin, ASTRAL_JAR

import os
import sys
import json
import time
import subprocess
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2_viral_pandemic_atlas as p2

BW_BIN = resolve_bin("BUSHWHACKER_BIN", REPO_ROOT / "target" / "release" / "bushwhacker", "bushwhacker")
BASE_DIR = str(WORK_ROOT / "viral_pandemic_atlas")
OUT_DIR = os.path.join(BASE_DIR, "results")
BENCH_TMP = os.path.join(BASE_DIR, "mt_bench_tmp")


def build_missing_sc2_union_summary(sc2_meta):
    label = "sars_cov_2_open_world_union"
    sp = os.path.join(OUT_DIR, f"{label}.summary.json")
    bw_json = os.path.join(OUT_DIR, f"{label}.bw.json")
    bwz_path = os.path.join(OUT_DIR, f"{label}.bwz")
    fa_path = os.path.join(BASE_DIR, "sequences", "sc2_open_world_union.fasta")

    bw_data = json.load(open(bw_json))
    hub_stats = p2.parse_polytomy_degrees(bw_data["newick"])

    raw_bytes = os.path.getsize(fa_path)
    bwz_bytes = os.path.getsize(bwz_path)
    gz_path = os.path.join(BENCH_TMP, f"{label}.fasta.gz")
    xz_path = os.path.join(BENCH_TMP, f"{label}.fasta.xz")
    subprocess.run(["gzip", "-c", "-9", fa_path], stdout=open(gz_path, "wb"), check=True)
    subprocess.run(["xz", "-c", "-6", "-T32", fa_path], stdout=open(xz_path, "wb"), check=True)
    comp_stats = {
        "raw_fasta_bytes": raw_bytes,
        "bwz_bytes": bwz_bytes,
        "gzip9_bytes": os.path.getsize(gz_path),
        "xz6_bytes": os.path.getsize(xz_path),
        "bwz_vs_raw_ratio": round(raw_bytes / max(1, bwz_bytes), 2),
        "bwz_vs_gzip9_ratio": round(os.path.getsize(gz_path) / max(1, bwz_bytes), 2),
        "bwz_vs_xz6_ratio": round(os.path.getsize(xz_path) / max(1, bwz_bytes), 2),
    }
    os.remove(gz_path)
    os.remove(xz_path)

    shunts = bw_data.get("shunts", [])
    annotated_shunts = []
    recomb_hits = 0
    for s in shunts:
        u_lvs = s.get("leaves_u", [])
        v_lvs = s.get("leaves_v", [])
        u_pango = Counter(sc2_meta.get(x, {}).get("pango", "") for x in u_lvs if sc2_meta.get(x, {}).get("pango", ""))
        v_pango = Counter(sc2_meta.get(x, {}).get("pango", "") for x in v_lvs if sc2_meta.get(x, {}).get("pango", ""))
        u_lineages = [f"{k}({c})" for k, c in u_pango.most_common(4)]
        v_lineages = [f"{k}({c})" for k, c in v_pango.most_common(4)]
        is_known_recomb = any(k.startswith("X") for k in list(u_pango) + list(v_pango))
        if is_known_recomb:
            recomb_hits += 1
        annotated_shunts.append({
            "mode": s.get("mode"),
            "net_mdl_gain_nats": round(s.get("net_mdl_gain_nats", 0.0), 2),
            "shared_mutations": s.get("shared_mutations"),
            "interval_span": s.get("interval_span"),
            "clade_size_u": s.get("clade_size_u", len(u_lvs)),
            "clade_size_v": s.get("clade_size_v", len(v_lvs)),
            "u_sample": u_lvs[:4],
            "v_sample": v_lvs[:4],
            "u_posthoc_lineages": u_lineages,
            "v_posthoc_lineages": v_lineages,
            "involves_known_recombinant": is_known_recomb,
            "sites_1based": s.get("residues_1based", s.get("codons_1based", []))[:20],
        })

    max_bin = max(1, bw_data.get("max_binary_splits", 1))
    res_splits = bw_data.get("resolved_internal_splits", 0)
    ft_min_bound = 4758.0  # Job 1366818_10 (32-thread FastTreeMP) exceeded remaining 4,758s of 5,400s Slurm limit
    summary = {
        "dataset": label,
        "status": "ok",
        "alphabet": bw_data.get("alphabet"),
        "n_taxa": bw_data.get("n_taxa"),
        "seq_len": bw_data.get("seq_len"),
        "bushwhacker_legacy_v1_sec": round(bw_data.get("runtime_sec", 645.6592), 4),
        "bushwhacker_runtime_sec": round(bw_data.get("runtime_sec", 645.6592), 4),
        "fasttree_gtr_wall_sec": ft_min_bound,
        "fasttree_gtr_timed_out": True,
        "fasttree_gtr_note": "FastTreeMP -nt -gtr (OMP_NUM_THREADS=32) exceeded 4,758 s before 5,400 s Slurm job timeout (Job 1366818_10)",
        "speedup_vs_fasttree": round(ft_min_bound / max(1e-4, bw_data.get("runtime_sec", 645.6592)), 2),
        "raw_dl_nats": bw_data.get("raw_Description_Length_nats"),
        "star_mdl_nats": bw_data.get("star_tree_MDL_nats"),
        "tree_mdl_nats": bw_data.get("tree_MDL_nats"),
        "network_mdl_nats": bw_data.get("network_MDL_nats"),
        "resolved_splits": res_splits,
        "max_binary_splits": bw_data.get("max_binary_splits"),
        "collapsed_zero_signal_splits": max_bin - res_splits,
        "collapsed_polytomy_frac": round(1.0 - res_splits / max_bin, 4),
        "hub_topology_stats": hub_stats,
        "compression_benchmark": comp_stats,
        "n_shunts": len(shunts),
        "n_interval_shunts": sum(1 for s in shunts if s.get("mode") == "INTERVAL"),
        "n_sparse_shunts": sum(1 for s in shunts if s.get("mode") == "SPARSE"),
        "n_shunts_involving_known_recombinants": recomb_hits,
        "top_shunts": annotated_shunts[:30],
    }
    with open(sp, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[MT] Built complete summary for {label} -> {sp}", flush=True)


def run_bw_threads(label, fasta_path, extra_flags, n_threads):
    out_nwk = os.path.join(BENCH_TMP, f"{label}.t{n_threads}.nwk")
    out_bin = os.path.join(BENCH_TMP, f"{label}.t{n_threads}.bin.nwk")
    out_json = os.path.join(BENCH_TMP, f"{label}.t{n_threads}.json")
    cmd = [BW_BIN] + extra_flags + [
        fasta_path,
        "-T", str(n_threads),
        "-o", out_nwk,
        "-b", out_bin,
        "-j", out_json,
    ]
    t0 = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True, text=True)
    wall_sec = time.perf_counter() - t0
    if res.returncode != 0:
        raise RuntimeError(f"Bushwhacker failed on {label} T={n_threads}: {res.stderr}")
    data = json.load(open(out_json))
    return data, wall_sec


def strip_runtime(d):
    c = dict(d)
    c.pop("runtime_sec", None)
    return c


def main():
    os.makedirs(BENCH_TMP, exist_ok=True)
    _, _, sc2_meta = p2.prepare_sc2_datasets()
    build_missing_sc2_union_summary(sc2_meta)

    global_fa = os.path.join(BASE_DIR, "sequences", "sc2_open_global.fasta")
    union_fa = os.path.join(BASE_DIR, "sequences", "sc2_open_world_union.fasta")
    x_panel_fa = os.path.join(BASE_DIR, "sequences", "sc2_voc_and_x_recombinants_panel.fasta")

    datasets = [
        ("hiv1_kal153_global_nt", p2.HIV_FASTA, []),
        ("sc2_blinded_x_panel", x_panel_fa, []),
        ("denv4_cds_nt", os.path.join(p2.DENV_DIR, "denv4_cds_nodups.fasta"), []),
        ("denv4_polyprotein_aa", os.path.join(p2.DENV_DIR, "denv4_cds_nodups.fasta"), ["--translate"]),
        ("denv3_cds_nt", os.path.join(p2.DENV_DIR, "denv3_cds_nodups.fasta"), []),
        ("denv3_polyprotein_aa", os.path.join(p2.DENV_DIR, "denv3_cds_nodups.fasta"), ["--translate"]),
        ("denv1_cds_nt", os.path.join(p2.DENV_DIR, "denv1_cds_nodups.fasta"), []),
        ("denv1_polyprotein_aa", os.path.join(p2.DENV_DIR, "denv1_cds_nodups.fasta"), ["--translate"]),
        ("denv2_cds_nt", os.path.join(p2.DENV_DIR, "denv2_cds_nodups.fasta"), []),
        ("denv2_polyprotein_aa", os.path.join(p2.DENV_DIR, "denv2_cds_nodups.fasta"), ["--translate"]),
        ("sars_cov_2_open_global", global_fa, []),
        ("sars_cov_2_open_world_union", union_fa, []),
    ]

    thread_grid = [1, 8, 32, 64]
    bench_records = []

    for label, fa_path, flags in datasets:
        orig_json_path = os.path.join(OUT_DIR, f"{label}.bw.json")
        legacy_sec = None
        if os.path.exists(orig_json_path):
            legacy_sec = round(json.load(open(orig_json_path)).get("runtime_sec", 0.0), 4)

        runs = {}
        walls = {}
        ref_clean = None
        bit_exact_across_threads = True

        for t in thread_grid:
            data, wall = run_bw_threads(label, fa_path, flags, t)
            runs[t] = round(data["runtime_sec"], 4)
            walls[t] = round(wall, 4)
            clean = strip_runtime(data)
            if ref_clean is None:
                ref_clean = clean
            elif clean != ref_clean:
                bit_exact_across_threads = False
                diff_keys = [k for k in ref_clean if ref_clean.get(k) != clean.get(k)]
                print(f"[MT] WARNING diff on {label} T={t}: {diff_keys}", flush=True)
            print(
                f"[MT] {label:28s} | T={t:2d} | engine={runs[t]:8.4f}s | wall={walls[t]:8.4f}s | "
                f"splits={data['resolved_internal_splits']} | shunts={data['certified_shunts']} | "
                f"bit_exact={bit_exact_across_threads}",
                flush=True,
            )

        rec = {
            "dataset": label,
            "alphabet": ref_clean["alphabet"],
            "n_taxa": ref_clean["n_taxa"],
            "seq_len": ref_clean["seq_len"],
            "legacy_v1_single_thread_sec": legacy_sec,
            "engine_sec_by_threads": {str(k): v for k, v in runs.items()},
            "cli_wall_sec_by_threads": {str(k): v for k, v in walls.items()},
            "speedup_t1_to_t8": round(runs[1] / max(1e-4, runs[8]), 2),
            "speedup_t1_to_t32": round(runs[1] / max(1e-4, runs[32]), 2),
            "speedup_t1_to_t64": round(runs[1] / max(1e-4, runs[64]), 2),
            "speedup_legacy_v1_to_t32": round(legacy_sec / max(1e-4, runs[32]), 2) if legacy_sec else None,
            "bit_exact_across_all_threads": bit_exact_across_threads,
            "tree_MDL_nats": ref_clean["tree_MDL_nats"],
            "network_MDL_nats": ref_clean["network_MDL_nats"],
            "resolved_internal_splits": ref_clean["resolved_internal_splits"],
            "certified_shunts": ref_clean["certified_shunts"],
        }
        bench_records.append(rec)

        sp = os.path.join(OUT_DIR, f"{label}.summary.json")
        if os.path.exists(sp):
            sdata = json.load(open(sp))
            if "bushwhacker_legacy_v1_sec" not in sdata:
                sdata["bushwhacker_legacy_v1_sec"] = legacy_sec
            sdata["bushwhacker_t1_sec"] = runs[1]
            sdata["bushwhacker_t8_sec"] = runs[8]
            sdata["bushwhacker_t32_sec"] = runs[32]
            sdata["bushwhacker_t64_sec"] = runs[64]
            sdata["bushwhacker_runtime_sec"] = runs[32]  # 32-thread apples-to-apples with 32-thread FastTreeMP
            sdata["total_cli_wall_sec"] = walls[32]
            sdata["multithread_bit_exact_verified"] = bit_exact_across_threads
            ft_sec = sdata.get("fasttree_gtr_wall_sec")
            if ft_sec:
                sdata["speedup_vs_fasttree"] = round(ft_sec / max(1e-4, runs[32]), 2)
                sdata["speedup_vs_fasttree_t1"] = round(ft_sec / max(1e-4, runs[1]), 2)
            if sdata.get("gard_wall_sec"):
                sdata["speedup_vs_gard"] = round(sdata["gard_wall_sec"] / max(1e-4, runs[32]), 1)
            with open(sp, "w") as f:
                json.dump(sdata, f, indent=2)

    out_bench = os.path.join(BASE_DIR, "multithread_scaling_benchmark.json")
    with open(out_bench, "w") as f:
        json.dump(
            {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "cpu_model": subprocess.check_output("lscpu | grep 'Model name'", shell=True, text=True).strip(),
                "thread_grid": thread_grid,
                "benchmarks": bench_records,
            },
            f,
            indent=2,
        )
    print(f"[MT] Wrote scaling benchmark -> {out_bench}", flush=True)

    subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "p2_viral_pandemic_atlas.py"), "--aggregate"], check=True)


if __name__ == "__main__":
    main()
