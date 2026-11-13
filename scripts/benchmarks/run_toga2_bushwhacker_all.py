#!/usr/bin/env python3
"""
Cluster-scale work-stealing orchestrator for running Bushwhacker across all 19,953
TOGA2 human-referenced (hg38 v1) mammalian coding alignments on silverback.

Usage:
  python3 run_toga2_bushwhacker_all.py <task_id> <workers_per_task> <total_virtual_workers>
"""


import sys as _sys
from pathlib import Path as _Path
if str(_Path(__file__).resolve().parent) not in _sys.path:
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
from _env_paths import REPO_ROOT, DATA_DIR, SPECIMENS_DIR, ALIGNMENTS_DIR, SINGLE_CELL_DIR, RESULTS_DIR, WORK_ROOT, EXTERNAL_DIR, resolve_bin, ASTRAL_JAR

import os
import re
import sys
import json
import gzip
import time
import shutil
import tempfile
import subprocess
from concurrent.futures import ProcessPoolExecutor

BUSHWHACKER_BIN = resolve_bin("BUSHWHACKER_BIN", REPO_ROOT / "target" / "release" / "bushwhacker", "bushwhacker")
SPECIES_TREE_PATH = "data/alignments/supermatrices/toga2_speciesTree.nh"
MANIFEST_PATH = str(EXTERNAL_DIR / "toga2_hg38_v1" / "manifest.txt")
WALRUS_MSAS_DIR = str(EXTERNAL_DIR / "toga2_hg38_v1" / "msas")
BASE_DIR = str(WORK_ROOT / "toga2_nt")
LOCKS_DIR = os.path.join(BASE_DIR, "locks")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
SUMMARIES_DIR = os.path.join(BASE_DIR, "summaries")


def parse_newick(nwk_str):
    s = nwk_str.strip()
    if s.endswith(";"):
        s = s[:-1]
    tokens = re.findall(r"\(|\)|,|;|:[^(),;]+|[^(),;:]+", s)
    stack = [[]]
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        if tok == "(":
            stack.append([])
        elif tok == ",":
            pass
        elif tok == ")":
            children = stack.pop()
            lbl = None
            if i + 1 < n and not tokens[i + 1].startswith(":") and tokens[i + 1] not in ("(", ")", ",", ";"):
                i += 1
                lbl = tokens[i]
            if i + 1 < n and tokens[i + 1].startswith(":"):
                i += 1
            node = {"children": children, "name": lbl}
            stack[-1].append(node)
        elif tok.startswith(":"):
            pass
        else:
            lbl = tok
            if i + 1 < n and tokens[i + 1].startswith(":"):
                i += 1
            node = {"children": [], "name": lbl}
            stack[-1].append(node)
        i += 1
    return stack[0][0] if stack[0] else {"children": [], "name": None}


def prune_tree(node, keep_tips):
    if not node["children"]:
        if node["name"] in keep_tips:
            return {"children": [], "name": node["name"]}
        return None
    kept_children = []
    for c in node["children"]:
        pc = prune_tree(c, keep_tips)
        if pc is not None:
            kept_children.append(pc)
    if not kept_children:
        return None
    if len(kept_children) == 1:
        return kept_children[0]
    return {"children": kept_children, "name": node.get("name")}


def to_newick(node):
    if not node["children"]:
        return f"{node['name']}:0.01"
    inner = ",".join(to_newick(c) for c in node["children"])
    return f"({inner}):0.01"


def get_nontrivial_splits(node, all_tips=None):
    if all_tips is None:
        all_tips = set()
        def collect(u):
            if not u["children"]:
                if u["name"]:
                    all_tips.add(u["name"])
            else:
                for c in u["children"]:
                    collect(c)
        collect(node)
    total = len(all_tips)
    splits = set()
    if total < 4:
        return splits, all_tips

    def dfs(u):
        if not u["children"]:
            return {u["name"]} if u["name"] in all_tips else set()
        sub = set()
        for c in u["children"]:
            sub |= dfs(c)
        k = len(sub)
        if 2 <= k <= total - 2:
            comp = all_tips - sub
            if len(sub) < len(comp):
                canon = frozenset(sub)
            elif len(comp) < len(sub):
                canon = frozenset(comp)
            else:
                s1 = min(sub)
                s2 = min(comp)
                canon = frozenset(sub) if s1 < s2 else frozenset(comp)
            splits.add(canon)
        return sub

    dfs(node)
    return splits, all_tips


def splits_compatible(s1, s2, all_tips):
    if not (s1 & s2):
        return True
    c1 = all_tips - s1
    if not (c1 & s2):
        return True
    c2 = all_tips - s2
    if not (s1 & c2):
        return True
    if not (c1 & c2):
        return True
    return False


def format_shunts(shunts, limit=8):
    records = []
    for s in shunts[:limit]:
        u_leaves = s.get("leaves_u", [])
        v_leaves = s.get("leaves_v", [])
        records.append({
            "label_u": s.get("label_u"),
            "label_v": s.get("label_v"),
            "u_tips_sample": u_leaves[:5],
            "u_clade_size": len(u_leaves),
            "v_tips_sample": v_leaves[:5],
            "v_clade_size": len(v_leaves),
            "mode": s.get("mode"),
            "shared_mutations": s.get("shared_mutations"),
            "net_mdl_gain_nats": round(s.get("net_mdl_gain_nats", 0.0), 2),
            "data_savings_nats": round(s.get("data_savings_nats", 0.0), 2),
            "interval_span": s.get("interval_span"),
            "codons_1based": s.get("codons_1based", [])[:18],
        })
    return records


def process_one_gene(fa_path, stem, species_tree_root, env):
    t0 = time.perf_counter()
    tmpdir = tempfile.mkdtemp(prefix=f"bw_{os.getpid()}_", dir="/dev/shm")
    try:
        out_nwk = os.path.join(tmpdir, "bw_mdl.nwk")
        out_bin = os.path.join(tmpdir, "bw_bin.nwk")
        out_json = os.path.join(tmpdir, "bw_mdl.json")

        cmd = [
            BUSHWHACKER_BIN,
            fa_path,
            "-o", out_nwk,
            "-b", out_bin,
            "-j", out_json,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
        if res.returncode != 0 or not os.path.exists(out_json):
            return {
                "gene": stem,
                "status": "error",
                "error": (res.stderr[-300:] if res.stderr else f"exit_{res.returncode}"),
            }, None

        with open(out_json, "r") as f:
            bw_data = json.load(f)

        t_denovo = time.perf_counter() - t0

        mdl_tree = parse_newick(bw_data["newick"])
        bin_tree = parse_newick(bw_data["binary_newick"])
        mdl_splits, kept_tips = get_nontrivial_splits(mdl_tree)
        bin_splits, _ = get_nontrivial_splits(bin_tree, kept_tips)

        n_kept = len(kept_tips)
        pruned_sp = prune_tree(species_tree_root, kept_tips) if n_kept >= 4 else None
        sp_splits, _ = get_nontrivial_splits(pruned_sp, kept_tips) if pruned_sp else (set(), kept_tips)

        bin_shared = len(bin_splits & sp_splits)
        bin_only = bin_splits - sp_splits
        bin_conflicting = sum(
            1 for s in bin_only if any(not splits_compatible(s, sp_s, kept_tips) for sp_s in sp_splits)
        )

        mdl_shared = len(mdl_splits & sp_splits)
        mdl_only = mdl_splits - sp_splits
        mdl_conflicting = sum(
            1 for s in mdl_only if any(not splits_compatible(s, sp_s, kept_tips) for sp_s in sp_splits)
        )

        # Evaluate pruned speciesTree directly under Bushwhacker MDL (--contract-tree)
        sp_tree_mdl_nats = None
        sp_net_mdl_nats = None
        sp_supported_splits = None
        sp_shunts_count = None
        sp_top_shunts = []
        if pruned_sp and len(sp_splits) > 0:
            sp_nwk_path = os.path.join(tmpdir, "sp_pruned.nwk")
            sp_out_nwk = os.path.join(tmpdir, "sp_contracted.nwk")
            sp_out_json = os.path.join(tmpdir, "sp_contracted.json")
            with open(sp_nwk_path, "w") as f:
                f.write(to_newick(pruned_sp) + ";\n")
            cmd_sp = [
                BUSHWHACKER_BIN,
                fa_path,
                "-t", sp_nwk_path,
                "-o", sp_out_nwk,
                "-j", sp_out_json,
            ]
            res_sp = subprocess.run(cmd_sp, capture_output=True, text=True, env=env, timeout=300)
            if res_sp.returncode == 0 and os.path.exists(sp_out_json):
                with open(sp_out_json, "r") as f:
                    sp_data = json.load(f)
                sp_tree_mdl_nats = sp_data.get("tree_MDL_nats")
                sp_net_mdl_nats = sp_data.get("network_MDL_nats")
                sp_supported_splits = sp_data.get("resolved_internal_splits")
                sp_shunts_raw = sp_data.get("shunts", [])
                sp_shunts_count = len(sp_shunts_raw)
                sp_top_shunts = format_shunts(sp_shunts_raw, limit=5)

        shunts = bw_data.get("shunts", [])
        interval_shunts = [s for s in shunts if s.get("mode") == "INTERVAL"]
        sparse_shunts = [s for s in shunts if s.get("mode") == "SPARSE"]

        # Optional comparison with WALRUS de novo MSA if present
        walrus_comparison = None
        w_msa_path = os.path.join(WALRUS_MSAS_DIR, f"{stem}.msa.fa.gz")
        if os.path.exists(w_msa_path) and os.path.getsize(w_msa_path) > 100:
            w_fa = os.path.join(tmpdir, "walrus_msa.fa")
            with gzip.open(w_msa_path, "rt", errors="replace") as fin, open(w_fa, "w") as fout:
                shutil.copyfileobj(fin, fout)
            w_nwk = os.path.join(tmpdir, "w_mdl.nwk")
            w_bin = os.path.join(tmpdir, "w_bin.nwk")
            w_json = os.path.join(tmpdir, "w_mdl.json")
            res_w = subprocess.run(
                [BUSHWHACKER_BIN, w_fa, "-o", w_nwk, "-b", w_bin, "-j", w_json],
                capture_output=True, text=True, env=env, timeout=300
            )
            if res_w.returncode == 0 and os.path.exists(w_json):
                with open(w_json, "r") as f:
                    w_data = json.load(f)
                w_tree = parse_newick(w_data["newick"])
                w_splits, w_tips = get_nontrivial_splits(w_tree)
                w_sp = prune_tree(species_tree_root, w_tips) if len(w_tips) >= 4 else None
                w_sp_splits, _ = get_nontrivial_splits(w_sp, w_tips) if w_sp else (set(), w_tips)
                w_shared = len(w_splits & w_sp_splits)
                w_only = w_splits - w_sp_splits
                w_conflict = sum(
                    1 for s in w_only if any(not splits_compatible(s, sp_s, w_tips) for sp_s in w_sp_splits)
                )
                w_shunts = w_data.get("shunts", [])
                walrus_comparison = {
                    "n_taxa": len(w_tips),
                    "seq_len": w_data.get("seq_len"),
                    "tree_MDL_nats": round(w_data.get("tree_MDL_nats", 0.0), 1),
                    "network_MDL_nats": round(w_data.get("network_MDL_nats", 0.0), 1),
                    "resolved_splits": len(w_splits),
                    "sp_shared_splits": w_shared,
                    "sp_conflict_splits": w_conflict,
                    "sp_conflict_rate": round(w_conflict / max(1, len(w_splits)), 4),
                    "n_interval_shunts": sum(1 for s in w_shunts if s.get("mode") == "INTERVAL"),
                    "n_sparse_shunts": sum(1 for s in w_shunts if s.get("mode") == "SPARSE"),
                }

        tree_mdl = bw_data.get("tree_MDL_nats", 0.0)
        gene_sym = stem.split("#")[-1] if "#" in stem else stem
        summary = {
            "status": "ok",
            "gene": stem,
            "gene_symbol": gene_sym,
            "n_taxa": n_kept,
            "seq_len": bw_data.get("seq_len"),
            "runtime_sec": round(t_denovo, 4),
            "raw_dl_nats": round(bw_data.get("raw_Description_Length_nats", 0.0), 1),
            "star_tree_MDL_nats": round(bw_data.get("star_tree_MDL_nats", 0.0), 1),
            "tree_MDL_nats": round(tree_mdl, 1),
            "network_MDL_nats": round(bw_data.get("network_MDL_nats", 0.0), 1),
            "sp_tree_MDL_nats": round(sp_tree_mdl_nats, 1) if sp_tree_mdl_nats is not None else None,
            "sp_net_MDL_nats": round(sp_net_mdl_nats, 1) if sp_net_mdl_nats is not None else None,
            "delta_mdl_gene_vs_sp_nats": (
                round(sp_tree_mdl_nats - tree_mdl, 2)
                if sp_tree_mdl_nats is not None
                else None
            ),
            "max_binary_splits": max(0, n_kept - 3),
            "sp_splits_count": len(sp_splits),
            "sp_supported_splits_count": sp_supported_splits,
            "sp_supported_frac": (
                round(sp_supported_splits / max(1, len(sp_splits)), 4)
                if sp_supported_splits is not None
                else None
            ),
            "sp_shunts_count": sp_shunts_count,
            "bin_splits_count": len(bin_splits),
            "bin_sp_shared": bin_shared,
            "bin_sp_conflict": bin_conflicting,
            "bin_sp_conflict_rate": round(bin_conflicting / max(1, len(bin_splits)), 4),
            "mdl_splits_count": len(mdl_splits),
            "polytomy_nodes": bw_data.get("polytomy_nodes"),
            "mdl_polytomy_contracted_frac": round(1.0 - len(mdl_splits) / max(1, len(bin_splits)), 4),
            "mdl_sp_shared": mdl_shared,
            "mdl_sp_conflict": mdl_conflicting,
            "mdl_sp_conflict_rate": round(mdl_conflicting / max(1, len(mdl_splits)), 4),
            "false_conflict_eliminated_frac": (
                round(1.0 - (mdl_conflicting / max(1, bin_conflicting)), 4)
                if bin_conflicting > 0 else 1.0
            ),
            "n_shunts": len(shunts),
            "n_interval_shunts": len(interval_shunts),
            "n_sparse_shunts": len(sparse_shunts),
            "max_shunt_gain_nats": round(max((s.get("net_mdl_gain_nats", 0.0) for s in shunts), default=0.0), 2),
            "top_shunts": format_shunts(shunts, limit=8),
            "sp_top_shunts": sp_top_shunts,
            "walrus_comparison": walrus_comparison,
        }
        return summary, bw_data
    except Exception as e:
        return {"gene": stem, "status": "error", "error": str(e)}, None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def worker_loop(vid, total_virtual_workers):
    with open(SPECIES_TREE_PATH, "r") as f:
        species_tree_root = parse_newick(f.read())
    with open(MANIFEST_PATH, "r") as f:
        manifest = [line.strip() for line in f if line.strip()]

    n_total = len(manifest)
    start_idx = (vid * 31) % n_total
    summary_file = os.path.join(SUMMARIES_DIR, f"worker_{vid:04d}.jsonl")

    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = "2"

    processed = 0
    with open(summary_file, "a") as sf:
        for offset in range(n_total):
            idx = (start_idx + offset) % n_total
            fa_path = manifest[idx]
            stem = os.path.basename(fa_path).replace(".codonified.fa", "")
            res_gz = os.path.join(RESULTS_DIR, f"{stem}.bw.json.gz")
            lock_f = os.path.join(LOCKS_DIR, f"{stem}.lock")

            if os.path.exists(res_gz) and os.path.getsize(res_gz) > 100:
                continue

            try:
                fd = os.open(lock_f, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{vid}\n".encode())
                os.close(fd)
            except FileExistsError:
                continue

            summary, full_json = process_one_gene(fa_path, stem, species_tree_root, env)
            if full_json is not None:
                tmp_gz = res_gz + f".tmp.{vid}"
                with gzip.open(tmp_gz, "wt") as gz_out:
                    json.dump(full_json, gz_out)
                os.replace(tmp_gz, res_gz)

            sf.write(json.dumps(summary) + "\n")
            sf.flush()
            processed += 1

    return vid, processed


def main():
    task_id = int(sys.argv[1])
    workers_per_task = int(sys.argv[2])
    total_virtual_workers = int(sys.argv[3])

    os.makedirs(LOCKS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(SUMMARIES_DIR, exist_ok=True)

    vids = [task_id * workers_per_task + j for j in range(workers_per_task)]
    print(f"Task {task_id}: starting {workers_per_task} workers (vids {vids[0]}..{vids[-1]})", flush=True)

    with ProcessPoolExecutor(max_workers=workers_per_task) as ex:
        futs = [ex.submit(worker_loop, vid, total_virtual_workers) for vid in vids]
        for fut in futs:
            vid, cnt = fut.result()
            print(f"  Worker {vid:04d} finished ({cnt} alignments processed)", flush=True)


if __name__ == "__main__":
    main()
