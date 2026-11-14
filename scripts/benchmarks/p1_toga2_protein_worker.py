#!/usr/bin/env python3
"""
Pipeline 1: 19,952-Protein x 882-Mammal Unsupervised Convergence & Horizontal Transfer Atlas
Runs 20-state Amino-Acid Bushwhacker (--translate) across all TOGA2 human-referenced
mammalian coding alignments in BOTH:
  (A) Unsupervised de novo protein graph induction (no species tree input)
  (B) Pruned mammalian species-tree MDL evaluation (-t speciesTree.nh)
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
BASE_DIR = str(WORK_ROOT / "toga2_aa")
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
            stack[-1].append({"children": children, "name": lbl})
        elif tok.startswith(":"):
            pass
        else:
            lbl = tok
            if i + 1 < n and tokens[i + 1].startswith(":"):
                i += 1
            stack[-1].append({"children": [], "name": lbl})
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
            if k < len(comp):
                canon = frozenset(sub)
            elif len(comp) < k:
                canon = frozenset(comp)
            else:
                canon = frozenset(sub) if min(sub) < min(comp) else frozenset(comp)
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


def compact_shunts(shunts, limit=24):
    out = []
    for s in shunts[:limit]:
        u_leaves = s.get("leaves_u", [])
        v_leaves = s.get("leaves_v", [])
        out.append({
            "label_u": s.get("label_u"),
            "label_v": s.get("label_v"),
            "clade_size_u": s.get("clade_size_u", len(u_leaves)),
            "clade_size_v": s.get("clade_size_v", len(v_leaves)),
            "leaves_u": u_leaves,
            "leaves_v": v_leaves,
            "mode": s.get("mode"),
            "shared_mutations": s.get("shared_mutations"),
            "net_mdl_gain_nats": round(s.get("net_mdl_gain_nats", 0.0), 3),
            "data_savings_nats": round(s.get("data_savings_nats", 0.0), 3),
            "mask_cost_nats": round(s.get("mask_cost_nats", 0.0), 3),
            "endpoint_cost_nats": round(s.get("endpoint_cost_nats", 0.0), 3),
            "posterior_prob": round(s.get("posterior_prob", 0.0), 4),
            "interval_span": s.get("interval_span"),
            "residues_1based": s.get("residues_1based", s.get("codons_1based", []))[:32],
        })
    return out


def process_one_gene(fa_path, stem, species_tree_root, env):
    t0 = time.perf_counter()
    shm_root = "/dev/shm" if os.path.isdir("/dev/shm") else None
    tmpdir = tempfile.mkdtemp(prefix=f"bwaa_{os.getpid()}_", dir=shm_root)
    try:
        out_nwk = os.path.join(tmpdir, "free_mdl.nwk")
        out_bin = os.path.join(tmpdir, "free_bin.nwk")
        out_json = os.path.join(tmpdir, "free.json")

        cmd = [
            BUSHWHACKER_BIN,
            "--translate",
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
            free_data = json.load(f)
        t_free = time.perf_counter() - t0

        mdl_tree = parse_newick(free_data["newick"])
        bin_tree = parse_newick(free_data["binary_newick"])
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

        sp_data = None
        t_sp = 0.0
        if pruned_sp and len(sp_splits) > 0:
            t1 = time.perf_counter()
            sp_nwk_path = os.path.join(tmpdir, "sp_pruned.nwk")
            sp_out_nwk = os.path.join(tmpdir, "sp_mdl.nwk")
            sp_out_json = os.path.join(tmpdir, "sp.json")
            with open(sp_nwk_path, "w") as f:
                f.write(to_newick(pruned_sp) + ";\n")
            cmd_sp = [
                BUSHWHACKER_BIN,
                "--translate",
                fa_path,
                "-t", sp_nwk_path,
                "-o", sp_out_nwk,
                "-j", sp_out_json,
            ]
            res_sp = subprocess.run(cmd_sp, capture_output=True, text=True, env=env, timeout=300)
            t_sp = time.perf_counter() - t1
            if res_sp.returncode == 0 and os.path.exists(sp_out_json):
                with open(sp_out_json, "r") as f:
                    sp_data = json.load(f)

        free_shunts = free_data.get("shunts", [])
        sp_shunts = sp_data.get("shunts", []) if sp_data else []

        gene_symbol = stem.split("#", 1)[1] if "#" in stem else stem
        summary = {
            "gene": stem,
            "symbol": gene_symbol,
            "status": "ok",
            "alphabet": "protein",
            "n_taxa": free_data.get("n_taxa", n_kept),
            "aa_len": free_data.get("seq_len"),
            "runtime_free_sec": round(t_free, 4),
            "runtime_sp_sec": round(t_sp, 4),
            "raw_dl_nats": free_data.get("raw_Description_Length_nats"),
            "star_mdl_nats": free_data.get("star_tree_MDL_nats"),
            "free_tree_mdl_nats": free_data.get("tree_MDL_nats"),
            "free_net_mdl_nats": free_data.get("network_MDL_nats"),
            "sp_tree_mdl_nats": sp_data.get("tree_MDL_nats") if sp_data else None,
            "sp_net_mdl_nats": sp_data.get("network_MDL_nats") if sp_data else None,
            "sp_splits_total": len(sp_splits),
            "sp_resolved_splits": sp_data.get("resolved_internal_splits") if sp_data else None,
            "sp_polytomy_nodes": sp_data.get("polytomy_nodes") if sp_data else None,
            "bin_splits": len(bin_splits),
            "bin_sp_shared": bin_shared,
            "bin_sp_conflict": bin_conflicting,
            "mdl_splits": len(mdl_splits),
            "mdl_polytomy_nodes": free_data.get("polytomy_nodes"),
            "mdl_sp_shared": mdl_shared,
            "mdl_sp_conflict": mdl_conflicting,
            "false_conflict_eliminated_frac": (
                round(1.0 - (mdl_conflicting / max(1, bin_conflicting)), 4)
                if bin_conflicting > 0 else 1.0
            ),
            "n_free_shunts": len(free_shunts),
            "n_free_interval": sum(1 for s in free_shunts if s.get("mode") == "INTERVAL"),
            "n_free_sparse": sum(1 for s in free_shunts if s.get("mode") == "SPARSE"),
            "n_sp_shunts": len(sp_shunts),
            "n_sp_interval": sum(1 for s in sp_shunts if s.get("mode") == "INTERVAL"),
            "n_sp_sparse": sum(1 for s in sp_shunts if s.get("mode") == "SPARSE"),
            "max_sp_shunt_gain_nats": round(max((s.get("net_mdl_gain_nats", 0.0) for s in sp_shunts), default=0.0), 3),
            "max_free_shunt_gain_nats": round(max((s.get("net_mdl_gain_nats", 0.0) for s in free_shunts), default=0.0), 3),
            "sp_shunts": compact_shunts(sp_shunts, limit=24),
            "free_shunts": compact_shunts(free_shunts, limit=16),
        }
        full_payload = {
            "summary": summary,
            "free_newick": free_data.get("newick"),
            "sp_newick": sp_data.get("newick") if sp_data else None,
        }
        return summary, full_payload
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
    start_idx = (vid * 37) % n_total
    summary_file = os.path.join(SUMMARIES_DIR, f"worker_{vid:04d}.jsonl")

    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = "2"

    processed = 0
    with open(summary_file, "a") as sf:
        for offset in range(n_total):
            idx = (start_idx + offset) % n_total
            fa_path = manifest[idx]
            stem = os.path.basename(fa_path).replace(".codonified.fa", "")
            res_gz = os.path.join(RESULTS_DIR, f"{stem}.bwaa.json.gz")
            lock_f = os.path.join(LOCKS_DIR, f"{stem}.lock")

            if os.path.exists(res_gz) and os.path.getsize(res_gz) > 80:
                continue

            try:
                fd = os.open(lock_f, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{vid}\n".encode())
                os.close(fd)
            except FileExistsError:
                continue

            summary, full_payload = process_one_gene(fa_path, stem, species_tree_root, env)
            if full_payload is not None:
                tmp_gz = res_gz + f".tmp.{vid}"
                with gzip.open(tmp_gz, "wt") as gz_out:
                    json.dump(full_payload, gz_out)
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
    print(f"[P1 TOGA2 Protein] Task {task_id}: starting {workers_per_task} workers ({vids[0]}..{vids[-1]})", flush=True)

    with ProcessPoolExecutor(max_workers=workers_per_task) as ex:
        futs = [ex.submit(worker_loop, vid, total_virtual_workers) for vid in vids]
        for fut in futs:
            vid, cnt = fut.result()
            print(f"  Worker {vid:04d} finished ({cnt} proteins processed)", flush=True)


if __name__ == "__main__":
    main()
