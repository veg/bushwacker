#!/usr/bin/env python3
"""
Genome-scale TOGA2 mammalian coding alignment analysis with Bushwhacker:
1. Gene tree vs. Species tree (speciesTree.nh) concordance & MDL calibration
   - Binary vs. Calibrated (G_MDL > 0) topological conflict against speciesTree.nh
   - Direct MDL evaluation of pruned speciesTree.nh via `bushwhacker -t`
2. Ectopic gene conversion & projection chimera discovery (`INTERVAL` shunts)
3. Adaptive functional convergence discovery (`SPARSE` shunts)
4. Reference-projected (.codonified.fa) vs. de novo (walrus .msa.fa.gz) MDL comparison
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
import json
import gzip
import time
import shutil
import tempfile
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

BUSHWHACKER_BIN = resolve_bin("BUSHWHACKER_BIN", REPO_ROOT / "target" / "release" / "bushwhacker", "bushwhacker")
SPECIES_TREE_PATH = "data/alignments/supermatrices/toga2_speciesTree.nh"
MANIFEST_PATH = str(EXTERNAL_DIR / "toga2_hg38_v1" / "manifest.txt")
GRANT_DIR = str(WORK_ROOT / "toga2_grant_examples")
WALRUS_MSAS_DIR = str(EXTERNAL_DIR / "toga2_hg38_v1" / "msas")
OUT_DIR = str(WORK_ROOT / "toga2_nt")


def parse_newick(nwk_str):
    """Parse Newick string into a tree dict: children list, name."""
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
    """Prune a parsed Newick tree to keep_tips, collapsing degree-1 internal nodes."""
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
    """Return set of canonical frozenset bipartitions."""
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
    """Buneman pairwise split compatibility: at least one of the 4 intersections is empty."""
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


SPECIES_TREE_ROOT = None

def init_worker(sp_tree_path):
    global SPECIES_TREE_ROOT
    with open(sp_tree_path, "r") as f:
        SPECIES_TREE_ROOT = parse_newick(f.read())


def analyze_single_alignment(args):
    fa_path, gene_label, compare_walrus_msa = args
    global SPECIES_TREE_ROOT
    t0 = time.perf_counter()
    tmpdir = tempfile.mkdtemp(prefix="bw_toga2_", dir="/dev/shm")
    try:
        if fa_path.endswith(".gz"):
            local_fa = os.path.join(tmpdir, "input.fa")
            with gzip.open(fa_path, "rt", errors="replace") as fin, open(local_fa, "w") as fout:
                shutil.copyfileobj(fin, fout)
        else:
            local_fa = fa_path

        out_nwk = os.path.join(tmpdir, "bw_mdl.nwk")
        out_bin = os.path.join(tmpdir, "bw_bin.nwk")
        out_json = os.path.join(tmpdir, "bw_mdl.json")
        env = os.environ.copy()
        env["RAYON_NUM_THREADS"] = "2"
        cmd = [
            BUSHWHACKER_BIN,
            local_fa,
            "-o", out_nwk,
            "-b", out_bin,
            "-j", out_json,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=180)
        if res.returncode != 0 or not os.path.exists(out_json):
            return {"gene": gene_label, "error": res.stderr[-300:] if res.stderr else "failed"}

        with open(out_json, "r") as f:
            bw_data = json.load(f)

        t_denovo = time.perf_counter() - t0

        mdl_tree = parse_newick(bw_data["newick"])
        bin_tree = parse_newick(bw_data["binary_newick"])
        mdl_splits, kept_tips = get_nontrivial_splits(mdl_tree)
        bin_splits, _ = get_nontrivial_splits(bin_tree, kept_tips)

        n_kept = len(kept_tips)
        if n_kept < 4:
            return None

        # Prune speciesTree.nh to kept_tips
        pruned_sp = prune_tree(SPECIES_TREE_ROOT, kept_tips)
        sp_splits, _ = get_nontrivial_splits(pruned_sp, kept_tips) if pruned_sp else (set(), kept_tips)

        # Binary gene tree vs species tree
        bin_shared = len(bin_splits & sp_splits)
        bin_only = bin_splits - sp_splits
        bin_conflicting = sum(
            1 for s in bin_only if any(not splits_compatible(s, sp_s, kept_tips) for sp_s in sp_splits)
        )

        # MDL-calibrated gene tree vs species tree
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
        if pruned_sp and len(sp_splits) > 0:
            sp_nwk_path = os.path.join(tmpdir, "sp_pruned.nwk")
            sp_out_nwk = os.path.join(tmpdir, "sp_contracted.nwk")
            sp_out_json = os.path.join(tmpdir, "sp_contracted.json")
            with open(sp_nwk_path, "w") as f:
                f.write(to_newick(pruned_sp) + ";\n")
            cmd_sp = [
                BUSHWHACKER_BIN,
                local_fa,
                "-t", sp_nwk_path,
                "-o", sp_out_nwk,
                "-j", sp_out_json,
            ]
            res_sp = subprocess.run(cmd_sp, capture_output=True, text=True, env=env, timeout=180)
            if res_sp.returncode == 0 and os.path.exists(sp_out_json):
                with open(sp_out_json, "r") as f:
                    sp_data = json.load(f)
                sp_tree_mdl_nats = sp_data.get("tree_MDL_nats")
                sp_net_mdl_nats = sp_data.get("network_MDL_nats")
                sp_supported_splits = sp_data.get("resolved_internal_splits")
                sp_shunts_count = sp_data.get("certified_shunts")

        shunts = bw_data.get("shunts", [])
        interval_shunts = [s for s in shunts if s.get("mode") == "INTERVAL"]
        sparse_shunts = [s for s in shunts if s.get("mode") == "SPARSE"]

        # Optional comparison with WALRUS de novo MSA
        walrus_comparison = None
        if compare_walrus_msa and os.path.exists(compare_walrus_msa):
            w_fa = os.path.join(tmpdir, "walrus_msa.fa")
            with gzip.open(compare_walrus_msa, "rt", errors="replace") as fin, open(w_fa, "w") as fout:
                shutil.copyfileobj(fin, fout)
            w_nwk = os.path.join(tmpdir, "w_mdl.nwk")
            w_bin = os.path.join(tmpdir, "w_bin.nwk")
            w_json = os.path.join(tmpdir, "w_mdl.json")
            res_w = subprocess.run(
                [BUSHWHACKER_BIN, w_fa, "-o", w_nwk, "-b", w_bin, "-j", w_json],
                capture_output=True, text=True, env=env, timeout=180
            )
            if res_w.returncode == 0 and os.path.exists(w_json):
                with open(w_json, "r") as f:
                    w_data = json.load(f)
                w_tree = parse_newick(w_data["newick"])
                w_splits, w_tips = get_nontrivial_splits(w_tree)
                w_sp = prune_tree(SPECIES_TREE_ROOT, w_tips)
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

        top_shunt_records = []
        for s in shunts[:6]:
            u_leaves = s.get("leaves_u", [])
            v_leaves = s.get("leaves_v", [])
            top_shunt_records.append({
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

        tree_mdl = bw_data.get("tree_MDL_nats", 0.0)
        return {
            "gene": gene_label,
            "n_taxa": n_kept,
            "seq_len": bw_data.get("seq_len"),
            "runtime_sec": round(t_denovo, 4),
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
            "max_binary_splits": n_kept - 3,
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
            "top_shunts": top_shunt_records,
            "walrus_comparison": walrus_comparison,
        }
    except Exception as e:
        return {"gene": gene_label, "error": str(e)}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Analyze the 10 curated grant examples first
    grant_files = sorted(glob.glob(os.path.join(GRANT_DIR, "*.codonified.fa")))
    print(f"=== Part 1: Analyzing {len(grant_files)} TOGA2 Grant Examples ===", flush=True)
    init_worker(SPECIES_TREE_PATH)

    grant_results = []
    for gf in grant_files:
        gene_sym = os.path.basename(gf).replace(".codonified.fa", "")
        r = analyze_single_alignment((gf, gene_sym, None))
        if r and "error" not in r:
            grant_results.append(r)
            print(
                f"  {gene_sym:10s} | N={r.get('n_taxa',0):3d}, L={r.get('seq_len',0):4d} | "
                f"Splits: {r.get('mdl_splits_count',0):3d}/{r.get('bin_splits_count',0):3d} "
                f"(SpTree sup: {r.get('sp_supported_splits_count',0)}/{r.get('sp_splits_count',0)}) | "
                f"Conflict: Bin={r.get('bin_sp_conflict',0):3d} -> MDL={r.get('mdl_sp_conflict',0):3d} "
                f"(-{100.0*r.get('false_conflict_eliminated_frac',0):.1f}%) | "
                f"Shunts: {r.get('n_interval_shunts',0)} INT, {r.get('n_sparse_shunts',0)} SPARSE "
                f"(max +{r.get('max_shunt_gain_nats',0):.1f}n)",
                flush=True
            )

    with open(os.path.join(OUT_DIR, "grant_examples_summary.json"), "w") as f:
        json.dump(grant_results, f, indent=2)

    # 2. Build a 300-gene genome-wide stratified + candidate sample from TOGA2 manifest.txt
    with open(MANIFEST_PATH, "r") as f:
        all_manifest = [line.strip() for line in f if line.strip()]

    priority_symbols = {
        "SLC26A5", "TMC1", "CDH23", "PCDH15", "OTOF", "KCNQ4", "DFNB59",
        "MB", "HBA1", "HBA2", "HBB", "HBG1", "HBG2", "EPAS1", "EGLN1", "HIF1A",
        "RNASE1", "LYZ", "PRNP", "APOBEC3A", "APOBEC3G", "IFITM3", "IFITM1",
        "CEACAM1", "CEACAM3", "CEACAM7", "AKR1C4", "H2AC18", "IDI2", "IQCF5",
        "OPN1SW", "RHO", "GJA8", "CRYAA", "UCP1", "LEP", "MC1R", "TAS1R1", "TAS1R2", "TAS2R38",
        "TP53", "ACE2", "TLR4", "TLR7", "MX1", "OAS1"
    }

    selected_tasks = []
    seen_paths = set()

    for path in all_manifest:
        base = os.path.basename(path).replace(".codonified.fa", "")
        sym = base.split("#")[-1] if "#" in base else base
        if sym in priority_symbols and os.path.getsize(path) <= 3_500_000:
            w_msa = os.path.join(WALRUS_MSAS_DIR, f"{base}.msa.fa.gz")
            selected_tasks.append((path, base, w_msa if os.path.exists(w_msa) else None))
            seen_paths.add(path)

    # Add stratified sample up to index 16,000 (<= 2 MB alignments) for fast turnaround
    target_total = 300
    pool = all_manifest[200:16000]
    step = max(1, len(pool) // (target_total - len(selected_tasks) + 5))
    for path in pool[::step]:
        if len(selected_tasks) >= target_total:
            break
        if path in seen_paths:
            continue
        base = os.path.basename(path).replace(".codonified.fa", "")
        w_msa = os.path.join(WALRUS_MSAS_DIR, f"{base}.msa.fa.gz")
        selected_tasks.append((path, base, w_msa if os.path.exists(w_msa) else None))
        seen_paths.add(path)

    print(f"\n=== Part 2: Running Parallel Genome-Wide Sweep on {len(selected_tasks)} TOGA2 Alignments ===", flush=True)
    t_start = time.perf_counter()
    results = []
    n_workers = 16
    with ProcessPoolExecutor(max_workers=n_workers, initializer=init_worker, initargs=(SPECIES_TREE_PATH,)) as ex:
        futs = {ex.submit(analyze_single_alignment, task): task[1] for task in selected_tasks}
        done_cnt = 0
        for fut in as_completed(futs):
            done_cnt += 1
            r = fut.result()
            if r and "error" not in r:
                results.append(r)
            if done_cnt % 25 == 0 or done_cnt == len(selected_tasks):
                elapsed = time.perf_counter() - t_start
                print(f"  [{done_cnt:4d}/{len(selected_tasks):4d}] completed in {elapsed:.1f}s (valid: {len(results)})", flush=True)

    with open(os.path.join(OUT_DIR, "toga2_300_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} gene reports to {OUT_DIR}/toga2_300_results.json", flush=True)


if __name__ == "__main__":
    main()
