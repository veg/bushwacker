#!/usr/bin/env python3
"""
Pipeline 3: Microbial 'Tree vs. Web' MDL Information Budget Across Pathogen Pangenomes & GTDB
Runs Bushwhacker (Nucleotide + 20-state Amino-Acid modes) across:
  1. Klebsiella pneumoniae complex core/accessory gene alignments (.fna + .aa.fa.aln),
     the 214 kb core super-matrix, and the blaKPC carbapenemase AMR horizontal transfer composite.
  2. All 122 GTDB R95 Archaeal domain-wide marker protein alignments (ar122_msa/*.faa, ~1,672 species),
     evaluated both de novo and against the GTDB R95 Archaeal reference tree (ar122_r95.tree),
     plus all 68 GTDB Bacterial/Archaeal clade alignments (NT + --translate AA).
  3. Stratifies HGT shunt prevalence and lateral nats savings by Informational vs. Operational
     functional categories (Jain, Rivera, & Lake 1999 Complexity Hypothesis) and maps
     inter-phylum / inter-class HGT highways.
"""


import sys as _sys
from pathlib import Path as _Path
if str(_Path(__file__).resolve().parent) not in _sys.path:
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
from _env_paths import REPO_ROOT, DATA_DIR, SPECIMENS_DIR, ALIGNMENTS_DIR, SINGLE_CELL_DIR, RESULTS_DIR, WORK_ROOT, EXTERNAL_DIR, resolve_bin, ASTRAL_JAR

import os
import re
import glob
import json
import time
import shutil
import tempfile
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

BUSHWHACKER_BIN = resolve_bin("BUSHWHACKER_BIN", REPO_ROOT / "target" / "release" / "bushwhacker", "bushwhacker")
BASE_DIR = str(WORK_ROOT / "microbial_web_atlas")

KLEB_DIR = str(ALIGNMENTS_DIR / "microbial_and_16s")
KLEB_CORE_DIR = os.path.join(KLEB_DIR, "core_alignments")
KLEB_SUPER_FA = os.path.join(KLEB_DIR, "super_matrix", "klebsiella_core_super_matrix_214kb.fna")
KLEB_SUPER_TREE = os.path.join(KLEB_DIR, "super_matrix", "super_fasttree.tree")
KLEB_AMR_COMPOSITE = os.path.join(KLEB_DIR, "amr_hgt", "klebsiella_core_plus_blaKPC_composite.fna")
KLEB_BLAKPC_FA = os.path.join(KLEB_DIR, "amr_hgt", "blaKPC_9taxa.fna")

GTDB_DIR = str(EXTERNAL_DIR / "gtdb_r95")
AR122_MSA_DIR = os.path.join(GTDB_DIR, "ar122_msa")
AR122_TREE = os.path.join(GTDB_DIR, "ar122_r95.tree")
AR122_TAX = os.path.join(GTDB_DIR, "ar122_taxonomy_r95.tsv")
AR122_INFO = os.path.join(GTDB_DIR, "ar122_marker_info_r95.tsv")
BAC120_TAX = os.path.join(GTDB_DIR, "bac120_taxonomy_r95.tsv")
BAC120_INFO = os.path.join(GTDB_DIR, "bac120_marker_info_r95.tsv")
CLADE_WORK_DIR = os.path.join(GTDB_DIR, "clade_shunt_work")


def parse_newick(nwk_str):
    s = nwk_str.strip()
    if s.endswith(";"):
        s = s[:-1]
    tokens = re.findall(r"'[^']*'|\(|\)|,|;|:[^(),;]+|[^(),;:]+", s)
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
                lbl = tokens[i].strip("'")
            if i + 1 < n and tokens[i + 1].startswith(":"):
                i += 1
            stack[-1].append({"children": children, "name": lbl})
        elif tok.startswith(":"):
            pass
        else:
            lbl = tok.strip("'")
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
    kept = []
    for c in node["children"]:
        pc = prune_tree(c, keep_tips)
        if pc is not None:
            kept.append(pc)
    if not kept:
        return None
    if len(kept) == 1:
        return kept[0]
    return {"children": kept, "name": node.get("name")}


def to_newick(node):
    if not node["children"]:
        return f"{node['name']}:0.01"
    return "(" + ",".join(to_newick(c) for c in node["children"]) + "):0.01"


def load_gtdb_taxonomy(*tsv_paths):
    tax = {}
    for p in tsv_paths:
        if not os.path.exists(p):
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                gid, ranks_str = parts[0].strip(), parts[1].strip()
                ranks = {}
                for item in ranks_str.split(";"):
                    if "__" in item:
                        k, v = item.split("__", 1)
                        ranks[k.strip()] = v.strip()
                tax[gid] = ranks
    return tax


def load_marker_info(*info_paths):
    info = {}
    info_kw = ("ribosom", "rnap", "rna polymerase", "trna", "translation", "elongation factor", "initiation factor", "nus", "secY", "ffh")
    for p in info_paths:
        if not os.path.exists(p):
            continue
        with open(p, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                mid = parts[0].replace("PFAM_", "").replace("TIGRFAM_", "").strip()
                name, desc = parts[1].strip(), parts[2].strip()
                txt = (name + " " + desc).lower()
                cat = "Informational" if any(k in txt for k in info_kw) else "Operational"
                info[mid] = {"name": name, "description": desc, "category": cat}
    return info


def run_bw_single(task):
    fa_path, label, extra_flags, ref_nwk_str = task
    shm_root = "/dev/shm" if os.path.isdir("/dev/shm") else None
    tmpdir = tempfile.mkdtemp(prefix="bw_mic_", dir=shm_root)
    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = "2"
    try:
        out_nwk = os.path.join(tmpdir, "free.nwk")
        out_json = os.path.join(tmpdir, "free.json")
        cmd = [BUSHWHACKER_BIN] + extra_flags + [fa_path, "-o", out_nwk, "-j", out_json]
        res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
        if res.returncode != 0 or not os.path.exists(out_json):
            return {"label": label, "status": "error", "error": res.stderr[-250:] if res.stderr else "fail"}
        free_d = json.load(open(out_json))

        ref_d = None
        if ref_nwk_str:
            tips = set()
            with open(fa_path, "r", errors="replace") as ff:
                for line in ff:
                    if line.startswith(">"):
                        tips.add(line[1:].strip().split()[0])
            root = parse_newick(ref_nwk_str)
            pruned = prune_tree(root, tips) if len(tips) >= 4 else None
            if pruned and pruned.get("children"):
                ref_p = os.path.join(tmpdir, "ref.nwk")
                ref_o = os.path.join(tmpdir, "ref_out.nwk")
                ref_j = os.path.join(tmpdir, "ref_out.json")
                with open(ref_p, "w") as rf:
                    rf.write(to_newick(pruned) + ";\n")
                r2 = subprocess.run([BUSHWHACKER_BIN] + extra_flags + [fa_path, "-t", ref_p, "-o", ref_o, "-j", ref_j],
                                    capture_output=True, text=True, env=env, timeout=600)
                if r2.returncode == 0 and os.path.exists(ref_j):
                    ref_d = json.load(open(ref_j))

        free_shunts = free_d.get("shunts", [])
        ref_shunts = ref_d.get("shunts", []) if ref_d else []
        tree_savings = max(0.0, free_d.get("star_tree_MDL_nats", 0.0) - free_d.get("tree_MDL_nats", 0.0))
        shunt_savings_free = sum(s.get("net_mdl_gain_nats", 0.0) for s in free_shunts)
        shunt_savings_ref = sum(s.get("net_mdl_gain_nats", 0.0) for s in ref_shunts)

        return {
            "label": label,
            "path": fa_path,
            "status": "ok",
            "alphabet": free_d.get("alphabet"),
            "n_taxa": free_d.get("n_taxa"),
            "seq_len": free_d.get("seq_len"),
            "runtime_sec": free_d.get("runtime_sec"),
            "star_mdl_nats": free_d.get("star_tree_MDL_nats"),
            "tree_mdl_nats": free_d.get("tree_MDL_nats"),
            "network_mdl_nats": free_d.get("network_MDL_nats"),
            "vertical_tree_savings_nats": round(tree_savings, 3),
            "free_shunt_net_savings_nats": round(shunt_savings_free, 3),
            "ref_tree_mdl_nats": ref_d.get("tree_MDL_nats") if ref_d else None,
            "ref_network_mdl_nats": ref_d.get("network_MDL_nats") if ref_d else None,
            "ref_shunt_net_savings_nats": round(shunt_savings_ref, 3),
            "resolved_splits": free_d.get("resolved_internal_splits"),
            "max_binary_splits": free_d.get("max_binary_splits"),
            "ref_resolved_splits": ref_d.get("resolved_internal_splits") if ref_d else None,
            "n_free_shunts": len(free_shunts),
            "n_free_interval": sum(1 for s in free_shunts if s.get("mode") == "INTERVAL"),
            "n_free_sparse": sum(1 for s in free_shunts if s.get("mode") == "SPARSE"),
            "n_ref_shunts": len(ref_shunts),
            "n_ref_interval": sum(1 for s in ref_shunts if s.get("mode") == "INTERVAL"),
            "n_ref_sparse": sum(1 for s in ref_shunts if s.get("mode") == "SPARSE"),
            "free_shunts": free_shunts[:20],
            "ref_shunts": ref_shunts[:20],
        }
    except Exception as e:
        return {"label": label, "status": "error", "error": str(e)}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def annotate_gtdb_shunts(records, tax_map, marker_info):
    phylum_pairs = Counter()
    class_pairs = Counter()
    by_category = defaultdict(lambda: {"loci": 0, "loci_with_shunts": 0, "total_shunts": 0, "interval_shunts": 0, "sparse_shunts": 0, "inter_phylum_shunts": 0, "inter_class_shunts": 0, "total_shunt_nats": 0.0, "total_tree_nats": 0.0})
    top_inter_phylum = []

    for r in records:
        if r.get("status") != "ok":
            continue
        lbl = r["label"]
        mid = lbl
        for pref in ("ar122_", "clade_nt_", "clade_aa_"):
            if mid.startswith(pref):
                mid = mid[len(pref):]
        # Extract PFxxxx or TIGRxxxx
        m_match = re.search(r"(PF\d+\.\d+|TIGR\d+)", mid)
        m_key = m_match.group(1) if m_match else mid
        cat = marker_info.get(m_key, {}).get("category", "Operational")
        m_name = marker_info.get(m_key, {}).get("name", m_key)

        shunts = r.get("ref_shunts") if r.get("ref_shunts") else r.get("free_shunts", [])
        st = by_category[cat]
        st["loci"] += 1
        if shunts:
            st["loci_with_shunts"] += 1
        st["total_shunts"] += len(shunts)
        st["interval_shunts"] += sum(1 for s in shunts if s.get("mode") == "INTERVAL")
        st["sparse_shunts"] += sum(1 for s in shunts if s.get("mode") == "SPARSE")
        st["total_shunt_nats"] += sum(s.get("net_mdl_gain_nats", 0.0) for s in shunts)
        st["total_tree_nats"] += r.get("vertical_tree_savings_nats", 0.0)

        for s in shunts:
            u_lvs = s.get("leaves_u", [])
            v_lvs = s.get("leaves_v", [])
            u_phy = sorted(set(tax_map.get(x, {}).get("p", "") for x in u_lvs if tax_map.get(x, {}).get("p", "")))
            v_phy = sorted(set(tax_map.get(x, {}).get("p", "") for x in v_lvs if tax_map.get(x, {}).get("p", "")))
            u_cls = sorted(set(tax_map.get(x, {}).get("c", "") for x in u_lvs if tax_map.get(x, {}).get("c", "")))
            v_cls = sorted(set(tax_map.get(x, {}).get("c", "") for x in v_lvs if tax_map.get(x, {}).get("c", "")))

            if u_phy and v_phy and set(u_phy).isdisjoint(set(v_phy)):
                st["inter_phylum_shunts"] += 1
                pair = tuple(sorted([u_phy[0], v_phy[0]]))
                phylum_pairs[f"{pair[0]} <-> {pair[1]}"] += 1
                top_inter_phylum.append({
                    "locus": lbl,
                    "marker_name": m_name,
                    "category": cat,
                    "mode": s.get("mode"),
                    "net_mdl_gain_nats": round(s.get("net_mdl_gain_nats", 0.0), 2),
                    "shared_mutations": s.get("shared_mutations"),
                    "interval_span": s.get("interval_span"),
                    "u_phyla": u_phy,
                    "v_phyla": v_phy,
                    "u_classes": u_cls[:3],
                    "v_classes": v_cls[:3],
                    "clade_size_u": s.get("clade_size_u", len(u_lvs)),
                    "clade_size_v": s.get("clade_size_v", len(v_lvs)),
                    "sites_1based": s.get("residues_1based", s.get("codons_1based", []))[:16],
                })
            elif u_cls and v_cls and set(u_cls).isdisjoint(set(v_cls)):
                st["inter_class_shunts"] += 1
                pair = tuple(sorted([u_cls[0], v_cls[0]]))
                class_pairs[f"{pair[0]} <-> {pair[1]}"] += 1

    top_inter_phylum.sort(key=lambda x: x["net_mdl_gain_nats"], reverse=True)
    return {
        "complexity_hypothesis_by_category": dict(by_category),
        "top_inter_phylum_highways": phylum_pairs.most_common(25),
        "top_inter_class_highways": class_pairs.most_common(25),
        "top_inter_phylum_shunts": top_inter_phylum[:40],
    }


def main():
    os.makedirs(BASE_DIR, exist_ok=True)
    kleb_ref_nwk = open(KLEB_SUPER_TREE).read() if os.path.exists(KLEB_SUPER_TREE) else None
    ar122_ref_nwk = open(AR122_TREE).read() if os.path.exists(AR122_TREE) else None

    tasks = []
    # 1. Klebsiella core/accessory DNA (.fna) and AA (.aa.fa.aln)
    for fna in sorted(glob.glob(os.path.join(KLEB_CORE_DIR, "*.fna"))):
        stem = os.path.basename(fna)[:-4]
        tasks.append((fna, f"kleb_nt_{stem}", [], kleb_ref_nwk))
    for faa in sorted(glob.glob(os.path.join(KLEB_CORE_DIR, "*.aa.fa.aln"))):
        stem = os.path.basename(faa).replace(".aa.fa.aln", "")
        tasks.append((faa, f"kleb_aa_{stem}", ["--protein"], kleb_ref_nwk))

    # 2. Klebsiella 214 kb Core Super-Matrix & blaKPC AMR Plasmid HGT Composite
    if os.path.exists(KLEB_SUPER_FA):
        tasks.append((KLEB_SUPER_FA, "kleb_super_matrix_214kb", [], kleb_ref_nwk))
    if os.path.exists(KLEB_AMR_COMPOSITE):
        tasks.append((KLEB_AMR_COMPOSITE, "kleb_core_plus_blaKPC_composite", [], kleb_ref_nwk))
    if os.path.exists(KLEB_BLAKPC_FA):
        tasks.append((KLEB_BLAKPC_FA, "kleb_blaKPC_9taxa", [], None))

    # 3. GTDB R95 122 Archaeal Domain-Wide Protein Markers (ar122_msa/*.faa)
    for faa in sorted(glob.glob(os.path.join(AR122_MSA_DIR, "*.faa"))):
        stem = os.path.basename(faa).replace("gtdb_r95_ar_", "").replace(".faa", "")
        tasks.append((faa, f"ar122_{stem}", ["--protein"], ar122_ref_nwk))

    # 4. GTDB R95 68 Bacterial/Archaeal Clade Coding Alignments (NT + AA)
    for fna in sorted(glob.glob(os.path.join(CLADE_WORK_DIR, "*.fna"))):
        stem = os.path.basename(fna)[:-4]
        tasks.append((fna, f"clade_nt_{stem}", [], None))
        tasks.append((fna, f"clade_aa_{stem}", ["--translate"], None))

    print(f"[P3 Microbial Web] Launching {len(tasks)} alignments across 32 workers...", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=32) as ex:
        for idx, res in enumerate(ex.map(run_bw_single, tasks, chunksize=2), 1):
            results.append(res)
            if idx % 100 == 0 or idx == len(tasks):
                print(f"  [P3] Completed {idx}/{len(tasks)} alignments", flush=True)

    tax_map = load_gtdb_taxonomy(AR122_TAX, BAC120_TAX)
    marker_info = load_marker_info(AR122_INFO, BAC120_INFO)

    kleb_nt = [r for r in results if r.get("status") == "ok" and r["label"].startswith("kleb_nt_")]
    kleb_aa = [r for r in results if r.get("status") == "ok" and r["label"].startswith("kleb_aa_")]
    ar122_res = [r for r in results if r.get("status") == "ok" and r["label"].startswith("ar122_")]
    clade_nt = [r for r in results if r.get("status") == "ok" and r["label"].startswith("clade_nt_")]
    clade_aa = [r for r in results if r.get("status") == "ok" and r["label"].startswith("clade_aa_")]
    specials = {r["label"]: r for r in results if r["label"].startswith("kleb_super") or "blaKPC" in r["label"]}

    def summarize_group(grp):
        if not grp:
            return {}
        tot_tree = sum(r["vertical_tree_savings_nats"] for r in grp)
        tot_free_shunt = sum(r["free_shunt_net_savings_nats"] for r in grp)
        tot_ref_shunt = sum(r["ref_shunt_net_savings_nats"] for r in grp)
        return {
            "n_loci": len(grp),
            "loci_with_free_shunts": sum(1 for r in grp if r["n_free_shunts"] > 0),
            "loci_with_ref_shunts": sum(1 for r in grp if r["n_ref_shunts"] > 0),
            "total_free_shunts": sum(r["n_free_shunts"] for r in grp),
            "total_free_interval": sum(r["n_free_interval"] for r in grp),
            "total_free_sparse": sum(r["n_free_sparse"] for r in grp),
            "total_ref_shunts": sum(r["n_ref_shunts"] for r in grp),
            "total_ref_interval": sum(r["n_ref_interval"] for r in grp),
            "total_ref_sparse": sum(r["n_ref_sparse"] for r in grp),
            "total_vertical_tree_savings_nats": round(tot_tree, 2),
            "total_free_shunt_savings_nats": round(tot_free_shunt, 2),
            "total_ref_shunt_savings_nats": round(tot_ref_shunt, 2),
            "mean_resolved_split_frac": round(
                sum(r["resolved_splits"] / max(1, r["max_binary_splits"]) for r in grp) / len(grp), 4
            ),
            "total_runtime_sec": round(sum(r.get("runtime_sec", 0.0) or 0.0 for r in grp), 2),
        }

    gtdb_ar122_annot = annotate_gtdb_shunts(ar122_res, tax_map, marker_info)
    gtdb_clade_aa_annot = annotate_gtdb_shunts(clade_aa, tax_map, marker_info)

    # Top Klebsiella HGT loci
    top_kleb_hgt = sorted(kleb_nt, key=lambda r: r["ref_shunt_net_savings_nats"], reverse=True)[:25]
    top_kleb_summary = [
        {
            "gene": r["label"].replace("kleb_nt_", ""),
            "seq_len": r["seq_len"],
            "n_ref_shunts": r["n_ref_shunts"],
            "n_ref_interval": r["n_ref_interval"],
            "ref_shunt_net_savings_nats": r["ref_shunt_net_savings_nats"],
            "top_shunts": [
                {
                    "mode": s.get("mode"),
                    "gain_nats": round(s.get("net_mdl_gain_nats", 0.0), 2),
                    "shared_muts": s.get("shared_mutations"),
                    "span": s.get("interval_span"),
                    "u": s.get("leaves_u", [])[:4],
                    "v": s.get("leaves_v", [])[:4],
                }
                for s in r.get("ref_shunts", [])[:4]
            ],
        }
        for r in top_kleb_hgt
    ]

    atlas = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "klebsiella_core_nt_summary": summarize_group(kleb_nt),
        "klebsiella_core_aa_summary": summarize_group(kleb_aa),
        "klebsiella_special_benchmarks": specials,
        "top_klebsiella_hgt_genes": top_kleb_summary,
        "gtdb_ar122_protein_summary": summarize_group(ar122_res),
        "gtdb_ar122_hgt_and_complexity": gtdb_ar122_annot,
        "gtdb_clade_nt_summary": summarize_group(clade_nt),
        "gtdb_clade_aa_summary": summarize_group(clade_aa),
        "gtdb_clade_aa_hgt_and_complexity": gtdb_clade_aa_annot,
    }

    out_path = os.path.join(BASE_DIR, "microbial_web_atlas.json")
    with open(out_path, "w") as f:
        json.dump(atlas, f, indent=2)
    with open(os.path.join(BASE_DIR, "all_loci_raw.jsonl"), "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"[P3] Wrote complete microbial web atlas to {out_path}", flush=True)


if __name__ == "__main__":
    main()
