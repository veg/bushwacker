#!/usr/bin/env python3
"""
Large-Scale GTDB R95 Microbial Tree of Life & Horizontal Gene Transfer Benchmark
================================================================================
Evaluates BUSHWHACKER (MDL & Binary) and FastTree 2 across all 242 universal
prokaryotic marker gene families in GTDB Release 95:
  - ar122: All 122 Archaeal marker genes across up to 1,672 species representatives
           (spanning all 19 Archaeal phyla).
  - bac120: All 120 Bacterial marker genes across a deterministic, phylum-stratified
            panel of 1,000 Bacterial species representatives covering all 111
            GTDB Bacterial phyla.

For every marker gene family, this script:
  1. Aligns the protein sequences (.faa) using MAFFT (--retree 1) and threads
     the exact Prodigal nucleotide CDS codons (.fna) into a codon MSA, trimming
     columns with >50% gaps (matching GTDB's MSA trimming standard).
  2. Runs BUSHWHACKER (with Stage 4 INTERVAL & SPARSE shunt detection enabled)
     and FastTree 2 (-nt -gtr -gamma).
  3. Evaluates split precision, recall, RF, nRF, soft RF (RF_sp), and taxonomic
     clade monophyly/compatibility across Phylum, Class, Order, Family, and Genus
     against the GTDB R95 reference species trees.
  4. Maps all MDL-certified reticulate shunts (INTERVAL HGT & SPARSE convergence)
     onto the 7-rank GTDB taxonomy.
"""


import sys as _sys
from pathlib import Path as _Path
if str(_Path(__file__).resolve().parent) not in _sys.path:
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
from _env_paths import REPO_ROOT, DATA_DIR, SPECIMENS_DIR, ALIGNMENTS_DIR, SINGLE_CELL_DIR, RESULTS_DIR, WORK_ROOT, EXTERNAL_DIR, resolve_bin, ASTRAL_JAR

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from io import StringIO
from pathlib import Path
from Bio import Phylo, SeqIO

GTDB_DIR = EXTERNAL_DIR / "gtdb_r95"
OUT_DIR = GTDB_DIR / "results"
WORK_DIR = GTDB_DIR / "work"
BW_BIN = resolve_bin("BUSHWHACKER_BIN", REPO_ROOT / "target" / "release" / "bushwhacker", "bushwhacker")
FASTTREE_BIN = resolve_bin("FASTTREE_BIN", "FastTree", "FastTreeMP", "fasttree")
MAFFT_BIN = "/opt/ohpc/pub/apps/mafft/7.526/bin/mafft"


def classify_marker_function(name: str, desc: str) -> str:
    txt = f"{name} {desc}".lower()
    if any(
        k in txt
        for k in [
            "trna-synt",
            "trna ligase",
            "aminoacyl",
            "-trna",
            "trna (",
            "trna_synt",
            "synthetase",
            "tils",
            "miab",
            "trub",
            "mnma",
            "cca ",
        ]
    ) and "atpase" not in txt and "thymidylate" not in txt and "gmp" not in txt and "ctp" not in txt:
        return "Aminoacyl-tRNA & tRNA Mod"
    if any(
        k in txt
        for k in [
            "ribosom",
            "bs20",
            "bl17",
            "bl21",
            "us1",
            "ul1",
            "release factor",
            "initiation factor",
            "elongation factor",
            "prfa",
            "prfb",
            "infb",
            "infc",
            "tsf",
            "fusa",
            "efp",
            "rbfa",
            "ybey",
            "frr",
            "smpb",
        ]
    ):
        return "Ribosomal & Translational"
    if any(
        k in txt
        for k in [
            "rna_pol",
            "rna polymerase",
            "rpo",
            "nusa",
            "nusg",
            "transcription",
            "dnag",
            "dnaa",
            "dna polymerase",
            "reca",
            "ruva",
            "ruvb",
            "ruvc",
            "gyra",
            "gyrb",
            "topa",
            "liga",
            "smc",
            "rad",
            "mfd",
            "uvr",
            "helicase",
            "primase",
            "fibrillarin",
        ]
    ):
        return "Transcription & DNA Replication"
    return "Metabolic & Cellular Operational"


def load_taxonomy(tsv_path: Path) -> dict:
    tax = {}
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            gid, ranks_str = parts[0], parts[1]
            ranks = {}
            for item in ranks_str.split(";"):
                if "__" in item:
                    k, v = item.split("__", 1)
                    ranks[k] = v
            tax[gid] = ranks
    return tax


def select_stratified_bac_panel(tax_dict: dict, ref_tree_path: Path, target_n: int = 1000) -> list:
    panel_file = GTDB_DIR / f"bac120_panel_{target_n}.txt"
    if panel_file.exists():
        with open(panel_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    t_ref = Phylo.read(str(ref_tree_path), "newick")
    tree_leaves = {t.name.strip("'") for t in t_ref.get_terminals() if t.name}

    by_phylum = defaultdict(list)
    for gid, rk in sorted(tax_dict.items()):
        if gid in tree_leaves:
            p = rk.get("p", "") or "Unclassified"
            c = rk.get("c", "")
            o = rk.get("o", "")
            fam = rk.get("f", "")
            by_phylum[p].append((c, o, fam, gid))

    selected = []
    remaining_pools = {}
    # Step 1: Take up to 4 species from EVERY phylum (selecting across distinct classes/orders)
    for p in sorted(by_phylum.keys()):
        items = sorted(by_phylum[p])
        if len(items) <= 4:
            selected.extend(x[3] for x in items)
        else:
            # Stratified across items
            idxs = [int(round(i * (len(items) - 1) / 3.0)) for i in range(4)]
            chosen_set = set(idxs)
            for idx in sorted(chosen_set):
                selected.append(items[idx][3])
            remaining_pools[p] = [items[idx][3] for idx in range(len(items)) if idx not in chosen_set]

    # Step 2: Fill remaining quota up to target_n proportionally to log1p(phylum size) acrossdistinct orders/families
    rem_needed = target_n - len(selected)
    if rem_needed > 0 and remaining_pools:
        weights = {p: math.sqrt(len(pool)) for p, pool in remaining_pools.items()}
        tot_w = sum(weights.values())
        quotas = {
            p: min(len( remaining_pools[p]), int(round(rem_needed * weights[p] / tot_w)))
            for p in remaining_pools
        }
        # Adjust exact sum
        diff = rem_needed - sum(quotas.values())
        for p in sorted(remaining_pools.keys(), key=lambda x: len(remaining_pools[x]) - quotas[x], reverse=True):
            if diff == 0:
                break
            if diff > 0 and quotas[p] < len(remaining_pools[p]):
                quotas[p] += 1
                diff -= 1
            elif diff < 0 and quotas[p] > 0:
                quotas[p] -= 1
                diff += 1

        for p, q in sorted(quotas.items()):
            pool = remaining_pools[p]
            if q > 0:
                if q >= len(pool):
                    selected.extend(pool)
                else:
                    for i in range(q):
                        idx = int(round(i * (len(pool) - 1) / max(1, q - 1)))
                        selected.append(pool[idx])

    selected = sorted(set(selected))[:target_n]
    with open(panel_file, "w", encoding="utf-8") as f:
        for gid in selected:
            f.write(f"{gid}\n")
    return selected


def build_trimmed_codon_alignment(
    faa_path: Path,
    fna_path: Path,
    allowed_taxa: set,
    out_codon_path: Path,
    threads: int = 4,
    max_gap_frac: float = 0.50,
) -> tuple:
    aa_dict = {}
    for rec in SeqIO.parse(str(faa_path), "fasta"):
        if rec.id in allowed_taxa:
            s = str(rec.seq).upper().rstrip("*")
            if len(s) >= 20:
                aa_dict[rec.id] = s

    nt_dict = {}
    for rec in SeqIO.parse(str(fna_path), "fasta"):
        if rec.id in aa_dict:
            s = str(rec.seq).upper().replace("-", "")
            if len(s) >= 3 * len(aa_dict[rec.id]):
                nt_dict[rec.id] = s

    shared_ids = sorted(nt_dict.keys())
    if len(shared_ids) < 10:
        return 0, 0, 0

    tmp_in_faa = out_codon_path.with_suffix(".in.faa")
    tmp_aln_faa = out_codon_path.with_suffix(".aln.faa")
    with open(tmp_in_faa, "w", encoding="utf-8") as f:
        for gid in shared_ids:
            f.write(f">{gid}\n{aa_dict[gid]}\n")

    subprocess.run(
        [MAFFT_BIN, "--retree", "1", "--thread", str(threads), "--quiet", str(tmp_in_faa)],
        stdout=open(tmp_aln_faa, "w"),
        stderr=subprocess.DEVNULL,
        check=True,
    )

    aln_aa_recs = list(SeqIO.parse(str(tmp_aln_faa), "fasta"))
    n_seqs = len(aln_aa_recs)
    l_aa_raw = len(aln_aa_recs[0].seq)

    # Identify columns with <= max_gap_frac gaps
    keep_cols = []
    for col in range(l_aa_raw):
        gaps = sum(1 for r in aln_aa_recs if r.seq[col] == "-")
        if (gaps / n_seqs) <= max_gap_frac:
            keep_cols.append(col)

    if len(keep_cols) < 20:
        keep_cols = list(range(l_aa_raw))
    keep_set = set(keep_cols)

    with open(out_codon_path, "w", encoding="utf-8") as out:
        for r in aln_aa_recs:
            nt = nt_dict[r.id]
            codons = []
            pos = 0
            for col, ch in enumerate(str(r.seq)):
                if ch == "-":
                    if col in keep_set:
                        codons.append("---")
                else:
                    cod = nt[pos : pos + 3]
                    if len(cod) < 3:
                        cod = "---"
                    if col in keep_set:
                        codons.append(cod)
                    pos += 3
            out.write(f">{r.id}\n{''.join(codons)}\n")

    tmp_in_faa.unlink(missing_ok=True)
    tmp_aln_faa.unlink(missing_ok=True)
    return n_seqs, 3 * l_aa_raw, 3 * len(keep_cols)


def get_canonical_splits(tree, all_taxa: set) -> set:
    n = len(all_taxa)
    splits = set()
    clade_leaves = {}
    for clade in tree.find_clades(order="postorder"):
        if clade.is_terminal():
            nm = clade.name.strip("'") if clade.name else ""
            clade_leaves[id(clade)] = frozenset([nm]) if nm in all_taxa else frozenset()
        else:
            s = frozenset().union(*(clade_leaves[id(c)] for c in clade.clades))
            clade_leaves[id(clade)] = s
            if clade != tree.root:
                k = len(s)
                if 1 < k < n - 1:
                    comp = all_taxa - s
                    if k < len(comp):
                        c = s
                    elif len(comp) < k:
                        c = frozenset(comp)
                    else:
                        c = frozenset(min(s, comp, key=lambda x: min(x)))
                    splits.add(c)
    return splits


def evaluate_taxonomic_ranks(gene_splits: set, all_taxa: set, tax_dict: dict) -> dict:
    n = len(all_taxa)
    # Build ground-truth clades for each taxonomic rank (p, c, o, f, g)
    rank_stats = {}
    for rk_code, rk_name in [("p", "Phylum"), ("c", "Class"), ("o", "Order"), ("f", "Family"), ("g", "Genus")]:
        groups = defaultdict(set)
        for gid in all_taxa:
            val = tax_dict.get(gid, {}).get(rk_code, "")
            if val:
                groups[val].add(gid)
        total_clades = 0
        recovered_clades = 0
        compatible_clades = 0
        for grp_name, members in groups.items():
            k = len(members)
            if 2 <= k <= n - 2:
                total_clades += 1
                comp = all_taxa - members
                if k < len(comp):
                    canon = frozenset(members)
                elif len(comp) < k:
                    canon = frozenset(comp)
                else:
                    canon = frozenset(min(members, comp, key=lambda s: min(s)))
                if canon in gene_splits:
                    recovered_clades += 1
                    compatible_clades += 1
                else:
                    # Check Four-Gamete compatibility with all resolved gene_splits
                    conflict = False
                    for sp in gene_splits:
                        q11 = len(members & sp)
                        if q11 > 0 and q11 < k and q11 < len(sp) and (len(all_taxa) - k - len(sp) + q11) > 0:
                            conflict = True
                            break
                    if not conflict:
                        compatible_clades += 1
        rank_stats[rk_name] = {
            "total": total_clades,
            "recovered": recovered_clades,
            "compatible": compatible_clades,
        }
    return rank_stats


def classify_shunt_taxonomy(u_leaves: list, v_leaves: list, tax_dict: dict) -> dict:
    ranks_order = [("d", "Domain"), ("p", "Phylum"), ("c", "Class"), ("o", "Order"), ("f", "Family"), ("g", "Genus"), ("s", "Species")]
    u_tax = [tax_dict.get(x, {}) for x in u_leaves if x in tax_dict]
    v_tax = [tax_dict.get(x, {}) for x in v_leaves if x in tax_dict]
    if not u_tax or not v_tax:
        return {"bridge_level": "Unknown", "u_summary": "Unknown", "v_summary": "Unknown"}

    def summarize_clade(t_list):
        for code, rname in reversed(ranks_order):
            vals = {t.get(code, "") for t in t_list if t.get(code, "")}
            if len(vals) == 1:
                p_val = next(iter({t.get("p", "") for t in t_list if t.get("p", "")}), "")
                v = next(iter(vals))
                return f"{code}__{v} (p__{p_val})" if code != "p" and p_val else f"{code}__{v}"
        p_vals = sorted({t.get("p", "") for t in t_list if t.get("p", "")})
        return f"Multi-clade ({','.join(p_vals[:2])})"

    # Determine highest taxonomic rank where u_tax and v_tax are completely disjoint
    bridge_level = "Intra-Genus"
    for code, rname in [("p", "Inter-Phylum"), ("c", "Inter-Class"), ("o", "Inter-Order"), ("f", "Inter-Family"), ("g", "Inter-Genus")]:
        u_vals = {t.get(code, "") for t in u_tax if t.get(code, "")}
        v_vals = {t.get(code, "") for t in v_tax if t.get(code, "")}
        if u_vals and v_vals and u_vals.isdisjoint(v_vals):
            bridge_level = rname
            break

    return {
        "bridge_level": bridge_level,
        "u_summary": summarize_clade(u_tax),
        "v_summary": summarize_clade(v_tax),
        "u_phyla": sorted({t.get("p", "") for t in u_tax if t.get("p", "")}),
        "v_phyla": sorted({t.get("p", "") for t in v_tax if t.get("p", "")}),
    }


def parse_shunts_with_clades(bw_json: dict, tax_dict: dict, all_leaves: set) -> list:
    shunts_out = []

    for sh in bw_json.get("shunts", []):
        lu = sh["label_u"]
        lv = sh["label_v"]
        u_leaves = sh.get("leaves_u") or ([lu] if lu in all_leaves else [])
        v_leaves = sh.get("leaves_v") or ([lv] if lv in all_leaves else [])
        tax_info = classify_shunt_taxonomy(u_leaves, v_leaves, tax_dict)
        shunts_out.append({
            "mode": sh["mode"],
            "node_u": sh["node_u"],
            "node_v": sh["node_v"],
            "label_u": lu,
            "label_v": lv,
            "leaves_u": u_leaves[:5],
            "leaves_v": v_leaves[:5],
            "shared_mutations": sh["shared_mutations"],
            "interval_span": sh["interval_span"],
            "net_mdl_gain_nats": sh["net_mdl_gain_nats"],
            "posterior_prob": sh["posterior_prob"],
            "bridge_level": tax_info["bridge_level"],
            "u_summary": tax_info["u_summary"],
            "v_summary": tax_info["v_summary"],
            "u_phyla": tax_info.get("u_phyla", []),
            "v_phyla": tax_info.get("v_phyla", []),
        })
    return shunts_out


def load_markers_list() -> list:
    markers = []
    for domain, info_tsv in [("ar122", GTDB_DIR / "ar122_marker_info_r95.tsv"), ("bac120", GTDB_DIR / "bac120_marker_info_r95.tsv")]:
        with open(info_tsv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                mid = row["Marker Id"].replace("PFAM_", "").replace("TIGR_", "")
                markers.append({
                    "domain": domain,
                    "marker_id": mid,
                    "raw_id": row["Marker Id"],
                    "name": row["Name"],
                    "description": row["Description"],
                    "func_class": classify_marker_function(row["Name"], row["Description"]),
                    "nom_len_aa": int(row["Length (bp)"]),
                    "single_copy_pct": float(row["Single copy (%)"]),
                    "ubiquity_pct": float(row["Ubiquity (%)"]),
                })
    return markers


def run_single_marker(idx: int, markers: list, threads: int = 4):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    m = markers[idx]
    domain = m["domain"]
    mid = m["marker_id"]
    out_json_path = OUT_DIR / f"{domain}_{mid}.json"
    if out_json_path.exists():
        print(f"[{idx}] {domain}_{mid} already completed.")
        return

    if domain == "ar122":
        tax_dict = load_taxonomy(GTDB_DIR / "ar122_taxonomy_r95.tsv")
        ref_tree_path = GTDB_DIR / "ar122_r95.tree"
        allowed_taxa = set(tax_dict.keys())
        base_dir = GTDB_DIR / "ar122_marker_genes_reps_r95"
    else:
        tax_dict = load_taxonomy(GTDB_DIR / "bac120_taxonomy_r95.tsv")
        ref_tree_path = GTDB_DIR / "bac120_r95.tree"
        allowed_taxa = set(select_stratified_bac_panel(tax_dict, ref_tree_path, target_n=1000))
        base_dir = GTDB_DIR / "bac120_marker_genes_reps_r95"

    faa_path = base_dir / "faa" / f"{mid}.faa"
    fna_path = base_dir / "fna" / f"{mid}.fna"
    codon_path = WORK_DIR / f"{domain}_{mid}.codon.fna"

    t_align0 = time.time()
    n_seqs, l_raw_bp, l_trim_bp = build_trimmed_codon_alignment(
        faa_path, fna_path, allowed_taxa, codon_path, threads=threads, max_gap_frac=0.50
    )
    align_sec = time.time() - t_align0

    # Run Bushwhacker (with shunts enabled)
    bw_nwk_path = WORK_DIR / f"{domain}_{mid}.bw_mdl.nwk"
    bw_bin_nwk_path = WORK_DIR / f"{domain}_{mid}.bw_bin.nwk"
    bw_rep_path = WORK_DIR / f"{domain}_{mid}.bw.json"

    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = str(threads)

    t_bw0 = time.time()
    subprocess.run(
        [BW_BIN, str(codon_path), "-o", str(bw_nwk_path), "-j", str(bw_rep_path)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    bw_wall_sec = time.time() - t_bw0

    with open(bw_rep_path, "r", encoding="utf-8") as f:
        bw_data = json.load(f)

    with open(bw_bin_nwk_path, "w", encoding="utf-8") as f:
        f.write(bw_data["binary_newick"] + "\n")

    # Run FastTree 2 (-nt -gtr -gamma)
    ft_nwk_path = WORK_DIR / f"{domain}_{mid}.fasttree.nwk"
    env_ft = os.environ.copy()
    env_ft["OMP_NUM_THREADS"] = str(threads)
    t_ft0 = time.time()
    subprocess.run(
        [FASTTREE_BIN, "-nt", "-gtr", "-gamma", "-quiet", str(codon_path)],
        env=env_ft,
        stdout=open(ft_nwk_path, "w"),
        stderr=subprocess.DEVNULL,
        check=True,
    )
    ft_wall_sec = time.time() - t_ft0

    # Evaluate against GTDB R95 Reference Species Tree
    t_ref = Phylo.read(str(ref_tree_path), "newick")
    t_bw_mdl = Phylo.read(StringIO(bw_data["newick"]), "newick")
    t_bw_bin = Phylo.read(StringIO(bw_data["binary_newick"]), "newick")
    t_ft = Phylo.read(str(ft_nwk_path), "newick")

    mdl_leaves = {t.name.strip("'") for t in t_bw_mdl.get_terminals() if t.name}
    ref_leaves = {t.name.strip("'") for t in t_ref.get_terminals() if t.name}
    shared_taxa = mdl_leaves & ref_leaves

    ref_splits = get_canonical_splits(t_ref, shared_taxa)
    mdl_splits = get_canonical_splits(t_bw_mdl, shared_taxa)
    bin_splits = get_canonical_splits(t_bw_bin, shared_taxa)
    ft_splits = get_canonical_splits(t_ft, shared_taxa)

    max_sp = max(1, len(ref_splits))

    def summarize_splits(sp_set):
        tp = len(sp_set & ref_splits)
        fp = len(sp_set - ref_splits)
        fn = len(ref_splits - sp_set)
        prec = tp / max(1, tp + fp)
        rec = tp / max_sp
        rf = fp + fn
        nrf = rf / (2.0 * max_sp)
        rf_sp = fp + 0.5 * max(0, max_sp - len(sp_set))
        nrf_sp = rf_sp / max_sp
        return {
            "resolved": len(sp_set),
            "max_splits": max_sp,
            "resolution_rate": len(sp_set) / max_sp,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": prec,
            "recall": rec,
            "rf": rf,
            "nrf": nrf,
            "rf_sp": rf_sp,
            "nrf_sp": nrf_sp,
            "rank_stats": evaluate_taxonomic_ranks(sp_set, shared_taxa, tax_dict),
        }

    shunts_annotated = parse_shunts_with_clades(bw_data, tax_dict, shared_taxa)

    result = {
        "idx": idx,
        "domain": domain,
        "marker_id": mid,
        "name": m["name"],
        "description": m["description"],
        "func_class": m["func_class"],
        "n_taxa": len(shared_taxa),
        "l_raw_bp": l_raw_bp,
        "l_trim_bp": l_trim_bp,
        "align_sec": align_sec,
        "bw_runtime_sec": bw_data["runtime_sec"],
        "bw_wall_sec": bw_wall_sec,
        "ft_wall_sec": ft_wall_sec,
        "speedup_vs_fasttree": ft_wall_sec / max(1e-6, bw_data["runtime_sec"]),
        "raw_nats": bw_data["raw_Description_Length_nats"],
        "star_nats": bw_data["star_tree_MDL_nats"],
        "tree_nats": bw_data["tree_MDL_nats"],
        "network_nats": bw_data["network_MDL_nats"],
        "rho_compression": 1.0 - (bw_data["network_MDL_nats"] / max(1.0, bw_data["star_tree_MDL_nats"])),
        "polytomy_nodes": bw_data["polytomy_nodes"],
        "n_shunts": len(shunts_annotated),
        "n_interval_shunts": sum(1 for s in shunts_annotated if s["mode"] == "INTERVAL"),
        "n_sparse_shunts": sum(1 for s in shunts_annotated if s["mode"] == "SPARSE"),
        "shunts": shunts_annotated,
        "bw_mdl": summarize_splits(mdl_splits),
        "bw_bin": summarize_splits(bin_splits),
        "fasttree": summarize_splits(ft_splits),
    }

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    codon_path.unlink(missing_ok=True)
    bw_rep_path.unlink(missing_ok=True)
    print(
        f"[{idx}] {domain}_{mid} ({m['name']}, N={len(shared_taxa)}, L={l_trim_bp}): "
        f"BW={bw_data['runtime_sec']:.2f}s vs FT={ft_wall_sec:.2f}s ({result['speedup_vs_fasttree']:.1f}x) | "
        f"BW_MDL prec={result['bw_mdl']['precision']:.3f} | Shunts={len(shunts_annotated)}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=None, help="Single marker index (0..241)")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=242)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    markers = load_markers_list()
    if args.idx is not None:
        run_single_marker(args.idx, markers, threads=args.threads)
    else:
        for i in range(args.start, min(args.end, len(markers))):
            run_single_marker(i, markers, threads=args.threads)


if __name__ == "__main__":
    main()
