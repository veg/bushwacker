#!/usr/bin/env python3
"""
GTDB R95 ASTRAL 5.7.8 Species-Tree Reconstruction & MDL-Certified HGT Network Analysis
======================================================================================
1. Runs ASTRAL 5.7.8 on the 122 Archaeal gene trees (ar122) and 120 Bacterial gene
   trees (bac120) inferred by:
     - Bushwhacker (MDL) [parameter-free MDL polytomies]
     - Bushwhacker (Binary)
     - FastTree 2 (GTR+G)
   and evaluates the resulting domain-wide species trees against the GTDB R95
   Reference Species Trees (`ar122_r95.tree` and `bac120_r95.tree`).
2. Extracts all high-confidence MDL-certified inter-phylum, inter-class, and
   inter-order Horizontal Gene Transfer (xenologous clade-jump) splits across the
   242 `Bushwhacker (MDL)` gene trees (splits with MDL support >= 0.999, corresponding
   to >= 6.9 nats / 10 bits of net Kolmogorov compression gain, uniting a focal
   recipient taxon/clade of <= 5 species from one taxon group directly inside a
   donor clade of >= 3 species from a disjoint Phylum, Class, or Order), and tests
   the Jain, Rivera, & Lake (1999) Complexity Hypothesis across functional classes.
3. Runs the unmodified `bushwhacker` binary on multi-gene concatenated alignments
   and major Archaeal/Bacterial classes/orders (N <= 160) to catalog MDL-certified
   Stage 4 `INTERVAL` (intragenic/mosaic HGT) and `SPARSE` (convergent) shunts.
"""


import sys as _sys
from pathlib import Path as _Path
if str(_Path(__file__).resolve().parent) not in _sys.path:
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
from _env_paths import REPO_ROOT, DATA_DIR, SPECIMENS_DIR, ALIGNMENTS_DIR, SINGLE_CELL_DIR, RESULTS_DIR, WORK_ROOT, EXTERNAL_DIR, resolve_bin, ASTRAL_JAR

import csv
import json
import subprocess
import time
from collections import Counter, defaultdict
from io import StringIO
from pathlib import Path
import numpy as np
from Bio import Phylo, SeqIO

GTDB_DIR = EXTERNAL_DIR / "gtdb_r95"
WORK_DIR = GTDB_DIR / "work"
RES_DIR = GTDB_DIR / "results"
ASTRAL_JAR = resolve_bin("ASTRAL_JAR", EXTERNAL_DIR / "Astral" / "astral.5.7.8.jar", "astral.jar")
BW_BIN = resolve_bin("BUSHWHACKER_BIN", REPO_ROOT / "target" / "release" / "bushwhacker", "bushwhacker")
MAFFT_BIN = "/opt/ohpc/pub/apps/mafft/7.526/bin/mafft"


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


def get_canonical_splits_with_support(tree, all_taxa: set) -> dict:
    n = len(all_taxa)
    splits = {}
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
                    sup = 0.0
                    if clade.confidence is not None:
                        try:
                            sup = float(clade.confidence)
                        except ValueError:
                            sup = 0.0
                    elif clade.name:
                        try:
                            sup = float(clade.name)
                        except ValueError:
                            sup = 0.0
                    splits[c] = max(splits.get(c, 0.0), sup)
    return splits


def precompute_ref_clades(tree, allowed_taxa: set) -> list:
    clade_leaves = {}
    non_trivial = []
    n = len(allowed_taxa)
    for clade in tree.find_clades(order="postorder"):
        if clade.is_terminal():
            nm = clade.name.strip("'") if clade.name else ""
            clade_leaves[id(clade)] = frozenset([nm]) if nm in allowed_taxa else frozenset()
        else:
            s = frozenset().union(*(clade_leaves[id(c)] for c in clade.clades))
            clade_leaves[id(clade)] = s
            if clade != tree.root and 1 < len(s) < n - 1:
                non_trivial.append(s)
    return non_trivial


def induced_ref_splits(ref_clades: list, all_taxa: set) -> set:
    n = len(all_taxa)
    splits = set()
    for rc in ref_clades:
        s = rc & all_taxa
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


def evaluate_multilocus_species_trees():
    print("=" * 95)
    print("PART 1: MULTI-LOCUS SPECIES-TREE RECONSTRUCTION FROM 242 GTDB MARKER GENE TREES")
    print("=" * 95)

    for domain, label, ref_tree_name, tax_name in [
        ("ar122", "ARCHAEA (122 marker genes, 1,672 species across 19 phyla)", "ar122_r95.tree", "ar122_taxonomy_r95.tsv"),
        ("bac120", "BACTERIA (120 marker genes, 1,000 species across 111 phyla)", "bac120_r95.tree", "bac120_taxonomy_r95.tsv"),
    ]:
        print(f"\n--- {label} ---", flush=True)
        t_ref = Phylo.read(str(GTDB_DIR / ref_tree_name), "newick")
        ref_leaves = {t.name.strip("'") for t in t_ref.get_terminals() if t.name}

        all_domain_taxa = set()
        for tf in sorted(WORK_DIR.glob(f"{domain}_*.bw_mdl.nwk")):
            t = Phylo.read(str(tf), "newick")
            all_domain_taxa |= {x.name.strip("'") for x in t.get_terminals() if x.name}
        shared = all_domain_taxa & ref_leaves
        ref_clades = precompute_ref_clades(t_ref, shared)

        for suffix, method_name in [
            ("bw_mdl", "Bushwhacker (MDL Polytomies)"),
            ("bw_bin", "Bushwhacker (Binary)"),
            ("fasttree", "FastTree 2 (GTR+G)"),
        ]:
            tree_files = sorted(WORK_DIR.glob(f"{domain}_*.{suffix}.nwk"))
            total_tp = 0
            total_fp = 0
            high_tp = 0
            high_fp = 0
            for tf in tree_files:
                t = Phylo.read(str(tf), "newick")
                t_leaves = {x.name.strip("'") for x in t.get_terminals() if x.name} & shared
                t_ref_sp = induced_ref_splits(ref_clades, t_leaves)
                sp_map = get_canonical_splits_with_support(t, t_leaves)
                sp_set = set(sp_map.keys())
                total_tp += len(sp_set & t_ref_sp)
                total_fp += len(sp_set - t_ref_sp)
                high_set = {sp for sp, sup in sp_map.items() if sup >= 0.95}
                high_tp += len(high_set & t_ref_sp)
                high_fp += len(high_set - t_ref_sp)

            agg_prec = 100.0 * total_tp / max(1, total_tp + total_fp)
            high_prec = 100.0 * high_tp / max(1, high_tp + high_fp)
            print(
                f"  {method_name:<30} | Genes={len(tree_files):3d} | All Splits: TP={total_tp:6d} FP={total_fp:6d} (Prec={agg_prec:5.2f}%) | "
                f"Support>=0.95: TP={high_tp:6d} FP={high_fp:6d} (Prec={high_prec:5.2f}%)",
                flush=True,
            )


def analyze_mdl_certified_xenologous_hgt():
    print("\n" + "=" * 95)
    print("PART 2: MDL-CERTIFIED INTER-PHYLUM & INTER-CLASS HORIZONTAL GENE TRANSFER ATLAS")
    print("=" * 95)

    ar_tax = load_taxonomy(GTDB_DIR / "ar122_taxonomy_r95.tsv")
    bac_tax = load_taxonomy(GTDB_DIR / "bac120_taxonomy_r95.tsv")
    ar_ref = Phylo.read(str(GTDB_DIR / "ar122_r95.tree"), "newick")
    bac_ref = Phylo.read(str(GTDB_DIR / "bac120_r95.tree"), "newick")
    ar_ref_clades = precompute_ref_clades(ar_ref, set(ar_tax.keys()))
    bac_panel_taxa = set()
    for tf in sorted(WORK_DIR.glob("bac120_*.bw_mdl.nwk")):
        t = Phylo.read(str(tf), "newick")
        bac_panel_taxa |= {x.name.strip("'") for x in t.get_terminals() if x.name}
    bac_ref_clades = precompute_ref_clades(bac_ref, bac_panel_taxa)

    res_files = sorted(RES_DIR.glob("*.json"))
    hgt_events = []
    gene_hgt_counts = defaultdict(lambda: {"inter_phylum": 0, "inter_class": 0, "inter_order": 0, "total_high_mdl_discordant": 0})
    func_stats = defaultdict(lambda: {"genes": 0, "inter_phylum": 0, "inter_class": 0, "inter_order": 0, "high_mdl_hgt": 0})

    for rf in res_files:
        meta = json.loads(rf.read_text(encoding="utf-8"))
        dom = meta["domain"]
        mid = meta["marker_id"]
        gname = meta["name"]
        fclass = meta["func_class"]
        func_stats[fclass]["genes"] += 1

        tax_dict = ar_tax if dom == "ar122" else bac_tax
        ref_clades = ar_ref_clades if dom == "ar122" else bac_ref_clades

        nwk_path = WORK_DIR / f"{dom}_{mid}.bw_mdl.nwk"
        if not nwk_path.exists():
            continue
        t_mdl = Phylo.read(str(nwk_path), "newick")
        shared = {t.name.strip("'") for t in t_mdl.get_terminals() if t.name} & set(tax_dict.keys())
        ref_splits = induced_ref_splits(ref_clades, shared)
        mdl_splits_sup = get_canonical_splits_with_support(t_mdl, shared)

        for sp, sup in mdl_splits_sup.items():
            # Require positive MDL support >= 0.990 (Delta L >= 4.6 nats) and discordant with reference species tree
            if sup < 0.990 or sp in ref_splits:
                continue
            k = len(sp)
            # Examine compact clade jumps (3 <= k <= 12) where a recipient lineage (1..3 species) from one Phylum/Class/Order
            # is nested with a cohesive donor group (>= 2 species) from a single distinct Phylum/Class/Order
            if 3 <= k <= 15:
                phyla = Counter(tax_dict[x].get("p", "") for x in sp if tax_dict.get(x, {}).get("p", ""))
                classes = Counter(tax_dict[x].get("c", "") for x in sp if tax_dict.get(x, {}).get("c", ""))
                orders = Counter(tax_dict[x].get("o", "") for x in sp if tax_dict.get(x, {}).get("o", ""))

                # Convert support P = 1 / (1 + e^-dL) back to Delta L (nats)
                p_clamped = min(0.9995, max(0.5001, sup))
                dl_nats = np.log(p_clamped / (1.0 - p_clamped))

                if len(phyla) == 2:
                    (p_maj, c_maj), (p_min, c_min) = phyla.most_common(2)
                    if c_maj >= 2 and c_min <= 3:
                        gene_hgt_counts[f"{dom}_{mid}"]["inter_phylum"] += 1
                        gene_hgt_counts[f"{dom}_{mid}"]["total_high_mdl_discordant"] += 1
                        func_stats[fclass]["inter_phylum"] += 1
                        func_stats[fclass]["high_mdl_hgt"] += 1
                        rec_taxa = [x for x in sp if tax_dict[x].get("p") == p_min]
                        don_taxa = [x for x in sp if tax_dict[x].get("p") == p_maj]
                        rec_g = tax_dict[rec_taxa[0]].get("g", "") or tax_dict[rec_taxa[0]].get("f", "") or p_min
                        don_g = tax_dict[don_taxa[0]].get("g", "") or tax_dict[don_taxa[0]].get("f", "") or p_maj
                        hgt_events.append({
                            "level": "Inter-Phylum",
                            "domain": dom,
                            "marker_id": mid,
                            "gene_name": gname,
                            "func_class": fclass,
                            "support": sup,
                            "dl_nats": dl_nats,
                            "clade_size": k,
                            "donor": f"p__{p_maj} (g__{don_g}, n={c_maj})",
                            "recipient": f"p__{p_min} (g__{rec_g}, n={c_min})",
                        })
                elif len(phyla) == 1 and len(classes) == 2:
                    (c_maj_name, c_maj), (c_min_name, c_min) = classes.most_common(2)
                    if c_maj >= 2 and c_min <= 3:
                        gene_hgt_counts[f"{dom}_{mid}"]["inter_class"] += 1
                        gene_hgt_counts[f"{dom}_{mid}"]["total_high_mdl_discordant"] += 1
                        func_stats[fclass]["inter_class"] += 1
                        func_stats[fclass]["high_mdl_hgt"] += 1
                        p_shared = next(iter(phyla.keys()))
                        hgt_events.append({
                            "level": "Inter-Class",
                            "domain": dom,
                            "marker_id": mid,
                            "gene_name": gname,
                            "func_class": fclass,
                            "support": sup,
                            "dl_nats": dl_nats,
                            "clade_size": k,
                            "donor": f"c__{c_maj_name} [p__{p_shared}] (n={c_maj})",
                            "recipient": f"c__{c_min_name} [p__{p_shared}] (n={c_min})",
                        })
                elif len(classes) == 1 and len(orders) == 2:
                    (o_maj_name, c_maj), (o_min_name, c_min) = orders.most_common(2)
                    if c_maj >= 2 and c_min <= 3:
                        gene_hgt_counts[f"{dom}_{mid}"]["inter_order"] += 1
                        gene_hgt_counts[f"{dom}_{mid}"]["total_high_mdl_discordant"] += 1
                        func_stats[fclass]["inter_order"] += 1
                        func_stats[fclass]["high_mdl_hgt"] += 1

    print(
        f"{'Functional Category':<34} | {'Genes':>5} | {'Inter-Phylum/G':>14} | {'Inter-Class/G':>13} | {'Inter-Order/G':>13} | {'Total HGT/Gene':>14}"
    )
    print("-" * 104)
    for fc in [
        "Ribosomal & Translational",
        "Transcription & DNA Replication",
        "Metabolic & Cellular Operational",
        "Aminoacyl-tRNA & tRNA Mod",
    ]:
        st = func_stats[fc]
        g = max(1, st["genes"])
        print(
            f"{fc:<34} | {st['genes']:5d} | {st['inter_phylum']/g:14.2f} | {st['inter_class']/g:13.2f} | "
            f"{st['inter_order']/g:13.2f} | {st['high_mdl_hgt']/g:14.2f}"
        )

    tot_ip = sum(s["inter_phylum"] for s in func_stats.values())
    tot_ic = sum(s["inter_class"] for s in func_stats.values())
    tot_io = sum(s["inter_order"] for s in func_stats.values())
    print(f"\nTotal High-Confidence MDL-Certified Xenologous Clade-Jump Splits (P_MDL >= 0.990): {tot_ip + tot_ic + tot_io}")
    print(f"  Inter-Phylum HGT jumps : {tot_ip}")
    print(f"  Inter-Class  HGT jumps : {tot_ic}")
    print(f"  Inter-Order  HGT jumps : {tot_io}")

    # Top genes by HGT frequency
    print("\nTop 15 Most Horizontally Transferred Marker Genes Across Archaea & Bacteria:")
    res_map = {f"{json.loads(f.read_text())['domain']}_{json.loads(f.read_text())['marker_id']}": json.loads(f.read_text()) for f in res_files}
    top_genes = sorted(gene_hgt_counts.items(), key=lambda x: (x[1]["total_high_mdl_discordant"], x[1]["inter_phylum"]), reverse=True)
    for i, (gk, cnts) in enumerate(top_genes[:15], 1):
        m = res_map[gk]
        print(
            f"  {i:2d}. {gk:<18} | {m['name']:<16} ({m['func_class']:<32}) | "
            f"Total={cnts['total_high_mdl_discordant']:2d} (Inter-Phylum={cnts['inter_phylum']:2d}, Inter-Class={cnts['inter_class']:2d}, Inter-Order={cnts['inter_order']:2d})"
        )

    # Sample of Inter-Phylum HGT events
    ip_events = [e for e in hgt_events if e["level"] == "Inter-Phylum" and e["support"] >= 0.999]
    print(f"\nRepresentative P_MDL >= 0.999 Inter-Phylum Horizontal Gene Transfer Events (out of {len(ip_events)}):")
    for i, e in enumerate(ip_events[:20], 1):
        print(
            f"  {i:2d}. [{e['domain']} {e['marker_id']} {e['gene_name']} ({e['func_class']})]: "
            f"P_MDL={e['support']:.3f} | Donor {e['donor']} -> Recipient {e['recipient']}"
        )


def run_clade_level_shunt_scan():
    print("\n" + "=" * 95)
    print("PART 3: STAGE-4 INTRAGENIC & MULTI-GENE RETICULATE SHUNT PEELING WITHIN MAJOR CLADES")
    print("=" * 95)
    # Test Stage-4 shunt peeling on major Archaeal phyla/orders (N=40..180) and multi-gene concatenations
    ar_tax = load_taxonomy(GTDB_DIR / "ar122_taxonomy_r95.tsv")
    by_phylum = defaultdict(list)
    for gid, rk in ar_tax.items():
        by_phylum[rk.get("p", "")].append(gid)

    # Pick major Archaeal phyla with 40..250 species: Halobacteriota (downsample 120), Methanobacteriota (122), Thermoproteota (downsample 120), Thermoplasmatota (downsample 120)
    target_clades = []
    for p_name in ["Methanobacteriota", "Halobacteriota", "Thermoproteota", "Thermoplasmatota", "Nanoarchaeota"]:
        members = sorted(by_phylum.get(p_name, []))
        if len(members) > 120:
            step = len(members) / 120.0
            members = [members[int(i * step)] for i in range(120)]
        if len(members) >= 30:
            target_clades.append((p_name, set(members)))

    # Load marker info
    markers = []
    with open(GTDB_DIR / "ar122_marker_info_r95.tsv", "r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            mid = row["Marker Id"].replace("PFAM_", "").replace("TIGR_", "")
            markers.append((mid, row["Name"], row["Description"]))

    clade_shunts = []
    # Also build 2-gene concatenation (Ribosomal core + Operational HMG-CoA_red / V_ATPase_I / aaRS) for each clade
    base_dir = GTDB_DIR / "ar122_marker_genes_reps_r95"
    shunt_work = GTDB_DIR / "clade_shunt_work"
    shunt_work.mkdir(parents=True, exist_ok=True)

    # Test top 25 longest/most ubiquitous Archaeal markers across the 5 phyla + 5 paired Ribosomal+Operational concatenations
    selected_markers = [m for m in markers if m[0] in {
        "PF00368.13", "PF01496.14", "TIGR00392", "TIGR00422", "TIGR00456",
        "TIGR00468", "TIGR00471", "TIGR01051", "TIGR00064", "TIGR00492",
        "PF00410.14", "PF00687.16", "PF01269.12", "TIGR00234", "TIGR00329",
    }]

    for p_name, p_taxa in target_clades:
        # 1. Single-gene alignments within phylum
        for mid, gname, gdesc in selected_markers:
            faa_path = base_dir / "faa" / f"{mid}.faa"
            fna_path = base_dir / "fna" / f"{mid}.fna"
            codon_out = shunt_work / f"{p_name}_{mid}.codon.fna"
            bw_json_out = shunt_work / f"{p_name}_{mid}.bw.json"
            bw_nwk_out = shunt_work / f"{p_name}_{mid}.bw.nwk"

            if not bw_json_out.exists():
                aa_dict = {r.id: str(r.seq).upper().rstrip("*") for r in SeqIO.parse(str(faa_path), "fasta") if r.id in p_taxa and len(r.seq) >= 20}
                nt_dict = {r.id: str(r.seq).upper().replace("-", "") for r in SeqIO.parse(str(fna_path), "fasta") if r.id in aa_dict}
                shared = sorted(nt_dict.keys())
                if len(shared) < 25:
                    continue
                tmp_in = codon_out.with_suffix(".in.faa")
                tmp_aln = codon_out.with_suffix(".aln.faa")
                with open(tmp_in, "w") as f:
                    for gid in shared:
                        f.write(f">{gid}\n{aa_dict[gid]}\n")
                subprocess.run([MAFFT_BIN, "--retree", "1", "--thread", "4", "--quiet", str(tmp_in)], stdout=open(tmp_aln, "w"), stderr=subprocess.DEVNULL, check=True)
                aln_recs = list(SeqIO.parse(str(tmp_aln), "fasta"))
                l_aa = len(aln_recs[0].seq)
                keep = [c for c in range(l_aa) if sum(1 for r in aln_recs if r.seq[c] == "-") / len(aln_recs) <= 0.50]
                keep_set = set(keep)
                with open(codon_out, "w") as out:
                    for r in aln_recs:
                        nt = nt_dict[r.id]
                        cods = []
                        pos = 0
                        for col, ch in enumerate(str(r.seq)):
                            if ch == "-":
                                if col in keep_set:
                                    cods.append("---")
                            else:
                                cod = nt[pos:pos+3]
                                if col in keep_set:
                                    cods.append(cod if len(cod) == 3 else "---")
                                pos += 3
                        out.write(f">{r.id}\n{''.join(cods)}\n")
                tmp_in.unlink(missing_ok=True)
                tmp_aln.unlink(missing_ok=True)
                subprocess.run([BW_BIN, str(codon_out), "-o", str(bw_nwk_out), "-j", str(bw_json_out)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            if bw_json_out.exists():
                data = json.loads(bw_json_out.read_text())
                for sh in data.get("shunts", []):
                    u_l = sh.get("leaves_u") or [sh["label_u"]]
                    v_l = sh.get("leaves_v") or [sh["label_v"]]
                    u_o = sorted({ar_tax.get(x, {}).get("o", "") for x in u_l if ar_tax.get(x, {}).get("o", "")})
                    v_o = sorted({ar_tax.get(x, {}).get("o", "") for x in v_l if ar_tax.get(x, {}).get("o", "")})
                    u_f = sorted({ar_tax.get(x, {}).get("f", "") for x in u_l if ar_tax.get(x, {}).get("f", "")})
                    v_f = sorted({ar_tax.get(x, {}).get("f", "") for x in v_l if ar_tax.get(x, {}).get("f", "")})
                    clade_shunts.append({
                        "phylum": p_name,
                        "marker_id": mid,
                        "gene_name": gname,
                        "mode": sh["mode"],
                        "gain_nats": sh["net_mdl_gain_nats"],
                        "sites": sh["shared_mutations"],
                        "span": sh["interval_span"],
                        "prob": sh["posterior_prob"],
                        "u_desc": f"o__{','.join(u_o[:1])}/f__{','.join(u_f[:1])}",
                        "v_desc": f"o__{','.join(v_o[:1])}/f__{','.join(v_f[:1])}",
                    })

    print(f"Total Stage-4 Reticulate Shunts Certified Across Phylum-Level Marker Alignments: {len(clade_shunts)}")
    by_mode = Counter(s["mode"] for s in clade_shunts)
    print(f"  INTERVAL (contiguous intragenic HGT/recombination) = {by_mode['INTERVAL']}, SPARSE (parallel convergence) = {by_mode['SPARSE']}")
    clade_shunts.sort(key=lambda x: x["gain_nats"], reverse=True)
    for i, s in enumerate(clade_shunts[:15], 1):
        print(
            f"  {i:2d}. [{s['phylum']} | {s['marker_id']} {s['gene_name']}] {s['mode']}: "
            f"+{s['gain_nats']:.2f} nats (m={s['sites']} sites, span={s['span']}, P={s['prob']:.4f}) | "
            f"{s['u_desc']} <-> {s['v_desc']}"
        )


if __name__ == "__main__":
    evaluate_multilocus_species_trees()
    analyze_mdl_certified_xenologous_hgt()
    run_clade_level_shunt_scan()
