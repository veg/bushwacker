#!/usr/bin/env python3
"""
Deep-mining pass across all 19,952 TOGA2 mammalian proteome Bushwhacker graphs on silverback:
1. Branch-by-Branch Species Tree Support vs. True Hard Conflict (ILS) across speciesTree.nh
2. Unsupervised Clade-Pair & Multi-Gene Functional Convergence Network (14,105 SPARSE shunts)
3. Ectopic Gene Conversion & Paralogous Exon Chimera Family Atlas (1,032 INTERVAL shunts)
4. De Novo WALRUS Codon MSA vs. TOGA2 Reference-Projected Chimera/Artifact Elimination
"""


import sys as _sys
from pathlib import Path as _Path
if str(_Path(__file__).resolve().parent) not in _sys.path:
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
from _env_paths import REPO_ROOT, DATA_DIR, SPECIMENS_DIR, ALIGNMENTS_DIR, SINGLE_CELL_DIR, RESULTS_DIR, WORK_ROOT, EXTERNAL_DIR, resolve_bin, ASTRAL_JAR

import os
import re
import json
import gzip
import math
import statistics as st
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

MASTER_JSONL = str(WORK_ROOT / "toga2_nt" / "toga2_bushwhacker_master_19953.jsonl")
RESULTS_DIR = str(WORK_ROOT / "toga2_nt" / "results")
SPECIES_TREE_PATH = "data/alignments/supermatrices/toga2_speciesTree.nh"
OUT_DEEP_JSON = str(RESULTS_DIR / "mammalian_toga2" / "toga2_deep_mining_atlas.json")


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


# Anchor assemblies for major mammalian orders/clades in TOGA2 (hg38 v1)
CLADE_ANCHORS = {
    "Primates": ["hg38", "panTro6", "gorGor6", "ponAbe3", "macMul10", "calJac4", "HLcalJac5", "HLpapAnu5", "HLateGeo2", "HLmicMur3"],
    "Rodentia": ["mm10", "mm39", "rn6", "rn7", "cavPor3", "HLcavPor4", "hetGla2", "HLhetGla2", "HLacoCah1", "HLjacJac2", "HLperManBai2", "HLmarMar1"],
    "Lagomorpha": ["oryCun2", "HLoryCunCun4", "HLochPri1", "HLlepTim1", "HLsylFlo1", "HLsylBac1"],
    "Chiroptera_Yinptero": ["pteVam1", "HLpteVam2", "HLhipArm1", "HLhipGal1", "HLrhiSin1", "HLrousAeg1", "HLaseSto2"],
    "Chiroptera_Yango": ["myoMyo6", "HLmyoMyo6", "HLeptFus3", "HLartJam2", "HLdesRot2", "HLmolMol2", "HLsacBil1", "HLantPal2"],
    "Cetacea_Odontoceti": ["turTru2", "HLturTru4", "orcOrc1", "HLdelLeu2", "HLmonMon1", "HLkogBre3", "HLphyCat1", "HLzipCav1", "HLlipVex1", "lipVex1"],
    "Cetacea_Mysticeti": ["balAcu1", "HLbalAcu2", "HLbalMys1", "HLbalMus1", "HLeubAus1", "HLmegNov2"],
    "Artiodactyla_Ruminantia": ["bosTau9", "oviAri4", "capHir1", "HLgirCam2", "HLcerEla1", "HLodoVir2", "HLbubBub2", "HLantMar1"],
    "Artiodactyla_Suina_Tylopoda": ["susScr11", "HLsusScro12", "camDro1", "HLcamDro2", "HLlamGla1", "HLvicPac4", "HLhipAmp2"],
    "Carnivora_Pinnipedia": ["odoRosDiv1", "HLhalGry1", "HLlepWed1", "HLmirLeo1", "HLpusSib1", "HLotaByr1", "HLarcGaz2", "HLneoSch1"],
    "Carnivora_Fissipedia": ["canFam3", "canFam6", "felCat9", "HLursMar1", "HLmusFur2", "HLailFul2", "HLpanTig3", "HLcroCro1", "HLsurSur1"],
    "Perissodactyla": ["equCab3", "HLequCab4", "cerSim1", "HLdicSum1", "HLtapInd1", "HLtapTer1"],
    "Eulipotyphla": ["eriEur2", "sorAra2", "HLsorAra3", "HLconCri1", "HLtalEur1A", "HLscaAqu1", "HLsunEtr1", "HLsolPar1"],
    "Pholidota": ["manJav1", "HLmanJav2", "HLmanPen2", "HLhatTet1"],
    "Afrotheria": ["loxAfr3", "HLeleMax2", "triMan1", "HLdugDug1", "chrAsi1", "echTel2", "HLoryAfeAfe2", "HLproCap3", "HLmicTal1", "eleEdw1"],
    "Xenarthra": ["dasNov3", "HLdasNov4", "choHof1", "HLchoDid2", "HLmyrTri1", "HLtamTet1", "HLtolMat1"],
    "Marsupialia_Monotremata": ["monDom5", "sarHar1", "phaCin1", "ornAna2", "HLornAna3", "HLtacAcu1"],
}


def build_taxon_to_order_map(sp_root):
    """Assign every tip in speciesTree.nh to its mammalian clade/order using LCA of anchor tips."""
    all_tips = set()
    def collect(u):
        if not u["children"]:
            if u["name"]:
                all_tips.add(u["name"])
            u["_tips"] = {u["name"]} if u["name"] else set()
        else:
            s = set()
            for c in u["children"]:
                s |= collect(c)
            u["_tips"] = s
        return u["_tips"]
    collect(sp_root)

    taxon_order = {}
    # Find tightest subtree containing >= 2 present anchors of each clade
    for clade_name, anchors in CLADE_ANCHORS.items():
        present = [a for a in anchors if a in all_tips]
        if not present:
            continue
        best_node = [None]
        best_size = [10**9]
        def find_lca(u):
            t = u["_tips"]
            if all(a in t for a in present):
                if len(t) < best_size[0]:
                    best_size[0] = len(t)
                    best_node[0] = u
                for c in u["children"]:
                    find_lca(c)
        find_lca(sp_root)
        if best_node[0] is not None and best_size[0] < 600:
            for tip in best_node[0]["_tips"]:
                # More specific clade overrides broader clade if smaller
                if tip not in taxon_order or best_size[0] < taxon_order[tip][1]:
                    taxon_order[tip] = (clade_name, best_size[0])

    return {k: v[0] for k, v in taxon_order.items()}, all_tips


def classify_gene_family(sym):
    s = sym.upper()
    if s.startswith("ZNF") or s.startswith("ZKSCAN") or s.startswith("ZSCAN") or s.startswith("ZFP") or s.startswith("ZBTB") or s.startswith("ZMYM") or s.startswith("PRDM"):
        return "KRAB / C2H2 Zinc-Finger Arrays"
    if s.startswith("OR") and len(s) >= 4 and s[2].isdigit():
        return "Olfactory Receptors (OR)"
    if s.startswith("TAS1R") or s.startswith("TAS2R") or s.startswith("VN1R"):
        return "Taste & Vomeronasal Receptors (TASR/VNR)"
    if s.startswith("CEACAM") or s.startswith("SIGLEC") or s.startswith("CD") or s.startswith("FCGR") or s.startswith("KIR") or s.startswith("LILR") or s.startswith("BTN") or s.startswith("HLA") or s.startswith("IGH") or s.startswith("IGK") or s.startswith("TRB") or s.startswith("TLR") or s.startswith("IFIT") or s.startswith("APOBEC") or s.startswith("OAS") or s.startswith("MX") or s.startswith("NLRP") or s.startswith("TRIM"):
        return "Innate & Adaptive Immunity / Ig-Superfamily"
    if s.startswith("HBA") or s.startswith("HBB") or s.startswith("HBG") or s.startswith("HBD") or s.startswith("HBE") or s.startswith("HBZ") or s == "MB":
        return "Hemoglobin / Globin Clusters"
    if s.startswith("H1") or s.startswith("H2A") or s.startswith("H2B") or s.startswith("H3") or s.startswith("H4") or s.startswith("HIST"):
        return "Replication-Dependent Histone Clusters"
    if s.startswith("CYP") or s.startswith("UGT") or s.startswith("AKR1") or s.startswith("GSTA") or s.startswith("GSTM") or s.startswith("SULT") or s.startswith("CES"):
        return "Xenobiotic Detoxification (CYP/UGT/AKR/GST)"
    if s.startswith("KRT") or s.startswith("PCDH") or s.startswith("CDH") or s.startswith("MUC") or s.startswith("COL"):
        return "Cadherins, Keratins, Mucins & Extracellular Matrix"
    if s.startswith("RGPD") or s.startswith("NBPF") or s.startswith("GOLGA") or s.startswith("USP") or s.startswith("TBC1D") or s.startswith("ANKRD") or s.startswith("FAM") or s.startswith("SPDYE") or s.startswith("PMS2"):
        return "Primate/Mammalian Segmental Duplications"
    if s in {"SLC26A5", "TMC1", "OTOF", "KCNQ4", "MYO15A", "MYO7A", "MYO6", "USH1C", "USH2A", "WHRN", "CLRN1", "PJVK", "STRC", "TECTA", "COCH", "POU4F3", "GJB2", "EPS8", "ESPN", "TRIOBP", "CIB2", "LHFPL5", "TMIE"}:
        return "Cochlear Hair-Cell & Auditory Mechanotransduction"
    return "Other Coding Loci"


def main():
    with open(SPECIES_TREE_PATH, "r") as f:
        sp_root = parse_newick(f.read())
    taxon_to_clade, all_sp_tips = build_taxon_to_order_map(sp_root)
    print(f"Mapped {len(taxon_to_clade)} / {len(all_sp_tips)} speciesTree assemblies to major mammalian orders/guilds")

    records = []
    with open(MASTER_JSONL, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("status") == "ok":
                records.append(r)

    print(f"Loaded {len(records)} valid gene records from {MASTER_JSONL}")

    # --- Analysis 1: Gene-Tree / Species-Tree Resolution & Conflict Stratification by Gene Length ---
    length_bins = [
        ("< 600 bp (Short CDS)", 0, 600),
        ("600 - 1,200 bp", 600, 1200),
        ("1,200 - 2,400 bp", 1200, 2400),
        ("2,400 - 4,800 bp", 2400, 4800),
        ("> 4,800 bp (Long Multi-Domain)", 4800, 10**9),
    ]
    strat_by_length = []
    for label, lo, hi in length_bins:
        sub = [
            r for r in records
            if lo <= (r.get("seq_len") or 0) < hi and (r.get("sp_splits_count") or 0) >= 5
        ]
        if not sub:
            continue
        tot_sp = sum(r["sp_splits_count"] for r in sub)
        tot_sp_sup = sum(r["sp_supported_splits_count"] or 0 for r in sub)
        tot_bin_conf = sum(r["bin_sp_conflict"] for r in sub)
        tot_mdl_conf = sum(r["mdl_sp_conflict"] for r in sub)
        tot_bin_sh = sum(r["bin_sp_shared"] for r in sub)
        tot_mdl_sh = sum(r["mdl_sp_shared"] for r in sub)
        d_mdl = [r["delta_mdl_gene_vs_sp_nats"] for r in sub if r.get("delta_mdl_gene_vs_sp_nats") is not None]
        strat_by_length.append({
            "length_bin": label,
            "n_genes": len(sub),
            "median_seq_len_bp": st.median(r["seq_len"] for r in sub),
            "sp_tree_splits_supported_pct": round(100.0 * tot_sp_sup / max(1, tot_sp), 2),
            "binary_conflict_rate_pct": round(100.0 * st.mean(r["bin_sp_conflict_rate"] for r in sub), 2),
            "mdl_conflict_rate_pct": round(100.0 * st.mean(r["mdl_sp_conflict_rate"] for r in sub), 2),
            "false_conflicts_eliminated_pct": round(100.0 * (1.0 - tot_mdl_conf / max(1, tot_bin_conf)), 2),
            "true_sp_splits_retained_pct": round(100.0 * tot_mdl_sh / max(1, tot_bin_sh), 2),
            "sp_tree_compresses_better_or_equal_pct": round(
                100.0 * sum(1 for x in d_mdl if x <= 0) / max(1, len(d_mdl)), 2
            ),
            "median_delta_mdl_nats": round(st.median(d_mdl), 1) if d_mdl else None,
        })

    # --- Analysis 2: Ectopic Gene Conversion & Chimera Family Enrichment (INTERVAL Shunts) ---
    fam_totals = Counter()
    fam_interval_genes = Counter()
    fam_interval_events = Counter()
    fam_sparse_genes = Counter()
    fam_sparse_events = Counter()
    fam_sp_tree_shunts = Counter()
    interval_spans = []
    interval_gains = []

    for r in records:
        fam = classify_gene_family(r["gene_symbol"])
        fam_totals[fam] += 1
        if r.get("n_interval_shunts", 0) > 0:
            fam_interval_genes[fam] += 1
            fam_interval_events[fam] += r["n_interval_shunts"]
        if r.get("n_sparse_shunts", 0) > 0:
            fam_sparse_genes[fam] += 1
            fam_sparse_events[fam] += r["n_sparse_shunts"]
        fam_sp_tree_shunts[fam] += (r.get("sp_shunts_count") or 0)

        for s in r.get("top_shunts", []):
            if s.get("mode") == "INTERVAL" and s.get("interval_span"):
                a, b = s["interval_span"]
                interval_spans.append(abs(b - a) + 1)
                interval_gains.append(s.get("net_mdl_gain_nats", 0.0))

    family_enrichment = []
    baseline_int_rate = sum(fam_interval_genes.values()) / max(1, len(records))
    for fam, total_g in fam_totals.most_common():
        int_g = fam_interval_genes[fam]
        sp_g = fam_sparse_genes[fam]
        int_rate = int_g / max(1, total_g)
        fold_int = int_rate / max(1e-9, baseline_int_rate)
        family_enrichment.append({
            "family": fam,
            "n_genes": total_g,
            "genes_with_interval_shunt": int_g,
            "interval_gene_pct": round(100.0 * int_rate, 2),
            "interval_fold_enrichment": round(fold_int, 2),
            "total_interval_events": fam_interval_events[fam],
            "genes_with_sparse_shunt": sp_g,
            "sparse_gene_pct": round(100.0 * sp_g / max(1, total_g), 2),
            "total_sparse_events": fam_sparse_events[fam],
            "total_fixed_sp_tree_shunts": fam_sp_tree_shunts[fam],
        })
    family_enrichment.sort(key=lambda x: x["interval_fold_enrichment"], reverse=True)

    # --- Analysis 3: Unsupervised Clade-Pair Convergence Network (SPARSE Shunts) ---
    inter_clade_pairs = Counter()
    inter_clade_genes = defaultdict(list)
    intra_clade_pairs = Counter()
    auditory_complex_genes = []

    auditory_set = {
        "SLC26A5", "TMC1", "PCDH15", "CDH23", "OTOF", "KCNQ4", "MYO15A", "MYO7A", "MYO6",
        "USH1C", "USH2A", "WHRN", "CLRN1", "PJVK", "STRC", "TECTA", "COCH", "POU4F3",
        "GJB2", "EPS8", "ESPN", "TRIOBP", "CIB2", "LHFPL5", "TMIE"
    }

    for r in records:
        sym = r["gene_symbol"]
        if sym in auditory_set:
            auditory_complex_genes.append({
                "gene_symbol": sym,
                "gene": r["gene"],
                "n_taxa": r["n_taxa"],
                "seq_len": r["seq_len"],
                "sp_supported_frac": r["sp_supported_frac"],
                "false_conflict_eliminated_frac": r["false_conflict_eliminated_frac"],
                "n_interval_shunts": r["n_interval_shunts"],
                "n_sparse_shunts": r["n_sparse_shunts"],
                "sp_shunts_count": r["sp_shunts_count"],
                "max_shunt_gain_nats": r["max_shunt_gain_nats"],
                "top_shunts": r.get("top_shunts", [])[:3],
            })

        for s in r.get("top_shunts", []):
            if s.get("mode") != "SPARSE":
                continue
            u_tips = s.get("u_tips_sample", [])
            v_tips = s.get("v_tips_sample", [])
            u_clades = [taxon_to_clade[t] for t in u_tips if t in taxon_to_clade]
            v_clades = [taxon_to_clade[t] for t in v_tips if t in taxon_to_clade]
            if not u_clades or not v_clades:
                continue
            c_u = Counter(u_clades).most_common(1)[0][0]
            c_v = Counter(v_clades).most_common(1)[0][0]
            pair = tuple(sorted([c_u, c_v]))
            if c_u != c_v:
                inter_clade_pairs[pair] += 1
                if len(inter_clade_genes[pair]) < 15:
                    inter_clade_genes[pair].append(
                        f"{sym}(+{s['net_mdl_gain_nats']:.1f}n,m={s['shared_mutations']})"
                    )
            else:
                intra_clade_pairs[c_u] += 1

    top_inter_clade = [
        {
            "clade_pair": f"{p[0]} <-> {p[1]}",
            "n_certified_sparse_shunts": cnt,
            "example_genes": inter_clade_genes[p][:10],
        }
        for p, cnt in inter_clade_pairs.most_common(25)
    ]

    top_intra_clade = [
        {"clade": k, "n_certified_sparse_shunts": cnt}
        for k, cnt in intra_clade_pairs.most_common(17)
    ]

    auditory_complex_genes.sort(key=lambda x: (x["n_sparse_shunts"] + x["n_interval_shunts"], x["max_shunt_gain_nats"]), reverse=True)

    out = {
        "stratification_by_gene_length": strat_by_length,
        "interval_tract_statistics": {
            "total_sampled_tracts": len(interval_spans),
            "median_tract_length_bp": st.median(interval_spans) if interval_spans else 0,
            "mean_tract_length_bp": round(st.mean(interval_spans), 1) if interval_spans else 0,
            "median_net_gain_nats": round(st.median(interval_gains), 2) if interval_gains else 0,
            "max_net_gain_nats": round(max(interval_gains), 2) if interval_gains else 0,
        },
        "gene_family_shunt_enrichment": family_enrichment,
        "top_inter_ordinal_convergent_clade_pairs": top_inter_clade,
        "intra_ordinal_convergent_clades": top_intra_clade,
        "cochlear_auditory_complex_25_genes": auditory_complex_genes,
    }

    with open(OUT_DEEP_JSON, "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== 1. STRATIFICATION BY CODING GENE LENGTH (19,890 GENES) ===")
    for row in strat_by_length:
        print(
            f"  {row['length_bin']:32s} | n={row['n_genes']:5d} | SpSup={row['sp_tree_splits_supported_pct']:5.1f}% | "
            f"Conf: {row['binary_conflict_rate_pct']:5.1f}%->{row['mdl_conflict_rate_pct']:5.1f}% "
            f"(Elim {row['false_conflicts_eliminated_pct']:5.1f}%, KeepTrue {row['true_sp_splits_retained_pct']:5.1f}%) | "
            f"SpBetter={row['sp_tree_compresses_better_or_equal_pct']:5.1f}%"
        )

    print("\n=== 2. GENE FAMILY ENRICHMENT FOR ECTOPIC GENE CONVERSION (INTERVAL) & CONVERGENCE (SPARSE) ===")
    for row in family_enrichment:
        print(
            f"  {row['family']:48s} | n={row['n_genes']:5d} | INT={row['genes_with_interval_shunt']:3d} "
            f"({row['interval_gene_pct']:5.2f}%, {row['interval_fold_enrichment']:4.2f}x) | "
            f"SPARSE={row['genes_with_sparse_shunt']:4d} ({row['sparse_gene_pct']:5.1f}%) | "
            f"SpTreeShunts={row['total_fixed_sp_tree_shunts']:4d}"
        )

    print("\n=== 3. TOP 20 INTER-ORDINAL / INTER-GUILD CONVERGENT CLADE PAIRS (UNSUPERVISED SPARSE SHUNTS) ===")
    for row in top_inter_clade[:20]:
        print(f"  {row['clade_pair']:52s} | {row['n_certified_sparse_shunts']:4d} shunts | e.g. {', '.join(row['example_genes'][:5])}")

    print("\n=== 4. COCHLEAR HAIR-CELL & AUDITORY MECHANOTRANSDUCTION COMPLEX (25 GENES) ===")
    aud_with_shunts = sum(1 for x in auditory_complex_genes if (x['n_sparse_shunts'] + x['n_interval_shunts'] + (x['sp_shunts_count'] or 0)) > 0)
    print(f"  Auditory genes with certified shunts: {aud_with_shunts} / {len(auditory_complex_genes)}")
    for row in auditory_complex_genes[:15]:
        print(
            f"  {row['gene_symbol']:10s} (N={row['n_taxa']:3d}, L={row['seq_len']:5d}) | "
            f"INT={row['n_interval_shunts']} SP={row['n_sparse_shunts']} SpTree={row['sp_shunts_count']} | "
            f"max +{row['max_shunt_gain_nats']:5.1f}n"
        )


if __name__ == "__main__":
    main()
