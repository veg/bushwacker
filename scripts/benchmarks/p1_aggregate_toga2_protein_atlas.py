#!/usr/bin/env python3
"""
Post-hoc Taxonomic, Ecological-Guild, and Order-Pair Aggregator for Pipeline 1:
19,952-Protein x 882-Mammal TOGA2 Amino-Acid Bushwhacker Atlas.
"""


import sys as _sys
from pathlib import Path as _Path
if str(_Path(__file__).resolve().parent) not in _sys.path:
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
from _env_paths import REPO_ROOT, DATA_DIR, SPECIMENS_DIR, ALIGNMENTS_DIR, SINGLE_CELL_DIR, RESULTS_DIR, WORK_ROOT, EXTERNAL_DIR, resolve_bin, ASTRAL_JAR

import os
import glob
import json
import math
import time
from collections import Counter, defaultdict

BASE_DIR = str(WORK_ROOT / "toga2_aa")
SUMMARIES_DIR = os.path.join(BASE_DIR, "summaries")
TSV_PATH = str(ALIGNMENTS_DIR / "supermatrices" / "toga2_assemblies_and_species.tsv")
NT_ATLAS_PATH = str(RESULTS_DIR / "mammalian_toga2" / "toga2_taxonomic_atlas.json")

MAMMALIAN_ORDERS = {
    "Primates", "Rodentia", "Lagomorpha", "Scandentia", "Dermoptera",
    "Chiroptera", "Cetacea", "Artiodactyla", "Perissodactyla", "Carnivora",
    "Pholidota", "Eulipotyphla", "Proboscidea", "Sirenia", "Hyracoidea",
    "Tubulidentata", "Macroscelidea", "Afrosoricida", "Cingulata", "Pilosa",
    "Didelphimorphia", "Diprotodontia", "Dasyuromorphia", "Microbiotheria",
    "Peramelemorphia", "Monotremata"
}

ECOLOGICAL_GUILDS = {
    "echolocators": lambda sp, cm, lin: (
        ("Chiroptera" in lin and "Pteropodidae" not in lin)
        or any(x in lin for x in ["Odontoceti", "Delphinidae", "Phocoenidae", "Physeteridae", "Ziphiidae", "Kogiidae", "Monodontidae", "Iniidae", "Pontoporiidae", "Platanistidae", "Lipotes"])
        or any(x in sp.lower() or x in cm.lower() for x in ["solenodon", "blarina", "sorex", "tenrec", "echinops"])
    ),
    "echolocating_bats_and_toothed_whales": lambda sp, cm, lin: (
        ("Chiroptera" in lin and "Pteropodidae" not in lin)
        or any(x in lin for x in ["Odontoceti", "Delphinidae", "Phocoenidae", "Physeteridae", "Ziphiidae", "Kogiidae", "Monodontidae", "Iniidae", "Pontoporiidae", "Platanistidae", "Lipotes"])
    ),
    "marine_aquatic": lambda sp, cm, lin: (
        any(x in lin for x in ["Cetacea", "Sirenia", "Pinnipedia", "Phocidae", "Otariidae", "Odobenidae"])
        or any(x in sp.lower() or x in cm.lower() for x in ["otter", "lutra", "enhydra", "pteronura", "castor", "beaver", "ondatra", "ornithorhynchus", "platypus", "hippopotamus", "desman", "neomys"])
    ),
    "subterranean_blind": lambda sp, cm, lin: any(
        x in sp.lower() or x in cm.lower()
        for x in ["mole", "zokor", "eospalax", "myospalax", "heterocephalus", "fukomys", "bathyergus", "spalax", "nannospalax", "rhizomys", "tachyoryctes", "chrysochloris", "amblysomus", "condylura", "scalopus", "talpa", "mogera", "urotrichus", "ctenomys"]
    ),
    "myrmecophagous_insectivorous_armor": lambda sp, cm, lin: (
        any(x in lin for x in ["Pholidota", "Cingulata", "Tubulidentata"])
        or any(x in sp.lower() or x in cm.lower() for x in ["anteater", "myrmecophaga", "tamandua", "cyclopes", "pangolin", "manis", "phataginus", "smutsia", "aardvark", "orycteropus", "armadillo", "dasypus", "tolypeutes", "chaetophractus", "echidna", "tachyglossus"])
    ),
}

SENSORY_COCHLEAR_GENES = {
    "SLC26A5", "TMC1", "OTOF", "PJVK", "CDH23", "PCDH15", "USH2A", "MYO7A",
    "MYO15A", "WHRN", "CLRN1", "STRC", "TECTA", "KCNQ4", "GJB2", "GJB6",
    "POU4F3", "EYA4", "ACTG1", "TRIOBP", "ESPN", "LHFPL5", "TMIE", "CIB2", "TPRN", "ADGRV1"
}


def load_taxonomy():
    meta = {}
    with open(TSV_PATH, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 7:
                continue
            sp, common, tax_id, lineage_str, asm = parts[1].strip(), parts[2].strip(), parts[4].strip(), parts[5].strip(), parts[6].strip()
            lin_tokens = [t.strip() for t in lineage_str.split(";") if t.strip()]
            order = "Unknown"
            family = "Unknown"
            for tok in lin_tokens:
                if tok in MAMMALIAN_ORDERS:
                    order = tok
                if tok.endswith("idae") and family == "Unknown":
                    family = tok
            if "Cetacea" in lin_tokens:
                order = "Cetacea"
            guilds = [g for g, fn in ECOLOGICAL_GUILDS.items() if fn(sp, common, lineage_str)]
            meta[asm] = {
                "species": sp,
                "common": common,
                "order": order,
                "family": family,
                "guilds": guilds,
            }
    return meta


def classify_clade(leaves, meta):
    orders = Counter(meta.get(x, {}).get("order", "Unknown") for x in leaves if x in meta)
    fams = Counter(meta.get(x, {}).get("family", "Unknown") for x in leaves if x in meta)
    guilds = Counter()
    for x in leaves:
        for g in meta.get(x, {}).get("guilds", []):
            guilds[g] += 1
    n_known = max(1, sum(orders.values()))
    top_ord = orders.most_common(1)[0][0] if orders else "Unknown"
    top_fam = fams.most_common(1)[0][0] if fams else "Unknown"
    maj_guilds = {g for g, c in guilds.items() if c >= 0.5 * n_known}
    return top_ord, set(orders.keys()) - {"Unknown"}, top_fam, set(fams.keys()) - {"Unknown"}, maj_guilds


def main():
    meta = load_taxonomy()
    seen_genes = {}
    for jf in sorted(glob.glob(os.path.join(SUMMARIES_DIR, "worker_*.jsonl"))):
        with open(jf, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("status") == "ok":
                    seen_genes[rec["gene"]] = rec

    genes = list(seen_genes.values())
    n_ok = len(genes)
    print(f"[P1 Aggregate] Loaded {n_ok} completed mammalian protein genes", flush=True)

    tot_bin_splits = sum(g["bin_splits"] for g in genes)
    tot_mdl_splits = sum(g["mdl_splits"] for g in genes)
    tot_bin_conf = sum(g["bin_sp_conflict"] for g in genes)
    tot_mdl_conf = sum(g["mdl_sp_conflict"] for g in genes)
    tot_sp_splits = sum(g["sp_splits_total"] for g in genes)
    tot_sp_res = sum(g["sp_resolved_splits"] or 0 for g in genes)

    # Shunt taxonomy analysis on speciesTree (sp_shunts) & de novo (free_shunts)
    inter_order_sparse = []
    inter_order_interval = []
    inter_family_sparse = []
    inter_family_interval = []
    order_pair_sparse = Counter()
    order_pair_interval = Counter()
    guild_convergent_counts = Counter()
    guild_convergent_genes = defaultdict(list)
    cochlear_hits = []

    tot_sp_sparse = 0
    tot_sp_interval = 0

    for g in genes:
        sym = g["symbol"]
        for s in g.get("sp_shunts", []):
            mode = s["mode"]
            if mode == "SPARSE":
                tot_sp_sparse += 1
            else:
                tot_sp_interval += 1

            u_ord, u_ords, u_fam, u_fams, u_guilds = classify_clade(s["leaves_u"], meta)
            v_ord, v_ords, v_fam, v_fams, v_guilds = classify_clade(s["leaves_v"], meta)

            u_names = [f"{meta.get(x, {}).get('common', x)} ({meta.get(x, {}).get('species', x)})" for x in s["leaves_u"][:3]]
            v_names = [f"{meta.get(x, {}).get('common', x)} ({meta.get(x, {}).get('species', x)})" for x in s["leaves_v"][:3]]

            annot = {
                "gene": g["gene"],
                "symbol": sym,
                "aa_len": g["aa_len"],
                "mode": mode,
                "gain_nats": s["net_mdl_gain_nats"],
                "shared_aa": s["shared_mutations"],
                "posterior_prob": s["posterior_prob"],
                "interval_span": s.get("interval_span"),
                "residues_1based": s.get("residues_1based", []),
                "clade_size_u": s["clade_size_u"],
                "clade_size_v": s["clade_size_v"],
                "u_order": u_ord,
                "v_order": v_ord,
                "u_family": u_fam,
                "v_family": v_fam,
                "u_taxa_sample": u_names,
                "v_taxa_sample": v_names,
                "shared_guilds": sorted(u_guilds & v_guilds),
            }

            is_inter_order = bool(u_ords and v_ords and u_ords.isdisjoint(v_ords))
            is_inter_family = bool(u_fams and v_fams and u_fams.isdisjoint(v_fams))

            if sym in SENSORY_COCHLEAR_GENES and is_inter_family:
                cochlear_hits.append(annot)

            if is_inter_order:
                op = " <-> ".join(sorted([u_ord, v_ord]))
                if mode == "SPARSE":
                    inter_order_sparse.append(annot)
                    order_pair_sparse[op] += 1
                else:
                    inter_order_interval.append(annot)
                    order_pair_interval[op] += 1

                for sg in (u_guilds & v_guilds):
                    if mode == "SPARSE":
                        guild_convergent_counts[sg] += 1
                        guild_convergent_genes[sg].append(annot)
            elif is_inter_family:
                if mode == "SPARSE":
                    inter_family_sparse.append(annot)
                else:
                    inter_family_interval.append(annot)
                    # Suborder echolocation (e.g., Yangochiroptera <-> Rhinolophoidea or Delphinida <-> Kogiidae)
                    for sg in (u_guilds & v_guilds):
                        guild_convergent_counts[f"{sg}_inter_family"] += 1
                        guild_convergent_genes[f"{sg}_inter_family"].append(annot)
                for sg in (u_guilds & v_guilds):
                    if mode == "SPARSE":
                        guild_convergent_counts[f"{sg}_all_independent_clades"] += 1
                        guild_convergent_genes[f"{sg}_all_independent_clades"].append(annot)

    inter_order_sparse.sort(key=lambda x: x["gain_nats"], reverse=True)
    inter_order_interval.sort(key=lambda x: x["gain_nats"], reverse=True)
    inter_family_sparse.sort(key=lambda x: x["gain_nats"], reverse=True)
    cochlear_hits.sort(key=lambda x: x["gain_nats"], reverse=True)
    for k in guild_convergent_genes:
        guild_convergent_genes[k].sort(key=lambda x: x["gain_nats"], reverse=True)

    # Expected baseline guild sharing fraction among random inter-order pairs
    known_asms = [a for a, m in meta.items() if m["order"] != "Unknown"]
    guild_enrichment = {}
    n_io_sparse = max(1, len(inter_order_sparse))
    for g_name in ECOLOGICAL_GUILDS:
        # Compute analytical inter-order pair probability p_0 that two randomly chosen tips from DIFFERENT orders both belong to guild g_name
        ord_tot = Counter(meta[a]["order"] for a in known_asms)
        ord_g = Counter(meta[a]["order"] for a in known_asms if g_name in meta[a]["guilds"])
        ords = sorted(ord_tot.keys())
        tot_pairs = 0
        both_g_pairs = 0
        for i in range(len(ords)):
            for j in range(i + 1, len(ords)):
                o1, o2 = ords[i], ords[j]
                tot_pairs += ord_tot[o1] * ord_tot[o2]
                both_g_pairs += ord_g[o1] * ord_g[o2]
        p0 = both_g_pairs / max(1, tot_pairs)
        obs = guild_convergent_counts[g_name]
        exp = n_io_sparse * p0
        fold = round(obs / max(1e-6, exp), 2) if exp > 0 else 0.0
        guild_enrichment[g_name] = {
            "observed_inter_order_sparse_shunts": obs,
            "expected_random_inter_order": round(exp, 2),
            "fold_enrichment": fold,
            "null_pair_prob": round(p0, 6),
            "observed_including_inter_family_independent_clades": (
                obs + guild_convergent_counts[f"{g_name}_all_independent_clades"]
            ),
            "top_genes": guild_convergent_genes[g_name][:15],
            "top_inter_family_genes": guild_convergent_genes[f"{g_name}_all_independent_clades"][:15],
        }

    atlas = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_proteins_completed": n_ok,
        "total_cpu_free_sec": round(sum(g["runtime_free_sec"] for g in genes), 1),
        "total_cpu_sp_sec": round(sum(g["runtime_sp_sec"] for g in genes), 1),
        "mean_free_runtime_sec": round(sum(g["runtime_free_sec"] for g in genes) / max(1, n_ok), 3),
        "tree_resolution_and_concordance": {
            "total_forced_binary_splits": tot_bin_splits,
            "total_mdl_resolved_splits": tot_mdl_splits,
            "mdl_polytomy_contracted_frac": round(1.0 - tot_mdl_splits / max(1, tot_bin_splits), 4),
            "species_tree_total_splits": tot_sp_splits,
            "species_tree_mdl_supported_splits": tot_sp_res,
            "species_tree_supported_frac": round(tot_sp_res / max(1, tot_sp_splits), 4),
            "forced_binary_sp_conflict": tot_bin_conf,
            "mdl_sp_conflict": tot_mdl_conf,
            "spurious_gene_tree_conflict_eliminated_frac": round(1.0 - tot_mdl_conf / max(1, tot_bin_conf), 4),
        },
        "shunt_totals_on_species_tree": {
            "proteins_with_any_shunt": sum(1 for g in genes if g["n_sp_shunts"] > 0),
            "proteins_with_sparse_shunt": sum(1 for g in genes if g["n_sp_sparse"] > 0),
            "proteins_with_interval_shunt": sum(1 for g in genes if g["n_sp_interval"] > 0),
            "total_sparse_shunts": tot_sp_sparse,
            "total_interval_shunts": tot_sp_interval,
            "inter_order_sparse_shunts": len(inter_order_sparse),
            "inter_order_interval_shunts": len(inter_order_interval),
            "inter_family_intra_order_sparse_shunts": len(inter_family_sparse),
            "inter_family_intra_order_interval_shunts": len(inter_family_interval),
        },
        "ecological_guild_enrichment": guild_enrichment,
        "top_order_pairs_sparse_convergence": order_pair_sparse.most_common(30),
        "top_order_pairs_interval_transfer": order_pair_interval.most_common(30),
        "cochlear_mechanotransduction_shunts": cochlear_hits[:30],
        "top_50_inter_order_sparse_convergence": inter_order_sparse[:50],
        "top_50_inter_order_interval_transfers": inter_order_interval[:50],
        "top_30_inter_family_sparse_convergence": inter_family_sparse[:30],
    }

    out_path = os.path.join(BASE_DIR, "toga2_protein_atlas.json")
    with open(out_path, "w") as f:
        json.dump(atlas, f, indent=2)
    print(f"[P1 Aggregate] Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
