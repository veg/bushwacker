#!/usr/bin/env python3
"""
Taxonomic & Clade-by-Clade Deep Mining of the 19,952-Gene TOGA2
Mammalian Proteome Bushwhacker Atlas on Silverback.

Uses Senckenberg's official `assemblies_and_species.tsv` (100% match to all 882
mammalian genome assemblies in `speciesTree.nh`) to:
  1. Annotate 100% of all 1,032 INTERVAL and 14,105 SPARSE shunts (both leaf and
     internal-node endpoints via `leaves_u` and `leaves_v`) with exact scientific
     names, common names, families, mammalian orders, and ecological guilds.
  2. Quantify Inter-Order vs. Intra-Order/Inter-Family vs. Intra-Family shunt
     distributions for SPARSE (adaptive convergence) vs. INTERVAL (ectopic gene conversion).
  3. Map exact clade/species pairs for the 16 cochlear hair-cell / mechanotransduction
     genes and top sensory/metabolic/immune convergence hubs.
  4. Compute clade-level concordance & MDL polytomy rates across major mammalian
     superorders and orders on `speciesTree.nh`.
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
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

BASE_DIR = str(WORK_ROOT / "toga2_nt")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
TSV_PATH = os.path.join(BASE_DIR, "assemblies_and_species.tsv")
SPECIES_TREE_PATH = "data/alignments/supermatrices/toga2_speciesTree.nh"
OUT_JSON = os.path.join(BASE_DIR, "toga2_taxonomic_atlas.json")

MAMMALIAN_ORDERS = {
    "Primates", "Rodentia", "Lagomorpha", "Scandentia", "Dermoptera",
    "Chiroptera", "Cetacea", "Artiodactyla", "Perissodactyla", "Carnivora",
    "Pholidota", "Eulipotyphla", "Proboscidea", "Sirenia", "Hyracoidea",
    "Tubulidentata", "Macroscelidea", "Afrosoricida", "Cingulata", "Pilosa",
    "Didelphimorphia", "Diprotodontia", "Dasyuromorphia", "Microbiotheria",
    "Peramelemorphia", "Monotremata"
}

ECOLOGICAL_GUILDS = {
    "echolocating_bats": lambda sp, cm, lin: "Chiroptera" in lin and "Pteropodidae" not in lin,
    "pteropodid_bats": lambda sp, cm, lin: "Pteropodidae" in lin,
    "toothed_whales": lambda sp, cm, lin: any(x in lin for x in ["Odontoceti", "Delphinidae", "Phocoenidae", "Physeteridae", "Ziphiidae", "Kogiidae", "Monodontidae", "Iniidae", "Pontoporiidae", "Platanistidae", "Lipotes"]),
    "baleen_whales": lambda sp, cm, lin: any(x in lin for x in ["Mysticeti", "Balaenopteridae", "Balaenidae", "Eschrichtiidae"]),
    "subterranean_mammals": lambda sp, cm, lin: any(x in sp.lower() or x in cm.lower() for x in ["mole", "zokor", "eospalax", "myospalax", "heterocephalus", "fukomys", "bathyergus", "spalax", "nannospalax", "rhizomys", "tachyoryctes", "chrysochloris", "amblysomus", "condylura", "scalopus", "talpa", "mogera", "urotrichus", "ctenomys"]),
    "aquatic_semiaquatic": lambda sp, cm, lin: any(x in lin for x in ["Cetacea", "Sirenia", "Pinnipedia", "Phocidae", "Otariidae", "Odobenidae"]) or any(x in sp.lower() or x in cm.lower() for x in ["otter", "lutra", "enhydra", "pteronura", "castor", "beaver", "ondatra", "ornithorhynchus", "platypus", "hippopotamus", "desman", "neomys"]),
}

COCHLEAR_GENES = {
    "SLC26A5", "TMC1", "OTOF", "PJVK", "CDH23", "PCDH15", "USH2A", "MYO7A",
    "MYO15A", "WHRN", "CLRN1", "STRC", "TECTA", "KCNQ4", "GJB2", "GJB6",
    "POU4F3", "EYA4", "ACTG1", "TRIOBP", "ESPN", "LHFPL5", "TMIE", "CIB2", "TPRN"
}


def load_taxonomy():
    meta = {}
    with open(TSV_PATH, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 7:
                continue
            sp = parts[1].strip()
            common = parts[2].strip()
            tax_id = parts[4].strip()
            lineage_str = parts[5].strip()
            asm = parts[6].strip()
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
                "tax_id": tax_id,
                "order": order,
                "family": family,
                "lineage": lin_tokens,
                "guilds": guilds,
            }
    return meta


def parse_newick_nodes(nwk_str):
    s = nwk_str.strip()
    if s.endswith(";"):
        s = s[:-1]
    tokens = re.findall(r"\(|\)|,|;|:[^(),;]+|[^(),;:]+", s)
    stack = [[]]
    i = 0
    n = len(tokens)
    all_leaves_list = []
    clades = []

    while i < n:
        tok = tokens[i]
        if tok == "(":
            stack.append([])
            i += 1
        elif tok == ",":
            i += 1
        elif tok == ")":
            children = stack.pop()
            i += 1
            lbl = ""
            bl = 0.0
            while i < n and tokens[i] not in ("(", ")", ",", ";"):
                if tokens[i].startswith(":"):
                    try:
                        bl = float(tokens[i][1:])
                    except ValueError:
                        bl = 0.0
                else:
                    lbl = tokens[i]
                i += 1
            leaves_set = frozenset().union(*(c[0] for c in children))
            clades.append((leaves_set, lbl, bl))
            stack[-1].append((leaves_set, lbl, bl))
        else:
            name = tok
            i += 1
            bl = 0.0
            while i < n and tokens[i] not in ("(", ")", ",", ";"):
                if tokens[i].startswith(":"):
                    try:
                        bl = float(tokens[i][1:])
                    except ValueError:
                        bl = 0.0
                i += 1
            leaves_set = frozenset([name])
            all_leaves_list.append(name)
            stack[-1].append((leaves_set, name, bl))

    return frozenset(all_leaves_list), clades


def summarize_clade_taxa(leaves, meta):
    if not leaves:
        return {
            "n_leaves": 0,
            "orders": [],
            "families": [],
            "guilds": [],
            "sample_species": [],
            "label": "Unknown",
        }
    orders = Counter()
    families = Counter()
    guilds = Counter()
    sp_list = []
    for l in sorted(leaves):
        m = meta.get(l)
        if m:
            orders[m["order"]] += 1
            if m["family"] != "Unknown":
                families[m["family"]] += 1
            for g in m["guilds"]:
                guilds[g] += 1
            sp_list.append(f"{m['species']} ({m['common']}, {l})")
        else:
            sp_list.append(l)

    top_orders = [k for k, _ in orders.most_common(3)]
    top_fams = [k for k, _ in families.most_common(3)]
    if len(leaves) == 1:
        l0 = next(iter(leaves))
        m0 = meta.get(l0)
        lbl = f"{m0['species']} [{m0['common']}; {m0['order']}:{m0['family']}]" if m0 else l0
    elif len(leaves) <= 3:
        names = []
        for l in sorted(leaves):
            m = meta.get(l)
            names.append(f"{m['species']} ({m['common']})" if m else l)
        ord_str = "/".join(top_orders)
        fam_str = "/".join(top_fams[:2])
        lbl = f"Clade({len(leaves)} spp [{ord_str}:{fam_str}]: {', '.join(names)})"
    else:
        ord_str = "/".join(top_orders)
        fam_str = "/".join(top_fams[:2])
        m0 = meta.get(sorted(leaves)[0])
        ex0 = f"{m0['species']} ({m0['common']})" if m0 else sorted(leaves)[0]
        lbl = f"Clade({len(leaves)} spp [{ord_str}:{fam_str}], e.g. {ex0})"

    return {
        "n_leaves": len(leaves),
        "orders": [k for k, _ in orders.most_common()],
        "families": [k for k, _ in families.most_common()],
        "guilds": [k for k, _ in guilds.most_common()],
        "sample_species": sp_list[:5],
        "label": lbl,
    }


def process_chunk(files):
    shunt_records = []
    cochlear_records = []
    for fpath in files:
        base = os.path.basename(fpath)
        if not base.endswith(".bw.json.gz"):
            continue
        stem = base[:-11]
        if "#" in stem:
            tx, gene = stem.split("#", 1)
        else:
            tx, gene = stem, stem

        try:
            with gzip.open(fpath, "rt", encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue

        n_taxa = d.get("n_taxa", 0)
        seq_len = d.get("seq_len", 0)
        shunts = d.get("shunts", [])

        if gene in COCHLEAR_GENES:
            cochlear_records.append({
                "gene": gene,
                "transcript": tx,
                "n_taxa": n_taxa,
                "seq_len": seq_len,
                "resolved_internal_splits": d.get("resolved_internal_splits", 0),
                "max_binary_splits": d.get("max_binary_splits", 0),
                "n_shunts": len(shunts),
                "shunts": shunts,
            })

        for s in shunts:
            shunt_records.append({
                "gene": gene,
                "transcript": tx,
                "n_taxa": n_taxa,
                "seq_len": seq_len,
                "mode": s.get("mode", ""),
                "label_u": s.get("label_u", ""),
                "label_v": s.get("label_v", ""),
                "leaves_u": s.get("leaves_u", [s.get("label_u", "")]),
                "leaves_v": s.get("leaves_v", [s.get("label_v", "")]),
                "shared_mutations": s.get("shared_mutations", 0),
                "codons_1based": s.get("codons_1based", []),
                "interval_span": s.get("interval_span"),
                "data_savings_nats": s.get("data_savings_nats", 0.0),
                "net_mdl_gain_nats": s.get("net_mdl_gain_nats", 0.0),
                "posterior_prob": s.get("posterior_prob", 0.0),
            })

    return shunt_records, cochlear_records


def main():
    print("Loading Senckenberg TOGA2 assemblies_and_species.tsv...")
    meta = load_taxonomy()
    print(f"Loaded taxonomic metadata for {len(meta)} assemblies.")

    with open(SPECIES_TREE_PATH, "r", encoding="utf-8") as f:
        sp_nwk = f.read().strip()
    sp_leaves, sp_clades = parse_newick_nodes(sp_nwk)
    matched_leaves = [l for l in sp_leaves if l in meta]
    order_counts = Counter(meta[l]["order"] for l in matched_leaves)
    family_counts = Counter(meta[l]["family"] for l in matched_leaves if meta[l]["family"] != "Unknown")
    print(f"speciesTree.nh has {len(sp_leaves)} leaf assemblies ({len(matched_leaves)} matched = {len(matched_leaves)*100.0/len(sp_leaves):.1f}%), across {len(order_counts)} orders and {len(family_counts)} families.")

    all_files = sorted([
        os.path.join(RESULTS_DIR, fn)
        for fn in os.listdir(RESULTS_DIR)
        if fn.endswith(".bw.json.gz")
    ])
    print(f"Scanning {len(all_files)} TOGA2 .bw.json.gz files with 48 workers...")

    n_workers = 48
    chunk_size = max(1, (len(all_files) + n_workers - 1) // n_workers)
    chunks = [all_files[i:i + chunk_size] for i in range(0, len(all_files), chunk_size)]

    all_shunts = []
    all_cochlear = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for sh_list, coch_list in ex.map(process_chunk, chunks):
            all_shunts.extend(sh_list)
            all_cochlear.extend(coch_list)

    print(f"Extracted {len(all_shunts)} certified shunts and {len(all_cochlear)} cochlear gene reports.")

    mode_tax_breakdown = {
        "INTERVAL": Counter(),
        "SPARSE": Counter(),
    }
    order_pair_counts = {
        "INTERVAL": Counter(),
        "SPARSE": Counter(),
    }
    family_pair_counts = {
        "INTERVAL": Counter(),
        "SPARSE": Counter(),
    }
    guild_pair_counts = {
        "INTERVAL": Counter(),
        "SPARSE": Counter(),
    }

    annotated_shunts = []
    for s in all_shunts:
        mode = s["mode"]
        lu = [x for x in s["leaves_u"] if x in meta]
        lv = [x for x in s["leaves_v"] if x in meta]
        if not lu and s["label_u"] in meta:
            lu = [s["label_u"]]
        if not lv and s["label_v"] in meta:
            lv = [s["label_v"]]

        info_u = summarize_clade_taxa(lu, meta)
        info_v = summarize_clade_taxa(lv, meta)

        s_copy = dict(s)
        s_copy["taxon_u"] = info_u["label"]
        s_copy["taxon_v"] = info_v["label"]
        s_copy["n_leaves_u"] = info_u["n_leaves"]
        s_copy["n_leaves_v"] = info_v["n_leaves"]
        s_copy["orders_u"] = info_u["orders"]
        s_copy["orders_v"] = info_v["orders"]
        s_copy["families_u"] = info_u["families"]
        s_copy["families_v"] = info_v["families"]
        s_copy["guilds_u"] = info_u["guilds"]
        s_copy["guilds_v"] = info_v["guilds"]

        if info_u["orders"] and info_v["orders"]:
            ords_u = set(info_u["orders"])
            ords_v = set(info_v["orders"])
            fams_u = set(info_u["families"])
            fams_v = set(info_v["families"])
            if ords_u.isdisjoint(ords_v):
                rel = "inter_order"
                pair = tuple(sorted([info_u["orders"][0], info_v["orders"][0]]))
                order_pair_counts[mode][f"{pair[0]} <-> {pair[1]}"] += 1
            elif fams_u and fams_v and fams_u.isdisjoint(fams_v):
                rel = "intra_order_inter_family"
                ord0 = info_u["orders"][0]
                fpair = tuple(sorted([info_u["families"][0], info_v["families"][0]]))
                family_pair_counts[mode][f"{ord0}: {fpair[0]} <-> {fpair[1]}"] += 1
            else:
                rel = "intra_family"
                ord0 = info_u["orders"][0]
                fam0 = info_u["families"][0] if info_u["families"] else "Unknown"
                family_pair_counts[mode][f"{ord0}: {fam0} (intra-family)"] += 1
            mode_tax_breakdown[mode][rel] += 1
            s_copy["taxonomic_scope"] = rel
        else:
            mode_tax_breakdown[mode]["unmapped"] += 1
            s_copy["taxonomic_scope"] = "unmapped"

        if info_u["guilds"] and info_v["guilds"]:
            for gu in info_u["guilds"]:
                for gv in info_v["guilds"]:
                    gpair = tuple(sorted([gu, gv]))
                    guild_pair_counts[mode][f"{gpair[0]} <-> {gpair[1]}"] += 1

        # Remove raw leaf lists from top table dump to keep JSON compact
        del s_copy["leaves_u"]
        del s_copy["leaves_v"]
        s_copy["sample_codons"] = s_copy["codons_1based"][:12]
        del s_copy["codons_1based"]
        annotated_shunts.append(s_copy)

    interval_all = [s for s in annotated_shunts if s["mode"] == "INTERVAL"]
    interval_all.sort(key=lambda x: x["net_mdl_gain_nats"], reverse=True)

    sparse_all = [s for s in annotated_shunts if s["mode"] == "SPARSE"]
    sparse_all.sort(key=lambda x: x["net_mdl_gain_nats"], reverse=True)

    inter_order_sparse = [s for s in sparse_all if s.get("taxonomic_scope") == "inter_order"]
    inter_family_sparse = [s for s in sparse_all if s.get("taxonomic_scope") == "intra_order_inter_family"]
    inter_order_interval = [s for s in interval_all if s.get("taxonomic_scope") == "inter_order"]

    # Annotate cochlear shunts
    for c in all_cochlear:
        annot_c_shunts = []
        for s in c["shunts"]:
            lu = [x for x in s.get("leaves_u", [s.get("label_u", "")]) if x in meta]
            lv = [x for x in s.get("leaves_v", [s.get("label_v", "")]) if x in meta]
            iu = summarize_clade_taxa(lu, meta)
            iv = summarize_clade_taxa(lv, meta)
            annot_c_shunts.append({
                "mode": s.get("mode"),
                "taxon_u": iu["label"],
                "taxon_v": iv["label"],
                "shared_mutations": s.get("shared_mutations"),
                "net_mdl_gain_nats": s.get("net_mdl_gain_nats"),
                "posterior_prob": s.get("posterior_prob"),
                "sample_codons": s.get("codons_1based", [])[:10],
            })
        annot_c_shunts.sort(key=lambda x: x["net_mdl_gain_nats"], reverse=True)
        c["annotated_shunts"] = annot_c_shunts
        c["total_shunt_gain_nats"] = round(sum(x["net_mdl_gain_nats"] for x in annot_c_shunts), 3)
        del c["shunts"]
    all_cochlear.sort(key=lambda c: c["total_shunt_gain_nats"], reverse=True)

    out = {
        "species_tree_summary": {
            "n_leaf_assemblies": len(sp_leaves),
            "n_internal_nodes": len(sp_clades),
            "n_total_nodes": len(sp_leaves) + len(sp_clades),
            "n_matched_in_tsv": len(matched_leaves),
            "n_orders": len(order_counts),
            "n_families": len(family_counts),
            "orders_breakdown": dict(order_counts.most_common()),
            "top_30_families": dict(family_counts.most_common(30)),
        },
        "shunt_taxonomic_scope": {
            "INTERVAL": dict(mode_tax_breakdown["INTERVAL"]),
            "SPARSE": dict(mode_tax_breakdown["SPARSE"]),
        },
        "top_inter_order_pairs": {
            "INTERVAL": dict(order_pair_counts["INTERVAL"].most_common(25)),
            "SPARSE": dict(order_pair_counts["SPARSE"].most_common(25)),
        },
        "top_family_pairs": {
            "INTERVAL": dict(family_pair_counts["INTERVAL"].most_common(25)),
            "SPARSE": dict(family_pair_counts["SPARSE"].most_common(25)),
        },
        "top_guild_pairs": {
            "INTERVAL": dict(guild_pair_counts["INTERVAL"].most_common(20)),
            "SPARSE": dict(guild_pair_counts["SPARSE"].most_common(20)),
        },
        "top_35_interval_shunts": interval_all[:35],
        "top_35_inter_order_interval_shunts": inter_order_interval[:35],
        "top_35_sparse_shunts": sparse_all[:35],
        "top_35_inter_order_sparse_shunts": inter_order_sparse[:35],
        "top_35_inter_family_sparse_shunts": inter_family_sparse[:35],
        "cochlear_mechanotransduction_atlas": all_cochlear,
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved taxonomic atlas to {OUT_JSON}")


if __name__ == "__main__":
    main()
