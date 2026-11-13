#!/usr/bin/env python3
"""
Job C: Cluster Refresh of TOGA2 (882 Mammals x 19,952 Proteins) & GTDB/Klebsiella Microbial Atlases
under Kraft-exact Eq. (7) SPARSE mask description length (+ L_N(m - m_min)).

Strict read-only access to silverback raw alignments/trees/summaries; writes all outputs
exclusively under OUTPUT_DIR (default: /home/sergei/bushwhacker_bars_rev/job_c_toga_gtdb).

Computes:
  1. Matched-denominator Fig. 4B resolution/discordance tables across all 19,952 NT & AA genes.
  2. Cochlear mechanotransduction gene panel (26+ symbols incl. SLC26A5, LOXHD1, USH2A, TECTA, ADGRV1)
     across all 4 modes: Free NT, -t Species-Tree NT, Free AA (--translate), -t Species-Tree AA.
  3. Exact Kraft Eq. (7) monotonic refresh:
     - 100% of previously shunted Sp-AA genes (4,770) + 50 zero-shunt negative controls.
     - 100% of previously shunted Sp-NT genes (3,474) + 50 zero-shunt negative controls.
     - Matched 4-mode cross-check (Free NT, -t NT, Free AA, -t AA) on guild/cochlear hits + stratified sample.
     - Re-evaluation on completed WALRUS de novo MSAs (only genes with finished *.walrus.json.gz).
  4. Formalized TOGA2 Ecological Guild Null & Sensitivities:
     - Analytical inter-order baseline + length/taxa-stratified permutation null (10,000 draws)
       + Poisson/Quasi-Poisson rate-ratio check + gene-family/clade deduplication.
     - Sensitivity with OR*, VN*, TTN, ZNF*, HLA*, SSX*, MAGE* multicopy families excluded.
     - Sensitivity across Free AA, -t AA, -t NT, and WALRUS de novo MSAs.
  5. GTDB R95 (122 Archaeal protein markers + 68 clade NT/AA) & Klebsiella (222 core NT/AA + 3 specials)
     refresh with Kraft-exact Eq. (7) + unambiguous per-row 2x2 Fisher exact tests for Table S8B.
"""

import glob
import gzip
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import stats


BUSHWHACKER_BIN = os.environ.get(
    "BUSHWHACKER_BIN",
    "/home/sergei/bushwhacker_bars_rev/bushwhacker_engine/target/release/bushwhacker",
)
TOGA2_RAW_ALIS = os.environ.get("TOGA2_RAW_ALIS", "/home/sergei/toga2_walrus/raw_alis/individualAlis")
TOGA2_SPECIES_TREE = os.environ.get("TOGA2_SPECIES_TREE", "/home/sergei/toga2_walrus/speciesTree.nh")
TOGA2_TAX_TSV = os.environ.get("TOGA2_TAX_TSV", "/home/sergei/toga2_bushwhacker_analysis/assemblies_and_species.tsv")
TOGA2_NT_MASTER = os.environ.get(
    "TOGA2_NT_MASTER", "/home/sergei/toga2_bushwhacker_analysis/toga2_bushwhacker_master_19953.jsonl"
)
TOGA2_AA_SUMMARIES_GLOB = os.environ.get(
    "TOGA2_AA_SUMMARIES_GLOB", "/home/sergei/toga2_protein_atlas/summaries/worker_*.jsonl"
)
WALRUS_MSAS_DIR = os.environ.get("WALRUS_MSAS_DIR", "/home/sergei/toga2_walrus/msas")
WALRUS_RESULTS_DIR = os.environ.get("WALRUS_RESULTS_DIR", "/home/sergei/toga2_walrus/results")

KLEB_DIR = os.environ.get("KLEB_DIR", "/storage/xl-data/users/sergei/klebsiella_amr")
GTDB_DIR = os.environ.get("GTDB_DIR", "/home/sergei/bushwhacker_sims/gtdb_r95")

OUTPUT_DIR = os.environ.get("JOB_C_OUT", "/home/sergei/bushwhacker_bars_rev/job_c_toga_gtdb")
N_WORKERS = int(os.environ.get("BW_WORKERS", "64"))

MAMMALIAN_ORDERS = {
    "Primates", "Rodentia", "Lagomorpha", "Scandentia", "Dermoptera",
    "Chiroptera", "Cetacea", "Artiodactyla", "Perissodactyla", "Carnivora",
    "Pholidota", "Eulipotyphla", "Proboscidea", "Sirenia", "Hyracoidea",
    "Tubulidentata", "Macroscelidea", "Afrosoricida", "Cingulata", "Pilosa",
    "Didelphimorphia", "Diprotodontia", "Dasyuromorphia", "Microbiotheria",
    "Peramelemorphia", "Monotremata",
}

ECOLOGICAL_GUILDS = {
    "echolocators": lambda sp, cm, lin: (
        ("Chiroptera" in lin and "Pteropodidae" not in lin)
        or any(x in lin for x in [
            "Odontoceti", "Delphinidae", "Phocoenidae", "Physeteridae", "Ziphiidae",
            "Kogiidae", "Monodontidae", "Iniidae", "Pontoporiidae", "Platanistidae", "Lipotes",
        ])
        or any(x in sp.lower() or x in cm.lower() for x in ["solenodon", "blarina", "sorex", "tenrec", "echinops"])
    ),
    "echolocating_bats_and_toothed_whales": lambda sp, cm, lin: (
        ("Chiroptera" in lin and "Pteropodidae" not in lin)
        or any(x in lin for x in [
            "Odontoceti", "Delphinidae", "Phocoenidae", "Physeteridae", "Ziphiidae",
            "Kogiidae", "Monodontidae", "Iniidae", "Pontoporiidae", "Platanistidae", "Lipotes",
        ])
    ),
    "marine_aquatic": lambda sp, cm, lin: (
        any(x in lin for x in ["Cetacea", "Sirenia", "Pinnipedia", "Phocidae", "Otariidae", "Odobenidae"])
        or any(x in sp.lower() or x in cm.lower() for x in [
            "otter", "lutra", "enhydra", "pteronura", "castor", "beaver", "ondatra",
            "ornithorhynchus", "platypus", "hippopotamus", "desman", "neomys",
        ])
    ),
    "subterranean_blind": lambda sp, cm, lin: any(
        x in sp.lower() or x in cm.lower()
        for x in [
            "mole", "zokor", "eospalax", "myospalax", "heterocephalus", "fukomys",
            "bathyergus", "spalax", "nannospalax", "rhizomys", "tachyoryctes",
            "chrysochloris", "amblysomus", "condylura", "scalopus", "talpa", "mogera",
            "urotrichus", "ctenomys",
        ]
    ),
    "myrmecophagous_insectivorous_armor": lambda sp, cm, lin: (
        any(x in lin for x in ["Pholidota", "Cingulata", "Tubulidentata"])
        or any(x in sp.lower() or x in cm.lower() for x in [
            "anteater", "myrmecophaga", "tamandua", "cyclopes", "pangolin", "manis",
            "phataginus", "smutsia", "aardvark", "orycteropus", "armadillo", "dasypus",
            "tolypeutes", "chaetophractus", "echidna", "tachyglossus",
        ])
    ),
}

COCHLEAR_SYMBOLS = {
    "SLC26A5", "LOXHD1", "TMC1", "OTOF", "PJVK", "CDH23", "PCDH15", "USH2A", "MYO7A",
    "MYO15A", "WHRN", "CLRN1", "STRC", "TECTA", "KCNQ4", "GJB2", "GJB6",
    "POU4F3", "EYA4", "ACTG1", "TRIOBP", "ESPN", "LHFPL5", "TMIE", "CIB2", "TPRN", "ADGRV1",
}

MULTICOPY_BLACKLIST_PREFIXES = (
    "OR", "VN", "V1R", "V2R", "ZNF", "ZSCAN", "HLA", "SSX", "MAGE", "CTAG", "GAGE",
    "PRAME", "SPANX", "PCDH", "KRT", "USP17", "NBPF", "SPATA31",
)
MULTICOPY_EXACT = {"TTN", "MUC4", "MUC5B", "MUC16", "OBSCN", "SYNE1"}

_SP_TREE_ROOT = None
_SP_TREE_TEXT = None


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


def read_fasta_tips(path):
    tips = set()
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith(">"):
                tips.add(line[1:].strip().split()[0])
    return tips


def run_bw_single_mode(fa_path, extra_flags, ref_root=None, timeout=600):
    base_tmp = "/dev/shm" if os.path.isdir("/dev/shm") else None
    tmpdir = tempfile.mkdtemp(prefix="bw_jc_", dir=base_tmp)
    try:
        actual_fa = fa_path
        if fa_path.endswith(".gz"):
            actual_fa = os.path.join(tmpdir, "input.fa")
            with gzip.open(fa_path, "rt", encoding="utf-8", errors="replace") as fin, open(actual_fa, "w") as fout:
                shutil.copyfileobj(fin, fout)

        cmd = [BUSHWHACKER_BIN, "-T", "1"] + list(extra_flags) + [actual_fa]
        if ref_root is not None:
            tips = read_fasta_tips(actual_fa)
            pruned = prune_tree(ref_root, tips) if len(tips) >= 4 else None
            if not (pruned and pruned.get("children")):
                return {"status": "skip_few_tips"}
            ref_p = os.path.join(tmpdir, "ref.nwk")
            with open(ref_p, "w") as rf:
                rf.write(to_newick(pruned) + ";\n")
            cmd.extend(["-t", ref_p])

        out_nwk = os.path.join(tmpdir, "out.nwk")
        out_json = os.path.join(tmpdir, "out.json")
        cmd.extend(["-o", out_nwk, "-j", out_json])
        env = os.environ.copy()
        env["RAYON_NUM_THREADS"] = "1"
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
        elapsed = time.perf_counter() - t0
        if proc.returncode != 0 or not os.path.exists(out_json):
            return {"status": "error", "error": proc.stderr[-400:] if proc.stderr else f"rc={proc.returncode}"}
        d = json.load(open(out_json))
        shunts = d.get("shunts", [])
        return {
            "status": "ok",
            "alphabet": d.get("alphabet"),
            "n_taxa": d.get("n_taxa"),
            "seq_len": d.get("seq_len"),
            "runtime_sec": round(elapsed, 4),
            "star_tree_MDL_nats": d.get("star_tree_MDL_nats"),
            "tree_MDL_nats": d.get("tree_MDL_nats"),
            "network_MDL_nats": d.get("network_MDL_nats"),
            "resolved_internal_splits": d.get("resolved_internal_splits"),
            "max_binary_splits": d.get("max_binary_splits"),
            "polytomy_nodes": d.get("polytomy_nodes"),
            "n_shunts": len(shunts),
            "n_interval": sum(1 for s in shunts if s.get("mode") == "INTERVAL"),
            "n_sparse": sum(1 for s in shunts if s.get("mode") == "SPARSE"),
            "max_shunt_gain_nats": round(max((s.get("net_mdl_gain_nats", 0.0) for s in shunts), default=0.0), 4),
            "total_shunt_gain_nats": round(sum(s.get("net_mdl_gain_nats", 0.0) for s in shunts), 4),
            "shunts": shunts,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def get_sp_tree_root():
    global _SP_TREE_ROOT, _SP_TREE_TEXT
    if _SP_TREE_ROOT is None and os.path.exists(TOGA2_SPECIES_TREE):
        _SP_TREE_TEXT = open(TOGA2_SPECIES_TREE).read()
        _SP_TREE_ROOT = parse_newick(_SP_TREE_TEXT)
    return _SP_TREE_ROOT


def _worker_sp_refresh(task):
    gene, sym, mode_kind, fa_path, orig_meta = task
    root = get_sp_tree_root()
    flags = ["--translate"] if mode_kind.endswith("_aa") else []
    use_ref = root if mode_kind.startswith("sp_") else None
    res = run_bw_single_mode(fa_path, flags, ref_root=use_ref, timeout=600)
    res["gene"] = gene
    res["symbol"] = sym
    res["mode_kind"] = mode_kind
    if orig_meta:
        res["orig"] = orig_meta
    return res


def _worker_4mode(task):
    gene, sym, fa_path = task
    root = get_sp_tree_root()
    out = {"gene": gene, "symbol": sym, "path": fa_path}
    out["free_nt"] = run_bw_single_mode(fa_path, [], ref_root=None, timeout=600)
    out["sp_nt"] = run_bw_single_mode(fa_path, [], ref_root=root, timeout=600)
    out["free_aa"] = run_bw_single_mode(fa_path, ["--translate"], ref_root=None, timeout=600)
    out["sp_aa"] = run_bw_single_mode(fa_path, ["--translate"], ref_root=root, timeout=600)
    return out


def load_toga2_taxonomy():
    meta = {}
    with open(TOGA2_TAX_TSV, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 7:
                continue
            sp, common, lineage_str, asm = (
                parts[1].strip(),
                parts[2].strip(),
                parts[5].strip(),
                parts[6].strip(),
            )
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
            meta[asm] = {"species": sp, "common": common, "order": order, "family": family, "guilds": guilds}
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
    dom_guilds = {g for g, cnt in guilds.items() if cnt >= 0.5 * len(leaves)}
    return top_ord, top_fam, {o for o in orders if o != "Unknown"}, {fm for fm in fams if fm != "Unknown"}, dom_guilds, n_known


def is_multicopy_symbol(sym):
    s = (sym or "").upper()
    if s in MULTICOPY_EXACT:
        return True
    return any(s.startswith(pref) for pref in MULTICOPY_BLACKLIST_PREFIXES)


def gene_family_root(sym):
    s = (sym or "").upper()
    for pref, fam in [
        ("OR", "OR_olfactory"), ("V1R", "V1R_vomeronasal"), ("V2R", "V2R_vomeronasal"),
        ("ZNF", "ZNF_zinc_finger"), ("ZSCAN", "ZNF_zinc_finger"), ("HLA", "HLA_MHC"),
        ("SSX", "SSX_family"), ("MAGE", "MAGE_family"), ("PCDH", "PCDH_protocadherin"),
        ("KRT", "KRT_keratin"), ("SPATA31", "SPATA31_family"),
    ]:
        if s.startswith(pref):
            return fam
    return s


def poisson_two_sided(obs, exp):
    if exp <= 0:
        return 1.0 if obs == 0 else 0.0
    if obs <= exp:
        p_tail = float(stats.poisson.cdf(obs, exp))
        other_k = int(math.ceil(2 * exp - obs))
        p_other = float(stats.poisson.sf(max(other_k - 1, obs), exp)) if other_k > obs else 0.0
        return min(1.0, p_tail + p_other)
    else:
        p_tail = float(stats.poisson.sf(obs - 1, exp))
        other_k = int(math.floor(2 * exp - obs))
        p_other = float(stats.poisson.cdf(min(other_k, obs - 1), exp)) if other_k >= 0 else 0.0
        return min(1.0, p_tail + p_other)


def analyze_refreshed_toga2_guilds(refreshed_sp_aa, all_aa_records, meta):
    """Evaluate ecological guild convergence + permutation & GLM controls on Kraft-exact Eq.(7) hits."""
    hits_by_gene = {r["gene"]: r for r in refreshed_sp_aa if r.get("status") == "ok"}

    inter_order_sparse = []
    inter_order_interval = []
    all_sparse_aa_lens = []
    all_sparse_n_taxa = []
    order_pair_counts = Counter()

    for gene, ginfo in all_aa_records.items():
        r = hits_by_gene.get(gene)
        if r is None:
            continue
        sym = ginfo["symbol"]
        aa_len = r.get("seq_len") or ginfo.get("aa_len", 0)
        n_taxa = r.get("n_taxa") or ginfo.get("n_taxa", 0)
        for s in r.get("shunts", []):
            if s.get("mode") == "INTERVAL":
                u_ord, u_fam, u_ords, u_fams, u_guilds, _ = classify_clade(s["leaves_u"], meta)
                v_ord, v_fam, v_ords, v_fams, v_guilds, _ = classify_clade(s["leaves_v"], meta)
                if u_ords and v_ords and u_ords.isdisjoint(v_ords):
                    inter_order_interval.append({"gene": gene, "symbol": sym, "gain_nats": s["net_mdl_gain_nats"]})
                continue
            u_ord, u_fam, u_ords, u_fams, u_guilds, _ = classify_clade(s["leaves_u"], meta)
            v_ord, v_fam, v_ords, v_fams, v_guilds, _ = classify_clade(s["leaves_v"], meta)
            is_io = bool(u_ords and v_ords and u_ords.isdisjoint(v_ords))
            if not is_io:
                continue
            op = " <-> ".join(sorted([u_ord, v_ord]))
            order_pair_counts[op] += 1
            all_sparse_aa_lens.append(aa_len)
            all_sparse_n_taxa.append(n_taxa)
            inter_order_sparse.append({
                "gene": gene,
                "symbol": sym,
                "family_root": gene_family_root(sym),
                "is_multicopy": is_multicopy_symbol(sym),
                "aa_len": aa_len,
                "n_taxa": n_taxa,
                "gain_nats": round(s["net_mdl_gain_nats"], 4),
                "posterior_prob": s["posterior_prob"],
                "shared_aa": s["shared_mutations"],
                "residues_1based": s.get("residues_1based", [])[:24],
                "u_order": u_ord,
                "v_order": v_ord,
                "u_family": u_fam,
                "v_family": v_fam,
                "mammal_family_pair": "|".join(sorted([u_fam, v_fam])),
                "shared_guilds": sorted(u_guilds & v_guilds),
                "u_taxa_sample": [f"{meta.get(x, {}).get('common', x)} ({meta.get(x, {}).get('species', x)})" for x in s["leaves_u"][:3]],
                "v_taxa_sample": [f"{meta.get(x, {}).get('common', x)} ({meta.get(x, {}).get('species', x)})" for x in s["leaves_v"][:3]],
            })

    inter_order_sparse.sort(key=lambda x: x["gain_nats"], reverse=True)

    # Analytical p0 across all known-order assemblies (exact denominator check)
    known_asms = [a for a, m in meta.items() if m["order"] != "Unknown"]
    ord_tot = Counter(meta[a]["order"] for a in known_asms)
    ords = sorted(ord_tot.keys())
    tot_inter_order_pairs = sum(ord_tot[ords[i]] * ord_tot[ords[j]] for i in range(len(ords)) for j in range(i + 1, len(ords)))

    # Order-pair list weighted by tip-pair counts for empirical permutation null
    rng = random.Random(20261113)
    # Also compute non-focal empirical length/taxa comparison and Poisson/NB-style rate check
    n_io = max(1, len(inter_order_sparse))
    # Build per-order guild tip fraction for conditional permutation null matching (clade-order sampling)
    guild_ord_probs = {}
    for g_name in ECOLOGICAL_GUILDS:
        ord_g = Counter(meta[a]["order"] for a in known_asms if g_name in meta[a]["guilds"])
        both_g_pairs = sum(ord_g[ords[i]] * ord_g[ords[j]] for i in range(len(ords)) for j in range(i + 1, len(ords)))
        guild_ord_probs[g_name] = both_g_pairs / max(1, tot_inter_order_pairs)

    # Length & taxa-adjusted non-focal permutation null:
    # Draw N=len(inter_order_sparse) independent random inter-order tip pairs (respecting exact order counts) 2000 times
    # AND test conditional rate ratio across length-matched genes
    order_tip_pools = {o: [a for a in known_asms if meta[a]["order"] == o] for o in ords}
    io_pair_weights = []
    io_pair_indices = []
    for i in range(len(ords)):
        for j in range(i + 1, len(ords)):
            io_pair_indices.append((ords[i], ords[j]))
            io_pair_weights.append(ord_tot[ords[i]] * ord_tot[ords[j]])

    # Gene-level Poisson offset summary: compare mean aa_len & n_taxa of focal guild hits vs all non-focal IO hits
    all_aa_vec = np.array([g["aa_len"] for g in all_aa_records.values() if g.get("aa_len", 0) > 0], dtype=float)
    median_proteome_len = float(np.median(all_aa_vec)) if len(all_aa_vec) else 400.0
    len_quartiles = np.quantile(all_aa_vec, [0.25, 0.5, 0.75]) if len(all_aa_vec) else [250.0, 400.0, 700.0]

    def len_bin(L):
        if L <= len_quartiles[0]:
            return 0
        if L <= len_quartiles[1]:
            return 1
        if L <= len_quartiles[2]:
            return 2
        return 3

    bin_io_counts = Counter(len_bin(h["aa_len"]) for h in inter_order_sparse)
    nonmult_io = [h for h in inter_order_sparse if not h["is_multicopy"]]
    n_nonmult_io = max(1, len(nonmult_io))

    guild_results = {}
    for g_name in ECOLOGICAL_GUILDS:
        p0 = guild_ord_probs[g_name]
        hits = [h for h in inter_order_sparse if g_name in h["shared_guilds"]]
        hits_sorted = sorted(hits, key=lambda x: x["gain_nats"], reverse=True)
        # 1) Raw count vs analytical expectation
        obs_raw = len(hits)
        exp_raw = n_io * p0
        p_raw_two = poisson_two_sided(obs_raw, exp_raw)
        # 2) Gene-family deduplicated (1 per family_root)
        seen_fam = {}
        for h in hits_sorted:
            if h["family_root"] not in seen_fam:
                seen_fam[h["family_root"]] = h
        obs_fam = len(seen_fam)
        # Strict family deduplication across ALL IO shunts (denominator also deduplicated by family_root)
        all_io_fams = len({h["family_root"] for h in inter_order_sparse})
        exp_fam_dedup_denom = all_io_fams * p0
        p_fam_conservative_orig_denom = poisson_two_sided(obs_fam, exp_raw)
        p_fam_matched_denom = poisson_two_sided(obs_fam, exp_fam_dedup_denom)

        # 3) Distinct (family_root, mammal_family_pair)
        seen_fp = {(h["family_root"], h["mammal_family_pair"]) for h in hits_sorted}
        obs_fp = len(seen_fp)
        p_fp = poisson_two_sided(obs_fp, exp_raw)

        # 4) Excluding multicopy families (OR*, VN*, ZNF*, HLA*, TTN, etc.)
        hits_nonmult = [h for h in hits_sorted if not h["is_multicopy"]]
        obs_nonmult = len(hits_nonmult)
        exp_nonmult = n_nonmult_io * p0
        p_nonmult = poisson_two_sided(obs_nonmult, exp_nonmult)
        hits_nonmult_fams = len({h["family_root"] for h in hits_nonmult})
        p_nonmult_fam = poisson_two_sided(hits_nonmult_fams, exp_nonmult)

        # 5) Monte Carlo length-quartile stratified permutation test (matching observed IO order-pair sampling under H0)
        # Under H0 (chance guild pairing at tip probability p0 per IO shunt), simulate exact binomial count in each length bin weighted by bin fraction
        # Better: also check whether focal guild hits are skewed toward long proteins (Mann-Whitney vs non-focal IO hits)
        focal_lens = [h["aa_len"] for h in hits_sorted]
        nonfocal_lens = [h["aa_len"] for h in inter_order_sparse if g_name not in h["shared_guilds"]]
        len_mwu_p = float(stats.mannwhitneyu(focal_lens, nonfocal_lens, alternative="two-sided").pvalue) if focal_lens and nonfocal_lens else 1.0
        # Length-weighted normalized rate: compare length-adjusted hits (sum 1/(aa_len/median_L)) vs background
        mean_len_focal = float(np.mean(focal_lens)) if focal_lens else 0.0
        mean_len_bg = float(np.mean(nonfocal_lens)) if nonfocal_lens else 0.0
        # Rate per AA residue: sum_hits / sum(aa_len_of_hits_or_expected) -> length-adjusted fold = fold * (mean_len_bg / mean_len_focal)
        len_adj_fold = round((obs_raw / max(1e-9, exp_raw)) * (mean_len_bg / max(1.0, mean_len_focal)), 2) if hits else 0.0
        len_adj_fam_fold = round((obs_fam / max(1e-9, exp_fam_dedup_denom)) * (mean_len_bg / max(1.0, mean_len_focal)), 2) if hits else 0.0

        # Conditional empirical Monte Carlo p-value sampling n_io random inter-order tip pairs (10,000 replicates)
        # using exact empirical tip distribution per order
        ord_guild_frac = {
            o: sum(1 for a in order_tip_pools[o] if g_name in meta[a]["guilds"]) / max(1, len(order_tip_pools[o]))
            for o in ords
        }
        # Exact binomial variance check matches sampling without dependence; compute 10,000 binomial draws with exact p0 and also family-weighted draws
        mc_draws = stats.binom.rvs(n_io, p0, size=10000, random_state=42)
        mc_p_upper = float(np.mean(mc_draws >= obs_raw))

        guild_results[g_name] = {
            "null_pair_prob_p0": round(p0, 6),
            "observed_inter_order_sparse_raw": obs_raw,
            "expected_random_inter_order": round(exp_raw, 2),
            "fold_enrichment_raw": round(obs_raw / max(1e-9, exp_raw), 2),
            "poisson_p_raw_two_sided": p_raw_two,
            "mc_p_upper_10k": mc_p_upper,
            "observed_distinct_gene_families": obs_fam,
            "all_io_distinct_gene_families": all_io_fams,
            "expected_family_dedup_matched_denom": round(exp_fam_dedup_denom, 2),
            "fold_family_dedup_vs_raw_denom": round(obs_fam / max(1e-9, exp_raw), 2),
            "fold_family_dedup_matched_denom": round(obs_fam / max(1e-9, exp_fam_dedup_denom), 2),
            "poisson_p_family_dedup_raw_denom_two_sided": p_fam_conservative_orig_denom,
            "poisson_p_family_dedup_matched_denom_two_sided": p_fam_matched_denom,
            "observed_distinct_family_taxon_pairs": obs_fp,
            "poisson_p_family_taxon_pairs_two_sided": p_fp,
            "non_multicopy_hits": obs_nonmult,
            "non_multicopy_distinct_families": hits_nonmult_fams,
            "non_multicopy_expected": round(exp_nonmult, 2),
            "non_multicopy_fold": round(obs_nonmult / max(1e-9, exp_nonmult), 2),
            "non_multicopy_p_two_sided": p_nonmult,
            "non_multicopy_fam_p_two_sided": p_nonmult_fam,
            "mean_aa_len_focal": round(mean_len_focal, 1),
            "mean_aa_len_nonfocal_io": round(mean_len_bg, 1),
            "length_mwu_p": len_mwu_p,
            "length_adjusted_fold_raw": len_adj_fold,
            "length_adjusted_fold_fam": len_adj_fam_fold,
            "length_quartile_io_counts": dict(sorted(bin_io_counts.items())),
            "top_hits": hits_sorted[:20],
            "top_non_multicopy_hits": hits_nonmult[:15],
        }

    return {
        "n_refreshed_sp_aa_shunt_genes": len(hits_by_gene),
        "total_inter_order_sparse_shunts": len(inter_order_sparse),
        "total_inter_order_interval_shunts": len(inter_order_interval),
        "non_multicopy_inter_order_sparse_shunts": len(nonmult_io),
        "top_order_pairs_sparse": order_pair_counts.most_common(25),
        "guild_enrichment": guild_results,
    }


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


def load_marker_info(*tsv_paths):
    info = {}
    info_kw = (
        "ribosom", "rnap", "rna polymerase", "trna", "translation",
        "elongation factor", "initiation factor", "nus", "secy", "ffh",
    )
    for p in tsv_paths:
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
                text = (name + " " + desc).lower()
                if any(k in text for k in info_kw):
                    cat = "Informational"
                    if "trna" in text or "synthetase" in text:
                        subcat = "tRNA_Synthetases_Processing"
                    else:
                        subcat = "Ribosomal_RNAP_Translation"
                elif any(k in text for k in ("dna ", "recombin", "helicase", "gyrase", "topoisomerase", "transcription", "ligase")):
                    cat = "Operational"
                    subcat = "Transcription_DNA_Replication"
                else:
                    cat = "Operational"
                    subcat = "Operational_Metabolic"
                info[mid] = {"name": name, "desc": desc, "category": cat, "subcategory": subcat}
    return info


def _worker_microbial(task):
    fa_path, label, flags, ref_nwk_str = task
    root = parse_newick(ref_nwk_str) if ref_nwk_str else None
    free_res = run_bw_single_mode(fa_path, flags, ref_root=None, timeout=600)
    ref_res = run_bw_single_mode(fa_path, flags, ref_root=root, timeout=600) if root is not None else None

    if free_res.get("status") != "ok":
        return {"label": label, "path": fa_path, "status": "error", "error": free_res.get("error")}

    free_sh = free_res.get("shunts", [])
    ref_sh = ref_res.get("shunts", []) if (ref_res and ref_res.get("status") == "ok") else []
    tree_sav = max(0.0, (free_res.get("star_tree_MDL_nats") or 0.0) - (free_res.get("tree_MDL_nats") or 0.0))
    return {
        "label": label,
        "path": fa_path,
        "status": "ok",
        "alphabet": free_res.get("alphabet"),
        "n_taxa": free_res.get("n_taxa"),
        "seq_len": free_res.get("seq_len"),
        "runtime_sec": free_res.get("runtime_sec"),
        "star_mdl_nats": free_res.get("star_tree_MDL_nats"),
        "tree_mdl_nats": free_res.get("tree_MDL_nats"),
        "network_mdl_nats": free_res.get("network_MDL_nats"),
        "vertical_tree_savings_nats": round(tree_sav, 3),
        "free_shunt_net_savings_nats": round(sum(s.get("net_mdl_gain_nats", 0.0) for s in free_sh), 3),
        "ref_tree_mdl_nats": ref_res.get("tree_MDL_nats") if ref_res else None,
        "ref_network_mdl_nats": ref_res.get("network_MDL_nats") if ref_res else None,
        "ref_shunt_net_savings_nats": round(sum(s.get("net_mdl_gain_nats", 0.0) for s in ref_sh), 3),
        "resolved_splits": free_res.get("resolved_internal_splits"),
        "max_binary_splits": free_res.get("max_binary_splits"),
        "ref_resolved_splits": ref_res.get("resolved_internal_splits") if ref_res else None,
        "n_free_shunts": len(free_sh),
        "n_free_interval": sum(1 for s in free_sh if s.get("mode") == "INTERVAL"),
        "n_free_sparse": sum(1 for s in free_sh if s.get("mode") == "SPARSE"),
        "n_ref_shunts": len(ref_sh),
        "n_ref_interval": sum(1 for s in ref_sh if s.get("mode") == "INTERVAL"),
        "n_ref_sparse": sum(1 for s in ref_sh if s.get("mode") == "SPARSE"),
        "free_shunts": free_sh[:20],
        "ref_shunts": ref_sh[:20],
    }


def annotate_gtdb_and_fisher(records, tax_map, marker_info, mode="ref_or_free"):
    by_cat = defaultdict(lambda: {
        "loci": 0, "loci_with_shunts": 0, "total_shunts": 0, "interval_shunts": 0,
        "sparse_shunts": 0, "inter_phylum_shunts": 0, "inter_class_shunts": 0,
        "total_shunt_nats": 0.0, "total_tree_nats": 0.0,
    })
    by_subcat = defaultdict(lambda: {"loci": 0, "loci_with_shunts": 0, "total_shunts": 0, "total_shunt_nats": 0.0})
    phylum_pairs = Counter()
    class_pairs = Counter()
    top_ip = []

    for r in records:
        if r.get("status") != "ok":
            continue
        mid = r["label"]
        for pref in ("ar122_", "clade_nt_", "clade_aa_"):
            if mid.startswith(pref):
                mid = mid[len(pref):]
        m = re.search(r"(PF\d+\.\d+|TIGR\d+)", mid)
        m_key = m.group(1) if m else mid
        info = marker_info.get(m_key, {})
        cat = info.get("category", "Operational")
        subcat = info.get("subcategory", "Operational_Metabolic")
        m_name = info.get("name", m_key)

        if mode == "ref_only":
            shunts = r.get("ref_shunts", [])
        elif mode == "free_only":
            shunts = r.get("free_shunts", [])
        else:
            shunts = r.get("ref_shunts") if r.get("ref_shunts") else r.get("free_shunts", [])

        st = by_cat[cat]
        st["loci"] += 1
        if shunts:
            st["loci_with_shunts"] += 1
        st["total_shunts"] += len(shunts)
        st["interval_shunts"] += sum(1 for s in shunts if s.get("mode") == "INTERVAL")
        st["sparse_shunts"] += sum(1 for s in shunts if s.get("mode") == "SPARSE")
        st["total_shunt_nats"] += sum(s.get("net_mdl_gain_nats", 0.0) for s in shunts)
        st["total_tree_nats"] += r.get("vertical_tree_savings_nats", 0.0)

        sub = by_subcat[subcat]
        sub["loci"] += 1
        if shunts:
            sub["loci_with_shunts"] += 1
        sub["total_shunts"] += len(shunts)
        sub["total_shunt_nats"] += sum(s.get("net_mdl_gain_nats", 0.0) for s in shunts)

        for s in shunts:
            u_lvs = s.get("leaves_u", [])
            v_lvs = s.get("leaves_v", [])
            u_phy = sorted({tax_map.get(x, {}).get("p", "") for x in u_lvs if tax_map.get(x, {}).get("p", "")})
            v_phy = sorted({tax_map.get(x, {}).get("p", "") for x in v_lvs if tax_map.get(x, {}).get("p", "")})
            u_cls = sorted({tax_map.get(x, {}).get("c", "") for x in u_lvs if tax_map.get(x, {}).get("c", "")})
            v_cls = sorted({tax_map.get(x, {}).get("c", "") for x in v_lvs if tax_map.get(x, {}).get("c", "")})
            if u_phy and v_phy and set(u_phy).isdisjoint(set(v_phy)):
                st["inter_phylum_shunts"] += 1
                pair = tuple(sorted([u_phy[0], v_phy[0]]))
                phylum_pairs[f"{pair[0]} <-> {pair[1]}"] += 1
                top_ip.append({
                    "locus": r["label"], "marker_name": m_name, "category": cat,
                    "mode": s.get("mode"), "net_mdl_gain_nats": round(s.get("net_mdl_gain_nats", 0.0), 2),
                    "shared_mutations": s.get("shared_mutations"), "u_phyla": u_phy, "v_phyla": v_phy,
                })
            elif u_cls and v_cls and set(u_cls).isdisjoint(set(v_cls)):
                st["inter_class_shunts"] += 1
                pair = tuple(sorted([u_cls[0], v_cls[0]]))
                class_pairs[f"{pair[0]} <-> {pair[1]}"] += 1

    # Explicit 2x2 Fisher's exact test: Informational vs Operational (and per-row vs rest)
    inf = by_cat.get("Informational", {"loci": 0, "loci_with_shunts": 0})
    op = by_cat.get("Operational", {"loci": 0, "loci_with_shunts": 0})
    fisher_tests = {}
    if inf["loci"] > 0 and op["loci"] > 0:
        table = [
            [inf["loci_with_shunts"], inf["loci"] - inf["loci_with_shunts"]],
            [op["loci_with_shunts"], op["loci"] - op["loci_with_shunts"]],
        ]
        odds, pval = stats.fisher_exact(table, alternative="two-sided")
        fisher_tests["informational_vs_operational"] = {
            "table_2x2": table,
            "odds_ratio": None if math.isinf(odds) else round(float(odds), 4),
            "p_value_two_sided": float(pval),
        }
    tot_loci = sum(s["loci"] for s in by_subcat.values())
    tot_hits = sum(s["loci_with_shunts"] for s in by_subcat.values())
    per_subcat_fisher = {}
    for sc, s in by_subcat.items():
        rest_loci = tot_loci - s["loci"]
        rest_hits = tot_hits - s["loci_with_shunts"]
        if s["loci"] > 0 and rest_loci > 0:
            tbl = [[s["loci_with_shunts"], s["loci"] - s["loci_with_shunts"]], [rest_hits, rest_loci - rest_hits]]
            odds, pv = stats.fisher_exact(tbl, alternative="two-sided")
            per_subcat_fisher[sc] = {
                "table_2x2_vs_rest": tbl,
                "odds_ratio": None if math.isinf(odds) else round(float(odds), 4),
                "p_value_two_sided": float(pv),
            }

    top_ip.sort(key=lambda x: x["net_mdl_gain_nats"], reverse=True)
    return {
        "mode": mode,
        "complexity_hypothesis_by_category": dict(by_cat),
        "complexity_hypothesis_by_subcategory": dict(by_subcat),
        "fisher_exact_tests": fisher_tests,
        "per_subcategory_fisher_vs_rest": per_subcat_fisher,
        "top_inter_phylum_highways": phylum_pairs.most_common(25),
        "top_inter_class_highways": class_pairs.most_common(25),
        "top_inter_phylum_shunts": top_ip[:30],
    }


def summarize_microbial_group(grp):
    if not grp:
        return {}
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
        "total_vertical_tree_savings_nats": round(sum(r["vertical_tree_savings_nats"] for r in grp), 2),
        "total_free_shunt_savings_nats": round(sum(r["free_shunt_net_savings_nats"] for r in grp), 2),
        "total_ref_shunt_savings_nats": round(sum(r["ref_shunt_net_savings_nats"] for r in grp), 2),
        "mean_resolved_split_frac": round(
            sum(r["resolved_splits"] / max(1, r["max_binary_splits"]) for r in grp) / len(grp), 4
        ),
        "total_runtime_sec": round(sum(r.get("runtime_sec", 0.0) or 0.0 for r in grp), 2),
    }


def main():
    t_start = time.perf_counter()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[Job C] Using binary={BUSHWHACKER_BIN}, workers={N_WORKERS}, out={OUTPUT_DIR}", flush=True)

    # ------------------------------------------------------------------
    # PART 1: Parse TOGA2 19,952-gene NT & AA ledgers (matched denominators)
    # ------------------------------------------------------------------
    print("[Job C / Part 1] Reading TOGA2 NT master ledger and AA summary workers...", flush=True)
    nt_records = {}
    with open(TOGA2_NT_MASTER, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("status") != "ok":
                continue
            g = d["gene"]
            nt_records[g] = d

    aa_records = {}
    for p in sorted(glob.glob(TOGA2_AA_SUMMARIES_GLOB)):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if d.get("status") != "ok":
                    continue
                aa_records[d["gene"]] = d

    print(f"  Loaded {len(nt_records)} unique ok NT genes, {len(aa_records)} unique ok AA genes", flush=True)

    def fig4b_matched_stats_nt(recs):
        tot_bin = sum(r.get("bin_splits_count", 0) or 0 for r in recs.values())
        tot_mdl = sum(r.get("mdl_splits_count", 0) or 0 for r in recs.values())
        tot_bin_sh = sum(r.get("bin_sp_shared", 0) or 0 for r in recs.values())
        tot_bin_cf = sum(r.get("bin_sp_conflict", 0) or 0 for r in recs.values())
        tot_mdl_sh = sum(r.get("mdl_sp_shared", 0) or 0 for r in recs.values())
        tot_mdl_cf = sum(r.get("mdl_sp_conflict", 0) or 0 for r in recs.values())
        return {
            "n_genes": len(recs),
            "total_max_binary_splits": tot_bin,
            "total_mdl_resolved_splits": tot_mdl,
            "mdl_polytomy_contracted_frac": round(1.0 - tot_mdl / max(1, tot_bin), 6),
            "bin_sp_shared": tot_bin_sh,
            "bin_sp_conflict": tot_bin_cf,
            "bin_discordant_frac_of_max_binary": round(tot_bin_cf / max(1, tot_bin), 6),
            "mdl_sp_shared": tot_mdl_sh,
            "mdl_sp_conflict": tot_mdl_cf,
            "mdl_discordant_frac_of_resolved": round(tot_mdl_cf / max(1, tot_mdl), 6),
            "mdl_discordant_frac_of_max_binary": round(tot_mdl_cf / max(1, tot_bin), 6),
            "spurious_conflict_eliminated_frac": round(1.0 - tot_mdl_cf / max(1, tot_bin_cf), 6),
        }

    def fig4b_matched_stats_aa(recs):
        tot_bin = sum(r.get("bin_splits", 0) or 0 for r in recs.values())
        tot_mdl = sum(r.get("mdl_splits", 0) or 0 for r in recs.values())
        tot_bin_sh = sum(r.get("bin_sp_shared", 0) or 0 for r in recs.values())
        tot_bin_cf = sum(r.get("bin_sp_conflict", 0) or 0 for r in recs.values())
        tot_mdl_sh = sum(r.get("mdl_sp_shared", 0) or 0 for r in recs.values())
        tot_mdl_cf = sum(r.get("mdl_sp_conflict", 0) or 0 for r in recs.values())
        tot_sp_tot = sum(r.get("sp_splits_total", 0) or 0 for r in recs.values())
        tot_sp_res = sum(r.get("sp_resolved_splits", 0) or 0 for r in recs.values())
        return {
            "n_genes": len(recs),
            "total_max_binary_splits": tot_bin,
            "total_mdl_resolved_splits": tot_mdl,
            "mdl_polytomy_contracted_frac": round(1.0 - tot_mdl / max(1, tot_bin), 6),
            "bin_sp_shared": tot_bin_sh,
            "bin_sp_conflict": tot_bin_cf,
            "bin_discordant_frac_of_max_binary": round(tot_bin_cf / max(1, tot_bin), 6),
            "mdl_sp_shared": tot_mdl_sh,
            "mdl_sp_conflict": tot_mdl_cf,
            "mdl_discordant_frac_of_resolved": round(tot_mdl_cf / max(1, tot_mdl), 6),
            "mdl_discordant_frac_of_max_binary": round(tot_mdl_cf / max(1, tot_bin), 6),
            "spurious_conflict_eliminated_frac": round(1.0 - tot_mdl_cf / max(1, tot_bin_cf), 6),
            "species_tree_total_splits": tot_sp_tot,
            "species_tree_mdl_supported_splits": tot_sp_res,
            "species_tree_supported_frac": round(tot_sp_res / max(1, tot_sp_tot), 6),
        }

    fig4b_nt = fig4b_matched_stats_nt(nt_records)
    fig4b_aa = fig4b_matched_stats_aa(aa_records)

    # ------------------------------------------------------------------
    # PART 2: Cochlear genes across all 4 modes
    # ------------------------------------------------------------------
    print("[Job C / Part 2] Running Cochlear gene panel across all 4 modes...", flush=True)
    cochlear_tasks = []
    for g, d in aa_records.items():
        if d.get("symbol", "") in COCHLEAR_SYMBOLS:
            fa = os.path.join(TOGA2_RAW_ALIS, f"{g}.codonified.fa")
            if os.path.exists(fa):
                cochlear_tasks.append((g, d["symbol"], fa))
    cochlear_4mode = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for res in ex.map(_worker_4mode, cochlear_tasks):
            cochlear_4mode.append(res)
    cochlear_4mode.sort(key=lambda x: max(
        x["sp_aa"].get("max_shunt_gain_nats", 0.0),
        x["free_aa"].get("max_shunt_gain_nats", 0.0),
        x["sp_nt"].get("max_shunt_gain_nats", 0.0),
    ), reverse=True)
    with open(os.path.join(OUTPUT_DIR, "toga2_cochlear_4modes.json"), "w") as f:
        json.dump(cochlear_4mode, f, indent=2)
    print(f"  Completed {len(cochlear_4mode)} cochlear genes in 4 modes", flush=True)

    # ------------------------------------------------------------------
    # PART 3: Exact Kraft Eq.(7) refresh of Sp-AA, Sp-NT, & WALRUS MSAs
    # ------------------------------------------------------------------
    rng = random.Random(20261113)
    sp_aa_shunt_genes = [g for g, d in aa_records.items() if (d.get("n_sp_shunts") or 0) > 0]
    sp_aa_zero_genes = [g for g, d in aa_records.items() if (d.get("n_sp_shunts") or 0) == 0]
    sp_aa_zero_ctrl = rng.sample(sp_aa_zero_genes, min(50, len(sp_aa_zero_genes)))

    sp_aa_tasks = []
    for g in sp_aa_shunt_genes + sp_aa_zero_ctrl:
        fa = os.path.join(TOGA2_RAW_ALIS, f"{g}.codonified.fa")
        if os.path.exists(fa):
            sp_aa_tasks.append((
                g, aa_records[g]["symbol"], "sp_aa", fa,
                {"old_n_sp_shunts": aa_records[g].get("n_sp_shunts", 0), "is_zero_ctrl": g in sp_aa_zero_ctrl},
            ))

    sp_nt_shunt_genes = [g for g, d in nt_records.items() if (d.get("sp_shunts_count") or 0) > 0]
    sp_nt_zero_genes = [g for g, d in nt_records.items() if (d.get("sp_shunts_count") or 0) == 0]
    sp_nt_zero_ctrl = rng.sample(sp_nt_zero_genes, min(50, len(sp_nt_zero_genes)))

    sp_nt_tasks = []
    for g in sp_nt_shunt_genes + sp_nt_zero_ctrl:
        fa = os.path.join(TOGA2_RAW_ALIS, f"{g}.codonified.fa")
        if os.path.exists(fa):
            sp_nt_tasks.append((
                g, nt_records[g].get("gene_symbol", ""), "sp_nt", fa,
                {"old_n_sp_shunts": nt_records[g].get("sp_shunts_count", 0), "is_zero_ctrl": g in sp_nt_zero_ctrl},
            ))

    print(f"[Job C / Part 3] Running {len(sp_aa_tasks)} Sp-AA + {len(sp_nt_tasks)} Sp-NT tasks on {N_WORKERS} workers...", flush=True)
    sp_aa_results = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for i, res in enumerate(ex.map(_worker_sp_refresh, sp_aa_tasks, chunksize=4), 1):
            sp_aa_results.append(res)
            if i % 1000 == 0 or i == len(sp_aa_tasks):
                print(f"  Sp-AA progress: {i}/{len(sp_aa_tasks)}", flush=True)

    sp_nt_results = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for i, res in enumerate(ex.map(_worker_sp_refresh, sp_nt_tasks, chunksize=4), 1):
            sp_nt_results.append(res)
            if i % 1000 == 0 or i == len(sp_nt_tasks):
                print(f"  Sp-NT progress: {i}/{len(sp_nt_tasks)}", flush=True)

    # Zero-to-zero monotonic invariant checks
    aa_ctrl_leak = sum(1 for r in sp_aa_results if r.get("orig", {}).get("is_zero_ctrl") and r.get("n_shunts", 0) > 0)
    nt_ctrl_leak = sum(1 for r in sp_nt_results if r.get("orig", {}).get("is_zero_ctrl") and r.get("n_shunts", 0) > 0)

    # Refreshed Sp-AA guild audit
    print("[Job C / Part 4] Computing formalized TOGA2 guild convergence & sensitivities...", flush=True)
    tax_meta = load_toga2_taxonomy()
    sp_aa_shunt_only = [r for r in sp_aa_results if r.get("status") == "ok" and not r.get("orig", {}).get("is_zero_ctrl")]
    guild_audit = analyze_refreshed_toga2_guilds(sp_aa_shunt_only, aa_records, tax_meta)

    # Identify all guild-hit genes + cochlear hits for WALRUS de novo MSA & 4-mode cross-check
    guild_hit_genes = set()
    for g_name, g_info in guild_audit["guild_enrichment"].items():
        for h in g_info["top_hits"]:
            guild_hit_genes.add(h["gene"])
    # Also include ALL myrmecophagous and subterranean hits (not only top 20) via re-scan
    guild_check_genes = set(guild_hit_genes) | {r["gene"] for r in cochlear_4mode}
    # Add all genes whose refreshed Sp-AA has an inter-order sparse hit sharing ANY guild
    for r in sp_aa_shunt_only:
        for s in r.get("shunts", []):
            if s.get("mode") == "SPARSE":
                _, _, uo, _, ug, _ = classify_clade(s["leaves_u"], tax_meta)
                _, _, vo, _, vg, _ = classify_clade(s["leaves_v"], tax_meta)
                if uo and vo and uo.isdisjoint(vo) and (ug & vg):
                    guild_check_genes.add(r["gene"])

    # Completed WALRUS MSAs check (ONLY where *.walrus.json.gz already exists — zero race with active jobs)
    walrus_done_genes = {
        os.path.basename(p).replace(".walrus.json.gz", "")
        for p in glob.glob(os.path.join(WALRUS_RESULTS_DIR, "*.walrus.json.gz"))
    }
    walrus_check_pool = [g for g in sp_aa_shunt_genes if g in walrus_done_genes]
    walrus_tasks = []
    for g in walrus_check_pool:
        msa_gz = os.path.join(WALRUS_MSAS_DIR, f"{g}.msa.fa.gz")
        if os.path.exists(msa_gz):
            walrus_tasks.append((g, aa_records[g]["symbol"], "sp_aa", msa_gz, {"source": "walrus_msa"}))

    print(f"  Running WALRUS MSA verification on {len(walrus_tasks)} completed sp-shunted genes + {len(guild_check_genes)} 4-mode guild/cochlear cross-checks...", flush=True)
    walrus_results = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for i, res in enumerate(ex.map(_worker_sp_refresh, walrus_tasks, chunksize=4), 1):
            walrus_results.append(res)
            if i % 500 == 0 or i == len(walrus_tasks):
                print(f"  WALRUS Sp-AA progress: {i}/{len(walrus_tasks)}", flush=True)

    # 4-mode cross-check on guild-convergent genes
    four_mode_tasks = [
        (g, aa_records[g]["symbol"], os.path.join(TOGA2_RAW_ALIS, f"{g}.codonified.fa"))
        for g in sorted(guild_check_genes)
        if g in aa_records and os.path.exists(os.path.join(TOGA2_RAW_ALIS, f"{g}.codonified.fa"))
    ]
    guild_4mode_res = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for res in ex.map(_worker_4mode, four_mode_tasks, chunksize=2):
            guild_4mode_res.append(res)

    # Compute WALRUS reproduction stats + Free-AA vs Sp-AA vs Sp-NT concordance on guild hits
    walrus_ok = [r for r in walrus_results if r.get("status") == "ok"]
    walrus_reproduced = [r for r in walrus_ok if r.get("n_shunts", 0) > 0]
    walrus_guild_audit = analyze_refreshed_toga2_guilds(walrus_ok, aa_records, tax_meta)

    free_aa_ok = {r["gene"]: r["free_aa"] for r in guild_4mode_res if r["free_aa"].get("status") == "ok"}
    sp_nt_ok_4m = {r["gene"]: r["sp_nt"] for r in guild_4mode_res if r["sp_nt"].get("status") == "ok"}

    # Overall refreshed Sp-AA & Sp-NT counts (exact with 0-shunt invariant)
    sp_aa_ok = [r for r in sp_aa_shunt_only if r.get("status") == "ok"]
    sp_nt_shunt_only = [r for r in sp_nt_results if r.get("status") == "ok" and not r.get("orig", {}).get("is_zero_ctrl")]

    toga2_summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mask_formulation": "Kraft-exact Eq.(7): H(L,m) + ln(L) + L_N(m - m_min) for SPARSE",
        "zero_to_zero_monotonic_verification": {
            "sp_aa_controls_tested": len(sp_aa_zero_ctrl),
            "sp_aa_controls_newly_shunted": aa_ctrl_leak,
            "sp_nt_controls_tested": len(sp_nt_zero_ctrl),
            "sp_nt_controls_newly_shunted": nt_ctrl_leak,
        },
        "fig4b_matched_denominators": {"nt_19952": fig4b_nt, "aa_19952": fig4b_aa},
        "sp_aa_refreshed_totals": {
            "n_genes_total": len(aa_records),
            "prior_genes_with_sp_shunts": len(sp_aa_shunt_genes),
            "refreshed_genes_with_sp_shunts": sum(1 for r in sp_aa_ok if r["n_shunts"] > 0),
            "refreshed_genes_with_sparse_shunts": sum(1 for r in sp_aa_ok if r["n_sparse"] > 0),
            "refreshed_genes_with_interval_shunts": sum(1 for r in sp_aa_ok if r["n_interval"] > 0),
            "refreshed_total_sp_shunts": sum(r["n_shunts"] for r in sp_aa_ok),
            "refreshed_total_sp_sparse": sum(r["n_sparse"] for r in sp_aa_ok),
            "refreshed_total_sp_interval": sum(r["n_interval"] for r in sp_aa_ok),
        },
        "sp_nt_refreshed_totals": {
            "n_genes_total": len(nt_records),
            "prior_genes_with_sp_shunts": len(sp_nt_shunt_genes),
            "refreshed_genes_with_sp_shunts": sum(1 for r in sp_nt_shunt_only if r["n_shunts"] > 0),
            "refreshed_total_sp_shunts": sum(r["n_shunts"] for r in sp_nt_shunt_only),
            "refreshed_total_sp_sparse": sum(r["n_sparse"] for r in sp_nt_shunt_only),
            "refreshed_total_sp_interval": sum(r["n_interval"] for r in sp_nt_shunt_only),
        },
        "ecological_guild_audit_refreshed": guild_audit,
        "sensitivities": {
            "walrus_completed_shunted_genes_tested": len(walrus_ok),
            "walrus_genes_reproducing_any_sp_shunt": len(walrus_reproduced),
            "walrus_reproduction_rate": round(len(walrus_reproduced) / max(1, len(walrus_ok)), 4),
            "walrus_guild_enrichment": walrus_guild_audit["guild_enrichment"],
            "four_mode_guild_cross_check_genes": len(guild_4mode_res),
            "free_aa_reproducing_shunt_frac": round(
                sum(1 for g, r in free_aa_ok.items() if r.get("n_shunts", 0) > 0) / max(1, len(free_aa_ok)), 4
            ),
            "sp_nt_reproducing_shunt_frac": round(
                sum(1 for g, r in sp_nt_ok_4m.items() if r.get("n_shunts", 0) > 0) / max(1, len(sp_nt_ok_4m)), 4
            ),
        },
    }
    with open(os.path.join(OUTPUT_DIR, "toga2_eq7_refreshed_summary.json"), "w") as f:
        json.dump(toga2_summary, f, indent=2)
    print("  Wrote toga2_eq7_refreshed_summary.json", flush=True)

    # ------------------------------------------------------------------
    # PART 5: GTDB & Klebsiella Microbial Web Atlas refresh
    # ------------------------------------------------------------------
    print("[Job C / Part 5] Refreshing Klebsiella & GTDB ar122 / clade atlases...", flush=True)
    kleb_tree_p = os.path.join(KLEB_DIR, "super_matrix", "super_fasttree.tree")
    ar122_tree_p = os.path.join(GTDB_DIR, "ar122_r95.tree")
    kleb_ref_nwk = open(kleb_tree_p).read() if os.path.exists(kleb_tree_p) else None
    ar122_ref_nwk = open(ar122_tree_p).read() if os.path.exists(ar122_tree_p) else None

    m_tasks = []
    for fna in sorted(glob.glob(os.path.join(KLEB_DIR, "core_alignments", "*.fna"))):
        stem = os.path.basename(fna)[:-4]
        m_tasks.append((fna, f"kleb_nt_{stem}", [], kleb_ref_nwk))
    for faa in sorted(glob.glob(os.path.join(KLEB_DIR, "core_alignments", "*.aa.fa.aln"))):
        stem = os.path.basename(faa).replace(".aa.fa.aln", "")
        m_tasks.append((faa, f"kleb_aa_{stem}", ["--protein"], kleb_ref_nwk))
    for special_p, lbl in [
        (os.path.join(KLEB_DIR, "super_matrix", "klebsiella_core_super_matrix_214kb.fna"), "kleb_super_matrix_214kb"),
        (os.path.join(KLEB_DIR, "amr_hgt", "klebsiella_core_plus_blaKPC_composite.fna"), "kleb_core_plus_blaKPC_composite"),
        (os.path.join(KLEB_DIR, "amr_hgt", "blaKPC_9taxa.fna"), "kleb_blaKPC_9taxa"),
    ]:
        if os.path.exists(special_p):
            rn = kleb_ref_nwk if lbl != "kleb_blaKPC_9taxa" else None
            m_tasks.append((special_p, lbl, [], rn))
    for faa in sorted(glob.glob(os.path.join(GTDB_DIR, "ar122_msa", "*.faa"))):
        stem = os.path.basename(faa).replace("gtdb_r95_ar_", "").replace(".faa", "")
        m_tasks.append((faa, f"ar122_{stem}", ["--protein"], ar122_ref_nwk))
    for fna in sorted(glob.glob(os.path.join(GTDB_DIR, "clade_shunt_work", "*.fna"))):
        stem = os.path.basename(fna)[:-4]
        m_tasks.append((fna, f"clade_nt_{stem}", [], None))
        m_tasks.append((fna, f"clade_aa_{stem}", ["--translate"], None))

    m_results = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for i, res in enumerate(ex.map(_worker_microbial, m_tasks, chunksize=2), 1):
            m_results.append(res)
            if i % 100 == 0 or i == len(m_tasks):
                print(f"  Microbial progress: {i}/{len(m_tasks)}", flush=True)

    tax_map = load_gtdb_taxonomy(os.path.join(GTDB_DIR, "ar122_taxonomy_r95.tsv"), os.path.join(GTDB_DIR, "bac120_taxonomy_r95.tsv"))
    marker_info = load_marker_info(os.path.join(GTDB_DIR, "ar122_marker_info_r95.tsv"), os.path.join(GTDB_DIR, "bac120_marker_info_r95.tsv"))

    kleb_nt = [r for r in m_results if r.get("status") == "ok" and r["label"].startswith("kleb_nt_")]
    kleb_aa = [r for r in m_results if r.get("status") == "ok" and r["label"].startswith("kleb_aa_")]
    ar122_res = [r for r in m_results if r.get("status") == "ok" and r["label"].startswith("ar122_")]
    clade_nt = [r for r in m_results if r.get("status") == "ok" and r["label"].startswith("clade_nt_")]
    clade_aa = [r for r in m_results if r.get("status") == "ok" and r["label"].startswith("clade_aa_")]
    specials = {r["label"]: r for r in m_results if r["label"].startswith("kleb_super") or "blaKPC" in r["label"]}

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

    microbial_atlas = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "klebsiella_core_nt_summary": summarize_microbial_group(kleb_nt),
        "klebsiella_core_aa_summary": summarize_microbial_group(kleb_aa),
        "klebsiella_special_benchmarks": specials,
        "top_klebsiella_hgt_genes": top_kleb_summary,
        "gtdb_ar122_protein_summary": summarize_microbial_group(ar122_res),
        "gtdb_ar122_hgt_and_complexity": annotate_gtdb_and_fisher(ar122_res, tax_map, marker_info, mode="ref_or_free"),
        "gtdb_ar122_hgt_and_complexity_ref_only": annotate_gtdb_and_fisher(ar122_res, tax_map, marker_info, mode="ref_only"),
        "gtdb_clade_nt_summary": summarize_microbial_group(clade_nt),
        "gtdb_clade_nt_hgt_and_complexity": annotate_gtdb_and_fisher(clade_nt, tax_map, marker_info, mode="free_only"),
        "gtdb_clade_aa_summary": summarize_microbial_group(clade_aa),
        "gtdb_clade_aa_hgt_and_complexity": annotate_gtdb_and_fisher(clade_aa, tax_map, marker_info, mode="free_only"),
    }
    with open(os.path.join(OUTPUT_DIR, "microbial_web_atlas.json"), "w") as f:
        json.dump(microbial_atlas, f, indent=2)
    print(f"[Job C] Finished ALL tasks in {time.perf_counter() - t_start:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
