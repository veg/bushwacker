#!/usr/bin/env python3
"""
Pipeline 4: Cracking Biology's Classic 'Hard Polytomies' Across All 19 Rokas/Shen (2018)
Phylogenomic Matrices (55,479 Single-Gene Alignments in 20-State Amino Acids & DNA).

Usage:
  python3 p4_macroevo_rokas_worker.py <dataset_index_0_to_18_or_name> <num_workers>
  python3 p4_macroevo_rokas_worker.py --aggregate
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
import time
import shutil
import tempfile
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

BUSHWHACKER_BIN = resolve_bin("BUSHWHACKER_BIN", REPO_ROOT / "target" / "release" / "bushwhacker", "bushwhacker")
ROKAS_ROOT = str(EXTERNAL_DIR / "rokas2018")
ALN_ROOT = os.path.join(ROKAS_ROOT, "single-gene_alignments")
COAL_ROOT = os.path.join(ROKAS_ROOT, "coalescent_species_trees")
BASE_DIR = str(WORK_ROOT / "macroevo_polytomy_atlas")

DATASETS = [
    "BoroA6", "WhelA7", "ChenA4", "MisoA2", "MisoD2a", "MisoD2b",
    "WickA3", "WickD3a", "WickD3b", "XiD4", "YangA8", "NagyA1",
    "ShenA9", "StruA5", "PrumD6", "SongD1", "JarvD5b", "TarvD7", "JarvD5a"
]

PROTEIN_DATASETS = {
    "BoroA6", "WhelA7", "ChenA4", "MisoA2", "WickA3", "YangA8", "NagyA1", "ShenA9", "StruA5"
}


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
            bl = 0.0
            if i + 1 < n and not tokens[i + 1].startswith(":") and tokens[i + 1] not in ("(", ")", ",", ";"):
                i += 1
                lbl = tokens[i].strip("'")
            if i + 1 < n and tokens[i + 1].startswith(":"):
                i += 1
                try:
                    bl = float(tokens[i][1:])
                except ValueError:
                    bl = 0.0
            stack[-1].append({"children": children, "name": lbl, "bl": bl})
        elif tok.startswith(":"):
            pass
        else:
            lbl = tok.strip("'")
            bl = 0.0
            if i + 1 < n and tokens[i + 1].startswith(":"):
                i += 1
                try:
                    bl = float(tokens[i][1:])
                except ValueError:
                    bl = 0.0
            stack[-1].append({"children": [], "name": lbl, "bl": bl})
        i += 1
    return stack[0][0] if stack[0] else {"children": [], "name": None, "bl": 0.0}


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


def get_splits_with_support(node, all_tips=None):
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
    splits = {}
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
            sup = 1.0
            if u.get("name"):
                try:
                    sup = float(u["name"])
                except ValueError:
                    sup = 1.0
            splits[canon] = sup
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


def canon_split(subset, all_tips):
    s = frozenset(subset & all_tips)
    k = len(s)
    total = len(all_tips)
    if k < 2 or k > total - 2:
        return None
    comp = all_tips - s
    if k < len(comp):
        return s
    if len(comp) < k:
        return frozenset(comp)
    return s if min(s) < min(comp) else frozenset(comp)


def get_hypothesis_splits(ds_name, all_tips):
    """Returns dict of {hypothesis_name: canonical_bipartition_frozenset} for focal controversies."""
    hyps = {}
    if ds_name == "BoroA6":
        outg = {"Monosiga", "Salpingoeca"} & all_tips
        cteno = {"Mnemiopsis"} & all_tips
        sponge = {"Amphimedon"} & all_tips
        rest = all_tips - outg - cteno - sponge
        if outg and cteno and sponge and len(rest) >= 2:
            # Ctenophora-sister: (Porifera + Rest) | (Outgroup + Ctenophora)
            hyps["Ctenophora_sister"] = canon_split(sponge | rest, all_tips)
            # Porifera-sister: (Ctenophora + Rest) | (Outgroup + Porifera)
            hyps["Porifera_sister"] = canon_split(cteno | rest, all_tips)
            # Coelenterata/Diploblast monophyly check: (Ctenophora + Porifera) | (Outgroup + Rest)
            hyps["Ctenophora_plus_Porifera"] = canon_split(cteno | sponge, all_tips)
    elif ds_name == "WhelA7":
        outg = {
            "Salpingoeca_rosetta", "Monosiga_ovata", "Capsaspora_owczarzaki", "Ministeria_vibrans",
            "Sphaeroforma_arctica", "Amoebidium_parasiticum", "Allomyces_macrogynus",
            "Spizellomyces_punctatus", "Rhizopus_oryzae", "Mortierella_verticillata"
        } & all_tips
        cteno = {
            "Euplokamis_dunlapae", "Pleurobrachia_atlantica", "Pleurobrachia_bachei",
            "Dryodora_glandiformis", "Beroe_abyssicola", "Mnemiopsis_leidyi",
            "Mertensiidae_sp", "Bolinopsis_infundibulum", "Vallicula_sp", "Coeloplana_astericola"
        } & all_tips
        sponge = {
            "Oscarella_carmela", "Corticium_candelabrum", "Sycon_ciliatum", "Sycon_coactum",
            "Hyalonema_populiferum", "Aphrocallistes_vastus", "Rossella_fibulata", "Sympagella_nux",
            "Chondrilla_nucula", "Ircinia_fasciculata", "Amphimedon_queenslandica", "Petrosia_ficiformis",
            "Spongilla_alba", "Ephydatia_muelleri", "Tethya_wilhelma", "Pseudospongosorites_suberitoides",
            "Latrunculia_apicalis", "Crella_elegans", "Kirkpatrickia_variolosa"
        } & all_tips
        rest = all_tips - outg - cteno - sponge
        if outg and cteno and sponge and len(rest) >= 2:
            hyps["Ctenophora_sister"] = canon_split(sponge | rest, all_tips)
            hyps["Porifera_sister"] = canon_split(cteno | rest, all_tips)
            hyps["Ctenophora_plus_Porifera"] = canon_split(cteno | sponge, all_tips)
    elif ds_name == "ChenA4":
        lungfish = {"Lepidosiren", "Protopterus"} & all_tips
        coelacanth = {"Latimeria"} & all_tips
        fishes_out = {
            "Polypterus", "Acipenser", "Lepisosteus", "Anguilla", "Osteoglossum", "Danio",
            "Astyanax", "Ictalurus", "Gadus", "Gasterosteus", "Takifugu", "Oreochromis",
            "Oryzias", "Xiphophorus", "Cmi", "Ler", "Sca"
        } & all_tips
        tetrapods = all_tips - lungfish - coelacanth - fishes_out
        if lungfish and coelacanth and fishes_out and len(tetrapods) >= 2:
            hyps["Lungfish_sister_to_Tetrapods"] = canon_split(lungfish | tetrapods, all_tips)
            hyps["Coelacanth_sister_to_Tetrapods"] = canon_split(coelacanth | tetrapods, all_tips)
            hyps["Lungfish_plus_Coelacanth"] = canon_split(lungfish | coelacanth, all_tips)
        turtles = {"Pelodiscus", "Trachemys"} & all_tips
        archosaurs = {"Alligator", "TGE", "Emu", "Ostrich", "Gallus", "Anas", "falco", "pigeon", "Melopsittacus", "Ficedula", "Sparrow", "Taeniopygia"} & all_tips
        lepidosaurs = {"Sphenodon", "Gecko", "skink", "Anolis", "YBK", "python"} & all_tips
        if turtles and archosaurs and lepidosaurs and (all_tips - turtles - archosaurs - lepidosaurs):
            hyps["Archelosauria_Turtles_plus_Archosaurs"] = canon_split(turtles | archosaurs, all_tips)
            hyps["Ankylopoda_Turtles_plus_Lepidosaurs"] = canon_split(turtles | lepidosaurs, all_tips)
            hyps["Archosaurs_plus_Lepidosaurs_Turtles_basal"] = canon_split(archosaurs | lepidosaurs, all_tips)
    elif ds_name in ("SongD1", "TarvD7"):
        afro = {"Ech", "Lox", "Pro"} & all_tips
        xena = {"Cho", "Das"} & all_tips
        outg = {"Mac", "Mon", "Orn", "Gal"} & all_tips
        boreo = all_tips - afro - xena - outg
        if afro and xena and outg and len(boreo) >= 2:
            hyps["Atlantogenata_Afro_plus_Xena"] = canon_split(afro | xena, all_tips)
            hyps["Exafroplacentalia_Boreo_plus_Xena"] = canon_split(boreo | xena, all_tips)
            hyps["Epitheria_Boreo_plus_Afro"] = canon_split(boreo | afro, all_tips)
    return {k: v for k, v in hyps.items() if v is not None}


def support_to_nats(sup):
    sup_c = min(max(sup, 1e-12), 1.0 - 1e-12)
    return round(float(__import__("math").log(sup_c / (1.0 - sup_c))), 3)


def process_one_locus(args):
    fa_path, ds_name, is_prot, sp_nwk_str = args
    shm_root = "/dev/shm" if os.path.isdir("/dev/shm") else None
    tmpdir = tempfile.mkdtemp(prefix="bw_p4_", dir=shm_root)
    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = "1"
    flags = ["--protein"] if is_prot else []
    try:
        out_nwk = os.path.join(tmpdir, "mdl.nwk")
        out_bin = os.path.join(tmpdir, "bin.nwk")
        out_json = os.path.join(tmpdir, "mdl.json")
        res = subprocess.run(
            [BUSHWHACKER_BIN] + flags + [fa_path, "-o", out_nwk, "-b", out_bin, "-j", out_json],
            capture_output=True, text=True, env=env, timeout=300
        )
        if res.returncode != 0 or not os.path.exists(out_json):
            return {"locus": os.path.basename(fa_path), "status": "error"}

        d = json.load(open(out_json))
        mdl_root = parse_newick(d["newick"])
        bin_root = parse_newick(d["binary_newick"])
        mdl_splits_map, kept_tips = get_splits_with_support(mdl_root)
        bin_splits_map, _ = get_splits_with_support(bin_root, kept_tips)
        mdl_splits = set(mdl_splits_map.keys())
        bin_splits = set(bin_splits_map.keys())

        sp_splits = set()
        sp_eval = None
        if sp_nwk_str and len(kept_tips) >= 4:
            sp_root = parse_newick(sp_nwk_str)
            pruned = prune_tree(sp_root, kept_tips)
            if pruned and pruned.get("children"):
                sp_map, _ = get_splits_with_support(pruned, kept_tips)
                sp_splits = set(sp_map.keys())
                sp_p = os.path.join(tmpdir, "sp.nwk")
                sp_o = os.path.join(tmpdir, "sp_out.nwk")
                sp_j = os.path.join(tmpdir, "sp_out.json")
                with open(sp_p, "w") as sf:
                    sf.write(to_newick(pruned) + ";\n")
                r2 = subprocess.run(
                    [BUSHWHACKER_BIN] + flags + [fa_path, "-t", sp_p, "-o", sp_o, "-j", sp_j],
                    capture_output=True, text=True, env=env, timeout=300
                )
                if r2.returncode == 0 and os.path.exists(sp_j):
                    sp_eval = json.load(open(sp_j))

        bin_shared = len(bin_splits & sp_splits)
        bin_conf = sum(1 for s in (bin_splits - sp_splits) if any(not splits_compatible(s, x, kept_tips) for x in sp_splits))
        mdl_shared = len(mdl_splits & sp_splits)
        mdl_conf = sum(1 for s in (mdl_splits - sp_splits) if any(not splits_compatible(s, x, kept_tips) for x in sp_splits))

        # Check focal controversies
        hyps = get_hypothesis_splits(ds_name, kept_tips)
        hyp_outcomes = {}
        for h_name, h_split in hyps.items():
            if h_split in mdl_splits_map:
                hyp_outcomes[h_name] = {"state": "mdl_resolved", "nats": support_to_nats(mdl_splits_map[h_split]), "bin": h_split in bin_splits}
            elif all(splits_compatible(h_split, ms, kept_tips) for ms in mdl_splits):
                hyp_outcomes[h_name] = {"state": "mdl_polytomy_compatible", "nats": 0.0, "bin": h_split in bin_splits}
            else:
                hyp_outcomes[h_name] = {"state": "mdl_conflicted", "nats": 0.0, "bin": h_split in bin_splits}

        shunts = d.get("shunts", [])
        sp_shunts = sp_eval.get("shunts", []) if sp_eval else []

        return {
            "locus": os.path.basename(fa_path),
            "status": "ok",
            "n_taxa": d.get("n_taxa"),
            "seq_len": d.get("seq_len"),
            "runtime_sec": d.get("runtime_sec"),
            "star_mdl_nats": d.get("star_tree_MDL_nats"),
            "tree_mdl_nats": d.get("tree_MDL_nats"),
            "network_mdl_nats": d.get("network_MDL_nats"),
            "resolved_splits": d.get("resolved_internal_splits"),
            "max_binary_splits": d.get("max_binary_splits"),
            "sp_splits_total": len(sp_splits),
            "sp_resolved_splits": sp_eval.get("resolved_internal_splits") if sp_eval else None,
            "bin_sp_shared": bin_shared,
            "bin_sp_conflict": bin_conf,
            "mdl_sp_shared": mdl_shared,
            "mdl_sp_conflict": mdl_conf,
            "n_free_shunts": len(shunts),
            "n_free_interval": sum(1 for s in shunts if s.get("mode") == "INTERVAL"),
            "n_free_sparse": sum(1 for s in shunts if s.get("mode") == "SPARSE"),
            "n_sp_shunts": len(sp_shunts),
            "n_sp_interval": sum(1 for s in sp_shunts if s.get("mode") == "INTERVAL"),
            "n_sp_sparse": sum(1 for s in sp_shunts if s.get("mode") == "SPARSE"),
            "hypotheses": hyp_outcomes,
            "top_sp_shunts": [
                {
                    "mode": s.get("mode"),
                    "gain_nats": round(s.get("net_mdl_gain_nats", 0.0), 2),
                    "m": s.get("shared_mutations"),
                    "u": s.get("leaves_u", [])[:4],
                    "v": s.get("leaves_v", [])[:4],
                }
                for s in sp_shunts[:4]
            ],
        }
    except Exception as e:
        return {"locus": os.path.basename(fa_path), "status": "error", "error": str(e)}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_dataset(ds_name, num_workers):
    os.makedirs(os.path.join(BASE_DIR, "datasets"), exist_ok=True)
    aln_dir = os.path.join(ALN_ROOT, ds_name)
    files = sorted(
        os.path.join(aln_dir, f) for f in os.listdir(aln_dir)
        if not f.startswith(".") and os.path.isfile(os.path.join(aln_dir, f))
    )
    ref_candidates = [
        os.path.join(COAL_ROOT, ds_name, f"{ds_name}.Published.ASTRAL.tre"),
        os.path.join(COAL_ROOT, ds_name, f"{ds_name}.Best_observed.ASTRAL.tre"),
    ]
    sp_nwk_str = None
    for rc in ref_candidates:
        if os.path.exists(rc):
            sp_nwk_str = open(rc).read().strip()
            break

    is_prot = ds_name in PROTEIN_DATASETS
    tasks = [(fp, ds_name, is_prot, sp_nwk_str) for fp in files]
    print(f"[P4 {ds_name}] Processing {len(tasks)} loci ({'protein' if is_prot else 'nucleotide'}) with {num_workers} workers...", flush=True)
    t0 = time.perf_counter()

    ok_recs = []
    err_cnt = 0
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        for idx, rec in enumerate(ex.map(process_one_locus, tasks, chunksize=8), 1):
            if rec.get("status") == "ok":
                ok_recs.append(rec)
            else:
                err_cnt += 1
            if idx % 500 == 0 or idx == len(tasks):
                print(f"  [{ds_name}] {idx}/{len(tasks)} done ({time.perf_counter()-t0:.1f}s)", flush=True)

    # Aggregate dataset summary
    n_ok = max(1, len(ok_recs))
    tot_bin_splits = sum(r["max_binary_splits"] for r in ok_recs)
    tot_mdl_splits = sum(r["resolved_splits"] for r in ok_recs)
    tot_sp_splits = sum(r["sp_splits_total"] for r in ok_recs)
    tot_sp_resolved = sum(r["sp_resolved_splits"] or 0 for r in ok_recs)
    tot_bin_conf = sum(r["bin_sp_conflict"] for r in ok_recs)
    tot_mdl_conf = sum(r["mdl_sp_conflict"] for r in ok_recs)
    tot_bin_shared = sum(r["bin_sp_shared"] for r in ok_recs)
    tot_mdl_shared = sum(r["mdl_sp_shared"] for r in ok_recs)

    hyp_agg = defaultdict(lambda: {"informative_loci": 0, "mdl_resolved": 0, "mdl_polytomy": 0, "mdl_conflicted": 0, "forced_binary_count": 0, "cumulative_mdl_nats": 0.0})
    for r in ok_recs:
        for h_name, ho in r.get("hypotheses", {}).items():
            ha = hyp_agg[h_name]
            ha["informative_loci"] += 1
            if ho["state"] == "mdl_resolved":
                ha["mdl_resolved"] += 1
                ha["cumulative_mdl_nats"] = round(ha["cumulative_mdl_nats"] + ho["nats"], 2)
            elif ho["state"] == "mdl_polytomy_compatible":
                ha["mdl_polytomy"] += 1
            else:
                ha["mdl_conflicted"] += 1
            if ho["bin"]:
                ha["forced_binary_count"] += 1

    top_shunts = []
    for r in ok_recs:
        for s in r.get("top_sp_shunts", []):
            top_shunts.append({"locus": r["locus"], **s})
    top_shunts.sort(key=lambda x: x["gain_nats"], reverse=True)

    ds_summary = {
        "dataset": ds_name,
        "alphabet": "protein" if is_prot else "nucleotide",
        "total_files": len(files),
        "ok_loci": len(ok_recs),
        "error_loci": err_cnt,
        "wall_sec": round(time.perf_counter() - t0, 2),
        "sum_cpu_runtime_sec": round(sum(r["runtime_sec"] for r in ok_recs), 2),
        "mean_taxa": round(sum(r["n_taxa"] for r in ok_recs) / n_ok, 2),
        "mean_seq_len": round(sum(r["seq_len"] for r in ok_recs) / n_ok, 1),
        "total_max_binary_splits": tot_bin_splits,
        "total_mdl_resolved_splits": tot_mdl_splits,
        "polytomy_contracted_frac": round(1.0 - tot_mdl_splits / max(1, tot_bin_splits), 4),
        "sp_tree_total_splits": tot_sp_splits,
        "sp_tree_mdl_supported_splits": tot_sp_resolved,
        "sp_tree_hard_polytomy_frac": round(1.0 - tot_sp_resolved / max(1, tot_sp_splits), 4),
        "forced_binary_sp_shared": tot_bin_shared,
        "forced_binary_sp_conflict": tot_bin_conf,
        "mdl_sp_shared": tot_mdl_shared,
        "mdl_sp_conflict": tot_mdl_conf,
        "spurious_discordance_eliminated_frac": round(1.0 - tot_mdl_conf / max(1, tot_bin_conf), 4),
        "loci_with_sp_shunts": sum(1 for r in ok_recs if r["n_sp_shunts"] > 0),
        "total_sp_shunts": sum(r["n_sp_shunts"] for r in ok_recs),
        "total_sp_interval": sum(r["n_sp_interval"] for r in ok_recs),
        "total_sp_sparse": sum(r["n_sp_sparse"] for r in ok_recs),
        "total_free_shunts": sum(r["n_free_shunts"] for r in ok_recs),
        "focal_hypotheses": dict(hyp_agg),
        "top_sp_shunts": top_shunts[:25],
    }

    out_json = os.path.join(BASE_DIR, "datasets", f"{ds_name}.summary.json")
    with open(out_json, "w") as f:
        json.dump(ds_summary, f, indent=2)
    print(f"[P4 {ds_name}] Wrote summary to {out_json}", flush=True)


def aggregate_all():
    summaries = []
    for ds in DATASETS:
        p = os.path.join(BASE_DIR, "datasets", f"{ds}.summary.json")
        if os.path.exists(p):
            summaries.append(json.load(open(p)))
    out = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_datasets_completed": len(summaries),
        "total_loci": sum(s["ok_loci"] for s in summaries),
        "total_max_binary_splits": sum(s["total_max_binary_splits"] for s in summaries),
        "total_mdl_resolved_splits": sum(s["total_mdl_resolved_splits"] for s in summaries),
        "total_forced_binary_conflict": sum(s["forced_binary_sp_conflict"] for s in summaries),
        "total_mdl_conflict": sum(s["mdl_sp_conflict"] for s in summaries),
        "overall_spurious_discordance_eliminated_frac": round(
            1.0 - sum(s["mdl_sp_conflict"] for s in summaries) / max(1, sum(s["forced_binary_sp_conflict"] for s in summaries)),
            4,
        ),
        "datasets": summaries,
    }
    dest = os.path.join(BASE_DIR, "macroevo_polytomy_atlas.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[P4] Aggregated {len(summaries)} datasets ({out['total_loci']} loci) -> {dest}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "--aggregate":
        aggregate_all()
    else:
        arg = sys.argv[1]
        ds = DATASETS[int(arg)] if arg.isdigit() else arg
        workers = int(sys.argv[2]) if len(sys.argv) > 2 else 32
        run_dataset(ds, workers)
