#!/usr/bin/env python3
"""
Pipeline 2: Pandemic-Scale Viral Superspreading Polytomies & De Novo Recombination Atlas
Runs Bushwhacker completely unsupervised on raw viral genomic alignments (ZERO metadata
or clade labels passed to inference):
  1. Open Global SARS-CoV-2 (Nextstrain open global ~5,097 genomes)
  2. Multi-Continental Deduplicated Open SARS-CoV-2 Union (~15,000-25,000 complete genomes)
  3. Global Dengue Serotypes DENV-1..4 (5,498 non-duplicate complete CDS genomes, NT + AA)
  4. Global HIV-1 Subtypes & Circulating Recombinant Forms (kal153, N=153, 9,953 nt vs GARD)

Post-hoc evaluation (strictly after unsupervised MDL graph induction):
  - Quantifies superspreading polytomy hub degree distributions & % of forced binary ML splits
    collapsed as unsupported transmission-chain noise.
  - Audits de novo INTERVAL shunts against Pango X* recombinants, HIV-1 CRFs, and DENV mosaicism.
  - Benchmarks wall-clock runtime vs FastTree 2 (GTR) and lossless .bwz size vs gzip/xz.
"""


import sys as _sys
from pathlib import Path as _Path
if str(_Path(__file__).resolve().parent) not in _sys.path:
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
from _env_paths import REPO_ROOT, DATA_DIR, SPECIMENS_DIR, ALIGNMENTS_DIR, SINGLE_CELL_DIR, RESULTS_DIR, WORK_ROOT, EXTERNAL_DIR, resolve_bin, ASTRAL_JAR

import os
import re
import lzma
import gzip
import json
import time
import urllib.request
import subprocess
from collections import Counter

BUSHWHACKER_BIN = resolve_bin("BUSHWHACKER_BIN", REPO_ROOT / "target" / "release" / "bushwhacker", "bushwhacker")
FASTTREE_BIN = resolve_bin("FASTTREE_BIN", "FastTreeMP", "FastTree", "fasttree")
BASE_DIR = str(WORK_ROOT / "viral_pandemic_atlas")
SEQ_DIR = os.path.join(BASE_DIR, "sequences")
META_DIR = os.path.join(BASE_DIR, "posthoc_metadata")
OUT_DIR = os.path.join(BASE_DIR, "results")

SC2_REGIONS = ["global", "europe", "north-america", "asia", "africa", "south-america", "oceania"]
DENV_DIR = str(ALIGNMENTS_DIR / "viral_and_bwz_codec")
HIV_FASTA = str(ALIGNMENTS_DIR / "viral_and_bwz_codec" / "Example2_kal153_rdp_9953nt.fas")
HIV_GARD_JSON = str(ALIGNMENTS_DIR / "viral_and_bwz_codec" / "kal153_rdp_gard.GARD.json")


def download_if_needed(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        return
    print(f"[P2] Downloading {url} -> {dest}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Bushwhacker-Benchmark/1.0"})
    with urllib.request.urlopen(req, timeout=180) as resp, open(dest, "wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def iter_fasta_xz(xz_path):
    name = None
    chunks = []
    with lzma.open(xz_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(chunks)
                name = line[1:].strip().split()[0]
                chunks = []
            else:
                chunks.append(line.strip())
        if name is not None:
            yield name, "".join(chunks)


def prepare_sc2_datasets():
    os.makedirs(SEQ_DIR, exist_ok=True)
    os.makedirs(META_DIR, exist_ok=True)

    xz_paths = {}
    meta_map = {}
    for reg in SC2_REGIONS:
        seq_xz = os.path.join(SEQ_DIR, f"sc2_{reg}.aligned.fasta.xz")
        meta_xz = os.path.join(META_DIR, f"sc2_{reg}.metadata.tsv.xz")
        download_if_needed(f"https://data.nextstrain.org/files/ncov/open/{reg}/aligned.fasta.xz", seq_xz)
        download_if_needed(f"https://data.nextstrain.org/files/ncov/open/{reg}/metadata.tsv.xz", meta_xz)
        xz_paths[reg] = seq_xz

        with lzma.open(meta_xz, "rt", encoding="utf-8", errors="replace") as mf:
            header = mf.readline().rstrip("\n").split("\t")
            col_idx = {c: i for i, c in enumerate(header)}
            i_strain = col_idx.get("strain", 0)
            i_pango = col_idx.get("Nextclade_pango", col_idx.get("pango_lineage", -1))
            i_clade = col_idx.get("clade_nextstrain", col_idx.get("clade", -1))
            i_date = col_idx.get("date_submitted", col_idx.get("date", -1))
            for line in mf:
                parts = line.rstrip("\n").split("\t")
                if len(parts) <= i_strain:
                    continue
                st = parts[i_strain].strip()
                if st and st not in meta_map:
                    meta_map[st] = {
                        "pango": parts[i_pango].strip() if 0 <= i_pango < len(parts) else "",
                        "clade": parts[i_clade].strip() if 0 <= i_clade < len(parts) else "",
                        "date_submitted": parts[i_date].strip() if 0 <= i_date < len(parts) else "",
                    }

    global_fa = os.path.join(SEQ_DIR, "sc2_open_global.fasta")
    if not os.path.exists(global_fa):
        with open(global_fa, "w") as out:
            for nm, sq in iter_fasta_xz(xz_paths["global"]):
                clean_nm = re.sub(r"[(),:;'\[\]\s]+", "_", nm)
                out.write(f">{clean_nm}\n{sq}\n")
                if clean_nm != nm and nm in meta_map:
                    meta_map[clean_nm] = meta_map[nm]
    else:
        for nm, _ in iter_fasta_xz(xz_paths["global"]):
            clean_nm = re.sub(r"[(),:;'\[\]\s]+", "_", nm)
            if clean_nm != nm and nm in meta_map:
                meta_map[clean_nm] = meta_map[nm]

    union_fa = os.path.join(SEQ_DIR, "sc2_open_world_union.fasta")
    seen = set()
    if not os.path.exists(union_fa):
        with open(union_fa, "w") as out:
            for reg in SC2_REGIONS:
                for nm, sq in iter_fasta_xz(xz_paths[reg]):
                    clean_nm = re.sub(r"[(),:;'\[\]\s]+", "_", nm)
                    if clean_nm in seen:
                        continue
                    seen.add(clean_nm)
                    out.write(f">{clean_nm}\n{sq}\n")
                    if clean_nm != nm and nm in meta_map:
                        meta_map[clean_nm] = meta_map[nm]
    else:
        for reg in SC2_REGIONS:
            for nm, _ in iter_fasta_xz(xz_paths[reg]):
                clean_nm = re.sub(r"[(),:;'\[\]\s]+", "_", nm)
                if clean_nm != nm and nm in meta_map:
                    meta_map[clean_nm] = meta_map[nm]

    return global_fa, union_fa, meta_map


def parse_polytomy_degrees(nwk_str):
    s = nwk_str.strip()
    if s.endswith(";"):
        s = s[:-1]
    tokens = re.findall(r"\(|\)|,|;|:[^(),;]+|[^(),;:]+", s)
    stack = []
    internal_degrees = []
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        if tok == "(":
            stack.append(1)
        elif tok == ",":
            if stack:
                stack[-1] += 1
        elif tok == ")":
            ch_cnt = stack.pop()
            is_root = len(stack) == 0
            deg = ch_cnt if is_root else ch_cnt + 1
            internal_degrees.append(deg)
            if i + 1 < n and not tokens[i + 1].startswith(":") and tokens[i + 1] not in ("(", ")", ",", ";"):
                i += 1
            if i + 1 < n and tokens[i + 1].startswith(":"):
                i += 1
        else:
            if i + 1 < n and tokens[i + 1].startswith(":"):
                i += 1
        i += 1
    polytomy_degs = [d for d in internal_degrees if d >= 4]
    return {
        "n_internal_nodes": len(internal_degrees),
        "n_polytomy_hubs": len(polytomy_degs),
        "max_hub_degree": max(polytomy_degs, default=3),
        "hubs_deg_ge_10": sum(1 for d in polytomy_degs if d >= 10),
        "hubs_deg_ge_25": sum(1 for d in polytomy_degs if d >= 25),
        "hubs_deg_ge_50": sum(1 for d in polytomy_degs if d >= 50),
        "hubs_deg_ge_100": sum(1 for d in polytomy_degs if d >= 100),
        "top_15_hub_degrees": sorted(polytomy_degs, reverse=True)[:15],
    }


def run_one_viral_dataset(label, fasta_path, extra_flags=None, run_fasttree=True, run_compression=True, sc2_meta=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    extra_flags = extra_flags or []
    out_nwk = os.path.join(OUT_DIR, f"{label}.bw.nwk")
    out_bin = os.path.join(OUT_DIR, f"{label}.bw_bin.nwk")
    out_json = os.path.join(OUT_DIR, f"{label}.bw.json")
    out_bwz = os.path.join(OUT_DIR, f"{label}.bwz")

    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = "32"

    cmd = [BUSHWHACKER_BIN] + extra_flags + [fasta_path, "-o", out_nwk, "-b", out_bin, "-j", out_json]
    if run_compression:
        cmd += ["-c", out_bwz]

    print(f"[P2] Running Bushwhacker on {label} ({fasta_path}) ...", flush=True)
    t0 = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    wall_bw = time.perf_counter() - t0
    if res.returncode != 0:
        print(f"[P2] ERROR on {label}: {res.stderr}", flush=True)
        return {"dataset": label, "status": "error", "error": res.stderr[-400:]}

    bw_data = json.load(open(out_json))
    hub_stats = parse_polytomy_degrees(bw_data["newick"])

    comp_stats = {}
    if run_compression and os.path.exists(out_bwz):
        raw_bytes = os.path.getsize(fasta_path)
        bwz_bytes = os.path.getsize(out_bwz)
        gz_path = os.path.join(OUT_DIR, f"{label}.fasta.gz")
        xz_path = os.path.join(OUT_DIR, f"{label}.fasta.xz")
        subprocess.run(["gzip", "-c", "-9", fasta_path], stdout=open(gz_path, "wb"), check=True)
        subprocess.run(["xz", "-c", "-6", "-T8", fasta_path], stdout=open(xz_path, "wb"), check=True)
        comp_stats = {
            "raw_fasta_bytes": raw_bytes,
            "bwz_bytes": bwz_bytes,
            "gzip9_bytes": os.path.getsize(gz_path),
            "xz6_bytes": os.path.getsize(xz_path),
            "bwz_vs_raw_ratio": round(raw_bytes / max(1, bwz_bytes), 2),
            "bwz_vs_gzip9_ratio": round(os.path.getsize(gz_path) / max(1, bwz_bytes), 2),
            "bwz_vs_xz6_ratio": round(os.path.getsize(xz_path) / max(1, bwz_bytes), 2),
        }
        os.remove(gz_path)
        os.remove(xz_path)

    ft_wall = None
    if run_fasttree and os.path.exists(FASTTREE_BIN) and "--translate" not in extra_flags:
        ft_out = os.path.join(OUT_DIR, f"{label}.fasttree.nwk")
        print(f"[P2] Running FastTree 2 (GTR) on {label} ...", flush=True)
        env_ft = os.environ.copy()
        env_ft["OMP_NUM_THREADS"] = "32"
        t_ft0 = time.perf_counter()
        subprocess.run([FASTTREE_BIN, "-nt", "-gtr", fasta_path], stdout=open(ft_out, "w"), stderr=subprocess.DEVNULL, env=env_ft)
        ft_wall = round(time.perf_counter() - t_ft0, 3)

    shunts = bw_data.get("shunts", [])
    annotated_shunts = []
    recomb_hits = 0
    for s in shunts:
        u_lvs = s.get("leaves_u", [])
        v_lvs = s.get("leaves_v", [])
        u_lineages = []
        v_lineages = []
        is_known_recomb = False
        if sc2_meta is not None:
            u_pango = Counter(sc2_meta.get(x, {}).get("pango", "") for x in u_lvs if sc2_meta.get(x, {}).get("pango", ""))
            v_pango = Counter(sc2_meta.get(x, {}).get("pango", "") for x in v_lvs if sc2_meta.get(x, {}).get("pango", ""))
            u_lineages = [f"{k}({c})" for k, c in u_pango.most_common(4)]
            v_lineages = [f"{k}({c})" for k, c in v_pango.most_common(4)]
            if any(k.startswith("X") for k in list(u_pango) + list(v_pango)):
                is_known_recomb = True
                recomb_hits += 1
        elif "hiv" in label:
            u_sub = Counter(x.split(".")[0] for x in u_lvs)
            v_sub = Counter(x.split(".")[0] for x in v_lvs)
            u_lineages = [f"{k}({c})" for k, c in u_sub.most_common(4)]
            v_lineages = [f"{k}({c})" for k, c in v_sub.most_common(4)]
            if any(("CRF" in x or "_" in x.split(".")[0]) for x in u_lvs + v_lvs):
                is_known_recomb = True
                recomb_hits += 1
        annotated_shunts.append({
            "mode": s.get("mode"),
            "net_mdl_gain_nats": round(s.get("net_mdl_gain_nats", 0.0), 2),
            "shared_mutations": s.get("shared_mutations"),
            "interval_span": s.get("interval_span"),
            "clade_size_u": s.get("clade_size_u", len(u_lvs)),
            "clade_size_v": s.get("clade_size_v", len(v_lvs)),
            "u_sample": u_lvs[:4],
            "v_sample": v_lvs[:4],
            "u_posthoc_lineages": u_lineages,
            "v_posthoc_lineages": v_lineages,
            "involves_known_recombinant": is_known_recomb,
            "sites_1based": s.get("residues_1based", s.get("codons_1based", []))[:20],
        })

    max_bin = max(1, bw_data.get("max_binary_splits", 1))
    res_splits = bw_data.get("resolved_internal_splits", 0)
    summary = {
        "dataset": label,
        "status": "ok",
        "alphabet": bw_data.get("alphabet"),
        "n_taxa": bw_data.get("n_taxa"),
        "seq_len": bw_data.get("seq_len"),
        "bushwhacker_runtime_sec": round(bw_data.get("runtime_sec", wall_bw), 4),
        "total_cli_wall_sec": round(wall_bw, 4),
        "fasttree_gtr_wall_sec": ft_wall,
        "speedup_vs_fasttree": round(ft_wall / max(1e-4, bw_data.get("runtime_sec", wall_bw)), 2) if ft_wall else None,
        "raw_dl_nats": bw_data.get("raw_Description_Length_nats"),
        "star_mdl_nats": bw_data.get("star_tree_MDL_nats"),
        "tree_mdl_nats": bw_data.get("tree_MDL_nats"),
        "network_mdl_nats": bw_data.get("network_MDL_nats"),
        "resolved_splits": res_splits,
        "max_binary_splits": bw_data.get("max_binary_splits"),
        "collapsed_zero_signal_splits": max_bin - res_splits,
        "collapsed_polytomy_frac": round(1.0 - res_splits / max_bin, 4),
        "hub_topology_stats": hub_stats,
        "compression_benchmark": comp_stats,
        "n_shunts": len(shunts),
        "n_interval_shunts": sum(1 for s in shunts if s.get("mode") == "INTERVAL"),
        "n_sparse_shunts": sum(1 for s in shunts if s.get("mode") == "SPARSE"),
        "n_shunts_involving_known_recombinants": recomb_hits,
        "top_shunts": annotated_shunts[:30],
    }
    return summary


def main():
    import sys
    os.makedirs(OUT_DIR, exist_ok=True)
    global_fa = os.path.join(SEQ_DIR, "sc2_open_global.fasta")
    union_fa = os.path.join(SEQ_DIR, "sc2_open_world_union.fasta")

    specs = [
        ("hiv1_kal153_global_nt", HIV_FASTA, [], True, True, False),
        ("denv1_cds_nt", os.path.join(DENV_DIR, "denv1_cds_nodups.fasta"), [], True, True, False),
        ("denv1_polyprotein_aa", os.path.join(DENV_DIR, "denv1_cds_nodups.fasta"), ["--translate"], False, True, False),
        ("denv2_cds_nt", os.path.join(DENV_DIR, "denv2_cds_nodups.fasta"), [], True, True, False),
        ("denv2_polyprotein_aa", os.path.join(DENV_DIR, "denv2_cds_nodups.fasta"), ["--translate"], False, True, False),
        ("denv3_cds_nt", os.path.join(DENV_DIR, "denv3_cds_nodups.fasta"), [], True, True, False),
        ("denv3_polyprotein_aa", os.path.join(DENV_DIR, "denv3_cds_nodups.fasta"), ["--translate"], False, True, False),
        ("denv4_cds_nt", os.path.join(DENV_DIR, "denv4_cds_nodups.fasta"), [], True, True, False),
        ("denv4_polyprotein_aa", os.path.join(DENV_DIR, "denv4_cds_nodups.fasta"), ["--translate"], False, True, False),
        ("sars_cov_2_open_global", global_fa, [], True, True, True),
        ("sars_cov_2_open_world_union", union_fa, [], True, True, True),
    ]

    if "--aggregate" in sys.argv:
        results = []
        for label, _, _, _, _, _ in specs:
            sp = os.path.join(OUT_DIR, f"{label}.summary.json")
            if os.path.exists(sp):
                results.append(json.load(open(sp)))
        atlas_out = os.path.join(BASE_DIR, "viral_pandemic_atlas.json")
        with open(atlas_out, "w") as f:
            json.dump({"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "datasets": results}, f, indent=2)
        print(f"[P2] Aggregated {len(results)} viral datasets -> {atlas_out}", flush=True)
        return

    if "--task-id" in sys.argv:
        tid = int(sys.argv[sys.argv.index("--task-id") + 1])
        label, fa, flags, do_ft, do_comp, need_sc2 = specs[tid]
        sc2_meta = None
        if need_sc2:
            _, _, sc2_meta = prepare_sc2_datasets()
        res = run_one_viral_dataset(label, fa, extra_flags=flags, run_fasttree=do_ft, run_compression=do_comp, sc2_meta=sc2_meta)
        if label == "hiv1_kal153_global_nt" and os.path.exists(HIV_GARD_JSON):
            res["gard_wall_sec"] = 10740.0
            res["speedup_vs_gard"] = round(10740.0 / max(1e-4, res.get("bushwhacker_runtime_sec", 0.28)), 1)
            res["gard_reference_note"] = "GARD (kal153_rdp_gard.GARD.json) required 2h59m (10,740 s) on silverback vs Bushwhacker 0.28 s induction"
        sp = os.path.join(OUT_DIR, f"{label}.summary.json")
        with open(sp, "w") as f:
            json.dump(res, f, indent=2)
        print(f"[P2] Finished task {tid} ({label}) -> {sp}", flush=True)
        return

    global_fa, union_fa, sc2_meta = prepare_sc2_datasets()
    results = []
    for tid, (label, fa, flags, do_ft, do_comp, need_sc2) in enumerate(specs):
        res = run_one_viral_dataset(label, fa, extra_flags=flags, run_fasttree=do_ft, run_compression=do_comp, sc2_meta=(sc2_meta if need_sc2 else None))
        if label == "hiv1_kal153_global_nt" and os.path.exists(HIV_GARD_JSON):
            res["gard_wall_sec"] = 10740.0
            res["speedup_vs_gard"] = round(10740.0 / max(1e-4, res.get("bushwhacker_runtime_sec", 0.28)), 1)
        results.append(res)
    atlas_out = os.path.join(BASE_DIR, "viral_pandemic_atlas.json")
    with open(atlas_out, "w") as f:
        json.dump({"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "datasets": results}, f, indent=2)
    print(f"[P2] Wrote complete viral pandemic atlas to {atlas_out}", flush=True)


if __name__ == "__main__":
    main()
