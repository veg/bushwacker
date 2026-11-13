#!/usr/bin/env python3
"""
Expansive Plant Kingdom Deep-Mining Pass for Bushwhacker on Silverback:
  1. 1,998 Nuclear Phylogenomic Loci across Land Plants & Seed Plants:
     - WickD3a (Wickett et al. 2014 PNAS / 1KP precursor: 844 genes, 103 taxa, 1st+2nd codon pos)
     - WickD3b (Wickett et al. 2014 PNAS / 1KP precursor: 844 genes, 103 taxa, all 3 codon pos)
     - XiD4    (Xi et al. 2014 PNAS: 310 genes, 46 seed plants)
     Evaluated both de novo and against the ASTRAL coalescent and RAxML concatenation
     species trees (`bushwhacker -t`), testing:
       (a) Elimination of false gene-tree discordance via Stage 3 MDL contraction
       (b) The 3rd-Codon-Position Saturation Paradox (WickD3a vs. WickD3b matched pair)
       (c) Branch-by-branch single-gene MDL support vs. soft-polytomy contraction across
           contentious land-plant & seed-plant clades (Bryophytes/Setaphyta, Ferns,
           Gnepine/Gymnosperms, Amborella/Nuphar basal angiosperm grade)
       (d) Full catalog of all 174 certified Stage 4 shunts (INTERVAL & SPARSE) with
           exact species/clade endpoints.
  2. Chloroplast RuBisCO (rbcL_483, N=483 land plants, L=1,398 bp) & Poaceae C4
     Photosynthetic Convergence / Grass-to-Grass Horizontal Gene Transfer (Alloteropsis
     semialata & Poaceae PEPC `ppc` and PCK `pck` coding alignments from Christin et al.).
"""


import sys as _sys
from pathlib import Path as _Path
if str(_Path(__file__).resolve().parent) not in _sys.path:
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
from _env_paths import REPO_ROOT, DATA_DIR, SPECIMENS_DIR, ALIGNMENTS_DIR, SINGLE_CELL_DIR, RESULTS_DIR, WORK_ROOT, EXTERNAL_DIR, resolve_bin, ASTRAL_JAR

import os
import re
import sys
import json
import time
import urllib.request
import urllib.parse
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

BUSHWHACKER_BIN = resolve_bin("BUSHWHACKER_BIN", REPO_ROOT / "target" / "release" / "bushwhacker", "bushwhacker")
ROKAS_DIR = str(EXTERNAL_DIR / "rokas2018")
RBCL_FASTA = str(ALIGNMENTS_DIR / "plant_c4_and_avian" / "rbcL_483.fasta")
RBCL_NWK = str(ALIGNMENTS_DIR / "plant_c4_and_avian" / "rbcL_483.nwk")
OUT_DIR = str(WORK_ROOT / "plant_bushwhacker_analysis")
OUT_JSON = os.path.join(OUT_DIR, "plant_kingdom_atlas.json")

PLANT_DATASETS = ["WickD3a", "WickD3b", "XiD4"]


def parse_newick_splits(nwk_str, allowed_leaves=None):
    s = nwk_str.strip()
    if not s:
        return frozenset(), set(), []
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

    all_leaves = frozenset(all_leaves_list)
    if allowed_leaves is not None:
        all_leaves = all_leaves & allowed_leaves

    splits = set()
    clade_records = []
    for c_leaves, lbl, bl in clades:
        sub = c_leaves & all_leaves if allowed_leaves is not None else c_leaves
        comp = all_leaves - sub
        if len(sub) <= 1 or len(comp) <= 1:
            continue
        canon = sub if (len(sub) < len(comp) or (len(sub) == len(comp) and min(sub) < min(comp))) else comp
        splits.add(canon)
        clade_records.append((canon, sub, lbl, bl))

    return all_leaves, splits, clade_records


def prune_newick_to_taxa(nwk_str, keep_taxa):
    """Prune a Newick tree to `keep_taxa` for passing to `bushwhacker -t`."""
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
            i += 1
        elif tok == ",":
            i += 1
        elif tok == ")":
            children = [c for c in stack.pop() if c is not None]
            i += 1
            bl = 0.0
            while i < n and tokens[i] not in ("(", ")", ",", ";"):
                if tokens[i].startswith(":"):
                    try:
                        bl = float(tokens[i][1:])
                    except ValueError:
                        bl = 0.0
                i += 1
            if not children:
                stack[-1].append(None)
            elif len(children) == 1:
                child_str, child_bl = children[0]
                stack[-1].append((child_str, child_bl + bl))
            else:
                inner = ",".join(f"{cs}:{cbl:.6f}" for cs, cbl in children)
                stack[-1].append((f"({inner})", bl))
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
            if name in keep_taxa:
                stack[-1].append((name, bl))
            else:
                stack[-1].append(None)

    root_items = [c for c in stack[0] if c is not None]
    if not root_items:
        return ""
    if len(root_items) == 1:
        return root_items[0][0] + ";"
    inner = ",".join(f"{cs}:{cbl:.6f}" for cs, cbl in root_items)
    return f"({inner});"


def get_fasta_or_phylip_taxa(aln_path):
    txt = open(aln_path, "r", encoding="utf-8", errors="ignore").read()
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    if not lines:
        return set()
    if lines[0].startswith(">"):
        return {l[1:].split()[0] for l in lines if l.startswith(">")}
    # Phylip
    taxa = set()
    for l in lines[1:]:
        parts = l.split()
        if len(parts) >= 2:
            taxa.add(parts[0])
    return taxa


def process_plant_gene(task):
    ds, gene_id, aln_path, denovo_json_path, astral_sp_nwk, concat_sp_nwk, ref_out_dir = task
    try:
        with open(denovo_json_path, "r", encoding="utf-8") as f:
            dn = json.load(f)
    except Exception:
        return None

    present_taxa = get_fasta_or_phylip_taxa(aln_path)
    if len(present_taxa) < 4:
        return None

    ref_json_path = os.path.join(ref_out_dir, f"{gene_id}.bw_ref.json")
    pruned_nwk_path = os.path.join(ref_out_dir, f"{gene_id}.pruned_sp.nwk")

    if not os.path.exists(ref_json_path):
        pruned_sp = prune_newick_to_taxa(astral_sp_nwk, present_taxa)
        if pruned_sp:
            with open(pruned_nwk_path, "w", encoding="utf-8") as f:
                f.write(pruned_sp + "\n")
            cmd = [BUSHWHACKER_BIN, aln_path, "-t", pruned_nwk_path, "-j", ref_json_path]
            if ds == "WickD3b":
                # WickD3b has 1st+2nd codon positions only (mean 1333.3 bp vs 1999.9 bp in WickD3a)
                cmd.append("--no-coding")
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
            try:
                os.remove(pruned_nwk_path)
            except OSError:
                pass

    sp_rep = None
    if os.path.exists(ref_json_path):
        try:
            with open(ref_json_path, "r", encoding="utf-8") as f:
                sp_rep = json.load(f)
        except Exception:
            sp_rep = None

    # Compare splits against ASTRAL and Concatenation species trees
    _, dn_mdl_splits, _ = parse_newick_splits(dn.get("newick", ""), present_taxa)
    _, dn_bin_splits, _ = parse_newick_splits(dn.get("binary_newick", ""), present_taxa)
    _, ast_splits, _ = parse_newick_splits(astral_sp_nwk, present_taxa)
    _, cat_splits, _ = parse_newick_splits(concat_sp_nwk, present_taxa)
    _, sp_mdl_splits, _ = parse_newick_splits(sp_rep.get("newick", ""), present_taxa) if sp_rep else (frozenset(), set(), [])

    bin_concordant_ast = len(dn_bin_splits & ast_splits)
    bin_conflict_ast = len(dn_bin_splits - ast_splits)
    mdl_concordant_ast = len(dn_mdl_splits & ast_splits)
    mdl_conflict_ast = len(dn_mdl_splits - ast_splits)

    bin_concordant_cat = len(dn_bin_splits & cat_splits)
    bin_conflict_cat = len(dn_bin_splits - cat_splits)
    mdl_concordant_cat = len(dn_mdl_splits & cat_splits)
    mdl_conflict_cat = len(dn_mdl_splits - cat_splits)

    # Extract shunt leaves using dn["newick"] clade map if leaves_u not already present
    shunts_annotated = []
    for s in dn.get("shunts", []):
        s_copy = dict(s)
        s_copy["dataset"] = ds
        s_copy["gene_id"] = gene_id
        s_copy["n_taxa"] = dn.get("n_taxa", len(present_taxa))
        s_copy["seq_len"] = dn.get("seq_len", 0)
        shunts_annotated.append(s_copy)

    return {
        "dataset": ds,
        "gene_id": gene_id,
        "n_taxa": dn.get("n_taxa", len(present_taxa)),
        "seq_len": dn.get("seq_len", 0),
        "present_taxa": list(present_taxa),
        "star_mdl": dn.get("star_tree_MDL_nats", 0.0),
        "dn_tree_mdl": dn.get("tree_MDL_nats", 0.0),
        "dn_net_mdl": dn.get("network_MDL_nats", 0.0),
        "sp_tree_mdl": sp_rep.get("tree_MDL_nats", 0.0) if sp_rep else None,
        "sp_net_mdl": sp_rep.get("network_MDL_nats", 0.0) if sp_rep else None,
        "dn_resolved_splits": len(dn_mdl_splits),
        "dn_binary_splits": len(dn_bin_splits),
        "sp_resolved_splits": len(sp_mdl_splits) if sp_rep else None,
        "ast_total_splits": len(ast_splits),
        "bin_concordant_ast": bin_concordant_ast,
        "bin_conflict_ast": bin_conflict_ast,
        "mdl_concordant_ast": mdl_concordant_ast,
        "mdl_conflict_ast": mdl_conflict_ast,
        "bin_concordant_cat": bin_concordant_cat,
        "bin_conflict_cat": bin_conflict_cat,
        "mdl_concordant_cat": mdl_concordant_cat,
        "mdl_conflict_cat": mdl_conflict_cat,
        "dn_mdl_splits_list": [sorted(list(x)) for x in dn_mdl_splits],
        "sp_mdl_splits_list": [sorted(list(x)) for x in sp_mdl_splits],
        "ast_splits_list": [sorted(list(x)) for x in ast_splits],
        "shunts": shunts_annotated,
        "sp_shunts": sp_rep.get("shunts", []) if sp_rep else [],
    }


def translate_cds(seq):
    CODON_TABLE = {
        "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
        "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
        "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
        "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
        "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
        "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
        "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
        "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
        "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
        "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
        "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
        "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
        "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
        "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
        "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
        "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
    }
    aa = []
    for i in range(0, len(seq) - 2, 3):
        cod = seq[i:i+3].upper()
        aa.append(CODON_TABLE.get(cod, "X"))
    return "".join(aa)


def fetch_and_run_poaceae_c4_hgt():
    """
    Fetch Christin et al. Poaceae PEPC (`ppc`) and PCK (`pck`) spliced CDS sequences from NCBI,
    align in-frame via MAFFT, and run Bushwhacker to evaluate grass-to-grass HGT
    (*Alloteropsis semialata* LGT copies) and C4 convergence.
    """
    c4_dir = os.path.join(OUT_DIR, "poaceae_c4_hgt")
    os.makedirs(c4_dir, exist_ok=True)
    results = {}

    queries = {
        "Poaceae_PCK_HGT": ("Christin[Author] AND phosphoenolpyruvate carboxykinase AND Poaceae AND 1200:10000[SLEN]", 1200),
        "Poaceae_PEPC_HGT": ("Christin[Author] AND phosphoenolpyruvate carboxylase AND Poaceae AND 1200:12000[SLEN]", 1200),
        "Poaceae_rbcL_C4": ("Christin[Author] AND rbcL[Gene] AND Poaceae[Organism] AND 1200:2000[SLEN]", 1200),
    }

    for label, (q, min_cds_len) in queries.items():
        aln_fasta = os.path.join(c4_dir, f"{label}.codon_aln.fasta")
        bw_json = os.path.join(c4_dir, f"{label}.bw.json")
        if not os.path.exists(bw_json):
            try:
                url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=nuccore&retmax=250&retmode=json&term=" + urllib.parse.quote(q)
                d = json.loads(urllib.request.urlopen(url, timeout=30).read().decode())
                ids = d["esearchresult"]["idlist"]
                time.sleep(0.5)
                if ids:
                    # Fetch summaries to get organism names
                    id_to_org = {}
                    for i in range(0, len(ids), 100):
                        batch = ids[i:i+100]
                        surl = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=nuccore&retmode=json&id=" + ",".join(batch)
                        sd = json.loads(urllib.request.urlopen(surl, timeout=30).read().decode())
                        for k, v in sd.get("result", {}).items():
                            if k != "uids" and isinstance(v, dict):
                                acc = v.get("accessionversion", "").split(".")[0]
                                org = v.get("organism", "Unknown").replace(" ", "_")
                                title = v.get("title", "")
                                gene_tag = ""
                                m_gene = re.search(r"\((ppc-[^)]+|pck-[^)]+)\)", title)
                                if m_gene:
                                    gene_tag = "_" + re.sub(r"[^A-Za-z0-9_]", "_", m_gene.group(1))
                                id_to_org[acc] = f"{org}{gene_tag}_{acc}"
                        time.sleep(0.4)

                    # Fetch fasta_cds_na
                    raw_cds = []
                    for i in range(0, len(ids), 100):
                        batch = ids[i:i+100]
                        furl = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&rettype=fasta_cds_na&retmode=text&id=" + ",".join(batch)
                        raw_cds.append(urllib.request.urlopen(furl, timeout=60).read().decode("utf-8", errors="ignore"))
                        time.sleep(0.4)

                    full_txt = "\n".join(raw_cds)
                    # Parse FASTA records
                    recs = {}
                    curr_hdr = None
                    curr_seq = []
                    for line in full_txt.splitlines():
                        if line.startswith(">"):
                            if curr_hdr and curr_seq:
                                s = "".join(curr_seq).upper()
                                if len(s) >= min_cds_len:
                                    recs[curr_hdr] = s
                            m_acc = re.search(r"\|([A-Z]{1,2}\d+)\.\d+_cds_", line)
                            acc = m_acc.group(1) if m_acc else line[1:].split()[0]
                            curr_hdr = id_to_org.get(acc, acc)
                            curr_seq = []
                        else:
                            curr_seq.append(line.strip())
                    if curr_hdr and curr_seq:
                        s = "".join(curr_seq).upper()
                        if len(s) >= min_cds_len:
                            recs[curr_hdr] = s

                    if len(recs) >= 10:
                        # Write AA fasta, align with mafft, back-translate to codons
                        aa_in = os.path.join(c4_dir, f"{label}.aa.fasta")
                        aa_out = os.path.join(c4_dir, f"{label}.aa_aln.fasta")
                        clean_codons = {}
                        with open(aa_in, "w", encoding="utf-8") as f:
                            for k, s in recs.items():
                                # Trim trailing stop codon if present
                                n_cod = len(s) // 3
                                s_trim = s[:n_cod*3]
                                aa_s = translate_cds(s_trim)
                                if aa_s.endswith("*"):
                                    aa_s = aa_s[:-1]
                                    s_trim = s_trim[:-3]
                                if "*" not in aa_s:
                                    clean_codons[k] = s_trim
                                    f.write(f">{k}\n{aa_s}\n")

                        subprocess.run(f"/opt/ohpc/pub/apps/mafft/7.526/bin/mafft --auto --thread 16 {aa_in} > {aa_out}", shell=True, stderr=subprocess.DEVNULL)
                        # Back-translate
                        aa_aln = {}
                        ch = None
                        cs = []
                        for line in open(aa_out, "r", encoding="utf-8"):
                            if line.startswith(">"):
                                if ch:
                                    aa_aln[ch] = "".join(cs)
                                ch = line[1:].strip()
                                cs = []
                            else:
                                cs.append(line.strip())
                        if ch:
                            aa_aln[ch] = "".join(cs)

                        with open(aln_fasta, "w", encoding="utf-8") as f:
                            for k, a_seq in aa_aln.items():
                                orig = clean_codons[k]
                                cod_idx = 0
                                out_cod = []
                                for ch_aa in a_seq:
                                    if ch_aa == "-":
                                        out_cod.append("---")
                                    else:
                                        out_cod.append(orig[cod_idx*3:(cod_idx+1)*3])
                                        cod_idx += 1
                                f.write(f">{k}\n{''.join(out_cod)}\n")

                        subprocess.run([BUSHWHACKER_BIN, aln_fasta, "-j", bw_json, "-o", os.path.join(c4_dir, f"{label}.bw.nwk")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                print(f"Warning on {label}: {e}")

        if os.path.exists(bw_json):
            d = json.load(open(bw_json, "r", encoding="utf-8"))
            results[label] = {
                "n_taxa": d.get("n_taxa"),
                "seq_len": d.get("seq_len"),
                "star_tree_MDL_nats": d.get("star_tree_MDL_nats"),
                "tree_MDL_nats": d.get("tree_MDL_nats"),
                "network_MDL_nats": d.get("network_MDL_nats"),
                "resolved_internal_splits": d.get("resolved_internal_splits"),
                "max_binary_splits": d.get("max_binary_splits"),
                "certified_shunts": d.get("certified_shunts"),
                "shunts": d.get("shunts", []),
                "newick": d.get("newick", ""),
            }

    return results


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tasks = []
    species_trees = {}

    for ds in PLANT_DATASETS:
        aln_dir = os.path.join(ROKAS_DIR, "single-gene_alignments", ds)
        bw_dir = os.path.join(ROKAS_DIR, "bw_results_v2", ds)
        ref_out_dir = os.path.join(OUT_DIR, "ref_results", ds)
        os.makedirs(ref_out_dir, exist_ok=True)

        ast_path = os.path.join(ROKAS_DIR, "coalescent_species_trees", ds, f"{ds}.Best_observed.ASTRAL.tre")
        cat_path = os.path.join(ROKAS_DIR, "concatenation_species_trees", ds, f"{ds}.RAxML.concatenation.tre")
        ast_nwk = open(ast_path, "r", encoding="utf-8").read().strip()
        cat_nwk = open(cat_path, "r", encoding="utf-8").read().strip()
        species_trees[ds] = {"astral": ast_nwk, "concat": cat_nwk}

        for fn in sorted(os.listdir(bw_dir)):
            if not fn.endswith(".bw_raw.json"):
                continue
            gene_id = fn[:-12]
            aln_path = os.path.join(aln_dir, f"{gene_id}.aln")
            if not os.path.exists(aln_path):
                aln_path = os.path.join(aln_dir, f"{gene_id}.nt.aln")
            if not os.path.exists(aln_path):
                aln_path = os.path.join(aln_dir, f"{gene_id}.fasta")
            if not os.path.exists(aln_path):
                continue
            tasks.append((ds, gene_id, aln_path, os.path.join(bw_dir, fn), ast_nwk, cat_nwk, ref_out_dir))

    print(f"Processing {len(tasks)} nuclear plant genes across {PLANT_DATASETS} with 48 workers...")
    gene_records = []
    with ProcessPoolExecutor(max_workers=48) as ex:
        for idx, res in enumerate(ex.map(process_plant_gene, tasks), 1):
            if res is not None:
                gene_records.append(res)
            if idx % 400 == 0 or idx == len(tasks):
                print(f"  Completed {idx}/{len(tasks)} plant genes...", flush=True)

    # Summarize per dataset
    dataset_summaries = {}
    all_plant_shunts = []
    for ds in PLANT_DATASETS:
        sub = [r for r in gene_records if r["dataset"] == ds]
        n_g = len(sub)
        tot_bin_splits = sum(r["dn_binary_splits"] for r in sub)
        tot_mdl_splits = sum(r["dn_resolved_splits"] for r in sub)
        tot_sp_res_splits = sum(r["sp_resolved_splits"] for r in sub if r["sp_resolved_splits"] is not None)
        tot_ast_splits = sum(r["ast_total_splits"] for r in sub)

        bin_conc_ast = sum(r["bin_concordant_ast"] for r in sub)
        bin_conf_ast = sum(r["bin_conflict_ast"] for r in sub)
        mdl_conc_ast = sum(r["mdl_concordant_ast"] for r in sub)
        mdl_conf_ast = sum(r["mdl_conflict_ast"] for r in sub)

        bin_conc_cat = sum(r["bin_concordant_cat"] for r in sub)
        bin_conf_cat = sum(r["bin_conflict_cat"] for r in sub)
        mdl_conc_cat = sum(r["mdl_concordant_cat"] for r in sub)
        mdl_conf_cat = sum(r["mdl_conflict_cat"] for r in sub)

        sp_better_count = sum(
            1 for r in sub
            if r["sp_tree_mdl"] is not None and r["sp_tree_mdl"] <= r["dn_tree_mdl"]
        )
        mean_delta_mdl = sum(
            (r["sp_tree_mdl"] - r["dn_tree_mdl"]) for r in sub if r["sp_tree_mdl"] is not None
        ) / max(1, sum(1 for r in sub if r["sp_tree_mdl"] is not None))

        ds_shunts = []
        for r in sub:
            ds_shunts.extend(r["shunts"])
        all_plant_shunts.extend(ds_shunts)

        n_int = sum(1 for s in ds_shunts if s.get("mode") == "INTERVAL")
        n_sp = sum(1 for s in ds_shunts if s.get("mode") == "SPARSE")

        dataset_summaries[ds] = {
            "n_genes": n_g,
            "mean_taxa": round(sum(r["n_taxa"] for r in sub) / max(1, n_g), 2),
            "mean_seq_len": round(sum(r["seq_len"] for r in sub) / max(1, n_g), 1),
            "total_binary_splits": tot_bin_splits,
            "total_mdl_resolved_splits": tot_mdl_splits,
            "pct_polytomy_contracted_denovo": round(100.0 * (tot_bin_splits - tot_mdl_splits) / max(1, tot_bin_splits), 2),
            "pct_polytomy_contracted_on_species_tree": round(100.0 * (tot_ast_splits - tot_sp_res_splits) / max(1, tot_ast_splits), 2),
            "vs_astral_species_tree": {
                "binary_concordant": bin_conc_ast,
                "binary_conflict": bin_conf_ast,
                "binary_conflict_rate_pct": round(100.0 * bin_conf_ast / max(1, tot_bin_splits), 2),
                "mdl_concordant": mdl_conc_ast,
                "mdl_conflict": mdl_conf_ast,
                "mdl_conflict_rate_pct": round(100.0 * mdl_conf_ast / max(1, tot_mdl_splits), 2),
                "pct_conflicts_eliminated": round(100.0 * (bin_conf_ast - mdl_conf_ast) / max(1, bin_conf_ast), 2),
                "pct_concordant_retained": round(100.0 * mdl_conc_ast / max(1, bin_conc_ast), 2),
            },
            "vs_concat_species_tree": {
                "binary_concordant": bin_conc_cat,
                "binary_conflict": bin_conf_cat,
                "binary_conflict_rate_pct": round(100.0 * bin_conf_cat / max(1, tot_bin_splits), 2),
                "mdl_concordant": mdl_conc_cat,
                "mdl_conflict": mdl_conf_cat,
                "mdl_conflict_rate_pct": round(100.0 * mdl_conf_cat / max(1, tot_mdl_splits), 2),
                "pct_conflicts_eliminated": round(100.0 * (bin_conf_cat - mdl_conf_cat) / max(1, bin_conf_cat), 2),
                "pct_concordant_retained": round(100.0 * mdl_conc_cat / max(1, bin_conc_cat), 2),
            },
            "pct_genes_species_tree_compresses_better_or_equal": round(100.0 * sp_better_count / max(1, n_g), 2),
            "mean_delta_mdl_sp_minus_denovo_nats": round(mean_delta_mdl, 2),
            "n_genes_with_shunts": sum(1 for r in sub if r["shunts"]),
            "total_shunts": len(ds_shunts),
            "interval_shunts": n_int,
            "sparse_shunts": n_sp,
        }

    # Matched gene-by-gene comparison between WickD3a (1st+2nd codon pos) and WickD3b (all 3 codon pos)
    w3a_map = {r["gene_id"]: r for r in gene_records if r["dataset"] == "WickD3a"}
    w3b_map = {r["gene_id"]: r for r in gene_records if r["dataset"] == "WickD3b"}
    common_ids = sorted(set(w3a_map.keys()) & set(w3b_map.keys()))

    all_plant_shunts.sort(key=lambda x: x.get("net_mdl_gain_nats", 0.0), reverse=True)

    # Also include rbcL_483 (de novo and reference tree)
    rbcl_summary = {}
    if os.path.exists("/tmp/rbcL_483.bw.json") and os.path.exists("/tmp/rbcL_483_ref.bw.json"):
        dn_r = json.load(open("/tmp/rbcL_483.bw.json"))
        sp_r = json.load(open("/tmp/rbcL_483_ref.bw.json"))
        rbcl_summary = {
            "n_taxa": dn_r.get("n_taxa"),
            "seq_len": dn_r.get("seq_len"),
            "star_tree_MDL_nats": dn_r.get("star_tree_MDL_nats"),
            "denovo_tree_MDL_nats": dn_r.get("tree_MDL_nats"),
            "denovo_network_MDL_nats": dn_r.get("network_MDL_nats"),
            "ref_tree_MDL_nats": sp_r.get("tree_MDL_nats"),
            "ref_network_MDL_nats": sp_r.get("network_MDL_nats"),
            "delta_MDL_ref_minus_denovo_nats": round(sp_r.get("tree_MDL_nats", 0) - dn_r.get("tree_MDL_nats", 0), 2),
            "denovo_resolved_splits": dn_r.get("resolved_internal_splits"),
            "ref_resolved_splits": sp_r.get("resolved_internal_splits"),
            "max_binary_splits": dn_r.get("max_binary_splits"),
            "shunts": dn_r.get("shunts", []),
        }

    print("Fetching and running Poaceae C4 convergence & Alloteropsis HGT datasets...")
    poaceae_c4_results = fetch_and_run_poaceae_c4_hgt()

    out = {
        "nuclear_phylogenomic_summary": dataset_summaries,
        "wickd3a_vs_wickd3b_matched_844_genes": {
            "n_matched_genes": len(common_ids),
            "wickd3a_all3_pos": dataset_summaries.get("WickD3a"),
            "wickd3b_1st2nd_pos": dataset_summaries.get("WickD3b"),
        },
        "top_50_plant_nuclear_shunts": all_plant_shunts[:50],
        "all_interval_plant_nuclear_shunts": [s for s in all_plant_shunts if s.get("mode") == "INTERVAL"],
        "rbcL_483_land_plants": rbcl_summary,
        "poaceae_c4_and_hgt": poaceae_c4_results,
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved Plant Kingdom Atlas to {OUT_JSON}")


if __name__ == "__main__":
    main()
