#!/usr/bin/env python3
"""
Benchmark #1: The Complete 882-Species Mammalian Tree of Life (TOGA2 Supermatrix)
=================================================================================
Assembles a leak-free, unsupervised 253-locus (n_taxa >= 875/882) TOGA2 mammalian
coding supermatrix in both Nucleotide (NT) and translated 20-State Protein (AA)
formats, and benchmarks Bushwhacker head-to-head against SOTA ML phylogenetics:

  1. Bushwhacker Coding-Aware NT (-T 32 --no-shunts): MDL & Binary trees
  2. Bushwhacker 20-State LG08+CAT8 AA (-T 32 --protein --no-shunts): MDL & Binary trees
  3. FastTreeMP (-gtr -nt, OMP_NUM_THREADS=32)
  4. IQ-TREE 2 (-st DNA -m GTR+G4 --fast -T 32)
  5. FastTreeMP (-lg, OMP_NUM_THREADS=32)
  6. IQ-TREE 2 (-st AA -m LG+G4 --fast -T 32)

Evaluated against:
  - Curated Senckenberg Mammalian Reference Phylogeny (data/alignments/supermatrices/toga2_speciesTree.nh)
  - 4-Rank NCBI Mammalian Taxonomy (Superorder, Order, Family, Genus; 100% 882/882 coverage)
  - Pairwise unrooted Robinson-Foulds (nRF) agreement across all methods
  - 32-core wall-clock runtime on identical silverback hardware
"""


import sys as _sys
from pathlib import Path as _Path
if str(_Path(__file__).resolve().parent) not in _sys.path:
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
from _env_paths import REPO_ROOT, DATA_DIR, SPECIMENS_DIR, ALIGNMENTS_DIR, SINGLE_CELL_DIR, RESULTS_DIR, WORK_ROOT, EXTERNAL_DIR, resolve_bin, ASTRAL_JAR

import os
import re
import json
import time
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict

BUSHWHACKER_BIN = resolve_bin("BUSHWHACKER_BIN", REPO_ROOT / "target" / "release" / "bushwhacker", "bushwhacker")
FASTTREE_BIN = resolve_bin("FASTTREE_BIN", "FastTreeMP", "FastTree", "fasttree")
IQTREE_BIN = resolve_bin("IQTREE_BIN", "iqtree2", "iqtree")

SPECIES_TREE_PATH = Path("data/alignments/supermatrices/toga2_speciesTree.nh")
ASSEMBLIES_TSV = ALIGNMENTS_DIR / "supermatrices" / "toga2_assemblies_and_species.tsv"
MASTER_JSONL = WORK_ROOT / "toga2_nt" / "toga2_bushwhacker_master_19953.jsonl"
INDIV_ALIS_DIR = EXTERNAL_DIR / "toga2_hg38_v1" / "individualAlis"

WORK_DIR = ALIGNMENTS_DIR / "supermatrices"
METHODS_DIR = WORK_DIR / "methods"
RUNS_DIR = WORK_DIR / "runs"
NT_FASTA = WORK_DIR / "toga2_mammal882_nt.fna"
AA_FASTA = WORK_DIR / "toga2_mammal882_aa.faa"
MANIFEST_JSON = WORK_DIR / "dataset_manifest.json"
OUT_JSON = RESULTS_DIR / "mammalian_toga2" / "toga2_mammal882_sota_bench.json"

SUPERORDERS = {"Laurasiatheria", "Euarchontoglires", "Afrotheria", "Xenarthra"}
ORDERS = {
    "Primates",
    "Rodentia",
    "Lagomorpha",
    "Scandentia",
    "Dermoptera",
    "Chiroptera",
    "Artiodactyla",  # Includes Whippomorpha/Cetacea in NCBI Taxonomy (Cetartiodactyla)
    "Perissodactyla",
    "Carnivora",
    "Pholidota",
    "Eulipotyphla",
    "Proboscidea",
    "Sirenia",
    "Hyracoidea",
    "Tubulidentata",
    "Macroscelidea",
    "Afrosoricida",
    "Cingulata",
    "Pilosa",
}

STANDARD_GENETIC_CODE = {
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
VALID_NT_CHARS = set("ACGTN-")


class TreeNode:
    __slots__ = ("name", "length", "children")

    def __init__(self, name="", length=None, children=None):
        self.name = name
        self.length = length
        self.children = children if children is not None else []


def parse_newick_tree(nwk_str: str) -> TreeNode:
    s = nwk_str.strip()
    if s.endswith(";"):
        s = s[:-1]
    tokens = re.findall(r"'[^']*'|\[[^\]]*\]|\(|\)|,|:[^(),;\[\]]+|[^(),;:\[\]]+", s)
    root = TreeNode()
    stack = [root]
    curr = root
    for tok in tokens:
        if tok.startswith("["):
            continue
        if tok == "(":
            child = TreeNode()
            curr.children.append(child)
            stack.append(curr)
            curr = child
        elif tok == ",":
            parent = stack[-1]
            child = TreeNode()
            parent.children.append(child)
            curr = child
        elif tok == ")":
            curr = stack.pop()
        elif tok.startswith(":"):
            try:
                curr.length = float(tok[1:])
            except ValueError:
                curr.length = 0.0
        else:
            curr.name = tok.strip("'")
    while len(root.children) == 1 and not root.name:
        root = root.children[0]
    return root


def get_leaf_names_and_hub_degrees(root: TreeNode) -> tuple:
    leaves = []
    hub_degrees = []
    stack = [(root, True)]
    while stack:
        nd, is_rt = stack.pop()
        if not nd.children:
            if nd.name:
                leaves.append(nd.name)
        else:
            deg = len(nd.children) + (0 if is_rt else 1)
            if deg > 3:
                hub_degrees.append(deg)
            for c in reversed(nd.children):
                stack.append((c, False))
    return leaves, hub_degrees


def extract_canonical_splits(root: TreeNode, taxon_to_bit: dict, full_mask: int) -> set:
    n_shared = bin(full_mask).count("1")
    if n_shared < 4:
        return set()
    splits = set()
    stack = [(root, False)]
    masks = {}
    while stack:
        nd, visited = stack.pop()
        if not nd.children:
            masks[id(nd)] = taxon_to_bit.get(nd.name, 0)
        elif not visited:
            stack.append((nd, True))
            for c in nd.children:
                stack.append((c, False))
        else:
            m = 0
            for c in nd.children:
                m |= masks.pop(id(c), 0)
            masks[id(nd)] = m
            bc = bin(m).count("1")
            if 2 <= bc <= n_shared - 2:
                comp = full_mask ^ m
                splits.add(m if m < comp else comp)
    return splits


def compare_unrooted_trees(root_pred: TreeNode, root_ref: TreeNode) -> dict:
    lp, _ = get_leaf_names_and_hub_degrees(root_pred)
    lr, _ = get_leaf_names_and_hub_degrees(root_ref)
    shared = sorted(set(lp) & set(lr))
    n = len(shared)
    t2b = {tx: (1 << i) for i, tx in enumerate(shared)}
    full = (1 << n) - 1
    sp_p = extract_canonical_splits(root_pred, t2b, full)
    sp_r = extract_canonical_splits(root_ref, t2b, full)
    tp = len(sp_p & sp_r)
    fp = len(sp_p - sp_r)
    fn = len(sp_r - sp_p)
    rf = fp + fn
    max_sp = n - 3
    return {
        "n_shared": n,
        "pred_splits": len(sp_p),
        "ref_splits": len(sp_r),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "rf": rf,
        "nrf": round(rf / (2.0 * max_sp), 5),
        "false_positive_rate_pct": round(100.0 * fp / max(1, len(sp_p)), 2),
        "precision_pct": round(100.0 * tp / max(1, len(sp_p)), 2),
        "recall_pct": round(100.0 * tp / max(1, len(sp_r)), 2),
    }


def load_mammal_taxonomy(mammal_set: set) -> dict:
    tax = {}
    with open(ASSEMBLIES_TSV) as f:
        f.readline()  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= 6:
                continue
            asm = parts[6].strip()
            if asm not in mammal_set:
                continue
            species = parts[1].strip()
            genus = species.split()[0] if species else ""
            tokens = [x.strip() for x in parts[5].split(";")]
            so = next((t for t in tokens if t in SUPERORDERS), "")
            od = next((t for t in tokens if t in ORDERS), "")
            if od == "Artiodactyla":
                od = "Cetartiodactyla"
            fm = next((t for t in tokens if t.endswith("idae")), "")
            tax[asm] = {
                "species": species,
                "superorder": so,
                "order": od,
                "family": fm,
                "genus": genus,
            }
    return tax


def evaluate_4rank_taxonomy(root: TreeNode, tax_map: dict) -> dict:
    leaves, _ = get_leaf_names_and_hub_degrees(root)
    n_all = len(leaves)
    t2b_all = {tx: (1 << i) for i, tx in enumerate(leaves)}
    full_all = (1 << n_all) - 1
    splits_all = list(extract_canonical_splits(root, t2b_all, full_all))

    out_ranks = {}
    for rank in ("superorder", "order", "family", "genus"):
        ann_mask = 0
        grp_masks = defaultdict(int)
        grp_counts = defaultdict(int)
        n_ann = 0
        for tx in leaves:
            rk_val = tax_map.get(tx, {}).get(rank)
            if rk_val:
                b = t2b_all[tx]
                ann_mask |= b
                grp_masks[rk_val] |= b
                grp_counts[rk_val] += 1
                n_ann += 1
        if n_ann < 4:
            continue

        proj_splits = set()
        for s in splits_all:
            ps = s & ann_mask
            bc = bin(ps).count("1")
            if 2 <= bc <= n_ann - 2:
                comp = ann_mask ^ ps
                proj_splits.add(ps if ps < comp else comp)
        proj_list = list(proj_splits)

        valid_grps = {g: m for g, m in grp_masks.items() if 2 <= grp_counts[g] <= n_ann - 2}
        strict = 0
        compat = 0
        total_crossing_splits = 0

        for g, gm in valid_grps.items():
            gcomp = ann_mask ^ gm
            canon = gm if gm < gcomp else gcomp
            if canon in proj_splits:
                strict += 1
                compat += 1
            else:
                cross_cnt = 0
                for s in proj_list:
                    if (s & gm) and (s & gcomp) and ((ann_mask ^ s) & gm) and ((ann_mask ^ s) & gcomp):
                        cross_cnt += 1
                if cross_cnt == 0:
                    compat += 1
                total_crossing_splits += cross_cnt

        ng = len(valid_grps)
        out_ranks[rank] = {
            "n_groups": ng,
            "strict_monophyletic": strict,
            "strict_pct": round(100.0 * strict / max(1, ng), 2),
            "compatible_uncontradicted": compat,
            "compatible_pct": round(100.0 * compat / max(1, ng), 2),
            "mean_contradicting_splits": round(total_crossing_splits / max(1, ng), 3),
        }
    return out_ranks


def evaluate_and_record(
    name: str,
    modality: str,
    nwk_path: Path,
    ref_root: TreeNode,
    tax_map: dict,
    wall_sec: float,
    extra: dict = None,
) -> dict:
    root = parse_newick_tree(nwk_path.read_text())
    vs_ref = compare_unrooted_trees(root, ref_root)
    tax_4 = evaluate_4rank_taxonomy(root, tax_map)
    rec = {
        "method": name,
        "modality": modality,
        "tree_path": str(nwk_path),
        "wall_sec": round(wall_sec, 2),
        "vs_species_tree": vs_ref,
        "taxonomy_4ranks": tax_4,
    }
    if extra:
        rec.update(extra)
    print(
        f"  [{name:28s}] ({modality}) wall={wall_sec:7.2f}s | nRF={vs_ref['nrf']:.4f} "
        f"(Prec={vs_ref['precision_pct']:5.1f}%, Rec={vs_ref['recall_pct']:5.1f}%, FP={vs_ref['false_positive_rate_pct']:4.2f}%) | "
        f"Superord={tax_4['superorder']['strict_pct']:5.1f}%, "
        f"Order={tax_4['order']['strict_pct']:5.1f}%, "
        f"Family={tax_4['family']['strict_pct']:5.1f}%, "
        f"Genus={tax_4['genus']['strict_pct']:5.1f}%",
        flush=True,
    )
    return rec


def parse_fasta_dict(fa_path: Path) -> dict:
    seqs = {}
    duplicates = set()
    cur_id = None
    buf = []
    with open(fa_path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith(">"):
                if cur_id is not None:
                    if cur_id in seqs:
                        duplicates.add(cur_id)
                    seqs[cur_id] = "".join(buf)
                cur_id = s[1:].split()[0]
                buf = []
            else:
                buf.append(s.upper())
        if cur_id is not None:
            if cur_id in seqs:
                duplicates.add(cur_id)
            seqs[cur_id] = "".join(buf)
    return seqs, duplicates


def clean_and_translate_codons(raw_nt: str) -> tuple:
    """
    Cleans an in-frame nucleotide string (len % 3 == 0):
      - Any character outside A,C,G,T,- becomes N
      - In-frame stop codons (TAA, TAG, TGA) become '---' in NT and '-' in AA
      - Sense codons of pure A/C/G/T translate to standard 20-state AA; all other codons become '-'
    """
    n = len(raw_nt)
    nt_out = []
    aa_out = []
    for i in range(0, n, 3):
        codon = raw_nt[i : i + 3]
        if codon in STANDARD_GENETIC_CODE:
            aa = STANDARD_GENETIC_CODE[codon]
            if aa == "*":
                nt_out.append("---")
                aa_out.append("-")
            else:
                nt_out.append(codon)
                aa_out.append(aa)
        else:
            c0 = codon[0] if codon[0] in VALID_NT_CHARS else "N"
            c1 = codon[1] if codon[1] in VALID_NT_CHARS else "N"
            c2 = codon[2] if codon[2] in VALID_NT_CHARS else "N"
            nt_out.append(c0 + c1 + c2)
            aa_out.append("-")
    return "".join(nt_out), "".join(aa_out)


def build_supermatrices(mammal_ids: list, min_taxa: int = 875) -> dict:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    if NT_FASTA.exists() and AA_FASTA.exists() and MANIFEST_JSON.exists():
        manifest = json.loads(MANIFEST_JSON.read_text())
        if (
            manifest.get("n_taxa") == len(mammal_ids)
            and manifest.get("min_taxa_threshold") == min_taxa
            and NT_FASTA.stat().st_size > 100_000_000
            and AA_FASTA.stat().st_size > 30_000_000
        ):
            print(
                f"[CACHE HIT] Reusing {NT_FASTA.name} ({manifest['nt_sites']:,} bp) "
                f"and {AA_FASTA.name} ({manifest['aa_sites']:,} AA) across {manifest['n_loci']} loci.",
                flush=True,
            )
            return manifest

    t0 = time.perf_counter()
    candidate_genes = []
    with open(MASTER_JSONL) as f:
        for line in f:
            d = json.loads(line)
            if d.get("status") == "ok" and d.get("n_taxa", 0) >= min_taxa:
                candidate_genes.append(d["gene"])
    candidate_genes.sort()

    mammal_set = set(mammal_ids)
    nt_chunks = {m: [] for m in mammal_ids}
    aa_chunks = {m: [] for m in mammal_ids}

    kept_loci = []
    skipped_loci = []
    total_nt_len = 0
    total_present_pairs = 0

    print(f"Building TOGA2 mammal882 supermatrices from {len(candidate_genes)} candidate loci (n_taxa >= {min_taxa})...", flush=True)
    for g in candidate_genes:
        fa_path = INDIV_ALIS_DIR / f"{g}.codonified.fa"
        if not fa_path.exists():
            skipped_loci.append((g, "missing_fasta"))
            continue
        seqs, dups = parse_fasta_dict(fa_path)
        overlap_dups = dups & mammal_set
        if overlap_dups:
            skipped_loci.append((g, f"duplicate_taxa:{len(overlap_dups)}"))
            continue
        lengths = {len(seq) for m, seq in seqs.items() if m in mammal_set}
        if len(lengths) != 1:
            skipped_loci.append((g, f"inconsistent_lengths:{lengths}"))
            continue
        l_nt = next(iter(lengths))
        if l_nt == 0 or l_nt % 3 != 0:
            skipped_loci.append((g, f"non_codon_length:{l_nt}"))
            continue

        l_aa = l_nt // 3
        gap_nt = "-" * l_nt
        gap_aa = "-" * l_aa
        present_here = 0
        for m in mammal_ids:
            s = seqs.get(m)
            if s is None:
                nt_chunks[m].append(gap_nt)
                aa_chunks[m].append(gap_aa)
            else:
                cnt, caa = clean_and_translate_codons(s)
                nt_chunks[m].append(cnt)
                aa_chunks[m].append(caa)
                present_here += 1

        kept_loci.append(g)
        total_nt_len += l_nt
        total_present_pairs += present_here

    total_aa_len = total_nt_len // 3
    occupancy_pct = round(100.0 * total_present_pairs / max(1, len(kept_loci) * len(mammal_ids)), 3)

    nt_tmp = NT_FASTA.with_suffix(".fna.tmp")
    aa_tmp = AA_FASTA.with_suffix(".faa.tmp")
    with open(nt_tmp, "w") as f_nt, open(aa_tmp, "w") as f_aa:
        for m in mammal_ids:
            nt_seq = "".join(nt_chunks[m])
            aa_seq = "".join(aa_chunks[m])
            assert len(nt_seq) == total_nt_len
            assert len(aa_seq) == total_aa_len
            f_nt.write(f">{m}\n{nt_seq}\n")
            f_aa.write(f">{m}\n{aa_seq}\n")
    nt_tmp.replace(NT_FASTA)
    aa_tmp.replace(AA_FASTA)

    manifest = {
        "n_taxa": len(mammal_ids),
        "min_taxa_threshold": min_taxa,
        "n_candidate_loci": len(candidate_genes),
        "n_loci": len(kept_loci),
        "n_skipped_loci": len(skipped_loci),
        "skipped_loci": skipped_loci,
        "nt_sites": total_nt_len,
        "aa_sites": total_aa_len,
        "taxon_locus_occupancy_pct": occupancy_pct,
        "build_wall_sec": round(time.perf_counter() - t0, 2),
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2))
    print(
        f"[SUPERMATRIX BUILT] {len(mammal_ids)} mammals x {len(kept_loci)} loci | "
        f"NT={total_nt_len:,} bp | AA={total_aa_len:,} AA | "
        f"Occupancy={occupancy_pct:.2f}% | Built in {manifest['build_wall_sec']:.2f}s",
        flush=True,
    )
    return manifest


def phase_build_and_bw(threads: int):
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    METHODS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    ref_root = parse_newick_tree(SPECIES_TREE_PATH.read_text())
    mammal_ids, _ = get_leaf_names_and_hub_degrees(ref_root)
    tax_map = load_mammal_taxonomy(set(mammal_ids))
    build_supermatrices(mammal_ids, min_taxa=875)

    bw_dir = RUNS_DIR / "bushwhacker"
    bw_dir.mkdir(parents=True, exist_ok=True)

    # 1. Bushwhacker Coding-Aware Nucleotide
    print(f"\n=== Running Bushwhacker Coding-Aware NT (-T {threads} --no-shunts) ===", flush=True)
    bw_nt_json = bw_dir / "bw_nt.json"
    t0 = time.perf_counter()
    subprocess.run(
        [
            BUSHWHACKER_BIN,
            str(NT_FASTA),
            "--no-shunts",
            "-T",
            str(threads),
            "-j",
            str(bw_nt_json),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    bw_nt_wall = time.perf_counter() - t0
    bw_nt_data = json.loads(bw_nt_json.read_text())

    bw_nt_mdl_nwk = bw_dir / "bw_nt_mdl.nwk"
    bw_nt_bin_nwk = bw_dir / "bw_nt_binary.nwk"
    bw_nt_mdl_nwk.write_text(bw_nt_data["newick"] + "\n")
    bw_nt_bin_nwk.write_text(bw_nt_data["binary_newick"] + "\n")

    rec_nt_mdl = evaluate_and_record(
        "Bushwhacker_NT_MDL",
        "NT",
        bw_nt_mdl_nwk,
        ref_root,
        tax_map,
        bw_nt_wall,
        {
            "engine_sec": round(bw_nt_data["runtime_sec"], 2),
            "resolved_splits": bw_nt_data["resolved_internal_splits"],
            "max_binary_splits": bw_nt_data["max_binary_splits"],
            "resolved_pct": round(
                100.0 * bw_nt_data["resolved_internal_splits"] / max(1, bw_nt_data["max_binary_splits"]), 2
            ),
        },
    )
    rec_nt_bin = evaluate_and_record(
        "Bushwhacker_NT_Binary",
        "NT",
        bw_nt_bin_nwk,
        ref_root,
        tax_map,
        bw_nt_wall,
        {"engine_sec": round(bw_nt_data["runtime_sec"], 2)},
    )
    (METHODS_DIR / "Bushwhacker_NT_MDL.json").write_text(json.dumps(rec_nt_mdl, indent=2))
    (METHODS_DIR / "Bushwhacker_NT_Binary.json").write_text(json.dumps(rec_nt_bin, indent=2))

    # 2. Bushwhacker Upgraded 20-State LG08+CAT8 Protein
    print(f"\n=== Running Bushwhacker 20-State LG08+CAT8 AA (-T {threads} --protein --no-shunts) ===", flush=True)
    bw_aa_json = bw_dir / "bw_aa.json"
    t0 = time.perf_counter()
    subprocess.run(
        [
            BUSHWHACKER_BIN,
            str(AA_FASTA),
            "--protein",
            "--no-shunts",
            "-T",
            str(threads),
            "-j",
            str(bw_aa_json),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    bw_aa_wall = time.perf_counter() - t0
    bw_aa_data = json.loads(bw_aa_json.read_text())

    bw_aa_mdl_nwk = bw_dir / "bw_aa_mdl.nwk"
    bw_aa_bin_nwk = bw_dir / "bw_aa_binary.nwk"
    bw_aa_mdl_nwk.write_text(bw_aa_data["newick"] + "\n")
    bw_aa_bin_nwk.write_text(bw_aa_data["binary_newick"] + "\n")

    rec_aa_mdl = evaluate_and_record(
        "Bushwhacker_AA_LG08_CAT8_MDL",
        "AA",
        bw_aa_mdl_nwk,
        ref_root,
        tax_map,
        bw_aa_wall,
        {
            "engine_sec": round(bw_aa_data["runtime_sec"], 2),
            "resolved_splits": bw_aa_data["resolved_internal_splits"],
            "max_binary_splits": bw_aa_data["max_binary_splits"],
            "resolved_pct": round(
                100.0 * bw_aa_data["resolved_internal_splits"] / max(1, bw_aa_data["max_binary_splits"]), 2
            ),
        },
    )
    rec_aa_bin = evaluate_and_record(
        "Bushwhacker_AA_LG08_CAT8_Binary",
        "AA",
        bw_aa_bin_nwk,
        ref_root,
        tax_map,
        bw_aa_wall,
        {"engine_sec": round(bw_aa_data["runtime_sec"], 2)},
    )
    (METHODS_DIR / "Bushwhacker_AA_LG08_CAT8_MDL.json").write_text(json.dumps(rec_aa_mdl, indent=2))
    (METHODS_DIR / "Bushwhacker_AA_LG08_CAT8_Binary.json").write_text(json.dumps(rec_aa_bin, indent=2))

    phase_merge(threads)


def phase_run_method(method_code: str, threads: int):
    METHODS_DIR.mkdir(parents=True, exist_ok=True)
    rdir = RUNS_DIR / method_code.lower()
    rdir.mkdir(parents=True, exist_ok=True)

    ref_root = parse_newick_tree(SPECIES_TREE_PATH.read_text())
    mammal_ids, _ = get_leaf_names_and_hub_degrees(ref_root)
    tax_map = load_mammal_taxonomy(set(mammal_ids))
    env_omp = {**os.environ, "OMP_NUM_THREADS": str(threads)}

    if method_code == "FASTTREE_NT":
        name = "FastTreeMP_GTR_NT"
        modality = "NT"
        out_nwk = rdir / "fasttree_gtr_nt.nwk"
        log_path = rdir / "fasttree_gtr_nt.log"
        print(f"=== Running {name} (OMP_NUM_THREADS={threads}) ===", flush=True)
        t0 = time.perf_counter()
        with open(out_nwk, "w") as fout, open(log_path, "w") as flog:
            subprocess.run(
                [FASTTREE_BIN, "-gtr", "-nt", str(NT_FASTA)],
                stdout=fout,
                stderr=flog,
                env=env_omp,
                check=True,
            )
        wall = time.perf_counter() - t0

    elif method_code == "IQTREE_FAST_NT":
        name = "IQTREE2_GTR_G4_fast_NT"
        modality = "NT"
        prefix = rdir / "iqtree_gtr_g4_fast_nt"
        out_nwk = Path(str(prefix) + ".treefile")
        log_path = rdir / "iqtree_gtr_g4_fast_nt.stdout.log"
        print(f"=== Running {name} (-T {threads}) ===", flush=True)
        t0 = time.perf_counter()
        with open(log_path, "w") as flog:
            subprocess.run(
                [
                    IQTREE_BIN,
                    "-s",
                    str(NT_FASTA),
                    "-st",
                    "DNA",
                    "-m",
                    "GTR+G4",
                    "--fast",
                    "-T",
                    str(threads),
                    "-pre",
                    str(prefix),
                    "-redo",
                ],
                stdout=flog,
                stderr=subprocess.STDOUT,
                check=True,
            )
        wall = time.perf_counter() - t0

    elif method_code == "FASTTREE_AA":
        name = "FastTreeMP_LG_AA"
        modality = "AA"
        out_nwk = rdir / "fasttree_lg_aa.nwk"
        log_path = rdir / "fasttree_lg_aa.log"
        print(f"=== Running {name} (OMP_NUM_THREADS={threads}) ===", flush=True)
        t0 = time.perf_counter()
        with open(out_nwk, "w") as fout, open(log_path, "w") as flog:
            subprocess.run(
                [FASTTREE_BIN, "-lg", str(AA_FASTA)],
                stdout=fout,
                stderr=flog,
                env=env_omp,
                check=True,
            )
        wall = time.perf_counter() - t0

    elif method_code == "IQTREE_FAST_AA":
        name = "IQTREE2_LG_G4_fast_AA"
        modality = "AA"
        prefix = rdir / "iqtree_lg_g4_fast_aa"
        out_nwk = Path(str(prefix) + ".treefile")
        log_path = rdir / "iqtree_lg_g4_fast_aa.stdout.log"
        print(f"=== Running {name} (-T {threads}) ===", flush=True)
        t0 = time.perf_counter()
        with open(log_path, "w") as flog:
            subprocess.run(
                [
                    IQTREE_BIN,
                    "-s",
                    str(AA_FASTA),
                    "-st",
                    "AA",
                    "-m",
                    "LG+G4",
                    "--fast",
                    "-T",
                    str(threads),
                    "-pre",
                    str(prefix),
                    "-redo",
                ],
                stdout=flog,
                stderr=subprocess.STDOUT,
                check=True,
            )
        wall = time.perf_counter() - t0

    else:
        raise ValueError(f"Unknown --method: {method_code}")

    rec = evaluate_and_record(name, modality, out_nwk, ref_root, tax_map, wall)
    (METHODS_DIR / f"{name}.json").write_text(json.dumps(rec, indent=2))
    print(f"[SAVED] {METHODS_DIR / f'{name}.json'}", flush=True)


def phase_merge(threads: int):
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    ref_root = parse_newick_tree(SPECIES_TREE_PATH.read_text())
    mammal_ids, _ = get_leaf_names_and_hub_degrees(ref_root)
    tax_map = load_mammal_taxonomy(set(mammal_ids))
    manifest = json.loads(MANIFEST_JSON.read_text()) if MANIFEST_JSON.exists() else {}

    method_order = [
        "Bushwhacker_NT_MDL",
        "Bushwhacker_NT_Binary",
        "FastTreeMP_GTR_NT",
        "IQTREE2_GTR_G4_fast_NT",
        "Bushwhacker_AA_LG08_CAT8_MDL",
        "Bushwhacker_AA_LG08_CAT8_Binary",
        "FastTreeMP_LG_AA",
        "IQTREE2_LG_G4_fast_AA",
    ]

    methods_dict = {}
    parsed_trees = {"Senckenberg_SpeciesTree": ref_root}

    for mname in method_order:
        mfile = METHODS_DIR / f"{mname}.json"
        if mfile.exists():
            rec = json.loads(mfile.read_text())
            methods_dict[mname] = rec
            tpath = Path(rec["tree_path"])
            if tpath.exists():
                parsed_trees[mname] = parse_newick_tree(tpath.read_text())

    # Compute speedups relative to Bushwhacker NT and AA wall-clock times
    bw_nt_wall = methods_dict.get("Bushwhacker_NT_MDL", {}).get("wall_sec")
    bw_aa_wall = methods_dict.get("Bushwhacker_AA_LG08_CAT8_MDL", {}).get("wall_sec")
    for mname, rec in methods_dict.items():
        w = rec.get("wall_sec", 0.0)
        if w > 0:
            if rec.get("modality") == "NT" and bw_nt_wall and bw_nt_wall > 0:
                rec["speedup_of_bushwhacker_same_modality_x"] = round(w / bw_nt_wall, 2)
            elif rec.get("modality") == "AA" and bw_aa_wall and bw_aa_wall > 0:
                rec["speedup_of_bushwhacker_same_modality_x"] = round(w / bw_aa_wall, 2)

    pw_keys = [k for k in method_order if k in parsed_trees]
    pairwise_nrf = {}
    for i, k1 in enumerate(pw_keys):
        for k2 in pw_keys[i + 1 :]:
            pairwise_nrf[f"{k1}__vs__{k2}"] = compare_unrooted_trees(parsed_trees[k1], parsed_trees[k2])["nrf"]

    payload = {
        "benchmark": "TOGA2 882-Species Placental Mammalian Tree of Life (Coding NT & 20-State AA Supermatrix)",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "threads_per_method": threads,
        "dataset": manifest,
        "senckenberg_ref_taxonomy_4ranks": evaluate_4rank_taxonomy(ref_root, tax_map),
        "methods": methods_dict,
        "pairwise_nrf": pairwise_nrf,
    }
    tmp = OUT_JSON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(OUT_JSON)
    print(f"\n[MERGED] Wrote {OUT_JSON} ({len(methods_dict)}/{len(method_order)} method records).", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["build_and_bw", "run_method", "merge"])
    ap.add_argument(
        "--method",
        choices=["FASTTREE_NT", "IQTREE_FAST_NT", "FASTTREE_AA", "IQTREE_FAST_AA"],
        help="Required when --phase run_method",
    )
    ap.add_argument("--threads", type=int, default=int(os.environ.get("BENCH_THREADS", "32")))
    args = ap.parse_args()

    if args.phase == "build_and_bw":
        phase_build_and_bw(args.threads)
    elif args.phase == "run_method":
        if not args.method:
            ap.error("--method is required when --phase run_method")
        phase_run_method(args.method, args.threads)
    elif args.phase == "merge":
        phase_merge(args.threads)


if __name__ == "__main__":
    main()
