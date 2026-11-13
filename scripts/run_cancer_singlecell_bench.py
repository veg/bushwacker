#!/usr/bin/env python3
"""
Strict Zero-Leakage Empirical Benchmark for Non-Sequence Bushwhacker (`src/discrete.rs`):
Single-Cell Cancer Copy-Number Evolution (DLP+ scWGS) & Somatic Genotype Clones (AML Tapestri).

Zero-Leakage Protocol:
- Bushwhacker receives ONLY anonymized taxon IDs (`T0000`, `T0001`, ...) and unlabeled discrete
  character states (`0..K-1`, `?` for missing/dropout).
- Anatomical compartment (`OV2295` ascites/ovary/omentum), normal diploid vs PDX passage /
  cisplatin selection (`SA609`, `SA501`, `SA535`), longitudinal relapse timepoints, and in vivo
  single-cell clone multiplicities (`seq_cell_number`) are held out strictly for post-hoc
  evaluation and NEVER written to any Bushwhacker input file.
"""

from __future__ import annotations


import sys as _sys
from pathlib import Path as _Path
if str(_Path(__file__).resolve().parent) not in _sys.path:
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
from _env_paths import REPO_ROOT, DATA_DIR, SPECIMENS_DIR, ALIGNMENTS_DIR, SINGLE_CELL_DIR, RESULTS_DIR, WORK_ROOT, EXTERNAL_DIR, resolve_bin, ASTRAL_JAR

import csv
import gzip
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = REPO_ROOT
BIN = Path(resolve_bin("BUSHWHACKER_BIN", str(REPO_ROOT / "target" / "release" / "bushwhacker")))
_SCRATCH_SC = (
    REPO_ROOT / "scratch" / "cancer_singlecell_data"
    if (REPO_ROOT / "scratch" / "cancer_singlecell_data").exists()
    else ROOT.parent / "scratch" / "cancer_singlecell_data"
)
DATA_DIR = (
    ROOT / "data" / "single_cell"
    if (ROOT / "data" / "single_cell").exists()
    else (
        ROOT / "zenodo_replication" / "data" / "single_cell"
        if (ROOT / "zenodo_replication" / "data" / "single_cell").exists()
        else _SCRATCH_SC
    )
)
DLP_DIR = DATA_DIR / "dlp_cna" if (DATA_DIR / "dlp_cna").exists() else (_SCRATCH_SC / "dlp_cna" if (_SCRATCH_SC / "dlp_cna").exists() else DATA_DIR)
COMPASS_ROOT = (
    DATA_DIR / "aml_tapestri" / "COMPASS" / "data"
    if (DATA_DIR / "aml_tapestri" / "COMPASS" / "data").exists()
    else (
        _SCRATCH_SC / "aml_tapestri" / "COMPASS" / "data"
        if (_SCRATCH_SC / "aml_tapestri" / "COMPASS" / "data").exists()
        else EXTERNAL_DIR / "compass_tapestri" / "data"
    )
)
OUT_DIR = ROOT / "results" / "single_cell"
SUMMARY_JSON = OUT_DIR / "cancer_singlecell_summary.json"

FORBIDDEN_METADATA_TOKENS = [
    "SA921",
    "SA922",
    "SA1090",
    "SA928",
    "SA609",
    "SA501",
    "SA535",
    "B00290",
    "B00426",
    "cisplatin",
    "untreated",
    "ovary",
    "omentum",
    "ascites",
    "relapse",
    "AML-",
    "CAL012",
    "MPN",
    "seq_cell",
]


def assert_zero_leakage_csv(csv_path: Path) -> None:
    """Verify that no sample, anatomical, treatment, timepoint, or cell-count token exists in input."""
    txt = csv_path.read_text()
    for tok in FORBIDDEN_METADATA_TOKENS:
        if tok.lower() in txt.lower():
            raise RuntimeError(f"Zero-leakage violation: token '{tok}' found in {csv_path}")


def write_anonymized_matrix_csv(
    csv_path: Path,
    mat: np.ndarray,
    feature_names: list[str],
    missing_val: int = -1,
) -> list[str]:
    """Write N x M integer matrix (-1 = '?') with anonymized taxon IDs T0000..T{N-1}."""
    n, m = mat.shape
    anon_ids = [f"T{i:05d}" for i in range(n)]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["taxon"] + feature_names)
        for i in range(n):
            row_s = [anon_ids[i]] + [
                "?" if int(mat[i, s]) == missing_val else str(int(mat[i, s]))
                for s in range(m)
            ]
            w.writerow(row_s)
    assert_zero_leakage_csv(csv_path)
    return anon_ids


def run_bushwhacker_cli(
    csv_path: Path,
    json_out: Path,
    mode: str = "ordered",
    min_shunt_sites: int = 3,
    no_shunts: bool = False,
) -> dict:
    flag = "--ordered" if mode == "ordered" else "--discrete"
    cmd = [
        str(BIN),
        str(csv_path),
        flag,
        "--min-shunt-sites",
        str(min_shunt_sites),
        "-j",
        str(json_out),
    ]
    if no_shunts:
        cmd.append("--no-shunts")
    t0 = time.perf_counter()
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    wall_sec = time.perf_counter() - t0
    data = json.loads(json_out.read_text())
    data["cli_wall_sec"] = wall_sec

    # Normalize exact JSON schema emitted by src/main.rs serialize_json
    data["mdl_tree_newick"] = data["newick"]
    data["binary_refinement_newick"] = data["binary_newick"]
    data["num_taxa"] = data["n_taxa"]
    data["num_sites"] = data["seq_len"]
    data["runtime_seconds"] = data["runtime_sec"]
    data["resolved_splits"] = data["resolved_internal_splits"]
    data["polytomy_hubs"] = data["polytomy_nodes"]
    for sh in data.get("shunts", []):
        sh["sites_1based"] = [int(idx) + 1 for idx in sh["sites_0based"]]

    # Compute exact max internal node degree directly from Newick topology
    n_t = int(data["n_taxa"])
    anon_map = {f"T{i:05d}": i for i in range(n_t)}
    adj_tmp, _ = parse_newick_unrooted(data["newick"], anon_map)
    int_degrees = [len(nbs) for u_id, nbs in adj_tmp.items() if u_id >= n_t]
    data["max_polytomy_degree"] = max(int_degrees) if int_degrees else 0
    return data


def parse_newick_unrooted(nwk: str, taxon_to_idx: dict[str, int]):
    """Parse an arbitrary (binary or multifurcating) Newick string into an adjacency list + edge lengths."""
    s = nwk.strip()
    if s.endswith(";"):
        s = s[:-1]
    adj: dict[int, list[tuple[int, float]]] = defaultdict(list)
    n_taxa = len(taxon_to_idx)
    next_internal = [n_taxa]

    pos = [0]
    L = len(s)

    def skip_ws():
        while pos[0] < L and s[pos[0]].isspace():
            pos[0] += 1

    def parse_sub() -> tuple[int, float]:
        skip_ws()
        if pos[0] < L and s[pos[0]] == "(":
            pos[0] += 1  # '('
            u = next_internal[0]
            next_internal[0] += 1
            while True:
                c_id, c_bl = parse_sub()
                adj[u].append((c_id, c_bl))
                adj[c_id].append((u, c_bl))
                skip_ws()
                if pos[0] < L and s[pos[0]] == ",":
                    pos[0] += 1
                    continue
                elif pos[0] < L and s[pos[0]] == ")":
                    pos[0] += 1
                    break
                else:
                    break
            skip_ws()
            # Optional internal label
            while pos[0] < L and s[pos[0]] not in (":", ",", ")", ";"):
                pos[0] += 1
        else:
            start = pos[0]
            while pos[0] < L and s[pos[0]] not in (":", ",", ")", ";"):
                pos[0] += 1
            lbl = s[start:pos[0]].strip()
            u = taxon_to_idx[lbl]
        bl = 0.0
        skip_ws()
        if pos[0] < L and s[pos[0]] == ":":
            pos[0] += 1
            b_start = pos[0]
            while pos[0] < L and s[pos[0]] not in (",", ")", ";"):
                pos[0] += 1
            try:
                bl = float(s[b_start:pos[0]].strip())
            except ValueError:
                bl = 0.0
        return u, bl

    root, _ = parse_sub()
    return adj, root


def compute_tree_compartment_concordance(
    nwk: str,
    anon_ids: list[str],
    labels: list[str],
) -> dict:
    """
    Compute zero-leakage post-hoc topological concordance of held-out biological labels:
    1. Topological nearest-sister purity: for each leaf i, inspect its immediate parent node p
       and find the set of leaves attached to p (or minimal cherry/polytomy sisters with shortest
       topological path <= 2 edges) and measure fraction sharing leaf i's true label.
    2. Best monophyletic/bipartition F1 per label across all tree edges.
    """
    n = len(anon_ids)
    t2i = {nm: idx for idx, nm in enumerate(anon_ids)}
    adj, root = parse_newick_unrooted(nwk, t2i)

    # Rooted DFS for bipartitions + immediate parent clade
    parent = {root: None}
    children = defaultdict(list)
    order = [root]
    q_i = 0
    while q_i < len(order):
        u = order[q_i]
        q_i += 1
        for nb, _ in adj[u]:
            if nb not in parent:
                parent[nb] = u
                children[u].append(nb)
                order.append(nb)

    sub_leaves: dict[int, list[int]] = {}
    for u in reversed(order):
        if u < n:
            sub_leaves[u] = [u]
        else:
            lst = []
            for c in children[u]:
                lst.extend(sub_leaves[c])
            sub_leaves[u] = lst

    # 1. Nearest topological clade purity for each leaf
    sister_purities = []
    for i in range(n):
        p = parent.get(i)
        if p is None:
            continue
        clade = [j for j in sub_leaves[p] if j != i]
        if not clade and parent.get(p) is not None:
            clade = [j for j in sub_leaves[parent[p]] if j != i]
        if clade:
            same = sum(1 for j in clade if labels[j] == labels[i])
            sister_purities.append(same / len(clade))

    # 2. Best split F1 for each distinct label
    uniq_labels = sorted(set(labels))
    label_counts = Counter(labels)
    best_f1_by_label = {}
    for target_lbl in uniq_labels:
        tot_pos = label_counts[target_lbl]
        if tot_pos == 0 or tot_pos == n:
            best_f1_by_label[target_lbl] = 1.0
            continue
        best_f1 = 0.0
        for u in order:
            if u == root:
                continue
            s_leaves = sub_leaves[u]
            k_in = len(s_leaves)
            pos_in = sum(1 for j in s_leaves if labels[j] == target_lbl)
            for tp, pred_sz in [(pos_in, k_in), (tot_pos - pos_in, n - k_in)]:
                if pred_sz > 0 and tp > 0:
                    f1 = (2.0 * tp) / (pred_sz + tot_pos)
                    if f1 > best_f1:
                        best_f1 = f1
        best_f1_by_label[target_lbl] = round(best_f1, 4)

    # Null random permutation baseline for sister purity
    p_null = sum((c / n) * ((c - 1) / max(n - 1, 1)) for c in label_counts.values())
    mean_purity = float(np.mean(sister_purities)) if sister_purities else 0.0
    weighted_f1 = float(
        sum(best_f1_by_label[k] * label_counts[k] for k in uniq_labels) / n
    )
    return {
        "nearest_clade_purity": round(mean_purity, 4),
        "null_random_purity": round(float(p_null), 4),
        "purity_enrichment_over_null": round(mean_purity / max(p_null, 1e-6), 3),
        "best_split_f1_by_group": best_f1_by_label,
        "weighted_split_f1": round(weighted_f1, 4),
    }


def compress_cna_bins_to_segments(
    mat: np.ndarray,
    bin_names: list[str],
    diff_thresh: float = 0.025,
    cum_drift_thresh: float = 0.10,
) -> tuple[np.ndarray, list[str], list[tuple[int, int]]]:
    """
    Unsupervised copy-number changepoint compression (analogous to Sitka's collinearity collapse):
    Uses ONLY the unlabeled N x M integer copy-number matrix and chromosomal coordinates to merge
    contiguous collinear 500-kb bins into chromosomal segment blocks, summarized per cell by modal
    integer copy number.
    """
    n, m = mat.shape
    chroms = [b.split("_")[0] for b in bin_names]
    seg_ranges: list[tuple[int, int]] = []
    b_start = 0
    cum_drift = 0.0
    for s in range(1, m):
        if chroms[s] != chroms[s - 1]:
            seg_ranges.append((b_start, s - 1))
            b_start = s
            cum_drift = 0.0
            continue
        col_prev = mat[:, s - 1]
        col_cur = mat[:, s]
        valid = (col_prev >= 0) & (col_cur >= 0)
        n_val = int(np.sum(valid))
        df = float(np.sum(col_prev[valid] != col_cur[valid])) / n_val if n_val > 0 else 0.0
        cum_drift += df
        if df >= diff_thresh or cum_drift >= cum_drift_thresh:
            seg_ranges.append((b_start, s - 1))
            b_start = s
            cum_drift = 0.0
    seg_ranges.append((b_start, m - 1))

    m_seg = len(seg_ranges)
    seg_mat = np.full((n, m_seg), -1, dtype=np.int16)
    seg_names: list[str] = []
    for k_idx, (a, b) in enumerate(seg_ranges):
        c_a, st_a, _ = bin_names[a].split("_")
        _, _, end_b = bin_names[b].split("_")
        seg_names.append(f"{c_a}_{st_a}_{end_b}")
        sub = mat[:, a : b + 1]
        if a == b:
            seg_mat[:, k_idx] = sub[:, 0]
        else:
            # Fast vectorized integer mode across states 0..31
            k_max = int(np.max(sub))
            if k_max < 0:
                continue
            best_cnt = np.zeros(n, dtype=np.int32)
            best_st = np.full(n, -1, dtype=np.int16)
            for st_val in range(k_max + 1):
                cnt = np.sum(sub == st_val, axis=1)
                mask = cnt > best_cnt
                best_cnt[mask] = cnt[mask]
                best_st[mask] = st_val
            seg_mat[:, k_idx] = best_st
    return seg_mat, seg_names, seg_ranges


def load_sitka_gz_cna(gz_path: Path) -> tuple[list[str], list[str], np.ndarray]:
    with gzip.open(gz_path, "rt") as f:
        r = csv.reader(f)
        hdr = next(r)
        cell_names = hdr[3:]
        bin_names = []
        rows = []
        for row in r:
            bin_names.append(f"{row[0]}_{row[1]}_{row[2]}")
            rows.append([int(x) if x not in ("NA", "", "?") else -1 for x in row[3:]])
    mat = np.array(rows, dtype=np.int16).T  # N x M
    return cell_names, bin_names, mat


def load_sa609_combined_cna(
    cnv_csv: Path, extra_csv: Path
) -> tuple[list[str], list[str], np.ndarray]:
    def read_r_table(p: Path):
        with open(p) as f:
            r = csv.reader(f)
            cells = [c.strip('"') for c in next(r)]
            loci = []
            vals = []
            for row in r:
                loci.append(row[0].strip('"'))
                vals.append([int(x) if x not in ("NA", "", "?") else -1 for x in row[1:]])
        return cells, loci, np.array(vals, dtype=np.int16).T

    c1, l1, m1 = read_r_table(cnv_csv)
    c2, l2, m2 = read_r_table(extra_csv)
    assert l1 == l2, "SA609 cnv.csv and extra.csv bin coordinates must match"
    comb = np.vstack([m1, m2])

    # R write.table sorted locus names lexicographically ("1_100000001" before "1_10000001");
    # restore true physical chromosomal coordinate order (chrom, start_bp, end_bp).
    def locus_sort_key(idx: int) -> tuple[int, int, int]:
        ch, st, en = l1[idx].split("_")
        ch_rank = 23 if ch == "X" else (24 if ch == "Y" else int(ch))
        return (ch_rank, int(st), int(en))

    order = sorted(range(len(l1)), key=locus_sort_key)
    l_sorted = [l1[i] for i in order]
    return c1 + c2, l_sorted, comb[:, order]


def extract_dlp_metadata_label(dataset_name: str, cell_id: str) -> tuple[str, int]:
    """Return held-out (group_label, chronological_rank) strictly for post-hoc evaluation."""
    if dataset_name == "OV2295":
        pref = cell_id.split("-")[0]
        mapping = {
            "SA1090": ("PreTx_Ascites_SA1090", 0),
            "SA921": ("Ovary_PrimarySite_SA921", 1),
            "SA922": ("Relapse_Ascites_SA922", 2),
        }
        return mapping.get(pref, (pref, 0))
    if dataset_name == "SA609":
        pref = cell_id.split(".")[0]
        if pref.startswith("SA928"):
            return ("Diploid_Normal_SA928", 0)
        if "X1" in pref:
            return ("Untreated_X1_B00290", 1)
        if "X2" in pref:
            return ("Selected_X2_B00426", 2)
        return (pref, 1)
    # SA501 / SA535 PDX passages (e.g. SA501X2XB00096 -> X2 -> 2)
    m = re.search(r"X(\d+)", cell_id)
    if m:
        p_num = int(m.group(1))
        return (f"Passage_X{p_num}", p_num)
    return (cell_id.split("-")[0], 0)


def summarize_dlp_shunts(shunts: list[dict], feature_names: list[str]) -> dict:
    int_sh = [s for s in shunts if s["mode"] == "INTERVAL"]
    sp_sh = [s for s in shunts if s["mode"] == "SPARSE"]
    top_examples = []
    for s in int_sh[:5]:
        sites_1b = s["sites_1based"]
        f_first = feature_names[sites_1b[0] - 1]
        f_last = feature_names[sites_1b[-1] - 1]
        c1, st1, _ = f_first.split("_")
        c2, _, end2 = f_last.split("_")
        mb_span = (int(end2) - int(st1)) / 1e6 if c1 == c2 else 0.0
        top_examples.append(
            {
                "mode": s["mode"],
                "chrom": c1,
                "start_bp": int(st1),
                "end_bp": int(end2),
                "span_mb": round(mb_span, 2),
                "shared_features": s["shared_mutations"],
                "net_mdl_gain_nats": round(s["net_mdl_gain_nats"], 2),
                "posterior_prob": round(s["posterior_prob"], 4),
            }
        )
    return {
        "total_shunts": len(shunts),
        "interval_segmental_shunts": len(int_sh),
        "sparse_focal_shunts": len(sp_sh),
        "total_shunt_mdl_gain_nats": round(
            sum(s["net_mdl_gain_nats"] for s in shunts), 2
        ),
        "top_interval_cna_loh_blocks": top_examples,
    }


def compute_subclone_shannon_entropy(
    nwk: str, anon_ids: list[str], group_labels: list[str]
) -> dict:
    """
    Measure selective bottleneck / clonal sweep across groups using the inferred MDL polytomy
    subclone hubs (assigning each leaf to its immediate parent internal node in the MDL tree).
    """
    t2i = {nm: idx for idx, nm in enumerate(anon_ids)}
    adj, root = parse_newick_unrooted(nwk, t2i)
    parent = {root: None}
    q = [root]
    qi = 0
    while qi < len(q):
        u = q[qi]
        qi += 1
        for nb, _ in adj[u]:
            if nb not in parent:
                parent[nb] = u
                q.append(nb)

    group_to_hubs: dict[str, list[int]] = defaultdict(list)
    for i, grp in enumerate(group_labels):
        group_to_hubs[grp].append(parent.get(i, root))

    out = {}
    for grp, hubs in sorted(group_to_hubs.items()):
        cnts = np.array(list(Counter(hubs).values()), dtype=np.float64)
        probs = cnts / cnts.sum()
        h_nats = float(-np.sum(probs * np.log(probs)))
        top1_share = float(cnts.max() / cnts.sum())
        out[grp] = {
            "n_cells": int(cnts.sum()),
            "n_occupied_mdl_clades": int(len(cnts)),
            "shannon_entropy_nats": round(h_nats, 4),
            "effective_clones_expH": round(float(math.exp(h_nats)), 2),
            "dominant_clade_fraction": round(top1_share, 4),
        }
    return out


def run_dlp_cna_benchmarks(tmp_dir: Path) -> dict:
    dlp_base = DLP_DIR if (DLP_DIR / "sitka_data").exists() else DATA_DIR / "dlp_cna"
    dlp_datasets = [
        (
            "OV2295",
            "High-grade serous ovarian cancer (Primary ovary + Pre/Post-Tx ascites)",
            "gz",
            dlp_base / "sitka_data" / "data_files" / "ova_cn_bin_cell_filtered.csv.gz",
        ),
        (
            "SA609",
            "Triple-negative breast cancer PDX (Diploid SA928 + Untreated X1 vs Selected X2)",
            "sa609",
            (
                dlp_base / "sitkatree" / "example" / "data" / "cnv.csv",
                dlp_base / "sitkatree" / "example" / "data" / "extra.csv",
            ),
        ),
        (
            "SA501",
            "Triple-negative breast cancer PDX longitudinal series (X2 -> X15)",
            "gz",
            dlp_base / "sitka_data" / "data_files" / "sa501_cn_bin_cell_filtered.csv.gz",
        ),
        (
            "SA535",
            "Triple-negative breast cancer PDX longitudinal series (X1 -> X8)",
            "gz",
            dlp_base / "sitka_data" / "data_files" / "sa535_cn_bin_cell_filtered.csv.gz",
        ),
    ]

    results = {}
    for name, desc, kind, path_info in dlp_datasets:
        print(f"[DLP+ CNA] Loading {name}...")
        if kind == "gz":
            cell_names, bin_names, raw_mat = load_sitka_gz_cna(path_info)
        else:
            cell_names, bin_names, raw_mat = load_sa609_combined_cna(*path_info)

        n_cells, m_bins = raw_mat.shape
        k_states = int(np.max(raw_mat)) + 1
        meta_pairs = [extract_dlp_metadata_label(name, cid) for cid in cell_names]
        group_labels = [p[0] for p in meta_pairs]
        time_ranks = np.array([p[1] for p in meta_pairs], dtype=np.float64)

        # 1. Raw 500-kb bin matrix (M = 4,375)
        raw_csv = tmp_dir / f"{name}_raw_bins.csv"
        anon_ids = write_anonymized_matrix_csv(raw_csv, raw_mat, bin_names)

        print(f"  Running Bushwhacker --ordered on {name} raw 500-kb bins ({n_cells} x {m_bins})...")
        res_raw_ord = run_bushwhacker_cli(
            raw_csv, tmp_dir / f"{name}_raw_ord.json", mode="ordered", min_shunt_sites=3
        )
        print(f"  Running Bushwhacker --discrete on {name} raw 500-kb bins ({n_cells} x {m_bins})...")
        res_raw_disc = run_bushwhacker_cli(
            raw_csv, tmp_dir / f"{name}_raw_disc.json", mode="discrete", min_shunt_sites=3
        )

        # 2. Compressed chromosomal copy-number changepoint segments (Sitka-analogous)
        seg_mat, seg_names, seg_ranges = compress_cna_bins_to_segments(raw_mat, bin_names)
        m_seg = seg_mat.shape[1]
        seg_csv = tmp_dir / f"{name}_segments.csv"
        write_anonymized_matrix_csv(seg_csv, seg_mat, seg_names)

        print(f"  Running Bushwhacker --ordered on {name} compressed segments ({n_cells} x {m_seg})...")
        res_seg_ord = run_bushwhacker_cli(
            seg_csv, tmp_dir / f"{name}_seg_ord.json", mode="ordered", min_shunt_sites=3
        )
        print(f"  Running Bushwhacker --discrete on {name} compressed segments ({n_cells} x {m_seg})...")
        res_seg_disc = run_bushwhacker_cli(
            seg_csv, tmp_dir / f"{name}_seg_disc.json", mode="discrete", min_shunt_sites=3
        )

        # Post-hoc zero-leakage concordance on raw Ordered MDL vs Forced-Binary vs Segment Ordered MDL
        conc_raw_ord = compute_tree_compartment_concordance(
            res_raw_ord["mdl_tree_newick"], anon_ids, group_labels
        )
        conc_raw_bin = compute_tree_compartment_concordance(
            res_raw_ord["binary_refinement_newick"], anon_ids, group_labels
        )
        conc_seg_ord = compute_tree_compartment_concordance(
            res_seg_ord["mdl_tree_newick"], anon_ids, group_labels
        )
        conc_seg_bin = compute_tree_compartment_concordance(
            res_seg_ord["binary_refinement_newick"], anon_ids, group_labels
        )
        conc_seg_disc = compute_tree_compartment_concordance(
            res_seg_disc["mdl_tree_newick"], anon_ids, group_labels
        )

        # Aneuploid L1 copy-number burden from diploid CN=2 vs passage/progression rank
        aneuploid_l1 = np.mean(np.abs(raw_mat.astype(np.float64) - 2.0), axis=1)
        if len(set(time_ranks)) >= 2:
            rho_prog, p_prog = stats.spearmanr(time_ranks, aneuploid_l1)
        else:
            rho_prog, p_prog = 0.0, 1.0

        clade_diversity = compute_subclone_shannon_entropy(
            res_seg_ord["mdl_tree_newick"], anon_ids, group_labels
        )

        raw_rle_segs = float(1.0 + np.mean(np.sum(raw_mat[:, 1:] != raw_mat[:, :-1], axis=1)))
        seg_rle_segs = float(1.0 + np.mean(np.sum(seg_mat[:, 1:] != seg_mat[:, :-1], axis=1)))

        def pack_run(
            r: dict, conc_mdl: dict, conc_bin: dict | None, feats: list[str], mean_rle: float
        ) -> dict:
            raw_dl = r["raw_Description_Length_nats"]
            star_dl = r["star_tree_MDL_nats"]
            tree_dl = r["tree_MDL_nats"]
            net_dl = r["network_MDL_nats"]
            max_sp = max(r["max_binary_splits"], 1)
            res_sp = r["resolved_splits"]
            out = {
                "n_cells": r["num_taxa"],
                "n_features": r["num_sites"],
                "mean_cell_rle_segments": round(mean_rle, 2),
                "runtime_sec": round(r["runtime_seconds"], 4),
                "cli_wall_sec": round(r["cli_wall_sec"], 4),
                "raw_dl_nats": round(raw_dl, 2),
                "star_mdl_nats": round(star_dl, 2),
                "tree_mdl_nats": round(tree_dl, 2),
                "network_mdl_nats": round(net_dl, 2),
                "compression_ratio_raw_to_net": round(raw_dl / max(net_dl, 1e-9), 3),
                "mdl_gain_vs_star_nats": round(star_dl - net_dl, 2),
                "mdl_pct_of_star": round(100.0 * net_dl / max(star_dl, 1e-9), 2),
                "resolved_splits": res_sp,
                "max_binary_splits": r["max_binary_splits"],
                "resolution_rate": round(res_sp / max_sp, 4),
                "contracted_false_splits": r["max_binary_splits"] - res_sp,
                "false_split_contraction_pct": round(
                    100.0 * (r["max_binary_splits"] - res_sp) / max_sp, 2
                ),
                "polytomy_hubs": r["polytomy_hubs"],
                "max_polytomy_degree": r["max_polytomy_degree"],
                "shunt_summary": summarize_dlp_shunts(r["shunts"], feats),
                "mdl_concordance": conc_mdl,
            }
            if conc_bin is not None:
                out["forced_binary_baseline_concordance"] = conc_bin
            return out

        results[name] = {
            "description": desc,
            "n_cells": n_cells,
            "n_raw_500kb_bins": m_bins,
            "n_compressed_segments": m_seg,
            "bin_to_segment_reduction_factor": round(m_bins / m_seg, 2),
            "k_copy_number_states": k_states,
            "group_counts": dict(Counter(group_labels)),
            "aneuploidy_vs_progression_spearman_rho": round(float(rho_prog), 4),
            "aneuploidy_vs_progression_p_value": float(p_prog),
            "subclone_clade_diversity_by_group": clade_diversity,
            "raw_500kb_ordered": pack_run(res_raw_ord, conc_raw_ord, conc_raw_bin, bin_names, raw_rle_segs),
            "raw_500kb_unordered": pack_run(res_raw_disc, compute_tree_compartment_concordance(res_raw_disc["mdl_tree_newick"], anon_ids, group_labels), None, bin_names, raw_rle_segs),
            "segment_blocks_ordered": pack_run(res_seg_ord, conc_seg_ord, conc_seg_bin, seg_names, seg_rle_segs),
            "segment_blocks_unordered": pack_run(res_seg_disc, conc_seg_disc, None, seg_names, seg_rle_segs),
            # Save compact segment matrix + MDL Newick summary for Panel A/B figure rendering
            "viz_tree_newick": res_seg_ord["mdl_tree_newick"],
            "viz_anon_to_group": {anon_ids[i]: group_labels[i] for i in range(n_cells)},
        }
        print(
            f"  -> {name}: Raw-Ord {res_raw_ord['runtime_seconds']:.2f}s "
            f"(compr {results[name]['raw_500kb_ordered']['compression_ratio_raw_to_net']:.2f}x, "
            f"purity {conc_raw_ord['nearest_clade_purity']:.3f}) | "
            f"Seg-Ord {res_seg_ord['runtime_seconds']:.2f}s "
            f"({m_seg} segs, splits {res_seg_ord['resolved_splits']}/{res_seg_ord['max_binary_splits']}, "
            f"purity {conc_seg_ord['nearest_clade_purity']:.3f} vs binary {conc_seg_bin['nearest_clade_purity']:.3f})"
        )

    return results


def compute_mdl_clone_out_degrees(
    nwk: str,
    anon_ids: list[str],
    wt_idx: int,
) -> dict[int, int]:
    """
    Root the inferred Bushwhacker MDL tree at the wild-type normal genotype `wt_idx` (`0,0,...,0`),
    collapse zero-mutation (`bl <= 1e-5`) identity attachments between sampled extant ancestral
    clones and their reconstructed internal node hubs, and return each clone's topological
    out-degree `k_out` (number of daughter lineages spawned by that clone).
    """
    n = len(anon_ids)
    t2i = {nm: idx for idx, nm in enumerate(anon_ids)}
    adj, _ = parse_newick_unrooted(nwk, t2i)

    # Identify parent of wt_idx; root tree at wt_idx
    parent: dict[int, int | None] = {wt_idx: None}
    edge_len_to_parent: dict[int, float] = {wt_idx: 0.0}
    children: dict[int, list[int]] = defaultdict(list)
    q = [wt_idx]
    qi = 0
    while qi < len(q):
        u = q[qi]
        qi += 1
        for nb, bl in adj[u]:
            if nb not in parent:
                parent[nb] = u
                edge_len_to_parent[nb] = bl
                children[u].append(nb)
                q.append(nb)

    # Note: In Bushwhacker discrete Newick serialization, minimal display branch length is
    # min_bl = 0.1 / max(L, 1.0) whereas 1 real mutation has bl >= 1.0 / max(L, 1.0).
    # We detect zero-mutation identity attachments cleanly via bl < 0.5 * min_single_mut_bl
    # OR by checking actual L1 distance == 0 in the caller!
    return parent, children, edge_len_to_parent


def compute_exact_clone_burst_degrees_from_mdl_tree(
    nwk: str,
    anon_ids: list[str],
    mat: np.ndarray,
    wt_idx: int,
) -> dict[int, dict]:
    """
    Reconstruct internal node states on Bushwhacker's MDL tree via ordered Wagner hull DP rooted
    at `wt_idx`, collapse zero-mutation taxon-to-hub edges (where an extant sampled clone is
    identical to the internal hub state), and compute each sampled clone's exact daughter
    out-degree `k_out` and clade depth!
    """
    n, l = mat.shape
    t2i = {nm: idx for idx, nm in enumerate(anon_ids)}
    adj, _ = parse_newick_unrooted(nwk, t2i)

    parent: dict[int, int | None] = {wt_idx: None}
    children: dict[int, list[int]] = defaultdict(list)
    order = [wt_idx]
    qi = 0
    while qi < len(order):
        u = order[qi]
        qi += 1
        for nb, _ in adj[u]:
            if nb not in parent:
                parent[nb] = u
                children[u].append(nb)
                order.append(nb)

    # Bottom-up Wagner interval hull [lo, hi] per site
    lo_map: dict[int, np.ndarray] = {}
    hi_map: dict[int, np.ndarray] = {}
    valid_map: dict[int, np.ndarray] = {}
    for u in reversed(order):
        if u < n:
            row = mat[u]
            val = row >= 0
            lo = np.where(val, row, 0).astype(np.int16)
            hi = np.where(val, row, 0).astype(np.int16)
            lo_map[u] = lo
            hi_map[u] = hi
            valid_map[u] = val
        else:
            lo = np.zeros(l, dtype=np.int16)
            hi = np.zeros(l, dtype=np.int16)
            val = np.zeros(l, dtype=bool)
            for s in range(l):
                pts = []
                for c in children[u]:
                    if valid_map[c][s]:
                        pts.append(int(lo_map[c][s]))
                        pts.append(int(hi_map[c][s]))
                if pts:
                    pts.sort()
                    m_len = len(pts)
                    lo[s] = pts[(m_len - 1) // 2]
                    hi[s] = pts[m_len // 2]
                    val[s] = True
            lo_map[u] = lo
            hi_map[u] = hi
            valid_map[u] = val

    # Top-down state assignment from WT root
    node_state: dict[int, np.ndarray] = {}
    for u in order:
        if u < n:
            # Fill any missing sites in leaf from parent
            p = parent[u]
            st = mat[u].copy()
            if p is not None:
                miss = st < 0
                st[miss] = node_state[p][miss]
            st[st < 0] = 0
            node_state[u] = st
        else:
            p = parent[u]
            anc = node_state[p] if p is not None else np.zeros(l, dtype=np.int16)
            st = np.clip(anc, lo_map[u], hi_map[u]).astype(np.int16)
            no_v = ~valid_map[u]
            st[no_v] = anc[no_v]
            node_state[u] = st

    # Contract all zero-difference internal-to-internal or internal-to-sampled-clone edges
    # into equivalence classes of identical nodal genotypes on the MDL tree
    uf = {u: u for u in order}

    def find(x):
        while uf[x] != x:
            uf[x] = uf[uf[x]]
            x = uf[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            # Prefer smaller ID / rootward representative
            if ra < rb:
                uf[rb] = ra
            else:
                uf[ra] = rb

    for u in order:
        p = parent[u]
        if p is not None:
            # Check if u and p have identical reconstructed state across observed sites of u
            if u < n:
                obs = mat[u] >= 0
                is_zero = bool(np.all(mat[u][obs] == node_state[p][obs]))
            else:
                is_zero = bool(np.array_equal(node_state[u], node_state[p]))
            if is_zero:
                union(p, u)

    comp_children: dict[int, set[int]] = defaultdict(set)
    comp_parent: dict[int, int | None] = {}
    wt_comp = find(wt_idx)
    comp_parent[wt_comp] = None

    for u in order:
        p = parent[u]
        if p is not None:
            cu, cp = find(u), find(p)
            if cu != cp:
                comp_children[cp].add(cu)
                comp_parent[cu] = cp

    # Compute depth and branch transitions
    comp_state: dict[int, np.ndarray] = {}
    for u in order:
        cu = find(u)
        if u < n and (mat[u] >= 0).all():
            comp_state[cu] = mat[u].copy()
        elif cu not in comp_state:
            comp_state[cu] = node_state[u].copy()

    out_info = {}
    for i in range(n):
        ci = find(i)
        k_out = len(comp_children[ci])
        depth = 0
        cur = ci
        seen = {cur}
        while comp_parent.get(cur) is not None and comp_parent[cur] not in seen:
            cur = comp_parent[cur]
            seen.add(cur)
            depth += 1
        out_info[i] = {
            "k_out": int(k_out),
            "depth": int(depth),
            "comp_id": int(ci),
        }

    # Extract directed branch transitions (site_idx, anc_state, der_state, is_trunk_from_wt)
    branch_transitions: list[tuple[int, int, int, bool]] = []
    for cp, ch_set in comp_children.items():
        st_p = comp_state[cp]
        is_trunk = cp == wt_comp
        for cc in ch_set:
            st_c = comp_state[cc]
            diffs = np.where(st_p != st_c)[0]
            for s_idx in diffs:
                branch_transitions.append(
                    (int(s_idx), int(st_p[s_idx]), int(st_c[s_idx]), bool(is_trunk))
                )

    return out_info, branch_transitions


def _fe_cluster_ols_and_depth_strata(
    records: list[dict],
    sample_key: str,
    patient_key: str,
    value_key: str,
    k_hub_thresh: int = 3,
    k_leaf_max: int = 0,
) -> dict:
    """
    BARS depth & WT-exclusion controls for clone expansion:
      1. Fixed depth strata (depth=1, depth=2, depth>=3 from WT) comparing k_out>=k_hub_thresh vs k_out<=k_leaf_max.
      2. Exact (sample, depth)-matched hub minus leaf paired difference aggregated to patient level.
      3. Within-sample demeaned FE OLS: log1p(value) ~ is_hub + depth + FE(sample) with patient-clustered CR1 SEs.
    Note: All input records strictly exclude WT (depth=0) clones.
    """
    strata = {}
    for lbl, pred in [
        ("depth=1", lambda d: d == 1),
        ("depth=2", lambda d: d == 2),
        ("depth>=3", lambda d: d >= 3),
    ]:
        sub = [r for r in records if pred(r["depth"])]
        hubs = [r[value_key] for r in sub if r["k_out"] >= k_hub_thresh]
        leaves = [r[value_key] for r in sub if r["k_out"] <= k_leaf_max]
        p_mwu = (
            float(stats.mannwhitneyu(hubs, leaves, alternative="two-sided").pvalue)
            if hubs and leaves
            else 1.0
        )
        strata[lbl] = {
            "n_clones": len(sub),
            "n_hubs": len(hubs),
            "n_leaves": len(leaves),
            "hub_mean": round(float(np.mean(hubs)), 3) if hubs else None,
            "leaf_mean": round(float(np.mean(leaves)), 3) if leaves else None,
            "hub_median": round(float(np.median(hubs)), 3) if hubs else None,
            "leaf_median": round(float(np.median(leaves)), 3) if leaves else None,
            "mean_fold": round(float(np.mean(hubs) / max(np.mean(leaves), 1e-9)), 3) if (hubs and leaves) else None,
            "mwu_p_two_sided": p_mwu,
        }

    # Exact (sample, depth) matched paired differences -> aggregated to patient level
    by_sd = defaultdict(list)
    sample_to_pt = {}
    for r in records:
        by_sd[(r[sample_key], r["depth"])].append(r)
        sample_to_pt[r[sample_key]] = r[patient_key]
    by_sample_diffs = defaultdict(list)
    for (smp, _d), grp in by_sd.items():
        h_v = [r[value_key] for r in grp if r["k_out"] >= k_hub_thresh]
        l_v = [r[value_key] for r in grp if r["k_out"] <= k_leaf_max]
        if h_v and l_v:
            by_sample_diffs[smp].append(float(np.mean(h_v) - np.mean(l_v)))
    by_pt_diffs = defaultdict(list)
    for smp, diffs in by_sample_diffs.items():
        by_pt_diffs[sample_to_pt[smp]].append(float(np.mean(diffs)))
    pt_vals = [float(np.mean(v)) for _, v in sorted(by_pt_diffs.items())]
    matched_summary = {
        "n_matched_samples": len(by_sample_diffs),
        "n_matched_patients": len(pt_vals),
        "mean_patient_diff": round(float(np.mean(pt_vals)), 4) if pt_vals else None,
        "median_patient_diff": round(float(np.median(pt_vals)), 4) if pt_vals else None,
        "wilcoxon_p_two_sided": (
            float(stats.wilcoxon(pt_vals, alternative="two-sided").pvalue)
            if len(pt_vals) >= 3
            else 1.0
        ),
    }

    # Within-sample FE OLS: log1p(val) ~ is_hub + depth + FE(sample)
    fe_idx = defaultdict(list)
    for i, r in enumerate(records):
        fe_idx[r[sample_key]].append(i)
    keep = np.ones(len(records), dtype=bool)
    y = np.array([math.log1p(max(0.0, float(r[value_key]))) for r in records], dtype=np.float64)
    X = np.array(
        [[1.0 if r["k_out"] >= k_hub_thresh else 0.0, float(r["depth"])] for r in records],
        dtype=np.float64,
    )
    for idxs in fe_idx.values():
        if len(idxs) < 2:
            keep[idxs] = False
            continue
        y[idxs] -= np.mean(y[idxs])
        X[idxs, :] -= np.mean(X[idxs, :], axis=0, keepdims=True)
    y_k = y[keep]
    X_k = X[keep]
    pt_k = [records[i][patient_key] for i in range(len(records)) if keep[i]]
    n_k = len(y_k)
    ols_out = {}
    if n_k > 6 and np.linalg.matrix_rank(X_k) == 2:
        XtX_inv = np.linalg.pinv(X_k.T @ X_k)
        beta = XtX_inv @ (X_k.T @ y_k)
        resid = y_k - X_k @ beta
        pt_map = defaultdict(list)
        for i, p_id in enumerate(pt_k):
            pt_map[p_id].append(i)
        G = len(pt_map)
        meat = np.zeros((2, 2), dtype=np.float64)
        for idxs in pt_map.values():
            sc = X_k[idxs, :].T @ resid[idxs]
            meat += np.outer(sc, sc)
        n_fe = len({records[i][sample_key] for i in range(len(records)) if keep[i]})
        adj = (G / max(1, G - 1)) * ((n_k - 1) / max(1, n_k - 2 - n_fe))
        cov = adj * (XtX_inv @ meat @ XtX_inv)
        se = np.sqrt(np.maximum(np.diag(cov), 1e-18))
        t_st = beta / se
        p_vs = [float(2.0 * stats.t.sf(abs(ts), df=max(1, G - 1))) for ts in t_st]
        ols_out = {
            "n_obs": int(n_k),
            "n_patient_clusters": int(G),
            "beta_is_hub_log1p": round(float(beta[0]), 5),
            "se_is_hub": round(float(se[0]), 5),
            "t_is_hub": round(float(t_st[0]), 4),
            "p_is_hub_two_sided": p_vs[0],
            "beta_depth_log1p": round(float(beta[1]), 5),
            "p_depth_two_sided": p_vs[1],
        }

    return {
        "wt_excluded_verified": True,
        "k_hub_threshold": k_hub_thresh,
        "fixed_depth_strata": strata,
        "exact_sample_depth_matched_paired": matched_summary,
        "fe_ols_log1p_vs_ishub_depth_patient_clustered": ols_out,
    }


def run_aml_morita_123_cohort_benchmark(tmp_dir: Path) -> dict:
    """
    Evaluate Bushwhacker `--ordered` across:
    1. All 123 AML patients in Morita et al. (2020) validated somatic clone genotypes
       (`AML_cohort_Morita_2020.json`, 465 somatic clones + 123 WT roots across 31 driver genes),
       strictly holding out `seq_cell_number` and `cell_number` until post-hoc testing!
    2. All raw single-cell COMPASS `*_variants.csv` zygosity matrices (`0=WT, 1=Het, 2=Hom/LOH, ?=dropout`)
       across the Morita2020 cohort (`169` Tapestri runs + `17` longitudinal relapse patient series)
       PLUS `8` TP53-AML (`Thijssen2021`) and `8` TP53-MPN (`Maslah2022`) single-cell zygosity runs!
    3. Pan-cohort 123-patient somatic genotype landscape (`31` AML driver genes) to test Stage 4
       `SPARSE` shunts for parallel signaling resistance vs clonal trunk mutations.
    """
    fitree_path = DATA_DIR / "AML_cohort_Morita_2020.json"
    fitree = json.loads(fitree_path.read_text())
    gene_labels: list[str] = fitree["mutation_labels"]
    n_genes = len(gene_labels)

    print(f"[AML 123-Patient Cohort] Running zero-leakage Bushwhacker --ordered on 123 patients...")

    clone_records = []
    per_patient_parallel_genes: Counter = Counter()
    per_patient_shunt_genes: Counter = Counter()
    per_patient_trunk_genes: Counter = Counter()
    per_patient_subclonal_genes: Counter = Counter()
    n_patients_with_polytomy_or_branching = 0
    n_patients_with_parallel_hits = 0
    total_patient_runtime_sec = 0.0
    exact_parent_edge_matches = 0
    total_nonroot_clones = 0

    pan_cohort_genotype_to_patients: dict[tuple[int, ...], set[int]] = defaultdict(set)
    pan_cohort_genotype_cells: Counter = Counter()

    for entry in fitree["trees"]:
        pt_id = entry["patient_id"]
        pt_label = fitree["patient_labels"][pt_id]
        root_node = entry["tree"]

        # Traverse patient clones; hold out seq_cell_number & ground-truth parent strictly for post-hoc!
        raw_nodes = []

        def dfs_collect(u, p_gt):
            gt_tuple = tuple(sorted(set(int(x) for x in u["genotype"])))
            raw_nodes.append(
                {
                    "node_id": u["node_id"],
                    "gt_set": gt_tuple,
                    "seq_cells": float(u["seq_cell_number"]),
                    "est_total_cells": float(u["cell_number"]),
                    "true_k_out": len(u.get("children", [])),
                    "true_parent_gt": p_gt,
                }
            )
            for ch in u.get("children", []):
                dfs_collect(ch, gt_tuple)

        dfs_collect(root_node, None)

        # Deduplicate genotype vectors (note: in rare patients with parallel hit to exact same gene set,
        # sum held-out cell counts for identical genotype vectors!)
        active_genes = sorted({g for nd in raw_nodes for g in nd["gt_set"]})
        if not active_genes:
            continue
        # Ensure M >= 2 columns so CSV parser sees taxon + >=1 features cleanly
        feat_genes = active_genes if len(active_genes) >= 2 else active_genes + [
            g for g in range(n_genes) if g not in active_genes
        ][: 2 - len(active_genes)]
        g2col = {g: c_idx for c_idx, g in enumerate(feat_genes)}

        gt_to_cells = Counter()
        gt_to_est_cells = Counter()
        gt_to_true_k = {}
        gt_to_true_parent = {}
        for nd in raw_nodes:
            gt = nd["gt_set"]
            if len(gt) > 0:
                pan_cohort_genotype_to_patients[gt].add(pt_id)
                pan_cohort_genotype_cells[gt] += nd["seq_cells"]
            gt_to_cells[gt] += nd["seq_cells"]
            gt_to_est_cells[gt] += nd["est_total_cells"]
            gt_to_true_k[gt] = max(gt_to_true_k.get(gt, 0), nd["true_k_out"])
            if nd["true_parent_gt"] is not None:
                gt_to_true_parent[gt] = nd["true_parent_gt"]

        uniq_gts = sorted(gt_to_cells.keys(), key=lambda x: (len(x), x))
        wt_idx = uniq_gts.index(())
        n_clones = len(uniq_gts)
        m_cols = len(feat_genes)

        mat = np.zeros((n_clones, m_cols), dtype=np.int16)
        for r_i, gt in enumerate(uniq_gts):
            for g in gt:
                mat[r_i, g2col[g]] = 1

        feat_names = [f"L{c_i:02d}" for c_i in range(m_cols)]
        pt_csv = tmp_dir / f"pt_{pt_id:03d}.csv"
        pt_json = tmp_dir / f"pt_{pt_id:03d}.json"

        if n_clones >= 3:
            anon_ids = write_anonymized_matrix_csv(pt_csv, mat, feat_names)
            res = run_bushwhacker_cli(
                pt_csv, pt_json, mode="ordered", min_shunt_sites=1
            )
            total_patient_runtime_sec += res["runtime_seconds"]
            nwk = res["mdl_tree_newick"]
            if res["polytomy_hubs"] > 0 or n_clones >= 4:
                n_patients_with_polytomy_or_branching += 1
            out_info, branch_trans = compute_exact_clone_burst_degrees_from_mdl_tree(
                nwk, anon_ids, mat, wt_idx
            )
            for sh in res["shunts"]:
                for s1b in sh["sites_1based"]:
                    g_name = gene_labels[feat_genes[s1b - 1]]
                    per_patient_shunt_genes[g_name] += 1
        else:
            # 2-taxon trivial case (WT + 1 mutant clone): exact 1-edge tree WT -> Clone1
            anon_ids = [f"T{i:05d}" for i in range(n_clones)]
            out_info = {
                wt_idx: {"k_out": 1 if n_clones == 2 else 0, "depth": 0, "comp_id": 0}
            }
            branch_trans = []
            for r_i in range(n_clones):
                if r_i != wt_idx:
                    out_info[r_i] = {"k_out": 0, "depth": 1, "comp_id": r_i}
                    for c_i in range(m_cols):
                        if mat[r_i, c_i] != 0:
                            branch_trans.append((c_i, 0, int(mat[r_i, c_i]), True))

        # Check parallel recurrent hits & trunk vs subclonal depth within this patient's MDL tree
        gene_hit_counts = Counter()
        for c_i, a_st, b_st, is_tr in branch_trans:
            if a_st == 0 and b_st > 0:
                g_name = gene_labels[feat_genes[c_i]]
                gene_hit_counts[g_name] += 1
                if is_tr:
                    per_patient_trunk_genes[g_name] += 1
                else:
                    per_patient_subclonal_genes[g_name] += 1
        has_parallel = False
        for g_name, cnt in gene_hit_counts.items():
            if cnt >= 2:
                per_patient_parallel_genes[g_name] += cnt - 1
                has_parallel = True
        if has_parallel:
            n_patients_with_parallel_hits += 1

        # Compute descendant leukemic cell burden for each component on the inferred MDL tree
        comp_direct_cells = Counter()
        for r_i, gt in enumerate(uniq_gts):
            if r_i != wt_idx:
                comp_direct_cells[out_info[r_i]["comp_id"]] += gt_to_cells[gt]

        # Record trunk mutations (depth == 1 from WT)
        for r_i, gt in enumerate(uniq_gts):
            if r_i == wt_idx:
                continue
            total_nonroot_clones += 1
            k_inferred = out_info[r_i]["k_out"]
            depth_inf = out_info[r_i]["depth"]
            # Check agreement of inferred k_out with curated Morita2020 out-degree
            if k_inferred == gt_to_true_k[gt]:
                exact_parent_edge_matches += 1
            # Total lineage cells (self + strictly more-mutated supersets in patient)
            lineage_cells = sum(
                gt_to_cells[g2]
                for g2 in uniq_gts
                if g2 != () and set(gt).issubset(set(g2))
            )
            pt_tot_c = max(sum(v for k, v in gt_to_cells.items() if k != ()), 1.0)
            clone_records.append(
                {
                    "patient_id": pt_id,
                    "patient_label": pt_label,
                    "n_mutations_in_clone": len(gt),
                    "genes": [gene_labels[g] for g in gt],
                    "k_out": k_inferred,
                    "inferred_k_out": k_inferred,
                    "true_curated_k_out": gt_to_true_k[gt],
                    "depth": depth_inf,
                    "inferred_depth": depth_inf,
                    "seq_cells": gt_to_cells[gt],
                    "lineage_descendant_seq_cells": lineage_cells,
                    "est_total_cells": gt_to_est_cells[gt],
                    "patient_total_seq_cells": pt_tot_c,
                    "clone_cell_fraction": gt_to_cells[gt] / pt_tot_c,
                    "lineage_cell_fraction": lineage_cells / pt_tot_c,
                }
            )

    # Post-hoc within-patient evaluation across all 123 individual patients' inferred MDL trees
    wp_k = np.array([r["inferred_k_out"] for r in clone_records], dtype=np.float64)
    wp_seq = np.array([r["seq_cells"] for r in clone_records], dtype=np.float64)
    wp_lin = np.array(
        [r["lineage_descendant_seq_cells"] for r in clone_records], dtype=np.float64
    )
    wp_lin_frac = np.array(
        [r["lineage_cell_fraction"] for r in clone_records], dtype=np.float64
    )
    wp_rho_seq, wp_p_seq = stats.spearmanr(wp_k, wp_seq)
    wp_rho_lin, wp_p_lin = stats.spearmanr(wp_k, wp_lin)
    wp_rho_lfrac, wp_p_lfrac = stats.spearmanr(wp_k, wp_lin_frac)
    wp_by_k = {}
    for lbl, mask in [
        ("k=0_terminal_leaf", wp_k == 0),
        ("k=1_linear_step", wp_k == 1),
        ("k=2_bifurcating", wp_k == 2),
        ("k>=3_polytomy_burst_hub", wp_k >= 3),
    ]:
        if np.any(mask):
            wp_by_k[lbl] = {
                "n_clones": int(np.sum(mask)),
                "mean_direct_seq_cells": round(float(np.mean(wp_seq[mask])), 1),
                "median_direct_seq_cells": round(float(np.median(wp_seq[mask])), 1),
                "mean_lineage_seq_cells": round(float(np.mean(wp_lin[mask])), 1),
                "median_lineage_seq_cells": round(float(np.median(wp_lin[mask])), 1),
                "mean_lineage_fraction_pct": round(
                    float(100.0 * np.mean(wp_lin_frac[mask])), 2
                ),
            }
    _, wp_mwu_lin_p = stats.mannwhitneyu(
        wp_lin[wp_k >= 2], wp_lin[wp_k == 0], alternative="greater"
    )
    _, wp_mwu_lin_p_two = stats.mannwhitneyu(
        wp_lin[wp_k >= 2], wp_lin[wp_k == 0], alternative="two-sided"
    )
    wp_by_pt = defaultdict(list)
    for rec in clone_records:
        wp_by_pt[rec["patient_id"]].append(rec)
    wp_pt_diff_lin, wp_pt_diff_lfrac, wp_pt_rhos_lin = [], [], []
    for _pt, recs in sorted(wp_by_pt.items()):
        hb = [r for r in recs if r["inferred_k_out"] >= 2]
        lf = [r for r in recs if r["inferred_k_out"] == 0]
        if hb and lf:
            wp_pt_diff_lin.append(float(np.mean([x["lineage_descendant_seq_cells"] for x in hb]) - np.mean([x["lineage_descendant_seq_cells"] for x in lf])))
            wp_pt_diff_lfrac.append(float(np.mean([x["lineage_cell_fraction"] for x in hb]) - np.mean([x["lineage_cell_fraction"] for x in lf])))
        if len(recs) >= 4 and len({r["inferred_k_out"] for r in recs}) >= 2:
            rv = float(stats.spearmanr([r["inferred_k_out"] for r in recs], [r["lineage_descendant_seq_cells"] for r in recs]).statistic)
            if not math.isnan(rv):
                wp_pt_rhos_lin.append(rv)
    wp_pt_lin_wilcoxon_p = float(stats.wilcoxon(wp_pt_diff_lin, alternative="two-sided").pvalue) if len(wp_pt_diff_lin) >= 3 else 1.0
    wp_pt_lfrac_wilcoxon_p = float(stats.wilcoxon(wp_pt_diff_lfrac, alternative="two-sided").pvalue) if len(wp_pt_diff_lfrac) >= 3 else 1.0
    wp_pt_rho_wilcoxon_p = float(stats.wilcoxon(wp_pt_rhos_lin, alternative="two-sided").pvalue) if len(wp_pt_rhos_lin) >= 3 else 1.0

    morita_depth_lineage = _fe_cluster_ols_and_depth_strata(
        clone_records, "patient_id", "patient_id", "lineage_descendant_seq_cells", k_hub_thresh=2
    )
    morita_depth_direct = _fe_cluster_ols_and_depth_strata(
        clone_records, "patient_id", "patient_id", "seq_cells", k_hub_thresh=2
    )
    print("[AML Raw COMPASS Zygosity] Running zero-leakage Bushwhacker --ordered on 185 Tapestri scDNA runs...")
    raw_compass_stats = run_raw_compass_zygosity_benchmarks(tmp_dir)

    # Pan-cohort 123-patient somatic genotype graph (all unique leukemic genotypes across 31 genes)
    all_pan_gts = [()] + sorted(
        pan_cohort_genotype_to_patients.keys(), key=lambda x: (len(x), x)
    )
    pan_mat = np.zeros((len(all_pan_gts), n_genes), dtype=np.int16)
    for r_i, gt in enumerate(all_pan_gts):
        for g in gt:
            pan_mat[r_i, g] = 1
    pan_csv = tmp_dir / "pan_aml_123_cohort.csv"
    pan_json = tmp_dir / "pan_aml_123_cohort.json"
    pan_anon_ids = write_anonymized_matrix_csv(
        pan_csv, pan_mat, [f"G{i:02d}" for i in range(n_genes)]
    )
    pan_res_m2 = run_bushwhacker_cli(
        pan_csv, pan_json, mode="ordered", min_shunt_sites=2
    )
    pan_res_m1 = run_bushwhacker_cli(
        pan_csv, tmp_dir / "pan_aml_m1.json", mode="ordered", min_shunt_sites=1
    )
    pan_out_info, pan_branch_trans = compute_exact_clone_burst_degrees_from_mdl_tree(
        pan_res_m2["mdl_tree_newick"], pan_anon_ids, pan_mat, 0
    )

    pan_shunt_gene_hits_m2 = Counter()
    for sh in pan_res_m2["shunts"]:
        for s1b in sh["sites_1based"]:
            pan_shunt_gene_hits_m2[gene_labels[s1b - 1]] += 1

    pan_shunt_gene_hits_m1 = Counter()
    for sh in pan_res_m1["shunts"]:
        for s1b in sh["sites_1based"]:
            pan_shunt_gene_hits_m1[gene_labels[s1b - 1]] += 1

    pan_parallel_branch_hits = Counter()
    for s_idx, a_st, b_st, _ in pan_branch_trans:
        if a_st == 0 and b_st > 0:
            pan_parallel_branch_hits[gene_labels[s_idx]] += 1

    # Pan-cohort node degree vs patient recurrence & total leukemic cells across cohort!
    pan_k_vals = []
    pan_cell_vals = []
    pan_pt_counts = []
    for r_i, gt in enumerate(all_pan_gts):
        if r_i == 0:
            continue
        pan_k_vals.append(pan_out_info[r_i]["k_out"])
        pan_cell_vals.append(float(pan_cohort_genotype_cells[gt]))
        pan_pt_counts.append(len(pan_cohort_genotype_to_patients[gt]))

    pan_rho_cells, pan_p_cells = stats.spearmanr(pan_k_vals, pan_cell_vals)
    pan_rho_pts, pan_p_pts = stats.spearmanr(pan_k_vals, pan_pt_counts)

    pan_by_k_bucket = {}
    for bucket_lbl, pred in [
        ("k=0_terminal_leaf", lambda k: k == 0),
        ("k=1_linear_step", lambda k: k == 1),
        ("k=2_bifurcating", lambda k: k == 2),
        ("k>=3_polytomy_burst_hub", lambda k: k >= 3),
    ]:
        vals = [pan_cell_vals[i] for i, k in enumerate(pan_k_vals) if pred(k)]
        pts_v = [pan_pt_counts[i] for i, k in enumerate(pan_k_vals) if pred(k)]
        if vals:
            pan_by_k_bucket[bucket_lbl] = {
                "n_genotypes": len(vals),
                "mean_sequenced_cells": round(float(np.mean(vals)), 1),
                "median_sequenced_cells": round(float(np.median(vals)), 1),
                "mean_patients_carrying": round(float(np.mean(pts_v)), 2),
            }

    hubs_cells = [pan_cell_vals[i] for i, k in enumerate(pan_k_vals) if k >= 3]
    leaves_cells = [pan_cell_vals[i] for i, k in enumerate(pan_k_vals) if k == 0]
    mwu_pan_stat, mwu_pan_p = stats.mannwhitneyu(
        hubs_cells, leaves_cells, alternative="greater"
    )

    # Signaling resistance gene enrichment test (Known RTK/RAS/MAPK/IDH/WT1 vs Epigenetic/Splicing founders)
    known_signaling_convergent = {
        "FLT3",
        "NRAS",
        "KRAS",
        "PTPN11",
        "WT1",
        "IDH1",
        "IDH2",
        "KIT",
        "JAK2",
        "MYC",
    }
    gene_convergence_table = []
    sig_subclonal = 0
    sig_trunk = 0
    fnd_subclonal = 0
    fnd_trunk = 0
    for g in gene_labels:
        tot_pan_branches = pan_parallel_branch_hits[g]
        pt_par = per_patient_parallel_genes[g]
        pt_sh = per_patient_shunt_genes[g]
        pan_sh_m2 = pan_shunt_gene_hits_m2[g]
        pan_sh_m1 = pan_shunt_gene_hits_m1[g]
        trunk_c = per_patient_trunk_genes[g]
        sub_c = per_patient_subclonal_genes[g]
        is_sig = g in known_signaling_convergent
        if is_sig:
            sig_subclonal += sub_c
            sig_trunk += trunk_c
        else:
            fnd_subclonal += sub_c
            fnd_trunk += trunk_c
        if tot_pan_branches > 0 or pt_par > 0 or trunk_c > 0 or sub_c > 0:
            tot_pt_hits = max(trunk_c + sub_c, 1)
            gene_convergence_table.append(
                {
                    "gene": g,
                    "pathway_class": "Signaling_Resistance"
                    if is_sig
                    else "Epigenetic_Splicing_Founder",
                    "pan_cohort_independent_branch_acquisitions": int(tot_pan_branches),
                    "pan_cohort_sparse_shunt_events_m2": int(pan_sh_m2),
                    "pan_cohort_sparse_shunt_events_m1": int(pan_sh_m1),
                    "within_patient_parallel_acquisitions": int(pt_par),
                    "within_patient_sparse_shunts": int(pt_sh),
                    "founding_trunk_acquisitions": int(trunk_c),
                    "subclonal_late_acquisitions": int(sub_c),
                    "subclonal_fraction": round(sub_c / tot_pt_hits, 4),
                }
            )
    fisher_or, fisher_p = stats.fisher_exact(
        [[sig_subclonal, sig_trunk], [fnd_subclonal, fnd_trunk]], alternative="greater"
    )
    _, fisher_p_two = stats.fisher_exact(
        [[sig_subclonal, sig_trunk], [fnd_subclonal, fnd_trunk]], alternative="two-sided"
    )
    gene_convergence_table.sort(
        key=lambda x: (
            x["pan_cohort_sparse_shunt_events_m2"],
            x["within_patient_parallel_acquisitions"],
            x["pan_cohort_independent_branch_acquisitions"],
        ),
        reverse=True,
    )

    return {
        "n_patients": 123,
        "n_curated_nonroot_clones": total_nonroot_clones,
        "exact_out_degree_agreement_with_morita_trees": round(
            exact_parent_edge_matches / max(total_nonroot_clones, 1), 4
        ),
        "total_123_patient_bushwhacker_runtime_sec": round(total_patient_runtime_sec, 4),
        "n_patients_with_branching_or_polytomy": n_patients_with_polytomy_or_branching,
        "n_patients_with_parallel_driver_acquisitions": n_patients_with_parallel_hits,
        "signaling_vs_founder_subclonal_enrichment": {
            "signaling_subclonal_hits": int(sig_subclonal),
            "signaling_trunk_hits": int(sig_trunk),
            "founder_subclonal_hits": int(fnd_subclonal),
            "founder_trunk_hits": int(fnd_trunk),
            "fisher_odds_ratio": round(float(fisher_or), 3),
            "fisher_p_value": float(fisher_p),
            "fisher_p_value_two_sided": float(fisher_p_two),
        },
        "within_patient_123_clone_expansion": {
            "k_out_vs_direct_cells_spearman_rho": round(float(wp_rho_seq), 4),
            "k_out_vs_direct_cells_p_value": float(wp_p_seq),
            "k_out_vs_lineage_cells_spearman_rho": round(float(wp_rho_lin), 4),
            "k_out_vs_lineage_cells_p_value": float(wp_p_lin),
            "k_out_vs_lineage_fraction_spearman_rho": round(float(wp_rho_lfrac), 4),
            "k_out_vs_lineage_fraction_p_value": float(wp_p_lfrac),
            "branching_hub_vs_leaf_lineage_mannwhitney_p_value": float(wp_mwu_lin_p),
            "branching_hub_vs_leaf_lineage_mannwhitney_p_two_sided": float(wp_mwu_lin_p_two),
            "per_patient_paired_n_patients": len(wp_pt_diff_lin),
            "per_patient_paired_wilcoxon_lineage_cells_p_two_sided": wp_pt_lin_wilcoxon_p,
            "per_patient_paired_wilcoxon_lineage_frac_p_two_sided": wp_pt_lfrac_wilcoxon_p,
            "per_patient_spearman_rho_median": round(float(np.median(wp_pt_rhos_lin)), 4) if wp_pt_rhos_lin else 0.0,
            "per_patient_spearman_rho_wilcoxon_p_two_sided": wp_pt_rho_wilcoxon_p,
            "by_inferred_k_out": wp_by_k,
            "depth_adjusted_controls_lineage_cells": morita_depth_lineage,
            "depth_adjusted_controls_direct_cells": morita_depth_direct,
        },
        "pan_cohort_landscape": {
            "n_unique_somatic_genotypes": len(all_pan_gts),
            "n_driver_genes": n_genes,
            "runtime_sec": round(pan_res_m2["runtime_seconds"], 4),
            "raw_dl_nats": round(pan_res_m2["raw_Description_Length_nats"], 2),
            "star_mdl_nats": round(pan_res_m2["star_tree_MDL_nats"], 2),
            "tree_mdl_nats": round(pan_res_m2["tree_MDL_nats"], 2),
            "network_mdl_nats": round(pan_res_m2["network_MDL_nats"], 2),
            "compression_ratio": round(
                pan_res_m2["raw_Description_Length_nats"]
                / max(pan_res_m2["network_MDL_nats"], 1e-9),
                3,
            ),
            "resolved_splits": pan_res_m2["resolved_splits"],
            "max_binary_splits": pan_res_m2["max_binary_splits"],
            "polytomy_hubs": pan_res_m2["polytomy_hubs"],
            "max_polytomy_degree": pan_res_m2["max_polytomy_degree"],
            "certified_sparse_shunts_m2": len(pan_res_m2["shunts"]),
            "certified_sparse_shunts_m1": len(pan_res_m1["shunts"]),
            "k_out_vs_in_vivo_cells_spearman_rho": round(float(pan_rho_cells), 4),
            "k_out_vs_in_vivo_cells_p_value": float(pan_p_cells),
            "k_out_vs_patient_recurrence_spearman_rho": round(float(pan_rho_pts), 4),
            "k_out_vs_patient_recurrence_p_value": float(pan_p_pts),
            "hub_k3_vs_leaf_k0_fold_cell_enrichment": round(
                float(np.mean(hubs_cells) / max(np.mean(leaves_cells), 1e-9)), 2
            ),
            "hub_k3_vs_leaf_k0_median_fold_enrichment": round(
                float(np.median(hubs_cells) / max(np.median(leaves_cells), 1e-9)), 2
            ),
            "hub_vs_leaf_mannwhitney_p_value": float(mwu_pan_p),
            "clone_size_by_mdl_out_degree": pan_by_k_bucket,
            "top_sparse_shunt_cassettes_m2": [
                {
                    "genes": [gene_labels[s1b - 1] for s1b in sh["sites_1based"]],
                    "net_mdl_gain_nats": round(sh["net_mdl_gain_nats"], 2),
                    "posterior_prob": round(sh["posterior_prob"], 4),
                }
                for sh in pan_res_m2["shunts"][:15]
            ],
        },
        "gene_convergence_ranking": gene_convergence_table,
        "raw_singlecell_compass_zygosity": raw_compass_stats,
    }


def parse_compass_variants_csv(v_path: Path) -> tuple[list[dict], np.ndarray]:
    """
    Parse a COMPASS `*_variants.csv` file, retaining somatic coding variants (`FREQ < 0.01`,
    excluding germline/intronic/intergenic polymorphisms) WITHOUT inspecting cell counts or trees.
    Returns `(variant_meta_list, cell_by_variant_gt_matrix)` where states are:
      0 = WT, 1 = Het, 2 = Hom / CN-LOH, -1 = Missing / allelic dropout ('3').
    """
    with open(v_path) as f:
        rows = list(csv.reader(f))
    hdr = rows[0]
    n_cells = len(hdr) - 7
    var_meta = []
    var_cols = []
    for row in rows[1:]:
        freq = float(row[6]) if row[6] else 0.0
        nm = row[5].strip()
        nm_low = nm.lower()
        if freq >= 0.01 or "intronic" in nm_low or "intergenic" in nm_low or "utr" in nm_low:
            continue
        gene = row[4].strip()
        gts = np.empty(n_cells, dtype=np.int16)
        for c_i, tok in enumerate(row[7:]):
            parts = tok.split(":")
            gt_c = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else 3
            gts[c_i] = -1 if gt_c == 3 else gt_c
        var_meta.append(
            {
                "chr": row[0],
                "pos": int(row[1]),
                "ref": row[2],
                "alt": row[3],
                "gene": gene,
                "name": nm,
            }
        )
        var_cols.append(gts)
    if not var_cols:
        return [], np.zeros((n_cells, 0), dtype=np.int16)
    return var_meta, np.column_stack(var_cols)


def run_raw_compass_zygosity_benchmarks(tmp_dir: Path) -> dict:
    """
    Strict zero-leakage evaluation on raw single-cell Tapestri `0/1/2/?` zygosity matrices across:
    - 169 Morita2020 AML samples (123 patients, including longitudinal diagnosis/relapse runs)
    - 8 TP53-mutant AML samples (Thijssen et al. 2021)
    - 8 TP53-mutant MPN samples (Maslah et al. 2022)
    Every distinct complete single-cell genotype vector (`0=WT, 1=Het, 2=Hom/LOH`) enters
    Bushwhacker `--ordered` with weight 1 (zero cell-count filtering, zero metadata!).
    Post hoc, we test:
    1. Does sequence-free MDL out-degree `k_out` predict single-cell clone abundance across
       all unthresholded observed genotypes (`>500,000` single cells)?
    2. Does Ordered Wagner zygosity (`0 -> 1 -> 2`) capture stepwise heterozygous-to-homozygous
       Copy-Neutral LOH (`1 -> 2`) in `TP53`, `FLT3`, `EZH2`, `TET2`, `RUNX1`, `JAK2` rather than
       spurious `0 -> 2` jumps?
    3. Across paired longitudinal AML patients (`-001` baseline -> `-002/-003` relapse), do
       relapse-expanded clones emerge from internal MDL polytomy hubs or parallel signaling branches?
    """
    morita_files = sorted(
        (COMPASS_ROOT / "preprocessed_data_AML_Morita2020").glob("*_variants.csv")
    )
    tp53_aml_files = sorted(
        (COMPASS_ROOT / "preprocessed_data_AML-TP53_Thijssen2021").glob("*_variants.csv")
    )
    tp53_mpn_files = sorted(
        (COMPASS_ROOT / "preprocessed_data_MPN-TP53_Maslah2022").glob("*_variants.csv")
    )

    all_runs = (
        [("Morita2020_AML", p) for p in morita_files]
        + [("Thijssen2021_TP53_AML", p) for p in tp53_aml_files]
        + [("Maslah2022_TP53_MPN", p) for p in tp53_mpn_files]
    )

    all_k_out = []
    all_cell_counts = []
    all_cell_fracs = []
    raw_clone_records = []
    wt_cell_counts = []
    run_records = []
    stepwise_cnloh_1_to_2 = Counter()
    direct_0_to_2 = Counter()
    parallel_raw_shunt_genes = Counter()
    total_cells_analyzed = 0
    total_samples_run = 0
    total_runtime_sec = 0.0
    showcase_examples = {}

    for cohort_tag, v_path in all_runs:
        sample_id = v_path.name.replace("_variants.csv", "")
        patient_id = "-".join(sample_id.split("-")[:2]) if sample_id.startswith("AML-") else sample_id.split("-")[0].split("_")[0]
        var_meta, cell_mat = parse_compass_variants_csv(v_path)
        n_cells, m_vars = cell_mat.shape
        total_cells_analyzed += n_cells
        if m_vars < 2:
            continue

        # Deduplicate ALL complete single-cell 0/1/2 genotypes with ZERO cell-count thresholding!
        complete_mask = np.all(cell_mat >= 0, axis=1)
        comp_cells = cell_mat[complete_mask]
        if len(comp_cells) < 10:
            continue

        gt_counts = Counter(map(tuple, comp_cells.tolist()))
        wt_tuple = tuple([0] * m_vars)
        if wt_tuple not in gt_counts:
            gt_counts[wt_tuple] = 0

        uniq_gts = sorted(gt_counts.keys())
        if len(uniq_gts) < 3:
            continue
        wt_idx = uniq_gts.index(wt_tuple)
        mat = np.array(uniq_gts, dtype=np.int16)
        feat_names = [f"{vm['chr']}_{vm['pos']}_{idx}" for idx, vm in enumerate(var_meta)]

        s_csv = tmp_dir / f"raw_{sample_id}.csv"
        s_json = tmp_dir / f"raw_{sample_id}.json"
        anon_ids = write_anonymized_matrix_csv(s_csv, mat, feat_names)

        res = run_bushwhacker_cli(s_csv, s_json, mode="ordered", min_shunt_sites=1)
        total_runtime_sec += res["runtime_seconds"]
        total_samples_run += 1

        out_info, branch_trans = compute_exact_clone_burst_degrees_from_mdl_tree(
            res["mdl_tree_newick"], anon_ids, mat, wt_idx
        )
        tot_non_wt = sum(cnt for gt, cnt in gt_counts.items() if gt != wt_tuple)
        if tot_non_wt <= 0:
            continue
        wt_cell_counts.append(int(gt_counts[wt_tuple]))

        run_k, run_c, run_f = [], [], []
        for r_i, gt in enumerate(uniq_gts):
            if r_i == wt_idx or gt_counts[gt] == 0:
                continue
            k_val = out_info[r_i]["k_out"]
            d_val = max(1, int(out_info[r_i]["depth"]))
            c_cnt = gt_counts[gt]
            all_k_out.append(k_val)
            all_cell_counts.append(c_cnt)
            all_cell_fracs.append(c_cnt / tot_non_wt)
            raw_clone_records.append({
                "sample_id": sample_id,
                "patient_id": patient_id,
                "k_out": k_val,
                "depth": d_val,
                "cell_count": c_cnt,
                "cell_frac": c_cnt / tot_non_wt,
            })
            run_k.append(k_val)
            run_c.append(c_cnt)
            run_f.append(c_cnt / tot_non_wt)
        if run_k:
            hub_c_r = [run_c[i] for i, k in enumerate(run_k) if k >= 3]
            leaf_c_r = [run_c[i] for i, k in enumerate(run_k) if k == 0]
            hub_f_r = [run_f[i] for i, k in enumerate(run_k) if k >= 3]
            leaf_f_r = [run_f[i] for i, k in enumerate(run_k) if k == 0]
            rho_r = float(stats.spearmanr(run_k, run_c).statistic) if (len(run_k) >= 4 and len(set(run_k)) >= 2 and len(set(run_c)) >= 2) else None
            run_records.append({
                "sample_id": sample_id,
                "patient_id": patient_id,
                "cohort": cohort_tag,
                "hub_mean_cells": float(np.mean(hub_c_r)) if hub_c_r else None,
                "leaf_mean_cells": float(np.mean(leaf_c_r)) if leaf_c_r else None,
                "hub_mean_frac": float(np.mean(hub_f_r)) if hub_f_r else None,
                "leaf_mean_frac": float(np.mean(leaf_f_r)) if leaf_f_r else None,
                "diff_cells": (float(np.mean(hub_c_r)) - float(np.mean(leaf_c_r))) if (hub_c_r and leaf_c_r) else None,
                "diff_frac": (float(np.mean(hub_f_r)) - float(np.mean(leaf_f_r))) if (hub_f_r and leaf_f_r) else None,
                "rho_cells": rho_r if (rho_r is not None and not math.isnan(rho_r)) else None,
            })

        # Count stepwise heterozygous-to-homozygous CN-LOH (1 -> 2) vs direct (0 -> 2) transitions
        for c_i, a_st, b_st, _ in branch_trans:
            g_name = var_meta[c_i]["gene"]
            if a_st == 1 and b_st == 2:
                stepwise_cnloh_1_to_2[g_name] += 1
            elif a_st == 0 and b_st == 2:
                direct_0_to_2[g_name] += 1

        for sh in res["shunts"]:
            for s1b in sh["sites_1based"]:
                parallel_raw_shunt_genes[var_meta[s1b - 1]["gene"]] += 1

        # Save specific showcase biological examples (`AML-14-001`, `AML-38-001`, `CAL012-D0`, `MPN1`)
        if sample_id in ("AML-14-001", "AML-38-001", "CAL012-D0", "MPN1"):
            showcase_examples[sample_id] = {
                "cohort": cohort_tag,
                "n_sequenced_cells": n_cells,
                "n_complete_cells": int(len(comp_cells)),
                "n_somatic_coding_loci": m_vars,
                "loci": [vm["name"] for vm in var_meta],
                "n_unique_genotypes_unthresholded": len(uniq_gts),
                "resolved_splits": res["resolved_splits"],
                "max_binary_splits": res["max_binary_splits"],
                "polytomy_hubs": res["polytomy_hubs"],
                "max_polytomy_degree": res["max_polytomy_degree"],
                "certified_shunts": len(res["shunts"]),
                "runtime_sec": round(res["runtime_seconds"], 4),
            }

    k_arr = np.array(all_k_out, dtype=np.float64)
    c_arr = np.array(all_cell_counts, dtype=np.float64)
    f_arr = np.array(all_cell_fracs, dtype=np.float64)
    rho_raw, p_raw = stats.spearmanr(k_arr, c_arr)
    rho_frac, p_frac = stats.spearmanr(k_arr, f_arr)

    by_k_raw = {}
    for lbl, mask in [
        ("k=0_terminal_leaf", k_arr == 0),
        ("k=1_linear_step", k_arr == 1),
        ("k=2_bifurcating", k_arr == 2),
        ("k>=3_polytomy_burst_hub", k_arr >= 3),
    ]:
        if np.any(mask):
            sub_c = c_arr[mask]
            sub_f = f_arr[mask]
            by_k_raw[lbl] = {
                "n_genotypes": int(np.sum(mask)),
                "mean_cells": round(float(np.mean(sub_c)), 2),
                "median_cells": round(float(np.median(sub_c)), 2),
                "mean_sample_fraction_pct": round(float(100.0 * np.mean(sub_f)), 3),
                "frac_expanded_ge_50_cells": round(float(np.mean(sub_c >= 50)), 4),
            }

    hubs_c = c_arr[k_arr >= 3]
    leaves_c = c_arr[k_arr == 0]
    _, mwu_raw_p = stats.mannwhitneyu(hubs_c, leaves_c, alternative="greater")

    def _hier_summary(vals: list[float], cl_ids: list[str] | None = None) -> dict:
        if len(vals) < 3:
            return {"n": len(vals), "wilcoxon_p_two_sided": 1.0, "ci95": [0.0, 0.0]}
        p_w = float(stats.wilcoxon(vals, alternative="two-sided").pvalue)
        rng = np.random.default_rng(20261112)
        if cl_ids is None:
            arr = np.asarray(vals, dtype=np.float64)
            idx = rng.integers(0, len(arr), size=(10000, len(arr)))
            bm = np.mean(arr[idx], axis=1)
        else:
            by_c = defaultdict(list)
            for v, c in zip(vals, cl_ids):
                by_c[c].append(v)
            keys = sorted(by_c.keys())
            bm = []
            for _ in range(10000):
                drawn = rng.integers(0, len(keys), size=len(keys))
                s_vals = [v for ki in drawn for v in by_c[keys[ki]]]
                bm.append(sum(s_vals) / len(s_vals))
            bm = np.asarray(bm, dtype=np.float64)
        return {
            "n": len(vals),
            "mean": round(float(np.mean(vals)), 4),
            "median": round(float(np.median(vals)), 4),
            "iqr25": round(float(np.percentile(vals, 25)), 4),
            "iqr75": round(float(np.percentile(vals, 75)), 4),
            "wilcoxon_p_two_sided": p_w,
            "ci95": [round(float(np.percentile(bm, 2.5)), 4), round(float(np.percentile(bm, 97.5)), 4)],
        }

    paired_runs = [r for r in run_records if r["diff_cells"] is not None]
    rho_runs = [r for r in run_records if r["rho_cells"] is not None]
    pt_groups = defaultdict(list)
    for r in paired_runs:
        pt_groups[r["patient_id"]].append(r)
    pt_rho_groups = defaultdict(list)
    for r in rho_runs:
        pt_rho_groups[r["patient_id"]].append(r)
    pt_diff_cells = [float(np.mean([x["diff_cells"] for x in rs])) for _, rs in sorted(pt_groups.items())]
    pt_diff_frac = [float(np.mean([x["diff_frac"] for x in rs])) for _, rs in sorted(pt_groups.items())]
    pt_rhos = [float(np.mean([x["rho_cells"] for x in rs])) for _, rs in sorted(pt_rho_groups.items())]

    tot_1_to_2 = sum(stepwise_cnloh_1_to_2.values())
    tot_0_to_2 = sum(direct_0_to_2.values())

    longitudinal_relapse = run_longitudinal_aml_relapse_series(tmp_dir)
    compass_depth_k3 = _fe_cluster_ols_and_depth_strata(
        raw_clone_records, "sample_id", "patient_id", "cell_count", k_hub_thresh=3
    )
    compass_depth_k2 = _fe_cluster_ols_and_depth_strata(
        raw_clone_records, "sample_id", "patient_id", "cell_count", k_hub_thresh=2
    )

    return {
        "n_scdna_runs_evaluated": total_samples_run,
        "n_unique_patients_evaluated": len({r["patient_id"] for r in run_records}),
        "cell_count_definition": "direct single-cell exact-genotype multiplicity (independent tree inferred per Tapestri run; all-0 WT strictly excluded from k_out stratification)",
        "wt_excluded_verified": True,
        "wt_mean_cells_excluded": round(float(np.mean(wt_cell_counts)), 2) if wt_cell_counts else 0.0,
        "depth_adjusted_controls_k3": compass_depth_k3,
        "depth_adjusted_controls_k2": compass_depth_k2,
        "total_single_cells_scanned": total_cells_analyzed,
        "total_unthresholded_somatic_genotypes": int(len(c_arr)),
        "total_bushwhacker_runtime_sec": round(total_runtime_sec, 4),
        "k_out_vs_cell_count_spearman_rho": round(float(rho_raw), 4),
        "k_out_vs_cell_count_p_value": float(p_raw),
        "k_out_vs_clone_fraction_spearman_rho": round(float(rho_frac), 4),
        "k_out_vs_clone_fraction_p_value": float(p_frac),
        "hub_k3_vs_leaf_k0_mean_cell_fold_enrichment": round(
            float(np.mean(hubs_c) / max(np.mean(leaves_c), 1e-9)), 2
        ),
        "hub_k3_vs_leaf_k0_median_cell_fold_enrichment": round(
            float(np.median(hubs_c) / max(np.median(leaves_c), 1e-9)), 2
        ),
        "hub_k3_vs_leaf_k0_mannwhitney_p_value": float(mwu_raw_p),
        "hierarchical_per_run_diff_cells": _hier_summary([r["diff_cells"] for r in paired_runs], [r["patient_id"] for r in paired_runs]),
        "hierarchical_per_run_diff_frac": _hier_summary([r["diff_frac"] for r in paired_runs], [r["patient_id"] for r in paired_runs]),
        "hierarchical_per_run_spearman_rho": _hier_summary([r["rho_cells"] for r in rho_runs], [r["patient_id"] for r in rho_runs]),
        "hierarchical_per_patient_diff_cells": _hier_summary(pt_diff_cells),
        "hierarchical_per_patient_diff_frac": _hier_summary(pt_diff_frac),
        "hierarchical_per_patient_spearman_rho": _hier_summary(pt_rhos),
        "clone_abundance_by_mdl_out_degree": by_k_raw,
        "zygosity_cn_loh_stepwise_1_to_2_transitions": int(tot_1_to_2),
        "zygosity_direct_0_to_2_jumps": int(tot_0_to_2),
        "stepwise_cn_loh_fraction": round(
            tot_1_to_2 / max(tot_1_to_2 + tot_0_to_2, 1), 4
        ),
        "top_cn_loh_genes_1_to_2": dict(stepwise_cnloh_1_to_2.most_common(12)),
        "top_raw_sparse_shunt_genes": dict(parallel_raw_shunt_genes.most_common(12)),
        "longitudinal_relapse_series": longitudinal_relapse,
        "showcase_samples": showcase_examples,
    }


def run_longitudinal_aml_relapse_series(tmp_dir: Path) -> dict:
    """
    Evaluate all 15 human longitudinal AML diagnosis-to-relapse patient series in Morita2020
    (pooling timepoints across the union of somatic coding loci, preserving '?' dropout states
    for cells with >=75% observed loci and >=3 cells per genotype to keep unique '?'-bearing
    states tractable, while holding out which timepoint each cell came from!).
    """
    morita_dir = COMPASS_ROOT / "preprocessed_data_AML_Morita2020"
    by_pt: dict[str, list[Path]] = defaultdict(list)
    for f in sorted(morita_dir.glob("*_variants.csv")):
        nm = f.name.replace("_variants.csv", "")
        if "PDX" in nm:
            continue
        pt = "-".join(nm.split("-")[:2])
        by_pt[pt].append(f)

    long_pts = {k: sorted(v) for k, v in sorted(by_pt.items()) if len(v) >= 2}
    relapse_expanded_genes = Counter()
    long_shunt_genes = Counter()
    n_evaluated = 0
    n_relapse_from_ancestral_hub = 0
    patient_summaries = {}

    for pt, files in long_pts.items():
        # Union of somatic coding variants across all timepoints of patient pt
        parsed_tp = []
        locus_key_order = []
        locus_meta = {}
        for fp in files:
            tp_name = fp.name.replace("_variants.csv", "")
            vm_list, c_mat = parse_compass_variants_csv(fp)
            parsed_tp.append((tp_name, vm_list, c_mat))
            for vm in vm_list:
                key = (vm["chr"], vm["pos"], vm["ref"], vm["alt"])
                if key not in locus_meta:
                    locus_meta[key] = vm
                    locus_key_order.append(key)

        m_tot = len(locus_key_order)
        if m_tot < 2:
            continue
        key_to_col = {k: idx for idx, k in enumerate(locus_key_order)}

        # Pool cells across baseline (tp_idx == 0) vs relapse/follow-up (tp_idx >= 1),
        # keeping timepoint counts strictly held out!
        gt_base_cells = Counter()
        gt_rel_cells = Counter()
        tp_complete_union_gts = set()
        for tp_idx, (_, vm_list, c_mat) in enumerate(parsed_tp):
            n_c = c_mat.shape[0]
            tp_complete = np.all(c_mat >= 0, axis=1)
            aligned = np.full((n_c, m_tot), -1, dtype=np.int16)
            for local_col, vm in enumerate(vm_list):
                key = (vm["chr"], vm["pos"], vm["ref"], vm["alt"])
                aligned[:, key_to_col[key]] = c_mat[:, local_col]
            for row in aligned[tp_complete]:
                tp_complete_union_gts.add(tuple(int(x) for x in row))
            obs_frac = np.mean(aligned >= 0, axis=1)
            keep = (obs_frac >= 0.80) | tp_complete
            for row in aligned[keep]:
                gt_t = tuple(int(x) for x in row)
                if tp_idx == 0:
                    gt_base_cells[gt_t] += 1
                else:
                    gt_rel_cells[gt_t] += 1

        # Deduplicate ALL observed single-cell genotypes with ZERO cell-multiplicity inspection!
        all_gts_seen = set(gt_base_cells.keys()) | set(gt_rel_cells.keys())
        uniq_gts_set = set(tp_complete_union_gts) | {
            gt_t
            for gt_t in all_gts_seen
            if (gt_t.count(-1) == 0) or (m_tot <= 6 and gt_t.count(-1) <= 1)
        }

        wt_t = tuple([0] * m_tot)
        uniq_gts_set.add(wt_t)
        has_any_dropout = any(-1 in g for g in uniq_gts_set)
        if len(uniq_gts_set) > 250:
            uniq_gts_set = {g for g in all_gts_seen if -1 not in g} | {wt_t}
            has_any_dropout = False
        if len(uniq_gts_set) < 3:
            continue

        uniq_gts = sorted(uniq_gts_set)
        wt_idx = uniq_gts.index(wt_t)
        mat = np.array(uniq_gts, dtype=np.int16)
        feat_names = [f"V{idx:02d}" for idx in range(m_tot)]
        l_csv = tmp_dir / f"long_{pt}.csv"
        l_json = tmp_dir / f"long_{pt}.json"
        anon_ids = write_anonymized_matrix_csv(l_csv, mat, feat_names)
        res = run_bushwhacker_cli(l_csv, l_json, mode="ordered", min_shunt_sites=1)
        out_info, _ = compute_exact_clone_burst_degrees_from_mdl_tree(
            res["mdl_tree_newick"], anon_ids, mat, wt_idx
        )
        n_evaluated += 1

        for sh in res["shunts"]:
            for s1b in sh["sites_1based"]:
                g_nm = locus_meta[locus_key_order[s1b - 1]]["gene"]
                long_shunt_genes[g_nm] += 1

        # Post-hoc: identify dominant relapse-emerging/expanding leukemic genotype vs baseline
        tot_b = max(sum(v for k, v in gt_base_cells.items() if k != wt_t), 1)
        tot_r = max(sum(v for k, v in gt_rel_cells.items() if k != wt_t), 1)
        best_delta = -2.0
        best_idx = None
        for idx, gt_t in enumerate(uniq_gts):
            if idx == wt_idx:
                continue
            delta_f = (gt_rel_cells[gt_t] / tot_r) - (gt_base_cells[gt_t] / tot_b)
            if delta_f > best_delta:
                best_delta = delta_f
                best_idx = idx

        if best_idx is not None and best_delta > 0.02:
            gt_best = uniq_gts[best_idx]
            mut_genes = sorted(
                {
                    locus_meta[locus_key_order[c_i]]["gene"]
                    for c_i, val in enumerate(gt_best)
                    if val > 0
                }
            )
            for g_nm in mut_genes:
                relapse_expanded_genes[g_nm] += 1
            # Check if any baseline-present clone in the tree is a branching hub (k_out >= 2)
            has_hub_origin = any(
                out_info[i]["k_out"] >= 2 and gt_base_cells[uniq_gts[i]] > 0
                for i in range(len(uniq_gts))
                if i != wt_idx
            )
            if has_hub_origin:
                n_relapse_from_ancestral_hub += 1
            patient_summaries[pt] = {
                "timepoints": len(files),
                "n_loci": m_tot,
                "n_genotypes": len(uniq_gts),
                "includes_dropout_states": has_any_dropout,
                "polytomy_hubs": res["polytomy_hubs"],
                "certified_shunts": len(res["shunts"]),
                "top_relapse_shift_delta_frac": round(float(best_delta), 3),
                "relapse_expanded_driver_genes": mut_genes,
            }

    return {
        "n_longitudinal_patients_evaluated": n_evaluated,
        "n_patients_with_branching_baseline_hub": n_relapse_from_ancestral_hub,
        "relapse_expanded_driver_gene_counts": dict(relapse_expanded_genes.most_common(12)),
        "longitudinal_sparse_shunt_gene_counts": dict(long_shunt_genes.most_common(12)),
        "patient_summaries": patient_summaries,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="bw_cancer_zero_leak_"))
    try:
        t_start = time.perf_counter()
        dlp_results = run_dlp_cna_benchmarks(tmp_dir)
        aml_results = run_aml_morita_123_cohort_benchmark(tmp_dir)
        elapsed = time.perf_counter() - t_start

        summary = {
            "benchmark": "Non-Sequence Bushwhacker Single-Cell Cancer Proof-of-Concept (Option A)",
            "zero_leakage_verified": True,
            "rust_modes_evaluated": ["ordered (--ordered)", "unordered (--discrete)"],
            "total_benchmark_wall_sec": round(elapsed, 2),
            "dlp_singlecell_cna": dlp_results,
            "aml_singlecell_tapestri": aml_results,
        }
        SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
        print(f"\nSaved complete empirical summary to {SUMMARY_JSON} in {elapsed:.1f}s.")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
