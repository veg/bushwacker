#!/usr/bin/env python3
"""
Full-Scale Benchmark Worker on the Abadi et al. (2019, Nature Communications) &
Gerth (2019, bioRxiv 849018) 7,200-Dataset Corpus (OSF.IO/T3PF2).

For each dataset ID (1..7200):
  1. EMPIRICAL RE-ANALYSIS (`empirical/dataset_{id}.phy`):
     - Runs IQ-TREE 2 ModelFinder (BIC across the 24 standard DNA models),
       IQ-TREE 2 fixed GTR+I+G4, IQ-TREE 2 fixed GTR+I+G4 --polytomy,
       IQ-TREE 2 fixed JC, FastTree 2, BioNJ, Bushwhacker (MDL), and Bushwhacker (Binary).
     - Extracts Gerth (2019) metrics: N, L, N*L, BIC best model, total tree length (TBL),
       low-stemminess fraction (internal branches < 0.1% TBL, and < 1 expected mutation),
       pairwise RF / nRF, and conflict analysis showing how often Bushwhacker (MDL)
       contracts the discordant branches where GTR+I+G4 and BIC disagree.
  2. MATCHED GROUND-TRUTH SIMULATION (`sim_bic`, matching c0 on empirical BIC tree & model):
     - Simulates an alignment of length L on the empirical BIC tree & fitted model parameters
       via `iqtree2 --alisim`, where the exact ground-truth tree and branch lengths are known.
     - Evaluates Bushwhacker (MDL), Bushwhacker_Binary, IQ-TREE 2 (BIC), IQ-TREE 2 (GTR+I+G4),
       IQ-TREE 2 (GTR+I+G4 --polytomy), IQ-TREE 2 (JC), FastTree 2, and BioNJ on:
       RF_bin, nRF_bin, Realized RF_ge1, FP, FN, Precision, Recall, and runtime.
  3. ABADI ET AL. (2019) COMPLEX SIMULATED DATASETS (`c1`, `c2`, `c3` from OSF.IO/T3PF2):
     - Reconstructs Abadi et al.'s exact deterministic generating tree (`phyml -m JC -o lr` on
       the empirical alignment, per `simulate_c1.py`, `simulate_c2.py`, `simulate_c3.py`) and
       evaluates Bushwhacker (MDL), Bushwhacker_Binary, FastTree 2, IQ-TREE 2 (GTR+I+G4),
       IQ-TREE 2 (GTR+I+G4 --polytomy), and IQ-TREE 2 (JC) on `c1`, `c2`, and `c3`.
"""


import sys as _sys
from pathlib import Path as _Path
if str(_Path(__file__).resolve().parent) not in _sys.path:
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
from _env_paths import REPO_ROOT, DATA_DIR, SPECIMENS_DIR, ALIGNMENTS_DIR, SINGLE_CELL_DIR, RESULTS_DIR, WORK_ROOT, EXTERNAL_DIR, resolve_bin, ASTRAL_JAR

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from io import StringIO
from pathlib import Path
from Bio import Phylo


def phy_to_fasta_and_clean_phy(phy_path: Path, fa_out: Path, clean_phy_out: Path):
    """Parse sequential or interleaved PHYLIP and write clean FASTA and sequential PHYLIP."""
    raw_lines = [l.rstrip("\r\n") for l in phy_path.read_text(errors="ignore").splitlines() if l.strip()]
    header = raw_lines[0].split()
    ntaxa, nchars = int(header[0]), int(header[1])
    seqs = []
    cur_idx = 0
    for line in raw_lines[1:]:
        s = line.strip()
        if not s:
            continue
        if len(seqs) < ntaxa:
            parts = s.split(None, 1)
            name = parts[0]
            seq = re.sub(r"\s+", "", parts[1]) if len(parts) > 1 else ""
            seqs.append([name, seq])
        else:
            chunk = re.sub(r"\s+", "", s)
            seqs[cur_idx][1] += chunk
            cur_idx = (cur_idx + 1) % ntaxa

    with open(fa_out, "w") as f:
        for name, seq in seqs:
            f.write(f">{name}\n{seq}\n")

    with open(clean_phy_out, "w") as f:
        f.write(f" {ntaxa} {nchars}\n")
        for name, seq in seqs:
            f.write(f"{name}  {seq}\n")

    return ntaxa, nchars, [name for name, _ in seqs]


def get_unrooted_splits(newick_str_or_path, is_path=True, L=None):
    """
    Extract unrooted non-trivial splits as canonical frozensets of leaf names,
    along with branch lengths and internal branch statistics.
    """
    try:
        if is_path:
            txt = Path(newick_str_or_path).read_text().strip()
        else:
            txt = newick_str_or_path.strip()
        if not txt:
            return None
        tree = Phylo.read(StringIO(txt), "newick")
    except Exception:
        return None

    terminals = [t.name for t in tree.get_terminals()]
    all_leaves = frozenset(terminals)
    n_leaves = len(all_leaves)
    if n_leaves < 4:
        return {
            "splits": {},
            "n_leaves": n_leaves,
            "tbl": 0.0,
            "n_internal": 0,
            "n_short_001": 0,
            "n_sub1_mut": 0,
        }

    # Postorder leaf sets
    clade_leaves = {}
    all_branch_lengths = []
    for clade in tree.find_clades(order="postorder"):
        if clade.is_terminal():
            clade_leaves[id(clade)] = frozenset([clade.name])
            if clade.branch_length is not None:
                all_branch_lengths.append(float(clade.branch_length))
        else:
            s = frozenset().union(*(clade_leaves[id(c)] for c in clade.clades))
            clade_leaves[id(clade)] = s
            if clade != tree.root and clade.branch_length is not None:
                all_branch_lengths.append(float(clade.branch_length))

    splits = {}
    for clade in tree.find_clades(order="postorder"):
        if clade == tree.root or clade.is_terminal():
            continue
        s = clade_leaves[id(clade)]
        comp = all_leaves - s
        if len(s) <= 1 or len(comp) <= 1:
            continue
        canon = s if len(s) < len(comp) or (len(s) == len(comp) and min(s) < min(comp)) else comp
        bl = float(clade.branch_length) if clade.branch_length is not None else 0.0
        splits[canon] = splits.get(canon, 0.0) + bl

    tbl = sum(all_branch_lengths)
    internal_bls = list(splits.values())
    n_internal = len(internal_bls)
    thresh_001 = 0.001 * tbl if tbl > 0 else 0.0
    n_short_001 = sum(1 for b in internal_bls if b < thresh_001)
    n_sub1_mut = sum(1 for b in internal_bls if (b * L < 1.0)) if L is not None else 0

    return {
        "splits": splits,
        "n_leaves": n_leaves,
        "tbl": tbl,
        "n_internal": n_internal,
        "n_short_001": n_short_001,
        "n_sub1_mut": n_sub1_mut,
    }


def evaluate_against_ref(ref_info, est_info, L):
    """Compute exact TP, FP, FN, RF_bin, nRF_bin, Realized RF_ge1, Precision, Recall."""
    if ref_info is None or est_info is None:
        return None
    ref_splits = set(ref_info["splits"].keys())
    est_splits = set(est_info["splits"].keys())
    n_true = len(ref_splits)
    n_est = len(est_splits)
    if n_true == 0:
        return None

    tp = len(ref_splits & est_splits)
    fp = len(est_splits - ref_splits)
    fn = len(ref_splits - est_splits)
    rf_bin = fp + fn
    nrf_bin = rf_bin / (2.0 * n_true)
    prec = tp / n_est if n_est > 0 else 1.0
    rec = tp / n_true

    # Realized / Active reference tree (true branches with expected mutations l*L >= 1.0)
    active_ref = {s for s, bl in ref_info["splits"].items() if bl * L >= 1.0}
    tp_ge1 = len(active_ref & est_splits)
    fp_ge1 = len(est_splits - active_ref)
    fn_ge1 = len(active_ref - est_splits)
    rf_ge1 = fp_ge1 + fn_ge1
    denom_ge1 = len(active_ref) + max(len(active_ref), n_est)
    nrf_ge1 = rf_ge1 / denom_ge1 if denom_ge1 > 0 else 0.0

    # Active recall on branches with >= 1 and >= 2 expected mutations
    active_ge2 = {s for s, bl in ref_info["splits"].items() if bl * L >= 2.0}
    rec_ge1 = (len(active_ref & est_splits) / len(active_ref)) if active_ref else 1.0
    rec_ge2 = (len(active_ge2 & est_splits) / len(active_ge2)) if active_ge2 else 1.0

    return {
        "n_true": n_true,
        "n_active_ge1": len(active_ref),
        "n_est": n_est,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "rf_bin": rf_bin,
        "nrf_bin": nrf_bin,
        "rf_ge1": rf_ge1,
        "nrf_ge1": nrf_ge1,
        "precision": prec,
        "recall": rec,
        "recall_ge1": rec_ge1,
        "recall_ge2": rec_ge2,
    }


def parse_iqtree_bic_model(iqtree_report_path: Path):
    """Extract BIC best model name and AliSim-compatible model string from .iqtree report."""
    txt = iqtree_report_path.read_text(errors="ignore")
    m = re.search(r"Best-fit model according to BIC:\s+(\S+)", txt)
    best_bic = m.group(1).strip() if m else "GTR+F+I+G4"
    base_cls = best_bic.split("+")[0]
    if base_cls in ("K2P", "K80"):
        base_cls = "K80"
    elif base_cls in ("JC69", "JC"):
        base_cls = "JC"
    return best_bic, base_cls


def run_bushwhacker(bw_bin: str, fa_path: Path, L: int):
    out_j = Path(str(fa_path) + ".bw.json")
    t0 = time.time()
    proc = subprocess.run(
        [bw_bin, str(fa_path), "-j", str(out_j)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=120,
    )
    dt = time.time() - t0
    if proc.returncode != 0 or not out_j.exists():
        return None, None, dt, 0
    data = json.loads(out_j.read_text())
    mdl_info = get_unrooted_splits(data.get("newick", ""), is_path=False, L=L)
    bin_info = get_unrooted_splits(data.get("binary_newick", ""), is_path=False, L=L)
    n_shunts = len(data.get("shunts", []))
    return mdl_info, bin_info, dt, n_shunts


def run_iqtree(iqtree_bin: str, fa_path: Path, prefix: Path, extra_args: list, L: int, seed: int | None = None):
    t0 = time.time()
    cmd = [iqtree_bin, "-s", str(fa_path), "-pre", str(prefix), "-nt", "1", "-quiet", "-redo"] + extra_args
    if seed is not None:
        cmd.extend(["-seed", str(seed)])
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
    dt = time.time() - t0
    tf = Path(str(prefix) + ".treefile")
    info = get_unrooted_splits(tf, is_path=True, L=L) if tf.exists() else None
    return info, dt


def run_fasttree(fasttree_bin: str, fa_path: Path, out_tree: Path, L: int):
    t0 = time.time()
    with open(out_tree, "w") as out_f:
        subprocess.run(
            [fasttree_bin, "-nt", "-gtr", "-gamma", "-quiet", str(fa_path)],
            stdout=out_f,
            stderr=subprocess.DEVNULL,
            timeout=120,
        )
    dt = time.time() - t0
    info = get_unrooted_splits(out_tree, is_path=True, L=L) if out_tree.exists() else None
    return info, dt


def process_dataset(ds_id: int, base_dir: Path, run_ml: bool = True, seed: int | None = None):
    abadi_dir = base_dir / "abadi_gerth"
    emp_phy = abadi_dir / "empirical" / f"dataset_{ds_id}.phy"
    if not emp_phy.exists():
        return None

    out_dir = abadi_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"ds_{ds_id}.json"

    bw_bin = str(base_dir / "target" / "release" / "bushwhacker")
    iqtree_bin = "/opt/ohpc/pub/apps/iqtree/2.4.0/bin/iqtree2"
    fasttree_bin = "/usr/local/bin/FastTree"
    phyml_bin = str(base_dir / "phyml" / "src" / "phyml")

    with tempfile.TemporaryDirectory(prefix=f"ag_ds{ds_id}_") as tmp:
        tmp_path = Path(tmp)
        emp_fa = tmp_path / "emp.fa"
        emp_clean_phy = tmp_path / "emp.phy"
        N, L, taxa = phy_to_fasta_and_clean_phy(emp_phy, emp_fa, emp_clean_phy)
        if N < 4 or L < 10:
            return None

        result = {
            "dataset_id": ds_id,
            "N": N,
            "L": L,
            "NxL": N * L,
        }

        # 1. Bushwhacker on empirical alignment
        bw_emp_mdl, bw_emp_bin, bw_emp_time, bw_emp_shunts = run_bushwhacker(bw_bin, emp_fa, L)
        result["emp_bw_time_s"] = bw_emp_time
        result["emp_bw_shunts"] = bw_emp_shunts
        result["emp_bw_mdl_resolved"] = len(bw_emp_mdl["splits"]) if bw_emp_mdl else 0

        # 2. PhyML BioNJ(JC) on empirical alignment -> Exact trueTree.txt for c1, c2, c3!
        phyml_tree_info = None
        if os.path.exists(phyml_bin):
            subprocess.run(
                [
                    phyml_bin,
                    "-i", str(emp_clean_phy),
                    "-m", "000000",
                    "-f", "0.25,0.25,0.25,0.25",
                    "-c", "1",
                    "-o", "lr",
                    "-d", "nt",
                    "-n", "1",
                    "-b", "0",
                    "--no_memory_check",
                    "--run_id", "JC",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=60,
            )
            ptf = Path(str(emp_clean_phy) + "_phyml_tree_JC.txt")
            if ptf.exists():
                phyml_tree_info = get_unrooted_splits(ptf, is_path=True, L=L)

        # 3. Evaluate on Abadi et al.'s c0, c1, c2, c3 against their generating tree (phyml_tree_info)
        for c_set in ("c0", "c1", "c2", "c3"):
            c_phy = abadi_dir / c_set / f"dataset_{ds_id}.phy"
            if c_phy.exists() and phyml_tree_info is not None:
                c_fa = tmp_path / f"{c_set}.fa"
                c_clean = tmp_path / f"{c_set}.phy"
                try:
                    cN, cL, _ = phy_to_fasta_and_clean_phy(c_phy, c_fa, c_clean)
                    c_bw_mdl, c_bw_bin, c_bw_t, _ = run_bushwhacker(bw_bin, c_fa, cL)
                    c_ft, c_ft_t = run_fasttree(fasttree_bin, c_fa, tmp_path / f"{c_set}.ft.tree", cL)
                    result[f"{c_set}_bw_mdl"] = evaluate_against_ref(phyml_tree_info, c_bw_mdl, cL)
                    result[f"{c_set}_bw_bin"] = evaluate_against_ref(phyml_tree_info, c_bw_bin, cL)
                    result[f"{c_set}_fasttree"] = evaluate_against_ref(phyml_tree_info, c_ft, cL)
                    result[f"{c_set}_bw_time_s"] = c_bw_t
                    result[f"{c_set}_ft_time_s"] = c_ft_t
                    if run_ml:
                        c_gtr, c_gtr_t = run_iqtree(iqtree_bin, c_fa, tmp_path / f"{c_set}_gtr", ["-m", "GTR+F+I+G4", "-fast"], cL, seed=seed)
                        c_poly, c_poly_t = run_iqtree(iqtree_bin, c_fa, tmp_path / f"{c_set}_poly", ["-m", "GTR+F+I+G4", "--polytomy", "-fast"], cL, seed=seed)
                        result[f"{c_set}_iqtree_gtr"] = evaluate_against_ref(phyml_tree_info, c_gtr, cL)
                        result[f"{c_set}_iqtree_poly"] = evaluate_against_ref(phyml_tree_info, c_poly, cL)
                        result[f"{c_set}_iqtree_time_s"] = c_gtr_t
                except Exception:
                    pass

        if not run_ml:
            out_json.write_text(json.dumps(result))
            return result

        # 4. Run IQ-TREE 2 ModelFinder (BIC over the 24 models), GTR+I+G4, GTR+I+G4 --polytomy, JC, and FastTree 2 on empirical
        bic_pre = tmp_path / "emp_bic"
        ml_Args = ["-fast"] if N > 35 else []
        emp_bic_info, emp_bic_time = run_iqtree(
            iqtree_bin,
            emp_fa,
            bic_pre,
            ["-m", "MFP", "-mset", "JC,F81,K80,HKY,SYM,GTR", "-mrate", "E,I,G,I+G", "-cmax", "4"] + ml_Args,
            L,
            seed=seed,
        )
        emp_gtr_info, emp_gtr_time = run_iqtree(
            iqtree_bin, emp_fa, tmp_path / "emp_gtr", ["-m", "GTR+F+I+G4"] + ml_Args, L, seed=seed
        )
        emp_poly_info, emp_poly_time = run_iqtree(
            iqtree_bin, emp_fa, tmp_path / "emp_poly", ["-m", "GTR+F+I+G4", "--polytomy"] + ml_Args, L, seed=seed
        )
        emp_jc_info, emp_jc_time = run_iqtree(
            iqtree_bin, emp_fa, tmp_path / "emp_jc", ["-m", "JC"] + ml_Args, L, seed=seed
        )
        emp_ft_info, emp_ft_time = run_fasttree(fasttree_bin, emp_fa, tmp_path / "emp.ft.tree", L)

        bionj_path = Path(str(bic_pre) + ".bionj")
        emp_bionj_info = get_unrooted_splits(bionj_path, is_path=True, L=L) if bionj_path.exists() else phyml_tree_info

        iqtree_rep = Path(str(bic_pre) + ".iqtree")
        if iqtree_rep.exists() and emp_bic_info is not None:
            best_bic_model, base_cls = parse_iqtree_bic_model(iqtree_rep)
            result["best_bic_model"] = best_bic_model
            result["best_bic_base"] = base_cls
            result["emp_bic_time_s"] = emp_bic_time
            result["emp_gtr_time_s"] = emp_gtr_time
            result["emp_tbl"] = emp_bic_info["tbl"]
            n_int = max(1, emp_bic_info["n_internal"])
            result["emp_prop_short_001"] = emp_bic_info["n_short_001"] / n_int
            result["emp_prop_sub1_mut"] = emp_bic_info["n_sub1_mut"] / n_int

            # Gerth (2019) empirical comparisons against BIC tree
            if emp_gtr_info is not None:
                cmp_bic_gtr = evaluate_against_ref(emp_bic_info, emp_gtr_info, L)
                result["emp_bic_vs_gtr"] = cmp_bic_gtr

                # Analyze consensus vs conflict splits between BIC and GTR+I+G4
                s_bic = set(emp_bic_info["splits"].keys())
                s_gtr = set(emp_gtr_info["splits"].keys())
                s_agree = s_bic & s_gtr
                s_conflict = s_bic - s_gtr
                s_bw = set(bw_emp_mdl["splits"].keys()) if bw_emp_mdl else set()

                result["emp_n_agree_bic_gtr"] = len(s_agree)
                result["emp_n_conflict_bic_gtr"] = len(s_conflict)
                result["emp_bw_recovers_agree"] = len(s_agree & s_bw)
                result["emp_bw_contracts_conflict"] = len(s_conflict - s_bw)
                result["emp_conflict_sub1_mut"] = sum(
                    1 for s in s_conflict if emp_bic_info["splits"].get(s, 0.0) * L < 1.0
                )

            result["emp_bic_vs_jc"] = evaluate_against_ref(emp_bic_info, emp_jc_info, L)
            result["emp_bic_vs_bw_mdl"] = evaluate_against_ref(emp_bic_info, bw_emp_mdl, L)
            result["emp_bic_vs_bw_bin"] = evaluate_against_ref(emp_bic_info, bw_emp_bin, L)
            result["emp_bic_vs_poly"] = evaluate_against_ref(emp_bic_info, emp_poly_info, L)

            # 5. Matched Ground-Truth Simulation (`sim_bic`) via IQ-TREE 2 AliSim on the empirical BIC tree & model
            sim_pre = tmp_path / "sim_bic"
            bic_treefile = Path(str(bic_pre) + ".treefile")
            try:
                subprocess.run(
                    [
                        iqtree_bin,
                        "--alisim", str(sim_pre),
                        "-t", str(bic_treefile),
                        "-m", best_bic_model,
                        "--length", str(L),
                        "-af", "fasta",
                        "-seed", str(100000 + ds_id),
                        "-quiet",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=120,
                    check=True,
                )
                sim_fa = Path(str(sim_pre) + ".fa")
                if sim_fa.exists():
                    s_bw_mdl, s_bw_bin, s_bw_t, s_bw_shunts = run_bushwhacker(bw_bin, sim_fa, L)
                    s_gtr, s_gtr_t = run_iqtree(iqtree_bin, sim_fa, tmp_path / "s_gtr", ["-m", "GTR+F+I+G4"] + ml_Args, L, seed=seed)
                    s_poly, s_poly_t = run_iqtree(iqtree_bin, sim_fa, tmp_path / "s_poly", ["-m", "GTR+F+I+G4", "--polytomy"] + ml_Args, L, seed=seed)
                    s_bic, s_bic_t = run_iqtree(iqtree_bin, sim_fa, tmp_path / "s_bic", ["-m", best_bic_model] + ml_Args, L, seed=seed)
                    s_jc, s_jc_t = run_iqtree(iqtree_bin, sim_fa, tmp_path / "s_jc", ["-m", "JC"] + ml_Args, L, seed=seed)
                    s_ft, s_ft_t = run_fasttree(fasttree_bin, sim_fa, tmp_path / "s.ft.tree", L)
                    s_bionj_path = tmp_path / "s_gtr.bionj"
                    s_bionj = get_unrooted_splits(s_bionj_path, is_path=True, L=L) if s_bionj_path.exists() else None

                    result["sim_bic"] = {
                        "bw_mdl": evaluate_against_ref(emp_bic_info, s_bw_mdl, L),
                        "bw_bin": evaluate_against_ref(emp_bic_info, s_bw_bin, L),
                        "iqtree_gtr": evaluate_against_ref(emp_bic_info, s_gtr, L),
                        "iqtree_poly": evaluate_against_ref(emp_bic_info, s_poly, L),
                        "iqtree_bic": evaluate_against_ref(emp_bic_info, s_bic, L),
                        "iqtree_jc": evaluate_against_ref(emp_bic_info, s_jc, L),
                        "fasttree": evaluate_against_ref(emp_bic_info, s_ft, L),
                        "bionj": evaluate_against_ref(emp_bic_info, s_bionj, L),
                        "bw_time_s": s_bw_t,
                        "gtr_time_s": s_gtr_t,
                        "bic_time_s": s_bic_t,
                        "ft_time_s": s_ft_t,
                        "bw_shunts": s_bw_shunts,
                    }
            except Exception:
                pass

        out_json.write_text(json.dumps(result))
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--base-dir", type=str, default=str(EXTERNAL_DIR))
    parser.add_argument("--fast-only", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help="Optional fixed random seed passed to IQ-TREE 2 (-seed)")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    for ds_id in range(args.start, args.end + 1):
        out_json = base_dir / "abadi_gerth" / "results" / f"ds_{ds_id}.json"
        if out_json.exists():
            continue
        try:
            process_dataset(ds_id, base_dir, run_ml=not args.fast_only, seed=args.seed)
        except Exception as e:
            sys.stderr.write(f"Dataset {ds_id} error: {e}\n")


if __name__ == "__main__":
    main()
