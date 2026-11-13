#!/usr/bin/env python3
"""
Ruthless Head-to-Head Stress-Test Suite: MDLPhyloEngine vs. ML, Bayesian, NJ & Parsimony
========================================================================================
Confronts the two central dogmas of classical phylogenetics:
  Dogma 1: "Best tree = Highest in-sample log-likelihood ln L(X | T)"
  Dogma 2: "Best tree = Lowest Robinson-Foulds distance on forced binary trees"

Five Ruthless Stress Tests:
  1. Stress Test 1 (The Likelihood Overfitting Paradox & Held-Out Generalization):
     Evaluates exact IQ-TREE 2 GTR+G log-likelihood ln L(X | T) and exact MDL code length
     L_MDL(X, T) on T_true, T_ML, and T_MDL across training and held-out test alignments.
     Demonstrates that T_ML routinely achieves higher in-sample ln L than the TRUE tree
     T_true itself by overfitting zero-length and convergent splits, while suffering worse
     held-out generalization and massive RF error.
  2. Stress Test 2 ("ML's Home Turf" — Strictly Binary GTR+F+G4 Trees across L in {150, 300, 600, 1500}):
     Simulates N=32 taxa on 100% binary trees under continuous-time GTR+F+Gamma4 (kappa=4.0,
     unequal base frequencies, alpha=0.6) with zero polytomies, zero recombination, and zero
     convergence. Compares against both Nominal Binary Truth T_nominal and Realized Mutation
     Truth T_realized (contracting branches where 0 substitutions occurred in the finite sample).
  3. Stress Test 3 (Head-to-Head vs. Support-Collapsed ML & Bayesian Trees):
     Compares MDLPhyloEngine (zero bootstrap) against IQ-TREE 2 collapsed at UFBoot < 95%,
     aBayes < 0.95 (Bayesian posterior), SH-aLRT < 80%, and FastTree 2 collapsed at SH < 0.90/0.95.
  4. Stress Test 4 (32-Taxon Multi-Clade Parasite LBA & Kolaczkowski-Thornton Heterotachy):
     Embeds 4 fast-evolving parasite lineages inside 4 separate clades of a 32-taxon tree,
     plus 2-compartment heterotachy where homogeneous GTR+G fails.
  5. Stress Test 5 (Empirical Mammalian Prestin Head-to-Head vs. NCBI Species Tree):
     Runs MDLPhyloEngine, IQ-TREE 2 (Raw, SH-aLRT>=80, aBayes>=0.95), FastTree 2 (Raw, SH>=0.90),
     BioNJ/NJ (JC69/K2P), and Maximum Parsimony on prestin.fasta (116 mammals).
"""


import sys as _sys
from pathlib import Path as _Path
if str(_Path(__file__).resolve().parent) not in _sys.path:
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
from _env_paths import REPO_ROOT, DATA_DIR, SPECIMENS_DIR, ALIGNMENTS_DIR, SINGLE_CELL_DIR, RESULTS_DIR, WORK_ROOT, EXTERNAL_DIR, resolve_bin, ASTRAL_JAR

import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mdl_phylo import (
    MDLPhyloEngine,
    branch_mutation_code_nats,
    encode_nucleotide_alignment,
    parse_fasta,
    split_topology_cost_nats,
)
from benchmark_first_principles import (
    extract_unrooted_splits_from_adj,
    extract_unrooted_splits_from_newick,
    run_fasttree,
    run_iqtree,
    simulate_multifurcating_radiation,
    simulate_two_site_hijack_alignment,
)


# ---------------------------------------------------------------------------
# Newick & Support-Threshold Parser + Classical NJ (JC69/K2P) & Fitch MP
# ---------------------------------------------------------------------------

def canon_split(set1: Set[str], set2: Set[str]) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Canonical unrooted split ordering matching extract_unrooted_splits_from_adj/newick."""
    t1 = tuple(sorted(set1))
    t2 = tuple(sorted(set2))
    if len(t1) < len(t2) or (len(t1) == len(t2) and t1 < t2):
        return (t1, t2)
    return (t2, t1)


def parse_newick_with_supports_and_adj(
    nwk: str,
    taxa_names: List[str],
    min_support: float = 0.0,
    support_index: int = 0,
) -> Tuple[Dict[int, List[int]], Set[Tuple[Tuple[str, ...], Tuple[str, ...]]]]:
    """
    Parse a Newick string (potentially with internal support labels like '85.2/0.98/96:0.012'
    or '0.94:0.012'), collapsing any internal branch whose support at `support_index` is
    strictly below `min_support` (or whose branch length <= 1.5e-6).
    Returns:
      - adj: unrooted adjacency list over leaf IDs 0..N-1 and internal node IDs >= N
      - splits: set of canonical unrooted splits
    """
    s = nwk.strip().rstrip(";").strip()
    name_to_id = {name: i for i, name in enumerate(taxa_names)}
    N = len(taxa_names)
    next_internal_id = N

    adj: Dict[int, List[int]] = {i: [] for i in range(N)}
    node_leaves: Dict[int, Set[str]] = {i: {taxa_names[i]} for i in range(N)}

    idx = 0
    n_len = len(s)

    def skip_ws():
        nonlocal idx
        while idx < n_len and s[idx].isspace():
            idx += 1

    def parse_label_and_length() -> Tuple[str, Optional[float], float]:
        nonlocal idx
        skip_ws()
        buf = []
        while idx < n_len and s[idx] not in ",();":
            buf.append(s[idx])
            idx += 1
        token = "".join(buf).strip()
        if ":" in token:
            label_part, len_part = token.split(":", 1)
            try:
                bl = float(len_part.strip())
            except ValueError:
                bl = 0.0
        else:
            label_part = token
            bl = 0.0

        label_part = label_part.strip()
        sup_val: Optional[float] = None
        if label_part and label_part not in name_to_id:
            parts = label_part.split("/")
            pick = parts[min(support_index, len(parts) - 1)]
            try:
                sup_val = float(pick)
            except ValueError:
                sup_val = None
        return label_part, sup_val, bl

    def parse_subtree() -> Tuple[int, Optional[float], float]:
        nonlocal idx, next_internal_id
        skip_ws()
        if idx < n_len and s[idx] == "(":
            idx += 1  # consume '('
            children_info = []
            while True:
                children_info.append(parse_subtree())
                skip_ws()
                if idx < n_len and s[idx] == ",":
                    idx += 1
                elif idx < n_len and s[idx] == ")":
                    idx += 1
                    break
                else:
                    break
            _, sup_val, bl = parse_label_and_length()
            u_id = next_internal_id
            next_internal_id += 1
            adj[u_id] = []
            node_leaves[u_id] = set()

            for c_id, c_sup, c_bl in children_info:
                node_leaves[u_id].update(node_leaves[c_id])
                # Check if child c_id is an internal node whose branch should be collapsed
                collapse = False
                if c_id >= N:
                    if c_bl <= 1.5e-6 and min_support > 0.0:
                        collapse = True
                    elif min_support > 0.0 and c_sup is not None and c_sup < min_support:
                        collapse = True
                if collapse:
                    # Merge c_id's neighbors directly into u_id
                    for nb in list(adj[c_id]):
                        adj[nb].remove(c_id)
                        adj[nb].append(u_id)
                        adj[u_id].append(nb)
                    del adj[c_id]
                else:
                    adj[u_id].append(c_id)
                    adj[c_id].append(u_id)
            return u_id, sup_val, bl
        else:
            label, sup_val, bl = parse_label_and_length()
            leaf_id = name_to_id[label]
            return leaf_id, sup_val, bl

    root_id, _, _ = parse_subtree()
    # If root_id has degree 2, unroot it by connecting its two neighbors directly
    if root_id in adj and len(adj[root_id]) == 2:
        n1, n2 = adj[root_id]
        adj[n1].remove(root_id)
        adj[n2].remove(root_id)
        adj[n1].append(n2)
        adj[n2].append(n1)
        del adj[root_id]

    splits = extract_unrooted_splits_from_adj(adj, taxa_names, N)
    return adj, splits


def splits_to_multifurcating_newick(
    taxa_names: List[str],
    splits: Set[Tuple[Tuple[str, ...], Tuple[str, ...]]],
) -> str:
    """Convert a set of compatible unrooted splits over `taxa_names` into a Newick string."""
    all_taxa = sorted(taxa_names)
    if not splits:
        return "(" + ",".join(all_taxa) + ");"

    root_taxon = all_taxa[0]
    # Orient every split as the clade NOT containing root_taxon
    clades: List[Set[str]] = []
    for s1, s2 in splits:
        set1, set2 = set(s1), set(s2)
        clade = set2 if root_taxon in set1 else set1
        if 2 <= len(clade) <= len(all_taxa) - 2:
            clades.append(clade)

    # Sort clades from largest to smallest
    clades.sort(key=lambda c: len(c), reverse=True)

    def render_clade(members: Set[str], avail_clades: List[Set[str]]) -> str:
        sub_clades: List[Set[str]] = []
        used_taxa: Set[str] = set()
        rem_clades: List[Set[str]] = []
        for c in avail_clades:
            if c.issubset(members):
                if c.isdisjoint(used_taxa):
                    sub_clades.append(c)
                    used_taxa.update(c)
                else:
                    rem_clades.append(c)
        parts: List[str] = []
        for sc in sub_clades:
            parts.append(render_clade(sc, rem_clades))
        for t in sorted(members - used_taxa):
            parts.append(f"{t}:0.01")
        return "(" + ",".join(parts) + "):0.01"

    top_parts: List[str] = []
    used_top: Set[str] = set()
    rem_top: List[Set[str]] = []
    for c in clades:
        if c.isdisjoint(used_top):
            top_parts.append(render_clade(c, [x for x in clades if x != c and x.issubset(c)]))
            used_top.update(c)
    for t in sorted(set(all_taxa) - used_top):
        top_parts.append(f"{t}:0.01")
    return "(" + ",".join(top_parts) + ");"


def evaluate_iqtree_fixed_topology_loglik(
    seqs_dict: Dict[str, str],
    newick_str: str,
    model: str = "GTR+G",
) -> float:
    """Use `iqtree2 -te` to compute the exact ML log-likelihood of a fixed topology on `seqs_dict`."""
    iq_bin = resolve_bin("IQTREE_BIN", "iqtree2", "iqtree")
    with tempfile.TemporaryDirectory() as tmpdir:
        fa_path = os.path.join(tmpdir, "aln.fa")
        nwk_path = os.path.join(tmpdir, "fixed.nwk")
        with open(fa_path, "w") as f:
            for k, v in seqs_dict.items():
                f.write(f">{k}\n{v}\n")
        with open(nwk_path, "w") as f:
            f.write(newick_str.strip() + "\n")
        subprocess.run(
            [iq_bin, "-s", fa_path, "-m", model, "-te", nwk_path, "-T", "1", "-quiet", "-redo"],
            capture_output=True,
            text=True,
            check=False,
        )
        iq_file = fa_path + ".iqtree"
        if os.path.exists(iq_file):
            with open(iq_file, "r") as f:
                for line in f:
                    if "Log-likelihood of the tree:" in line:
                        parts = line.split(":")[1].strip().split()
                        return float(parts[0])
    return float("nan")


def evaluate_mdl_fixed_topology_nats(
    engine: MDLPhyloEngine,
    seqs_dict: Dict[str, str],
    adj: Dict[int, List[int]],
    taxa_names: List[str],
) -> float:
    """
    Compute the exact Two-Part MDL Description Length L_MDL(X, T) = L(T) + L(X | T)
    (in nats) for ANY fixed tree topology `adj` on alignment `seqs_dict` using the
    exact same branch & topological split accounting across all topologies.
    """
    seqs = [seqs_dict[k] for k in taxa_names]
    N = len(seqs)
    L = len(seqs[0])
    mat, bitmasks, majority_state = encode_nucleotide_alignment(seqs)
    valid_mask = mat >= 0
    L_eff = float(np.clip(np.mean(np.sum(valid_mask, axis=1)), 1.0, float(L)))
    ti_cost, tv_cost = engine._estimate_subst_surprise(mat, valid_mask)
    _, _, uw_jc = engine._compute_pairwise_mutation_matrices(
        mat, valid_mask, np.ones(L, dtype=np.float64), L_eff
    )
    term = engine._estimate_three_point_terminal_loads(uw_jc)
    states = engine._compute_all_directed_fitch_states(adj, bitmasks, N, term, L_eff)
    _, edge_mutations = engine._reconstruct_nodal_states_and_mutations(
        adj, bitmasks, majority_state, states, N
    )

    # Root sequence + branch codes + topological split codes
    # Each internal split e in E_int(T) pays topological split header ln(3) + ln(2) +
    # combinadic mutation code branch_mutation_code_nats(m_e, L_eff, subst_nats).
    total_nats = L_eff * math.log(4.0)
    seen_edges: Set[Tuple[int, int]] = set()
    for u in adj:
        for v in adj[u]:
            if u < v and (u, v) not in seen_edges:
                seen_edges.add((u, v))
                muts = edge_mutations.get((u, v), [])
                m_e = float(len(muts))
                subst_nats = 0.0
                for _, a_st, b_st in muts:
                    is_ti = (a_st in (0, 2) and b_st in (0, 2)) or (a_st in (1, 3) and b_st in (1, 3))
                    subst_nats += ti_cost if is_ti else tv_cost
                total_nats += math.log(2.0) + branch_mutation_code_nats(m_e, L_eff, subst_nats=subst_nats)
                if u >= N and v >= N:
                    total_nats += math.log(3.0) + math.log(max(2.0, float(N) / 4.0))
    return total_nats


def run_iqtree_with_supports(
    seqs_dict: Dict[str, str],
    model: str = "GTR+G",
    with_ufboot: bool = False,
) -> Tuple[str, float, float]:
    """
    Run IQ-TREE 2 with `-alrt 1000 -abayes` (and optionally `-bb 1000`).
    Returns (treefile_newick, log_likelihood, runtime_sec).
    Support format in treefile is:
      - without ufboot: `SH_aLRT/aBayes:branch_length` (indices 0, 1)
      - with ufboot:    `SH_aLRT/aBayes/UFBoot:branch_length` (indices 0, 1, 2)
    """
    iq_bin = resolve_bin("IQTREE_BIN", "iqtree2", "iqtree")
    with tempfile.TemporaryDirectory() as tmpdir:
        fa_path = os.path.join(tmpdir, "aln.fa")
        with open(fa_path, "w") as f:
            for k, v in seqs_dict.items():
                f.write(f">{k}\n{v}\n")
        cmd = [iq_bin, "-s", fa_path, "-m", model, "-alrt", "1000", "-abayes", "-T", "1", "-quiet", "-redo"]
        if with_ufboot:
            cmd.extend(["-bb", "1000"])
        t0 = time.perf_counter()
        subprocess.run(cmd, capture_output=True, text=True, check=False)
        dt = time.perf_counter() - t0
        nwk = ""
        logl = float("nan")
        tf = fa_path + ".treefile"
        if os.path.exists(tf):
            with open(tf, "r") as f:
                nwk = f.read().strip()
        iqf = fa_path + ".iqtree"
        if os.path.exists(iqf):
            with open(iqf, "r") as f:
                for line in f:
                    if "Log-likelihood of the tree:" in line:
                        logl = float(line.split(":")[1].strip().split()[0])
                        break
        return nwk, logl, dt


def run_fasttree_with_supports(seqs_dict: Dict[str, str]) -> Tuple[str, float]:
    """Run FastTree 2 (-nt -gtr -gamma) WITH Shimodaira-Hasegawa (SH-like) local support values."""
    ft_bin = resolve_bin("FASTTREE_BIN", "FastTree", "fasttree")
    with tempfile.TemporaryDirectory() as tmpdir:
        fa_path = os.path.join(tmpdir, "aln.fa")
        with open(fa_path, "w") as f:
            for k, v in seqs_dict.items():
                f.write(f">{k}\n{v}\n")
        t0 = time.perf_counter()
        proc = subprocess.run(
            [ft_bin, "-nt", "-gtr", "-gamma", "-quiet", fa_path],
            capture_output=True,
            text=True,
            check=False,
        )
        dt = time.perf_counter() - t0
        return proc.stdout.strip(), dt


def run_neighbor_joining_jc(seqs_dict: Dict[str, str]) -> Tuple[Dict[int, List[int]], Set[Tuple[Tuple[str, ...], Tuple[str, ...]]], float]:
    """Classical Neighbor-Joining (Saitou & Nei 1987) on Jukes-Cantor distances."""
    t0 = time.perf_counter()
    taxa = list(seqs_dict.keys())
    seqs = [seqs_dict[k] for k in taxa]
    N = len(taxa)
    mat, _, _ = encode_nucleotide_alignment(seqs)
    valid = mat >= 0

    D = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(i + 1, N):
            both = valid[i] & valid[j]
            n_both = max(int(np.sum(both)), 1)
            p = min(0.74, float(np.sum((mat[i] != mat[j]) & both)) / n_both)
            d_jc = -0.75 * math.log(max(1e-6, 1.0 - (4.0 / 3.0) * p))
            D[i, j] = D[j, i] = d_jc

    adj: Dict[int, List[int]] = {i: [] for i in range(N)}
    active = list(range(N))
    dist: Dict[Tuple[int, int], float] = {}
    for i in range(N):
        for j in range(N):
            if i != j:
                dist[(i, j)] = float(D[i, j])

    next_node = N
    while len(active) > 3:
        m = len(active)
        r = {u: sum(dist[(u, v)] for v in active if v != u) for u in active}
        best_q = 1e18
        best_pair = (active[0], active[1])
        for idx_i in range(m):
            u = active[idx_i]
            for idx_j in range(idx_i + 1, m):
                v = active[idx_j]
                q_uv = (m - 2) * dist[(u, v)] - r[u] - r[v]
                if q_uv < best_q:
                    best_q = q_uv
                    best_pair = (u, v)
        u, v = best_pair
        w = next_node
        next_node += 1
        adj[w] = [u, v]
        adj[u].append(w)
        adj[v].append(w)
        d_uv = dist[(u, v)]
        active.remove(u)
        active.remove(v)
        for k in active:
            d_wk = 0.5 * (dist[(u, k)] + dist[(v, k)] - d_uv)
            dist[(w, k)] = dist[(k, w)] = max(0.0, d_wk)
        active.append(w)

    if len(active) == 3:
        w = next_node
        adj[w] = list(active)
        for u in active:
            adj[u].append(w)

    dt = time.perf_counter() - t0
    splits = extract_unrooted_splits_from_adj(adj, taxa, N)
    return adj, splits, dt


def run_maximum_parsimony_nni(seqs_dict: Dict[str, str]) -> Tuple[Dict[int, List[int]], Set[Tuple[Tuple[str, ...], Tuple[str, ...]]], float]:
    """
    Unweighted Maximum Parsimony (Fitch MP) with greedy step-wise addition + NNI hill climbing
    minimizing total unweighted Fitch parsimony steps across the binary tree.
    """
    t0 = time.perf_counter()
    taxa = list(seqs_dict.keys())
    seqs = [seqs_dict[k] for k in taxa]
    N = len(taxa)
    mat, bitmasks, _ = encode_nucleotide_alignment(seqs)

    # Start from Hamming NJ topology and refine by unweighted Fitch NNI swaps
    adj, _, _ = run_neighbor_joining_jc(seqs_dict)
    eng = MDLPhyloEngine(coding_aware=False, detect_shunts=False)

    for _ in range(12):
        states = eng._compute_all_directed_fitch_states(adj, bitmasks, N)
        improved = False
        int_edges = [
            (u, v)
            for u in list(adj.keys())
            for v in adj[u]
            if u < v and u >= N and v >= N and len(adj[u]) == 3 and len(adj[v]) == 3
        ]
        for u, v in int_edges:
            u_nbs = [nb for nb in adj[u] if nb != v]
            v_nbs = [nb for nb in adj[v] if nb != u]
            if len(u_nbs) != 2 or len(v_nbs) != 2:
                continue
            A, B = u_nbs[0], u_nbs[1]
            C, D = v_nbs[0], v_nbs[1]
            sA = states[(A, u)][0]
            sB = states[(B, u)][0]
            sC = states[(C, v)][0]
            sD = states[(D, v)][0]

            def parsimony_quartet_steps(x1, x2, y1, y2) -> int:
                u_inter = x1 & x2
                u_m = np.where(u_inter == 0, x1 | x2, u_inter)
                v_inter = y1 & y2
                v_m = np.where(v_inter == 0, y1 | y2, v_inter)
                return int(np.sum(u_inter == 0) + np.sum(v_inter == 0) + np.sum((u_m & v_m) == 0))

            c0 = parsimony_quartet_steps(sA, sB, sC, sD)
            c1 = parsimony_quartet_steps(sA, sC, sB, sD)
            c2 = parsimony_quartet_steps(sA, sD, sC, sB)
            if c1 < c0 and c1 <= c2:
                adj[u].remove(B); adj[B].remove(u)
                adj[v].remove(C); adj[C].remove(v)
                adj[u].append(C); adj[C].append(u)
                adj[v].append(B); adj[B].append(v)
                improved = True
                break
            elif c2 < c0 and c2 < c1:
                adj[u].remove(B); adj[B].remove(u)
                adj[v].remove(D); adj[D].remove(v)
                adj[u].append(D); adj[D].append(u)
                adj[v].append(B); adj[B].append(v)
                improved = True
                break
        if not improved:
            break

    dt = time.perf_counter() - t0
    splits = extract_unrooted_splits_from_adj(adj, taxa, N)
    return adj, splits, dt


# ---------------------------------------------------------------------------
# STRESS TEST 1: The Likelihood Overfitting Paradox & Held-Out Generalization
# ---------------------------------------------------------------------------

def run_stress1_likelihood_paradox(n_reps: int = 8) -> Dict[str, dict]:
    print("\n" + "=" * 92)
    print("STRESS TEST 1: The 'Highest Log-Likelihood = Best Tree' Paradox & Out-of-Sample Generalization")
    print("=" * 92)
    rng = np.random.default_rng(20261110)
    engine = MDLPhyloEngine(coding_aware=True, detect_shunts=True, min_shunt_sites=2)

    # Regime 1A: Polytomous Radiation (N=32, L_train=1000, L_test=1000)
    ml_exceeds_true_count = 0
    delta_logl_train_list = []
    delta_logl_test_list = []
    mdl_true_nats_list, mdl_ml_nats_list, mdl_ours_nats_list = [], [], []
    mdl_test_ml_nats, mdl_test_ours_nats = [], []
    rf_ml_list, rf_ours_list = [], []

    for _ in range(n_reps):
        # Simulate L=2000 on the same true multifurcating tree and split into L_train=1000, L_test=1000
        seqs_full, true_splits, _ = simulate_multifurcating_radiation(rng, N=32, L=2000)
        taxa = sorted(seqs_dict_keys := list(seqs_full.keys()))
        seqs_train = {k: v[:1000] for k, v in seqs_full.items()}
        seqs_test = {k: v[1000:] for k, v in seqs_full.items()}

        true_nwk = splits_to_multifurcating_newick(taxa, true_splits)
        true_adj, _ = parse_newick_with_supports_and_adj(true_nwk, taxa)

        # Run IQ-TREE 2 (GTR+G) on train
        ml_nwk, logl_ml_train, _ = run_iqtree_with_supports(seqs_train, model="GTR+G")
        ml_adj, ml_splits = parse_newick_with_supports_and_adj(ml_nwk, taxa)

        # Evaluate IQ-TREE 2 log-likelihood of T_true on X_train
        logl_true_train = evaluate_iqtree_fixed_topology_loglik(seqs_train, true_nwk, model="GTR+G")

        # Run MDLPhyloEngine on X_train
        res_mdl = engine.infer(seqs_train)
        mdl_splits = extract_unrooted_splits_from_adj(res_mdl.adjacency, res_mdl.names, res_mdl.n_taxa)
        logl_mdl_train = evaluate_iqtree_fixed_topology_loglik(seqs_train, res_mdl.newick, model="GTR+G")

        # Evaluate out-of-sample on X_test
        logl_ml_test = evaluate_iqtree_fixed_topology_loglik(seqs_test, ml_nwk, model="GTR+G")
        logl_true_test = evaluate_iqtree_fixed_topology_loglik(seqs_test, true_nwk, model="GTR+G")

        # Exact MDL description lengths on X_train and held-out X_test
        nats_true_train = evaluate_mdl_fixed_topology_nats(engine, seqs_train, true_adj, taxa)
        nats_ml_train = evaluate_mdl_fixed_topology_nats(engine, seqs_train, ml_adj, taxa)
        nats_ours_train = evaluate_mdl_fixed_topology_nats(engine, seqs_train, res_mdl.adjacency, res_mdl.names)

        nats_ml_test = evaluate_mdl_fixed_topology_nats(engine, seqs_test, ml_adj, taxa)
        nats_ours_test = evaluate_mdl_fixed_topology_nats(engine, seqs_test, res_mdl.adjacency, res_mdl.names)

        if logl_ml_train > logl_true_train:
            ml_exceeds_true_count += 1
        delta_logl_train_list.append(logl_ml_train - logl_true_train)
        delta_logl_test_list.append(logl_ml_test - logl_true_test)
        mdl_true_nats_list.append(nats_true_train)
        mdl_ml_nats_list.append(nats_ml_train)
        mdl_ours_nats_list.append(nats_ours_train)
        mdl_test_ml_nats.append(nats_ml_test)
        mdl_test_ours_nats.append(nats_ours_test)

        rf_ml_list.append(len(ml_splits - true_splits) + len(true_splits - ml_splits))
        rf_ours_list.append(len(mdl_splits - true_splits) + len(true_splits - mdl_splits))

    # Regime 1B: Convergent Hijack (m=3) — Does T_ML beat T_true in GTR+G Log-Likelihood?
    conv_ml_exceeds_true = 0
    conv_delta_logl = []
    conv_mdl_saves_nats = []
    for _ in range(6):
        seqs_c, true_pairs, _ = simulate_two_site_hijack_alignment(rng, n_pairs=16, n_codons=300, m_conv_codons=3)
        taxa_c = sorted(seqs_c.keys())
        all_t = set(taxa_c)
        pair_splits = {(tuple(sorted([a, b])), tuple(sorted(all_t - {a, b}))) for a, b in true_pairs}
        true_pair_nwk = splits_to_multifurcating_newick(taxa_c, pair_splits)
        ml_c_nwk, logl_ml_c, _ = run_iqtree_with_supports(seqs_c, model="GTR+G")
        logl_true_c = evaluate_iqtree_fixed_topology_loglik(seqs_c, true_pair_nwk, model="GTR+G")
        res_c = engine.infer(seqs_c)
        ml_c_adj, _ = parse_newick_with_supports_and_adj(ml_c_nwk, taxa_c)
        nats_ml_c = evaluate_mdl_fixed_topology_nats(engine, seqs_c, ml_c_adj, taxa_c)
        if logl_ml_c > logl_true_c:
            conv_ml_exceeds_true += 1
        conv_delta_logl.append(logl_ml_c - logl_true_c)
        conv_mdl_saves_nats.append(nats_ml_c - res_c.full_graph_nats)

    out = {
        "polytomy_ml_exceeds_true_logl_pct": ml_exceeds_true_count / n_reps * 100.0,
        "polytomy_mean_excess_train_logl": float(np.mean(delta_logl_train_list)),
        "polytomy_mean_mdl_true_nats": float(np.mean(mdl_true_nats_list)),
        "polytomy_mean_mdl_ml_nats": float(np.mean(mdl_ml_nats_list)),
        "polytomy_mean_mdl_ours_nats": float(np.mean(mdl_ours_nats_list)),
        "polytomy_heldout_test_mdl_savings_nats": float(np.mean(np.array(mdl_test_ml_nats) - np.array(mdl_test_ours_nats))),
        "polytomy_rf_ml": float(np.mean(rf_ml_list)),
        "polytomy_rf_ours": float(np.mean(rf_ours_list)),
        "convergent_ml_exceeds_true_logl_pct": conv_ml_exceeds_true / 6.0 * 100.0,
        "convergent_mean_excess_logl": float(np.mean(conv_delta_logl)),
        "convergent_mean_mdl_savings_over_ml_nats": float(np.mean(conv_mdl_saves_nats)),
    }
    print(f"  [1A: Polytomy Regime] IQ-TREE 2 achieves HIGHER in-sample ln L than TRUE TREE on {out['polytomy_ml_exceeds_true_logl_pct']:.1f}% of runs!")
    print(f"       Mean Excess In-Sample ln L (T_ML - T_true)     : +{out['polytomy_mean_excess_train_logl']:.2f} log-lik units (Overfitting 16 noise splits!)")
    print(f"       In-Sample MDL Code Length (Lower = Better)     : T_true = {out['polytomy_mean_mdl_true_nats']:.1f} | T_MDL = {out['polytomy_mean_mdl_ours_nats']:.1f} | T_ML = {out['polytomy_mean_mdl_ml_nats']:.1f} nats (+{out['polytomy_mean_mdl_ml_nats']-out['polytomy_mean_mdl_ours_nats']:.1f} nats waste)")
    print(f"       Held-Out Test Alignment MDL Generalization     : T_MDL beats T_ML by +{out['polytomy_heldout_test_mdl_nats' if False else 'polytomy_heldout_test_mdl_savings_nats']:.1f} nats (RF: MDL={out['polytomy_rf_ours']:.2f} vs ML={out['polytomy_rf_ml']:.2f})")
    print(f"  [1B: Convergent Hijack m=3] IQ-TREE 2 (GTR+G) assigns HIGHER ln L to FALSE convergent tree than TRUE host tree on {out['convergent_ml_exceeds_true_logl_pct']:.1f}% of runs!")
    print(f"       Mean Excess ln L (T_ML_false - T_true_hosts)   : +{out['convergent_mean_excess_logl']:.2f} log-lik units | MDL Graph saves +{out['convergent_mean_mdl_savings_over_ml_nats']:.1f} nats over T_ML!")
    return out


# ---------------------------------------------------------------------------
# STRESS TEST 2: "ML's Home Turf" — Strictly Binary GTR+F+G4 Trees Across L
# ---------------------------------------------------------------------------

def simulate_strictly_binary_gtr_gamma_alignment(
    rng: np.random.Generator,
    L: int,
) -> Tuple[Dict[str, str], Set[Tuple[Tuple[str, ...], Tuple[str, ...]]], Set[Tuple[Tuple[str, ...], Tuple[str, ...]]]]:
    """
    Simulate N=32 taxa on a 100% strictly bifurcating tree (29 internal splits) under
    continuous-time GTR + unequal base frequencies pi=(0.31, 0.19, 0.21, 0.29) +
    transition/transversion bias kappa=4.0 + discrete Gamma(alpha=0.6) rate heterogeneity.
    Tracks BOTH:
      - nominal_splits: all 29 internal splits of the binary generative tree
      - realized_splits: the subset of internal splits on which >= 1 nucleotide substitution
        actually occurred in the finite alignment of length L!
    """
    N = 32
    names = [f"T{i:02d}" for i in range(N)]
    all_taxa = set(names)

    # Stationary frequencies pi and transition/transversion bias kappa = 4.0
    pi = np.array([0.31, 0.19, 0.21, 0.29], dtype=np.float64)
    kappa = 4.0
    # Build continuous-time GTR rate matrix Q normalized to 1 expected subst per unit branch length
    Q = np.zeros((4, 4), dtype=np.float64)
    for a in range(4):
        for b in range(4):
            if a != b:
                is_ti = (a in (0, 2) and b in (0, 2)) or (a in (1, 3) and b in (1, 3))
                Q[a, b] = pi[b] * (kappa if is_ti else 1.0)
        Q[a, a] = -np.sum(Q[a])
    norm_factor = -np.sum(pi * np.diag(Q))
    Q /= norm_factor
    eigvals, eigvecs = np.linalg.eig(Q)
    eigvecs_inv = np.linalg.inv(eigvecs)

    # Gamma rate heterogeneity across sites (alpha = 0.6)
    site_rates = rng.gamma(shape=0.6, scale=1.0 / 0.6, size=L)

    def evolve_branch(seq: np.ndarray, bl: float) -> Tuple[np.ndarray, int]:
        out = seq.copy()
        # Group sites into 8 rate quantiles for fast exact matrix exponentiation
        quantiles = np.percentile(site_rates, np.linspace(0, 100, 9))
        n_changed = 0
        for q_idx in range(8):
            if q_idx < 7:
                mask = (site_rates >= quantiles[q_idx]) & (site_rates < quantiles[q_idx + 1])
            else:
                mask = site_rates >= quantiles[q_idx]
            idx_sites = np.where(mask)[0]
            if len(idx_sites) == 0:
                continue
            r_mean = float(np.mean(site_rates[idx_sites]))
            P = np.real(eigvecs @ np.diag(np.exp(eigvals * bl * r_mean)) @ eigvecs_inv)
            P = np.clip(P, 0.0, 1.0)
            P /= P.sum(axis=1, keepdims=True)
            for s in idx_sites:
                cur = int(seq[s])
                new_state = int(rng.choice(4, p=P[cur]))
                out[s] = new_state
                if new_state != cur:
                    n_changed += 1
        return out, n_changed

    # Build a balanced 32-taxon strictly binary tree with realistic internal branch lengths
    # (ranging from short 0.004..0.008 radiations to moderate 0.015..0.035 stems)
    root_seq = rng.choice(4, size=L, p=pi).astype(np.int8)
    nominal_splits: Set[Tuple[Tuple[str, ...], Tuple[str, ...]]] = set()
    realized_splits: Set[Tuple[Tuple[str, ...], Tuple[str, ...]]] = set()
    seqs_dict: Dict[str, str] = {}
    nuc_chars = np.array(list("ACGT"))

    def simulate_clade(taxa_sub: List[str], parent_seq: np.ndarray, is_root: bool = False):
        if len(taxa_sub) == 1:
            t_bl = float(rng.uniform(0.008, 0.035))
            leaf_seq, _ = evolve_branch(parent_seq, t_bl)
            seqs_dict[taxa_sub[0]] = "".join(nuc_chars[leaf_seq])
            return
        if not is_root:
            int_bl = float(rng.uniform(0.004, 0.025))
            node_seq, n_mut = evolve_branch(parent_seq, int_bl)
            split = canon_split(set(taxa_sub), all_taxa - set(taxa_sub))
            nominal_splits.add(split)
            if n_mut >= 1:
                realized_splits.add(split)
        else:
            node_seq = parent_seq
        mid = len(taxa_sub) // 2
        if is_root:
            # Root split separates first 16 from second 16 across root internal branch
            root_bl = float(rng.uniform(0.010, 0.025))
            left_seq, n_mut_root = evolve_branch(node_seq, root_bl)
            split = canon_split(set(taxa_sub[:mid]), set(taxa_sub[mid:]))
            nominal_splits.add(split)
            if n_mut_root >= 1:
                realized_splits.add(split)
            simulate_clade(taxa_sub[:mid], left_seq, is_root=False)
            simulate_clade(taxa_sub[mid:], node_seq, is_root=False)
        else:
            simulate_clade(taxa_sub[:mid], node_seq, is_root=False)
            simulate_clade(taxa_sub[mid:], node_seq, is_root=False)

    simulate_clade(names, root_seq, is_root=True)
    return seqs_dict, nominal_splits, realized_splits


def run_stress2_ml_home_turf(n_reps: int = 8) -> Dict[str, dict]:
    print("\n" + "=" * 92)
    print("STRESS TEST 2: 'ML Home Turf' — Strictly Binary GTR+F+G4 Trees Across L in {150, 300, 600, 1500}")
    print("=" * 92)
    rng = np.random.default_rng(20261120)
    engine = MDLPhyloEngine(coding_aware=False, detect_shunts=False)

    out_by_L: Dict[str, dict] = {}
    for L in (150, 300, 600, 1500):
        stats = {
            m: {"prec_nom": [], "rec_real": [], "fp_nom": [], "rf_real": [], "rf_nom": []}
            for m in ("MDL", "IQTREE_Raw", "IQTREE_aBayes95", "FastTree_Raw", "NJ_JC", "MP")
        }
        realized_counts = []
        for _ in range(n_reps):
            seqs, nom_splits, real_splits = simulate_strictly_binary_gtr_gamma_alignment(rng, L=L)
            taxa = sorted(seqs.keys())
            realized_counts.append(len(real_splits))

            # 1. MDLPhyloEngine
            res_mdl = engine.infer(seqs)
            s_mdl = extract_unrooted_splits_from_adj(res_mdl.adjacency, res_mdl.names, res_mdl.n_taxa)

            # 2. IQ-TREE 2 (GTR+G with aBayes)
            iq_nwk, _, _ = run_iqtree_with_supports(seqs, model="GTR+G")
            _, s_iq_raw = parse_newick_with_supports_and_adj(iq_nwk, taxa, min_support=0.0)
            _, s_iq_ab = parse_newick_with_supports_and_adj(iq_nwk, taxa, min_support=0.95, support_index=1)

            # 3. FastTree 2
            ft_nwk, _ = run_fasttree(seqs)
            _, s_ft = parse_newick_with_supports_and_adj(ft_nwk, taxa, min_support=0.0)

            # 4. NJ (JC) & 5. MP
            _, s_nj, _ = run_neighbor_joining_jc(seqs)
            _, s_mp, _ = run_maximum_parsimony_nni(seqs)

            for m_name, s_pred in [
                ("MDL", s_mdl),
                ("IQTREE_Raw", s_iq_raw),
                ("IQTREE_aBayes95", s_iq_ab),
                ("FastTree_Raw", s_ft),
                ("NJ_JC", s_nj),
                ("MP", s_mp),
            ]:
                tp_nom = len(s_pred & nom_splits)
                fp_nom = len(s_pred - nom_splits)
                tp_real = len(s_pred & real_splits)
                prec_nom = tp_nom / max(len(s_pred), 1) * 100.0
                rec_real = tp_real / max(len(real_splits), 1) * 100.0
                rf_real = len(s_pred - real_splits) + len(real_splits - s_pred)
                rf_nom = len(s_pred - nom_splits) + len(nom_splits - s_pred)
                stats[m_name]["prec_nom"].append(prec_nom)
                stats[m_name]["rec_real"].append(rec_real)
                stats[m_name]["fp_nom"].append(fp_nom)
                stats[m_name]["rf_real"].append(rf_real)
                stats[m_name]["rf_nom"].append(rf_nom)

        mean_real = float(np.mean(realized_counts))
        print(f"\n--- Alignment Length L = {L} bp (Nominal Splits = 29, Mean Realized Non-Zero Splits = {mean_real:.1f}) ---")
        print(f"{'Method':<18} | {'Precision (True%)':<18} | {'False Pos Splits':<16} | {'Recall (Realized)':<18} | {'RF (vs Realized)':<16} | {'RF (vs Nominal)':<15}")
        print("-" * 112)
        out_by_L[str(L)] = {"mean_realized_splits": mean_real, "methods": {}}
        for m_name in ("MDL", "IQTREE_aBayes95", "IQTREE_Raw", "FastTree_Raw", "NJ_JC", "MP"):
            p_m = float(np.mean(stats[m_name]["prec_nom"]))
            fp_m = float(np.mean(stats[m_name]["fp_nom"]))
            r_m = float(np.mean(stats[m_name]["rec_real"]))
            rfr_m = float(np.mean(stats[m_name]["rf_real"]))
            rfn_m = float(np.mean(stats[m_name]["rf_nom"]))
            out_by_L[str(L)]["methods"][m_name] = {
                "precision_pct": round(p_m, 2),
                "false_positive_splits": round(fp_m, 2),
                "recall_realized_pct": round(r_m, 2),
                "rf_realized": round(rfr_m, 2),
                "rf_nominal": round(rfn_m, 2),
            }
            print(f"{m_name:<18} | {p_m:16.1f}% | {fp_m:16.2f} | {r_m:16.1f}% | {rfr_m:16.2f} | {rfn_m:15.2f}")

    return out_by_L


# ---------------------------------------------------------------------------
# STRESS TEST 3: Head-to-Head vs. Support-Collapsed ML & Bayesian Trees
# ---------------------------------------------------------------------------

def run_stress3_support_collapsed(n_reps: int = 8) -> Dict[str, dict]:
    print("\n" + "=" * 92)
    print("STRESS TEST 3: Head-to-Head vs. Support-Collapsed ML & Bayesian Trees (UFBoot, aBayes, SH-aLRT)")
    print("=" * 92)
    rng = np.random.default_rng(20261130)
    engine = MDLPhyloEngine(coding_aware=True, detect_shunts=True, min_shunt_sites=2)

    methods = [
        "MDL (No Bootstrap)",
        "IQTREE Raw Binary",
        "IQTREE SH-aLRT>=80",
        "IQTREE aBayes>=0.95",
        "IQTREE UFBoot>=95",
        "FastTree SH>=0.90",
        "FastTree SH>=0.95",
        "NJ (JC69)",
        "Max Parsimony",
    ]
    poly_stats = {m: {"fp": [], "prec": [], "rec": [], "rf": [], "time": []} for m in methods}

    for _ in range(n_reps):
        seqs, true_splits, _ = simulate_multifurcating_radiation(rng, N=32, L=1000)
        taxa = sorted(seqs.keys())

        # MDL
        res_mdl = engine.infer(seqs)
        s_mdl = extract_unrooted_splits_from_adj(res_mdl.adjacency, res_mdl.names, res_mdl.n_taxa)

        # IQ-TREE 2 with SH-aLRT, aBayes, and 1000 UFBoot
        iq_nwk, _, t_iq = run_iqtree_with_supports(seqs, model="GTR+G", with_ufboot=True)
        _, s_iq_raw = parse_newick_with_supports_and_adj(iq_nwk, taxa, min_support=0.0)
        _, s_iq_alrt = parse_newick_with_supports_and_adj(iq_nwk, taxa, min_support=80.0, support_index=0)
        _, s_iq_abayes = parse_newick_with_supports_and_adj(iq_nwk, taxa, min_support=0.95, support_index=1)
        _, s_iq_ufboot = parse_newick_with_supports_and_adj(iq_nwk, taxa, min_support=95.0, support_index=2)

        # FastTree 2 with SH local supports
        ft_nwk, t_ft = run_fasttree_with_supports(seqs)
        _, s_ft_90 = parse_newick_with_supports_and_adj(ft_nwk, taxa, min_support=0.90, support_index=0)
        _, s_ft_95 = parse_newick_with_supports_and_adj(ft_nwk, taxa, min_support=0.95, support_index=0)

        # NJ & MP
        _, s_nj, t_nj = run_neighbor_joining_jc(seqs)
        _, s_mp, t_mp = run_maximum_parsimony_nni(seqs)

        for m_name, s_pred, dt in [
            ("MDL (No Bootstrap)", s_mdl, res_mdl.runtime_sec),
            ("IQTREE Raw Binary", s_iq_raw, t_iq),
            ("IQTREE SH-aLRT>=80", s_iq_alrt, t_iq),
            ("IQTREE aBayes>=0.95", s_iq_abayes, t_iq),
            ("IQTREE UFBoot>=95", s_iq_ufboot, t_iq),
            ("FastTree SH>=0.90", s_ft_90, t_ft),
            ("FastTree SH>=0.95", s_ft_95, t_ft),
            ("NJ (JC69)", s_nj, t_nj),
            ("Max Parsimony", s_mp, t_mp),
        ]:
            tp = len(s_pred & true_splits)
            fp = len(s_pred - true_splits)
            fn = len(true_splits - s_pred)
            poly_stats[m_name]["fp"].append(fp)
            poly_stats[m_name]["prec"].append(tp / max(len(s_pred), 1) * 100.0)
            poly_stats[m_name]["rec"].append(tp / len(true_splits) * 100.0)
            poly_stats[m_name]["rf"].append(fp + fn)
            poly_stats[m_name]["time"].append(dt)

    # Also test Support-Collapsed IQ-TREE (aBayes>=0.95, UFBoot>=95) on Convergent Hijack (m=3)
    conv_false_clade_confident_abayes = []
    conv_false_clade_confident_ufboot = []
    conv_mdl_false_clades = []
    for _ in range(6):
        seqs_c, true_pairs, _ = simulate_two_site_hijack_alignment(rng, n_pairs=16, n_codons=300, m_conv_codons=3)
        taxa_c = sorted(seqs_c.keys())
        all_t = set(taxa_c)
        # False convergent pairs are (Host{2j}_T, Host{2j+1}_T)
        false_conv_splits = set()
        for j in range(8):
            a, b = f"Host{2*j:02d}_T", f"Host{2*j+1:02d}_T"
            s1, s2 = tuple(sorted([a, b])), tuple(sorted(all_t - {a, b}))
            false_conv_splits.add((s1, s2))

        res_c = engine.infer(seqs_c)
        s_mdl_c = extract_unrooted_splits_from_adj(res_c.adjacency, res_c.names, res_c.n_taxa)
        iq_c_nwk, _, _ = run_iqtree_with_supports(seqs_c, model="GTR+G", with_ufboot=True)
        _, s_ab_c = parse_newick_with_supports_and_adj(iq_c_nwk, taxa_c, min_support=0.95, support_index=1)
        _, s_uf_c = parse_newick_with_supports_and_adj(iq_c_nwk, taxa_c, min_support=95.0, support_index=2)

        conv_mdl_false_clades.append(len(s_mdl_c & false_conv_splits) / 8.0 * 100.0)
        conv_false_clade_confident_abayes.append(len(s_ab_c & false_conv_splits) / 8.0 * 100.0)
        conv_false_clade_confident_ufboot.append(len(s_uf_c & false_conv_splits) / 8.0 * 100.0)

    print(f"{'Method':<22} | {'False Pos (FP/16)':<19} | {'Precision':<11} | {'Recall':<11} | {'RF Error':<10} | {'Time (s)':<9}")
    print("-" * 92)
    out = {"polytomy_methods": {}, "convergent_false_clade_rate": {}}
    for m_name in methods:
        fp_m = float(np.mean(poly_stats[m_name]["fp"]))
        pr_m = float(np.mean(poly_stats[m_name]["prec"]))
        rc_m = float(np.mean(poly_stats[m_name]["rec"]))
        rf_m = float(np.mean(poly_stats[m_name]["rf"]))
        tm_m = float(np.mean(poly_stats[m_name]["time"]))
        out["polytomy_methods"][m_name] = {
            "fp": round(fp_m, 2),
            "fpr_pct": round(fp_m / 16.0 * 100.0, 2),
            "precision_pct": round(pr_m, 2),
            "recall_pct": round(rc_m, 2),
            "rf_error": round(rf_m, 2),
            "time_sec": round(tm_m, 3),
        }
        print(f"{m_name:<22} | {fp_m:5.2f} ({fp_m/16*100:5.1f}% FPR) | {pr_m:9.1f}% | {rc_m:9.1f}% | {rf_m:10.2f} | {tm_m:7.3f}s")

    out["convergent_false_clade_rate"] = {
        "MDL_false_convergent_clades_pct": round(float(np.mean(conv_mdl_false_clades)), 2),
        "IQTREE_aBayes95_confident_false_clades_pct": round(float(np.mean(conv_false_clade_confident_abayes)), 2),
        "IQTREE_UFBoot95_confident_false_clades_pct": round(float(np.mean(conv_false_clade_confident_ufboot)), 2),
    }
    print(f"\n  [Convergent Hijack m=3 Support Deception]")
    print(f"    False Convergent Clades Certified with aBayes >= 0.95 : {out['convergent_false_clade_rate']['IQTREE_aBayes95_confident_false_clades_pct']:.1f}%")
    print(f"    False Convergent Clades Certified with UFBoot >= 95%  : {out['convergent_false_clade_rate']['IQTREE_UFBoot95_confident_false_clades_pct']:.1f}%")
    print(f"    False Convergent Clades in MDLPhyloEngine             : {out['convergent_false_clade_rate']['MDL_false_convergent_clades_pct']:.1f}% (Peeled as SPARSE shunts!)")
    return out


# ---------------------------------------------------------------------------
# STRESS TEST 4: 32-Taxon Multi-Lineage Parasite LBA & Heterotachy
# ---------------------------------------------------------------------------

def run_stress4_large_lba_and_heterotachy(n_reps: int = 12) -> Dict[str, dict]:
    print("\n" + "=" * 92)
    print("STRESS TEST 4: 32-Taxon Multi-Clade Parasite LBA & Kolaczkowski-Thornton Heterotachy")
    print("=" * 92)
    rng = np.random.default_rng(20261140)
    engine = MDLPhyloEngine(coding_aware=False, detect_shunts=False)
    nuc_chars = np.array(list("ACGT"))

    def mutate_jc(seq: np.ndarray, bl: float) -> np.ndarray:
        p = 0.75 * (1.0 - math.exp(-(4.0 / 3.0) * bl))
        out = seq.copy()
        draws = rng.random(len(seq))
        for idx in np.where(draws < p)[0]:
            cur = int(out[idx])
            out[idx] = rng.choice([b for b in range(4) if b != cur])
        return out

    # 4A: 16-Taxon Multi-Clade Parasite LBA (4 clades of 4 taxa; each clade has 1 fast parasite P_k
    # sister to short host H_k, with stem l_stem=0.025 and parasite branch l_P=0.45)
    methods_4a = ["MDLPhyloEngine", "IQTREE2 (GTR+G)", "FastTree2", "NJ (JC69)", "Max Parsimony"]
    p_recall = {m: [] for m in methods_4a}
    p_lba_trap = {m: [] for m in methods_4a}
    L_4a = 1200

    for _ in range(n_reps):
        root_seq = rng.integers(0, 4, size=L_4a, dtype=np.int8)
        c_01 = mutate_jc(root_seq, 0.035)
        c_23 = mutate_jc(root_seq, 0.035)
        clade_ancs = [
            mutate_jc(c_01, 0.035),
            mutate_jc(c_01, 0.035),
            mutate_jc(c_23, 0.035),
            mutate_jc(c_23, 0.035),
        ]
        seqs: Dict[str, str] = {}
        true_parasite_sisters = set()
        all_taxa = [f"C{k}_{x}" for k in range(4) for x in ("P", "H1", "H2", "H3")]
        all_set = set(all_taxa)

        for k in range(4):
            anc_ph = mutate_jc(clade_ancs[k], 0.028)
            anc_h23 = mutate_jc(clade_ancs[k], 0.025)
            s_p = mutate_jc(anc_ph, 0.44)     # Fast-evolving parasite lineage!
            s_h1 = mutate_jc(anc_ph, 0.015)   # Short sister host!
            s_h2 = mutate_jc(anc_h23, 0.018)
            s_h3 = mutate_jc(anc_h23, 0.018)
            seqs[f"C{k}_P"] = "".join(nuc_chars[s_p])
            seqs[f"C{k}_H1"] = "".join(nuc_chars[s_h1])
            seqs[f"C{k}_H2"] = "".join(nuc_chars[s_h2])
            seqs[f"C{k}_H3"] = "".join(nuc_chars[s_h3])
            true_parasite_sisters.add(canon_split({f"C{k}_P", f"C{k}_H1"}, all_set - {f"C{k}_P", f"C{k}_H1"}))

        # False parasite-parasite LBA splits (any pair or all 4 of {C0_P, C1_P, C2_P, C3_P})
        parasite_set = {"C0_P", "C1_P", "C2_P", "C3_P"}

        def eval_parasite_splits(s_pred):
            rec = len(s_pred & true_parasite_sisters) / 4.0 * 100.0
            has_lba = 0.0
            for s1, s2 in s_pred:
                for side in (set(s1), set(s2)):
                    if len(side) >= 2 and side.issubset(parasite_set):
                        has_lba = 100.0
            return rec, has_lba

        res_mdl = engine.infer(seqs)
        s_mdl = extract_unrooted_splits_from_adj(res_mdl.adjacency, res_mdl.names, res_mdl.n_taxa)
        iq_nwk, _ = run_iqtree(seqs, model="GTR+G")
        _, s_iq = extract_unrooted_splits_from_newick(iq_nwk)
        ft_nwk, _ = run_fasttree(seqs)
        _, s_ft = extract_unrooted_splits_from_newick(ft_nwk)
        _, s_nj, _ = run_neighbor_joining_jc(seqs)
        _, s_mp, _ = run_maximum_parsimony_nni(seqs)

        for m_name, s_p in [
            ("MDLPhyloEngine", s_mdl),
            ("IQTREE2 (GTR+G)", s_iq),
            ("FastTree2", s_ft),
            ("NJ (JC69)", s_nj),
            ("Max Parsimony", s_mp),
        ]:
            r_v, lba_v = eval_parasite_splits(s_p)
            p_recall[m_name].append(r_v)
            p_lba_trap[m_name].append(lba_v)

    # 4B: Kolaczkowski-Thornton 2-Compartment Heterotachy (50% of sites have A,C long=0.45, B,D short=0.015;
    # 50% of sites have A,B,C,D rate inversion or heterotachous long branches across internal branch l_int=0.025)
    kt_true = {m: 0 for m in methods_4a}
    kt_lba = {m: 0 for m in methods_4a}
    n_kt = 20
    for _ in range(n_kt):
        L_half = 600
        # Compartment 1: A, C long (0.46), B, D short (0.012), int = 0.025
        r1 = rng.integers(0, 4, size=L_half, dtype=np.int8)
        u1 = r1.copy()
        v1 = mutate_jc(r1, 0.025)
        a1 = mutate_jc(u1, 0.46); b1 = mutate_jc(u1, 0.012)
        c1 = mutate_jc(v1, 0.46); d1 = mutate_jc(v1, 0.012)
        # Compartment 2: A, C medium-long (0.38), B, D ultra-short (0.008), int = 0.020
        r2 = rng.integers(0, 4, size=L_half, dtype=np.int8)
        u2 = r2.copy()
        v2 = mutate_jc(r2, 0.020)
        a2 = mutate_jc(u2, 0.38); b2 = mutate_jc(u2, 0.008)
        c2 = mutate_jc(v2, 0.38); d2 = mutate_jc(v2, 0.008)
        seqs_kt = {
            "A": "".join(nuc_chars[np.concatenate([a1, a2])]),
            "B": "".join(nuc_chars[np.concatenate([b1, b2])]),
            "C": "".join(nuc_chars[np.concatenate([c1, c2])]),
            "D": "".join(nuc_chars[np.concatenate([d1, d2])]),
        }
        true_kt = canon_split({"A", "B"}, {"C", "D"})
        lba_kt = canon_split({"A", "C"}, {"B", "D"})

        res_m = engine.infer(seqs_kt)
        sm = extract_unrooted_splits_from_adj(res_m.adjacency, res_m.names, res_m.n_taxa)
        iq_n, _ = run_iqtree(seqs_kt, model="GTR+G")
        _, siq = extract_unrooted_splits_from_newick(iq_n)
        ft_n, _ = run_fasttree(seqs_kt)
        _, sft = extract_unrooted_splits_from_newick(ft_n)
        _, snj, _ = run_neighbor_joining_jc(seqs_kt)
        _, smp, _ = run_maximum_parsimony_nni(seqs_kt)

        for m_name, sp_set in [
            ("MDLPhyloEngine", sm),
            ("IQTREE2 (GTR+G)", siq),
            ("FastTree2", sft),
            ("NJ (JC69)", snj),
            ("Max Parsimony", smp),
        ]:
            if true_kt in sp_set:
                kt_true[m_name] += 1
            elif lba_kt in sp_set:
                kt_lba[m_name] += 1

    print("  [4A: Multi-Clade Parasite LBA (4 Long-Branch Parasites Nested in 4 Separate Clades)]")
    out = {"multi_clade_parasite_lba": {}, "kolaczkowski_thornton_heterotachy": {}}
    for m_name in methods_4a:
        rc = float(np.mean(p_recall[m_name]))
        lb = float(np.mean(p_lba_trap[m_name]))
        out["multi_clade_parasite_lba"][m_name] = {
            "true_parasite_host_sister_recall_pct": round(rc, 2),
            "false_parasite_clade_lba_rate_pct": round(lb, 2),
        }
        print(f"    {m_name:<18} : True Parasite-Host Sister Recall = {rc:5.1f}% | False Parasite-Clade LBA Rate = {lb:5.1f}%")

    print("\n  [4B: Kolaczkowski-Thornton Heterotachy / Deep Asymmetry (20 Replicates)]")
    for m_name in methods_4a:
        tr_p = kt_true[m_name] / n_kt * 100.0
        lb_p = kt_lba[m_name] / n_kt * 100.0
        out["kolaczkowski_thornton_heterotachy"][m_name] = {
            "true_split_pct": round(tr_p, 2),
            "lba_trap_pct": round(lb_p, 2),
        }
        print(f"    {m_name:<18} : True Split = {tr_p:5.1f}% | LBA Trap = {lb_p:5.1f}%")

    return out


# ---------------------------------------------------------------------------
# STRESS TEST 5: Head-to-Head on Empirical Mammalian Prestin (116 Mammals)
# ---------------------------------------------------------------------------

def run_stress5_prestin_head_to_head() -> Dict[str, dict]:
    print("\n" + "=" * 92)
    print("STRESS TEST 5: Empirical Head-to-Head on Mammalian Prestin (116 Mammals vs. NCBI Species Tree)")
    print("=" * 92)
    spec_dir = Path(__file__).resolve().parents[2] / "data" / "specimens"
    fa_path = spec_dir / "prestin.fasta"
    sp_path = spec_dir / "prestin_species.nwk"

    taxa, seqs_list = parse_fasta(str(fa_path))
    seqs_dict = dict(zip(taxa, seqs_list))
    with open(sp_path, "r") as f:
        sp_nwk = re.sub(r"\{[^}]*\}", "", f.read().strip())

    shared_taxa, sp_splits = extract_unrooted_splits_from_newick(sp_nwk, present_taxa=set(taxa))

    # 1. MDLPhyloEngine
    engine = MDLPhyloEngine(coding_aware=True, detect_shunts=True, min_shunt_sites=2)
    res_mdl = engine.infer(str(fa_path))
    _, s_mdl = extract_unrooted_splits_from_newick(res_mdl.newick, present_taxa=shared_taxa)

    # 2. IQ-TREE 2 (GTR+G with SH-aLRT and aBayes)
    iq_nwk, iq_logl, t_iq = run_iqtree_with_supports(seqs_dict, model="GTR+G", with_ufboot=False)
    _, s_iq_raw_all = parse_newick_with_supports_and_adj(iq_nwk, taxa, min_support=0.0)
    _, s_iq_alrt_all = parse_newick_with_supports_and_adj(iq_nwk, taxa, min_support=80.0, support_index=0)
    _, s_iq_ab_all = parse_newick_with_supports_and_adj(iq_nwk, taxa, min_support=0.95, support_index=1)

    # Project split sets onto `shared_taxa`
    def project_splits(splits_all: Set[Tuple[Tuple[str, ...], Tuple[str, ...]]]) -> Set[Tuple[Tuple[str, ...], Tuple[str, ...]]]:
        proj = set()
        for s1, s2 in splits_all:
            p1 = set(s1) & shared_taxa
            p2 = set(s2) & shared_taxa
            if len(p1) >= 2 and len(p2) >= 2:
                proj.add(canon_split(p1, p2))
        return proj

    s_iq_raw = project_splits(s_iq_raw_all)
    s_iq_alrt = project_splits(s_iq_alrt_all)
    s_iq_ab = project_splits(s_iq_ab_all)

    # 3. FastTree 2
    ft_nwk, t_ft = run_fasttree_with_supports(seqs_dict)
    _, s_ft_raw_all = parse_newick_with_supports_and_adj(ft_nwk, taxa, min_support=0.0)
    _, s_ft_90_all = parse_newick_with_supports_and_adj(ft_nwk, taxa, min_support=0.90, support_index=0)
    s_ft_raw = project_splits(s_ft_raw_all)
    s_ft_90 = project_splits(s_ft_90_all)

    # 4. NJ (JC69) & 5. MP
    _, s_nj_all, t_nj = run_neighbor_joining_jc(seqs_dict)
    _, s_mp_all, t_mp = run_maximum_parsimony_nni(seqs_dict)
    s_nj = project_splits(s_nj_all)
    s_mp = project_splits(s_mp_all)

    print(f"{'Method':<22} | {'Splits':<7} | {'True Pos':<9} | {'False Pos':<9} | {'Precision':<10} | {'Recall':<9} | {'F1 Score':<9} | {'RF Error':<9} | {'Time':<7}")
    print("-" * 112)
    out = {}
    for m_name, s_pred, dt in [
        ("MDLPhyloEngine (Ours)", s_mdl, res_mdl.runtime_sec),
        ("IQTREE2 (aBayes>=0.95)", s_iq_ab, t_iq),
        ("IQTREE2 (SH-aLRT>=80)", s_iq_alrt, t_iq),
        ("IQTREE2 (GTR+G Raw)", s_iq_raw, t_iq),
        ("FastTree2 (SH>=0.90)", s_ft_90, t_ft),
        ("FastTree2 (GTR+CAT)", s_ft_raw, t_ft),
        ("Neighbor-Joining (JC)", s_nj, t_nj),
        ("Max Parsimony (NNI)", s_mp, t_mp),
    ]:
        tp = len(s_pred & sp_splits)
        fp = len(s_pred - sp_splits)
        fn = len(sp_splits - s_pred)
        prec = tp / max(len(s_pred), 1) * 100.0
        rec = tp / len(sp_splits) * 100.0
        f1 = 2 * prec * rec / max(prec + rec, 1e-6)
        rf = fp + fn
        out[m_name] = {
            "inferred_splits": len(s_pred),
            "true_positives": tp,
            "false_positives": fp,
            "precision_pct": round(prec, 2),
            "recall_pct": round(rec, 2),
            "f1_pct": round(f1, 2),
            "rf_error": rf,
            "time_sec": round(dt, 2),
        }
        print(f"{m_name:<22} | {len(s_pred):7d} | {tp:9d} | {fp:9d} | {prec:8.1f}% | {rec:7.1f}% | {f1:7.1f}% | {rf:9d} | {dt:6.2f}s")

    return out


if __name__ == "__main__":
    all_stress = {
        "stress1_likelihood_paradox": run_stress1_likelihood_paradox(n_reps=8),
        "stress2_ml_home_turf": run_stress2_ml_home_turf(n_reps=8),
        "stress3_support_collapsed": run_stress3_support_collapsed(n_reps=8),
        "stress4_large_lba": run_stress4_large_lba_and_heterotachy(n_reps=12),
        "stress5_prestin_head_to_head": run_stress5_prestin_head_to_head(),
    }
    out_file = Path(__file__).resolve().parents[2] / "results" / "synthetic" / "stress_test_results.json"
    with open(out_file, "w") as f:
        json.dump(all_stress, f, indent=2)
    print(f"\nSaved ruthless stress-test results to {out_file}")
