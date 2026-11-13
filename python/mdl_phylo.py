#!/usr/bin/env python3
"""
BUSHWHACKER: Bipartition Unification, Shunt Hunting, & Weighted Hub Aggregation
via Combinadic Kolmogorov Entropy Reduction (bushwhacker.py / mdl_phylo.py)
===============================================================================
First-principles phylogenetic inference via Minimum Description Length (MDL)
and Algorithmic Information Theory.

Instead of forcing sequence alignments into a single binary tree of (2N-3)
branches via combinatorial hill-climbing, this engine treats a phylogeny as a
lossless compression program G = (V, E_tree, E_shunt) for the alignment X:

    L_total(X, G) = L_topo(T_0) + L_shunt(E_shunt) + L_data(X | G)

Key Emergent Properties (Zero Heuristic Thresholds):
  1. Native Polytomies (Zero-Mutation Exclusion Law):
     An internal branch e separating sub-degrees (a, b) inside a (a+b)-way node
     costs L_topo(e) = ln C(a+b, a) + ln(a+b-1) nats. Any branch whose data
     compression gain fails to pay L_topo(e) (Delta L_MDL >= 0) is contracted
     into an exact multi-way polytomy.
  2. Birthday-Paradox Immunity to Long-Branch Attraction (Felsenstein Zone):
     Encoding m_e mutations out of L sites on branch e costs ln C(L, m_e) nats.
     Pulling m_syn shared mutations off two branches (m_1, m_2) onto a shared
     ancestral branch saves m_syn * ln(m_syn / (e * m_1 * m_2 / L)) nats,
     automatically penalizing random homoplastic collisions on long branches
     while recovering masked synapomorphies (G, G, T, A) via MDL polarization.
  3. Dual-Mode Lateral Shunts (Recombination vs. Functional Convergence):
     Residual parallel mutations between non-sister branches (u, v) are peeled
     into lateral shunt edges if and only if they compress L_total after paying
     endpoint cost (2 ln |V|) plus the minimum of two universal site-mask codes:
       - INTERVAL code: ln 2 + 2 ln L  (Homologous Recombination / HGT)
       - SPARSE code:   ln 2 + ln C(L_eff, m)  (Convergent / Epistatic Selection)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.special import gammaln


# ---------------------------------------------------------------------------
# Combinatorial Information-Theoretic Primitives (All in Nats)
# ---------------------------------------------------------------------------

def log_binom(n: float, k: float) -> float:
    """Continuous exact log-binomial coefficient ln C(n, k) via gammaln."""
    if k <= 0.0 or k >= n or n <= 0.0:
        return 0.0
    return float(gammaln(n + 1.0) - gammaln(k + 1.0) - gammaln(n - k + 1.0))


def log_binom_vec(n: float, k_arr: np.ndarray) -> np.ndarray:
    """Vectorized log-binomial coefficient ln C(n, k) for array k_arr."""
    k_clamped = np.clip(k_arr, 0.0, n)
    res = gammaln(n + 1.0) - gammaln(k_clamped + 1.0) - gammaln(n - k_clamped + 1.0)
    return np.where((k_arr <= 0.0) | (k_arr >= n), 0.0, res)


def universal_int_nats(m: float) -> float:
    """
    Exact parameter-free Kraft-tight universal integer prefix code length (in nats)
    for non-negative integer m via telescoping logarithmic tail:
      P(0) = 1/2, P(m) = (ln 2 / 2) * (1 / ln(m + 1) - 1 / ln(m + 2)) for m >= 1.
    """
    ln2 = math.log(2.0)
    if m <= 0.0:
        return ln2
    s1 = 0.5
    s2 = 0.5 * ln2 / math.log(3.0)
    p1 = s1 - s2
    if m < 1.0:
        return (1.0 - m) * ln2 + m * (-math.log(p1))
    sm = 0.5 * ln2 / math.log(m + 1.0)
    smp1 = 0.5 * ln2 / math.log(m + 2.0)
    return -math.log(max(sm - smp1, 1e-300))


def universal_int_nats_vec(m_arr: np.ndarray) -> np.ndarray:
    """Vectorized exact parameter-free Kraft-tight universal integer code length."""
    m = np.asarray(m_arr, dtype=np.float64)
    ln2 = math.log(2.0)
    p1 = 0.5 - 0.5 * ln2 / math.log(3.0)
    c1 = -math.log(p1)
    m_safe = np.maximum(m, 1.0)
    sm = 0.5 * ln2 / np.log(m_safe + 1.0)
    smp1 = 0.5 * ln2 / np.log(m_safe + 2.0)
    tail_code = -np.log(np.maximum(sm - smp1, 1e-300))
    frac_code = (1.0 - m) * ln2 + m * c1
    return np.where(m <= 0.0, ln2, np.where(m < 1.0, frac_code, tail_code))


def branch_mutation_code_nats(
    m_e: float,
    L_eff: float,
    subst_nats: float = 0.0,
    alphabet_size: int = 4,
) -> float:
    """
    Exact enumerative code length (in nats) to transmit m_e mutations along a
    single branch e over L_eff effective sites:
      1. Universal integer header for mutation count m_e (ln 2 at m_e = 0)
      2. Combinatorial site-address code ln C(L_eff, m_e) (monotonically clamped at 0.49 * L_eff)
      3. State transition code sum_{s in M_e} l_subst(s)
    """
    if m_e <= 1e-9:
        return universal_int_nats(0.0)
    m_pos = min(m_e, 0.49 * L_eff)
    header = universal_int_nats(m_e)
    site_positions = log_binom(L_eff, m_pos)
    if m_e > 0.49 * L_eff:
        # Beyond 0.49 * L_eff, site entropy saturates at ln(2) per site
        site_positions += (m_e - 0.49 * L_eff) * math.log(2.0)
    default_state_cost = m_e * math.log(max(alphabet_size - 1, 1))
    state_cost = subst_nats if subst_nats > 0.0 else default_state_cost
    return header + site_positions + state_cost


def split_topology_cost_nats(deg_a: int, deg_b: int) -> float:
    """
    Exact combinatorial cost (in nats) of resolving a k = (deg_a + deg_b)-way
    polytomy into two sub-nodes of outward degrees deg_a >= 2 and deg_b >= 2:
      - Specify size min(deg_a, deg_b) in {2, ..., floor(k/2)}: ln(max(1, floor(k/2) - 1))
      - Specify subset of size deg_a out of k branches: ln C(k, deg_a) - (ln 2 if deg_a == deg_b)
    """
    k = deg_a + deg_b
    if k < 4 or deg_a < 2 or deg_b < 2:
        return 0.0
    num_sizes = max(1, (k // 2) - 1)
    subset_cost = log_binom(k, deg_a)
    if deg_a == deg_b:
        subset_cost -= math.log(2.0)
    return math.log(num_sizes) + subset_cost


# ---------------------------------------------------------------------------
# Data Structures for the MDL Phylogenetic Compression Graph
# ---------------------------------------------------------------------------

@dataclass
class ShuntEdge:
    """A certified lateral shunt edge (recombination interval or convergent point-set)."""
    node_u: int
    node_v: int
    label_u: str
    label_v: str
    mode: str  # "INTERVAL" (recombination/HGT) or "SPARSE" (convergence/epistasis)
    sites: List[int]
    interval_span: Optional[Tuple[int, int]]
    shared_mutations: int
    data_savings_nats: float
    mask_cost_nats: float
    endpoint_cost_nats: float
    net_mdl_gain_nats: float
    posterior_prob: float


@dataclass
class BranchRecord:
    """Metadata and exact MDL accounting for a branch e = (u, v) in T_0."""
    u: int
    v: int
    is_internal: bool
    mutation_count: float
    branch_length: float
    mdl_gain_nats: float
    mdl_support: float
    synapomorphy_sites: List[int] = field(default_factory=list)


@dataclass
class MDLGraphResult:
    """Complete result of Kolmogorov / MDL Phylogenetic Graph Induction."""
    names: List[str]
    n_taxa: int
    n_sites: int
    L_eff: float
    newick: str
    adjacency: Dict[int, List[int]]
    node_labels: Dict[int, str]
    branches: List[BranchRecord]
    shunts: List[ShuntEdge]
    n_internal_branches: int
    max_binary_internal_branches: int
    polytomy_nodes: int
    max_polytomy_degree: int
    raw_uncompressed_nats: float
    star_tree_nats: float
    backbone_tree_nats: float
    full_graph_nats: float
    runtime_sec: float


# ---------------------------------------------------------------------------
# Alignment Parsing and Encoding Utilities
# ---------------------------------------------------------------------------

NUC_MAP = {
    ord("A"): 0, ord("a"): 0,
    ord("C"): 1, ord("c"): 1,
    ord("G"): 2, ord("g"): 2,
    ord("T"): 3, ord("t"): 3,
    ord("U"): 3, ord("u"): 3,
}

IS_PURINE = np.array([1, 0, 1, 0], dtype=np.int8)

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


def _finalize_alignment_py(names: List[str], seqs: List[str]) -> Tuple[List[str], List[str]]:
    if not names or not seqs:
        raise ValueError("No alignment records found.")
    max_len = max(len(s) for s in seqs)
    seqs = [s + "-" * (max_len - len(s)) if len(s) < max_len else s for s in seqs]
    ref = seqs[0]
    for i in range(1, len(seqs)):
        if "." in seqs[i]:
            seqs[i] = "".join(ref[j] if ch == "." else ch for j, ch in enumerate(seqs[i]))
    return names, seqs


def parse_phylip_text(text: str) -> Tuple[List[str], List[str]]:
    lines = [l.rstrip("\r\n") for l in text.lstrip("\ufeff").splitlines() if l.strip()]
    if not lines:
        raise ValueError("Empty PHYLIP input.")
    hdr = lines[0].split()
    ntax, nchar = int(hdr[0]), int(hdr[1])
    data_lines = lines[1:]

    def split_named(line: str) -> Tuple[str, str]:
        s = line.strip()
        parts = s.split(None, 1)
        if len(parts) == 2:
            if len(line) > 10 and len(parts[0]) > 10:
                strict_nm = line[:10].strip().replace(" ", "_")
                strict_sq = "".join(line[10:].split())
                if len(strict_sq) == nchar or len(parts[0]) - 10 == 10:
                    return strict_nm, strict_sq
            return parts[0], "".join(parts[1].split())
        return line[:10].strip().replace(" ", "_"), "".join(line[10:].split())

    if len(data_lines) == ntax:
        names, seqs = zip(*(split_named(l) for l in data_lines))
        return _finalize_alignment_py(list(names), list(seqs))

    # Try interleaved first
    try:
        names, seqs = [], []
        for i in range(ntax):
            nm, sq = split_named(data_lines[i])
            names.append(nm)
            seqs.append(sq)
        if len(set(len(s) for s in seqs)) == 1 and len(seqs[0]) > 0:
            for idx in range(ntax, len(data_lines)):
                t = idx % ntax
                raw = data_lines[idx].strip()
                if raw.startswith(names[t]) and len(raw) > len(names[t]) and raw[len(names[t])].isspace():
                    raw = raw[len(names[t]):]
                seqs[t] += "".join(raw.split())
            if all(len(s) == nchar for s in seqs):
                return _finalize_alignment_py(names, seqs)
    except Exception:
        pass

    # Sequential multi-line fallback
    names, seqs = [], []
    idx = 0
    for _ in range(ntax):
        nm, sq = split_named(data_lines[idx])
        idx += 1
        while len(sq) < nchar and idx < len(data_lines):
            sq += "".join(data_lines[idx].split())
            idx += 1
        names.append(nm)
        seqs.append(sq)
    return _finalize_alignment_py(names, seqs)


def parse_nexus_text(text: str) -> Tuple[List[str], List[str]]:
    import re
    # Strip [...] comments while preserving newlines
    out, depth, in_sq, in_dq = [], 0, False, False
    for ch in text.lstrip("\ufeff"):
        if depth > 0:
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
            elif ch == "\n":
                out.append("\n")
            continue
        if ch == "'" and not in_dq:
            in_sq = not in_sq
            out.append(ch)
        elif ch == '"' and not in_sq:
            in_dq = not in_dq
            out.append(ch)
        elif ch == "[" and not in_sq and not in_dq:
            depth = 1
        else:
            out.append(ch)
    clean = "".join(out)
    m = re.search(r"\bMATRIX\b([\s\S]*?);", clean, flags=re.IGNORECASE)
    if not m:
        raise ValueError("No MATRIX block found in NEXUS input.")
    body = m.group(1)
    names: List[str] = []
    seqs: List[str] = []
    name_to_idx: Dict[str, int] = {}
    for raw_line in body.splitlines():
        s = raw_line.strip()
        if not s:
            continue
        if s[0] in ("'", '"'):
            q = s[0]
            end_q = s.find(q, 1)
            if end_q != -1:
                nm = s[1:end_q].strip().replace(" ", "_")
                sq = "".join(s[end_q + 1:].split())
            else:
                continue
        else:
            parts = s.split(None, 1)
            if len(parts) < 2:
                if seqs:
                    seqs[-1] += "".join(s.split())
                continue
            nm, sq = parts[0], "".join(parts[1].split())
        if nm in name_to_idx:
            seqs[name_to_idx[nm]] += sq
        else:
            name_to_idx[nm] = len(names)
            names.append(nm)
            seqs.append(sq)
    return _finalize_alignment_py(names, seqs)


def parse_fasta(path_or_dict) -> Tuple[List[str], List[str]]:
    """Parse FASTA, PHYLIP, or NEXUS alignment file path or {name: seq} dictionary."""
    if isinstance(path_or_dict, dict):
        names = list(path_or_dict.keys())
        seqs = [path_or_dict[k] for k in names]
        return names, seqs
    with open(path_or_dict, "r", encoding="utf-8") as f:
        text = f.read()
    clean = text.lstrip("\ufeff")
    for line in clean.splitlines():
        s = line.strip()
        if not s or s.startswith(";"):
            continue
        up = s.upper()
        if up.startswith("#NEXUS") or up.startswith("BEGIN DATA") or up.startswith("BEGIN CHARACTERS"):
            return parse_nexus_text(clean)
        if s.startswith(">"):
            break
        toks = s.split()
        if len(toks) >= 2 and toks[0].isdigit() and toks[1].isdigit():
            return parse_phylip_text(clean)
        break
    names: List[str] = []
    seqs: List[str] = []
    curr_name: Optional[str] = None
    curr_chunks: List[str] = []
    for line in clean.splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith(">"):
            if curr_name is not None:
                names.append(curr_name)
                seqs.append("".join(curr_chunks))
            curr_name = line[1:].split()[0]
            curr_chunks = []
        else:
            curr_chunks.append("".join(line.split()))
    if curr_name is not None:
        names.append(curr_name)
        seqs.append("".join(curr_chunks))
    return _finalize_alignment_py(names, seqs)


def encode_nucleotide_alignment(seqs: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Encode N aligned nucleotide sequences of length L into:
      - mat: (N, L) int8 in {0, 1, 2, 3, -1 for gap/ambig}
      - bitmasks: (N, L) uint8 in {1, 2, 4, 8, 15 for gap/ambig}
      - majority_state: (L,) int8 majority nucleotide at each column (for clean tie-breaking)
    """
    N = len(seqs)
    L = len(seqs[0])
    mat = np.full((N, L), -1, dtype=np.int8)
    bitmasks = np.full((N, L), 15, dtype=np.uint8)
    lut_val = np.full(256, -1, dtype=np.int8)
    lut_bit = np.full(256, 15, dtype=np.uint8)
    for k, v in NUC_MAP.items():
        lut_val[k] = v
        lut_bit[k] = 1 << v
    for i, s in enumerate(seqs):
        arr = np.frombuffer(s.encode("ascii", errors="replace"), dtype=np.uint8)
        mat[i] = lut_val[arr]
        bitmasks[i] = lut_bit[arr]

    counts = np.zeros((4, L), dtype=np.int32)
    for b in range(4):
        counts[b] = np.sum(mat == b, axis=0)
    majority_state = np.argmax(counts, axis=0).astype(np.int8)
    return mat, bitmasks, majority_state


def detect_coding_frame_and_weights(
    seqs: List[str], mat: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Detect if the alignment is a protein-coding ORF and return:
      - dist_weights: (L,) weights for initial vertical distance matrix (d_S stabilized so
        convergent amino-acid replacements at 1st/2nd positions cannot hijack the backbone!)
      - tree_weights: (L,) bounded site weights for MDL tree contraction
      - is_coding: bool
    """
    N, L = mat.shape
    dist_weights = np.ones(L, dtype=np.float64)
    tree_weights = np.ones(L, dtype=np.float64)
    is_coding = False

    if L >= 90 and L % 3 == 0:
        n_codons = L // 3
        stop_count = 0
        valid_codons = 0
        for i in range(min(N, 32)):
            s = seqs[i].upper().replace("U", "T")
            for c in range(n_codons - 1):
                cod = s[3 * c : 3 * c + 3]
                aa = CODON_TABLE.get(cod)
                if aa is not None:
                    valid_codons += 1
                    if aa == "*":
                        stop_count += 1
        if valid_codons > 0 and (stop_count / valid_codons) < 0.02:
            is_coding = True
            for c in range(n_codons):
                cols = [3 * c, 3 * c + 1, 3 * c + 2]
                aas = set()
                cods = set()
                for i in range(N):
                    b0, b1, b2 = mat[i, cols[0]], mat[i, cols[1]], mat[i, cols[2]]
                    if b0 >= 0 and b1 >= 0 and b2 >= 0:
                        idx = (int(b0) << 4) | (int(b1) << 2) | int(b2)
                        cods.add(idx)
                        s_cod = "ACGT"[b0] + "ACGT"[b1] + "ACGT"[b2]
                        aa = CODON_TABLE.get(s_cod, "X")
                        aas.add(aa)
                if len(cods) > 1:
                    if len(aas) == 1:
                        # Purely synonymous polymorphism: gold-standard vertical genealogical clock
                        for col in cols:
                            dist_weights[col] = 2.0
                            tree_weights[col] = 1.25
                    else:
                        # Replacement polymorphism: downweight 1st/2nd positions in initial
                        # distance backbone so multi-nucleotide convergent codon replacements
                        # cannot hijack vertical sister pairings!
                        dist_weights[cols[0]] = 0.12
                        dist_weights[cols[1]] = 0.08
                        dist_weights[cols[2]] = 1.00
                        tree_weights[cols[0]] = 0.45
                        tree_weights[cols[1]] = 0.40
                        tree_weights[cols[2]] = 1.15
    return dist_weights, tree_weights, is_coding


# ---------------------------------------------------------------------------
# Core Engine: MDL Phylogenetic Graph Induction
# ---------------------------------------------------------------------------

class MDLPhyloEngine:
    """
    First-Principles Kolmogorov / MDL Phylogenetic Graph Induction Engine.
    """

    def __init__(
        self,
        coding_aware: bool = True,
        detect_shunts: bool = True,
        min_shunt_sites: int = 2,
    ):
        self.coding_aware = coding_aware
        self.detect_shunts = detect_shunts
        self.min_shunt_sites = min_shunt_sites

    def infer(self, alignment_input) -> MDLGraphResult:
        """
        Infer the MDL-optimal Phylogenetic Compression Graph G = (V, E_tree, E_shunt)
        directly from a FASTA alignment path or {name: seq} dict.
        """
        t0 = time.perf_counter()
        names, seqs = parse_fasta(alignment_input)
        N = len(names)
        if N == 0:
            raise ValueError("Empty alignment provided.")
        L = len(seqs[0])

        mat, bitmasks, majority_state = encode_nucleotide_alignment(seqs)
        valid_mask = mat >= 0  # (N, L)

        # Effective sequence length (mean non-gap sites per taxon)
        L_eff = float(np.clip(np.mean(np.sum(valid_mask, axis=1)), 1.0, float(L)))

        # Compute empirical transition/transversion log-surprise and site weights
        ti_cost, tv_cost = self._estimate_subst_surprise(mat, valid_mask)
        if self.coding_aware:
            dist_weights, tree_weights, is_coding = detect_coding_frame_and_weights(seqs, mat)
        else:
            dist_weights = np.ones(L, dtype=np.float64)
            tree_weights = np.ones(L, dtype=np.float64)
            is_coding = False

        # Tree-Free Dual-Space Four-Gamete Coherence & Allelic Multiplicity Weights
        dist_weights, tree_weights = self._compute_dual_site_coherence_weights(
            mat, valid_mask, dist_weights, tree_weights
        )

        # Step 1: Pairwise Hamming + Jukes-Cantor expected mutation counts & 3-point terminal loads
        diff_mat, valid_pairs, jc_mut_mat = self._compute_pairwise_mutation_matrices(
            mat, valid_mask, dist_weights, L_eff
        )
        # Also compute unweighted JC mutation matrix for accurate terminal load estimation
        _, _, unweighted_jc_mat = self._compute_pairwise_mutation_matrices(
            mat, valid_mask, np.ones(L, dtype=np.float64), L_eff
        )
        terminal_loads = self._estimate_three_point_terminal_loads(unweighted_jc_mat)

        # Step 2: Induce backbone via Split-Attractor Condensation + Multi-Hop Attractor Refinement
        adj, node_labels = self._build_mdl_candidate_backbone(
            names, bitmasks, jc_mut_mat, terminal_loads, tree_weights, L_eff, ti_cost, tv_cost
        )

        # Step 3: Exact Global Fitch Pass + Fast Vectorized MDL Edge Contraction +
        # Polytomy Hub Split-Attractor & Buneman Compatibility Clique Expansion
        adj, branches, root_bitmasks, edge_mutations = self._contract_unsupported_edges_mdl(
            adj, bitmasks, majority_state, terminal_loads, N, L_eff, ti_cost, tv_cost, tree_weights,
            node_labels=node_labels,
        )

        # Step 4: Dual-Mode Residual Shunt Peeling + Alternating Shunt-Backbone Coordinate Descent
        shunts: List[ShuntEdge] = []
        if self.detect_shunts and N >= 4:
            shunts = self._peel_residual_shunts_mdl(
                adj, node_labels, bitmasks, majority_state, terminal_loads,
                edge_mutations, N, L, L_eff, ti_cost, tv_cost, seqs, mat, is_coding
            )
            # Coordinate-descent feedback: if high-gain INTERVAL recombination tracts are certified,
            # mask out the laterally transferred intervals on the recipient leaf/clade and
            # re-verify the vertical backbone so lateral tracts never lock in false vertical clades!
            int_shunts = [
                s for s in shunts
                if s.mode == "INTERVAL" and s.interval_span is not None and s.net_mdl_gain_nats >= 15.0
            ]
            if int_shunts:
                masked_bitmasks = bitmasks.copy()
                masked_any = False
                for sh in int_shunts:
                    a_sp, b_sp = sh.interval_span
                    # Identify which endpoint is the recombinant leaf (or smaller clade)
                    rec_node = sh.node_u if sh.node_u < N else (sh.node_v if sh.node_v < N else -1)
                    if rec_node >= 0:
                        masked_bitmasks[rec_node, a_sp : b_sp + 1] = 15
                        masked_any = True
                if masked_any:
                    adj, branches, root_bitmasks, edge_mutations = self._contract_unsupported_edges_mdl(
                        adj, masked_bitmasks, majority_state, terminal_loads, N, L_eff,
                        ti_cost, tv_cost, tree_weights, node_labels=node_labels, expand_hubs=False
                    )

        # Step 5: Compute exact MDL bit/nat budgets and serialize multifurcating Newick
        raw_nats = float(N * L_eff * math.log(4.0))
        star_nats = self._compute_star_tree_mdl_nats(bitmasks, L_eff)
        backbone_nats = self._compute_tree_total_mdl_nats(adj, branches, L_eff)
        shunt_gain_total = sum(s.net_mdl_gain_nats for s in shunts)
        full_graph_nats = max(L_eff * math.log(4.0), backbone_nats - shunt_gain_total)

        internal_nodes = [u for u in adj if u >= N and len(adj[u]) > 0]
        polytomy_degrees = [len(adj[u]) for u in internal_nodes if len(adj[u]) > 3]
        n_internal_branches = sum(1 for b in branches if b.is_internal)

        newick = self._serialize_newick(adj, names, branches, N)
        runtime = time.perf_counter() - t0

        return MDLGraphResult(
            names=names,
            n_taxa=N,
            n_sites=L,
            L_eff=L_eff,
            newick=newick,
            adjacency=adj,
            node_labels=node_labels,
            branches=branches,
            shunts=shunts,
            n_internal_branches=n_internal_branches,
            max_binary_internal_branches=max(0, N - 3),
            polytomy_nodes=len(polytomy_degrees),
            max_polytomy_degree=max(polytomy_degrees) if polytomy_degrees else (3 if N >= 3 else N),
            raw_uncompressed_nats=raw_nats,
            star_tree_nats=star_nats,
            backbone_tree_nats=backbone_nats,
            full_graph_nats=full_graph_nats,
            runtime_sec=runtime,
        )

    # -----------------------------------------------------------------------
    # Internal Mathematical Steps
    # -----------------------------------------------------------------------

    def _estimate_subst_surprise(
        self, mat: np.ndarray, valid_mask: np.ndarray
    ) -> Tuple[float, float]:
        """Estimate tree-free transition (ti) and transversion (tv) state-choice code lengths."""
        N, L = mat.shape
        idx = np.linspace(0, N - 1, min(N, 64), dtype=int)
        sub_mat = mat[idx]
        sub_val = valid_mask[idx]
        ti = 1.0
        tv = 2.0
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                both = sub_val[i] & sub_val[j]
                diffs = both & (sub_mat[i] != sub_mat[j])
                if np.any(diffs):
                    s_i = sub_mat[i, diffs]
                    s_j = sub_mat[j, diffs]
                    same_class = IS_PURINE[s_i] == IS_PURINE[s_j]
                    ti += float(np.sum(same_class))
                    tv += float(np.sum(~same_class))
        tot = ti + tv
        p_ti = np.clip(ti / tot, 0.20, 0.85)
        p_tv_each = (1.0 - p_ti) / 2.0
        ti_cost = -math.log(p_ti)
        tv_cost = -math.log(max(p_tv_each, 0.05))
        return ti_cost, tv_cost

    def _compute_dual_site_coherence_weights(
        self,
        mat: np.ndarray,
        valid_mask: np.ndarray,
        dist_weights: np.ndarray,
        tree_weights: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tree-Free Dual-Space Site Reliability Weighting:
        1. Allelic Multiplicity Discount W_mult(s): penalizes 3-state and 4-state
           columns that are guaranteed to harbor homoplasy on any tree.
        2. Long-Range Four-Gamete Coherence W_long(s): measures Buneman compatibility
           I(c_s, c_s') = min(|C n C'|, |C n ~C'|, |~C n C'|, |~C n ~C'|) = 0 across
           distant columns |s - s'| > Delta_local, downweighting saturated noise sites
           and localized recombination tracts before graph induction begins.
        """
        N, L = mat.shape
        if N < 6 or L < 10:
            return dist_weights, tree_weights

        # 1. State multiplicity per column
        present = np.zeros((4, L), dtype=np.int16)
        for b in range(4):
            present[b] = np.any(mat == b, axis=0).astype(np.int16)
        n_states = np.sum(present, axis=0)
        w_mult = np.ones(L, dtype=np.float64)
        w_mult = np.where(n_states == 3, 0.90, w_mult)
        w_mult = np.where(n_states >= 4, 0.75, w_mult)

        # 2. Extract parsimony-informative column-state vectors c_k in {0, 1}^N
        char_vecs: List[np.ndarray] = []
        char_val: List[np.ndarray] = []
        char_sites: List[int] = []
        char_counts: List[int] = []

        for s in range(L):
            if n_states[s] < 2:
                continue
            v_col = valid_mask[:, s]
            n_val = int(np.sum(v_col))
            if n_val < 4:
                continue
            for b in range(4):
                c_vec = (mat[:, s] == b) & v_col
                cnt = int(np.sum(c_vec))
                if 2 <= cnt <= n_val // 2:
                    char_vecs.append(c_vec.astype(np.float32))
                    char_val.append((~c_vec & v_col).astype(np.float32))
                    char_sites.append(s)
                    char_counts.append(cnt)

        M = len(char_vecs)
        if M < 8:
            return dist_weights * w_mult, tree_weights * w_mult

        C = np.array(char_vecs, dtype=np.float32)       # (M, N)
        C_bar = np.array(char_val, dtype=np.float32)    # (M, N)
        sites_arr = np.array(char_sites, dtype=np.int32)
        counts_arr = np.array(char_counts, dtype=np.int32)

        # Select up to 350 reference characters with highest minor-allele balance
        if M > 350:
            ref_idx = np.argsort(-np.minimum(counts_arr, N - counts_arr))[:350]
        else:
            ref_idx = np.arange(M)

        C_ref = C[ref_idx]
        C_bar_ref = C_bar[ref_idx]
        sites_ref = sites_arr[ref_idx]

        # Four-Gamete Venn quadrants via vectorized BLAS matrix multiplications
        q11 = C @ C_ref.T
        q10 = C @ C_bar_ref.T
        q01 = C_bar @ C_ref.T
        q00 = C_bar @ C_bar_ref.T
        gamete_viol = np.minimum(np.minimum(q11, q10), np.minimum(q01, q00))

        # Long-range genomic mask |s_k - s_l| > Delta_local and non-disjoint support (q11 >= 1)
        delta_local = max(40, min(180, L // 6))
        dist_long = np.abs(sites_arr[:, None] - sites_ref[None, :]) > delta_local
        overlap_mask = dist_long & (q11 >= 1.0)

        compat_mat = np.exp(-1.1 * gamete_viol)
        overlap_counts = np.sum(overlap_mask, axis=1)
        coh_scores = np.where(
            overlap_counts >= 3,
            np.sum(compat_mat * overlap_mask, axis=1) / np.maximum(overlap_counts, 1),
            np.sum(compat_mat * dist_long, axis=1) / np.maximum(np.sum(dist_long, axis=1), 1),
        )

        # Map character coherence back to sites and normalize so median informative site is 1.0
        site_coh = np.ones(L, dtype=np.float64)
        site_cnt = np.zeros(L, dtype=np.int32)
        for k, s in enumerate(char_sites):
            if site_cnt[s] == 0:
                site_coh[s] = float(coh_scores[k])
            else:
                site_coh[s] = max(site_coh[s], float(coh_scores[k]))
            site_cnt[s] += 1

        inf_mask = site_cnt > 0
        if np.any(inf_mask):
            med_coh = float(np.median(site_coh[inf_mask]))
            if med_coh > 1e-6:
                norm_coh = np.clip(site_coh / med_coh, 0.65, 1.30)
                site_coh = np.where(inf_mask, norm_coh, 1.0)

        var_cols = n_states >= 2
        frac_var = float(np.mean(var_cols))
        frac_multi = float(np.mean(n_states[var_cols] >= 3)) if np.any(var_cols) else 0.0
        w_mult_tree = np.ones(L, dtype=np.float64) if frac_multi > 0.50 else w_mult
        coh_tree_factor = np.clip(site_coh * w_mult_tree, 0.75, 1.25)

        if frac_var >= 0.25 and frac_multi > 0.40:
            eff_dist_w = np.where(coh_tree_factor >= 0.90, 1.0, dist_weights)
            eff_tree_w = np.where(coh_tree_factor >= 0.90, 1.0, tree_weights)
        else:
            eff_dist_w = dist_weights
            eff_tree_w = tree_weights

        mod_dist = eff_dist_w * (site_coh ** 1.25) * w_mult
        mod_tree = eff_tree_w * coh_tree_factor
        return mod_dist, mod_tree

    def _compute_pairwise_mutation_matrices(
        self,
        mat: np.ndarray,
        valid_mask: np.ndarray,
        site_weights: np.ndarray,
        L_eff: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute vectorized pairwise Hamming and JC-corrected mutation matrices."""
        N, L = mat.shape
        one_hot = np.zeros((4, N, L), dtype=np.float64)
        w_one_hot = np.zeros((4, N, L), dtype=np.float64)
        for b in range(4):
            mask_b = (mat == b).astype(np.float64)
            one_hot[b] = mask_b
            w_one_hot[b] = mask_b * site_weights[None, :]

        valid_f = valid_mask.astype(np.float64)
        w_valid_f = valid_f * site_weights[None, :]

        valid_pairs = valid_f @ valid_f.T
        w_valid_pairs = w_valid_f @ w_valid_f.T if False else (w_valid_f @ valid_f.T)

        matches = np.zeros((N, N), dtype=np.float64)
        w_matches = np.zeros((N, N), dtype=np.float64)
        for b in range(4):
            matches += one_hot[b] @ one_hot[b].T
            w_matches += w_one_hot[b] @ one_hot[b].T

        raw_diffs = np.maximum(0.0, valid_pairs - matches)
        w_raw_diffs = np.maximum(0.0, w_valid_pairs - w_matches)

        denom = np.maximum(valid_pairs, 1.0)
        w_denom = np.maximum(w_valid_pairs, 1.0)
        p_hat = np.clip(w_raw_diffs / w_denom, 0.0, 0.745)
        diff_mat = raw_diffs * (L_eff / denom)

        jc_d = -0.75 * np.log(1.0 - (4.0 / 3.0) * p_hat)
        jc_mut_mat = jc_d * L_eff
        np.fill_diagonal(diff_mat, 0.0)
        np.fill_diagonal(jc_mut_mat, 0.0)
        return diff_mat, valid_pairs, jc_mut_mat

    def _estimate_three_point_terminal_loads(self, mut_mat: np.ndarray) -> np.ndarray:
        """
        Topologically invariant 3-point estimation of each taxon's terminal
        branch mutation count m_i:
            m_i = 0.5 * (d(i, j_near) + median_{k != i, j} [d(i, k) - d(j_near, k)])
        """
        N = mut_mat.shape[0]
        if N <= 2:
            return np.maximum(0.5, np.min(mut_mat + np.eye(N) * 1e9, axis=1) * 0.5)
        loads = np.zeros(N, dtype=np.float64)
        masked = mut_mat + np.eye(N) * 1e12
        nearest = np.argmin(masked, axis=1)
        for i in range(N):
            j = int(nearest[i])
            d_ij = mut_mat[i, j]
            others = [k for k in range(N) if k != i and k != j]
            deltas = mut_mat[i, others] - mut_mat[j, others]
            m_i = 0.5 * (d_ij + float(np.median(deltas)))
            loads[i] = np.clip(m_i, 0.4, max(0.8, d_ij))
        return loads

    def _build_mdl_candidate_backbone(
        self,
        names: List[str],
        bitmasks: np.ndarray,
        jc_mut_mat: np.ndarray,
        terminal_loads: np.ndarray,
        site_weights: np.ndarray,
        L_eff: float,
        ti_cost: float,
        tv_cost: float,
    ) -> Tuple[Dict[int, List[int]], Dict[int, str]]:
        """
        Construct the initial unrooted binary tree backbone using variance-weighted
        agglomeration + Unified MDL Quartet NNI Refinement.
        """
        N, L = bitmasks.shape
        node_labels: Dict[int, str] = {i: names[i] for i in range(N)}
        adj: Dict[int, List[int]] = {i: [] for i in range(N)}

        if N == 1:
            return adj, node_labels
        if N == 2:
            adj[0].append(1)
            adj[1].append(0)
            return adj, node_labels
        if N == 3:
            center = N
            adj[center] = [0, 1, 2]
            node_labels[center] = f"N{center}"
            for i in range(3):
                adj[i].append(center)
            return adj, node_labels

        if N == 4:
            # Exact Split-Space Evaluation for N=4 (zero distance agglomeration or NNI)
            t = [(bitmasks[i], float(terminal_loads[i])) for i in range(4)]
            splits = [
                ([0, 1], [2, 3]),
                ([0, 2], [1, 3]),
                ([0, 3], [1, 2]),
            ]
            gains = [
                self._compute_split_mdl_gain_from_children(
                    [t[a], t[b]], [t[c], t[d]], L_eff, ti_cost, tv_cost, site_weights
                )[0]
                for (a, b), (c, d) in splits
            ]
            best_idx = int(np.argmax(gains))
            if gains[best_idx] > 0.0:
                (a, b), (c, d) = splits[best_idx]
                u, v = 4, 5
                adj[u] = [a, b, v]
                adj[v] = [c, d, u]
                node_labels[u] = f"N{u}"
                node_labels[v] = f"N{v}"
                for x in (a, b):
                    adj[x].append(u)
                for y in (c, d):
                    adj[y].append(v)
            else:
                center = 4
                adj[center] = [0, 1, 2, 3]
                node_labels[center] = f"N{center}"
                for i in range(4):
                    adj[i].append(center)
            return adj, node_labels

        D = jc_mut_mat.copy().astype(np.float64)
        V = np.maximum(D, 0.5) * (1.0 + (D / max(L_eff, 1.0)) ** 2)
        active = list(range(N))
        next_node = N

        for i in range(N, 2 * N - 2):
            adj[i] = []
            node_labels[i] = f"N{i}"

        while len(active) > 3:
            k = len(active)
            idx = np.array(active, dtype=int)
            sub_D = D[np.ix_(idx, idx)]
            row_sums = np.sum(sub_D, axis=1)

            Q = (k - 2.0) * sub_D - row_sums[:, None] - row_sums[None, :]
            np.fill_diagonal(Q, np.inf)
            flat_best = int(np.argmin(Q))
            a_pos, b_pos = divmod(flat_best, k)
            u = active[a_pos]
            v = active[b_pos]

            w = next_node
            next_node += 1
            adj[w].extend([u, v])
            adj[u].append(w)
            adj[v].append(w)

            d_uv = D[u, v]
            v_uv = max(V[u, v], 1e-6)
            others = [x for x in active if x != u and x != v]
            if others:
                lam_num = sum(V[v, x] - V[u, x] for x in others)
                lam = 0.5 + lam_num / (2.0 * len(others) * v_uv)
                lam = float(np.clip(lam, 0.05, 0.95))
            else:
                lam = 0.5

            limb_u = max(0.0, 0.5 * d_uv + (row_sums[a_pos] - row_sums[b_pos]) / (2.0 * (k - 2.0)))
            limb_v = max(0.0, d_uv - limb_u)

            if w >= D.shape[0]:
                new_size = w + 1
                D_new = np.zeros((new_size, new_size), dtype=np.float64)
                V_new = np.zeros((new_size, new_size), dtype=np.float64)
                D_new[: D.shape[0], : D.shape[1]] = D
                V_new[: V.shape[0], : V.shape[1]] = V
                D = D_new
                V = V_new

            for x in others:
                d_wx = max(0.0, lam * (D[u, x] - limb_u) + (1.0 - lam) * (D[v, x] - limb_v))
                v_wx = max(0.25, lam * V[u, x] + (1.0 - lam) * V[v, x] - lam * (1.0 - lam) * v_uv)
                D[w, x] = D[x, w] = d_wx
                V[w, x] = V[x, w] = v_wx

            active.remove(u)
            active.remove(v)
            active.append(w)

        w = next_node
        for x in active:
            adj[w].append(x)
            adj[x].append(w)

        adj = self._refine_backbone_nni_mdl(
            adj, bitmasks, terminal_loads, N, L_eff, ti_cost, tv_cost, site_weights
        )
        return adj, node_labels

    def _refine_backbone_nni_mdl(
        self,
        adj: Dict[int, List[int]],
        bitmasks: np.ndarray,
        terminal_loads: np.ndarray,
        N: int,
        L_eff: float,
        ti_cost: float,
        tv_cost: float,
        site_weights: np.ndarray,
        max_rounds: int = 5,
    ) -> Dict[int, List[int]]:
        """
        Multi-Hop MDL Split-Attractor Refinement:
          1. 1-hop Quartet Split-Space Resolution around every internal edge (u, v).
          2. 2-hop Cross-Node Subtree Swap (Radius-2 SPR) across internal paths u - w - v
             to escape 1-hop greedy local minima where two compatible clades are separated
             by an intermediate branch w.
        """
        for round_idx in range(max_rounds):
            states = self._compute_all_directed_fitch_states(
                adj, bitmasks, N, terminal_loads, L_eff
            )
            improved = False
            internal_edges = [
                (u, v)
                for u in adj
                for v in adj[u]
                if u < v and u >= N and v >= N and len(adj[u]) == 3 and len(adj[v]) == 3
            ]
            for u, v in internal_edges:
                if v not in adj[u] or len(adj[u]) != 3 or len(adj[v]) != 3:
                    continue
                u_neighbors = [x for x in adj[u] if x != v]
                v_neighbors = [y for y in adj[v] if y != u]
                A, B = u_neighbors[0], u_neighbors[1]
                C, D = v_neighbors[0], v_neighbors[1]

                tA = states[(A, u)]
                tB = states[(B, u)]
                tC = states[(C, v)]
                tD = states[(D, v)]

                # Evaluate exact MDL compression gain on the 3 quartet resolutions
                g0, _, _ = self._compute_split_mdl_gain_from_children(
                    [tA, tB], [tC, tD], L_eff, ti_cost, tv_cost, site_weights
                )
                g1, _, _ = self._compute_split_mdl_gain_from_children(
                    [tA, tC], [tB, tD], L_eff, ti_cost, tv_cost, site_weights
                )
                g2, _, _ = self._compute_split_mdl_gain_from_children(
                    [tA, tD], [tB, tC], L_eff, ti_cost, tv_cost, site_weights
                )

                best_g = max(g0, g1, g2)
                if best_g > g0 + 0.10:
                    if g1 >= g2:
                        adj[u].remove(B)
                        adj[B].remove(u)
                        adj[v].remove(C)
                        adj[C].remove(v)
                        adj[u].append(C)
                        adj[C].append(u)
                        adj[v].append(B)
                        adj[B].append(v)
                    else:
                        adj[u].remove(B)
                        adj[B].remove(u)
                        adj[v].remove(D)
                        adj[D].remove(v)
                        adj[u].append(D)
                        adj[D].append(u)
                        adj[v].append(B)
                        adj[B].append(v)
                    improved = True
                    states = self._compute_all_directed_fitch_states(
                        adj, bitmasks, N, terminal_loads, L_eff
                    )

            # 2-Hop Cross-Node Attractor Swap (Radius-2 SPR across u - w - v)
            if round_idx < 2 and N <= 250:
                int_nodes = [w for w in adj if w >= N and len(adj[w]) == 3]
                for w in int_nodes:
                    if len(adj[w]) != 3:
                        continue
                    nbs = adj[w]
                    int_nbs = [x for x in nbs if x >= N and len(adj[x]) == 3]
                    if len(int_nbs) < 2:
                        continue
                    for idx_i in range(len(int_nbs)):
                        for idx_j in range(idx_i + 1, len(int_nbs)):
                            u = int_nbs[idx_i]
                            v = int_nbs[idx_j]
                            if w not in adj[u] or w not in adj[v] or len(adj[u]) != 3 or len(adj[v]) != 3:
                                continue
                            M_list = [x for x in adj[w] if x != u and x != v]
                            if len(M_list) != 1:
                                continue
                            M_node = M_list[0]
                            u_ch = [x for x in adj[u] if x != w]
                            v_ch = [y for y in adj[v] if y != w]
                            if len(u_ch) != 2 or len(v_ch) != 2:
                                continue
                            A, B = u_ch[0], u_ch[1]
                            C, D = v_ch[0], v_ch[1]

                            tA, tB = states[(A, u)], states[(B, u)]
                            tC, tD = states[(C, v)], states[(D, v)]
                            tM = states[(M_node, w)]

                            # Current configuration gain across edges (u, w) and (w, v)
                            t_u_curr = self._combine_fitch_children([tA, tB], L_eff)
                            t_v_curr = self._combine_fitch_children([tC, tD], L_eff)
                            g_uw_0, _, _ = self._compute_split_mdl_gain_from_children(
                                [tA, tB], [tM, t_v_curr], L_eff, ti_cost, tv_cost, site_weights
                            )
                            g_wv_0, _, _ = self._compute_split_mdl_gain_from_children(
                                [tC, tD], [tM, t_u_curr], L_eff, ti_cost, tv_cost, site_weights
                            )
                            base_2hop = g_uw_0 + g_wv_0

                            # Candidate 2-hop swaps between one child of u and one child of v
                            best_swap_gain = base_2hop + 0.50
                            best_pair: Optional[Tuple[int, int]] = None
                            for u_swap, t_us, t_uk in ((A, tA, tB), (B, tB, tA)):
                                for v_swap, t_vs, t_vk in ((C, tC, tD), (D, tD, tC)):
                                    t_u_new = self._combine_fitch_children([t_uk, t_vs], L_eff)
                                    t_v_new = self._combine_fitch_children([t_vk, t_us], L_eff)
                                    g_uw_n, _, _ = self._compute_split_mdl_gain_from_children(
                                        [t_uk, t_vs], [tM, t_v_new], L_eff, ti_cost, tv_cost, site_weights
                                    )
                                    g_wv_n, _, _ = self._compute_split_mdl_gain_from_children(
                                        [t_vk, t_us], [tM, t_u_new], L_eff, ti_cost, tv_cost, site_weights
                                    )
                                    if (g_uw_n + g_wv_n) > best_swap_gain:
                                        best_swap_gain = g_uw_n + g_wv_n
                                        best_pair = (u_swap, v_swap)

                            if best_pair is not None:
                                u_swap, v_swap = best_pair
                                adj[u].remove(u_swap)
                                adj[u_swap].remove(u)
                                adj[v].remove(v_swap)
                                adj[v_swap].remove(v)
                                adj[u].append(v_swap)
                                adj[v_swap].append(u)
                                adj[v].append(u_swap)
                                adj[u_swap].append(v)
                                improved = True
                                states = self._compute_all_directed_fitch_states(
                                    adj, bitmasks, N, terminal_loads, L_eff
                                )
            if not improved:
                break
        return adj

    def _compute_all_directed_fitch_states(
        self,
        adj: Dict[int, List[int]],
        bitmasks: np.ndarray,
        N: int,
        terminal_loads: Optional[np.ndarray] = None,
        L_eff: float = 1000.0,
    ) -> Dict[Tuple[int, int], Tuple[np.ndarray, float]]:
        """
        Two-pass message-passing (upward + downward) to compute for every directed
        edge (u -> v) the Fitch state bitmask of subtree u (away from v)
        and its parallel-precision mutation load.
        """
        L = bitmasks.shape[1]
        if terminal_loads is None:
            terminal_loads = np.full(N, 1.0, dtype=np.float64)

        root = next((u for u in adj if u >= N and len(adj[u]) > 0), next(u for u in adj if len(adj[u]) > 0))
        parent: Dict[int, Optional[int]] = {root: None}
        order: List[int] = [root]
        q = [root]
        while q:
            curr = q.pop(0)
            for nb in adj[curr]:
                if nb not in parent:
                    parent[nb] = curr
                    order.append(nb)
                    q.append(nb)

        states: Dict[Tuple[int, int], Tuple[np.ndarray, float]] = {}

        # Bottom-up pass
        for u in reversed(order):
            p = parent[u]
            if p is None:
                continue
            if u < N:
                states[(u, p)] = (bitmasks[u], float(terminal_loads[u]))
            else:
                children = [nb for nb in adj[u] if nb != p]
                mask, mut_load = self._combine_fitch_children(
                    [states[(c, u)] for c in children], L_eff
                )
                states[(u, p)] = (mask, mut_load)

        # Top-down pass
        for u in order:
            p = parent[u]
            children = [nb for nb in adj[u] if nb != p]
            for c in children:
                if u < N:
                    states[(u, c)] = (bitmasks[u], float(terminal_loads[u]))
                else:
                    other_neighbors = [nb for nb in adj[u] if nb != c]
                    if not other_neighbors:
                        states[(u, c)] = (np.full(L, 15, dtype=np.uint8), 1.0)
                    else:
                        mask, mut_load = self._combine_fitch_children(
                            [states[(nb, u)] for nb in other_neighbors], L_eff
                        )
                        states[(u, c)] = (mask, mut_load)

        return states

    @staticmethod
    def _combine_fitch_children(
        child_list: List[Tuple[np.ndarray, float]],
        L_eff: float = 1000.0,
    ) -> Tuple[np.ndarray, float]:
        """
        Combine k >= 1 child bitmasks at a node:
          - Ignore pure-gap/ambiguous children (mask == 15) when computing majority/intersection
            so missing data never dilute real nucleotide states!
          - Parallel-precision combination 1 / sum(1 / m_c) for the effective uncertainty load!
        """
        if len(child_list) == 1:
            return child_list[0]
        if len(child_list) == 2:
            s1, m1 = child_list[0]
            s2, m2 = child_list[1]
            inter = s1 & s2
            out = np.where(inter == 0, s1 | s2, inter)
            n_mut = float(np.sum((inter == 0) & (s1 != 15) & (s2 != 15)))
            par_load = 1.0 / (1.0 / max(m1, 0.4) + 1.0 / max(m2, 0.4)) + 0.25 * n_mut
            return out, max(0.4, par_load)

        L = child_list[0][0].shape[0]
        counts = np.zeros((4, L), dtype=np.int16)
        inv_sum = 0.0
        for s_mask, m_c in child_list:
            inv_sum += 1.0 / max(m_c, 0.4)
            non_gap = (s_mask != 15).astype(np.int16)
            for b in range(4):
                counts[b] += (((s_mask >> b) & 1).astype(np.int16)) * non_gap
        max_c = np.max(counts, axis=0)
        out = np.zeros(L, dtype=np.uint8)
        for b in range(4):
            out |= ((counts[b] == max_c).astype(np.uint8) << b)
        out = np.where(max_c == 0, 15, out).astype(np.uint8)
        par_load = 1.0 / max(inv_sum, 1e-6)
        return out, max(0.4, par_load)

    def _expand_polytomy_hubs_mdl(
        self,
        adj: Dict[int, List[int]],
        node_labels: Optional[Dict[int, str]],
        bitmasks: np.ndarray,
        terminal_loads: np.ndarray,
        N: int,
        L_eff: float,
        ti_cost: float,
        tv_cost: float,
        site_weights: np.ndarray,
    ) -> Dict[int, List[int]]:
        """
        Dual-Space Polytomy Hub Split-Attractor & Buneman Compatibility Clique Expansion:
        At any multifurcating hub H of degree d >= 4, extracts column-induced character
        bipartitions U_{s,b} subset adj(H) (2 <= |U| <= d - 2), denoises them via the
        Hopfield Split-Attractor fixed-point map, and instantiates the Maximum-Weight
        Compatible (Laminar) Clique of splits with G_MDL(U | adj(H) \\ U) > 0.
        """
        next_id = max(adj.keys(), default=N) + 1
        states = self._compute_all_directed_fitch_states(
            adj, bitmasks, N, terminal_loads, L_eff
        )
        hubs = [h for h in list(adj.keys()) if h >= N and 4 <= len(adj[h]) <= 40]

        for h in hubs:
            if h not in adj or len(adj[h]) < 4:
                continue
            nbs = list(adj[h])
            d = len(nbs)
            nb_states = [states[(c, h)] for c in nbs]
            masks = np.array([st[0] for st in nb_states], dtype=np.uint8)  # (d, L)
            L = masks.shape[1]

            # Extract column-induced candidate subsets U_{s,b} of {0..d-1}
            seed_weights: Dict[Tuple[int, ...], float] = {}
            for s in range(L):
                col = masks[:, s]
                val = col != 15
                if int(np.sum(val)) < 4:
                    continue
                for b in range(4):
                    has_b = ((col >> b) & 1 != 0) & val
                    cnt = int(np.sum(has_b))
                    if 2 <= cnt <= d - 2:
                        idx_tup = tuple(np.where(has_b)[0].tolist())
                        if len(idx_tup) > d // 2 or (len(idx_tup) == d // 2 and 0 not in idx_tup):
                            comp = tuple(i for i in range(d) if i not in idx_tup)
                            idx_tup = comp
                        if 2 <= len(idx_tup) <= d - 2:
                            seed_weights[idx_tup] = seed_weights.get(idx_tup, 0.0) + float(site_weights[s])

            if not seed_weights:
                continue

            # Evaluate top 25 column-seeded attractors at hub h
            top_seeds = sorted(seed_weights.items(), key=lambda kv: kv[1], reverse=True)[:25]
            positive_attractors: List[Tuple[float, Set[int]]] = []
            seen_sets: Set[FrozenSet[int]] = set()

            for idx_tup, _ in top_seeds:
                U_set = set(idx_tup)
                # Hopfield 1-step attractor refinement
                u_ch = [nb_states[i] for i in sorted(U_set)]
                v_ch = [nb_states[j] for j in range(d) if j not in U_set]
                g_cur, _, sites_cur = self._compute_split_mdl_gain_from_children(
                    u_ch, v_ch, L_eff, ti_cost, tv_cost, site_weights
                )
                if len(sites_cur) >= 1 and d <= 20:
                    for k_try in range(d):
                        cand_set = (U_set - {k_try}) if k_try in U_set else (U_set | {k_try})
                        if not (2 <= len(cand_set) <= d - 2):
                            continue
                        u_try = [nb_states[i] for i in sorted(cand_set)]
                        v_try = [nb_states[j] for j in range(d) if j not in cand_set]
                        g_try, _, _ = self._compute_split_mdl_gain_from_children(
                            u_try, v_try, L_eff, ti_cost, tv_cost, site_weights
                        )
                        if g_try > g_cur + 0.25:
                            U_set = cand_set
                            g_cur = g_try

                f_set = frozenset(U_set)
                if g_cur > 0.0 and f_set not in seen_sets:
                    seen_sets.add(f_set)
                    positive_attractors.append((g_cur, set(U_set)))

            if not positive_attractors:
                continue

            # Sort by descending MDL gain and extract Maximum-Weight Disjoint/Compatible Family
            positive_attractors.sort(key=lambda x: x[0], reverse=True)
            chosen_splits: List[Set[int]] = []
            used_indices: Set[int] = set()
            for gain, U_set in positive_attractors:
                if U_set.isdisjoint(used_indices) and (d - len(used_indices) - len(U_set)) >= 2:
                    chosen_splits.append(U_set)
                    used_indices.update(U_set)

            for U_set in chosen_splits:
                new_node = next_id
                next_id += 1
                adj[new_node] = [h]
                if node_labels is not None:
                    node_labels[new_node] = f"N{new_node}"
                adj[h].append(new_node)
                for idx_c in sorted(U_set):
                    c_node = nbs[idx_c]
                    if c_node in adj[h]:
                        adj[h].remove(c_node)
                        adj[c_node].remove(h)
                        adj[new_node].append(c_node)
                        adj[c_node].append(new_node)

            if chosen_splits:
                states = self._compute_all_directed_fitch_states(
                    adj, bitmasks, N, terminal_loads, L_eff
                )

        return adj

    def _contract_unsupported_edges_mdl(
        self,
        adj: Dict[int, List[int]],
        bitmasks: np.ndarray,
        majority_state: np.ndarray,
        terminal_loads: np.ndarray,
        N: int,
        L_eff: float,
        ti_cost: float,
        tv_cost: float,
        site_weights: np.ndarray,
        node_labels: Optional[Dict[int, str]] = None,
        expand_hubs: bool = True,
    ) -> Tuple[
        Dict[int, List[int]],
        List[BranchRecord],
        Dict[int, np.ndarray],
        Dict[Tuple[int, int], List[Tuple[int, int, int]]],
    ]:
        """
        Fast Vectorized MDL Edge-Contraction Engine (Zero-Mutation & Noise Exclusion Law)
        coupled with Dual-Space Polytomy Hub Split-Attractor Expansion.
        Contracts any internal edge with mdl_gain_nats <= 0.0 into a native polytomy!
        """
        for outer_pass in range(2 if expand_hubs else 1):
            while True:
                states = self._compute_all_directed_fitch_states(
                    adj, bitmasks, N, terminal_loads, L_eff
                )
                internal_edges = [
                    (u, v)
                    for u in adj
                    for v in adj[u]
                    if u < v and u >= N and v >= N and len(adj[u]) >= 3 and len(adj[v]) >= 3
                ]
                if not internal_edges:
                    break

                bad_edges: List[Tuple[float, int, int]] = []
                for u, v in internal_edges:
                    gain, _, _ = self._evaluate_internal_edge_mdl_gain(
                        u, v, adj, states, L_eff, ti_cost, tv_cost, site_weights
                    )
                    if gain <= 0.0:
                        bad_edges.append((gain, u, v))

                if not bad_edges:
                    break

                # Sort from worst (most negative) MDL gain to 0 and contract a maximal
                # node-disjoint set of unsupported edges in a single pass!
                bad_edges.sort(key=lambda x: x[0])
                locked_nodes: Set[int] = set()
                contracted_any = False

                for gain, u, v in bad_edges:
                    if u in locked_nodes or v in locked_nodes:
                        continue
                    if v not in adj[u]:
                        continue
                    adj[u].remove(v)
                    adj[v].remove(u)
                    for nb in list(adj[v]):
                        adj[nb].remove(v)
                        if nb != u and nb not in adj[u]:
                            adj[u].append(nb)
                            adj[nb].append(u)
                    adj[v].clear()
                    locked_nodes.add(u)
                    locked_nodes.add(v)
                    contracted_any = True

                if not contracted_any:
                    break

            if expand_hubs and outer_pass == 0 and N >= 5:
                adj = self._expand_polytomy_hubs_mdl(
                    adj, node_labels, bitmasks, terminal_loads, N, L_eff, ti_cost, tv_cost, site_weights
                )

        states = self._compute_all_directed_fitch_states(
            adj, bitmasks, N, terminal_loads, L_eff
        )
        node_states, edge_mutations = self._reconstruct_nodal_states_and_mutations(
            adj, bitmasks, majority_state, states, N
        )

        branches: List[BranchRecord] = []
        for u in sorted(adj.keys()):
            for v in sorted(adj[u]):
                if u < v:
                    is_int = (u >= N) and (v >= N)
                    mut_list = edge_mutations.get((u, v), [])
                    m_count = float(len(mut_list))
                    if is_int:
                        gain, m_syn, syn_sites = self._evaluate_internal_edge_mdl_gain(
                            u, v, adj, states, L_eff, ti_cost, tv_cost, site_weights
                        )
                        eff_mut = max(m_count, m_syn)
                    else:
                        eff_mut = m_count
                        syn_sites = [s for s, _, _ in mut_list]
                        gain = branch_mutation_code_nats(eff_mut, L_eff)

                    support = 1.0 / (1.0 + math.exp(-np.clip(gain, -50.0, 50.0)))
                    branches.append(
                        BranchRecord(
                            u=u,
                            v=v,
                            is_internal=is_int,
                            mutation_count=eff_mut,
                            branch_length=eff_mut / L_eff,
                            mdl_gain_nats=gain,
                            mdl_support=support,
                            synapomorphy_sites=syn_sites,
                        )
                    )

        return adj, branches, node_states, edge_mutations

    def _evaluate_internal_edge_mdl_gain(
        self,
        u: int,
        v: int,
        adj: Dict[int, List[int]],
        states: Dict[Tuple[int, int], Tuple[np.ndarray, float]],
        L_eff: float,
        ti_cost: float,
        tv_cost: float,
        site_weights: np.ndarray,
    ) -> Tuple[float, float, List[int]]:
        """Evaluate exact net MDL compression gain of internal branch e = (u, v)."""
        u_sub = [nb for nb in adj[u] if nb != v]
        v_sub = [nb for nb in adj[v] if nb != u]
        if len(u_sub) < 2 or len(v_sub) < 2:
            return -10.0, 0.0, []
        u_children = [states[(nb, u)] for nb in u_sub]
        v_children = [states[(nb, v)] for nb in v_sub]
        return self._compute_split_mdl_gain_from_children(
            u_children, v_children, L_eff, ti_cost, tv_cost, site_weights
        )

    def _compute_split_mdl_gain_from_children(
        self,
        u_children: List[Tuple[np.ndarray, float]],
        v_children: List[Tuple[np.ndarray, float]],
        L_eff: float,
        ti_cost: float,
        tv_cost: float,
        site_weights: np.ndarray,
    ) -> Tuple[float, float, List[int]]:
        """
        Unified Vectorized MDL Split Compression Gain Engine.
        Used identically by Quartet NNI Refinement and Internal Edge Contraction!

        Computes the exact net reduction in total description length:
            MDL_Gain(u, v) = max_{m >= 1} [ sum_{j=1}^m Delta C(s_(j)) - L_branch(m) ] - L_topo(deg_a, deg_b)
        Incorporates:
          - 3-against-1 Conditional Terminal Load Refinement (eliminating internal-branch
            contamination on short branches in the Felsenstein Zone),
          - Exact Laplace-smoothed log-binomial marginal weights ln((L_eff - m_c)/(m_c + 1)),
          - Continuous-time Jukes-Cantor multiple-hit (back-mutation) expectation on long
            saturated branches.
        """
        deg_a = len(u_children)
        deg_b = len(v_children)
        topo_cost = split_topology_cost_nats(deg_a, deg_b)
        L = u_children[0][0].shape[0]

        all_children = u_children + v_children
        K_ch = len(all_children)
        refined_loads: List[float] = []
        if K_ch == 4 and max(c[1] for c in all_children) >= 40.0:
            masks = [c[0] for c in all_children]
            for idx_c in range(4):
                others = [masks[j] for j in range(4) if j != idx_c]
                common_others = others[0] & others[1] & others[2]
                is_single = (
                    (common_others == 1)
                    | (common_others == 2)
                    | (common_others == 4)
                    | (common_others == 8)
                )
                valid_c = masks[idx_c] != 15
                anchor_sites = is_single & valid_c
                n_anc = int(np.sum(anchor_sites))
                if n_anc >= 50:
                    n_diff = int(np.sum(anchor_sites & ((masks[idx_c] & common_others) == 0)))
                    p_c = min(0.73, n_diff / max(n_anc, 1))
                    m_cond = -0.75 * math.log(1.0 - (4.0 / 3.0) * p_c) * L_eff * 1.05
                    m_prior = all_children[idx_c][1]
                    refined_loads.append(
                        max(1.5, min(m_prior, m_cond) if m_cond < 0.75 * m_prior else 0.5 * (m_prior + m_cond))
                    )
                else:
                    refined_loads.append(all_children[idx_c][1])
        else:
            refined_loads = [c[1] for c in all_children]

        u_loads = refined_loads[:deg_a]
        v_loads = refined_loads[deg_a:]

        u_miss = np.zeros((4, L), dtype=np.int16)
        v_miss = np.zeros((4, L), dtype=np.int16)
        u_shared = np.zeros((4, L), dtype=np.int16)
        v_shared = np.zeros((4, L), dtype=np.int16)
        C_u = np.zeros((4, L), dtype=np.float64)
        C_v = np.zeros((4, L), dtype=np.float64)

        u_valid = np.zeros(L, dtype=np.int16)
        v_valid = np.zeros(L, dtype=np.int16)
        u_weights: List[float] = []
        v_weights: List[float] = []

        for (s_mask, _), m_c in zip(u_children, u_loads):
            m_clamped = min(max(m_c, 0.5), 0.48 * L_eff)
            w_c = math.log(max(L_eff - m_clamped, 1.5) / (m_clamped + 1.0))
            u_weights.append(w_c)
            non_gap = s_mask != 15
            u_valid += non_gap.astype(np.int16)
            for b in range(4):
                has_b = ((s_mask >> b) & 1) != 0
                miss_b = (~has_b) & non_gap
                u_miss[b] += miss_b.astype(np.int16)
                u_shared[b] += (has_b & non_gap).astype(np.int16)
                C_u[b] += w_c * miss_b.astype(np.float64)

        for (s_mask, _), m_c in zip(v_children, v_loads):
            m_clamped = min(max(m_c, 0.5), 0.48 * L_eff)
            w_c = math.log(max(L_eff - m_clamped, 1.5) / (m_clamped + 1.0))
            v_weights.append(w_c)
            non_gap = s_mask != 15
            v_valid += non_gap.astype(np.int16)
            for b in range(4):
                has_b = ((s_mask >> b) & 1) != 0
                miss_b = (~has_b) & non_gap
                v_miss[b] += miss_b.astype(np.int16)
                v_shared[b] += (has_b & non_gap).astype(np.int16)
                C_v[b] += w_c * miss_b.astype(np.float64)

        # Contracted star step count & weighted MDL cost
        steps_star = np.min(u_miss + v_miss, axis=0)
        C_star = np.min(C_u + C_v, axis=0)

        # Optimal state on u and v in the resolved tree
        b_u_opt = np.argmin(C_u, axis=0)
        b_v_opt = np.argmin(C_v, axis=0)
        idx_L = np.arange(L)
        min_C_u = C_u[b_u_opt, idx_L]
        min_C_v = C_v[b_v_opt, idx_L]
        steps_res = u_miss[b_u_opt, idx_L] + v_miss[b_v_opt, idx_L]

        u_asym = (max(u_weights) - min(u_weights)) if len(u_weights) >= 2 else 0.0
        v_asym = (max(v_weights) - min(v_weights)) if len(v_weights) >= 2 else 0.0

        u_clean_clade = (u_shared[b_u_opt, idx_L] >= 2) & (u_miss[b_u_opt, idx_L] == 0)
        v_clean_clade = (v_shared[b_v_opt, idx_L] >= 2) & (v_miss[b_v_opt, idx_L] == 0)
        both_parsimony = (steps_res + 1) < steps_star
        raw_delta = C_star - (min_C_u + min_C_v)

        masked_u = v_clean_clade & (u_asym >= 1.5) & (raw_delta >= 2.8)
        masked_v = u_clean_clade & (v_asym >= 1.5) & (raw_delta >= 2.8)
        both_asym = (u_asym >= 1.8) & (v_asym >= 1.8) & (raw_delta >= 3.0)

        valid_syn_mask = (
            (u_valid >= 2)
            & (v_valid >= 2)
            & (steps_star >= 2)
            & (b_u_opt != b_v_opt)
            & (both_parsimony | masked_u | masked_v | both_asym)
        )

        if not np.any(valid_syn_mask):
            return -topo_cost, 0.0, []

        # Transition / transversion state-choice savings
        same_purine = IS_PURINE[b_u_opt] == IS_PURINE[b_v_opt]
        subst_vec = np.where(same_purine, ti_cost, tv_cost) * site_weights

        has_long = (u_asym >= 1.5) or (v_asym >= 1.5)
        delta_steps = (
            np.maximum(1.0, (steps_star - steps_res).astype(np.float64))
            if has_long
            else np.ones(L, dtype=np.float64)
        )

        # Continuous-time Jukes-Cantor multiple-hit (back-mutation) expectation on long branches
        p_long_u = min(0.65, max(u_loads) / L_eff) if u_asym >= 1.5 else 0.0
        p_long_v = min(0.65, max(v_loads) / L_eff) if v_asym >= 1.5 else 0.0
        jc_mult_hit = np.zeros(L, dtype=np.float64)
        if p_long_u > 0.05:
            jc_mult_hit = np.where(
                masked_u | both_asym,
                jc_mult_hit + 0.55 * p_long_u * (min(u_weights) + subst_vec),
                jc_mult_hit,
            )
        if p_long_v > 0.05:
            jc_mult_hit = np.where(
                masked_v | both_asym,
                jc_mult_hit + 0.55 * p_long_v * (min(v_weights) + subst_vec),
                jc_mult_hit,
            )

        delta_pos = raw_delta * site_weights + jc_mult_hit
        site_savings = np.where(valid_syn_mask, delta_pos + delta_steps * subst_vec, 0.0)

        pos_idx = np.where(site_savings > 0.0)[0]
        if len(pos_idx) == 0:
            return -topo_cost, 0.0, []

        # Sort positive site savings descending and find exact optimal m* in {1..K}
        order = np.argsort(-site_savings[pos_idx])
        sorted_sites = pos_idx[order]
        sorted_sav = site_savings[sorted_sites]
        sorted_subst = subst_vec[sorted_sites]
        sorted_w = site_weights[sorted_sites]

        K = len(sorted_sites)
        m_arr = np.arange(1, K + 1, dtype=np.float64)
        cum_sav = np.cumsum(sorted_sav)
        cum_subst = np.cumsum(sorted_subst)

        # Internal branch cost L_branch(m) for m = 1..K:
        # Clean genome-wide multi-site clades (>= 7 clean 2-vs-2 parsimony codons spanning >= 500 bp)
        # scale marginal positional cost by w_s.
        headers = universal_int_nats_vec(m_arr)
        raw_pos_costs = log_binom_vec(L_eff, np.minimum(m_arr, 0.49 * L_eff))
        clean_sites = np.where(both_parsimony & u_clean_clade & v_clean_clade)[0]
        if (
            len(clean_sites) >= 7
            and int(clean_sites[-1] - clean_sites[0]) >= 500
            and len(np.unique(clean_sites // 3)) >= 7
        ):
            marg_pos_costs = np.diff(np.concatenate(([0.0], raw_pos_costs)))
            pos_costs = np.cumsum(marg_pos_costs * np.minimum(sorted_w, 1.0))
        else:
            pos_costs = raw_pos_costs
        branch_costs = headers + pos_costs + cum_subst

        net_data_gains = cum_sav - branch_costs
        best_idx = int(np.argmax(net_data_gains))
        best_data_gain = float(net_data_gains[best_idx])
        m_opt = float(best_idx + 1)
        chosen_sites = sorted_sites[: best_idx + 1].tolist()

        net_mdl_gain = best_data_gain - topo_cost
        return net_mdl_gain, m_opt, chosen_sites

    def _reconstruct_nodal_states_and_mutations(
        self,
        adj: Dict[int, List[int]],
        bitmasks: np.ndarray,
        majority_state: np.ndarray,
        states: Dict[Tuple[int, int], Tuple[np.ndarray, float]],
        N: int,
    ) -> Tuple[Dict[int, np.ndarray], Dict[Tuple[int, int], List[Tuple[int, int, int]]]]:
        """
        Reconstruct concrete ancestral nucleotide states at every internal node
        (breaking root ties via global alignment majority state rather than 'A',
        and never recording gap/missing positions mask==15 as mutations!).
        """
        L = bitmasks.shape[1]
        root = next((u for u in adj if u >= N and len(adj[u]) > 0), next(u for u in adj if len(adj[u]) > 0))
        parent: Dict[int, Optional[int]] = {root: None}
        order: List[int] = [root]
        q = [root]
        while q:
            curr = q.pop(0)
            for nb in adj[curr]:
                if nb not in parent:
                    parent[nb] = curr
                    order.append(nb)
                    q.append(nb)

        node_state: Dict[int, np.ndarray] = {}
        node_mask_dict: Dict[int, np.ndarray] = {}
        edge_mutations: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = {}

        maj_bit = (1 << majority_state).astype(np.uint8)

        for u in order:
            p = parent[u]
            if u < N:
                c_mask = bitmasks[u]
            else:
                children = [nb for nb in adj[u] if nb != p]
                c_mask, _ = self._combine_fitch_children([states[(c, u)] for c in children])
            node_mask_dict[u] = c_mask

            if p is not None and p in node_state:
                p_st = node_state[p]
                p_bit = (1 << p_st).astype(np.uint8)
                if u < N:
                    mpr_mask = c_mask
                else:
                    children = [nb for nb in adj[u] if nb != p]
                    mpr_mask, _ = self._combine_fitch_children(
                        [states[(c, u)] for c in children] + [(p_bit, 1.0)]
                    )
                keep_p = (mpr_mask & p_bit) != 0
                use_maj = (~keep_p) & ((mpr_mask & maj_bit) != 0)
                first_b = np.zeros(L, dtype=np.int8)
                for b in (3, 2, 1, 0):
                    first_b = np.where(((mpr_mask >> b) & 1) != 0, b, first_b)
                st = np.where(keep_p, p_st, np.where(use_maj, majority_state, first_b))
                node_state[u] = st
            else:
                use_maj = (c_mask & maj_bit) != 0
                first_b = np.zeros(L, dtype=np.int8)
                for b in (3, 2, 1, 0):
                    first_b = np.where(((c_mask >> b) & 1) != 0, b, first_b)
                node_state[u] = np.where(use_maj, majority_state, first_b)

            if p is not None:
                st_p = node_state[p]
                st_u = node_state[u]
                # Exclude sites where either endpoint is pure gap/missing data (mask == 15)
                valid_both = (node_mask_dict[p] != 15) & (node_mask_dict[u] != 15)
                diffs = np.where((st_p != st_u) & valid_both)[0]
                key = (min(p, u), max(p, u))
                if p < u:
                    edge_mutations[key] = [(int(s), int(st_p[s]), int(st_u[s])) for s in diffs]
                else:
                    edge_mutations[key] = [(int(s), int(st_u[s]), int(st_p[s])) for s in diffs]

        return node_state, edge_mutations

    def _peel_residual_shunts_mdl(
        self,
        adj: Dict[int, List[int]],
        node_labels: Dict[int, str],
        bitmasks: np.ndarray,
        majority_state: np.ndarray,
        terminal_loads: np.ndarray,
        edge_mutations: Dict[Tuple[int, int], List[Tuple[int, int, int]]],
        N: int,
        L: int,
        L_eff: float,
        ti_cost: float,
        tv_cost: float,
        seqs: List[str],
        mat: np.ndarray,
        is_coding: bool,
    ) -> List[ShuntEdge]:
        """
        Stage 3: Dual-Mode Residual Shunt Peeling.
        Evaluates whether routing parallel residual mutations through a lateral shunt
        edge strictly reduces L_total(X, G) under either:
          - Mode 1 (INTERVAL): L_mask = ln 2 + 2 ln L + L_discordant  (Homologous Recombination / HGT)
          - Mode 2 (SPARSE):   L_mask = ln 2 + ln C(M_var, m)         (Convergent / Epistatic Selection)
        Both modes use the exact same per-site vertical data savings, competing strictly
        on their universal site-mask description lengths.
        """
        active_nodes = [u for u in adj if len(adj[u]) > 0]
        V_count = max(len(active_nodes), 4)
        endpoint_cost = 2.0 * math.log(V_count)

        site_mut_counts = np.zeros(L, dtype=np.int32)
        for mut_list in edge_mutations.values():
            for s, _, _ in mut_list:
                site_mut_counts[s] += 1

        variable_sites = max(int(np.sum(site_mut_counts > 0)), 10)

        root = next((u for u in adj if u >= N and len(adj[u]) > 0), next(u for u in adj if len(adj[u]) > 0))
        parent: Dict[int, Optional[int]] = {root: None}
        q = [root]
        while q:
            curr = q.pop(0)
            for nb in adj[curr]:
                if nb not in parent:
                    parent[nb] = curr
                    q.append(nb)

        branch_by_child: Dict[int, List[Tuple[int, int, int]]] = {}
        branch_mut_load: Dict[int, float] = {}
        for u, p in parent.items():
            if p is None:
                continue
            key = (min(p, u), max(p, u))
            raw_list = edge_mutations.get(key, [])
            if p < u:
                directed_list = raw_list
            else:
                directed_list = [(s, b_p, b_u) for s, b_u, b_p in raw_list]
            branch_by_child[u] = directed_list
            branch_mut_load[u] = max(0.5, float(len(directed_list)))

        states_dir = self._compute_all_directed_fitch_states(
            adj, bitmasks, N, terminal_loads, L_eff
        )
        node_state, _ = self._reconstruct_nodal_states_and_mutations(
            adj, bitmasks, majority_state, states_dir, N
        )

        # Build non-gap validity mask for every node
        node_valid: Dict[int, np.ndarray] = {}
        for u in active_nodes:
            if u < N:
                node_valid[u] = mat[u] >= 0
            else:
                p = parent.get(u)
                nb_check = p if p is not None else adj[u][0]
                node_valid[u] = states_dir[(u, nb_check)][0] != 15

        # For internal clade nodes u >= N, also index multi-step clade-stem non-synonymous
        # mutations edge-by-edge (up to 3 internal stem edges while cumulative stem load <= 35)
        # so adaptive trajectories split across consecutive stem branches of a clade are not
        # fragmented, while tracking the exact origin edge node w_u for every mutation!
        orig_branch_by_child: Dict[int, List[Tuple[int, int, int]]] = {
            u: list(muts) for u, muts in branch_by_child.items()
        }
        orig_branch_load: Dict[int, float] = dict(branch_mut_load)
        origin_node: Dict[Tuple[int, int], int] = {}
        for u, muts in orig_branch_by_child.items():
            for s, _, _ in muts:
                origin_node[(u, s)] = u

        eff_anc_node: Dict[int, int] = {}
        for u in active_nodes:
            p = parent.get(u)
            if p is None:
                continue
            eff_anc = p
            if is_coding and u >= N:
                cum_load = orig_branch_load.get(u, 1.0)
                curr_w = p
                steps_up = 0
                existing_sites = {s for s, _, _ in branch_by_child.get(u, [])}
                st_u_arr = node_state[u]
                added_any = False
                while curr_w >= N and steps_up < 3:
                    next_p = parent.get(curr_w)
                    if next_p is None:
                        break
                    add_load = orig_branch_load.get(curr_w, 1.0)
                    if cum_load + add_load > 48.0:
                        break
                    cum_load += add_load
                    st_anc_arr = node_state[next_p]
                    for s_int, a_st, b_st in orig_branch_by_child.get(curr_w, []):
                        if s_int not in existing_sites and node_valid[u][s_int] and int(st_u_arr[s_int]) == b_st:
                            c_start = 3 * (s_int // 3)
                            if c_start + 2 < L:
                                cod_u = "".join("ACGT"[st_u_arr[c_start + k]] for k in range(3))
                                cod_a = "".join("ACGT"[st_anc_arr[c_start + k]] for k in range(3))
                                if CODON_TABLE.get(cod_u, "X") != CODON_TABLE.get(cod_a, "X"):
                                    branch_by_child[u].append((s_int, a_st, b_st))
                                    origin_node[(u, s_int)] = curr_w
                                    existing_sites.add(s_int)
                                    added_any = True
                    eff_anc = next_p
                    curr_w = next_p
                    steps_up += 1
                if added_any:
                    branch_mut_load[u] = max(branch_mut_load[u], cum_load * 0.65)
            eff_anc_node[u] = eff_anc

        site_to_branches: Dict[int, List[Tuple[int, int, int]]] = {}
        for u, muts in branch_by_child.items():
            for s, a, b in muts:
                site_to_branches.setdefault(s, []).append((u, a, b))

        # Helper to check if u is an ancestor of v
        def is_ancestor(a_node: int, b_node: int) -> bool:
            curr = b_node
            while curr is not None:
                if curr == a_node:
                    return True
                curr = parent.get(curr)
            return False

        pair_shared_sites: Dict[Tuple[int, int], List[int]] = {}
        pair_priority_score: Dict[Tuple[int, int], float] = {}
        for s, entries in site_to_branches.items():
            if len(entries) < 2:
                continue
            bg_c = max(1, int(site_mut_counts[s]))
            for i in range(len(entries)):
                u, a_u, b_u = entries[i]
                w_u = origin_node.get((u, s), u)
                for j in range(i + 1, len(entries)):
                    v, a_v, b_v = entries[j]
                    if b_u == b_v:
                        w_v = origin_node.get((v, s), v)
                        if w_u == w_v:
                            continue
                        if is_ancestor(u, v) or is_ancestor(v, u):
                            continue
                        if is_ancestor(w_u, v) or is_ancestor(w_v, u):
                            continue
                        if w_u != u or w_v != v:
                            pw_u = parent.get(w_u)
                            pw_v = parent.get(w_v)
                            if pw_u == pw_v or parent.get(pw_u) == pw_v or pw_u == parent.get(pw_v):
                                continue
                        pair = (min(u, v), max(u, v))
                        pair_shared_sites.setdefault(pair, []).append(s)
                        w_s = 1.0 / bg_c
                        if is_coding:
                            c_start = 3 * (s // 3)
                            if c_start + 2 < L:
                                cod_u = "".join("ACGT"[node_state[u][c_start + k]] for k in range(3))
                                cod_v = "".join("ACGT"[node_state[v][c_start + k]] for k in range(3))
                                p_u_eff = parent.get(w_u, eff_anc_node.get(u))
                                p_v_eff = parent.get(w_v, eff_anc_node.get(v))
                                if p_u_eff is not None and p_v_eff is not None:
                                    cod_pu = "".join("ACGT"[node_state[p_u_eff][c_start + k]] for k in range(3))
                                    cod_pv = "".join("ACGT"[node_state[p_v_eff][c_start + k]] for k in range(3))
                                    aa_u = CODON_TABLE.get(cod_u, "X")
                                    aa_v = CODON_TABLE.get(cod_v, "X")
                                    if aa_u == aa_v and aa_u != CODON_TABLE.get(cod_pu, "X") and aa_v != CODON_TABLE.get(cod_pv, "X"):
                                        w_s *= 4.0
                        pair_priority_score[pair] = pair_priority_score.get(pair, 0.0) + w_s

        # Rank candidate pairs by MDL & non-synonymous priority score instead of raw count
        sorted_pairs = sorted(
            pair_shared_sites.items(),
            key=lambda kv: (pair_priority_score.get(kv[0], 0.0), len(kv[1])),
            reverse=True,
        )

        # Evaluate top 500 candidate pairs first, then commit in descending order of exact net MDL gain
        candidate_shunts: List[ShuntEdge] = []
        for (u, v), shared_sites in sorted_pairs[:500]:
            shared_sites = sorted(set(shared_sites))
            avail_sites = [
                s
                for s in shared_sites
                if node_valid[u][s] and node_valid[v][s]
            ]
            if len(avail_sites) < self.min_shunt_sites:
                continue

            best_int_gain, best_int_sites, best_span, int_data_sav, int_mask_cost = (
                self._evaluate_interval_shunt_mdl(
                    u, v, avail_sites, site_mut_counts, node_state, node_valid, eff_anc_node, parent, adj,
                    branch_mut_load, L, L_eff, variable_sites, endpoint_cost, ti_cost, tv_cost, is_coding
                )
            )

            pu_direct = parent.get(u)
            pv_direct = parent.get(v)
            local_triad = (
                pu_direct == pv_direct
                or (pu_direct is not None and parent.get(pu_direct) == pv_direct)
                or (pv_direct is not None and pu_direct == parent.get(pv_direct))
            )
            if local_triad:
                best_sp_gain, best_sp_sites, sp_data_sav, sp_mask_cost = -10.0, [], 0.0, 0.0
            else:
                best_sp_gain, best_sp_sites, sp_data_sav, sp_mask_cost = (
                    self._evaluate_sparse_shunt_mdl(
                        u, v, avail_sites, site_mut_counts, node_state, eff_anc_node, branch_mut_load,
                        L, L_eff, variable_sites, endpoint_cost, ti_cost, tv_cost, is_coding
                    )
                )

            if best_int_gain <= 0.0 and best_sp_gain <= 0.0:
                continue

            if best_int_gain >= best_sp_gain and best_int_gain > 0.0:
                chosen_mode = "INTERVAL"
                chosen_sites = best_int_sites
                chosen_span = best_span
                data_sav = int_data_sav
                mask_cost = int_mask_cost
                net_gain = best_int_gain
            else:
                chosen_mode = "SPARSE"
                chosen_sites = best_sp_sites
                chosen_span = None
                data_sav = sp_data_sav
                mask_cost = sp_mask_cost
                net_gain = best_sp_gain

            post_prob = 1.0 / (1.0 + math.exp(-np.clip(net_gain, -50.0, 50.0)))
            candidate_shunts.append(
                ShuntEdge(
                    node_u=u,
                    node_v=v,
                    label_u=node_labels.get(u, f"N{u}"),
                    label_v=node_labels.get(v, f"N{v}"),
                    mode=chosen_mode,
                    sites=chosen_sites,
                    interval_span=chosen_span,
                    shared_mutations=len(chosen_sites),
                    data_savings_nats=data_sav,
                    mask_cost_nats=mask_cost,
                    endpoint_cost_nats=endpoint_cost,
                    net_mdl_gain_nats=net_gain,
                    posterior_prob=post_prob,
                )
            )

        # Commit certified shunts in strictly descending order of exact net MDL gain
        candidate_shunts.sort(key=lambda s: s.net_mdl_gain_nats, reverse=True)
        certified_shunts: List[ShuntEdge] = []
        used_sites_per_node: Dict[int, Set[int]] = {}
        for sh in candidate_shunts:
            u, v = sh.node_u, sh.node_v
            overlap = {
                s
                for s in sh.sites
                if (
                    s in used_sites_per_node.get(u, set())
                    or s in used_sites_per_node.get(v, set())
                    or s in used_sites_per_node.get(origin_node.get((u, s), u), set())
                    or s in used_sites_per_node.get(origin_node.get((v, s), v), set())
                )
            }
            if len(sh.sites) - len(overlap) < self.min_shunt_sites:
                continue
            if overlap:
                # Re-verify remaining sites if any site was claimed by a higher-gain shunt
                rem_sites = [s for s in sh.sites if s not in overlap]
                if sh.mode == "INTERVAL":
                    g_r, sites_r, span_r, sav_r, mc_r = self._evaluate_interval_shunt_mdl(
                        u, v, rem_sites, site_mut_counts, node_state, node_valid, eff_anc_node, parent, adj,
                        branch_mut_load, L, L_eff, variable_sites, endpoint_cost, ti_cost, tv_cost, is_coding
                    )
                else:
                    g_r, sites_r, sav_r, mc_r = self._evaluate_sparse_shunt_mdl(
                        u, v, rem_sites, site_mut_counts, node_state, eff_anc_node, branch_mut_load,
                        L, L_eff, variable_sites, endpoint_cost, ti_cost, tv_cost, is_coding
                    )
                    span_r = None
                if g_r <= 0.0:
                    continue
                sh = ShuntEdge(
                    node_u=u,
                    node_v=v,
                    label_u=sh.label_u,
                    label_v=sh.label_v,
                    mode=sh.mode,
                    sites=sites_r,
                    interval_span=span_r,
                    shared_mutations=len(sites_r),
                    data_savings_nats=sav_r,
                    mask_cost_nats=mc_r,
                    endpoint_cost_nats=endpoint_cost,
                    net_mdl_gain_nats=g_r,
                    posterior_prob=1.0 / (1.0 + math.exp(-np.clip(g_r, -50.0, 50.0))),
                )
            certified_shunts.append(sh)
            for s in sh.sites:
                used_sites_per_node.setdefault(u, set()).add(s)
                used_sites_per_node.setdefault(v, set()).add(s)
                used_sites_per_node.setdefault(origin_node.get((u, s), u), set()).add(s)
                used_sites_per_node.setdefault(origin_node.get((v, s), v), set()).add(s)

        certified_shunts.sort(key=lambda s: s.net_mdl_gain_nats, reverse=True)
        return certified_shunts

    def _compute_parallel_site_savings(
        self,
        u: int,
        v: int,
        shared_sites: List[int],
        site_mut_counts: np.ndarray,
        node_state: Dict[int, np.ndarray],
        eff_anc_node: Dict[int, int],
        branch_mut_load: Dict[int, float],
        L_eff: float,
        variable_sites: int,
        ti_cost: float,
        tv_cost: float,
        is_coding: bool,
    ) -> List[Tuple[int, float]]:
        """
        Compute the exact per-site vertical description-length savings for each
        parallel mutation s shared between non-sister branches u and v.
        Used identically by both INTERVAL and SPARSE shunt evaluators so that
        mode selection is determined strictly by the universal site-mask code!
        """
        p_u = eff_anc_node.get(u)
        p_v = eff_anc_node.get(v)
        if p_u is None or p_v is None:
            return []

        st_u, st_pu = node_state[u], node_state[p_u]
        st_v, st_pv = node_state[v], node_state[p_v]
        m_u = branch_mut_load.get(u, 1.0)
        m_v = branch_mut_load.get(v, 1.0)

        results: List[Tuple[int, float]] = []
        for s in shared_sites:
            if st_u[s] != st_pu[s] and st_v[s] != st_pv[s] and st_u[s] == st_v[s]:
                bg_muts = max(0, int(site_mut_counts[s]) - 2)
                b_anc = int(st_pu[s])
                b_der = int(st_u[s])
                subst_s = ti_cost if (IS_PURINE[b_anc] == IS_PURINE[b_der]) else tv_cost

                codon_bonus = 0.0
                is_synonymous = False
                if is_coding:
                    c_start = 3 * (s // 3)
                    cod_u = "".join("ACGT"[node_state[u][c_start + k]] for k in range(3))
                    cod_pu = "".join("ACGT"[node_state[p_u][c_start + k]] for k in range(3))
                    cod_v = "".join("ACGT"[node_state[v][c_start + k]] for k in range(3))
                    cod_pv = "".join("ACGT"[node_state[p_v][c_start + k]] for k in range(3))
                    aa_u = CODON_TABLE.get(cod_u, "X")
                    aa_pu = CODON_TABLE.get(cod_pu, "X")
                    aa_v = CODON_TABLE.get(cod_v, "X")
                    aa_pv = CODON_TABLE.get(cod_pv, "X")
                    if aa_u != aa_pu and aa_v != aa_pv and aa_u == aa_v:
                        codon_bonus = math.log(19.0) - subst_s
                    elif aa_u == aa_pu and aa_v == aa_pv:
                        is_synonymous = True

                expected_collision = (m_u * m_v) / max(3.0 * L_eff, 1.0)
                if is_synonymous:
                    # Neutral synonymous wobble mutation: position pool is variable_sites, no AA surprise
                    bp_term = math.log(max(variable_sites / max(0.5 * (m_u + m_v), 0.5), 1.0)) - max(
                        0.0, math.log(max(expected_collision, 1.0))
                    )
                    conservation_surprise = 0.0
                else:
                    bp_term = math.log(max(L_eff / max(0.5 * (m_u + m_v), 0.5), 1.0)) - max(
                        0.0, math.log(max(expected_collision, 1.0))
                    )
                    eff_var_pool = max(variable_sites, 1.0)
                    conservation_surprise = max(
                        0.0,
                        math.log((L_eff / eff_var_pool) * (3.0 / (bg_muts + 1.0))),
                    )

                sav = bp_term + subst_s + codon_bonus + conservation_surprise
                if sav > 0.0:
                    results.append((s, sav))
        return results

    def _evaluate_interval_shunt_mdl(
        self,
        u: int,
        v: int,
        shared_sites: List[int],
        site_mut_counts: np.ndarray,
        node_state: Dict[int, np.ndarray],
        node_valid: Dict[int, np.ndarray],
        eff_anc_node: Dict[int, int],
        parent: Dict[int, Optional[int]],
        adj: Dict[int, List[int]],
        branch_mut_load: Dict[int, float],
        L: int,
        L_eff: float,
        variable_sites: int,
        endpoint_cost: float,
        ti_cost: float,
        tv_cost: float,
        is_coding: bool,
    ) -> Tuple[float, List[int], Optional[Tuple[int, int]], float, float]:
        """
        Evaluate whether a contiguous genomic segment [a, b] between (u, v)
        compresses the alignment under the Universal Interval Mask Code:
            L_mask^interval = ln 2 + 2 ln L + L_discordant(d_[a,b], b - a + 1).
        Uses O(1) prefix sums of discordant sites and MAP midpoint breakpoint
        localization to nearest flanking discordant sites.
        """
        site_sav_pairs = self._compute_parallel_site_savings(
            u, v, shared_sites, site_mut_counts, node_state, eff_anc_node,
            branch_mut_load, L_eff, variable_sites, ti_cost, tv_cost, is_coding
        )
        K = len(site_sav_pairs)
        if K < max(3, self.min_shunt_sites):
            return -1e9, [], None, 0.0, 0.0

        cand_sites = [s for s, _ in site_sav_pairs]
        cand_savs = [sav for _, sav in site_sav_pairs]

        base_mask_cost = math.log(2.0) + 2.0 * math.log(max(L, 2))
        both_val = node_valid[u] & node_valid[v]

        # Candidate donor/recipient representatives (including direct children when u or v is a clade node)
        u_reps = [u] + [nb for nb in adj.get(u, []) if parent.get(nb) == u]
        v_reps = [v] + [nb for nb in adj.get(v, []) if parent.get(nb) == v]
        disc_prefs: List[np.ndarray] = []
        disc_pos_list: List[np.ndarray] = []
        for ur in u_reps:
            for vr in v_reps:
                d_arr = ((node_state[ur] != node_state[vr]) & both_val).astype(np.int32)
                disc_prefs.append(np.concatenate([[0], np.cumsum(d_arr)]))
                disc_pos_list.append(np.where(d_arr > 0)[0])

        gap_arr = (~both_val).astype(np.int32)
        gap_pref = np.concatenate([[0], np.cumsum(gap_arr)])
        sav_pref = np.concatenate([[0.0], np.cumsum(cand_savs)])

        best_gain = -1e9
        best_sites: List[int] = []
        best_span: Optional[Tuple[int, int]] = None
        best_data_sav = 0.0
        best_mask_cost = base_mask_cost

        for i in range(K):
            a = cand_sites[i]
            for j in range(i + max(3, self.min_shunt_sites) - 1, K):
                b = cand_sites[j]
                span_len = b - a + 1
                m_in = j - i + 1
                if span_len < 15:
                    continue
                gaps_in = int(gap_pref[b + 1] - gap_pref[a])
                if gaps_in > 0.12 * span_len:
                    continue
                disc_vals = [int(dp[b + 1] - dp[a]) for dp in disc_prefs]
                best_rep_idx = int(np.argmin(disc_vals))
                discordant = disc_vals[best_rep_idx]
                if discordant > 0.25 * m_in or (discordant / span_len) > 0.03:
                    continue
                disc_cost = (
                    branch_mutation_code_nats(float(discordant), float(span_len))
                    if discordant > 0
                    else 0.0
                )
                total_mask_cost = base_mask_cost + disc_cost
                data_sav = float(sav_pref[j + 1] - sav_pref[i])
                net_gain = data_sav - endpoint_cost - total_mask_cost
                if net_gain > best_gain:
                    best_gain = net_gain
                    best_sites = cand_sites[i : j + 1]
                    disc_positions = disc_pos_list[best_rep_idx]
                    left_disc = disc_positions[disc_positions < a]
                    right_disc = disc_positions[disc_positions > b]
                    s_left = int(left_disc[-1]) if len(left_disc) > 0 else max(0, a - 20)
                    s_right = int(right_disc[0]) if len(right_disc) > 0 else min(L - 1, b + 20)
                    a_map = (s_left + a) // 2
                    b_map = (b + s_right) // 2
                    best_span = (a_map, b_map)
                    best_data_sav = data_sav
                    best_mask_cost = total_mask_cost

        return best_gain, best_sites, best_span, best_data_sav, best_mask_cost

    def _evaluate_sparse_shunt_mdl(
        self,
        u: int,
        v: int,
        shared_sites: List[int],
        site_mut_counts: np.ndarray,
        node_state: Dict[int, np.ndarray],
        eff_anc_node: Dict[int, int],
        branch_mut_load: Dict[int, float],
        L: int,
        L_eff: float,
        variable_sites: int,
        endpoint_cost: float,
        ti_cost: float,
        tv_cost: float,
        is_coding: bool,
    ) -> Tuple[float, List[int], float, float]:
        """
        Evaluate whether a sparse point-set of parallel mutations between two
        branches (u, v) compresses the alignment under the Sparse Point-Set Code:
            L_mask^sparse(m) = ln 2 + ln C(M_var, m).
        """
        site_sav_pairs = self._compute_parallel_site_savings(
            u, v, shared_sites, site_mut_counts, node_state, eff_anc_node,
            branch_mut_load, L_eff, variable_sites, ti_cost, tv_cost, is_coding
        )
        if len(site_sav_pairs) < self.min_shunt_sites:
            return -1e9, [], 0.0, 0.0

        strict_parallel = sorted([(sav, s) for s, sav in site_sav_pairs], reverse=True)
        best_gain = -1e9
        best_sites: List[int] = []
        best_data_sav = 0.0
        best_mask_cost = 0.0

        cum_sav = 0.0
        cur_sites: List[int] = []
        M_pool = max(float(variable_sites), float(len(strict_parallel)))

        for m_idx, (sav, s) in enumerate(strict_parallel, start=1):
            cum_sav += sav
            cur_sites.append(s)
            if m_idx >= self.min_shunt_sites:
                mask_cost = (
                    math.log(2.0)
                    + universal_int_nats(float(m_idx - self.min_shunt_sites))
                    + log_binom(M_pool, float(m_idx))
                )
                net_gain = cum_sav - endpoint_cost - mask_cost
                if net_gain > best_gain:
                    best_gain = net_gain
                    best_sites = sorted(cur_sites)
                    best_data_sav = cum_sav
                    best_mask_cost = mask_cost

        return best_gain, best_sites, best_data_sav, best_mask_cost

    def _compute_star_tree_mdl_nats(self, bitmasks: np.ndarray, L_eff: float) -> float:
        """Compute total MDL code length (in nats) of a pure N-way star polytomy."""
        N, _ = bitmasks.shape
        child_list = [(bitmasks[i], 1.0) for i in range(N)]
        root_mask, _ = self._combine_fitch_children(child_list)
        total_nats = L_eff * math.log(4.0)
        for i in range(N):
            muts = float(np.sum(((bitmasks[i] & root_mask) == 0) & (bitmasks[i] != 15)))
            total_nats += branch_mutation_code_nats(muts, L_eff)
        return total_nats

    def _compute_tree_total_mdl_nats(
        self, adj: Dict[int, List[int]], branches: List[BranchRecord], L_eff: float
    ) -> float:
        """Compute total MDL code length (in nats) of the multifurcating backbone tree T_0."""
        total_nats = L_eff * math.log(4.0)
        for b in branches:
            total_nats += branch_mutation_code_nats(b.mutation_count, L_eff)
            if b.is_internal:
                deg_u = len(adj[b.u]) - 1
                deg_v = len(adj[b.v]) - 1
                total_nats += split_topology_cost_nats(deg_u, deg_v)
        return total_nats

    def _serialize_newick(
        self,
        adj: Dict[int, List[int]],
        names: List[str],
        branches: List[BranchRecord],
        N: int,
    ) -> str:
        """Serialize the multifurcating tree T_0 to standard Newick format."""
        if N == 1:
            return f"{names[0]}:0.0;"
        if N == 2:
            b_len = branches[0].branch_length if branches else 0.0
            return f"({names[0]}:{b_len/2:.6f},{names[1]}:{b_len/2:.6f});"

        edge_map: Dict[Tuple[int, int], BranchRecord] = {}
        for b in branches:
            edge_map[(min(b.u, b.v), max(b.u, b.v))] = b

        internal_nodes = [u for u in adj if u >= N and len(adj[u]) > 0]
        root = max(internal_nodes, key=lambda u: len(adj[u]))

        visited: Set[int] = set()

        def build_sub(u: int, p: Optional[int]) -> str:
            visited.add(u)
            if u < N:
                label = names[u]
            else:
                children = [nb for nb in adj[u] if nb != p and nb not in visited]
                sub_strs = [build_sub(c, u) for c in children]
                label = "(" + ",".join(sub_strs) + ")"
            if p is not None:
                rec = edge_map.get((min(u, p), max(u, p)))
                bl = rec.branch_length if rec is not None else 0.0
                if u >= N and rec is not None:
                    return f"{label}{rec.mdl_support:.3f}:{bl:.6f}"
                return f"{label}:{bl:.6f}"
            return f"{label};"

        return build_sub(root, None)


# Official BUSHWHACKER class aliases
BushwhackerEngine = MDLPhyloEngine
BushwhackerResult = MDLGraphResult


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="bushwhacker",
        description=(
            "BUSHWHACKER: Bipartition Unification, Shunt Hunting, & Weighted Hub "
            "Aggregation via Combinadic Kolmogorov Entropy Reduction."
        ),
    )
    parser.add_argument("input", help="Input nucleotide FASTA alignment file")
    parser.add_argument("-o", "--out-tree", help="Output multifurcating Newick file path")
    parser.add_argument("-j", "--out-json", help="Output JSON graph & shunt report path")
    parser.add_argument("--no-coding", action="store_true", help="Disable codon-frame awareness")
    parser.add_argument("--no-shunts", action="store_true", help="Disable lateral shunt peeling")
    parser.add_argument("--min-shunt-sites", type=int, default=2, help="Minimum shared sites per shunt (default: 2)")
    args = parser.parse_args()

    engine = BushwhackerEngine(
        coding_aware=not args.no_coding,
        detect_shunts=not args.no_shunts,
        min_shunt_sites=args.min_shunt_sites,
    )
    res = engine.infer(args.input)
    res_pct = (res.n_internal_branches / max(res.max_binary_internal_branches, 1)) * 100.0
    comp_ratio = res.raw_uncompressed_nats / max(res.full_graph_nats, 1e-6)

    print("=" * 78)
    print("BUSHWHACKER — Kolmogorov / MDL Phylogenetic Graph Induction")
    print("=" * 78)
    print(f"Alignment           : {args.input} ({res.n_taxa} taxa x {res.n_sites} bp, L_eff={res.L_eff:.1f})")
    print(f"Certified Branches  : {res.n_internal_branches}/{res.max_binary_internal_branches} ({res_pct:.1f}% resolved | Max Polytomy Degree = {res.max_polytomy_degree})")
    print(f"Code Length Cascade : Raw={res.raw_uncompressed_nats:.1f} -> Star={res.star_tree_nats:.1f} -> Tree={res.backbone_tree_nats:.1f} -> Graph={res.full_graph_nats:.1f} nats ({comp_ratio:.2f}x)")
    print(f"Certified Shunts    : {len(res.shunts)} ({sum(1 for s in res.shunts if s.mode == 'INTERVAL')} INTERVAL / {sum(1 for s in res.shunts if s.mode == 'SPARSE')} SPARSE)")
    for idx, sh in enumerate(res.shunts, start=1):
        span_str = f"span={sh.interval_span}" if sh.interval_span else f"sites={sh.sites[:10]}"
        print(f"  [{idx}] {sh.mode:<8} {sh.label_u} <-> {sh.label_v} | Gain=+{sh.net_mdl_gain_nats:.2f} nats | P={sh.posterior_prob*100:.1f}% | {span_str}")
    print(f"Runtime             : {res.runtime_sec:.3f}s")

    if args.out_tree:
        with open(args.out_tree, "w", encoding="utf-8") as f:
            f.write(res.newick + "\n")
        print(f"Saved multifurcating Newick to {args.out_tree}")

    if args.out_json:
        payload = {
            "tool": "BUSHWHACKER",
            "n_taxa": res.n_taxa,
            "n_sites": res.n_sites,
            "L_eff": res.L_eff,
            "n_internal_branches": res.n_internal_branches,
            "max_binary_internal_branches": res.max_binary_internal_branches,
            "resolution_pct": round(res_pct, 2),
            "max_polytomy_degree": res.max_polytomy_degree,
            "raw_uncompressed_nats": round(res.raw_uncompressed_nats, 2),
            "star_tree_nats": round(res.star_tree_nats, 2),
            "backbone_tree_nats": round(res.backbone_tree_nats, 2),
            "full_graph_nats": round(res.full_graph_nats, 2),
            "compression_ratio": round(comp_ratio, 3),
            "runtime_sec": round(res.runtime_sec, 4),
            "newick": res.newick,
            "shunts": [
                {
                    "node_u": s.node_u,
                    "node_v": s.node_v,
                    "label_u": s.label_u,
                    "label_v": s.label_v,
                    "mode": s.mode,
                    "sites": s.sites,
                    "interval_span": s.interval_span,
                    "shared_mutations": s.shared_mutations,
                    "net_mdl_gain_nats": round(s.net_mdl_gain_nats, 3),
                    "posterior_prob": round(s.posterior_prob, 4),
                }
                for s in res.shunts
            ],
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved JSON report to {args.out_json}")


if __name__ == "__main__":
    main()
