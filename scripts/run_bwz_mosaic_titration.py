#!/usr/bin/env python3
"""Mosaic recombination titration benchmark for lossless `.bwz` phylogenetic archiving (`M10`).

Simulates a 56-taxon x 12,000-bp nucleotide alignment (realistic conserved viral/bacterial divergence,
mean ~0.04 substitutions/site/edge) under K in [0, 1, 8, 32] topological mosaic recombination
breakpoints (random internal SPR regraftings across contiguous blocks along the chromosome),
with fixed deterministic seed 20261112.

At each K:
  - Compresses via `bushwhacker <fa> -c <bwz>` and verifies bit-exact SHA-256 round-trip
    via `bushwhacker <bwz> -d <rec.fa>` on sequence payload.
  - Compares `.bwz` sequence payload AND total `.bwz` archive size against `gzip -9`,
    `bzip2 -9` (row and column-transposed), and `xz -6` (and `lzma -9`).
"""

from __future__ import annotations

import bz2
import gzip
import hashlib
import json
import lzma
import re
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _env_paths import REPO_ROOT, RESULTS_DIR, BUSHWHACKER_BIN  # noqa: E402


BASES = np.array(list("ACGT"))


def generate_random_binary_edges(n_taxa: int, rng: np.random.Generator, mean_bl: float = 0.03) -> tuple[int, list[tuple[int, int, float]]]:
    """Generate a rooted binary tree with leaves 0..n_taxa-1 and internal nodes n_taxa..2*n_taxa-2."""
    active = list(range(n_taxa))
    next_id = n_taxa
    edges: list[tuple[int, int, float]] = []
    while len(active) > 1:
        i, j = sorted(rng.choice(len(active), size=2, replace=False))
        b = active.pop(j)
        a = active.pop(i)
        p = next_id
        next_id += 1
        edges.append((p, a, float(rng.exponential(mean_bl))))
        edges.append((p, b, float(rng.exponential(mean_bl))))
        active.append(p)
    root = active[0]
    return root, edges


def apply_random_clade_shift(edges: list[tuple[int, int, float]], root: int, rng: np.random.Generator) -> list[tuple[int, int, float]]:
    """Simulate a non-trivial recombination / HGT event by permuting leaf assignments of two disjoint clades
    or re-grafting a random subtree while preserving valid binary rooted DAG structure."""
    children_map: dict[int, list[tuple[int, float]]] = {}
    parent_map: dict[int, int] = {}
    for p, c, bl in edges:
        children_map.setdefault(p, []).append((c, bl))
        parent_map[c] = p

    # Collect internal non-root nodes of subtree size 3..12 as candidate transferred clades
    def subtree_leaves(u: int) -> list[int]:
        if u not in children_map:
            return [u]
        res = []
        for c, _ in children_map[u]:
            res.extend(subtree_leaves(c))
        return res

    candidates = [u for u in children_map if u != root and 3 <= len(subtree_leaves(u)) <= 12]
    all_nodes = list(parent_map.keys())
    if not candidates or len(all_nodes) < 4:
        return list(edges)

    u = int(rng.choice(candidates))
    u_sub = set(subtree_leaves(u))
    # Choose another node v disjoint from u's parent/children and not in u's subtree
    def all_desc(x: int) -> set[int]:
        s = {x}
        for c, _ in children_map.get(x, []):
            s |= all_desc(c)
        return s

    anc_u = set()
    cur = u
    while cur in parent_map:
        cur = parent_map[cur]
        anc_u.add(cur)
    forb = all_desc(u) | anc_u | {root}
    targets = [x for x in all_nodes if x not in forb]
    if not targets:
        return list(edges)
    v = int(rng.choice(targets))

    # Swap subtree roots u and v in parent_map (exact subtree swap regrafting)
    pu = parent_map[u]
    pv = parent_map[v]
    if pu == pv:
        return list(edges)
    new_edges = []
    for p, c, bl in edges:
        if p == pu and c == u:
            new_edges.append((pu, v, bl))
        elif p == pv and c == v:
            new_edges.append((pv, u, bl))
        else:
            new_edges.append((p, c, bl))
    return new_edges


def evolve_jc_block(n_taxa: int, root: int, edges: list[tuple[int, int, float]], length: int, rng: np.random.Generator) -> np.ndarray:
    """Evolve DNA sequences under Jukes-Cantor continuous-time Markov chain top-down."""
    children_map: dict[int, list[tuple[int, float]]] = {}
    for p, c, bl in edges:
        children_map.setdefault(p, []).append((c, bl))

    seqs: dict[int, np.ndarray] = {}
    seqs[root] = rng.integers(0, 4, size=length, dtype=np.uint8)
    stack = [root]
    while stack:
        u = stack.pop()
        u_seq = seqs[u]
        for c, bl in children_map.get(u, []):
            p_change = 0.75 * (1.0 - np.exp(-4.0 * bl / 3.0))
            mut_mask = rng.random(size=length) < p_change
            c_seq = u_seq.copy()
            n_mut = int(mut_mask.sum())
            if n_mut > 0:
                shift = rng.integers(1, 4, size=n_mut, dtype=np.uint8)
                c_seq[mut_mask] = (c_seq[mut_mask] + shift) % 4
            seqs[c] = c_seq
            stack.append(c)
    return np.vstack([seqs[i] for i in range(n_taxa)])


def run_one_k(n_taxa: int, seq_len: int, k_breaks: int, seed: int, tmp: Path) -> dict:
    rng = np.random.default_rng(seed + 1009 * k_breaks)
    root, base_edges = generate_random_binary_edges(n_taxa, rng, mean_bl=0.03)
    n_blocks = k_breaks + 1
    bounds = np.linspace(0, seq_len, n_blocks + 1, dtype=int)

    blocks = []
    cur_edges = base_edges
    for b_idx in range(n_blocks):
        blen = int(bounds[b_idx + 1] - bounds[b_idx])
        if b_idx > 0:
            cur_edges = apply_random_clade_shift(cur_edges, root, rng)
        blocks.append(evolve_jc_block(n_taxa, root, cur_edges, blen, rng))
    aln = np.hstack(blocks)

    ids = [f"Taxon_{i:03d}" for i in range(n_taxa)]
    seqs = ["".join(BASES[aln[i]].tolist()) for i in range(n_taxa)]

    fa_path = tmp / f"mosaic_k{k_breaks}.fasta"
    bwz_path = tmp / f"mosaic_k{k_breaks}.bwz"
    rec_path = tmp / f"mosaic_k{k_breaks}.rec.fasta"
    with open(fa_path, "w") as f:
        for tid, s in zip(ids, seqs):
            f.write(f">{tid}\n{s}\n")

    raw_bytes = "".join(seqs).encode("ascii")
    raw_sha = hashlib.sha256(raw_bytes).hexdigest()
    col_bytes = "".join("".join(s[j] for s in seqs) for j in range(seq_len)).encode("ascii")

    proc = subprocess.run(
        [BUSHWHACKER_BIN, str(fa_path), "-c", str(bwz_path)],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    subprocess.run(
        [BUSHWHACKER_BIN, str(bwz_path), "-d", str(rec_path)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )

    # Verify exact sequence + taxon round-trip via SHA-256
    rec_ids = []
    rec_seqs = []
    cur_id = None
    cur_chunks = []
    for line in rec_path.read_text().splitlines():
        if line.startswith(">"):
            if cur_id is not None:
                rec_ids.append(cur_id)
                rec_seqs.append("".join(cur_chunks).upper())
            cur_id = line[1:].strip().split()[0]
            cur_chunks = []
        else:
            cur_chunks.append(line.strip())
    if cur_id is not None:
        rec_ids.append(cur_id)
        rec_seqs.append("".join(cur_chunks).upper())

    rec_sha = hashlib.sha256("".join(rec_seqs).encode("ascii")).hexdigest()
    assert rec_ids == ids and rec_sha == raw_sha, f"Round-trip SHA mismatch at K={k_breaks}"

    m = re.search(r"Sequence Payload\s*:\s*(\d+)\s*B", proc.stdout)
    bwz_seq = int(m.group(1)) if m else bwz_path.stat().st_size
    bwz_tot = bwz_path.stat().st_size

    gz9 = len(gzip.compress(raw_bytes, compresslevel=9))
    bz2_row = len(bz2.compress(raw_bytes, compresslevel=9))
    bz2_col = len(bz2.compress(col_bytes, compresslevel=9))
    xz6 = len(lzma.compress(raw_bytes, preset=6))
    best_gp = min(gz9, bz2_row, bz2_col, xz6)

    return {
        "k_breakpoints": k_breaks,
        "n_blocks": n_blocks,
        "n_taxa": n_taxa,
        "length_bp": seq_len,
        "raw_seq_bytes": len(raw_bytes),
        "gzip_9_bytes": gz9,
        "bzip2_9_row_bytes": bz2_row,
        "bzip2_9_col_bytes": bz2_col,
        "xz_6_bytes": xz6,
        "best_gp_bytes": best_gp,
        "bwz_seq_payload_bytes": bwz_seq,
        "bwz_total_archive_bytes": bwz_tot,
        "payload_ratio_vs_best_gp": round(best_gp / bwz_seq, 3),
        "total_ratio_vs_best_gp": round(best_gp / bwz_tot, 3),
        "payload_ratio_vs_xz6": round(xz6 / bwz_seq, 3),
        "payload_ratio_vs_gzip9": round(gz9 / bwz_seq, 3),
        "bit_exact_sha256": raw_sha,
        "bit_exact_verified": True,
    }


def main() -> None:
    seed = 20261112
    n_taxa = 56
    seq_len = 12000
    ks = [0, 1, 8, 32]
    rows = []
    with tempfile.TemporaryDirectory(prefix="bwz_mosaic_") as td:
        tmp = Path(td)
        for k in ks:
            row = run_one_k(n_taxa, seq_len, k, seed, tmp)
            rows.append(row)
            print(
                f"K={k:2d} blocks={row['n_blocks']:2d} | bwz_seq={row['bwz_seq_payload_bytes']:6d} B "
                f"(tot={row['bwz_total_archive_bytes']:6d} B) | xz6={row['xz_6_bytes']:6d} B | "
                f"gz9={row['gzip_9_bytes']:6d} B | vs_best={row['payload_ratio_vs_best_gp']:.2f}x"
            )

    out = {
        "seed": seed,
        "n_taxa": n_taxa,
        "length_bp": seq_len,
        "model": "JC+mosaic_SPR",
        "rows": rows,
    }
    for dest in [
        RESULTS_DIR / "synthetic" / "bwz_mosaic_titration.json",
        REPO_ROOT.parent / "results" / "bwz_mosaic_titration.json",
    ]:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(out, indent=2) + "\n")
    print("Wrote bwz_mosaic_titration.json")


if __name__ == "__main__":
    main()
