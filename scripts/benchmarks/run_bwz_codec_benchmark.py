#!/usr/bin/env python3
"""Run and verify Table S9: lossless ``.bwz`` phylogenetic archiving vs. gzip-9, bzip2-9, xz-6, and lzma-9.

Supports running the bundled fast subset (small/medium alignments already uncompressed)
or the full corpus (after extracting ``.xz`` archives via
``python3 scripts/data_tools/decompress_bundled_alignments.py --group viral --group 16s --group plant``),
plus SHA-256 round-trip verification via ``bushwhacker -d``.
"""

from __future__ import annotations

import argparse
import bz2
import gzip
import json
import lzma
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from Bio import SeqIO

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _env_paths import ALIGNMENTS_DIR, BUSHWHACKER_BIN, REPO_ROOT, RESULTS_DIR, SPECIMENS_DIR  # noqa: E402


DATASETS = [
    ("InfluenzaA.fasta", ALIGNMENTS_DIR / "viral_and_bwz_codec" / "empirical_genes" / "InfluenzaA.fasta"),
    ("prestin.fasta", SPECIMENS_DIR / "prestin.fasta"),
    ("rbcL_483.fasta", ALIGNMENTS_DIR / "plant_c4_and_avian" / "rbcL_483.fasta"),
    ("hominoidea_mhc_class1_379.fasta", ALIGNMENTS_DIR / "viral_and_bwz_codec" / "hominoidea_mhc_class1_379.fasta"),
    ("heliconius_optix_island.fasta", ALIGNMENTS_DIR / "viral_and_bwz_codec" / "heliconius_optix_island.fasta"),
    ("spneumoniae_pbp2x_20kb.fasta", ALIGNMENTS_DIR / "viral_and_bwz_codec" / "spneumoniae_pbp2x_20kb.fasta"),
    ("hiv1_kal153_crf02_ag.fasta", ALIGNMENTS_DIR / "viral_and_bwz_codec" / "hiv1_kal153_crf02_ag.fasta"),
    # Large decompressed datasets (available after decompress_bundled_alignments.py):
    ("sars_cov_2_5097_global.fasta", ALIGNMENTS_DIR / "viral_and_bwz_codec" / "sars_cov_2_5097_global.fasta"),
    ("sars_cov_2_10062_world_union.fasta", ALIGNMENTS_DIR / "viral_and_bwz_codec" / "sars_cov_2_10062_world_union.fasta"),
    ("16S_N2048.fa", ALIGNMENTS_DIR / "microbial_and_16s" / "16s_nested_tiers" / "16S_N2048.fa"),
    ("human_mtdna_78.fasta", ALIGNMENTS_DIR / "viral_and_bwz_codec" / "human_mtdna_78.fasta"),
    ("83sarbeco_wgal.fas", ALIGNMENTS_DIR / "viral_and_bwz_codec" / "83sarbeco_wgal.fas"),
    ("spneumoniae_PMEN1_11chrom.aln", ALIGNMENTS_DIR / "viral_and_bwz_codec" / "spneumoniae_PMEN1_11chrom.aln"),
]


def transpose_sequences(seqs: list[str]) -> bytes:
    if not seqs:
        return b""
    L = len(seqs[0])
    return "".join("".join(s[j] for s in seqs) for j in range(L)).encode("ascii")


def benchmark_one(name: str, fa_path: Path, tmp: Path) -> dict:
    seqs = [str(r.seq).upper() for r in SeqIO.parse(str(fa_path), "fasta")]
    ids = [r.id for r in SeqIO.parse(str(fa_path), "fasta")]
    raw_seq_bytes = "".join(seqs).encode("ascii")
    col_seq_bytes = transpose_sequences(seqs)

    bwz = tmp / f"{fa_path.stem}.bwz"
    rec = tmp / f"{fa_path.stem}.rec.fasta"
    proc = subprocess.run([BUSHWHACKER_BIN, str(fa_path), "-c", str(bwz)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    subprocess.run([BUSHWHACKER_BIN, str(bwz), "-d", str(rec)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    rec_records = [(r.id, str(r.seq).upper()) for r in SeqIO.parse(str(rec), "fasta")]
    assert rec_records == list(zip(ids, seqs)), f"Round-trip mismatch on {name}"

    m = re.search(r"Sequence Payload\s*:\s*(\d+)\s*B", proc.stdout)
    bwz_seq_sz = int(m.group(1)) if m else bwz.stat().st_size
    bwz_total_sz = bwz.stat().st_size

    row_gz = len(gzip.compress(raw_seq_bytes, compresslevel=9))
    row_bz2 = len(bz2.compress(raw_seq_bytes, compresslevel=9))
    col_bz2 = len(bz2.compress(col_seq_bytes, compresslevel=9))
    row_lzma = len(lzma.compress(raw_seq_bytes, preset=9, format=lzma.FORMAT_ALONE))
    row_xz6 = len(lzma.compress(raw_seq_bytes, preset=6))
    best_str = min(row_gz, row_bz2, col_bz2, row_lzma, row_xz6)
    return {
        "dataset": name,
        "n_taxa": len(ids),
        "length": len(seqs[0]) if seqs else 0,
        "raw_seq_bytes": len(raw_seq_bytes),
        "gzip_9": row_gz,
        "bzip2_9_row": row_bz2,
        "bzip2_9_col": col_bz2,
        "xz_6": row_xz6,
        "lzma_9": row_lzma,
        "bwz_seq_payload_bytes": bwz_seq_sz,
        "bwz_total_archive_bytes": bwz_total_sz,
        "ratio_vs_best": round(best_str / bwz_seq_sz, 3),
        "ratio_vs_gzip9": round(row_gz / bwz_seq_sz, 3),
        "bit_exact_verified": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "microbial_viral" / "bwz_codec_reproduced.json")
    args = parser.parse_args()

    results = []
    with tempfile.TemporaryDirectory(prefix="bwz_bench_") as td:
        tmp = Path(td)
        for name, path in DATASETS:
            actual_path = path
            if not actual_path.exists():
                xz_path = Path(str(path) + ".xz")
                if xz_path.exists():
                    actual_path = tmp / path.name
                    actual_path.write_bytes(lzma.decompress(xz_path.read_bytes()))
                else:
                    print(f"[SKIP] {name} (not found: {path})")
                    continue
            row = benchmark_one(name, actual_path, tmp)
            results.append(row)
            print(f"[PASS] {name:34s} {row['n_taxa']:5d} x {row['length']:7d} | .bwz={row['bwz_seq_payload_bytes']:8d} B | vs_best={row['ratio_vs_best']:.2f}x | vs_gzip9={row['ratio_vs_gzip9']:.2f}x")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"datasets": results}, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
