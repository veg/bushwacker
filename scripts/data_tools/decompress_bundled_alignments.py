#!/usr/bin/env python3
"""Decompress bundled ``.xz`` and ``.tar.xz`` alignments in ``data/``.

Examples:
    python3 scripts/data_tools/decompress_bundled_alignments.py --list
    python3 scripts/data_tools/decompress_bundled_alignments.py --group plant
    python3 scripts/data_tools/decompress_bundled_alignments.py --all
"""

from __future__ import annotations

import argparse
import lzma
import shutil
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"

GROUPS = {
    "supermatrices": [
        "alignments/supermatrices/toga2_mammal882_nt.fna.xz",
        "alignments/supermatrices/toga2_mammal882_aa.faa.xz",
        "alignments/supermatrices/ar122_supermatrix.faa.xz",
        "alignments/supermatrices/bac120_supermatrix_1000.faa.xz",
    ],
    "plant": [
        "alignments/plant_c4_and_avian/avian_shultz2019_50loci.tar.xz",
    ],
    "microbial": [
        "alignments/microbial_and_16s/klebsiella_222_core_loci.tar.xz",
    ],
    "16s": [
        "alignments/microbial_and_16s/16s_nested_tiers/97_otu_taxonomy.txt.xz",
        "alignments/microbial_and_16s/16s_nested_tiers/16S_N256.fa.xz",
        "alignments/microbial_and_16s/16s_nested_tiers/16S_N512.fa.xz",
        "alignments/microbial_and_16s/16s_nested_tiers/16S_N1024.fa.xz",
        "alignments/microbial_and_16s/16s_nested_tiers/16S_N2048.fa.xz",
        "alignments/microbial_and_16s/16s_nested_tiers/16S_N4096.fa.xz",
        "alignments/microbial_and_16s/16s_nested_tiers/16S_N8192.fa.xz",
        "alignments/microbial_and_16s/16s_nested_tiers/16S_N16384.fa.xz",
        "alignments/microbial_and_16s/16s_nested_tiers/16S_N32768.fa.xz",
    ],
    "viral": [
        "alignments/viral_and_bwz_codec/sars_cov_2_10062_world_union.fasta.xz",
        "alignments/viral_and_bwz_codec/sars_cov_2_5097_global.fasta.xz",
        "alignments/viral_and_bwz_codec/sc2_africa.aligned.fasta.xz",
        "alignments/viral_and_bwz_codec/sc2_asia.aligned.fasta.xz",
        "alignments/viral_and_bwz_codec/sc2_europe.aligned.fasta.xz",
        "alignments/viral_and_bwz_codec/sc2_global.aligned.fasta.xz",
        "alignments/viral_and_bwz_codec/sc2_north-america.aligned.fasta.xz",
        "alignments/viral_and_bwz_codec/sc2_oceania.aligned.fasta.xz",
        "alignments/viral_and_bwz_codec/sc2_south-america.aligned.fasta.xz",
        "alignments/viral_and_bwz_codec/denv1_cds_nodups.fasta.xz",
        "alignments/viral_and_bwz_codec/denv2_cds_nodups.fasta.xz",
        "alignments/viral_and_bwz_codec/denv3_cds_nodups.fasta.xz",
        "alignments/viral_and_bwz_codec/denv4_cds_nodups.fasta.xz",
        "alignments/viral_and_bwz_codec/human_mtdna_78.fasta.xz",
        "alignments/viral_and_bwz_codec/83sarbeco_wgal.fas.xz",
        "alignments/viral_and_bwz_codec/spneumoniae_PMEN1_11chrom.aln.xz",
    ],
    "single_cell": [
        "single_cell/bcr_gcreplay/observed-seqs.csv.xz",
        "single_cell/dlp_cna_sitka.tar.xz",
    ],
}


def extract_file(src: Path, force: bool = False) -> None:
    if not src.exists():
        print(f"[SKIP missing] {src.relative_to(REPO_ROOT)}")
        return
    if src.name.endswith(".tar.xz"):
        dest_dir = src.parent
        marker = dest_dir / f".extracted_{src.name}"
        if marker.exists() and not force:
            print(f"[OK cached]    {src.relative_to(REPO_ROOT)}")
            return
        with tarfile.open(src, mode="r:xz") as tf:
            tf.extractall(dest_dir)
        marker.write_text("ok\n", encoding="utf-8")
        print(f"[EXTRACTED]    {src.relative_to(REPO_ROOT)} -> {dest_dir.relative_to(REPO_ROOT)}")
    elif src.name.endswith(".xz"):
        out = src.with_suffix("")
        if out.exists() and not force:
            print(f"[OK cached]    {out.relative_to(REPO_ROOT)}")
            return
        with lzma.open(src, "rb") as fin, out.open("wb") as fout:
            shutil.copyfileobj(fin, fout, length=4 * 1024 * 1024)
        print(f"[EXTRACTED]    {src.relative_to(REPO_ROOT)} -> {out.relative_to(REPO_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List bundled archive groups and files")
    parser.add_argument("--group", choices=sorted(GROUPS.keys()), action="append", help="Group(s) to extract")
    parser.add_argument("--all", action="store_true", help="Extract all bundled archives")
    parser.add_argument("--force", action="store_true", help="Re-extract even if already present")
    args = parser.parse_args()

    if args.list or (not args.all and not args.group):
        for g, items in GROUPS.items():
            print(f"{g}:")
            for rel in items:
                p = DATA_DIR / rel
                sz = f"{p.stat().st_size / 1e6:.2f} MB" if p.exists() else "missing"
                print(f"  - data/{rel} ({sz})")
        if not args.all and not args.group:
            return

    selected = list(GROUPS.keys()) if args.all else args.group
    for g in selected:
        for rel in GROUPS[g]:
            extract_file(DATA_DIR / rel, force=args.force)


if __name__ == "__main__":
    main()
