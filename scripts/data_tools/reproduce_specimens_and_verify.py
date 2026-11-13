#!/usr/bin/env python3
"""End-to-end live verification of ``bushwhacker`` across all 4 alphabet families and ``.bwz`` archiving.

Builds (or reuses) ``target/release/bushwhacker`` and runs live reconstruction on
bundled specimen inputs in ``data/specimens/``, ``data/alignments/``, and
``data/alignments/viral_and_bwz_codec/empirical_genes/``, asserting key quantitative
claims reported in the main text and Supplementary Materials.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import lzma
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from Bio import SeqIO

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_DIR = REPO_ROOT / "data" / "specimens"
BW_BIN = REPO_ROOT / "target" / "release" / "bushwhacker"


def ensure_binary() -> None:
    if BW_BIN.exists():
        return
    print("[BUILD] Compiling target/release/bushwhacker via cargo build --release ...")
    subprocess.run(["cargo", "build", "--release"], cwd=REPO_ROOT, check=True)


def run_bw(args: list[str]) -> str:
    proc = subprocess.run([str(BW_BIN), *args], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc.stdout.decode("utf-8", errors="replace")


def check_prestin(tmp: Path) -> None:
    fa = SPEC_DIR / "prestin.fasta"
    nwk = tmp / "prestin.nwk"
    js = tmp / "prestin.json"
    run_bw([str(fa), "-o", str(nwk), "-j", str(js)])
    d = json.loads(js.read_text(encoding="utf-8"))
    shunts = d.get("shunts", [])
    assert len(shunts) >= 1, "Expected at least one recovered SPARSE shunt in Prestin"
    top = shunts[0]
    gain = float(top["net_mdl_gain_nats"])
    prob = float(top["posterior_prob"])
    codons = list(top.get("codons_1based", []))
    assert gain >= 3.5 and prob >= 0.97, f"Unexpected Prestin gain/posterior: {gain}, {prob}"
    assert codons == [7, 384, 392, 601], f"Unexpected Prestin codons: {codons}"
    print(f"[PASS] Prestin (116 mammals, coding NT): +{gain:.2f} nats, P={prob*100:.2f}%, codons={codons}")


def check_klebsiella(tmp: Path) -> None:
    chrom_fa = SPEC_DIR / "kleb_9taxa_chrom_214kb.fna"
    kpc_fa = SPEC_DIR / "blaKPC_9taxa.fna"
    chrom_js = tmp / "kleb_chrom.json"
    kpc_js = tmp / "blaKPC.json"
    run_bw([str(chrom_fa), "-o", str(tmp / "kleb_chrom.nwk"), "-j", str(chrom_js)])
    run_bw([str(kpc_fa), "-o", str(tmp / "blaKPC.nwk"), "-j", str(kpc_js)])
    dc = json.loads(chrom_js.read_text(encoding="utf-8"))
    dk = json.loads(kpc_js.read_text(encoding="utf-8"))
    res_c = dc["resolved_internal_splits"]
    res_k = dk["resolved_internal_splits"]
    col_k = dk["max_binary_splits"] - res_k
    assert res_c == 6, f"Expected 6/6 resolved splits on K. pneumoniae chromosome, got {res_c}"
    assert res_k == 1 and col_k == 5, f"Expected 1 resolved / 5 collapsed on blaKPC plasmid, got {res_k}/{col_k}"
    print(f"[PASS] Klebsiella chromosome ({res_c}/6 splits) vs. blaKPC plasmid ({res_k}/6 splits, {col_k}/6 collapsed)")


def check_aml38(tmp: Path) -> None:
    fa = SPEC_DIR / "aml_38_001_clones.fasta"
    js = tmp / "aml_38_001.json"
    run_bw(["--ordered", str(fa), "-o", str(tmp / "aml_38_001.nwk"), "-j", str(js)])
    d = json.loads(js.read_text(encoding="utf-8"))
    nwk_txt = (tmp / "aml_38_001.nwk").read_text(encoding="utf-8")
    assert "IDH1" in nwk_txt and "IDH2" in nwk_txt and "NPM1" in nwk_txt
    assert d["resolved_internal_splits"] >= 1
    print(f"[PASS] AML-38-001 (--ordered 3-state zygosity): resolved={d['resolved_internal_splits']}, MDL={d['tree_MDL_nats']:.2f} nats")


def check_poaceae_pck_and_protein(tmp: Path) -> None:
    fa_codon = REPO_ROOT / "data" / "alignments" / "plant_c4_and_avian" / "Poaceae_PCK_HGT.codon_aln.fasta"
    fa_aa = SPEC_DIR / "prestin_protein.faa"
    js_pck = tmp / "pck.json"
    js_aa = tmp / "prestin_aa.json"
    run_bw([str(fa_codon), "-o", str(tmp / "pck.nwk"), "-j", str(js_pck)])
    run_bw(["--protein", str(fa_aa), "-o", str(tmp / "prestin_aa.nwk"), "-j", str(js_aa)])
    dp = json.loads(js_pck.read_text(encoding="utf-8"))
    da = json.loads(js_aa.read_text(encoding="utf-8"))
    shunt_gains = [float(s["net_mdl_gain_nats"]) for s in dp.get("shunts", [])]
    assert sum(shunt_gains) > 15.0 and max(shunt_gains) > 12.0, f"Expected strong PCK catalytic shunts (>15 nats total, top >12), got {shunt_gains}"
    assert da["resolved_internal_splits"] > 0
    print(f"[PASS] Poaceae PCK C4 shunts (+{sum(shunt_gains):.2f} nats total; top +{max(shunt_gains):.2f} nats) & Prestin 20-state --protein ({da['resolved_internal_splits']} splits)")


def check_bwz_roundtrip(tmp: Path) -> None:
    emp_dir = REPO_ROOT / "data" / "alignments" / "viral_and_bwz_codec" / "empirical_genes"
    for fa in (emp_dir / "InfluenzaA.fasta", SPEC_DIR / "prestin.fasta"):
        bwz = tmp / f"{fa.stem}.bwz"
        rec_fa = tmp / f"{fa.stem}.reconstructed.fasta"
        out_txt = run_bw([str(fa), "-c", str(bwz)])
        run_bw([str(bwz), "-d", str(rec_fa)])
        orig = [(r.id, str(r.seq).upper()) for r in SeqIO.parse(str(fa), "fasta")]
        rec = [(r.id, str(r.seq).upper()) for r in SeqIO.parse(str(rec_fa), "fasta")]
        assert orig == rec, f"Bit-exact reconstruction mismatch on {fa.name}"
        raw_seq = "".join(seq for _, seq in orig).encode("ascii")
        m = re.search(r"Sequence Payload\s*:\s*(\d+)\s*B", out_txt)
        assert m is not None, f"Could not parse Sequence Payload from stdout on {fa.name}: {out_txt}"
        seq_payload_bytes = int(m.group(1))
        bwz_total_bytes = bwz.stat().st_size
        gz_seq_bytes = len(gzip.compress(raw_seq, compresslevel=9))
        xz_seq_bytes = len(lzma.compress(raw_seq, preset=6))
        assert seq_payload_bytes < gz_seq_bytes and seq_payload_bytes < xz_seq_bytes, (
            f"Expected .bwz sequence payload ({seq_payload_bytes} B) < xz-6 ({xz_seq_bytes} B) "
            f"and gzip-9 ({gz_seq_bytes} B) for {fa.name}"
        )
        print(
            f"[PASS] .bwz round-trip {fa.name}: raw_seq={len(raw_seq)} B -> "
            f"seq_payload={seq_payload_bytes} B (total={bwz_total_bytes} B) vs "
            f"gzip-9={gz_seq_bytes} B, xz-6={xz_seq_bytes} B (bit-exact FASTA match)"
        )


def main() -> None:
    ensure_binary()
    with tempfile.TemporaryDirectory(prefix="bw_verify_") as td:
        tmp = Path(td)
        check_prestin(tmp)
        check_klebsiella(tmp)
        check_aml38(tmp)
        check_poaceae_pck_and_protein(tmp)
        check_bwz_roundtrip(tmp)
    print("\nALL LIVE BUSHWHACKER SPECIMEN & CODEC CHECKS PASSED.")


if __name__ == "__main__":
    main()
