#!/usr/bin/env python3
"""Regenerate all 12 publication figures (Figs. 1-6 and S1-S6) and optionally compile LaTeX PDFs."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_SCRIPTS_DIR = REPO_ROOT / "scripts" / "figures"
PAPER_DIR = REPO_ROOT / "paper"

SCRIPTS = [
    "fig1_concept.py",
    "fig2_performance_sota.py",
    "fig3_macroevo_clean.py",
    "fig4_mammalian_atlas.py",
    "fig5_microbial_viral.py",
    "fig6_tree_scales.py",
    "figS1_didactic_mdl.py",
    "figS2_S3_synthetic_and_stress.py",
    "figS4_prestin_convergence.py",
    "figS5_kingdom_synthesis.py",
    "figS6_cancer_singlecell.py",
    "figS7_chst13_bootstrap.py",
]

EXPECTED_PDFS = [
    "fig_science_concept.pdf",
    "fig_science_performance_sota.pdf",
    "fig_science_macroevo_clean.pdf",
    "fig_science_mammalian_atlas.pdf",
    "fig_science_microbial_viral.pdf",
    "fig_science_tree_scales.pdf",
    "fig_didactic_mdl.pdf",
    "fig1_synthetic_benchmarks.pdf",
    "fig3_stress_tests.pdf",
    "fig_prestin_convergence.pdf",
    "fig_kingdom_synthesis.pdf",
    "fig_science_cancer_singlecell.pdf",
    "fig_supp_chst13_bootstrap.pdf",
]


def compile_latex(doc_stem: str) -> None:
    for step in (
        ["pdflatex", "-interaction=nonstopmode", f"{doc_stem}.tex"],
        ["bibtex", doc_stem],
        ["pdflatex", "-interaction=nonstopmode", f"{doc_stem}.tex"],
        ["pdflatex", "-interaction=nonstopmode", f"{doc_stem}.tex"],
    ):
        subprocess.run(step, cwd=PAPER_DIR, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    log_text = (PAPER_DIR / f"{doc_stem}.log").read_text(errors="replace")
    undef = re.findall(r"(?:Warning: Citation|Warning: Reference|There were undefined|multiply-defined).*", log_text)
    overfull = re.findall(r"Overfull \\hbox.*", log_text)
    pages = re.findall(r"Output written on .*? \((\d+) pages", log_text)
    assert not undef, f"{doc_stem}.log has undefined/duplicate references: {undef}"
    if doc_stem == "science_supplement":
        assert not overfull, f"{doc_stem}.log has Overfull hboxes: {overfull}"
    print(f"[PASS] Compiled {doc_stem}.pdf ({pages[0] if pages else '?'} pages, 0 ref warnings, {len(overfull)} overfull)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-latex", action="store_true", help="Also compile science_main.pdf and science_supplement.pdf")
    args = parser.parse_args()

    for name in SCRIPTS:
        script_path = FIG_SCRIPTS_DIR / name
        print(f"[RUN]  {name}")
        subprocess.run([sys.executable, str(script_path)], cwd=REPO_ROOT, check=True)

    for pdf in EXPECTED_PDFS:
        p = PAPER_DIR / pdf
        assert p.exists() and p.stat().st_size > 10_000, f"Missing or tiny PDF: {p}"
        print(f"[OK]   paper/{pdf} ({p.stat().st_size:,} B)")

    if args.compile_latex:
        compile_latex("science_main")
        compile_latex("science_supplement")
        main_bbl = re.findall(r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}", (PAPER_DIR / "science_main.bbl").read_text())
        supp_bbl = re.findall(r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}", (PAPER_DIR / "science_supplement.bbl").read_text())
        assert supp_bbl[: len(main_bbl)] == main_bbl, "Citation 1..62 lockstep mismatch between main and supplement"
        print(f"[PASS] Citation lockstep verified: main={len(main_bbl)}, supplement={len(supp_bbl)}")


if __name__ == "__main__":
    main()
