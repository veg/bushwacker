"""Centralized path and external-tool resolver for Bushwhacker replication benchmarks.

Every benchmark script in ``scripts/benchmarks/`` imports paths from here so that
bundled datasets, precomputed results, and compiled binaries resolve relative to
``REPO_ROOT`` with optional environment overrides (``BUSHWHACKER_BIN``,
``IQTREE_BIN``, ``FASTTREE_BIN``, ``ASTRAL4_BIN``, ``ASTRAL_JAR``, ``MAFFT_BIN``,
``EXTERNAL_CORPORA_DIR``).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
SPECIMENS_DIR = DATA_DIR / "specimens"
ALIGNMENTS_DIR = DATA_DIR / "alignments"
SINGLE_CELL_DIR = DATA_DIR / "single_cell"
RESULTS_DIR = REPO_ROOT / "results"
WORK_ROOT = Path(os.environ.get("BUSHWHACKER_WORK_DIR", str(REPO_ROOT / "work")))
EXTERNAL_DIR = Path(os.environ.get("EXTERNAL_CORPORA_DIR", str(REPO_ROOT / "external_corpora")))


def resolve_bin(env_var: str, *candidates: str | Path) -> str:
    """Resolve an executable from env var, explicit candidates, or system PATH."""
    val = os.environ.get(env_var)
    if val:
        return val
    for c in candidates:
        p = Path(c)
        if p.exists():
            return str(p)
        found = shutil.which(str(c))
        if found:
            return found
    return str(candidates[0]) if candidates else ""


BUSHWHACKER_BIN = resolve_bin(
    "BUSHWHACKER_BIN",
    REPO_ROOT / "target" / "release" / "bushwhacker",
    "bushwhacker",
)
IQTREE_BIN = resolve_bin("IQTREE_BIN", "iqtree2", "iqtree")
FASTTREE_BIN = resolve_bin("FASTTREE_BIN", "FastTreeMP", "FastTree", "fasttree")
ASTRAL4_BIN = resolve_bin("ASTRAL4_BIN", "astral4")
ASTRAL_JAR = resolve_bin("ASTRAL_JAR", EXTERNAL_DIR / "Astral" / "astral.5.7.8.jar", "astral.jar")
MAFFT_BIN = resolve_bin("MAFFT_BIN", "mafft")
