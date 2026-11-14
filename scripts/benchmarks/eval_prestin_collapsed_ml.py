#!/usr/bin/env python3
"""Evaluate collapsed-ML IQ-TREE 2 (--polytomy and support-thresholded consensus)
vs. Bushwhacker MDL on the 116-taxon mammalian Prestin coding alignment (`M6`).
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
from pathlib import Path
from ete3 import Tree

from _env_paths import REPO_ROOT, SPECIMENS_DIR, RESULTS_DIR, BUSHWHACKER_BIN, resolve_bin


IQTREE_BIN = resolve_bin("IQTREE_BIN", "iqtree2")


def get_bipartitions(tree: Tree, shared_leaves: list[str], min_alrt: float | None = None, min_ufboot: float | None = None) -> set[frozenset[str]]:
    all_set = frozenset(shared_leaves)
    splits: set[frozenset[str]] = set()
    for node in tree.traverse():
        if node.is_leaf() or node.is_root():
            continue
        if min_alrt is not None or min_ufboot is not None:
            parts = str(node.name).split("/")
            if len(parts) == 2:
                try:
                    alrt, ufb = float(parts[0]), float(parts[1])
                    if min_alrt is not None and alrt < min_alrt:
                        continue
                    if min_ufboot is not None and ufb < min_ufboot:
                        continue
                except ValueError:
                    pass
        clade = frozenset(node.get_leaf_names())
        if len(clade) <= 1 or len(clade) >= len(all_set) - 1:
            continue
        comp = all_set - clade
        splits.add(clade if sorted(clade)[0] < sorted(comp)[0] else comp)
    return splits


def eval_vs_species(cand: set[frozenset[str]], ref: set[frozenset[str]]) -> dict:
    tp = len(cand & ref)
    fp = len(cand - ref)
    fn = len(ref - cand)
    prec = tp / len(cand) if cand else 0.0
    rec = tp / len(ref) if ref else 0.0
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    nrf = (fp + fn) / (len(cand) + len(ref)) if (cand or ref) else 0.0
    return {
        "resolved_splits": len(cand),
        "max_binary_splits": len(ref),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1": round(f1, 6),
        "nrf": round(nrf, 6),
    }


def main() -> None:
    prestin_fa = SPECIMENS_DIR / "prestin.fasta"
    species_nwk_path = SPECIMENS_DIR / "prestin_species.nwk"

    sp_raw = species_nwk_path.read_text().strip().replace("{Foreground}", "")
    t_sp = Tree(sp_raw, format=1)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        bw_nwk = tmp / "prestin_bw.nwk"
        bw_json = tmp / "prestin_bw.json"
        t0 = time.perf_counter()
        subprocess.run(
            [BUSHWHACKER_BIN, str(prestin_fa), "-o", str(bw_nwk), "-j", str(bw_json)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        bw_sec = time.perf_counter() - t0
        bw_res = json.loads(bw_json.read_text())
        t_bw = Tree(bw_nwk.read_text().strip(), format=1)

        pre_ml = Path("/tmp/prestin_ml_poly")
        if (pre_ml.with_suffix(".treefile")).exists() and (pre_ml.with_suffix(".iqtree")).exists():
            iq_treefile = pre_ml.with_suffix(".treefile")
            iq_log = pre_ml.with_suffix(".iqtree").read_text()
            iq_sec = None
            for line in iq_log.splitlines():
                if "Total wall-clock time used:" in line:
                    iq_sec = float(line.split(":")[1].strip().split()[0])
                    break
            if iq_sec is None:
                iq_sec = 0.0
        else:
            t0 = time.perf_counter()
            pre_tmp = tmp / "prestin_ml_poly"
            subprocess.run(
                [IQTREE_BIN, "-s", str(prestin_fa), "-m", "GTR+G4", "--polytomy",
                 "-B", "1000", "-alrt", "1000", "-seed", "42", "-nt", "AUTO",
                 "-pre", str(pre_tmp)],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            iq_sec = time.perf_counter() - t0
            iq_treefile = pre_tmp.with_suffix(".treefile")

        t_ml = Tree(iq_treefile.read_text().strip(), format=1)

    shared = sorted(set(t_sp.get_leaf_names()) & set(t_bw.get_leaf_names()) & set(t_ml.get_leaf_names()))
    ref_splits = get_bipartitions(t_sp, shared)

    models = {
        "Bushwhacker_MDL": {
            "wall_sec": round(bw_sec, 4),
            "internal_runtime_sec": round(float(bw_res.get("runtime_seconds", bw_sec)), 4),
            "certified_shunts": len(bw_res.get("shunts", [])),
            **eval_vs_species(get_bipartitions(t_bw, shared), ref_splits),
        },
        "IQTREE2_GTR_G4_polytomy": {
            "wall_sec": round(iq_sec, 3),
            "certified_shunts": 0,
            **eval_vs_species(get_bipartitions(t_ml, shared), ref_splits),
        },
        "IQTREE2_UFBoot95": {
            "wall_sec": round(iq_sec, 3),
            "certified_shunts": 0,
            **eval_vs_species(get_bipartitions(t_ml, shared, min_ufboot=95.0), ref_splits),
        },
        "IQTREE2_SHaLRT80": {
            "wall_sec": round(iq_sec, 3),
            "certified_shunts": 0,
            **eval_vs_species(get_bipartitions(t_ml, shared, min_alrt=80.0), ref_splits),
        },
        "IQTREE2_SHaLRT80_UFBoot95": {
            "wall_sec": round(iq_sec, 3),
            "certified_shunts": 0,
            **eval_vs_species(get_bipartitions(t_ml, shared, min_alrt=80.0, min_ufboot=95.0), ref_splits),
        },
    }

    out = {
        "specimen": "prestin.fasta",
        "species_reference": "prestin_species.nwk ({Foreground} markers stripped)",
        "n_taxa": len(shared),
        "n_bp": 2235,
        "models": models,
    }

    for dest in [
        RESULTS_DIR / "mammalian_toga2" / "prestin_collapsed_ml_comparison.json",
        REPO_ROOT.parent / "results" / "prestin_collapsed_ml_comparison.json",
    ]:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
