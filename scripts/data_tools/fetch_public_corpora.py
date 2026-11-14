#!/usr/bin/env python3
"""Document and fetch multi-GB external public empirical archives referenced in the paper.

Smaller custom/curated alignments are already bundled directly inside ``data/``.
This script covers the external public multi-gigabyte repositories cited in
``DATA_SOURCES.md`` and ``paper/science_supplement.tex``.

Examples:
    python3 scripts/data_tools/fetch_public_corpora.py --list
    python3 scripts/data_tools/fetch_public_corpora.py --dataset gcreplay --dest external_corpora
"""

from __future__ import annotations

import argparse
import subprocess
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

CORPORA = {
    "abadi_gerth_osf": {
        "description": "Abadi et al. (2019) / Gerth (2019): 7,200 empirical + 27,500 INDELible alignments",
        "doi_or_url": "https://doi.org/10.17605/OSF.IO/T3PF2",
        "method": "osf_browser_or_cli",
        "notes": "Requires OSF client (`pip install osfclient && osf -p t3pf2 clone`) or browser download from OSF T3PF2.",
    },
    "rokas_shen_zhou_figshare": {
        "description": "Zhou et al. (2018) / Shen et al. (2017): 19 eukaryotic phylogenomic datasets (45,205 loci)",
        "doi_or_url": "https://figshare.com/projects/Evaluating_fast_maximum_likelihood-based_phylogenetic_programs_using_empirical_phylogenomic_data_sets/22040",
        "method": "figshare_project",
        "notes": "Download Figshare project 22040 archives via Figshare web/API into external_corpora/rokas2018/.",
    },
    "avian_shultz2019": {
        "description": "Shultz & Sackton (2019) whole-genome avian radiation alignments",
        "doi_or_url": "https://doi.org/10.5061/dryad.4b8gt83",
        "method": "dryad",
        "notes": "The 50 clean coding loci used in ASTRAL-IV evaluation are already bundled in data/alignments/plant_c4_and_avian/avian_shultz2019_50loci.tar.xz.",
    },
    "toga2_hg38_v1": {
        "description": "TOGA2 MultiCodonAlignments hg38 v1 (882 mammals; 19,890 coding / 19,952 protein loci)",
        "doi_or_url": "https://genome.senckenberg.de/download/TOGA2/MultiCodonAlignments/reference_human_hg38/v1/",
        "method": "http_index",
        "url": "https://genome.senckenberg.de/download/TOGA2/MultiCodonAlignments/reference_human_hg38/v1/allCodonAlis.tar.gz",
        "notes": "882-species supermatrices and species tree are already bundled in data/alignments/supermatrices/.",
    },
    "gtdb_r95": {
        "description": "GTDB Release 95.0 archaeal ar122 + bacterial bac120 marker alignments and reference trees",
        "doi_or_url": "https://data.gtdb.ecogenomic.org/releases/release95/95.0/",
        "method": "http_multi",
        "urls": [
            "https://data.gtdb.ecogenomic.org/releases/release95/95.0/ar122_r95.tree",
            "https://data.gtdb.ecogenomic.org/releases/release95/95.0/bac120_r95.tree",
            "https://data.gtdb.ecogenomic.org/releases/release95/95.0/ar122_msa_individual_genes_r95.tar.gz",
            "https://data.gtdb.ecogenomic.org/releases/release95/95.0/bac120_msa_individual_genes_r95.tar.gz",
        ],
        "notes": "Domain supermatrices (ar122_supermatrix.faa.xz, bac120_supermatrix_1000.faa.xz) are already bundled in data/alignments/supermatrices/.",
    },
    "greengenes_13_5": {
        "description": "Greengenes 13_5 aligned 16S rRNA archive",
        "doi_or_url": "https://greengenes.secondgenome.com/",
        "method": "http_index",
        "notes": "The 8 nested 16S tiers (N=256..32,768) evaluated in the paper are already bundled in data/alignments/microbial_and_16s/16s_nested_tiers/.",
    },
    "gcreplay": {
        "description": "GCReplay mouse germinal-center B-cell lineages & yeast-display DMS affinities",
        "doi_or_url": "https://github.com/matsengrp/gcreplay",
        "method": "git_clone",
        "url": "https://github.com/matsengrp/gcreplay.git",
        "notes": "Filtered input CSVs are already bundled in data/single_cell/bcr_gcreplay/.",
    },
    "compass_tapestri": {
        "description": "Sollier et al. (2023) COMPASS 168 myeloid Tapestri runs (1,001,609 cells)",
        "doi_or_url": "https://doi.org/10.5281/zenodo.10822292",
        "method": "zenodo_api",
        "record_id": "10822292",
    },
    "morita_aml": {
        "description": "Morita et al. (2020) 123-patient AML single-cell DNA cohort (GEO GSE146150 / Zenodo 15096570)",
        "doi_or_url": "https://zenodo.org/records/15096570",
        "method": "zenodo_api",
        "record_id": "15096570",
        "notes": "Compiled cohort JSON AML_cohort_Morita_2020.json is already bundled in data/single_cell/.",
    },
    "dlp_sitka": {
        "description": "Salehi et al. (2023) DLP+ scWGS copy-number matrices (4,950 tumor cells)",
        "doi_or_url": "https://doi.org/10.24072/pcjournal.292",
        "method": "bundled_subset",
        "notes": "Copy-number matrices are already bundled in data/single_cell/dlp_cna_sitka.tar.xz.",
    },
}


def download_url(url: str, dest_file: Path) -> None:
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    if dest_file.exists():
        print(f"[CACHED] {dest_file}")
        return
    print(f"[FETCH]  {url} -> {dest_file}")
    urllib.request.urlretrieve(url, dest_file)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List all external public corpora and DOIs/URLs")
    parser.add_argument("--dataset", choices=sorted(CORPORA.keys()), help="Fetch a specific downloadable corpus")
    parser.add_argument("--dest", type=Path, default=REPO_ROOT / "external_corpora", help="Destination directory")
    args = parser.parse_args()

    if args.list or not args.dataset:
        for k, meta in CORPORA.items():
            print(f"- {k}: {meta['description']}")
            print(f"  Source: {meta['doi_or_url']}")
            if "notes" in meta:
                print(f"  Note:   {meta['notes']}")
        return

    meta = CORPORA[args.dataset]
    target_dir = args.dest / args.dataset
    target_dir.mkdir(parents=True, exist_ok=True)

    method = meta["method"]
    if method == "git_clone":
        if (target_dir / ".git").exists():
            print(f"[CACHED] {target_dir}")
        else:
            subprocess.run(["git", "clone", "--depth", "1", meta["url"], str(target_dir)], check=True)
    elif method == "http_multi":
        for url in meta["urls"]:
            download_url(url, target_dir / Path(url).name)
    elif method == "http_index" and "url" in meta:
        download_url(meta["url"], target_dir / Path(meta["url"]).name)
    elif method == "zenodo_api":
        import json
        api_url = f"https://zenodo.org/api/records/{meta['record_id']}"
        print(f"[QUERY]  {api_url}")
        with urllib.request.urlopen(api_url) as resp:
            rec = json.loads(resp.read().decode("utf-8"))
        for f in rec.get("files", []):
            link = f["links"]["self"]
            download_url(link, target_dir / f["key"])
    else:
        print(f"Dataset '{args.dataset}' must be retrieved from: {meta['doi_or_url']}")
        if "notes" in meta:
            print(f"Instructions: {meta['notes']}")


if __name__ == "__main__":
    main()
