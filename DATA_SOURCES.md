# Data Sources & Complete Provenance

Datasets fall into two cleanly separated classes: **(A) Custom, assembled, subsetted, or simulated alignments bundled directly inside `data/`**, and **(B) Multi-gigabyte public third-party archives** cited by accession/DOI/URL and downloadable via `scripts/data_tools/fetch_public_corpora.py`.

---

## Part A: Bundled Custom, Assembled, & Specimen Datasets (`data/`)

Use `python3 scripts/data_tools/decompress_bundled_alignments.py --all` to unpack compressed `.xz` / `.tar.xz` files before running full Tree-of-Life or `.bwz` sweeps.

### 1. Runnable Specimens (`data/specimens/`)
- `prestin.fasta`, `prestin_protein.faa`, `prestin.nwk`, `prestin.json`, `prestin_species.nwk` — $116$-mammal *Prestin* (`SLC26A5`, $2,235\text{ bp}$ coding alignment; Fig. 4A, Fig. S4) adapted from Liu et al. (2010) / Li et al. (2010).
- `CHST13_ENSG00000180767_22mammals.fasta` — Mammalian `CHST13` ($22$ species, $1,176\text{ bp}$ from `TarvD7`; Fig. 3A).
- `kleb_9taxa_chrom_214kb.{fna,nwk,json}`, `blaKPC_9taxa.{fna,nwk,json}`, `klebsiella_core_super_matrix_214kb.fna`, `klebsiella_core_plus_blaKPC_composite.fna`, `kleb_16taxa_super_fasttree.nwk` — Clinical *Klebsiella pneumoniae* chromosomal supermatrices and matched *blaKPC* plasmid alignments (`PRJNA246471`, Wyres et al. 2020; Fig. 5A).
- `150228_Clone_3-8.fasta` — Germinal-center B-cell clonal tree (`66` cells; Tas et al. 2016; `gctree`).
- `aml_38_001_clones.fasta`, `aml_38_001.nwk`, `aml_38_001.json`, `aml_clones_binary.csv`, `AML-38-001_variants.csv` — `AML-38-001` clonal somatic zygosity specimen ($11$ clones $\times 31$ loci representing $7,016$ cells; Morita et al. 2020; Fig. 6B).

### 2. Assembled Tree-of-Life Supermatrices (`data/alignments/supermatrices/`)
- `toga2_mammal882_nt.fna.xz` & `toga2_mammal882_aa.faa.xz` — 253-locus single-copy mammalian supermatrices ($882$ assemblies, $20$ orders, $116$ families; $254,524\text{ bp}$ DNA / $126,683\text{ AA}$; Fig. 2, Table S11).
- `toga2_speciesTree.nh`, `toga2_assemblies_and_species.tsv` — Senckenberg TOGA2 reference mammal species tree and assembly-to-taxonomy map.
- `ar122_supermatrix.faa.xz` ($1,672$ Archaea $\times 32,673\text{ AA}$) and `bac120_supermatrix_1000.faa.xz` ($1,000$ Bacteria $\times 45,755\text{ AA}$ across $111$ phyla) — Assembled GTDB r95 domain supermatrices (Fig. 2, Table S11).

### 3. Botanical, Poaceae $\text{C}_4$, & Clean Avian Coding Corpus (`data/alignments/plant_c4_and_avian/`)
- `Poaceae_PCK_HGT.*`, `Poaceae_PEPC_HGT.*`, `Poaceae_rbcL_C4.*` — Curated Poaceae $\text{C}_3$/$\text{C}_4$ (`pck`, `ppc`, `rbcL`) ortholog & paralog nucleotide/protein alignments (including *Alloteropsis semialata* HGT copies; Christin et al. 2007/2012, Dunning et al. 2019; Fig. 3C).
- `rbcL_483.fasta`, `rbcL_483.nwk` — 483-taxon land-plant plastid `rbcL` coding alignment ($1,398\text{ bp}$; Table S9).
- `avian_shultz2019_50loci.tar.xz` — 50 clean avian coding orthologs + `avian_39_species_topology.nwk` (Shultz & Sackton 2019) evaluated in `ASTRAL-IV`.

### 4. Clinical *Klebsiella* Core Loci & Nested 16S Scaling Tiers (`data/alignments/microbial_and_16s/`)
- `klebsiella_222_core_loci.tar.xz`, `klebsiella_222_core_benchmark.csv` — 222 single-copy core loci (DNA + AA) across 10 *K. pneumoniae* strains (`PRJNA246471`; Table S8).
- `16s_nested_tiers/16S_N{256..32768}.fa.xz` + `97_otu_taxonomy.txt.xz` — 8 nested Greengenes `13_5` 16S rRNA scaling alignments (`256` to `32,768` OTUs $\times 1,348\text{ bp}$; Fig. 2, Fig. 5D).

### 5. Pandemic Viral, Mosaic Recombination, & `.bwz` Benchmark Alignments (`data/alignments/viral_and_bwz_codec/`)
- `sars_cov_2_10062_world_union.fasta.xz` ($10,062 \times 29,903\text{ bp}$), `sars_cov_2_5097_global.fasta.xz`, regional aligned subsets + `sc2_*.metadata.tsv.xz`, and `sc2_voc_and_x_recombinants_panel.fasta` (curated from Nextstrain Open `ncov`).
- `denv{1,2,3,4}_cds_nodups.fasta.xz` + `denv{1,2,3,4}_metadata.tsv` — 5,498 non-redundant global Dengue CDS genomes (`DENV-1..4`).
- `Example2_kal153_rdp_9953nt.fas`, `hiv1_kal153_aligned.fasta`, `kal153_rdp_gard.GARD.json`, `hiv1_kal153_crf02_ag.fasta` — HIV-1 `KAL153`/`CRF02_AG` whole-genome mosaic benchmark + GARD baseline.
- `human_mtdna_78.fasta.xz` ($78 \times 16,585\text{ bp}$), `83sarbeco_wgal.fas.xz` ($83 \times 30,965\text{ bp}$), `spneumoniae_PMEN1_11chrom.aln.xz` ($11 \times 2.22\text{ Mb}$), `spneumoniae_pbp2x_20kb.fasta`, `hominoidea_mhc_class1_379.fasta`, `heliconius_optix_island.fasta` (Table S9).
- `empirical_genes/*.fasta` — 26 single-locus empirical alignments used in Tables S1, S5, and S9 (`InfluenzaA.fasta`, `bglobin.fasta`, `camelid.fasta`, `HIV_RT.fasta`, etc.).

### 6. Single-Cell Immune & Cancer Cohorts (`data/single_cell/`)
- `bcr_gcreplay/` — Processed input CSVs from `GCReplay` (`CGGnaive_sites.csv`, `MK_RS5NF_mutability.csv`, `MK_RS5NF_substitution.csv`, `final_variant_scores.csv`, `observed-seqs.csv.xz`, `gctree-node-data.csv.xz`).
- `AML_cohort_Morita_2020.json` — 123-patient AML cohort clonal genotypes (`Morita et al. 2020`, FiTree export).
- `dlp_cna_sitka.tar.xz` — 4,950 DLP+ scWGS copy-number matrices (`OV2295`, `SA501`, `SA535`, `SA609`; `Salehi et al. 2023`).

---

## Part B: Multi-GB External Public Archives (`fetch_public_corpora.py`)

Because the full raw phylogenomic benchmarks below exceed multiple gigabytes uncompressed and are permanently hosted in curated public archives, they are fetched on demand rather than duplicated verbatim:

| Corpus | Size / Scope | Public Source / DOI / Accession |
|---|---|---|
| Model-Selection Benchmark Archive (`abadi_gerth_osf`) | 7,200 empirical + 27,500 simulated alignments | OSF https://doi.org/10.17605/OSF.IO/T3PF2 |
| Eukaryotic Radiations (`rokas_shen_zhou_figshare`) | 19 datasets, 45,205 loci (83,180 ML searches) | Figshare Project `22040` |
| Avian Whole-Genome Radiation (`avian_shultz2019`) | Complete genome-wide avian loci | Dryad https://doi.org/10.5061/dryad.4b8gt83 |
| TOGA2 882-Mammal Proteome (`toga2_hg38_v1`) | 19,952 coding / 19,953 protein loci | Senckenberg TOGA2 `hg38 v1` (`allCodonAlis.tar.gz`) |
| GTDB Release 95.0 (`gtdb_r95`) | `ar122` + `bac120` marker MSAs & trees | https://data.gtdb.ecogenomic.org/releases/release95/95.0/ |
| Greengenes 16S (`greengenes_13_5`) | Full 16S rRNA release | https://greengenes.secondgenome.com/ |
| COMPASS Tapestri Myeloid (`compass_tapestri`) | 168 runs, 1,001,609 cells | Zenodo https://doi.org/10.5281/zenodo.10822292 |
| Morita AML Tapestri (`morita_aml`) | 123 patients | NCBI GEO `GSE146150` / Zenodo `15096570` |
| DLP+ scWGS (`dlp_sitka`) | 4,950 tumor genomes | Peer Community Journal https://doi.org/10.24072/pcjournal.292 |
