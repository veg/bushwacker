# Agent Execution & Scientific Audit Guide (`AGENTS.md`)

Machine-actionable instructions for AI coding, verification, and peer-review agents working inside this Zenodo replication package.

---

## 1. Environment & Binary Contract

- Run all commands from the root of this package (`REPO_ROOT`, containing `Cargo.toml`).
- **Build binary:**
  ```bash
  cargo build --release
  # Produces target/release/bushwhacker
  ```
- **Deterministic multi-threading:** `bushwhacker -T <n>` produces bit-identical trees/ledgers across `-T 1` through `-T 64` using deterministic canonical ordering and tiled `64`-site likelihood reductions.
- **Alphabet & mode flags:**
  - Default (no flag): Nucleotide DNA/RNA + automatic third-codon phase detection (`--codon-phase` / coding-DNA).
  - `--protein`: 20-state amino acid with empirical Dayhoff/BLOSUM substitution costs (`Eq. 8`).
  - `--translate`: 64-codon to 20-state universal genetic code translation.
  - `--discrete`: Unordered multi-state discrete somatic SNVs (`2 <= K <= 32`; `0=WT`, `1=Mut`, `?/-=dropout`).
  - `--ordered`: Ordered step-weighted zygosity / copy-number states (`3 <= K <= 16`; `0=WT, 1=HET, 2=HOM` or `0..11+` CNA bins).
  - Default continuous-time rate-category models: Fixed 8-category unit-mean $\Gamma(1,1)$ octile rate models (`GTR+CAT8` DNA for $N \ge 4$ / `LG08+CAT8` Protein for $N \ge 3$) scored with Fisher/Rissanen contraction penalties (`Eq. 9`).
  - `<input.fa> -c <out.bwz>` / `<in.bwz> -d <restored.fa>`: Lossless causal arithmetic compressor & decompressor.
  - `-j <out.json>`: Structured JSON ledger (`tree_MDL_nats`, `network_MDL_nats`, `resolved_internal_splits`, `max_binary_splits`, `polytomy_nodes`, `internal_branches[]`, `shunts[]`).
- **Benchmark path resolution:** Every script in `scripts/benchmarks/` imports `scripts/benchmarks/_env_paths.py`. No user-specific home paths (`/Users/...`, `/home/...`, `/storage/...`) are used anywhere; external tools and large public corpora can be pointed to via `BUSHWHACKER_BIN`, `IQTREE_BIN`, `FASTTREE_BIN`, `ASTRAL4_BIN`, `MAFFT_BIN`, and `EXTERNAL_CORPORA_DIR`.

---

## 2. Clean-Room Verification Sequence

Run these four checks in order; all must exit `0`:

```bash
python3 scripts/data_tools/verify_manifest_checksums.py
cargo build --release
python3 scripts/data_tools/reproduce_specimens_and_verify.py
python3 scripts/data_tools/reproduce_all_figures.py --compile-latex
```

Expected pass invariants checked by `reproduce_specimens_and_verify.py`:
1. **`data/specimens/prestin.fasta`**: Recovered `SPARSE` shunt gain `+3.91 nats` (`abs(gain - 3.912) < 0.02`), conditional local posterior `P_local = 0.9804` (`abs(prob - 0.9804) < 0.002`), exact 1-based codons `[7, 384, 392, 601]`.
2. **`data/specimens/kleb_9taxa_chrom_214kb.fna` vs. `blaKPC_9taxa.fna`**: Chromosome resolves `6/6` internal splits; *blaKPC* plasmid resolves `1/6` splits and contracts `5/6` into polytomies.
3. **`data/specimens/aml_38_001_clones.fasta --ordered`**: Recovers clonal *NPM1* trunk branching into parallel *IDH1* and *IDH2* hubs.
4. **`data/alignments/plant_c4_and_avian/Poaceae_PCK_HGT.codon_aln.fasta`**: Recovers catalytic $\text{C}_4$ convergence shunts (`>15 nats` cumulative, `+16.01 nats`; top `+12.58 nats`).
5. **`.bwz` round-trip (`InfluenzaA.fasta`, `prestin.fasta`)**: Taxon IDs and sequences after `-c` followed by `-d` match original inputs 100% identically, and `.bwz` sequence payloads compress smaller than both `gzip -9` and `xz -6`.

---

## 3. Claim-to-Artifact Traceability

- **Benchmark Tables & Figures:** All plotted values in `scripts/figures/*.py` are read verbatim from JSON ledgers in `results/` or live `bushwhacker` outputs in `data/specimens/`.
- **Main Text / Supplement Citations (`paper/`):** `science_supplement.bbl` contains `91` entries (`1..49` identical to `science_main.bbl`; `50..91` supplement-only). Cross-references cover `Eqs. 1–9`, `Figs. 1–6`, `Figs. S1–S7`, and `Tables S1–S11`.
- **Single-Cell Benchmarks:** `scripts/benchmarks/run_cancer_singlecell_bench.py` enforces `assert_zero_leakage_csv()` and `FORBIDDEN_METADATA_TOKENS` so anonymized taxa (`T00000..`) never expose clinical labels, timepoints, sample compartments, or clone sizes prior to post-hoc enrichment.

---

## 4. Scientific Integrity Rules for Automated Auditing

1. **Zero synthetic placeholders:** Never substitute analytic sigmoids, logistic curves, or hardcoded arrays for executed benchmark outputs.
2. **Zero lookahead or metadata leakage:** Strip cell multiplicities, DMS affinity scores, sampling dates, treatment/relapse tags, and anatomic labels before invoking `bushwhacker`.
3. **Exact log & checksum reporting:** Cite only counts, runtimes, SHA-256 digests, and p-values directly observed in tool output.
