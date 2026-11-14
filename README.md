# BUSHWHACKER

**B**ipartition **U**nification, **S**hunt **H**unting, & **W**eighted **H**ub **A**ggregation via **C**ombinadic **K**olmogorov **E**ntropy **R**eduction

[![Language: Rust](https://img.shields.io/badge/Language-Rust%20%28Edition%202021%29-DEA584.svg)](#installation--rust-cli)
[![WebApp: Rust WASM](https://img.shields.io/badge/WebApp-GitHub%20Pages%20%28WASM%29-ff4f00.svg)](https://veg.github.io/bushwhacker/)
[![Technical Report](https://img.shields.io/badge/Manuscript-Science%20PDF-blue.svg)](paper/science_main.pdf)

> *"Clear the statistical underbrush."*

**BUSHWHACKER** is a first-principles **Minimum Description Length (MDL) / algorithmic Kolmogorov complexity** engine for phylogenetic graph induction, implemented natively in zero-dependency **Rust** ([`src/lib.rs`](src/lib.rs), [`src/main.rs`](src/main.rs)) with a reference **Python/NumPy** implementation ([`python/mdl_phylo.py`](python/mdl_phylo.py), [`python/bushwhacker.py`](python/bushwhacker.py)).

Instead of forcing every multiple sequence alignment onto a strictly bifurcating tree and evaluating it by in-sample point-estimate maximum likelihood `ln L(X | T, θ)`, **BUSHWHACKER** treats an alignment `X` (`N` taxa × `L` sites) as a discrete data table to be transmitted losslessly over a noiseless two-part Rissanen–Wallace sender–receiver channel:

$$
G^* = \arg\min_{G = (V, E_{\text{tree}}, E_{\text{shunt}})} \left( L_{\text{model}}(G) + L_{\text{data}}(X \mid G) \right)
$$

Every proposed evolutionary structure—an internal tree split `e = (u, v)`, a multifurcating polytomy hub, a recombinant mosaic tract (`INTERVAL` shunt), or a sparse convergent replacement event (`SPARSE` shunt)—must pay for its own exact combinatorial address cost in **nats** before it is admitted into the induced evolutionary graph.

---

## Key Capabilities

1. **Zero-Parameter, Bootstrap-Free Polytomy Calibration (`0.0% FPR`):**
   Every internal split dividing a `(d_u + d_v)`-way node into subtrees of degrees `d_u ≥ 2` and `d_v ≥ 2` pays an exact combinatorial topology penalty `L_topo(d_u, d_v) = ln(max(1, floor((d_u + d_v)/2) - 1)) + ln C(d_u + d_v, d_u) - I(d_u=d_v) ln 2` plus a Kraft-tight combinadic branch header `L_N(m_e) + ln C(L_eff, m_e)`, where `L_N(0) = ln 2` and `L_N(m ≥ 1) = -ln[0.5 ln 2 (1/ln(m+1) - 1/ln(m+2))]` satisfies `sum_{m>=0} exp(-L_N(m)) = 1`. Continuous-time 8-category rate-model refinement (`GTR+CAT8` / `LG08+CAT8` on DNA and proteins) and exact bitwise/interval search (`--discrete` / `--ordered`) pair with deterministic contraction (`MDL_Gain ≤ 0 nats`) and conditional local posteriors `P_local = 1 / (1 + exp(-G_MDL))` without bootstrap replicates or arbitrary support cutoffs.
2. **Intrinsic Immunity to Long-Branch Attraction (`96.0%` True Split in the Felsenstein Zone):**
   In the Felsenstein Zone (`b_long = 0.40`, `b_short = 0.03`, `b_int = 0.035`), a random mutation on a long branch with `m_c` mutations carries marginal combinadic positional cost `ln((L_eff - m_c) / (m_c + 1)) ≈ 0.6 nats`, whereas a mutation on a short branch carries `≈ 3.8 nats`. Uniting the two short branches yields `+5.8 nats` of compression per synapomorphy, while uniting the two long branches fails to overcome the quartet topology penalty `L_topo(2, 2) = ln 3 nats`.
3. **Dual-Mode Reticulate Shunt Peeling (`INTERVAL` vs. `SPARSE`):**
   When localized recombination or multi-site adaptive convergence occurs across distant lineages, BUSHWHACKER preserves the vertical species backbone `T_0` and peels off the horizontal/convergent signal as a reticulate shunt `e_s = (u, v, M_s)`, automatically classifying contiguous mosaic blocks (`INTERVAL` mode, `L_mask = ln 2 + 2 ln L`) versus dispersed functional convergence (`SPARSE` mode, `L_mask(m) = ln 2 + L_N(m) + ln C(M_pool, m)`).

---

## Interactive WebApp (Rust + SIMD WebAssembly)

**Live GitHub Pages App:** **[https://veg.github.io/bushwhacker/](https://veg.github.io/bushwhacker/)**

BUSHWHACKER ships with a zero-server, 100% client-side **Rust + WebAssembly** web application ([`docs/index.html`](docs/index.html)) built with the Dieter Rams / Braun industrial functionalist design language (`RhizType`) and high-DPI HTML5 `<canvas>` large-tree rendering inspired by `bigtreeviewer.net`:

- **Alignment & Character Matrix $\to$ MDL Graph Mode (Rust WASM):** Drag-and-drop any **Nucleotide**, **Protein (`20-state AA` or in-frame CDS `--translate`)**, or **Binary / Discrete / Ordered (`0/1` somatic mutations, copy-number states, morphological traits)** dataset in **FASTA/FAA**, **PHYLIP**, **NEXUS**, or **CSV/TSV** format—or load built-in presets (*Mammalian Prestin CDS & 20-State Protein `N = 116`*, *Human AML Single-Cell Clones Binary CSV `N = 256 × 31 genes`*, *Vertebrate β-Globin `N = 17`*, *Adaptive Polytomy Radiation `N = 64`*, *HGT / Recombination Mosaic `N = 24`*)—to induce the MDL tree, reticulate shunts, and lossless `.bwz` archive in milliseconds directly in the browser.
- **Big-Tree Newick / NEXUS Explorer Mode:** Paste or upload arbitrary Newick or NEXUS (`BEGIN TREES;` + `TRANSLATE`) trees (or load synthetic 250 / 1,000 / 2,500-tip super-trees) with optional zero-branch-length ($\ell \le \epsilon$) polytomy contraction.
- **Multi-Scale Visual Encoding:**
  - **Phylogram**, **Cladogram (Equal)**, and **Circular Ring** layouts with smooth pan/zoom, viewport culling, semantic Level-of-Detail (LOD) label thinning, and a live minimap navigator.
  - **Automatic Taxonomic / Clade Ribbons:** Translates UCSC assembly accessions (`HLturTru3` $\to$ `Bottlenose dolphin [HLturTru3]`) or prefix guilds (`Clade01_...`) into translucent clade sector bands and outer color bars.
  - **MDL Polytomy Diamond Hubs ($d \ge 4$) & Collapsible Subtree Wedges:** Click any internal node to fold/unfold its descendant subtree into a proportional summary triangle; `Shift + Click` to isolate a subtree.
  - **Reticulate Shunt Bezier Chords & Genome-Wide Barcode Track:** Visualizes `SPARSE` convergent shunts (orange dashed arcs) and `INTERVAL` recombinant tracts (blue solid arcs) alongside an interactive site-coherence & synapomorphy coordinate track.

---

## Installation & Rust CLI

BUSHWHACKER is written in native Rust (`Edition 2021`) with 64-site bit-plane SIMD (`POPCNT` / `TZCNT` / 64-bit SWAR Fitch) and multi-threaded `rayon` parallelism (`0.015s` on 17-taxon β-globin; `0.123s` on 116-taxon mammalian *Prestin*; `0.923s` on 4,096 taxa).

```bash
cargo build --release
```

### Usage

```text
./target/release/bushwhacker <input.fasta|faa|phy|nex|csv|tsv> [options]

Supported Input Formats & Alphabets (auto-detected):
  - FASTA / FAA (.fa, .fasta, .fna, .faa; DNA, RNA, 20-state Protein, or 0/1/0-9 discrete states)
  - PHYLIP      (.phy, .phylip; strict 10-char or relaxed, sequential or interleaved)
  - NEXUS       (.nex, .nexus, .nxs; DATA or CHARACTERS blocks, sequential or interleaved)
  - CSV / TSV   (.csv, .tsv; binary 0/1 somatic mutations, unordered K-state traits, ordered copy numbers)

Options:
  -o, --out-newick <PATH>         Output Newick file path (with MDL branch supports & lengths)
  -b, --out-binary <PATH>         Output strictly binary Newick file path (for iqtree3 -t)
  -t, --contract-tree <PATH>      Contract unsupported splits & peel shunts on an input Newick tree
  -j, --out-json <PATH>           Output JSON report path (full MDL ledger, splits, & shunts)
  -c, --compress <PATH.bwz>       Lossless phylogenetic .bwz archive (`bushwhacker <in.fa> -c <out.bwz>`)
  -d, --decompress <PATH.bwz>     Bit-identical decompression (`bushwhacker <in.bwz> -d <out.fa>`)
  --protein                       Force 20-state amino-acid LG08+CAT8 / Dayhoff-6 MDL
  --translate                     Translate in-frame coding nucleotide alignment to 20-state AA
  --discrete                      Force unordered K-state discrete character MDL (including 0/1 binary)
  --ordered                       Force ordered Wagner L1 integer state MDL (e.g. single-cell CNAs)
  --no-coding                     Disable automatic coding-frame 3rd-position entropy weighting
  --no-shunts                     Disable Stage 4 reticulate shunt peeling
  --min-shunt-sites <N>           Minimum parallel mutations to evaluate a shunt (default: 2)
  -T, --threads <N>               Worker threads for deterministic parallel execution (default: all cores)
  -h, --help                      Print help message
```

### Example: Mammalian *Prestin* (`data/specimens/prestin.fasta`, 116 Mammals, 2,235 bp)

```bash
./target/release/bushwhacker data/specimens/prestin.fasta -o prestin.nwk -j prestin.json -T 1
```

```text
==============================================================================
BUSHWHACKER Summary: data/specimens/prestin.fasta  (N=116 taxa, L=2235 sites)
------------------------------------------------------------------------------
  Raw Uncompressed DL   :     351367.4 nats
  Star Tree MDL (Null)  :      80342.7 nats
  Induced Tree MDL      :      46306.1 nats  (57.64% of Star)
  Induced Network MDL   :      46302.2 nats  (57.63% of Star)
  Resolved Splits       :   81 /  113 binary splits (19 polytomy hubs)
  Retained Shunts       :    1
    [1] SPARSE  N148 <-> N169  (+3.91 nats, P_local=98.0%, m=4 sites, codons=[7, 384, 392, 601])
  Runtime               : 0.12 s
==============================================================================
```

Without any species-tree supervision or empirical amino acid matrices, **BUSHWHACKER**:
- Resolves the 116-mammal species backbone at **93.8% precision** (`76 TP`, `5 FP`, `RF = 42` across all 116 taxa) against the curated NCBI Taxonomy (compared against `83.6%` / `18 FP` for `IQ-TREE --polytomy` and `83.2%` / `19 FP` for raw `IQ-TREE 2`), and
- Automatically recovers the classic **echolocation convergence shunt** (`SPARSE N148 Delphinidae <-> N169 Rhinolophoidea Bats`, `+3.91 nats`, `P_local = 98.04%`) on the **exact four classic echolocation codons `7, 384, 392, 601` (`N7T, I384T, S392A, G601A`)** of Liu et al. (*PNAS* 2010) and Li et al. (*Curr. Biol.* 2010).

### End-to-End Clean-Room Verification

```bash
python3 scripts/data_tools/verify_manifest_checksums.py
cargo build --release
python3 scripts/data_tools/reproduce_specimens_and_verify.py
python3 scripts/data_tools/reproduce_all_figures.py --compile-latex
```

---

## Summary of Benchmark & Stress-Test Results

Full derivations, empirical cohorts, and multi-panel figures are available in [`paper/science_main.pdf`](paper/science_main.pdf) and [`paper/science_supplement.pdf`](paper/science_supplement.pdf).

### 1. Polytomy Calibration & Support-Threshold Comparison (`N = 20` Taxa, 16 Zero-Length Polytomy Branches)

| Method / Support Cutoff | Mean False Splits (FP / 16) | False Positive Rate (FPR) | Precision (%) | Recall (%) | RF Error |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **BUSHWHACKER (Zero Bootstrap)** | **0.00** | **0.0%** | **100.0%** | **96.9%** | **0.40** |
| IQ-TREE 2 (Raw ML) | 16.00 | 100.0% | 19.6% | 97.5% | 16.10 |
| IQ-TREE 2 (`SH-aLRT ≥ 80%`) | 0.90 | 5.6% | 83.2% | 95.0% | 1.10 |
| IQ-TREE 2 (`aBayes ≥ 0.95`) | 0.70 | 4.4% | 86.2% | 95.0% | 0.90 |
| IQ-TREE 2 (`UFBoot ≥ 95%`) | 0.30 | 1.9% | 94.0% | 87.5% | 0.80 |
| FastTree 2 (Raw) | 16.00 | 100.0% | 19.6% | 97.5% | 16.10 |
| FastTree 2 (`SH ≥ 0.90`) | 0.50 | 3.1% | 88.9% | 95.0% | 0.70 |
| BioNJ (`JC69`) | 16.00 | 100.0% | 20.0% | 100.0% | 16.00 |
| Maximum Parsimony (`NNI`) | 16.00 | 100.0% | 20.0% | 100.0% | 16.00 |

### 2. Classical Felsenstein Zone (`L = 1,000 bp`, 50 Replicates)

| Method | True Split `(A,B) \| (C,D)` (%) | LBA Trap `(A,C) \| (B,D)` (%) | Polytomy (%) |
| :--- | :---: | :---: | :---: |
| **BUSHWHACKER (Ours)** | **96.0%** | **0.0%** | **2.0%** |
| IQ-TREE 2 (`GTR+G`) | 94.0% | 4.0% | 0.0% |
| FastTree 2 (`GTR+CAT`) | 90.0% | 4.0% | 0.0% |

### 3. Multi-Clade Parasite LBA (32 Taxa, 4 Long-Branch Parasites Nested in 4 Clades)

| Method | True Parasite–Host Sister Recall (%) | False Parasite–Parasite LBA Trap (%) |
| :--- | :---: | :---: |
| **BUSHWHACKER (Ours)** | **91.7%** | **0.0%** |
| IQ-TREE 2 (`GTR+G`) | 79.2% | 8.3% |
| FastTree 2 (`GTR+CAT`) | 45.8% | 4.2% |
| BioNJ (`JC69`) | 29.2% | 58.3% |
| Maximum Parsimony (`NNI`) | 8.3% | 75.0% |

### 4. Empirical Mammalian *Prestin* (116 Mammals vs. Curated NCBI Species Tree; 116 Shared Taxa)

| Method / Support Rule | Resolved Splits | True Positives (TP) | False Positives (FP) | Precision (%) | Recall (%) | F1 Score (%) | nRF Distance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BUSHWHACKER (+2 Echoloc. Shunts)** | **81** | **76** | **5** | **93.8%** | **67.3%** | **78.4%** | **0.2165** |
| IQ-TREE 2 (`GTR+G4 --polytomy`) | 110 | 92 | 18 | 83.6% | 81.4% | 82.5% | 0.1749 |
| IQ-TREE 2 (`SH-aLRT ≥ 80%`) | 83 | 79 | 4 | 95.2% | 69.9% | 80.6% | 0.1939 |
| IQ-TREE 2 (`UFBoot ≥ 95%`) | 86 | 83 | 3 | 96.5% | 73.5% | 83.4% | 0.1658 |
| IQ-TREE 2 (`SH-aLRT80 + UFBoot95`) | 77 | 77 | 0 | 100.0% | 68.1% | 81.1% | 0.1895 |

---

## Scaling Performance

**BUSHWHACKER** is engineered to be **strictly sub-quadratic in the number of taxa `N` (`O(N log N · L)`)** and **near-linear in alignment length `L` (`O(L log L)`)**:
- **64-Site Bit-Plane SIMD (`BitPlaneWord64`) & 64-Bit SWAR Fitch:** Packs 64 alignment columns into four `u64` bit-planes (`b0, b1, b2, b3`) plus a `valid: u64` mask (`6` bitwise instructions + `2` hardware `POPCNT`s per 64 sites for unweighted distances; hardware `TZCNT`/`BLSR` over non-zero difference bits for weighted Jukes–Cantor distances; and 8-site-per-instruction 64-bit SWAR Fitch parsimony union/intersection).
- **128-Bit Popcount Four-Gamete Coherence (`O(L)`):** Packs column character states across at most `128` stratified anchor taxa into stack-allocated `[u64; 2]` bitsets, evaluating Four-Gamete intersection in parallel via `rayon` and hardware `POPCNT` instructions.
- **Top-Down Recursive Pivot-Split Partitioning & Tournament Backbone (`O(N log N · L)`):** Partitions `N > 128` taxa via recursive 2-pivot Jukes–Cantor projection into bounded leaf clusters (`|C_m| ≤ 64`), computes 3-point terminal loads in parallel across leaf clusters plus `32` global anchors (`O(N L)`), and agglomerates cluster Fitch consensus roots bottom-up without ever allocating an `N × N` distance matrix or executing an `O(N³)` loop.
- **Bounded-Radius Delta-Fitch NNI & Leave-One-Out Hub Propagation (`O(N · L log L)`):** Propagates directed Fitch state updates locally across swapped edges and computes top-down polytomy hub states via `O(d · L)` leave-one-out bit-count subtraction rather than `O(d² · L)` re-aggregation.

| Taxa (`N`) | Sites (`L`) | Rust SIMD + Rayon Runtime (s) | Speedup vs. Initial (`O(N³)`) | Scaling Mode |
| :---: | :---: | :---: | :---: | :--- |
| **16** | 1,000 | **0.0034 s** | 2.1× | Taxa scaling (`L = 1,000 bp`) |
| **64** | 1,000 | **0.0092 s** | 5.9× | Taxa scaling (`L = 1,000 bp`) |
| **256** | 1,000 | **0.0202 s** | 55.4× | Taxa scaling (`L = 1,000 bp`, hierarchical `O(N log N)`) |
| **512** | 1,000 | **0.0474 s** | 164.6× | Taxa scaling (`L = 1,000 bp`, hierarchical `O(N log N)`) |
| **1,024** | 1,000 | **0.1169 s** | **517.2×** | Taxa scaling (`L = 1,000 bp`, hierarchical `O(N log N)`) |
| **2,048** | 1,000 | **0.3046 s** | — | Taxa scaling (`L = 1,000 bp`, hierarchical `O(N log N)`) |
| **4,096** | 1,000 | **0.9226 s** | — | Taxa scaling (`L = 1,000 bp`, **sub-second for 4,096 taxa**) |
| **8,192** | 1,000 | **3.2519 s** | — | Taxa scaling (`L = 1,000 bp`, hierarchical `O(N log N)`) |

---

## Repository Structure

```text
bushwhacker/
├── Cargo.toml                             # Rust crate manifest (native rayon + wasm32 wasm-bindgen)
├── README.md                              # Project documentation
├── AGENTS.md                              # Automated verification & scientific audit contract
├── DATA_SOURCES.md                        # Dataset accessions & external corpus provenance
├── MANIFEST_SHA256.json                   # SHA-256 verification ledger across package files
├── src/                                   # Native Rust + SIMD + WASM engine (CLI & core library)
├── docs/                                  # Interactive WebApp (Rust WASM bundle & examples)
├── python/
│   ├── bushwhacker.py                     # Reference Python CLI
│   └── mdl_phylo.py                       # Reference Python/NumPy MDL engine
├── data/                                  # Curated specimen alignments & reproduction inputs
├── results/                               # Precomputed benchmark JSON ledgers & figure PNGs
├── scripts/
│   ├── benchmarks/                        # Cross-domain & single-cell benchmark harnesses
│   ├── figures/                           # Main (Figs. 1–6) & supplementary (Figs. S1–S7) figure generators
│   └── data_tools/                        # Manifest, specimen, & LaTeX reproduction harnesses
└── paper/
    ├── science_main.tex                   # Main Science manuscript LaTeX source
    ├── science_main.pdf                   # Compiled main Science manuscript
    ├── science_supplement.tex             # Supplementary Materials LaTeX source
    ├── science_supplement.pdf             # Compiled Supplementary Materials
    └── fig*.pdf                           # Publication vector PDF figures (Figs. 1–6 & S1–S7)
```
