# BUSHWHACKER

**B**ipartition **U**nification, **S**hunt **H**unting, & **W**eighted **H**ub **A**ggregation via **C**ombinadic **K**olmogorov **E**ntropy **R**eduction

[![Language: Rust](https://img.shields.io/badge/Language-Rust%20%28Edition%202021%29-DEA584.svg)](#installation--rust-cli)
[![WebApp: Rust WASM](https://img.shields.io/badge/WebApp-GitHub%20Pages%20%28WASM%29-ff4f00.svg)](https://veg.github.io/bushwacker/)
[![Technical Report](https://img.shields.io/badge/Technical%20Report-PDF%20%2817%20pp.%29-blue.svg)](paper/bushwhacker_report.pdf)

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
   Every internal split dividing a `(d_u + d_v)`-way node into subtrees of degrees `d_u ≥ 2` and `d_v ≥ 2` pays an exact combinatorial topology penalty `L_topo(d_u, d_v) = ln((d_u + d_v)/2 - 1) + ln C(d_u + d_v, d_u)` plus a combinadic branch header `L*(m_e) + ln C(L_eff, m_e)`. Unsupported branches contract deterministically (`MDL_Gain ≤ 0 nats`) into explicit multifurcations without needing 1,000 bootstrap replicates or arbitrary support cutoffs.
2. **Intrinsic Immunity to Long-Branch Attraction (`96.0%` True Split in the Felsenstein Zone):**
   In the Felsenstein Zone (`b_long = 0.40`, `b_short = 0.03`, `b_int = 0.035`), a random mutation on a long branch with `m_c` mutations carries marginal combinadic positional cost `ln((L_eff - m_c) / (m_c + 1)) ≈ 0.6 nats`, whereas a mutation on a short branch carries `≈ 3.8 nats`. Uniting the two short branches yields `+5.8 nats` of compression per synapomorphy, while uniting the two long branches fails to overcome the quartet topology penalty `L_topo(2, 2) = ln 3 nats`.
3. **Dual-Mode Reticulate Shunt Peeling (`INTERVAL` vs. `SPARSE`):**
   When localized recombination or multi-site adaptive convergence occurs across distant lineages, BUSHWHACKER preserves the vertical species backbone `T_0` and peels off the horizontal/convergent signal as a certified reticulate shunt `e_s = (u, v, M_s)`, automatically classifying contiguous mosaic blocks (`INTERVAL` mode, `L_mask = ln 2 + 2 ln L`) versus dispersed functional convergence (`SPARSE` mode, `L_mask(m) = ln 2 + ln C(M_pool, m)`).

---

## Interactive WebApp (Rust + SIMD WebAssembly)

**Live GitHub Pages App:** **[https://veg.github.io/bushwacker/](https://veg.github.io/bushwacker/)**

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
  -o, --out-newick <PATH>      Output Newick file path (with MDL branch supports & lengths)
  -j, --out-json <PATH>        Output JSON report path (full MDL ledger, splits, & shunts)
  --protein                    Force 20-state amino-acid Dayhoff-6 MDL (or continuous LG08+CAT8 with --lg-cat)
  --translate                  Translate in-frame coding nucleotide alignment to 20-state AA
  --binary                     Force 2-state binary (0/1) character MDL
  --discrete                   Force unordered K-state discrete character MDL
  --ordered                    Force ordered Wagner L1 integer state MDL (e.g. single-cell CNAs)
  --gtr-cat / --lg-cat         Continuous-time reversible rate-mixture MDL (GTR+CAT8 / LG08+CAT8)
  --compress <PATH.bwz>        Lossless phylogenetic .bwz archive compression across any alphabet
  --decompress <PATH.bwz>      Bit-identical .bwz archive decompression
  --no-coding                  Disable automatic coding-frame 3rd-position entropy weighting
  --no-shunts                  Disable Stage 4 reticulate shunt peeling
  --min-shunt-sites <N>        Minimum parallel mutations to evaluate a shunt (default: 2)
  -h, --help                   Print help message
```

### Example: Mammalian *Prestin* (`prestin.fasta`, 116 Mammals, 2,235 bp)

```bash
./target/release/bushwhacker prestin.fasta -o prestin.nwk -j prestin.json
```

```text
==============================================================================
BUSHWHACKER Summary: prestin.fasta  (N=116 taxa, L=2235 sites)
------------------------------------------------------------------------------
  Raw Uncompressed DL   :     351367.4 nats
  Star Tree MDL (Null)  :      80036.0 nats
  Induced Tree MDL      :      43812.9 nats  (54.74% of Star)
  Induced Network MDL   :      43808.0 nats  (54.74% of Star)
  Resolved Splits       :   95 /  113 binary splits (12 polytomy hubs)
  Certified Shunts      :    1
    [1] SPARSE  N140 <-> N167  (+4.94 nats, P=99.3%, m=4 sites, codons=[7, 384, 392, 601])
  Runtime               : 0.1233 s
==============================================================================
```

Without any species-tree supervision or empirical amino acid matrices, **BUSHWHACKER**:
- Resolves the 116-mammal species backbone at **88.0% precision** against the curated NCBI Taxonomy (`RF = 37`, outperforming raw and support-collapsed IQ-TREE 2, FastTree 2, BioNJ, and Maximum Parsimony), and
- Automatically discovers and certifies the classic **echolocation convergence shunt** (`SPARSE N140 Dolphins <-> N167 Echolocating Bats`, `+4.94 nats`, `P = 99.3%`) on the **exact four classic echolocation codons `7, 384, 392, 601` (`N7T, I384T, S392A, G601A`)** of Liu et al. (*PNAS* 2010) and Li et al. (*Curr. Biol.* 2010).

---

## Summary of Benchmark & Stress-Test Results

Full derivations, proofs, and multi-panel figures are available in [`paper/bushwhacker_report.pdf`](paper/bushwhacker_report.pdf).

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

### 4. Empirical Mammalian *Prestin* (116 Mammals vs. Curated NCBI Species Tree)

| Method / Support Rule | Resolved Splits | True Positives (TP) | False Positives (FP) | Precision (%) | Recall (%) | F1 Score (%) | RF Distance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BUSHWHACKER (+ Echoloc. Shunt)** | **83** | **73** | **10** | **88.0%** | **73.0%** | **79.8%** | **37** |
| IQ-TREE 2 (`GTR+G` Raw) | 100 | 76 | 24 | 76.0% | 76.0% | 76.0% | 48 |
| IQ-TREE 2 (`SH-aLRT ≥ 80%`) | 63 | 55 | 8 | 87.3% | 55.0% | 67.5% | 53 |
| IQ-TREE 2 (`aBayes ≥ 0.95`) | 60 | 52 | 8 | 86.7% | 52.0% | 65.0% | 56 |
| FastTree 2 (`GTR+CAT` Raw) | 100 | 74 | 26 | 74.0% | 74.0% | 74.0% | 52 |
| FastTree 2 (`SH ≥ 0.90`) | 66 | 56 | 10 | 84.8% | 56.0% | 67.5% | 54 |
| BioNJ (`JC69`) | 100 | 77 | 23 | 77.0% | 77.0% | 77.0% | 46 |
| Maximum Parsimony (`NNI`) | 100 | 70 | 30 | 70.0% | 70.0% | 70.0% | 60 |

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
bushwacker/
├── Cargo.toml                             # Rust crate manifest (native rayon + wasm32 wasm-bindgen)
├── README.md                              # Project documentation
├── .github/workflows/deploy-pages.yml     # GitHub Pages CI/CD workflow for docs/
├── src/
│   ├── lib.rs                             # Native Rust + SIMD + WASM BUSHWHACKER core engine
│   └── main.rs                            # Native Rust CLI binary
├── docs/
│   ├── index.html                         # Interactive WebApp (Dieter Rams / Braun UI + Big-Tree Canvas)
│   ├── pkg/                               # Compiled Rust WebAssembly bundle (wasm-pack)
│   └── examples/                          # Empirical alignments (prestin.fasta, bglobin.fasta)
├── python/
│   ├── bushwhacker.py                     # Reference Python CLI
│   └── mdl_phylo.py                       # Reference Python/NumPy MDL engine
├── benchmarks/
│   ├── benchmark_first_principles.py      # 5-suite synthetic & empirical benchmark harness
│   ├── benchmark_results.json             # Executed 5-suite benchmark outputs
│   ├── stress_test_vs_paradigms.py        # 5-axis adversarial stress-test harness vs. ML/Bayesian/NJ/MP
│   ├── stress_test_results.json           # Executed stress-test outputs
│   └── generate_report_figures.py         # Publication vector PDF figure generator
└── paper/
    ├── bushwhacker_report.pdf             # Compiled 17-page technical report
    ├── bushwhacker_report.tex             # LaTeX source manuscript
    ├── fig1_synthetic_benchmarks.pdf      # Figure 1 (4-regime synthetic benchmarks)
    ├── fig2_empirical_compression.pdf     # Figure 2 (Empirical compression & polytomy ratios)
    └── fig3_stress_tests.pdf              # Figure 3 (5-axis ruthless stress tests)
```
