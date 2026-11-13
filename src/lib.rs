//! BUSHWHACKER: Bipartition Unification, Shunt Hunting, & Weighted Hub Aggregation
//! via Combinadic Kolmogorov Entropy Reduction.
//!
//! Native Rust implementation of the first-principles Minimum Description Length (MDL)
//! phylogenetic graph induction engine.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

pub mod discrete;
pub mod nt_gtr;
pub mod protein;

/// Configure the global Rayon worker thread pool on native targets (`-T` / `--threads`).
pub fn configure_thread_pool(num_threads: usize) {
    #[cfg(not(target_arch = "wasm32"))]
    if num_threads >= 1 {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global();
    }
    #[cfg(target_arch = "wasm32")]
    let _ = num_threads;
}

// ---------------------------------------------------------------------------
// Combinatorial Information-Theoretic Primitives (All in Nats)
// ---------------------------------------------------------------------------

/// High-precision Lanczos approximation of ln(Gamma(z)) for z > 0.
#[inline]
pub fn ln_gamma(z: f64) -> f64 {
    const COEFFS: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    if z < 0.5 {
        let pi = std::f64::consts::PI;
        return pi.ln() - (pi * z).sin().ln() - ln_gamma(1.0 - z);
    }
    let z_m1 = z - 1.0;
    let mut x = COEFFS[0];
    for (i, &c) in COEFFS.iter().enumerate().skip(1) {
        x += c / (z_m1 + i as f64);
    }
    let t = z_m1 + 7.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (z_m1 + 0.5) * t.ln() - t + x.ln()
}

/// Continuous exact log-binomial coefficient ln C(n, k) via ln_gamma.
#[inline]
pub fn log_binom(n: f64, k: f64) -> f64 {
    if k <= 0.0 || k >= n || n <= 0.0 {
        return 0.0;
    }
    ln_gamma(n + 1.0) - ln_gamma(k + 1.0) - ln_gamma(n - k + 1.0)
}

/// Exact parameter-free Kraft-tight universal integer prefix code length (in nats)
/// for non-negative integer `m` via telescoping logarithmic tail:
/// P(0) = 1/2, P(m) = (ln 2 / 2) * (1 / ln(m + 1) - 1 / ln(m + 2)) for m >= 1,
/// satisfying sum_{m >= 0} exp(-L_N(m)) = 1.0 identically.
#[inline]
pub fn universal_int_nats(m: f64) -> f64 {
    let ln2 = 2.0_f64.ln();
    if m <= 0.0 {
        return ln2;
    }
    let s1 = 0.5 * ln2 / 2.0_f64.ln(); // 0.5
    let s2 = 0.5 * ln2 / 3.0_f64.ln();
    let p1 = s1 - s2;
    if m < 1.0 {
        return (1.0 - m) * ln2 + m * (-p1.ln());
    }
    let sm = 0.5 * ln2 / (m + 1.0).ln();
    let smp1 = 0.5 * ln2 / (m + 2.0).ln();
    -((sm - smp1).max(1e-300)).ln()
}

/// Regularized lower incomplete gamma function P(a, x) = gamma(a, x) / Gamma(a).
pub fn regularized_gamma_p(a: f64, x: f64) -> f64 {
    if a <= 0.0 || x <= 0.0 {
        return 0.0;
    }
    let ln_ga = ln_gamma(a);
    if x < a + 1.0 {
        let mut ap = a;
        let mut sum = 1.0 / a;
        let mut del = sum;
        for _ in 0..400 {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() <= sum.abs() * 1e-15 {
                break;
            }
        }
        let log_pref = -x + a * x.ln() - ln_ga;
        (sum * log_pref.exp()).clamp(0.0, 1.0)
    } else {
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / 1e-30;
        let mut d = if b.abs() < 1e-30 { 1e30 } else { 1.0 / b };
        let mut h = d;
        for i in 1..400 {
            let fi = i as f64;
            let an = -fi * (fi - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < 1e-30 {
                d = 1e-30;
            }
            c = b + an / c;
            if c.abs() < 1e-30 {
                c = 1e-30;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() <= 1e-15 {
                break;
            }
        }
        let log_pref = -x + a * x.ln() - ln_ga;
        let q = (h * log_pref.exp()).clamp(0.0, 1.0);
        (1.0 - q).clamp(0.0, 1.0)
    }
}

/// Exact analytical equiprobable octile means of the unit-mean Gamma(alpha, alpha) rate distribution.
pub fn gamma_octile_rates(alpha: f64) -> [f32; 8] {
    let a = alpha.clamp(0.15, 25.0);
    let mut quantiles = [0.0_f64; 9];
    for i in 1..8 {
        let p = (i as f64) / 8.0;
        let mut lo = 0.0_f64;
        let mut hi = (1.0_f64).max(2.0 / a);
        while regularized_gamma_p(a, a * hi) < p && hi < 1e6 {
            hi *= 2.0;
        }
        for _ in 0..80 {
            let mid = 0.5 * (lo + hi);
            if regularized_gamma_p(a, a * mid) < p {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        quantiles[i] = 0.5 * (lo + hi);
    }
    let mut partial_means = [0.0_f64; 9];
    for i in 1..8 {
        partial_means[i] = regularized_gamma_p(a + 1.0, a * quantiles[i]);
    }
    partial_means[8] = 1.0;

    let mut raw = [0.0_f64; 8];
    let mut sum = 0.0_f64;
    for c in 0..8 {
        raw[c] = (8.0 * (partial_means[c + 1] - partial_means[c])).max(1e-6);
        sum += raw[c];
    }
    let norm = 8.0 / sum;
    let mut out = [0.0_f32; 8];
    for c in 0..8 {
        out[c] = (raw[c] * norm) as f32;
    }
    out
}

/// Method-of-moments Gamma(alpha, alpha)-Poisson shape estimator from per-site Fitch step counts.
pub fn estimate_gamma_alpha_from_fitch(steps_and_nv: &[(u32, u32)]) -> f64 {
    if steps_and_nv.is_empty() {
        return 1.0;
    }
    let mut sum_steps = 0.0_f64;
    let mut sum_nv = 0.0_f64;
    for &(st, nv) in steps_and_nv {
        sum_steps += st as f64;
        sum_nv += nv as f64;
    }
    if sum_steps <= 1.0 || sum_nv <= 1.0 {
        return 1.0;
    }
    let rate_per_nv = sum_steps / sum_nv;
    let mut sum_l2 = 0.0_f64;
    let mut sum_var_excess = 0.0_f64;
    for &(st, nv) in steps_and_nv {
        let lam = rate_per_nv * (nv as f64);
        let diff = (st as f64) - lam;
        sum_var_excess += diff * diff - lam;
        sum_l2 += lam * lam;
    }
    if sum_l2 <= 1e-12 || sum_var_excess <= 0.05 * sum_l2 {
        return 20.0;
    }
    (sum_l2 / sum_var_excess).clamp(0.15, 20.0)
}

/// Exact enumerative code length (in nats) to transmit m_e mutations along a
/// single branch e over L_eff effective sites.
#[inline]
pub fn branch_mutation_code_nats(
    m_e: f64,
    l_eff: f64,
    subst_nats: f64,
    alphabet_size: usize,
) -> f64 {
    if m_e <= 1e-9 {
        return universal_int_nats(0.0);
    }
    let m_pos = m_e.min(0.49 * l_eff);
    let header = universal_int_nats(m_e);
    let mut site_positions = log_binom(l_eff, m_pos);
    if m_e > 0.49 * l_eff {
        site_positions += (m_e - 0.49 * l_eff) * 2.0_f64.ln();
    }
    let default_state_cost = m_e * ((alphabet_size.saturating_sub(1).max(1)) as f64).ln();
    let state_cost = if subst_nats > 0.0 {
        subst_nats
    } else {
        default_state_cost
    };
    header + site_positions + state_cost
}

/// Exact combinatorial cost (in nats) of resolving a k = (deg_a + deg_b)-way
/// polytomy into two sub-nodes of outward degrees deg_a >= 2 and deg_b >= 2.
#[inline]
pub fn split_topology_cost_nats(deg_a: usize, deg_b: usize) -> f64 {
    let k = deg_a + deg_b;
    if k < 4 || deg_a < 2 || deg_b < 2 {
        return 0.0;
    }
    let num_sizes = ((k / 2).saturating_sub(1)).max(1) as f64;
    let mut subset_cost = log_binom(k as f64, deg_a as f64);
    if deg_a == deg_b {
        subset_cost -= 2.0_f64.ln();
    }
    num_sizes.ln() + subset_cost
}

// ---------------------------------------------------------------------------
// Data Structures for the MDL Phylogenetic Compression Graph
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShuntEdge {
    pub node_u: usize,
    pub node_v: usize,
    pub label_u: String,
    pub label_v: String,
    pub mode: String, // "INTERVAL" or "SPARSE"
    pub sites: Vec<usize>,
    pub interval_span: Option<(usize, usize)>,
    pub shared_mutations: usize,
    pub data_savings_nats: f64,
    pub mask_cost_nats: f64,
    pub endpoint_cost_nats: f64,
    pub net_mdl_gain_nats: f64,
    pub posterior_prob: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BranchRecord {
    pub u: usize,
    pub v: usize,
    pub is_internal: bool,
    pub mutation_count: f64,
    pub branch_length: f64,
    pub mdl_gain_nats: f64,
    pub mdl_support: f64,
    pub synapomorphy_sites: Vec<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BushwhackerResult {
    pub names: Vec<String>,
    pub n_taxa: usize,
    pub n_sites: usize,
    pub l_eff: f64,
    pub newick: String,
    pub binary_newick: String,
    pub adjacency: BTreeMap<usize, Vec<usize>>,
    pub node_labels: BTreeMap<usize, String>,
    pub branches: Vec<BranchRecord>,
    pub shunts: Vec<ShuntEdge>,
    pub n_internal_branches: usize,
    pub max_binary_internal_branches: usize,
    pub polytomy_nodes: usize,
    pub max_polytomy_degree: usize,
    pub raw_uncompressed_nats: f64,
    pub star_tree_nats: f64,
    pub backbone_tree_nats: f64,
    pub full_graph_nats: f64,
    pub runtime_sec: f64,
    pub site_weights: Vec<f64>,
    pub terminal_loads: Vec<f64>,
}

// ---------------------------------------------------------------------------
// 64-Site Bit-Plane SIMD Sequence Representation
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default)]
pub struct BitPlaneWord64 {
    pub b0: u64,    // 1 bit per site for A (state 0)
    pub b1: u64,    // 1 bit per site for C (state 1)
    pub b2: u64,    // 1 bit per site for G (state 2)
    pub b3: u64,    // 1 bit per site for T/U (state 3)
    pub valid: u64, // b0 | b1 | b2 | b3
}

#[derive(Clone, Debug)]
pub struct PackedSeq64 {
    pub words: Vec<BitPlaneWord64>,
    pub len: usize,
}

impl PackedSeq64 {
    #[inline]
    pub fn from_row(row: &[i8]) -> Self {
        let len = row.len();
        let n_words = (len + 63) / 64;
        let mut words = vec![BitPlaneWord64::default(); n_words];
        for (w_idx, chunk) in row.chunks(64).enumerate() {
            let mut b0 = 0_u64;
            let mut b1 = 0_u64;
            let mut b2 = 0_u64;
            let mut b3 = 0_u64;
            for (bit_idx, &v) in chunk.iter().enumerate() {
                let bit = 1_u64 << bit_idx;
                match v {
                    0 => b0 |= bit,
                    1 => b1 |= bit,
                    2 => b2 |= bit,
                    3 => b3 |= bit,
                    _ => {}
                }
            }
            words[w_idx] = BitPlaneWord64 {
                b0,
                b1,
                b2,
                b3,
                valid: b0 | b1 | b2 | b3,
            };
        }
        Self { words, len }
    }

    /// Reconstruct the ancestral packed sequence at internal node `w = parent(u, v)`
    /// using the 64-site parallel bitwise Steiner Median `(u & v) | (u & r) | (v & r)`
    /// against the nearest outgroup witness `r`. This strips terminal mutations on both
    /// `(w, u)` and `(w, v)` while preserving a clean single-bit state at every site.
    #[inline]
    pub fn merge_steiner_median(u: &Self, v: &Self, r: &Self) -> Self {
        let mut words = Vec::with_capacity(u.words.len());
        for ((wu, wv), wr) in u.words.iter().zip(&v.words).zip(&r.words) {
            let mb0 = (wu.b0 & wv.b0) | (wu.b0 & wr.b0) | (wv.b0 & wr.b0);
            let mb1 = (wu.b1 & wv.b1) | (wu.b1 & wr.b1) | (wv.b1 & wr.b1);
            let mb2 = (wu.b2 & wv.b2) | (wu.b2 & wr.b2) | (wv.b2 & wr.b2);
            let mb3 = (wu.b3 & wv.b3) | (wu.b3 & wr.b3) | (wv.b3 & wr.b3);
            let has_med = mb0 | mb1 | mb2 | mb3;
            let fb_u = !has_med & wu.valid;
            let fb_v = !has_med & !wu.valid & wv.valid;
            let b0 = mb0 | (fb_u & wu.b0) | (fb_v & wv.b0);
            let b1 = mb1 | (fb_u & wu.b1) | (fb_v & wv.b1);
            let b2 = mb2 | (fb_u & wu.b2) | (fb_v & wv.b2);
            let b3 = mb3 | (fb_u & wu.b3) | (fb_v & wv.b3);
            words.push(BitPlaneWord64 {
                b0,
                b1,
                b2,
                b3,
                valid: b0 | b1 | b2 | b3,
            });
        }
        Self { words, len: u.len }
    }
}

// ---------------------------------------------------------------------------
// Genetic Code & Alignment Encoding Utilities
// ---------------------------------------------------------------------------

pub const IS_PURINE: [u8; 4] = [1, 0, 1, 0];

/// Translate a 6-bit encoded codon `(b0 << 4) | (b1 << 2) | b2` (where A=0, C=1, G=2, T=3)
/// into its single-letter amino acid (or b'*' for stop).
#[inline]
pub fn translate_codon_idx(idx: u8) -> u8 {
    // 64 codons in order AAA(0), AAC(1), AAG(2), AAT(3), ACA(4), ...
    const TABLE: &[u8; 64] = b"KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*Y*YSSSS*CWCLFLF";
    if (idx as usize) < 64 {
        TABLE[idx as usize]
    } else {
        b'X'
    }
}

#[inline]
pub fn translate_bases(b0: i8, b1: i8, b2: i8) -> u8 {
    if b0 < 0 || b1 < 0 || b2 < 0 {
        return b'X';
    }
    let idx = ((b0 as u8) << 4) | ((b1 as u8) << 2) | (b2 as u8);
    translate_codon_idx(idx)
}

#[inline]
fn is_valid_seq_char(ch: char) -> bool {
    matches!(
        ch.to_ascii_uppercase(),
        'A' | 'C'
            | 'D'
            | 'E'
            | 'F'
            | 'G'
            | 'H'
            | 'I'
            | 'J'
            | 'K'
            | 'L'
            | 'M'
            | 'N'
            | 'O'
            | 'P'
            | 'Q'
            | 'R'
            | 'S'
            | 'T'
            | 'U'
            | 'V'
            | 'W'
            | 'X'
            | 'Y'
            | 'Z'
            | 'B'
            | '-'
            | '?'
            | '.'
            | '*'
    )
}

/// Resolve `.` (match-first-taxon character used in PHYLIP and NEXUS) and pad sequences to uniform length.
fn finalize_alignment(
    names: Vec<String>,
    mut seqs: Vec<String>,
) -> Result<(Vec<String>, Vec<String>), String> {
    if names.is_empty() || seqs.is_empty() {
        return Err("No alignment records found.".to_string());
    }
    let max_len = seqs.iter().map(|s| s.len()).max().unwrap_or(0);
    if max_len == 0 {
        return Err("Alignment sequences have zero length.".to_string());
    }
    for s in &mut seqs {
        if s.len() < max_len {
            s.extend(std::iter::repeat('-').take(max_len - s.len()));
        }
    }
    // Resolve '.' match characters against the first sequence
    let ref_bytes = seqs[0].as_bytes().to_vec();
    for s in seqs.iter_mut().skip(1) {
        if s.contains('.') {
            let mut bytes = s.as_bytes().to_vec();
            for j in 0..bytes.len().min(ref_bytes.len()) {
                if bytes[j] == b'.' {
                    bytes[j] = ref_bytes[j];
                }
            }
            if let Ok(resolved) = String::from_utf8(bytes) {
                *s = resolved;
            }
        }
    }
    Ok((names, seqs))
}

/// Detect alignment format (`"FASTA"`, `"PHYLIP"`, or `"NEXUS"`) from raw text.
pub fn detect_alignment_format(text: &str) -> &'static str {
    let clean = text.trim_start_matches('\u{feff}');
    for line in clean.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with(';') {
            continue;
        }
        let upper = trimmed.to_ascii_uppercase();
        if upper.starts_with("#NEXUS")
            || upper.starts_with("BEGIN DATA")
            || upper.starts_with("BEGIN CHARACTERS")
            || upper.starts_with("BEGIN TAXA")
        {
            return "NEXUS";
        }
        if trimmed.starts_with('>') {
            return "FASTA";
        }
        let mut toks = trimmed.split_whitespace();
        if let (Some(t0), Some(t1)) = (toks.next(), toks.next()) {
            if t0.parse::<usize>().is_ok() && t1.parse::<usize>().is_ok() {
                return "PHYLIP";
            }
        }
        break;
    }
    let upper_all = clean.to_ascii_uppercase();
    if upper_all.contains("BEGIN DATA") || upper_all.contains("BEGIN CHARACTERS") {
        return "NEXUS";
    }
    "FASTA"
}

pub fn parse_fasta_only_str(fasta_text: &str) -> Result<(Vec<String>, Vec<String>), String> {
    let clean = fasta_text.trim_start_matches('\u{feff}');
    let mut names = Vec::new();
    let mut seqs = Vec::new();
    let mut curr_name: Option<String> = None;
    let mut curr_seq = String::new();

    for line in clean.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with(';') {
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix('>') {
            if let Some(name) = curr_name.take() {
                names.push(name);
                seqs.push(std::mem::take(&mut curr_seq));
            }
            let first_tok = rest.split_whitespace().next().unwrap_or("").to_string();
            curr_name = Some(if first_tok.is_empty() {
                format!("Seq_{}", names.len() + 1)
            } else {
                first_tok
            });
        } else {
            for ch in trimmed.chars() {
                if !ch.is_whitespace() {
                    curr_seq.push(ch);
                }
            }
        }
    }
    if let Some(name) = curr_name {
        names.push(name);
        seqs.push(curr_seq);
    }
    if names.is_empty() {
        return Err("No FASTA records found.".to_string());
    }
    finalize_alignment(names, seqs)
}

/// Split a PHYLIP line that begins a taxon record into `(taxon_name, sequence_chars)`.
/// Supports both Relaxed PHYLIP (whitespace-delimited) and Strict 10-character PHYLIP
/// (even when the 10-character taxon name immediately abuts nucleotide position 1).
fn split_phylip_named_line(raw_line: &str, expected_nchar: usize) -> Option<(String, String)> {
    let line = raw_line.trim_end_matches(&['\r', '\n'][..]);
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }

    // Check if there is whitespace separating a leading token from the rest
    if let Some((first_tok, rest)) = trimmed.split_once(char::is_whitespace) {
        let rest_seq: String = rest.chars().filter(|c| !c.is_whitespace()).collect();
        // Could this be Strict 10-char PHYLIP where the 10-char name had no space at column 10,
        // so `first_tok` absorbed the first sequence block?
        if line.len() > 10 && first_tok.len() > 10 {
            let strict_name = line[..10].trim();
            let strict_seq: String = line[10..].chars().filter(|c| !c.is_whitespace()).collect();
            let suffix_after_10 = &first_tok[10.min(first_tok.len())..];
            if !strict_name.is_empty()
                && !strict_seq.is_empty()
                && suffix_after_10.chars().all(is_valid_seq_char)
                && strict_seq.chars().all(is_valid_seq_char)
                && (strict_seq.len() == expected_nchar || rest_seq.is_empty() || suffix_after_10.len() == 10)
            {
                return Some((strict_name.replace(' ', "_"), strict_seq));
            }
        }
        if !rest_seq.is_empty() {
            return Some((first_tok.to_string(), rest_seq));
        }
    }

    // No whitespace separator found: try Strict 10-character fixed-width PHYLIP column split
    if line.len() > 10 {
        let strict_name = line[..10].trim().replace(' ', "_");
        let strict_seq: String = line[10..].chars().filter(|c| !c.is_whitespace()).collect();
        if !strict_name.is_empty()
            && !strict_seq.is_empty()
            && strict_seq.chars().all(is_valid_seq_char)
        {
            return Some((strict_name, strict_seq));
        }
    }
    None
}

/// Parse PHYLIP alignment string (supports Strict 10-char & Relaxed, Sequential & Interleaved).
pub fn parse_phylip_str(phylip_text: &str) -> Result<(Vec<String>, Vec<String>), String> {
    let clean = phylip_text.trim_start_matches('\u{feff}');
    let mut lines_iter = clean
        .lines()
        .map(|l| l.trim_end_matches('\r'))
        .filter(|l| !l.trim().is_empty());

    let header = lines_iter
        .next()
        .ok_or_else(|| "Empty PHYLIP input.".to_string())?;
    let header_toks: Vec<&str> = header.split_whitespace().collect();
    if header_toks.len() < 2 {
        return Err("Invalid PHYLIP header: expected '<ntax> <nchar>'.".to_string());
    }
    let ntax: usize = header_toks[0]
        .parse()
        .map_err(|_| format!("Invalid PHYLIP ntax '{}'", header_toks[0]))?;
    let nchar: usize = header_toks[1]
        .parse()
        .map_err(|_| format!("Invalid PHYLIP nchar '{}'", header_toks[1]))?;
    if ntax == 0 || nchar == 0 {
        return Err("PHYLIP ntax and nchar must be positive.".to_string());
    }
    let explicit_interleaved = header_toks
        .get(2)
        .map(|f| f.eq_ignore_ascii_case("I"))
        .unwrap_or(false);
    let explicit_sequential = header_toks
        .get(2)
        .map(|f| f.eq_ignore_ascii_case("S"))
        .unwrap_or(false);

    let data_lines: Vec<&str> = lines_iter.collect();
    if data_lines.len() < ntax {
        return Err(format!(
            "PHYLIP header specifies {ntax} taxa, but only {} non-empty lines follow.",
            data_lines.len()
        ));
    }

    // Fast path: exactly 1 line per taxon
    if data_lines.len() == ntax {
        let mut names = Vec::with_capacity(ntax);
        let mut seqs = Vec::with_capacity(ntax);
        for (i, &line) in data_lines.iter().enumerate() {
            let (nm, sq) = split_phylip_named_line(line, nchar).ok_or_else(|| {
                format!("Failed to parse PHYLIP taxon row {} ('{}')", i + 1, line.trim())
            })?;
            names.push(nm);
            seqs.push(sq);
        }
        return finalize_alignment(names, seqs);
    }

    // Helper closure to try Sequential PHYLIP parsing
    let try_sequential = || -> Result<(Vec<String>, Vec<String>), String> {
        let mut names = Vec::with_capacity(ntax);
        let mut seqs = Vec::with_capacity(ntax);
        let mut line_idx = 0;
        for t in 0..ntax {
            if line_idx >= data_lines.len() {
                return Err(format!("Unexpected EOF at taxon {} in sequential PHYLIP", t + 1));
            }
            let (nm, mut sq) = split_phylip_named_line(data_lines[line_idx], nchar)
                .ok_or_else(|| format!("Invalid taxon start line in sequential PHYLIP: '{}'", data_lines[line_idx]))?;
            line_idx += 1;
            while sq.len() < nchar && line_idx < data_lines.len() {
                let cont: String = data_lines[line_idx]
                    .chars()
                    .filter(|c| !c.is_whitespace())
                    .collect();
                if !cont.chars().all(is_valid_seq_char) {
                    return Err("Continuation line contains non-sequence characters".to_string());
                }
                sq.push_str(&cont);
                line_idx += 1;
            }
            if sq.len() != nchar {
                return Err(format!(
                    "Sequential PHYLIP taxon '{nm}' length {} != expected {nchar}",
                    sq.len()
                ));
            }
            names.push(nm);
            seqs.push(sq);
        }
        if line_idx != data_lines.len() {
            return Err("Trailing unconsumed lines in sequential PHYLIP".to_string());
        }
        finalize_alignment(names, seqs)
    };

    // Helper closure to try Interleaved PHYLIP parsing
    let try_interleaved = || -> Result<(Vec<String>, Vec<String>), String> {
        let mut names = Vec::with_capacity(ntax);
        let mut seqs = Vec::with_capacity(ntax);
        for i in 0..ntax {
            let (nm, sq) = split_phylip_named_line(data_lines[i], nchar)
                .ok_or_else(|| format!("Invalid first-block line {} in interleaved PHYLIP", i + 1))?;
            names.push(nm);
            seqs.push(sq);
        }
        let first_len = seqs[0].len();
        if first_len == 0 || seqs.iter().any(|s| s.len() != first_len) {
            return Err("First-block sequence lengths are not uniform in interleaved PHYLIP".to_string());
        }
        for (idx, &line) in data_lines.iter().enumerate().skip(ntax) {
            let t = idx % ntax;
            let trimmed = line.trim();
            // Support both standard interleaved lines (sequence only) and lines that repeat the taxon name
            let chunk: String = if let Some(rest) = trimmed.strip_prefix(names[t].as_str()) {
                if rest.starts_with(char::is_whitespace) {
                    rest.chars().filter(|c| !c.is_whitespace()).collect()
                } else {
                    trimmed.chars().filter(|c| !c.is_whitespace()).collect()
                }
            } else {
                trimmed.chars().filter(|c| !c.is_whitespace()).collect()
            };
            if chunk.is_empty() || !chunk.chars().all(is_valid_seq_char) {
                return Err(format!("Invalid interleaved continuation line {}", idx + 1));
            }
            seqs[t].push_str(&chunk);
        }
        if seqs.iter().any(|s| s.len() != nchar) {
            return Err("Interleaved PHYLIP sequence lengths do not match nchar".to_string());
        }
        finalize_alignment(names, seqs)
    };

    if explicit_sequential {
        return try_sequential().or_else(|_| try_interleaved());
    }
    if explicit_interleaved {
        return try_interleaved().or_else(|_| try_sequential());
    }

    // Auto-disambiguate Interleaved vs Multi-line Sequential:
    // Check if line 1 (second data line) starts with a non-sequence taxon name or if interleaved succeeds cleanly
    if let Ok(res) = try_interleaved() {
        return Ok(res);
    }
    try_sequential()
}

/// Strip NEXUS `[...]` comments (supporting nested brackets) while preserving quotes and line breaks.
fn strip_nexus_comments(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut depth = 0usize;
    let mut in_single_quote = false;
    let mut in_double_quote = false;

    for ch in input.chars() {
        if depth > 0 {
            if ch == '[' {
                depth += 1;
            } else if ch == ']' {
                depth -= 1;
            } else if ch == '\n' {
                out.push('\n');
            }
            continue;
        }
        if ch == '\'' && !in_double_quote {
            in_single_quote = !in_single_quote;
            out.push(ch);
        } else if ch == '"' && !in_single_quote {
            in_double_quote = !in_double_quote;
            out.push(ch);
        } else if ch == '[' && !in_single_quote && !in_double_quote {
            depth = 1;
        } else {
            out.push(ch);
        }
    }
    out
}

/// Parse a NEXUS alignment line into `(taxon_name, sequence_chunk)`.
fn parse_nexus_matrix_line(trimmed: &str) -> Option<(String, String)> {
    if trimmed.is_empty() {
        return None;
    }
    if let Some(rest) = trimmed.strip_prefix('\'') {
        if let Some(end_q) = rest.find('\'') {
            let name = rest[..end_q].trim().replace(' ', "_");
            let seq: String = rest[end_q + 1..]
                .chars()
                .filter(|c| !c.is_whitespace())
                .collect();
            return Some((name, seq));
        }
    }
    if let Some(rest) = trimmed.strip_prefix('"') {
        if let Some(end_q) = rest.find('"') {
            let name = rest[..end_q].trim().replace(' ', "_");
            let seq: String = rest[end_q + 1..]
                .chars()
                .filter(|c| !c.is_whitespace())
                .collect();
            return Some((name, seq));
        }
    }
    let (first_tok, rest) = trimmed.split_once(char::is_whitespace)?;
    let seq: String = rest.chars().filter(|c| !c.is_whitespace()).collect();
    if seq.is_empty() {
        return None;
    }
    Some((first_tok.to_string(), seq))
}

/// Extract the value of a NEXUS `KEY=VALUE` token (case-insensitive) from header text.
fn extract_nexus_int_param(header_upper: &str, key: &str) -> Option<usize> {
    let idx = header_upper.find(key)?;
    let after = header_upper[idx + key.len()..].trim_start();
    let after_eq = after.strip_prefix('=')?.trim_start();
    let digits: String = after_eq
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse().ok()
}

/// Parse NEXUS alignment string (supports `DATA` and `CHARACTERS` blocks, sequential & interleaved,
/// `[...]` comments, quoted taxon names, and `.` matchchar).
pub fn parse_nexus_str(nexus_text: &str) -> Result<(Vec<String>, Vec<String>), String> {
    let stripped = strip_nexus_comments(nexus_text.trim_start_matches('\u{feff}'));
    let upper = stripped.to_ascii_uppercase();

    // Find MATRIX keyword bounded by whitespace/newline/semicolon
    let mut matrix_body_start = None;
    let mut header_end = 0;
    let mut search_from = 0;
    while let Some(pos) = upper[search_from..].find("MATRIX") {
        let abs_pos = search_from + pos;
        let before_ok = abs_pos == 0
            || stripped.as_bytes()[abs_pos - 1].is_ascii_whitespace()
            || stripped.as_bytes()[abs_pos - 1] == b';';
        let after_pos = abs_pos + 6;
        let after_ok = after_pos >= stripped.len()
            || stripped.as_bytes()[after_pos].is_ascii_whitespace();
        if before_ok && after_ok {
            header_end = abs_pos;
            matrix_body_start = Some(after_pos);
            break;
        }
        search_from = after_pos;
    }

    let start_idx = matrix_body_start
        .ok_or_else(|| "No MATRIX block found in NEXUS input.".to_string())?;
    let header_upper = &upper[..header_end];
    let expected_ntax = extract_nexus_int_param(header_upper, "NTAX");
    let expected_nchar = extract_nexus_int_param(header_upper, "NCHAR");

    // Extract matrix body up to the first unquoted ';'
    let after_matrix = &stripped[start_idx..];
    let mut end_rel = after_matrix.len();
    let mut in_sq = false;
    let mut in_dq = false;
    for (idx, ch) in after_matrix.char_indices() {
        if ch == '\'' && !in_dq {
            in_sq = !in_sq;
        } else if ch == '"' && !in_sq {
            in_dq = !in_dq;
        } else if ch == ';' && !in_sq && !in_dq {
            end_rel = idx;
            break;
        }
    }
    let matrix_body = &after_matrix[..end_rel];

    let mut names: Vec<String> = Vec::new();
    let mut seqs: Vec<String> = Vec::new();
    let mut name_to_idx: HashMap<String, usize> = HashMap::new();

    for raw_line in matrix_body.lines() {
        let trimmed = raw_line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Check if this is an unnamed sequence continuation line for the current taxon
        // in a multi-line sequential NEXUS matrix
        if let Some(last_idx) = seqs.len().checked_sub(1) {
            let need_more = expected_nchar
                .map(|nc| seqs[last_idx].len() < nc)
                .unwrap_or(false);
            let first_tok = trimmed.split_whitespace().next().unwrap_or("");
            let is_known_taxon = name_to_idx.contains_key(first_tok);
            let all_taxa_seen = expected_ntax
                .map(|nt| names.len() >= nt)
                .unwrap_or(false);
            let line_only_seq_chars = trimmed
                .chars()
                .all(|c| c.is_whitespace() || is_valid_seq_char(c));

            if need_more && !is_known_taxon && line_only_seq_chars && (all_taxa_seen || !trimmed.contains(char::is_whitespace)) {
                let cont: String = trimmed.chars().filter(|c| !c.is_whitespace()).collect();
                seqs[last_idx].push_str(&cont);
                continue;
            }
        }

        if let Some((name, seq_chunk)) = parse_nexus_matrix_line(trimmed) {
            if let Some(&idx) = name_to_idx.get(&name) {
                seqs[idx].push_str(&seq_chunk);
            } else {
                let idx = names.len();
                name_to_idx.insert(name.clone(), idx);
                names.push(name);
                seqs.push(seq_chunk);
            }
        } else if let Some(last_idx) = seqs.len().checked_sub(1) {
            // Single-token continuation line
            let cont: String = trimmed.chars().filter(|c| !c.is_whitespace()).collect();
            if !cont.is_empty() && cont.chars().all(is_valid_seq_char) {
                seqs[last_idx].push_str(&cont);
            }
        }
    }

    if names.is_empty() {
        return Err("No sequence records found in NEXUS MATRIX block.".to_string());
    }
    finalize_alignment(names, seqs)
}

/// Auto-detect and parse an alignment string in **FASTA**, **PHYLIP** (strict/relaxed,
/// sequential/interleaved), or **NEXUS** format, returning `(names, seqs, format_name)`.
pub fn parse_alignment_str_with_format(
    text: &str,
) -> Result<(Vec<String>, Vec<String>, &'static str), String> {
    let fmt = detect_alignment_format(text);
    match fmt {
        "NEXUS" => {
            let (n, s) = parse_nexus_str(text)?;
            Ok((n, s, "NEXUS"))
        }
        "PHYLIP" => {
            let (n, s) = parse_phylip_str(text)?;
            Ok((n, s, "PHYLIP"))
        }
        _ => {
            let (n, s) = parse_fasta_only_str(text)?;
            Ok((n, s, "FASTA"))
        }
    }
}

/// Auto-detect and parse an alignment string in **FASTA**, **PHYLIP**, or **NEXUS** format.
pub fn parse_alignment_str(text: &str) -> Result<(Vec<String>, Vec<String>), String> {
    let (names, seqs, _) = parse_alignment_str_with_format(text)?;
    Ok((names, seqs))
}

/// Backwards-compatible alias that auto-detects **FASTA**, **PHYLIP**, or **NEXUS** input.
pub fn parse_fasta_str(text: &str) -> Result<(Vec<String>, Vec<String>), String> {
    parse_alignment_str(text)
}

/// Auto-detect and parse an alignment file in **FASTA**, **PHYLIP**, or **NEXUS** format.
pub fn parse_alignment_file<P: AsRef<Path>>(path: P) -> Result<(Vec<String>, Vec<String>), String> {
    let file = File::open(path.as_ref())
        .map_err(|e| format!("Failed to open alignment file: {e}"))?;
    let mut reader = BufReader::new(file);
    let mut buf = String::new();
    use std::io::Read;
    reader
        .read_to_string(&mut buf)
        .map_err(|e| format!("Read error: {e}"))?;
    parse_alignment_str(&buf)
}

/// Backwards-compatible alias that auto-detects **FASTA**, **PHYLIP**, or **NEXUS** files.
pub fn parse_fasta_file<P: AsRef<Path>>(path: P) -> Result<(Vec<String>, Vec<String>), String> {
    parse_alignment_file(path)
}

/// Encode N aligned nucleotide sequences of length L into flat row-major arrays:
/// - `mat`: `(N * L)` `i8` in `{0, 1, 2, 3, -1}`
/// - `bitmasks`: `(N * L)` `u8` in `{1, 2, 4, 8, 15}`
/// - `majority_state`: `(L)` `i8`
pub fn encode_nucleotide_alignment(seqs: &[String]) -> (Vec<i8>, Vec<u8>, Vec<i8>, usize, usize) {
    let n = seqs.len();
    let l = seqs.first().map(|s| s.len()).unwrap_or(0);
    let mut mat = vec![-1_i8; n * l];
    let mut bitmasks = vec![15_u8; n * l];
    let mut counts = vec![[0_u32; 4]; l];

    for (i, s) in seqs.iter().enumerate() {
        let row_off = i * l;
        for (j, &byte) in s.as_bytes().iter().take(l).enumerate() {
            let v: i8 = match byte {
                b'A' | b'a' => 0,
                b'C' | b'c' => 1,
                b'G' | b'g' => 2,
                b'T' | b't' | b'U' | b'u' => 3,
                _ => -1,
            };
            if v >= 0 {
                mat[row_off + j] = v;
                bitmasks[row_off + j] = 1_u8 << (v as u8);
                counts[j][v as usize] += 1;
            }
        }
    }

    let mut majority_state = vec![0_i8; l];
    for j in 0..l {
        let mut best_b = 0_i8;
        let mut best_c = counts[j][0];
        for b in 1..4 {
            if counts[j][b] > best_c {
                best_c = counts[j][b];
                best_b = b as i8;
            }
        }
        majority_state[j] = best_b;
    }

    (mat, bitmasks, majority_state, n, l)
}

pub fn detect_coding_frame_and_weights(
    mat: &[i8],
    n: usize,
    l: usize,
) -> (Vec<f64>, Vec<f64>, bool) {
    let mut dist_weights = vec![1.0_f64; l];
    let mut tree_weights = vec![1.0_f64; l];
    let mut is_coding = false;

    if l >= 90 && l % 3 == 0 {
        let n_codons = l / 3;
        let mut stop_count = 0usize;
        let mut valid_codons = 0usize;
        let check_n = n.min(32);
        for i in 0..check_n {
            let row = &mat[i * l..(i + 1) * l];
            for c in 0..(n_codons.saturating_sub(1)) {
                let b0 = row[3 * c];
                let b1 = row[3 * c + 1];
                let b2 = row[3 * c + 2];
                if b0 >= 0 && b1 >= 0 && b2 >= 0 {
                    valid_codons += 1;
                    if translate_bases(b0, b1, b2) == b'*' {
                        stop_count += 1;
                    }
                }
            }
        }
        if valid_codons > 0 && (stop_count as f64 / valid_codons as f64) < 0.02 {
            is_coding = true;
            for c in 0..n_codons {
                let c0 = 3 * c;
                let c1 = 3 * c + 1;
                let c2 = 3 * c + 2;
                let mut cod_mask = 0_u64;
                let mut aa_mask = 0_u32;
                for i in 0..n {
                    let row = &mat[i * l..(i + 1) * l];
                    let b0 = row[c0];
                    let b1 = row[c1];
                    let b2 = row[c2];
                    if b0 >= 0 && b1 >= 0 && b2 >= 0 {
                        let idx = ((b0 as u8) << 4) | ((b1 as u8) << 2) | (b2 as u8);
                        cod_mask |= 1_u64 << idx;
                        let aa = translate_codon_idx(idx);
                        let aa_bit = if aa == b'*' {
                            26
                        } else {
                            (aa.wrapping_sub(b'A')).min(25)
                        };
                        aa_mask |= 1_u32 << aa_bit;
                    }
                }
                if cod_mask.count_ones() > 1 {
                    if aa_mask.count_ones() == 1 {
                        for col in [c0, c1, c2] {
                            dist_weights[col] = 2.0;
                            tree_weights[col] = 1.25;
                        }
                    } else {
                        dist_weights[c0] = 0.12;
                        dist_weights[c1] = 0.08;
                        dist_weights[c2] = 1.00;
                        tree_weights[c0] = 0.45;
                        tree_weights[c1] = 0.40;
                        tree_weights[c2] = 1.15;
                    }
                }
            }
        }
    }
    (dist_weights, tree_weights, is_coding)
}

#[inline]
fn median_f64(vals: &mut [f64]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = vals.len() / 2;
    if vals.len() % 2 == 1 {
        vals[mid]
    } else {
        0.5 * (vals[mid - 1] + vals[mid])
    }
}

// ---------------------------------------------------------------------------
// Core Engine: BushwhackerEngine
// ---------------------------------------------------------------------------

pub struct BushwhackerEngine {
    pub coding_aware: bool,
    pub detect_shunts: bool,
    pub min_shunt_sites: usize,
    pub force_protein: bool,
    pub translate_coding: bool,
    pub discrete_mode: Option<discrete::DiscreteMode>,
    pub no_mult_pen: bool,
    pub empirical_alpha: bool,
}

impl Default for BushwhackerEngine {
    fn default() -> Self {
        Self {
            coding_aware: true,
            detect_shunts: true,
            min_shunt_sites: 2,
            force_protein: false,
            translate_coding: false,
            discrete_mode: None,
            no_mult_pen: false,
            empirical_alpha: false,
        }
    }
}

impl BushwhackerEngine {
    pub fn new(coding_aware: bool, detect_shunts: bool, min_shunt_sites: usize) -> Self {
        Self {
            coding_aware,
            detect_shunts,
            min_shunt_sites,
            force_protein: false,
            translate_coding: false,
            discrete_mode: None,
            no_mult_pen: false,
            empirical_alpha: false,
        }
    }

    pub fn infer_from_fasta<P: AsRef<Path>>(&self, path: P) -> Result<BushwhackerResult, String> {
        if let Some(d_mode) = self.discrete_mode {
            let cm = discrete::parse_discrete_file(path)?;
            return discrete::infer_discrete_mdl(self, &cm, d_mode, None);
        }
        let (names, seqs) = parse_fasta_file(path)?;
        self.infer(&names, &seqs)
    }

    pub fn infer(&self, names: &[String], seqs: &[String]) -> Result<BushwhackerResult, String> {
        if names.is_empty() {
            return Err("Empty alignment provided.".to_string());
        }
        if let Some(d_mode) = self.discrete_mode {
            let cm = discrete::encode_discrete_sequences(names, seqs)?;
            return discrete::infer_discrete_mdl(self, &cm, d_mode, None);
        }
        if self.translate_coding && !protein::is_protein_alignment(seqs) {
            let aa_seqs = protein::translate_coding_alignment(seqs)?;
            return protein::infer_protein_mdl(self, names, &aa_seqs, None);
        }
        if self.force_protein || protein::is_protein_alignment(seqs) {
            return protein::infer_protein_mdl(self, names, seqs, None);
        }
        #[cfg(not(target_arch = "wasm32"))]
        let t0 = Instant::now();
        let n = names.len();
        let (mat, bitmasks, majority_state, _, l) = encode_nucleotide_alignment(seqs);

        let total_valid: usize = mat.iter().filter(|&&x| x >= 0).count();
        let l_eff = ((total_valid as f64) / (n as f64)).clamp(1.0, l as f64);

        let (ti_cost, tv_cost) = self.estimate_subst_surprise(&mat, n, l);
        let (_, _, is_coding) = if self.coding_aware {
            detect_coding_frame_and_weights(&mat, n, l)
        } else {
            (vec![1.0; l], vec![1.0; l], false)
        };
        let tree_weights = vec![1.0; l];

        // Step 1 & 2: Exact Jin-Nei BioNJ + 64-site SoA GTR+CAT8 Eigen-Felsenstein-Rissanen MDL backbone
        let (terminal_loads, mut adj, node_labels, gtr_edge_stats) =
            self.build_nt_gtr_loads_and_backbone(names, &mat, &bitmasks, n, l, l_eff);

        let pre_step3_adj = adj.clone();
        let need_fitch_muts = self.detect_shunts && n >= 4;

        // Step 3: MDL Edge Contraction + Polytomy Hub Clique Expansion
        let (adj_c, mut branches, _, edge_mutations) = self.contract_unsupported_edges_mdl(
            adj,
            &bitmasks,
            &majority_state,
            &terminal_loads,
            n,
            l,
            l_eff,
            ti_cost,
            tv_cost,
            &tree_weights,
            &mut node_labels.clone(),
            true,
            gtr_edge_stats.as_ref(),
            need_fitch_muts,
        );
        adj = adj_c;

        // Build the strict binary refinement of the MDL-certified tree `adj` (preserving 100% of
        // certified positive-MDL splits; when `gtr_edge_stats` is present and `pre_step3_adj` is
        // already binary, `pre_step3_adj` is an exact topological refinement of `adj`).
        let bin_adj = if gtr_edge_stats.is_some()
            && pre_step3_adj
                .iter()
                .all(|(&u, nbs)| u < n || nbs.is_empty() || nbs.len() == 3)
        {
            pre_step3_adj
        } else {
            self.build_binary_refinement_from_mdl(
                &adj,
                &pre_step3_adj,
                &bitmasks,
                &terminal_loads,
                n,
                l,
                l_eff,
                ti_cost,
                tv_cost,
                &tree_weights,
            )
        };
        let binary_newick = self.serialize_binary_newick(
            &bin_adj,
            names,
            &bitmasks,
            &majority_state,
            &terminal_loads,
            n,
            l,
            l_eff,
            gtr_edge_stats.as_ref().map(|x| &x.0),
        );
        // Refresh node labels for any hub-expanded nodes
        let mut full_labels = node_labels;
        for &u in adj.keys() {
            full_labels.entry(u).or_insert_with(|| format!("N{u}"));
        }

        // Step 4: Dual-Mode Residual Shunt Peeling + Coordinate-Descent Re-Verification
        let mut shunts = Vec::new();
        if self.detect_shunts && n >= 4 {
            shunts = self.peel_residual_shunts_mdl(
                &adj,
                &full_labels,
                &bitmasks,
                &majority_state,
                &terminal_loads,
                &edge_mutations,
                n,
                l,
                l_eff,
                ti_cost,
                tv_cost,
                &mat,
                is_coding,
            );
            let int_shunts: Vec<&ShuntEdge> = shunts
                .iter()
                .filter(|s| {
                    s.mode == "INTERVAL" && s.interval_span.is_some() && s.net_mdl_gain_nats >= 15.0
                })
                .collect();
            if !int_shunts.is_empty() {
                let mut masked_bitmasks = bitmasks.clone();
                let mut masked_any = false;
                for sh in int_shunts {
                    if let Some((a_sp, b_sp)) = sh.interval_span {
                        let rec_node = if sh.node_u < n {
                            Some(sh.node_u)
                        } else if sh.node_v < n {
                            Some(sh.node_v)
                        } else {
                            None
                        };
                        if let Some(r_idx) = rec_node {
                            let row = &mut masked_bitmasks[r_idx * l..(r_idx + 1) * l];
                            for s_i in a_sp..=b_sp.min(l - 1) {
                                row[s_i] = 15;
                            }
                            masked_any = true;
                        }
                    }
                }
                if masked_any {
                    let (adj_re, branches_re, _, _) = self.contract_unsupported_edges_mdl(
                        adj,
                        &masked_bitmasks,
                        &majority_state,
                        &terminal_loads,
                        n,
                        l,
                        l_eff,
                        ti_cost,
                        tv_cost,
                        &tree_weights,
                        &mut full_labels,
                        false,
                        None,
                        true,
                    );
                    adj = adj_re;
                    branches = branches_re;
                }
            }
        }

        // Step 5: MDL bit/nat budgets & Newick serialization
        let raw_nats = (n as f64) * l_eff * 4.0_f64.ln();
        let star_nats = self.compute_star_tree_mdl_nats(&bitmasks, n, l, l_eff);
        let backbone_nats = self.compute_tree_total_mdl_nats(&adj, &branches, l_eff);
        let shunt_gain_total: f64 = shunts.iter().map(|s| s.net_mdl_gain_nats).sum();
        let full_graph_nats = (l_eff * 4.0_f64.ln()).max(backbone_nats - shunt_gain_total);

        let mut polytomy_degrees = Vec::new();
        for (&u, nbs) in &adj {
            if u >= n && nbs.len() > 3 {
                polytomy_degrees.push(nbs.len());
            }
        }
        let n_internal_branches = branches.iter().filter(|b| b.is_internal).count();
        let max_binary_internal_branches = n.saturating_sub(3);
        let max_polytomy_degree = polytomy_degrees
            .iter()
            .copied()
            .max()
            .unwrap_or(if n >= 3 { 3 } else { n });

        let newick = self.serialize_newick(&adj, names, &branches, n);
        #[cfg(not(target_arch = "wasm32"))]
        let runtime_sec = t0.elapsed().as_secs_f64();
        #[cfg(target_arch = "wasm32")]
        let runtime_sec = 0.0;

        Ok(BushwhackerResult {
            names: names.to_vec(),
            n_taxa: n,
            n_sites: l,
            l_eff,
            newick,
            binary_newick,
            adjacency: adj,
            node_labels: full_labels,
            branches,
            shunts,
            n_internal_branches,
            max_binary_internal_branches,
            polytomy_nodes: polytomy_degrees.len(),
            max_polytomy_degree,
            raw_uncompressed_nats: raw_nats,
            star_tree_nats: star_nats,
            backbone_tree_nats: backbone_nats,
            full_graph_nats,
            runtime_sec,
            site_weights: tree_weights,
            terminal_loads,
        })
    }

    fn estimate_subst_surprise(&self, mat: &[i8], n: usize, l: usize) -> (f64, f64) {
        let k_sub = n.min(64);
        let indices: Vec<usize> = if n <= 64 {
            (0..n).collect()
        } else {
            (0..k_sub)
                .map(|i| (i * (n - 1)) / (k_sub - 1))
                .collect()
        };
        let mut ti = 1.0_f64;
        let mut tv = 2.0_f64;
        for i in 0..indices.len() {
            let r_i = &mat[indices[i] * l..(indices[i] + 1) * l];
            for j in (i + 1)..indices.len() {
                let r_j = &mat[indices[j] * l..(indices[j] + 1) * l];
                for s in 0..l {
                    let bi = r_i[s];
                    let bj = r_j[s];
                    if bi >= 0 && bj >= 0 && bi != bj {
                        if IS_PURINE[bi as usize] == IS_PURINE[bj as usize] {
                            ti += 1.0;
                        } else {
                            tv += 1.0;
                        }
                    }
                }
            }
        }
        let tot = ti + tv;
        let p_ti = (ti / tot).clamp(0.20, 0.85);
        let p_tv_each = ((1.0 - p_ti) / 2.0).max(0.05);
        (-p_ti.ln(), -p_tv_each.ln())
    }

    fn compute_dual_site_coherence_weights(
        &self,
        mat: &[i8],
        n: usize,
        l: usize,
        dist_weights: &[f64],
        tree_weights: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        if n < 6 || l < 10 {
            return (dist_weights.to_vec(), tree_weights.to_vec());
        }

        let mut n_states = vec![0_u8; l];
        let mut w_mult = vec![1.0_f64; l];
        for s in 0..l {
            let mut mask = 0_u8;
            for i in 0..n {
                let v = mat[i * l + s];
                if v >= 0 {
                    mask |= 1 << (v as u8);
                }
            }
            let st = mask.count_ones() as u8;
            n_states[s] = st;
            if st == 3 {
                w_mult[s] = 0.90;
            } else if st >= 4 {
                w_mult[s] = 0.75;
            }
        }

        // Extract parsimony-informative column-state vectors packed into 128-bit [u64; 2] words
        let k_coh = n.min(128);
        let coh_taxa: Vec<usize> = if n <= 128 {
            (0..n).collect()
        } else {
            (0..k_coh).map(|i| (i * (n - 1)) / (k_coh - 1)).collect()
        };

        let mut char_vecs: Vec<[u64; 2]> = Vec::new();
        let mut char_bar: Vec<[u64; 2]> = Vec::new();
        let mut char_sites: Vec<usize> = Vec::new();
        let mut char_counts: Vec<usize> = Vec::new();

        for s in 0..l {
            if n_states[s] < 2 {
                continue;
            }
            let mut n_val = 0usize;
            for &ti in &coh_taxa {
                if mat[ti * l + s] >= 0 {
                    n_val += 1;
                }
            }
            if n_val < 4 {
                continue;
            }
            for b in 0..4_i8 {
                let mut cnt = 0usize;
                for &ti in &coh_taxa {
                    if mat[ti * l + s] == b {
                        cnt += 1;
                    }
                }
                if cnt >= 2 && cnt <= n_val / 2 {
                    let mut c_bits = [0_u64; 2];
                    let mut cb_bits = [0_u64; 2];
                    for (idx_t, &ti) in coh_taxa.iter().enumerate() {
                        let v = mat[ti * l + s];
                        let word = idx_t >> 6;
                        let bit = 1_u64 << (idx_t & 63);
                        if v == b {
                            c_bits[word] |= bit;
                        } else if v >= 0 {
                            cb_bits[word] |= bit;
                        }
                    }
                    char_vecs.push(c_bits);
                    char_bar.push(cb_bits);
                    char_sites.push(s);
                    char_counts.push(cnt);
                }
            }
        }

        let m_chars = char_vecs.len();
        if m_chars < 8 {
            let md = dist_weights
                .iter()
                .zip(&w_mult)
                .map(|(d, w)| d * w)
                .collect();
            let mt = tree_weights
                .iter()
                .zip(&w_mult)
                .map(|(t, w)| t * w)
                .collect();
            return (md, mt);
        }

        let mut ref_idx: Vec<usize> = (0..m_chars).collect();
        if m_chars > 350 {
            ref_idx.sort_by_key(|&idx| {
                let c = char_counts[idx];
                let bal = c.min(k_coh.saturating_sub(c));
                (std::cmp::Reverse(bal), idx)
            });
            ref_idx.truncate(350);
        }

        let delta_local = (l / 6).clamp(40, 180);
        let eval_char = |k: usize| -> f64 {
            let ck = char_vecs[k];
            let cbk = char_bar[k];
            let sk = char_sites[k];
            let mut sum_overlap = 0.0_f64;
            let mut cnt_overlap = 0usize;
            let mut sum_long = 0.0_f64;
            let mut cnt_long = 0usize;

            for &r in &ref_idx {
                let sr = char_sites[r];
                if sk.abs_diff(sr) <= delta_local {
                    continue;
                }
                let cr = char_vecs[r];
                let cbr = char_bar[r];
                let q11 = (ck[0] & cr[0]).count_ones() + (ck[1] & cr[1]).count_ones();
                let q10 = (ck[0] & cbr[0]).count_ones() + (ck[1] & cbr[1]).count_ones();
                let q01 = (cbk[0] & cr[0]).count_ones() + (cbk[1] & cr[1]).count_ones();
                let q00 = (cbk[0] & cbr[0]).count_ones() + (cbk[1] & cbr[1]).count_ones();

                let viol = q11.min(q10).min(q01.min(q00)) as f64;
                let compat = (-1.1 * viol).exp();
                sum_long += compat;
                cnt_long += 1;
                if q11 >= 1 {
                    sum_overlap += compat;
                    cnt_overlap += 1;
                }
            }
            if cnt_overlap >= 3 {
                sum_overlap / (cnt_overlap as f64)
            } else if cnt_long >= 1 {
                sum_long / (cnt_long as f64)
            } else {
                1.0
            }
        };

        #[cfg(not(target_arch = "wasm32"))]
        let coh_scores: Vec<f64> = (0..m_chars).into_par_iter().map(eval_char).collect();
        #[cfg(target_arch = "wasm32")]
        let coh_scores: Vec<f64> = (0..m_chars).map(eval_char).collect();

        let mut site_coh = vec![1.0_f64; l];
        let mut site_cnt = vec![0_u32; l];
        for (k, &s) in char_sites.iter().enumerate() {
            if site_cnt[s] == 0 {
                site_coh[s] = coh_scores[k];
            } else {
                site_coh[s] = site_coh[s].max(coh_scores[k]);
            }
            site_cnt[s] += 1;
        }

        let mut inf_vals: Vec<f64> = (0..l)
            .filter(|&s| site_cnt[s] > 0)
            .map(|s| site_coh[s])
            .collect();
        if !inf_vals.is_empty() {
            let med_coh = median_f64(&mut inf_vals);
            if med_coh > 1e-6 {
                for s in 0..l {
                    if site_cnt[s] > 0 {
                        site_coh[s] = (site_coh[s] / med_coh).clamp(0.65, 1.30);
                    }
                }
            }
        }

        let var_count = n_states.iter().filter(|&&st| st >= 2).count();
        let multi_count = n_states.iter().filter(|&&st| st >= 3).count();
        let frac_multi = if var_count > 0 {
            (multi_count as f64) / (var_count as f64)
        } else {
            0.0
        };

        let mut mod_dist = vec![0.0_f64; l];
        let mut mod_tree = vec![0.0_f64; l];
        let mut var_dist_sum = 0.0_f64;
        let mut var_dist_base_sum = 0.0_f64;

        for s in 0..l {
            let w_mt = if frac_multi > 0.50 { 1.0 } else { w_mult[s] };
            let coh_tree_factor = (site_coh[s] * w_mt).clamp(0.75, 1.25);
            // Only damp nonsynonymous sites that exhibit Four-Gamete incompatibility (coh_tree_factor < 0.90);
            // preserve Four-Gamete compatible synapomorphies at (1.0, 1.0).
            let (eff_d, eff_t) = if coh_tree_factor >= 0.90 {
                (1.0, 1.0)
            } else {
                (dist_weights[s], tree_weights[s])
            };
            mod_dist[s] = eff_d * site_coh[s].powf(1.25) * w_mult[s];
            mod_tree[s] = eff_t * coh_tree_factor;
            if n_states[s] >= 2 {
                var_dist_sum += mod_dist[s];
                var_dist_base_sum += eff_d;
            }
        }
        // Normalize variable-site distance weights so their average scale matches invariant sites,
        // preserving relative Four-Gamete compatibility weights without biasing P/Q saturation fractions.
        if var_dist_sum > 1e-6 && var_dist_base_sum > 1e-6 {
            let norm_factor = var_dist_base_sum / var_dist_sum;
            for s in 0..l {
                if n_states[s] >= 2 {
                    mod_dist[s] *= norm_factor;
                }
            }
        }
        (mod_dist, mod_tree)
    }

    #[allow(clippy::too_many_arguments)]
    fn select_coding_weights_by_mdl(
        &self,
        names: &[String],
        mat: &[i8],
        bitmasks: &[u8],
        majority_state: &[i8],
        dw_h0: &[f64],
        tw_h0: &[f64],
        dw_h1: &[f64],
        tw_h1: &[f64],
        n: usize,
        l: usize,
        l_eff: f64,
        ti_cost: f64,
        tv_cost: f64,
    ) -> bool {
        // Evaluate exact Two-Part MDL description length L(T, X) on up to K = 128 stratified taxa
        let k_sub = n.min(128);
        let sub_idx: Vec<usize> = if n <= 128 {
            (0..n).collect()
        } else {
            (0..k_sub)
                .map(|i| (i * (n - 1)) / (k_sub - 1))
                .collect()
        };
        let sub_names: Vec<String> = sub_idx.iter().map(|&i| names[i].clone()).collect();
        let mut sub_mat = Vec::with_capacity(k_sub * l);
        let mut sub_masks = Vec::with_capacity(k_sub * l);
        for &i in &sub_idx {
            sub_mat.extend_from_slice(&mat[i * l..(i + 1) * l]);
            sub_masks.extend_from_slice(&bitmasks[i * l..(i + 1) * l]);
        }

        let eval_hypothesis = |dw: &[f64], tw: &[f64]| -> f64 {
            let (loads, adj_0, mut labels_0) = self.build_subquadratic_loads_and_backbone(
                &sub_names,
                &sub_mat,
                &sub_masks,
                dw,
                tw,
                k_sub,
                l,
                l_eff,
                ti_cost,
                tv_cost,
                true,
            );
            let (adj_c, branches_c, _, _) = self.contract_unsupported_edges_mdl(
                adj_0,
                &sub_masks,
                majority_state,
                &loads,
                k_sub,
                l,
                l_eff,
                ti_cost,
                tv_cost,
                tw,
                &mut labels_0,
                true,
                None,
                true,
            );
            self.compute_tree_total_mdl_nats(&adj_c, &branches_c, l_eff)
        };

        #[cfg(not(target_arch = "wasm32"))]
        let (mdl_h0, mdl_h1) = rayon::join(
            || eval_hypothesis(dw_h0, tw_h0),
            || eval_hypothesis(dw_h1, tw_h1),
        );
        #[cfg(target_arch = "wasm32")]
        let (mdl_h0, mdl_h1) = (eval_hypothesis(dw_h0, tw_h0), eval_hypothesis(dw_h1, tw_h1));
        // Require H1 to compress the alignment by at least ln(2) nats (1-bit model choice)
        mdl_h1 + 2.0_f64.ln() < mdl_h0
    }

    #[inline]
    fn compute_word_weight_sums(site_weights: &[f64]) -> Vec<f64> {
        site_weights
            .chunks(64)
            .map(|ch| ch.iter().sum::<f64>())
            .collect()
    }

    /// 64-site bit-plane SIMD weighted Jin-Nei (K80 + Gamma) distance using hardware TZCNT/BLSR.
    #[inline]
    fn compute_pair_jc_dist_packed(
        pi: &PackedSeq64,
        pj: &PackedSeq64,
        site_weights: &[f64],
        word_weight_sums: &[f64],
        l_eff: f64,
        inv_alpha: f64,
    ) -> f64 {
        let mut w_valid = 0.0_f64;
        let mut w_ti = 0.0_f64;
        let mut w_tv = 0.0_f64;
        let n_words = pi.words.len();
        for w_idx in 0..n_words {
            let wi = pi.words[w_idx];
            let wj = pj.words[w_idx];
            let valid_both = wi.valid & wj.valid;
            if valid_both == 0 {
                continue;
            }
            let base = w_idx * 64;
            let w_len = (pi.len - base).min(64);
            let full_mask = if w_len == 64 {
                u64::MAX
            } else {
                (1_u64 << w_len) - 1
            };
            if valid_both == full_mask {
                w_valid += word_weight_sums[w_idx];
            } else {
                let mut vm = valid_both;
                while vm != 0 {
                    let tz = vm.trailing_zeros() as usize;
                    w_valid += site_weights[base + tz];
                    vm &= vm - 1;
                }
            }
            let same = (wi.b0 & wj.b0) | (wi.b1 & wj.b1) | (wi.b2 & wj.b2) | (wi.b3 & wj.b3);
            let diff = valid_both & !same;
            if diff != 0 {
                let ri = wi.b0 | wi.b2;
                let rj = wj.b0 | wj.b2;
                let mut tvm = diff & (ri ^ rj);
                let mut tim = diff & !tvm;
                while tim != 0 {
                    let tz = tim.trailing_zeros() as usize;
                    w_ti += site_weights[base + tz];
                    tim &= tim - 1;
                }
                while tvm != 0 {
                    let tz = tvm.trailing_zeros() as usize;
                    w_tv += site_weights[base + tz];
                    tvm &= tvm - 1;
                }
            }
        }
        let denom = w_valid.max(1.0);
        let p_ti = (w_ti / denom).clamp(0.0, 0.70);
        let q_tv = (w_tv / denom).clamp(0.0, 0.70);
        let a1 = (1.0 - 2.0 * p_ti - q_tv).clamp(0.02, 1.0);
        let a2 = (1.0 - 2.0 * q_tv).clamp(0.02, 1.0);
        if inv_alpha <= 1e-4 {
            (-0.5 * a1.ln() - 0.25 * a2.ln()) * l_eff
        } else {
            let alpha = 1.0 / inv_alpha;
            alpha * (0.5 * (a1.powf(-inv_alpha) - 1.0) + 0.25 * (a2.powf(-inv_alpha) - 1.0)) * l_eff
        }
    }

    /// 64-site bit-plane SIMD unweighted Jin-Nei (K80 + Gamma) distance using hardware POPCNT.
    #[inline]
    fn compute_pair_unweighted_jc_dist_packed(
        pi: &PackedSeq64,
        pj: &PackedSeq64,
        l_eff: f64,
        inv_alpha: f64,
    ) -> f64 {
        let mut n_valid = 0usize;
        let mut n_ti = 0usize;
        let mut n_tv = 0usize;
        for (wi, wj) in pi.words.iter().zip(&pj.words) {
            let valid_both = wi.valid & wj.valid;
            let same = (wi.b0 & wj.b0) | (wi.b1 & wj.b1) | (wi.b2 & wj.b2) | (wi.b3 & wj.b3);
            let diff = valid_both & !same;
            let ri = wi.b0 | wi.b2;
            let rj = wj.b0 | wj.b2;
            let tv = diff & (ri ^ rj);
            let ti = diff & !tv;
            n_valid += valid_both.count_ones() as usize;
            n_ti += ti.count_ones() as usize;
            n_tv += tv.count_ones() as usize;
        }
        let denom = n_valid.max(1) as f64;
        let p_ti = ((n_ti as f64) / denom).clamp(0.0, 0.70);
        let q_tv = ((n_tv as f64) / denom).clamp(0.0, 0.70);
        let a1 = (1.0 - 2.0 * p_ti - q_tv).clamp(0.02, 1.0);
        let a2 = (1.0 - 2.0 * q_tv).clamp(0.02, 1.0);
        if inv_alpha <= 1e-4 {
            (-0.5 * a1.ln() - 0.25 * a2.ln()) * l_eff
        } else {
            let alpha = 1.0 / inv_alpha;
            alpha * (0.5 * (a1.powf(-inv_alpha) - 1.0) + 0.25 * (a2.powf(-inv_alpha) - 1.0)) * l_eff
        }
    }

    /// First-principles Negative Binomial / Laplace zero-class estimator for site-rate
    /// dispersion `1 / alpha = Var(r_s)` on up to `K = min(N, 128)` stratified taxa.
    ///
    /// Under homogeneous site rates (`alpha -> infinity`), the Fitch step count `S_s` at site `s`
    /// is a sum of Bernoulli branch indicators with `Var(S_s) <= E[S_s]`, returning `0.0` (exact K80).
    /// Under heterogeneous rates `r_s ~ Gamma(alpha, alpha)`, `Var(S_s) > E[S_s]` and the
    /// invariant-site Laplace transform `f_0 = E[exp(-r_s T)] = (1 + T/alpha)^(-alpha)`
    /// together with the saturation-corrected Method-of-Moments yields `1 / alpha`.
    fn estimate_site_rate_inv_alpha(
        packed_vec: &[PackedSeq64],
        bitmasks: &[u8],
        n: usize,
        l: usize,
        l_eff: f64,
    ) -> f64 {
        if n < 6 || l < 20 {
            return 0.0;
        }
        let k_sub = n.min(128);
        let sub_idx: Vec<usize> = if n <= 128 {
            (0..n).collect()
        } else {
            (0..k_sub).map(|i| (i * (n - 1)) / (k_sub - 1)).collect()
        };
        let cap = 2 * k_sub + 2;
        let mut d = vec![0.0_f64; cap * cap];
        for a in 0..k_sub {
            let pa = &packed_vec[sub_idx[a]];
            for b in (a + 1)..k_sub {
                let pb = &packed_vec[sub_idx[b]];
                let dist = Self::compute_pair_unweighted_jc_dist_packed(pa, pb, l_eff, 0.0);
                d[a * cap + b] = dist;
                d[b * cap + a] = dist;
            }
        }
        let mut active: Vec<usize> = (0..k_sub).collect();
        let mut row_sums = vec![0.0_f64; cap];
        for a in 0..k_sub {
            row_sums[a] = (0..k_sub).map(|b| d[a * cap + b]).sum();
        }
        let mut masks: Vec<Vec<u8>> = Vec::with_capacity(cap);
        for &gi in &sub_idx {
            masks.push(bitmasks[gi * l..(gi + 1) * l].to_vec());
        }
        let mut steps = vec![0_u16; l];
        let chunks = l / 8;

        while active.len() > 1 {
            let k = active.len();
            let (a_pos, b_pos) = if k > 2 {
                let k_m2 = (k as f64) - 2.0;
                let mut best_q = f64::INFINITY;
                let mut best_p = (0usize, 1usize);
                for a in 0..k {
                    let la = active[a];
                    let ra = row_sums[la];
                    for b in (a + 1)..k {
                        let lb = active[b];
                        let q = k_m2 * d[la * cap + lb] - ra - row_sums[lb];
                        if q < best_q {
                            best_q = q;
                            best_p = (a, b);
                        }
                    }
                }
                best_p
            } else {
                (0usize, 1usize)
            };
            let lu = active[a_pos];
            let lv = active[b_pos];
            let lw = masks.len();
            let mut out = vec![0_u8; l];
            {
                let s1 = &masks[lu];
                let s2 = &masks[lv];
                for c in 0..chunks {
                    let off = c * 8;
                    let w1 = u64::from_ne_bytes(s1[off..off + 8].try_into().unwrap());
                    let w2 = u64::from_ne_bytes(s2[off..off + 8].try_into().unwrap());
                    let inter = w1 & w2;
                    let zero_hi =
                        !(inter.wrapping_add(0x7F7F_7F7F_7F7F_7F7F)) & 0x8080_8080_8080_8080;
                    if zero_hi == 0 {
                        out[off..off + 8].copy_from_slice(&inter.to_ne_bytes());
                    } else {
                        for i in off..(off + 8) {
                            let b1 = s1[i];
                            let b2 = s2[i];
                            let m_and = b1 & b2;
                            if m_and == 0 {
                                out[i] = b1 | b2;
                                if b1 != 15 && b2 != 15 {
                                    steps[i] += 1;
                                }
                            } else {
                                out[i] = m_and;
                            }
                        }
                    }
                }
                for i in (chunks * 8)..l {
                    let b1 = s1[i];
                    let b2 = s2[i];
                    let m_and = b1 & b2;
                    if m_and == 0 {
                        out[i] = b1 | b2;
                        if b1 != 15 && b2 != 15 {
                            steps[i] += 1;
                        }
                    } else {
                        out[i] = m_and;
                    }
                }
            }
            masks.push(out);

            let others: Vec<usize> = active
                .iter()
                .copied()
                .filter(|&x| x != lu && x != lv)
                .collect();
            if k > 2 {
                let k_m2 = (k as f64) - 2.0;
                let d_uv = d[lu * cap + lv];
                let limb_u = (0.5 * d_uv + (row_sums[lu] - row_sums[lv]) / (2.0 * k_m2)).max(0.0);
                let limb_v = (d_uv - limb_u).max(0.0);
                let mut rw = 0.0_f64;
                for &lx in &others {
                    let d_ux = d[lu * cap + lx];
                    let d_vx = d[lv * cap + lx];
                    let d_wx = (0.5 * (d_ux - limb_u + d_vx - limb_v)).max(0.0);
                    d[lw * cap + lx] = d_wx;
                    d[lx * cap + lw] = d_wx;
                    row_sums[lx] += d_wx - d_ux - d_vx;
                    rw += d_wx;
                }
                row_sums[lw] = rw;
            }
            active = others;
            active.push(lw);
        }

        let lf = l as f64;
        let mean_s: f64 = steps.iter().map(|&s| s as f64).sum::<f64>() / lf;
        let var_s: f64 = steps
            .iter()
            .map(|&s| {
                let df = (s as f64) - mean_s;
                df * df
            })
            .sum::<f64>()
            / lf;
        if mean_s < 0.05 || var_s <= mean_s * 1.05 {
            return 0.0;
        }
        let k_sat = 0.50 * (k_sub as f64);
        let mut sum_m = 0.0_f64;
        let mut m_vec = Vec::with_capacity(l);
        let mut zero_cnt = 0usize;
        for &s in &steps {
            if s == 0 {
                zero_cnt += 1;
            }
            let p_s = ((s as f64) / k_sat).clamp(0.0, 0.92);
            let m_s = (-k_sat * (1.0 - p_s).ln() - 0.5 * p_s).max(0.0);
            sum_m += m_s;
            m_vec.push(m_s);
        }
        let mean_m = sum_m / lf;
        let var_m: f64 = m_vec
            .iter()
            .map(|&m| {
                let df = m - mean_m;
                df * df
            })
            .sum::<f64>()
            / lf;
        let alpha_mom = (mean_m * mean_m) / (var_m - mean_m).max(1e-6);
        let f0 = (zero_cnt as f64) / lf;
        let mut alpha_f0 = alpha_mom;
        if f0 > 0.005 && f0 < 0.99 && f0 > (-mean_m * 0.98).exp() {
            let mut lo = 0.10_f64;
            let mut hi = 100.0_f64;
            for _ in 0..40 {
                let mid = 0.5 * (lo + hi);
                if (1.0 + mean_m / mid).powf(-mid) > f0 {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            alpha_f0 = 0.5 * (lo + hi);
        }
        let alpha = alpha_mom.min(alpha_f0).clamp(0.15, 100.0);
        if alpha >= 50.0 {
            0.0
        } else {
            1.0 / alpha
        }
    }

    /// Partition `items` into metric clades/slices of size `<= max_cluster_size` using
    /// Farthest-Point Voronoi Landmark Clustering ordered along a greedy nearest-neighbor chain.
    ///
    /// Unlike 1D two-pivot projection `d(x, p1) - d(x, p2)` (where root-to-tip distances cancel
    /// out to a constant across all clades orthogonal to `(p1, p2)`), Voronoi assignment
    /// `argmin_k d(x, L_k)` groups every taxon with the landmark inside its own clade or
    /// local trunk neighborhood in O(M * K * L / 64) time.
    fn partition_items_recursive(
        items: Vec<usize>,
        seqs: &HashMap<usize, PackedSeq64>,
        site_weights: &[f64],
        word_weight_sums: &[f64],
        l_eff: f64,
        inv_alpha: f64,
        max_cluster_size: usize,
        out_clusters: &mut Vec<Vec<usize>>,
    ) {
        let m = items.len();
        if m <= max_cluster_size {
            if !items.is_empty() {
                out_clusters.push(items);
            }
            return;
        }

        let k_land = (m / (max_cluster_size / 2).max(16)).clamp(3, 32);
        let p0 = items[0];
        let seq_p0 = &seqs[&p0];
        let mut l1 = p0;
        let mut max_d0 = -1.0_f64;
        for &x in &items {
            let d = Self::compute_pair_jc_dist_packed(
                seq_p0,
                &seqs[&x],
                site_weights,
                word_weight_sums,
                l_eff,
                inv_alpha,
            );
            if d > max_d0 {
                max_d0 = d;
                l1 = x;
            }
        }

        let mut landmarks: Vec<usize> = Vec::with_capacity(k_land);
        landmarks.push(l1);
        let mut min_dist = vec![f64::INFINITY; m];
        let mut sum_dist = vec![0.0_f64; m];
        let mut best_k = vec![0usize; m];
        let mut land_dists = vec![vec![0.0_f64; m]; k_land];

        for k_idx in 0..k_land {
            let lk = landmarks[k_idx];
            let seq_lk = &seqs[&lk];
            for (i, &x) in items.iter().enumerate() {
                let d = if x == lk {
                    0.0
                } else {
                    Self::compute_pair_jc_dist_packed(
                        seq_lk,
                        &seqs[&x],
                        site_weights,
                        word_weight_sums,
                        l_eff,
                        inv_alpha,
                    )
                };
                land_dists[k_idx][i] = d;
                sum_dist[i] += d;
                if d < min_dist[i] {
                    min_dist[i] = d;
                    best_k[i] = k_idx;
                }
            }
            if k_idx + 1 < k_land {
                // Select the 80th-percentile unrepresented sequence rather than the 100th-percentile
                // solitary outlier tip so landmarks land inside genuine clades.
                let mut cand_dists: Vec<(f64, usize)> = items
                    .iter()
                    .copied()
                    .enumerate()
                    .filter(|(_, x)| !landmarks.contains(x))
                    .map(|(i, x)| (min_dist[i], x))
                    .collect();
                if !cand_dists.is_empty() {
                    cand_dists.sort_by(|a, b| {
                        a.0.partial_cmp(&b.0)
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then_with(|| a.1.cmp(&b.1))
                    });
                    let q_idx = ((cand_dists.len() * 4) / 5).min(cand_dists.len() - 1);
                    landmarks.push(cand_dists[q_idx].1);
                }
            }
        }

        let mean_l_dist: Vec<f64> = sum_dist
            .into_iter()
            .map(|s| s / (k_land as f64))
            .collect();
        let mut cells: Vec<Vec<(f64, Vec<usize>)>> = vec![Vec::new(); k_land];

        // Lock sister cherries/clades into indivisible micro-clusters (size <= 8)
        // in strict O(M log M * L / 64) time using Johnson-Lindenstrauss 64-bit bit-plane LSH
        // projections on shared core valid sites + landmark distance projections.
        let n_words = seqs[&items[0]].words.len();
        let mut core_valid = vec![!0u64; n_words];
        let step_s = (m / 32).max(1);
        for s_idx in (0..m).step_by(step_s) {
            for (cw, w) in core_valid.iter_mut().zip(&seqs[&items[s_idx]].words) {
                *cw &= w.valid;
            }
        }
        let core_ones: u32 = core_valid.iter().map(|w| w.count_ones()).sum();
        if core_ones < 64 {
            core_valid.fill(!0u64);
        }

        let n_proj = 12usize;
        let mut proj_orders: Vec<Vec<usize>> = Vec::with_capacity(n_proj + 6);

        for r in 0..n_proj {
            let mut masks = Vec::with_capacity(n_words);
            let mut seed = 0x9E37_79B9_7F4A_7C15_u64
                .wrapping_mul((r as u64) + 1)
                .wrapping_add(0xBF58_476D_1CE4_E5B9_u64);
            for &cw in &core_valid {
                let mut next_u64 = || -> u64 {
                    seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15_u64);
                    let mut z = seed;
                    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9_u64);
                    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB_u64);
                    z ^ (z >> 31)
                };
                // Random 2-bit selector per site restricted to shared core valid sites `cw`
                let u = next_u64();
                let v = next_u64();
                let m0 = u & v;
                let m1 = u & !v;
                let m2 = !u & v;
                let m3 = !u & !v;
                masks.push((
                    (m0 | m1) & cw,
                    (m0 | m2) & cw,
                    (m1 | m3) & cw,
                    (m2 | m3) & cw,
                    cw,
                ));
            }
            let mut scored: Vec<(f64, usize)> = (0..m)
                .map(|i| {
                    let pi = &seqs[&items[i]];
                    let mut hits_sum = 0i32;
                    let mut val_sum = 0i32;
                    for (w, &(p0, p1, p2, p3, cw)) in pi.words.iter().zip(&masks) {
                        let h = (w.b0 & p0) | (w.b1 & p1) | (w.b2 & p2) | (w.b3 & p3);
                        hits_sum += h.count_ones() as i32;
                        val_sum += (w.valid & cw).count_ones() as i32;
                    }
                    let sc = (hits_sum as f64) / (val_sum.max(1) as f64);
                    (sc, i)
                })
                .collect();
            scored.sort_by(|a, b| {
                a.0.partial_cmp(&b.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.1.cmp(&b.1))
            });
            proj_orders.push(scored.into_iter().map(|x| x.1).collect());
        }

        // Also project centered landmark distance vectors (d(i, L_k) - mean_d(i)) along 4 random axes
        for r in 0..4usize {
            let mut l_signs = vec![1.0_f64; k_land];
            let mut seed = 0x517C_C1B7_2722_0A95_u64.wrapping_mul((r as u64) + 1);
            for s in &mut l_signs {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                *s = if (seed >> 63) == 0 { 1.0 } else { -1.0 };
            }
            let mut scored: Vec<(f64, usize)> = (0..m)
                .map(|i| {
                    let mut acc = 0.0_f64;
                    let mi = mean_l_dist[i];
                    for k_idx in 0..k_land {
                        acc += l_signs[k_idx] * (land_dists[k_idx][i] - mi);
                    }
                    (acc, i)
                })
                .collect();
            scored.sort_by(|a, b| {
                a.0.partial_cmp(&b.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.1.cmp(&b.1))
            });
            proj_orders.push(scored.into_iter().map(|x| x.1).collect());
        }

        // Also include 2 landmark-sorted orders (by primary cell + min_dist, and by mean_l_dist)
        let mut by_cell: Vec<usize> = (0..m).collect();
        by_cell.sort_by(|&a, &b| {
            best_k[a]
                .cmp(&best_k[b])
                .then_with(|| {
                    min_dist[a]
                        .partial_cmp(&min_dist[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| a.cmp(&b))
        });
        proj_orders.push(by_cell);

        let mut by_mean: Vec<usize> = (0..m).collect();
        by_mean.sort_by(|&a, &b| {
            mean_l_dist[a]
                .partial_cmp(&mean_l_dist[b])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });
        proj_orders.push(by_mean);

        let win_radius = 20usize;
        let mut cand_neighbors: Vec<Vec<usize>> =
            vec![Vec::with_capacity(win_radius * proj_orders.len() * 2); m];
        for order in &proj_orders {
            for pos in 0..m {
                let u = order[pos];
                let end = (pos + 1 + win_radius).min(m);
                for &v in &order[(pos + 1)..end] {
                    cand_neighbors[u].push(v);
                    cand_neighbors[v].push(u);
                }
            }
        }

        let mut nn_edges: Vec<(f64, usize, usize)> = Vec::with_capacity(m);
        for i in 0..m {
            let cands = &mut cand_neighbors[i];
            cands.sort_unstable();
            cands.dedup();
            let si = &seqs[&items[i]];
            let ri = mean_l_dist[i];
            let mut best_q = f64::INFINITY;
            let mut best_j = i;
            for &j in cands.iter() {
                if i != j {
                    let d = Self::compute_pair_jc_dist_packed(
                        si,
                        &seqs[&items[j]],
                        site_weights,
                        word_weight_sums,
                        l_eff,
                        inv_alpha,
                    );
                    let q = d - ri - mean_l_dist[j];
                    if q < best_q {
                        best_q = q;
                        best_j = j;
                    }
                }
            }
            if best_j != i {
                nn_edges.push((best_q, i, best_j));
            }
        }
        nn_edges.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });

        let mut uf_parent: Vec<usize> = (0..m).collect();
        let mut uf_members: Vec<Vec<usize>> = (0..m).map(|i| vec![i]).collect();
        for (_, u, v) in nn_edges {
            let mut ru = u;
            while uf_parent[ru] != ru {
                ru = uf_parent[ru];
            }
            let mut rv = v;
            while uf_parent[rv] != rv {
                rv = uf_parent[rv];
            }
            if ru != rv && uf_members[ru].len() + uf_members[rv].len() <= 8 {
                let moved = std::mem::take(&mut uf_members[rv]);
                uf_members[ru].extend(moved);
                uf_parent[rv] = ru;
            }
        }

        for r in 0..m {
            if uf_parent[r] == r && !uf_members[r].is_empty() {
                let mut best_cell = best_k[r];
                let mut best_cell_d = min_dist[r];
                for &mem_i in &uf_members[r] {
                    if min_dist[mem_i] < best_cell_d {
                        best_cell_d = min_dist[mem_i];
                        best_cell = best_k[mem_i];
                    }
                }
                let global_group: Vec<usize> =
                    uf_members[r].iter().map(|&idx| items[idx]).collect();
                cells[best_cell].push((best_cell_d, global_group));
            }
        }
        for cell in &mut cells {
            cell.sort_by(|a, b| {
                a.0.partial_cmp(&b.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.1[0].cmp(&b.1[0]))
            });
        }

        // Order landmarks along a greedy nearest-neighbor chain starting from landmark 0
        let mut visited_l = vec![false; k_land];
        let mut chain_order = Vec::with_capacity(k_land);
        let mut cur_l = 0usize;
        visited_l[0] = true;
        chain_order.push(0);
        for _ in 1..k_land {
            let seq_cur = &seqs[&landmarks[cur_l]];
            let mut best_next = 0usize;
            let mut best_d = f64::INFINITY;
            for cand in 0..k_land {
                if !visited_l[cand] {
                    let d = Self::compute_pair_jc_dist_packed(
                        seq_cur,
                        &seqs[&landmarks[cand]],
                        site_weights,
                        word_weight_sums,
                        l_eff,
                        inv_alpha,
                    );
                    if d < best_d {
                        best_d = d;
                        best_next = cand;
                    }
                }
            }
            visited_l[best_next] = true;
            chain_order.push(best_next);
            cur_l = best_next;
        }

        let mut current_pack: Vec<usize> = Vec::new();
        for &c_idx in &chain_order {
            for (_, group) in &cells[c_idx] {
                if current_pack.len() + group.len() > max_cluster_size && !current_pack.is_empty() {
                    out_clusters.push(std::mem::take(&mut current_pack));
                }
                current_pack.extend(group.iter().copied());
            }
        }
        if !current_pack.is_empty() {
            out_clusters.push(current_pack);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn agglomerate_bounded_cluster(
        &self,
        cluster_nodes: &[usize],
        seqs: &mut HashMap<usize, PackedSeq64>,
        site_weights: &[f64],
        word_weight_sums: &[f64],
        _l: usize,
        l_eff: f64,
        inv_alpha: f64,
        adj: &mut BTreeMap<usize, Vec<usize>>,
        node_labels: &mut BTreeMap<usize, String>,
        next_node: &mut usize,
        stop_at_one: bool,
    ) -> usize {
        let m = cluster_nodes.len();
        if m == 1 {
            return cluster_nodes[0];
        }
        if m == 2 {
            let u = cluster_nodes[0];
            let v = cluster_nodes[1];
            if stop_at_one {
                let w = *next_node;
                *next_node += 1;
                adj.entry(w).or_default().extend([u, v]);
                adj.entry(u).or_default().push(w);
                adj.entry(v).or_default().push(w);
                node_labels.insert(w, format!("N{w}"));
                let sw = PackedSeq64::merge_steiner_median(&seqs[&u], &seqs[&v], &seqs[&u]);
                seqs.insert(w, sw);
                return w;
            } else {
                adj.entry(u).or_default().push(v);
                adj.entry(v).or_default().push(u);
                return u;
            }
        }

        let cap = 2 * m + 2;
        let mut d_mat = vec![0.0_f64; cap * cap];
        let mut v_mat = vec![0.0_f64; cap * cap];
        let mut local_to_global: Vec<usize> = cluster_nodes.to_vec();

        #[cfg(not(target_arch = "wasm32"))]
        if m >= 64 {
            let seq_refs: Vec<&PackedSeq64> = (0..m).map(|i| &seqs[&local_to_global[i]]).collect();
            let rows: Vec<Vec<(f64, f64)>> = (0..m)
                .into_par_iter()
                .map(|i| {
                    let si = seq_refs[i];
                    let mut row = Vec::with_capacity(m - i - 1);
                    for j in (i + 1)..m {
                        let d_val = Self::compute_pair_jc_dist_packed(
                            si,
                            seq_refs[j],
                            site_weights,
                            word_weight_sums,
                            l_eff,
                            inv_alpha,
                        );
                        let v_val = d_val.max(0.5)
                            * (1.0 + (1.0 + inv_alpha) * (d_val / l_eff.max(1.0)).powi(2));
                        row.push((d_val, v_val));
                    }
                    row
                })
                .collect();
            for (i, row) in rows.into_iter().enumerate() {
                for (offset, (d_val, v_val)) in row.into_iter().enumerate() {
                    let j = i + 1 + offset;
                    d_mat[i * cap + j] = d_val;
                    d_mat[j * cap + i] = d_val;
                    v_mat[i * cap + j] = v_val;
                    v_mat[j * cap + i] = v_val;
                }
            }
        } else {
            for i in 0..m {
                let si = &seqs[&local_to_global[i]];
                for j in (i + 1)..m {
                    let sj = &seqs[&local_to_global[j]];
                    let d_val = Self::compute_pair_jc_dist_packed(
                        si,
                        sj,
                        site_weights,
                        word_weight_sums,
                        l_eff,
                        inv_alpha,
                    );
                    let v_val = d_val.max(0.5)
                        * (1.0 + (1.0 + inv_alpha) * (d_val / l_eff.max(1.0)).powi(2));
                    d_mat[i * cap + j] = d_val;
                    d_mat[j * cap + i] = d_val;
                    v_mat[i * cap + j] = v_val;
                    v_mat[j * cap + i] = v_val;
                }
            }
        }
        #[cfg(target_arch = "wasm32")]
        for i in 0..m {
            let si = &seqs[&local_to_global[i]];
            for j in (i + 1)..m {
                let sj = &seqs[&local_to_global[j]];
                let d_val = Self::compute_pair_jc_dist_packed(
                    si,
                    sj,
                    site_weights,
                    word_weight_sums,
                    l_eff,
                    inv_alpha,
                );
                let v_val = d_val.max(0.5)
                    * (1.0 + (1.0 + inv_alpha) * (d_val / l_eff.max(1.0)).powi(2));
                d_mat[i * cap + j] = d_val;
                d_mat[j * cap + i] = d_val;
                v_mat[i * cap + j] = v_val;
                v_mat[j * cap + i] = v_val;
            }
        }

        let mut active: Vec<usize> = (0..m).collect();
        let mut row_sums = vec![0.0_f64; cap];
        for i in 0..m {
            let mut s = 0.0_f64;
            for j in 0..m {
                s += d_mat[i * cap + j];
            }
            row_sums[i] = s;
        }

        let min_active = if stop_at_one { 2 } else { 3 };
        while active.len() > min_active {
            let k = active.len();
            let k_m2 = ((k as f64) - 2.0).max(1.0);
            #[cfg(not(target_arch = "wasm32"))]
            let best_pair = if k >= 256 {
                (0..(k - 1))
                    .into_par_iter()
                    .map(|a| {
                        let la = active[a];
                        let ra = row_sums[la];
                        let row_d = &d_mat[la * cap..];
                        let mut loc_q = f64::INFINITY;
                        let mut loc_b = a + 1;
                        for b in (a + 1)..k {
                            let lb = active[b];
                            let q_val = k_m2 * row_d[lb] - ra - row_sums[lb];
                            if q_val < loc_q {
                                loc_q = q_val;
                                loc_b = b;
                            }
                        }
                        (loc_q, a, loc_b)
                    })
                    .reduce(
                        || (f64::INFINITY, 0usize, 1usize),
                        |x, y| {
                            if y.0 < x.0 || (y.0 == x.0 && (y.1, y.2) < (x.1, x.2)) {
                                y
                            } else {
                                x
                            }
                        },
                    )
            } else {
                let mut best_q = f64::INFINITY;
                let mut bp = (0usize, 1usize);
                for a in 0..k {
                    let la = active[a];
                    let ra = row_sums[la];
                    let row_d = &d_mat[la * cap..];
                    for b in (a + 1)..k {
                        let lb = active[b];
                        let q_val = k_m2 * row_d[lb] - ra - row_sums[lb];
                        if q_val < best_q {
                            best_q = q_val;
                            bp = (a, b);
                        }
                    }
                }
                (best_q, bp.0, bp.1)
            };
            #[cfg(not(target_arch = "wasm32"))]
            let best_pair = (best_pair.1, best_pair.2);

            #[cfg(target_arch = "wasm32")]
            let best_pair = {
                let mut best_q = f64::INFINITY;
                let mut bp = (0usize, 1usize);
                for a in 0..k {
                    let la = active[a];
                    let ra = row_sums[la];
                    for b in (a + 1)..k {
                        let lb = active[b];
                        let q_val = k_m2 * d_mat[la * cap + lb] - ra - row_sums[lb];
                        if q_val < best_q {
                            best_q = q_val;
                            bp = (a, b);
                        }
                    }
                }
                bp
            };

            let (a_pos, b_pos) = best_pair;
            let lu = active[a_pos];
            let lv = active[b_pos];
            let gu = local_to_global[lu];
            let gv = local_to_global[lv];

            let gw = *next_node;
            *next_node += 1;
            let lw = local_to_global.len();
            local_to_global.push(gw);

            adj.entry(gw).or_default().extend([gu, gv]);
            adj.entry(gu).or_default().push(gw);
            adj.entry(gv).or_default().push(gw);
            node_labels.insert(gw, format!("N{gw}"));

            let others: Vec<usize> = active
                .iter()
                .copied()
                .filter(|&x| x != lu && x != lv)
                .collect();
            if stop_at_one {
                let gr = others
                    .iter()
                    .copied()
                    .min_by(|&a, &b| {
                        (d_mat[lu * cap + a] + d_mat[lv * cap + a])
                            .partial_cmp(&(d_mat[lu * cap + b] + d_mat[lv * cap + b]))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|lx| local_to_global[lx])
                    .unwrap_or(gu);
                let sw = PackedSeq64::merge_steiner_median(&seqs[&gu], &seqs[&gv], &seqs[&gr]);
                seqs.insert(gw, sw);
            }

            let d_uv = d_mat[lu * cap + lv];
            let v_uv = v_mat[lu * cap + lv].max(1e-6);
            let lam = if !others.is_empty() {
                let lam_num: f64 = others
                    .iter()
                    .map(|&x| v_mat[lv * cap + x] - v_mat[lu * cap + x])
                    .sum();
                (0.5 + lam_num / (2.0 * (others.len() as f64) * v_uv)).clamp(0.05, 0.95)
            } else {
                0.5
            };

            let limb_u = (0.5 * d_uv + (row_sums[lu] - row_sums[lv]) / (2.0 * k_m2)).max(0.0);
            let limb_v = (d_uv - limb_u).max(0.0);

            let mut rw_sum = 0.0_f64;
            for &lx in &others {
                let d_ux = d_mat[lu * cap + lx];
                let d_vx = d_mat[lv * cap + lx];
                let d_wx = (lam * (d_ux - limb_u) + (1.0 - lam) * (d_vx - limb_v)).max(0.0);
                let v_wx = (lam * v_mat[lu * cap + lx]
                    + (1.0 - lam) * v_mat[lv * cap + lx]
                    - lam * (1.0 - lam) * v_uv)
                    .max(0.25);
                d_mat[lw * cap + lx] = d_wx;
                d_mat[lx * cap + lw] = d_wx;
                v_mat[lw * cap + lx] = v_wx;
                v_mat[lx * cap + lw] = v_wx;
                row_sums[lx] += d_wx - d_ux - d_vx;
                rw_sum += d_wx;
            }
            row_sums[lw] = rw_sum;

            active.retain(|&x| x != lu && x != lv);
            active.push(lw);
        }

        if stop_at_one {
            let gu = local_to_global[active[0]];
            let gv = local_to_global[active[1]];
            let gw = *next_node;
            *next_node += 1;
            adj.entry(gw).or_default().extend([gu, gv]);
            adj.entry(gu).or_default().push(gw);
            adj.entry(gv).or_default().push(gw);
            node_labels.insert(gw, format!("N{gw}"));
            let sw = PackedSeq64::merge_steiner_median(&seqs[&gu], &seqs[&gv], &seqs[&gu]);
            seqs.insert(gw, sw);
            gw
        } else {
            let gw = *next_node;
            *next_node += 1;
            node_labels.insert(gw, format!("N{gw}"));
            for &lx in &active {
                let gx = local_to_global[lx];
                adj.entry(gw).or_default().push(gx);
                adj.entry(gx).or_default().push(gw);
            }
            gw
        }
    }

    /// Sub-quadratic O(N log N * L / 64) Outgroup-Witnessed Partial Agglomeration.
    ///
    /// Runs local Neighbor-Joining on `cluster_nodes` in the presence of `witness_nodes`
    /// (external outgroups from adjacent clusters and global anchors). The witness nodes
    /// participate in all Q-criterion row sums so paraphyletic trunk slices retain exact
    /// global orientation. Whenever a cluster node's minimum-Q partner is an external witness
    /// (or once `target_roots` subtrees remain), that subtree root graduates directly to the
    /// next tournament level rather than being forced into an artificial monophyletic cluster.
    #[allow(clippy::too_many_arguments)]
    fn agglomerate_witnessed_cluster_pure(
        &self,
        cluster_nodes: &[usize],
        witness_nodes: &[usize],
        seqs: &HashMap<usize, PackedSeq64>,
        site_weights: &[f64],
        word_weight_sums: &[f64],
        l_eff: f64,
        inv_alpha: f64,
        target_roots: usize,
    ) -> (Vec<usize>, Vec<(usize, usize, PackedSeq64)>) {
        const TEMP_BASE: usize = usize::MAX / 2;
        let m_c = cluster_nodes.len();
        if m_c <= target_roots.max(1) {
            return (cluster_nodes.to_vec(), Vec::new());
        }
        let mut local_to_global: Vec<usize> =
            Vec::with_capacity(m_c + witness_nodes.len());
        let mut is_witness: Vec<bool> =
            Vec::with_capacity(2 * m_c + witness_nodes.len() + 2);
        for &u in cluster_nodes {
            local_to_global.push(u);
            is_witness.push(false);
        }
        for &w in witness_nodes {
            if !local_to_global.contains(&w) {
                local_to_global.push(w);
                is_witness.push(true);
            }
        }
        let m_tot = local_to_global.len();
        let cap = m_tot + m_c + 2;
        let mut d_mat = vec![0.0_f64; cap * cap];
        let mut v_mat = vec![0.0_f64; cap * cap];

        for i in 0..m_tot {
            let si = &seqs[&local_to_global[i]];
            for j in (i + 1)..m_tot {
                let sj = &seqs[&local_to_global[j]];
                let d_val = Self::compute_pair_jc_dist_packed(
                    si,
                    sj,
                    site_weights,
                    word_weight_sums,
                    l_eff,
                    inv_alpha,
                );
                let v_val = d_val.max(0.5)
                    * (1.0 + (1.0 + inv_alpha) * (d_val / l_eff.max(1.0)).powi(2));
                d_mat[i * cap + j] = d_val;
                d_mat[j * cap + i] = d_val;
                v_mat[i * cap + j] = v_val;
                v_mat[j * cap + i] = v_val;
            }
        }

        let mut active: Vec<usize> = (0..m_tot).collect();
        let mut row_sums = vec![0.0_f64; cap];
        for i in 0..m_tot {
            let mut s = 0.0_f64;
            for j in 0..m_tot {
                s += d_mat[i * cap + j];
            }
            row_sums[i] = s;
        }

        let mut active_c_count = m_c;
        let mut graduated: Vec<usize> = Vec::new();
        let mut merges: Vec<(usize, usize, PackedSeq64)> = Vec::with_capacity(m_c);

        while active_c_count > target_roots && active_c_count >= 2 {
            let k = active.len();
            let k_m2 = ((k as f64) - 2.0).max(1.0);
            let mut best_q = f64::INFINITY;
            let mut best_pair = (0usize, 1usize);
            let mut best_cc_q = f64::INFINITY;
            let mut best_cc_pair = (0usize, 1usize);

            for a in 0..k {
                let la = active[a];
                let ra = row_sums[la];
                for b in (a + 1)..k {
                    let lb = active[b];
                    if is_witness[la] && is_witness[lb] {
                        continue;
                    }
                    let q_val = k_m2 * d_mat[la * cap + lb] - ra - row_sums[lb];
                    if q_val < best_q {
                        best_q = q_val;
                        best_pair = (a, b);
                    }
                    if !is_witness[la] && !is_witness[lb] && q_val < best_cc_q {
                        best_cc_q = q_val;
                        best_cc_pair = (a, b);
                    }
                }
            }

            let (qa, qb) = best_pair;
            if is_witness[active[qa]] != is_witness[active[qb]] {
                let c_local = if !is_witness[active[qa]] {
                    active[qa]
                } else {
                    active[qb]
                };
                graduated.push(local_to_global[c_local]);
                is_witness[c_local] = true;
                active_c_count -= 1;
                continue;
            }

            let (a_pos, b_pos) = best_cc_pair;
            let lu = active[a_pos];
            let lv = active[b_pos];
            let gu = local_to_global[lu];
            let gv = local_to_global[lv];

            let gw = TEMP_BASE + merges.len();
            let lw = local_to_global.len();
            local_to_global.push(gw);
            is_witness.push(false);

            let others: Vec<usize> = active
                .iter()
                .copied()
                .filter(|&x| x != lu && x != lv)
                .collect();
            let gr = others
                .iter()
                .copied()
                .min_by(|&a, &b| {
                    (d_mat[lu * cap + a] + d_mat[lv * cap + a])
                        .partial_cmp(&(d_mat[lu * cap + b] + d_mat[lv * cap + b]))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|lx| local_to_global[lx])
                .unwrap_or(gu);
            let sw = PackedSeq64::merge_steiner_median(
                if gu >= TEMP_BASE { &merges[gu - TEMP_BASE].2 } else { &seqs[&gu] },
                if gv >= TEMP_BASE { &merges[gv - TEMP_BASE].2 } else { &seqs[&gv] },
                if gr >= TEMP_BASE { &merges[gr - TEMP_BASE].2 } else { &seqs[&gr] },
            );
            merges.push((gu, gv, sw));

            let d_uv = d_mat[lu * cap + lv];
            let v_uv = v_mat[lu * cap + lv].max(1e-6);
            let lam = if !others.is_empty() {
                let lam_num: f64 = others
                    .iter()
                    .map(|&x| v_mat[lv * cap + x] - v_mat[lu * cap + x])
                    .sum();
                (0.5 + lam_num / (2.0 * (others.len() as f64) * v_uv)).clamp(0.05, 0.95)
            } else {
                0.5
            };

            let limb_u = (0.5 * d_uv + (row_sums[lu] - row_sums[lv]) / (2.0 * k_m2)).max(0.0);
            let limb_v = (d_uv - limb_u).max(0.0);

            let mut rw_sum = 0.0_f64;
            for &lx in &others {
                let d_ux = d_mat[lu * cap + lx];
                let d_vx = d_mat[lv * cap + lx];
                let d_wx = (lam * (d_ux - limb_u) + (1.0 - lam) * (d_vx - limb_v)).max(0.0);
                let v_wx = (lam * v_mat[lu * cap + lx]
                    + (1.0 - lam) * v_mat[lv * cap + lx]
                    - lam * (1.0 - lam) * v_uv)
                    .max(0.25);
                d_mat[lw * cap + lx] = d_wx;
                d_mat[lx * cap + lw] = d_wx;
                v_mat[lw * cap + lx] = v_wx;
                v_mat[lx * cap + lw] = v_wx;
                row_sums[lx] += d_wx - d_ux - d_vx;
                rw_sum += d_wx;
            }
            row_sums[lw] = rw_sum;

            active.retain(|&x| x != lu && x != lv);
            active.push(lw);
            active_c_count -= 1;
        }

        for &lx in &active {
            if !is_witness[lx] {
                graduated.push(local_to_global[lx]);
            }
        }
        (graduated, merges)
    }

    fn build_nt_gtr_loads_and_backbone(
        &self,
        names: &[String],
        mat: &[i8],
        bitmasks: &[u8],
        n: usize,
        l: usize,
        l_eff: f64,
    ) -> (
        Vec<f64>,
        BTreeMap<usize, Vec<usize>>,
        BTreeMap<usize, String>,
        Option<(HashMap<(usize, usize), (f64, f64)>, f64)>,
    ) {
        let mut node_labels: BTreeMap<usize, String> =
            (0..n).map(|i| (i, names[i].clone())).collect();
        let mut adj: BTreeMap<usize, Vec<usize>> = (0..n).map(|i| (i, Vec::new())).collect();

        if n == 1 {
            return (vec![0.5; 1], adj, node_labels, None);
        }
        #[cfg(not(target_arch = "wasm32"))]
        let packed_vec: Vec<PackedSeq64> = (0..n)
            .into_par_iter()
            .map(|i| PackedSeq64::from_row(&mat[i * l..(i + 1) * l]))
            .collect();
        #[cfg(target_arch = "wasm32")]
        let packed_vec: Vec<PackedSeq64> = (0..n)
            .map(|i| PackedSeq64::from_row(&mat[i * l..(i + 1) * l]))
            .collect();

        if n == 2 {
            let d01 = Self::compute_pair_unweighted_jc_dist_packed(
                &packed_vec[0],
                &packed_vec[1],
                l_eff,
                0.0,
            );
            let ld = (d01 * 0.5).max(0.5);
            adj.get_mut(&0).unwrap().push(1);
            adj.get_mut(&1).unwrap().push(0);
            return (vec![ld, ld], adj, node_labels, None);
        }

        let inv_alpha = Self::estimate_site_rate_inv_alpha(&packed_vec, bitmasks, n, l, l_eff);
        if n <= 4096 {
            #[cfg(not(target_arch = "wasm32"))]
            let uw_rows: Vec<Vec<f64>> = (0..n)
                .into_par_iter()
                .map(|i| {
                    let mut row = vec![0.0_f64; n];
                    let pi = &packed_vec[i];
                    for j in 0..n {
                        if i != j {
                            row[j] = Self::compute_pair_unweighted_jc_dist_packed(
                                pi,
                                &packed_vec[j],
                                l_eff,
                                inv_alpha,
                            );
                        }
                    }
                    row
                })
                .collect();
            #[cfg(target_arch = "wasm32")]
            let uw_rows: Vec<Vec<f64>> = (0..n)
                .map(|i| {
                    let mut row = vec![0.0_f64; n];
                    let pi = &packed_vec[i];
                    for j in 0..n {
                        if i != j {
                            row[j] = Self::compute_pair_unweighted_jc_dist_packed(
                                pi,
                                &packed_vec[j],
                                l_eff,
                                inv_alpha,
                            );
                        }
                    }
                    row
                })
                .collect();

            let calc_load = |i: usize| -> f64 {
                let row_i = &uw_rows[i];
                let mut j_near = if i == 0 { 1 } else { 0 };
                let mut d_min = row_i[j_near];
                for j in 0..n {
                    if j != i && row_i[j] < d_min {
                        d_min = row_i[j];
                        j_near = j;
                    }
                }
                let row_near = &uw_rows[j_near];
                let mut deltas: Vec<f64> = (0..n)
                    .filter(|&k| k != i && k != j_near)
                    .map(|k| row_i[k] - row_near[k])
                    .collect();
                let med = median_f64(&mut deltas);
                (0.5 * (d_min + med)).clamp(0.4, d_min.max(0.8))
            };
            #[cfg(not(target_arch = "wasm32"))]
            let loads: Vec<f64> = (0..n).into_par_iter().map(calc_load).collect();
            #[cfg(target_arch = "wasm32")]
            let loads: Vec<f64> = (0..n).map(calc_load).collect();

            if n == 3 {
                let center = n;
                adj.insert(center, vec![0, 1, 2]);
                node_labels.insert(center, format!("N{center}"));
                for i in 0..3 {
                    adj.get_mut(&i).unwrap().push(center);
                }
                return (loads, adj, node_labels, None);
            }

            let mut uw_dist: Vec<f64> = Vec::with_capacity(n * n);
            for row in uw_rows {
                uw_dist.extend_from_slice(&row);
            }
            let all_ids: Vec<usize> = (0..n).collect();
            let mut next_node = n;
            let mut edge_lens: HashMap<(usize, usize), f64> = HashMap::with_capacity(2 * n);
            nt_gtr::agglomerate_nt_bionj_from_dist(
                &all_ids,
                uw_dist,
                Some((&mut edge_lens, l_eff)),
                &mut adj,
                &mut node_labels,
                &mut next_node,
            );
            let gtr_stats = nt_gtr::refine_and_score_nt_gtr_mdl_opts(
                &mut adj,
                &mut edge_lens,
                mat,
                bitmasks,
                n,
                l,
                l_eff,
                self.empirical_alpha,
            );
            return (loads, adj, node_labels, Some(gtr_stats));
        }

        // Subquadratic Voronoi-portal initial skeleton + 64-site SoA GTR+CAT8 refinement for N > 4096
        let ones = vec![1.0_f64; l];
        let (loads, mut adj_sub, labels_sub) = self.build_subquadratic_loads_and_backbone(
            names, mat, bitmasks, &ones, &ones, n, l, l_eff, 1.0, 1.5, false,
        );
        let mut edge_lens: HashMap<(usize, usize), f64> = HashMap::with_capacity(2 * n);
        for (&u, nbs) in &adj_sub {
            for &v in nbs {
                if u < v {
                    let t_init = if u < n {
                        (loads[u] / l_eff.max(1.0)).clamp(1e-4, 1.5)
                    } else if v < n {
                        (loads[v] / l_eff.max(1.0)).clamp(1e-4, 1.5)
                    } else {
                        0.02
                    };
                    edge_lens.insert((u, v), t_init);
                }
            }
        }
        let gtr_stats = nt_gtr::refine_and_score_nt_gtr_mdl_opts(
            &mut adj_sub,
            &mut edge_lens,
            mat,
            bitmasks,
            n,
            l,
            l_eff,
            self.empirical_alpha,
        );
        (loads, adj_sub, labels_sub, Some(gtr_stats))
    }

    #[allow(clippy::too_many_arguments)]
    fn build_subquadratic_loads_and_backbone(
        &self,
        names: &[String],
        mat: &[i8],
        bitmasks: &[u8],
        dist_weights: &[f64],
        tree_weights: &[f64],
        n: usize,
        l: usize,
        l_eff: f64,
        ti_cost: f64,
        tv_cost: f64,
        run_fitch_nni: bool,
    ) -> (Vec<f64>, BTreeMap<usize, Vec<usize>>, BTreeMap<usize, String>) {
        let mut node_labels: BTreeMap<usize, String> =
            (0..n).map(|i| (i, names[i].clone())).collect();
        let mut adj: BTreeMap<usize, Vec<usize>> = (0..n).map(|i| (i, Vec::new())).collect();

        if n == 1 {
            return (vec![0.5; 1], adj, node_labels);
        }
        #[cfg(not(target_arch = "wasm32"))]
        let packed_vec: Vec<PackedSeq64> = (0..n)
            .into_par_iter()
            .map(|i| PackedSeq64::from_row(&mat[i * l..(i + 1) * l]))
            .collect();
        #[cfg(target_arch = "wasm32")]
        let packed_vec: Vec<PackedSeq64> = (0..n)
            .map(|i| PackedSeq64::from_row(&mat[i * l..(i + 1) * l]))
            .collect();
        let word_weight_sums = Self::compute_word_weight_sums(dist_weights);

        if n == 2 {
            let d01 = Self::compute_pair_unweighted_jc_dist_packed(
                &packed_vec[0],
                &packed_vec[1],
                l_eff,
                0.0,
            );
            let ld = (d01 * 0.5).max(0.5);
            adj.get_mut(&0).unwrap().push(1);
            adj.get_mut(&1).unwrap().push(0);
            return (vec![ld, ld], adj, node_labels);
        }
        let inv_alpha = Self::estimate_site_rate_inv_alpha(&packed_vec, bitmasks, n, l, l_eff);
        if n <= 512 {
            // Exact SIMD + parallel O(N^2) agglomeration path for N <= 512
            #[cfg(not(target_arch = "wasm32"))]
            let uw_rows: Vec<Vec<f64>> = (0..n)
                .into_par_iter()
                .map(|i| {
                    let mut row = vec![0.0_f64; n];
                    let pi = &packed_vec[i];
                    for j in 0..n {
                        if i != j {
                            row[j] = Self::compute_pair_unweighted_jc_dist_packed(
                                pi,
                                &packed_vec[j],
                                l_eff,
                                inv_alpha,
                            );
                        }
                    }
                    row
                })
                .collect();
            #[cfg(target_arch = "wasm32")]
            let uw_rows: Vec<Vec<f64>> = (0..n)
                .map(|i| {
                    let mut row = vec![0.0_f64; n];
                    let pi = &packed_vec[i];
                    for j in 0..n {
                        if i != j {
                            row[j] = Self::compute_pair_unweighted_jc_dist_packed(
                                pi,
                                &packed_vec[j],
                                l_eff,
                                inv_alpha,
                            );
                        }
                    }
                    row
                })
                .collect();

            let mut loads = vec![0.0_f64; n];
            for i in 0..n {
                let row_i = &uw_rows[i];
                let mut j_near = if i == 0 { 1 } else { 0 };
                let mut d_min = row_i[j_near];
                for j in 0..n {
                    if j != i && row_i[j] < d_min {
                        d_min = row_i[j];
                        j_near = j;
                    }
                }
                let row_near = &uw_rows[j_near];
                let mut deltas: Vec<f64> = (0..n)
                    .filter(|&k| k != i && k != j_near)
                    .map(|k| row_i[k] - row_near[k])
                    .collect();
                let med = median_f64(&mut deltas);
                loads[i] = (0.5 * (d_min + med)).clamp(0.4, d_min.max(0.8));
            }

            if n == 3 {
                let center = n;
                adj.insert(center, vec![0, 1, 2]);
                node_labels.insert(center, format!("N{center}"));
                for i in 0..3 {
                    adj.get_mut(&i).unwrap().push(center);
                }
                return (loads, adj, node_labels);
            }
            if n == 4 {
                let t: Vec<(Vec<u8>, f64)> = (0..4)
                    .map(|i| (bitmasks[i * l..(i + 1) * l].to_vec(), loads[i]))
                    .collect();
                let splits = [([0, 1], [2, 3]), ([0, 2], [1, 3]), ([0, 3], [1, 2])];
                let mut best_idx = 0;
                let mut best_gain = f64::NEG_INFINITY;
                for (idx, (p1, p2)) in splits.iter().enumerate() {
                    let (g, _, _) = self.compute_split_mdl_gain_from_children(
                        &[t[p1[0]].clone(), t[p1[1]].clone()],
                        &[t[p2[0]].clone(), t[p2[1]].clone()],
                        l,
                        l_eff,
                        ti_cost,
                        tv_cost,
                        tree_weights,
                    );
                    if g > best_gain {
                        best_gain = g;
                        best_idx = idx;
                    }
                }
                if best_gain > 0.0 {
                    let (p1, p2) = splits[best_idx];
                    let (u, v) = (4, 5);
                    adj.insert(u, vec![p1[0], p1[1], v]);
                    adj.insert(v, vec![p2[0], p2[1], u]);
                    node_labels.insert(u, format!("N{u}"));
                    node_labels.insert(v, format!("N{v}"));
                    for &x in &p1 {
                        adj.get_mut(&x).unwrap().push(u);
                    }
                    for &x in &p2 {
                        adj.get_mut(&x).unwrap().push(v);
                    }
                } else {
                    let center = 4;
                    adj.insert(center, vec![0, 1, 2, 3]);
                    node_labels.insert(center, format!("N{center}"));
                    for i in 0..4 {
                        adj.get_mut(&i).unwrap().push(center);
                    }
                }
                return (loads, adj, node_labels);
            }

            let mut seqs_map: HashMap<usize, PackedSeq64> =
                packed_vec.into_iter().enumerate().collect();
            let all_nodes: Vec<usize> = (0..n).collect();
            let mut next_node = n;
            self.agglomerate_bounded_cluster(
                &all_nodes,
                &mut seqs_map,
                dist_weights,
                &word_weight_sums,
                l,
                l_eff,
                inv_alpha,
                &mut adj,
                &mut node_labels,
                &mut next_node,
                false,
            );
            if run_fitch_nni {
                let nni_rounds = if n <= 512 { 5 } else { 2 };
                self.refine_backbone_nni_mdl(
                    &mut adj,
                    bitmasks,
                    &loads,
                    n,
                    l,
                    l_eff,
                    ti_cost,
                    tv_cost,
                    tree_weights,
                    nni_rounds,
                );
            }
            return (loads, adj, node_labels);
        }

        // Sub-Quadratic O(N log N * L) Outgroup-Portal Divide-and-Conquer Path for N > 128
        let mut seqs_map: HashMap<usize, PackedSeq64> =
            packed_vec.into_iter().enumerate().collect();
        let mut leaf_clusters: Vec<Vec<usize>> = Vec::new();
        Self::partition_items_recursive(
            (0..n).collect(),
            &seqs_map,
            dist_weights,
            &word_weight_sums,
            l_eff,
            inv_alpha,
            64,
            &mut leaf_clusters,
        );

        // Compute 3-point terminal loads within each leaf cluster + 32 global anchors in O(N * L / 64)
        let n_anc = 32.min(n);
        let global_anchors: Vec<usize> = (0..n_anc).map(|i| (i * (n - 1)) / (n_anc - 1)).collect();
        let compute_cluster_loads = |cluster: &Vec<usize>| -> Vec<(usize, f64)> {
            let mut ref_set: Vec<usize> = cluster.clone();
            for &ga in &global_anchors {
                if !ref_set.contains(&ga) {
                    ref_set.push(ga);
                }
            }
            let m_ref = ref_set.len();
            let mut dist_to_ref = vec![0.0_f64; m_ref * m_ref];
            for a in 0..m_ref {
                let sa = &seqs_map[&ref_set[a]];
                for b in (a + 1)..m_ref {
                    let sb = &seqs_map[&ref_set[b]];
                    let d =
                        Self::compute_pair_unweighted_jc_dist_packed(sa, sb, l_eff, inv_alpha);
                    dist_to_ref[a * m_ref + b] = d;
                    dist_to_ref[b * m_ref + a] = d;
                }
            }
            let mut out = Vec::with_capacity(cluster.len());
            for (loc_i, &gi) in cluster.iter().enumerate() {
                let mut best_b = if loc_i == 0 { 1 } else { 0 };
                let mut d_min = dist_to_ref[loc_i * m_ref + best_b];
                for b in 0..m_ref {
                    if b != loc_i && dist_to_ref[loc_i * m_ref + b] < d_min {
                        d_min = dist_to_ref[loc_i * m_ref + b];
                        best_b = b;
                    }
                }
                let mut deltas: Vec<f64> = (0..m_ref)
                    .filter(|&k| k != loc_i && k != best_b)
                    .map(|k| dist_to_ref[loc_i * m_ref + k] - dist_to_ref[best_b * m_ref + k])
                    .collect();
                let med = median_f64(&mut deltas);
                out.push((gi, (0.5 * (d_min + med)).clamp(0.4, d_min.max(0.8))));
            }
            out
        };

        #[cfg(not(target_arch = "wasm32"))]
        let cluster_load_lists: Vec<Vec<(usize, f64)>> = leaf_clusters
            .par_iter()
            .map(compute_cluster_loads)
            .collect();
        #[cfg(target_arch = "wasm32")]
        let cluster_load_lists: Vec<Vec<(usize, f64)>> =
            leaf_clusters.iter().map(compute_cluster_loads).collect();

        let mut loads = vec![0.5_f64; n];
        for list in cluster_load_lists {
            for (gi, ld) in list {
                loads[gi] = ld;
            }
        }

        // Outgroup-Witnessed Partial Agglomeration tournament in O(N log N * L / 64)
        const TEMP_BASE: usize = usize::MAX / 2;
        let mut next_node = n;
        let mut current_clusters = leaf_clusters;
        let mut prev_root_count = n;
        let current_roots: Vec<usize> = loop {
            let n_cl = current_clusters.len();
            let run_one_cluster = |c_idx: usize| -> (Vec<usize>, Vec<(usize, usize, PackedSeq64)>) {
                let cluster = &current_clusters[c_idx];
                let mut witnesses: Vec<usize> = Vec::with_capacity(20);
                if c_idx > 0 {
                    for &w in current_clusters[c_idx - 1].iter().rev().take(6) {
                        witnesses.push(w);
                    }
                }
                if c_idx + 1 < n_cl {
                    for &w in current_clusters[c_idx + 1].iter().take(6) {
                        witnesses.push(w);
                    }
                }
                for (step_i, &ga) in global_anchors.iter().enumerate() {
                    if step_i % 4 == 0 && !cluster.contains(&ga) {
                        witnesses.push(ga);
                    }
                }
                self.agglomerate_witnessed_cluster_pure(
                    cluster,
                    &witnesses,
                    &seqs_map,
                    dist_weights,
                    &word_weight_sums,
                    l_eff,
                    inv_alpha,
                    8,
                )
            };

            #[cfg(not(target_arch = "wasm32"))]
            let cluster_results: Vec<(Vec<usize>, Vec<(usize, usize, PackedSeq64)>)> =
                (0..n_cl).into_par_iter().map(run_one_cluster).collect();
            #[cfg(target_arch = "wasm32")]
            let cluster_results: Vec<(Vec<usize>, Vec<(usize, usize, PackedSeq64)>)> =
                (0..n_cl).map(run_one_cluster).collect();

            let mut next_roots = Vec::new();
            for (raw_grad, merges) in cluster_results {
                let base_id = next_node;
                next_node += merges.len();
                let remap = |x: usize| -> usize {
                    if x >= TEMP_BASE {
                        base_id + (x - TEMP_BASE)
                    } else {
                        x
                    }
                };
                for (m_idx, (raw_u, raw_v, sw)) in merges.into_iter().enumerate() {
                    let gw = base_id + m_idx;
                    let gu = remap(raw_u);
                    let gv = remap(raw_v);
                    adj.entry(gw).or_default().extend([gu, gv]);
                    adj.entry(gu).or_default().push(gw);
                    adj.entry(gv).or_default().push(gw);
                    node_labels.insert(gw, format!("N{gw}"));
                    seqs_map.insert(gw, sw);
                }
                next_roots.extend(raw_grad.into_iter().map(remap));
            }
            if next_roots.len() <= 160 || next_roots.len() * 5 >= prev_root_count * 4 {
                break next_roots;
            }
            prev_root_count = next_roots.len();
            let mut super_clusters: Vec<Vec<usize>> = Vec::new();
            Self::partition_items_recursive(
                next_roots,
                &seqs_map,
                dist_weights,
                &word_weight_sums,
                l_eff,
                inv_alpha,
                64,
                &mut super_clusters,
            );
            current_clusters = super_clusters;
        };

        self.agglomerate_bounded_cluster(
            &current_roots,
            &mut seqs_map,
            dist_weights,
            &word_weight_sums,
            l,
            l_eff,
            inv_alpha,
            &mut adj,
            &mut node_labels,
            &mut next_node,
            false,
        );

        if run_fitch_nni {
            let nni_rounds = if n <= 1024 { 5 } else { 3 };
            self.refine_backbone_nni_mdl(
                &mut adj,
                bitmasks,
                &loads,
                n,
                l,
                l_eff,
                ti_cost,
                tv_cost,
                tree_weights,
                nni_rounds,
            );
        }
        (loads, adj, node_labels)
    }

    fn propagate_directed_fitch_outward(
        adj: &BTreeMap<usize, Vec<usize>>,
        states: &mut HashMap<(usize, usize), (Vec<u8>, f64)>,
        mut queue: VecDeque<(usize, usize, u8)>,
        n: usize,
        mut dirty_out: Option<&mut HashSet<(usize, usize)>>,
    ) {
        while let Some((x, y, depth)) = queue.pop_front() {
            if x < n {
                continue;
            }
            let child_refs: Vec<(&[u8], f64)> = adj[&x]
                .iter()
                .copied()
                .filter(|&nb| nb != y)
                .map(|nb| {
                    let st = &states[&(nb, x)];
                    (st.0.as_slice(), st.1)
                })
                .collect();
            let new_state = Self::combine_fitch_children_refs(&child_refs);
            let changed = match states.get(&(x, y)) {
                Some((old_mask, old_load)) => {
                    new_state.0 != *old_mask || (new_state.1 - *old_load).abs() > 1e-4
                }
                None => true,
            };
            states.insert((x, y), new_state);
            if let Some(ref mut dset) = dirty_out {
                dset.insert((x, y));
            }
            let max_d = if n <= 1024 { 12 } else { 4 };
            if changed && depth < max_d && y >= n {
                for &z in &adj[&y] {
                    if z != x {
                        queue.push_back((y, z, depth + 1));
                    }
                }
            }
        }
    }

    fn propagate_directed_fitch_outward_exact(
        adj: &BTreeMap<usize, Vec<usize>>,
        states: &mut HashMap<(usize, usize), (Vec<u8>, f64)>,
        mut queue: VecDeque<(usize, usize)>,
        n: usize,
    ) {
        while let Some((x, y)) = queue.pop_front() {
            if x < n {
                continue;
            }
            let child_refs: Vec<(&[u8], f64)> = adj[&x]
                .iter()
                .copied()
                .filter(|&nb| nb != y)
                .map(|nb| {
                    let st = &states[&(nb, x)];
                    (st.0.as_slice(), st.1)
                })
                .collect();
            let new_state = Self::combine_fitch_children_refs(&child_refs);
            let changed = match states.get(&(x, y)) {
                Some((old_mask, old_load)) => {
                    new_state.0 != *old_mask || (new_state.1 - *old_load).abs() > 1e-11
                }
                None => true,
            };
            states.insert((x, y), new_state);
            if changed && y >= n {
                for &z in &adj[&y] {
                    if z != x {
                        queue.push_back((y, z));
                    }
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn refine_backbone_nni_mdl(
        &self,
        adj: &mut BTreeMap<usize, Vec<usize>>,
        bitmasks: &[u8],
        terminal_loads: &[f64],
        n: usize,
        l: usize,
        l_eff: f64,
        ti_cost: f64,
        tv_cost: f64,
        site_weights: &[f64],
        max_rounds: usize,
    ) {
        let mut states =
            self.compute_all_directed_fitch_states(adj, bitmasks, n, l, terminal_loads, l_eff);

        for round_idx in 0..max_rounds {
            let mut improved = false;

            let mut internal_edges = Vec::new();
            for (&u, nbs) in adj.iter() {
                if u >= n && nbs.len() == 3 {
                    for &v in nbs {
                        if u < v && v >= n && adj.get(&v).map(|x| x.len()) == Some(3) {
                            internal_edges.push((u, v));
                        }
                    }
                }
            }

            let eval_nni_edge = |u: usize, v: usize, cur_adj: &BTreeMap<usize, Vec<usize>>, cur_states: &HashMap<(usize, usize), (Vec<u8>, f64)>| -> (usize, usize, usize, usize, f64, f64, f64) {
                let mut u_it = cur_adj[&u].iter().copied().filter(|&x| x != v);
                let mut v_it = cur_adj[&v].iter().copied().filter(|&y| y != u);
                let (a, b) = (u_it.next().unwrap(), u_it.next().unwrap());
                let (c, d) = (v_it.next().unwrap(), v_it.next().unwrap());
                let sa = &cur_states[&(a, u)];
                let sb = &cur_states[&(b, u)];
                let sc = &cur_states[&(c, v)];
                let sd = &cur_states[&(d, v)];
                let ra = (sa.0.as_slice(), sa.1);
                let rb = (sb.0.as_slice(), sb.1);
                let rc = (sc.0.as_slice(), sc.1);
                let rd = (sd.0.as_slice(), sd.1);
                let (g0, _, _) = self.compute_split_mdl_gain_from_child_refs(
                    &[ra, rb],
                    &[rc, rd],
                    l,
                    l_eff,
                    ti_cost,
                    tv_cost,
                    site_weights,
                );
                let (g1, _, _) = self.compute_split_mdl_gain_from_child_refs(
                    &[ra, rc],
                    &[rb, rd],
                    l,
                    l_eff,
                    ti_cost,
                    tv_cost,
                    site_weights,
                );
                let (g2, _, _) = self.compute_split_mdl_gain_from_child_refs(
                    &[ra, rd],
                    &[rb, rc],
                    l,
                    l_eff,
                    ti_cost,
                    tv_cost,
                    site_weights,
                );
                (a, b, c, d, g0, g1, g2)
            };

            #[cfg(not(target_arch = "wasm32"))]
            let pre_evals: Vec<(usize, usize, usize, usize, f64, f64, f64)> = internal_edges
                .par_iter()
                .map(|&(u, v)| eval_nni_edge(u, v, adj, &states))
                .collect();

            #[cfg(target_arch = "wasm32")]
            let pre_evals: Vec<(usize, usize, usize, usize, f64, f64, f64)> = internal_edges
                .iter()
                .map(|&(u, v)| eval_nni_edge(u, v, adj, &states))
                .collect();

            let mut dirty_directed: HashSet<(usize, usize)> = HashSet::new();

            for (idx, (u, v)) in internal_edges.into_iter().enumerate() {
                if !adj.get(&u).map(|x| x.contains(&v)).unwrap_or(false)
                    || adj[&u].len() != 3
                    || adj[&v].len() != 3
                {
                    continue;
                }
                let mut u_it = adj[&u].iter().copied().filter(|&x| x != v);
                let mut v_it = adj[&v].iter().copied().filter(|&y| y != u);
                let (a, b) = (u_it.next().unwrap(), u_it.next().unwrap());
                let (c, d) = (v_it.next().unwrap(), v_it.next().unwrap());

                let (a0, b0, c0, d0, pg0, pg1, pg2) = pre_evals[idx];
                let is_clean = (a, b, c, d) == (a0, b0, c0, d0)
                    && !dirty_directed.contains(&(a, u))
                    && !dirty_directed.contains(&(b, u))
                    && !dirty_directed.contains(&(c, v))
                    && !dirty_directed.contains(&(d, v));

                let (g0, g1, g2) = if is_clean {
                    (pg0, pg1, pg2)
                } else {
                    let (_, _, _, _, rg0, rg1, rg2) = eval_nni_edge(u, v, adj, &states);
                    (rg0, rg1, rg2)
                };

                let best_g = g0.max(g1).max(g2);
                if best_g > g0 + 0.10 {
                    let (swap_v, keep_v) = if g1 >= g2 { (c, d) } else { (d, c) };
                    let tb = states.remove(&(b, u)).unwrap();
                    let t_swap_v = states.remove(&(swap_v, v)).unwrap();
                    states.remove(&(u, b));
                    states.remove(&(v, swap_v));

                    adj.get_mut(&u).unwrap().retain(|&x| x != b);
                    adj.get_mut(&b).unwrap().retain(|&x| x != u);
                    adj.get_mut(&v).unwrap().retain(|&x| x != swap_v);
                    adj.get_mut(&swap_v).unwrap().retain(|&x| x != v);

                    adj.get_mut(&u).unwrap().push(swap_v);
                    adj.get_mut(&swap_v).unwrap().push(u);
                    adj.get_mut(&v).unwrap().push(b);
                    adj.get_mut(&b).unwrap().push(v);

                    states.insert((b, v), tb);
                    states.insert((swap_v, u), t_swap_v);
                    dirty_directed.insert((b, v));
                    dirty_directed.insert((swap_v, u));

                    let st_uv = {
                        let sa = &states[&(a, u)];
                        let ssw = &states[&(swap_v, u)];
                        Self::combine_fitch_children_refs(&[
                            (sa.0.as_slice(), sa.1),
                            (ssw.0.as_slice(), ssw.1),
                        ])
                    };
                    let st_vu = {
                        let skv = &states[&(keep_v, v)];
                        let sbv = &states[&(b, v)];
                        Self::combine_fitch_children_refs(&[
                            (skv.0.as_slice(), skv.1),
                            (sbv.0.as_slice(), sbv.1),
                        ])
                    };
                    states.insert((u, v), st_uv);
                    states.insert((v, u), st_vu);
                    dirty_directed.insert((u, v));
                    dirty_directed.insert((v, u));

                    let q = VecDeque::from([(u, a, 0), (u, swap_v, 0), (v, keep_v, 0), (v, b, 0)]);
                    Self::propagate_directed_fitch_outward(
                        adj,
                        &mut states,
                        q,
                        n,
                        Some(&mut dirty_directed),
                    );
                    improved = true;
                }
            }

            // 2-Hop Cross-Node Attractor Swap (Radius-2 SPR across u - w - v)
            if round_idx < 2 && n <= 1024 {
                let int_nodes: Vec<usize> = adj
                    .iter()
                    .filter(|(&w, nbs)| w >= n && nbs.len() == 3)
                    .map(|(&w, _)| w)
                    .collect();

                let eval_2hop_triple =
                    |w: usize,
                     u: usize,
                     v: usize,
                     adj_map: &BTreeMap<usize, Vec<usize>>,
                     st_map: &HashMap<(usize, usize), (Vec<u8>, f64)>|
                     -> Option<(usize, usize, usize, usize, usize, Option<(usize, usize)>)> {
                        if adj_map.get(&w).map(|x| x.len()) != Some(3)
                            || adj_map.get(&u).map(|x| x.len()) != Some(3)
                            || adj_map.get(&v).map(|x| x.len()) != Some(3)
                            || !adj_map[&u].contains(&w)
                            || !adj_map[&v].contains(&w)
                        {
                            return None;
                        }
                        let m_list: Vec<usize> = adj_map[&w]
                            .iter()
                            .copied()
                            .filter(|&x| x != u && x != v)
                            .collect();
                        if m_list.len() != 1 {
                            return None;
                        }
                        let m_node = m_list[0];
                        let u_ch: Vec<usize> =
                            adj_map[&u].iter().copied().filter(|&x| x != w).collect();
                        let v_ch: Vec<usize> =
                            adj_map[&v].iter().copied().filter(|&y| y != w).collect();
                        if u_ch.len() != 2 || v_ch.len() != 2 {
                            return None;
                        }
                        let (a, b) = (u_ch[0], u_ch[1]);
                        let (c, d) = (v_ch[0], v_ch[1]);

                        let ta = st_map.get(&(a, u))?;
                        let tb = st_map.get(&(b, u))?;
                        let tc = st_map.get(&(c, v))?;
                        let td = st_map.get(&(d, v))?;
                        let tm = st_map.get(&(m_node, w))?;

                        let ra = (ta.0.as_slice(), ta.1);
                        let rb = (tb.0.as_slice(), tb.1);
                        let rc = (tc.0.as_slice(), tc.1);
                        let rd = (td.0.as_slice(), td.1);
                        let rm = (tm.0.as_slice(), tm.1);

                        let t_u_curr = Self::combine_fitch_children_refs(&[ra, rb]);
                        let t_v_curr = Self::combine_fitch_children_refs(&[rc, rd]);
                        let (g_uw_0, _, _) = self.compute_split_mdl_gain_from_child_refs(
                            &[ra, rb],
                            &[rm, (t_v_curr.0.as_slice(), t_u_curr.1.max(0.0) * 0.0 + t_v_curr.1)],
                            l,
                            l_eff,
                            ti_cost,
                            tv_cost,
                            site_weights,
                        );
                        let (g_wv_0, _, _) = self.compute_split_mdl_gain_from_child_refs(
                            &[rc, rd],
                            &[rm, (t_u_curr.0.as_slice(), t_u_curr.1)],
                            l,
                            l_eff,
                            ti_cost,
                            tv_cost,
                            site_weights,
                        );
                        let base_2hop = g_uw_0 + g_wv_0;
                        let mut best_swap_gain = base_2hop + 0.50;
                        let mut best_pair: Option<(usize, usize)> = None;

                        for (u_swap, r_us, r_uk) in [(a, ra, rb), (b, rb, ra)] {
                            for (v_swap, r_vs, r_vk) in [(c, rc, rd), (d, rd, rc)] {
                                let t_u_new = Self::combine_fitch_children_refs(&[r_uk, r_vs]);
                                let t_v_new = Self::combine_fitch_children_refs(&[r_vk, r_us]);
                                let (g_uw_n, _, _) = self.compute_split_mdl_gain_from_child_refs(
                                    &[r_uk, r_vs],
                                    &[rm, (t_v_new.0.as_slice(), t_v_new.1)],
                                    l,
                                    l_eff,
                                    ti_cost,
                                    tv_cost,
                                    site_weights,
                                );
                                let (g_wv_n, _, _) = self.compute_split_mdl_gain_from_child_refs(
                                    &[r_vk, r_us],
                                    &[rm, (t_u_new.0.as_slice(), t_u_new.1)],
                                    l,
                                    l_eff,
                                    ti_cost,
                                    tv_cost,
                                    site_weights,
                                );
                                if (g_uw_n + g_wv_n) > best_swap_gain {
                                    best_swap_gain = g_uw_n + g_wv_n;
                                    best_pair = Some((u_swap, v_swap));
                                }
                            }
                        }
                        Some((a, b, c, d, m_node, best_pair))
                    };

                let mut init_triples: Vec<(usize, usize, usize)> = Vec::new();
                for &w in &int_nodes {
                    if let Some(nbs) = adj.get(&w) {
                        if nbs.len() == 3 {
                            let int_nbs: Vec<usize> = nbs
                                .iter()
                                .copied()
                                .filter(|&x| x >= n && adj.get(&x).map(|v| v.len()) == Some(3))
                                .collect();
                            for idx_i in 0..int_nbs.len() {
                                for idx_j in (idx_i + 1)..int_nbs.len() {
                                    init_triples.push((w, int_nbs[idx_i], int_nbs[idx_j]));
                                }
                            }
                        }
                    }
                }

                #[cfg(not(target_arch = "wasm32"))]
                let pre_2hop_vec: Vec<(
                    (usize, usize, usize),
                    Option<(usize, usize, usize, usize, usize, Option<(usize, usize)>)>,
                )> = init_triples
                    .par_iter()
                    .map(|&(w, u, v)| ((w, u, v), eval_2hop_triple(w, u, v, adj, &states)))
                    .collect();
                #[cfg(target_arch = "wasm32")]
                let pre_2hop_vec: Vec<(
                    (usize, usize, usize),
                    Option<(usize, usize, usize, usize, usize, Option<(usize, usize)>)>,
                )> = init_triples
                    .iter()
                    .map(|&(w, u, v)| ((w, u, v), eval_2hop_triple(w, u, v, adj, &states)))
                    .collect();

                let mut pre_2hop: HashMap<
                    (usize, usize, usize),
                    (usize, usize, usize, usize, usize, Option<(usize, usize)>),
                > = HashMap::with_capacity(pre_2hop_vec.len());
                for (k, val) in pre_2hop_vec {
                    if let Some(v_some) = val {
                        pre_2hop.insert(k, v_some);
                    }
                }

                let mut dirty_2hop: HashSet<(usize, usize)> = HashSet::new();

                for w in int_nodes {
                    if adj.get(&w).map(|x| x.len()) != Some(3) {
                        continue;
                    }
                    let nbs = adj[&w].clone();
                    let int_nbs: Vec<usize> = nbs
                        .iter()
                        .copied()
                        .filter(|&x| x >= n && adj.get(&x).map(|v| v.len()) == Some(3))
                        .collect();
                    if int_nbs.len() < 2 {
                        continue;
                    }
                    for idx_i in 0..int_nbs.len() {
                        for idx_j in (idx_i + 1)..int_nbs.len() {
                            let u = int_nbs[idx_i];
                            let v = int_nbs[idx_j];
                            if !adj[&u].contains(&w)
                                || !adj[&v].contains(&w)
                                || adj[&u].len() != 3
                                || adj[&v].len() != 3
                            {
                                continue;
                            }
                            let m_list: Vec<usize> =
                                adj[&w].iter().copied().filter(|&x| x != u && x != v).collect();
                            if m_list.len() != 1 {
                                continue;
                            }
                            let m_node = m_list[0];
                            let u_ch: Vec<usize> =
                                adj[&u].iter().copied().filter(|&x| x != w).collect();
                            let v_ch: Vec<usize> =
                                adj[&v].iter().copied().filter(|&y| y != w).collect();
                            if u_ch.len() != 2 || v_ch.len() != 2 {
                                continue;
                            }
                            let (a, b) = (u_ch[0], u_ch[1]);
                            let (c, d) = (v_ch[0], v_ch[1]);

                            let best_pair = if let Some(&(pa, pb, pc, pd, pm, pbest)) =
                                pre_2hop.get(&(w, u, v))
                            {
                                if (a, b, c, d, m_node) == (pa, pb, pc, pd, pm)
                                    && !dirty_2hop.contains(&(a, u))
                                    && !dirty_2hop.contains(&(b, u))
                                    && !dirty_2hop.contains(&(c, v))
                                    && !dirty_2hop.contains(&(d, v))
                                    && !dirty_2hop.contains(&(m_node, w))
                                {
                                    pbest
                                } else {
                                    eval_2hop_triple(w, u, v, adj, &states).and_then(|x| x.5)
                                }
                            } else {
                                eval_2hop_triple(w, u, v, adj, &states).and_then(|x| x.5)
                            };

                            if let Some((u_swap, v_swap)) = best_pair {
                                let u_keep = if u_swap == a { b } else { a };
                                let v_keep = if v_swap == c { d } else { c };
                                let t_us = states.remove(&(u_swap, u)).unwrap();
                                let t_vs = states.remove(&(v_swap, v)).unwrap();
                                states.remove(&(u, u_swap));
                                states.remove(&(v, v_swap));

                                adj.get_mut(&u).unwrap().retain(|&x| x != u_swap);
                                adj.get_mut(&u_swap).unwrap().retain(|&x| x != u);
                                adj.get_mut(&v).unwrap().retain(|&x| x != v_swap);
                                adj.get_mut(&v_swap).unwrap().retain(|&x| x != v);

                                adj.get_mut(&u).unwrap().push(v_swap);
                                adj.get_mut(&v_swap).unwrap().push(u);
                                adj.get_mut(&v).unwrap().push(u_swap);
                                adj.get_mut(&u_swap).unwrap().push(v);

                                states.insert((u_swap, v), t_us);
                                states.insert((v_swap, u), t_vs);
                                dirty_2hop.insert((u_swap, v));
                                dirty_2hop.insert((v_swap, u));

                                let st_uw = {
                                    let ruk = &states[&(u_keep, u)];
                                    let rvs = &states[&(v_swap, u)];
                                    Self::combine_fitch_children_refs(&[
                                        (ruk.0.as_slice(), ruk.1),
                                        (rvs.0.as_slice(), rvs.1),
                                    ])
                                };
                                let st_vw = {
                                    let rvk = &states[&(v_keep, v)];
                                    let rus = &states[&(u_swap, v)];
                                    Self::combine_fitch_children_refs(&[
                                        (rvk.0.as_slice(), rvk.1),
                                        (rus.0.as_slice(), rus.1),
                                    ])
                                };
                                let st_wu = {
                                    let rm = &states[&(m_node, w)];
                                    Self::combine_fitch_children_refs(&[
                                        (rm.0.as_slice(), rm.1),
                                        (st_vw.0.as_slice(), st_vw.1),
                                    ])
                                };
                                let st_wv = {
                                    let rm = &states[&(m_node, w)];
                                    Self::combine_fitch_children_refs(&[
                                        (rm.0.as_slice(), rm.1),
                                        (st_uw.0.as_slice(), st_uw.1),
                                    ])
                                };
                                states.insert((u, w), st_uw);
                                states.insert((v, w), st_vw);
                                states.insert((w, u), st_wu);
                                states.insert((w, v), st_wv);
                                dirty_2hop.insert((u, w));
                                dirty_2hop.insert((v, w));
                                dirty_2hop.insert((w, u));
                                dirty_2hop.insert((w, v));

                                let q = VecDeque::from([
                                    (u, u_keep, 0),
                                    (u, v_swap, 0),
                                    (v, v_keep, 0),
                                    (v, u_swap, 0),
                                    (w, m_node, 0),
                                ]);
                                Self::propagate_directed_fitch_outward(
                                    adj,
                                    &mut states,
                                    q,
                                    n,
                                    Some(&mut dirty_2hop),
                                );
                                improved = true;
                            }
                        }
                    }
                }
            }

            // 3. Subtree-Prune-and-Regraft (MDL-Certified Fitch-Guided SPR in O(N log N * L / 64))
            // Recovers leaves, clades, or inverted caterpillar segments across multiple hops
            // via zero-cost component-separated BFS and JL sequence-neighbor candidate edges,
            // leaving the final round for 1-hop MDL NNI local polishing.
            if round_idx + 1 < max_rounds {
                if self.refine_backbone_spr_step(
                    adj,
                    &mut states,
                    n,
                    l,
                    l_eff,
                    ti_cost,
                    tv_cost,
                    site_weights,
                ) {
                    improved = true;
                }
            }

            if !improved {
                break;
            }
        }
    }

    #[inline]
    fn fitch_insert_cost_pair(
        s_mask: &[u8],
        x_mask: &[u8],
        y_mask: &[u8],
        ti_w: f64,
        tv_w: f64,
        site_weights: &[f64],
    ) -> f64 {
        let l = s_mask.len();
        let chunks = l / 8;
        let mut cost = 0.0_f64;
        for c in 0..chunks {
            let off = c * 8;
            let ws = u64::from_ne_bytes(s_mask[off..off + 8].try_into().unwrap());
            let wx = u64::from_ne_bytes(x_mask[off..off + 8].try_into().unwrap());
            let wy = u64::from_ne_bytes(y_mask[off..off + 8].try_into().unwrap());
            let xy_and = wx & wy;
            let xy_zero_hi =
                !(xy_and.wrapping_add(0x7F7F_7F7F_7F7F_7F7F)) & 0x8080_8080_8080_8080;
            let u_xy = if xy_zero_hi == 0 {
                xy_and
            } else {
                let mut m = xy_zero_hi >> 7;
                m |= m << 1;
                m |= m << 2;
                m |= m << 4;
                (xy_and & !m) | ((wx | wy) & m)
            };
            let inter = ws & u_xy;
            let zero_hi = !(inter.wrapping_add(0x7F7F_7F7F_7F7F_7F7F)) & 0x8080_8080_8080_8080;
            if zero_hi != 0 {
                for i in off..(off + 8) {
                    let b1 = s_mask[i];
                    if b1 != 15 {
                        let bx = x_mask[i];
                        let by = y_mask[i];
                        let b2 = if bx == 15 {
                            by
                        } else if by == 15 {
                            bx
                        } else if (bx & by) != 0 {
                            bx & by
                        } else {
                            bx | by
                        };
                        if b2 != 15 && (b1 & b2) == 0 {
                            let is_ti = (((b1 | (b1 >> 2)) & (b2 | (b2 >> 2))) & 3) != 0;
                            cost += (if is_ti { ti_w } else { tv_w }) * site_weights[i];
                        }
                    }
                }
            }
        }
        for i in (chunks * 8)..l {
            let b1 = s_mask[i];
            if b1 != 15 {
                let bx = x_mask[i];
                let by = y_mask[i];
                let b2 = if bx == 15 {
                    by
                } else if by == 15 {
                    bx
                } else if (bx & by) != 0 {
                    bx & by
                } else {
                    bx | by
                };
                if b2 != 15 && (b1 & b2) == 0 {
                    let is_ti = (((b1 | (b1 >> 2)) & (b2 | (b2 >> 2))) & 3) != 0;
                    cost += (if is_ti { ti_w } else { tv_w }) * site_weights[i];
                }
            }
        }
        cost
    }

    #[allow(clippy::too_many_arguments)]
    fn refine_backbone_spr_step(
        &self,
        adj: &mut BTreeMap<usize, Vec<usize>>,
        states: &mut HashMap<(usize, usize), (Vec<u8>, f64)>,
        n: usize,
        l: usize,
        l_eff: f64,
        ti_cost: f64,
        tv_cost: f64,
        site_weights: &[f64],
    ) -> bool {
        // Every directed edge (s, u) where u >= n has degree 3 defines a valid SPR prune:
        // pruning u from between its other two neighbors (a, b) separates C_s from C_ab.
        let mut movable: Vec<(usize, usize)> = Vec::new();
        for (&u, nbs) in adj.iter() {
            if u >= n && nbs.len() == 3 {
                for &s in nbs {
                    if s < n || adj.get(&s).map(|v| v.len()) == Some(3) {
                        movable.push((s, u));
                    }
                }
            }
        }
        let m = movable.len();
        if m < 4 {
            return false;
        }

        // Build O(M log M * L / 8) JL projection neighbor table over `states[&(s, u)].0`
        let n_proj = 8usize;
        let mut cand_peers: Vec<Vec<usize>> = vec![Vec::with_capacity(64); m];
        for r in 0..n_proj {
            let mut seed = 0x9E37_79B9_7F4A_7C15_u64
                .wrapping_mul((r as u64) + 1)
                .wrapping_add(0xBF58_476D_1CE4_E5B9_u64);
            let mut masks = Vec::with_capacity(l);
            for _ in 0..l {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let b_pick = ((seed >> 60) as u8) & 3;
                let b_alt = ((seed >> 58) as u8) & 3;
                masks.push((1u8 << b_pick) | (1u8 << b_alt));
            }
            #[cfg(not(target_arch = "wasm32"))]
            let mut scored: Vec<(f64, usize)> = (0..m)
                .into_par_iter()
                .map(|idx| {
                    let (s, u) = movable[idx];
                    if let Some((s_mask, _)) = states.get(&(s, u)) {
                        let mut hits = 0i32;
                        let mut val = 0i32;
                        for (&b, &pm) in s_mask.iter().zip(&masks) {
                            if b != 15 {
                                val += 1;
                                if (b & pm) != 0 {
                                    hits += 1;
                                }
                            }
                        }
                        ((hits as f64) / (val.max(1) as f64), idx)
                    } else {
                        (0.0, idx)
                    }
                })
                .collect();

            #[cfg(target_arch = "wasm32")]
            let mut scored: Vec<(f64, usize)> = (0..m)
                .map(|idx| {
                    let (s, u) = movable[idx];
                    if let Some((s_mask, _)) = states.get(&(s, u)) {
                        let mut hits = 0i32;
                        let mut val = 0i32;
                        for (&b, &pm) in s_mask.iter().zip(&masks) {
                            if b != 15 {
                                val += 1;
                                if (b & pm) != 0 {
                                    hits += 1;
                                }
                            }
                        }
                        ((hits as f64) / (val.max(1) as f64), idx)
                    } else {
                        (0.0, idx)
                    }
                })
                .collect();
            scored.sort_by(|a, b| {
                a.0.partial_cmp(&b.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.1.cmp(&b.1))
            });
            let win = 6usize;
            for pos in 0..m {
                let ia = scored[pos].1;
                let end = (pos + 1 + win).min(m);
                for &(_, ib) in &scored[(pos + 1)..end] {
                    cand_peers[ia].push(ib);
                    cand_peers[ib].push(ia);
                }
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        cand_peers.par_iter_mut().for_each(|peers| {
            peers.sort_unstable();
            peers.dedup();
        });
        #[cfg(target_arch = "wasm32")]
        cand_peers.iter_mut().for_each(|peers| {
            peers.sort_unstable();
            peers.dedup();
        });

        let mean_sub = (0.5 * (ti_cost + tv_cost)).max(1e-6);
        let ti_w = (ti_cost / mean_sub).clamp(0.85, 1.05);
        let tv_w = (tv_cost / mean_sub).clamp(0.95, 1.18);
        let min_step = 0.85 * ti_w;

        let eval_spr_prescreen =
            |idx: usize,
             adj_map: &BTreeMap<usize, Vec<usize>>,
             st_map: &HashMap<(usize, usize), (Vec<u8>, f64)>|
             -> Option<(
                usize,
                usize,
                f64,
                Vec<usize>,
                Vec<(usize, usize)>,
                Vec<(f64, usize, usize)>,
            )> {
                let (s, u) = movable[idx];
                if adj_map.get(&u).map(|v| v.len()) != Some(3) || !adj_map[&u].contains(&s) {
                    return None;
                }
                let mut u_it = adj_map[&u].iter().copied().filter(|&x| x != s);
                let (Some(a), Some(b)) = (u_it.next(), u_it.next()) else {
                    return None;
                };
                if adj_map.get(&a).map(|v| v.contains(&b)).unwrap_or(false) {
                    return None;
                }
                let (Some(ts), Some(ta), Some(tb)) =
                    (st_map.get(&(s, u)), st_map.get(&(a, u)), st_map.get(&(b, u)))
                else {
                    return None;
                };
                let base_cost =
                    Self::fitch_insert_cost_pair(&ts.0, &ta.0, &tb.0, ti_w, tv_w, site_weights);
                if base_cost <= min_step {
                    return Some((a, b, base_cost, Vec::new(), Vec::new(), Vec::new()));
                }

                let mut sub_s: Vec<usize> = vec![s];
                let mut q_idx = 0;
                while q_idx < sub_s.len() && sub_s.len() <= 17 {
                    let curr = sub_s[q_idx];
                    q_idx += 1;
                    if let Some(cnbs) = adj_map.get(&curr) {
                        for &nx in cnbs {
                            if nx != u && !sub_s.contains(&nx) {
                                sub_s.push(nx);
                                if sub_s.len() > 17 {
                                    break;
                                }
                            }
                        }
                    }
                }
                let s_is_small = sub_s.len() <= 16;

                let mut cand_edges: Vec<(usize, usize)> = Vec::with_capacity(96);
                let mut bfs_vis: HashSet<usize> = HashSet::with_capacity(96);
                bfs_vis.insert(u);
                bfs_vis.insert(s);
                let mut bfs_q: VecDeque<(usize, u8)> = VecDeque::new();
                bfs_vis.insert(a);
                bfs_vis.insert(b);
                bfs_q.push_back((a, 0));
                bfs_q.push_back((b, 0));
                while let Some((curr, d_hop)) = bfs_q.pop_front() {
                    if d_hop >= 28 || bfs_vis.len() >= 128 {
                        continue;
                    }
                    if let Some(cnbs) = adj_map.get(&curr) {
                        for &nx in cnbs {
                            if nx != u && nx != s {
                                let e = if curr < nx { (curr, nx) } else { (nx, curr) };
                                cand_edges.push(e);
                                if bfs_vis.len() < 128 && bfs_vis.insert(nx) {
                                    bfs_q.push_back((nx, d_hop + 1));
                                }
                            }
                        }
                    }
                }

                let mut watched_nodes: Vec<usize> = bfs_vis.into_iter().collect();
                watched_nodes.extend(sub_s.iter().copied());
                if s_is_small {
                    for &p_idx in &cand_peers[idx] {
                        let (ps, pu) = movable[p_idx];
                        for center in [ps, pu] {
                            if center != u && !sub_s.contains(&center) {
                                watched_nodes.push(center);
                                if let Some(cnbs) = adj_map.get(&center) {
                                    for &nx in cnbs {
                                        if nx != u && !sub_s.contains(&nx) {
                                            let e = if center < nx {
                                                (center, nx)
                                            } else {
                                                (nx, center)
                                            };
                                            cand_edges.push(e);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                cand_edges.sort_unstable();
                cand_edges.dedup();

                let best_pre_cost = base_cost - min_step;
                let mut top_cands: Vec<(f64, usize, usize)> = Vec::new();
                for &(x, y) in &cand_edges {
                    if let (Some(tx), Some(ty)) = (st_map.get(&(x, y)), st_map.get(&(y, x))) {
                        let c_xy = Self::fitch_insert_cost_pair(
                            &ts.0,
                            &tx.0,
                            &ty.0,
                            ti_w,
                            tv_w,
                            site_weights,
                        );
                        if c_xy < best_pre_cost {
                            top_cands.push((c_xy, x, y));
                        }
                    }
                }
                if !top_cands.is_empty() {
                    top_cands.sort_by(|x, y| {
                        x.0.partial_cmp(&y.0)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    top_cands.truncate(6);
                }
                Some((a, b, base_cost, watched_nodes, cand_edges, top_cands))
            };

        let batch_size = 256usize;
        let mut any_moved = false;

        for batch_start in (0..m).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(m);

            #[cfg(not(target_arch = "wasm32"))]
            let use_par_batch = rayon::current_num_threads() > 1 && (batch_end - batch_start) >= 8;
            #[cfg(not(target_arch = "wasm32"))]
            let pre_spr: Vec<
                Option<(
                    usize,
                    usize,
                    f64,
                    Vec<usize>,
                    Vec<(usize, usize)>,
                    Vec<(f64, usize, usize)>,
                )>,
            > = if use_par_batch {
                (batch_start..batch_end)
                    .into_par_iter()
                    .map(|idx| eval_spr_prescreen(idx, adj, states))
                    .collect()
            } else {
                Vec::new()
            };

            #[cfg(target_arch = "wasm32")]
            let pre_spr: Vec<
                Option<(
                    usize,
                    usize,
                    f64,
                    Vec<usize>,
                    Vec<(usize, usize)>,
                    Vec<(f64, usize, usize)>,
                )>,
            > = Vec::new();

            let mut dirty_spr: HashSet<(usize, usize)> = HashSet::new();
            let mut dirty_nodes: HashSet<usize> = HashSet::new();

            for idx in batch_start..batch_end {
                let (s, u) = movable[idx];
                if adj.get(&u).map(|v| v.len()) != Some(3) || !adj[&u].contains(&s) {
                    continue;
                }
                let u_others: Vec<usize> = adj[&u].iter().copied().filter(|&x| x != s).collect();
                if u_others.len() != 2 {
                    continue;
                }
                let (a, b) = (u_others[0], u_others[1]);
                if adj[&a].contains(&b) {
                    continue;
                }

                if !states.contains_key(&(s, u))
                    || !states.contains_key(&(a, u))
                    || !states.contains_key(&(b, u))
                {
                    continue;
                }

                let top_cands = if !pre_spr.is_empty() {
                    if let Some((pa, pb, pcost, pwatched, pcands, ptop)) =
                        &pre_spr[idx - batch_start]
                    {
                        let clean_base = (*pa, *pb) == (a, b)
                            && !dirty_spr.contains(&(s, u))
                            && !dirty_spr.contains(&(a, u))
                            && !dirty_spr.contains(&(b, u));
                        if clean_base && *pcost <= min_step {
                            continue;
                        }
                        let clean_full = clean_base
                            && (dirty_nodes.is_empty()
                                || pwatched.iter().all(|nd| !dirty_nodes.contains(nd)))
                            && (dirty_spr.is_empty()
                                || pcands.iter().all(|&(x, y)| {
                                    !dirty_spr.contains(&(x, y)) && !dirty_spr.contains(&(y, x))
                                }));
                        if clean_full {
                            if ptop.is_empty() {
                                continue;
                            }
                            ptop.clone()
                        } else {
                            match eval_spr_prescreen(idx, adj, states) {
                                Some((_, _, bcost, _, _, rtop))
                                    if bcost > min_step && !rtop.is_empty() =>
                                {
                                    rtop
                                }
                                _ => continue,
                            }
                        }
                    } else {
                        match eval_spr_prescreen(idx, adj, states) {
                            Some((_, _, bcost, _, _, rtop))
                                if bcost > min_step && !rtop.is_empty() =>
                            {
                                rtop
                            }
                            _ => continue,
                        }
                    }
                } else {
                    match eval_spr_prescreen(idx, adj, states) {
                        Some((_, _, bcost, _, _, rtop)) if bcost > min_step && !rtop.is_empty() => {
                            rtop
                        }
                        _ => continue,
                    }
                };

                let ts = states[&(s, u)].clone();
                let ta = states.remove(&(a, u)).unwrap();
                let tb = states.remove(&(b, u)).unwrap();
                states.remove(&(u, a));
                states.remove(&(u, b));

                // Temporarily prune `u` from between `a` and `b` so `T \ s` states are 100% exact
                adj.get_mut(&a).unwrap().retain(|&z| z != u);
                adj.get_mut(&b).unwrap().retain(|&z| z != u);
                adj.get_mut(&a).unwrap().push(b);
                adj.get_mut(&b).unwrap().push(a);
                adj.get_mut(&u).unwrap().retain(|&z| z == s);

                states.insert((a, b), ta);
                states.insert((b, a), tb);
                dirty_spr.insert((a, b));
                dirty_spr.insert((b, a));
                Self::propagate_directed_fitch_outward(
                    adj,
                    states,
                    VecDeque::from([(a, b, 0), (b, a, 0)]),
                    n,
                    Some(&mut dirty_spr),
                );

                let exact_old_cost = Self::fitch_insert_cost_pair(
                    &ts.0,
                    &states[&(a, b)].0,
                    &states[&(b, a)].0,
                    ti_w,
                    tv_w,
                    site_weights,
                );

                let eval_edge_mdl_delta =
                    |ex: usize,
                     ey: usize,
                     adj_map: &BTreeMap<usize, Vec<usize>>,
                     st_map: &HashMap<(usize, usize), (Vec<u8>, f64)>|
                     -> f64 {
                        let (Some(t_ex), Some(t_ey)) =
                            (st_map.get(&(ex, ey)), st_map.get(&(ey, ex)))
                        else {
                            return f64::NEG_INFINITY;
                        };
                        let x_ch: Vec<(&[u8], f64)> = if ex >= n {
                            adj_map[&ex]
                                .iter()
                                .copied()
                                .filter(|&nb| nb != ey)
                                .filter_map(|nb| {
                                    st_map.get(&(nb, ex)).map(|st| (st.0.as_slice(), st.1))
                                })
                                .collect()
                        } else {
                            Vec::new()
                        };
                        let y_ch: Vec<(&[u8], f64)> = if ey >= n {
                            adj_map[&ey]
                                .iter()
                                .copied()
                                .filter(|&nb| nb != ex)
                                .filter_map(|nb| {
                                    st_map.get(&(nb, ey)).map(|st| (st.0.as_slice(), st.1))
                                })
                                .collect()
                        } else {
                            Vec::new()
                        };
                        let g_base = if x_ch.len() >= 2 && y_ch.len() >= 2 {
                            let raw = self
                                .compute_split_mdl_gain_from_child_refs(
                                    &x_ch,
                                    &y_ch,
                                    l,
                                    l_eff,
                                    ti_cost,
                                    tv_cost,
                                    site_weights,
                                )
                                .0;
                            (raw + split_topology_cost_nats(x_ch.len(), y_ch.len())).max(0.0)
                        } else {
                            0.0
                        };
                        let g_ux = if x_ch.len() >= 2 {
                            let raw = self
                                .compute_split_mdl_gain_from_child_refs(
                                    &[(ts.0.as_slice(), ts.1), (t_ey.0.as_slice(), t_ey.1)],
                                    &x_ch,
                                    l,
                                    l_eff,
                                    ti_cost,
                                    tv_cost,
                                    site_weights,
                                )
                                .0;
                            (raw + split_topology_cost_nats(2, x_ch.len())).max(0.0)
                        } else {
                            0.0
                        };
                        let g_uy = if y_ch.len() >= 2 {
                            let raw = self
                                .compute_split_mdl_gain_from_child_refs(
                                    &[(ts.0.as_slice(), ts.1), (t_ex.0.as_slice(), t_ex.1)],
                                    &y_ch,
                                    l,
                                    l_eff,
                                    ti_cost,
                                    tv_cost,
                                    site_weights,
                                )
                                .0;
                            (raw + split_topology_cost_nats(2, y_ch.len())).max(0.0)
                        } else {
                            0.0
                        };
                        g_ux + g_uy - g_base
                    };

                let old_mdl_delta = eval_edge_mdl_delta(a, b, adj, states);
                let mut best_edge = (a, b);
                let mut best_exact_cost = exact_old_cost - min_step;
                let mut best_mdl_delta = old_mdl_delta + 0.10;
                for (_, x, y) in top_cands {
                    if adj.get(&x).map(|v| v.contains(&y)).unwrap_or(false) {
                        if let (Some(tx), Some(ty)) = (states.get(&(x, y)), states.get(&(y, x))) {
                            let c_xy = Self::fitch_insert_cost_pair(
                                &ts.0,
                                &tx.0,
                                &ty.0,
                                ti_w,
                                tv_w,
                                site_weights,
                            );
                            if c_xy <= exact_old_cost - min_step {
                                let cand_mdl_delta = eval_edge_mdl_delta(x, y, adj, states);
                                if cand_mdl_delta > best_mdl_delta
                                    || ((cand_mdl_delta - best_mdl_delta).abs() <= 0.10
                                        && c_xy < best_exact_cost)
                                {
                                    best_mdl_delta = cand_mdl_delta;
                                    best_exact_cost = c_xy;
                                    best_edge = (x, y);
                                }
                            }
                        }
                    }
                }

                let (rx, ry) = best_edge;
                // Insert `u` onto `(rx, ry)` (either the improved target edge or restoring `(a, b)`)
                let tx = states.remove(&(rx, ry)).unwrap();
                let ty = states.remove(&(ry, rx)).unwrap();
                adj.get_mut(&rx).unwrap().retain(|&z| z != ry);
                adj.get_mut(&ry).unwrap().retain(|&z| z != rx);
                adj.get_mut(&rx).unwrap().push(u);
                adj.get_mut(&ry).unwrap().push(u);
                adj.get_mut(&u).unwrap().extend([rx, ry]);

                let st_urx = Self::combine_fitch_children_refs(&[
                    (ts.0.as_slice(), ts.1),
                    (ty.0.as_slice(), ty.1),
                ]);
                let st_ury = Self::combine_fitch_children_refs(&[
                    (ts.0.as_slice(), ts.1),
                    (tx.0.as_slice(), tx.1),
                ]);
                let st_us = Self::combine_fitch_children_refs(&[
                    (tx.0.as_slice(), tx.1),
                    (ty.0.as_slice(), ty.1),
                ]);
                states.insert((rx, u), tx);
                states.insert((ry, u), ty);
                states.insert((u, rx), st_urx);
                states.insert((u, ry), st_ury);
                states.insert((u, s), st_us);
                dirty_spr.insert((rx, u));
                dirty_spr.insert((ry, u));
                dirty_spr.insert((u, rx));
                dirty_spr.insert((u, ry));
                dirty_spr.insert((u, s));
                Self::propagate_directed_fitch_outward(
                    adj,
                    states,
                    VecDeque::from([(u, rx, 0), (u, ry, 0), (u, s, 0)]),
                    n,
                    Some(&mut dirty_spr),
                );

                if best_edge != (a, b) {
                    dirty_nodes.extend([a, b, rx, ry, u, s]);
                    any_moved = true;
                }
            }
        }
        any_moved
    }

    pub fn compute_all_directed_fitch_states(
        &self,
        adj: &BTreeMap<usize, Vec<usize>>,
        bitmasks: &[u8],
        n: usize,
        l: usize,
        terminal_loads: &[f64],
        _l_eff: f64,
    ) -> HashMap<(usize, usize), (Vec<u8>, f64)> {
        let root = adj
            .iter()
            .find(|(&u, nbs)| u >= n && !nbs.is_empty())
            .or_else(|| adj.iter().find(|(_, nbs)| !nbs.is_empty()))
            .map(|(&u, _)| u)
            .unwrap_or(0);

        let mut parent: HashMap<usize, Option<usize>> = HashMap::new();
        parent.insert(root, None);
        let mut levels: Vec<Vec<usize>> = vec![vec![root]];
        let mut total_nodes = 1usize;

        loop {
            let mut next_level = Vec::new();
            for &curr in levels.last().unwrap() {
                if let Some(nbs) = adj.get(&curr) {
                    for &nb in nbs {
                        if !parent.contains_key(&nb) {
                            parent.insert(nb, Some(curr));
                            next_level.push(nb);
                        }
                    }
                }
            }
            if next_level.is_empty() {
                break;
            }
            total_nodes += next_level.len();
            levels.push(next_level);
        }

        let mut states: HashMap<(usize, usize), (Vec<u8>, f64)> =
            HashMap::with_capacity(total_nodes * 2);

        // Level-parallel bottom-up pass (depth d only reads depth d + 1)
        for d in (1..levels.len()).rev() {
            let level = &levels[d];
            let compute_up = |&u: &usize| -> ((usize, usize), (Vec<u8>, f64)) {
                let p = parent[&u].unwrap();
                if u < n {
                    (
                        (u, p),
                        (bitmasks[u * l..(u + 1) * l].to_vec(), terminal_loads[u]),
                    )
                } else {
                    let child_refs: Vec<(&[u8], f64)> = adj[&u]
                        .iter()
                        .copied()
                        .filter(|&nb| nb != p)
                        .map(|c| {
                            let st = &states[&(c, u)];
                            (st.0.as_slice(), st.1)
                        })
                        .collect();
                    ((u, p), Self::combine_fitch_children_refs(&child_refs))
                }
            };

            #[cfg(not(target_arch = "wasm32"))]
            let up_batch: Vec<((usize, usize), (Vec<u8>, f64))> = if level.len() >= 8 {
                level.par_iter().map(compute_up).collect()
            } else {
                level.iter().map(compute_up).collect()
            };

            #[cfg(target_arch = "wasm32")]
            let up_batch: Vec<((usize, usize), (Vec<u8>, f64))> =
                level.iter().map(compute_up).collect();

            states.extend(up_batch);
        }

        // Level-parallel top-down pass (with O(d * L) leave-one-out hub acceleration for degree >= 4)
        for level in &levels {
            let compute_down = |&u: &usize| -> Vec<((usize, usize), (Vec<u8>, f64))> {
                let p_opt = parent[&u];
                let children: Vec<usize> = adj[&u]
                    .iter()
                    .copied()
                    .filter(|&nb| Some(nb) != p_opt)
                    .collect();
                if children.is_empty() {
                    return Vec::new();
                }
                if u < n {
                    let leaf_slice = &bitmasks[u * l..(u + 1) * l];
                    let tl = terminal_loads[u];
                    children
                        .into_iter()
                        .map(|c| ((u, c), (leaf_slice.to_vec(), tl)))
                        .collect()
                } else if adj[&u].len() >= 4 {
                    let all_nbs = &adj[&u];
                    let mut tot_inv = 0.0_f64;
                    let mut tot_counts = vec![[0_u16; 4]; l];
                    for &nb in all_nbs {
                        let (ref s_mask, m_c) = states[&(nb, u)];
                        tot_inv += 1.0 / m_c.max(0.4);
                        for s in 0..l {
                            let m = s_mask[s];
                            if m != 15 {
                                tot_counts[s][0] += (m & 1) as u16;
                                tot_counts[s][1] += ((m >> 1) & 1) as u16;
                                tot_counts[s][2] += ((m >> 2) & 1) as u16;
                                tot_counts[s][3] += ((m >> 3) & 1) as u16;
                            }
                        }
                    }
                    let build_hub_child = |&c: &usize| -> ((usize, usize), (Vec<u8>, f64)) {
                        let (ref s_c, m_c) = states[&(c, u)];
                        let inv_rem = (tot_inv - 1.0 / m_c.max(0.4)).max(1e-6);
                        let par_load = (1.0 / inv_rem).max(0.4);
                        let mut out_mask = vec![15_u8; l];
                        for s in 0..l {
                            let mc = s_c[s];
                            let mut c0 = tot_counts[s][0];
                            let mut c1 = tot_counts[s][1];
                            let mut c2 = tot_counts[s][2];
                            let mut c3 = tot_counts[s][3];
                            if mc != 15 {
                                c0 -= (mc & 1) as u16;
                                c1 -= ((mc >> 1) & 1) as u16;
                                c2 -= ((mc >> 2) & 1) as u16;
                                c3 -= ((mc >> 3) & 1) as u16;
                            }
                            let max_c = c0.max(c1).max(c2.max(c3));
                            if max_c > 0 {
                                let mut mask = 0_u8;
                                if c0 == max_c {
                                    mask |= 1;
                                }
                                if c1 == max_c {
                                    mask |= 2;
                                }
                                if c2 == max_c {
                                    mask |= 4;
                                }
                                if c3 == max_c {
                                    mask |= 8;
                                }
                                out_mask[s] = mask;
                            }
                        }
                        ((u, c), (out_mask, par_load))
                    };
                    #[cfg(not(target_arch = "wasm32"))]
                    if children.len() >= 16 {
                        return children.par_iter().map(build_hub_child).collect();
                    }
                    children.iter().map(build_hub_child).collect()
                } else {
                    children
                        .into_iter()
                        .map(|c| {
                            let other_refs: Vec<(&[u8], f64)> = adj[&u]
                                .iter()
                                .copied()
                                .filter(|&nb| nb != c)
                                .map(|nb| {
                                    let st = &states[&(nb, u)];
                                    (st.0.as_slice(), st.1)
                                })
                                .collect();
                            if other_refs.is_empty() {
                                ((u, c), (vec![15_u8; l], 1.0))
                            } else {
                                ((u, c), Self::combine_fitch_children_refs(&other_refs))
                            }
                        })
                        .collect()
                }
            };

            #[cfg(not(target_arch = "wasm32"))]
            let down_batches: Vec<Vec<((usize, usize), (Vec<u8>, f64))>> = if level.len() >= 4 {
                level.par_iter().map(compute_down).collect()
            } else {
                level.iter().map(compute_down).collect()
            };

            #[cfg(target_arch = "wasm32")]
            let down_batches: Vec<Vec<((usize, usize), (Vec<u8>, f64))>> =
                level.iter().map(compute_down).collect();

            for batch in down_batches {
                states.extend(batch);
            }
        }

        states
    }

    /// 64-bit SWAR SIMD Fitch combination processing 8 sites per u64 word (zero-clone slice API).
    #[inline]
    pub fn combine_fitch_children_refs(child_list: &[(&[u8], f64)]) -> (Vec<u8>, f64) {
        if child_list.len() == 1 {
            return (child_list[0].0.to_vec(), child_list[0].1);
        }
        let l = child_list[0].0.len();
        if child_list.len() == 2 {
            let (s1, m1) = child_list[0];
            let (s2, m2) = child_list[1];
            let mut out = vec![0_u8; l];
            let mut n_mut = 0usize;
            let chunks = l / 8;
            for c in 0..chunks {
                let off = c * 8;
                let w1 = u64::from_ne_bytes(s1[off..off + 8].try_into().unwrap());
                let w2 = u64::from_ne_bytes(s2[off..off + 8].try_into().unwrap());
                let inter = w1 & w2;
                let zero_hi = !(inter.wrapping_add(0x7F7F_7F7F_7F7F_7F7F)) & 0x8080_8080_8080_8080;
                if zero_hi == 0 {
                    out[off..off + 8].copy_from_slice(&inter.to_ne_bytes());
                } else {
                    n_mut += zero_hi.count_ones() as usize;
                    let zero_mask = (zero_hi >> 3) - (zero_hi >> 7);
                    let out_w = inter | ((w1 | w2) & zero_mask);
                    out[off..off + 8].copy_from_slice(&out_w.to_ne_bytes());
                }
            }
            for i in (chunks * 8)..l {
                let b1 = s1[i];
                let b2 = s2[i];
                let inter = b1 & b2;
                if inter == 0 {
                    out[i] = b1 | b2;
                    if b1 != 15 && b2 != 15 {
                        n_mut += 1;
                    }
                } else {
                    out[i] = inter;
                }
            }
            let par_load =
                1.0 / (1.0 / m1.max(0.4) + 1.0 / m2.max(0.4)) + 0.25 * (n_mut as f64);
            return (out, par_load.max(0.4));
        }

        let mut out = vec![15_u8; l];
        let mut inv_sum = 0.0_f64;
        for &(_, m_c) in child_list {
            inv_sum += 1.0 / m_c.max(0.4);
        }
        let chunks = l / 8;
        for c in 0..chunks {
            let off = c * 8;
            let mut w_and = !0_u64;
            for &(s_mask, _) in child_list {
                let w = u64::from_ne_bytes(s_mask[off..off + 8].try_into().unwrap());
                w_and &= w;
            }
            let zero_hi = !(w_and.wrapping_add(0x7F7F_7F7F_7F7F_7F7F)) & 0x8080_8080_8080_8080;
            if zero_hi == 0 {
                out[off..off + 8].copy_from_slice(&w_and.to_ne_bytes());
            } else {
                for s in off..(off + 8) {
                    let mut counts = [0_u16; 4];
                    for &(s_mask, _) in child_list {
                        let m = s_mask[s];
                        if m != 15 {
                            counts[0] += (m & 1) as u16;
                            counts[1] += ((m >> 1) & 1) as u16;
                            counts[2] += ((m >> 2) & 1) as u16;
                            counts[3] += ((m >> 3) & 1) as u16;
                        }
                    }
                    let max_c = counts[0].max(counts[1]).max(counts[2].max(counts[3]));
                    if max_c > 0 {
                        let mut mask = 0_u8;
                        for b in 0..4 {
                            if counts[b] == max_c {
                                mask |= 1 << b;
                            }
                        }
                        out[s] = mask;
                    }
                }
            }
        }
        for s in (chunks * 8)..l {
            let mut counts = [0_u16; 4];
            for &(s_mask, _) in child_list {
                let m = s_mask[s];
                if m != 15 {
                    counts[0] += (m & 1) as u16;
                    counts[1] += ((m >> 1) & 1) as u16;
                    counts[2] += ((m >> 2) & 1) as u16;
                    counts[3] += ((m >> 3) & 1) as u16;
                }
            }
            let max_c = counts[0].max(counts[1]).max(counts[2].max(counts[3]));
            if max_c > 0 {
                let mut mask = 0_u8;
                for b in 0..4 {
                    if counts[b] == max_c {
                        mask |= 1 << b;
                    }
                }
                out[s] = mask;
            }
        }
        let par_load = (1.0 / inv_sum.max(1e-6)).max(0.4);
        (out, par_load)
    }

    #[inline]
    pub fn combine_fitch_children(child_list: &[(Vec<u8>, f64)]) -> (Vec<u8>, f64) {
        if child_list.len() == 1 {
            return child_list[0].clone();
        }
        if child_list.len() == 2 {
            return Self::combine_fitch_children_refs(&[
                (child_list[0].0.as_slice(), child_list[0].1),
                (child_list[1].0.as_slice(), child_list[1].1),
            ]);
        }
        let refs: Vec<(&[u8], f64)> = child_list
            .iter()
            .map(|(s, m)| (s.as_slice(), *m))
            .collect();
        Self::combine_fitch_children_refs(&refs)
    }

    #[allow(clippy::too_many_arguments)]
    fn expand_polytomy_hubs_mdl(
        &self,
        adj: &mut BTreeMap<usize, Vec<usize>>,
        node_labels: &mut BTreeMap<usize, String>,
        bitmasks: &[u8],
        terminal_loads: &[f64],
        n: usize,
        l: usize,
        l_eff: f64,
        ti_cost: f64,
        tv_cost: f64,
        site_weights: &[f64],
    ) {
        let mut next_id = adj.keys().copied().max().unwrap_or(n) + 1;
        let mut states =
            self.compute_all_directed_fitch_states(adj, bitmasks, n, l, terminal_loads, l_eff);
        let hubs: Vec<usize> = adj
            .iter()
            .filter(|(&h, nbs)| h >= n && (4..=40).contains(&nbs.len()))
            .map(|(&h, _)| h)
            .collect();

        for h in hubs {
            let Some(nbs_ref) = adj.get(&h) else {
                continue;
            };
            if nbs_ref.len() < 4 {
                continue;
            };
            let nbs = nbs_ref.clone();
            let d = nbs.len();

            let chosen_splits: Vec<BTreeSet<usize>> = {
                let nb_refs: Vec<(&[u8], f64)> = nbs
                    .iter()
                    .map(|&c| {
                        let st = &states[&(c, h)];
                        (st.0.as_slice(), st.1)
                    })
                    .collect();

                let mut seed_weights: HashMap<Vec<usize>, f64> = HashMap::new();
                for s in 0..l {
                    let mut n_val = 0usize;
                    for &(s_mask, _) in &nb_refs {
                        if s_mask[s] != 15 {
                            n_val += 1;
                        }
                    }
                    if n_val < 4 {
                        continue;
                    }
                    for b in 0..4_u8 {
                        let mut idx_vec: Vec<usize> = (0..d)
                            .filter(|&i| {
                                let m = nb_refs[i].0[s];
                                m != 15 && ((m >> b) & 1) != 0
                            })
                            .collect();
                        let cnt = idx_vec.len();
                        if cnt >= 2 && cnt <= d - 2 {
                            if cnt > d / 2 || (cnt == d / 2 && !idx_vec.contains(&0)) {
                                idx_vec = (0..d).filter(|i| !idx_vec.contains(i)).collect();
                            }
                            if idx_vec.len() >= 2 && idx_vec.len() <= d - 2 {
                                *seed_weights.entry(idx_vec).or_insert(0.0) += site_weights[s];
                            }
                        }
                    }
                }

                if seed_weights.is_empty() {
                    Vec::new()
                } else {
                    let mut top_seeds: Vec<(Vec<usize>, f64)> = seed_weights.into_iter().collect();
                    top_seeds.sort_by(|a, b| {
                        b.1.partial_cmp(&a.1)
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then_with(|| a.0.cmp(&b.0))
                    });
                    top_seeds.truncate(25);

                    let mut positive_attractors: Vec<(f64, BTreeSet<usize>)> = Vec::new();
                    let mut seen_sets: HashSet<BTreeSet<usize>> = HashSet::new();

                    for (idx_vec, _) in top_seeds {
                        let mut u_set: BTreeSet<usize> = idx_vec.into_iter().collect();
                        let u_ch: Vec<(&[u8], f64)> =
                            u_set.iter().map(|&i| nb_refs[i]).collect();
                        let v_ch: Vec<(&[u8], f64)> = (0..d)
                            .filter(|j| !u_set.contains(j))
                            .map(|j| nb_refs[j])
                            .collect();
                        let (mut g_cur, _, sites_cur) = self.compute_split_mdl_gain_from_child_refs(
                            &u_ch,
                            &v_ch,
                            l,
                            l_eff,
                            ti_cost,
                            tv_cost,
                            site_weights,
                        );
                        if !sites_cur.is_empty() && d <= 20 {
                            for k_try in 0..d {
                                let mut cand_set = u_set.clone();
                                if !cand_set.remove(&k_try) {
                                    cand_set.insert(k_try);
                                }
                                if cand_set.len() < 2 || cand_set.len() > d - 2 {
                                    continue;
                                }
                                let u_try: Vec<(&[u8], f64)> =
                                    cand_set.iter().map(|&i| nb_refs[i]).collect();
                                let v_try: Vec<(&[u8], f64)> = (0..d)
                                    .filter(|j| !cand_set.contains(j))
                                    .map(|j| nb_refs[j])
                                    .collect();
                                let (g_try, _, _) = self.compute_split_mdl_gain_from_child_refs(
                                    &u_try,
                                    &v_try,
                                    l,
                                    l_eff,
                                    ti_cost,
                                    tv_cost,
                                    site_weights,
                                );
                                if g_try > g_cur + 0.25 {
                                    u_set = cand_set;
                                    g_cur = g_try;
                                }
                            }
                        }
                        if g_cur > 0.0 && seen_sets.insert(u_set.clone()) {
                            positive_attractors.push((g_cur, u_set));
                        }
                    }

                    positive_attractors
                        .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                    let mut c_splits: Vec<BTreeSet<usize>> = Vec::new();
                    let mut used_indices: BTreeSet<usize> = BTreeSet::new();
                    for (_, u_set) in positive_attractors {
                        if u_set.is_disjoint(&used_indices)
                            && (d - used_indices.len() - u_set.len()) >= 2
                        {
                            used_indices.extend(&u_set);
                            c_splits.push(u_set);
                        }
                    }
                    c_splits
                }
            };

            if chosen_splits.is_empty() {
                continue;
            }

            let mut new_nodes_created: Vec<usize> = Vec::with_capacity(chosen_splits.len());
            for u_set in chosen_splits {
                let new_node = next_id;
                next_id += 1;
                new_nodes_created.push(new_node);
                adj.insert(new_node, vec![h]);
                node_labels.insert(new_node, format!("N{new_node}"));
                adj.get_mut(&h).unwrap().push(new_node);
                for idx_c in u_set {
                    let c_node = nbs[idx_c];
                    if adj[&h].contains(&c_node) {
                        adj.get_mut(&h).unwrap().retain(|&x| x != c_node);
                        adj.get_mut(&c_node).unwrap().retain(|&x| x != h);
                        adj.get_mut(&new_node).unwrap().push(c_node);
                        adj.get_mut(&c_node).unwrap().push(new_node);
                        if let Some(st_ch) = states.remove(&(c_node, h)) {
                            states.insert((c_node, new_node), st_ch);
                        }
                        states.remove(&(h, c_node));
                    }
                }
                let ch_refs: Vec<(&[u8], f64)> = adj[&new_node]
                    .iter()
                    .copied()
                    .filter(|&x| x != h)
                    .map(|c_node| {
                        let st = &states[&(c_node, new_node)];
                        (st.0.as_slice(), st.1)
                    })
                    .collect();
                let st_new_h = Self::combine_fitch_children_refs(&ch_refs);
                states.insert((new_node, h), st_new_h);
            }

            let mut prop_q: VecDeque<(usize, usize)> = VecDeque::new();
            for &nb in &adj[&h] {
                prop_q.push_back((h, nb));
            }
            Self::propagate_directed_fitch_outward_exact(adj, &mut states, prop_q, n);
        }
    }

    /// Construct a strictly binary tree (`N - 3` internal splits) that is a guaranteed topological
    /// refinement of the MDL-certified tree `adj_mdl`.
    ///
    /// By Buneman's Split Compatibility Theorem, any split from `pre_step3_adj` refines a polytomy
    /// hub `h` of degree `d >= 4` in `adj_mdl` without conflicting with any certified split of
    /// `adj_mdl` if and only if its leaf set is a union of `k` (`2 <= k <= d - 2`) neighbor
    /// component leaf-sets of `h`. Distinct polytomy hubs are topologically independent and
    /// resolved in parallel across threads, with deterministic node-ID assignment.
    #[allow(clippy::too_many_arguments)]
    fn build_binary_refinement_from_mdl(
        &self,
        adj_mdl: &BTreeMap<usize, Vec<usize>>,
        pre_step3_adj: &BTreeMap<usize, Vec<usize>>,
        bitmasks: &[u8],
        terminal_loads: &[f64],
        n: usize,
        l: usize,
        l_eff: f64,
        ti_cost: f64,
        tv_cost: f64,
        site_weights: &[f64],
    ) -> BTreeMap<usize, Vec<usize>> {
        let hubs: Vec<usize> = adj_mdl
            .iter()
            .filter(|(&u, nbs)| u >= n && nbs.len() >= 4)
            .map(|(&u, _)| u)
            .collect();
        if n <= 3 || hubs.is_empty() {
            return adj_mdl.clone();
        }
        let n_words = (n + 63) / 64;

        let compute_sub_bits =
            |graph: &BTreeMap<usize, Vec<usize>>| -> (
                HashMap<usize, Option<usize>>,
                HashMap<usize, Vec<u64>>,
                Vec<u64>,
            ) {
                let root = graph
                    .iter()
                    .find(|(&u, nbs)| u >= n && !nbs.is_empty())
                    .or_else(|| graph.iter().find(|(_, nbs)| !nbs.is_empty()))
                    .map(|(&u, _)| u)
                    .unwrap_or(0);
                let mut parent: HashMap<usize, Option<usize>> = HashMap::new();
                parent.insert(root, None);
                let mut order = vec![root];
                let mut q_idx = 0usize;
                while q_idx < order.len() {
                    let curr = order[q_idx];
                    q_idx += 1;
                    if let Some(nbs) = graph.get(&curr) {
                        for &nb in nbs {
                            if !parent.contains_key(&nb) {
                                parent.insert(nb, Some(curr));
                                order.push(nb);
                            }
                        }
                    }
                }
                let mut sub_bits: HashMap<usize, Vec<u64>> = HashMap::with_capacity(order.len());
                for &u in order.iter().rev() {
                    let mut bits = vec![0_u64; n_words];
                    if u < n {
                        bits[u >> 6] |= 1_u64 << (u & 63);
                    }
                    if let Some(nbs) = graph.get(&u) {
                        let p_opt = parent[&u];
                        for &nb in nbs {
                            if Some(nb) != p_opt {
                                if let Some(cb) = sub_bits.get(&nb) {
                                    for w in 0..n_words {
                                        bits[w] |= cb[w];
                                    }
                                }
                            }
                        }
                    }
                    sub_bits.insert(u, bits);
                }
                let all_bits = sub_bits
                    .get(&root)
                    .cloned()
                    .unwrap_or_else(|| vec![0_u64; n_words]);
                (parent, sub_bits, all_bits)
            };

        let edge_bits = |start: usize,
                         blocked: usize,
                         parent: &HashMap<usize, Option<usize>>,
                         sub_bits: &HashMap<usize, Vec<u64>>,
                         all_bits: &[u64]|
         -> Vec<u64> {
            if parent.get(&start) == Some(&Some(blocked)) {
                sub_bits[&start].clone()
            } else if parent.get(&blocked) == Some(&Some(start)) {
                let sb = &sub_bits[&blocked];
                (0..n_words).map(|w| all_bits[w] ^ sb[w]).collect()
            } else {
                vec![0_u64; n_words]
            }
        };

        let (pre_parent, pre_sub, pre_all) = compute_sub_bits(pre_step3_adj);
        let mut pre_splits: Vec<Vec<u64>> = Vec::new();
        for (&u, nbs) in pre_step3_adj {
            if u >= n {
                for &v in nbs {
                    if u < v && v >= n {
                        let b = edge_bits(u, v, &pre_parent, &pre_sub, &pre_all);
                        let pop: u32 = b.iter().map(|w| w.count_ones()).sum();
                        if (pop as usize) >= 2 && (pop as usize) <= n - 2 {
                            pre_splits.push(b);
                        }
                    }
                }
            }
        }

        let (mdl_parent, mdl_sub, mdl_all) = compute_sub_bits(adj_mdl);
        let states = self.compute_all_directed_fitch_states(
            adj_mdl,
            bitmasks,
            n,
            l,
            terminal_loads,
            l_eff,
        );

        let resolve_single_hub = |&h: &usize| -> (usize, Vec<usize>, Vec<(usize, usize)>) {
            let nbs = &adj_mdl[&h];
            let d = nbs.len();
            let d_words = (d + 63) / 64;
            let comp_masks: Vec<Vec<u64>> = nbs
                .iter()
                .map(|&c| edge_bits(c, h, &mdl_parent, &mdl_sub, &mdl_all))
                .collect();
            let rep_leaf: Vec<Option<usize>> = comp_masks
                .iter()
                .map(|cm| {
                    for (w, &word) in cm.iter().enumerate() {
                        if word != 0 {
                            return Some(w * 64 + word.trailing_zeros() as usize);
                        }
                    }
                    None
                })
                .collect();

            let mut hub_splits: Vec<Vec<u64>> = Vec::new();
            for sp in &pre_splits {
                let mut cnt_rep = 0usize;
                for &rl_opt in &rep_leaf {
                    if let Some(lf) = rl_opt {
                        if ((sp[lf >> 6] >> (lf & 63)) & 1) != 0 {
                            cnt_rep += 1;
                        }
                    }
                }
                if cnt_rep < 2 || cnt_rep > d - 2 {
                    continue;
                }
                let mut pm = vec![0_u64; d_words];
                let mut u_cnt = 0usize;
                let mut clean = true;
                for (r, cm) in comp_masks.iter().enumerate() {
                    let in_sp = rep_leaf[r]
                        .map(|lf| ((sp[lf >> 6] >> (lf & 63)) & 1) != 0)
                        .unwrap_or(false);
                    if in_sp {
                        let mut ok = true;
                        for w in 0..n_words {
                            if (cm[w] & !sp[w]) != 0 {
                                ok = false;
                                break;
                            }
                        }
                        if !ok {
                            clean = false;
                            break;
                        }
                        pm[r >> 6] |= 1_u64 << (r & 63);
                        u_cnt += 1;
                    } else {
                        let mut ok = true;
                        for w in 0..n_words {
                            if (cm[w] & sp[w]) != 0 {
                                ok = false;
                                break;
                            }
                        }
                        if !ok {
                            clean = false;
                            break;
                        }
                    }
                }
                if clean && u_cnt >= 2 && u_cnt <= d - 2 {
                    hub_splits.push(pm);
                }
            }

            // Local graph: 0..d are external ports, d is hub center `h`, >d are new internal nodes
            let mut local_adj: Vec<Vec<usize>> = Vec::with_capacity(2 * d);
            for _ in 0..d {
                local_adj.push(vec![d]);
            }
            local_adj.push((0..d).collect());

            let mut item_pm: Vec<Vec<u64>> = (0..=d)
                .map(|idx| {
                    let mut v = vec![0_u64; d_words];
                    if idx < d {
                        v[idx >> 6] |= 1_u64 << (idx & 63);
                    }
                    v
                })
                .collect();

            let mut in_states: Vec<(Vec<u8>, f64)> = Vec::with_capacity(2 * d);
            for &c in nbs {
                in_states.push(
                    states
                        .get(&(c, h))
                        .cloned()
                        .unwrap_or_else(|| (vec![15_u8; l], 1.0)),
                );
            }
            in_states.push((vec![15_u8; l], 1.0)); // dummy slot for center `d`

            // 1. Apply all compatible pre-step3 splits locally
            while local_adj[d].len() >= 4 {
                let cur_nbs = local_adj[d].clone();
                let cur_d = cur_nbs.len();
                let mut matched: Option<Vec<usize>> = None;
                for sp_pm in &hub_splits {
                    let mut u_set = Vec::new();
                    let mut clean = true;
                    for &item in &cur_nbs {
                        let pm = &item_pm[item];
                        let mut has_in = false;
                        let mut has_out = false;
                        for w in 0..d_words {
                            if (pm[w] & sp_pm[w]) != 0 {
                                has_in = true;
                            }
                            if (pm[w] & !sp_pm[w]) != 0 {
                                has_out = true;
                            }
                        }
                        if has_in && has_out {
                            clean = false;
                            break;
                        }
                        if has_in {
                            u_set.push(item);
                        }
                    }
                    if clean && u_set.len() >= 2 && u_set.len() <= cur_d - 2 {
                        matched = Some(u_set);
                        break;
                    }
                }
                let Some(chosen_items) = matched else {
                    break;
                };
                let w_loc = local_adj.len();
                local_adj.push(vec![d]);
                local_adj[d].push(w_loc);
                let mut pm_w = vec![0_u64; d_words];
                for &it in &chosen_items {
                    for w in 0..d_words {
                        pm_w[w] |= item_pm[it][w];
                    }
                    local_adj[d].retain(|&x| x != it);
                    local_adj[it].retain(|&x| x != d);
                    local_adj[w_loc].push(it);
                    local_adj[it].push(w_loc);
                }
                let ch_refs: Vec<(&[u8], f64)> = chosen_items
                    .iter()
                    .map(|&it| (in_states[it].0.as_slice(), in_states[it].1))
                    .collect();
                let st_w = Self::combine_fitch_children_refs(&ch_refs);
                item_pm.push(pm_w);
                in_states.push(st_w);
            }

            // 2. If remaining degree > 4, reduce down to 4 via SWAR-accelerated Neighbor-Joining
            if local_adj[d].len() > 4 {
                let mut nb_items: Vec<usize> = local_adj[d].clone();
                let rem_d = nb_items.len();
                let cap = 2 * rem_d + 2;
                let mut d_mat = vec![0.0_f64; cap * cap];
                let mut row_sums = vec![0.0_f64; cap];
                let chunks = l / 8;
                for a in 0..rem_d {
                    let s1 = &in_states[nb_items[a]].0;
                    for b in (a + 1)..rem_d {
                        let s2 = &in_states[nb_items[b]].0;
                        let mut w_diff = 0.0_f64;
                        for c_idx in 0..chunks {
                            let off = c_idx * 8;
                            let w1 = u64::from_ne_bytes(s1[off..off + 8].try_into().unwrap());
                            let w2 = u64::from_ne_bytes(s2[off..off + 8].try_into().unwrap());
                            let inter = w1 & w2;
                            let zero_hi = !(inter.wrapping_add(0x7F7F_7F7F_7F7F_7F7F))
                                & 0x8080_8080_8080_8080;
                            if zero_hi != 0 {
                                for s in off..(off + 8) {
                                    let b1 = s1[s];
                                    let b2 = s2[s];
                                    if b1 != 15 && b2 != 15 && (b1 & b2) == 0 {
                                        w_diff += site_weights[s];
                                    }
                                }
                            }
                        }
                        for s in (chunks * 8)..l {
                            let b1 = s1[s];
                            let b2 = s2[s];
                            if b1 != 15 && b2 != 15 && (b1 & b2) == 0 {
                                w_diff += site_weights[s];
                            }
                        }
                        d_mat[a * cap + b] = w_diff;
                        d_mat[b * cap + a] = w_diff;
                        row_sums[a] += w_diff;
                        row_sums[b] += w_diff;
                    }
                }
                let mut active: Vec<usize> = (0..rem_d).collect();
                while active.len() > 4 {
                    let k = active.len();
                    let k_m2 = (k as f64) - 2.0;
                    let mut best_q = f64::INFINITY;
                    let mut best_pair = (0usize, 1usize);
                    for a in 0..k {
                        let la = active[a];
                        let ra = row_sums[la];
                        for b in (a + 1)..k {
                            let lb = active[b];
                            let q = k_m2 * d_mat[la * cap + lb] - ra - row_sums[lb];
                            if q < best_q {
                                best_q = q;
                                best_pair = (a, b);
                            }
                        }
                    }
                    let lu = active[best_pair.0];
                    let lv = active[best_pair.1];
                    let iu = nb_items[lu];
                    let iv = nb_items[lv];
                    let w_loc = local_adj.len();
                    let lw = nb_items.len();
                    nb_items.push(w_loc);

                    let st_w = Self::combine_fitch_children_refs(&[
                        (in_states[iu].0.as_slice(), in_states[iu].1),
                        (in_states[iv].0.as_slice(), in_states[iv].1),
                    ]);
                    in_states.push(st_w);
                    item_pm.push(vec![0_u64; d_words]);

                    local_adj.push(vec![d, iu, iv]);
                    local_adj[d].retain(|&x| x != iu && x != iv);
                    local_adj[d].push(w_loc);
                    local_adj[iu].retain(|&x| x != d);
                    local_adj[iu].push(w_loc);
                    local_adj[iv].retain(|&x| x != d);
                    local_adj[iv].push(w_loc);

                    let d_uv = d_mat[lu * cap + lv];
                    let limb_u =
                        (0.5 * d_uv + (row_sums[lu] - row_sums[lv]) / (2.0 * k_m2)).max(0.0);
                    let limb_v = (d_uv - limb_u).max(0.0);
                    active.retain(|&x| x != lu && x != lv);
                    let mut rw = 0.0_f64;
                    for &lx in &active {
                        let d_ux = d_mat[lu * cap + lx];
                        let d_vx = d_mat[lv * cap + lx];
                        let d_wx = (0.5 * (d_ux - limb_u + d_vx - limb_v)).max(0.0);
                        d_mat[lw * cap + lx] = d_wx;
                        d_mat[lx * cap + lw] = d_wx;
                        row_sums[lx] += d_wx - d_ux - d_vx;
                        rw += d_wx;
                    }
                    row_sums[lw] = rw;
                    active.push(lw);
                }
            }

            // 3. Exact 3-quartet MDL resolution if remaining degree == 4
            if local_adj[d].len() == 4 {
                let nbs4 = local_adj[d].clone();
                let r4: Vec<(&[u8], f64)> = nbs4
                    .iter()
                    .map(|&it| (in_states[it].0.as_slice(), in_states[it].1))
                    .collect();
                let pairings = [([0usize, 1], [2usize, 3]), ([0, 2], [1, 3]), ([0, 3], [1, 2])];
                let mut best_p = [0usize, 1];
                let mut best_g = f64::NEG_INFINITY;
                for (p1, p2) in pairings {
                    let (g, _, _) = self.compute_split_mdl_gain_from_child_refs(
                        &[r4[p1[0]], r4[p1[1]]],
                        &[r4[p2[0]], r4[p2[1]]],
                        l,
                        l_eff,
                        ti_cost,
                        tv_cost,
                        site_weights,
                    );
                    if g > best_g {
                        best_g = g;
                        best_p = p1;
                    }
                }
                let w_loc = local_adj.len();
                let c0 = nbs4[best_p[0]];
                let c1 = nbs4[best_p[1]];
                local_adj.push(vec![d, c0, c1]);
                local_adj[d].retain(|&x| x != c0 && x != c1);
                local_adj[d].push(w_loc);
                local_adj[c0].retain(|&x| x != d);
                local_adj[c0].push(w_loc);
                local_adj[c1].retain(|&x| x != d);
                local_adj[c1].push(w_loc);
            }

            let k_new = local_adj.len() - (d + 1);
            let port_owner: Vec<usize> = (0..d).map(|r| local_adj[r][0] - d).collect();
            let mut local_int_edges: Vec<(usize, usize)> = Vec::with_capacity(k_new);
            for a in d..local_adj.len() {
                for &b in &local_adj[a] {
                    if b >= d && a < b {
                        local_int_edges.push((a - d, b - d));
                    }
                }
            }
            (k_new, port_owner, local_int_edges)
        };

        #[cfg(not(target_arch = "wasm32"))]
        let hub_resolutions: Vec<(usize, Vec<usize>, Vec<(usize, usize)>)> =
            hubs.par_iter().map(resolve_single_hub).collect();

        #[cfg(target_arch = "wasm32")]
        let hub_resolutions: Vec<(usize, Vec<usize>, Vec<(usize, usize)>)> =
            hubs.iter().map(resolve_single_hub).collect();

        // Deterministically assign global node IDs and splice all local hub resolutions into `bin_adj`
        let mut next_id = adj_mdl.keys().copied().max().unwrap_or(n) + 1;
        let mut bin_adj: BTreeMap<usize, Vec<usize>> =
            adj_mdl.keys().map(|&u| (u, Vec::new())).collect();
        let mut end_owner: HashMap<(usize, usize), usize> = HashMap::new();

        for (&h, (k_new, port_owner, local_int_edges)) in hubs.iter().zip(&hub_resolutions) {
            let base_id = next_id;
            next_id += *k_new;
            for j in 0..*k_new {
                bin_adj.insert(base_id + j, Vec::with_capacity(3));
            }
            let map_loc = |loc: usize| -> usize {
                if loc == 0 {
                    h
                } else {
                    base_id + loc - 1
                }
            };
            let nbs = &adj_mdl[&h];
            for (r, &nb) in nbs.iter().enumerate() {
                end_owner.insert((h, nb), map_loc(port_owner[r]));
            }
            for &(lu, lv) in local_int_edges {
                let gu = map_loc(lu);
                let gv = map_loc(lv);
                bin_adj.get_mut(&gu).unwrap().push(gv);
                bin_adj.get_mut(&gv).unwrap().push(gu);
            }
        }

        for (&u, nbs) in adj_mdl {
            for &v in nbs {
                if u < v {
                    let ou = end_owner.get(&(u, v)).copied().unwrap_or(u);
                    let ov = end_owner.get(&(v, u)).copied().unwrap_or(v);
                    bin_adj.get_mut(&ou).unwrap().push(ov);
                    bin_adj.get_mut(&ov).unwrap().push(ou);
                }
            }
        }
        for nbs in bin_adj.values_mut() {
            nbs.sort_unstable();
        }
        bin_adj
    }

    #[allow(clippy::too_many_arguments)]
    fn contract_unsupported_edges_mdl(
        &self,
        mut adj: BTreeMap<usize, Vec<usize>>,
        bitmasks: &[u8],
        majority_state: &[i8],
        terminal_loads: &[f64],
        n: usize,
        l: usize,
        l_eff: f64,
        ti_cost: f64,
        tv_cost: f64,
        site_weights: &[f64],
        node_labels: &mut BTreeMap<usize, String>,
        expand_hubs: bool,
        gtr_edge_stats: Option<&(HashMap<(usize, usize), (f64, f64)>, f64)>,
        need_fitch_muts: bool,
    ) -> (
        BTreeMap<usize, Vec<usize>>,
        Vec<BranchRecord>,
        HashMap<usize, Vec<i8>>,
        HashMap<(usize, usize), Vec<(usize, i8, i8)>>,
    ) {
        if let Some((init_stats, emp_scale)) = gtr_edge_stats {
            let mut cur_stats = init_stats.clone();
            let block_w = 512usize;
            let n_blocks = (l + block_w - 1) / block_w;
            let count_var_block = |b_idx: usize| -> usize {
                let cs = b_idx * block_w;
                let ce = (cs + block_w).min(l);
                let cw = ce - cs;
                let mut u_masks = [0_u8; 512];
                for i in 0..n {
                    let row = &bitmasks[i * l + cs..i * l + ce];
                    for k in 0..cw {
                        let m = row[k] & 15;
                        if m != 15 {
                            u_masks[k] |= m;
                        }
                    }
                }
                (0..cw).filter(|&k| u_masks[k].count_ones() >= 2).count()
            };
            #[cfg(not(target_arch = "wasm32"))]
            let m_var: usize = (0..n_blocks).into_par_iter().map(count_var_block).sum();
            #[cfg(target_arch = "wasm32")]
            let m_var: usize = (0..n_blocks).map(count_var_block).sum();

            let fisher_n = ((m_var as f64) / std::f64::consts::TAU).max(1.0);
            let mult_pen = if self.no_mult_pen {
                0.0
            } else {
                0.5 * ((n.saturating_sub(3)).max(1) as f64).ln()
            };
            let base_pen = 0.5 * fisher_n.ln() + mult_pen;
            let min_bl = (0.20 / l_eff.max(1.0)).clamp(1e-6, 1e-3);

            loop {
                let mut bad_edges: Vec<(f64, usize, usize)> = Vec::new();
                for (&u, nbs) in &adj {
                    if u >= n && nbs.len() >= 3 {
                        let deg_u = nbs.len().saturating_sub(1);
                        for &v in nbs {
                            if u < v {
                                if let Some(v_nbs) = adj.get(&v) {
                                    if v >= n && v_nbs.len() >= 3 {
                                        let deg_v = v_nbs.len().saturating_sub(1);
                                        let (dll, _) =
                                            cur_stats.get(&(u, v)).copied().unwrap_or((0.0, min_bl));
                                        let rissanen_pen =
                                            base_pen + split_topology_cost_nats(deg_u, deg_v);
                                        let gain = dll - rissanen_pen;
                                        if gain <= 0.0 {
                                            bad_edges.push((gain, u, v));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                if bad_edges.is_empty() {
                    break;
                }
                bad_edges.sort_by(|a, b| {
                    a.0.partial_cmp(&b.0)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.1.cmp(&b.1))
                        .then_with(|| a.2.cmp(&b.2))
                });
                let mut locked_nodes: HashSet<usize> = HashSet::new();
                let mut alias: HashMap<usize, usize> = HashMap::new();
                let mut contracted_any = false;

                for (_gain, raw_u, raw_v) in bad_edges {
                    let mut u = raw_u;
                    while let Some(&au) = alias.get(&u) {
                        u = au;
                    }
                    let mut v = raw_v;
                    while let Some(&av) = alias.get(&v) {
                        v = av;
                    }
                    if u == v || locked_nodes.contains(&u) || locked_nodes.contains(&v) {
                        continue;
                    }
                    if !adj.get(&u).map(|x| x.contains(&v)).unwrap_or(false) {
                        continue;
                    }
                    cur_stats.remove(&(u.min(v), u.max(v)));
                    adj.get_mut(&u).unwrap().retain(|&x| x != v);
                    adj.get_mut(&v).unwrap().retain(|&x| x != u);
                    let v_nbs = adj[&v].clone();
                    for nb in v_nbs {
                        let old_stat = cur_stats.remove(&(v.min(nb), v.max(nb)));
                        adj.get_mut(&nb).unwrap().retain(|&x| x != v);
                        if nb != u && !adj[&u].contains(&nb) {
                            adj.get_mut(&u).unwrap().push(nb);
                            adj.get_mut(&nb).unwrap().push(u);
                            if let Some(st_val) = old_stat {
                                cur_stats.insert((u.min(nb), u.max(nb)), st_val);
                            }
                        }
                    }
                    adj.get_mut(&v).unwrap().clear();
                    alias.insert(v, u);
                    locked_nodes.insert(u);
                    locked_nodes.insert(v);
                    contracted_any = true;
                }
                if !contracted_any {
                    break;
                }
            }
            adj.retain(|&u, nbs| u < n || !nbs.is_empty());

            let (node_states, edge_mutations) = if need_fitch_muts {
                let states = self.compute_all_directed_fitch_states(
                    &adj,
                    bitmasks,
                    n,
                    l,
                    terminal_loads,
                    l_eff,
                );
                self.reconstruct_nodal_states_and_mutations(
                    &adj,
                    bitmasks,
                    majority_state,
                    &states,
                    n,
                    l,
                )
            } else {
                (HashMap::new(), HashMap::new())
            };

            let mut edge_pairs: Vec<(usize, usize)> = Vec::new();
            for (&u, nbs) in &adj {
                let mut sorted_nbs = nbs.clone();
                sorted_nbs.sort_unstable();
                for v in sorted_nbs {
                    if u < v {
                        edge_pairs.push((u, v));
                    }
                }
            }

            let branches: Vec<BranchRecord> = edge_pairs
                .into_iter()
                .map(|(u, v)| {
                    let is_int = u >= n && v >= n;
                    let (dll, tmle) = cur_stats.get(&(u, v)).copied().unwrap_or((0.0, min_bl));
                    let gain = if is_int {
                        let deg_u = adj[&u].len().saturating_sub(1);
                        let deg_v = adj[&v].len().saturating_sub(1);
                        dll - (base_pen + split_topology_cost_nats(deg_u, deg_v))
                    } else {
                        dll
                    };
                    let syn_sites: Vec<usize> = edge_mutations
                        .get(&(u, v))
                        .map(|ms| ms.iter().map(|x| x.0).collect())
                        .unwrap_or_default();
                    let m_cnt = edge_mutations
                        .get(&(u, v))
                        .map(|m| m.len() as f64)
                        .unwrap_or(0.0);
                    let eff_mut = (tmle * (*emp_scale) * l_eff).max(m_cnt);
                    let support = 1.0 / (1.0 + (-gain.clamp(-50.0, 50.0)).exp());
                    BranchRecord {
                        u,
                        v,
                        is_internal: is_int,
                        mutation_count: eff_mut,
                        branch_length: tmle.max(min_bl),
                        mdl_gain_nats: gain,
                        mdl_support: support,
                        synapomorphy_sites: syn_sites,
                    }
                })
                .collect();

            return (adj, branches, node_states, edge_mutations);
        }
        let passes = if expand_hubs { 2 } else { 1 };
        for outer_pass in 0..passes {
            loop {
                let states = self.compute_all_directed_fitch_states(
                    &adj,
                    bitmasks,
                    n,
                    l,
                    terminal_loads,
                    l_eff,
                );
                let mut internal_edges = Vec::new();
                for (&u, nbs) in &adj {
                    if u >= n && nbs.len() >= 3 {
                        for &v in nbs {
                            if u < v && v >= n && adj.get(&v).map(|x| x.len()).unwrap_or(0) >= 3 {
                                internal_edges.push((u, v));
                            }
                        }
                    }
                }
                if internal_edges.is_empty() {
                    break;
                }

                #[cfg(not(target_arch = "wasm32"))]
                let mut bad_edges: Vec<(f64, usize, usize)> = internal_edges
                    .into_par_iter()
                    .filter_map(|(u, v)| {
                        let (gain, _, _) = self.evaluate_internal_edge_mdl_gain(
                            u,
                            v,
                            &adj,
                            &states,
                            l,
                            l_eff,
                            ti_cost,
                            tv_cost,
                            site_weights,
                        );
                        if gain <= 0.0 {
                            Some((gain, u, v))
                        } else {
                            None
                        }
                    })
                    .collect();

                #[cfg(target_arch = "wasm32")]
                let mut bad_edges: Vec<(f64, usize, usize)> = internal_edges
                    .into_iter()
                    .filter_map(|(u, v)| {
                        let (gain, _, _) = self.evaluate_internal_edge_mdl_gain(
                            u,
                            v,
                            &adj,
                            &states,
                            l,
                            l_eff,
                            ti_cost,
                            tv_cost,
                            site_weights,
                        );
                        if gain <= 0.0 {
                            Some((gain, u, v))
                        } else {
                            None
                        }
                    })
                    .collect();

                if bad_edges.is_empty() {
                    break;
                }

                bad_edges.sort_by(|a, b| {
                    a.0.partial_cmp(&b.0)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.1.cmp(&b.1))
                        .then_with(|| a.2.cmp(&b.2))
                });
                let mut locked_nodes: HashSet<usize> = HashSet::new();
                let mut alias: HashMap<usize, usize> = HashMap::new();
                let mut contracted_any = false;

                for (_gain, raw_u, raw_v) in bad_edges {
                    let mut u = raw_u;
                    while let Some(&au) = alias.get(&u) {
                        u = au;
                    }
                    let mut v = raw_v;
                    while let Some(&av) = alias.get(&v) {
                        v = av;
                    }
                    if u == v {
                        continue;
                    }
                    if locked_nodes.contains(&u) || locked_nodes.contains(&v) {
                        continue;
                    }
                    if !adj.get(&u).map(|x| x.contains(&v)).unwrap_or(false) {
                        continue;
                    }
                    adj.get_mut(&u).unwrap().retain(|&x| x != v);
                    adj.get_mut(&v).unwrap().retain(|&x| x != u);
                    let v_nbs = adj[&v].clone();
                    for nb in v_nbs {
                        adj.get_mut(&nb).unwrap().retain(|&x| x != v);
                        if nb != u && !adj[&u].contains(&nb) {
                            adj.get_mut(&u).unwrap().push(nb);
                            adj.get_mut(&nb).unwrap().push(u);
                        }
                    }
                    adj.get_mut(&v).unwrap().clear();
                    alias.insert(v, u);
                    locked_nodes.insert(u);
                    locked_nodes.insert(v);
                    contracted_any = true;
                }
                if !contracted_any {
                    break;
                }
            }

            if expand_hubs && outer_pass == 0 && n >= 5 {
                self.expand_polytomy_hubs_mdl(
                    &mut adj,
                    node_labels,
                    bitmasks,
                    terminal_loads,
                    n,
                    l,
                    l_eff,
                    ti_cost,
                    tv_cost,
                    site_weights,
                );
            }
        }

        let states =
            self.compute_all_directed_fitch_states(&adj, bitmasks, n, l, terminal_loads, l_eff);
        let (node_states, edge_mutations) = self.reconstruct_nodal_states_and_mutations(
            &adj,
            bitmasks,
            majority_state,
            &states,
            n,
            l,
        );

        let mut edge_pairs: Vec<(usize, usize)> = Vec::new();
        for (&u, nbs) in &adj {
            let mut sorted_nbs = nbs.clone();
            sorted_nbs.sort_unstable();
            for v in sorted_nbs {
                if u < v {
                    edge_pairs.push((u, v));
                }
            }
        }

        let build_branch_rec = |&(u, v): &(usize, usize)| -> BranchRecord {
            let is_int = u >= n && v >= n;
            let m_count = edge_mutations
                .get(&(u, v))
                .map(|m| m.len() as f64)
                .unwrap_or(0.0);
            let (gain, eff_mut, syn_sites) = if is_int {
                let (g, m_syn, sites) = self.evaluate_internal_edge_mdl_gain(
                    u,
                    v,
                    &adj,
                    &states,
                    l,
                    l_eff,
                    ti_cost,
                    tv_cost,
                    site_weights,
                );
                (g, m_count.max(m_syn), sites)
            } else {
                (m_count * 2.0, m_count, Vec::new())
            };
            let support = 1.0 / (1.0 + (-gain.clamp(-50.0, 50.0)).exp());
            BranchRecord {
                u,
                v,
                is_internal: is_int,
                mutation_count: eff_mut,
                branch_length: eff_mut / l_eff,
                mdl_gain_nats: gain,
                mdl_support: support,
                synapomorphy_sites: syn_sites,
            }
        };

        #[cfg(not(target_arch = "wasm32"))]
        let branches: Vec<BranchRecord> = edge_pairs.par_iter().map(build_branch_rec).collect();

        #[cfg(target_arch = "wasm32")]
        let branches: Vec<BranchRecord> = edge_pairs.iter().map(build_branch_rec).collect();

        (adj, branches, node_states, edge_mutations)
    }

    #[allow(clippy::too_many_arguments)]
    fn evaluate_internal_edge_mdl_gain(
        &self,
        u: usize,
        v: usize,
        adj: &BTreeMap<usize, Vec<usize>>,
        states: &HashMap<(usize, usize), (Vec<u8>, f64)>,
        l: usize,
        l_eff: f64,
        ti_cost: f64,
        tv_cost: f64,
        site_weights: &[f64],
    ) -> (f64, f64, Vec<usize>) {
        let mut u_sub: Vec<usize> = adj[&u].iter().copied().filter(|&nb| nb != v).collect();
        let mut v_sub: Vec<usize> = adj[&v].iter().copied().filter(|&nb| nb != u).collect();
        let deg_u = u_sub.len();
        let deg_v = v_sub.len();
        if deg_u < 2 || deg_v < 2 {
            return (-10.0, 0.0, Vec::new());
        }
        if u_sub.len() > 16 {
            u_sub.sort_by(|&a, &b| {
                states[&(a, u)]
                    .1
                    .partial_cmp(&states[&(b, u)].1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            u_sub.truncate(12);
        }
        if v_sub.len() > 16 {
            v_sub.sort_by(|&a, &b| {
                states[&(a, v)]
                    .1
                    .partial_cmp(&states[&(b, v)].1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            v_sub.truncate(12);
        }
        let u_refs: Vec<(&[u8], f64)> = u_sub
            .iter()
            .map(|&nb| {
                let st = &states[&(nb, u)];
                (st.0.as_slice(), st.1)
            })
            .collect();
        let v_refs: Vec<(&[u8], f64)> = v_sub
            .iter()
            .map(|&nb| {
                let st = &states[&(nb, v)];
                (st.0.as_slice(), st.1)
            })
            .collect();
        let (gain_sub, m_opt, sites) = self.compute_split_mdl_gain_from_child_refs(
            &u_refs,
            &v_refs,
            l,
            l_eff,
            ti_cost,
            tv_cost,
            site_weights,
        );
        let extra_topo = split_topology_cost_nats(deg_u, deg_v)
            - split_topology_cost_nats(u_refs.len(), v_refs.len());
        (gain_sub - extra_topo, m_opt, sites)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn compute_split_mdl_gain_from_children(
        &self,
        u_children: &[(Vec<u8>, f64)],
        v_children: &[(Vec<u8>, f64)],
        l: usize,
        l_eff: f64,
        ti_cost: f64,
        tv_cost: f64,
        site_weights: &[f64],
    ) -> (f64, f64, Vec<usize>) {
        let u_refs: Vec<(&[u8], f64)> = u_children
            .iter()
            .map(|(s, m)| (s.as_slice(), *m))
            .collect();
        let v_refs: Vec<(&[u8], f64)> = v_children
            .iter()
            .map(|(s, m)| (s.as_slice(), *m))
            .collect();
        self.compute_split_mdl_gain_from_child_refs(
            &u_refs,
            &v_refs,
            l,
            l_eff,
            ti_cost,
            tv_cost,
            site_weights,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn compute_split_mdl_gain_from_child_refs(
        &self,
        u_children: &[(&[u8], f64)],
        v_children: &[(&[u8], f64)],
        l: usize,
        l_eff: f64,
        ti_cost: f64,
        tv_cost: f64,
        site_weights: &[f64],
    ) -> (f64, f64, Vec<usize>) {
        let deg_a = u_children.len();
        let deg_b = v_children.len();
        let topo_cost = split_topology_cost_nats(deg_a, deg_b);

        let all_children: Vec<(&[u8], f64)> =
            u_children.iter().copied().chain(v_children.iter().copied()).collect();
        let k_ch = all_children.len();
        let mut refined_loads = Vec::with_capacity(k_ch);

        let max_load = all_children.iter().map(|c| c.1).fold(0.0_f64, f64::max);
        if k_ch == 4 && max_load >= 40.0 {
            for idx_c in 0..4 {
                let others: Vec<&[u8]> = (0..4)
                    .filter(|&j| j != idx_c)
                    .map(|j| all_children[j].0)
                    .collect();
                let mask_c = all_children[idx_c].0;
                let mut n_anc = 0usize;
                let mut n_diff = 0usize;
                for s in 0..l {
                    let co = others[0][s] & others[1][s] & others[2][s];
                    let is_single = co == 1 || co == 2 || co == 4 || co == 8;
                    if is_single && mask_c[s] != 15 {
                        n_anc += 1;
                        if (mask_c[s] & co) == 0 {
                            n_diff += 1;
                        }
                    }
                }
                if n_anc >= 50 {
                    let p_c = ((n_diff as f64) / (n_anc as f64)).min(0.73);
                    let m_cond = -0.75 * (1.0 - (4.0 / 3.0) * p_c).ln() * l_eff * 1.05;
                    let m_prior = all_children[idx_c].1;
                    let r_load = if m_cond < 0.75 * m_prior {
                        m_prior.min(m_cond)
                    } else {
                        0.5 * (m_prior + m_cond)
                    };
                    refined_loads.push(r_load.max(1.5));
                } else {
                    refined_loads.push(all_children[idx_c].1);
                }
            }
        } else {
            for &(_, mc) in &all_children {
                refined_loads.push(mc);
            }
        }

        let u_loads = &refined_loads[..deg_a];
        let v_loads = &refined_loads[deg_a..];

        let mut u_weights = Vec::with_capacity(deg_a);
        let mut v_weights = Vec::with_capacity(deg_b);
        for &m_c in u_loads {
            let mc = m_c.clamp(0.5, 0.48 * l_eff);
            u_weights.push(((l_eff - mc).max(1.5) / (mc + 1.0)).ln());
        }
        for &m_c in v_loads {
            let mc = m_c.clamp(0.5, 0.48 * l_eff);
            v_weights.push(((l_eff - mc).max(1.5) / (mc + 1.0)).ln());
        }

        let u_asym = if u_weights.len() >= 2 {
            u_weights.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                - u_weights.iter().copied().fold(f64::INFINITY, f64::min)
        } else {
            0.0
        };
        let v_asym = if v_weights.len() >= 2 {
            v_weights.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                - v_weights.iter().copied().fold(f64::INFINITY, f64::min)
        } else {
            0.0
        };

        let has_long = u_asym >= 1.5 || v_asym >= 1.5;
        let p_long_u = if u_asym >= 1.5 {
            (u_loads.iter().copied().fold(0.0_f64, f64::max) / l_eff).min(0.65)
        } else {
            0.0
        };
        let p_long_v = if v_asym >= 1.5 {
            (v_loads.iter().copied().fold(0.0_f64, f64::max) / l_eff).min(0.65)
        } else {
            0.0
        };
        let min_uw = u_weights.iter().copied().fold(f64::INFINITY, f64::min);
        let min_vw = v_weights.iter().copied().fold(f64::INFINITY, f64::min);

        let mut pos_sites: Vec<(f64, usize, f64, f64)> = Vec::new();
        let mut clean_sites: Vec<usize> = Vec::new();

        let mut eval_polymorphic_site = |s: usize| {
            let mut u_valid = 0i16;
            let mut v_valid = 0i16;
            let mut u_miss = [0i16; 4];
            let mut v_miss = [0i16; 4];
            let mut u_shared = [0i16; 4];
            let mut v_shared = [0i16; 4];
            let mut cu = [0.0_f64; 4];
            let mut cv = [0.0_f64; 4];

            for (&(s_mask, _), &w_c) in u_children.iter().zip(&u_weights) {
                let m = s_mask[s];
                if m != 15 {
                    u_valid += 1;
                    for b in 0..4 {
                        if ((m >> b) & 1) != 0 {
                            u_shared[b] += 1;
                        } else {
                            u_miss[b] += 1;
                            cu[b] += w_c;
                        }
                    }
                }
            }
            if u_valid < 2 {
                return;
            }
            for (&(s_mask, _), &w_c) in v_children.iter().zip(&v_weights) {
                let m = s_mask[s];
                if m != 15 {
                    v_valid += 1;
                    for b in 0..4 {
                        if ((m >> b) & 1) != 0 {
                            v_shared[b] += 1;
                        } else {
                            v_miss[b] += 1;
                            cv[b] += w_c;
                        }
                    }
                }
            }
            if v_valid < 2 {
                return;
            }

            let mut steps_star = i16::MAX;
            let mut c_star = f64::INFINITY;
            let mut b_u_opt = 0usize;
            let mut b_v_opt = 0usize;
            for b in 0..4 {
                let st = u_miss[b] + v_miss[b];
                if st < steps_star {
                    steps_star = st;
                }
                let cs = cu[b] + cv[b];
                if cs < c_star {
                    c_star = cs;
                }
                if cu[b] < cu[b_u_opt] {
                    b_u_opt = b;
                }
                if cv[b] < cv[b_v_opt] {
                    b_v_opt = b;
                }
            }

            if steps_star < 2 || b_u_opt == b_v_opt {
                return;
            }

            let min_cu = cu[b_u_opt];
            let min_cv = cv[b_v_opt];
            let steps_res = u_miss[b_u_opt] + v_miss[b_v_opt];

            let u_clean = u_shared[b_u_opt] >= 2 && u_miss[b_u_opt] == 0;
            let v_clean = v_shared[b_v_opt] >= 2 && v_miss[b_v_opt] == 0;
            let both_parsimony = (steps_res + 1) < steps_star;
            let raw_delta = c_star - (min_cu + min_cv);

            if both_parsimony && u_clean && v_clean {
                clean_sites.push(s);
            }

            let masked_u = v_clean && u_asym >= 1.5 && raw_delta >= 2.8;
            let masked_v = u_clean && v_asym >= 1.5 && raw_delta >= 2.8;

            if !(both_parsimony || masked_u || masked_v) {
                return;
            }

            let same_purine = IS_PURINE[b_u_opt] == IS_PURINE[b_v_opt];
            let subst_s = (if same_purine { ti_cost } else { tv_cost }) * site_weights[s];
            let delta_steps = if has_long || clean_sites.len() >= 2 {
                ((steps_star - steps_res) as f64).max(1.0)
            } else {
                1.0
            };

            let mut jc_mult_hit = 0.0_f64;
            if p_long_u > 0.05 && masked_u {
                jc_mult_hit += 0.55 * p_long_u * (min_uw + subst_s);
            }
            if p_long_v > 0.05 && masked_v {
                jc_mult_hit += 0.55 * p_long_v * (min_vw + subst_s);
            }

            let delta_pos = raw_delta * site_weights[s] + jc_mult_hit;
            let site_sav = delta_pos + delta_steps * subst_s;
            if site_sav > 0.0 {
                pos_sites.push((site_sav, s, subst_s, site_weights[s]));
            }
        };

        // 64-bit SWAR 8-site-at-a-time invariant skipping (visits polymorphic sites in exact ascending order)
        let chunks = l / 8;
        for c in 0..chunks {
            let off = c * 8;
            let mut w_and = 0x0F0F_0F0F_0F0F_0F0F_u64;
            for &(s_mask, _) in &all_children {
                w_and &= u64::from_le_bytes(s_mask[off..off + 8].try_into().unwrap());
            }
            let mut zb = !(w_and.wrapping_add(0x7F7F_7F7F_7F7F_7F7F)) & 0x8080_8080_8080_8080;
            while zb != 0 {
                let j = (zb.trailing_zeros() / 8) as usize;
                eval_polymorphic_site(off + j);
                zb &= zb - 1;
            }
        }
        for s in (chunks * 8)..l {
            let mut m_and = 15_u8;
            for &(s_mask, _) in &all_children {
                m_and &= s_mask[s];
            }
            if m_and == 0 {
                eval_polymorphic_site(s);
            }
        }

        if pos_sites.is_empty() {
            return (-topo_cost, 0.0, Vec::new());
        }

        pos_sites.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });

        let scale_pos_cost = if clean_sites.len() >= 7
            && (clean_sites.last().unwrap() - clean_sites[0]) >= 500
        {
            let uniq_codons: HashSet<usize> = clean_sites.iter().map(|&s| s / 3).collect();
            uniq_codons.len() >= 7
        } else {
            false
        };

        let mut cum_sav = 0.0_f64;
        let mut cum_subst = 0.0_f64;
        let mut cum_pos_cost = 0.0_f64;
        let mut prev_raw_pos = 0.0_f64;

        let mut best_data_gain = f64::NEG_INFINITY;
        let mut best_idx = 0usize;
        let mut first_thresh = 1.0_f64;

        for (idx, &(sav, _, subst_s, w_s)) in pos_sites.iter().enumerate() {
            let m = (idx + 1) as f64;
            cum_sav += sav;
            cum_subst += subst_s;
            let header = universal_int_nats(m);
            let raw_pos = log_binom(l_eff, m.min(0.49 * l_eff));
            if scale_pos_cost {
                let marg = raw_pos - prev_raw_pos;
                cum_pos_cost += marg * w_s.min(1.0);
                prev_raw_pos = raw_pos;
            } else {
                cum_pos_cost = raw_pos;
            }
            let branch_cost = header + cum_pos_cost + cum_subst;
            if idx == 0 {
                first_thresh = (branch_cost + topo_cost).max(1.0);
            }
            let net_gain = cum_sav - branch_cost;
            if net_gain > best_data_gain {
                best_data_gain = net_gain;
                best_idx = idx;
            }
        }

        let m_opt = (best_idx + 1) as f64;
        let chosen_sites: Vec<usize> = pos_sites[..=best_idx].iter().map(|x| x.1).collect();
        let raw_gain = best_data_gain - topo_cost;
        let final_gain = if raw_gain > 0.0 {
            raw_gain
        } else {
            let frac = (pos_sites[0].0 / first_thresh).clamp(0.0, 0.98);
            -topo_cost * (1.0 - 0.95 * frac)
        };
        (final_gain, m_opt, chosen_sites)
    }

    pub fn reconstruct_nodal_states_and_mutations(
        &self,
        adj: &BTreeMap<usize, Vec<usize>>,
        bitmasks: &[u8],
        majority_state: &[i8],
        states: &HashMap<(usize, usize), (Vec<u8>, f64)>,
        n: usize,
        l: usize,
    ) -> (
        HashMap<usize, Vec<i8>>,
        HashMap<(usize, usize), Vec<(usize, i8, i8)>>,
    ) {
        let root = adj
            .iter()
            .find(|(&u, nbs)| u >= n && !nbs.is_empty())
            .or_else(|| adj.iter().find(|(_, nbs)| !nbs.is_empty()))
            .map(|(&u, _)| u)
            .unwrap_or(0);

        let mut parent: HashMap<usize, Option<usize>> = HashMap::new();
        parent.insert(root, None);
        let mut levels: Vec<Vec<usize>> = vec![vec![root]];
        let mut total_nodes = 1usize;
        loop {
            let mut next_level = Vec::new();
            for &curr in levels.last().unwrap() {
                if let Some(nbs) = adj.get(&curr) {
                    for &nb in nbs {
                        if !parent.contains_key(&nb) {
                            parent.insert(nb, Some(curr));
                            next_level.push(nb);
                        }
                    }
                }
            }
            if next_level.is_empty() {
                break;
            }
            total_nodes += next_level.len();
            levels.push(next_level);
        }

        let mut node_state: HashMap<usize, Vec<i8>> = HashMap::with_capacity(total_nodes);
        let mut node_mask_dict: HashMap<usize, Vec<u8>> = HashMap::with_capacity(total_nodes);
        let mut edge_mutations: HashMap<(usize, usize), Vec<(usize, i8, i8)>> =
            HashMap::with_capacity(total_nodes);

        let maj_bit: Vec<u8> = majority_state.iter().map(|&b| 1_u8 << (b as u8)).collect();

        for level in &levels {
            let compute_node =
                |&u: &usize| -> (usize, Vec<u8>, Vec<i8>, Option<((usize, usize), Vec<(usize, i8, i8)>)>) {
                    let p_opt = parent[&u];
                    let c_mask = if u < n {
                        bitmasks[u * l..(u + 1) * l].to_vec()
                    } else if let Some(p) = p_opt {
                        states[&(u, p)].0.clone()
                    } else {
                        let child_refs: Vec<(&[u8], f64)> = adj[&u]
                            .iter()
                            .copied()
                            .map(|c| {
                                let st = &states[&(c, u)];
                                (st.0.as_slice(), st.1)
                            })
                            .collect();
                        Self::combine_fitch_children_refs(&child_refs).0
                    };

                    if let Some(p) = p_opt {
                        let p_st = &node_state[&p];
                        let p_bit: Vec<u8> = p_st.iter().map(|&b| 1_u8 << (b as u8)).collect();
                        let mpr_mask = if u < n {
                            c_mask.clone()
                        } else {
                            let mut star_refs: Vec<(&[u8], f64)> = adj[&u]
                                .iter()
                                .copied()
                                .filter(|&nb| nb != p)
                                .map(|c| {
                                    let st = &states[&(c, u)];
                                    (st.0.as_slice(), st.1)
                                })
                                .collect();
                            star_refs.push((p_bit.as_slice(), 1.0));
                            Self::combine_fitch_children_refs(&star_refs).0
                        };
                        let mut st = vec![0_i8; l];
                        for s in 0..l {
                            let m = mpr_mask[s];
                            if (m & p_bit[s]) != 0 {
                                st[s] = p_st[s];
                            } else if (m & maj_bit[s]) != 0 {
                                st[s] = majority_state[s];
                            } else {
                                for b in 0..4_i8 {
                                    if ((m >> b) & 1) != 0 {
                                        st[s] = b;
                                        break;
                                    }
                                }
                            }
                        }

                        let mask_p = &node_mask_dict[&p];
                        let mut muts = Vec::new();
                        for s in 0..l {
                            if mask_p[s] != 15 && c_mask[s] != 15 && p_st[s] != st[s] {
                                if p < u {
                                    muts.push((s, p_st[s], st[s]));
                                } else {
                                    muts.push((s, st[s], p_st[s]));
                                }
                            }
                        }
                        (u, c_mask, st, Some(((p.min(u), p.max(u)), muts)))
                    } else {
                        let mut st = vec![0_i8; l];
                        for s in 0..l {
                            let m = c_mask[s];
                            if (m & maj_bit[s]) != 0 {
                                st[s] = majority_state[s];
                            } else {
                                for b in 0..4_i8 {
                                    if ((m >> b) & 1) != 0 {
                                        st[s] = b;
                                        break;
                                    }
                                }
                            }
                        }
                        (u, c_mask, st, None)
                    }
                };

            #[cfg(not(target_arch = "wasm32"))]
            let results: Vec<(
                usize,
                Vec<u8>,
                Vec<i8>,
                Option<((usize, usize), Vec<(usize, i8, i8)>)>,
            )> = if level.len() >= 8 {
                level.par_iter().map(compute_node).collect()
            } else {
                level.iter().map(compute_node).collect()
            };

            #[cfg(target_arch = "wasm32")]
            let results: Vec<(
                usize,
                Vec<u8>,
                Vec<i8>,
                Option<((usize, usize), Vec<(usize, i8, i8)>)>,
            )> = level.iter().map(compute_node).collect();

            for (u, cm, st, em_opt) in results {
                node_mask_dict.insert(u, cm);
                node_state.insert(u, st);
                if let Some((ekey, muts)) = em_opt {
                    edge_mutations.insert(ekey, muts);
                }
            }
        }

        (node_state, edge_mutations)
    }

    #[allow(clippy::too_many_arguments)]
    fn peel_residual_shunts_mdl(
        &self,
        adj: &BTreeMap<usize, Vec<usize>>,
        node_labels: &BTreeMap<usize, String>,
        bitmasks: &[u8],
        majority_state: &[i8],
        terminal_loads: &[f64],
        edge_mutations: &HashMap<(usize, usize), Vec<(usize, i8, i8)>>,
        n: usize,
        l: usize,
        l_eff: f64,
        ti_cost: f64,
        tv_cost: f64,
        mat: &[i8],
        is_coding: bool,
    ) -> Vec<ShuntEdge> {
        let active_nodes: Vec<usize> = adj
            .iter()
            .filter(|(_, nbs)| !nbs.is_empty())
            .map(|(&u, _)| u)
            .collect();
        let v_count = active_nodes.len().max(4) as f64;
        let endpoint_cost = 2.0 * v_count.ln();

        let mut site_mut_counts = vec![0_i32; l];
        for mut_list in edge_mutations.values() {
            for &(s, _, _) in mut_list {
                site_mut_counts[s] += 1;
            }
        }
        let variable_sites = site_mut_counts.iter().filter(|&&c| c > 0).count().max(1);
        let aa_variable_codons = if is_coding {
            let mut n_aa_var = 0usize;
            let mut c_start = 0usize;
            while c_start + 2 < l {
                let mut seen_aas: HashSet<u8> = HashSet::new();
                for i in 0..n {
                    let row = &mat[i * l..(i + 1) * l];
                    let b0 = row[c_start];
                    let b1 = row[c_start + 1];
                    let b2 = row[c_start + 2];
                    if (0..=3).contains(&b0) && (0..=3).contains(&b1) && (0..=3).contains(&b2) {
                        let aa = translate_bases(b0, b1, b2);
                        if aa != b'*' && aa != b'X' {
                            seen_aas.insert(aa);
                            if seen_aas.len() >= 2 {
                                break;
                            }
                        }
                    }
                }
                if seen_aas.len() >= 2 {
                    n_aa_var += 1;
                }
                c_start += 3;
            }
            n_aa_var.max(1)
        } else {
            variable_sites
        };
        let shunt_var_pool = if is_coding {
            aa_variable_codons
        } else {
            variable_sites
        };

        let root = adj
            .iter()
            .find(|(&u, nbs)| u >= n && !nbs.is_empty())
            .or_else(|| adj.iter().find(|(_, nbs)| !nbs.is_empty()))
            .map(|(&u, _)| u)
            .unwrap_or(0);

        let mut parent: HashMap<usize, Option<usize>> = HashMap::new();
        parent.insert(root, None);
        let mut q = VecDeque::new();
        q.push_back(root);
        while let Some(curr) = q.pop_front() {
            if let Some(nbs) = adj.get(&curr) {
                for &nb in nbs {
                    if !parent.contains_key(&nb) {
                        parent.insert(nb, Some(curr));
                        q.push_back(nb);
                    }
                }
            }
        }

        let mut branch_by_child: HashMap<usize, Vec<(usize, i8, i8)>> = HashMap::new();
        let mut branch_mut_load: HashMap<usize, f64> = HashMap::new();
        for (&u, &p_opt) in &parent {
            let Some(p) = p_opt else { continue };
            let key = (p.min(u), p.max(u));
            let raw_list = edge_mutations.get(&key).cloned().unwrap_or_default();
            let directed: Vec<(usize, i8, i8)> = if p < u {
                raw_list
            } else {
                raw_list.into_iter().map(|(s, bu, bp)| (s, bp, bu)).collect()
            };
            branch_mut_load.insert(u, (directed.len() as f64).max(0.5));
            branch_by_child.insert(u, directed);
        }

        let states_dir =
            self.compute_all_directed_fitch_states(adj, bitmasks, n, l, terminal_loads, l_eff);
        let (node_state, _) = self.reconstruct_nodal_states_and_mutations(
            adj,
            bitmasks,
            majority_state,
            &states_dir,
            n,
            l,
        );

        let build_valid = |&u: &usize| -> (usize, Vec<bool>) {
            if u < n {
                let row = &mat[u * l..(u + 1) * l];
                (u, row.iter().map(|&x| x >= 0).collect())
            } else {
                let nb_check = parent
                    .get(&u)
                    .and_then(|&p| p)
                    .unwrap_or_else(|| adj[&u][0]);
                let m = &states_dir[&(u, nb_check)].0;
                (u, m.iter().map(|&x| x != 15).collect())
            }
        };
        #[cfg(not(target_arch = "wasm32"))]
        let node_valid: HashMap<usize, Vec<bool>> =
            active_nodes.par_iter().map(build_valid).collect();
        #[cfg(target_arch = "wasm32")]
        let node_valid: HashMap<usize, Vec<bool>> =
            active_nodes.iter().map(build_valid).collect();

        let orig_branch_by_child = branch_by_child.clone();
        let orig_branch_load = branch_mut_load.clone();
        let mut origin_node: HashMap<(usize, usize), usize> = HashMap::new();
        for (&u, muts) in &orig_branch_by_child {
            for &(s, _, _) in muts {
                origin_node.insert((u, s), u);
            }
        }

        let mut eff_anc_node: HashMap<usize, usize> = HashMap::new();
        for &u in &active_nodes {
            let Some(Some(p)) = parent.get(&u) else {
                continue;
            };
            let mut eff_anc = *p;
            if is_coding && u >= n {
                let mut cum_load = orig_branch_load.get(&u).copied().unwrap_or(1.0);
                let mut curr_w = *p;
                let mut steps_up = 0usize;
                let mut existing_sites: HashSet<usize> = branch_by_child
                    .get(&u)
                    .map(|v| v.iter().map(|x| x.0).collect())
                    .unwrap_or_default();
                let st_u_arr = &node_state[&u];
                let mut added_any = false;
                while curr_w >= n && steps_up < 3 {
                    let Some(Some(next_p)) = parent.get(&curr_w).copied() else {
                        break;
                    };
                    let add_load = orig_branch_load.get(&curr_w).copied().unwrap_or(1.0);
                    if cum_load + add_load > 48.0 {
                        break;
                    }
                    cum_load += add_load;
                    let st_anc_arr = &node_state[&next_p];
                    if let Some(w_muts) = orig_branch_by_child.get(&curr_w) {
                        for &(s_int, a_st, b_st) in w_muts {
                            if !existing_sites.contains(&s_int)
                                && node_valid[&u][s_int]
                                && st_u_arr[s_int] == b_st
                            {
                                let c_start = 3 * (s_int / 3);
                                if c_start + 2 < l {
                                    let aa_u = translate_bases(
                                        st_u_arr[c_start],
                                        st_u_arr[c_start + 1],
                                        st_u_arr[c_start + 2],
                                    );
                                    let aa_a = translate_bases(
                                        st_anc_arr[c_start],
                                        st_anc_arr[c_start + 1],
                                        st_anc_arr[c_start + 2],
                                    );
                                    if aa_u != aa_a {
                                        branch_by_child
                                            .entry(u)
                                            .or_default()
                                            .push((s_int, a_st, b_st));
                                        origin_node.insert((u, s_int), curr_w);
                                        existing_sites.insert(s_int);
                                        added_any = true;
                                    }
                                }
                            }
                        }
                    }
                    eff_anc = next_p;
                    curr_w = next_p;
                    steps_up += 1;
                }
                if added_any {
                    let entry = branch_mut_load.entry(u).or_insert(1.0);
                    *entry = entry.max(cum_load * 0.65);
                }
            }
            eff_anc_node.insert(u, eff_anc);
        }

        let mut site_to_branches: Vec<Vec<(usize, i8, i8)>> = vec![Vec::new(); l];
        for (&u, muts) in &branch_by_child {
            for &(s, a, b) in muts {
                site_to_branches[s].push((u, a, b));
            }
        }

        // O(1) Euler-tour ancestor table (zero HashMap pointer chasing inside site loop)
        let max_node_id = active_nodes.iter().copied().max().unwrap_or(0).max(root);
        let mut tin = vec![0_u32; max_node_id + 1];
        let mut tout = vec![0_u32; max_node_id + 1];
        let mut tree_children: HashMap<usize, Vec<usize>> = HashMap::with_capacity(parent.len());
        for (&u, &p_opt) in &parent {
            if let Some(p) = p_opt {
                tree_children.entry(p).or_default().push(u);
            }
        }
        let mut timer = 0_u32;
        let mut dfs_stack: Vec<(usize, bool)> = vec![(root, false)];
        while let Some((u, exiting)) = dfs_stack.pop() {
            if exiting {
                tout[u] = timer;
            } else {
                timer += 1;
                tin[u] = timer;
                dfs_stack.push((u, true));
                if let Some(ch) = tree_children.get(&u) {
                    for &c in ch.iter().rev() {
                        dfs_stack.push((c, false));
                    }
                }
            }
        }
        let is_ancestor = |a_node: usize, b_node: usize| -> bool {
            tin.get(a_node).copied().unwrap_or(0) != 0
                && tin.get(b_node).copied().unwrap_or(0) != 0
                && tin[a_node] <= tin[b_node]
                && tout[b_node] <= tout[a_node]
        };

        let mut pair_shared_sites: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        let mut pair_priority_score: HashMap<(usize, usize), f64> = HashMap::new();

        for (s, entries) in site_to_branches.iter().enumerate() {
            if entries.len() < 2 || entries.len() > 36 {
                continue;
            }
            let bg_c = site_mut_counts[s].max(1) as f64;
            for i in 0..entries.len() {
                let (u, _, b_u) = entries[i];
                let w_u = origin_node.get(&(u, s)).copied().unwrap_or(u);
                for j in (i + 1)..entries.len() {
                    let (v, _, b_v) = entries[j];
                    if b_u == b_v {
                        let w_v = origin_node.get(&(v, s)).copied().unwrap_or(v);
                        if w_u == w_v {
                            continue;
                        }
                        if is_ancestor(u, v) || is_ancestor(v, u) {
                            continue;
                        }
                        if is_ancestor(w_u, v) || is_ancestor(w_v, u) {
                            continue;
                        }
                        if w_u != u || w_v != v {
                            let pw_u = parent.get(&w_u).copied().flatten();
                            let pw_v = parent.get(&w_v).copied().flatten();
                            let p_pw_u = pw_u.and_then(|x| parent.get(&x).copied().flatten());
                            let p_pw_v = pw_v.and_then(|x| parent.get(&x).copied().flatten());
                            if pw_u == pw_v || p_pw_u == pw_v || pw_u == p_pw_v {
                                continue;
                            }
                        }
                        let pair = (u.min(v), u.max(v));
                        pair_shared_sites.entry(pair).or_default().push(s);
                        let mut w_s = 1.0 / bg_c;
                        if is_coding {
                            let c_start = 3 * (s / 3);
                            if c_start + 2 < l {
                                let p_u_eff = parent
                                    .get(&w_u)
                                    .copied()
                                    .flatten()
                                    .or_else(|| eff_anc_node.get(&u).copied());
                                let p_v_eff = parent
                                    .get(&w_v)
                                    .copied()
                                    .flatten()
                                    .or_else(|| eff_anc_node.get(&v).copied());
                                if let (Some(pu), Some(pv)) = (p_u_eff, p_v_eff) {
                                    let st_u = &node_state[&u];
                                    let st_v = &node_state[&v];
                                    let st_pu = &node_state[&pu];
                                    let st_pv = &node_state[&pv];
                                    let aa_u = translate_bases(
                                        st_u[c_start],
                                        st_u[c_start + 1],
                                        st_u[c_start + 2],
                                    );
                                    let aa_v = translate_bases(
                                        st_v[c_start],
                                        st_v[c_start + 1],
                                        st_v[c_start + 2],
                                    );
                                    let aa_pu = translate_bases(
                                        st_pu[c_start],
                                        st_pu[c_start + 1],
                                        st_pu[c_start + 2],
                                    );
                                    let aa_pv = translate_bases(
                                        st_pv[c_start],
                                        st_pv[c_start + 1],
                                        st_pv[c_start + 2],
                                    );
                                    if aa_u == aa_v && aa_u != aa_pu && aa_v != aa_pv {
                                        w_s *= 4.0;
                                    }
                                }
                            }
                        }
                        *pair_priority_score.entry(pair).or_insert(0.0) += w_s;
                    }
                }
            }
        }

        let mut sorted_pairs: Vec<((usize, usize), Vec<usize>)> =
            pair_shared_sites.into_iter().collect();
        sorted_pairs.sort_by(|a, b| {
            let sa = pair_priority_score.get(&a.0).copied().unwrap_or(0.0);
            let sb = pair_priority_score.get(&b.0).copied().unwrap_or(0.0);
            sb.partial_cmp(&sa)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.1.len().cmp(&a.1.len()))
                .then_with(|| a.0.cmp(&b.0))
        });
        sorted_pairs.truncate(500);

        let eval_shunt_pair = |((u, v), mut shared_sites): ((usize, usize), Vec<usize>)| -> Option<ShuntEdge> {
            shared_sites.sort_unstable();
            shared_sites.dedup();
            let avail_sites: Vec<usize> = shared_sites
                .into_iter()
                .filter(|&s| node_valid[&u][s] && node_valid[&v][s])
                .collect();
            if avail_sites.len() < self.min_shunt_sites {
                return None;
            }

            let (best_int_gain, best_int_sites, best_span, int_data_sav, int_mask_cost) = self
                .evaluate_interval_shunt_mdl(
                    u,
                    v,
                    &avail_sites,
                    &site_mut_counts,
                    &node_state,
                    &node_valid,
                    &eff_anc_node,
                    &parent,
                    adj,
                    &branch_mut_load,
                    l,
                    l_eff,
                    variable_sites,
                    endpoint_cost,
                    ti_cost,
                    tv_cost,
                    is_coding,
                );

            let pu_direct = parent.get(&u).copied().flatten();
            let pv_direct = parent.get(&v).copied().flatten();
            let local_triad = pu_direct == pv_direct
                || (pu_direct.is_some()
                    && parent.get(&pu_direct.unwrap()).copied().flatten() == pv_direct)
                || (pv_direct.is_some()
                    && pu_direct == parent.get(&pv_direct.unwrap()).copied().flatten());

            let (best_sp_gain, best_sp_sites, sp_data_sav, sp_mask_cost) = if local_triad {
                (-10.0, Vec::new(), 0.0, 0.0)
            } else {
                self.evaluate_sparse_shunt_mdl(
                    u,
                    v,
                    &avail_sites,
                    &site_mut_counts,
                    &node_state,
                    &eff_anc_node,
                    &branch_mut_load,
                    l_eff,
                    shunt_var_pool,
                    endpoint_cost,
                    ti_cost,
                    tv_cost,
                    is_coding,
                )
            };

            if best_int_gain <= 0.0 && best_sp_gain <= 0.0 {
                return None;
            }

            let (chosen_mode, chosen_sites, chosen_span, data_sav, mask_cost, net_gain) =
                if best_int_gain >= best_sp_gain && best_int_gain > 0.0 {
                    (
                        "INTERVAL".to_string(),
                        best_int_sites,
                        best_span,
                        int_data_sav,
                        int_mask_cost,
                        best_int_gain,
                    )
                } else {
                    (
                        "SPARSE".to_string(),
                        best_sp_sites,
                        None,
                        sp_data_sav,
                        sp_mask_cost,
                        best_sp_gain,
                    )
                };

            let post_prob = 1.0 / (1.0 + (-net_gain.clamp(-50.0, 50.0)).exp());
            Some(ShuntEdge {
                node_u: u,
                node_v: v,
                label_u: node_labels
                    .get(&u)
                    .cloned()
                    .unwrap_or_else(|| format!("N{u}")),
                label_v: node_labels
                    .get(&v)
                    .cloned()
                    .unwrap_or_else(|| format!("N{v}")),
                mode: chosen_mode,
                shared_mutations: chosen_sites.len(),
                sites: chosen_sites,
                interval_span: chosen_span,
                data_savings_nats: data_sav,
                mask_cost_nats: mask_cost,
                endpoint_cost_nats: endpoint_cost,
                net_mdl_gain_nats: net_gain,
                posterior_prob: post_prob,
            })
        };

        #[cfg(not(target_arch = "wasm32"))]
        let mut candidate_shunts: Vec<ShuntEdge> =
            sorted_pairs.into_par_iter().filter_map(eval_shunt_pair).collect();

        #[cfg(target_arch = "wasm32")]
        let mut candidate_shunts: Vec<ShuntEdge> =
            sorted_pairs.into_iter().filter_map(eval_shunt_pair).collect();

        candidate_shunts.sort_by(|a, b| {
            b.net_mdl_gain_nats
                .partial_cmp(&a.net_mdl_gain_nats)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.node_u.cmp(&b.node_u))
                .then_with(|| a.node_v.cmp(&b.node_v))
        });

        let mut certified_shunts: Vec<ShuntEdge> = Vec::new();
        let mut used_sites_per_node: HashMap<usize, HashSet<usize>> = HashMap::new();

        for mut sh in candidate_shunts {
            let (u, v) = (sh.node_u, sh.node_v);
            let overlap: HashSet<usize> = sh
                .sites
                .iter()
                .copied()
                .filter(|&s| {
                    let wu = origin_node.get(&(u, s)).copied().unwrap_or(u);
                    let wv = origin_node.get(&(v, s)).copied().unwrap_or(v);
                    used_sites_per_node
                        .get(&u)
                        .map(|set| set.contains(&s))
                        .unwrap_or(false)
                        || used_sites_per_node
                            .get(&v)
                            .map(|set| set.contains(&s))
                            .unwrap_or(false)
                        || used_sites_per_node
                            .get(&wu)
                            .map(|set| set.contains(&s))
                            .unwrap_or(false)
                        || used_sites_per_node
                            .get(&wv)
                            .map(|set| set.contains(&s))
                            .unwrap_or(false)
                })
                .collect();

            if sh.sites.len().saturating_sub(overlap.len()) < self.min_shunt_sites {
                continue;
            }
            if !overlap.is_empty() {
                let rem_sites: Vec<usize> = sh
                    .sites
                    .iter()
                    .copied()
                    .filter(|s| !overlap.contains(s))
                    .collect();
                let (g_r, sites_r, span_r, sav_r, mc_r) = if sh.mode == "INTERVAL" {
                    self.evaluate_interval_shunt_mdl(
                        u,
                        v,
                        &rem_sites,
                        &site_mut_counts,
                        &node_state,
                        &node_valid,
                        &eff_anc_node,
                        &parent,
                        adj,
                        &branch_mut_load,
                        l,
                        l_eff,
                        variable_sites,
                        endpoint_cost,
                        ti_cost,
                        tv_cost,
                        is_coding,
                    )
                } else {
                    let (g, st, sv, mc) = self.evaluate_sparse_shunt_mdl(
                        u,
                        v,
                        &rem_sites,
                        &site_mut_counts,
                        &node_state,
                        &eff_anc_node,
                        &branch_mut_load,
                        l_eff,
                        shunt_var_pool,
                        endpoint_cost,
                        ti_cost,
                        tv_cost,
                        is_coding,
                    );
                    (g, st, None, sv, mc)
                };
                if g_r <= 0.0 {
                    continue;
                }
                sh.shared_mutations = sites_r.len();
                sh.sites = sites_r;
                sh.interval_span = span_r;
                sh.data_savings_nats = sav_r;
                sh.mask_cost_nats = mc_r;
                sh.net_mdl_gain_nats = g_r;
                sh.posterior_prob = 1.0 / (1.0 + (-g_r.clamp(-50.0, 50.0)).exp());
            }

            for &s in &sh.sites {
                let wu = origin_node.get(&(u, s)).copied().unwrap_or(u);
                let wv = origin_node.get(&(v, s)).copied().unwrap_or(v);
                used_sites_per_node.entry(u).or_default().insert(s);
                used_sites_per_node.entry(v).or_default().insert(s);
                used_sites_per_node.entry(wu).or_default().insert(s);
                used_sites_per_node.entry(wv).or_default().insert(s);
            }
            certified_shunts.push(sh);
        }

        certified_shunts.sort_by(|a, b| {
            b.net_mdl_gain_nats
                .partial_cmp(&a.net_mdl_gain_nats)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        certified_shunts
    }

    #[allow(clippy::too_many_arguments)]
    fn compute_parallel_site_savings(
        &self,
        u: usize,
        v: usize,
        shared_sites: &[usize],
        site_mut_counts: &[i32],
        node_state: &HashMap<usize, Vec<i8>>,
        eff_anc_node: &HashMap<usize, usize>,
        branch_mut_load: &HashMap<usize, f64>,
        l_eff: f64,
        variable_sites: usize,
        ti_cost: f64,
        tv_cost: f64,
        is_coding: bool,
    ) -> Vec<(usize, f64)> {
        let (Some(&p_u), Some(&p_v)) = (eff_anc_node.get(&u), eff_anc_node.get(&v)) else {
            return Vec::new();
        };
        let st_u = &node_state[&u];
        let st_pu = &node_state[&p_u];
        let st_v = &node_state[&v];
        let st_pv = &node_state[&p_v];
        let l = st_u.len();
        let m_u = branch_mut_load.get(&u).copied().unwrap_or(1.0);
        let m_v = branch_mut_load.get(&v).copied().unwrap_or(1.0);

        let mut results = Vec::new();
        for &s in shared_sites {
            if st_u[s] != st_pu[s] && st_v[s] != st_pv[s] && st_u[s] == st_v[s] {
                let bg_muts = (site_mut_counts[s] - 2).max(0) as f64;
                let b_anc = st_pu[s] as usize;
                let b_der = st_u[s] as usize;
                let subst_s = if IS_PURINE[b_anc] == IS_PURINE[b_der] {
                    ti_cost
                } else {
                    tv_cost
                };

                let mut codon_bonus = 0.0_f64;
                let mut is_synonymous = false;
                if is_coding {
                    let c_start = 3 * (s / 3);
                    if c_start + 2 < l {
                        let aa_u =
                            translate_bases(st_u[c_start], st_u[c_start + 1], st_u[c_start + 2]);
                        let aa_pu =
                            translate_bases(st_pu[c_start], st_pu[c_start + 1], st_pu[c_start + 2]);
                        let aa_v =
                            translate_bases(st_v[c_start], st_v[c_start + 1], st_v[c_start + 2]);
                        let aa_pv =
                            translate_bases(st_pv[c_start], st_pv[c_start + 1], st_pv[c_start + 2]);
                        if aa_u != aa_pu && aa_v != aa_pv && aa_u == aa_v {
                            codon_bonus = 19.0_f64.ln() - subst_s;
                        } else if aa_u == aa_pu && aa_v == aa_pv {
                            is_synonymous = true;
                        }
                    }
                }

                let expected_collision = (m_u * m_v) / (3.0 * l_eff).max(1.0);
                let (bp_term, conservation_surprise) = if is_synonymous {
                    let bp = ((variable_sites as f64) / (0.5 * (m_u + m_v)).max(0.5))
                        .max(1.0)
                        .ln()
                        - expected_collision.max(1.0).ln().max(0.0);
                    (bp, 0.0)
                } else {
                    let bp = (l_eff / (0.5 * (m_u + m_v)).max(0.5)).max(1.0).ln()
                        - expected_collision.max(1.0).ln().max(0.0);
                    let eff_var_pool = (variable_sites as f64).max(1.0);
                    let cs = ((l_eff / eff_var_pool) * (3.0 / (bg_muts + 1.0)))
                        .ln()
                        .max(0.0);
                    (bp, cs)
                };

                let sav = bp_term + subst_s + codon_bonus + conservation_surprise;
                if sav > 0.0 {
                    results.push((s, sav));
                }
            }
        }
        results
    }

    #[allow(clippy::too_many_arguments)]
    fn evaluate_interval_shunt_mdl(
        &self,
        u: usize,
        v: usize,
        shared_sites: &[usize],
        site_mut_counts: &[i32],
        node_state: &HashMap<usize, Vec<i8>>,
        node_valid: &HashMap<usize, Vec<bool>>,
        eff_anc_node: &HashMap<usize, usize>,
        parent: &HashMap<usize, Option<usize>>,
        adj: &BTreeMap<usize, Vec<usize>>,
        branch_mut_load: &HashMap<usize, f64>,
        l: usize,
        l_eff: f64,
        variable_sites: usize,
        endpoint_cost: f64,
        ti_cost: f64,
        tv_cost: f64,
        is_coding: bool,
    ) -> (f64, Vec<usize>, Option<(usize, usize)>, f64, f64) {
        let site_sav_pairs = self.compute_parallel_site_savings(
            u,
            v,
            shared_sites,
            site_mut_counts,
            node_state,
            eff_anc_node,
            branch_mut_load,
            l_eff,
            variable_sites,
            ti_cost,
            tv_cost,
            is_coding,
        );
        let k = site_sav_pairs.len();
        let min_k = self.min_shunt_sites.max(3);
        if k < min_k {
            return (-1e9, Vec::new(), None, 0.0, 0.0);
        }

        let cand_sites: Vec<usize> = site_sav_pairs.iter().map(|x| x.0).collect();
        let cand_savs: Vec<f64> = site_sav_pairs.iter().map(|x| x.1).collect();
        let base_mask_cost = 2.0_f64.ln() + 2.0 * (l.max(2) as f64).ln();

        let val_u = &node_valid[&u];
        let val_v = &node_valid[&v];
        let both_val: Vec<bool> = (0..l).map(|s| val_u[s] && val_v[s]).collect();

        let mut u_reps = vec![u];
        if let Some(nbs) = adj.get(&u) {
            for &nb in nbs {
                if parent.get(&nb).copied().flatten() == Some(u) {
                    u_reps.push(nb);
                }
            }
        }
        let mut v_reps = vec![v];
        if let Some(nbs) = adj.get(&v) {
            for &nb in nbs {
                if parent.get(&nb).copied().flatten() == Some(v) {
                    v_reps.push(nb);
                }
            }
        }

        let mut disc_prefs: Vec<Vec<i32>> = Vec::new();
        let mut disc_pos_list: Vec<Vec<usize>> = Vec::new();
        for &ur in &u_reps {
            let st_ur = &node_state[&ur];
            for &vr in &v_reps {
                let st_vr = &node_state[&vr];
                let mut pref = Vec::with_capacity(l + 1);
                pref.push(0);
                let mut pos = Vec::new();
                let mut acc = 0i32;
                for s in 0..l {
                    if both_val[s] && st_ur[s] != st_vr[s] {
                        acc += 1;
                        pos.push(s);
                    }
                    pref.push(acc);
                }
                disc_prefs.push(pref);
                disc_pos_list.push(pos);
            }
        }

        let mut gap_pref = Vec::with_capacity(l + 1);
        gap_pref.push(0i32);
        let mut g_acc = 0i32;
        for &bv in &both_val {
            if !bv {
                g_acc += 1;
            }
            gap_pref.push(g_acc);
        }

        let mut sav_pref = Vec::with_capacity(k + 1);
        sav_pref.push(0.0_f64);
        let mut s_acc = 0.0_f64;
        for &sv in &cand_savs {
            s_acc += sv;
            sav_pref.push(s_acc);
        }

        let mut best_gain = -1e9_f64;
        let mut best_sites = Vec::new();
        let mut best_span = None;
        let mut best_data_sav = 0.0_f64;
        let mut best_mask_cost = base_mask_cost;

        let l_val_total = (l.saturating_sub(gap_pref[l] as usize)).max(1) as f64;
        for i in 0..k {
            let a = cand_sites[i];
            for j in (i + min_k - 1)..k {
                let b = cand_sites[j];
                let span_len = b - a + 1;
                let m_in = j - i + 1;
                // Unrooted identifiability: a lateral tract spans at most half the alignment
                if span_len < min_k || span_len > (l / 2) {
                    continue;
                }
                let gaps_in = (gap_pref[b + 1] - gap_pref[a]) as usize;
                let mut best_rep_idx = 0usize;
                let mut discordant = i32::MAX;
                let mut total_disc_best = 0i32;
                for (r_idx, dp) in disc_prefs.iter().enumerate() {
                    let d_val = dp[b + 1] - dp[a];
                    if d_val < discordant {
                        discordant = d_val;
                        best_rep_idx = r_idx;
                        total_disc_best = dp[l];
                    }
                }
                let disc_f = discordant as f64;
                let d_all = (total_disc_best as f64).max(disc_f);
                let d_out = (d_all - disc_f).max(0.0);
                let w_val = (span_len.saturating_sub(gaps_in)).max(1) as f64;
                let out_val = (l_val_total - w_val).max(1.0);
                if (disc_f / w_val) >= (d_out / out_val) {
                    continue;
                }
                // Exact combinadic block-homogeneity gain of partitioning (u, v) divergence at (a, b):
                // Delta L_block = ln C(L_val, D_all) - ln C(W_val, D_in) - ln C(L_out, D_out)
                let block_kl_gain = (log_binom(l_val_total, d_all)
                    - log_binom(w_val, disc_f)
                    - log_binom(out_val, d_out))
                    .max(0.0);
                let w_pool = (w_val * (variable_sites as f64) / l_val_total)
                    .max(m_in as f64);
                let window_addr_cost =
                    (log_binom(w_pool, m_in as f64) - block_kl_gain).max(0.0);
                let eff_disc = disc_f + (gaps_in as f64);
                let disc_cost = if eff_disc > 0.0 {
                    branch_mutation_code_nats(eff_disc, span_len as f64, 0.0, 4)
                } else {
                    0.0
                };
                let total_mask_cost = base_mask_cost + disc_cost + window_addr_cost;
                let data_sav = sav_pref[j + 1] - sav_pref[i];
                let net_gain = data_sav - endpoint_cost - total_mask_cost;
                if net_gain > best_gain {
                    best_gain = net_gain;
                    best_sites = cand_sites[i..=j].to_vec();
                    let disc_positions = &disc_pos_list[best_rep_idx];
                    let s_left = disc_positions
                        .iter()
                        .copied()
                        .rfind(|&p| p < a)
                        .unwrap_or_else(|| a.saturating_sub(20));
                    let s_right = disc_positions
                        .iter()
                        .copied()
                        .find(|&p| p > b)
                        .unwrap_or_else(|| (b + 20).min(l - 1));
                    best_span = Some(((s_left + a) / 2, (b + s_right) / 2));
                    best_data_sav = data_sav;
                    best_mask_cost = total_mask_cost;
                }
            }
        }

        (
            best_gain,
            best_sites,
            best_span,
            best_data_sav,
            best_mask_cost,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn evaluate_sparse_shunt_mdl(
        &self,
        u: usize,
        v: usize,
        shared_sites: &[usize],
        site_mut_counts: &[i32],
        node_state: &HashMap<usize, Vec<i8>>,
        eff_anc_node: &HashMap<usize, usize>,
        branch_mut_load: &HashMap<usize, f64>,
        l_eff: f64,
        variable_sites: usize,
        endpoint_cost: f64,
        ti_cost: f64,
        tv_cost: f64,
        is_coding: bool,
    ) -> (f64, Vec<usize>, f64, f64) {
        let site_sav_pairs = self.compute_parallel_site_savings(
            u,
            v,
            shared_sites,
            site_mut_counts,
            node_state,
            eff_anc_node,
            branch_mut_load,
            l_eff,
            variable_sites,
            ti_cost,
            tv_cost,
            is_coding,
        );
        if site_sav_pairs.len() < self.min_shunt_sites {
            return (-1e9, Vec::new(), 0.0, 0.0);
        }

        let mut strict_parallel: Vec<(f64, usize)> =
            site_sav_pairs.into_iter().map(|(s, sav)| (sav, s)).collect();
        strict_parallel.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.1.cmp(&a.1))
        });

        let m_pool = (variable_sites as f64).max(strict_parallel.len() as f64);
        let _ = (l_eff, is_coding);

        let mut best_gain = -1e9_f64;
        let mut best_sites = Vec::new();
        let mut best_data_sav = 0.0_f64;
        let mut best_mask_cost = 0.0_f64;
        let mut cum_sav = 0.0_f64;
        let mut cur_sites = Vec::new();

        for (idx, &(sav, s)) in strict_parallel.iter().enumerate() {
            let m_idx = idx + 1;
            cum_sav += sav;
            cur_sites.push(s);
            if m_idx >= self.min_shunt_sites {
                let mask_cost = 2.0_f64.ln()
                    + universal_int_nats((m_idx - self.min_shunt_sites) as f64)
                    + log_binom(m_pool, m_idx as f64);
                let net_gain = cum_sav - endpoint_cost - mask_cost;
                if net_gain > best_gain {
                    best_gain = net_gain;
                    let mut sorted_s = cur_sites.clone();
                    sorted_s.sort_unstable();
                    best_sites = sorted_s;
                    best_data_sav = cum_sav;
                    best_mask_cost = mask_cost;
                }
            }
        }

        (best_gain, best_sites, best_data_sav, best_mask_cost)
    }

    fn compute_star_tree_mdl_nats(
        &self,
        bitmasks: &[u8],
        n: usize,
        l: usize,
        l_eff: f64,
    ) -> f64 {
        let block_w = 512usize;
        let n_blocks = (l + block_w - 1) / block_w;
        let build_root_block = |b_idx: usize| -> Vec<u8> {
            let cs = b_idx * block_w;
            let ce = (cs + block_w).min(l);
            let cw = ce - cs;
            let mut counts = [[0_u16; 4]; 512];
            for i in 0..n {
                let row = &bitmasks[i * l + cs..i * l + ce];
                for k in 0..cw {
                    let m = row[k];
                    if m != 15 {
                        counts[k][0] += (m & 1) as u16;
                        counts[k][1] += ((m >> 1) & 1) as u16;
                        counts[k][2] += ((m >> 2) & 1) as u16;
                        counts[k][3] += ((m >> 3) & 1) as u16;
                    }
                }
            }
            let mut out = vec![15_u8; cw];
            for k in 0..cw {
                let c = &counts[k];
                let max_c = c[0].max(c[1]).max(c[2].max(c[3]));
                if max_c > 0 {
                    let mut mask = 0_u8;
                    for b in 0..4 {
                        if c[b] == max_c {
                            mask |= 1 << b;
                        }
                    }
                    out[k] = mask;
                }
            }
            out
        };
        #[cfg(not(target_arch = "wasm32"))]
        let root_blocks: Vec<Vec<u8>> = (0..n_blocks).into_par_iter().map(build_root_block).collect();
        #[cfg(target_arch = "wasm32")]
        let root_blocks: Vec<Vec<u8>> = (0..n_blocks).map(build_root_block).collect();

        let mut root_mask = Vec::with_capacity(l);
        for rb in root_blocks {
            root_mask.extend_from_slice(&rb);
        }

        let eval_leaf = |i: usize| -> f64 {
            let row = &bitmasks[i * l..(i + 1) * l];
            let mut muts = 0usize;
            for s in 0..l {
                if row[s] != 15 && (row[s] & root_mask[s]) == 0 {
                    muts += 1;
                }
            }
            branch_mutation_code_nats(muts as f64, l_eff, 0.0, 4)
        };
        #[cfg(not(target_arch = "wasm32"))]
        let leaf_sum: f64 = (0..n).into_par_iter().map(eval_leaf).sum();
        #[cfg(target_arch = "wasm32")]
        let leaf_sum: f64 = (0..n).map(eval_leaf).sum();

        l_eff * 4.0_f64.ln() + leaf_sum
    }

    fn compute_tree_total_mdl_nats(
        &self,
        adj: &BTreeMap<usize, Vec<usize>>,
        branches: &[BranchRecord],
        l_eff: f64,
    ) -> f64 {
        let mut total_nats = l_eff * 4.0_f64.ln();
        for b in branches {
            total_nats += branch_mutation_code_nats(b.mutation_count, l_eff, 0.0, 4);
            if b.is_internal {
                let deg_u = adj[&b.u].len().saturating_sub(1);
                let deg_v = adj[&b.v].len().saturating_sub(1);
                total_nats += split_topology_cost_nats(deg_u, deg_v);
            }
        }
        total_nats
    }

    fn serialize_newick(
        &self,
        adj: &BTreeMap<usize, Vec<usize>>,
        names: &[String],
        branches: &[BranchRecord],
        n: usize,
    ) -> String {
        if n == 1 {
            return format!("{}:0.0;", names[0]);
        }
        if n == 2 {
            let b_len = branches.first().map(|b| b.branch_length).unwrap_or(0.0);
            return format!("({}:{:.6},{}:{:.6});", names[0], b_len / 2.0, names[1], b_len / 2.0);
        }

        let mut edge_map: HashMap<(usize, usize), &BranchRecord> = HashMap::new();
        for b in branches {
            edge_map.insert((b.u.min(b.v), b.u.max(b.v)), b);
        }

        let root = adj
            .iter()
            .filter(|(&u, nbs)| u >= n && !nbs.is_empty())
            .max_by_key(|(_, nbs)| nbs.len())
            .map(|(&u, _)| u)
            .unwrap_or(n);

        let mut visited: HashSet<usize> = HashSet::new();

        fn build_sub(
            u: usize,
            p: Option<usize>,
            adj: &BTreeMap<usize, Vec<usize>>,
            names: &[String],
            edge_map: &HashMap<(usize, usize), &BranchRecord>,
            visited: &mut HashSet<usize>,
            n: usize,
        ) -> String {
            visited.insert(u);
            let label = if u < n {
                names[u].clone()
            } else {
                let children: Vec<usize> = adj[&u]
                    .iter()
                    .copied()
                    .filter(|&nb| Some(nb) != p && !visited.contains(&nb))
                    .collect();
                let sub_strs: Vec<String> = children
                    .into_iter()
                    .map(|c| build_sub(c, Some(u), adj, names, edge_map, visited, n))
                    .collect();
                format!("({})", sub_strs.join(","))
            };
            if let Some(parent_node) = p {
                let rec = edge_map.get(&(u.min(parent_node), u.max(parent_node)));
                let bl = rec.map(|r| r.branch_length).unwrap_or(0.0);
                if u >= n {
                    if let Some(r) = rec {
                        return format!("{label}{:.3}:{bl:.6}", r.mdl_support);
                    }
                }
                format!("{label}:{bl:.6}")
            } else {
                format!("{label};")
            }
        }

        build_sub(root, None, adj, names, &edge_map, &mut visited, n)
    }

    #[allow(clippy::too_many_arguments)]
    fn serialize_binary_newick(
        &self,
        bin_adj: &BTreeMap<usize, Vec<usize>>,
        names: &[String],
        bitmasks: &[u8],
        majority_state: &[i8],
        terminal_loads: &[f64],
        n: usize,
        l: usize,
        l_eff: f64,
        gtr_edge_stats: Option<&HashMap<(usize, usize), (f64, f64)>>,
    ) -> String {
        if n == 1 {
            return format!("{}:0.000010;", names[0]);
        }
        let min_bl = (0.20 / l_eff.max(1.0)).clamp(1e-6, 1e-3);
        if n == 2 {
            return format!("({}:{min_bl:.6},{}:{min_bl:.6});", names[0], names[1]);
        }
        let mut edge_bl: HashMap<(usize, usize), f64> = HashMap::new();
        if let Some(stats) = gtr_edge_stats {
            let mut all_covered = true;
            for (&u, nbs) in bin_adj {
                for &v in nbs {
                    if u < v {
                        if let Some(&(_, tmle)) = stats.get(&(u, v)) {
                            edge_bl.insert((u, v), tmle.max(min_bl));
                        } else {
                            all_covered = false;
                            break;
                        }
                    }
                }
                if !all_covered {
                    break;
                }
            }
            if !all_covered {
                edge_bl.clear();
            }
        }
        if edge_bl.is_empty() {
            let states = self.compute_all_directed_fitch_states(
                bin_adj,
                bitmasks,
                n,
                l,
                terminal_loads,
                l_eff,
            );
            let (_, edge_mutations) = self.reconstruct_nodal_states_and_mutations(
                bin_adj,
                bitmasks,
                majority_state,
                &states,
                n,
                l,
            );
            for (&u, nbs) in bin_adj {
                for &v in nbs {
                    if u < v {
                        let m_cnt = edge_mutations
                            .get(&(u, v))
                            .map(|m| m.len() as f64)
                            .unwrap_or(0.0);
                        let bl = (m_cnt / l_eff.max(1.0)).max(min_bl);
                        edge_bl.insert((u, v), bl);
                    }
                }
            }
        }

        let root = bin_adj
            .iter()
            .filter(|(&u, nbs)| u >= n && !nbs.is_empty())
            .max_by_key(|(_, nbs)| nbs.len())
            .map(|(&u, _)| u)
            .unwrap_or(n);
        let mut visited: HashSet<usize> = HashSet::new();

        fn build_bin_sub(
            u: usize,
            p: Option<usize>,
            adj: &BTreeMap<usize, Vec<usize>>,
            names: &[String],
            edge_bl: &HashMap<(usize, usize), f64>,
            visited: &mut HashSet<usize>,
            n: usize,
            min_bl: f64,
        ) -> String {
            visited.insert(u);
            let label = if u < n {
                names[u].clone()
            } else {
                let children: Vec<usize> = adj[&u]
                    .iter()
                    .copied()
                    .filter(|&nb| Some(nb) != p && !visited.contains(&nb))
                    .collect();
                let sub_strs: Vec<String> = children
                    .into_iter()
                    .map(|c| build_bin_sub(c, Some(u), adj, names, edge_bl, visited, n, min_bl))
                    .collect();
                format!("({})", sub_strs.join(","))
            };
            if let Some(parent_node) = p {
                let bl = edge_bl
                    .get(&(u.min(parent_node), u.max(parent_node)))
                    .copied()
                    .unwrap_or(min_bl);
                format!("{label}:{bl:.6}")
            } else {
                format!("{label};")
            }
        }

        build_bin_sub(root, None, bin_adj, names, &edge_bl, &mut visited, n, min_bl)
    }

    pub fn parse_newick_to_unrooted_adj(
        newick_str: &str,
        names: &[String],
    ) -> Result<(BTreeMap<usize, Vec<usize>>, BTreeMap<usize, String>), String> {
        let n = names.len();
        let mut name_to_idx: HashMap<String, usize> = HashMap::with_capacity(n);
        for (i, nm) in names.iter().enumerate() {
            name_to_idx.insert(nm.trim().to_string(), i);
        }
        let s = newick_str.trim().trim_end_matches(';').trim();
        let bytes = s.as_bytes();
        let mut pos = 0usize;
        let mut next_id = n;
        let mut children_map: BTreeMap<usize, Vec<usize>> = BTreeMap::new();

        fn skip_ws(bytes: &[u8], pos: &mut usize) {
            while *pos < bytes.len() && bytes[*pos].is_ascii_whitespace() {
                *pos += 1;
            }
        }

        fn parse_node(
            bytes: &[u8],
            pos: &mut usize,
            next_id: &mut usize,
            name_to_idx: &HashMap<String, usize>,
            children_map: &mut BTreeMap<usize, Vec<usize>>,
        ) -> Result<usize, String> {
            skip_ws(bytes, pos);
            let my_id = if *pos < bytes.len() && bytes[*pos] == b'(' {
                *pos += 1;
                let id = *next_id;
                *next_id += 1;
                children_map.insert(id, Vec::new());
                loop {
                    let ch_id = parse_node(bytes, pos, next_id, name_to_idx, children_map)?;
                    children_map.get_mut(&id).unwrap().push(ch_id);
                    skip_ws(bytes, pos);
                    if *pos < bytes.len() && bytes[*pos] == b',' {
                        *pos += 1;
                    } else if *pos < bytes.len() && bytes[*pos] == b')' {
                        *pos += 1;
                        break;
                    } else {
                        break;
                    }
                }
                while *pos < bytes.len() && !matches!(bytes[*pos], b':' | b',' | b')' | b';') {
                    *pos += 1;
                }
                id
            } else {
                let start = *pos;
                while *pos < bytes.len() && !matches!(bytes[*pos], b':' | b',' | b')' | b';') {
                    *pos += 1;
                }
                let raw_name = std::str::from_utf8(&bytes[start..*pos])
                    .map_err(|e| e.to_string())?
                    .trim()
                    .trim_matches('\'')
                    .trim_matches('"');
                *name_to_idx
                    .get(raw_name)
                    .ok_or_else(|| format!("Unknown taxon '{}' in input Newick", raw_name))?
            };

            skip_ws(bytes, pos);
            if *pos < bytes.len() && bytes[*pos] == b':' {
                *pos += 1;
                while *pos < bytes.len() && !matches!(bytes[*pos], b',' | b')' | b';') {
                    *pos += 1;
                }
            }
            Ok(my_id)
        }

        let _root = parse_node(bytes, &mut pos, &mut next_id, &name_to_idx, &mut children_map)?;

        let mut adj: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for i in 0..n {
            adj.insert(i, Vec::new());
        }
        for (&u, chs) in &children_map {
            adj.entry(u).or_default();
            for &c in chs {
                adj.entry(u).or_default().push(c);
                adj.entry(c).or_default().push(u);
            }
        }

        // Smooth out any degree-2 internal nodes (such as a rooted binary root)
        loop {
            let deg2 = adj
                .iter()
                .find(|(&u, nbs)| u >= n && nbs.len() == 2)
                .map(|(&u, nbs)| (u, nbs[0], nbs[1]));
            let Some((u, a, b)) = deg2 else {
                break;
            };
            adj.remove(&u);
            if let Some(na) = adj.get_mut(&a) {
                na.retain(|&x| x != u);
                if !na.contains(&b) {
                    na.push(b);
                }
            }
            if let Some(nb) = adj.get_mut(&b) {
                nb.retain(|&x| x != u);
                if !nb.contains(&a) {
                    nb.push(a);
                }
            }
        }

        let mut node_labels = BTreeMap::new();
        for (i, nm) in names.iter().enumerate() {
            node_labels.insert(i, nm.clone());
        }
        for &u in adj.keys() {
            if u >= n {
                node_labels.insert(u, format!("N{u}"));
            }
        }
        Ok((adj, node_labels))
    }

    pub fn infer_with_starting_newick(
        &self,
        names: &[String],
        seqs: &[String],
        input_newick: &str,
        expand_hubs: bool,
    ) -> Result<BushwhackerResult, String> {
        if names.is_empty() {
            return Err("Empty alignment provided.".to_string());
        }
        if let Some(d_mode) = self.discrete_mode {
            let cm = discrete::encode_discrete_sequences(names, seqs)?;
            return discrete::infer_discrete_mdl(
                self,
                &cm,
                d_mode,
                Some((input_newick, expand_hubs)),
            );
        }
        if self.translate_coding && !protein::is_protein_alignment(seqs) {
            let aa_seqs = protein::translate_coding_alignment(seqs)?;
            return protein::infer_protein_mdl(
                self,
                names,
                &aa_seqs,
                Some((input_newick, expand_hubs)),
            );
        }
        if self.force_protein || protein::is_protein_alignment(seqs) {
            return protein::infer_protein_mdl(
                self,
                names,
                seqs,
                Some((input_newick, expand_hubs)),
            );
        }
        #[cfg(not(target_arch = "wasm32"))]
        let t0 = Instant::now();
        let n = names.len();
        let (mat, bitmasks, majority_state, _, l) = encode_nucleotide_alignment(seqs);
        let total_valid: usize = mat.iter().filter(|&&x| x >= 0).count();
        let l_eff = ((total_valid as f64) / (n as f64)).clamp(1.0, l as f64);

        let (ti_cost, tv_cost) = self.estimate_subst_surprise(&mat, n, l);
        let (raw_dist_w, raw_tree_w, is_coding) = if self.coding_aware {
            detect_coding_frame_and_weights(&mat, n, l)
        } else {
            (vec![1.0; l], vec![1.0; l], false)
        };
        let (dw_h0, tw_h0) =
            self.compute_dual_site_coherence_weights(&mat, n, l, &vec![1.0; l], &vec![1.0; l]);
        let (dist_weights, tree_weights) = if self.coding_aware && is_coding && n >= 5 {
            let (dw_h1, tw_h1) =
                self.compute_dual_site_coherence_weights(&mat, n, l, &raw_dist_w, &raw_tree_w);
            if self.select_coding_weights_by_mdl(
                names,
                &mat,
                &bitmasks,
                &majority_state,
                &dw_h0,
                &tw_h0,
                &dw_h1,
                &tw_h1,
                n,
                l,
                l_eff,
                ti_cost,
                tv_cost,
            ) {
                (dw_h1, tw_h1)
            } else {
                (dw_h0, tw_h0)
            }
        } else {
            (dw_h0, tw_h0)
        };

        let (terminal_loads, _, _) = self.build_subquadratic_loads_and_backbone(
            names,
            &mat,
            &bitmasks,
            &dist_weights,
            &tree_weights,
            n,
            l,
            l_eff,
            ti_cost,
            tv_cost,
            false,
        );
        let (mut adj, node_labels) = Self::parse_newick_to_unrooted_adj(input_newick, names)?;
        let pre_step3_adj = adj.clone();

        let (adj_c, mut branches, _, edge_mutations) = self.contract_unsupported_edges_mdl(
            adj,
            &bitmasks,
            &majority_state,
            &terminal_loads,
            n,
            l,
            l_eff,
            ti_cost,
            tv_cost,
            &tree_weights,
            &mut node_labels.clone(),
            expand_hubs,
            None,
            true,
        );
        adj = adj_c;

        let bin_adj = self.build_binary_refinement_from_mdl(
            &adj,
            &pre_step3_adj,
            &bitmasks,
            &terminal_loads,
            n,
            l,
            l_eff,
            ti_cost,
            tv_cost,
            &tree_weights,
        );
        let binary_newick = self.serialize_binary_newick(
            &bin_adj,
            names,
            &bitmasks,
            &majority_state,
            &terminal_loads,
            n,
            l,
            l_eff,
            None,
        );
        let mut full_labels = node_labels;
        for &u in adj.keys() {
            full_labels.entry(u).or_insert_with(|| format!("N{u}"));
        }

        let mut shunts = Vec::new();
        if self.detect_shunts && n >= 4 {
            shunts = self.peel_residual_shunts_mdl(
                &adj,
                &full_labels,
                &bitmasks,
                &majority_state,
                &terminal_loads,
                &edge_mutations,
                n,
                l,
                l_eff,
                ti_cost,
                tv_cost,
                &mat,
                is_coding,
            );
            let int_shunts: Vec<&ShuntEdge> = shunts
                .iter()
                .filter(|s| {
                    s.mode == "INTERVAL" && s.interval_span.is_some() && s.net_mdl_gain_nats >= 15.0
                })
                .collect();
            if !int_shunts.is_empty() {
                let mut masked_bitmasks = bitmasks.clone();
                let mut masked_any = false;
                for sh in int_shunts {
                    if let Some((a_sp, b_sp)) = sh.interval_span {
                        let rec_node = if sh.node_u < n {
                            Some(sh.node_u)
                        } else if sh.node_v < n {
                            Some(sh.node_v)
                        } else {
                            None
                        };
                        if let Some(r_idx) = rec_node {
                            let row = &mut masked_bitmasks[r_idx * l..(r_idx + 1) * l];
                            for s_i in a_sp..=b_sp.min(l - 1) {
                                row[s_i] = 15;
                            }
                            masked_any = true;
                        }
                    }
                }
                if masked_any {
                    let (adj_re, branches_re, _, _) = self.contract_unsupported_edges_mdl(
                        adj,
                        &masked_bitmasks,
                        &majority_state,
                        &terminal_loads,
                        n,
                        l,
                        l_eff,
                        ti_cost,
                        tv_cost,
                        &tree_weights,
                        &mut full_labels,
                        false,
                        None,
                        true,
                    );
                    adj = adj_re;
                    branches = branches_re;
                }
            }
        }

        let raw_nats = (n as f64) * l_eff * 4.0_f64.ln();
        let star_nats = self.compute_star_tree_mdl_nats(&bitmasks, n, l, l_eff);
        let backbone_nats = self.compute_tree_total_mdl_nats(&adj, &branches, l_eff);
        let shunt_gain_total: f64 = shunts.iter().map(|s| s.net_mdl_gain_nats).sum();
        let full_graph_nats = (l_eff * 4.0_f64.ln()).max(backbone_nats - shunt_gain_total);

        let mut polytomy_degrees = Vec::new();
        for (&u, nbs) in &adj {
            if u >= n && nbs.len() > 3 {
                polytomy_degrees.push(nbs.len());
            }
        }
        let n_internal_branches = branches.iter().filter(|b| b.is_internal).count();
        let max_binary_internal_branches = n.saturating_sub(3);
        let max_polytomy_degree = polytomy_degrees
            .iter()
            .copied()
            .max()
            .unwrap_or(if n >= 3 { 3 } else { n });

        let newick = self.serialize_newick(&adj, names, &branches, n);
        #[cfg(not(target_arch = "wasm32"))]
        let runtime_sec = t0.elapsed().as_secs_f64();
        #[cfg(target_arch = "wasm32")]
        let runtime_sec = 0.0;

        Ok(BushwhackerResult {
            names: names.to_vec(),
            n_taxa: n,
            n_sites: l,
            l_eff,
            newick,
            binary_newick,
            adjacency: adj,
            node_labels: full_labels,
            branches,
            shunts,
            n_internal_branches,
            max_binary_internal_branches,
            polytomy_nodes: polytomy_degrees.len(),
            max_polytomy_degree,
            raw_uncompressed_nats: raw_nats,
            star_tree_nats: star_nats,
            backbone_tree_nats: backbone_nats,
            full_graph_nats,
            runtime_sec,
            site_weights: tree_weights,
            terminal_loads,
        })
    }
}

// ---------------------------------------------------------------------------
// Lossless Phylogenetic Alignment Codec (.bwz) via Causal Arithmetic Coding
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BwzCompressionStats {
    pub n_taxa: usize,
    pub n_sites: usize,
    pub raw_seq_bytes: usize,
    pub compressed_seq_bytes: usize,
    pub compressed_header_bytes: usize,
    pub total_bwz_bytes: usize,
    pub seq_mdl_nats: f64,
    pub total_bwz_nats: f64,
}

struct ArithEncoder {
    low: u64,
    high: u64,
    pending: usize,
    bits: Vec<u8>,
}

impl ArithEncoder {
    fn new() -> Self {
        Self {
            low: 0,
            high: 0xFFFF_FFFF,
            pending: 0,
            bits: Vec::with_capacity(65536),
        }
    }

    #[inline]
    fn put_bit(&mut self, b: u8) {
        self.bits.push(b);
        while self.pending > 0 {
            self.bits.push(1 - b);
            self.pending -= 1;
        }
    }

    #[inline]
    fn encode_bit(&mut self, bit: u8, p1: u32) {
        let p1_clamped = p1.clamp(1, 65535) as u64;
        let span = self.high - self.low + 1;
        let mid = self.low + ((span * (65536 - p1_clamped)) >> 16) - 1;
        if bit == 0 {
            self.high = mid;
        } else {
            self.low = mid + 1;
        }
        loop {
            if self.high < 0x8000_0000 {
                self.put_bit(0);
            } else if self.low >= 0x8000_0000 {
                self.put_bit(1);
                self.low -= 0x8000_0000;
                self.high -= 0x8000_0000;
            } else if self.low >= 0x4000_0000 && self.high < 0xC000_0000 {
                self.pending += 1;
                self.low -= 0x4000_0000;
                self.high -= 0x4000_0000;
            } else {
                break;
            }
            self.low = (self.low << 1) & 0xFFFF_FFFF;
            self.high = ((self.high << 1) | 1) & 0xFFFF_FFFF;
        }
    }

    fn encode_int_elias(&mut self, m: usize, p_zero: u32) {
        if m == 0 {
            self.encode_bit(0, 65536 - p_zero.clamp(1, 65535));
            return;
        }
        self.encode_bit(1, 65536 - p_zero.clamp(1, 65535));
        let mut blocks: Vec<(usize, usize)> = Vec::new();
        let mut v = m;
        while v > 1 {
            let bit_len = (usize::BITS - v.leading_zeros()) as usize;
            blocks.push((v, bit_len));
            v = bit_len - 1;
        }
        for &(val, bit_len) in blocks.iter().rev() {
            self.encode_bit(1, 32768);
            for shift in (0..(bit_len - 1)).rev() {
                let b = ((val >> shift) & 1) as u8;
                self.encode_bit(b, 32768);
            }
        }
        self.encode_bit(0, 32768);
    }

    fn encode_categorical(&mut self, sym_idx: usize, weights: &[u32]) {
        let mut rem: u64 = weights.iter().map(|&w| w as u64).sum();
        for (i, &w) in weights.iter().enumerate().take(weights.len().saturating_sub(1)) {
            if rem == 0 {
                break;
            }
            let w64 = w as u64;
            if w64 == 0 {
                continue;
            }
            if w64 >= rem {
                break;
            }
            let p_other = (((65536.0 * ((rem - w64) as f64)) / (rem as f64)).round() as u32)
                .clamp(1, 65535);
            if sym_idx == i {
                self.encode_bit(0, p_other);
                return;
            } else {
                self.encode_bit(1, p_other);
                rem -= w64;
            }
        }
    }

    fn finish(mut self) -> Vec<u8> {
        self.pending += 1;
        if self.low < 0x4000_0000 {
            self.put_bit(0);
        } else {
            self.put_bit(1);
        }
        let mut out = vec![0_u8; (self.bits.len() + 7) / 8];
        for (i, &b) in self.bits.iter().enumerate() {
            if b != 0 {
                out[i >> 3] |= 1_u8 << (7 - (i & 7));
            }
        }
        out
    }
}

struct ArithDecoder<'a> {
    data: &'a [u8],
    bit_pos: usize,
    low: u64,
    high: u64,
    val: u64,
}

impl<'a> ArithDecoder<'a> {
    fn new(data: &'a [u8]) -> Self {
        let mut dec = Self {
            data,
            bit_pos: 0,
            low: 0,
            high: 0xFFFF_FFFF,
            val: 0,
        };
        for _ in 0..32 {
            dec.val = (dec.val << 1) | (dec.get_bit() as u64);
        }
        dec
    }

    #[inline]
    fn get_bit(&mut self) -> u8 {
        let byte_idx = self.bit_pos >> 3;
        if byte_idx >= self.data.len() {
            return 0;
        }
        let b = (self.data[byte_idx] >> (7 - (self.bit_pos & 7))) & 1;
        self.bit_pos += 1;
        b
    }

    #[inline]
    fn decode_bit(&mut self, p1: u32) -> u8 {
        let p1_clamped = p1.clamp(1, 65535) as u64;
        let span = self.high - self.low + 1;
        let mid = self.low + ((span * (65536 - p1_clamped)) >> 16) - 1;
        let bit = if self.val <= mid {
            self.high = mid;
            0
        } else {
            self.low = mid + 1;
            1
        };
        loop {
            if self.high < 0x8000_0000 {
                // E1 scaling
            } else if self.low >= 0x8000_0000 {
                self.val -= 0x8000_0000;
                self.low -= 0x8000_0000;
                self.high -= 0x8000_0000;
            } else if self.low >= 0x4000_0000 && self.high < 0xC000_0000 {
                self.val -= 0x4000_0000;
                self.low -= 0x4000_0000;
                self.high -= 0x4000_0000;
            } else {
                break;
            }
            self.low = (self.low << 1) & 0xFFFF_FFFF;
            self.high = ((self.high << 1) | 1) & 0xFFFF_FFFF;
            self.val = ((self.val << 1) | (self.get_bit() as u64)) & 0xFFFF_FFFF;
        }
        bit
    }

    fn decode_int_elias(&mut self, p_zero: u32) -> usize {
        if self.decode_bit(65536 - p_zero.clamp(1, 65535)) == 0 {
            return 0;
        }
        let mut n = 1usize;
        let mut steps = 0usize;
        while steps < 16 && self.decode_bit(32768) == 1 {
            steps += 1;
            let len = n.min(60);
            let mut v = 1usize;
            for _ in 0..len {
                v = (v << 1) | (self.decode_bit(32768) as usize);
            }
            n = v;
        }
        n
    }

    fn decode_categorical(&mut self, weights: &[u32]) -> usize {
        let mut rem: u64 = weights.iter().map(|&w| w as u64).sum();
        let mut last_valid = 0usize;
        for (i, &w) in weights.iter().enumerate() {
            if w > 0 {
                last_valid = i;
            }
        }
        for (i, &w) in weights.iter().enumerate().take(weights.len().saturating_sub(1)) {
            if rem == 0 {
                break;
            }
            let w64 = w as u64;
            if w64 == 0 {
                continue;
            }
            if w64 >= rem {
                return i;
            }
            let p_other = (((65536.0 * ((rem - w64) as f64)) / (rem as f64)).round() as u32)
                .clamp(1, 65535);
            if self.decode_bit(p_other) == 0 {
                return i;
            }
            rem -= w64;
        }
        last_valid
    }
}

fn arith_compress_bytes_order0(input: &[u8]) -> Vec<u8> {
    let mut enc = ArithEncoder::new();
    let mut match_p1 = 16384_u32;
    let mut ctx_counts = vec![[1_u32; 256]; 256];
    let mut prev = 0_usize;
    let mut pos = 0_usize;
    while pos < input.len() {
        let win_start = pos.saturating_sub(32767);
        let max_len = (input.len() - pos).min(512);
        let mut best_len = 0_usize;
        let mut best_dist = 1_usize;
        if max_len >= 4 && pos > win_start {
            for cand in (win_start..pos).rev() {
                if input[cand] == input[pos] && input[cand + 1] == input[pos + 1] {
                    let mut l = 2_usize;
                    while l < max_len && input[cand + l] == input[pos + l] {
                        l += 1;
                    }
                    if l > best_len {
                        best_len = l;
                        best_dist = pos - cand;
                        if best_len == max_len {
                            break;
                        }
                    }
                }
            }
        }
        if best_len >= 4 {
            enc.encode_bit(1, match_p1);
            match_p1 = ((match_p1 * 31 + 65535) >> 5).clamp(1, 65535);
            enc.encode_int_elias(best_len - 4, 32768);
            enc.encode_int_elias(best_dist - 1, 32768);
            prev = input[pos + best_len - 1] as usize;
            pos += best_len;
        } else {
            enc.encode_bit(0, match_p1);
            match_p1 = ((match_p1 * 31) >> 5).clamp(1, 65535);
            let b = input[pos] as usize;
            enc.encode_categorical(b, &ctx_counts[prev]);
            ctx_counts[prev][b] += 4;
            prev = b;
            pos += 1;
        }
    }
    enc.finish()
}

fn arith_decompress_bytes_order0(compressed: &[u8], out_len: usize) -> Vec<u8> {
    let mut dec = ArithDecoder::new(compressed);
    let mut match_p1 = 16384_u32;
    let mut ctx_counts = vec![[1_u32; 256]; 256];
    let mut prev = 0_usize;
    let mut out = Vec::with_capacity(out_len);
    while out.len() < out_len {
        let is_match = dec.decode_bit(match_p1) != 0;
        if is_match {
            match_p1 = ((match_p1 * 31 + 65535) >> 5).clamp(1, 65535);
            let len = dec.decode_int_elias(32768) + 4;
            let dist = dec.decode_int_elias(32768) + 1;
            let start = out.len().saturating_sub(dist);
            for i in 0..len {
                if out.len() >= out_len {
                    break;
                }
                let b = out[start + i];
                out.push(b);
                prev = b as usize;
            }
        } else {
            match_p1 = ((match_p1 * 31) >> 5).clamp(1, 65535);
            let sym = dec.decode_categorical(&ctx_counts[prev]);
            out.push(sym as u8);
            ctx_counts[prev][sym] += 4;
            prev = sym;
        }
    }
    out
}

/// Compress an alignment (`names`, `seqs`) and its induced `BushwhackerResult`
/// into a self-contained, bit-for-bit lossless `.bwz` phylogenetic archive.
pub fn compress_alignment_to_bwz(
    names: &[String],
    seqs: &[String],
    res: &BushwhackerResult,
) -> Result<(Vec<u8>, BwzCompressionStats), String> {
    let n = names.len();
    if n == 0 || seqs.len() != n {
        return Err("Invalid alignment dimensions for .bwz compression".to_string());
    }
    let l = seqs[0].len();
    let leaf_seqs: Vec<&[u8]> = seqs.iter().map(|s| s.as_bytes()).collect();

    // 1. Build preorder tree traversal from `res.adjacency`
    let adj = &res.adjacency;
    let root = adj
        .iter()
        .filter(|(&u, nbs)| u >= n && !nbs.is_empty())
        .max_by_key(|(_, nbs)| nbs.len())
        .map(|(&u, _)| u)
        .unwrap_or(0);

    let mut parent_map: HashMap<usize, Option<usize>> = HashMap::new();
    parent_map.insert(root, None);
    let mut order: Vec<usize> = vec![root];
    let mut q: VecDeque<usize> = VecDeque::from([root]);
    while let Some(u) = q.pop_front() {
        if let Some(nbs) = adj.get(&u) {
            for &v in nbs {
                if !parent_map.contains_key(&v) {
                    parent_map.insert(v, Some(u));
                    order.push(v);
                    q.push_back(v);
                }
            }
        }
    }
    // Include any disconnected leaves if present
    for leaf_i in 0..n {
        if !parent_map.contains_key(&leaf_i) {
            parent_map.insert(leaf_i, Some(root));
            order.push(leaf_i);
        }
    }

    let n_nodes = order.len();
    let mut node_to_pidx: HashMap<usize, usize> = HashMap::with_capacity(n_nodes);
    for (pidx, &u) in order.iter().enumerate() {
        node_to_pidx.insert(u, pidx);
    }

    let mut children_pidx: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
    let mut parent_pidx: Vec<usize> = vec![0; n_nodes];
    for (pidx, &u) in order.iter().enumerate().skip(1) {
        let p_node = parent_map[&u].unwrap_or(root);
        let p_i = node_to_pidx[&p_node];
        parent_pidx[pidx] = p_i;
        children_pidx[p_i].push(pidx);
    }

    // 2. Two-pass Fitch parsimony reconstruction over full ASCII byte alphabet
    let mut alphabet_set: BTreeSet<u8> = BTreeSet::new();
    for &ls in &leaf_seqs {
        for &b in ls {
            alphabet_set.insert(b);
        }
    }
    let alphabet: Vec<u8> = alphabet_set.into_iter().collect();
    let k_alpha = alphabet.len();
    let mut alpha_idx = [0_usize; 256];
    for (i, &b) in alphabet.iter().enumerate() {
        alpha_idx[b as usize] = i;
    }

    let mut node_seqs: Vec<Vec<u8>> = vec![vec![0_u8; l]; n_nodes];
    let mut gap_mask: Vec<Vec<bool>> = vec![vec![false; l]; n_nodes];
    let mut n_mask: Vec<Vec<bool>> = vec![vec![false; l]; n_nodes];
    for (pidx, &u) in order.iter().enumerate() {
        if u < n {
            node_seqs[pidx].copy_from_slice(leaf_seqs[u]);
            for s in 0..l {
                gap_mask[pidx][s] = leaf_seqs[u][s] == b'-';
                n_mask[pidx][s] = leaf_seqs[u][s] == b'N';
            }
        }
    }

    let gap_sym_opt = alphabet.iter().position(|&x| x == b'-');
    let n_sym_opt = alphabet.iter().position(|&x| x == b'N');
    let first_nongap_idx = (0..k_alpha)
        .find(|&i| Some(i) != gap_sym_opt && Some(i) != n_sym_opt)
        .unwrap_or(0);
    let mut all_nongap_mask = 0_u64;
    for i in 0..k_alpha.min(64) {
        if Some(i) != gap_sym_opt && Some(i) != n_sym_opt {
            all_nongap_mask |= 1_u64 << i;
        }
    }
    if all_nongap_mask == 0 {
        all_nongap_mask = 1;
    }

    let mut f_masks: Vec<u64> = vec![0_u64; n_nodes];
    let mut g_masks: Vec<u8> = vec![0_u8; n_nodes];
    let mut nm_masks: Vec<u8> = vec![0_u8; n_nodes];
    for s in 0..l {
        for pidx in (0..n_nodes).rev() {
            let u = order[pidx];
            if u < n {
                let b = leaf_seqs[u][s];
                if b == b'-' {
                    g_masks[pidx] = 2;
                    nm_masks[pidx] = 1;
                    f_masks[pidx] = all_nongap_mask;
                } else if b == b'N' {
                    g_masks[pidx] = 1;
                    nm_masks[pidx] = 2;
                    f_masks[pidx] = all_nongap_mask;
                } else {
                    g_masks[pidx] = 1;
                    nm_masks[pidx] = 1;
                    let sym_i = alpha_idx[b as usize];
                    f_masks[pidx] = 1_u64 << sym_i;
                }
            } else {
                let ch = &children_pidx[pidx];
                if ch.is_empty() {
                    g_masks[pidx] = 1;
                    nm_masks[pidx] = 1;
                    f_masks[pidx] = 1_u64 << first_nongap_idx;
                } else {
                    let mut g_cnt = [0_u16; 2];
                    let mut nm_cnt = [0_u16; 2];
                    let mut counts = [0_u16; 64];
                    let mut max_c = 0_u16;
                    for &c_i in ch {
                        let gm = g_masks[c_i];
                        if (gm & 1) != 0 {
                            g_cnt[0] += 1;
                        }
                        if (gm & 2) != 0 {
                            g_cnt[1] += 1;
                        }
                        let nmm = nm_masks[c_i];
                        if (nmm & 1) != 0 {
                            nm_cnt[0] += 1;
                        }
                        if (nmm & 2) != 0 {
                            nm_cnt[1] += 1;
                        }
                        let m = f_masks[c_i];
                        let mut bits = m;
                        while bits != 0 {
                            let b_idx = bits.trailing_zeros() as usize;
                            counts[b_idx] += 1;
                            if counts[b_idx] > max_c {
                                max_c = counts[b_idx];
                            }
                            bits &= bits - 1;
                        }
                    }
                    g_masks[pidx] = if g_cnt[0] > g_cnt[1] {
                        1
                    } else if g_cnt[1] > g_cnt[0] {
                        2
                    } else {
                        3
                    };
                    nm_masks[pidx] = if nm_cnt[0] > nm_cnt[1] {
                        1
                    } else if nm_cnt[1] > nm_cnt[0] {
                        2
                    } else {
                        3
                    };
                    let mut win_mask = 0_u64;
                    for b_idx in 0..k_alpha.min(64) {
                        if counts[b_idx] == max_c {
                            win_mask |= 1_u64 << b_idx;
                        }
                    }
                    f_masks[pidx] = if win_mask != 0 {
                        win_mask
                    } else {
                        1_u64 << first_nongap_idx
                    };
                }
            }
        }
        for pidx in 0..n_nodes {
            let u = order[pidx];
            if u >= n {
                let gm = g_masks[pidx];
                if pidx > 0 {
                    let pg = gap_mask[parent_pidx[pidx]][s];
                    let p_bit = if pg { 2 } else { 1 };
                    if (gm & p_bit) != 0 {
                        gap_mask[pidx][s] = pg;
                    } else {
                        gap_mask[pidx][s] = (gm & 2) != 0;
                    }
                } else {
                    gap_mask[pidx][s] = gm == 2;
                }
                let nmm = nm_masks[pidx];
                if pidx > 0 {
                    let pn = n_mask[parent_pidx[pidx]][s];
                    let p_bit = if pn { 2 } else { 1 };
                    if (nmm & p_bit) != 0 {
                        n_mask[pidx][s] = pn;
                    } else {
                        n_mask[pidx][s] = (nmm & 2) != 0;
                    }
                } else {
                    n_mask[pidx][s] = nmm == 2;
                }
                if gap_mask[pidx][s] {
                    n_mask[pidx][s] = false;
                }
            }
            let is_masked = gap_mask[pidx][s] || n_mask[pidx][s];
            if u < n && !is_masked {
                continue;
            }
            if u < n && is_masked {
                node_seqs[pidx][s] = if pidx > 0 {
                    node_seqs[parent_pidx[pidx]][s]
                } else {
                    alphabet[first_nongap_idx]
                };
                continue;
            }
            let mask = f_masks[pidx];
            if pidx > 0 {
                let p_sym = alpha_idx[node_seqs[parent_pidx[pidx]][s] as usize];
                if (mask & (1_u64 << p_sym)) != 0 {
                    node_seqs[pidx][s] = alphabet[p_sym];
                    continue;
                }
            }
            let best_sym = mask.trailing_zeros() as usize;
            node_seqs[pidx][s] = alphabet[best_sym];
        }
    }

    // 3. Identify beneficial shunts in preorder
    let mut shunt_map: HashMap<usize, (usize, Vec<usize>)> = HashMap::new();
    for sh in &res.shunts {
        if let (Some(&idx_u), Some(&idx_v)) =
            (node_to_pidx.get(&sh.node_u), node_to_pidx.get(&sh.node_v))
        {
            let (donor_idx, rec_idx) = if idx_u < idx_v {
                (idx_u, idx_v)
            } else {
                (idx_v, idx_u)
            };
            if rec_idx > 0 && !shunt_map.contains_key(&rec_idx) {
                let valid_sites: Vec<usize> =
                    sh.sites.iter().copied().filter(|&s| s < l).collect();
                if !valid_sites.is_empty() {
                    shunt_map.insert(rec_idx, (donor_idx, valid_sites));
                }
            }
        }
    }

    // 4. Encode sequence payload via Causal Phylogenetic Arithmetic Coder
    let mut enc = ArithEncoder::new();
    let is_canon = |b: u8| matches!(b, b'A' | b'C' | b'G' | b'T' | b'U' | b'a' | b'c' | b'g' | b't' | b'u');
    let mut base_root_row = vec![1_u32; k_alpha];
    for (i, &b) in alphabet.iter().enumerate() {
        if is_canon(b) {
            base_root_row[i] = 32;
        }
    }
    if let Some(g_idx) = gap_sym_opt {
        base_root_row[g_idx] = 0;
    }
    if let Some(n_idx) = n_sym_opt {
        base_root_row[n_idx] = 0;
    }
    let mut root_ctx = vec![base_root_row; 16];
    let mut r_prev1 = 0_usize;
    let mut r_prev2 = 0_usize;
    for s in 0..l {
        let ctx = ((r_prev2 & 3) << 2) | (r_prev1 & 3);
        let sym = alpha_idx[node_seqs[0][s] as usize];
        enc.encode_categorical(sym, &root_ctx[ctx]);
        root_ctx[ctx][sym] += 16;
        r_prev2 = r_prev1;
        r_prev1 = sym;
    }

    let mut gap_start_cnt = [2_u64, (l as u64) * 2];
    let mut gap_cont_cnt = [20_u64, 4_u64];
    if gap_sym_opt.is_some() {
        let has_root_gap = gap_mask[0].iter().any(|&g| g);
        enc.encode_bit(u8::from(has_root_gap), 4096);
        if has_root_gap {
            let mut in_run = false;
            for s in 0..l {
                let dg = gap_mask[0][s];
                if !in_run {
                    let p1 = ((65536 * gap_start_cnt[0]) / (gap_start_cnt[0] + gap_start_cnt[1]))
                        .clamp(1, 65535) as u32;
                    enc.encode_bit(u8::from(dg), p1);
                    if dg {
                        gap_start_cnt[0] += 1;
                        in_run = true;
                    } else {
                        gap_start_cnt[1] += 1;
                    }
                } else {
                    let p1 = ((65536 * gap_cont_cnt[0]) / (gap_cont_cnt[0] + gap_cont_cnt[1]))
                        .clamp(1, 65535) as u32;
                    enc.encode_bit(u8::from(dg), p1);
                    if dg {
                        gap_cont_cnt[0] += 1;
                    } else {
                        gap_cont_cnt[1] += 1;
                        in_run = false;
                    }
                }
            }
        }
    }

    let mut n_start_cnt = [2_u64, (l as u64) * 2];
    let mut n_cont_cnt = [20_u64, 4_u64];
    if n_sym_opt.is_some() {
        let has_root_n = n_mask[0].iter().any(|&g| g);
        enc.encode_bit(u8::from(has_root_n), 4096);
        if has_root_n {
            let mut in_run = false;
            for s in 0..l {
                let dn = n_mask[0][s];
                if !in_run {
                    let p1 = ((65536 * n_start_cnt[0]) / (n_start_cnt[0] + n_start_cnt[1]))
                        .clamp(1, 65535) as u32;
                    enc.encode_bit(u8::from(dn), p1);
                    if dn {
                        n_start_cnt[0] += 1;
                        in_run = true;
                    } else {
                        n_start_cnt[1] += 1;
                    }
                } else {
                    let p1 = ((65536 * n_cont_cnt[0]) / (n_cont_cnt[0] + n_cont_cnt[1]))
                        .clamp(1, 65535) as u32;
                    enc.encode_bit(u8::from(dn), p1);
                    if dn {
                        n_cont_cnt[0] += 1;
                    } else {
                        n_cont_cnt[1] += 1;
                        in_run = false;
                    }
                }
            }
        }
    }

    let mut mut_hist = vec![0_u32; l];
    let mut codon_counts = [10_u32; 3];
    let mut trans_counts = vec![vec![1_u32; k_alpha]; k_alpha];
    for i in 0..k_alpha {
        for j in 0..k_alpha {
            if is_canon(alphabet[j]) {
                trans_counts[i][j] = 16;
            }
        }
        trans_counts[i][i] = 0;
        if let Some(g_idx) = gap_sym_opt {
            trans_counts[i][g_idx] = 0;
        }
        if let Some(n_idx) = n_sym_opt {
            trans_counts[i][n_idx] = 0;
        }
    }
    for &(b1, b2) in &[
        (b'A', b'G'),
        (b'G', b'A'),
        (b'C', b'T'),
        (b'T', b'C'),
        (b'a', b'g'),
        (b'g', b'a'),
        (b'c', b't'),
        (b't', b'c'),
    ] {
        if let (Some(i1), Some(i2)) = (
            alphabet.iter().position(|&x| x == b1),
            alphabet.iter().position(|&x| x == b2),
        ) {
            trans_counts[i1][i2] += 32;
        }
    }

    for pidx in 1..n_nodes {
        let p_i = parent_pidx[pidx];
        let mut pred = node_seqs[p_i].clone();
        if let Some((donor_i, sites)) = shunt_map.get(&pidx) {
            for &s in sites {
                pred[s] = node_seqs[*donor_i][s];
            }
        }
        for s in 0..l {
            if gap_mask[pidx][s] || n_mask[pidx][s] {
                node_seqs[pidx][s] = pred[s];
            }
        }
        let u_seq = &node_seqs[pidx];

        if gap_sym_opt.is_some() {
            let has_gap_diff = (0..l).any(|s| gap_mask[pidx][s] != gap_mask[p_i][s]);
            enc.encode_bit(u8::from(has_gap_diff), 4096);
            if has_gap_diff {
                let mut in_run = false;
                for s in 0..l {
                    let diff_g = gap_mask[pidx][s] != gap_mask[p_i][s];
                    if !in_run {
                        let p1 = ((65536 * gap_start_cnt[0]) / (gap_start_cnt[0] + gap_start_cnt[1]))
                            .clamp(1, 65535) as u32;
                        enc.encode_bit(u8::from(diff_g), p1);
                        if diff_g {
                            gap_start_cnt[0] += 1;
                            in_run = true;
                        } else {
                            gap_start_cnt[1] += 1;
                        }
                    } else {
                        let p1 = ((65536 * gap_cont_cnt[0]) / (gap_cont_cnt[0] + gap_cont_cnt[1]))
                            .clamp(1, 65535) as u32;
                        enc.encode_bit(u8::from(diff_g), p1);
                        if diff_g {
                            gap_cont_cnt[0] += 1;
                        } else {
                            gap_cont_cnt[1] += 1;
                            in_run = false;
                        }
                    }
                }
            }
        }

        if n_sym_opt.is_some() {
            let has_n_diff = (0..l).any(|s| n_mask[pidx][s] != n_mask[p_i][s]);
            enc.encode_bit(u8::from(has_n_diff), 4096);
            if has_n_diff {
                let mut in_run = false;
                for s in 0..l {
                    let diff_n = n_mask[pidx][s] != n_mask[p_i][s];
                    if !in_run {
                        let p1 = ((65536 * n_start_cnt[0]) / (n_start_cnt[0] + n_start_cnt[1]))
                            .clamp(1, 65535) as u32;
                        enc.encode_bit(u8::from(diff_n), p1);
                        if diff_n {
                            n_start_cnt[0] += 1;
                            in_run = true;
                        } else {
                            n_start_cnt[1] += 1;
                        }
                    } else {
                        let p1 = ((65536 * n_cont_cnt[0]) / (n_cont_cnt[0] + n_cont_cnt[1]))
                            .clamp(1, 65535) as u32;
                        enc.encode_bit(u8::from(diff_n), p1);
                        if diff_n {
                            n_cont_cnt[0] += 1;
                        } else {
                            n_cont_cnt[1] += 1;
                            in_run = false;
                        }
                    }
                }
            }
        }

        let valid_sites: Vec<usize> = (0..l)
            .filter(|&s| !gap_mask[pidx][s] && !n_mask[pidx][s])
            .collect();
        let mut_sites: Vec<usize> = valid_sites
            .iter()
            .copied()
            .filter(|&s| pred[s] != u_seq[s])
            .collect();
        let m_u = mut_sites.len();
        enc.encode_int_elias(m_u, 24000);
        if m_u > 0 {
            let mut_set: HashSet<usize> = mut_sites.into_iter().collect();
            let c_sum = (codon_counts[0] + codon_counts[1] + codon_counts[2]) as f64;
            let c_mult = [
                3.0 * (codon_counts[0] as f64) / c_sum,
                3.0 * (codon_counts[1] as f64) / c_sum,
                3.0 * (codon_counts[2] as f64) / c_sum,
            ];
            let weights: Vec<f64> = valid_sites
                .iter()
                .map(|&s| (0.20 + 2.8 * (mut_hist[s] as f64)) * c_mult[s % 3])
                .collect();
            let mut w_rem: f64 = weights.iter().sum();
            let mut m_rem = m_u;
            let mut n_rem = valid_sites.len();

            for (idx_s, &s) in valid_sites.iter().enumerate() {
                if m_rem == 0 {
                    break;
                }
                let is_mut = if m_rem == n_rem {
                    true
                } else {
                    let prob = ((m_rem as f64) * weights[idx_s] / w_rem.max(1e-9)).clamp(1e-6, 0.999);
                    let p1 = ((65536.0 * prob).round() as u32).clamp(1, 65535);
                    let flag = mut_set.contains(&s);
                    enc.encode_bit(u8::from(flag), p1);
                    flag
                };
                w_rem -= weights[idx_s];
                n_rem -= 1;
                if is_mut {
                    m_rem -= 1;
                    let b_from = alpha_idx[pred[s] as usize];
                    let b_to = alpha_idx[u_seq[s] as usize];
                    let mut w_sub = trans_counts[b_from].clone();
                    w_sub[b_from] = 0;
                    if let Some(g_idx) = gap_sym_opt {
                        w_sub[g_idx] = 0;
                    }
                    if let Some(n_idx) = n_sym_opt {
                        w_sub[n_idx] = 0;
                    }
                    enc.encode_categorical(b_to, &w_sub);
                    trans_counts[b_from][b_to] += 3;
                    mut_hist[s] += 1;
                    codon_counts[s % 3] += 1;
                }
            }
        }
    }

    let seq_payload = enc.finish();

    // 5. Serialize metadata header (taxon names, Newick string, preorder node table, and shunts)
    let mut meta_raw: Vec<u8> = Vec::new();
    let names_joined = names.join("\n");
    let names_bytes = names_joined.as_bytes();
    meta_raw.extend_from_slice(&(names_bytes.len() as u32).to_le_bytes());
    meta_raw.extend_from_slice(names_bytes);

    let nwk_bytes = res.newick.as_bytes();
    meta_raw.extend_from_slice(&(nwk_bytes.len() as u32).to_le_bytes());
    meta_raw.extend_from_slice(nwk_bytes);

    for pidx in 0..n_nodes {
        let u = order[pidx];
        let is_leaf = u < n;
        meta_raw.push(u8::from(is_leaf));
        meta_raw.extend_from_slice(&(if is_leaf { u as u32 } else { 0 }).to_le_bytes());
        meta_raw.extend_from_slice(&(parent_pidx[pidx] as u32).to_le_bytes());
    }

    meta_raw.extend_from_slice(&(shunt_map.len() as u16).to_le_bytes());
    for (&rec_idx, (donor_idx, sites)) in &shunt_map {
        meta_raw.extend_from_slice(&(rec_idx as u32).to_le_bytes());
        meta_raw.extend_from_slice(&(*donor_idx as u32).to_le_bytes());
        meta_raw.extend_from_slice(&(sites.len() as u32).to_le_bytes());
        for &s in sites {
            meta_raw.extend_from_slice(&(s as u32).to_le_bytes());
        }
    }

    let meta_compressed = arith_compress_bytes_order0(&meta_raw);

    // 6. Assemble final .bwz binary container
    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(b"BWZ1");
    out.extend_from_slice(&(n as u32).to_le_bytes());
    out.extend_from_slice(&(l as u32).to_le_bytes());
    out.extend_from_slice(&(n_nodes as u32).to_le_bytes());
    out.extend_from_slice(&(k_alpha as u16).to_le_bytes());
    out.extend_from_slice(&alphabet);
    out.extend_from_slice(&(meta_raw.len() as u32).to_le_bytes());
    out.extend_from_slice(&(meta_compressed.len() as u32).to_le_bytes());
    out.extend_from_slice(&meta_compressed);
    out.extend_from_slice(&(seq_payload.len() as u32).to_le_bytes());
    out.extend_from_slice(&seq_payload);

    let ln2 = std::f64::consts::LN_2;
    let stats = BwzCompressionStats {
        n_taxa: n,
        n_sites: l,
        raw_seq_bytes: n * l,
        compressed_seq_bytes: seq_payload.len(),
        compressed_header_bytes: out.len() - seq_payload.len(),
        total_bwz_bytes: out.len(),
        seq_mdl_nats: (seq_payload.len() as f64) * 8.0 * ln2,
        total_bwz_nats: (out.len() as f64) * 8.0 * ln2,
    };
    Ok((out, stats))
}

/// Decompress a `.bwz` phylogenetic archive back into `(names, seqs, newick)` bit-for-bit.
pub fn decompress_alignment_from_bwz(
    data: &[u8],
) -> Result<(Vec<String>, Vec<String>, String), String> {
    if data.len() < 26 || &data[0..4] != b"BWZ1" {
        return Err("Invalid .bwz file header (expected magic 'BWZ1')".to_string());
    }
    let mut pos = 4usize;
    let read_u32 = |buf: &[u8], p: &mut usize| -> Result<u32, String> {
        if *p + 4 > buf.len() {
            return Err("Unexpected EOF in .bwz header".to_string());
        }
        let v = u32::from_le_bytes([buf[*p], buf[*p + 1], buf[*p + 2], buf[*p + 3]]);
        *p += 4;
        Ok(v)
    };
    let read_u16 = |buf: &[u8], p: &mut usize| -> Result<u16, String> {
        if *p + 2 > buf.len() {
            return Err("Unexpected EOF in .bwz header".to_string());
        }
        let v = u16::from_le_bytes([buf[*p], buf[*p + 1]]);
        *p += 2;
        Ok(v)
    };

    let n = read_u32(data, &mut pos)? as usize;
    let l = read_u32(data, &mut pos)? as usize;
    let n_nodes = read_u32(data, &mut pos)? as usize;
    let k_alpha = read_u16(data, &mut pos)? as usize;
    if pos + k_alpha > data.len() {
        return Err("Unexpected EOF reading .bwz alphabet".to_string());
    }
    let alphabet: Vec<u8> = data[pos..pos + k_alpha].to_vec();
    pos += k_alpha;

    let mut alpha_idx = [0_usize; 256];
    for (i, &b) in alphabet.iter().enumerate() {
        alpha_idx[b as usize] = i;
    }

    let meta_raw_len = read_u32(data, &mut pos)? as usize;
    let meta_comp_len = read_u32(data, &mut pos)? as usize;
    if pos + meta_comp_len > data.len() {
        return Err("Unexpected EOF reading .bwz metadata block".to_string());
    }
    let meta_raw = arith_decompress_bytes_order0(&data[pos..pos + meta_comp_len], meta_raw_len);
    pos += meta_comp_len;

    let seq_comp_len = read_u32(data, &mut pos)? as usize;
    if pos + seq_comp_len > data.len() {
        return Err("Unexpected EOF reading .bwz sequence payload".to_string());
    }
    let seq_payload = &data[pos..pos + seq_comp_len];

    // Parse decompressed metadata
    let mut mpos = 0usize;
    let names_len = read_u32(&meta_raw, &mut mpos)? as usize;
    let names_str = String::from_utf8(meta_raw[mpos..mpos + names_len].to_vec())
        .map_err(|e| format!("UTF-8 error in .bwz taxon names: {e}"))?;
    mpos += names_len;
    let names: Vec<String> = names_str.split('\n').map(String::from).collect();

    let nwk_len = read_u32(&meta_raw, &mut mpos)? as usize;
    let newick = String::from_utf8(meta_raw[mpos..mpos + nwk_len].to_vec())
        .map_err(|e| format!("UTF-8 error in .bwz Newick string: {e}"))?;
    mpos += nwk_len;

    let mut is_leaf_vec = Vec::with_capacity(n_nodes);
    let mut leaf_id_vec = Vec::with_capacity(n_nodes);
    let mut parent_pidx = Vec::with_capacity(n_nodes);
    for _ in 0..n_nodes {
        let is_l = meta_raw[mpos] != 0;
        mpos += 1;
        let lid = read_u32(&meta_raw, &mut mpos)? as usize;
        let pid = read_u32(&meta_raw, &mut mpos)? as usize;
        is_leaf_vec.push(is_l);
        leaf_id_vec.push(lid);
        parent_pidx.push(pid);
    }

    let n_shunts = read_u16(&meta_raw, &mut mpos)? as usize;
    let mut shunt_map: HashMap<usize, (usize, Vec<usize>)> = HashMap::new();
    for _ in 0..n_shunts {
        let rec_idx = read_u32(&meta_raw, &mut mpos)? as usize;
        let donor_idx = read_u32(&meta_raw, &mut mpos)? as usize;
        let s_len = read_u32(&meta_raw, &mut mpos)? as usize;
        let mut sites = Vec::with_capacity(s_len);
        for _ in 0..s_len {
            sites.push(read_u32(&meta_raw, &mut mpos)? as usize);
        }
        shunt_map.insert(rec_idx, (donor_idx, sites));
    }

    // Decode sequence payload in lockstep preorder
    let mut dec = ArithDecoder::new(seq_payload);
    let mut node_seqs: Vec<Vec<u8>> = vec![vec![0_u8; l]; n_nodes];
    let mut gap_mask: Vec<Vec<bool>> = vec![vec![false; l]; n_nodes];
    let mut n_mask: Vec<Vec<bool>> = vec![vec![false; l]; n_nodes];
    let gap_sym_opt = alphabet.iter().position(|&x| x == b'-');
    let n_sym_opt = alphabet.iter().position(|&x| x == b'N');
    let is_canon = |b: u8| matches!(b, b'A' | b'C' | b'G' | b'T' | b'U' | b'a' | b'c' | b'g' | b't' | b'u');
    let mut base_root_row = vec![1_u32; k_alpha];
    for (i, &b) in alphabet.iter().enumerate() {
        if is_canon(b) {
            base_root_row[i] = 32;
        }
    }
    if let Some(g_idx) = gap_sym_opt {
        base_root_row[g_idx] = 0;
    }
    if let Some(n_idx) = n_sym_opt {
        base_root_row[n_idx] = 0;
    }
    let mut root_ctx = vec![base_root_row; 16];
    let mut r_prev1 = 0_usize;
    let mut r_prev2 = 0_usize;
    for s in 0..l {
        let ctx = ((r_prev2 & 3) << 2) | (r_prev1 & 3);
        let sym = dec.decode_categorical(&root_ctx[ctx]);
        node_seqs[0][s] = alphabet[sym];
        root_ctx[ctx][sym] += 16;
        r_prev2 = r_prev1;
        r_prev1 = sym;
    }

    let mut gap_start_cnt = [2_u64, (l as u64) * 2];
    let mut gap_cont_cnt = [20_u64, 4_u64];
    if gap_sym_opt.is_some() {
        let has_root_gap = dec.decode_bit(4096) != 0;
        if has_root_gap {
            let mut in_run = false;
            for s in 0..l {
                let dg = if !in_run {
                    let p1 = ((65536 * gap_start_cnt[0]) / (gap_start_cnt[0] + gap_start_cnt[1]))
                        .clamp(1, 65535) as u32;
                    let flag = dec.decode_bit(p1) != 0;
                    if flag {
                        gap_start_cnt[0] += 1;
                        in_run = true;
                    } else {
                        gap_start_cnt[1] += 1;
                    }
                    flag
                } else {
                    let p1 = ((65536 * gap_cont_cnt[0]) / (gap_cont_cnt[0] + gap_cont_cnt[1]))
                        .clamp(1, 65535) as u32;
                    let flag = dec.decode_bit(p1) != 0;
                    if flag {
                        gap_cont_cnt[0] += 1;
                    } else {
                        gap_cont_cnt[1] += 1;
                        in_run = false;
                    }
                    flag
                };
                gap_mask[0][s] = dg;
            }
        }
    }

    let mut n_start_cnt = [2_u64, (l as u64) * 2];
    let mut n_cont_cnt = [20_u64, 4_u64];
    if n_sym_opt.is_some() {
        let has_root_n = dec.decode_bit(4096) != 0;
        if has_root_n {
            let mut in_run = false;
            for s in 0..l {
                let dn = if !in_run {
                    let p1 = ((65536 * n_start_cnt[0]) / (n_start_cnt[0] + n_start_cnt[1]))
                        .clamp(1, 65535) as u32;
                    let flag = dec.decode_bit(p1) != 0;
                    if flag {
                        n_start_cnt[0] += 1;
                        in_run = true;
                    } else {
                        n_start_cnt[1] += 1;
                    }
                    flag
                } else {
                    let p1 = ((65536 * n_cont_cnt[0]) / (n_cont_cnt[0] + n_cont_cnt[1]))
                        .clamp(1, 65535) as u32;
                    let flag = dec.decode_bit(p1) != 0;
                    if flag {
                        n_cont_cnt[0] += 1;
                    } else {
                        n_cont_cnt[1] += 1;
                        in_run = false;
                    }
                    flag
                };
                n_mask[0][s] = dn;
            }
        }
    }

    let mut mut_hist = vec![0_u32; l];
    let mut codon_counts = [10_u32; 3];
    let mut trans_counts = vec![vec![1_u32; k_alpha]; k_alpha];
    for i in 0..k_alpha {
        for j in 0..k_alpha {
            if is_canon(alphabet[j]) {
                trans_counts[i][j] = 16;
            }
        }
        trans_counts[i][i] = 0;
        if let Some(g_idx) = gap_sym_opt {
            trans_counts[i][g_idx] = 0;
        }
        if let Some(n_idx) = n_sym_opt {
            trans_counts[i][n_idx] = 0;
        }
    }
    for &(b1, b2) in &[
        (b'A', b'G'),
        (b'G', b'A'),
        (b'C', b'T'),
        (b'T', b'C'),
        (b'a', b'g'),
        (b'g', b'a'),
        (b'c', b't'),
        (b't', b'c'),
    ] {
        if let (Some(i1), Some(i2)) = (
            alphabet.iter().position(|&x| x == b1),
            alphabet.iter().position(|&x| x == b2),
        ) {
            trans_counts[i1][i2] += 32;
        }
    }

    for pidx in 1..n_nodes {
        let p_i = parent_pidx[pidx];
        let mut pred = node_seqs[p_i].clone();
        if let Some((donor_i, sites)) = shunt_map.get(&pidx) {
            for &s in sites {
                pred[s] = node_seqs[*donor_i][s];
            }
        }
        let mut work_seq = pred.clone();
        let mut work_gap = gap_mask[p_i].clone();
        let mut work_n = n_mask[p_i].clone();

        if gap_sym_opt.is_some() {
            let has_gap_diff = dec.decode_bit(4096) != 0;
            if has_gap_diff {
                let mut in_run = false;
                for s in 0..l {
                    let diff_g = if !in_run {
                        let p1 = ((65536 * gap_start_cnt[0]) / (gap_start_cnt[0] + gap_start_cnt[1]))
                            .clamp(1, 65535) as u32;
                        let flag = dec.decode_bit(p1) != 0;
                        if flag {
                            gap_start_cnt[0] += 1;
                            in_run = true;
                        } else {
                            gap_start_cnt[1] += 1;
                        }
                        flag
                    } else {
                        let p1 = ((65536 * gap_cont_cnt[0]) / (gap_cont_cnt[0] + gap_cont_cnt[1]))
                            .clamp(1, 65535) as u32;
                        let flag = dec.decode_bit(p1) != 0;
                        if flag {
                            gap_cont_cnt[0] += 1;
                        } else {
                            gap_cont_cnt[1] += 1;
                            in_run = false;
                        }
                        flag
                    };
                    if diff_g {
                        work_gap[s] = !work_gap[s];
                    }
                }
            }
        }

        if n_sym_opt.is_some() {
            let has_n_diff = dec.decode_bit(4096) != 0;
            if has_n_diff {
                let mut in_run = false;
                for s in 0..l {
                    let diff_n = if !in_run {
                        let p1 = ((65536 * n_start_cnt[0]) / (n_start_cnt[0] + n_start_cnt[1]))
                            .clamp(1, 65535) as u32;
                        let flag = dec.decode_bit(p1) != 0;
                        if flag {
                            n_start_cnt[0] += 1;
                            in_run = true;
                        } else {
                            n_start_cnt[1] += 1;
                        }
                        flag
                    } else {
                        let p1 = ((65536 * n_cont_cnt[0]) / (n_cont_cnt[0] + n_cont_cnt[1]))
                            .clamp(1, 65535) as u32;
                        let flag = dec.decode_bit(p1) != 0;
                        if flag {
                            n_cont_cnt[0] += 1;
                        } else {
                            n_cont_cnt[1] += 1;
                            in_run = false;
                        }
                        flag
                    };
                    if diff_n {
                        work_n[s] = !work_n[s];
                    }
                }
            }
        }

        let valid_sites: Vec<usize> = (0..l)
            .filter(|&s| !work_gap[s] && !work_n[s])
            .collect();
        let m_u = dec.decode_int_elias(24000);
        if m_u > 0 {
            let c_sum = (codon_counts[0] + codon_counts[1] + codon_counts[2]) as f64;
            let c_mult = [
                3.0 * (codon_counts[0] as f64) / c_sum,
                3.0 * (codon_counts[1] as f64) / c_sum,
                3.0 * (codon_counts[2] as f64) / c_sum,
            ];
            let weights: Vec<f64> = valid_sites
                .iter()
                .map(|&s| (0.20 + 2.8 * (mut_hist[s] as f64)) * c_mult[s % 3])
                .collect();
            let mut w_rem: f64 = weights.iter().sum();
            let mut m_rem = m_u;
            let mut n_rem = valid_sites.len();

            for (idx_s, &s) in valid_sites.iter().enumerate() {
                if m_rem == 0 {
                    break;
                }
                let is_mut = if m_rem == n_rem {
                    true
                } else {
                    let prob = ((m_rem as f64) * weights[idx_s] / w_rem.max(1e-9)).clamp(1e-6, 0.999);
                    let p1 = ((65536.0 * prob).round() as u32).clamp(1, 65535);
                    dec.decode_bit(p1) != 0
                };
                w_rem -= weights[idx_s];
                n_rem -= 1;
                if is_mut {
                    m_rem -= 1;
                    let b_from = alpha_idx[pred[s] as usize];
                    let mut w_sub = trans_counts[b_from].clone();
                    w_sub[b_from] = 0;
                    if let Some(g_idx) = gap_sym_opt {
                        w_sub[g_idx] = 0;
                    }
                    if let Some(n_idx) = n_sym_opt {
                        w_sub[n_idx] = 0;
                    }
                    let b_to = dec.decode_categorical(&w_sub);
                    work_seq[s] = alphabet[b_to];
                    trans_counts[b_from][b_to] += 3;
                    mut_hist[s] += 1;
                    codon_counts[s % 3] += 1;
                }
            }
        }
        node_seqs[pidx] = work_seq;
        gap_mask[pidx] = work_gap;
        n_mask[pidx] = work_n;
    }

    let mut restored_seqs: Vec<String> = vec![String::new(); n];
    for pidx in 0..n_nodes {
        if is_leaf_vec[pidx] {
            let tid = leaf_id_vec[pidx];
            if tid < n {
                let mut leaf_bytes = node_seqs[pidx].clone();
                for s in 0..l {
                    if gap_mask[pidx][s] {
                        leaf_bytes[s] = b'-';
                    } else if n_mask[pidx][s] {
                        leaf_bytes[s] = b'N';
                    }
                }
                restored_seqs[tid] = String::from_utf8(leaf_bytes)
                    .map_err(|e| format!("UTF-8 error in restored sequence {tid}: {e}"))?;
            }
        }
    }

    Ok((names, restored_seqs, newick))
}

// ---------------------------------------------------------------------------
// Unified Multi-Alphabet Dispatch & WebAssembly Bindings (wasm32-unknown-unknown)
// ---------------------------------------------------------------------------

fn looks_like_csv_or_tsv_matrix(text: &str) -> bool {
    let Some(first) = text
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty() && (!l.starts_with('#') || l.eq_ignore_ascii_case("#nexus")))
    else {
        return false;
    };
    if first.starts_with('>')
        || first.starts_with(';')
        || first.starts_with('(')
        || first.eq_ignore_ascii_case("#nexus")
    {
        return false;
    }
    // Exclude PHYLIP header "N L"
    let mut sp = first.split_whitespace();
    if let (Some(a), Some(b), None) = (sp.next(), sp.next(), sp.next()) {
        if a.chars().all(|c| c.is_ascii_digit()) && b.chars().all(|c| c.is_ascii_digit()) {
            return false;
        }
    }
    first.contains(',') || first.contains('\t')
}

fn detect_sequence_alphabet_kind(seqs: &[String]) -> &'static str {
    let mut digits_01 = 0usize;
    let mut digits_2_9 = 0usize;
    let mut letters_non_nx = 0usize;
    for s in seqs.iter().take(128) {
        for &b in s.as_bytes().iter().take(4096) {
            let up = b.to_ascii_uppercase();
            if matches!(up, b'-' | b'?' | b'.' | b'*') {
                continue;
            }
            if matches!(up, b'0' | b'1') {
                digits_01 += 1;
            } else if up.is_ascii_digit() {
                digits_2_9 += 1;
            } else if up.is_ascii_alphabetic() && up != b'N' && up != b'X' {
                letters_non_nx += 1;
            }
        }
    }
    if digits_01 > 0 && digits_2_9 == 0 && letters_non_nx == 0 {
        "binary"
    } else if (digits_01 + digits_2_9) > 0 && letters_non_nx == 0 {
        "ordered"
    } else if protein::is_protein_alignment(seqs) {
        "protein"
    } else {
        "nucleotide"
    }
}

fn discrete_matrix_to_symbol_seqs(cm: &discrete::DiscreteCharacterMatrix) -> Vec<String> {
    let mut seqs = Vec::with_capacity(cm.n_taxa);
    for i in 0..cm.n_taxa {
        let off = i * cm.n_sites;
        let mut row = Vec::with_capacity(cm.n_sites);
        for j in 0..cm.n_sites {
            let st = cm.mat[off + j];
            let ch = if st < 0 {
                b'?'
            } else if st < 10 {
                b'0' + (st as u8)
            } else if st < 32 {
                b'A' + ((st - 10) as u8)
            } else {
                b'V'
            };
            row.push(ch);
        }
        seqs.push(String::from_utf8(row).unwrap_or_default());
    }
    seqs
}

/// Induce an MDL phylogenetic graph across any supported data modality (`auto`, `nucleotide`,
/// `protein`, `translate`, `binary`, `discrete`, `ordered`) from FASTA, PHYLIP, NEXUS, CSV, or TSV
/// text and return a JSON-serialized `BushwhackerResult` enriched with alphabet metadata.
pub fn infer_any_alignment_json(
    input_text: &str,
    data_mode: &str,
    coding_aware: bool,
    detect_shunts: bool,
    min_shunt_sites: usize,
) -> Result<String, String> {
    let mode_norm = data_mode.trim().to_ascii_lowercase();
    let engine = BushwhackerEngine::new(coding_aware, detect_shunts, min_shunt_sites.max(1));

    let (res, alphabet_mode, k_states, feature_names): (
        BushwhackerResult,
        &'static str,
        usize,
        Vec<String>,
    ) = match mode_norm.as_str() {
        "binary" => {
            let cm = discrete::parse_discrete_str(input_text)?;
            let r = discrete::infer_discrete_mdl(
                &engine,
                &cm,
                discrete::DiscreteMode::Unordered,
                None,
            )?;
            (r, "binary", cm.k_states.max(2), cm.feature_names)
        }
        "discrete" | "unordered" => {
            let cm = discrete::parse_discrete_str(input_text)?;
            let r = discrete::infer_discrete_mdl(
                &engine,
                &cm,
                discrete::DiscreteMode::Unordered,
                None,
            )?;
            (r, "discrete", cm.k_states, cm.feature_names)
        }
        "ordered" | "cna" => {
            let cm = discrete::parse_discrete_str(input_text)?;
            let r = discrete::infer_discrete_mdl(
                &engine,
                &cm,
                discrete::DiscreteMode::Ordered,
                None,
            )?;
            (r, "ordered", cm.k_states, cm.feature_names)
        }
        "protein" | "aa" => {
            let (names, seqs) = parse_alignment_str(input_text)?;
            let r = protein::infer_protein_mdl(&engine, &names, &seqs, None)?;
            (r, "protein", 20, Vec::new())
        }
        "translate" | "codon_translate" => {
            let (names, seqs) = parse_alignment_str(input_text)?;
            let r = if protein::is_protein_alignment(&seqs) {
                protein::infer_protein_mdl(&engine, &names, &seqs, None)?
            } else {
                let aa_seqs = protein::translate_coding_alignment(&seqs)?;
                protein::infer_protein_mdl(&engine, &names, &aa_seqs, None)?
            };
            (r, "translate", 20, Vec::new())
        }
        "nucleotide" | "dna" | "rna" => {
            let (names, seqs) = parse_alignment_str(input_text)?;
            let r = engine.infer(&names, &seqs)?;
            (r, "nucleotide", 4, Vec::new())
        }
        _ => {
            // Auto-detect CSV/TSV character matrix vs FASTA/PHYLIP/NEXUS alignment
            if looks_like_csv_or_tsv_matrix(input_text) {
                let cm = discrete::parse_discrete_str(input_text)?;
                if cm.k_states <= 2 {
                    let r = discrete::infer_discrete_mdl(
                        &engine,
                        &cm,
                        discrete::DiscreteMode::Unordered,
                        None,
                    )?;
                    (r, "binary", 2, cm.feature_names)
                } else {
                    let r = discrete::infer_discrete_mdl(
                        &engine,
                        &cm,
                        discrete::DiscreteMode::Ordered,
                        None,
                    )?;
                    (r, "ordered", cm.k_states, cm.feature_names)
                }
            } else {
                let (names, seqs) = parse_alignment_str(input_text)?;
                match detect_sequence_alphabet_kind(&seqs) {
                    "binary" => {
                        let cm = discrete::encode_discrete_sequences(&names, &seqs)?;
                        let r = discrete::infer_discrete_mdl(
                            &engine,
                            &cm,
                            discrete::DiscreteMode::Unordered,
                            None,
                        )?;
                        (r, "binary", 2, cm.feature_names)
                    }
                    "ordered" => {
                        let cm = discrete::encode_discrete_sequences(&names, &seqs)?;
                        let r = discrete::infer_discrete_mdl(
                            &engine,
                            &cm,
                            discrete::DiscreteMode::Ordered,
                            None,
                        )?;
                        (r, "ordered", cm.k_states, cm.feature_names)
                    }
                    "protein" => {
                        let r = protein::infer_protein_mdl(&engine, &names, &seqs, None)?;
                        (r, "protein", 20, Vec::new())
                    }
                    _ => {
                        let r = engine.infer(&names, &seqs)?;
                        (r, "nucleotide", 4, Vec::new())
                    }
                }
            }
        }
    };

    let mut val = serde_json::to_value(&res).map_err(|e| e.to_string())?;
    if let Some(obj) = val.as_object_mut() {
        obj.insert(
            "alphabet_mode".to_string(),
            serde_json::Value::String(alphabet_mode.to_string()),
        );
        obj.insert(
            "k_states".to_string(),
            serde_json::Value::Number(serde_json::Number::from(k_states)),
        );
        if !feature_names.is_empty() {
            obj.insert(
                "feature_names".to_string(),
                serde_json::to_value(&feature_names).unwrap_or(serde_json::Value::Null),
            );
        }
    }
    serde_json::to_string(&val).map_err(|e| e.to_string())
}

/// Compress any supported alignment or discrete/binary character matrix (`input_text`) and its
/// JSON-serialized `BushwhackerResult` into a bit-for-bit lossless `.bwz` binary archive.
pub fn compress_any_alignment_to_bwz(
    input_text: &str,
    result_json: &str,
) -> Result<Vec<u8>, String> {
    let res: BushwhackerResult = serde_json::from_str(result_json)
        .map_err(|e| format!("Invalid BushwhackerResult JSON: {e}"))?;
    let (names, seqs) = if looks_like_csv_or_tsv_matrix(input_text) {
        let cm = discrete::parse_discrete_str(input_text)?;
        let sqs = discrete_matrix_to_symbol_seqs(&cm);
        (cm.names, sqs)
    } else {
        match parse_alignment_str(input_text) {
            Ok(pair) => pair,
            Err(_) => {
                let cm = discrete::parse_discrete_str(input_text)?;
                let sqs = discrete_matrix_to_symbol_seqs(&cm);
                (cm.names, sqs)
            }
        }
    };
    let (bytes, _stats) = compress_alignment_to_bwz(&names, &seqs, &res)?;
    Ok(bytes)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct WasmBushwhacker;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl WasmBushwhacker {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self
    }

    /// Induce an MDL phylogenetic graph with explicit or auto-detected data modality
    /// (`auto`, `nucleotide`, `protein`, `translate`, `binary`, `discrete`, `ordered`).
    pub fn infer_alignment(
        &self,
        input_text: &str,
        data_mode: &str,
        coding_aware: bool,
        detect_shunts: bool,
        min_shunt_sites: usize,
    ) -> Result<String, JsValue> {
        infer_any_alignment_json(
            input_text,
            data_mode,
            coding_aware,
            detect_shunts,
            min_shunt_sites,
        )
        .map_err(|e| JsValue::from_str(&e))
    }

    /// Backwards-compatible entrypoint that auto-detects Nucleotide, Protein, Binary (`0/1`),
    /// or Discrete/Ordered CSV/TSV matrices.
    pub fn infer_fasta(
        &self,
        fasta_text: &str,
        coding_aware: bool,
        detect_shunts: bool,
        min_shunt_sites: usize,
    ) -> Result<String, JsValue> {
        infer_any_alignment_json(
            fasta_text,
            "auto",
            coding_aware,
            detect_shunts,
            min_shunt_sites.max(2),
        )
        .map_err(|e| JsValue::from_str(&e))
    }

    /// Compress an alignment or character matrix (`fasta_text`) and its JSON-serialized
    /// `BushwhackerResult` into a self-contained, bit-for-bit lossless `.bwz` binary archive.
    pub fn compress_bwz(
        &self,
        fasta_text: &str,
        result_json: &str,
    ) -> Result<Vec<u8>, JsValue> {
        compress_any_alignment_to_bwz(fasta_text, result_json).map_err(|e| JsValue::from_str(&e))
    }

    /// Decompress a `.bwz` binary archive back into a JSON object `{ fasta, newick, n_taxa, n_sites }`.
    pub fn decompress_bwz(&self, bwz_bytes: &[u8]) -> Result<String, JsValue> {
        let (names, seqs, newick) =
            decompress_alignment_from_bwz(bwz_bytes).map_err(|e| JsValue::from_str(&e))?;
        let mut fasta = String::new();
        for (name, seq) in names.iter().zip(seqs.iter()) {
            fasta.push('>');
            fasta.push_str(name);
            fasta.push('\n');
            fasta.push_str(seq);
            fasta.push('\n');
        }
        let n_taxa = names.len();
        let n_sites = seqs.first().map(|s| s.len()).unwrap_or(0);
        let payload = serde_json::json!({
            "fasta": fasta,
            "newick": newick,
            "n_taxa": n_taxa,
            "n_sites": n_sites,
        });
        serde_json::to_string(&payload).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_star_radiation_contracts_to_zero_splits() {
        // 8 taxa radiating from an ancestral AAA... sequence with independent terminal mutations
        let l = 300;
        let names: Vec<String> = (0..8).map(|i| format!("T{i}")).collect();
        let mut seqs = Vec::new();
        for i in 0..8 {
            let mut s = vec![b'A'; l];
            for k in 0..5 {
                s[i * 10 + k] = b'G';
            }
            seqs.push(String::from_utf8(s).unwrap());
        }
        let engine = BushwhackerEngine::default();
        let res = engine.infer(&names, &seqs).unwrap();
        assert_eq!(res.n_internal_branches, 0);
        assert!(res.polytomy_nodes >= 1);
    }

    #[test]
    fn test_clean_clade_resolves_with_high_support() {
        let l = 300;
        let names: Vec<String> = vec!["A1", "A2", "B1", "B2", "B3"]
            .into_iter()
            .map(String::from)
            .collect();
        let mut s_a1 = vec![b'A'; l];
        let mut s_a2 = vec![b'A'; l];
        let mut s_b1 = vec![b'A'; l];
        let mut s_b2 = vec![b'A'; l];
        let mut s_b3 = vec![b'A'; l];
        // 5 clean synapomorphies uniting (A1, A2) vs (B1, B2, B3)
        for pos in [11, 41, 89, 149, 221] {
            s_a1[pos] = b'G';
            s_a2[pos] = b'G';
        }
        // Terminal mutations
        s_a1[1] = b'C';
        s_a2[2] = b'T';
        s_b1[3] = b'C';
        s_b2[4] = b'T';
        s_b3[5] = b'G';
        let seqs = vec![
            String::from_utf8(s_a1).unwrap(),
            String::from_utf8(s_a2).unwrap(),
            String::from_utf8(s_b1).unwrap(),
            String::from_utf8(s_b2).unwrap(),
            String::from_utf8(s_b3).unwrap(),
        ];
        let engine = BushwhackerEngine::default();
        let res = engine.infer(&names, &seqs).unwrap();
        assert_eq!(res.n_internal_branches, 1);
        assert!(res.backbone_tree_nats < res.star_tree_nats);
    }

    #[test]
    fn test_phylip_and_nexus_parsers_match_fasta() {
        let fasta = "\
>Taxon_A1
ACGTACGTACGTACGTACGT
>Taxon_A2
ACGTACGTACGTACGTACGA
>Taxon_B1
TGCATGCATGCATGCATGCA
>Taxon_B2
TGCATGCATGCATGCATGCT
";
        // 1. Relaxed Sequential PHYLIP
        let phylip_relaxed = "\
4 20
Taxon_A1   ACGTACGTACGTACGTACGT
Taxon_A2   ACGTACGTACGTACGTACGA
Taxon_B1   TGCATGCATGCATGCATGCA
Taxon_B2   TGCATGCATGCATGCATGCT
";
        // 2. Strict 10-char Sequential PHYLIP (no space after 10-char name on row 1, plus '.' matchchar)
        let phylip_strict = "\
 4 20
Taxon_A1  ACGTACGTACGTACGTACGT
Taxon_A2  ...................A
Taxon_B1  TGCATGCATGCATGCATGCA
Taxon_B2  TGCATGCATGCATGCATGCT
";
        // 3. Interleaved PHYLIP (2 blocks of 10 bp)
        let phylip_interleaved = "\
4 20
Taxon_A1   ACGTACGTAC
Taxon_A2   ACGTACGTAC
Taxon_B1   TGCATGCATG
Taxon_B2   TGCATGCATG

           GTACGTACGT
           GTACGTACGA
           CATGCATGCA
           CATGCATGCT
";
        // 4. NEXUS Interleaved with comments, quoted names, and '.' matchchar
        let nexus_interleaved = "\
#NEXUS
[ multi-line comment [with nested brackets] ]
BEGIN DATA;
  DIMENSIONS NTAX=4 NCHAR=20;
  FORMAT DATATYPE=DNA INTERLEAVE=YES GAP=- MISSING=? MATCHCHAR=.;
  MATRIX
    'Taxon A1'  ACGTACGTAC [block 1]
    'Taxon A2'  ..........
    Taxon_B1    TGCATGCATG
    Taxon_B2    TGCATGCATG

    'Taxon A1'  GTACGTACGT [block 2]
    'Taxon A2'  .........A
    Taxon_B1    CATGCATGCA
    Taxon_B2    CATGCATGCT
  ;
END;
";

        let (nf, sf, fmt_f) = parse_alignment_str_with_format(fasta).unwrap();
        let (np1, sp1, fmt_p1) = parse_alignment_str_with_format(phylip_relaxed).unwrap();
        let (np2, sp2, fmt_p2) = parse_alignment_str_with_format(phylip_strict).unwrap();
        let (np3, sp3, fmt_p3) = parse_alignment_str_with_format(phylip_interleaved).unwrap();
        let (nn, sn, fmt_n) = parse_alignment_str_with_format(nexus_interleaved).unwrap();

        assert_eq!(fmt_f, "FASTA");
        assert_eq!(fmt_p1, "PHYLIP");
        assert_eq!(fmt_p2, "PHYLIP");
        assert_eq!(fmt_p3, "PHYLIP");
        assert_eq!(fmt_n, "NEXUS");

        assert_eq!(nf, np1);
        assert_eq!(nf, np2);
        assert_eq!(nf, np3);
        assert_eq!(nf, nn);

        assert_eq!(sf, sp1);
        assert_eq!(sf, sp2);
        assert_eq!(sf, sp3);
        assert_eq!(sf, sn);
    }

    #[test]
    fn test_bwz_lossless_roundtrip() {
        let fasta = "\
>Taxon_A1
ACGTACGT---TACGTACGTACGTACGT
>Taxon_A2
ACGTACGT---TACGTACGAACGTACGT
>Taxon_B1
TGCATGCATGCTGCATGCATGCATGCAT
>Taxon_B2
TGCATGCATGCTGCATGCTTGCATGCAT
>Taxon_C1
TGCATGCANNNTGCATGCTTGCATGCAT
";
        let (names, seqs) = parse_fasta_str(fasta).unwrap();
        let engine = BushwhackerEngine::default();
        let res = engine.infer(&names, &seqs).unwrap();
        let (bwz_bytes, stats) = compress_alignment_to_bwz(&names, &seqs, &res).unwrap();
        assert!(stats.compressed_seq_bytes > 0);
        let (dec_names, dec_seqs, dec_nwk) = decompress_alignment_from_bwz(&bwz_bytes).unwrap();
        assert_eq!(names, dec_names);
        assert_eq!(seqs, dec_seqs);
        assert_eq!(res.newick, dec_nwk);
    }

    #[test]
    fn test_protein_auto_detection_and_bwz_roundtrip() {
        let aa_fasta = "\
>Bat_1
MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHG
>Bat_2
MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHG
>Dolphin_1
MVLSPAEKTNVKAAWGKVGGHAGEYGAEALERMFLGFPTTKTYFPHFDLSHGSAQVKAHG
>Dolphin_2
MVLSPAEKTNVKAAWGKVGGHAGEYGAEALERMFLGFPTTKTYFPHFDLSHGSAQVKAHG
>Outgroup
MVLSAADKNNVKGIFTKIAGHAEEYGAETLERMFTTYPPTKTYFPHFDLSHGSAQIKAHG
";
        let (names, seqs) = parse_fasta_str(aa_fasta).unwrap();
        assert!(protein::is_protein_alignment(&seqs));
        let engine = BushwhackerEngine::default();
        let res = engine.infer(&names, &seqs).unwrap();
        assert!(res.n_internal_branches >= 1);
        let (bwz_bytes, _) = compress_alignment_to_bwz(&names, &seqs, &res).unwrap();
        let (dec_names, dec_seqs, _) = decompress_alignment_from_bwz(&bwz_bytes).unwrap();
        assert_eq!(names, dec_names);
        assert_eq!(seqs, dec_seqs);
    }

    #[test]
    fn test_multithreaded_bit_exact_determinism() {
        let n_taxa = 560usize;
        let l_codons = 60usize;
        let codons = [b"ATG", b"GCT", b"TTC", b"CAG", b"GAC", b"AAG", b"GTG", b"TGG"];
        let mut names = Vec::with_capacity(n_taxa);
        let mut seqs = Vec::with_capacity(n_taxa);
        for i in 0..n_taxa {
            names.push(format!("Taxon_{:04}", i));
            let clade = i / 35;
            let subclade = (i % 35) / 7;
            let mut s = String::with_capacity(l_codons * 3);
            for c in 0..l_codons {
                let mut idx = (c * 3 + (clade & 3)) & 7;
                if c >= 10 && c < 20 && subclade >= 2 {
                    idx = (idx + subclade) & 7;
                }
                if c >= 45 && c < 50 && (clade == 2 || clade == 11) {
                    idx = 7;
                }
                if c == (i % l_codons) {
                    idx = (idx + 1) & 7;
                }
                s.push_str(std::str::from_utf8(codons[idx]).unwrap());
            }
            seqs.push(s);
        }

        let mut nt_refs: Vec<BushwhackerResult> = Vec::new();
        let mut aa_refs: Vec<BushwhackerResult> = Vec::new();
        for &nth in &[1usize, 2, 4, 8, 16, 64] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(nth)
                .build()
                .unwrap();
            let engine_nt = BushwhackerEngine::default();
            nt_refs.push(pool.install(|| engine_nt.infer(&names, &seqs).unwrap()));
            let mut engine_aa = BushwhackerEngine::default();
            engine_aa.translate_coding = true;
            aa_refs.push(pool.install(|| engine_aa.infer(&names, &seqs).unwrap()));
        }
        for i in 1..nt_refs.len() {
            assert_eq!(nt_refs[0].newick, nt_refs[i].newick);
            assert_eq!(nt_refs[0].binary_newick, nt_refs[i].binary_newick);
            assert_eq!(
                nt_refs[0].backbone_tree_nats.to_bits(),
                nt_refs[i].backbone_tree_nats.to_bits()
            );
            assert_eq!(
                nt_refs[0].full_graph_nats.to_bits(),
                nt_refs[i].full_graph_nats.to_bits()
            );
            assert_eq!(nt_refs[0].shunts.len(), nt_refs[i].shunts.len());

            assert_eq!(aa_refs[0].newick, aa_refs[i].newick);
            assert_eq!(aa_refs[0].binary_newick, aa_refs[i].binary_newick);
            assert_eq!(
                aa_refs[0].backbone_tree_nats.to_bits(),
                aa_refs[i].backbone_tree_nats.to_bits()
            );
            assert_eq!(
                aa_refs[0].full_graph_nats.to_bits(),
                aa_refs[i].full_graph_nats.to_bits()
            );
            assert_eq!(aa_refs[0].shunts.len(), aa_refs[i].shunts.len());
        }
    }

    #[test]
    fn test_universal_int_kraft_exact() {
        let ln2 = 2.0_f64.ln();
        let partial: f64 = (0..200_000)
            .map(|m| (-universal_int_nats(m as f64)).exp())
            .sum();
        let tail = 0.5 * ln2 / 200_001.0_f64.ln();
        assert!(
            ((partial + tail) - 1.0).abs() < 1e-11,
            "Kraft sum must equal 1.0 identically, got {}",
            partial + tail
        );
    }

    #[test]
    fn test_cat8_exact_unit_mean() {
        let sum_r: f64 = nt_gtr::NT_CAT_RATES.iter().map(|&r| r as f64).sum();
        assert!(((sum_r / (nt_gtr::NT_N_CATS as f64)) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_protein_lg08_kraft_normalization() {
        let costs = protein::estimate_aa_subst_costs(&[], 0, 0);
        for a in 0..20 {
            assert_eq!(costs[a][a], 0.0);
            let kraft_sum: f64 = (0..20)
                .filter(|&b| b != a)
                .map(|b| (-costs[a][b]).exp())
                .sum();
            assert!(
                (kraft_sum - 1.0).abs() < 1e-12,
                "Kraft sum for AA {} = {}",
                a,
                kraft_sum
            );
        }
        // Alphabetical order: A=0,C=1,D=2,E=3,F=4,G=5,H=6,I=7,K=8,L=9,M=10,N=11,P=12,Q=13,R=14,S=15,T=16,V=17,W=18,Y=19
        assert!(costs[7][17] < costs[7][2]); // I->V cheaper than I->D
        assert!(costs[8][14] < costs[8][18]); // K->R cheaper than K->W
        assert!(costs[4][19] < costs[4][12]); // F->Y cheaper than F->P
    }

    #[test]
    fn test_protein_star_radiation_contracts_to_zero_splits() {
        let base_motif = b"MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLS"; // 50 AA
        let l = 1000usize;
        let aa_alphabet = b"ACDEFGHIKLMNPQRSTVWY";
        let costs = protein::estimate_aa_subst_costs(&[], 0, 0);
        let mut names = Vec::new();
        let mut seqs = Vec::new();
        let mut rng_state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut next_f64 = || -> f64 {
            rng_state ^= rng_state >> 12;
            rng_state ^= rng_state << 25;
            rng_state ^= rng_state >> 27;
            let u = rng_state.wrapping_mul(0x2545_F491_4F6C_DD1D);
            ((u >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
        };
        for i in 0..16usize {
            names.push(format!("StarAA_{:02}", i));
            let mut row: Vec<u8> = (0..l).map(|s| base_motif[s % base_motif.len()]).collect();
            for s in 0..l {
                if next_f64() < 0.20 {
                    let a = aa_alphabet.iter().position(|&c| c == row[s]).unwrap();
                    let u = next_f64();
                    let mut cum = 0.0_f64;
                    let mut chosen = a;
                    for b in 0..20 {
                        if b != a {
                            cum += (-costs[a][b]).exp();
                            if u <= cum {
                                chosen = b;
                                break;
                            }
                        }
                    }
                    row[s] = aa_alphabet[chosen];
                }
            }
            seqs.push(String::from_utf8(row).unwrap());
        }
        let mut engine = BushwhackerEngine::default();
        engine.force_protein = true;
        engine.detect_shunts = false;
        let res = engine.infer(&names, &seqs).unwrap();
        assert_eq!(
            res.n_internal_branches, 0,
            "Expected 0 internal branches on LG08 20-state star radiation, got {} (nwk={})",
            res.n_internal_branches,
            res.newick
        );
    }

    #[test]
    fn test_protein_deep_biochemical_clades_resolve() {
        // 8 taxa: Clade A (0..4) vs Clade B (4..8) across 800 AA where diagnostic sites
        // exhibit within-class terminal divergence (I vs V in Clade A; K vs R in Clade B),
        // so unweighted 4-taxon Fitch steps tie (3 vs 3) while LG08 exchangeabilities resolve A|B.
        let l = 800usize;
        let bg = b"MGLSDGEWQLVLNVWGKVEADIPGHGQEVLIRLFKGHPETLEKFDKFKHL";
        let mut names = Vec::new();
        let mut seqs = Vec::new();
        for i in 0..8usize {
            names.push(format!("Clade{}_{}", if i < 4 { "A" } else { "B" }, i % 4));
            let mut row: Vec<u8> = (0..l).map(|s| bg[s % bg.len()]).collect();
            for s in (0..l).step_by(5) {
                row[s] = match (i < 4, i % 2 == 0) {
                    (true, true) => b'I',
                    (true, false) => b'V',
                    (false, true) => b'K',
                    (false, false) => b'R',
                };
            }
            // Add independent leaf-specific noise at non-diagnostic sites
            for s in 0..l {
                if s % 5 != 0 && ((s * 13 + i * 37) % 11 == 0) {
                    let noise_aa = b"ADEFGHLMNPQSTWY"[(s + i * 3) % 15];
                    row[s] = noise_aa;
                }
            }
            seqs.push(String::from_utf8(row).unwrap());
        }
        let mut engine = BushwhackerEngine::default();
        engine.force_protein = true;
        engine.detect_shunts = false;
        let res = engine.infer(&names, &seqs).unwrap();
        assert!(
            res.n_internal_branches >= 1,
            "Expected biochemical split A|B to resolve, got {} internal branches (nwk={})",
            res.n_internal_branches,
            res.newick
        );
    }

    #[test]
    fn test_nt_gtr_eigendecomposition_and_reversibility() {
        let fasta = "\
>Sp1
ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT
>Sp2
ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGA
>Sp3
TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA
>Sp4
TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCT
";
        let (_, seqs) = parse_fasta_str(fasta).unwrap();
        let (mat, _, _, _, l) = encode_nucleotide_alignment(&seqs);
        let gtr = nt_gtr::estimate_empirical_gtr_eigensystem(&mat, seqs.len(), l);
        let pi_sum: f64 = gtr.pi.iter().sum();
        assert!((pi_sum - 1.0).abs() < 1e-12);
        assert_eq!(gtr.evals[0], 0.0);
        for &ev in &gtr.evals[1..] {
            assert!(ev < -1e-4, "Non-stationary GTR eigenvalue must be negative, got {ev}");
        }
        let p0 = gtr.trans_mat4(0.0);
        let pt = gtr.trans_mat4(0.35);
        for a in 0..4 {
            let mut row_s = 0.0_f64;
            for b in 0..4 {
                let target0 = if a == b { 1.0 } else { 0.0 };
                assert!((p0[a][b] as f64 - target0).abs() < 1e-5);
                row_s += pt[a][b] as f64;
                // Detailed balance: pi[a] * P(t)[a][b] == pi[b] * P(t)[b][a]
                let lhs = gtr.pi[a] * (pt[a][b] as f64);
                let rhs = gtr.pi[b] * (pt[b][a] as f64);
                assert!(
                    (lhs - rhs).abs() < 1e-5,
                    "Detailed balance violated for ({a},{b}): {lhs} vs {rhs}"
                );
            }
            assert!((row_s - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_unified_multi_alphabet_dispatch_and_bwz_roundtrip() {
        // 1. Protein auto-detection & lossless .bwz roundtrip (including Asparagine 'N' and gaps '-')
        let prot_fasta = "\
>TaxA1
MGLSDGEWQLVLNVWGKVEADIPGHGQEVLIRLFKGHPETLEKFDKFKHLKSEDEMKASE
>TaxA2
MGLSDGEWQLVLNVWGKVEADIPGHGQEVLIRLFKGHPETLEKFDKFKHLKSEDEMKASD
>TaxB1
MGLSDGEWQLVLNVWGKVEADIPGHGQEVLIRLFKGHPEVLEKFDKFKHLKSEDEMKA--
>TaxB2
MGLSDGEWQLVLNVWGKVEADIPGHGQEVLIRLFKGHPEVLEKFDKFKHLKSEDEMKANN
";
        let prot_json = infer_any_alignment_json(prot_fasta, "auto", true, true, 2).unwrap();
        let prot_val: serde_json::Value = serde_json::from_str(&prot_json).unwrap();
        assert_eq!(prot_val["alphabet_mode"], "protein");
        assert_eq!(prot_val["k_states"], 20);
        let bwz_prot = compress_any_alignment_to_bwz(prot_fasta, &prot_json).unwrap();
        let (dec_names, dec_seqs, _) = decompress_alignment_from_bwz(&bwz_prot).unwrap();
        let (orig_names, orig_seqs) = parse_alignment_str(prot_fasta).unwrap();
        assert_eq!(dec_names, orig_names);
        assert_eq!(dec_seqs, orig_seqs);

        // 2. Codon translation ("translate") mode
        let cds_fasta = "\
>SpA1
ATGGTGCTGTCTGATGGGGAGTGGCAGCTGGTCTTGAATGTCTGGGGGAAGGTGGAGGCC
>SpA2
ATGGTGCTGTCTGATGGGGAGTGGCAGCTGGTCTTGAATGTCTGGGGGAAGGTGGAGGCT
>SpB1
ATGGTGCTGTCTGATGGGGAGTGGCAGCTGATCTTGAATGTCTGGGGGAAGATAGAGGCC
>SpB2
ATGGTGCTGTCTGATGGGGAGTGGCAGCTGATCTTGAATGTCTGGGGGAAGATAGAGGCA
";
        let tr_json = infer_any_alignment_json(cds_fasta, "translate", true, false, 2).unwrap();
        let tr_val: serde_json::Value = serde_json::from_str(&tr_json).unwrap();
        assert_eq!(tr_val["alphabet_mode"], "translate");
        assert_eq!(tr_val["n_sites"], 20);
        let bwz_tr = compress_any_alignment_to_bwz(cds_fasta, &tr_json).unwrap();
        let (_, dec_cds, _) = decompress_alignment_from_bwz(&bwz_tr).unwrap();
        let (_, orig_cds) = parse_alignment_str(cds_fasta).unwrap();
        assert_eq!(dec_cds, orig_cds);

        // 3. Binary 0/1 FASTA & CSV auto-detection + lossless .bwz roundtrip
        let bin_csv = "\
clone,DNMT3A,NPM1,FLT3,NRAS,TET2,SRSF2,IDH2,WT1
WT_Root,0,0,0,0,0,0,0,0
Clone_A1,1,1,1,0,0,0,0,0
Clone_A2,1,1,1,0,0,0,0,1
Clone_B1,0,0,0,1,1,1,0,0
Clone_B2,0,0,0,1,1,1,1,?
";
        let bin_json = infer_any_alignment_json(bin_csv, "auto", false, true, 2).unwrap();
        let bin_val: serde_json::Value = serde_json::from_str(&bin_json).unwrap();
        assert_eq!(bin_val["alphabet_mode"], "binary");
        assert_eq!(bin_val["k_states"], 2);
        assert_eq!(bin_val["feature_names"][0], "DNMT3A");
        let bwz_bin = compress_any_alignment_to_bwz(bin_csv, &bin_json).unwrap();
        let (dec_bnames, dec_bseqs, _) = decompress_alignment_from_bwz(&bwz_bin).unwrap();
        assert_eq!(dec_bnames.len(), 5);
        assert_eq!(dec_bseqs[4], "0001111?");
    }

    #[test]
    fn test_sparse_shunt_mask_kraft_bound() {
        let min_sites = 2usize;
        for &m_donor in &[2usize, 5usize, 12usize, 40usize] {
            let mut sparse_sum = 0.0_f64;
            for m in min_sites..=m_donor {
                let mask_cost = 2.0_f64.ln()
                    + universal_int_nats((m - min_sites) as f64)
                    + log_binom(m_donor as f64, m as f64);
                // C(m_donor, m) distinct m-subsets each have code length `mask_cost`
                sparse_sum += (log_binom(m_donor as f64, m as f64) - mask_cost).exp();
            }
            // Must be strictly <= 0.5 so INTERVAL + SPARSE combined strictly satisfy Kraft < 1.0
            assert!(
                sparse_sum < 0.5,
                "SPARSE Kraft sum exceeded 0.5 for M={}: {}",
                m_donor,
                sparse_sum
            );
            let n_intervals = (m_donor * (m_donor + 1)) / 2;
            let interval_sum =
                (n_intervals as f64) * (-2.0_f64.ln() - 2.0 * (m_donor as f64).ln()).exp();
            assert!(
                interval_sum < 0.5,
                "INTERVAL Kraft sum exceeded 0.5 for M={}: {}",
                m_donor,
                interval_sum
            );
            assert!(
                sparse_sum + interval_sum < 1.0,
                "Total mask Kraft sum exceeded 1.0 for M={}: {}",
                m_donor,
                sparse_sum + interval_sum
            );
        }
    }

    #[test]
    fn test_empirical_alpha_octiles_unit_mean() {
        let r1 = gamma_octile_rates(1.0);
        for c in 0..8 {
            assert!(
                ((r1[c] - nt_gtr::NT_CAT_RATES[c]) as f64).abs() < 1e-5,
                "alpha=1 octile {} mismatch: {} vs {}",
                c,
                r1[c],
                nt_gtr::NT_CAT_RATES[c]
            );
        }
        for &alpha in &[0.25, 0.5, 1.0, 2.0, 5.0, 15.0] {
            let rates = gamma_octile_rates(alpha);
            let mean: f64 = rates.iter().map(|&x| x as f64).sum::<f64>() / 8.0;
            assert!(
                (mean - 1.0).abs() < 1e-6,
                "Octile mean mismatch for alpha={}: {}",
                alpha,
                mean
            );
            for c in 0..7 {
                assert!(
                    rates[c] < rates[c + 1],
                    "Octiles not strictly increasing for alpha={}: {:?}",
                    alpha,
                    rates
                );
            }
        }
    }

    #[test]
    fn test_split_scorers_route_through_universal_int_nats() {
        // Verify that both NT (compute_split_mdl_gain_from_child_refs) and Protein
        // (compute_protein_split_mdl_gain_refs) split scorers use exact Kraft-tight
        // universal_int_nats(m) rather than the legacy divergent ln(2)+ln(1+m)+0.5*ln(1+ln(1+m)).
        let l = 32usize;
        let l_eff = 32.0_f64;
        let weights = vec![1.0_f64; l];
        let old_header_2 = 2.0_f64.ln() + 3.0_f64.ln() + 0.5 * (1.0 + 3.0_f64.ln()).ln();
        let new_header_2 = universal_int_nats(2.0);
        assert!((new_header_2 - old_header_2).abs() > 0.1);

        // 1. Nucleotide quartet with 2 clean transitions (A=0 in U, G=2 in V)
        let u_nt = vec![1_u8; l];
        let mut v_nt = vec![1_u8; l];
        for s in 0..2 {
            v_nt[s] = 4; // G
        }
        let load = 0.5_f64;
        let u_children_nt = [(&u_nt[..], load), (&u_nt[..], load)];
        let v_children_nt = [(&v_nt[..], load), (&v_nt[..], load)];
        let engine = BushwhackerEngine::default();
        let (ti_cost, tv_cost) = (1.0_f64, 1.5_f64);
        let (nt_gain, nt_m, _) = engine.compute_split_mdl_gain_from_child_refs(
            &u_children_nt,
            &v_children_nt,
            l,
            l_eff,
            ti_cost,
            tv_cost,
            &weights,
        );
        assert!(
            nt_gain > 0.0 && (nt_m - 2.0).abs() < 1e-12,
            "Unexpected NT split result: gain={}, m={}",
            nt_gain,
            nt_m
        );
        let w_c = ((l_eff - load) / (load + 1.0)).ln();
        // Site s=0 has clean_sites.len()==1 -> delta_steps=1; site s=1 has clean_sites.len()==2 -> delta_steps=2.
        let total_sav = 4.0 * w_c + 3.0 * ti_cost;
        let expected_nt_gain = total_sav
            - (new_header_2 + log_binom(l_eff, 2.0) + 2.0 * ti_cost)
            - split_topology_cost_nats(2, 2);
        assert!(
            (nt_gain - expected_nt_gain).abs() < 1e-10,
            "NT split scorer did not match universal_int_nats(2): {} vs {}",
            nt_gain,
            expected_nt_gain
        );

        // 2. Protein quartet with 2 clean substitutions (Ala=0 in U, Trp=18 in V)
        let u_aa = vec![1_u32; l]; // all Ala (bit 0)
        let mut v_aa = vec![1_u32; l];
        for s in 0..2 {
            v_aa[s] = 1_u32 << 18; // Trp
        }
        let subst_costs = protein::estimate_aa_subst_costs(&[], 0, 0);
        let u_children_aa = [(&u_aa[..], load), (&u_aa[..], load)];
        let v_children_aa = [(&v_aa[..], load), (&v_aa[..], load)];
        let (aa_gain, aa_m, _) = protein::compute_protein_split_mdl_gain_refs(
            &u_children_aa,
            &v_children_aa,
            l,
            l_eff,
            &subst_costs,
            &weights,
        );
        assert!(
            aa_gain > 0.0 && (aa_m - 2.0).abs() < 1e-12,
            "Unexpected AA split result: gain={}, m={}",
            aa_gain,
            aa_m
        );
        // Add a 3rd identical Ala->Trp site: exact marginal header delta must equal
        // -(universal_int_nats(3.0) - universal_int_nats(2.0)).
        v_aa[2] = 1_u32 << 18;
        let (aa_gain_3, aa_m3, _) = protein::compute_protein_split_mdl_gain_refs(
            &[(&u_aa[..], load), (&u_aa[..], load)],
            &[(&v_aa[..], load), (&v_aa[..], load)],
            l,
            l_eff,
            &subst_costs,
            &weights,
        );
        assert!((aa_m3 - 3.0).abs() < 1e-12);
        let total_site_sav_2 = aa_gain + new_header_2 + log_binom(l_eff, 2.0) + split_topology_cost_nats(2, 2);
        let expected_aa_gain_3 = 1.5 * total_site_sav_2
            - universal_int_nats(3.0)
            - log_binom(l_eff, 3.0)
            - split_topology_cost_nats(2, 2);
        assert!(
            (aa_gain_3 - expected_aa_gain_3).abs() < 1e-4,
            "Protein split scorer header mismatch: {} vs {}",
            aa_gain_3,
            expected_aa_gain_3
        );
    }
}

