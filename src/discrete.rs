//! General `K`-State Discrete & Ordered Wagner Character-Matrix MDL Graph Induction (`2 <= K <= 32`).
//!
//! Prices phylogenetic trees, multifurcating clonal polytomies, and dual-mode (`INTERVAL` / `SPARSE`)
//! lateral shunts directly in natural logarithmic information units (nats) over arbitrary non-sequence
//! character matrices `C in {0, ..., K - 1, missing}^{N x M}` with zero DNA or protein substitution priors:
//!   1. `DiscreteMode::Unordered` (`--discrete`): Sankoff-Fitch intersection/union `u32` bitmasks with
//!      Kraft-exact Laplace-smoothed empirical / uniform off-diagonal state surprise `-ln P_hat(b | a)`.
//!   2. `DiscreteMode::Ordered` (`--ordered`): Farris-Wagner `L1` interval-hull dynamic programming
//!      `[l_v, r_v]` across binary and arbitrary-degree polytomy nodes, paired with a Kraft-exact
//!      geometric step-size code `l_step(b | a)` normalized over `b != a` and lossless segmental
//!      run-length breakpoint encoding (`L_RLE`) across collinear chromosomal copy-number bins.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::fs;
use std::path::Path;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use crate::{
    branch_mutation_code_nats, log_binom, parse_alignment_str, split_topology_cost_nats,
    universal_int_nats, BranchRecord, BushwhackerEngine, BushwhackerResult, ShuntEdge,
};

pub const MAX_DISCRETE_STATES: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiscreteMode {
    Unordered,
    Ordered,
}

#[derive(Clone, Debug)]
pub struct DiscreteCharacterMatrix {
    pub names: Vec<String>,
    pub mat: Vec<i8>,
    pub bitmasks: Vec<u32>,
    pub majority_state: Vec<i8>,
    pub n_taxa: usize,
    pub n_sites: usize,
    pub k_states: usize,
    pub feature_names: Vec<String>,
    pub chrom_ids: Vec<u16>,
}

#[derive(Clone, Debug)]
pub struct DiscreteSubstModel {
    pub mode: DiscreteMode,
    pub k: usize,
    pub all_mask: u32,
    pub pi: [f64; MAX_DISCRETE_STATES],
    /// Off-diagonal conditional code length `-ln P(b | a)` for `b != a` (`0.0` on diagonal).
    /// Satisfies exact Kraft equality `sum_{b != a} exp(-cost[a][b]) == 1.0` for every `a in 0..k`.
    pub cost: [[f64; MAX_DISCRETE_STATES]; MAX_DISCRETE_STATES],
    pub step_gamma: f64,
    pub is_collinear_bins: bool,
    pub l_eff_seg: f64,
    pub avg_block_len: f64,
    pub dist_weights: Vec<f64>,
    pub tree_weights: Vec<f64>,
}

#[inline]
pub fn all_states_mask(k: usize) -> u32 {
    if k >= 32 {
        u32::MAX
    } else if k == 0 {
        0
    } else {
        (1_u32 << (k as u32)) - 1
    }
}

#[inline]
pub fn interval_mask(lo: usize, hi: usize) -> u32 {
    let lo_c = lo.min(31);
    let hi_c = hi.min(31).max(lo_c);
    let top = if hi_c == 31 {
        u64::from(u32::MAX) + 1
    } else {
        1_u64 << ((hi_c + 1) as u32)
    };
    let bot = 1_u64 << (lo_c as u32);
    (top - bot) as u32
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

fn parse_chrom_id_from_feature(feat: &str, chrom_dict: &mut HashMap<String, u16>) -> u16 {
    let trimmed = feat.trim().trim_matches('"').trim_matches('\'');
    let sep_pos = trimmed.find('_').or_else(|| trimmed.find(':'));
    let Some(idx) = sep_pos else {
        return 0;
    };
    let prefix = trimmed[..idx].trim();
    if prefix.is_empty() {
        return 0;
    }
    let clean = prefix
        .strip_prefix("chr")
        .or_else(|| prefix.strip_prefix("CHR"))
        .or_else(|| prefix.strip_prefix("Chr"))
        .unwrap_or(prefix);
    let valid_chr = clean.chars().all(|c| c.is_ascii_digit())
        || matches!(clean, "X" | "Y" | "M" | "MT" | "x" | "y");
    if !valid_chr {
        return 0;
    }
    let key = clean.to_ascii_uppercase();
    if let Some(&id) = chrom_dict.get(&key) {
        id
    } else {
        let next_id = (chrom_dict.len() as u16).saturating_add(1);
        chrom_dict.insert(key, next_id);
        next_id
    }
}

#[inline]
fn parse_cell_token(tok: &str) -> Result<Option<i32>, ()> {
    let t = tok.trim().trim_matches('"').trim_matches('\'');
    if t.is_empty()
        || matches!(
            t,
            "?" | "-" | "." | "*" | "NA" | "na" | "NaN" | "nan" | "NULL" | "null" | "None"
        )
    {
        return Ok(None);
    }
    if let Ok(v) = t.parse::<i32>() {
        if v < 0 {
            Ok(None)
        } else {
            Ok(Some(v))
        }
    } else {
        Err(())
    }
}

fn finalize_character_matrix(
    names: Vec<String>,
    raw_mat: Vec<i8>,
    n: usize,
    l: usize,
    feature_names: Vec<String>,
) -> Result<DiscreteCharacterMatrix, String> {
    if n == 0 || l == 0 {
        return Err("Discrete character matrix must have at least 1 taxon and 1 character.".to_string());
    }
    if raw_mat.len() != n * l {
        return Err(format!(
            "Internal matrix dimension mismatch: got {} entries for {} x {}",
            raw_mat.len(),
            n,
            l
        ));
    }
    let mut max_st: i8 = 1;
    for &v in &raw_mat {
        if v > max_st {
            max_st = v;
        }
    }
    let k_states = ((max_st as usize) + 1).clamp(2, MAX_DISCRETE_STATES);
    let all_mask = all_states_mask(k_states);
    let mut bitmasks = vec![all_mask; n * l];
    let mut counts = vec![[0_u32; MAX_DISCRETE_STATES]; l];
    for i in 0..n {
        let off = i * l;
        for s in 0..l {
            let st = raw_mat[off + s];
            if st >= 0 && (st as usize) < k_states {
                bitmasks[off + s] = 1_u32 << (st as u32);
                counts[s][st as usize] += 1;
            } else {
                bitmasks[off + s] = all_mask;
            }
        }
    }
    let mut majority_state = vec![0_i8; l];
    for s in 0..l {
        let mut best_a = 0_i8;
        let mut best_c = counts[s][0];
        for a in 1..k_states {
            if counts[s][a] > best_c {
                best_c = counts[s][a];
                best_a = a as i8;
            }
        }
        majority_state[s] = best_a;
    }

    let feats = if feature_names.len() == l {
        feature_names
    } else {
        (0..l).map(|s| format!("F{s}")).collect()
    };
    let mut chrom_dict: HashMap<String, u16> = HashMap::new();
    let chrom_ids: Vec<u16> = feats
        .iter()
        .map(|f| parse_chrom_id_from_feature(f, &mut chrom_dict))
        .collect();

    Ok(DiscreteCharacterMatrix {
        names,
        mat: raw_mat,
        bitmasks,
        majority_state,
        n_taxa: n,
        n_sites: l,
        k_states,
        feature_names: feats,
        chrom_ids,
    })
}

/// Encode symbol sequences (`0..9, A..V` or pure `A..V` morphological states; `?`/`-`/`.` as missing)
/// into a `DiscreteCharacterMatrix`.
pub fn encode_discrete_sequences(
    names: &[String],
    seqs: &[String],
) -> Result<DiscreteCharacterMatrix, String> {
    if names.is_empty() || seqs.is_empty() || names.len() != seqs.len() {
        return Err("Empty or mismatched names/sequences in discrete alignment.".to_string());
    }
    let n = names.len();
    let l = seqs[0].len();
    if l == 0 {
        return Err("Zero-length discrete sequence alignment.".to_string());
    }

    let mut has_digit = false;
    let mut has_alpha_non_nx = false;
    for s in seqs {
        if s.len() != l {
            return Err(format!(
                "Sequence length mismatch in discrete alignment: expected {}, got {}",
                l,
                s.len()
            ));
        }
        for &b in s.as_bytes() {
            let up = b.to_ascii_uppercase();
            if up.is_ascii_digit() {
                has_digit = true;
            } else if (b'A'..=b'V').contains(&up) && up != b'N' {
                has_alpha_non_nx = true;
            }
        }
    }

    let mut raw_mat = vec![-1_i8; n * l];
    for (i, s) in seqs.iter().enumerate() {
        let off = i * l;
        for (j, &b) in s.as_bytes().iter().enumerate() {
            let up = b.to_ascii_uppercase();
            let st: i8 = if matches!(up, b'-' | b'?' | b'.' | b'*') {
                -1
            } else if has_digit && !has_alpha_non_nx {
                // Pure numeric 0..9 encoding: treat 'N'/'X' and all non-digits as missing
                if up.is_ascii_digit() {
                    (up - b'0') as i8
                } else {
                    -1
                }
            } else if has_digit {
                // Base-32 symbol encoding: '0'..'9' -> 0..9, 'A'..'V' -> 10..31
                if up.is_ascii_digit() {
                    (up - b'0') as i8
                } else if (b'A'..=b'V').contains(&up) {
                    (10 + (up - b'A')) as i8
                } else {
                    -1
                }
            } else if (b'A'..=b'V').contains(&up) {
                // Pure alphabetical symbol matrix ('A'..'V' -> 0..21)
                (up - b'A') as i8
            } else {
                -1
            };
            raw_mat[off + j] = st;
        }
    }

    finalize_character_matrix(names.to_vec(), raw_mat, n, l, Vec::new())
}

/// Parse a character matrix from **CSV**, **TSV**, **FASTA**, **PHYLIP**, or **NEXUS** text.
///
/// For CSV/TSV inputs:
/// - Standard orientation: rows are taxa/cells (`col 0 = taxon_id`, `cols 1..=M = integer states 0..31`
///   or `?`/`-`/`.`/`NA` as missing), with an optional header row of feature names.
/// - Genomic bin-by-cell orientation (`chr,start,end,cell_1,cell_2,...`): automatically detected and
///   transposed into `N_cells x M_bins` with chromosomal coordinates preserved in `feature_names`!
pub fn parse_discrete_str(text: &str) -> Result<DiscreteCharacterMatrix, String> {
    let first_non_empty = text
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty() && !line.starts_with('#') || line.starts_with("#NEXUS") || line.starts_with("#nexus"))
        .ok_or_else(|| "Empty discrete input.".to_string())?;

    if first_non_empty.starts_with('>')
        || first_non_empty.eq_ignore_ascii_case("#NEXUS")
        || (!first_non_empty.contains(',') && !first_non_empty.contains('\t'))
    {
        if let Ok((names, seqs)) = parse_alignment_str(text) {
            return encode_discrete_sequences(&names, &seqs);
        }
    }

    let delim = if first_non_empty.contains('\t') && !first_non_empty.contains(',') {
        '\t'
    } else {
        ','
    };

    let lines: Vec<&str> = text
        .lines()
        .map(|l| l.trim_end_matches('\r'))
        .filter(|l| !l.trim().is_empty())
        .collect();
    if lines.is_empty() {
        return Err("No non-empty rows in CSV/TSV character matrix.".to_string());
    }

    let row0: Vec<&str> = lines[0].split(delim).map(str::trim).collect();
    if row0.len() < 2 {
        let (names, seqs) = parse_alignment_str(text)?;
        return encode_discrete_sequences(&names, &seqs);
    }

    // Auto-detect R `write.table` feature-by-cell CSV (`cell1,cell2,...` on row 0; `locus,v1,v2,...` on rows 1..)
    if lines.len() >= 2 {
        let row1: Vec<&str> = lines[1].split(delim).map(str::trim).collect();
        if row1.len() == row0.len() + 1 && parse_cell_token(row1[1]).is_ok() {
            let names: Vec<String> = row0
                .iter()
                .map(|s| s.trim().trim_matches('"').to_string())
                .collect();
            let n = names.len();
            let l = lines.len() - 1;
            let mut feature_names = Vec::with_capacity(l);
            let mut raw_mat = vec![-1_i8; n * l];
            for (s_idx, &line) in lines[1..].iter().enumerate() {
                let toks: Vec<&str> = line.split(delim).collect();
                if toks.len() != n + 1 {
                    return Err(format!(
                        "Row {} in R-style feature-by-cell CSV has {} columns, expected {}",
                        s_idx + 2,
                        toks.len(),
                        n + 1
                    ));
                }
                feature_names.push(toks[0].trim().trim_matches('"').to_string());
                for c_idx in 0..n {
                    let st = match parse_cell_token(toks[c_idx + 1]) {
                        Ok(Some(v)) => v.min(31) as i8,
                        Ok(None) => -1_i8,
                        Err(()) => {
                            return Err(format!(
                                "Invalid state token '{}' at row {}, col {}",
                                toks[c_idx + 1],
                                s_idx + 2,
                                c_idx + 2
                            ))
                        }
                    };
                    raw_mat[c_idx * l + s_idx] = st;
                }
            }
            // If all loci follow `chr_start_end` coordinates (e.g. R `write.table` lexicographically sorted bins),
            // restore physical chromosomal coordinate order `(chrom_rank, start_bp, end_bp)`.
            let parsed_coords: Option<Vec<(u32, u64, u64, usize)>> = feature_names
                .iter()
                .enumerate()
                .map(|(idx, fnm)| {
                    let mut sp = fnm.split('_');
                    let c_str = sp.next()?;
                    let st_bp: u64 = sp.next()?.parse().ok()?;
                    let en_bp: u64 = sp.next()?.parse().ok()?;
                    if sp.next().is_some() {
                        return None;
                    }
                    let c_clean = c_str
                        .strip_prefix("chr")
                        .or_else(|| c_str.strip_prefix("CHR"))
                        .unwrap_or(c_str);
                    let c_rank: u32 = match c_clean {
                        "X" | "x" => 23,
                        "Y" | "y" => 24,
                        "M" | "MT" | "m" | "mt" => 25,
                        _ => c_clean.parse().ok()?,
                    };
                    Some((c_rank, st_bp, en_bp, idx))
                })
                .collect();
            if let Some(mut coords) = parsed_coords {
                coords.sort_by_key(|&(cr, sb, eb, oi)| (cr, sb, eb, oi));
                let is_identity = coords.iter().enumerate().all(|(new_i, t)| t.3 == new_i);
                if !is_identity {
                    let mut sorted_feats = Vec::with_capacity(l);
                    let mut sorted_mat = vec![-1_i8; n * l];
                    for (new_s, &(_, _, _, old_s)) in coords.iter().enumerate() {
                        sorted_feats.push(feature_names[old_s].clone());
                        for c_idx in 0..n {
                            sorted_mat[c_idx * l + new_s] = raw_mat[c_idx * l + old_s];
                        }
                    }
                    feature_names = sorted_feats;
                    raw_mat = sorted_mat;
                }
            }
            return finalize_character_matrix(names, raw_mat, n, l, feature_names);
        }
    }

    // Auto-detect Sitka-style genomic bin-by-cell CSV (`chr,start,end,cell1,cell2,...`)
    if row0.len() >= 5
        && row0[0].trim_matches('"').eq_ignore_ascii_case("chr")
        && row0[1].trim_matches('"').eq_ignore_ascii_case("start")
        && row0[2].trim_matches('"').eq_ignore_ascii_case("end")
    {
        let names: Vec<String> = row0[3..]
            .iter()
            .map(|s| s.trim().trim_matches('"').to_string())
            .collect();
        let n = names.len();
        let l = lines.len().saturating_sub(1);
        let mut feature_names = Vec::with_capacity(l);
        let mut raw_mat = vec![-1_i8; n * l];
        for (s_idx, &line) in lines[1..].iter().enumerate() {
            let toks: Vec<&str> = line.split(delim).collect();
            if toks.len() != n + 3 {
                return Err(format!(
                    "Row {} in bin-by-cell CSV has {} columns, expected {}",
                    s_idx + 2,
                    toks.len(),
                    n + 3
                ));
            }
            let chr_s = toks[0].trim().trim_matches('"');
            let start_s = toks[1].trim().trim_matches('"');
            let end_s = toks[2].trim().trim_matches('"');
            feature_names.push(format!("{chr_s}_{start_s}_{end_s}"));
            for c_idx in 0..n {
                let st = match parse_cell_token(toks[c_idx + 3]) {
                    Ok(Some(v)) => v.min(31) as i8,
                    Ok(None) => -1_i8,
                    Err(()) => {
                        return Err(format!(
                            "Invalid state token '{}' at row {}, col {}",
                            toks[c_idx + 3],
                            s_idx + 2,
                            c_idx + 4
                        ))
                    }
                };
                raw_mat[c_idx * l + s_idx] = st;
            }
        }
        return finalize_character_matrix(names, raw_mat, n, l, feature_names);
    }

    // Check whether row 0 is a header row in standard taxon-by-feature orientation
    let col0_low = row0[0].trim_matches('"').to_ascii_lowercase();
    let known_header_col0 = matches!(
        col0_low.as_str(),
        "" | "taxon" | "taxa" | "cell" | "cells" | "cell_id" | "id" | "name" | "sample" | "clone"
    );
    let any_non_numeric_in_row0 = row0[1..].iter().any(|t| parse_cell_token(t).is_err());
    let has_header = known_header_col0 || any_non_numeric_in_row0;

    let (feature_names, start_row) = if has_header {
        let feats: Vec<String> = row0[1..]
            .iter()
            .map(|s| s.trim().trim_matches('"').to_string())
            .collect();
        (feats, 1usize)
    } else {
        (Vec::new(), 0usize)
    };

    if lines.len() <= start_row {
        return Err("CSV/TSV character matrix contains a header but zero taxon rows.".to_string());
    }
    let l = row0.len() - 1;
    let n = lines.len() - start_row;
    let mut names = Vec::with_capacity(n);
    let mut raw_mat = vec![-1_i8; n * l];

    for (i, &line) in lines[start_row..].iter().enumerate() {
        let toks: Vec<&str> = line.split(delim).collect();
        if toks.len() != l + 1 {
            return Err(format!(
                "Row {} in CSV/TSV character matrix has {} columns, expected {}",
                i + start_row + 1,
                toks.len(),
                l + 1
            ));
        }
        names.push(toks[0].trim().trim_matches('"').to_string());
        let off = i * l;
        for s in 0..l {
            let st = match parse_cell_token(toks[s + 1]) {
                Ok(Some(v)) => v.min(31) as i8,
                Ok(None) => -1_i8,
                Err(()) => {
                    return Err(format!(
                        "Invalid discrete state '{}' at row {}, column {}",
                        toks[s + 1],
                        i + start_row + 1,
                        s + 2
                    ))
                }
            };
            raw_mat[off + s] = st;
        }
    }

    finalize_character_matrix(names, raw_mat, n, l, feature_names)
}

pub fn parse_discrete_file<P: AsRef<Path>>(path: P) -> Result<DiscreteCharacterMatrix, String> {
    let text = fs::read_to_string(path.as_ref())
        .map_err(|e| format!("Failed to read discrete matrix '{}': {e}", path.as_ref().display()))?;
    parse_discrete_str(&text)
}

/// Estimate the parameter-free Kraft-exact substitution / step model and collinearity structure
/// directly from the observed character matrix `mat` (zero DNA or protein priors).
pub fn estimate_discrete_subst_model(
    char_mat: &DiscreteCharacterMatrix,
    mode: DiscreteMode,
) -> DiscreteSubstModel {
    let n = char_mat.n_taxa;
    let l = char_mat.n_sites;
    let k = char_mat.k_states.clamp(2, MAX_DISCRETE_STATES);
    let all_mask = all_states_mask(k);
    let mat = &char_mat.mat;

    // 1. Laplace-1 empirical stationary state distribution pi[0..k]
    let mut st_counts = [0_u64; MAX_DISCRETE_STATES];
    for &v in mat {
        if v >= 0 && (v as usize) < k {
            st_counts[v as usize] += 1;
        }
    }
    let tot_st: f64 = (0..k).map(|a| (st_counts[a] as f64) + 1.0).sum();
    let mut pi = [0.0_f64; MAX_DISCRETE_STATES];
    for a in 0..k {
        pi[a] = ((st_counts[a] as f64) + 1.0) / tot_st;
    }

    // 2. Detect collinear chromosomal bin structure in Ordered mode (`M >= 128`)
    let mut adj_diff_frac = vec![1.0_f64; l];
    let mut var_adj_conc = Vec::new();
    if mode == DiscreteMode::Ordered && l >= 2 && n >= 2 {
        for s in 1..l {
            if char_mat.chrom_ids[s] != char_mat.chrom_ids[s - 1] {
                adj_diff_frac[s] = 1.0;
                continue;
            }
            let mut n_both = 0usize;
            let mut n_diff = 0usize;
            let mut has_var = false;
            let mut first_v = -1_i8;
            for i in 0..n {
                let v_prev = mat[i * l + s - 1];
                let v_cur = mat[i * l + s];
                if v_cur >= 0 {
                    if first_v < 0 {
                        first_v = v_cur;
                    } else if v_cur != first_v {
                        has_var = true;
                    }
                }
                if v_prev >= 0 && v_cur >= 0 {
                    n_both += 1;
                    if v_prev != v_cur {
                        n_diff += 1;
                    }
                }
            }
            if n_both > 0 {
                let df = (n_diff as f64) / (n_both as f64);
                adj_diff_frac[s] = df;
                if has_var {
                    var_adj_conc.push(1.0 - df);
                }
            }
        }
    }

    let med_conc = if !var_adj_conc.is_empty() {
        median_f64(&mut var_adj_conc)
    } else {
        0.0
    };
    let is_collinear_bins =
        mode == DiscreteMode::Ordered && l >= 128 && var_adj_conc.len() >= 32 && med_conc >= 0.90;

    let mut dist_weights = vec![1.0_f64; l];
    let tree_weights = vec![1.0_f64; l];
    let l_eff_seg: f64;
    let mut avg_block_len = 1.0_f64;

    if is_collinear_bins {
        // Partition ordered bins into empirical copy-number changepoint blocks
        let mut block_ranges: Vec<(usize, usize)> = Vec::new();
        let mut b_start = 0usize;
        let mut cum_drift = 0.0_f64;
        for s in 1..l {
            let chrom_break = char_mat.chrom_ids[s] != char_mat.chrom_ids[s - 1];
            let df = adj_diff_frac[s];
            cum_drift += df;
            if chrom_break || df >= 0.025 || cum_drift >= 0.08 {
                block_ranges.push((b_start, s - 1));
                b_start = s;
                cum_drift = 0.0;
            }
        }
        block_ranges.push((b_start, l - 1));
        l_eff_seg = (block_ranges.len() as f64).max(4.0);
        avg_block_len = ((l as f64) / l_eff_seg).max(1.0);
        for &(a, b) in &block_ranges {
            let blen = (b - a + 1) as f64;
            let w = 1.0 / blen;
            for s in a..=b {
                dist_weights[s] = w;
            }
        }
    } else {
        let total_valid: usize = mat.iter().filter(|&&x| x >= 0).count();
        l_eff_seg = ((total_valid as f64) / (n.max(1) as f64)).clamp(1.0, l as f64);
    }

    // 3. Sample nearest-neighbor taxon pairs to estimate empirical transitions / step size
    let max_sample = n.min(256);
    let step_i = (n / max_sample.max(1)).max(1);
    let sample_indices: Vec<usize> = (0..n).step_by(step_i).take(max_sample).collect();
    let mut off_counts = [[0_u64; MAX_DISCRETE_STATES]; MAX_DISCRETE_STATES];
    let mut total_abs_delta = 0_u64;
    let mut total_neq = 0_u64;

    if sample_indices.len() >= 2 {
        let site_stride = (l / 1024).max(1);
        for &i in &sample_indices {
            let ri = &mat[i * l..(i + 1) * l];
            let mut best_j = None;
            let mut best_d = f64::INFINITY;
            for &j in &sample_indices {
                if i == j {
                    continue;
                }
                let rj = &mat[j * l..(j + 1) * l];
                let mut diff = 0usize;
                let mut val = 0usize;
                for s in (0..l).step_by(site_stride) {
                    let a = ri[s];
                    let b = rj[s];
                    if a >= 0 && b >= 0 {
                        val += 1;
                        if a != b {
                            diff += match mode {
                                DiscreteMode::Unordered => 1,
                                DiscreteMode::Ordered => (a - b).unsigned_abs() as usize,
                            };
                        }
                    }
                }
                if val > 0 {
                    let d = (diff as f64) / (val as f64);
                    if d > 0.0 && d < best_d {
                        best_d = d;
                        best_j = Some(j);
                    }
                }
            }
            if let Some(jn) = best_j {
                let rj = &mat[jn * l..(jn + 1) * l];
                for s in (0..l).step_by(site_stride) {
                    let a = ri[s];
                    let b = rj[s];
                    if a >= 0 && b >= 0 && a != b && (a as usize) < k && (b as usize) < k {
                        off_counts[a as usize][b as usize] += 1;
                        off_counts[b as usize][a as usize] += 1;
                        total_abs_delta += (a - b).unsigned_abs() as u64;
                        total_neq += 1;
                    }
                }
            }
        }
    }

    let mut cost = [[0.0_f64; MAX_DISCRETE_STATES]; MAX_DISCRETE_STATES];
    let mut step_gamma = 0.0_f64;

    match mode {
        DiscreteMode::Unordered => {
            if k == 2 {
                // Binary alphabet: knowing a != b uniquely determines b = 1 - a -> 0.0 nats
                cost[0][1] = 0.0;
                cost[1][0] = 0.0;
            } else {
                for a in 0..k {
                    let row_tot: f64 = (0..k)
                        .filter(|&b| b != a)
                        .map(|b| (off_counts[a][b] as f64) + 1.0)
                        .sum();
                    for b in 0..k {
                        if b != a {
                            let p_ab = ((off_counts[a][b] as f64) + 1.0) / row_tot;
                            cost[a][b] = (-p_ab.ln()).max(0.0);
                        }
                    }
                }
            }
        }
        DiscreteMode::Ordered => {
            if k == 2 {
                cost[0][1] = 0.0;
                cost[1][0] = 0.0;
                step_gamma = 0.0;
            } else {
                let mean_delta = if total_neq >= 8 {
                    (total_abs_delta as f64) / (total_neq as f64)
                } else {
                    1.30
                };
                step_gamma = ((mean_delta - 1.0) / mean_delta.max(1.0001)).clamp(0.12, 0.68);
                for a in 0..k {
                    let z_a: f64 = (0..k)
                        .filter(|&b| b != a)
                        .map(|b| {
                            let delta = a.abs_diff(b);
                            step_gamma.powi((delta as i32) - 1)
                        })
                        .sum();
                    let ln_za = z_a.ln();
                    let ln_inv_g = (-step_gamma.ln()).max(0.0);
                    for b in 0..k {
                        if b != a {
                            let delta = a.abs_diff(b);
                            cost[a][b] = (ln_za + ((delta - 1) as f64) * ln_inv_g).max(0.0);
                        }
                    }
                }
            }
        }
    }

    DiscreteSubstModel {
        mode,
        k,
        all_mask,
        pi,
        cost,
        step_gamma,
        is_collinear_bins,
        l_eff_seg,
        avg_block_len,
        dist_weights,
        tree_weights,
    }
}

/// Lossless two-part branch code length `L_branch(e)` in nats between parent state vector `u_st`
/// and child state vector `v_st` (restricted to sites where neither mask is `all_mask`).
///
/// - In `Unordered` mode: exact Combinadic site-position code + Kraft-normalized state code.
/// - In `Ordered` mode: Kraft-exact union code `ln(2) + min(L_point(e), L_RLE(e))`, where
///   `L_RLE(e)` transmits contiguous chromosomal runs `[a_q, b_q]` of constant copy-number shift
///   `delta_q = v_st[s] - u_st[s] != 0` via the exact Combinadic endpoint bijection `ln C(M_val + r, 2r)`.
pub fn ordered_or_discrete_branch_code_nats(
    u_st: &[i8],
    v_st: &[i8],
    u_valid: &[bool],
    v_valid: &[bool],
    chrom_ids: &[u16],
    l_eff: f64,
    subst: &DiscreteSubstModel,
) -> (f64, usize, usize) {
    let l = u_st.len();
    let mut m_pt = 0usize;
    let mut pt_state_nats = 0.0_f64;

    let mut r_runs = 0usize;
    let mut rle_state_nats = 0.0_f64;
    let mut prev_delta: i16 = 0;
    let mut prev_chr: u16 = u16::MAX;
    let mut m_val = 0usize;

    for s in 0..l {
        if !u_valid[s] || !v_valid[s] {
            continue;
        }
        let a = u_st[s];
        let b = v_st[s];
        if a < 0 || b < 0 {
            continue;
        }
        m_val += 1;
        let chr = chrom_ids[s];
        let delta = (b as i16) - (a as i16);
        if delta != 0 {
            m_pt += 1;
            let c_ab = subst.cost[(a as usize).min(subst.k - 1)][(b as usize).min(subst.k - 1)];
            pt_state_nats += c_ab;
            if delta != prev_delta || chr != prev_chr {
                r_runs += 1;
                rle_state_nats += c_ab;
            }
        }
        prev_delta = delta;
        prev_chr = chr;
    }

    if m_pt == 0 {
        return (0.0, 0, 0);
    }

    let eff_len = (m_val as f64).max(l_eff).max(1.0);
    let m_f = m_pt as f64;
    let m_pos = m_f.min(0.49 * eff_len);
    let mut l_point = universal_int_nats(m_f) + log_binom(eff_len, m_pos) + pt_state_nats;
    if m_f > 0.49 * eff_len {
        l_point += (m_f - 0.49 * eff_len) * 2.0_f64.ln();
    }

    if subst.mode == DiscreteMode::Unordered {
        return (l_point, m_pt, m_pt);
    }

    let r_f = r_runs as f64;
    let top_n = eff_len + r_f;
    let choose_2r = (2.0 * r_f).min(0.49 * top_n);
    let mut l_rle = universal_int_nats(r_f) + log_binom(top_n, choose_2r) + rle_state_nats;
    if 2.0 * r_f > 0.49 * top_n {
        l_rle += (2.0 * r_f - 0.49 * top_n) * 2.0_f64.ln();
    }

    let union_nats = 2.0_f64.ln() + l_point.min(l_rle);
    (union_nats, m_pt, r_runs)
}

#[inline]
fn discrete_leaf_dist(
    ri: &[i8],
    rj: &[i8],
    weights: &[f64],
    l_eff_target: f64,
    subst: &DiscreteSubstModel,
) -> f64 {
    let mut w_val = 0.0_f64;
    let mut w_diff = 0.0_f64;
    for s in 0..ri.len() {
        let a = ri[s];
        let b = rj[s];
        if a >= 0 && b >= 0 {
            let w = weights[s];
            w_val += w;
            if a != b {
                match subst.mode {
                    DiscreteMode::Unordered => {
                        w_diff += w;
                    }
                    DiscreteMode::Ordered => {
                        w_diff += w * ((a - b).unsigned_abs() as f64);
                    }
                }
            }
        }
    }
    if w_val <= 1e-12 {
        return l_eff_target * 0.5;
    }
    match subst.mode {
        DiscreteMode::Unordered => {
            let p = (w_diff / w_val).clamp(0.0, 0.999);
            let q_max = ((subst.k - 1) as f64) / (subst.k as f64);
            let ratio = (p / q_max).min(0.96);
            let alpha = 2.0_f64;
            let t = q_max * alpha * ((1.0 - ratio).powf(-1.0 / alpha) - 1.0);
            t * l_eff_target
        }
        DiscreteMode::Ordered => {
            // Exact weighted Wagner L1 additive step distance
            (w_diff / w_val) * l_eff_target
        }
    }
}

#[inline]
fn discrete_mask_dist(
    si: &[u32],
    sj: &[u32],
    weights: &[f64],
    l_eff_target: f64,
    subst: &DiscreteSubstModel,
) -> f64 {
    let all_m = subst.all_mask;
    let mut w_val = 0.0_f64;
    let mut w_diff = 0.0_f64;
    for s in 0..si.len() {
        let mi = si[s];
        let mj = sj[s];
        if mi != all_m && mj != all_m && mi != 0 && mj != 0 {
            let w = weights[s];
            w_val += w;
            if (mi & mj) == 0 {
                match subst.mode {
                    DiscreteMode::Unordered => {
                        w_diff += w;
                    }
                    DiscreteMode::Ordered => {
                        let loi = mi.trailing_zeros() as usize;
                        let hii = (31 - mi.leading_zeros()) as usize;
                        let loj = mj.trailing_zeros() as usize;
                        let hij = (31 - mj.leading_zeros()) as usize;
                        let step = if hii < loj {
                            loj - hii
                        } else {
                            loi.saturating_sub(hij)
                        };
                        w_diff += w * (step as f64);
                    }
                }
            }
        }
    }
    if w_val <= 1e-12 {
        return l_eff_target * 0.5;
    }
    match subst.mode {
        DiscreteMode::Unordered => {
            let p = (w_diff / w_val).clamp(0.0, 0.999);
            let q_max = ((subst.k - 1) as f64) / (subst.k as f64);
            let ratio = (p / q_max).min(0.96);
            let alpha = 2.0_f64;
            q_max * alpha * ((1.0 - ratio).powf(-1.0 / alpha) - 1.0) * l_eff_target
        }
        DiscreteMode::Ordered => (w_diff / w_val) * l_eff_target,
    }
}

#[inline]
fn merge_discrete_steiner_median(
    u: &[u32],
    v: &[u32],
    r: &[u32],
    subst: &DiscreteSubstModel,
) -> Vec<u32> {
    let l = u.len();
    let all_m = subst.all_mask;
    let mut out = vec![all_m; l];
    match subst.mode {
        DiscreteMode::Unordered => {
            for s in 0..l {
                let (mu, mv, mr) = (u[s], v[s], r[s]);
                let i3 = mu & mv & mr;
                if i3 != 0 {
                    out[s] = i3;
                } else {
                    let p_uv = mu & mv;
                    let p_ur = mu & mr;
                    let p_vr = mv & mr;
                    let maj = p_uv | p_ur | p_vr;
                    out[s] = if maj != 0 { maj } else { mu | mv | mr };
                }
            }
        }
        DiscreteMode::Ordered => {
            for s in 0..l {
                let mut best_cost = u32::MAX;
                let mut lo_opt = 0usize;
                let mut hi_opt = 0usize;
                let mut any_valid = false;
                let masks = [u[s], v[s], r[s]];
                let mut ranges = [(0usize, 0usize, false); 3];
                for (idx, &m) in masks.iter().enumerate() {
                    if m != all_m && m != 0 {
                        any_valid = true;
                        ranges[idx] = (
                            m.trailing_zeros() as usize,
                            (31 - m.leading_zeros()) as usize,
                            true,
                        );
                    }
                }
                if !any_valid {
                    out[s] = all_m;
                    continue;
                }
                for x in 0..subst.k {
                    let mut c = 0_u32;
                    for &(lo, hi, val) in &ranges {
                        if val {
                            c += (lo.saturating_sub(x) + x.saturating_sub(hi)) as u32;
                        }
                    }
                    if c < best_cost {
                        best_cost = c;
                        lo_opt = x;
                        hi_opt = x;
                    } else if c == best_cost {
                        hi_opt = x;
                    }
                }
                out[s] = interval_mask(lo_opt, hi_opt);
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn agglomerate_discrete_bionj_from_dist(
    active_ids: &[usize],
    mut dist: Vec<f64>,
    mut seqs_map: Option<&mut HashMap<usize, Vec<u32>>>,
    subst: &DiscreteSubstModel,
    adj: &mut BTreeMap<usize, Vec<usize>>,
    node_labels: &mut BTreeMap<usize, String>,
    next_node: &mut usize,
) {
    let m = active_ids.len();
    if m == 0 {
        return;
    }
    if m == 1 {
        adj.entry(active_ids[0]).or_default();
        return;
    }
    if m == 2 {
        let u = active_ids[0];
        let v = active_ids[1];
        adj.entry(u).or_default().push(v);
        adj.entry(v).or_default().push(u);
        return;
    }
    if m == 3 {
        let w = *next_node;
        *next_node += 1;
        node_labels.insert(w, format!("N{w}"));
        let (u0, u1, u2) = (active_ids[0], active_ids[1], active_ids[2]);
        adj.insert(w, vec![u0, u1, u2]);
        for &x in &[u0, u1, u2] {
            adj.entry(x).or_default().push(w);
        }
        if let Some(ref mut smap) = seqs_map {
            let med = merge_discrete_steiner_median(&smap[&u0], &smap[&u1], &smap[&u2], subst);
            smap.insert(w, med);
        }
        return;
    }

    let mut vmat = dist.clone();
    let mut alive = vec![true; m];
    let mut cur_ids = active_ids.to_vec();
    let mut r_sum = vec![0.0_f64; m];
    for i in 0..m {
        let mut s = 0.0_f64;
        for j in 0..m {
            if i != j {
                s += dist[i * m + j];
            }
        }
        r_sum[i] = s;
    }

    let mut n_rem = m;
    while n_rem > 3 {
        let denom = (n_rem - 2) as f64;
        let mut best_q = f64::INFINITY;
        let mut best_i = 0usize;
        let mut best_j = 1usize;
        for i in 0..m {
            if !alive[i] {
                continue;
            }
            let ri = r_sum[i];
            for j in (i + 1)..m {
                if alive[j] {
                    let q = denom * dist[i * m + j] - ri - r_sum[j];
                    if q < best_q {
                        best_q = q;
                        best_i = i;
                        best_j = j;
                    }
                }
            }
        }

        let dij = dist[best_i * m + best_j].max(1e-6);
        let u_node = cur_ids[best_i];
        let v_node = cur_ids[best_j];
        let w_node = *next_node;
        *next_node += 1;
        node_labels.insert(w_node, format!("N{w_node}"));
        adj.insert(w_node, vec![u_node, v_node]);
        adj.entry(u_node).or_default().push(w_node);
        adj.entry(v_node).or_default().push(w_node);

        if let Some(ref mut smap) = seqs_map {
            let mut best_k = 0usize;
            let mut best_dk = f64::INFINITY;
            for k_idx in 0..m {
                if alive[k_idx] && k_idx != best_i && k_idx != best_j {
                    let dk = dist[best_i * m + k_idx] + dist[best_j * m + k_idx];
                    if dk < best_dk {
                        best_dk = dk;
                        best_k = k_idx;
                    }
                }
            }
            let r_node = cur_ids[best_k];
            let med_seq =
                merge_discrete_steiner_median(&smap[&u_node], &smap[&v_node], &smap[&r_node], subst);
            smap.insert(w_node, med_seq);
        }

        let li_raw = 0.5 * dij + (r_sum[best_i] - r_sum[best_j]) / (2.0 * denom);
        let li = li_raw.max(0.0);
        let lj = (dij - li).max(0.0);
        let vij = vmat[best_i * m + best_j];
        let lambda = if vij > 1e-12 {
            let mut v_diff = 0.0_f64;
            for k_idx in 0..m {
                if alive[k_idx] && k_idx != best_i && k_idx != best_j {
                    v_diff += vmat[best_j * m + k_idx] - vmat[best_i * m + k_idx];
                }
            }
            (0.5 + v_diff / (2.0 * denom * vij)).clamp(0.0, 1.0)
        } else {
            0.5
        };

        let mut new_r_i = 0.0_f64;
        for k_idx in 0..m {
            if alive[k_idx] && k_idx != best_i && k_idx != best_j {
                let d_ik = dist[best_i * m + k_idx];
                let d_jk = dist[best_j * m + k_idx];
                let d_wk = (lambda * (d_ik - li) + (1.0 - lambda) * (d_jk - lj)).max(0.0);
                let v_wk = (lambda * vmat[best_i * m + k_idx]
                    + (1.0 - lambda) * vmat[best_j * m + k_idx]
                    - lambda * (1.0 - lambda) * vij)
                    .max(0.0);
                r_sum[k_idx] = r_sum[k_idx] - d_ik - d_jk + d_wk;
                dist[best_i * m + k_idx] = d_wk;
                dist[k_idx * m + best_i] = d_wk;
                vmat[best_i * m + k_idx] = v_wk;
                vmat[k_idx * m + best_i] = v_wk;
                new_r_i += d_wk;
            }
        }
        alive[best_j] = false;
        dist[best_i * m + best_i] = 0.0;
        vmat[best_i * m + best_i] = 0.0;
        r_sum[best_i] = new_r_i;
        cur_ids[best_i] = w_node;
        n_rem -= 1;
    }

    let rem_idx: Vec<usize> = (0..m).filter(|&i| alive[i]).collect();
    if rem_idx.len() == 3 {
        let w = *next_node;
        *next_node += 1;
        node_labels.insert(w, format!("N{w}"));
        let (u0, u1, u2) = (cur_ids[rem_idx[0]], cur_ids[rem_idx[1]], cur_ids[rem_idx[2]]);
        adj.insert(w, vec![u0, u1, u2]);
        for &x in &[u0, u1, u2] {
            adj.entry(x).or_default().push(w);
        }
        if let Some(ref mut smap) = seqs_map {
            let med = merge_discrete_steiner_median(&smap[&u0], &smap[&u1], &smap[&u2], subst);
            smap.insert(w, med);
        }
    }
}

fn agglomerate_discrete_bionj(
    active_ids: &[usize],
    seqs_map: &mut HashMap<usize, Vec<u32>>,
    subst: &DiscreteSubstModel,
    adj: &mut BTreeMap<usize, Vec<usize>>,
    node_labels: &mut BTreeMap<usize, String>,
    next_node: &mut usize,
) {
    let m = active_ids.len();
    if m <= 3 {
        agglomerate_discrete_bionj_from_dist(
            active_ids,
            Vec::new(),
            Some(seqs_map),
            subst,
            adj,
            node_labels,
            next_node,
        );
        return;
    }
    let mut dist = vec![0.0_f64; m * m];
    #[cfg(not(target_arch = "wasm32"))]
    {
        let rows: Vec<Vec<f64>> = (0..m)
            .into_par_iter()
            .map(|i| {
                let mut r = vec![0.0_f64; m];
                let si = &seqs_map[&active_ids[i]];
                for j in (i + 1)..m {
                    let sj = &seqs_map[&active_ids[j]];
                    r[j] = discrete_mask_dist(
                        si,
                        sj,
                        &subst.dist_weights,
                        subst.l_eff_seg,
                        subst,
                    );
                }
                r
            })
            .collect();
        for i in 0..m {
            for j in (i + 1)..m {
                dist[i * m + j] = rows[i][j];
                dist[j * m + i] = rows[i][j];
            }
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        for i in 0..m {
            let si = &seqs_map[&active_ids[i]];
            for j in (i + 1)..m {
                let sj = &seqs_map[&active_ids[j]];
                let d = discrete_mask_dist(si, sj, &subst.dist_weights, subst.l_eff_seg, subst);
                dist[i * m + j] = d;
                dist[j * m + i] = d;
            }
        }
    }
    agglomerate_discrete_bionj_from_dist(
        active_ids,
        dist,
        Some(seqs_map),
        subst,
        adj,
        node_labels,
        next_node,
    );
}

fn partition_discrete_voronoi(
    items: Vec<usize>,
    mat: &[i8],
    l: usize,
    subst: &DiscreteSubstModel,
    max_cluster: usize,
    out: &mut Vec<Vec<usize>>,
) {
    let m = items.len();
    if m <= max_cluster {
        if !items.is_empty() {
            out.push(items);
        }
        return;
    }
    let k_land = (m / (max_cluster / 2).max(32)).clamp(4, 32);
    let row_of = |idx: usize| -> &[i8] { &mat[idx * l..(idx + 1) * l] };

    let p0 = items[0];
    let mut l1 = p0;
    let mut max_d0 = -1.0_f64;
    for &x in &items {
        let d = discrete_leaf_dist(row_of(p0), row_of(x), &subst.dist_weights, subst.l_eff_seg, subst);
        if d > max_d0 {
            max_d0 = d;
            l1 = x;
        }
    }

    let mut landmarks = Vec::with_capacity(k_land);
    landmarks.push(l1);
    let mut min_dist = vec![f64::INFINITY; m];
    let mut best_k = vec![0usize; m];

    for k_idx in 0..k_land {
        let lk = landmarks[k_idx];
        let r_lk = row_of(lk);
        #[cfg(not(target_arch = "wasm32"))]
        let dists_k: Vec<f64> = items
            .par_iter()
            .map(|&x| {
                if x == lk {
                    0.0
                } else {
                    discrete_leaf_dist(r_lk, row_of(x), &subst.dist_weights, subst.l_eff_seg, subst)
                }
            })
            .collect();
        #[cfg(target_arch = "wasm32")]
        let dists_k: Vec<f64> = items
            .iter()
            .map(|&x| {
                if x == lk {
                    0.0
                } else {
                    discrete_leaf_dist(r_lk, row_of(x), &subst.dist_weights, subst.l_eff_seg, subst)
                }
            })
            .collect();

        for (i, &d) in dists_k.iter().enumerate() {
            if d < min_dist[i] {
                min_dist[i] = d;
                best_k[i] = k_idx;
            }
        }
        if k_idx + 1 < k_land {
            let mut cands: Vec<(f64, usize)> = items
                .iter()
                .copied()
                .enumerate()
                .filter(|(_, x)| !landmarks.contains(x))
                .map(|(i, x)| (min_dist[i], x))
                .collect();
            if !cands.is_empty() {
                cands.sort_by(|a, b| {
                    a.0.partial_cmp(&b.0)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.1.cmp(&b.1))
                });
                let q_idx = ((cands.len() * 4) / 5).min(cands.len() - 1);
                landmarks.push(cands[q_idx].1);
            }
        }
    }

    let mut cells: Vec<Vec<(f64, usize)>> = vec![Vec::new(); k_land];
    for (i, &x) in items.iter().enumerate() {
        cells[best_k[i]].push((min_dist[i], x));
    }
    for cell in &mut cells {
        cell.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });
    }

    let mut visited = vec![false; k_land];
    let mut chain = Vec::with_capacity(k_land);
    let mut cur = 0usize;
    visited[0] = true;
    chain.push(0);
    for _ in 1..k_land {
        let r_cur = row_of(landmarks[cur]);
        let mut best_next = 0usize;
        let mut best_d = f64::INFINITY;
        for cand in 0..k_land {
            if !visited[cand] {
                let d = discrete_leaf_dist(
                    r_cur,
                    row_of(landmarks[cand]),
                    &subst.dist_weights,
                    subst.l_eff_seg,
                    subst,
                );
                if d < best_d {
                    best_d = d;
                    best_next = cand;
                }
            }
        }
        visited[best_next] = true;
        chain.push(best_next);
        cur = best_next;
    }

    let mut pack: Vec<usize> = Vec::new();
    for c_idx in chain {
        let cell_items: Vec<usize> = cells[c_idx].iter().map(|t| t.1).collect();
        if cell_items.len() > max_cluster {
            if !pack.is_empty() {
                out.push(std::mem::take(&mut pack));
            }
            partition_discrete_voronoi(cell_items, mat, l, subst, max_cluster, out);
        } else {
            if pack.len() + cell_items.len() > max_cluster && !pack.is_empty() {
                out.push(std::mem::take(&mut pack));
            }
            pack.extend(cell_items);
        }
    }
    if !pack.is_empty() {
        out.push(pack);
    }
}

/// Exact nodal state combination across `child_list`:
/// - `Unordered`: Sankoff-Fitch intersection/union/plurality `u32` bitmasks (`0..k`).
/// - `Ordered`: Farris-Wagner `L1` convex interval-hull `[l_v, r_v]` stored as contiguous `u32` bitmasks.
#[inline]
pub fn combine_discrete_children_refs(
    child_list: &[(&[u32], f64)],
    subst: &DiscreteSubstModel,
) -> (Vec<u32>, f64) {
    if child_list.len() == 1 {
        return (child_list[0].0.to_vec(), child_list[0].1);
    }
    let l = child_list[0].0.len();
    let all_m = subst.all_mask;
    let k = subst.k;

    if child_list.len() == 2 {
        let (s1, m1) = child_list[0];
        let (s2, m2) = child_list[1];
        let mut out = vec![all_m; l];
        let mut n_mut = 0.0_f64;
        match subst.mode {
            DiscreteMode::Unordered => {
                for s in 0..l {
                    let b1 = s1[s];
                    let b2 = s2[s];
                    let inter = b1 & b2;
                    if inter == 0 {
                        out[s] = b1 | b2;
                        if b1 != all_m && b2 != all_m {
                            n_mut += subst.dist_weights[s];
                        }
                    } else {
                        out[s] = inter;
                    }
                }
            }
            DiscreteMode::Ordered => {
                for s in 0..l {
                    let b1 = s1[s];
                    let b2 = s2[s];
                    if b1 == all_m {
                        out[s] = b2;
                    } else if b2 == all_m {
                        out[s] = b1;
                    } else {
                        let inter = b1 & b2;
                        if inter != 0 {
                            out[s] = inter;
                        } else {
                            let lo1 = b1.trailing_zeros() as usize;
                            let hi1 = (31 - b1.leading_zeros()) as usize;
                            let lo2 = b2.trailing_zeros() as usize;
                            let hi2 = (31 - b2.leading_zeros()) as usize;
                            if hi1 < lo2 {
                                out[s] = interval_mask(hi1, lo2);
                                n_mut += subst.dist_weights[s] * ((lo2 - hi1) as f64);
                            } else {
                                out[s] = interval_mask(hi2, lo1);
                                n_mut += subst.dist_weights[s] * ((lo1 - hi2) as f64);
                            }
                        }
                    }
                }
            }
        }
        let par_load = 1.0 / (1.0 / m1.max(0.05) + 1.0 / m2.max(0.05)) + 0.25 * n_mut;
        return (out, par_load.max(0.05));
    }

    let mut out = vec![all_m; l];
    let mut inv_sum = 0.0_f64;
    for &(_, m_c) in child_list {
        inv_sum += 1.0 / m_c.max(0.05);
    }

    match subst.mode {
        DiscreteMode::Unordered => {
            for s in 0..l {
                let mut m_and = all_m;
                for &(s_mask, _) in child_list {
                    m_and &= s_mask[s];
                }
                if m_and != 0 {
                    out[s] = m_and;
                } else {
                    let mut counts = [0_u16; MAX_DISCRETE_STATES];
                    let mut max_c = 0_u16;
                    for &(s_mask, _) in child_list {
                        let mut m = s_mask[s];
                        if m != all_m {
                            while m != 0 {
                                let b = m.trailing_zeros() as usize;
                                if b < k {
                                    counts[b] += 1;
                                    if counts[b] > max_c {
                                        max_c = counts[b];
                                    }
                                }
                                m &= m - 1;
                            }
                        }
                    }
                    if max_c > 0 {
                        let mut mask = 0_u32;
                        for b in 0..k {
                            if counts[b] == max_c {
                                mask |= 1_u32 << (b as u32);
                            }
                        }
                        out[s] = mask;
                    }
                }
            }
        }
        DiscreteMode::Ordered => {
            for s in 0..l {
                let mut m_and = all_m;
                let mut any_valid = false;
                for &(s_mask, _) in child_list {
                    let m = s_mask[s];
                    if m != all_m && m != 0 {
                        any_valid = true;
                        m_and &= m;
                    }
                }
                if !any_valid {
                    out[s] = all_m;
                } else if m_and != 0 {
                    out[s] = m_and;
                } else {
                    let mut costs = [0_u32; MAX_DISCRETE_STATES];
                    for &(s_mask, _) in child_list {
                        let m = s_mask[s];
                        if m != all_m && m != 0 {
                            let lo = m.trailing_zeros() as usize;
                            let hi = (31 - m.leading_zeros()) as usize;
                            for x in 0..k {
                                costs[x] += (lo.saturating_sub(x) + x.saturating_sub(hi)) as u32;
                            }
                        }
                    }
                    let mut min_c = u32::MAX;
                    let mut lo_opt = 0usize;
                    let mut hi_opt = 0usize;
                    for x in 0..k {
                        let c = costs[x];
                        if c < min_c {
                            min_c = c;
                            lo_opt = x;
                            hi_opt = x;
                        } else if c == min_c {
                            hi_opt = x;
                        }
                    }
                    out[s] = interval_mask(lo_opt, hi_opt);
                }
            }
        }
    }

    let par_load = (1.0 / inv_sum.max(1e-6)).max(0.05);
    (out, par_load)
}

#[inline]
pub fn combine_discrete_children(
    child_list: &[(Vec<u32>, f64)],
    subst: &DiscreteSubstModel,
) -> (Vec<u32>, f64) {
    let refs: Vec<(&[u32], f64)> = child_list
        .iter()
        .map(|(s, m)| (s.as_slice(), *m))
        .collect();
    combine_discrete_children_refs(&refs, subst)
}

pub fn compute_all_directed_discrete_states(
    adj: &BTreeMap<usize, Vec<usize>>,
    bitmasks: &[u32],
    n: usize,
    l: usize,
    terminal_loads: &[f64],
    subst: &DiscreteSubstModel,
) -> HashMap<(usize, usize), (Vec<u32>, f64)> {
    let all_m = subst.all_mask;
    let k = subst.k;
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

    let mut states: HashMap<(usize, usize), (Vec<u32>, f64)> =
        HashMap::with_capacity(total_nodes * 2);

    // Level-parallel bottom-up pass
    for d in (1..levels.len()).rev() {
        let level = &levels[d];
        let compute_up = |&u: &usize| -> ((usize, usize), (Vec<u32>, f64)) {
            let p = parent[&u].unwrap();
            if u < n {
                (
                    (u, p),
                    (bitmasks[u * l..(u + 1) * l].to_vec(), terminal_loads[u]),
                )
            } else {
                let child_refs: Vec<(&[u32], f64)> = adj[&u]
                    .iter()
                    .copied()
                    .filter(|&nb| nb != p)
                    .map(|c| {
                        let st = &states[&(c, u)];
                        (st.0.as_slice(), st.1)
                    })
                    .collect();
                ((u, p), combine_discrete_children_refs(&child_refs, subst))
            }
        };
        #[cfg(not(target_arch = "wasm32"))]
        let up_batch: Vec<((usize, usize), (Vec<u32>, f64))> = if level.len() >= 8 {
            level.par_iter().map(compute_up).collect()
        } else {
            level.iter().map(compute_up).collect()
        };
        #[cfg(target_arch = "wasm32")]
        let up_batch: Vec<((usize, usize), (Vec<u32>, f64))> =
            level.iter().map(compute_up).collect();
        states.extend(up_batch);
    }

    // Level-parallel top-down pass with exact O(d * K * M) leave-one-out hub acceleration
    for level in &levels {
        let compute_down = |&u: &usize| -> Vec<((usize, usize), (Vec<u32>, f64))> {
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
                match subst.mode {
                    DiscreteMode::Unordered => {
                        let mut tot_counts = vec![[0_u16; MAX_DISCRETE_STATES]; l];
                        for &nb in all_nbs {
                            let (ref s_mask, m_c) = states[&(nb, u)];
                            tot_inv += 1.0 / m_c.max(0.05);
                            for s in 0..l {
                                let mut m = s_mask[s];
                                if m != all_m {
                                    while m != 0 {
                                        let b = m.trailing_zeros() as usize;
                                        if b < k {
                                            tot_counts[s][b] += 1;
                                        }
                                        m &= m - 1;
                                    }
                                }
                            }
                        }
                        let build_hub_child = |&c: &usize| -> ((usize, usize), (Vec<u32>, f64)) {
                            let (ref s_c, m_c) = states[&(c, u)];
                            let inv_rem = (tot_inv - 1.0 / m_c.max(0.05)).max(1e-6);
                            let par_load = (1.0 / inv_rem).max(0.05);
                            let mut out_mask = vec![all_m; l];
                            for s in 0..l {
                                let mut counts = tot_counts[s];
                                let mut mc = s_c[s];
                                if mc != all_m {
                                    while mc != 0 {
                                        let b = mc.trailing_zeros() as usize;
                                        if b < k {
                                            counts[b] = counts[b].saturating_sub(1);
                                        }
                                        mc &= mc - 1;
                                    }
                                }
                                let mut max_c = 0_u16;
                                for &v in &counts[..k] {
                                    if v > max_c {
                                        max_c = v;
                                    }
                                }
                                if max_c > 0 {
                                    let mut mask = 0_u32;
                                    for b in 0..k {
                                        if counts[b] == max_c {
                                            mask |= 1_u32 << (b as u32);
                                        }
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
                    }
                    DiscreteMode::Ordered => {
                        let mut tot_l1 = vec![[0_u32; MAX_DISCRETE_STATES]; l];
                        let mut tot_valid = vec![0_u16; l];
                        for &nb in all_nbs {
                            let (ref s_mask, m_c) = states[&(nb, u)];
                            tot_inv += 1.0 / m_c.max(0.05);
                            for s in 0..l {
                                let m = s_mask[s];
                                if m != all_m && m != 0 {
                                    tot_valid[s] += 1;
                                    let lo = m.trailing_zeros() as usize;
                                    let hi = (31 - m.leading_zeros()) as usize;
                                    let row = &mut tot_l1[s];
                                    for x in 0..k {
                                        row[x] +=
                                            (lo.saturating_sub(x) + x.saturating_sub(hi)) as u32;
                                    }
                                }
                            }
                        }
                        let build_hub_child = |&c: &usize| -> ((usize, usize), (Vec<u32>, f64)) {
                            let (ref s_c, m_c) = states[&(c, u)];
                            let inv_rem = (tot_inv - 1.0 / m_c.max(0.05)).max(1e-6);
                            let par_load = (1.0 / inv_rem).max(0.05);
                            let mut out_mask = vec![all_m; l];
                            for s in 0..l {
                                let mc = s_c[s];
                                let c_is_val = mc != all_m && mc != 0;
                                let rem_val = tot_valid[s].saturating_sub(u16::from(c_is_val));
                                if rem_val == 0 {
                                    out_mask[s] = all_m;
                                    continue;
                                }
                                let mut row = tot_l1[s];
                                if c_is_val {
                                    let lo = mc.trailing_zeros() as usize;
                                    let hi = (31 - mc.leading_zeros()) as usize;
                                    for x in 0..k {
                                        row[x] = row[x].saturating_sub(
                                            (lo.saturating_sub(x) + x.saturating_sub(hi)) as u32,
                                        );
                                    }
                                }
                                let mut min_c = u32::MAX;
                                let mut lo_opt = 0usize;
                                let mut hi_opt = 0usize;
                                for x in 0..k {
                                    let cv = row[x];
                                    if cv < min_c {
                                        min_c = cv;
                                        lo_opt = x;
                                        hi_opt = x;
                                    } else if cv == min_c {
                                        hi_opt = x;
                                    }
                                }
                                out_mask[s] = interval_mask(lo_opt, hi_opt);
                            }
                            ((u, c), (out_mask, par_load))
                        };
                        #[cfg(not(target_arch = "wasm32"))]
                        if children.len() >= 16 {
                            return children.par_iter().map(build_hub_child).collect();
                        }
                        children.iter().map(build_hub_child).collect()
                    }
                }
            } else {
                children
                    .into_iter()
                    .map(|c| {
                        let other_refs: Vec<(&[u32], f64)> = adj[&u]
                            .iter()
                            .copied()
                            .filter(|&nb| nb != c)
                            .map(|nb| {
                                let st = &states[&(nb, u)];
                                (st.0.as_slice(), st.1)
                            })
                            .collect();
                        if other_refs.is_empty() {
                            ((u, c), (vec![all_m; l], 1.0))
                        } else {
                            ((u, c), combine_discrete_children_refs(&other_refs, subst))
                        }
                    })
                    .collect()
            }
        };

        #[cfg(not(target_arch = "wasm32"))]
        let down_batches: Vec<Vec<((usize, usize), (Vec<u32>, f64))>> = if level.len() >= 4 {
            level.par_iter().map(compute_down).collect()
        } else {
            level.iter().map(compute_down).collect()
        };
        #[cfg(target_arch = "wasm32")]
        let down_batches: Vec<Vec<((usize, usize), (Vec<u32>, f64))>> =
            level.iter().map(compute_down).collect();

        for batch in down_batches {
            states.extend(batch);
        }
    }

    states
}

fn propagate_directed_discrete_outward_exact(
    adj: &BTreeMap<usize, Vec<usize>>,
    states: &mut HashMap<(usize, usize), (Vec<u32>, f64)>,
    mut queue: VecDeque<(usize, usize)>,
    n: usize,
    subst: &DiscreteSubstModel,
    mut dirty_out: Option<&mut HashSet<(usize, usize)>>,
) {
    while let Some((x, y)) = queue.pop_front() {
        if x < n {
            continue;
        }
        let child_refs: Vec<(&[u32], f64)> = adj[&x]
            .iter()
            .copied()
            .filter(|&nb| nb != y)
            .map(|nb| {
                let st = &states[&(nb, x)];
                (st.0.as_slice(), st.1)
            })
            .collect();
        let new_state = combine_discrete_children_refs(&child_refs, subst);
        let changed = match states.get(&(x, y)) {
            Some((old_mask, old_load)) => {
                new_state.0 != *old_mask || (new_state.1 - *old_load).abs() > 1e-11
            }
            None => true,
        };
        states.insert((x, y), new_state);
        if let Some(ref mut dset) = dirty_out {
            dset.insert((x, y));
        }
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
fn compute_discrete_split_mdl_gain_core(
    u_children: &[(&[u32], f64)],
    v_children: &[(&[u32], f64)],
    l: usize,
    l_eff_raw: f64,
    chrom_ids: &[u16],
    subst: &DiscreteSubstModel,
) -> (f64, f64, f64, Vec<usize>) {
    let deg_a = u_children.len();
    let deg_b = v_children.len();
    let topo_cost = split_topology_cost_nats(deg_a, deg_b);
    if deg_a < 2 || deg_b < 2 || l == 0 {
        return (-topo_cost, -topo_cost - 10.0, 0.0, Vec::new());
    }

    let k = subst.k;
    let all_m = subst.all_mask;
    let l_eff_model = if subst.is_collinear_bins {
        subst.l_eff_seg
    } else {
        l_eff_raw
    };

    let mut subst_from_parent = [[0.0_f32; MAX_DISCRETE_STATES]; MAX_DISCRETE_STATES];
    let mut nlp_arr = [0.0_f32; MAX_DISCRETE_STATES];
    for b_par in 0..k {
        let nlp = (-subst.pi[b_par].max(1e-12).ln()) as f32;
        nlp_arr[b_par] = nlp;
        for a_ch in 0..k {
            subst_from_parent[b_par][a_ch] = subst.cost[b_par][a_ch] as f32;
        }
    }

    let all_children: Vec<(&[u32], f64)> = u_children
        .iter()
        .copied()
        .chain(v_children.iter().copied())
        .collect();
    let k_ch = all_children.len();
    let min_ld = if l_eff_model <= 64.0 { 0.05 } else { 0.45 };
    let mut refined_loads = Vec::with_capacity(k_ch);
    for &(_, mc) in &all_children {
        refined_loads.push(mc.max(min_ld));
    }

    let marginal_pos_code = |m_c: f64| -> f32 {
        if l_eff_model > 64.0 {
            (l_eff_model / (m_c + 1.0)).max(1.25).ln() as f32
        } else {
            let l_cap = l_eff_model.max(2.0);
            let pos_at = |x: f64| -> f64 {
                if x <= 1e-9 {
                    0.0
                } else if x < 1.0 {
                    x * (universal_int_nats(1.0) + log_binom(l_cap, 1.0_f64.min(0.49 * l_cap)))
                } else {
                    universal_int_nats(x) + log_binom(l_cap, x.min(0.49 * l_cap))
                }
            };
            (pos_at(m_c + 1.0) - pos_at(m_c)).max(0.25) as f32
        }
    };

    let u_loads = &refined_loads[..deg_a];
    let v_loads = &refined_loads[deg_a..];
    let mut u_weights = Vec::with_capacity(deg_a);
    let mut v_weights = Vec::with_capacity(deg_b);
    for &m_c in u_loads {
        let mc = m_c.clamp(min_ld, 0.85 * l_eff_model.max(2.0));
        u_weights.push(marginal_pos_code(mc));
    }
    for &m_c in v_loads {
        let mc = m_c.clamp(min_ld, 0.85 * l_eff_model.max(2.0));
        v_weights.push(marginal_pos_code(mc));
    }

    // (site_idx, b_u_opt, b_v_opt, raw_sav, subst_s)
    let mut raw_candidates: Vec<(usize, u8, u8, f64, f64)> = Vec::new();

    for s in 0..l {
        if k_ch == 4 {
            let m0 = u_children[0].0[s];
            let m1 = u_children[1].0[s];
            let m2 = v_children[0].0[s];
            let m3 = v_children[1].0[s];
            let u01 = m0 & m1;
            let v23 = m2 & m3;
            if (u01 & v23) != 0 {
                continue;
            }
        } else {
            let mut m_and = all_m;
            for &(s_mask, _) in &all_children {
                m_and &= s_mask[s];
            }
            if m_and != 0 {
                continue;
            }
        }

        let mut u_valid = 0i16;
        let mut v_valid = 0i16;
        let mut u_miss = [0i16; MAX_DISCRETE_STATES];
        let mut v_miss = [0i16; MAX_DISCRETE_STATES];
        let mut cu = [0.0_f32; MAX_DISCRETE_STATES];
        let mut cv = [0.0_f32; MAX_DISCRETE_STATES];

        for (&(s_mask, _), &w_c) in u_children.iter().zip(&u_weights) {
            let m = s_mask[s];
            if m != all_m && m != 0 {
                u_valid += 1;
                if subst.mode == DiscreteMode::Ordered {
                    let lo = m.trailing_zeros() as usize;
                    let hi = (31 - m.leading_zeros()) as usize;
                    for b in 0..lo {
                        u_miss[b] += 1;
                        cu[b] += w_c + subst_from_parent[b][lo];
                    }
                    for b in (hi + 1)..k {
                        u_miss[b] += 1;
                        cu[b] += w_c + subst_from_parent[b][hi];
                    }
                } else if m.is_power_of_two() {
                    let a = m.trailing_zeros() as usize;
                    for b in 0..k {
                        if b != a {
                            u_miss[b] += 1;
                            cu[b] += w_c + subst_from_parent[b][a];
                        }
                    }
                } else {
                    for b in 0..k {
                        if ((m >> b) & 1) == 0 {
                            let mut min_sub = f32::INFINITY;
                            let mut mm = m;
                            while mm != 0 {
                                let a = mm.trailing_zeros() as usize;
                                if a < k {
                                    min_sub = min_sub.min(subst_from_parent[b][a]);
                                }
                                mm &= mm - 1;
                            }
                            u_miss[b] += 1;
                            cu[b] += w_c + min_sub;
                        }
                    }
                }
            }
        }
        if u_valid < 2 {
            continue;
        }

        for (&(s_mask, _), &w_c) in v_children.iter().zip(&v_weights) {
            let m = s_mask[s];
            if m != all_m && m != 0 {
                v_valid += 1;
                if subst.mode == DiscreteMode::Ordered {
                    let lo = m.trailing_zeros() as usize;
                    let hi = (31 - m.leading_zeros()) as usize;
                    for b in 0..lo {
                        v_miss[b] += 1;
                        cv[b] += w_c + subst_from_parent[b][lo];
                    }
                    for b in (hi + 1)..k {
                        v_miss[b] += 1;
                        cv[b] += w_c + subst_from_parent[b][hi];
                    }
                } else if m.is_power_of_two() {
                    let a = m.trailing_zeros() as usize;
                    for b in 0..k {
                        if b != a {
                            v_miss[b] += 1;
                            cv[b] += w_c + subst_from_parent[b][a];
                        }
                    }
                } else {
                    for b in 0..k {
                        if ((m >> b) & 1) == 0 {
                            let mut min_sub = f32::INFINITY;
                            let mut mm = m;
                            while mm != 0 {
                                let a = mm.trailing_zeros() as usize;
                                if a < k {
                                    min_sub = min_sub.min(subst_from_parent[b][a]);
                                }
                                mm &= mm - 1;
                            }
                            v_miss[b] += 1;
                            cv[b] += w_c + min_sub;
                        }
                    }
                }
            }
        }
        if v_valid < 2 {
            continue;
        }

        let mut steps_star = i16::MAX;
        let mut c_star = f32::INFINITY;
        for b in 0..k {
            let st = u_miss[b] + v_miss[b];
            if st < steps_star {
                steps_star = st;
            }
            let cs = nlp_arr[b] + cu[b] + cv[b];
            if cs < c_star {
                c_star = cs;
            }
        }
        if steps_star < 2 {
            continue;
        }

        // Exact symmetric two-part root + directed internal edge cost:
        // One endpoint acts as local root (`-ln pi[b_anc]`) and transmits `b_anc -> b_der` (`cost[b_anc][b_der]`),
        // avoiding double-charging `-ln pi[b_der]` on skewed non-reversible somatic/CNA alphabets.
        let mut bu_best = 0usize;
        let mut bv_best = 0usize;
        let mut best_tot = f64::INFINITY;
        let mut best_sub = 0.0_f64;
        for bu in 0..k {
            let cu_f = cu[bu] as f64;
            let nlp_u = nlp_arr[bu] as f64;
            for bv in 0..k {
                if bu == bv {
                    continue;
                }
                let cv_f = cv[bv] as f64;
                let nlp_v = nlp_arr[bv] as f64;
                let dir_uv = nlp_u + subst.cost[bu][bv];
                let dir_vu = nlp_v + subst.cost[bv][bu];
                let (root_plus_edge, sub_uv) = if dir_uv <= dir_vu {
                    (dir_uv, subst.cost[bu][bv])
                } else {
                    (dir_vu, subst.cost[bv][bu])
                };
                let tot = cu_f + cv_f + root_plus_edge;
                if tot < best_tot {
                    best_tot = tot;
                    bu_best = bu;
                    bv_best = bv;
                    best_sub = sub_uv;
                }
            }
        }
        let (b_u_opt, b_v_opt, best_split_total, subst_s) = (bu_best, bv_best, best_tot, best_sub);

        let sw = subst.tree_weights[s];
        let net_site = ((c_star as f64) - best_split_total) * sw;
        let raw_sav = net_site + subst_s * sw;
        if net_site > 0.0 {
            raw_candidates.push((s, b_u_opt as u8, b_v_opt as u8, raw_sav, subst_s * sw));
        }
    }

    if raw_candidates.is_empty() {
        return (-topo_cost, -topo_cost - 6.0, 0.0, Vec::new());
    }

    // In collinear chromosomal bin matrices, collapse contiguous concordant runs of `(chrom, b_u, b_v)`
    // so a single chromosomal arm CNA spanning 200 adjacent 500-kb bins is priced as 1 segmental event!
    let mut pos_events: Vec<(f64, f64, usize, f64)> = Vec::new();
    if subst.is_collinear_bins {
        let mut idx = 0usize;
        while idx < raw_candidates.len() {
            let (s0, bu0, bv0, mut sum_sav, sub0) = raw_candidates[idx];
            let chr0 = chrom_ids[s0];
            let mut last_s = s0;
            let mut cnt = 1usize;
            let mut j = idx + 1;
            while j < raw_candidates.len() {
                let (sj, buj, bvj, savj, _) = raw_candidates[j];
                if chrom_ids[sj] == chr0 && buj == bu0 && bvj == bv0 && sj <= last_s + 4 {
                    sum_sav += savj;
                    last_s = sj;
                    cnt += 1;
                    j += 1;
                } else {
                    break;
                }
            }
            let mean_sav = sum_sav / (cnt as f64);
            let net_ev = mean_sav - sub0;
            if net_ev > 0.0 {
                pos_events.push((net_ev, mean_sav, s0, sub0));
            }
            idx = j;
        }
    } else {
        for (s, _, _, raw_sav, subst_s) in raw_candidates {
            pos_events.push((raw_sav - subst_s, raw_sav, s, subst_s));
        }
    }

    if pos_events.is_empty() {
        return (-topo_cost, -topo_cost - 6.0, 0.0, Vec::new());
    }

    pos_events.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.2.cmp(&b.2))
    });

    let l_vis = l_eff_model.max((pos_events.len() as f64) * 2.05).max(2.0);
    let mut cum_sav = 0.0_f64;
    let mut cum_subst = 0.0_f64;
    let mut best_data_gain = f64::NEG_INFINITY;
    let mut best_idx = 0usize;
    let mut first_thresh = 1.0_f64;

    for (idx, &(_, sav, _, subst_s)) in pos_events.iter().enumerate() {
        let m = (idx + 1) as f64;
        cum_sav += sav;
        cum_subst += subst_s;
        let header = universal_int_nats(m);
        let pos_cost = log_binom(l_vis, m.min(0.49 * l_vis));
        let branch_cost = header + pos_cost + cum_subst;
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
    let chosen_sites: Vec<usize> = pos_events[..=best_idx].iter().map(|x| x.2).collect();
    let raw_gain = best_data_gain - topo_cost;
    let final_gain = if raw_gain > 0.0 {
        raw_gain
    } else {
        let frac = (pos_events[0].1 / first_thresh).clamp(0.0, 0.98);
        -topo_cost * (1.0 - 0.95 * frac)
    };
    (final_gain, raw_gain, m_opt, chosen_sites)
}

#[allow(clippy::too_many_arguments)]
pub fn compute_discrete_split_mdl_gain_refs(
    u_children: &[(&[u32], f64)],
    v_children: &[(&[u32], f64)],
    l: usize,
    l_eff: f64,
    chrom_ids: &[u16],
    subst: &DiscreteSubstModel,
) -> (f64, f64, Vec<usize>) {
    let (final_gain, _, m_opt, chosen_sites) =
        compute_discrete_split_mdl_gain_core(u_children, v_children, l, l_eff, chrom_ids, subst);
    (final_gain, m_opt, chosen_sites)
}

#[allow(clippy::too_many_arguments)]
fn evaluate_discrete_internal_edge(
    u: usize,
    v: usize,
    adj: &BTreeMap<usize, Vec<usize>>,
    states: &HashMap<(usize, usize), (Vec<u32>, f64)>,
    l: usize,
    l_eff: f64,
    chrom_ids: &[u16],
    subst: &DiscreteSubstModel,
) -> (f64, f64, Vec<usize>) {
    let mut u_sub: Vec<usize> = adj[&u].iter().copied().filter(|&nb| nb != v).collect();
    let mut v_sub: Vec<usize> = adj[&v].iter().copied().filter(|&nb| nb != u).collect();
    let deg_u = u_sub.len();
    let deg_v = v_sub.len();
    if deg_u < 2 || deg_v < 2 {
        return (-10.0, 0.0, Vec::new());
    }
    if u_sub.len() > 64 {
        u_sub.sort_by(|&a, &b| {
            states[&(a, u)]
                .1
                .partial_cmp(&states[&(b, u)].1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        u_sub.truncate(48);
    }
    if v_sub.len() > 64 {
        v_sub.sort_by(|&a, &b| {
            states[&(a, v)]
                .1
                .partial_cmp(&states[&(b, v)].1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        v_sub.truncate(48);
    }
    let u_refs: Vec<(&[u32], f64)> = u_sub
        .iter()
        .map(|&nb| {
            let st = &states[&(nb, u)];
            (st.0.as_slice(), st.1)
        })
        .collect();
    let v_refs: Vec<(&[u32], f64)> = v_sub
        .iter()
        .map(|&nb| {
            let st = &states[&(nb, v)];
            (st.0.as_slice(), st.1)
        })
        .collect();
    let (gain_sub, m_opt, sites) =
        compute_discrete_split_mdl_gain_refs(&u_refs, &v_refs, l, l_eff, chrom_ids, subst);
    let extra_topo = split_topology_cost_nats(deg_u, deg_v)
        - split_topology_cost_nats(u_refs.len(), v_refs.len());
    (gain_sub - extra_topo, m_opt, sites)
}

#[allow(clippy::too_many_arguments)]
fn refine_discrete_nni_mdl(
    adj: &mut BTreeMap<usize, Vec<usize>>,
    bitmasks: &[u32],
    terminal_loads: &[f64],
    n: usize,
    l: usize,
    l_eff: f64,
    chrom_ids: &[u16],
    subst: &DiscreteSubstModel,
    max_rounds: usize,
) {
    let mut states =
        compute_all_directed_discrete_states(adj, bitmasks, n, l, terminal_loads, subst);
    for _ in 0..max_rounds {
        let mut improved = false;
        let mut internal_edges = Vec::new();
        for (&u, nbs) in adj.iter() {
            if u >= n && nbs.len() == 3 {
                for &v in nbs {
                    if u < v && v >= n && adj.get(&v).map(|x| x.len()).unwrap_or(0) == 3 {
                        internal_edges.push((u, v));
                    }
                }
            }
        }
        if internal_edges.is_empty() {
            break;
        }

        let eval_nni_edge = |u: usize,
                             v: usize,
                             cur_adj: &BTreeMap<usize, Vec<usize>>,
                             cur_states: &HashMap<(usize, usize), (Vec<u32>, f64)>|
         -> (usize, usize, usize, usize, f64, f64, f64) {
            let mut u_it = cur_adj[&u].iter().copied().filter(|&x| x != v);
            let mut v_it = cur_adj[&v].iter().copied().filter(|&x| x != u);
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
            let (_, g0, _, _) =
                compute_discrete_split_mdl_gain_core(&[ra, rb], &[rc, rd], l, l_eff, chrom_ids, subst);
            let (_, g1, _, _) =
                compute_discrete_split_mdl_gain_core(&[ra, rc], &[rb, rd], l, l_eff, chrom_ids, subst);
            let (_, g2, _, _) =
                compute_discrete_split_mdl_gain_core(&[ra, rd], &[rb, rc], l, l_eff, chrom_ids, subst);
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
            if adj.get(&u).map(|x| x.len()).unwrap_or(0) != 3
                || adj.get(&v).map(|x| x.len()).unwrap_or(0) != 3
                || !adj[&u].contains(&v)
            {
                continue;
            }
            let u_nbs: Vec<usize> = adj[&u].iter().copied().filter(|&x| x != v).collect();
            let v_nbs: Vec<usize> = adj[&v].iter().copied().filter(|&x| x != u).collect();
            if u_nbs.len() != 2 || v_nbs.len() != 2 {
                continue;
            }
            let (a, b) = (u_nbs[0], u_nbs[1]);
            let (c, d) = (v_nbs[0], v_nbs[1]);

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

            let (best_alt, swap_pair) = if g1 >= g2 {
                (g1, (b, c))
            } else {
                (g2, (b, d))
            };
            if best_alt > g0 + 0.15 {
                let (u_swap, v_swap) = swap_pair;
                let keep_v = if v_swap == c { d } else { c };
                let t_uswap = states.remove(&(u_swap, u)).unwrap();
                let t_vswap = states.remove(&(v_swap, v)).unwrap();
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

                states.insert((u_swap, v), t_uswap);
                states.insert((v_swap, u), t_vswap);
                dirty_directed.insert((u_swap, v));
                dirty_directed.insert((v_swap, u));

                let st_uv = {
                    let sa = &states[&(a, u)];
                    let ssw = &states[&(v_swap, u)];
                    combine_discrete_children_refs(
                        &[(sa.0.as_slice(), sa.1), (ssw.0.as_slice(), ssw.1)],
                        subst,
                    )
                };
                let st_vu = {
                    let skv = &states[&(keep_v, v)];
                    let sus = &states[&(u_swap, v)];
                    combine_discrete_children_refs(
                        &[(skv.0.as_slice(), skv.1), (sus.0.as_slice(), sus.1)],
                        subst,
                    )
                };
                states.insert((u, v), st_uv);
                states.insert((v, u), st_vu);
                dirty_directed.insert((u, v));
                dirty_directed.insert((v, u));

                let q = VecDeque::from([(u, a), (u, v_swap), (v, keep_v), (v, u_swap)]);
                propagate_directed_discrete_outward_exact(
                    adj,
                    &mut states,
                    q,
                    n,
                    subst,
                    Some(&mut dirty_directed),
                );
                improved = true;
            }
        }
        if !improved {
            break;
        }
    }
}

/// Parallel 2-hop Subtree Prune and Regraft (`SPR`) swaps across 5-subtree internal caterpillars
/// `v0 - v1 - v2` under the exact two-part Discrete / Ordered Wagner MDL objective.
#[allow(clippy::too_many_arguments)]
fn refine_discrete_spr_mdl(
    adj: &mut BTreeMap<usize, Vec<usize>>,
    bitmasks: &[u32],
    terminal_loads: &[f64],
    n: usize,
    l: usize,
    l_eff: f64,
    chrom_ids: &[u16],
    subst: &DiscreteSubstModel,
    max_rounds: usize,
) {
    if n < 5 || max_rounds == 0 {
        return;
    }
    const SPR_PERMS: [[usize; 5]; 8] = [
        [0, 2, 4, 1, 3],
        [0, 2, 3, 1, 4],
        [1, 2, 4, 0, 3],
        [1, 2, 3, 0, 4],
        [0, 3, 1, 2, 4],
        [1, 3, 0, 2, 4],
        [0, 4, 1, 2, 3],
        [1, 4, 0, 2, 3],
    ];

    for _ in 0..max_rounds {
        let states =
            compute_all_directed_discrete_states(adj, bitmasks, n, l, terminal_loads, subst);

        let mut triples: Vec<(usize, usize, usize)> = Vec::new();
        for (&v1, nbs1) in adj.iter() {
            if v1 >= n && nbs1.len() == 3 {
                let int_nbs: Vec<usize> = nbs1
                    .iter()
                    .copied()
                    .filter(|&x| x >= n && adj.get(&x).map(|v| v.len()).unwrap_or(0) == 3)
                    .collect();
                for i in 0..int_nbs.len() {
                    for j in (i + 1)..int_nbs.len() {
                        let (u, w) = if int_nbs[i] < int_nbs[j] {
                            (int_nbs[i], int_nbs[j])
                        } else {
                            (int_nbs[j], int_nbs[i])
                        };
                        triples.push((u, v1, w));
                    }
                }
            }
        }
        if triples.is_empty() {
            break;
        }

        let eval_triple = |&(v0, v1, v2): &(usize, usize, usize)|
         -> Option<(f64, usize, usize, usize, [usize; 5], [usize; 5])> {
            let mut it0 = adj[&v0].iter().copied().filter(|&x| x != v1);
            let (a, b) = (it0.next()?, it0.next()?);
            let m = adj[&v1]
                .iter()
                .copied()
                .find(|&x| x != v0 && x != v2)?;
            let mut it2 = adj[&v2].iter().copied().filter(|&x| x != v1);
            let (c, d) = (it2.next()?, it2.next()?);

            let orig_nodes = [a, b, m, c, d];
            let sub_refs: [(&[u32], f64); 5] = [
                {
                    let st = &states[&(a, v0)];
                    (st.0.as_slice(), st.1)
                },
                {
                    let st = &states[&(b, v0)];
                    (st.0.as_slice(), st.1)
                },
                {
                    let st = &states[&(m, v1)];
                    (st.0.as_slice(), st.1)
                },
                {
                    let st = &states[&(c, v2)];
                    (st.0.as_slice(), st.1)
                },
                {
                    let st = &states[&(d, v2)];
                    (st.0.as_slice(), st.1)
                },
            ];

            let score_caterpillar = |p: &[usize; 5], check_cap: bool| -> Option<f64> {
                let (tl0, tl1, tm, tr0, tr1) = (
                    sub_refs[p[0]],
                    sub_refs[p[1]],
                    sub_refs[p[2]],
                    sub_refs[p[3]],
                    sub_refs[p[4]],
                );
                let comb_l = combine_discrete_children_refs(&[tl0, tl1], subst);
                let comb_r = combine_discrete_children_refs(&[tr0, tr1], subst);
                let (_, g_left, _, _) = compute_discrete_split_mdl_gain_core(
                    &[tl0, tl1],
                    &[tm, (comb_r.0.as_slice(), comb_r.1)],
                    l,
                    l_eff,
                    chrom_ids,
                    subst,
                );
                let (_, g_right, _, _) = compute_discrete_split_mdl_gain_core(
                    &[(comb_l.0.as_slice(), comb_l.1), tm],
                    &[tr0, tr1],
                    l,
                    l_eff,
                    chrom_ids,
                    subst,
                );
                if check_cap && g_left.min(g_right) > 35.0 {
                    return None;
                }
                Some(g_left + g_right)
            };

            let curr_score = score_caterpillar(&[0, 1, 2, 3, 4], true)?;
            let mut best_score = curr_score;
            let mut best_perm: Option<[usize; 5]> = None;
            for perm in &SPR_PERMS {
                if let Some(sc) = score_caterpillar(perm, false) {
                    if sc > best_score + 0.30 {
                        best_score = sc;
                        best_perm = Some(*perm);
                    }
                }
            }
            let p = best_perm?;
            let new_nodes = [
                orig_nodes[p[0]],
                orig_nodes[p[1]],
                orig_nodes[p[2]],
                orig_nodes[p[3]],
                orig_nodes[p[4]],
            ];
            Some((best_score - curr_score, v0, v1, v2, orig_nodes, new_nodes))
        };

        #[cfg(not(target_arch = "wasm32"))]
        let mut moves: Vec<(f64, usize, usize, usize, [usize; 5], [usize; 5])> =
            triples.par_iter().filter_map(eval_triple).collect();
        #[cfg(target_arch = "wasm32")]
        let mut moves: Vec<(f64, usize, usize, usize, [usize; 5], [usize; 5])> =
            triples.iter().filter_map(eval_triple).collect();

        if moves.is_empty() {
            break;
        }
        moves.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
                .then_with(|| a.2.cmp(&b.2))
                .then_with(|| a.3.cmp(&b.3))
        });

        let mut locked: HashSet<usize> = HashSet::new();
        let mut committed = 0usize;
        for (_, v0, v1, v2, orig_o, new_o) in moves {
            if locked.contains(&v0)
                || locked.contains(&v1)
                || locked.contains(&v2)
                || orig_o.iter().any(|x| locked.contains(x))
            {
                continue;
            }
            let [a, b, m, c, d] = orig_o;
            let [nl0, nl1, nmid, nr0, nr1] = new_o;

            adj.get_mut(&v0).unwrap().retain(|&x| x == v1);
            adj.get_mut(&v1).unwrap().retain(|&x| x == v0 || x == v2);
            adj.get_mut(&v2).unwrap().retain(|&x| x == v1);

            adj.get_mut(&a).unwrap().retain(|&x| x != v0);
            adj.get_mut(&b).unwrap().retain(|&x| x != v0);
            adj.get_mut(&m).unwrap().retain(|&x| x != v1);
            adj.get_mut(&c).unwrap().retain(|&x| x != v2);
            adj.get_mut(&d).unwrap().retain(|&x| x != v2);

            adj.get_mut(&v0).unwrap().extend([nl0, nl1]);
            adj.get_mut(&nl0).unwrap().push(v0);
            adj.get_mut(&nl1).unwrap().push(v0);

            adj.get_mut(&v1).unwrap().push(nmid);
            adj.get_mut(&nmid).unwrap().push(v1);

            adj.get_mut(&v2).unwrap().extend([nr0, nr1]);
            adj.get_mut(&nr0).unwrap().push(v2);
            adj.get_mut(&nr1).unwrap().push(v2);

            locked.extend([v0, v1, v2, a, b, m, c, d]);
            committed += 1;
        }
        if committed == 0 {
            break;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn expand_discrete_polytomy_hubs_mdl(
    adj: &mut BTreeMap<usize, Vec<usize>>,
    node_labels: &mut BTreeMap<usize, String>,
    bitmasks: &[u32],
    terminal_loads: &[f64],
    n: usize,
    l: usize,
    l_eff: f64,
    chrom_ids: &[u16],
    subst: &DiscreteSubstModel,
) {
    let all_m = subst.all_mask;
    let k = subst.k;
    let mut next_id = adj.keys().copied().max().unwrap_or(n) + 1;
    let mut states =
        compute_all_directed_discrete_states(adj, bitmasks, n, l, terminal_loads, subst);
    let hubs: Vec<usize> = adj
        .iter()
        .filter(|(&h, nbs)| h >= n && (4..=256).contains(&nbs.len()))
        .map(|(&h, _)| h)
        .collect();

    for h in hubs {
        let Some(nbs_ref) = adj.get(&h) else { continue };
        if nbs_ref.len() < 4 {
            continue;
        }
        let nbs = nbs_ref.clone();
        let d = nbs.len();

        let chosen_splits: Vec<BTreeSet<usize>> = {
            let nb_refs: Vec<(&[u32], f64)> = nbs
                .iter()
                .map(|&c| {
                    let st = &states[&(c, h)];
                    (st.0.as_slice(), st.1)
                })
                .collect();

            let mut seed_weights: HashMap<Vec<usize>, f64> = HashMap::new();
            let site_stride = if subst.is_collinear_bins {
                (subst.avg_block_len as usize / 2).max(1)
            } else {
                1
            };
            for s in (0..l).step_by(site_stride) {
                let mut n_val = 0usize;
                for &(s_mask, _) in &nb_refs {
                    if s_mask[s] != all_m {
                        n_val += 1;
                    }
                }
                if n_val < 4 {
                    continue;
                }
                for b in 0..k {
                    let mut idx_vec: Vec<usize> = if subst.mode == DiscreteMode::Ordered {
                        // Threshold cut `state >= b` respects ordered copy-number / zygosity steps!
                        (0..d)
                            .filter(|&i| {
                                let m = nb_refs[i].0[s];
                                m != all_m && m != 0 && (m.trailing_zeros() as usize) >= b
                            })
                            .collect()
                    } else {
                        (0..d)
                            .filter(|&i| {
                                let m = nb_refs[i].0[s];
                                m != all_m && ((m >> b) & 1) != 0
                            })
                            .collect()
                    };
                    let cnt = idx_vec.len();
                    if cnt >= 2 && cnt <= d - 2 {
                        if cnt > d / 2 || (cnt == d / 2 && !idx_vec.contains(&0)) {
                            idx_vec = (0..d).filter(|i| !idx_vec.contains(i)).collect();
                        }
                        if idx_vec.len() >= 2 && idx_vec.len() <= d - 2 {
                            *seed_weights.entry(idx_vec).or_insert(0.0) += subst.dist_weights[s];
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
                top_seeds.truncate(24);

                let mut positive_attractors: Vec<(f64, BTreeSet<usize>)> = Vec::new();
                let mut seen_sets: HashSet<BTreeSet<usize>> = HashSet::new();

                for (idx_vec, _) in top_seeds {
                    let mut u_set: BTreeSet<usize> = idx_vec.into_iter().collect();
                    let u_ch: Vec<(&[u32], f64)> = u_set.iter().map(|&i| nb_refs[i]).collect();
                    let v_ch: Vec<(&[u32], f64)> = (0..d)
                        .filter(|j| !u_set.contains(j))
                        .map(|j| nb_refs[j])
                        .collect();
                    let (mut g_cur, _, sites_cur) = compute_discrete_split_mdl_gain_refs(
                        &u_ch,
                        &v_ch,
                        l,
                        l_eff,
                        chrom_ids,
                        subst,
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
                            let u_try: Vec<(&[u32], f64)> =
                                cand_set.iter().map(|&i| nb_refs[i]).collect();
                            let v_try: Vec<(&[u32], f64)> = (0..d)
                                .filter(|j| !cand_set.contains(j))
                                .map(|j| nb_refs[j])
                                .collect();
                            let (g_try, _, _) = compute_discrete_split_mdl_gain_refs(
                                &u_try,
                                &v_try,
                                l,
                                l_eff,
                                chrom_ids,
                                subst,
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
        for u_set in chosen_splits {
            let new_node = next_id;
            next_id += 1;
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
            let ch_refs: Vec<(&[u32], f64)> = adj[&new_node]
                .iter()
                .copied()
                .filter(|&x| x != h)
                .map(|c_node| {
                    let st = &states[&(c_node, new_node)];
                    (st.0.as_slice(), st.1)
                })
                .collect();
            states.insert(
                (new_node, h),
                combine_discrete_children_refs(&ch_refs, subst),
            );
        }
        let mut prop_q: VecDeque<(usize, usize)> = VecDeque::new();
        for &nb in &adj[&h] {
            prop_q.push_back((h, nb));
        }
        propagate_directed_discrete_outward_exact(adj, &mut states, prop_q, n, subst, None);
    }
}

pub fn reconstruct_discrete_nodal_states_and_mutations(
    adj: &BTreeMap<usize, Vec<usize>>,
    bitmasks: &[u32],
    majority_state: &[i8],
    states: &HashMap<(usize, usize), (Vec<u32>, f64)>,
    n: usize,
    l: usize,
    subst: &DiscreteSubstModel,
) -> (
    HashMap<usize, Vec<i8>>,
    HashMap<usize, Vec<bool>>,
    HashMap<(usize, usize), Vec<(usize, i8, i8)>>,
) {
    let all_m = subst.all_mask;
    let k_max_i8 = (subst.k.saturating_sub(1)) as i8;
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
    let mut node_valid: HashMap<usize, Vec<bool>> = HashMap::with_capacity(total_nodes);
    let mut node_mask_dict: HashMap<usize, Vec<u32>> = HashMap::with_capacity(total_nodes);
    let mut edge_mutations: HashMap<(usize, usize), Vec<(usize, i8, i8)>> =
        HashMap::with_capacity(total_nodes);
    let maj_bit: Vec<u32> = majority_state
        .iter()
        .map(|&b| 1_u32 << (b.max(0) as u32))
        .collect();

    for level in &levels {
        let compute_node = |&u: &usize| -> (
            usize,
            Vec<u32>,
            Vec<i8>,
            Vec<bool>,
            Option<((usize, usize), Vec<(usize, i8, i8)>)>,
        ) {
            let p_opt = parent[&u];
            let c_mask = if u < n {
                bitmasks[u * l..(u + 1) * l].to_vec()
            } else if let Some(p) = p_opt {
                states[&(u, p)].0.clone()
            } else {
                let child_refs: Vec<(&[u32], f64)> = adj[&u]
                    .iter()
                    .copied()
                    .map(|c| {
                        let st = &states[&(c, u)];
                        (st.0.as_slice(), st.1)
                    })
                    .collect();
                combine_discrete_children_refs(&child_refs, subst).0
            };
            let val_vec: Vec<bool> = c_mask.iter().map(|&m| m != all_m && m != 0).collect();

            if let Some(p) = p_opt {
                let p_st = &node_state[&p];
                let p_bit: Vec<u32> = p_st.iter().map(|&b| 1_u32 << (b.max(0) as u32)).collect();
                let mpr_mask = if u < n {
                    c_mask.clone()
                } else {
                    let mut star_refs: Vec<(&[u32], f64)> = adj[&u]
                        .iter()
                        .copied()
                        .filter(|&nb| nb != p)
                        .map(|c| {
                            let st = &states[&(c, u)];
                            (st.0.as_slice(), st.1)
                        })
                        .collect();
                    star_refs.push((p_bit.as_slice(), 1.0));
                    combine_discrete_children_refs(&star_refs, subst).0
                };
                let mut st = vec![0_i8; l];
                for s in 0..l {
                    let m = mpr_mask[s];
                    if m == 0 || m == all_m {
                        st[s] = p_st[s];
                    } else if subst.mode == DiscreteMode::Ordered {
                        // Exact Farris-Wagner L1 top-down projection onto [lo, hi]
                        let lo = (m.trailing_zeros() as i8).clamp(0, k_max_i8);
                        let hi = ((31 - m.leading_zeros()) as i8).clamp(lo, k_max_i8);
                        st[s] = p_st[s].clamp(lo, hi);
                    } else if (m & p_bit[s]) != 0 {
                        st[s] = p_st[s];
                    } else if (m & maj_bit[s]) != 0 {
                        st[s] = majority_state[s];
                    } else {
                        st[s] = (m.trailing_zeros() as i8).clamp(0, k_max_i8);
                    }
                }

                let mask_p = &node_mask_dict[&p];
                let mut muts = Vec::new();
                for s in 0..l {
                    if mask_p[s] != all_m && c_mask[s] != all_m && p_st[s] != st[s] {
                        if p < u {
                            muts.push((s, p_st[s], st[s]));
                        } else {
                            muts.push((s, st[s], p_st[s]));
                        }
                    }
                }
                (u, c_mask, st, val_vec, Some(((p.min(u), p.max(u)), muts)))
            } else {
                let mut st = vec![0_i8; l];
                for s in 0..l {
                    let m = c_mask[s];
                    if m == 0 || m == all_m {
                        st[s] = majority_state[s];
                    } else if subst.mode == DiscreteMode::Ordered {
                        let lo = (m.trailing_zeros() as i8).clamp(0, k_max_i8);
                        let hi = ((31 - m.leading_zeros()) as i8).clamp(lo, k_max_i8);
                        st[s] = majority_state[s].clamp(lo, hi);
                    } else if (m & maj_bit[s]) != 0 {
                        st[s] = majority_state[s];
                    } else {
                        st[s] = (m.trailing_zeros() as i8).clamp(0, k_max_i8);
                    }
                }
                (u, c_mask, st, val_vec, None)
            }
        };

        #[cfg(not(target_arch = "wasm32"))]
        let results = if level.len() >= 8 {
            level.par_iter().map(compute_node).collect::<Vec<_>>()
        } else {
            level.iter().map(compute_node).collect::<Vec<_>>()
        };
        #[cfg(target_arch = "wasm32")]
        let results = level.iter().map(compute_node).collect::<Vec<_>>();

        for (u, cm, st, vv, em_opt) in results {
            node_mask_dict.insert(u, cm);
            node_state.insert(u, st);
            node_valid.insert(u, vv);
            if let Some((ekey, muts)) = em_opt {
                edge_mutations.insert(ekey, muts);
            }
        }
    }

    (node_state, node_valid, edge_mutations)
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn contract_discrete_edges_mdl(
    mut adj: BTreeMap<usize, Vec<usize>>,
    bitmasks: &[u32],
    majority_state: &[i8],
    terminal_loads: &[f64],
    n: usize,
    l: usize,
    l_eff: f64,
    chrom_ids: &[u16],
    subst: &DiscreteSubstModel,
    node_labels: &mut BTreeMap<usize, String>,
    expand_hubs: bool,
) -> (
    BTreeMap<usize, Vec<usize>>,
    Vec<BranchRecord>,
    HashMap<usize, Vec<i8>>,
    HashMap<usize, Vec<bool>>,
    HashMap<(usize, usize), Vec<(usize, i8, i8)>>,
) {
    let passes = if expand_hubs { 2 } else { 1 };
    for outer_pass in 0..passes {
        loop {
            let states = compute_all_directed_discrete_states(
                &adj,
                bitmasks,
                n,
                l,
                terminal_loads,
                subst,
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
                    let (gain, _, _) = evaluate_discrete_internal_edge(
                        u,
                        v,
                        &adj,
                        &states,
                        l,
                        l_eff,
                        chrom_ids,
                        subst,
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
                    let (gain, _, _) = evaluate_discrete_internal_edge(
                        u,
                        v,
                        &adj,
                        &states,
                        l,
                        l_eff,
                        chrom_ids,
                        subst,
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
                if u == v || locked_nodes.contains(&u) || locked_nodes.contains(&v) {
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
        adj.retain(|&u, nbs| u < n || !nbs.is_empty());

        if expand_hubs && outer_pass == 0 && n >= 5 {
            expand_discrete_polytomy_hubs_mdl(
                &mut adj,
                node_labels,
                bitmasks,
                terminal_loads,
                n,
                l,
                l_eff,
                chrom_ids,
                subst,
            );
        }
    }

    let states =
        compute_all_directed_discrete_states(&adj, bitmasks, n, l, terminal_loads, subst);
    let (node_states, node_valid, edge_mutations) = reconstruct_discrete_nodal_states_and_mutations(
        &adj,
        bitmasks,
        majority_state,
        &states,
        n,
        l,
        subst,
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

    let l_denom = if subst.is_collinear_bins {
        subst.l_eff_seg
    } else {
        l_eff
    };

    let build_branch_rec = |&(u, v): &(usize, usize)| -> BranchRecord {
        let is_int = u >= n && v >= n;
        let (_b_nats, m_pt, r_runs) = ordered_or_discrete_branch_code_nats(
            &node_states[&u],
            &node_states[&v],
            &node_valid[&u],
            &node_valid[&v],
            chrom_ids,
            l_eff,
            subst,
        );
        let m_eff_branch = if subst.is_collinear_bins {
            r_runs as f64
        } else {
            m_pt as f64
        };
        let (gain, eff_mut, syn_sites) = if is_int {
            let (g, m_syn, sites) = evaluate_discrete_internal_edge(
                u,
                v,
                &adj,
                &states,
                l,
                l_eff,
                chrom_ids,
                subst,
            );
            (g, m_eff_branch.max(m_syn), sites)
        } else {
            (m_eff_branch * 2.5, m_eff_branch, Vec::new())
        };
        let support = 1.0 / (1.0 + (-gain.clamp(-50.0, 50.0)).exp());
        BranchRecord {
            u,
            v,
            is_internal: is_int,
            mutation_count: eff_mut,
            branch_length: eff_mut / l_denom.max(1.0),
            mdl_gain_nats: gain,
            mdl_support: support,
            synapomorphy_sites: syn_sites,
        }
    };

    #[cfg(not(target_arch = "wasm32"))]
    let branches: Vec<BranchRecord> = edge_pairs.par_iter().map(build_branch_rec).collect();
    #[cfg(target_arch = "wasm32")]
    let branches: Vec<BranchRecord> = edge_pairs.iter().map(build_branch_rec).collect();

    (adj, branches, node_states, node_valid, edge_mutations)
}

#[inline]
fn log_falling_factorial(m: f64, k: usize) -> f64 {
    if k == 0 {
        return 0.0;
    }
    let m_eff = m.max(k as f64);
    let mut acc = 0.0_f64;
    for j in 0..k {
        acc += (m_eff - (j as f64)).max(1.0).ln();
    }
    acc
}

#[allow(clippy::too_many_arguments)]
fn compute_discrete_parallel_site_savings(
    u: usize,
    v: usize,
    shared_sites: &[usize],
    site_mut_counts: &[i32],
    site_target_counts: &[[u16; MAX_DISCRETE_STATES]],
    total_tree_muts: f64,
    node_state: &HashMap<usize, Vec<i8>>,
    eff_anc_node: &HashMap<usize, usize>,
    branch_mut_load: &HashMap<usize, f64>,
    l_eff_model: f64,
    subst: &DiscreteSubstModel,
) -> Vec<(usize, f64)> {
    let (Some(&p_u), Some(&p_v)) = (eff_anc_node.get(&u), eff_anc_node.get(&v)) else {
        return Vec::new();
    };
    let k = subst.k;
    let st_u = &node_state[&u];
    let st_pu = &node_state[&p_u];
    let st_v = &node_state[&v];
    let st_pv = &node_state[&p_v];
    let m_u = branch_mut_load.get(&u).copied().unwrap_or(1.0);
    let m_v = branch_mut_load.get(&v).copied().unwrap_or(1.0);
    let m_bg_total = (total_tree_muts - m_u - m_v).max(0.0);
    let pos_denom = (m_bg_total + l_eff_model).max(1.0);

    let mut results = Vec::new();
    for &s in shared_sites {
        if st_u[s] != st_pu[s] && st_v[s] != st_pv[s] && st_u[s] == st_v[s] {
            let w_s = subst.dist_weights[s];
            let bg_muts = ((site_mut_counts[s] - 2).max(0) as f64) * w_s;
            let b_der = (st_u[s].max(0) as usize).min(k - 1);
            let a_recip = if m_v <= m_u {
                (st_pv[s].max(0) as usize).min(k - 1)
            } else {
                (st_pu[s].max(0) as usize).min(k - 1)
            };
            let bg_to_target =
                (((site_target_counts[s][b_der] as i32) - 2).max(0) as f64) * w_s;

            // Kraft-exact site-rate occupancy probability: sum_s q_pos(s) == 1
            let q_pos = (bg_muts + 1.0) / pos_denom;
            let pos_info = (-q_pos.ln()).max(0.0);

            // Kraft-exact conditional target state probability over K - 1 off-diagonal states
            let state_info = if k <= 2 {
                0.0
            } else {
                let alpha_st = (k - 1) as f64;
                let p0 = (-subst.cost[a_recip][b_der]).exp();
                let q_st = (bg_to_target + alpha_st * p0) / (bg_muts + alpha_st);
                (-q_st.ln()).max(0.0)
            };

            let raw_site_info = (pos_info + state_info) * w_s;
            if raw_site_info > 0.0 {
                results.push((s, raw_site_info));
            }
        }
    }
    results
}

#[allow(clippy::too_many_arguments)]
fn evaluate_discrete_interval_shunt_mdl(
    min_shunt_sites: usize,
    u: usize,
    v: usize,
    shared_sites: &[usize],
    site_mut_counts: &[i32],
    site_target_counts: &[[u16; MAX_DISCRETE_STATES]],
    total_tree_muts: f64,
    node_state: &HashMap<usize, Vec<i8>>,
    node_valid: &HashMap<usize, Vec<bool>>,
    eff_anc_node: &HashMap<usize, usize>,
    parent: &HashMap<usize, Option<usize>>,
    adj: &BTreeMap<usize, Vec<usize>>,
    branch_mut_load: &HashMap<usize, f64>,
    l: usize,
    l_eff_model: f64,
    chrom_ids: &[u16],
    endpoint_cost: f64,
    subst: &DiscreteSubstModel,
) -> (f64, Vec<usize>, Option<(usize, usize)>, f64, f64) {
    let site_sav_pairs = compute_discrete_parallel_site_savings(
        u,
        v,
        shared_sites,
        site_mut_counts,
        site_target_counts,
        total_tree_muts,
        node_state,
        eff_anc_node,
        branch_mut_load,
        l_eff_model,
        subst,
    );
    let k_cand = site_sav_pairs.len();
    let min_k = min_shunt_sites.max(3);
    if k_cand < min_k || l < 6 {
        return (-1e9, Vec::new(), None, 0.0, 0.0);
    }

    let m_u = branch_mut_load.get(&u).copied().unwrap_or(1.0);
    let m_v = branch_mut_load.get(&v).copied().unwrap_or(1.0);
    let m_donor_base = m_u.max(m_v);
    let m_recip_base = m_u.min(m_v);

    let cand_sites: Vec<usize> = site_sav_pairs.iter().map(|x| x.0).collect();
    let cand_savs: Vec<f64> = site_sav_pairs.iter().map(|x| x.1).collect();
    let base_mask_cost = 2.0_f64.ln() + log_binom((l_eff_model + 1.0).max(2.0), 2.0);

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

    let mut disc_prefs: Vec<Vec<f64>> = Vec::new();
    let mut disc_pos_list: Vec<Vec<usize>> = Vec::new();
    for &ur in &u_reps {
        let st_ur = &node_state[&ur];
        for &vr in &v_reps {
            let st_vr = &node_state[&vr];
            let mut pref = Vec::with_capacity(l + 1);
            pref.push(0.0);
            let mut pos = Vec::new();
            let mut acc = 0.0_f64;
            for s in 0..l {
                if both_val[s] && st_ur[s] != st_vr[s] {
                    acc += subst.dist_weights[s];
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

    let mut w_pref = Vec::with_capacity(l + 1);
    w_pref.push(0.0_f64);
    let mut w_acc = 0.0_f64;
    for s in 0..l {
        if both_val[s] {
            w_acc += subst.dist_weights[s];
        }
        w_pref.push(w_acc);
    }

    let mut sav_pref = Vec::with_capacity(k_cand + 1);
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

    let w_val_total = w_pref[l].max(1.0);
    let min_span = min_k.max(3);
    for i in 0..k_cand {
        let a = cand_sites[i];
        for j in (i + min_k - 1)..k_cand {
            let b = cand_sites[j];
            if chrom_ids[a] != chrom_ids[b] {
                break;
            }
            let span_len = b - a + 1;
            if span_len < min_span || (span_len as f64) > (0.55 * (l as f64)).min(2500.0) {
                continue;
            }
            let gaps_in = (gap_pref[b + 1] - gap_pref[a]) as usize;
            if (gaps_in as f64) > 0.20 * (span_len as f64) {
                continue;
            }
            let eff_m_in = (w_pref[b + 1] - w_pref[a]).max(1.0);
            let m_in_eff_int = ((j - i + 1) as f64 / subst.avg_block_len).ceil() as usize;
            if m_in_eff_int < min_k {
                continue;
            }

            let mut best_rep_idx = 0usize;
            let mut discordant = f64::INFINITY;
            let mut total_disc_best = 0.0_f64;
            for (r_idx, dp) in disc_prefs.iter().enumerate() {
                let d_val = dp[b + 1] - dp[a];
                if d_val < discordant {
                    discordant = d_val;
                    best_rep_idx = r_idx;
                    total_disc_best = dp[l];
                }
            }
            let d_all = total_disc_best.max(discordant);
            let d_out = (d_all - discordant).max(0.0);
            let out_val = (w_val_total - eff_m_in).max(1.0);
            if discordant > 0.25 * (m_in_eff_int as f64)
                || (discordant / eff_m_in) >= (d_out / out_val)
            {
                continue;
            }
            let total_mask_cost = if discordant > 1e-6 {
                base_mask_cost + branch_mutation_code_nats(discordant, eff_m_in, 0.0, subst.k)
            } else {
                let m_donor = m_donor_base.max(m_in_eff_int as f64);
                let donor_int_cost =
                    2.0_f64.ln() + log_binom((m_donor + 1.0).max(2.0), 2.0);
                donor_int_cost.min(base_mask_cost)
            };
            let data_sav = ((sav_pref[j + 1] - sav_pref[i])
                - log_falling_factorial(m_recip_base, m_in_eff_int))
            .max(0.0);
            let net_gain = data_sav - endpoint_cost - total_mask_cost;
            if net_gain > best_gain {
                best_gain = net_gain;
                best_sites = cand_sites[i..=j].to_vec();
                let disc_positions = &disc_pos_list[best_rep_idx];
                let s_left = disc_positions
                    .iter()
                    .copied()
                    .rfind(|&p| p < a)
                    .unwrap_or_else(|| a.saturating_sub(4));
                let s_right = disc_positions
                    .iter()
                    .copied()
                    .find(|&p| p > b)
                    .unwrap_or_else(|| (b + 4).min(l - 1));
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
fn evaluate_discrete_sparse_shunt_mdl(
    min_shunt_sites: usize,
    u: usize,
    v: usize,
    shared_sites: &[usize],
    site_mut_counts: &[i32],
    site_target_counts: &[[u16; MAX_DISCRETE_STATES]],
    total_tree_muts: f64,
    node_state: &HashMap<usize, Vec<i8>>,
    eff_anc_node: &HashMap<usize, usize>,
    branch_mut_load: &HashMap<usize, f64>,
    l_eff_model: f64,
    chrom_ids: &[u16],
    endpoint_cost: f64,
    subst: &DiscreteSubstModel,
) -> (f64, Vec<usize>, f64, f64) {
    let raw_pairs = compute_discrete_parallel_site_savings(
        u,
        v,
        shared_sites,
        site_mut_counts,
        site_target_counts,
        total_tree_muts,
        node_state,
        eff_anc_node,
        branch_mut_load,
        l_eff_model,
        subst,
    );

    // If collinear bins, deduplicate contiguous runs into 1 sparse representative per chromosomal block
    let site_sav_pairs: Vec<(usize, f64)> = if subst.is_collinear_bins {
        let mut dedup = Vec::new();
        let mut idx = 0usize;
        while idx < raw_pairs.len() {
            let (s0, mut sum_sav) = raw_pairs[idx];
            let chr0 = chrom_ids[s0];
            let mut last_s = s0;
            let mut j = idx + 1;
            while j < raw_pairs.len() {
                let (sj, savj) = raw_pairs[j];
                if chrom_ids[sj] == chr0 && sj <= last_s + 4 {
                    sum_sav += savj;
                    last_s = sj;
                    j += 1;
                } else {
                    break;
                }
            }
            dedup.push((s0, sum_sav));
            idx = j;
        }
        dedup
    } else {
        raw_pairs
    };

    if site_sav_pairs.len() < min_shunt_sites {
        return (-1e9, Vec::new(), 0.0, 0.0);
    }

    let m_u = branch_mut_load.get(&u).copied().unwrap_or(1.0);
    let m_v = branch_mut_load.get(&v).copied().unwrap_or(1.0);
    let m_donor_base = m_u.max(m_v);
    let m_recip_base = m_u.min(m_v);

    let mut strict_parallel: Vec<(f64, usize)> =
        site_sav_pairs.into_iter().map(|(s, sav)| (sav, s)).collect();
    strict_parallel.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.1.cmp(&a.1))
    });

    let mut best_gain = -1e9_f64;
    let mut best_sites = Vec::new();
    let mut best_data_sav = 0.0_f64;
    let mut best_mask_cost = 0.0_f64;
    let mut cum_raw_info = 0.0_f64;
    let mut cur_sites = Vec::new();

    for (idx, &(raw_info, s)) in strict_parallel.iter().enumerate() {
        let k_ev = idx + 1;
        cum_raw_info += raw_info;
        cur_sites.push(s);
        if k_ev >= min_shunt_sites {
            let m_donor = m_donor_base.max(k_ev as f64);
            let mask_cost = 2.0_f64.ln()
                + crate::universal_int_nats((k_ev - min_shunt_sites) as f64)
                + log_binom(m_donor, k_ev as f64);
            let data_sav = (cum_raw_info - log_falling_factorial(m_recip_base, k_ev)).max(0.0);
            let net_gain = data_sav - endpoint_cost - mask_cost;
            if net_gain > best_gain {
                best_gain = net_gain;
                let mut sorted_s = cur_sites.clone();
                sorted_s.sort_unstable();
                best_sites = sorted_s;
                best_data_sav = data_sav;
                best_mask_cost = mask_cost;
            }
        }
    }

    (best_gain, best_sites, best_data_sav, best_mask_cost)
}

#[allow(clippy::too_many_arguments)]
fn peel_discrete_shunts_mdl(
    min_shunt_sites: usize,
    adj: &BTreeMap<usize, Vec<usize>>,
    node_labels: &BTreeMap<usize, String>,
    bitmasks: &[u32],
    majority_state: &[i8],
    terminal_loads: &[f64],
    edge_mutations: &HashMap<(usize, usize), Vec<(usize, i8, i8)>>,
    n: usize,
    l: usize,
    l_eff_raw: f64,
    chrom_ids: &[u16],
    subst: &DiscreteSubstModel,
) -> Vec<ShuntEdge> {
    let min_sites = min_shunt_sites.max(1);
    let active_nodes: Vec<usize> = adj
        .iter()
        .filter(|(_, nbs)| !nbs.is_empty())
        .map(|(&u, _)| u)
        .collect();
    let v_count = active_nodes.len().max(4) as f64;
    let endpoint_cost = 2.0 * v_count.ln();
    let l_eff_model = if subst.is_collinear_bins {
        subst.l_eff_seg
    } else {
        l_eff_raw
    };

    let root = adj
        .iter()
        .find(|(&u, nbs)| u >= n && !nbs.is_empty())
        .or_else(|| adj.iter().find(|(_, nbs)| !nbs.is_empty()))
        .map(|(&u, _)| u)
        .unwrap_or(0);

    let mut parent: HashMap<usize, Option<usize>> = HashMap::new();
    parent.insert(root, None);
    let mut q = VecDeque::from([root]);
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
    let mut site_mut_counts = vec![0_i32; l];
    let mut site_target_counts = vec![[0_u16; MAX_DISCRETE_STATES]; l];
    let mut total_tree_muts = 0.0_f64;

    for (&u, &p_opt) in &parent {
        let Some(p) = p_opt else { continue };
        let key = (p.min(u), p.max(u));
        let raw_list = edge_mutations.get(&key).cloned().unwrap_or_default();
        let directed: Vec<(usize, i8, i8)> = if p < u {
            raw_list
        } else {
            raw_list.into_iter().map(|(s, bu, bp)| (s, bp, bu)).collect()
        };
        let mut eff_load = 0.0_f64;
        for &(s, _, b_der) in &directed {
            site_mut_counts[s] += 1;
            site_target_counts[s][(b_der.max(0) as usize).min(subst.k - 1)] += 1;
            eff_load += subst.dist_weights[s];
        }
        total_tree_muts += eff_load;
        branch_mut_load.insert(u, eff_load.max(0.5));
        branch_by_child.insert(u, directed);
    }

    let states_dir =
        compute_all_directed_discrete_states(adj, bitmasks, n, l, terminal_loads, subst);
    let (node_state, node_valid, _) = reconstruct_discrete_nodal_states_and_mutations(
        adj,
        bitmasks,
        majority_state,
        &states_dir,
        n,
        l,
        subst,
    );

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
        if u >= n {
            let mut cum_load = orig_branch_load.get(&u).copied().unwrap_or(1.0);
            let mut curr_w = *p;
            let mut steps_up = 0usize;
            let mut existing_sites: HashSet<usize> = branch_by_child
                .get(&u)
                .map(|v| v.iter().map(|x| x.0).collect())
                .unwrap_or_default();
            let st_u_arr = &node_state[&u];
            let mut added_any = false;
            while curr_w >= n && steps_up < 8 {
                let Some(Some(next_p)) = parent.get(&curr_w).copied() else {
                    break;
                };
                let add_load = orig_branch_load.get(&curr_w).copied().unwrap_or(1.0);
                if cum_load + add_load > 28.0 {
                    break;
                }
                cum_load += add_load;
                let st_anc_arr = &node_state[&next_p];
                if let Some(w_muts) = orig_branch_by_child.get(&curr_w) {
                    for &(s_int, a_st, b_st) in w_muts {
                        if !existing_sites.contains(&s_int)
                            && node_valid[&u][s_int]
                            && st_u_arr[s_int] == b_st
                            && st_u_arr[s_int] != st_anc_arr[s_int]
                        {
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
                eff_anc = next_p;
                curr_w = next_p;
                steps_up += 1;
            }
            if added_any {
                let entry = branch_mut_load.entry(u).or_insert(1.0);
                *entry = entry.max(cum_load);
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

    let is_bin_node = |nd: usize| -> bool { adj.get(&nd).map(|v| v.len() <= 3).unwrap_or(false) };

    let mut pair_shared_sites: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    let mut pair_priority_score: HashMap<(usize, usize), f64> = HashMap::new();

    for (s, entries) in site_to_branches.iter().enumerate() {
        if entries.len() < 2 || entries.len() > 160 {
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
                    if w_u == w_v
                        || is_ancestor(u, v)
                        || is_ancestor(v, u)
                        || is_ancestor(w_u, v)
                        || is_ancestor(w_v, u)
                    {
                        continue;
                    }
                    if w_u != u || w_v != v {
                        let pw_u = parent.get(&w_u).copied().flatten();
                        let pw_v = parent.get(&w_v).copied().flatten();
                        let p_pw_u = pw_u.and_then(|x| parent.get(&x).copied().flatten());
                        let p_pw_v = pw_v.and_then(|x| parent.get(&x).copied().flatten());
                        let bin_sister = matches!((pw_u, pw_v), (Some(p1), Some(p2)) if p1 == p2 && is_bin_node(p1))
                            || matches!((p_pw_u, pw_v), (Some(p1), Some(p2)) if p1 == p2 && is_bin_node(p1))
                            || matches!((pw_u, p_pw_v), (Some(p1), Some(p2)) if p1 == p2 && is_bin_node(p1));
                        if bin_sister {
                            continue;
                        }
                    }
                    let pair = (u.min(v), u.max(v));
                    pair_shared_sites.entry(pair).or_default().push(s);
                    *pair_priority_score.entry(pair).or_insert(0.0) += subst.dist_weights[s] / bg_c;
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

    let eval_shunt_pair =
        |((u, v), mut shared_sites): ((usize, usize), Vec<usize>)| -> Option<ShuntEdge> {
            shared_sites.sort_unstable();
            shared_sites.dedup();
            let avail_sites: Vec<usize> = shared_sites
                .into_iter()
                .filter(|&s| node_valid[&u][s] && node_valid[&v][s])
                .collect();
            if avail_sites.len() < min_sites {
                return None;
            }

            let (best_int_gain, best_int_sites, best_span, int_data_sav, int_mask_cost) =
                evaluate_discrete_interval_shunt_mdl(
                    min_sites,
                    u,
                    v,
                    &avail_sites,
                    &site_mut_counts,
                    &site_target_counts,
                    total_tree_muts,
                    &node_state,
                    &node_valid,
                    &eff_anc_node,
                    &parent,
                    adj,
                    &branch_mut_load,
                    l,
                    l_eff_model,
                    chrom_ids,
                    endpoint_cost,
                    subst,
                );

            let pu_direct = parent.get(&u).copied().flatten();
            let pv_direct = parent.get(&v).copied().flatten();
            let local_triad = matches!((pu_direct, pv_direct), (Some(p1), Some(p2)) if p1 == p2 && is_bin_node(p1))
                || matches!((pu_direct.and_then(|p| parent.get(&p).copied().flatten()), pv_direct), (Some(p1), Some(p2)) if p1 == p2 && is_bin_node(p1))
                || matches!((pu_direct, pv_direct.and_then(|p| parent.get(&p).copied().flatten())), (Some(p1), Some(p2)) if p1 == p2 && is_bin_node(p1));

            let (best_sp_gain, best_sp_sites, sp_data_sav, sp_mask_cost) = if local_triad {
                (-10.0, Vec::new(), 0.0, 0.0)
            } else {
                evaluate_discrete_sparse_shunt_mdl(
                    min_sites,
                    u,
                    v,
                    &avail_sites,
                    &site_mut_counts,
                    &site_target_counts,
                    total_tree_muts,
                    &node_state,
                    &eff_anc_node,
                    &branch_mut_load,
                    l_eff_model,
                    chrom_ids,
                    endpoint_cost,
                    subst,
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

        if sh.sites.len().saturating_sub(overlap.len()) < min_sites {
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
                evaluate_discrete_interval_shunt_mdl(
                    min_sites,
                    u,
                    v,
                    &rem_sites,
                    &site_mut_counts,
                    &site_target_counts,
                    total_tree_muts,
                    &node_state,
                    &node_valid,
                    &eff_anc_node,
                    &parent,
                    adj,
                    &branch_mut_load,
                    l,
                    l_eff_model,
                    chrom_ids,
                    endpoint_cost,
                    subst,
                )
            } else {
                let (g, st, sv, mc) = evaluate_discrete_sparse_shunt_mdl(
                    min_sites,
                    u,
                    v,
                    &rem_sites,
                    &site_mut_counts,
                    &site_target_counts,
                    total_tree_muts,
                    &node_state,
                    &eff_anc_node,
                    &branch_mut_load,
                    l_eff_model,
                    chrom_ids,
                    endpoint_cost,
                    subst,
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

fn build_discrete_binary_refinement(
    adj_mdl: &BTreeMap<usize, Vec<usize>>,
    pre_step3_adj: &BTreeMap<usize, Vec<usize>>,
    bitmasks: &[u32],
    terminal_loads: &[f64],
    n: usize,
    l: usize,
    subst: &DiscreteSubstModel,
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
    let all_m = subst.all_mask;

    let compute_sub_bits = |graph: &BTreeMap<usize, Vec<usize>>| -> (
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
    let states =
        compute_all_directed_discrete_states(adj_mdl, bitmasks, n, l, terminal_loads, subst);

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
                    if !(0..n_words).all(|w| (cm[w] & !sp[w]) == 0) {
                        clean = false;
                        break;
                    }
                    pm[r >> 6] |= 1_u64 << (r & 63);
                    u_cnt += 1;
                } else if !(0..n_words).all(|w| (cm[w] & sp[w]) == 0) {
                    clean = false;
                    break;
                }
            }
            if clean && u_cnt >= 2 && u_cnt <= d - 2 {
                hub_splits.push(pm);
            }
        }

        let mut local_adj: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for r in 0..d {
            local_adj.insert(r, vec![d]);
        }
        local_adj.insert(d, (0..d).collect());
        let mut next_loc = d + 1;

        let mut item_pm: HashMap<usize, Vec<u64>> = HashMap::new();
        let mut local_seqs: HashMap<usize, Vec<u32>> = HashMap::new();
        for (r, &c) in nbs.iter().enumerate() {
            let mut v = vec![0_u64; d_words];
            v[r >> 6] |= 1_u64 << (r & 63);
            item_pm.insert(r, v);
            local_seqs.insert(
                r,
                states
                    .get(&(c, h))
                    .map(|st| st.0.clone())
                    .unwrap_or_else(|| vec![all_m; l]),
            );
        }

        while local_adj[&d].len() >= 4 {
            let cur_nbs = local_adj[&d].clone();
            let cur_d = cur_nbs.len();
            let mut matched: Option<Vec<usize>> = None;
            for sp_pm in &hub_splits {
                let mut u_set = Vec::new();
                let mut clean = true;
                for &it in &cur_nbs {
                    let pm = &item_pm[&it];
                    let has_in = (0..d_words).any(|w| (pm[w] & sp_pm[w]) != 0);
                    let has_out = (0..d_words).any(|w| (pm[w] & !sp_pm[w]) != 0);
                    if has_in && has_out {
                        clean = false;
                        break;
                    }
                    if has_in {
                        u_set.push(it);
                    }
                }
                if clean && u_set.len() >= 2 && u_set.len() <= cur_d - 2 {
                    matched = Some(u_set);
                    break;
                }
            }
            let Some(chosen) = matched else { break };
            let w_loc = next_loc;
            next_loc += 1;
            local_adj.insert(w_loc, vec![d]);
            local_adj.get_mut(&d).unwrap().push(w_loc);
            let mut pm_w = vec![0_u64; d_words];
            for &it in &chosen {
                for w in 0..d_words {
                    pm_w[w] |= item_pm[&it][w];
                }
                local_adj.get_mut(&d).unwrap().retain(|&x| x != it);
                local_adj.get_mut(&it).unwrap().retain(|&x| x != d);
                local_adj.get_mut(&w_loc).unwrap().push(it);
                local_adj.get_mut(&it).unwrap().push(w_loc);
            }
            let ch_refs: Vec<(&[u32], f64)> = chosen
                .iter()
                .map(|&it| (local_seqs[&it].as_slice(), 1.0))
                .collect();
            let st_w = combine_discrete_children_refs(&ch_refs, subst).0;
            item_pm.insert(w_loc, pm_w);
            local_seqs.insert(w_loc, st_w);
        }

        if local_adj[&d].len() >= 4 {
            let rem_nbs = local_adj[&d].clone();
            local_adj.get_mut(&d).unwrap().clear();
            for &it in &rem_nbs {
                local_adj.get_mut(&it).unwrap().retain(|&x| x != d);
            }
            let mut sub_labels: BTreeMap<usize, String> = BTreeMap::new();
            let bionj_start = next_loc;
            agglomerate_discrete_bionj(
                &rem_nbs,
                &mut local_seqs,
                subst,
                &mut local_adj,
                &mut sub_labels,
                &mut next_loc,
            );
            let center_alias = next_loc - 1;
            if center_alias >= bionj_start {
                let cnbs = local_adj.remove(&center_alias).unwrap_or_default();
                for &x in &cnbs {
                    if let Some(xnbs) = local_adj.get_mut(&x) {
                        for entry in xnbs.iter_mut() {
                            if *entry == center_alias {
                                *entry = d;
                            }
                        }
                    }
                }
                local_adj.insert(d, cnbs);
                next_loc -= 1;
            }
        }

        let k_new = next_loc - (d + 1);
        let port_owner: Vec<usize> = (0..d).map(|r| local_adj[&r][0] - d).collect();
        let mut local_int_edges: Vec<(usize, usize)> = Vec::with_capacity(k_new);
        for a in d..next_loc {
            if let Some(anbs) = local_adj.get(&a) {
                for &b in anbs {
                    if b >= d && a < b {
                        local_int_edges.push((a - d, b - d));
                    }
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

fn serialize_discrete_newick(
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
        return format!(
            "({}:{:.6},{}:{:.6});",
            names[0],
            b_len / 2.0,
            names[1],
            b_len / 2.0
        );
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
fn serialize_discrete_binary_newick(
    bin_adj: &BTreeMap<usize, Vec<usize>>,
    names: &[String],
    bitmasks: &[u32],
    majority_state: &[i8],
    terminal_loads: &[f64],
    n: usize,
    l: usize,
    l_eff: f64,
    chrom_ids: &[u16],
    subst: &DiscreteSubstModel,
) -> String {
    let l_denom = if subst.is_collinear_bins {
        subst.l_eff_seg
    } else {
        l_eff
    };
    let min_bl = (0.5 / l_denom.max(1.0)).max(1e-5);
    if n == 1 {
        return format!("{}:0.000010;", names[0]);
    }
    if n == 2 {
        return format!("({}:{min_bl:.6},{}:{min_bl:.6});", names[0], names[1]);
    }
    let states =
        compute_all_directed_discrete_states(bin_adj, bitmasks, n, l, terminal_loads, subst);
    let (node_states, node_valid, _) = reconstruct_discrete_nodal_states_and_mutations(
        bin_adj,
        bitmasks,
        majority_state,
        &states,
        n,
        l,
        subst,
    );
    let mut edge_bl: HashMap<(usize, usize), f64> = HashMap::new();
    for (&u, nbs) in bin_adj {
        for &v in nbs {
            if u < v {
                let (_, m_pt, r_runs) = ordered_or_discrete_branch_code_nats(
                    &node_states[&u],
                    &node_states[&v],
                    &node_valid[&u],
                    &node_valid[&v],
                    chrom_ids,
                    l_eff,
                    subst,
                );
                let m_cnt = if subst.is_collinear_bins {
                    r_runs as f64
                } else {
                    m_pt as f64
                };
                edge_bl.insert((u, v), (m_cnt / l_denom.max(1.0)).max(min_bl));
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

    fn build_bin(
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
                .map(|c| build_bin(c, Some(u), adj, names, edge_bl, visited, n, min_bl))
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

    build_bin(root, None, bin_adj, names, &edge_bl, &mut visited, n, min_bl)
}

#[allow(clippy::too_many_arguments)]
fn build_discrete_loads_and_backbone(
    names: &[String],
    mat: &[i8],
    bitmasks: &[u32],
    n: usize,
    l: usize,
    l_eff: f64,
    chrom_ids: &[u16],
    subst: &DiscreteSubstModel,
    build_backbone: bool,
) -> (Vec<f64>, BTreeMap<usize, Vec<usize>>, BTreeMap<usize, String>) {
    let mut node_labels: BTreeMap<usize, String> =
        (0..n).map(|i| (i, names[i].clone())).collect();
    let mut adj: BTreeMap<usize, Vec<usize>> = (0..n).map(|i| (i, Vec::new())).collect();

    if n == 1 {
        return (vec![0.5; 1], adj, node_labels);
    }

    let l_target = subst.l_eff_seg;

    if n <= 4096 {
        #[cfg(not(target_arch = "wasm32"))]
        let pair_rows: Vec<Vec<f64>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut uw_r = vec![0.0_f64; n];
                let ri = &mat[i * l..(i + 1) * l];
                for j in (i + 1)..n {
                    let rj = &mat[j * l..(j + 1) * l];
                    uw_r[j] = discrete_leaf_dist(ri, rj, &subst.dist_weights, l_target, subst);
                }
                uw_r
            })
            .collect();
        #[cfg(target_arch = "wasm32")]
        let pair_rows: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let mut uw_r = vec![0.0_f64; n];
                let ri = &mat[i * l..(i + 1) * l];
                for j in (i + 1)..n {
                    let rj = &mat[j * l..(j + 1) * l];
                    uw_r[j] = discrete_leaf_dist(ri, rj, &subst.dist_weights, l_target, subst);
                }
                uw_r
            })
            .collect();

        let mut uw_dist = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in (i + 1)..n {
                let du = pair_rows[i][j];
                uw_dist[i * n + j] = du;
                uw_dist[j * n + i] = du;
            }
        }
        drop(pair_rows);

        let mut loads = vec![0.5_f64; n];
        for i in 0..n {
            let row_i = &uw_dist[i * n..(i + 1) * n];
            let mut j_near = if i == 0 { 1 } else { 0 };
            let mut d_min = row_i[j_near];
            for j in 0..n {
                if j != i && row_i[j] < d_min {
                    d_min = row_i[j];
                    j_near = j;
                }
            }
            if n == 2 {
                loads[i] = (0.5 * d_min).clamp(0.45, 0.85 * l_target.max(2.0));
            } else {
                let row_near = &uw_dist[j_near * n..(j_near + 1) * n];
                let mut deltas: Vec<f64> = (0..n)
                    .filter(|&k_i| k_i != i && k_i != j_near)
                    .map(|k_i| row_i[k_i] - row_near[k_i])
                    .collect();
                let med = median_f64(&mut deltas);
                let min_ld = if l_target <= 64.0 { 0.05 } else { 0.45 };
                let raw_add = (0.5 * (d_min + med)).clamp(min_ld, d_min.max(0.8));
                loads[i] = raw_add.clamp(min_ld, 0.85 * l_target.max(2.0));
            }
        }

        if !build_backbone {
            return (loads, adj, node_labels);
        }
        let all_ids: Vec<usize> = (0..n).collect();
        let mut next_node = n;
        agglomerate_discrete_bionj_from_dist(
            &all_ids,
            uw_dist,
            None,
            subst,
            &mut adj,
            &mut node_labels,
            &mut next_node,
        );
        if n >= 4 {
            refine_discrete_nni_mdl(
                &mut adj,
                bitmasks,
                &loads,
                n,
                l,
                l_eff,
                chrom_ids,
                subst,
                3,
            );
            refine_discrete_spr_mdl(
                &mut adj,
                bitmasks,
                &loads,
                n,
                l,
                l_eff,
                chrom_ids,
                subst,
                2,
            );
            refine_discrete_nni_mdl(
                &mut adj,
                bitmasks,
                &loads,
                n,
                l,
                l_eff,
                chrom_ids,
                subst,
                2,
            );
        }
        return (loads, adj, node_labels);
    }

    // Sub-quadratic Voronoi landmark divide-and-conquer for N > 4,096 single cells
    let mut leaf_clusters: Vec<Vec<usize>> = Vec::new();
    partition_discrete_voronoi((0..n).collect(), mat, l, subst, 512, &mut leaf_clusters);
    let n_anc = 32.min(n);
    let anchors: Vec<usize> = (0..n_anc).map(|i| (i * (n - 1)) / (n_anc - 1)).collect();

    let eval_cluster_loads = |cluster: &Vec<usize>| -> Vec<(usize, f64)> {
        let mut ref_set = cluster.clone();
        for &ga in &anchors {
            if !ref_set.contains(&ga) {
                ref_set.push(ga);
            }
        }
        let m_r = ref_set.len();
        let mut dmat = vec![0.0_f64; m_r * m_r];
        for a in 0..m_r {
            let ra = &mat[ref_set[a] * l..(ref_set[a] + 1) * l];
            for b in (a + 1)..m_r {
                let rb = &mat[ref_set[b] * l..(ref_set[b] + 1) * l];
                let d = discrete_leaf_dist(ra, rb, &subst.dist_weights, l_target, subst);
                dmat[a * m_r + b] = d;
                dmat[b * m_r + a] = d;
            }
        }
        cluster
            .iter()
            .enumerate()
            .map(|(loc_i, &gi)| {
                let mut best_b = if loc_i == 0 { 1 } else { 0 };
                let mut d_min = dmat[loc_i * m_r + best_b];
                for b in 0..m_r {
                    if b != loc_i && dmat[loc_i * m_r + b] < d_min {
                        d_min = dmat[loc_i * m_r + b];
                        best_b = b;
                    }
                }
                let mut deltas: Vec<f64> = (0..m_r)
                    .filter(|&k_i| k_i != loc_i && k_i != best_b)
                    .map(|k_i| dmat[loc_i * m_r + k_i] - dmat[best_b * m_r + k_i])
                    .collect();
                let med = median_f64(&mut deltas);
                let min_ld = if l_target <= 64.0 { 0.05 } else { 0.45 };
                let raw_add = (0.5 * (d_min + med)).clamp(min_ld, d_min.max(0.8));
                (gi, raw_add.clamp(min_ld, 0.85 * l_target.max(2.0)))
            })
            .collect()
    };

    #[cfg(not(target_arch = "wasm32"))]
    let load_batches: Vec<Vec<(usize, f64)>> =
        leaf_clusters.par_iter().map(eval_cluster_loads).collect();
    #[cfg(target_arch = "wasm32")]
    let load_batches: Vec<Vec<(usize, f64)>> =
        leaf_clusters.iter().map(eval_cluster_loads).collect();

    let mut loads = vec![0.5_f64; n];
    for batch in load_batches {
        for (gi, lv) in batch {
            loads[gi] = lv;
        }
    }

    if !build_backbone {
        return (loads, adj, node_labels);
    }

    let mut seqs_map: HashMap<usize, Vec<u32>> = HashMap::new();
    const TEMP_BASE: usize = usize::MAX / 2;
    let build_cluster_bionj = |cluster: &Vec<usize>| -> (
        usize,
        BTreeMap<usize, Vec<usize>>,
        HashMap<usize, Vec<u32>>,
    ) {
        let mut loc_seqs: HashMap<usize, Vec<u32>> = cluster
            .iter()
            .map(|&gi| (gi, bitmasks[gi * l..(gi + 1) * l].to_vec()))
            .collect();
        let mut loc_adj: BTreeMap<usize, Vec<usize>> =
            cluster.iter().map(|&gi| (gi, Vec::new())).collect();
        let mut loc_labels: BTreeMap<usize, String> = BTreeMap::new();
        let mut loc_next = TEMP_BASE;
        agglomerate_discrete_bionj(
            cluster,
            &mut loc_seqs,
            subst,
            &mut loc_adj,
            &mut loc_labels,
            &mut loc_next,
        );
        (loc_next - TEMP_BASE, loc_adj, loc_seqs)
    };

    #[cfg(not(target_arch = "wasm32"))]
    let cluster_trees = leaf_clusters
        .par_iter()
        .map(build_cluster_bionj)
        .collect::<Vec<_>>();
    #[cfg(target_arch = "wasm32")]
    let cluster_trees = leaf_clusters
        .iter()
        .map(build_cluster_bionj)
        .collect::<Vec<_>>();

    let anchor_seqs: Vec<Vec<u32>> = anchors
        .iter()
        .map(|&ga| bitmasks[ga * l..(ga + 1) * l].to_vec())
        .collect();

    let mut next_node = n;
    let mut portal_reps = Vec::with_capacity(leaf_clusters.len());
    for (cluster, (k_int, loc_adj, mut loc_seqs)) in leaf_clusters.iter().zip(cluster_trees) {
        let base_id = next_node;
        next_node += k_int;
        let map_id = |x: usize| -> usize {
            if x >= TEMP_BASE {
                base_id + (x - TEMP_BASE)
            } else {
                x
            }
        };
        for &gi in cluster {
            if let Some(seq) = loc_seqs.remove(&gi) {
                seqs_map.insert(gi, seq);
            }
        }
        for j in 0..k_int {
            let gid = base_id + j;
            node_labels.insert(gid, format!("N{gid}"));
            if let Some(seq) = loc_seqs.remove(&(TEMP_BASE + j)) {
                seqs_map.insert(gid, seq);
            }
        }
        for (raw_u, raw_nbs) in loc_adj {
            let gu = map_id(raw_u);
            let entry = adj.entry(gu).or_default();
            for raw_v in raw_nbs {
                entry.push(map_id(raw_v));
            }
        }

        let graft_u = if k_int >= 1 {
            let mut best_g = base_id + k_int - 1;
            let mut best_ad = f64::INFINITY;
            let step_g = (k_int / 16).max(1);
            for j in (0..k_int).step_by(step_g) {
                let gid = base_id + j;
                if let Some(gs) = seqs_map.get(&gid) {
                    let ad: f64 = anchor_seqs
                        .iter()
                        .take(8)
                        .map(|asq| discrete_mask_dist(gs, asq, &subst.dist_weights, l_target, subst))
                        .sum();
                    if ad < best_ad {
                        best_ad = ad;
                        best_g = gid;
                    }
                }
            }
            best_g
        } else {
            cluster[0]
        };
        if adj[&graft_u].is_empty() {
            portal_reps.push(graft_u);
            continue;
        }
        let hook = adj[&graft_u]
            .iter()
            .copied()
            .find(|&nb| nb >= n)
            .unwrap_or(adj[&graft_u][0]);
        let portal = next_node;
        next_node += 1;
        node_labels.insert(portal, format!("N{portal}"));
        adj.get_mut(&graft_u).unwrap().retain(|&x| x != hook);
        adj.get_mut(&hook).unwrap().retain(|&x| x != graft_u);
        adj.insert(portal, vec![graft_u, hook]);
        adj.get_mut(&graft_u).unwrap().push(portal);
        adj.get_mut(&hook).unwrap().push(portal);
        let p_seq = merge_discrete_steiner_median(
            &seqs_map[&graft_u],
            &seqs_map[&hook],
            &anchor_seqs[0],
            subst,
        );
        seqs_map.insert(portal, p_seq);
        portal_reps.push(portal);
    }

    if portal_reps.len() >= 2 {
        agglomerate_discrete_bionj(
            &portal_reps,
            &mut seqs_map,
            subst,
            &mut adj,
            &mut node_labels,
            &mut next_node,
        );
    }
    if n >= 4 {
        refine_discrete_nni_mdl(
            &mut adj,
            bitmasks,
            &loads,
            n,
            l,
            l_eff,
            chrom_ids,
            subst,
            2,
        );
        refine_discrete_spr_mdl(
            &mut adj,
            bitmasks,
            &loads,
            n,
            l,
            l_eff,
            chrom_ids,
            subst,
            1,
        );
    }

    (loads, adj, node_labels)
}

/// Primary entry point for general `K`-state `Unordered` (`--discrete`) or `Ordered` (`--ordered`)
/// character-matrix MDL graph induction.
pub fn infer_discrete_mdl(
    engine: &BushwhackerEngine,
    char_mat: &DiscreteCharacterMatrix,
    mode: DiscreteMode,
    starting_newick: Option<(&str, bool)>,
) -> Result<BushwhackerResult, String> {
    #[cfg(not(target_arch = "wasm32"))]
    let t0 = Instant::now();

    let n = char_mat.n_taxa;
    let l = char_mat.n_sites;
    if n == 0 || l == 0 {
        return Err("Empty discrete character matrix.".to_string());
    }
    let names = &char_mat.names;
    let mat = &char_mat.mat;
    let bitmasks = &char_mat.bitmasks;
    let majority_state = &char_mat.majority_state;
    let chrom_ids = &char_mat.chrom_ids;

    let total_valid: usize = mat.iter().filter(|&&x| x >= 0).count();
    let l_eff = ((total_valid as f64) / (n as f64)).clamp(1.0, l as f64);
    let subst = estimate_discrete_subst_model(char_mat, mode);

    let (terminal_loads, mut adj, node_labels, expand_hubs) =
        if let Some((nwk, exp_hubs)) = starting_newick {
            let (loads, _, _) = build_discrete_loads_and_backbone(
                names,
                mat,
                bitmasks,
                n,
                l,
                l_eff,
                chrom_ids,
                &subst,
                false,
            );
            let (parsed_adj, parsed_labels) =
                BushwhackerEngine::parse_newick_to_unrooted_adj(nwk, names)?;
            (loads, parsed_adj, parsed_labels, exp_hubs)
        } else {
            let (loads, built_adj, built_labels) = build_discrete_loads_and_backbone(
                names,
                mat,
                bitmasks,
                n,
                l,
                l_eff,
                chrom_ids,
                &subst,
                true,
            );
            (loads, built_adj, built_labels, true)
        };

    let pre_step3_adj = adj.clone();
    let mut full_labels = node_labels;
    let (adj_c, mut branches, mut node_states, mut node_valid, edge_mutations) =
        contract_discrete_edges_mdl(
            adj,
            bitmasks,
            majority_state,
            &terminal_loads,
            n,
            l,
            l_eff,
            chrom_ids,
            &subst,
            &mut full_labels,
            expand_hubs,
        );
    adj = adj_c;

    let mut bin_adj = build_discrete_binary_refinement(
        &adj,
        &pre_step3_adj,
        bitmasks,
        &terminal_loads,
        n,
        l,
        &subst,
    );
    if n >= 4 {
        refine_discrete_nni_mdl(
            &mut bin_adj,
            bitmasks,
            &terminal_loads,
            n,
            l,
            l_eff,
            chrom_ids,
            &subst,
            2,
        );
    }
    let binary_newick = serialize_discrete_binary_newick(
        &bin_adj,
        names,
        bitmasks,
        majority_state,
        &terminal_loads,
        n,
        l,
        l_eff,
        chrom_ids,
        &subst,
    );

    for &u in adj.keys() {
        full_labels.entry(u).or_insert_with(|| format!("N{u}"));
    }

    let mut shunts = Vec::new();
    if engine.detect_shunts && n >= 4 {
        shunts = peel_discrete_shunts_mdl(
            engine.min_shunt_sites,
            &adj,
            &full_labels,
            bitmasks,
            majority_state,
            &terminal_loads,
            &edge_mutations,
            n,
            l,
            l_eff,
            chrom_ids,
            &subst,
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
                        for item in row.iter_mut().take(b_sp.min(l - 1) + 1).skip(a_sp) {
                            *item = subst.all_mask;
                        }
                        masked_any = true;
                    }
                }
            }
            if masked_any {
                let (adj_re, branches_re, ns_re, nv_re, _) = contract_discrete_edges_mdl(
                    adj,
                    &masked_bitmasks,
                    majority_state,
                    &terminal_loads,
                    n,
                    l,
                    l_eff,
                    chrom_ids,
                    &subst,
                    &mut full_labels,
                    false,
                );
                adj = adj_re;
                branches = branches_re;
                node_states = ns_re;
                node_valid = nv_re;
            }
        }
    }

    let ln_k = (subst.k as f64).ln();
    let raw_nats = (n as f64) * l_eff * ln_k;

    // Exact Star-Tree MDL baseline in K-state Discrete / Ordered RLE space
    let star_children: Vec<(Vec<u32>, f64)> = (0..n)
        .map(|i| (bitmasks[i * l..(i + 1) * l].to_vec(), 1.0))
        .collect();
    let (root_mask, _) = combine_discrete_children(&star_children, &subst);
    let mut root_st = vec![0_i8; l];
    let root_valid = vec![true; l];
    let k_max_i8 = (subst.k.saturating_sub(1)) as i8;
    for s in 0..l {
        let m = root_mask[s];
        if m == 0 || m == subst.all_mask {
            root_st[s] = majority_state[s];
        } else if subst.mode == DiscreteMode::Ordered {
            let lo = (m.trailing_zeros() as i8).clamp(0, k_max_i8);
            let hi = ((31 - m.leading_zeros()) as i8).clamp(lo, k_max_i8);
            root_st[s] = majority_state[s].clamp(lo, hi);
        } else if (m & (1_u32 << (majority_state[s].max(0) as u32))) != 0 {
            root_st[s] = majority_state[s];
        } else {
            root_st[s] = (m.trailing_zeros() as i8).clamp(0, k_max_i8);
        }
    }

    let root_profile_nats = if subst.is_collinear_bins {
        // Transmit root chromosomal profile via RLE from diploid/modal baseline
        let baseline = vec![majority_state[0]; l];
        let (r_rle, _, _) = ordered_or_discrete_branch_code_nats(
            &baseline,
            &root_st,
            &root_valid,
            &root_valid,
            chrom_ids,
            l_eff,
            &subst,
        );
        (ln_k + r_rle).min(l_eff * ln_k)
    } else {
        l_eff * ln_k
    };

    let mut star_nats = root_profile_nats;
    for i in 0..n {
        let row_st = &mat[i * l..(i + 1) * l];
        let row_val: Vec<bool> = row_st.iter().map(|&x| x >= 0).collect();
        let (b_nats, _, _) = ordered_or_discrete_branch_code_nats(
            &root_st,
            row_st,
            &root_valid,
            &row_val,
            chrom_ids,
            l_eff,
            &subst,
        );
        star_nats += b_nats;
    }

    let mut backbone_nats = root_profile_nats;
    for b in &branches {
        let (b_nats, _, _) = ordered_or_discrete_branch_code_nats(
            &node_states[&b.u],
            &node_states[&b.v],
            &node_valid[&b.u],
            &node_valid[&b.v],
            chrom_ids,
            l_eff,
            &subst,
        );
        backbone_nats += b_nats;
        if b.is_internal {
            let deg_u = adj[&b.u].len().saturating_sub(1);
            let deg_v = adj[&b.v].len().saturating_sub(1);
            backbone_nats += split_topology_cost_nats(deg_u, deg_v);
        }
    }
    let shunt_gain_total: f64 = shunts.iter().map(|s| s.net_mdl_gain_nats).sum();
    let full_graph_nats = root_profile_nats.max(backbone_nats - shunt_gain_total);

    let mut polytomy_degrees = Vec::new();
    for (&u, nbs) in &adj {
        if u >= n && nbs.len() > 3 {
            polytomy_degrees.push(nbs.len());
        }
    }
    let n_int = branches.iter().filter(|b| b.is_internal).count();
    let newick = serialize_discrete_newick(&adj, names, &branches, n);

    #[cfg(not(target_arch = "wasm32"))]
    let runtime_sec = t0.elapsed().as_secs_f64();
    #[cfg(target_arch = "wasm32")]
    let runtime_sec = 0.0;

    Ok(BushwhackerResult {
        names: names.clone(),
        n_taxa: n,
        n_sites: l,
        l_eff: if subst.is_collinear_bins {
            subst.l_eff_seg
        } else {
            l_eff
        },
        newick,
        binary_newick,
        adjacency: adj,
        node_labels: full_labels,
        branches,
        shunts,
        n_internal_branches: n_int,
        max_binary_internal_branches: n.saturating_sub(3),
        polytomy_nodes: polytomy_degrees.len(),
        max_polytomy_degree: polytomy_degrees.into_iter().max().unwrap_or(3),
        raw_uncompressed_nats: raw_nats,
        star_tree_nats: star_nats,
        backbone_tree_nats: backbone_nats,
        full_graph_nats,
        runtime_sec,
        site_weights: subst.dist_weights,
        terminal_loads,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kraft_normalization_and_ordered_step_monotonicity() {
        let csv = "\
cell,c0,c1,c2,c3,c4,c5
C0,0,1,2,3,4,11
C1,0,1,2,3,5,10
C2,1,2,3,4,5,9
C3,2,2,4,5,6,8
";
        let cm = parse_discrete_str(csv).unwrap();
        assert_eq!(cm.n_taxa, 4);
        assert_eq!(cm.n_sites, 6);
        assert_eq!(cm.k_states, 12);

        for mode in [DiscreteMode::Unordered, DiscreteMode::Ordered] {
            let subst = estimate_discrete_subst_model(&cm, mode);
            let pi_sum: f64 = subst.pi[..subst.k].iter().sum();
            assert!((pi_sum - 1.0).abs() < 1e-12);
            for a in 0..subst.k {
                let mut kraft_sum = 0.0_f64;
                for b in 0..subst.k {
                    if b != a {
                        kraft_sum += (-subst.cost[a][b]).exp();
                    }
                }
                assert!(
                    (kraft_sum - 1.0).abs() < 1e-11,
                    "Kraft equality violated for mode={mode:?}, a={a}: sum={kraft_sum}"
                );
            }
            if mode == DiscreteMode::Ordered {
                assert!(subst.cost[2][3] < subst.cost[2][4]);
                assert!(subst.cost[2][4] < subst.cost[2][5]);
                assert!(subst.cost[2][5] < subst.cost[2][8]);
            }
        }
    }

    #[test]
    fn test_csv_and_symbol_fasta_parsing_with_missing() {
        let fasta = "\
>CloneA
012?A-V
>CloneB
0122B?V
>CloneC
1120A10
>CloneD
1120A10
";
        let cm_fa = parse_discrete_str(fasta).unwrap();
        assert_eq!(cm_fa.n_taxa, 4);
        assert_eq!(cm_fa.n_sites, 7);
        assert_eq!(cm_fa.k_states, 32); // 'V' -> 31 -> K = 32
        assert_eq!(cm_fa.mat[0], 0);
        assert_eq!(cm_fa.mat[3], -1); // '?' -> missing
        assert_eq!(cm_fa.mat[4], 10); // 'A' -> 10
        assert_eq!(cm_fa.mat[5], -1); // '-' -> missing
        assert_eq!(cm_fa.mat[6], 31); // 'V' -> 31

        let bin_csv = "\
chr,start,end,Cell1,Cell2,Cell3,Cell4
1,1,500000,2,2,3,3
1,500001,1000000,2,NA,3,4
X,1,500000,2,2,1,1
";
        let cm_csv = parse_discrete_str(bin_csv).unwrap();
        assert_eq!(cm_csv.n_taxa, 4);
        assert_eq!(cm_csv.n_sites, 3);
        assert_eq!(cm_csv.k_states, 5);
        assert_eq!(cm_csv.mat[1 * 3 + 1], -1); // Cell2 bin 1 == NA
        assert_ne!(cm_csv.chrom_ids[0], cm_csv.chrom_ids[2]); // chr1 vs chrX
    }

    #[test]
    fn test_pure_star_radiation_contracts_to_zero_false_splits() {
        // 10 subclones radiating from a diploid (CN=2) ancestor with strictly private copy-number hits
        let mut lines = vec!["cell,".to_string()
            + &(0..40)
                .map(|s| format!("seg_{s}"))
                .collect::<Vec<_>>()
                .join(",")];
        for i in 0..10 {
            let mut row = vec![2_i32; 40];
            row[i * 3] = 3;
            row[i * 3 + 1] = 1;
            lines.push(format!(
                "Cell{i},{}",
                row.iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            ));
        }
        let cm = parse_discrete_str(&lines.join("\n")).unwrap();
        let engine = BushwhackerEngine::new(false, false, 2);
        let res = infer_discrete_mdl(&engine, &cm, DiscreteMode::Ordered, None).unwrap();
        assert_eq!(
            res.n_internal_branches, 0,
            "Pure clonal star radiation must contract to 0 internal splits, got {}",
            res.n_internal_branches
        );
        assert_eq!(res.polytomy_nodes, 1);
    }

    #[test]
    fn test_ordered_cna_clade_and_interval_shunt_recovery() {
        // 8 tumor subclones across 60 chromosomal segments:
        // Clade A (0..4) vs Clade B (4..8) separated by 8 clean trunk segments (0..8),
        // plus a parallel 6-segment chromosomal LOH block (segments 30..36: CN 2 -> 1)
        // shared between subclone A0 and subclone B4!
        let l = 60usize;
        let mut lines = vec![
            "cell,".to_string()
                + &(0..l)
                    .map(|s| format!("1_{}_{}", s * 500_000 + 1, (s + 1) * 500_000))
                    .collect::<Vec<_>>()
                    .join(","),
        ];
        for i in 0..8 {
            let mut row = vec![2_i32; l];
            if i < 4 {
                for s in 0..8 {
                    row[s] = 3;
                }
                if i < 2 {
                    for s in 8..13 {
                        row[s] = 4;
                    }
                }
            } else {
                for s in 14..22 {
                    row[s] = 1;
                }
                if i >= 6 {
                    for s in 22..27 {
                        row[s] = 0;
                    }
                }
            }
            // Private terminal noise so branches have realistic positive load
            row[40 + i * 2] = 3;
            row[40 + i * 2 + 1] = 1;

            // Convergent segmental LOH block across segments 30..36 in Clone0 and Clone4
            if i == 0 || i == 4 {
                for s in 30..36 {
                    row[s] = 1;
                }
            }
            lines.push(format!(
                "Clone{i},{}",
                row.iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            ));
        }
        let cm = parse_discrete_str(&lines.join("\n")).unwrap();
        let engine = BushwhackerEngine::new(false, true, 3);
        let res = infer_discrete_mdl(&engine, &cm, DiscreteMode::Ordered, None).unwrap();
        assert!(res.n_internal_branches >= 2);
        assert!(res.backbone_tree_nats < res.star_tree_nats);
        assert!(
            res.shunts.iter().any(|s| s.mode == "INTERVAL"),
            "Expected INTERVAL segmental LOH shunt across segments 30..35, got {:?}",
            res.shunts
        );
    }

    #[test]
    fn test_small_panel_somatic_genotype_clones_resolution() {
        // 6-locus somatic Tapestri zygosity panel (0=WT, 1=Het, 2=Hom/LOH, ?=dropout)
        // with extant ancestral hubs + daughter signaling clones.
        let csv = "\
taxon,DNMT3A,NPM1,FLT3,NRAS,KRAS,TP53
T00,0,0,0,0,0,0
T01,1,0,0,0,0,0
T02,1,1,0,0,0,0
T03,1,1,1,0,0,0
T04,1,1,2,0,0,0
T05,1,1,0,1,0,0
T06,1,1,0,0,1,?
T07,0,0,0,0,0,1
T08,0,0,0,0,0,2
";
        let cm = parse_discrete_str(csv).unwrap();
        let engine = BushwhackerEngine::new(false, true, 1);
        let res = infer_discrete_mdl(&engine, &cm, DiscreteMode::Ordered, None).unwrap();
        assert!(
            res.n_internal_branches >= 2,
            "Expected resolved somatic clone splits on L=6 panel, got {}",
            res.n_internal_branches
        );
        assert!(res.backbone_tree_nats < res.star_tree_nats);
    }
}
