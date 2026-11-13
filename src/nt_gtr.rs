//! Site-Tiled 4-State `GTR+CAT8` Eigen-Felsenstein-Rissanen MDL Engine for Nucleotide Supermatrices.
//!
//! Mirrors the 64-site Structure-of-Arrays (`SoA`) cache-tiled architecture of `src/protein.rs`,
//! specialized to 4-state nucleotides (`A, C, G, T` + all 15 non-empty 4-bit IUPAC ambiguity masks):
//!   1. Empirical General Time-Reversible (`GTR`) rate matrix estimation + exact 4x4 Jacobi
//!      symmetric eigendecomposition (`B = Pi^{1/2} Q Pi^{-1/2} = V Lambda V^T`).
//!   2. Parallel Empirical-Bayes Gamma(1.5)-Poisson `CAT8` relative site-rate categorization and
//!      parallel cache-blocked column-pattern deduplication (`100%` thread-count deterministic).
//!   3. Vectorized 64-site `SoA` (`[[f32; 64]; 4]`) `NNI + LeafMLE + SPR2 + NNI` topology and
//!      branch-length optimization with exact Rissanen Fisher-information MDL contraction support.

use std::collections::{BTreeMap, HashMap, HashSet};

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

pub const NT_TILE: usize = 64;
pub const NT_N_CATS: usize = 8;
/// Exact analytical equiprobable octile means of the unit-mean Gamma(alpha=1, beta=1) rate distribution.
pub const NT_CAT_RATES: [f32; NT_N_CATS] = [
    0.06528030, 0.20862730, 0.37607430, 0.57742940, 0.83010100, 1.16989900, 1.69314720,
    3.07944150,
];
pub const NT_N_TGRID: usize = 15;
pub const NT_T_GRID: [f32; NT_N_TGRID] = [
    0.0, 0.00004, 0.00015, 0.0005, 0.0015, 0.004, 0.010, 0.024, 0.055, 0.115, 0.230, 0.450, 0.850,
    1.450, 2.300,
];
pub const NT_T_LEAF_GRID: [f32; NT_N_TGRID] = [
    0.0, 0.00003, 0.00012, 0.0004, 0.0012, 0.0033, 0.0085, 0.020, 0.048, 0.105, 0.220, 0.440,
    0.850, 1.500, 2.500,
];

const NT_SPR_PERMS: [[usize; 5]; 15] = [
    [0, 1, 2, 3, 4],
    [0, 1, 3, 2, 4],
    [0, 1, 4, 2, 3],
    [0, 2, 1, 3, 4],
    [1, 2, 0, 3, 4],
    [0, 3, 1, 2, 4],
    [0, 3, 2, 1, 4],
    [0, 3, 4, 1, 2],
    [0, 4, 1, 2, 3],
    [0, 4, 2, 1, 3],
    [0, 4, 3, 1, 2],
    [1, 3, 0, 2, 4],
    [1, 3, 2, 0, 4],
    [1, 4, 0, 2, 3],
    [1, 4, 2, 0, 3],
];

/// Pure IEEE-754 branchless `ln(x)` for positive `f32` (`x >= 1e-37`), accurate to `< 1 ULP`
/// with zero `libm`/`errno` side effects so LLVM vectorizes 64-site `SoA` loops cleanly.
#[inline(always)]
pub fn ln_f32_noerrno(x: f32) -> f32 {
    let xc = x.max(1e-37_f32);
    let bits = xc.to_bits();
    let raw_exp = ((bits >> 23) as i32) - 127;
    let m0 = f32::from_bits((bits & 0x007f_ffff) | 0x3f80_0000);
    let gt = (m0 > std::f32::consts::SQRT_2) as i32;
    let e = (raw_exp + gt) as f32;
    let m = if gt != 0 { 0.5_f32 * m0 } else { m0 };
    let u = (m - 1.0_f32) / (m + 1.0_f32);
    let u2 = u * u;
    let poly = 2.0_f32
        * u
        * (1.0_f32
            + u2 * (0.333_333_34_f32
                + u2 * (0.2_f32 + u2 * (0.142_857_15_f32 + u2 * 0.111_111_11_f32))));
    e * std::f32::consts::LN_2 + poly
}

#[derive(Clone, Debug)]
pub struct GtrEigen4 {
    pub pi: [f64; 4],
    pub evals: [f64; 4],
    /// `w_mat[r][a] = sqrt(pi[a]) * V[a][r]`, satisfying
    /// `\sum_r exp(evals[r] * t) * w_mat[r][a] * w_mat[r][b] == pi[a] * P(t)[a][b]`.
    pub w_mat: [[f32; 4]; 4],
    pub w_mat_f64: [[f64; 4]; 4],
    /// Instantaneous exit rate `-sum_a f_obs[a] Q[a][a]` under raw empirical base composition
    /// (`1.0` on natural genomes where `f_obs == pi`, and exact mutation-count conversion on skewed synthetic tests).
    pub emp_rate_scale: f64,
}

impl GtrEigen4 {
    /// Transition probability matrix `P(tau)[a][b]` for `a, b in 0..4`.
    #[inline]
    pub fn trans_mat4(&self, tau: f32) -> [[f32; 4]; 4] {
        let tau64 = (tau as f64).max(0.0);
        let mut e_lam = [1.0_f64; 4];
        for r in 1..4 {
            e_lam[r] = (self.evals[r] * tau64).exp();
        }
        let mut out = [[0.0_f32; 4]; 4];
        for a in 0..4 {
            let inv_pia = 1.0 / self.pi[a];
            let mut row_sum = 0.0_f64;
            let mut row64 = [0.0_f64; 4];
            for b in 0..4 {
                let mut s = 0.0_f64;
                for r in 0..4 {
                    s += e_lam[r] * self.w_mat_f64[r][a] * self.w_mat_f64[r][b];
                }
                let p = (s * inv_pia).max(1e-30);
                row64[b] = p;
                row_sum += p;
            }
            let inv_s = 1.0 / row_sum.max(1e-30);
            for b in 0..4 {
                out[a][b] = (row64[b] * inv_s) as f32;
            }
        }
        out
    }

    /// Lookup table `pm[a][mask]` for all 4-bit IUPAC masks `mask in 0..16`:
    /// - `mask == 0` or `mask == 15` (`N` / `-`): `1.0`
    /// - `mask in 1..15`: `\sum_{b: (mask & (1 << b)) != 0} P(tau)[a][b]`
    #[inline]
    pub fn trans_mat_iupac16(&self, tau: f32) -> [[f32; 16]; 4] {
        let p4 = self.trans_mat4(tau);
        let mut out = [[1.0_f32; 16]; 4];
        for a in 0..4 {
            let pa = &p4[a];
            for m in 1..15_usize {
                let mut s = 0.0_f32;
                if (m & 1) != 0 {
                    s += pa[0];
                }
                if (m & 2) != 0 {
                    s += pa[1];
                }
                if (m & 4) != 0 {
                    s += pa[2];
                }
                if (m & 8) != 0 {
                    s += pa[3];
                }
                out[a][m] = s.max(1e-30_f32);
            }
            out[a][0] = 1.0_f32;
            out[a][15] = 1.0_f32;
        }
        out
    }
}

/// Exact cyclic Jacobi eigensolver for a `4x4` real symmetric matrix.
fn jacobi_eigen_4x4(mut a: [[f64; 4]; 4]) -> ([f64; 4], [[f64; 4]; 4]) {
    let mut v = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    for _sweep in 0..32 {
        let mut max_off = 0.0_f64;
        for p in 0..4 {
            for q in (p + 1)..4 {
                let val = a[p][q].abs();
                if val > max_off {
                    max_off = val;
                }
            }
        }
        if max_off < 1e-15 {
            break;
        }
        for p in 0..4 {
            for q in (p + 1)..4 {
                let apq = a[p][q];
                if apq.abs() < 1e-18 {
                    continue;
                }
                let app = a[p][p];
                let aqq = a[q][q];
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                a[p][p] = app - t * apq;
                a[q][q] = aqq + t * apq;
                a[p][q] = 0.0;
                a[q][p] = 0.0;
                for r in 0..4 {
                    if r != p && r != q {
                        let arp = a[r][p];
                        let arq = a[r][q];
                        a[r][p] = c * arp - s * arq;
                        a[p][r] = a[r][p];
                        a[r][q] = s * arp + c * arq;
                        a[q][r] = a[r][q];
                    }
                }
                for r in 0..4 {
                    let vrp = v[r][p];
                    let vrq = v[r][q];
                    v[r][p] = c * vrp - s * vrq;
                    v[r][q] = s * vrp + c * vrq;
                }
            }
        }
    }

    let mut order = [0usize, 1, 2, 3];
    order.sort_by(|&i, &j| {
        a[j][j]
            .partial_cmp(&a[i][i])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| i.cmp(&j))
    });

    let mut evals = [0.0_f64; 4];
    let mut v_sorted = [[0.0_f64; 4]; 4];
    for (new_r, &old_r) in order.iter().enumerate() {
        evals[new_r] = a[old_r][old_r];
        let mut sign = 1.0_f64;
        for row in 0..4 {
            if v[row][old_r].abs() > 1e-8 {
                if v[row][old_r] < 0.0 {
                    sign = -1.0;
                }
                break;
            }
        }
        for row in 0..4 {
            v_sorted[row][new_r] = sign * v[row][old_r];
        }
    }
    evals[0] = 0.0;
    for r in 1..4 {
        evals[r] = evals[r].min(-1e-6);
    }
    (evals, v_sorted)
}

/// Estimate an empirical 4-state reversible `GTR` eigensystem from `mat` (`n x l`).
pub fn estimate_empirical_gtr_eigensystem(mat: &[i8], n: usize, l: usize) -> GtrEigen4 {
    let mut base_cnt = [0_u64; 4];
    for &x in mat {
        if (0..4).contains(&x) {
            base_cnt[x as usize] += 1;
        }
    }
    let tot_bases: f64 = base_cnt.iter().map(|&c| c as f64).sum::<f64>().max(1.0);
    let mut f_obs = [0.25_f64; 4];
    if base_cnt.iter().sum::<u64>() > 0 {
        for a in 0..4 {
            f_obs[a] = (base_cnt[a] as f64) / tot_bases;
        }
    }

    // Regularize equilibrium frequencies into [0.10, 0.45] so reversible rates remain well-conditioned
    // even on short or single-base synthetic test alignments.
    let mut pi = [0.0_f64; 4];
    let mut pi_sum = 0.0_f64;
    for a in 0..4 {
        pi[a] = (((base_cnt[a] as f64) + 8.0) / (tot_bases + 32.0)).clamp(0.10, 0.45);
        pi_sum += pi[a];
    }
    for a in 0..4 {
        pi[a] /= pi_sum;
    }

    // Stratified subsample of up to 64 taxa for O(1)-scaling GTR exchangeability estimation
    let k_sub = n.min(64);
    let sub_idx: Vec<usize> = if n <= 64 {
        (0..n).collect()
    } else {
        (0..k_sub).map(|i| (i * (n - 1)) / (k_sub - 1)).collect()
    };

    // HKY85-like informative Dirichlet prior on symmetric pair counts (transitions: (0,2)=A<->G, (1,3)=C<->T)
    let mut sym_c = [[0.0_f64; 4]; 4];
    for a in 0..4 {
        for b in (a + 1)..4 {
            let is_ti = (a == 0 && b == 2) || (a == 1 && b == 3);
            let prior = (if is_ti { 8.0 } else { 3.5 }) * 2.0 * pi[a] * pi[b];
            sym_c[a][b] = prior;
            sym_c[b][a] = prior;
        }
    }

    if k_sub >= 2 && l > 0 {
        let step_s = (l / 32768).max(1);
        for ia in 0..k_sub {
            let ra = &mat[sub_idx[ia] * l..(sub_idx[ia] + 1) * l];
            for ib in (ia + 1)..k_sub {
                let rb = &mat[sub_idx[ib] * l..(sub_idx[ib] + 1) * l];
                let mut pair_c = [[0u32; 4]; 4];
                let mut n_val = 0u32;
                let mut n_diff = 0u32;
                let mut s = 0usize;
                while s < l {
                    let ba = ra[s];
                    let bb = rb[s];
                    if ba >= 0 && bb >= 0 {
                        n_val += 1;
                        if ba != bb {
                            n_diff += 1;
                            pair_c[ba as usize][bb as usize] += 1;
                        }
                    }
                    s += step_s;
                }
                if n_val >= 8 && n_diff > 0 {
                    let p_diff = (n_diff as f64) / (n_val as f64);
                    // Inverse-divergence weight suppresses saturated deep-pair homoplasy
                    let w = 1.0 / (p_diff + 0.05);
                    for a in 0..4 {
                        for b in (a + 1)..4 {
                            let c_ab = (pair_c[a][b] + pair_c[b][a]) as f64 * w;
                            sym_c[a][b] += c_ab;
                            sym_c[b][a] += c_ab;
                        }
                    }
                }
            }
        }
    }

    let mut q = [[0.0_f64; 4]; 4];
    for a in 0..4 {
        let mut row_s = 0.0_f64;
        for b in 0..4 {
            if a != b {
                let s_ab = sym_c[a][b] / (2.0 * pi[a] * pi[b]);
                q[a][b] = s_ab * pi[b];
                row_s += q[a][b];
            }
        }
        q[a][a] = -row_s;
    }

    let mut mean_rate = 0.0_f64;
    for a in 0..4 {
        mean_rate -= pi[a] * q[a][a];
    }
    let scale = 1.0 / mean_rate.max(1e-12);
    for a in 0..4 {
        for b in 0..4 {
            q[a][b] *= scale;
        }
    }

    let mut emp_rate = 0.0_f64;
    for a in 0..4 {
        emp_rate -= f_obs[a] * q[a][a];
    }
    let emp_rate_scale = emp_rate.clamp(0.40, 2.00);

    // Symmetrize B = Pi^{1/2} Q Pi^{-1/2}
    let mut b_sym = [[0.0_f64; 4]; 4];
    for a in 0..4 {
        b_sym[a][a] = q[a][a];
        for b in (a + 1)..4 {
            let v = 0.5
                * ((pi[a] / pi[b]).sqrt() * q[a][b] + (pi[b] / pi[a]).sqrt() * q[b][a]);
            b_sym[a][b] = v;
            b_sym[b][a] = v;
        }
    }

    let (mut evals, mut v_mat) = jacobi_eigen_4x4(b_sym);
    // Pin exact stationary eigenvector v_0[a] = +sqrt(pi[a]) so at tau = 0, sum_r w[r][a]*w[r][b] == pi[a]*delta_{ab}
    for a in 0..4 {
        v_mat[a][0] = pi[a].sqrt();
    }
    // Re-orthogonalize columns 1..4 against v_0 via Gram-Schmidt for machine-exact stochasticity
    for r in 1..4 {
        for prev in 0..r {
            let mut dot = 0.0_f64;
            for a in 0..4 {
                dot += v_mat[a][r] * v_mat[a][prev];
            }
            for a in 0..4 {
                v_mat[a][r] -= dot * v_mat[a][prev];
            }
        }
        let norm = (0..4).map(|a| v_mat[a][r] * v_mat[a][r]).sum::<f64>().sqrt();
        if norm > 1e-12 {
            for a in 0..4 {
                v_mat[a][r] /= norm;
            }
        }
    }
    evals[0] = 0.0;

    let mut w_mat = [[0.0_f32; 4]; 4];
    let mut w_mat_f64 = [[0.0_f64; 4]; 4];
    for r in 0..4 {
        for a in 0..4 {
            let w = pi[a].sqrt() * v_mat[a][r];
            w_mat_f64[r][a] = w;
            w_mat[r][a] = w as f32;
        }
    }

    GtrEigen4 {
        pi,
        evals,
        w_mat,
        w_mat_f64,
        emp_rate_scale,
    }
}

/// Variance-weighted Gascuel (1997) BioNJ directly on an `m x m` nucleotide distance matrix.
pub fn agglomerate_nt_bionj_from_dist(
    active_ids: &[usize],
    mut dist: Vec<f64>,
    mut out_edge_lens: Option<(&mut HashMap<(usize, usize), f64>, f64)>,
    adj: &mut BTreeMap<usize, Vec<usize>>,
    node_labels: &mut BTreeMap<usize, String>,
    next_node: &mut usize,
) {
    let m = active_ids.len();
    if m <= 1 {
        return;
    }
    if m == 2 {
        let (a, b) = (active_ids[0], active_ids[1]);
        adj.entry(a).or_default().push(b);
        adj.entry(b).or_default().push(a);
        if let Some((ref mut el_map, leff)) = out_edge_lens {
            let d = if dist.len() == 4 { dist[1] } else { 0.02 * leff };
            el_map.insert((a.min(b), a.max(b)), (d / leff.max(1.0)).clamp(5e-5, 2.5));
        }
        return;
    }
    if m == 3 {
        let w = *next_node;
        *next_node += 1;
        node_labels.insert(w, format!("N{w}"));
        adj.insert(w, vec![active_ids[0], active_ids[1], active_ids[2]]);
        for &x in active_ids {
            adj.entry(x).or_default().push(w);
        }
        if let Some((ref mut el_map, leff)) = out_edge_lens {
            if dist.len() == 9 {
                let (d01, d02, d12) = (dist[1], dist[2], dist[5]);
                let ls = [
                    0.5 * (d01 + d02 - d12),
                    0.5 * (d01 + d12 - d02),
                    0.5 * (d02 + d12 - d01),
                ];
                for (idx, &x) in active_ids.iter().enumerate() {
                    el_map.insert(
                        (x.min(w), x.max(w)),
                        (ls[idx] / leff.max(1.0)).clamp(5e-5, 2.5),
                    );
                }
            }
        }
        return;
    }

    let mut vmat = dist.clone();
    let mut cur_ids: Vec<usize> = active_ids.to_vec();
    let mut alive = vec![true; m];
    let mut r_sum = vec![0.0_f64; m];
    for i in 0..m {
        r_sum[i] = (0..m).map(|j| dist[i * m + j]).sum();
    }
    let mut n_rem = m;

    while n_rem > 3 {
        let denom = (n_rem - 2) as f64;
        #[cfg(not(target_arch = "wasm32"))]
        let (best_q, best_i, best_j) = if n_rem >= 384 {
            (0..m)
                .into_par_iter()
                .filter(|&i| alive[i])
                .map(|i| {
                    let ri = r_sum[i];
                    let row_off = i * m;
                    let mut loc_q = f64::INFINITY;
                    let mut loc_j = i;
                    for j in (i + 1)..m {
                        if alive[j] {
                            let q = denom * dist[row_off + j] - ri - r_sum[j];
                            if q < loc_q {
                                loc_q = q;
                                loc_j = j;
                            }
                        }
                    }
                    (loc_q, i, loc_j)
                })
                .filter(|&(_, i, j)| j > i)
                .reduce(
                    || (f64::INFINITY, 0usize, 1usize),
                    |a, b| {
                        if b.0 < a.0 || (b.0 == a.0 && (b.1, b.2) < (a.1, a.2)) {
                            b
                        } else {
                            a
                        }
                    },
                )
        } else {
            let mut bq = f64::INFINITY;
            let mut bi = 0usize;
            let mut bj = 1usize;
            for i in 0..m {
                if !alive[i] {
                    continue;
                }
                let ri = r_sum[i];
                for j in (i + 1)..m {
                    if alive[j] {
                        let q = denom * dist[i * m + j] - ri - r_sum[j];
                        if q < bq {
                            bq = q;
                            bi = i;
                            bj = j;
                        }
                    }
                }
            }
            (bq, bi, bj)
        };

        #[cfg(target_arch = "wasm32")]
        let (best_q, best_i, best_j) = {
            let mut bq = f64::INFINITY;
            let mut bi = 0usize;
            let mut bj = 1usize;
            for i in 0..m {
                if !alive[i] {
                    continue;
                }
                let ri = r_sum[i];
                for j in (i + 1)..m {
                    if alive[j] {
                        let q = denom * dist[i * m + j] - ri - r_sum[j];
                        if q < bq {
                            bq = q;
                            bi = i;
                            bj = j;
                        }
                    }
                }
            }
            (bq, bi, bj)
        };
        let _ = best_q;

        let dij = dist[best_i * m + best_j].max(1e-6);
        let u_node = cur_ids[best_i];
        let v_node = cur_ids[best_j];
        let w_node = *next_node;
        *next_node += 1;
        node_labels.insert(w_node, format!("N{w_node}"));
        adj.insert(w_node, vec![u_node, v_node]);
        adj.entry(u_node).or_default().push(w_node);
        adj.entry(v_node).or_default().push(w_node);

        let li_raw = 0.5 * dij + (r_sum[best_i] - r_sum[best_j]) / (2.0 * denom);
        let li = li_raw.max(0.0);
        let lj = (dij - li).max(0.0);
        if let Some((ref mut el_map, leff)) = out_edge_lens {
            el_map.insert(
                (u_node.min(w_node), u_node.max(w_node)),
                (li / leff.max(1.0)).clamp(5e-5, 2.5),
            );
            el_map.insert(
                (v_node.min(w_node), v_node.max(w_node)),
                (lj / leff.max(1.0)).clamp(5e-5, 2.5),
            );
        }
        let vij = vmat[best_i * m + best_j];
        let lambda = if vij > 1e-12 {
            let mut v_diff = 0.0_f64;
            for k in 0..m {
                if alive[k] && k != best_i && k != best_j {
                    v_diff += vmat[best_j * m + k] - vmat[best_i * m + k];
                }
            }
            (0.5 + v_diff / (2.0 * denom * vij)).clamp(0.0, 1.0)
        } else {
            0.5
        };

        let mut new_r_i = 0.0_f64;
        for k in 0..m {
            if alive[k] && k != best_i && k != best_j {
                let d_ik = dist[best_i * m + k];
                let d_jk = dist[best_j * m + k];
                let d_wk = (lambda * (d_ik - li) + (1.0 - lambda) * (d_jk - lj)).max(0.0);
                let v_wk = (lambda * vmat[best_i * m + k]
                    + (1.0 - lambda) * vmat[best_j * m + k]
                    - lambda * (1.0 - lambda) * vij)
                    .max(0.0);
                r_sum[k] = r_sum[k] - d_ik - d_jk + d_wk;
                dist[best_i * m + k] = d_wk;
                dist[k * m + best_i] = d_wk;
                vmat[best_i * m + k] = v_wk;
                vmat[k * m + best_i] = v_wk;
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
        let (i0, i1, i2) = (rem_idx[0], rem_idx[1], rem_idx[2]);
        let (u0, u1, u2) = (cur_ids[i0], cur_ids[i1], cur_ids[i2]);
        adj.insert(w, vec![u0, u1, u2]);
        for &x in &[u0, u1, u2] {
            adj.entry(x).or_default().push(w);
        }
        if let Some((ref mut el_map, leff)) = out_edge_lens {
            let (d01, d02, d12) = (dist[i0 * m + i1], dist[i0 * m + i2], dist[i1 * m + i2]);
            let ls = [
                0.5 * (d01 + d02 - d12),
                0.5 * (d01 + d12 - d02),
                0.5 * (d02 + d12 - d01),
            ];
            for (&ux, &lx) in [u0, u1, u2].iter().zip(&ls) {
                el_map.insert(
                    (ux.min(w), ux.max(w)),
                    (lx / leff.max(1.0)).clamp(5e-5, 2.5),
                );
            }
        }
    }
}

#[derive(Clone, Copy)]
struct NtUpLeafOp {
    dst: usize,
    e_idx: usize,
    leaf: usize,
}

#[derive(Clone, Copy)]
struct NtMsgIntOp {
    dst: usize,
    c0: usize,
    c1: usize,
    e_idx: usize,
}

#[derive(Clone, Copy)]
struct NtNniTask {
    a0: usize,
    b0: usize,
    c0: usize,
    d0: usize,
    eval_swaps: bool,
}

#[derive(Clone, Copy)]
struct NtLeafTask {
    leaf: usize,
    o0: usize,
    o1: usize,
}

#[derive(Clone, Copy)]
struct NtSprTask {
    m_slots: [usize; 5],
    e_left: usize,
    e_right: usize,
}

struct NtGtrTile {
    cat: usize,
    weights: [f32; NT_TILE],
    /// 4-bit IUPAC masks (`1..15`) for each leaf across the 64 sites in this tile.
    leaf_masks: Vec<[u8; NT_TILE]>,
}

#[inline(always)]
fn nt_mul_norm4_soa(
    m1: &[[f32; NT_TILE]; 4],
    m2: &[[f32; NT_TILE]; 4],
    out_norm: &mut [[f32; NT_TILE]; 4],
    out_lsc: &mut [f32; NT_TILE],
) {
    let mut mx = [1e-37_f32; NT_TILE];
    for a in 0..4 {
        let r1 = &m1[a];
        let r2 = &m2[a];
        let ro = &mut out_norm[a];
        for s in 0..NT_TILE {
            let v = r1[s] * r2[s];
            ro[s] = v;
            if v > mx[s] {
                mx[s] = v;
            }
        }
    }
    let mut inv_mx = [0.0_f32; NT_TILE];
    for s in 0..NT_TILE {
        inv_mx[s] = 1.0_f32 / mx[s];
        out_lsc[s] = ln_f32_noerrno(mx[s]);
    }
    for a in 0..4 {
        let ro = &mut out_norm[a];
        for s in 0..NT_TILE {
            ro[s] *= inv_mx[s];
        }
    }
}

#[inline(always)]
fn nt_matmul4_soa(
    mat: &[[f32; 4]; 4],
    inp: &[[f32; NT_TILE]; 4],
    out: &mut [[f32; NT_TILE]; 4],
) {
    let (in0, in1, in2, in3) = (&inp[0], &inp[1], &inp[2], &inp[3]);
    for a in 0..4 {
        let m = &mat[a];
        let (m0, m1, m2, m3) = (m[0], m[1], m[2], m[3]);
        let oa = &mut out[a];
        for s in 0..NT_TILE {
            oa[s] = m0 * in0[s] + m1 * in1[s] + m2 * in2[s] + m3 * in3[s];
        }
    }
}

fn nt_compile_tree_ops(
    adj: &BTreeMap<usize, Vec<usize>>,
    edge_lens: &HashMap<(usize, usize), f64>,
    n: usize,
) -> (
    usize,
    Vec<f32>,
    HashMap<(usize, usize), usize>,
    HashMap<(usize, usize), usize>,
    Vec<NtUpLeafOp>,
    Vec<NtMsgIntOp>,
    Vec<NtMsgIntOp>,
) {
    let root = adj
        .iter()
        .find(|(&u, nbs)| u >= n && nbs.len() == 3)
        .map(|(&u, _)| u)
        .unwrap_or(n);
    let mut order = vec![root];
    let mut parent: HashMap<usize, usize> = HashMap::with_capacity(2 * n);
    let mut head = 0usize;
    while head < order.len() {
        let u = order[head];
        head += 1;
        if let Some(nbs) = adj.get(&u) {
            for &nb in nbs {
                if nb != root && !parent.contains_key(&nb) {
                    parent.insert(nb, u);
                    order.push(nb);
                }
            }
        }
    }

    let mut undir_idx: HashMap<(usize, usize), usize> = HashMap::with_capacity(2 * n);
    let mut undir_lens: Vec<f32> = Vec::with_capacity(2 * n);
    for (&u, nbs) in adj {
        for &v in nbs {
            if u < v {
                let idx = undir_lens.len();
                undir_idx.insert((u, v), idx);
                let t = edge_lens.get(&(u, v)).copied().unwrap_or(0.02) as f32;
                undir_lens.push(t.clamp(3e-5_f32, 2.5_f32));
            }
        }
    }

    let mut dir_slot: HashMap<(usize, usize), usize> = HashMap::with_capacity(3 * n);
    let mut n_slots = 0usize;
    for (&v, nbs) in adj {
        if v >= n {
            for &u in nbs {
                dir_slot.insert((u, v), n_slots);
                n_slots += 1;
            }
        }
    }

    let mut up_leaf = Vec::with_capacity(n);
    let mut up_int = Vec::with_capacity(n);
    for &u in order[1..].iter().rev() {
        let p = parent[&u];
        let dst = dir_slot[&(u, p)];
        let e_idx = undir_idx[&(u.min(p), u.max(p))];
        if u < n {
            up_leaf.push(NtUpLeafOp { dst, e_idx, leaf: u });
        } else {
            let mut chs = adj[&u].iter().copied().filter(|&nb| nb != p);
            let ch0 = chs.next().unwrap();
            let ch1 = chs.next().unwrap();
            up_int.push(NtMsgIntOp {
                dst,
                c0: dir_slot[&(ch0, u)],
                c1: dir_slot[&(ch1, u)],
                e_idx,
            });
        }
    }

    let mut down_int = Vec::with_capacity(2 * n);
    for &u in &order {
        if u < n {
            continue;
        }
        let p_opt = parent.get(&u).copied();
        let nbs = &adj[&u];
        for &ch in nbs {
            if Some(ch) == p_opt || ch < n {
                continue;
            }
            let mut oths = nbs.iter().copied().filter(|&nb| nb != ch);
            let o0 = oths.next().unwrap();
            let o1 = oths.next().unwrap();
            down_int.push(NtMsgIntOp {
                dst: dir_slot[&(u, ch)],
                c0: dir_slot[&(o0, u)],
                c1: dir_slot[&(o1, u)],
                e_idx: undir_idx[&(u.min(ch), u.max(ch))],
            });
        }
    }

    (n_slots, undir_lens, undir_idx, dir_slot, up_leaf, up_int, down_int)
}

#[inline]
fn nt_populate_tile_messages(
    tile: &NtGtrTile,
    p_iupac_cat: &[[[f32; 16]; 4]],
    p_int_cat: &[[[f32; 4]; 4]],
    up_leaf: &[NtUpLeafOp],
    up_int: &[NtMsgIntOp],
    down_int: &[NtMsgIntOp],
    msg_buf: &mut [[[f32; NT_TILE]; 4]],
    lsc_buf: &mut [[f32; NT_TILE]],
) {
    let mut vn = [[0.0_f32; NT_TILE]; 4];
    let mut mx_ln = [0.0_f32; NT_TILE];

    for op in up_leaf {
        let pm = &p_iupac_cat[op.e_idx];
        let st = &tile.leaf_masks[op.leaf];
        let out_m = &mut msg_buf[op.dst];
        for a in 0..4 {
            let pa = &pm[a];
            let oa = &mut out_m[a];
            for s in 0..NT_TILE {
                oa[s] = pa[(st[s] & 15) as usize];
            }
        }
        lsc_buf[op.dst] = [0.0_f32; NT_TILE];
    }

    for op in up_int {
        nt_mul_norm4_soa(&msg_buf[op.c0], &msg_buf[op.c1], &mut vn, &mut mx_ln);
        let l0 = lsc_buf[op.c0];
        let l1 = lsc_buf[op.c1];
        let lo = &mut lsc_buf[op.dst];
        for s in 0..NT_TILE {
            lo[s] = l0[s] + l1[s] + mx_ln[s];
        }
        nt_matmul4_soa(&p_int_cat[op.e_idx], &vn, &mut msg_buf[op.dst]);
    }

    for op in down_int {
        nt_mul_norm4_soa(&msg_buf[op.c0], &msg_buf[op.c1], &mut vn, &mut mx_ln);
        let l0 = lsc_buf[op.c0];
        let l1 = lsc_buf[op.c1];
        let lo = &mut lsc_buf[op.dst];
        for s in 0..NT_TILE {
            lo[s] = l0[s] + l1[s] + mx_ln[s];
        }
        nt_matmul4_soa(&p_int_cat[op.e_idx], &vn, &mut msg_buf[op.dst]);
    }
}

/// Parabolic log-space sub-grid branch-length, peak log-likelihood, and local Fisher curvature penalty.
#[inline]
fn refine_grid_peak_ln_t_with_curv(
    ll_slice: &[f64],
    grid: &[f32],
    min_bl: f64,
) -> (f64, f64, f64) {
    let mut kb = 0usize;
    for k in 1..NT_N_TGRID {
        if ll_slice[k] > ll_slice[kb] {
            kb = k;
        }
    }
    let mut best_ll = ll_slice[kb];
    let mut best_t = (grid[kb] as f64).max(min_bl);
    let mut fisher_pen = 0.0_f64;

    if kb >= 2 && kb + 1 < NT_N_TGRID {
        let d_minus = ll_slice[kb] - ll_slice[kb - 1];
        let d_plus = ll_slice[kb] - ll_slice[kb + 1];
        let denom = d_minus + d_plus;
        if d_minus >= 0.0 && d_plus >= 0.0 && denom > 1e-9 {
            let delta = (0.5 * (d_minus - d_plus) / denom).clamp(-0.45, 0.45);
            let x_m = (grid[kb - 1] as f64).ln();
            let x_0 = (grid[kb] as f64).ln();
            let x_p = (grid[kb + 1] as f64).ln();
            let step = 0.5 * (x_p - x_m);
            best_t = (x_0 + delta * step).exp().max(min_bl);
            best_ll += 0.25 * delta * (d_minus - d_plus);
            let curv_ln_t = (denom / (step * step)).max(0.0);
            fisher_pen = 0.5 * (1.0 + curv_ln_t / std::f64::consts::TAU).ln();
        }
    }
    let dll = (best_ll - ll_slice[0]).max(0.0);
    (dll, best_t, fisher_pen)
}

#[inline]
fn refine_grid_peak_ln_t(ll_slice: &[f64], grid: &[f32], min_bl: f64) -> (f64, f64) {
    let (dll, best_t, _) = refine_grid_peak_ln_t_with_curv(ll_slice, grid, min_bl);
    (dll, best_t)
}

/// Site-tiled 4-state `GTR+CAT8` Eigen-Felsenstein-Rissanen MDL topology refinement (`NNI + LeafMLE + SPR2`)
/// and branch-support / MLE branch-length scorer.
/// Returns `(HashMap<(u_min, v_max), (delta_ll_vs_zero, t_mle)>, emp_rate_scale)`.
pub fn refine_and_score_nt_gtr_mdl(
    adj: &mut BTreeMap<usize, Vec<usize>>,
    edge_lens: &mut HashMap<(usize, usize), f64>,
    mat: &[i8],
    bitmasks: &[u8],
    n: usize,
    l: usize,
    l_eff: f64,
) -> (HashMap<(usize, usize), (f64, f64)>, f64) {
    let gtr = estimate_empirical_gtr_eigensystem(mat, n, l);
    let min_bl = (0.20 / l_eff.max(1.0)).clamp(1e-6, 1e-3);
    if n < 4 || l == 0 {
        let map = edge_lens
            .iter()
            .map(|(&k, &t)| (k, ((t * l_eff * 2.0).max(0.0), t.max(min_bl))))
            .collect();
        return (map, gtr.emp_rate_scale);
    }

    for (&u, nbs) in adj.iter() {
        for &v in nbs {
            if u < v {
                edge_lens.entry((u, v)).or_insert(0.02);
            }
        }
    }

    // 1. Preorder traversal on initial BioNJ tree for parallel Fitch step counts
    let root0 = adj
        .iter()
        .find(|(&u, nbs)| u >= n && nbs.len() == 3)
        .map(|(&u, _)| u)
        .unwrap_or(n);
    let mut order0 = vec![root0];
    let mut parent0: HashMap<usize, usize> = HashMap::with_capacity(2 * n);
    let mut h0 = 0usize;
    while h0 < order0.len() {
        let u = order0[h0];
        h0 += 1;
        if let Some(nbs) = adj.get(&u) {
            for &nb in nbs {
                if nb != root0 && !parent0.contains_key(&nb) {
                    parent0.insert(nb, u);
                    order0.push(nb);
                }
            }
        }
    }

    let post_int: Vec<(usize, Vec<usize>)> = order0
        .iter()
        .rev()
        .copied()
        .filter(|&u| u >= n)
        .map(|u| {
            let p_opt = parent0.get(&u).copied();
            let chs: Vec<usize> = adj[&u]
                .iter()
                .copied()
                .filter(|&nb| Some(nb) != p_opt)
                .collect();
            (u, chs)
        })
        .collect();

    // Parallel block-wise computation of `n_val` and Fitch `steps` across 512-column blocks
    let block_w = 512usize;
    let n_blocks = (l + block_w - 1) / block_w;
    let eval_fitch_block = |b_idx: usize| -> Vec<(usize, u32, u32)> {
        let col_s = b_idx * block_w;
        let col_e = (col_s + block_w).min(l);
        let cw = col_e - col_s;
        let mut nv = vec![0_u32; cw];
        for i in 0..n {
            let row = &bitmasks[i * l + col_s..i * l + col_e];
            for k in 0..cw {
                if row[k] != 15 && row[k] != 0 {
                    nv[k] += 1;
                }
            }
        }
        let active_local: Vec<usize> = (0..cw).filter(|&k| nv[k] >= 2).collect();
        if active_local.is_empty() {
            return Vec::new();
        }
        let aw = active_local.len();
        let mut st_cnt = vec![0_u32; aw];
        let mut fmask: Vec<Vec<u8>> = vec![Vec::new(); 2 * n + 4];
        for i in 0..n {
            let row = &bitmasks[i * l + col_s..i * l + col_e];
            let mut v = Vec::with_capacity(aw);
            for &k in &active_local {
                let m = row[k] & 15;
                v.push(if m == 0 { 15 } else { m });
            }
            fmask[i] = v;
        }
        for (u, chs) in &post_int {
            if chs.is_empty() {
                continue;
            }
            let mut cur = std::mem::take(&mut fmask[chs[0]]);
            for &cn in &chs[1..] {
                let mc = std::mem::take(&mut fmask[cn]);
                for idx in 0..aw {
                    let inter = cur[idx] & mc[idx];
                    if inter != 0 {
                        cur[idx] = inter;
                    } else {
                        if cur[idx] != 15 && mc[idx] != 15 {
                            st_cnt[idx] += 1;
                        }
                        cur[idx] |= mc[idx];
                    }
                }
            }
            if *u >= fmask.len() {
                fmask.resize(*u + 1, Vec::new());
            }
            fmask[*u] = cur;
        }
        active_local
            .into_iter()
            .enumerate()
            .map(|(idx, k)| (col_s + k, nv[k], st_cnt[idx]))
            .collect()
    };

    #[cfg(not(target_arch = "wasm32"))]
    let block_res: Vec<Vec<(usize, u32, u32)>> =
        (0..n_blocks).into_par_iter().map(eval_fitch_block).collect();
    #[cfg(target_arch = "wasm32")]
    let block_res: Vec<Vec<(usize, u32, u32)>> =
        (0..n_blocks).map(eval_fitch_block).collect();

    let mut active_meta: Vec<(usize, u32, u32)> = Vec::with_capacity(l);
    for br in block_res {
        active_meta.extend(br);
    }
    if active_meta.is_empty() {
        let mut out = HashMap::new();
        for (&u, nbs) in adj.iter() {
            for &v in nbs {
                if u < v {
                    out.insert((u, v), (0.0, min_bl));
                }
            }
        }
        return (out, gtr.emp_rate_scale);
    }

    let mut sum_steps = 0.0_f64;
    let mut sum_nv = 0.0_f64;
    for &(_, nv, st) in &active_meta {
        sum_steps += st as f64;
        sum_nv += nv as f64;
    }
    let steps_per_nv = (sum_steps / sum_nv.max(1.0)).max(1e-6);
    let ln_cats: [f64; NT_N_CATS] = std::array::from_fn(|c| (NT_CAT_RATES[c] as f64).ln());

    // Parallel block-transposed extraction of `(cat, col_pattern)`
    let extract_pat_chunk = |meta_chunk: &[(usize, u32, u32)]| -> Vec<(u8, Vec<u8>)> {
        let cw = meta_chunk.len();
        let mut pats: Vec<Vec<u8>> = (0..cw).map(|_| vec![15_u8; n]).collect();
        for i in 0..n {
            let row = &bitmasks[i * l..(i + 1) * l];
            for (k, &(col, _, _)) in meta_chunk.iter().enumerate() {
                let m = row[col] & 15;
                pats[k][i] = if m == 0 { 15 } else { m };
            }
        }
        meta_chunk
            .iter()
            .zip(pats)
            .map(|(&(_, nv, st), pat)| {
                let exp_steps = steps_per_nv * (nv as f64);
                let rel = ((1.5 + st as f64) / (1.5 + exp_steps)).clamp(0.065, 3.10);
                let ln_rel = rel.ln();
                let mut best_c = 0usize;
                let mut best_d = (ln_rel - ln_cats[0]).abs();
                for c in 1..NT_N_CATS {
                    let d = (ln_rel - ln_cats[c]).abs();
                    if d < best_d {
                        best_d = d;
                        best_c = c;
                    }
                }
                (best_c as u8, pat)
            })
            .collect()
    };

    #[cfg(not(target_arch = "wasm32"))]
    let pat_batches: Vec<Vec<(u8, Vec<u8>)>> = active_meta
        .par_chunks(256)
        .map(extract_pat_chunk)
        .collect();
    #[cfg(target_arch = "wasm32")]
    let pat_batches: Vec<Vec<(u8, Vec<u8>)>> =
        active_meta.chunks(256).map(extract_pat_chunk).collect();
    active_meta.clear();

    let mut cat_vecs: Vec<Vec<Vec<u8>>> = (0..NT_N_CATS).map(|_| Vec::new()).collect();
    for batch in pat_batches {
        for (c, pat) in batch {
            cat_vecs[c as usize].push(pat);
        }
    }

    // Parallel sort + exact RLE deduplication within each category (100% deterministic order)
    #[cfg(not(target_arch = "wasm32"))]
    let dedup_cats: Vec<(usize, Vec<(Vec<u8>, u32)>)> = cat_vecs
        .into_par_iter()
        .enumerate()
        .map(|(c, mut v)| {
            if v.is_empty() {
                return (c, Vec::new());
            }
            v.par_sort_unstable();
            let mut entries: Vec<(Vec<u8>, u32)> = Vec::with_capacity(v.len());
            for pat in v {
                if let Some(last) = entries.last_mut() {
                    if last.0 == pat {
                        last.1 += 1;
                        continue;
                    }
                }
                entries.push((pat, 1));
            }
            (c, entries)
        })
        .collect();

    #[cfg(target_arch = "wasm32")]
    let dedup_cats: Vec<(usize, Vec<(Vec<u8>, u32)>)> = cat_vecs
        .into_iter()
        .enumerate()
        .map(|(c, mut v)| {
            if v.is_empty() {
                return (c, Vec::new());
            }
            v.sort_unstable();
            let mut entries: Vec<(Vec<u8>, u32)> = Vec::with_capacity(v.len());
            for pat in v {
                if let Some(last) = entries.last_mut() {
                    if last.0 == pat {
                        last.1 += 1;
                        continue;
                    }
                }
                entries.push((pat, 1));
            }
            (c, entries)
        })
        .collect();

    let mut tiles: Vec<NtGtrTile> = Vec::new();
    for (c, entries) in dedup_cats {
        if entries.is_empty() {
            continue;
        }
        for chunk in entries.chunks(NT_TILE) {
            let mut weights = [0.0_f32; NT_TILE];
            let mut leaf_masks = vec![[15_u8; NT_TILE]; n];
            for (s, (pat, mult)) in chunk.iter().enumerate() {
                weights[s] = *mult as f32;
                for i in 0..n {
                    leaf_masks[i][s] = pat[i];
                }
            }
            tiles.push(NtGtrTile {
                cat: c,
                weights,
                leaf_masks,
            });
        }
    }

    // Precompute eigen-grid tables and leaf IUPAC lookup tables in f32
    let w_mat = gtr.w_mat;
    let mut eg_cat_grid = [[[0.0_f32; 4]; NT_N_TGRID]; NT_N_CATS];
    let mut pb_leaf_cat_grid = [[[[0.0_f32; 4]; 16]; NT_N_TGRID]; NT_N_CATS];
    for c in 0..NT_N_CATS {
        let cr = NT_CAT_RATES[c];
        for k in 0..NT_N_TGRID {
            let tau = (cr * NT_T_GRID[k]) as f64;
            eg_cat_grid[c][k][0] = 1.0_f32;
            for r in 1..4 {
                eg_cat_grid[c][k][r] = (gtr.evals[r] * tau).exp() as f32;
            }
            let pm16 = gtr.trans_mat_iupac16(cr * NT_T_LEAF_GRID[k]);
            for m in 0..16 {
                for a in 0..4 {
                    pb_leaf_cat_grid[c][k][m][a] = (gtr.pi[a] as f32) * pm16[a][m];
                }
            }
        }
    }

    let build_cat_pmats =
        |undir_lens: &[f32]| -> (Vec<Vec<[[f32; 16]; 4]>>, Vec<Vec<[[f32; 4]; 4]>>) {
            let mut iupac_cats = Vec::with_capacity(NT_N_CATS);
            let mut int_cats = Vec::with_capacity(NT_N_CATS);
            for &cr in &NT_CAT_RATES {
                let mut iu = Vec::with_capacity(undir_lens.len());
                let mut im = Vec::with_capacity(undir_lens.len());
                for &t in undir_lens {
                    let tau = cr * t;
                    iu.push(gtr.trans_mat_iupac16(tau));
                    im.push(gtr.trans_mat4(tau));
                }
                iupac_cats.push(iu);
                int_cats.push(im);
            }
            (iupac_cats, int_cats)
        };

    let run_nni_pass = |adj: &mut BTreeMap<usize, Vec<usize>>,
                        edge_lens: &mut HashMap<(usize, usize), f64>,
                        prev_sup: &HashMap<(usize, usize), f64>,
                        touched: &HashSet<usize>,
                        apply_swaps: bool|
     -> (usize, HashMap<(usize, usize), (f64, f64)>, HashSet<usize>) {
        let (n_slots, undir_lens, _undir_idx, dir_slot, up_leaf, up_int, down_int) =
            nt_compile_tree_ops(adj, edge_lens, n);
        let mut int_edges = Vec::new();
        for (&u, nbs) in adj.iter() {
            if u >= n && nbs.len() == 3 {
                for &v in nbs {
                    if u < v && v >= n && adj.get(&v).map(|x| x.len()).unwrap_or(0) == 3 {
                        int_edges.push((u, v));
                    }
                }
            }
        }
        if int_edges.is_empty() {
            return (0, HashMap::new(), HashSet::new());
        }

        let nni_tasks: Vec<NtNniTask> = int_edges
            .iter()
            .map(|&(u, v)| {
                let mut nu = adj[&u].iter().copied().filter(|&x| x != v);
                let mut nv = adj[&v].iter().copied().filter(|&x| x != u);
                let a = nu.next().unwrap();
                let b = nu.next().unwrap();
                let c = nv.next().unwrap();
                let d = nv.next().unwrap();
                let eval_swaps = apply_swaps
                    && (prev_sup.is_empty()
                        || prev_sup.get(&(u, v)).copied().unwrap_or(0.0) < 30.0
                        || touched.contains(&u)
                        || touched.contains(&v)
                        || touched.contains(&a)
                        || touched.contains(&b)
                        || touched.contains(&c)
                        || touched.contains(&d));
                NtNniTask {
                    a0: dir_slot[&(a, u)],
                    b0: dir_slot[&(b, u)],
                    c0: dir_slot[&(c, v)],
                    d0: dir_slot[&(d, v)],
                    eval_swaps,
                }
            })
            .collect();

        let (p_iupac, p_int) = build_cat_pmats(&undir_lens);
        let n_ie = int_edges.len();
        let stride = 3 * NT_N_TGRID;

        let eval_chunk = |chunk: &[NtGtrTile]| -> Vec<f64> {
            let mut acc = vec![0.0_f64; n_ie * stride];
            let mut msg_buf = vec![[[0.0_f32; NT_TILE]; 4]; n_slots];
            let mut lsc_buf = vec![[0.0_f32; NT_TILE]; n_slots];
            let mut u_n = [[0.0_f32; NT_TILE]; 4];
            let mut v_n = [[0.0_f32; NT_TILE]; 4];
            let mut lsc_u = [0.0_f32; NT_TILE];
            let mut lsc_v = [0.0_f32; NT_TILE];
            let mut wu = [[0.0_f32; NT_TILE]; 4];
            let mut wv = [[0.0_f32; NT_TILE]; 4];
            let mut prod = [[0.0_f32; NT_TILE]; 4];

            for tile in chunk {
                nt_populate_tile_messages(
                    tile,
                    &p_iupac[tile.cat],
                    &p_int[tile.cat],
                    &up_leaf,
                    &up_int,
                    &down_int,
                    &mut msg_buf,
                    &mut lsc_buf,
                );
                let eg_grid = &eg_cat_grid[tile.cat];
                let wts = &tile.weights;

                for (ei, task) in nni_tasks.iter().enumerate() {
                    let la = &lsc_buf[task.a0];
                    let lb = &lsc_buf[task.b0];
                    let lc = &lsc_buf[task.c0];
                    let ld = &lsc_buf[task.d0];
                    let mut bsc = 0.0_f64;
                    for s in 0..NT_TILE {
                        bsc += (wts[s] as f64) * ((la[s] + lb[s] + lc[s] + ld[s]) as f64);
                    }
                    let n_topos = if task.eval_swaps { 3 } else { 1 };
                    let pairs = [
                        (task.a0, task.b0, task.c0, task.d0),
                        (task.a0, task.c0, task.b0, task.d0),
                        (task.a0, task.d0, task.c0, task.b0),
                    ];
                    for ti in 0..n_topos {
                        let (i_l1, i_l2, i_r1, i_r2) = pairs[ti];
                        nt_mul_norm4_soa(&msg_buf[i_l1], &msg_buf[i_l2], &mut u_n, &mut lsc_u);
                        nt_mul_norm4_soa(&msg_buf[i_r1], &msg_buf[i_r2], &mut v_n, &mut lsc_v);
                        let mut sc_uv = bsc;
                        for s in 0..NT_TILE {
                            sc_uv += (wts[s] as f64) * ((lsc_u[s] + lsc_v[s]) as f64);
                        }
                        nt_matmul4_soa(&w_mat, &u_n, &mut wu);
                        nt_matmul4_soa(&w_mat, &v_n, &mut wv);
                        for r in 0..4 {
                            for s in 0..NT_TILE {
                                prod[r][s] = wu[r][s] * wv[r][s];
                            }
                        }
                        let (pr0, pr1, pr2, pr3) = (&prod[0], &prod[1], &prod[2], &prod[3]);
                        let base_off = ei * stride + ti * NT_N_TGRID;
                        for k in 0..NT_N_TGRID {
                            let eg = &eg_grid[k];
                            let (e0, e1, e2, e3) = (eg[0], eg[1], eg[2], eg[3]);
                            let mut sum_ln = 0.0_f64;
                            for s in 0..NT_TILE {
                                let ps =
                                    e0 * pr0[s] + e1 * pr1[s] + e2 * pr2[s] + e3 * pr3[s];
                                sum_ln += (wts[s] as f64) * (ln_f32_noerrno(ps) as f64);
                            }
                            acc[base_off + k] += sc_uv + sum_ln;
                        }
                    }
                }
            }
            acc
        };

        let chunk_size = (tiles.len() / 64).clamp(1, 16);
        #[cfg(not(target_arch = "wasm32"))]
        let parts: Vec<Vec<f64>> = tiles.par_chunks(chunk_size).map(eval_chunk).collect();
        #[cfg(target_arch = "wasm32")]
        let parts: Vec<Vec<f64>> = tiles.chunks(chunk_size).map(eval_chunk).collect();

        let mut total_ll = vec![0.0_f64; n_ie * stride];
        for p in parts {
            for idx in 0..total_ll.len() {
                total_ll[idx] += p[idx];
            }
        }

        let mut stats: HashMap<(usize, usize), (f64, f64)> = HashMap::with_capacity(n_ie);
        let mut moves: Vec<(f64, usize, usize, usize, f64)> = Vec::new();
        for (ei, &(u, v)) in int_edges.iter().enumerate() {
            let off0 = ei * stride;
            let (dll0, tmle0, pen0) = refine_grid_peak_ln_t_with_curv(
                &total_ll[off0..off0 + NT_N_TGRID],
                &NT_T_GRID,
                min_bl,
            );
            let mdl0 = total_ll[off0] + dll0 - pen0;
            stats.insert((u, v), (dll0, tmle0));
            edge_lens.insert((u, v), tmle0.max(3e-5));

            if nni_tasks[ei].eval_swaps {
                let off1 = off0 + NT_N_TGRID;
                let off2 = off0 + 2 * NT_N_TGRID;
                let (dll1, tmle1, pen1) = refine_grid_peak_ln_t_with_curv(
                    &total_ll[off1..off1 + NT_N_TGRID],
                    &NT_T_GRID,
                    min_bl,
                );
                let (dll2, tmle2, pen2) = refine_grid_peak_ln_t_with_curv(
                    &total_ll[off2..off2 + NT_N_TGRID],
                    &NT_T_GRID,
                    min_bl,
                );
                let mdl1 = total_ll[off1] + dll1 - pen1;
                let mdl2 = total_ll[off2] + dll2 - pen2;
                if mdl1 >= mdl2 && mdl1 > mdl0 + 1e-2 {
                    moves.push((mdl1 - mdl0, u, v, 1, tmle1.max(3e-5)));
                } else if mdl2 > mdl1 && mdl2 > mdl0 + 1e-2 {
                    moves.push((mdl2 - mdl0, u, v, 2, tmle2.max(3e-5)));
                }
            }
        }

        if !apply_swaps || moves.is_empty() {
            return (0, stats, HashSet::new());
        }

        moves.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
                .then_with(|| a.2.cmp(&b.2))
        });
        let mut locked: HashSet<usize> = HashSet::new();
        let mut n_sw = 0usize;
        for (_gain, u, v, topo, nt) in moves {
            let mut nu = adj[&u].iter().copied().filter(|&x| x != v);
            let mut nv = adj[&v].iter().copied().filter(|&x| x != u);
            let a = nu.next().unwrap();
            let b = nu.next().unwrap();
            let c = nv.next().unwrap();
            let d = nv.next().unwrap();
            let all6 = [u, v, a, b, c, d];
            if all6.iter().any(|x| locked.contains(x)) {
                continue;
            }
            for &x in &all6 {
                locked.insert(x);
            }
            let sw = if topo == 1 { c } else { d };
            let len_v_sw = edge_lens
                .remove(&(v.min(sw), v.max(sw)))
                .unwrap_or(0.02);
            let len_u_b = edge_lens
                .remove(&(u.min(b), u.max(b)))
                .unwrap_or(0.02);
            adj.get_mut(&u).unwrap().retain(|&x| x != b);
            adj.get_mut(&b).unwrap().retain(|&x| x != u);
            adj.get_mut(&v).unwrap().retain(|&x| x != sw);
            adj.get_mut(&sw).unwrap().retain(|&x| x != v);
            adj.get_mut(&u).unwrap().push(sw);
            adj.get_mut(&sw).unwrap().push(u);
            adj.get_mut(&v).unwrap().push(b);
            adj.get_mut(&b).unwrap().push(v);
            edge_lens.insert((u.min(sw), u.max(sw)), len_v_sw);
            edge_lens.insert((v.min(b), v.max(b)), len_u_b);
            edge_lens.insert((u, v), nt);
            n_sw += 1;
        }
        (n_sw, stats, locked)
    };

    let run_leaf_len_pass = |adj: &BTreeMap<usize, Vec<usize>>,
                             edge_lens: &mut HashMap<(usize, usize), f64>|
     -> HashMap<(usize, usize), (f64, f64)> {
        let (n_slots, undir_lens, _undir_idx, dir_slot, up_leaf, up_int, down_int) =
            nt_compile_tree_ops(adj, edge_lens, n);
        let mut leaf_tasks = Vec::with_capacity(n);
        let mut leaf_edges = Vec::with_capacity(n);
        for leaf in 0..n {
            if let Some(nbs) = adj.get(&leaf) {
                if nbs.len() == 1 {
                    let p = nbs[0];
                    if p >= n && adj.get(&p).map(|x| x.len()).unwrap_or(0) == 3 {
                        let mut oths = adj[&p].iter().copied().filter(|&x| x != leaf);
                        let o0 = oths.next().unwrap();
                        let o1 = oths.next().unwrap();
                        leaf_tasks.push(NtLeafTask {
                            leaf,
                            o0: dir_slot[&(o0, p)],
                            o1: dir_slot[&(o1, p)],
                        });
                        leaf_edges.push((leaf, p));
                    }
                }
            }
        }
        if leaf_tasks.is_empty() {
            return HashMap::new();
        }
        let (p_iupac, p_int) = build_cat_pmats(&undir_lens);
        let n_lf = leaf_tasks.len();

        let eval_chunk = |chunk: &[NtGtrTile]| -> Vec<f64> {
            let mut acc = vec![0.0_f64; n_lf * NT_N_TGRID];
            let mut msg_buf = vec![[[0.0_f32; NT_TILE]; 4]; n_slots];
            let mut lsc_buf = vec![[0.0_f32; NT_TILE]; n_slots];
            let mut vp_n = [[0.0_f32; NT_TILE]; 4];
            let mut lsc_vp = [0.0_f32; NT_TILE];

            for tile in chunk {
                nt_populate_tile_messages(
                    tile,
                    &p_iupac[tile.cat],
                    &p_int[tile.cat],
                    &up_leaf,
                    &up_int,
                    &down_int,
                    &mut msg_buf,
                    &mut lsc_buf,
                );
                let pb_grid = &pb_leaf_cat_grid[tile.cat];
                let wts = &tile.weights;

                for (li, task) in leaf_tasks.iter().enumerate() {
                    nt_mul_norm4_soa(&msg_buf[task.o0], &msg_buf[task.o1], &mut vp_n, &mut lsc_vp);
                    let st = &tile.leaf_masks[task.leaf];
                    let off = li * NT_N_TGRID;
                    for s in 0..NT_TILE {
                        let w = wts[s];
                        let m = (st[s] & 15) as usize;
                        if w > 0.0 && m != 15 && m != 0 {
                            let wd = w as f64;
                            let (v0, v1, v2, v3) =
                                (vp_n[0][s], vp_n[1][s], vp_n[2][s], vp_n[3][s]);
                            for k in 0..NT_N_TGRID {
                                let pb = &pb_grid[k][m];
                                let dot =
                                    pb[0] * v0 + pb[1] * v1 + pb[2] * v2 + pb[3] * v3;
                                acc[off + k] += wd * (ln_f32_noerrno(dot) as f64);
                            }
                        }
                    }
                }
            }
            acc
        };

        let chunk_size = (tiles.len() / 64).clamp(1, 16);
        #[cfg(not(target_arch = "wasm32"))]
        let parts: Vec<Vec<f64>> = tiles.par_chunks(chunk_size).map(eval_chunk).collect();
        #[cfg(target_arch = "wasm32")]
        let parts: Vec<Vec<f64>> = tiles.chunks(chunk_size).map(eval_chunk).collect();

        let mut total_ll = vec![0.0_f64; n_lf * NT_N_TGRID];
        for p in parts {
            for idx in 0..total_ll.len() {
                total_ll[idx] += p[idx];
            }
        }

        let mut leaf_stats = HashMap::with_capacity(n_lf);
        for (li, &(leaf, p)) in leaf_edges.iter().enumerate() {
            let off = li * NT_N_TGRID;
            let (dll, tmle) = refine_grid_peak_ln_t(
                &total_ll[off..off + NT_N_TGRID],
                &NT_T_LEAF_GRID,
                min_bl,
            );
            let key = (leaf.min(p), leaf.max(p));
            edge_lens.insert(key, tmle.max(3e-5));
            leaf_stats.insert(key, (dll, tmle));
        }
        leaf_stats
    };

    let run_spr2_pass = |adj: &mut BTreeMap<usize, Vec<usize>>,
                         edge_lens: &mut HashMap<(usize, usize), f64>,
                         supports: &HashMap<(usize, usize), f64>,
                         max_sup_thresh: f64|
     -> (usize, HashSet<usize>) {
        let mut triples = Vec::new();
        for (&v1, nbs) in adj.iter() {
            if v1 < n || nbs.len() != 3 {
                continue;
            }
            let inbs: Vec<usize> = nbs
                .iter()
                .copied()
                .filter(|&x| x >= n && adj.get(&x).map(|y| y.len()).unwrap_or(0) == 3)
                .collect();
            for i in 0..inbs.len() {
                for j in (i + 1)..inbs.len() {
                    let (v0, v2) = if inbs[i] < inbs[j] {
                        (inbs[i], inbs[j])
                    } else {
                        (inbs[j], inbs[i])
                    };
                    let s1 = supports
                        .get(&(v0.min(v1), v0.max(v1)))
                        .copied()
                        .unwrap_or(0.0);
                    let s2 = supports
                        .get(&(v1.min(v2), v1.max(v2)))
                        .copied()
                        .unwrap_or(0.0);
                    if s1.min(s2) < max_sup_thresh {
                        triples.push((v0, v1, v2));
                    }
                }
            }
        }
        if triples.is_empty() {
            return (0, HashSet::new());
        }

        let (n_slots, undir_lens, undir_idx, dir_slot, up_leaf, up_int, down_int) =
            nt_compile_tree_ops(adj, edge_lens, n);
        let (p_iupac, p_int) = build_cat_pmats(&undir_lens);

        let spr_floor = (min_bl as f32).max(5e-4_f32);
        let mut spr_left_cat: Vec<Vec<[[f32; 4]; 4]>> = Vec::with_capacity(NT_N_CATS);
        let mut spr_right_pi_cat: Vec<Vec<[[f32; 4]; 4]>> = Vec::with_capacity(NT_N_CATS);
        for c in 0..NT_N_CATS {
            let cr = NT_CAT_RATES[c];
            let mut l_vec = Vec::with_capacity(undir_lens.len());
            let mut r_vec = Vec::with_capacity(undir_lens.len());
            for &t in &undir_lens {
                let pm = gtr.trans_mat4(cr * t.max(spr_floor));
                let mut pr_pi = [[0.0_f32; 4]; 4];
                for a in 0..4 {
                    let pia = gtr.pi[a] as f32;
                    for b in 0..4 {
                        pr_pi[a][b] = pia * pm[a][b];
                    }
                }
                l_vec.push(pm);
                r_vec.push(pr_pi);
            }
            spr_left_cat.push(l_vec);
            spr_right_pi_cat.push(r_vec);
        }

        let spr_tasks: Vec<NtSprTask> = triples
            .iter()
            .map(|&(v0, v1, v2)| {
                let mut ab = adj[&v0].iter().copied().filter(|&x| x != v1);
                let a = ab.next().unwrap();
                let b = ab.next().unwrap();
                let m = adj[&v1]
                    .iter()
                    .copied()
                    .find(|&x| x != v0 && x != v2)
                    .unwrap();
                let mut cd = adj[&v2].iter().copied().filter(|&x| x != v1);
                let c = cd.next().unwrap();
                let d = cd.next().unwrap();
                NtSprTask {
                    m_slots: [
                        dir_slot[&(a, v0)],
                        dir_slot[&(b, v0)],
                        dir_slot[&(m, v1)],
                        dir_slot[&(c, v2)],
                        dir_slot[&(d, v2)],
                    ],
                    e_left: undir_idx[&(v0.min(v1), v0.max(v1))],
                    e_right: undir_idx[&(v1.min(v2), v1.max(v2))],
                }
            })
            .collect();

        let n_tr = spr_tasks.len();
        let eval_chunk = |chunk: &[NtGtrTile]| -> Vec<f64> {
            let mut acc = vec![0.0_f64; n_tr * 15];
            let mut msg_buf = vec![[[0.0_f32; NT_TILE]; 4]; n_slots];
            let mut lsc_buf = vec![[0.0_f32; NT_TILE]; n_slots];
            let mut l_ch = [[0.0_f32; NT_TILE]; 4];
            let mut r_ch = [[0.0_f32; NT_TILE]; 4];
            let mut mid_p = [[0.0_f32; NT_TILE]; 4];
            let mut mid_n = [[0.0_f32; NT_TILE]; 4];
            let mut r_pi_m = [[0.0_f32; NT_TILE]; 4];
            let mut lsc_l = [0.0_f32; NT_TILE];
            let mut lsc_r = [0.0_f32; NT_TILE];
            let mut lsc_m = [0.0_f32; NT_TILE];

            for tile in chunk {
                nt_populate_tile_messages(
                    tile,
                    &p_iupac[tile.cat],
                    &p_int[tile.cat],
                    &up_leaf,
                    &up_int,
                    &down_int,
                    &mut msg_buf,
                    &mut lsc_buf,
                );
                let pl_cat = &spr_left_cat[tile.cat];
                let pr_cat = &spr_right_pi_cat[tile.cat];
                let wts = &tile.weights;

                for (ti, task) in spr_tasks.iter().enumerate() {
                    let ms = &task.m_slots;
                    let mut bsc = 0.0_f64;
                    for s in 0..NT_TILE {
                        bsc += (wts[s] as f64)
                            * ((lsc_buf[ms[0]][s]
                                + lsc_buf[ms[1]][s]
                                + lsc_buf[ms[2]][s]
                                + lsc_buf[ms[3]][s]
                                + lsc_buf[ms[4]][s]) as f64);
                    }
                    let pl = &pl_cat[task.e_left];
                    let pr_pi = &pr_cat[task.e_right];
                    let off = ti * 15;
                    for (pi_idx, perm) in NT_SPR_PERMS.iter().enumerate() {
                        nt_mul_norm4_soa(
                            &msg_buf[ms[perm[0]]],
                            &msg_buf[ms[perm[1]]],
                            &mut l_ch,
                            &mut lsc_l,
                        );
                        nt_mul_norm4_soa(
                            &msg_buf[ms[perm[3]]],
                            &msg_buf[ms[perm[4]]],
                            &mut r_ch,
                            &mut lsc_r,
                        );
                        nt_matmul4_soa(pl, &l_ch, &mut mid_p);
                        nt_mul_norm4_soa(&mid_p, &msg_buf[ms[perm[2]]], &mut mid_n, &mut lsc_m);
                        nt_matmul4_soa(pr_pi, &mid_n, &mut r_pi_m);
                        let mut sum_s = 0.0_f64;
                        for s in 0..NT_TILE {
                            let ps = r_pi_m[0][s] * r_ch[0][s]
                                + r_pi_m[1][s] * r_ch[1][s]
                                + r_pi_m[2][s] * r_ch[2][s]
                                + r_pi_m[3][s] * r_ch[3][s];
                            sum_s += (wts[s] as f64)
                                * ((lsc_l[s] + lsc_r[s] + lsc_m[s] + ln_f32_noerrno(ps)) as f64);
                        }
                        acc[off + pi_idx] += bsc + sum_s;
                    }
                }
            }
            acc
        };

        let chunk_size = (tiles.len() / 64).clamp(1, 16);
        #[cfg(not(target_arch = "wasm32"))]
        let parts: Vec<Vec<f64>> = tiles.par_chunks(chunk_size).map(eval_chunk).collect();
        #[cfg(target_arch = "wasm32")]
        let parts: Vec<Vec<f64>> = tiles.chunks(chunk_size).map(eval_chunk).collect();

        let mut total_ll = vec![0.0_f64; n_tr * 15];
        for p in parts {
            for idx in 0..total_ll.len() {
                total_ll[idx] += p[idx];
            }
        }

        let mut moves: Vec<(f64, usize, usize, usize, [usize; 5])> = Vec::new();
        for (ti, &(v0, v1, v2)) in triples.iter().enumerate() {
            let off = ti * 15;
            let cur_ll = total_ll[off];
            let mut bi = 1usize;
            for p_i in 2..15 {
                if total_ll[off + p_i] > total_ll[off + bi] {
                    bi = p_i;
                }
            }
            if total_ll[off + bi] > cur_ll + 5e-2 {
                moves.push((total_ll[off + bi] - cur_ll, v0, v1, v2, NT_SPR_PERMS[bi]));
            }
        }
        if moves.is_empty() {
            return (0, HashSet::new());
        }
        moves.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
                .then_with(|| a.2.cmp(&b.2))
                .then_with(|| a.3.cmp(&b.3))
        });

        let mut locked: HashSet<usize> = HashSet::new();
        let mut n_mv = 0usize;
        for (_gain, v0, v1, v2, perm) in moves {
            let mut ab = adj[&v0].iter().copied().filter(|&x| x != v1);
            let a = ab.next().unwrap();
            let b = ab.next().unwrap();
            let m = adj[&v1]
                .iter()
                .copied()
                .find(|&x| x != v0 && x != v2)
                .unwrap();
            let mut cd = adj[&v2].iter().copied().filter(|&x| x != v1);
            let c = cd.next().unwrap();
            let d = cd.next().unwrap();
            let orig = [a, b, m, c, d];
            let all8 = [v0, v1, v2, a, b, m, c, d];
            if all8.iter().any(|x| locked.contains(x)) {
                continue;
            }
            for &x in &all8 {
                locked.insert(x);
            }
            let old_pairs = [(a, v0), (b, v0), (m, v1), (c, v2), (d, v2)];
            let mut old_lens = [0.02_f64; 5];
            for (idx, &(x, vx)) in old_pairs.iter().enumerate() {
                old_lens[idx] = edge_lens
                    .remove(&(x.min(vx), x.max(vx)))
                    .unwrap_or(0.02);
                adj.get_mut(&vx).unwrap().retain(|&y| y != x);
                adj.get_mut(&x).unwrap().retain(|&y| y != vx);
            }
            let new_assign = [
                (orig[perm[0]], old_lens[perm[0]], v0),
                (orig[perm[1]], old_lens[perm[1]], v0),
                (orig[perm[2]], old_lens[perm[2]], v1),
                (orig[perm[3]], old_lens[perm[3]], v2),
                (orig[perm[4]], old_lens[perm[4]], v2),
            ];
            for (nd, el, vx) in new_assign {
                adj.get_mut(&vx).unwrap().push(nd);
                adj.get_mut(&nd).unwrap().push(vx);
                edge_lens.insert((nd.min(vx), nd.max(vx)), el);
            }
            n_mv += 1;
        }
        (n_mv, locked)
    };

    // Execute NNI(3) -> LeafMLE -> SPR2-1 -> NNI(3) -> [SPR2-2 -> NNI(2)] -> Final Score
    let mut prev_sup: HashMap<(usize, usize), f64> = HashMap::new();
    let mut touched: HashSet<usize> = HashSet::new();
    let mut last_int_stats: HashMap<(usize, usize), (f64, f64)> = HashMap::new();

    for r in 0..3 {
        let (nsw, st, lk) = run_nni_pass(adj, edge_lens, &prev_sup, &touched, true);
        prev_sup = st.iter().map(|(&k, &(dll, _))| (k, dll)).collect();
        last_int_stats = st;
        touched = lk;
        if nsw == 0 && r >= 1 {
            break;
        }
    }

    let _ = run_leaf_len_pass(adj, edge_lens);
    let mut last_clean = false;
    let (n_spr1, lk_spr1) = run_spr2_pass(adj, edge_lens, &prev_sup, 35.0);
    if n_spr1 > 0 {
        last_clean = false;
        prev_sup.clear();
        touched = lk_spr1;
        for _ in 0..3 {
            let (nsw, st, lk) = run_nni_pass(adj, edge_lens, &prev_sup, &touched, true);
            prev_sup = st.iter().map(|(&k, &(dll, _))| (k, dll)).collect();
            last_int_stats = st;
            touched = lk;
            if nsw == 0 {
                last_clean = true;
                break;
            }
        }
        let (n_spr2, lk_spr2) = run_spr2_pass(adj, edge_lens, &prev_sup, 25.0);
        if n_spr2 > 0 {
            last_clean = false;
            prev_sup.clear();
            touched = lk_spr2;
            for _ in 0..2 {
                let (nsw, st, lk) = run_nni_pass(adj, edge_lens, &prev_sup, &touched, true);
                prev_sup = st.iter().map(|(&k, &(dll, _))| (k, dll)).collect();
                last_int_stats = st;
                touched = lk;
                if nsw == 0 {
                    last_clean = true;
                    break;
                }
            }
        }
    }

    if !last_clean {
        let (_, st, _) = run_nni_pass(adj, edge_lens, &HashMap::new(), &HashSet::new(), false);
        last_int_stats = st;
    }
    let leaf_stats = run_leaf_len_pass(adj, edge_lens);
    for (k, v) in leaf_stats {
        last_int_stats.insert(k, v);
    }
    (last_int_stats, gtr.emp_rate_scale)
}
