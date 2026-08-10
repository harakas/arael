//! Numeric rank and null-space basis of a sparse [`Jacobian`].
//!
//! The rank of the column-normalised Jacobian decides how many degrees
//! of freedom a model has left (`nullity = num_params - rank`) and
//! whether a candidate constraint row would remove one. Two paths
//! produce the same decision:
//!
//! - **Dense** (small problems): full SVD of the normalised Jacobian.
//! - **Iterative** (large problems): shift-inverted block subspace
//!   iteration. A sparse Cholesky of the normalised `J^T J + lambda*I`
//!   amplifies null directions by `1/lambda`; a few solve+orthonormalise
//!   sweeps converge a block containing the null space, and the rank
//!   decision is taken from the singular values of `J` restricted to
//!   that block -- the same metric as the dense path, no
//!   condition-number squaring in the decision.
//!
//! Both fill [`RankResult`] with an orthonormal null-space basis, so a
//! candidate row can be tested against the free directions in
//! microseconds ([`RankResult::reduces_rank`]) instead of recomputing
//! the rank.

use crate::model::Jacobian;
use faer::sparse::linalg::cholesky as fchol;
use std::mem::MaybeUninit;

/// Options for [`Jacobian::numeric_rank`].
#[derive(Clone, Debug)]
pub struct RankOptions {
    /// Shift added to the diagonal of the normalised normal matrix.
    pub lambda: f64,
    /// Inverse-iteration sweeps per attempt.
    pub sweeps: usize,
    /// Expected nullity (e.g. the previous DOF of the same model). The
    /// block starts at `hint + margin` and grows until the boundary is
    /// clean.
    pub null_hint: Option<usize>,
    /// Extra block columns above the hint.
    pub margin: usize,
    /// At or below this parameter count the dense path runs instead.
    pub dense_cutoff: usize,
    /// Tolerance for [`RankResult::reduces_rank`]: a candidate row
    /// whose relative null-space component exceeds this reduces rank.
    pub row_tol: f64,
}

impl Default for RankOptions {
    fn default() -> Self {
        Self {
            lambda: 1e-10,
            sweeps: 2,
            null_hint: None,
            margin: 16,
            dense_cutoff: 64,
            row_tol: 1e-6,
        }
    }
}

/// Which path produced a [`RankResult`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RankMethod {
    /// Dense SVD of the normalised Jacobian.
    Dense,
    /// Subspace iteration; `block` is the final block width, `grew`
    /// counts how many times the block had to grow.
    Iterative { block: usize, grew: usize },
}

/// Rank computation failure.
#[derive(Clone, Debug)]
pub enum RankError {
    /// The Jacobian contains a non-finite entry; the label names the
    /// first offending constraint row.
    NonFinite { label: &'static str },
    /// The shifted normal matrix could not be factorized.
    Factorization,
}

impl std::fmt::Display for RankError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            RankError::NonFinite { label } => {
                write!(f, "non-finite Jacobian entry in constraint row '{}'", label)
            }
            RankError::Factorization => write!(f, "sparse factorization of J^T J failed"),
        }
    }
}

/// Numeric rank of a column-normalised Jacobian, with the null basis.
#[derive(Clone, Debug)]
pub struct RankResult {
    /// Numeric rank of the column-normalised Jacobian.
    pub rank: usize,
    /// `num_params - rank`: the model's free directions (DOF).
    pub nullity: usize,
    /// Ratio across the zero/real boundary of the decided spectrum.
    /// `f64::INFINITY` when one side is empty.
    pub gap: f64,
    /// Which path ran.
    pub method: RankMethod,
    row_tol: f64,
    /// Column scales of the normalisation (already floored).
    scales: Vec<f64>,
    /// Orthonormal null basis, column-major `n x nullity`.
    basis: Vec<f64>,
    n: usize,
}

impl RankResult {
    /// Component of a candidate row inside the null space, relative to
    /// the row's own normalised norm (0 = row already in the span of
    /// the existing rows, 1 = row entirely in the free directions).
    /// Entries are `(param_index, d_residual/d_param)`, raw scale --
    /// the stored column normalisation is applied here.
    pub fn null_component(&self, entries: &[(u32, f64)]) -> f64 {
        let k = self.nullity;
        if k == 0 {
            return 0.0;
        }
        let mut row_norm2 = 0.0f64;
        let mut proj2 = 0.0f64;
        let mut proj = vec![0.0f64; k];
        for &(idx, v) in entries {
            let i = idx as usize;
            if i >= self.n {
                continue;
            }
            let vn = v / self.scales[i];
            row_norm2 += vn * vn;
            for (c, p) in proj.iter_mut().enumerate() {
                *p += vn * self.basis[c * self.n + i];
            }
        }
        for p in &proj {
            proj2 += p * p;
        }
        if row_norm2 <= 0.0 {
            return 0.0;
        }
        (proj2 / row_norm2).sqrt()
    }

    /// True when adding this row to the Jacobian would reduce the
    /// nullity: it constrains at least one currently free direction.
    pub fn reduces_rank(&self, entries: &[(u32, f64)]) -> bool {
        self.null_component(entries) > self.row_tol
    }

    /// The orthonormal null basis: column-major slice and its column
    /// count (`num_params x nullity`), in normalised parameter space.
    pub fn null_basis(&self) -> (&[f64], usize) {
        (&self.basis, self.nullity)
    }
}

/// The shared rank decision: gap search over an ascending spectrum.
/// Returns `(count_below_cut, gap_ratio)`. The search considers cuts
/// whose low side is under 1% of the largest value, values floored at
/// `max * 1e-20`; when no gap reaches 1e3 the fallback counts values
/// below the absolute 1e-15.
pub fn rank_cut(sorted_ascending: &[f64]) -> (usize, f64) {
    let max_sv = sorted_ascending.last().copied().unwrap_or(0.0);
    let upper_bound = max_sv * 0.01;
    let floor = max_sv * 1e-20;
    let mut best_gap = 0.0f64;
    let mut best_cut = 0;
    for i in 0..sorted_ascending.len().saturating_sub(1) {
        let lo = sorted_ascending[i].max(floor);
        let hi = sorted_ascending[i + 1].max(floor);
        if lo > upper_bound {
            break;
        }
        let gap = hi / lo;
        if gap > best_gap {
            best_gap = gap;
            best_cut = i + 1;
        }
    }
    if best_gap < 1e3 {
        best_cut = sorted_ascending.iter().filter(|&&v| v < 1e-15).count();
    }
    (best_cut, best_gap)
}

impl Jacobian<f64> {
    /// Numeric rank of the column-normalised Jacobian. Errors on
    /// non-finite entries instead of producing NaN spectra.
    pub fn numeric_rank(&self, opts: &RankOptions) -> Result<RankResult, RankError> {
        self.rank_impl(opts, None)
    }

    /// [`Self::numeric_rank`] warm-started from a previous result's
    /// null basis -- for re-evaluation after value-only parameter
    /// changes, where the null space moved only slightly. Typically
    /// converges with `sweeps: 1`.
    pub fn numeric_rank_warm(
        &self,
        opts: &RankOptions,
        prev: &RankResult,
    ) -> Result<RankResult, RankError> {
        if prev.n != self.num_params {
            return self.rank_impl(opts, None);
        }
        self.rank_impl(opts, Some(prev))
    }

    fn rank_impl(
        &self,
        opts: &RankOptions,
        warm: Option<&RankResult>,
    ) -> Result<RankResult, RankError> {
        let n = self.num_params;
        let m = self.num_residuals();
        if n == 0 {
            return Ok(RankResult {
                rank: 0,
                nullity: 0,
                gap: f64::INFINITY,
                method: RankMethod::Dense,
                row_tol: opts.row_tol,
                scales: Vec::new(),
                basis: Vec::new(),
                n,
            });
        }
        for row in &self.rows {
            for &(_, v) in &row.entries {
                if !v.is_finite() {
                    return Err(RankError::NonFinite { label: row.label });
                }
            }
        }
        let scales: Vec<f64> = self.column_l2_norms().iter().map(|c| c.max(1e-15)).collect();
        if m == 0 {
            // No residuals: every parameter is free, identity basis.
            let mut basis = vec![0.0f64; n * n];
            for i in 0..n {
                basis[i * n + i] = 1.0;
            }
            return Ok(RankResult {
                rank: 0,
                nullity: n,
                gap: f64::INFINITY,
                method: RankMethod::Dense,
                row_tol: opts.row_tol,
                scales,
                basis,
                n,
            });
        }
        if n <= opts.dense_cutoff {
            return Ok(self.rank_dense(opts, scales));
        }
        self.rank_iterative(opts, scales, warm)
    }

    fn rank_dense(&self, opts: &RankOptions, scales: Vec<f64>) -> RankResult {
        let n = self.num_params;
        let (svd, _) = self.svd_column_normalised();
        let k = svd.singular_values.len();
        let mut sorted: Vec<f64> = svd.singular_values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let (zeros_in_spectrum, gap) = rank_cut(&sorted);
        let rank = k - zeros_in_spectrum;
        let nullity = n - rank;

        // Null basis: right singular vectors of the zero cluster (the
        // spectrum is descending, so those are the last columns), plus
        // an orthonormal completion for the n - k directions a thin
        // m < n SVD cannot represent.
        let mut basis = vec![0.0f64; n * nullity];
        for (c, sv_col) in (rank..k).enumerate() {
            for i in 0..n {
                basis[c * n + i] = svd.v[i * k + sv_col];
            }
        }
        let have = k - rank;
        if have < nullity {
            complete_orthonormal(&mut basis, n, have, nullity, &svd.v, k, rank);
        }
        RankResult {
            rank,
            nullity,
            gap,
            method: RankMethod::Dense,
            row_tol: opts.row_tol,
            scales,
            basis,
            n,
        }
    }

    fn rank_iterative(
        &self,
        opts: &RankOptions,
        scales: Vec<f64>,
        warm: Option<&RankResult>,
    ) -> Result<RankResult, RankError> {
        let n = self.num_params;
        let m = self.num_residuals();

        // Assemble the normalised H + lambda*I as full symmetric CSC.
        let mut upper: std::collections::HashMap<(u32, u32), f64> = std::collections::HashMap::new();
        for row in &self.rows {
            for (ai, &(ia, va)) in row.entries.iter().enumerate() {
                let van = va / scales[ia as usize];
                for &(ib, vb) in &row.entries[ai..] {
                    let vbn = vb / scales[ib as usize];
                    let key = if ia <= ib { (ia, ib) } else { (ib, ia) };
                    *upper.entry(key).or_insert(0.0) += van * vbn;
                }
            }
        }
        let mut cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        for (&(i, j), &v) in &upper {
            let (i, j) = (i as usize, j as usize);
            cols[j].push((i, v));
            if i != j {
                cols[i].push((j, v));
            }
        }
        let mut col_ptr: Vec<usize> = Vec::with_capacity(n + 1);
        let mut row_idx: Vec<usize> = Vec::new();
        let mut vals: Vec<f64> = Vec::new();
        col_ptr.push(0);
        for (j, col) in cols.iter_mut().enumerate() {
            if !col.iter().any(|&(i, _)| i == j) {
                col.push((j, 0.0));
            }
            col.sort_by_key(|&(i, _)| i);
            for &(i, v) in col.iter() {
                row_idx.push(i);
                vals.push(if i == j { v + opts.lambda } else { v });
            }
            col_ptr.push(row_idx.len());
        }

        let sym_ref =
            faer::sparse::SymbolicSparseColMatRef::new_checked(n, n, &col_ptr, None, &row_idx);
        let symbolic = fchol::factorize_symbolic_cholesky(
            sym_ref,
            faer::Side::Upper,
            fchol::SymmetricOrdering::Amd,
            fchol::CholeskySymbolicParams::default(),
        )
        .map_err(|_| RankError::Factorization)?;
        let mut l_vals = vec![0.0f64; symbolic.len_val()];
        let factor_req =
            symbolic.factorize_numeric_llt_scratch::<f64>(faer::Par::Seq, faer::Spec::default());
        let mut factor_mem: Vec<MaybeUninit<u8>> =
            vec![MaybeUninit::uninit(); factor_req.unaligned_bytes_required()];
        let stack = faer::dyn_stack::MemStack::new(&mut factor_mem);
        let mat_ref = faer::sparse::SparseColMatRef::new(sym_ref, &vals);
        symbolic
            .factorize_numeric_llt(
                &mut l_vals,
                mat_ref,
                faer::Side::Upper,
                faer::linalg::cholesky::llt::factor::LltRegularization::default(),
                faer::Par::Seq,
                stack,
                faer::Spec::default(),
            )
            .map_err(|_| RankError::Factorization)?;
        let llt = fchol::LltRef::new(&symbolic, &l_vals);


        let hint = opts.null_hint.or(warm.map(|w| w.nullity)).unwrap_or(opts.margin);
        let mut k = (hint + opts.margin).clamp(1, n);
        let mut grew = 0usize;
        let mut rng = 0x9e3779b97f4a7c15u64;
        loop {
            // Start block: warm basis columns first, random fill after.
            let mut v = vec![0.0f64; n * k];
            let warm_cols = match warm {
                Some(w) if grew == 0 => {
                    let take = w.nullity.min(k);
                    v[..n * take].copy_from_slice(&w.basis[..n * take]);
                    take
                }
                _ => 0,
            };
            for x in v[n * warm_cols..].iter_mut() {
                *x = xorshift(&mut rng);
            }

            let solve_req = symbolic.solve_in_place_scratch::<f64>(k, faer::Par::Seq);
            let mut solve_mem: Vec<MaybeUninit<u8>> =
                vec![MaybeUninit::uninit(); solve_req.unaligned_bytes_required()];
            for _ in 0..opts.sweeps.max(1) {
                let stack = faer::dyn_stack::MemStack::new(&mut solve_mem);
                let rhs = faer::mat::MatMut::from_column_major_slice_mut(&mut v, n, k);
                llt.solve_in_place_with_conj(faer::Conj::No, rhs, faer::Par::Seq, stack);
                orthonormalize(&mut v, n, k);
            }

            // Decision in the J metric: singular values of B = Jn * V.
            let mut b = vec![0.0f64; m * k];
            for (r, row) in self.rows.iter().enumerate() {
                for &(ci, val) in &row.entries {
                    let van = val / scales[ci as usize];
                    let base = ci as usize;
                    for c in 0..k {
                        b[c * m + r] += van * v[c * n + base];
                    }
                }
            }
            // sigma(B) = sigma(R) and B's right singular vectors are
            // R's: decompose the small R instead of the tall B.
            let bmat = faer::mat::MatRef::from_column_major_slice(&b, m, k);
            let bqr = bmat.qr();
            let bsvd = bqr.thin_R().thin_svd().map_err(|_| RankError::Factorization)?;
            let s = bsvd.S().column_vector();
            let mn = s.nrows();
            let mut order: Vec<usize> = (0..mn).collect();
            order.sort_by(|&a, &c| s[a].partial_cmp(&s[c]).unwrap());
            // When k > m the thin SVD reports only m singular values;
            // the remaining k - m directions of the block are
            // structurally zero and take part in the gap search.
            let pad = k - mn;
            let mut sorted: Vec<f64> = vec![0.0; pad];
            sorted.extend(order.iter().map(|&i| s[i]));
            let (cut, gap) = rank_cut(&sorted);

            // Clean boundary: a real gap, the null cluster strictly
            // inside the block (cut == k would mean the null space may
            // extend beyond it), and the real side genuinely real -- a
            // block living entirely inside the null space produces
            // noise-to-noise ratios that can fake a gap. At k == n the
            // block is the whole space and the decision is exact.
            let clean = cut < k && gap >= 1e3 && sorted[cut] > 1e-10;
            if clean || k == n {
                let nullity = cut;
                let rank = n - nullity;
                // Rotate the block onto the null directions: N = V * W.
                // W's columns are the right singular vectors of the zero
                // cluster, then (for k > m) an orthonormal completion of
                // the subspace the thin SVD cannot represent -- those
                // directions map to zero under B by construction.
                let v_mat = bsvd.V();
                let computed_zeros = nullity.saturating_sub(pad);
                let mut w = vec![0.0f64; k * nullity];
                for (c, &sv_i) in order[..computed_zeros].iter().enumerate() {
                    for col in 0..k {
                        w[c * k + col] = v_mat[(col, sv_i)];
                    }
                }
                for c in computed_zeros..nullity {
                    loop {
                        let mut cand: Vec<f64> = (0..k).map(|_| xorshift(&mut rng)).collect();
                        for r in 0..mn {
                            let mut dot = 0.0f64;
                            for col in 0..k {
                                dot += cand[col] * v_mat[(col, r)];
                            }
                            for col in 0..k {
                                cand[col] -= dot * v_mat[(col, r)];
                            }
                        }
                        for p in 0..c {
                            let mut dot = 0.0f64;
                            for col in 0..k {
                                dot += cand[col] * w[p * k + col];
                            }
                            for col in 0..k {
                                cand[col] -= dot * w[p * k + col];
                            }
                        }
                        let norm: f64 = cand.iter().map(|x| x * x).sum::<f64>().sqrt();
                        if norm > 1e-8 {
                            for (col, x) in cand.iter().enumerate() {
                                w[c * k + col] = x / norm;
                            }
                            break;
                        }
                    }
                }
                let mut basis = vec![0.0f64; n * nullity];
                if nullity > 0 {
                    faer::linalg::matmul::matmul(
                        faer::mat::MatMut::from_column_major_slice_mut(&mut basis, n, nullity),
                        faer::Accum::Replace,
                        faer::mat::MatRef::from_column_major_slice(&v, n, k),
                        faer::mat::MatRef::from_column_major_slice(&w, k, nullity),
                        1.0,
                        faer::Par::Seq,
                    );
                }
                return Ok(RankResult {
                    rank,
                    nullity,
                    gap,
                    method: RankMethod::Iterative { block: k, grew },
                    row_tol: opts.row_tol,
                    scales,
                    basis,
                    n,
                });
            }
            k = (k * 2 + opts.margin).min(n);
            grew += 1;
        }
    }
}

fn xorshift(state: &mut u64) -> f64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    (x >> 11) as f64 / (1u64 << 53) as f64 - 0.5
}

/// Orthonormalize the columns of column-major `v` (n x k) in place via
/// a Householder thin Q -- orthonormal even when the input columns are
/// linearly dependent.
fn orthonormalize(v: &mut [f64], n: usize, k: usize) {
    let q = faer::mat::MatRef::from_column_major_slice(&*v, n, k)
        .qr()
        .compute_thin_Q();
    for c in 0..k {
        for i in 0..n {
            v[c * n + i] = q[(i, c)];
        }
    }
}

/// Extend `basis` (n x nullity column-major, first `have` columns
/// filled) to `nullity` orthonormal columns that are also orthogonal to
/// the `rank` row-space columns of `v` (n x k row-major).
fn complete_orthonormal(
    basis: &mut [f64],
    n: usize,
    have: usize,
    nullity: usize,
    v: &[f64],
    k: usize,
    rank: usize,
) {
    let mut rng = 0x2545f4914f6cdd1du64;
    for c in have..nullity {
        loop {
            let mut cand: Vec<f64> = (0..n).map(|_| xorshift(&mut rng)).collect();
            // Orthogonalize against the row-space right vectors...
            for sv_col in 0..rank {
                let mut dot = 0.0f64;
                for i in 0..n {
                    dot += cand[i] * v[i * k + sv_col];
                }
                for i in 0..n {
                    cand[i] -= dot * v[i * k + sv_col];
                }
            }
            // ...and the basis columns built so far.
            for p in 0..c {
                let mut dot = 0.0f64;
                for i in 0..n {
                    dot += cand[i] * basis[p * n + i];
                }
                for i in 0..n {
                    cand[i] -= dot * basis[p * n + i];
                }
            }
            let norm: f64 = cand.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-8 {
                for (i, x) in cand.iter().enumerate() {
                    basis[c * n + i] = x / norm;
                }
                break;
            }
        }
    }
}
