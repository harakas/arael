//! Parameter covariance recovery.
//!
//! At the solution the parameter covariance is `Sigma = (J^T J)^-1`. arael
//! assembles `H = 2 J^T J` (the cost is `sum r^2`, so `add_residual` carries a
//! factor of 2), hence `Sigma = 2 H^-1`. Because the rotation parameters are
//! already minimal 3-DOF retractions, `Sigma` is in local tangent coordinates
//! -- no manifold projection is needed.
//!
//! [`assemble_covariance`](crate::covariance::Covariance::assemble_covariance)
//! re-assembles `H` at the solution and prepares it for querying. The dense
//! inverse is never formed. Query per-entity
//! blocks through the entity itself: any `Model` reports its parameter span via
//! `collect_param_blocks`.
//!
//! ```ignore
//! use arael::covariance::{Covariance, CovMode};
//! path.solve_sparse(&cfg);              // solution written back into the model
//! let cov = path.assemble_covariance(CovMode::PerQuery)?;
//! let lm0  = cov.marginal_cov(&path.landmarks[0]);        // one landmark, solves for its columns
//! let lmc  = cov.conditional_cov(&path.landmarks[0]);     // its own info block, others fixed
//! let x    = cov.cross_cov(&path.poses[0], &path.landmarks[3]);
//! ```
//!
//! [`CovMode`](crate::covariance::CovMode) chosen at assembly picks the strategy:
//! - [`CovMode::PerQuery`](crate::covariance::CovMode::PerQuery) factors `H` and
//!   answers each query by solving for its columns (faer picks supernodal where
//!   it pays). Best for a few entities.
//! - [`CovMode::AllMarginals`](crate::covariance::CovMode::AllMarginals) also runs a selected inverse up front -- the
//!   block Takahashi recursion over a supernodal factor (BLAS-3 dense kernels),
//!   computing every covariance entry inside the factor's sparsity pattern -- so
//!   every marginal and coupled cross block becomes a lookup. Best for many/all
//!   marginals.
//! - [`CovMode::TriDiagonal`](crate::covariance::CovMode::TriDiagonal) is for a block-tridiagonal `H` (localization: a
//!   pose chain with a fixed map, no loop closures). It runs a forward Schur pass
//!   over the band -- no factorization at all -- so the last pose's covariance is
//!   `2 S_last^-1`, computed front to back. Querying an interior pose runs a
//!   backward pass once (cached); marginals are then `2 (S_i + R_i - D_i)^-1`.
//!
//! Either way, **conditional** covariance (`conditional_cov`) just inverts the
//! entity's own `H_ee` block (`O(dof^3)`, no factor solve).

use crate::model::Model;
use crate::simple_lm::{BlockSupernodalMode, CooMatrix, CscMatrix, LmProblem, RootProblem};
use crate::utils::Float;
use faer::sparse::linalg::cholesky as fchol;
use nalgebra::DMatrix;
use std::cell::OnceCell;
use std::mem::MaybeUninit;

/// Why a covariance could not be assembled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CovError {
    /// `H` is not positive definite. The usual cause is an unfixed gauge: with
    /// no anchor the problem has unobservable directions and `H` is singular
    /// (e.g. free-gauge SLAM). Fix a pose / add a prior.
    NotPositiveDefinite,
    /// The model has no optimizable parameters.
    Empty,
    /// [`CovMode::TriDiagonal`] was requested but `H` is not block-tridiagonal in
    /// serialization order: some off-band block couples non-adjacent entities (a
    /// loop closure or a free landmark). Use `PerQuery` / `AllMarginals` instead.
    NotTriDiagonal,
    /// A query this assembly's backend cannot answer. The
    /// [`CovMode::TriDiagonal`] backend stores only the band: it serves
    /// single-band-block entity queries, so an entity spanning several
    /// blocks or none, or any cross block, has no answer there. Assemble
    /// with `PerQuery` or `AllMarginals` for those.
    UnsupportedQuery {
        /// The query that failed (`"marginal_cov"`, `"cross_cov"`, ...).
        op: &'static str,
    },
}

impl std::fmt::Display for CovError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CovError::NotPositiveDefinite => write!(f,
                "Hessian is not positive definite (unfixed gauge? add an anchor or a prior)"),
            CovError::Empty => write!(f, "model has no optimizable parameters"),
            CovError::NotTriDiagonal => write!(f,
                "Hessian is not block-tridiagonal (loop closure or free landmark?); use PerQuery or AllMarginals"),
            CovError::UnsupportedQuery { op } => write!(f,
                "{} is unsupported on the TriDiagonal backend for this query                  (entity spans several band blocks or none, or an off-diagonal                  block); assemble with PerQuery or AllMarginals", op),
        }
    }
}

impl std::error::Error for CovError {}

/// How much of the covariance to prepare when assembling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CovMode {
    /// Factor `H` and answer each query by solving for its columns. faer may pick
    /// a supernodal factor when it pays off, so a handful of entities are as fast
    /// as possible. Choose this when you need a few covariances.
    PerQuery,
    /// Also compute the selected inverse up front (a block Takahashi pass over a
    /// supernodal factor), so every marginal and coupled cross block is a lookup.
    /// Choose this when you need many or all marginals (e.g. an ellipse per pose).
    AllMarginals,
    /// Forward Schur pass over a block-tridiagonal `H` (a localization pose chain:
    /// fixed map, no loop closures). The last pose's covariance is then free
    /// (forward only); an interior pose triggers a backward pass once. No
    /// factorization. Errors with `NotTriDiagonal` if `H` is not banded.
    TriDiagonal,
}

/// Elimination ordering for a covariance assembly. Ordering does not change
/// the covariance, only what it costs to compute: it decides how much fill the
/// factor carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CovOrdering {
    /// Minimum degree over the model's BLOCK graph when the model has one,
    /// over `H`'s scalar columns when it does not. The default.
    ///
    /// The block graph is the same coupling with the entity sizes divided
    /// out: a 6-DOF pose against a 3-DOF landmark is one edge there and 18
    /// scalar ones in `H`, so minimum degree walks a much smaller graph for
    /// the same answer.
    #[default]
    Auto,
    /// Minimum degree over `H`'s scalar columns, whatever structure the model
    /// has.
    Amd,
    /// Nested dissection over the model's block graph
    /// ([`arael_faer::nd`]).
    ///
    /// For a trajectory that revisits -- a loop closure, a figure-8 crossing
    /// -- where poses far apart in the ordering are coupled. That leaves a
    /// separator structure dissection exploits and minimum degree does not.
    /// Needs a block structure; without one this falls back to scalar AMD.
    NestedDissection,
    /// Natural order, no permutation. For a model already in a good order,
    /// and for checking what the ordering is worth.
    Natural,
}

/// How to assemble a covariance. A covariance is built from the model, not
/// from a solve, so this is separate from the solver's
/// [`SparseFaerOptions`](crate::simple_lm::SparseFaerOptions) -- the fields
/// that struct carries about damping, marginalization and iterative solves
/// have no meaning here.
#[derive(Debug, Clone, Default)]
pub struct CovOptions {
    /// Elimination ordering -- see [`CovOrdering`].
    pub ordering: CovOrdering,
    /// Factorize in block form with arael's supernodal Cholesky instead of
    /// faer's scalar one -- see [`BlockSupernodalMode`].
    ///
    /// [`CovMode::AllMarginals`] ignores this and stays on the scalar factor:
    /// its selected inverse reads faer's supernode panels. Ask for the block
    /// route there and [`CovAssembly::took_block_route`] reports `false`.
    pub block_supernodal: BlockSupernodalMode,
}

impl CovOptions {
    /// The defaults: ordering and factorization picked from the model's
    /// structure.
    pub fn auto() -> Self {
        Self::default()
    }

    /// Set the elimination ordering.
    pub fn with_ordering(mut self, ordering: CovOrdering) -> Self {
        self.ordering = ordering;
        self
    }

    /// Set when the block supernodal Cholesky factorizes.
    pub fn with_block_supernodal(mut self, mode: BlockSupernodalMode) -> Self {
        self.block_supernodal = mode;
        self
    }
}

/// A covariance prepared at the solution. Build it with
/// [`Covariance::assemble_covariance`], then query per-entity blocks. An owned
/// value: querying does not borrow the problem, so
/// `let cov = p.assemble_covariance(mode)?;` followed by
/// `cov.marginal_cov(&p.landmarks[0])` borrow-checks.
pub struct CovAssembly {
    n: usize,
    backend: Backend,
}

enum Backend {
    // faer sparse Cholesky: PerQuery solves on demand, AllMarginals also holds a
    // selected-inverse cache. `h` (the CSC upper triangle) serves conditional_cov.
    Factored {
        symbolic: fchol::SymbolicCholesky<usize>,
        l_vals: Vec<f64>,
        h: CscMatrix<f64>,
        sel: Option<SelInv>,
    },
    // arael's block supernodal Cholesky over the model's entity partition. No
    // scalar CSC and no scalar symbolic: the block H serves conditional_cov
    // directly. PerQuery only -- the selected inverse reads faer's supernode
    // panels, so AllMarginals stays on Factored.
    BlockFactored {
        sn: arael_faer::supernodal::SupernodalSymbolic,
        factor: Vec<f64>,
        hb: arael_faer::bsc::SparseBlockColMat<arael_faer::SparseIndex, f64>,
        // supernodal_solve wants scratch, and the queries take &self.
        ctx: std::cell::RefCell<arael_faer::supernodal::SupernodalContext<f64>>,
    },
    // Block-tridiagonal forward/backward Schur blocks. The diagonal blocks serve
    // conditional_cov, so no CSC is kept.
    Band(BandData),
}

// The block-tridiagonal representation, in serialization (chain) order. `fwd`
// holds the forward Schur info blocks S_i (S_0 = D_0, S_i = D_i - B_{i-1}
// S_{i-1}^-1 B_{i-1}^T); `bwd` the backward blocks R_i, filled lazily on the
// first interior-pose query. `off[i]` is H[block i, block i+1].
struct BandData {
    spans: Vec<(usize, usize)>,
    diag: Vec<DMatrix<f64>>,
    off: Vec<DMatrix<f64>>,
    fwd: Vec<DMatrix<f64>>,
    bwd: OnceCell<Vec<DMatrix<f64>>>,
}

impl BandData {
    // Block index of an entity from its (contiguous) parameter span, or None
    // when the entity has no parameters or does not line up with a single band
    // block. The TriDiagonal backend can only answer single-block queries; the
    // caller turns None into the empty-matrix sentinel (see marginal_cov).
    fn block_index(&self, idx: &[usize]) -> Option<usize> {
        let offset = *idx.iter().min()?;
        self.spans.iter().position(|&(o, w)| o == offset && w == idx.len())
    }

    // Marginal covariance of block bi: last block is forward-only; an interior
    // block triggers the backward pass (cached), then combines both sides.
    fn marginal(&self, bi: usize) -> Result<DMatrix<f64>, CovError> {
        let nb = self.spans.len();
        let info = if bi == nb - 1 {
            self.fwd[bi].clone()
        } else {
            let bwd = self.bwd.get_or_init(|| self.compute_backward());
            &self.fwd[bi] + &bwd[bi] - &self.diag[bi]
        };
        info.try_inverse()
            .map(|inv| inv * 2.0)
            .ok_or(CovError::NotPositiveDefinite)
    }

    // Backward Schur: R_{n-1} = D_{n-1}, R_i = D_i - B_i R_{i+1}^-1 B_i^T.
    fn compute_backward(&self) -> Vec<DMatrix<f64>> {
        let nb = self.spans.len();
        let mut bwd: Vec<DMatrix<f64>> = self.diag.clone();
        for i in (0..nb - 1).rev() {
            let r = bwd[i + 1].clone();
            let (rn, rc) = (r.nrows(), r.ncols());
            // A singular tail block is unobservable. Fall back to the
            // pseudo-inverse so the pass completes; a genuinely unobservable
            // block then surfaces as NotPositiveDefinite when `marginal`
            // inverts the combined information, matching the singular
            // handling there. The fast LU inverse carries the healthy blocks.
            let r_inv = r.clone().try_inverse().unwrap_or_else(|| {
                r.pseudo_inverse(1e-12).unwrap_or_else(|_| DMatrix::zeros(rn, rc))
            });
            let b = &self.off[i]; // H[i, i+1]
            bwd[i] = &self.diag[i] - b * &r_inv * b.transpose();
        }
        bwd
    }
}

impl CovAssembly {
    /// Number of optimized scalar parameters.
    pub fn dim(&self) -> usize {
        self.n
    }

    /// Whether this assembly factorized in block form
    /// ([`CovOptions::block_supernodal`]). `false` when the model has no block
    /// structure to use, when its blocks are one scalar wide, or for
    /// [`CovMode::AllMarginals`] and [`CovMode::TriDiagonal`], which take
    /// other routes -- so a caller who asked for the block route can check
    /// whether it got it.
    pub fn took_block_route(&self) -> bool {
        matches!(self.backend, Backend::BlockFactored { .. })
    }

    /// Scalar parameter indices covered by a model (an entity's contiguous
    /// live-parameter ranges, or every element's for a collection).
    fn indices<M: Model + ?Sized>(m: &M) -> Vec<usize> {
        let mut spans: Vec<(u32, u32)> = Vec::new();
        m.collect_param_blocks(&mut spans);
        let mut idx = Vec::new();
        for (off, width) in spans {
            for i in off..off + width {
                idx.push(i as usize);
            }
        }
        idx
    }

    /// Marginal covariance of a model: one entity for its own covariance, or a
    /// whole collection for the joint over all its entities. In tangent
    /// coordinates.
    ///
    /// Errors: [`CovError::UnsupportedQuery`] when the
    /// [`CovMode::TriDiagonal`] backend is asked a non-single-band-block
    /// query; [`CovError::NotPositiveDefinite`] when the block is singular
    /// (unobservable).
    pub fn marginal_cov<M: Model + ?Sized>(&self, m: &M) -> Result<DMatrix<f64>, CovError> {
        let idx = Self::indices(m);
        match &self.backend {
            Backend::Factored { .. } | Backend::BlockFactored { .. } => {
                Ok(self.factored_block(&idx, &idx))
            }
            Backend::Band(b) => match b.block_index(&idx) {
                Some(bi) => b.marginal(bi),
                None => Err(CovError::UnsupportedQuery { op: "marginal_cov" }),
            },
        }
    }

    /// Conditional covariance of a model: its uncertainty with every *other*
    /// parameter held fixed. This is `2 (H_ee)^-1` -- the inverse of the
    /// entity's own information block -- distinct from the marginal (which folds
    /// in the uncertainty of the variables it couples to) and never larger than
    /// it. An entity with no self-information yields infinities.
    ///
    /// Errors as [`marginal_cov`](Self::marginal_cov).
    pub fn conditional_cov<M: Model + ?Sized>(&self, m: &M) -> Result<DMatrix<f64>, CovError> {
        let idx = Self::indices(m);
        let k = idx.len();
        let hb = match &self.backend {
            Backend::Factored { h, .. } => {
                let mut hb = DMatrix::zeros(k, k);
                for (r, &i) in idx.iter().enumerate() {
                    for (c, &j) in idx.iter().enumerate() {
                        hb[(r, c)] = h.get_sym(i, j);
                    }
                }
                hb
            }
            // Same H_ee, read out of the block matrix instead of a CSC.
            Backend::BlockFactored { hb, .. } => {
                let mut hbm = DMatrix::zeros(k, k);
                for (r, &i) in idx.iter().enumerate() {
                    for (c, &j) in idx.iter().enumerate() {
                        hbm[(r, c)] = block_get_sym(hb, i, j);
                    }
                }
                hbm
            }
            // The entity's own diagonal block is exactly H_ee.
            Backend::Band(b) => match b.block_index(&idx) {
                Some(bi) => b.diag[bi].clone(),
                None => return Err(CovError::UnsupportedQuery { op: "conditional_cov" }),
            },
        };
        hb.try_inverse()
            .map(|inv| inv * 2.0)
            .ok_or(CovError::NotPositiveDefinite)
    }

    /// Standard deviations: the square root of the marginal covariance
    /// diagonal, one per scalar parameter of the model.
    ///
    /// Errors as [`marginal_cov`](Self::marginal_cov).
    pub fn std_dev<M: Model + ?Sized>(&self, m: &M) -> Result<Vec<f64>, CovError> {
        let idx = Self::indices(m);
        let block = match &self.backend {
            Backend::Factored { .. } | Backend::BlockFactored { .. } => {
                self.factored_block(&idx, &idx)
            }
            Backend::Band(b) => match b.block_index(&idx) {
                Some(bi) => b.marginal(bi)?,
                None => return Err(CovError::UnsupportedQuery { op: "std_dev" }),
            },
        };
        Ok((0..idx.len()).map(|i| block[(i, i)].sqrt()).collect())
    }

    /// Cross-covariance block between two models (the off-diagonal `A x B`
    /// block of the joint covariance). In tangent coordinates.
    ///
    /// Errors: [`CovError::UnsupportedQuery`] on the
    /// [`CovMode::TriDiagonal`] backend, which stores only the band and has
    /// no off-diagonal blocks -- assemble with `AllMarginals` or `PerQuery`.
    pub fn cross_cov<A: Model + ?Sized, B: Model + ?Sized>(&self, a: &A, b: &B) -> Result<DMatrix<f64>, CovError> {
        let ia = Self::indices(a);
        let ib = Self::indices(b);
        match &self.backend {
            Backend::Factored { .. } | Backend::BlockFactored { .. } => {
                Ok(self.factored_block(&ia, &ib))
            }
            Backend::Band(_) => Err(CovError::UnsupportedQuery { op: "cross_cov" }),
        }
    }

    // Sigma[rows, cols] = 2 (H^-1)[rows, cols] for the factored backends. A
    // selected-inverse hit needs no solve; a miss (out-of-pattern cross block)
    // falls through to a column solve.
    fn factored_block(&self, rows: &[usize], cols: &[usize]) -> DMatrix<f64> {
        // One batched solve for the whole query: the factor is read once for
        // all its columns. There is no selected inverse on this route
        // (AllMarginals stays on the scalar factor), so every query solves.
        if let Backend::BlockFactored { sn, factor, ctx, .. } = &self.backend {
            let k = cols.len();
            let mut e = vec![0.0_f64; self.n * k]; // column-major n x k
            for (c, &j) in cols.iter().enumerate() {
                e[c * self.n + j] = 1.0;
            }
            arael_faer::supernodal::supernodal_solve_multi(
                sn,
                factor,
                &mut e,
                k,
                &mut ctx.borrow_mut(),
            );
            let mut m = DMatrix::zeros(rows.len(), k);
            for c in 0..k {
                for (r, &i) in rows.iter().enumerate() {
                    m[(r, c)] = 2.0 * e[c * self.n + i];
                }
            }
            return m;
        }
        let Backend::Factored { symbolic, l_vals, sel, .. } = &self.backend else {
            // INVARIANT: every call site is inside a factored arm.
            unreachable!("factored_block on a band backend")
        };
        if let Some(s) = sel {
            if let Some(m) = s.try_block(rows, cols) {
                return m;
            }
        }
        let k = cols.len();
        let mut e = vec![0.0_f64; self.n * k]; // column-major n x k
        for (c, &j) in cols.iter().enumerate() {
            e[c * self.n + j] = 1.0;
        }
        solve_cols(symbolic, l_vals, self.n, &mut e, k);
        let mut m = DMatrix::zeros(rows.len(), k);
        for c in 0..k {
            for (r, &i) in rows.iter().enumerate() {
                m[(r, c)] = 2.0 * e[c * self.n + i];
            }
        }
        m
    }
}

// The selected inverse: the entries of Sigma = 2 H^-1 that lie inside the factor
// pattern, stored in the factor's own lower-triangular CSC (permuted ordering),
// values holding the raw H^-1 (the factor of 2 is applied on lookup). `inv` maps
// an original scalar index to its permuted position.
struct SelInv {
    col_ptr: Vec<usize>,
    row_idx: Vec<usize>,
    vals: Vec<f64>,
    inv: Vec<usize>,
}

impl SelInv {
    // Raw H^-1 entry (a, b) in original indices, or None if outside the pattern.
    // Off-diagonal rows within a column are sorted, so binary search.
    fn get(&self, a: usize, b: usize) -> Option<f64> {
        let (i, j) = (self.inv[a], self.inv[b]);
        let (r, c) = if i >= j { (i, j) } else { (j, i) };
        let cs = self.col_ptr[c];
        if r == c {
            return Some(self.vals[cs]);
        }
        self.row_idx[cs + 1..self.col_ptr[c + 1]]
            .binary_search(&r)
            .ok()
            .map(|off| self.vals[cs + 1 + off])
    }

    // Sigma[rows, cols] from the cache, or None if any entry is out of pattern.
    fn try_block(&self, rows: &[usize], cols: &[usize]) -> Option<DMatrix<f64>> {
        let mut m = DMatrix::zeros(rows.len(), cols.len());
        for (cc, &b) in cols.iter().enumerate() {
            for (rr, &a) in rows.iter().enumerate() {
                m[(rr, cc)] = 2.0 * self.get(a, b)?;
            }
        }
        Some(m)
    }
}

// Supernodal block selected inverse: the Takahashi recursion over faer's
// supernodal factor, with BLAS-3 dense-block kernels. Per supernode J with
// diagonal block L_JJ (sj x sj lower-tri) and below block L_RJ (sr x sj, rows R
// = pattern):
//   M    = L_RJ L_JJ^-1                    (triangular solve)
//   S_RJ = -S_RR M                          (matmul; S_RR gathered from later supernodes)
//   S_JJ =  L_JJ^-T L_JJ^-1 - S_RJ^T M      (matmul)
// The result is flattened into the scalar-CSC SelInv used by the query path.
fn selected_inverse_supernodal(symbolic: &fchol::SymbolicCholesky<usize>, l_vals: &[f64], n: usize) -> SelInv {
    use faer::linalg::matmul::matmul;
    use faer::linalg::triangular_solve::{solve_lower_triangular_in_place, solve_upper_triangular_in_place};
    use faer::{Accum, Mat, MatRef, Par};

    let sup = match symbolic.raw() {
        fchol::SymbolicCholeskyRaw::Supernodal(s) => s,
        // INVARIANT: AllMarginals sets FORCE_SUPERNODAL before this runs.
        fchol::SymbolicCholeskyRaw::Simplicial(_) => unreachable!("assemble_covariance forced supernodal"),
    };
    let inv: Vec<usize> = symbolic
        .perm()
        .map(|p| p.arrays().1.to_vec())
        .unwrap_or_else(|| (0..n).collect());

    let ns = sup.n_supernodes();
    let sbegin = sup.supernode_begin();
    let send = sup.supernode_end();
    let vptr = sup.col_ptr_for_val();

    // Global column/row -> owning supernode (supernodes partition [0, n)).
    let mut owner = vec![0usize; n];
    for s in 0..ns {
        for c in sbegin[s]..send[s] {
            owner[c] = s;
        }
    }

    let mut sig = vec![0.0_f64; l_vals.len()]; // Sigma panels, same layout as l_vals
    let mut rpos = vec![usize::MAX; n]; // global row -> local index in the current R

    for s in (0..ns).rev() {
        let start = sbegin[s];
        let sj = send[s] - start;
        let pattern = sup.supernode(s).pattern();
        let sr = pattern.len();
        let ld = sj + sr;
        let v0 = vptr[s];

        let lpanel = MatRef::from_column_major_slice(&l_vals[v0..vptr[s + 1]], ld, sj);
        let (l_jj, l_rj) = lpanel.split_at_row(sj);

        // Gather Sigma_RR (sr x sr symmetric) from the already-computed later
        // supernodes: read column b (b in R) from its owning supernode's panel.
        for (li, &r) in pattern.iter().enumerate() {
            rpos[r] = li;
        }
        let mut srr = Mat::<f64>::zeros(sr, sr);
        for (lb, &b) in pattern.iter().enumerate() {
            let sb = owner[b];
            let sb_start = sbegin[sb];
            let sb_sj = send[sb] - sb_start;
            let sb_pat = sup.supernode(sb).pattern();
            let sb_ld = sb_sj + sb_pat.len();
            let base = vptr[sb] + (b - sb_start) * sb_ld;
            for ro in 0..sb_ld {
                let ga = if ro < sb_sj { sb_start + ro } else { sb_pat[ro - sb_sj] };
                let la = rpos[ga];
                if la != usize::MAX {
                    let val = sig[base + ro];
                    srr[(la, lb)] = val;
                    srr[(lb, la)] = val;
                }
            }
        }
        for &r in pattern {
            rpos[r] = usize::MAX;
        }

        // M = L_RJ L_JJ^-1: solve L_JJ^T M^T = L_RJ^T.
        let mut m = l_rj.to_owned();
        if sr > 0 {
            solve_upper_triangular_in_place(l_jj.transpose(), m.as_mut().transpose_mut(), Par::Seq);
        }

        // Sigma_RJ = -Sigma_RR M.
        let mut srj = Mat::<f64>::zeros(sr, sj);
        if sr > 0 {
            matmul(srj.as_mut(), Accum::Replace, srr.as_ref(), m.as_ref(), -1.0, Par::Seq);
        }

        // Sigma_JJ = L_JJ^-T L_JJ^-1 - Sigma_RJ^T M.
        let mut linv = Mat::<f64>::identity(sj, sj);
        solve_lower_triangular_in_place(l_jj, linv.as_mut(), Par::Seq); // linv = L_JJ^-1
        let mut sjj = Mat::<f64>::zeros(sj, sj);
        matmul(sjj.as_mut(), Accum::Replace, linv.as_ref().transpose(), linv.as_ref(), 1.0, Par::Seq);
        if sr > 0 {
            matmul(sjj.as_mut(), Accum::Add, srj.as_ref().transpose(), m.as_ref(), -1.0, Par::Seq);
        }

        // Write the Sigma panel: top = Sigma_JJ (full), bottom = Sigma_RJ.
        for lc in 0..sj {
            let base = v0 + lc * ld;
            for ro in 0..sj {
                sig[base + ro] = sjj[(ro, lc)];
            }
            for ro in 0..sr {
                sig[base + sj + ro] = srj[(ro, lc)];
            }
        }
    }

    // Flatten the Sigma panels into a lower-triangular scalar CSC: per column,
    // the diagonal first, then the block rows below it, then the pattern rows
    // (already ascending).
    let mut col_ptr = vec![0usize; n + 1];
    for s in 0..ns {
        let start = sbegin[s];
        let sj = send[s] - start;
        let sr = sup.supernode(s).pattern().len();
        for lc in 0..sj {
            col_ptr[start + lc + 1] = 1 + (sj - 1 - lc) + sr;
        }
    }
    for c in 0..n {
        col_ptr[c + 1] += col_ptr[c];
    }
    let mut row_idx = vec![0usize; col_ptr[n]];
    let mut vals = vec![0.0_f64; col_ptr[n]];
    for s in 0..ns {
        let start = sbegin[s];
        let sj = send[s] - start;
        let pattern = sup.supernode(s).pattern();
        let sr = pattern.len();
        let ld = sj + sr;
        let v0 = vptr[s];
        for lc in 0..sj {
            let c = start + lc;
            let base = v0 + lc * ld;
            let mut p = col_ptr[c];
            row_idx[p] = c;
            vals[p] = sig[base + lc];
            p += 1;
            for ro in (lc + 1)..sj {
                row_idx[p] = start + ro;
                vals[p] = sig[base + ro];
                p += 1;
            }
            for (i, &r) in pattern.iter().enumerate() {
                row_idx[p] = r;
                vals[p] = sig[base + sj + i];
                p += 1;
            }
        }
    }

    SelInv { col_ptr, row_idx, vals, inv }
}

// Solve H X = E in place (E is column-major n x k), leaving X = H^-1 E.
fn solve_cols(symbolic: &fchol::SymbolicCholesky<usize>, l_vals: &[f64], n: usize, e: &mut [f64], k: usize) {
    let llt = fchol::LltRef::new(symbolic, l_vals);
    let req = symbolic.solve_in_place_scratch::<f64>(k, faer::Par::Seq);
    let mut mem: Vec<MaybeUninit<u8>> = vec![MaybeUninit::uninit(); req.unaligned_bytes_required()];
    let stack = faer::dyn_stack::MemStack::new(&mut mem);
    let rhs = faer::mat::MatMut::from_column_major_slice_mut(e, n, k);
    llt.solve_in_place_with_conj(faer::Conj::No, rhs, faer::Par::Seq, stack);
}

// The block graph over the model's entity partition. `None` when there is no
// block structure to work with (a hand-built problem), or when the blocks are
// one scalar wide and there is nothing to divide out -- the block graph is
// then the scalar graph and building it would only add cost.
fn block_graph(
    spans: &[(u32, u32)],
    cells: &[(u32, u32)],
    n: usize,
) -> Option<(Vec<usize>, arael_faer::bsc::SymbolicSparseBlockColMat<arael_faer::SparseIndex>)> {
    if spans.is_empty() || cells.is_empty() {
        return None;
    }
    let part = crate::simple_lm::block_partition_from_spans(spans, n);
    if part.len() - 1 >= n {
        return None;
    }
    let part_idx: Vec<arael_faer::SparseIndex> = part.iter().map(|&p| p as _).collect();
    let (hsym, _) = arael_faer::bsc::SymbolicSparseBlockColMat::from_scalar_coords(
        part_idx.clone(),
        part_idx,
        cells.len(),
        |k| (cells[k].0 as usize, cells[k].1 as usize),
    );
    Some((part, hsym))
}

// The block elimination order `ordering` asks for, and the symbolic that
// prices it -- the caller factorizes through that symbolic rather than
// rebuilding it. `Natural` gets no order.
//
// Auto prices minimum degree against nested dissection by the flops each
// factor would take and keeps the cheaper, the determination the solver makes
// for a solve: minimum degree wins on a chain, dissection on a trajectory that
// revisits, where poses far apart in the ordering are coupled.
fn block_order(
    ordering: CovOrdering,
    hsym: &arael_faer::bsc::SymbolicSparseBlockColMat<arael_faer::SparseIndex>,
    params: &arael_faer::supernodal::SupernodalParams,
) -> (Option<Vec<usize>>, Option<arael_faer::supernodal::SupernodalSymbolic>) {
    use arael_faer::supernodal as sn;
    let build = |o: Option<&[usize]>| sn::SupernodalSymbolic::new(hsym, o, params).ok();
    match ordering {
        CovOrdering::Amd => {
            let o = sn::amd_block_order(hsym);
            let s = build(Some(&o));
            (Some(o), s)
        }
        CovOrdering::NestedDissection => {
            let o = sn::nd_block_order(hsym);
            let s = build(Some(&o));
            (Some(o), s)
        }
        CovOrdering::Natural => (None, build(None)),
        // Minimum degree first, so it keeps a tie: it is the cheaper of the
        // two to build.
        CovOrdering::Auto => {
            let candidates = vec![sn::amd_block_order(hsym), sn::nd_block_order(hsym)];
            match sn::cheapest_block_order(hsym, params, candidates) {
                Some(c) => (Some(c.order), Some(c.symbolic)),
                None => (Some(sn::amd_block_order(hsym)), None),
            }
        }
    }
}

// A block ordering as a scalar permutation faer can take. `None` leaves the
// caller on its own scalar ordering.
//
// Returned as (forward, inverse) in faer's convention: forward[k] is the
// column eliminated k-th, inverse[i] where column i ended up.
fn block_perm(
    ordering: CovOrdering,
    spans: &[(u32, u32)],
    cells: &[(u32, u32)],
    n: usize,
) -> Option<(Vec<usize>, Vec<usize>)> {
    let (part, hsym) = block_graph(spans, cells, n)?;
    let (order, _) =
        block_order(ordering, &hsym, &arael_faer::supernodal::SupernodalParams::default());
    let order = order?;

    // Block order -> scalar order: each block contributes its own columns, in
    // their natural order, at the position the block landed in.
    let mut forward: Vec<usize> = Vec::with_capacity(n);
    for &b in &order {
        forward.extend(part[b]..part[b + 1]);
    }
    debug_assert_eq!(forward.len(), n, "the block partition did not cover every column");
    let mut inverse = vec![0usize; n];
    for (k, &c) in forward.iter().enumerate() {
        inverse[c] = k;
    }
    Some((forward, inverse))
}

// Assemble and factorize in block form. `Ok(None)` when the model's blocks
// turn out to be one scalar wide, where the block route has nothing to divide
// out and the scalar one is the better answer.
fn block_assemble<T: Float, M: Covariance<T> + ?Sized>(
    m: &mut M,
    params: &[T],
    grad: &mut [T],
    spans: &[(u32, u32)],
    cells: &[(u32, u32)],
    n: usize,
    opts: &CovOptions,
) -> Result<Option<CovAssembly>, CovError> {
    use arael_faer::supernodal as sn;

    let Some((_, hsym)) = block_graph(spans, cells, n) else {
        return Ok(None);
    };

    // One value slot per scalar entry the block traversal emits, in that
    // traversal's order -- what the indexed assembly writes through.
    let mut resolver = arael_faer::bsc::PositionResolver::new(&hsym);
    let mut positions = Vec::new();
    LmProblem::bind_hessian_positions(
        m,
        &mut crate::model::HessianBinder::Tiled(&mut |i, j| {
            resolver.resolve_tile(i as usize, j as usize)
        }),
        &mut positions,
    );

    // Assemble at the model's precision, then carry the values to f64: a
    // covariance is computed in f64 whatever the model is.
    let mut vals_t = vec![T::zero(); hsym.val_count()];
    m.calc_grad_hessian_sparse_indexed(params, grad, &mut vals_t, &positions);
    let vals: Vec<f64> = vals_t.iter().map(|&x| x.to_f64().unwrap_or(f64::NAN)).collect();

    let params = sn::SupernodalParams::default();
    let (_, symbolic) = block_order(opts.ordering, &hsym, &params);
    let symbolic = symbolic.ok_or(CovError::NotPositiveDefinite)?;

    let hb = arael_faer::bsc::SparseBlockColMat::new(hsym, vals);
    let mut factor = vec![0.0_f64; symbolic.factor_val_count()];
    let mut ctx = sn::SupernodalContext::new();
    sn::supernodal_factorize(&symbolic, &hb, &mut factor, &mut ctx, faer::Par::Seq)
        .map_err(|_| CovError::NotPositiveDefinite)?;

    Ok(Some(CovAssembly {
        n,
        backend: Backend::BlockFactored { sn: symbolic, factor, hb, ctx: std::cell::RefCell::new(ctx) },
    }))
}

#[cfg(test)]
mod tests {
    use super::{block_perm, CovOrdering};

    #[test]
    fn block_perm_is_a_permutation_that_keeps_blocks_whole() {
        // Four 3-wide blocks in a chain: diagonal cells plus 0-1, 1-2, 2-3.
        let spans = [(0u32, 3u32), (3, 3), (6, 3), (9, 3)];
        let cells = [(0u32, 0u32), (3, 3), (6, 6), (9, 9), (0, 3), (3, 6), (6, 9)];
        let (fwd, inv) = block_perm(CovOrdering::Auto, &spans, &cells, 12).expect("3-wide blocks divide out");

        assert_eq!(fwd.len(), 12);
        let mut seen = vec![false; 12];
        for (k, &c) in fwd.iter().enumerate() {
            assert!(!seen[c], "column {c} appears twice");
            seen[c] = true;
            assert_eq!(inv[c], k, "inverse disagrees at column {c}");
        }

        // A block is eliminated whole: its columns stay consecutive and in
        // their own order, whatever position the block landed in.
        for k in (0..12).step_by(3) {
            assert_eq!(fwd[k] % 3, 0, "block does not start on a boundary at {k}");
            assert_eq!(fwd[k + 1], fwd[k] + 1);
            assert_eq!(fwd[k + 2], fwd[k] + 2);
        }
    }

    #[test]
    fn one_scalar_per_block_declines() {
        // The block graph is then the scalar graph: nothing to divide out, and
        // building it would only add to what scalar AMD already costs.
        let spans = [(0u32, 1u32), (1, 1), (2, 1)];
        let cells = [(0u32, 0u32), (1, 1), (2, 2), (0, 1)];
        assert!(block_perm(CovOrdering::Auto, &spans, &cells, 3).is_none());
    }

    #[test]
    fn no_block_structure_declines() {
        assert!(block_perm(CovOrdering::Auto, &[], &[(0, 0)], 4).is_none());
        assert!(block_perm(CovOrdering::Auto, &[(0, 4)], &[], 4).is_none());
    }
}

// Symmetric scalar lookup into a block matrix holding the upper triangle:
// H[i,j] from the block that covers (i, j), zero when that block is absent.
fn block_get_sym(
    hb: &arael_faer::bsc::SparseBlockColMat<arael_faer::SparseIndex, f64>,
    i: usize,
    j: usize,
) -> f64 {
    let (lo, hi) = if i <= j { (i, j) } else { (j, i) };
    let sym = hb.symbolic();
    let part = sym.col_part();
    // Which block owns a scalar column: the partition is ascending, so the
    // block is the last boundary at or below it.
    let blk_of = |c: usize| match part.binary_search(&(c as arael_faer::SparseIndex)) {
        Ok(b) => b,
        Err(b) => b - 1,
    };
    let (br, bc) = (blk_of(lo), blk_of(hi));
    match hb.get_block(br, bc) {
        Some(m) => m[(lo - part[br] as usize, hi - part[bc] as usize)],
        None => 0.0,
    }
}

// Upper-triangle CSC symmetric lookup: H[i,j] with H stored for i <= j.
impl CscMatrix<f64> {
    fn get_sym(&self, i: usize, j: usize) -> f64 {
        let (lo, hi) = if i <= j { (i, j) } else { (j, i) };
        let start = self.col_ptr[hi];
        let end = self.col_ptr[hi + 1];
        for p in start..end {
            if self.row_idx[p] as usize == lo {
                return self.vals[p];
            }
        }
        0.0
    }
}

// Half-bandwidth kd for a block-tridiagonal partition: the widest coupling is
// the first parameter of a block to the last of the next (w_i + w_{i+1} - 1);
// within a single block it is w_i - 1.
fn band_half_width(spans: &[(usize, usize)]) -> usize {
    let mut kd = spans.iter().map(|&(_, w)| w.saturating_sub(1)).max().unwrap_or(0);
    for pair in spans.windows(2) {
        kd = kd.max(pair[0].1 + pair[1].1 - 1);
    }
    kd
}

// Build the block-tridiagonal representation from the assembled upper-band
// buffer (LAPACK layout: A[i,j], i <= j, at band[(kd + i - j) + j*(kd+1)]).
// Extract the diagonal and first-off-diagonal blocks (a nonzero outside them is
// not block-tridiagonal), then run the forward Schur pass.
fn build_band<T: Float>(band: &[T], kd: usize, n: usize, spans: &[(usize, usize)]) -> Result<BandData, CovError> {
    let nb = spans.len();
    let ldab = kd + 1;
    let mut coord_block = vec![usize::MAX; n];
    for (bi, &(o, w)) in spans.iter().enumerate() {
        for c in o..o + w {
            coord_block[c] = bi;
        }
    }

    let mut diag: Vec<DMatrix<f64>> = spans.iter().map(|&(_, w)| DMatrix::zeros(w, w)).collect();
    let mut off: Vec<DMatrix<f64>> = (0..nb.saturating_sub(1))
        .map(|i| DMatrix::zeros(spans[i].1, spans[i + 1].1))
        .collect();

    // Walk the band (i <= j). A structurally absent coupling is exactly zero.
    for j in 0..n {
        let bb = coord_block[j];
        for p in 0..=kd {
            let Some(i) = (j + p).checked_sub(kd) else { continue }; // i = j - kd + p
            let val = band[p + j * ldab].to_f64().unwrap_or(f64::NAN);
            if val == 0.0 {
                continue;
            }
            let ba = coord_block[i];
            if ba == bb {
                let (lo, hi) = (i - spans[ba].0, j - spans[bb].0);
                diag[ba][(lo, hi)] = val;
                diag[ba][(hi, lo)] = val;
            } else if bb - ba == 1 {
                off[ba][(i - spans[ba].0, j - spans[bb].0)] = val;
            } else {
                return Err(CovError::NotTriDiagonal);
            }
        }
    }

    // Forward Schur: S_0 = D_0, S_i = D_i - B_{i-1} S_{i-1}^-1 B_{i-1}^T, where
    // B_{i-1}^T = off[i-1] = H[i-1, i].
    let mut fwd: Vec<DMatrix<f64>> = Vec::with_capacity(nb);
    fwd.push(diag[0].clone());
    for i in 1..nb {
        let s_prev_inv = fwd[i - 1].clone().try_inverse().ok_or(CovError::NotPositiveDefinite)?;
        let b = &off[i - 1];
        fwd.push(&diag[i] - b.transpose() * &s_prev_inv * b);
    }

    Ok(BandData { spans: spans.to_vec(), diag, off, fwd, bwd: OnceCell::new() })
}

/// Post-solve covariance. Implemented for every `#[arael(root)]` model. Call
/// with the trait in scope after the solution has been written into the model
/// (i.e. after a `solve_*` / `deserialize`), since it reads the model's current
/// parameters as the linearization point.
pub trait Covariance<T: Float>: LmProblem<T> + RootProblem<T> + Model {
    /// Re-assemble `H` at the current parameters and prepare it for querying, per
    /// `mode`. The dense inverse is never formed. `Err` if `H` is singular, or
    /// (for [`CovMode::TriDiagonal`]) not block-tridiagonal.
    fn assemble_covariance(&mut self, mode: CovMode) -> Result<CovAssembly, CovError> {
        self.assemble_covariance_with(mode, &CovOptions::auto())
    }

    /// [`assemble_covariance`](Self::assemble_covariance) with the assembly
    /// spelled out instead of left to the defaults -- see [`CovOptions`].
    /// The covariance is the same either way; the options decide what it
    /// costs to produce.
    fn assemble_covariance_with(
        &mut self,
        mode: CovMode,
        opts: &CovOptions,
    ) -> Result<CovAssembly, CovError> {
        let mut params: Vec<T> = Vec::new();
        self.serialize(&mut params);
        let n = params.len();
        if n == 0 {
            return Err(CovError::Empty);
        }
        let mut grad = vec![T::zero(); n];

        // TriDiagonal: assemble straight into the band (no COO/CSC), extract the
        // block-tridiagonal structure, run the forward Schur pass. A coupling
        // beyond the band makes calc_grad_hessian_band fail -> not tridiagonal.
        if mode == CovMode::TriDiagonal {
            let mut spans_raw: Vec<(u32, u32)> = Vec::new();
            self.collect_param_blocks(&mut spans_raw);
            let mut spans: Vec<(usize, usize)> =
                spans_raw.iter().map(|&(o, w)| (o as usize, w as usize)).collect();
            spans.sort_by_key(|&(o, _)| o);
            let kd = band_half_width(&spans);
            let mut band = vec![T::zero(); (kd + 1) * n];
            self.calc_grad_hessian_band(&params, &mut grad, &mut band, kd)
                .map_err(|_| CovError::NotTriDiagonal)?;
            let bd = build_band(&band, kd, n, &spans)?;
            return Ok(CovAssembly { n, backend: Backend::Band(bd) });
        }

        // The model's block structure, when it has one. Both the block route
        // and Auto's block ordering are built on it.
        let mut spans: Vec<(u32, u32)> = Vec::new();
        self.collect_param_blocks(&mut spans);
        let mut cells: Vec<(u32, u32)> = Vec::new();
        LmProblem::collect_hessian_cells(self, &mut cells);
        let blocked = !spans.is_empty() && !cells.is_empty();

        // The block route: assemble straight into block form, order over the
        // block graph, factorize with the block supernodal Cholesky. No COO,
        // no scalar CSC, no scalar symbolic. AllMarginals is excluded because
        // its selected inverse reads faer's supernode panels.
        let want_block = match opts.block_supernodal {
            BlockSupernodalMode::Auto | BlockSupernodalMode::Always => true,
            BlockSupernodalMode::Never => false,
        };
        if want_block && blocked && mode == CovMode::PerQuery {
            if let Some(a) = block_assemble(self, &params, &mut grad, &spans, &cells, n, opts)? {
                return Ok(a);
            }
        }

        let mut coo = CooMatrix::new(n);
        self.calc_grad_hessian_sparse(&params, &mut grad, &mut coo);
        let csc_t = coo.to_csc().map_err(|_| CovError::NotPositiveDefinite)?;

        // Upper-triangle CSC of H in f64 (covariance is computed in f64
        // regardless of the model's precision).
        let h = CscMatrix::<f64> {
            n,
            col_ptr: csc_t.col_ptr.clone(),
            row_idx: csc_t.row_idx.clone(),
            vals: csc_t.vals.iter().map(|&x| x.to_f64().unwrap_or(f64::NAN)).collect(),
            diag_pos: csc_t.diag_pos.clone(),
        };

        // faer wants usize row indices.
        let row_usize: Vec<usize> = h.row_idx.iter().map(|&r| r as usize).collect();
        let sym_ref = faer::sparse::SymbolicSparseColMatRef::new_checked(n, n, &h.col_ptr, None, &row_usize);
        // AllMarginals forces a supernodal factor: the block (BLAS-3) selected
        // inverse reads its dense supernode panels. PerQuery leaves faer's AUTO
        // threshold (it may pick either; the solve handles both).
        let mut chol_params = fchol::CholeskySymbolicParams::default();
        if mode == CovMode::AllMarginals {
            chol_params.supernodal_flop_ratio_threshold =
                faer::sparse::linalg::SupernodalThreshold::FORCE_SUPERNODAL;
        }
        // Auto orders over the model's block graph when there is one. The
        // permutation has to outlive the symbolic factorization that borrows it.
        let block_perm = match opts.ordering {
            CovOrdering::Auto | CovOrdering::NestedDissection => {
                block_perm(opts.ordering, &spans, &cells, n)
            }
            CovOrdering::Amd | CovOrdering::Natural => None,
        };
        let ordering = match (&block_perm, opts.ordering) {
            (Some((fwd, inv)), _) => fchol::SymmetricOrdering::Custom(
                faer::perm::PermRef::new_checked(fwd, inv, n),
            ),
            (None, CovOrdering::Natural) => fchol::SymmetricOrdering::Identity,
            // Auto with no block structure to use falls through to scalar AMD.
            (None, _) => fchol::SymmetricOrdering::Amd,
        };
        let symbolic = fchol::factorize_symbolic_cholesky(
            sym_ref,
            faer::Side::Upper,
            ordering,
            chol_params,
        )
        .map_err(|_| CovError::NotPositiveDefinite)?;

        let mut l_vals = vec![0.0_f64; symbolic.len_val()];
        let factor_req = symbolic.factorize_numeric_llt_scratch::<f64>(faer::Par::Seq, faer::Spec::default());
        let mut factor_mem: Vec<MaybeUninit<u8>> = vec![MaybeUninit::uninit(); factor_req.unaligned_bytes_required()];
        let stack = faer::dyn_stack::MemStack::new(&mut factor_mem);
        let mat_ref = faer::sparse::SparseColMatRef::new(sym_ref, &h.vals);
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
            .map_err(|_| CovError::NotPositiveDefinite)?;

        let sel = if mode == CovMode::AllMarginals {
            Some(selected_inverse_supernodal(&symbolic, &l_vals, n))
        } else {
            None
        };

        Ok(CovAssembly { n, backend: Backend::Factored { symbolic, l_vals, h, sel } })
    }
}

impl<T: Float, P: LmProblem<T> + RootProblem<T> + Model> Covariance<T> for P {}
