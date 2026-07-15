//! Parameter covariance recovery.
//!
//! At the solution the parameter covariance is `Sigma = (J^T J)^-1`. arael
//! assembles `H = 2 J^T J` (the cost is `sum r^2`, so `add_residual` carries a
//! factor of 2), hence `Sigma = 2 H^-1`. Because the rotation parameters are
//! already minimal 3-DOF retractions, `Sigma` is in local tangent coordinates
//! -- no manifold projection is needed.
//!
//! [`Covariance::assemble_covariance`] factors `H` sparsely at the solution and
//! keeps the factorization. Each query then solves for only the requested
//! entity's columns, so the dense inverse is never formed -- it scales to large
//! problems. Query per-entity blocks through the entity itself: any `Model`
//! reports its parameter span via `collect_param_blocks`.
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
//! [`CovMode`] chosen at assembly picks the strategy:
//! - [`CovMode::PerQuery`] factors `H` and answers each query by solving for its
//!   columns (faer picks supernodal where it pays). Best for a few entities.
//! - [`CovMode::AllMarginals`] also runs a selected inverse up front -- the
//!   Takahashi recursion over a simplicial factor, computing every covariance
//!   entry inside the factor's sparsity pattern in one `O(fill)` pass -- so every
//!   marginal and coupled cross block becomes a lookup. Best for many/all
//!   marginals.
//!
//! Either way, **conditional** covariance (`conditional_cov`) just inverts the
//! entity's own `H_ee` block (`O(dof^3)`, no factor solve), and out-of-pattern
//! cross blocks fall back to a solve.

use crate::model::Model;
use crate::simple_lm::{CooMatrix, CscMatrix, LmProblem, RootProblem};
use crate::utils::Float;
use faer::sparse::linalg::cholesky as fchol;
use nalgebra::DMatrix;
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
}

impl std::fmt::Display for CovError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CovError::NotPositiveDefinite => write!(f,
                "Hessian is not positive definite (unfixed gauge? add an anchor or a prior)"),
            CovError::Empty => write!(f, "model has no optimizable parameters"),
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
    /// Also compute the selected inverse up front (one `O(fill)` pass over a
    /// simplicial factor), so every marginal and coupled cross block is a lookup.
    /// Choose this when you need many or all marginals (e.g. an ellipse per pose).
    AllMarginals,
}

/// A factored covariance at the solution. Build it with
/// [`Covariance::assemble_covariance`], then query per-entity blocks. An owned
/// value: querying does not borrow the problem, so
/// `let cov = p.assemble_covariance()?;` followed by
/// `cov.marginal_cov(&p.landmarks[0])` borrow-checks.
///
/// Holds the sparse Cholesky factor of `H` and the CSC `H` itself (for
/// conditional queries). Covariance blocks are solved for on demand; assembling
/// with [`CovMode::AllMarginals`] fills a selected-inverse cache so marginals
/// become lookups.
pub struct CovAssembly {
    n: usize,
    symbolic: fchol::SymbolicCholesky<usize>,
    l_vals: Vec<f64>,
    h: CscMatrix<f64>,
    sel: Option<SelInv>,
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
    fn get(&self, a: usize, b: usize) -> Option<f64> {
        let (i, j) = (self.inv[a], self.inv[b]);
        let (r, c) = if i >= j { (i, j) } else { (j, i) };
        let cs = self.col_ptr[c];
        if r == c {
            return Some(self.vals[cs]);
        }
        // Off-diagonal row indices within a column are unsorted -> linear scan.
        for p in (cs + 1)..self.col_ptr[c + 1] {
            if self.row_idx[p] == r {
                return Some(self.vals[p]);
            }
        }
        None
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

// Raw H^-1 entry (r, c), r >= c, from a partially built selected inverse in the
// factor's own indexing. Every entry the recursion asks for is in the pattern.
fn sel_read(vals: &[f64], col_ptr: &[usize], row_idx: &[usize], r: usize, c: usize) -> f64 {
    let cs = col_ptr[c];
    if r == c {
        return vals[cs];
    }
    for p in (cs + 1)..col_ptr[c + 1] {
        if row_idx[p] == r {
            return vals[p];
        }
    }
    panic!("selected inverse: entry ({r}, {c}) outside factor pattern");
}

impl CovAssembly {
    /// Number of optimized scalar parameters.
    pub fn dim(&self) -> usize {
        self.n
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

    // Solve H X = E in place (E is column-major n x k), leaving X = H^-1 E.
    fn solve_cols(&self, e: &mut [f64], k: usize) {
        let llt = fchol::LltRef::new(&self.symbolic, &self.l_vals);
        let req = self.symbolic.solve_in_place_scratch::<f64>(k, faer::Par::Seq);
        let mut mem: Vec<MaybeUninit<u8>> = vec![MaybeUninit::uninit(); req.unaligned_bytes_required()];
        let stack = faer::dyn_stack::MemStack::new(&mut mem);
        let rhs = faer::mat::MatMut::from_column_major_slice_mut(e, self.n, k);
        llt.solve_in_place_with_conj(faer::Conj::No, rhs, faer::Par::Seq, stack);
    }

    // The covariance sub-block Sigma[rows, cols] = 2 (H^-1)[rows, cols].
    fn sigma_block(&self, rows: &[usize], cols: &[usize]) -> DMatrix<f64> {
        // Selected-inverse cache: a hit needs no solve. A miss (out-of-pattern
        // cross block) falls through to a column solve.
        if let Some(sel) = &self.sel {
            if let Some(m) = sel.try_block(rows, cols) {
                return m;
            }
        }
        // Solve for the requested columns of H^-1, then read the rows.
        let k = cols.len();
        let mut e = vec![0.0_f64; self.n * k]; // column-major n x k
        for (c, &j) in cols.iter().enumerate() {
            e[c * self.n + j] = 1.0;
        }
        self.solve_cols(&mut e, k);
        let mut m = DMatrix::zeros(rows.len(), k);
        for c in 0..k {
            for (r, &i) in rows.iter().enumerate() {
                m[(r, c)] = 2.0 * e[c * self.n + i];
            }
        }
        m
    }

    /// Marginal covariance of a model: one entity for its own covariance, or a
    /// whole collection for the joint over all its entities. In tangent
    /// coordinates.
    pub fn marginal_cov<M: Model + ?Sized>(&self, m: &M) -> DMatrix<f64> {
        let idx = Self::indices(m);
        self.sigma_block(&idx, &idx)
    }

    /// Conditional covariance of a model: its uncertainty with every *other*
    /// parameter held fixed. This is `2 (H_ee)^-1` -- the inverse of the
    /// entity's own information block -- distinct from the marginal (which folds
    /// in the uncertainty of the variables it couples to) and never larger than
    /// it. Useful when a reference frame is already pinned elsewhere: an entity
    /// that shares no factor with its peers (e.g. a landmark, given the poses)
    /// has a block-diagonal `H_ee`, so this is its pose-fixed uncertainty. An
    /// entity with no self-information yields infinities.
    pub fn conditional_cov<M: Model + ?Sized>(&self, m: &M) -> DMatrix<f64> {
        let idx = Self::indices(m);
        let k = idx.len();
        let mut hb = DMatrix::zeros(k, k);
        for (r, &i) in idx.iter().enumerate() {
            for (c, &j) in idx.iter().enumerate() {
                hb[(r, c)] = self.h.get_sym(i, j);
            }
        }
        hb.try_inverse()
            .map(|inv| inv * 2.0)
            .unwrap_or_else(|| DMatrix::from_element(k, k, f64::INFINITY))
    }

    /// Standard deviations: the square root of the marginal covariance
    /// diagonal, one per scalar parameter of the model.
    pub fn std_dev<M: Model + ?Sized>(&self, m: &M) -> Vec<f64> {
        let idx = Self::indices(m);
        let block = self.sigma_block(&idx, &idx);
        (0..idx.len()).map(|i| block[(i, i)].sqrt()).collect()
    }

    /// Cross-covariance block between two models (the off-diagonal `A x B`
    /// block of the joint covariance).
    pub fn cross_cov<A: Model + ?Sized, B: Model + ?Sized>(&self, a: &A, b: &B) -> DMatrix<f64> {
        let ia = Self::indices(a);
        let ib = Self::indices(b);
        self.sigma_block(&ia, &ib)
    }

    // Selected inverse: the Takahashi recursion runs backward over the
    // simplicial factor, filling every Sigma entry inside the factor pattern in
    // one O(fill) pass. Requires a simplicial factor (AllMarginals forces one).
    fn build_selected_inverse(&mut self) {
        let n = self.n;
        // Factor pattern (permuted ordering). First row of each column is the
        // diagonal; the values in `l_vals` align 1:1 with these row indices.
        let (col_ptr, row_idx) = match self.symbolic.raw() {
            fchol::SymbolicCholeskyRaw::Simplicial(s) => (s.col_ptr().to_vec(), s.row_idx().to_vec()),
            fchol::SymbolicCholeskyRaw::Supernodal(_) => {
                unreachable!("assemble_covariance forces a simplicial factorization")
            }
        };
        // original index -> permuted position.
        let inv: Vec<usize> = self
            .symbolic
            .perm()
            .map(|p| p.arrays().1.to_vec())
            .unwrap_or_else(|| (0..n).collect());

        // Sigma_ij for i >= j inside the pattern, from H = L L^T:
        //   Sigma_ij = -(1/L_jj) sum_{k>j} L_kj Sigma_ik        (i > j)
        //   Sigma_jj = 1/L_jj^2 - (1/L_jj) sum_{k>j} L_kj Sigma_kj
        // Both indices of every Sigma_ik touched exceed j, so it is already done.
        let mut vals = vec![0.0_f64; self.l_vals.len()];
        {
            let l = &self.l_vals;
            for j in (0..n).rev() {
                let (cs, ce) = (col_ptr[j], col_ptr[j + 1]);
                let inv_ljj = 1.0 / l[cs];
                for p in (cs + 1)..ce {
                    let i = row_idx[p];
                    let mut acc = 0.0;
                    for q in (cs + 1)..ce {
                        let k = row_idx[q];
                        let (r, c) = if i >= k { (i, k) } else { (k, i) };
                        acc += l[q] * sel_read(&vals, &col_ptr, &row_idx, r, c);
                    }
                    vals[p] = -inv_ljj * acc;
                }
                let mut dacc = 0.0;
                for q in (cs + 1)..ce {
                    dacc += l[q] * vals[q];
                }
                vals[cs] = inv_ljj * inv_ljj - inv_ljj * dacc;
            }
        }
        self.sel = Some(SelInv { col_ptr, row_idx, vals, inv });
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

/// Post-solve covariance. Implemented for every `#[arael(root)]` model. Call
/// with the trait in scope after the solution has been written into the model
/// (i.e. after a `solve_*` / `deserialize`), since it reads the model's current
/// parameters as the linearization point.
pub trait Covariance<T: Float>: LmProblem<T> + RootProblem<T> {
    /// Re-assemble `H` at the current parameters, factor it sparsely, and keep
    /// the factorization for querying. The dense inverse is never formed. `mode`
    /// picks the strategy: [`CovMode::PerQuery`] solves each query on demand;
    /// [`CovMode::AllMarginals`] also computes the selected inverse up front so
    /// marginals are lookups. `Err` if `H` is singular.
    fn assemble_covariance(&mut self, mode: CovMode) -> Result<CovAssembly, CovError> {
        let mut params: Vec<T> = Vec::new();
        self.serialize(&mut params);
        let n = params.len();
        if n == 0 {
            return Err(CovError::Empty);
        }
        let mut grad = vec![T::zero(); n];
        let mut coo = CooMatrix::new(n);
        self.calc_grad_hessian_sparse(&params, &mut grad, &mut coo);
        let csc_t = coo.to_csc();

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
        // AllMarginals forces a simplicial factor: the selected inverse reads a
        // plain column-major CSC `L` directly. PerQuery leaves faer's AUTO
        // threshold, so it may pick a supernodal factor (faster dense-block
        // solves) for the few-columns-at-a-time queries.
        let mut chol_params = fchol::CholeskySymbolicParams::default();
        if mode == CovMode::AllMarginals {
            chol_params.supernodal_flop_ratio_threshold =
                faer::sparse::linalg::SupernodalThreshold::FORCE_SIMPLICIAL;
        }
        let symbolic = fchol::factorize_symbolic_cholesky(
            sym_ref,
            faer::Side::Upper,
            fchol::SymmetricOrdering::Amd,
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

        let mut cov = CovAssembly { n, symbolic, l_vals, h, sel: None };
        if mode == CovMode::AllMarginals {
            cov.build_selected_inverse();
        }
        Ok(cov)
    }
}

impl<T: Float, P: LmProblem<T> + RootProblem<T>> Covariance<T> for P {}
