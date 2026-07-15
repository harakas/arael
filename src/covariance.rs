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
//! use arael::covariance::Covariance;
//! path.solve_sparse(&cfg);              // solution written back into the model
//! let cov = path.assemble_covariance()?;
//! let lm0  = cov.marginal_cov(&path.landmarks[0]);        // one landmark, solves for its columns
//! let lmc  = cov.conditional_cov(&path.landmarks[0]);     // its own info block, others fixed
//! let x    = cov.cross_cov(&path.poses[0], &path.landmarks[3]);
//! ```
//!
//! Access patterns and their costs:
//! - **conditional** of an entity (`conditional_cov`) -- invert its own `H_ee`
//!   block, `O(dof^3)`, no factor solve.
//! - **a few marginals** (`marginal_cov` per entity) -- one `O(fill)` solve each.
//! - **all/many marginals** -- a selected inverse (Takahashi / Meurant / Schur)
//!   computing every diagonal block in one `O(fill)` pass. Future work (COV.md);
//!   it will land as an opt-in precompute behind the same query API, not a
//!   second entry point.

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

/// A factored covariance at the solution. Build it with
/// [`Covariance::assemble_covariance`], then query per-entity blocks. An owned
/// value: querying does not borrow the problem, so
/// `let cov = p.assemble_covariance()?;` followed by
/// `cov.marginal_cov(&p.landmarks[0])` borrow-checks.
///
/// Holds the sparse Cholesky factor of `H` and the CSC `H` itself (for
/// conditional queries). Covariance blocks are solved for on demand.
pub struct CovAssembly {
    n: usize,
    symbolic: fchol::SymbolicCholesky<usize>,
    l_vals: Vec<f64>,
    h: CscMatrix<f64>,
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
    /// the factorization for querying. The dense inverse is never formed; each
    /// query solves for only the columns it needs. `Err` if `H` is singular.
    fn assemble_covariance(&mut self) -> Result<CovAssembly, CovError> {
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
        let symbolic = fchol::factorize_symbolic_cholesky(
            sym_ref,
            faer::Side::Upper,
            fchol::SymmetricOrdering::Amd,
            Default::default(),
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

        Ok(CovAssembly { n, symbolic, l_vals, h })
    }
}

impl<T: Float, P: LmProblem<T> + RootProblem<T>> Covariance<T> for P {}
