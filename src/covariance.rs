//! Parameter covariance recovery.
//!
//! At the solution the parameter covariance is `Sigma = (J^T J)^-1`. arael
//! assembles `H = 2 J^T J` (the cost is `sum r^2`, so `add_residual` carries a
//! factor of 2), hence `Sigma = 2 H^-1`. Because the rotation parameters are
//! already minimal 3-DOF retractions, `Sigma` is in local tangent coordinates
//! -- no manifold projection is needed.
//!
//! Query per-entity blocks through the entity itself: any `Model` reports its
//! parameter span via `collect_param_blocks`, so the covariance is extracted
//! generically.
//!
//! ```ignore
//! use arael::covariance::Covariance;
//! path.solve_sparse(&cfg);              // solution written back into the model
//! let cov = path.assemble_covariance()?;
//! let lm0  = cov.marginal_cov(&path.landmarks[0]);   // one landmark
//! let last = cov.marginal_cov(&path.poses[path.poses.len() - 1]);
//! ```
//!
//! This is the dense path (re-assemble `H` at the solution, factor once). It is
//! exact and correct for any problem that fits in memory; scalable per-marginal
//! recovery for large sparse problems is future work (see COV.md).

use crate::model::Model;
use crate::simple_lm::{LmProblem, RootProblem};
use crate::utils::Float;
use nalgebra::DMatrix;

/// Why a covariance could not be assembled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CovError {
    /// `H` is not positive definite. The usual cause is an unfixed gauge: with
    /// no anchor the problem has unobservable directions and `H` is singular
    /// (e.g. free-gauge SLAM). Fix a pose / add a prior, or -- for a small
    /// problem -- a pseudo-inverse path (not yet implemented).
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
/// [`Covariance::assemble_covariance`], then query per-entity blocks.
///
/// Holds the full `Sigma = 2 H^-1` in `f64` (covariance is computed in `f64`
/// regardless of the model's precision). It is an owned value: querying it does
/// not borrow the problem, so `let cov = p.assemble_covariance()?;` followed by
/// `cov.marginal_cov(&p.landmarks[0])` borrow-checks.
pub struct CovAssembly {
    n: usize,
    sigma: DMatrix<f64>,
    hessian: DMatrix<f64>,
}

impl CovAssembly {
    /// Number of optimized scalar parameters.
    pub fn dim(&self) -> usize {
        self.n
    }

    /// The full covariance matrix (`n x n`), in serialize order.
    pub fn full(&self) -> &DMatrix<f64> {
        &self.sigma
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

    fn extract(&self, rows: &[usize], cols: &[usize]) -> DMatrix<f64> {
        let mut m = DMatrix::zeros(rows.len(), cols.len());
        for (r, &i) in rows.iter().enumerate() {
            for (c, &j) in cols.iter().enumerate() {
                m[(r, c)] = self.sigma[(i, j)];
            }
        }
        m
    }

    /// Marginal covariance of a model: one entity for its own covariance, or a
    /// whole collection for the joint over all its entities. `K x K` where `K`
    /// is the entity's DOF, in tangent coordinates.
    pub fn marginal_cov<M: Model + ?Sized>(&self, m: &M) -> DMatrix<f64> {
        let idx = Self::indices(m);
        self.extract(&idx, &idx)
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
                hb[(r, c)] = self.hessian[(i, j)];
            }
        }
        hb.try_inverse()
            .map(|inv| inv * 2.0)
            .unwrap_or_else(|| DMatrix::from_element(k, k, f64::INFINITY))
    }

    /// Standard deviations: the square root of the marginal covariance
    /// diagonal, one per scalar parameter of the model.
    pub fn std_dev<M: Model + ?Sized>(&self, m: &M) -> Vec<f64> {
        Self::indices(m).iter().map(|&i| self.sigma[(i, i)].sqrt()).collect()
    }

    /// Cross-covariance block between two models (the off-diagonal `A x B`
    /// block of the joint covariance).
    pub fn cross_cov<A: Model + ?Sized, B: Model + ?Sized>(&self, a: &A, b: &B) -> DMatrix<f64> {
        let ia = Self::indices(a);
        let ib = Self::indices(b);
        self.extract(&ia, &ib)
    }
}

/// Post-solve covariance. Implemented for every `#[arael(root)]` model. Call
/// with the trait in scope after the solution has been written into the model
/// (i.e. after a `solve_*` / `deserialize`), since it reads the model's current
/// parameters as the linearization point.
pub trait Covariance<T: Float>: LmProblem<T> + RootProblem<T> {
    /// Re-assemble `H` at the current parameters, factor it, and return the
    /// covariance `2 H^-1` for querying. `Err` if `H` is singular (see
    /// [`CovError`]).
    fn assemble_covariance(&mut self) -> Result<CovAssembly, CovError> {
        let mut params: Vec<T> = Vec::new();
        self.serialize(&mut params);
        let n = params.len();
        if n == 0 {
            return Err(CovError::Empty);
        }
        let mut grad = vec![T::zero(); n];
        let mut hess = vec![T::zero(); n * n];
        self.calc_grad_hessian_dense(&params, &mut grad, &mut hess);

        // Covariance is computed in f64 regardless of the model's precision.
        let h: Vec<f64> = hess.iter().map(|&x| x.to_f64().unwrap_or(f64::NAN)).collect();
        let hmat = DMatrix::from_row_slice(n, n, &h);
        let chol = nalgebra::linalg::Cholesky::new(hmat.clone()).ok_or(CovError::NotPositiveDefinite)?;
        // Sigma = 2 H^-1: the factor of 2 undoes the one add_residual applies.
        let sigma = chol.inverse() * 2.0;
        Ok(CovAssembly { n, sigma, hessian: hmat })
    }
}

impl<T: Float, P: LmProblem<T> + RootProblem<T>> Covariance<T> for P {}
