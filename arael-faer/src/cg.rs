//! Preconditioned conjugate gradients on a symmetric block matrix.
//!
//! Solves `A x = b` for symmetric positive-definite `A` held as the upper
//! block triangle (see [`crate::bsc`]), by repeated multiplication -- no
//! factorization, so no fill and no factor to store. The step it returns is
//! inexact by construction: it stops when the residual has fallen far enough,
//! and how far is the caller's [`CgOptions::tol`].
//!
//! The preconditioner is block Jacobi: the Cholesky factor of each diagonal
//! block, applied per block. It costs one small dense factorization per block
//! per call and is what makes the iteration count tolerable.
//!
//! Reductions run in f64 whatever `T` is. The dot products are where a single-
//! precision solve loses its digits first, and they are O(n) against an
//! O(nnz) matrix-vector product, so widening them is close to free.

use crate::bsc::SparseBlockColMat;
use crate::{value_index, ValueIndex};
use crate::schur::{llt_in_place, llt_solve_panel, SchurReal};
use faer::Index;

/// Settings for one CG solve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CgOptions {
    /// Stop once the preconditioned residual `r^T M^-1 r` falls to this
    /// fraction of its value at entry. That quantity is squared, so 1e-6
    /// here is a factor of 1e-3 on the residual norm.
    pub tol: f64,
    /// Iteration cap. 0 means the system dimension, at which CG would be
    /// exact in exact arithmetic.
    pub max_iters: usize,
    /// Recompute `r = b - A x` from scratch every this many iterations
    /// instead of trusting the update recurrence; 0 never does.
    ///
    /// The recurrence accumulates rounding, so the residual CG tracks drifts
    /// away from the true one -- far enough in single precision that CG can
    /// report a convergence it has not reached. Recomputing costs one extra
    /// matrix-vector product each time.
    pub restart_every: usize,
}

impl Default for CgOptions {
    fn default() -> Self {
        CgOptions { tol: 1e-6, max_iters: 0, restart_every: 0 }
    }
}

/// What one CG solve did.
#[derive(Clone, Copy, Debug, Default)]
pub struct CgStats {
    /// Iterations run, each one matrix-vector product.
    pub iters: usize,
    /// Whether the tolerance was met. `false` means the cap was reached or
    /// the iteration broke down, and `x` holds the best iterate so far.
    pub converged: bool,
    /// `r^T M^-1 r` at exit, relative to its value at entry.
    pub relative_residual: f64,
}

/// Why a CG solve could not run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CgError {
    /// A diagonal block of `A` is not positive definite, so the block-Jacobi
    /// preconditioner does not exist. The system is not usable here.
    IndefiniteBlock(usize),
}

/// Block-Jacobi preconditioner: the Cholesky factor of every diagonal block.
#[derive(Debug)]
pub struct BlockJacobi<T> {
    /// Lower Cholesky factors, column-major, concatenated block by block.
    factors: Vec<T>,
    /// Where each block's factor starts in `factors`.
    at: Vec<ValueIndex>,
    /// Scalar offset and width of each block.
    span: Vec<(usize, usize)>,
}

impl<T: SchurReal> BlockJacobi<T> {
    /// Factor every diagonal block of `a`.
    ///
    /// The diagonal tiles hold their upper triangle (the storage convention),
    /// while the Cholesky reads the lower, so each is mirrored into scratch
    /// on the way in.
    pub fn build<I: Index>(a: &SparseBlockColMat<I, T>) -> Result<Self, CgError> {
        let sym = a.symbolic();
        let nblk = sym.nblk_cols();
        let mut factors = Vec::new();
        let mut at = Vec::with_capacity(nblk);
        let mut span = Vec::with_capacity(nblk);
        let mut scratch: Vec<T> = Vec::new();

        for j in 0..nblk {
            let cols = sym.col_span(j);
            let w = cols.len();
            at.push(value_index(factors.len()));
            span.push((cols.start, w));

            let tile = a
                .get_block(j, j)
                .ok_or(CgError::IndefiniteBlock(j))?;
            // Column-major (row, col) of the stored tile lives at
            // [col * w + row], and only row <= col was written, so the
            // symmetric counterpart of any (i, k) is at [max * w + min].
            scratch.clear();
            scratch.resize(w * w, T::ZERO);
            for k in 0..w {
                for i in 0..w {
                    let (hi, lo) = if i >= k { (i, k) } else { (k, i) };
                    scratch[i + k * w] = tile[(lo, hi)];
                }
            }
            if !llt_in_place(&mut scratch, w) {
                return Err(CgError::IndefiniteBlock(j));
            }
            factors.extend_from_slice(&scratch);
        }
        Ok(BlockJacobi { factors, at, span })
    }

    /// Factor diagonal blocks handed over directly, for an operator with no
    /// matrix to read them off. `spans` is `(scalar offset, width)` per block
    /// in ascending order; `blocks` holds each one FULL and symmetric,
    /// column-major, concatenated in the same order.
    pub fn from_diagonal_blocks(
        spans: &[(usize, usize)],
        blocks: &[T],
    ) -> Result<Self, CgError> {
        let mut factors = Vec::with_capacity(blocks.len());
        let mut at = Vec::with_capacity(spans.len());
        let mut scratch: Vec<T> = Vec::new();
        let mut read = 0usize;
        for (j, &(_, w)) in spans.iter().enumerate() {
            at.push(value_index(factors.len()));
            scratch.clear();
            scratch.extend_from_slice(&blocks[read..read + w * w]);
            read += w * w;
            if !llt_in_place(&mut scratch, w) {
                return Err(CgError::IndefiniteBlock(j));
            }
            factors.extend_from_slice(&scratch);
        }
        Ok(BlockJacobi { factors, at, span: spans.to_vec() })
    }

    /// `z = M^-1 r`, block by block.
    pub fn apply(&self, r: &[T], z: &mut [T]) {
        for (b, &(start, w)) in self.span.iter().enumerate() {
            let seg = &mut z[start..start + w];
            seg.copy_from_slice(&r[start..start + w]);
            llt_solve_panel(&self.factors[self.at[b] as usize..], seg, w, 1);
        }
    }
}

/// Scratch vectors, reused across solves so an LM iteration allocates none.
pub struct CgWorkspace<T> {
    r: Vec<T>,
    z: Vec<T>,
    p: Vec<T>,
    q: Vec<T>,
}

// Hand-written: deriving Default would demand `T: Default`, which empty Vecs
// do not need and the scalar types would then have to carry.
impl<T> Default for CgWorkspace<T> {
    fn default() -> Self {
        CgWorkspace { r: Vec::new(), z: Vec::new(), p: Vec::new(), q: Vec::new() }
    }
}

impl<T: SchurReal> CgWorkspace<T> {
    /// Size the scratch for an `n`-dimensional system.
    pub fn resize(&mut self, n: usize) {
        for v in [&mut self.r, &mut self.z, &mut self.p, &mut self.q] {
            v.clear();
            v.resize(n, T::ZERO);
        }
    }
}

/// f64 dot product over `T` storage.
fn dot<T: SchurReal>(a: &[T], b: &[T]) -> f64 {
    let mut acc = 0.0f64;
    for i in 0..a.len() {
        acc += a[i].to_f64() * b[i].to_f64();
    }
    acc
}

/// Solve `A x = b` by preconditioned conjugate gradients, starting from
/// `x = 0`. `x` is overwritten.
///
/// `apply` is the operator: `apply(x, y)` must leave `y = A x`. Taking it as a
/// closure rather than a matrix lets the caller supply either a system it has
/// built ([`SparseBlockColMat::mul_symmetric_upper`]) or one it applies
/// without building, which is the whole point of an iterative solve.
pub fn solve<T: SchurReal>(
    mut apply: impl FnMut(&[T], &mut [T]),
    m: &BlockJacobi<T>,
    b: &[T],
    x: &mut [T],
    opts: &CgOptions,
    w: &mut CgWorkspace<T>,
) -> CgStats {
    let n = b.len();
    w.resize(n);
    x.iter_mut().for_each(|v| *v = T::ZERO);
    // Destructured so `apply` can borrow two of them at once.
    let CgWorkspace { r, z, p, q } = w;

    r.copy_from_slice(b);
    m.apply(r, z);
    p.copy_from_slice(z);
    let mut rho = dot(r, z);
    let rho0 = rho;

    // A zero right-hand side is already solved; so is a starting residual the
    // preconditioner reports as non-positive, which cannot be descended.
    if !(rho0 > 0.0) {
        return CgStats { iters: 0, converged: true, relative_residual: 0.0 };
    }
    let target = opts.tol * rho0;
    let cap = if opts.max_iters == 0 { n } else { opts.max_iters };

    let mut iters = 0;
    let mut converged = rho <= target;
    while !converged && iters < cap {
        apply(p, q);
        let pq = dot(p, q);
        // Non-positive curvature means A is not positive definite along p;
        // continuing would step the wrong way. Stop with what we have.
        if !(pq > 0.0) {
            break;
        }
        let alpha = rho / pq;
        let alpha_t = T::from_f64(alpha);
        for i in 0..n {
            x[i] = x[i] + alpha_t * p[i];
        }
        iters += 1;

        if opts.restart_every != 0 && iters % opts.restart_every == 0 {
            apply(x, q);
            for i in 0..n {
                r[i] = b[i] - q[i];
            }
        } else {
            for i in 0..n {
                r[i] = r[i] - alpha_t * q[i];
            }
        }

        m.apply(r, z);
        let rho_new = dot(r, z);
        converged = rho_new <= target;
        if converged {
            rho = rho_new;
            break;
        }
        let beta = T::from_f64(rho_new / rho);
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
        rho = rho_new;
    }

    CgStats { iters, converged, relative_residual: rho / rho0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bsc::SymbolicSparseBlockColMat;

    /// Two 2x2 diagonal blocks with one off-diagonal tile, upper triangle
    /// only, diagonally dominant so it is positive definite:
    ///     [ 4 1 | 1 0 ]
    ///     [ 1 5 | 0 1 ]
    ///     [ ----+---- ]
    ///     [ 1 0 | 6 2 ]
    ///     [ 0 1 | 2 7 ]
    fn spd() -> SparseBlockColMat<usize, f64> {
        let part = vec![0usize, 2, 4];
        let symbolic = SymbolicSparseBlockColMat::new_checked(
            part.clone(),
            part,
            vec![0usize, 1, 3],   // col0: {0}; col1: {0, 1}
            vec![0usize, 0, 1],
            vec![0usize, 4, 8, 12],
        );
        let vals = vec![
            // (0,0): upper triangle written, lower left as zero
            4., 0., 1., 5.,
            // (0,1): full off-diagonal tile
            1., 0., 0., 1.,
            // (1,1)
            6., 0., 2., 7.,
        ];
        SparseBlockColMat::new(symbolic, vals)
    }

    fn dense_of(m: &SparseBlockColMat<usize, f64>) -> [[f64; 4]; 4] {
        let mut out = [[0.0; 4]; 4];
        for k in 0..4 {
            let mut e = vec![0.0; 4];
            e[k] = 1.0;
            let mut col = vec![0.0; 4];
            m.mul_symmetric_upper(&e, &mut col);
            for i in 0..4 {
                out[i][k] = col[i];
            }
        }
        out
    }

    #[test]
    fn cg_reaches_the_exact_solution() {
        let a = spd();
        // Expected system, mirrored by hand -- pins mul_symmetric_upper too.
        assert_eq!(dense_of(&a), [
            [4., 1., 1., 0.],
            [1., 5., 0., 1.],
            [1., 0., 6., 2.],
            [0., 1., 2., 7.],
        ]);

        let m = BlockJacobi::build(&a).unwrap();
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let mut x = vec![0.0; 4];
        let mut w = CgWorkspace::default();
        let opts = CgOptions { tol: 1e-14, ..Default::default() };
        let stats = solve(|u, v| a.mul_symmetric_upper(u, v), &m, &b, &mut x, &opts, &mut w);
        assert!(stats.converged, "did not converge: {:?}", stats);
        assert!(stats.iters <= 4, "took {} iterations on a 4x4", stats.iters);

        // A x must reproduce b
        let mut ax = vec![0.0; 4];
        a.mul_symmetric_upper(&x, &mut ax);
        for i in 0..4 {
            assert!((ax[i] - b[i]).abs() < 1e-10, "row {}: {} vs {}", i, ax[i], b[i]);
        }
    }

    #[test]
    fn cg_stops_at_the_iteration_cap() {
        let a = spd();
        let m = BlockJacobi::build(&a).unwrap();
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let mut x = vec![0.0; 4];
        let mut w = CgWorkspace::default();
        let opts = CgOptions { tol: 1e-14, max_iters: 1, ..Default::default() };
        let stats = solve(|u, v| a.mul_symmetric_upper(u, v), &m, &b, &mut x, &opts, &mut w);
        assert_eq!(stats.iters, 1);
        assert!(!stats.converged);
    }

    #[test]
    fn restart_reaches_the_same_solution() {
        let a = spd();
        let m = BlockJacobi::build(&a).unwrap();
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let mut w = CgWorkspace::default();
        let opts = CgOptions { tol: 1e-14, restart_every: 1, ..Default::default() };
        let mut x = vec![0.0; 4];
        let stats = solve(|u, v| a.mul_symmetric_upper(u, v), &m, &b, &mut x, &opts, &mut w);
        assert!(stats.converged);
        let mut ax = vec![0.0; 4];
        a.mul_symmetric_upper(&x, &mut ax);
        for i in 0..4 {
            assert!((ax[i] - b[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn indefinite_block_is_reported() {
        let mut a = spd();
        a.vals_mut()[0] = -4.0; // break the first diagonal block
        assert_eq!(BlockJacobi::build(&a).unwrap_err(), CgError::IndefiniteBlock(0));
    }

    #[test]
    fn f32_solves_the_same_system() {
        let a64 = spd();
        let vals32: Vec<f32> = a64.vals().iter().map(|&v| v as f32).collect();
        let a = SparseBlockColMat::new(a64.symbolic().clone(), vals32);
        let m = BlockJacobi::build(&a).unwrap();
        let b = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut x = vec![0.0f32; 4];
        let mut w = CgWorkspace::default();
        let stats = solve(|u, v| a.mul_symmetric_upper(u, v), &m, &b, &mut x, &CgOptions::default(), &mut w);
        assert!(stats.converged, "{:?}", stats);
        let mut ax = vec![0.0f32; 4];
        a.mul_symmetric_upper(&x, &mut ax);
        for i in 0..4 {
            assert!((ax[i] - b[i]).abs() < 1e-4, "row {}: {} vs {}", i, ax[i], b[i]);
        }
    }
}
