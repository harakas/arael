// faer's sparse Cholesky is generic over its index type. The reduced Schur
// system is handed to it as a scalar CSC whose row indices are half the
// solver's peak on the factorizing route, so whether `u32` works there is a
// memory question, not a taste one.
//
// Same matrix, same ordering, factorized once with each index type: the
// factors must agree to the bit, and the solve must return the same x.

use arael_faer::faer::sparse::linalg::cholesky::*;
use arael_faer::faer::sparse::{SparseColMatRef, SymbolicSparseColMatRef};
use arael_faer::faer::{Index, Par, Side};
use arael_faer::faer::dyn_stack::MemStack;

/// An SPD pentadiagonal system in upper-triangle CSC, over any index type.
fn build<I: Index>(n: usize) -> (Vec<I>, Vec<I>, Vec<f64>) {
    let mut col_ptr = Vec::with_capacity(n + 1);
    let mut row_idx = Vec::new();
    let mut vals = Vec::new();
    col_ptr.push(I::truncate(0));
    for j in 0..n {
        for i in j.saturating_sub(2)..=j {
            row_idx.push(I::truncate(i));
            vals.push(if i == j { 8.0 + j as f64 } else { -1.0 });
        }
        col_ptr.push(I::truncate(row_idx.len()));
    }
    (col_ptr, row_idx, vals)
}

/// Factorize and solve, returning the factor values and the solution.
fn solve<I: Index>(n: usize, rhs: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let (col_ptr, row_idx, vals) = build::<I>(n);
    let sym = SymbolicSparseColMatRef::new_checked(n, n, &col_ptr, None, &row_idx);
    let symbolic = factorize_symbolic_cholesky(
        sym,
        Side::Upper,
        SymmetricOrdering::Amd,
        Default::default(),
    )
    .expect("symbolic");

    let a = SparseColMatRef::new(sym, &vals);
    let mut l = vec![0.0f64; symbolic.len_val()];

    let need = symbolic.factorize_numeric_llt_scratch::<f64>(Par::Seq, arael_faer::faer::Spec::default());
    let mut factor_mem = vec![
        core::mem::MaybeUninit::<u8>::uninit();
        need.unaligned_bytes_required()
    ];
    let llt = symbolic
        .factorize_numeric_llt::<f64>(
            &mut l,
            a,
            Side::Upper,
            arael_faer::faer::linalg::cholesky::llt::factor::LltRegularization::default(),
            Par::Seq,
            MemStack::new(&mut factor_mem),
            arael_faer::faer::Spec::default(),
        )
        .expect("numeric");

    let mut x = rhs.to_vec();
    let need = symbolic.solve_in_place_scratch::<f64>(1, Par::Seq);
    let mut solve_mem = vec![
        core::mem::MaybeUninit::<u8>::uninit();
        need.unaligned_bytes_required()
    ];
    let rhs_col = arael_faer::faer::col::ColMut::from_slice_mut(&mut x);
    llt.solve_in_place_with_conj(
        arael_faer::faer::Conj::No,
        rhs_col.as_mat_mut(),
        Par::Seq,
        MemStack::new(&mut solve_mem),
    );
    (l, x)
}

/// A `u32`-indexed factorization is the same computation as a `usize` one --
/// same factor, same solution -- so the index width is free to choose.
#[test]
fn u32_indices_factorize_and_solve_identically() {
    let n = 400;
    let rhs: Vec<f64> = (0..n).map(|i| 1.0 + (i % 7) as f64).collect();

    let (l32, x32) = solve::<u32>(n, &rhs);
    let (l64, x64) = solve::<usize>(n, &rhs);

    assert_eq!(l32.len(), l64.len(), "factor sizes differ");
    assert_eq!(l32, l64, "factor values differ between index widths");
    assert_eq!(x32, x64, "solutions differ between index widths");

    // and the solution really solves the system
    let (col_ptr, row_idx, vals) = build::<usize>(n);
    let mut ax = vec![0.0f64; n];
    for j in 0..n {
        for k in col_ptr[j]..col_ptr[j + 1] {
            let i = row_idx[k];
            ax[i] += vals[k] * x64[j];
            if i != j {
                ax[j] += vals[k] * x64[i];
            }
        }
    }
    for (got, want) in std::iter::zip(&ax, &rhs) {
        assert!((got - want).abs() < 1e-9, "A x != b: {} vs {}", got, want);
    }
}
