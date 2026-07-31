//! faer extensions. Everything here is built on faer's public API and laid out
//! the way it would sit in faer itself.
//!
//! The block-structured pieces a large sparse solve needs:
//!
//! - **Block CSC** ([`bsc`]) -- sparse matrix storage over a *variable* block
//!   partition. A Hessian assembled from entities (6-wide poses, 3-wide
//!   points) is a matrix of small dense tiles, and storing it that way means
//!   one index lookup per tile instead of per scalar, and dense kernels
//!   inside the tile.
//! - **Schur complement** ([`schur`]) -- eliminate a set of mutually
//!   uncoupled blocks from a block-CSC matrix and factorize only what is
//!   left. This is the landmark/point marginalization that makes bundle
//!   adjustment and SLAM tractable, and it needs the block structure to be
//!   cheap.
//! - **Band Cholesky** ([`band`]) -- factorize a block-CSC matrix that is
//!   banded in natural order directly in block form, fill confined to each
//!   column's envelope. A trajectory's Hessian, and its reduced pose system,
//!   are banded, so this needs no fill-reducing ordering and no symbolic phase.
//! - **Nested dissection** ([`nd`]) -- a fill-reducing ordering for matrices
//!   with no band and no small degrees, where minimum degree has nothing to
//!   chew on. faer offers AMD, natural, or a custom permutation; this computes
//!   the custom one.
//!
//! # bsc -- block CSC
//!
//! [`SymbolicSparseBlockColMat`](bsc::SymbolicSparseBlockColMat) is the
//! structure (row/column partitions, which tiles exist),
//! [`SparseBlockColMat`](bsc::SparseBlockColMat) adds the values.
//! Upper-triangle storage is the convention for symmetric matrices; diagonal
//! tiles carry only their own upper triangle.
//!
//! * `SymbolicSparseBlockColMat::from_scalar_coords` -- build the structure
//!   from scalar (row, col) coordinates and a partition
//! * `SparseBlockColMat::zeroed` / `new` -- allocate values for a structure
//! * `block(b)` / `block_mut(b)` -- a tile as a faer `MatRef` / `MatMut` --
//!   dense, so faer's kernels apply
//! * [`PositionResolver`](bsc::PositionResolver) -- scalar (i, j) -> offset
//!   in the value array; build a scatter map once, assemble by index forever
//!   after
//! * `csc_pattern` / `csc_vals_into` / `to_csc` -- hand the matrix to a
//!   scalar sparse factorization
//! * `to_dense` -- expand, for tests and debugging
//!
//! # nd -- nested dissection
//!
//! A fill-reducing ordering for matrices with no band and no small degrees --
//! bundle adjustment, where every 3D point makes a clique of the cameras that
//! see it and minimum degree drowns in them. It dissects the BLOCK graph, so a
//! block's parameters stay contiguous and the factor keeps its supernodes.
//! [`NestedDissection::of_blocks`](nd::NestedDissection::of_blocks) gives a
//! permutation faer takes as-is (`SymmetricOrdering::Custom`).
//!
//! Not a general win: a banded system (a SLAM trajectory) is 3.4x SLOWER
//! dissected, and a pose graph prefers AMD. The caller must know its matrix.
//!
//! Nested dissection normally comes from METIS, a C library -- it is what
//! CHOLMOD uses, and faer's `SymmetricOrdering` offers `Amd`, `Identity` and
//! `Custom` and nothing else. So a pure Rust implementation was written here.
//!
//! # schur -- Schur complement
//!
//! Given `H` in block CSC and a set of blocks to eliminate,
//! `S = Hkk - Hke Hee^-1 Hek`. The eliminated blocks must be mutually
//! uncoupled (no tile joins two of them), which makes `Hee` block-diagonal
//! and lets the reduction decompose into one independent contribution per
//! eliminated block.
//!
//! * [`schur_symbolic`](schur::schur_symbolic) -- analyse once: what S looks
//!   like, where every contribution lands. Returns
//!   [`SchurSymbolic`](schur::SchurSymbolic), or
//!   [`SchurError`](schur::SchurError) if the set is not eliminable
//! * [`schur_reduce`](schur::schur_reduce) -- numeric: fill S and the reduced
//!   right-hand side from a (damped) H
//! * [`schur_backsub`](schur::schur_backsub) -- recover the eliminated blocks
//!   once the reduced system is solved
//! * `SchurSymbolic::{kept_size, kept_bandwidth, reduce_flops, pair_count}`
//!   -- what the reduction will cost and how big S is -- free, from the
//!   symbolic pass, for deciding whether to reduce at all
//! * [`SchurContext`](schur::SchurContext) -- reusable workspace across
//!   iterations; `enable_timing` breaks a reduction down by stage
//! * [`FIXED_SHAPES`](schur::FIXED_SHAPES) /
//!   [`has_fixed_kernel`](schur::has_fixed_kernel) -- the tile shapes with a
//!   fully unrolled GEMM kernel (the ones SLAM systems use: 3/6/7/9-wide
//!   observers through 1/2/3/4-wide marginalized blocks, and those same widths
//!   one column wide, for the right-hand side and back-substitution). Anything
//!   else works, through the nano-gemm fallback, at about 1.2-1.4x
//! * [`gemm_shapes`](schur::SchurSymbolic::gemm_shapes) -- which shapes a given
//!   problem needs, so a caller can see whether it is on the slow path
//!
//! The caller factorizes S itself (`csc_pattern` + faer's sparse Cholesky, or
//! any other solver), then calls `schur_backsub`. See
//! `arael::simple_lm::SparseFaer` for the whole loop.
//!
//! # envelope -- envelope (profile, skyline) Cholesky
//!
//! Given a symmetric positive-definite block-CSC matrix in natural order,
//! factor `R^T R = S` in block form. Cholesky preserves each column's
//! envelope, so the factor's pattern follows from the matrix: no fill-reducing
//! ordering, no symbolic analysis, no scalar-CSC round trip. Works on the
//! whole Hessian (a pose graph or localization system) or on a reduced Schur
//! system -- anything left in its natural order.
//!
//! * [`EnvelopeSymbolic::new`](envelope::EnvelopeSymbolic::new) -- analyse once: the
//!   factor's envelope pattern and the map back to the matrix's values. The
//!   envelope falls out of the block structure; the caller supplies nothing
//! * [`envelope_factorize`](envelope::envelope_factorize) -- numeric: `R^T R = S` into a
//!   factor buffer. Left-looking by block column, reusing schur's unrolled tile
//!   kernels
//! * [`envelope_solve`](envelope::envelope_solve) -- solve `S x = rhs` from the factor
//! * [`EnvelopeError`](envelope::EnvelopeError) -- the matrix was not positive definite
//!
//! It holds less than an ordered sparse factor while the envelope stays
//! narrow, and more once it widens, so the choice is worth pricing:
//! [`envelope_flops`](envelope::envelope_flops) costs one pass over the
//! pattern. `arael::simple_lm::EnvelopeMode` does that for the reduced Schur
//! system; `SparseFaer::with_narrow_band` takes the whole Hessian and warns
//! when its band is too wide to pay.

pub use faer;

/// The index type the sparse structures are instantiated with: block rows,
/// column pointers, permutations, and the offsets the Schur analysis keeps.
///
/// These are the largest structures a solve holds after the values
/// themselves -- the Schur symbolic alone runs to tens of megabytes on a
/// bundle problem -- and every one of them is generic, so the width is a
/// single choice. 32 bits addresses 4e9 of anything, which no problem that
/// fits in memory reaches; widening the library is this alias and a rebuild.
///
/// faer's `Index` is implemented for `u32`, `u64` and `usize`, so its sparse
/// Cholesky takes whichever this names (verified identical factor and
/// solution in `tests/u32_cholesky.rs`).
pub type SparseIndex = u32;

/// An offset into a matrix's value buffer, as the position maps and tile
/// origins store it.
///
/// One entry per stored scalar is the largest index structure a solve carries,
/// so the width is worth spending a type on: 32 bits addresses 4e9 values --
/// 34 GB of `f64` -- which is past what any problem that fits in memory
/// reaches. Widening the library is this alias and a rebuild.
pub type ValueIndex = u32;

/// Convert a `usize` offset into a [`ValueIndex`].
///
/// Checked rather than assumed: overflowing would put a value in the wrong
/// slot instead of failing, which is a silently wrong matrix. Goes through
/// `TryFrom` so widening the alias needs no edit here -- and costs nothing
/// once the range is statically known.
#[inline]
pub fn value_index(p: usize) -> ValueIndex {
    ValueIndex::try_from(p).unwrap_or_else(|_| {
        panic!(
            "value buffer holds {} entries; a ValueIndex addresses at most {}",
            p,
            ValueIndex::MAX,
        )
    })
}

pub mod envelope;
pub mod bsc;
pub mod cg;
pub mod nd;
pub mod schur;

#[cfg(test)]
mod value_index_tests {
    use super::*;

    /// The conversion is checked, so a buffer too large for the alias fails
    /// loudly instead of scattering into a wrapped-around slot.
    #[test]
    fn value_index_rejects_an_unaddressable_offset() {
        let max = ValueIndex::MAX as usize;
        assert_eq!(value_index(max), ValueIndex::MAX);
        assert_eq!(value_index(0), 0);
        if max < usize::MAX {
            assert!(std::panic::catch_unwind(|| value_index(max + 1)).is_err());
        }
    }
}
