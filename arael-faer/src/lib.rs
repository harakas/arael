//! faer extensions. Everything here is built on faer's public API and laid out
//! the way it would sit in faer itself.
//!
//! Three things faer does not ship:
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
//!   observers through 1/2/3/4-wide marginalized blocks). Anything else works,
//!   through a generic loop, at roughly half the speed
//! * [`gemm_shapes`](schur::SchurSymbolic::gemm_shapes) -- which shapes a given
//!   problem needs, so a caller can see whether it is on the slow path
//!
//! The caller factorizes S itself (`csc_pattern` + faer's sparse Cholesky, or
//! any other solver), then calls `schur_backsub`. See
//! `arael::simple_lm::SparseFaer` for the whole loop.

pub use faer;

pub mod bsc;
pub mod nd;
pub mod schur;
