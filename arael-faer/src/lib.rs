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
//! - **Conjugate gradients** ([`cg`]) -- solve a symmetric system by repeated
//!   multiplication instead of factorizing it, preconditioned by the Cholesky
//!   factor of each diagonal block. No fill and no factor to store, at the
//!   price of an inexact solution. The operator is a closure, so it can be a
//!   matrix that was never formed.
//! - **Supernodal block Cholesky** ([`supernodal`]) -- factorize a block-CSC
//!   matrix in block form under a fill-reducing ordering. The elimination tree
//!   is built over the blocks, columns with the same pattern are amalgamated
//!   into one dense panel, and each update between panels is one GEMM. Where
//!   the envelope route needs a narrow envelope and no ordering, this needs an
//!   ordering and handles any pattern.
//! - **Envelope Cholesky** ([`envelope`]) -- factorize a block-CSC matrix in
//!   natural order directly in block form, fill confined to each column's
//!   envelope. A trajectory's Hessian, and its reduced pose system, keep a
//!   narrow one, so this needs no fill-reducing ordering and no symbolic phase.
//! - **Nested dissection** ([`nd`]) -- a fill-reducing ordering for matrices
//!   with no band and no small degrees, where minimum degree has nothing to
//!   chew on. faer offers AMD, natural, or a custom permutation; this computes
//!   the custom one.
//! - **Worker threads** ([`pool`]) -- the threads the stages above that do
//!   not run on faer's kernels dispatch to, and that arael's sweeps share.
//!   Spawned on first use, parked between dispatches, one dispatch at a
//!   time. faer's own kernels run on rayon's pool.
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
//!   iterations; `set_threads` runs the reduction and the
//!   back-substitution on the pool, one range of S's block columns per
//!   thread, same sums at any count; `set_chunk_columns` sets how many
//!   columns a thread forms at a time; `enable_timing` breaks a
//!   reduction down by stage
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
//! # cg -- conjugate gradients
//!
//! Solve `A x = b` for symmetric positive-definite `A` by repeated
//! multiplication: no factorization, so no fill and no factor to store. The
//! step it returns is inexact by construction -- it stops when the residual
//! has fallen far enough -- which suits a damped solve, where the step is a
//! trial anyway.
//!
//! * [`cg::solve`] -- takes the operator as a closure, so `A` can
//!   be a matrix held in block CSC
//!   ([`mul_symmetric_upper`](bsc::SparseBlockColMat::mul_symmetric_upper)) or
//!   one that is never built at all
//! * [`BlockJacobi`](cg::BlockJacobi) -- the preconditioner: the Cholesky
//!   factor of each diagonal block, built from a matrix or handed the blocks
//!   directly
//! * [`CgOptions`](cg::CgOptions) / [`CgStats`](cg::CgStats) -- tolerance and
//!   iteration cap in, iterations and final residual out
//!
//! The reductions run in f64 whatever the storage type. Paired with
//! [`schur_apply`](schur::schur_apply) this solves the reduced camera system
//! without ever forming it.
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
//!
//! # supernodal -- supernodal block Cholesky
//!
//! Factor `L L^T = A` for a symmetric positive-definite block-CSC matrix, in
//! block form, under a block-level ordering. The elimination tree is built
//! over the blocks rather than the scalars; block columns sharing a pattern
//! are amalgamated into supernodes, each held as one dense column-major panel;
//! and every update between two supernodes is packed into a scratch buffer,
//! spent as one GEMM, and scattered back. The permutation is baked into the
//! scatter map, so the matrix is never permuted and no scalar copy of it is
//! ever built.
//!
//! * [`SupernodalSymbolic::new`](supernodal::SupernodalSymbolic::new) --
//!   analyse once: elimination tree, supernodes, panel patterns and the
//!   scatter map. The ordering is a block permutation, `None` keeping the
//!   natural one
//! * [`supernodal_factorize`](supernodal::supernodal_factorize) -- numeric:
//!   `L L^T = A` into a factor buffer, left-looking over the descendant
//!   graph. Its `Par` runs independent subtrees on separate threads and
//!   hands the pool to the dense kernels of the panels too big to chunk;
//!   the result is bit-identical at any thread count
//! * [`supernodal_solve`](supernodal::supernodal_solve) -- solve `A x = rhs`
//!   from the factor
//! * [`SupernodalParams`](supernodal::SupernodalParams) -- amalgamation table,
//!   update-batching ratio, postordering;
//!   [`memory_lean`](supernodal::SupernodalParams::memory_lean) trades a
//!   little speed for a smaller factor
//! * [`amd_block_order`](supernodal::amd_block_order) -- AMD over the block
//!   adjacency, blocks kept whole
//! * [`SupernodalError`](supernodal::SupernodalError) -- not positive
//!   definite, or the factor overflowed the index type
//!
//! Consecutive small updates into one target panel are batched: packed
//! zero-padded into a joint operand pair and spent as a single larger GEMM,
//! accepted while the padding stays under the params' ratio.
//! `arael::simple_lm::BlockSupernodalMode` selects this route in arael, where
//! it is the default on a sequential solve.

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

/// Sort `keys` by their bits `bits`, keeping the order of the keys that
/// are equal in them: counting sort, lowest digit first. The digit is as
/// wide as the key count asks for, 8 to 16 bits, so the histogram holds
/// about one counter per key; the passes stop at the highest bit any key
/// sets in the range, and a digit every key shares costs its count and
/// no placement.
pub fn sort_keys(keys: &mut Vec<u64>, bits: core::ops::Range<u32>) {
    assert!(bits.end <= 64 && keys.len() <= u32::MAX as usize);
    let n = keys.len() as u32;
    if n < 2 || bits.start >= bits.end {
        return;
    }
    let below = (1u64 << bits.start) - 1;
    let within = if bits.end == 64 { u64::MAX } else { (1u64 << bits.end) - 1 };
    let set = keys.iter().fold(0u64, |acc, &k| acc | k) & within & !below;
    if set == 0 {
        return;
    }
    let end = 64 - set.leading_zeros();
    let width_max = (u32::BITS - n.leading_zeros()).clamp(8, 16);
    let mut counts = vec![0u32; (1usize << width_max) + 1];
    let mut tmp: Vec<u64> = Vec::new();
    let mut next = bits.start;
    while next < end {
        let shift = next;
        let width = (end - shift).min(width_max);
        next += width;
        let mask = (1u64 << width) - 1;
        let digit = |k: u64| ((k >> shift) & mask) as usize;
        // Per digit, where its first key goes.
        let start = &mut counts[..(1usize << width) + 1];
        start.fill(0);
        for &k in keys.iter() {
            start[digit(k) + 1] += 1;
        }
        if start.iter().any(|&c| c == n) {
            continue;
        }
        for d in 0..start.len() - 1 {
            start[d + 1] += start[d];
        }
        tmp.resize(keys.len(), 0);
        for &k in keys.iter() {
            let d = digit(k);
            tmp[start[d] as usize] = k;
            start[d] += 1;
        }
        core::mem::swap(keys, &mut tmp);
    }
}

pub mod envelope;
pub mod bsc;
pub mod cg;
pub mod nd;
pub mod schur;
pub mod supernodal;
pub mod pool;

#[cfg(test)]
mod sort_keys_tests {
    use super::*;

    fn keys(n: u64, bits: u32) -> Vec<u64> {
        let mut x = 12345u64;
        (0..n).map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            if bits == 64 { x } else { (x >> 11) & ((1u64 << bits) - 1) }
        }).collect()
    }

    /// Over every bit it is the plain sort, whatever the keys' range,
    /// the digits all keys share included.
    #[test]
    fn over_every_bit_it_is_the_plain_sort() {
        for bits in [64, 48, 33, 20, 16, 3] {
            let mut radix = keys(5000, bits);
            let mut plain = radix.clone();
            sort_keys(&mut radix, 0..64);
            plain.sort_unstable();
            assert_eq!(radix, plain, "{} bits", bits);
        }
        let mut empty = Vec::new();
        sort_keys(&mut empty, 0..64);
        assert!(empty.is_empty());
        let mut same = vec![7u64; 100];
        sort_keys(&mut same, 0..64);
        assert_eq!(same, vec![7u64; 100]);
    }

    /// Over a part of the bits the keys equal in it keep their order.
    #[test]
    fn keys_equal_in_the_sorted_bits_keep_their_order() {
        // High word from a small range, low word the key's place.
        for n in [5000, 455, 9, 2] {
            let mut radix: Vec<u64> = keys(n, 6).iter().enumerate()
                .map(|(k, v)| (v << 32) | k as u64).collect();
            let mut plain = radix.clone();
            sort_keys(&mut radix, 32..64);
            plain.sort_unstable();
            assert_eq!(radix, plain, "{} keys", n);
        }
    }

    /// A few keys take narrow digits and few passes; the result is the
    /// plain sort all the same, keys above the sorted range included.
    #[test]
    fn few_keys_sort_the_same() {
        for n in [2, 3, 7, 100, 455, 4000] {
            for bits in [64, 40, 33, 16, 9, 3] {
                let mut radix = keys(n, bits);
                let mut plain = radix.clone();
                sort_keys(&mut radix, 0..64);
                plain.sort_unstable();
                assert_eq!(radix, plain, "{} keys, {} bits", n, bits);
            }
        }
        // Bits above the range are not sorted on, however high they are.
        let mut radix: Vec<u64> = keys(300, 20).iter().enumerate()
            .map(|(k, v)| (((k % 3) as u64) << 60) | (v << 16) | k as u64).collect();
        let mut plain = radix.clone();
        sort_keys(&mut radix, 16..36);
        plain.sort_by_key(|k| (k >> 16) & 0xf_ffff);
        assert_eq!(radix, plain);
    }
}

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
