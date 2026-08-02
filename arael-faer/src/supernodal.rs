//! Supernodal block Cholesky, symbolic analysis (docs/dev/BLOCK.md).
//!
//! Given a symmetric block-CSC matrix (upper-tile storage) and an
//! optional block elimination order, compute everything the numeric
//! factorization needs, on the block graph -- no scalar pattern is ever
//! formed:
//!
//! - the block elimination tree and per-column counts (block and scalar
//!   units, one pass);
//! - fundamental block supernodes, then relaxed amalgamation under a
//!   zero-fill budget counted in scalar units;
//! - each supernode's sorted block pattern, via an up-looking reach
//!   over the supernodal etree;
//! - the descendant graph (who updates whom), transposed and
//!   deduplicated;
//! - the dense panel layout (one column-major panel per supernode, the
//!   factor stored as lower L), and the tile source map: for every
//!   stored tile of the matrix, its target offset, stride and
//!   orientation inside the panels. The permutation lives in this map;
//!   the matrix is never permuted or copied.

use crate::bsc::{SparseBlockColMat, SymbolicSparseBlockColMat};
use crate::nd::Graph;
use crate::schur::SchurReal;
use crate::{SparseIndex, ValueIndex};

const NONE: u32 = u32::MAX;

/// Amalgamation and batching control.
///
/// `relax`: merge a supernode into its parent while the merged panel
/// stays under a zero-fill budget. Each `(max_cols, max_zero_fraction)`
/// row permits a merge whose combined scalar column count is at most
/// `max_cols` and whose accumulated explicit zeros stay under
/// `max_zero_fraction` of the merged panel. `None` disables
/// amalgamation (fundamental supernodes only). The default table is
/// the one faer tunes for scalar supernodes; measured as the best of
/// six candidates on the pose-graph and bundle benchmarks alike.
///
/// `batch_ratio`: batch consecutive small-depth descendant updates of
/// one target into a single zero-padded GEMM while the padded flops
/// stay within this factor of the individual updates' flops. Trades
/// arithmetic for one pass over the shared target region -- the win
/// where rank-3 landmark updates are memory-bound. `None` disables
/// batching. The 1.5 default is measured: 1.5-2.0 are ahead of 1.2 by
/// ~3% per full iteration on the landmark-heavy whole-Hessian
/// factorization, all are neutral elsewhere, and 3.0 loses outright --
/// past ~2, the padding's arithmetic outruns the traffic it saves.
/// Since the batch product accumulates straight into the panel, the
/// ratio carries no memory cost: only the span-by-depth pack buffers
/// remain, in the noise at any measured ratio.
#[derive(Clone, Debug)]
pub struct SupernodalParams {
    pub relax: Option<Vec<(usize, f64)>>,
    pub batch_ratio: Option<f64>,
}

impl Default for SupernodalParams {
    fn default() -> Self {
        Self {
            relax: Some(vec![(4, 1.0), (16, 0.8), (48, 0.1), (usize::MAX, 0.05)]),
            batch_ratio: Some(1.5),
        }
    }
}

/// A descendant wider than this never joins a batch: its own update is
/// already arithmetic-bound.
const BATCH_DEPTH_MAX: usize = 16;
/// Depth cap per batch, bounding the packed operand buffers.
const BATCH_K_MAX: usize = 512;

/// What a supernodal pass can fail with: the symbolic analysis rejects
/// a factor a [`ValueIndex`] cannot address, the numeric factorization
/// a matrix that is not positive definite.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SupernodalError {
    /// The factor holds more scalars than a [`ValueIndex`] addresses.
    IndexOverflow { required: u64 },
    /// A diagonal pivot was not strictly positive; the caller raises
    /// the damping and retries, as with the other factorizations.
    NotPositiveDefinite,
}

/// Where one stored tile of the matrix lands inside the factor panels.
#[derive(Clone, Copy, Debug)]
pub struct TileTarget {
    /// Offset of the tile's (0, 0) scalar in the factor value buffer.
    pub dst: ValueIndex,
    /// Column stride at the target -- the panel's scalar height.
    pub stride: u32,
    /// The tile is written transposed (an upper tile crossing to the
    /// lower factor, or a diagonal tile mirrored into its lower half).
    pub trans: bool,
}

/// The symbolic factorization: supernode structure, patterns, panel
/// layout, descendant lists and the seed scatter map. Built once per
/// sparsity structure; every damped attempt reuses it.
pub struct SupernodalSymbolic {
    /// scalar dimension
    n: usize,
    /// block dimension
    nblk: usize,
    /// number of supernodes
    ns: usize,
    /// block elimination order: `order[k]` = old block eliminated k-th
    order: Vec<u32>,
    /// inverse: `inv[old]` = position in the elimination order
    inv: Vec<u32>,
    /// permuted scalar column starts, `nblk + 1`
    sc_start: Vec<u32>,
    /// permuted block column -> supernode
    index_to_super: Vec<u32>,
    /// supernode -> first permuted block column, `ns + 1`
    sup_begin: Vec<u32>,
    /// supernodal elimination tree, `NONE` at a root
    super_etree: Vec<u32>,
    /// pattern bounds per supernode, `ns + 1`
    pat_ptr: Vec<u32>,
    /// concatenated patterns: permuted block columns below each
    /// supernode, sorted ascending
    pat_blk: Vec<u32>,
    /// per pattern entry, the scalar row of the block inside the panel
    pat_row: Vec<u32>,
    /// panel value offsets, `ns + 1`
    val_ptr: Vec<ValueIndex>,
    /// panel scalar heights
    nrows: Vec<u32>,
    /// descendant bounds per supernode, `ns + 1`
    desc_ptr: Vec<u32>,
    /// concatenated descendant lists, sorted ascending
    desc_idx: Vec<u32>,
    /// per descendant pair, the split of the descendant's pattern:
    /// (first entry inside the target, entries inside the target)
    desc_split: Vec<(u32, u32)>,
    /// old (unpermuted) scalar column starts per old block
    old_start: Vec<u32>,
    /// largest descendant update, in scalars
    max_update: usize,
    /// widest supernode, in scalar columns
    max_ncols: usize,
    /// update batches: one-past-last pair index per bucket (empty when
    /// batching is off; buckets partition each target's pair range)
    bucket_end: Vec<u32>,
    /// bucket range per target, `ns + 1` (empty when batching is off)
    tb_ptr: Vec<u32>,
    /// per bucket, the union target span: row_lo, row_hi, col_lo, col_hi
    bucket_span: Vec<[u32; 4]>,
    /// per bucket, the packed depth (sum of member widths)
    bucket_k: Vec<u32>,
    /// packed-operand scratch bounds for the batched path
    max_a_cat: usize,
    max_b_cat: usize,
    /// how many descendant pairs landed in multi-member buckets
    batched_pairs: usize,
    /// per stored tile of the matrix, its panel target
    tiles: Vec<TileTarget>,
    /// tile indices grouped by target supernode, `tile_ptr` bounds
    tile_ptr: Vec<u32>,
    tile_order: Vec<u32>,
    /// factor scalar count (panel volumes)
    n_vals: usize,
    /// factorization cost, one squared-height term per scalar column
    flops: f64,
}

impl SupernodalSymbolic {
    /// Analyse a symmetric block matrix under an optional block
    /// elimination order (`order[k]` = block eliminated k-th, e.g. from
    /// [`crate::nd::order_graph`]); `None` is the natural order.
    pub fn new(
        a: &SymbolicSparseBlockColMat<SparseIndex>,
        order: Option<&[usize]>,
        params: &SupernodalParams,
    ) -> Result<Self, SupernodalError> {
        let nblk = a.nblk_cols();
        assert_eq!(a.nblk_rows(), nblk, "the matrix must be square in blocks");
        assert_eq!(a.nrows(), a.ncols());
        let n = a.ncols();

        // Permutation, both directions, in block units.
        let order: Vec<u32> = match order {
            Some(o) => {
                assert_eq!(o.len(), nblk);
                o.iter().map(|&b| b as u32).collect()
            }
            None => (0..nblk as u32).collect(),
        };
        let mut inv = vec![NONE; nblk];
        for (k, &b) in order.iter().enumerate() {
            debug_assert!(inv[b as usize] == NONE, "order names a block twice");
            inv[b as usize] = k as u32;
        }

        // Permuted scalar widths and column starts.
        let width_p: Vec<u32> =
            (0..nblk).map(|k| a.col_span(order[k] as usize).len() as u32).collect();
        let mut sc_start = vec![0u32; nblk + 1];
        for k in 0..nblk {
            sc_start[k + 1] = sc_start[k] + width_p[k];
        }
        debug_assert_eq!(sc_start[nblk] as usize, n);

        // Symmetric block adjacency, old indices, no diagonal.
        let g = Graph::of_blocks(a);

        // Block elimination tree and column counts, block and scalar
        // units in one up-looking pass over the permuted columns.
        let mut parent = vec![NONE; nblk];
        let mut visited = vec![NONE; nblk];
        let mut cc_blk = vec![0u32; nblk];
        let mut cc_sc = vec![0u32; nblk];
        for k in 0..nblk {
            visited[k] = k as u32;
            cc_blk[k] = 1;
            cc_sc[k] = width_p[k];
            for &nbr in g.neighbours(order[k] as usize) {
                let mut i = inv[nbr] as usize;
                if i >= k {
                    continue;
                }
                while visited[i] != k as u32 {
                    let next = if parent[i] == NONE {
                        parent[i] = k as u32;
                        k
                    } else {
                        parent[i] as usize
                    };
                    cc_blk[i] += 1;
                    cc_sc[i] += width_p[k];
                    visited[i] = k as u32;
                    i = next;
                }
            }
        }

        // Fundamental supernodes: a column joins its predecessor's
        // supernode iff it is the predecessor's parent, its only child,
        // and the patterns nest (block counts differ by exactly one).
        let mut child_count = vec![0u32; nblk];
        for k in 0..nblk {
            if parent[k] != NONE {
                child_count[parent[k] as usize] += 1;
            }
        }
        let mut fund_begin: Vec<u32> = vec![0];
        for k in 1..nblk {
            let merge = parent[k - 1] == k as u32
                && child_count[k] == 1
                && cc_blk[k - 1] == cc_blk[k] + 1;
            if !merge {
                fund_begin.push(k as u32);
            }
        }
        fund_begin.push(nblk as u32);
        let nf = fund_begin.len() - 1;

        // Fundamental supernodal etree.
        let mut fund_of = vec![0u32; nblk];
        for s in 0..nf {
            for k in fund_begin[s]..fund_begin[s + 1] {
                fund_of[k as usize] = s as u32;
            }
        }
        let fund_parent = |s: usize| -> u32 {
            let last = fund_begin[s + 1] as usize - 1;
            if parent[last] == NONE { NONE } else { fund_of[parent[last] as usize] }
        };

        // Relaxed amalgamation, scalar units. A supernode may absorb
        // the live supernode immediately before it when that one is its
        // child in the supernodal etree and the zero-fill budget holds.
        let sc_of = |s: usize, begin: &[u32]| -> u32 {
            sc_start[fund_begin[s + 1] as usize] - sc_start[begin[s] as usize]
        };
        let mut begin: Vec<u32> = (0..nf).map(|s| fund_begin[s]).collect();
        let mut alive = vec![true; nf];
        let mut rep: Vec<u32> = (0..nf as u32).collect();
        if let Some(relax) = &params.relax {
            let mut ncols_sc: Vec<u64> = (0..nf).map(|s| sc_of(s, &begin) as u64).collect();
            let degree_sc: Vec<u64> = (0..nf)
                .map(|s| {
                    let last = fund_begin[s + 1] as usize - 1;
                    (cc_sc[last] - width_p[last]) as u64
                })
                .collect();
            let mut nzeros = vec![0u64; nf];
            let mut prev: Vec<u32> = (0..nf).map(|s| s.wrapping_sub(1) as u32).collect();
            let find = |rep: &mut [u32], mut s: u32| -> u32 {
                while rep[s as usize] != s {
                    let up = rep[rep[s as usize] as usize];
                    rep[s as usize] = up;
                    s = up;
                }
                s
            };
            for p in 1..nf {
                loop {
                    let c = prev[p];
                    if c == NONE {
                        break;
                    }
                    let c = c as usize;
                    let cp = fund_parent(c);
                    if cp == NONE || find(&mut rep, cp) != p as u32 {
                        break;
                    }
                    // faer's budget: pad the child's columns to the
                    // parent's height, accept per the relax table.
                    let new_zeros =
                        (ncols_sc[p] + degree_sc[p] - degree_sc[c]) * ncols_sc[c];
                    let total_zeros = new_zeros + nzeros[p] + nzeros[c];
                    let ok = if new_zeros == 0 {
                        true
                    } else {
                        let combined = ncols_sc[p] + ncols_sc[c];
                        let expanded =
                            combined * (combined + 1) / 2 + degree_sc[p] * combined;
                        relax.iter().any(|&(max_n, max_z)| {
                            combined as usize <= max_n
                                && (expanded as f64) * max_z >= total_zeros as f64
                        })
                    };
                    if !ok {
                        break;
                    }
                    ncols_sc[p] += ncols_sc[c];
                    nzeros[p] = total_zeros;
                    begin[p] = begin[c];
                    alive[c] = false;
                    rep[c] = p as u32;
                    prev[p] = prev[c];
                }
            }
        }

        // Compact to the final supernodes.
        let mut sup_begin: Vec<u32> = Vec::with_capacity(nf + 1);
        for s in 0..nf {
            if alive[s] {
                sup_begin.push(begin[s]);
            }
        }
        sup_begin.push(nblk as u32);
        let ns = sup_begin.len() - 1;
        let mut index_to_super = vec![0u32; nblk];
        for s in 0..ns {
            for k in sup_begin[s]..sup_begin[s + 1] {
                index_to_super[k as usize] = s as u32;
            }
        }
        let mut super_etree = vec![NONE; ns];
        for s in 0..ns {
            let last = sup_begin[s + 1] as usize - 1;
            if parent[last] != NONE {
                super_etree[s] = index_to_super[parent[last] as usize];
            }
        }

        // Pattern bounds from the block degrees (exact: the merged
        // supernode's pattern is its last column's), then the patterns
        // by an up-looking reach. Emission in ascending column order
        // leaves every pattern sorted.
        let mut pat_ptr = vec![0u32; ns + 1];
        for s in 0..ns {
            let last = sup_begin[s + 1] as usize - 1;
            pat_ptr[s + 1] = pat_ptr[s] + (cc_blk[last] - 1);
        }
        let mut pat_blk = vec![0u32; pat_ptr[ns] as usize];
        {
            let mut pos: Vec<u32> = pat_ptr[..ns].to_vec();
            let mut vis = vec![NONE; ns];
            for k in 0..nblk {
                let sk = index_to_super[k] as usize;
                vis[sk] = k as u32;
                for &nbr in g.neighbours(order[k] as usize) {
                    let i = inv[nbr] as usize;
                    if i >= k {
                        continue;
                    }
                    let mut d = index_to_super[i] as usize;
                    while vis[d] != k as u32 {
                        pat_blk[pos[d] as usize] = k as u32;
                        pos[d] += 1;
                        vis[d] = k as u32;
                        d = super_etree[d] as usize;
                    }
                }
            }
            for s in 0..ns {
                assert_eq!(
                    pos[s], pat_ptr[s + 1],
                    "pattern fill disagrees with the degree bound"
                );
            }
        }

        // Panel layout: scalar rows per pattern entry, heights, value
        // offsets (checked against the ValueIndex range), flops.
        let mut pat_row = vec![0u32; pat_blk.len()];
        let mut nrows = vec![0u32; ns];
        let mut val_ptr = vec![0 as ValueIndex; ns + 1];
        let mut n_vals: u64 = 0;
        let mut flops = 0.0f64;
        for s in 0..ns {
            let ncols = sc_start[sup_begin[s + 1] as usize] - sc_start[sup_begin[s] as usize];
            let mut row = ncols;
            for p in pat_ptr[s]..pat_ptr[s + 1] {
                pat_row[p as usize] = row;
                row += width_p[pat_blk[p as usize] as usize];
            }
            nrows[s] = row;
            if n_vals <= ValueIndex::MAX as u64 {
                val_ptr[s] = n_vals as ValueIndex;
            }
            n_vals += row as u64 * ncols as u64;
            let (h, q) = (row as f64, ncols as f64);
            // sum_{c=0..q} (h - c)^2
            flops += q * h * h - h * q * (q - 1.0) + q * (q - 1.0) * (2.0 * q - 1.0) / 6.0;
        }
        if n_vals > ValueIndex::MAX as u64 {
            return Err(SupernodalError::IndexOverflow { required: n_vals });
        }
        val_ptr[ns] = n_vals as ValueIndex;

        // Descendant lists: transpose the pattern's supernode
        // occurrences. Patterns are sorted, so occurrences of one
        // target are consecutive; emitting s ascending leaves each
        // target's list sorted.
        let mut desc_ptr = vec![0u32; ns + 1];
        for s in 0..ns {
            let mut last_t = NONE;
            for p in pat_ptr[s]..pat_ptr[s + 1] {
                let t = index_to_super[pat_blk[p as usize] as usize];
                if t != last_t {
                    desc_ptr[t as usize + 1] += 1;
                    last_t = t;
                }
            }
        }
        for s in 0..ns {
            desc_ptr[s + 1] += desc_ptr[s];
        }
        let mut desc_idx = vec![0u32; desc_ptr[ns] as usize];
        let mut desc_split = vec![(0u32, 0u32); desc_ptr[ns] as usize];
        let mut max_update = 0usize;
        {
            // One entry per (target, descendant) pair, with the
            // descendant's pattern split precomputed so the numeric
            // loop never searches: entries [start, start+mid) lie in
            // the target's columns, everything from start participates
            // as update rows.
            let row_at = |s: usize, rel: u32| -> u32 {
                let p = pat_ptr[s] + rel;
                if p < pat_ptr[s + 1] { pat_row[p as usize] } else { nrows[s] }
            };
            let mut pos: Vec<u32> = desc_ptr[..ns].to_vec();
            for s in 0..ns {
                let mut p = pat_ptr[s];
                while p < pat_ptr[s + 1] {
                    let t = index_to_super[pat_blk[p as usize] as usize];
                    let start = p - pat_ptr[s];
                    let mut mid = 0u32;
                    while p < pat_ptr[s + 1]
                        && index_to_super[pat_blk[p as usize] as usize] == t
                    {
                        mid += 1;
                        p += 1;
                    }
                    desc_idx[pos[t as usize] as usize] = s as u32;
                    desc_split[pos[t as usize] as usize] = (start, mid);
                    pos[t as usize] += 1;
                    let m = (nrows[s] - row_at(s, start)) as usize;
                    let k = (row_at(s, start + mid) - row_at(s, start)) as usize;
                    max_update = max_update.max(m * k);
                }
            }
        }

        // The tile source map. A stored tile (i, j), i <= j, holds
        // A[rows(i), cols(j)]; in the permuted lower factor it lands at
        // block (max, min) of the permuted pair -- as-is when the panel
        // column is old j, transposed when it is old i (and a diagonal
        // tile mirrors its upper triangle into the panel's lower).
        let sup_col_start =
            |s: usize| -> u32 { sc_start[sup_begin[s] as usize] };
        let mut tiles = Vec::with_capacity(a.nblocks());
        let mut tile_super = vec![0u32; a.nblocks()];
        for j_old in 0..nblk {
            for b in a.col_range(j_old) {
                let i_old = a.blk_row(b);
                let (pi, pj) = (inv[i_old] as usize, inv[j_old] as usize);
                let (c, r, trans) = if i_old == j_old {
                    (pi, pi, true)
                } else if pj < pi {
                    (pj, pi, false)
                } else {
                    (pi, pj, true)
                };
                let s = index_to_super[c] as usize;
                let col_off = sc_start[c] - sup_col_start(s);
                let row_off = if r < sup_begin[s + 1] as usize {
                    debug_assert!(r >= sup_begin[s] as usize);
                    sc_start[r] - sup_col_start(s)
                } else {
                    let pat = &pat_blk[pat_ptr[s] as usize..pat_ptr[s + 1] as usize];
                    let p = pat.binary_search(&(r as u32)).expect(
                        "a stored tile's row block is missing from the factor pattern",
                    );
                    pat_row[pat_ptr[s] as usize + p]
                };
                let dst = val_ptr[s]
                    + row_off as ValueIndex
                    + col_off as ValueIndex * nrows[s] as ValueIndex;
                tile_super[tiles.len()] = s as u32;
                tiles.push(TileTarget { dst, stride: nrows[s], trans });
            }
        }
        // Tiles grouped by target supernode, so the seed can run
        // panel-by-panel right after each panel is zeroed.
        let mut tile_ptr = vec![0u32; ns + 1];
        for &s in &tile_super {
            tile_ptr[s as usize + 1] += 1;
        }
        for s in 0..ns {
            tile_ptr[s + 1] += tile_ptr[s];
        }
        let mut tile_order = vec![0u32; tiles.len()];
        {
            let mut pos: Vec<u32> = tile_ptr[..ns].to_vec();
            for (b, &s) in tile_super.iter().enumerate() {
                tile_order[pos[s as usize] as usize] = b as u32;
                pos[s as usize] += 1;
            }
        }

        let old_start: Vec<u32> =
            (0..nblk).map(|b| a.col_span(b).start as u32).collect();
        let max_ncols = (0..ns)
            .map(|s| {
                (sc_start[sup_begin[s + 1] as usize] - sc_start[sup_begin[s] as usize]) as usize
            })
            .max()
            .unwrap_or(0);

        // Update batching: partition each target's descendant pairs into
        // consecutive buckets whose zero-padded joint update stays within
        // `batch_ratio` of the members' individual flops. Spans are in the
        // target panel's scalar coordinates, so the batched product is one
        // dense sub-panel.
        let mut bucket_end: Vec<u32> = Vec::new();
        let mut tb_ptr: Vec<u32> = Vec::new();
        let mut bucket_span: Vec<[u32; 4]> = Vec::new();
        let mut bucket_k: Vec<u32> = Vec::new();
        let mut max_a_cat = 0usize;
        let mut max_b_cat = 0usize;
        let mut batched_pairs = 0usize;
        if let Some(ratio) = params.batch_ratio {
            tb_ptr.push(0);
            // Target row of a permuted block inside t's panel, rebuilt per
            // target exactly as the numeric pass does.
            let mut blk_row = vec![0u32; nblk];
            for t in 0..ns {
                let col0 = sc_start[sup_begin[t] as usize];
                for k in sup_begin[t]..sup_begin[t + 1] {
                    blk_row[k as usize] = sc_start[k as usize] - col0;
                }
                for p in pat_ptr[t]..pat_ptr[t + 1] {
                    blk_row[pat_blk[p as usize] as usize] = pat_row[p as usize];
                }
                // Current bucket state.
                let mut members = 0u32;
                let mut span = [0u32; 4];
                let mut kk = 0usize;
                let mut flops_ind = 0f64;
                let flush = |members: &mut u32,
                             span: &mut [u32; 4],
                             kk: &mut usize,
                             flops_ind: &mut f64,
                             end_pair: u32,
                             bucket_end: &mut Vec<u32>,
                             bucket_span: &mut Vec<[u32; 4]>,
                             bucket_k: &mut Vec<u32>,
                             batched_pairs: &mut usize,
                             max_a: &mut usize,
                             max_b: &mut usize| {
                    if *members == 0 {
                        return;
                    }
                    bucket_end.push(end_pair);
                    bucket_span.push(*span);
                    bucket_k.push(*kk as u32);
                    if *members > 1 {
                        *batched_pairs += *members as usize;
                        let rs = (span[1] - span[0]) as usize;
                        let cs = (span[3] - span[2]) as usize;
                        *max_a = (*max_a).max(rs * *kk);
                        *max_b = (*max_b).max(cs * *kk);
                    }
                    *members = 0;
                    *kk = 0;
                    *flops_ind = 0.0;
                };
                for pi in desc_ptr[t]..desc_ptr[t + 1] {
                    let d = desc_idx[pi as usize] as usize;
                    let (start, mid) = desc_split[pi as usize];
                    let dq = (sc_start[sup_begin[d + 1] as usize]
                        - sc_start[sup_begin[d] as usize]) as usize;
                    let dpat = &pat_blk[pat_ptr[d] as usize..pat_ptr[d + 1] as usize];
                    let drow = &pat_row[pat_ptr[d] as usize..pat_ptr[d + 1] as usize];
                    // The pair's own span in target coordinates.
                    let first = dpat[start as usize] as usize;
                    let last = dpat[dpat.len() - 1] as usize;
                    let row_lo = blk_row[first];
                    let row_hi = blk_row[last] + (sc_start[last + 1] - sc_start[last]);
                    let fmid = dpat[start as usize] as usize;
                    let lmid = dpat[(start + mid - 1) as usize] as usize;
                    let col_lo = sc_start[fmid] - col0;
                    let col_hi = sc_start[lmid + 1] - col0;
                    let m_i = (nrows[d] - drow[start as usize]) as f64;
                    let kw_i = (col_hi - col_lo) as f64;
                    let f_i = m_i * kw_i * dq as f64;
                    let joinable = dq <= BATCH_DEPTH_MAX;
                    let mut push_single = true;
                    if joinable && members > 0 && kk + dq <= BATCH_K_MAX {
                        let u = [
                            span[0].min(row_lo),
                            span[1].max(row_hi),
                            span[2].min(col_lo),
                            span[3].max(col_hi),
                        ];
                        let batched = (u[1] - u[0]) as f64
                            * (u[3] - u[2]) as f64
                            * (kk + dq) as f64;
                        if batched <= ratio * (flops_ind + f_i) {
                            span = u;
                            kk += dq;
                            flops_ind += f_i;
                            members += 1;
                            push_single = false;
                        }
                    }
                    if push_single {
                        flush(
                            &mut members, &mut span, &mut kk, &mut flops_ind, pi,
                            &mut bucket_end, &mut bucket_span, &mut bucket_k,
                            &mut batched_pairs, &mut max_a_cat, &mut max_b_cat,
                        );
                        span = [row_lo, row_hi, col_lo, col_hi];
                        kk = dq;
                        flops_ind = f_i;
                        members = 1;
                        if !joinable {
                            flush(
                                &mut members, &mut span, &mut kk, &mut flops_ind, pi + 1,
                                &mut bucket_end, &mut bucket_span, &mut bucket_k,
                                &mut batched_pairs, &mut max_a_cat, &mut max_b_cat,
                            );
                        }
                    }
                }
                flush(
                    &mut members, &mut span, &mut kk, &mut flops_ind, desc_ptr[t + 1],
                    &mut bucket_end, &mut bucket_span, &mut bucket_k,
                    &mut batched_pairs, &mut max_a_cat, &mut max_b_cat,
                );
                tb_ptr.push(bucket_end.len() as u32);
            }
        }

        Ok(Self {
            n,
            nblk,
            ns,
            order,
            inv,
            sc_start,
            index_to_super,
            sup_begin,
            super_etree,
            pat_ptr,
            pat_blk,
            pat_row,
            val_ptr,
            nrows,
            desc_ptr,
            desc_idx,
            desc_split,
            old_start,
            max_update,
            max_ncols,
            bucket_end,
            tb_ptr,
            bucket_span,
            bucket_k,
            max_a_cat,
            max_b_cat,
            batched_pairs,
            tiles,
            tile_ptr,
            tile_order,
            n_vals: n_vals as usize,
            flops,
        })
    }

    /// Scalar dimension of the matrix.
    pub fn dim(&self) -> usize {
        self.n
    }

    /// Block dimension of the matrix.
    pub fn n_blk(&self) -> usize {
        self.nblk
    }

    /// The supernode a permuted block column belongs to.
    pub fn supernode_of(&self, permuted_block: usize) -> usize {
        self.index_to_super[permuted_block] as usize
    }

    /// The supernode's panel in the factor value buffer.
    pub fn panel_range(&self, s: usize) -> core::ops::Range<usize> {
        self.val_ptr[s] as usize..self.val_ptr[s + 1] as usize
    }

    /// Number of supernodes.
    pub fn n_supernodes(&self) -> usize {
        self.ns
    }

    /// Scalars in the factor buffer (panel volumes, the diagonal
    /// blocks' unused upper triangles included).
    pub fn factor_val_count(&self) -> usize {
        self.n_vals
    }

    /// The factor's structural scalar count: the lower triangle only,
    /// amalgamation padding included. Comparable to a scalar symbolic
    /// factorization's `len_val` when amalgamation is off.
    pub fn factor_scalar_nnz(&self) -> u64 {
        let mut nnz = 0u64;
        for s in 0..self.ns {
            let q = (self.sc_start[self.sup_begin[s + 1] as usize]
                - self.sc_start[self.sup_begin[s] as usize]) as u64;
            let h = self.nrows[s] as u64;
            nnz += q * (q + 1) / 2 + q * (h - q);
        }
        nnz
    }

    /// The factorization's cost: one squared-height term per scalar
    /// column, the same yardstick as
    /// [`envelope_flops`](crate::envelope::envelope_flops).
    pub fn flops(&self) -> f64 {
        self.flops
    }

    /// The supernode's block-column range, in elimination order.
    pub fn supernode_cols(&self, s: usize) -> core::ops::Range<usize> {
        self.sup_begin[s] as usize..self.sup_begin[s + 1] as usize
    }

    /// The supernode's panel shape: scalar columns and scalar rows
    /// (columns included).
    pub fn supernode_dims(&self, s: usize) -> (usize, usize) {
        let q = self.sc_start[self.sup_begin[s + 1] as usize]
            - self.sc_start[self.sup_begin[s] as usize];
        (q as usize, self.nrows[s] as usize)
    }

    /// The supernode's below-diagonal pattern: permuted block columns,
    /// sorted ascending.
    pub fn supernode_pattern(&self, s: usize) -> &[u32] {
        &self.pat_blk[self.pat_ptr[s] as usize..self.pat_ptr[s + 1] as usize]
    }

    /// The supernodes that update `s`, sorted ascending.
    pub fn descendants(&self, s: usize) -> &[u32] {
        &self.desc_idx[self.desc_ptr[s] as usize..self.desc_ptr[s + 1] as usize]
    }

    /// The pattern entries' scalar rows inside the panel, parallel to
    /// [`supernode_pattern`](Self::supernode_pattern).
    pub fn supernode_pattern_rows(&self, s: usize) -> &[u32] {
        &self.pat_row[self.pat_ptr[s] as usize..self.pat_ptr[s + 1] as usize]
    }

    /// The supernode's parent in the supernodal elimination tree, or
    /// `None` at a root.
    pub fn supernode_parent(&self, s: usize) -> Option<usize> {
        (self.super_etree[s] != NONE).then_some(self.super_etree[s] as usize)
    }

    /// The seed scatter map, one entry per stored tile of the matrix in
    /// storage order.
    pub fn tile_targets(&self) -> &[TileTarget] {
        &self.tiles
    }

    /// The block elimination order the analysis ran under.
    pub fn order(&self) -> &[u32] {
        &self.order
    }

    /// How many descendant pairs landed in multi-member update batches
    /// (0 when batching is off or nothing qualified).
    pub fn batched_pairs(&self) -> usize {
        self.batched_pairs
    }

    /// Where an old (unpermuted) block sits in the elimination order.
    pub fn position_of(&self, old_block: usize) -> usize {
        self.inv[old_block] as usize
    }
}

/// Approximate-minimum-degree order of the BLOCK graph: faer's amd run
/// on the block adjacency, so blocks stay whole by construction. On the
/// pose-graph benchmarks it matches scalar AMD's fill while ordering a
/// graph 3-6x smaller. Returns the block elimination order
/// [`SupernodalSymbolic::new`] takes.
pub fn amd_block_order(a: &SymbolicSparseBlockColMat<SparseIndex>) -> Vec<usize> {
    let nblk = a.nblk_cols();
    let (_, _, bcp, bri, _) = a.parts();
    let pattern =
        faer::sparse::SymbolicSparseColMatRef::new_checked(nblk, nblk, bcp, None, bri);
    let mut fwd = vec![0 as SparseIndex; nblk];
    let mut inv = vec![0 as SparseIndex; nblk];
    let mut mem = vec![
        core::mem::MaybeUninit::<u8>::uninit();
        faer::sparse::linalg::amd::order_maybe_unsorted_scratch::<SparseIndex>(
            nblk,
            a.nblocks(),
        )
        .unaligned_bytes_required()
    ];
    faer::sparse::linalg::amd::order(
        &mut fwd,
        &mut inv,
        pattern,
        faer::sparse::linalg::amd::Control::default(),
        faer::dyn_stack::MemStack::new(&mut mem),
    )
    .expect("amd on a checked block pattern");
    fwd.into_iter().map(|b| b as usize).collect()
}

fn one<T: SchurReal>() -> T {
    T::from_f64(1.0)
}

fn minus_one<T: SchurReal>() -> T {
    T::ZERO - one::<T>()
}


/// One descendant's contribution to the current panel: a triangular
/// GEMM for the mid-by-mid lower half plus a rectangular one into
/// packed scratch (the mid part's upper triangle lands in the diagonal
/// block's unused storage, so the scatter needn't mask it), then a
/// block-run scatter.
#[allow(clippy::too_many_arguments)]
fn apply_pair<T: SchurReal>(
    sym: &SupernodalSymbolic,
    pi: usize,
    head: &[T],
    panel: &mut [T],
    blk_row: &[u32],
    upd: &mut [T],
    runs: &mut Vec<(u32, u32, u32)>,
    h: usize,
    col0: u32,
) {
    let d = sym.desc_idx[pi] as usize;
    let (start, mid) = sym.desc_split[pi];
    let dq = {
        let (dq, _) = sym.supernode_dims(d);
        dq
    };
    let dh = sym.nrows[d] as usize;
    let dpat_row = sym.supernode_pattern_rows(d);
    let dpat_blk = sym.supernode_pattern(d);
    let r0 = dpat_row[start as usize] as usize;
    let m = dh - r0;
    let kend = start + mid;
    let kw = (if (kend as usize) < dpat_row.len() {
        dpat_row[kend as usize] as usize
    } else {
        dh
    }) - r0;
    let dpanel = &head[sym.val_ptr[d] as usize..sym.val_ptr[d + 1] as usize];

    let b_v = unsafe {
        faer::MatRef::from_raw_parts(dpanel.as_ptr().add(r0), kw, dq, 1, dh as isize)
    };

    // When the target rows and columns are each one contiguous range
    // (banded and trajectory-shaped patterns, mostly), accumulate the
    // two GEMMs straight into the panel: no product scratch, no
    // scatter. The mid block lands at (tc0, tc0) because a block's
    // diagonal-panel row equals its column offset.
    {
        let first = dpat_blk[start as usize] as usize;
        let last = dpat_blk[dpat_blk.len() - 1] as usize;
        let lmid = dpat_blk[(kend - 1) as usize] as usize;
        let rows_contig = (blk_row[last] as usize
            + (sym.sc_start[last + 1] - sym.sc_start[last]) as usize
            - blk_row[first] as usize)
            == m;
        let cols_contig = (sym.sc_start[lmid + 1] - sym.sc_start[first]) as usize == kw;
        if rows_contig && cols_contig {
            let tc0 = (sym.sc_start[first] - col0) as usize;
            let top = unsafe {
                faer::MatMut::from_raw_parts_mut(
                    panel.as_mut_ptr().add(tc0 * h + tc0),
                    kw,
                    kw,
                    1,
                    h as isize,
                )
            };
            faer::linalg::matmul::triangular::matmul(
                top,
                faer::linalg::matmul::triangular::BlockStructure::TriangularLower,
                faer::Accum::Add,
                b_v,
                faer::linalg::matmul::triangular::BlockStructure::Rectangular,
                b_v.transpose(),
                faer::linalg::matmul::triangular::BlockStructure::Rectangular,
                minus_one::<T>(),
                faer::Par::Seq,
            );
            if m > kw {
                let a_bot = unsafe {
                    faer::MatRef::from_raw_parts(
                        dpanel.as_ptr().add(r0 + kw),
                        m - kw,
                        dq,
                        1,
                        dh as isize,
                    )
                };
                let bot = unsafe {
                    faer::MatMut::from_raw_parts_mut(
                        panel.as_mut_ptr().add(tc0 * h + tc0 + kw),
                        m - kw,
                        kw,
                        1,
                        h as isize,
                    )
                };
                faer::linalg::matmul::matmul(
                    bot,
                    faer::Accum::Add,
                    a_bot,
                    b_v.transpose(),
                    minus_one::<T>(),
                    faer::Par::Seq,
                );
            }
            return;
        }
    }

    let u_top = unsafe {
        faer::MatMut::from_raw_parts_mut(upd.as_mut_ptr(), kw, kw, 1, m as isize)
    };
    faer::linalg::matmul::triangular::matmul(
        u_top,
        faer::linalg::matmul::triangular::BlockStructure::TriangularLower,
        faer::Accum::Replace,
        b_v,
        faer::linalg::matmul::triangular::BlockStructure::Rectangular,
        b_v.transpose(),
        faer::linalg::matmul::triangular::BlockStructure::Rectangular,
        one::<T>(),
        faer::Par::Seq,
    );
    if m > kw {
        let a_bot = unsafe {
            faer::MatRef::from_raw_parts(dpanel.as_ptr().add(r0 + kw), m - kw, dq, 1, dh as isize)
        };
        let u_bot = unsafe {
            faer::MatMut::from_raw_parts_mut(upd.as_mut_ptr().add(kw), m - kw, kw, 1, m as isize)
        };
        faer::linalg::matmul::matmul(
            u_bot,
            faer::Accum::Replace,
            a_bot,
            b_v.transpose(),
            one::<T>(),
            faer::Par::Seq,
        );
    }

    // Row runs: (target row, update row, width) per pattern block from
    // `start` on -- every one a contiguous segment.
    runs.clear();
    for e in start as usize..dpat_blk.len() {
        let kb = dpat_blk[e] as usize;
        let dr = dpat_row[e] as usize - r0;
        let w = sym.sc_start[kb + 1] - sym.sc_start[kb];
        runs.push((blk_row[kb], dr as u32, w));
    }
    let mut cc = 0usize;
    for f in start as usize..kend as usize {
        let kf = dpat_blk[f] as usize;
        let wf = (sym.sc_start[kf + 1] - sym.sc_start[kf]) as usize;
        let tc0 = (sym.sc_start[kf] - col0) as usize;
        for c in 0..wf {
            let pcol = (tc0 + c) * h;
            let ucol = (cc + c) * m;
            for &(tr, dr, w) in runs.iter() {
                let dst = pcol + tr as usize;
                let srcx = ucol + dr as usize;
                for r in 0..w as usize {
                    panel[dst + r] = panel[dst + r] - upd[srcx + r];
                }
            }
        }
        cc += wf;
    }
}

/// A bucket of consecutive descendants applied as ONE update: their
/// panels packed (zero-padded) into a joint A and B over the union
/// target span, then one GEMM accumulating straight into the panel --
/// the span is a contiguous sub-panel, so no product buffer and no
/// scatter, one pass over the target instead of one per member.
#[allow(clippy::too_many_arguments)]
fn apply_bucket<T: SchurReal>(
    sym: &SupernodalSymbolic,
    b: usize,
    p0: u32,
    p1: u32,
    head: &[T],
    panel: &mut [T],
    blk_row: &[u32],
    a_cat: &mut [T],
    b_cat: &mut [T],
    h: usize,
    col0: u32,
) {
    let span = sym.bucket_span[b];
    let (row_lo, col_lo) = (span[0] as usize, span[2] as usize);
    let rs = (span[1] - span[0]) as usize;
    let cs = (span[3] - span[2]) as usize;
    let kk = sym.bucket_k[b] as usize;
    a_cat[..rs * kk].fill(T::ZERO);
    b_cat[..cs * kk].fill(T::ZERO);

    let mut koff = 0usize;
    for pi in p0..p1 {
        let d = sym.desc_idx[pi as usize] as usize;
        let (start, mid) = sym.desc_split[pi as usize];
        let dq = {
            let (dq, _) = sym.supernode_dims(d);
            dq
        };
        let dh = sym.nrows[d] as usize;
        let dpat = sym.supernode_pattern(d);
        let drow = sym.supernode_pattern_rows(d);
        let dpanel = &head[sym.val_ptr[d] as usize..sym.val_ptr[d + 1] as usize];
        for e in start as usize..dpat.len() {
            let kb = dpat[e] as usize;
            let w = (sym.sc_start[kb + 1] - sym.sc_start[kb]) as usize;
            let ar = blk_row[kb] as usize - row_lo;
            let a0 = drow[e] as usize;
            for t in 0..dq {
                a_cat[(koff + t) * rs + ar..(koff + t) * rs + ar + w]
                    .clone_from_slice(&dpanel[a0 + t * dh..a0 + t * dh + w]);
            }
        }
        for f in start as usize..(start + mid) as usize {
            let kf = dpat[f] as usize;
            let wf = (sym.sc_start[kf + 1] - sym.sc_start[kf]) as usize;
            let bc = (sym.sc_start[kf] - col0) as usize - col_lo;
            let b0 = drow[f] as usize;
            for t in 0..dq {
                b_cat[(koff + t) * cs + bc..(koff + t) * cs + bc + wf]
                    .clone_from_slice(&dpanel[b0 + t * dh..b0 + t * dh + wf]);
            }
        }
        koff += dq;
    }
    debug_assert_eq!(koff, kk);

    let a_v = unsafe { faer::MatRef::from_raw_parts(a_cat.as_ptr(), rs, kk, 1, rs as isize) };
    let b_v = unsafe { faer::MatRef::from_raw_parts(b_cat.as_ptr(), cs, kk, 1, cs as isize) };
    let dst = unsafe {
        faer::MatMut::from_raw_parts_mut(
            panel.as_mut_ptr().add(col_lo * h + row_lo),
            rs,
            cs,
            1,
            h as isize,
        )
    };
    faer::linalg::matmul::matmul(
        dst,
        faer::Accum::Add,
        a_v,
        b_v.transpose(),
        minus_one::<T>(),
        faer::Par::Seq,
    );
}

/// Reusable workspace for [`supernodal_factorize`] and
/// [`supernodal_solve`]. Allocates nothing on construction; each call
/// grows what it needs to the symbolic-derived bound and never
/// allocates again -- across damped attempts, and across problems (a
/// bigger structure regrows it).
#[derive(Default)]
pub struct SupernodalContext<T> {
    upd: Vec<T>,
    blk_row: Vec<u32>,
    runs: Vec<(u32, u32, u32)>,
    a_cat: Vec<T>,
    b_cat: Vec<T>,
    chol_mem: Vec<core::mem::MaybeUninit<u8>>,
    x: Vec<T>,
    tmp: Vec<T>,
}

impl<T> SupernodalContext<T> {
    pub fn new() -> Self {
        Self {
            upd: Vec::new(),
            blk_row: Vec::new(),
            runs: Vec::new(),
            a_cat: Vec::new(),
            b_cat: Vec::new(),
            chol_mem: Vec::new(),
            x: Vec::new(),
            tmp: Vec::new(),
        }
    }
}

/// Factor `L L^T = P A P^T` into the panels, left-looking by supernode.
///
/// `a` is the matrix the symbolic analysis was built from (same
/// structure, any values); the permutation and the upper-to-lower
/// transposition live in the precomputed tile map, so the matrix is
/// read exactly once, tile by tile. Damping is the caller's: write it
/// into `a`'s diagonal tiles before calling, as the other block routes
/// do. The factor buffer and the context are caller-owned and reused
/// across attempts; each panel is zeroed and seeded at its own turn,
/// so no pass over the whole buffer precedes the work.
pub fn supernodal_factorize<T: SchurReal>(
    sym: &SupernodalSymbolic,
    a: &SparseBlockColMat<SparseIndex, T>,
    factor: &mut [T],
    ctx: &mut SupernodalContext<T>,
) -> Result<(), SupernodalError> {
    assert_eq!(factor.len(), sym.factor_val_count());
    let asym = a.symbolic();
    assert_eq!(asym.nblocks(), sym.tiles.len());

    ctx.upd.resize(sym.max_update, T::ZERO);
    ctx.blk_row.resize(sym.nblk, 0);
    ctx.a_cat.resize(sym.max_a_cat, T::ZERO);
    ctx.b_cat.resize(sym.max_b_cat, T::ZERO);
    let chol_req = faer::linalg::cholesky::llt::factor::cholesky_in_place_scratch::<T>(
        sym.max_ncols,
        faer::Par::Seq,
        faer::Spec::default(),
    )
    .unaligned_bytes_required();
    ctx.chol_mem.resize(chol_req, core::mem::MaybeUninit::uninit());
    let (upd, blk_row, runs, a_cat, b_cat, chol_mem) = (
        &mut ctx.upd,
        &mut ctx.blk_row,
        &mut ctx.runs,
        &mut ctx.a_cat,
        &mut ctx.b_cat,
        &mut ctx.chol_mem,
    );

    let vals = a.vals();
    for s in 0..sym.ns {
        let (q, h) = sym.supernode_dims(s);
        let col0 = sym.sc_start[sym.sup_begin[s] as usize];

        // Permuted block -> scalar row inside this panel.
        for k in sym.sup_begin[s]..sym.sup_begin[s + 1] {
            blk_row[k as usize] = sym.sc_start[k as usize] - col0;
        }
        for p in sym.pat_ptr[s]..sym.pat_ptr[s + 1] {
            blk_row[sym.pat_blk[p as usize] as usize] = sym.pat_row[p as usize];
        }

        let (head, tail) = factor.split_at_mut(sym.val_ptr[s] as usize);
        let panel = &mut tail[..h * q];

        // Zero and seed this panel now, while it is about to be hot:
        // every stored tile of its columns lands at its precomputed
        // target, columns contiguous when the orientations agree,
        // transposed otherwise.
        panel.fill(T::ZERO);
        let base = sym.val_ptr[s] as usize;
        for &bi in &sym.tile_order[sym.tile_ptr[s] as usize..sym.tile_ptr[s + 1] as usize] {
            let b = bi as usize;
            let t = sym.tiles[b];
            let (wi, wj) = asym.block_dims(b);
            let src = &vals[asym.val_range(b)];
            let stride = t.stride as usize;
            let dst0 = t.dst as usize - base;
            if !t.trans {
                for c in 0..wj {
                    panel[dst0 + c * stride..dst0 + c * stride + wi]
                        .clone_from_slice(&src[c * wi..(c + 1) * wi]);
                }
            } else {
                for c in 0..wi {
                    for r in 0..wj {
                        panel[dst0 + c * stride + r] = src[c + r * wi];
                    }
                }
            }
        }

        // Left-looking: subtract every descendant's contribution --
        // pair by pair, or bucket by bucket when the analysis batched
        // them.
        if sym.bucket_end.is_empty() {
            for pi in sym.desc_ptr[s]..sym.desc_ptr[s + 1] {
                apply_pair(sym, pi as usize, head, panel, blk_row, upd, runs, h, col0);
            }
        } else {
            let mut p0 = sym.desc_ptr[s];
            for b in sym.tb_ptr[s]..sym.tb_ptr[s + 1] {
                let p1 = sym.bucket_end[b as usize];
                if p1 - p0 == 1 {
                    apply_pair(sym, p0 as usize, head, panel, blk_row, upd, runs, h, col0);
                } else {
                    apply_bucket(
                        sym, b as usize, p0, p1, head, panel, blk_row, a_cat, b_cat, h, col0,
                    );
                }
                p0 = p1;
            }
        }

        // Dense diagonal factor, then the panel below it.
        let top = unsafe {
            faer::MatMut::from_raw_parts_mut(panel.as_mut_ptr(), q, q, 1, h as isize)
        };
        let stack = faer::dyn_stack::MemStack::new(chol_mem);
        faer::linalg::cholesky::llt::factor::cholesky_in_place(
            top,
            faer::linalg::cholesky::llt::factor::LltRegularization::default(),
            faer::Par::Seq,
            stack,
            faer::Spec::default(),
        )
        .map_err(|_| SupernodalError::NotPositiveDefinite)?;
        if h > q {
            let l11 =
                unsafe { faer::MatRef::from_raw_parts(panel.as_ptr(), q, q, 1, h as isize) };
            let bot = unsafe {
                faer::MatMut::from_raw_parts_mut(panel.as_mut_ptr().add(q), h - q, q, 1, h as isize)
            };
            faer::linalg::triangular_solve::solve_lower_triangular_in_place(
                l11,
                bot.transpose_mut(),
                faer::Par::Seq,
            );
        }
    }
    Ok(())
}

/// Solve `A x = rhs` in place from a factor produced by
/// [`supernodal_factorize`]. The permutation is applied on entry and
/// undone on exit; `rhs` stays in the matrix's own ordering.
pub fn supernodal_solve<T: SchurReal>(
    sym: &SupernodalSymbolic,
    factor: &[T],
    rhs: &mut [T],
    ctx: &mut SupernodalContext<T>,
) {
    assert_eq!(factor.len(), sym.factor_val_count());
    assert_eq!(rhs.len(), sym.n);
    let max_below = (0..sym.ns)
        .map(|s| {
            let (q, h) = sym.supernode_dims(s);
            h - q
        })
        .max()
        .unwrap_or(0);
    ctx.x.resize(sym.n, T::ZERO);
    ctx.tmp.resize(max_below, T::ZERO);
    let (x, tmp) = (&mut ctx.x, &mut ctx.tmp);

    // Into elimination order.
    for k in 0..sym.nblk {
        let ob = sym.order[k] as usize;
        let w = (sym.sc_start[k + 1] - sym.sc_start[k]) as usize;
        let src = sym.old_start[ob] as usize;
        let dst = sym.sc_start[k] as usize;
        x[dst..dst + w].clone_from_slice(&rhs[src..src + w]);
    }

    // Forward sweep: L y = x.
    for s in 0..sym.ns {
        let (q, h) = sym.supernode_dims(s);
        let c0 = sym.sc_start[sym.sup_begin[s] as usize] as usize;
        let panel = &factor[sym.val_ptr[s] as usize..sym.val_ptr[s + 1] as usize];
        let l11 = unsafe { faer::MatRef::from_raw_parts(panel.as_ptr(), q, q, 1, h as isize) };
        {
            let x_top =
                faer::col::ColMut::from_slice_mut(&mut x[c0..c0 + q]).as_mat_mut();
            faer::linalg::triangular_solve::solve_lower_triangular_in_place(
                l11,
                x_top,
                faer::Par::Seq,
            );
        }
        if h > q {
            let bot = unsafe {
                faer::MatRef::from_raw_parts(panel.as_ptr().add(q), h - q, q, 1, h as isize)
            };
            let x_top = faer::col::ColRef::from_slice(&x[c0..c0 + q]).as_mat();
            let t = unsafe {
                faer::MatMut::from_raw_parts_mut(tmp.as_mut_ptr(), h - q, 1, 1, (h - q) as isize)
            };
            faer::linalg::matmul::matmul(
                t,
                faer::Accum::Replace,
                bot,
                x_top,
                one::<T>(),
                faer::Par::Seq,
            );
            for (e, &k) in sym.supernode_pattern(s).iter().enumerate() {
                let dr = sym.supernode_pattern_rows(s)[e] as usize - q;
                let w = (sym.sc_start[k as usize + 1] - sym.sc_start[k as usize]) as usize;
                let dst = sym.sc_start[k as usize] as usize;
                for r in 0..w {
                    x[dst + r] = x[dst + r] - tmp[dr + r];
                }
            }
        }
    }

    // Backward sweep: L^T x = y.
    for s in (0..sym.ns).rev() {
        let (q, h) = sym.supernode_dims(s);
        let c0 = sym.sc_start[sym.sup_begin[s] as usize] as usize;
        let panel = &factor[sym.val_ptr[s] as usize..sym.val_ptr[s + 1] as usize];
        let l11 = unsafe { faer::MatRef::from_raw_parts(panel.as_ptr(), q, q, 1, h as isize) };
        if h > q {
            for (e, &k) in sym.supernode_pattern(s).iter().enumerate() {
                let dr = sym.supernode_pattern_rows(s)[e] as usize - q;
                let w = (sym.sc_start[k as usize + 1] - sym.sc_start[k as usize]) as usize;
                let src = sym.sc_start[k as usize] as usize;
                tmp[dr..dr + w].clone_from_slice(&x[src..src + w]);
            }
            let bot = unsafe {
                faer::MatRef::from_raw_parts(panel.as_ptr().add(q), h - q, q, 1, h as isize)
            };
            let t = unsafe {
                faer::MatRef::from_raw_parts(tmp.as_ptr(), h - q, 1, 1, (h - q) as isize)
            };
            let x_top =
                faer::col::ColMut::from_slice_mut(&mut x[c0..c0 + q]).as_mat_mut();
            faer::linalg::matmul::matmul(
                x_top,
                faer::Accum::Add,
                bot.transpose(),
                t,
                minus_one::<T>(),
                faer::Par::Seq,
            );
        }
        let x_top = faer::col::ColMut::from_slice_mut(&mut x[c0..c0 + q]).as_mat_mut();
        faer::linalg::triangular_solve::solve_upper_triangular_in_place(
            l11.transpose(),
            x_top,
            faer::Par::Seq,
        );
    }

    // Back to the matrix's ordering.
    for k in 0..sym.nblk {
        let ob = sym.order[k] as usize;
        let w = (sym.sc_start[k + 1] - sym.sc_start[k]) as usize;
        let dst = sym.old_start[ob] as usize;
        let src = sym.sc_start[k] as usize;
        rhs[dst..dst + w].clone_from_slice(&x[src..src + w]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bsc::SymbolicSparseBlockColMat;
    use crate::nd::{order_graph, Graph, NdParams};

    /// Deterministic PRNG, the house pattern.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() >> 33) as usize % n
        }
    }

    /// A random symmetric block structure: random widths, a chain for
    /// connectivity, extra random couplings, full diagonal.
    fn random_structure(
        nblk: usize,
        max_width: usize,
        extra_per_col: usize,
        seed: u64,
    ) -> SymbolicSparseBlockColMat<SparseIndex> {
        let mut rng = Lcg(seed);
        let mut part: Vec<SparseIndex> = vec![0];
        for _ in 0..nblk {
            let w = 1 + rng.below(max_width);
            part.push(part.last().unwrap() + w as SparseIndex);
        }
        let mut cells: Vec<(usize, usize)> = Vec::new();
        for j in 0..nblk {
            cells.push((part[j] as usize, part[j] as usize));
            if j > 0 {
                cells.push((part[j - 1] as usize, part[j] as usize));
            }
            for _ in 0..extra_per_col {
                let i = rng.below(j + 1);
                cells.push((part[i] as usize, part[j] as usize));
            }
        }
        let (sym, _) = SymbolicSparseBlockColMat::from_scalar_coords(
            part.clone(),
            part,
            cells.len(),
            |k| cells[k],
        );
        sym
    }

    /// Textbook scalar etree + column counts on the tile-expanded,
    /// permuted pattern -- an independent reference for the block
    /// analysis.
    fn scalar_factor_nnz(
        sym: &SymbolicSparseBlockColMat<SparseIndex>,
        order: &[usize],
    ) -> u64 {
        let nblk = sym.nblk_cols();
        let mut inv = vec![0usize; nblk];
        for (k, &b) in order.iter().enumerate() {
            inv[b] = k;
        }
        let mut sc_start = vec![0usize; nblk + 1];
        for k in 0..nblk {
            sc_start[k + 1] = sc_start[k] + sym.col_span(order[k]).len();
        }
        let n = sc_start[nblk];
        // Upper scalar adjacency of the permuted expansion.
        let mut cols: Vec<Vec<usize>> = vec![Vec::new(); n];
        for j in 0..nblk {
            for b in sym.col_range(j) {
                let i = sym.blk_row(b);
                let (pi, pj) = (inv[i], inv[j]);
                let ri = sc_start[pi]..sc_start[pi] + sym.col_span(i).len();
                let rj = sc_start[pj]..sc_start[pj] + sym.col_span(j).len();
                for r in ri.clone() {
                    for c in rj.clone() {
                        if r < c {
                            cols[c].push(r);
                        } else if c < r {
                            cols[r].push(c);
                        }
                    }
                }
            }
        }
        let mut parent = vec![usize::MAX; n];
        let mut visited = vec![usize::MAX; n];
        let mut count = vec![0u64; n];
        for j in 0..n {
            visited[j] = j;
            count[j] = 1;
            for &e in &cols[j] {
                let mut i = e;
                while visited[i] != j {
                    let next = if parent[i] == usize::MAX {
                        parent[i] = j;
                        j
                    } else {
                        parent[i]
                    };
                    count[i] += 1;
                    visited[i] = j;
                    i = next;
                }
            }
        }
        count.iter().sum()
    }

    /// With amalgamation off, the block analysis must reproduce the
    /// scalar factorization's fill exactly -- natural, reversed and
    /// nested-dissection orders alike.
    #[test]
    fn fundamental_fill_matches_the_scalar_reference() {
        let no_relax = SupernodalParams { relax: None, ..Default::default() };
        for seed in [1u64, 7, 42] {
            for nblk in [1usize, 2, 13, 60] {
                let sym = random_structure(nblk, 4, 2, seed);
                let natural: Vec<usize> = (0..nblk).collect();
                let reversed: Vec<usize> = (0..nblk).rev().collect();
                let nd = order_graph(&Graph::of_blocks(&sym), NdParams::default());
                for order in [&natural, &reversed, &nd] {
                    let sn = SupernodalSymbolic::new(&sym, Some(order), &no_relax).unwrap();
                    assert_eq!(
                        sn.factor_scalar_nnz(),
                        scalar_factor_nnz(&sym, order),
                        "seed {} nblk {} order {:?}",
                        seed,
                        nblk,
                        &order[..order.len().min(8)]
                    );
                }
            }
        }
    }

    /// Amalgamation only ever pads: the panel volume grows or stays,
    /// the supernode count shrinks or stays, and the structural fill
    /// (which counts the padding) never shrinks.
    #[test]
    fn amalgamation_only_grows_the_factor() {
        for seed in [3u64, 9] {
            let sym = random_structure(80, 3, 2, seed);
            let fund =
                SupernodalSymbolic::new(&sym, None, &SupernodalParams { relax: None, ..Default::default() }).unwrap();
            let relaxed =
                SupernodalSymbolic::new(&sym, None, &SupernodalParams::default()).unwrap();
            assert!(relaxed.factor_val_count() >= fund.factor_val_count());
            assert!(relaxed.n_supernodes() <= fund.n_supernodes());
            assert!(relaxed.factor_scalar_nnz() >= fund.factor_scalar_nnz());
        }
    }

    /// Structural invariants: sorted patterns strictly below their
    /// supernode, panel rows within bounds, supernode ranges contiguous
    /// and exhaustive, descendant lists sorted and consistent with the
    /// patterns.
    #[test]
    fn structure_invariants_hold() {
        for (order_kind, seed) in [(0, 11u64), (1, 12), (2, 13)] {
            let sym = random_structure(50, 4, 3, seed);
            let nblk = sym.nblk_cols();
            let order: Vec<usize> = match order_kind {
                0 => (0..nblk).collect(),
                1 => (0..nblk).rev().collect(),
                _ => order_graph(&Graph::of_blocks(&sym), NdParams::default()),
            };
            let sn = SupernodalSymbolic::new(&sym, Some(&order), &SupernodalParams::default())
                .unwrap();
            let ns = sn.n_supernodes();
            let mut col_cursor = 0usize;
            for s in 0..ns {
                let cols = sn.supernode_cols(s);
                assert_eq!(cols.start, col_cursor);
                col_cursor = cols.end;
                let (q, h) = sn.supernode_dims(s);
                assert!(q >= 1 && h >= q);
                let pat = sn.supernode_pattern(s);
                for w in pat.windows(2) {
                    assert!(w[0] < w[1], "pattern not strictly sorted");
                }
                for &k in pat {
                    assert!((k as usize) >= cols.end, "pattern block inside the supernode");
                }
                for &d in sn.descendants(s) {
                    assert!((d as usize) < s);
                    let dpat = sn.supernode_pattern(d as usize);
                    let hit = dpat.iter().any(|&k| {
                        let t = sn.supernode_cols(s);
                        (k as usize) >= t.start && (k as usize) < t.end
                    });
                    assert!(hit, "descendant without a pattern block in the target");
                }
            }
            assert_eq!(col_cursor, nblk);
            assert_eq!(sn.tile_targets().len(), sym.nblocks());
            assert_eq!(sn.dim(), sym.ncols());
        }
    }

    /// A random SPD block matrix on a random structure: values filled
    /// tile by tile (diagonal tiles upper-only, per the storage
    /// convention), mirrored into a dense twin, diagonal made dominant
    /// in both.
    fn random_spd(
        nblk: usize,
        max_width: usize,
        extra_per_col: usize,
        seed: u64,
    ) -> (SparseBlockColMat<SparseIndex, f64>, Vec<f64>, Vec<f64>) {
        let sym = random_structure(nblk, max_width, extra_per_col, seed);
        let n = sym.ncols();
        let mut rng = Lcg(seed ^ 0x9e3779b97f4a7c15);
        let mut a = SparseBlockColMat::<SparseIndex, f64>::zeroed(sym);
        let mut dense = vec![0.0f64; n * n];
        let asym = a.symbolic().clone();
        for j in 0..asym.nblk_cols() {
            let cj = asym.col_span(j).start;
            for b in asym.col_range(j) {
                let i = asym.blk_row(b);
                let ri = asym.row_span(i).start;
                let (wi, wj) = asym.block_dims(b);
                let base = asym.val_range(b).start;
                for cc in 0..wj {
                    for rr in 0..wi {
                        if i == j && rr > cc {
                            continue;
                        }
                        let v = (rng.below(2001) as f64 - 1000.0) / 1000.0;
                        a.vals_mut()[base + rr + cc * wi] = v;
                        dense[(ri + rr) + (cj + cc) * n] = v;
                        dense[(cj + cc) + (ri + rr) * n] = v;
                    }
                }
            }
        }
        // Dominance, written through both forms.
        for j in 0..asym.nblk_cols() {
            let b = asym.col_range(j).find(|&b| asym.blk_row(b) == j).unwrap();
            let (w, _) = asym.block_dims(b);
            let base = asym.val_range(b).start;
            let cj = asym.col_span(j).start;
            for k in 0..w {
                let boost = 2.0 * n as f64;
                a.vals_mut()[base + k * (w + 1)] += boost;
                dense[(cj + k) * (n + 1)] += boost;
            }
        }
        let rhs: Vec<f64> =
            (0..n).map(|_| (rng.below(2001) as f64 - 1000.0) / 500.0).collect();
        (a, dense, rhs)
    }

    fn rel_resid(dense: &[f64], n: usize, x: &[f64], rhs: &[f64]) -> f64 {
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for r in 0..n {
            let mut ax = 0.0;
            for c in 0..n {
                ax += dense[r + c * n] * x[c];
            }
            num += (ax - rhs[r]) * (ax - rhs[r]);
            den += rhs[r] * rhs[r];
        }
        (num / den).sqrt()
    }

    /// Factor + solve against the dense twin, across orders and with
    /// amalgamation on and off.
    #[test]
    fn factorize_and_solve_match_the_dense_reference() {
        for seed in [2u64, 21, 77] {
            for nblk in [1usize, 2, 17, 48] {
                let (a, dense, rhs) = random_spd(nblk, 4, 2, seed);
                let n = a.symbolic().ncols();
                let natural: Vec<usize> = (0..nblk).collect();
                let reversed: Vec<usize> = (0..nblk).rev().collect();
                let nd = order_graph(&Graph::of_blocks(a.symbolic()), NdParams::default());
                for order in [&natural, &reversed, &nd] {
                    for params in [
                        SupernodalParams { relax: None, ..Default::default() },
                        SupernodalParams::default(),
                        SupernodalParams { batch_ratio: Some(3.0), ..Default::default() },
                        SupernodalParams {
                            relax: None,
                            batch_ratio: Some(1.2),
                        },
                    ] {
                        let sn =
                            SupernodalSymbolic::new(a.symbolic(), Some(order), &params).unwrap();
                        let mut factor = vec![0.0f64; sn.factor_val_count()];
                        supernodal_factorize(&sn, &a, &mut factor, &mut SupernodalContext::new()).unwrap();
                        let mut x = rhs.clone();
                        supernodal_solve(&sn, &factor, &mut x, &mut SupernodalContext::new());
                        let resid = rel_resid(&dense, n, &x, &rhs);
                        assert!(
                            resid < 1e-10,
                            "resid {} seed {} nblk {} relax {}",
                            resid,
                            seed,
                            nblk,
                            params.relax.is_some(),
                        );
                    }
                }
            }
        }
    }

    /// The same in f32: storage, kernels and solve all at single
    /// precision, with the tolerance that precision supports.
    #[test]
    fn f32_factorize_and_solve_hold_their_tolerance() {
        let (a64, dense, rhs) = random_spd(40, 3, 2, 5);
        let sym32 = a64.symbolic().clone();
        let mut a = SparseBlockColMat::<SparseIndex, f32>::zeroed(sym32);
        for (dst, &src) in a.vals_mut().iter_mut().zip(a64.vals()) {
            *dst = src as f32;
        }
        let n = a.symbolic().ncols();
        let nd = order_graph(&Graph::of_blocks(a.symbolic()), NdParams::default());
        let sn = SupernodalSymbolic::new(a.symbolic(), Some(&nd), &SupernodalParams::default())
            .unwrap();
        let mut factor = vec![0.0f32; sn.factor_val_count()];
        supernodal_factorize(&sn, &a, &mut factor, &mut SupernodalContext::new()).unwrap();
        let mut x32: Vec<f32> = rhs.iter().map(|&v| v as f32).collect();
        supernodal_solve(&sn, &factor, &mut x32, &mut SupernodalContext::new());
        let x: Vec<f64> = x32.iter().map(|&v| v as f64).collect();
        let resid = rel_resid(&dense, n, &x, &rhs);
        assert!(resid < 1e-3, "resid {}", resid);
    }

    /// Batching must actually fire on the shape it exists for -- many
    /// narrow blocks eliminated first, all updating the same trailing
    /// blocks -- and change nothing about the answer.
    #[test]
    fn batched_updates_fire_and_match_the_unbatched_result() {
        // 40 narrow "landmark" blocks, each coupled to a window of wide
        // "pose" blocks at the end; landmarks eliminated first.
        let nlm = 40usize;
        let npose = 8usize;
        let mut part: Vec<SparseIndex> = vec![0];
        for _ in 0..nlm {
            part.push(part.last().unwrap() + 3);
        }
        for _ in 0..npose {
            part.push(part.last().unwrap() + 6);
        }
        let mut cells: Vec<(usize, usize)> = Vec::new();
        for b in 0..nlm + npose {
            cells.push((part[b] as usize, part[b] as usize));
        }
        for l in 0..nlm {
            let w0 = (l * npose) / nlm;
            for p in w0..(w0 + 3).min(npose) {
                cells.push((part[l] as usize, part[nlm + p] as usize));
            }
        }
        for p in 1..npose {
            cells.push((part[nlm + p - 1] as usize, part[nlm + p] as usize));
        }
        let (sym, _) = SymbolicSparseBlockColMat::from_scalar_coords(
            part.clone(),
            part,
            cells.len(),
            |k| cells[k],
        );
        let n = sym.ncols();
        // Values on that structure, SPD, mirrored dense.
        let mut rng = Lcg(31);
        let mut a = SparseBlockColMat::<SparseIndex, f64>::zeroed(sym);
        let mut dense = vec![0.0f64; n * n];
        let asym = a.symbolic().clone();
        for j in 0..asym.nblk_cols() {
            let cj = asym.col_span(j).start;
            for b in asym.col_range(j) {
                let i = asym.blk_row(b);
                let ri = asym.row_span(i).start;
                let (wi, wj) = asym.block_dims(b);
                let base = asym.val_range(b).start;
                for cc in 0..wj {
                    for rr in 0..wi {
                        if i == j && rr > cc {
                            continue;
                        }
                        let v = (rng.below(2001) as f64 - 1000.0) / 1000.0;
                        a.vals_mut()[base + rr + cc * wi] = v;
                        dense[(ri + rr) + (cj + cc) * n] = v;
                        dense[(cj + cc) + (ri + rr) * n] = v;
                    }
                }
            }
        }
        for j in 0..asym.nblk_cols() {
            let b = asym.col_range(j).find(|&b| asym.blk_row(b) == j).unwrap();
            let (w, _) = asym.block_dims(b);
            let base = asym.val_range(b).start;
            let cj = asym.col_span(j).start;
            for k in 0..w {
                a.vals_mut()[base + k * (w + 1)] += 2.0 * n as f64;
                dense[(cj + k) * (n + 1)] += 2.0 * n as f64;
            }
        }
        let rhs: Vec<f64> = (0..n).map(|_| (rng.below(2001) as f64 - 1000.0) / 500.0).collect();

        let solve_with = |batch: Option<f64>| -> (Vec<f64>, usize) {
            let sn = SupernodalSymbolic::new(
                a.symbolic(),
                None,
                &SupernodalParams { batch_ratio: batch, ..Default::default() },
            )
            .unwrap();
            let mut factor = vec![0.0f64; sn.factor_val_count()];
            supernodal_factorize(&sn, &a, &mut factor, &mut SupernodalContext::new()).unwrap();
            let mut x = rhs.clone();
            supernodal_solve(&sn, &factor, &mut x, &mut SupernodalContext::new());
            (x, sn.batched_pairs())
        };
        let (x_plain, b_plain) = solve_with(None);
        let (x_batch, b_batch) = solve_with(Some(3.0));
        assert_eq!(b_plain, 0);
        assert!(b_batch >= 4, "bucketing never fired: {} pairs", b_batch);
        assert!(rel_resid(&dense, n, &x_batch, &rhs) < 1e-10);
        for (a, b) in std::iter::zip(&x_plain, &x_batch) {
            assert!((a - b).abs() < 1e-9, "batched diverges: {} vs {}", a, b);
        }
    }

    /// One context serves repeated attempts and then a bigger problem:
    /// the buffers regrow as needed and the answers stay exact.
    #[test]
    fn a_context_is_reusable_across_attempts_and_problems() {
        let mut ctx = SupernodalContext::new();
        for (nblk, seed) in [(12usize, 4u64), (48, 5), (20, 6)] {
            let (a, dense, rhs) = random_spd(nblk, 4, 2, seed);
            let n = a.symbolic().ncols();
            let sn =
                SupernodalSymbolic::new(a.symbolic(), None, &SupernodalParams::default()).unwrap();
            let mut factor = vec![0.0f64; sn.factor_val_count()];
            for _ in 0..3 {
                supernodal_factorize(&sn, &a, &mut factor, &mut ctx).unwrap();
                let mut x = rhs.clone();
                supernodal_solve(&sn, &factor, &mut x, &mut ctx);
                let resid = rel_resid(&dense, n, &x, &rhs);
                assert!(resid < 1e-10, "resid {} nblk {}", resid, nblk);
            }
        }
    }

    /// An indefinite matrix is a clean error, not a factor of garbage.
    #[test]
    fn an_indefinite_matrix_is_rejected() {
        let (mut a, _, _) = random_spd(12, 3, 2, 8);
        let asym = a.symbolic().clone();
        let b = asym.col_range(5).find(|&b| asym.blk_row(b) == 5).unwrap();
        let base = asym.val_range(b).start;
        a.vals_mut()[base] = -1.0;
        let sn = SupernodalSymbolic::new(&asym, None, &SupernodalParams::default()).unwrap();
        let mut factor = vec![0.0f64; sn.factor_val_count()];
        assert_eq!(
            supernodal_factorize(&sn, &a, &mut factor, &mut SupernodalContext::new()),
            Err(SupernodalError::NotPositiveDefinite),
        );
    }

    /// A factor bigger than a ValueIndex addresses is an error from the
    /// symbolic phase, before anything numeric is allocated. The matrix
    /// itself fits the index comfortably (a 100-wide block band, 6e7
    /// stored scalars); its fill does not (about 1e10).
    #[test]
    fn an_unaddressable_factor_is_rejected() {
        let nblk = 2000usize;
        let w = 100usize;
        let band = 500usize;
        let part: Vec<SparseIndex> = (0..=nblk).map(|i| (i * w) as SparseIndex).collect();
        let mut cells: Vec<(usize, usize)> = Vec::new();
        for j in 0..nblk {
            cells.push((j * w, j * w));
            if j >= 1 {
                cells.push(((j - 1) * w, j * w));
            }
            if j >= band {
                cells.push(((j - band) * w, j * w));
            }
        }
        let (sym, _) = SymbolicSparseBlockColMat::from_scalar_coords(
            part.clone(),
            part,
            cells.len(),
            |k| cells[k],
        );
        match SupernodalSymbolic::new(&sym, None, &SupernodalParams::default()) {
            Err(SupernodalError::IndexOverflow { required }) => {
                assert!(required > ValueIndex::MAX as u64);
            }
            other => panic!("expected IndexOverflow, got {:?}", other.map(|_| ())),
        }
    }
}
