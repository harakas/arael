//! The host side of the device factorization: what the kernels need to know
//! about a [`SupernodalSymbolic`], flattened into integer arrays and cut into
//! the per-level work lists the launches walk.
//!
//! Everything here is built once per symbolic analysis and reused by every
//! numeric factorization; nothing depends on the matrix values.

// Index loops over parallel arrays (pattern blocks beside pattern rows).
#![allow(clippy::needless_range_loop)]

use std::collections::HashMap;

use arael_faer::bsc::SymbolicSparseBlockColMat;
use arael_faer::supernodal::SupernodalSymbolic;

/// The block index type of arael-faer's sparse matrices.
pub type SparseIndex = u32;

/// A piece of a small descendant update: at most PAIR_NI of the pair's
/// rows against PAIR_NJ of its columns, PAIR_EPT elements a thread.
pub const PAIR_NI: usize = 128;
pub const PAIR_NJ: usize = 64;
pub const PAIR_EPT: usize = 4;
/// The descendant's columns staged per pass of the small-update kernel.
pub const PAIR_KC: usize = 16;
/// Tile of the big-descendant update and of the panel's trailing update.
pub const TB: usize = 64;
/// The k slab of the big-descendant GEMM core.
pub const TK: usize = 16;
/// The panel step: columns factored per round of the three panel kernels.
pub const NB: usize = 32;

/// The trailing update (`k_syrk`) of panel step `step` in blocks of
/// `block` columns, a multiple of [`NB`]: the solved columns from `k0`,
/// `kn` of them at most, update the columns before `cend` (all of them at
/// `u32::MAX`). A step inside a block updates the rest of its block; the
/// block's last step updates everything past the block with the block's
/// columns, so the trailing part is read and written once per block. In
/// blocks of [`NB`] every step updates everything past it.
///
/// ```ignore
/// assert_eq!(syrk_update(5, 32), (160, 32, u32::MAX));
/// assert_eq!(syrk_update(5, 128), (160, 32, 256)); // inside the block [128, 256)
/// assert_eq!(syrk_update(7, 128), (128, 128, u32::MAX)); // the block's last step
/// ```
pub fn syrk_update(step: usize, block: usize) -> (u32, u32, u32) {
    let per = block / NB;
    let b = step / per;
    if step % per == per - 1 {
        ((b * block) as u32, block as u32, u32::MAX)
    } else {
        ((step * NB) as u32, NB as u32, ((b + 1) * block) as u32)
    }
}
/// Rows of x staged per pass of the backward GEMV.
pub const XR_CH: usize = 1024;

/// The header of the one `plan` buffer: where each array starts. A kernel
/// reads the offsets it needs once per unit; they stay out of the kernel's
/// compile-time identity, so a process compiles each kernel once, not once
/// per sparsity structure.
pub const H_DIMS: usize = 0;
pub const H_VAL_PTR: usize = 1;
pub const H_SUP_BEGIN: usize = 2;
pub const H_SC_START: usize = 3;
pub const H_PAIR_PTR: usize = 4;
pub const H_PR_D: usize = 5;
pub const H_PR_R0: usize = 6;
pub const H_PR_RMAP: usize = 7;
pub const H_PR_CMAP: usize = 8;
pub const H_RMAP: usize = 9;
pub const H_CMAP: usize = 10;
pub const H_ROW_SC: usize = 11;
pub const H_ROW_SC_OFF: usize = 12;
/// The permutation, for the device-side solve: the block elimination order,
/// the permuted block widths, and the matrix's scalar partition.
pub const H_ORDER: usize = 13;
pub const H_BLK_WIDTH: usize = 14;
pub const H_PART: usize = 15;
/// The solve's header of every supernode, [`SOLVE_HEAD`] words: its
/// columns q, its rows h, its first scalar column, its panel's offset, and
/// the offset of its rows below in the row map (`H_ROW_SC`).
pub const H_SOLVE_HEAD: usize = 16;
pub const SOLVE_HEAD: usize = 5;
/// Words of the header.
pub const HEADER: usize = 32;

/// How the work is cut. The defaults are the measured ones (docs/dev/GPU.md, E3).
#[derive(Clone, Copy, Debug)]
pub struct GpuSupernodalParams {
    /// Supernodes narrower than this (scalar columns) are factored by the
    /// one-block small kernel, unless their rows below need more than
    /// `small_flops`; the others by the batched panel steps. Capped to what
    /// one block holds: 256 columns, and the device's shared memory.
    pub small_q: u32,
    /// The most multiply-adds of a small supernode's solve of its rows below
    /// (`q^2 h / 2`) the one-block kernel takes: a taller supernode goes to
    /// the panel steps, whose solve runs on many blocks.
    pub small_flops: f64,
    /// A descendant update of at least this many multiply-adds goes through
    /// the GEMM-core kernel; smaller ones through the small-update kernel, unless
    /// `wide_pair_k` and `wide_pair_flops` send them there.
    pub big_pair_flops: f64,
    /// A descendant at least this many columns wide sends its updates of
    /// at least `wide_pair_flops` multiply-adds through the GEMM-core
    /// kernel too: it reuses each k slab across its tile, where the
    /// small-update kernel computes every element as a dot product of that
    /// length.
    pub wide_pair_k: u32,
    pub wide_pair_flops: f64,
    /// Multiply-adds per block of the solve's gather and GEMV kernels.
    pub solve_work: usize,
    /// Loads in flight per iteration of the backward dot products; `None`
    /// picks 8 for f64 and 4 for f32.
    pub lane_unroll: Option<usize>,
    /// A run of levels of one small supernode each (a path of the tree),
    /// without big updates, each of at most `chain_flops_f32` (in an f32
    /// factor) or `chain_flops_f64` (f64) multiply-adds of small updates and
    /// factor and `chain_items` update pieces, is factored by one block in
    /// one launch; 0 turns chains off. In f64 one block's multiply-add rate
    /// is the limit where f64 is slow.
    pub chain_flops_f32: f64,
    pub chain_flops_f64: f64,
    pub chain_items: u32,
    /// A run of levels of one supernode each, each of at most this many
    /// multiply-adds of solve (its gather, its GEMV, its two trsvs), is
    /// solved by one block in one launch forward and one backward; 0 turns
    /// it off.
    pub solve_chain_work: f64,
    /// Blocks per multiprocessor a level's updates are cut to fill: a level
    /// whose big tiles make fewer blocks has their updates split across
    /// blocks, which then subtract atomically; the small updates are cut
    /// into pieces of the level's share of the work.
    pub blocks_per_sm: u32,
    /// Columns per block of the panel steps' trailing update
    /// ([`syrk_update`]), a multiple of [`NB`]: in blocks of [`NB`] every
    /// step updates everything past it; in wider blocks the trailing part
    /// is read and written once per block. `None` takes the device's choice
    /// ([`Self::syrk_block_on`]).
    pub syrk_block: Option<u32>,
    /// Rows per block of the panel steps' `k_trsm`, one thread each: at
    /// least [`NB`] and a divisor of `NB * NB`. Smaller blocks spread a
    /// step's rows over more blocks, and every block copies the step's
    /// diagonal tile. `None` takes the device's choice
    /// ([`Self::trsm_rows_on`]).
    pub trsm_rows: Option<u32>,
}

impl GpuSupernodalParams {
    /// [`syrk_block`](Self::syrk_block) on the device `arch`, unset: 128 on
    /// an integrated GPU or the CPU device, [`NB`] on a discrete GPU.
    /// Measured on bundle adjustment and SLAM reduced systems: blocks of 128
    /// are faster on AMD integrated GPUs, whose memory bandwidth bounds the
    /// update, and slower on an NVIDIA discrete GPU, where the update's
    /// launches are bound by latency. Discrete AMD GPUs are not measured.
    ///
    /// ```ignore
    /// let block = GpuSupernodalParams::default().syrk_block_on(dev.arch());
    /// ```
    pub fn syrk_block_on(&self, arch: &crate::Arch) -> u32 {
        self.syrk_block.unwrap_or(if arch.integrated { 128 } else { NB as u32 })
    }

    /// [`trsm_rows`](Self::trsm_rows) on the device `arch`, unset: 64 on an
    /// integrated GPU or the CPU device, a warp (at least [`NB`]) on a
    /// discrete GPU. Measured on bundle adjustment and SLAM reduced systems:
    /// a warp per block is faster on an NVIDIA discrete GPU, where most
    /// `k_trsm` launches have fewer blocks than multiprocessors, and 64 rows
    /// on an AMD integrated GPU, which the launches fill, so the blocks'
    /// copies of the diagonal tile are the cost. Discrete AMD GPUs are not
    /// measured.
    ///
    /// ```ignore
    /// let rows = GpuSupernodalParams::default().trsm_rows_on(dev.arch());
    /// ```
    pub fn trsm_rows_on(&self, arch: &crate::Arch) -> u32 {
        self.trsm_rows.unwrap_or(if arch.integrated { 64 } else { arch.warp.max(NB as u32) })
    }
}

impl Default for GpuSupernodalParams {
    fn default() -> Self {
        GpuSupernodalParams {
            small_q: 96,
            small_flops: 3e5,
            big_pair_flops: 2e6,
            wide_pair_k: 64,
            wide_pair_flops: 1e5,
            chain_flops_f32: 3e5,
            chain_flops_f64: 0.0,
            chain_items: 32,
            solve_chain_work: 1e5,
            solve_work: 65536,
            lane_unroll: None,
            blocks_per_sm: 4,
            syrk_block: None,
            trsm_rows: None,
        }
    }
}

/// One descendant update (target `s` <- descendant `d`): the descendant's
/// rows from `r0` (`m` of them, `kw` of them landing in `s`'s columns),
/// placed by the plan's row and column maps.
#[derive(Clone)]
pub(crate) struct Pair {
    pub s: u32,
    pub d: u32,
    pub r0: u32,
    pub m: u32,
    pub kw: u32,
}

/// The symbolic's arrays in the kernels' form, plus the descendant pairs
/// with their row and column maps.
#[derive(Default)]
pub(crate) struct Plan {
    pub n: usize,
    pub nb: usize,
    pub ns: usize,
    pub nlevels: usize,
    pub factor_vals: usize,
    pub order: Vec<u32>,
    pub blk_width: Vec<u32>,
    pub sup_begin: Vec<u32>,
    pub dims: Vec<u32>,
    pub val_ptr: Vec<u32>,
    pub sc_start: Vec<u32>,
    pub part: Vec<u32>,
    pub level: Vec<usize>,
    /// 6 per stored tile: destination, stride, transposed, value offset, rows, columns.
    pub tiles: Vec<u32>,
    pub pairs: Vec<Pair>,
    pub pair_ptr: Vec<u32>,
    pub rmap: Vec<u32>,
    pub cmap: Vec<u32>,
    pub pr_rmap: Vec<u32>,
    pub pr_cmap: Vec<u32>,
    pub row_sc: Vec<u32>,
    pub row_sc_off: Vec<u32>,
}

impl Plan {
    /// The symbolic analysis of `asym` (the matrix's block structure, for the
    /// block widths and the stored tiles) read into flat arrays.
    pub fn new(sn: &SupernodalSymbolic, asym: &SymbolicSparseBlockColMat<SparseIndex>) -> Plan {
        let ns = sn.n_supernodes();
        let nb = sn.n_blk();
        let n = sn.dim();
        let part: Vec<u32> = asym.col_part().to_vec();
        let order: Vec<u32> = sn.order().to_vec();
        assert_eq!(order.len(), nb);
        let blk_width: Vec<u32> = (0..nb).map(|k| part[order[k] as usize + 1] - part[order[k] as usize]).collect();
        let mut sc_start = vec![0u32; nb + 1];
        for k in 0..nb {
            sc_start[k + 1] = sc_start[k] + blk_width[k];
        }
        let mut sup_begin = Vec::with_capacity(ns + 1);
        let mut dims = Vec::with_capacity(2 * ns);
        let mut val_ptr = Vec::with_capacity(ns + 1);
        let mut level = vec![0usize; ns];
        let mut nlevels = 0;
        for s in 0..ns {
            sup_begin.push(sn.supernode_cols(s).start as u32);
            let (q, h) = sn.supernode_dims(s);
            dims.extend_from_slice(&[q as u32, h as u32]);
            val_ptr.push(sn.panel_range(s).start as u32);
        }
        sup_begin.push(if ns > 0 { sn.supernode_cols(ns - 1).end as u32 } else { 0 });
        val_ptr.push(sn.factor_val_count() as u32);
        for s in 0..ns {
            if let Some(p) = sn.supernode_parent(s) {
                level[p] = level[p].max(level[s] + 1);
            }
            nlevels = nlevels.max(level[s] + 1);
        }
        // the stored tiles, in the matrix's block storage order, which is the
        // order of the symbolic's tile targets
        let targets = sn.tile_targets();
        assert_eq!(targets.len(), asym.nblocks());
        let mut tiles = Vec::with_capacity(6 * targets.len());
        for j in 0..asym.nblk_cols() {
            for b in asym.col_range(j) {
                let t = &targets[b];
                let (wr, wc) = asym.block_dims(b);
                tiles.extend_from_slice(&[t.dst, t.stride, t.trans as u32, asym.val_range(b).start as u32, wr as u32, wc as u32]);
            }
        }
        let mut p = Plan {
            n,
            nb,
            ns,
            nlevels,
            factor_vals: sn.factor_val_count(),
            order,
            blk_width,
            sup_begin,
            dims,
            val_ptr,
            sc_start,
            part,
            level,
            tiles,
            ..Plan::default()
        };
        p.pairs_and_maps(sn);
        p
    }

    /// The descendant pairs of every supernode with their flat row and
    /// column maps into the target panel, the running count of mid rows, and
    /// the permuted scalar index of every row below a diagonal block.
    fn pairs_and_maps(&mut self, sn: &SupernodalSymbolic) {
        self.pair_ptr.push(0);
        let mut row_of: HashMap<u32, u32> = HashMap::new();
        for s in 0..self.ns {
            let (c0, c1) = (self.sup_begin[s], self.sup_begin[s + 1]);
            let col0 = self.sc_start[c0 as usize];
            let pat_blk = sn.supernode_pattern(s);
            let pat_row = sn.supernode_pattern_rows(s);
            row_of.clear();
            for k in c0..c1 {
                row_of.insert(k, self.sc_start[k as usize] - col0);
            }
            for (e, &kb) in pat_blk.iter().enumerate() {
                row_of.insert(kb, pat_row[e]);
            }
            for &d in sn.descendants(s) {
                let d = d as usize;
                let dblk = sn.supernode_pattern(d);
                let drow = sn.supernode_pattern_rows(d);
                let mut start = 0;
                while start < dblk.len() && dblk[start] < c0 {
                    start += 1;
                }
                let mut mid = 0;
                while start + mid < dblk.len() && dblk[start + mid] < c1 {
                    mid += 1;
                }
                assert!(mid > 0, "descendant {d} of supernode {s} has no columns in it");
                let r0 = drow[start];
                let dh = self.dims[2 * d + 1];
                let m = dh - r0;
                let kw: u32 = (start..start + mid).map(|e| self.blk_width[dblk[e] as usize]).sum();
                // the target row of each of the pair's rows, the target column
                // of each of its first kw
                self.pr_rmap.push(self.rmap.len() as u32);
                self.pr_cmap.push(self.cmap.len() as u32);
                for e in start..dblk.len() {
                    let (kb, w) = (dblk[e], self.blk_width[dblk[e] as usize]);
                    debug_assert_eq!(drow[e] - r0, self.rmap.len() as u32 - self.pr_rmap[self.pr_rmap.len() - 1], "the pair's rows are contiguous");
                    self.rmap.extend(row_of[&kb]..row_of[&kb] + w);
                }
                for e in start..start + mid {
                    let (kb, w) = (dblk[e], self.blk_width[dblk[e] as usize]);
                    let tc = self.sc_start[kb as usize] - col0;
                    self.cmap.extend(tc..tc + w);
                }
                debug_assert_eq!(self.rmap.len() as u32 - self.pr_rmap[self.pr_rmap.len() - 1], m);
                debug_assert_eq!(self.cmap.len() as u32 - self.pr_cmap[self.pr_cmap.len() - 1], kw);
                self.pairs.push(Pair { s: s as u32, d: d as u32, r0, m, kw });
            }
            self.pair_ptr.push(self.pairs.len() as u32);
            self.row_sc_off.push(self.row_sc.len() as u32);
            for &kb in pat_blk {
                for k in 0..self.blk_width[kb as usize] {
                    self.row_sc.push(self.sc_start[kb as usize] + k);
                }
            }
            debug_assert_eq!(self.row_sc.len() as u32 - self.row_sc_off[s], self.dims[2 * s + 1] - self.dims[2 * s]);
        }
        self.row_sc_off.push(self.row_sc.len() as u32);
    }

    /// The plan arrays packed into one buffer behind the header that says
    /// where each starts.
    pub fn pack(&self) -> Vec<u32> {
        let mut buf: Vec<u32> = vec![0; HEADER];
        let pr_d: Vec<u32> = self.pairs.iter().map(|x| x.d).collect();
        let pr_r0: Vec<u32> = self.pairs.iter().map(|x| x.r0).collect();
        let head: Vec<u32> = (0..self.ns)
            .flat_map(|s| {
                let col0 = self.sc_start[self.sup_begin[s] as usize];
                [self.dims[2 * s], self.dims[2 * s + 1], col0, self.val_ptr[s], self.row_sc_off[s]]
            })
            .collect();
        let arrays: [(usize, &[u32]); 17] = [
            (H_ORDER, &self.order),
            (H_BLK_WIDTH, &self.blk_width),
            (H_PART, &self.part),
            (H_DIMS, &self.dims),
            (H_VAL_PTR, &self.val_ptr),
            (H_SUP_BEGIN, &self.sup_begin),
            (H_SC_START, &self.sc_start),
            (H_PAIR_PTR, &self.pair_ptr),
            (H_PR_D, &pr_d),
            (H_PR_R0, &pr_r0),
            (H_PR_RMAP, &self.pr_rmap),
            (H_PR_CMAP, &self.pr_cmap),
            (H_RMAP, &self.rmap),
            (H_CMAP, &self.cmap),
            (H_ROW_SC, &self.row_sc),
            (H_ROW_SC_OFF, &self.row_sc_off),
            (H_SOLVE_HEAD, &head),
        ];
        for (slot, v) in arrays {
            buf[slot] = buf.len() as u32;
            buf.extend_from_slice(v);
        }
        buf
    }
}

/// A big descendant's rows `[da, db)` and columns `[ca, cb)` landing in one tile.
struct BigRect {
    pair: u32,
    da: u32,
    db: u32,
    ca: u32,
    cb: u32,
}

/// One level's launches, as the kernels read them.
#[derive(Default)]
pub(crate) struct LevelWork {
    /// [`PAIR_ITEM`] per piece of a small descendant update: the
    /// descendant's panel at the pair's first row, its height and width, the
    /// target's panel and height, the pair's row and column maps (offsets
    /// into `rmap`, `cmap`), and the piece's rows `[i0, i1)` and columns
    /// `[j0, j1)` of the pair.
    pub pair_items: Vec<u32>,
    /// The multiply-adds of the small updates and of the small targets' factors.
    pub pair_work: f64,
    pub small_work: f64,
    /// The multiply-adds of the level's solve: its gather, GEMV and trsvs.
    pub solve_work: f64,
    /// [`BIG_ITEM`] per item: target, row0, col0, first rectangle, end, the
    /// k range `[k0, k1)` of a rectangle cut along k (`k1` 0: every k), and
    /// whether the tile is shared with other items (subtracted atomically).
    pub big_items: Vec<u32>,
    /// 5 per rectangle: pair, da, db, ca, cb.
    pub big_rects: Vec<u32>,
    pub small_targets: Vec<u32>,
    pub big_targets: Vec<u32>,
    pub level_sn: Vec<u32>,
    /// [`GATHER_ROW`] fields per mid row of the level's pairs, field by
    /// field (field f of row r at `f * rows + r`, so a warp's loads of a field
    /// are contiguous): the descendant's row at its column 0 (a factor
    /// offset), the target's x index, and the descendant (its height, width
    /// and first column in its solve header, `H_SOLVE_HEAD`).
    pub g_rows: Vec<u32>,
    /// The gather rows' fields while the level is built, one vector a field.
    g_fields: [Vec<u32>; GATHER_ROW],
    /// 2 per gather item: its first row of `g_rows` and its rows.
    pub g_items: Vec<u32>,
    /// 3 per GEMV item: supernode, columns [c0, c1).
    pub bg_items: Vec<u32>,
    /// The widest big target: the number of panel steps is `ceil(maxq_big / NB)`.
    pub maxq_big: u32,
    /// Per panel step, the trsm and syrk grid sizes, as the kernels count
    /// them (the syrk's for [`build_levels`]' blocks, [`syrk_update`]).
    pub trsm_grid: Vec<u32>,
    pub syrk_grid: Vec<u32>,
}

/// Words per big update item ([`LevelWork::big_items`]).
pub const BIG_ITEM: usize = 8;

/// Words per gather row ([`LevelWork::g_rows`]).
pub const GATHER_ROW: usize = 3;

/// The most rows of a gather item: one a thread of the gather's block.
pub const GATHER_ITEM_ROWS: u32 = 256;

/// Words per small update piece ([`LevelWork::pair_items`]).
pub const PAIR_ITEM: usize = 11;

/// The least work (multiply-adds) a small update piece is cut down to.
const MIN_PAIR_WORK: f64 = 2048.0;

/// The least work (multiply-adds) a big item is cut down to.
const MIN_ITEM_WORK: f64 = (TB * TB * TK * 2) as f64;

/// The level schedule of `p` on a device of `multiprocessors`: every
/// supernode's work sorted into its level, the panel steps' grids for
/// `k_syrk` in blocks of `syrk_block` columns and `k_trsm` in blocks of
/// `trsm_rows` rows.
pub(crate) fn build_levels(p: &Plan, kn: &GpuSupernodalParams, multiprocessors: u32, syrk_block: usize, trsm_rows: u32) -> Vec<LevelWork> {
    let mut lw: Vec<LevelWork> = (0..p.nlevels).map(|_| LevelWork::default()).collect();
    let mut g_works = vec![0usize; p.nlevels];
    let mut small_pairs: Vec<Vec<usize>> = vec![Vec::new(); p.nlevels];
    for s in 0..p.ns {
        let w = &mut lw[p.level[s]];
        let g_work = &mut g_works[p.level[s]];
        let q = p.dims[2 * s];
        let h = p.dims[2 * s + 1];
        if q < kn.small_q && (q as f64) * (q as f64) * (h as f64) / 2.0 <= kn.small_flops {
            w.small_targets.push(s as u32);
            let (q, h) = (q as f64, h as f64);
            w.small_work += q * q * q / 6.0 + (h - q) * q * q / 2.0;
        } else {
            w.big_targets.push(s as u32);
            w.maxq_big = w.maxq_big.max(q);
        }
        w.level_sn.push(s as u32);
        w.solve_work += (q as f64) * (q as f64) + (h - q) as f64 * q as f64;
        // the solve's gather: a record per mid row of s's pairs, appended to
        // the level's open item while it stays under the work bound and
        // GATHER_ITEM_ROWS
        let col0 = p.sc_start[p.sup_begin[s] as usize];
        for pi in p.pair_ptr[s] as usize..p.pair_ptr[s + 1] as usize {
            let pr = &p.pairs[pi];
            let d = pr.d as usize;
            let dq = p.dims[2 * d];
            w.solve_work += (pr.kw * dq) as f64;
            for i in 0..pr.kw {
                let row = w.g_fields[0].len() as u32;
                let tc = p.cmap[(p.pr_cmap[pi] + i) as usize];
                for (f, v) in [p.val_ptr[d] + pr.r0 + i, col0 + tc, d as u32].into_iter().enumerate() {
                    w.g_fields[f].push(v);
                }
                let ng = w.g_items.len() / 2;
                if ng > 0 && *g_work + dq as usize <= kn.solve_work && w.g_items[2 * ng - 1] < GATHER_ITEM_ROWS {
                    w.g_items[2 * ng - 1] += 1;
                    *g_work += dq as usize;
                } else {
                    w.g_items.extend_from_slice(&[row, 1]);
                    *g_work = dq as usize;
                }
            }
        }
        if h > q {
            let per = ((kn.solve_work / (h - q) as usize) as u32).max(1);
            let mut c0 = 0;
            while c0 < q {
                w.bg_items.extend_from_slice(&[s as u32, c0, q.min(c0 + per)]);
                c0 += per;
            }
        }
        // the pairs: the small ones cut into pieces once the level is
        // complete, the big ones as rectangles per tile
        let mut bigs = Vec::new();
        for pi in p.pair_ptr[s] as usize..p.pair_ptr[s + 1] as usize {
            let pr = &p.pairs[pi];
            let dq = p.dims[2 * pr.d as usize];
            let flops = pr.m as f64 * pr.kw as f64 * dq as f64;
            if flops >= kn.big_pair_flops || (dq >= kn.wide_pair_k && flops >= kn.wide_pair_flops) {
                bigs.push(pi);
            } else {
                small_pairs[p.level[s]].push(pi);
            }
        }
        // the big pairs' rectangles per tile of the target: the maps are monotonic,
        // so the rows of a pair landing in one tile row are a range, likewise the
        // columns; tiles above the diagonal are skipped
        let mut btiles: HashMap<u64, Vec<BigRect>> = HashMap::new();
        for &pi in &bigs {
            let pr = &p.pairs[pi];
            let rmap = &p.rmap[p.pr_rmap[pi] as usize..][..pr.m as usize];
            let cmap = &p.cmap[p.pr_cmap[pi] as usize..][..pr.kw as usize];
            let tb = TB as u32;
            let ranges = |map: &[u32]| -> Vec<(u32, u32, u32)> {
                let mut out = Vec::new();
                let mut i = 0usize;
                while i < map.len() {
                    let t = map[i] / tb;
                    let mut i1 = i;
                    while i1 < map.len() && map[i1] / tb == t {
                        i1 += 1;
                    }
                    out.push((t, i as u32, i1 as u32));
                    i = i1;
                }
                out
            };
            let rr = ranges(rmap);
            let cc = ranges(cmap);
            for r in &rr {
                for c in &cc {
                    if r.0 >= c.0 {
                        btiles
                            .entry(((r.0 as u64) << 20) | c.0 as u64)
                            .or_default()
                            .push(BigRect { pair: pi as u32, da: r.1, db: r.2, ca: c.1, cb: c.2 });
                    }
                }
            }
        }
        let mut bkeys: Vec<u64> = btiles.keys().copied().collect();
        bkeys.sort_unstable();
        for key in bkeys {
            let rs = &btiles[&key];
            let rect0 = (w.big_rects.len() / 5) as u32;
            for r in rs {
                w.big_rects.extend_from_slice(&[r.pair, r.da, r.db, r.ca, r.cb]);
            }
            w.big_items.extend_from_slice(&[
                s as u32,
                (key >> 20) as u32 * TB as u32,
                (key & 0xfffff) as u32 * TB as u32,
                rect0,
                (w.big_rects.len() / 5) as u32,
                0,
                0,
                0,
            ]);
        }
    }
    // the gather rows field by field
    for w in &mut lw {
        let fields = std::mem::take(&mut w.g_fields);
        w.g_rows = Vec::with_capacity(fields.iter().map(Vec::len).sum());
        for f in &fields {
            w.g_rows.extend_from_slice(f);
        }
    }
    let blocks = (multiprocessors * kn.blocks_per_sm) as usize;
    for (w, pairs) in lw.iter_mut().zip(&small_pairs) {
        split_big(w, p, blocks);
        pair_pieces(w, p, pairs, blocks);
    }
    // the panel steps' grid sizes
    for w in &mut lw {
        let mut j = 0u32;
        while j < w.maxq_big {
            let (k0, kn, cend) = syrk_update(j as usize / NB, syrk_block);
            let (mut tg, mut sg) = (0u32, 0u32);
            for &s in &w.big_targets {
                let (q, h) = (p.dims[2 * s as usize], p.dims[2 * s as usize + 1]);
                if k0 < q {
                    let (k1, c1) = ((k0 + kn).min(q), cend.min(q));
                    if k1 < c1 {
                        sg += (h - k1).div_ceil(TB as u32) * (c1 - k1).div_ceil(TB as u32);
                    }
                }
                if j < q {
                    let nb = (NB as u32).min(q - j);
                    tg += (h - j - nb).div_ceil(trsm_rows);
                }
            }
            w.trsm_grid.push(tg);
            w.syrk_grid.push(sg);
            j += NB as u32;
        }
    }
    lw
}

/// The runs of consecutive levels of `lw` that `k_chain` factors, one block
/// a run: levels of one small supernode (a path of the tree; the chain
/// factors a level's supernodes in turn), without big updates, each of at
/// most `chain_flops` multiply-adds and `chain_items` update pieces.
pub(crate) fn chains(lw: &[LevelWork], chain_flops: f64, chain_items: u32) -> Vec<std::ops::Range<usize>> {
    runs(lw, |w| {
        w.big_items.is_empty()
            && w.big_targets.is_empty()
            && w.small_targets.len() == 1
            && w.pair_work + w.small_work <= chain_flops
            && w.pair_items.len() / PAIR_ITEM <= chain_items as usize
    })
}

/// The runs of consecutive levels of `lw` that the solve's chain kernels
/// solve, one block a run: levels of one supernode, each of at most
/// `solve_chain_work` multiply-adds of solve.
pub(crate) fn solve_chains(lw: &[LevelWork], solve_chain_work: f64) -> Vec<std::ops::Range<usize>> {
    runs(lw, |w| w.level_sn.len() == 1 && w.solve_work <= solve_chain_work)
}

/// The maximal runs of consecutive levels of `lw` that `fits`.
fn runs(lw: &[LevelWork], fits: impl Fn(&LevelWork) -> bool) -> Vec<std::ops::Range<usize>> {
    let mut out = Vec::new();
    let mut l = 0;
    while l < lw.len() {
        if fits(&lw[l]) {
            let first = l;
            while l < lw.len() && fits(&lw[l]) {
                l += 1;
            }
            out.push(first..l);
        } else {
            l += 1;
        }
    }
    out
}

/// The level's small updates, `pairs`, cut into [`LevelWork::pair_items`]:
/// a pair's columns in runs of at most PAIR_NJ, its rows from the first
/// column of the run on (the rows above are above the diagonal) in runs of
/// at most PAIR_NI and PAIR_NI * PAIR_EPT elements, fewer while a piece's
/// work (elements times the descendant's width) is over the level's share
/// of `blocks` blocks.
fn pair_pieces(w: &mut LevelWork, p: &Plan, pairs: &[usize], blocks: usize) {
    let dq = |pi: usize| p.dims[2 * p.pairs[pi].d as usize];
    let total: f64 = pairs.iter().map(|&pi| p.pairs[pi].m as f64 * p.pairs[pi].kw as f64 * dq(pi) as f64).sum();
    w.pair_work = total;
    let cap = (total / blocks as f64).max(MIN_PAIR_WORK);
    for &pi in pairs {
        let pr = &p.pairs[pi];
        let (d, s) = (pr.d as usize, pr.s as usize);
        let head = [
            p.val_ptr[d] + pr.r0,
            p.dims[2 * d + 1],
            dq(pi),
            p.val_ptr[s],
            p.dims[2 * s + 1],
            p.pr_rmap[pi],
            p.pr_cmap[pi],
        ];
        let mut j0 = 0;
        while j0 < pr.kw {
            let j1 = pr.kw.min(j0 + PAIR_NJ as u32);
            let nj = j1 - j0;
            let by_work = (cap / (nj as f64 * dq(pi) as f64)) as u32;
            let ni = (PAIR_NI as u32).min((PAIR_NI * PAIR_EPT) as u32 / nj).min(by_work).max(1);
            let mut i0 = j0;
            while i0 < pr.m {
                let i1 = pr.m.min(i0 + ni);
                w.pair_items.extend_from_slice(&head);
                w.pair_items.extend_from_slice(&[i0, i1, j0, j1]);
                i0 = i1;
            }
            j0 = j1;
        }
    }
}

/// The level's big updates cut to fill `blocks` blocks and ordered longest
/// first: an item over the level's share of the work is cut by rectangles,
/// and a rectangle over it by k ranges (multiples of TK); the items of a
/// tile cut so are marked shared. The work is what `k_big_update` executes,
/// a whole TB x TB tile per k slab of TK, whatever the rectangle's own
/// rows and columns. Blocks are dispatched in item order, so the longest
/// items start first and the short ones fill the gaps.
fn split_big(w: &mut LevelWork, p: &Plan, blocks: usize) {
    let n_items = w.big_items.len() / BIG_ITEM;
    if n_items == 0 {
        return;
    }
    let rects = &w.big_rects;
    // a rectangle's executed multiply-adds and its k extent
    let work = |r: u32| -> (f64, u32) {
        let dq = p.dims[2 * p.pairs[rects[5 * r as usize] as usize].d as usize];
        (slab_work(dq), dq)
    };
    let total: f64 = (0..(rects.len() / 5) as u32).map(|r| work(r).0).sum();
    let cap = (total / blocks as f64).max(MIN_ITEM_WORK);
    let old = std::mem::take(&mut w.big_items);
    for it in old.chunks(BIG_ITEM) {
        let (s, row0, col0, r0, r1) = (it[0], it[1], it[2], it[3], it[4]);
        let first = w.big_items.len();
        let push = |w: &mut Vec<u32>, ra: u32, rb: u32, k0: u32, k1: u32| w.extend_from_slice(&[s, row0, col0, ra, rb, k0, k1, 0]);
        let (mut start, mut acc) = (r0, 0.0);
        for r in r0..r1 {
            let (wr, dq) = work(r);
            if wr > cap {
                if start < r {
                    push(&mut w.big_items, start, r, 0, 0);
                }
                let step = dq.div_ceil((wr / cap).ceil() as u32).next_multiple_of(TK as u32);
                let mut k0 = 0;
                while k0 < dq {
                    push(&mut w.big_items, r, r + 1, k0, dq.min(k0 + step));
                    k0 += step;
                }
                (start, acc) = (r + 1, 0.0);
            } else if acc > 0.0 && acc + wr > cap {
                push(&mut w.big_items, start, r, 0, 0);
                (start, acc) = (r, wr);
            } else {
                acc += wr;
            }
        }
        if start < r1 {
            push(&mut w.big_items, start, r1, 0, 0);
        }
        if w.big_items.len() - first > BIG_ITEM {
            for item in w.big_items[first..].chunks_mut(BIG_ITEM) {
                item[7] = 1;
            }
        }
    }
    // longest first: an item's executed work over its rectangles and k range
    let item_work = |it: &[u32]| -> f64 {
        let (r0, r1, k0, k1) = (it[3], it[4], it[5], it[6]);
        (r0..r1).map(|r| if k1 > k0 { slab_work(k1 - k0) } else { work(r).0 }).sum()
    };
    let mut items: Vec<&[u32]> = w.big_items.chunks(BIG_ITEM).collect();
    items.sort_by(|a, b| item_work(b).total_cmp(&item_work(a)));
    w.big_items = items.concat();
}

/// The multiply-adds `k_big_update` executes for `k` columns of a
/// rectangle: a whole TB x TB tile per k slab of TK.
fn slab_work(k: u32) -> f64 {
    (TB * TB) as f64 * k.next_multiple_of(TK as u32) as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::grid_spd;
    use arael_faer::supernodal::{SupernodalParams, nd_block_order};

    fn analyse(k: usize) -> (Plan, SupernodalSymbolic, SymbolicSparseBlockColMat<SparseIndex>) {
        let (a, _, _) = grid_spd(k, &[3, 6], 7);
        let asym = a.symbolic().clone();
        let order = nd_block_order(&asym);
        let sn = SupernodalSymbolic::new(&asym, Some(&order), &SupernodalParams::default()).unwrap();
        let p = Plan::new(&sn, &asym);
        (p, sn, asym)
    }

    /// Every element of every descendant update that the factor needs (at
    /// or below the target's diagonal) is in exactly one small piece or big
    /// rectangle, over all of its k once (a big item may cover a k range),
    /// and no element above it more than once (a big tile at the diagonal
    /// computes its upper part); the solve's items cover every mid row and
    /// every column once.
    fn check_coverage(kn: &GpuSupernodalParams, multiprocessors: u32, both_kernels: bool) {
        check_coverage_of(12, kn, multiprocessors, both_kernels);
    }

    /// [`check_coverage`] on a `k x k` grid.
    fn check_coverage_of(k: usize, kn: &GpuSupernodalParams, multiprocessors: u32, both_kernels: bool) {
        let (p, _, _) = analyse(k);
        let lw = build_levels(&p, kn, multiprocessors, NB, 32);
        let dq = |pi: usize| p.dims[2 * p.pairs[pi].d as usize];
        // per element of a pair, the k it is summed over
        let mut hits: Vec<Vec<u32>> = p.pairs.iter().map(|pr| vec![0u32; (pr.m * pr.kw) as usize]).collect();
        // a piece's pair, by its maps' offset
        let pair_of: HashMap<u32, usize> = (0..p.pairs.len()).map(|pi| (p.pr_rmap[pi], pi)).collect();
        let (mut small_pieces, mut big_rects) = (0usize, 0usize);
        for w in &lw {
            for it in w.pair_items.chunks(PAIR_ITEM) {
                let pi = pair_of[&it[5]];
                let pr = &p.pairs[pi];
                assert_eq!(it[..7], [p.val_ptr[pr.d as usize] + pr.r0, p.dims[2 * pr.d as usize + 1], dq(pi), p.val_ptr[pr.s as usize], p.dims[2 * pr.s as usize + 1], p.pr_rmap[pi], p.pr_cmap[pi]]);
                let (i0, i1, j0, j1) = (it[7], it[8], it[9], it[10]);
                assert!(i0 < i1 && i1 <= pr.m && j0 < j1 && j1 <= pr.kw, "piece {it:?} of a pair {} x {}", pr.m, pr.kw);
                assert!((i1 - i0) as usize <= PAIR_NI && (j1 - j0) as usize <= PAIR_NJ && ((i1 - i0) * (j1 - j0)) as usize <= PAIR_NI * PAIR_EPT);
                for j in j0..j1 {
                    for i in i0.max(j)..i1 {
                        hits[pi][(j * pr.m + i) as usize] += dq(pi);
                    }
                }
                small_pieces += 1;
            }
            for it in w.big_items.chunks(BIG_ITEM) {
                assert!(it[6] == 0 || it[4] == it[3] + 1, "a k range is of one rectangle");
                for r in it[3] as usize..it[4] as usize {
                    let (pi, da, db, ca, cb) = (w.big_rects[5 * r] as usize, w.big_rects[5 * r + 1], w.big_rects[5 * r + 2], w.big_rects[5 * r + 3], w.big_rects[5 * r + 4]);
                    let m = p.pairs[pi].m;
                    let k = if it[6] == 0 { dq(pi) } else { it[6] - it[5] };
                    for j in ca..cb {
                        for i in da..db {
                            hits[pi][(j * m + i) as usize] += k;
                        }
                    }
                    big_rects += 1;
                }
            }
        }
        assert!(small_pieces > 0, "the small-update kernel in use");
        assert!(!both_kernels || big_rects > 0, "both update kernels in use: {small_pieces} small, {big_rects} big");
        for (pi, pr) in p.pairs.iter().enumerate() {
            let rmap = &p.rmap[p.pr_rmap[pi] as usize..][..pr.m as usize];
            let cmap = &p.cmap[p.pr_cmap[pi] as usize..][..pr.kw as usize];
            for j in 0..pr.kw as usize {
                for i in 0..pr.m as usize {
                    let needed = rmap[i] >= cmap[j];
                    let h = hits[pi][j * pr.m as usize + i];
                    assert!(h == dq(pi) || (!needed && h == 0), "pair {pi} element ({i}, {j}) summed over {h} k, needed {needed}");
                }
            }
        }
        // the gather items: every mid row of every pair once, its record right
        let mut want: HashMap<[u32; GATHER_ROW], u32> = HashMap::new();
        for (pi, pr) in p.pairs.iter().enumerate() {
            let (s, d) = (pr.s as usize, pr.d as usize);
            for i in 0..pr.kw {
                let tgt = p.sc_start[p.sup_begin[s] as usize] + p.cmap[(p.pr_cmap[pi] + i) as usize];
                let rec = [p.val_ptr[d] + pr.r0 + i, tgt, d as u32];
                *want.entry(rec).or_default() += 1;
            }
        }
        let mut got: HashMap<[u32; GATHER_ROW], u32> = HashMap::new();
        for w in &lw {
            let n = (w.g_rows.len() / GATHER_ROW) as u32;
            let mut next = 0;
            for it in w.g_items.chunks(2) {
                assert_eq!(it[0], next, "the items cover the level's rows in turn");
                for r in it[0]..it[0] + it[1] {
                    let rec: [u32; GATHER_ROW] = std::array::from_fn(|f| w.g_rows[(f as u32 * n + r) as usize]);
                    *got.entry(rec).or_default() += 1;
                }
                next = it[0] + it[1];
            }
            assert_eq!(next, n);
        }
        assert_eq!(got, want, "gather rows");
        // the GEMV items: every column of every supernode with rows below, once
        let mut cols: Vec<u32> = vec![0; p.ns];
        for w in &lw {
            for it in w.bg_items.chunks(3) {
                cols[it[0] as usize] += it[2] - it[1];
            }
        }
        for s in 0..p.ns {
            let (q, h) = (p.dims[2 * s], p.dims[2 * s + 1]);
            assert_eq!(cols[s], if h > q { q } else { 0 }, "supernode {s}: GEMV columns");
        }
        // every supernode in one level, as a small or a big target
        let mut seen = vec![0u8; p.ns];
        for w in &lw {
            for &s in &w.level_sn {
                seen[s as usize] += 1;
            }
            assert_eq!(w.small_targets.len() + w.big_targets.len(), w.level_sn.len());
            assert_eq!(w.trsm_grid.len(), (w.maxq_big as usize).div_ceil(NB));
        }
        assert!(seen.iter().all(|&c| c == 1));
    }

    #[test]
    fn plan_covers_every_update_once() {
        // the default threshold leaves a 12 x 12 grid without big pairs
        check_coverage(&GpuSupernodalParams::default(), 128, false);
    }

    /// The parameters that put every kernel to work on a 12 x 12 grid.
    fn every_kernel_busy() -> GpuSupernodalParams {
        GpuSupernodalParams {
            small_q: 8,
            small_flops: f64::INFINITY,
            big_pair_flops: 100.0,
            wide_pair_k: u32::MAX,
            wide_pair_flops: f64::INFINITY,
            chain_flops_f32: 0.0,
            chain_flops_f64: 0.0,
            chain_items: 0,
            solve_chain_work: 0.0,
            solve_work: 512,
            lane_unroll: None,
            blocks_per_sm: 4,
            syrk_block: None,
            trsm_rows: None,
        }
    }

    #[test]
    fn plan_covers_every_update_once_with_every_kernel_busy() {
        check_coverage(&every_kernel_busy(), 1, true);
    }

    /// On a device too large for the grid's levels the big updates are cut
    /// by rectangles and along k, each element still summed over its k once,
    /// and the items of a cut tile are marked shared.
    #[test]
    fn big_updates_are_cut_to_fill_the_device() {
        let kn = every_kernel_busy();
        check_coverage_of(40, &kn, 10_000, true);
        let (p, _, _) = analyse(40);
        let (whole, cut) = (build_levels(&p, &kn, 1, NB, 32), build_levels(&p, &kn, 10_000, NB, 32));
        let items = |lw: &[LevelWork]| lw.iter().map(|w| w.big_items.len() / BIG_ITEM).sum::<usize>();
        assert!(items(&cut) > items(&whole), "{} items against {}", items(&cut), items(&whole));
        let k_ranges = cut.iter().flat_map(|w| w.big_items.chunks(BIG_ITEM)).filter(|it| it[6] != 0).count();
        assert!(k_ranges > 0, "some rectangle cut along k");
        for w in &cut {
            let mut tiles: HashMap<(u32, u32, u32), Vec<u32>> = HashMap::new();
            for it in w.big_items.chunks(BIG_ITEM) {
                tiles.entry((it[0], it[1], it[2])).or_default().push(it[7]);
            }
            for (tile, shared) in tiles {
                let want = (shared.len() > 1) as u32;
                assert!(shared.iter().all(|&f| f == want), "tile {tile:?}: {} items, shared flags {shared:?}", shared.len());
            }
        }
    }

    /// A level's big items, on a device they fill and on one they do not,
    /// are each within the level's share of the executed work (whole TB x
    /// TB tiles per k slab) but for one slab, and come longest first; every
    /// element is still summed over its k once.
    #[test]
    fn big_items_share_the_executed_work_longest_first() {
        let kn = every_kernel_busy();
        let (p, _, _) = analyse(40);
        let dq = |pi: u32| p.dims[2 * p.pairs[pi as usize].d as usize];
        for multiprocessors in [1, 4, 10_000] {
            check_coverage_of(40, &kn, multiprocessors, true);
            let blocks = (multiprocessors * kn.blocks_per_sm) as f64;
            let mut levels = 0;
            for w in build_levels(&p, &kn, multiprocessors, NB, 32) {
                let rect_work = |r: u32| slab_work(dq(w.big_rects[5 * r as usize]));
                let total: f64 = (0..(w.big_rects.len() / 5) as u32).map(rect_work).sum();
                let cap = (total / blocks).max(MIN_ITEM_WORK);
                let works: Vec<f64> = w
                    .big_items
                    .chunks(BIG_ITEM)
                    .map(|it| (it[3]..it[4]).map(|r| if it[6] > it[5] { slab_work(it[6] - it[5]) } else { rect_work(r) }).sum())
                    .collect();
                assert!(works.windows(2).all(|x| x[0] >= x[1]), "{multiprocessors} SMs: items not longest first: {works:?}");
                let slab = (TB * TB * TK) as f64;
                assert!(works.iter().all(|&x| x <= cap + slab), "{multiprocessors} SMs: an item over the share {cap}: {works:?}");
                levels += !works.is_empty() as usize;
            }
            assert!(levels > 0, "{multiprocessors} SMs: levels with big items");
        }
    }

    /// A supernode narrower than `small_q` goes to the small kernel only
    /// while its solve of the rows below stays within `small_flops`.
    #[test]
    fn tall_narrow_supernodes_take_the_panel_steps() {
        let (p, _, _) = analyse(12);
        let work = |s: usize| {
            let (q, h) = (p.dims[2 * s] as f64, p.dims[2 * s + 1] as f64);
            q * q * h / 2.0
        };
        let narrow: Vec<usize> = (0..p.ns).filter(|&s| p.dims[2 * s] < 96).collect();
        let mut works: Vec<f64> = narrow.iter().map(|&s| work(s)).collect();
        works.sort_by(f64::total_cmp);
        let bound = works[works.len() / 2];
        assert!(works[0] <= bound && bound < works[works.len() - 1], "the grid has narrow supernodes either side of {bound}");
        let kn = GpuSupernodalParams { small_flops: bound, ..GpuSupernodalParams::default() };
        let lw = build_levels(&p, &kn, 128, NB, 32);
        for w in &lw {
            for &s in &w.small_targets {
                assert!(work(s as usize) <= bound, "supernode {s} is small at {} multiply-adds", work(s as usize));
            }
            for &s in &w.big_targets {
                let s = s as usize;
                assert!(p.dims[2 * s] >= 96 || work(s) > bound, "supernode {s} takes the panel steps");
            }
        }
        check_coverage(&kn, 128, false);
    }

    /// On a device too large for the grid's levels the small updates are
    /// cut into more pieces, each element still in exactly one.
    #[test]
    fn small_updates_are_cut_to_fill_the_device() {
        let (p, _, _) = analyse(40);
        let kn = GpuSupernodalParams::default();
        let (whole, cut) = (build_levels(&p, &kn, 1, NB, 32), build_levels(&p, &kn, 10_000, NB, 32));
        let items = |lw: &[LevelWork]| lw.iter().map(|w| w.pair_items.len() / PAIR_ITEM).sum::<usize>();
        assert!(items(&cut) > items(&whole), "{} pieces against {}", items(&cut), items(&whole));
        check_coverage_of(40, &kn, 1, false);
        check_coverage_of(40, &kn, 10_000, false);
    }

    /// With the plain threshold off, the updates of descendants at least
    /// `wide_pair_k` wide go to the GEMM-core kernel and the others to the
    /// small-update kernel.
    #[test]
    fn wide_descendants_take_the_gemm_core() {
        let (p, _, _) = analyse(24);
        let dq = |pi: u32| p.dims[2 * p.pairs[pi as usize].d as usize];
        let mut widths: Vec<u32> = (0..p.pairs.len() as u32).map(dq).collect();
        widths.sort_unstable();
        let k = widths[widths.len() / 2];
        assert!(widths[0] < k, "the grid has narrower and wider descendants than {k}");
        let kn = GpuSupernodalParams { big_pair_flops: f64::INFINITY, wide_pair_k: k, wide_pair_flops: 0.0, ..GpuSupernodalParams::default() };
        let lw = build_levels(&p, &kn, 128, NB, 32);
        let (mut big, mut small) = (0, 0);
        for w in &lw {
            for r in w.big_rects.chunks(5) {
                assert!(dq(r[0]) >= k, "a big rectangle of a descendant {} wide", dq(r[0]));
                big += 1;
            }
            for it in w.pair_items.chunks(PAIR_ITEM) {
                assert!(it[2] < k, "a small piece of a descendant {} wide", it[2]);
                small += 1;
            }
        }
        assert!(big > 0 && small > 0, "{big} big rectangles and {small} small pieces");
    }

    /// Chains are runs of consecutive levels without big work, within the
    /// chain bounds, each level in one at most; the levels left out are
    /// those outside the bounds.
    #[test]
    fn chains_take_runs_of_small_levels() {
        let (p, _, _) = analyse(24);
        let lw = build_levels(&p, &GpuSupernodalParams::default(), 128, NB, 32);
        let work = |w: &LevelWork| w.pair_work + w.small_work;
        let items = |w: &LevelWork| (w.pair_items.len() / PAIR_ITEM) as u32;
        let mut works: Vec<f64> = lw.iter().map(work).collect();
        works.sort_by(f64::total_cmp);
        let mut counts: Vec<u32> = lw.iter().map(items).collect();
        counts.sort_unstable();
        let (flops, n) = (works[works.len() / 2], counts[counts.len() / 2]);
        let small = |w: &LevelWork| {
            w.big_items.is_empty() && w.big_targets.is_empty() && w.small_targets.len() == 1 && work(w) <= flops && items(w) <= n
        };
        let cs = chains(&lw, flops, n);
        assert!(!cs.is_empty() && lw.iter().any(|w| !small(w)), "the grid has levels in chains and out");
        let mut end = 0;
        for c in &cs {
            assert!(c.start >= end && c.start < c.end, "chains {cs:?} in order, apart");
            end = c.end;
        }
        for (l, w) in lw.iter().enumerate() {
            assert_eq!(cs.iter().any(|c| c.contains(&l)), small(w), "level {l}");
        }
        assert!(chains(&lw, 0.0, n).is_empty() && chains(&lw, flops, 0).is_empty(), "chains off");
    }

    /// The solve's chains are runs of consecutive levels of one supernode
    /// within the bound, each level in one at most.
    #[test]
    fn solve_chains_take_runs_of_single_supernodes() {
        let (p, _, _) = analyse(24);
        let lw = build_levels(&p, &GpuSupernodalParams::default(), 128, NB, 32);
        let mut works: Vec<f64> = lw.iter().filter(|w| w.level_sn.len() == 1).map(|w| w.solve_work).collect();
        works.sort_by(f64::total_cmp);
        assert!(works.len() > 2, "the grid has levels of one supernode");
        let bound = works[works.len() / 2];
        let cs = solve_chains(&lw, bound);
        let mut end = 0;
        for c in &cs {
            assert!(c.start >= end && c.start < c.end, "chains {cs:?} in order, apart");
            end = c.end;
        }
        for (l, w) in lw.iter().enumerate() {
            let fits = w.level_sn.len() == 1 && w.solve_work <= bound;
            assert_eq!(cs.iter().any(|c| c.contains(&l)), fits, "level {l}");
        }
        assert!(!cs.is_empty() && solve_chains(&lw, 0.0).is_empty());
    }

    /// Unset, the trailing update takes blocks of 128 columns on an
    /// integrated device and of NB on a discrete one; set, the blocks are
    /// what it says on every device.
    #[test]
    fn the_syrk_block_is_the_devices_unless_set() {
        let arch = |integrated| crate::Arch { name: String::new(), target: String::new(), warp: 32, shared_bytes: 49152, multiprocessors: 2, has_f64: true, integrated };
        let unset = GpuSupernodalParams::default();
        assert_eq!(unset.syrk_block_on(&arch(true)), 128);
        assert_eq!(unset.syrk_block_on(&arch(false)), NB as u32);
        let set = GpuSupernodalParams { syrk_block: Some(256), ..unset };
        assert_eq!((set.syrk_block_on(&arch(true)), set.syrk_block_on(&arch(false))), (256, 256));
    }

    /// Unset, `k_trsm` takes blocks of 64 rows on an integrated device and
    /// of a warp, at least NB, on a discrete one; set, the rows are what it
    /// says on every device.
    #[test]
    fn the_trsm_rows_are_the_devices_unless_set() {
        let arch = |integrated, warp| crate::Arch { name: String::new(), target: String::new(), warp, shared_bytes: 49152, multiprocessors: 2, has_f64: true, integrated };
        let unset = GpuSupernodalParams::default();
        assert_eq!(unset.trsm_rows_on(&arch(true, 32)), 64);
        assert_eq!(unset.trsm_rows_on(&arch(false, 32)), 32);
        assert_eq!(unset.trsm_rows_on(&arch(false, 64)), 64);
        assert_eq!(unset.trsm_rows_on(&arch(false, 16)), NB as u32);
        let set = GpuSupernodalParams { trsm_rows: Some(128), ..unset };
        assert_eq!((set.trsm_rows_on(&arch(true, 32)), set.trsm_rows_on(&arch(false, 32))), (128, 128));
    }

    /// The trailing updates of a block's steps cover every column past each
    /// solved column once: in blocks of NB, 64, 128 and 256 the columns each
    /// solved column updates are all those past it.
    #[test]
    fn the_syrk_updates_cover_the_trailing_columns_once() {
        let q = 1000u32;
        for block in [NB, 64, 128, 256] {
            // updates[c][k]: how often solved column k updated column c
            let mut updates = vec![vec![0u32; q as usize]; q as usize];
            for step in 0..(q as usize).div_ceil(NB) {
                let (k0, kn, cend) = syrk_update(step, block);
                let (k1, c1) = ((k0 + kn).min(q), cend.min(q));
                for c in k1..c1 {
                    for k in k0..k1 {
                        updates[c as usize][k as usize] += 1;
                    }
                }
            }
            for c in 0..q as usize {
                // column c is updated by every column of the steps before its own
                let solved_before = c / NB * NB;
                for k in 0..q as usize {
                    assert_eq!(updates[c][k], (k < solved_before) as u32, "block {block}: column {c} by {k}");
                }
            }
        }
    }

    #[test]
    fn plan_seeds_every_tile() {
        let (p, sn, asym) = analyse(8);
        assert_eq!(p.tiles.len() / 6, asym.nblocks());
        // every seeded scalar lands inside the factor, and a tile's destination
        // is its target's panel
        for t in p.tiles.chunks(6) {
            let (dst, stride, wr, wc) = (t[0] as usize, t[1] as usize, t[4] as usize, t[5] as usize);
            let (rows, cols) = if t[2] == 0 { (wr, wc) } else { (wc, wr) };
            assert!(dst + (cols - 1) * stride + rows <= sn.factor_val_count());
        }
    }
}
