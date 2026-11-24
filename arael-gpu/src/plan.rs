//! The settings of the device factorization ([`GpuSupernodalParams`]).

// The rest is the host side of the factorization's plan: a
// SupernodalSymbolic flattened into the integer arrays and the per-level
// work lists the kernels read. The plan is built once per symbolic analysis
// and reused by every numeric factorization; nothing in it depends on the
// matrix values.

// Index loops over parallel arrays (pattern blocks beside pattern rows).
#![allow(clippy::needless_range_loop)]

use std::collections::HashMap;

use arael_faer::bsc::SymbolicSparseBlockColMat;
use arael_faer::supernodal::SupernodalSymbolic;

/// The block index type of arael-faer's sparse matrices.
pub type SparseIndex = u32;

/// A piece of a small descendant update: at most PAIR_NI of the pair's
/// rows against PAIR_NJ of its columns, PAIR_EPT elements a thread.
pub(crate) const PAIR_NI: usize = 128;
pub(crate) const PAIR_NJ: usize = 64;
pub(crate) const PAIR_EPT: usize = 4;
/// The descendant's columns staged per pass of the small-update kernel.
pub(crate) const PAIR_KC: usize = 16;
/// Rows and columns of a tile of the big descendant update and of the
/// panel's trailing update.
pub(crate) const TB: usize = 64;
/// The descendant's columns the big descendant update's kernel multiplies
/// per pass.
pub(crate) const TK: usize = 16;
/// The panel step: columns factored per round of the three panel kernels.
pub(crate) const NB: usize = 32;

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
pub(crate) fn syrk_update(step: usize, block: usize) -> (u32, u32, u32) {
    let per = block / NB;
    let b = step / per;
    if step % per == per - 1 {
        ((b * block) as u32, block as u32, u32::MAX)
    } else {
        ((step * NB) as u32, NB as u32, ((b + 1) * block) as u32)
    }
}

/// The column where a lookahead splits panel step `step`'s update (blocks
/// of `block` columns): a block's last step updates the next block first,
/// up to this column, and the rest from it on, alongside the next block's
/// steps; `None` for a step inside its block.
///
/// ```ignore
/// assert_eq!(lookahead_split(7, 128), Some(384)); // the block [128, 256)'s last step
/// assert_eq!(lookahead_split(5, 128), None);
/// ```
pub(crate) fn lookahead_split(step: usize, block: usize) -> Option<u32> {
    let (k0, kn, cend) = syrk_update(step, block);
    (cend == u32::MAX).then_some(k0 + kn + block as u32)
}
/// Rows of x staged per pass of the backward solve's product.
pub(crate) const XR_CH: usize = 1024;

// The header of the one `plan` buffer: where each array starts, a slot
// (`H_*`) each. A kernel reads the offsets it needs once per unit; they stay
// out of the kernel's compile-time identity, so a process compiles each
// kernel once, not once per sparsity structure.
/// Per permuted block: its first permuted scalar.
pub(crate) const H_SC_START: usize = 0;
/// Per pair: the descendant supernode.
pub(crate) const H_PR_D: usize = 1;
/// Per pair: its first row in the descendant's panel.
pub(crate) const H_PR_R0: usize = 2;
/// Per pair: its offset into the row map.
pub(crate) const H_PR_RMAP: usize = 3;
/// Per pair: its offset into the column map.
pub(crate) const H_PR_CMAP: usize = 4;
/// Per pair row: the target panel's row.
pub(crate) const H_RMAP: usize = 5;
/// Per pair column: the target panel's column.
pub(crate) const H_CMAP: usize = 6;
/// Per row below a diagonal block: its index in the solve's vector.
pub(crate) const H_ROW_SC: usize = 7;
/// The block elimination order, for the device-side solve: per permuted
/// block, the original block. A permuted block's width is the distance of
/// its first permuted scalar ([`H_SC_START`]) to the next's.
pub(crate) const H_ORDER: usize = 8;
/// The matrix's scalar partition: per original block, its first scalar.
pub(crate) const H_PART: usize = 9;
/// The header of every supernode, [`SN_HEAD`] words: its columns q, its
/// rows h, its first scalar column, its panel's offset in the factor, the
/// offset of its rows below in the row map (`H_ROW_SC`), and its first
/// inverted diagonal tile ([`H_INV_TILES`]; 0 for a supernode of at most
/// [`NB`] columns, which has none).
pub(crate) const H_SN_HEAD: usize = 10;
/// Words of a supernode's header ([`H_SN_HEAD`]).
pub(crate) const SN_HEAD: usize = 6;

// The words of a big panel's record in a level's list of them (the panel
// steps' `k_potf2`, `k_trsm` and `k_syrk`): what the steps need of the
// supernode, so that finding a block's panel reads no head.
/// The supernode.
pub(crate) const PANEL_S: usize = 0;
/// Its columns.
pub(crate) const PANEL_Q: usize = 1;
/// Its rows.
pub(crate) const PANEL_H: usize = 2;
/// Its panel's offset in the factor.
pub(crate) const PANEL_OFF: usize = 3;
/// Words of a big panel's record.
pub(crate) const PANEL_REC: usize = 4;
/// The inverted 32 x 32 diagonal tiles of the supernodes wider than
/// [`NB`], 2 words each: the supernode and the tile's first column.
pub(crate) const H_INV_TILES: usize = 11;
/// Values per inverted diagonal tile: the inverse column-major (for the
/// forward solve), then row-major (for the backward solve).
pub(crate) const INV_TILE: usize = 2 * 32 * 32;
/// Words of the header.
pub(crate) const HEADER: usize = 32;

/// The settings of a [`GpuSupernodal`](crate::GpuSupernodal): at what size
/// a supernode, an update or a solve changes kernel, and how much one block
/// of a kernel takes. Every field has a default. A field that is `None`
/// takes a value chosen for the device, which the `*_on` method of the same
/// name returns.
///
/// The terms used below. A supernode is a run of the factor's columns
/// stored as one dense panel of `q` columns and `h` rows. A descendant
/// update subtracts an earlier supernode's columns from a later one. A
/// level is a set of supernodes that the same launches factor. A block is
/// one group of threads of a launch, a warp 32 threads that run in
/// lockstep. The panel steps factor a supernode 32 columns at a time over
/// many blocks: each step factors a diagonal tile, solves the rows below it
/// and updates the columns after it (the trailing update). Work is counted
/// in multiply-adds. The field names use the BLAS and LAPACK names of the
/// steps: potf2 factors a diagonal tile, trsm solves the rows below it,
/// syrk is the trailing update, trsv a triangular solve and gemv a
/// matrix-vector product.
///
/// ```
/// use arael_gpu::GpuSupernodalParams;
/// // a trailing update in groups of 128 columns, the rest as the device has it
/// let params = GpuSupernodalParams { syrk_block: Some(128), ..Default::default() };
/// assert_eq!(params.small_q, 96);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct GpuSupernodalParams {
    /// Supernodes narrower than this many columns are factored by one block
    /// each, unless [`small_flops`](Self::small_flops) sends them to the
    /// panel steps; the others take the panel steps. A value above 257 acts
    /// as 257 (one block's 256 threads), and the value is lowered to what
    /// one block's shared memory holds. Default 96.
    pub small_q: u32,
    /// The most multiply-adds, counted as `q^2 h / 2`, of a supernode that
    /// one block factors: one with more takes the panel steps. Default 3e5.
    pub small_flops: f64,
    /// A descendant update of at least this many multiply-adds is a big
    /// update, cut into 64 x 64 tiles over many blocks; a smaller one is a
    /// small update, unless [`wide_pair_k`](Self::wide_pair_k) makes it
    /// big. Default 2e6.
    pub big_pair_flops: f64,
    /// A descendant at least this many columns wide makes its updates big
    /// from [`wide_pair_flops`](Self::wide_pair_flops) multiply-adds on.
    /// Default 64.
    pub wide_pair_k: u32,
    /// The least multiply-adds of a big update from a descendant of at
    /// least [`wide_pair_k`](Self::wide_pair_k) columns. Default 1e5.
    pub wide_pair_flops: f64,
    /// Multiply-adds per block of the forward solve's gather, which
    /// subtracts the solved supernodes' parts from a level's right-hand
    /// side: a block takes rows while their work stays within this, at
    /// least one row and at most 256. Default 65536.
    pub gather_work: usize,
    /// Multiply-adds per block of the backward solve's product of the rows
    /// below a supernode's diagonal block with the solution. `None` (the
    /// default) takes the device's value ([`Self::gemv_work_on`]).
    pub gemv_work: Option<usize>,
    /// Loads a thread of the backward solve's dot products issues per loop
    /// pass: even, at least 2. `None` (the default) takes 8 in an f64
    /// factor and 4 in an f32 one.
    pub lane_unroll: Option<usize>,
    /// In an f32 factor, the most multiply-adds (small updates and factor)
    /// of a level that joins a chain. A chain is a run of consecutive
    /// levels that one block factors in one launch: each level one
    /// supernode, factored by one block, without big updates. 0 turns
    /// chains off. Default 3e5.
    pub chain_flops_f32: f64,
    /// [`chain_flops_f32`](Self::chain_flops_f32) for an f64 factor.
    /// Default 0: no chains.
    pub chain_flops_f64: f64,
    /// The most small update pieces of a level that joins a chain. Default
    /// 32.
    pub chain_items: u32,
    /// The most multiply-adds of a level's solve for the level to join a
    /// solve chain: a run of consecutive levels of one supernode each,
    /// solved by one block in one launch forward and one backward. 0 turns
    /// solve chains off. Default 1e5.
    pub solve_chain_work: f64,
    /// Blocks per multiprocessor ([`Arch::multiprocessors`]; `sm` stands for
    /// NVIDIA's streaming multiprocessor) a level's updates are cut to make:
    /// big updates that make fewer blocks are split across blocks, and the
    /// small updates are cut into pieces of the level's share of the work.
    /// Default 4.
    ///
    /// [`Arch::multiprocessors`]: crate::Arch::multiprocessors
    pub blocks_per_sm: u32,
    /// The column group of the panel steps' trailing update, in columns: a
    /// multiple of 32. At 32 every step updates all the columns after it.
    /// At more, a step updates the rest of its group only, and the group's
    /// last step updates everything after the group. `None` (the default)
    /// takes the device's value ([`Self::syrk_block_on`]).
    pub syrk_block: Option<u32>,
    /// Rows per block of the panel steps' solve of the rows below a
    /// diagonal tile, a thread per row: at least 32 and a divisor of 1024.
    /// `None` (the default) takes the device's value
    /// ([`Self::trsm_rows_on`]).
    pub trsm_rows: Option<u32>,
    /// Rows and columns of a tile of the panel steps' trailing update, a
    /// block per tile: 32 or 64. `None` (the default) takes the device's
    /// value ([`Self::syrk_tile_on`]).
    pub syrk_tile: Option<u32>,
    /// Whether a warp that factors a 32-column diagonal tile alone runs its
    /// 32 pivot steps unrolled (`true`) or in a loop (`false`). Without
    /// effect on a tile that 4 warps factor
    /// ([`diag_warps`](Self::diag_warps)). `None` (the default) takes the
    /// device's value ([`Self::potf2_unroll_on`]).
    pub potf2_unroll: Option<bool>,
    /// Warps that factor a panel step's 32-column diagonal tile: 1 or 4. A
    /// tile narrower than 32 columns takes one. `None` (the default) takes
    /// the device's value ([`Self::diag_warps_on`]).
    pub diag_warps: Option<u32>,
    /// The narrowest supernode, in columns (more than 32), whose triangular
    /// solves run over many blocks, [`trsv_block`](Self::trsv_block)
    /// columns per launch; one block solves a narrower one. A supernode too
    /// wide for one block's shared memory always takes many blocks. `None`
    /// (the default) takes the device's value ([`Self::trsv_wide_on`]).
    pub trsv_wide: Option<u32>,
    /// Columns per launch of the triangular solves over many blocks: a
    /// multiple of 32. `None` (the default) takes the device's value
    /// ([`Self::trsv_block_on`]).
    pub trsv_block: Option<u32>,
    /// Whether a big descendant update is cut into tiles of its own rows
    /// and columns, each added into the target atomically (`true`), or into
    /// the target's 64 x 64 tiles, each summing the parts of several
    /// updates before it is written (`false`). `None` (the default) takes
    /// the device's value ([`Self::pair_tiles_on`]).
    pub pair_tiles: Option<bool>,
    /// Whether the panel steps look ahead: the last step of a column group
    /// ([`syrk_block`](Self::syrk_block)) updates the next group's columns
    /// first, and the rest of its update runs on the main stream while the
    /// next group's steps run on the side stream ([`crate::Lane`]). `None`
    /// (the default) takes the device's value ([`Self::lookahead_on`]).
    pub lookahead: Option<bool>,
}

impl GpuSupernodalParams {
    /// The [`syrk_block`](Self::syrk_block) used on the device `arch` for a
    /// factor of `fs`-byte scalars (4 or 8): the value set, else 256 on an
    /// Apple GPU; on another integrated GPU and the CPU device 256 for an
    /// f32 factor and 128 for an f64 one; 32 on a discrete GPU.
    ///
    /// ```
    /// use arael_gpu::{Arch, GpuSupernodalParams, Vendor};
    /// let arch = Arch {
    ///     name: "a discrete GPU".into(),
    ///     vendor: Vendor::Nvidia,
    ///     target: "sm_89".into(),
    ///     warp: 32,
    ///     shared_bytes: 48 << 10,
    ///     multiprocessors: 128,
    ///     has_f64: true,
    ///     flushes_subnormals: false,
    ///     integrated: false,
    /// };
    /// let params = GpuSupernodalParams::default();
    /// assert_eq!(params.syrk_block_on(&arch, 4), 32);
    /// assert_eq!(GpuSupernodalParams { syrk_block: Some(128), ..params }.syrk_block_on(&arch, 4), 128);
    /// ```
    pub fn syrk_block_on(&self, arch: &crate::Arch, fs: usize) -> u32 {
        self.syrk_block.unwrap_or(match arch.vendor {
            crate::Vendor::Apple => 256,
            _ if arch.integrated && fs == 4 => 256,
            _ if arch.integrated => 128,
            _ => NB as u32,
        })
    }

    /// The [`trsm_rows`](Self::trsm_rows) used on the device `arch`: the
    /// value set, else 64 on an integrated GPU and the CPU device, and the
    /// device's lockstep width ([`Arch::warp`], at least 32) on a discrete
    /// GPU.
    ///
    /// [`Arch::warp`]: crate::Arch::warp
    pub fn trsm_rows_on(&self, arch: &crate::Arch) -> u32 {
        self.trsm_rows.unwrap_or(if arch.integrated { 64 } else { arch.warp.max(NB as u32) })
    }

    /// The [`gemv_work`](Self::gemv_work) used on the device `arch`: the
    /// value set, else 16384 on an integrated GPU and the CPU device, 2048
    /// on a discrete GPU.
    pub fn gemv_work_on(&self, arch: &crate::Arch) -> usize {
        self.gemv_work.unwrap_or(if arch.integrated { 16384 } else { 2048 })
    }

    /// The [`syrk_tile`](Self::syrk_tile) used on the device `arch` for a
    /// factor of `fs`-byte scalars (4 or 8): the value set, else 64 for an
    /// f32 factor on an integrated GPU and the CPU device, 32 otherwise.
    pub fn syrk_tile_on(&self, arch: &crate::Arch, fs: usize) -> u32 {
        self.syrk_tile.unwrap_or(if fs == 4 && arch.integrated { 64 } else { 32 })
    }

    /// The [`potf2_unroll`](Self::potf2_unroll) used on the device `arch`:
    /// the value set, else `true` on a discrete GPU, `false` on an
    /// integrated GPU and the CPU device.
    pub fn potf2_unroll_on(&self, arch: &crate::Arch) -> bool {
        self.potf2_unroll.unwrap_or(!arch.integrated)
    }

    /// The [`diag_warps`](Self::diag_warps) used on the device `arch`: the
    /// value set, else 4 on NVIDIA and AMD GPUs, 1 on Apple GPUs and the
    /// CPU device.
    pub fn diag_warps_on(&self, arch: &crate::Arch) -> u32 {
        self.diag_warps.unwrap_or(match arch.vendor {
            crate::Vendor::Nvidia | crate::Vendor::Amd => 4,
            crate::Vendor::Apple | crate::Vendor::Cpu => 1,
        })
    }

    /// The [`trsv_wide`](Self::trsv_wide) used on the device `arch`: the
    /// value set, else 512 on a discrete GPU, 1024 on an integrated GPU and
    /// the CPU device.
    pub fn trsv_wide_on(&self, arch: &crate::Arch) -> u32 {
        self.trsv_wide.unwrap_or(if arch.integrated { 1024 } else { 512 })
    }

    /// The [`trsv_block`](Self::trsv_block) used on the device `arch`: the
    /// value set, else 128 on a discrete GPU, 256 on an integrated GPU and
    /// the CPU device.
    pub fn trsv_block_on(&self, arch: &crate::Arch) -> u32 {
        self.trsv_block.unwrap_or(if arch.integrated { 256 } else { 128 })
    }

    /// The [`pair_tiles`](Self::pair_tiles) used on the device `arch` for a
    /// factor of `fs`-byte scalars (4 or 8): the value set, else `true` on
    /// an integrated GPU and the CPU device, and on a discrete GPU `true`
    /// for an f64 factor, `false` for an f32 one.
    pub fn pair_tiles_on(&self, arch: &crate::Arch, fs: usize) -> bool {
        self.pair_tiles.unwrap_or(arch.integrated || fs == 8)
    }

    /// The [`lookahead`](Self::lookahead) used on the device `arch`: the
    /// value set, else `false` on every device.
    pub fn lookahead_on(&self, arch: &crate::Arch) -> bool {
        let _ = arch;
        self.lookahead.unwrap_or(false)
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
            gather_work: 65536,
            gemv_work: None,
            lane_unroll: None,
            blocks_per_sm: 4,
            syrk_block: None,
            trsm_rows: None,
            syrk_tile: None,
            potf2_unroll: None,
            diag_warps: None,
            trsv_wide: None,
            trsv_block: None,
            pair_tiles: None,
            lookahead: None,
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
        let inv = self.inv_tiles();
        let mut tile = 0;
        let head: Vec<u32> = (0..self.ns)
            .flat_map(|s| {
                let col0 = self.sc_start[self.sup_begin[s] as usize];
                let q = self.dims[2 * s];
                let first = if q as usize > NB { tile } else { 0 };
                if q as usize > NB {
                    tile += q.div_ceil(32);
                }
                [q, self.dims[2 * s + 1], col0, self.val_ptr[s], self.row_sc_off[s], first]
            })
            .collect();
        // What a supernode has (its dims, its panel, its rows below) is in
        // its head and nowhere else; a pair carries its descendant and
        // maps; a permuted block's width is the distance between two of
        // sc_start. So the kernels read no dims, val_ptr, sup_begin,
        // pair_ptr, row_sc_off or blk_width.
        let arrays: [(usize, &[u32]); 12] = [
            (H_ORDER, &self.order),
            (H_PART, &self.part),
            (H_SC_START, &self.sc_start),
            (H_PR_D, &pr_d),
            (H_PR_R0, &pr_r0),
            (H_PR_RMAP, &self.pr_rmap),
            (H_PR_CMAP, &self.pr_cmap),
            (H_RMAP, &self.rmap),
            (H_CMAP, &self.cmap),
            (H_ROW_SC, &self.row_sc),
            (H_SN_HEAD, &head),
            (H_INV_TILES, &inv),
        ];
        for (slot, v) in arrays {
            buf[slot] = buf.len() as u32;
            buf.extend_from_slice(v);
        }
        buf
    }

    /// The inverted diagonal tiles ([`H_INV_TILES`]): a tile per 32
    /// columns of every supernode wider than [`NB`], as (supernode, first
    /// column), supernode by supernode.
    pub fn inv_tiles(&self) -> Vec<u32> {
        (0..self.ns)
            .filter(|&s| self.dims[2 * s] as usize > NB)
            .flat_map(|s| (0..self.dims[2 * s]).step_by(32).flat_map(move |j0| [s as u32, j0]))
            .collect()
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
    /// [`PAIR_TILE`] per item of the big updates tiled in the pairs' own
    /// rows and columns (`pair_tiles`): target, pair, the tile's first row
    /// i0 and column j0 of the pair's m x kw block, its rows and columns,
    /// and the k range `[k0, k1)` (`k1` 0: every k).
    pub pair_tiles: Vec<u32>,
    pub small_targets: Vec<u32>,
    pub big_targets: Vec<u32>,
    pub level_sn: Vec<u32>,
    /// [`GATHER_ROW`] fields per mid row of the level's pairs, field by
    /// field (field f of row r at `f * rows + r`, so a warp's loads of a field
    /// are contiguous): the descendant's row at its column 0 (a factor
    /// offset), the target's x index, and the descendant (its height, width
    /// and first column in its solve header, `H_SN_HEAD`).
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
    /// Per panel step, with a lookahead ([`GpuSupernodalParams::lookahead`]),
    /// the syrk's grid split at [`lookahead_split`]: the next block's
    /// columns and the rest; both 0 at a step inside its block or without a
    /// lookahead.
    pub syrk_near_grid: Vec<u32>,
    pub syrk_far_grid: Vec<u32>,
    /// The level's supernodes whose triangular solves run on one block each
    /// (`k_fwd_trsv`, `k_bwd_trsv`).
    pub narrow: Vec<u32>,
    /// The other supernodes' forward solve over many blocks
    /// (`k_fwd_trsv_wide`): per launch its items, [`TRSV_ITEM`] words each,
    /// the supernode, the launch's block j and rows `[a, b)`; `a == b` is
    /// block j's own solve.
    pub wide_fwd: Vec<Vec<u32>>,
    /// Their backward solve (`k_bwd_trsv_wide`): per launch its items, as
    /// `wide_fwd`'s with the block k it solves and columns `[a, b)`.
    pub wide_bwd: Vec<Vec<u32>>,
}

/// Words per big update item ([`LevelWork::big_items`]).
pub(crate) const BIG_ITEM: usize = 8;

/// Words per pair tile item ([`LevelWork::pair_tiles`]).
pub(crate) const PAIR_TILE: usize = 8;

/// Words per item of the triangular solves over many blocks
/// ([`LevelWork::wide_fwd`]).
pub(crate) const TRSV_ITEM: usize = 4;

/// The most rows of a forward update item of the solves over many blocks:
/// one a thread of the block.
pub(crate) const TRSV_UPDATE_ROWS: u32 = 256;

/// The most columns of a backward update item of the solves over many
/// blocks, a warp per column.
pub(crate) const TRSV_UPDATE_COLS: u32 = 32;

/// Words per gather row ([`LevelWork::g_rows`]).
pub(crate) const GATHER_ROW: usize = 3;

/// The most rows of a gather item: one a thread of the gather's block.
pub(crate) const GATHER_ITEM_ROWS: u32 = 256;

/// Words per small update piece ([`LevelWork::pair_items`]).
pub(crate) const PAIR_ITEM: usize = 11;

/// The least work (multiply-adds) a small update piece is cut down to.
const MIN_PAIR_WORK: f64 = 2048.0;

/// The least work (multiply-adds) a big item is cut down to.
const MIN_ITEM_WORK: f64 = (TB * TB * TK * 2) as f64;

/// The level schedule of `p` on the device `arch` for a factor of `fs`-byte
/// scalars: every supernode's work sorted into its level and cut into
/// blocks, with the sizes of `kn` that the device chooses (the panel steps'
/// `syrk_block`, `syrk_tile` and `trsm_rows`, the solve's `gemv_work`).
pub(crate) fn build_levels(p: &Plan, kn: &GpuSupernodalParams, arch: &crate::Arch, fs: usize) -> Vec<LevelWork> {
    let syrk_block = kn.syrk_block_on(arch, fs) as usize;
    let syrk_tile = kn.syrk_tile_on(arch, fs);
    let trsm_rows = kn.trsm_rows_on(arch);
    let gemv_work = kn.gemv_work_on(arch);
    let pair_tiles = kn.pair_tiles_on(arch, fs);
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
                if ng > 0 && *g_work + dq as usize <= kn.gather_work && w.g_items[2 * ng - 1] < GATHER_ITEM_ROWS {
                    w.g_items[2 * ng - 1] += 1;
                    *g_work += dq as usize;
                } else {
                    w.g_items.extend_from_slice(&[row, 1]);
                    *g_work = dq as usize;
                }
            }
        }
        if h > q {
            let per = ((gemv_work / (h - q) as usize) as u32).max(1);
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
        if pair_tiles {
            // the big pairs' TB x TB tiles of their own m x kw blocks, those
            // reaching the lower part (pair row i >= column j: the first kw
            // rows are the target's columns, and the maps increase)
            let tb = TB as u32;
            for &pi in &bigs {
                let pr = &p.pairs[pi];
                for i0 in (0..pr.m).step_by(TB) {
                    for j0 in (0..pr.kw.min(i0 + tb)).step_by(TB) {
                        w.pair_tiles.extend_from_slice(&[s as u32, pi as u32, i0, j0, (pr.m - i0).min(tb), (pr.kw - j0).min(tb), 0, 0]);
                    }
                }
            }
            continue;
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
    let blocks = (arch.multiprocessors * kn.blocks_per_sm) as usize;
    // a supernode wider than the one-block solve's shared memory holds
    // takes the many blocks
    let wide = kn.trsv_wide_on(arch).min((arch.shared_bytes as usize / fs) as u32 + 1);
    let trsv_block = kn.trsv_block_on(arch);
    assert!(trsv_block > 0 && trsv_block.is_multiple_of(32), "trsv_block {trsv_block} is a multiple of 32");
    for (w, pairs) in lw.iter_mut().zip(&small_pairs) {
        split_big(w, p, blocks);
        split_pair_tiles(w, p, blocks);
        pair_pieces(w, p, pairs, blocks);
        wide_trsv(w, p, wide, trsv_block);
    }
    // the panel steps' grid sizes
    let lookahead = kn.lookahead_on(arch);
    for w in &mut lw {
        let mut j = 0u32;
        while j < w.maxq_big {
            let step = j as usize / NB;
            let (k0, kn, cend) = syrk_update(step, syrk_block);
            // k_syrk's tiles of the update of columns [k1, cend) from cbeg on
            let syrk_tiles = |cend: u32, cbeg: u32| -> u32 {
                let mut tiles = 0;
                for &s in &w.big_targets {
                    let (q, h) = (p.dims[2 * s as usize], p.dims[2 * s as usize + 1]);
                    if k0 < q {
                        let (k1, c1) = ((k0 + kn).min(q), cend.min(q));
                        let cs = k1.max(cbeg);
                        if cs < c1 {
                            tiles += (h - cs).div_ceil(syrk_tile) * (c1 - cs).div_ceil(syrk_tile);
                        }
                    }
                }
                tiles
            };
            let (near, far) = match lookahead_split(step, syrk_block) {
                Some(split) if lookahead => (syrk_tiles(split, 0), syrk_tiles(u32::MAX, split)),
                _ => (0, 0),
            };
            let mut tg = 0u32;
            for &s in &w.big_targets {
                let (q, h) = (p.dims[2 * s as usize], p.dims[2 * s as usize + 1]);
                if j < q {
                    let nb = (NB as u32).min(q - j);
                    tg += (h - j - nb).div_ceil(trsm_rows);
                }
            }
            w.syrk_grid.push(syrk_tiles(cend, 0));
            w.syrk_near_grid.push(near);
            w.syrk_far_grid.push(far);
            w.trsm_grid.push(tg);
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
            && w.pair_tiles.is_empty()
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

/// The level's triangular solves: a supernode of at least `wide` columns
/// (and more than [`NB`]) over many blocks, a launch per `block` columns
/// ([`LevelWork::wide_fwd`], [`LevelWork::wide_bwd`]), the others one block
/// each ([`LevelWork::narrow`]). Forward launch j: block j's solve, and the
/// rows past block j in items of [`TRSV_UPDATE_ROWS`] taking block j - 1
/// off. Backward launch j solves block k = blocks - 1 - j, and the columns
/// before block k in items of [`TRSV_UPDATE_COLS`] take block k + 1 off. A
/// launch's solve items come first.
fn wide_trsv(w: &mut LevelWork, p: &Plan, wide: u32, block: u32) {
    let wide = wide.max(NB as u32 + 1);
    let mut wides = Vec::new();
    for &s in &w.level_sn {
        let q = p.dims[2 * s as usize];
        if q >= wide {
            wides.push((s, q));
        } else {
            w.narrow.push(s);
        }
    }
    let launches = wides.iter().map(|&(_, q)| q.div_ceil(block)).max().unwrap_or(0) as usize;
    w.wide_fwd = vec![Vec::new(); launches];
    w.wide_bwd = vec![Vec::new(); launches];
    for &(s, q) in &wides {
        let nd = q.div_ceil(block);
        for j in 0..nd {
            w.wide_fwd[j as usize].extend_from_slice(&[s, j, 0, 0]);
            let k = nd - 1 - j;
            w.wide_bwd[j as usize].extend_from_slice(&[s, k, 0, 0]);
        }
    }
    for &(s, q) in &wides {
        let nd = q.div_ceil(block);
        for j in 1..nd {
            let f = &mut w.wide_fwd[j as usize];
            for r in ((j + 1) * block..q).step_by(TRSV_UPDATE_ROWS as usize) {
                f.extend_from_slice(&[s, j, r, (r + TRSV_UPDATE_ROWS).min(q)]);
            }
            let k = nd - 1 - j;
            let b = &mut w.wide_bwd[j as usize];
            for c in (0..k * block).step_by(TRSV_UPDATE_COLS as usize) {
                b.extend_from_slice(&[s, k, c, (c + TRSV_UPDATE_COLS).min(k * block)]);
            }
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

/// The multiply-adds `k_big_pair` executes for a pair tile over `k`
/// columns: the whole TB x TB tile, 10 of its 16 sub-blocks on a diagonal
/// tile (`diag`, the others lie above the diagonal).
fn pair_tile_work(k: u32, diag: bool) -> f64 {
    (TB * TB) as f64 * k as f64 * if diag { 10.0 / 16.0 } else { 1.0 }
}

/// The level's pair tiles cut to fill `blocks` blocks and ordered longest
/// first: a tile over the level's share of the work is cut into k ranges
/// (multiples of TK). Blocks are dispatched in item order, so the longest
/// items start first and the short ones fill the gaps.
fn split_pair_tiles(w: &mut LevelWork, p: &Plan, blocks: usize) {
    if w.pair_tiles.is_empty() {
        return;
    }
    let dq = |pi: u32| p.dims[2 * p.pairs[pi as usize].d as usize];
    let item_work = |it: &[u32]| pair_tile_work(if it[7] > 0 { it[7] - it[6] } else { dq(it[1]) }, it[2] == it[3]);
    let total: f64 = w.pair_tiles.chunks(PAIR_TILE).map(item_work).sum();
    let cap = (total / blocks as f64).max(MIN_ITEM_WORK);
    let old = std::mem::take(&mut w.pair_tiles);
    for it in old.chunks(PAIR_TILE) {
        let work = item_work(it);
        if work <= cap {
            w.pair_tiles.extend_from_slice(it);
            continue;
        }
        let k = dq(it[1]);
        let step = k.div_ceil((work / cap).ceil() as u32).next_multiple_of(TK as u32);
        for k0 in (0..k).step_by(step as usize) {
            w.pair_tiles.extend_from_slice(&it[..6]);
            w.pair_tiles.extend_from_slice(&[k0, k.min(k0 + step)]);
        }
    }
    let mut items: Vec<&[u32]> = w.pair_tiles.chunks(PAIR_TILE).collect();
    items.sort_by(|a, b| item_work(b).total_cmp(&item_work(a)));
    w.pair_tiles = items.concat();
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

    /// A device of `multiprocessors` with `warp` lanes, integrated (an AMD
    /// one) or discrete (an NVIDIA one).
    fn arch(multiprocessors: u32, warp: u32, integrated: bool) -> crate::Arch {
        let vendor = if integrated { crate::Vendor::Amd } else { crate::Vendor::Nvidia };
        crate::Arch { name: String::new(), vendor, target: String::new(), warp, shared_bytes: 49152, multiprocessors, has_f64: true, flushes_subnormals: false, integrated }
    }

    /// A discrete device of `multiprocessors` with 32-lane warps.
    fn discrete(multiprocessors: u32) -> crate::Arch {
        arch(multiprocessors, 32, false)
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
        let lw = build_levels(&p, kn, &discrete(multiprocessors), 8);
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
            // a pair tile writes its elements at or below the pair's diagonal
            for it in w.pair_tiles.chunks(PAIR_TILE) {
                let (pi, i0, j0, nr, nc) = (it[1] as usize, it[2], it[3], it[4], it[5]);
                let pr = &p.pairs[pi];
                assert!(it[0] == pr.s && i0 + nr <= pr.m && j0 + nc <= pr.kw && nr <= TB as u32 && nc <= TB as u32, "pair tile {it:?}");
                let k = if it[7] == 0 { dq(pi) } else { it[7] - it[6] };
                for j in j0..j0 + nc {
                    for i in i0.max(j)..i0 + nr {
                        hits[pi][(j * pr.m + i) as usize] += k;
                    }
                }
                big_rects += 1;
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
        // the GEMV items: every column of every supernode with rows below,
        // once; a supernode's items are consecutive and follow its columns
        // from 0 (a solve chain runs them as one range)
        let mut cols: Vec<u32> = vec![0; p.ns];
        for w in &lw {
            let (mut prev_s, mut prev_c1) = (u32::MAX, 0);
            for it in w.bg_items.chunks(3) {
                cols[it[0] as usize] += it[2] - it[1];
                let c0 = if it[0] == prev_s { prev_c1 } else { 0 };
                assert_eq!(it[1], c0, "supernode {}: a GEMV item out of column order", it[0]);
                (prev_s, prev_c1) = (it[0], it[2]);
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
            gather_work: 512,
            gemv_work: Some(512),
            lane_unroll: None,
            blocks_per_sm: 4,
            syrk_block: None,
            trsm_rows: None,
            syrk_tile: None,
            potf2_unroll: None,
            diag_warps: None,
            trsv_wide: None,
            trsv_block: None,
            pair_tiles: Some(false),
            lookahead: None,
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
        let (whole, cut) = (build_levels(&p, &kn, &discrete(1), 8), build_levels(&p, &kn, &discrete(10_000), 8));
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
            for w in build_levels(&p, &kn, &discrete(multiprocessors), 8) {
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

    /// Unset, the big updates take pair tiles on an integrated device and
    /// for an f64 factor on a discrete one; set, what it says on every
    /// device.
    #[test]
    fn the_pair_tiles_are_the_devices_unless_set() {
        let unset = GpuSupernodalParams::default();
        let (int, dis) = (arch(2, 32, true), discrete(128));
        assert_eq!((unset.pair_tiles_on(&int, 8), unset.pair_tiles_on(&int, 4)), (true, true));
        assert_eq!((unset.pair_tiles_on(&dis, 8), unset.pair_tiles_on(&dis, 4)), (true, false));
        let set = GpuSupernodalParams { pair_tiles: Some(false), ..unset };
        assert_eq!((set.pair_tiles_on(&int, 4), set.pair_tiles_on(&dis, 8)), (false, false));
    }

    /// The big updates tiled in the pairs' own rows and columns: every
    /// element at or below a pair's diagonal is summed over its k once, on
    /// a device the tiles fill and on one so large that they are cut along
    /// k; no target tiles are built then.
    #[test]
    fn pair_tiles_cover_every_update_once() {
        let kn = GpuSupernodalParams { pair_tiles: Some(true), ..every_kernel_busy() };
        check_coverage(&kn, 128, true);
        check_coverage_of(40, &kn, 1, true);
        check_coverage_of(40, &kn, 10_000, true);
        let (p, _, _) = analyse(40);
        let cut = build_levels(&p, &kn, &discrete(10_000), 8);
        assert!(cut.iter().all(|w| w.big_items.is_empty() && w.big_rects.is_empty()));
        assert!(cut.iter().flat_map(|w| w.pair_tiles.chunks(PAIR_TILE)).any(|it| it[7] != 0), "a pair tile cut along k");
    }

    /// A level's pair tiles, on a device they fill and on one they do not,
    /// are each within the level's share of the executed work but for one
    /// slab, and come longest first.
    #[test]
    fn pair_tiles_share_the_executed_work_longest_first() {
        let kn = GpuSupernodalParams { pair_tiles: Some(true), ..every_kernel_busy() };
        let (p, _, _) = analyse(40);
        let dq = |pi: u32| p.dims[2 * p.pairs[pi as usize].d as usize];
        let work = |it: &[u32]| pair_tile_work(if it[7] > 0 { it[7] - it[6] } else { dq(it[1]) }, it[2] == it[3]);
        for multiprocessors in [1, 4, 10_000] {
            let blocks = (multiprocessors * kn.blocks_per_sm) as f64;
            let mut levels = 0;
            for w in build_levels(&p, &kn, &discrete(multiprocessors), 8) {
                let works: Vec<f64> = w.pair_tiles.chunks(PAIR_TILE).map(work).collect();
                let cap = (works.iter().sum::<f64>() / blocks).max(MIN_ITEM_WORK);
                assert!(works.windows(2).all(|x| x[0] >= x[1]), "{multiprocessors} SMs: items not longest first: {works:?}");
                let slab = (TB * TB * TK) as f64;
                assert!(works.iter().all(|&x| x <= cap + slab), "{multiprocessors} SMs: an item over the share {cap}: {works:?}");
                levels += !works.is_empty() as usize;
            }
            assert!(levels > 0, "{multiprocessors} SMs: levels with pair tiles");
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
        let lw = build_levels(&p, &kn, &discrete(128), 8);
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
        let (whole, cut) = (build_levels(&p, &kn, &discrete(1), 8), build_levels(&p, &kn, &discrete(10_000), 8));
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
        let lw = build_levels(&p, &kn, &discrete(128), 8);
        let (mut big, mut small) = (0, 0);
        for w in &lw {
            for r in w.big_rects.chunks(5) {
                assert!(dq(r[0]) >= k, "a big rectangle of a descendant {} wide", dq(r[0]));
                big += 1;
            }
            for it in w.pair_tiles.chunks(PAIR_TILE) {
                assert!(dq(it[1]) >= k, "a pair tile of a descendant {} wide", dq(it[1]));
                big += 1;
            }
            for it in w.pair_items.chunks(PAIR_ITEM) {
                assert!(it[2] < k, "a small piece of a descendant {} wide", it[2]);
                small += 1;
            }
        }
        assert!(big > 0 && small > 0, "{big} big rectangles or pair tiles and {small} small pieces");
    }

    /// Chains are runs of consecutive levels without big work, within the
    /// chain bounds, each level in one at most; the levels left out are
    /// those outside the bounds.
    #[test]
    fn chains_take_runs_of_small_levels() {
        let (p, _, _) = analyse(24);
        let lw = build_levels(&p, &GpuSupernodalParams::default(), &discrete(128), 8);
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
        let lw = build_levels(&p, &GpuSupernodalParams::default(), &discrete(128), 8);
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

    /// A lookahead splits a block's last update at the end of the next
    /// block, and no step inside a block; unset, it is off.
    #[test]
    fn the_lookahead_splits_a_blocks_last_update() {
        assert_eq!(lookahead_split(7, 128), Some(384));
        assert_eq!(lookahead_split(5, 128), None);
        // in blocks of NB every step is its block's last
        assert_eq!(lookahead_split(3, NB), Some(160));
        let arch = |integrated| arch(2, 32, integrated);
        assert!(!GpuSupernodalParams::default().lookahead_on(&arch(true)));
        assert!(!GpuSupernodalParams::default().lookahead_on(&arch(false)));
        assert!(GpuSupernodalParams { lookahead: Some(true), ..Default::default() }.lookahead_on(&arch(false)));
    }

    /// Unset, the trailing update takes blocks of 256 columns on an Apple
    /// GPU, on another integrated device 256 for an f32 factor and 128 for
    /// an f64 one, and NB on a discrete one; set, the blocks are what it
    /// says on every device.
    #[test]
    fn the_syrk_block_is_the_devices_unless_set() {
        let arch = |integrated| arch(2, 32, integrated);
        let apple = crate::Arch { vendor: crate::Vendor::Apple, ..arch(true) };
        let unset = GpuSupernodalParams::default();
        assert_eq!([4, 8].map(|fs| unset.syrk_block_on(&apple, fs)), [256, 256]);
        assert_eq!([4, 8].map(|fs| unset.syrk_block_on(&arch(true), fs)), [256, 128]);
        assert_eq!([4, 8].map(|fs| unset.syrk_block_on(&arch(false), fs)), [NB as u32; 2]);
        let set = GpuSupernodalParams { syrk_block: Some(64), ..unset };
        assert_eq!([&apple, &arch(true), &arch(false)].map(|a| set.syrk_block_on(a, 4)), [64; 3]);
    }

    /// Unset, `k_trsm` takes blocks of 64 rows on an integrated device and
    /// of a warp, at least NB, on a discrete one; set, the rows are what it
    /// says on every device.
    #[test]
    fn the_trsm_rows_are_the_devices_unless_set() {
        let arch = |integrated, warp| arch(2, warp, integrated);
        let unset = GpuSupernodalParams::default();
        assert_eq!(unset.trsm_rows_on(&arch(true, 32)), 64);
        assert_eq!(unset.trsm_rows_on(&arch(false, 32)), 32);
        assert_eq!(unset.trsm_rows_on(&arch(false, 64)), 64);
        assert_eq!(unset.trsm_rows_on(&arch(false, 16)), NB as u32);
        let set = GpuSupernodalParams { trsm_rows: Some(128), ..unset };
        assert_eq!((set.trsm_rows_on(&arch(true, 32)), set.trsm_rows_on(&arch(false, 32))), (128, 128));
    }

    /// Every supernode wider than NB has a tile per 32 columns in the
    /// inverted tile list, in column order from its solve header's first
    /// tile; a narrower one has none (first tile 0).
    #[test]
    fn the_inverse_tiles_follow_the_solve_heads() {
        let (p, _, _) = analyse(40);
        let words = p.pack();
        let head = &words[words[H_SN_HEAD] as usize..];
        let inv = &words[words[H_INV_TILES] as usize..];
        let (mut next, mut wide) = (0u32, 0);
        for s in 0..p.ns {
            let q = p.dims[2 * s];
            let first = head[SN_HEAD * s + 5];
            if q as usize > NB {
                assert_eq!(first, next, "supernode {s}: its first tile");
                for (i, j0) in (0..q).step_by(32).enumerate() {
                    let t = first as usize + i;
                    assert_eq!(inv[2 * t..2 * t + 2], [s as u32, j0], "supernode {s}: tile {i}");
                }
                next += q.div_ceil(32);
                wide += 1;
            } else {
                assert_eq!(first, 0, "supernode {s} of {q} columns");
            }
        }
        assert!(wide > 1, "supernodes wider than NB: {wide}");
        assert_eq!(p.inv_tiles().len(), 2 * next as usize);
    }

    /// Unset, the triangular solves run over many blocks from 512 columns
    /// on, in blocks of 128, on a discrete device, from 1024 in blocks of
    /// 256 on an integrated one; set, what they say on every device.
    #[test]
    fn the_wide_trsv_sizes_are_the_devices_unless_set() {
        let unset = GpuSupernodalParams::default();
        let (int, dis) = (arch(2, 32, true), discrete(128));
        assert_eq!((unset.trsv_wide_on(&int), unset.trsv_block_on(&int)), (1024, 256));
        assert_eq!((unset.trsv_wide_on(&dis), unset.trsv_block_on(&dis)), (512, 128));
        let set = GpuSupernodalParams { trsv_wide: Some(100), trsv_block: Some(64), ..unset };
        assert_eq!((set.trsv_wide_on(&int), set.trsv_block_on(&int)), (100, 64));
        assert_eq!((set.trsv_wide_on(&dis), set.trsv_block_on(&dis)), (100, 64));
    }

    /// The solves over many blocks: a level's supernodes are narrow or wide,
    /// once; forward launch j solves block j of each wide supernode (its
    /// first item there) and takes block j - 1 off every row past block j
    /// once; backward launch j solves block k = blocks - 1 - j and takes
    /// block k + 1 off every column before block k once.
    #[test]
    fn the_wide_trsv_items_cover_every_block_row_and_column() {
        let (p, _, _) = analyse(40);
        for block in [32u32, 64, 128] {
            let kn = GpuSupernodalParams { trsv_wide: Some(33), trsv_block: Some(block), ..GpuSupernodalParams::default() };
            let mut wides = 0;
            for w in build_levels(&p, &kn, &discrete(128), 8) {
                let wide: Vec<(u32, u32)> = w.level_sn.iter().map(|&s| (s, p.dims[2 * s as usize])).filter(|&(_, q)| q >= 33).collect();
                let mut seen: Vec<u32> = w.narrow.iter().copied().chain(wide.iter().map(|&(s, _)| s)).collect();
                let mut all = w.level_sn.clone();
                seen.sort_unstable();
                all.sort_unstable();
                assert_eq!(seen, all, "narrow and wide supernodes");
                for &(s, q) in &wide {
                    wides += 1;
                    let nd = q.div_ceil(block);
                    for j in 0..nd {
                        let items: Vec<&[u32]> = w.wide_fwd[j as usize].chunks(TRSV_ITEM).filter(|it| it[0] == s).collect();
                        assert_eq!(items[0], &[s, j, 0, 0][..], "forward launch {j}: supernode {s} solves block {j} first");
                        let mut rows = vec![0u32; q as usize];
                        for it in &items[1..] {
                            assert!(it[1] == j && it[2] < it[3] && it[3] - it[2] <= TRSV_UPDATE_ROWS, "forward item {it:?}");
                            (it[2]..it[3]).for_each(|r| rows[r as usize] += 1);
                        }
                        for r in 0..q {
                            assert_eq!(rows[r as usize], u32::from(j > 0 && r >= (j + 1) * block), "forward launch {j}, supernode {s}, row {r}");
                        }
                        let k = nd - 1 - j;
                        let items: Vec<&[u32]> = w.wide_bwd[j as usize].chunks(TRSV_ITEM).filter(|it| it[0] == s).collect();
                        assert_eq!(items[0], &[s, k, 0, 0][..], "backward launch {j}: supernode {s} solves block {k} first");
                        let mut cols = vec![0u32; q as usize];
                        for it in &items[1..] {
                            assert!(it[1] == k && it[2] < it[3] && it[3] - it[2] <= TRSV_UPDATE_COLS, "backward item {it:?}");
                            (it[2]..it[3]).for_each(|c| cols[c as usize] += 1);
                        }
                        for c in 0..q {
                            assert_eq!(cols[c as usize], u32::from(j > 0 && c < k * block), "backward launch {j}, supernode {s}, column {c}");
                        }
                    }
                    let past = |l: &Vec<Vec<u32>>| l[nd as usize..].iter().all(|items| items.chunks(TRSV_ITEM).all(|it| it[0] != s));
                    assert!(past(&w.wide_fwd) && past(&w.wide_bwd), "supernode {s}: items past its {nd} launches");
                }
                for items in w.wide_fwd.iter().chain(&w.wide_bwd) {
                    let solves: Vec<bool> = items.chunks(TRSV_ITEM).map(|it| it[2] == it[3]).collect();
                    assert!(solves.windows(2).all(|x| x[0] || !x[1]), "a launch's solve items come first");
                }
            }
            assert!(wides > 1, "wide supernodes: {wides}");
        }
    }

    /// Unset, the panel steps unroll a full diagonal tile's pivots on a
    /// discrete device, not on an integrated one; set, what it says on
    /// every device.
    #[test]
    fn the_potf2_unroll_is_the_devices_unless_set() {
        let unset = GpuSupernodalParams::default();
        assert_eq!((unset.potf2_unroll_on(&arch(2, 32, true)), unset.potf2_unroll_on(&discrete(128))), (false, true));
        let set = GpuSupernodalParams { potf2_unroll: Some(true), ..unset };
        assert_eq!((set.potf2_unroll_on(&arch(2, 32, true)), set.potf2_unroll_on(&discrete(128))), (true, true));
    }

    /// Unset, four warps factor a full diagonal tile on NVIDIA and AMD
    /// devices, one on Apple devices and the CPU device; set, what it says
    /// on every device.
    #[test]
    fn the_diag_warps_are_the_devices_unless_set() {
        let unset = GpuSupernodalParams::default();
        let apple = crate::Arch { vendor: crate::Vendor::Apple, ..discrete(10) };
        let cpu = crate::Arch { vendor: crate::Vendor::Cpu, ..arch(1, 32, true) };
        let devices = [arch(2, 32, true), discrete(128), apple, cpu];
        assert_eq!(devices.each_ref().map(|d| unset.diag_warps_on(d)), [4, 4, 1, 1]);
        let set = GpuSupernodalParams { diag_warps: Some(1), ..unset };
        assert_eq!(devices.each_ref().map(|d| set.diag_warps_on(d)), [1, 1, 1, 1]);
    }

    /// Unset, `k_syrk` takes tiles of 32, except 64 for an f32 factor on an
    /// integrated device; set, what it says on every device.
    #[test]
    fn the_syrk_tile_is_the_devices_unless_set() {
        let unset = GpuSupernodalParams::default();
        assert_eq!((unset.syrk_tile_on(&arch(2, 32, true), 8), unset.syrk_tile_on(&arch(2, 32, true), 4)), (32, 64));
        assert_eq!((unset.syrk_tile_on(&discrete(128), 8), unset.syrk_tile_on(&discrete(128), 4)), (32, 32));
        let set = GpuSupernodalParams { syrk_tile: Some(64), ..unset };
        assert_eq!((set.syrk_tile_on(&arch(2, 32, true), 8), set.syrk_tile_on(&discrete(128), 4)), (64, 64));
    }

    /// Unset, the solve's GEMV takes blocks of 16384 multiply-adds on an
    /// integrated device and of 2048 on a discrete one; set, what it says
    /// on every device.
    #[test]
    fn the_gemv_work_is_the_devices_unless_set() {
        let unset = GpuSupernodalParams::default();
        assert_eq!(unset.gemv_work_on(&arch(2, 32, true)), 16384);
        assert_eq!(unset.gemv_work_on(&arch(128, 32, false)), 2048);
        let set = GpuSupernodalParams { gemv_work: Some(1000), ..unset };
        assert_eq!((set.gemv_work_on(&arch(2, 32, true)), set.gemv_work_on(&arch(128, 32, false))), (1000, 1000));
    }

    /// The GEMV items of a supernode with rows below hold at most the
    /// device's `gemv_work` multiply-adds (or one column), so a discrete
    /// device gets more of them than an integrated one; the gather items
    /// hold at most `gather_work` (or one row).
    #[test]
    fn the_solve_items_hold_the_devices_work() {
        let (p, _, _) = analyse(40);
        let kn = GpuSupernodalParams { gather_work: 4096, ..GpuSupernodalParams::default() };
        let count = |a: &crate::Arch| {
            let work = kn.gemv_work_on(a);
            let mut items = 0;
            for w in build_levels(&p, &kn, a, 8) {
                for it in w.bg_items.chunks(3) {
                    let (s, c0, c1) = (it[0] as usize, it[1], it[2]);
                    let rows = (p.dims[2 * s + 1] - p.dims[2 * s]) as usize;
                    assert!((c1 - c0) as usize * rows <= work.max(rows), "a GEMV item of {} columns x {rows} rows over {work}", c1 - c0);
                }
                // g_rows holds the rows field by field: the descendant is field 2
                let n = (w.g_rows.len() / GATHER_ROW) as u32;
                for g in w.g_items.chunks(2) {
                    let work: usize = (g[0]..g[0] + g[1]).map(|r| p.dims[2 * w.g_rows[(2 * n + r) as usize] as usize] as usize).sum();
                    assert!(g[1] == 1 || work <= kn.gather_work, "a gather item of {} rows and {work} multiply-adds", g[1]);
                }
                items += w.bg_items.len() / 3;
            }
            items
        };
        let (int, dis) = (count(&arch(2, 32, true)), count(&discrete(128)));
        assert!(dis > int, "GEMV items: discrete {dis}, integrated {int}");
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
