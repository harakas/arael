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

/// Tile of the small-descendant update kernel (rows x columns of a target panel).
pub const TR: usize = 64;
pub const TC: usize = 64;
/// Elements per small rectangle.
pub const RECT_CAP: u32 = 64;
/// Units per small rectangle.
pub const GROUP: usize = 8;
/// Tile of the big-descendant update and of the panel's trailing update.
pub const TB: usize = 64;
/// The k slab of the big-descendant GEMM core.
pub const TK: usize = 16;
/// The panel step: columns factored per round of the three panel kernels.
pub const NB: usize = 32;
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
pub const H_PR_S: usize = 7;
pub const H_PR_RMAP: usize = 8;
pub const H_PR_CMAP: usize = 9;
pub const H_RMAP: usize = 10;
pub const H_CMAP: usize = 11;
pub const H_PAIR_KW_PTR: usize = 12;
pub const H_ROW_SC: usize = 13;
pub const H_ROW_SC_OFF: usize = 14;
/// The permutation, for the device-side solve: the block elimination order,
/// the permuted block widths, and the matrix's scalar partition.
pub const H_ORDER: usize = 15;
pub const H_BLK_WIDTH: usize = 16;
pub const H_PART: usize = 17;
/// Words of the header.
pub const HEADER: usize = 32;

/// How the work is cut. The defaults are the measured ones (docs/dev/GPU.md, E3).
#[derive(Clone, Copy, Debug)]
pub struct GpuSupernodalParams {
    /// Supernodes narrower than this (scalar columns) are factored by the
    /// one-cube small kernel; the others by the batched panel steps.
    pub small_q: u32,
    /// A descendant update of at least this many multiply-adds goes through
    /// the GEMM-core kernel; smaller ones through the tile kernel.
    pub big_pair_flops: f64,
    /// Elements of a target tile per cube of the small-update kernel; a tile
    /// with more is cut into chunks.
    pub elem_cap: usize,
    /// Multiply-adds per cube of the solve's gather and GEMV kernels.
    pub solve_work: usize,
    /// Loads in flight per iteration of the backward dot products; `None`
    /// picks 8 for f64 and 4 for f32.
    pub lane_unroll: Option<usize>,
}

impl Default for GpuSupernodalParams {
    fn default() -> Self {
        GpuSupernodalParams { small_q: 96, big_pair_flops: 2e6, elem_cap: 16384, solve_work: 65536, lane_unroll: None }
    }
}

/// One descendant update (target `s` <- descendant `d`): the descendant's
/// rows from `r0` (`m` of them, `kw` of them landing in `s`'s columns), and
/// the runs that place them.
#[derive(Clone)]
pub(crate) struct Pair {
    pub s: u32,
    pub d: u32,
    pub r0: u32,
    pub m: u32,
    pub kw: u32,
    pub row_run0: usize,
    pub row_runs: usize,
    pub col_run0: usize,
    pub col_runs: usize,
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
    pub run_tr: Vec<u32>,
    pub run_dr: Vec<u32>,
    pub run_w: Vec<u32>,
    pub run_tc: Vec<u32>,
    pub run_uc: Vec<u32>,
    pub run_cw: Vec<u32>,
    pub rmap: Vec<u32>,
    pub cmap: Vec<u32>,
    pub pr_rmap: Vec<u32>,
    pub pr_cmap: Vec<u32>,
    pub pair_kw_ptr: Vec<u32>,
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

    /// The descendant pairs of every supernode with the runs that place their
    /// rows and columns in the target panel, the flat row and column maps,
    /// the running count of mid rows, and the permuted scalar index of every
    /// row below a diagonal block.
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
                let (row_run0, col_run0) = (self.run_tr.len(), self.run_tc.len());
                for e in start..dblk.len() {
                    let kb = dblk[e];
                    self.run_tr.push(row_of[&kb]);
                    self.run_dr.push(drow[e] - r0);
                    self.run_w.push(self.blk_width[kb as usize]);
                }
                let mut uc = 0;
                for e in start..start + mid {
                    let kb = dblk[e];
                    self.run_tc.push(self.sc_start[kb as usize] - col0);
                    self.run_uc.push(uc);
                    self.run_cw.push(self.blk_width[kb as usize]);
                    uc += self.blk_width[kb as usize];
                }
                let (row_runs, col_runs) = (self.run_tr.len() - row_run0, self.run_tc.len() - col_run0);
                self.pr_rmap.push(self.rmap.len() as u32);
                self.pr_cmap.push(self.cmap.len() as u32);
                for r in 0..row_runs {
                    for w in 0..self.run_w[row_run0 + r] {
                        self.rmap.push(self.run_tr[row_run0 + r] + w);
                    }
                }
                for c in 0..col_runs {
                    for w in 0..self.run_cw[col_run0 + c] {
                        self.cmap.push(self.run_tc[col_run0 + c] + w);
                    }
                }
                debug_assert_eq!(self.rmap.len() as u32 - self.pr_rmap[self.pr_rmap.len() - 1], m);
                debug_assert_eq!(self.cmap.len() as u32 - self.pr_cmap[self.pr_cmap.len() - 1], kw);
                self.pairs.push(Pair { s: s as u32, d: d as u32, r0, m, kw, row_run0, row_runs, col_run0, col_runs });
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
        self.pair_kw_ptr = vec![0u32; self.pairs.len() + 1];
        for i in 0..self.pairs.len() {
            self.pair_kw_ptr[i + 1] = self.pair_kw_ptr[i] + self.pairs[i].kw;
        }
    }

    /// The plan arrays packed into one buffer behind the header that says
    /// where each starts.
    pub fn pack(&self) -> Vec<u32> {
        let mut buf: Vec<u32> = vec![0; HEADER];
        let pr_d: Vec<u32> = self.pairs.iter().map(|x| x.d).collect();
        let pr_r0: Vec<u32> = self.pairs.iter().map(|x| x.r0).collect();
        let pr_s: Vec<u32> = self.pairs.iter().map(|x| x.s).collect();
        let arrays: [(usize, &[u32]); 18] = [
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
            (H_PR_S, &pr_s),
            (H_PR_RMAP, &self.pr_rmap),
            (H_PR_CMAP, &self.pr_cmap),
            (H_RMAP, &self.rmap),
            (H_CMAP, &self.cmap),
            (H_PAIR_KW_PTR, &self.pair_kw_ptr),
            (H_ROW_SC, &self.row_sc),
            (H_ROW_SC_OFF, &self.row_sc_off),
        ];
        for (slot, v) in arrays {
            buf[slot] = buf.len() as u32;
            buf.extend_from_slice(v);
        }
        buf
    }
}

/// A small descendant's update clipped to one tile: `ni x nj` elements from
/// (`i0`, `j0`) of the tile, read from the descendant's rows `dra..` and
/// `dca..` (offsets from the pair's first row).
struct Rect {
    pair: u32,
    i0: u32,
    ni: u32,
    j0: u32,
    nj: u32,
    dra: u32,
    dca: u32,
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
    /// 5 per chunk: target, row0, col0, first rectangle, end.
    pub tile_items: Vec<u32>,
    /// 4 per rectangle: pair, geometry (i0 | ni << 8 | j0 << 16 | nj << 24), dra, dca.
    pub tile_rects: Vec<u32>,
    /// 5 per (target, tile): target, row0, col0, first rectangle, end.
    pub big_items: Vec<u32>,
    /// 5 per rectangle: pair, da, db, ca, cb.
    pub big_rects: Vec<u32>,
    pub small_targets: Vec<u32>,
    pub big_targets: Vec<u32>,
    pub level_sn: Vec<u32>,
    /// 4 per gather item: pairs [pa, pb), mid rows [e0, e1) counted from pair pa's first.
    pub g_items: Vec<u32>,
    /// 3 per GEMV item: supernode, columns [c0, c1).
    pub bg_items: Vec<u32>,
    /// The widest big target: the number of panel steps is `ceil(maxq_big / NB)`.
    pub maxq_big: u32,
    /// Per panel step, the trsm and syrk grid sizes, as the kernels count them.
    pub trsm_grid: Vec<u32>,
    pub syrk_grid: Vec<u32>,
}

/// The level schedule of `p`: every supernode's work sorted into its level.
pub(crate) fn build_levels(p: &Plan, kn: &GpuSupernodalParams) -> Vec<LevelWork> {
    let mut lw: Vec<LevelWork> = (0..p.nlevels).map(|_| LevelWork::default()).collect();
    let mut g_works = vec![0usize; p.nlevels];
    for s in 0..p.ns {
        let w = &mut lw[p.level[s]];
        let g_work = &mut g_works[p.level[s]];
        let q = p.dims[2 * s];
        let h = p.dims[2 * s + 1];
        if q < kn.small_q {
            w.small_targets.push(s as u32);
        } else {
            w.big_targets.push(s as u32);
            w.maxq_big = w.maxq_big.max(q);
        }
        w.level_sn.push(s as u32);
        // the solve's items: the pairs of s appended to the level's open gather
        // item while it stays under the work bound, a wide pair cut into row ranges
        for pi in p.pair_ptr[s] as usize..p.pair_ptr[s + 1] as usize {
            let pr = &p.pairs[pi];
            let dq = p.dims[2 * pr.d as usize] as usize;
            let work = pr.kw as usize * dq;
            let ng = w.g_items.len() / 4;
            let open = ng > 0
                && w.g_items[4 * ng - 3] as usize == pi
                && w.g_items[4 * ng - 2] == 0
                && w.g_items[4 * ng - 1] == p.pair_kw_ptr[pi] - p.pair_kw_ptr[w.g_items[4 * ng - 4] as usize];
            if work > kn.solve_work {
                let per = ((kn.solve_work / dq) as u32).max(1);
                let mut e0 = 0;
                while e0 < pr.kw {
                    w.g_items.extend_from_slice(&[pi as u32, pi as u32 + 1, e0, pr.kw.min(e0 + per)]);
                    e0 += per;
                }
                *g_work = kn.solve_work + 1;
            } else if open && *g_work + work <= kn.solve_work {
                w.g_items[4 * ng - 3] = pi as u32 + 1;
                w.g_items[4 * ng - 1] = p.pair_kw_ptr[pi + 1] - p.pair_kw_ptr[w.g_items[4 * ng - 4] as usize];
                *g_work += work;
            } else {
                w.g_items.extend_from_slice(&[pi as u32, pi as u32 + 1, 0, pr.kw]);
                *g_work = work;
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
        // the pairs: the small ones as rectangles per tile, the big ones listed
        let mut tiles: HashMap<u64, Vec<Rect>> = HashMap::new();
        let mut bigs = Vec::new();
        for pi in p.pair_ptr[s] as usize..p.pair_ptr[s + 1] as usize {
            let pr = &p.pairs[pi];
            let dq = p.dims[2 * pr.d as usize];
            if pr.m as f64 * pr.kw as f64 * dq as f64 >= kn.big_pair_flops {
                bigs.push(pi);
                continue;
            }
            for r in 0..pr.row_runs {
                let (tr, dr, wr) = (p.run_tr[pr.row_run0 + r], p.run_dr[pr.row_run0 + r], p.run_w[pr.row_run0 + r]);
                for c in 0..pr.col_runs {
                    let (tc, uc, cw) = (p.run_tc[pr.col_run0 + c], p.run_uc[pr.col_run0 + c], p.run_cw[pr.col_run0 + c]);
                    let (trr, tcc) = (TR as u32, TC as u32);
                    for rt in tr / trr..=(tr + wr - 1) / trr {
                        let (ra, rb) = (tr.max(rt * trr), (tr + wr).min((rt + 1) * trr));
                        for ct in tc / tcc..=(tc + cw - 1) / tcc {
                            if rt < ct {
                                continue; // above the diagonal
                            }
                            let (ca, cb) = (tc.max(ct * tcc), (tc + cw).min((ct + 1) * tcc));
                            let nj = cb - ca;
                            let step = (RECT_CAP / nj).max(1);
                            let mut r0 = ra;
                            while r0 < rb {
                                let r1 = rb.min(r0 + step);
                                tiles.entry(((rt as u64) << 20) | ct as u64).or_default().push(Rect {
                                    pair: pi as u32,
                                    i0: r0 - rt * trr,
                                    ni: r1 - r0,
                                    j0: ca - ct * tcc,
                                    nj,
                                    dra: dr + (r0 - tr),
                                    dca: uc + (ca - tc),
                                });
                                r0 += step;
                            }
                        }
                    }
                }
            }
        }
        let mut keys: Vec<u64> = tiles.keys().copied().collect();
        keys.sort_unstable();
        for key in keys {
            let rs = tiles.get_mut(&key).unwrap();
            // alike sizes together, then chunks of at most elem_cap elements
            rs.sort_by_key(|r| std::cmp::Reverse(r.ni * r.nj));
            let mut at = 0;
            while at < rs.len() {
                let rect0 = (w.tile_rects.len() / 4) as u32;
                let mut off = 0u32;
                while at < rs.len() && (off == 0 || (off + rs[at].ni * rs[at].nj) as usize <= kn.elem_cap) {
                    let r = &rs[at];
                    w.tile_rects.extend_from_slice(&[r.pair, r.i0 | (r.ni << 8) | (r.j0 << 16) | (r.nj << 24), r.dra, r.dca]);
                    off += r.ni * r.nj;
                    at += 1;
                }
                w.tile_items.extend_from_slice(&[
                    s as u32,
                    (key >> 20) as u32 * TR as u32,
                    (key & 0xfffff) as u32 * TC as u32,
                    rect0,
                    (w.tile_rects.len() / 4) as u32,
                ]);
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
            ]);
        }
    }
    // the panel steps' grid sizes
    for w in &mut lw {
        let mut j = 0u32;
        while j < w.maxq_big {
            let (mut tg, mut sg) = (0u32, 0u32);
            for &s in &w.big_targets {
                let (q, h) = (p.dims[2 * s as usize], p.dims[2 * s as usize + 1]);
                if j >= q {
                    continue;
                }
                let nb = (NB as u32).min(q - j);
                tg += (h - j - nb).div_ceil(TB as u32);
                if j + nb < q {
                    sg += (h - j - nb).div_ceil(TB as u32) * (q - j - nb).div_ceil(TB as u32);
                }
            }
            w.trsm_grid.push(tg);
            w.syrk_grid.push(sg);
            j += NB as u32;
        }
    }
    lw
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

    /// Every element of every descendant update that the factor needs (the
    /// tiles at or below the diagonal) is in exactly one rectangle, small or
    /// big; the solve's items cover every mid row and every column once.
    fn check_coverage(kn: &GpuSupernodalParams, both_kernels: bool) {
        let (p, _, _) = analyse(12);
        let lw = build_levels(&p, kn);
        let mut hits: Vec<Vec<u8>> = p.pairs.iter().map(|pr| vec![0u8; (pr.m * pr.kw) as usize]).collect();
        let (mut small_rects, mut big_rects) = (0usize, 0usize);
        for w in &lw {
            for it in w.tile_items.chunks(5) {
                for r in it[3] as usize..it[4] as usize {
                    let (pi, geo, dra, dca) = (w.tile_rects[4 * r] as usize, w.tile_rects[4 * r + 1], w.tile_rects[4 * r + 2], w.tile_rects[4 * r + 3]);
                    let (ni, nj) = ((geo >> 8) & 0xff, geo >> 24);
                    let m = p.pairs[pi].m;
                    for jj in 0..nj {
                        for ii in 0..ni {
                            hits[pi][((dca + jj) * m + dra + ii) as usize] += 1;
                        }
                    }
                    small_rects += 1;
                }
            }
            for it in w.big_items.chunks(5) {
                for r in it[3] as usize..it[4] as usize {
                    let (pi, da, db, ca, cb) = (w.big_rects[5 * r] as usize, w.big_rects[5 * r + 1], w.big_rects[5 * r + 2], w.big_rects[5 * r + 3], w.big_rects[5 * r + 4]);
                    let m = p.pairs[pi].m;
                    for j in ca..cb {
                        for i in da..db {
                            hits[pi][(j * m + i) as usize] += 1;
                        }
                    }
                    big_rects += 1;
                }
            }
        }
        assert!(small_rects > 0, "the tile kernel in use");
        assert!(!both_kernels || big_rects > 0, "both update kernels in use: {small_rects} small, {big_rects} big");
        for (pi, pr) in p.pairs.iter().enumerate() {
            let rmap = &p.rmap[p.pr_rmap[pi] as usize..][..pr.m as usize];
            let cmap = &p.cmap[p.pr_cmap[pi] as usize..][..pr.kw as usize];
            for j in 0..pr.kw as usize {
                for i in 0..pr.m as usize {
                    let needed = rmap[i] as usize / TR >= cmap[j] as usize / TC;
                    let h = hits[pi][j * pr.m as usize + i];
                    assert_eq!(h, needed as u8, "pair {pi} element ({i}, {j}) covered {h} times, needed {needed}");
                }
            }
        }
        // the gather items: every mid row of every pair once
        let mut rows: Vec<u32> = vec![0; p.pairs.len()];
        for w in &lw {
            for it in w.g_items.chunks(4) {
                let (pa, pb, e0, e1) = (it[0] as usize, it[1] as usize, it[2], it[3]);
                let base = p.pair_kw_ptr[pa];
                for e in e0..e1 {
                    let pi = (pa..pb).rev().find(|&q| p.pair_kw_ptr[q] - base <= e).unwrap();
                    rows[pi] += 1;
                }
            }
        }
        for (pi, pr) in p.pairs.iter().enumerate() {
            assert_eq!(rows[pi], pr.kw, "pair {pi}: gather rows");
        }
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
        check_coverage(&GpuSupernodalParams::default(), false);
    }

    #[test]
    fn plan_covers_every_update_once_with_every_kernel_busy() {
        check_coverage(&GpuSupernodalParams { small_q: 8, big_pair_flops: 100.0, elem_cap: 256, solve_work: 512, lane_unroll: None }, true);
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
