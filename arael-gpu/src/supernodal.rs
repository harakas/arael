//! [`GpuSupernodal`]: the block supernodal Cholesky and its level-set
//! solve on a [`Device`]: the kernels of `kernels/supernodal.cpp` launched
//! over the plan of [`crate::plan`].

use std::fmt::Write as _;
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arael_faer::bsc::SymbolicSparseBlockColMat;
use arael_faer::schur::SchurReal;
use arael_faer::supernodal::SupernodalSymbolic;

use crate::plan::{self, GpuSupernodalParams, NB, Plan, SparseIndex, build_levels};
use crate::{Arg, Device, GpuError, Lane, Launch, Scalar, WIDE, Wide, is_f64, scalar_bytes};

/// Arrays packed into one block, each starting on a 256-byte boundary.
#[derive(Default)]
pub(crate) struct Packer {
    pub words: Vec<u32>,
}

impl Packer {
    /// A packer with room for `words` words.
    pub fn with_capacity(words: usize) -> Self {
        Packer { words: Vec::with_capacity(words) }
    }

    /// Append `v`, returning where it starts, in words.
    pub fn push(&mut self, v: &[u32]) -> usize {
        let at = self.words.len().div_ceil(64) * 64;
        self.words.resize(at, 0);
        self.words.extend_from_slice(v);
        at
    }
}

/// The kernels, compiled per scalar and lane unroll.
const SOURCE: &str = include_str!("../kernels/supernodal.cpp");

/// Threads per block of every kernel but `k_trsm`, which runs
/// [`GpuSupernodalParams::trsm_rows`].
const THREADS: u32 = 256;
const _: () = assert!(plan::GATHER_ITEM_ROWS == THREADS, "a gather item is a row a thread");
const _: () = assert!(plan::TRSV_UPDATE_ROWS == THREADS, "a forward update item of the wide trsv is a row a thread");

/// The dynamic shared memory of `k_small_factor` on a supernode of `q`
/// columns in a scalar of `fs` bytes: the diagonal block, the step's
/// diagonal tile (its columns NB apart) and its diagonal's reciprocals.
fn small_shared_bytes(q: usize, fs: usize) -> usize {
    (q * q + q.min(NB) * NB + NB) * fs
}

/// Why [`GpuSupernodal::factorize`] failed.
#[derive(Debug)]
pub enum FactorizeError {
    /// A diagonal pivot was not positive. The factor is unusable; the caller
    /// raises the damping, as with the CPU factorization.
    NotPositiveDefinite,
    Device(GpuError),
}

impl From<GpuError> for FactorizeError {
    fn from(e: GpuError) -> Self {
        FactorizeError::Device(e)
    }
}

impl core::fmt::Display for FactorizeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            FactorizeError::NotPositiveDefinite => write!(f, "a pivot of the factorization is not positive"),
            FactorizeError::Device(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for FactorizeError {}

struct Kernels<K> {
    zero: K,
    zero_u32: K,
    seed_f32: K,
    /// None on a device without f64: f64 values are rounded on the host.
    seed_f64: Option<K>,
    seed_wide: K,
    pair_update: K,
    big_update: K,
    small_factor: K,
    chain: K,
    potf2: K,
    trsm: K,
    syrk: K,
    fwd_gather: K,
    fwd_trsv: K,
    bwd_gemv: K,
    bwd_trsv: K,
    fwd_chain: K,
    bwd_chain: K,
    perm_in: K,
    perm_out: K,
    tile_inv: K,
    fwd_trsv_wide: K,
    bwd_trsv_wide: K,
    big_pair: K,
}

/// What the factorization launches, in order: one level's kernels, or a
/// chain of levels in one `k_chain` launch.
enum Step {
    Level(usize),
    /// The chain's level records (4 words a level, a byte offset into the
    /// structure block), its levels, and its dynamic shared memory.
    Chain { records: usize, n: u32, shared: u32 },
}

/// One level's work lists, as byte offsets into the structure block, and
/// its launch sizes.
struct Level {
    pair_items: usize,
    n_pair_items: u32,
    big_items: usize,
    big_rects: usize,
    n_big_items: u32,
    /// The big updates tiled in the pairs' own rows and columns
    /// (`k_big_pair`), and their count.
    pair_tiles: usize,
    n_pair_tiles: u32,
    small: usize,
    n_small: u32,
    /// Dynamic shared memory of `k_small_factor` ([`small_shared_bytes`]) for the widest small supernode.
    small_shared: u32,
    /// The block of `k_small_factor`: a thread per row of the tallest small
    /// supernode, in whole warps, at most THREADS.
    small_threads: u32,
    big: usize,
    n_big: u32,
    /// Per panel step, the grid sizes of `k_trsm` and `k_syrk`, and of the
    /// `k_syrk` split by a lookahead (plan's `syrk_near_grid`).
    trsm_grid: Vec<u32>,
    syrk_grid: Vec<u32>,
    syrk_near_grid: Vec<u32>,
    syrk_far_grid: Vec<u32>,
    /// The level's supernodes (for tests).
    #[cfg(test)]
    n_all: u32,
    /// Dynamic shared memory of a solve chain through the level: q values
    /// for its widest supernode.
    trsv_shared: u32,
    /// The supernodes the one-block trsv kernels solve, and their dynamic
    /// shared memory, q values for the widest of them.
    narrow: usize,
    n_narrow: u32,
    narrow_shared: u32,
    /// Per launch of the trsvs over many blocks, forward and backward:
    /// the items' byte offset in the structure block and count.
    wide_fwd: Vec<(usize, u32)>,
    wide_bwd: Vec<(usize, u32)>,
    /// The level's gather rows: their word offset in the structure block and count.
    g_rows: (u32, u32),
    g_items: usize,
    n_g: u32,
    bg_items: usize,
    n_bg: u32,
}

/// Byte offsets of the work block's sections, each on a 256-byte boundary.
struct Sections {
    factor: usize,
    /// The solve's working vector, in the factor's scalar and ordering.
    xb: usize,
    /// The pivot flag of [`GpuSupernodal::factorize`], a `u32`.
    fail: usize,
    /// The matrix's values from the host, `f32` or `f64` as given.
    vals: usize,
    /// The right-hand side and solution of [`GpuSupernodal::solve`], `f64`.
    rhs: usize,
    /// The inverted diagonal tiles of the factor (`k_tile_inv`,
    /// [`plan::INV_TILE`] values each), in the factor's scalar.
    inv: usize,
    bytes: usize,
}

impl Sections {
    fn new(factor: usize, xb: usize, vals: usize, rhs: usize, inv: usize) -> Self {
        let mut at = 0;
        let mut take = |bytes: usize| {
            let start = at;
            at = (at + bytes.max(1)).next_multiple_of(256);
            start
        };
        let (factor, xb, fail, vals, rhs, inv) = (take(factor), take(xb), take(4), take(vals), take(rhs), take(inv));
        Sections { factor, xb, fail, vals, rhs, inv, bytes: at }
    }
}

/// The device factorization of one sparsity structure on a [`Device`]: the plan built from a [`SupernodalSymbolic`] once, the factor
/// kept on the device, and the kernels that fill it from a matrix's values
/// and solve against it.
///
/// `F` is the factor's scalar, `f32` or `f64`; the matrix handed to
/// [`factorize`](Self::factorize) and the right-hand side of
/// [`solve`](Self::solve) may be in either. Every device buffer is made in
/// [`new`](Self::new): the structure's integer data (the plan, the seed map,
/// every level's work lists) in one block, uploaded in one copy, and the
/// factor, the solve's vectors and the uploads in another.
///
/// ```ignore
/// let dev = Arc::new(CudaDevice::new(0)?);
/// let mut g = GpuSupernodal::<f32, _>::new(&dev, &sn, a.symbolic(), &Default::default(), Wide::F64)?;
/// g.factorize(a.vals())?;          // Err(NotPositiveDefinite) when a pivot is not positive
/// g.solve(&mut rhs)?;              // rhs becomes x, in the matrix's ordering
/// ```
pub struct GpuSupernodal<F: Scalar, D: Device> {
    dev: Arc<D>,
    k: Kernels<D::Kernel>,
    /// The plan words (at 0, as the plan's header offsets require), the
    /// tiles, a word holding 1 (the gate of a solve that always runs), and
    /// every level's lists.
    ints: D::Block,
    tiles: usize,
    on: usize,
    ntiles: u32,
    /// The seed's threads per stored tile, a block each.
    seed_lanes: u32,
    /// Columns per block of `k_syrk`'s trailing update
    /// ([`GpuSupernodalParams::syrk_block`]).
    syrk_block: u32,
    /// Rows per block of `k_trsm` ([`GpuSupernodalParams::trsm_rows`]).
    trsm_rows: u32,
    /// The inverted diagonal tiles, a `k_tile_inv` block each.
    n_inv: u32,
    levels: Vec<Level>,
    steps: Vec<Step>,
    /// The solve's steps: a level's kernels, or a chain of levels (forward
    /// in order, backward in reverse).
    solve_steps: Vec<Step>,
    work: D::Block,
    sec: Sections,
    n: usize,
    nb: u32,
    nvals: usize,
    factor_vals: usize,
    /// The scalar of the vectors it solves and of the values a device
    /// factorization reads.
    wide: Wide,
    /// The right-hand side of [`solve`](Self::solve) in `f64`, and encoded.
    host_x: Vec<f64>,
    host_bytes: Vec<u8>,
    id: u64,
    _f: PhantomData<F>,
}

/// The next factor's [`GpuSupernodal::id`].
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

impl<F: Scalar, D: Device> GpuSupernodal<F, D> {
    /// The device plan of `sn`, the analysis of a matrix with the block
    /// structure `asym`, uploaded to `dev`, with the factor's buffers and
    /// the kernels; the vectors it solves and the values a device
    /// factorization reads are in `wide`. [`GpuError::Limit`] when a
    /// supernode needs more shared memory than the device gives;
    /// [`GpuError::Unavailable`] for an f64 factor or [`Wide::F64`] on a
    /// device without f64.
    pub fn new(
        dev: &Arc<D>,
        sn: &SupernodalSymbolic,
        asym: &SymbolicSparseBlockColMat<SparseIndex>,
        params: &GpuSupernodalParams,
        wide: Wide,
    ) -> Result<Self, GpuError> {
        let fs = core::mem::size_of::<F>();
        crate::check_f64(dev.arch(), fs == 8, "an f64 factor")?;
        crate::check_f64(dev.arch(), wide == Wide::F64, "Wide::F64")?;
        // a supernode the small kernel's shared memory does not hold, or
        // wider than its block (its diagonal rows are the first pass's), takes
        // the panel steps
        let limit = dev.arch().shared_bytes as usize;
        let fits = |q: usize| small_shared_bytes(q, fs) <= limit;
        let mut small_q = (params.small_q as usize).min(THREADS as usize + 1);
        while small_q > 1 && !fits(small_q - 1) {
            small_q -= 1;
        }
        let params = &GpuSupernodalParams { small_q: small_q as u32, ..*params };
        let syrk_block = params.syrk_block_on(dev.arch(), fs);
        assert!(syrk_block as usize >= NB && (syrk_block as usize).is_multiple_of(NB), "syrk_block {syrk_block} is a multiple of {NB}");
        let trsm_rows = params.trsm_rows_on(dev.arch());
        assert!(trsm_rows as usize >= NB && (NB * NB).is_multiple_of(trsm_rows as usize), "trsm_rows {trsm_rows} is at least {NB} and divides {NB} * {NB}");
        let syrk_tile = params.syrk_tile_on(dev.arch(), fs);
        assert!(syrk_tile == 32 || syrk_tile == 64, "syrk_tile {syrk_tile} is 32 or 64");
        let sizes = DeviceSizes { syrk_block, syrk_tile, trsm_rows, potf2_unroll: params.potf2_unroll_on(dev.arch()), trsv_block: params.trsv_block_on(dev.arch()) };
        let plan = Plan::new(sn, asym);
        let lw = build_levels(&plan, params, dev.arch(), fs);
        let lu = params.lane_unroll.unwrap_or(if fs == 4 { 4 } else { 8 });
        assert!(lu >= 2 && lu.is_multiple_of(2), "the lane unroll is even and at least 2");
        // room for every array and its alignment, so the block is not regrown
        let plan_words = plan.pack();
        let level_words: usize = lw
            .iter()
            .map(|w| {
                let lists = [&w.pair_items, &w.big_items, &w.big_rects, &w.pair_tiles, &w.small_targets, &w.big_targets, &w.level_sn, &w.g_rows, &w.bg_items, &w.narrow];
                let wide: usize = w.wide_fwd.iter().chain(&w.wide_bwd).map(|v| v.len() + 64).sum();
                lists.iter().map(|v| v.len()).sum::<usize>() + w.g_items.len() / 2 * 3 + wide + 16 * 64
            })
            .sum();
        // (the chains' records: 4 and 5 words a level)
        let mut pk = Packer::with_capacity(plan_words.len() + plan.tiles.len() + level_words + 9 * lw.len() + 1024);
        let at_plan = pk.push(&plan_words);
        debug_assert_eq!(at_plan, 0, "the plan's header offsets are relative to the block start");
        let at_tiles = pk.push(&plan.tiles);
        let at_on = pk.push(&[1]);
        let q = |s: &u32| plan.dims[2 * *s as usize] as usize;
        let chain_flops = if fs == 4 { params.chain_flops_f32 } else { params.chain_flops_f64 };
        let chains = plan::chains(&lw, chain_flops, params.chain_items);
        let solve_chains = plan::solve_chains(&lw, params.solve_chain_work);
        let first_sn: Vec<u32> = lw.iter().map(|w| w.level_sn[0]).collect();
        let mut levels = Vec::with_capacity(lw.len());
        for w in lw {
            let small_q = w.small_targets.iter().map(q).max().unwrap_or(0);
            let small_h = w.small_targets.iter().map(|&s| plan.dims[2 * s as usize + 1]).max().unwrap_or(0);
            let max_q = w.level_sn.iter().map(q).max().unwrap_or(0);
            let narrow_q = w.narrow.iter().map(q).max().unwrap_or(0);
            let (small_shared, trsv_shared, narrow_shared) = (small_shared_bytes(small_q, fs), max_q * fs, narrow_q * fs);
            assert!(small_shared <= limit, "small_q is capped to the shared memory: {small_shared} bytes for {small_q} columns, {limit} given");
            assert!(narrow_shared <= limit, "a supernode wider than the shared memory holds takes the trsvs over many blocks");
            let push_launches = |pk: &mut Packer, launches: &[Vec<u32>]| -> Vec<(usize, u32)> {
                launches.iter().map(|items| (4 * pk.push(items), (items.len() / plan::TRSV_ITEM) as u32)).collect()
            };
            let (wide_fwd, wide_bwd) = (push_launches(&mut pk, &w.wide_fwd), push_launches(&mut pk, &w.wide_bwd));
            // the gather items with their first row's word offset in the structure block
            let g_at = pk.push(&w.g_rows) as u32;
            let g_n = (w.g_rows.len() / plan::GATHER_ROW) as u32;
            let g_items: Vec<u32> = w.g_items.chunks(2).flat_map(|it| [g_at + it[0], it[1], g_n]).collect();
            levels.push(Level {
                pair_items: 4 * pk.push(&w.pair_items),
                n_pair_items: (w.pair_items.len() / plan::PAIR_ITEM) as u32,
                big_items: 4 * pk.push(&w.big_items),
                big_rects: 4 * pk.push(&w.big_rects),
                n_big_items: (w.big_items.len() / plan::BIG_ITEM) as u32,
                pair_tiles: 4 * pk.push(&w.pair_tiles),
                n_pair_tiles: (w.pair_tiles.len() / plan::PAIR_TILE) as u32,
                small: 4 * pk.push(&w.small_targets),
                n_small: w.small_targets.len() as u32,
                small_shared: small_shared as u32,
                small_threads: small_h.next_multiple_of(32).clamp(32, THREADS),
                big: 4 * pk.push(&w.big_targets),
                n_big: w.big_targets.len() as u32,
                trsm_grid: w.trsm_grid,
                syrk_grid: w.syrk_grid,
                syrk_near_grid: w.syrk_near_grid,
                syrk_far_grid: w.syrk_far_grid,
                #[cfg(test)]
                n_all: w.level_sn.len() as u32,
                trsv_shared: trsv_shared as u32,
                narrow: 4 * pk.push(&w.narrow),
                n_narrow: w.narrow.len() as u32,
                narrow_shared: narrow_shared as u32,
                wide_fwd,
                wide_bwd,
                g_rows: (g_at, g_n),
                g_items: 4 * pk.push(&g_items),
                n_g: (w.g_items.len() / 2) as u32,
                bg_items: 4 * pk.push(&w.bg_items),
                n_bg: (w.bg_items.len() / 3) as u32,
            });
        }
        // the chains' level records, the other levels as they are
        let mut steps = Vec::new();
        let mut l = 0;
        for c in &chains {
            steps.extend((l..c.start).map(Step::Level));
            let records: Vec<u32> = levels[c.clone()]
                .iter()
                .flat_map(|v: &Level| [v.pair_items as u32 / 4, v.n_pair_items, v.small as u32 / 4, v.n_small])
                .collect();
            let shared = levels[c.clone()].iter().map(|v| v.small_shared).max().unwrap_or(0);
            steps.push(Step::Chain { records: 4 * pk.push(&records), n: c.len() as u32, shared });
            l = c.end;
        }
        steps.extend((l..levels.len()).map(Step::Level));
        // the solve's chains: per level its gather items, its supernode, its GEMV items
        let mut solve_steps = Vec::new();
        let mut l = 0;
        for c in &solve_chains {
            solve_steps.extend((l..c.start).map(Step::Level));
            let records: Vec<u32> = c
                .clone()
                .flat_map(|i| {
                    let v = &levels[i];
                    [v.g_rows.0, v.g_rows.1, first_sn[i], v.bg_items as u32 / 4, v.n_bg]
                })
                .collect();
            let shared = levels[c.clone()].iter().map(|v| v.trsv_shared).max().unwrap_or(0);
            if shared as usize > limit {
                return Err(GpuError::Limit(format!("a solve chain needs {shared} bytes of shared memory, the device gives {limit}")));
            }
            solve_steps.push(Step::Chain { records: 4 * pk.push(&records), n: c.len() as u32, shared });
            l = c.end;
        }
        solve_steps.extend((l..levels.len()).map(Step::Level));
        let mut ints = dev.alloc(pk.words.len() * 4)?;
        dev.write(&mut ints, 0, bytemuck::cast_slice(&pk.words))?;
        let nvals = asym.val_count();
        let n_inv = plan.inv_tiles().len() / 2;
        let sec = Sections::new(plan.factor_vals * fs, plan.n * fs, nvals * 8, plan.n * 8, n_inv * plan::INV_TILE * fs);
        let work = dev.alloc(sec.bytes)?;
        // the seed's threads per stored tile: the device's lockstep width
        let seed_lanes = dev.arch().warp;
        let defines = format!("{}{}", wide.defines(), defines(F::CPP, fs, lu, seed_lanes, sizes));
        let source = format!("{WIDE}\n{SOURCE}");
        let entries = ["k_zero", "k_zero_u32", "k_seed_f32", "k_seed_wide", "k_pair_update", "k_big_update", "k_small_factor", "k_chain", "k_potf2", "k_trsm", "k_syrk", "k_fwd_gather", "k_fwd_trsv", "k_bwd_gemv", "k_bwd_trsv", "k_fwd_chain", "k_bwd_chain", "k_perm_in", "k_perm_out", "k_tile_inv", "k_fwd_trsv_wide", "k_bwd_trsv_wide", "k_big_pair"];
        let kernels = dev.compile_many(&source, &defines, &entries)?;
        let seed_f64 = if dev.arch().has_f64 { Some(dev.compile(&source, &defines, "k_seed_f64")?) } else { None };
        let [zero, zero_u32, seed_f32, seed_wide, pair_update, big_update, small_factor, chain, potf2, trsm, syrk, fwd_gather, fwd_trsv, bwd_gemv, bwd_trsv, fwd_chain, bwd_chain, perm_in, perm_out, tile_inv, fwd_trsv_wide, bwd_trsv_wide, big_pair]: [D::Kernel; 23] =
            kernels.try_into().ok().expect("a kernel per entry point");
        let k = Kernels { zero, zero_u32, seed_f32, seed_f64, seed_wide, pair_update, big_update, small_factor, chain, potf2, trsm, syrk, fwd_gather, fwd_trsv, bwd_gemv, bwd_trsv, fwd_chain, bwd_chain, perm_in, perm_out, tile_inv, fwd_trsv_wide, bwd_trsv_wide, big_pair };
        Ok(GpuSupernodal {
            dev: dev.clone(),
            k,
            ints,
            tiles: 4 * at_tiles,
            on: 4 * at_on,
            ntiles: (plan.tiles.len() / 6) as u32,
            seed_lanes,
            syrk_block,
            trsm_rows,
            n_inv: n_inv as u32,
            levels,
            steps,
            solve_steps,
            work,
            sec,
            n: plan.n,
            nb: plan.nb as u32,
            nvals,
            factor_vals: plan.factor_vals,
            wide,
            host_x: vec![0.0; plan.n],
            host_bytes: Vec::new(),
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            _f: PhantomData,
        })
    }

    /// The scalar dimension of the system.
    pub fn dim(&self) -> usize {
        self.n
    }

    /// The factor's identity, unique in the process: a graph that recorded
    /// this factor's launches replays only with it.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// The device the factor runs on.
    pub fn device(&self) -> &Arc<D> {
        &self.dev
    }

    /// Factorize the matrix whose stored tile values are `vals`, in the
    /// block storage order of the structure the plan was built from, and
    /// wait for the pivot flag.
    pub fn factorize<M: SchurReal + 'static>(&mut self, vals: &[M]) -> Result<(), FactorizeError> {
        assert_eq!(vals.len(), self.nvals, "the values of the structure the plan was built from");
        let fail = self.sec.fail;
        self.run(&self.k.zero_u32, 1, 32, 0, &[Arg::Block(&self.work, fail), Arg::U32(1)])?;
        let seed = match (&self.k.seed_f64, is_f64::<M>()) {
            (Some(seed_f64), true) => {
                self.dev.write(&mut self.work, self.sec.vals, scalar_bytes(vals))?;
                seed_f64
            }
            // a device without f64 seeds from the values rounded here
            (None, true) => {
                let rounded: Vec<f32> = bytemuck::cast_slice::<u8, f64>(scalar_bytes(vals)).iter().map(|&v| v as f32).collect();
                self.dev.write(&mut self.work, self.sec.vals, bytemuck::cast_slice(&rounded))?;
                &self.k.seed_f32
            }
            (_, false) => {
                self.dev.write(&mut self.work, self.sec.vals, scalar_bytes(vals))?;
                &self.k.seed_f32
            }
        };
        self.factorize_with((&self.work, self.sec.vals), seed, (&self.work, fail))?;
        let mut flag = [0u32];
        self.dev.read(&self.work, fail, bytemuck::cast_slice_mut(&mut flag))?;
        if flag[0] != 0 { Err(FactorizeError::NotPositiveDefinite) } else { Ok(()) }
    }

    /// [`factorize`](Self::factorize) from tile values in the wide scalar
    /// already on the device at `vals` (block, byte offset), without
    /// waiting: the values are cast to `F` as they land in the panels, and a
    /// pivot that is not positive sets the `u32` at `fail` nonzero (the
    /// caller zeroes it, and reads it with its result).
    pub fn factorize_device(&self, vals: (&D::Block, usize), fail: (&D::Block, usize)) -> Result<(), GpuError> {
        self.factorize_with(vals, &self.k.seed_wide, fail)
    }

    fn factorize_with(&self, vals: (&D::Block, usize), seed: &D::Kernel, fail: (&D::Block, usize)) -> Result<(), GpuError> {
        let plan = Arg::Block(&self.ints, 0);
        let list = |at: usize| Arg::Block(&self.ints, at);
        let factor = Arg::Block(&self.work, self.sec.factor);
        let fail = Arg::Block(fail.0, fail.1);
        let flen = self.factor_vals as u32;
        self.run(&self.k.zero, flen.div_ceil(THREADS).max(1), THREADS, 0, &[factor, Arg::U32(flen)])?;
        let seed_args = [list(self.tiles), Arg::Block(vals.0, vals.1), factor, Arg::U32(self.ntiles)];
        self.run(seed, self.ntiles.max(1), self.seed_lanes, 0, &seed_args)?;
        for step in &self.steps {
            let l = match *step {
                Step::Chain { records, n, shared } => {
                    self.run(&self.k.chain, 1, THREADS, shared, &[list(records), Arg::U32(n), plan, factor, fail])?;
                    continue;
                }
                Step::Level(i) => &self.levels[i],
            };
            if l.n_pair_items > 0 {
                let args = [list(l.pair_items), plan, factor, factor];
                self.run(&self.k.pair_update, l.n_pair_items, THREADS, 0, &args)?;
            }
            if l.n_big_items > 0 {
                let args = [list(l.big_items), list(l.big_rects), plan, factor];
                self.run(&self.k.big_update, l.n_big_items, THREADS, 0, &args)?;
            }
            if l.n_pair_tiles > 0 {
                self.run(&self.k.big_pair, l.n_pair_tiles, THREADS, 0, &[list(l.pair_tiles), plan, factor])?;
            }
            if l.n_small > 0 {
                self.run(&self.k.small_factor, l.n_small, l.small_threads, l.small_shared, &[list(l.small), plan, factor, fail])?;
            }
            // the first step's diagonal tiles; each later one is factored by
            // the syrk of the step before it
            if !l.trsm_grid.is_empty() {
                self.run(&self.k.potf2, l.n_big, NB as u32, 0, &[list(l.big), plan, factor, fail])?;
            }
            // With a lookahead, a block's last step updates the next block's
            // columns, forks, and updates the rest on the main stream while
            // the next block's steps run on the side stream; they write the
            // next block's columns only, the rest the columns past them. The
            // join comes before the next block's last update, and after a
            // failure while forked, so the next work does not run alongside
            // the side's.
            let mut lane = Lane::Main;
            let stepped = (|| -> Result<(), GpuError> {
                for (step, (&tg, &sg)) in l.trsm_grid.iter().zip(&l.syrk_grid).enumerate() {
                    let j = Arg::U32((step * NB) as u32);
                    let args = [list(l.big), Arg::U32(l.n_big), j, plan, factor];
                    if tg > 0 {
                        self.run_on(lane, &self.k.trsm, tg, self.trsm_rows, 0, &args)?;
                    }
                    let (k0, kn, cend) = plan::syrk_update(step, self.syrk_block as usize);
                    // the update of columns [k1, cend) from cbeg on
                    let syrk = |lane: Lane, grid: u32, cend: u32, cbeg: u32| -> Result<(), GpuError> {
                        if grid == 0 {
                            return Ok(());
                        }
                        let args = [list(l.big), Arg::U32(l.n_big), Arg::U32(k0), Arg::U32(kn), Arg::U32(cend), Arg::U32(cbeg), plan, factor, fail];
                        self.run_on(lane, &self.k.syrk, grid, THREADS, 0, &args)
                    };
                    if cend == u32::MAX {
                        if lane == Lane::Side {
                            self.dev.join()?;
                            lane = Lane::Main;
                        }
                        if l.syrk_far_grid[step] > 0 {
                            let split = plan::lookahead_split(step, self.syrk_block as usize).expect("a block's last step splits");
                            syrk(Lane::Main, l.syrk_near_grid[step], split, 0)?;
                            self.dev.fork()?;
                            lane = Lane::Side;
                            syrk(Lane::Main, l.syrk_far_grid[step], u32::MAX, split)?;
                            continue;
                        }
                    }
                    syrk(lane, sg, cend, 0)?;
                }
                Ok(())
            })();
            let joined = if lane == Lane::Side { self.dev.join() } else { Ok(()) };
            // the steps' error first: the join's, if any, follows from it
            stepped.and(joined)?;
        }
        // the solve's inverted diagonal tiles
        if self.n_inv > 0 {
            self.run(&self.k.tile_inv, self.n_inv, 32, 0, &[plan, factor, Arg::Block(&self.work, self.sec.inv)])?;
        }
        Ok(())
    }

    /// Solve `A x = rhs` in place through the last factorization, and wait
    /// for `x`; `rhs` is in the matrix's own ordering and comes back as `x`.
    pub fn solve<M: SchurReal + 'static>(&mut self, rhs: &mut [M]) -> Result<(), GpuError> {
        assert_eq!(rhs.len(), self.n);
        let at = self.sec.rhs;
        for (dst, &v) in self.host_x.iter_mut().zip(rhs.iter()) {
            *dst = v.to_f64();
        }
        self.host_bytes.clear();
        self.wide.encode_into(&self.host_x, &mut self.host_bytes);
        self.dev.write(&mut self.work, at, &self.host_bytes)?;
        self.solve_with((&self.work, at), (&self.ints, self.on))?;
        self.dev.read(&self.work, at, &mut self.host_bytes)?;
        self.wide.decode(&self.host_bytes, &mut self.host_x);
        for (dst, &v) in rhs.iter_mut().zip(&self.host_x) {
            *dst = M::from_f64(v);
        }
        Ok(())
    }

    /// [`solve`](Self::solve) on a vector already on the device at `x`
    /// (block, byte offset), `f64` in the matrix's ordering, in place,
    /// without waiting: the permutation into the factor's ordering and
    /// scalar and back, and the level-set solve between.
    pub fn solve_device(&self, x: (&D::Block, usize)) -> Result<(), GpuError> {
        self.solve_with(x, (&self.ints, self.on))
    }

    /// [`solve_device`](Self::solve_device) when the `u32` at `gate` is
    /// nonzero, as the device finds it; otherwise its kernels do nothing.
    pub fn solve_device_if(&self, x: (&D::Block, usize), gate: (&D::Block, usize)) -> Result<(), GpuError> {
        self.solve_with(x, gate)
    }

    fn solve_with(&self, x: (&D::Block, usize), gate: (&D::Block, usize)) -> Result<(), GpuError> {
        let gate = Arg::Block(gate.0, gate.1);
        let plan = Arg::Block(&self.ints, 0);
        let list = |at: usize| Arg::Block(&self.ints, at);
        let factor = Arg::Block(&self.work, self.sec.factor);
        let inv = Arg::Block(&self.work, self.sec.inv);
        let xb = Arg::Block(&self.work, self.sec.xb);
        let x = Arg::Block(x.0, x.1);
        let blocks = self.nb.div_ceil(THREADS).max(1);
        self.run(&self.k.perm_in, blocks, THREADS, 0, &[gate, plan, x, xb, Arg::U32(self.nb)])?;
        for step in &self.solve_steps {
            let l = match *step {
                Step::Chain { records, n, shared } => {
                    self.run(&self.k.fwd_chain, 1, THREADS, shared, &[gate, list(records), Arg::U32(n), plan, factor, inv, xb])?;
                    continue;
                }
                Step::Level(i) => &self.levels[i],
            };
            if l.n_g > 0 {
                self.run(&self.k.fwd_gather, l.n_g, THREADS, 0, &[gate, list(l.g_items), plan, factor, xb, xb])?;
            }
            if l.n_narrow > 0 {
                self.run(&self.k.fwd_trsv, l.n_narrow, THREADS, l.narrow_shared, &[gate, list(l.narrow), plan, factor, inv, xb])?;
            }
            for &(items, n) in &l.wide_fwd {
                self.run(&self.k.fwd_trsv_wide, n, THREADS, 0, &[gate, list(items), plan, factor, inv, xb])?;
            }
        }
        for step in self.solve_steps.iter().rev() {
            let l = match *step {
                Step::Chain { records, n, shared } => {
                    self.run(&self.k.bwd_chain, 1, THREADS, shared, &[gate, list(records), Arg::U32(n), plan, factor, inv, xb])?;
                    continue;
                }
                Step::Level(i) => &self.levels[i],
            };
            if l.n_bg > 0 {
                self.run(&self.k.bwd_gemv, l.n_bg, THREADS, 0, &[gate, list(l.bg_items), plan, factor, xb])?;
            }
            for &(items, n) in &l.wide_bwd {
                self.run(&self.k.bwd_trsv_wide, n, THREADS, 0, &[gate, list(items), plan, factor, inv, xb])?;
            }
            if l.n_narrow > 0 {
                self.run(&self.k.bwd_trsv, l.n_narrow, THREADS, l.narrow_shared, &[gate, list(l.narrow), plan, factor, inv, xb])?;
            }
        }
        self.run(&self.k.perm_out, blocks, THREADS, 0, &[gate, plan, xb, x, Arg::U32(self.nb)])
    }

    /// The factor's values, panel by panel in the symbolic's layout, read
    /// back from the device (for tests and inspection).
    pub fn factor_values(&self) -> Result<Vec<F>, GpuError> {
        let mut v = vec![F::zeroed(); self.factor_vals];
        self.dev.read(&self.work, self.sec.factor, bytemuck::cast_slice_mut(&mut v))?;
        Ok(v)
    }

    /// The inverted diagonal tiles as `k_tile_inv` wrote them.
    #[cfg(test)]
    fn inv_values(&self) -> Result<Vec<F>, GpuError> {
        let mut v = vec![F::zeroed(); self.n_inv as usize * plan::INV_TILE];
        self.dev.read(&self.work, self.sec.inv, bytemuck::cast_slice_mut(&mut v))?;
        Ok(v)
    }

    /// The levels the factorization runs in chains.
    #[cfg(test)]
    fn chained_levels(&self) -> usize {
        self.steps.iter().map(|s| if let Step::Chain { n, .. } = s { *n as usize } else { 0 }).sum()
    }

    /// The levels the solve runs in chains.
    #[cfg(test)]
    fn solve_chained_levels(&self) -> usize {
        self.solve_steps.iter().map(|s| if let Step::Chain { n, .. } = s { *n as usize } else { 0 }).sum()
    }

    /// The levels whose trsvs run over many blocks.
    #[cfg(test)]
    fn wide_levels(&self) -> usize {
        self.levels.iter().filter(|l| !l.wide_fwd.is_empty()).count()
    }

    /// The levels of one supernode.
    #[cfg(test)]
    fn single_levels(&self) -> usize {
        self.levels.iter().filter(|l| l.n_all == 1).count()
    }

    /// The levels of one small supernode and no big updates.
    #[cfg(test)]
    fn path_levels(&self) -> usize {
        self.levels.iter().filter(|l| l.n_small == 1 && l.n_big == 0 && l.n_big_items == 0 && l.n_pair_tiles == 0).count()
    }

    fn run(&self, kernel: &D::Kernel, blocks: u32, threads: u32, shared_bytes: u32, args: &[Arg<'_, D::Block>]) -> Result<(), GpuError> {
        self.run_on(Lane::Main, kernel, blocks, threads, shared_bytes, args)
    }

    fn run_on(&self, lane: Lane, kernel: &D::Kernel, blocks: u32, threads: u32, shared_bytes: u32, args: &[Arg<'_, D::Block>]) -> Result<(), GpuError> {
        let launch = Launch { grid: [blocks, 1, 1], block: [threads, 1, 1], shared_bytes };
        self.dev.launch_on(lane, kernel, launch, args)
    }
}

/// A [`GpuSupernodal`] of either scalar behind one type, for a caller
/// whose own scalar is generic.
pub enum GpuFactor<D: Device> {
    F32(GpuSupernodal<f32, D>),
    F64(GpuSupernodal<f64, D>),
}

impl<D: Device> GpuFactor<D> {
    /// As [`GpuSupernodal::new`], in `precision`.
    pub fn new(
        dev: &Arc<D>,
        precision: crate::GpuPrecision,
        sn: &SupernodalSymbolic,
        asym: &SymbolicSparseBlockColMat<SparseIndex>,
        params: &GpuSupernodalParams,
        wide: Wide,
    ) -> Result<Self, GpuError> {
        Ok(match precision {
            crate::GpuPrecision::F32 => GpuFactor::F32(GpuSupernodal::new(dev, sn, asym, params, wide)?),
            crate::GpuPrecision::F64 => GpuFactor::F64(GpuSupernodal::new(dev, sn, asym, params, wide)?),
        })
    }

    /// As [`GpuSupernodal::factorize`].
    pub fn factorize<M: SchurReal + 'static>(&mut self, vals: &[M]) -> Result<(), FactorizeError> {
        match self {
            GpuFactor::F32(f) => f.factorize(vals),
            GpuFactor::F64(f) => f.factorize(vals),
        }
    }

    /// As [`GpuSupernodal::solve`].
    pub fn solve<M: SchurReal + 'static>(&mut self, rhs: &mut [M]) -> Result<(), GpuError> {
        match self {
            GpuFactor::F32(f) => f.solve(rhs),
            GpuFactor::F64(f) => f.solve(rhs),
        }
    }

    /// The factor's scalar.
    pub fn precision(&self) -> crate::GpuPrecision {
        match self {
            GpuFactor::F32(_) => crate::GpuPrecision::F32,
            GpuFactor::F64(_) => crate::GpuPrecision::F64,
        }
    }

    /// As [`GpuSupernodal::factorize_device`].
    pub fn factorize_device(&self, vals: (&D::Block, usize), fail: (&D::Block, usize)) -> Result<(), GpuError> {
        match self {
            GpuFactor::F32(f) => f.factorize_device(vals, fail),
            GpuFactor::F64(f) => f.factorize_device(vals, fail),
        }
    }

    /// As [`GpuSupernodal::solve_device`].
    pub fn solve_device(&self, x: (&D::Block, usize)) -> Result<(), GpuError> {
        match self {
            GpuFactor::F32(f) => f.solve_device(x),
            GpuFactor::F64(f) => f.solve_device(x),
        }
    }

    /// As [`GpuSupernodal::solve_device_if`].
    pub fn solve_device_if(&self, x: (&D::Block, usize), gate: (&D::Block, usize)) -> Result<(), GpuError> {
        match self {
            GpuFactor::F32(f) => f.solve_device_if(x, gate),
            GpuFactor::F64(f) => f.solve_device_if(x, gate),
        }
    }

    /// As [`GpuSupernodal::id`].
    pub fn id(&self) -> u64 {
        match self {
            GpuFactor::F32(f) => f.id(),
            GpuFactor::F64(f) => f.id(),
        }
    }

    /// As [`GpuSupernodal::device`].
    pub fn device(&self) -> &Arc<D> {
        match self {
            GpuFactor::F32(f) => f.device(),
            GpuFactor::F64(f) => f.device(),
        }
    }

    /// As [`GpuSupernodal::dim`].
    pub fn dim(&self) -> usize {
        match self {
            GpuFactor::F32(f) => f.dim(),
            GpuFactor::F64(f) => f.dim(),
        }
    }
}

/// The kernel sizes the device chooses ([`GpuSupernodalParams`]'s
/// `*_on`), compiled into `supernodal.cpp`.
#[derive(Clone, Copy)]
struct DeviceSizes {
    syrk_block: u32,
    syrk_tile: u32,
    trsm_rows: u32,
    potf2_unroll: bool,
    trsv_block: u32,
}

/// The `#define`s of `supernodal.cpp`: the scalar, the lane unroll, the
/// block sizes, the device's sizes, and the tile sizes and plan header
/// slots of [`crate::plan`].
fn defines(scalar: &str, scalar_bytes: usize, lu: usize, seed_lanes: u32, sizes: DeviceSizes) -> String {
    use plan::*;
    let DeviceSizes { syrk_block, syrk_tile, trsm_rows, potf2_unroll, trsv_block } = sizes;
    let mut s = format!(
        "#define SCALAR {scalar}\n#define SCALAR_BYTES {scalar_bytes}u\n#define LU {lu}u\n#define THREADS {THREADS}u\n#define SEED_LANES {seed_lanes}u\n\
         #define SYRK_BLOCK {syrk_block}u\n#define SYRK_TB {syrk_tile}u\n#define TRSM_ROWS {trsm_rows}u\n#define POTF2_UNROLL {}\n#define TRSV_BLOCK {trsv_block}u\n",
        potf2_unroll as u32
    );
    let consts = [
        ("NB", NB),
        ("TB", TB),
        ("TK", TK),
        ("BIG_ITEM", plan::BIG_ITEM),
        ("PAIR_TILE", plan::PAIR_TILE),
        ("PAIR_ITEM", plan::PAIR_ITEM),
        ("PAIR_NI", PAIR_NI),
        ("PAIR_NJ", PAIR_NJ),
        ("PAIR_EPT", PAIR_EPT),
        ("PAIR_KC", PAIR_KC),
        ("XR_CH", XR_CH),
        ("H_DIMS", H_DIMS),
        ("H_VAL_PTR", H_VAL_PTR),
        ("H_SUP_BEGIN", H_SUP_BEGIN),
        ("H_SC_START", H_SC_START),
        ("H_PAIR_PTR", H_PAIR_PTR),
        ("H_PR_D", H_PR_D),
        ("H_PR_R0", H_PR_R0),
        ("H_PR_RMAP", H_PR_RMAP),
        ("H_PR_CMAP", H_PR_CMAP),
        ("H_RMAP", H_RMAP),
        ("H_CMAP", H_CMAP),
        ("H_ROW_SC", H_ROW_SC),
        ("H_ROW_SC_OFF", H_ROW_SC_OFF),
        ("H_ORDER", H_ORDER),
        ("H_BLK_WIDTH", H_BLK_WIDTH),
        ("H_PART", H_PART),
        ("H_SOLVE_HEAD", H_SOLVE_HEAD),
        ("H_INV_TILES", H_INV_TILES),
        ("INV_TILE", INV_TILE),
        ("TRSV_ITEM", TRSV_ITEM),
        ("SOLVE_HEAD", plan::SOLVE_HEAD),
        ("GATHER_ROW", plan::GATHER_ROW),
    ];
    for (name, v) in consts {
        writeln!(s, "#define {name} {v}u").unwrap();
    }
    s
}

#[cfg(all(test, any(feature = "cuda", feature = "hip", feature = "cpu", all(feature = "metal", target_vendor = "apple"))))]
mod tests {
    use super::*;
    use crate::TestDevice;
    use crate::test_util::{grid_spd, grid_spd_sparse, has_f64, rel_residual, wide};
    use arael_faer::bsc::SparseBlockColMat;
    use arael_faer::supernodal::{SupernodalContext, SupernodalParams, nd_block_order, supernodal_factorize, supernodal_solve};

    type Matrix = SparseBlockColMat<SparseIndex, f64>;

    fn device() -> Arc<TestDevice> {
        Arc::new(TestDevice::new(0).unwrap())
    }

    fn setup(k: usize) -> (Matrix, Vec<f64>, Vec<f64>, SupernodalSymbolic) {
        let (a, dense, rhs) = grid_spd(k, &[3, 6], 11);
        let order = nd_block_order(a.symbolic());
        let sn = SupernodalSymbolic::new(a.symbolic(), Some(&order), &SupernodalParams::default()).unwrap();
        (a, dense, rhs, sn)
    }

    /// The parameters that put every kernel to work on a small grid.
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
            trsv_wide: None,
            trsv_block: None,
            pair_tiles: None,
            lookahead: None,
        }
    }

    /// Every supernode on the panel steps, however narrow: what the
    /// `small_flops` bound sends a tall one to.
    fn panel_steps_only() -> GpuSupernodalParams {
        GpuSupernodalParams { small_flops: 0.0, ..GpuSupernodalParams::default() }
    }

    /// The panel steps with `k_syrk` in blocks of NB and of 128 columns:
    /// both on every device, whichever it takes by default.
    fn both_syrks() -> [GpuSupernodalParams; 2] {
        [NB as u32, 128].map(|b| GpuSupernodalParams { syrk_block: Some(b), ..panel_steps_only() })
    }

    /// The big updates cut by rectangles and along k into shared tiles, as
    /// on a device far larger than the grid's levels.
    fn cut_updates() -> GpuSupernodalParams {
        GpuSupernodalParams { blocks_per_sm: 1000, ..every_kernel_busy() }
    }

    /// The big updates in target tiles and in pair tiles, cut as
    /// `cut_updates`: both on every device, whichever it takes by default.
    fn both_big_updates() -> [GpuSupernodalParams; 2] {
        [false, true].map(|p| GpuSupernodalParams { pair_tiles: Some(p), ..cut_updates() })
    }

    /// Every supernode the small kernel holds on it, however wide or tall.
    fn small_kernel_only() -> GpuSupernodalParams {
        GpuSupernodalParams { small_q: u32::MAX, small_flops: f64::INFINITY, ..GpuSupernodalParams::default() }
    }

    /// Every level the chain kernel can take in chains, in either precision:
    /// every update small, every supernode the small kernel holds, no bound
    /// on a level's work.
    fn chains_everywhere() -> GpuSupernodalParams {
        GpuSupernodalParams {
            chain_flops_f32: f64::INFINITY,
            chain_flops_f64: f64::INFINITY,
            chain_items: u32::MAX,
            solve_chain_work: f64::INFINITY,
            big_pair_flops: f64::INFINITY,
            wide_pair_flops: f64::INFINITY,
            ..small_kernel_only()
        }
    }

    /// The widest q the small kernel takes in `F` on `dev`, as `new` caps it.
    fn small_kernel_widest<F>(dev: &TestDevice) -> usize {
        let fs = core::mem::size_of::<F>();
        (1..=THREADS as usize).take_while(|&q| small_shared_bytes(q, fs) <= dev.arch().shared_bytes as usize).last().unwrap()
    }

    /// The factor's lower parts against a reference, as err / max.
    fn factor_err(sn: &SupernodalSymbolic, got: &[f64], reference: &[f64]) -> f64 {
        let (mut mx, mut df) = (0.0f64, 0.0f64);
        for s in 0..sn.n_supernodes() {
            let (q, h) = sn.supernode_dims(s);
            let base = sn.panel_range(s).start;
            for c in 0..q {
                for r in c..h {
                    mx = mx.max(reference[base + c * h + r].abs());
                    df = df.max((got[base + c * h + r] - reference[base + c * h + r]).abs());
                }
            }
        }
        df / mx
    }

    fn widen<F: Scalar>(v: &[F]) -> Vec<f64> {
        v.iter().map(|&x| x.to_f64()).collect()
    }

    #[test]
    fn f64_factor_and_solve_match_the_cpu() {
        let (a, dense, rhs, sn) = setup(24);
        let n = sn.dim();
        let mut cpu = vec![0.0f64; sn.factor_val_count()];
        let mut ctx = SupernodalContext::new();
        supernodal_factorize(&sn, &a, &mut cpu, &mut ctx, 1).unwrap();
        let mut x_cpu = rhs.clone();
        supernodal_solve(&sn, &cpu, &mut x_cpu, &mut ctx, 1);
        let dev = device();
        if !has_f64(&*dev, "the f64 factor") {
            return;
        }
        // the cut parameters do cut: some target tile is shared between
        // items, some pair tile cut along k
        let [targets, pairs] = both_big_updates().map(|kn| plan::build_levels(&Plan::new(&sn, a.symbolic()), &kn, dev.arch(), 8));
        assert!(targets.iter().any(|w| w.big_items.chunks(plan::BIG_ITEM).any(|it| it[7] == 1)), "a tile cut between items");
        assert!(pairs.iter().any(|w| w.pair_tiles.chunks(plan::PAIR_TILE).any(|it| it[7] != 0)), "a pair tile cut along k");
        for params in [GpuSupernodalParams::default(), every_kernel_busy(), cut_updates(), small_kernel_only(), chains_everywhere()].into_iter().chain(both_syrks()).chain(both_big_updates()) {
            let mut g = GpuSupernodal::<f64, _>::new(&dev, &sn, a.symbolic(), &params, Wide::F64).unwrap();
            assert!(sn.n_supernodes() > 20, "a tree with levels");
            if params.chain_flops_f64 == f64::INFINITY {
                assert!(g.chained_levels() > 0 && g.chained_levels() == g.path_levels(), "the levels of one small supernode in chains");
            }
            if params.solve_chain_work == f64::INFINITY {
                assert!(g.solve_chained_levels() > 0 && g.solve_chained_levels() == g.single_levels(), "the levels of one supernode in solve chains");
            }
            g.factorize(a.vals()).unwrap();
            let err = factor_err(&sn, &g.factor_values().unwrap(), &cpu);
            assert!(err < 1e-12, "factor err/max {err}");
            let mut x = rhs.clone();
            g.solve(&mut x).unwrap();
            let res = rel_residual(&dense, n, &x, &rhs);
            assert!(res < 1e-10, "residual {res}");
            let dx = x.iter().zip(&x_cpu).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            assert!(dx < 1e-9, "solution differs from the CPU's by {dx}");
            // a second factorization of the same structure reuses the plan
            g.factorize(a.vals()).unwrap();
            let mut x2 = rhs.clone();
            g.solve(&mut x2).unwrap();
            assert!(rel_residual(&dense, n, &x2, &rhs) < 1e-10);
        }
    }

    /// An f32 factor is the CPU's to single precision under every kernel
    /// route (on Metal, the tiles of `k_syrk` and `k_big_pair` on SIMD-group
    /// matrices), solves to single precision, and refines.
    #[test]
    fn f32_factor_solves_to_single_precision_and_refines() {
        let (a, dense, rhs, sn) = setup(24);
        let n = sn.dim();
        let mut cpu = vec![0.0f64; sn.factor_val_count()];
        supernodal_factorize(&sn, &a, &mut cpu, &mut SupernodalContext::new(), 1).unwrap();
        let dev = device();
        // the pair tiles run: diagonal ones, others, and some cut along k
        let pairs = plan::build_levels(&Plan::new(&sn, a.symbolic()), &both_big_updates()[1], dev.arch(), 4);
        let tiles: Vec<&[u32]> = pairs.iter().flat_map(|w| w.pair_tiles.chunks(plan::PAIR_TILE)).collect();
        assert!(tiles.iter().any(|it| it[2] == it[3]) && tiles.iter().any(|it| it[2] != it[3]), "diagonal and other pair tiles");
        assert!(tiles.iter().any(|it| it[7] != 0), "a pair tile cut along k");
        for params in [GpuSupernodalParams::default(), every_kernel_busy(), cut_updates(), small_kernel_only(), chains_everywhere()].into_iter().chain(both_syrks()).chain(both_big_updates()) {
            let mut g = GpuSupernodal::<f32, _>::new(&dev, &sn, a.symbolic(), &params, wide(&*dev)).unwrap();
            if params.chain_flops_f32 == f64::INFINITY {
                assert!(g.chained_levels() > 0 && g.chained_levels() == g.path_levels(), "the levels of one small supernode in chains");
            }
            g.factorize(a.vals()).unwrap();
            let err = factor_err(&sn, &widen(&g.factor_values().unwrap()), &cpu);
            assert!(err < 1e-5, "f32 factor err/max {err}, {params:?}");
            let mut x = rhs.clone();
            g.solve(&mut x).unwrap();
            let res0 = rel_residual(&dense, n, &x, &rhs);
            assert!(res0 < 1e-4 && res0 > 1e-9, "an f32 solve: residual {res0}");
            // one refinement step against the f64 matrix
            let mut r = vec![0.0f64; n];
            for i in 0..n {
                let mut ax = 0.0;
                for c in 0..n {
                    ax += dense[i + c * n] * x[c];
                }
                r[i] = rhs[i] - ax;
            }
            g.solve(&mut r).unwrap();
            for i in 0..n {
                x[i] += r[i];
            }
            let res1 = rel_residual(&dense, n, &x, &rhs);
            assert!(res1 < res0 * 1e-3, "refinement {res0} -> {res1}");
        }
    }

    /// A supernode too wide for the small kernel's shared memory takes the
    /// panel steps, whatever `small_q` says.
    #[test]
    fn a_supernode_beyond_the_shared_memory_takes_the_panel_steps() {
        let (a, dense, rhs, sn) = setup(32);
        let dev = device();
        fn check<F: Scalar>(dev: &Arc<TestDevice>, a: &Matrix, dense: &[f64], rhs: &[f64], sn: &SupernodalSymbolic, tol: f64) {
            let fs = core::mem::size_of::<F>();
            let widest = (0..sn.n_supernodes()).map(|s| sn.supernode_dims(s).0).max().unwrap();
            assert!(small_shared_bytes(widest, fs) > dev.arch().shared_bytes as usize, "a supernode of {widest} columns fits the device");
            let params = GpuSupernodalParams { small_q: 100_000, small_flops: f64::INFINITY, ..GpuSupernodalParams::default() };
            let mut g = GpuSupernodal::<F, _>::new(dev, sn, a.symbolic(), &params, wide(&**dev)).unwrap();
            g.factorize(a.vals()).unwrap();
            let mut x = rhs.to_vec();
            g.solve(&mut x).unwrap();
            let res = rel_residual(dense, sn.dim(), &x, rhs);
            assert!(res < tol, "{}: residual {res}", F::CPP);
        }
        if has_f64(&*dev, "the f64 factor") {
            check::<f64>(&dev, &a, &dense, &rhs, &sn, 1e-10);
        } else {
            check::<f32>(&dev, &a, &dense, &rhs, &sn, 1e-4);
        }
    }

    /// The kernels' SYRK_BLOCK, SYRK_TB, TRSM_ROWS, POTF2_UNROLL and
    /// TRSV_BLOCK are the sizes asked for (`defines` brings `plan`'s names into scope, which
    /// once shadowed its flag).
    #[test]
    fn the_defines_carry_the_device_sizes() {
        let base = DeviceSizes { syrk_block: 32, syrk_tile: 64, trsm_rows: 64, potf2_unroll: false, trsv_block: 128 };
        let d = |sizes| defines("float", 4, 1, 32, sizes);
        assert!(d(base).contains("#define SYRK_BLOCK 32u\n"));
        assert!(d(DeviceSizes { syrk_block: 256, ..base }).contains("#define SYRK_BLOCK 256u\n"));
        assert!(d(DeviceSizes { syrk_tile: 32, ..base }).contains("#define SYRK_TB 32u\n"));
        assert!(d(base).contains("#define SYRK_TB 64u\n"));
        assert!(d(DeviceSizes { trsm_rows: 32, ..base }).contains("#define TRSM_ROWS 32u\n"));
        assert!(d(DeviceSizes { trsm_rows: 128, ..base }).contains("#define TRSM_ROWS 128u\n"));
        assert!(d(base).contains("#define POTF2_UNROLL 0\n"));
        assert!(d(DeviceSizes { potf2_unroll: true, ..base }).contains("#define POTF2_UNROLL 1\n"));
        assert!(d(DeviceSizes { trsv_block: 256, ..base }).contains("#define TRSV_BLOCK 256u\n"));
    }

    /// The trsvs over many blocks (every supernode wider than 32 columns,
    /// in blocks of 32, 64 and 128 columns, widths no multiple of the
    /// block) solve as the one-block trsvs do: in f64 to rounding, and with
    /// an f32 factor to single precision.
    #[test]
    fn wide_trsvs_solve_as_one_block_trsvs() {
        let (a, dense, rhs) = grid_spd(24, &[3, 6], 11);
        let order = nd_block_order(a.symbolic());
        let sp = SupernodalParams { relax: Some(vec![(320, 1.0)]), ..SupernodalParams::default() };
        let sn = SupernodalSymbolic::new(a.symbolic(), Some(&order), &sp).unwrap();
        let n = sn.dim();
        let dev = device();
        let f64s = has_f64(&*dev, "the f64 factor");
        let one = GpuSupernodalParams { trsv_wide: Some(u32::MAX), solve_chain_work: 0.0, ..GpuSupernodalParams::default() };
        let mut x1 = rhs.clone();
        if f64s {
            let mut g = GpuSupernodal::<f64, _>::new(&dev, &sn, a.symbolic(), &one, Wide::F64).unwrap();
            assert_eq!(g.wide_levels(), 0);
            g.factorize(a.vals()).unwrap();
            g.solve(&mut x1).unwrap();
        }
        let max = x1.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        for block in [32, 64, 128] {
            let wide_trsv = GpuSupernodalParams { trsv_wide: Some(33), trsv_block: Some(block), ..one };
            if f64s {
                let mut g = GpuSupernodal::<f64, _>::new(&dev, &sn, a.symbolic(), &wide_trsv, Wide::F64).unwrap();
                assert!(g.wide_levels() > 2, "levels over many blocks: {}", g.wide_levels());
                g.factorize(a.vals()).unwrap();
                let mut x = rhs.clone();
                g.solve(&mut x).unwrap();
                let diff = x.iter().zip(&x1).fold(0.0f64, |m, (u, v)| m.max((u - v).abs())) / max;
                assert!(diff < 1e-12, "blocks of {block}: the solution off the one-block one by {diff}");
                let res = rel_residual(&dense, n, &x, &rhs);
                assert!(res < 1e-10, "blocks of {block}: residual {res}");
            }
            let mut g32 = GpuSupernodal::<f32, _>::new(&dev, &sn, a.symbolic(), &wide_trsv, wide(&*dev)).unwrap();
            assert!(g32.wide_levels() > 2, "levels over many blocks: {}", g32.wide_levels());
            g32.factorize(a.vals()).unwrap();
            let mut x = rhs.clone();
            g32.solve(&mut x).unwrap();
            let res = rel_residual(&dense, n, &x, &rhs);
            assert!(res < 1e-4, "blocks of {block}, an f32 factor: residual {res}");
        }
    }

    /// Supernodes wider than two blocks of 128 columns, with rows below them
    /// and widths no multiple of NB: the factor of `k_syrk` in tiles of 32
    /// and 64, each in blocks of NB, 64, 128 and 256 columns
    /// (`plan::syrk_update`), with the lookahead and without, and of
    /// `k_trsm` in blocks of 32, 64 and 128 rows is the CPU's, in f64, and
    /// an f32 factor solves to single precision.
    #[test]
    fn wide_supernodes_with_rows_below_factor_as_on_the_cpu() {
        let (a, dense, rhs) = grid_spd(24, &[3, 6], 11);
        let order = nd_block_order(a.symbolic());
        let sp = SupernodalParams { relax: Some(vec![(320, 1.0)]), ..SupernodalParams::default() };
        let sn = SupernodalSymbolic::new(a.symbolic(), Some(&order), &sp).unwrap();
        let wide_sn = (0..sn.n_supernodes()).map(|s| sn.supernode_dims(s)).filter(|&(q, h)| q > 256 && h > q && q % plan::NB != 0).count();
        assert!(wide_sn >= 2, "wide supernodes with rows below: {wide_sn}");
        let mut cpu = vec![0.0f64; sn.factor_val_count()];
        let mut ctx = SupernodalContext::new();
        supernodal_factorize(&sn, &a, &mut cpu, &mut ctx, 1).unwrap();
        let dev = device();
        let blocks = [32, 64].into_iter().flat_map(|t| {
            [NB as u32, 64, 128, 256].into_iter().flat_map(move |b| {
                [false, true].map(|la| GpuSupernodalParams { syrk_block: Some(b), syrk_tile: Some(t), lookahead: Some(la), ..GpuSupernodalParams::default() })
            })
        });
        let rows = [32, 64, 128].map(|r| GpuSupernodalParams { trsm_rows: Some(r), ..GpuSupernodalParams::default() });
        let unrolls = [false, true].map(|u| GpuSupernodalParams { potf2_unroll: Some(u), ..GpuSupernodalParams::default() });
        let f64s = has_f64(&*dev, "the f64 factor");
        for params in blocks.chain(rows).chain(unrolls).chain(both_syrks()) {
            let sizes = (params.syrk_block, params.syrk_tile, params.trsm_rows, params.potf2_unroll, params.lookahead);
            let widest = (0..sn.n_supernodes()).map(|s| sn.supernode_dims(s).0).max().unwrap();
            if params.lookahead == Some(true) && widest > 2 * params.syrk_block.unwrap() as usize {
                // the lookahead splits some block's last update: there are
                // columns past the next block
                let g = GpuSupernodal::<f32, _>::new(&dev, &sn, a.symbolic(), &params, wide(&*dev)).unwrap();
                assert!(g.levels.iter().any(|l| l.syrk_far_grid.iter().any(|&n| n > 0)), "no update split by the lookahead, {sizes:?}");
            }
            if f64s {
                let mut g = GpuSupernodal::<f64, _>::new(&dev, &sn, a.symbolic(), &params, Wide::F64).unwrap();
                g.factorize(a.vals()).unwrap();
                let err = factor_err(&sn, &g.factor_values().unwrap(), &cpu);
                assert!(err < 1e-12, "factor err/max {err}, syrk block, syrk tile, trsm rows, potf2 unroll {sizes:?}");
                let mut x = rhs.clone();
                g.solve(&mut x).unwrap();
                let res = rel_residual(&dense, sn.dim(), &x, &rhs);
                assert!(res < 1e-10, "residual {res}");
            }
            let mut g32 = GpuSupernodal::<f32, _>::new(&dev, &sn, a.symbolic(), &params, wide(&*dev)).unwrap();
            g32.factorize(a.vals()).unwrap();
            let err = factor_err(&sn, &widen(&g32.factor_values().unwrap()), &cpu);
            assert!(err < 1e-5, "f32 factor err/max {err}, syrk block, syrk tile, trsm rows, potf2 unroll {sizes:?}");
            let mut x = rhs.clone();
            g32.solve(&mut x).unwrap();
            let res = rel_residual(&dense, sn.dim(), &x, &rhs);
            assert!(res < 1e-4, "an f32 solve: residual {res}");
        }
    }

    /// `k_tile_inv` writes the inverse X of every 32 x 32 diagonal tile L
    /// of the supernodes wider than NB twice, column-major and row-major:
    /// L X = I to rounding, and X is zero past a partial tile's columns, in
    /// f64 and f32.
    #[test]
    fn the_inverted_tiles_invert_the_diagonal_tiles() {
        let (a, _, _) = grid_spd(24, &[3, 6], 11);
        let order = nd_block_order(a.symbolic());
        let sp = SupernodalParams { relax: Some(vec![(320, 1.0)]), ..SupernodalParams::default() };
        let sn = SupernodalSymbolic::new(a.symbolic(), Some(&order), &sp).unwrap();
        let plan = Plan::new(&sn, a.symbolic());
        let tiles = plan.inv_tiles();
        let dev = device();
        fn check<F: Scalar>(dev: &Arc<TestDevice>, sn: &SupernodalSymbolic, a: &Matrix, plan: &Plan, tiles: &[u32], tol: f64) {
            let mut g = GpuSupernodal::<F, _>::new(dev, sn, a.symbolic(), &GpuSupernodalParams::default(), wide(&**dev)).unwrap();
            g.factorize(a.vals()).unwrap();
            let (l, inv) = (widen(&g.factor_values().unwrap()), widen(&g.inv_values().unwrap()));
            let mut partial = 0;
            for (t, st) in tiles.chunks(2).enumerate() {
                let (s, j0) = (st[0] as usize, st[1] as usize);
                let (q, h, panel) = (plan.dims[2 * s] as usize, plan.dims[2 * s + 1] as usize, plan.val_ptr[s] as usize);
                let nb = (q - j0).min(32);
                partial += (nb < 32) as usize;
                let x = &inv[t * plan::INV_TILE..(t + 1) * plan::INV_TILE];
                for r in 0..32 {
                    for c in 0..32 {
                        let xrc = x[c * 32 + r];
                        assert_eq!(xrc, x[1024 + r * 32 + c], "tile {t}: the two copies at ({r}, {c})");
                        if r >= nb || c >= nb {
                            assert_eq!(xrc, 0.0, "tile {t} of {nb} columns at ({r}, {c})");
                            continue;
                        }
                        let lx: f64 = (0..=r).map(|k| l[panel + (j0 + k) * h + j0 + r] * x[c * 32 + k]).sum();
                        let err = (lx - if r == c { 1.0 } else { 0.0 }).abs();
                        assert!(err < tol, "{}: tile {t}, (L X)[{r}][{c}] off by {err}", F::CPP);
                    }
                }
            }
            assert!(partial > 0, "a tile narrower than 32 columns");
        }
        assert!(tiles.len() > 20, "inverted tiles: {}", tiles.len() / 2);
        if has_f64(&*dev, "the f64 factor") {
            check::<f64>(&dev, &sn, &a, &plan, &tiles, 1e-12);
        }
        check::<f32>(&dev, &sn, &a, &plan, &tiles, 1e-4);
    }

    /// The small kernel alone, on supernodes of three steps (the last
    /// narrower than a warp) and of rows past its block's first pass,
    /// matches the CPU's factor in f64 and f32.
    #[test]
    fn the_small_kernel_factors_wide_and_tall_supernodes() {
        let (a, _) = grid_spd_sparse(48, &[3, 6], 5);
        let order = nd_block_order(a.symbolic());
        let sn = SupernodalSymbolic::new(a.symbolic(), Some(&order), &SupernodalParams::default()).unwrap();
        let dev = device();
        let small = |widest: usize| -> Vec<(usize, usize)> {
            (0..sn.n_supernodes()).map(|s| sn.supernode_dims(s)).filter(|&(q, _)| q <= widest).collect()
        };
        let f64s = has_f64(&*dev, "the f64 factor");
        let widths = if f64s { vec![small_kernel_widest::<f64>(&dev), small_kernel_widest::<f32>(&dev)] } else { vec![small_kernel_widest::<f32>(&dev)] };
        for widest in widths {
            let small = small(widest);
            assert!(small.iter().any(|&(q, _)| q > 2 * plan::NB && q % plan::NB != 0), "a small supernode of three steps, the last narrower");
            assert!(small.iter().any(|&(q, h)| q > plan::NB && h > THREADS as usize), "a small supernode of rows past the first pass");
        }
        let mut cpu = vec![0.0f64; sn.factor_val_count()];
        supernodal_factorize(&sn, &a, &mut cpu, &mut SupernodalContext::new(), 1).unwrap();
        if f64s {
            let mut g = GpuSupernodal::<f64, _>::new(&dev, &sn, a.symbolic(), &small_kernel_only(), Wide::F64).unwrap();
            g.factorize(a.vals()).unwrap();
            let err = factor_err(&sn, &g.factor_values().unwrap(), &cpu);
            assert!(err < 1e-12, "f64 factor err/max {err}");
        }
        let mut g = GpuSupernodal::<f32, _>::new(&dev, &sn, a.symbolic(), &small_kernel_only(), wide(&*dev)).unwrap();
        g.factorize(a.vals()).unwrap();
        let err = factor_err(&sn, &widen(&g.factor_values().unwrap()), &cpu);
        assert!(err < 1e-5, "f32 factor err/max {err}");
    }

    /// The kernels holding a row per thread keep it in registers: none uses
    /// local memory, in f64 or f32 (a GPU's), but k_syrk in f64 on CUDA,
    /// held to two blocks a multiprocessor, spills a little in its potf2
    /// tail (measured worth the occupancy), within a bound.
    #[test]
    #[cfg(any(feature = "cuda", feature = "hip"))]
    fn the_row_kernels_keep_their_rows_in_registers() {
        let (a, _, _, sn) = setup(10);
        let dev = device();
        fn check<F: Scalar>(dev: &Arc<TestDevice>, sn: &SupernodalSymbolic, a: &Matrix) {
            let g = GpuSupernodal::<F, _>::new(dev, sn, a.symbolic(), &GpuSupernodalParams::default(), Wide::F64).unwrap();
            let k = &g.k;
            let kernels = [
                ("small_factor", &k.small_factor),
                ("chain", &k.chain),
                ("pair_update", &k.pair_update),
                ("potf2", &k.potf2),
                ("trsm", &k.trsm),
                ("syrk", &k.syrk),
                ("fwd_trsv", &k.fwd_trsv),
                ("bwd_trsv", &k.bwd_trsv),
                ("fwd_chain", &k.fwd_chain),
                ("bwd_chain", &k.bwd_chain),
                ("tile_inv", &k.tile_inv),
                ("big_update", &k.big_update),
                ("big_pair", &k.big_pair),
            ];
            for (name, kernel) in kernels {
                let local = kernel.local_bytes().unwrap();
                let allowed = if name == "syrk" && F::CPP == "double" && cfg!(feature = "cuda") { 256 } else { 0 };
                assert!(local <= allowed, "k_{name} in {}: {} registers, {local} bytes of local memory", F::CPP, kernel.registers().unwrap());
            }
        }
        check::<f64>(&dev, &sn, &a);
        check::<f32>(&dev, &sn, &a);
    }

    #[test]
    fn a_pivot_that_is_not_positive_is_reported() {
        let (mut a, _, _, sn) = setup(10);
        let asym = a.symbolic().clone();
        // the last block's diagonal made negative: the root supernode fails
        let j = asym.nblk_cols() - 1;
        let b = asym.col_range(j).find(|&b| asym.blk_row(b) == j).unwrap();
        let base = asym.val_range(b).start;
        let good = a.vals()[base];
        a.vals_mut()[base] = -1.0;
        // on the panel steps the bad pivot is in a later step's diagonal
        // tile, which the syrk before it factors
        let (q, _) = sn.supernode_dims(sn.n_supernodes() - 1);
        assert!(q > crate::plan::NB, "the root spans more than one panel step: {q} columns");
        let dev = device();
        fn check<F: Scalar>(dev: &Arc<TestDevice>, sn: &SupernodalSymbolic, asym: &SymbolicSparseBlockColMat<SparseIndex>, a: &mut Matrix, base: usize, good: f64) {
            let f64s = core::mem::size_of::<F>() == 8;
            for params in [GpuSupernodalParams::default(), every_kernel_busy(), cut_updates(), small_kernel_only(), chains_everywhere()].into_iter().chain(both_syrks()).chain(both_big_updates()) {
                let mut g = GpuSupernodal::<F, _>::new(dev, sn, asym, &params, wide(&**dev)).unwrap();
                if (if f64s { params.chain_flops_f64 } else { params.chain_flops_f32 }) == f64::INFINITY {
                    assert!(matches!(g.steps.last(), Some(Step::Chain { .. })), "the bad root in a chain");
                }
                assert!(matches!(g.factorize(a.vals()), Err(FactorizeError::NotPositiveDefinite)));
                // the flag is cleared by the next factorization
                a.vals_mut()[base] = good;
                g.factorize(a.vals()).unwrap();
                a.vals_mut()[base] = -1.0;
            }
        }
        if has_f64(&*dev, "the f64 factor") {
            check::<f64>(&dev, &sn, &asym, &mut a, base, good);
        }
        check::<f32>(&dev, &sn, &asym, &mut a, base, good);
    }

    /// Values and right-hand sides in f32 go through the same kernels as
    /// f64 ones rounded to f32.
    #[test]
    fn f32_values_and_rhs_match_their_f64_twins() {
        let (a, _, rhs, sn) = setup(16);
        let dev = device();
        // the factor f64, or f32 on a device without f64 (whose f64 values the host rounds)
        fn check<F: Scalar>(dev: &Arc<TestDevice>, a: &Matrix, rhs: &[f64], sn: &SupernodalSymbolic, tol: f64) {
            let mut g = GpuSupernodal::<F, _>::new(dev, sn, a.symbolic(), &every_kernel_busy(), wide(&**dev)).unwrap();
            let v32: Vec<f32> = a.vals().iter().map(|&v| v as f32).collect();
            let v64: Vec<f64> = v32.iter().map(|&v| v as f64).collect();
            g.factorize(&v64).unwrap();
            let from64 = widen(&g.factor_values().unwrap());
            g.factorize(&v32).unwrap();
            let from32 = widen(&g.factor_values().unwrap());
            // the tile updates add atomically, in no fixed order
            let err = factor_err(sn, &from32, &from64);
            assert!(err < tol, "{}: factor err/max {err}", F::CPP);
            let mut x32: Vec<f32> = rhs.iter().map(|&v| v as f32).collect();
            let mut x64: Vec<f64> = x32.iter().map(|&v| v as f64).collect();
            g.solve(&mut x32).unwrap();
            g.solve(&mut x64).unwrap();
            for (a, b) in x32.iter().zip(&x64) {
                assert!((*a as f64 - b).abs() <= 1e-6 * b.abs().max(1.0), "{}: {a} against {b}", F::CPP);
            }
        }
        if has_f64(&*dev, "the f64 factor") {
            check::<f64>(&dev, &a, &rhs, &sn, 1e-12);
        } else {
            check::<f32>(&dev, &a, &rhs, &sn, 1e-6);
        }
    }

    /// The device entry points: the factorization from f64 values on the
    /// device, and the solve that runs only when its gate is set.
    #[test]
    fn device_factorize_and_gated_solve() {
        let (a, dense, rhs, sn) = setup(16);
        let n = sn.dim();
        let dev = device();
        // the values and the vectors on the device in the wide scalar W
        let w = wide(&*dev);
        let mut g = GpuSupernodal::<f32, _>::new(&dev, &sn, a.symbolic(), &Default::default(), w).unwrap();
        g.factorize(a.vals()).unwrap();
        let from_host = widen(&g.factor_values().unwrap());
        let mut x_host = rhs.clone();
        g.solve(&mut x_host).unwrap();

        // one block: the values, then the flag, the gate and x, each on 256 bytes
        let nvals = a.vals().len();
        let at_fail = (nvals * 8).next_multiple_of(256);
        let (at_gate, at_x) = (at_fail + 256, at_fail + 512);
        let mut block = dev.alloc(at_x + n * 8).unwrap();
        dev.write(&mut block, 0, &w.encode(a.vals())).unwrap();
        // a factor of other values first, so the comparison below sees the device path's work
        g.factorize(&vec![0.0f64; nvals]).unwrap_err();
        g.factorize_device((&block, 0), (&block, at_fail)).unwrap();
        let mut flag = [7u32];
        dev.read(&block, at_fail, bytemuck::cast_slice_mut(&mut flag)).unwrap();
        assert_eq!(flag[0], 0, "no pivot failed");
        assert!(factor_err(&sn, &widen(&g.factor_values().unwrap()), &from_host) < 1e-6);

        let read_x = |block: &<TestDevice as Device>::Block| {
            let mut bytes = vec![0u8; n * 8];
            dev.read(block, at_x, &mut bytes).unwrap();
            let mut x = vec![0.0f64; n];
            w.decode(&bytes, &mut x);
            x
        };
        let rhs_w = w.encode(&rhs);
        let mut rhs_back = vec![0.0f64; n];
        w.decode(&rhs_w, &mut rhs_back);
        dev.write(&mut block, at_x, &rhs_w).unwrap();
        dev.write(&mut block, at_gate, bytemuck::cast_slice(&[0u32])).unwrap();
        g.solve_device_if((&block, at_x), (&block, at_gate)).unwrap();
        assert_eq!(read_x(&block), rhs_back, "a closed gate leaves x alone");

        dev.write(&mut block, at_gate, bytemuck::cast_slice(&[1u32])).unwrap();
        g.solve_device_if((&block, at_x), (&block, at_gate)).unwrap();
        let x = read_x(&block);
        let res = rel_residual(&dense, n, &x, &rhs);
        assert!(res < 1e-4, "an open gate solves: residual {res}");
        let dx = x.iter().zip(&x_host).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(dx < 1e-5, "the device solve against the host one: {dx}");

        dev.write(&mut block, at_x, &rhs_w).unwrap();
        g.solve_device((&block, at_x)).unwrap();
        assert!(rel_residual(&dense, n, &read_x(&block), &rhs) < 1e-4);
    }

}
