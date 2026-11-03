//! [`NativeSupernodal`]: the block supernodal Cholesky and its level-set
//! solve on a [`Device`]: the kernels of `kernels/supernodal.cpp` launched
//! over the plan of [`crate::plan`].

use std::fmt::Write as _;
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arael_faer::bsc::SymbolicSparseBlockColMat;
use arael_faer::schur::SchurReal;
use arael_faer::supernodal::SupernodalSymbolic;

use super::{Arg, Device, Launch, NativeError, Scalar, WIDE, Wide, is_f64, scalar_bytes};
use crate::plan::{self, GpuSupernodalParams, NB, Plan, SparseIndex, TB, build_levels};
use crate::supernodal::Packer;

/// The kernels, compiled per scalar and lane unroll.
const SOURCE: &str = include_str!("../../kernels/supernodal.cpp");

/// Threads per block of every kernel but `k_trsm`, which runs TB.
const THREADS: u32 = 256;

/// Why [`NativeSupernodal::factorize`] failed.
#[derive(Debug)]
pub enum FactorizeError {
    /// A diagonal pivot was not positive. The factor is unusable; the caller
    /// raises the damping, as with the CPU factorization.
    NotPositiveDefinite,
    Device(NativeError),
}

impl From<NativeError> for FactorizeError {
    fn from(e: NativeError) -> Self {
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
    seed_f64: K,
    seed_wide: K,
    tile_update: K,
    big_update: K,
    small_factor: K,
    potf2: K,
    trsm: K,
    syrk: K,
    fwd_gather: K,
    fwd_trsv: K,
    bwd_gemv: K,
    bwd_trsv: K,
    perm_in: K,
    perm_out: K,
}

/// One level's work lists, as byte offsets into the structure block, and
/// its launch sizes.
struct Level {
    tile_items: usize,
    tile_rects: usize,
    n_tile_items: u32,
    big_items: usize,
    big_rects: usize,
    n_big_items: u32,
    small: usize,
    n_small: u32,
    /// Dynamic shared memory of `k_small_factor`: q * q values for the widest small supernode.
    small_shared: u32,
    big: usize,
    n_big: u32,
    /// Per panel step, the grid sizes of `k_trsm` and `k_syrk`.
    trsm_grid: Vec<u32>,
    syrk_grid: Vec<u32>,
    all: usize,
    n_all: u32,
    /// Dynamic shared memory of the trsv kernels: 1024 + q values for the widest supernode.
    trsv_shared: u32,
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
    /// The pivot flag of [`NativeSupernodal::factorize`], a `u32`.
    fail: usize,
    /// The matrix's values from the host, `f32` or `f64` as given.
    vals: usize,
    /// The right-hand side and solution of [`NativeSupernodal::solve`], `f64`.
    rhs: usize,
    bytes: usize,
}

impl Sections {
    fn new(factor: usize, xb: usize, vals: usize, rhs: usize) -> Self {
        let mut at = 0;
        let mut take = |bytes: usize| {
            let start = at;
            at = (at + bytes.max(1)).next_multiple_of(256);
            start
        };
        let (factor, xb, fail, vals, rhs) = (take(factor), take(xb), take(4), take(vals), take(rhs));
        Sections { factor, xb, fail, vals, rhs, bytes: at }
    }
}

/// The device factorization of one sparsity structure on a native
/// [`Device`]: the plan built from a [`SupernodalSymbolic`] once, the factor
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
/// let mut g = NativeSupernodal::<f32, _>::new(&dev, &sn, a.symbolic(), &Default::default(), Wide::F64)?;
/// g.factorize(a.vals())?;          // Err(NotPositiveDefinite) when a pivot is not positive
/// g.solve(&mut rhs)?;              // rhs becomes x, in the matrix's ordering
/// ```
pub struct NativeSupernodal<F: Scalar, D: Device> {
    dev: Arc<D>,
    k: Kernels<D::Kernel>,
    /// The plan words (at 0, as the plan's header offsets require), the
    /// tiles, a word holding 1 (the gate of a solve that always runs), and
    /// every level's lists.
    ints: D::Block,
    tiles: usize,
    on: usize,
    ntiles: u32,
    levels: Vec<Level>,
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

/// The next factor's [`NativeSupernodal::id`].
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

impl<F: Scalar, D: Device> NativeSupernodal<F, D> {
    /// The device plan of `sn`, the analysis of a matrix with the block
    /// structure `asym`, uploaded to `dev`, with the factor's buffers and
    /// the kernels; the vectors it solves and the values a device
    /// factorization reads are in `wide`. [`NativeError::Limit`] when a
    /// supernode needs more shared memory than the device gives.
    pub fn new(
        dev: &Arc<D>,
        sn: &SupernodalSymbolic,
        asym: &SymbolicSparseBlockColMat<SparseIndex>,
        params: &GpuSupernodalParams,
        wide: Wide,
    ) -> Result<Self, NativeError> {
        let plan = Plan::new(sn, asym);
        let lw = build_levels(&plan, params);
        let fs = core::mem::size_of::<F>();
        let lu = params.lane_unroll.unwrap_or(if fs == 4 { 4 } else { 8 });
        assert!(lu >= 2 && lu.is_multiple_of(2), "the lane unroll is even and at least 2");
        let mut pk = Packer::default();
        let at_plan = pk.push(&plan.pack());
        debug_assert_eq!(at_plan, 0, "the plan's header offsets are relative to the block start");
        let at_tiles = pk.push(&plan.tiles);
        let at_on = pk.push(&[1]);
        let q = |s: &u32| plan.dims[2 * *s as usize] as usize;
        let limit = dev.arch().shared_bytes as usize;
        let mut levels = Vec::with_capacity(lw.len());
        for w in lw {
            let small_q = w.small_targets.iter().map(q).max().unwrap_or(0);
            let max_q = w.level_sn.iter().map(q).max().unwrap_or(0);
            let (small_shared, trsv_shared) = (small_q * small_q * fs, (1024 + max_q) * fs);
            if small_shared > limit {
                return Err(NativeError::Limit(format!(
                    "the small factor kernel needs {small_shared} bytes of shared memory for a supernode of {small_q} columns, the device gives {limit}; lower small_q"
                )));
            }
            if trsv_shared > limit {
                return Err(NativeError::Limit(format!(
                    "the solve needs {trsv_shared} bytes of shared memory for a supernode of {max_q} columns, the device gives {limit}"
                )));
            }
            levels.push(Level {
                tile_items: 4 * pk.push(&w.tile_items),
                tile_rects: 4 * pk.push(&w.tile_rects),
                n_tile_items: (w.tile_items.len() / 5) as u32,
                big_items: 4 * pk.push(&w.big_items),
                big_rects: 4 * pk.push(&w.big_rects),
                n_big_items: (w.big_items.len() / 5) as u32,
                small: 4 * pk.push(&w.small_targets),
                n_small: w.small_targets.len() as u32,
                small_shared: small_shared as u32,
                big: 4 * pk.push(&w.big_targets),
                n_big: w.big_targets.len() as u32,
                trsm_grid: w.trsm_grid,
                syrk_grid: w.syrk_grid,
                all: 4 * pk.push(&w.level_sn),
                n_all: w.level_sn.len() as u32,
                trsv_shared: trsv_shared as u32,
                g_items: 4 * pk.push(&w.g_items),
                n_g: (w.g_items.len() / 4) as u32,
                bg_items: 4 * pk.push(&w.bg_items),
                n_bg: (w.bg_items.len() / 3) as u32,
            });
        }
        let mut ints = dev.alloc(pk.words.len() * 4)?;
        dev.write(&mut ints, 0, bytemuck::cast_slice(&pk.words))?;
        let nvals = asym.val_count();
        let sec = Sections::new(plan.factor_vals * fs, plan.n * fs, nvals * 8, plan.n * 8);
        let work = dev.alloc(sec.bytes)?;
        let defines = format!("{}{}", wide.defines(), defines(F::CPP, lu));
        let source = format!("{WIDE}\n{SOURCE}");
        let c = |entry: &str| dev.compile(&source, &defines, entry);
        let k = Kernels {
            zero: c("k_zero")?,
            zero_u32: c("k_zero_u32")?,
            seed_f32: c("k_seed_f32")?,
            seed_f64: c("k_seed_f64")?,
            seed_wide: c("k_seed_wide")?,
            tile_update: c("k_tile_update")?,
            big_update: c("k_big_update")?,
            small_factor: c("k_small_factor")?,
            potf2: c("k_potf2")?,
            trsm: c("k_trsm")?,
            syrk: c("k_syrk")?,
            fwd_gather: c("k_fwd_gather")?,
            fwd_trsv: c("k_fwd_trsv")?,
            bwd_gemv: c("k_bwd_gemv")?,
            bwd_trsv: c("k_bwd_trsv")?,
            perm_in: c("k_perm_in")?,
            perm_out: c("k_perm_out")?,
        };
        Ok(NativeSupernodal {
            dev: dev.clone(),
            k,
            ints,
            tiles: 4 * at_tiles,
            on: 4 * at_on,
            ntiles: (plan.tiles.len() / 6) as u32,
            levels,
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
        self.dev.write(&mut self.work, self.sec.vals, scalar_bytes(vals))?;
        let seed = if is_f64::<M>() { &self.k.seed_f64 } else { &self.k.seed_f32 };
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
    pub fn factorize_device(&self, vals: (&D::Block, usize), fail: (&D::Block, usize)) -> Result<(), NativeError> {
        self.factorize_with(vals, &self.k.seed_wide, fail)
    }

    fn factorize_with(&self, vals: (&D::Block, usize), seed: &D::Kernel, fail: (&D::Block, usize)) -> Result<(), NativeError> {
        let plan = Arg::Block(&self.ints, 0);
        let list = |at: usize| Arg::Block(&self.ints, at);
        let factor = Arg::Block(&self.work, self.sec.factor);
        let fail = Arg::Block(fail.0, fail.1);
        let flen = self.factor_vals as u32;
        self.run(&self.k.zero, flen.div_ceil(THREADS).max(1), THREADS, 0, &[factor, Arg::U32(flen)])?;
        let seed_args = [list(self.tiles), Arg::Block(vals.0, vals.1), factor, Arg::U32(self.ntiles)];
        self.run(seed, self.ntiles.div_ceil(THREADS).max(1), THREADS, 0, &seed_args)?;
        for l in &self.levels {
            if l.n_tile_items > 0 {
                let args = [list(l.tile_items), list(l.tile_rects), plan, factor, factor];
                self.run(&self.k.tile_update, l.n_tile_items, THREADS, 0, &args)?;
            }
            if l.n_big_items > 0 {
                let args = [list(l.big_items), list(l.big_rects), plan, factor];
                self.run(&self.k.big_update, l.n_big_items, THREADS, 0, &args)?;
            }
            if l.n_small > 0 {
                self.run(&self.k.small_factor, l.n_small, THREADS, l.small_shared, &[list(l.small), plan, factor, fail])?;
            }
            for (step, (&tg, &sg)) in l.trsm_grid.iter().zip(&l.syrk_grid).enumerate() {
                let j = Arg::U32((step * NB) as u32);
                self.run(&self.k.potf2, l.n_big, THREADS, 0, &[list(l.big), j, plan, factor, fail])?;
                let args = [list(l.big), Arg::U32(l.n_big), j, plan, factor];
                if tg > 0 {
                    self.run(&self.k.trsm, tg, TB as u32, 0, &args)?;
                }
                if sg > 0 {
                    self.run(&self.k.syrk, sg, THREADS, 0, &args)?;
                }
            }
        }
        Ok(())
    }

    /// Solve `A x = rhs` in place through the last factorization, and wait
    /// for `x`; `rhs` is in the matrix's own ordering and comes back as `x`.
    pub fn solve<M: SchurReal + 'static>(&mut self, rhs: &mut [M]) -> Result<(), NativeError> {
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
    pub fn solve_device(&self, x: (&D::Block, usize)) -> Result<(), NativeError> {
        self.solve_with(x, (&self.ints, self.on))
    }

    /// [`solve_device`](Self::solve_device) when the `u32` at `gate` is
    /// nonzero, as the device finds it; otherwise its kernels do nothing.
    pub fn solve_device_if(&self, x: (&D::Block, usize), gate: (&D::Block, usize)) -> Result<(), NativeError> {
        self.solve_with(x, gate)
    }

    fn solve_with(&self, x: (&D::Block, usize), gate: (&D::Block, usize)) -> Result<(), NativeError> {
        let gate = Arg::Block(gate.0, gate.1);
        let plan = Arg::Block(&self.ints, 0);
        let list = |at: usize| Arg::Block(&self.ints, at);
        let factor = Arg::Block(&self.work, self.sec.factor);
        let xb = Arg::Block(&self.work, self.sec.xb);
        let x = Arg::Block(x.0, x.1);
        let blocks = self.nb.div_ceil(THREADS).max(1);
        self.run(&self.k.perm_in, blocks, THREADS, 0, &[gate, plan, x, xb, Arg::U32(self.nb)])?;
        for l in &self.levels {
            if l.n_g > 0 {
                self.run(&self.k.fwd_gather, l.n_g, THREADS, 0, &[gate, list(l.g_items), plan, factor, xb, xb])?;
            }
            self.run(&self.k.fwd_trsv, l.n_all, THREADS, l.trsv_shared, &[gate, list(l.all), plan, factor, xb])?;
        }
        for l in self.levels.iter().rev() {
            if l.n_bg > 0 {
                self.run(&self.k.bwd_gemv, l.n_bg, THREADS, 0, &[gate, list(l.bg_items), plan, factor, xb])?;
            }
            self.run(&self.k.bwd_trsv, l.n_all, THREADS, l.trsv_shared, &[gate, list(l.all), plan, factor, xb])?;
        }
        self.run(&self.k.perm_out, blocks, THREADS, 0, &[gate, plan, xb, x, Arg::U32(self.nb)])
    }

    /// The factor's values, panel by panel in the symbolic's layout, read
    /// back from the device (for tests and inspection).
    pub fn factor_values(&self) -> Result<Vec<F>, NativeError> {
        let mut v = vec![F::zeroed(); self.factor_vals];
        self.dev.read(&self.work, self.sec.factor, bytemuck::cast_slice_mut(&mut v))?;
        Ok(v)
    }

    fn run(&self, kernel: &D::Kernel, blocks: u32, threads: u32, shared_bytes: u32, args: &[Arg<'_, D::Block>]) -> Result<(), NativeError> {
        let launch = Launch { grid: [blocks, 1, 1], block: [threads, 1, 1], shared_bytes };
        self.dev.launch(kernel, launch, args)
    }
}

/// A [`NativeSupernodal`] of either scalar behind one type, for a caller
/// whose own scalar is generic.
pub enum NativeFactor<D: Device> {
    F32(NativeSupernodal<f32, D>),
    F64(NativeSupernodal<f64, D>),
}

impl<D: Device> NativeFactor<D> {
    /// As [`NativeSupernodal::new`], in `precision`.
    pub fn new(
        dev: &Arc<D>,
        precision: crate::GpuPrecision,
        sn: &SupernodalSymbolic,
        asym: &SymbolicSparseBlockColMat<SparseIndex>,
        params: &GpuSupernodalParams,
        wide: Wide,
    ) -> Result<Self, NativeError> {
        Ok(match precision {
            crate::GpuPrecision::F32 => NativeFactor::F32(NativeSupernodal::new(dev, sn, asym, params, wide)?),
            crate::GpuPrecision::F64 => NativeFactor::F64(NativeSupernodal::new(dev, sn, asym, params, wide)?),
        })
    }

    /// As [`NativeSupernodal::factorize`].
    pub fn factorize<M: SchurReal + 'static>(&mut self, vals: &[M]) -> Result<(), FactorizeError> {
        match self {
            NativeFactor::F32(f) => f.factorize(vals),
            NativeFactor::F64(f) => f.factorize(vals),
        }
    }

    /// As [`NativeSupernodal::solve`].
    pub fn solve<M: SchurReal + 'static>(&mut self, rhs: &mut [M]) -> Result<(), NativeError> {
        match self {
            NativeFactor::F32(f) => f.solve(rhs),
            NativeFactor::F64(f) => f.solve(rhs),
        }
    }

    /// The factor's scalar.
    pub fn precision(&self) -> crate::GpuPrecision {
        match self {
            NativeFactor::F32(_) => crate::GpuPrecision::F32,
            NativeFactor::F64(_) => crate::GpuPrecision::F64,
        }
    }

    /// As [`NativeSupernodal::factorize_device`].
    pub fn factorize_device(&self, vals: (&D::Block, usize), fail: (&D::Block, usize)) -> Result<(), NativeError> {
        match self {
            NativeFactor::F32(f) => f.factorize_device(vals, fail),
            NativeFactor::F64(f) => f.factorize_device(vals, fail),
        }
    }

    /// As [`NativeSupernodal::solve_device`].
    pub fn solve_device(&self, x: (&D::Block, usize)) -> Result<(), NativeError> {
        match self {
            NativeFactor::F32(f) => f.solve_device(x),
            NativeFactor::F64(f) => f.solve_device(x),
        }
    }

    /// As [`NativeSupernodal::solve_device_if`].
    pub fn solve_device_if(&self, x: (&D::Block, usize), gate: (&D::Block, usize)) -> Result<(), NativeError> {
        match self {
            NativeFactor::F32(f) => f.solve_device_if(x, gate),
            NativeFactor::F64(f) => f.solve_device_if(x, gate),
        }
    }

    /// As [`NativeSupernodal::id`].
    pub fn id(&self) -> u64 {
        match self {
            NativeFactor::F32(f) => f.id(),
            NativeFactor::F64(f) => f.id(),
        }
    }

    /// As [`NativeSupernodal::device`].
    pub fn device(&self) -> &Arc<D> {
        match self {
            NativeFactor::F32(f) => f.device(),
            NativeFactor::F64(f) => f.device(),
        }
    }

    /// As [`NativeSupernodal::dim`].
    pub fn dim(&self) -> usize {
        match self {
            NativeFactor::F32(f) => f.dim(),
            NativeFactor::F64(f) => f.dim(),
        }
    }
}

/// The `#define`s of `supernodal.cpp`: the scalar, the lane unroll, the
/// block size, and the tile sizes and plan header slots of [`crate::plan`].
fn defines(scalar: &str, lu: usize) -> String {
    use plan::*;
    let mut s = format!("#define SCALAR {scalar}\n#define LU {lu}u\n#define THREADS {THREADS}u\n");
    let consts = [
        ("NB", NB),
        ("TB", TB),
        ("TK", TK),
        ("TR", TR),
        ("TC", TC),
        ("GROUP", GROUP),
        ("XR_CH", XR_CH),
        ("H_DIMS", H_DIMS),
        ("H_VAL_PTR", H_VAL_PTR),
        ("H_SUP_BEGIN", H_SUP_BEGIN),
        ("H_SC_START", H_SC_START),
        ("H_PAIR_PTR", H_PAIR_PTR),
        ("H_PR_D", H_PR_D),
        ("H_PR_R0", H_PR_R0),
        ("H_PR_S", H_PR_S),
        ("H_PR_RMAP", H_PR_RMAP),
        ("H_PR_CMAP", H_PR_CMAP),
        ("H_RMAP", H_RMAP),
        ("H_CMAP", H_CMAP),
        ("H_PAIR_KW_PTR", H_PAIR_KW_PTR),
        ("H_ROW_SC", H_ROW_SC),
        ("H_ROW_SC_OFF", H_ROW_SC_OFF),
        ("H_ORDER", H_ORDER),
        ("H_BLK_WIDTH", H_BLK_WIDTH),
        ("H_PART", H_PART),
    ];
    for (name, v) in consts {
        writeln!(s, "#define {name} {v}u").unwrap();
    }
    s
}

#[cfg(all(test, feature = "native-cuda"))]
mod tests {
    use super::*;
    use crate::native::cuda::CudaDevice;
    use crate::test_util::{grid_spd, rel_residual};
    use arael_faer::bsc::SparseBlockColMat;
    use arael_faer::supernodal::{SupernodalContext, SupernodalParams, nd_block_order, supernodal_factorize, supernodal_solve};

    type Matrix = SparseBlockColMat<SparseIndex, f64>;

    fn device() -> Arc<CudaDevice> {
        Arc::new(CudaDevice::new(0).unwrap())
    }

    fn setup(k: usize) -> (Matrix, Vec<f64>, Vec<f64>, SupernodalSymbolic) {
        let (a, dense, rhs) = grid_spd(k, &[3, 6], 11);
        let order = nd_block_order(a.symbolic());
        let sn = SupernodalSymbolic::new(a.symbolic(), Some(&order), &SupernodalParams::default()).unwrap();
        (a, dense, rhs, sn)
    }

    /// The parameters that put every kernel to work on a small grid.
    fn every_kernel_busy() -> GpuSupernodalParams {
        GpuSupernodalParams { small_q: 8, big_pair_flops: 100.0, elem_cap: 256, solve_work: 512, lane_unroll: None }
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
        for params in [GpuSupernodalParams::default(), every_kernel_busy()] {
            let mut g = NativeSupernodal::<f64, _>::new(&dev, &sn, a.symbolic(), &params, Wide::F64).unwrap();
            assert!(sn.n_supernodes() > 20, "a tree with levels");
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

    #[test]
    fn f32_factor_solves_to_single_precision_and_refines() {
        let (a, dense, rhs, sn) = setup(24);
        let n = sn.dim();
        let dev = device();
        for params in [GpuSupernodalParams::default(), every_kernel_busy()] {
            let mut g = NativeSupernodal::<f32, _>::new(&dev, &sn, a.symbolic(), &params, Wide::F64).unwrap();
            g.factorize(a.vals()).unwrap();
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
        let dev = device();
        for params in [GpuSupernodalParams::default(), every_kernel_busy()] {
            let mut g = NativeSupernodal::<f64, _>::new(&dev, &sn, &asym, &params, Wide::F64).unwrap();
            assert!(matches!(g.factorize(a.vals()), Err(FactorizeError::NotPositiveDefinite)));
            // the flag is cleared by the next factorization
            a.vals_mut()[base] = good;
            g.factorize(a.vals()).unwrap();
            a.vals_mut()[base] = -1.0;
        }
    }

    /// Values and right-hand sides in f32 go through the same kernels as
    /// f64 ones rounded to f32.
    #[test]
    fn f32_values_and_rhs_match_their_f64_twins() {
        let (a, _, rhs, sn) = setup(16);
        let dev = device();
        let mut g = NativeSupernodal::<f64, _>::new(&dev, &sn, a.symbolic(), &every_kernel_busy(), Wide::F64).unwrap();
        let v32: Vec<f32> = a.vals().iter().map(|&v| v as f32).collect();
        let v64: Vec<f64> = v32.iter().map(|&v| v as f64).collect();
        g.factorize(&v64).unwrap();
        let from64 = g.factor_values().unwrap();
        g.factorize(&v32).unwrap();
        let from32 = g.factor_values().unwrap();
        // the tile updates add atomically, in no fixed order
        assert!(factor_err(&sn, &from32, &from64) < 1e-12);
        let mut x32: Vec<f32> = rhs.iter().map(|&v| v as f32).collect();
        let mut x64: Vec<f64> = x32.iter().map(|&v| v as f64).collect();
        g.solve(&mut x32).unwrap();
        g.solve(&mut x64).unwrap();
        for (a, b) in x32.iter().zip(&x64) {
            assert!((*a as f64 - b).abs() <= 1e-6 * b.abs().max(1.0), "{a} against {b}");
        }
    }

    /// The device entry points: the factorization from f64 values on the
    /// device, and the solve that runs only when its gate is set.
    #[test]
    fn device_factorize_and_gated_solve() {
        let (a, dense, rhs, sn) = setup(16);
        let n = sn.dim();
        let dev = device();
        let mut g = NativeSupernodal::<f32, _>::new(&dev, &sn, a.symbolic(), &Default::default(), Wide::F64).unwrap();
        g.factorize(a.vals()).unwrap();
        let from_host = widen(&g.factor_values().unwrap());
        let mut x_host = rhs.clone();
        g.solve(&mut x_host).unwrap();

        // one block: the values, then the flag, the gate and x, each on 256 bytes
        let nvals = a.vals().len();
        let at_fail = (nvals * 8).next_multiple_of(256);
        let (at_gate, at_x) = (at_fail + 256, at_fail + 512);
        let mut block = dev.alloc(at_x + n * 8).unwrap();
        dev.write(&mut block, 0, bytemuck::cast_slice(a.vals())).unwrap();
        // a factor of other values first, so the comparison below sees the device path's work
        g.factorize(&vec![0.0f64; nvals]).unwrap_err();
        g.factorize_device((&block, 0), (&block, at_fail)).unwrap();
        let mut flag = [7u32];
        dev.read(&block, at_fail, bytemuck::cast_slice_mut(&mut flag)).unwrap();
        assert_eq!(flag[0], 0, "no pivot failed");
        assert!(factor_err(&sn, &widen(&g.factor_values().unwrap()), &from_host) < 1e-6);

        dev.write(&mut block, at_x, bytemuck::cast_slice(&rhs)).unwrap();
        dev.write(&mut block, at_gate, bytemuck::cast_slice(&[0u32])).unwrap();
        g.solve_device_if((&block, at_x), (&block, at_gate)).unwrap();
        let mut x = vec![0.0f64; n];
        dev.read(&block, at_x, bytemuck::cast_slice_mut(&mut x)).unwrap();
        assert_eq!(x, rhs, "a closed gate leaves x alone");

        dev.write(&mut block, at_gate, bytemuck::cast_slice(&[1u32])).unwrap();
        g.solve_device_if((&block, at_x), (&block, at_gate)).unwrap();
        dev.read(&block, at_x, bytemuck::cast_slice_mut(&mut x)).unwrap();
        let res = rel_residual(&dense, n, &x, &rhs);
        assert!(res < 1e-4, "an open gate solves: residual {res}");
        let dx = x.iter().zip(&x_host).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(dx < 1e-5, "the device solve against the host one: {dx}");

        dev.write(&mut block, at_x, bytemuck::cast_slice(&rhs)).unwrap();
        g.solve_device((&block, at_x)).unwrap();
        dev.read(&block, at_x, bytemuck::cast_slice_mut(&mut x)).unwrap();
        assert!(rel_residual(&dense, n, &x, &rhs) < 1e-4);
    }

    /// The native kernels against the CubeCL ones, factor and solution, in
    /// both scalars: equal to rounding (the tile updates add atomically, in
    /// no fixed order).
    #[cfg(feature = "cuda")]
    #[test]
    fn native_matches_cubecl() {
        fn compare<F: Scalar + crate::Real>(tol: f64) {
            let (a, _, rhs, sn) = setup(24);
            let dev = device();
            let ctx = crate::GpuContext::new(crate::GpuBackend::Cuda, 0).unwrap();
            for params in [GpuSupernodalParams::default(), every_kernel_busy()] {
                let mut native = NativeSupernodal::<F, _>::new(&dev, &sn, a.symbolic(), &params, Wide::F64).unwrap();
                let mut cubecl = crate::GpuSupernodal::<F>::new(&ctx, &sn, a.symbolic(), &params);
                native.factorize(a.vals()).unwrap();
                cubecl.factorize(a.vals()).unwrap();
                let (fn_, fc) = (native.factor_values().unwrap(), cubecl.factor_values());
                let identical = |a: &[F], b: &[F]| a.iter().zip(b).filter(|&(&a, &b)| Scalar::to_f64(a).to_bits() == Scalar::to_f64(b).to_bits()).count();
                let err = factor_err(&sn, &widen(&fn_), &widen(&fc));
                // a second native run: what the atomics' order alone changes
                native.factorize(a.vals()).unwrap();
                let again = identical(&fn_, &native.factor_values().unwrap());
                println!(
                    "{}: factor err/max {err:.2e}; bit-identical values of {}: {} against CubeCL, {again} against a second native run",
                    F::CPP,
                    fn_.len(),
                    identical(&fn_, &fc)
                );
                assert!(err < tol, "{}: factor err/max {err}", F::CPP);
                let (mut xn, mut xc) = (rhs.clone(), rhs.clone());
                native.solve(&mut xn).unwrap();
                cubecl.solve(&mut xc);
                let dx = xn.iter().zip(&xc).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
                let xmax = xc.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                println!("{}: solution max diff {:.2e} of max {xmax:.2e}", F::CPP, dx);
                assert!(dx < tol * 10.0 * xmax, "{}: solutions differ by {dx}", F::CPP);
            }
        }
        compare::<f64>(1e-13);
        compare::<f32>(1e-5);
    }

    /// Median wall times of factorize and solve (each with its upload and
    /// its blocking read), native against CubeCL:
    /// `cargo test -r -p arael-gpu --features cuda,native-cuda -- --ignored --nocapture native_against_cubecl`.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "timing"]
    fn time_native_against_cubecl() {
        fn median(mut v: Vec<f64>) -> f64 {
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        }
        fn time<F: Scalar + crate::Real>(a: &Matrix, rhs: &[f64], sn: &SupernodalSymbolic) {
            const RUNS: usize = 30;
            let dev = device();
            let ctx = crate::GpuContext::new(crate::GpuBackend::Cuda, 0).unwrap();
            let params = GpuSupernodalParams::default();
            let mut native = NativeSupernodal::<F, _>::new(&dev, sn, a.symbolic(), &params, Wide::F64).unwrap();
            let mut cubecl = crate::GpuSupernodal::<F>::new(&ctx, sn, a.symbolic(), &params);
            let ms = |t: std::time::Instant| t.elapsed().as_secs_f64() * 1e3;
            let (mut nf, mut ns, mut cf, mut cs) = (vec![], vec![], vec![], vec![]);
            for run in 0..RUNS + 3 {
                let mut x = rhs.to_vec();
                let t = std::time::Instant::now();
                native.factorize(a.vals()).unwrap();
                let tf = ms(t);
                let t = std::time::Instant::now();
                native.solve(&mut x).unwrap();
                let ts = ms(t);
                if run >= 3 {
                    nf.push(tf);
                    ns.push(ts);
                }
                let mut x = rhs.to_vec();
                let t = std::time::Instant::now();
                cubecl.factorize(a.vals()).unwrap();
                let tf = ms(t);
                let t = std::time::Instant::now();
                cubecl.solve(&mut x);
                let ts = ms(t);
                if run >= 3 {
                    cf.push(tf);
                    cs.push(ts);
                }
            }
            println!(
                "  {:6} factor: native {:7.3} ms, CubeCL {:7.3} ms; solve: native {:7.3} ms, CubeCL {:7.3} ms",
                F::CPP,
                median(nf),
                median(cf),
                median(ns),
                median(cs)
            );
        }
        for k in [24, 64] {
            let (a, rhs, sn) = if k <= 24 {
                let (a, _, rhs, sn) = setup(k);
                (a, rhs, sn)
            } else {
                large_grid(k)
            };
            println!("grid {k} x {k}: {} scalars, {} supernodes, {} factor values", sn.dim(), sn.n_supernodes(), sn.factor_val_count());
            time::<f64>(&a, &rhs, &sn);
            time::<f32>(&a, &rhs, &sn);
        }
    }

    /// `grid_spd`'s structure at a size whose dense twin would not fit: the
    /// same couplings, values that keep it positive definite.
    #[cfg(feature = "cuda")]
    fn large_grid(k: usize) -> (Matrix, Vec<f64>, SupernodalSymbolic) {
        let widths = [3usize, 6];
        let nblk = k * k;
        let mut part = vec![0u32];
        for b in 0..nblk {
            part.push(part[b] + widths[b % 2] as u32);
        }
        let mut cells = Vec::new();
        for r in 0..k {
            for c in 0..k {
                let b = r * k + c;
                cells.push((b, b));
                if c + 1 < k {
                    cells.push((b, b + 1));
                }
                if r + 1 < k {
                    cells.push((b, b + k));
                }
            }
        }
        let (sym, _) = SymbolicSparseBlockColMat::<SparseIndex>::from_scalar_coords(part.clone(), part.clone(), cells.len(), |i| {
            (part[cells[i].0] as usize, part[cells[i].1] as usize)
        });
        let mut a = Matrix::zeroed(sym);
        let asym = a.symbolic().clone();
        let mut rng = crate::test_util::Lcg(3);
        for j in 0..asym.nblk_cols() {
            for b in asym.col_range(j) {
                let (wi, wj) = asym.block_dims(b);
                let base = asym.val_range(b).start;
                let diag = asym.blk_row(b) == j;
                for cc in 0..wj {
                    for rr in 0..wi {
                        if !(diag && rr > cc) {
                            a.vals_mut()[base + rr + cc * wi] = rng.unit() + if diag && rr == cc { 120.0 } else { 0.0 };
                        }
                    }
                }
            }
        }
        let rhs: Vec<f64> = (0..part[nblk] as usize).map(|_| rng.unit()).collect();
        let order = nd_block_order(a.symbolic());
        let sn = SupernodalSymbolic::new(a.symbolic(), Some(&order), &SupernodalParams::default()).unwrap();
        (a, rhs, sn)
    }
}
