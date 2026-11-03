//! [`NativeSchur`]: the Schur reduction on a [`Device`], the reduced system
//! kept there for a [`NativeFactor`], the refinement and the
//! back-substitution: the kernels of `kernels/schur.cpp` over the plan of
//! [`crate::schur`], whose module documentation describes the stages.
//!
//! ```ignore
//! let dev = Arc::new(CudaDevice::new(0)?);
//! let mut g = NativeSchur::new(&dev, &schur_sym, h.symbolic(), Wide::Df32)?;
//! let mut f = NativeFactor::new(&dev, GpuPrecision::F32, &sn_of_s, &schur_sym.s, &Default::default(), Wide::Df32)?;
//! // the full step; Err when an eliminated block or the reduced system is not definite
//! let stats = g.solve(&mut f, h.vals(), &rhs, Some((4, 1e-10)), &mut delta)?;
//! ```

use std::sync::Arc;

use arael_faer::bsc::SymbolicSparseBlockColMat;
use arael_faer::schur::{SchurReal, SchurSymbolic};

use super::supernodal::NativeFactor;
use super::{Arg, Device, Launch, Layout, NativeError, WIDE, Wide, define_u32s};
use crate::plan::SparseIndex;
use crate::schur::{self as host, GpuSchurError, RefineStats, SchurPlan};
use crate::supernodal::Packer;

/// The kernels, compiled per block-width class.
const SOURCE: &str = include_str!("../../kernels/schur.cpp");

/// Threads per block of the kernels over entries and blocks.
const THREADS: u32 = 256;
/// Threads per block of the kernels over the tiles of S and the kept blocks.
const PLANE: u32 = 32;

/// Bytes of a solve's status at [`NativeSchur::status_view`]: the
/// refinement's state and the flags, for [`NativeSchur::outcome`].
pub const STATUS_BYTES: usize = host::STATUS_BYTES;

/// Why a [`NativeSchur`] solve failed.
#[derive(Clone, Debug)]
pub enum NativeSchurError {
    /// The diagonal tile of the eliminated block `block` is not positive
    /// definite; the caller raises the damping.
    Eliminated { block: usize },
    /// The reduced system is not positive definite; the caller raises the
    /// damping.
    Reduced,
    Device(NativeError),
}

impl From<NativeError> for NativeSchurError {
    fn from(e: NativeError) -> Self {
        NativeSchurError::Device(e)
    }
}

impl From<GpuSchurError> for NativeSchurError {
    fn from(e: GpuSchurError) -> Self {
        match e {
            GpuSchurError::Eliminated { block } => NativeSchurError::Eliminated { block },
            GpuSchurError::Reduced => NativeSchurError::Reduced,
        }
    }
}

impl core::fmt::Display for NativeSchurError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            NativeSchurError::Eliminated { block } => write!(f, "the diagonal tile of eliminated block {block} is not positive definite"),
            NativeSchurError::Reduced => write!(f, "the reduced system is not positive definite"),
            NativeSchurError::Device(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for NativeSchurError {}

/// What a captured attempt depends on beyond the reduction's own blocks:
/// the factor whose kernels it recorded and the refinement's launches and
/// tolerance.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GraphKey {
    factor: u64,
    max_steps: usize,
    tol_bits: u64,
}

struct Kernels<K> {
    zero_u32: K,
    elim: K,
    w: K,
    tiles: K,
    rhs: K,
    scatter_kept: K,
    backsub: K,
    symv_sub: K,
    copy: K,
    refine_start: K,
    refine_apply: K,
    refine_judge: K,
    refine_revert: K,
}

/// Byte offsets of the work block's sections.
struct Sections {
    /// The Hessian's tile values and the gradient, f64.
    h: usize,
    rhs: usize,
    dinv: usize,
    z: usize,
    w: usize,
    s: usize,
    rhs_kept: usize,
    x_kept: usize,
    x_prev: usize,
    r: usize,
    /// What a solve reads back, in one copy: the full step (`n` f64), then
    /// at `status_at` from here the refinement's state and the flags.
    out: usize,
    bytes: usize,
}

/// The device plan of a [`SchurSymbolic`] and the device state of one
/// reduction on a native [`Device`]: the Hessian's values and gradient as
/// uploaded, the block inverses, W, S and the reduced right-hand side, the
/// solution. Every device buffer is made in [`new`](Self::new): the plan in
/// one block, uploaded in one copy, and the rest in a work block.
pub struct NativeSchur<D: Device> {
    /// The scalar of H, the gradient, S and the reduction's vectors.
    /// The last attempt's launches as a graph, with what it was captured
    /// against; first, so it is dropped before the blocks it records.
    graph: Option<(GraphKey, D::Graph)>,
    graphs: bool,
    replayed: bool,
    dev: Arc<D>,
    wide: Wide,
    k: Kernels<D::Kernel>,
    /// The plan words (at 0, as the plan's header offsets require), then a
    /// word holding 1, at `on`: the gate of kernels that always run; then
    /// per kept block the tiles of S touching it, `kt_ptr` and `kt` (byte
    /// offsets), for the residual.
    ints: D::Block,
    on: usize,
    kt_ptr: usize,
    kt: usize,
    work: D::Block,
    sec: Sections,
    status_at: usize,
    ne: u32,
    no: u32,
    nt: u32,
    nk: u32,
    n: usize,
    nks: usize,
    nh: usize,
    s_len: usize,
    elim_blk: Vec<u32>,
    /// The inputs encoded in the wide scalar.
    conv: Vec<u8>,
    /// The step and the status as a solve reads them.
    host_out: Vec<u8>,
    step: Vec<f64>,
}

impl<D: Device> NativeSchur<D> {
    /// The device plan of `sym`, the reduction of a matrix with the block
    /// structure `hsym`, uploaded to `dev`, with the reduction's blocks and
    /// kernels, in `wide`.
    pub fn new(dev: &Arc<D>, sym: &SchurSymbolic<SparseIndex>, hsym: &SymbolicSparseBlockColMat<SparseIndex>, wide: Wide) -> Result<Self, NativeError> {
        Self::build(dev, SchurPlan::reduction(sym, hsym), hsym.val_count(), wide)
    }

    /// [`new`](Self::new) with nothing eliminated: S is H, and a solve is
    /// the whole system's.
    pub fn whole(dev: &Arc<D>, hsym: &SymbolicSparseBlockColMat<SparseIndex>, wide: Wide) -> Result<Self, NativeError> {
        Self::build(dev, SchurPlan::whole(hsym), hsym.val_count(), wide)
    }

    fn build(dev: &Arc<D>, p: SchurPlan, nh: usize, wide: Wide) -> Result<Self, NativeError> {
        let defines = format!("{}{}", wide.defines(), defines(&p));
        let (kt_ptr_words, kt_words) = tiles_by_kept_block(&p);
        let mut pk = Packer { words: p.words };
        let on = 4 * pk.push(&[1]);
        let kt_ptr = 4 * pk.push(&kt_ptr_words);
        let kt = 4 * pk.push(&kt_words);
        let mut ints = dev.alloc(pk.words.len() * 4)?;
        dev.write(&mut ints, 0, bytemuck::cast_slice(&pk.words))?;
        let status_at = host::status_at(p.n);
        let f = core::mem::size_of::<f64>();
        let mut l = Layout::new();
        let sec = Sections {
            h: l.take(nh * f),
            rhs: l.take(p.n * f),
            dinv: l.take(p.dinv_len * f),
            z: l.take(p.z_len * f),
            w: l.take(p.w_len * f),
            s: l.take(p.s_len * f),
            rhs_kept: l.take(p.nks * f),
            x_kept: l.take(p.nks * f),
            x_prev: l.take(p.nks * f),
            r: l.take(p.nks * f),
            out: l.take(status_at + STATUS_BYTES),
            bytes: l.bytes(),
        };
        let work = dev.alloc(sec.bytes)?;
        let source = format!("{WIDE}\n{SOURCE}");
        let c = |entry: &str| dev.compile(&source, &defines, entry);
        let k = Kernels {
            zero_u32: c("k_zero_u32")?,
            elim: c("k_schur_elim")?,
            w: c("k_schur_w")?,
            tiles: c("k_schur_tiles")?,
            rhs: c("k_schur_rhs")?,
            scatter_kept: c("k_scatter_kept")?,
            backsub: c("k_schur_backsub")?,
            symv_sub: c("k_symv_sub")?,
            copy: c("k_copy_f64")?,
            refine_start: c("k_refine_start")?,
            refine_apply: c("k_refine_apply")?,
            refine_judge: c("k_refine_judge")?,
            refine_revert: c("k_refine_revert")?,
        };
        Ok(NativeSchur {
            graph: None,
            graphs: true,
            replayed: false,
            dev: dev.clone(),
            wide,
            k,
            ints,
            on,
            kt_ptr,
            kt,
            work,
            sec,
            status_at,
            ne: p.ne,
            no: p.no,
            nt: p.nt,
            nk: p.nk,
            n: p.n,
            nks: p.nks,
            nh,
            s_len: p.s_len,
            elim_blk: p.elim_blk,
            conv: Vec::new(),
            host_out: vec![0; status_at + STATUS_BYTES],
            step: vec![0.0; p.n],
        })
    }

    /// The device the reduction runs on; its factor runs there too.
    pub fn device(&self) -> &Arc<D> {
        &self.dev
    }

    /// The Hessian's scalar dimension.
    pub fn dim(&self) -> usize {
        self.n
    }

    /// The reduced system's scalar dimension.
    pub fn kept_size(&self) -> usize {
        self.nks
    }

    /// S's values on the device (block, byte offset) and their count, in the
    /// storage order of the symbolic's `s`, as the last reduction left them.
    pub fn s_values(&self) -> ((&D::Block, usize), usize) {
        ((&self.work, self.sec.s), self.s_len)
    }

    /// The scalar of H, the gradient, S and the reduction's vectors.
    pub fn wide(&self) -> Wide {
        self.wide
    }

    /// The reduced right-hand side on the device, [`kept_size`](Self::kept_size) values.
    pub fn rhs_kept(&self) -> (&D::Block, usize) {
        (&self.work, self.sec.rhs_kept)
    }

    /// Where [`launch`](Self::launch) reads the Hessian's tile values: the
    /// structure's value count, in [`wide`](Self::wide), for a device
    /// assembly to write.
    pub fn h_target(&self) -> (&D::Block, usize) {
        (&self.work, self.sec.h)
    }

    /// Where [`launch`](Self::launch) reads the gradient: [`dim`](Self::dim) values.
    pub fn rhs_target(&self) -> (&D::Block, usize) {
        (&self.work, self.sec.rhs)
    }

    /// The step of the last solve: [`dim`](Self::dim) values.
    pub fn step_view(&self) -> (&D::Block, usize) {
        (&self.work, self.sec.out)
    }

    /// The last solve's status ([`STATUS_BYTES`] bytes), to read and hand to
    /// [`outcome`](Self::outcome).
    pub fn status_view(&self) -> (&D::Block, usize) {
        (&self.work, self.sec.out + self.status_at)
    }

    /// The Hessian's tile values as the last upload or device assembly left
    /// them, read back (for diagnostics).
    pub fn h_values(&self) -> Result<Vec<f64>, NativeError> {
        self.read_values(self.sec.h, self.nh)
    }

    /// S's values and the reduced right-hand side read back (for tests).
    pub fn download(&self) -> Result<(Vec<f64>, Vec<f64>), NativeError> {
        Ok((self.read_values(self.sec.s, self.s_len)?, self.read_values(self.sec.rhs_kept, self.nks)?))
    }

    /// `n` values of the work block from byte `at` on, read back.
    fn read_values(&self, at: usize, n: usize) -> Result<Vec<f64>, NativeError> {
        let mut bytes = vec![0u8; 8 * n];
        self.dev.read(&self.work, at, &mut bytes)?;
        let mut v = vec![0.0f64; n];
        self.wide.decode(&bytes, &mut v);
        Ok(v)
    }

    /// Byte offset of flag `f` in the work block.
    fn flag_at(&self, f: usize) -> usize {
        self.sec.out + self.status_at + host::FLAGS_AT + 4 * f
    }

    /// Write the Hessian's tile values (`hsym`'s value count) and the
    /// gradient at [`h_target`](Self::h_target) and
    /// [`rhs_target`](Self::rhs_target), in [`wide`](Self::wide), for
    /// [`launch`](Self::launch) (no wait).
    pub fn upload<M: SchurReal + 'static>(&mut self, hvals: &[M], rhs: &[M]) -> Result<(), NativeError> {
        assert_eq!(hvals.len(), self.nh, "the Hessian's values, in the structure the reduction was built for");
        assert_eq!(rhs.len(), self.n);
        let wide = self.wide;
        self.conv.clear();
        self.conv.extend(hvals.iter().flat_map(|v| wide.bits(v.to_f64()).to_le_bytes()));
        self.dev.write(&mut self.work, self.sec.h, &self.conv)?;
        self.conv.clear();
        self.conv.extend(rhs.iter().flat_map(|v| wide.bits(v.to_f64()).to_le_bytes()));
        self.dev.write(&mut self.work, self.sec.rhs, &self.conv)
    }

    /// `kernel` over `blocks` blocks of `threads`; nothing when there are no blocks.
    fn run(&self, kernel: &D::Kernel, blocks: u32, threads: u32, args: &[Arg<'_, D::Block>]) -> Result<(), NativeError> {
        if blocks == 0 {
            return Ok(());
        }
        self.dev.launch(kernel, Launch { grid: [blocks, 1, 1], block: [threads, 1, 1], shared_bytes: 0 }, args)
    }

    /// Zero the flags and launch the reduction, from the uploaded values.
    fn launch_reduce(&self) -> Result<(), NativeError> {
        let (s, plan) = (&self.sec, Arg::Block(&self.ints, 0));
        let w = |at: usize| Arg::Block(&self.work, at);
        self.run(&self.k.zero_u32, 1, PLANE, &[w(self.flag_at(0)), Arg::U32(host::N_FLAGS as u32)])?;
        let elim = [plan, w(s.h), w(s.rhs), w(s.dinv), w(s.z), w(self.flag_at(host::F_ELIM)), Arg::U32(self.ne)];
        self.run(&self.k.elim, self.ne.div_ceil(THREADS), THREADS, &elim)?;
        self.run(&self.k.w, self.no.div_ceil(THREADS), THREADS, &[plan, w(s.h), w(s.dinv), w(s.w), Arg::U32(self.no)])?;
        self.run(&self.k.tiles, self.nt, PLANE, &[plan, w(s.h), w(s.w), w(s.s)])?;
        self.run(&self.k.rhs, self.nk, PLANE, &[plan, w(s.h), w(s.rhs), w(s.z), w(s.rhs_kept)])
    }

    /// Form S and the reduced right-hand side from the damped Hessian's tile
    /// values `hvals` (in `hsym`'s storage order) and the gradient `rhs`, and
    /// wait for the pivot flags (for inspection: [`solve`](Self::solve)
    /// reduces as its first stage).
    pub fn reduce<M: SchurReal + 'static>(&mut self, hvals: &[M], rhs: &[M]) -> Result<(), NativeSchurError> {
        self.upload(hvals, rhs)?;
        self.launch_reduce()?;
        let mut failed = [0u32];
        self.dev.read(&self.work, self.flag_at(host::F_ELIM), bytemuck::cast_slice_mut(&mut failed))?;
        if failed[0] != 0 { Err(NativeSchurError::Eliminated { block: self.elim_blk[failed[0] as usize - 1] as usize }) } else { Ok(()) }
    }

    /// Whether [`solve`](Self::solve) captures its launches as a graph and
    /// replays it (the default): the first solve with a factor and a
    /// refinement setting runs the kernels and records them, and the solves
    /// after it replay the recording, one launch instead of hundreds.
    pub fn use_graphs(&mut self, on: bool) {
        self.graphs = on;
        if !on {
            self.graph = None;
        }
    }

    /// Whether solves capture and replay graphs ([`use_graphs`](Self::use_graphs)).
    pub fn graphs(&self) -> bool {
        self.graphs
    }

    /// Whether the last [`solve`](Self::solve) replayed a captured graph.
    pub fn replayed(&self) -> bool {
        self.replayed
    }

    /// The launches of one damped solve after the uploads: the reduction,
    /// the factorization, the solve, `max_steps` refinement steps, the
    /// back-substitution. The same sequence every time for the same factor
    /// and refinement, which is what makes it a graph.
    fn launch_attempt(&self, factor: &NativeFactor<D>, max_steps: usize, tol_rel: f64) -> Result<(), NativeError> {
        self.launch_reduce()?;
        let (s, plan) = (&self.sec, Arg::Block(&self.ints, 0));
        let w = |at: usize| Arg::Block(&self.work, at);
        factor.factorize_device((&self.work, s.s), (&self.work, self.flag_at(host::F_FACTOR)))?;
        let nks = self.nks as u32;
        let per_entry = nks.div_ceil(THREADS);
        let on = Arg::Block(&self.ints, self.on);
        // the flags, and the one that gates a refinement step's kernels
        let flags = w(self.flag_at(0));
        let gate = w(self.flag_at(host::F_ACTIVE));
        let state = w(s.out + self.status_at);
        let copy = |gate, dst: usize, src: usize| self.run(&self.k.copy, per_entry, THREADS, &[gate, w(dst), w(src), Arg::U32(nks)]);
        // r = b - S x
        let residual = |gate| {
            copy(gate, s.r, s.rhs_kept)?;
            let (kt_ptr, kt) = (Arg::Block(&self.ints, self.kt_ptr), Arg::Block(&self.ints, self.kt));
            self.run(&self.k.symv_sub, self.nk, PLANE, &[gate, plan, kt_ptr, kt, w(s.s), w(s.x_kept), w(s.r)])
        };
        copy(on, s.x_kept, s.rhs_kept)?;
        factor.solve_device((&self.work, s.x_kept))?;
        if max_steps > 0 {
            residual(on)?;
            self.run(&self.k.refine_start, 1, THREADS, &[w(s.rhs_kept), w(s.r), state, flags, Arg::U32(nks), self.wide.arg(tol_rel)])?;
            for _ in 0..max_steps {
                factor.solve_device_if((&self.work, s.r), (&self.work, self.flag_at(host::F_ACTIVE)))?;
                self.run(&self.k.refine_apply, per_entry, THREADS, &[flags, w(s.x_kept), w(s.x_prev), w(s.r), Arg::U32(nks)])?;
                residual(gate)?;
                self.run(&self.k.refine_judge, 1, THREADS, &[w(s.r), state, flags, Arg::U32(nks)])?;
            }
            self.run(&self.k.refine_revert, per_entry, THREADS, &[flags, w(s.x_kept), w(s.x_prev), Arg::U32(nks)])?;
        }
        // the kept blocks into the full vector, then the eliminated ones
        self.run(&self.k.scatter_kept, self.nk.div_ceil(THREADS), THREADS, &[plan, w(s.x_kept), w(s.out), Arg::U32(self.nk)])?;
        self.run(&self.k.backsub, self.ne.div_ceil(THREADS), THREADS, &[plan, w(s.h), w(s.rhs), w(s.dinv), w(s.out), Arg::U32(self.ne)])
    }

    /// One damped solve, entirely on the device, waiting once: form S and
    /// the reduced right-hand side from the Hessian's tile values `hvals`
    /// (in `hsym`'s storage order) and the gradient `rhs`, factorize S with
    /// `factor`, solve for the kept blocks, refine the solution against S
    /// when `refine` is `Some((max_steps, tol))` (an f32 factor; `tol`
    /// relative to the right-hand side), recover the eliminated blocks, and
    /// read the full step into `delta` with the pivot flags and the
    /// refinement's state.
    ///
    /// The refinement decides on the device: `max_steps` steps are
    /// launched, and those after it has converged or stalled do nothing. A
    /// failed pivot is found with the result, so a failing solve runs to the
    /// end; `delta` is then left as it was. The launches are captured as a
    /// graph and replayed by the solves after it (see
    /// [`use_graphs`](Self::use_graphs)).
    pub fn solve<M: SchurReal + 'static>(
        &mut self,
        factor: &mut NativeFactor<D>,
        hvals: &[M],
        rhs: &[M],
        refine: Option<(usize, f64)>,
        delta: &mut [M],
    ) -> Result<RefineStats, NativeSchurError> {
        assert_eq!(delta.len(), self.n);
        self.upload(hvals, rhs)?;
        self.enqueue(factor, refine)?;
        self.dev.read(&self.work, self.sec.out, &mut self.host_out)?;
        let stats = self.outcome(&self.host_out[self.status_at..], refine)?;
        self.wide.decode(&self.host_out[..8 * self.n], &mut self.step);
        for (d, &v) in delta.iter_mut().zip(&self.step) {
            *d = M::from_f64(v);
        }
        Ok(stats)
    }

    /// The attempt's launches, replayed when a graph of them is captured,
    /// without waiting; with graphs on and none captured for this factor and
    /// refinement, the launches are recorded right after they are queued, so
    /// the recording overlaps their run.
    fn enqueue(&mut self, factor: &NativeFactor<D>, refine: Option<(usize, f64)>) -> Result<(), NativeError> {
        assert!(Arc::ptr_eq(&self.dev, factor.device()), "the factor runs on the reduction's device, so S never leaves it");
        let (max_steps, tol_rel) = refine.filter(|r| r.0 > 0).unwrap_or((0, 0.0));
        let key = self.graph_key(factor, refine);
        let replay = self.graphs && self.graph.as_ref().is_some_and(|g| g.0 == key);
        self.replayed = replay;
        if replay {
            return self.dev.replay(&self.graph.as_ref().unwrap().1);
        }
        self.launch_attempt(factor, max_steps, tol_rel)?;
        if self.graphs {
            self.graph = None;
            let this = &*self;
            let graph = self.dev.capture(|| this.launch_attempt(factor, max_steps, tol_rel))?;
            self.graph = Some((key, graph));
        }
        Ok(())
    }

    /// The launches of [`solve`](Self::solve) and nothing else, from the
    /// damped Hessian's values and the gradient already on the device (at
    /// [`h_target`](Self::h_target) and [`rhs_target`](Self::rhs_target)):
    /// the step lands at [`step_view`](Self::step_view), the status at
    /// [`status_view`](Self::status_view), for the caller to read with its
    /// own results and hand to [`outcome`](Self::outcome). Launches only, so
    /// a caller can capture them in a graph of its own (keyed by
    /// [`graph_key`](Self::graph_key)).
    pub fn launch(&self, factor: &NativeFactor<D>, refine: Option<(usize, f64)>) -> Result<(), NativeError> {
        assert!(Arc::ptr_eq(&self.dev, factor.device()), "the factor runs on the reduction's device, so S never leaves it");
        let (max_steps, tol_rel) = refine.filter(|r| r.0 > 0).unwrap_or((0, 0.0));
        self.launch_attempt(factor, max_steps, tol_rel)
    }

    /// What a captured [`launch`](Self::launch) depends on beyond the
    /// reduction's own blocks: the factor and the refinement. A graph
    /// replays only under its key.
    pub fn graph_key(&self, factor: &NativeFactor<D>, refine: Option<(usize, f64)>) -> GraphKey {
        let (max_steps, tol_rel) = refine.filter(|r| r.0 > 0).unwrap_or((0, 0.0));
        GraphKey { factor: factor.id(), max_steps, tol_bits: tol_rel.to_bits() }
    }

    /// The outcome of a solve with refinement `refine` from its status bytes
    /// (as read from [`status_view`](Self::status_view)).
    pub fn outcome(&self, status: &[u8], refine: Option<(usize, f64)>) -> Result<RefineStats, NativeSchurError> {
        // the state's words as f64, the flags after them as they are
        let mut f64_status = status[..STATUS_BYTES].to_vec();
        for c in f64_status[..host::FLAGS_AT].as_chunks_mut::<8>().0 {
            *c = self.wide.value(u64::from_le_bytes(*c)).to_le_bytes();
        }
        Ok(host::decode_status(&f64_status, &self.elim_blk, refine)?)
    }
}

/// Per kept block, the tiles of S touching it, for `k_symv_sub`: `(ptr, tiles)`
/// with block k's at `tiles[ptr[k]..ptr[k + 1]]`, each a tile index times 2,
/// plus 1 when k is the tile's column block and not its row block.
fn tiles_by_kept_block(p: &SchurPlan) -> (Vec<u32>, Vec<u32>) {
    let (nt, nk) = (p.nt as usize, p.nk as usize);
    let array = |slot: usize| &p.words[p.words[slot] as usize..][..nt];
    let (ta, tb) = (array(host::ST_A), array(host::ST_B));
    let mut ptr = vec![0u32; nk + 1];
    for t in 0..nt {
        ptr[ta[t] as usize + 1] += 1;
        if ta[t] != tb[t] {
            ptr[tb[t] as usize + 1] += 1;
        }
    }
    for k in 0..nk {
        ptr[k + 1] += ptr[k];
    }
    let mut fill = ptr.clone();
    let mut tiles = vec![0u32; ptr[nk] as usize];
    for t in 0..nt {
        let a = ta[t] as usize;
        tiles[fill[a] as usize] = 2 * t as u32;
        fill[a] += 1;
        if ta[t] != tb[t] {
            let b = tb[t] as usize;
            tiles[fill[b] as usize] = 2 * t as u32 + 1;
            fill[b] += 1;
        }
    }
    (ptr, tiles)
}

/// The `#define`s of `schur.cpp`: the block size, the width classes, and
/// the plan header's and the status's slots of [`crate::schur`].
fn defines(p: &SchurPlan) -> String {
    use host::*;
    let mut s = String::new();
    define_u32s(
        &mut s,
        &[
            ("THREADS", THREADS as usize),
            ("EW", p.ew_cap),
            ("KW", p.kw_cap),
            ("ACC", p.acc_cap),
            ("SE_DIAG_OFF", SE_DIAG_OFF),
            ("SE_W", SE_W),
            ("SE_ORIG_OFF", SE_ORIG_OFF),
            ("SE_OBS_PTR", SE_OBS_PTR),
            ("SE_DINV_OFF", SE_DINV_OFF),
            ("SE_Z_OFF", SE_Z_OFF),
            ("SO_CA_OFF", SO_CA_OFF),
            ("SO_TRANS", SO_TRANS),
            ("SO_W", SO_W),
            ("SO_E", SO_E),
            ("SO_ORIG_OFF", SO_ORIG_OFF),
            ("SO_W_OFF", SO_W_OFF),
            ("ST_VAL_OFF", ST_VAL_OFF),
            ("ST_WR", ST_WR),
            ("ST_WC", ST_WC),
            ("ST_A", ST_A),
            ("ST_B", ST_B),
            ("ST_A_OFF", ST_A_OFF),
            ("ST_B_OFF", ST_B_OFF),
            ("ST_COPY_OFF", ST_COPY_OFF),
            ("ST_PAIR_PTR", ST_PAIR_PTR),
            ("SP_OA", SP_OA),
            ("SP_OB", SP_OB),
            ("SK_OBS_PTR", SK_OBS_PTR),
            ("SK_OBS", SK_OBS),
            ("SK_S_OFF", SK_S_OFF),
            ("SK_ORIG_OFF", SK_ORIG_OFF),
            ("SK_W", SK_W),
            ("R_BNORM", ST_BNORM),
            ("R_TOL", ST_TOL),
            ("R_RNORM", ST_RNORM),
            ("F_ACTIVE", F_ACTIVE),
            ("F_STEPS", F_STEPS),
            ("F_REVERT", F_REVERT),
        ],
    );
    s
}

#[cfg(all(test, any(feature = "native-cuda", feature = "native-hip")))]
mod tests {
    use super::*;
    use crate::GpuPrecision;
    use crate::native::TestDevice;
    use crate::test_util::{ba_like, rel_residual};
    use arael_faer::schur::{SchurContext, schur_reduce, schur_symbolic};
    use arael_faer::supernodal::{SupernodalParams, SupernodalSymbolic, nd_block_order};

    type Matrix = arael_faer::bsc::SparseBlockColMat<SparseIndex, f64>;

    fn device() -> Arc<TestDevice> {
        Arc::new(TestDevice::new(0).unwrap())
    }

    fn err_max(a: &[f64], b: &[f64]) -> f64 {
        let m = b.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        a.iter().zip(b).fold(0.0f64, |e, (x, y)| e.max((x - y).abs())) / m
    }

    /// Both wide scalars: the reduction's tests run in each.
    const WIDES: [Wide; 2] = [Wide::F64, Wide::Df32];

    fn factor(dev: &Arc<TestDevice>, precision: GpuPrecision, sn: &SupernodalSymbolic, s: &SymbolicSparseBlockColMat<SparseIndex>, wide: Wide) -> NativeFactor<TestDevice> {
        NativeFactor::new(dev, precision, sn, s, &Default::default(), wide).unwrap()
    }

    /// A BA-like system, its Schur symbolic and the supernodal analysis of S.
    fn setup(np: usize, nl: usize, seed: u64) -> (Matrix, Vec<f64>, Vec<f64>, Vec<usize>, SchurSymbolic<SparseIndex>, SupernodalSymbolic) {
        let (h, dense, rhs, elim) = ba_like(np, nl, seed);
        let sym = schur_symbolic(h.symbolic(), &elim).unwrap();
        let order = nd_block_order(&sym.s);
        let sn = SupernodalSymbolic::new(&sym.s, Some(&order), &SupernodalParams::default()).unwrap();
        (h, dense, rhs, elim, sym, sn)
    }

    #[test]
    fn the_reduction_matches_the_cpu() {
        for wide in WIDES {
            the_reduction_matches_the_cpu_in(wide);
        }
    }

    fn the_reduction_matches_the_cpu_in(wide: Wide) {
        let (h, _, rhs, elim) = ba_like(40, 300, 5);
        let sym = schur_symbolic(h.symbolic(), &elim).unwrap();
        let mut s_cpu = sym.alloc_s::<f64>();
        let mut rhs_cpu = vec![0.0f64; sym.kept_size()];
        let mut ctx = SchurContext::new();
        schur_reduce(&sym, &h, &rhs, &mut ctx, &mut s_cpu, &mut rhs_cpu).unwrap();
        let dev = device();
        let mut g = NativeSchur::new(&dev, &sym, h.symbolic(), wide).unwrap();
        g.reduce(h.vals(), &rhs).unwrap();
        let (s_gpu, rhs_gpu) = g.download().unwrap();
        assert_eq!(s_gpu.len(), s_cpu.vals().len());
        let es = err_max(&s_gpu, s_cpu.vals());
        let er = err_max(&rhs_gpu, &rhs_cpu);
        assert!(es < 1e-12 && er < 1e-12, "S err/max {es}, rhs err/max {er}");
        assert!(sym.obs_trans().iter().any(|&t| t) && sym.obs_trans().iter().any(|&t| !t), "both tile orientations in the fixture");
        // the same from f32 inputs, widened on the host
        let (h32, r32): (Vec<f32>, Vec<f32>) = (h.vals().iter().map(|&v| v as f32).collect(), rhs.iter().map(|&v| v as f32).collect());
        g.reduce(&h32, &r32).unwrap();
        let (s32, _) = g.download().unwrap();
        let h64: Vec<f64> = h32.iter().map(|&v| v as f64).collect();
        let r64: Vec<f64> = r32.iter().map(|&v| v as f64).collect();
        g.reduce(&h64, &r64).unwrap();
        assert_eq!(s32, g.download().unwrap().0, "f32 inputs are their f64 twins");
    }

    #[test]
    fn the_reduced_solve_solves_the_whole_system() {
        for wide in WIDES {
            the_reduced_solve_solves_the_whole_system_in(wide);
        }
    }

    fn the_reduced_solve_solves_the_whole_system_in(wide: Wide) {
        let (h, dense, rhs, _, sym, sn) = setup(40, 300, 9);
        let n = h.symbolic().ncols();
        let dev = device();
        let mut g = NativeSchur::new(&dev, &sym, h.symbolic(), wide).unwrap();
        for (precision, refine, tol) in [(GpuPrecision::F64, None, 1e-10), (GpuPrecision::F32, Some((4usize, 1e-10f64)), 1e-8)] {
            let mut f = factor(&dev, precision, &sn, &sym.s, wide);
            let mut delta = vec![0.0f64; n];
            let stats = g.solve(&mut f, h.vals(), &rhs, refine, &mut delta).unwrap();
            let res = rel_residual(&dense, n, &delta, &rhs);
            assert!(res < tol, "{precision:?}: residual {res}");
            if refine.is_some() {
                assert!(stats.steps > 0 && !stats.short && stats.rnorm <= 1e-10 * stats.bnorm, "{stats:?}");
            } else {
                assert_eq!(stats.steps, 0);
            }
            // a second solve of the same structure
            let mut delta2 = vec![0.0f64; n];
            g.solve(&mut f, h.vals(), &rhs, refine, &mut delta2).unwrap();
            assert!(rel_residual(&dense, n, &delta2, &rhs) < tol);
        }
    }

    /// With nothing eliminated S is H: the reduction solves the whole
    /// system, refinement included.
    #[test]
    fn nothing_eliminated_solves_the_whole_system() {
        for wide in WIDES {
            nothing_eliminated_solves_the_whole_system_in(wide);
        }
    }

    fn nothing_eliminated_solves_the_whole_system_in(wide: Wide) {
        let (h, dense, rhs, _) = ba_like(30, 200, 5);
        let n = h.symbolic().ncols();
        let order = nd_block_order(h.symbolic());
        let sn = SupernodalSymbolic::new(h.symbolic(), Some(&order), &SupernodalParams::default()).unwrap();
        let dev = device();
        let mut g = NativeSchur::whole(&dev, h.symbolic(), wide).unwrap();
        for (precision, refine, tol) in [(GpuPrecision::F64, None, 1e-10), (GpuPrecision::F32, Some((4usize, 1e-10f64)), 1e-8)] {
            let mut f = factor(&dev, precision, &sn, h.symbolic(), wide);
            for _ in 0..2 {
                let mut delta = vec![0.0f64; n];
                let stats = g.solve(&mut f, h.vals(), &rhs, refine, &mut delta).unwrap();
                let res = rel_residual(&dense, n, &delta, &rhs);
                assert!(res < tol, "{precision:?}: residual {res}");
                if refine.is_some() {
                    assert!(stats.steps > 0 && !stats.short, "{stats:?}");
                }
            }
        }
    }

    /// The refinement decides on the device: it stops once below the
    /// tolerance, launched steps after that doing nothing; with a tolerance
    /// it cannot reach it stops at the first step that does not reduce the
    /// residual, and undoes that step.
    #[test]
    fn the_refinement_stops_on_the_device() {
        for wide in WIDES {
            the_refinement_stops_on_the_device_in(wide);
        }
    }

    fn the_refinement_stops_on_the_device_in(wide: Wide) {
        let (h, dense, rhs, _, sym, sn) = setup(40, 300, 9);
        let n = h.symbolic().ncols();
        let dev = device();
        let mut g = NativeSchur::new(&dev, &sym, h.symbolic(), wide).unwrap();
        let mut f = factor(&dev, GpuPrecision::F32, &sn, &sym.s, wide);
        let mut delta = vec![0.0f64; n];
        let reached = g.solve(&mut f, h.vals(), &rhs, Some((12, 1e-10)), &mut delta).unwrap();
        assert!(reached.steps > 0 && reached.steps < 12 && !reached.short, "{reached:?}");
        let res = rel_residual(&dense, n, &delta, &rhs);
        assert!(res < 1e-8, "residual {res}");
        let mut delta0 = vec![0.0f64; n];
        let stalled = g.solve(&mut f, h.vals(), &rhs, Some((40, 0.0)), &mut delta0).unwrap();
        assert!(stalled.short && stalled.steps > reached.steps && stalled.steps < 40, "{stalled:?}");
        assert!(stalled.rnorm <= reached.rnorm, "{stalled:?} against {reached:?}");
        let res0 = rel_residual(&dense, n, &delta0, &rhs);
        assert!(res0 < 1e-8, "residual {res0} after the undone step");
    }

    /// `h2` and `dense2`: the diagonal of every diagonal tile scaled by 1.5.
    fn damped(h: &Matrix, dense: &[f64]) -> (Matrix, Vec<f64>) {
        let hs = h.symbolic().clone();
        let n = hs.ncols();
        let mut h2 = h.clone();
        for blk in 0..hs.nblk_cols() {
            let t = hs.col_range(blk).find(|&b| hs.blk_row(b) == blk).unwrap();
            let (w, _) = hs.block_dims(t);
            let base = hs.val_range(t).start;
            for i in 0..w {
                h2.vals_mut()[base + i * w + i] *= 1.5;
            }
        }
        let mut dense2 = dense.to_vec();
        for i in 0..n {
            dense2[i + i * n] *= 1.5;
        }
        (h2, dense2)
    }

    /// The second solve with a factor and a refinement setting replays the
    /// graph the first captured, against new values, and lands exactly where
    /// the launched kernels land; a new factor or another refinement setting
    /// is captured anew.
    #[test]
    fn replayed_solves_match_launched_ones() {
        for wide in WIDES {
            replayed_solves_match_launched_ones_in(wide);
        }
    }

    fn replayed_solves_match_launched_ones_in(wide: Wide) {
        let (h, dense, rhs, _, sym, sn) = setup(40, 300, 9);
        let n = h.symbolic().ncols();
        let dev = device();
        // another damping: the replay must see the values written before it
        let (h2, dense2) = damped(&h, &dense);
        for precision in [GpuPrecision::F64, GpuPrecision::F32] {
            let refine = (precision == GpuPrecision::F32).then_some((4, 1e-10));
            let mut launched = NativeSchur::new(&dev, &sym, h.symbolic(), wide).unwrap();
            launched.use_graphs(false);
            let mut fl = factor(&dev, precision, &sn, &sym.s, wide);
            let mut g = NativeSchur::new(&dev, &sym, h.symbolic(), wide).unwrap();
            let mut f = factor(&dev, precision, &sn, &sym.s, wide);
            for (round, (hm, dm)) in [(&h, &dense), (&h2, &dense2), (&h, &dense)].into_iter().enumerate() {
                let (mut a, mut b) = (vec![0.0f64; n], vec![0.0f64; n]);
                let sa = launched.solve(&mut fl, hm.vals(), &rhs, refine, &mut a).unwrap();
                let sb = g.solve(&mut f, hm.vals(), &rhs, refine, &mut b).unwrap();
                assert!(!launched.replayed());
                assert_eq!(g.replayed(), round > 0, "{precision:?} round {round}");
                assert_eq!(sa.steps, sb.steps);
                let res = rel_residual(dm, n, &b, &rhs);
                assert!(res < 1e-8, "{precision:?} round {round}: residual {res}");
                let dx = a.iter().zip(&b).fold(0.0f64, |m, (x, y)| m.max((x - y).abs())) / a.iter().fold(0.0f64, |m, x| m.max(x.abs()));
                assert!(dx < 1e-12, "{precision:?} round {round}: replay against launches {dx}");
            }
            // another factor of the same structure, then another refinement setting
            let mut f2 = factor(&dev, precision, &sn, &sym.s, wide);
            let mut d = vec![0.0f64; n];
            g.solve(&mut f2, h.vals(), &rhs, refine, &mut d).unwrap();
            assert!(!g.replayed());
            g.solve(&mut f2, h.vals(), &rhs, Some((2, 1e-10)), &mut d).unwrap();
            assert!(!g.replayed());
            g.solve(&mut f2, h.vals(), &rhs, Some((2, 1e-10)), &mut d).unwrap();
            assert!(g.replayed());
            assert!(rel_residual(&dense, n, &d, &rhs) < 1e-8);
        }
    }

    /// The launches alone, from values written at the targets, read back
    /// through the views, give what `solve` gives; and a graph the caller
    /// captures of them replays to the same.
    #[test]
    fn the_launches_alone_match_solve() {
        for wide in WIDES {
            the_launches_alone_match_solve_in(wide);
        }
    }

    fn the_launches_alone_match_solve_in(wide: Wide) {
        let (h, _, rhs, _, sym, sn) = setup(30, 250, 4);
        let n = h.symbolic().ncols();
        let dev = device();
        let refine = Some((4usize, 1e-10f64));
        let mut g = NativeSchur::new(&dev, &sym, h.symbolic(), wide).unwrap();
        g.use_graphs(false);
        let mut f = factor(&dev, GpuPrecision::F32, &sn, &sym.s, wide);
        let mut want = vec![0.0f64; n];
        let stats = g.solve(&mut f, h.vals(), &rhs, refine, &mut want).unwrap();
        // fresh blocks, the inputs written at the targets
        let mut g = NativeSchur::new(&dev, &sym, h.symbolic(), wide).unwrap();
        let f = factor(&dev, GpuPrecision::F32, &sn, &sym.s, wide);
        g.upload(h.vals(), &rhs).unwrap();
        let held: Vec<f64> = h.vals().iter().map(|&v| wide.value(wide.bits(v))).collect();
        assert_eq!(g.h_values().unwrap(), held, "the values as the wide scalar holds them");
        let read_result = |g: &NativeSchur<TestDevice>| {
            let mut step = vec![0.0f64; n];
            let mut bytes = vec![0u8; 8 * n];
            let mut status = vec![0u8; STATUS_BYTES];
            let (b, at) = g.step_view();
            dev.read(b, at, &mut bytes).unwrap();
            wide.decode(&bytes, &mut step);
            let (b, at) = g.status_view();
            dev.read(b, at, &mut status).unwrap();
            (step, g.outcome(&status, refine).unwrap())
        };
        g.launch(&f, refine).unwrap();
        let (step, got) = read_result(&g);
        assert_eq!(got.steps, stats.steps);
        let dx = err_max(&step, &want);
        assert!(dx < 1e-12, "launches against solve: {dx}");
        let graph = dev.capture(|| g.launch(&f, refine)).unwrap();
        dev.replay(&graph).unwrap();
        let (step2, got2) = read_result(&g);
        assert_eq!(got2.steps, stats.steps);
        assert!(err_max(&step2, &want) < 1e-12);
        assert_eq!(g.graph_key(&f, refine), g.graph_key(&f, Some((4, 1e-10))));
        assert_ne!(g.graph_key(&f, refine), g.graph_key(&f, Some((3, 1e-10))));
    }

    /// Solves, their captures and replays included, allocate nothing: the
    /// blocks are made when the reduction and the factor are.
    #[test]
    fn solves_allocate_nothing() {
        for wide in WIDES {
            solves_allocate_nothing_in(wide);
        }
    }

    fn solves_allocate_nothing_in(wide: Wide) {
        let dev = device();
        for (np, nl, seed) in [(30, 400, 1u64), (60, 900, 2), (20, 200, 3)] {
            let (h, dense, rhs, _, sym, sn) = setup(np, nl, seed);
            let n = h.symbolic().ncols();
            let before = dev.allocations();
            let mut g = NativeSchur::new(&dev, &sym, h.symbolic(), wide).unwrap();
            assert_eq!(dev.allocations() - before, 2, "the plan block and the work block");
            let mut f = factor(&dev, GpuPrecision::F32, &sn, &sym.s, wide);
            let made = dev.allocations();
            let mut delta = vec![0.0f64; n];
            for _ in 0..3 {
                g.solve(&mut f, h.vals(), &rhs, Some((4, 1e-10)), &mut delta).unwrap();
                assert!(rel_residual(&dense, n, &delta, &rhs) < 1e-8);
            }
            assert!(g.replayed());
            assert_eq!(dev.allocations(), made, "no block made by a solve");
        }
    }

    /// A pivot that fails, in an eliminated block or in S, comes back with
    /// the solve's result and leaves the step as it was; the next solve of
    /// the same objects starts clean.
    #[test]
    fn bad_pivots_are_reported_with_the_result() {
        for wide in WIDES {
            bad_pivots_are_reported_with_the_result_in(wide);
        }
    }

    fn bad_pivots_are_reported_with_the_result_in(wide: Wide) {
        let (good, dense, rhs, elim, sym, sn) = setup(10, 30, 2);
        let n = good.symbolic().ncols();
        let asym = good.symbolic().clone();
        let diag0 = |blk: usize| asym.val_range(asym.col_range(blk).find(|&b| asym.blk_row(b) == blk).unwrap()).start;
        let dev = device();
        let mut g = NativeSchur::new(&dev, &sym, &asym, wide).unwrap();
        for precision in [GpuPrecision::F64, GpuPrecision::F32] {
            let mut f = factor(&dev, precision, &sn, &sym.s, wide);
            let refine = (precision == GpuPrecision::F32).then_some((4, 1e-10));
            // an eliminated block
            let e = elim[3];
            let mut h = good.clone();
            h.vals_mut()[diag0(e)] = -1.0;
            assert!(matches!(g.reduce(h.vals(), &rhs), Err(NativeSchurError::Eliminated { block }) if block == e));
            let mut delta = vec![7.0f64; n];
            let err = g.solve(&mut f, h.vals(), &rhs, refine, &mut delta).unwrap_err();
            assert!(matches!(err, NativeSchurError::Eliminated { block } if block == e), "{err:?}");
            assert!(delta.iter().all(|&v| v == 7.0), "the step is left as it was");
            // a kept block: S(k, k) is H(k, k) less a semidefinite sum
            let k = sym.kept[0] as usize;
            let mut h = good.clone();
            h.vals_mut()[diag0(k)] = -1e8;
            let err = g.solve(&mut f, h.vals(), &rhs, refine, &mut delta).unwrap_err();
            assert!(matches!(err, NativeSchurError::Reduced), "{err:?}");
            // and the good matrix again
            g.solve(&mut f, good.vals(), &rhs, refine, &mut delta).unwrap();
            assert!(rel_residual(&dense, n, &delta, &rhs) < 1e-8);
        }
    }

    /// At BA widths (eliminated blocks of 3, kept of 6) the per-thread
    /// arrays stay in registers: no kernel uses local memory.
    #[test]
    fn the_kernels_keep_their_arrays_in_registers() {
        for wide in WIDES {
            the_kernels_keep_their_arrays_in_registers_in(wide);
        }
    }

    fn the_kernels_keep_their_arrays_in_registers_in(wide: Wide) {
        let (h, _, _, _, sym, _) = setup(10, 30, 2);
        let g = NativeSchur::new(&device(), &sym, h.symbolic(), wide).unwrap();
        let k = &g.k;
        for (name, kernel) in [("elim", &k.elim), ("tiles", &k.tiles), ("rhs", &k.rhs), ("backsub", &k.backsub)] {
            assert_eq!(kernel.local_bytes().unwrap(), 0, "k_schur_{name}: {} registers", kernel.registers().unwrap());
        }
    }

    /// The native kernels against the CubeCL ones on the same system: the
    /// reduction (no atomics) bit for bit, the solves equal to rounding (the
    /// factor's updates and the symv add atomically, in no fixed order).
    #[cfg(all(feature = "cuda", feature = "native-cuda"))]
    #[test]
    fn native_matches_cubecl() {
        let (h, _, rhs, _, sym, sn) = setup(60, 900, 2);
        let n = h.symbolic().ncols();
        let dev = device();
        let ctx = crate::GpuContext::new(crate::GpuBackend::Cuda, 0).unwrap();
        let mut native = NativeSchur::new(&dev, &sym, h.symbolic(), Wide::F64).unwrap();
        let mut cubecl = crate::GpuSchur::new(&ctx, &sym, h.symbolic());
        native.reduce(h.vals(), &rhs).unwrap();
        cubecl.reduce(h.vals(), &rhs).unwrap();
        let ((sn_, rn), (sc, rc)) = (native.download().unwrap(), cubecl.download());
        let same = |a: &[f64], b: &[f64]| a.iter().zip(b).filter(|(x, y)| x.to_bits() == y.to_bits()).count();
        println!("reduction: S {} of {} values bit-identical, rhs {} of {}", same(&sn_, &sc), sc.len(), same(&rn, &rc), rc.len());
        assert_eq!(sn_, sc, "S bit for bit");
        assert_eq!(rn, rc, "the reduced right-hand side bit for bit");
        for (precision, refine, tol) in [(GpuPrecision::F64, None, 1e-11), (GpuPrecision::F32, Some((4usize, 1e-10f64)), 1e-8)] {
            let mut fnat = factor(&dev, precision, &sn, &sym.s, Wide::F64);
            let mut fcub = crate::GpuFactor::new(&ctx, precision, &sn, &sym.s, &Default::default());
            for round in 0..2 {
                let (mut a, mut b) = (vec![0.0f64; n], vec![0.0f64; n]);
                let sa = native.solve(&mut fnat, h.vals(), &rhs, refine, &mut a).unwrap();
                let sb = cubecl.solve(&mut fcub, h.vals(), &rhs, refine, &mut b).unwrap();
                let dx = err_max(&a, &b);
                println!(
                    "{precision:?} round {round}: step err/max {dx:.2e}, {} of {n} bit-identical; refinement steps {} / {}, rnorm {:.3e} / {:.3e}",
                    same(&a, &b),
                    sa.steps,
                    sb.steps,
                    sa.rnorm,
                    sb.rnorm
                );
                assert!(dx < tol, "{precision:?}: native against CubeCL {dx}");
                assert_eq!(native.replayed(), round > 0);
            }
        }
    }

    /// Median wall time per attempt (`solve`: the uploads of H and the
    /// gradient, the reduction, the factorization, the solve, the
    /// refinement, the back-substitution and the one read), native against
    /// CubeCL, with and without graphs:
    /// `cargo test -r -p arael-gpu --features cuda,native-cuda -- --ignored --nocapture time_schur`.
    #[cfg(all(feature = "cuda", feature = "native-cuda"))]
    #[test]
    #[ignore = "timing"]
    fn time_schur_native_against_cubecl() {
        const RUNS: usize = 40;
        fn median(mut v: Vec<f64>) -> f64 {
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        }
        let (np, nl, seed) = (80, 1200, 2);
        let (h, _, rhs, _, sym, sn) = setup(np, nl, seed);
        let n = h.symbolic().ncols();
        println!("ba_like({np}, {nl}): {n} scalars, {} kept, {} tiles of S, {} H values", sym.kept_size(), sym.s.nblocks(), h.vals().len());
        let dev = device();
        let ctx = crate::GpuContext::new(crate::GpuBackend::Cuda, 0).unwrap();
        let ms = |t: std::time::Instant| t.elapsed().as_secs_f64() * 1e3;
        for (precision, refine) in [(GpuPrecision::F64, None), (GpuPrecision::F32, Some((4usize, 1e-10f64)))] {
            for graphs in [false, true] {
                let mut native = NativeSchur::new(&dev, &sym, h.symbolic(), Wide::F64).unwrap();
                native.use_graphs(graphs);
                let mut fnat = factor(&dev, precision, &sn, &sym.s, Wide::F64);
                let mut cubecl = crate::GpuSchur::new(&ctx, &sym, h.symbolic());
                cubecl.use_graphs(graphs);
                let mut fcub = crate::GpuFactor::new(&ctx, precision, &sn, &sym.s, &Default::default());
                let (mut tn, mut tc) = (Vec::new(), Vec::new());
                let mut d = vec![0.0f64; n];
                for run in 0..RUNS + 3 {
                    let t = std::time::Instant::now();
                    native.solve(&mut fnat, h.vals(), &rhs, refine, &mut d).unwrap();
                    let a = ms(t);
                    let t = std::time::Instant::now();
                    cubecl.solve(&mut fcub, h.vals(), &rhs, refine, &mut d).unwrap();
                    let b = ms(t);
                    if run >= 3 {
                        tn.push(a);
                        tc.push(b);
                    }
                }
                println!(
                    "  {precision:?} factor{}, {}: native {:.3} ms, CubeCL {:.3} ms per attempt",
                    if refine.is_some() { " + 4 refinement steps" } else { "" },
                    if graphs { "graph replay" } else { "launches" },
                    median(tn),
                    median(tc)
                );
            }
        }
    }
}
