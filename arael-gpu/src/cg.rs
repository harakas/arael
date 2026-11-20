//! [`GpuCg`]: preconditioned conjugate gradients on the reduced system of a
//! [`GpuSchur`], on its device: the solver of the reduced system that takes
//! the place of a [`GpuFactor`](crate::GpuFactor). No factor, so no fill and
//! no factor memory; the step is inexact, to [`CgParams::tol`].
//!
//! The iteration is `arael_faer::cg::solve`'s, with its block-Jacobi
//! preconditioner (the Cholesky factor of each diagonal tile of S), its
//! stopping rule and its settings ([`CgParams`]); the vectors are in the
//! reduction's scalar ([`GpuSchur::reduced`]), the dot products in the
//! route's wide scalar. The kernels are `kernels/cg.cpp`.
//!
//! The number of iterations is data, so the scalars and the decision to
//! stop stay on the device: the kernels of an iteration return at once when
//! the solve has stopped. A round queues the launches of some iterations
//! (replays of one captured graph of [`GpuCgParams::graph_iters`]) and
//! reads the status, the one wait of the round; a solve takes as many
//! rounds as it needs, usually one, sized from the solve before it.
//!
//! ```ignore
//! let mut g = GpuSchur::new(&dev, &sym, h.symbolic(), Wide::F64)?;
//! let mut cg = GpuCg::new(&g, &CgParams::default(), &GpuCgParams::default())?;
//! let stats = cg.solve(&mut g, h.vals(), &rhs, &mut delta)?;
//! ```

use std::sync::Arc;

use arael_faer::cg::CgParams;
use arael_faer::schur::SchurReal;

use crate::schur::{F_ELIM, F_FACTOR, GpuSchur, GpuSchurError, PLANE, ReducedSystem, SHARED, THREADS};
use crate::{Arg, Device, GpuError, Launch, Layout, Scope, Section, WIDE, Wide, define_u32s};

/// The kernels, compiled after [`SHARED`] with the reduction's #defines.
const SOURCE: &str = include_str!("../kernels/cg.cpp");

// The words of a kept block's record, written on the device once per
// structure (`k_cg_blocks`).
/// Its offset in the reduced vector.
const KB_OFF: usize = 0;
const KB_W: usize = 1;
/// The value offset in s of its diagonal tile.
const KB_DIAG: usize = 2;
const KB_REC: usize = 3;

// The state on the device: wide words, then the flags, read in one copy.
/// `r^T M^-1 r` of the last iteration.
const CG_RHO: usize = 0;
/// Its starting value.
const CG_RHO0: usize = 1;
/// `tol` times the starting value.
const CG_TARGET: usize = 2;
const CG_ALPHA: usize = 3;
const CG_BETA: usize = 4;
const N_STATE: usize = 8;
/// More iterations are due: the gate of an iteration's kernels.
const CF_ACTIVE: usize = 0;
const CF_ITERS: usize = 1;
const CF_CONVERGED: usize = 2;
/// `p^T S p` was not positive.
const CF_BREAKDOWN: usize = 3;
/// This iteration recomputes the residual ([`CgParams::restart_every`]).
const CF_RESTART: usize = 4;
const N_FLAGS: usize = 8;
/// Bytes of a solve's status at [`GpuCg::status_view`], for
/// [`GpuCg::status`].
pub const CG_STATUS_BYTES: usize = 8 * N_STATE + 4 * N_FLAGS;

/// The most warps a kept block takes in the product with S: a block of
/// 1024 threads.
const MAX_MATVEC_WARPS: u32 = 32;

/// How [`GpuCg`] queues its iterations and cuts its product with S.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuCgParams {
    /// Iterations in the captured graph a round replays.
    pub graph_iters: u32,
    /// Iterations queued before the first status read of the first solve;
    /// a later solve starts from the count of the solve before it.
    pub first_iters: u32,
    /// Warps (32 threads) a kept block takes in the product with S, 1 to
    /// 32; `None` takes [`Self::matvec_warps_for`]'s. Kept blocks wider
    /// than 32 columns take one.
    pub matvec_warps: Option<u32>,
}

impl GpuCgParams {
    /// [`matvec_warps`](Self::matvec_warps), unset: from the tile entries
    /// the `kept_blocks` kept blocks have in all (`tile_entries`) and
    /// their width rounded up (`kw_cap`, at most 32), by
    /// [`kept_warps_for`](crate::schur::kept_warps_for), up to 8 (more
    /// measured no faster for the product).
    ///
    /// ```ignore
    /// // 372 cameras of 9 parameters, 54692 tile entries: 8 warps
    /// assert_eq!(GpuCgParams::default().matvec_warps_for(54692, 372, 16), 8);
    /// ```
    pub fn matvec_warps_for(&self, tile_entries: usize, kept_blocks: usize, kw_cap: usize) -> u32 {
        self.matvec_warps.unwrap_or_else(|| crate::schur::kept_warps_for(tile_entries, kept_blocks, kw_cap, 8))
    }
}

impl Default for GpuCgParams {
    fn default() -> Self {
        GpuCgParams { graph_iters: 8, first_iters: 16, matvec_warps: None }
    }
}

/// Where a solve stands, from its status ([`GpuCg::status`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CgStatus {
    /// More iterations are due: the solve has neither converged nor
    /// stopped.
    pub active: bool,
    /// Iterations run, each one product with S.
    pub iters: usize,
    /// Whether the tolerance was met.
    pub converged: bool,
    /// Whether `p^T S p` was not positive: S is not positive definite along
    /// a direction (or a value is not a number), and the solve stopped
    /// with the iterate it had.
    pub breakdown: bool,
    /// `r^T M^-1 r` relative to its starting value.
    pub relative_residual: f64,
}

/// What one solve took ([`GpuCg::run`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuCgStats {
    /// Iterations run, each one product with S.
    pub iters: usize,
    /// Whether the tolerance was met. `false` at the iteration cap or after
    /// a breakdown; the solution is then the best iterate.
    pub converged: bool,
    /// [`CgStatus::breakdown`].
    pub breakdown: bool,
    /// `r^T M^-1 r` at the end, relative to its starting value.
    pub relative_residual: f64,
    /// Status reads: the waits the iterations took.
    pub rounds: usize,
    /// Iterations queued, those launched after the solve had stopped
    /// included (they do nothing): `iters` of them ran.
    pub queued: usize,
}

struct Kernels<K> {
    blocks: K,
    precond: K,
    start: K,
    begin: K,
    matvec: K,
    alpha: K,
    update: K,
    apply: K,
    rho: K,
    direction: K,
}

/// Byte offsets of the work block's sections.
struct Sections {
    kb: usize,
    mfac: usize,
    z: usize,
    p: usize,
    q: usize,
    part: usize,
    /// The state's words, then the flags.
    status: usize,
    bytes: usize,
}

/// Conjugate gradients on the reduced system of one [`GpuSchur`]: the
/// preconditioner's factors, the iteration's vectors and its state on the
/// device, made in [`new`](Self::new).
pub struct GpuCg<D: Device> {
    /// The launches of [`GpuCgParams::graph_iters`] iterations as a graph;
    /// first, so it is dropped before the blocks it records.
    body: Option<D::Graph>,
    graphs: bool,
    dev: Arc<D>,
    /// The reduction this was built over ([`GpuSchur::id`]).
    schur: u64,
    k: Kernels<D::Kernel>,
    work: D::Block,
    sec: Sections,
    wide: Wide,
    cg: CgParams,
    params: GpuCgParams,
    /// The iteration cap: [`CgParams::max_iters`], or the dimension.
    cap: u32,
    /// Threads a block of the product's kernel.
    matvec_threads: u32,
    nk: u32,
    nks: u32,
    /// The iterations of the solve before this one.
    last: Option<u32>,
    /// The status as a round reads it.
    host: [u8; CG_STATUS_BYTES],
}

impl<D: Device> GpuCg<D> {
    /// Conjugate gradients under `cg` on the reduced system of `gs`, on
    /// its device, with the vectors, the preconditioner's factors and the
    /// kernels.
    ///
    /// # Panics
    ///
    /// When `gs` keeps no block, or [`GpuCgParams::graph_iters`] is 0.
    pub fn new(gs: &GpuSchur<D>, cg: &CgParams, params: &GpuCgParams) -> Result<Self, GpuError> {
        let dev = gs.device();
        let rs = gs.reduced_system();
        assert!(rs.nk > 0, "conjugate gradients solve a reduced system, and this reduction keeps no block");
        assert!(params.graph_iters > 0, "a graph of no iteration");
        let _scope = (Scope::new(&**dev, "setup"), Scope::new(&**dev, "cg"));
        let reduced = gs.reduced();
        let elem = reduced.bytes();
        let (nk, nks) = (rs.nk as usize, rs.nks);
        // a block's factor in the widest kept block's square
        let mstride = rs.kw_max * rs.kw_max;
        let mut defines = rs.defines.to_string();
        define_u32s(
            &mut defines,
            &[
                ("KB_REC", KB_REC),
                ("KB_OFF", KB_OFF),
                ("KB_W", KB_W),
                ("KB_DIAG", KB_DIAG),
                ("MSTRIDE", mstride),
                ("CG_RHO", CG_RHO),
                ("CG_RHO0", CG_RHO0),
                ("CG_TARGET", CG_TARGET),
                ("CG_ALPHA", CG_ALPHA),
                ("CG_BETA", CG_BETA),
                ("CF_ACTIVE", CF_ACTIVE),
                ("CF_ITERS", CF_ITERS),
                ("CF_CONVERGED", CF_CONVERGED),
                ("CF_BREAKDOWN", CF_BREAKDOWN),
                ("CF_RESTART", CF_RESTART),
                ("F_ELIM", F_ELIM),
                ("F_FACTOR", F_FACTOR),
            ],
        );
        // the product's warps a kept block; blocks wider than a plane's
        // lanes take one warp, its lanes over the tile entries
        let warps = if rs.kw_cap <= PLANE as usize { params.matvec_warps_for(rs.kt_entries, nk, rs.kw_cap) } else { 1 };
        assert!((1..=MAX_MATVEC_WARPS).contains(&warps), "{warps} warps a kept block in the product, 1 to {MAX_MATVEC_WARPS}");
        define_u32s(&mut defines, &[("MWARPS", warps as usize)]);
        let matvec_threads = warps * PLANE;
        let source = format!("{WIDE}\n{SHARED}\n{SOURCE}");
        let entries = ["k_cg_blocks", "k_cg_precond", "k_cg_start", "k_cg_begin", "k_cg_matvec", "k_cg_alpha", "k_cg_update", "k_cg_apply", "k_cg_rho", "k_cg_direction"];
        let [blocks, precond, start, begin, matvec, alpha, update, apply, rho, direction]: [D::Kernel; 10] =
            dev.compile_many(&source, &defines, &entries)?.try_into().ok().expect("a kernel per entry point");
        let k = Kernels { blocks, precond, start, begin, matvec, alpha, update, apply, rho, direction };
        let mut l = Layout::new();
        let sec = Sections {
            kb: l.take(4 * KB_REC * nk),
            mfac: l.take(nk * mstride * elem),
            z: l.take(nks * elem),
            p: l.take(nks * elem),
            q: l.take(nks * elem),
            part: l.take(8 * nk),
            status: l.take(CG_STATUS_BYTES),
            bytes: l.bytes(),
        };
        let work = dev.alloc(sec.bytes)?;
        // the kept blocks' records, written on the device from the plan
        let args = [
            Arg::Block(rs.plan, 0),
            Arg::Bytes(bytemuck::cast_slice(rs.header)),
            Arg::Block(rs.plan, rs.kt_ptr),
            Arg::Block(rs.expanded, rs.kt),
            Arg::Block(&work, sec.kb),
            Arg::U32(rs.nk),
        ];
        dev.launch(&k.blocks, Launch::linear(rs.nk, THREADS), &args)?;
        dev.describe(&work, "cg.work", &|| {
            let wide = gs.wide();
            let r = |name: &str, at: usize, count: usize| Section::new(name, at, count * elem, elem, if elem == 4 { "f32" } else { wide.name() }).role("R");
            vec![
                Section::u32s("kept_blocks", sec.kb, KB_REC * nk).record(KB_REC).doc("per kept block: its offset in the reduced vector, its width, the value offset in s of its diagonal tile"),
                r("mfac", sec.mfac, nk * mstride).doc("the preconditioner: per kept block the Cholesky factor of its diagonal tile of S"),
                r("z", sec.z, nks).doc("the preconditioned residual"),
                r("p", sec.p, nks).doc("the search direction"),
                r("q", sec.q, nks).doc("S times the search direction"),
                Section::wide("parts", sec.part, nk, wide).doc("per kept block: its part of the dot product being summed"),
                Section::wide("state", sec.status, N_STATE, wide).doc("rho, its starting value, the target, alpha, beta"),
                Section::u32s("flags", sec.status + 8 * N_STATE, N_FLAGS).doc("active, iterations, converged, breakdown, restart"),
            ]
        });
        let cap = if cg.max_iters == 0 { nks } else { cg.max_iters };
        Ok(GpuCg {
            body: None,
            graphs: true,
            dev: dev.clone(),
            schur: gs.id(),
            k,
            work,
            sec,
            wide: gs.wide(),
            cg: *cg,
            params: *params,
            cap: cap.min(u32::MAX as usize) as u32,
            matvec_threads,
            nk: rs.nk,
            nks: nks as u32,
            last: None,
            host: [0; CG_STATUS_BYTES],
        })
    }

    /// The settings the solves run under.
    pub fn params(&self) -> &CgParams {
        &self.cg
    }

    /// Whether [`queue`](Self::queue) captures the iterations' launches as
    /// a graph and replays it (the default).
    pub fn use_graphs(&mut self, on: bool) {
        self.graphs = on;
        if !on {
            self.body = None;
        }
    }

    /// `gs`'s reduced system; `gs` is the reduction this was built over.
    fn system<'a>(&self, gs: &'a GpuSchur<D>) -> ReducedSystem<'a, D> {
        assert_eq!(gs.id(), self.schur, "conjugate gradients run on the reduction they were built over");
        gs.reduced_system()
    }

    /// Byte offset of flag `f` in the work block.
    fn flag_at(&self, f: usize) -> usize {
        self.sec.status + 8 * N_STATE + 4 * f
    }

    /// The launches before the iterations, after `gs`'s reduction
    /// ([`GpuSchur::launch_reduce`]): the preconditioner from S's diagonal
    /// tiles, then `x = 0`, the first residual and direction, and the
    /// state. A diagonal tile that is not positive definite fails the
    /// solve as a failed factorization does
    /// ([`GpuSchurError::Reduced`]).
    pub fn launch_start(&self, gs: &GpuSchur<D>) -> Result<(), GpuError> {
        let rs = self.system(gs);
        let _scope = Scope::new(&*self.dev, "cg start");
        let (s, nk) = (&self.sec, Arg::U32(self.nk));
        let w = |at: usize| Arg::Block(&self.work, at);
        let g = |at: usize| Arg::Block(rs.work, at);
        let per_block = Launch::linear(self.nk, THREADS);
        self.dev.launch(&self.k.precond, per_block, &[w(s.kb), g(rs.s), w(s.mfac), g(rs.flags + 4 * F_FACTOR), nk])?;
        self.dev.launch(&self.k.start, per_block, &[w(s.kb), w(s.mfac), g(rs.b), g(rs.x), g(rs.r), w(s.z), w(s.p), w(s.part), nk])?;
        let begin = [w(s.part), w(s.status), w(self.flag_at(0)), g(rs.flags), nk, self.wide.arg(self.cg.tol)];
        self.dev.launch(&self.k.begin, Launch::linear(1, THREADS), &begin)
    }

    /// The launches of `count` iterations, each doing nothing once the
    /// solve has stopped: launches only, so a caller can capture them.
    pub fn launch_iterations(&self, gs: &GpuSchur<D>, count: u32) -> Result<(), GpuError> {
        let rs = self.system(gs);
        let _scope = Scope::new(&*self.dev, "cg");
        let (s, nk) = (&self.sec, Arg::U32(self.nk));
        let w = |at: usize| Arg::Block(&self.work, at);
        let g = |at: usize| Arg::Block(rs.work, at);
        let (fl, st) = (w(self.flag_at(0)), w(s.status));
        let per_block = Launch::linear(self.nk, THREADS);
        let one = Launch::linear(1, THREADS);
        let restart = self.cg.restart_every.min(u32::MAX as usize) as u32;
        for _ in 0..count {
            let matvec = [fl, Arg::Block(rs.plan, rs.kt_ptr), Arg::Block(rs.expanded, rs.kt), w(s.kb), g(rs.s), w(s.p), w(s.q), w(s.part)];
            self.dev.launch(&self.k.matvec, Launch { grid: [self.nk, 1, 1], block: [self.matvec_threads, 1, 1], shared_bytes: 0 }, &matvec)?;
            self.dev.launch(&self.k.alpha, one, &[fl, w(s.part), st, nk, Arg::U32(restart)])?;
            self.dev.launch(&self.k.update, per_block, &[fl, st, w(s.kb), w(s.mfac), w(s.p), w(s.q), g(rs.x), g(rs.r), w(s.z), w(s.part), nk])?;
            if restart != 0 {
                // r = b - S x on the iterations that recompute it, then z
                // and the parts of r.z from it
                let gate = w(self.flag_at(CF_RESTART));
                gs.launch_residual(gate)?;
                self.dev.launch(&self.k.apply, per_block, &[gate, w(s.kb), w(s.mfac), g(rs.r), w(s.z), w(s.part), nk])?;
            }
            self.dev.launch(&self.k.rho, one, &[fl, w(s.part), st, nk, Arg::U32(self.cap)])?;
            self.dev.launch(&self.k.direction, Launch::linear(self.nks, THREADS), &[fl, st, w(s.z), w(s.p), Arg::U32(self.nks)])?;
        }
        Ok(())
    }

    /// Queue `graphs` times the launches of
    /// [`GpuCgParams::graph_iters`] iterations, without waiting: replays of
    /// their graph, captured at the first call (which launches them once
    /// and records them while they run).
    pub fn queue(&mut self, gs: &GpuSchur<D>, graphs: u32) -> Result<(), GpuError> {
        let per = self.params.graph_iters;
        if !self.graphs {
            return self.launch_iterations(gs, graphs * per);
        }
        let mut replays = graphs;
        if self.body.is_none() && graphs > 0 {
            self.launch_iterations(gs, per)?;
            let this = &*self;
            let body = self.dev.capture(|| this.launch_iterations(gs, per))?;
            self.body = Some(body);
            replays -= 1;
        }
        for _ in 0..replays {
            self.dev.replay(self.body.as_ref().expect("the iterations' graph"))?;
        }
        Ok(())
    }

    /// Where the solve's status is ([`CG_STATUS_BYTES`] bytes), for a
    /// caller that reads it with results of its own and hands it to
    /// [`status`](Self::status).
    pub fn status_view(&self) -> (&D::Block, usize) {
        (&self.work, self.sec.status)
    }

    /// Where a solve stands, from its status bytes (as read from
    /// [`status_view`](Self::status_view)).
    pub fn status(&self, bytes: &[u8]) -> CgStatus {
        let word = |i: usize| self.wide.value(u64::from_le_bytes(bytes[8 * i..8 * i + 8].try_into().expect("eight bytes")));
        let flag = |f: usize| bytemuck::pod_read_unaligned::<u32>(&bytes[8 * N_STATE + 4 * f..8 * N_STATE + 4 * f + 4]);
        let (rho, rho0) = (word(CG_RHO), word(CG_RHO0));
        CgStatus {
            active: flag(CF_ACTIVE) != 0,
            iters: flag(CF_ITERS) as usize,
            converged: flag(CF_CONVERGED) != 0,
            breakdown: flag(CF_BREAKDOWN) != 0,
            // a right-hand side of zero is solved as it stands
            relative_residual: if rho0 > 0.0 { rho / rho0 } else { 0.0 },
        }
    }

    /// The graphs a solve queues before its first status read: enough for
    /// the iterations of the solve before it and a quarter more, or for
    /// [`GpuCgParams::first_iters`].
    pub fn first_graphs(&self) -> u32 {
        let guess = self.last.map_or(self.params.first_iters, |l| l.saturating_add((l / 4).max(2)));
        guess.div_ceil(self.params.graph_iters).max(1)
    }

    /// The graphs a round queues after `queued` of them left the solve
    /// unfinished: as many again. An iteration launched after the stop
    /// costs little against a round's wait (measured on an NVIDIA discrete
    /// GPU: about 3 us against 100), so a miss doubles what is queued.
    pub fn more_graphs(&self, queued: u32) -> u32 {
        queued.max(1)
    }

    /// The iterations a finished solve took, the next one's first guess.
    pub fn note_iterations(&mut self, iters: usize) {
        self.last = Some(iters.min(u32::MAX as usize) as u32);
    }

    /// Run the iterations to the end of the solve, after
    /// [`launch_start`](Self::launch_start): rounds of queued iterations
    /// and one status read each (the wait), until the solve has converged,
    /// reached its cap or stopped. The solution is in `gs`, where its
    /// back-substitution reads it ([`GpuSchur::launch_backsub`]).
    pub fn run(&mut self, gs: &GpuSchur<D>) -> Result<GpuCgStats, GpuError> {
        let (mut graphs, mut queued, mut rounds) = (self.first_graphs(), 0u32, 0usize);
        loop {
            self.queue(gs, graphs)?;
            queued = queued.saturating_add(graphs);
            rounds += 1;
            let mut host = self.host;
            self.dev.read(&self.work, self.sec.status, &mut host)?;
            self.host = host;
            let st = self.status(&host);
            if !st.active {
                self.note_iterations(st.iters);
                let queued = queued as usize * self.params.graph_iters as usize;
                return Ok(GpuCgStats { iters: st.iters, converged: st.converged, breakdown: st.breakdown, relative_residual: st.relative_residual, rounds, queued });
            }
            graphs = self.more_graphs(queued);
        }
    }

    /// One damped solve on the device from host values: `gs` forms S and
    /// the reduced right-hand side from the Hessian's tile values `hvals`
    /// and the gradient `rhs`, the kept blocks are solved for by conjugate
    /// gradients, the eliminated blocks recovered, and the full step read
    /// into `delta` (left as it was by a failed solve).
    ///
    /// An eliminated block that is not positive definite is
    /// [`GpuSchurError::Eliminated`], a diagonal tile of S that is not
    /// [`GpuSchurError::Reduced`]: the caller raises the damping.
    pub fn solve<M: SchurReal + 'static>(&mut self, gs: &mut GpuSchur<D>, hvals: &[M], rhs: &[M], delta: &mut [M]) -> Result<GpuCgStats, GpuSchurError> {
        gs.start_solve(hvals, rhs)?;
        gs.launch_reduce()?;
        self.launch_start(gs)?;
        let stats = self.run(gs)?;
        gs.launch_backsub()?;
        gs.finish_solve(None, delta)?;
        Ok(stats)
    }
}

#[cfg(all(test, any(feature = "cuda", feature = "hip", feature = "cpu", all(feature = "metal", target_vendor = "apple"))))]
mod tests {
    use super::*;
    use crate::TestDevice;
    use crate::plan::SparseIndex;
    use crate::test_util::{ba_like, grid_spd, rel_residual, wide, wides};
    use crate::{GpuSchurParams, Values};
    use arael_faer::bsc::{SparseBlockColMat, SymbolicSparseBlockColMat};
    use arael_faer::cg::{BlockJacobi, CgStats, CgWorkspace};
    use arael_faer::schur::{SchurContext, SchurSymbolic, schur_backsub, schur_reduce, schur_symbolic};

    type Matrix = SparseBlockColMat<SparseIndex, f64>;

    fn device() -> Arc<TestDevice> {
        Arc::new(TestDevice::new(0).unwrap())
    }

    fn err_max(a: &[f64], b: &[f64]) -> f64 {
        let m = b.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        a.iter().zip(b).fold(0.0f64, |e, (x, y)| e.max((x - y).abs())) / m
    }

    fn tight() -> CgParams {
        CgParams { tol: 1e-20, ..Default::default() }
    }

    /// A BA-like system and its Schur symbolic.
    fn setup(np: usize, nl: usize, seed: u64) -> (Matrix, Vec<f64>, Vec<f64>, Vec<usize>, SchurSymbolic<SparseIndex>) {
        let (h, dense, rhs, elim) = ba_like(np, nl, seed);
        let sym = schur_symbolic(h.symbolic(), &elim).unwrap();
        (h, dense, rhs, elim, sym)
    }

    /// The CPU's route in the scalar `T`: S formed, conjugate gradients
    /// under `cg`, the back-substitution; the full step and the stats.
    fn cpu_solve<T: SchurReal>(h: &Matrix, rhs: &[f64], sym: &SchurSymbolic<SparseIndex>, cg: &CgParams) -> (Vec<f64>, CgStats) {
        let h = SparseBlockColMat::new(h.symbolic().clone(), h.vals().iter().map(|&v| T::from_f64(v)).collect::<Vec<T>>());
        let rhs: Vec<T> = rhs.iter().map(|&v| T::from_f64(v)).collect();
        let mut s = sym.alloc_s::<T>();
        let mut b = vec![T::ZERO; sym.kept_size()];
        let mut ctx = SchurContext::new();
        schur_reduce(sym, &h, &rhs, &mut ctx, &mut s, &mut b).unwrap();
        let m = BlockJacobi::build(&s).unwrap();
        let mut x = vec![T::ZERO; b.len()];
        let stats = arael_faer::cg::solve(|u, v| s.mul_symmetric_upper(u, v), &m, &b, &mut x, cg, &mut CgWorkspace::default());
        let mut step = vec![T::ZERO; rhs.len()];
        schur_backsub(sym, &h, &rhs, &x, &mut ctx, &mut step).unwrap();
        (step.iter().map(|v| v.to_f64()).collect(), stats)
    }

    /// The CPU's conjugate gradients on the whole of `h`.
    fn cpu_whole(h: &Matrix, rhs: &[f64], cg: &CgParams) -> (Vec<f64>, CgStats) {
        let m = BlockJacobi::build(h).unwrap();
        let mut x = vec![0.0; rhs.len()];
        let stats = arael_faer::cg::solve(|u, v| h.mul_symmetric_upper(u, v), &m, rhs, &mut x, cg, &mut CgWorkspace::default());
        (x, stats)
    }

    /// The kept blocks' solution as the device holds it.
    fn kept_solution(g: &GpuSchur<TestDevice>) -> Vec<f64> {
        let (rs, r) = (g.reduced_system(), g.reduced());
        let mut bytes = vec![0u8; r.bytes() * rs.nks];
        g.device().read(rs.work, rs.x, &mut bytes).unwrap();
        let mut v = vec![0.0f64; rs.nks];
        r.decode(&bytes, &mut v);
        v
    }

    /// On each wide scalar, the device's solve is the CPU's: to a tight
    /// tolerance the whole system's solution, and at the default tolerance
    /// the CPU's inexact step in the CPU's iterations (the sums are taken
    /// in another order, so a count may differ by a few).
    #[test]
    fn the_device_solves_as_the_cpu() {
        let (h, dense, rhs, _, sym) = setup(40, 300, 5);
        let n = h.symbolic().ncols();
        let dev = device();
        for wide in wides(&*dev) {
            let mut g = GpuSchur::new(&dev, &sym, h.symbolic(), wide).unwrap();
            for cg in [tight(), CgParams::default()] {
                let (want, cpu) = cpu_solve::<f64>(&h, &rhs, &sym, &cg);
                let mut solver = GpuCg::new(&g, &cg, &GpuCgParams::default()).unwrap();
                let mut delta = vec![0.0f64; n];
                let stats = solver.solve(&mut g, h.vals(), &rhs, &mut delta).unwrap();
                assert_eq!(stats.converged, cpu.converged, "{wide:?} tol {}: {stats:?} against {cpu:?}", cg.tol);
                assert!(stats.iters.abs_diff(cpu.iters) <= 2 && stats.iters > 0, "{wide:?} tol {}: {stats:?} against {cpu:?}", cg.tol);
                assert!(!stats.breakdown && stats.rounds >= 1);
                let e = err_max(&delta, &want);
                assert!(e < 1e-7, "{wide:?} tol {}: the step against the CPU's, err/max {e}", cg.tol);
                if cg.tol < 1e-12 {
                    let res = rel_residual(&dense, n, &delta, &rhs);
                    assert!(res < 1e-9, "{wide:?}: residual {res}");
                } else {
                    assert!(stats.converged && stats.relative_residual <= cg.tol, "{stats:?}");
                }
            }
        }
    }

    /// The f32 system (`GpuSchurParams::f32_reduction`): S, the vectors and
    /// the products in f32 with wide dot products, as the CPU's f32 route.
    /// From an f32 root's values, and from wide values rounded to f32 on
    /// the device (an f64 solve under mixed precision), whose step comes
    /// back in the wide scalar.
    #[test]
    fn the_f32_system_solves_as_the_cpus_f32_route() {
        let (h, dense, rhs, _, sym) = setup(40, 300, 5);
        let n = h.symbolic().ncols();
        let dev = device();
        let cg = CgParams { tol: 1e-10, ..Default::default() };
        let (want, cpu) = cpu_solve::<f32>(&h, &rhs, &sym, &cg);
        let (h32, r32): (Vec<f32>, Vec<f32>) = (h.vals().iter().map(|&v| v as f32).collect(), rhs.iter().map(|&v| v as f32).collect());
        for wide in wides(&*dev) {
            let params = GpuSchurParams { f32_reduction: true, ..Default::default() };
            let mut g = GpuSchur::with_params(&dev, &sym, h.symbolic(), Values::f32(wide), &params).unwrap();
            let mut solver = GpuCg::new(&g, &cg, &GpuCgParams::default()).unwrap();
            let mut delta = vec![0.0f32; n];
            let stats = solver.solve(&mut g, &h32, &r32, &mut delta).unwrap();
            let delta: Vec<f64> = delta.iter().map(|&v| v as f64).collect();
            assert!(stats.iters > 0 && stats.iters.abs_diff(cpu.iters) <= 4, "{wide:?}: {stats:?} against {cpu:?}");
            let e = err_max(&delta, &want);
            assert!(e < 1e-3, "{wide:?}: the f32 step against the CPU's f32 one, err/max {e}");
            let res = rel_residual(&dense, n, &delta, &rhs);
            assert!(res < 1e-3, "{wide:?}: residual {res}");
            // wide values, the system in f32
            let mut g = GpuSchur::with_params(&dev, &sym, h.symbolic(), wide, &params).unwrap();
            assert!(g.f32_reduction() && !g.values().is_f32());
            let mut solver = GpuCg::new(&g, &cg, &GpuCgParams::default()).unwrap();
            let mut delta = vec![0.0f64; n];
            let stats = solver.solve(&mut g, h.vals(), &rhs, &mut delta).unwrap();
            assert!(stats.iters > 0 && stats.iters.abs_diff(cpu.iters) <= 4, "{wide:?}, wide values: {stats:?} against {cpu:?}");
            let e = err_max(&delta, &want);
            assert!(e < 1e-3, "{wide:?}, wide values: the step against the CPU's f32 one, err/max {e}");
            assert!(rel_residual(&dense, n, &delta, &rhs) < 1e-3, "{wide:?}, wide values");
        }
    }

    /// The solve stops at the iteration cap with the iterate it has, as the
    /// CPU's; a right-hand side of zero is solved in no iteration.
    #[test]
    fn the_cap_and_a_zero_right_hand_side_end_the_solve() {
        let (h, _, rhs, _, sym) = setup(40, 300, 5);
        let n = h.symbolic().ncols();
        let dev = device();
        let mut g = GpuSchur::new(&dev, &sym, h.symbolic(), wide(&*dev)).unwrap();
        let cg = CgParams { tol: 1e-20, max_iters: 3, ..Default::default() };
        let (want, cpu) = cpu_solve::<f64>(&h, &rhs, &sym, &cg);
        let mut solver = GpuCg::new(&g, &cg, &GpuCgParams::default()).unwrap();
        let mut delta = vec![0.0f64; n];
        let stats = solver.solve(&mut g, h.vals(), &rhs, &mut delta).unwrap();
        assert_eq!((stats.iters, stats.converged, stats.breakdown), (3, false, false), "{stats:?}");
        assert_eq!((cpu.iters, cpu.converged), (3, false));
        assert!((stats.relative_residual / cpu.relative_residual - 1.0).abs() < 1e-8, "{stats:?} against {cpu:?}");
        assert!(err_max(&delta, &want) < 1e-10);
        let zero = vec![0.0f64; n];
        let stats = solver.solve(&mut g, h.vals(), &zero, &mut delta).unwrap();
        assert_eq!((stats.iters, stats.converged, stats.relative_residual), (0, true, 0.0), "{stats:?}");
        assert!(delta.iter().all(|&v| v == 0.0));
    }

    /// The iterations are queued in rounds: graphs of one iteration and of
    /// many reach the same solution in the same iterations, the small ones
    /// in several rounds, and a second solve sizes its first round from the
    /// first one's count. Launched and replayed iterations agree.
    #[test]
    fn rounds_of_any_size_reach_the_same_solution() {
        let (h, _, rhs, _, sym) = setup(40, 300, 5);
        let n = h.symbolic().ncols();
        let dev = device();
        let mut g = GpuSchur::new(&dev, &sym, h.symbolic(), wide(&*dev)).unwrap();
        let cg = tight();
        let run = |g: &mut GpuSchur<TestDevice>, params: GpuCgParams, graphs: bool| {
            let mut solver = GpuCg::new(g, &cg, &params).unwrap();
            solver.use_graphs(graphs);
            let mut delta = vec![0.0f64; n];
            let first = solver.solve(g, h.vals(), &rhs, &mut delta).unwrap();
            let mut again = vec![0.0f64; n];
            let second = solver.solve(g, h.vals(), &rhs, &mut again).unwrap();
            assert!(err_max(&again, &delta) < 1e-13, "{params:?}: a second solve");
            (delta, first, second)
        };
        let (long, long_stats, _) = run(&mut g, GpuCgParams { graph_iters: 256, first_iters: 256, ..Default::default() }, true);
        assert!(long_stats.converged && long_stats.iters < 256 && long_stats.rounds == 1, "{long_stats:?}");
        for graphs in [true, false] {
            let (short, first, second) = run(&mut g, GpuCgParams { graph_iters: 1, first_iters: 1, ..Default::default() }, graphs);
            assert_eq!(first.iters, long_stats.iters, "graphs {graphs}");
            assert!(first.rounds > 1 && second.rounds == 1 && second.iters == first.iters, "graphs {graphs}: {first:?} then {second:?}");
            assert!(err_max(&short, &long) < 1e-13, "graphs {graphs}");
        }
    }

    /// The product with S cut over any number of warps a kept block is the
    /// same product: the solve is the CPU's with 1 warp, with as many as
    /// leave most without a tile, and with a count that is no power of
    /// two; on both scalars of the system.
    #[test]
    fn the_product_over_any_warps_is_the_same_product() {
        let (h, _, rhs, _, sym) = setup(40, 300, 5);
        let n = h.symbolic().ncols();
        let dev = device();
        let cg = tight();
        let (want, cpu) = cpu_solve::<f64>(&h, &rhs, &sym, &cg);
        let (want32, cpu32) = cpu_solve::<f32>(&h, &rhs, &sym, &CgParams { tol: 1e-10, ..Default::default() });
        let (h32, r32): (Vec<f32>, Vec<f32>) = (h.vals().iter().map(|&v| v as f32).collect(), rhs.iter().map(|&v| v as f32).collect());
        let mut g = GpuSchur::new(&dev, &sym, h.symbolic(), wide(&*dev)).unwrap();
        let f32_system = GpuSchurParams { f32_reduction: true, ..Default::default() };
        let mut g32 = GpuSchur::with_params(&dev, &sym, h.symbolic(), Values::f32(wide(&*dev)), &f32_system).unwrap();
        for warps in [1u32, 2, 5, 8, 32] {
            let params = GpuCgParams { matvec_warps: Some(warps), ..Default::default() };
            let mut solver = GpuCg::new(&g, &cg, &params).unwrap();
            let mut delta = vec![0.0f64; n];
            let stats = solver.solve(&mut g, h.vals(), &rhs, &mut delta).unwrap();
            assert!(stats.converged && stats.iters.abs_diff(cpu.iters) <= 2, "{warps} warps: {stats:?} against {cpu:?}");
            assert!(err_max(&delta, &want) < 1e-8, "{warps} warps");
            let mut solver = GpuCg::new(&g32, &CgParams { tol: 1e-10, ..Default::default() }, &params).unwrap();
            let mut delta = vec![0.0f32; n];
            let stats = solver.solve(&mut g32, &h32, &r32, &mut delta).unwrap();
            let delta: Vec<f64> = delta.iter().map(|&v| v as f64).collect();
            assert!(stats.iters.abs_diff(cpu32.iters) <= 4, "{warps} warps, f32: {stats:?} against {cpu32:?}");
            assert!(err_max(&delta, &want32) < 1e-3, "{warps} warps, f32");
        }
    }

    /// The warps a kept block takes in the product: about 8 tiles a group
    /// of lanes, a power of two from 1 to 8; as set when set.
    #[test]
    fn the_products_warps_follow_the_tiles_a_block() {
        let p = GpuCgParams::default();
        // 372 cameras of 9 parameters (two groups a warp): 147 tiles a block
        assert_eq!(p.matvec_warps_for(54692, 372, 16), 8);
        // a SLAM map's 900 poses of 6 (four groups a warp): 272 tiles a block
        assert_eq!(p.matvec_warps_for(244820, 900, 8), 8);
        // 40 tiles a block in two groups: 2.5 groups' worth a warp
        assert_eq!(p.matvec_warps_for(40 * 100, 100, 16), 2);
        assert_eq!(p.matvec_warps_for(5 * 100, 100, 8), 1);
        assert_eq!(p.matvec_warps_for(0, 0, 4), 1);
        assert_eq!(GpuCgParams { matvec_warps: Some(3), ..p }.matvec_warps_for(54692, 372, 16), 3);
    }

    /// Iterations queued after the solve has stopped do nothing: the
    /// solution and the status stay.
    #[test]
    fn iterations_after_the_stop_change_nothing() {
        let (h, _, rhs, _, sym) = setup(30, 200, 7);
        let n = h.symbolic().ncols();
        let dev = device();
        let mut g = GpuSchur::new(&dev, &sym, h.symbolic(), wide(&*dev)).unwrap();
        let mut solver = GpuCg::new(&g, &CgParams::default(), &GpuCgParams::default()).unwrap();
        let mut delta = vec![0.0f64; n];
        let stats = solver.solve(&mut g, h.vals(), &rhs, &mut delta).unwrap();
        assert!(stats.converged);
        let before = kept_solution(&g);
        solver.queue(&g, 3).unwrap();
        let mut bytes = [0u8; CG_STATUS_BYTES];
        let (b, at) = solver.status_view();
        dev.read(b, at, &mut bytes).unwrap();
        let st = solver.status(&bytes);
        assert_eq!((st.active, st.iters, st.converged), (false, stats.iters, true));
        assert_eq!(kept_solution(&g), before);
    }

    /// With the residual recomputed every iteration or every few, the
    /// solve reaches the CPU's solution.
    #[test]
    fn a_recomputed_residual_reaches_the_same_solution() {
        let (h, dense, rhs, _, sym) = setup(40, 300, 5);
        let n = h.symbolic().ncols();
        let dev = device();
        for wide in wides(&*dev) {
            let mut g = GpuSchur::new(&dev, &sym, h.symbolic(), wide).unwrap();
            for restart_every in [1usize, 3] {
                let cg = CgParams { tol: 1e-20, restart_every, ..Default::default() };
                let (want, cpu) = cpu_solve::<f64>(&h, &rhs, &sym, &cg);
                let mut solver = GpuCg::new(&g, &cg, &GpuCgParams::default()).unwrap();
                let mut delta = vec![0.0f64; n];
                let stats = solver.solve(&mut g, h.vals(), &rhs, &mut delta).unwrap();
                assert!(stats.converged && cpu.converged, "{wide:?} every {restart_every}: {stats:?} against {cpu:?}");
                assert!(stats.iters.abs_diff(cpu.iters) <= 3, "{wide:?} every {restart_every}: {stats:?} against {cpu:?}");
                assert!(err_max(&delta, &want) < 1e-8, "{wide:?} every {restart_every}");
                assert!(rel_residual(&dense, n, &delta, &rhs) < 1e-9);
            }
        }
    }

    /// A system that is not positive definite fails as on the CPU: an
    /// eliminated block or a diagonal tile of S is reported with the
    /// result, the step left as it was, and the next solve is good again.
    #[test]
    fn blocks_that_are_not_definite_are_reported() {
        let (good, dense, rhs, elim, sym) = setup(10, 30, 2);
        let n = good.symbolic().ncols();
        let asym = good.symbolic().clone();
        let diag0 = |blk: usize| asym.val_range(asym.col_range(blk).find(|&b| asym.blk_row(b) == blk).unwrap()).start;
        let dev = device();
        let mut g = GpuSchur::new(&dev, &sym, &asym, wide(&*dev)).unwrap();
        let mut solver = GpuCg::new(&g, &tight(), &GpuCgParams::default()).unwrap();
        let mut delta = vec![7.0f64; n];
        let e = elim[3];
        let mut h = good.clone();
        h.vals_mut()[diag0(e)] = -1.0;
        let err = solver.solve(&mut g, h.vals(), &rhs, &mut delta).unwrap_err();
        assert!(matches!(err, GpuSchurError::Eliminated { block } if block == e), "{err:?}");
        assert!(delta.iter().all(|&v| v == 7.0), "the step is left as it was");
        // a kept block: S(k, k) is H(k, k) less a semidefinite sum
        let k = sym.kept[0] as usize;
        let mut h = good.clone();
        h.vals_mut()[diag0(k)] = -1e8;
        let err = solver.solve(&mut g, h.vals(), &rhs, &mut delta).unwrap_err();
        assert!(matches!(err, GpuSchurError::Reduced), "{err:?}");
        assert!(delta.iter().all(|&v| v == 7.0));
        let stats = solver.solve(&mut g, good.vals(), &rhs, &mut delta).unwrap();
        assert!(stats.converged && rel_residual(&dense, n, &delta, &rhs) < 1e-9, "{stats:?}");
    }

    /// Two blocks of two columns, identity on the diagonal and `off` times
    /// the identity between them: not positive definite for `off` above 1,
    /// with definite diagonal blocks.
    fn two_blocks(off: f64) -> Matrix {
        let part: Vec<SparseIndex> = vec![0, 2, 4];
        let symbolic = SymbolicSparseBlockColMat::new_checked(part.clone(), part, vec![0, 1, 3], vec![0, 0, 1], vec![0, 4, 8, 12]);
        SparseBlockColMat::new(symbolic, vec![1., 0., 0., 1., off, 0., 0., off, 1., 0., 0., 1.])
    }

    /// A direction along which S is not positive definite stops the solve
    /// with the iterate it has, as on the CPU; the same system made
    /// definite is solved.
    #[test]
    fn a_direction_of_negative_curvature_stops_the_solve() {
        let dev = device();
        let rhs = vec![1.0f64, 0.5, -1.0, -0.5];
        for (off, breaks) in [(10.0, true), (0.5, false)] {
            let h = two_blocks(off);
            let (want, cpu) = cpu_whole(&h, &rhs, &tight());
            let mut g = GpuSchur::whole(&dev, h.symbolic(), wide(&*dev)).unwrap();
            let mut solver = GpuCg::new(&g, &tight(), &GpuCgParams::default()).unwrap();
            let mut delta = vec![0.0f64; 4];
            let stats = solver.solve(&mut g, h.vals(), &rhs, &mut delta).unwrap();
            assert_eq!((stats.breakdown, stats.converged, stats.iters), (breaks, !breaks, cpu.iters), "off {off}: {stats:?} against {cpu:?}");
            assert_eq!(cpu.converged, !breaks);
            if breaks {
                assert_eq!(delta, want, "the iterate so far: zero");
            } else {
                assert!(err_max(&delta, &want) < 1e-12);
            }
        }
    }

    /// Kept blocks of mixed widths up to 30 and up to 64 columns, the whole
    /// system (nothing eliminated): the preconditioner factors and solves
    /// them in place. A device without the shared memory the reduction's
    /// kernels take at 64 columns refuses that reduction, not the solve.
    #[test]
    fn wide_kept_blocks_are_preconditioned_and_solved() {
        let dev = device();
        let mut solved = 0;
        for widths in [&[24usize, 12, 30][..], &[40, 12, 64]] {
            let (h, dense, rhs) = grid_spd(3, widths, 11);
            let n = h.symbolic().ncols();
            let mut g = match GpuSchur::whole(&dev, h.symbolic(), wide(&*dev)) {
                Err(GpuError::Limit(_)) => continue,
                other => other.unwrap(),
            };
            let (want, cpu) = cpu_whole(&h, &rhs, &tight());
            let mut solver = GpuCg::new(&g, &tight(), &GpuCgParams::default()).unwrap();
            let mut delta = vec![0.0f64; n];
            let stats = solver.solve(&mut g, h.vals(), &rhs, &mut delta).unwrap();
            assert!(stats.converged && stats.iters.abs_diff(cpu.iters) <= 3, "{widths:?}: {stats:?} against {cpu:?}");
            assert!(err_max(&delta, &want) < 1e-8, "{widths:?}");
            assert!(rel_residual(&dense, n, &delta, &rhs) < 1e-9, "{widths:?}");
            solved += 1;
        }
        assert!(solved >= 1, "the reduction of blocks up to 30 columns fits every device");
    }

    /// Solves allocate nothing: the blocks are made with the solver, the
    /// graph captured by the first solve.
    #[test]
    fn solves_allocate_nothing() {
        let (h, _, rhs, _, sym) = setup(30, 200, 3);
        let n = h.symbolic().ncols();
        let dev = device();
        let mut g = GpuSchur::new(&dev, &sym, h.symbolic(), wide(&*dev)).unwrap();
        let mut solver = GpuCg::new(&g, &CgParams::default(), &GpuCgParams::default()).unwrap();
        let mut delta = vec![0.0f64; n];
        solver.solve(&mut g, h.vals(), &rhs, &mut delta).unwrap();
        let made = dev.allocations();
        for _ in 0..3 {
            solver.solve(&mut g, h.vals(), &rhs, &mut delta).unwrap();
        }
        assert_eq!(dev.allocations(), made);
    }

    /// Conjugate gradients run on the reduction they were built over.
    #[test]
    #[should_panic(expected = "the reduction they were built over")]
    fn another_reduction_is_refused() {
        let (h, _, _, _, sym) = setup(10, 30, 2);
        let dev = device();
        let g = GpuSchur::new(&dev, &sym, h.symbolic(), wide(&*dev)).unwrap();
        let other = GpuSchur::new(&dev, &sym, h.symbolic(), wide(&*dev)).unwrap();
        let solver = GpuCg::new(&g, &CgParams::default(), &GpuCgParams::default()).unwrap();
        solver.launch_start(&other).unwrap();
    }
}
