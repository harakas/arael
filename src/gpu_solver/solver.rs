//! [`DeviceSolver`]: the [`GpuSolver`] on any [`Device`].

use std::sync::Arc;

use arael_faer::bsc::SymbolicSparseBlockColMat;
use arael_faer::cg::CgParams;
use arael_faer::schur::{SchurReal, SchurSymbolic};
use arael_faer::{SparseIndex, ValueIndex};
use arael_gpu::lm::{N_SCALARS, S_COST_TRY, S_COST_X, Scalars};
use arael_gpu::schur::STATUS_BYTES;
use arael_gpu::{Device, GpuCg, GpuCgParams, GpuError, GpuFactor, GpuPrecision, GpuSchur, GpuSchurError, LmVectors, Scope, Values, Wide};

use super::{GpuSolver, Method, SolveStats, narrow, resident_attempt};
use crate::device::DeviceRoot;
use crate::device::DeviceAssembly;
use crate::simple_lm::{DeviceFailure, GpuSolve, LmProblemInternals, ResidentAttempt};

/// What solves the system, the reduction and the route on one [`Device`].
/// In the order they are dropped: a graph before the blocks it records (the
/// route's record everything, the conjugate gradients' the reduction's
/// blocks, the reduction's the factor's).
pub(crate) struct DeviceSolver<T, D: Device> {
    route: Option<Route<T, D>>,
    /// Conjugate gradients over the reduction, built with it
    /// ([`Solve::Cg`]).
    cg: Option<GpuCg<D>>,
    schur: Option<GpuSchur<D>>,
    solve: Solve<D>,
    dev: Arc<D>,
    /// The route's wide scalar: the cost's and the loop's scalars'
    /// ([`values`](Self::values) for the vectors, H and the gradient).
    wide: Wide,
    /// Where the device assembly's image is written before it is copied.
    staging: crate::device::Staging,
    /// The reduction in f32 (`GpuSchurParams::f32_reduction`).
    f32_reduction: bool,
    /// The device's work is timed ([`GpuSolver::set_timing`]), its times so
    /// far, and whether a linearization was stamped since they were read.
    timed: bool,
    times: crate::simple_lm::GpuTiming,
    linearized: bool,
}

/// What solves the system on the device ([`Method`]).
enum Solve<D: Device> {
    /// The factor, its solves refined against S under `refine`
    /// (`(max_steps, tolerance)`): under mixed precision, where it is f32.
    Factor { factor: GpuFactor<D>, refine: Option<(usize, f64)> },
    /// Conjugate gradients under these settings, built over the reduction
    /// ([`DeviceSolver::cg`]).
    Cg(CgParams),
}

/// The stamps ([`Device::stamp`]) around a linearization, around an
/// attempt, and within it around the conjugate gradients' iterations.
const LIN: (usize, usize) = (0, 1);
const ATTEMPT: (usize, usize) = (2, 3);
const CG: (usize, usize) = (4, 5);

/// The device assembly of the reduced route and the LM loop's vectors
/// beside it.
struct Route<T, D: Device> {
    /// An attempt's launches and an accept's with the linearization after
    /// it, captured as graphs, with the assembly's generation (and the
    /// diagonal's floor) they were captured against; first, so they are
    /// dropped before the blocks they record. With a factor an attempt is
    /// one graph (damping, solve, trial point, trial cost). With conjugate
    /// gradients, whose iterations are queued between them, it is two:
    /// `attempt_graph` up to the iterations' start, `tail_graph` after
    /// them.
    attempt_graph: Option<(u64, D::Graph)>,
    tail_graph: Option<(u64, D::Graph)>,
    step_graph: Option<((u64, u64), D::Graph)>,
    asm: DeviceAssembly<T, D>,
    vec: LmVectors<D>,
    /// The undamped diagonal of the last compute that read it back.
    diag_host: Vec<T>,
    /// What the reads land in, then decoded, before they are narrowed to `T`.
    bytes: Vec<u8>,
    host: Vec<f64>,
    /// H's tile values.
    nh: usize,
    /// The device holds H and the gradient at the loop's parameters.
    live: bool,
    /// The model's data on the device predates this solve.
    stale: bool,
    /// [`LmConfig::min_diagonal`](crate::simple_lm::LmConfig::min_diagonal).
    floor: Option<f64>,
    /// An accepted trial point waits to become the parameters: the next
    /// linearization, or the end of the solve, takes it.
    accept_due: bool,
}

/// `r`'s value, or the failure of the device operation `operation`.
fn on_device<V>(r: Result<V, GpuError>, operation: &'static str) -> Result<V, DeviceFailure> {
    r.map_err(|e| DeviceFailure { operation, message: e.to_string() })
}

impl<T, D: Device> DeviceSolver<T, D> {
    /// A solver by `method` on `dev`: a factor is built here, conjugate
    /// gradients with the reduction. The reduction and the route built
    /// after it work in `wide`, the route's image staged in `staging`, the
    /// reduction in f32 with `f32_reduction`.
    pub fn new(dev: &Arc<D>, method: Method<'_>, wide: Wide, staging: crate::device::Staging, f32_reduction: bool) -> Result<Self, GpuError> {
        let solve = match method {
            Method::Factor { precision, sn, asym, refine } => {
                let _scope = (Scope::new(&**dev, "setup"), Scope::new(&**dev, "factor"));
                Solve::Factor { factor: GpuFactor::new(dev, precision, sn, asym, &Default::default(), wide)?, refine }
            }
            Method::Cg(params) => Solve::Cg(params),
        };
        Ok(DeviceSolver { route: None, cg: None, schur: None, solve, dev: dev.clone(), wide, staging, f32_reduction, timed: false, times: Default::default(), linearized: false })
    }

    /// The scalar of the vectors, H and the gradient: the root's, as the
    /// CPU's route keeps them.
    fn values(&self) -> Values {
        Values::of(core::mem::size_of::<T>(), self.wide)
    }
}

/// The values of `ranges` (block, byte offset, count, scalar), read with
/// one wait into `bytes` and decoded into `host`, one range after another.
fn read_values<D: Device>(
    dev: &D,
    ranges: &[(&D::Block, usize, usize, Values)],
    bytes: &mut Vec<u8>,
    host: &mut Vec<f64>,
) -> Result<(), GpuError> {
    bytes.resize(ranges.iter().map(|r| r.2 * r.3.bytes()).sum(), 0);
    let mut reads = Vec::with_capacity(ranges.len());
    let mut rest = &mut bytes[..];
    for &(b, at, n, v) in ranges {
        let (head, tail) = rest.split_at_mut(n * v.bytes());
        reads.push((b, at, head));
        rest = tail;
    }
    dev.read_many(&mut reads)?;
    host.resize(ranges.iter().map(|r| r.2).sum(), 0.0);
    let (mut b, mut h) = (&bytes[..], &mut host[..]);
    for &(_, _, n, v) in ranges {
        let (bh, bt) = b.split_at(n * v.bytes());
        let (hh, ht) = h.split_at_mut(n);
        v.decode(bh, hh);
        (b, h) = (bt, ht);
    }
    Ok(())
}

/// An accepted trial point becomes the parameters: the reference rotations
/// moved by its steps (the last gather was the trial point's), the point
/// copied, its rotation steps zeroed.
fn accept_on_device<T: crate::utils::Float, D: Device>(asm: &DeviceAssembly<T, D>, vec: &LmVectors<D>) -> Result<(), GpuError> {
    asm.advance()?;
    vec.accept()?;
    asm.zero_steps(vec.x())
}

/// Run `launches` (kernel launches only) as the graph `slot` holds when it
/// was captured under `key`; otherwise run them, and record them into
/// `slot` while they run.
fn run_graph<D: Device, K: PartialEq>(
    dev: &D,
    slot: &mut Option<(K, D::Graph)>,
    key: K,
    launches: impl Fn() -> Result<(), GpuError>,
) -> Result<(), GpuError> {
    if let Some((k, graph)) = slot.as_ref()
        && *k == key
    {
        return dev.replay(graph);
    }
    *slot = None;
    launches()?;
    *slot = Some((key, dev.capture(&launches)?));
    Ok(())
}

/// Stamp `dev`'s queue at `slot` when its work is `timed`.
fn stamp<D: Device>(dev: &D, timed: bool, slot: usize) -> Result<(), DeviceFailure> {
    if !timed {
        return Ok(());
    }
    on_device(dev.stamp(slot), "timestamp")
}

impl<T: crate::utils::Float, D: Device> DeviceSolver<T, D> {
    /// Add the spans of the stamps made since the last call: an attempt's
    /// when `attempt`, a linearization's when one was stamped. The device
    /// has passed them: the caller has just read from it.
    fn collect(&mut self, attempt: bool) -> Result<(), DeviceFailure> {
        if !self.timed {
            return Ok(());
        }
        let dev = &self.dev;
        let span = |(a, b): (usize, usize)| on_device(dev.span(a, b), "timestamp").map(|s| s.map(std::time::Duration::from_secs_f64));
        if std::mem::take(&mut self.linearized) && let Some(d) = span(LIN)? {
            self.times.assembly += d;
            self.times.assembly_count += 1;
        }
        if attempt && let Some(d) = span(ATTEMPT)? {
            self.times.attempts += d;
            self.times.attempt_count += 1;
        }
        // the iterations of the attempt's conjugate gradients, when it ran
        // any: its rounds since the last attempt
        if attempt && let Some(cg) = self.cg.as_mut() {
            let rounds = cg.take_rounds();
            if rounds > 0 && let Some(d) = span(CG)? {
                self.times.cg += d;
                self.times.cg_rounds += rounds;
            }
        }
        Ok(())
    }

    /// The pending accept applied, without waiting.
    fn settle(&mut self) -> Result<(), DeviceFailure> {
        let r = self.route.as_mut().expect("a device route");
        if std::mem::take(&mut r.accept_due) {
            let _scope = Scope::new(&**r.asm.device(), "accept");
            on_device(accept_on_device(&r.asm, &r.vec), "accept")?;
        }
        Ok(())
    }
}

impl<T: crate::utils::Float + SchurReal, D: Device> GpuSolver<T> for DeviceSolver<T, D> {
    fn solve(&self) -> GpuSolve {
        match &self.solve {
            Solve::Factor { factor, .. } => GpuSolve::Factor { f32: factor.precision() == GpuPrecision::F32 },
            Solve::Cg(_) => GpuSolve::ConjugateGradients,
        }
    }

    fn arch(&self) -> &arael_gpu::Arch {
        self.dev.arch()
    }

    fn set_timing(&mut self, on: bool) {
        self.timed = on;
        self.times = Default::default();
        self.linearized = false;
        if let Some(gs) = self.schur.as_mut() {
            gs.stamp_solves(on.then_some(ATTEMPT));
        }
        if let Some(cg) = self.cg.as_mut() {
            cg.stamp_runs(on.then_some(CG));
            cg.take_rounds();
        }
    }

    fn timing(&self) -> Option<crate::simple_lm::GpuTiming> {
        (self.timed && self.times.assembly_count + self.times.attempt_count > 0).then_some(self.times)
    }

    fn prepare_reduction(&mut self, schur: Option<&SchurSymbolic<SparseIndex>>, hsym: &SymbolicSparseBlockColMat<SparseIndex>) -> Result<(), GpuError> {
        // the old blocks go first, so the device holds one structure's at a time
        self.drop_reduction();
        let dev = &self.dev;
        let _scope = (Scope::new(&**dev, "setup"), Scope::new(&**dev, "reduction"));
        let params = arael_gpu::GpuSchurParams { f32_reduction: self.f32_reduction, ..Default::default() };
        let values = self.values();
        let mut gs = match schur {
            Some(s) => GpuSchur::with_params(dev, s, hsym, values, &params)?,
            None => GpuSchur::whole_with_params(dev, hsym, values, &params)?,
        };
        gs.stamp_solves(self.timed.then_some(ATTEMPT));
        if let Solve::Cg(params) = &self.solve {
            let mut cg = GpuCg::new(&gs, params, &GpuCgParams::default())?;
            cg.stamp_runs(self.timed.then_some(CG));
            self.cg = Some(cg);
        }
        self.schur = Some(gs);
        Ok(())
    }

    fn drop_reduction(&mut self) {
        self.route = None;
        self.cg = None;
        self.schur = None;
    }

    fn reduces(&self) -> bool {
        self.schur.is_some()
    }

    fn f32_reduction(&self) -> bool {
        self.schur.as_ref().is_some_and(|s| s.f32_reduction())
    }

    fn solve_reduced(&mut self, hvals: &[T], grad: &[T], delta: &mut [T]) -> Result<Option<SolveStats>, DeviceFailure> {
        let dev = self.dev.clone();
        let _scope = Scope::new(&*dev, "solve_reduced");
        let gs = self.schur.as_mut().expect("the device reduction is built with the Schur symbolic");
        let solved = match &mut self.solve {
            Solve::Factor { factor, refine } => gs.solve(factor, hvals, grad, *refine, delta).map(SolveStats::Refined),
            Solve::Cg(_) => self.cg.as_mut().expect("conjugate gradients are built with the reduction").solve(gs, hvals, grad, delta).map(SolveStats::Cg),
        };
        let solved = match solved {
            Ok(stats) => Some(stats),
            Err(GpuSchurError::Eliminated { .. } | GpuSchurError::Reduced) => None,
            Err(GpuSchurError::Device(e)) => return on_device(Err(e), "reduced solve"),
        };
        self.collect(true)?;
        Ok(solved)
    }

    fn build_route(&mut self, root: &dyn DeviceRoot<T>, hsym: &SymbolicSparseBlockColMat<SparseIndex>, bdiag_pos: &[ValueIndex], threads: usize) -> Result<(), GpuError> {
        self.route = None;
        if self.schur.is_none() {
            return Ok(());
        }
        let dev = &self.dev;
        let _scope = (Scope::new(&**dev, "setup"), Scope::new(&**dev, "route"));
        let asm = DeviceAssembly::new(dev, root, hsym, threads, self.wide, self.staging)?;
        self.route = Some(Route {
            attempt_graph: None,
            tail_graph: None,
            step_graph: None,
            asm,
            vec: LmVectors::new(dev, bdiag_pos, self.values())?,
            diag_host: vec![T::zero(); bdiag_pos.len()],
            bytes: Vec::new(),
            host: Vec::new(),
            nh: hsym.val_count(),
            live: false,
            stale: false,
            floor: None,
            accept_due: false,
        });
        Ok(())
    }

    fn has_route(&self) -> bool {
        self.route.is_some()
    }

    fn route_live(&self) -> bool {
        self.route.as_ref().is_some_and(|r| r.live)
    }

    fn mark_stale(&mut self) {
        if let Some(r) = self.route.as_mut() {
            r.stale = true;
        }
    }

    fn route_compute(&mut self, problem: &mut dyn LmProblemInternals<T>, params: &[T], grad: &mut [T]) -> Result<T, DeviceFailure> {
        let (values, wide) = (self.values(), Values::from(self.wide));
        let dev = self.dev.clone();
        let _scope = Scope::new(&*dev, "compute");
        let timed = self.timed;
        let gs = self.schur.as_ref().expect("a device route has its reduction");
        let r = self.route.as_mut().expect("a device route");
        // what the device holds is the last compute's only once this one
        // ends
        r.live = false;
        if r.stale {
            problem.update_params(params);
            on_device(r.asm.refresh(problem.device_root().expect("a device route's problem has a device root")), "upload of the model's data")?;
            r.stale = false;
        }
        let n = grad.len();
        r.host.clear();
        r.host.extend(params.iter().map(|&v| SchurReal::to_f64(v)));
        on_device(r.vec.write_x(&r.host), "upload of the parameters")?;
        let (g, h, cost) = (gs.rhs_target(), gs.h_target(), r.vec.slot(S_COST_X));
        stamp(&*dev, timed, LIN.0)?;
        let queued = r.asm.assemble_at(r.vec.x(), g, n, h, r.nh, cost).and_then(|()| r.vec.gather_diag(h));
        on_device(queued, "assembly")?;
        stamp(&*dev, timed, LIN.1)?;
        let diag = r.vec.diag();
        let read = read_values(&**r.asm.device(), &[(g.0, g.1, n, values), (diag.0, diag.1, n, values), (cost.0, cost.1, 1, wide)], &mut r.bytes, &mut r.host);
        on_device(read, "read of the gradient")?;
        narrow(grad, &r.host[..n]);
        narrow(&mut r.diag_host, &r.host[n..2 * n]);
        r.live = true;
        let cost = T::from(r.host[2 * n]).unwrap();
        self.linearized = timed;
        self.collect(false)?;
        Ok(cost)
    }

    fn route_diagonal(&self, diagonal: &mut [T]) {
        diagonal.copy_from_slice(&self.route.as_ref().expect("a device route").diag_host);
    }

    fn start(&mut self, x: &[T], floor: Option<f64>) -> Result<(), DeviceFailure> {
        let dev = self.dev.clone();
        let _scope = Scope::new(&*dev, "start");
        let r = self.route.as_mut().expect("a resident solve has its device route");
        r.floor = floor;
        r.accept_due = false;
        r.host.clear();
        r.host.extend(x.iter().map(|&v| SchurReal::to_f64(v)));
        on_device(r.vec.write_x(&r.host), "upload of the parameters")?;
        if r.live {
            // this linearization was assembled there: its diagonal pass only
            let gs = self.schur.as_ref().expect("a device route has its reduction");
            let _scope = Scope::new(&*dev, "pass");
            on_device(r.vec.linearized(gs.rhs_target(), gs.h_target(), r.floor), "linearization")
        } else {
            self.linearize()
        }
    }

    /// The pending accept, when there is one, and the linearization at the
    /// parameters after it (the assembly and the pass over the diagonal and
    /// the gradient), without waiting. An accept with its linearization is
    /// one captured graph.
    fn linearize(&mut self) -> Result<(), DeviceFailure> {
        let dev = self.dev.clone();
        let _scope = Scope::new(&*dev, "linearize");
        let timed = self.timed;
        let gs = self.schur.as_ref().expect("a device route has its reduction");
        let r = self.route.as_mut().expect("a device route");
        r.live = false;
        stamp(&*dev, timed, LIN.0)?;
        let accept = std::mem::take(&mut r.accept_due);
        let key = (r.asm.generation(), r.floor.map_or(u64::MAX, f64::to_bits));
        let (asm, vec, nh, floor) = (&r.asm, &r.vec, r.nh, r.floor);
        let (g, h) = (gs.rhs_target(), gs.h_target());
        let launches = || {
            if accept {
                let _scope = Scope::new(&**asm.device(), "accept");
                accept_on_device(asm, vec)?;
            }
            asm.assemble_at(vec.x(), g, vec.len(), h, nh, vec.slot(S_COST_X))?;
            let _scope = Scope::new(&**asm.device(), "pass");
            vec.linearized(g, h, floor)
        };
        let queued = if accept { run_graph(&**asm.device(), &mut r.step_graph, key, launches) } else { launches() };
        on_device(queued, "linearization")?;
        stamp(&*dev, timed, LIN.1)?;
        r.live = true;
        // its span is read at the next wait, an attempt's
        self.linearized = timed;
        Ok(())
    }

    /// With a factor, one captured graph and one wait. With conjugate
    /// gradients, a graph up to the iterations' start, the iterations (a
    /// wait a round, [`GpuCg::run`]), a graph after them, and the wait for
    /// the result.
    fn attempt(&mut self, lambda: f64) -> Result<(Option<SolveStats>, ResidentAttempt), DeviceFailure> {
        let dev = self.dev.clone();
        let _scope = Scope::new(&*dev, "attempt");
        let timed = self.timed;
        let gs = self.schur.as_ref().expect("a device route has its reduction");
        let r = self.route.as_mut().expect("a resident solve has its device route");
        stamp(&*dev, timed, ATTEMPT.0)?;
        on_device(r.vec.set_lambda(lambda), "upload of the damping")?;
        let key = r.asm.generation();
        let (asm, vec) = (&r.asm, &r.vec);
        let damp = || {
            let _scope = Scope::new(&**asm.device(), "damp");
            vec.damp(gs.h_target())
        };
        // the trial point from the step, and its cost
        let trial = || {
            {
                let _scope = Scope::new(&**asm.device(), "step");
                vec.step(gs.step_view(), gs.rhs_target())?;
            }
            asm.cost_at(vec.x_try(), vec.slot(S_COST_TRY))
        };
        let (refine, iterated) = match &self.solve {
            Solve::Factor { factor, refine } => {
                let launches = || {
                    damp()?;
                    gs.launch(factor, *refine)?;
                    trial()
                };
                on_device(run_graph(&**asm.device(), &mut r.attempt_graph, key, launches), "attempt")?;
                (*refine, None)
            }
            Solve::Cg(_) => {
                let cg = self.cg.as_mut().expect("conjugate gradients are built with the reduction");
                let head = || {
                    damp()?;
                    gs.launch_reduce()?;
                    cg.launch_start(gs)
                };
                on_device(run_graph(&**asm.device(), &mut r.attempt_graph, key, head), "attempt")?;
                let stats = on_device(cg.run(gs), "conjugate gradients")?;
                let tail = || {
                    gs.launch_backsub()?;
                    trial()
                };
                on_device(run_graph(&**asm.device(), &mut r.tail_graph, key, tail), "attempt")?;
                (None, Some(stats))
            }
        };
        stamp(&*dev, timed, ATTEMPT.1)?;
        let (mut status, mut scalars) = ([0u8; STATUS_BYTES], [0u8; 8 * N_SCALARS]);
        let (st, sc) = (gs.status_view(), vec.scalars());
        on_device(asm.device().read_many(&mut [(st.0, st.1, &mut status[..]), (sc.0, sc.1, &mut scalars[..])]), "read of the attempt")?;
        let solved = match gs.outcome(&status, refine) {
            Ok(stats) => Some(iterated.map_or(SolveStats::Refined(stats), SolveStats::Cg)),
            Err(GpuSchurError::Eliminated { .. } | GpuSchurError::Reduced) => None,
            Err(GpuSchurError::Device(e)) => return on_device(Err(e), "attempt"),
        };
        let mut values = [0.0f64; N_SCALARS];
        self.wide.decode(&scalars, &mut values);
        // the read above waited for the attempt, and for the linearization
        // queued before it
        self.collect(true)?;
        Ok((solved, resident_attempt(solved.is_some(), &Scalars::from_values(&values))))
    }

    fn accept(&mut self) {
        // applied with the next linearization, in its graph
        self.route.as_mut().expect("a resident solve has its device route").accept_due = true;
    }

    fn params(&mut self, x: &mut [T]) -> Result<(), DeviceFailure> {
        self.settle()?;
        let values = self.values();
        let r = self.route.as_mut().expect("a resident solve has its device route");
        let at = r.vec.x();
        on_device(read_values(&**r.asm.device(), &[(at.0, at.1, x.len(), values)], &mut r.bytes, &mut r.host), "read of the parameters")?;
        narrow(x, &r.host);
        Ok(())
    }

    fn vectors(&mut self, grad: &mut [T], diagonal: &mut [T], step: &mut [T]) -> Result<(), DeviceFailure> {
        let values = self.values();
        let gs = self.schur.as_ref().expect("a device route has its reduction");
        let r = self.route.as_mut().expect("a resident solve has its device route");
        let n = grad.len();
        let (g, d, s) = (gs.rhs_target(), r.vec.diag(), gs.step_view());
        let read = read_values(&**r.asm.device(), &[(g.0, g.1, n, values), (d.0, d.1, n, values), (s.0, s.1, n, values)], &mut r.bytes, &mut r.host);
        on_device(read, "read of the vectors")?;
        narrow(grad, &r.host[..n]);
        narrow(diagonal, &r.host[n..2 * n]);
        narrow(step, &r.host[2 * n..]);
        Ok(())
    }

    fn nonfinite(&mut self) -> Result<[usize; 4], DeviceFailure> {
        let values = self.values();
        let gs = self.schur.as_ref().expect("a device route has its reduction");
        let r = self.route.as_mut().expect("a resident solve has its device route");
        let n = r.vec.len();
        let (g, d, x, h) = (gs.rhs_target(), r.vec.diag(), r.vec.x(), gs.h_target());
        let ranges = [(g.0, g.1, n, values), (d.0, d.1, n, values), (x.0, x.1, n, values), (h.0, h.1, r.nh, values)];
        let read = read_values(&**r.asm.device(), &ranges, &mut r.bytes, &mut r.host);
        on_device(read, "read of the vectors")?;
        let count = |v: &[f64]| v.iter().filter(|v| !v.is_finite()).count();
        let v = &r.host;
        Ok([count(&v[..n]), count(&v[n..2 * n]), count(&v[2 * n..3 * n]), count(&v[3 * n..])])
    }

    fn finish(&mut self, problem: &mut dyn LmProblemInternals<T>, x: &mut [T]) -> Result<(), DeviceFailure> {
        let dev = self.dev.clone();
        let _scope = Scope::new(&*dev, "finish");
        self.params(x)?;
        // a linearization no attempt followed
        self.collect(false)?;
        let r = self.route.as_ref().expect("a resident solve has its device route");
        on_device(r.asm.restore(problem.device_root_mut().expect("a device route's problem has a device root")), "read of the rotations")
    }
}
