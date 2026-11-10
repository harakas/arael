//! [`DeviceSolver`]: the [`GpuSolver`] on any [`Device`].

use std::sync::Arc;

use arael_faer::bsc::SymbolicSparseBlockColMat;
use arael_faer::schur::{SchurReal, SchurSymbolic};
use arael_faer::supernodal::{SupernodalError, SupernodalSymbolic};
use arael_faer::{SparseIndex, ValueIndex};
use arael_gpu::lm::{N_SCALARS, S_COST_TRY, S_COST_X, Scalars};
use arael_gpu::schur::{GraphKey, STATUS_BYTES};
use arael_gpu::FactorizeError;
use arael_gpu::{Device, GpuError, GpuFactor, LmVectors, GpuSchur, GpuSchurError, Wide};
use arael_gpu::{GpuPrecision, RefineStats};

use super::{GpuSolver, narrow, resident_attempt};
use crate::device::DeviceRoot;
use crate::device::DeviceAssembly;
use crate::simple_lm::{DeviceFailure, LmProblemInternals, ResidentAttempt};

/// The factor, the reduction and the route on one [`Device`]; the route
/// first, so its graphs are dropped before the blocks they record.
pub(crate) struct DeviceSolver<T, D: Device> {
    route: Option<Route<T, D>>,
    schur: Option<GpuSchur<D>>,
    factor: GpuFactor<D>,
    /// The scalar of the reduction, the assembly's sums and the loop's vectors.
    wide: Wide,
    /// Where the device assembly's image is written before it is copied.
    staging: crate::device::Staging,
}

/// The device assembly of the reduced route and the LM loop's vectors
/// beside it.
struct Route<T, D: Device> {
    /// An attempt's launches (damping, solve, trial point, trial cost) and
    /// an accept's with the linearization after it, captured as graphs,
    /// with what they were captured against; first, so they are dropped
    /// before the blocks they record.
    attempt_graph: Option<((GraphKey, u64), D::Graph)>,
    step_graph: Option<((GraphKey, u64, u64), D::Graph)>,
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
    /// The factor of `sn`, the analysis of the block structure `asym`, in
    /// `precision`, on `dev`; the reduction and the route built after it
    /// work in `wide`, the route's image staged in `staging`.
    pub fn new(
        dev: &Arc<D>,
        precision: GpuPrecision,
        sn: &SupernodalSymbolic,
        asym: &SymbolicSparseBlockColMat<SparseIndex>,
        wide: Wide,
        staging: crate::device::Staging,
    ) -> Result<Self, GpuError> {
        let factor = GpuFactor::new(dev, precision, sn, asym, &Default::default(), wide)?;
        Ok(DeviceSolver { route: None, schur: None, factor, wide, staging })
    }
}

/// The values of `ranges` (block, byte offset, count), read with one wait
/// into `bytes` and decoded from `wide` into `host`, one range after another.
fn read_values<D: Device>(
    dev: &D,
    wide: Wide,
    ranges: &[(&D::Block, usize, usize)],
    bytes: &mut Vec<u8>,
    host: &mut Vec<f64>,
) -> Result<(), GpuError> {
    let total: usize = ranges.iter().map(|r| r.2).sum();
    bytes.resize(8 * total, 0);
    let mut reads = Vec::with_capacity(ranges.len());
    let mut rest = &mut bytes[..];
    for &(b, at, n) in ranges {
        let (head, tail) = rest.split_at_mut(8 * n);
        reads.push((b, at, head));
        rest = tail;
    }
    dev.read_many(&mut reads)?;
    host.resize(total, 0.0);
    wide.decode(bytes, host);
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

impl<T: crate::utils::Float, D: Device> DeviceSolver<T, D> {
    /// The pending accept applied, without waiting.
    fn settle(&mut self) -> Result<(), DeviceFailure> {
        let r = self.route.as_mut().expect("a device route");
        if std::mem::take(&mut r.accept_due) {
            on_device(accept_on_device(&r.asm, &r.vec), "accept")?;
        }
        Ok(())
    }
}

impl<T: crate::utils::Float + SchurReal, D: Device> GpuSolver<T> for DeviceSolver<T, D> {
    fn runtime(&self) -> String {
        match self.wide {
            Wide::F64 => D::RUNTIME.to_string(),
            Wide::Df32 => format!("{} df32", D::RUNTIME),
        }
    }

    fn precision(&self) -> GpuPrecision {
        self.factor.precision()
    }

    fn has_f64(&self) -> bool {
        self.factor.device().arch().has_f64
    }

    fn factorize(&mut self, vals: &[T]) -> Result<Result<(), SupernodalError>, DeviceFailure> {
        match self.factor.factorize(vals) {
            Ok(()) => Ok(Ok(())),
            Err(FactorizeError::NotPositiveDefinite) => Ok(Err(SupernodalError::NotPositiveDefinite)),
            Err(FactorizeError::Device(e)) => on_device(Err(e), "factorization"),
        }
    }

    fn solve(&mut self, x: &mut [T]) -> Result<(), DeviceFailure> {
        on_device(self.factor.solve(x), "solve")
    }

    fn solve_f64(&mut self, x: &mut [f64]) -> Result<(), DeviceFailure> {
        on_device(self.factor.solve(x), "solve")
    }

    fn prepare_reduction(&mut self, schur: Option<&SchurSymbolic<SparseIndex>>, hsym: &SymbolicSparseBlockColMat<SparseIndex>) -> Result<(), GpuError> {
        // the old blocks go first, so the device holds one structure's at a time
        self.drop_reduction();
        let dev = self.factor.device();
        self.schur = Some(match schur {
            Some(s) => GpuSchur::new(dev, s, hsym, self.wide)?,
            None => GpuSchur::whole(dev, hsym, self.wide)?,
        });
        Ok(())
    }

    fn drop_reduction(&mut self) {
        self.route = None;
        self.schur = None;
    }

    fn reduces(&self) -> bool {
        self.schur.is_some()
    }

    fn solve_reduced(&mut self, hvals: &[T], grad: &[T], refine: Option<(usize, f64)>, delta: &mut [T]) -> Result<Option<RefineStats>, DeviceFailure> {
        let gs = self.schur.as_mut().expect("the device reduction is built with the Schur symbolic");
        match gs.solve(&mut self.factor, hvals, grad, refine, delta) {
            Ok(stats) => Ok(Some(stats)),
            Err(GpuSchurError::Eliminated { .. } | GpuSchurError::Reduced) => Ok(None),
            Err(GpuSchurError::Device(e)) => on_device(Err(e), "reduced solve"),
        }
    }

    fn build_route(&mut self, root: &dyn DeviceRoot<T>, hsym: &SymbolicSparseBlockColMat<SparseIndex>, bdiag_pos: &[ValueIndex], threads: usize) -> Result<(), GpuError> {
        self.route = None;
        if self.schur.is_none() {
            return Ok(());
        }
        let dev = self.factor.device();
        let asm = DeviceAssembly::new(dev, root, hsym, threads, self.wide, self.staging)?;
        self.route = Some(Route {
            attempt_graph: None,
            step_graph: None,
            asm,
            vec: LmVectors::new(dev, bdiag_pos, self.wide)?,
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
        let queued = r.asm.assemble_at(r.vec.x(), g, n, h, r.nh, cost).and_then(|()| r.vec.gather_diag(h));
        on_device(queued, "assembly")?;
        let diag = r.vec.diag();
        let read = read_values(&**r.asm.device(), self.wide, &[(g.0, g.1, n), (diag.0, diag.1, n), (cost.0, cost.1, 1)], &mut r.bytes, &mut r.host);
        on_device(read, "read of the gradient")?;
        narrow(grad, &r.host[..n]);
        narrow(&mut r.diag_host, &r.host[n..2 * n]);
        r.live = true;
        Ok(T::from(r.host[2 * n]).unwrap())
    }

    fn route_diagonal(&self, diagonal: &mut [T]) {
        diagonal.copy_from_slice(&self.route.as_ref().expect("a device route").diag_host);
    }

    fn start(&mut self, x: &[T], floor: Option<f64>) -> Result<(), DeviceFailure> {
        let r = self.route.as_mut().expect("a resident solve has its device route");
        r.floor = floor;
        r.accept_due = false;
        r.host.clear();
        r.host.extend(x.iter().map(|&v| SchurReal::to_f64(v)));
        on_device(r.vec.write_x(&r.host), "upload of the parameters")?;
        if r.live {
            // this linearization was assembled there: its diagonal pass only
            let gs = self.schur.as_ref().expect("a device route has its reduction");
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
        let gs = self.schur.as_ref().expect("a device route has its reduction");
        let r = self.route.as_mut().expect("a device route");
        r.live = false;
        let accept = std::mem::take(&mut r.accept_due);
        let key = (gs.graph_key(&self.factor, None), r.asm.generation(), r.floor.map_or(u64::MAX, f64::to_bits));
        let (asm, vec, nh, floor) = (&r.asm, &r.vec, r.nh, r.floor);
        let (g, h) = (gs.rhs_target(), gs.h_target());
        let launches = || {
            if accept {
                accept_on_device(asm, vec)?;
            }
            asm.assemble_at(vec.x(), g, vec.len(), h, nh, vec.slot(S_COST_X))?;
            vec.linearized(g, h, floor)
        };
        let queued = if accept { run_graph(&**asm.device(), &mut r.step_graph, key, launches) } else { launches() };
        on_device(queued, "linearization")?;
        r.live = true;
        Ok(())
    }

    fn attempt(&mut self, lambda: f64, refine: Option<(usize, f64)>) -> Result<(Option<RefineStats>, ResidentAttempt), DeviceFailure> {
        let gs = self.schur.as_ref().expect("a device route has its reduction");
        let gf = &self.factor;
        let r = self.route.as_mut().expect("a resident solve has its device route");
        on_device(r.vec.set_lambda(lambda), "upload of the damping")?;
        let key = (gs.graph_key(gf, refine), r.asm.generation());
        let (asm, vec) = (&r.asm, &r.vec);
        let launches = || {
            vec.damp(gs.h_target())?;
            gs.launch(gf, refine)?;
            vec.step(gs.step_view(), gs.rhs_target())?;
            asm.cost_at(vec.x_try(), vec.slot(S_COST_TRY))
        };
        on_device(run_graph(&**asm.device(), &mut r.attempt_graph, key, launches), "attempt")?;
        let (mut status, mut scalars) = ([0u8; STATUS_BYTES], [0u8; 8 * N_SCALARS]);
        let (st, sc) = (gs.status_view(), vec.scalars());
        on_device(asm.device().read_many(&mut [(st.0, st.1, &mut status[..]), (sc.0, sc.1, &mut scalars[..])]), "read of the attempt")?;
        let solved = match gs.outcome(&status, refine) {
            Ok(stats) => Some(stats),
            Err(GpuSchurError::Eliminated { .. } | GpuSchurError::Reduced) => None,
            Err(GpuSchurError::Device(e)) => return on_device(Err(e), "attempt"),
        };
        let mut values = [0.0f64; N_SCALARS];
        self.wide.decode(&scalars, &mut values);
        Ok((solved, resident_attempt(solved.is_some(), &Scalars::from_values(&values))))
    }

    fn accept(&mut self) {
        // applied with the next linearization, in its graph
        self.route.as_mut().expect("a resident solve has its device route").accept_due = true;
    }

    fn params(&mut self, x: &mut [T]) -> Result<(), DeviceFailure> {
        self.settle()?;
        let r = self.route.as_mut().expect("a resident solve has its device route");
        let at = r.vec.x();
        on_device(read_values(&**r.asm.device(), self.wide, &[(at.0, at.1, x.len())], &mut r.bytes, &mut r.host), "read of the parameters")?;
        narrow(x, &r.host);
        Ok(())
    }

    fn vectors(&mut self, grad: &mut [T], diagonal: &mut [T], step: &mut [T]) -> Result<(), DeviceFailure> {
        let gs = self.schur.as_ref().expect("a device route has its reduction");
        let r = self.route.as_mut().expect("a resident solve has its device route");
        let n = grad.len();
        let (g, d, s) = (gs.rhs_target(), r.vec.diag(), gs.step_view());
        let read = read_values(&**r.asm.device(), self.wide, &[(g.0, g.1, n), (d.0, d.1, n), (s.0, s.1, n)], &mut r.bytes, &mut r.host);
        on_device(read, "read of the vectors")?;
        narrow(grad, &r.host[..n]);
        narrow(diagonal, &r.host[n..2 * n]);
        narrow(step, &r.host[2 * n..]);
        Ok(())
    }

    fn nonfinite(&mut self) -> Result<[usize; 4], DeviceFailure> {
        let gs = self.schur.as_ref().expect("a device route has its reduction");
        let r = self.route.as_mut().expect("a resident solve has its device route");
        let n = r.vec.len();
        let (g, d, x, h) = (gs.rhs_target(), r.vec.diag(), r.vec.x(), gs.h_target());
        let read = read_values(&**r.asm.device(), self.wide, &[(g.0, g.1, n), (d.0, d.1, n), (x.0, x.1, n), (h.0, h.1, r.nh)], &mut r.bytes, &mut r.host);
        on_device(read, "read of the vectors")?;
        let count = |v: &[f64]| v.iter().filter(|v| !v.is_finite()).count();
        let v = &r.host;
        Ok([count(&v[..n]), count(&v[n..2 * n]), count(&v[2 * n..3 * n]), count(&v[3 * n..])])
    }

    fn finish(&mut self, problem: &mut dyn LmProblemInternals<T>, x: &mut [T]) -> Result<(), DeviceFailure> {
        self.params(x)?;
        let r = self.route.as_ref().expect("a resident solve has its device route");
        on_device(r.asm.restore(problem.device_root_mut().expect("a device route's problem has a device root")), "read of the rotations")
    }
}
