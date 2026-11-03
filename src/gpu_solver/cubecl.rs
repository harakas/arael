//! [`CubeclSolver`]: the [`GpuSolver`] on CubeCL's runtime (features
//! `gpu-cuda`, `gpu-hip`).

use arael_faer::bsc::{PositionResolver, SymbolicSparseBlockColMat};
use arael_faer::schur::{SchurReal, SchurSymbolic};
use arael_faer::supernodal::{SupernodalError, SupernodalSymbolic};
use arael_faer::{SparseIndex, ValueIndex};
use arael_gpu::cubecl::CubeElement as _;
use arael_gpu::lm::{LmVectors, S_COST_TRY, S_COST_X, Scalars};
use arael_gpu::native::NativeError;
use arael_gpu::{GpuContext, GpuFactor, GpuPrecision, GpuSchur, RefineStats};

use super::{GpuSolver, narrow, resident_attempt};
use crate::device::{DeviceAssembly, DeviceRoot, DeviceTargets};
use crate::simple_lm::{LmProblemInternals, ResidentAttempt};

/// The factor, the reduction and the route on a [`GpuContext`]; the route
/// first, so its graphs are dropped before the buffers they record.
pub(crate) struct CubeclSolver<T> {
    route: Option<Route<T>>,
    schur: Option<GpuSchur>,
    factor: GpuFactor,
}

/// The device assembly of the reduced route and the LM loop's vectors
/// beside it.
struct Route<T> {
    /// An attempt's launches (damping, solve, trial point, trial cost) and
    /// an accept's with the linearization after it, captured as graphs,
    /// with what they were captured against; first, so they are dropped
    /// before the buffers they record.
    attempt_graph: Option<((arael_gpu::GraphKey, u64), arael_gpu::Graph)>,
    step_graph: Option<((arael_gpu::GraphKey, u64, u64), arael_gpu::Graph)>,
    asm: DeviceAssembly<T>,
    vec: LmVectors,
    /// The undamped diagonal of the last compute that read it back.
    diag_host: Vec<T>,
    /// H's tile values.
    nh: usize,
    /// The device holds H and the gradient at the loop's parameters (a
    /// structure's first compute assembles on the CPU).
    live: bool,
    /// The model's data on the device predates this solve.
    stale: bool,
    /// [`LmConfig::min_diagonal`](crate::simple_lm::LmConfig::min_diagonal).
    floor: Option<f64>,
    /// An accepted trial point waits to become the parameters: the next
    /// linearization, or the end of the solve, takes it.
    accept_due: bool,
}

impl<T> CubeclSolver<T> {
    /// The factor of `sn`, the analysis of the block structure `asym`, in `precision`.
    pub fn new(ctx: &GpuContext, precision: GpuPrecision, sn: &SupernodalSymbolic, asym: &SymbolicSparseBlockColMat<SparseIndex>) -> Self {
        CubeclSolver { route: None, schur: None, factor: GpuFactor::new(ctx, precision, sn, asym, &Default::default()) }
    }
}

/// An accepted trial point becomes the parameters: the reference rotations
/// moved by its steps (the last gather was the trial point's), the point
/// copied, its rotation steps zeroed.
fn accept_on_device<T: crate::utils::Float>(asm: &DeviceAssembly<T>, vec: &LmVectors) {
    asm.advance();
    vec.accept();
    asm.zero_steps(&vec.x(), vec.len());
}

/// Run `launches` (kernel launches only) as the graph `slot` holds when it
/// was captured under `key`; otherwise, with `graphs` on, run them once to
/// warm the pools and capture them into `slot`; without graphs, run them.
fn run_graph<K: PartialEq>(client: &arael_gpu::cubecl::prelude::Client, graphs: bool, slot: &mut Option<(K, arael_gpu::Graph)>, key: K, mut launches: impl FnMut() + Send) {
    if !graphs {
        launches();
        return;
    }
    if let Some((k, graph)) = slot.as_ref()
        && *k == key
    {
        // SAFETY: the graph records the route's buffers, the reduction's and
        // the factor's (both in the key), all alive while the route is; the
        // client is pinned to the stream it was captured on, which orders the
        // writes before and the read after it.
        unsafe { graph.replay() }.expect("replay a captured graph");
        return;
    }
    *slot = None;
    client.graph_prepare().expect("prepare a graph capture");
    launches();
    let graph = arael_gpu::capture(client, launches);
    *slot = Some((key, graph));
}

impl<T: crate::utils::Float + SchurReal> CubeclSolver<T> {
    /// The pending accept applied, without waiting.
    fn settle(&mut self) {
        let r = self.route.as_mut().expect("a device route");
        if std::mem::take(&mut r.accept_due) {
            accept_on_device(&r.asm, &r.vec);
        }
    }
}

impl<T: crate::utils::Float + SchurReal> GpuSolver<T> for CubeclSolver<T> {
    fn runtime(&self) -> &'static str {
        "cubecl"
    }

    fn precision(&self) -> GpuPrecision {
        self.factor.precision()
    }

    fn factorize(&mut self, vals: &[T]) -> Result<(), SupernodalError> {
        self.factor.factorize(vals)
    }

    fn solve(&mut self, x: &mut [T]) {
        self.factor.solve(x)
    }

    fn solve_f64(&mut self, x: &mut [f64]) {
        self.factor.solve(x)
    }

    fn prepare_reduction(&mut self, schur: Option<&SchurSymbolic<SparseIndex>>, hsym: &SymbolicSparseBlockColMat<SparseIndex>) -> Result<(), NativeError> {
        // the old buffers go back to the context first, so the new ones
        // reuse them rather than add to them
        self.drop_reduction();
        let ctx = self.factor.context();
        self.schur = Some(match schur {
            Some(s) => GpuSchur::new(ctx, s, hsym),
            None => GpuSchur::whole(ctx, hsym),
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

    fn solve_reduced(&mut self, hvals: &[T], grad: &[T], refine: Option<(usize, f64)>, delta: &mut [T]) -> Option<RefineStats> {
        let gs = self.schur.as_mut().expect("the device reduction is built with the Schur symbolic");
        gs.solve(&mut self.factor, hvals, grad, refine, delta).ok()
    }

    fn h_nonfinite(&self) -> usize {
        let gs = self.schur.as_ref().expect("a device route has its reduction");
        gs.h_values().iter().filter(|v| !v.is_finite()).count()
    }

    fn build_route(&mut self, root: &dyn DeviceRoot<T>, hsym: &SymbolicSparseBlockColMat<SparseIndex>, bdiag_pos: &[ValueIndex]) -> Result<(), NativeError> {
        self.route = None;
        let Some(gs) = self.schur.as_ref() else { return Ok(()) };
        let ctx = gs.context().clone();
        let mut resolver = PositionResolver::new(hsym);
        let asm = DeviceAssembly::new(&ctx, root, &mut |i, j| resolver.resolve_tile(i as usize, j as usize));
        self.route = Some(Route {
            attempt_graph: None,
            step_graph: None,
            asm,
            vec: LmVectors::new(&ctx, bdiag_pos),
            diag_host: vec![T::zero(); bdiag_pos.len()],
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

    fn route_compute(&mut self, problem: &mut dyn LmProblemInternals<T>, params: &[T], grad: &mut [T]) -> T {
        let gs = self.schur.as_mut().expect("a device route has its reduction");
        let r = self.route.as_mut().expect("a device route");
        if r.stale {
            problem.update_params(params);
            r.asm.refresh(problem.device_root().expect("a device route's problem has a device root"));
            r.stale = false;
        }
        let n = grad.len();
        let (h, g) = (gs.h_target(r.nh), gs.rhs_target());
        let cost = r.asm.assemble(params, &g, n, &h, r.nh);
        r.vec.gather_diag(&h, r.nh);
        let back = arael_gpu::read(r.asm.context().client(), vec![g, r.vec.diag()]).expect("read the gradient and the diagonal");
        narrow(grad, f64::from_bytes(&back[0]));
        narrow(&mut r.diag_host, f64::from_bytes(&back[1]));
        r.live = true;
        T::from(cost).unwrap()
    }

    fn route_diagonal(&self, diagonal: &mut [T]) {
        diagonal.copy_from_slice(&self.route.as_ref().expect("a device route").diag_host);
    }

    fn start(&mut self, x: &[T], floor: Option<f64>) {
        let r = self.route.as_mut().expect("a resident solve has its device route");
        r.floor = floor;
        r.accept_due = false;
        r.vec.write_x(x.iter().map(|&v| SchurReal::to_f64(v)).collect());
        if r.live {
            // this linearization was assembled there: its diagonal pass only
            let gs = self.schur.as_mut().expect("a device route has its reduction");
            let (h, g) = (gs.h_target(r.nh), gs.rhs_target());
            r.vec.linearized(&g, &h, r.nh, r.floor);
        } else {
            self.linearize();
        }
    }

    /// The pending accept, when there is one, and the linearization at the
    /// parameters after it (the assembly and the pass over the diagonal and
    /// the gradient), without waiting. An accept with its linearization is
    /// one captured graph (graphs on).
    fn linearize(&mut self) {
        let gs = self.schur.as_mut().expect("a device route has its reduction");
        let gf = &self.factor;
        let r = self.route.as_mut().expect("a device route");
        let accept = std::mem::take(&mut r.accept_due);
        let (h, g) = (gs.h_target(r.nh), gs.rhs_target());
        let key = (gs.graph_key(gf, None), r.asm.generation(), r.floor.map_or(u64::MAX, f64::to_bits));
        let (asm, vec, nh, floor) = (&r.asm, &r.vec, r.nh, r.floor);
        let launches = move || {
            if accept {
                accept_on_device(asm, vec);
            }
            let n = vec.len();
            asm.assemble_at(&vec.x(), n, &DeviceTargets { g: &g, ng: n, h: &h, nh, cost: &vec.slot(S_COST_X) });
            vec.linearized(&g, &h, nh, floor);
        };
        if accept {
            run_graph(asm.context().client(), gs.graphs(), &mut r.step_graph, key, launches);
        } else {
            launches();
        }
        r.live = true;
    }

    fn attempt(&mut self, lambda: f64, refine: Option<(usize, f64)>) -> (Option<RefineStats>, ResidentAttempt) {
        let gs = self.schur.as_mut().expect("a device route has its reduction");
        let gf = &mut self.factor;
        let r = self.route.as_mut().expect("a resident solve has its device route");
        let (h, g) = (gs.h_target(r.nh), gs.rhs_target());
        let key = (gs.graph_key(gf, refine), r.asm.generation());
        let graphs = gs.graphs();
        r.vec.set_lambda(lambda);
        {
            let (asm, vec, nh, solver) = (&r.asm, &r.vec, r.nh, &*gs);
            let launches = move || {
                vec.damp(&h, nh);
                solver.launch(gf, refine);
                vec.step(&solver.step_view(), &g);
                asm.cost_at(&vec.x_try(), vec.len(), &vec.slot(S_COST_TRY));
            };
            run_graph(asm.context().client(), graphs, &mut r.attempt_graph, key, launches);
        }
        let c = r.asm.context().client();
        let back = arael_gpu::read(c, vec![gs.status_view(), r.vec.scalars()]).expect("read the attempt");
        // the queue's rotation returns the staging of the lambda's write
        c.flush().expect("device flush");
        let solved = gs.outcome(&back[0], refine).ok();
        (solved, resident_attempt(solved.is_some(), &Scalars::decode(&back[1])))
    }

    fn accept(&mut self) {
        // applied with the next linearization, in its graph
        self.route.as_mut().expect("a resident solve has its device route").accept_due = true;
    }

    fn params(&mut self, x: &mut [T]) {
        self.settle();
        let r = self.route.as_ref().expect("a resident solve has its device route");
        let back = arael_gpu::read_one(r.asm.context().client(), r.vec.x()).expect("read the parameters");
        narrow(x, f64::from_bytes(&back));
    }

    fn vectors(&mut self, grad: &mut [T], diagonal: &mut [T], step: &mut [T]) {
        let gs = self.schur.as_mut().expect("a device route has its reduction");
        let r = self.route.as_ref().expect("a resident solve has its device route");
        let back = arael_gpu::read(r.asm.context().client(), vec![gs.rhs_target(), r.vec.diag(), gs.step_view()]).expect("read the vectors");
        for (out, b) in [grad, diagonal, step].into_iter().zip(&back) {
            narrow(out, f64::from_bytes(b));
        }
    }

    fn nonfinite(&mut self) -> [usize; 4] {
        let gs = self.schur.as_mut().expect("a device route has its reduction");
        let r = self.route.as_ref().expect("a resident solve has its device route");
        let h = gs.h_target(r.nh);
        let back = arael_gpu::read(r.asm.context().client(), vec![gs.rhs_target(), r.vec.diag(), r.vec.x(), h]).expect("read the vectors");
        let count = |b: &[u8]| f64::from_bytes(b).iter().filter(|v| !v.is_finite()).count();
        [count(&back[0]), count(&back[1]), count(&back[2]), count(&back[3])]
    }

    fn finish(&mut self, problem: &mut dyn LmProblemInternals<T>, x: &mut [T]) {
        self.params(x);
        let r = self.route.as_ref().expect("a resident solve has its device route");
        r.asm.restore(problem.device_root_mut().expect("a device route's problem has a device root"));
    }
}
