//! The GPU side of [`SparseFaer`](crate::simple_lm::SparseFaer):
//! [`GpuSolver`], implemented by [`solver::DeviceSolver`] for every
//! [`Device`](arael_gpu::Device).

use arael_faer::bsc::SymbolicSparseBlockColMat;
use arael_faer::cg::CgParams;
use arael_faer::schur::SchurSymbolic;
use arael_faer::supernodal::SupernodalSymbolic;
use arael_faer::{SparseIndex, ValueIndex};
use arael_gpu::GpuError;
use arael_gpu::{GpuCgStats, GpuPrecision, RefineStats};

use crate::device::DeviceRoot;
use crate::simple_lm::{DeviceFailure, GpuSolve, LmProblemInternals, ResidentAttempt};

pub(crate) mod solver;

/// How a device solver solves its system (the reduced one, or the whole
/// one with nothing eliminated), chosen when it is built.
pub(crate) enum Method<'a> {
    /// The block supernodal Cholesky factor of the symbolic `sn`, the
    /// analysis of the block structure `asym`, in `precision`, its solves
    /// refined against the system under `refine` (`(max_steps,
    /// tolerance)`: an f32 factor under mixed precision).
    Factor {
        precision: GpuPrecision,
        sn: &'a SupernodalSymbolic,
        asym: &'a SymbolicSparseBlockColMat<SparseIndex>,
        refine: Option<(usize, f64)>,
    },
    /// Preconditioned conjugate gradients.
    Cg(CgParams),
}

/// What a damped solve on the device took.
#[derive(Clone, Copy, Debug)]
pub(crate) enum SolveStats {
    /// A factor's solve: its refinement (no step without one).
    Refined(RefineStats),
    /// Conjugate gradients.
    Cg(GpuCgStats),
}

/// The GPU side of the solver: what solves the system on one device (a
/// factor or conjugate gradients, [`Method`]), and two layers built on it:
///
/// - with [`prepare_reduction`](Self::prepare_reduction), the reduction,
///   of the reduced system or of the whole one with nothing eliminated: a
///   damped solve ([`solve_reduced`](Self::solve_reduced)) forms S from
///   the damped H, solves it (a factor: factorizes, solves, refines,
///   waiting once; conjugate gradients: a wait a round of iterations) and
///   back-substitutes on the device;
/// - with [`build_route`](Self::build_route), the device assembly and the
///   LM loop's vectors: from the first linearization on, the loop runs on
///   the device ([`ResidentLoop`](crate::simple_lm::ResidentLoop)), the
///   parameters f64 there whatever `T`.
///
/// Every device operation can fail with a [`DeviceFailure`], which ends
/// the solve; a damped solve's `None` is the numerical outcome the loop
/// raises lambda on. `Send`, so a solver moves between threads as a host
/// solver does.
pub(crate) trait GpuSolver<T>: Send {
    /// How the system is solved, as the report says it.
    fn solve(&self) -> GpuSolve;
    /// The device: its name, its compile target, whether it computes in
    /// f64 (Metal does not: an f64 root assembles on the host there).
    fn arch(&self) -> &arael_gpu::Arch;
    /// Time the device's work from here on (`on`), or stop: stamps in its
    /// queue around each linearization and attempt, their spans read at the
    /// waits a solve already makes. Starts the times from zero.
    fn set_timing(&mut self, on: bool);
    /// The device's times since [`set_timing`](Self::set_timing)`(true)`;
    /// `None` when not timing, and on a device that keeps no stamps.
    fn timing(&self) -> Option<crate::simple_lm::GpuTiming>;

    /// Build the reduction by `schur` of matrices with the structure
    /// `hsym`, or with `None` the whole system (nothing eliminated), on the
    /// solver's device; a factor is the reduced system's, and conjugate
    /// gradients are built over the reduction. Replaces any earlier
    /// reduction and route.
    fn prepare_reduction(&mut self, schur: Option<&SchurSymbolic<SparseIndex>>, hsym: &SymbolicSparseBlockColMat<SparseIndex>, threads: usize) -> Result<(), GpuError>;
    /// Release the reduction and the route.
    fn drop_reduction(&mut self);
    /// Whether a reduction is built.
    fn reduces(&self) -> bool;
    /// Whether the reduction computes in f32
    /// (`GpuSchurParams::f32_reduction`).
    fn f32_reduction(&self) -> bool;
    /// One damped solve on the device from the damped H's tile values and
    /// the gradient, the step into `delta`, with what it took; `None` when
    /// an eliminated block or S is not positive definite.
    fn solve_reduced(&mut self, hvals: &[T], grad: &[T], delta: &mut [T]) -> Result<Option<SolveStats>, DeviceFailure>;

    /// Lay `root` out for the device assembly against the reduction's
    /// structure `hsym`, whose scalar diagonal is at `bdiag_pos` in its
    /// values; the model's working copies are current. Nothing without a
    /// reduction.
    fn build_route(&mut self, root: &dyn DeviceRoot<T>, hsym: &SymbolicSparseBlockColMat<SparseIndex>, bdiag_pos: &[ValueIndex], threads: usize) -> Result<(), GpuError>;
    /// Whether the route is built.
    fn has_route(&self) -> bool;
    /// Whether the device holds the last compute's H and gradient.
    fn route_live(&self) -> bool;
    /// The model's data on the device predates this solve: the solve's
    /// first compute reads it again.
    fn mark_stale(&mut self);
    /// Assemble at `params` on the device into the reduction's inputs,
    /// returning the cost; the gradient and the undamped diagonal come back
    /// to the host.
    fn route_compute(&mut self, problem: &mut dyn LmProblemInternals<T>, params: &[T], grad: &mut [T]) -> Result<T, DeviceFailure>;
    /// The undamped diagonal of the last [`route_compute`](Self::route_compute).
    fn route_diagonal(&self, diagonal: &mut [T]);
    /// A damped solve of the last [`route_compute`](Self::route_compute)'s
    /// system for a caller that drives the loop from the host
    /// ([`LmSolver::solve_damped`](crate::simple_lm::LmSolver::solve_damped)):
    /// H's diagonal set to `diagonal + lambda damp` on the device, the
    /// system solved there and the step read into `delta`, with what the
    /// solve took. `None` for a system that is not positive definite, which
    /// leaves `delta`.
    fn route_solve(&mut self, diagonal: &[T], damp: &[T], lambda: f64, delta: &mut [T]) -> Result<Option<SolveStats>, DeviceFailure>;

    /// [`ResidentLoop::start`](crate::simple_lm::ResidentLoop::start).
    fn start(&mut self, x: &[T], floor: Option<f64>) -> Result<(), DeviceFailure>;
    /// [`ResidentLoop::linearize`](crate::simple_lm::ResidentLoop::linearize).
    fn linearize(&mut self) -> Result<(), DeviceFailure>;
    /// [`ResidentLoop::attempt`](crate::simple_lm::ResidentLoop::attempt)
    /// at `lambda`, with what the damped solve took (`None` for a system
    /// that is not positive definite).
    fn attempt(&mut self, lambda: f64) -> Result<(Option<SolveStats>, ResidentAttempt), DeviceFailure>;
    /// [`ResidentLoop::accept`](crate::simple_lm::ResidentLoop::accept).
    fn accept(&mut self);
    /// [`ResidentLoop::params`](crate::simple_lm::ResidentLoop::params).
    fn params(&mut self, x: &mut [T]) -> Result<(), DeviceFailure>;
    /// [`ResidentLoop::vectors`](crate::simple_lm::ResidentLoop::vectors).
    fn vectors(&mut self, grad: &mut [T], diagonal: &mut [T], step: &mut [T]) -> Result<(), DeviceFailure>;
    /// [`ResidentLoop::nonfinite`](crate::simple_lm::ResidentLoop::nonfinite).
    fn nonfinite(&mut self) -> Result<[usize; 4], DeviceFailure>;
    /// [`ResidentLoop::finish`](crate::simple_lm::ResidentLoop::finish).
    fn finish(&mut self, problem: &mut dyn LmProblemInternals<T>, x: &mut [T]) -> Result<(), DeviceFailure>;
}

/// The scalars of an attempt as [`ResidentAttempt`], `solved` from its outcome.
fn resident_attempt(solved: bool, s: &arael_gpu::lm::Scalars) -> ResidentAttempt {
    use crate::simple_lm::DiagonalFault;
    use arael_gpu::lm::DiagFault;
    ResidentAttempt {
        solved,
        new_cost: s.cost_try,
        step2: s.step2,
        try2: s.try2,
        predicted: s.predicted,
        grad_max: s.grad_max,
        grad_scaled: s.grad_scaled,
        fault: s.fault.map(|(i, f)| {
            (i, match f {
                DiagFault::Nan => DiagonalFault::Nan,
                DiagFault::Negative => DiagonalFault::Negative,
                DiagFault::Zero => DiagonalFault::Zero,
            })
        }),
    }
}

/// `src`, f64 on the device, into `dst` in the solver's scalar.
fn narrow<T: crate::utils::Float>(dst: &mut [T], src: &[f64]) {
    for (d, &v) in dst.iter_mut().zip(src) {
        *d = T::from(v).unwrap();
    }
}
