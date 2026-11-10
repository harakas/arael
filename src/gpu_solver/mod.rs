//! The GPU side of the supernodal solver ([`SparseFaer`](crate::simple_lm::SparseFaer)):
//! [`GpuSolver`], implemented by [`solver::DeviceSolver`] for every
//! [`Device`](arael_gpu::Device).

use arael_faer::bsc::SymbolicSparseBlockColMat;
use arael_faer::schur::SchurSymbolic;
use arael_faer::supernodal::SupernodalError;
use arael_faer::{SparseIndex, ValueIndex};
use arael_gpu::GpuError;
use arael_gpu::{GpuPrecision, RefineStats};

use crate::device::DeviceRoot;
use crate::simple_lm::{DeviceFailure, LmProblemInternals, ResidentAttempt};

pub(crate) mod solver;

/// The GPU side of the supernodal solver, in three layers built one on the
/// other:
///
/// - the factor of one symbolic analysis, factorized and solved from host
///   values (the reduction on the CPU);
/// - with [`prepare_reduction`](Self::prepare_reduction), the reduction
///   too: an attempt forms S from the damped H, factorizes, solves, refines
///   and back-substitutes on the device, waiting once;
/// - with [`build_route`](Self::build_route), the device assembly and the
///   LM loop's vectors: from the first linearization on, the loop runs on
///   the device ([`ResidentLoop`](crate::simple_lm::ResidentLoop)), the
///   parameters f64 there whatever `T`.
///
/// Every device operation can fail with a [`DeviceFailure`], which ends
/// the solve; the inner results of a factorization or a damped solve are
/// the numerical outcomes the loop raises lambda on. `Send`, so a solver
/// moves between threads as on the CPU.
pub(crate) trait GpuSolver<T>: Send {
    /// The device's runtime and the wide scalar, for the verbose log:
    /// `cuda`, `cuda df32`, `hip`, `hip df32`.
    fn runtime(&self) -> String;
    /// The factor's scalar.
    fn precision(&self) -> GpuPrecision;
    /// Whether the device computes in f64 (Metal does not): an f64 root
    /// assembles on the CPU without it.
    fn has_f64(&self) -> bool;

    /// Factorize the matrix whose tile values are `vals`, in the storage
    /// order of the structure the factor was built for; the inner `Err`
    /// for a matrix that is not positive definite.
    fn factorize(&mut self, vals: &[T]) -> Result<Result<(), SupernodalError>, DeviceFailure>;
    /// Solve in place with the last factorization.
    fn solve(&mut self, x: &mut [T]) -> Result<(), DeviceFailure>;
    /// [`solve`](Self::solve) of an f64 vector: a refinement's correction.
    fn solve_f64(&mut self, x: &mut [f64]) -> Result<(), DeviceFailure>;

    /// Build the reduction by `schur` of matrices with the structure
    /// `hsym`, or with `None` the whole system (nothing eliminated), on the
    /// factor's device; the factor is the reduced system's. Replaces any
    /// earlier reduction and route.
    fn prepare_reduction(&mut self, schur: Option<&SchurSymbolic<SparseIndex>>, hsym: &SymbolicSparseBlockColMat<SparseIndex>) -> Result<(), GpuError>;
    /// Release the reduction and the route.
    fn drop_reduction(&mut self);
    /// Whether a reduction is built.
    fn reduces(&self) -> bool;
    /// One damped solve on the device from the damped H's tile values and
    /// the gradient, refined under `refine` (`(max_steps, tolerance)`), the
    /// step into `delta`; `None` when an eliminated block or S is not
    /// positive definite.
    fn solve_reduced(&mut self, hvals: &[T], grad: &[T], refine: Option<(usize, f64)>, delta: &mut [T]) -> Result<Option<RefineStats>, DeviceFailure>;

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

    /// [`ResidentLoop::start`](crate::simple_lm::ResidentLoop::start).
    fn start(&mut self, x: &[T], floor: Option<f64>) -> Result<(), DeviceFailure>;
    /// [`ResidentLoop::linearize`](crate::simple_lm::ResidentLoop::linearize).
    fn linearize(&mut self) -> Result<(), DeviceFailure>;
    /// [`ResidentLoop::attempt`](crate::simple_lm::ResidentLoop::attempt)
    /// at `lambda`, refined under `refine`, with the refinement's outcome
    /// (`None` for a system that is not positive definite).
    fn attempt(&mut self, lambda: f64, refine: Option<(usize, f64)>) -> Result<(Option<RefineStats>, ResidentAttempt), DeviceFailure>;
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
