// arael C++ solver types, shared by every generated interface: the
// enums, the configuration, the options, the result and its reports,
// and the error types of the Levenberg-Marquardt (LM) solver. A
// generated header instantiates the templates at its root's precision
// (`using LmResult = LmResultT<double>;`). The structs' layouts are part
// of the C ABI, so their field order must not change.
#pragma once

#include <cstdint>
#include <stdexcept>
#include "result.hpp"

namespace arael {

/// A Rust panic caught at the FFI boundary -- a programmer error (a
/// stale ref, an unguarded Option read; validate() reports these as
/// text). The model's parameters are unchanged, and a session in use
/// was invalidated; the message is the Rust panic text.
struct PanicError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

/// Why a solve stopped. Non-negative codes come from the solver;
/// SolverFailed carries text via last_error(), Panicked likewise.
enum class LmStatus : int32_t {
    Converged = 0,
    CostThreshold = 1,
    MaxIterations = 2,
    GradientTolerance = 3,
    ParameterTolerance = 4,
    PredictedReduction = 5,
    LambdaCeiling = 6,
    DriverTerminated = 7,
    ObserverTerminated = 8,
    TimeLimit = 9,
    RetryBudgetExhausted = 10,
    Aborted = 11,
    SolverFailed = -1,
    Panicked = -2,
};

/// Rust's LmStatus::is_success: did the solve reach a minimum, as
/// opposed to running out of something?
inline bool is_success(LmStatus s) {
    switch (s) {
        case LmStatus::Converged:
        case LmStatus::CostThreshold:
        case LmStatus::GradientTolerance:
        case LmStatus::ParameterTolerance:
        case LmStatus::PredictedReduction:
        case LmStatus::DriverTerminated:
        case LmStatus::ObserverTerminated:
            return true;
        default:
            return false;
    }
}

/// Rust's LmStatus::as_str (plus the two skin-only codes).
inline const char* as_str(LmStatus s) {
    switch (s) {
        case LmStatus::Converged: return "converged";
        case LmStatus::CostThreshold: return "cost threshold reached";
        case LmStatus::MaxIterations: return "hit max_iters";
        case LmStatus::GradientTolerance: return "gradient flat";
        case LmStatus::ParameterTolerance: return "step negligible";
        case LmStatus::PredictedReduction: return "predicted gain negligible";
        case LmStatus::LambdaCeiling: return "damping exhausted";
        case LmStatus::DriverTerminated: return "driver stopped it";
        case LmStatus::ObserverTerminated: return "observer stopped it";
        case LmStatus::TimeLimit: return "out of time";
        case LmStatus::RetryBudgetExhausted: return "retry budget exhausted";
        case LmStatus::Aborted: return "aborted";
        case LmStatus::SolverFailed: return "solver failed";
        case LmStatus::Panicked: return "panicked";
    }
    return "?";
}

/// What broke a solve (mirrors Rust's SolveFailureKind with its
/// Setup payload flattened in).
enum class SolveFailureKind : int32_t {
    /// No structured failure stored (the last solve did not fail).
    None = 0,
    /// A Hessian element fell outside the declared band half-bandwidth.
    BandOverflow = 1,
    /// A parameter no constraint touches: the system is singular.
    UnconstrainedParameter = 2,
    /// The symbolic Cholesky factorization failed.
    SymbolicFactorization = 3,
    /// Two blocks selected for marginalization are coupled by a
    /// constraint.
    CoupledMarginalization = 4,
    /// A block selected for marginalization has no diagonal tile.
    MarginalizeMissingDiagonal = 5,
    /// The marginalization set is not a valid whole-block set.
    BadMarginalizeSet = 6,
    /// Iterative Schur was asked for but nothing was marginalized.
    IterativeSchurWithoutReduction = 7,
    /// The requested solver is not available: not compiled in (a GPU
    /// runtime among them), or not for this scalar (f64 on Metal).
    SolverUnavailable = 8,
    /// A parameter's Gauss-Newton diagonal went bad at this iterate.
    DegenerateDiagonal = 9,
    /// A GPU operation failed, in the setup or a step of the solve; the
    /// error's message holds the operation and the runtime's message.
    Device = 10,
};

/// How a Gauss-Newton diagonal entry was bad (DegenerateDiagonal).
enum class DiagonalFault : int32_t {
    /// Not applicable (the failure is not DegenerateDiagonal).
    None = -1,
    /// NaN: a residual or derivative evaluated to NaN.
    Nan = 0,
    /// Negative: impossible for a sum of squares; poisoned assembly.
    Negative = 1,
    /// Zero: no constraint curvature reaches the parameter here.
    Zero = 2,
};

/// The structured failure behind a SolverFailed result (Rust's
/// SolveFailureKind with the setup error's data flattened in): the
/// kind and the indices a caller can act on (-1 where not applicable).
struct SolveFailure {
    SolveFailureKind kind = SolveFailureKind::None;
    /// DegenerateDiagonal only.
    DiagonalFault fault = DiagonalFault::None;
    /// Scalar parameter index (UnconstrainedParameter,
    /// DegenerateDiagonal).
    int64_t param = -1;
    /// Element row (BandOverflow) or block row (CoupledMarginalization).
    int64_t row = -1;
    /// Element column (BandOverflow) or block column
    /// (CoupledMarginalization).
    int64_t col = -1;
    /// The declared half-bandwidth (BandOverflow).
    int64_t kd = -1;
    /// Block index (MarginalizeMissingDiagonal).
    int64_t block = -1;
    /// SymbolicFactorization: the reduced system, not the whole one.
    bool reduced = false;
};

/// The base preset a config starts from; it also supplies the one
/// Rust field the struct does not expose, the lambda driver.
/// IllConditioned selects the Nielsen driver (its other fields match
/// Conservative); Defaults and Conservative are the same config.
enum class LmPreset : uint32_t {
    Defaults = 0,
    Conservative = 1,
    WellConditioned = 2,
    IllConditioned = 3,
};

/// Severity threshold for arael's diagnostics (mirrors arael's
/// log::Level): a level admits everything at or below itself.
enum class LogLevel : uint32_t {
    Off = 0,
    Error = 1,
    Warn = 2,
    Info = 3,
};

/// One damped attempt, as the observer callback sees it. `params`
/// points at the CURRENT parameter vector for this attempt; valid
/// only during the callback.
template<class F>
struct LmIterT {
    uint32_t iter;
    uint32_t inner;
    bool accepted;
    bool factorization_failed;
    F cost;
    F new_cost;
    F lambda;
    uint32_t accepted_total;
    const F* params;
    uint32_t params_len;
};

enum class CovMode : uint32_t;
enum class CovOrdering : uint32_t;
enum class BlockSupernodalMode : uint32_t;

/// The solver configuration, holding the preset's Rust values.
/// Inspect them, edit them, pass the struct back whole; `option`
/// fields mirror the Rust `Option` fields (assign a value or `{}`).
/// `observer` (with `observer_user` passed back as its first
/// argument) is called once per damped attempt; return false to stop
/// the solve (status ObserverTerminated).
template<class F>
struct LmConfigT {
    LmPreset preset;
    uint32_t max_iters;
    uint32_t min_iters;
    uint32_t patience;
    uint32_t num_threads;
    bool verbose;
    bool gather_timing;
    F abs_precision;
    F rel_precision;
    F initial_lambda;
    F cost_threshold;
    F lambda_floor;
    option<F> gradient_tolerance;
    option<F> parameter_tolerance;
    option<F> predicted_reduction_tolerance;
    option<F> min_diagonal;
    /// Rust's `time_limit: Option<Duration>`, in seconds.
    option<double> time_limit_seconds;
    bool (*observer)(void*, const LmIterT<F>*);
    void* observer_user;
    /// Threads for the cost and assembly sweeps alone. Empty leaves them
    /// on `num_threads`; a value gives the sweeps that count and leaves
    /// the linear solve on `num_threads`.
    option<uint32_t> assembly_threads;
    /// Accepted steps after which the solve stops (status
    /// MaxIterations). Empty turns it off.
    option<uint32_t> max_accepted_iters;
    /// Assemble the parameter covariance at the solution in this mode
    /// and carry it in the result (LmResult::covariance()). Empty
    /// assembles nothing.
    option<CovMode> covariance;
    /// The elimination ordering of that covariance's assembly.
    CovOrdering covariance_ordering;
    /// Whether that assembly factorizes with the block supernodal
    /// Cholesky.
    BlockSupernodalMode covariance_block_supernodal;
    /// With the LM loop on a GPU (GpuWork::Loop), read the parameters
    /// back after each accepted step, for the observer; off, the
    /// observer's `params` are empty there.
    bool device_observer_params;
};

/// Whether and when the sparse backend marginalizes (mirrors arael's
/// SchurPolicy). Auto prices the reduction against factorizing the
/// whole system; its tuning lives in SparseOptionsT.
enum class SchurPolicy : uint32_t {
    Auto = 0,
    Force = 1,
    Never = 2,
};

/// Elimination ordering of the factorized system (mirrors arael's
/// SolveOrdering).
enum class SolveOrdering : uint32_t {
    Auto = 0,
    Amd = 1,
    MarginalizeFirst = 2,
    Natural = 3,
    NestedDissection = 4,
};

/// Former name of SolveOrdering.
using FaerOrdering [[deprecated("renamed to SolveOrdering")]] = SolveOrdering;

/// How the reduced Schur system is factored (mirrors arael's
/// EnvelopeMode): in block form under its envelope, or by the general
/// sparse Cholesky. The envelope route uses less memory on suitable
/// systems; Auto prices it per problem.
enum class EnvelopeMode : uint32_t {
    Auto = 0,
    Always = 1,
    Never = 2,
};

/// When the block supernodal Cholesky factorizes instead of faer's
/// scalar one (mirrors arael's BlockSupernodalMode), in the seats the
/// scalar route otherwise holds: the whole Hessian, and a reduced
/// Schur system the envelope route declined. Auto takes it wherever
/// the model has block structure, at any thread count.
enum class BlockSupernodalMode : uint32_t {
    Auto = 0,
    Always = 1,
    Never = 2,
};

/// How the reduced Schur system is solved (mirrors arael's
/// SchurMethod). The iterative routes need a reduction: pair them with
/// SchurPolicy::Force, or a solve without one fails.
enum class SchurMethod : uint32_t {
    /// Cholesky factorization.
    Factorize = 0,
    /// Preconditioned conjugate gradients on the formed reduced system;
    /// runs on a GPU.
    Iterative = 1,
    /// The same conjugate gradients without forming the reduced system;
    /// on the host only.
    IterativeImplicit = 2,
};

/// Former name of SchurMethod.
using SchurSolve [[deprecated("renamed to SchurMethod")]] = SchurMethod;

/// The GPU runtime of SparseOptionsT::gpu.
enum class GpuBackend : uint32_t {
    /// No GPU; the solve runs on the host.
    None = 0,
    /// CUDA, on an NVIDIA GPU.
    Cuda = 1,
    /// HIP, on an AMD GPU.
    Hip = 2,
    /// The CPU device: the GPU kernels run on the host, for tests without
    /// a GPU.
    Cpu = 3,
    /// Metal, on Apple silicon, which has no f64.
    Metal = 4,
};

/// Whether `b`'s GPUs compute in f64: every runtime but Metal. Without
/// f64 an f64 solve that factorizes needs mixed_precision, and gpu_wide
/// F64 fails the solve.
constexpr bool has_f64(GpuBackend b) { return b != GpuBackend::Metal; }

/// The GPU's wide scalar (SparseOptionsT::gpu_wide): what the GPU
/// computes in where the host computes in f64, such as an f64 model's
/// values, the norms and the refinement's residual.
enum class GpuWide : uint32_t {
    /// f64.
    F64 = 0,
    /// Two f32 whose sum is the value, about 48 bits of significand, in
    /// f32 arithmetic only. For GPUs whose f64 arithmetic is slow.
    Df32 = 1,
    /// The GPU's default: Df32 on Metal, else F64.
    Default = 2,
};

/// Where the GPU assembly writes the model's data before copying it to
/// the GPU (SparseOptionsT::gpu_staging).
enum class GpuStaging : uint32_t {
    /// Pinned host memory, whose copies run while the host goes on.
    Pinned = 0,
    /// Ordinary host memory, which the runtime copies itself. For a
    /// platform where copies from pinned memory are slow.
    Pageable = 1,
};

/// The sparse backend's options (mirrors arael's SparseFaerOptions).
/// The generated SparseOptions holds the Rust defaults; edit fields and
/// pass it to solve_sparse. The marginalize range list is not exposed:
/// the model's marginalize hint covers it.
struct SparseOptionsT {
    /// Whether and when to marginalize (Rust's `policy`).
    SchurPolicy schur;
    /// The elimination ordering of the factorized system.
    SolveOrdering ordering;
    /// Whether the reduced system is factored under its envelope.
    EnvelopeMode envelope;
    /// Envelope panel width; 0 picks it automatically.
    uint32_t envelope_panel_width;
    /// Factorize supernodally, in dense panels.
    bool supernodal;
    /// Factor a banded whole system in block form under its band.
    bool narrow_band;
    /// SchurPolicy::Auto tuning: the factor by which the reduction must
    /// beat the whole system in flops to be taken.
    double flop_margin;
    /// SchurPolicy::Auto tuning: below this cheap ratio the reduction is
    /// taken without the exact pricing.
    double obvious_flop_ratio;
    /// Conjugate-gradient tolerance of the iterative routes.
    double cg_tol;
    /// How the reduced system is solved.
    SchurMethod schur_solve;
    /// Conjugate-gradient iteration cap; 0 = unlimited.
    uint32_t cg_max_iters;
    /// Conjugate-gradient restart interval; 0 = never.
    uint32_t cg_restart_every;
    /// When the block supernodal Cholesky factorizes.
    BlockSupernodalMode block_supernodal;
    /// Update-batching acceptance ratio for that route; 0 disables
    /// batching.
    double block_supernodal_batch;
    /// Memory-lean amalgamation for that route: a smaller factor, at
    /// some cost in speed.
    bool block_supernodal_memory_lean;
    /// Factor in f32 on that route and refine each damped solve against
    /// the matrix (arael's MixedPrecision). On a GPU it also sets the
    /// precision of the system the GPU solves (docs/SOLVERS.md#precision).
    /// Off by default.
    bool mixed_precision;
    /// Refinement steps at most per damped solve, under mixed_precision.
    uint32_t max_refinements;
    /// The relative residual at which refinement stops, under
    /// mixed_precision.
    double refine_tol;
    /// The GPU runtime to solve on; None, the default, solves on the
    /// host. The interface must be built with that runtime (ARAEL_GPU),
    /// or the solve fails with SolverUnavailable.
    GpuBackend gpu;
    /// The GPU of that runtime, as the runtime numbers them; 0 by
    /// default.
    uint32_t gpu_index;
    /// Assemble on the GPU too and keep the LM loop there, when the model
    /// has a device form; LmResult::gpu() says whether it did. On by
    /// default; off, the host assembles.
    bool gpu_assembly;
    /// The GPU's wide scalar; Default by default.
    GpuWide gpu_wide;
    /// Where the GPU assembly writes the model's data; Pinned by default.
    GpuStaging gpu_staging;
};

/// Where a solve's one-time setup went: the parts of LmTiming::analysis
/// the sparse backend times, in seconds. A part a route does not have is
/// zero; the parts add up to at most the analysis.
struct SetupTiming {
    /// Finding the model's blocks and the ones to marginalize.
    double blocks;
    /// The Hessian's block pattern, the blocks' positions in it and, when
    /// reducing, the reduced system's pattern.
    double structure;
    /// On a GPU, opening it: the process's first solve on that GPU
    /// creates its state; later solves take nothing here.
    double gpu_open;
    /// Ordering the system that is factorized.
    double ordering;
    /// Pricing the reduced route against the whole system; zero when the
    /// comparison was skipped.
    double weighing;
    /// The symbolic factorization and the factor's buffers; on a GPU, the
    /// GPU factor's plan and kernels.
    double factor;
    /// On a GPU, setting the reduction up there.
    double gpu_plan;
    /// On a GPU that assembles, setting the GPU assembly up: the model's
    /// data written for the GPU, and its kernels.
    double gpu_layout;
};

/// The GPU's own time in a GPU solve, in seconds. The host's phases show
/// waits, not the GPU's work: with the LM loop on the GPU, LmTiming's
/// assembly and cost_eval read near zero and linear_solve holds the GPU's
/// assembly and trial cost with its solve.
struct GpuTiming {
    /// The linearizations on the GPU, summed; zero when the host
    /// assembles.
    double assembly;
    /// The damped attempts on the GPU, summed; when the host assembles,
    /// from the upload of the Hessian's values.
    double attempts;
    /// The linearizations timed.
    uint32_t assembly_count;
    /// The attempts timed.
    uint32_t attempt_count;
    /// The part of `attempts` spent in conjugate gradients, the waits
    /// between their batches included; zero for a solve that factorizes.
    double cg;
    /// The batches of conjugate-gradient iterations, each a wait for the
    /// GPU.
    uint32_t cg_rounds;
};

/// Per-phase wall-clock seconds plus call counts, gathered when
/// LmConfigT::gather_timing is set; the per-attempt records are
/// LmResult::steps().
///
///     if (r.has_timing && r.timing.gpu)
///         printf("an attempt takes %.3f ms on the GPU\n",
///                1e3 * r.timing.gpu->attempts / r.timing.gpu->attempt_count);
struct LmTiming {
    /// The whole solve.
    double total;
    /// The model's residual, Jacobian and Hessian work, over all
    /// iterations.
    double assembly;
    /// The first assembly.
    double first_assembly;
    /// The backend's one-time work on the structure: the pattern, the
    /// ordering, the symbolic factorization.
    double analysis;
    /// The damped numeric factorizations and solves, over all attempts.
    double linear_solve;
    /// The first linear solve.
    double first_linear_solve;
    /// The trial-point cost evaluations.
    double cost_eval;
    /// The first cost evaluation.
    double first_cost_eval;
    /// The post-step re-centering.
    double advance;
    /// The first re-centering.
    double first_advance;
    uint32_t assembly_count;
    uint32_t analysis_count;
    uint32_t linear_solve_count;
    uint32_t cost_eval_count;
    uint32_t advance_count;
    /// The covariance assembly the config asked for, after the solve
    /// and outside `total`; zero when none was asked.
    double covariance;
    /// Where `analysis` went, from the sparse backend; empty otherwise.
    option<SetupTiming> setup;
    /// The GPU's own time, in a solve on a GPU; empty without one.
    option<GpuTiming> gpu;
};

/// One attempted LM step: the per-attempt timeline behind
/// LmResult::steps() (Rust's LmStep). A damping retry is its own
/// record, so rejected steps and failed factorizations appear too.
/// Durations are wall-clock seconds; assembly is charged to the
/// first attempt at each linearization (inner == 0), zero on a
/// retry.
struct LmStep {
    /// Attempt number, 1-based, counting damping retries.
    uint32_t iter;
    /// Retry index within this linearization (0 = first attempt).
    uint32_t inner;
    /// Did the step reduce the cost and get kept?
    bool accepted;
    /// Damped Cholesky not positive definite: no step was produced,
    /// new_cost is NaN and step_norm is zero.
    bool factorization_failed;
    double lambda;
    /// Cost before the attempt / at the trial point.
    double cost;
    double new_cost;
    /// |delta|_2 of the proposed move.
    double step_norm;
    /// max|g_i| at this linearization point.
    double grad_max;
    /// The whole attempt (excludes assembly and analysis).
    double time;
    double assembly;
    double analysis;
    double linear_solve;
    double cost_eval;
    /// Post-step re-centering; zero unless the step was accepted.
    double advance;
};

template<class F>
struct LmResultT {
    F start_cost;
    F end_cost;
    uint32_t iterations;
    uint32_t accepted_iterations;
    LmStatus status;
    F final_lambda;
    /// Valid iff has_timing (config.gather_timing was set).
    LmTiming timing;
    bool has_timing;
    /// The full Rust result behind this one (reports, the backend's
    /// plan). Owned: the generated LmResult class manages it; null on
    /// a failure that produced no partial result.
    void* detail;
};

/// How the reduced system was ordered (mirrors arael's
/// ReducedOrdering).
enum class ReducedOrdering : int32_t {
    NaturalBanded = 0,
    NaturalDense = 1,
    Amd = 2,
    Nd = 3,
    Natural = 4,
};

/// Per-iteration flops down the two routes the Auto policy priced:
/// the reduction plus the reduced factor, against the whole system's
/// factor.
struct RouteFlops {
    double reduced;
    double full;
};

/// What the sparse backend decided (mirrors arael's SchurPlan); read
/// it with LmResult::plan().
struct SchurPlan {
    /// Whether the Schur reduction was used; false means the full
    /// system was factorized instead.
    bool reduced;
    /// Parameter blocks marginalized.
    uint32_t eliminated_blocks;
    /// Parameters marginalized.
    uint32_t eliminated_params;
    /// Parameters left in the reduced system.
    uint32_t kept_params;
    /// fill(L_S) / fill(L_H), when the exact route comparison ran.
    option<double> fill_ratio;
    /// The exact comparison's verdict; present exactly when
    /// fill_ratio is.
    option<RouteFlops> route_flops;
    /// Conjugate-gradient iterations, when the reduced system was
    /// solved iteratively.
    option<uint32_t> cg_iterations;
    /// The cheap statistic the Auto policy screens with; present
    /// whenever the policy was Auto.
    option<double> flop_ratio;
    /// How the reduced system was ordered; absent when there was no
    /// reduction.
    option<ReducedOrdering> ordering;
    /// The reduced system's half-bandwidth in scalars; 0 when there
    /// was no reduction.
    uint32_t kept_bandwidth;
    /// Factored in block form under its envelope rather than by the
    /// general sparse Cholesky: the reduced system when `reduced`,
    /// the whole Hessian otherwise.
    bool envelope;
    /// Factored by the block supernodal Cholesky rather than faer's
    /// scalar one, over the same system `envelope` refers to.
    bool block_supernodal;
    /// The factor was f32: an f32 solve's, or one under mixed_precision.
    bool f32_factor;
    /// Refinement steps summed over the solve, under mixed_precision.
    option<uint32_t> refinements;
    /// Damped solves that ended above the refinement tolerance, under
    /// mixed_precision.
    option<uint32_t> refinement_shortfalls;
};

/// How the GPU solved the system, the reduced one or the whole one
/// (GpuReport::solve).
enum class GpuSolve : uint32_t {
    /// Not at all: the solve did not use the GPU.
    None = 0,
    /// The block supernodal Cholesky, with an f64 factor.
    Factor = 1,
    /// The block supernodal Cholesky, with an f32 factor: an f32 solve, or
    /// mixed_precision.
    FactorF32 = 2,
    /// Preconditioned conjugate gradients on the reduced system
    /// (SchurMethod::Iterative).
    ConjugateGradients = 3,
};

/// How much of a solve ran on its GPU (GpuReport::work).
enum class GpuWork : uint32_t {
    /// Nothing. The solve took a route that runs on the host (the
    /// envelope or narrow band factorization, faer's scalar Cholesky,
    /// SchurMethod::IterativeImplicit) and logged a warning naming it.
    Unused = 0,
    /// The damped solves. The host assembled (GpuReport::host_assembly
    /// says why) and uploaded the Hessian's values for each.
    Solves = 1,
    /// The assembly, the damped solves and the LM loop.
    Loop = 2,
};

/// Why a GPU solve assembled on the host (GpuReport::host_assembly).
enum class HostAssembly : uint32_t {
    /// It did not: the GPU assembled, or was not used.
    None = 0,
    /// SparseOptions::gpu_assembly was off.
    Asked = 1,
    /// The model has no device form (GPU assembly kernels);
    /// GpuReport::no_device_form says why.
    NoDeviceForm = 2,
    /// An f64 model on a GPU without f64 (Metal).
    NoF64 = 3,
};

/// A GPU of the machine, as a model's available_gpus() and detect_gpu()
/// give it.
struct GpuInfo {
    GpuBackend backend;
    /// The device, as the runtime numbers them (SparseOptions::gpu_index).
    uint32_t index;
    /// Whether it shares the host's memory: an integrated GPU.
    bool integrated;
    /// The device's name, as its driver reports it.
    char name[128];
};

/// What a solve did on the GPU it was given; read it with
/// LmResult::gpu(). The strings live as long as the result.
struct GpuReport {
    /// The runtime, as SparseOptions::gpu.
    GpuBackend backend;
    /// The device, as SparseOptions::gpu_index.
    uint32_t index;
    /// The device's name, as its driver reports it; empty when work is
    /// Unused.
    const char* name;
    /// What the kernels were compiled for (sm_89, gfx1036); empty when
    /// work is Unused.
    const char* target;
    /// Whether the GPU shares the host's memory: an integrated GPU, or the
    /// CPU device. False when work is Unused.
    bool integrated;
    /// The wide scalar: F64 or Df32.
    GpuWide wide;
    /// How the GPU solved the system.
    GpuSolve solve;
    /// Whether the system the GPU formed and solved is f32; otherwise it
    /// is in the wide scalar (docs/SOLVERS.md#precision).
    bool f32_system;
    /// How much of the solve ran on the GPU.
    GpuWork work;
    /// Why the host assembled, when work is Solves; None otherwise.
    HostAssembly host_assembly;
    /// Why the model has no device form, when host_assembly is
    /// NoDeviceForm; empty otherwise.
    const char* no_device_form;
    /// The GPU memory its state holds after the solve, in use or free, in
    /// bytes.
    uint64_t reserved_bytes;
    /// The GPU buffers its state has allocated over its life; unchanged
    /// across a solve that allocated nothing.
    uint64_t buffers_created;
};

/// One form's share of a sweep phase, summed over its calls (mirrors
/// arael's FormTiming). Seconds.
struct FormTiming {
    uint32_t calls;
    double region;
    double task_sum;
    double task_max;
    double task_min;
};

/// A sweep phase's clocks: the dispatched calls and the ones run on
/// the calling thread (mirrors arael's PhaseTiming).
struct PhaseTiming {
    FormTiming par;
    FormTiming seq;
};

/// Where the sweeps' time went (mirrors arael's ParTiming); all zero
/// unless the solve gathered timing, and `on` says which.
struct ParTiming {
    bool on;
    double bind;
    double assembly_update;
    double assembly_zero;
    PhaseTiming assembly;
    double gather_grad;
    double assembly_zero_vals;
    double scatter;
    double cost_update;
    PhaseTiming cost;
};

/// What one store holds (mirrors arael's StoreFootprint).
struct StoreFootprint {
    uint64_t self_blocks;
    uint64_t cross_blocks;
    uint64_t coo_entries;
    uint64_t bytes;
};

/// What one phase's sweeps did over a solve (mirrors arael's
/// PhaseChoice): whether it was dispatched over the stores, and how
/// many calls there were.
struct PhaseChoice {
    bool threaded;
    uint32_t calls;
};

/// What the sweeps did (mirrors arael's SweepReport). `held` is the
/// store count; LmResult::threads_held() reads the footprints.
struct SweepReport {
    uint32_t threads;
    PhaseChoice assembly;
    PhaseChoice cost;
    ParTiming timing;
    StoreFootprint whole;
    uint32_t held;
};

/// What a solve's threads did (mirrors arael's ThreadReport); read it
/// with LmResult::threads(). `sweeps` is absent when the model has no
/// threaded sweep path.
struct ThreadReport {
    uint32_t sweeps_asked;
    uint32_t linear;
    bool fell_back;
    option<SweepReport> sweeps;
};

/// The arael version a shim was built with (mirrors arael's Version):
/// the three numbers and `pre`, the tag after the dash ("dev" in
/// 0.9.0-dev, empty for a release); read it with arael_version().
struct AraelVersion {
    uint32_t major;
    uint32_t minor;
    uint32_t patch;
    const char* pre;
};

/// How much covariance to prepare (mirrors arael's CovMode).
enum class CovMode : uint32_t {
    PerQuery = 0,
    AllMarginals = 1,
    TriDiagonal = 2,
};

/// Elimination ordering for a covariance assembly (mirrors arael's
/// CovOrdering). It decides what the covariance costs, never what it is.
enum class CovOrdering : uint32_t {
    /// Price minimum degree against nested dissection over the model's block
    /// graph and keep the cheaper factor. The default.
    Auto = 0,
    /// Minimum degree, whatever structure the model has.
    Amd = 1,
    /// Nested dissection. For a trajectory that revisits -- a loop closure, a
    /// figure-8 crossing -- where poses far apart in the ordering are coupled.
    NestedDissection = 2,
    /// Natural order, no permutation.
    Natural = 3,
};

/// How to assemble a covariance (mirrors arael's CovOptions).
struct CovOptions {
    /// Elimination ordering.
    CovOrdering ordering = CovOrdering::Auto;
    /// Factorize in block form with arael's supernodal Cholesky instead of
    /// faer's scalar one. AllMarginals ignores this and stays on the scalar
    /// factor: its selected inverse reads faer's supernode panels.
    BlockSupernodalMode block_supernodal = BlockSupernodalMode::Auto;
};

/// Factor flops down each candidate ordering (mirrors arael's CovPlan).
struct CovCandidateFlops {
    double amd;
    double nd;
};

/// What a covariance assembly decided (mirrors arael's CovPlan).
struct CovPlan {
    /// The ordering that produced the factor. CovOrdering::Auto resolves to
    /// the candidate it kept, so this never reads Auto.
    CovOrdering ordering;
    /// What each candidate priced at, when Auto priced them.
    option<CovCandidateFlops> candidate_flops;
    /// Symbolic analyses built to reach the factor. Auto builds one per
    /// candidate and keeps one.
    uint32_t symbolics_built;
    /// Whether the block supernodal route ran.
    bool block_route;
};

/// A failed covariance operation; message points at last_error().
struct CovError {
    const char* message;
};

} // namespace arael
