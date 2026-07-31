// arael C++ solver surface, shared by every generated interface:
// status and preset enums, the configuration and result layouts
// (templated on the solve precision), and the error types. A generated
// header instantiates these (`using LmResult = LmResultT<float>;`) and
// wires LmConfig construction to its root's FFI, which copies the
// preset's Rust values into the struct.
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

/// The solver configuration, holding the preset's Rust values.
/// Inspect them, edit them, pass the struct back whole; `option`
/// fields mirror the Rust `Option` fields (assign a value or `{}`).
/// `observer` (with `observer_user` passed back as its first
/// argument) is called once per damped attempt; return false to stop
/// the solve (status ObserverTerminated). Layout is part of the C
/// ABI -- field order matters.
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
/// FaerOrdering).
enum class FaerOrdering : uint32_t {
    Auto = 0,
    Amd = 1,
    MarginalizeFirst = 2,
    Natural = 3,
    NestedDissection = 4,
};

/// How the reduced Schur system is factored (mirrors arael's
/// EnvelopeMode): in block form under its envelope, or by the general
/// sparse Cholesky. The envelope route uses less memory on suitable
/// systems; Auto prices it per problem.
enum class EnvelopeMode : uint32_t {
    Auto = 0,
    Always = 1,
    Never = 2,
};

/// How the reduced Schur system is solved (mirrors arael's
/// SchurSolve): factorized, or by preconditioned conjugate gradients
/// (Iterative forms the reduced matrix, IterativeImplicit never
/// does). Pair the iterative routes with SchurPolicy::Force --
/// without a reduction the solve fails rather than taking another
/// route.
enum class SchurSolve : uint32_t {
    Factorize = 0,
    Iterative = 1,
    IterativeImplicit = 2,
};

/// The sparse backend's options (mirrors arael's SparseFaerOptions).
/// The generated SparseOptions fills the Rust defaults at
/// construction; edit fields and pass to solve_sparse. Layout is part
/// of the C ABI -- field order matters. Not exposed: the marginalize
/// range list (the model's own marginalize hint covers it) and the
/// iterative Schur routes.
struct SparseOptionsT {
    SchurPolicy schur;
    FaerOrdering ordering;
    EnvelopeMode envelope;
    /// Envelope panel width; 0 picks it automatically.
    uint32_t envelope_panel_width;
    /// Factorize supernodally (dense panels, BLAS3).
    bool supernodal;
    /// Factor a banded whole system in block form under its band.
    bool narrow_band;
    /// SchurPolicy::Auto tuning: the reduction must beat the whole
    /// system by this flop factor to be taken...
    double flop_margin;
    /// ...and below this cheap ratio it is taken without the exact
    /// pricing.
    double obvious_flop_ratio;
    /// Conjugate-gradient tolerance for the iterative routes.
    double cg_tol;
    SchurSolve schur_solve;
    /// CG iteration cap; 0 = unlimited.
    uint32_t cg_max_iters;
    /// CG restart interval; 0 = never.
    uint32_t cg_restart_every;
};

/// Per-phase wall-clock seconds plus call counts, gathered when
/// LmConfigT::gather_timing is set (see Rust's LmTiming for the
/// phase definitions; the per-step records stay Rust-side).
struct LmTiming {
    double total;
    double assembly;
    double first_assembly;
    double analysis;
    double linear_solve;
    double first_linear_solve;
    double cost_eval;
    double first_cost_eval;
    double advance;
    double first_advance;
    uint32_t assembly_count;
    uint32_t analysis_count;
    uint32_t linear_solve_count;
    uint32_t cost_eval_count;
    uint32_t advance_count;
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
};

/// Per-iteration flops down the two routes the Auto policy priced:
/// the reduction plus the reduced factor, against the whole system's
/// factor.
struct RouteFlops {
    double reduced;
    double full;
};

/// What the sparse backend decided (mirrors arael's SchurPlan); read
/// it with LmResult::plan(). Layout is part of the C ABI -- field
/// order matters.
struct SchurPlan {
    /// Whether the Schur reduction was used; false means the full
    /// system was factorized instead.
    bool reduced;
    /// Parameter blocks marginalized.
    uint32_t eliminated_blocks;
    /// Parameters marginalized, and parameters left in the reduced
    /// system.
    uint32_t eliminated_params;
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
};

/// How much covariance to prepare (mirrors arael's CovMode).
enum class CovMode : uint32_t {
    PerQuery = 0,
    AllMarginals = 1,
    TriDiagonal = 2,
};

/// A failed covariance operation; message points at last_error().
struct CovError {
    const char* message;
};

} // namespace arael
