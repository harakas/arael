// arael C++ solver surface, shared by every generated interface:
// status and preset enums, the configuration and result layouts
// (templated on the solve precision), and the error types. A generated
// header instantiates these (`using LmResult = LmResultT<float>;`) and
// wires LmConfig construction to its root's FFI, which copies the
// preset's Rust values into the struct.
#pragma once

#include <cstdint>
#include "result.hpp"

namespace arael {

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

/// The base preset a config starts from; it also supplies the Rust
/// fields the struct does not expose (lambda driver, observer,
/// gather_timing).
enum class LmPreset : uint32_t {
    Defaults = 0,
    Conservative = 1,
    WellConditioned = 2,
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
};

/// The Err side of a solve: SolverFailed or Panicked, with the text
/// from last_error() (valid until the next call on the model).
struct SolveError {
    LmStatus status;
    const char* message;
};

template<class F>
using SolveResultT = result<LmResultT<F>, SolveError>;

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
