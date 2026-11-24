// The arael pipeline: build the model, probe it, solve it, read the solution.
//
// The model structs cannot be shared -- the arael macro bakes the scalar into
// the generated code, so an f32 model and an f64 model are different types, and
// each benchmark's problem is its own. The pipeline around them is the same for
// all of them, and keeping a copy per model is how one benchmark's f32 config
// lost the verbose flag its f64 twin had.

use ::arael::simple_lm::{LmConfig, LmProblem, LmResult, SolveFailure};
use ::arael::utils::Float;

use crate::table::Row;

pub trait Model: Clone + LmProblem<Self::Scalar> {
    type Scalar: Float;
    /// Whatever the benchmark builds its model from: a parsed dataset, a
    /// generated scene. It also carries whatever else distinguishes one row from
    /// another over the same model -- bundle adjustment runs the same camera/point
    /// model through two different linear solvers, and that choice lives here.
    type Input;
    type Solution;

    /// Problem-appropriate initial damping, before the ARAEL_LAMBDA0 override.
    /// Solver benchmarks give each algorithm the damping the problem wants, not
    /// the one it ships with; see the benchmark READMEs' damping policy. It sees
    /// the input because the right value can depend on the problem's size.
    fn lambda0(input: &Self::Input) -> f64;

    /// Whether this problem wants the gain-ratio damping driver by default. The
    /// pose graphs do not -- they are well-initialized and every step is accepted,
    /// so an adaptive schedule only over-damps. Bundle adjustment does: it is far
    /// less linear, and the fixed ladder walks into damping spirals there. DRIVER
    /// overrides either way.
    const NIELSEN: bool = false;

    fn build(input: &Self::Input) -> Self;
    fn serialize(&mut self, out: &mut Vec<Self::Scalar>);
    fn deserialize(&mut self, x: &[Self::Scalar]);
    fn solution(&self) -> Self::Solution;

    /// The scalar picks the solver (`SparseFaer` / `SparseFaer<f32>`, or a band
    /// solver), which is the one thing a generic function cannot choose for
    /// itself. It sees the input too, so a benchmark can run one model through
    /// several linear solvers and get a row for each.
    ///
    /// Hand the failure back rather than unwrapping it. A solve CAN fail on a
    /// real dataset -- arael's f32 rows hit a NaN Hessian diagonal on
    /// Ladybug-1723, from observations sitting on the optical centre -- and a
    /// benchmark that panics there loses every other row in the run.
    ///
    /// `ctx` is the row's solve context, one per model, kept across the
    /// probes so what a solve reuses (the root's block stores) carries
    /// over; pass it to `lm_solve_with_context`.
    fn solve(
        input: &Self::Input,
        params: &[Self::Scalar],
        model: &mut Self,
        cfg: &LmConfig<Self::Scalar>,
        ctx: &mut arael::threads::Context,
    ) -> Result<LmResult<Self::Scalar>, SolveFailure<Self::Scalar>>;

    /// Anything else this benchmark wants on the config. The defaults below are
    /// the ones every benchmark has agreed on so far.
    fn tune(_cfg: &mut LmConfig<Self::Scalar>) {}

    /// Whether this row's linear solve is inexact (conjugate gradients). It
    /// reads the input because the route is what distinguishes such a row from
    /// an exact one over the same model. See [`Row::inexact`].
    fn inexact(_input: &Self::Input) -> bool { false }

    /// The threaded sweeps' timing of the last solve, read from the row's
    /// context (the context's sweep `timing`); printed under TIMING=1.
    fn par_timing(_ctx: &arael::threads::Context) -> Option<String> { None }
}

/// The mixed precision setting of the arael rows, for
/// `SparseFaerOptions::with_mixed_precision`: `None` unless
/// ARAEL_F32_FACTOR is set.
///
/// ARAEL_F32_FACTOR=<n> is `MixedPrecision` with at most n refinement
/// steps (0 for none). ARAEL_REFINE_TOL=<t> is the relative residual the
/// refinement stops at, the library's default when unset. A value that
/// does not parse, and ARAEL_REFINE_TOL without ARAEL_F32_FACTOR, stop
/// the run.
pub fn mixed_precision() -> Option<arael::simple_lm::MixedPrecision> {
    mixed_precision_from(std::env::var("ARAEL_F32_FACTOR").ok().as_deref(), std::env::var("ARAEL_REFINE_TOL").ok().as_deref())
}

/// [`mixed_precision`] of the two variables' values.
fn mixed_precision_from(steps: Option<&str>, tol: Option<&str>) -> Option<arael::simple_lm::MixedPrecision> {
    let Some(steps) = steps else {
        if let Some(tol) = tol {
            panic!("ARAEL_REFINE_TOL={tol} without ARAEL_F32_FACTOR: there is no refinement for it to stop");
        }
        return None;
    };
    let max_refinements = steps.parse().unwrap_or_else(|_| panic!("ARAEL_F32_FACTOR={steps}: expected a refinement step count"));
    let mut mp = arael::simple_lm::MixedPrecision { max_refinements, ..Default::default() };
    if let Some(tol) = tol {
        mp.tol = tol.parse().unwrap_or_else(|_| panic!("ARAEL_REFINE_TOL={tol}: expected a relative residual"));
    }
    Some(mp)
}

/// The header's value for the single-precision factor ([`mixed_precision`]).
pub fn f32_factor_value() -> String {
    match mixed_precision() {
        None => "off [ARAEL_F32_FACTOR]".to_string(),
        Some(mp) => format!(
            "on, at most {} refinement step{} to a relative residual of {:e} [ARAEL_F32_FACTOR, ARAEL_REFINE_TOL]",
            mp.max_refinements,
            if mp.max_refinements == 1 { "" } else { "s" },
            mp.tol
        ),
    }
}

/// The header's value for the GPU the arael rows are given ([`gpu`]): the
/// device as its runtime lists it, its wide scalar and its staging, or
/// `none`. A device the build does not find on the machine says so: its
/// rows will fail.
pub fn gpu_value() -> String {
    gpu_value_of(gpu().as_ref())
}

/// [`gpu_value`] of `device`.
fn gpu_value_of(device: Option<&arael::simple_lm::GpuDevice>) -> String {
    use arael::simple_lm::{GpuBackend, GpuDevice, GpuStaging, GpuWide};
    let Some(d) = device else {
        return "none [ARAEL_GPU: cuda|hip|metal|cpu, with :index]".to_string();
    };
    let device = match d.backend() {
        GpuBackend::Cpu => "cpu:0, the CPU running the GPU's kernels".to_string(),
        backend => match GpuDevice::available().into_iter().find(|g| g.backend == backend && g.index == d.index()) {
            Some(g) => g.to_string(),
            None => format!(
                "{}:{} NOT FOUND on this machine by this build (its gpu-* feature, its driver): the arael rows will fail",
                format!("{backend:?}").to_lowercase(),
                d.index()
            ),
        },
    };
    let wide = match d.wide() {
        GpuWide::F64 => "f64",
        GpuWide::Df32 => "df32",
    };
    let staging = match d.staging() {
        GpuStaging::Pinned => "pinned",
        GpuStaging::Pageable => "pageable",
    };
    format!("{device}, wide {wide}, staging {staging} [ARAEL_GPU, ARAEL_GPU_WIDE, ARAEL_GPU_STAGING]")
}

/// The GPU the arael rows are given, for `SparseFaerOptions::with_gpu`:
/// `None` unless ARAEL_GPU is set. Every call returns a clone of one
/// device, so the rows of a run share its memory.
///
/// - ARAEL_GPU: `cuda`, `hip` or `metal`, with an optional device index
///   (`cuda:1`), or `cpu`, the CPU device. The binary needs the matching
///   feature (`gpu-cuda`, `gpu-hip`, `gpu-metal`, `gpu-cpu`).
/// - ARAEL_GPU_WIDE: `f64` or `df32` (`GpuDevice::with_wide`); unset is
///   the device's default.
/// - ARAEL_GPU_STAGING: `pinned` (the default) or `pageable`
///   (`GpuDevice::with_staging`).
///
/// Any other value stops the run. What a row did on the GPU is printed
/// under it ([`Row::gpu`]).
pub fn gpu() -> Option<arael::simple_lm::GpuDevice> {
    static DEVICE: std::sync::OnceLock<Option<arael::simple_lm::GpuDevice>> = std::sync::OnceLock::new();
    DEVICE
        .get_or_init(|| {
            let v = std::env::var("ARAEL_GPU").ok()?;
            let (name, index) = v.split_once(':').unwrap_or((v.as_str(), "0"));
            let expected = "expected cuda[:index], hip[:index], metal[:index] or cpu";
            let index: usize = index.parse().unwrap_or_else(|_| panic!("ARAEL_GPU={}: {}", v, expected));
            let device = match name {
                "cuda" => arael::simple_lm::GpuDevice::cuda(index),
                "hip" => arael::simple_lm::GpuDevice::hip(index),
                "metal" => arael::simple_lm::GpuDevice::metal(index),
                "cpu" => arael::simple_lm::GpuDevice::cpu(),
                _ => panic!("ARAEL_GPU={}: {}", v, expected),
            };
            let wide = match std::env::var("ARAEL_GPU_WIDE").as_deref() {
                Err(_) => device.wide(),
                Ok("f64") => arael::simple_lm::GpuWide::F64,
                Ok("df32") => arael::simple_lm::GpuWide::Df32,
                Ok(other) => panic!("ARAEL_GPU_WIDE={other}: expected f64 or df32"),
            };
            let staging = match std::env::var("ARAEL_GPU_STAGING").as_deref() {
                Err(_) | Ok("pinned") => arael::simple_lm::GpuStaging::Pinned,
                Ok("pageable") => arael::simple_lm::GpuStaging::Pageable,
                Ok(other) => panic!("ARAEL_GPU_STAGING={other}: expected pinned or pageable"),
            };
            Some(device.with_wide(wide).with_staging(staging))
        })
        .clone()
}

/// The first lambda of the arael rows: ARAEL_LAMBDA0, or the model's own
/// ([`Model::lambda0`]) when unset.
pub fn lambda0<M: Model>(input: &M::Input) -> f64 {
    lambda0_override().unwrap_or_else(|| M::lambda0(input))
}

/// ARAEL_LAMBDA0, when set.
pub fn lambda0_override() -> Option<f64> {
    crate::env::parsed("ARAEL_LAMBDA0", "an initial lambda")
}

/// Whether the arael rows use the gain-ratio (Nielsen) lambda driver
/// instead of arael's default one: DRIVER=nielsen or DRIVER=fixed
/// (`default` means `fixed`), or the model's own choice
/// ([`Model::NIELSEN`]) when unset. Any other value stops the run.
pub fn nielsen<M: Model>() -> bool {
    driver_from(std::env::var("DRIVER").ok().as_deref()).unwrap_or(M::NIELSEN)
}

/// DRIVER's value as "use the Nielsen driver", `None` when unset.
fn driver_from(value: Option<&str>) -> Option<bool> {
    match value? {
        "nielsen" => Some(true),
        "fixed" | "default" => Some(false),
        other => panic!("DRIVER={other}: expected nielsen or fixed"),
    }
}

/// ARAEL_BLOCK_SUPERNODAL, for `SparseFaerOptions::with_block_supernodal`:
/// `auto` or unset (arael's default), `1` or `always`, `0`, `off` or
/// `scalar` (never). Any other value stops the run.
pub fn block_supernodal() -> arael::simple_lm::BlockSupernodalMode {
    block_supernodal_from(std::env::var("ARAEL_BLOCK_SUPERNODAL").ok().as_deref())
}

fn block_supernodal_from(value: Option<&str>) -> arael::simple_lm::BlockSupernodalMode {
    use arael::simple_lm::BlockSupernodalMode as M;
    match value {
        None | Some("auto") => M::Auto,
        Some("1") | Some("always") => M::Always,
        Some("0") | Some("off") | Some("scalar") => M::Never,
        Some(other) => panic!("ARAEL_BLOCK_SUPERNODAL={other}: expected auto, 1/always, or 0/off/scalar"),
    }
}

/// ARAEL_BLOCK_SUPERNODAL_BATCH, for
/// `SparseFaerOptions::with_block_supernodal_batching`: a ratio (1.5),
/// `0` or `off` for no batching, unset for arael's default. Any other
/// value stops the run.
pub fn block_supernodal_batch() -> Option<f64> {
    block_supernodal_batch_from(std::env::var("ARAEL_BLOCK_SUPERNODAL_BATCH").ok().as_deref())
}

fn block_supernodal_batch_from(value: Option<&str>) -> Option<f64> {
    match value {
        None => arael::simple_lm::SparseFaerOptions::auto().block_supernodal_batch,
        Some("0") | Some("off") => None,
        Some(v) => Some(v.parse().unwrap_or_else(|_| panic!("ARAEL_BLOCK_SUPERNODAL_BATCH={v}: expected a ratio, or 0/off"))),
    }
}

/// ARAEL_BLOCK_SUPERNODAL_LEAN=1, for
/// `SparseFaerOptions::with_block_supernodal_memory_lean`.
pub fn block_supernodal_lean() -> bool {
    crate::env::flag("ARAEL_BLOCK_SUPERNODAL_LEAN")
}

/// ARAEL_ORDERING, for `SparseFaerOptions::with_ordering`: `auto`, `amd`,
/// `nd` (nested dissection), `natural` or `marginalize-first`. `None`
/// when unset: the benchmark's own choice. Any other value stops the run.
///
/// ```ignore
/// let ordering = bench_harness::arael::ordering().unwrap_or(SolveOrdering::NestedDissection);
/// ```
pub fn ordering() -> Option<arael::simple_lm::SolveOrdering> {
    ordering_from(std::env::var("ARAEL_ORDERING").ok().as_deref())
}

fn ordering_from(value: Option<&str>) -> Option<arael::simple_lm::SolveOrdering> {
    use arael::simple_lm::SolveOrdering as O;
    Some(match value? {
        "auto" => O::Auto,
        "amd" => O::Amd,
        "nd" => O::NestedDissection,
        "natural" => O::Natural,
        "marginalize-first" => O::MarginalizeFirst,
        other => panic!("ARAEL_ORDERING={other}: expected auto, amd, nd, natural or marginalize-first"),
    })
}

/// The name of `ordering` as ARAEL_ORDERING spells it, for a header.
pub fn ordering_name(ordering: arael::simple_lm::SolveOrdering) -> &'static str {
    use arael::simple_lm::SolveOrdering as O;
    match ordering {
        O::Auto => "auto (arael's rule)",
        O::Amd => "amd",
        O::NestedDissection => "nd (nested dissection)",
        O::Natural => "natural",
        O::MarginalizeFirst => "marginalize-first",
    }
}

/// ARAEL_SCHUR, for `SparseFaerOptions::with_policy`: `auto`, `force` or
/// `never`. `None` when unset: the benchmark's own choice. Any other
/// value stops the run.
pub fn schur_policy() -> Option<arael::simple_lm::SchurPolicy> {
    schur_policy_from(std::env::var("ARAEL_SCHUR").ok().as_deref())
}

fn schur_policy_from(value: Option<&str>) -> Option<arael::simple_lm::SchurPolicy> {
    use arael::simple_lm::SchurPolicy;
    Some(match value? {
        "auto" => SchurPolicy::default(),
        "force" => SchurPolicy::Force,
        "never" => SchurPolicy::Never,
        other => panic!("ARAEL_SCHUR={other}: expected auto, force or never"),
    })
}

/// The name of `policy` as ARAEL_SCHUR spells it, for a header.
pub fn schur_policy_name(policy: &arael::simple_lm::SchurPolicy) -> &'static str {
    use arael::simple_lm::SchurPolicy;
    match policy {
        SchurPolicy::Force => "force",
        SchurPolicy::Never => "never",
        _ => "auto",
    }
}

/// ARAEL_ENVELOPE, for `SparseFaerOptions::with_envelope_schur`: `auto`,
/// `always` or `never`; unset is `auto`. Any other value stops the run.
pub fn envelope() -> arael::simple_lm::EnvelopeMode {
    envelope_from(std::env::var("ARAEL_ENVELOPE").ok().as_deref())
}

fn envelope_from(value: Option<&str>) -> arael::simple_lm::EnvelopeMode {
    use arael::simple_lm::EnvelopeMode;
    match value {
        None | Some("auto") => EnvelopeMode::Auto,
        Some("always") => EnvelopeMode::Always,
        Some("never") => EnvelopeMode::Never,
        Some(other) => panic!("ARAEL_ENVELOPE={other}: expected auto, always or never"),
    }
}

/// The conjugate gradient settings of the rows that solve with them, for
/// `SparseFaerOptions::with_iterative_schur`: ARAEL_CG_TOL (the benchmark's
/// `default_tol` when unset), ARAEL_CG_MAXITER and ARAEL_CG_RESTART
/// (arael's defaults when unset).
pub fn cg(default_tol: f64) -> arael::simple_lm::CgParams {
    let d = arael::simple_lm::CgParams::default();
    arael::simple_lm::CgParams {
        tol: crate::env::parsed("ARAEL_CG_TOL", "a relative tolerance").unwrap_or(default_tol),
        max_iters: crate::env::parsed("ARAEL_CG_MAXITER", "an iteration count").unwrap_or(d.max_iters),
        restart_every: crate::env::parsed("ARAEL_CG_RESTART", "an iteration count").unwrap_or(d.restart_every),
    }
}

/// ARAEL_COV_ORDERING, for `CovOptions::with_ordering`: `auto`, `amd`,
/// `nd` or `natural`. `None` when unset: the benchmark's own choice. Any
/// other value stops the run.
pub fn cov_ordering() -> Option<arael::covariance::CovOrdering> {
    cov_ordering_from(std::env::var("ARAEL_COV_ORDERING").ok().as_deref())
}

fn cov_ordering_from(value: Option<&str>) -> Option<arael::covariance::CovOrdering> {
    use arael::covariance::CovOrdering as O;
    Some(match value? {
        "auto" => O::Auto,
        "amd" => O::Amd,
        "nd" => O::NestedDissection,
        "natural" => O::Natural,
        other => panic!("ARAEL_COV_ORDERING={other}: expected auto, amd, nd or natural"),
    })
}

/// The `LmConfig` of an arael row: the termination every system of the
/// benchmarks is held to, the first lambda and the driver ([`lambda0`],
/// [`nielsen`]), the threads ([`crate::pin::threads`]), VERBOSE, and the
/// model's own settings ([`Model::tune`]).
pub fn config<M: Model>(input: &M::Input, max_iters: usize) -> LmConfig<M::Scalar> {
    // Float: num::NumCast, so a literal reaches either scalar the same way.
    let f = |v: f64| <M::Scalar as num::NumCast>::from(v).unwrap();
    let mut cfg = LmConfig {
        verbose: crate::env::verbose(),
        gather_timing: true,
        // The same count every other system's pool was capped at (BENCH_THREADS,
        // resolved by pin::enforce_cores), so a threaded run stays a fair race.
        // Needs arael built with the `threads` feature, or it warns and runs
        // sequentially.
        num_threads: crate::pin::threads(),
        // The termination class every system in these benchmarks is held to.
        abs_precision: f(1e-5),
        rel_precision: f(1e-5),
        // Stop as soon as the tolerance test says converged, the way Ceres and
        // g2o do. The library defaults (min_iters 5, patience 3) put a floor
        // under the iteration count, which on a problem that converges in four
        // iterations would time a fifth that improves the cost by 4e-14.
        patience: 1,
        min_iters: 1,
        max_iters,
        initial_lambda: f(lambda0::<M>(input)),
        ..Default::default()
    };
    M::tune(&mut cfg);
    if nielsen::<M>() {
        cfg.with_driver(::arael::simple_lm::NielsenLambdaDriver::default())
    } else {
        cfg
    }
}

/// Prints the phase times of one arael solve to stderr, under TIMING=1
/// ([`crate::env::timing`]): a `[timing]` line (total, assembly, analysis,
/// linear solve, cost evaluation), a `[setup]` line (the parts of the
/// first solve's setup) and, on a GPU, `[gpu]` lines (the time the device
/// spent on assembly, attempts and conjugate gradients).
///
/// Every solve prints, the warmup and the probes included, so a row
/// prints several sets. The solver gathers the times whether or not they
/// are printed.
pub fn print_timing<T>(r: &LmResult<T>) {
    if !crate::env::timing() {
        return;
    }
    if let Some(t) = &r.timing {
        eprintln!(
            "  [timing] total {:.1} ms = assembly {:.1} + analysis {:.1} + linear solve {:.1} \
             + cost eval {:.1} (first assembly {:.1}), {} iters, {} cost evals",
            t.total.as_secs_f64() * 1e3,
            t.assembly.as_secs_f64() * 1e3,
            t.analysis.as_secs_f64() * 1e3,
            t.linear_solve.as_secs_f64() * 1e3,
            t.cost_eval.as_secs_f64() * 1e3,
            t.first_assembly.as_secs_f64() * 1e3,
            r.iterations,
            t.cost_eval_count,
        );
        let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
        if let Some(setup) = &t.setup {
            let parts: Vec<String> = setup.parts().iter().filter(|p| ms(p.1) >= 0.005).map(|(name, d)| format!("{name} {:.2}", ms(*d))).collect();
            eprintln!("  [setup] {} ms", parts.join(" + "));
        }
        if let Some(g) = &t.gpu {
            eprintln!(
                "  [gpu] on the device: assembly {:.2} ms in {} calls ({:.3} each), attempts {:.2} ms in {} calls ({:.3} each)",
                ms(g.assembly),
                g.assembly_count,
                ms(g.assembly) / g.assembly_count.max(1) as f64,
                ms(g.attempts),
                g.attempt_count,
                ms(g.attempts) / g.attempt_count.max(1) as f64,
            );
            if g.cg_rounds > 0 {
                eprintln!(
                    "  [gpu] of the attempts, conjugate gradients {:.2} ms in {} rounds ({:.3} an attempt)",
                    ms(g.cg),
                    g.cg_rounds,
                    ms(g.cg) / g.attempt_count.max(1) as f64,
                );
            }
        }
    }
}

/// The probes run on throwaway copies of the model: re-supplying the initial
/// parameter vector does not reset parametrization state held outside it, such
/// as the reference rotation a QuaternionParam re-centres on after a step. A
/// probe that ran on the real model left it half-advanced, and the solve that
/// followed inherited a warm start it never asked for.
///
/// `Err` carries why the solve failed, for the caller to put in the table.
/// Every probe runs the same solve, so the first failure ends the row: there is
/// no partial row worth reporting, and re-running it would only repeat itself.
pub fn run<M: Model>(input: &M::Input) -> Result<Row<M::Solution>, String> {
    let mut model = M::build(input);
    let mut params: Vec<M::Scalar> = Vec::new();
    model.serialize(&mut params);
    // One context for the row: the model is cloned per probe, the
    // context is not, so the block stores carry over. Its phase
    // timing runs only under TIMING, where it is printed.
    let mut ctx = arael::threads::Context::new();
    let timing = crate::env::timing();
    ctx.set_timing(timing);

    // REPORT keeps the report of the full solve, for the table to print
    // under the row. The probes are capped solves and are not reported.
    const FULL_ITERS: usize = 100;
    let full_iters = crate::solver::full_iters(FULL_ITERS);
    let report = crate::env::report();
    let mut full_report: Option<String> = None;
    // What the full solve did on the GPU it was given, for the table to
    // print under the row whether or not the solver is verbose.
    let mut gpu_note: Option<String> = None;
    let mut failure: Option<String> = None;
    let row = crate::solver::run(FULL_ITERS, |max_iters| {
        // The clone and the config are the probe's reset, not the solve: the
        // clock starts below.
        let mut m = model.clone();
        if failure.is_some() {
            // Already broken. The driver still wants its remaining probes;
            // re-running a solve that fails deterministically only costs time.
            return crate::solver::Outcome {
                ms: f64::INFINITY,
                accepted: 0,
                attempts: 1,
                solution: m.solution(),
            };
        }
        let cfg = config::<M>(input, max_iters);
        // Qualified: `M::solve` alone is ambiguous against `LmProblem::solve`
        // (the SolverKind entry point), which Model also carries.
        let (ms, r) = crate::solver::timed(|| <M as Model>::solve(input, &params, &mut m, &cfg, &mut ctx));
        let r = match r {
            Ok(r) => r,
            Err(e) => {
                failure.get_or_insert_with(|| describe(&e));
                // The driver needs an Outcome to carry on with. Nothing reads
                // it: the caller drops the row the moment `failure` is set.
                return crate::solver::Outcome {
                    ms: f64::INFINITY,
                    accepted: 0,
                    attempts: 1,
                    solution: m.solution(),
                };
            }
        };
        print_timing(&r);
        if timing {
            if let Some(par) = M::par_timing(&ctx) {
                println!("  [par] {}", par);
            }
        }
        if report && max_iters == full_iters {
            full_report = Some(r.pretty_report());
        }
        if max_iters == full_iters {
            gpu_note = r.gpu.as_ref().map(|g| g.to_string());
        }
        m.deserialize(&r.x);
        crate::solver::Outcome {
            ms,
            accepted: r.accepted_iterations,
            attempts: r.iterations,
            solution: m.solution(),
        }
    });
    match failure {
        Some(why) => Err(why),
        None => Ok(row.inexact(M::inexact(input)).report(full_report).gpu(gpu_note)),
    }
}

/// One line naming the fault, for a table cell: `SolveFailure`'s Display, which
/// is the short form. Its Debug carries the partial result.
pub fn describe<T>(e: &SolveFailure<T>) -> String {
    e.to_string()
}

/// `problem`'s Hessian at its serialized parameters, in the tiled block
/// layout the solver assembles into, the gradient, and the block partition
/// of the parameters. The diagonal is damped by `1 + 1e-8`, as a solve's
/// first attempt about does.
///
/// ```ignore
/// let (h, grad, partition) = bench_harness::arael::hessian::<u32, _>(&mut model);
/// ```
pub fn hessian<I, P>(problem: &mut P) -> (arael_faer::bsc::SparseBlockColMat<I, f64>, Vec<f64>, Vec<usize>)
where
    I: arael_faer::faer::Index,
    P: ::arael::simple_lm::RootProblem<f64> + ::arael::simple_lm::LmProblemInternals<f64>,
{
    hessian_in::<I, f64, P>(problem)
}

/// [`hessian`] in the model's scalar `T`: an f32 model's Hessian as its
/// solve assembles it.
///
/// ```ignore
/// let (h, grad, partition) = bench_harness::arael::hessian_in::<u32, f32, _>(&mut model);
/// ```
pub fn hessian_in<I, T, P>(problem: &mut P) -> (arael_faer::bsc::SparseBlockColMat<I, T>, Vec<T>, Vec<usize>)
where
    I: arael_faer::faer::Index,
    T: ::arael::utils::Float + arael_faer::schur::SchurReal,
    P: ::arael::simple_lm::RootProblem<T> + ::arael::simple_lm::LmProblemInternals<T>,
{
    use arael_faer::bsc::{PositionResolver, SparseBlockColMat, SymbolicSparseBlockColMat};
    let mut ctx = ::arael::threads::Context::new();
    let mut params: Vec<T> = Vec::new();
    problem.serialize(&mut params);
    let n = params.len();
    let mut spans = Vec::new();
    problem.collect_param_block_spans(&mut spans, &mut ctx);
    let partition = ::arael::store::block_partition_from_spans(&spans, n);
    let mut cells: Vec<(u32, u32)> = Vec::new();
    problem.collect_hessian_cells(&mut cells, &mut ctx);
    let part: Vec<I> = partition.iter().map(|&p| I::truncate(p)).collect();
    let (hsym, _) = SymbolicSparseBlockColMat::<I>::from_scalar_coords(part.clone(), part, cells.len(), |k| (cells[k].0 as usize, cells[k].1 as usize));
    let mut resolver = PositionResolver::new(&hsym);
    let mut positions = ::arael::store::PositionStream::new();
    problem.bind_hessian_positions(
        &mut ::arael::store::HessianBinder::Tiled(&mut |i, j| resolver.resolve_tile(i as usize, j as usize)),
        &mut positions,
        &mut ctx,
    );
    let mut h = SparseBlockColMat::<I, T>::zeroed(hsym);
    let mut grad = vec![T::zero(); n];
    problem.calc_grad_hessian_sparse_indexed(&params, &mut grad, h.vals_mut(), &positions, &mut ctx);
    // the structure once: cloned per block, it made this quadratic
    let hs = h.symbolic().clone();
    let damp = T::one() + T::from(1e-8).unwrap();
    for b in 0..partition.len() - 1 {
        let w = partition[b + 1] - partition[b];
        let d = hs.col_range(b).find(|&x| hs.blk_row(x) == b).expect("every block has its diagonal tile");
        let base = hs.val_range(d).start;
        for k in 0..w {
            h.vals_mut()[base + k * (w + 1)] *= damp;
        }
    }
    (h, grad, partition)
}

#[cfg(test)]
mod tests {
    use super::{
        block_supernodal_batch_from, block_supernodal_from, cov_ordering_from, driver_from, envelope_from, gpu_value_of,
        mixed_precision_from, ordering_from, schur_policy_from,
    };
    use arael::covariance::CovOrdering;
    use arael::simple_lm::{BlockSupernodalMode, EnvelopeMode, GpuDevice, GpuWide, SchurPolicy, SolveOrdering};

    /// ARAEL_F32_FACTOR turns mixed precision on with its step count, and
    /// ARAEL_REFINE_TOL sets the tolerance; the tolerance alone is an
    /// error, as is a value that does not parse.
    #[test]
    fn the_refinement_tolerance_needs_the_f32_factor() {
        assert!(mixed_precision_from(None, None).is_none());
        let mp = mixed_precision_from(Some("3"), None).unwrap();
        assert_eq!((mp.max_refinements, mp.tol), (3, arael::simple_lm::MixedPrecision::default().tol));
        let mp = mixed_precision_from(Some("0"), Some("1e-6")).unwrap();
        assert_eq!((mp.max_refinements, mp.tol), (0, 1e-6));
        assert!(std::panic::catch_unwind(|| mixed_precision_from(None, Some("1e-6"))).is_err(), "the tolerance alone was accepted");
        assert!(std::panic::catch_unwind(|| mixed_precision_from(Some("x"), None)).is_err());
        assert!(std::panic::catch_unwind(|| mixed_precision_from(Some("2"), Some("x"))).is_err());
    }

    /// ARAEL_ORDERING and ARAEL_SCHUR: unset leaves the benchmark its own
    /// choice, a known value names an option, anything else is an error.
    #[test]
    fn the_solver_options_parse_or_stop() {
        assert!(ordering_from(None).is_none() && schur_policy_from(None).is_none());
        assert!(matches!(ordering_from(Some("nd")), Some(SolveOrdering::NestedDissection)));
        assert!(matches!(ordering_from(Some("marginalize-first")), Some(SolveOrdering::MarginalizeFirst)));
        assert!(matches!(schur_policy_from(Some("force")), Some(SchurPolicy::Force)));
        assert!(matches!(schur_policy_from(Some("auto")), Some(SchurPolicy::Auto { .. })));
        assert!(std::panic::catch_unwind(|| ordering_from(Some("ND"))).is_err());
        assert!(std::panic::catch_unwind(|| schur_policy_from(Some("1"))).is_err());
    }

    /// ARAEL_ENVELOPE, ARAEL_COV_ORDERING and DRIVER: unset is the default
    /// (or the benchmark's choice), a known value names an option, anything
    /// else is an error.
    #[test]
    fn the_envelope_cov_ordering_and_driver_parse_or_stop() {
        assert!(matches!(envelope_from(None), EnvelopeMode::Auto));
        assert!(matches!(envelope_from(Some("always")), EnvelopeMode::Always));
        assert!(std::panic::catch_unwind(|| envelope_from(Some("on"))).is_err());
        assert!(cov_ordering_from(None).is_none());
        assert!(matches!(cov_ordering_from(Some("amd")), Some(CovOrdering::Amd)));
        assert!(std::panic::catch_unwind(|| cov_ordering_from(Some("marginalize-first"))).is_err());
        assert_eq!((driver_from(None), driver_from(Some("nielsen")), driver_from(Some("fixed")), driver_from(Some("default"))), (None, Some(true), Some(false), Some(false)));
        assert!(std::panic::catch_unwind(|| driver_from(Some("gain"))).is_err());
    }

    /// The block supernodal options: every documented spelling, the
    /// library's batching default when unset, and an error otherwise.
    #[test]
    fn the_block_supernodal_options_parse_or_stop() {
        assert!(matches!(block_supernodal_from(None), BlockSupernodalMode::Auto));
        assert!(matches!(block_supernodal_from(Some("1")), BlockSupernodalMode::Always));
        assert!(matches!(block_supernodal_from(Some("scalar")), BlockSupernodalMode::Never));
        assert!(std::panic::catch_unwind(|| block_supernodal_from(Some("yes"))).is_err());
        assert_eq!(block_supernodal_batch_from(None), arael::simple_lm::SparseFaerOptions::auto().block_supernodal_batch);
        assert_eq!((block_supernodal_batch_from(Some("off")), block_supernodal_batch_from(Some("1.5"))), (None, Some(1.5)));
        assert!(std::panic::catch_unwind(|| block_supernodal_batch_from(Some("x"))).is_err());
    }

    /// The header's GPU line: none without a device, the CPU device by its
    /// role, a device's wide scalar and staging, and a device the machine
    /// does not have called out.
    #[test]
    fn the_headers_gpu_line_says_what_the_rows_are_given() {
        assert!(gpu_value_of(None).starts_with("none [ARAEL_GPU"));
        let cpu = gpu_value_of(Some(&GpuDevice::cpu().with_wide(GpuWide::Df32)));
        assert_eq!(cpu, "cpu:0, the CPU running the GPU's kernels, wide df32, staging pinned [ARAEL_GPU, ARAEL_GPU_WIDE, ARAEL_GPU_STAGING]");
        // no machine has an eighth HIP device
        let missing = gpu_value_of(Some(&GpuDevice::hip(7)));
        assert!(missing.starts_with("hip:7 NOT FOUND") && missing.contains("the arael rows will fail"), "{missing}");
    }
}
