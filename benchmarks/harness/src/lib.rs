// The benchmark harness: what every arael benchmark does the same way.
//
// pgo, slam, loc and bal all measure the same thing -- what one iteration of a
// nonlinear solver costs on an identical problem -- and each of them had grown
// its own copy of the machinery. The copies drifted, and every timing bug found
// in pgo (a probe mutating the model, a first iteration full of rejected steps,
// retries counted for one system and not another, a metric that amortized setup
// into the per-iteration number) was a bug the others still have.
//
// A benchmark supplies its problem: how to build it, how to score a solution,
// which systems to run. Everything else lives here.

pub mod arael;
pub mod cov;
pub mod env;
pub mod external;
pub mod header;
#[cfg(feature = "factrs")]
pub mod factrs;
#[cfg(any(feature = "gpu-cuda", feature = "gpu-hip", feature = "gpu-metal"))]
pub mod gpu_factor;
/// The GPU factor comparison of a build without a GPU runtime: it stops,
/// so `gpu_factor_bench` builds with every feature set.
#[cfg(not(any(feature = "gpu-cuda", feature = "gpu-hip", feature = "gpu-metal")))]
pub mod gpu_factor {
    /// Stops the run: the comparison needs the feature `gpu-cuda`,
    /// `gpu-hip` or `gpu-metal`.
    pub fn compare(what: &str, _a: &arael_faer::bsc::SparseBlockColMat<arael_faer::SparseIndex, f64>, _rhs: &[f64]) {
        eprintln!("{what}: the GPU factor comparison needs the feature gpu-cuda, gpu-hip or gpu-metal");
        std::process::exit(2);
    }
}
pub mod mem;
pub mod pin;
pub mod probe;
pub mod solver;
pub mod table;
