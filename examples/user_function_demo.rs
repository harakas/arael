//! `#[arael::function]` demo.
//!
//! Two user-defined functions wired into a single Gauss-Newton fit:
//!
//! - `sigmoid(x: E) -> E` -- Form A, purely symbolic. The macro stashes
//!   `1 / (1 + exp(-x))` as an arael-sym source string; the fn body is
//!   rewritten to delegate to `arael_sym::parse_with_functions` so
//!   ordinary Rust callers still see `sigmoid(e) -> E`. Inside a
//!   constraint the interpreter inlines the tree directly.
//!
//! - `my_safe_asin(x: f64) -> f64` -- Form B, opaque numerical eval +
//!   explicit symbolic derivative. The eval clamps to [-1, 1] before
//!   calling the libm `asin`; the derivative string
//!   `1.0 / sqrt(1.0 - x * x + 1e-12)` is parsed by arael-sym at
//!   constraint-expansion and differentiated automatically through
//!   the chain rule.
//!
//! The toy problem: choose a scalar `x` that simultaneously satisfies
//! two residuals -- `sigmoid(x) = 0.8` and `my_safe_asin(x) = 0.5` --
//! weighted equally. LM converges to a single best-fit x.

use arael::info;
use arael::model::{Model, Param, SelfBlock};
use arael::simple_lm::{self, LmConfig, LmProblem};
use arael_sym::E;

// --- Form A: purely symbolic sigmoid. No derivs, auto-differentiated
// at constraint-expansion via the stashed arael-sym source. The body
// is captured verbatim as a string and handed to arael-sym's parser,
// so `exp` resolves to arael-sym's built-in rather than a Rust name.
#[arael::function]
fn sigmoid(x: E) -> E {
    1.0 / (1.0 + exp(-x))
}

// --- Form B: opaque clamped asin. The derivative of asin is
// 1/sqrt(1 - x^2), which blows up at |x| = 1. The `identity(...)`
// guard stops arael-sym's simplifier from reordering
// `1 - x*x + 1e-12` into `1 + 1e-12 - x*x`: near |x| = 1, the
// subtraction cancels most significant bits, and the reordered form
// would lose the 1e-12 floor to catastrophic cancellation. This is
// the same pattern the built-in `safe_asin` uses.
#[arael::function(my_safe_asin,
    derivs = [1.0 / sqrt(identity(1.0 - x * x) + 1e-12)])]
fn my_safe_asin_eval(x: f64) -> f64 {
    x.clamp(-1.0, 1.0).asin()
}

#[arael::model]
#[arael(root, jacobian)]
#[arael(constraint(hb, name = "sigmoid_eq_0p8", {
    // sigmoid(x) - 0.8 = 0  =>  x = logit(0.8) ~= 1.3863
    [(sigmoid(m.x) - 0.8) * m.isigma]
}))]
#[arael(constraint(hb, name = "asin_eq_0p5", {
    // my_safe_asin(x) - 0.5 = 0  =>  x = sin(0.5) ~= 0.4794
    [(my_safe_asin(m.x) - 0.5) * m.isigma]
}))]
struct M {
    x: Param<f64>,
    isigma: f64,
    hb: SelfBlock<M>,
}

fn main() {
    // Standalone evaluation of the symbolic sibling fns.
    let x_sym = arael_sym::symbol("x");
    info!("sigmoid(x) symbolic form: {}", sigmoid(x_sym.clone()));
    info!("my_safe_asin(x) symbolic form: {}", my_safe_asin(x_sym));
    info!("");

    let mut m = M {
        x: Param::new(0.0),
        isigma: 1.0,
        hb: SelfBlock::new(),
    };

    let mut params = Vec::new();
    m.serialize64(&mut params);
    info!("Start x = {:.6}, cost = {:.6}", params[0], m.calc_cost(&params));

    let config = LmConfig::<f64> {
        verbose: true,
        max_iters: 50,
        ..Default::default()
    };
    let result = simple_lm::solve(&params, &mut m, &config);
    m.deserialize64(&result.x);

    let r1 = sigmoid_f64(m.x.value) - 0.8;
    let r2 = my_safe_asin_eval(m.x.value) - 0.5;
    info!("");
    info!("LM: {} iterations, cost {:.6} -> {:.6}",
          result.iterations, result.start_cost, result.end_cost);
    info!("x                 = {:.6}", m.x.value);
    info!("sigmoid(x) - 0.8  = {:+.6}", r1);
    info!("my_safe_asin(x) - 0.5 = {:+.6}", r2);
    info!("(balanced fit: neither residual goes to zero alone)");
}

fn sigmoid_f64(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
