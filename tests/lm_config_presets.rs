// Behavioral validation of the LmConfig presets. Value assertions would only
// prove the fields are set; these check the presets actually solve, and that
// each fits the regime it is named for. Rosenbrock is the vehicle: stiff and
// curved from a far start (the ill-conditioned regime), locally well-behaved
// near its minimum (1, 1) (the well-conditioned regime).

use arael::simple_lm::{
    solve, BandError, CooMatrix, CscMatrix, LmConfig, LmProblem, LmResult,
};

/// Rosenbrock as least squares: r = [10*(y - x^2), 1 - x], cost = 0.5*|r|^2,
/// minimized at (1, 1) with cost 0.
struct Rosenbrock;

impl LmProblem<f64> for Rosenbrock {
    fn calc_cost(&mut self, p: &[f64]) -> f64 {
        let (x, y) = (p[0], p[1]);
        let (r1, r2) = (10.0 * (y - x * x), 1.0 - x);
        0.5 * (r1 * r1 + r2 * r2)
    }

    fn calc_grad_hessian_dense(&mut self, p: &[f64], grad: &mut [f64], h: &mut [f64]) -> f64 {
        let (x, y) = (p[0], p[1]);
        let r = [10.0 * (y - x * x), 1.0 - x];
        // j[k] = [dr_k/dx, dr_k/dy]
        let j = [[-20.0 * x, 10.0], [-1.0, 0.0]];
        grad.fill(0.0);
        h.fill(0.0);
        for k in 0..2 {
            for a in 0..2 {
                grad[a] += j[k][a] * r[k];
                for b in 0..2 {
                    h[a + b * 2] += j[k][a] * j[k][b];
                }
            }
        }
        self.calc_cost(p)
    }

    fn calc_grad_hessian_band(&mut self, _p: &[f64], _g: &mut [f64], _b: &mut [f64], _kd: usize)
        -> Result<f64, BandError> {
        unimplemented!("dense only")
    }
    fn calc_grad_hessian_sparse(&mut self, _p: &[f64], _g: &mut [f64], _c: &mut CooMatrix<f64>) -> f64 {
        unimplemented!("dense only")
    }
    fn calc_grad_hessian_sparse_direct(&mut self, _p: &[f64], _g: &mut [f64], _c: &mut CscMatrix<f64>) -> f64 {
        unimplemented!("dense only")
    }
    fn calc_grad_hessian_sparse_indexed(&mut self, _p: &[f64], _g: &mut [f64], _v: &mut [f64], _pos: &[usize]) -> f64 {
        unimplemented!("dense only")
    }
}

fn run(cfg: &LmConfig<f64>, x0: [f64; 2]) -> LmResult<f64> {
    solve(&x0, &mut Rosenbrock, cfg)
}

fn at_minimum(r: &LmResult<f64>) -> bool {
    (r.x[0] - 1.0).abs() < 1e-3 && (r.x[1] - 1.0).abs() < 1e-3
}

const FAR: [f64; 2] = [-1.2, 1.0];
const NEAR: [f64; 2] = [0.9, 0.81];

/// Default is the conservative preset.
#[test]
fn default_is_conservative() {
    let (d, c) = (LmConfig::<f64>::default(), LmConfig::<f64>::conservative());
    assert_eq!(d.initial_lambda, c.initial_lambda);
    assert_eq!(d.min_iters, c.min_iters);
    assert!(d.gradient_tolerance.is_none() && c.gradient_tolerance.is_none());
}

/// The robust presets solve the stiff (far-start) problem to the minimum.
#[test]
fn robust_presets_solve_the_stiff_problem() {
    for cfg in [LmConfig::conservative(), LmConfig::ill_conditioned()] {
        let r = run(&cfg.with_max_iters(200), FAR);
        assert!(at_minimum(&r), "x = {:?}, status = {:?}", r.x, r.status);
    }
}

/// well_conditioned fits a good start: it converges, in no more accepted steps
/// than conservative. On Rosenbrock's flat valley the near-Gauss-Newton step
/// can converge on the cost short of the analytic minimum -- that is a
/// convergence, so check the status, not the point.
#[test]
fn well_conditioned_fits_a_good_start() {
    let wc = run(&LmConfig::well_conditioned(), NEAR);
    let cons = run(&LmConfig::conservative(), NEAR);
    assert!(wc.status.is_success(), "well_conditioned status = {:?}", wc.status);
    assert!(at_minimum(&cons), "conservative x = {:?}", cons.x);
    assert!(
        wc.accepted_iterations <= cons.accepted_iterations,
        "well_conditioned took {} accepted steps, conservative {}",
        wc.accepted_iterations, cons.accepted_iterations,
    );
}

/// continue_from resumes a capped solve at its ending lambda and finishes,
/// without paying to re-climb: total accepted steps stay near a single solve.
#[test]
fn continue_from_resumes_a_capped_solve() {
    let partial = run(&LmConfig::conservative().with_max_iters(3), FAR);
    assert!(!at_minimum(&partial), "the cap should stop it short: x = {:?}", partial.x);

    let cfg = LmConfig::conservative().continue_from(&partial);
    let full = solve(&partial.x, &mut Rosenbrock, &cfg.with_max_iters(200));
    assert!(at_minimum(&full), "resumed x = {:?}, status = {:?}", full.x, full.status);

    let one_shot = run(&LmConfig::conservative().with_max_iters(200), FAR);
    assert!(
        partial.accepted_iterations + full.accepted_iterations <= one_shot.accepted_iterations + 2,
        "resume {}+{} vs one-shot {}",
        partial.accepted_iterations, full.accepted_iterations, one_shot.accepted_iterations,
    );
}
