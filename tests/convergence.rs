// Integration tests: verify solver convergence on known problems.
// Checks both final cost and iteration count bounds.

use arael::simple_lm::*;

#[test]
fn quadratic_converges_fast() {
    // f(x,y) = (x-3)^2 + (y-7)^2, minimum at (3,7), cost=0
    let mut p = FnProblem {
        cost: |x: &[f64]| (x[0] - 3.0).powi(2) + (x[1] - 7.0).powi(2),
        grad_hessian: |x: &[f64], g: &mut [f64], h: &mut [f64]| {
            g[0] = 2.0 * (x[0] - 3.0);
            g[1] = 2.0 * (x[1] - 7.0);
            h[0] = 2.0; h[1] = 0.0; h[2] = 0.0; h[3] = 2.0;
            (x[0] - 3.0).powi(2) + (x[1] - 7.0).powi(2)
        },
    };
    let result = solve(&[0.0, 0.0], &mut p, &LmConfig::default());
    assert!(result.end_cost < 1e-10, "cost={}", result.end_cost);
    assert!(result.iterations <= 30, "iters={}", result.iterations);
}

#[test]
fn rosenbrock_converges() {
    // f(x,y) = (1-x)^2 + 100*(y-x^2)^2, minimum at (1,1), cost=0
    let mut p = FnProblem {
        cost: |p: &[f64]| (1.0 - p[0]).powi(2) + 100.0 * (p[1] - p[0] * p[0]).powi(2),
        grad_hessian: |p: &[f64], g: &mut [f64], h: &mut [f64]| {
            let (x, y) = (p[0], p[1]);
            g[0] = -2.0 * (1.0 - x) - 400.0 * x * (y - x * x);
            g[1] = 200.0 * (y - x * x);
            h[0] = 2.0 + 1200.0 * x * x - 400.0 * y;
            h[1] = -400.0 * x;
            h[2] = -400.0 * x;
            h[3] = 200.0;
            (1.0 - p[0]).powi(2) + 100.0 * (p[1] - p[0] * p[0]).powi(2)
        },
    };
    let result = solve(
        &[-1.0, 1.0], &mut p,
        &LmConfig { max_iters: 500, ..Default::default() },
    );
    assert!((result.x[0] - 1.0).abs() < 1e-3, "x={}", result.x[0]);
    assert!((result.x[1] - 1.0).abs() < 1e-3, "y={}", result.x[1]);
    assert!(result.iterations <= 100, "iters={}", result.iterations);
}

#[test]
fn high_dimensional_quadratic() {
    // f(x) = sum_i (x_i - i)^2, 20 params, minimum at x_i = i
    let n = 20;
    let mut p = FnProblem {
        cost: |x: &[f64]| {
            x.iter().enumerate().map(|(i, &xi)| (xi - i as f64).powi(2)).sum()
        },
        grad_hessian: |x: &[f64], g: &mut [f64], h: &mut [f64]| {
            let n = x.len();
            for i in 0..n {
                g[i] = 2.0 * (x[i] - i as f64);
                for j in 0..n { h[i * n + j] = 0.0; }
                h[i * n + i] = 2.0;
            }
            x.iter().enumerate().map(|(i, &xi)| (xi - i as f64).powi(2)).sum()
        },
    };
    let x0: Vec<f64> = vec![0.0; n];
    let result = solve(&x0, &mut p, &LmConfig::default());
    assert!(result.end_cost < 1e-10, "cost={}", result.end_cost);
    assert!(result.iterations <= 30, "iters={}", result.iterations);
    for i in 0..n {
        assert!((result.x[i] - i as f64).abs() < 1e-5, "x[{}]={}", i, result.x[i]);
    }
}

#[test]
fn solver_does_not_waste_iterations_on_converged_problem() {
    // Already at minimum -- should terminate quickly
    let mut p = FnProblem {
        cost: |x: &[f64]| x[0].powi(2) + x[1].powi(2),
        grad_hessian: |x: &[f64], g: &mut [f64], h: &mut [f64]| {
            g[0] = 2.0 * x[0];
            g[1] = 2.0 * x[1];
            h[0] = 2.0; h[1] = 0.0; h[2] = 0.0; h[3] = 2.0;
            x[0].powi(2) + x[1].powi(2)
        },
    };
    let result = solve(&[0.0, 0.0], &mut p, &LmConfig::default());
    assert!(result.end_cost < 1e-15, "cost={}", result.end_cost);
    assert!(result.iterations <= 20, "iters={} (should stop early at minimum)", result.iterations);
}
