// Port of Ceres's robust_curve_fitting example: fit y = exp(m*x + c) to data
// with two gross outliers. Three fits from one model, selected per solve by a
// `mode` field that guards three constraints:
//   0 naive     -- plain residual y - exp(m*x + c)
//   1 cauchy    -- block loss loss_cauchy(s, 0.25)  (== Ceres CauchyLoss(0.5); the scale is squared)
//   2 starship  -- per-element gamma*atan(r/gamma) wrapper
//
// The fit parameters m, c live on Curve. Each Obs is nested in a Batch (the
// container the root iterates) and targets Curve's Hessian block through its
// `curve` reference (a remote block), so only m and c are optimized.
use arael::covariance::{CovMode, Covariance};
use arael::model::{Param, SelfBlock};
use arael::refs::{self, Ref};
use arael::simple_lm::{LmConfig, LmProblem};

#[arael::model]
struct Curve {
    m: Param<f64>,
    c: Param<f64>,
    hb: SelfBlock<Curve>,
}

#[arael::model]
#[arael(constraint(curve.hb, parent = batch, guard = self.mode == 0, { // naive
    [obs.y - exp(curve.m * obs.x + curve.c)]
}))]
#[arael(constraint(curve.hb, parent = batch, guard = self.mode == 1, loss = |s| loss_cauchy(s, 0.25), { // cauchy (squared scale)
    [obs.y - exp(curve.m * obs.x + curve.c)]
}))]
#[arael(constraint(curve.hb, parent = batch, guard = self.mode == 2, { // starship
    let r = obs.y - exp(curve.m * obs.x + curve.c);
    [obs.gamma * atan(r / obs.gamma)]
}))]
struct Obs {
    #[arael(ref = root.curves)]
    curve: Ref<Curve>,
    x: f64,
    y: f64,
    gamma: f64,
    mode: i32,
}

#[arael::model]
struct Batch {
    obs: std::vec::Vec<Obs>,
}

#[arael::model]
#[arael(root)]
struct Fit {
    curves: refs::Vec<Curve>,
    batches: refs::Vec<Batch>,
}

const DATA: &[(f64, f64)] = &[
    (0.000, 1.133898), (0.075, 1.334902), (0.150, 1.213546), (0.225, 1.252016),
    (0.300, 1.392265), (0.375, 1.314458), (0.450, 1.472541), (0.525, 1.536218),
    (0.600, 1.355679), (0.675, 1.463566), (0.750, 1.490201), (0.825, 1.658699),
    (0.900, 1.067574), (0.975, 1.464629), (1.050, 1.402653), (1.125, 1.713141),
    (1.200, 1.527021), (1.275, 1.702632), (1.350, 1.423899), (1.425, 5.543078),
    (1.500, 5.664015), (1.575, 1.732484), (1.650, 1.543296), (1.725, 1.959523),
    (1.800, 1.685132), (1.875, 1.951791), (1.950, 2.095346), (2.025, 2.361460),
    (2.100, 2.169119), (2.175, 2.061745), (2.250, 2.178641), (2.325, 2.104346),
    (2.400, 2.584470), (2.475, 1.914158), (2.550, 2.368375), (2.625, 2.686125),
    (2.700, 2.712395), (2.775, 2.499511), (2.850, 2.558897), (2.925, 2.309154),
    (3.000, 2.869503), (3.075, 3.116645), (3.150, 3.094907), (3.225, 2.471759),
    (3.300, 3.017131), (3.375, 3.232381), (3.450, 2.944596), (3.525, 3.385343),
    (3.600, 3.199826), (3.675, 3.423039), (3.750, 3.621552), (3.825, 3.559255),
    (3.900, 3.530713), (3.975, 3.561766), (4.050, 3.544574), (4.125, 3.867945),
    (4.200, 4.049776), (4.275, 3.885601), (4.350, 4.110505), (4.425, 4.345320),
    (4.500, 4.161241), (4.575, 4.363407), (4.650, 4.161576), (4.725, 4.619728),
    (4.800, 4.737410), (4.875, 4.727863), (4.950, 4.669206),
];

// gamma sets the starship suppression scale (in residual units). Inlier noise
// is ~0.15, the outliers are ~4 off; 0.5 matches the Cauchy scale.
const GAMMA: f64 = 0.5;

fn build() -> Fit {
    let mut fit = Fit { curves: refs::Vec::new(), batches: refs::Vec::new() };
    let cref = fit.curves.push(Curve { m: Param::new(0.0), c: Param::new(0.0), hb: SelfBlock::new() });
    let obs = DATA.iter()
        .map(|&(x, y)| Obs { curve: cref, x, y, gamma: GAMMA, mode: 0 })
        .collect();
    fit.batches.push(Batch { obs });
    fit
}

// Returns (m, c, [sigma_m, sigma_c]). The std devs come from the covariance API
// at the fit: Sigma = 2 H^-1 over Curve's m, c. The residuals here are not
// whitened, so it is the unit-noise covariance (scale by the residual RMS for an
// absolute interval); under the robust losses it is a local curvature interval.
fn fit_mode(fit: &mut Fit, name: &str, mode: i32) -> (f64, f64, Option<Vec<f64>>) {
    // Label the verbose solver output so it is clear which fit it belongs to.
    eprintln!("\n=== solving: {name} ===");
    for b in fit.batches.iter_mut() {
        for o in b.obs.iter_mut() {
            o.mode = mode;
        }
    }
    // Fit each mode from m=c=0; solve_dense reads the start point from the model.
    for c in fit.curves.iter_mut() {
        c.m = Param::new(0.0);
        c.c = Param::new(0.0);
    }
    // Start well-damped: m=c=0 is far from the solution.
    // conservative, with a high starting lambda: robust losses flatten the
    // cost surface around outliers, and small first steps keep the naive fit
    // from chasing them.
    let cfg = LmConfig::conservative().with_verbose(true).with_initial_lambda(1.0);
    let result = fit.solve_dense(&cfg);
    result.pretty_print();
    let sd = fit.assemble_covariance(CovMode::PerQuery).ok().map(|cov| cov.std_dev(&fit.curves[0]));
    (result.x[0], result.x[1], sd)
}

fn main() {
    let mut fit = build();
    let naive = fit_mode(&mut fit, "naive", 0);
    let cauchy = fit_mode(&mut fit, "cauchy", 1);
    let starship = fit_mode(&mut fit, "starship", 2);

    let y = |(m, c): (f64, f64), x: f64| (m * x + c).exp();
    let truth = (0.3, 0.1);
    let row = |name: &str, f: &(f64, f64, Option<Vec<f64>>)| match &f.2 {
        Some(sd) => println!("  {name:<9} m={:.4} +/- {:.4}  c={:.4} +/- {:.4}", f.0, sd[0], f.1, sd[1]),
        None => println!("  {name:<9} m={:.4}  c={:.4}", f.0, f.1),
    };

    println!("\nfit  y = exp(m*x + c)  (+/- 1 sigma from the covariance API):");
    println!("  true      m={:.4}  c={:.4}", truth.0, truth.1);
    row("naive", &naive);
    row("cauchy", &cauchy);
    row("starship", &starship);

    // Reading vs the true curve vs each fit, evaluated at the data points. The
    // two outliers (y ~ 5.5 near x = 1.4) drag the naive column up; cauchy and
    // starship stay close to the true curve.
    println!("\n     x   reading     true    naive   cauchy  starship");
    for &(x, reading) in DATA {
        println!(
            "  {:5.3}   {:6.3}   {:6.3}   {:6.3}   {:6.3}   {:6.3}",
            x, reading, y(truth, x), y((naive.0, naive.1), x), y((cauchy.0, cauchy.1), x), y((starship.0, starship.1), x)
        );
    }
}
