// Tests for the #[arael(fit(...))] flagship feature (REVIEW2 section 5).
// fit() had only a doctest; this pins it against closed-form OLS, checks the
// Starship robust variant ignores an outlier, and exercises fit_with.

use arael::model::Param;
use arael::simple_lm::{FitProblem, LmConfig};

#[arael::model]
struct Pt { x: f32, y: f32 }

// Robust linear model y = a*x + b with Starship-suppressed residuals.
#[arael::model]
#[arael(fit(data, |e| {
    let r = (a * e.x + b - e.y) / sigma;
    gamma * atan(r / gamma)
}))]
struct LinearModel {
    a: Param<f32>,
    b: Param<f32>,
    data: Vec<Pt>,
    sigma: f32,
    gamma: f32,
}

/// Closed-form ordinary least squares on the same data (ground truth).
fn ols(data: &[Pt]) -> (f32, f32) {
    let n = data.len() as f64;
    let (mut sx, mut sxx, mut sy, mut sxy) = (0.0, 0.0, 0.0, 0.0);
    for p in data {
        let (x, y) = (p.x as f64, p.y as f64);
        sx += x; sxx += x * x; sy += y; sxy += x * y;
    }
    let a = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    let b = (sy - a * sx) / n;
    (a as f32, b as f32)
}

fn line_data(a: f32, b: f32, xs: &[f32]) -> Vec<Pt> {
    xs.iter().map(|&x| Pt { x, y: a * x + b }).collect()
}

fn model(data: Vec<Pt>, gamma: f32) -> LinearModel {
    LinearModel { a: Param::new(0.0), b: Param::new(0.0), data, sigma: 1.0, gamma }
}

#[test]
fn fit_matches_closed_form_ols() {
    // Clean data on a known line; a huge gamma makes the Starship kernel
    // ~identity, so the robust fit should land on the OLS solution.
    let xs: Vec<f32> = (0..20).map(|i| i as f32 * 0.5 - 5.0).collect();
    let data = line_data(2.0, -1.0, &xs);
    let (a_ols, b_ols) = ols(&data);
    let mut m = model(data, 1e6);
    let r = m.fit();
    assert!(r.end_cost < r.start_cost);
    assert!((m.a.value - a_ols).abs() < 1e-3, "a: fit {} vs ols {}", m.a.value, a_ols);
    assert!((m.b.value - b_ols).abs() < 1e-3, "b: fit {} vs ols {}", m.b.value, b_ols);
    assert!((m.a.value - 2.0).abs() < 1e-2 && (m.b.value + 1.0).abs() < 1e-2,
        "should recover the true line, got a={} b={}", m.a.value, m.b.value);
}

#[test]
fn starship_ignores_outlier() {
    // True line y = 2x - 1 plus one gross outlier. The robust fit should
    // track the inliers; plain OLS is dragged off by the outlier.
    let xs: Vec<f32> = (0..20).map(|i| i as f32 * 0.5 - 5.0).collect();
    let mut data = line_data(2.0, -1.0, &xs);
    data.push(Pt { x: 0.0, y: 50.0 }); // ~51 off the line

    let (a_ols, b_ols) = ols(&data);
    let mut m = model(data, 0.5); // tight gamma suppresses the outlier
    let r = m.fit();
    assert!(r.end_cost < r.start_cost);

    let robust_err = (m.a.value - 2.0).abs() + (m.b.value + 1.0).abs();
    let ols_err = (a_ols - 2.0).abs() + (b_ols + 1.0).abs();
    assert!(robust_err < 0.2, "robust should recover the true line, err {}", robust_err);
    assert!(robust_err < ols_err * 0.5,
        "robust (err {}) should clearly beat OLS (err {})", robust_err, ols_err);
}

#[test]
fn fit_with_custom_config() {
    let xs: Vec<f32> = (0..10).map(|i| i as f32).collect();
    let data = line_data(1.5, 0.5, &xs);
    let mut m = model(data, 1e6);
    let r = m.fit_with(&LmConfig { max_iters: 50, ..Default::default() });
    assert!(r.iterations <= 50);
    assert!((m.a.value - 1.5).abs() < 1e-2 && (m.b.value - 0.5).abs() < 1e-2,
        "should recover the line, got a={} b={}", m.a.value, m.b.value);
}

// fit64: the f64 variant of the fit attribute -- same shorthand, f64
// parameters, config, and result throughout.
#[arael::model]
#[arael(fit64(data, |e| a * e.x + b - e.y))]
struct LinearModel64 {
    a: Param<f64>,
    b: Param<f64>,
    data: std::vec::Vec<XY64>,
}

#[arael::model]
struct XY64 {
    x: f64,
    y: f64,
}

#[test]
fn fit64_recovers_line() {
    let mut m = LinearModel64 {
        a: Param::new(0.0),
        b: Param::new(0.0),
        data: (0..20).map(|i| {
            let x = i as f64 * 0.5;
            XY64 { x, y: 2.0 * x - 1.0 }
        }).collect(),
    };
    let r = m.fit_with(&LmConfig::<f64> { max_iters: 50, ..Default::default() });
    assert!(r.end_cost < 1e-18, "cost {}", r.end_cost);
    assert!((m.a.value - 2.0).abs() < 1e-9, "a = {}", m.a.value);
    assert!((m.b.value + 1.0).abs() < 1e-9, "b = {}", m.b.value);
}
