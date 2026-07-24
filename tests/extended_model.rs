// The runtime-differentiation escape hatch: `#[arael(root, extended)]` +
// `ExtendedModel`, with the parameters living in `#[arael(skip_self_block)]`
// entities whose gradient and Hessian arrive through a root TripletBlock
// instead of per-entity SelfBlocks (the runtime_fit_demo shape). A line fit
// with residuals and derivatives supplied at runtime must reach the
// closed-form least-squares solution, on the dense and the sparse path.

use arael::model::{ExtendedModel, Param, TripletBlock};
use arael::refs;
use arael::simple_lm::{self, LmConfig, LmProblem};

#[arael::model]
#[arael(skip_self_block)]
struct Coefficient {
    value: Param<f64>,
}

#[arael::model]
#[arael(root, extended)]
struct Fit {
    coeffs: refs::Vec<Coefficient>,
    hb: TripletBlock<f64>,
    #[arael(skip)]
    data: Vec<(f64, f64)>,
}

impl ExtendedModel for Fit {
    fn extended_cost64(&self, params: &[f64]) -> f64 {
        let a = params[self.coeffs[0].value.index() as usize];
        let b = params[self.coeffs[1].value.index() as usize];
        self.data.iter().map(|&(x, y)| {
            let r = a * x + b - y;
            r * r
        }).sum()
    }

    fn extended_compute64(&mut self, params: &[f64], grad: &mut [f64]) {
        let ia = self.coeffs[0].value.index();
        let ib = self.coeffs[1].value.index();
        let a = params[ia as usize];
        let b = params[ib as usize];
        for &(x, y) in &self.data {
            let r = a * x + b - y;
            self.hb.add_residual(r, &[ia, ib], &[x, 1.0], grad);
        }
    }
}

fn data() -> Vec<(f64, f64)> {
    (0..10).map(|i| {
        let x = i as f64 * 0.3;
        (x, 2.0 * x + 1.0 + if i % 2 == 0 { 0.01 } else { -0.01 })
    }).collect()
}

/// Closed-form least-squares line through the data.
fn normal_equations(data: &[(f64, f64)]) -> (f64, f64) {
    let n = data.len() as f64;
    let (sx, sy, sxx, sxy) = data.iter().fold((0.0, 0.0, 0.0, 0.0),
        |(sx, sy, sxx, sxy), &(x, y)| (sx + x, sy + y, sxx + x * x, sxy + x * y));
    let a = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    let b = (sy - a * sx) / n;
    (a, b)
}

fn build() -> Fit {
    let mut m = Fit {
        coeffs: refs::Vec::new(),
        hb: TripletBlock::new(),
        data: data(),
    };
    m.coeffs.push(Coefficient { value: Param::new(0.0) });
    m.coeffs.push(Coefficient { value: Param::new(0.0) });
    m
}

#[test]
fn extended_line_fit_matches_normal_equations() {
    let (a_ref, b_ref) = normal_equations(&data());
    let cfg = LmConfig::<f64> {
        abs_precision: 1e-14,
        rel_precision: 1e-12,
        max_iters: 100,
        ..Default::default()
    };

    let mut dense = build();
    let r = dense.solve_dense(&cfg);
    assert!(r.end_cost < r.start_cost);
    assert!((dense.coeffs[0].value.value - a_ref).abs() < 1e-8,
        "dense a {} vs {}", dense.coeffs[0].value.value, a_ref);
    assert!((dense.coeffs[1].value.value - b_ref).abs() < 1e-8,
        "dense b {} vs {}", dense.coeffs[1].value.value, b_ref);

    // The sparse path must handle the TripletBlock-only structure (no
    // entity SelfBlocks exist to declare the pattern).
    let mut sparse = build();
    let mut p = Vec::new();
    sparse.serialize64(&mut p);
    let r = simple_lm::solve_sparse_faer(&p, &mut sparse, &cfg);
    sparse.deserialize64(&r.x);
    assert!((sparse.coeffs[0].value.value - a_ref).abs() < 1e-8,
        "sparse a {} vs {}", sparse.coeffs[0].value.value, a_ref);
    assert!((sparse.coeffs[1].value.value - b_ref).abs() < 1e-8,
        "sparse b {} vs {}", sparse.coeffs[1].value.value, b_ref);
}
