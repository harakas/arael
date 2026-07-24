// Graduated optimization as a tested feature: a robust-loss scale field
// on the root is stepped across LmSession warm re-solves. The loss
// expression must re-read the field on every solve (not bake in the value
// seen at first analysis), and the warm session must accept the changed
// problem.
//
// Problem: one scalar `x` measured by five inliers around 0 and one gross
// outlier at 10, under a Geman-McClure block loss with threshold
// `root.c2`. At c2 = 1e4 the loss is near-quadratic and the optimum is
// the contaminated mean; annealed down to c2 = 1 through the same
// session, the outlier is rejected and x lands on the inlier mean. Both
// ends are asserted, so the test fails either if graduation has no effect
// (field baked in) or if the warm re-solve breaks.

use arael::model::{Param, SelfBlock};
use arael::simple_lm::{Dense, LmConfig, LmProblem, LmSession};

#[arael::model]
#[arael(constraint(root.hb, loss = |s| loss_geman_mcclure(s, root.c2), {
    [root.x - m.z]
}))]
struct M {
    z: f64,
}

#[arael::model]
#[arael(root)]
struct W {
    x: Param<f64>,
    c2: f64,
    hb: SelfBlock<W>,
    data: std::vec::Vec<M>,
}

const MEASUREMENTS: [f64; 6] = [-0.2, -0.1, 0.0, 0.1, 0.2, 10.0];

fn build(x0: f64, c2: f64) -> W {
    W {
        x: Param::new(x0),
        c2,
        hb: SelfBlock::new(),
        data: MEASUREMENTS.iter().map(|&z| M { z }).collect(),
    }
}

#[test]
fn field_graduation_through_warm_session() {
    let cfg = LmConfig::<f64> {
        abs_precision: 1e-14,
        rel_precision: 1e-12,
        max_iters: 100,
        ..Default::default()
    };
    let mut w = build(0.5, 1e4);
    let mut session = LmSession::new(Dense);

    // Near-quadratic rung: the optimum is (close to) the contaminated
    // mean -- the outlier pulls with almost full weight.
    let r = session.solve(&mut w, &cfg);
    assert!(r.end_cost <= r.start_cost);
    let contaminated_mean = MEASUREMENTS.iter().sum::<f64>() / MEASUREMENTS.len() as f64;
    assert!((w.x.value - contaminated_mean).abs() < 0.05,
        "quadratic rung x = {} vs mean {}", w.x.value, contaminated_mean);

    // Anneal to the true loss through the same session: the changed field
    // must take effect on every re-solve.
    for c2 in [100.0, 10.0, 1.0] {
        w.c2 = c2;
        session.solve(&mut w, &cfg);
    }
    assert!(w.x.value.abs() < 0.02, "robust rung x = {}", w.x.value);

    // The graduated result matches a cold solve at the final threshold
    // started inside the inlier basin.
    let mut cold = build(0.0, 1.0);
    cold.solve_dense(&cfg);
    assert!((w.x.value - cold.x.value).abs() < 1e-6,
        "graduated {} vs cold {}", w.x.value, cold.x.value);
}
