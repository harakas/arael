// The minimal import surface to define a model and solve it -- pinned as
// a test so API changes that grow the required imports get noticed.

use arael::model::{Param, SelfBlock};
use arael::simple_lm::{LmConfig, LmProblem};

#[arael::model]
#[arael(root)]
#[arael(constraint(hb, { [m.x - 3.0] }))]
struct M {
    x: Param<f64>,
    hb: SelfBlock<M>,
}

#[test]
fn two_use_lines_suffice() {
    let mut m = M { x: Param::new(0.0), hb: SelfBlock::new() };
    let result = m.solve_sparse(&LmConfig::default());
    assert!(result.end_cost < 1e-12);
    assert!((m.x.value - 3.0).abs() < 1e-6);
}
