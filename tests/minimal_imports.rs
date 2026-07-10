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

// The same program through the prelude: one glob import covers it.
mod via_prelude {
    use arael::prelude::*;

    #[arael::model]
    #[arael(root)]
    #[arael(constraint(hb, { [p.x - p.t.x, p.y - p.t.y] }))]
    struct P {
        x: Param<f64>,
        y: Param<f64>,
        t: vect2d,
        hb: SelfBlock<P>,
    }

    #[test]
    fn one_glob_import_suffices() {
        let mut p = P { x: Param::new(0.0), y: Param::new(0.0),
                        t: vect2d::new(1.0, 2.0), hb: SelfBlock::new() };
        let result = p.solve_sparse(&LmConfig::default());
        assert!(result.end_cost < 1e-12);
        assert!((p.x.value - 1.0).abs() < 1e-6 && (p.y.value - 2.0).abs() < 1e-6);
    }
}
