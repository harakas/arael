//! Data fields whose paths spell alike as identifiers (`p.x` and `p_x`, an
//! entity's and the root's) build and solve. The device head names each
//! data path a record field of its own; a clash there once failed the
//! build of such a model, with or without the `gpu` feature.

use arael::model::{Param, SelfBlock};
use arael::refs;
#[allow(unused_imports)]
use arael::simple_lm::{LmConfig, LmProblem, RootProblem};
use arael::vect::vect2;

#[arael::model]
#[arael(constraint(hb, { [(alike.x - alike.p.x - alike.p_x) * 2.0, (alike.x - alike.p.x - alike.p_x) * (root.q.x - root.q_x)] }))]
struct Alike {
    x: Param<f64>,
    p: vect2<f64>,
    p_x: f64,
    hb: SelfBlock<Alike>,
}

#[arael::model]
#[arael(root)]
struct AlikeRoot {
    items: refs::Vec<Alike>,
    q: vect2<f64>,
    q_x: f64,
}

#[test]
fn alike_data_paths_build_and_solve() {
    let mut items = refs::Vec::new();
    for k in 0..4 {
        let t = 0.5 * k as f64;
        items.push(Alike { x: Param::new(t + 0.3), p: vect2::new(t, 3.0), p_x: 0.2, hb: SelfBlock::new() });
    }
    let mut m = AlikeRoot { items, q: vect2::new(0.7, -4.0), q_x: -0.2 };
    let r = m.solve_dense(&LmConfig::default()).unwrap();
    assert!(r.end_cost < 1e-20, "{}", r.end_cost);
    for (k, it) in m.items.iter().enumerate() {
        assert!((it.x.value - (0.5 * k as f64 + 0.2)).abs() < 1e-9, "item {k}: {}", it.x.value);
    }
}
