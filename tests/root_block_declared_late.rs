// A root's SelfBlock<Self> declared AFTER its collections: the block
// span walk emits each span at its block FIELD's position while the
// offsets follow the params' serialize positions, so the root's span
// arrived last with offset 0 -- and every sparse solve panicked in
// block_partition_from_spans ("param block spans overlap or are
// unsorted"). The partition builder now sorts; dense and sparse must
// agree on this shape.

use arael::model::{Param, SelfBlock};
use arael::refs;
use arael::simple_lm::{LmConfig, LmProblem};

#[arael::model]
#[arael(constraint(root.hb, {
    [obs.y - (fit.m * obs.x + fit.c)]
}))]
struct Obs {
    x: f64,
    y: f64,
}

#[arael::model]
#[arael(constraint(hb, {
    [(n.v - n.t) * n.w]
}))]
struct N {
    v: Param<f64>,
    t: f64,
    w: f64,
    hb: SelfBlock<N>,
}

#[arael::model]
#[arael(root)]
struct Fit {
    m: Param<f64>,
    c: Param<f64>,
    obs: std::vec::Vec<Obs>,
    items: refs::Vec<N>,
    // Deliberately last: the params above serialize first, so this
    // block's span has offset 0 yet is emitted after the items' spans.
    hb: SelfBlock<Fit>,
}

fn build() -> Fit {
    let mut fit = Fit {
        m: Param::new(0.0),
        c: Param::new(0.0),
        obs: Vec::new(),
        items: refs::Vec::new(),
        hb: SelfBlock::new(),
    };
    for i in 0..6 {
        let x = i as f64;
        fit.obs.push(Obs { x, y: 2.0 * x + 1.0 + if i % 2 == 0 { 0.05 } else { -0.05 } });
    }
    for (t, w) in [(1.5, 1.0), (-0.3, 2.0), (0.7, 0.5)] {
        fit.items.push(N { v: Param::new(0.0), t, w, hb: SelfBlock::new() });
    }
    fit
}

#[test]
fn sparse_solve_with_a_late_root_block() {
    let cfg = LmConfig { max_iters: 50, ..Default::default() };
    let mut dense = build();
    let rd = dense.solve_dense(&cfg).unwrap();
    assert!(rd.status.is_success(), "{:?}", rd.status);

    let mut sparse = build();
    let rs = sparse.solve_sparse(&cfg).unwrap();
    assert!(rs.status.is_success(), "{:?}", rs.status);

    assert!((rd.end_cost - rs.end_cost).abs() < 1e-12);
    assert!((dense.m.value - sparse.m.value).abs() < 1e-9);
    assert!((dense.c.value - sparse.c.value).abs() < 1e-9);
    for i in 0..3 {
        assert!((dense.items[i].v.value - sparse.items[i].v.value).abs() < 1e-9);
    }
}
