// `Model::release_blocks` between solves must not change a result. A
// block owns no storage any more -- its values are in the solve's block
// store -- so the call frees nothing, but it is public API and a solve
// after one has to come out the same.

use arael::model::{CrossBlock, Model, Param, SelfBlock};
use arael::simple_lm::LmConfig;
use arael::simple_lm::LmProblem;
use arael::vect::vect2d;
use arael::refs::{self, Ref};

// ---- inline model ----
#[arael::model]
#[arael(constraint(hb, guard = self.is_anchor, {
    [point.pos.x * chain.anchor, point.pos.y * chain.anchor]
}))]
#[arael(constraint(hb, {
    let d = point.pos - point.pos_value;
    [d.x * chain.drift, d.y * chain.drift]
}))]
struct Point {
    pos: Param<vect2d>,
    is_anchor: bool,
    hb: SelfBlock<Point>,
}
#[arael::model]
#[arael(constraint(hb, {
    let d = b.pos - a.pos;
    [(d.norm() - link.rest) * chain.spring]
}))]
struct Link {
    #[arael(ref = root.points)] a: Ref<Point>,
    #[arael(ref = root.points)] b: Ref<Point>,
    rest: f64,
    hb: CrossBlock<Point, Point>,
}
#[arael::model]
#[arael(root)]
struct Chain {
    points: refs::Vec<Point>,
    links: std::vec::Vec<Link>,
    anchor: f64,
    drift: f64,
    spring: f64,
}

const N: usize = 6;

fn init_pos(i: usize) -> vect2d {
    let x = i as f64 + ((i * 7 % 3) as f64 - 1.0) * 0.3;
    let y = if i % 2 == 0 { 0.0 } else { 0.8 } + ((i * 5 % 3) as f64 - 1.0) * 0.2;
    vect2d::new(x, y)
}

fn build_inline() -> Chain {
    let mut c = Chain {
        points: refs::Vec::new(), links: std::vec::Vec::new(),
        anchor: 100.0, drift: 0.01, spring: 1.0,
    };
    let mut pts = std::vec::Vec::new();
    for i in 0..N {
        pts.push(c.points.push(Point { pos: Param::new(init_pos(i)), is_anchor: i == 0, hb: SelfBlock::new() }));
    }
    for i in 1..N {
        let a = pts[i - 1];
        let b = pts[i];
        c.links.push(Link { a, b, rest: 1.1, hb: CrossBlock::new() });
    }
    c
}

fn build_inline_fixed(fixed_upto: usize) -> Chain {
    let mut c = Chain {
        points: refs::Vec::new(), links: std::vec::Vec::new(),
        anchor: 100.0, drift: 0.01, spring: 1.0,
    };
    for i in 0..N {
        let pos = if i < fixed_upto { Param::fixed(init_pos(i)) } else { Param::new(init_pos(i)) };
        c.points.push(Point { pos, is_anchor: i == 0, hb: SelfBlock::new() });
    }
    for i in 1..N {
        let a = c.points.ref_at(i - 1);
        let b = c.points.ref_at(i);
        c.links.push(Link { a, b, rest: 1.1, hb: CrossBlock::new() });
    }
    c
}

/// Boxed chain with the first `fixed_upto` points frozen (`Param::fixed`), so
/// only the tail of the chain is optimized.

fn cfg() -> LmConfig<f64> {
    LmConfig { abs_precision: 1e-14, rel_precision: 1e-12, max_iters: 200, ..Default::default() }
}

#[test]
fn release_between_solves_does_not_change_the_result() {
    // Two solves in a row, with and without a release between them: the
    // release is what has to make no difference. (A second solve starts
    // from the first one's answer, so it does not equal a fresh solve.)
    let mut plain = build_inline();
    plain.solve_sparse(&cfg()).unwrap();
    let plain_x = plain.solve_sparse(&cfg()).unwrap().x;

    let mut released = build_inline();
    released.solve_sparse(&cfg()).unwrap();
    released.release_blocks();
    let released_x = released.solve_sparse(&cfg()).unwrap().x;

    assert_eq!(released_x, plain_x, "a release between solves changed the next one");
}

#[test]
fn a_release_before_the_first_solve_changes_nothing() {
    let plain = build_inline().solve_sparse(&cfg()).unwrap().x;
    let mut m = build_inline();
    m.release_blocks();
    assert_eq!(m.solve_sparse(&cfg()).unwrap().x, plain);
}

#[test]
fn release_with_fixed_params_does_not_change_the_result() {
    // A frozen sub-tree used to leave its blocks unallocated; now there is
    // nothing to allocate either way, and the answer must not move.
    const FIXED: usize = 3;
    let plain = build_inline_fixed(FIXED).solve_sparse(&cfg()).unwrap();
    let frozen: std::vec::Vec<vect2d> = (0..FIXED).map(init_pos).collect();

    let mut m = build_inline_fixed(FIXED);
    m.release_blocks();
    let got = m.solve_sparse(&cfg()).unwrap();
    assert_eq!(got.x, plain.x);
    assert_eq!(got.end_cost, plain.end_cost);

    let pts: std::vec::Vec<_> = m.points.refs().collect();
    for i in 0..FIXED {
        let (p, want) = (m.points[pts[i]].pos.value, frozen[i]);
        assert!(p.x == want.x && p.y == want.y, "frozen point {i} moved");
    }
}
