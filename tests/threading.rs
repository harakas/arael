// LmConfig::num_threads: the sparse factorization and triangular solve run on
// faer's rayon pool. Everything else -- assembly, the Schur reduction, every
// other backend -- stays sequential.
//
// The default route throughout: BlockSupernodalMode::Auto takes the block
// supernodal at any thread count, and its dense kernels are the thing the
// threads go into. Bit-identity across thread counts is therefore a property
// of one factorization, not a coincidence between two -- faer's kernels split
// their output, never their reduction.

use arael::model::{CrossBlock, Param, SelfBlock};
use arael::refs::{self, Ref};
use arael::simple_lm::{LmConfig, LmProblem, LmResult};
use arael::vect::vect2d;

#[arael::model]
#[arael(constraint(hb, guard = self.is_anchor, {
    [point.pos.x * chain.anchor, point.pos.y * chain.anchor]
}))]
#[arael(constraint(hb, {
    let d = point.pos - point.pos_value;
    [d.x * chain.drift, d.y * chain.drift]
}))]
struct Point { pos: Param<vect2d>, is_anchor: bool, hb: SelfBlock<Point> }

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
    anchor: f64, drift: f64, spring: f64,
}

fn build_chain(n: usize) -> Chain {
    let mut c = Chain {
        points: refs::Vec::new(), links: std::vec::Vec::new(),
        anchor: 100.0, drift: 0.01, spring: 1.0,
    };
    for i in 0..n {
        let pos = vect2d::new(i as f64 * 0.5, if i % 2 == 0 { 0.7 } else { -0.7 });
        c.points.push(Point { pos: Param::new(pos), is_anchor: i == 0, hb: SelfBlock::new() });
    }
    for i in 1..n {
        let (a, b) = (c.points.ref_at(i - 1), c.points.ref_at(i));
        c.links.push(Link { a, b, rest: 1.0, hb: CrossBlock::new() });
    }
    c
}

/// One 60-point chain solved by the default sparse backend at `num_threads`.
fn solve(num_threads: usize) -> LmResult<f64> {
    let cfg = LmConfig::<f64> { max_iters: 200, num_threads, ..Default::default() };
    build_chain(60).solve_sparse(&cfg).unwrap()
}

/// Parameters agree to rounding, and the cost with them.
fn assert_same_answer(a: &LmResult<f64>, b: &LmResult<f64>) {
    assert_eq!(a.status, b.status);
    assert!((a.end_cost - b.end_cost).abs() <= 1e-12 * (1.0 + a.end_cost.abs()),
        "same cost: {} vs {}", a.end_cost, b.end_cost);
    assert_eq!(a.x.len(), b.x.len());
    for (x, y) in a.x.iter().zip(&b.x) {
        assert!((x - y).abs() <= 1e-9 * (1.0 + x.abs()), "same parameters: {} vs {}", x, y);
    }
}

/// Threads must not change the answer. The factorization is exact either
/// way, and the threaded assembly sums each entity per thread and then
/// across the threads, so the two solves agree to rounding rather than to
/// the bit.
#[test]
fn threads_do_not_change_the_answer() {
    let seq = solve(1);
    let par = solve(4);

    assert_eq!(seq.iterations, par.iterations, "same steps");
    assert_eq!(seq.accepted_iterations, par.accepted_iterations);
    assert_same_answer(&seq, &par);
}

/// 0 means "every core", and must also not change the answer.
#[test]
fn zero_threads_means_all_cores() {
    assert_same_answer(&solve(1), &solve(0));
}

/// The default is sequential. Whatever else changes, that must not.
#[test]
fn the_default_is_one_thread() {
    assert_eq!(LmConfig::<f64>::default().num_threads, 1);
    assert_eq!(LmConfig::<f32>::default().num_threads, 1);
}
