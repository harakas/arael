// LmConfig::num_threads: the sparse factorization and triangular solve run on
// faer's rayon pool. Everything else -- assembly, the Schur reduction, every
// other backend -- stays sequential.
//
// The route is pinned to faer's scalar Cholesky throughout. What these tests
// are about is one factorization run at two thread counts; the default
// (BlockSupernodalMode::Auto) takes the block supernodal only when sequential,
// so leaving the route to it would compare two different factorizations
// instead, and they agree only to the last ulp.

use arael::model::{CrossBlock, Param, SelfBlock};
use arael::refs::{self, Ref};
use arael::simple_lm::{
    lm_solve, BlockSupernodalMode, LmConfig, LmResult, RootProblem, SparseFaer,
};
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

/// One 60-point chain solved on faer's scalar Cholesky at `num_threads`.
fn solve_scalar(num_threads: usize) -> LmResult<f64> {
    let cfg = LmConfig::<f64> { max_iters: 200, num_threads, ..Default::default() };
    let mut chain = build_chain(60);
    let mut params = Vec::new();
    RootProblem::serialize(&mut chain, &mut params);
    let mut solver = SparseFaer::new().with_block_supernodal(BlockSupernodalMode::Never);
    lm_solve(&params, &mut solver, &mut chain, &cfg).unwrap()
}

/// Threads must not change the answer. The factorization is exact either way, so
/// the two solves take the same steps and land on the same parameters.
#[test]
fn threads_do_not_change_the_answer() {
    let seq = solve_scalar(1);
    let par = solve_scalar(4);

    assert_eq!(seq.status, par.status);
    assert_eq!(seq.iterations, par.iterations, "same steps");
    assert_eq!(seq.accepted_iterations, par.accepted_iterations);
    assert_eq!(seq.end_cost, par.end_cost, "same cost, to the bit");
    assert_eq!(seq.x, par.x, "same parameters, to the bit");
}

/// 0 means "every core", and must also not change the answer.
#[test]
fn zero_threads_means_all_cores() {
    let seq = solve_scalar(1);
    let all = solve_scalar(0);
    assert_eq!(seq.end_cost, all.end_cost);
    assert_eq!(seq.x, all.x);
}

/// The default is sequential. Whatever else changes, that must not.
#[test]
fn the_default_is_one_thread() {
    assert_eq!(LmConfig::<f64>::default().num_threads, 1);
    assert_eq!(LmConfig::<f32>::default().num_threads, 1);
}
