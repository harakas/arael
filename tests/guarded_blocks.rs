// A cross block is in the Hessian pattern only while a constraint with a
// true guard writes it. The block store keeps the block's entry with its
// parameter indices unset, so the entry keeps its slot and lists no cell.
// A block with two writers stays while either is kept, a block with an
// unguarded writer always, a parent-owned block while one child is kept.
// A warm LmSession panics when a guard switches a block in or out without
// invalidate().

use arael::model::{CrossBlock, Param, SelfBlock};
use arael::refs::{self, Ref};
use arael::simple_lm::{lm_solve, LmConfig, LmProblemInternals, LmSession, RootProblem, SparseFaer};

#[arael::model]
#[arael(constraint(hb, {
    [(pose.x - pose.ax) * 0.1, (pose.y - pose.ay) * 0.1]
}))]
struct Pose {
    x: Param<f64>,
    y: Param<f64>,
    ax: f64,
    ay: f64,
    hb: SelfBlock<Pose>,
}

// One guarded constraint on the block.
#[arael::model]
#[arael(constraint(hb, guard = self.on, {
    [b.x - a.x - link.dx, b.y - a.y - link.dy]
}))]
struct Link {
    #[arael(ref = root.poses)]
    a: Ref<Pose>,
    #[arael(ref = root.poses)]
    b: Ref<Pose>,
    dx: f64,
    dy: f64,
    on: bool,
    hb: CrossBlock<Pose, Pose>,
}

// Two constraints on the block, one of them kept whatever `strong` is.
#[arael::model]
#[arael(constraint(hb, guard = self.strong, {
    [(b.x - a.x - either.dx) * 2.0, (b.y - a.y - either.dy) * 2.0]
}))]
#[arael(constraint(hb, guard = !self.strong, {
    [(b.x - a.x - either.dx) * 0.5, (b.y - a.y - either.dy) * 0.5]
}))]
struct Either {
    #[arael(ref = root.poses)]
    a: Ref<Pose>,
    #[arael(ref = root.poses)]
    b: Ref<Pose>,
    dx: f64,
    dy: f64,
    strong: bool,
    hb: CrossBlock<Pose, Pose>,
}

// A guarded and an unguarded constraint on the block.
#[arael::model]
#[arael(constraint(hb, guard = self.on, {
    [b.x - a.x - always.dx, b.y - a.y - always.dy]
}))]
#[arael(constraint(hb, {
    [(b.x - a.x) * 0.01]
}))]
struct Always {
    #[arael(ref = root.poses)]
    a: Ref<Pose>,
    #[arael(ref = root.poses)]
    b: Ref<Pose>,
    dx: f64,
    dy: f64,
    on: bool,
    hb: CrossBlock<Pose, Pose>,
}

// A block on the parent, written by every kept child.
#[arael::model]
struct Pair {
    #[arael(ref = root.poses)]
    a: Ref<Pose>,
    #[arael(ref = root.poses)]
    b: Ref<Pose>,
    matches: std::vec::Vec<Hit>,
    hb: CrossBlock<Pose, Pose>,
}

#[arael::model]
#[arael(constraint(parent.hb, guard = self.on, {
    [parent.b.x - parent.a.x - hit.d]
}))]
struct Hit {
    d: f64,
    on: bool,
}

#[arael::model]
#[arael(root)]
struct World {
    poses: refs::Vec<Pose>,
    links: std::vec::Vec<Link>,
    eithers: std::vec::Vec<Either>,
    always: std::vec::Vec<Always>,
    pairs: std::vec::Vec<Pair>,
}

const N: usize = 30;

fn poses() -> refs::Vec<Pose> {
    let mut poses = refs::Vec::new();
    for i in 0..N {
        let (tx, ty) = (i as f64, 0.5 * i as f64);
        poses.push(Pose { x: Param::new(tx + 0.3), y: Param::new(ty - 0.2), ax: tx, ay: ty, hb: SelfBlock::new() });
    }
    poses
}

fn empty() -> World {
    World { poses: poses(), links: Vec::new(), eithers: Vec::new(), always: Vec::new(), pairs: Vec::new() }
}

/// A chain of links, and one from every pose to the pose three ahead:
/// `on(k)` says whether link `k` is on. A long link is the only one
/// between its two poses.
fn chain(on: impl Fn(usize) -> bool) -> World {
    let mut w = empty();
    let mut k = 0;
    for step in [1, 3] {
        for i in 0..N - step {
            let (a, b) = (w.poses.ref_at(i), w.poses.ref_at(i + step));
            w.links.push(Link { a, b, dx: step as f64, dy: 0.5 * step as f64, on: on(k), hb: CrossBlock::new() });
            k += 1;
        }
    }
    w
}

/// [`chain`] with the links that are off left out of the model.
fn chain_without(on: impl Fn(usize) -> bool) -> World {
    let mut w = chain(&on);
    let mut k = 0;
    w.links.retain(|_| {
        k += 1;
        on(k - 1)
    });
    w
}

/// H's cells as the block stores of `threads` threads list them, sorted.
fn cells(w: &mut World, threads: usize) -> Vec<(u32, u32)> {
    let mut x = Vec::new();
    w.serialize(&mut x);
    let mut ctx = arael::threads::Context::new();
    ctx.set_threads(threads);
    w.begin_with_context(&mut ctx);
    let mut cells = Vec::new();
    w.collect_hessian_cells(&mut cells, &mut ctx);
    cells.sort_unstable();
    cells.dedup();
    cells
}

fn config(threads: usize) -> LmConfig<f64> {
    LmConfig { max_iters: 50, num_threads: threads, ..Default::default() }
}

/// Every seventh link off: 7 does not divide the chain's length, so both
/// short and long links are among them.
fn seventh(k: usize) -> bool {
    k % 7 != 2
}

/// The pattern of a model with links turned off is the pattern of the
/// model without them, at any store count, and smaller than with every
/// link on.
#[test]
fn a_block_no_kept_constraint_writes_is_out_of_the_pattern() {
    for threads in [1, 3, 8] {
        let guarded = cells(&mut chain(seventh), threads);
        let without = cells(&mut chain_without(seventh), threads);
        let whole = cells(&mut chain(|_| true), threads);
        assert_eq!(guarded, without, "{threads} threads");
        assert!(guarded.len() < whole.len(), "{threads} threads: {} cells, {} with every link on", guarded.len(), whole.len());
    }
}

/// A model with links turned off solves as the model without them.
#[test]
fn a_guarded_model_solves_as_the_model_without_the_dropped_instances() {
    for threads in [1, 4] {
        let solve = |w: &mut World| {
            let mut x = Vec::new();
            w.serialize(&mut x);
            lm_solve(&x, &mut SparseFaer::new(), w, &config(threads)).unwrap()
        };
        let guarded = solve(&mut chain(seventh));
        let without = solve(&mut chain_without(seventh));
        assert!((guarded.end_cost - without.end_cost).abs() <= 1e-9 * without.end_cost, "{threads} threads: {} against {}", guarded.end_cost, without.end_cost);
        for (g, w) in guarded.x.iter().zip(&without.x) {
            assert!((g - w).abs() < 1e-6, "{threads} threads: {g} against {w}");
        }
    }
}

/// A block two constraints with opposite guards write is kept whichever is
/// on.
#[test]
fn two_writers_with_opposite_guards_keep_their_block() {
    let build = |strong: &dyn Fn(usize) -> bool| {
        let mut w = empty();
        for i in 0..N - 2 {
            let (a, b) = (w.poses.ref_at(i), w.poses.ref_at(i + 2));
            w.eithers.push(Either { a, b, dx: 2.0, dy: 1.0, strong: strong(i), hb: CrossBlock::new() });
        }
        w
    };
    let mixed = cells(&mut build(&|i| i % 3 == 0), 3);
    assert_eq!(mixed, cells(&mut build(&|_| true), 3));
    assert_eq!(mixed, cells(&mut build(&|_| false), 3));
    assert_eq!(mixed.len(), N + N - 2);
}

/// A block with an unguarded writer is kept with its guarded writer off.
#[test]
fn an_unguarded_writer_keeps_its_block() {
    let build = |on: bool| {
        let mut w = empty();
        for i in 0..N - 2 {
            let (a, b) = (w.poses.ref_at(i), w.poses.ref_at(i + 2));
            w.always.push(Always { a, b, dx: 2.0, dy: 1.0, on, hb: CrossBlock::new() });
        }
        w
    };
    let off = cells(&mut build(false), 3);
    assert_eq!(off, cells(&mut build(true), 3));
    assert_eq!(off.len(), N + N - 2);
}

/// A parent-owned block is kept while one child under the parent is, and
/// dropped with every child off, as with no child.
#[test]
fn a_parent_owned_block_stays_while_one_child_is_kept() {
    let build = |children: &[bool]| {
        let mut w = empty();
        let (a, b) = (w.poses.ref_at(2), w.poses.ref_at(9));
        // the poses' own constraints put them 7 apart: a kept child pulls
        // against that
        let matches = children.iter().map(|&on| Hit { d: 5.0, on }).collect();
        w.pairs.push(Pair { a, b, matches, hb: CrossBlock::new() });
        w
    };
    for threads in [1, 3] {
        let none = cells(&mut build(&[]), threads);
        assert_eq!(none.len(), N, "{threads} threads: the poses' own blocks");
        assert_eq!(cells(&mut build(&[false, false]), threads), none, "{threads} threads: every child off");
        for kept in [&[true, false][..], &[false, true], &[true, true]] {
            assert_eq!(cells(&mut build(kept), threads).len(), N + 1, "{threads} threads: {kept:?}");
        }
    }
    // the kept child's residual is in the solve
    let mut w = build(&[false, true]);
    let mut x = Vec::new();
    w.serialize(&mut x);
    let with = lm_solve(&x, &mut SparseFaer::new(), &mut w, &config(1)).unwrap();
    let mut w = build(&[false, false]);
    let mut x = Vec::new();
    w.serialize(&mut x);
    let without = lm_solve(&x, &mut SparseFaer::new(), &mut w, &config(1)).unwrap();
    assert!(with.end_cost > without.end_cost + 1e-6, "{} against {}", with.end_cost, without.end_cost);
}

/// A guard that switches a cross block out of the pattern, in a warm
/// session without `invalidate()`: the session panics.
#[test]
#[should_panic(expected = "a guard changed without invalidate()")]
fn a_guard_flipped_in_a_warm_session_panics() {
    let mut w = chain(seventh);
    let mut session = LmSession::new(SparseFaer::new());
    session.solve(&mut w, &config(1)).unwrap();
    // link 2 is the only one between its poses
    assert!(!w.links[2].on);
    w.links[3].on = false;
    let _ = session.solve(&mut w, &config(1));
}

/// The same with one block switched in for every block switched out: the
/// block counts stay, and the session still panics.
#[test]
#[should_panic(expected = "a guard changed without invalidate()")]
fn a_guard_flip_that_keeps_the_count_of_kept_blocks_panics() {
    let mut w = chain(seventh);
    let mut session = LmSession::new(SparseFaer::new());
    session.solve(&mut w, &config(1)).unwrap();
    w.links[2].on = true;
    w.links[3].on = false;
    let _ = session.solve(&mut w, &config(1));
}

/// After `invalidate()` the solve runs cold over the new guards and gives
/// what a fresh solve gives.
#[test]
fn after_invalidate_a_flipped_guard_solves_cold() {
    let flipped = |k: usize| seventh(k) && k != 3;
    let mut w = chain(seventh);
    let mut session = LmSession::new(SparseFaer::new());
    session.solve(&mut w, &config(1)).unwrap();
    let mut fresh = chain(flipped);
    for (p, q) in w.poses.iter().zip(fresh.poses.iter_mut()) {
        q.x.value = p.x.value;
        q.y.value = p.y.value;
    }
    w.links[3].on = false;
    session.invalidate();
    let warm = session.solve(&mut w, &config(1)).unwrap();
    let mut x = Vec::new();
    fresh.serialize(&mut x);
    let cold = lm_solve(&x, &mut SparseFaer::new(), &mut fresh, &config(1)).unwrap();
    assert_eq!(warm.end_cost.to_bits(), cold.end_cost.to_bits());
    assert_eq!(warm.iterations, cold.iterations);
}

/// A guard flip that changes no block's writers from some to none is not
/// a change of the pattern: the warm solve runs and gives the cold one's
/// result.
#[test]
fn a_flip_between_two_writers_of_a_block_keeps_the_session() {
    let build = |strong: &dyn Fn(usize) -> bool| {
        let mut w = empty();
        for i in 0..N - 1 {
            let (a, b) = (w.poses.ref_at(i), w.poses.ref_at(i + 1));
            w.eithers.push(Either { a, b, dx: 1.0, dy: 0.5, strong: strong(i), hb: CrossBlock::new() });
        }
        w
    };
    let mut w = build(&|i| i % 2 == 0);
    let mut session = LmSession::new(SparseFaer::new());
    session.solve(&mut w, &config(1)).unwrap();
    let mut fresh = build(&|i| i % 2 == 1);
    for (p, q) in w.poses.iter().zip(fresh.poses.iter_mut()) {
        q.x.value = p.x.value;
        q.y.value = p.y.value;
    }
    for e in w.eithers.iter_mut() {
        e.strong = !e.strong;
    }
    let warm = session.solve(&mut w, &config(1)).unwrap();
    let mut x = Vec::new();
    fresh.serialize(&mut x);
    let cold = lm_solve(&x, &mut SparseFaer::new(), &mut fresh, &config(1)).unwrap();
    assert_eq!(warm.end_cost.to_bits(), cold.end_cost.to_bits());
}
