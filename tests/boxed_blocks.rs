// BoxedSelfBlock / BoxedCrossBlock must produce a bit-for-bit identical solve
// to the inline SelfBlock / CrossBlock (same math, heap storage), and the
// release / auto-realloc round trip must not change the result.

use arael::model::{Model, Param, SelfBlock, CrossBlock, BoxedSelfBlock, BoxedCrossBlock};
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

// ---- boxed model (identical math, Boxed* blocks) ----
#[arael::model]
#[arael(constraint(hb, guard = self.is_anchor, {
    [bpoint.pos.x * bchain.anchor, bpoint.pos.y * bchain.anchor]
}))]
#[arael(constraint(hb, {
    let d = bpoint.pos - bpoint.pos_value;
    [d.x * bchain.drift, d.y * bchain.drift]
}))]
struct BPoint {
    pos: Param<vect2d>,
    is_anchor: bool,
    hb: BoxedSelfBlock<BPoint>,
}
#[arael::model]
#[arael(constraint(hb, {
    let d = b.pos - a.pos;
    [(d.norm() - blink.rest) * bchain.spring]
}))]
struct BLink {
    #[arael(ref = root.points)] a: Ref<BPoint>,
    #[arael(ref = root.points)] b: Ref<BPoint>,
    rest: f64,
    hb: BoxedCrossBlock<BPoint, BPoint>,
}
#[arael::model]
#[arael(root)]
struct BChain {
    points: refs::Arena<BPoint>,
    links: std::vec::Vec<BLink>,
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
    for i in 0..N {
        c.points.push(Point { pos: Param::new(init_pos(i)), is_anchor: i == 0, hb: SelfBlock::new() });
    }
    for i in 1..N {
        let a = c.points.ref_at(i - 1);
        let b = c.points.ref_at(i);
        c.links.push(Link { a, b, rest: 1.1, hb: CrossBlock::new() });
    }
    c
}

fn build_boxed() -> BChain {
    let mut c = BChain {
        points: refs::Arena::new(), links: std::vec::Vec::new(),
        anchor: 100.0, drift: 0.01, spring: 1.0,
    };
    for i in 0..N {
        c.points.push(BPoint { pos: Param::new(init_pos(i)), is_anchor: i == 0, hb: BoxedSelfBlock::new() });
    }
    for i in 1..N {
        let a = c.points.ref_at(i - 1);
        let b = c.points.ref_at(i);
        c.links.push(BLink { a, b, rest: 1.1, hb: BoxedCrossBlock::new() });
    }
    c
}

/// Inline chain with the first `fixed_upto` points frozen (`Param::fixed`).
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
fn build_boxed_fixed(fixed_upto: usize) -> BChain {
    let mut c = BChain {
        points: refs::Arena::new(), links: std::vec::Vec::new(),
        anchor: 100.0, drift: 0.01, spring: 1.0,
    };
    for i in 0..N {
        let pos = if i < fixed_upto { Param::fixed(init_pos(i)) } else { Param::new(init_pos(i)) };
        c.points.push(BPoint { pos, is_anchor: i == 0, hb: BoxedSelfBlock::new() });
    }
    for i in 1..N {
        let a = c.points.ref_at(i - 1);
        let b = c.points.ref_at(i);
        c.links.push(BLink { a, b, rest: 1.1, hb: BoxedCrossBlock::new() });
    }
    c
}

fn cfg() -> LmConfig<f64> {
    LmConfig { abs_precision: 1e-14, rel_precision: 1e-12, max_iters: 200, ..Default::default() }
}

#[test]
fn boxed_blocks_solve_identically_to_inline() {
    let inline_x = build_inline().solve_sparse(&cfg()).x;
    let boxed_x = build_boxed().solve_sparse(&cfg()).x;
    assert_eq!(inline_x, boxed_x, "boxed blocks must give a bit-identical solve");
}

#[test]
fn boxed_blocks_release_and_realloc() {
    // Release every boxed Hessian (they become None) up front, then solve:
    // zero() must auto-reallocate on the first pass and produce a result
    // bit-identical to a plain solve -- release/realloc is transparent.
    let mut released = build_boxed();
    released.release_blocks();
    let released_x = released.solve_sparse(&cfg()).x;

    let plain_x = build_boxed().solve_sparse(&cfg()).x;
    assert_eq!(released_x, plain_x, "release + auto-realloc must not change the solve");

    // Releasing again after a solve (frees the materialized Hessians) and
    // re-solving must not crash and must stay finite.
    released.release_blocks();
    let r = released.solve_sparse(&cfg());
    assert!(r.end_cost.is_finite());
}

#[test]
fn boxed_blocks_partial_optimization_allocates_only_active() {
    // Freeze the first FIXED points (Param::fixed); optimize only the tail.
    // A boxed block must allocate its Hessian only when its entity is active,
    // so the frozen sub-tree stays unallocated -- that is the memory win.
    const FIXED: usize = 3;
    let mut c = build_boxed_fixed(FIXED);

    // Snapshot frozen positions to assert they never move.
    let frozen: std::vec::Vec<vect2d> =
        (0..FIXED).map(|i| c.points[c.points.ref_at(i)].pos.value).collect();

    // The gated partial solve must match an inline partial solve bit-for-bit
    // (exercises mixed active/fixed cross-blocks, e.g. the link into point 2).
    let boxed_x = c.solve_sparse(&cfg()).x;
    let inline_x = build_inline_fixed(FIXED).solve_sparse(&cfg()).x;
    assert_eq!(boxed_x, inline_x, "gated partial solve must match inline");

    // Self-blocks: allocated iff the point is active (index >= FIXED).
    for i in 0..N {
        let allocated = c.points[c.points.ref_at(i)].hb.is_allocated();
        assert_eq!(allocated, i >= FIXED,
            "point {i}: expected allocated={}, got {allocated}", i >= FIXED);
    }

    // Cross-blocks: link at vec index i couples points i and i+1; allocated iff
    // at least one endpoint is active.
    for (i, link) in c.links.iter().enumerate() {
        let touches_active = i >= FIXED || (i + 1) >= FIXED;
        assert_eq!(link.hb.is_allocated(), touches_active,
            "link {i} ({}-{}): expected allocated={touches_active}", i, i + 1);
    }

    // Frozen points must not have moved.
    for i in 0..FIXED {
        let p = c.points[c.points.ref_at(i)].pos.value;
        assert_eq!((p.x, p.y), (frozen[i].x, frozen[i].y), "frozen point {i} moved");
    }
}

#[test]
fn boxed_blocks_allocation_follows_activity_across_solves() {
    // Guards the core invariant: a boxed block is allocated iff its entity is
    // active, and that is re-decided on every solve. If the solver flow ever
    // stopped gating allocation on activity this would fail.
    let mut c = build_boxed(); // all active
    c.solve_sparse(&cfg());
    for i in 0..N {
        let r = c.points.ref_at(i);
        assert!(c.points[r].hb.is_allocated(), "point {i} allocated while active");
    }

    // Freeze the first three points and re-solve: their self-blocks must be
    // released by set_indices (all-fixed indices), active ones stay allocated.
    for i in 0..3 {
        let r = c.points.ref_at(i);
        c.points[r].pos.optimize = false;
    }
    c.solve_sparse(&cfg());
    for i in 0..N {
        let r = c.points.ref_at(i);
        assert_eq!(c.points[r].hb.is_allocated(), i >= 3, "point {i} allocation must track activity");
    }

    // Unfreeze: re-solving must re-allocate every block.
    for i in 0..3 {
        let r = c.points.ref_at(i);
        c.points[r].pos.optimize = true;
    }
    c.solve_sparse(&cfg());
    for i in 0..N {
        let r = c.points.ref_at(i);
        assert!(c.points[r].hb.is_allocated(), "point {i} must re-allocate when active again");
    }
}
