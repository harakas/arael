//! A cross-constraint struct reachable through TWO nested containment
//! paths: iteration drives from a single resolved location, so the
//! second path's constraints would silently never run -- rejected at
//! expansion, listing the dotted paths.

use arael::model::{CrossBlock, Param, SelfBlock};
use arael::refs::{self, Ref};

#[arael::model]
struct P {
    x: Param<f64>,
    hb: SelfBlock<P>,
}

#[arael::model]
#[arael(constraint(hb, { [(a.x - b.x) * 2.0] }))]
struct Tie {
    #[arael(ref = root.points)]
    a: Ref<P>,
    #[arael(ref = root.points)]
    b: Ref<P>,
    hb: CrossBlock<P, P>,
}

#[arael::model]
struct BundleA {
    ties: std::vec::Vec<Tie>,
}

#[arael::model]
struct BundleB {
    ties: std::vec::Vec<Tie>,
}

#[arael::model]
#[arael(root)]
struct W {
    points: refs::Vec<P>,
    ba: std::vec::Vec<BundleA>,
    bb: std::vec::Vec<BundleB>,
}

fn main() {}
