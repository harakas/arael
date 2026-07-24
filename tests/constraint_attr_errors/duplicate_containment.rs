//! A cross-constraint struct contained in two root fields would get its
//! sweep emitted for the first field only (the second's constraints
//! silently contribute nothing) -- rejected at expansion. SelfBlock
//! entities in several collections ARE supported (one sweep each); the
//! guard covers the remaining shapes.

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
#[arael(root)]
struct W {
    points: refs::Vec<P>,
    ties: std::vec::Vec<Tie>,
    more_ties: std::vec::Vec<Tie>,
}

fn main() {}
