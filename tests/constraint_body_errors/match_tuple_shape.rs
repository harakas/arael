//! Tuple arms of a body `match` have the same element kinds: a vec2
//! beside a vec3 in the first element is rejected, naming both shapes.

use arael::model::{Param, SelfBlock};
use arael::vect::{vect2, vect3};

#[arael::model]
#[arael(root)]
#[arael(constraint(hb, {
    let (d, s) = match p.kind {
        0 => (p.v, 1.0),
        _ => (p.w, 2.0),
    };
    [d.x * s]
}))]
struct P {
    v: Param<vect2<f64>>,
    w: Param<vect3<f64>>,
    kind: u32,
    hb: SelfBlock<P>,
}

fn main() {}
