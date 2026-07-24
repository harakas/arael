//! An f32 block under a plain (f64) root: block precision must match the
//! root's solve precision. Used to surface as an E0308 inside generated
//! code pointing at the macro attribute; now the mismatch names the
//! struct and block field.

use arael::model::{Param, SelfBlock};
use arael::refs;

#[arael::model]
#[arael(constraint(hb, {
    [(nf.v - nf.t) * 0.3]
}))]
struct Nf {
    v: Param<f32>,
    t: f32,
    hb: SelfBlock<Nf, f32>,
}

#[arael::model]
#[arael(root)]
struct W {
    nodes: refs::Vec<Nf>,
}

fn main() {}
