//! Two block fields of one struct disagreeing on precision: rejected at
//! the struct's own expansion, before any root is involved.

use arael::model::{Param, SelfBlock, TripletBlock};

#[arael::model]
#[arael(root, f32)]
struct W {
    v: Param<f32>,
    hb: SelfBlock<W, f32>,
    hbt: TripletBlock<f64>,
}

fn main() {}
