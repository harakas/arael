//! A root-owned TripletBlock is a secondary block, never the primary:
//! `root.<triplet>` alone is rejected with a pointer at the bracketed form.

use arael::model::{Param, SelfBlock, TripletBlock};

#[arael::model]
#[arael(constraint(root.hbt, { [e.y - root.a * e.x] }))]
struct E {
    x: f64,
    y: f64,
}

#[arael::model]
#[arael(root)]
struct Fit {
    a: Param<f64>,
    hb: SelfBlock<Fit>,
    hbt: TripletBlock<f64>,
    data: std::vec::Vec<E>,
}

fn main() {}
