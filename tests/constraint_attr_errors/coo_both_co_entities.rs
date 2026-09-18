//! `[hb, coo]` whose body reads params of BOTH the root and the
//! containing parent: a constraint couples to one co-entity.

use arael::model::{Param, SelfBlock};

#[arael::model]
#[arael(constraint([hb, coo], { [(e.x - curve.m - w.b) * 2.0] }))]
struct E {
    x: Param<f64>,
    hb: SelfBlock<E>,
}

#[arael::model]
struct Curve {
    m: Param<f64>,
    items: std::vec::Vec<E>,
    hb: SelfBlock<Curve>,
}

#[arael::model]
#[arael(root)]
struct W {
    b: Param<f64>,
    curves: std::vec::Vec<Curve>,
    hb: SelfBlock<W>,
}

fn main() {}
