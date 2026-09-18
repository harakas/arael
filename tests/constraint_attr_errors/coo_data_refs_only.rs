//! `constraint(coo, ..)` whose refs all point at param-less records: there
//! is no entity to couple.

use arael::refs::{self, Ref};

#[arael::model]
struct Mark {
    anchor: f64,
}

#[arael::model]
#[arael(constraint(coo, { [(m.anchor - l.target) * 0.5] }))]
struct L {
    #[arael(ref = root.marks)] m: Ref<Mark>,
    target: f64,
}

#[arael::model]
#[arael(root)]
struct W {
    marks: refs::Vec<Mark>,
    ls: std::vec::Vec<L>,
}

fn main() {}
