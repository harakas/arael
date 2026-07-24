//! A containment + ref cycle: A holds a collection of B, and B's ref
//! targets A's collection. The sweep would iterate `aa` mutably while
//! writing ref-target blocks into it -- unsound to emit. This shape
//! used to overflow rustc's stack during binding registration (SIGSEGV,
//! no diagnostic); now the recursion is cycle-guarded and the shape is
//! rejected with the paths named.

use arael::model::{Param, SelfBlock, CrossBlock};
use arael::refs::{self, Ref};

#[arael::model]
#[arael(constraint(hb, {
    [(a2.v - a1.v - b.d) * 1.5]
}))]
struct B {
    #[arael(ref = root.aa)]
    a1: Ref<A>,
    #[arael(ref = root.aa)]
    a2: Ref<A>,
    d: f64,
    hb: CrossBlock<A, A>,
}

#[arael::model]
#[arael(constraint(hb, {
    [(a.v - a.t) * 0.3]
}))]
struct A {
    v: Param<f64>,
    t: f64,
    bs: std::vec::Vec<B>,
    hb: SelfBlock<A>,
}

#[arael::model]
#[arael(root)]
struct W {
    aa: refs::Vec<A>,
}

fn main() {}
