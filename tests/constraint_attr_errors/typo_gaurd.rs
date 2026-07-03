//! A typoed key inside constraint(...) must be rejected -- `gaurd` used
//! to be silently swallowed, compiling as an unguarded, always-active
//! constraint.

use arael::model::{Param, SelfBlock};

#[arael::model]
#[arael(root)]
#[arael(constraint(hb, gaurd = self.active, {
    [(m.x - 1.0) * m.isigma]
}))]
struct M {
    x: Param<f64>,
    active: bool,
    isigma: f64,
    hb: SelfBlock<M>,
}

fn main() {}
