//! A semicolon-terminated expression statement before the residual array
//! was silently treated as extra residuals.

use arael::model::{Param, SelfBlock};

#[arael::model]
#[arael(root)]
#[arael(constraint(hb, {
    (m.x - 1.0) * m.isigma;
    [(m.x - 2.0) * m.isigma]
}))]
struct M {
    x: Param<f64>,
    isigma: f64,
    hb: SelfBlock<M>,
}

fn main() {}
