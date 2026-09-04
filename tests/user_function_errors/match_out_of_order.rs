//! Match arm patterns in a `#[arael::function]` body must be the
//! integer literals 0, 1, ... in order.

#[arael::function]
fn f(kind: arael::sym::E, a: arael::sym::E, b: arael::sym::E) -> arael::sym::E {
    match kind {
        1 => a,
        0 => b,
    }
}

fn main() {}
