//! A `match` in a `#[arael::function]` body takes no arm guards.

#[arael::function]
fn f(kind: arael::sym::E, a: arael::sym::E, b: arael::sym::E) -> arael::sym::E {
    match kind {
        0 if a > b => a,
        _ => b,
    }
}

fn main() {}
