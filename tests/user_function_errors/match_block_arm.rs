//! A match arm in a `#[arael::function]` body is one expression; a
//! block with statements is rejected.

#[arael::function]
fn f(kind: arael::sym::E, a: arael::sym::E, b: arael::sym::E) -> arael::sym::E {
    match kind {
        0 => { let y = a; y },
        _ => b,
    }
}

fn main() {}
