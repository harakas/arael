//! `#[arael::function]` on a fn whose signature matches neither
//! Form A (E -> E) nor Form B (f32/f64 -> same) should fail with a
//! message pointing at the signature.

#[arael::function]
fn bad(x: f32) -> arael_sym::E {
    arael_sym::symbol("x")
}

fn main() {}
