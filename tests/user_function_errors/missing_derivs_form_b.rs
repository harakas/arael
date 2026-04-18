//! Form B (opaque numerical eval fn) requires explicit `derivs = [...]`.

#[arael::function(foo)]
fn foo_eval(k: f32) -> f32 { k }

fn main() {}
