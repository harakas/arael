//! Symbolic sibling name must differ from the eval fn name.

#[arael::function(foo, derivs = [k])]
fn foo(k: f32) -> f32 { k }

fn main() {}
