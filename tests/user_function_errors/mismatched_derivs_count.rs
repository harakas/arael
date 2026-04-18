//! `derivs = [...]` entry count must match fn arity.

#[arael::function(foo, derivs = [k, k])]
fn foo_eval(k: f32) -> f32 { k }

fn main() {}
