//! One partial per scalar parameter.

#[arael::function(f)]
fn f_eval(x: f64, y: f64, derivs: bool) -> (f64, [f64; 1]) { let _ = derivs; (x * y, [y]) }

fn main() {}
