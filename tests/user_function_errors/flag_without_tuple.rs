//! A trailing `derivs: bool` on a fn that returns no partials.

#[arael::function(f)]
fn f_eval(x: f64, derivs: bool) -> f64 { let _ = derivs; x * x }

fn main() {}
