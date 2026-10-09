//! A fn that returns its partials takes no `derivs = [...]`.

#[arael::function(f, derivs = [2.0 * x])]
fn f_eval(x: f64, derivs: bool) -> (f64, [f64; 1]) { (x * x, [if derivs { 2.0 * x } else { 0.0 }]) }

fn main() {}
