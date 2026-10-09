//! A fn returning its partials needs the trailing `derivs: bool`.

#[arael::function(f)]
fn f_eval(x: f64) -> (f64, [f64; 1]) { (x * x, [2.0 * x]) }

fn main() {}
