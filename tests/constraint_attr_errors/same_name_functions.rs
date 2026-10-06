//! Two `#[arael::function]` items with the same name in one crate. The
//! macro keys functions by bare name, so the second one is refused
//! instead of replacing the first in every constraint body.

use arael_sym::E;

#[arael::function]
fn square(x: E) -> E {
    x * x
}

mod other {
    use arael_sym::E;

    #[arael::function]
    fn square(x: E) -> E {
        x + x
    }
}

fn main() {}
