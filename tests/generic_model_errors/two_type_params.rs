// Exactly one Float-bounded type parameter is supported.

use arael::model::Param;
use arael::utils::Float;

#[arael::model]
#[arael(skip_self_block)]
struct Pair<T: Float, U: Float> {
    a: Param<T>,
    b: Param<U>,
}

fn main() {}
