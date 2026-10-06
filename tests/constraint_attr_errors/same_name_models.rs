//! Two `#[arael::model]` structs with the same name in one crate. The
//! macro's registry is keyed by bare struct name, so the second one is
//! refused instead of sharing the first one's constraints.

use arael::model::{Param, SelfBlock};

#[arael::model]
#[arael(root)]
#[arael(constraint(hb, { [m.x - 3.0] }))]
struct M {
    x: Param<f64>,
    hb: SelfBlock<M>,
}

mod other {
    use arael::model::{Param, SelfBlock};

    #[arael::model]
    #[arael(root)]
    #[arael(constraint(hb, { [m.x - 3.0, m.x - 4.0] }))]
    struct M {
        x: Param<f64>,
        hb: SelfBlock<M>,
    }
}

fn main() {
    let _ = M { x: Param::new(0.0), hb: SelfBlock::new() };
}
