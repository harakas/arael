//! The macro registry is keyed by bare struct name: two #[arael::model]
//! structs with the same name (different modules) used to last-write-win
//! and corrupt each other's generated code.

mod first {
    use arael::model::{Param, SelfBlock};

    #[arael::model]
    pub struct Twin {
        pub x: arael::model::Param<f64>,
        pub hb: SelfBlock<Twin>,
    }

    // Reference the imports so the fixture stays warning-free.
    pub fn _touch(_: &Param<f64>) {}
}

mod second {
    use arael::model::SelfBlock;

    #[arael::model]
    pub struct Twin {
        pub y: arael::model::Param<f64>,
        pub hb: SelfBlock<Twin>,
    }
}

fn main() {}
