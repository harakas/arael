// Any primitive type is usable as a plain model field without `#[arael(skip)]`.
// The macro needs both a no-op Model and a ModelSym for the field's type; those
// are provided for the whole primitive set (integers of every width, floats,
// bool, char, String). Non-parameter primitives are read at runtime, e.g. in a
// guard.

use arael::simple_lm::RootProblem;
use arael::model::{Param, SelfBlock};
use arael::refs;
use arael::simple_lm::LmProblem;

#[arael::model]
#[arael(constraint(hb, guard = self.kind == 2, {
    [pt.x - pt.target]
}))]
struct Pt {
    x: Param<f64>,
    target: f64,
    // A spread of primitive types as plain data fields -- every one must
    // compile with no #[arael(skip)].
    kind: i32,
    flag: bool,
    tiny: i8,
    small: u8,
    wide: i64,
    huge: u128,
    idx: usize,
    letter: char,
    label: String,
    hb: SelfBlock<Pt>,
}

#[arael::model]
#[arael(root)]
struct W {
    pts: refs::Vec<Pt>,
}

fn world(kind: i32) -> (W, Vec<f64>) {
    let mut w = W { pts: refs::Vec::new() };
    w.pts.push(Pt {
        x: Param::new(1.0),
        target: 3.0, // residual x - target = -2, so an active constraint costs 4
        kind,
        flag: true,
        tiny: -1,
        small: 7,
        wide: 1_000_000_000_000,
        huge: u128::MAX,
        idx: 42,
        letter: 'z',
        label: String::from("pt"),
        hb: SelfBlock::new(),
    });
    let mut params = Vec::new();
    w.serialize(&mut params);
    (w, params)
}

#[test]
fn primitive_field_read_in_guard() {
    // kind == 2 activates the constraint (cost 4); anything else filters it.
    let (mut on, p_on) = world(2);
    assert!((on.calc_cost(&p_on) - 4.0).abs() < 1e-12);

    let (mut off, p_off) = world(0);
    assert_eq!(off.calc_cost(&p_off), 0.0);
}
