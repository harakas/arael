// Integer and bool data fields in constraint bodies. A body computes in the
// root's float type, so such a field may only pick a `match` arm or a
// `loss_select` kernel; any other read is a macro error naming the field.
// It used to reach the generated code and fail there ("cannot multiply
// `u32` by `f64`", reported at the root).

use arael::model::{Param, SelfBlock};
use arael::simple_lm::{LmConfig, LmProblem};

#[arael::model]
#[arael(constraint(hb, { [match a.k { 0 => a.x - 1.0, _ => a.x - 2.0 }] }))]
struct A { x: Param<f64>, k: u32, hb: SelfBlock<A> }

#[arael::model]
#[arael(constraint(hb, { [match b.k { 0 => b.x - 1.0, 1 => b.x - 3.0, _ => b.x }] }))]
struct B { x: Param<f64>, k: u64, hb: SelfBlock<B> }

#[arael::model]
#[arael(constraint(hb, { [match c.on { 0 => c.x - 1.0, _ => c.x - 4.0 }] }))]
struct C { x: Param<f64>, on: bool, hb: SelfBlock<C> }

#[arael::model]
#[arael(constraint(hb, loss = |s| loss_select(d.kind, s, 1.0), { [d.x - 5.0] }))]
struct D { x: Param<f64>, kind: i32, hb: SelfBlock<D> }

#[arael::model]
#[arael(root)]
struct R {
    a: std::vec::Vec<A>,
    b: std::vec::Vec<B>,
    c: std::vec::Vec<C>,
    d: std::vec::Vec<D>,
}

#[test]
fn integer_and_bool_fields_pick_arms() {
    let mut r = R {
        a: vec![A { x: Param::new(0.0), k: 1, hb: SelfBlock::new() }],
        b: vec![B { x: Param::new(0.0), k: 1, hb: SelfBlock::new() }],
        c: vec![C { x: Param::new(0.0), on: true, hb: SelfBlock::new() }],
        d: vec![D { x: Param::new(0.0), kind: 3, hb: SelfBlock::new() }],
    };
    let res = r.solve_dense(&LmConfig::conservative()).unwrap();
    assert!(res.status.is_success(), "{:?}", res.status);
    for (got, want) in [(r.a[0].x.value, 2.0), (r.b[0].x.value, 3.0),
                        (r.c[0].x.value, 4.0), (r.d[0].x.value, 5.0)] {
        assert!((got - want).abs() < 1e-9, "{got} vs {want}");
    }
}

/// Regenerate the snapshots with:
///     TRYBUILD=overwrite cargo test --test integer_fields
#[test]
fn integer_and_bool_fields_elsewhere_are_rejected() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/integer_fields/arithmetic.rs");
    t.compile_fail("tests/integer_fields/bool_branch.rs");
}
