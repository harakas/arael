//! The root keywords `cost_plain`, `cost_kahan` and `cost_f64`. The cost
//! of 200k rows whose squares are exact dyadic rationals is compared
//! against the exact sum, formed in integers: the compensated and the
//! widened accumulators land within a few ulps, the plain ones hundreds
//! away, and the cost-only sweep agrees with the cost the block assembly
//! computes alongside.

use arael::model::{Param, SelfBlock};
use arael::refs;
use arael::simple_lm::{CooMatrix, LmProblem, RootProblem};
use arael::utils::Float;

/// One row: `v * d`, with `v` a parameter held at 1 so the residual is
/// the datum `d` exactly.
#[arael::model]
#[arael(constraint(hb, { [term.v * term.d] }))]
struct Term<T: Float> {
    v: Param<T>,
    d: T,
    hb: SelfBlock<Term<T>, T>,
}

#[arael::model]
#[arael(root, f32, cost_plain)]
struct PlainF32 { terms: refs::Vec<Term<f32>> }

#[arael::model]
#[arael(root, f32, cost_kahan)]
struct KahanF32 { terms: refs::Vec<Term<f32>> }

#[arael::model]
#[arael(root, f32, cost_f64)]
struct WideF32 { terms: refs::Vec<Term<f32>> }

#[arael::model]
#[arael(root, f32, cost_f64, cost_kahan)]
struct WideKahanF32 { terms: refs::Vec<Term<f32>> }

#[arael::model]
#[arael(root)]
struct PlainF64 { terms: refs::Vec<Term<f64>> }

/// The same rows under a parent: each group's loop is a scope of the
/// sum, so a row's chain of adds is the group's length, not the model's.
#[arael::model]
struct Group {
    terms: std::vec::Vec<Term<f64>>,
}

#[arael::model]
#[arael(root)]
struct NestedF64 { groups: refs::Vec<Group> }

#[arael::model]
#[arael(root, cost_kahan)]
struct KahanF64 { terms: refs::Vec<Term<f64>> }

const N: usize = 200_000;
const SCALE: f64 = 16_777_216.0; // 2^24

/// Integers below 2^24 spread over the whole range, so the squares span
/// magnitudes and every add rounds.
fn datum(i: usize) -> u64 {
    ((i as u64).wrapping_mul(2_654_435_761) % (1 << 24)) + 1
}

fn terms<T: Float>() -> refs::Vec<Term<T>> {
    let mut v = refs::Vec::new();
    for i in 0..N {
        let d = T::from(datum(i) as f64 / SCALE).unwrap();
        v.push(Term { v: Param::new(T::one()), d, hb: SelfBlock::new() });
    }
    v
}

/// The exact sum of the squares, as integers over 2^48.
fn exact_sum() -> f64 {
    let s: u128 = (0..N).map(|i| { let m = datum(i) as u128; m * m }).sum();
    s as f64 / (SCALE * SCALE)
}

/// The exact sum of the squares as f32 rounds them, which is what an f32
/// accumulator receives.
fn exact_sum_of_f32_squares() -> f64 {
    let s: u128 = (0..N).map(|i| {
        let d = (datum(i) as f64 / SCALE) as f32;
        let sq = d * d;
        (sq as f64 * SCALE * SCALE) as u128
    }).sum();
    s as f64 / (SCALE * SCALE)
}

fn ulps_f32(a: f64, b: f64) -> f64 { (a - b).abs() / (b as f32).abs() as f64 * (1.0 / f32::EPSILON as f64) }
fn ulps_f64(a: f64, b: f64) -> f64 { (a - b).abs() / b.abs() / f64::EPSILON }

/// Cost from the cost-only sweep and from the block assembly.
fn costs<P: LmProblem<T> + RootProblem<T>, T: Float>(m: &mut P) -> (f64, f64) {
    let mut x = Vec::new();
    RootProblem::serialize(m, &mut x);
    let c = m.calc_cost(&x).to_f64().unwrap();
    let mut g = vec![T::zero(); x.len()];
    let mut coo = CooMatrix::new(x.len());
    let cg = m.calc_grad_hessian_sparse(&x, &mut g, &mut coo).to_f64().unwrap();
    (c, cg)
}

#[test]
fn plain_f32_drifts() {
    let mut m = PlainF32 { terms: terms() };
    let (c, cg) = costs(&mut m);
    let e = ulps_f32(c, exact_sum_of_f32_squares());
    assert!(e > 16.0, "plain f32 error {e} ulps, expected the drift of a 200k-term sum");
    assert!(ulps_f32(cg, c) <= 1.0, "assembly cost {cg} != sweep cost {c}");
}

#[test]
fn kahan_f32_is_exact_to_a_few_ulps() {
    let mut m = KahanF32 { terms: terms() };
    let (c, cg) = costs(&mut m);
    let e = ulps_f32(c, exact_sum_of_f32_squares());
    assert!(e <= 4.0, "kahan f32 error {e} ulps");
    assert!(ulps_f32(cg, c) <= 1.0, "assembly cost {cg} != sweep cost {c}");
}

#[test]
fn f64_accumulator_on_f32_root() {
    let mut m = WideF32 { terms: terms() };
    let (c, cg) = costs(&mut m);
    // The squares are exact in f64 and the sum is rounded once to f32.
    let e = ulps_f32(c, exact_sum());
    assert!(e <= 1.0, "f64-accumulated f32 error {e} ulps");
    assert!(ulps_f32(cg, c) <= 1.0, "assembly cost {cg} != sweep cost {c}");
}

#[test]
fn f64_and_kahan_combine() {
    let mut m = WideKahanF32 { terms: terms() };
    let (c, _) = costs(&mut m);
    assert!(ulps_f32(c, exact_sum()) <= 1.0);
}

#[test]
fn plain_f64_drifts() {
    let mut m = PlainF64 { terms: terms() };
    let (c, cg) = costs(&mut m);
    let e = ulps_f64(c, exact_sum());
    assert!(e > 16.0, "plain f64 error {e} ulps, expected the drift of a 200k-term sum");
    assert!(ulps_f64(cg, c) <= 1.0, "assembly cost {cg} != sweep cost {c}");
}

#[test]
fn nested_scopes_cut_the_drift() {
    let mut flat = PlainF64 { terms: terms() };
    let (flat_cost, _) = costs(&mut flat);
    let flat_err = ulps_f64(flat_cost, exact_sum());

    // 450 groups of about 445 rows: both levels of the sum are short.
    let mut nested = NestedF64 { groups: refs::Vec::new() };
    let per_group = 445;
    let mut i = 0;
    while i < N {
        let mut g = Group { terms: Vec::new() };
        for _ in 0..per_group {
            if i == N { break; }
            g.terms.push(Term { v: Param::new(1.0), d: datum(i) as f64 / SCALE, hb: SelfBlock::new() });
            i += 1;
        }
        nested.groups.push(g);
    }
    let (nested_cost, nested_cg) = costs(&mut nested);
    let nested_err = ulps_f64(nested_cost, exact_sum());
    assert!(nested_err * 2.0 < flat_err,
        "nested {nested_err} ulps against flat {flat_err}: the scopes did not cut the drift");
    assert!(ulps_f64(nested_cg, nested_cost) <= 1.0, "assembly cost {nested_cg} != sweep cost {nested_cost}");
}

#[test]
fn kahan_f64_is_exact_to_a_few_ulps() {
    let mut m = KahanF64 { terms: terms() };
    let (c, cg) = costs(&mut m);
    let e = ulps_f64(c, exact_sum());
    assert!(e <= 4.0, "kahan f64 error {e} ulps");
    assert!(ulps_f64(cg, c) <= 1.0, "assembly cost {cg} != sweep cost {c}");
}
