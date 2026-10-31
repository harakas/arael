//! The root keywords `grad_plain`, `grad_kahan` and `grad_f64`. One
//! parameter shared by 200k rows receives one gradient term from each row;
//! the stash's sum is compared against the exact value, formed in integers:
//! the widened and the compensated stashes land within an ulp or a few,
//! the plain one far away, on both block kinds and at both precisions.

use arael::model::{Param, SelfBlock, VarSelfBlock};
use arael::refs::{self, Ref};
use arael::simple_lm::{CooMatrix, LmProblem, RootProblem};
use arael::utils::Float;

/// The shared parameter, held at 1.
#[arael::model]
struct Shared<T: Float> {
    v: Param<T>,
    hb: SelfBlock<Shared<T>, T>,
}

/// The same with the block stored at its live size.
#[arael::model]
struct SharedVar<T: Float> {
    v: Param<T>,
    hb: VarSelfBlock<SharedVar<T>, T>,
}

/// One row `v * d`: with `v` at 1 the residual is the datum `d` and the
/// row's gradient term for `v` is `2 d^2`.
#[arael::model]
#[arael(constraint(s.hb, { [s.v * row.d] }))]
struct Row<T: Float> {
    #[arael(ref = root.shared)]
    s: Ref<Shared<T>>,
    d: T,
}

#[arael::model]
#[arael(constraint(s.hb, { [s.v * rowvar.d] }))]
struct RowVar<T: Float> {
    #[arael(ref = root.shared)]
    s: Ref<SharedVar<T>>,
    d: T,
}

#[arael::model]
#[arael(root, f32, grad_plain)]
struct PlainF32 { shared: refs::Vec<Shared<f32>>, rows: std::vec::Vec<Row<f32>> }

#[arael::model]
#[arael(root, f32, grad_kahan)]
struct KahanF32 { shared: refs::Vec<Shared<f32>>, rows: std::vec::Vec<Row<f32>> }

#[arael::model]
#[arael(root, f32, grad_f64)]
struct WideF32 { shared: refs::Vec<Shared<f32>>, rows: std::vec::Vec<Row<f32>> }

#[arael::model]
#[arael(root, f32, grad_f64, grad_kahan)]
struct WideKahanF32 { shared: refs::Vec<Shared<f32>>, rows: std::vec::Vec<Row<f32>> }

#[arael::model]
#[arael(root, f32)]
struct PlainVarF32 { shared: refs::Vec<SharedVar<f32>>, rows: std::vec::Vec<RowVar<f32>> }

#[arael::model]
#[arael(root, f32, grad_f64)]
struct WideVarF32 { shared: refs::Vec<SharedVar<f32>>, rows: std::vec::Vec<RowVar<f32>> }

#[arael::model]
#[arael(root, f32, grad_kahan)]
struct KahanVarF32 { shared: refs::Vec<SharedVar<f32>>, rows: std::vec::Vec<RowVar<f32>> }

#[arael::model]
#[arael(root)]
struct PlainF64 { shared: refs::Vec<Shared<f64>>, rows: std::vec::Vec<Row<f64>> }

#[arael::model]
#[arael(root, grad_kahan)]
struct KahanF64 { shared: refs::Vec<Shared<f64>>, rows: std::vec::Vec<Row<f64>> }

/// `grad_f64` on an f64 root is the plain sum.
#[arael::model]
#[arael(root, grad_f64)]
struct WideF64 { shared: refs::Vec<Shared<f64>>, rows: std::vec::Vec<Row<f64>> }

const N: usize = 200_000;
const SCALE: f64 = 16_777_216.0; // 2^24

/// Integers below 2^24 spread over the whole range, so the squares span
/// magnitudes and every add rounds.
fn datum(i: usize) -> u64 {
    ((i as u64).wrapping_mul(2_654_435_761) % (1 << 24)) + 1
}

/// The exact gradient `2 * sum d^2`, the squares exact over 2^48.
fn exact_grad() -> f64 {
    let s: u128 = (0..N).map(|i| { let m = datum(i) as u128; m * m }).sum();
    2.0 * s as f64 / (SCALE * SCALE)
}

/// The exact sum of the terms as f32 forms them, `2 * fl(d * d)`, which is
/// what an f32 stash receives.
fn exact_grad_of_f32_terms() -> f64 {
    let s: u128 = (0..N).map(|i| {
        let d = (datum(i) as f64 / SCALE) as f32;
        let sq = d * d;
        (sq as f64 * SCALE * SCALE) as u128
    }).sum();
    2.0 * s as f64 / (SCALE * SCALE)
}

fn ulps_f32(a: f64, b: f64) -> f64 { (a - b).abs() / (b as f32).abs() as f64 * (1.0 / f32::EPSILON as f64) }
fn ulps_f64(a: f64, b: f64) -> f64 { (a - b).abs() / b.abs() / f64::EPSILON }

macro_rules! build {
    ($root:ident, $shared:ident, $row:ident, $t:ty, $hb:expr) => {{
        let mut shared = refs::Vec::new();
        shared.push($shared { v: Param::new(1.0 as $t), hb: $hb });
        let mut rows = std::vec::Vec::with_capacity(N);
        for i in 0..N {
            rows.push($row { s: shared.ref_at(0), d: (datum(i) as f64 / SCALE) as $t });
        }
        $root { shared, rows }
    }};
}

/// The gradient of the one parameter, from the block assembly.
fn gradient<P: LmProblem<T> + RootProblem<T>, T: Float>(m: &mut P) -> f64 {
    let mut x = Vec::new();
    RootProblem::serialize(m, &mut x);
    assert_eq!(x.len(), 1);
    let mut g = vec![T::zero(); 1];
    let mut coo = CooMatrix::new(1);
    m.calc_grad_hessian_sparse(&x, &mut g, &mut coo);
    g[0].to_f64().unwrap()
}

#[test]
fn plain_f32_drifts() {
    let mut m = build!(PlainF32, Shared, Row, f32, SelfBlock::new());
    let e = ulps_f32(gradient(&mut m), exact_grad_of_f32_terms());
    assert!(e > 16.0, "plain f32 gradient error {e} ulps, expected the drift of a 200k-term sum");
}

#[test]
fn kahan_f32_is_exact_to_a_few_ulps() {
    let mut m = build!(KahanF32, Shared, Row, f32, SelfBlock::new());
    let e = ulps_f32(gradient(&mut m), exact_grad_of_f32_terms());
    assert!(e <= 4.0, "kahan f32 gradient error {e} ulps");
}

#[test]
fn f64_stash_on_f32_root() {
    let mut m = build!(WideF32, Shared, Row, f32, SelfBlock::new());
    // The terms are formed in f64 from the f32 data, so they are exact,
    // and the sum is rounded once to f32.
    let e = ulps_f32(gradient(&mut m), exact_grad());
    assert!(e <= 1.0, "f64-stashed f32 gradient error {e} ulps");
}

#[test]
fn f64_and_kahan_combine() {
    let mut m = build!(WideKahanF32, Shared, Row, f32, SelfBlock::new());
    let e = ulps_f32(gradient(&mut m), exact_grad());
    assert!(e <= 1.0, "f64 and kahan f32 gradient error {e} ulps");
}

#[test]
fn the_var_block_follows_the_keywords() {
    let mut plain = build!(PlainVarF32, SharedVar, RowVar, f32, VarSelfBlock::new());
    let e = ulps_f32(gradient(&mut plain), exact_grad_of_f32_terms());
    assert!(e > 16.0, "plain var f32 gradient error {e} ulps, expected drift");
    let mut wide = build!(WideVarF32, SharedVar, RowVar, f32, VarSelfBlock::new());
    let e = ulps_f32(gradient(&mut wide), exact_grad());
    assert!(e <= 1.0, "f64-stashed var f32 gradient error {e} ulps");
    let mut kahan = build!(KahanVarF32, SharedVar, RowVar, f32, VarSelfBlock::new());
    let e = ulps_f32(gradient(&mut kahan), exact_grad_of_f32_terms());
    assert!(e <= 4.0, "kahan var f32 gradient error {e} ulps");
}

#[test]
fn plain_f64_drifts_and_kahan_f64_does_not() {
    let mut plain = build!(PlainF64, Shared, Row, f64, SelfBlock::new());
    let e = ulps_f64(gradient(&mut plain), exact_grad());
    assert!(e > 16.0, "plain f64 gradient error {e} ulps, expected the drift of a 200k-term sum");
    let mut kahan = build!(KahanF64, Shared, Row, f64, SelfBlock::new());
    let e = ulps_f64(gradient(&mut kahan), exact_grad());
    assert!(e <= 4.0, "kahan f64 gradient error {e} ulps");
}

#[test]
fn grad_f64_on_an_f64_root_is_the_plain_sum() {
    let mut plain = build!(PlainF64, Shared, Row, f64, SelfBlock::new());
    let mut wide = build!(WideF64, Shared, Row, f64, SelfBlock::new());
    assert_eq!(gradient(&mut plain), gradient(&mut wide));
}
