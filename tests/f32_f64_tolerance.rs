// f32-vs-f64 tolerance test (REVIEW2 section 5): the same model solved in both
// precisions should reach essentially the same optimum. The f32 pipeline is
// exercised standalone elsewhere; this pins the two precisions against each
// other on one model with an explicit error bound.

use arael::model::{Model, Param, SelfBlock, CrossBlock};
use arael::simple_lm::LmConfig;
use arael::vect::{vect2d, vect2f};
use arael::refs::{self, Ref};

// ---- f64 model (a spring chain with an anchored end + weak drift) ----
#[arael::model]
#[arael(constraint(hb, guard = self.is_anchor,
    { [p64.pos.x * chain64.anchor, p64.pos.y * chain64.anchor] }))]
#[arael(constraint(hb, { let d = p64.pos - p64.pos_value; [d.x * chain64.drift, d.y * chain64.drift] }))]
struct P64 { pos: Param<vect2d>, is_anchor: bool, hb: SelfBlock<P64> }
#[arael::model]
#[arael(constraint(hb, { let d = b.pos - a.pos; [(d.norm() - l64.rest) * chain64.spring] }))]
struct L64 {
    #[arael(ref = root.points)] a: Ref<P64>,
    #[arael(ref = root.points)] b: Ref<P64>,
    rest: f64, hb: CrossBlock<P64, P64>,
}
#[arael::model]
#[arael(root)]
struct Chain64 { points: refs::Vec<P64>, links: std::vec::Vec<L64>, anchor: f64, drift: f64, spring: f64 }

// ---- f32 model (identical math, f32 throughout) ----
#[arael::model]
#[arael(constraint(hb, guard = self.is_anchor,
    { [p32.pos.x * chain32.anchor, p32.pos.y * chain32.anchor] }))]
#[arael(constraint(hb, { let d = p32.pos - p32.pos_value; [d.x * chain32.drift, d.y * chain32.drift] }))]
struct P32 { pos: Param<vect2f>, is_anchor: bool, hb: SelfBlock<P32, f32> }
#[arael::model]
#[arael(constraint(hb, { let d = b.pos - a.pos; [(d.norm() - l32.rest) * chain32.spring] }))]
struct L32 {
    #[arael(ref = root.points)] a: Ref<P32>,
    #[arael(ref = root.points)] b: Ref<P32>,
    rest: f32, hb: CrossBlock<P32, P32, f32>,
}
#[arael::model]
#[arael(root, f32)]
struct Chain32 { points: refs::Vec<P32>, links: std::vec::Vec<L32>, anchor: f32, drift: f32, spring: f32 }

const N: usize = 6;
fn init(i: usize) -> (f64, f64) { (i as f64 * 0.5, if i % 2 == 0 { 0.7 } else { -0.7 }) }

fn build64() -> Chain64 {
    let mut c = Chain64 { points: refs::Vec::new(), links: std::vec::Vec::new(),
        anchor: 100.0, drift: 0.01, spring: 1.0 };
    for i in 0..N { let (x, y) = init(i);
        c.points.push(P64 { pos: Param::new(vect2d::new(x, y)), is_anchor: i == 0, hb: SelfBlock::new() }); }
    for i in 1..N { let a = c.points.ref_at(i - 1); let b = c.points.ref_at(i);
        c.links.push(L64 { a, b, rest: 1.0, hb: CrossBlock::new() }); }
    c
}
fn build32() -> Chain32 {
    let mut c = Chain32 { points: refs::Vec::new(), links: std::vec::Vec::new(),
        anchor: 100.0, drift: 0.01, spring: 1.0 };
    for i in 0..N { let (x, y) = init(i);
        c.points.push(P32 { pos: Param::new(vect2f::new(x as f32, y as f32)), is_anchor: i == 0, hb: SelfBlock::new() }); }
    for i in 1..N { let a = c.points.ref_at(i - 1); let b = c.points.ref_at(i);
        c.links.push(L32 { a, b, rest: 1.0, hb: CrossBlock::new() }); }
    c
}

#[test]
fn f32_and_f64_agree_within_tolerance() {
    let mut c64 = build64();
    c64.solve_sparse(&LmConfig { max_iters: 200, ..Default::default() });
    let mut c32 = build32();
    c32.solve_sparse(&LmConfig { max_iters: 200, ..Default::default() });

    for i in 0..N {
        let p64 = c64.points[c64.points.ref_at(i)].pos.value;
        let p32 = c32.points[c32.points.ref_at(i)].pos.value;
        let dx = (p64.x - p32.x as f64).abs();
        let dy = (p64.y - p32.y as f64).abs();
        assert!(dx < 1e-3 && dy < 1e-3,
            "point {i}: f64 ({:.6},{:.6}) vs f32 ({:.6},{:.6}) diff ({dx:.2e},{dy:.2e})",
            p64.x, p64.y, p32.x, p32.y);
    }
}
