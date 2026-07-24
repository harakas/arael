// Macro robustness: guard rewriting on the AST, passive SelfBlock
// wiring for Arena and direct-composed entities.

use arael::model::{Model, Param, SelfBlock, CrossBlock};
use arael::simple_lm::{self, LmConfig, LmProblem};
use arael::refs::{self, Ref};

// --- Guards: AST rewrite instead of string surgery ---------------------
//
// The old rewrite was `g.replacen("self.", ..., 10)`: an 11th `self.`
// survived verbatim (binding to the generated fn's real `self`, the
// root), and a block expression in the guard was truncated at the brace
// by the attribute parser.

#[arael::model]
#[arael(constraint(hb, guard = self.a + self.b + self.c + self.d + self.e2
    + self.f + self.g + self.h + self.i + self.j + self.k > 0.0, {
    [(g11.x - g11.target) * w25.isigma]
}))]
struct G11 {
    x: Param<f64>,
    target: f64,
    a: f64, b: f64, c: f64, d: f64, e2: f64, f: f64,
    g: f64, h: f64, i: f64, j: f64, k: f64,
    hb: SelfBlock<G11>,
}

#[arael::model]
#[arael(constraint(hb, guard = { let t = self.threshold; self.lvl > t }, {
    [(gblk.x - gblk.target) * w25.isigma]
}))]
struct GBlk {
    x: Param<f64>,
    target: f64,
    threshold: f64,
    lvl: f64,
    hb: SelfBlock<GBlk>,
}

// --- Passive wiring: Arena collection and direct-composed field --------
//
// ArenaPt has params + SelfBlock but NO self-constraint; it participates
// only through the cross constraint below. The passive index wiring
// matched Vec|Deque only, so an Arena-held entity kept u32::MAX indices:
// its Hessian diagonal stayed zero (now caught by the zero-diagonal
// fail-fast in lm_solve,
// previously 20 silent failures per iteration).

#[arael::model]
struct ArenaPt {
    x: Param<f64>,
    hb: SelfBlock<ArenaPt>,
}

#[arael::model]
#[arael(constraint(hb, {
    [(anchor.x - tie.gap - pt.x) * w25.isigma]
}))]
struct Tie {
    #[arael(ref = root.anchors)]
    anchor: Ref<AnchorPt>,
    #[arael(ref = root.pts)]
    pt: Ref<ArenaPt>,
    gap: f64,
    hb: CrossBlock<AnchorPt, ArenaPt>,
}

#[arael::model]
#[arael(constraint(hb, {
    [(anchorpt.x - anchorpt.target) * w25.isigma]
}))]
struct AnchorPt {
    x: Param<f64>,
    target: f64,
    hb: SelfBlock<AnchorPt>,
}

#[arael::model]
#[arael(root)]
struct W25 {
    anchors: refs::Vec<AnchorPt>,
    pts: refs::Arena<ArenaPt>,
    ties: std::vec::Vec<Tie>,
    g11s: refs::Vec<G11>,
    gblks: refs::Vec<GBlk>,
    isigma: f64,
}

fn base_model() -> W25 {
    W25 {
        anchors: refs::Vec::new(),
        pts: refs::Arena::new(),
        ties: std::vec::Vec::new(),
        g11s: refs::Vec::new(),
        gblks: refs::Vec::new(),
        isigma: 1.0,
    }
}

// Direct-composed passive entity in its own root: params + SelfBlock, no
// constraint of its own. The passive index wiring used to cover
// collections only; a direct-composed field kept u32::MAX indices, so
// anything writing through its block (e.g. ExtendedModel runtime
// constraints, as the sketch does) was silently dropped. Wiring is the
// claim here, not solvability -- no constraint touches the param, so a
// solve would (correctly) fail fast on the zero diagonal.
#[arael::model]
struct DirectPt {
    x: Param<f64>,
    hb: SelfBlock<DirectPt>,
}

#[arael::model]
#[arael(root)]
struct WD {
    direct: DirectPt,
}

#[test]
fn direct_composed_passive_entity_is_wired() {
    let mut wd = WD { direct: DirectPt { x: Param::new(1.0), hb: SelfBlock::new() } };
    let mut params = Vec::new();
    wd.serialize64(&mut params);
    assert_eq!(params.len(), 1);
    assert!(wd.direct.hb.is_active(),
        "direct-composed passive SelfBlock must have wired indices after serialize");
}

#[test]
fn guard_with_more_than_ten_self_references() {
    let mut w = base_model();
    let mk = |on: f64| G11 {
        x: Param::new(0.0), target: 5.0,
        a: on, b: on, c: on, d: on, e2: on, f: on,
        g: on, h: on, i: on, j: on, k: on,
        hb: SelfBlock::new(),
    };
    w.g11s.push(mk(1.0));  // guard sum 11 > 0: active
    w.g11s.push(mk(-1.0)); // guard sum -11: inactive
    let mut params = Vec::new();
    w.serialize64(&mut params);
    let cost = w.calc_cost(&params);
    // Only the active instance contributes (5^2); the inactive one's
    // residual must be guarded off. If the 11th `self.` survived the
    // rewrite this would not even compile (or bind to the root).
    assert!((cost - 25.0).abs() < 1e-12, "cost={}", cost);
}

#[test]
fn guard_with_block_expression() {
    let mut w = base_model();
    w.gblks.push(GBlk { x: Param::new(0.0), target: 3.0, threshold: 0.5, lvl: 1.0, hb: SelfBlock::new() });
    w.gblks.push(GBlk { x: Param::new(0.0), target: 3.0, threshold: 0.5, lvl: 0.0, hb: SelfBlock::new() });
    let mut params = Vec::new();
    w.serialize64(&mut params);
    let cost = w.calc_cost(&params);
    assert!((cost - 9.0).abs() < 1e-12, "cost={}", cost);
}

#[test]
fn arena_passive_entity_is_wired() {
    let mut w = base_model();
    w.anchors.push(AnchorPt { x: Param::new(0.0), target: 2.0, hb: SelfBlock::new() });
    let pt = w.pts.push(ArenaPt { x: Param::new(0.0), hb: SelfBlock::new() });
    w.ties.push(Tie { anchor: w.anchors.ref_at(0), pt, gap: 1.5, hb: CrossBlock::new() });
    let mut params = Vec::new();
    w.serialize64(&mut params);
    // Pre-fix, the Arena entity's SelfBlock kept u32::MAX indices: its
    // diagonal stayed zero, which now terminates the solve immediately
    // (B20). Post-fix the solve converges: anchor -> 2, pt -> 0.5.
    let result = simple_lm::solve(&params, &mut w, &LmConfig::default()).unwrap();
    assert!(result.end_cost < 1e-12, "cost={} iters={}", result.end_cost, result.iterations);
    assert!(result.iterations > 0, "solve must actually iterate");
    w.deserialize64(&result.x);
    let a = w.anchors[0].x.value;
    assert!((a - 2.0).abs() < 1e-6, "anchor={}", a);
}
