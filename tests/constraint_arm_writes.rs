// A `match` in a residual body whose arms touch different parameters:
// the store writes are emitted once per arm over the parameters the arm
// touches (docs/dev/ARM_WRITES.md). The cost, gradient and Hessian of the
// dispatching constraint equal those of the constraint written for each
// arm directly, including under a `branch` guard around the match and
// with a loss selected by another field.

use arael::model::{Param, SelfBlock, CrossBlock};
use arael::refs::{self, Ref};
use arael::simple_lm::{LmProblem, RootProblem};

#[arael::model]
struct Pt {
    x: Param<f64>,
    hb: SelfBlock<Pt>,
}

/// A camera with three parameters; a model uses one or two of them.
#[arael::model]
struct Cam {
    model: u32,
    f: Param<f64>,
    k: Param<f64>,
    c: Param<f64>,
    hb: SelfBlock<Cam>,
}

// --- The dispatching observation: model 0 reads f, model 1 f and k,
// any other model c. ---

#[arael::model]
#[arael(constraint(hb, {
    [match cam.model {
        0 => pt.x * cam.f - obs.m,
        1 => pt.x * cam.f * (1.0 + cam.k * pt.x * pt.x) - obs.m,
        _ => pt.x + cam.c - obs.m,
    }]
}))]
struct Obs {
    #[arael(ref = root.pts)]
    pt: Ref<Pt>,
    #[arael(ref = root.cams)]
    cam: Ref<Cam>,
    m: f64,
    hb: CrossBlock<Pt, Cam>,
}

// --- Each arm as its own constraint. ---

#[arael::model]
#[arael(constraint(hb, { [pt.x * cam.f - obs0.m] }))]
struct Obs0 {
    #[arael(ref = root.pts)]
    pt: Ref<Pt>,
    #[arael(ref = root.cams)]
    cam: Ref<Cam>,
    m: f64,
    hb: CrossBlock<Pt, Cam>,
}

#[arael::model]
#[arael(constraint(hb, { [pt.x * cam.f * (1.0 + cam.k * pt.x * pt.x) - obs1.m] }))]
struct Obs1 {
    #[arael(ref = root.pts)]
    pt: Ref<Pt>,
    #[arael(ref = root.cams)]
    cam: Ref<Cam>,
    m: f64,
    hb: CrossBlock<Pt, Cam>,
}

#[arael::model]
#[arael(constraint(hb, { [pt.x + cam.c - obs2.m] }))]
struct Obs2 {
    #[arael(ref = root.pts)]
    pt: Ref<Pt>,
    #[arael(ref = root.cams)]
    cam: Ref<Cam>,
    m: f64,
    hb: CrossBlock<Pt, Cam>,
}

// --- Without a default arm. ---

#[arael::model]
#[arael(constraint(hb, {
    [match cam.model {
        0 => pt.x * cam.f - obsn.m,
        1 => pt.x * cam.f * (1.0 + cam.k * pt.x * pt.x) - obsn.m,
    }]
}))]
struct ObsN {
    #[arael(ref = root.pts)]
    pt: Ref<Pt>,
    #[arael(ref = root.cams)]
    cam: Ref<Cam>,
    m: f64,
    hb: CrossBlock<Pt, Cam>,
}

// --- The match under a `branch` guard (a depth check), and beside a
// loss picked by another field. ---

#[arael::model]
#[arael(constraint(hb, {
    [branch(pt.x, match cam.model {
        0 => pt.x * cam.f - obsg.m,
        _ => pt.x + cam.c - obsg.m,
    }, 0.0)]
}))]
struct ObsG {
    #[arael(ref = root.pts)]
    pt: Ref<Pt>,
    #[arael(ref = root.cams)]
    cam: Ref<Cam>,
    m: f64,
    hb: CrossBlock<Pt, Cam>,
}

#[arael::model]
#[arael(constraint(hb, { [branch(pt.x, pt.x * cam.f - obsg0.m, 0.0)] }))]
struct ObsG0 {
    #[arael(ref = root.pts)]
    pt: Ref<Pt>,
    #[arael(ref = root.cams)]
    cam: Ref<Cam>,
    m: f64,
    hb: CrossBlock<Pt, Cam>,
}

#[arael::model]
#[arael(constraint(hb, loss = |s| match obsl.kind {
    0 => s,
    _ => loss_cauchy(s, obsl.k2),
}, {
    [match cam.model {
        0 => pt.x * cam.f - obsl.m,
        _ => pt.x + cam.c - obsl.m,
    }]
}))]
struct ObsL {
    #[arael(ref = root.pts)]
    pt: Ref<Pt>,
    #[arael(ref = root.cams)]
    cam: Ref<Cam>,
    m: f64,
    k2: f64,
    kind: u32,
    hb: CrossBlock<Pt, Cam>,
}

#[arael::model]
#[arael(constraint(hb, loss = |s| loss_cauchy(s, obsl2.k2), { [pt.x + cam.c - obsl2.m] }))]
struct ObsL2 {
    #[arael(ref = root.pts)]
    pt: Ref<Pt>,
    #[arael(ref = root.cams)]
    cam: Ref<Cam>,
    m: f64,
    k2: f64,
    hb: CrossBlock<Pt, Cam>,
}

#[arael::model]
#[arael(root)]
struct W {
    pts: refs::Vec<Pt>,
    cams: refs::Vec<Cam>,
    obs: std::vec::Vec<Obs>,
    obs0: std::vec::Vec<Obs0>,
    obs1: std::vec::Vec<Obs1>,
    obs2: std::vec::Vec<Obs2>,
    obsn: std::vec::Vec<ObsN>,
    obsg: std::vec::Vec<ObsG>,
    obsg0: std::vec::Vec<ObsG0>,
    obsl: std::vec::Vec<ObsL>,
    obsl2: std::vec::Vec<ObsL2>,
}

/// Two points and two cameras of the given models.
fn world(models: [u32; 2]) -> W {
    let mut w = W {
        pts: refs::Vec::new(), cams: refs::Vec::new(),
        obs: std::vec::Vec::new(), obs0: std::vec::Vec::new(),
        obs1: std::vec::Vec::new(), obs2: std::vec::Vec::new(),
        obsn: std::vec::Vec::new(),
        obsg: std::vec::Vec::new(), obsg0: std::vec::Vec::new(),
        obsl: std::vec::Vec::new(), obsl2: std::vec::Vec::new(),
    };
    for x in [0.7, -1.3] {
        w.pts.push(Pt { x: Param::new(x), hb: SelfBlock::new() });
    }
    for (i, model) in models.into_iter().enumerate() {
        w.cams.push(Cam {
            model,
            f: Param::new(1.5 + 0.25 * i as f64),
            k: Param::new(-0.1 * (i + 1) as f64),
            c: Param::new(0.4 - 0.3 * i as f64),
            hb: SelfBlock::new(),
        });
    }
    w
}

/// Cost, gradient and dense Hessian at the model's current values.
fn passes(w: &mut W) -> (f64, Vec<f64>, Vec<f64>) {
    let mut params = Vec::new();
    w.serialize(&mut params);
    let cost = w.calc_cost(&params);
    let n = params.len();
    let mut grad = vec![0.0; n];
    let mut hess = vec![0.0; n * n];
    let cost2 = w.calc_grad_hessian_dense(&params, &mut grad, &mut hess);
    assert_eq!(cost, cost2);
    (cost, grad, hess)
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * (1.0 + a.abs().max(b.abs()))
}

fn assert_same(what: &str, a: (f64, Vec<f64>, Vec<f64>), b: (f64, Vec<f64>, Vec<f64>)) {
    assert!(close(a.0, b.0), "{what}: cost {} vs {}", a.0, b.0);
    assert_eq!(a.1.len(), b.1.len());
    for (i, (x, y)) in a.1.iter().zip(&b.1).enumerate() {
        assert!(close(*x, *y), "{what}: grad[{i}] {x} vs {y}");
    }
    for (i, (x, y)) in a.2.iter().zip(&b.2).enumerate() {
        assert!(close(*x, *y), "{what}: hess[{i}] {x} vs {y}");
    }
}

/// Every point observed by every camera, with a measurement off the
/// prediction.
const OBS: [(usize, usize, f64); 4] = [(0, 0, 0.9), (0, 1, -0.2), (1, 0, 1.1), (1, 1, 0.3)];

#[test]
fn each_arm_matches_its_own_constraint() {
    for model in 0..3u32 {
        let mut d = world([model, model]);
        let mut r = world([model, model]);
        for (p, c, m) in OBS {
            d.obs.push(Obs { pt: d.pts.ref_at(p), cam: d.cams.ref_at(c), m, hb: CrossBlock::new() });
            match model {
                0 => r.obs0.push(Obs0 { pt: r.pts.ref_at(p), cam: r.cams.ref_at(c), m, hb: CrossBlock::new() }),
                1 => r.obs1.push(Obs1 { pt: r.pts.ref_at(p), cam: r.cams.ref_at(c), m, hb: CrossBlock::new() }),
                _ => r.obs2.push(Obs2 { pt: r.pts.ref_at(p), cam: r.cams.ref_at(c), m, hb: CrossBlock::new() }),
            }
        }
        let (cost, grad, hess) = passes(&mut d);
        assert!(cost > 0.0);
        assert!(grad.iter().any(|g| *g != 0.0));
        assert_same(&format!("model {model}"), (cost, grad, hess), passes(&mut r));
    }
}

#[test]
fn mixed_models_in_one_solve() {
    let mut d = world([1, 2]);
    let mut r = world([1, 2]);
    for (p, c, m) in OBS {
        d.obs.push(Obs { pt: d.pts.ref_at(p), cam: d.cams.ref_at(c), m, hb: CrossBlock::new() });
        if c == 0 {
            r.obs1.push(Obs1 { pt: r.pts.ref_at(p), cam: r.cams.ref_at(c), m, hb: CrossBlock::new() });
        } else {
            r.obs2.push(Obs2 { pt: r.pts.ref_at(p), cam: r.cams.ref_at(c), m, hb: CrossBlock::new() });
        }
    }
    assert_same("mixed", passes(&mut d), passes(&mut r));
}

#[test]
fn under_a_branch_guard() {
    // The first point is on the guard's live side, the second off it.
    let mut d = world([0, 0]);
    let mut r = world([0, 0]);
    for (p, c, m) in OBS {
        d.obsg.push(ObsG { pt: d.pts.ref_at(p), cam: d.cams.ref_at(c), m, hb: CrossBlock::new() });
        r.obsg0.push(ObsG0 { pt: r.pts.ref_at(p), cam: r.cams.ref_at(c), m, hb: CrossBlock::new() });
    }
    let (cost, grad, hess) = passes(&mut d);
    assert!(cost > 0.0);
    assert_same("guarded", (cost, grad, hess), passes(&mut r));
}

#[test]
fn beside_a_loss_picked_by_another_field() {
    // Model 2 cameras under the Cauchy kernel, one inlier and one outlier.
    let mut d = world([2, 2]);
    let mut r = world([2, 2]);
    for (p, c, m) in [(0, 0, 0.9), (1, 1, 30.0)] {
        d.obsl.push(ObsL {
            pt: d.pts.ref_at(p), cam: d.cams.ref_at(c), m, k2: 1.0, kind: 1, hb: CrossBlock::new(),
        });
        r.obsl2.push(ObsL2 {
            pt: r.pts.ref_at(p), cam: r.cams.ref_at(c), m, k2: 1.0, hb: CrossBlock::new(),
        });
    }
    assert_same("loss", passes(&mut d), passes(&mut r));
}

#[test]
fn without_a_default_arm() {
    let mut d = world([1, 1]);
    let mut r = world([1, 1]);
    for (p, c, m) in OBS {
        d.obsn.push(ObsN { pt: d.pts.ref_at(p), cam: d.cams.ref_at(c), m, hb: CrossBlock::new() });
        r.obs1.push(Obs1 { pt: r.pts.ref_at(p), cam: r.cams.ref_at(c), m, hb: CrossBlock::new() });
    }
    assert_same("no default", passes(&mut d), passes(&mut r));
}

#[test]
#[should_panic(expected = "select index")]
fn a_model_no_arm_covers_panics() {
    let mut d = world([0, 0]);
    d.obsn.push(ObsN { pt: d.pts.ref_at(0), cam: d.cams.ref_at(0), m: 0.5, hb: CrossBlock::new() });
    d.cams[0].model = 7;
    passes(&mut d);
}
