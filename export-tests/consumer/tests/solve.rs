use arael::model::{CrossBlock, Model, Param, SelfBlock};
use arael::refs;
use arael::simple_lm::{LmConfig, LmProblem, LmStatus};
use arael::utils::Float;
use arael::vect::{vect2, vect3, vect3d};
use export_consumer::{Bias, BiasLink, World32, World64};
use export_models::{Beacon, Dir, Kind, Spring};

fn target(k: usize) -> vect3d {
    vect3d::new(0.3, 1.0 + k as f64, -0.4 * k as f64).unit()
}

macro_rules! build_world {
    ($world:ident, $t:ty) => {{
        let mut w = $world {
            beacons: refs::Vec::new(),
            biases: refs::Vec::new(),
            springs: std::vec::Vec::new(),
            links: std::vec::Vec::new(),
        };
        let mut brefs = std::vec::Vec::new();
        for k in 0..3 {
            let m = target(k);
            brefs.push(w.beacons.push(Beacon {
                pos: Param::new(vect2::new((k as f64 + 0.4) as $t, -0.3 as $t)),
                dir: Dir::new(vect3::new(1.0 as $t, 0.1 as $t, 0.0 as $t)),
                prior: vect2::new(k as f64 as $t, 0.0 as $t),
                target: vect3::new(m.x as $t, m.y as $t, m.z as $t),
                kind: if k == 0 { Kind::Fixed } else { Kind::Free },
                hb: SelfBlock::new(),
            }));
        }
        for k in 0..2 {
            w.springs.push(Spring {
                a: brefs[k],
                b: brefs[k + 1],
                rest: 0.8 as $t,
                w: 2.0 as $t,
                hb: CrossBlock::new(),
            });
        }
        let bias = w.biases.push(Bias { v: Param::new(0.2 as $t), hb: SelfBlock::new() });
        w.links.push(BiasLink { bk: brefs[1], bl: bias, m: 1.1 as $t, hb: CrossBlock::new() });
        w
    }};
}

/// Imported params fold correctly: the component's 2 DOF inside the
/// imported entity, at both precisions.
#[test]
fn imported_param_counts() {
    assert_eq!(<Dir<f64> as Model>::PARAM_COUNT, 2);
    assert_eq!(<Beacon<f64> as Model>::PARAM_COUNT, 4);
    assert_eq!(<Beacon<f32> as Model>::PARAM_COUNT, 4);
    assert_eq!(<Spring<f64> as Model>::PARAM_COUNT, 0);
}

/// The mixed local + imported model solves at both precisions and lands
/// on the same optimum.
#[test]
fn imported_models_solve_in_both_roots() {
    let mut w64 = build_world!(World64, f64);
    let r64 = w64.solve_sparse(&LmConfig { max_iters: 100, ..Default::default() }).unwrap();
    assert!(matches!(r64.status, LmStatus::Converged), "{:?}", r64.status);

    let mut w32 = build_world!(World32, f32);
    let r32 = w32.solve_sparse(&LmConfig { max_iters: 100, ..Default::default() }).unwrap();
    assert!((r32.end_cost as f64 - r64.end_cost).abs() < 1e-3,
        "f32 cost {} vs f64 {}", r32.end_cost, r64.end_cost);

    // Imported constraints reached the solve: directions moved to their
    // targets (remote data lives on the imported entity itself).
    for (k, (b64, b32)) in w64.beacons.iter().zip(w32.beacons.iter()).enumerate() {
        let m = target(k);
        assert!((b64.dir.unit - m).norm() < 1e-4, "beacon {k} f64 dir off");
        let d32 = vect3d::new(b32.dir.unit.x as f64, b32.dir.unit.y as f64,
            b32.dir.unit.z as f64);
        assert!((d32 - b64.dir.unit).norm() < 2e-3, "beacon {k} f32 vs f64");
        let dx = b64.pos.value.x - b32.pos.value.x as f64;
        let dy = b64.pos.value.y - b32.pos.value.y as f64;
        assert!(dx.abs() < 2e-3 && dy.abs() < 2e-3, "beacon {k} pos disagree");
    }
    // The cross-crate BiasLink pulled the bias off its weak zero prior.
    let bias64 = w64.biases.iter().next().unwrap().v.value;
    assert!(bias64.abs() > 1e-3, "bias unmoved: {}", bias64);
}
