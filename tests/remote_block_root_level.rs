// A remote-block constraint struct living in a ROOT-LEVEL collection.
//
// The remote sweep iterates "parent collection -> frines"; when the
// parent is the root itself the sweep is a single loop over the root
// collection. This used to fall through the emission's location gate:
// the constraint's residuals were silently dropped from cost, gradient
// and Hessian, and the wired-but-unwritten target block surfaced as a
// zero Hessian diagonal at solve time.

use arael::model::{Param, SelfBlock};
use arael::refs::{self, Ref};
use arael::simple_lm::{LmConfig, LmProblem, LmStatus};
use arael::vect::vect2d;

#[arael::model]
struct Beacon {
    pos: Param<vect2d>,
    hb: SelfBlock<Beacon>,
}

/// Observation of a beacon's position, directly on the root.
#[arael::model]
#[arael(constraint(b.hb, {
    [b.pos.x - obs.m.x, b.pos.y - obs.m.y]
}))]
struct Obs {
    #[arael(ref = root.beacons)]
    b: Ref<Beacon>,
    m: vect2d,
}

#[arael::model]
#[arael(root)]
struct World {
    beacons: refs::Vec<Beacon>,
    obs: std::vec::Vec<Obs>,
}

/// Two observations per beacon pull it to their midpoint; every residual
/// must reach the cost AND the beacon's Hessian block.
#[test]
fn root_level_remote_constraints_are_not_dropped() {
    let mut w = World { beacons: refs::Vec::new(), obs: std::vec::Vec::new() };
    let mut refs_v = std::vec::Vec::new();
    for i in 0..3 {
        refs_v.push(w.beacons.push(Beacon {
            pos: Param::new(vect2d::new(0.0, 0.0)),
            hb: SelfBlock::new(),
        }));
        let c = vect2d::new(i as f64, -(i as f64));
        w.obs.push(Obs { b: refs_v[i], m: vect2d::new(c.x - 0.5, c.y) });
        w.obs.push(Obs { b: refs_v[i], m: vect2d::new(c.x + 0.5, c.y) });
    }
    let r = w.solve_sparse(&LmConfig::default());
    // 6 observations, each contributing (+-0.5, 0) at the optimum.
    assert!(matches!(r.status, LmStatus::Converged), "{:?}", r.status);
    assert!((r.end_cost - 6.0 * 0.25).abs() < 1e-9, "cost {}", r.end_cost);
    for (i, b) in w.beacons.iter().enumerate() {
        let c = vect2d::new(i as f64, -(i as f64));
        assert!((b.pos.value - c).norm() < 1e-6,
            "beacon {i} at ({}, {})", b.pos.value.x, b.pos.value.y);
    }
}
