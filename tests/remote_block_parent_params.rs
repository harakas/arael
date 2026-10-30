// A remote-block constraint whose containing parent has params of its own.
// The block holds the target's params only, but the derivative list used
// to start with the parent's, so every write landed that many positions
// too far and the first assembly panicked (out of bounds). Covers a parent
// that is the root and one nested below it; reading the parent's params in
// such a body is a compile error.

use arael::model::{Param, SelfBlock};
use arael::refs::{self, Ref};
use arael::simple_lm::{LmConfig, LmProblem};
use arael::vect::vect2d;

#[arael::model]
struct Beacon {
    pos: Param<vect2d>,
    hb: SelfBlock<Beacon>,
}

// ---------------------------------------------------------- root parent

#[arael::model]
#[arael(constraint(b.hb, { [b.pos.x - obs.m.x, b.pos.y - obs.m.y] }))]
struct Obs {
    #[arael(ref = root.beacons)]
    b: Ref<Beacon>,
    m: vect2d,
}

#[arael::model]
#[arael(constraint(root.hb, { [g.v - world.shift] }))]
struct G { v: f64 }

#[arael::model]
#[arael(root)]
struct World {
    shift: Param<f64>,
    beacons: refs::Vec<Beacon>,
    obs: std::vec::Vec<Obs>,
    gs: std::vec::Vec<G>,
    hb: SelfBlock<World>,
}

#[test]
fn root_parent_with_params() {
    let mut w = World {
        shift: Param::new(0.0),
        beacons: refs::Vec::new(),
        obs: vec![],
        gs: vec![G { v: 1.0 }],
        hb: SelfBlock::new(),
    };
    let b = w.beacons.push(Beacon { pos: Param::new(vect2d::new(0.0, 0.0)), hb: SelfBlock::new() });
    w.obs.push(Obs { b, m: vect2d::new(1.0, 2.0) });
    let r = w.solve_sparse(&LmConfig::conservative()).unwrap();
    assert!(r.status.is_success(), "{:?}", r.status);
    assert!((w.beacons[0].pos.value - vect2d::new(1.0, 2.0)).norm() < 1e-9);
    assert!((w.shift.value - 1.0).abs() < 1e-9);
}

// -------------------------------------------------------- nested parent

#[arael::model]
#[arael(constraint(b.hb, parent = st, { [b.pos.x - sobs.m.x, b.pos.y - sobs.m.y] }))]
struct SObs {
    #[arael(ref = root.beacons)]
    b: Ref<Beacon>,
    m: vect2d,
}

#[arael::model]
#[arael(constraint(hb, { [station.bias - station.prior] }))]
struct Station {
    bias: Param<f64>,
    prior: f64,
    obs: std::vec::Vec<SObs>,
    hb: SelfBlock<Station>,
}

#[arael::model]
#[arael(root)]
struct Field {
    beacons: refs::Vec<Beacon>,
    stations: std::vec::Vec<Station>,
}

#[test]
fn nested_parent_with_params() {
    let mut f = Field { beacons: refs::Vec::new(), stations: vec![] };
    let b = f.beacons.push(Beacon { pos: Param::new(vect2d::new(0.0, 0.0)), hb: SelfBlock::new() });
    f.stations.push(Station {
        bias: Param::new(0.0),
        prior: 0.5,
        obs: vec![SObs { b, m: vect2d::new(1.0, 2.0) }],
        hb: SelfBlock::new(),
    });
    let r = f.solve_sparse(&LmConfig::conservative()).unwrap();
    assert!(r.status.is_success(), "{:?}", r.status);
    assert!((f.beacons[0].pos.value - vect2d::new(1.0, 2.0)).norm() < 1e-9);
    assert!((f.stations[0].bias.value - 0.5).abs() < 1e-9);
}

/// Regenerate the snapshot with:
///     TRYBUILD=overwrite cargo test --test remote_block_parent_params
#[test]
fn reading_the_parents_params_is_rejected() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/remote_block_parent_params/reads_parent_param.rs");
}
