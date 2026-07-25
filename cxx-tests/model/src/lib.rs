//! The cxx-tests fixture model: root params behind `root.hb`
//! observations in a std::vec::Vec, own-param entities in a refs::Vec,
//! a 3D pose chain in a Deque with cross-block ties, an Arena with
//! removals, a nested sub-model with an Option entity, math-typed data
//! and params, a fixed euler rotation param, and opaque fields. The
//! parity test builds the same problem from C++ and from Rust and
//! compares the solves exactly.

use arael::model::{CrossBlock, Param, SelfBlock, SimpleEulerAngleParam};
use arael::refs::{self, Ref};
use arael::vect::{vect2d, vect3d};

/// A data-only observation of the root's line: y ~ m * x + c.
#[arael::model]
#[arael(constraint(root.hb, {
    [obs.y - (fit.m * obs.x + fit.c)]
}))]
#[derive(Default)]
pub struct Obs {
    pub x: f64,
    pub y: f64,
}

/// An entity with its own parameter pulled to its target.
#[arael::model]
#[arael(constraint(hb, {
    [(n.v - n.t) * n.w]
}))]
#[derive(Default)]
pub struct N {
    pub v: Param<f64>,
    pub t: f64,
    pub w: f64,
    pub hb: SelfBlock<N>,
}

/// A pose with an optimized position, a FIXED rotation (storage only:
/// `optimize` cleared from the interface), and a prior target.
#[arael::model]
#[arael(constraint(hb, {
    [(pose.pos.x - pose.target.x) * 2.0,
     (pose.pos.y - pose.target.y) * 2.0,
     (pose.pos.z - pose.target.z) * 2.0]
}))]
#[derive(Default)]
pub struct Pose {
    pub ea: SimpleEulerAngleParam<f64>,
    pub pos: Param<vect3d>,
    pub target: vect3d,
    pub info: Info,
    pub hb: SelfBlock<Pose>,
}

/// Nested sub-model: an Option entity plus opaque data.
#[arael::model]
#[derive(Default)]
pub struct Info {
    pub gps: Option<GpsObs>,
    pub note: String,
}

/// Data-only optional observation.
#[arael::model]
#[derive(Default)]
pub struct GpsObs {
    pub pos: vect3d,
    pub isigma: f32,
}

/// Relative-position tie between two poses.
#[arael::model]
#[arael(constraint(hb, {
    [(b.pos.x - a.pos.x - tie.d.x) * tie.w,
     (b.pos.y - a.pos.y - tie.d.y) * tie.w,
     (b.pos.z - a.pos.z - tie.d.z) * tie.w]
}))]
#[derive(Default)]
pub struct Tie {
    #[arael(ref = root.poses)]
    pub a: Ref<Pose>,
    #[arael(ref = root.poses)]
    pub b: Ref<Pose>,
    pub d: vect3d,
    pub w: f64,
    pub hb: CrossBlock<Pose, Pose>,
}

#[arael::model]
#[arael(root)]
#[derive(Default)]
pub struct Fit {
    pub m: Param<f64>,
    pub c: Param<f64>,
    pub cal: vect2d,
    pub tag: String,
    pub obs: std::vec::Vec<Obs>,
    pub items: refs::Vec<N>,
    pub poses: refs::Deque<Pose>,
    pub ties: std::vec::Vec<Tie>,
    pub marks: refs::Arena<N>,
    pub hb: SelfBlock<Fit>,
}
