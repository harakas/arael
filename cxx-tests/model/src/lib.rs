//! The cxx-tests fixture model: root params behind `root.hb`
//! observations in a std::vec::Vec, own-param entities in a refs::Vec,
//! a 3D pose chain in a Deque with cross-block ties, an Arena with
//! removals, a nested sub-model with an Option entity, math-typed data
//! and params, a fixed euler rotation param, and opaque fields. The
//! parity test builds the same problem from C++ and from Rust and
//! compares the solves exactly.

use arael::angle::AngleParam;
use arael::model::{
    Component, CrossBlock, EulerAngleParam, Param, QuaternionParam, SelfBlock,
    SimpleEulerAngleParam,
};
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
/// `optimize` cleared from the interface), a 2D heading (`AngleParam`),
/// and prior targets. The heading residual pins the rotation matrix's
/// first column to `target_dir`, exercising the computed `rotation_matrix`.
#[arael::model]
#[arael(constraint(hb, {
    let hd = pose.heading.rotation_matrix.col(0) - pose.target_dir;
    [(pose.pos.x - pose.target.x) * 2.0,
     (pose.pos.y - pose.target.y) * 2.0,
     (pose.pos.z - pose.target.z) * 2.0,
     hd.x * 1.5,
     hd.y * 1.5]
}))]
#[derive(Default)]
pub struct Pose {
    pub ea: SimpleEulerAngleParam<f64>,
    pub heading: AngleParam<f64>,
    pub pos: Param<vect3d>,
    pub target: vect3d,
    pub target_dir: vect2d,
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

/// A user component: reference + zero-centred delta, the manifold
/// lifecycle shape. The symbolic field carries d(g)/d(d) = 1 through
/// constraints; `g` is the user-facing value (set before, read after).
#[arael::model]
#[arael(component)]
#[derive(Default)]
pub struct Gain {
    pub ref_g: f64,
    pub d: Param<f64>,
    #[arael(symbolic = ref_g + d)]
    pub g: f64,
}

impl Component for Gain {
    fn start(&mut self) {
        self.ref_g = self.g;
        self.d.value = 0.0;
    }
    fn update(&mut self) {
        self.ref_g += self.d.value;
        self.d.value = 0.0;
    }
    fn finish(&mut self) {
        self.g = self.ref_g + self.d.value;
    }
}

/// The compound-parameter zoo: the universal euler-angle param, the
/// quaternion param, and a user component, each pinned to a target
/// (two rotation rows fully determine each rotation).
#[arael::model]
#[arael(constraint(hb, {
    let ru = rig.ea_u.rotation_matrix();
    let rq = rig.q.rotation_matrix();
    let du0 = ru.row(0) - rig.target_u0;
    let du2 = ru.row(2) - rig.target_u2;
    let dq0 = rq.row(0) - rig.target_q0;
    let dq2 = rq.row(2) - rig.target_q2;
    [du0.x, du0.y, du0.z, du2.x, du2.y, du2.z,
     dq0.x, dq0.y, dq0.z, dq2.x, dq2.y, dq2.z,
     (rig.gain.g - rig.target_g) * 2.0]
}))]
#[derive(Default)]
pub struct Rig {
    pub ea_u: EulerAngleParam<f64>,
    pub q: QuaternionParam<f64>,
    pub gain: Gain,
    pub target_u0: vect3d,
    pub target_u2: vect3d,
    pub target_q0: vect3d,
    pub target_q2: vect3d,
    pub target_g: f64,
    pub hb: SelfBlock<Rig>,
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
    pub rigs: refs::Vec<Rig>,
    pub hb: SelfBlock<Fit>,
}
