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
use arael::matrix::matrixd;
use arael::transform::{ScaledTransformParam, TransformParam};
use arael::unitvec::UnitVecParam;
use arael::vect::{vect2d, vect3d, vectd};

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

/// An entity with its own parameter pulled to its target. Its
/// hand-written `Default` (unit weight) is what a keyword-less
/// `push()` must reproduce through every interface.
#[arael::model]
#[arael(constraint(hb, {
    [(n.v - n.t) * n.w]
}))]
pub struct N {
    pub v: Param<f64>,
    pub t: f64,
    pub w: f64,
    pub hb: SelfBlock<N>,
}

impl Default for N {
    fn default() -> Self {
        N { v: Param::default(), t: 0.0, w: 1.0, hb: SelfBlock::new() }
    }
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

/// N-dof entity: a const-generic vect parameter with per-component
/// priors plus a matrix-projected residual -- exercises the
/// vect<T, N> / matrix<T, R, C> export surface.
#[arael::model]
#[arael(constraint(hb, {
    let d = vn.v - vn.t;
    let p = vn.h * d;
    [d[0] * vn.wp, d[1] * vn.wp, d[2] * vn.wp, d[3] * vn.wp,
     p[0] * vn.w, p[1] * vn.w]
}))]
#[derive(Default)]
pub struct Vn {
    pub v: Param<vectd<4>>,
    pub t: vectd<4>,
    pub h: matrixd<2, 4>,
    pub wp: f64,
    pub w: f64,
    pub hb: SelfBlock<Vn>,
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

/// The pose builtins -- a rigid transform, its scaled sister and a unit
/// direction -- with an `i32` and an `f32` beside them: the leaf kinds
/// the keyword and column calls must carry, and the fields the C++ and
/// Python transform views wrap. Unconstrained; the parity suite builds
/// and reads it back without solving.
#[arael::model]
#[derive(Default)]
pub struct Frame {
    pub pose: TransformParam<f64>,
    pub st: ScaledTransformParam<f64>,
    pub dir: UnitVecParam<f64>,
    pub anchor: vect3d,
    pub tag: i32,
    pub scale: f32,
    pub hb: SelfBlock<Frame>,
}

/// An entity with no scalar surface of its own -- a user component
/// and its block only -- so a keyword-less push has nothing to name.
#[arael::model]
#[derive(Default)]
pub struct Wrap {
    pub gain: Gain,
    pub hb: SelfBlock<Wrap>,
}

#[arael::model]
#[arael(root, jacobian)]
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
    pub vns: refs::Vec<Vn>,
    pub wraps: std::vec::Vec<Wrap>,
    pub frames: refs::Vec<Frame>,
    pub hb: SelfBlock<Fit>,
}
