//! The 2D pose-graph model of examples/m3500_demo.rs, packaged for the
//! C++ interface: the model and solver are Rust; loading the g2o file,
//! composing the graph, and reporting live in ../main.cpp. See the
//! Rust example for the model walkthrough -- the structs and
//! constraints here are the same, with pub fields and Defaults for the
//! generated interface.

use arael::angle::AngleParam;
use arael::model::{Param, SelfBlock, CrossBlock};
use arael::refs::{self, Ref};
use arael::vect::{vect2d, vect3d};

/// A 2D pose. `rot` is an `AngleParam`: the heading is optimized directly,
/// and its rotation matrix is built from cached sin/cos so the edge
/// constraint reads them instead of recomputing trig per observation.
#[arael::model]
#[derive(Default)]
pub struct Pose2 {
    pub pos: Param<vect2d>,
    pub rot: AngleParam<f64>,
    pub hb: SelfBlock<Pose2>,
}

/// The gauge anchor: ONE optional prior on the root instead of prior
/// fields carried by every pose. It pulls the referenced pose toward a
/// fixed value with unit weight -- without it, any rigid motion of the
/// whole graph would leave the cost unchanged and the Hessian
/// singular. The residuals write into the pose's own block (`p.hb`);
/// when the Option is None the constraint simply does not exist.
#[arael::model]
#[arael(constraint(p.hb, {
    [p.pos.x - prior.pos.x,
     p.pos.y - prior.pos.y,
     p.rot.angle - prior.th]
}))]
#[derive(Default)]
pub struct Prior {
    #[arael(ref = root.poses)]
    pub p: Ref<Pose2>,
    pub pos: vect2d,
    pub th: f64,
}

/// One relative SE2 measurement between two poses, in the g2o
/// convention: the residual is expressed in pose a's (the
/// measurement's) frame. s0/s1/s2 are the rows of the sqrt-information
/// factor diag(w) * R^T from the information matrix eigendecomposition
/// (info = R diag(w)^2 R^T) -- identity rows in unweighted mode.
#[arael::model]
#[arael(constraint(hb, {
    let local = a.rot.rotation_matrix.transpose() * (b.pos - a.pos)
        - edge.delta;
    let rr = rad_diff(b.rot.angle, a.rot.angle + edge.dth);
    [edge.s0.x * local.x + edge.s0.y * local.y + edge.s0.z * rr,
     edge.s1.x * local.x + edge.s1.y * local.y + edge.s1.z * rr,
     edge.s2.x * local.x + edge.s2.y * local.y + edge.s2.z * rr]
}))]
#[derive(Default)]
pub struct Edge {
    #[arael(ref = root.poses)]
    pub a: Ref<Pose2>,
    #[arael(ref = root.poses)]
    pub b: Ref<Pose2>,
    pub delta: vect2d,
    pub dth: f64,
    pub s0: vect3d,
    pub s1: vect3d,
    pub s2: vect3d,
    pub hb: CrossBlock<Pose2, Pose2>,
}

#[arael::model]
#[arael(root)]
#[derive(Default)]
pub struct Graph {
    pub poses: refs::Vec<Pose2>,
    pub edges: std::vec::Vec<Edge>,
    pub prior: Option<Prior>,
}
