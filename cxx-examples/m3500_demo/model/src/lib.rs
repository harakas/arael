//! The 2D pose-graph model of examples/m3500_demo.rs, packaged for the
//! C++ interface: the model and solver are Rust; loading the g2o file,
//! composing the graph, and reporting live in ../main.cpp. See the
//! Rust example for the model walkthrough -- the structs and
//! constraints here are the same, with pub fields and Defaults for the
//! generated interface.

use arael::angle::AngleParam;
use arael::model::{Param, SelfBlock, CrossBlock};
use arael::refs::{self, Ref};
use arael::vect::vect2d;

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

/// One relative SE2 measurement between two poses. wt/wr are the
/// square roots of the (diagonal) information matrix entries -- 1.0 in
/// unweighted mode.
#[arael::model]
#[arael(constraint(hb, {
    let local = b.rot.rotation_matrix.transpose()
        * (a.pos + a.rot.rotation_matrix * edge.delta - b.pos);
    [local.x * edge.wt,
     local.y * edge.wt,
     rad_diff(a.rot.angle + edge.dth, b.rot.angle) * edge.wr]
}))]
#[derive(Default)]
pub struct Edge {
    #[arael(ref = root.poses)]
    pub a: Ref<Pose2>,
    #[arael(ref = root.poses)]
    pub b: Ref<Pose2>,
    pub delta: vect2d,
    pub dth: f64,
    pub wt: f64,
    pub wr: f64,
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
