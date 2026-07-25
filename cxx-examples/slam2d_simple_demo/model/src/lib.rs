//! The 2D SLAM model of examples/slam2d_simple_demo.rs, packaged for
//! the C++ interface: the model and solver are Rust; composing the
//! problem, reading results, and plotting live in ../main.cpp. See
//! that Rust example for the model walkthrough -- the structs and
//! constraints here are the same, with pub fields and Defaults for
//! the generated interface.

use arael::prelude::*;

/// Robot pose (x, y, gamma) plus the odometry measured since the
/// previous pose, in that pose's local frame.
#[arael::model]
#[derive(Default)]
pub struct Pose {
    pub pos: Param<vect2f>,
    pub gamma: Param<f32>,
    pub delta_pos: vect2f,
    pub delta_gamma: f32,
    pub delta_pos_isigma: f32,
    pub delta_gamma_isigma: f32,
    pub hb_pose: SelfBlock<Pose, f32>,
}

/// Odometry constraint between two consecutive poses.
#[arael::model]
#[arael(constraint(hb, {
    let local = matrix2sym::rotation(prev.gamma).transpose() * (cur.pos - prev.pos);
    [(local.x - cur.delta_pos.x) * cur.delta_pos_isigma,
     (local.y - cur.delta_pos.y) * cur.delta_pos_isigma,
     rad_diff(cur.gamma - prev.gamma, cur.delta_gamma) * cur.delta_gamma_isigma]
}))]
#[derive(Default)]
pub struct PosePair {
    #[arael(ref = root.poses)]
    pub prev: Ref<Pose>,
    #[arael(ref = root.poses)]
    pub cur: Ref<Pose>,
    pub hb: CrossBlock<Pose, Pose, f32>,
}

/// A building corner and its bearing sightings.
#[arael::model]
#[derive(Default)]
pub struct Landmark {
    pub pos: Param<vect2f>,
    pub frines: std::vec::Vec<Frine>,
    pub hb: SelfBlock<Landmark, f32>,
}

/// One bearing sighting linking a landmark to a pose.
#[arael::model]
#[arael(constraint(hb, parent = lm, {
    let world_angle = pose.gamma + frine.bearing;
    let aligned = matrix2sym::rotation(world_angle).transpose() * (lm.pos - pose.pos);
    [atan2(aligned.y, aligned.x) * frine.isigma]
}))]
#[derive(Default)]
pub struct Frine {
    #[arael(ref = root.poses)]
    pub pose: Ref<Pose>,
    pub bearing: f32,
    pub isigma: f32,
    pub hb: CrossBlock<Landmark, Pose, f32>,
}

#[arael::model]
#[arael(root, f32)]
#[derive(Default)]
pub struct Path {
    pub poses: refs::Deque<Pose>,
    pub pose_pairs: std::vec::Vec<PosePair>,
    pub landmarks: refs::Arena<Landmark>,
}
