//! `parent.parent = <name>` may not reuse a field name of the constraint
//! struct.

use arael::model::{CrossBlock, Param, SelfBlock};
use arael::refs::{self, Ref};

#[arael::model]
#[arael(constraint(hb, { [cam.f - cam.pf] }))]
struct Cam {
    f: Param<f64>,
    pf: f64,
    hb: SelfBlock<Cam>,
}

#[arael::model]
#[arael(constraint(hb, { [point.p - point.pp] }))]
struct Point {
    p: Param<f64>,
    pp: f64,
    hb: SelfBlock<Point>,
}

#[arael::model]
#[arael(constraint(hb, { [pose.t - pose.pt] }))]
struct Pose {
    t: Param<f64>,
    pt: f64,
    images: std::vec::Vec<Image>,
    hb: SelfBlock<Pose>,
}

#[arael::model]
struct Image {
    #[arael(ref = root.cams)] cam: Ref<Cam>,
    obs: std::vec::Vec<Obs>,
    hb_pose_cam: CrossBlock<Pose, Cam>,
}

#[arael::model]
#[arael(constraint([hb_point_pose, hb_point_cam, parent.hb_pose_cam],
    parent = image, parent.parent = m, {
    [image.cam.f * (point.p - m.t) - obs.m]
}))]
struct Obs {
    #[arael(ref = root.points)] point: Ref<Point>,
    m: f64,
    hb_point_pose: CrossBlock<Point, Pose>,
    hb_point_cam: CrossBlock<Point, Cam>,
}

#[arael::model]
#[arael(root)]
struct World {
    poses: refs::Vec<Pose>,
    cams: refs::Vec<Cam>,
    points: refs::Vec<Point>,
}

fn main() {}
