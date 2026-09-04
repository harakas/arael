//! A parent-owned CrossBlock is shared by every instance under the
//! parent, so an instance's own ref cannot be one of its sides.

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
#[arael(constraint([hb_point_cam, parent.hb_shared], parent = image, {
    [image.cam.f * point.p - obs.m]
}))]
struct Obs {
    #[arael(ref = root.points)] point: Ref<Point>,
    m: f64,
    hb_point_cam: CrossBlock<Point, Cam>,
}

#[arael::model]
struct Image {
    #[arael(ref = root.cams)] cam: Ref<Cam>,
    obs: std::vec::Vec<Obs>,
    hb_shared: CrossBlock<Point, Cam>,
}

#[arael::model]
#[arael(root)]
struct World {
    cams: refs::Vec<Cam>,
    points: refs::Vec<Point>,
    images: std::vec::Vec<Image>,
}

fn main() {}
