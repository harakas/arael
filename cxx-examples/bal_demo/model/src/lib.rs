//! The bundle-adjustment model of examples/bal_demo.rs, packaged for the
//! C++ interface: the model and the solver are Rust; loading a BAL file,
//! composing the scene, choosing the route (CPU or GPU) and reporting
//! are ../cxx/main.cpp. The structs and the constraint are the Rust
//! example's, with pub fields for the generated interface.

use arael::model::{CrossBlock, EulerAngleParam, Param, SelfBlock};
use arael::refs::{self, Ref};
use arael::vect::{vect2d, vect3d};

/// A camera: its translation, its world-to-camera rotation, and its
/// intrinsics (focal length and two radial distortion coefficients).
#[arael::model]
#[derive(Default)]
pub struct Camera {
    pub t: Param<vect3d>,
    pub ea: EulerAngleParam<f64>,
    pub intr: Param<vect3d>,
    pub hb: SelfBlock<Camera>,
}

/// A 3D point.
#[arael::model]
#[derive(Default)]
pub struct Point {
    pub pos: Param<vect3d>,
    pub hb: SelfBlock<Point>,
}

/// A pixel observation of a point by a camera: the point rotated and
/// translated into the camera frame, projected (BAL cameras look down
/// -z), distorted by `1 + k1 r^2 + k2 r^4`, scaled by the focal length.
#[arael::model]
#[derive(Default)]
#[arael(constraint(hb, {
    let pc = cam.ea.rotation_matrix() * pt.pos + cam.t;
    let px = -pc.x / pc.z;
    let py = -pc.y / pc.z;
    let r2 = px * px + py * py;
    let d = 1.0 + r2 * (cam.intr.y + cam.intr.z * r2);
    [cam.intr.x * d * px - obs.xy.x,
     cam.intr.x * d * py - obs.xy.y]
}))]
pub struct Obs {
    #[arael(ref = root.cameras)]
    pub cam: Ref<Camera>,
    #[arael(ref = root.points)]
    pub pt: Ref<Point>,
    pub xy: vect2d,
    pub hb: CrossBlock<Camera, Point>,
}

/// The scene: cameras, points and their observations.
#[arael::model]
#[derive(Default)]
#[arael(root)]
pub struct Scene {
    pub cameras: refs::Vec<Camera>,
    pub points: refs::Vec<Point>,
    pub observations: std::vec::Vec<Obs>,
}
