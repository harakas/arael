//! Geometric primitives for camera models and projections.

use crate::matrix::matrix3;
use crate::utils::Float;
use crate::vect::{vect2, vect3};

/// Pinhole camera model with intrinsics and extrinsics, generic over
/// the scalar like the math types; use [`cameraf`] / [`camerad`].
#[allow(non_camel_case_types)]
pub struct camera<T: Float> {
    // Intrinsics
    pub fx: T,
    pub fy: T,
    pub cx: T,
    pub cy: T,
    pub width: u32,
    pub height: u32,
    // Extrinsics: position and orientation in robot frame
    pub camera_pos: vect3<T>,
    /// Rotation from **c**amera frame to **r**obot frame (`mc2r` = M_camera_to_robot).
    pub mc2r: matrix3<T>,
}

/// f32 camera.
#[allow(non_camel_case_types)]
pub type cameraf = camera<f32>;
/// f64 camera.
#[allow(non_camel_case_types)]
pub type camerad = camera<f64>;
/// Legacy alias of [`cameraf`]; new code names the precision.
pub type Camera = cameraf;

impl<T: Float> camera<T> {
    /// Project a 3D point in camera frame to 2D pixel coordinates.
    pub fn project(&self, p_cam: vect3<T>) -> vect2<T> {
        vect2::new(
            self.fx * p_cam.x / p_cam.z + self.cx,
            self.fy * p_cam.y / p_cam.z + self.cy,
        )
    }

    /// Unproject a pixel to a unit direction in camera frame.
    pub fn unproject(&self, px: vect2<T>) -> vect3<T> {
        let dir = vect3::new(
            (px.x - self.cx) / self.fx,
            (px.y - self.cy) / self.fy,
            T::one(),
        );
        dir * (T::one() / dir.norm())
    }

    /// Transform a world point into this camera's frame given robot pose.
    pub fn world_to_camera(&self, p_world: vect3<T>, robot_pos: vect3<T>, mr2w: matrix3<T>) -> vect3<T> {
        let mw2r = mr2w.transpose();
        let mr2c = self.mc2r.transpose();
        let p_robot = mw2r * (p_world - robot_pos);
        mr2c * (p_robot - self.camera_pos)
    }

    /// Unproject pixel to unit direction in robot frame.
    pub fn unproject_to_robot(&self, px: vect2<T>) -> vect3<T> {
        let dir_cam = self.unproject(px);
        self.mc2r * dir_cam
    }

    /// Angular size of one pixel at position `px`, in radians per axis.
    ///
    /// In a pinhole camera, pixel angular size varies across the image:
    ///   size_x = fx / ((px.x - cx)^2 + fx^2)
    ///   size_y = fy / ((py.y - cy)^2 + fy^2)
    /// At image center this equals 1/fx; at edges it is smaller.
    pub fn pixel_angular_size(&self, px: vect2<T>) -> vect2<T> {
        let dx = px.x - self.cx;
        let dy = px.y - self.cy;
        vect2::new(
            self.fx / (dx * dx + self.fx * self.fx),
            self.fy / (dy * dy + self.fy * self.fy),
        )
    }

    /// Check if a pixel coordinate is within the image bounds.
    pub fn is_visible(&self, px: vect2<T>) -> bool {
        px.x >= T::zero() && px.x < T::from(self.width).unwrap()
            && px.y >= T::zero() && px.y < T::from(self.height).unwrap()
    }
}
