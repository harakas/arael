//! Geometric primitives for camera models and projections.

use crate::vect::{vect2f, vect3f};
use crate::matrix::matrix3f;

/// Pinhole camera model with intrinsics and extrinsics.
pub struct Camera {
    // Intrinsics
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub width: u32,
    pub height: u32,
    // Extrinsics: position and orientation in robot frame
    pub camera_pos: vect3f,
    /// Rotation from **c**amera frame to **r**obot frame (`mc2r` = M_camera_to_robot).
    pub mc2r: matrix3f,
}

impl Camera {
    /// Project a 3D point in camera frame to 2D pixel coordinates.
    pub fn project(&self, p_cam: vect3f) -> vect2f {
        vect2f::new(
            self.fx * p_cam.x / p_cam.z + self.cx,
            self.fy * p_cam.y / p_cam.z + self.cy,
        )
    }

    /// Unproject a pixel to a unit direction in camera frame.
    pub fn unproject(&self, px: vect2f) -> vect3f {
        let dir = vect3f::new(
            (px.x - self.cx) / self.fx,
            (px.y - self.cy) / self.fy,
            1.0,
        );
        dir * (1.0 / dir.norm())
    }

    /// Transform a world point into this camera's frame given robot pose.
    pub fn world_to_camera(&self, p_world: vect3f, robot_pos: vect3f, mr2w: matrix3f) -> vect3f {
        let mw2r = mr2w.transpose();
        let mr2c = self.mc2r.transpose();
        let p_robot = mw2r * (p_world - robot_pos);
        mr2c * (p_robot - self.camera_pos)
    }

    /// Unproject pixel to unit direction in robot frame.
    pub fn unproject_to_robot(&self, px: vect2f) -> vect3f {
        let dir_cam = self.unproject(px);
        self.mc2r * dir_cam
    }

    /// Angular size of one pixel at position `px`, in radians per axis.
    ///
    /// In a pinhole camera, pixel angular size varies across the image:
    ///   size_x = fx / ((px.x - cx)^2 + fx^2)
    ///   size_y = fy / ((py.y - cy)^2 + fy^2)
    /// At image center this equals 1/fx; at edges it is smaller.
    pub fn pixel_angular_size(&self, px: vect2f) -> vect2f {
        let dx = px.x - self.cx;
        let dy = px.y - self.cy;
        vect2f::new(
            self.fx / (dx * dx + self.fx * self.fx),
            self.fy / (dy * dy + self.fy * self.fy),
        )
    }

    /// Check if a pixel coordinate is within the image bounds.
    pub fn is_visible(&self, px: vect2f) -> bool {
        px.x >= 0.0 && px.x < self.width as f32
            && px.y >= 0.0 && px.y < self.height as f32
    }
}
