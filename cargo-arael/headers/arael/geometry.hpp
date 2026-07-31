// arael C++ geometry: pinhole camera. Mirrors arael's src/geometry.rs
// (intrinsics + extrinsics in robot frame, mc2r = camera-to-robot
// rotation), generic over the scalar like the math types: cameraf /
// camerad, with Camera as a legacy alias of cameraf. Header-only,
// C++17.
#pragma once

#include <cstdint>
#include "vect.hpp"
#include "matrix.hpp"

namespace arael {

/// Pinhole camera model with intrinsics and extrinsics.
template<class T>
struct camera {
    // Intrinsics
    T fx, fy, cx, cy;
    uint32_t width, height;
    // Extrinsics: position and orientation in robot frame
    vect3<T> camera_pos;
    /// Rotation from camera frame to robot frame.
    matrix3<T> mc2r;

    /// Project a 3D point in camera frame to 2D pixel coordinates.
    vect2<T> project(vect3<T> p_cam) const {
        return {fx * p_cam.x / p_cam.z + cx, fy * p_cam.y / p_cam.z + cy};
    }

    /// Unproject a pixel to a unit direction in camera frame.
    vect3<T> unproject(vect2<T> px) const {
        vect3<T> dir{(px.x - cx) / fx, (px.y - cy) / fy, T(1)};
        return dir * (T(1) / dir.norm());
    }

    /// Transform a world point into this camera's frame given robot pose.
    vect3<T> world_to_camera(vect3<T> p_world, vect3<T> robot_pos, matrix3<T> mr2w) const {
        vect3<T> p_robot = mr2w.transpose() * (p_world - robot_pos);
        return mc2r.transpose() * (p_robot - camera_pos);
    }

    /// Unproject pixel to unit direction in robot frame.
    vect3<T> unproject_to_robot(vect2<T> px) const {
        return mc2r * unproject(px);
    }

    /// Angular size of one pixel at position px, in radians per axis.
    /// At image center this equals 1/fx; at edges it is smaller.
    vect2<T> pixel_angular_size(vect2<T> px) const {
        T dx = px.x - cx;
        T dy = px.y - cy;
        return {fx / (dx * dx + fx * fx), fy / (dy * dy + fy * fy)};
    }

    /// Check if a pixel coordinate is within the image bounds.
    bool is_visible(vect2<T> px) const {
        return px.x >= T(0) && px.x < T(width)
            && px.y >= T(0) && px.y < T(height);
    }
};

using cameraf = camera<float>;
using camerad = camera<double>;
/// Legacy alias of cameraf; new code names the precision.
using Camera = cameraf;

} // namespace arael
