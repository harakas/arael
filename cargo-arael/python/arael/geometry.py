# arael Python geometry: pinhole camera. Mirrors arael's
# src/geometry.rs (intrinsics + extrinsics in robot frame, mc2r =
# camera-to-robot rotation), generic over the scalar like the math
# types: cameraf / camerad, with Camera as a legacy alias of cameraf.

from .math import matrix3d, matrix3f, vect2d, vect2f, vect3d, vect3f


def _make_camera(name, vec2, vec3, mat3):
    class cam:
        """Pinhole camera model with intrinsics and extrinsics."""

        def __init__(self, fx, fy, cx, cy, width, height, camera_pos, mc2r):
            self.fx = float(fx)
            self.fy = float(fy)
            self.cx = float(cx)
            self.cy = float(cy)
            self.width = int(width)
            self.height = int(height)
            self.camera_pos = (camera_pos if isinstance(camera_pos, vec3)
                               else vec3(camera_pos))
            self.mc2r = mc2r if isinstance(mc2r, mat3) else mat3(mc2r)

        def project(self, p_cam):
            """3D point in camera frame -> 2D pixel coordinates."""
            return vec2(self.fx * p_cam.x / p_cam.z + self.cx,
                        self.fy * p_cam.y / p_cam.z + self.cy)

        def unproject(self, px):
            """Pixel -> unit direction in camera frame."""
            d = vec3((px[0] - self.cx) / self.fx,
                     (px[1] - self.cy) / self.fy, 1.0)
            return d * (1.0 / d.norm())

        def world_to_camera(self, p_world, robot_pos, mr2w):
            """World point -> this camera's frame, given the robot pose."""
            p_robot = mr2w.transpose() * (p_world - robot_pos)
            return self.mc2r.transpose() * (p_robot - self.camera_pos)

        def unproject_to_robot(self, px):
            """Pixel -> unit direction in robot frame."""
            return self.mc2r * self.unproject(px)

        def pixel_angular_size(self, px):
            """Angular size of one pixel at px, radians per axis."""
            dx = px[0] - self.cx
            dy = px[1] - self.cy
            return vec2(self.fx / (dx * dx + self.fx * self.fx),
                        self.fy / (dy * dy + self.fy * self.fy))

        def is_visible(self, px):
            """Whether a pixel coordinate is inside the image bounds."""
            return (0.0 <= px[0] < float(self.width)
                    and 0.0 <= px[1] < float(self.height))

    cam.__name__ = name
    cam.__qualname__ = name
    return cam


cameraf = _make_camera("cameraf", vect2f, vect3f, matrix3f)
camerad = _make_camera("camerad", vect2d, vect3d, matrix3d)
# Legacy alias of cameraf; new code names the precision.
Camera = cameraf
