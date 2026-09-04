// arael C++ math: quatern<T>. Mirrors arael's src/quatern.rs: scalar
// part FIRST ({t, v}), same euler convention and gimbal-lock handling
// as matrix3, Shepperd's method for matrix conversion. Header-only,
// C++17.
#pragma once

#include <limits>
#include "vect.hpp"
#include "matrix.hpp"

namespace arael {

/// Quaternion: scalar part t, vector part v. Unit quaternions
/// represent 3D rotations.
template<class T>
struct quatern {
    T t;
    vect3<T> v;

    static quatern identity() { return {T(1), {T(0), T(0), T(0)}}; }

    T dot(quatern q) const { return t * q.t + v * q.v; }
    T norm() const { return std::sqrt(dot(*this)); }
    quatern unit() const { return *this * (T(1) / norm()); }
    quatern conj() const { return {t, -v}; }
    bool is_finite() const { return std::isfinite(t) && v.is_finite(); }
    template<class K> quatern<K> cast() const { return {K(t), v.template cast<K>()}; }
    /// `Eigen::Quaternion<T>` with the same rotation (`w()` is `t`).
    template<class Dummy = void> auto to_eigen() const { return arael_to_eigen(*this); }   // provided by <arael/eigen.hpp>
    /// From an Eigen quaternion, through `w() x() y() z()`.
    template<class Q> static quatern from_eigen(const Q& q) {
        return {T(q.w()), {T(q.x()), T(q.y()), T(q.z())}};
    }
    bool similar(quatern other) const {
        return std::abs(t - other.t)
                < T(10) * (std::abs(t) + std::abs(other.t)
                    + std::numeric_limits<T>::epsilon())
                    * std::numeric_limits<T>::epsilon()
            && v.similar(other.v);
    }

    /// Rotate a vector by this unit quaternion: q * v * q'.
    vect3<T> rotate(vect3<T> p) const {
        return (*this * quatern{T(0), p} * conj()).v;
    }

    matrix3<T> rotation_matrix() const {
        T x2 = v.x * v.x, y2 = v.y * v.y, z2 = v.z * v.z;
        return matrix3<T>::from_rows(
            {T(1) - T(2) * (y2 + z2), T(2) * (v.x * v.y - v.z * t), T(2) * (v.x * v.z + v.y * t)},
            {T(2) * (v.x * v.y + v.z * t), T(1) - T(2) * (x2 + z2), T(2) * (v.y * v.z - v.x * t)},
            {T(2) * (v.x * v.z - v.y * t), T(2) * (v.y * v.z + v.x * t), T(1) - T(2) * (x2 + y2)});
    }

    /// (unit axis, angle in radians, normalized to [-pi, pi]); identity
    /// yields the x axis, angle 0.
    void get_axis_angle(vect3<T>& axis, T& angle) const {
        angle = detail::rad2rad(T(2) * detail::safe_acos(t));
        T s2 = T(1) - t * t;
        if (s2 > std::numeric_limits<T>::epsilon() * std::numeric_limits<T>::epsilon()) {
            axis = v * (T(1) / std::sqrt(s2));
        } else {
            axis = {T(1), T(0), T(0)};
            angle = T(0);
        }
    }

    /// Euler angles (x=roll, y=pitch, z=yaw). At and near gimbal lock
    /// only roll -+ yaw is determined; roll = 0 convention. Matches
    /// matrix3::get_euler_angles for the same rotation.
    vect3<T> get_euler_angles() const {
        T pitch = detail::safe_asin(T(2) * (t * v.y - v.z * v.x));
        T roll_num = T(2) * (t * v.x + v.y * v.z);
        T roll_den = T(1) - T(2) * (v.x * v.x + v.y * v.y);
        T cp2 = roll_num * roll_num + roll_den * roll_den;
        if (cp2 > std::numeric_limits<T>::epsilon()) {
            return {std::atan2(roll_num, roll_den), pitch,
                    std::atan2(T(2) * (t * v.z + v.x * v.y),
                               T(1) - T(2) * (v.y * v.y + v.z * v.z))};
        }
        return {T(0), pitch,
                std::atan2(T(2) * (t * v.z - v.x * v.y),
                           T(1) - T(2) * (v.x * v.x + v.z * v.z))};
    }

    static quatern from_euler_angles(vect3<T> ea) {
        vect3<T> ha = ea * T(0.5);
        T shax = std::sin(ha.x), chax = std::cos(ha.x);
        T shay = std::sin(ha.y), chay = std::cos(ha.y);
        T shaz = std::sin(ha.z), chaz = std::cos(ha.z);
        return {chax * chay * chaz + shax * shay * shaz,
                {shax * chay * chaz - chax * shay * shaz,
                 chax * shay * chaz + shax * chay * shaz,
                 chax * chay * shaz - shax * shay * chaz}};
    }

    static quatern from_axis_angle(vect3<T> normal, T angle) {
        T half_angle = T(0.5) * angle;
        return {std::cos(half_angle), normal * std::sin(half_angle)};
    }

    /// Exp map of so(3): v = axis * angle. Taylor branch below
    /// |v|^2 = 1e-8, so no singularity at v = 0.
    static quatern from_rotation_vector(vect3<T> v) {
        T s = v.x * v.x + v.y * v.y + v.z * v.z;
        T theta = std::sqrt(s);
        T half = T(0.5) * theta;
        T qt, scale;
        if (s >= T(1e-8)) {
            qt = std::cos(half);
            scale = std::sin(half) / theta;
        } else {
            qt = T(1) - s * T(0.125) + s * s * T(1.0 / 384.0);
            scale = T(0.5) - s * T(1.0 / 48.0);
        }
        return {qt, v * scale};
    }

    /// First-order retraction normalize(1, v/2); branch-free, agrees
    /// with the exp map to first order.
    static quatern from_rotation_vector_small(vect3<T> v) {
        return quatern{T(1), v * T(0.5)}.unit();
    }

    /// Shepperd's method: largest squared component first, no branch
    /// divides by a small number.
    static quatern from_rotation_matrix(matrix3<T> m) {
        T tr = m[0].x + m[1].y + m[2].z;
        quatern q;
        if (tr > T(0)) {
            T s = std::sqrt(tr + T(1)) * T(2);
            q = {T(0.25) * s,
                 {(m[2].y - m[1].z) / s, (m[0].z - m[2].x) / s, (m[1].x - m[0].y) / s}};
        } else if (m[0].x > m[1].y && m[0].x > m[2].z) {
            T s = std::sqrt(T(1) + m[0].x - m[1].y - m[2].z) * T(2);
            q = {(m[2].y - m[1].z) / s,
                 {T(0.25) * s, (m[0].y + m[1].x) / s, (m[0].z + m[2].x) / s}};
        } else if (m[1].y > m[2].z) {
            T s = std::sqrt(T(1) + m[1].y - m[0].x - m[2].z) * T(2);
            q = {(m[0].z - m[2].x) / s,
                 {(m[0].y + m[1].x) / s, T(0.25) * s, (m[1].z + m[2].y) / s}};
        } else {
            T s = std::sqrt(T(1) + m[2].z - m[0].x - m[1].y) * T(2);
            q = {(m[1].x - m[0].y) / s,
                 {(m[0].z + m[2].x) / s, (m[1].z + m[2].y) / s, T(0.25) * s}};
        }
        return q.unit();
    }

    /// Scale the rotation angle by f.
    quatern pow(T f) const {
        vect3<T> axis;
        T angle;
        get_axis_angle(axis, angle);
        return from_axis_angle(axis, f * angle);
    }

    /// Pure quaternion with vector part axis * angle.
    quatern log() const {
        vect3<T> axis;
        T angle;
        get_axis_angle(axis, angle);
        return {T(0), axis * angle};
    }

    /// Inverse of log().
    quatern exp() const {
        T angle = v.norm();
        if (angle < std::numeric_limits<T>::epsilon()) {
            return identity();
        }
        vect3<T> axis = v * (T(1) / angle);
        return from_axis_angle(axis, angle);
    }

    /// Shortest arc rotating unit vector `from` onto unit vector `to`.
    static quatern from_two_vectors(vect3<T> from, vect3<T> to) {
        vect3<T> mid = (from + to) * T(0.5);
        T mid_len2 = mid * mid;
        if (mid_len2 < std::numeric_limits<T>::epsilon()) {
            return from_axis_angle(from.across(), detail::pi<T>());
        }
        mid = mid * (T(1) / std::sqrt(mid_len2));
        return {mid * to, mid % to};
    }

    /// Spherical linear interpolation; f=0 -> from, f=1 -> to.
    static quatern slerp(quatern from, quatern to, T f) {
        return from * (from.conj() * to).pow(f);
    }

    quatern operator+(quatern q) const { return {t + q.t, v + q.v}; }
    quatern operator-(quatern q) const { return {t - q.t, v - q.v}; }
    quatern operator-() const { return {-t, -v}; }
    quatern operator*(T s) const { return {t * s, v * s}; }
    quatern operator*(quatern q) const {
        return {t * q.t - v * q.v, q.v * t + v * q.t + (v % q.v)};
    }
};

template<class T> inline quatern<T> operator*(T s, quatern<T> q) { return q * s; }

using quaternf = quatern<float>;
using quaternd = quatern<double>;

static_assert(sizeof(quaternf) == 16 && sizeof(quaternd) == 32, "quatern layout");

} // namespace arael
