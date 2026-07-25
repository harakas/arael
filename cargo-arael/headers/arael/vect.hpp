// arael C++ math: vect2<T>, vect3<T>. Mirrors arael's src/vect.rs --
// same fields, same operations, same conventions: `*` between vectors
// is the DOT product, `%` is the CROSS product, `vect * scalar` scales.
// Header-only, C++17.
#pragma once

#include <cmath>
#include <cstddef>
#include <limits>

namespace arael {

template<class T> struct matrix3;

namespace detail {

// asin/acos with the argument clamped to [-1, 1]: float noise can push
// a valid rotation's entry just past the domain, where the raw calls
// return NaN.
template<class T> inline T safe_asin(T v) {
    if (v > T(1)) v = T(1);
    if (v < T(-1)) v = T(-1);
    return std::asin(v);
}
template<class T> inline T safe_acos(T v) {
    if (v > T(1)) v = T(1);
    if (v < T(-1)) v = T(-1);
    return std::acos(v);
}

template<class T> inline constexpr T pi() { return T(3.14159265358979323846264338327950288L); }

// Normalize radians to [-pi, pi].
template<class T> inline T rad2similar(T x, T similar) {
    T delta = x - similar;
    T two_pi = T(2) * pi<T>();
    return x - two_pi * std::floor(delta / two_pi + T(0.5));
}
template<class T> inline T rad2rad(T x) {
    if (x < -pi<T>() || x > pi<T>()) return rad2similar(x, T(0));
    return x;
}

} // namespace detail

/// 2D vector with x, y components.
template<class T>
struct vect2 {
    T x, y;

    T square() const { return x * x + y * y; }
    T norm() const { return std::sqrt(square()); }
    vect2 unit() const { T r = T(1) / norm(); return {x * r, y * r}; }
    /// Perpendicular vector (90-degree counterclockwise rotation).
    vect2 across() const { return {-y, x}; }
    /// 2D cross product (z of the 3D cross).
    T cross(vect2 rhs) const { return x * rhs.y - y * rhs.x; }
    T dot(vect2 rhs) const { return x * rhs.x + y * rhs.y; }
    bool is_finite() const { return std::isfinite(x) && std::isfinite(y); }
    vect2 deg2rad() const { T k = detail::pi<T>() / T(180); return {x * k, y * k}; }
    vect2 rad2deg() const { T k = T(180) / detail::pi<T>(); return {x * k, y * k}; }
    template<class K> vect2<K> cast() const { return {K(x), K(y)}; }
    bool similar(vect2 other) const {
        return (*this - other).norm()
            < T(10) * (norm() + other.norm() + std::numeric_limits<T>::epsilon())
                * std::numeric_limits<T>::epsilon();
    }

    T& operator[](std::size_t i) { return (&x)[i]; }
    const T& operator[](std::size_t i) const { return (&x)[i]; }
    vect2 operator+(vect2 rhs) const { return {x + rhs.x, y + rhs.y}; }
    vect2 operator-(vect2 rhs) const { return {x - rhs.x, y - rhs.y}; }
    vect2 operator-() const { return {-x, -y}; }
    /// Dot product.
    T operator*(vect2 rhs) const { return x * rhs.x + y * rhs.y; }
    vect2 operator*(T s) const { return {x * s, y * s}; }
};

/// 3D vector with x, y, z components.
template<class T>
struct vect3 {
    T x, y, z;

    T square() const { return x * x + y * y + z * z; }
    T norm() const { return std::sqrt(square()); }
    vect3 unit() const { T r = T(1) / norm(); return {x * r, y * r, z * r}; }
    /// Some unit vector perpendicular to this one.
    vect3 across() const {
        if (std::abs(y) < std::abs(x)) {
            return vect3{-z, T(0), x}.unit();
        }
        return vect3{T(0), z, -y}.unit();
    }
    T dot(vect3 rhs) const { return x * rhs.x + y * rhs.y + z * rhs.z; }
    vect3 cross(vect3 rhs) const {
        return {y * rhs.z - z * rhs.y, z * rhs.x - x * rhs.z, x * rhs.y - y * rhs.x};
    }
    bool is_finite() const { return std::isfinite(x) && std::isfinite(y) && std::isfinite(z); }
    vect3 deg2rad() const { T k = detail::pi<T>() / T(180); return {x * k, y * k, z * k}; }
    vect3 rad2deg() const { T k = T(180) / detail::pi<T>(); return {x * k, y * k, z * k}; }
    template<class K> vect3<K> cast() const { return {K(x), K(y), K(z)}; }
    bool similar(vect3 other) const {
        return (*this - other).norm()
            < T(10) * (norm() + other.norm() + std::numeric_limits<T>::epsilon())
                * std::numeric_limits<T>::epsilon();
    }
    /// Element-wise (sin, cos).
    void sincos(vect3& s, vect3& c) const {
        s = {std::sin(x), std::sin(y), std::sin(z)};
        c = {std::cos(x), std::cos(y), std::cos(z)};
    }
    /// Rotation matrix of this vector read as euler angles (x=roll,
    /// y=pitch, z=yaw; R = R(z)*R(y)*R(x)). Defined in matrix.hpp.
    matrix3<T> rotation_matrix() const;

    T& operator[](std::size_t i) { return (&x)[i]; }
    const T& operator[](std::size_t i) const { return (&x)[i]; }
    vect3 operator+(vect3 rhs) const { return {x + rhs.x, y + rhs.y, z + rhs.z}; }
    vect3 operator-(vect3 rhs) const { return {x - rhs.x, y - rhs.y, z - rhs.z}; }
    vect3 operator-() const { return {-x, -y, -z}; }
    /// Dot product.
    T operator*(vect3 rhs) const { return x * rhs.x + y * rhs.y + z * rhs.z; }
    vect3 operator*(T s) const { return {x * s, y * s, z * s}; }
    /// Cross product.
    vect3 operator%(vect3 rhs) const { return cross(rhs); }
};

template<class T> inline vect2<T> operator*(T s, vect2<T> v) { return v * s; }
template<class T> inline vect3<T> operator*(T s, vect3<T> v) { return v * s; }

using vect2f = vect2<float>;
using vect2d = vect2<double>;
using vect3f = vect3<float>;
using vect3d = vect3<double>;

static_assert(sizeof(vect2f) == 8 && sizeof(vect2d) == 16, "vect2 layout");
static_assert(sizeof(vect3f) == 12 && sizeof(vect3d) == 24, "vect3 layout");

} // namespace arael
