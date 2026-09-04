// arael C++ math: vect2<T>, vect3<T>. Mirrors arael's src/vect.rs --
// same fields, same operations, same conventions: `*` between vectors
// is the DOT product, `%` is the CROSS product, `vect * scalar` scales.
// Header-only, C++17.
#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>

#include "assert.hpp"

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

// The shape check behind every `from_eigen`: a fixed-size Eigen
// expression is checked at compile time, a dynamic one at runtime.
template<class X, class = void>
struct has_eigen_shape : std::false_type {};
template<class X>
struct has_eigen_shape<X, std::void_t<decltype(X::RowsAtCompileTime), decltype(X::ColsAtCompileTime)>>
    : std::true_type {};

template<class X, int R, int C>
inline void eigen_shape(const X& x) {
    (void)x;
    if constexpr (has_eigen_shape<X>::value) {
        static_assert((X::RowsAtCompileTime == R || X::RowsAtCompileTime == -1)
                      && (X::ColsAtCompileTime == C || X::ColsAtCompileTime == -1),
                      "from_eigen: the expression has the wrong shape");
        if constexpr (X::RowsAtCompileTime == -1 || X::ColsAtCompileTime == -1) {
            arael_assert_true(x.rows() == R && x.cols() == C);
        }
    } else {
        arael_assert_true(x.rows() == R && x.cols() == C);
    }
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
    /// The Eigen value, `Eigen::Matrix<T, 2, 1>`.
    template<class Dummy = void> auto to_eigen() const { return arael_to_eigen(*this); }   // provided by <arael/eigen.hpp>
    /// An `Eigen::Map` over this storage, readable or assignable in place.
    template<class Dummy = void> auto eigen_map() { return arael_eigen_map(*this); }       // provided by <arael/eigen.hpp>
    template<class Dummy = void> auto eigen_map() const { return arael_eigen_map(*this); } // provided by <arael/eigen.hpp>
    /// From any 2x1 Eigen expression.
    template<class X> static vect2 from_eigen(const X& x) {
        detail::eigen_shape<X, 2, 1>(x);
        return {T(x(0)), T(x(1))};
    }
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
    /// The Eigen value, `Eigen::Matrix<T, 3, 1>`.
    template<class Dummy = void> auto to_eigen() const { return arael_to_eigen(*this); }   // provided by <arael/eigen.hpp>
    /// An `Eigen::Map` over this storage, readable or assignable in place.
    template<class Dummy = void> auto eigen_map() { return arael_eigen_map(*this); }       // provided by <arael/eigen.hpp>
    template<class Dummy = void> auto eigen_map() const { return arael_eigen_map(*this); } // provided by <arael/eigen.hpp>
    /// From any 3x1 Eigen expression.
    template<class X> static vect3 from_eigen(const X& x) {
        detail::eigen_shape<X, 3, 1>(x);
        return {T(x(0)), T(x(1)), T(x(2))};
    }
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

/// N-dimensional vector over T e[N] -- mirrors arael's vect<T, N>
/// (same layout as the FFI mirror structs). `*` between vectors is the
/// DOT product; `vect * scalar` scales; `v[i]` indexes.
template<class T, std::size_t N>
struct vect {
    T e[N];

    T& operator[](std::size_t i) { return e[i]; }
    const T& operator[](std::size_t i) const { return e[i]; }
    static constexpr std::size_t size() { return N; }

    /// The Eigen value, `Eigen::Matrix<T, N, 1>`.
    template<class Dummy = void> auto to_eigen() const { return arael_to_eigen(*this); }   // provided by <arael/eigen.hpp>
    /// An `Eigen::Map` over this storage, readable or assignable in place.
    template<class Dummy = void> auto eigen_map() { return arael_eigen_map(*this); }       // provided by <arael/eigen.hpp>
    template<class Dummy = void> auto eigen_map() const { return arael_eigen_map(*this); } // provided by <arael/eigen.hpp>
    /// From any Nx1 Eigen expression.
    template<class X> static vect from_eigen(const X& x) {
        detail::eigen_shape<X, int(N), 1>(x);
        vect r{};
        for (std::size_t i = 0; i < N; i++) r.e[i] = T(x(i));
        return r;
    }

    vect operator+(const vect& o) const {
        vect r{};
        for (std::size_t i = 0; i < N; i++) r.e[i] = e[i] + o.e[i];
        return r;
    }
    vect operator-(const vect& o) const {
        vect r{};
        for (std::size_t i = 0; i < N; i++) r.e[i] = e[i] - o.e[i];
        return r;
    }
    vect operator-() const {
        vect r{};
        for (std::size_t i = 0; i < N; i++) r.e[i] = -e[i];
        return r;
    }
    vect operator*(T s) const {
        vect r{};
        for (std::size_t i = 0; i < N; i++) r.e[i] = e[i] * s;
        return r;
    }
    vect operator/(T s) const {
        vect r{};
        for (std::size_t i = 0; i < N; i++) r.e[i] = e[i] / s;
        return r;
    }
    T operator*(const vect& o) const {  // dot
        T s = T(0);
        for (std::size_t i = 0; i < N; i++) s += e[i] * o.e[i];
        return s;
    }
    T square() const { return (*this) * (*this); }
    T norm() const { return std::sqrt(square()); }
};

template<class T, std::size_t N>
inline vect<T, N> operator*(T s, const vect<T, N>& v) { return v * s; }

template<std::size_t N> using vectf = vect<float, N>;
template<std::size_t N> using vectd = vect<double, N>;

static_assert(sizeof(vect<double, 4>) == 32, "vect<N> layout");

} // namespace arael
