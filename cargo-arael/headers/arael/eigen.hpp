// arael C++ math: Eigen interop. Include this header, after Eigen is on
// the include path, to use `to_eigen()`, `eigen_map()` and
// `from_eigen()` on the arael value types. Nothing else in the math
// headers depends on Eigen: a translation unit that does not include
// this file compiles without it.
//
//   arael type            to_eigen()                        eigen_map()
//   vect2/3<T>, vect<T,N> Eigen::Matrix<T, N, 1>            Map over x, y, z / e[]
//   matrix2/3, matrix<T,R,C> Eigen::Matrix<T, R, C>         Map (row-major) over rows[]
//   quatern<T>            Eigen::Quaternion<T>, w = t       --
//   transform3<T>         Eigen::Transform<T,3,Isometry>    --
//   scaled_transform3<T>  Eigen::Transform<T,3,Affine>, linear = s R
//
// `from_eigen(x)` takes any Eigen expression of the right shape,
// blocks included; the shape is checked at compile time when fixed and
// at runtime when dynamic. Storage: arael matrices are row-major, so a
// map declares `Eigen::RowMajor`; `to_eigen()` copies into Eigen's
// default column-major value. Quaternions cross through the
// constructor and `w() x() y() z()`, never `coeffs()`, since arael
// stores the scalar part first and Eigen last.
#pragma once

#if defined(__has_include)
#if !__has_include(<Eigen/Core>)
#error "arael/eigen.hpp: Eigen is not on the include path"
#endif
#endif

#include <cstddef>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "math.hpp"

namespace arael {

namespace detail {
// Maps over the row-major arael storage. Eigen insists that a column
// matrix is column-major (and a row matrix row-major), which for one
// column is the same memory either way.
template<class T, int R, int C>
using eigen_map_t = Eigen::Map<Eigen::Matrix<T, R, C, (C == 1 ? Eigen::ColMajor : Eigen::RowMajor)>>;
template<class T, int R, int C>
using eigen_cmap_t = Eigen::Map<const Eigen::Matrix<T, R, C, (C == 1 ? Eigen::ColMajor : Eigen::RowMajor)>>;
} // namespace detail

// --- values ---

template<class T>
inline Eigen::Matrix<T, 2, 1> arael_to_eigen(const vect2<T>& v) { return {v.x, v.y}; }

template<class T>
inline Eigen::Matrix<T, 3, 1> arael_to_eigen(const vect3<T>& v) { return {v.x, v.y, v.z}; }

template<class T, std::size_t N>
inline Eigen::Matrix<T, int(N), 1> arael_to_eigen(const vect<T, N>& v) {
    Eigen::Matrix<T, int(N), 1> r;
    for (std::size_t i = 0; i < N; i++) r(Eigen::Index(i)) = v.e[i];
    return r;
}

template<class T>
inline Eigen::Matrix<T, 2, 2> arael_to_eigen(const matrix2<T>& m) {
    Eigen::Matrix<T, 2, 2> r;
    r << m.rows[0].x, m.rows[0].y,
         m.rows[1].x, m.rows[1].y;
    return r;
}

template<class T>
inline Eigen::Matrix<T, 3, 3> arael_to_eigen(const matrix3<T>& m) {
    Eigen::Matrix<T, 3, 3> r;
    r << m.rows[0].x, m.rows[0].y, m.rows[0].z,
         m.rows[1].x, m.rows[1].y, m.rows[1].z,
         m.rows[2].x, m.rows[2].y, m.rows[2].z;
    return r;
}

template<class T, std::size_t R, std::size_t C>
inline Eigen::Matrix<T, int(R), int(C)> arael_to_eigen(const matrix<T, R, C>& m) {
    Eigen::Matrix<T, int(R), int(C)> r;
    for (std::size_t i = 0; i < R; i++)
        for (std::size_t j = 0; j < C; j++)
            r(Eigen::Index(i), Eigen::Index(j)) = m.rows[i].e[j];
    return r;
}

template<class T>
inline Eigen::Quaternion<T> arael_to_eigen(const quatern<T>& q) {
    return Eigen::Quaternion<T>(q.t, q.v.x, q.v.y, q.v.z);
}

template<class T>
inline Eigen::Transform<T, 3, Eigen::Isometry> arael_to_eigen(const transform3<T>& t) {
    Eigen::Transform<T, 3, Eigen::Isometry> r = Eigen::Transform<T, 3, Eigen::Isometry>::Identity();
    r.linear() = arael_to_eigen(t.rotation_matrix);
    r.translation() = arael_to_eigen(t.translation);
    return r;
}

template<class T>
inline Eigen::Transform<T, 3, Eigen::Affine> arael_to_eigen(const scaled_transform3<T>& t) {
    Eigen::Transform<T, 3, Eigen::Affine> r = Eigen::Transform<T, 3, Eigen::Affine>::Identity();
    r.linear() = arael_to_eigen(t.rotation_matrix) * t.scale;
    r.translation() = arael_to_eigen(t.translation);
    return r;
}

// --- maps over the arael storage ---

template<class T>
inline detail::eigen_map_t<T, 2, 1> arael_eigen_map(vect2<T>& v) { return detail::eigen_map_t<T, 2, 1>(&v.x); }
template<class T>
inline detail::eigen_cmap_t<T, 2, 1> arael_eigen_map(const vect2<T>& v) { return detail::eigen_cmap_t<T, 2, 1>(&v.x); }

template<class T>
inline detail::eigen_map_t<T, 3, 1> arael_eigen_map(vect3<T>& v) { return detail::eigen_map_t<T, 3, 1>(&v.x); }
template<class T>
inline detail::eigen_cmap_t<T, 3, 1> arael_eigen_map(const vect3<T>& v) { return detail::eigen_cmap_t<T, 3, 1>(&v.x); }

template<class T, std::size_t N>
inline detail::eigen_map_t<T, int(N), 1> arael_eigen_map(vect<T, N>& v) { return detail::eigen_map_t<T, int(N), 1>(v.e); }
template<class T, std::size_t N>
inline detail::eigen_cmap_t<T, int(N), 1> arael_eigen_map(const vect<T, N>& v) { return detail::eigen_cmap_t<T, int(N), 1>(v.e); }

template<class T>
inline detail::eigen_map_t<T, 2, 2> arael_eigen_map(matrix2<T>& m) { return detail::eigen_map_t<T, 2, 2>(&m.rows[0].x); }
template<class T>
inline detail::eigen_cmap_t<T, 2, 2> arael_eigen_map(const matrix2<T>& m) { return detail::eigen_cmap_t<T, 2, 2>(&m.rows[0].x); }

template<class T>
inline detail::eigen_map_t<T, 3, 3> arael_eigen_map(matrix3<T>& m) { return detail::eigen_map_t<T, 3, 3>(&m.rows[0].x); }
template<class T>
inline detail::eigen_cmap_t<T, 3, 3> arael_eigen_map(const matrix3<T>& m) { return detail::eigen_cmap_t<T, 3, 3>(&m.rows[0].x); }

template<class T, std::size_t R, std::size_t C>
inline detail::eigen_map_t<T, int(R), int(C)> arael_eigen_map(matrix<T, R, C>& m) {
    return detail::eigen_map_t<T, int(R), int(C)>(m.rows[0].e);
}
template<class T, std::size_t R, std::size_t C>
inline detail::eigen_cmap_t<T, int(R), int(C)> arael_eigen_map(const matrix<T, R, C>& m) {
    return detail::eigen_cmap_t<T, int(R), int(C)>(m.rows[0].e);
}

// --- free spellings of the same ---

/// `to_eigen(v)`, the same as `v.to_eigen()`.
template<class V> inline auto to_eigen(const V& v) { return arael_to_eigen(v); }
/// `from_eigen<vect3d>(x)`, the same as `vect3d::from_eigen(x)`.
template<class V, class X> inline V from_eigen(const X& x) { return V::from_eigen(x); }
/// `eigen_map(v)`, the same as `v.eigen_map()`.
template<class V> inline auto eigen_map(V& v) { return arael_eigen_map(v); }

} // namespace arael
