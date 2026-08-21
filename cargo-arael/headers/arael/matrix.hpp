// arael C++ math: matrix2<T>, matrix3<T>. Mirrors arael's
// src/matrix.rs: row-major storage (rows are vectors), same euler
// convention (x=roll, y=pitch, z=yaw, R = R(z)*R(y)*R(x)), same
// gimbal-lock handling. Header-only, C++17. The solver-internal
// derivative helpers are not ported.
#pragma once

#include <limits>
#include <utility>
#include "vect.hpp"

namespace arael {

/// 3x3 matrix, row-major.
template<class T>
struct matrix3 {
    vect3<T> rows[3];

    static matrix3 from_rows(vect3<T> r0, vect3<T> r1, vect3<T> r2) {
        return {{r0, r1, r2}};
    }
    static matrix3 from_cols(vect3<T> c0, vect3<T> c1, vect3<T> c2) {
        return {{{c0.x, c1.x, c2.x}, {c0.y, c1.y, c2.y}, {c0.z, c1.z, c2.z}}};
    }
    static matrix3 from_elements(T a00, T a01, T a02, T a10, T a11, T a12,
                                 T a20, T a21, T a22) {
        return {{{a00, a01, a02}, {a10, a11, a12}, {a20, a21, a22}}};
    }
    static matrix3 zero_matrix() {
        return from_elements(T(0), T(0), T(0), T(0), T(0), T(0), T(0), T(0), T(0));
    }
    static matrix3 identity() {
        return from_elements(T(1), T(0), T(0), T(0), T(1), T(0), T(0), T(0), T(1));
    }

    vect3<T> row(std::size_t i) const { return rows[i]; }
    vect3<T> col(std::size_t i) const { return {rows[0][i], rows[1][i], rows[2][i]}; }
    matrix3 transpose() const { return from_cols(rows[0], rows[1], rows[2]); }
    T det() const {
        return rows[0].x * (rows[1].y * rows[2].z - rows[1].z * rows[2].y)
             - rows[0].y * (rows[1].x * rows[2].z - rows[1].z * rows[2].x)
             + rows[0].z * (rows[1].x * rows[2].y - rows[1].y * rows[2].x);
    }
    bool is_finite() const {
        return rows[0].is_finite() && rows[1].is_finite() && rows[2].is_finite();
    }
    template<class K> matrix3<K> cast() const {
        return {{rows[0].template cast<K>(), rows[1].template cast<K>(),
                 rows[2].template cast<K>()}};
    }

    /// Rotation from euler angles (x=roll, y=pitch, z=yaw).
    static matrix3 rotation_from_euler_angles(vect3<T> ea) {
        vect3<T> s, c;
        ea.sincos(s, c);
        return rotation_from_euler_angles_sincos(s, c);
    }
    static matrix3 rotation_from_euler_angles_sincos(vect3<T> s, vect3<T> c) {
        return from_rows(
            {c.y * c.z, -c.x * s.z + c.z * s.x * s.y, c.x * c.z * s.y + s.x * s.z},
            {c.y * s.z, c.x * c.z + s.x * s.y * s.z, c.x * s.y * s.z - c.z * s.x},
            {-s.y, c.y * s.x, c.x * c.y});
    }

    /// Rotation about a unit axis by phi radians.
    static matrix3 rotation_from_axis_angle(vect3<T> axis, T phi) {
        return rotation_from_axis_angle_sincos(axis, std::sin(phi), std::cos(phi));
    }
    static matrix3 rotation_from_axis_angle_sincos(vect3<T> a, T sp, T cp) {
        T k = T(1) - cp;
        return from_rows(
            {cp + a.x * a.x * k, a.x * a.y * k - a.z * sp, a.x * a.z * k + a.y * sp},
            {a.y * a.x * k + a.z * sp, cp + a.y * a.y * k, a.y * a.z * k - a.x * sp},
            {a.z * a.x * k - a.y * sp, a.z * a.y * k + a.x * sp, cp + a.z * a.z * k});
    }

    /// Rotation matrix of the small-angle retraction normalize(1, v/2),
    /// sqrt-free.
    static matrix3 from_rotation_vector_small(vect3<T> v) {
        T x = v.x * T(0.5), y = v.y * T(0.5), z = v.z * T(0.5);
        T x2 = x * x, y2 = y * y, z2 = z * z;
        T s = T(2) / (T(1) + x2 + y2 + z2);
        return from_rows(
            {T(1) - s * (y2 + z2), s * (x * y - z), s * (x * z + y)},
            {s * (x * y + z), T(1) - s * (x2 + z2), s * (y * z - x)},
            {s * (x * z - y), s * (y * z + x), T(1) - s * (x2 + y2)});
    }

    /// Rotation vector of a NEAR-IDENTITY rotation (the extraction
    /// companion of from_rotation_vector_small; use on error rotations,
    /// not full attitudes).
    vect3<T> get_rotation_vector_small() const {
        return {(rows[2][1] - rows[1][2]) * T(0.5),
                (rows[0][2] - rows[2][0]) * T(0.5),
                (rows[1][0] - rows[0][1]) * T(0.5)};
    }

    /// Euler angles (x=roll, y=pitch, z=yaw). At and near gimbal lock
    /// only roll -+ yaw is determined; roll = 0 convention, yaw carries
    /// the combined angle.
    vect3<T> get_euler_angles() const {
        T y = detail::safe_asin(-rows[2][0]);
        T cp2 = rows[2][1] * rows[2][1] + rows[2][2] * rows[2][2];
        if (cp2 > std::numeric_limits<T>::epsilon()) {
            return {std::atan2(rows[2][1], rows[2][2]), y,
                    std::atan2(rows[1][0], rows[0][0])};
        }
        return {T(0), y, std::atan2(-rows[0][1], rows[1][1])};
    }

    /// Eigen-decomposition of a symmetric matrix: (R, d) with the
    /// columns of R the eigenvectors of the eigenvalues in d,
    /// ascending. R has orthonormal columns but is not necessarily a
    /// proper rotation (e.g. covariance whitening); if a proper
    /// rotation is needed, negate one column when det(R) < 0.
    /// Non-finite input propagates: NaN in yields NaN out.
    /// (Cyclic Jacobi in double; the Rust twin runs nalgebra --
    /// results agree to precision, not bit-exactly.)
    std::pair<matrix3, vect3<T>> symmetric_eigen() const {
        double a[3][3], v[3][3];
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++) {
                a[r][c] = double(rows[r][c]);
                v[r][c] = r == c ? 1.0 : 0.0;
            }
        for (int sweep = 0; sweep < 64; sweep++) {
            double off = std::abs(a[0][1]) + std::abs(a[0][2]) + std::abs(a[1][2]);
            if (!(off > 1e-300)) break; // converged, zero, or NaN
            for (int p = 0; p < 2; p++) {
                for (int q = p + 1; q < 3; q++) {
                    if (a[p][q] == 0.0) continue;
                    double theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                    double t = (theta >= 0.0 ? 1.0 : -1.0)
                        / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
                    double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
                    for (int k = 0; k < 3; k++) {
                        double akp = a[k][p], akq = a[k][q];
                        a[k][p] = c * akp - s * akq;
                        a[k][q] = s * akp + c * akq;
                    }
                    for (int k = 0; k < 3; k++) {
                        double apk = a[p][k], aqk = a[q][k];
                        a[p][k] = c * apk - s * aqk;
                        a[q][k] = s * apk + c * aqk;
                    }
                    for (int k = 0; k < 3; k++) {
                        double vkp = v[k][p], vkq = v[k][q];
                        v[k][p] = c * vkp - s * vkq;
                        v[k][q] = s * vkp + c * vkq;
                    }
                }
            }
        }
        int idx[3] = {0, 1, 2};
        for (int i = 0; i < 2; i++)
            for (int j = i + 1; j < 3; j++)
                if (a[idx[j]][idx[j]] < a[idx[i]][idx[i]]) std::swap(idx[i], idx[j]);
        vect3<T> d{T(a[idx[0]][idx[0]]), T(a[idx[1]][idx[1]]), T(a[idx[2]][idx[2]])};
        matrix3 r = from_cols(
            {T(v[0][idx[0]]), T(v[1][idx[0]]), T(v[2][idx[0]])},
            {T(v[0][idx[1]]), T(v[1][idx[1]]), T(v[2][idx[1]])},
            {T(v[0][idx[2]]), T(v[1][idx[2]]), T(v[2][idx[2]])});
        return {r, d};
    }

    bool similar(matrix3 other) const {
        return rows[0].similar(other.rows[0]) && rows[1].similar(other.rows[1])
            && rows[2].similar(other.rows[2]);
    }
    /// Orthonormal basis with unit `n` as the third column (the
    /// caller supplies a unit vector).
    static matrix3 null_space(vect3<T> n) {
        vect3<T> x = n.across();
        vect3<T> y = n % x;
        return from_cols(x, y, n);
    }
    vect3<T>& operator[](std::size_t i) { return rows[i]; }
    const vect3<T>& operator[](std::size_t i) const { return rows[i]; }
    matrix3 operator+(matrix3 m) const {
        return {{rows[0] + m.rows[0], rows[1] + m.rows[1], rows[2] + m.rows[2]}};
    }
    matrix3 operator-(matrix3 m) const {
        return {{rows[0] - m.rows[0], rows[1] - m.rows[1], rows[2] - m.rows[2]}};
    }
    matrix3 operator-() const { return {{-rows[0], -rows[1], -rows[2]}}; }
    vect3<T> operator*(vect3<T> v) const {
        return {rows[0] * v, rows[1] * v, rows[2] * v};
    }
    matrix3 operator*(matrix3 m) const {
        return from_rows(
            {rows[0] * m.col(0), rows[0] * m.col(1), rows[0] * m.col(2)},
            {rows[1] * m.col(0), rows[1] * m.col(1), rows[1] * m.col(2)},
            {rows[2] * m.col(0), rows[2] * m.col(1), rows[2] * m.col(2)});
    }
    matrix3 operator*(T s) const {
        return {{rows[0] * s, rows[1] * s, rows[2] * s}};
    }
};

/// Row vector times matrix (v^T * M).
template<class T>
inline vect3<T> operator*(vect3<T> v, matrix3<T> m) {
    return {v * m.col(0), v * m.col(1), v * m.col(2)};
}

template<class T>
inline matrix3<T> vect3<T>::rotation_matrix() const {
    return matrix3<T>::rotation_from_euler_angles(*this);
}

/// 2x2 matrix, row-major.
template<class T>
struct matrix2 {
    vect2<T> rows[2];

    static matrix2 from_rows(vect2<T> r0, vect2<T> r1) { return {{r0, r1}}; }
    static matrix2 from_cols(vect2<T> c0, vect2<T> c1) {
        return {{{c0.x, c1.x}, {c0.y, c1.y}}};
    }
    static matrix2 from_elements(T a00, T a01, T a10, T a11) {
        return {{{a00, a01}, {a10, a11}}};
    }
    static matrix2 zero_matrix() { return from_elements(T(0), T(0), T(0), T(0)); }
    static matrix2 identity() { return from_elements(T(1), T(0), T(0), T(1)); }

    vect2<T> row(std::size_t i) const { return rows[i]; }
    vect2<T> col(std::size_t i) const { return {rows[0][i], rows[1][i]}; }
    matrix2 transpose() const { return from_cols(rows[0], rows[1]); }
    T det() const { return rows[0].x * rows[1].y - rows[0].y * rows[1].x; }
    bool is_finite() const { return rows[0].is_finite() && rows[1].is_finite(); }
    template<class K> matrix2<K> cast() const {
        return {{rows[0].template cast<K>(), rows[1].template cast<K>()}};
    }

    /// 2D rotation by angle radians (counterclockwise).
    static matrix2 rotation(T angle) {
        return rotation_from_sincos(std::sin(angle), std::cos(angle));
    }
    static matrix2 rotation_from_sincos(T s, T c) {
        return from_rows({c, -s}, {s, c});
    }
    T get_rotation_angle() const { return std::atan2(rows[1][0], rows[0][0]); }

    /// Eigen-decomposition of a symmetric matrix: (R, d) with the
    /// columns of R the eigenvectors of the eigenvalues in d,
    /// ascending. Same contract as matrix3::symmetric_eigen.
    std::pair<matrix2, vect2<T>> symmetric_eigen() const {
        double a = double(rows[0][0]), b = double(rows[0][1]), c = double(rows[1][1]);
        double half = (a + c) * 0.5;
        double disc = std::sqrt((a - c) * (a - c) * 0.25 + b * b);
        double l0 = half - disc, l1 = half + disc;
        // Eigenvector of l1 from the numerically larger residual row.
        double vx, vy;
        if (std::abs(l1 - a) >= std::abs(l1 - c)) { vx = b; vy = l1 - a; }
        else { vx = l1 - c; vy = b; }
        double n = std::sqrt(vx * vx + vy * vy);
        if (n > 0.0) { vx /= n; vy /= n; }
        else { vx = 1.0; vy = 0.0; } // diagonal input: axis-aligned
        // Columns: eigenvector of l0 (perpendicular), then of l1.
        matrix2 r = from_cols({T(-vy), T(vx)}, {T(vx), T(vy)});
        return {r, vect2<T>{T(l0), T(l1)}};
    }

    bool similar(matrix2 other) const {
        return rows[0].similar(other.rows[0]) && rows[1].similar(other.rows[1]);
    }
    vect2<T>& operator[](std::size_t i) { return rows[i]; }
    const vect2<T>& operator[](std::size_t i) const { return rows[i]; }
    matrix2 operator+(matrix2 m) const {
        return {{rows[0] + m.rows[0], rows[1] + m.rows[1]}};
    }
    matrix2 operator-(matrix2 m) const {
        return {{rows[0] - m.rows[0], rows[1] - m.rows[1]}};
    }
    matrix2 operator-() const { return {{-rows[0], -rows[1]}}; }
    vect2<T> operator*(vect2<T> v) const { return {rows[0] * v, rows[1] * v}; }
    matrix2 operator*(matrix2 m) const {
        return from_rows({rows[0] * m.col(0), rows[0] * m.col(1)},
                         {rows[1] * m.col(0), rows[1] * m.col(1)});
    }
    matrix2 operator*(T s) const { return {{rows[0] * s, rows[1] * s}}; }
};

template<class T> inline matrix2<T> operator*(T s, matrix2<T> m) { return m * s; }
template<class T> inline matrix3<T> operator*(T s, matrix3<T> m) { return m * s; }

using matrix2f = matrix2<float>;
using matrix2d = matrix2<double>;
using matrix3f = matrix3<float>;
using matrix3d = matrix3<double>;

static_assert(sizeof(matrix3f) == 36 && sizeof(matrix3d) == 72, "matrix3 layout");
static_assert(sizeof(matrix2f) == 16 && sizeof(matrix2d) == 32, "matrix2 layout");

/// R x C matrix stored as R row vectors -- mirrors arael's
/// matrix<T, R, C> (same layout as the FFI mirror structs). `m[i]`
/// yields the row; `m * v` and `m * m2` multiply; `transpose()` flips.
template<class T, std::size_t R, std::size_t C>
struct matrix {
    vect<T, C> rows[R];

    vect<T, C>& operator[](std::size_t i) { return rows[i]; }
    const vect<T, C>& operator[](std::size_t i) const { return rows[i]; }

    matrix operator+(const matrix& o) const {
        matrix r{};
        for (std::size_t i = 0; i < R; i++) r.rows[i] = rows[i] + o.rows[i];
        return r;
    }
    matrix operator-(const matrix& o) const {
        matrix r{};
        for (std::size_t i = 0; i < R; i++) r.rows[i] = rows[i] - o.rows[i];
        return r;
    }
    matrix operator*(T s) const {
        matrix r{};
        for (std::size_t i = 0; i < R; i++) r.rows[i] = rows[i] * s;
        return r;
    }
    vect<T, R> operator*(const vect<T, C>& v) const {
        vect<T, R> r{};
        for (std::size_t i = 0; i < R; i++) r.e[i] = rows[i] * v;
        return r;
    }
    template<std::size_t K>
    matrix<T, R, K> operator*(const matrix<T, C, K>& o) const {
        matrix<T, R, K> r{};
        for (std::size_t i = 0; i < R; i++)
            for (std::size_t j = 0; j < C; j++)
                for (std::size_t k = 0; k < K; k++)
                    r.rows[i].e[k] += rows[i].e[j] * o.rows[j].e[k];
        return r;
    }
    matrix<T, C, R> transpose() const {
        matrix<T, C, R> r{};
        for (std::size_t i = 0; i < R; i++)
            for (std::size_t j = 0; j < C; j++)
                r.rows[j].e[i] = rows[i].e[j];
        return r;
    }
};

template<std::size_t R, std::size_t C> using matrixf = matrix<float, R, C>;
template<std::size_t R, std::size_t C> using matrixd = matrix<double, R, C>;

static_assert(sizeof(matrix<double, 2, 4>) == 64, "matrix<R, C> layout");

} // namespace arael
