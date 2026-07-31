// arael C++ g2o pose-graph file I/O, mirroring arael's src/g2o.rs:
// VERTEX_SE2 / EDGE_SE2 and VERTEX_SE3:QUAT / EDGE_SE3:QUAT. Unknown
// record types are skipped, vertex ids must be dense and ordered.
// Errors abort with the offending record via arael_assert_true.
// Header-only, C++17.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include "vect.hpp"
#include "matrix.hpp"
#include "quatern.hpp"
#include "assert.hpp"

namespace arael {
namespace g2o {

/// One 2D pose from a VERTEX_SE2 record.
struct Pose2 {
    vect2d t;
    double th;
};

/// One relative SE2 measurement from an EDGE_SE2 record: pose `b` seen
/// from pose `a`'s body frame.
struct DeltaPose2 {
    uint32_t a, b;
    /// Measured translation in `a`'s frame.
    vect2d dt;
    /// Measured heading change.
    double dth;
    /// Information matrix upper triangle in file order:
    /// I11 I12 I13 I22 I23 I33, rows ordered (x y theta).
    double info[6];

    /// Sqrt-information row weights (wt, wr) when the information
    /// matrix is diagonal with equal translation entries; false when
    /// it is anything else.
    bool iso_sqrt_info(double& wt, double& wr) const {
        if (std::abs(info[1]) < 1e-9 && std::abs(info[2]) < 1e-9
            && std::abs(info[4]) < 1e-9 && std::abs(info[0] - info[3]) < 1e-9) {
            wt = std::sqrt(info[0]);
            wr = std::sqrt(info[5]);
            return true;
        }
        return false;
    }
};

/// A 2D pose graph: poses and the relative measurements between them.
struct Dataset2 {
    std::vector<Pose2> poses;
    std::vector<DeltaPose2> deltas;

    /// Parse a 2D pose graph from .g2o text. Aborts on malformed
    /// records (with the 1-based line number).
    static Dataset2 parse(const std::string& text) {
        Dataset2 ds;
        size_t pos = 0, line_no = 0;
        while (pos < text.size()) {
            size_t eol = text.find('\n', pos);
            if (eol == std::string::npos) eol = text.size();
            std::string line = text.substr(pos, eol - pos);
            pos = eol + 1;
            line_no++;

            char tag[32];
            if (std::sscanf(line.c_str(), "%31s", tag) != 1) continue;
            if (std::string(tag) == "VERTEX_SE2") {
                unsigned long id;
                double x, y, th;
                arael_assert_true(std::sscanf(line.c_str(),
                    "%*s %lu %lf %lf %lf", &id, &x, &y, &th) == 4);
                arael_assert_true(id == ds.poses.size()); // dense, ordered
                ds.poses.push_back({{x, y}, th});
            } else if (std::string(tag) == "EDGE_SE2") {
                DeltaPose2 d;
                unsigned long a, b;
                arael_assert_true(std::sscanf(line.c_str(),
                    "%*s %lu %lu %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                    &a, &b, &d.dt.x, &d.dt.y, &d.dth,
                    &d.info[0], &d.info[1], &d.info[2],
                    &d.info[3], &d.info[4], &d.info[5]) == 11);
                d.a = uint32_t(a);
                d.b = uint32_t(b);
                ds.deltas.push_back(d);
            }
        }
        for (const auto& d : ds.deltas)
            arael_assert_true(d.a < ds.poses.size() && d.b < ds.poses.size());
        return ds;
    }

    /// Read a 2D pose graph from a .g2o file. Aborts when the file
    /// cannot be read or parsed.
    static Dataset2 load(const char* path) {
        std::FILE* f = std::fopen(path, "rb");
        arael_assert_true(f != nullptr);
        std::string text;
        char buf[65536];
        size_t n;
        while ((n = std::fread(buf, 1, sizeof buf, f)) > 0)
            text.append(buf, n);
        std::fclose(f);
        return parse(text);
    }
};

/// The file's qx qy qz qw, normalized (a zero-length quaternion
/// aborts).
inline quaternd unit_quat(double qw, vect3d v) {
    double n = std::sqrt(qw * qw + v.square());
    arael_assert_true(n >= 1e-12);
    return {qw / n, v * (1.0 / n)};
}

/// One 3D pose from a VERTEX_SE3:QUAT record.
struct Pose3 {
    vect3d t;
    /// Orientation (unit quaternion; normalized on load).
    quaternd q;

    /// The pose's rotation matrix.
    matrix3d rot() const { return q.rotation_matrix(); }
};

/// One relative SE3 measurement from an EDGE_SE3:QUAT record: pose `b`
/// seen from pose `a`'s body frame.
struct DeltaPose3 {
    uint32_t a, b;
    /// Measured translation in `a`'s frame.
    vect3d dt;
    /// Measured rotation change (unit quaternion; normalized on load).
    quaternd dq;
    /// Full symmetric 6x6 information matrix, rows ordered
    /// (x y z qx qy qz).
    double info[6][6];

    /// Upper Cholesky factor `u` of the information matrix
    /// (`info = u^T u`). Aborts when the matrix is not positive
    /// definite -- that is a data error, not something to paper over.
    void sqrt_info_upper(double u[6][6]) const {
        // Lower Cholesky info = l l^T, returned transposed.
        double l[6][6] = {};
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j <= i; j++) {
                double s = info[i][j];
                for (int k = 0; k < j; k++) s -= l[i][k] * l[j][k];
                if (i == j) {
                    arael_assert_true(s > 0.0);
                    l[i][j] = std::sqrt(s);
                } else {
                    l[i][j] = s / l[j][j];
                }
            }
        }
        for (int i = 0; i < 6; i++)
            for (int j = 0; j < 6; j++)
                u[i][j] = l[j][i];
    }

    /// The three 3x3 blocks of the upper-triangular sqrt-info factor:
    /// `[ u_tt u_tr ; 0 u_rr ]`.
    struct U {
        matrix3d u_tt, u_tr, u_rr;
    };
    U u_blocks() const {
        double u[6][6];
        sqrt_info_upper(u);
        auto b = [&](int r0, int c0) {
            return matrix3d::from_elements(
                u[r0][c0], u[r0][c0 + 1], u[r0][c0 + 2],
                u[r0 + 1][c0], u[r0 + 1][c0 + 1], u[r0 + 1][c0 + 2],
                u[r0 + 2][c0], u[r0 + 2][c0 + 1], u[r0 + 2][c0 + 2]);
        };
        return {b(0, 0), b(0, 3), b(3, 3)};
    }
};

/// A 3D pose graph: poses and the relative measurements between them.
struct Dataset3 {
    std::vector<Pose3> poses;
    std::vector<DeltaPose3> deltas;

    /// Parse a 3D pose graph from .g2o text. Aborts on malformed
    /// records.
    static Dataset3 parse(const std::string& text) {
        Dataset3 ds;
        size_t pos = 0;
        while (pos < text.size()) {
            size_t eol = text.find('\n', pos);
            if (eol == std::string::npos) eol = text.size();
            std::string line = text.substr(pos, eol - pos);
            pos = eol + 1;

            char tag[32];
            if (std::sscanf(line.c_str(), "%31s", tag) != 1) continue;
            if (std::string(tag) == "VERTEX_SE3:QUAT") {
                // VERTEX_SE3:QUAT id x y z qx qy qz qw
                unsigned long id;
                double x, y, z, qx, qy, qz, qw;
                arael_assert_true(std::sscanf(line.c_str(),
                    "%*s %lu %lf %lf %lf %lf %lf %lf %lf",
                    &id, &x, &y, &z, &qx, &qy, &qz, &qw) == 8);
                arael_assert_true(id == ds.poses.size()); // dense, ordered
                ds.poses.push_back({{x, y, z}, unit_quat(qw, {qx, qy, qz})});
            } else if (std::string(tag) == "EDGE_SE3:QUAT") {
                // EDGE_SE3:QUAT a b dx dy dz qx qy qz qw then the 21
                // upper-triangular information entries, row-major,
                // rows ordered (x y z qx qy qz).
                DeltaPose3 d;
                unsigned long a, b;
                double v[7];
                int consumed = 0;
                arael_assert_true(std::sscanf(line.c_str(),
                    "%*s %lu %lu %lf %lf %lf %lf %lf %lf %lf%n",
                    &a, &b, &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6],
                    &consumed) == 9);
                d.a = uint32_t(a);
                d.b = uint32_t(b);
                d.dt = {v[0], v[1], v[2]};
                d.dq = unit_quat(v[6], {v[3], v[4], v[5]});
                const char* s = line.c_str() + consumed;
                for (int i = 0; i < 6; i++) {
                    for (int j = i; j < 6; j++) {
                        char* end = nullptr;
                        double e = std::strtod(s, &end);
                        arael_assert_true(end != s);
                        s = end;
                        d.info[i][j] = e;
                        d.info[j][i] = e;
                    }
                }
                ds.deltas.push_back(d);
            }
        }
        for (const auto& d : ds.deltas)
            arael_assert_true(d.a < ds.poses.size() && d.b < ds.poses.size());
        return ds;
    }

    /// Read a 3D pose graph from a .g2o file. Aborts when the file
    /// cannot be read or parsed.
    static Dataset3 load(const char* path) {
        std::FILE* f = std::fopen(path, "rb");
        arael_assert_true(f != nullptr);
        std::string text;
        char buf[65536];
        size_t n;
        while ((n = std::fread(buf, 1, sizeof buf, f)) > 0)
            text.append(buf, n);
        std::fclose(f);
        return parse(text);
    }
};

} // namespace g2o
} // namespace arael
