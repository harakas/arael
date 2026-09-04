// Eigen interop through the vendored arael/eigen.hpp: every value type
// both ways, blocks, maps, the quaternion component order, transforms
// applied to a point, and a Frame's anchor and pose through the
// generated interface. Prints "ok"; a failed check aborts with its name.
#include <fit.hpp>
#include <arael/eigen.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>

using cxx_fit::Fit;

static void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "eigen interop: %s failed\n", what);
        std::exit(1);
    }
}
static bool near(double a, double b) {
    return std::abs(a - b) <= 1e-12 * (1.0 + std::abs(a) + std::abs(b));
}

int main() {
    // --- vectors: value, from any expression, map read and write ---
    arael::vect3d v3{1.0, 2.0, 3.0};
    Eigen::Vector3d e3 = v3.to_eigen();
    check(e3 == Eigen::Vector3d(1.0, 2.0, 3.0), "vect3 to_eigen");
    check(arael::vect3d::from_eigen(e3 * 2.0).similar({2.0, 4.0, 6.0}), "vect3 from_eigen");
    v3.eigen_map() = e3 * 3.0;
    check(v3.x == 3.0 && v3.y == 6.0 && v3.z == 9.0, "vect3 eigen_map write");
    const arael::vect3d& cv3 = v3;
    check(cv3.eigen_map().sum() == 18.0, "vect3 eigen_map read");

    arael::vect2d v2{0.5, -1.5};
    check(v2.to_eigen() == Eigen::Vector2d(0.5, -1.5), "vect2 to_eigen");
    v2.eigen_map() *= 2.0;
    check(arael::vect2d::from_eigen(v2.to_eigen()).similar({1.0, -3.0}), "vect2 round trip");

    // A 6x6 with S(i, j) = 10 i + j: blocks and columns into the N-types.
    Eigen::Matrix<double, 6, 6> S;
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 6; j++) S(i, j) = 10.0 * i + j;
    auto v6 = arael::vect<double, 6>::from_eigen(S.col(2));
    check(v6.e[3] == 32.0 && v6.e[0] == 2.0, "vect<6> from a column");
    check(v6.to_eigen() == S.col(2), "vect<6> to_eigen");
    v6.eigen_map() = S.col(5);
    check(v6.e[1] == 15.0, "vect<6> eigen_map write");

    // --- matrices: a 3x3 block of the 6x6, row-major map, in-place assign ---
    arael::matrix3d b = arael::matrix3d::from_eigen(S.block<3, 3>(0, 3));
    check(b.rows[1].y == S(1, 4) && b.rows[2].x == S(2, 3), "matrix3 from a block");
    check(b.to_eigen() == S.block<3, 3>(0, 3), "matrix3 to_eigen");
    check(b.eigen_map() == S.block<3, 3>(0, 3), "matrix3 eigen_map read");
    b.eigen_map() = Eigen::Matrix3d::Identity() * 5.0;
    check(b.rows[2].z == 5.0 && b.rows[0].y == 0.0, "matrix3 eigen_map write");
    auto tri = arael::matrix3d::from_eigen(S.topLeftCorner<3, 3>().triangularView<Eigen::Upper>().toDenseMatrix());
    check(tri.rows[1].x == 0.0 && tri.rows[0].z == 2.0, "matrix3 from an upper triangle");

    arael::matrix2d m2 = arael::matrix2d::from_eigen(S.block<2, 2>(4, 4));
    check(m2.rows[0].y == 45.0 && m2.to_eigen() == S.block<2, 2>(4, 4), "matrix2 round trip");
    m2.eigen_map().transposeInPlace();
    check(m2.rows[0].y == 54.0, "matrix2 eigen_map write");

    auto m24 = arael::matrix<double, 2, 4>::from_eigen(S.block<2, 4>(1, 2));
    check(m24.rows[1].e[3] == 25.0 && m24.to_eigen() == S.block<2, 4>(1, 2), "matrix<2,4> round trip");
    m24.eigen_map().row(0).setZero();
    check(m24.rows[0].e[2] == 0.0 && m24.rows[1].e[0] == 22.0, "matrix<2,4> eigen_map write");
    auto m31 = arael::matrix<double, 3, 1>::from_eigen(S.block<3, 1>(0, 0));
    check(m31.rows[2].e[0] == 20.0 && m31.eigen_map()(1, 0) == 10.0, "matrix<3,1> column map");

    // Dynamic-size expressions are checked at runtime and work.
    Eigen::MatrixXd dyn = Eigen::MatrixXd::Identity(3, 3) * 4.0;
    check(arael::matrix3d::from_eigen(dyn).rows[1].y == 4.0, "matrix3 from a dynamic matrix");

    // --- quaternions: scalar first in arael, last in Eigen's coeffs() ---
    arael::quaternd q{0.5, {0.1, -0.2, 0.3}};
    Eigen::Quaterniond eq = q.to_eigen();
    check(eq.w() == 0.5 && eq.x() == 0.1 && eq.y() == -0.2 && eq.z() == 0.3, "quatern component order");
    check(arael::quaternd::from_eigen(eq).similar(q), "quatern round trip");
    Eigen::Matrix3d R = q.unit().rotation_matrix().to_eigen();
    check((R - eq.normalized().toRotationMatrix()).norm() < 1e-12, "quatern rotation agrees");

    // --- transforms: apply to a point both ways, recover the parts ---
    Eigen::Vector3d x(0.3, -0.7, 1.1);
    arael::transform3d t = arael::transform3d::from({1.0, 2.0, 3.0}, q.unit());
    Eigen::Isometry3d et = t.to_eigen();
    check((et * x - t.transform(arael::vect3d::from_eigen(x)).to_eigen()).norm() < 1e-12, "transform3 apply");
    arael::transform3d tb = arael::transform3d::from_eigen(et);
    check(tb.translation.similar(t.translation) && tb.rotation_matrix.similar(t.rotation_matrix), "transform3 round trip");

    arael::scaled_transform3d st = arael::scaled_transform3d::from({1.0, 2.0, 3.0}, q.unit(), 2.5);
    Eigen::Affine3d est = st.to_eigen();
    check((est * x - st.transform(arael::vect3d::from_eigen(x)).to_eigen()).norm() < 1e-12, "scaled_transform3 apply");
    arael::scaled_transform3d sb = arael::scaled_transform3d::from_eigen(est);
    check(near(sb.scale, 2.5) && sb.rotation_matrix.similar(st.rotation_matrix)
          && sb.translation.similar(st.translation), "scaled_transform3 round trip");

    // --- free spellings ---
    check(arael::to_eigen(v3) == v3.to_eigen(), "free to_eigen");
    check(arael::from_eigen<arael::vect3d>(e3).similar({1.0, 2.0, 3.0}), "free from_eigen");
    arael::eigen_map(v3).setZero();
    check(v3.x == 0.0 && v3.z == 0.0, "free eigen_map");

    // --- through the generated interface ---
    Fit fit;
    fit.frames().reserve(1);
    auto f = fit.frames().push();
    f.set_anchor(arael::vect3d::from_eigen(Eigen::Vector3d(4.0, 5.0, 6.0)));
    check(f.anchor().to_eigen() == Eigen::Vector3d(4.0, 5.0, 6.0), "anchor through the ffi");
    f.pose().set_translation(arael::vect3d::from_eigen(Eigen::Vector3d(7.0, 8.0, 9.0)));
    f.pose().set_rotation(arael::quaternd::from_eigen(eq.normalized()));
    arael::transform3d pv = f.pose();
    Eigen::Isometry3d ep = pv.to_eigen();
    check((ep.translation() - Eigen::Vector3d(7.0, 8.0, 9.0)).norm() < 1e-12, "pose translation through the ffi");
    check((ep.linear() - eq.normalized().toRotationMatrix()).norm() < 1e-12, "pose rotation through the ffi");

    std::printf("ok\n");
    return 0;
}
