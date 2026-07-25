// Parity program for the arael C++ math headers: computes the same
// operations as tests/cxx_math.rs from the same inputs and prints
// "name value" lines. The Rust test compiles, runs, and compares.
#include <arael/math.hpp>
#include <cstdio>

using namespace arael;

static void p(const char* n, double x) { std::printf("%s %.17e\n", n, x); }
static void pv2(const char* n, vect2d v) {
    std::printf("%s.x %.17e\n%s.y %.17e\n", n, v.x, n, v.y);
}
static void pv3(const char* n, vect3d v) {
    std::printf("%s.x %.17e\n%s.y %.17e\n%s.z %.17e\n", n, v.x, n, v.y, n, v.z);
}
static void pq(const char* n, quaternd q) {
    std::printf("%s.t %.17e\n", n, q.t);
    std::printf("%s.vx %.17e\n%s.vy %.17e\n%s.vz %.17e\n", n, q.v.x, n, q.v.y, n, q.v.z);
}
static void pm3(const char* n, matrix3d m) {
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            std::printf("%s.%d%d %.17e\n", n, r, c, m[r][c]);
}
static void pm2(const char* n, matrix2d m) {
    for (int r = 0; r < 2; r++)
        for (int c = 0; c < 2; c++)
            std::printf("%s.%d%d %.17e\n", n, r, c, m[r][c]);
}

int main() {
    vect3d a{1.2, -0.5, 2.0};
    vect3d b{0.3, 0.9, -1.1};
    pv3("v3_add", a + b);
    pv3("v3_sub", a - b);
    pv3("v3_neg", -a);
    p("v3_dot", a * b);
    pv3("v3_cross", a % b);
    pv3("v3_scale", a * 2.5);
    pv3("v3_lscale", 2.5 * a);
    p("v3_norm", a.norm());
    p("v3_square", a.square());
    pv3("v3_unit", a.unit());
    pv3("v3_across_a", a.across());
    pv3("v3_across_x", vect3d{1.0, 0.0, 0.0}.across());
    pv3("v3_deg2rad", a.rad2deg().deg2rad());

    vect2d u{0.8, -1.3};
    vect2d w{-0.2, 0.7};
    pv2("v2_add", u + w);
    p("v2_dot", u * w);
    p("v2_cross", u.cross(w));
    pv2("v2_across", u.across());
    pv2("v2_unit", u.unit());

    vect3d ea{0.3, -0.7, 1.9};
    matrix3d r = matrix3d::rotation_from_euler_angles(ea);
    pm3("m3_rot", r);
    pv3("m3_rot_ea", r.get_euler_angles());
    pm3("m3_rot_t", r.transpose());
    p("m3_rot_det", r.det());
    pv3("v3_ea_rotmat", ea.rotation_matrix().col(0));

    // Near gimbal lock.
    vect3d ea_lock{0.4, 1.5707963258, -1.2};
    matrix3d rl = matrix3d::rotation_from_euler_angles(ea_lock);
    pv3("m3_lock_ea", rl.get_euler_angles());

    matrix3d am = matrix3d::from_elements(1.1, -0.2, 0.4, 0.0, 2.2, -1.0, 0.7, 0.3, 1.9);
    pm3("m3_mul", r * am);
    pv3("m3_mul_v", am * a);
    pv3("v3_mul_m", a * am);
    pm3("m3_add", r + am);
    pm3("m3_scale", am * -1.5);
    p("m3_det", am.det());
    pv3("m3_col1", am.col(1));

    vect3d axis = vect3d{1.0, 2.0, -2.0}.unit();
    matrix3d ra = matrix3d::rotation_from_axis_angle(axis, 2.1);
    pm3("m3_axis", ra);

    vect3d small{0.01, -0.02, 0.03};
    pm3("m3_rvs", matrix3d::from_rotation_vector_small(small));
    pv3("m3_rvs_get", matrix3d::from_rotation_vector_small(small).get_rotation_vector_small());

    quaternd q = quaternd::from_euler_angles(ea);
    pq("q_ea", q);
    pv3("q_ea_back", q.get_euler_angles());
    pm3("q_rotmat", q.rotation_matrix());
    pq("q_from_m", quaternd::from_rotation_matrix(r));
    pv3("q_lock_ea", quaternd::from_euler_angles(ea_lock).get_euler_angles());

    quaternd qa = quaternd::from_axis_angle(axis, 2.1);
    pq("q_axis", qa);
    vect3d qax;
    double qaa;
    qa.get_axis_angle(qax, qaa);
    pv3("q_axis_get", qax);
    p("q_angle_get", qaa);

    pq("q_mul", q * qa);
    pq("q_conj", q.conj());
    p("q_dot", q.dot(qa));
    pv3("q_rotate", q.rotate(a));
    pq("q_pow", qa.pow(0.37));
    pq("q_slerp", quaternd::slerp(q, qa, 0.3));
    pq("q_log", qa.log());
    pq("q_exp", qa.log().exp());
    pq("q_rv", quaternd::from_rotation_vector(vect3d{0.4, -0.2, 0.9}));
    pq("q_rv_tiny", quaternd::from_rotation_vector(vect3d{1e-5, 0.0, 0.0}));
    pq("q_rvs", quaternd::from_rotation_vector_small(small));
    pq("q_two", quaternd::from_two_vectors(vect3d{1.0, 0.0, 0.0},
                                           vect3d{0.2, 0.9, -0.1}.unit()));
    pq("q_two_anti", quaternd::from_two_vectors(vect3d{1.0, 0.0, 0.0},
                                                vect3d{-1.0, 0.0, 0.0}));

    matrix2d r2 = matrix2d::rotation(0.7);
    pm2("m2_rot", r2);
    p("m2_angle", r2.get_rotation_angle());
    pm2("m2_mul", r2 * matrix2d::from_elements(1.2, -0.3, 0.5, 0.8));
    pv2("m2_mul_v", r2 * u);
    p("m2_det", r2.det());

    // f32 smoke: euler round trip, printed at double precision.
    vect3<float> eaf{0.3f, -0.7f, 1.9f};
    vect3<float> eaf_back = quatern<float>::from_euler_angles(eaf).get_euler_angles();
    pv3("f32_ea_back", vect3d{double(eaf_back.x), double(eaf_back.y), double(eaf_back.z)});

    return 0;
}
