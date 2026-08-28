// Parity program for the arael C++ math headers: computes the same
// operations as tests/cxx_math.rs from the same inputs and prints
// "name value" lines. The Rust test compiles, runs, and compares.
#include <arael/math.hpp>
#include <arael/geometry.hpp>
#include <arael/g2o.hpp>
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

    // Symmetric eigen: eigenvalues directly; eigenvectors through the
    // reconstruction R diag(d) R^T (sign/algorithm independent). The
    // Rust twin runs nalgebra, so these agree to precision, not bits.
    {
        matrix3d s3 = matrix3d::from_elements(
            4.0, 0.5, -0.3, 0.5, 2.5, 0.7, -0.3, 0.7, 1.2);
        auto [r, d] = s3.symmetric_eigen();
        pv3("eig3_d", d);
        pm3("eig3_recon", r * matrix3d::from_elements(
            d.x, 0.0, 0.0, 0.0, d.y, 0.0, 0.0, 0.0, d.z) * r.transpose());
        matrix2d s2 = matrix2d::from_elements(2.0, 0.6, 0.6, 1.1);
        auto [r2e, d2] = s2.symmetric_eigen();
        pv2("eig2_d", d2);
        pm2("eig2_recon", r2e * matrix2d::from_elements(d2.x, 0.0, 0.0, d2.y)
            * r2e.transpose());
    }

    // Pinhole camera (cameraf, printed at double precision).
    {
        cameraf cam{800.0f, 820.0f, 512.0f, 384.0f, 1024, 768,
            vect3f{0.1f, -0.05f, 0.3f},
            matrix3f::rotation_from_euler_angles({0.1f, -0.2f, 0.5f})};
        vect2f px{600.0f, 300.0f};
        vect2f proj = cam.project({0.4f, -0.3f, 2.0f});
        pv2("f32_cam_proj", proj.cast<double>());
        pv3("f32_cam_unproj", cam.unproject(px).cast<double>());
        pv3("f32_cam_w2c", cam.world_to_camera({3.0f, 1.0f, 0.5f}, {1.0f, 0.2f, 0.0f},
            matrix3f::rotation_from_euler_angles({0.02f, 0.05f, 1.1f})).cast<double>());
        pv3("f32_cam_unproj_robot", cam.unproject_to_robot(px).cast<double>());
        pv2("f32_cam_pixang", cam.pixel_angular_size(px).cast<double>());
        p("f32_cam_vis_in", cam.is_visible(px) ? 1.0 : 0.0);
        p("f32_cam_vis_out", cam.is_visible({-1.0f, 300.0f}) ? 1.0 : 0.0);
    }

    // Pinhole camera at f64 (camerad), compared exactly.
    {
        camerad cam{800.0, 820.0, 512.0, 384.0, 1024, 768,
            vect3d{0.1, -0.05, 0.3},
            matrix3d::rotation_from_euler_angles({0.1, -0.2, 0.5})};
        vect2d px{600.0, 300.0};
        pv2("cam_d_proj", cam.project({0.4, -0.3, 2.0}));
        pv3("cam_d_w2c", cam.world_to_camera({3.0, 1.0, 0.5}, {1.0, 0.2, 0.0},
            matrix3d::rotation_from_euler_angles({0.02, 0.05, 1.1})));
        pv2("cam_d_pixang", cam.pixel_angular_size(px));
    }

    // g2o SE2 parsing (exact doubles through the text round trip).
    {
        arael::g2o::Dataset2 ds = arael::g2o::Dataset2::parse(
            "VERTEX_SE2 0 0.25 -1.5 0.125\n"
            "VERTEX_SE2 1 1.75 0.5 -0.25\n"
            "FIX 0\n"
            "EDGE_SE2 0 1 1.5 2.0 -0.375 100 0 0 100 0 400\n"
            "EDGE_SE2 1 0 0.1 0.2 0.05 1.78 0.027 0.0 3.85 0.0 388.7\n");
        p("g2o_n_poses", double(ds.poses.size()));
        p("g2o_n_deltas", double(ds.deltas.size()));
        pv2("g2o_p1_t", ds.poses[1].t);
        p("g2o_p1_th", ds.poses[1].th);
        pv2("g2o_d0_dt", ds.deltas[0].dt);
        p("g2o_d0_dth", ds.deltas[0].dth);
        double wt = 0, wr = 0;
        p("g2o_d0_iso", ds.deltas[0].iso_sqrt_info(wt, wr) ? 1.0 : 0.0);
        p("g2o_d0_wt", wt);
        p("g2o_d0_wr", wr);
        // Correlated info: sqrt eigenvalues directly, eigenvectors
        // through the reconstruction (sign/algorithm independent)
        auto [er, ew] = ds.deltas[1].eigen_sqrt_info();
        pv3("g2o_d1_ew", ew);
        pm3("g2o_d1_erec", er * matrix3d::from_elements(
            ew.x * ew.x, 0.0, 0.0, 0.0, ew.y * ew.y, 0.0, 0.0, 0.0, ew.z * ew.z)
            * er.transpose());
    }

    // g2o SE3 parsing: quaternion normalization, the symmetric
    // information matrix, and its Cholesky blocks.
    {
        arael::g2o::Dataset3 ds = arael::g2o::Dataset3::parse(
            "VERTEX_SE3:QUAT 0 0 0 0 0 0 0 1\n"
            "VERTEX_SE3:QUAT 1 1.0 2.0 -0.5 0.2 0.4 -0.4 1.0\n"
            "EDGE_SE3:QUAT 0 1 1.5 -0.25 0.75 0.1 -0.2 0.3 0.9 "
            "100 1 2 0 0 0 100 3 0 0 0 100 0 0 0 400 4 0 400 5 400\n");
        p("g2o3_n_poses", double(ds.poses.size()));
        p("g2o3_n_deltas", double(ds.deltas.size()));
        pv3("g2o3_p1_t", ds.poses[1].t);
        pq("g2o3_p1_q", ds.poses[1].q);
        pm3("g2o3_p1_rot", ds.poses[1].rot());
        const arael::g2o::DeltaPose3& d3 = ds.deltas[0];
        pv3("g2o3_d0_dt", d3.dt);
        pq("g2o3_d0_dq", d3.dq);
        p("g2o3_d0_i03", d3.info[0][3]);
        p("g2o3_d0_i34", d3.info[3][4]);
        arael::g2o::DeltaPose3::U u = d3.u_blocks();
        pm3("g2o3_u_tt", u.u_tt);
        pm3("g2o3_u_tr", u.u_tr);
        pm3("g2o3_u_rr", u.u_rr);
    }

    // g2o write-back: the rendered text re-parses to the same values
    // and is byte-identical across the three writers (length +
    // FNV-1a pins).
    {
        auto fnv32 = [](const std::string& s) {
            uint32_t h = 2166136261u;
            for (unsigned char c : s) {
                h ^= c;
                h *= 16777619u;
            }
            return double(h);
        };
        arael::g2o::Dataset2 ds = arael::g2o::Dataset2::parse(
            "VERTEX_SE2 0 0.25 -1.5 0.125\n"
            "VERTEX_SE2 1 1.75 0.5 -0.25\n"
            "FIX 0\n"
            "EDGE_SE2 0 1 1.5 2.0 -0.375 100 0 0 100 0 400\n");
        std::string txt = ds.to_g2o();
        p("g2o_save_len", double(txt.size()));
        p("g2o_save_fnv", fnv32(txt));
        arael::g2o::Dataset2 rt = arael::g2o::Dataset2::parse(txt);
        p("g2o_save_p1_th", rt.poses[1].th);
        p("g2o_save_d0_i5", rt.deltas[0].info[5]);
        arael::g2o::Dataset3 ds3 = arael::g2o::Dataset3::parse(
            "VERTEX_SE3:QUAT 0 0 0 0 0 0 0 1\n"
            "VERTEX_SE3:QUAT 1 1.0 2.0 -0.5 0.2 0.4 -0.4 1.0\n"
            "EDGE_SE3:QUAT 0 1 1.5 -0.25 0.75 0.1 -0.2 0.3 0.9 "
            "100 1 2 0 0 0 100 3 0 0 0 100 0 0 0 400 4 0 400 5 400\n");
        std::string t3 = ds3.to_g2o();
        p("g2o3_save_len", double(t3.size()));
        p("g2o3_save_fnv", fnv32(t3));
        arael::g2o::Dataset3 rt3 = arael::g2o::Dataset3::parse(t3);
        p("g2o3_save_qx", rt3.poses[1].q.v.x);
        p("g2o3_save_i34", rt3.deltas[0].info[3][4]);
    }

    // similar / is_finite / null_space / quatern cast.
    {
        vect3d a2{1.2, -0.5, 2.0};
        vect3d b2{0.3, 0.9, -1.1};
        matrix3d r2 = matrix3d::rotation_from_euler_angles({0.3, -0.7, 1.9});
        quaternd q2 = quaternd::from_euler_angles({0.3, -0.7, 1.9});
        p("sim_v3_same", a2.similar(a2) ? 1.0 : 0.0);
        p("sim_v3_diff", a2.similar(b2) ? 1.0 : 0.0);
        p("sim_m3_same", r2.similar(r2) ? 1.0 : 0.0);
        p("sim_q_same", q2.similar(q2) ? 1.0 : 0.0);
        p("fin_v3", a2.is_finite() ? 1.0 : 0.0);
        p("fin_v3_nan",
          vect3d{std::nan(""), 0.0, 0.0}.is_finite() ? 1.0 : 0.0);
        vect3d ax = vect3d{1.0, 2.0, -2.0}.unit();
        pm3("m3_nullspace", matrix3d::null_space(ax));
        quaternf qf = quaternf::from_euler_angles({0.3f, -0.7f, 1.9f});
        pq("f32_q_cast", qf.cast<double>());
    }

    // f32 smoke: euler round trip, printed at double precision.
    vect3<float> eaf{0.3f, -0.7f, 1.9f};
    vect3<float> eaf_back = quatern<float>::from_euler_angles(eaf).get_euler_angles();
    pv3("f32_ea_back", vect3d{double(eaf_back.x), double(eaf_back.y), double(eaf_back.z)});

    return 0;
}
