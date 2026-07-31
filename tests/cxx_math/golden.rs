// The golden math values shared by the C++ (tests/cxx_math.rs) and
// Python (tests/py_math.rs) parity tests: computed with arael's Rust
// types, compared against each ported library's output.

use arael::matrix::{matrix2d, matrix3d};
use arael::quatern::{quatern, quaternd};
use arael::vect::{vect2d, vect3, vect3d};

pub fn golden() -> Vec<(String, f64)> {
    let mut out: Vec<(String, f64)> = Vec::new();
    let p = |out: &mut Vec<(String, f64)>, n: &str, x: f64| out.push((n.to_string(), x));
    let pv2 = |out: &mut Vec<(String, f64)>, n: &str, v: vect2d| {
        out.push((format!("{}.x", n), v.x));
        out.push((format!("{}.y", n), v.y));
    };
    let pv3 = |out: &mut Vec<(String, f64)>, n: &str, v: vect3d| {
        out.push((format!("{}.x", n), v.x));
        out.push((format!("{}.y", n), v.y));
        out.push((format!("{}.z", n), v.z));
    };
    let pq = |out: &mut Vec<(String, f64)>, n: &str, q: quaternd| {
        out.push((format!("{}.t", n), q.t));
        out.push((format!("{}.vx", n), q.v.x));
        out.push((format!("{}.vy", n), q.v.y));
        out.push((format!("{}.vz", n), q.v.z));
    };
    let pm3 = |out: &mut Vec<(String, f64)>, n: &str, m: matrix3d| {
        for r in 0..3 {
            for c in 0..3 {
                out.push((format!("{}.{}{}", n, r, c), m[r][c]));
            }
        }
    };
    let pm2 = |out: &mut Vec<(String, f64)>, n: &str, m: matrix2d| {
        for r in 0..2 {
            for c in 0..2 {
                out.push((format!("{}.{}{}", n, r, c), m[r][c]));
            }
        }
    };

    let a = vect3d::new(1.2, -0.5, 2.0);
    let b = vect3d::new(0.3, 0.9, -1.1);
    pv3(&mut out, "v3_add", a + b);
    pv3(&mut out, "v3_sub", a - b);
    pv3(&mut out, "v3_neg", -a);
    p(&mut out, "v3_dot", a * b);
    pv3(&mut out, "v3_cross", a % b);
    pv3(&mut out, "v3_scale", a * 2.5);
    pv3(&mut out, "v3_lscale", 2.5 * a);
    p(&mut out, "v3_norm", a.norm());
    p(&mut out, "v3_square", a.square());
    pv3(&mut out, "v3_unit", a.unit());
    pv3(&mut out, "v3_across_a", a.across());
    pv3(&mut out, "v3_across_x", vect3d::new(1.0, 0.0, 0.0).across());
    pv3(&mut out, "v3_deg2rad", a.rad2deg().deg2rad());

    let u = vect2d::new(0.8, -1.3);
    let w = vect2d::new(-0.2, 0.7);
    pv2(&mut out, "v2_add", u + w);
    p(&mut out, "v2_dot", u * w);
    p(&mut out, "v2_cross", u.cross(w));
    pv2(&mut out, "v2_across", u.across());
    pv2(&mut out, "v2_unit", u.unit());

    let ea = vect3d::new(0.3, -0.7, 1.9);
    let r = matrix3d::rotation_from_euler_angles(ea);
    pm3(&mut out, "m3_rot", r);
    pv3(&mut out, "m3_rot_ea", r.get_euler_angles());
    pm3(&mut out, "m3_rot_t", r.transpose());
    p(&mut out, "m3_rot_det", r.det());
    pv3(&mut out, "v3_ea_rotmat", ea.rotation_matrix().col(0));

    let ea_lock = vect3d::new(0.4, 1.5707963258, -1.2);
    let rl = matrix3d::rotation_from_euler_angles(ea_lock);
    pv3(&mut out, "m3_lock_ea", rl.get_euler_angles());

    let am = matrix3d::from_elements(1.1, -0.2, 0.4, 0.0, 2.2, -1.0, 0.7, 0.3, 1.9);
    pm3(&mut out, "m3_mul", r * am);
    pv3(&mut out, "m3_mul_v", am * a);
    pv3(&mut out, "v3_mul_m", a * am);
    pm3(&mut out, "m3_add", r + am);
    pm3(&mut out, "m3_scale", am * -1.5);
    p(&mut out, "m3_det", am.det());
    pv3(&mut out, "m3_col1", am.col(1));

    let axis = vect3d::new(1.0, 2.0, -2.0).unit();
    pm3(&mut out, "m3_axis", matrix3d::rotation_from_axis_angle(axis, 2.1));

    let small = vect3d::new(0.01, -0.02, 0.03);
    pm3(&mut out, "m3_rvs", matrix3d::from_rotation_vector_small(small));
    pv3(&mut out, "m3_rvs_get",
        matrix3d::from_rotation_vector_small(small).get_rotation_vector_small());

    let q = quaternd::from_euler_angles(ea);
    pq(&mut out, "q_ea", q);
    pv3(&mut out, "q_ea_back", q.get_euler_angles());
    pm3(&mut out, "q_rotmat", q.rotation_matrix());
    pq(&mut out, "q_from_m", quaternd::from_rotation_matrix(r));
    pv3(&mut out, "q_lock_ea", quaternd::from_euler_angles(ea_lock).get_euler_angles());

    let qa = quaternd::from_axis_angle(axis, 2.1);
    pq(&mut out, "q_axis", qa);
    let (qax, qaa) = qa.get_axis_angle();
    pv3(&mut out, "q_axis_get", qax);
    p(&mut out, "q_angle_get", qaa);

    pq(&mut out, "q_mul", q * qa);
    pq(&mut out, "q_conj", q.conj());
    p(&mut out, "q_dot", q.dot(qa));
    pv3(&mut out, "q_rotate", q.rotate(a));
    pq(&mut out, "q_pow", qa.pow(0.37));
    pq(&mut out, "q_slerp", quaternd::slerp(q, qa, 0.3));
    pq(&mut out, "q_log", qa.log());
    pq(&mut out, "q_exp", qa.log().exp());
    pq(&mut out, "q_rv", quaternd::from_rotation_vector(vect3d::new(0.4, -0.2, 0.9)));
    pq(&mut out, "q_rv_tiny", quaternd::from_rotation_vector(vect3d::new(1e-5, 0.0, 0.0)));
    pq(&mut out, "q_rvs", quaternd::from_rotation_vector_small(small));
    pq(&mut out, "q_two", quaternd::from_two_vectors(
        vect3d::new(1.0, 0.0, 0.0), vect3d::new(0.2, 0.9, -0.1).unit()));
    pq(&mut out, "q_two_anti", quaternd::from_two_vectors(
        vect3d::new(1.0, 0.0, 0.0), vect3d::new(-1.0, 0.0, 0.0)));

    let r2 = matrix2d::rotation(0.7);
    pm2(&mut out, "m2_rot", r2);
    p(&mut out, "m2_angle", r2.get_rotation_angle());
    pm2(&mut out, "m2_mul", r2 * matrix2d::from_elements(1.2, -0.3, 0.5, 0.8));
    pv2(&mut out, "m2_mul_v", r2 * u);
    p(&mut out, "m2_det", r2.det());

    // Symmetric eigen: eigenvalues directly; eigenvectors through the
    // reconstruction R diag(d) R^T (sign/algorithm independent). The
    // C++ twin runs its own Jacobi, so these agree to precision, not
    // bits.
    {
        let s3 = matrix3d::from_elements(
            4.0, 0.5, -0.3, 0.5, 2.5, 0.7, -0.3, 0.7, 1.2);
        let (r, d) = s3.symmetric_eigen();
        pv3(&mut out, "eig3_d", d);
        pm3(&mut out, "eig3_recon", r * matrix3d::from_elements(
            d.x, 0.0, 0.0, 0.0, d.y, 0.0, 0.0, 0.0, d.z) * r.transpose());
        let s2 = matrix2d::from_elements(2.0, 0.6, 0.6, 1.1);
        let (r2e, d2) = s2.symmetric_eigen();
        pv2(&mut out, "eig2_d", d2);
        pm2(&mut out, "eig2_recon", r2e * matrix2d::from_elements(d2.x, 0.0, 0.0, d2.y)
            * r2e.transpose());
    }

    // Pinhole camera (f32, compared at f32 accuracy).
    {
        use arael::geometry::Camera;
        use arael::matrix::matrix3f;
        use arael::vect::{vect2f, vect3f};
        let cam = Camera {
            fx: 800.0, fy: 820.0, cx: 512.0, cy: 384.0, width: 1024, height: 768,
            camera_pos: vect3f::new(0.1, -0.05, 0.3),
            mc2r: matrix3f::rotation_from_euler_angles(vect3f::new(0.1, -0.2, 0.5)),
        };
        let px = vect2f::new(600.0, 300.0);
        let v2d = |v: vect2f| vect2d::new(v.x as f64, v.y as f64);
        let v3d = |v: vect3f| vect3d::new(v.x as f64, v.y as f64, v.z as f64);
        pv2(&mut out, "f32_cam_proj", v2d(cam.project(vect3f::new(0.4, -0.3, 2.0))));
        pv3(&mut out, "f32_cam_unproj", v3d(cam.unproject(px)));
        pv3(&mut out, "f32_cam_w2c", v3d(cam.world_to_camera(
            vect3f::new(3.0, 1.0, 0.5), vect3f::new(1.0, 0.2, 0.0),
            matrix3f::rotation_from_euler_angles(vect3f::new(0.02, 0.05, 1.1)))));
        pv3(&mut out, "f32_cam_unproj_robot", v3d(cam.unproject_to_robot(px)));
        pv2(&mut out, "f32_cam_pixang", v2d(cam.pixel_angular_size(px)));
        p(&mut out, "f32_cam_vis_in", if cam.is_visible(px) { 1.0 } else { 0.0 });
        p(&mut out, "f32_cam_vis_out",
            if cam.is_visible(vect2f::new(-1.0, 300.0)) { 1.0 } else { 0.0 });
    }

    // g2o SE2 parsing (exact doubles through the text round trip).
    {
        let ds = arael::g2o::Dataset2::parse(
            "VERTEX_SE2 0 0.25 -1.5 0.125\n\
             VERTEX_SE2 1 1.75 0.5 -0.25\n\
             FIX 0\n\
             EDGE_SE2 0 1 1.5 2.0 -0.375 100 0 0 100 0 400\n").unwrap();
        p(&mut out, "g2o_n_poses", ds.poses.len() as f64);
        p(&mut out, "g2o_n_deltas", ds.deltas.len() as f64);
        pv2(&mut out, "g2o_p1_t", ds.poses[1].t);
        p(&mut out, "g2o_p1_th", ds.poses[1].th);
        pv2(&mut out, "g2o_d0_dt", ds.deltas[0].dt);
        p(&mut out, "g2o_d0_dth", ds.deltas[0].dth);
        let iso = ds.deltas[0].iso_sqrt_info();
        p(&mut out, "g2o_d0_iso", if iso.is_some() { 1.0 } else { 0.0 });
        let (wt, wr) = iso.unwrap();
        p(&mut out, "g2o_d0_wt", wt);
        p(&mut out, "g2o_d0_wr", wr);
    }

    // g2o SE3 parsing: quaternion normalization, the symmetric
    // information matrix, and its Cholesky blocks.
    {
        let ds = arael::g2o::Dataset3::parse(
            "VERTEX_SE3:QUAT 0 0 0 0 0 0 0 1\n\
             VERTEX_SE3:QUAT 1 1.0 2.0 -0.5 0.2 0.4 -0.4 1.0\n\
             EDGE_SE3:QUAT 0 1 1.5 -0.25 0.75 0.1 -0.2 0.3 0.9 \
             100 1 2 0 0 0 100 3 0 0 0 100 0 0 0 400 4 0 400 5 400\n").unwrap();
        p(&mut out, "g2o3_n_poses", ds.poses.len() as f64);
        p(&mut out, "g2o3_n_deltas", ds.deltas.len() as f64);
        pv3(&mut out, "g2o3_p1_t", ds.poses[1].t);
        pq(&mut out, "g2o3_p1_q", ds.poses[1].q);
        pm3(&mut out, "g2o3_p1_rot", ds.poses[1].rot());
        let d = &ds.deltas[0];
        pv3(&mut out, "g2o3_d0_dt", d.dt);
        pq(&mut out, "g2o3_d0_dq", d.dq);
        p(&mut out, "g2o3_d0_i03", d.info[0][3]);
        p(&mut out, "g2o3_d0_i34", d.info[3][4]);
        let (u_tt, u_tr, u_rr) = d.u_blocks();
        pm3(&mut out, "g2o3_u_tt", u_tt);
        pm3(&mut out, "g2o3_u_tr", u_tr);
        pm3(&mut out, "g2o3_u_rr", u_rr);
    }

    let eaf = vect3::<f32>::new(0.3, -0.7, 1.9);
    let eaf_back = quatern::<f32>::from_euler_angles(eaf).get_euler_angles();
    pv3(&mut out, "f32_ea_back",
        vect3d::new(eaf_back.x as f64, eaf_back.y as f64, eaf_back.z as f64));

    out
}
