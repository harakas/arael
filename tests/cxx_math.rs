// Parity test for the arael C++ math headers
// (cargo-arael/headers/arael/): computes golden values with arael's
// Rust types, compiles and runs tests/cxx_math/main.cpp against the
// headers, and compares every printed value. The euler convention and
// every ported formula must match; skipped (with a note) when no C++
// compiler is available.

use arael::matrix::{matrix2d, matrix3d};
use arael::quatern::{quatern, quaternd};
use arael::vect::{vect2d, vect3, vect3d};

fn golden() -> Vec<(String, f64)> {
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

    let eaf = vect3::<f32>::new(0.3, -0.7, 1.9);
    let eaf_back = quatern::<f32>::from_euler_angles(eaf).get_euler_angles();
    pv3(&mut out, "f32_ea_back",
        vect3d::new(eaf_back.x as f64, eaf_back.y as f64, eaf_back.z as f64));

    out
}

fn find_compiler() -> Option<&'static str> {
    for cc in ["c++", "g++", "clang++"] {
        if std::process::Command::new(cc).arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status().map(|s| s.success()).unwrap_or(false)
        {
            return Some(cc);
        }
    }
    None
}

#[test]
fn cxx_math_headers_match_rust() {
    let Some(cc) = find_compiler() else {
        eprintln!("cxx_math: no C++ compiler found, skipping");
        return;
    };
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let headers = manifest.join("cargo-arael/headers");
    let src = manifest.join("tests/cxx_math/main.cpp");
    let bin = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("cxx_math_parity");

    let status = std::process::Command::new(cc)
        .arg("-std=c++17").arg("-O2").arg("-ffp-contract=off")
        .arg("-I").arg(&headers)
        .arg(&src).arg("-o").arg(&bin)
        .status().expect("compiler spawn");
    assert!(status.success(), "C++ compile failed");

    let output = std::process::Command::new(&bin).output().expect("run");
    assert!(output.status.success(), "C++ run failed");
    let text = String::from_utf8(output.stdout).unwrap();

    let mut got: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    for line in text.lines() {
        let mut it = line.split_whitespace();
        let (Some(name), Some(val)) = (it.next(), it.next()) else { continue };
        got.insert(name.to_string(), val.parse::<f64>().unwrap());
    }

    let expected = golden();
    assert_eq!(got.len(), expected.len(),
        "value count mismatch: C++ printed {}, Rust computed {}", got.len(), expected.len());
    let mut worst = 0.0f64;
    for (name, want) in &expected {
        let have = *got.get(name).unwrap_or_else(|| panic!("C++ output missing `{}`", name));
        // f64 paths agree to a few ulps (same formulas, same libm);
        // the f32 round trip is compared at f32 accuracy.
        let tol = if name.starts_with("f32") { 1e-6 } else { 1e-13 };
        let err = (have - want).abs() / (1.0 + want.abs());
        worst = worst.max(if name.starts_with("f32") { 0.0 } else { err });
        assert!(err <= tol, "`{}`: C++ {} vs Rust {} (rel err {})", name, have, want, err);
    }
    eprintln!("cxx_math: {} values matched, worst f64 rel err {:.3e}", expected.len(), worst);
}
