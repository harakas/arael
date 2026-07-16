// Euler angle round-trip precision: proving that matrix3::get_euler_angles
// and quatern::get_euler_angles are well-behaved through and around gimbal
// lock, in both f32 and f64. The two share the near-lock algorithm and must
// agree; each is run through the same spot / worst-case / stochastic battery.
//
// # Background
//
// Extracting euler angles from a rotation matrix is ill-conditioned near
// |pitch| = 90 degrees: the roll/yaw split lives in matrix entries scaled
// by cos(pitch), and once cos(pitch)^2 drops below eps those entries are
// float noise -- no algorithm can recover roll and yaw separately,
// because a sub-ulp perturbation of the matrix moves them O(1). This is
// an information floor of the representation, not an implementation
// artifact. matrix3::get_euler_angles therefore switches to the
// gimbal-lock convention (roll = 0, yaw carries the combined angle) when
// m21^2 + m22^2 <= eps, i.e. |cos(pitch)| <= sqrt(eps). See
// src/matrix.rs::get_euler_angles.
//
// # The guaranteed bound (what these tests pin)
//
// The recomposed orientation differs from the input by a geodesic angle of at
// most a small multiple of sqrt(eps) radians. The matrix path tops out at
// ~2*sqrt(eps); the quaternion path at ~3.6*sqrt(eps) -- its near-lock matrix
// entries are reconstructed from half-angle products, so sin(pitch) carries
// more rounding for the asin near +-1 to amplify. Both are O(sqrt(eps)), the
// information floor of extracting euler angles from a float representation:
//
//   matrix f64:  2.0 * 1.49e-8 rad = 1.71e-6 degrees
//   matrix f32:  2.0 * 3.45e-4 rad = 3.96e-2 degrees
//   quat   f64:  3.6 * 1.49e-8 rad = 3.08e-6 degrees
//   quat   f32:  3.4 * 3.45e-4 rad = 6.73e-2 degrees
//
// The worst case sits at roll = +-180 with pitch just inside the lock band
// (error law there: err ~ d + sqrt(eps), d = distance from lock); it decays
// away from that point in every direction and collapses to ~eps at exact lock
// and far from lock. The quaternion supremum is at a yaw the deterministic
// grid does not sample, so the stochastic sweep is what pins it -- extend it
// with SWEEP_SECS to reproduce (0.25 s finds ~3.2, several seconds ~3.6).
//
// # Error metric
//
// Geodesic rotation error: the angle of R_in^T * R_recomposed --
// by how many radians/degrees the round-tripped orientation differs
// from the input orientation. Angle components are NOT compared
// directly: near lock the euler triple is not a continuous function of
// the rotation (crossing the band switches decomposition convention),
// but the rotation it encodes must stay put. That is the contract.
//
// # Running
//
//   cargo test -r --test euler_lock_precision -- --nocapture --test-threads=1
//
// The stochastic sweep runs 0.25 s per precision by default; extend it
// and vary the sampling with environment variables:
//
//   SWEEP_SECS=10 SEED=7 cargo test -r --test euler_lock_precision stochastic -- --nocapture

use arael::matrix::matrix3;
use arael::quatern::quatern;
use arael::vect::vect3;
use arael::utils::Float;

/// A round-trip: (roll, pitch, yaw) as f64 -> (geodesic error [rad], extracted
/// angles, lock branch?). `lock` is read from the reference matrix so the two
/// representations are classified identically.
type Roundtrip<T> = fn(f64, f64, f64) -> (f64, vect3<T>, bool);

/// Geodesic rotation error in radians, small-angle-accurate:
/// |axis * sin(theta)| read from the antisymmetric part of a^T b.
fn geodesic_rad<T: Float>(a: &matrix3<T>, b: &matrix3<T>) -> f64 {
    let d = a.transpose() * *b;
    let sx = (d[2][1] - d[1][2]).to_f64().unwrap() * 0.5;
    let sy = (d[0][2] - d[2][0]).to_f64().unwrap() * 0.5;
    let sz = (d[1][0] - d[0][1]).to_f64().unwrap() * 0.5;
    let s = (sx * sx + sy * sy + sz * sz).sqrt().min(1.0);
    s.asin()
}

/// Round-trip through the MATRIX extraction: build a rotation from
/// (roll, pitch, yaw), extract euler, recompose, measure geodesic error.
fn roundtrip<T: Float>(roll: f64, pitch: f64, yaw: f64) -> (f64, vect3<T>, bool) {
    let f = |v: f64| T::from(v).unwrap();
    let m = matrix3::<T>::rotation_from_euler_angles(
        vect3::<T>::new(f(roll), f(pitch), f(yaw)));
    let cp2 = m[2][1] * m[2][1] + m[2][2] * m[2][2];
    let lock = !(cp2 > T::epsilon());
    let ea = m.get_euler_angles();
    let m2 = matrix3::<T>::rotation_from_euler_angles(ea);
    (geodesic_rad(&m, &m2), ea, lock)
}

/// Round-trip through the QUATERNION extraction: build the rotation as a
/// quaternion, extract euler, recompose to a matrix, measure geodesic error
/// against the quaternion's OWN rotation. The contract is q -> euler ->
/// recompose ~ q, so the reference is q itself, not a separately-built matrix
/// (which would fold in the euler->quat vs euler->matrix build discrepancy).
fn quat_roundtrip<T: Float>(roll: f64, pitch: f64, yaw: f64) -> (f64, vect3<T>, bool) {
    let f = |v: f64| T::from(v).unwrap();
    let q = quatern::<T>::from_euler_angles(vect3::<T>::new(f(roll), f(pitch), f(yaw)));
    let m = q.rotation_matrix();
    let cp2 = m[2][1] * m[2][1] + m[2][2] * m[2][2];
    let lock = !(cp2 > T::epsilon());
    let ea = q.get_euler_angles();
    let m2 = matrix3::<T>::rotation_from_euler_angles(ea);
    (geodesic_rad(&m, &m2), ea, lock)
}

/// Spot checks across the regimes: exact lock, inside the lock band,
/// just above the branch threshold, and far from lock. Asserts the
/// global 2*sqrt(eps) bound everywhere, the branch identity where it is
/// unambiguous, and -- on the main branch -- that the extracted angles
/// match the inputs (the "no garbage angles above the threshold" sanity
/// check; generous margins so architecture/libm variation cannot trip
/// it, while actual garbage is orders of magnitude away).
fn spot_checks<T: Float>(label: &str, rt: Roundtrip<T>, hi_mult: f64) {
    let sqrt_eps = T::epsilon().to_f64().unwrap().sqrt(); // = band edge, rad
    let bound = hi_mult * sqrt_eps; // path's worst-case constant + ~10% margin

    // Distance from lock, in units of the band edge. 0.99 stays clearly
    // inside the band, 1.5 clearly above it; exactly 1.0 is roundoff
    // territory and deliberately not asserted for branch identity.
    let d_mults = [0.0, 0.01, 0.5, 0.99, 1.5, 2.0, 10.0, 1000.0];
    let rolls = [0.0, 0.7854, 2.5, std::f64::consts::PI - 1e-3, std::f64::consts::PI];
    let yaws = [0.3, -2.0];

    let mut worst = 0.0_f64;
    println!("\n== {} spot checks ==  band edge sqrt(eps) = {:.3e} rad", label, sqrt_eps);
    for &hemi in &[1.0_f64, -1.0] {
        for &dm in &d_mults {
            for &r in &rolls {
                for &y in &yaws {
                    let d = dm * sqrt_eps;
                    let pitch = hemi * (std::f64::consts::FRAC_PI_2 - d);
                    let (err, ea, lock) = rt(r, pitch, y);
                    worst = worst.max(err);

                    assert!(err < bound,
                        "{}: err {:.3e} rad exceeds {:.1}*sqrt(eps) at roll={} d={}xedge",
                        label, err, hi_mult, r, dm);

                    if dm <= 0.99 {
                        assert!(lock, "{}: expected lock branch at d={}xedge", label, dm);
                        // Lock convention: roll is exactly zero.
                        assert!(ea.x == T::zero(),
                            "{}: lock branch must return roll = 0", label);
                    }
                    if dm >= 1.5 {
                        assert!(!lock, "{}: expected main branch at d={}xedge", label, dm);
                        // Angles must match the inputs. Worst legitimate
                        // error on the main branch is ~eps/cos(pitch) =
                        // sqrt(eps)/dm; garbage would be O(1).
                        let tol = 5.0 * sqrt_eps / dm + 16.0 * T::epsilon().to_f64().unwrap();
                        let dr = (ea.x.to_f64().unwrap() - r).abs();
                        let dy = (ea.z.to_f64().unwrap() - y).abs();
                        // roll = pi and roll = -pi are the same angle.
                        let dr = dr.min((dr - 2.0 * std::f64::consts::PI).abs());
                        assert!(dr < tol && dy < tol,
                            "{}: main-branch angles off: droll={:.3e} dyaw={:.3e} tol={:.3e} at d={}xedge roll={}",
                            label, dr, dy, tol, dm, r);
                    }
                }
            }
        }
    }
    println!("worst error over all spot checks: {:.3e} rad = {:.3e} deg (bound {:.3e} rad)",
        worst, worst.to_degrees(), bound);
}

// The matrix worst case is ~2*sqrt(eps); the quaternion path tops out at
// ~3.6*sqrt(eps) (pinned by the stochastic sweep) -- its near-lock matrix
// entries are reconstructed from half-angle products, so sin(pitch) carries
// more rounding for asin to amplify. Both are O(sqrt(eps)); the 4.5 bound is
// ~24% over the measured quaternion supremum.
#[test]
fn spot_checks_matrix_f64() { spot_checks::<f64>("matrix f64", roundtrip, 2.2); }
#[test]
fn spot_checks_matrix_f32() { spot_checks::<f32>("matrix f32", roundtrip, 2.2); }
#[test]
fn spot_checks_quat_f64() { spot_checks::<f64>("quat f64", quat_roundtrip, 4.5); }
#[test]
fn spot_checks_quat_f32() { spot_checks::<f32>("quat f32", quat_roundtrip, 4.5); }

/// Deterministic worst-case search: coarse-to-fine maximization over
/// (roll, distance-from-lock) inside the lock band, over both hemispheres and
/// a few fixed yaws. Seed-independent confirmation that a large near-lock error
/// is exercised: the lower assert proves the search still FINDS one, the upper
/// that it stays bounded. For the matrix path this reaches the true supremum
/// (~2*sqrt(eps) at roll = +-180); the quaternion supremum sits at other yaws,
/// so its true peak is pinned by the stochastic sweep instead.
fn worst_case_search<T: Float>(label: &str, rt: Roundtrip<T>, lo_mult: f64, hi_mult: f64) {
    let sqrt_eps = T::epsilon().to_f64().unwrap().sqrt();
    let mut best = (0.0_f64, 0.0_f64, 0.0_f64); // (err rad, roll, d)
    for &yw in &[0.0_f64, 1.0, -2.5] {
        for &hemi in &[1.0_f64, -1.0] {
            let mut center = std::f64::consts::PI;
            let mut span = std::f64::consts::PI;
            for _ in 0..4 {
                for i in 0..81 {
                    let r = center - span + 2.0 * span * (i as f64) / 80.0;
                    for j in 1..40 {
                        let d = sqrt_eps * (j as f64) / 40.0;
                        let pitch = hemi * (std::f64::consts::FRAC_PI_2 - d);
                        let (err, _, lock) = rt(r, pitch, yw);
                        if !lock { continue; }
                        if err > best.0 { best = (err, r, d); }
                    }
                }
                center = best.1;
                span /= 10.0;
            }
        }
    }
    println!("\n== {} worst-case search ==", label);
    println!("supremum: {:.6e} rad = {:.6e} deg at roll={:.4} deg, {:.3}x band edge",
        best.0, best.0.to_degrees(), best.1.to_degrees(), best.2 / sqrt_eps);
    // Search grid tops out at d = 0.975 * edge, just inside the band.
    assert!(best.0 > lo_mult * sqrt_eps,
        "{}: worst case unexpectedly small -- error law changed?", label);
    assert!(best.0 < hi_mult * sqrt_eps,
        "{}: worst case exceeds the {:.1}*sqrt(eps) law", label, hi_mult);
}

#[test]
fn worst_case_search_matrix_f64() { worst_case_search::<f64>("matrix f64", roundtrip, 1.8, 2.1); }
#[test]
fn worst_case_search_matrix_f32() { worst_case_search::<f32>("matrix f32", roundtrip, 1.8, 2.1); }
#[test]
fn worst_case_search_quat_f64() { worst_case_search::<f64>("quat f64", quat_roundtrip, 2.5, 4.5); }
#[test]
fn worst_case_search_quat_f32() { worst_case_search::<f32>("quat f32", quat_roundtrip, 2.5, 4.5); }

/// Stochastic sweep: random roll/yaw over +-180, pitch log-uniform
/// inside the lock band (exact lock every 16th sample), both
/// hemispheres. Complements the deterministic search with coverage of
/// arbitrary angle combinations. 0.25 s per precision by default;
/// SWEEP_SECS / SEED environment variables extend or reseed it.
fn stochastic_lock_sweep<T: Float>(label: &str, seed_offset: u64, rt: Roundtrip<T>, hi_mult: f64) {
    use rand::prelude::*;
    use rand::rngs::StdRng;
    use std::time::Instant;

    let secs: f64 = std::env::var("SWEEP_SECS").ok()
        .and_then(|v| v.parse().ok()).unwrap_or(0.25);
    let seed: u64 = std::env::var("SEED").ok()
        .and_then(|v| v.parse().ok()).unwrap_or(42);
    let mut rng = StdRng::seed_from_u64(seed + seed_offset);

    let sqrt_eps = T::epsilon().to_f64().unwrap().sqrt();
    let mut worst = (0.0_f64, 0.0, 0.0, 0.0); // (err rad, roll, pitch, yaw)
    let mut n = 0_u64;
    let t0 = Instant::now();
    while t0.elapsed().as_secs_f64() < secs {
        let r = rng.random_range(-std::f64::consts::PI..std::f64::consts::PI);
        let y = rng.random_range(-std::f64::consts::PI..std::f64::consts::PI);
        let d = if n % 16 == 0 { 0.0 }
            else { sqrt_eps * 10f64.powf(rng.random_range(-12.0..0.0)) };
        let hemi = if rng.random_bool(0.5) { 1.0 } else { -1.0 };
        let pitch = hemi * (std::f64::consts::FRAC_PI_2 - d);
        let (err, _, lock) = rt(r, pitch, y);
        if !lock { continue; }
        if err > worst.0 { worst = (err, r, pitch, y); }
        n += 1;
    }
    println!("\n== {} stochastic lock sweep ==  {} samples in {}s", label, n, secs);
    println!("worst: {:.3e} rad = {:.3e} deg at roll={:.4} pitch={:.8} yaw={:.4} deg",
        worst.0, worst.0.to_degrees(),
        worst.1.to_degrees(), worst.2.to_degrees(), worst.3.to_degrees());
    assert!(worst.0 < hi_mult * sqrt_eps,
        "{}: stochastic worst exceeds the {:.1}*sqrt(eps) law", label, hi_mult);
}

#[test]
fn stochastic_lock_sweep_matrix_f64() { stochastic_lock_sweep::<f64>("matrix f64", 0, roundtrip, 2.1); }
#[test]
fn stochastic_lock_sweep_matrix_f32() { stochastic_lock_sweep::<f32>("matrix f32", 1, roundtrip, 2.1); }
#[test]
fn stochastic_lock_sweep_quat_f64() { stochastic_lock_sweep::<f64>("quat f64", 2, quat_roundtrip, 4.5); }
#[test]
fn stochastic_lock_sweep_quat_f32() { stochastic_lock_sweep::<f32>("quat f32", 3, quat_roundtrip, 4.5); }
