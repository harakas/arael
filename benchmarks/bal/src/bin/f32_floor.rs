//! Where an f32 bundle adjustment loses, measured at the f64 optimum of a
//! BAL problem.
//!
//!   f32_floor <dataset> <lambda0> [max_iters]
//!
//! 1. The representation floor: the f64 optimum's parameter vector rounded
//!    to f32, the reference cost and the pixel-error percentiles evaluated
//!    in f64 at that point. What an f32 model can represent at best.
//! 2. The evaluation floor: the residuals at the rounded point computed in
//!    f32 arithmetic against the same in f64, per observation.
//! 3. The assembly: the gradient and the Hessian diagonal the f32 model
//!    forms at the rounded point against the f64 model's at the same point,
//!    with the f64 model's gradient at the optimum for scale.
//! 4. The near-plane observations at the initial point and at the optimum:
//!    how many, how deep, and what their residuals do in f32.

#[path = "../bal.rs"]
mod bal;
#[path = "../arael_runner.rs"]
mod arael_runner;

use std::rc::Rc;

use arael::matrix::matrix3;
use arael::model::{Jacobian, JacobianModel};
use arael::refs;
use arael::simple_lm::{CooMatrix, LmProblem, RootProblem};
use arael::utils::Float;
use arael::vect::{vect2, vect3, vect3d};
use bench_harness::arael::Model;

use arael_runner::{Camera, Obs, Point, Problem, Route, Scene, SceneF, Solution};
use bal::{CameraIn, Dataset};

/// The two models again with the Jacobian diagnostics on, for the
/// decomposition of the f32 gradient's error into the Jacobian's and the
/// residuals' shares.
#[arael::model]
#[arael(root, jacobian)]
struct SceneJ {
    cameras: refs::Vec<Camera<f64>>,
    points: refs::Vec<Point<f64>>,
    observations: std::vec::Vec<Obs<f64>>,
}

#[arael::model]
#[arael(root, f32, jacobian)]
struct SceneFJ {
    cameras: refs::Vec<Camera<f32>>,
    points: refs::Vec<Point<f32>>,
    observations: std::vec::Vec<Obs<f32>>,
}

/// `2 J^T r`, arael's gradient convention, accumulated in f64 from a
/// Jacobian in `T` and residuals given in f64.
fn gradient_of<T: Float>(jac: &Jacobian<T>, r: &[f64], n: usize) -> Vec<f64> {
    let mut g = vec![0.0; n];
    for (row, &ri) in jac.rows.iter().zip(r) {
        for &(idx, d) in &row.entries {
            g[idx as usize] += 2.0 * d.to_f64().unwrap() * ri;
        }
    }
    g
}

/// The Rodrigues vector to a rotation, in `T`.
fn rodrigues<T: Float>(w: vect3<T>) -> matrix3<T> {
    let theta2 = w.square();
    if theta2 > T::from(1e-24).unwrap() {
        let theta = theta2.sqrt();
        matrix3::rotation_from_axis_angle(w * (T::one() / theta), theta)
    } else {
        matrix3::from_elements(
            T::one(), -w.z, w.y,
            w.z, T::one(), -w.x,
            -w.y, w.x, T::one(),
        )
    }
}

/// One camera's parameters in `T`.
#[derive(Clone, Copy)]
struct Cam<T: Float> {
    rot: matrix3<T>,
    t: vect3<T>,
    f: T,
    k1: T,
    k2: T,
}

impl<T: Float> Cam<T> {
    fn of(c: &CameraIn) -> Self {
        let w: vect3<T> = c.rodrigues.cast();
        Cam { rot: rodrigues(w), t: c.t.cast(), f: T::from(c.f).unwrap(), k1: T::from(c.k1).unwrap(), k2: T::from(c.k2).unwrap() }
    }
}

/// The point in the camera frame and the Snavely residual, in `T`.
fn project<T: Float>(c: &Cam<T>, p: vect3<T>, xy: vect2<T>) -> (vect3<T>, [T; 2]) {
    let pc = c.rot * p + c.t;
    let px = -pc.x / pc.z;
    let py = -pc.y / pc.z;
    let r2 = px * px + py * py;
    let d = T::one() + r2 * (c.k1 + c.k2 * r2);
    (pc, [c.f * d * px - xy.x, c.f * d * py - xy.y])
}

/// The f32 residual's error by stage. The chain is `pc = R p + t`,
/// `px = -pc.x / pc.z`, `d = 1 + r2 (k1 + k2 r2)`, `pred = f d px`,
/// `r = pred - obs`. Each variant runs the chain in f64 except for the
/// part named, against the all-f64 residual: rms over every residual.
fn stage_noise(ds: &Dataset, s: &Solution) {
    let cams64: Vec<Cam<f64>> = s.cameras.iter().map(Cam::of).collect();
    let cams32: Vec<Cam<f32>> = s.cameras.iter().map(Cam::of).collect();
    let names = [
        "all f32 (the model)",
        "f32 chain, the final subtraction in f64",
        "f32 chain, final step fma(f d, px, -obs) in f32",
        "f64 chain, only pc rounded to f32",
        "f64 chain, only px, py rounded to f32",
        "f64 chain, only pred = f d px rounded to f32",
        "f64 chain, only the observation stored as f32",
        "f32 chain in normalized coordinates: f (d px - obs/f), obs/f stored f32",
        "f32 transform R p + t (f32 R, products and sums), the rest in f64",
        "f64 chain, only R built in f32 (its entries rounded)",
        "f64 chain, only pc kept in double-single (two f32s)",
        "f32 chain with the camera by its centre: pc = R (p - c), c stored f32",
        "f64 chain, only R built in f32, applied to (p - c)",
    ];
    let mut ss = [0.0f64; 13];
    let mut count = 0usize;
    for o in &ds.observations {
        let c = &cams64[o.cam as usize];
        let c32 = &cams32[o.cam as usize];
        let p = s.points[o.point as usize];
        let p32: vect3<f32> = p.cast();
        let xy = o.xy;
        let xy32: vect2<f32> = xy.cast();
        // f64 chain
        let pc = c.rot * p + c.t;
        let (px, py) = (-pc.x / pc.z, -pc.y / pc.z);
        let dist = |px: f64, py: f64| { let r2 = px * px + py * py; 1.0 + r2 * (c.k1 + c.k2 * r2) };
        let d = dist(px, py);
        let r = [c.f * d * px - xy.x, c.f * d * py - xy.y];
        // f32 chain
        let pc32 = c32.rot * p32 + c32.t;
        let (px32, py32) = (-pc32.x / pc32.z, -pc32.y / pc32.z);
        let r2_32 = px32 * px32 + py32 * py32;
        let d32 = 1.0f32 + r2_32 * (c32.k1 + c32.k2 * r2_32);
        let pred32 = [c32.f * d32 * px32, c32.f * d32 * py32];
        let fd32 = c32.f * d32;
        let from_pc = |q: vect3d| {
            let (qx, qy) = (-q.x / q.z, -q.y / q.z);
            let dq = dist(qx, qy);
            [c.f * dq * qx - xy.x, c.f * dq * qy - xy.y]
        };
        // The camera centre, c = -R^T t, in f64 and rounded to f32: the
        // parameter a centre-based camera would hold.
        let centre = -(c.rot.transpose() * c.t);
        let centre32: vect3<f32> = centre.cast();
        let v: [[f64; 2]; 13] = [
            [(pred32[0] - xy32.x) as f64, (pred32[1] - xy32.y) as f64],
            [pred32[0] as f64 - xy.x, pred32[1] as f64 - xy.y],
            [fd32.mul_add(px32, -xy32.x) as f64, fd32.mul_add(py32, -xy32.y) as f64],
            {
                let q: vect3d = pc.cast::<f32>().cast();
                let (qx, qy) = (-q.x / q.z, -q.y / q.z);
                let dq = dist(qx, qy);
                [c.f * dq * qx - xy.x, c.f * dq * qy - xy.y]
            },
            {
                let (qx, qy) = (px as f32 as f64, py as f32 as f64);
                let dq = dist(qx, qy);
                [c.f * dq * qx - xy.x, c.f * dq * qy - xy.y]
            },
            [(c.f * d * px) as f32 as f64 - xy.x, (c.f * d * py) as f32 as f64 - xy.y],
            [c.f * d * px - xy32.x as f64, c.f * d * py - xy32.y as f64],
            {
                let (nx, ny) = ((xy.x / c.f) as f32, (xy.y / c.f) as f32);
                [(c32.f * (d32 * px32 - nx)) as f64, (c32.f * (d32 * py32 - ny)) as f64]
            },
            from_pc(pc32.cast()),
            {
                let r32: matrix3<f64> = c32.rot.cast();
                from_pc(r32 * p + c.t)
            },
            {
                // pc as a double-single: the f64 value split into an f32 and
                // an f32 remainder, which is what two-float arithmetic keeps.
                let ds = |x: f64| { let hi = x as f32; let lo = (x - hi as f64) as f32; hi as f64 + lo as f64 };
                from_pc(vect3d::new(ds(pc.x), ds(pc.y), ds(pc.z)))
            },
            {
                let q32 = c32.rot * (p32 - centre32);
                let (qx, qy) = (-q32.x / q32.z, -q32.y / q32.z);
                let r2 = qx * qx + qy * qy;
                let dq = 1.0f32 + r2 * (c32.k1 + c32.k2 * r2);
                [(c32.f * dq * qx - xy32.x) as f64, (c32.f * dq * qy - xy32.y) as f64]
            },
            {
                let r32: matrix3<f64> = c32.rot.cast();
                from_pc(r32 * (p - centre))
            },
        ];
        for (k, rv) in v.iter().enumerate() {
            if rv[0].is_finite() && rv[1].is_finite() {
                let (e0, e1) = (rv[0] - r[0], rv[1] - r[1]);
                ss[k] += e0 * e0 + e1 * e1;
            }
        }
        count += 2;
    }
    for (k, name) in names.iter().enumerate() {
        println!("  {:<72} rms {:.2e} px", name, (ss[k] / count as f64).sqrt());
    }
}

/// Cameras and points rounded to f32 and widened again.
fn round_solution(s: &Solution) -> Solution {
    let r3 = |v: vect3d| -> vect3d { v.cast::<f32>().cast() };
    Solution {
        cameras: s.cameras.iter().map(|c| CameraIn {
            rodrigues: r3(c.rodrigues), t: r3(c.t),
            f: c.f as f32 as f64, k1: c.k1 as f32 as f64, k2: c.k2 as f32 as f64,
        }).collect(),
        points: s.points.iter().map(|&p| r3(p)).collect(),
    }
}

fn dataset_at(ds: &Dataset, s: &Solution) -> Dataset {
    Dataset { cameras: s.cameras.clone(), points: s.points.clone(), observations: ds.observations.clone() }
}

fn report_cost(label: &str, ds: &Dataset, s: &Solution) -> f64 {
    let c = bal::reference_cost(ds, &s.cameras, &s.points);
    let (p50, p95) = bal::reprojection_percentiles(ds, &s.cameras, &s.points);
    println!("{:<44} cost {:.4}  px p50/p95 {:.4}/{:.4}", label, c, p50, p95);
    c
}

/// The residuals in `T` at `s` (the values exactly representable in `T`),
/// against the f64 residuals of the same values: RMS and max of the
/// difference, the count of non-finite ones, and the cost of the `T`
/// residuals summed in f64.
fn evaluation_floor<T: Float>(ds: &Dataset, s: &Solution) -> (f64, f64, usize, f64) {
    let cams: Vec<Cam<T>> = s.cameras.iter().map(Cam::of).collect();
    let cams64: Vec<Cam<f64>> = s.cameras.iter().map(Cam::of).collect();
    let (mut ss, mut worst, mut bad, mut cost) = (0.0f64, 0.0f64, 0usize, 0.0f64);
    for o in &ds.observations {
        let p: vect3<T> = s.points[o.point as usize].cast();
        let (_, r) = project(&cams[o.cam as usize], p, o.xy.cast());
        let (_, r64) = project(&cams64[o.cam as usize], s.points[o.point as usize], o.xy);
        let r0 = r[0].to_f64().unwrap();
        let r1 = r[1].to_f64().unwrap();
        if !(r0.is_finite() && r1.is_finite()) {
            bad += 1;
            continue;
        }
        let (d0, d1) = (r0 - r64[0], r1 - r64[1]);
        ss += d0 * d0 + d1 * d1;
        worst = worst.max(d0.abs()).max(d1.abs());
        cost += r0 * r0 + r1 * r1;
    }
    ((ss / (2.0 * ds.observations.len() as f64)).sqrt(), worst, bad, cost)
}

fn norm(v: &[f64]) -> f64 { v.iter().map(|x| x * x).sum::<f64>().sqrt() }

fn diagonal(coo: &CooMatrix<f64>, n: usize) -> Vec<f64> {
    let mut d = vec![0.0; n];
    for ((&r, &c), &v) in coo.rows.iter().zip(&coo.cols).zip(&coo.vals) {
        if r == c { d[r as usize] += v; }
    }
    d
}

fn diagonal32(coo: &CooMatrix<f32>, n: usize) -> Vec<f64> {
    let mut d = vec![0.0; n];
    for ((&r, &c), &v) in coo.rows.iter().zip(&coo.cols).zip(&coo.vals) {
        if r == c { d[r as usize] += v as f64; }
    }
    d
}

/// Relative difference of two vectors over a range: ||a - b|| / ||b||, and
/// the largest entrywise relative difference among the entries of `b` above
/// `floor` in magnitude.
fn compare(a: &[f64], b: &[f64], floor: f64) -> (f64, f64) {
    let diff: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    let worst = a.iter().zip(b).filter(|(_, y)| y.abs() > floor)
        .map(|(x, y)| ((x - y) / y).abs()).fold(0.0, f64::max);
    (norm(&diff) / norm(b), worst)
}

/// Round an f64 scene's observations to the values the f32 scene holds, so
/// that the two models stand at one point. The rotations need nothing: a
/// `MixedEulerAngleParam` holds its angles in f64 under either scalar.
/// Followed by `update_self` on the root.
fn at_f32_values(_cams: &mut refs::Vec<Camera<f64>>, obs: &mut [Obs<f64>]) {
    for o in obs.iter_mut() {
        o.xy = o.xy.cast::<f32>().cast();
    }
}

/// The assembly of the f32 model against the f64 model's at the rounded
/// point, and the f64 gradient at the optimum for scale.
fn assembly(ds: &Dataset, lambda0: f64, opt: &Solution, ncams: usize) {
    let p_opt = Problem { ds: Rc::new(dataset_at(ds, opt)), lambda0, route: Route::Schur };
    let mut scene = Scene::build(&p_opt);
    let mut x = Vec::new();
    Model::serialize(&mut scene, &mut x);
    let n = x.len();
    let ncam_params = ncams * 9;

    let mut g_opt = vec![0.0; n];
    let mut coo = CooMatrix::new(n);
    scene.calc_grad_hessian_sparse(&x, &mut g_opt, &mut coo);

    // The same point rounded to f32: the f64 model's and the f32 model's
    // gradient and Hessian diagonal there. The observations are not in
    // the parameter vector, so the f64 model gets them rounded directly;
    // the rotations are f64 in both models.
    let x32: Vec<f64> = x.iter().map(|&v| v as f32 as f64).collect();
    let x32f: Vec<f32> = x.iter().map(|&v| v as f32).collect();
    Model::deserialize(&mut scene, &x32);
    at_f32_values(&mut scene.cameras, &mut scene.observations);
    arael::model::Model::update_self(&mut scene);
    let mut g64 = vec![0.0; n];
    let mut coo64 = CooMatrix::new(n);
    scene.calc_grad_hessian_sparse(&x32, &mut g64, &mut coo64);
    let d64 = diagonal(&coo64, n);
    let sol32 = scene.solution();
    report_cost("  f64 optimum rounded to f32 (model params)", ds, &sol32);

    // serialize assigns the parameter indices; deserialize alone would
    // leave the entities unplaced.
    let mut scene32 = SceneF::build(&p_opt);
    let mut x32_own = Vec::new();
    Model::serialize(&mut scene32, &mut x32_own);
    assert_eq!(x32_own.len(), n);
    Model::deserialize(&mut scene32, &x32f);
    let mut g32f = vec![0.0f32; n];
    let mut coo32 = CooMatrix::new(n);
    let cost32 = scene32.calc_grad_hessian_sparse(&x32f, &mut g32f, &mut coo32);
    let g32: Vec<f64> = g32f.iter().map(|&v| v as f64).collect();
    let d32 = diagonal32(&coo32, n);

    println!("  gradient norms: f64 at optimum {:.3e}, f64 at rounded point {:.3e}, f32 model at rounded point {:.3e}",
        norm(&g_opt), norm(&g64), norm(&g32));
    let (rel_c, worst_c) = compare(&g32[..ncam_params], &g64[..ncam_params], 1e-6 * norm(&g64));
    let (rel_p, worst_p) = compare(&g32[ncam_params..], &g64[ncam_params..], 1e-6 * norm(&g64));
    println!("  f32 gradient vs f64 at the same point: cameras rel {:.2e} (worst entry {:.1e}), points rel {:.2e} (worst entry {:.1e})",
        rel_c, worst_c, rel_p, worst_p);
    let gnoise: Vec<f64> = g32.iter().zip(&g64).map(|(a, b)| a - b).collect();
    println!("  f32 gradient error norm {:.3e} against the rounding gradient {:.3e} and the optimum's {:.3e}",
        norm(&gnoise), norm(&g64), norm(&g_opt));
    let (rel_dc, worst_dc) = compare(&d32[..ncam_params], &d64[..ncam_params], 0.0);
    let (rel_dp, worst_dp) = compare(&d32[ncam_params..], &d64[ncam_params..], 0.0);
    println!("  f32 Hessian diagonal vs f64: cameras rel {:.2e} (worst entry {:.1e}), points rel {:.2e} (worst entry {:.1e})",
        rel_dc, worst_dc, rel_dp, worst_dp);
    println!("  f32 model's own cost at the rounded point {:.4}", cost32);

    // The gradient's error decomposed: the Jacobians and residuals of both
    // models at the rounded point, recombined across precisions.
    let (cams, pts, obs) = arael_runner::build_parts::<f64>(&p_opt.ds);
    let mut sj = SceneJ { cameras: cams, points: pts, observations: obs };
    let mut xj = Vec::new();
    RootProblem::serialize(&mut sj, &mut xj);
    RootProblem::deserialize(&mut sj, &x32);
    at_f32_values(&mut sj.cameras, &mut sj.observations);
    arael::model::Model::update_self(&mut sj);
    let j64 = sj.calc_jacobian(&x32);
    let (cams, pts, obs) = arael_runner::build_parts::<f32>(&p_opt.ds);
    let mut sjf = SceneFJ { cameras: cams, points: pts, observations: obs };
    let mut xjf = Vec::new();
    RootProblem::serialize(&mut sjf, &mut xjf);
    RootProblem::deserialize(&mut sjf, &x32f);
    let j32 = sjf.calc_jacobian(&x32f);
    assert_eq!(j64.rows.len(), j32.rows.len(), "the two Jacobians must have the same rows");
    let r64: Vec<f64> = j64.rows.iter().map(|r| r.residual).collect();
    let r32: Vec<f64> = j32.rows.iter().map(|r| r.residual as f64).collect();
    let dr: Vec<f64> = r32.iter().zip(&r64).map(|(a, b)| a - b).collect();
    println!("  f32 model's residuals against f64 at the same point: rms {:.3e} px, max {:.3e} px",
        norm(&dr) / (dr.len() as f64).sqrt(), dr.iter().fold(0.0f64, |m: f64, v: &f64| m.max(v.abs())));
    let g_jj = gradient_of(&j64, &r64, n);
    let g_ff = gradient_of(&j32, &r32, n);
    let (san64, _) = compare(&g_jj, &g64, 0.0);
    let (san32, _) = compare(&g_ff, &g32, 0.0);
    println!("  sanity: 2 J^T r reproduces the f64 gradient to {:.1e} and the f32 model's to {:.1e}", san64, san32);
    let g_mixed = gradient_of(&j32, &r64, n);
    let g_jr = gradient_of(&j64, &r32, n);
    let (rel_mixed, _) = compare(&g_mixed, &g64, 0.0);
    let (rel_jr, _) = compare(&g_jr, &g64, 0.0);
    let err = |g: &[f64]| norm(&g.iter().zip(&g64).map(|(a, b)| a - b).collect::<Vec<_>>());
    println!("  f32 Jacobian with f64 residuals: gradient error rel {:.2e}, norm {:.3e} ({:.2} of the optimum's gradient)",
        rel_mixed, err(&g_mixed), err(&g_mixed) / norm(&g_opt));
    println!("  f64 Jacobian with f32 residuals: gradient error rel {:.2e}, norm {:.3e} ({:.2} of the optimum's gradient)",
        rel_jr, err(&g_jr), err(&g_jr) / norm(&g_opt));
    println!("  f32 Jacobian with f32 residuals: gradient error rel {:.2e}, norm {:.3e} ({:.2} of the optimum's gradient)",
        compare(&g_ff, &g64, 0.0).0, err(&g_ff), err(&g_ff) / norm(&g_opt));
}

/// The near-plane observations: by depth, their residuals in f64 and f32.
fn near_plane(label: &str, ds: &Dataset, s: &Solution) {
    let cams64: Vec<Cam<f64>> = s.cameras.iter().map(Cam::of).collect();
    let cams32: Vec<Cam<f32>> = s.cameras.iter().map(Cam::of).collect();
    let bounds = [1e-6, 1e-4, 1e-3, 1e-2, 1e-1];
    let mut count = [0usize; 5];
    let mut cost_share = [0.0f64; 5];
    let mut nonfinite32 = [0usize; 5];
    let mut big32 = [0usize; 5];
    let mut behind = 0usize;
    let mut total = 0.0;
    let mut worst_z = f64::INFINITY;
    for o in &ds.observations {
        let p = s.points[o.point as usize];
        let (pc, r) = project(&cams64[o.cam as usize], p, o.xy);
        let c = r[0] * r[0] + r[1] * r[1];
        total += c;
        if pc.z > 0.0 { behind += 1; }
        let z = pc.z.abs();
        worst_z = worst_z.min(z);
        let (_, r32) = project(&cams32[o.cam as usize], p.cast(), o.xy.cast());
        let fin = r32[0].is_finite() && r32[1].is_finite();
        let big = fin && (r32[0].abs() > 1e6 || r32[1].abs() > 1e6);
        for (i, &b) in bounds.iter().enumerate() {
            if z < b {
                count[i] += 1;
                cost_share[i] += c;
                if !fin { nonfinite32[i] += 1; }
                if big { big32[i] += 1; }
            }
        }
    }
    println!("  {}: {} observations, {} with the point behind the camera, smallest |pc.z| {:.2e}",
        label, ds.observations.len(), behind, worst_z);
    for (i, &b) in bounds.iter().enumerate() {
        println!("    |pc.z| < {:>5.0e}: {:>6} observations, {:.2e} of the cost, f32 residual non-finite {}, above 1e6 px {}",
            b, count[i], cost_share[i] / total, nonfinite32[i], big32[i]);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: f32_floor <dataset> <lambda0> [max_iters]");
        std::process::exit(2);
    }
    let path = &args[1];
    let lambda0: f64 = args[2].parse().expect("lambda0");
    let max_iters: usize = args.get(3).map_or(80, |v| v.parse().expect("max_iters"));
    let ds = Rc::new(bal::load(path));
    let ncams = ds.cameras.len();
    println!("== {}: {} cameras, {} points, {} observations", path, ncams, ds.points.len(), ds.observations.len());

    let initial = Solution { cameras: ds.cameras.clone(), points: ds.points.clone() };
    println!("-- near-plane observations at the initial point");
    near_plane("initial", &ds, &initial);

    let p = Problem { ds: Rc::clone(&ds), lambda0, route: Route::Schur };
    let opt = arael_runner::run_f64_capped(&p, max_iters);
    println!("-- the f64 optimum ({} iterations at most)", max_iters);
    let c_opt = report_cost("  f64 optimum", &ds, &opt);
    near_plane("optimum", &ds, &opt);

    println!("-- 1. the representation floor");
    let rounded = round_solution(&opt);
    let c_round = report_cost("  f64 optimum rounded to f32 (Rodrigues, t, f, k)", &ds, &rounded);
    println!("  relative cost rise from rounding the parameters: {:.2e}", (c_round - c_opt) / c_opt);

    println!("-- 2. the evaluation floor at the rounded point");
    let (rms64, worst64, bad64, _) = evaluation_floor::<f64>(&ds, &rounded);
    let (rms32, worst32, bad32, cost32) = evaluation_floor::<f32>(&ds, &rounded);
    println!("  f64 arithmetic against itself: rms {:.1e} max {:.1e} (non-finite {})", rms64, worst64, bad64);
    println!("  f32 arithmetic against f64:   rms {:.3e} px, max {:.3e} px, non-finite {}; cost of the f32 residuals {:.4} (f64: {:.4})",
        rms32, worst32, bad32, cost32, c_round);
    let typical = (c_round / (2.0 * ds.observations.len() as f64)).sqrt();
    println!("  rms residual at the optimum {:.4} px, so the f32 evaluation noise is {:.1e} of it", typical, rms32 / typical);
    println!("-- 2b. the f32 residual's error by stage (f64 chain except the part named)");
    stage_noise(&ds, &rounded);

    println!("-- 3. the assembly at the rounded point");
    assembly(&ds, lambda0, &opt, ncams);
}
