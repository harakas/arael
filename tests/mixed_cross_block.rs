//! The mixed parent-cross form: a constraint's own CrossBlocks beside
//! CrossBlocks owned by the containing parent. The bundle-adjustment
//! shape -- an image holds its pose and camera refs and the shared
//! (pose, cam) tile, each observation its point ref and the two point
//! tiles -- must match the flat form with three per-observation tiles
//! in cost, gradient, Hessian on every route, and solution. Guards and
//! a robust loss stay per observation; an image may hold no
//! observations; a camera may be frozen.

use arael::model::{CrossBlock, Param, SelfBlock};
use arael::refs::{self, Ref};
use arael::simple_lm::{CooMatrix, LmConfig, LmProblem, RootProblem};
use arael::vect::vect2d;

const TOL: f64 = 1e-9;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs()))
}

/// Cost + all-route + FD + validate battery; returns the dense (g, H).
fn check_model<P>(label: &str, m: &mut P, manual_cost: f64) -> (Vec<f64>, Vec<f64>)
where
    P: LmProblem<f64> + RootProblem<f64>,
{
    let mut x = Vec::new();
    RootProblem::serialize(m, &mut x);
    let n = x.len();
    let cost = m.calc_cost(&x);
    assert!(close(cost, manual_cost, TOL),
        "{label}: calc_cost {} != manual {}", cost, manual_cost);

    let mut gd = vec![0.0; n];
    let mut hd = vec![0.0; n * n];
    let cd = m.calc_grad_hessian_dense(&x, &mut gd, &mut hd);
    assert!(close(cd, cost, TOL), "{label}: dense cost");

    let mut gs = vec![0.0; n];
    let mut coo = CooMatrix::new(n);
    let cs = m.calc_grad_hessian_sparse(&x, &mut gs, &mut coo);
    assert!(close(cs, cost, TOL), "{label}: coo cost");
    let mut hs = vec![0.0; n * n];
    for k in 0..coo.rows.len() {
        let (r, c) = (coo.rows[k] as usize, coo.cols[k] as usize);
        hs[r * n + c] += coo.vals[k];
        if r != c { hs[c * n + r] += coo.vals[k]; }
    }
    for i in 0..n {
        assert!(close(gs[i], gd[i], TOL), "{label}: coo grad[{i}]");
        for j in 0..n {
            assert!(close(hs[i * n + j], hd[i * n + j], TOL),
                "{label}: coo H[{i},{j}] {} != dense {}", hs[i * n + j], hd[i * n + j]);
        }
    }

    let (csc, positions) = coo.to_csc_with_map().unwrap();
    let mut gi = vec![0.0; n];
    let mut vals = vec![0.0; csc.vals.len()];
    let ci = m.calc_grad_hessian_sparse_indexed(&x, &mut gi, &mut vals, &positions);
    assert!(close(ci, cost, TOL), "{label}: indexed cost");
    for i in 0..n {
        assert!(close(gi[i], gd[i], TOL), "{label}: indexed grad[{i}]");
    }

    let kd = n - 1;
    let ldab = kd + 1;
    let mut gb = vec![0.0; n];
    let mut band = vec![0.0; ldab * n];
    let cb = m.calc_grad_hessian_band(&x, &mut gb, &mut band, kd)
        .unwrap_or_else(|e| panic!("{label}: band overflow: {e}"));
    assert!(close(cb, cost, TOL), "{label}: band cost");
    for i in 0..n {
        for j in i..n {
            assert!(close(band[(kd + i - j) + j * ldab], hd[i * n + j], TOL),
                "{label}: band H[{i},{j}]");
        }
    }

    let d = m.check_gradients(&x);
    assert!(d.is_clean(), "{label}: gradient check:\n{}", d);
    let d = m.validate();
    assert!(d.is_clean(), "{label}: validate:\n{}", d);
    (gd, hd)
}

// --- Entities, each with a prior so the solve is well posed ---------

#[arael::model]
#[arael(constraint(hb, {
    [(pose.t.x - pose.px) * 0.3, (pose.t.y - pose.py) * 0.3]
}))]
struct Pose {
    t: Param<vect2d>,
    px: f64,
    py: f64,
    hb: SelfBlock<Pose>,
}

#[arael::model]
#[arael(constraint(hb, {
    [(cam.f - cam.pf) * 0.5]
}))]
struct Cam {
    f: Param<f64>,
    pf: f64,
    hb: SelfBlock<Cam>,
}

#[arael::model]
#[arael(constraint(hb, {
    [(point.p.x - point.px) * 0.2, (point.p.y - point.py) * 0.2]
}))]
struct Point {
    p: Param<vect2d>,
    px: f64,
    py: f64,
    hb: SelfBlock<Point>,
}

// --- Build A: the flat form, three tiles per observation -----------

#[arael::model]
#[arael(constraint([hb_point_pose, hb_point_cam, hb_pose_cam],
    guard = self.on, loss = |s| loss_huber(s, flatobs.k2), {
    [cam.f * (point.p.x - pose.t.x) - flatobs.m.x,
     cam.f * (point.p.y - pose.t.y) - flatobs.m.y]
}))]
struct FlatObs {
    #[arael(ref = root.poses)] pose: Ref<Pose>,
    #[arael(ref = root.cams)] cam: Ref<Cam>,
    #[arael(ref = root.points)] point: Ref<Point>,
    m: vect2d,
    k2: f64,
    on: bool,
    hb_point_pose: CrossBlock<Point, Pose>,
    hb_point_cam: CrossBlock<Point, Cam>,
    hb_pose_cam: CrossBlock<Pose, Cam>,
}

#[arael::model]
#[arael(root)]
struct FlatWorld {
    poses: refs::Vec<Pose>,
    cams: refs::Vec<Cam>,
    points: refs::Vec<Point>,
    obs: std::vec::Vec<FlatObs>,
}

// --- Build B: the mixed form, the image owns the (pose, cam) tile ---

#[arael::model]
#[arael(constraint([hb_point_pose, hb_point_cam, parent.hb_pose_cam],
    parent = image, guard = self.on, loss = |s| loss_huber(s, imgobs.k2), {
    [image.cam.f * (point.p.x - image.pose.t.x) - imgobs.m.x,
     image.cam.f * (point.p.y - image.pose.t.y) - imgobs.m.y]
}))]
struct ImgObs {
    #[arael(ref = root.points)] point: Ref<Point>,
    m: vect2d,
    k2: f64,
    on: bool,
    hb_point_pose: CrossBlock<Point, Pose>,
    hb_point_cam: CrossBlock<Point, Cam>,
}

#[arael::model]
struct Image {
    #[arael(ref = root.poses)] pose: Ref<Pose>,
    #[arael(ref = root.cams)] cam: Ref<Cam>,
    obs: std::vec::Vec<ImgObs>,
    hb_pose_cam: CrossBlock<Pose, Cam>,
}

#[arael::model]
#[arael(root)]
struct ImgWorld {
    poses: refs::Vec<Pose>,
    cams: refs::Vec<Cam>,
    points: refs::Vec<Point>,
    images: std::vec::Vec<Image>,
}

// --- One scene, two builds --------------------------------------------

const NP: usize = 3;
const NC: usize = 2;
const NK: usize = 4;
const K2: f64 = 0.09;

fn t_gt(i: usize) -> (f64, f64) { (0.4 * i as f64, -0.2 * i as f64 + 0.1) }
fn f_gt(j: usize) -> f64 { [1.5, 0.8][j] }
fn p_gt(k: usize) -> (f64, f64) { (1.0 + 0.3 * k as f64, 0.5 - 0.25 * k as f64) }

/// Measurement of point k from pose i through cam j, with a deterministic
/// perturbation that leaves some residuals past the Huber knee.
fn meas(i: usize, j: usize, k: usize) -> (f64, f64) {
    let (tx, ty) = t_gt(i);
    let (px, py) = p_gt(k);
    let noise = 0.05 * (((i * 5 + j * 3 + k * 7) % 9) as f64 - 4.0);
    (f_gt(j) * (px - tx) + noise, f_gt(j) * (py - ty) - 0.5 * noise)
}
fn on(i: usize, j: usize, k: usize) -> bool { (i * 7 + j * 3 + k) % 5 != 0 }
/// Pose 2 through cam 1 has no observations: an empty image.
fn image_present(i: usize, j: usize) -> bool { !(i == 2 && j == 1) }

/// Initial values: the truth, perturbed.
fn pose_init(i: usize) -> (f64, f64) { let (x, y) = t_gt(i); (x + 0.07, y - 0.05 * i as f64) }
fn cam_init(j: usize) -> f64 { f_gt(j) * 1.1 }
fn point_init(k: usize) -> (f64, f64) { let (x, y) = p_gt(k); (x - 0.06, y + 0.04 * k as f64) }

fn huber(s: f64) -> f64 { if s <= K2 { s } else { 2.0 * (K2 * s).sqrt() - K2 } }

fn manual_cost() -> f64 {
    let mut c = 0.0;
    for i in 0..NP {
        let (x, y) = pose_init(i);
        let (px, py) = t_gt(i);
        c += ((x - px) * 0.3).powi(2) + ((y - py) * 0.3).powi(2);
    }
    for j in 0..NC {
        c += ((cam_init(j) - f_gt(j)) * 0.5).powi(2);
    }
    for k in 0..NK {
        let (x, y) = point_init(k);
        let (px, py) = p_gt(k);
        c += ((x - px) * 0.2).powi(2) + ((y - py) * 0.2).powi(2);
    }
    for i in 0..NP {
        for j in 0..NC {
            if !image_present(i, j) { continue; }
            for k in 0..NK {
                if !on(i, j, k) { continue; }
                let (tx, ty) = pose_init(i);
                let f = cam_init(j);
                let (px, py) = point_init(k);
                let (mx, my) = meas(i, j, k);
                let rx = f * (px - tx) - mx;
                let ry = f * (py - ty) - my;
                c += huber(rx * rx + ry * ry);
            }
        }
    }
    c
}

fn entities() -> (refs::Vec<Pose>, refs::Vec<Cam>, refs::Vec<Point>) {
    let mut poses = refs::Vec::new();
    for i in 0..NP {
        let (x, y) = pose_init(i);
        let (px, py) = t_gt(i);
        poses.push(Pose { t: Param::new(vect2d::new(x, y)), px, py, hb: SelfBlock::new() });
    }
    let mut cams = refs::Vec::new();
    for j in 0..NC {
        cams.push(Cam { f: Param::new(cam_init(j)), pf: f_gt(j), hb: SelfBlock::new() });
    }
    // The second camera is held constant.
    cams.iter_mut().nth(1).unwrap().f.optimize = false;
    let mut points = refs::Vec::new();
    for k in 0..NK {
        let (x, y) = point_init(k);
        let (px, py) = p_gt(k);
        points.push(Point { p: Param::new(vect2d::new(x, y)), px, py, hb: SelfBlock::new() });
    }
    (poses, cams, points)
}

fn build_flat() -> FlatWorld {
    let (poses, cams, points) = entities();
    let prefs: Vec<Ref<Pose>> = poses.iter_refs().map(|(r, _)| r).collect();
    let crefs: Vec<Ref<Cam>> = cams.iter_refs().map(|(r, _)| r).collect();
    let krefs: Vec<Ref<Point>> = points.iter_refs().map(|(r, _)| r).collect();
    let mut obs = Vec::new();
    for i in 0..NP {
        for j in 0..NC {
            if !image_present(i, j) { continue; }
            for k in 0..NK {
                let (mx, my) = meas(i, j, k);
                obs.push(FlatObs {
                    pose: prefs[i], cam: crefs[j], point: krefs[k],
                    m: vect2d::new(mx, my), k2: K2, on: on(i, j, k),
                    hb_point_pose: CrossBlock::new(),
                    hb_point_cam: CrossBlock::new(),
                    hb_pose_cam: CrossBlock::new(),
                });
            }
        }
    }
    FlatWorld { poses, cams, points, obs }
}

fn build_images() -> ImgWorld {
    let (poses, cams, points) = entities();
    let prefs: Vec<Ref<Pose>> = poses.iter_refs().map(|(r, _)| r).collect();
    let crefs: Vec<Ref<Cam>> = cams.iter_refs().map(|(r, _)| r).collect();
    let krefs: Vec<Ref<Point>> = points.iter_refs().map(|(r, _)| r).collect();
    let mut images = Vec::new();
    for i in 0..NP {
        for j in 0..NC {
            let mut obs = Vec::new();
            if image_present(i, j) {
                for k in 0..NK {
                    let (mx, my) = meas(i, j, k);
                    obs.push(ImgObs {
                        point: krefs[k], m: vect2d::new(mx, my), k2: K2, on: on(i, j, k),
                        hb_point_pose: CrossBlock::new(),
                        hb_point_cam: CrossBlock::new(),
                    });
                }
            }
            images.push(Image {
                pose: prefs[i], cam: crefs[j], obs, hb_pose_cam: CrossBlock::new(),
            });
        }
    }
    ImgWorld { poses, cams, points, images }
}

#[test]
fn image_owned_tile_matches_the_flat_form() {
    let manual = manual_cost();
    let mut a = build_flat();
    let mut b = build_images();
    let (ga, ha) = check_model("flat", &mut a, manual);
    let (gb, hb) = check_model("images", &mut b, manual);
    assert_eq!(ga.len(), gb.len());
    for i in 0..ga.len() {
        assert!(close(ga[i], gb[i], TOL), "grad[{i}] {} != {}", ga[i], gb[i]);
    }
    for k in 0..ha.len() {
        assert!(close(ha[k], hb[k], TOL), "H[{k}] {} != {}", ha[k], hb[k]);
    }
}

#[test]
fn image_owned_tile_solves_to_the_same_solution() {
    let mut a = build_flat();
    let mut b = build_images();
    a.solve_sparse(&LmConfig::well_conditioned()).unwrap();
    b.solve_sparse(&LmConfig::well_conditioned()).unwrap();
    let (mut xa, mut xb) = (Vec::new(), Vec::new());
    RootProblem::serialize(&mut a, &mut xa);
    RootProblem::serialize(&mut b, &mut xb);
    assert_eq!(xa.len(), xb.len());
    for i in 0..xa.len() {
        assert!(close(xa[i], xb[i], 1e-7), "x[{i}] {} != {}", xa[i], xb[i]);
    }
    // The frozen camera kept its value; the free one moved.
    assert!(close(b.cams.iter().nth(1).unwrap().f.value, cam_init(1), 1e-15));
    assert!(!close(b.cams.iter().next().unwrap().f.value, cam_init(0), 1e-6));
}
