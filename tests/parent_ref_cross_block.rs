//! The parent-ref form: an observation owns its (point, cam) tile, the
//! point comes from its own ref, the camera from a ref on the containing
//! image, read as `image.cam` through `parent = image` or as
//! `parent.cam`. No parent-owned tile is involved. The image's plain
//! data reads as `image.t` / `parent.t`. Cost, gradient, Hessian on
//! every route, and the solution must match the flat form with both
//! refs on the observation.

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

#[arael::model]
struct Cam {
    f: Param<f64>,
    hb: SelfBlock<Cam>,
}

#[arael::model]
struct Point {
    p: Param<vect2d>,
    hb: SelfBlock<Point>,
}

// --- Build A: flat, both refs on the observation ------------------------

#[arael::model]
#[arael(constraint([hb], {
    [cam.f * (point.p.x + flatobs.t) - flatobs.mx,
     cam.f * (point.p.y - flatobs.t) - flatobs.my]
}))]
struct FlatObs {
    #[arael(ref = root.cams)] cam: Ref<Cam>,
    #[arael(ref = root.points)] point: Ref<Point>,
    t: f64,
    mx: f64,
    my: f64,
    hb: CrossBlock<Point, Cam>,
}

#[arael::model]
#[arael(root)]
struct FlatWorld {
    cams: refs::Vec<Cam>,
    points: refs::Vec<Point>,
    obs: std::vec::Vec<FlatObs>,
}

// --- Build B: the camera ref and the data on the image ---------------------

#[arael::model]
struct Image {
    #[arael(ref = root.cams)] cam: Ref<Cam>,
    t: f64,
    obs: std::vec::Vec<Obs>,
}

// Aliased: the image reads as `image`.
#[arael::model]
#[arael(constraint([hb], parent = image, {
    [image.cam.f * (point.p.x + image.t) - obs.mx,
     image.cam.f * (point.p.y - image.t) - obs.my]
}))]
struct Obs {
    #[arael(ref = root.points)] point: Ref<Point>,
    mx: f64,
    my: f64,
    hb: CrossBlock<Point, Cam>,
}

#[arael::model]
#[arael(root)]
struct NestedWorld {
    cams: refs::Vec<Cam>,
    points: refs::Vec<Point>,
    images: refs::Vec<Image>,
}

// --- Build C: the same, spelled `parent.` -----------------------------------

#[arael::model]
struct PImage {
    #[arael(ref = root.cams)] cam: Ref<Cam>,
    t: f64,
    obs: std::vec::Vec<PObs>,
}

#[arael::model]
#[arael(constraint([hb], {
    [parent.cam.f * (point.p.x + parent.t) - pobs.mx,
     parent.cam.f * (point.p.y - parent.t) - pobs.my]
}))]
struct PObs {
    #[arael(ref = root.points)] point: Ref<Point>,
    mx: f64,
    my: f64,
    hb: CrossBlock<Point, Cam>,
}

#[arael::model]
#[arael(root)]
struct ParentWorld {
    cams: refs::Vec<Cam>,
    points: refs::Vec<Point>,
    images: refs::Vec<PImage>,
}

const NI: usize = 3;
const NC: usize = 2;
const NK: usize = 4;

fn f_gt(j: usize) -> f64 { [1.5, 0.8][j] }
fn p_gt(k: usize) -> (f64, f64) { (1.0 + 0.3 * k as f64, 0.5 - 0.25 * k as f64) }
fn t_of(i: usize) -> f64 { 0.4 * i as f64 - 0.3 }
fn cam_of(i: usize) -> usize { i % NC }
fn meas(i: usize, k: usize) -> (f64, f64) {
    let (x, y) = p_gt(k);
    let f = f_gt(cam_of(i));
    (f * (x + t_of(i)) + 0.02 * (i + k) as f64, f * (y - t_of(i)) - 0.01 * k as f64)
}
fn cam_init(j: usize) -> f64 { f_gt(j) * 1.1 }
fn point_init(k: usize) -> (f64, f64) { let (x, y) = p_gt(k); (x - 0.06, y + 0.04 * k as f64) }

fn manual_cost() -> f64 {
    let mut c = 0.0;
    for i in 0..NI {
        for k in 0..NK {
            let (mx, my) = meas(i, k);
            let (x, y) = point_init(k);
            let f = cam_init(cam_of(i));
            let rx = f * (x + t_of(i)) - mx;
            let ry = f * (y - t_of(i)) - my;
            c += rx * rx + ry * ry;
        }
    }
    c
}

fn cams_and_points() -> (refs::Vec<Cam>, refs::Vec<Point>) {
    let mut cams = refs::Vec::new();
    for j in 0..NC {
        cams.push(Cam { f: Param::new(cam_init(j)), hb: SelfBlock::new() });
    }
    let mut points = refs::Vec::new();
    for k in 0..NK {
        let (x, y) = point_init(k);
        points.push(Point { p: Param::new(vect2d::new(x, y)), hb: SelfBlock::new() });
    }
    (cams, points)
}

fn build_flat() -> FlatWorld {
    let (cams, points) = cams_and_points();
    let crefs: Vec<Ref<Cam>> = cams.iter_refs().map(|(r, _)| r).collect();
    let krefs: Vec<Ref<Point>> = points.iter_refs().map(|(r, _)| r).collect();
    let mut obs = Vec::new();
    for i in 0..NI {
        for k in 0..NK {
            let (mx, my) = meas(i, k);
            obs.push(FlatObs {
                cam: crefs[cam_of(i)], point: krefs[k], t: t_of(i), mx, my,
                hb: CrossBlock::new(),
            });
        }
    }
    FlatWorld { cams, points, obs }
}

fn build_nested() -> NestedWorld {
    let (cams, points) = cams_and_points();
    let crefs: Vec<Ref<Cam>> = cams.iter_refs().map(|(r, _)| r).collect();
    let krefs: Vec<Ref<Point>> = points.iter_refs().map(|(r, _)| r).collect();
    let mut images = refs::Vec::new();
    for i in 0..NI {
        let obs = (0..NK).map(|k| {
            let (mx, my) = meas(i, k);
            Obs { point: krefs[k], mx, my, hb: CrossBlock::new() }
        }).collect();
        images.push(Image { cam: crefs[cam_of(i)], t: t_of(i), obs });
    }
    NestedWorld { cams, points, images }
}

fn build_parent() -> ParentWorld {
    let (cams, points) = cams_and_points();
    let crefs: Vec<Ref<Cam>> = cams.iter_refs().map(|(r, _)| r).collect();
    let krefs: Vec<Ref<Point>> = points.iter_refs().map(|(r, _)| r).collect();
    let mut images = refs::Vec::new();
    for i in 0..NI {
        let obs = (0..NK).map(|k| {
            let (mx, my) = meas(i, k);
            PObs { point: krefs[k], mx, my, hb: CrossBlock::new() }
        }).collect();
        images.push(PImage { cam: crefs[cam_of(i)], t: t_of(i), obs });
    }
    ParentWorld { cams, points, images }
}

fn expect_same(label: &str, (ga, ha): &(Vec<f64>, Vec<f64>), (gb, hb): &(Vec<f64>, Vec<f64>)) {
    assert_eq!(ga.len(), gb.len(), "{label}: parameter count");
    for i in 0..ga.len() {
        assert!(close(ga[i], gb[i], TOL), "{label}: grad[{i}] {} != {}", ga[i], gb[i]);
    }
    for k in 0..ha.len() {
        assert!(close(ha[k], hb[k], TOL), "{label}: H[{k}] {} != {}", ha[k], hb[k]);
    }
}

#[test]
fn parent_ref_form_matches_the_flat_form() {
    let manual = manual_cost();
    let flat = check_model("flat", &mut build_flat(), manual);
    let nested = check_model("nested", &mut build_nested(), manual);
    let parent = check_model("parent", &mut build_parent(), manual);
    expect_same("nested", &flat, &nested);
    expect_same("parent", &flat, &parent);
}

#[test]
fn parent_ref_form_solves_to_the_same_solution() {
    let mut cfg = LmConfig::default();
    cfg.max_iters = 50;
    let mut a = build_flat();
    let mut b = build_nested();
    a.solve_dense(&cfg).unwrap();
    b.solve_dense(&cfg).unwrap();
    let (mut xa, mut xb) = (Vec::new(), Vec::new());
    RootProblem::serialize(&mut a, &mut xa);
    RootProblem::serialize(&mut b, &mut xb);
    assert_eq!(xa.len(), xb.len());
    for i in 0..xa.len() {
        assert!(close(xa[i], xb[i], 1e-7), "x[{i}] {} != {}", xa[i], xb[i]);
    }
}
