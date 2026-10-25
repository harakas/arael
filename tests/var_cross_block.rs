// A `VarCrossBlock` or `VarSelfBlock` field stores each block at its
// live size and must assemble exactly what the same model's `CrossBlock`
// and `SelfBlock` assemble: the same scene is declared twice, with the
// two sets of markers, and compared on every route. The cameras hold
// parameters at the front, in the middle, at the back and entirely, so
// the blocks' live rectangles differ from their tiles; one link is
// aliased (both sides one point) and the image shares its pose-camera
// block among its observations under a loss. The poses and cameras
// take the variable self block, the points keep the fixed one.

use arael::model::{CrossBlock, Param, SelfBlock, VarCrossBlock, VarSelfBlock};
use arael::refs::{self, Ref};
use arael::simple_lm::{CooMatrix, LmConfig, LmProblem, LmProblemInternals, RootProblem};
use arael::vect::vect2d;

/// The scene with `$cross` as its cross block marker and `$selfb` as the
/// poses' and cameras' self block marker.
macro_rules! scene {
    ($cross:ident, $selfb:ident, $Pose:ident as $pose:ident, $Cam:ident, $Point:ident as $point:ident, $Image:ident,
     $Obs:ident as $obs:ident, $Link:ident as $link:ident, $World:ident, $build:ident) => {
        #[arael::model]
        #[arael(constraint(hb, {
            let d = $pose.t - $pose.prior;
            [d.x * root.drift, d.y * root.drift]
        }))]
        struct $Pose {
            t: Param<vect2d>,
            prior: vect2d,
            hb: $selfb<$Pose>,
        }

        /// A camera whose parameters are commonly held: a focal length, a
        /// principal point and a distortion coefficient.
        #[arael::model]
        struct $Cam {
            f: Param<f64>,
            c: Param<vect2d>,
            k: Param<f64>,
            hb: $selfb<$Cam>,
        }

        #[arael::model]
        #[arael(constraint(hb, {
            let d = $point.pos - $point.prior;
            [d.x * root.drift, d.y * root.drift]
        }))]
        struct $Point {
            pos: Param<vect2d>,
            prior: vect2d,
            hb: SelfBlock<$Point>,
        }

        /// One image: a pose and a camera, with the pose-camera block
        /// shared by its observations.
        #[arael::model]
        struct $Image {
            #[arael(ref = root.poses)] pose: Ref<$Pose>,
            #[arael(ref = root.cams)] cam: Ref<$Cam>,
            obs: std::vec::Vec<$Obs>,
            hb_pose_cam: $cross<$Pose, $Cam>,
        }

        #[arael::model]
        #[arael(constraint([hb_point_pose, hb_point_cam, parent.hb_pose_cam],
            parent = image, loss = |s| loss_huber(s, root.k2), {
            let q = point.pos - image.pose.t;
            let r = q.x * q.x + q.y * q.y;
            let sc = image.cam.f * (1.0 + image.cam.k * r);
            [sc * q.x + image.cam.c.x - $obs.px.x,
             sc * q.y + image.cam.c.y - $obs.px.y]
        }))]
        struct $Obs {
            #[arael(ref = root.points)] point: Ref<$Point>,
            px: vect2d,
            hb_point_pose: $cross<$Point, $Pose>,
            hb_point_cam: $cross<$Point, $Cam>,
        }

        #[arael::model]
        #[arael(constraint(hb, {
            let d = b.pos - a.pos;
            [(d.x * d.x + d.y * d.y - $link.rest * $link.rest) * root.spring]
        }))]
        struct $Link {
            #[arael(ref = root.points)] a: Ref<$Point>,
            #[arael(ref = root.points)] b: Ref<$Point>,
            rest: f64,
            hb: $cross<$Point, $Point>,
        }

        #[arael::model]
        #[arael(root)]
        struct $World {
            poses: refs::Vec<$Pose>,
            cams: refs::Vec<$Cam>,
            points: refs::Vec<$Point>,
            images: std::vec::Vec<$Image>,
            links: std::vec::Vec<$Link>,
            drift: f64,
            spring: f64,
            k2: f64,
        }

        /// `n` points seen from `n / 2` images through four cameras:
        /// camera 0 holds its principal point (the middle slots), camera
        /// 1 its focal length and distortion (front and back), camera 2
        /// everything, camera 3 nothing. Every third point and the
        /// second pose are held.
        fn $build(n: usize) -> $World {
            let mut w = $World {
                poses: refs::Vec::new(),
                cams: refs::Vec::new(),
                points: refs::Vec::new(),
                images: std::vec::Vec::new(),
                links: std::vec::Vec::new(),
                drift: 0.05,
                spring: 0.7,
                k2: 0.4,
            };
            for j in 0..4 {
                let mut cam = $Cam {
                    f: Param::new(1.2 + 0.1 * j as f64),
                    c: Param::new(vect2d::new(0.1 * j as f64, -0.05)),
                    k: Param::new(0.02 * j as f64),
                    hb: $selfb::new(),
                };
                match j {
                    0 => cam.c.optimize = false,
                    1 => { cam.f.optimize = false; cam.k.optimize = false; }
                    2 => { cam.f.optimize = false; cam.c.optimize = false; cam.k.optimize = false; }
                    _ => {}
                }
                w.cams.push(cam);
            }
            for i in 0..n {
                let pos = vect2d::new(0.3 * i as f64, if i % 2 == 0 { 0.8 } else { -0.6 });
                let mut p = $Point { pos: Param::new(pos), prior: pos, hb: SelfBlock::new() };
                p.optimize_pos(i % 3 != 0);
                w.points.push(p);
            }
            for i in 0..n / 2 {
                let t = vect2d::new(0.15 * i as f64 - 1.0, 2.0);
                let mut pose = $Pose { t: Param::new(t), prior: t, hb: $selfb::new() };
                if i == 1 { pose.t.optimize = false; }
                w.poses.push(pose);
                let obs = (0..3).map(|k| {
                    let point = w.points.ref_at((2 * i + 3 * k) % n);
                    $Obs {
                        point,
                        px: vect2d::new(0.4 * k as f64 - 0.3, 0.1 * i as f64),
                        hb_point_pose: $cross::new(),
                        hb_point_cam: $cross::new(),
                    }
                }).collect();
                w.images.push($Image {
                    pose: w.poses.ref_at(i),
                    cam: w.cams.ref_at(i % 4),
                    obs,
                    hb_pose_cam: $cross::new(),
                });
            }
            for i in 1..n {
                let (a, b) = (w.points.ref_at(i - 1), w.points.ref_at(i));
                w.links.push($Link { a, b, rest: 0.9, hb: $cross::new() });
            }
            // An aliased link: both sides one point.
            let p = w.points.ref_at(n / 2);
            w.links.push($Link { a: p, b: p, rest: 0.0, hb: $cross::new() });
            w
        }

        impl $Point {
            fn optimize_pos(&mut self, on: bool) { self.pos.optimize = on; }
        }
    };
}

scene!(CrossBlock, SelfBlock, Pose as pose, Cam, Point as point, Image, Obs as obs, Link as link, World, build_fixed);
scene!(VarCrossBlock, VarSelfBlock, VPose as vpose, VCam, VPoint as vpoint, VImage, VObs as vobs, VLink as vlink, VWorld, build_var);

fn close(a: f64, b: f64, rel: f64) -> bool {
    (a - b).abs() <= rel * (1.0 + a.abs().max(b.abs()))
}

fn assert_close(what: &str, a: &[f64], b: &[f64], rel: f64) {
    assert_eq!(a.len(), b.len(), "{}: lengths", what);
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(close(*x, *y, rel), "{} differs at {}: {} vs {}", what, i, x, y);
    }
}

/// Both scenes serialized: the same parameter vector.
fn both(n: usize) -> (World, VWorld, Vec<f64>) {
    let mut fixed = build_fixed(n);
    let mut var = build_var(n);
    let mut x = Vec::new();
    fixed.serialize(&mut x);
    let mut x2 = Vec::new();
    var.serialize(&mut x2);
    assert_eq!(x, x2);
    (fixed, var, x)
}

/// The dense route: cost, gradient and Hessian agree.
#[test]
fn dense_assembly_matches_the_fixed_width_blocks() {
    let (mut fixed, mut var, x) = both(18);
    let n = x.len();
    let (mut g1, mut h1) = (vec![0.0; n], vec![0.0; n * n]);
    let (mut g2, mut h2) = (vec![0.0; n], vec![0.0; n * n]);
    let c1 = fixed.calc_grad_hessian_dense(&x, &mut g1, &mut h1);
    let c2 = var.calc_grad_hessian_dense(&x, &mut g2, &mut h2);
    assert!(close(c1, c2, 1e-13), "cost {} vs {}", c1, c2);
    assert_close("grad", &g1, &g2, 1e-12);
    assert_close("hessian", &h1, &h2, 1e-12);
    assert!(h1.iter().any(|&v| v != 0.0));
}

/// The COO, band and per-entry indexed routes agree with the dense one.
#[test]
fn sparse_routes_match_the_fixed_width_blocks() {
    let (mut fixed, mut var, x) = both(18);
    let n = x.len();
    let (mut g1, mut h1) = (vec![0.0; n], vec![0.0; n * n]);
    fixed.calc_grad_hessian_dense(&x, &mut g1, &mut h1);

    let mut g = vec![0.0; n];
    let mut coo = CooMatrix::new(n);
    var.calc_grad_hessian_sparse(&x, &mut g, &mut coo);
    assert_close("coo grad", &g1, &g, 1e-12);
    let mut from_coo = vec![0.0; n * n];
    for k in 0..coo.rows.len() {
        let (i, j, v) = (coo.rows[k] as usize, coo.cols[k] as usize, coo.vals[k]);
        from_coo[i * n + j] += v;
        if i != j { from_coo[j * n + i] += v; }
    }
    assert_close("coo hessian", &h1, &from_coo, 1e-12);

    let mut ctx = arael::threads::Context::new();
    let kd = n - 1;
    let ldab = kd + 1;
    let mut gb = vec![0.0; n];
    let mut band = vec![0.0; ldab * n];
    var.calc_grad_hessian_band(&x, &mut gb, &mut band, kd, &mut ctx).unwrap();
    assert_close("band grad", &g1, &gb, 1e-12);
    for i in 0..n {
        for j in i..n {
            assert!(close(band[(kd + i - j) + j * ldab], h1[i * n + j], 1e-12), "band H[{i},{j}]");
        }
    }

    let (csc, positions) = coo.to_csc_with_positions().unwrap();
    let mut gi = vec![0.0; n];
    let mut vals = vec![0.0; csc.vals.len()];
    var.calc_grad_hessian_sparse_indexed(&x, &mut gi, &mut vals, &positions, &mut ctx);
    assert_close("indexed grad", &g1, &gi, 1e-12);
    let mut from_csc = vec![0.0; n * n];
    for j in 0..n {
        for p in csc.col_ptr[j]..csc.col_ptr[j + 1] {
            let i = csc.row_idx[p] as usize;
            from_csc[i * n + j] += vals[p];
            if i != j { from_csc[j * n + i] += vals[p]; }
        }
    }
    assert_close("indexed hessian", &h1, &from_csc, 1e-12);
}

/// The tiled scatter, the route every sparse solve takes: both scenes
/// bind the same pattern and fill it alike, whatever the buffer held.
#[test]
fn the_tiled_scatter_matches_the_fixed_width_blocks() {
    use arael::store::{block_partition_from_spans, csc_from_cells, HessianBinder, PositionStream};
    fn fill<P: LmProblemInternals<f64> + RootProblem<f64>>(w: &mut P, x: &[f64], threads: usize) -> (Vec<f64>, Vec<f64>) {
        let mut ctx = arael::threads::Context::new();
        ctx.set_threads(threads);
        w.begin_with_context(&mut ctx);
        let n = x.len();
        let mut cells = Vec::new();
        w.collect_hessian_cells(&mut cells, &mut ctx);
        let mut spans = Vec::new();
        w.collect_param_block_spans(&mut spans, &mut ctx);
        let partition = block_partition_from_spans(&spans, n);
        let (csc, mut resolver) = csc_from_cells::<f64>(&partition, &cells);
        let mut stream = PositionStream::new();
        w.bind_hessian_positions(
            &mut HessianBinder::Tiled(&mut |i, j| resolver.resolve_tile(i, j)),
            &mut stream,
            &mut ctx,
        );
        assert!(stream.tiled());
        let mut vals = vec![1.0e9; csc.vals.len()];
        let mut g = vec![0.0; n];
        w.calc_grad_hessian_sparse_indexed(x, &mut g, &mut vals, &stream, &mut ctx);
        (vals, g)
    }
    let (mut fixed, mut var, x) = both(18);
    let (vf, gf) = fill(&mut fixed, &x, 1);
    let (vv, gv) = fill(&mut var, &x, 1);
    assert_close("tiled hessian", &vf, &vv, 1e-12);
    assert_close("grad", &gf, &gv, 1e-12);
    #[cfg(feature = "threads")]
    {
        let (v4, g4) = fill(&mut var, &x, 4);
        assert_close("tiled hessian, four stores", &vf, &v4, 1e-12);
        assert_close("grad, four stores", &gf, &g4, 1e-12);
    }
}

/// A sparse solve lands where the fixed-width scene's does.
#[test]
fn a_sparse_solve_matches_the_fixed_width_blocks() {
    let cfg = LmConfig::<f64> { max_iters: 50, ..Default::default() };
    let f = build_fixed(18).solve_sparse(&cfg).unwrap();
    let v = build_var(18).solve_sparse(&cfg).unwrap();
    assert_eq!(f.status, v.status);
    assert!(close(f.end_cost, v.end_cost, 1e-9), "{} vs {}", f.end_cost, v.end_cost);
    assert_close("x", &f.x, &v.x, 1e-7);
    assert!(f.end_cost < f.start_cost);
}

/// The variable-width store holds the same blocks in fewer bytes.
#[test]
fn the_store_is_smaller() {
    let (mut fixed, mut var, _x) = both(18);
    let mut cf = arael::threads::Context::new();
    fixed.begin_with_context(&mut cf);
    let mut cv = arael::threads::Context::new();
    var.begin_with_context(&mut cv);
    let rf = cf.sweeps().expect("the fixed scene reports its stores");
    let rv = cv.sweeps().expect("the variable scene reports its stores");
    assert_eq!(rf.whole.cross_blocks, rv.whole.cross_blocks);
    assert_eq!(rf.whole.self_blocks, rv.whole.self_blocks);
    assert!(rv.whole.bytes < rf.whole.bytes, "{} vs {}", rv.whole.bytes, rf.whole.bytes);
    let held = arael::store::StoreFootprint::sum(&rv.held);
    assert_eq!(held.bytes, rv.whole.bytes, "one store is the whole model");
    assert_eq!(held.cross_blocks, rv.whole.cross_blocks);
}

/// The threaded assembly over split stores agrees with the sequential
/// fixed-width one, and the whole-model footprint does not depend on
/// the split.
#[cfg(feature = "threads")]
#[test]
fn threaded_assembly_matches_the_fixed_width_blocks() {
    let (mut fixed, mut var, x) = both(30);
    let n = x.len();
    let (mut g1, mut h1) = (vec![0.0; n], vec![0.0; n * n]);
    let c1 = fixed.calc_grad_hessian_dense(&x, &mut g1, &mut h1);
    let mut one = arael::threads::Context::new();
    var.begin_with_context(&mut one);
    let whole = one.sweeps().unwrap().whole;
    for threads in [2, 4] {
        let mut ctx = arael::threads::Context::new();
        ctx.set_threads(threads);
        var.begin_with_context(&mut ctx);
        assert_eq!(ctx.blocks_list::<VWorldBlocks>().map(|v| v.len()), Some(threads));
        let (mut g2, mut h2) = (vec![0.0; n], vec![0.0; n * n]);
        let c2 = var.calc_grad_hessian_dense_with_context(&x, &mut g2, &mut h2, &mut ctx);
        assert!(close(c1, c2, 1e-13), "{} stores: cost {} vs {}", threads, c1, c2);
        assert_close("grad", &g1, &g2, 1e-12);
        assert_close("hessian", &h1, &h2, 1e-12);
        let r = ctx.sweeps().unwrap();
        assert_eq!(r.whole, whole, "the whole does not depend on the split");
        let held = arael::store::StoreFootprint::sum(&r.held);
        assert_eq!(held.cross_blocks, whole.cross_blocks, "a cross block sits in one store");
    }
}
