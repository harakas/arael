// Rotation-cache substitutions in the parent-refs cross form: a body
// reading `parent.a.ea.rotation_matrix()` must hit the same precomputed
// rotation caches as the own-refs form, and both builds must agree on
// cost, gradient, Hessian, and the solved rotations across recentering
// steps (EulerAngleParam re-centers every accepted step).

use arael::matrix::matrix3d;
use arael::model::{CrossBlock, EulerAngleParam, SelfBlock};
use arael::refs::{self, Ref};
use arael::simple_lm::{self, LmConfig, LmProblem, RootProblem};
use arael::vect::vect3d;

const TOL: f64 = 1e-9;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs()))
}

// Shared entity: a rotation with a soft prior on the angles (anchors
// the solve).
#[arael::model]
#[arael(constraint(hb, {
    let d = pose.ea - pose.target;
    [d.x * pose.pw, d.y * pose.pw, d.z * pose.pw]
}))]
struct Pose {
    ea: EulerAngleParam<f64>,
    target: vect3d,
    pw: f64,
    hb: SelfBlock<Pose>,
}

// Build A: per-instance CrossBlock, own refs.
#[arael::model]
#[arael(constraint(hb, {
    let d = b.ea.rotation_matrix() - a.ea.rotation_matrix() * link.rel;
    [d[0][0] * link.w, d[0][1] * link.w, d[0][2] * link.w,
     d[1][0] * link.w, d[1][1] * link.w, d[1][2] * link.w,
     d[2][0] * link.w, d[2][1] * link.w, d[2][2] * link.w]
}))]
struct Link {
    #[arael(ref = root.poses)] a: Ref<Pose>,
    #[arael(ref = root.poses)] b: Ref<Pose>,
    rel: matrix3d,
    w: f64,
    hb: CrossBlock<Pose, Pose>,
}

#[arael::model]
#[arael(root)]
struct NetA {
    poses: refs::Arena<Pose>,
    links: std::vec::Vec<Link>,
}

// Build B: parent-held refs, shared parent CrossBlock; the same body
// through `parent.a` / `pp.b`.
#[arael::model]
#[arael(constraint(parent.hb, parent = pp, {
    let d = pp.b.ea.rotation_matrix() - parent.a.ea.rotation_matrix() * plink.rel;
    [d[0][0] * plink.w, d[0][1] * plink.w, d[0][2] * plink.w,
     d[1][0] * plink.w, d[1][1] * plink.w, d[1][2] * plink.w,
     d[2][0] * plink.w, d[2][1] * plink.w, d[2][2] * plink.w]
}))]
struct PLink {
    rel: matrix3d,
    w: f64,
}

#[arael::model]
struct PosePair {
    #[arael(ref = root.poses)] a: Ref<Pose>,
    #[arael(ref = root.poses)] b: Ref<Pose>,
    links: std::vec::Vec<PLink>,
    hb: CrossBlock<Pose, Pose>,
}

#[arael::model]
#[arael(root)]
struct NetB {
    poses: refs::Arena<Pose>,
    pairs: std::vec::Vec<PosePair>,
}

// One data table drives both builds.
type PoseData = (vect3d, f64); // initial + prior target angles, weight
type LinkData = (vect3d, f64); // relative rotation (as angles), weight
struct Data {
    poses: Vec<PoseData>,
    pairs: Vec<((usize, usize), Vec<LinkData>)>,
}

fn data() -> Data {
    Data {
        poses: vec![
            (vect3d::new(0.1, -0.2, 0.3), 1.0),
            (vect3d::new(0.4, 0.1, -0.5), 0.3),
            (vect3d::new(-0.3, 0.25, 0.8), 0.2),
        ],
        pairs: vec![
            ((0, 1), vec![
                (vect3d::new(0.30, 0.28, -0.85), 1.5),
                (vect3d::new(0.28, 0.32, -0.78), 0.7),
            ]),
            ((0, 2), vec![
                (vect3d::new(-0.42, 0.44, 0.52), 0.8),
                (vect3d::new(-0.38, 0.46, 0.48), 1.1),
            ]),
            ((1, 2), vec![
                (vect3d::new(-0.72, 0.18, 1.30), 1.2),
            ]),
        ],
    }
}

fn poses_of(d: &Data) -> (refs::Arena<Pose>, Vec<Ref<Pose>>) {
    let mut poses = refs::Arena::new();
    let mut prefs = Vec::new();
    for &(ea, pw) in &d.poses {
        prefs.push(poses.push(Pose {
            ea: EulerAngleParam::new(ea), target: ea, pw, hb: SelfBlock::new(),
        }));
    }
    (poses, prefs)
}

fn build_a(d: &Data) -> NetA {
    let (poses, prefs) = poses_of(d);
    let mut links = Vec::new();
    for ((ia, ib), ls) in &d.pairs {
        for &(rel, w) in ls {
            links.push(Link {
                a: prefs[*ia], b: prefs[*ib],
                rel: matrix3d::rotation_from_euler_angles(rel), w,
                hb: CrossBlock::new(),
            });
        }
    }
    NetA { poses, links }
}

fn build_b(d: &Data) -> NetB {
    let (poses, prefs) = poses_of(d);
    let mut pairs = Vec::new();
    for ((ia, ib), ls) in &d.pairs {
        let links = ls.iter().map(|&(rel, w)| PLink {
            rel: matrix3d::rotation_from_euler_angles(rel), w,
        }).collect();
        pairs.push(PosePair {
            a: prefs[*ia], b: prefs[*ib], links, hb: CrossBlock::new(),
        });
    }
    NetB { poses, pairs }
}

// Cost/gradient/Hessian agreement at the initial point.
#[test]
fn parent_refs_euler_matches_own_refs() {
    let d = data();
    let mut ma = build_a(&d);
    let mut mb = build_b(&d);

    let mut xa = Vec::new();
    RootProblem::serialize(&mut ma, &mut xa);
    let mut xb = Vec::new();
    RootProblem::serialize(&mut mb, &mut xb);
    assert_eq!(xa.len(), xb.len());
    let n = xa.len();

    let ca = ma.calc_cost(&xa);
    let cb = mb.calc_cost(&xb);
    assert!(close(ca, cb, TOL), "cost: {} != {}", ca, cb);

    let mut ga = vec![0.0; n];
    let mut ha = vec![0.0; n * n];
    ma.calc_grad_hessian_dense(&xa, &mut ga, &mut ha);
    let mut gb = vec![0.0; n];
    let mut hb = vec![0.0; n * n];
    mb.calc_grad_hessian_dense(&xb, &mut gb, &mut hb);
    for i in 0..n {
        assert!(close(ga[i], gb[i], TOL), "grad[{i}]: {} != {}", ga[i], gb[i]);
        for j in 0..n {
            assert!(close(ha[i * n + j], hb[i * n + j], TOL),
                "H[{i},{j}]: {} != {}", ha[i * n + j], hb[i * n + j]);
        }
    }

    let da = ma.check_gradients(&xa);
    assert!(da.is_clean(), "build A gradient check:\n{}", da);
    let db = mb.check_gradients(&xb);
    assert!(db.is_clean(), "build B gradient check:\n{}", db);
}

// Full solves agree across recentering steps (advance() folds the delta
// into ref_rotation after every accepted step; the caches must follow).
#[test]
fn parent_refs_euler_solve_matches_own_refs() {
    let d = data();
    let mut ma = build_a(&d);
    let mut mb = build_b(&d);

    let cfg = LmConfig { max_iters: 50, ..Default::default() };
    let mut xa = Vec::new();
    RootProblem::serialize(&mut ma, &mut xa);
    let ra = simple_lm::solve(&xa, &mut ma, &cfg).unwrap();
    ma.deserialize(&ra.x);

    let mut xb = Vec::new();
    RootProblem::serialize(&mut mb, &mut xb);
    let rb = simple_lm::solve(&xb, &mut mb, &cfg).unwrap();
    mb.deserialize(&rb.x);

    assert!(ra.iterations > 1, "solve must take steps (recentering exercised)");
    assert!(close(ra.end_cost, rb.end_cost, 1e-7),
        "final cost: {} != {}", ra.end_cost, rb.end_cost);
    for (pa, pb) in ma.poses.iter().zip(mb.poses.iter()) {
        let (ra, rb) = (&pa.ea.rotation_matrix, &pb.ea.rotation_matrix);
        let err: f64 = (0..3).map(|i| (0..3).map(|j|
            (ra[i][j] - rb[i][j]).abs()).sum::<f64>()).sum();
        assert!(err < 1e-6, "solved rotations differ, err={}", err);
    }
}
