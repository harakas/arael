// AngleParam: the cached 2D rotation matrix and its Jacobian must be used
// by the generated constraint code (verified by `cargo expand`), and the
// assembled gradient must match finite differences.

use arael::angle::AngleParam;
use arael::model::{CrossBlock, Param, SelfBlock};
use arael::refs::{self, Ref};
use arael::simple_lm::LmProblem;
use arael::utils::rad_diff;
use arael::vect::vect2d;

// A 2D pose: position plus an AngleParam heading. The prior anchors pose 0.
#[arael::model]
#[arael(constraint(hb, guard = self.has_prior, {
    [pose2.pos.x - pose2.prior.x,
     pose2.pos.y - pose2.prior.y,
     pose2.rot.angle - pose2.prior_th]
}))]
struct Pose2 {
    pos: Param<vect2d>,
    rot: AngleParam<f64>,
    prior: vect2d,
    prior_th: f64,
    has_prior: bool,
    hb: SelfBlock<Pose2>,
}

// A relative SE2 measurement: rotation read through the cached matrices.
#[arael::model]
#[arael(constraint(hb, {
    let local = b.rot.rotation_matrix.transpose()
        * (a.pos + a.rot.rotation_matrix * edge.delta - b.pos);
    [local.x * edge.wt,
     local.y * edge.wt,
     rad_diff(a.rot.angle + edge.dth, b.rot.angle) * edge.wr]
}))]
struct Edge {
    #[arael(ref = root.poses)]
    a: Ref<Pose2>,
    #[arael(ref = root.poses)]
    b: Ref<Pose2>,
    delta: vect2d,
    dth: f64,
    wt: f64,
    wr: f64,
    hb: CrossBlock<Pose2, Pose2>,
}

#[arael::model]
#[arael(root)]
struct Graph {
    poses: refs::Vec<Pose2>,
    edges: std::vec::Vec<Edge>,
}

fn build() -> Graph {
    // A 4-pose square loop with a slight perturbation to leave residual.
    let gt = [(0.0, 0.0, 0.0), (1.0, 0.0, 1.5708), (1.0, 1.0, 3.1416), (0.0, 1.0, -1.5708)];
    let mut poses = refs::Vec::new();
    for (i, &(x, y, th)) in gt.iter().enumerate() {
        poses.push(Pose2 {
            pos: Param::new(vect2d::new(x + 0.03 * i as f64, y - 0.02 * i as f64)),
            rot: AngleParam::new(th + 0.01 * i as f64),
            prior: vect2d::new(x, y),
            prior_th: th,
            has_prior: i == 0,
            hb: SelfBlock::new(),
        });
    }
    let mut edges = std::vec::Vec::new();
    let mut add = |a: usize, b: usize| {
        let (ax, ay, ath) = gt[a];
        let (bx, by, bth) = gt[b];
        let (s, c) = ath.sin_cos();
        // delta = R(ath)^T (p_b - p_a); dth = wrap(bth - ath).
        let dx = bx - ax;
        let dy = by - ay;
        edges.push(Edge {
            a: refs::Vec::ref_at(&poses, a as u32),
            b: refs::Vec::ref_at(&poses, b as u32),
            delta: vect2d::new(c * dx + s * dy, -s * dx + c * dy),
            dth: rad_diff(bth, ath),
            wt: 1.0,
            wr: 1.0,
            hb: CrossBlock::new(),
        });
    };
    add(0, 1);
    add(1, 2);
    add(2, 3);
    add(3, 0);
    Graph { poses, edges }
}

#[test]
fn gradient_matches_finite_difference() {
    let mut g = build();
    let mut params = std::vec::Vec::new();
    g.serialize64(&mut params);
    let d = g.check_gradients(&params);
    assert!(d.is_clean(), "assembled gradient disagrees with FD:\n{}", d);
}

#[test]
fn solves_the_loop() {
    let mut g = build();
    let mut params = std::vec::Vec::new();
    g.serialize64(&mut params);
    let cfg = arael::simple_lm::LmConfig::default();
    let r = arael::simple_lm::solve_dense(&params, &mut g, &cfg).unwrap();
    assert!(r.status.is_success(), "{:?}", r.status);
    assert!(r.end_cost < r.start_cost * 1e-3,
        "cost {} -> {}", r.start_cost, r.end_cost);
}
