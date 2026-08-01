// What the solver DECIDES, on the structures our benchmarks are made of.
//
// SparseFaer makes three choices from the model's structure alone -- what to
// marginalize, whether marginalizing pays, and how to order what is left --
// and each of them is worth a large factor. They are cheap to get wrong and
// invisible when they are: a wrong choice still returns the right answer, just
// slower. So these tests assert the DECISION, not the answer.
//
// The structures are miniatures of the benchmarks, sized so a solve is
// milliseconds. What matters is the shape, which is what the decisions are
// made from:
//
//   benchmarks/slam    a trajectory, landmarks seen from a bounded stretch
//                      of it  -> reduce, and S comes out banded
//   benchmarks/bal     cameras and points, everything sees everything
//                      -> reduce, and S comes out dense
//   benchmarks/pgo     a pose graph, no landmarks at all
//                      -> nothing to marginalize, factorize the whole system
//
// The measured cost of each mistake is in the doc comments below.

use arael::model::{CrossBlock, Param, SelfBlock};
use arael::refs::{self, Ref};
use arael::simple_lm::{
    lm_solve, LmConfig, ReducedOrdering, SchurPlan, SchurPolicy, SparseFaer,
};

// --- a 2D SLAM model: 3-parameter poses, 2-parameter landmarks ------------
// The same widths every slam2d demo has, and the same widths g2o's VertexSE2
// and VertexPointXY have.

#[arael::model]
#[arael(constraint(hb, {
    [(pose.x - pose.px) * 0.01, (pose.y - pose.py) * 0.01, (pose.th - pose.pth) * 0.01]
}))]
struct Pose {
    x: Param<f64>,
    y: Param<f64>,
    th: Param<f64>,
    px: f64,
    py: f64,
    pth: f64,
    hb: SelfBlock<Pose>,
}

#[arael::model]
struct Landmark {
    x: Param<f64>,
    y: Param<f64>,
    hb: SelfBlock<Landmark>,
}

#[arael::model]
#[arael(constraint(hb, {
    [b.x - a.x - odo.dx, b.y - a.y - odo.dy, b.th - a.th - odo.dth]
}))]
struct Odo {
    #[arael(ref = root.poses)]
    a: Ref<Pose>,
    #[arael(ref = root.poses)]
    b: Ref<Pose>,
    dx: f64,
    dy: f64,
    dth: f64,
    hb: CrossBlock<Pose, Pose>,
}

#[arael::model]
#[arael(constraint(hb, {
    [l.x - p.x - obs.dx, l.y - p.y - obs.dy]
}))]
struct Obs {
    #[arael(ref = root.poses)]
    p: Ref<Pose>,
    #[arael(ref = root.landmarks)]
    l: Ref<Landmark>,
    dx: f64,
    dy: f64,
    hb: CrossBlock<Pose, Landmark>,
}

#[arael::model]
#[arael(root)]
struct World {
    poses: refs::Vec<Pose>,
    landmarks: refs::Vec<Landmark>,
    odos: std::vec::Vec<Odo>,
    obs: std::vec::Vec<Obs>,
}

/// A scene generator with the one knob that decides everything: how far along
/// the trajectory a landmark is seen from.
///
/// `span = 0` means every landmark is seen from every pose (the
/// bundle-adjustment shape). `odometry = false` drops the pose chain, which is
/// what makes the poses marginalizable too.
struct Scene {
    poses: usize,
    landmarks: usize,
    span: usize,
    odometry: bool,
}

impl Scene {
    fn build(&self) -> World {
        let mut w = World {
            poses: refs::Vec::new(),
            landmarks: refs::Vec::new(),
            odos: std::vec::Vec::new(),
            obs: std::vec::Vec::new(),
        };
        for i in 0..self.poses {
            let (x, y) = (i as f64, 0.0);
            w.poses.push(Pose {
                x: Param::new(x + 0.01),
                y: Param::new(y - 0.01),
                th: Param::new(0.01),
                px: x,
                py: y,
                pth: 0.0,
                hb: SelfBlock::new(),
            });
        }
        for j in 0..self.landmarks {
            w.landmarks.push(Landmark {
                x: Param::new(j as f64 * 0.5 + 0.01),
                y: Param::new(1.0),
                hb: SelfBlock::new(),
            });
        }
        if self.odometry {
            for i in 1..self.poses {
                w.odos.push(Odo {
                    a: w.poses.ref_at(i - 1),
                    b: w.poses.ref_at(i),
                    dx: 1.0,
                    dy: 0.0,
                    dth: 0.0,
                    hb: CrossBlock::new(),
                });
            }
        }
        for j in 0..self.landmarks {
            // which poses see landmark j: a bounded stretch of the trajectory,
            // or all of them when span == 0
            let (first, last) = if self.span == 0 {
                (0, self.poses)
            } else {
                let first = (j * self.poses / self.landmarks.max(1))
                    .min(self.poses.saturating_sub(self.span));
                (first, (first + self.span).min(self.poses))
            };
            for i in first..last {
                let (lx, ly) = (j as f64 * 0.5, 1.0);
                w.obs.push(Obs {
                    p: w.poses.ref_at(i),
                    l: w.landmarks.ref_at(j),
                    dx: lx - i as f64,
                    dy: ly,
                    hb: CrossBlock::new(),
                });
            }
        }
        w
    }
}

/// Run the solver's analysis and hand back what it decided. One iteration is
/// enough: every choice is made on the first assembly.
fn decide(scene: &Scene, policy: SchurPolicy) -> SchurPlan {
    let mut w = scene.build();
    let mut params = std::vec::Vec::new();
    arael::simple_lm::RootProblem::serialize(&mut w, &mut params);
    let cfg = LmConfig { max_iters: 1, ..Default::default() };
    let mut solver = SparseFaer::new().with_policy(policy);
    lm_solve(&params, &mut solver, &mut w, &cfg).unwrap();
    solver.plan().expect("the first compute must leave a plan")
}

// --- benchmarks/slam: a trajectory with local landmarks -------------------

/// The slam benchmark's shape. Landmarks are seen from a bounded stretch of
/// the trajectory, so marginalizing them leaves a BANDED pose system -- and on
/// a banded matrix the natural order is already at the fill limit.
///
/// Getting the ordering wrong here costs real time: at 6000 poses AMD spends
/// 452 ms of symbolic factorization to find an ordering 0.2% WORSE than the
/// natural one (65 ms), and METIS would be 3.7x slower still.
#[test]
fn a_trajectory_with_local_landmarks_reduces_and_keeps_the_natural_order() {
    let plan = decide(
        &Scene { poses: 200, landmarks: 400, span: 8, odometry: true },
        SchurPolicy::default(),
    );

    assert!(plan.reduced, "the landmarks are marginalizable and worth it");
    assert_eq!(plan.eliminated_blocks, 400);
    assert_eq!(plan.eliminated_params, 800, "2 parameters per landmark");
    assert_eq!(plan.kept_params, 600, "3 parameters per pose");

    // A landmark reaches 8 poses, so S reaches ~8 poses back: banded.
    assert!(
        plan.kept_bandwidth < 600 / 5,
        "S should be banded, got half-bandwidth {} of 600",
        plan.kept_bandwidth
    );
    assert_eq!(
        plan.ordering,
        Some(ReducedOrdering::NaturalBanded),
        "a banded S must not pay for a fill-reducing pass"
    );
}

/// Same trajectory, but every landmark is seen from everywhere -- which is
/// what a loop closure does to the structure, and what the slam-300 benchmark
/// looks like (its S comes out 69% dense). Marginalizing still pays, but the
/// band is gone and the reason for the natural order changes: there is simply
/// no fill left for AMD to reduce.
#[test]
fn globally_visible_landmarks_leave_a_dense_reduced_system() {
    let plan = decide(
        &Scene { poses: 60, landmarks: 120, span: 0, odometry: true },
        SchurPolicy::default(),
    );

    assert!(plan.reduced);
    assert_eq!(plan.eliminated_blocks, 120);
    assert_eq!(
        plan.ordering,
        Some(ReducedOrdering::NaturalDense),
        "every pose is coupled to every other -- AMD has nothing to find"
    );
    assert_eq!(
        plan.kept_bandwidth,
        plan.kept_params,
        "a landmark seen from the first and last pose couples them: no band"
    );
}

/// The inverse shape: a HANDFUL of globally visible landmarks under a long
/// odometry chain (the plane benchmark's shared-plane scene). Eliminating
/// them couples every pose to every other -- a dense reduced system and an
/// expensive reduction -- while the whole system is nearly banded and cheap.
/// Auto must decline: taking this reduction cost 14x per iteration before
/// the gate priced the whole-system route.
#[test]
fn a_small_global_family_is_declined() {
    let plan = decide(
        &Scene { poses: 120, landmarks: 6, span: 0, odometry: true },
        SchurPolicy::default(),
    );
    assert!(
        !plan.reduced,
        "6 global landmarks leave a dense reduced system over an almost-banded \
         whole system; the reduction must be declined: {:?}",
        plan
    );
    assert_eq!(plan.eliminated_blocks, 0, "a decline marginalizes nothing");
    assert!(
        plan.flop_ratio.is_some(),
        "the cheap filter ran and could not call it obvious"
    );
    assert!(
        plan.route_flops.is_some_and(|(red, full)| red > full),
        "the exact pricing is the evidence: {:?}",
        plan.route_flops
    );
}

// --- benchmarks/bal: cameras and points, no trajectory --------------------
//
// A separate model, because detection reads the model's TYPE coupling graph,
// not its instances: the World above DECLARES a pose-pose constraint (Odo), so
// its poses are never marginalizable even in a scene that contains no odometry
// at all. That is by design -- the graph is static, which is what makes
// detection free -- but it means the bundle-adjustment shape needs a model
// that has no pose-pose constraint to declare.

#[arael::model]
struct Camera {
    x: Param<f64>,
    y: Param<f64>,
    th: Param<f64>,
    hb: SelfBlock<Camera>,
}

#[arael::model]
struct Point {
    x: Param<f64>,
    y: Param<f64>,
    hb: SelfBlock<Point>,
}

#[arael::model]
#[arael(constraint(hb, {
    [(pt.x - cam.x) * cos(cam.th) - view.u, (pt.y - cam.y) - view.v]
}))]
struct View {
    #[arael(ref = root.cameras)]
    cam: Ref<Camera>,
    #[arael(ref = root.points)]
    pt: Ref<Point>,
    u: f64,
    v: f64,
    hb: CrossBlock<Camera, Point>,
}

#[arael::model]
#[arael(root)]
struct Bal {
    cameras: refs::Vec<Camera>,
    points: refs::Vec<Point>,
    views: std::vec::Vec<View>,
}

fn decide_bal(cameras: usize, points: usize) -> SchurPlan {
    let mut b = Bal {
        cameras: refs::Vec::new(),
        points: refs::Vec::new(),
        views: std::vec::Vec::new(),
    };
    for i in 0..cameras {
        b.cameras.push(Camera {
            x: Param::new(i as f64 + 0.01),
            y: Param::new(0.01),
            th: Param::new(0.01),
            hb: SelfBlock::new(),
        });
    }
    for j in 0..points {
        b.points.push(Point {
            x: Param::new(j as f64 * 0.5 + 0.01),
            y: Param::new(1.01),
            hb: SelfBlock::new(),
        });
    }
    // every camera sees every point -- BAL's covisibility, in miniature
    for i in 0..cameras {
        for j in 0..points {
            b.views.push(View {
                cam: b.cameras.ref_at(i),
                pt: b.points.ref_at(j),
                u: (j as f64 * 0.5 - i as f64),
                v: 1.0,
                hb: CrossBlock::new(),
            });
        }
    }
    let mut params = std::vec::Vec::new();
    arael::simple_lm::RootProblem::serialize(&mut b, &mut params);
    let cfg = LmConfig { max_iters: 1, ..Default::default() };
    let mut solver = SparseFaer::new();
    lm_solve(&params, &mut solver, &mut b, &cfg).unwrap();
    solver.plan().expect("a plan")
}

/// The bundle-adjustment shape: cameras couple only to points and points only
/// to cameras, so BOTH families are legally marginalizable and only the counts
/// say which is worth it. The solver must take the one that removes more
/// parameters -- eliminating the smaller family would leave the bigger one to
/// factorize, which is backwards.
#[test]
fn bundle_adjustment_marginalizes_the_larger_family() {
    // Many points, few cameras: the points go. This is BAL's own shape.
    let plan = decide_bal(20, 200);
    assert!(plan.reduced);
    assert_eq!(plan.eliminated_blocks, 200, "the 200 points, not the 20 cameras");
    assert_eq!(plan.eliminated_params, 400, "2 parameters per point");
    assert_eq!(plan.kept_params, 60, "20 cameras x 3");

    // Flip the counts and the choice must flip with them. Nothing in the model
    // says which family is "the landmarks" -- only the parameter counts do.
    let plan = decide_bal(200, 20);
    assert!(plan.reduced);
    assert_eq!(plan.eliminated_blocks, 200, "now the cameras are the bigger family");
    assert_eq!(plan.eliminated_params, 600, "3 parameters per camera");
    assert_eq!(plan.kept_params, 40, "20 points x 2");
}

// --- benchmarks/pgo: a pose graph ----------------------------------------

/// The pose-graph benchmark has no landmarks: odometry and loop closures
/// couple poses to poses, so nothing is marginalizable and the whole system is
/// factorized. The solver must see that from the coupling graph and never
/// build the block machinery at all.
///
/// This is also where the ordering mistake is most expensive in the other
/// direction: on a real pose graph (m3500) the natural order is 48x slower
/// than AMD -- 101 ms against 2.1. The band rule must not fire here.
#[test]
fn a_pose_graph_has_nothing_to_marginalize() {
    let plan = decide(
        &Scene { poses: 300, landmarks: 0, span: 0, odometry: true },
        SchurPolicy::default(),
    );

    assert!(!plan.reduced);
    assert_eq!(plan.eliminated_blocks, 0);
    assert_eq!(plan.kept_params, 900, "the whole system: 300 poses x 3");
    assert_eq!(plan.ordering, None, "there is no reduced system to order");
    // Neither test ran: there was nothing to weigh.
    assert!(plan.flop_ratio.is_none() && plan.fill_ratio.is_none());
}

// --- the policies override the decision, and say so ----------------------

/// Never means never, even on the shape that most wants a reduction.
#[test]
fn never_declines_a_reduction_that_would_obviously_pay() {
    let scene = Scene { poses: 200, landmarks: 400, span: 8, odometry: true };
    assert!(decide(&scene, SchurPolicy::default()).reduced);

    let plan = decide(&scene, SchurPolicy::Never);
    assert!(!plan.reduced);
    assert_eq!(plan.eliminated_blocks, 0);
    assert_eq!(plan.kept_params, 600 + 800, "the whole system, poses and landmarks");
    assert_eq!(plan.ordering, None);
}

/// Force skips the analysis entirely -- neither the cheap filter nor the
/// ordering comparison leaves a trace in the plan.
#[test]
fn force_reduces_without_weighing_it() {
    let plan = decide(
        &Scene { poses: 200, landmarks: 400, span: 8, odometry: true },
        SchurPolicy::Force,
    );
    assert!(plan.reduced);
    assert!(
        plan.flop_ratio.is_none() && plan.fill_ratio.is_none(),
        "Force must not pay for any analysis"
    );
    // It still orders the reduced system by its shape.
    assert_eq!(plan.ordering, Some(ReducedOrdering::NaturalBanded));
}

/// The cheap filter exists so that an obvious reduction does not pay for the
/// exact comparison. On the slam shape it must fire -- and the plan says which
/// test decided, by which of the two ratios is present.
#[test]
fn an_obvious_reduction_skips_the_expensive_comparison() {
    let plan = decide(
        &Scene { poses: 200, landmarks: 400, span: 8, odometry: true },
        SchurPolicy::default(),
    );
    assert!(plan.reduced);
    assert!(
        plan.flop_ratio.is_some(),
        "the cheap filter always runs under Auto"
    );
    assert!(
        plan.fill_ratio.is_none(),
        "and having settled it, the ordering comparison must be skipped"
    );
}
