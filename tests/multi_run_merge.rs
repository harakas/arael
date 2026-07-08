//! Multi-run merge on the nested model tree (the slam2d_multi_demo shape),
//! reduced to a deterministic, noise-free case so recovery is exact.
//!
//! Two runs drive parallel tracks and observe the SAME two landmarks. Run A's
//! first pose is fixed as the global frame; run B has no fixed pose and is
//! pinned ONLY through the shared landmarks (`Frine.lm = root.landmarks`). With
//! exact bearings and good parallax the joint solve must recover every pose and
//! landmark of BOTH runs -- which can only happen if the shared landmarks merge
//! the two runs into one frame.

use arael::model::{Model, Param, SelfBlock, CrossBlock};
use arael::vect::vect2f;
use arael::matrix::matrix2f;
use arael::refs::{self, Ref};
use arael::simple_lm::LmConfig;

#[arael::model]
struct Pose {
    pos: Param<vect2f>,
    gamma: Param<f32>,
    delta_pos: vect2f,
    delta_gamma: f32,
    delta_pos_isigma: f32,
    delta_gamma_isigma: f32,
    hb_pose: SelfBlock<Pose, f32>,
}

#[arael::model]
#[arael(constraint(hb, {
    let local = matrix2sym::rotation(prev.gamma).transpose() * (cur.pos - prev.pos);
    [(local.x - cur.delta_pos.x) * cur.delta_pos_isigma,
     (local.y - cur.delta_pos.y) * cur.delta_pos_isigma,
     rad_diff(cur.gamma - prev.gamma, cur.delta_gamma) * cur.delta_gamma_isigma]
}))]
struct PosePair {
    #[arael(ref = parent.poses)] prev: Ref<Pose>,
    #[arael(ref = parent.poses)] cur: Ref<Pose>,
    hb: CrossBlock<Pose, Pose, f32>,
}

#[arael::model]
struct Landmark {
    pos: Param<vect2f>,
    hb: SelfBlock<Landmark, f32>,
}

#[arael::model]
#[arael(constraint(hb, {
    let world_angle = pose.gamma + frine.bearing;
    let aligned = matrix2sym::rotation(world_angle).transpose() * (lm.pos - pose.pos);
    [atan2(aligned.y, aligned.x) * frine.isigma]
}))]
struct Frine {
    #[arael(ref = parent.poses)] pose: Ref<Pose>,
    #[arael(ref = root.landmarks)] lm: Ref<Landmark>,
    bearing: f32,
    isigma: f32,
    hb: CrossBlock<Landmark, Pose, f32>,
}

#[arael::model]
struct Path {
    poses: refs::Deque<Pose>,
    pose_pairs: std::vec::Vec<PosePair>,
    frines: std::vec::Vec<Frine>,
}

#[arael::model]
#[arael(root, f32)]
struct Map {
    paths: std::vec::Vec<Path>,
    landmarks: refs::Vec<Landmark>,
}

fn bearing(pos: vect2f, gamma: f32, lm: vect2f) -> f32 {
    let local = matrix2f::rotation(gamma).transpose() * (lm - pos);
    local.y.atan2(local.x)
}

// Build one straight run along +x at height `y`, starting at x=0. Exact
// odometry deltas; each pose observes both landmarks. `fix_first` pins pose 0.
// `init_jitter` offsets the initial pose/landmark guesses so the solve has real
// work to do (and cannot pass by starting at the answer).
fn build_run(truth: &[vect2f], gt_lms: &[vect2f], map_lm: &[u32],
             fix_first: bool, init_jitter: vect2f) -> (Path, Vec<(usize, u32)>) {
    let mut path = Path {
        poses: refs::Deque::new(),
        pose_pairs: std::vec::Vec::new(),
        frines: std::vec::Vec::new(),
    };
    for (i, &t) in truth.iter().enumerate() {
        let init = if fix_first && i == 0 { t } else { t + init_jitter };
        let mut p = Pose {
            pos: Param::new(init),
            gamma: Param::new(0.0),
            delta_pos: if i == 0 { vect2f::new(0.0, 0.0) } else { truth[i] - truth[i - 1] },
            delta_gamma: 0.0,
            delta_pos_isigma: if i == 0 { 0.0 } else { 100.0 },
            delta_gamma_isigma: if i == 0 { 0.0 } else { 100.0 },
            hb_pose: SelfBlock::new(),
        };
        if fix_first && i == 0 {
            p.pos.optimize = false;
            p.gamma.optimize = false;
        }
        path.poses.push_back(p);
        if i > 0 {
            path.pose_pairs.push(PosePair {
                prev: path.poses.ref_at(i - 1),
                cur: path.poses.ref_at(i),
                hb: CrossBlock::new(),
            });
        }
    }
    // Frines: every pose observes every landmark (exact bearings).
    let mut sightings = Vec::new();
    for (pi, &t) in truth.iter().enumerate() {
        for (li, &gl) in gt_lms.iter().enumerate() {
            path.frines.push(Frine {
                pose: path.poses.ref_at(pi),
                lm: Ref::new(map_lm[li]),
                bearing: bearing(t, 0.0, gl),
                isigma: 500.0,
                hb: CrossBlock::new(),
            });
            sightings.push((pi, map_lm[li]));
        }
    }
    (path, sightings)
}

fn run_merge<F>(solve: F)
where
    F: Fn(&mut Map, &LmConfig<f32>) -> arael::simple_lm::LmResult<f32>,
{
    // Two shared landmarks with good parallax off a pair of parallel tracks.
    let gt_lms = [vect2f::new(3.0, 6.0), vect2f::new(6.0, -4.0)];
    let run_a_truth = [vect2f::new(0.0, 0.0), vect2f::new(2.0, 0.0), vect2f::new(4.0, 0.0)];
    let run_b_truth = [vect2f::new(0.0, 2.0), vect2f::new(2.0, 2.0), vect2f::new(4.0, 2.0)];

    // Shared landmarks live once on the Map; both runs reference them by Ref.
    let mut map = Map { paths: std::vec::Vec::new(), landmarks: refs::Vec::new() };
    let mut map_lm = Vec::new();
    for &gl in &gt_lms {
        // Initialise off-truth so the solve must actually triangulate/merge.
        map_lm.push(map.landmarks.len() as u32);
        map.landmarks.push(Landmark {
            pos: Param::new(gl + vect2f::new(1.5, -1.0)),
            hb: SelfBlock::new(),
        });
    }

    let (path_a, _) = build_run(&run_a_truth, &gt_lms, &map_lm, true, vect2f::new(0.6, 0.4));
    // Run B has NO fixed pose -- it is pinned only through the shared landmarks.
    let (path_b, _) = build_run(&run_b_truth, &gt_lms, &map_lm, false, vect2f::new(-0.5, 0.7));
    map.paths.push(path_a);
    map.paths.push(path_b);

    let cfg = LmConfig::<f32>::default();
    let result = solve(&mut map, &cfg);
    assert!(result.end_cost < 1e-4, "did not converge: cost {}", result.end_cost);

    // Both runs recovered to the common frame.
    let truth = [run_a_truth, run_b_truth];
    for (r, path) in map.paths.iter().enumerate() {
        for (n, &t) in path.poses.iter().zip(truth[r].iter()) {
            assert!((n.pos.value - t).norm() < 1e-2,
                "run {} pose {:?} != truth {:?}", r, n.pos.value, t);
        }
    }
    // Shared landmarks recovered -- proving run B's observations merged onto
    // the same landmarks run A anchored.
    for (lm, &gl) in map.landmarks.iter().zip(gt_lms.iter()) {
        assert!((lm.pos.value - gl).norm() < 1e-2,
            "landmark {:?} != truth {:?}", lm.pos.value, gl);
    }
}

#[test]
fn two_runs_merge_through_shared_landmarks_dense() {
    run_merge(|m, c| m.solve_dense(c));
}

#[test]
fn two_runs_merge_through_shared_landmarks_sparse() {
    run_merge(|m, c| m.solve_sparse(c));
}
