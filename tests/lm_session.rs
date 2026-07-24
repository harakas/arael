// LmSession keeps the structure a backend learns (pattern, position map,
// ordering, symbolic factorization, Schur plan) across solves. Everything it
// reuses is value-independent, so a warm solve must compute exactly what a
// cold one does -- the tests compare bit-for-bit, not to a tolerance. The
// session must also run cold again by itself when the parameter count
// changes, and invalidate() must make a same-size structure change safe.

use arael::model::{CrossBlock, Param, SelfBlock};
use arael::refs::{self, Ref};
use arael::simple_lm::{
    lm_solve, Band, BandError, CooMatrix, CscMatrix, Dense, LmConfig, LmProblem, LmResult,
    LmSession, LmSolver, RootProblem, SchurPolicy, SparseFaer,
};

// --- a model with marginalizable blocks: poses seeing nearby landmarks ---

#[arael::model]
#[arael(constraint(hb, {
    [(pose.x - pose.ax) * 0.1, (pose.y - pose.ay) * 0.1]
}))]
struct Pose {
    x: Param<f64>,
    y: Param<f64>,
    ax: f64,
    ay: f64,
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
    [b.x - a.x - odo.dx, b.y - a.y - odo.dy]
}))]
struct Odo {
    #[arael(ref = root.poses)]
    a: Ref<Pose>,
    #[arael(ref = root.poses)]
    b: Ref<Pose>,
    dx: f64,
    dy: f64,
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

// Enough poses that the reduced (pose) system's bandwidth is small relative
// to its size -- that is what makes the ordering NaturalBanded, which the
// narrow-band route needs.
const N_POSES: usize = 24;
const N_LANDMARKS: usize = 24;
const N_PARAMS: usize = 2 * (N_POSES + N_LANDMARKS);

/// Each landmark is seen only from a small window of poses, so the reduced
/// (pose) system is banded -- which is what lets the narrow-band route
/// activate in the route sweep below.
fn build(off: f64) -> World {
    let mut w = World {
        poses: refs::Vec::new(),
        landmarks: refs::Vec::new(),
        odos: std::vec::Vec::new(),
        obs: std::vec::Vec::new(),
    };
    let pose_true = |i: usize| (i as f64, 0.5 * i as f64);
    let lm_true = |j: usize| (j as f64, 2.0 + j as f64);
    for i in 0..N_POSES {
        let (tx, ty) = pose_true(i);
        w.poses.push(Pose {
            x: Param::new(tx + off),
            y: Param::new(ty - off),
            ax: tx,
            ay: ty,
            hb: SelfBlock::new(),
        });
    }
    for j in 0..N_LANDMARKS {
        let (tx, ty) = lm_true(j);
        w.landmarks.push(Landmark {
            x: Param::new(tx - off),
            y: Param::new(ty + off),
            hb: SelfBlock::new(),
        });
    }
    for i in 1..N_POSES {
        let (ax, ay) = pose_true(i - 1);
        let (bx, by) = pose_true(i);
        w.odos.push(Odo {
            a: w.poses.ref_at((i - 1)),
            b: w.poses.ref_at(i),
            dx: bx - ax,
            dy: by - ay,
            hb: CrossBlock::new(),
        });
    }
    for j in 0..N_LANDMARKS {
        let first = (j * N_POSES / N_LANDMARKS).min(N_POSES - 2);
        for i in first..first + 2 {
            let (px, py) = pose_true(i);
            let (lx, ly) = lm_true(j);
            w.obs.push(Obs {
                p: w.poses.ref_at(i),
                l: w.landmarks.ref_at(j),
                dx: lx - px,
                dy: ly - py,
                hb: CrossBlock::new(),
            });
        }
    }
    w
}

fn cfg() -> LmConfig<f64> {
    LmConfig { max_iters: 60, ..Default::default() }
}

/// Everything a solve computed, for bit-for-bit comparison.
fn fingerprint(r: &LmResult<f64>) -> (Vec<f64>, f64, f64, usize, usize, f64) {
    (r.x.clone(), r.start_cost, r.end_cost, r.iterations, r.accepted_iterations, r.final_lambda)
}

/// A warm second solve through a session computes exactly what a cold solve
/// of the same problem does.
fn warm_equals_cold<S: LmSolver<f64>>(mk: impl Fn() -> S, label: &str) {
    // The cold reference for the SECOND start point, on a fresh solver.
    let mut w_ref = build(0.15);
    let cold = w_ref.solve_with(&mut mk(), &cfg()).unwrap();

    let mut session = LmSession::new(mk());
    let mut w1 = build(0.05);
    let first = session.solve(&mut w1, &cfg()).unwrap();
    assert!(first.status.is_success(), "{}: first solve failed: {:?}", label, first.status);

    let mut w2 = build(0.15);
    let warm = session.solve(&mut w2, &cfg()).unwrap();

    assert_eq!(fingerprint(&warm), fingerprint(&cold), "{}: warm != cold", label);
    assert_eq!(warm.status, cold.status, "{}", label);

    // The optimized values were written back into the model.
    let mut written = Vec::new();
    RootProblem::serialize(&mut w2, &mut written);
    assert_eq!(written, warm.x, "{}: model not updated from the warm solve", label);
}

#[test]
fn warm_equals_cold_sparse_auto() {
    warm_equals_cold(SparseFaer::new, "sparse auto");
}

#[test]
fn warm_equals_cold_whole_system() {
    warm_equals_cold(|| SparseFaer::new().with_policy(SchurPolicy::Never), "whole system");
}

#[test]
fn warm_equals_cold_schur() {
    warm_equals_cold(|| SparseFaer::new().with_policy(SchurPolicy::Force), "schur");
}

#[test]
fn warm_equals_cold_narrow_band() {
    let mk = || SparseFaer::new().with_policy(SchurPolicy::Force).with_narrow_band(true);
    warm_equals_cold(mk, "narrow band");

    // Prove the sweep exercised the narrow-band route, not a fallback.
    let mut session = LmSession::new(mk());
    let mut w = build(0.05);
    session.solve(&mut w, &cfg()).unwrap();
    let plan = session.solver().plan().expect("a plan");
    assert!(plan.reduced && plan.narrow_band, "expected the narrow-band route: {:?}", plan);
}

#[test]
fn warm_equals_cold_dense() {
    warm_equals_cold(|| Dense, "dense");
}

#[test]
fn warm_equals_cold_band() {
    warm_equals_cold(|| Band::new(N_PARAMS - 1), "band");
}

#[test]
#[allow(deprecated)] // exercises the direct-CSC validation baseline
fn warm_equals_cold_sparse_direct() {
    use arael::simple_lm::SparseDirectCsc;
    warm_equals_cold(SparseDirectCsc::new, "sparse direct");
}

// --- a hand-built problem that counts how it was assembled ---

/// Quadratic with an explicit sparsity pattern: an anchor residual per
/// parameter (the diagonal) plus one residual `p[a] + p[b] - t` per pair (the
/// couplings). The pair list IS the pattern, so two Spy instances with
/// different pairs are same-size problems with different structures. Counts
/// COO (discovery) and indexed (warm) assemblies.
struct Spy {
    target: Vec<f64>,
    pairs: Vec<(usize, usize)>,
    coo_calls: usize,
    indexed_calls: usize,
}

impl Spy {
    fn new(target: &[f64], pairs: &[(usize, usize)]) -> Spy {
        Spy {
            target: target.to_vec(),
            pairs: pairs.to_vec(),
            coo_calls: 0,
            indexed_calls: 0,
        }
    }

    /// residuals: anchors `p[i] - target[i]`, pairs
    /// `p[a] + p[b] - (t[a] + t[b] + PAIR_BIAS)`. The bias makes the two
    /// families disagree, so the optimum is a least-squares compromise with a
    /// nonzero cost -- an exactly-solvable quadratic would hit cost 0.0 in
    /// one step and spend the rest of min_iters unable to improve on it.
    fn residuals(&self, p: &[f64]) -> Vec<f64> {
        let mut r: Vec<f64> = std::iter::zip(p, &self.target).map(|(x, t)| x - t).collect();
        r.extend(
            self.pairs
                .iter()
                .map(|&(a, b)| p[a] + p[b] - self.target[a] - self.target[b] - PAIR_BIAS),
        );
        r
    }
}

impl LmProblem<f64> for Spy {
    fn calc_cost(&mut self, p: &[f64]) -> f64 {
        self.residuals(p).iter().map(|r| r * r).sum::<f64>() * 0.5
    }

    fn calc_grad_hessian_dense(&mut self, _p: &[f64], _g: &mut [f64], _h: &mut [f64]) -> f64 {
        unimplemented!("sparse only")
    }
    fn calc_grad_hessian_band(&mut self, _p: &[f64], _g: &mut [f64], _b: &mut [f64], _kd: usize)
        -> Result<f64, BandError> {
        unimplemented!("sparse only")
    }
    fn calc_grad_hessian_sparse_direct(&mut self, _p: &[f64], _g: &mut [f64], _c: &mut CscMatrix<f64>) -> f64 {
        unimplemented!("sparse only")
    }

    fn calc_grad_hessian_sparse(&mut self, p: &[f64], grad: &mut [f64], coo: &mut CooMatrix<f64>) -> f64 {
        self.coo_calls += 1;
        grad.fill(0.0);
        for (i, (&x, &t)) in std::iter::zip(p, &self.target).enumerate() {
            grad[i] += x - t;
            coo.push(i as u32, i as u32, 1.0);
        }
        for &(a, b) in &self.pairs {
            let r = p[a] + p[b] - self.target[a] - self.target[b] - PAIR_BIAS;
            grad[a] += r;
            grad[b] += r;
            coo.push(a as u32, a as u32, 1.0);
            coo.push(a as u32, b as u32, 1.0);
            coo.push(b as u32, b as u32, 1.0);
        }
        self.calc_cost(p)
    }

    fn calc_grad_hessian_sparse_indexed(&mut self, p: &[f64], grad: &mut [f64], vals: &mut [f64], positions: &[usize]) -> f64 {
        self.indexed_calls += 1;
        grad.fill(0.0);
        vals.fill(0.0);
        let mut k = 0;
        for (i, (&x, &t)) in std::iter::zip(p, &self.target).enumerate() {
            grad[i] += x - t;
            vals[positions[k]] += 1.0;
            k += 1;
        }
        for &(a, b) in &self.pairs {
            let r = p[a] + p[b] - self.target[a] - self.target[b] - PAIR_BIAS;
            grad[a] += r;
            grad[b] += r;
            vals[positions[k]] += 1.0;
            vals[positions[k + 1]] += 1.0;
            vals[positions[k + 2]] += 1.0;
            k += 3;
        }
        self.calc_cost(p)
    }
}

const TARGET3: [f64; 3] = [1.0, 2.0, 3.0];
const PAIR_BIAS: f64 = 0.5;

/// The warm solve runs no discovery pass at all: assembly goes through the
/// cached position map from the first solve.
#[test]
fn warm_solve_skips_discovery() {
    let mut spy = Spy::new(&TARGET3, &[(0, 1), (1, 2)]);
    let mut session = LmSession::new(SparseFaer::new());

    let r1 = session.solve_x0(&[0.0; 3], &mut spy, &cfg()).unwrap();
    assert!(r1.status.is_success(), "{:?}", r1.status);
    assert_eq!(spy.coo_calls, 1, "the cold solve discovers the pattern once");

    let r2 = session.solve_x0(&[5.0, -1.0, 2.5], &mut spy, &cfg()).unwrap();
    assert!(r2.status.is_success(), "{:?}", r2.status);
    assert_eq!(spy.coo_calls, 1, "the warm solve must not rediscover");
    assert!(spy.indexed_calls > 0);

    // ... and it still computes the right thing: bit-identical to a cold
    // solve of the same problem from the same start.
    let mut fresh = Spy::new(&TARGET3, &[(0, 1), (1, 2)]);
    let cold = lm_solve(&[5.0, -1.0, 2.5], &mut SparseFaer::new(), &mut fresh, &cfg()).unwrap();
    assert_eq!(fingerprint(&r2), fingerprint(&cold));
}

/// The parameter-count backstop: a different count drops the caches, and the
/// session rediscovers and solves correctly. (This is a heuristic only --
/// same-count structure changes are NOT caught; see invalidate below.)
#[test]
fn parameter_count_change_runs_cold() {
    let mut session = LmSession::new(SparseFaer::new());

    let mut p3 = Spy::new(&TARGET3, &[(0, 1)]);
    session.solve_x0(&[0.0; 3], &mut p3, &cfg()).unwrap();

    let mut p4 = Spy::new(&[1.0, 2.0, 3.0, 4.0], &[(0, 1), (2, 3)]);
    let r = session.solve_x0(&[0.0; 4], &mut p4, &cfg()).unwrap();
    assert_eq!(p4.coo_calls, 1, "the size change must force a fresh discovery");

    let mut fresh = Spy::new(&[1.0, 2.0, 3.0, 4.0], &[(0, 1), (2, 3)]);
    let cold = lm_solve(&[0.0; 4], &mut SparseFaer::new(), &mut fresh, &cfg()).unwrap();
    assert_eq!(fingerprint(&r), fingerprint(&cold));
}

/// invalidate() makes a same-size structure change safe: the next solve
/// rediscovers and matches a cold solve of the new problem.
#[test]
fn invalidate_handles_a_same_size_structure_change() {
    let mut session = LmSession::new(SparseFaer::new());

    let mut p1 = Spy::new(&TARGET3, &[(0, 1), (1, 2)]);
    session.solve_x0(&[0.0; 3], &mut p1, &cfg()).unwrap();

    // Same parameter count, different coupling pattern.
    let mut p2 = Spy::new(&TARGET3, &[(0, 2)]);
    session.invalidate();
    let r = session.solve_x0(&[0.0; 3], &mut p2, &cfg()).unwrap();
    assert_eq!(p2.coo_calls, 1, "invalidate must force a fresh discovery");

    let mut fresh = Spy::new(&TARGET3, &[(0, 2)]);
    let cold = lm_solve(&[0.0; 3], &mut SparseFaer::new(), &mut fresh, &cfg()).unwrap();
    assert_eq!(fingerprint(&r), fingerprint(&cold));
}

/// An empty problem through a session: the empty result, no state disturbed.
#[test]
fn empty_solve_is_a_no_op() {
    let mut spy = Spy::new(&[], &[]);
    let mut session = LmSession::new(SparseFaer::new());
    let r = session.solve_x0(&[], &mut spy, &cfg()).unwrap();
    assert!(r.x.is_empty());
    assert!(r.status.is_success());
    assert_eq!(spy.coo_calls, 0);
}

/// Changing num_threads between warm solves re-sizes the factorization
/// scratch (it is sized per Par); the solve must run and agree with the
/// single-threaded answer.
#[cfg(feature = "rayon")]
#[test]
fn thread_count_change_between_warm_solves() {
    let mut session = LmSession::new(SparseFaer::new());
    let mut w1 = build(0.05);
    session.solve(&mut w1, &cfg()).unwrap();

    let mut w2 = build(0.15);
    let threaded = LmConfig { num_threads: 2, ..cfg() };
    let warm = session.solve(&mut w2, &threaded).unwrap();
    assert!(warm.status.is_success(), "{:?}", warm.status);

    let mut w_ref = build(0.15);
    let cold = w_ref.solve_with(&mut SparseFaer::new(), &cfg()).unwrap();
    for (i, (a, b)) in std::iter::zip(&warm.x, &cold.x).enumerate() {
        assert!((a - b).abs() < 1e-8, "param {}: {} vs {}", i, a, b);
    }
}
