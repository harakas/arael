// arael runners: identical model in f64 and f32 (separate structs --
// the precision is a compile-time property of the generated code).

use crate::g2o::{Dataset, PoseIn};
use arael::model::{Model, Param, SelfBlock, CrossBlock};
use arael::refs::{self, Ref};
use arael::vect::{vect2d, vect2f};

// ---------------------------------------------------------------- f64

#[arael::model]
#[arael(constraint(hb, guard = self.has_prior, {
    [pose2.pos.x - pose2.prior.x,
     pose2.pos.y - pose2.prior.y,
     pose2.th - pose2.prior_th]
}))]
struct Pose2 {
    pos: Param<vect2d>,
    th: Param<f64>,
    prior: vect2d,
    prior_th: f64,
    has_prior: bool,
    hb: SelfBlock<Pose2>,
}

#[arael::model]
#[arael(constraint(hb, {
    let local = matrix2sym::rotation(b.th).transpose()
        * (a.pos + matrix2sym::rotation(a.th) * edge.delta - b.pos);
    [local.x * edge.wt,
     local.y * edge.wt,
     rad_diff(a.th + edge.dth, b.th) * edge.wr]
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

// ---------------------------------------------------------------- f32

#[arael::model]
#[arael(constraint(hb, guard = self.has_prior, {
    [pose2f.pos.x - pose2f.prior.x,
     pose2f.pos.y - pose2f.prior.y,
     pose2f.th - pose2f.prior_th]
}))]
struct Pose2F {
    pos: Param<vect2f>,
    th: Param<f32>,
    prior: vect2f,
    prior_th: f32,
    has_prior: bool,
    hb: SelfBlock<Pose2F, f32>,
}

#[arael::model]
#[arael(constraint(hb, {
    let local = matrix2sym::rotation(b.th).transpose()
        * (a.pos + matrix2sym::rotation(a.th) * edgef.delta - b.pos);
    [local.x * edgef.wt,
     local.y * edgef.wt,
     rad_diff(a.th + edgef.dth, b.th) * edgef.wr]
}))]
struct EdgeF {
    #[arael(ref = root.poses)]
    a: Ref<Pose2F>,
    #[arael(ref = root.poses)]
    b: Ref<Pose2F>,
    delta: vect2f,
    dth: f32,
    wt: f32,
    wr: f32,
    hb: CrossBlock<Pose2F, Pose2F, f32>,
}

#[arael::model]
#[arael(root, f32)]
struct GraphF {
    poses: refs::Vec<Pose2F>,
    edges: std::vec::Vec<EdgeF>,
}

// ---------------------------------------------------------------- runners

fn build_f64(ds: &Dataset) -> Graph {
    let mut g = Graph { poses: refs::Vec::new(), edges: std::vec::Vec::new() };
    for (i, p) in ds.poses.iter().enumerate() {
        g.poses.push(Pose2 {
            pos: Param::new(vect2d::new(p.x, p.y)),
            th: Param::new(p.th),
            prior: vect2d::new(p.x, p.y),
            prior_th: p.th,
            has_prior: i == 0,
            hb: SelfBlock::new(),
        });
    }
    for e in &ds.edges {
        g.edges.push(Edge {
            a: Ref::new(e.a),
            b: Ref::new(e.b),
            delta: vect2d::new(e.dx, e.dy),
            dth: e.dth,
            wt: e.wt,
            wr: e.wr,
            hb: CrossBlock::new(),
        });
    }
    g
}

fn build_f32(ds: &Dataset) -> GraphF {
    let mut g = GraphF { poses: refs::Vec::new(), edges: std::vec::Vec::new() };
    for (i, p) in ds.poses.iter().enumerate() {
        g.poses.push(Pose2F {
            pos: Param::new(vect2f::new(p.x as f32, p.y as f32)),
            th: Param::new(p.th as f32),
            prior: vect2f::new(p.x as f32, p.y as f32),
            prior_th: p.th as f32,
            has_prior: i == 0,
            hb: SelfBlock::new(),
        });
    }
    for e in &ds.edges {
        g.edges.push(EdgeF {
            a: Ref::new(e.a),
            b: Ref::new(e.b),
            delta: vect2f::new(e.dx as f32, e.dy as f32),
            dth: e.dth as f32,
            wt: e.wt as f32,
            wr: e.wr as f32,
            hb: CrossBlock::new(),
        });
    }
    g
}

pub struct RunOut {
    pub solve_ms: f64,
    pub first_iter_ms: f64,
    pub iterations: usize,
    pub poses: Vec<PoseIn>,
}

// Termination: same criterion class and thresholds as the tiny-solver and
// GTSAM defaults (stop when a step improves the cost by less than 1e-5
// absolute or 1e-5 relative). patience = 1 so ONE small step terminates,
// matching how both other systems check it.
fn cfg64(max_iters: usize) -> arael::simple_lm::LmConfig<f64> {
    arael::simple_lm::LmConfig {
        abs_precision: 1e-5,
        rel_precision: 1e-5,
        patience: 1,
        max_iters,
        ..Default::default()
    }
}

fn cfg32(max_iters: usize) -> arael::simple_lm::LmConfig<f32> {
    arael::simple_lm::LmConfig {
        abs_precision: 1e-5,
        rel_precision: 1e-5,
        patience: 1,
        max_iters,
        ..Default::default()
    }
}

pub fn run_f64(ds: &Dataset) -> RunOut {
    let mut g = build_f64(ds);
    let mut params: Vec<f64> = Vec::new();
    g.serialize64(&mut params);

    // First-iteration time: a fresh solve capped at one iteration
    // (setup + first assembly + symbolic + numeric factorization + step).
    let t0 = std::time::Instant::now();
    let _ = arael::simple_lm::solve_sparse_faer(&params, &mut g, &cfg64(1));
    let first_iter_ms = t0.elapsed().as_secs_f64() * 1e3;

    let t0 = std::time::Instant::now();
    let result = arael::simple_lm::solve_sparse_faer(&params, &mut g, &cfg64(100));
    let solve_ms = t0.elapsed().as_secs_f64() * 1e3;
    g.deserialize64(&result.x);
    let poses = g.poses.iter()
        .map(|p| PoseIn { x: p.pos.value.x, y: p.pos.value.y, th: p.th.value })
        .collect();
    RunOut { solve_ms, first_iter_ms, iterations: result.iterations, poses }
}

pub fn run_f32(ds: &Dataset) -> RunOut {
    let mut g = build_f32(ds);
    let mut params: Vec<f32> = Vec::new();
    g.serialize32(&mut params);

    let t0 = std::time::Instant::now();
    let _ = arael::simple_lm::solve_sparse_faer_f32(&params, &mut g, &cfg32(1));
    let first_iter_ms = t0.elapsed().as_secs_f64() * 1e3;

    let t0 = std::time::Instant::now();
    let result = arael::simple_lm::solve_sparse_faer_f32(&params, &mut g, &cfg32(100));
    let solve_ms = t0.elapsed().as_secs_f64() * 1e3;
    g.deserialize32(&result.x);
    let poses = g.poses.iter()
        .map(|p| PoseIn { x: p.pos.value.x as f64, y: p.pos.value.y as f64, th: p.th.value as f64 })
        .collect();
    RunOut { solve_ms, first_iter_ms, iterations: result.iterations, poses }
}
