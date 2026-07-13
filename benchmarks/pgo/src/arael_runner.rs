// arael runners: identical model in f64 and f32 (separate structs --
// the precision is a compile-time property of the generated code).

use crate::arael_pipeline::{run, Model as Pipeline, RunOut as Out};
use crate::g2o::{Dataset, PoseIn};
use arael::model::{Param, SelfBlock, CrossBlock};
use arael::refs::{self, Ref};
use arael::vect::{vect2d, vect2f};

// ---------------------------------------------------------------- f64

#[arael::model]
#[arael(constraint(hb, guard = self.has_prior, {
    [pose2.pos.x - pose2.prior.x,
     pose2.pos.y - pose2.prior.y,
     pose2.th - pose2.prior_th]
}))]
#[derive(Clone)]
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
#[derive(Clone)]
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
#[derive(Clone)]
pub struct Graph {
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
#[derive(Clone)]
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
#[derive(Clone)]
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
#[derive(Clone)]
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
        let a = g.poses.ref_at(e.a);
        let b = g.poses.ref_at(e.b);
        g.edges.push(Edge {
            a,
            b,
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
        let a = g.poses.ref_at(e.a);
        let b = g.poses.ref_at(e.b);
        g.edges.push(EdgeF {
            a,
            b,
            delta: vect2f::new(e.dx as f32, e.dy as f32),
            dth: e.dth as f32,
            wt: e.wt as f32,
            wr: e.wr as f32,
            hb: CrossBlock::new(),
        });
    }
    g
}

/// PGO_ORDERING=nd factorizes the whole system under a nested-dissection
/// ordering instead of AMD. On parking-garage -- a 3D pose graph dense enough
/// that AMD's ordering leaves faer no supernodes worth having -- it is worth a
/// lot; on the sparser graphs it is not. Default stays AMD.
pub(crate) fn ordering() -> arael::simple_lm::FaerOrdering {
    if std::env::var("PGO_ORDERING").as_deref() == Ok("nd") {
        arael::simple_lm::FaerOrdering::NestedDissection
    } else {
        arael::simple_lm::FaerOrdering::Auto
    }
}

/// TIMING=1 prints arael's internal breakdown for every solve it runs. That
/// includes the probes, so one round of one system emits several lines: the
/// discarded warmup plus PROBE_SUBROUNDS passes of the one- and two-iteration
/// probes, then the real solve.
///
/// Only the printing is gated. `gather_timing` stays on either way, so the
/// timing numbers do not depend on whether they are being looked at.
fn timing_enabled() -> bool {
    std::env::var("TIMING").is_ok()
}

fn print_timing<T>(r: &arael::simple_lm::LmResult<T>) {
    if !timing_enabled() {
        return;
    }
    if let Some(t) = &r.timing {
        eprintln!(
            "  [timing] total {:.1} ms = assembly {:.1} + linear solve {:.1} (first assembly {:.1}), {} iters",
            t.total.as_secs_f64() * 1e3,
            t.assembly.as_secs_f64() * 1e3,
            t.linear_solve.as_secs_f64() * 1e3,
            t.first_assembly.as_secs_f64() * 1e3,
            r.iterations,
        );
    }
}

/// The two solvers, one per scalar. This is the one thing the generic pipeline
/// cannot pick for itself.
pub fn solve_f64<P: arael::simple_lm::LmProblem<f64>>(
    params: &[f64],
    p: &mut P,
    cfg: &arael::simple_lm::LmConfig<f64>,
) -> arael::simple_lm::LmResult<f64> {
    let mut solver = arael::simple_lm::SparseFaer::new().with_ordering(ordering());
    let r = arael::simple_lm::lm_solve(params, &mut solver, p, cfg);
    print_timing(&r);
    r
}

pub fn solve_f32<P: arael::simple_lm::LmProblem<f32>>(
    params: &[f32],
    p: &mut P,
    cfg: &arael::simple_lm::LmConfig<f32>,
) -> arael::simple_lm::LmResult<f32> {
    let mut solver = arael::simple_lm::SparseFaerF32::new().with_ordering(ordering());
    let r = arael::simple_lm::lm_solve(params, &mut solver, p, cfg);
    print_timing(&r);
    r
}

// Initial damping, problem-appropriate for well-initialized 2D pose graphs (the
// LmConfig docs recommend a small initial_lambda for these); see the README's
// initial-damping policy.
const LAMBDA0_2D: f64 = 1e-8;

impl Pipeline for Graph {
    type Scalar = f64;
    type Dataset = Dataset;
    type Pose = PoseIn;
    fn lambda0() -> f64 { LAMBDA0_2D }
    fn build(ds: &Dataset) -> Self { build_f64(ds) }
    fn serialize(&mut self, out: &mut Vec<f64>) { self.serialize64(out); }
    fn deserialize(&mut self, x: &[f64]) { self.deserialize64(x); }
    fn poses(&self) -> Vec<PoseIn> {
        self.poses.iter()
            .map(|p| PoseIn { x: p.pos.value.x, y: p.pos.value.y, th: p.th.value })
            .collect()
    }
    fn solve(params: &[f64], m: &mut Self, cfg: &arael::simple_lm::LmConfig<f64>)
        -> arael::simple_lm::LmResult<f64> { solve_f64(params, m, cfg) }
}

impl Pipeline for GraphF {
    type Scalar = f32;
    type Dataset = Dataset;
    type Pose = PoseIn;
    fn lambda0() -> f64 { LAMBDA0_2D }
    fn build(ds: &Dataset) -> Self { build_f32(ds) }
    fn serialize(&mut self, out: &mut Vec<f32>) { self.serialize32(out); }
    fn deserialize(&mut self, x: &[f32]) { self.deserialize32(x); }
    fn poses(&self) -> Vec<PoseIn> {
        self.poses.iter()
            .map(|p| PoseIn {
                x: p.pos.value.x as f64,
                y: p.pos.value.y as f64,
                th: p.th.value as f64,
            })
            .collect()
    }
    fn solve(params: &[f32], m: &mut Self, cfg: &arael::simple_lm::LmConfig<f32>)
        -> arael::simple_lm::LmResult<f32> { solve_f32(params, m, cfg) }
}

pub type RunOut = Out<PoseIn>;

pub fn run_f64(ds: &Dataset) -> RunOut { run::<Graph>(ds) }
pub fn run_f32(ds: &Dataset) -> RunOut { run::<GraphF>(ds) }

