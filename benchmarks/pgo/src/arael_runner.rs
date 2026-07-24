// arael runners: one generic model, instantiated by the f64 and f32
// roots (the precision is a compile-time property of the generated code).

use bench_harness::arael::{run, Model as Pipeline};
use bench_harness::table::Row;
use crate::g2o::{Dataset, PoseIn};
use arael::model::{Param, SelfBlock, CrossBlock};
use arael::refs::{self, Ref};
use arael::utils::Float;
use arael::vect::vect2;

#[arael::model]
#[arael(constraint(hb, guard = self.has_prior, {
    [pose2.pos.x - pose2.prior.x,
     pose2.pos.y - pose2.prior.y,
     pose2.th - pose2.prior_th]
}))]
#[derive(Clone)]
struct Pose2<T: Float> {
    pos: Param<vect2<T>>,
    th: Param<T>,
    prior: vect2<T>,
    prior_th: T,
    has_prior: bool,
    hb: SelfBlock<Pose2<T>, T>,
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
struct Edge<T: Float> {
    #[arael(ref = root.poses)]
    a: Ref<Pose2<T>>,
    #[arael(ref = root.poses)]
    b: Ref<Pose2<T>>,
    delta: vect2<T>,
    dth: T,
    wt: T,
    wr: T,
    hb: CrossBlock<Pose2<T>, Pose2<T>, T>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct Graph {
    poses: refs::Vec<Pose2<f64>>,
    edges: std::vec::Vec<Edge<f64>>,
}

#[arael::model]
#[arael(root, f32)]
#[derive(Clone)]
struct GraphF {
    poses: refs::Vec<Pose2<f32>>,
    edges: std::vec::Vec<Edge<f32>>,
}

// ---------------------------------------------------------------- runners

fn build_parts<T: Float>(ds: &Dataset)
    -> (refs::Vec<Pose2<T>>, std::vec::Vec<Edge<T>>)
{
    let c = |x: f64| T::from(x).unwrap();
    let mut poses = refs::Vec::new();
    for (i, p) in ds.poses.iter().enumerate() {
        poses.push(Pose2 {
            pos: Param::new(vect2::new(c(p.t.x), c(p.t.y))),
            th: Param::new(c(p.th)),
            prior: vect2::new(c(p.t.x), c(p.t.y)),
            prior_th: c(p.th),
            has_prior: i == 0,
            hb: SelfBlock::new(),
        });
    }
    let mut edges = std::vec::Vec::new();
    for e in &ds.edges {
        edges.push(Edge {
            a: poses.ref_at(e.a),
            b: poses.ref_at(e.b),
            delta: vect2::new(c(e.dx), c(e.dy)),
            dth: c(e.dth),
            wt: c(e.wt),
            wr: c(e.wr),
            hb: CrossBlock::new(),
        });
    }
    (poses, edges)
}

fn solution_parts<T: Float>(poses: &refs::Vec<Pose2<T>>) -> Vec<PoseIn> {
    poses.iter()
        .map(|p| PoseIn {
            t: arael::vect::vect2d::new(
                p.pos.value.x.to_f64().unwrap(),
                p.pos.value.y.to_f64().unwrap(),
            ),
            th: p.th.value.to_f64().unwrap(),
        })
        .collect()
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

/// The two solvers, one per scalar. This is the one thing the generic pipeline
/// cannot pick for itself. The per-solve TIMING breakdown is the harness's.
pub fn solve_f64<P: arael::simple_lm::LmProblem<f64>>(
    params: &[f64],
    p: &mut P,
    cfg: &arael::simple_lm::LmConfig<f64>,
) -> arael::simple_lm::LmResult<f64> {
    let mut solver = arael::simple_lm::SparseFaer::new().with_ordering(ordering());
    arael::simple_lm::lm_solve(params, &mut solver, p, cfg).unwrap()
}

pub fn solve_f32<P: arael::simple_lm::LmProblem<f32>>(
    params: &[f32],
    p: &mut P,
    cfg: &arael::simple_lm::LmConfig<f32>,
) -> arael::simple_lm::LmResult<f32> {
    let mut solver = arael::simple_lm::SparseFaerF32::new().with_ordering(ordering());
    arael::simple_lm::lm_solve(params, &mut solver, p, cfg).unwrap()
}

// Initial damping, problem-appropriate for well-initialized 2D pose graphs (the
// LmConfig docs recommend a small initial_lambda for these); see the README's
// initial-damping policy.
pub const LAMBDA0_2D: f64 = 1e-8;

impl Pipeline for Graph {
    type Scalar = f64;
    type Input = Dataset;
    type Solution = Vec<PoseIn>;
    fn lambda0(_: &Dataset) -> f64 { LAMBDA0_2D }
    fn build(ds: &Dataset) -> Self {
        let (poses, edges) = build_parts(ds);
        Graph { poses, edges }
    }
    fn serialize(&mut self, out: &mut Vec<f64>) { self.serialize64(out); }
    fn deserialize(&mut self, x: &[f64]) { self.deserialize64(x); }
    fn solution(&self) -> Vec<PoseIn> { solution_parts(&self.poses) }
    fn solve(_: &Self::Input, params: &[f64], m: &mut Self, cfg: &arael::simple_lm::LmConfig<f64>)
        -> arael::simple_lm::LmResult<f64> { solve_f64(params, m, cfg) }
}

impl Pipeline for GraphF {
    type Scalar = f32;
    type Input = Dataset;
    type Solution = Vec<PoseIn>;
    fn lambda0(_: &Dataset) -> f64 { LAMBDA0_2D }
    fn build(ds: &Dataset) -> Self {
        let (poses, edges) = build_parts(ds);
        GraphF { poses, edges }
    }
    fn serialize(&mut self, out: &mut Vec<f32>) { self.serialize32(out); }
    fn deserialize(&mut self, x: &[f32]) { self.deserialize32(x); }
    fn solution(&self) -> Vec<PoseIn> { solution_parts(&self.poses) }
    fn solve(_: &Self::Input, params: &[f32], m: &mut Self, cfg: &arael::simple_lm::LmConfig<f32>)
        -> arael::simple_lm::LmResult<f32> { solve_f32(params, m, cfg) }
}

pub type RunOut = Row<Vec<PoseIn>>;

pub fn run_f64(ds: &Dataset) -> RunOut { run::<Graph>(ds) }
pub fn run_f32(ds: &Dataset) -> RunOut { run::<GraphF>(ds) }

