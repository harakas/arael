// arael runners: one generic model, instantiated by the f64 and f32
// roots (the precision is a compile-time property of the generated code).

use bench_harness::arael::{run, Model as Pipeline};
use bench_harness::table::Row;
use crate::g2o::{Dataset, PoseIn};
use arael::angle::AngleParam;
use arael::model::{Param, SelfBlock, CrossBlock};
use arael::refs::{self, Ref};
use arael::utils::Float;
use arael::vect::vect2;

#[arael::model]
#[derive(Clone)]
struct Pose2<T: Float> {
    pos: Param<vect2<T>>,
    rot: AngleParam<T>,
    hb: SelfBlock<Pose2<T>, T>,
}

// The gauge anchor: one prior on the root instead of prior fields on every
// pose. It pulls the referenced pose toward a fixed value; the residuals
// write into the pose's own block (`p.hb`). Without it any rigid motion of
// the whole graph would leave the cost unchanged and the Hessian singular.
#[arael::model]
#[arael(constraint(p.hb, {
    [p.pos.x - prior.pos.x,
     p.pos.y - prior.pos.y,
     p.rot.angle - prior.th]
}))]
#[derive(Clone)]
struct Prior<T: Float> {
    #[arael(ref = root.poses)]
    p: Ref<Pose2<T>>,
    pos: vect2<T>,
    th: T,
}

#[arael::model]
#[arael(constraint(hb, {
    let local = b.rot.rotation_matrix.transpose()
        * (a.pos + a.rot.rotation_matrix * edge.delta - b.pos);
    [local.x * edge.wt,
     local.y * edge.wt,
     rad_diff(a.rot.angle + edge.dth, b.rot.angle) * edge.wr]
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
    prior: Option<Prior<f64>>,
}

#[arael::model]
#[arael(root, f32)]
#[derive(Clone)]
struct GraphF {
    poses: refs::Vec<Pose2<f32>>,
    edges: std::vec::Vec<Edge<f32>>,
    prior: Option<Prior<f32>>,
}

// ---------------------------------------------------------------- runners

fn build_parts<T: Float>(ds: &Dataset)
    -> (refs::Vec<Pose2<T>>, std::vec::Vec<Edge<T>>, Option<Prior<T>>)
{
    let c = |x: f64| T::from(x).unwrap();
    let mut poses = refs::Vec::new();
    for p in &ds.poses {
        poses.push(Pose2 {
            pos: Param::new(vect2::new(c(p.t.x), c(p.t.y))),
            rot: AngleParam::new(c(p.th)),
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
    // Anchor the first pose at its initial value.
    let prior = ds.poses.first().map(|p| Prior {
        p: poses.ref_at(0),
        pos: vect2::new(c(p.t.x), c(p.t.y)),
        th: c(p.th),
    });
    (poses, edges, prior)
}

fn solution_parts<T: Float>(poses: &refs::Vec<Pose2<T>>) -> Vec<PoseIn> {
    poses.iter()
        .map(|p| PoseIn {
            t: arael::vect::vect2d::new(
                p.pos.value.x.to_f64().unwrap(),
                p.pos.value.y.to_f64().unwrap(),
            ),
            th: p.rot.angle.value.to_f64().unwrap(),
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

/// PGO_SCHUR overrides the marginalization decision the backend makes for
/// itself. A pose graph is not the shape Schur elimination was meant for --
/// every entity is a pose coupled to other poses, with no separate class to
/// eliminate -- so the default `Auto` policy weighs the reduction and declines
/// it. `force` takes it anyway, which is how to see what the analysis turned
/// down; `never` pins the whole-system factorization; `auto` is the default
/// spelled out. A typo is rejected rather than silently ignored: it would
/// otherwise produce a row labelled as one route and measured on another.
pub(crate) fn schur_policy() -> arael::simple_lm::SchurPolicy {
    parse_schur(std::env::var("PGO_SCHUR").ok().as_deref())
}

fn parse_schur(v: Option<&str>) -> arael::simple_lm::SchurPolicy {
    use arael::simple_lm::SchurPolicy;
    match v {
        Some("force") | Some("1") => SchurPolicy::Force,
        Some("never") | Some("0") => SchurPolicy::Never,
        Some("auto") | None => SchurPolicy::default(),
        Some(other) => panic!(
            "PGO_SCHUR={}: expected force (marginalize anyway), never (do not), \
             or auto (let the backend decide, the default)", other),
    }
}

pub type Solved<T> = Result<arael::simple_lm::LmResult<T>, arael::simple_lm::SolveFailure<T>>;

/// The two solvers, one per scalar. This is the one thing the generic pipeline
/// cannot pick for itself. The per-solve TIMING breakdown is the harness's.
pub fn solve_f64<P: arael::simple_lm::LmProblem<f64>>(
    params: &[f64],
    p: &mut P,
    cfg: &arael::simple_lm::LmConfig<f64>,
) -> Solved<f64> {
    let mut solver = arael::simple_lm::SparseFaer::new()
        .with_ordering(ordering())
        .with_policy(schur_policy());
    arael::simple_lm::lm_solve(params, &mut solver, p, cfg)
}

pub fn solve_f32<P: arael::simple_lm::LmProblem<f32>>(
    params: &[f32],
    p: &mut P,
    cfg: &arael::simple_lm::LmConfig<f32>,
) -> Solved<f32> {
    let mut solver = arael::simple_lm::SparseFaerF32::new()
        .with_ordering(ordering())
        .with_policy(schur_policy());
    arael::simple_lm::lm_solve(params, &mut solver, p, cfg)
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
        let (poses, edges, prior) = build_parts(ds);
        Graph { poses, edges, prior }
    }
    fn serialize(&mut self, out: &mut Vec<f64>) { arael::simple_lm::RootProblem::serialize(self, out); }
    fn deserialize(&mut self, x: &[f64]) { arael::simple_lm::RootProblem::deserialize(self, x); }
    fn solution(&self) -> Vec<PoseIn> { solution_parts(&self.poses) }
    fn solve(_: &Self::Input, params: &[f64], m: &mut Self, cfg: &arael::simple_lm::LmConfig<f64>)
        -> Solved<f64> { solve_f64(params, m, cfg) }
}

impl Pipeline for GraphF {
    type Scalar = f32;
    type Input = Dataset;
    type Solution = Vec<PoseIn>;
    fn lambda0(_: &Dataset) -> f64 { LAMBDA0_2D }
    fn build(ds: &Dataset) -> Self {
        let (poses, edges, prior) = build_parts(ds);
        GraphF { poses, edges, prior }
    }
    fn serialize(&mut self, out: &mut Vec<f32>) { arael::simple_lm::RootProblem::serialize(self, out); }
    fn deserialize(&mut self, x: &[f32]) { arael::simple_lm::RootProblem::deserialize(self, x); }
    fn solution(&self) -> Vec<PoseIn> { solution_parts(&self.poses) }
    fn solve(_: &Self::Input, params: &[f32], m: &mut Self, cfg: &arael::simple_lm::LmConfig<f32>)
        -> Solved<f32> { solve_f32(params, m, cfg) }
}

/// `Err` is why the solve failed, for the table to show in place of the row.
pub type RunOut = Result<Row<Vec<PoseIn>>, String>;

pub fn run_f64(ds: &Dataset) -> RunOut { run::<Graph>(ds) }
pub fn run_f32(ds: &Dataset) -> RunOut { run::<GraphF>(ds) }


#[cfg(test)]
mod tests {
    use arael::simple_lm::SchurPolicy;

    /// A mistyped value must not fall through to the default: the run would be
    /// labelled as one route in the header and measured on another.
    #[test]
    fn pgo_schur_parses_every_documented_spelling() {
        assert!(matches!(super::parse_schur(Some("force")), SchurPolicy::Force));
        assert!(matches!(super::parse_schur(Some("1")), SchurPolicy::Force));
        assert!(matches!(super::parse_schur(Some("never")), SchurPolicy::Never));
        assert!(matches!(super::parse_schur(Some("0")), SchurPolicy::Never));
        assert!(matches!(super::parse_schur(Some("auto")), SchurPolicy::Auto { .. }));
        assert!(matches!(super::parse_schur(None), SchurPolicy::Auto { .. }));
    }

    #[test]
    #[should_panic(expected = "PGO_SCHUR=froce")]
    fn pgo_schur_rejects_a_typo() {
        super::parse_schur(Some("froce"));
    }
}
