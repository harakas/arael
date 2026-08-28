// M3500 2D pose-graph optimization (Olson's Manhattan-world dataset).
//
// Reads a g2o file with VERTEX_SE2 / EDGE_SE2 entries and solves the
// classic pose-graph problem: 3500 poses (x, y, theta), ~5450 relative
// SE2 measurements between them, gauge fixed by a soft prior on pose 0.
//
// The residual is the g2o convention, expressed in the measurement
// frame (pose a):
//   r = [ R_a^T (p_b - p_a) - t_d ; wrap(th_b - th_a - dth) ]
// with unit weights by default (the dataset's information matrices are
// ignored to stay comparable with other minimal solvers that do the
// same); --weighted applies the full information matrices.
//
// Reads the dataset vendored under benchmarks/pgo/datasets by default;
// pass a path to run any other 2D g2o file:
//   cargo run -r --example m3500_demo [-- path/to/file.g2o] [--weighted]

use arael::simple_lm::RootProblem;
use arael::model::{Param, SelfBlock, CrossBlock};
use arael::refs::{self, Ref};
use arael::vect::{vect2d, vect3d};

// A 2D pose.
#[arael::model]
struct Pose2 {
    pos: Param<vect2d>,
    th: Param<f64>,
    hb: SelfBlock<Pose2>,
}

// The gauge anchor: ONE optional prior on the root instead of prior
// fields carried by every pose. It pulls the referenced pose toward a
// fixed value with unit weight -- without it, any rigid motion of the
// whole graph would leave the cost unchanged and the Hessian singular.
// The residuals write into the pose's own block (`p.hb`); when the
// Option is None the constraint simply does not exist.
#[arael::model]
#[arael(constraint(p.hb, {
    [p.pos.x - prior.pos.x,
     p.pos.y - prior.pos.y,
     p.th - prior.th]
}))]
struct Prior {
    #[arael(ref = root.poses)]
    p: Ref<Pose2>,
    pos: vect2d,
    th: f64,
}

// One relative SE2 measurement between two poses. s0/s1/s2 are the
// rows of the sqrt-information factor diag(w) * R^T from the
// information matrix eigendecomposition (info = R diag(w)^2 R^T) --
// identity rows in unweighted mode.
#[arael::model]
#[arael(constraint(hb, {
    let local = matrix2sym::rotation(a.th).transpose() * (b.pos - a.pos)
        - edge.delta;
    let rr = rad_diff(b.th, a.th + edge.dth);
    [edge.s0.x * local.x + edge.s0.y * local.y + edge.s0.z * rr,
     edge.s1.x * local.x + edge.s1.y * local.y + edge.s1.z * rr,
     edge.s2.x * local.x + edge.s2.y * local.y + edge.s2.z * rr]
}))]
struct Edge {
    #[arael(ref = root.poses)]
    a: Ref<Pose2>,
    #[arael(ref = root.poses)]
    b: Ref<Pose2>,
    delta: vect2d,
    dth: f64,
    s0: vect3d,
    s1: vect3d,
    s2: vect3d,
    hb: CrossBlock<Pose2, Pose2>,
}

#[arael::model]
#[arael(root)]
struct Graph {
    poses: refs::Vec<Pose2>,
    edges: std::vec::Vec<Edge>,
    prior: Option<Prior>,
}

fn load_g2o(path: &str, weighted: bool) -> Graph {
    let ds = arael::g2o::Dataset2::load(path).unwrap_or_else(|e| panic!("{}: {}", path, e));
    let mut graph = Graph {
        poses: refs::Vec::new(),
        edges: std::vec::Vec::new(),
        prior: None,
    };
    for p in &ds.poses {
        let r = graph.poses.push(Pose2 {
            pos: Param::new(p.t),
            th: Param::new(p.th),
            hb: SelfBlock::new(),
        });
        // The first pose anchors the gauge.
        if graph.prior.is_none() {
            graph.prior = Some(Prior { p: r, pos: p.t, th: p.th });
        }
    }
    for d in &ds.deltas {
        // Exact whitening for any symmetric information matrix: rows of
        // diag(w) * R^T from its eigendecomposition.
        let [s0, s1, s2] = if weighted {
            let (r, w) = d.eigen_sqrt_info();
            [
                vect3d::new(r[0].x * w.x, r[1].x * w.x, r[2].x * w.x),
                vect3d::new(r[0].y * w.y, r[1].y * w.y, r[2].y * w.y),
                vect3d::new(r[0].z * w.z, r[1].z * w.z, r[2].z * w.z),
            ]
        } else {
            [
                vect3d::new(1.0, 0.0, 0.0),
                vect3d::new(0.0, 1.0, 0.0),
                vect3d::new(0.0, 0.0, 1.0),
            ]
        };
        graph.edges.push(Edge {
            a: graph.poses.ref_at(d.a),
            b: graph.poses.ref_at(d.b),
            delta: d.dt,
            dth: d.dth,
            s0,
            s1,
            s2,
            hb: CrossBlock::new(),
        });
    }
    graph
}

// Reference metrics computed directly from the data, independent of the
// generated solver code: plain least-squares cost and the Huber(1.0)
// block metric other minimal solvers report (rho(s) = s for s <= 1,
// 2 sqrt(s) - 1 above).
fn metrics(graph: &Graph) -> (f64, f64) {
    let mut ls = 0.0;
    let mut huber = 0.0;
    let mut block = |r: [f64; 3]| {
        let s: f64 = r.iter().map(|v| v * v).sum();
        ls += s;
        huber += if s > 1.0 { 2.0 * s.sqrt() - 1.0 } else { s };
    };
    for e in &graph.edges {
        let a = &graph.poses[e.a];
        let b = &graph.poses[e.b];
        let (sa, ca) = a.th.value.sin_cos();
        let dx = b.pos.value.x - a.pos.value.x;
        let dy = b.pos.value.y - a.pos.value.y;
        let lx = ca * dx + sa * dy - e.delta.x;
        let ly = -sa * dx + ca * dy - e.delta.y;
        let rr = arael::utils::rad_diff(b.th.value, a.th.value + e.dth);
        block([
            e.s0.x * lx + e.s0.y * ly + e.s0.z * rr,
            e.s1.x * lx + e.s1.y * ly + e.s1.z * rr,
            e.s2.x * lx + e.s2.y * ly + e.s2.z * rr,
        ]);
    }
    if let Some(prior) = &graph.prior {
        let p = &graph.poses[prior.p];
        block([
            p.pos.value.x - prior.pos.x,
            p.pos.value.y - prior.pos.y,
            p.th.value - prior.th,
        ]);
    }
    (ls, huber)
}

// Minimal EPS scatter of pose positions (before = light gray, after =
// black), raw PostScript with no dependencies.
fn write_eps(before: &[(f64, f64)], after: &[(f64, f64)], out: &str) -> std::io::Result<()> {
    use std::io::Write;
    let all = before.iter().chain(after.iter());
    let (mut xmin, mut xmax, mut ymin, mut ymax) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
    for &(x, y) in all {
        xmin = xmin.min(x); xmax = xmax.max(x);
        ymin = ymin.min(y); ymax = ymax.max(y);
    }
    let size = 500.0;
    let scale = size / (xmax - xmin).max(ymax - ymin);
    let mut f = std::io::BufWriter::new(std::fs::File::create(out)?);
    writeln!(f, "%!PS-Adobe-3.0 EPSF-3.0")?;
    writeln!(f, "%%BoundingBox: 0 0 {} {}", size as i32 + 20, size as i32 + 20)?;
    for (points, gray) in [(before, 0.75), (after, 0.0)] {
        writeln!(f, "{} setgray", gray)?;
        for &(x, y) in points {
            writeln!(f, "{:.1} {:.1} 1.2 0 360 arc fill",
                10.0 + (x - xmin) * scale, 10.0 + (y - ymin) * scale)?;
        }
    }
    writeln!(f, "showpage")?;
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let weighted = args.iter().any(|a| a == "--weighted");
    let path = args.iter().skip(1).find(|a| !a.starts_with("--"))
        .cloned().unwrap_or_else(|| {
            concat!(env!("CARGO_MANIFEST_DIR"), "/benchmarks/pgo/datasets/input_M3500_g2o.g2o").to_string()
        });
    let mut graph = load_g2o(&path, weighted);
    if weighted { println!("using information-matrix (sqrt-info) weighting"); }
    println!("{}: {} poses, {} edges", path, graph.poses.len(), graph.edges.len());

    let (ls0, huber0) = metrics(&graph);
    println!("initial cost: LS={:.6} huber={:.6}", ls0, huber0);
    let before: Vec<(f64, f64)> = graph.poses.iter()
        .map(|p| (p.pos.value.x, p.pos.value.y)).collect();

    let mut params: Vec<f64> = Vec::new();
    graph.serialize(&mut params);
    println!("parameters: {}", params.len());

    let cfg = arael::simple_lm::LmConfig::well_conditioned()
        .with_verbose(std::env::var("VERBOSE").is_ok());
    let start = std::time::Instant::now();
    let result = arael::simple_lm::solve_sparse(&params, &mut graph, &cfg).unwrap();
    let elapsed = start.elapsed();
    graph.deserialize(&result.x);

    let (ls1, huber1) = metrics(&graph);
    println!("{} iterations, cost {:.6} -> {:.6}", result.iterations, result.start_cost, result.end_cost);
    println!("final cost:   LS={:.6} huber={:.6}", ls1, huber1);
    println!("solve time: {:?}", elapsed);

    let after: Vec<(f64, f64)> = graph.poses.iter()
        .map(|p| (p.pos.value.x, p.pos.value.y)).collect();
    let out = if weighted { "m3500_weighted.eps" } else { "m3500.eps" };
    write_eps(&before, &after, out).expect("eps write");
    println!("wrote {}", out);

    if let Some(dump) = args.iter().position(|a| a == "--dump").map(|i| args[i + 1].clone()) {
        use std::io::Write;
        let mut f = std::io::BufWriter::new(std::fs::File::create(&dump).unwrap());
        for p in graph.poses.iter() {
            writeln!(f, "{} {} {}", p.pos.value.x, p.pos.value.y, p.th.value).unwrap();
        }
        println!("dumped poses to {}", dump);
    }

    for (label, idx) in [("x0", 0usize), ("x1", 1), ("x3499", 3499)] {
        if idx < graph.poses.len() {
            let p = &graph.poses[idx];
            println!("{}: theta={:.6} x={:.6} y={:.6}", label, p.th.value, p.pos.value.x, p.pos.value.y);
        }
    }
}
