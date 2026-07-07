// M3500 2D pose-graph optimization (Olson's Manhattan-world dataset).
//
// Reads a g2o file with VERTEX_SE2 / EDGE_SE2 entries and solves the
// classic pose-graph problem: 3500 poses (x, y, theta), ~5450 relative
// SE2 measurements between them, gauge fixed by a soft prior on pose 0.
//
// The residual matches the standard between-factor
//   r = [ R_b^T (p_a + R_a t_d - p_b) ; wrap(th_a + dth - th_b) ]
// with unit weights (the dataset's information matrices are ignored to
// stay comparable with other minimal solvers that do the same).
//
// Reads the dataset vendored under benchmarks/pgo/datasets by default;
// pass a path to run any other 2D g2o file:
//   cargo run -r --example m3500_demo [-- path/to/file.g2o] [--weighted]

use arael::model::{Model, Param, SelfBlock, CrossBlock};
use arael::refs::{self, Ref};
use arael::vect::vect2d;

// A 2D pose. The prior constraint fires only on the pose that anchors
// the gauge (has_prior), pulling it toward its initial value with unit
// weight -- without it, any rigid motion of the whole graph would leave
// the cost unchanged and the Hessian singular.
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

// One relative SE2 measurement between two poses. wt/wr are the
// square roots of the (diagonal) information matrix entries -- 1.0 in
// unweighted mode.
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

fn load_g2o(path: &str, weighted: bool) -> Graph {
    let mut graph = Graph { poses: refs::Vec::new(), edges: std::vec::Vec::new() };
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {}", path, e));
    for line in text.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        match f.first().copied() {
            Some("VERTEX_SE2") => {
                // VERTEX_SE2 id x y theta
                let x: f64 = f[2].parse().unwrap();
                let y: f64 = f[3].parse().unwrap();
                let th: f64 = f[4].parse().unwrap();
                let id: usize = f[1].parse().unwrap();
                assert_eq!(id, graph.poses.len(), "vertices must be dense and ordered");
                graph.poses.push(Pose2 {
                    pos: Param::new(vect2d::new(x, y)),
                    th: Param::new(th),
                    prior: vect2d::new(x, y),
                    prior_th: th,
                    has_prior: id == 0,
                    hb: SelfBlock::new(),
                });
            }
            Some("EDGE_SE2") => {
                // EDGE_SE2 id_a id_b dx dy dtheta <info upper-triangle...>
                let ia: u32 = f[1].parse().unwrap();
                let ib: u32 = f[2].parse().unwrap();
                let dx: f64 = f[3].parse().unwrap();
                let dy: f64 = f[4].parse().unwrap();
                let dth: f64 = f[5].parse().unwrap();
                // Info matrix upper triangle: I11 I12 I13 I22 I23 I33.
                // M3500 is diagonal with I11 == I22; sqrt-info weighting
                // then reduces to two per-edge row scales.
                let (mut wt, mut wr) = (1.0, 1.0);
                if weighted && f.len() >= 12 {
                    let i11: f64 = f[6].parse().unwrap();
                    let i12: f64 = f[7].parse().unwrap();
                    let i13: f64 = f[8].parse().unwrap();
                    let i22: f64 = f[9].parse().unwrap();
                    let i23: f64 = f[10].parse().unwrap();
                    let i33: f64 = f[11].parse().unwrap();
                    assert!(i12.abs() < 1e-9 && i13.abs() < 1e-9 && i23.abs() < 1e-9,
                        "only diagonal information matrices are supported");
                    assert!((i11 - i22).abs() < 1e-9, "anisotropic translation info unsupported");
                    wt = i11.sqrt();
                    wr = i33.sqrt();
                }
                let a = graph.poses.ref_at(ia);
                let b = graph.poses.ref_at(ib);
                graph.edges.push(Edge {
                    a,
                    b,
                    delta: vect2d::new(dx, dy),
                    dth,
                    wt,
                    wr,
                    hb: CrossBlock::new(),
                });
            }
            _ => {}
        }
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
        let (sb, cb) = b.th.value.sin_cos();
        let gx = a.pos.value.x + ca * e.delta.x - sa * e.delta.y - b.pos.value.x;
        let gy = a.pos.value.y + sa * e.delta.x + ca * e.delta.y - b.pos.value.y;
        block([
            (cb * gx + sb * gy) * e.wt,
            (-sb * gx + cb * gy) * e.wt,
            arael::utils::rad_diff(a.th.value + e.dth, b.th.value) * e.wr,
        ]);
    }
    for p in graph.poses.iter() {
        if p.has_prior {
            block([
                p.pos.value.x - p.prior.x,
                p.pos.value.y - p.prior.y,
                p.th.value - p.prior_th,
            ]);
        }
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
    graph.serialize64(&mut params);
    println!("parameters: {}", params.len());

    let cfg = arael::simple_lm::LmConfig::<f64> {
        verbose: std::env::var("VERBOSE").is_ok(),
        ..Default::default()
    };
    let start = std::time::Instant::now();
    let result = arael::simple_lm::solve_sparse_faer(&params, &mut graph, &cfg);
    let elapsed = start.elapsed();
    graph.deserialize64(&result.x);

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
