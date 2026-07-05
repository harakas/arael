// factrs f32 runner, speaking the shared benchmark protocol:
//   factrs32-bench <file.g2o> <gn|lm> <poses_out> <info|unit>
// JSON {solve_ms, first_iter_ms, iterations, cpus_allowed} on stdout,
// "x y theta" lines to poses_out. factrs's f32 mode is a crate-global
// dtype switch, hence this separate binary (see ../src/factrs_runner.rs
// for the f64 in-process runner; the graph construction is identical).

use std::sync::atomic::{AtomicUsize, Ordering};

use factrs::assign_symbols;
use factrs::core::{BetweenResidual, GaussianNoise, GaussNewton, Graph, LevenMarquardt, PriorResidual, Values, SE2};
use factrs::fac;
use factrs::optimizers::{BaseOptParams, LevenParams, OptError, OptObserver};
use factrs::traits::Optimizer;

assign_symbols!(X: SE2);

static STEPS: AtomicUsize = AtomicUsize::new(0);

struct StepCounter;
impl OptObserver for StepCounter {
    fn on_step(&self, _values: &Values, _time: i64) {
        STEPS.fetch_add(1, Ordering::Relaxed);
    }
}

struct EdgeIn {
    a: u32,
    b: u32,
    dx: f64,
    dy: f64,
    dth: f64,
    it: f64, // translation information
    ir: f64, // rotation information
}

fn parse(path: &str, unit: bool) -> (Vec<(f64, f64, f64)>, Vec<EdgeIn>) {
    let text = std::fs::read_to_string(path).expect("cannot read g2o");
    let mut poses = Vec::new();
    let mut edges = Vec::new();
    for line in text.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        match f.first().copied() {
            Some("VERTEX_SE2") => {
                poses.push((f[2].parse().unwrap(), f[3].parse().unwrap(), f[4].parse().unwrap()));
            }
            Some("EDGE_SE2") => {
                let i11: f64 = f[6].parse().unwrap();
                let i33: f64 = f[11].parse().unwrap();
                edges.push(EdgeIn {
                    a: f[1].parse().unwrap(),
                    b: f[2].parse().unwrap(),
                    dx: f[3].parse().unwrap(),
                    dy: f[4].parse().unwrap(),
                    dth: f[5].parse().unwrap(),
                    it: if unit { 1.0 } else { i11 },
                    ir: if unit { 1.0 } else { i33 },
                });
            }
            _ => {}
        }
    }
    (poses, edges)
}

fn build(poses: &[(f64, f64, f64)], edges: &[EdgeIn]) -> (Graph, Values) {
    let mut graph = Graph::new();
    let mut values = Values::new();
    for (i, p) in poses.iter().enumerate() {
        values.insert(X(i as u32), SE2::new(p.2 as f32, p.0 as f32, p.1 as f32));
    }
    for e in edges {
        let delta = SE2::new(e.dth as f32, e.dx as f32, e.dy as f32);
        let inf = factrs::linalg::Vector3::new(e.ir as f32, e.it as f32, e.it as f32);
        let noise = GaussianNoise::<3>::from_vec_inf(inf.as_view());
        graph.add_factor(fac![BetweenResidual::new(delta), (X(e.a), X(e.b)), noise]);
    }
    let p0 = poses[0];
    graph.add_factor(fac![
        PriorResidual::new(SE2::new(p0.2 as f32, p0.0 as f32, p0.1 as f32)),
        X(0),
        1.0 as std
    ]);
    (graph, values)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (path, kind, poses_out, weights) = (&args[1], &args[2], &args[3], &args[4]);
    let (poses, edges) = parse(path, weights == "unit");

    let optimize = |max_iter: usize| -> (f64, usize, Values) {
        let (graph, init) = build(&poses, &edges);
        let base = BaseOptParams {
            max_iterations: max_iter,
            error_tol_relative: 1e-5,
            error_tol_absolute: 1e-5,
            ..Default::default()
        };
        let before = STEPS.load(Ordering::Relaxed);
        let t0 = std::time::Instant::now();
        let result = if kind == "gn" {
            let mut opt = GaussNewton::new(base, graph);
            opt.observers_mut().add(StepCounter);
            opt.optimize(init)
        } else {
            let mut opt = LevenMarquardt::new(LevenParams { base, ..Default::default() }, graph);
            opt.observers_mut().add(StepCounter);
            opt.optimize(init)
        };
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let values = match result {
            Ok(v) => v,
            Err(OptError::MaxIterations(v)) => v,
            Err(e) => panic!("factrs f32 failed: {:?}", e),
        };
        (ms, STEPS.load(Ordering::Relaxed) - before, values)
    };

    let (first_iter_ms, _, _) = optimize(1);
    let (solve_ms, iterations, values) = optimize(100);

    let mut out = String::new();
    for i in 0..poses.len() {
        let p: &SE2 = values.get(X(i as u32)).expect("missing pose");
        out.push_str(&format!("{} {} {}\n", p.x(), p.y(), p.theta()));
    }
    std::fs::write(poses_out, out).unwrap();

    let cpus = std::fs::read_to_string("/proc/self/status").unwrap()
        .lines()
        .find(|l| l.starts_with("Cpus_allowed_list"))
        .map(|l| l.split_whitespace().last().unwrap().to_string())
        .unwrap_or_else(|| "?".to_string());
    println!("{{\"solve_ms\": {:.3}, \"first_iter_ms\": {:.3}, \"iterations\": {}, \"cpus_allowed\": \"{}\"}}",
        solve_ms, first_iter_ms, iterations, cpus);
}
