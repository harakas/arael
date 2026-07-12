//! Dump a pose graph's Hessian in Matrix Market form, so the ordering
//! experiments can run on the structure real SLAM produces -- a trajectory
//! plus loop closures, which is exactly what the landmark benchmarks lack.
//!
//! Research scaffolding.
//!
//!     SCHUR_DUMP_DIR=/tmp/mtx cargo run -r --bin dump_h

#[path = "../g2o.rs"]
mod g2o;
#[path = "../g2o3.rs"]
mod g2o3;

use arael::model::{CrossBlock, Param, SelfBlock};
use arael::refs::{self, Ref};
use arael::simple_lm::{block_partition_from_spans, csc_from_cells, LmProblem, RootProblem};

// The 2D pose graph, same shape as the benchmark's.
#[arael::model]
struct Pose {
    x: Param<f64>,
    y: Param<f64>,
    theta: Param<f64>,
    hb: SelfBlock<Pose>,
}

#[arael::model]
#[arael(constraint(hb, {
    let c = cos(a.theta);
    let s = sin(a.theta);
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    [(c * dx + s * dy - edge.dx) * edge.w,
     (-s * dx + c * dy - edge.dy) * edge.w,
     (b.theta - a.theta - edge.dtheta) * edge.w]
}))]
struct Edge {
    #[arael(ref = root.poses)]
    a: Ref<Pose>,
    #[arael(ref = root.poses)]
    b: Ref<Pose>,
    dx: f64,
    dy: f64,
    dtheta: f64,
    w: f64,
    hb: CrossBlock<Pose, Pose>,
}

#[arael::model]
#[arael(root)]
struct Graph {
    poses: refs::Vec<Pose>,
    edges: std::vec::Vec<Edge>,
}

fn main() {
    let dir = std::env::var("SCHUR_DUMP_DIR").expect("set SCHUR_DUMP_DIR");
    for (name, path) in [
        ("pgo-m3500", "datasets/input_M3500_g2o.g2o"),
        ("pgo-city10000", "datasets/city10000.g2o"),
    ] {
        let ds = g2o::load(path, false);
        let mut g = Graph { poses: refs::Vec::new(), edges: std::vec::Vec::new() };
        for p in &ds.poses {
            g.poses.push(Pose {
                x: Param::new(p.x),
                y: Param::new(p.y),
                theta: Param::new(p.th),
                hb: SelfBlock::new(),
            });
        }
        for e in &ds.edges {
            g.edges.push(Edge {
                a: Ref::new(e.a as u32),
                b: Ref::new(e.b as u32),
                dx: e.dx,
                dy: e.dy,
                dtheta: e.dth,
                w: 1.0,
                hb: CrossBlock::new(),
            });
        }

        // Assemble H exactly as the solver's fast path does.
        let mut params = std::vec::Vec::new();
        RootProblem::serialize(&mut g, &mut params);
        let n = params.len();
        let mut cells = std::vec::Vec::new();
        LmProblem::collect_hessian_cells(&mut g, &mut cells);
        let mut spans = std::vec::Vec::new();
        LmProblem::collect_param_block_spans(&mut g, &mut spans);
        let partition = block_partition_from_spans(&spans, n);
        let (mut csc, mut resolver) = csc_from_cells::<f64>(&partition, &cells);
        let mut positions = std::vec::Vec::new();
        LmProblem::accumulate_hessian_positions(
            &mut g,
            &mut |i, j| resolver.resolve(i, j),
            &mut positions,
        );
        let mut grad = vec![0.0; n];
        LmProblem::calc_grad_hessian_sparse_indexed(
            &mut g, &params, &mut grad, &mut csc.vals, &positions,
        );
        // Levenberg damping, as a real solve would apply before factorizing.
        for i in 0..n {
            csc.vals[csc.diag_pos[i]] *= 1.0 + 1e-6;
            csc.vals[csc.diag_pos[i]] += 1e-9;
        }

        use std::io::Write;
        let out = format!("{}/{}.mtx", dir, name);
        let f = std::fs::File::create(&out).expect("create");
        let mut w = std::io::BufWriter::new(f);
        let mut emitted = 0usize;
        for j in 0..n {
            for k in csc.col_ptr[j]..csc.col_ptr[j + 1] {
                if csc.row_idx[k] as usize <= j {
                    emitted += 1;
                }
            }
        }
        writeln!(w, "%%MatrixMarket matrix coordinate real symmetric").unwrap();
        writeln!(w, "{} {} {}", n, n, emitted).unwrap();
        for j in 0..n {
            for k in csc.col_ptr[j]..csc.col_ptr[j + 1] {
                let i = csc.row_idx[k] as usize;
                if i <= j {
                    writeln!(w, "{} {} {:.17e}", j + 1, i + 1, csc.vals[k]).unwrap();
                }
            }
        }
        w.flush().unwrap();
        println!(
            "dumped {} ({} poses, {} edges, n = {}, {} nnz upper)",
            out,
            ds.poses.len(),
            ds.edges.len(),
            n,
            emitted
        );
    }
}
