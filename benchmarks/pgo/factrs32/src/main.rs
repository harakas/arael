// factrs f32 SE2 runner, speaking the shared benchmark protocol:
//   factrs32-bench <file.g2o> <gn|lm> <poses_out> <info|unit>
// JSON on stdout, "x y theta" lines to poses_out.
//
// The graph, the residuals and the optimizer setup are the parent crate's SE2
// runner compiled against factrs's f32 dtype -- included, not reimplemented,
// so the two precisions cannot drift apart. factrs's f32 mode is a
// crate-global dtype feature, which is why this is a separate binary.

// The included modules are shared with the f64 harness and expose more than
// this binary calls.
#![allow(dead_code)]

#[path = "../../src/factrs_counting.rs"]
mod factrs_counting;

#[path = "../../src/g2o.rs"]
mod g2o;

#[path = "../../src/factrs_runner.rs"]
mod factrs_runner;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (path, kind, poses_out, weights) = (&args[1], &args[2], &args[3], &args[4]);
    let ds = g2o::load(path, weights == "unit");

    let out = if kind == "gn" {
        factrs_runner::run_gn(&ds)
    } else {
        factrs_runner::run_lm(&ds)
    };

    let mut text = String::new();
    for p in &out.solution {
        text.push_str(&format!("{} {} {}\n", p.x, p.y, p.th));
    }
    std::fs::write(poses_out, text).unwrap();

    println!("{}", factrs_counting::protocol_line(
        out.solve_ms, out.first_iter_ms, out.iterations,
        out.accepted.unwrap_or(out.iterations), out.full_ms));
}
