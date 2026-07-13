// factrs f32 SE3 runner, speaking the shared benchmark protocol:
//   factrs32-bench3 <file.g2o> <gn|lm> <poses_out>
// JSON {solve_ms, first_iter_ms, iterations, cpus_allowed} on stdout,
// "x y z qx qy qz qw" lines to poses_out.
//
// The graph, the canonical residual and the optimizer setup are the parent
// crate's SE3 runner compiled against factrs's f32 dtype -- included, not
// reimplemented, so the two precisions cannot drift apart. factrs's f32 mode
// is a crate-global dtype feature, which is why this is a separate binary.

// The included modules are shared with the f64 harness and expose more than
// this binary calls.
#![allow(dead_code)]

#[path = "../../src/factrs_counting.rs"]
mod factrs_counting;

#[path = "../../src/probe.rs"]
mod probe;

#[path = "../../src/g2o3.rs"]
mod g2o3;

#[path = "../../src/factrs_runner3.rs"]
mod factrs_runner3;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (path, kind, poses_out) = (&args[1], &args[2], &args[3]);
    let ds = g2o3::load3(path);

    let out = if kind == "gn" {
        factrs_runner3::run_gn(&ds)
    } else {
        factrs_runner3::run_lm(&ds)
    };

    let mut text = String::new();
    for p in &out.poses {
        text.push_str(&format!(
            "{} {} {} {} {} {} {}\n",
            p.t.x, p.t.y, p.t.z, p.q[0], p.q[1], p.q[2], p.q[3]));
    }
    std::fs::write(poses_out, text).unwrap();

    println!("{}", factrs_counting::protocol_line(
        out.solve_ms, out.first_iter_ms, out.iterations, out.accepted, out.two_iter_ms));
}
