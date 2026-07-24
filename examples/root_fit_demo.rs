// Curve fitting on the regular model/root system: the parameters live on the
// root, the measurements are plain data entities, and each measurement's
// constraint writes straight into the root's own SelfBlock via the
// `root.<selfblock>` block spec.
//
// This is the "one shared parameter set, many observations" shape. Compared
// to `#[arael(root, fit(...))]` (see examples/linear_demo.rs) it trades the
// one-liner for the full constraint feature set: per-measurement guards,
// block robust losses, named residual groups, and measurements that can grow
// into real entities later. In constraint bodies `root` names the root
// (`root.a` here); the lowercased root type (`fit.a`) works too.
//
// Run: cargo run -r --example root_fit_demo

use arael::model::{Param, SelfBlock};
use arael::simple_lm::{LmConfig, LmProblem};

// One measurement: pure data, no params, no blocks. The residual reads only
// root params, so the constraint names the root's SelfBlock as its block.
// The Huber loss caps what a single bad point can contribute; the guard
// drops measurements flagged invalid.
#[arael::model]
#[arael(constraint(root.hb, guard = e.ok, loss = |s| loss_huber(s, 0.01), {
    [e.y - root.a * e.x - root.b]
}))]
struct E {
    x: f64,
    y: f64,
    ok: bool,
}

#[arael::model]
#[arael(root)]
struct Fit {
    a: Param<f64>,
    b: Param<f64>,
    hb: SelfBlock<Fit>,
    data: std::vec::Vec<E>,
}

fn main() {
    // y = 2x + 1 with mild noise, one gross outlier, one invalid reading.
    let mut rng = 123456789u64;
    let mut noise = || {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((rng >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 0.02
    };
    let mut data: Vec<E> = (0..40)
        .map(|i| {
            let x = i as f64 * 0.05;
            E { x, y: 2.0 * x + 1.0 + noise(), ok: true }
        })
        .collect();
    data[7].y = 50.0; // outlier: the Huber loss caps its influence
    data[9].ok = false; // invalid reading: the guard excludes it entirely
    data[9].y = -999.0;

    let mut m = Fit {
        a: Param::new(0.0),
        b: Param::new(0.0),
        hb: SelfBlock::new(),
        data,
    };

    let r = m.solve_dense(&LmConfig::well_conditioned()).unwrap();
    println!(
        "fit: y = {:.4} * x + {:.4}   (truth 2, 1; one outlier suppressed, one reading guarded out)",
        m.a.value, m.b.value
    );
    println!(
        "{} iterations, cost {:.4} -> {:.4}, status {:?}",
        r.iterations, r.start_cost, r.end_cost, r.status
    );
}
