// Minimal working example of a single-struct model+root with a direct-composed
// sub-model. The root carries its own SelfBlock<Self> plus two fix-value
// constraints; the embedded Sub carries its own SelfBlock<Sub> with another
// fix-value constraint. Three parameters total, three fix constraints,
// LM converges to (x, y, sub.z) = (3, 4, 5).
//
// Contrast with jacobian_demo / loc_demo / slam_demo, which use the
// `refs::Vec<Entity>` container pattern. Both shapes are now supported.

use arael::model::{JacobianModel, Model, Param, SelfBlock};
use arael::simple_lm::{self, LmConfig, LmProblem};

// Direct-composition sub-model: single parameter, single constraint pinning it
// to 5.0. Not wrapped in refs::Vec, just embedded as a plain field in the root.
#[arael::model]
#[arael(constraint(hb, name = "fix_z", {
    [(sub.z - 5.0) * singleroot.isigma]
}))]
struct Sub {
    z: Param<f64>,
    #[arael(constraint_index)]
    ci: u32,
    hb: SelfBlock<Sub>,
}

#[arael::model]
#[arael(root, jacobian)]
#[arael(constraint(hb, name = "fix_x", {
    [(singleroot.x - 3.0) * singleroot.isigma]
}))]
#[arael(constraint(hb, name = "fix_y", {
    [(singleroot.y - 4.0) * singleroot.isigma]
}))]
struct SingleRoot {
    x: Param<f64>,
    y: Param<f64>,
    isigma: f64,
    sub: Sub,
    #[arael(constraint_index)]
    ci: u32,
    hb: SelfBlock<SingleRoot>,
}

fn main() {
    let mut m = SingleRoot {
        x: Param::new(0.0),
        y: Param::new(0.0),
        isigma: 10.0,
        sub: Sub {
            z: Param::new(0.0),
            ci: 0,
            hb: SelfBlock::new(),
        },
        ci: 0,
        hb: SelfBlock::new(),
    };

    let mut params = Vec::new();
    m.serialize64(&mut params);
    println!("Serialized params: {:?}", params);
    println!("PARAM_COUNT = {}", SingleRoot::PARAM_COUNT);

    // Initial cost: (0-3)^2 * 100 + (0-4)^2 * 100 + (0-5)^2 * 100 = 2500 + 2500 = 5000.
    let cost0 = m.calc_cost(&params);
    println!("Initial cost = {:.6} (expect 5000.0)", cost0);

    let j = m.calc_jacobian(&params);
    println!(
        "Jacobian: {} residuals x {} params (expect 3 x 3)",
        j.num_residuals(),
        j.num_params
    );
    for (i, row) in j.rows.iter().enumerate() {
        let entries: Vec<String> = row
            .entries
            .iter()
            .map(|(idx, v)| format!("p{}={:+.4}", idx, v))
            .collect();
        println!(
            "  row {:2}: cid={} label={:<8} r={:+.6} [{}]",
            i,
            row.constraint,
            row.label,
            row.residual,
            entries.join(", ")
        );
    }

    let config = LmConfig::well_conditioned().with_verbose(true);
    let result = simple_lm::solve(&params, &mut m, &config);
    m.deserialize64(&result.x);

    println!(
        "\nLM: {} iterations, cost {:.6} -> {:.6}",
        result.iterations, result.start_cost, result.end_cost
    );
    println!("Final params: {:?}", result.x);
    println!(
        "x.value = {:.6} (expect 3.0), y.value = {:.6} (expect 4.0), sub.z.value = {:.6} (expect 5.0)",
        m.x.value, m.y.value, m.sub.z.value
    );
}
