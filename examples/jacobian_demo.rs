/// Jacobian computation demo.
///
/// Shows `#[arael(root, jacobian)]`, `#[arael(constraint_index)]`, and the
/// optional `name = "..."` on individual constraint attributes. The label
/// shows up on each emitted `JacobianRow`, which is handy for DOF /
/// sparsity analyses where you want to know which of a struct's several
/// residual groups produced a given row.
///
/// Use `cargo expand --example jacobian_demo` to inspect generated code.

use arael::model::{CrossBlock, JacobianModel, Param, SelfBlock};
use arael::simple_lm::LmProblem;
use arael::vect::vect2d;

// Point has TWO constraint attributes, both given explicit names. Without
// the names the labels would default to "Point:0" and "Point:1".
#[arael::model]
#[arael(constraint(hb, name = "drift", {
    let d = point.pos - point.pos_value;
    [d.x * jacmodel.isigma, d.y * jacmodel.isigma]
}))]
#[arael(constraint(hb, name = "fix_x", guard = self.has_fix_x, {
    [(point.pos.x - point.fix_x) * jacmodel.isigma]
}))]
struct Point {
    pos: Param<vect2d>,
    pos_value: vect2d,
    has_fix_x: bool,
    fix_x: f64,
    #[arael(constraint_index)]
    ci: u32,
    hb: SelfBlock<Point>,
}

// Coincident has a single constraint attribute and no `name`, so the
// default label is just the struct name: "Coincident".
#[arael::model]
#[arael(constraint(hb, {
    [(a.pos.x - b.pos.x) * jacmodel.isigma,
     (a.pos.y - b.pos.y) * jacmodel.isigma]
}))]
struct Coincident {
    #[arael(ref = root.points)]
    a: arael::refs::Ref<Point>,
    #[arael(ref = root.points)]
    b: arael::refs::Ref<Point>,
    #[arael(constraint_index)]
    ci: u32,
    hb: CrossBlock<Point, Point>,
}

#[arael::model]
#[arael(root, jacobian)]
struct JacModel {
    points: arael::refs::Vec<Point>,
    coincidents: arael::refs::Vec<Coincident>,
    isigma: f64,
}

fn main() {
    let mut m = JacModel {
        points: arael::refs::Vec::new(),
        coincidents: arael::refs::Vec::new(),
        isigma: 10.0,
    };
    m.points.push(Point {
        pos: Param::new(vect2d::new(0.0, 0.0)),
        pos_value: vect2d::new(0.0, 0.0),
        has_fix_x: true,          // exercises the guarded "fix_x" attribute
        fix_x: 0.0,
        ci: 0, hb: SelfBlock::new(),
    });
    m.points.push(Point {
        pos: Param::new(vect2d::new(1.0, 0.0)),
        pos_value: vect2d::new(1.0, 0.0),
        has_fix_x: false,         // "fix_x" attribute is inactive for this point
        fix_x: 0.0,
        ci: 0, hb: SelfBlock::new(),
    });
    let a = m.points.ref_at(0);
    let b = m.points.ref_at(1);
    m.coincidents.push(Coincident {
        a,
        b,
        ci: 0, hb: CrossBlock::new(),
    });

    let mut params = Vec::new();
    m.serialize64(&mut params);
    // Perturb
    params[0] += 0.1;
    params[3] -= 0.2;

    let j = m.calc_jacobian(&params);

    println!("Jacobian: {} residuals x {} params", j.num_residuals(), j.num_params);
    for (i, row) in j.rows.iter().enumerate() {
        let entries: Vec<String> = row.entries.iter()
            .map(|(idx, val)| format!("p{}={:.4}", idx, val))
            .collect();
        // row.label is the static string set by `name = "..."` (or
        // defaulted to the struct name / "<Struct>:<idx>" suffix).
        println!("  row {:2}: cid={} label={:<12} r={:+.6} [{}]",
            i, row.constraint, row.label, row.residual, entries.join(", "));
    }

    // --- Cost comparison ---
    let cost_j: f64 = j.rows.iter().map(|r| r.residual * r.residual).sum();
    let cost_c = m.calc_cost(&params);
    println!("\nCost comparison:");
    println!("  sum(r^2)  = {:.6}", cost_j);
    println!("  calc_cost = {:.6}", cost_c);
    assert!((cost_j - cost_c).abs() < 1e-12, "cost mismatch");

    // --- Gradient comparison: grad = 2 * J^T * r ---
    let n = j.num_params;
    let mut grad_j = vec![0.0f64; n];
    for row in &j.rows {
        for &(idx, d) in &row.entries {
            grad_j[idx as usize] += 2.0 * row.residual * d;
        }
    }
    let mut grad = vec![0.0f64; n];
    let mut hessian = vec![0.0f64; n * n];
    m.calc_grad_hessian_dense(&params, &mut grad, &mut hessian);

    println!("\nGradient comparison (2*J^T*r vs calc_grad_hessian):");
    for i in 0..n {
        let ok = (grad_j[i] - grad[i]).abs() < 1e-10;
        println!("  grad[{}]: J={:+.6}  GH={:+.6}  {}", i, grad_j[i], grad[i],
            if ok { "ok" } else { "MISMATCH" });
        assert!(ok, "gradient mismatch at {}", i);
    }

    // --- Hessian comparison: H = 2 * J^T * J ---
    let dense = j.to_dense();
    let m_rows = j.num_residuals();
    let mut jtj = vec![0.0f64; n * n];
    for i in 0..n {
        for k in 0..n {
            let mut s = 0.0;
            for r in 0..m_rows {
                s += dense[r * n + i] * dense[r * n + k];
            }
            jtj[i * n + k] = s;
        }
    }

    println!("\nHessian comparison (2*J^T*J vs calc_grad_hessian):");
    let mut all_ok = true;
    for i in 0..n {
        for k in 0..n {
            let expected = 2.0 * jtj[i * n + k];
            let actual = hessian[i * n + k];
            if (expected - actual).abs() > 1e-10 {
                println!("  H[{},{}]: 2*JtJ={:.6}  GH={:.6}  MISMATCH", i, k, expected, actual);
                all_ok = false;
            }
        }
    }
    if all_ok {
        println!("  all entries match");
    }
    assert!(all_ok, "hessian mismatch");

    // --- Constraint indices ---
    println!("\nConstraint indices:");
    for (i, p) in m.points.iter().enumerate() {
        println!("  point[{}].ci = {}", i, p.ci);
    }
    for (i, c) in m.coincidents.iter().enumerate() {
        println!("  coincident[{}].ci = {}", i, c.ci);
    }

    // --- Per-label cost breakdown ---
    // JacobianModel::calc_cost_table returns the squared-residual sum
    // grouped by each row's label. Row counts come from the Jacobian
    // we already have.
    use std::collections::BTreeMap;
    let mut rows_per_label: BTreeMap<&'static str, usize> = BTreeMap::new();
    for row in &j.rows { *rows_per_label.entry(row.label).or_insert(0) += 1; }
    let cost_table = m.calc_cost_table(&params);
    // BTreeMap the cost table so output order is deterministic.
    let cost_table: BTreeMap<_, _> = cost_table.into_iter().collect();
    println!("\nPer-label breakdown (rows, cost):");
    for (label, count) in &rows_per_label {
        let cost = cost_table.get(label).copied().unwrap_or(0.0);
        println!("  {:<12} rows={} cost={:.6}", label, count, cost);
    }
    let total: f64 = cost_table.values().sum();
    println!("  {:<12}          total={:.6}", "", total);

    // --- Singular value spectrum ---
    // Jacobian::singular_values returns sigma in descending order; small
    // sigma near zero indicates degrees of freedom (unconstrained
    // parameter-space directions). Always computed in f64 regardless of
    // the Jacobian's element type.
    let svs = j.singular_values();
    println!("\nSingular values (descending):");
    for (i, s) in svs.iter().enumerate() {
        println!("  sigma[{}] = {:.6e}", i, s);
    }

    println!("\nAll checks passed.");
}
