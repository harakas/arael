//! Differentiation must keep the sharing of its input. An expression is
//! a DAG of shared nodes; a fisheye camera projection references the
//! scaled image point through every power of the fisheye angle and the
//! Aria model feeds the radially distorted point into its tangential
//! and thin-prism terms on top. The derivative of such an expression
//! has to stay a small multiple of the expression's node count: a
//! derivative built afresh along every path unfolds the DAG into a
//! tree, and the model macro then spends gigabytes on it.

use arael_sym::{
    atan, branch, epsilon_for, safe_sqrt, symbol, quaternsym, vect3sym, E, Expr,
};
use std::collections::HashSet;

fn children(e: &E) -> Vec<&E> {
    match e.as_ref() {
        Expr::Sym(_) | Expr::Const(_) | Expr::NamedConst { .. } => vec![],
        Expr::Neg(a)
        | Expr::Sin(a)
        | Expr::Cos(a)
        | Expr::Tan(a)
        | Expr::Asin(a)
        | Expr::Acos(a)
        | Expr::Atan(a)
        | Expr::Sinh(a)
        | Expr::Cosh(a)
        | Expr::Tanh(a)
        | Expr::Exp(a)
        | Expr::Ln(a)
        | Expr::Log2(a)
        | Expr::Log10(a)
        | Expr::Sqrt(a)
        | Expr::Abs(a)
        | Expr::Heaviside(a) => vec![a],
        Expr::Add(a, b)
        | Expr::Sub(a, b)
        | Expr::Mul(a, b)
        | Expr::Div(a, b)
        | Expr::Pow(a, b)
        | Expr::Atan2(a, b) => vec![a, b],
        Expr::Clamp(a, b, c) | Expr::Branch(a, b, c) => vec![a, b, c],
        Expr::Select { index, arms, default } => {
            let mut v = vec![index];
            v.extend(arms.iter());
            v.extend(default.iter());
            v
        }
        Expr::Func { args, .. } => args.iter().collect(),
    }
}

/// Unique nodes reachable from `e`.
fn dag_size(e: &E) -> usize {
    fn walk(e: &E, seen: &mut HashSet<*const Expr>) -> usize {
        if !seen.insert(e.as_ref() as *const Expr) {
            return 0;
        }
        1 + children(e).into_iter().map(|c| walk(c, seen)).sum::<usize>()
    }
    walk(e, &mut HashSet::new())
}

/// Nodes of `e` unfolded into a tree, a shared node counted once per path.
fn tree_size(e: &E) -> usize {
    1 + children(e).into_iter().map(tree_size).sum::<usize>()
}

/// Peak resident memory of this process in megabytes, from the kernel.
fn peak_rss_mb() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with("VmHWM:"))?;
    line.split_whitespace().nth(1)?.parse::<u64>().ok().map(|kb| kb / 1024)
}

/// The pixel of a world point through a rotated camera under COLMAP's
/// RAD_TAN_THIN_PRISM_FISHEYE model, with the powers of the fisheye
/// angle written as products, as a model would.
fn aria_projection() -> E {
    let q = quaternsym {
        t: symbol("qw"),
        v: vect3sym::from_components(symbol("qx"), symbol("qy"), symbol("qz")),
    };
    let p = q.rotation_matrix() * vect3sym::new("point")
        + vect3sym::from_components(symbol("tx"), symbol("ty"), symbol("tz"));
    let size = |name: &str, e: &E| {
        println!("  {:<10} {:>7} nodes, {:>8} unfolded", name, dag_size(e), tree_size(e));
    };
    size("p.x", &p.x);
    let u = p.x.clone() / p.z.clone();
    let v = p.y / p.z;
    size("u", &u);
    let r = safe_sqrt(u.clone() * u.clone() + v.clone() * v.clone());
    size("r", &r);
    let scale = branch(r.clone() - epsilon_for(r.clone()), atan(r.clone()) / r, 1.0);
    size("scale", &scale);
    let uu = u * scale.clone();
    let vv = v * scale;
    size("uu", &uu);
    let t2 = uu.clone() * uu.clone() + vv.clone() * vv.clone();
    size("t2", &t2);
    let t4 = t2.clone() * t2.clone();
    let t6 = t4.clone() * t2.clone();
    size("t6", &t6);
    let radial = E::from(1.0)
        + symbol("k0") * t2
        + symbol("k1") * t4.clone()
        + symbol("k2") * t6.clone()
        + symbol("k3") * t4.clone() * t4.clone()
        + symbol("k4") * t6.clone() * t4
        + symbol("k5") * t6.clone() * t6;
    size("radial", &radial);
    let x = radial.clone() * uu;
    let y = radial * vv;
    size("x", &x);
    let x2 = x.clone() * x.clone();
    let y2 = y.clone() * y.clone();
    let xy = x.clone() * y;
    let r2 = x2.clone() + y2;
    size("r2", &r2);
    let r4 = r2.clone() * r2.clone();
    size("r4", &r4);
    let distorted = x
        + 2.0 * symbol("p1") * xy
        + symbol("p0") * (r2.clone() + 2.0 * x2)
        + symbol("s0") * r2
        + symbol("s1") * r4;
    size("distorted", &distorted);
    symbol("fx") * distorted + symbol("cx") - symbol("px")
}

/// The unique nodes of `e` that are structural copies of another unique
/// node: sharing the construction lost. Returns the count of redundant
/// nodes and the largest duplicated subexpression.
fn lost_sharing(e: &E) -> (usize, String) {
    fn collect<'a>(e: &'a E, seen: &mut HashSet<*const Expr>, out: &mut Vec<&'a E>) {
        if !seen.insert(e.as_ref() as *const Expr) {
            return;
        }
        out.push(e);
        for c in children(e) {
            collect(c, seen, out);
        }
    }
    let mut nodes = Vec::new();
    collect(e, &mut HashSet::new(), &mut nodes);
    let mut groups: std::collections::HashMap<&E, usize> = std::collections::HashMap::new();
    for n in &nodes {
        *groups.entry(n).or_insert(0) += 1;
    }
    let redundant: usize = groups.values().filter(|&&c| c > 1).map(|c| c - 1).sum();
    let largest = groups
        .iter()
        .filter(|(_, c)| **c > 1)
        .max_by_key(|(n, _)| tree_size(n))
        .map(|(n, c)| format!("{c} copies of {}", format!("{n}").chars().take(90).collect::<String>()))
        .unwrap_or_default();
    (redundant, largest)
}

/// Where the construction of an expression through the operators loses
/// sharing: each line one operation on a shared subexpression, with
/// the unique node count the operation adds.
#[test]
fn construction_keeps_the_sharing_of_its_operands() {
    let q = quaternsym {
        t: symbol("qw"),
        v: vect3sym::from_components(symbol("qx"), symbol("qy"), symbol("qz")),
    };
    let p = q.rotation_matrix() * vect3sym::new("point");
    let a = p.x.clone() / p.z.clone();
    let b = p.y / p.z;
    let na = dag_size(&a);
    println!("  a = p.x / p.z: {na} nodes");
    let probe = |name: &str, e: &E| {
        let (redundant, largest) = lost_sharing(e);
        println!(
            "  {:<24} {:>6} nodes  (+{} over a)  redundant {redundant}  {largest}",
            name,
            dag_size(e),
            dag_size(e) as i64 - na as i64
        );
    };
    probe("a * a", &(a.clone() * a.clone()));
    probe("a * b", &(a.clone() * b.clone()));
    let s = a.clone() * a.clone() + b.clone() * b.clone();
    probe("a*a + b*b", &s);
    probe("s * s", &(s.clone() * s.clone()));
    probe("k * s", &(symbol("k") * s.clone()));
    probe("1 + k*s", &(E::from(1.0) + symbol("k") * s.clone()));
    probe("1 + k1*s + k2*s*s", &(E::from(1.0) + symbol("k1") * s.clone() + symbol("k2") * s.clone() * s.clone()));
    let r = safe_sqrt(s.clone());
    probe("safe_sqrt(s)", &r);
    probe("atan(r) / r", &(atan(r.clone()) / r.clone()));
    let scale = branch(r.clone() - epsilon_for(r.clone()), atan(r.clone()) / r.clone(), 1.0);
    probe("branch(r - eps, ..., 1)", &scale);
    // Every operation adds a handful of nodes on top of its shared
    // operands; a rebuild that copies an operand shows up as a jump.
    let (redundant, largest) = lost_sharing(&scale);
    assert!(dag_size(&scale) <= na + 60 && redundant <= 40,
        "the construction lost sharing: {} nodes for a = {na}, {redundant} redundant, {largest}",
        dag_size(&scale));
}

#[test]
fn derivative_keeps_the_sharing_of_its_input() {
    let built = std::time::Instant::now();
    let expr = aria_projection();
    println!("built in {:.2} s", built.elapsed().as_secs_f64());
    let expr_nodes = dag_size(&expr);
    let raw_start = std::time::Instant::now();
    let raw = expr.diff_unsimplified("point.x");
    println!(
        "raw derivative {} nodes ({} unfolded) in {:.2} s",
        dag_size(&raw),
        tree_size(&raw),
        raw_start.elapsed().as_secs_f64()
    );
    let before = peak_rss_mb();
    let start = std::time::Instant::now();
    let d = expr.diff("point.x");
    let secs = start.elapsed().as_secs_f64();
    let after = peak_rss_mb();
    let d_nodes = dag_size(&d);
    println!(
        "expression {} nodes ({} unfolded), derivative {} nodes ({} unfolded, {:.1}x) in {:.2} s, \
         peak RSS {:?} -> {:?} MB",
        expr_nodes,
        tree_size(&expr),
        d_nodes,
        tree_size(&d),
        d_nodes as f64 / expr_nodes as f64,
        secs,
        before,
        after
    );
    // The projection is a few hundred operations on shared pieces; its
    // unique nodes must stay in that range, whatever the unfolded size.
    assert!(
        expr_nodes <= 600,
        "the expression has {} unique nodes for a few hundred operations: the operators copy their operands",
        expr_nodes
    );
    // A derivative shares the way its expression shares: a handful of new
    // nodes per node of the input, never a copy per path.
    assert!(
        d_nodes <= 8 * expr_nodes,
        "the derivative has {} nodes for an expression of {}: it unfolds the shared subexpressions",
        d_nodes,
        expr_nodes
    );
}
