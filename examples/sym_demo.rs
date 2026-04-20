use std::collections::HashMap;
use arael::sym::*;
use arael::sym;

fn main() {
    // FunctionBag for use at parse_with_functions
    let mut bag = FunctionBag::new();

    // ============================================================
    // Basics
    // ============================================================
    println!("=== Basics ===\n");

    sym! {
        let (x, y, a, b) = symbols!(x, y, a, b);

        println!("Symbols:    x = {x}, y = {y}");
        println!("Constants:  constant(3.0) = {}, c(3.0) = {}", constant(3.0), c(3.0));
        println!("Arithmetic: x + y = {}", x + y);
        println!("            x * y - 1 = {}", x * y - 1.0);
        println!("            x^2 = {}", pow(x, 2.0));

        // Auto-simplification (like SymPy)
        println!("\n--- Auto-simplification ---");
        println!("(x + y) / (x + y) = {}", (x + y) / (x + y));
        println!("x + 0 = {}", x + 0.0);
        println!("x * 1 = {}", x * 1.0);
        println!("x * 0 = {}", x * 0.0);
        println!("-(-x) = {}", -(-x));
        println!("x - x = {}", x - x);
        println!("3*x + 2*x = {}", 3.0 * x + 2.0 * x);
        println!("x * x = {}", x * x);

        // ============================================================
        // Derivatives — all rules (no .simplify() needed!)
        // ============================================================
        println!("\n=== Derivatives ===\n");

        // Power rule
        let x4 = pow(x, 4.0);
        let dx4 = x4.diff(x);
        println!("Power rule:     d/dx(x^4)  = {dx4}");
        let d2x4 = dx4.diff(x);
        println!("  2nd deriv:    d²/dx²(x^4) = {d2x4}");

        // Product rule
        let xsinx = x * sin(x);
        println!("Product rule:   d/dx(x·sin(x)) = {}", xsinx.diff(x));

        // Quotient rule
        let sinx_over_x = sin(x) / x;
        println!("Quotient rule:  d/dx(sin(x)/x) = {}", sinx_over_x.diff(x));

        // Chain rule
        let exp_sin = exp(sin(x));
        println!("Chain rule:     d/dx(exp(sin(x))) = {}", exp_sin.diff(x));

        let ln_sqrt = ln(sqrt(pow(x, 2.0) + c(1.0)));
        println!("  nested:       d/dx(ln(√(x²+1))) = {}", ln_sqrt.diff(x));

        // General power: x^x
        let xx = pow(x, x);
        println!("General power:  d/dx(x^x) = {}", xx.diff(x));

        // Trig derivatives
        println!("Trig:           d/dx(sin(x)) = {}", sin(x).diff(x));
        println!("                d/dx(cos(x)) = {}", cos(x).diff(x));
        println!("                d/dx(tan(x)) = {}", tan(x).diff(x));

        // Inverse trig
        println!("Inv trig:       d/dx(asin(x)) = {}", asin(x).diff(x));
        println!("                d/dx(acos(x)) = {}", acos(x).diff(x));
        println!("                d/dx(atan(x)) = {}", atan(x).diff(x));

        // ============================================================
        // Expansion
        // ============================================================
        println!("\n=== Expansion ===\n");

        let dist = x * (a + b);
        println!("Distributive:   ({dist}).expand() = {}", dist.expand());

        let sq = pow(x + y, 2.0);
        println!("(x + y)^2:      {}", sq.expand());

        let cb = pow(x - y, 3.0);
        println!("(x - y)^3:      {}", cb.expand());

        // ============================================================
        // Collection
        // ============================================================
        println!("\n=== Collection ===\n");

        let scattered = a * x + b * x;
        println!("({scattered}).collect(x) = {}", scattered.collect(&x));

        // ============================================================
        // Evaluation & Substitution
        // ============================================================
        println!("\n=== Evaluation & Substitution ===\n");

        let expr = pow(x, 2.0) + 3.0 * x + 1.0;
        let vars = HashMap::from([("x", 2.0)]);
        println!("f(x) = {expr}");
        println!("f(2) = {}", expr.eval(&vars).unwrap());

        let subst = expr.subs(x, &(y + 1.0));
        println!("f(y+1) = {subst}");

        // Kinematics example
        println!("\n--- Kinematics ---");
        let (t, v0, a) = symbols!(t, v0, a);
        let s = v0 * t + 0.5 * a * pow(t, 2.0);
        println!("Position:   s(t) = {s}");
        let v = s.diff(t);
        println!("Velocity:   v(t) = ds/dt = {v}");
        let a_deriv = v.diff(t);
        println!("Accel:      a(t) = dv/dt = {a_deriv}");

        // ============================================================
        // Free variables
        // ============================================================
        println!("\n=== Free Variables ===\n");

        let fv_expr = x * y + sin(symbol("z"));
        println!("free_vars({fv_expr}) = {:?}", fv_expr.free_vars());

        // ============================================================
        // Linear Algebra
        // ============================================================
        println!("\n=== Linear Algebra ===\n");

        let v = SymVec::new([x, y]);
        println!("v = {v}");
        println!("v·v = {}", v.dot(&v));

        let m = SymMat::new(2, 2, [1.0, 2.0, 3.0, 4.0]);
        let mv = m.clone() * v;
        println!("M = {m}");
        println!("M·v = {mv}");
        println!("M^T = {}", m.transpose());

        // Jacobian
        let f1 = x * y;
        let f2 = pow(x, 2.0) + sin(y);
        let j = jacobian(&[f1, f2], &["x", "y"]);
        println!("\nJ([x·y, x²+sin(y)]) = {}", j.simplify());

        // ============================================================
        // Output formats
        // ============================================================
        println!("\n=== Output Formats ===\n");

        let complex = sin(x) / pow(y, 2.0);
        println!("Expression: {complex}");
        println!("LaTeX:      {}", complex.to_latex());
        println!("Rust f64:   {}", complex.to_rust("f64"));
        println!("Rust f32:   {}", complex.to_rust("f32"));

        println!("\nLaTeX samples:");
        println!("  sqrt(x)   → {}", sqrt(x).to_latex());
        println!("  |x|       → {}", abs(x).to_latex());
        println!("  exp(x)    → {}", exp(x).to_latex());
        println!("  M (LaTeX) → {}", m.to_latex());

        // ============================================================
        // Error suppression (SymPy comparison)
        // ============================================================
        println!("\n=== Error suppression ===\n");
        let (gamma, a, b, sigma) = symbols!(gamma, a, b, sigma);
        let r = (y - (a * x + b)) / sigma;
        let rs = gamma * atan(r / gamma);
        #[allow(non_snake_case)]
        let S = pow(rs, 2.0);
        println!("S = {S}");
        println!("dS/dx = {}", S.diff(x));

        // ============================================================
        // User-defined functions
        // ============================================================
        //
        // `simple_func1` / `simple_func2` wrap a symbolic body as a
        // named callable. Diff and codegen see the body when they need
        // to (chain rule, inlined emission) and the name otherwise --
        // the function participates in expression building like a
        // built-in. `simple_func*_derivs` and `extern_func*` provide
        // explicit derivatives and native eval when the body is
        // opaque or brittle to auto-diff.
        println!("\n=== User-defined functions ===\n");

        let sq = simple_func1("sq", |t| t * t);
        let hypot = simple_func2("hypot",
            |a, b| sqrt(a * a + b * b));

        let f = sq(x + 1.0) + hypot(x, y);
        println!("f = {f}");
        println!("df/dx = {}", f.diff(x));

        // ============================================================
        // Parsing expressions from strings
        // ============================================================
        println!("\n=== Parsing ===\n");

        let e1 = parse("x^2 + 3*x + 1").unwrap();
        println!("parse(\"x^2 + 3*x + 1\") = {e1}");

        let e2 = parse("sqrt(atan2(y, x) + pi)").unwrap();
        println!("parse(\"sqrt(atan2(y, x) + pi)\") = {e2}");

        let e3 = parse("exp(sin(x)) * cos(x)").unwrap();
        println!("parse(\"exp(sin(x)) * cos(x)\") = {e3}");
        println!("  d/dx = {}", e3.diff(x));

        let e4: E = "-a * x^2 + b".parse().unwrap();
        println!("FromStr: \"-a * x^2 + b\" = {e4}");

        // `parse_with_functions` consults a FunctionBag before
        // falling back to built-ins, so expressions typed at runtime
        // can reference user-defined `sq` / `hypot`.
        bag.add1(simple_func1("sq", |t| t * t)).unwrap();
        bag.add2(simple_func2("hypot",
            |a, b| sqrt(a * a + b * b))).unwrap();

        let e5 = parse_with_functions("sq(3) + hypot(3, 4)", &bag).unwrap();
        println!("parse_with_functions(\"sq(3) + hypot(3, 4)\") = {e5}");
        println!("  eval = {}", e5.eval(&HashMap::new()).unwrap());

        // Parsed user-defined functions diff + codegen like built-ins.
        let e6 = parse_with_functions("sq(x) + hypot(x, y)", &bag).unwrap();
        println!("d/dx(sq(x) + hypot(x, y)) = {}", e6.diff(x));
    }
}
