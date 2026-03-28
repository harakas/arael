# Symbolic Math Library (arael-sym)

The symbolic math library provides expression trees with automatic differentiation, algebraic simplification, code generation, and common subexpression elimination.

## Basics

```rust
use arael::sym::*;

arael::sym! {
    let x = symbol("x");
    let y = symbol("y");

    println!("x + y = {}", x + y);             // x + y
    println!("x * y - 1 = {}", x * y - 1.0);   // x * y - 1
    println!("x^2 = {}", pow(x, c(2.0)));       // x^2
}
```

The `arael::sym!` macro auto-inserts `.clone()` on variable reuse, eliminating ownership boilerplate.

### Auto-simplification

All operations auto-simplify:

```rust
arael::sym! {
    let x = symbol("x");
    let y = symbol("y");
    println!("{}", (x + y) / (x + y)); // 1
    println!("{}", x + 0.0);          // x
    println!("{}", x * 1.0);          // x
    println!("{}", x * 0.0);          // 0
    println!("{}", -(-x));             // x
    println!("{}", x - x);            // 0
    println!("{}", 3.0 * x + 2.0 * x); // 5 * x
    println!("{}", x * x);            // x^2
}
```

## Derivatives

The library implements all standard calculus rules:

```rust
arael::sym! {
    let x = symbol("x");

    // Power rule
    println!("d/dx(x^4) = {}", pow(x, c(4.0)).diff("x"));
    // 4 * x^3

    // Product rule
    println!("d/dx(x*sin(x)) = {}", (x * sin(x)).diff("x"));
    // x * cos(x) + sin(x)

    // Quotient rule
    println!("d/dx(sin(x)/x) = {}", (sin(x) / x).diff("x"));
    // (x * cos(x) - sin(x)) / x^2

    // Chain rule
    println!("d/dx(exp(sin(x))) = {}", exp(sin(x)).diff("x"));
    // cos(x) * exp(sin(x))

    // Nested chain rule
    let e = ln(sqrt(pow(x, c(2.0)) + 1.0));
    println!("d/dx(ln(sqrt(x^2+1))) = {}", e.diff("x"));
    // x / sqrt(x^2 + 1)^2

    // General power
    println!("d/dx(x^x) = {}", pow(x, x).diff("x"));
    // x^x * (ln(x) + 1)
}
```

### Trigonometric derivatives

```rust
arael::sym! {
    let x = symbol("x");
    println!("d/dx(sin(x)) = {}", sin(x).diff("x"));  // cos(x)
    println!("d/dx(cos(x)) = {}", cos(x).diff("x"));  // -sin(x)
    println!("d/dx(tan(x)) = {}", tan(x).diff("x"));  // 1 / cos(x)^2
    println!("d/dx(asin(x)) = {}", asin(x).diff("x")); // 1 / sqrt(-x^2 + 1)
    println!("d/dx(acos(x)) = {}", acos(x).diff("x")); // -1 / sqrt(-x^2 + 1)
    println!("d/dx(atan(x)) = {}", atan(x).diff("x")); // 1 / (x^2 + 1)
}
```

## Expansion and Collection

```rust
arael::sym! {
    let a = symbol("a");
    let b = symbol("b");
    let x = symbol("x");
    let y = symbol("y");

    println!("{}", (x * (a + b)).expand());  // a * x + b * x
    println!("{}", pow(x + y, c(2.0)).expand());  // x^2 + 2 * x * y + y^2
    println!("{}", pow(x - y, c(3.0)).expand());  // x^3 - 3 * x^2 * y + 3 * x * y^2 - y^3
    println!("{}", (a * x + b * x).collect("x")); // x * (a + b)
}
```

## Evaluation and Substitution

```rust
arael::sym! {
    let x = symbol("x");
    let y = symbol("y");
    let f = pow(x, c(2.0)) + 3.0 * x + 1.0;

    let vars = HashMap::from([("x", 2.0)]);
    println!("f(2) = {}", f.eval(&vars).unwrap()); // 11

    println!("f(y+1) = {}", f.subs("x", &(y + 1.0)));
    // (y + 1)^2 + 3 * (y + 1) + 1
}
```

### Kinematics example

```rust
arael::sym! {
    let t = symbol("t");
    let v0 = symbol("v0");
    let a = symbol("a");

    let s = 0.5 * a * pow(t, c(2.0)) + v0 * t;
    let v = s.diff("t");   // a * t + v0
    let acc = v.diff("t"); // a

    println!("s(t) = {}", s);
    println!("v(t) = {}", v);
    println!("a(t) = {}", acc);
}
```

## Free Variables

```rust
arael::sym! {
    let e = symbol("x") * symbol("y") + sin(symbol("z"));
    println!("{:?}", e.free_vars()); // {"x", "y", "z"}
}
```

## Linear Algebra

```rust
arael::sym! {
    let v = SymVec(vec![symbol("x"), symbol("y")]);
    println!("v.v = {}", v.dot(&v));   // x^2 + y^2

    let m = SymMat::new(2, 2, vec![c(1.0), c(2.0), c(3.0), c(4.0)]);
    println!("M*v = {}", m * &v);      // [x + 2 * y, 3 * x + 4 * y]
    println!("M^T = {}", m.transpose()); // [1, 3; 2, 4]

    let exprs = vec![symbol("x") * symbol("y"), pow(symbol("x"), c(2.0)) + sin(symbol("y"))];
    let j = jacobian(&exprs, &["x", "y"]);
    println!("J = {}", j); // [y, x; 2 * x, cos(y)]
}
```

## Output Formats

```rust
arael::sym! {
    let e = sin(symbol("x")) / pow(symbol("y"), c(2.0));
    println!("Display: {}", e);               // sin(x) / y^2
    println!("LaTeX:   {}", e.to_latex());     // \frac{\sin\left(x\right)}{y^{2}}
    println!("Rust:    {}", e.to_rust("f64")); // x.sin() / y.powf(2.0_f64)
}
```

## Common Subexpression Elimination

```rust
arael::sym! {
    let x = symbol("x");
    let y = symbol("y");
    let common = sin(x * y);
    let e1 = common + 1.0;
    let e2 = common * 2.0;

    let (intermediates, simplified) = cse(&[e1, e2]);
    // intermediates: [("__x0", sin(x * y))]
    // simplified: [__x0 + 1, 2 * __x0]
}
```

CSE is applied automatically in the constraint code generation macro, reducing generated code size dramatically (e.g., 47000 ops down to ~400 for a SLAM constraint).

## Parsing

```rust
let e: E = "x^2 + 3*x + 1".parse().unwrap();
let f = parse("sqrt(atan2(y, x) + pi)").unwrap();
let g = parse("exp(sin(x)) * cos(x)").unwrap();
println!("d/dx = {}", g.diff("x")); // cos(x)^2 * exp(sin(x)) - exp(sin(x)) * sin(x)
```
