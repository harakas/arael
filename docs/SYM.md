# Symbolic Math Library (arael-sym)

`arael-sym` provides a lightweight computer algebra system built around a reference-counted expression tree (`E`). Expressions are constructed from symbols and constants, combined with standard arithmetic operators (which auto-simplify), and then differentiated, evaluated, pretty-printed, or compiled to Rust source code.

This crate is the symbolic engine behind the [`arael`](https://docs.rs/arael) optimization framework, where it powers compile-time constraint differentiation and code generation. It can also be used independently for any symbolic math task.

See [`examples/sym_demo.rs`](../examples/sym_demo.rs) for a runnable walkthrough covering every section below (`cargo run --example sym_demo`).

## Scope and limitations

`arael-sym` is focused on what's needed for nonlinear optimization: scalar expressions, differentiation, and code generation. Compared to a full CAS like Python's SymPy, it does **not** support:

- Symbolic integration
- Equation solving (solve for x)
- Symbolic matrix algebra (symbolic determinant, inverse, eigenvalues)
- Polynomial factoring, GCD, partial fractions
- Limits, series expansion, Taylor series
- Assumptions / domain reasoning (positive, real, integer)
- Pattern matching / rewrite rules
- Pretty-printing of intermediate simplification steps

## Basics

```rust
use arael::sym::*;
use arael::sym;

sym! {
    let (x, y) = symbols!(x, y);

    println!("x + y = {}", x + y);            // x + y
    println!("x * y - 1 = {}", x * y - 1.0);  // x * y - 1
    println!("x^2 = {}", pow(x, 2.0));        // x^2
}
```

The `symbols!` macro expands each bare identifier to
`symbol("<name>")` and returns a tuple.

The `sym!` macro auto-inserts `.clone()` on variable reuse, eliminating ownership boilerplate.

Every expression has type `arael::sym::E`, defined as `struct E(Rc<Expr>)`. Cloning is cheap (a reference-count bump) -- the `.clone()` calls `sym!` inserts don't duplicate the expression tree.

### Auto-simplification

All operations auto-simplify:

```rust
sym! {
    let (x, y) = symbols!(x, y);
    println!("{}", (x + y) / (x + y));  // 1
    println!("{}", x + 0.0);            // x
    println!("{}", x * 1.0);            // x
    println!("{}", x * 0.0);            // 0
    println!("{}", -(-x));              // x
    println!("{}", x - x);              // 0
    println!("{}", 3.0 * x + 2.0 * x);  // 5 * x
    println!("{}", x * x);              // x^2
}
```

## Derivatives

The library implements all standard calculus rules:

```rust
sym! {
    let x = symbol("x");

    // Power rule
    println!("d/dx(x^4) = {}", pow(x, 4.0).diff(x));
    // 4 * x^3

    // Product rule
    println!("d/dx(x*sin(x)) = {}", (x * sin(x)).diff(x));
    // x * cos(x) + sin(x)

    // Quotient rule
    println!("d/dx(sin(x)/x) = {}", (sin(x) / x).diff(x));
    // (x * cos(x) - sin(x)) / x^2

    // Chain rule
    println!("d/dx(exp(sin(x))) = {}", exp(sin(x)).diff(x));
    // cos(x) * exp(sin(x))

    // Nested chain rule
    let e = ln(sqrt(pow(x, 2.0) + 1.0));
    println!("d/dx(ln(sqrt(x^2+1))) = {}", e.diff(x));
    // x / sqrt(x^2 + 1)^2

    // General power
    println!("d/dx(x^x) = {}", pow(x, x).diff(x));
    // x^x * (ln(x) + 1)
}
```

### Trigonometric derivatives

```rust
sym! {
    let x = symbol("x");
    println!("d/dx(sin(x)) = {}", sin(x).diff(x));    // cos(x)
    println!("d/dx(cos(x)) = {}", cos(x).diff(x));    // -sin(x)
    println!("d/dx(tan(x)) = {}", tan(x).diff(x));    // 1 / cos(x)^2
    println!("d/dx(asin(x)) = {}", asin(x).diff(x));  // 1 / sqrt(-x^2 + 1)
    println!("d/dx(acos(x)) = {}", acos(x).diff(x));  // -1 / sqrt(-x^2 + 1)
    println!("d/dx(atan(x)) = {}", atan(x).diff(x));  // 1 / (x^2 + 1)
}
```

## Expansion and Collection

```rust
sym! {
    let (a, b, x, y) = symbols!(a, b, x, y);

    println!("{}", (x * (a + b)).expand());      // a * x + b * x
    println!("{}", pow(x + y, 2.0).expand());    // x^2 + 2 * x * y + y^2
    println!("{}", pow(x - y, 3.0).expand());    // x^3 - 3 * x^2 * y + 3 * x * y^2 - y^3
    println!("{}", (a * x + b * x).collect(x));  // x * (a + b)
}
```

## Evaluation and Substitution

```rust
use maplit::hashmap;

sym! {
    let (x, y) = symbols!(x, y);
    let f = pow(x, 2.0) + 3.0 * x + 1.0;

    let vars = hashmap!{ "x" => 2.0 };
    println!("f(2) = {}", f.eval(&vars).unwrap()); // 11

    println!("f(y+1) = {}", f.subs(x, &(y + 1.0)));
    // (y + 1)^2 + 3 * (y + 1) + 1
}
```

### Kinematics example

```rust
sym! {
    let (t, v0, a) = symbols!(t, v0, a);

    let s = 0.5 * a * pow(t, 2.0) + v0 * t;
    let v = s.diff(t);   // a * t + v0
    let acc = v.diff(t); // a

    println!("s(t) = {}", s);
    println!("v(t) = {}", v);
    println!("a(t) = {}", acc);
}
```

## Free Variables

```rust
sym! {
    let (x, y, z) = symbols!(x, y, z);
    let e = x * y + sin(z);
    println!("{:?}", e.free_vars()); // {"x", "y", "z"}
}
```

## Linear Algebra

```rust
sym! {
    let (x, y) = symbols!(x, y);
    let v = SymVec::new([x, y]);
    println!("v.v = {}", v.dot(&v));      // x^2 + y^2

    let m = SymMat::new(2, 2, [1.0, 2.0, 3.0, 4.0]);
    println!("M*v = {}", m * v);          // [x + 2 * y, 3 * x + 4 * y]
    println!("M^T = {}", m.transpose());  // [1, 3; 2, 4]

    let exprs = vec![x * y, pow(x, 2.0) + sin(y)];
    let j = jacobian(&exprs, &["x", "y"]);
    println!("J = {}", j);                // [y, x; 2 * x, cos(y)]
}
```

## Geometric Primitives

Fixed-shape symbolic vectors/matrices live in `arael_sym::geo`. They're used by `#[arael::model]` constraint bodies as the companions to the runtime `vect2f`/`vect3f`/`matrix2f`/`matrix3f` types -- the macro's body interpreter dispatches operators and methods through these.

| Symbolic         | Runtime companion        | Fields / shape                     |
| ---------------- | ------------------------ | ---------------------------------- |
| `vect2sym`       | `vect2f` / `vect2d`      | `.x .y`                            |
| `vect3sym`       | `vect3f` / `vect3d`      | `.x .y .z`                         |
| `matrix2sym`     | `matrix2f` / `matrix2d`  | `rows: [vect2sym; 2]`              |
| `matrix3sym`     | `matrix3f` / `matrix3d`  | `rows: [vect3sym; 3]`              |
| `quaternsym`     | `quaternf` / `quaternd`  | `.t` (scalar) + `.v` (vect3sym)    |

### vect2sym / vect3sym

```rust
let v = vect2sym::new("v");      // v.x, v.y as symbols
let u = vect2sym::new("u");
let w = vect2sym::from_components(symbol("a"), symbol("b")); // from scalar exprs
let dot     = v.clone() * u.clone();   // E (scalar)
let scaled  = v.clone() * symbol("k"); // vect2sym
let perp    = v.clone().across();      // (-v.y, v.x)
let cross_z = v.cross(&u);             // E: v.x*u.y - v.y*u.x
let n2      = v.square();              // v.x*v.x + v.y*v.y
let n       = v.norm();                // sqrt(v.square())
let u_hat   = v.unit();                // v / v.norm()
```

`vect3sym` has the same surface (`from_components`, `square`, `norm`,
`unit`, division by a scalar) with a 3D cross product -- `a.cross(&b)` or
the `%` operator, mirroring the runtime `vect3` -- and adds
`.rotation_matrix()` (intrinsic ZYX Euler angles -> `matrix3sym`) plus
the usual dot / scaled / negate / add / sub.

Both `from_components` constructors are also available inside
`#[arael(constraint(...))]` bodies (`vect3sym::from_components(a, b, c)`)
for assembling a vector from scalar expressions -- e.g. the vee of a
skew-symmetric matrix.

### matrix2sym

```rust
let r = matrix2sym::rotation(symbol("a"));    // 2D rotation
let rs = matrix2sym::rotation_from_sincos(symbol("s"), symbol("c"));
let rt = r.transpose();                       // R(-a)
let v  = vect2sym::new("v");
let rv = r.clone() * v.clone();               // matrix * vec  -> vect2sym
let vr = v * r.clone();                       // vec * matrix = M^T v
let rr = r.clone() * rt;                      // matrix * matrix -> matrix2sym
let a  = r.get_rotation_angle();              // E: atan2(m10, m00)
let d  = r.det();                             // E
```

Constructors: `identity()`, `from_rows(r0, r1)`, `from_cols(c0, c1)`,
`from_elements(m00, m01, m10, m11)` (row-major). Matrices add, subtract,
negate, and scale by a scalar (`m * k`, `k * m`).

### matrix3sym

```rust
let m = matrix3sym::new("m");                 // 9 symbols
let mt = m.transpose();
let ea = m.get_euler_angles();                // -> vect3sym
let c  = m.col(1);                            // column as vect3sym
let r  = m.row(2);                            // row as vect3sym
let w  = m.get_rotation_vector_small();       // vee((M - M^T)/2):
                                              // sin(theta) * axis, the
                                              // rotation vector near identity
let mv = m.clone() * vect3sym::new("v");      // -> vect3sym
let vm = vect3sym::new("v") * m.clone();      // vec * matrix = M^T v
let mm = m.clone() * m.clone();               // -> matrix3sym
let d  = m.det();                             // E
```

Constructors: `identity()`, `from_rows/from_cols` (three vectors),
`from_elements` (nine scalars, row-major),
`rotation_from_euler_angles(&ea)` (intrinsic ZYX, same as
`ea.rotation_matrix()`), `rotation_from_axis_angle(&axis, phi)` (axis must
be unit length). Add / Sub / Neg / scalar Mul as for `matrix2sym`.

### quaternsym

```rust
let q = quaternsym::new("q");                 // q.t + q.v (vect3sym)
let r = quaternsym::from_axis_angle(&vect3sym::new("ax"), symbol("ang"));
let p = quaternsym::from_euler_angles(&vect3sym::new("ea"));
let h = q.clone() * r.clone();                // Hamilton product
let c = q.conj();                             // negated vector part
let v = q.rotate(&vect3sym::new("v"));        // q * (0, v) * q'
let m = q.rotation_matrix();                  // -> matrix3sym
let e = q.get_euler_angles();                 // continuous: safe_asin pitch
let d = q.dot(&r);                            // E
let n = q.norm();                             // E
let u = q.clone().unit();
```

Mirrors the runtime `quatern` surface for everything branch-free; Add /
Sub / Neg / scalar Mul as for the vectors, plus `identity()`. The branchy
runtime operations (`pow`, `log`, `exp`, `slerp`, `from_two_vectors`,
`get_axis_angle`) are deliberately absent: constraint expressions must
stay continuous (see TODO.md). `get_euler_angles` uses `safe_asin` for
the pitch instead of the runtime's boundary branches.

### Use in constraint bodies

Inside `#[arael(constraint(..., { ... }))]` the body is interpreted (not compiled as Rust), so you write the *runtime* type names but the macro dispatches through the symbolic companions:

```rust
#[arael(constraint(hb, parent = lm, {
    // pose.gamma is a Param<f32>; lm.pos / pose.pos are Param<vect2f>.
    let world_angle = pose.gamma + frine.measured;
    let aligned = matrix2sym::rotation(world_angle).transpose()
                  * (lm.pos - pose.pos);
    [atan2(aligned.y, aligned.x) * frine.isigma]
}))]
```

You don't need `use arael::matrix::matrix2sym;` -- the macro matches `matrix2sym::rotation(...)` by path suffix, so any spelling that ends in `matrix2sym::rotation` works.

## Output Formatting / Code Generation

Any `E` renders three ways: `Display` for human reading, `to_latex()` for typeset output, and `to_rust("f64")` / `to_rust("f32")` for generated Rust code (the scalar type controls `powf` suffixes and literal formatting). Passing `""` omits the suffixes and lets literals infer from context. For code targeting a generic `T: Float` scope -- where an unsuffixed literal cannot infer -- `to_rust_generic()` wraps every literal as `__c(<lit>)` and expects the caller to provide a conversion helper `__c` returning the scalar type.

```rust
sym! {
    let (x, y) = symbols!(x, y);
    // Rosenbrock: a benchmark function with shared subterms.
    let f = pow(1.0 - x, 2.0) + 100.0 * pow(y - x * x, 2.0);

    println!("Display:  {f}");
    println!("LaTeX:    {}", f.to_latex());
    println!("Rust f64: {}", f.to_rust("f64"));
}
```

Output:

```text
Display:  (-x + 1)^2 + 100 * (-x^2 + y)^2
LaTeX:    \left(-x + 1\right)^{2} + 100 \cdot \left(-x^{2} + y\right)^{2}
Rust f64: (-x + 1.0_f64).powf(2.0_f64) + 100.0_f64 * (-x.powf(2.0_f64) + y).powf(2.0_f64)
```

## Common Subexpression Elimination

`E::is_zero()` tests for the literal constant zero -- after `simplify`, a
structurally-zero expression (e.g. a derivative of a residual with respect
to a parameter it does not touch) is exactly that, and codegen uses the
test to elide dead derivative emission and block-accumulation calls.

`cse(&[expr0, expr1, ...])` walks a batch of expressions, finds subtrees that appear more than once across the batch, and factors them into named intermediates. Paired with `to_rust`, it produces generated code that computes the shared work once.

Continuing the Rosenbrock example, its value and its two partial derivatives share `y - x*x` and `1 - x`:

```rust
sym! {
    let (x, y) = symbols!(x, y);
    let f = pow(1.0 - x, 2.0) + 100.0 * pow(y - x * x, 2.0);

    let batch = [f, f.diff(x), f.diff(y)];
    let (intermediates, simplified) = cse(&batch);

    for (name, val) in &intermediates {
        println!("let {name} = {};", val.to_rust("f64"));
    }
    let names = ["f", "df_dx", "df_dy"];
    for (i, s) in simplified.iter().enumerate() {
        println!("let {} = {};", names[i], s.to_rust("f64"));
    }
}
```

Output:

```text
let __x1 = -x + 1.0_f64;
let __x0 = -x.powf(2.0_f64) + y;
let f = __x1.powf(2.0_f64) + 100.0_f64 * __x0.powf(2.0_f64);
let df_dx = -400.0_f64 * (x * __x0) - 2.0_f64 * __x1;
let df_dy = 200.0_f64 * __x0;
```

`y - x*x` and `1 - x` each appear once, as `__x0` and `__x1`, rather than being recomputed at every use. CSE is applied automatically by `arael`'s constraint code-generation macro, where batches grow much larger -- one SLAM constraint went from 47000 ops to ~400 after CSE.

## Custom Functions

The library can build named function nodes (`Expr::Func`) that carry a body, formal parameters, and a behavioural kind. Use these when you want a function that participates in differentiation and code generation but stays distinct in the expression tree (e.g., to avoid CSE across the call boundary, or to call out to an extern Rust function at eval time).

Three families exist, picked based on how derivatives and numeric eval are produced:

| constructor                | body for diff / codegen | numeric eval                      | per-arg derivs |
| -------------------------- | ----------------------- | --------------------------------- | -------------- |
| `simple_func1` / `2` / `func`     | symbolic body inlined  | inlined body                      | auto-diffed    |
| `simple_func1_derivs` / `2_derivs` / `_derivs` | symbolic body inlined | inlined body | explicit       |
| `extern_func1` / `2` / `func`     | (none -- external)      | `eval_fn: fn(&[f64]) -> f64`      | explicit       |

Each constructor returns a *closure* that, when applied to actual argument expressions, produces an `Expr::Func` E.

### Symbolic with auto-diff

```rust
sym! {
    let (x, y) = symbols!(x, y);
    let sq = simple_func1("sq", |t| t * t);
    let f = sq(x) + sq(y);
    println!("f = {}", f);                  // sq(x) + sq(y)
    println!("df/dx = {}", f.diff(x));      // 2 * x
}
```

The body lambda runs once with placeholder symbols to capture the body expression; auto-differentiation operates on that body when the resulting Func is differentiated.

### Symbolic with explicit derivatives

When auto-diff would yield brittle or expensive derivatives, supply them explicitly:

```rust
sym! {
    let x = symbol("x");
    let safe_sq = simple_func1_derivs(
        "safe_sq",
        |t| t * t,
        |t| [2.0 * t],   // d/dt
    );
    let f = safe_sq(x);
    println!("df/dx = {}", f.diff(x));      // 2 * x
}
```

The arael-sym built-ins `safe_sqrt`, `safe_atan2`, `safe_asin`, `safe_acos`, `rad_diff`, and `rad_sum` are themselves built using these `simple_func*_derivs` / `extern_func*` constructors -- their source is a useful reference for non-trivial derivative wiring.

### Extern (call out to a Rust function at eval)

When the body is implemented natively (not as a symbolic expression), use `extern_func1/2/func`. The function is generated as a normal Rust call (`call_path(args...)`) by `to_rust(float_type)` codegen, and uses `eval_fn` for numeric evaluation.

```rust
sym! {
    // lerp(a, b, t) = a*(1-t) + b*t. Eval calls `my_crate::lerp`
    // at runtime; diff uses the supplied per-arg derivatives;
    // codegen emits `my_crate::lerp(a, b, t)`.
    let lerp = extern_func(
        "lerp", 3, "my_crate::lerp",
        // [d/da, d/db, d/dt] for lerp(a, b, t):
        |args| {
            let a = args[0].clone();
            let b = args[1].clone();
            let t = args[2].clone();
            vec![1.0 - t, t, b - a]
        },
        |args: &[f64]| args[0] * (1.0 - args[2]) + args[1] * args[2],
    );

    let (x, y, t) = symbols!(x, y, t);
    let e = lerp(vec![x, y, t]);
    println!("{e}");                   // lerp(x, y, t)
    println!("d/dx = {}", e.diff(x));  // 1 - t
    println!("d/dt = {}", e.diff(t));  // y - x
}
```

### `FuncKind`: the underlying enum

Every `Expr::Func` carries one of three `FuncKind` variants:

- `FuncKind::Symbolic { body }` -- the simplest case; the body is auto-differentiated and inlined for evaluation and codegen.
- `FuncKind::SymbolicDerivs { body, derivs }` -- body for evaluation/codegen, explicit per-argument derivatives.
- `FuncKind::Extern { derivs, eval_fn, call_path }` -- explicit derivatives, native eval function, codegen emits `call_path(args...)`.

You can construct `Expr::Func` values directly via `FuncKind` if you need to bypass the constructors above; usually the constructors are easier.

### Replacing functions in an expression

`e.replace_function(name, &|args| ...)` returns the expression with
every occurrence of the named function rebuilt by the closure from its
(already-transformed) arguments, recursing through the whole tree.
Built-in nodes match by their conventional name (`"sin"`, `"atan2"`,
`"pow"`, ... -- the names in `FUNCTIONS`), `Expr::Func` nodes by their
`name`; operators are not addressable.

```rust
sym! {
    let f = sin(atan2(y, x));
    let g = f.replace_function("atan2", &|args| fast_atan2(args[0].clone(), args[1].clone()));
    // sin(fast_atan2(y, x))
}
```

### `fast_atan` / `fast_atan2`

Approximate arctangents (max error < 1e-6 radians), the cheap
alternative when the residual tolerates it -- e.g. bearing residuals
in a localization pipeline. Codegen emits `arael::utils::fast_atan` /
`fast_atan2`, eval runs the same polynomial, and the derivatives are
the exact rational forms. Usable directly in constraint bodies by
name; the `#[arael(root, fast_atan)]` macro keyword rewrites every
plain `atan`/`atan2` in a model onto them via `replace_function`.

## Switching and Clamping: `heaviside`, `clamp`, `branch`, `min`, `max`, `sign`

### `heaviside(x)`

The Heaviside step function: 0 for `x < 0`, 1 for `x >= 0`, and 0 for NaN (interpreted eval and compiled code agree on this). Auto-differentiates to 0 everywhere -- the true derivative is a Dirac delta, whose applications in numeric calculations are limited, so we drop it. `H` is a parser-level alias: `parse("H(x)")` is the same as `parse("heaviside(x)")`.

### `branch(q, a, b)`

Selects between two subexpressions on the sign of a third: `branch(q, a, b) = q >= 0 ? a : b`, with the same `>= 0` / NaN sense as `heaviside` (a NaN condition selects `b`). It compiles to `if q >= 0.0 { a } else { b }` and interprets the same way, so **only the taken side is evaluated** -- the untaken arm may be undefined (a division by zero, a `ln` of a negative) without poisoning the result. Differentiation selects the taken side's derivative: `d/dvar branch(q, a, b) = branch(q, d a/dvar, d b/dvar)`. The switch `q` contributes nothing (the jump at `q = 0` is dropped, as with `heaviside`), so `branch` gives a piecewise residual the correct one-sided gradient with no spurious boundary term. Use it for asymmetric penalties, one-sided barriers, or guarding an inner computation valid on only one side of a condition.

### `min(a, b)`, `max(a, b)`, `sign(x)`

`min` and `max` return the smaller and larger of two arguments. They are built on `branch` (`min(a, b) = branch(b - a, a, b)`, `max(a, b) = branch(a - b, a, b)`), so they are differentiable away from the kink -- the derivative is the taken side's. At a tie (`a == b`) both return `a` and use `a`'s derivative.

`sign(x)` is `-1` for `x < 0`, `0` at `x == 0`, and `+1` for `x > 0`. It is built on `heaviside` (`sign(x) = heaviside(x) - heaviside(-x)`), so its derivative is 0 everywhere -- the jumps are dropped, as with `heaviside`.

All three accept `impl Into<E>`, so bare numeric arguments compose: `max(x, 0.0)` is a one-sided ramp.

### `clamp(value, lo, hi)`

Clamps the value to `[lo, hi]` for numeric evaluation. Differentiation passes through as if clamp were the identity on `value`: `d/dvar clamp(v, lo, hi)` is `v.diff(var)`, independent of the bounds. This makes clamp useful for *bounding the input* of an inner function whose math is undefined or numerically unstable outside `[lo, hi]`, without the derivative flattening to zero at the boundary.

```rust
sym! {
    let x = symbol("x");
    let safe = asin(clamp(x, -1.0, 1.0));
    println!("{}", safe);                   // asin(clamp(x, -1, 1))
    println!("d/dx = {}", safe.diff(x));    // 1 / sqrt(-clamp(x, -1, 1)^2 + 1)
}
```

The catch: at `|x| >= 1`, `clamp(x, -1, 1) = +/-1`, so `asin`'s derivative `1 / sqrt(1 - x^2)` becomes `1 / sqrt(0)` -- numerically NaN or infinite. The pass-through derivative is the right choice for inputs strictly inside `[lo, hi]`, but it doesn't tame a singularity at the boundary. When the inner function has one (as `asin` does at `|x| = 1`), the standard fix is to replace the auto-diffed derivative with an epsilon-regularised explicit derivative via `simple_func1_derivs`, as the next subsection shows.

### Example: building `safe_asin` from scratch

The arael-sym built-in `safe_asin` combines `clamp` for the body with an `epsilon`-regularised derivative supplied via `simple_func1_derivs`:

```rust
sym! {
    let x = symbol("x");
    let safe_asin = simple_func1_derivs(
        "safe_asin",
        // Body: clamp the input, then asin. Used for both numeric
        // evaluation and codegen.
        |x| asin(clamp(x, -1.0, 1.0)),
        // Derivative: 1 / sqrt(1 - xc^2 + eps^2), where `xc` is `x`
        // clamped to [-1, 1] so `1 - xc^2` stays non-negative for
        // any input (an unclamped `1 - x*x` at |x| > 1 goes
        // negative, sqrt NaNs, and the eps^2 term can't recover
        // it). The `identity` guard around `1 - xc^2` prevents the
        // simplifier from reordering the subtraction relative to
        // `+eps^2`, which would otherwise cancel in floating point
        // near |x| = 1. `epsilon_for(xc)` is machine epsilon anchored
        // to the TYPE of `xc` (see below).
        |x| {
            let xc = clamp(x, -1.0, 1.0);
            let e = epsilon_for(xc.clone());
            [1.0 / sqrt(identity(1.0 - xc.clone() * xc) + e.clone() * e)]
        },
    );
    let f = safe_asin(x);
    println!("{}",       f);             // safe_asin(x)
    println!("d/dx = {}", f.diff(x));    // uses epsilon_for(clamp(x, -1, 1)) as the guard
}
```

Why explicit derivatives? Auto-differentiating `asin(clamp(x, -1, 1))` would produce `(d/dx clamp) / sqrt(1 - clamp(x, -1, 1)^2)`, which still diverges at the boundary because `clamp`'s derivative is the identity. The regularised version replaces `sqrt(1 - x^2)` with `sqrt(1 - clamp(x, -1, 1)^2 + eps^2)` (clamp on both the body and the derivative input) and uses `identity` to defend the subtraction from simplifier reordering.

Why `epsilon_for(xc)` and not the nullary `epsilon()`? The constant machine epsilon is precision-DEPENDENT (`f32::EPSILON` != `f64::EPSILON`). In type-inferred codegen (unsuffixed literals, one body for both f32 and f64), `epsilon()` folds to a single literal and bakes in the f64 value -- an f32 constraint then gets a guard ~1e18x too small. `epsilon_for(anchor)` instead emits `arael::utils::epsilon_for(anchor)`, whose type parameter is inferred from a nearby concrete value (here `xc`, a field-typed expression), so f32 code gets `f32::EPSILON` and f64 code `f64::EPSILON`. Its symbolic eval is `f64::EPSILON` and its derivative w.r.t. the anchor is 0.

`epsilon_for` is a registered builtin, so you can call it directly in a constraint body (or in `parse`) to write your own precision-safe guard, e.g. `1 / sqrt(1 - x*x + epsilon_for(x)^2)` -- anchor it to a parameter or field so it picks up the model's precision.

The same pattern -- `simple_func*_derivs` plus `clamp` and/or `epsilon_for`-regularisation in the derivative -- is how `safe_acos`, `safe_sqrt`, `safe_atan2`, and similar are implemented.

## Robust loss kernels

`loss_geman_mcclure`, `loss_cauchy`, `loss_huber`, and `loss_tukey` take the squared residual norm `s = |r|^2` and return the robustified cost that replaces it. The scale (`k2`, `c2`) is SQUARED, in the same chi-square units as `s` -- the units inlier thresholds are naturally quoted in (a chi-square quantile such as 7.815 goes in unchanged). All four equal `s` near zero, so an inlier is left as plain least squares, and their weight `rho'(s)` is what a solver multiplies the block's gradient and Hessian by.

| Kernel | `rho(s)` | Weight `rho'(s)` | Behaviour |
|---|---|---|---|
| `loss_geman_mcclure(s, c2)` | `c2*s/(c2 + s)`, saturating at `c2` | `(c2/(c2 + s))^2` | smooth, redescends fast: a gross outlier's cost is capped and its weight falls toward zero |
| `loss_cauchy(s, c2)` | `c2 * ln(1 + s/c2)` | `c2/(c2 + s)` | smooth, redescends slowly |
| `loss_huber(s, k2)` | `s` if `s <= k2`, else `2*sqrt(k2*s) - k2` | `1`, then `sqrt(k2/s)` | convex, never rejects |
| `loss_tukey(s, c2)` | `(c2/3)(1 - (1 - s/c2)^3)`, capped at `c2/3` | `(1 - s/c2)^2`, then `0` | rejects `s >= c2` outright |

They are composed from `branch`, `sqrt`, and `ln`, so they differentiate, CSE, and generate code with no special handling.

```rust
let s = symbol("s");
let rho = loss_huber(s.clone(), constant(2.25)); // k2 = 2.25 (norm-axis 1.5)
let w = rho.diff("s");                   // the weight, sqrt(k2/s) past the knot
```

## Parsing

`parse(input)` reads an expression in standard infix notation: arithmetic, parentheses, function calls, the `^` operator for power, and the named constants `pi` and `e`. Numeric literals accept an optional scientific exponent (`1e-12`, `2.5E+2`). Anything else becomes a free symbol.

```rust
let e: E = "x^2 + 3*x + 1".parse().unwrap();
let f = parse("sqrt(atan2(y, x) + pi)").unwrap();
let g = parse("exp(sin(x)) * cos(x)").unwrap();
println!("d/dx = {}", g.diff("x")); // cos(x)^2 * exp(sin(x)) - exp(sin(x)) * sin(x)
```

Built-in functions recognised: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`, `exp`, `ln`, `log2`, `log10`, `sqrt`, `abs`, `heaviside` (alias `H`), `clamp`, `min`, `max`, `sign`, `pow`, `rad_diff`, `rad_sum`, `safe_atan2`, `safe_sqrt`, `safe_asin`, `safe_acos`, `fast_atan`, `fast_atan2`, `epsilon_for`, `cached`, `identity`. The full list is also enumerable at runtime via `function_names()` / `FUNCTIONS`.

### User-defined functions: `parse_with_functions` + `FunctionBag`

The plain `parse` only knows the built-in function set. To recognise additional functions defined at runtime, pass a `FunctionBag` to `parse_with_functions`:

```rust
let mut bag = FunctionBag::new();
bag.add1(simple_func1("sq", |t| t.clone() * t)).unwrap();

let e = parse_with_functions("sq(3) + 1", &bag).unwrap();
assert_eq!(e.eval(&HashMap::new()).unwrap(), 10.0);
```

The parser checks the bag first then falls back to built-ins, so:
- An empty `FunctionBag` behaves exactly like plain `parse` (built-ins always available).
- Adding a name that matches a built-in shadows it for the duration of the parse.
- `parse(s)` is shorthand for `parse_with_functions(s, &FunctionBag::new())`.

Ways to register a function in the bag:

```rust
// add1 / add2: pass the closure returned by simple_func1 / simple_func2
//              (or extern_func1 / extern_func2). The bag invokes it
//              once with placeholder symbols to extract name, params,
//              and kind.
bag.add1(simple_func1("sq", |t| t.clone() * t)).unwrap();
bag.add2(simple_func2("hypot",
    |a, b| sqrt(a.clone() * a + b.clone() * b))).unwrap();

// addN: n-ary closure. Takes `Vec<E>`, matching the shape of
//       `simple_func` / `simple_func_derivs` / `extern_func`. No
//       upper arity bound.
bag.addN(4, simple_func("blend", 4, |args: Vec<E>|
    args[0].clone() + args[1].clone() + args[2].clone() + args[3].clone()
)).unwrap();

// add: register an already-formed Expr::Func E directly (e.g. after
//      pre-applying a constructor to placeholder symbols).
let cube = simple_func1("cube", |t| t.clone() * t.clone() * t)(symbol("x"));
bag.add(cube).unwrap();

// add_symbolic: explicit name + parameter list + body. Use when the
//               body is an already-built E (e.g. from parse).
bag.add_symbolic("doublex", vec!["x".into()], parse("2*x").unwrap());
```

For escape-hatch cases there's also `add_with_kind(name, params, FuncKind)` that takes the parts directly.

Plus `remove(name) -> bool`, `contains(name)`, `names() -> Vec<String>`, `entries() -> impl Iterator<Item=(&str, usize)>` for management, and `get_info(name) -> Option<(&[String], &FuncKind)>` for read-only inspection.

#### Parameter shadowing

Formal parameters always shadow outer variables of the same name during the function body's evaluation. This is what you want for an interactive REPL: defining `sq(x) = x*x` after `x = 5` should still yield 9 when you call `sq(3)`, not 25.

```rust
use maplit::hashmap;

let mut bag = FunctionBag::new();
bag.add_symbolic("sq", vec!["x".into()], parse("x*x").unwrap());
let e = parse_with_functions("sq(3)", &bag).unwrap();
let vars = hashmap!{ "x" => 5.0 };
assert_eq!(e.eval(&vars).unwrap(), 9.0); // 3*3, not 5*5
```

See [`examples/calc_demo.rs`](https://github.com/harakas/arael/blob/master/examples/calc_demo.rs) for a complete bc-style REPL built on `FunctionBag` + `parse_with_functions`, with readline-style history.
