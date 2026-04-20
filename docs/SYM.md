# Symbolic Math Library (arael-sym)

The symbolic math library provides expression trees with automatic differentiation, algebraic simplification, code generation, and common subexpression elimination.

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
    println!("M*v = {}", m * &v);         // [x + 2 * y, 3 * x + 4 * y]
    println!("M^T = {}", m.transpose());  // [1, 3; 2, 4]

    let exprs = vec![x * y, pow(x, 2.0) + sin(y)];
    let j = jacobian(&exprs, &["x", "y"]);
    println!("J = {}", j);                // [y, x; 2 * x, cos(y)]
}
```

## Output Formats

```rust
sym! {
    let (x, y) = symbols!(x, y);
    let e = sin(x) / pow(y, 2.0);
    println!("Display: {}", e);                 // sin(x) / y^2
    println!("LaTeX:   {}", e.to_latex());      // \frac{\sin\left(x\right)}{y^{2}}
    println!("Rust:    {}", e.to_rust("f64"));  // x.sin() / y.powf(2.0_f64)
}
```

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

When the body is implemented natively (not as a symbolic expression), use `extern_func1/2/func`. The function is generated as a normal Rust call (`call_path(args...)`) in `to_rust_*` codegen, and uses `eval_fn` for numeric evaluation.

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

## Switching and Clamping: `heaviside`, `clamp`

Two built-ins act as conditional building blocks. Both are continuous in value but produce discontinuous (or zero) derivatives at their boundaries, so they're typically used inside larger expressions where the discontinuity is either intentional (a threshold penalty) or guarded by a clamp on the input domain.

### `heaviside(x)`

The Heaviside step function: 0 for `x < 0`, 1 for `x >= 0`. Auto-differentiates to 0 everywhere. `H` is a parser-level alias: `parse("H(x)")` is the same as `parse("heaviside(x)")`.

### `clamp(value, lo, hi)`

Clamps the value to `[lo, hi]`. Differentiation passes through `value` (the limits are treated as constant from the perspective of the derivative). Useful to *bound the input* of an inner function whose math is undefined or numerically unstable outside that range.

```rust
sym! {
    let x = symbol("x");
    let safe = asin(clamp(x, -1.0, 1.0));
    // safe is well-defined for any x; derivative at |x| > 1 is 0 (clamp's
    // derivative is the indicator of the interior).
    println!("{}", safe);                   // asin(clamp(x, -1, 1))
    println!("d/dx = {}", safe.diff(x));    // 1 / sqrt(-clamp(x, -1, 1)^2 + 1)
}
```

The catch: when `x` sits exactly at `-1` or `+1`, `asin`'s derivative is `1 / sqrt(1 - x^2)` which diverges. The next subsection shows the standard fix.

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
        // Derivative: 1 / sqrt(1 - x^2 + eps^2). The `identity` guard
        // around `1 - x^2` prevents the simplifier from reordering the
        // subtraction relative to the `+eps^2`, which would otherwise
        // cancel in floating point near |x| = 1.
        |x| [1.0 / sqrt(identity(1.0 - x * x) + epsilon() * epsilon())],
    );
    let f = safe_asin(x);
    println!("{}",       f);             // safe_asin(x)
    println!("d/dx = {}", f.diff(x));    // 1 / sqrt(identity(1 - x^2) + epsilon^2)
}
```

Why explicit derivatives? Auto-differentiating `asin(clamp(x, -1, 1))` would produce `(d/dx clamp) / sqrt(1 - clamp(x, -1, 1)^2)`, which still diverges at the boundary. The regularised version replaces `sqrt(1 - x^2)` with `sqrt(1 - x^2 + eps^2)` and uses `identity` to defend the `1 - x^2` subtraction from simplifier reordering.

The same pattern -- `simple_func*_derivs` plus `clamp` and/or `epsilon`-regularisation in the derivative -- is how `safe_acos`, `safe_sqrt`, `safe_atan2`, and similar are implemented. Their source is short and worth reading when you need a domain-safe primitive of your own.

## Common Subexpression Elimination

```rust
sym! {
    let (x, y) = symbols!(x, y);
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

`parse(input)` reads an expression in standard infix notation: arithmetic, parentheses, function calls, the `^` operator for power, and the named constants `pi` and `e`. Numeric literals accept an optional scientific exponent (`1e-12`, `2.5E+2`). Anything else becomes a free symbol.

```rust
let e: E = "x^2 + 3*x + 1".parse().unwrap();
let f = parse("sqrt(atan2(y, x) + pi)").unwrap();
let g = parse("exp(sin(x)) * cos(x)").unwrap();
println!("d/dx = {}", g.diff("x")); // cos(x)^2 * exp(sin(x)) - exp(sin(x)) * sin(x)
```

Built-in functions recognised: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`, `exp`, `ln`, `log2`, `log10`, `sqrt`, `abs`, `heaviside` (alias `H`), `clamp`, `pow`, `rad_diff`, `rad_sum`, `safe_atan2`, `safe_sqrt`, `safe_asin`, `safe_acos`, `identity`. The full list is also enumerable at runtime via `function_names()` / `FUNCTIONS`.

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
