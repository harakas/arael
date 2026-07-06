# Linear Regression with Robust Error Suppression

This example demonstrates fitting a linear model `y = a*x + b` to noisy data with outliers, using the Levenberg-Marquardt optimizer with robust error suppression.

## Model Definition

```rust
use arael::model::Param;
use arael::simple_lm::LmConfig;

#[arael::model]
struct DataEntry {
    x: f32,
    y: f32,
}

#[arael::model]
#[arael(fit(data, |e| {
    let plain_r = (a * e.x + b - e.y) / sigma;
    gamma * atan(plain_r / gamma)
}))]
struct LinearModel {
    a: Param<f32>,
    b: Param<f32>,
    data: Vec<DataEntry>,
    sigma: f32,
    gamma: f32,
}
```

### How it works

The `#[arael(fit(...))]` attribute defines the residual expression for each data point:

- `data` -- the field containing the data collection
- `|e| { ... }` -- closure over each data entry
- `a`, `b` -- references to `Param<f32>` fields (optimization variables)
- `e.x`, `e.y` -- data entry fields (constants per iteration)
- `sigma`, `gamma` -- plain `f32` fields (constants)

The residual uses robust error suppression: `gamma * atan(plain_r / gamma)`. For small residuals this behaves like the plain residual; for large residuals (outliers), the `atan` saturates, limiting their influence.

The macro auto-generates at compile time:
- `calc_cost(&mut self, params) -> f32` -- sum of squared residuals
- the `calc_grad_hessian_*` family (`_dense`, `_band`, `_sparse`, ...) --
  gradient and Gauss-Newton hessian per backend, returning the cost
- `fit(&mut self) -> LmResult` -- runs the LM optimizer
- `fit_with(&mut self, config) -> LmResult` -- with custom config

All derivatives are computed symbolically and compiled -- no runtime differentiation overhead.

## Data

21 data points with a nearly constant function around y = -0.087, plus several outliers (y ~ -0.21):

```rust
let data = vec![
    DataEntry { x: -0.15640527, y: -0.09394677 },
    DataEntry { x: -0.14665490, y: -0.09246022 },
    // ...
    DataEntry { x: -0.09716778, y: -0.21283103 },  // outlier
    // ...
    DataEntry { x: -0.07798443, y: -0.20681172 },  // outlier
    DataEntry { x: -0.06789591, y: -0.20985495 },  // outlier
    // ...
    DataEntry { x:  0.03921752, y: -0.08541373 },
];
```

## Fitting

### Step 1: Initial linear regression (closed-form)

```rust
model.fit_linear_regression();
// Linear regression: a=0.17499836, b=-0.09714453
```

The ordinary least squares fit is strongly influenced by the outliers, giving a steep slope.

### Step 2: Robust nonlinear fit

```rust
let result = model.fit_with(&LmConfig {
    abs_precision: 0.01,
    max_iters: 100,
    initial_lambda: 0.001,
    verbose: true,
    ..Default::default()
});
```

Output:
```
1/0: 95.0849->60.4124 / 34.6725, lambda=0.001
2/0: 60.4124->60.2542 / 0.158112, lambda=0.0002
3/0: 60.2542->60.2525 / 0.00171661, lambda=4e-5
4/0: 60.2525->60.2525 / 2.67029e-5, lambda=8e-6
5/0: 60.2525->60.2525 / -3.8147e-6, lambda=1.6e-6

Iterations: 5, cost: 95.084900 -> 60.252499
Robust fit: a=0.05899305, b=-0.08699960
```

The robust fit converges in 5 iterations. The slope drops from 0.175 to 0.059 -- the outliers are suppressed, and the fit tracks the majority of the data.

### Step 3: Comparison

```
X          Y           LINEAR       ROBUST
-0.156     -0.094      -0.125       -0.096
-0.097     -0.213      -0.114       -0.093   <-- outlier: LINEAR pulled, ROBUST stable
-0.078     -0.207      -0.111       -0.092   <-- outlier
-0.068     -0.210      -0.109       -0.091   <-- outlier
 0.010     -0.085      -0.095       -0.086
 0.039     -0.085      -0.090       -0.085
```

The ROBUST column shows nearly constant predictions (~-0.087) that match the inlier data, while LINEAR is tilted by the outliers.

## How the macro generates code

The `#[arael(fit(...))]` attribute:

1. Parses the residual expression from the attribute tokens
2. Identifies `Param` fields (`a`, `b`) vs constants (`sigma`, `gamma`) vs data fields (`e.x`, `e.y`)
3. Converts the expression to symbolic form using `arael-sym`
4. Differentiates symbolically w.r.t. each `Param`
5. Generates Rust code via `.to_rust("f32")`
6. Emits compiled `calc_cost`, `calc_grad_hessian`, `fit`, and `fit_with` methods

The generated gradient accumulation uses the Gauss-Newton approximation: `H = 2 * J^T * J`, which avoids computing second derivatives of the residual.

## Full source

See [examples/linear_demo.rs](../examples/linear_demo.rs).
