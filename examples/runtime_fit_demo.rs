/// Robust curve fitting with a runtime-parsed equation.
///
/// Demonstrates ExtendedModel: the model equation is a string parsed at
/// runtime with arael_sym, symbolically differentiated, then optimized
/// with Levenberg-Marquardt using TripletBlock for Gauss-Newton updates.
///
/// Usage:
///   cargo run --example runtime_fit_demo
///   cargo run --example runtime_fit_demo -- "a * x + b"
///   cargo run --example runtime_fit_demo -- "a * x^2 + b * x + c"
///   cargo run --example runtime_fit_demo -- --data points.csv "a * sin(x * b) + c"
///   cargo run --example runtime_fit_demo -- --init "a=1,b=0.5,c=-0.1" "a * sin(x * b) + c"
///
/// Variables named x and y are data columns. Everything else is a free
/// parameter to be optimized.
///
/// --data <path.csv>  Load x,y data from CSV (two columns, '#' comments)
/// --init <vals>      Initial parameter values, e.g. "a=1,b=0.5,c=-0.1"
/// --sigma <value>    Residual normalization (default: 0.01)

use std::collections::HashMap;
use arael::model::{ExtendedModel, Param, TripletBlock};
use arael::simple_lm::LmConfig;
use arael_sym::E;

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// One optimizable coefficient. Params are written by RegressionModel's
/// ExtendedModel::extended_compute directly into the global grad/hessian
/// via a TripletBlock, so Coefficient itself has no grad+diag to store.
#[arael::model]
#[arael(skip_self_block)]
struct Coefficient {
    value: Param<f64>,
}

/// Regression model whose equation is parsed at runtime.
#[arael::model]
#[arael(root, extended)]
struct RegressionModel {
    coeffs: arael::refs::Vec<Coefficient>,

    hb: TripletBlock<f64>,

    // --- runtime (not part of model tree) ---
    #[arael(skip)]
    residual_expr: Option<E>,
    #[arael(skip)]
    derivs: Vec<(String, u32, E)>,
    #[arael(skip)]
    data: Vec<(f64, f64)>,
    #[arael(skip)]
    param_names: Vec<String>,
}

impl ExtendedModel<f64> for RegressionModel {
    fn extended_cost(&self, params: &[f64]) -> f64 {
        let residual = match self.residual_expr { Some(ref e) => e, None => return 0.0 };
        let mut vars = self.build_vars(params);
        let mut cost = 0.0;
        for &(x, y) in &self.data {
            vars.insert("x", x);
            vars.insert("y", y);
            if let Ok(r) = residual.eval(&vars) {
                cost += r * r;
            }
        }
        cost
    }

    fn extended_compute(&mut self, params: &[f64], grad: &mut [f64]) {
        let residual = match self.residual_expr { Some(ref e) => e.clone(), None => return };
        let derivs: Vec<(u32, E)> = self.derivs.iter().map(|(_, idx, d)| (*idx, d.clone())).collect();
        let mut vars: HashMap<&str, f64> = HashMap::new();
        for (i, name) in self.param_names.iter().enumerate() {
            let idx = self.coeffs[i].value.index() as usize;
            vars.insert(name.as_str(), params[idx]);
        }
        let indices: Vec<u32> = derivs.iter().map(|(idx, _)| *idx).collect();

        for &(x, y) in &self.data {
            vars.insert("x", x);
            vars.insert("y", y);
            let r = match residual.eval(&vars) { Ok(v) => v, Err(_) => continue };
            let dr: Vec<f64> = derivs.iter()
                .filter_map(|(_, d)| d.eval(&vars).ok())
                .collect();
            if dr.len() == indices.len() {
                self.hb.add_residual(r, &indices, &dr, grad);
            }
        }
    }
}

impl RegressionModel {
    fn build_vars<'a>(&'a self, params: &[f64]) -> HashMap<&'a str, f64> {
        let mut vars = HashMap::new();
        for (i, name) in self.param_names.iter().enumerate() {
            let idx = self.coeffs[i].value.index() as usize;
            vars.insert(name.as_str(), params[idx]);
        }
        vars
    }
}

// ---------------------------------------------------------------------------
// Data (same as linear_demo.rs)
// ---------------------------------------------------------------------------

fn sample_data() -> Vec<(f64, f64)> {
    vec![
        (-0.15640527, -0.09394677), (-0.14665490, -0.09246022),
        (-0.13697288, -0.09540069), (-0.12694226, -0.12290291),
        (-0.11715084, -0.07633987), (-0.10758017, -0.09448499),
        (-0.09716778, -0.21283103), (-0.08797396, -0.09011850),
        (-0.07798443, -0.20681172), (-0.06789591, -0.20985495),
        (-0.05861036, -0.09025026), (-0.04905056, -0.08905702),
        (-0.03925969, -0.09053987), (-0.02946681, -0.08921083),
        (-0.01946522, -0.08783635), (-0.00987463, -0.08561212),
        (-0.00014984, -0.08510941), ( 0.00974805, -0.08513614),
        ( 0.01940540, -0.08678824), ( 0.02935162, -0.08533194),
        ( 0.03921752, -0.08541373),
    ]
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

fn parse_init(init_str: &str) -> HashMap<String, f64> {
    let mut map = HashMap::new();
    for pair in init_str.split(',') {
        let pair = pair.trim();
        if let Some((k, v)) = pair.split_once('=') {
            if let Ok(val) = v.trim().parse::<f64>() {
                map.insert(k.trim().to_string(), val);
            }
        }
    }
    map
}

fn build_model(equation: &str, data: Vec<(f64, f64)>, init: &HashMap<String, f64>, sigma: f64) -> RegressionModel {
    let model_expr = arael_sym::parse(equation)
        .unwrap_or_else(|e| panic!("failed to parse equation: {}", e.msg));

    // Partition symbols into data vars (x, y) and param vars
    let all_syms = model_expr.symbols();
    let data_vars: std::collections::HashSet<&str> = ["x", "y"].into();
    let mut param_names: Vec<String> = all_syms.iter()
        .filter(|s| !data_vars.contains(s.as_str()))
        .cloned()
        .collect();
    param_names.sort();

    println!("Equation:    y = {}", model_expr);
    println!("Parameters:  {:?}", param_names);
    print!("Initial:    ");
    for (i, name) in param_names.iter().enumerate() {
        let v = init.get(name).copied().unwrap_or(0.1 * (i as f64 + 1.0));
        print!(" {} = {}", name, v);
    }
    println!();
    println!("Sigma:       {}", sigma);
    println!("Data points: {}\n", data.len());

    // Robust residual: gamma * atan((model - y) / (sigma * gamma))
    let gamma = 2.0 * 25.0_f64.sqrt() / std::f64::consts::PI;
    let plain_r = (model_expr - arael_sym::symbol("y")) / arael_sym::constant(sigma);
    let residual_expr = arael_sym::constant(gamma)
        * arael_sym::atan(plain_r / arael_sym::constant(gamma));

    // Initial values: from --init if provided, otherwise small nonzero
    // defaults to avoid saddle points (e.g. a*sin(b*x)+c at all zeros).
    let mut coeffs = arael::refs::Vec::new();
    for (i, name) in param_names.iter().enumerate() {
        let v = init.get(name).copied().unwrap_or(0.1 * (i as f64 + 1.0));
        coeffs.push(Coefficient { value: Param::new(v) });
    }

    let mut model = RegressionModel {
        coeffs,
        hb: TripletBlock::new(),
        residual_expr: Some(residual_expr.clone()),
        derivs: Vec::new(),
        data,
        param_names: param_names.clone(),
    };

    // Serialize to assign param indices
    {
        let mut tmp = Vec::new();
        model.serialize64(&mut tmp);
    }

    // Pre-compute symbolic derivatives
    for (i, name) in param_names.iter().enumerate() {
        let d = residual_expr.diff(name.as_str());
        let idx = model.coeffs[i].value.index();
        model.derivs.push((name.clone(), idx, d));
    }

    model
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn load_csv(path: &str) -> Vec<(f64, f64)> {
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("failed to read {}: {}", path, e));
    let mut data = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') { continue; }
        // Skip header lines (first field not parseable as number)
        let parts: Vec<&str> = line.split([',', ' ', '\t'])
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();
        if parts.len() >= 2 {
            if let (Ok(x), Ok(y)) = (parts[0].parse::<f64>(), parts[1].parse::<f64>()) {
                data.push((x, y));
            }
        }
    }
    data
}

fn print_help() {
    println!(r#"Robust curve fitting with a runtime-parsed equation.

Usage:
  cargo run --example runtime_fit_demo -- [options] [equation]

Equation:
  Any expression in x; symbols other than x and y become free
  parameters to optimize. Default: "a * x + b".

Examples:
  cargo run --example runtime_fit_demo
  cargo run --example runtime_fit_demo -- "a * x + b"
  cargo run --example runtime_fit_demo -- "a * x^2 + b * x + c"
  cargo run --example runtime_fit_demo -- --data points.csv "a * sin(x * b) + c"
  cargo run --example runtime_fit_demo -- --init "a=1,b=0.5,c=-0.1" "a * sin(x * b) + c"

Options:
  --data <path.csv>   Load (x, y) data from CSV (two columns, '#' comments).
                      Default: a small built-in sample dataset.
  --init <vals>       Initial parameter values, e.g. "a=1,b=0.5,c=-0.1".
                      Defaults to 0.1, 0.2, 0.3, ... by parameter order.
  --sigma <value>     Residual normalization for the robust loss
                      (gamma * atan(r / (sigma * gamma))). Default: 0.01.
  -h, --help          Show this help and exit."#);
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print_help();
        return;
    }
    let mut equation = "a * x + b";
    let mut data_path: Option<&str> = None;
    let mut init_str: Option<&str> = None;
    let mut sigma = 0.01;
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--data" && i + 1 < args.len() {
            data_path = Some(&args[i + 1]);
            i += 2;
        } else if args[i] == "--init" && i + 1 < args.len() {
            init_str = Some(&args[i + 1]);
            i += 2;
        } else if args[i] == "--sigma" && i + 1 < args.len() {
            sigma = args[i + 1].parse().expect("--sigma value must be a number");
            i += 2;
        } else {
            equation = &args[i];
            i += 1;
        }
    }

    let data = if let Some(path) = data_path {
        load_csv(path)
    } else {
        sample_data()
    };
    let init = init_str.map_or_else(HashMap::new, parse_init);
    let mut model = build_model(equation, data, &init, sigma);

    let result = {
        let mut params = Vec::new();
        model.serialize64(&mut params);
        // conservative: a runtime-parsed user expression with no informed
        // starting values has unknown conditioning.
        let config = LmConfig::conservative().with_verbose(true);
        let result = arael::simple_lm::solve_sparse(&params, &mut model, &config).unwrap();
        model.deserialize64(&result.x);
        result
    };

    println!("\nIterations: {}, cost: {:.6} -> {:.6}", result.iterations, result.start_cost, result.end_cost);

    // arael's covariance API -- `assemble_covariance` (see docs/COVARIANCE.md) --
    // recovers this parameter covariance directly and is what you would normally
    // use. We compute it by hand here to show the derivation from the Hessian, and
    // to apply the reduced-chi-squared (Birge-ratio) scaling below, a correction
    // the API deliberately leaves to the caller.
    //
    // Parameter uncertainties from the inverse Hessian at the
    // solution. arael's `calc_grad_hessian_dense` writes the full
    // mathematical Hessian H = d2S/d theta2 = 2 JT J (under
    // Gauss-Newton; see TripletBlock::add_residual). The textbook
    // covariance is sigma_r2 * (JT J)^-1 = 2 sigma_r2 * H^-1, hence
    // the factor of 2 here. sigma_r2 is estimated from
    // s2 = end_cost / (N - p) -- the reduced-chi-squared / Birge-ratio
    // correction that compensates if the user-supplied `sigma`
    // mis-scales the residuals.
    // https://en.wikipedia.org/wiki/Reduced_chi-squared_statistic
    // With the robust gamma*atan loss the result is a local curvature
    // interval, not a strict Gaussian interval.
    use arael::simple_lm::LmProblem;
    let mut params_final = Vec::new();
    model.serialize64(&mut params_final);
    let n_p = params_final.len();
    let mut grad = vec![0.0_f64; n_p];
    let mut hess = vec![0.0_f64; n_p * n_p];
    model.calc_grad_hessian_dense(&params_final, &mut grad, &mut hess);
    let h = nalgebra::DMatrix::from_row_slice(n_p, n_p, &hess);
    let n_data = model.data.len() as f64;
    let dof = (n_data - n_p as f64).max(1.0);
    let s2 = result.end_cost / dof;
    let uncertainties: Vec<f64> = match h.try_inverse() {
        Some(inv) => (0..n_p).map(|i| (2.0 * s2 * inv[(i, i)]).max(0.0).sqrt()).collect(),
        None => vec![f64::NAN; n_p],
    };

    println!("Birge ratio:    s = {:.6} (s^2 = cost/(N-p) = {:.6})", s2.sqrt(), s2);
    print!("Result (+/- 1 sigma, ~68% CI):");
    for (i, name) in model.param_names.iter().enumerate() {
        let v = model.coeffs[i].value.value;
        let pi = model.coeffs[i].value.index() as usize;
        print!(" {} = {:.8} +/- {:.8}", name, v, uncertainties[pi]);
    }
    println!();

    // Comparison table
    println!("\n{:>12} {:>12} {:>12}", "x", "y", "model");
    for &(x, y) in &model.data {
        let mut vars: HashMap<&str, f64> = HashMap::new();
        vars.insert("x", x);
        for (i, name) in model.param_names.iter().enumerate() {
            vars.insert(name.as_str(), model.coeffs[i].value.value);
        }
        let model_y = arael_sym::parse(equation).unwrap().eval(&vars).unwrap_or(f64::NAN);
        println!("{:12.8} {:12.8} {:12.8}", x, y, model_y);
    }
}
