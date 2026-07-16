use arael::prelude::*;

#[arael::model]
struct DataEntry {
    x: f32,
    y: f32,
}

// Linear model y = a*x + b with robust (suppressed) residuals.
// fit(...) is f32 throughout; use fit64(...) for an f64 model (Param<f64>
// fields, f64 config and result) -- same shorthand otherwise.
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

impl LinearModel {
    fn new(data: Vec<DataEntry>, sigma: f32) -> Self {
        let gamma = 2.0 * (25.0_f32).sqrt() / std::f32::consts::PI;
        LinearModel {
            a: Param::new(0.0),
            b: Param::new(0.0),
            data,
            sigma,
            gamma,
        }
    }

    fn fit_linear_regression(&mut self) {
        let mut sum_x = 0.0_f64;
        let mut sum_x2 = 0.0_f64;
        let mut sum_y = 0.0_f64;
        let mut sum_xy = 0.0_f64;
        for e in &self.data {
            let (x, y) = (e.x as f64, e.y as f64);
            sum_x += x;
            sum_x2 += x * x;
            sum_y += y;
            sum_xy += x * y;
        }
        let n = self.data.len() as f64;
        let a = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        let b = (sum_y - a * sum_x) / n;
        self.a = Param::new(a as f32);
        self.b = Param::new(b as f32);
    }
}

fn main() {
    let data = vec![
        DataEntry { x: -0.15640527, y: -0.09394677 },
        DataEntry { x: -0.14665490, y: -0.09246022 },
        DataEntry { x: -0.13697288, y: -0.09540069 },
        DataEntry { x: -0.12694226, y: -0.12290291 },
        DataEntry { x: -0.11715084, y: -0.07633987 },
        DataEntry { x: -0.10758017, y: -0.09448499 },
        DataEntry { x: -0.09716778, y: -0.21283103 },
        DataEntry { x: -0.08797396, y: -0.09011850 },
        DataEntry { x: -0.07798443, y: -0.20681172 },
        DataEntry { x: -0.06789591, y: -0.20985495 },
        DataEntry { x: -0.05861036, y: -0.09025026 },
        DataEntry { x: -0.04905056, y: -0.08905702 },
        DataEntry { x: -0.03925969, y: -0.09053987 },
        DataEntry { x: -0.02946681, y: -0.08921083 },
        DataEntry { x: -0.01946522, y: -0.08783635 },
        DataEntry { x: -0.00987463, y: -0.08561212 },
        DataEntry { x: -0.00014984, y: -0.08510941 },
        DataEntry { x:  0.00974805, y: -0.08513614 },
        DataEntry { x:  0.01940540, y: -0.08678824 },
        DataEntry { x:  0.02935162, y: -0.08533194 },
        DataEntry { x:  0.03921752, y: -0.08541373 },
    ];

    let mut model = LinearModel::new(data, 0.01);

    // Step 1: fit with simple linear regression (initial values)
    model.fit_linear_regression();
    let lin_a = model.a.value;
    let lin_b = model.b.value;
    println!("Linear regression: a={:.8}, b={:.8}", lin_a, lin_b);

    // Step 2: robust nonlinear fit with suppressed residuals
    let result = model.fit_with(&LmConfig {
        verbose: true,
        ..Default::default()
    });
    println!(
        "\nIterations: {}, cost: {:.6} -> {:.6}",
        result.iterations, result.start_cost, result.end_cost
    );
    println!("Robust fit: a={:.8}, b={:.8}", model.a.value, model.b.value);
    if let Some(sd) = model.get_stdev() {
        println!("Std dev:   a +/- {:.8}, b +/- {:.8}", sd[0], sd[1]);
    }

    // Step 3: print comparison table
    println!("\nX Y LINEAR ROBUST");
    for e in &model.data {
        println!(
            "{:.8} {:.8} {:.8} {:.8}",
            e.x,
            e.y,
            e.x * lin_a + lin_b,
            e.x * model.a.value + model.b.value,
        );
    }
}
