// Bundle adjustment on a real "Bundle Adjustment in the Large" (Snavely et
// al.) Ladybug problem: N cameras and M 3D points tied together by pixel
// reprojection observations. Levenberg-Marquardt refines every camera pose +
// intrinsics and every point position at once, minimizing the total
// reprojection error.
//
// Run (defaults to the vendored 49-camera problem):
//   cargo run -r --example bal_demo [-- path/to/problem.txt]

use arael::model::{CrossBlock, EulerAngleParam, Param, SelfBlock};
use arael::matrix::matrix3d;
use arael::refs::{self, Ref};
use arael::simple_lm::{LmConfig, LmProblem};
use arael::vect::{vect2d, vect3d};

// ---------------------------------------------------------------------------
// Model (the BAL / Snavely reprojection model)
// ---------------------------------------------------------------------------

#[arael::model]
struct Camera {
    t: Param<vect3d>,         // translation
    ea: EulerAngleParam<f64>, // world-to-camera rotation
    intr: Param<vect3d>,      // (focal, k1, k2) -- focal + 2 radial-distortion coeffs
    hb: SelfBlock<Camera>,
}

#[arael::model]
struct Point {
    pos: Param<vect3d>,
    hb: SelfBlock<Point>,
}

// Reprojection residual: rotate+translate the point into the camera frame,
// project (BAL cameras look down -z), apply radial distortion 1 + k1 r^2 +
// k2 r^4 and the focal length, and compare to the observed pixel.
#[arael::model]
#[arael(constraint(hb, {
    let pc = cam.ea.rotation_matrix() * pt.pos + cam.t;
    let px = -pc.x / pc.z;
    let py = -pc.y / pc.z;
    let r2 = px * px + py * py;
    let d = 1.0 + r2 * (cam.intr.y + cam.intr.z * r2);
    [cam.intr.x * d * px - obs.xy.x,
     cam.intr.x * d * py - obs.xy.y]
}))]
struct Obs {
    #[arael(ref = root.cameras)] cam: Ref<Camera>,
    #[arael(ref = root.points)] pt: Ref<Point>,
    xy: vect2d,
    hb: CrossBlock<Camera, Point>,
}

#[arael::model]
#[arael(root)]
struct Scene {
    cameras: refs::Vec<Camera>,
    points: refs::Vec<Point>,
    observations: std::vec::Vec<Obs>,
}

// ---------------------------------------------------------------------------
// BAL file loader (self-contained). Format: "n_cams n_points n_obs", then
// n_obs lines "cam pt x y", then 9 values per camera (Rodrigues axis-angle 3,
// t 3, f, k1, k2), then 3 per point.
// ---------------------------------------------------------------------------

fn load(path: &str) -> Scene {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!("cannot read {path}: {e}\nfetch one with benchmarks/bal/fetch_datasets.sh, or pass a path")
    });
    let mut it = text.split_ascii_whitespace().map(|t| t.parse::<f64>().unwrap());
    let mut next = || it.next().expect("truncated BAL file");
    let (n_cams, n_points, n_obs) = (next() as usize, next() as usize, next() as usize);

    // Observations come first in the file; stash the raw indices, wire the
    // typed refs once the cameras and points exist.
    let mut obs_raw = std::vec::Vec::with_capacity(n_obs);
    for _ in 0..n_obs {
        obs_raw.push((next() as usize, next() as usize, vect2d::new(next(), next())));
    }

    let mut scene = Scene {
        cameras: refs::Vec::new(),
        points: refs::Vec::new(),
        observations: std::vec::Vec::new(),
    };
    for _ in 0..n_cams {
        let rodrigues = vect3d::new(next(), next(), next());
        let t = vect3d::new(next(), next(), next());
        let intr = vect3d::new(next(), next(), next()); // f, k1, k2
        scene.cameras.push(Camera {
            t: Param::new(t),
            ea: EulerAngleParam::new(rodrigues_to_matrix(rodrigues).get_euler_angles()),
            intr: Param::new(intr),
            hb: SelfBlock::new(),
        });
    }
    for _ in 0..n_points {
        scene.points.push(Point {
            pos: Param::new(vect3d::new(next(), next(), next())),
            hb: SelfBlock::new(),
        });
    }
    for (c, p, xy) in obs_raw {
        let cam = scene.cameras.ref_at(c);
        let pt = scene.points.ref_at(p);
        scene.observations.push(Obs { cam, pt, xy, hb: CrossBlock::new() });
    }
    scene
}

// Rodrigues axis-angle -> world-to-camera rotation matrix (Taylor guard at
// tiny angles).
fn rodrigues_to_matrix(w: vect3d) -> matrix3d {
    let t2 = w.square();
    if t2 > 1e-24 {
        let theta = t2.sqrt();
        matrix3d::rotation_from_axis_angle(w * (1.0 / theta), theta)
    } else {
        matrix3d::from_elements(1.0, -w.z, w.y, w.z, 1.0, -w.x, -w.y, w.x, 1.0)
    }
}

fn main() {
    let path = std::env::args().nth(1)
        .unwrap_or_else(|| "benchmarks/bal/datasets/problem-49-7776-pre.txt".to_string());
    let mut scene = load(&path);
    let n_obs = scene.observations.len();
    println!("BAL problem: {} cameras, {} points, {} observations\n",
        scene.cameras.len(), scene.points.len(), n_obs);

    // Gain-ratio (Nielsen) schedule -- the right default for the ill-
    // conditioned, gauge-free BA problem. No gauge prior: the 7-DOF gauge is
    // left to LM damping, as BAL is conventionally run.
    // Bundle adjustment is the regime the ill_conditioned preset is modeled
    // on: it brings the gain-ratio (Nielsen) damping driver. The overrides
    // tighten it for a short demo run.
    let cfg = LmConfig::ill_conditioned()
        .with_verbose(true) // print the per-iteration LM trace
        .with_max_iters(30)
        .with_initial_lambda(1e-4)
        .with_abs_precision(1e-5)
        .with_rel_precision(1e-5)
        .with_patience(1);

    let result = scene.solve_sparse(&cfg);

    // The cost is the summed squared pixel residual (2 per observation), so
    // the per-observation RMS reprojection error is sqrt(cost / n_obs).
    let rms = |cost: f64| (cost / n_obs as f64).sqrt();
    println!("\ncost {:.1} -> {:.1}  in {} iterations ({} accepted)",
        result.start_cost, result.end_cost, result.iterations, result.accepted_iterations);
    println!("reprojection RMS: {:.3} px -> {:.3} px", rms(result.start_cost), rms(result.end_cost));
}
