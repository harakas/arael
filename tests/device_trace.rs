//! A device solve traced (features `gpu-trace` and `gpu-cpu`): the trace of
//! a small bundle adjustment on the probed CPU device has the solve's
//! phases, every block's sections inside the block without overlap, and
//! the kernels' measured accesses inside described sections.
#![cfg(all(feature = "gpu-trace", feature = "gpu-cpu"))]

use arael::model::{CrossBlock, MixedEulerAngleParam, Param, SelfBlock};
use arael::refs::{self, Ref};
use arael::simple_lm::{GpuDevice, LmProblem};
use arael::utils::Float;
use arael::vect::{vect2, vect3};

/// The bundle-adjustment camera of benchmarks/bal: translation, rotation,
/// (f, k1, k2), and the depth under which an observation is dropped.
#[arael::model]
#[derive(Clone)]
pub struct Camera<T: Float> {
    pub t: Param<vect3<T>>,
    pub ea: MixedEulerAngleParam<T>,
    pub intr: Param<vect3<T>>,
    pub zdrop: T,
    pub hb: SelfBlock<Camera<T>, T>,
}

#[arael::model]
#[derive(Clone)]
pub struct Point<T: Float> {
    pub pos: Param<vect3<T>>,
    pub hb: SelfBlock<Point<T>, T>,
}

#[arael::model]
#[arael(constraint(hb, {
    let pc = cam.ea.rotation_matrix() * pt.pos + cam.t;
    let keep = -pc.z - cam.zdrop;
    let px = -pc.x / pc.z;
    let py = -pc.y / pc.z;
    let r2 = px * px + py * py;
    let d = 1.0 + r2 * (cam.intr.y + cam.intr.z * r2);
    [branch(keep, with_wide(cam.intr.x * d * px - obs.xy.x), 0.0),
     branch(keep, with_wide(cam.intr.x * d * py - obs.xy.y), 0.0)]
}))]
#[derive(Clone)]
pub struct Obs<T: Float> {
    #[arael(ref = root.cameras)]
    pub cam: Ref<Camera<T>>,
    #[arael(ref = root.points)]
    pub pt: Ref<Point<T>>,
    pub xy: vect2<T>,
    pub hb: CrossBlock<Camera<T>, Point<T>, T>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct Scene {
    pub cameras: refs::Vec<Camera<f64>>,
    pub points: refs::Vec<Point<f64>>,
    pub observations: std::vec::Vec<Obs<f64>>,
}

type Parts<T> = (refs::Vec<Camera<T>>, refs::Vec<Point<T>>, std::vec::Vec<Obs<T>>);

struct Lcg(u64);
impl Lcg {
    fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }
}

/// [`consistent_scene`] with `nc` cameras and `np` points, from `seed`.
fn consistent_scene_of<T: Float>(nc: usize, np: usize, seed: u64) -> Parts<T> {
    let mut rng = Lcg(seed);
    let c = |v: f64| T::from(v).unwrap();
    let mut cams = Vec::new();
    for k in 0..nc {
        let ea = arael::vect::vect3d::new(0.05 * rng.uniform(), 0.05 * rng.uniform(), 0.2 * rng.uniform());
        let t = arael::vect::vect3d::new(0.5 * rng.uniform(), 0.5 * rng.uniform(), 0.1 * rng.uniform());
        cams.push((ea, t, 500.0 + 10.0 * k as f64, 0.01 * rng.uniform(), 0.001 * rng.uniform()));
    }
    let pts: Vec<arael::vect::vect3d> = (0..np).map(|_| arael::vect::vect3d::new(2.0 * rng.uniform(), 2.0 * rng.uniform(), -5.0 + rng.uniform())).collect();
    let mut cameras = refs::Vec::new();
    for &(ea, t, f, k1, k2) in &cams {
        let moved = t + arael::vect::vect3d::new(0.01 * rng.uniform(), 0.01 * rng.uniform(), 0.01 * rng.uniform());
        cameras.push(Camera {
            t: Param::new(vect3::new(c(moved.x), c(moved.y), c(moved.z))),
            ea: MixedEulerAngleParam::new(ea),
            intr: Param::new(vect3::new(c(f), c(k1), c(k2))),
            zdrop: c(-1e30),
            hb: SelfBlock::new(),
        });
    }
    let mut points = refs::Vec::new();
    for p in &pts {
        let q = *p + arael::vect::vect3d::new(0.02 * rng.uniform(), 0.02 * rng.uniform(), 0.02 * rng.uniform());
        points.push(Point { pos: Param::new(vect3::new(c(q.x), c(q.y), c(q.z))), hb: SelfBlock::new() });
    }
    let mut observations = std::vec::Vec::new();
    for (pi, p) in pts.iter().enumerate() {
        for j in 0..(3 + pi % 2) {
            let ci = (pi + 2 * j) % nc;
            let (ea, t, f, k1, k2) = cams[ci];
            let pc = arael::matrix::matrix3d::rotation_from_euler_angles(ea) * *p + t;
            let (px, py) = (-pc.x / pc.z, -pc.y / pc.z);
            let r2 = px * px + py * py;
            let d = 1.0 + r2 * (k1 + k2 * r2);
            observations.push(Obs {
                cam: cameras.ref_at(ci),
                pt: points.ref_at(pi),
                xy: vect2::new(c(f * d * px + 0.5 * rng.uniform()), c(f * d * py + 0.5 * rng.uniform())),
                hb: CrossBlock::new(),
            });
        }
    }
    (cameras, points, observations)
}

/// The f32 twin of [`Scene`].
#[arael::model]
#[arael(root, f32)]
#[derive(Clone)]
pub struct SceneF {
    pub cameras: refs::Vec<Camera<f32>>,
    pub points: refs::Vec<Point<f32>>,
    pub observations: std::vec::Vec<Obs<f32>>,
}

/// Whether clang is there for the probe; says so when not.
fn has_clang() -> bool {
    let cxx = std::env::var("ARAEL_CPU_PROBE_CXX").unwrap_or_else(|_| "clang++".into());
    let has = std::process::Command::new(&cxx).arg("--version").output().is_ok_and(|o| o.status.success());
    if !has {
        eprintln!("the access probe needs clang (`{cxx}` not found): test skipped");
    }
    has
}

/// The number after `"key":` in a trace line.
fn number(line: &str, key: &str) -> Option<i64> {
    let at = line.find(&format!("\"{key}\":"))? + key.len() + 3;
    let end = line[at..].find(|c: char| !(c.is_ascii_digit() || c == '-')).map_or(line.len(), |e| at + e);
    line[at..end].parse().ok()
}

#[test]
fn a_traced_solve_names_its_phases_blocks_and_accesses() {
    if !has_clang() {
        return;
    }
    let path = std::env::temp_dir().join(format!("arael-device-trace-{}.jsonl", std::process::id()));
    // SAFETY: the test's process; nothing else reads the environment at this point.
    unsafe { std::env::set_var("ARAEL_CPU_PROBE", "1") };
    arael::__gpu::trace::start(&path).unwrap();
    let (cameras, points, observations) = consistent_scene_of::<f32>(6, 60, 11);
    let mut s = SceneF { cameras, points, observations };
    let options = arael::simple_lm::SparseFaerOptions::default()
        .with_policy(arael::simple_lm::SchurPolicy::Force)
        .with_envelope_schur(arael::simple_lm::EnvelopeMode::Never)
        .with_block_supernodal(arael::simple_lm::BlockSupernodalMode::Always)
        .with_gpu(Some(GpuDevice::cpu()))
        .with_gpu_assembly(true);
    let mut solver = arael::simple_lm::SparseFaer::<f32>::from_options(&options);
    let cfg = arael::simple_lm::LmConfig::<f32> { max_iters: 5, ..Default::default() };
    let r = s.solve_with(&mut solver, &cfg).unwrap();
    assert!(solver.plan().unwrap().gpu_assembly && r.iterations > 1);
    arael::__gpu::trace::stop();
    let text = std::fs::read_to_string(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    let lines: Vec<&str> = text.lines().collect();
    let kind = |l: &str| l.split('"').nth(3).unwrap().to_string();

    // the phases, as scopes at the top level
    let (mut depth, mut phases) = (0, Vec::new());
    for l in &lines {
        match kind(l).as_str() {
            "scope" => {
                if depth == 0 {
                    phases.push(l.split('"').nth(7).unwrap().to_string());
                }
                depth += 1;
            }
            "end" => depth -= 1,
            _ => {}
        }
    }
    assert_eq!(depth, 0, "every scope closes");
    for phase in ["setup", "compute", "start", "attempt", "linearize", "finish"] {
        assert!(phases.iter().any(|p| p == phase), "the phase `{phase}` in {phases:?}");
    }

    // every block's sections: inside the block, in no other's bytes
    let mut bytes = std::collections::HashMap::new();
    let mut described = std::collections::HashMap::new();
    for l in &lines {
        match kind(l).as_str() {
            "alloc" => {
                bytes.insert(number(l, "block").unwrap(), number(l, "bytes").unwrap());
            }
            "describe" => {
                let block = number(l, "block").unwrap();
                let name = l.split("\"name\":\"").nth(1).unwrap().split('"').next().unwrap().to_string();
                let mut sections: Vec<(i64, i64, String)> = l
                    .split("{\"name\":\"")
                    .skip(1)
                    .map(|s| (number(s, "at").unwrap(), number(s, "bytes").unwrap(), s.split('"').next().unwrap().to_string()))
                    .collect();
                assert!(!sections.is_empty(), "{name} has sections");
                sections.sort();
                for w in sections.windows(2) {
                    assert!(w[0].0 + w[0].1 <= w[1].0, "{name}: `{}` at {} of {} bytes runs into `{}` at {}", w[0].2, w[0].0, w[0].1, w[1].2, w[1].0);
                }
                let last = sections.last().unwrap();
                assert!(last.0 + last.1 <= bytes[&block], "{name}: `{}` ends past the block's {} bytes", last.2, bytes[&block]);
                described.insert(name, sections.len());
            }
            _ => {}
        }
    }
    for name in ["factor.plan", "factor.work", "schur.plan", "schur.work", "image", "lm"] {
        assert!(described.contains_key(name), "the block `{name}` is described: {described:?}");
    }
    assert!(described["image"] > 20 && described["schur.plan"] > 20, "the image's columns and the plan's arrays are sections: {described:?}");

    // an index section's words are in the trace: a tile column is records of
    // three words, classified, its first and last words kept
    let image = lines.iter().find(|l| kind(l) == "describe" && l.contains("\"name\":\"image\",\"sections\"")).expect("the image is described");
    let tile = image.split("{\"name\":\"").find(|s| s.starts_with("E0_TILE\"")).expect("the image has the column E0_TILE");
    assert!(tile.contains("\"record\":3") && tile.contains("\"sample\":{\"head\":["), "{tile}");
    // classified a field at a time: its stride is one value
    assert!(tile.contains("\"content\":{\"kind\":\"records\",") && tile.contains("\"period\":3,\"declared\":true,\"fields\":[{"), "{tile}");
    let fields: Vec<&str> = tile.split("\"fields\":[").nth(1).unwrap().split("},{").collect();
    assert!(fields.len() == 3 && fields[1].starts_with("\"kind\":\"constant\""), "the stride field is a constant: {tile}");
    let stride = tile.split("\"head\":[").nth(1).and_then(|h| h.split(',').nth(1)).and_then(|v| v.parse::<usize>().ok());
    assert!(stride.is_some_and(|s| s > 0), "the second word of a tile's record is its stride: {tile}");

    // launches are measured, and nothing they touch lies outside the sections
    let launches: Vec<&&str> = lines.iter().filter(|l| kind(l) == "launch").collect();
    assert!(launches.len() > 50 && launches.iter().all(|l| l.contains("\"acc\":[")), "{} launches, each measured", launches.len());
    let outside: Vec<&&&str> = launches.iter().filter(|l| l.contains("\"s\":-1")).collect();
    assert!(outside.is_empty(), "accesses outside the described sections: {}", outside.first().map_or("", |l| l));
    assert!(lines.iter().any(|l| kind(l) == "replay" && l.contains("\"acc\":[[")), "replayed graphs are measured too");
}
