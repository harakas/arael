//! Every built-in function of arael-sym in a constraint body assembles on
//! the device (feature `gpu-cuda` or `gpu-hip`) as on the CPU: per function
//! a model whose instances all take that function's constraint (the others
//! guarded off), checked in an f64 root, an f32 root and an f32 root with
//! `wide = df32`, in both of the device's wide scalars, at points across
//! the function's domain and off its kinks and cuts, and at its edges.
#![cfg(any(feature = "gpu-cuda", feature = "gpu-hip", feature = "gpu-cpu"))]

use arael::df32;
use arael::model::{Param, SelfBlock};
use arael::refs;
use arael::simple_lm::{LmProblemInternals, RootProblem};
use arael::utils::Float;

/// A function under test per constraint, `f` picking which one an
/// instance takes; `k` is the index of the selects, `a` a data argument.
#[arael::model]
#[arael(constraint(hb, guard = self.f == 0, { [sin(fz.x) * fz.y] }))]
#[arael(constraint(hb, guard = self.f == 1, { [cos(fz.x) * fz.y] }))]
#[arael(constraint(hb, guard = self.f == 2, { [tan(fz.x) * fz.y] }))]
#[arael(constraint(hb, guard = self.f == 3, { [asin(fz.x) * fz.y] }))]
#[arael(constraint(hb, guard = self.f == 4, { [acos(fz.x) * fz.y] }))]
#[arael(constraint(hb, guard = self.f == 5, { [atan(fz.x) * fz.y] }))]
#[arael(constraint(hb, guard = self.f == 6, { [sinh(fz.x) * fz.y] }))]
#[arael(constraint(hb, guard = self.f == 7, { [cosh(fz.x) * fz.y] }))]
#[arael(constraint(hb, guard = self.f == 8, { [tanh(fz.x) * fz.y] }))]
#[arael(constraint(hb, guard = self.f == 9, { [exp(fz.x) * fz.y] }))]
#[arael(constraint(hb, guard = self.f == 10, { [exp_m1(fz.x) * fz.y] }))]
#[arael(constraint(hb, guard = self.f == 11, { [ln(fz.x) * fz.y] }))]
#[arael(constraint(hb, guard = self.f == 12, { [ln_1p(fz.x) * fz.y] }))]
#[arael(constraint(hb, guard = self.f == 13, { [log2(fz.x) * fz.y] }))]
#[arael(constraint(hb, guard = self.f == 14, { [log10(fz.x) * fz.y] }))]
#[arael(constraint(hb, guard = self.f == 15, { [sqrt(fz.x) * fz.y] }))]
#[arael(constraint(hb, guard = self.f == 16, { [abs(fz.x) * fz.y] }))]
#[arael(constraint(hb, guard = self.f == 17, { [heaviside(fz.x) * fz.y] }))]
#[arael(constraint(hb, guard = self.f == 18, { [sign(fz.x) * fz.y] }))]
#[arael(constraint(hb, guard = self.f == 19, { [identity(fz.x - fz.y) * fz.y] }))]
#[arael(constraint(hb, guard = self.f == 20, { [cached(sin(fz.x)) * fz.y] }))]
#[arael(constraint(hb, guard = self.f == 21, { [with_wide(fz.x * fz.y - fz.x * fz.a)] }))]
#[arael(constraint(hb, guard = self.f == 22, { [safe_sqrt(fz.x) * fz.y] }))]
#[arael(constraint(hb, guard = self.f == 23, { [safe_asin(fz.x) * fz.y] }))]
#[arael(constraint(hb, guard = self.f == 24, { [safe_acos(fz.x) * fz.y] }))]
#[arael(constraint(hb, guard = self.f == 25, { [epsilon_for(fz.x) * fz.y * 1.0e6] }))]
#[arael(constraint(hb, guard = self.f == 26, { [fast_atan(fz.x) * fz.y] }))]
#[arael(constraint(hb, guard = self.f == 27, { [atan2(fz.y, fz.x)] }))]
#[arael(constraint(hb, guard = self.f == 28, { [pow(fz.x, fz.y)] }))]
#[arael(constraint(hb, guard = self.f == 29, { [pow(fz.x, 3.0) * fz.y] }))]
#[arael(constraint(hb, guard = self.f == 30, { [pow(fz.x, -2.0) * fz.y] }))]
#[arael(constraint(hb, guard = self.f == 31, { [pow(fz.x, 0.5) * fz.y] }))]
#[arael(constraint(hb, guard = self.f == 32, { [pow(fz.x, fz.a) * fz.y] }))]
#[arael(constraint(hb, guard = self.f == 33, { [safe_atan2(fz.y, fz.x)] }))]
#[arael(constraint(hb, guard = self.f == 34, { [fast_atan2(fz.y, fz.x)] }))]
#[arael(constraint(hb, guard = self.f == 35, { [rad_diff(fz.x, fz.y)] }))]
#[arael(constraint(hb, guard = self.f == 36, { [rad_sum(fz.x, fz.y)] }))]
#[arael(constraint(hb, guard = self.f == 37, { [min(fz.x, fz.y) * fz.y] }))]
#[arael(constraint(hb, guard = self.f == 38, { [max(fz.x, fz.y) * fz.y] }))]
#[arael(constraint(hb, guard = self.f == 39, { [loss_geman_mcclure(fz.x * fz.x, fz.a) * fz.y] }))]
#[arael(constraint(hb, guard = self.f == 40, { [loss_cauchy(fz.x * fz.x, fz.a) * fz.y] }))]
#[arael(constraint(hb, guard = self.f == 41, { [loss_huber(fz.x * fz.x, fz.a) * fz.y] }))]
#[arael(constraint(hb, guard = self.f == 42, { [loss_tukey(fz.x * fz.x, fz.a) * fz.y] }))]
#[arael(constraint(hb, guard = self.f == 43, { [loss_soft_l1(fz.x * fz.x, fz.a) * fz.y] }))]
#[arael(constraint(hb, guard = self.f == 44, { [clamp(fz.x, -0.5, 0.5) * fz.y] }))]
#[arael(constraint(hb, guard = self.f == 45, { [clamp(fz.x, fz.y - 1.0, fz.y + 1.0) * fz.y] }))]
#[arael(constraint(hb, guard = self.f == 46, { [branch(fz.x - fz.y, sin(fz.x), cos(fz.y))] }))]
#[arael(constraint(hb, guard = self.f == 47, { [loss_select(fz.k, fz.x * fz.x, fz.a) * fz.y] }))]
#[arael(constraint(hb, guard = self.f == 48, { [select(fz.k, sin(fz.x), fz.x * fz.y, cos(fz.y), fz.x - fz.y, exp(fz.x), fz.y * fz.y)] }))]
#[arael(constraint(hb, guard = self.f == 49, { [select_or(fz.k, sin(fz.x), fz.x * fz.y, cos(fz.y), fz.x - fz.y, fz.y * fz.y)] }))]
#[arael(constraint(hb, guard = self.f == 50, { [multibranch(fz.x, sin(fz.y), fz.y - 0.5, cos(fz.x), fz.x * fz.y)] }))]
#[arael(constraint(hb, guard = self.f == 51, { [piecewise(fz.x, fz.x * fz.y, -0.5, sin(fz.x), 0.5, fz.y * fz.y)] }))]
struct Fz<T: Float> {
    x: Param<T>,
    y: Param<T>,
    a: T,
    f: u32,
    k: u32,
    hb: SelfBlock<Fz<T>, T>,
}

#[arael::model]
#[arael(root)]
struct Zoo64 {
    fs: refs::Vec<Fz<f64>>,
}

#[arael::model]
#[arael(root, f32)]
struct Zoo32 {
    fs: refs::Vec<Fz<f32>>,
}

/// The f32 root whose `with_wide` scopes compute in df32 on the CPU.
#[arael::model]
#[arael(root, f32, wide = df32)]
struct Zoo32W {
    fs: refs::Vec<Fz<f32>>,
}

/// The functions in the order of their constraints: a label starting with
/// the built-in's name, the ranges of `x` and `y`, and `a`.
const ZOO: &[(&str, (f64, f64), (f64, f64), f64)] = &[
    ("sin", (-3.0, 3.0), Y, A),
    ("cos", (-3.0, 3.0), Y, A),
    ("tan", (-1.3, 1.3), Y, A),
    ("asin", (-0.95, 0.95), Y, A),
    ("acos", (-0.95, 0.95), Y, A),
    ("atan", (-3.0, 3.0), Y, A),
    ("sinh", (-2.0, 2.0), Y, A),
    ("cosh", (-2.0, 2.0), Y, A),
    ("tanh", (-2.0, 2.0), Y, A),
    ("exp", (-2.0, 2.0), Y, A),
    // small arguments, where exp(x) - 1 and ln(1 + x) lose their digits
    // (most of f64's, all of f32's)
    ("exp_m1", (-3e-7, 3e-7), Y, A),
    ("ln", (0.1, 5.0), Y, A),
    ("ln_1p", (-3e-7, 3e-7), Y, A),
    ("log2", (0.1, 5.0), Y, A),
    ("log10", (0.1, 5.0), Y, A),
    ("sqrt", (0.1, 5.0), Y, A),
    ("abs", (-2.0, 2.0), Y, A),
    ("heaviside", (-2.0, 2.0), Y, A),
    ("sign", (-2.0, 2.0), Y, A),
    ("identity", (-2.0, 2.0), Y, A),
    ("cached", (-3.0, 3.0), Y, A),
    ("with_wide", (-2.0, 2.0), Y, A),
    ("safe_sqrt", (0.01, 4.0), Y, A),
    ("safe_asin", (-0.95, 0.95), Y, A),
    ("safe_acos", (-0.95, 0.95), Y, A),
    // scaled so an f64 epsilon in an f32 root, or the reverse, shows
    ("epsilon_for", (-2.0, 2.0), Y, A),
    ("fast_atan", (-3.0, 3.0), Y, A),
    ("atan2", S, S, A),
    ("pow(x, y)", (0.2, 3.0), S, A),
    ("pow(x, 3)", S, Y, A),
    ("pow(x, -2)", (0.3, 2.0), Y, A),
    ("pow(x, 0.5)", (0.1, 4.0), Y, A),
    ("pow(x, a)", (0.2, 3.0), Y, 1.7),
    ("safe_atan2", S, S, A),
    ("fast_atan2", S, S, A),
    // differences and sums past +-pi, which wrap
    ("rad_diff", (-3.0, 3.0), (-3.0, 3.0), A),
    ("rad_sum", (-3.0, 3.0), (-3.0, 3.0), A),
    ("min", S, S, A),
    ("max", S, S, A),
    // the squared norm on both sides of each kernel's knee at s = a
    ("loss_geman_mcclure", L, Y, 1.0),
    ("loss_cauchy", L, Y, 1.0),
    ("loss_huber", L, Y, 1.0),
    ("loss_tukey", L, Y, 2.0),
    ("loss_soft_l1", L, Y, 1.0),
    ("clamp(x, -0.5, 0.5)", S, Y, A),
    ("clamp(x, y - 1, y + 1)", S, S, A),
    ("branch", S, S, A),
    ("loss_select", L, Y, 1.2),
    ("select", S, S, A),
    ("select_or", S, S, A),
    ("multibranch", S, S, A),
    ("piecewise", S, S, A),
];

/// The default range of `y` and value of `a`.
const Y: (f64, f64) = (0.5, 2.0);
const A: f64 = 0.5;
/// A range across zero.
const S: (f64, f64) = (-2.0, 2.0);
/// The range of `x` for the losses: `x * x` on both sides of their knees.
const L: (f64, f64) = (-2.4, 2.4);

/// Instances per function. `x` and `y` step through their ranges in two
/// orders, off the ends; the steps keep `x` and `y` 0.08 apart over `S`,
/// clear of the kinks of `min`, `max`, `branch` and the clamps, and the
/// sides of the cuts of `atan2` and `rad_diff`.
const N: usize = 12;

fn instance<T: Float>(f: usize, i: usize) -> Fz<T> {
    let (_, (x0, x1), (y0, y1), a) = ZOO[f];
    let t = (i as f64 + 0.37) / N as f64;
    let u = ((i * 5) % N) as f64 + 0.61;
    let u = u / N as f64;
    Fz {
        x: Param::new(T::from(x0 + (x1 - x0) * t).unwrap()),
        y: Param::new(T::from(y0 + (y1 - y0) * u).unwrap()),
        a: T::from(a).unwrap(),
        f: f as u32,
        k: (i % 6) as u32,
        hb: SelfBlock::new(),
    }
}

/// The instances of the functions `fs`, `N` each.
fn instances<T: Float>(fs: &[usize]) -> refs::Vec<Fz<T>> {
    let mut v = refs::Vec::new();
    for &f in fs {
        for i in 0..N {
            v.push(instance(f, i));
        }
    }
    v
}

/// The functions at their edges: a label, the zoo entry whose constraint
/// it takes, `x`, `y` and `a` there, and the f64 root's tolerance;
/// evaluated where they are, not moved off.
const EDGES: &[(&str, &str, f64, f64, f64, f64)] = &[
    ("safe_sqrt at 0", "safe_sqrt", 0.0, 1.5, A, T64),
    ("safe_asin at 1", "safe_asin", 1.0, 1.5, A, T64),
    ("safe_asin at -1", "safe_asin", -1.0, 1.5, A, T64),
    ("safe_acos at 1", "safe_acos", 1.0, 1.5, A, T64),
    ("safe_acos at -1", "safe_acos", -1.0, 1.5, A, T64),
    ("safe_atan2 at (0, 0)", "safe_atan2", 0.0, 0.0, A, T64),
    ("abs at 0", "abs", 0.0, 1.5, A, T64),
    ("heaviside at 0", "heaviside", 0.0, 1.5, A, T64),
    ("sign at 0", "sign", 0.0, 1.5, A, T64),
    ("min at a tie", "min", 0.7, 0.7, A, T64),
    ("max at a tie", "max", 0.7, 0.7, A, T64),
    ("branch at 0", "branch", 0.7, 0.7, A, T64),
    ("clamp at its upper bound", "clamp(x, -0.5, 0.5)", 0.5, 1.5, A, T64),
    ("clamp at its lower bound", "clamp(x, -0.5, 0.5)", -0.5, 1.5, A, T64),
    ("clamp at a bound of y", "clamp(x, y - 1, y + 1)", 1.5, 0.5, A, T64),
    ("piecewise at its first break", "piecewise", -0.5, 1.5, A, T64),
    ("piecewise at its second break", "piecewise", 0.5, 1.5, A, T64),
    ("multibranch at its first switch", "multibranch", 0.0, 1.5, A, T64),
    ("multibranch at its second switch", "multibranch", -1.0, 0.5, A, T64),
    ("loss_huber at its knee", "loss_huber", 1.0, 1.5, 1.0, T64),
    ("pow(x, 3) at 0", "pow(x, 3)", 0.0, 1.5, A, T64),
    // y = 1 + 2^-17, a = 1 + 2^-20: x * y - x * a cancels 17 bits. The f32
    // roots' products are exact in f64 and df32, while f32 rounds x * a
    // even where an fma takes x * y whole (a 0.5% error). In the f64 root
    // the CPU's two roundings and the device's fma differ there by 1e-11
    // of the cost; with df32 sums the device computes the scope in df32
    // (48 bits), 3.6e-10.
    ("with_wide in a cancellation", "with_wide", 1.3, 1.0 + 1.0 / 131072.0, 1.0 + 1.0 / 1048576.0, 1e-9),
];

/// The f64 root's tolerance.
const T64: f64 = 1e-11;

/// The instances of the edge `e`: its point, with each select index.
fn edge_instances<T: Float>(e: usize) -> refs::Vec<Fz<T>> {
    let (_, entry, x, y, a, _) = EDGES[e];
    let f = ZOO.iter().position(|(label, ..)| *label == entry).expect("an edge of a zoo entry");
    let mut v = refs::Vec::new();
    for k in 0..6 {
        v.push(Fz {
            x: Param::new(T::from(x).unwrap()),
            y: Param::new(T::from(y).unwrap()),
            a: T::from(a).unwrap(),
            f: f as u32,
            k,
            hb: SelfBlock::new(),
        });
    }
    v
}

/// The parameters moved by a thousandth of their own size, so they keep
/// their signs and the sides of the kinks.
fn perturbed<T: Float>(x: &[T]) -> Vec<T> {
    x.iter().enumerate().map(|(i, &v)| v * T::from(1.0 + 1e-3 * ((i % 7) as f64 - 3.0)).unwrap()).collect()
}

/// The device: CUDA when it is compiled in, else HIP, else the CPU device.
#[cfg(feature = "gpu-cuda")]
type Dev = arael::__gpu::cuda::CudaDevice;
#[cfg(all(feature = "gpu-hip", not(feature = "gpu-cuda")))]
type Dev = arael::__gpu::hip::HipDevice;
#[cfg(all(feature = "gpu-cpu", not(any(feature = "gpu-cuda", feature = "gpu-hip"))))]
type Dev = arael::__gpu::cpu::CpuDevice;

/// One device per test: its compiled kernels are found again in memory.
fn device() -> std::sync::Arc<Dev> {
    std::sync::Arc::new(Dev::new(0).unwrap())
}

/// The CPU's and the device's assembly of `root` against each other on
/// `dev`, in both wide scalars, at its parameters `moved` off (else where
/// they are): a line per disagreement beyond `tol` (cost relative,
/// gradient and Hessian relative to their largest CPU entries).
fn disagreements<T: Float, R: RootProblem<T> + LmProblemInternals<T>>(dev: &std::sync::Arc<Dev>, mut root: R, moved: bool, what: &str, tol: f64) -> Vec<String> {
    use arael::__gpu::Wide;
    let mut out = Vec::new();
    if let Some(why) = LmProblemInternals::<T>::device_unsupported(&root) {
        return vec![format!("{what}: no device form: {why}")];
    }
    let mut x = Vec::new();
    root.serialize(&mut x);
    let x1 = if moved { perturbed(&x) } else { x.clone() };
    for wide in [Wide::F64, Wide::Df32] {
        let chk = arael::__gpu::device_check(&mut root, &x, &x1, dev, wide);
        let cost_err = (chk.device_cost - chk.cpu_cost).abs() / chk.cpu_cost.abs().max(f64::MIN_POSITIVE);
        if !(cost_err <= tol && chk.grad_err <= tol && chk.hess_err <= tol) {
            out.push(format!("{what} {wide:?}: cost {} against the CPU's {}, gradient {:.1e}, Hessian {:.1e}", chk.device_cost, chk.cpu_cost, chk.grad_err, chk.hess_err));
        }
    }
    out
}

#[test]
fn the_zoo_covers_every_built_in_function() {
    let missing: Vec<&str> = arael::sym::function_names()
        .filter(|name| !ZOO.iter().any(|(label, ..)| label.split('(').next() == Some(*name)))
        .collect();
    assert!(missing.is_empty(), "built-ins without a constraint in the zoo: {missing:?}");
}

/// Each function on its own, in an f64 root (the CPU and the device
/// agreeing to rounding) and the f32 roots (the CPU summing in f32, the
/// device in its wide scalar; `with_wide` on the CPU in the root's wide
/// scalar, f64 or df32, on the device in its own).
#[test]
fn every_built_in_function_assembles_on_the_device_as_on_the_cpu() {
    let dev = device();
    let mut bad = Vec::new();
    for (f, (label, ..)) in ZOO.iter().enumerate() {
        bad.extend(disagreements(&dev, Zoo64 { fs: instances(&[f]) }, true, &format!("{label} f64"), T64));
        bad.extend(disagreements(&dev, Zoo32 { fs: instances(&[f]) }, true, &format!("{label} f32"), 1e-4));
        bad.extend(disagreements(&dev, Zoo32W { fs: instances(&[f]) }, true, &format!("{label} f32 wide df32"), 1e-4));
    }
    assert!(bad.is_empty(), "{} disagreements:\n{}", bad.len(), bad.join("\n"));
}

/// The functions exactly at their edges: domain ends, kinks, ties,
/// switches.
#[test]
fn the_built_in_functions_at_their_edges_assemble_on_the_device_as_on_the_cpu() {
    let dev = device();
    let mut bad = Vec::new();
    for (e, &(label, .., tol64)) in EDGES.iter().enumerate() {
        bad.extend(disagreements(&dev, Zoo64 { fs: edge_instances(e) }, false, &format!("{label} f64"), tol64));
        bad.extend(disagreements(&dev, Zoo32 { fs: edge_instances(e) }, false, &format!("{label} f32"), 1e-4));
        bad.extend(disagreements(&dev, Zoo32W { fs: edge_instances(e) }, false, &format!("{label} f32 wide df32"), 1e-4));
    }
    assert!(bad.is_empty(), "{} disagreements:\n{}", bad.len(), bad.join("\n"));
}

/// Every function in one model: every constraint has instances at once.
#[test]
fn the_whole_zoo_assembles_on_the_device_as_on_the_cpu() {
    let all: Vec<usize> = (0..ZOO.len()).collect();
    let dev = device();
    let mut bad = disagreements(&dev, Zoo64 { fs: instances(&all) }, true, "the zoo f64", T64);
    bad.extend(disagreements(&dev, Zoo32 { fs: instances(&all) }, true, "the zoo f32", 1e-4));
    bad.extend(disagreements(&dev, Zoo32W { fs: instances(&all) }, true, "the zoo f32 wide df32", 1e-4));
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}
