//! Does nano-gemm beat our own GEMM kernels on the shapes the Schur reduction
//! actually multiplies?
//!
//! The Schur reduction's inner operation is `dst -= C_a * Z_b` on small
//! column-major tiles whose widths are entity sizes: 3 (a point), 6 (a pose),
//! 9 (a BAL camera). Three contenders, for each shape and both orientations of
//! the left-hand tile:
//!
//!   * `fixed`   -- the const-generic kernel, fully unrolled at compile time.
//!                  Used for the shapes in `FIXED_SHAPES`.
//!   * `runtime` -- a plain loop with the widths known only at run time. What
//!                  the fallback USED to be.
//!   * `nano`    -- nano-gemm. What the fallback IS.
//!
//! The last table asks a separate question: nano-gemm needs a `Plan` for the
//! shape, and the fallback only learns the shape at run time. Build one per call,
//! or keep them in a cache? It measures both, and the cache does not pay.
//!
//! Run it:
//!
//! ```text
//! cargo run --release -p arael-faer --example gemmbench
//! ```
//!
//! On x86, nano-gemm picks AVX2 by itself; its AVX-512 microkernels are behind
//! a feature, and that is where its published numbers come from:
//!
//! ```text
//! RUSTFLAGS="-C target-cpu=native" \
//!     cargo run --release -p arael-faer --features x86-v4 --example gemmbench
//! ```
//!
//! The interesting column is `nano vs fixed`: above 1.0 the unrolled kernel
//! wins and the fallback should stay a fallback; below 1.0 on the hot shapes
//! ((6,3,6) for SLAM, (9,3,9) for bundle adjustment) nano-gemm would be worth
//! taking over the whole path on that machine.

use std::time::Instant;

/// (wa, we, wb, what it is). The first block is what `FIXED_SHAPES` covers; the
/// second has no unrolled kernel and goes to the fallback.
const SHAPES: &[(usize, usize, usize, &str)] = &[
    (3, 3, 3, "pgo 2D"),
    (6, 3, 6, "slam pose x point"),
    (6, 6, 6, "pgo 3D pose x pose"),
    (9, 3, 9, "bal camera x point"),
    (6, 1, 6, "slam, 1-wide elim"),
    (7, 3, 7, ""),
    (3, 2, 3, ""),
    (2, 3, 2, ""),
    (12, 3, 12, "no fixed kernel"),
    (9, 6, 9, "no fixed kernel"),
    (5, 3, 7, "no fixed kernel"),
];

/// The scalar: arael solves in both, and the SIMD width -- so the verdict --
/// differs between them.
trait Real: Copy + std::ops::Add<Output = Self> + std::ops::Sub<Output = Self> + std::ops::Mul<Output = Self> {
    const ZERO: Self;
    const ONE: Self;
    const NEG_ONE: Self;
    fn of(x: f64) -> Self;
    fn abs_diff(self, other: Self) -> f64;
    /// dst column-major always; lhs column-major only when untransposed.
    fn plan(m: usize, n: usize, k: usize, trans: bool) -> nano_gemm::Plan<Self>
    where
        Self: Sized;
}

impl Real for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const NEG_ONE: Self = -1.0;
    fn of(x: f64) -> Self {
        x
    }
    fn abs_diff(self, o: Self) -> f64 {
        (self - o).abs()
    }
    fn plan(m: usize, n: usize, k: usize, trans: bool) -> nano_gemm::Plan<Self> {
        if trans {
            nano_gemm::Plan::<f64>::new_f64(m, n, k)
        } else {
            nano_gemm::Plan::<f64>::new_colmajor_lhs_and_dst_f64(m, n, k)
        }
    }
}

impl Real for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const NEG_ONE: Self = -1.0;
    fn of(x: f64) -> Self {
        x as f32
    }
    fn abs_diff(self, o: Self) -> f64 {
        (self - o).abs() as f64
    }
    fn plan(m: usize, n: usize, k: usize, trans: bool) -> nano_gemm::Plan<Self> {
        if trans {
            nano_gemm::Plan::<f32>::new_f32(m, n, k)
        } else {
            nano_gemm::Plan::<f32>::new_colmajor_lhs_and_dst_f32(m, n, k)
        }
    }
}

/// The unrolled kernel, copied EXACTLY from `schur.rs`. The column slicing is
/// not cosmetic: it is what lets the compiler hoist the bounds checks and
/// vectorize the inner loop. Writing the same arithmetic with flat indexing
/// (`dst[i + c * WA]`) costs 2x, and benchmarking that instead would flatter
/// nano-gemm against a kernel we do not actually ship.
#[inline]
fn fixed<T: Real, const WA: usize, const WE: usize, const WB: usize>(
    dst: &mut [T],
    ca: &[T],
    zb: &[T],
    trans: bool,
) {
    if !trans {
        for c in 0..WB {
            for k in 0..WE {
                let z = zb[k + c * WE];
                let dcol = &mut dst[c * WA..(c + 1) * WA];
                let acol = &ca[k * WA..(k + 1) * WA];
                for i in 0..WA {
                    dcol[i] = dcol[i] - acol[i] * z;
                }
            }
        }
    } else {
        for c in 0..WB {
            for i in 0..WA {
                let arow = &ca[i * WE..(i + 1) * WE];
                let zcol = &zb[c * WE..(c + 1) * WE];
                let mut s = T::ZERO;
                for k in 0..WE {
                    s = s + arow[k] * zcol[k];
                }
                dst[i + c * WA] = dst[i + c * WA] - s;
            }
        }
    }
}

/// The loop the fallback used to be: the same code with the widths at run time,
/// so the trip counts -- and the vectorization -- are unknown to the compiler.
#[inline]
fn runtime<T: Real>(dst: &mut [T], ca: &[T], trans: bool, wa: usize, we: usize, zb: &[T], wb: usize) {
    if !trans {
        for c in 0..wb {
            for k in 0..we {
                let z = zb[k + c * we];
                let dcol = &mut dst[c * wa..(c + 1) * wa];
                let acol = &ca[k * wa..(k + 1) * wa];
                for i in 0..wa {
                    dcol[i] = dcol[i] - acol[i] * z;
                }
            }
        }
    } else {
        for c in 0..wb {
            for i in 0..wa {
                let arow = &ca[i * we..(i + 1) * we];
                let zcol = &zb[c * we..(c + 1) * we];
                let mut s = T::ZERO;
                for k in 0..we {
                    s = s + arow[k] * zcol[k];
                }
                dst[i + c * wa] = dst[i + c * wa] - s;
            }
        }
    }
}

/// nano-gemm: `dst = alpha*dst + beta*(lhs*rhs)`, so ours is alpha=1, beta=-1.
/// A transposed lhs is a stride swap, not a copy.
#[inline]
fn nano<T: Real>(
    plan: &nano_gemm::Plan<T>,
    dst: &mut [T],
    ca: &[T],
    trans: bool,
    wa: usize,
    we: usize,
    zb: &[T],
    wb: usize,
) {
    let (lhs_rs, lhs_cs) = if trans { (we as isize, 1) } else { (1, wa as isize) };
    unsafe {
        plan.execute_unchecked(
            wa, wb, we,
            dst.as_mut_ptr(), 1, wa as isize,
            ca.as_ptr(), lhs_rs, lhs_cs,
            zb.as_ptr(), 1, we as isize,
            T::ONE, T::NEG_ONE,
            false, false,
        );
    }
}

/// Best of `rounds`, each of `reps` GEMMs. Returns ns per GEMM.
fn best_ns(rounds: usize, reps: usize, mut f: impl FnMut()) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..rounds {
        let t = Instant::now();
        for _ in 0..reps {
            f();
        }
        best = best.min(t.elapsed().as_secs_f64() * 1e9 / reps as f64);
    }
    best
}

fn bench<T: Real + std::fmt::Debug>(scalar: &str, trans: bool) {
    const ROUNDS: usize = 12;
    const REPS: usize = 20_000;

    println!(
        "\n=== {scalar}, lhs {} ===",
        if trans { "transposed (stride swap)" } else { "direct" }
    );
    println!(
        "{:<12} {:>20} {:>9} {:>9} {:>9}  {:>13}",
        "(wa,we,wb)", "what it is", "fixed", "runtime", "nano", "nano vs fixed"
    );

    for &(wa, we, wb, what) in SHAPES {
        let ca: Vec<T> = (0..wa * we).map(|i| T::of(1.0 + i as f64 * 0.01)).collect();
        let zb: Vec<T> = (0..we * wb).map(|i| T::of(0.5 - i as f64 * 0.003)).collect();
        let (mut d1, mut d2, mut d3) = (
            vec![T::ZERO; wa * wb],
            vec![T::ZERO; wa * wb],
            vec![T::ZERO; wa * wb],
        );
        // Built once per shape: the reduction hits the same few shapes thousands
        // of times, so this is the fair comparison (and the library caches them).
        let plan = T::plan(wa, wb, we, trans);

        macro_rules! call_fixed {
            ($dst:expr) => {
                match (wa, we, wb) {
                    (3, 3, 3) => fixed::<T, 3, 3, 3>($dst, &ca, &zb, trans),
                    (6, 3, 6) => fixed::<T, 6, 3, 6>($dst, &ca, &zb, trans),
                    (6, 6, 6) => fixed::<T, 6, 6, 6>($dst, &ca, &zb, trans),
                    (9, 3, 9) => fixed::<T, 9, 3, 9>($dst, &ca, &zb, trans),
                    (6, 1, 6) => fixed::<T, 6, 1, 6>($dst, &ca, &zb, trans),
                    (7, 3, 7) => fixed::<T, 7, 3, 7>($dst, &ca, &zb, trans),
                    (3, 2, 3) => fixed::<T, 3, 2, 3>($dst, &ca, &zb, trans),
                    (2, 3, 2) => fixed::<T, 2, 3, 2>($dst, &ca, &zb, trans),
                    (12, 3, 12) => fixed::<T, 12, 3, 12>($dst, &ca, &zb, trans),
                    (9, 6, 9) => fixed::<T, 9, 6, 9>($dst, &ca, &zb, trans),
                    (5, 3, 7) => fixed::<T, 5, 3, 7>($dst, &ca, &zb, trans),
                    _ => unreachable!(),
                }
            };
        }

        // Correctness before timing: a fast wrong kernel is worth nothing.
        call_fixed!(&mut d1);
        runtime(&mut d2, &ca, trans, wa, we, &zb, wb);
        nano(&plan, &mut d3, &ca, trans, wa, we, &zb, wb);
        let tol = 1e-3;
        assert!(
            d1.iter().zip(&d2).all(|(a, b)| a.abs_diff(*b) < tol)
                && d1.iter().zip(&d3).all(|(a, b)| a.abs_diff(*b) < tol),
            "({wa},{we},{wb}) trans={trans} disagree: fixed {d1:?} nano {d3:?}"
        );

        let t_fixed = best_ns(ROUNDS, REPS, || {
            call_fixed!(&mut d1);
            std::hint::black_box(&d1);
        });
        let t_rt = best_ns(ROUNDS, REPS, || {
            runtime(&mut d2, &ca, trans, wa, we, &zb, wb);
            std::hint::black_box(&d2);
        });
        let t_nano = best_ns(ROUNDS, REPS, || {
            nano(&plan, &mut d3, &ca, trans, wa, we, &zb, wb);
            std::hint::black_box(&d3);
        });

        let ratio = t_nano / t_fixed;
        println!(
            "{:<12} {:>20} {:>8.1}n {:>8.1}n {:>8.1}n  {:>12.2}x{}",
            format!("({wa},{we},{wb})"),
            what,
            t_fixed,
            t_rt,
            t_nano,
            ratio,
            if ratio < 0.95 { "  <- nano wins" } else { "" }
        );
    }
}

/// What nano-gemm will actually dispatch to here -- an x86 result means nothing
/// without it, since AVX-512 is where its published numbers come from.
fn simd() -> String {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let avx512 = std::arch::is_x86_feature_detected!("avx512f");
        let avx2 = std::arch::is_x86_feature_detected!("avx2");
        let v4 = cfg!(feature = "x86-v4");
        return format!(
            "x86_64: avx512f={avx512} avx2={avx2}, x86-v4 feature={v4} -> nano-gemm uses {}",
            if avx512 && v4 {
                "AVX-512"
            } else if avx2 {
                "AVX2"
            } else {
                "scalar"
            }
        );
    }
    #[cfg(target_arch = "aarch64")]
    {
        return format!(
            "aarch64: neon={} -> nano-gemm uses {}",
            std::arch::is_aarch64_feature_detected!("neon"),
            if std::arch::is_aarch64_feature_detected!("neon") { "NEON" } else { "scalar" }
        );
    }
    #[allow(unreachable_code)]
    "unknown arch".to_string()
}

fn main() {
    plan_strategy();
    println!("{}", simd());
    println!(
        "\n`fixed` is what the Schur reduction uses for the shapes it has kernels for;\n\
         `nano` is what it uses for every other shape. `runtime` is what `nano` replaced."
    );
    for trans in [false, true] {
        bench::<f64>("f64", trans);
        bench::<f32>("f32", trans);
    }
}

/// Where does the plan come from? nano-gemm's `execute_unchecked` needs a `Plan`
/// for the shape, and the fallback is handed the widths at run time, so it has to
/// get one on every call. Three ways, and the answer is not the obvious one.
///
/// Everything here is behind `#[inline(never)]`, which is not a handicap but the
/// truth: the real call site is a `T::gemm_sub_nano` in a loop whose shapes vary,
/// so nothing about the plan is loop-invariant there. Let LLVM see through these
/// and it hoists the plan construction clean out of the timing loop, and the
/// per-call column silently re-measures the hoisted one.
mod plan_source {
    /// Build a plan every call. Four lookups into a const microkernel table plus a
    /// branch chain, into a struct that never leaves the stack.
    #[inline(never)]
    pub fn fresh(dst: &mut [f64], ca: &[f64], wa: usize, we: usize, zb: &[f64], wb: usize) {
        let plan = nano_gemm::Plan::<f64>::new_colmajor_lhs_and_dst_f64(wa, wb, we);
        super::nano(&plan, dst, ca, false, wa, we, zb, wb);
    }

    /// Keep the plans in a thread-local and find the one for this shape. A model has
    /// a handful of distinct tile shapes, so a linear scan finds it in a step or two
    /// -- and it is still a thread-local access, a RefCell borrow and a scan.
    #[inline(never)]
    pub fn cached(dst: &mut [f64], ca: &[f64], wa: usize, we: usize, zb: &[f64], wb: usize) {
        thread_local! {
            static PLANS: core::cell::RefCell<Vec<((usize, usize, usize), nano_gemm::Plan<f64>)>> =
                const { core::cell::RefCell::new(Vec::new()) };
        }
        PLANS.with(|plans| {
            let mut plans = plans.borrow_mut();
            let key = (wa, we, wb);
            let at = match plans.iter().position(|(k, _)| *k == key) {
                Some(at) => at,
                None => {
                    let plan = nano_gemm::Plan::<f64>::new_colmajor_lhs_and_dst_f64(wa, wb, we);
                    plans.push((key, plan));
                    plans.len() - 1
                }
            };
            super::nano(&plans[at].1, dst, ca, false, wa, we, zb, wb);
        });
    }
}

fn plan_strategy() {
    use arael_faer::schur::SchurReal;
    const ROUNDS: usize = 12;
    const REPS: usize = 20_000;
    println!("\n=== where the plan comes from (f64, lhs direct) ===");
    println!(
        "{:<12} {:>11} {:>10} {:>9} {:>9} {:>11}",
        "(wa,we,wb)", "plan hoisted", "runtime", "fresh", "cached", "the trait"
    );
    for &(wa, we, wb, _) in SHAPES {
        let ca = vec![1.0f64; wa * we];
        let zb = vec![0.5f64; we * wb];
        let mut d = vec![0.0f64; wa * wb];
        let plan = <f64 as Real>::plan(wa, wb, we, false);

        // The floor: the plan is already in hand. Unreachable through nano-gemm's
        // API here -- its plan fields are private, so there is nothing to hoist the
        // build into -- but it bounds what any scheme could win back.
        let hoisted = best_ns(ROUNDS, REPS, || {
            nano(&plan, &mut d, &ca, false, wa, we, &zb, wb);
            std::hint::black_box(&d);
        });
        // What the fallback used to be.
        let loops = best_ns(ROUNDS, REPS, || {
            runtime(&mut d, &ca, false, wa, we, &zb, wb);
            std::hint::black_box(&d);
        });
        let fresh = best_ns(ROUNDS, REPS, || {
            plan_source::fresh(&mut d, &ca, wa, we, &zb, wb);
            std::hint::black_box(&d);
        });
        let cached = best_ns(ROUNDS, REPS, || {
            plan_source::cached(&mut d, &ca, wa, we, &zb, wb);
            std::hint::black_box(&d);
        });
        // What the library actually ships: whichever of the two won, plus the three
        // length assertions that make its `unsafe` sound. The gap to that column is
        // what the assertions cost.
        let trait_ = best_ns(ROUNDS, REPS, || {
            <f64 as SchurReal>::gemm_sub_nano(&mut d, &ca, false, wa, we, &zb, wb);
            std::hint::black_box(&d);
        });
        println!(
            "{:<12} {:>10.1}n {:>9.1}n {:>8.1}n {:>8.1}n {:>10.1}n",
            format!("({wa},{we},{wb})"),
            hoisted,
            loops,
            fresh,
            cached,
            trait_
        );
    }
}
