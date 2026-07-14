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
//! the `x86-v4` feature:
//!
//! ```text
//! RUSTFLAGS="-C target-cpu=native" \
//!     cargo run --release -p arael-faer --features x86-v4 --example gemmbench
//! ```
//!
//! On the one x86 machine measured so far that feature made things WORSE -- see
//! the note on it in Cargo.toml -- so run both and compare rather than assuming
//! the wider kernel is the faster one. The last line of output says which SIMD
//! path was actually taken, which is the only way to tell the two runs apart.
//!
//! The interesting column is `nano vs fixed`: above 1.0 the unrolled kernel
//! wins and the fallback should stay a fallback; below 1.0 on the hot shapes
//! ((6,3,6) for SLAM, (9,3,9) for bundle adjustment) nano-gemm would be worth
//! taking over the whole path on that machine. Check BOTH orientations before
//! concluding anything: the reduction uses each, and nano-gemm is much weaker on
//! a transposed lhs, so a shape it wins direct can still lose overall.

use std::time::Instant;

/// (wa, we, wb, what it is). The first block is what `FIXED_SHAPES` covers; the
/// second has no unrolled kernel and goes to the fallback.
/// Every shape in `FIXED_SHAPES`, plus three with no unrolled kernel. All eleven
/// listed shapes are here: the transpose-first rule routes real shipped shapes, so
/// none of them may go unmeasured.
const SHAPES: &[(usize, usize, usize, &str)] = &[
    (3, 3, 3, "pgo 2D"),
    (6, 3, 6, "slam pose x point"),
    (6, 6, 6, "pgo 3D pose x pose"),
    (9, 3, 9, "bal camera x point"),
    (6, 1, 6, "slam, 1-wide elim"),
    (6, 2, 6, "slam, bearing"),
    (6, 4, 6, "slam, line/plane"),
    (7, 3, 7, "sim3 x point"),
    (3, 2, 3, "2D pose x 2D point"),
    (3, 4, 3, "2D pose x segment"),
    (2, 3, 2, "mirror: 2-wide obs"),
    (12, 3, 12, "no fixed kernel"),
    (9, 6, 9, "no fixed kernel"),
    (5, 3, 7, "no fixed kernel"),
];

/// Mirrors `schur::TRANS_PACK_MAX`. A transposed tile at or under this many
/// elements is transposed into a stack buffer and handed to nano-gemm column-major;
/// over it, it is passed with a row stride instead.
const TRANS_PACK_MAX: usize = 12 * 12;

/// Will this call hand nano-gemm a lhs whose row stride is not 1? That is the only
/// thing that decides which plan -- and which millikernel -- it gets.
fn strided_lhs(trans: bool, wa: usize, we: usize) -> bool {
    trans && wa * we > TRANS_PACK_MAX
}

/// The scalar: arael solves in both, and the SIMD width -- so the verdict --
/// differs between them.
trait Real: Copy + std::ops::Add<Output = Self> + std::ops::Sub<Output = Self> + std::ops::Mul<Output = Self> {
    const ZERO: Self;
    const ONE: Self;
    const NEG_ONE: Self;
    fn of(x: f64) -> Self;
    fn abs_diff(self, other: Self) -> f64;
    /// dst is column-major always. So is the lhs, unless it is an oversized
    /// transposed tile -- only then is the general-stride plan needed.
    fn plan(m: usize, n: usize, k: usize, strided: bool) -> nano_gemm::Plan<Self>
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
    fn plan(m: usize, n: usize, k: usize, strided: bool) -> nano_gemm::Plan<Self> {
        if strided {
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
    fn plan(m: usize, n: usize, k: usize, strided: bool) -> nano_gemm::Plan<Self> {
        if strided {
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

/// The transposed unrolled kernel, but transposing FIRST. `fixed`'s transposed body
/// computes a horizontal dot product per (i, c); its direct body is an axpy down a
/// contiguous column, which is the shape the machine wants. So pay a WA x WE
/// transpose into a stack buffer -- 18 elements for (6,3,6) -- and then run the
/// direct body on it.
///
/// `[[T; WA]; WE]` needs no `generic_const_exprs`: it is nested arrays, not an array
/// of length `WA * WE`. Each `a[k]` is then exactly the contiguous column the direct
/// body wants.
///
/// The zero-init is free and needs no `MaybeUninit`: the array is exactly WA x WE and
/// every element is written before it is read, so the stores are dead and LLVM drops
/// them -- measured identical to a `MaybeUninit` version. The nano-gemm fallback
/// cannot do this, because there the widths are run-time values, the buffer has to be
/// worst-case, and the part that is never written still gets zeroed.
#[inline]
fn fixed_packed<T: Real, const WA: usize, const WE: usize, const WB: usize>(
    dst: &mut [T],
    ca: &[T],
    zb: &[T],
) {
    // ca holds C_a^T (WE x WA): C_a[i, k] is at ca[k + i * WE].
    let mut a = [[T::ZERO; WA]; WE];
    for i in 0..WA {
        for k in 0..WE {
            a[k][i] = ca[k + i * WE];
        }
    }
    for c in 0..WB {
        for k in 0..WE {
            let z = zb[k + c * WE];
            let dcol = &mut dst[c * WA..(c + 1) * WA];
            let acol = &a[k];
            for i in 0..WA {
                dcol[i] = dcol[i] - acol[i] * z;
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
///
/// Copied from `SchurReal::gemm_sub_nano`, packing and all. A transposed lhs COULD
/// be expressed as a stride swap -- it is the same matrix read differently, no copy
/// needed -- but nano-gemm answers any lhs with a row stride other than 1 by packing
/// it into two 64 KB stack buffers, a cost that is flat and therefore ruinous on a
/// small tile. So the transpose is done here, into a buffer the size of the tile,
/// and nano-gemm gets a column-major lhs it can read in place.
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
    // MaybeUninit, not [T::ZERO; TRANS_PACK_MAX]: zeroing the whole buffer is 1152
    // bytes of memset per call, which measured as a flat ~64 ns -- worse than the
    // packing it exists to avoid. The library makes the same choice for the same
    // reason, and this has to match it or the benchmark is measuring fiction.
    let mut packed = [const { std::mem::MaybeUninit::<T>::uninit() }; TRANS_PACK_MAX];
    let pack = trans && wa * we <= TRANS_PACK_MAX;
    if pack {
        for i in 0..wa {
            for k in 0..we {
                packed[i + k * wa].write(ca[k + i * we]);
            }
        }
    }
    let (lhs, lhs_rs, lhs_cs): (*const T, isize, isize) = if !trans {
        (ca.as_ptr(), 1, wa as isize)
    } else if pack {
        (packed.as_ptr().cast::<T>(), 1, wa as isize)
    } else {
        (ca.as_ptr(), we as isize, 1)
    };
    unsafe {
        plan.execute_unchecked(
            wa, wb, we,
            dst.as_mut_ptr(), 1, wa as isize,
            lhs, lhs_rs, lhs_cs,
            zb.as_ptr(), 1, we as isize,
            T::ONE, T::NEG_ONE,
            false, false,
        );
    }
}

/// Best-of-`rounds` for each contender, each round `reps` GEMMs, with the rounds
/// INTERLEAVED: one round of each contender, then the next round of each. Returns ns
/// per GEMM, in the order given.
///
/// Running them back-to-back instead lets a frequency or thermal drift land entirely
/// on whichever contender happened to be executing. That is not hypothetical here:
/// back-to-back, (6,3,6) transposed came out 12.8 ns on one run and 16.5 ns on the
/// next, which is the difference between a 1.3x win and no win at all. Interleaved,
/// a drift hits every contender equally and the comparison survives it.
fn best_ns_each(rounds: usize, reps: usize, fs: &mut [&mut dyn FnMut()]) -> Vec<f64> {
    let mut best = vec![f64::INFINITY; fs.len()];
    for _ in 0..rounds {
        for (slot, f) in core::iter::zip(best.iter_mut(), fs.iter_mut()) {
            let t = Instant::now();
            for _ in 0..reps {
                f();
            }
            *slot = slot.min(t.elapsed().as_secs_f64() * 1e9 / reps as f64);
        }
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
    // `fixed+T` only means anything transposed: it IS the direct kernel there.
    if trans {
        println!(
            "{:<12} {:>20} {:>9} {:>9} {:>9} {:>9}  {:>13}",
            "(wa,we,wb)", "what it is", "fixed", "fixed+T", "runtime", "nano", "nano vs fixed"
        );
    } else {
        println!(
            "{:<12} {:>20} {:>9} {:>9} {:>9}  {:>13}",
            "(wa,we,wb)", "what it is", "fixed", "runtime", "nano", "nano vs fixed"
        );
    }

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
        let plan = T::plan(wa, wb, we, strided_lhs(trans, wa, we));

        macro_rules! on_shape {
            ($f:ident, $dst:expr $(, $extra:expr)*) => {
                match (wa, we, wb) {
                    (3, 3, 3) => $f::<T, 3, 3, 3>($dst, &ca, &zb $(, $extra)*),
                    (6, 3, 6) => $f::<T, 6, 3, 6>($dst, &ca, &zb $(, $extra)*),
                    (6, 6, 6) => $f::<T, 6, 6, 6>($dst, &ca, &zb $(, $extra)*),
                    (9, 3, 9) => $f::<T, 9, 3, 9>($dst, &ca, &zb $(, $extra)*),
                    (6, 1, 6) => $f::<T, 6, 1, 6>($dst, &ca, &zb $(, $extra)*),
                    (6, 2, 6) => $f::<T, 6, 2, 6>($dst, &ca, &zb $(, $extra)*),
                    (6, 4, 6) => $f::<T, 6, 4, 6>($dst, &ca, &zb $(, $extra)*),
                    (7, 3, 7) => $f::<T, 7, 3, 7>($dst, &ca, &zb $(, $extra)*),
                    (3, 2, 3) => $f::<T, 3, 2, 3>($dst, &ca, &zb $(, $extra)*),
                    (3, 4, 3) => $f::<T, 3, 4, 3>($dst, &ca, &zb $(, $extra)*),
                    (2, 3, 2) => $f::<T, 2, 3, 2>($dst, &ca, &zb $(, $extra)*),
                    (12, 3, 12) => $f::<T, 12, 3, 12>($dst, &ca, &zb $(, $extra)*),
                    (9, 6, 9) => $f::<T, 9, 6, 9>($dst, &ca, &zb $(, $extra)*),
                    (5, 3, 7) => $f::<T, 5, 3, 7>($dst, &ca, &zb $(, $extra)*),
                    _ => unreachable!(),
                }
            };
        }
        macro_rules! call_fixed {
            ($dst:expr) => {
                on_shape!(fixed, $dst, trans)
            };
        }
        macro_rules! call_fixed_packed {
            ($dst:expr) => {
                on_shape!(fixed_packed, $dst)
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
        let mut d4 = vec![T::ZERO; wa * wb];
        if trans {
            call_fixed_packed!(&mut d4);
            assert!(
                d1.iter().zip(&d4).all(|(a, b)| a.abs_diff(*b) < tol),
                "({wa},{we},{wb}) fixed+T disagrees: {d1:?} vs {d4:?}"
            );
        }

        // Interleaved, not back-to-back: see `best_ns_each`. `fixed_packed` reads ca
        // as C_a^T, so it is only in the race when the tile really is transposed.
        let mut f_fixed = || {
            call_fixed!(&mut d1);
            std::hint::black_box(&d1);
        };
        let mut f_rt = || {
            runtime(&mut d2, &ca, trans, wa, we, &zb, wb);
            std::hint::black_box(&d2);
        };
        let mut f_nano = || {
            nano(&plan, &mut d3, &ca, trans, wa, we, &zb, wb);
            std::hint::black_box(&d3);
        };
        let mut f_packed = || {
            call_fixed_packed!(&mut d4);
            std::hint::black_box(&d4);
        };

        let mut contenders: Vec<&mut dyn FnMut()> = vec![&mut f_fixed, &mut f_rt, &mut f_nano];
        if trans {
            contenders.push(&mut f_packed);
        }
        let t = best_ns_each(ROUNDS, REPS, &mut contenders);
        let (t_fixed, t_rt, t_nano) = (t[0], t[1], t[2]);
        let t_packed = if trans { t[3] } else { t_fixed };

        // The best of ours is what nano-gemm actually has to beat.
        let best_ours = if trans { t_fixed.min(t_packed) } else { t_fixed };
        let ratio = t_nano / best_ours;
        if trans {
            println!(
                "{:<12} {:>20} {:>8.1}n {:>8.1}n {:>8.1}n {:>8.1}n  {:>12.2}x{}",
                format!("({wa},{we},{wb})"),
                what,
                t_fixed,
                t_packed,
                t_rt,
                t_nano,
                ratio,
                if ratio < 0.95 { "  <- nano wins" } else { "" }
            );
            continue;
        }
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
        let plan = <f64 as Real>::plan(wa, wb, we, false);
        // One dst each: interleaving needs the contenders to be independent closures.
        let (mut d1, mut d2, mut d3, mut d4, mut d5) = (
            vec![0.0f64; wa * wb],
            vec![0.0f64; wa * wb],
            vec![0.0f64; wa * wb],
            vec![0.0f64; wa * wb],
            vec![0.0f64; wa * wb],
        );

        // The floor: the plan is already in hand. Unreachable through nano-gemm's
        // API here -- its plan fields are private, so there is nothing to hoist the
        // build into -- but it bounds what any scheme could win back.
        let mut f_hoisted = || {
            nano(&plan, &mut d1, &ca, false, wa, we, &zb, wb);
            std::hint::black_box(&d1);
        };
        // What the fallback used to be.
        let mut f_loops = || {
            runtime(&mut d2, &ca, false, wa, we, &zb, wb);
            std::hint::black_box(&d2);
        };
        let mut f_fresh = || {
            plan_source::fresh(&mut d3, &ca, wa, we, &zb, wb);
            std::hint::black_box(&d3);
        };
        let mut f_cached = || {
            plan_source::cached(&mut d4, &ca, wa, we, &zb, wb);
            std::hint::black_box(&d4);
        };
        // What the library actually ships: whichever of the two won, plus the three
        // length assertions that make its `unsafe` sound. The gap to that column is
        // what the assertions cost.
        let mut f_trait = || {
            <f64 as SchurReal>::gemm_sub_nano(&mut d5, &ca, false, wa, we, &zb, wb);
            std::hint::black_box(&d5);
        };
        let t = best_ns_each(
            ROUNDS,
            REPS,
            &mut [&mut f_hoisted, &mut f_loops, &mut f_fresh, &mut f_cached, &mut f_trait],
        );
        println!(
            "{:<12} {:>10.1}n {:>9.1}n {:>8.1}n {:>8.1}n {:>10.1}n",
            format!("({wa},{we},{wb})"),
            t[0],
            t[1],
            t[2],
            t[3],
            t[4]
        );
    }
}
