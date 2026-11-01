//! [`GpuSupernodal`]: the block supernodal Cholesky and its solve on the
//! device, in arael's own panel layout.

use arael_faer::bsc::SymbolicSparseBlockColMat;
use arael_faer::schur::SchurReal;
use arael_faer::supernodal::{SupernodalError, SupernodalSymbolic};
use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::kernels::*;
use crate::plan::{GpuSupernodalParams, LevelWork, NB, Plan, SparseIndex, TB, build_levels};
use crate::context::{GpuContext, Lease};
use crate::host::Upload;
use crate::{Real, sync};

/// One level's work lists on the device: views into the structure's
/// integer buffer, with their lengths.
struct DevLevel {
    tile_items: Handle,
    tile_rects: Handle,
    n_tile_items: u32,
    big_items: Handle,
    big_rects: Handle,
    n_big_items: u32,
    small: Handle,
    n_small: u32,
    big: Handle,
    n_big: u32,
    all: Handle,
    n_all: u32,
    g_items: Handle,
    n_g: u32,
    bg_items: Handle,
    n_bg: u32,
}

/// The device factorization of one sparsity structure: the plan built from
/// a [`SupernodalSymbolic`] once, the factor kept on the device, and the
/// kernels that fill it from a matrix's values and solve against it.
///
/// `F` is the factor's scalar, `f32` or `f64`; the matrix handed to
/// [`factorize`](Self::factorize) and the right-hand side of
/// [`solve`](Self::solve) may be in either and are converted on the way.
/// Every device buffer is leased from the [`GpuContext`] and goes back to it
/// on drop; the structure's integer data (the plan, the seed map, every
/// level's work lists) is one buffer, uploaded in one copy.
///
/// ```ignore
/// let mut g = GpuSupernodal::<f32>::new(&ctx, &sn, a.symbolic(), &Default::default());
/// g.factorize(a.vals())?;          // Err when a pivot is not positive
/// g.solve(&mut rhs);               // rhs becomes x, in the matrix's ordering
/// ```
pub struct GpuSupernodal<F: Real> {
    ctx: GpuContext,
    plan: Plan,
    lw: Vec<LevelWork>,
    /// The structure's integer data (the plan words first, then the rest),
    /// held for the views into it.
    _ints: Lease,
    d_plan: Handle,
    plan_len: usize,
    d_tiles: Handle,
    /// A word holding 1: the gate of a solve that always runs.
    d_on: Handle,
    levels: Vec<DevLevel>,
    /// The matrix's tile values as last uploaded.
    vals: Upload<F>,
    /// The right-hand side of a [`solve`](Self::solve) from the host.
    rhs: Upload<f64>,
    factor: Lease,
    fail: Lease,
    /// The solve's working vector, in the factor's scalar and ordering.
    xb: Lease,
    qq: usize,
    max_q: usize,
    lu: usize,
    id: u64,
    _f: core::marker::PhantomData<F>,
}

/// The next factor id.
static NEXT_FACTOR: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(1);

/// Arrays packed into one buffer, each starting on a 256-byte boundary.
#[derive(Default)]
pub(crate) struct Packer {
    pub words: Vec<u32>,
}

impl Packer {
    /// Append `v`, returning where it starts, in words.
    pub fn push(&mut self, v: &[u32]) -> usize {
        let at = self.words.len().div_ceil(64) * 64;
        self.words.resize(at, 0);
        self.words.extend_from_slice(v);
        at
    }

    /// A view of the array at word `at` of the buffer `h`.
    pub fn view(h: &Handle, at: usize) -> Handle {
        h.clone().offset_start((at * 4) as u64)
    }

    /// Lease a buffer for the words and upload them in one copy, handing
    /// CubeCL the words to own until the copy is done (no sync).
    pub fn upload(mut self, ctx: &GpuContext) -> Lease {
        self.words.push(0);
        let lease = ctx.lease(self.words.len() * 4);
        ctx.client().write(lease.handle(), cubecl::bytes::Bytes::from_elems(self.words));
        lease
    }
}

impl<F: Real> GpuSupernodal<F> {
    /// Build the device plan of `sn`, the analysis of a matrix with the block
    /// structure `asym`, and lease the factor's buffers from `ctx`.
    pub fn new(ctx: &GpuContext, sn: &SupernodalSymbolic, asym: &SymbolicSparseBlockColMat<SparseIndex>, params: &GpuSupernodalParams) -> Self {
        let plan = Plan::new(sn, asym);
        let lw = build_levels(&plan, params);
        let mut pk = Packer::default();
        let plan_words = plan.pack();
        let plan_len = plan_words.len();
        let at_plan = pk.push(&plan_words);
        debug_assert_eq!(at_plan, 0, "the plan's header offsets are relative to the buffer start");
        let at_tiles = pk.push(&plan.tiles);
        let at_on = pk.push(&[1]);
        let mut max_small_q = 0u32;
        let mut max_q = 0u32;
        for s in 0..plan.ns {
            max_q = max_q.max(plan.dims[2 * s]);
        }
        // each level's lists, as offsets first, views once the buffer exists
        let mut at_levels = Vec::with_capacity(lw.len());
        for w in &lw {
            for &t in &w.small_targets {
                max_small_q = max_small_q.max(plan.dims[2 * t as usize]);
            }
            at_levels.push([
                pk.push(&w.tile_items),
                pk.push(&w.tile_rects),
                pk.push(&w.big_items),
                pk.push(&w.big_rects),
                pk.push(&w.small_targets),
                pk.push(&w.big_targets),
                pk.push(&w.level_sn),
                pk.push(&w.g_items),
                pk.push(&w.bg_items),
            ]);
        }
        let ints = pk.upload(ctx);
        let h = ints.handle();
        let levels = lw
            .iter()
            .zip(&at_levels)
            .map(|(w, at)| DevLevel {
                tile_items: Packer::view(h, at[0]),
                tile_rects: Packer::view(h, at[1]),
                n_tile_items: (w.tile_items.len() / 5) as u32,
                big_items: Packer::view(h, at[2]),
                big_rects: Packer::view(h, at[3]),
                n_big_items: (w.big_items.len() / 5) as u32,
                small: Packer::view(h, at[4]),
                n_small: w.small_targets.len() as u32,
                big: Packer::view(h, at[5]),
                n_big: w.big_targets.len() as u32,
                all: Packer::view(h, at[6]),
                n_all: w.level_sn.len() as u32,
                g_items: Packer::view(h, at[7]),
                n_g: (w.g_items.len() / 4) as u32,
                bg_items: Packer::view(h, at[8]),
                n_bg: (w.bg_items.len() / 3) as u32,
            })
            .collect();
        let (d_plan, d_tiles, d_on) = (h.clone(), Packer::view(h, at_tiles), Packer::view(h, at_on));
        let fs = core::mem::size_of::<F>();
        let factor = ctx.lease(plan.factor_vals.max(1) * fs);
        let fail = ctx.lease(4);
        let xb = ctx.lease(plan.n.max(1) * fs);
        let lu = params.lane_unroll.unwrap_or(if fs == 4 { 4 } else { 8 });
        // the small kernel's shared tile: q rounded up to a multiple of 32
        let qq = ((max_small_q.max(1) as usize).div_ceil(32) * 32).pow(2);
        assert!(
            qq * fs <= 96 * 1024,
            "small_q {} needs a {} KB shared tile in the small factor kernel; the device gives 96",
            params.small_q,
            qq * fs / 1024
        );
        assert!(lu >= 2 && lu.is_multiple_of(2), "the lane unroll is even and at least 2");
        GpuSupernodal {
            ctx: ctx.clone(),
            plan,
            lw,
            _ints: ints,
            d_plan,
            plan_len,
            d_tiles,
            d_on,
            levels,
            vals: Upload::default(),
            rhs: Upload::default(),
            factor,
            fail,
            xb,
            qq,
            // rounded, so the kernels specialised on it are compiled for a
            // few sizes rather than per problem
            max_q: (max_q.max(1) as usize).div_ceil(256) * 256,
            lu,
            id: NEXT_FACTOR.fetch_add(1, core::sync::atomic::Ordering::Relaxed),
            _f: core::marker::PhantomData,
        }
    }

    /// The scalar dimension of the system.
    pub fn dim(&self) -> usize {
        self.plan.n
    }

    /// The context the factor runs on.
    pub fn context(&self) -> &GpuContext {
        &self.ctx
    }

    /// The factor's identity, unique in the process. A graph that captured
    /// this factor's kernels replays them only for the same id (a new factor
    /// may lease the same buffers with another plan in them).
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Launches per factorization: one per level for the small updates, the
    /// big updates and the small factors each, three per panel step, plus
    /// the zeroing and the seed.
    pub fn factorization_launches(&self) -> usize {
        self.lw.iter().map(|w| 3 + 3 * w.trsm_grid.len()).sum::<usize>() + 3
    }

    /// Factorize the matrix whose stored tile values are `vals`, in the
    /// block storage order of the structure the plan was built from. `Err`
    /// when a diagonal pivot is not positive; the factor is then unusable
    /// and the caller raises the damping, as with the CPU factorization.
    pub fn factorize<M: SchurReal + 'static>(&mut self, vals: &[M]) -> Result<(), SupernodalError> {
        let ctx = self.ctx.clone();
        let c = ctx.client();
        unsafe { k_zero_u32::launch_unchecked(c, CubeCount::Static(1, 1, 1), CubeDim::new_1d(32), BufferArg::from_raw_parts(self.fail.handle().clone(), 1), 1u32) };
        let flight = crate::host::InFlight::new(c);
        // The values in the factor's scalar, read in place when the scalars
        // agree, converted on the host otherwise.
        // SAFETY: `vals` stays borrowed until the read of the pivot flag below.
        unsafe { self.vals.write(&ctx, vals, |v| F::from(v.to_f64()).unwrap()) };
        let d_vals = self.vals.handle().clone();
        let fail = self.fail.handle().clone();
        self.factorize_with(&d_vals, vals.len(), false, &fail);
        let failed = bytemuck::cast_slice::<u8, u32>(&crate::read_one(c, self.fail.view(4)).expect("read the pivot flag"))[0];
        flight.done();
        if failed != 0 { Err(SupernodalError::NotPositiveDefinite) } else { Ok(()) }
    }

    /// [`factorize`](Self::factorize) from tile values already on the
    /// device, in `f64` (a reduced system the device formed), without waiting
    /// for it: the values are cast to the factor's scalar as they land in the
    /// panels, and a pivot that is not positive sets `fail[0]` nonzero (the
    /// caller zeroes it, and reads it with its result).
    pub fn factorize_device(&mut self, vals: &Handle, nvals: usize, fail: &Handle) {
        self.factorize_with(vals, nvals, true, fail)
    }

    fn factorize_with(&mut self, d_vals: &Handle, nvals: usize, from_f64: bool, fail: &Handle) {
        let c = self.ctx.client();
        let arr = |h: &Handle, len: usize| unsafe { BufferArg::from_raw_parts(h.clone(), len.max(1)) };
        let d256 = CubeDim::new_1d(256);
        let count = |n: u32| CubeCount::Static(n, 1, 1);
        let flen = self.plan.factor_vals;
        let plen = self.plan_len;
        let ntiles = (self.plan.tiles.len() / 6) as u32;
        unsafe {
            k_zero::launch_unchecked::<F>(c, count((flen as u32).div_ceil(256)), d256, arr(self.factor.handle(), flen), flen as u32);
            if from_f64 {
                k_seed_from::launch_unchecked::<f64, F>(c, count(ntiles.div_ceil(256).max(1)), d256, arr(&self.d_tiles, self.plan.tiles.len()), arr(d_vals, nvals), arr(self.factor.handle(), flen), ntiles);
            } else {
                k_seed::launch_unchecked::<F>(c, count(ntiles.div_ceil(256).max(1)), d256, arr(&self.d_tiles, self.plan.tiles.len()), arr(d_vals, nvals), arr(self.factor.handle(), flen), ntiles);
            }
            for (l, d) in self.levels.iter().enumerate() {
                let w = &self.lw[l];
                if d.n_tile_items > 0 {
                    k_tile_update::launch_unchecked::<F>(
                        c,
                        count(d.n_tile_items),
                        d256,
                        arr(&d.tile_items, w.tile_items.len()),
                        arr(&d.tile_rects, w.tile_rects.len()),
                        arr(&self.d_plan, plen),
                        arr(self.factor.handle(), flen),
                        arr(self.factor.handle(), flen),
                    );
                }
                if d.n_big_items > 0 {
                    k_big_update::launch_unchecked::<F>(
                        c,
                        count(d.n_big_items),
                        d256,
                        arr(&d.big_items, w.big_items.len()),
                        arr(&d.big_rects, w.big_rects.len()),
                        arr(&self.d_plan, plen),
                        arr(self.factor.handle(), flen),
                    );
                }
                if d.n_small > 0 {
                    k_small_factor::launch_unchecked::<F>(
                        c,
                        count(d.n_small),
                        d256,
                        arr(&d.small, w.small_targets.len()),
                        arr(&self.d_plan, plen),
                        arr(self.factor.handle(), flen),
                        arr(fail, 1),
                        self.qq,
                    );
                }
                for (st, (&tg, &sg)) in w.trsm_grid.iter().zip(&w.syrk_grid).enumerate() {
                    let j = (st * NB) as u32;
                    k_potf2::launch_unchecked::<F>(c, count(d.n_big), d256, arr(&d.big, w.big_targets.len()), j, arr(&self.d_plan, plen), arr(self.factor.handle(), flen), arr(fail, 1));
                    if tg > 0 {
                        k_trsm::launch_unchecked::<F>(c, count(tg), CubeDim::new_1d(TB as u32), arr(&d.big, w.big_targets.len()), d.n_big, j, arr(&self.d_plan, plen), arr(self.factor.handle(), flen));
                    }
                    if sg > 0 {
                        k_syrk::launch_unchecked::<F>(c, count(sg), d256, arr(&d.big, w.big_targets.len()), d.n_big, j, arr(&self.d_plan, plen), arr(self.factor.handle(), flen));
                    }
                }
            }
        }
    }

    /// Solve `A x = rhs` in place through the last factorization; `rhs` is in
    /// the matrix's own ordering and comes back as `x`.
    pub fn solve<M: SchurReal + 'static>(&mut self, rhs: &mut [M]) {
        let n = self.plan.n;
        assert_eq!(rhs.len(), n);
        let ctx = self.ctx.clone();
        let c = ctx.client();
        let flight = crate::host::InFlight::new(c);
        // SAFETY: `rhs` stays borrowed until the read of the solution below.
        unsafe { self.rhs.write(&ctx, rhs, |v| v.to_f64()) };
        let hx = self.rhs.handle().clone();
        self.solve_device(&hx);
        let bytes = crate::read_one(c, self.rhs.lease().view(n * 8)).expect("read the solution");
        flight.done();
        for (dst, &v) in rhs.iter_mut().zip(f64::from_bytes(&bytes)) {
            *dst = M::from_f64(v);
        }
    }

    /// [`solve`](Self::solve) on a vector already on the device, `f64` in
    /// the matrix's ordering, in place: the permutation into the factor's
    /// ordering and scalar and back, and the level-set solve between.
    pub fn solve_device(&mut self, x: &Handle) {
        let on = self.d_on.clone();
        self.solve_device_if(x, &on);
    }

    /// [`solve_device`](Self::solve_device) when `gate[0]` is nonzero, as
    /// the device finds it; otherwise its kernels do nothing.
    pub fn solve_device_if(&mut self, x: &Handle, gate: &Handle) {
        let p = &self.plan;
        let c = self.ctx.client();
        let arr = |h: &Handle, len: usize| unsafe { BufferArg::from_raw_parts(h.clone(), len.max(1)) };
        let d256 = CubeDim::new_1d(256);
        let count = |n: u32| CubeCount::Static(n, 1, 1);
        let flen = p.factor_vals;
        let plen = self.plan_len;
        let hx = self.xb.handle();
        let nb = p.nb as u32;
        unsafe {
            k_perm_in::launch_unchecked::<F>(c, count(nb.div_ceil(256).max(1)), d256, arr(gate, 1), arr(&self.d_plan, plen), arr(x, p.n), arr(hx, p.n), nb);
            for (l, d) in self.levels.iter().enumerate() {
                let w = &self.lw[l];
                if d.n_g > 0 {
                    k_fwd_gather::launch_unchecked::<F>(c, count(d.n_g), d256, arr(gate, 1), arr(&d.g_items, w.g_items.len()), arr(&self.d_plan, plen), arr(self.factor.handle(), flen), arr(hx, p.n), arr(hx, p.n));
                }
                k_fwd_trsv::launch_unchecked::<F>(c, count(d.n_all), d256, arr(gate, 1), arr(&d.all, w.level_sn.len()), arr(&self.d_plan, plen), arr(self.factor.handle(), flen), arr(hx, p.n), self.max_q);
            }
            for (l, d) in self.levels.iter().enumerate().rev() {
                let w = &self.lw[l];
                if d.n_bg > 0 {
                    k_bwd_gemv::launch_unchecked::<F>(c, count(d.n_bg), d256, arr(gate, 1), arr(&d.bg_items, w.bg_items.len()), arr(&self.d_plan, plen), arr(self.factor.handle(), flen), arr(hx, p.n), self.lu);
                }
                k_bwd_trsv::launch_unchecked::<F>(c, count(d.n_all), d256, arr(gate, 1), arr(&d.all, w.level_sn.len()), arr(&self.d_plan, plen), arr(self.factor.handle(), flen), arr(hx, p.n), self.max_q, self.lu);
            }
            k_perm_out::launch_unchecked::<F>(c, count(nb.div_ceil(256).max(1)), d256, arr(gate, 1), arr(&self.d_plan, plen), arr(hx, p.n), arr(x, p.n), nb);
        }
    }

    /// The factor's values, panel by panel in the symbolic's layout, read
    /// back from the device (for tests and inspection).
    pub fn factor_values(&self) -> Vec<F> {
        sync(self.ctx.client());
        let bytes = crate::read_one(self.ctx.client(), self.factor.view(self.plan.factor_vals * core::mem::size_of::<F>())).expect("read the factor");
        F::from_bytes(&bytes)[..self.plan.factor_vals].to_vec()
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::test_util::{grid_spd, rel_residual};
    use crate::{GpuBackend, GpuContext, GpuFactor, GpuPrecision};
    use arael_faer::supernodal::{SupernodalContext, SupernodalParams, nd_block_order, supernodal_factorize, supernodal_solve};

    fn setup(k: usize) -> (arael_faer::bsc::SparseBlockColMat<SparseIndex, f64>, Vec<f64>, Vec<f64>, SupernodalSymbolic) {
        let (a, dense, rhs) = grid_spd(k, &[3, 6], 11);
        let order = nd_block_order(a.symbolic());
        let sn = SupernodalSymbolic::new(a.symbolic(), Some(&order), &SupernodalParams::default()).unwrap();
        (a, dense, rhs, sn)
    }

    /// The factor's lower parts against the CPU's, as err / max.
    fn factor_err(sn: &SupernodalSymbolic, gpu: &[f64], cpu: &[f64]) -> f64 {
        let (mut mx, mut df) = (0.0f64, 0.0f64);
        for s in 0..sn.n_supernodes() {
            let (q, h) = sn.supernode_dims(s);
            let base = sn.panel_range(s).start;
            for c in 0..q {
                for r in c..h {
                    mx = mx.max(cpu[base + c * h + r].abs());
                    df = df.max((gpu[base + c * h + r] - cpu[base + c * h + r]).abs());
                }
            }
        }
        df / mx
    }

    #[test]
    fn f64_factor_and_solve_match_the_cpu() {
        let (a, dense, rhs, sn) = setup(24);
        let n = sn.dim();
        let mut cpu = vec![0.0f64; sn.factor_val_count()];
        let mut ctx = SupernodalContext::new();
        supernodal_factorize(&sn, &a, &mut cpu, &mut ctx, 1).unwrap();
        let mut x_cpu = rhs.clone();
        supernodal_solve(&sn, &cpu, &mut x_cpu, &mut ctx, 1);
        for params in [GpuSupernodalParams::default(), GpuSupernodalParams { small_q: 8, big_pair_flops: 100.0, elem_cap: 256, solve_work: 512, lane_unroll: None }] {
            let mut g = GpuSupernodal::<f64>::new(&GpuContext::new(GpuBackend::Cuda, 0).unwrap(), &sn, a.symbolic(), &params);
            assert!(sn.n_supernodes() > 20, "a tree with levels");
            g.factorize(a.vals()).unwrap();
            let err = factor_err(&sn, &g.factor_values(), &cpu);
            assert!(err < 1e-12, "factor err/max {err}");
            let mut x = rhs.clone();
            g.solve(&mut x);
            let res = rel_residual(&dense, n, &x, &rhs);
            assert!(res < 1e-10, "residual {res}");
            let dx = x.iter().zip(&x_cpu).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            assert!(dx < 1e-9, "solution differs from the CPU's by {dx}");
            // a second factorization of the same structure reuses the plan
            g.factorize(a.vals()).unwrap();
            let mut x2 = rhs.clone();
            g.solve(&mut x2);
            assert!(rel_residual(&dense, n, &x2, &rhs) < 1e-10);
        }
    }

    #[test]
    fn f32_factor_solves_to_single_precision_and_refines() {
        let (a, dense, rhs, sn) = setup(24);
        let n = sn.dim();
        let mut g = GpuFactor::new(&GpuContext::new(GpuBackend::Cuda, 0).unwrap(), GpuPrecision::F32, &sn, a.symbolic(), &Default::default());
        assert_eq!(g.precision(), GpuPrecision::F32);
        g.factorize(a.vals()).unwrap();
        let mut x = rhs.clone();
        g.solve(&mut x);
        let res0 = rel_residual(&dense, n, &x, &rhs);
        assert!(res0 < 1e-4 && res0 > 1e-9, "an f32 solve: residual {res0}");
        // one refinement step against the f64 matrix
        let mut r = vec![0.0f64; n];
        for i in 0..n {
            let mut ax = 0.0;
            for c in 0..n {
                ax += dense[i + c * n] * x[c];
            }
            r[i] = rhs[i] - ax;
        }
        g.solve(&mut r);
        for i in 0..n {
            x[i] += r[i];
        }
        let res1 = rel_residual(&dense, n, &x, &rhs);
        assert!(res1 < res0 * 1e-3, "refinement {res0} -> {res1}");
    }

    #[test]
    fn a_pivot_that_is_not_positive_is_reported() {
        let (mut a, _, _, sn) = setup(10);
        let asym = a.symbolic().clone();
        // the last block's diagonal made negative: the root supernode fails
        let j = asym.nblk_cols() - 1;
        let b = asym.col_range(j).find(|&b| asym.blk_row(b) == j).unwrap();
        let base = asym.val_range(b).start;
        a.vals_mut()[base] = -1.0;
        let mut g = GpuSupernodal::<f64>::new(&GpuContext::new(GpuBackend::Cuda, 0).unwrap(), &sn, &asym, &Default::default());
        assert_eq!(g.factorize(a.vals()), Err(SupernodalError::NotPositiveDefinite));
    }
}
