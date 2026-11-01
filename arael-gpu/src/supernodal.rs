//! [`GpuSupernodal`]: the block supernodal Cholesky and its solve on the
//! device, in arael's own panel layout.

use arael_faer::bsc::SymbolicSparseBlockColMat;
use arael_faer::schur::SchurReal;
use arael_faer::supernodal::{SupernodalError, SupernodalSymbolic};
use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::kernels::*;
use crate::plan::{GpuSupernodalParams, LevelWork, NB, Plan, SparseIndex, TB, build_levels};
use crate::{Real, sync};

/// One level's work lists on the device, with their lengths.
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
///
/// ```ignore
/// let mut g = GpuSupernodal::<f32>::new(client, &sn, a.symbolic(), &Default::default());
/// g.factorize(a.vals())?;          // Err when a pivot is not positive
/// g.solve(&mut rhs);               // rhs becomes x, in the matrix's ordering
/// ```
pub struct GpuSupernodal<F: Real> {
    client: Client,
    plan: Plan,
    lw: Vec<LevelWork>,
    d_plan: Handle,
    plan_len: usize,
    d_tiles: Handle,
    levels: Vec<DevLevel>,
    factor: Handle,
    fail: Handle,
    qq: usize,
    max_q: usize,
    lu: usize,
    _f: core::marker::PhantomData<F>,
}

fn upload_u32(client: &Client, v: &[u32]) -> Handle {
    if v.is_empty() {
        return client.create_from_slice(bytemuck::cast_slice(&[0u32]));
    }
    client.create_from_slice(bytemuck::cast_slice(v))
}

impl<F: Real> GpuSupernodal<F> {
    /// Build the device plan of `sn`, the analysis of a matrix with the block
    /// structure `asym`, and allocate the factor on the device.
    pub fn new(client: Client, sn: &SupernodalSymbolic, asym: &SymbolicSparseBlockColMat<SparseIndex>, params: &GpuSupernodalParams) -> Self {
        let plan = Plan::new(sn, asym);
        let lw = build_levels(&plan, params);
        let plan_buf = plan.pack();
        let d_plan = upload_u32(&client, &plan_buf);
        let d_tiles = upload_u32(&client, &plan.tiles);
        let mut max_small_q = 0u32;
        let mut max_q = 0u32;
        for s in 0..plan.ns {
            max_q = max_q.max(plan.dims[2 * s]);
        }
        let mut levels = Vec::with_capacity(lw.len());
        for w in &lw {
            for &t in &w.small_targets {
                max_small_q = max_small_q.max(plan.dims[2 * t as usize]);
            }
            levels.push(DevLevel {
                tile_items: upload_u32(&client, &w.tile_items),
                tile_rects: upload_u32(&client, &w.tile_rects),
                n_tile_items: (w.tile_items.len() / 5) as u32,
                big_items: upload_u32(&client, &w.big_items),
                big_rects: upload_u32(&client, &w.big_rects),
                n_big_items: (w.big_items.len() / 5) as u32,
                small: upload_u32(&client, &w.small_targets),
                n_small: w.small_targets.len() as u32,
                big: upload_u32(&client, &w.big_targets),
                n_big: w.big_targets.len() as u32,
                all: upload_u32(&client, &w.level_sn),
                n_all: w.level_sn.len() as u32,
                g_items: upload_u32(&client, &w.g_items),
                n_g: (w.g_items.len() / 4) as u32,
                bg_items: upload_u32(&client, &w.bg_items),
                n_bg: (w.bg_items.len() / 3) as u32,
            });
        }
        let factor = client.empty(plan.factor_vals.max(1) * core::mem::size_of::<F>());
        let fail = client.empty(4);
        let lu = params.lane_unroll.unwrap_or(if core::mem::size_of::<F>() == 4 { 4 } else { 8 });
        // the small kernel's shared tile: q rounded up to a multiple of 32
        let qq = ((max_small_q.max(1) as usize).div_ceil(32) * 32).pow(2);
        assert!(
            qq * core::mem::size_of::<F>() <= 96 * 1024,
            "small_q {} needs a {} KB shared tile in the small factor kernel; the device gives 96",
            params.small_q,
            qq * core::mem::size_of::<F>() / 1024
        );
        assert!(lu >= 2 && lu.is_multiple_of(2), "the lane unroll is even and at least 2");
        GpuSupernodal {
            client,
            plan,
            lw,
            d_plan,
            plan_len: plan_buf.len(),
            d_tiles,
            levels,
            factor,
            fail,
            // rounded, so the kernels specialised on them are compiled for a
            // few sizes rather than per problem
            qq,
            max_q: (max_q.max(1) as usize).div_ceil(256) * 256,
            lu,
            _f: core::marker::PhantomData,
        }
    }

    /// The scalar dimension of the system.
    pub fn dim(&self) -> usize {
        self.plan.n
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
        let nvals = vals.len();
        // The values in the factor's scalar: the bytes as they are when the
        // scalars agree, converted on the host otherwise.
        let d_vals = if core::any::TypeId::of::<M>() == core::any::TypeId::of::<F>() {
            let bytes = unsafe { core::slice::from_raw_parts(vals.as_ptr() as *const u8, core::mem::size_of_val(vals)) };
            self.client.create_from_slice(bytes)
        } else {
            let fv: Vec<F> = vals.iter().map(|v| F::from(v.to_f64()).unwrap()).collect();
            self.client.create_from_slice(F::as_bytes(&fv))
        };
        let c = &self.client;
        let arr = |h: &Handle, len: usize| unsafe { BufferArg::from_raw_parts(h.clone(), len.max(1)) };
        let d256 = CubeDim::new_1d(256);
        let count = |n: u32| CubeCount::Static(n, 1, 1);
        let flen = self.plan.factor_vals;
        let plen = self.plan_len;
        let ntiles = (self.plan.tiles.len() / 6) as u32;
        unsafe {
            k_zero::launch_unchecked::<F>(c, count((flen as u32).div_ceil(256)), d256, arr(&self.factor, flen), flen as u32);
            k_zero_u32::launch_unchecked(c, count(1), CubeDim::new_1d(32), arr(&self.fail, 1), 1u32);
            k_seed::launch_unchecked::<F>(c, count(ntiles.div_ceil(256).max(1)), d256, arr(&self.d_tiles, self.plan.tiles.len()), arr(&d_vals, nvals), arr(&self.factor, flen), ntiles);
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
                        arr(&self.factor, flen),
                        arr(&self.factor, flen),
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
                        arr(&self.factor, flen),
                    );
                }
                if d.n_small > 0 {
                    k_small_factor::launch_unchecked::<F>(
                        c,
                        count(d.n_small),
                        d256,
                        arr(&d.small, w.small_targets.len()),
                        arr(&self.d_plan, plen),
                        arr(&self.factor, flen),
                        arr(&self.fail, 1),
                        self.qq,
                    );
                }
                for (st, (&tg, &sg)) in w.trsm_grid.iter().zip(&w.syrk_grid).enumerate() {
                    let j = (st * NB) as u32;
                    k_potf2::launch_unchecked::<F>(c, count(d.n_big), d256, arr(&d.big, w.big_targets.len()), j, arr(&self.d_plan, plen), arr(&self.factor, flen), arr(&self.fail, 1));
                    if tg > 0 {
                        k_trsm::launch_unchecked::<F>(c, count(tg), CubeDim::new_1d(TB as u32), arr(&d.big, w.big_targets.len()), d.n_big, j, arr(&self.d_plan, plen), arr(&self.factor, flen));
                    }
                    if sg > 0 {
                        k_syrk::launch_unchecked::<F>(c, count(sg), d256, arr(&d.big, w.big_targets.len()), d.n_big, j, arr(&self.d_plan, plen), arr(&self.factor, flen));
                    }
                }
            }
        }
        let failed = bytemuck::cast_slice::<u8, u32>(&self.client.read_one(self.fail.clone()).expect("read the pivot flag"))[0];
        if failed != 0 { Err(SupernodalError::NotPositiveDefinite) } else { Ok(()) }
    }

    /// Solve `A x = rhs` in place through the last factorization; `rhs` is in
    /// the matrix's own ordering and comes back as `x`. The permutation is
    /// applied on the host, the level-set solve runs on the device.
    pub fn solve<M: SchurReal>(&mut self, rhs: &mut [M]) {
        let p = &self.plan;
        assert_eq!(rhs.len(), p.n);
        let mut xb = vec![F::from(0.0f64).unwrap(); p.n];
        for k in 0..p.nb {
            let old = p.order[k] as usize;
            for i in 0..p.blk_width[k] as usize {
                xb[p.sc_start[k] as usize + i] = F::from(rhs[p.part[old] as usize + i].to_f64()).unwrap();
            }
        }
        let hx = self.client.create_from_slice(F::as_bytes(&xb));
        let c = &self.client;
        let arr = |h: &Handle, len: usize| unsafe { BufferArg::from_raw_parts(h.clone(), len.max(1)) };
        let d256 = CubeDim::new_1d(256);
        let count = |n: u32| CubeCount::Static(n, 1, 1);
        let flen = p.factor_vals;
        let plen = self.plan_len;
        unsafe {
            for (l, d) in self.levels.iter().enumerate() {
                let w = &self.lw[l];
                if d.n_g > 0 {
                    k_fwd_gather::launch_unchecked::<F>(c, count(d.n_g), d256, arr(&d.g_items, w.g_items.len()), arr(&self.d_plan, plen), arr(&self.factor, flen), arr(&hx, p.n), arr(&hx, p.n));
                }
                k_fwd_trsv::launch_unchecked::<F>(c, count(d.n_all), d256, arr(&d.all, w.level_sn.len()), arr(&self.d_plan, plen), arr(&self.factor, flen), arr(&hx, p.n), self.max_q);
            }
            for (l, d) in self.levels.iter().enumerate().rev() {
                let w = &self.lw[l];
                if d.n_bg > 0 {
                    k_bwd_gemv::launch_unchecked::<F>(c, count(d.n_bg), d256, arr(&d.bg_items, w.bg_items.len()), arr(&self.d_plan, plen), arr(&self.factor, flen), arr(&hx, p.n), self.lu);
                }
                k_bwd_trsv::launch_unchecked::<F>(c, count(d.n_all), d256, arr(&d.all, w.level_sn.len()), arr(&self.d_plan, plen), arr(&self.factor, flen), arr(&hx, p.n), self.max_q, self.lu);
            }
        }
        let bytes = self.client.read_one(hx).expect("read the solution");
        let xr = F::from_bytes(&bytes);
        for k in 0..p.nb {
            let old = p.order[k] as usize;
            for i in 0..p.blk_width[k] as usize {
                rhs[p.part[old] as usize + i] = M::from_f64(xr[p.sc_start[k] as usize + i].to_f64().unwrap());
            }
        }
    }

    /// The factor's values, panel by panel in the symbolic's layout, read
    /// back from the device (for tests and inspection).
    pub fn factor_values(&self) -> Vec<F> {
        sync(&self.client);
        let bytes = self.client.read_one(self.factor.clone()).expect("read the factor");
        F::from_bytes(&bytes)[..self.plan.factor_vals].to_vec()
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::test_util::{grid_spd, rel_residual};
    use crate::{GpuBackend, GpuFactor, GpuPrecision, client};
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
            let mut g = GpuSupernodal::<f64>::new(client(GpuBackend::Cuda, 0).unwrap(), &sn, a.symbolic(), &params);
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
        let mut g = GpuFactor::new(client(GpuBackend::Cuda, 0).unwrap(), GpuPrecision::F32, &sn, a.symbolic(), &Default::default());
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
        let mut g = GpuSupernodal::<f64>::new(client(GpuBackend::Cuda, 0).unwrap(), &sn, &asym, &Default::default());
        assert_eq!(g.factorize(a.vals()), Err(SupernodalError::NotPositiveDefinite));
    }
}
