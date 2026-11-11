//! A factorization with the lookahead (`GpuSupernodalParams::lookahead`)
//! failing at each of its device operations in turn, its forks and joins
//! among them: each failure is reported, and the factorization after it is
//! the CPU's, the side stream joined back whatever failed. Fault injection is
//! process-wide, so this binary holds this one test. On CUDA (feature
//! `cuda`), else on HIP (`hip`), else on Metal (`metal`), else on the CPU
//! device (`cpu`).
#![cfg(any(feature = "cuda", feature = "hip", feature = "cpu", all(feature = "metal", target_vendor = "apple")))]

use std::sync::Arc;

use arael_faer::supernodal::{SupernodalContext, SupernodalParams, SupernodalSymbolic, nd_block_order, supernodal_factorize};
use arael_gpu::test_util::{grid_spd, wide};
use arael_gpu::{FactorizeError, GpuError, GpuSupernodal, GpuSupernodalParams, fault};

#[cfg(feature = "cuda")]
type Dev = arael_gpu::cuda::CudaDevice;
#[cfg(all(feature = "hip", not(feature = "cuda")))]
type Dev = arael_gpu::hip::HipDevice;
#[cfg(all(feature = "metal", target_vendor = "apple", not(any(feature = "cuda", feature = "hip"))))]
type Dev = arael_gpu::metal::MetalDevice;
#[cfg(all(feature = "cpu", not(any(feature = "cuda", feature = "hip", all(feature = "metal", target_vendor = "apple")))))]
type Dev = arael_gpu::cpu::CpuDevice;

/// The factor's lower parts against the CPU's, as err / max.
fn factor_err(sn: &SupernodalSymbolic, got: &[f32], cpu: &[f64]) -> f64 {
    let (mut mx, mut df) = (0.0f64, 0.0f64);
    for s in 0..sn.n_supernodes() {
        let (q, h) = sn.supernode_dims(s);
        let base = sn.panel_range(s).start;
        for c in 0..q {
            for r in c..h {
                mx = mx.max(cpu[base + c * h + r].abs());
                df = df.max((got[base + c * h + r] as f64 - cpu[base + c * h + r]).abs());
            }
        }
    }
    df / mx
}

#[test]
fn every_failure_in_a_lookahead_factorization_is_reported_and_the_next_is_right() {
    // supernodes wider than two blocks of NB: every step forks and joins
    let (a, _, _) = grid_spd(24, &[3, 6], 11);
    let order = nd_block_order(a.symbolic());
    let sp = SupernodalParams { relax: Some(vec![(320, 1.0)]), ..SupernodalParams::default() };
    let sn = SupernodalSymbolic::new(a.symbolic(), Some(&order), &sp).unwrap();
    let mut cpu = vec![0.0f64; sn.factor_val_count()];
    supernodal_factorize(&sn, &a, &mut cpu, &mut SupernodalContext::new(), 1).unwrap();
    let dev = Arc::new(Dev::new(0).unwrap());
    let params = GpuSupernodalParams { syrk_block: Some(32), lookahead: Some(true), ..GpuSupernodalParams::default() };
    let mut g = GpuSupernodal::<f32, _>::new(&dev, &sn, a.symbolic(), &params, wide(&*dev)).unwrap();
    // the kernels compiled
    g.factorize(a.vals()).unwrap();
    fault::arm(0);
    g.factorize(a.vals()).unwrap();
    let operations = fault::disarm();
    let (mut forks, mut joins) = (0, 0);
    for k in 1..=operations {
        fault::arm(k);
        let failed = g.factorize(a.vals());
        fault::disarm();
        match failed {
            Err(FactorizeError::Device(GpuError::Driver(why))) => {
                assert!(why.contains(&format!("injected failure of operation {k} ")), "operation {k}: {why}");
                forks += why.ends_with("(fork)") as usize;
                joins += why.ends_with("(join)") as usize;
            }
            other => panic!("operation {k} of {operations}: {other:?}"),
        }
        g.factorize(a.vals()).unwrap();
        let err = factor_err(&sn, &g.factor_values().unwrap(), &cpu);
        assert!(err < 1e-5, "after operation {k} of {operations}: f32 factor err/max {err}");
    }
    assert!(forks > 0 && joins > 0, "{forks} forks and {joins} joins among the {operations} failures");
}
