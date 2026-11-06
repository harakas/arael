//! The device's pool (CUDA, else HIP, else the CPU device): a dropped block
//! is handed out again, zeroed, a fresh one has room for a quarter more, so
//! solves of a slightly changed structure allocate nothing new.
#![cfg(any(feature = "cuda", feature = "hip", feature = "cpu"))]

use std::sync::Arc;

use arael_faer::schur::schur_symbolic;
use arael_faer::supernodal::{SupernodalParams, SupernodalSymbolic, nd_block_order};
#[cfg(feature = "cuda")]
use arael_gpu::cuda::CudaDevice as Dev;
#[cfg(all(feature = "hip", not(feature = "cuda")))]
use arael_gpu::hip::HipDevice as Dev;
#[cfg(all(feature = "cpu", not(any(feature = "cuda", feature = "hip"))))]
use arael_gpu::cpu::CpuDevice as Dev;
use arael_gpu::test_util::{ba_like, rel_residual};
use arael_gpu::{Device, GpuFactor, GpuPrecision, GpuSchur, Wide};

#[test]
fn dropped_blocks_are_reused_zeroed_and_with_room_to_grow() {
    let dev = Dev::new(0).unwrap();
    let sizes = [100usize, 4096, 1 << 20, 3 << 20];
    let blocks: Vec<_> = sizes
        .iter()
        .map(|&n| {
            let mut b = dev.alloc(n).unwrap();
            dev.write(&mut b, 0, &vec![7u8; n]).unwrap();
            b
        })
        .collect();
    let (made, reserved) = (dev.allocations(), dev.reserved_bytes());
    assert_eq!(made, sizes.len());
    assert!(reserved >= sizes.iter().map(|n| n + n / 4).sum::<usize>(), "{reserved}");
    drop(blocks);
    // the same sizes, then a fifth larger: the dropped blocks serve both
    for grow in [1.0, 1.2] {
        let again: Vec<_> = sizes.iter().map(|&n| dev.alloc((n as f64 * grow) as usize).unwrap()).collect();
        assert_eq!((dev.allocations(), dev.reserved_bytes()), (made, reserved), "grown by {grow}");
        for b in &again {
            let mut back = vec![1u8; b.len()];
            dev.read(b, 0, &mut back).unwrap();
            assert!(back.iter().all(|&v| v == 0), "a reused block comes back zeroed");
        }
    }
    // twice as large is a new block
    let big = dev.alloc(2 * sizes[3]).unwrap();
    assert_eq!(dev.allocations(), made + 1);
    assert_eq!(big.len(), 2 * sizes[3]);
}

/// A fresh reduction and factor per solve, as a caller solving problem after
/// problem makes them: after the first structure, a different structure of
/// the same size and one 5% larger allocate nothing.
#[test]
fn solves_of_slightly_changed_structures_allocate_nothing() {
    let dev = Arc::new(Dev::new(0).unwrap());
    let mut after = Vec::new();
    for (np, nl, seed) in [(80usize, 1200usize, 2u64), (80, 1200, 4), (84, 1260, 5), (76, 1140, 6)] {
        let (h, dense, rhs, elim) = ba_like(np, nl, seed);
        let sym = schur_symbolic(h.symbolic(), &elim).unwrap();
        let order = nd_block_order(&sym.s);
        let sn = SupernodalSymbolic::new(&sym.s, Some(&order), &SupernodalParams::default()).unwrap();
        let n = h.symbolic().ncols();
        let mut g = GpuSchur::new(&dev, &sym, h.symbolic(), Wide::F64).unwrap();
        let mut f = GpuFactor::new(&dev, GpuPrecision::F32, &sn, &sym.s, &Default::default(), Wide::F64).unwrap();
        let mut delta = vec![0.0f64; n];
        for _ in 0..3 {
            g.solve(&mut f, h.vals(), &rhs, Some((4, 1e-10)), &mut delta).unwrap();
        }
        assert!(rel_residual(&dense, n, &delta, &rhs) < 1e-8);
        after.push(dev.allocations());
    }
    assert!(after.iter().all(|&a| a == after[0]), "allocations after each structure: {after:?}");
}
