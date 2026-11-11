//! The side stream ([`Device::launch_on`], [`Device::fork`], [`Device::join`]):
//! side launches see the main stream's work before the fork, the main
//! stream's work after the join sees theirs, and a capture records both
//! streams. On CUDA (feature `cuda`), else on HIP (`hip`), else on Metal
//! (`metal`), else on the CPU device (`cpu`).
#![cfg(any(feature = "cuda", feature = "hip", feature = "cpu", all(feature = "metal", target_vendor = "apple")))]

use arael_gpu::{Arg, Device, GpuError, Lane, Launch};

#[cfg(feature = "cuda")]
type Dev = arael_gpu::cuda::CudaDevice;
#[cfg(all(feature = "hip", not(feature = "cuda")))]
type Dev = arael_gpu::hip::HipDevice;
#[cfg(all(feature = "metal", target_vendor = "apple", not(any(feature = "cuda", feature = "hip"))))]
type Dev = arael_gpu::metal::MetalDevice;
#[cfg(all(feature = "cpu", not(any(feature = "cuda", feature = "hip", all(feature = "metal", target_vendor = "apple")))))]
type Dev = arael_gpu::cpu::CpuDevice;

type Block = <Dev as Device>::Block;

// `spin_*` loop before their write, so a launch that did not wait for one
// reads the old value
const SOURCE: &str = r#"
KERNEL void spin_set(GLOBAL float* x, float v, u32 iters, u32 n) {
    u32 i = global_x();
    float acc = 0.0f;
    for (u32 k = 0; k < iters; k++) acc = acc * 0.5f;
    if (i < n) x[i] = v + acc;
}
KERNEL void spin_add(GLOBAL float* y, GLOBAL const float* x, u32 iters, u32 n) {
    u32 i = global_x();
    float acc = 0.0f;
    for (u32 k = 0; k < iters; k++) acc = acc * 0.5f;
    if (i < n) y[i] += x[i] + acc;
}
"#;

const N: u32 = 1 << 16;

type Kernel = <Dev as Device>::Kernel;

fn kernels(dev: &Dev) -> (Kernel, Kernel) {
    (dev.compile(SOURCE, "", "spin_set").unwrap(), dev.compile(SOURCE, "", "spin_add").unwrap())
}

/// x set to 1 on the main stream; forked, the side adds x into y twice
/// while the main stream sets z; joined, the main stream adds y into x.
fn sequence(dev: &Dev, (set, add): &(Kernel, Kernel), x: &Block, y: &Block, z: &Block) -> Result<(), GpuError> {
    let l = Launch::linear(N, 256);
    dev.launch(set, l, &[Arg::Block(x, 0), Arg::F32(1.0), Arg::U32(20000), Arg::U32(N)])?;
    dev.fork()?;
    dev.launch(set, l, &[Arg::Block(z, 0), Arg::F32(5.0), Arg::U32(200000), Arg::U32(N)])?;
    dev.launch_on(Lane::Side, add, l, &[Arg::Block(y, 0), Arg::Block(x, 0), Arg::U32(1), Arg::U32(N)])?;
    dev.launch_on(Lane::Side, add, l, &[Arg::Block(y, 0), Arg::Block(x, 0), Arg::U32(20000), Arg::U32(N)])?;
    dev.join()?;
    dev.launch(add, l, &[Arg::Block(x, 0), Arg::Block(y, 0), Arg::U32(1), Arg::U32(N)])
}

/// y is 2 (the side saw x set) and x is 3 (the main stream saw the side's y).
fn check(dev: &Dev, x: &Block, y: &Block) {
    let (mut xs, mut ys) = (vec![0f32; N as usize], vec![0f32; N as usize]);
    dev.read_many(&mut [(x, 0, bytemuck::cast_slice_mut(&mut xs)), (y, 0, bytemuck::cast_slice_mut(&mut ys))]).unwrap();
    assert!(ys.iter().all(|&v| v == 2.0), "the side saw x as {:?}", &ys[..4]);
    assert!(xs.iter().all(|&v| v == 3.0), "the main stream saw y as {:?}", &xs[..4]);
}

fn zero(dev: &Dev, b: &mut Block) {
    dev.write(b, 0, bytemuck::cast_slice(&vec![0f32; N as usize])).unwrap();
}

#[test]
fn a_fork_orders_the_side_stream_after_the_main_and_a_join_back() {
    let dev = Dev::new(0).unwrap();
    let k = kernels(&dev);
    let bytes = 4 * N as usize;
    let (mut x, mut y, z) = (dev.alloc(bytes).unwrap(), dev.alloc(bytes).unwrap(), dev.alloc(bytes).unwrap());
    sequence(&dev, &k, &x, &y, &z).unwrap();
    check(&dev, &x, &y);
    let graph = dev.capture(|| sequence(&dev, &k, &x, &y, &z)).unwrap();
    for _ in 0..2 {
        zero(&dev, &mut x);
        zero(&dev, &mut y);
        dev.replay(&graph).unwrap();
        check(&dev, &x, &y);
    }
    dev.sync().unwrap();
}

/// A capture that leaves its fork open fails, and the device goes on.
#[test]
fn a_capture_with_its_fork_open_fails() {
    let dev = Dev::new(0).unwrap();
    match dev.capture(|| dev.fork()) {
        Err(GpuError::Capture(why)) => assert!(why.contains("not joined"), "{why}"),
        Err(e) => panic!("{e}"),
        Ok(_) => panic!("a capture with its fork open succeeded"),
    }
    dev.join().unwrap();
    let k = kernels(&dev);
    let bytes = 4 * N as usize;
    let (x, y, z) = (dev.alloc(bytes).unwrap(), dev.alloc(bytes).unwrap(), dev.alloc(bytes).unwrap());
    sequence(&dev, &k, &x, &y, &z).unwrap();
    check(&dev, &x, &y);
}

#[test]
#[should_panic(expected = "a side launch outside a fork and its join")]
fn a_side_launch_without_a_fork_panics() {
    let dev = Dev::new(0).unwrap();
    let k = dev.compile(SOURCE, "", "spin_set").unwrap();
    let x = dev.alloc(4).unwrap();
    let _ = dev.launch_on(Lane::Side, &k, Launch::linear(1, 32), &[Arg::Block(&x, 0), Arg::F32(1.0), Arg::U32(1), Arg::U32(1)]);
}

#[test]
#[should_panic(expected = "a join without a fork")]
fn a_join_without_a_fork_panics() {
    let dev = Dev::new(0).unwrap();
    let _ = dev.join();
}
