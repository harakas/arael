//! The kernel cache (feature `cuda` or `hip`), in a process of its own: the
//! cache is chosen once per process, at its first GPU use.
#![cfg(any(feature = "cuda", feature = "hip"))]

#[cfg(feature = "cuda")]
use arael_gpu::cuda::{CudaDevice as Dev, compiles};
#[cfg(all(feature = "hip", not(feature = "cuda")))]
use arael_gpu::hip::{HipDevice as Dev, compiles};
use arael_gpu::{Arg, Device, KernelCache, Launch, set_kernel_cache};

/// The cache's directory for the device's runtime.
#[cfg(feature = "cuda")]
const TARGET: &str = "cuda";
#[cfg(all(feature = "hip", not(feature = "cuda")))]
const TARGET: &str = "hip";

const SOURCE: &str = "KERNEL void twice(GLOBAL float* RESTRICT x, u32 n) { u32 i = global_x(); if (i < n) x[i] *= 2.0f; }";

/// The kernel run on a fresh device: compiled, or loaded from the cache.
fn run_twice(scale: &str) -> Vec<f32> {
    let dev = Dev::new(0).unwrap();
    let k = dev.compile(SOURCE, scale, "twice").unwrap();
    let host = [1f32, 2.0, 3.0];
    let mut x = dev.alloc(12).unwrap();
    dev.write(&mut x, 0, bytemuck::cast_slice(&host)).unwrap();
    dev.launch(&k, Launch::linear(3, 256), &[Arg::Block(&x, 0), Arg::U32(3)]).unwrap();
    let mut out = [0f32; 3];
    dev.read(&x, 0, bytemuck::cast_slice_mut(&mut out)).unwrap();
    out.to_vec()
}

fn entries(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let Ok(read) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut v: Vec<_> = read.flatten().map(|e| e.path()).collect();
    v.sort();
    v
}

#[test]
fn a_compiled_kernel_is_found_again_by_the_next_device() {
    let dir = std::env::temp_dir().join(format!("arael-cache-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    assert!(set_kernel_cache(KernelCache::Dir(dir.clone())));
    let cached = dir.join(TARGET);
    let start = compiles();
    assert_eq!(run_twice(""), [2.0, 4.0, 6.0]);
    assert_eq!(compiles(), start + 1);
    let files = entries(&cached);
    assert_eq!(files.len(), 1, "{files:?}");
    // a new device has nothing in memory: the entry is loaded, not compiled
    assert_eq!(run_twice(""), [2.0, 4.0, 6.0]);
    assert_eq!(compiles(), start + 1);
    // other defines are another source, another entry
    assert_eq!(run_twice("#define UNUSED 1\n"), [2.0, 4.0, 6.0]);
    assert_eq!(compiles(), start + 2);
    assert_eq!(entries(&cached).len(), 2);
    // a damaged entry is a miss, compiled and written again: unreadable
    let good = std::fs::read(&files[0]).unwrap();
    std::fs::write(&files[0], b"not a kernel").unwrap();
    assert_eq!(run_twice(""), [2.0, 4.0, 6.0]);
    assert_eq!(compiles(), start + 3);
    assert_eq!(run_twice(""), [2.0, 4.0, 6.0]);
    assert_eq!(compiles(), start + 3);
    // or the key intact and the binary cut short
    std::fs::write(&files[0], &good[..good.len() - 100]).unwrap();
    assert_eq!(run_twice(""), [2.0, 4.0, 6.0]);
    assert_eq!(compiles(), start + 4);
    assert_eq!(run_twice(""), [2.0, 4.0, 6.0]);
    assert_eq!(compiles(), start + 4);
    assert_eq!(arael_gpu::take_cache_failure(), None);
    let _ = std::fs::remove_dir_all(&dir);
}
