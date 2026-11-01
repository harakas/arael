//! The kernel cache (feature `cuda`), in a process of its own: the cache
//! is chosen once per process, at its first GPU use.
#![cfg(feature = "cuda")]

use arael_gpu::cubecl::bytes::Bytes;
use arael_gpu::{GpuBackend, GpuContext, KernelCache, kernel_cache, set_kernel_cache};

fn files_under(dir: &std::path::Path) -> usize {
    let Ok(entries) = std::fs::read_dir(dir) else { return 0 };
    entries.flatten().map(|e| if e.path().is_dir() { files_under(&e.path()) } else { 1 }).sum()
}

#[test]
fn compiled_kernels_land_in_the_chosen_directory() {
    let dir = std::env::temp_dir().join(format!("arael-kernel-cache-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    assert!(set_kernel_cache(KernelCache::Dir(dir.clone())));
    let ctx = GpuContext::new(GpuBackend::Cuda, 0).unwrap();
    // fixed at the first GPU use
    assert!(!set_kernel_cache(KernelCache::Off));
    assert_eq!(kernel_cache(), Some(KernelCache::Dir(dir.clone())));
    // a kernel compiled and run
    let n = 300usize;
    let pos: Vec<u32> = (0..n as u32).collect();
    let h = ctx.lease(n * 8);
    ctx.client().write(h.handle(), Bytes::from_elems(vec![2.0f64; n]));
    let v = arael_gpu::lm::LmVectors::new(&ctx, &pos);
    v.gather_diag(h.handle(), n);
    let diag = arael_gpu::read_one(ctx.client(), v.diag()).unwrap();
    assert_eq!(bytemuck::cast_slice::<u8, f64>(&diag)[n - 1], 2.0);
    assert!(files_under(&dir) > 0, "nothing under {dir:?}");
    let _ = std::fs::remove_dir_all(&dir);
}
