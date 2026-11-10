//! A warp is 32 lanes on every device: `shfl`, `shfl_xor`, `warp_sum` and
//! `WARP_BARRIER` act within threads 32 w to 32 w + 31 of a block. A 64-lane
//! wavefront (AMD GCN, CDNA) holds two warps, which must not see each
//! other, also when only one of them runs. On CUDA (feature `cuda`), else on
//! HIP (`hip`), else on Metal (`metal`), else on the CPU device (`cpu`).
#![cfg(any(feature = "cuda", feature = "hip", feature = "cpu", all(feature = "metal", target_vendor = "apple")))]

use arael_gpu::{Arg, Device, Launch};

#[cfg(feature = "cuda")]
type Dev = arael_gpu::cuda::CudaDevice;
#[cfg(all(feature = "hip", not(feature = "cuda")))]
type Dev = arael_gpu::hip::HipDevice;
#[cfg(all(feature = "metal", target_vendor = "apple", not(any(feature = "cuda", feature = "hip"))))]
type Dev = arael_gpu::metal::MetalDevice;
#[cfg(all(feature = "cpu", not(any(feature = "cuda", feature = "hip", all(feature = "metal", target_vendor = "apple")))))]
type Dev = arael_gpu::cpu::CpuDevice;

const OUTS: usize = 9;

const SOURCE: &str = r#"
KERNEL_THREADS(256) void warps(GLOBAL float* out) {
    SHARED float s[256];
    u32 t = thread_x();
    u32 i = global_x();
    float v = (float)i;
    GLOBAL float* o = out + 9 * i;
    o[0] = shfl_xor(v, 1u);
    o[1] = shfl(v, 5u);
    o[2] = shfl(v, 37u);
#if HAS_F64
    double d = (double)i;
    o[3] = (float)shfl(d, 7u);
    o[4] = (float)warp_sum(d);
#else
    // a device without f64: the float forms
    o[3] = shfl(v, 7u);
    o[4] = warp_sum(v);
#endif
    o[5] = warp_sum(v);
    o[6] = -1.0f;
    o[7] = -1.0f;
    o[8] = -1.0f;
    // only the odd warps of the block run
    if ((t / 32u) % 2u == 1u) {
        o[6] = shfl(v, 3u);
        o[7] = warp_sum(v);
        s[t] = v;
        WARP_BARRIER();
        o[8] = s[(t & ~31u) | ((t + 1u) & 31u)];
    }
}
"#;

#[test]
fn warp_operations_stay_within_their_32_lanes() {
    let dev = Dev::new(0).unwrap();
    let kernel = dev.compile(SOURCE, "", "warps").unwrap();
    let n = 512usize;
    let out = dev.alloc(4 * OUTS * n).unwrap();
    dev.launch(&kernel, Launch::linear(n as u32, 256), &[Arg::Block(&out, 0)]).unwrap();
    let mut v = vec![0f32; OUTS * n];
    dev.read(&out, 0, bytemuck::cast_slice_mut(&mut v)).unwrap();
    for i in 0..n {
        let base = i / 32 * 32;
        let sum = (base..base + 32).sum::<usize>() as f32;
        let o = &v[OUTS * i..OUTS * (i + 1)];
        assert_eq!(o[0], (i ^ 1) as f32, "shfl_xor in thread {i}");
        assert_eq!(o[1], (base + 5) as f32, "shfl in thread {i}");
        assert_eq!(o[2], (base + 5) as f32, "shfl from lane 37 wraps to lane 5 in thread {i}");
        assert_eq!(o[3], (base + 7) as f32, "shfl of a double (a float without f64) in thread {i}");
        assert_eq!(o[4], sum, "warp_sum of a double (a float without f64) in thread {i}");
        assert_eq!(o[5], sum, "warp_sum in thread {i}");
        if (i % 256 / 32) % 2 == 1 {
            assert_eq!(o[6], (base + 3) as f32, "shfl, the other warp idle, in thread {i}");
            assert_eq!(o[7], sum, "warp_sum, the other warp idle, in thread {i}");
            assert_eq!(o[8], (base + (i + 1) % 32) as f32, "WARP_BARRIER, the other warp idle, in thread {i}");
        } else {
            assert_eq!(&o[6..], &[-1.0; 3], "an idle warp in thread {i}");
        }
    }
}
