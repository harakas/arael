//! The device df32 (`arael-gpu/kernels/wide.h`, feature `gpu-native-cuda`)
//! against the host's `arael::df32`: the same operations give the same bits.
#![cfg(feature = "gpu-native-cuda")]

use arael::__gpu::bytemuck;
use arael::__gpu::native::cuda::CudaDevice;
use arael::__gpu::native::{Arg, Device, Launch, WIDE, Wide};
use arael::df32;

const SOURCE: &str = r#"
KERNEL void k_ops(GLOBAL const df32* a, GLOBAL const df32* b, GLOBAL df32* out, u32 n) {
    u32 i = global_x();
    if (i >= n) return;
    df32 x = a[i], y = b[i];
    GLOBAL df32* o = out + 8 * i;
    o[0] = x + y;
    o[1] = x - y;
    o[2] = x * y;
    o[3] = x / y;
    o[4] = sqrt(fabs(x));
    o[5] = df32_mul_f32(x, y.hi);
    WidePair p = wide_sin_cos(x);
    o[6] = p.s;
    o[7] = p.c;
}

KERNEL void k_cmp(GLOBAL const df32* a, GLOBAL const df32* b, GLOBAL u32* out, u32 n) {
    u32 i = global_x();
    if (i >= n) return;
    df32 x = a[i], y = b[i];
    out[i] = (x < y ? 1u : 0u) | (x <= y ? 2u : 0u) | (x == y ? 4u : 0u) | (x > y ? 8u : 0u) | (x >= y ? 16u : 0u)
        | (wide_max(x, y) == x ? 32u : 0u) | (wide_min(x, y) == x ? 64u : 0u);
}
"#;

struct Lcg(u64);
impl Lcg {
    fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Values across magnitudes, both signs, with full trailing parts, then
/// special cases.
fn inputs(n: usize, seed: u64, max_exp: f64) -> Vec<df32> {
    let mut r = Lcg(seed);
    let mut v: Vec<df32> = (0..n)
        .map(|_| {
            let m = r.uniform() - 0.5;
            let e = (2.0 * r.uniform() - 1.0) * max_exp;
            df32::from_f64(m * 10f64.powf(e))
        })
        .collect();
    v.extend([0.0, -0.0, 1.0, -1.0, 1e-30, 3.0, f64::INFINITY, f64::NAN, 1e30, std::f64::consts::PI].map(df32::from_f64));
    v
}

fn same(a: df32, b: df32) -> bool {
    if a.hi.is_nan() || b.hi.is_nan() {
        return a.hi.is_nan() && b.hi.is_nan();
    }
    a.hi.to_bits() == b.hi.to_bits() && a.lo.to_bits() == b.lo.to_bits()
}

fn up(dev: &CudaDevice, v: &[df32]) -> arael::__gpu::native::cuda::CudaBlock {
    let flat: Vec<f32> = v.iter().flat_map(|x| [x.hi, x.lo]).collect();
    let mut b = dev.alloc(4 * flat.len()).unwrap();
    dev.write(&mut b, 0, bytemuck::cast_slice(&flat)).unwrap();
    b
}

#[test]
fn device_df32_matches_the_host_bit_for_bit() {
    let dev = CudaDevice::new(0).unwrap();
    let source = format!("{WIDE}\n{SOURCE}");
    let ops = dev.compile(&source, Wide::Df32.defines(), "k_ops").unwrap();
    let cmp = dev.compile(&source, Wide::Df32.defines(), "k_cmp").unwrap();
    for (seed, max_exp) in [(1, 8.0), (2, 2.0), (3, 30.0)] {
        let a = inputs(4000, seed, max_exp);
        let b = inputs(4000, seed + 100, max_exp);
        let n = a.len();
        let (da, db) = (up(&dev, &a), up(&dev, &b));
        let out = dev.alloc(64 * n).unwrap();
        let flags = dev.alloc(4 * n).unwrap();
        dev.launch(&ops, Launch::linear(n as u32, 256), &[Arg::Block(&da, 0), Arg::Block(&db, 0), Arg::Block(&out, 0), Arg::U32(n as u32)]).unwrap();
        dev.launch(&cmp, Launch::linear(n as u32, 256), &[Arg::Block(&da, 0), Arg::Block(&db, 0), Arg::Block(&flags, 0), Arg::U32(n as u32)]).unwrap();
        let mut got = vec![0f32; 16 * n];
        dev.read(&out, 0, bytemuck::cast_slice_mut(&mut got)).unwrap();
        let mut got_flags = vec![0u32; n];
        dev.read(&flags, 0, bytemuck::cast_slice_mut(&mut got_flags)).unwrap();
        let mut bad = 0;
        for i in 0..n {
            let (x, y) = (a[i], b[i]);
            // sin_cos where the reduction keeps the digits the host keeps
            let (s, c) = x.sin_cos();
            let want = [x + y, x - y, x * y, x / y, x.abs().sqrt(), x.mul_f32(y.hi), s, c];
            for (k, w) in want.iter().enumerate() {
                let g = df32 { hi: got[16 * i + 2 * k], lo: got[16 * i + 2 * k + 1] };
                if !same(g, *w) {
                    bad += 1;
                    if bad < 10 {
                        eprintln!("op {k} of ({:?}, {:?}) and ({:?}, {:?}): device ({:?}, {:?}), host ({:?}, {:?})", x.hi, x.lo, y.hi, y.lo, g.hi, g.lo, w.hi, w.lo);
                    }
                }
            }
            let host = (x < y) as u32 | ((x <= y) as u32) << 1 | ((x == y) as u32) << 2 | ((x > y) as u32) << 3 | ((x >= y) as u32) << 4
                | ((x.max(y) == x) as u32) << 5
                | ((x.min(y) == x) as u32) << 6;
            if got_flags[i] != host {
                bad += 1;
                if bad < 10 {
                    eprintln!("comparisons of ({:?}, {:?}) and ({:?}, {:?}): device {:07b}, host {:07b}", x.hi, x.lo, y.hi, y.lo, got_flags[i], host);
                }
            }
        }
        assert_eq!(bad, 0, "seed {seed}: {bad} results differ from the host");
    }
}

/// `Wide::Df32` splits and joins as the host's df32 does.
#[test]
fn the_host_encoding_is_the_df32_split() {
    for v in [0.1, -1.0 / 3.0, 1e20, 5e-30, f64::INFINITY] {
        let d = df32::from_f64(v);
        let bits = Wide::Df32.bits(v);
        assert_eq!(bits as u32, d.hi.to_bits());
        assert_eq!((bits >> 32) as u32, d.lo.to_bits());
        assert_eq!(Wide::Df32.value(bits), d.to_f64());
    }
    assert_eq!(Wide::F64.value(Wide::F64.bits(0.1)), 0.1);
}
