//! Graph capture and replay on the native runtime (CUDA, else HIP): a
//! capture records launches without running them, a replay runs them against
//! the blocks' contents at the time, operations other than launches are
//! refused while a capture is open, and a capture (thread-local) leaves other
//! threads' work alone, CubeCL's included.
#![cfg(any(feature = "native-cuda", feature = "native-hip"))]

#[cfg(feature = "native-cuda")]
use arael_gpu::native::cuda::CudaDevice as Dev;
#[cfg(all(feature = "native-hip", not(feature = "native-cuda")))]
use arael_gpu::native::hip::HipDevice as Dev;
use arael_gpu::native::{Arg, Device, Launch, NativeError};
#[cfg(all(feature = "cuda", feature = "native-cuda"))]
use cubecl::prelude::*;

type Block = <Dev as Device>::Block;
type Kernel = <Dev as Device>::Kernel;

const SOURCE: &str = r#"
KERNEL void axpc(GLOBAL const float* RESTRICT x, GLOBAL float* RESTRICT y, float a, u32 n) {
    u32 i = global_x();
    if (i < n) y[i] = a * x[i] + 1.0f;
}

// n dependent multiply-adds by one thread: work that keeps the device busy.
KERNEL void spin(GLOBAL float* RESTRICT out, u32 n) {
    float x = 0.0f;
    for (u32 i = 0; i < n; i++) x = x * 0.999f + 1.0f;
    out[0] = x;
}
"#;

const N: usize = 1000;

/// `y = a x + 1` over N entries.
fn axpc(dev: &Dev, k: &Kernel, x: &Block, y: &Block, a: f32) -> Result<(), NativeError> {
    dev.launch(k, Launch::linear(N as u32, 256), &[Arg::Block(x, 0), Arg::Block(y, 0), Arg::F32(a), Arg::U32(N as u32)])
}

fn read(dev: &Dev, b: &Block) -> Vec<f32> {
    let mut v = vec![0f32; N];
    dev.read(b, 0, bytemuck::cast_slice_mut(&mut v)).unwrap();
    v
}

#[test]
fn a_replay_runs_the_recorded_launches_against_the_blocks_contents() {
    let dev = Dev::new(0).unwrap();
    let k = dev.compile(SOURCE, "", "axpc").unwrap();
    let mut x = dev.alloc(4 * N).unwrap();
    let y = dev.alloc(4 * N).unwrap();
    let graph = dev.capture(|| axpc(&dev, &k, &x, &y, 3.0)).unwrap();
    assert!(read(&dev, &y).iter().all(|&v| v == 0.0), "a capture runs nothing");
    for round in 0..3 {
        let host: Vec<f32> = (0..N).map(|i| (i + round) as f32).collect();
        dev.write(&mut x, 0, bytemuck::cast_slice(&host)).unwrap();
        dev.replay(&graph).unwrap();
        let got = read(&dev, &y);
        for i in 0..N {
            assert_eq!(got[i], 3.0 * host[i] + 1.0, "round {round}, entry {i}");
        }
    }
}

/// `read_many` reads each range, of several blocks and offsets, after the
/// replay queued before it.
#[test]
fn a_read_of_several_ranges_comes_after_the_queued_work() {
    let dev = Dev::new(0).unwrap();
    let k = dev.compile(SOURCE, "", "axpc").unwrap();
    let mut x = dev.alloc(4 * N).unwrap();
    let y = dev.alloc(4 * N).unwrap();
    let host: Vec<f32> = (0..N).map(|i| i as f32).collect();
    dev.write(&mut x, 0, bytemuck::cast_slice(&host)).unwrap();
    let graph = dev.capture(|| axpc(&dev, &k, &x, &y, 2.0)).unwrap();
    dev.replay(&graph).unwrap();
    let (mut a, mut b, mut c) = ([0f32; 3], [0f32; 5], [0f32; 2]);
    dev.read_many(&mut [
        (&y, 4 * 10, bytemuck::cast_slice_mut(&mut a)),
        (&x, 4 * 100, bytemuck::cast_slice_mut(&mut b)),
        (&y, 4 * (N - 2), bytemuck::cast_slice_mut(&mut c)),
    ])
    .unwrap();
    assert_eq!(a, [21.0, 23.0, 25.0]);
    assert_eq!(b, [100.0, 101.0, 102.0, 103.0, 104.0]);
    assert_eq!(c, [2.0 * (N - 2) as f32 + 1.0, 2.0 * (N - 1) as f32 + 1.0]);
}

#[test]
fn only_launches_are_allowed_inside_a_capture() {
    let dev = Dev::new(0).unwrap();
    let k = dev.compile(SOURCE, "", "axpc").unwrap();
    let mut x = dev.alloc(4 * N).unwrap();
    let y = dev.alloc(4 * N).unwrap();
    let mut z = dev.alloc(16).unwrap();
    let refused = |r: Result<(), NativeError>| assert!(matches!(r, Err(NativeError::Capture(_))), "{r:?}");
    let mut host = [0f32; 4];
    let graph = dev
        .capture(|| {
            axpc(&dev, &k, &x, &y, 2.0)?;
            refused(dev.alloc(16).map(drop));
            refused(dev.write(&mut z, 0, &[0u8; 4]));
            refused(dev.read(&y, 0, bytemuck::cast_slice_mut(&mut host)));
            refused(dev.read_many(&mut [(&y, 0, bytemuck::cast_slice_mut(&mut host))]));
            refused(dev.sync());
            refused(dev.compile(SOURCE, "#define OTHER 1\n", "axpc").map(drop));
            refused(dev.capture(|| Ok(())).map(drop));
            Ok(())
        })
        .unwrap();
    // the refusals left the capture intact
    dev.write(&mut x, 0, bytemuck::cast_slice(&vec![1f32; N])).unwrap();
    dev.replay(&graph).unwrap();
    assert!(read(&dev, &y).iter().all(|&v| v == 3.0));
}

#[test]
fn a_failed_or_panicking_recording_leaves_the_device_usable() {
    let dev = Dev::new(0).unwrap();
    let k = dev.compile(SOURCE, "", "axpc").unwrap();
    let mut x = dev.alloc(4 * N).unwrap();
    let y = dev.alloc(4 * N).unwrap();
    let failed = dev.capture(|| {
        axpc(&dev, &k, &x, &y, 2.0)?;
        Err(NativeError::Limit("the recording's own error".into()))
    });
    assert!(matches!(failed, Err(NativeError::Limit(_))), "the recording's error comes back");
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = dev.capture(|| {
            axpc(&dev, &k, &x, &y, 2.0)?;
            panic!("a recording that panics");
        });
    }));
    assert!(panicked.is_err());
    // an empty recording replays as nothing
    let empty = dev.capture(|| Ok(())).unwrap();
    dev.replay(&empty).unwrap();
    // the stream is out of capture: writes, launches and a new capture work
    dev.write(&mut x, 0, bytemuck::cast_slice(&vec![2f32; N])).unwrap();
    axpc(&dev, &k, &x, &y, 2.0).unwrap();
    assert!(read(&dev, &y).iter().all(|&v| v == 5.0), "the discarded recordings never ran");
    let graph = dev.capture(|| axpc(&dev, &k, &x, &y, 2.0)).unwrap();
    dev.write(&mut x, 0, bytemuck::cast_slice(&vec![0f32; N])).unwrap();
    dev.replay(&graph).unwrap();
    assert!(read(&dev, &y).iter().all(|&v| v == 1.0));
}

/// A thread-local capture forbids nothing to other threads: one thread
/// captures and replays long recordings while another reads and waits on a
/// second device (and, with CubeCL in the build, a third runs CubeCL's own
/// captures and reads). In the driver's global mode the reads would fail.
#[test]
fn captures_leave_other_threads_alone() {
    let (capturing, reading) = (Dev::new(0).unwrap(), Dev::new(0).unwrap());
    let (kc, kr) = (capturing.compile(SOURCE, "", "spin").unwrap(), reading.compile(SOURCE, "", "spin").unwrap());
    let (bc, br) = (capturing.alloc(4).unwrap(), reading.alloc(4).unwrap());
    let one = Launch::linear(1, 1);
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(1500);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            let mut captures = 0;
            while std::time::Instant::now() < deadline {
                let graph = capturing.capture(|| (0..300).try_for_each(|_| capturing.launch(&kc, one, &[Arg::Block(&bc, 0), Arg::U32(10)]))).unwrap();
                capturing.replay(&graph).unwrap();
                capturing.sync().unwrap();
                captures += 1;
            }
            assert!(captures > 10, "{captures} captures");
        });
        scope.spawn(|| {
            let mut v = [0f32];
            while std::time::Instant::now() < deadline {
                reading.launch(&kr, one, &[Arg::Block(&br, 0), Arg::U32(1000)]).unwrap();
                reading.read(&br, 0, bytemuck::cast_slice_mut(&mut v)).expect("a read beside a capture");
                reading.sync().expect("a wait beside a capture");
            }
        });
        #[cfg(all(feature = "cuda", feature = "native-cuda"))]
        scope.spawn(|| cubecl_captures_until(deadline));
    });
}

/// `spin` in CubeCL.
#[cfg(all(feature = "cuda", feature = "native-cuda"))]
#[cubecl::cube(launch_unchecked)]
fn k_spin(out: &mut [f32], n: u32) {
    let mut x = 0.0f32;
    let mut i = 0u32;
    while i < n {
        x = x * 0.999f32 + 1.0f32;
        i += 1u32;
    }
    out[0] = x;
}

/// CubeCL's global-mode captures and reads on a context of its own, until `deadline`.
#[cfg(all(feature = "cuda", feature = "native-cuda"))]
fn cubecl_captures_until(deadline: std::time::Instant) {
    let ctx = arael_gpu::GpuContext::new(arael_gpu::GpuBackend::Cuda, 0).unwrap();
    let b = ctx.lease(4);
    let spin = |n: u32| unsafe { k_spin::launch_unchecked(ctx.client(), CubeCount::Static(1, 1, 1), CubeDim::new_1d(1), BufferArg::from_raw_parts(b.handle().clone(), 1), n) };
    while std::time::Instant::now() < deadline {
        ctx.client().graph_prepare().unwrap();
        spin(1);
        arael_gpu::sync(ctx.client());
        let graph = arael_gpu::capture(ctx.client(), || (0..100).for_each(|_| spin(10)));
        unsafe { graph.replay() }.unwrap();
        arael_gpu::read_one(ctx.client(), b.view(4)).unwrap();
    }
}
