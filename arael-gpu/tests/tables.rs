//! Struct arguments taken by value (`Arg::Bytes`, `TABLE_PARAM(T)`; CUDA,
//! else HIP, else Metal, else the CPU device): the kernel reads the bytes
//! given, launched directly, captured and replayed, and timed; a graph
//! keeps the bytes of its capture.
#![cfg(any(feature = "cuda", feature = "hip", feature = "cpu", all(feature = "metal", target_vendor = "apple")))]

#[cfg(feature = "cuda")]
use arael_gpu::cuda::CudaDevice as Dev;
#[cfg(all(feature = "hip", not(feature = "cuda")))]
use arael_gpu::hip::HipDevice as Dev;
#[cfg(all(feature = "metal", target_vendor = "apple", not(any(feature = "cuda", feature = "hip"))))]
use arael_gpu::metal::MetalDevice as Dev;
#[cfg(all(feature = "cpu", not(any(feature = "cuda", feature = "hip", all(feature = "metal", target_vendor = "apple")))))]
use arael_gpu::cpu::CpuDevice as Dev;
use arael_gpu::{Arg, Device, Launch};

type Block = <Dev as Device>::Block;
type Kernel = <Dev as Device>::Kernel;

/// `Wide` is 37 words of 8 bytes; `Odd` 5 of 4 bytes, 20 bytes, no multiple
/// of 8. Each thread i copies word i of both into `out`, the table between
/// two other arguments.
const SOURCE: &str = r#"
struct Wide { u32 w[2 * 37]; };
struct Odd { u32 w[5]; };
KERNEL void k_copy(GLOBAL u32* RESTRICT out, TABLE_PARAM(Wide) a, u32 n, TABLE_PARAM(Odd) b) {
    u32 i = global_x();
    if (i < n) out[i] = a.w[i];
    if (i < 5) out[n + i] = b.w[i];
}
"#;

const WIDE: usize = 2 * 37;

fn launch(dev: &Dev, k: &Kernel, out: &Block, a: &[u32; WIDE], b: &[u32; 5]) {
    let args = [Arg::Block(out, 0), Arg::Bytes(bytemuck::cast_slice(a)), Arg::U32(WIDE as u32), Arg::Bytes(bytemuck::cast_slice(b))];
    dev.launch(k, Launch::linear(WIDE as u32, 128), &args).unwrap();
}

fn read(dev: &Dev, out: &Block) -> Vec<u32> {
    let mut v = vec![0u32; WIDE + 5];
    dev.read(out, 0, bytemuck::cast_slice_mut(&mut v)).unwrap();
    v
}

fn tables(seed: u32) -> ([u32; WIDE], [u32; 5]) {
    (core::array::from_fn(|i| seed * 1000 + i as u32 * 7 + 1), core::array::from_fn(|i| seed * 1000 + 500 + i as u32))
}

fn expect(a: &[u32; WIDE], b: &[u32; 5]) -> Vec<u32> {
    a.iter().chain(b).copied().collect()
}

#[test]
fn a_kernel_reads_the_tables_it_is_given() {
    let dev = Dev::new(0).unwrap();
    let k = dev.compile(SOURCE, "", "k_copy").unwrap();
    let out = dev.alloc(4 * (WIDE + 5)).unwrap();
    for seed in 1..4 {
        let (a, b) = tables(seed);
        launch(&dev, &k, &out, &a, &b);
        assert_eq!(read(&dev, &out), expect(&a, &b), "launch {seed}");
    }
}

/// A graph replays the tables of its capture, whatever the host's buffers
/// hold since, also with every launch timed (a timed capture keeps its
/// launches to issue at each replay).
#[test]
fn a_graph_keeps_the_tables_of_its_capture() {
    let dev = Dev::new(0).unwrap();
    let k = dev.compile(SOURCE, "", "k_copy").unwrap();
    let out = dev.alloc(4 * (WIDE + 5)).unwrap();
    for timed in [false, true] {
        dev.time_launches(timed);
        let (mut a, mut b) = tables(7);
        let want = expect(&a, &b);
        let graph = dev.capture(|| {
            launch(&dev, &k, &out, &a, &b);
            Ok(())
        })
        .unwrap();
        a = [0; WIDE];
        b = [0; 5];
        dev.replay(&graph).unwrap();
        assert_eq!(read(&dev, &out), want, "timed {timed}: the captured tables");
        // a direct launch takes the new ones
        launch(&dev, &k, &out, &a, &b);
        assert_eq!(read(&dev, &out), expect(&a, &b), "timed {timed}: a launch after the capture");
        if timed {
            assert!(!dev.take_times().is_empty(), "the timed launches have times");
        }
    }
    dev.time_launches(false);
}

/// CUDA checks a table's bytes against the kernel's parameter (a driver
/// from 12.4 on reports the sizes): the driver would read the parameter's
/// size from a smaller argument.
#[test]
#[cfg(feature = "cuda")]
#[should_panic(expected = "a table argument of 16 bytes for parameter 3, which takes 20")]
fn cuda_refuses_a_table_of_the_wrong_size() {
    let dev = Dev::new(0).unwrap();
    let k = dev.compile(SOURCE, "", "k_copy").unwrap();
    let out = dev.alloc(4 * (WIDE + 5)).unwrap();
    let (a, _) = tables(1);
    let short = [0u32; 4];
    let args = [Arg::Block(&out, 0), Arg::Bytes(bytemuck::cast_slice(&a)), Arg::U32(WIDE as u32), Arg::Bytes(bytemuck::cast_slice(&short))];
    let _ = dev.launch(&k, Launch::linear(WIDE as u32, 128), &args);
}

/// A traced launch writes a table argument down by its size.
#[test]
fn a_trace_shows_a_table_by_its_size() {
    let path = std::env::temp_dir().join(format!("arael-tables-trace-{}.jsonl", std::process::id()));
    arael_gpu::trace::start(&path).unwrap();
    let dev = arael_gpu::trace::traced(&std::sync::Arc::new(Dev::new(0).unwrap()));
    let k = dev.compile(SOURCE, "", "k_copy").unwrap();
    let out = dev.alloc(4 * (WIDE + 5)).unwrap();
    let (a, b) = tables(1);
    let args = [Arg::Block(&out, 0), Arg::Bytes(bytemuck::cast_slice(&a)), Arg::U32(WIDE as u32), Arg::Bytes(bytemuck::cast_slice(&b))];
    dev.launch(&k, Launch::linear(WIDE as u32, 128), &args).unwrap();
    dev.sync().unwrap();
    drop(out);
    arael_gpu::trace::stop();
    let text = std::fs::read_to_string(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    let launched = text.lines().find(|l| l.contains("\"e\":\"launch\"")).expect("a launch line");
    assert!(launched.contains("{\"table\":\"296 bytes\"},{\"u\":74},{\"table\":\"20 bytes\"}"), "{launched}");
}
