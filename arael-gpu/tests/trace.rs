//! The CPU device's access probe and the trace of a device's operations.
#![cfg(feature = "cpu")]

use std::sync::Arc;

use arael_gpu::cpu::CpuDevice;
use arael_gpu::{Arg, Device, Launch, Scope, Section};

/// out[i] = x[idx[i]] * scale[0]: idx read in order, x gathered, scale
/// read by every thread, out written in order.
const SOURCE: &str = r#"
KERNEL_THREADS(256) void k_pick(GLOBAL const float* RESTRICT x, GLOBAL const u32* RESTRICT idx, GLOBAL float* RESTRICT out,
                                GLOBAL const float* RESTRICT scale, u32 n) {
    u32 i = global_x();
    if (i < n) out[i] = x[idx[i]] * scale[0];
}
"#;

const N: usize = 1000;

/// Whether clang is there for the probe; says so when not.
fn has_clang() -> bool {
    let cxx = std::env::var("ARAEL_CPU_PROBE_CXX").unwrap_or_else(|_| "clang++".into());
    let has = std::process::Command::new(&cxx).arg("--version").output().is_ok_and(|o| o.status.success());
    if !has {
        eprintln!("the access probe needs clang (`{cxx}` not found): test skipped");
    }
    has
}

/// The block of the test: x (declared as records of four values), idx, out
/// and scale, each on its own 4096 bytes.
fn sections() -> Vec<Section> {
    vec![
        Section::new("x", 0, 4 * N, 4, "f32").role("F").record(4),
        Section::u32s("idx", 4096, N),
        Section::new("out", 8192, 4 * N, 4, "f32").role("F"),
        Section::new("scale", 12288, 4, 4, "f32"),
    ]
}

fn fill<D: Device>(dev: &D, block: &mut D::Block) {
    let x: Vec<f32> = (0..N).map(|i| i as f32).collect();
    // a permutation far from the identity
    let idx: Vec<u32> = (0..N as u32).map(|i| (i * 7 + 3) % N as u32).collect();
    dev.write(block, 0, arael_gpu::bytemuck::cast_slice(&x)).unwrap();
    dev.write(block, 4096, arael_gpu::bytemuck::cast_slice(&idx)).unwrap();
    dev.write(block, 12288, arael_gpu::bytemuck::cast_slice(&[2.0f32])).unwrap();
}

fn args<B>(block: &B) -> [Arg<'_, B>; 5] {
    [Arg::Block(block, 0), Arg::Block(block, 4096), Arg::Block(block, 8192), Arg::Block(block, 12288), Arg::U32(N as u32)]
}

/// The probe counts a launch's reads and writes per section: how many,
/// how many distinct elements, and whether neighbouring threads touch
/// neighbouring elements or the same one.
#[test]
fn the_probe_counts_a_launch_s_accesses_per_section() {
    if !has_clang() {
        return;
    }
    let dev = CpuDevice::with_probe(0, true).unwrap();
    let k = dev.compile(SOURCE, "", "k_pick").unwrap();
    let mut block = dev.alloc(16384).unwrap();
    fill(&dev, &mut block);
    dev.describe(&block, "test", &sections);
    let launch = Launch::linear(N as u32, 256);
    dev.launch(&k, launch, &args(&block)).unwrap();
    let mut out = vec![0f32; N];
    dev.read(&block, 8192, arael_gpu::bytemuck::cast_slice_mut(&mut out)).unwrap();
    assert!((0..N).all(|i| out[i] == 2.0 * ((i * 7 + 3) % N) as f32), "the probed kernel computes as the plain one");
    let taken = dev.take_accesses();
    assert_eq!(taken.len(), 1, "one launch ran");
    let of = |s: usize| taken[0].iter().find(|a| a.section == Some(s)).unwrap_or_else(|| panic!("section {s} was touched"));
    let n = N as u64;
    // x: gathered, every element once, no thread next to the one before it
    let x = of(0);
    assert_eq!((x.arg, x.read.n, x.read.bytes, x.read.unique, x.write.n), (0, n, 4 * n, n, 0));
    assert!(x.read.adjacent < n / 10, "{:?}", x.read);
    // idx: in order, a thread an element
    let idx = of(1);
    assert_eq!((idx.read.n, idx.read.unique, idx.read.threads, idx.read.lo, idx.read.hi, idx.read.sizes), (n, n, n, 0, 4 * n - 1, 4));
    assert_eq!(idx.read.adjacent, n - 1, "coalesced: {:?}", idx.read);
    // out: written in order, never read
    let out = of(2);
    assert_eq!((out.read.n, out.write.n, out.write.unique, out.write.adjacent), (0, n, n, n - 1));
    // scale: one element, read by every thread
    let scale = of(3);
    assert_eq!((scale.read.n, scale.read.unique, scale.read.shared), (n, 1, n - 1));
    assert_eq!(taken[0].len(), 4, "nothing outside the sections: {:?}", taken[0]);

    // a replayed graph's launches are measured each
    let graph = dev.capture(|| dev.launch(&k, launch, &args(&block)).and_then(|()| dev.launch(&k, launch, &args(&block)))).unwrap();
    assert!(dev.take_accesses().is_empty(), "a capture runs nothing");
    dev.replay(&graph).unwrap();
    let replayed = dev.take_accesses();
    assert_eq!(replayed.len(), 2);
    assert_eq!(replayed[1], taken[0], "the same launch, the same accesses");
}

/// A device without the probe measures nothing and runs the plain library.
#[test]
fn a_device_without_the_probe_measures_nothing() {
    let dev = CpuDevice::with_probe(0, false).unwrap();
    let k = dev.compile(SOURCE, "", "k_pick").unwrap();
    let mut block = dev.alloc(16384).unwrap();
    fill(&dev, &mut block);
    dev.describe(&block, "test", &|| panic!("a device that inspects nothing does not ask for the sections"));
    dev.launch(&k, Launch::linear(N as u32, 256), &args(&block)).unwrap();
    assert!(dev.take_accesses().is_empty());
}

/// A traced device writes its operations down, a line each: the block and
/// its sections, the copies, the launch with its arguments and measured
/// accesses, the scopes, the captured graph and its replay.
#[test]
fn a_traced_device_writes_its_operations_down() {
    if !has_clang() {
        return;
    }
    let path = std::env::temp_dir().join(format!("arael-trace-test-{}.jsonl", std::process::id()));
    arael_gpu::trace::start(&path).unwrap();
    assert!(arael_gpu::trace::active());
    let dev = arael_gpu::trace::traced(&Arc::new(CpuDevice::with_probe(0, true).unwrap()));
    let k = dev.compile(SOURCE, "", "k_pick").unwrap();
    let mut block = dev.alloc(16384).unwrap();
    fill(&*dev, &mut block);
    dev.describe(&block, "test", &sections);
    let launch = Launch::linear(N as u32, 256);
    {
        let _scope = Scope::new(&*dev, "pick");
        dev.launch(&k, launch, &args(&block)).unwrap();
    }
    let graph = dev.capture(|| dev.launch(&k, launch, &args(&block))).unwrap();
    dev.replay(&graph).unwrap();
    let mut out = vec![0f32; N];
    dev.read(&block, 8192, arael_gpu::bytemuck::cast_slice_mut(&mut out)).unwrap();
    drop(block);
    arael_gpu::trace::stop();
    assert!(!arael_gpu::trace::active());
    let text = std::fs::read_to_string(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    let lines: Vec<&str> = text.lines().collect();
    let kinds: Vec<&str> = lines.iter().map(|l| l.split('"').nth(3).unwrap()).collect();
    assert_eq!(
        kinds,
        ["device", "unit", "alloc", "write", "write", "write", "describe", "scope", "launch", "end", "capture", "replay", "read", "free"],
        "{text}"
    );
    assert!(lines[1].contains("\"entries\":[\"k_pick\"]") && lines[1].contains("KERNEL_THREADS(256) void k_pick"), "the unit's source: {}", lines[1]);
    assert!(lines[6].contains("\"name\":\"idx\",\"at\":4096,\"bytes\":4000,\"elem\":4,\"ty\":\"u32\""), "{}", lines[6]);
    // a section of records says how many elements a record holds, a plain one nothing
    assert!(lines[6].contains("{\"name\":\"x\",\"at\":0,\"bytes\":4000,\"elem\":4,\"ty\":\"f32\",\"role\":\"F\",\"group\":\"\",\"doc\":\"\",\"record\":4}"), "{}", lines[6]);
    assert_eq!(lines[6].matches("\"record\":").count(), 1, "{}", lines[6]);
    // the index section's words, read back and classified: a permutation
    assert!(lines[6].contains("\"content\":{\"kind\":\"other\",\"n\":1000,") && lines[6].contains("\"distinct\":1000,\"gcd\":1,\"hash\":\""), "{}", lines[6]);
    // and its first and last words kept: idx[i] = (7 i + 3) mod 1000
    assert!(lines[6].contains("\"sample\":{\"head\":[3,10,17,24,") && lines[6].contains(",982,989,996]}"), "{}", lines[6]);
    let launched = lines[8];
    assert!(launched.contains("\"k\":\"k_pick\"") && launched.contains("\"grid\":[4,1,1]") && launched.contains("\"threads\":[256,1,1]"), "{launched}");
    assert!(launched.contains("{\"u\":1000}") && launched.contains("\"at\":12288"), "the arguments: {launched}");
    // the gather of x: a thousand reads of a thousand elements
    assert!(launched.contains("\"s\":0,\"r\":{\"n\":1000,\"bytes\":4000,\"unique\":1000"), "the measured accesses: {launched}");
    assert!(lines[10].contains("\"events\":[{\"e\":\"launch\"") && lines[11].contains("\"acc\":[[{\"arg\":0,\"s\":0"), "{}\n{}", lines[10], lines[11]);
    assert!(lines[0].contains("\"timed\":false") && !text.contains("\"us\":"), "an untimed trace has no times: {}", lines[0]);

    // a timed trace (the trace file is the process's: one test, one after the other):
    // a launch has its microseconds, a replay those of its graph's launches
    arael_gpu::trace::time(true);
    arael_gpu::trace::start(&path).unwrap();
    let dev = arael_gpu::trace::traced(&Arc::new(CpuDevice::with_probe(0, false).unwrap()));
    let k = dev.compile(SOURCE, "", "k_pick").unwrap();
    let mut block = dev.alloc(16384).unwrap();
    fill(&*dev, &mut block);
    dev.launch(&k, launch, &args(&block)).unwrap();
    let graph = dev.capture(|| dev.launch(&k, launch, &args(&block)).and_then(|()| dev.launch(&k, launch, &args(&block)))).unwrap();
    dev.replay(&graph).unwrap();
    arael_gpu::trace::stop();
    arael_gpu::trace::time(false);
    let text = std::fs::read_to_string(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    let us = |line: &str| -> Vec<f64> {
        let at = line.find("\"us\":").unwrap_or_else(|| panic!("a time in {line}")) + 5;
        line[at..].trim_start_matches('[').split(|c| c == ']' || c == '}').next().unwrap().split(',').map(|v| v.parse().unwrap()).collect()
    };
    let line = |kind: &str| text.lines().find(|l| l.split('"').nth(3) == Some(kind)).unwrap_or_else(|| panic!("a {kind} line in {text}"));
    assert!(line("device").contains("\"timed\":true"), "{}", line("device"));
    let launched = us(line("launch"));
    assert!(launched.len() == 1 && launched[0] > 0.0, "{launched:?}");
    let replayed = us(line("replay"));
    assert!(replayed.len() == 2 && replayed.iter().all(|&t| t > 0.0), "{replayed:?}");
    assert!(!line("capture").contains("\"us\":"), "a capture runs nothing: {}", line("capture"));
}
