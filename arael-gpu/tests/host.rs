//! Pinned host memory (CUDA, else HIP, else the CPU device): asynchronous
//! copies from it land where they are aimed, and taking it to write again
//! waits for the copies still reading it.
#![cfg(any(feature = "cuda", feature = "hip", feature = "cpu", all(feature = "metal", target_vendor = "apple")))]

#[cfg(feature = "cuda")]
use arael_gpu::cuda::CudaDevice as Dev;
#[cfg(all(feature = "hip", not(feature = "cuda")))]
use arael_gpu::hip::HipDevice as Dev;
#[cfg(all(feature = "metal", target_vendor = "apple", not(any(feature = "cuda", feature = "hip"))))]
use arael_gpu::metal::MetalDevice as Dev;
#[cfg(all(feature = "cpu", not(any(feature = "cuda", feature = "hip", all(feature = "metal", target_vendor = "apple")))))]
use arael_gpu::cpu::CpuDevice as Dev;
use arael_gpu::Device;

/// Ranges of the host block copied to offsets of a device block arrive
/// there, and nowhere else.
#[test]
fn copies_from_host_memory_land_at_their_offsets() {
    let dev = Dev::new(0).unwrap();
    let mut host = dev.alloc_host(1000).unwrap();
    let bytes: Vec<u8> = (0..1000u32).map(|i| (i % 251) as u8).collect();
    dev.host_mut(&mut host).unwrap().copy_from_slice(&bytes);
    let mut block = dev.alloc(4096).unwrap();
    dev.write_from_host(&mut block, 512, &host, 100..400).unwrap();
    dev.write_from_host(&mut block, 2048, &host, 0..1000).unwrap();
    dev.write_from_host(&mut block, 0, &host, 7..7).unwrap();
    let mut back = vec![0u8; 4096];
    dev.read(&block, 0, &mut back).unwrap();
    assert!(back[..512].iter().all(|&v| v == 0));
    assert_eq!(&back[512..812], &bytes[100..400]);
    assert!(back[812..2048].iter().all(|&v| v == 0));
    assert_eq!(&back[2048..3048], &bytes[..]);
    assert!(back[3048..].iter().all(|&v| v == 0));
}

/// A copy of a large range is queued and the host block is at once taken
/// to write other bytes: the first copy still delivers what the block held
/// when it was queued.
#[test]
fn writing_host_memory_again_waits_for_the_copies_from_it() {
    let dev = Dev::new(0).unwrap();
    let n = 64 << 20;
    let mut host = dev.alloc_host(n).unwrap();
    let mut block = dev.alloc(2 * n).unwrap();
    for round in 0..3u8 {
        dev.host_mut(&mut host).unwrap().fill(1 + 2 * round);
        dev.write_from_host(&mut block, 0, &host, 0..n).unwrap();
        dev.host_mut(&mut host).unwrap().fill(2 + 2 * round);
        dev.write_from_host(&mut block, n, &host, 0..n).unwrap();
        let mut back = vec![0u8; 2 * n];
        dev.read(&block, 0, &mut back).unwrap();
        assert!(back[..n].iter().all(|&v| v == 1 + 2 * round), "round {round}: the first copy saw the second write");
        assert!(back[n..].iter().all(|&v| v == 2 + 2 * round), "round {round}");
    }
}

/// A dropped host block goes back to the device's pool and serves the next
/// request it holds, so a fresh solver's staging costs no new pinning.
#[test]
fn dropped_host_blocks_are_reused() {
    let dev = Dev::new(0).unwrap();
    let before = dev.host_allocations();
    let a = dev.alloc_host(1 << 20).unwrap();
    assert_eq!(dev.host_allocations(), before + 1);
    drop(a);
    for bytes in [1 << 20, 1 << 19, (1 << 20) + (1 << 18)] {
        let b = dev.alloc_host(bytes).unwrap();
        assert_eq!(b.len(), bytes);
        assert_eq!(dev.host_allocations(), before + 1, "{bytes} bytes");
    }
    let _big = dev.alloc_host(4 << 20).unwrap();
    assert_eq!(dev.host_allocations(), before + 2);
}
