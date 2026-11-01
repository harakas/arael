//! Are CubeCL's copies asynchronous? Times the enqueue against the
//! completion of writes and reads, several in flight on one stream, and an
//! upload on one stream against a download on another.
//!
//!   cargo run -r -p arael-gpu --features cuda --example async_copies

use std::time::Instant;

use arael_gpu::{GpuBackend, client, sync};
use cubecl::bytes::Bytes;
use cubecl_environment::stream::StreamId;

const MB: usize = 1_000_000;

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

fn main() {
    let c = client(GpuBackend::Cuda, 0).unwrap();
    let big = Bytes::from_bytes_vec((0..160 * MB).map(|i| (i % 251) as u8).collect()).shared();
    let parts: Vec<Bytes> = (0..4).map(|k| Bytes::from_bytes_vec(vec![k as u8 + 1; 40 * MB]).shared()).collect();
    let h_big = c.empty(160 * MB);
    let h_parts: Vec<_> = (0..4).map(|_| c.empty(40 * MB)).collect();
    // warm up every path once
    c.write(&h_big, big.clone());
    for (h, b) in h_parts.iter().zip(&parts) {
        c.write(h, b.clone());
    }
    let _ = c.read(h_parts.clone()).unwrap();
    sync(&c);

    for round in 0..3 {
        println!("round {round}");
        let t = Instant::now();
        c.write(&h_big, big.clone());
        let enq = ms(t);
        sync(&c);
        println!("  write 160 MB, resident pageable:    enqueue {enq:7.3} ms, complete {:7.2} ms", ms(t));

        let t = Instant::now();
        for (h, b) in h_parts.iter().zip(&parts) {
            c.write(h, b.clone());
        }
        let enq = ms(t);
        sync(&c);
        println!("  4 writes of 40 MB, back to back:    enqueue {enq:7.3} ms, complete {:7.2} ms", ms(t));

        let t = Instant::now();
        for h in &h_parts {
            let _ = c.read_one(h.clone()).unwrap();
        }
        println!("  4 reads of 40 MB, one at a time:                       complete {:7.2} ms", ms(t));

        let t = Instant::now();
        let futs: Vec<_> = h_parts.iter().map(|h| c.read_async(vec![h.clone()])).collect();
        let enq = ms(t);
        for f in futs {
            let _ = cubecl::future::block_on(f).unwrap();
        }
        println!("  4 reads of 40 MB, all in flight:    enqueue {enq:7.3} ms, complete {:7.2} ms", ms(t));

        // an upload on a second stream against a download on the first
        let mut c2 = c.clone();
        unsafe { c2.set_stream(StreamId::allocate()) };
        let h2 = c2.empty(160 * MB);
        c2.write(&h2, big.clone());
        sync(&c2);
        let t = Instant::now();
        let fut = c.read_async(vec![h_big.clone()]);
        c2.write(&h2, big.clone());
        c2.flush().unwrap();
        let _ = cubecl::future::block_on(fut).unwrap();
        sync(&c2);
        let both = ms(t);
        let t = Instant::now();
        let _ = c.read_one(h_big.clone()).unwrap();
        let down = ms(t);
        let t = Instant::now();
        c2.write(&h2, big.clone());
        sync(&c2);
        let up = ms(t);
        println!("  160 MB down (stream 1) + 160 MB up (stream 2) together {both:7.2} ms (write flushed); alone: down {down:6.2}, up {up:6.2}");

        // the same with the upload's source pinned
        let mut pinned = Bytes::from_bytes_vec((0..160 * MB).map(|i| (i % 249) as u8).collect());
        c2.staging(std::iter::once(&mut pinned), false);
        let pinned = pinned.shared();
        c2.write(&h2, pinned.clone());
        sync(&c2);
        let t = Instant::now();
        let fut = c.read_async(vec![h_big.clone()]);
        c2.write(&h2, pinned.clone());
        c2.flush().unwrap();
        let _ = cubecl::future::block_on(fut).unwrap();
        sync(&c2);
        let both = ms(t);
        let t = Instant::now();
        c2.write(&h2, pinned.clone());
        sync(&c2);
        let up = ms(t);
        println!("  same, upload from pinned memory:                   together {both:7.2} ms; alone: up {up:6.2}");
    }
}
