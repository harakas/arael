//! [`Pool`]: the device memory a device has allocated, kept for reuse. A
//! block that is dropped goes back to its device's pool, and an allocation
//! takes the smallest free block that holds it before asking the driver.
//! A fresh block gets room to spare ([`capacity`]), so solving a structure
//! after a slightly larger or smaller one allocates nothing. The memory is
//! released when the device and its last block are gone.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Device memory of one runtime, `R` (released by its own drop), and what
/// the pool holds of it.
pub(crate) struct Pool<R> {
    /// Blocks not in use, with their sizes.
    free: Mutex<Vec<(usize, R)>>,
    /// Bytes of every block made, in use or free.
    reserved: AtomicUsize,
    /// Blocks made.
    allocations: AtomicUsize,
}

impl<R> Pool<R> {
    pub fn new() -> Self {
        Pool { free: Mutex::new(Vec::new()), reserved: AtomicUsize::new(0), allocations: AtomicUsize::new(0) }
    }

    /// The smallest free block of at least `bytes`, and its size, taken out of the pool.
    pub fn take(&self, bytes: usize) -> Option<(usize, R)> {
        let mut free = self.free.lock().unwrap_or_else(|e| e.into_inner());
        let best = free.iter().enumerate().filter(|(_, (cap, _))| *cap >= bytes).min_by_key(|(_, (cap, _))| *cap)?.0;
        Some(free.swap_remove(best))
    }

    /// A block of `cap` bytes back into the pool.
    pub fn give(&self, cap: usize, mem: R) {
        self.free.lock().unwrap_or_else(|e| e.into_inner()).push((cap, mem));
    }

    /// Counts a block of `cap` bytes just made.
    pub fn made(&self, cap: usize) {
        self.reserved.fetch_add(cap, Ordering::Relaxed);
        self.allocations.fetch_add(1, Ordering::Relaxed);
    }

    /// Blocks made over the pool's life.
    pub fn allocations(&self) -> usize {
        self.allocations.load(Ordering::Relaxed)
    }

    /// Bytes the pool holds, in use or free.
    pub fn reserved_bytes(&self) -> usize {
        self.reserved.load(Ordering::Relaxed)
    }
}

/// The size of a fresh block for a request of `bytes`: a quarter more,
/// rounded up to a size class (four per power of two, 256 bytes at least),
/// so a later request up to about 1.25 times as large fits it.
///
/// ```ignore
/// assert_eq!(capacity(1000), 1280);
/// assert_eq!(capacity(1 << 20), 1310720);
/// ```
pub(crate) fn capacity(bytes: usize) -> usize {
    let want = bytes.saturating_add(bytes / 4).max(256);
    let step = (1usize << (usize::BITS - 1 - want.leading_zeros())) / 4;
    want.next_multiple_of(step)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_class_holds_a_quarter_more_than_asked() {
        for bytes in [0, 1, 255, 256, 1000, 4096, 100_000, 1 << 20, 123_456_789] {
            let cap = capacity(bytes);
            assert!(cap >= bytes + bytes / 4 && cap >= 256, "{bytes}: {cap}");
            // a class wastes at most a quarter of its power of two
            assert!(cap <= (bytes + bytes / 4).max(256) * 5 / 4 + 1, "{bytes}: {cap}");
        }
        assert_eq!(capacity(1000), 1280);
        assert_eq!(capacity(1 << 20), 1310720);
    }

    #[test]
    fn a_request_takes_the_smallest_block_that_holds_it() {
        let pool = Pool::<&str>::new();
        for (cap, name) in [(1024, "a"), (4096, "b"), (2048, "c")] {
            pool.made(cap);
            pool.give(cap, name);
        }
        assert_eq!(pool.take(1500), Some((2048, "c")));
        assert_eq!(pool.take(1500), Some((4096, "b")));
        assert_eq!(pool.take(1500), None);
        assert_eq!(pool.take(100), Some((1024, "a")));
        assert_eq!((pool.allocations(), pool.reserved_bytes()), (3, 7168));
    }
}
