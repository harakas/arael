//! [`GpuContext`]: one stream and the device buffers that live on it.
//!
//! The solver objects ([`GpuSupernodal`](crate::GpuSupernodal),
//! [`GpuSchur`](crate::GpuSchur)) lease every buffer they use from the
//! context's arena and return it when they are dropped. The arena keeps what
//! it is given back and hands it out again: buffers grow to the largest
//! request seen and are not released, so a caller that solves problem after
//! problem through one context stops allocating once it has seen its largest.
//!
//! ```ignore
//! let ctx = GpuContext::new(GpuBackend::Cuda, 0)?;
//! for problem in problems {
//!     let mut f = GpuFactor::new(&ctx, GpuPrecision::F32, &problem.sn, problem.a.symbolic(), &Default::default());
//!     f.factorize(problem.a.vals())?;     // the buffers come from `ctx`, and go back to it on drop
//! }
//! assert_eq!(ctx.buffers_created(), /* what the largest problem needed */);
//! ```

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl_environment::stream::StreamId;

use crate::{GpuBackend, GpuError};

/// A buffer the arena owns.
struct Buf {
    handle: Handle,
    cap: usize,
}

struct Inner {
    client: Client,
    /// Buffers returned by dropped leases, ready to be leased again.
    free: Mutex<Vec<Buf>>,
    /// Bytes held by the arena, leased or free.
    reserved: AtomicUsize,
    /// Device buffers the arena has created over its life.
    created: AtomicUsize,
}

/// A client on its own stream, and an arena of device buffers on it.
///
/// Cloning shares both. Work submitted through one context runs in order on
/// its stream; solves on separate contexts run on separate streams and can
/// overlap. Sharing one context between threads is correct (leases never
/// overlap) but puts their work on one stream.
#[derive(Clone)]
pub struct GpuContext {
    inner: Arc<Inner>,
}

impl core::fmt::Debug for GpuContext {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("GpuContext")
            .field("reserved_bytes", &self.reserved_bytes())
            .field("buffers_created", &self.buffers_created())
            .finish()
    }
}

/// The granularities the arena rounds capacities to, below and above 64 KB.
const ROUND: usize = 64 * 1024;
const SMALL_ROUND: usize = 256;

impl GpuContext {
    /// A context on device `index` of `backend`, on a stream of its own.
    pub fn new(backend: GpuBackend, index: usize) -> Result<Self, GpuError> {
        let mut client = crate::client(backend, index)?;
        // SAFETY: the client is this context's alone; every handle the
        // context leases and every launch through it uses this stream.
        unsafe { client.set_stream(StreamId::allocate()) };
        Ok(GpuContext {
            inner: Arc::new(Inner { client, free: Mutex::new(Vec::new()), reserved: AtomicUsize::new(0), created: AtomicUsize::new(0) }),
        })
    }

    /// The client, pinned to the context's stream.
    pub fn client(&self) -> &Client {
        &self.inner.client
    }

    /// A buffer of at least `bytes`: the smallest free one that fits, else
    /// the largest free one grown to fit, else a new one.
    pub fn lease(&self, bytes: usize) -> Lease {
        let need = bytes.max(1);
        let mut free = self.inner.free.lock().unwrap();
        let fit = free.iter().enumerate().filter(|(_, b)| b.cap >= need).min_by_key(|(_, b)| b.cap).map(|(i, _)| i);
        let buf = match fit {
            Some(i) => free.swap_remove(i),
            None => {
                // nothing fits: the largest free buffer is replaced by a
                // bigger one, so the number of buffers stays what the
                // busiest moment needed
                if let Some(i) = free.iter().enumerate().max_by_key(|(_, b)| b.cap).map(|(i, _)| i) {
                    let old = free.swap_remove(i);
                    self.inner.reserved.fetch_sub(old.cap, Ordering::Relaxed);
                }
                // a quarter of headroom, so a size creeping up does not
                // reallocate at every step
                let round = if need < ROUND { SMALL_ROUND } else { ROUND };
                let cap = (need + need / 4).div_ceil(round) * round;
                self.inner.reserved.fetch_add(cap, Ordering::Relaxed);
                self.inner.created.fetch_add(1, Ordering::Relaxed);
                Buf { handle: self.inner.client.empty(cap), cap }
            }
        };
        Lease { buf: Some(buf), home: self.inner.clone() }
    }

    /// Whether `other` is this context (a clone of it): one stream, one arena.
    pub fn same(&self, other: &GpuContext) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// Bytes of device memory the arena holds, leased or free.
    pub fn reserved_bytes(&self) -> usize {
        self.inner.reserved.load(Ordering::Relaxed)
    }

    /// Device buffers the arena has created over its life: constant once
    /// the largest problem has been seen.
    pub fn buffers_created(&self) -> usize {
        self.inner.created.load(Ordering::Relaxed)
    }

    /// Release the free buffers back to the device (the leased ones stay).
    pub fn release_free(&self) {
        let mut free = self.inner.free.lock().unwrap();
        for b in free.drain(..) {
            self.inner.reserved.fetch_sub(b.cap, Ordering::Relaxed);
        }
    }
}

/// A buffer leased from a [`GpuContext`], returned to it on drop.
pub struct Lease {
    buf: Option<Buf>,
    home: Arc<Inner>,
}

impl Lease {
    /// The device buffer, its whole capacity.
    pub fn handle(&self) -> &Handle {
        &self.buf.as_ref().unwrap().handle
    }

    /// The buffer's capacity in bytes.
    pub fn capacity(&self) -> usize {
        self.buf.as_ref().unwrap().cap
    }

    /// The buffer's first `bytes`: what a read or a copy of the data in use
    /// should take (a read of [`handle`](Self::handle) moves the whole
    /// capacity).
    pub fn view(&self, bytes: usize) -> Handle {
        let b = self.buf.as_ref().unwrap();
        assert!(bytes <= b.cap, "a view of {bytes} bytes of a {}-byte lease", b.cap);
        b.handle.clone().offset_end((b.cap - bytes) as u64)
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        if let Some(b) = self.buf.take() {
            self.home.free.lock().unwrap().push(b);
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;

    #[test]
    fn leases_are_reused_and_grown_not_multiplied() {
        let ctx = GpuContext::new(GpuBackend::Cuda, 0).unwrap();
        let a = ctx.lease(1000);
        assert!(a.capacity() >= 1000);
        assert_eq!(ctx.buffers_created(), 1);
        drop(a);
        // a smaller request takes the returned buffer
        let b = ctx.lease(800);
        assert_eq!(ctx.buffers_created(), 1);
        drop(b);
        // a larger one replaces it: still one buffer held
        let c = ctx.lease(200_000);
        assert!(c.capacity() >= 200_000);
        assert_eq!(ctx.buffers_created(), 2);
        assert_eq!(ctx.reserved_bytes(), c.capacity());
        // two at once are two buffers
        let d = ctx.lease(100);
        assert_eq!(ctx.buffers_created(), 3);
        assert_eq!(ctx.reserved_bytes(), c.capacity() + d.capacity());
        drop((c, d));
        ctx.release_free();
        assert_eq!(ctx.reserved_bytes(), 0);
    }

    #[test]
    fn contexts_are_distinct_streams_and_clones_share() {
        let a = GpuContext::new(GpuBackend::Cuda, 0).unwrap();
        let b = GpuContext::new(GpuBackend::Cuda, 0).unwrap();
        assert!(a.same(&a.clone()) && !a.same(&b));
        let _l = a.clone().lease(10);
        assert_eq!(a.buffers_created(), 1);
        assert_eq!(b.buffers_created(), 0);
    }
}
