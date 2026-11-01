//! Uploads that read the caller's memory in place.
//!
//! CubeCL takes upload data as an owned `Bytes`, which from a slice means a
//! fresh copy: on a first touch of fresh memory the page faults alone cost
//! more than the transfer (160 MB: 57 ms to copy into a new `Vec`, 6.8 ms to
//! move over PCIe from resident memory). [`HostSlice`] lends the slice
//! instead.

use core::mem::MaybeUninit;

use cubecl::bytes::{AccessError, AccessPolicy, AllocationController, AllocationProperty, Bytes};
use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::context::{GpuContext, Lease};

/// A read-only `Bytes` controller over memory the caller owns.
struct HostSlice {
    ptr: *const u8,
    len: usize,
    align: usize,
}

// SAFETY: the controller only reads, and `write_borrowed`'s contract keeps the
// memory alive and unchanged until the device has read it.
unsafe impl Send for HostSlice {}
unsafe impl Sync for HostSlice {}

impl AllocationController for HostSlice {
    fn alloc_align(&self) -> usize {
        self.align
    }

    fn property(&self) -> AllocationProperty {
        AllocationProperty::Native
    }

    unsafe fn memory_mut(&mut self, _policy: AccessPolicy) -> Result<&mut [MaybeUninit<u8>], AccessError> {
        // Borrowed for reading only: a writer must copy, which this controller
        // never does on its own.
        Err(AccessError::WouldCopy)
    }

    fn memory(&self, _policy: AccessPolicy) -> Result<&[MaybeUninit<u8>], AccessError> {
        // SAFETY: `ptr` and `len` came from a live slice, kept alive and
        // unchanged by `write_borrowed`'s contract.
        Ok(unsafe { core::slice::from_raw_parts(self.ptr as *const MaybeUninit<u8>, self.len) })
    }
}

/// Write `src` into the device buffer `dst` in place, reading `src` where it
/// lies instead of copying it into a new allocation first.
///
/// # Safety
///
/// The device reads `src` after this returns. The caller must not modify or
/// free `src` until the client has been synchronized past this write (a
/// blocking read on the same client after it is enough: the stream is in
/// order).
pub unsafe fn write_borrowed<T: bytemuck::Pod>(client: &Client, dst: &Handle, src: &[T]) {
    if src.is_empty() {
        return;
    }
    let controller = HostSlice { ptr: src.as_ptr() as *const u8, len: core::mem::size_of_val(src), align: core::mem::align_of::<T>() };
    // SAFETY: the controller reports exactly `len` initialized bytes.
    let bytes = unsafe { Bytes::from_controller(Box::new(controller), core::mem::size_of_val(src)) };
    client.write(dst, bytes);
}

/// A device buffer of `F`, leased from a [`GpuContext`], written from host
/// values of any scalar: in place when the scalars agree, through a
/// conversion buffer kept between writes otherwise (so it is resident after
/// the first). It grows when a write is larger than its lease.
pub struct Upload<F> {
    lease: Option<Lease>,
    conv: Vec<F>,
}

impl<F: bytemuck::Pod + 'static> Default for Upload<F> {
    fn default() -> Self {
        Upload { lease: None, conv: Vec::new() }
    }
}

impl<F: bytemuck::Pod + 'static> Upload<F> {
    /// The device buffer holding the last write; written before it is read.
    pub fn handle(&self) -> &Handle {
        self.lease().handle()
    }

    /// The lease holding the last write.
    pub fn lease(&self) -> &Lease {
        self.lease.as_ref().expect("written before it is read")
    }

    /// Write `src` converted to `F` by `conv`, without waiting for the copy.
    ///
    /// # Safety
    ///
    /// As [`write_borrowed`]: the device reads `src` (or, when the scalars
    /// differ, the conversion buffer this keeps) after this returns. Until a
    /// blocking read on the context's client has returned, the caller keeps
    /// `src` alive and unchanged and makes no other write through this
    /// `Upload`.
    pub unsafe fn write<M: 'static + Copy>(&mut self, ctx: &GpuContext, src: &[M], conv: impl Fn(M) -> F) {
        let bytes = src.len().max(1) * core::mem::size_of::<F>();
        if self.lease.as_ref().is_none_or(|l| l.capacity() < bytes) {
            // returned first, so the arena can grow this very buffer
            self.lease = None;
            self.lease = Some(ctx.lease(bytes));
        }
        let client = ctx.client();
        let dst = self.lease.as_ref().unwrap().handle();
        if core::any::TypeId::of::<M>() == core::any::TypeId::of::<F>() {
            // SAFETY: M is F; the caller holds the borrow until the device has read it.
            let same = unsafe { core::slice::from_raw_parts(src.as_ptr() as *const F, src.len()) };
            unsafe { write_borrowed(client, dst, same) };
        } else {
            self.conv.clear();
            self.conv.extend(src.iter().map(|&v| conv(v)));
            // SAFETY: `conv` lives in self, untouched until the caller's next
            // write, which the contract puts after a blocking read.
            unsafe { write_borrowed(client, dst, &self.conv) };
        }
    }
}

/// Synchronizes the client on drop unless [`done`](Self::done) was called:
/// held across borrowed writes, so an early return or a panic before the
/// blocking read cannot free memory the device has yet to copy.
pub(crate) struct InFlight<'a> {
    client: &'a Client,
    done: bool,
}

impl<'a> InFlight<'a> {
    pub fn new(client: &'a Client) -> Self {
        InFlight { client, done: false }
    }

    /// A blocking read on the client has returned: nothing is in flight.
    pub fn done(mut self) {
        self.done = true;
    }
}

impl Drop for InFlight<'_> {
    fn drop(&mut self) {
        if !self.done {
            crate::sync(self.client);
        }
    }
}
