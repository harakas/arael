//! Fault injection for tests (not API): device operations counted, and one
//! of them failing, so a test can make every operation of a solve fail in
//! turn and see each failure reported. Process-wide: a test that arms it
//! runs alone in its binary.
//!
//! ```ignore
//! arael_gpu::fault::arm(0); // count only
//! solve();
//! let n = arael_gpu::fault::disarm();
//! for k in 1..=n {
//!     arael_gpu::fault::arm(k); // the k-th operation fails, once
//!     assert!(solve().is_err());
//!     arael_gpu::fault::disarm();
//! }
//! ```

use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::GpuError;

static ARMED: AtomicBool = AtomicBool::new(false);
/// Operations counted since [`arm`].
static COUNT: AtomicU64 = AtomicU64::new(0);
/// The operation that fails (1-based), 0 for none.
static FAIL_AT: AtomicU64 = AtomicU64::new(0);

/// Count the device operations from now on, the `fail_at`-th failing once
/// with [`GpuError::Driver`]; 0 counts only.
pub fn arm(fail_at: u64) {
    COUNT.store(0, Ordering::SeqCst);
    FAIL_AT.store(fail_at, Ordering::SeqCst);
    ARMED.store(true, Ordering::SeqCst);
}

/// Stop counting; the operations counted since [`arm`].
pub fn disarm() -> u64 {
    ARMED.store(false, Ordering::SeqCst);
    COUNT.load(Ordering::SeqCst)
}

/// A device operation `op` starts: its injected failure when it is the one
/// [`arm`] named. One relaxed load when unarmed.
#[inline]
pub(crate) fn check(op: &'static str) -> Result<(), GpuError> {
    if !ARMED.load(Ordering::Relaxed) {
        return Ok(());
    }
    let k = COUNT.fetch_add(1, Ordering::SeqCst) + 1;
    if k == FAIL_AT.load(Ordering::SeqCst) {
        return Err(GpuError::Driver(format!("injected failure of operation {k} ({op})")));
    }
    Ok(())
}
