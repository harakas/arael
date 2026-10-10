//! The upper halves of the YMM registers, which faer's x86 matrix kernels
//! leave dirty.
//!
//! `private-gemm-x86` (0.1.22 and earlier), which faer calls for its dense
//! products on x86_64, returns from its kernels without `vzeroupper`. The
//! calling thread keeps the upper halves of the YMM registers (the 256-bit AVX
//! registers) dirty. SSE code that then runs next to VEX code -- arael's own
//! kernels next to glibc's `memcpy` -- pays a state transition at every
//! switch, large on Zen 2 and small but present on Zen 4 and Skylake-X.
//!
//! It adds up where a thread goes from a faer product straight into a tight
//! loop of its own. Two places do: a pool worker, from a task that ends in a
//! product into the next stage, and the envelope factorization's caller. faer's
//! Cholesky and solves end in its own AVX code, which clears the state, so the
//! supernodal and the scalar sparse routes need nothing.
//!
//! Remove once arael requires a `private-gemm-x86` that clears the state
//! itself.

/// Clear the upper halves of the YMM registers (`vzeroupper`) on x86_64 CPUs
/// with AVX; nothing elsewhere.
#[inline]
pub(crate) fn clear_upper_ymm() {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx") {
        // SAFETY: the CPU has AVX.
        unsafe { vzeroupper() }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn vzeroupper() {
    core::arch::x86_64::_mm256_zeroupper();
}

/// Calls [`clear_upper_ymm`] when dropped. Bound to a named variable at the
/// top of a function, it clears the state on every return, `?` included:
/// `let _clear = ClearUpperYmmOnDrop;` (not `let _ = ...`, which drops it at
/// once).
pub(crate) struct ClearUpperYmmOnDrop;

impl Drop for ClearUpperYmmOnDrop {
    #[inline]
    fn drop(&mut self) {
        clear_upper_ymm();
    }
}

/// Whether the upper halves of YMM0-15 are dirty on this thread, from XINUSE
/// (`xgetbv` with `ecx = 1`, bit 2). `None` where the CPU cannot report it.
#[cfg(test)]
pub(crate) fn upper_ymm_in_use() -> Option<bool> {
    #[cfg(target_arch = "x86_64")]
    {
        use core::arch::x86_64::{__cpuid, __cpuid_count};
        // SAFETY: cpuid is always available on x86_64; xgetbv with ecx = 1
        // only runs when the CPU reports both XSAVE enabled by the OS
        // (OSXSAVE) and that form of xgetbv.
        unsafe {
            let osxsave = __cpuid(1).ecx & (1 << 27) != 0;
            let xgetbv1 = __cpuid_count(0xd, 1).eax & (1 << 2) != 0;
            if !osxsave || !xgetbv1 {
                return None;
            }
            let (lo, _hi): (u32, u32);
            core::arch::asm!(
                "xgetbv",
                in("ecx") 1u32, out("eax") lo, out("edx") _hi,
                options(nomem, nostack, preserves_flags),
            );
            Some(lo & (1 << 2) != 0)
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        None
    }
}

/// Make the upper halves dirty, as faer's kernels do. Returns false where it
/// cannot (no AVX2).
#[cfg(test)]
pub(crate) fn dirty_upper_ymm() -> bool {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: the CPU has AVX2. All ones into ymm0, from a function
        // compiled without AVX, so no `vzeroupper` follows it.
        unsafe {
            core::arch::asm!(
                "vpcmpeqd ymm0, ymm0, ymm0",
                out("xmm0") _,
                options(nomem, nostack, preserves_flags),
            );
        }
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dirty the state, check that it reads dirty: false where this machine
    /// cannot run or observe the test.
    fn dirtied() -> bool {
        dirty_upper_ymm() && upper_ymm_in_use() == Some(true)
    }

    #[test]
    fn clear_upper_ymm_clears_a_dirty_state() {
        if !dirtied() {
            return;
        }
        clear_upper_ymm();
        assert_eq!(upper_ymm_in_use(), Some(false));
    }

    #[test]
    fn the_guard_clears_when_dropped() {
        if !dirtied() {
            return;
        }
        {
            let _clear = ClearUpperYmmOnDrop;
            assert_eq!(upper_ymm_in_use(), Some(true), "not before the drop");
        }
        assert_eq!(upper_ymm_in_use(), Some(false));
    }
}
