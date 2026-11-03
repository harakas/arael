//! The kernel cache: compiled kernels kept on disk between processes, so
//! only a process's first run on a machine compiles them.
//!
//! The location comes from [`set_kernel_cache`] or, without it, from the
//! environment variable `ARAEL_KERNEL_CACHE`: a directory, `off` to keep
//! nothing, or unset for the user's cache directory (`~/.cache/arael/kernels`
//! on Linux, `$XDG_CACHE_HOME` respected). It is fixed at the first GPU use
//! of the process. Kernels go to `<runtime>` there (`cuda`, `hip`), one
//! file per kernel source, found again by the source, the target and the
//! compiler's version.
//!
//! ```ignore
//! // ARAEL_KERNEL_CACHE=/var/cache/myapp ./myapp, or in the program:
//! arael_gpu::set_kernel_cache(KernelCache::Dir("/var/cache/myapp".into()));
//! ```

use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

/// Where compiled kernels are kept between processes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KernelCache {
    /// Nowhere: every process compiles its kernels.
    Off,
    /// In this directory.
    Dir(PathBuf),
}

impl KernelCache {
    /// The cache `ARAEL_KERNEL_CACHE` names.
    pub fn from_env() -> KernelCache {
        Self::parse(std::env::var("ARAEL_KERNEL_CACHE").ok().as_deref())
    }

    /// The cache a value of `ARAEL_KERNEL_CACHE` names: `off` none, a
    /// directory that one, unset or empty the default.
    pub fn parse(value: Option<&str>) -> KernelCache {
        match value {
            Some("off") => KernelCache::Off,
            Some(dir) if !dir.is_empty() => KernelCache::Dir(PathBuf::from(dir)),
            _ => Self::default_dir().map_or(KernelCache::Off, KernelCache::Dir),
        }
    }

    /// `arael/kernels` in the user's cache directory, when there is one.
    pub fn default_dir() -> Option<PathBuf> {
        use etcetera::BaseStrategy;
        etcetera::choose_base_strategy().ok().map(|s| s.cache_dir().join("arael").join("kernels"))
    }
}

/// The cache asked for before the first GPU use.
static ASKED: Mutex<Option<KernelCache>> = Mutex::new(None);
/// The cache the process uses, fixed at its first GPU use.
static CHOSEN: OnceLock<KernelCache> = OnceLock::new();

/// Choose the kernel cache for the process, in place of
/// `ARAEL_KERNEL_CACHE`. Before the first GPU use only: `false` when the
/// GPU is already in use, and the cache chosen then stays.
pub fn set_kernel_cache(cache: KernelCache) -> bool {
    if CHOSEN.get().is_some() {
        return false;
    }
    *ASKED.lock().unwrap_or_else(|e| e.into_inner()) = Some(cache);
    true
}

/// The kernel cache the process uses, chosen now when the GPU is not in
/// use yet (see [`set_kernel_cache`]).
pub fn kernel_cache() -> KernelCache {
    chosen().clone()
}

/// The cache the process uses, chosen now when the GPU is not in use yet.
pub(crate) fn chosen() -> &'static KernelCache {
    CHOSEN.get_or_init(|| ASKED.lock().unwrap_or_else(|e| e.into_inner()).take().unwrap_or_else(KernelCache::from_env))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_variable_names_a_directory_off_or_the_default() {
        assert_eq!(KernelCache::parse(Some("off")), KernelCache::Off);
        assert_eq!(KernelCache::parse(Some("/tmp/k")), KernelCache::Dir(PathBuf::from("/tmp/k")));
        let default = KernelCache::default_dir().map_or(KernelCache::Off, KernelCache::Dir);
        assert_eq!(KernelCache::parse(None), default);
        assert_eq!(KernelCache::parse(Some("")), default);
        if let KernelCache::Dir(d) = default {
            assert!(d.ends_with("arael/kernels"), "{d:?}");
        }
    }
}
