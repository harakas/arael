//! [`DiskCache`]: a runtime's compiled kernels kept on disk, in the
//! process's kernel cache ([`crate::KernelCache`]).
//!
//! An entry is one file per kernel source, named by a hash of its key (the
//! source, the target, the options and the compiler's version). It holds
//! the key, then the binary's length and checksum, then the binary: a
//! different key under the same name, or a damaged binary, is a miss, never
//! a wrong kernel.
//!
//! ```ignore
//! let cache = DiskCache::open("cuda");
//! let key = format!("sm_89\nnvrtc 12.9\n{source}");
//! let binary = match cache.as_ref().and_then(|c| c.load(key.as_bytes())) {
//!     Some(b) => b,
//!     None => compile(source)?, // then cache.store(key.as_bytes(), &binary)
//! };
//! ```

use std::io::Write as _;
use std::path::PathBuf;
use std::sync::Mutex;

use crate::cache::{KernelCache, chosen};

/// A directory of compiled kernels.
pub(crate) struct DiskCache {
    dir: PathBuf,
}

/// The first failure to write the cache, once per process, for the caller
/// to report ([`take_cache_failure`]).
static FAILURE: Mutex<Option<(bool, String)>> = Mutex::new(None);

/// The first failure to write a kernel cache entry since the last
/// call, as a message; each process reports one at most.
pub fn take_cache_failure() -> Option<String> {
    let mut f = FAILURE.lock().unwrap_or_else(|e| e.into_inner());
    match f.as_mut() {
        Some((taken, msg)) if !*taken => {
            *taken = true;
            Some(msg.clone())
        }
        _ => None,
    }
}

/// 64-bit FNV-1a from `seed`.
fn fnv1a(seed: u64, bytes: &[u8]) -> u64 {
    bytes.iter().fold(seed, |h, &b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3))
}

/// A binary's checksum in its entry.
fn checksum(binary: &[u8]) -> u64 {
    fnv1a(0xcbf2_9ce4_8422_2325, binary)
}

impl DiskCache {
    /// `runtime`'s part of the process's kernel cache; `None` when the
    /// cache is off.
    pub fn open(runtime: &str) -> Option<DiskCache> {
        match chosen() {
            KernelCache::Off => None,
            KernelCache::Dir(dir) => Some(DiskCache { dir: dir.join(runtime) }),
        }
    }

    /// A cache in `dir`, for tests.
    #[cfg(test)]
    pub fn at(dir: PathBuf) -> DiskCache {
        DiskCache { dir }
    }

    fn path(&self, key: &[u8]) -> PathBuf {
        let (a, b) = (fnv1a(0xcbf2_9ce4_8422_2325, key), fnv1a(0x6c62_272e_07bb_0142, key));
        self.dir.join(format!("{a:016x}{b:016x}.bin"))
    }

    /// The binary compiled from `key`, when it is on disk whole.
    pub fn load(&self, key: &[u8]) -> Option<Vec<u8>> {
        let bytes = std::fs::read(self.path(key)).ok()?;
        let take = |at: &mut usize, n: usize| -> Option<&[u8]> {
            let s = bytes.get(*at..at.checked_add(n)?)?;
            *at += n;
            Some(s)
        };
        let word = |at: &mut usize| -> Option<u64> { Some(u64::from_le_bytes(take(at, 8)?.try_into().ok()?)) };
        let mut at = 0;
        let key_len = usize::try_from(word(&mut at)?).ok()?;
        if take(&mut at, key_len)? != key {
            return None;
        }
        let len = usize::try_from(word(&mut at)?).ok()?;
        let sum = word(&mut at)?;
        let binary = take(&mut at, len)?;
        (at == bytes.len() && checksum(binary) == sum).then(|| binary.to_vec())
    }

    /// Keep `binary`, compiled from `key`. The entry is written to a file of
    /// this process and renamed into place, so a reader sees all of it or
    /// none. A failure is kept for [`take_cache_failure`]; the kernel works
    /// either way.
    pub fn store(&self, key: &[u8], binary: &[u8]) {
        let path = self.path(key);
        let write = || -> std::io::Result<()> {
            std::fs::create_dir_all(&self.dir)?;
            let tmp = path.with_extension(format!("{}.tmp", std::process::id()));
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(&(key.len() as u64).to_le_bytes())?;
            f.write_all(key)?;
            f.write_all(&(binary.len() as u64).to_le_bytes())?;
            f.write_all(&checksum(binary).to_le_bytes())?;
            f.write_all(binary)?;
            drop(f);
            std::fs::rename(&tmp, &path)
        };
        if let Err(e) = write() {
            let mut f = FAILURE.lock().unwrap_or_else(|e| e.into_inner());
            if f.is_none() {
                *f = Some((false, format!("the kernel cache entry {} could not be written ({e}); kernels compile again in every process", path.display())));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache() -> (DiskCache, PathBuf) {
        let dir = std::env::temp_dir().join(format!("arael-disk-cache-test-{}-{:?}", std::process::id(), std::thread::current().id()));
        let _ = std::fs::remove_dir_all(&dir);
        (DiskCache { dir: dir.join("cuda") }, dir)
    }

    #[test]
    fn an_entry_comes_back_under_its_key_only() {
        let (c, dir) = cache();
        assert_eq!(c.load(b"k1"), None);
        c.store(b"k1", b"binary one");
        c.store(b"k2", b"binary two");
        assert_eq!(c.load(b"k1").as_deref(), Some(&b"binary one"[..]));
        assert_eq!(c.load(b"k2").as_deref(), Some(&b"binary two"[..]));
        // a file under k1's name holding another key is a miss
        std::fs::copy(c.path(b"k2"), c.path(b"k1")).unwrap();
        assert_eq!(c.load(b"k1"), None);
        // as is one cut short anywhere, one with a byte more, one with a binary byte changed
        let whole = std::fs::read(c.path(b"k2")).unwrap();
        for cut in [9, whole.len() - 1] {
            std::fs::write(c.path(b"k2"), &whole[..cut]).unwrap();
            assert_eq!(c.load(b"k2"), None, "cut at {cut}");
        }
        std::fs::write(c.path(b"k2"), [&whole[..], b"x"].concat()).unwrap();
        assert_eq!(c.load(b"k2"), None);
        let mut changed = whole.clone();
        *changed.last_mut().unwrap() ^= 1;
        std::fs::write(c.path(b"k2"), changed).unwrap();
        assert_eq!(c.load(b"k2"), None);
        std::fs::write(c.path(b"k2"), whole).unwrap();
        assert_eq!(c.load(b"k2").as_deref(), Some(&b"binary two"[..]));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_failed_write_is_reported_once() {
        // a file where the directory should be
        let (_, dir) = cache();
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("cuda"), b"").unwrap();
        let c = DiskCache { dir: dir.join("cuda").join("x") };
        c.store(b"k", b"b");
        let msg = take_cache_failure().expect("a failure");
        assert!(msg.contains("could not be written"), "{msg}");
        c.store(b"k", b"b");
        assert_eq!(take_cache_failure(), None);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
