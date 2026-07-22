//! Type-safe indexed collections with stable Ref-based access.

use std::marker::PhantomData;
use std::fmt;
use std::ops;
use std::sync::atomic::{AtomicU32, Ordering};

// ============================================================
// Ref<T> -- typed index reference
// ============================================================

/// Bits of a [`Ref`] holding the element index. The rest hold the
/// generation.
const INDEX_BITS: u32 = 24;
/// Largest index a collection can address: 16,777,215 elements.
pub const MAX_INDEX: u32 = (1 << INDEX_BITS) - 1;
const INDEX_MASK: u32 = MAX_INDEX;
/// Number of distinct generations before the counter repeats.
const GENERATIONS: u32 = 1 << (32 - INDEX_BITS);

/// Hands out a starting generation per collection, so a `Ref` from one
/// collection is rejected by another of the same type instead of silently
/// addressing whatever sits at that index. Seeded once per process, then
/// incremented, so collections are distinct within a run and almost
/// certainly across runs.
fn next_generation_base() -> u32 {
    use std::hash::{BuildHasher, Hasher};
    static NEXT: AtomicU32 = AtomicU32::new(0);
    static SEEDED: std::sync::Once = std::sync::Once::new();
    SEEDED.call_once(|| {
        let mut h = std::collections::hash_map::RandomState::new().build_hasher();
        h.write_u32(0);
        NEXT.store(h.finish() as u32, Ordering::Relaxed);
    });
    NEXT.fetch_add(1, Ordering::Relaxed) % GENERATIONS
}

/// Typed handle to an element of a collection. `Copy`, 4 bytes: a 24-bit
/// index and an 8-bit generation packed into one `u32`.
///
/// The phantom type keeps refs into collections of different types apart at
/// compile time. The generation keeps apart what the type cannot: a ref to
/// an element that has since been removed, and a ref belonging to a
/// different collection of the same type. Using either panics rather than
/// resolving to whatever now occupies the index.
///
/// The generation is 8 bits, so it repeats after 256 invalidations of the
/// same slot; a ref that survives that many is not detected. Collections
/// address at most [`MAX_INDEX`] + 1 elements.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Ref<T>(u32, PhantomData<T>);

impl<T> Clone for Ref<T> {
    fn clone(&self) -> Self { *self }
}

impl<T> Copy for Ref<T> {}

// Index and generation both take part: two refs with the same index but
// different generations denote different elements, one of them stale.
impl<T> PartialEq for Ref<T> {
    fn eq(&self, other: &Self) -> bool { self.0 == other.0 }
}

impl<T> Eq for Ref<T> {}

impl<T> std::hash::Hash for Ref<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) { self.0.hash(state); }
}

impl<T> Ref<T> {
    /// Packs an index and a generation. Crate-internal on purpose: a ref is
    /// only meaningful against the collection that issued it, so it is
    /// handed out by `push`/`alloc`/`ref_at` and never forged from a bare
    /// number. Keep a `Ref` (it is `Copy + Hash + Eq`) rather than storing
    /// an index and rebuilding one.
    pub(crate) fn new_gen(index: u32, generation: u32) -> Self {
        debug_assert!(index <= MAX_INDEX, "index {index} exceeds MAX_INDEX");
        Ref(((generation % GENERATIONS) << INDEX_BITS) | (index & INDEX_MASK), PhantomData)
    }

    /// The element index.
    pub fn index(&self) -> u32 {
        self.0 & INDEX_MASK
    }

    /// The generation this ref was issued at.
    pub(crate) fn generation(&self) -> u32 {
        self.0 >> INDEX_BITS
    }

    /// Panics with what the collection saw, for a ref whose generation does
    /// not match. This is a use-after-remove or a ref from another
    /// collection -- both are caller bugs, and the message has to be enough
    /// to find which.
    pub(crate) fn stale(&self, found: u32) -> ! {
        panic!("stale Ref<{}>: index {} carries generation {}, the collection holds {} \
-- the element was removed, or the ref belongs to another collection",
            std::any::type_name::<T>(), self.index(), self.generation(), found)
    }
}

impl<T> fmt::Debug for Ref<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ref({}g{})", self.index(), self.generation())
    }
}

// ============================================================
// RefIter -- iterator that yields Ref<T>
// ============================================================

/// Iterator yielding `Ref<T>` for each element in a contiguous index range.
/// Counts elements rather than comparing indices: Deque index ranges can
/// straddle the u32 wrap point (push_front on a fresh deque starts at
/// u32::MAX), where an end-index comparison would terminate immediately.
pub struct RefIter<T> {
    current: u32,
    remaining: u32,
    generation: u32,
    _marker: PhantomData<T>,
}

impl<T> Iterator for RefIter<T> {
    type Item = Ref<T>;
    fn next(&mut self) -> Option<Ref<T>> {
        if self.remaining > 0 {
            let r = Ref::new_gen(self.current, self.generation);
            self.current = self.current.wrapping_add(1);
            self.remaining -= 1;
            Some(r)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining as usize, Some(self.remaining as usize))
    }
}

impl<T> ExactSizeIterator for RefIter<T> {}

/// Iterator yielding `Ref<T>` for each element of a [`Deque`], front to
/// back. The position is 32 bits split across the ref's index and
/// generation, so it keeps counting across the index space wrapping.
pub struct DequeRefIter<T> {
    pos: u32,
    remaining: u32,
    seed: u32,
    _marker: PhantomData<T>,
}

impl<T> Iterator for DequeRefIter<T> {
    type Item = Ref<T>;
    fn next(&mut self) -> Option<Ref<T>> {
        if self.remaining == 0 { return None; }
        let pos = self.pos;
        self.pos = self.pos.wrapping_add(1) & INDEX_MASK;
        self.remaining -= 1;
        Some(Ref::new_gen(pos, self.seed))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining as usize, Some(self.remaining as usize))
    }
}

impl<T> ExactSizeIterator for DequeRefIter<T> {}

// ============================================================
// Vec<T> -- indexed vector with Ref-based access
// ============================================================

/// Indexed vector with Ref-based access. Like `std::vec::Vec` but indexed by `Ref<T>`.
///
/// Also supports plain `usize` indexing for convenience. Push returns a `Ref<T>` that
/// remains valid for the lifetime of the element.
///
/// A `Ref<T>` into a vector is a position. Removing elements (`pop`,
/// `truncate`, `clear`) leaves the refs to what remains valid -- nothing
/// moves -- and the ref to a removed element resolves out of range. What the
/// vector does NOT track is reuse: a later `push` takes the freed position,
/// and a `Ref` held across that resolves to the new element. This is the
/// contract, not an oversight; use an [`Arena`] for handles that outlive the
/// elements they name, which tracks a generation per slot.
///
/// The generation a vector stamps into its refs is fixed for its lifetime,
/// so a `Ref` from a different vector of the same type is still rejected.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Vec<T> {
    inner: std::vec::Vec<T>,
    /// Stamped into every `Ref` this vector hands out, so a ref from another
    /// vector of the same type is rejected. Fixed for the vector's life: an
    /// index freed by `pop` and handed to a later `push` is the same index,
    /// and a `Ref` to the old element resolves to the new one. Hold refs
    /// across removals in an [`Arena`], which tracks that per slot.
    generation: u32,
}

impl<T> Vec<T> {
    /// Creates an empty `Vec`.
    pub fn new() -> Self {
        Vec { inner: std::vec::Vec::new(), generation: next_generation_base() }
    }

    /// Creates an empty `Vec` with the given pre-allocated capacity.
    pub fn with_capacity(cap: usize) -> Self {
        Vec { inner: std::vec::Vec::with_capacity(cap),
              generation: next_generation_base() }
    }

    /// Wraps an existing `std::vec::Vec` into a Ref-indexed `Vec`.
    pub fn from_vec(v: std::vec::Vec<T>) -> Self {
        Vec { inner: v, generation: next_generation_base() }
    }

    /// Appends a value and returns a `Ref` to it.
    pub fn push(&mut self, val: T) -> Ref<T> {
        let idx = self.inner.len() as u32;
        assert!(idx <= MAX_INDEX, "refs::Vec cannot hold more than {} elements", MAX_INDEX);
        self.inner.push(val);
        Ref::new_gen(idx, self.generation)
    }

    /// Removes and returns the last element, or `None` if empty. The `Ref`
    /// to it leaves the range; a later `push` reuses that index and the ref
    /// then resolves to the new element (see the type docs).
    pub fn pop(&mut self) -> Option<T> {
        self.inner.pop()
    }

    /// Removes all elements. Refs resolve again as the indices are reused.
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Shortens to at most `len` elements, dropping the rest from the back.
    pub fn truncate(&mut self, len: usize) {
        self.inner.truncate(len);
    }

    /// Reserves capacity for at least `additional` more elements.
    pub fn reserve(&mut self, additional: usize) {
        self.inner.reserve(additional);
    }

    /// Returns the number of elements.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns true if the vector contains no elements.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns a reference to the first element, or `None` if empty.
    pub fn first(&self) -> Option<&T> {
        self.inner.first()
    }

    /// Returns a reference to the last element, or `None` if empty.
    pub fn last(&self) -> Option<&T> {
        self.inner.last()
    }

    /// Returns a reference to the element at `r`, or `None` if `r` is out of
    /// bounds or belongs to a different vector.
    pub fn get(&self, r: Ref<T>) -> Option<&T> {
        if r.generation() != self.generation { return None; }
        self.inner.get(r.index() as usize)
    }

    /// The mutable counterpart of [`get`](Self::get).
    pub fn get_mut(&mut self, r: Ref<T>) -> Option<&mut T> {
        if r.generation() != self.generation { return None; }
        self.inner.get_mut(r.index() as usize)
    }

    /// Returns true if `r` resolves to a live element of this vector.
    pub fn contains_ref(&self, r: Ref<T>) -> bool {
        r.generation() == self.generation && (r.index() as usize) < self.inner.len()
    }

    /// Returns an iterator over references to the elements.
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.inner.iter()
    }

    /// Returns an iterator over mutable references to the elements.
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.inner.iter_mut()
    }

    /// Returns the contents as a slice.
    pub fn as_slice(&self) -> &[T] {
        self.inner.as_slice()
    }

    /// Returns the contents as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.inner.as_mut_slice()
    }

    /// Returns an iterator yielding a `Ref<T>` for each element.
    pub fn refs(&self) -> RefIter<T> {
        RefIter { current: 0, remaining: self.inner.len() as u32,
                  generation: self.generation, _marker: PhantomData }
    }

    /// Returns an iterator yielding `(Ref<T>, &T)` for each element -- the
    /// stable handle alongside the value, so you never hand-build a `Ref`
    /// from a loop counter.
    pub fn iter_refs(&self) -> impl Iterator<Item = (Ref<T>, &T)> + '_ {
        let g = self.generation;
        self.inner.iter().enumerate().map(move |(i, v)| (Ref::new_gen(i as u32, g), v))
    }

    /// Returns an iterator yielding `(Ref<T>, &mut T)` for each element.
    pub fn iter_refs_mut(&mut self) -> impl Iterator<Item = (Ref<T>, &mut T)> + '_ {
        let g = self.generation;
        self.inner.iter_mut().enumerate().map(move |(i, v)| (Ref::new_gen(i as u32, g), v))
    }

    /// Returns the `Ref` of the `pos`-th element (panics if out of range).
    /// Turns a positional index into a stable handle instead of writing
    /// `Ref::new(pos as u32)` by hand. Accepts any integer index (`u32`,
    /// `usize`, ...) so callers never cast.
    pub fn ref_at(&self, pos: impl TryInto<usize>) -> Ref<T> {
        let pos = pos.try_into().ok().expect("ref_at: index out of usize range");
        assert!(pos < self.inner.len(),
            "ref_at: position {pos} out of range (len {})", self.inner.len());
        Ref::new_gen(pos as u32, self.generation)
    }
}

impl<T: Clone> Vec<T> {
    /// Creates a `Vec` by cloning elements from a slice.
    pub fn from_slice(s: &[T]) -> Self {
        Vec { inner: s.to_vec(), generation: next_generation_base() }
    }
}

impl<T> From<std::vec::Vec<T>> for Vec<T> {
    fn from(v: std::vec::Vec<T>) -> Self {
        Vec { inner: v, generation: next_generation_base() }
    }
}

impl<T> ops::Index<Ref<T>> for Vec<T> {
    type Output = T;
    fn index(&self, r: Ref<T>) -> &T {
        if r.generation() != self.generation { r.stale(self.generation); }
        &self.inner[r.index() as usize]
    }
}

impl<T> ops::IndexMut<Ref<T>> for Vec<T> {
    fn index_mut(&mut self, r: Ref<T>) -> &mut T {
        if r.generation() != self.generation { r.stale(self.generation); }
        &mut self.inner[r.index() as usize]
    }
}

impl<T> ops::Index<usize> for Vec<T> {
    type Output = T;
    fn index(&self, i: usize) -> &T {
        &self.inner[i]
    }
}

impl<T> ops::IndexMut<usize> for Vec<T> {
    fn index_mut(&mut self, i: usize) -> &mut T {
        &mut self.inner[i]
    }
}

impl<'a, T> IntoIterator for &'a Vec<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.inner.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut Vec<T> {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.inner.iter_mut()
    }
}

impl<T: fmt::Debug> fmt::Debug for Vec<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.fmt(f)
    }
}

// ============================================================
// Deque<T> -- double-ended queue with stable Ref indices
// ============================================================

/// Double-ended queue with Ref-based access. Like `VecDeque` but indexed by `Ref<T>`.
///
/// A `Ref<T>` stays valid as long as the element it points to is still in the
/// deque: pushing at either end, or removing a *different* element, does not
/// affect it. That is what the sliding-window pattern needs -- pop old elements
/// off the front, push new ones on the back, and every `Ref` to an element
/// still present keeps resolving. Removing the element a `Ref` points to
/// (`pop_front`, `pop_back`, `truncate`, `clear`) invalidates that `Ref` by
/// design: it resolves out of range until that position comes round again,
/// then names whatever now sits there, so prune constraints referencing
/// evicted elements. Use [`Arena`] to keep a removed handle permanently dead.
///
/// The position space is 24 bits, so a ref held across 16,777,216 pushes can
/// name a live element again. The generation a deque stamps into its refs is
/// fixed for its lifetime, so a `Ref` from a different deque is rejected.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Deque<T> {
    first_index: u32,
    inner: std::collections::VecDeque<T>,
    /// Added to the lap number when stamping a ref, so two deques disagree
    /// even at the same position.
    seed: u32,
}

impl<T> Deque<T> {
    /// Stamps a ref for a position in the index space.
    fn ref_for(&self, pos: u32) -> Ref<T> {
        Ref::new_gen(pos & INDEX_MASK, self.seed)
    }

    /// The position a ref names, or `None` if it belongs to another deque.
    fn pos_of(&self, r: Ref<T>) -> Option<u32> {
        (r.generation() == self.seed).then(|| r.index())
    }

    /// True when `pos` lies inside the live window.
    fn holds(&self, pos: u32) -> bool {
        (pos.wrapping_sub(self.first_index) & INDEX_MASK) < self.inner.len() as u32
    }

    /// Creates an empty `Deque`.
    pub fn new() -> Self {
        Deque { first_index: 0, inner: std::collections::VecDeque::new(), seed: next_generation_base() }
    }

    /// Creates an empty `Deque` with the given pre-allocated capacity.
    pub fn with_capacity(cap: usize) -> Self {
        Deque { first_index: 0, inner: std::collections::VecDeque::with_capacity(cap), seed: next_generation_base() }
    }

    /// Creates a `Deque` from a `std::vec::Vec`, with indices starting at 0.
    pub fn from_vec(v: std::vec::Vec<T>) -> Self {
        Deque { first_index: 0, inner: std::collections::VecDeque::from(v), seed: next_generation_base() }
    }

    /// Appends a value to the back and returns a `Ref` to it.
    pub fn push_back(&mut self, val: T) -> Ref<T> {
        let idx = self.first_index.wrapping_add(self.inner.len() as u32) & INDEX_MASK;
        self.inner.push_back(val);
        self.ref_for(idx)
    }

    /// Prepends a value to the front and returns a `Ref` to it.
    pub fn push_front(&mut self, val: T) -> Ref<T> {
        self.first_index = self.first_index.wrapping_sub(1) & INDEX_MASK;
        self.inner.push_front(val);
        self.ref_for(self.first_index)
    }

    /// Removes and returns the back element, or `None` if empty.
    /// Invalidates the `Ref` to it (see the type docs).
    pub fn pop_back(&mut self) -> Option<T> {
        self.inner.pop_back()
    }

    /// Removes and returns the front element, or `None` if empty. The front
    /// index advances, so `Ref`s to the remaining elements stay valid; only
    /// the `Ref` to the removed element is invalidated (see the type docs).
    pub fn pop_front(&mut self) -> Option<T> {
        let val = self.inner.pop_front();
        if val.is_some() {
            self.first_index = self.first_index.wrapping_add(1) & INDEX_MASK;
        }
        val
    }

    /// Removes all elements and resets the front index to 0, invalidating
    /// every `Ref` (see the type docs).
    pub fn clear(&mut self) {
        self.inner.clear();
        self.first_index = 0;
    }

    /// Reserves capacity for at least `additional` more elements.
    pub fn reserve(&mut self, additional: usize) {
        self.inner.reserve(additional);
    }

    /// Shortens to at most `len` elements, dropping from the back.
    /// Invalidates the `Ref`s to the dropped elements (see the type docs).
    pub fn truncate(&mut self, len: usize) {
        self.inner.truncate(len);
    }

    /// Returns the number of elements.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns true if the deque contains no elements.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns a reference to the front element, or `None` if empty.
    pub fn front(&self) -> Option<&T> {
        self.inner.front()
    }

    /// Returns a reference to the back element, or `None` if empty.
    pub fn back(&self) -> Option<&T> {
        self.inner.back()
    }

    /// Returns the `Ref` of the front element, or `None` if empty.
    pub fn front_ref(&self) -> Option<Ref<T>> {
        if self.inner.is_empty() { None }
        else { Some(self.ref_for(self.first_index)) }
    }

    /// Returns the `Ref` of the back element, or `None` if empty.
    pub fn back_ref(&self) -> Option<Ref<T>> {
        if self.inner.is_empty() { None }
        else { Some(self.ref_for(self.first_index.wrapping_add(self.inner.len() as u32 - 1) & INDEX_MASK)) }
    }

    /// Returns true if `r` refers to an element currently in the deque.
    pub fn contains_ref(&self, r: Ref<T>) -> bool {
        self.pos_of(r).is_some_and(|p| self.holds(p))
    }

    /// Returns a reference to the element at `r`, or `None` if out of range.
    pub fn get(&self, r: Ref<T>) -> Option<&T> {
        let pos = self.pos_of(r)?;
        self.inner.get((pos.wrapping_sub(self.first_index) & INDEX_MASK) as usize)
    }

    /// Returns a mutable reference to the element at `r`, or `None` if out of range.
    pub fn get_mut(&mut self, r: Ref<T>) -> Option<&mut T> {
        let pos = self.pos_of(r)?;
        let first = self.first_index;
        self.inner.get_mut((pos.wrapping_sub(first) & INDEX_MASK) as usize)
    }

    /// Returns an iterator over references to the elements, front to back.
    pub fn iter(&self) -> std::collections::vec_deque::Iter<'_, T> {
        self.inner.iter()
    }

    /// Returns an iterator over mutable references to the elements, front to back.
    pub fn iter_mut(&mut self) -> std::collections::vec_deque::IterMut<'_, T> {
        self.inner.iter_mut()
    }

    /// Returns an iterator yielding a `Ref<T>` for each element, front to back.
    pub fn refs(&self) -> DequeRefIter<T> {
        DequeRefIter { pos: self.first_index, remaining: self.inner.len() as u32,
                       seed: self.seed, _marker: PhantomData }
    }

    /// Returns an iterator yielding `(Ref<T>, &T)` for each element, front to
    /// back -- the stable handle alongside the value.
    pub fn iter_refs(&self) -> impl Iterator<Item = (Ref<T>, &T)> + '_ {
        let base = self.first_index;
        let seed = self.seed;
        self.inner.iter().enumerate().map(move |(i, v)| {
            let pos = base.wrapping_add(i as u32) & INDEX_MASK;
            (Ref::new_gen(pos, seed), v)
        })
    }

    /// Returns an iterator yielding `(Ref<T>, &mut T)` for each element, front to back.
    pub fn iter_refs_mut(&mut self) -> impl Iterator<Item = (Ref<T>, &mut T)> + '_ {
        let base = self.first_index;
        let seed = self.seed;
        self.inner.iter_mut().enumerate().map(move |(i, v)| {
            let pos = base.wrapping_add(i as u32) & INDEX_MASK;
            (Ref::new_gen(pos, seed), v)
        })
    }

    /// Returns the `Ref` of the `pos`-th element from the front (panics if
    /// out of range). Accounts for the front offset left by prior
    /// `pop_front`s, so it stays correct as a sliding window advances.
    /// Accepts any integer index (`u32`, `usize`, ...) so callers never cast.
    pub fn ref_at(&self, pos: impl TryInto<usize>) -> Ref<T> {
        let pos = pos.try_into().ok().expect("ref_at: index out of usize range");
        assert!(pos < self.inner.len(),
            "ref_at: position {pos} out of range (len {})", self.inner.len());
        self.ref_for(self.first_index.wrapping_add(pos as u32) & INDEX_MASK)
    }
}

impl<T: Clone> Deque<T> {
    /// Creates a `Deque` by cloning elements from a slice.
    pub fn from_slice(s: &[T]) -> Self {
        Deque { first_index: 0, inner: std::collections::VecDeque::from(s.to_vec()), seed: next_generation_base() }
    }
}

impl<T> ops::Index<Ref<T>> for Deque<T> {
    type Output = T;
    fn index(&self, r: Ref<T>) -> &T {
        let pos = self.pos_of(r).unwrap_or_else(|| r.stale(self.seed));
        if !self.holds(pos) { r.stale(self.seed); }
        &self.inner[(pos.wrapping_sub(self.first_index) & INDEX_MASK) as usize]
    }
}

impl<T> ops::IndexMut<Ref<T>> for Deque<T> {
    fn index_mut(&mut self, r: Ref<T>) -> &mut T {
        let pos = self.pos_of(r).unwrap_or_else(|| r.stale(self.seed));
        if !self.holds(pos) { r.stale(self.seed); }
        let first = self.first_index;
        &mut self.inner[(pos.wrapping_sub(first) & INDEX_MASK) as usize]
    }
}

impl<T> ops::Index<usize> for Deque<T> {
    type Output = T;
    /// Positional access from the front (`0` = front element), like
    /// `VecDeque`. Distinct from `Index<Ref<T>>`: a `usize` is a position
    /// that shifts as the front advances, a `Ref<T>` is a stable handle
    /// that does not.
    fn index(&self, pos: usize) -> &T {
        &self.inner[pos]
    }
}

impl<T> ops::IndexMut<usize> for Deque<T> {
    fn index_mut(&mut self, pos: usize) -> &mut T {
        &mut self.inner[pos]
    }
}

impl<'a, T> IntoIterator for &'a Deque<T> {
    type Item = &'a T;
    type IntoIter = std::collections::vec_deque::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.inner.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut Deque<T> {
    type Item = &'a mut T;
    type IntoIter = std::collections::vec_deque::IterMut<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.inner.iter_mut()
    }
}

impl<T: fmt::Debug> fmt::Debug for Deque<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Deque({}+{:?})", self.first_index, self.inner)
    }
}

// ============================================================
// Arena<T> -- stable-index arena with actual deletion
// ============================================================

/// A slot in an [`Arena`]: either holds a value, or is free and stores the
/// index of the next free slot. The free slots form an index-based linked
/// list (a "free list") -- the safe-Rust equivalent of a C pointer free list,
/// the same technique the `slab` crate uses. The links live inside the
/// vacated slots, so reclamation costs no extra memory.
enum Slot<T> {
    Occupied(T),
    Free { next: Option<u32> },
}

/// A slot and the generation it is currently at. Removing an element bumps
/// its slot's generation, so a `Ref` issued before the removal no longer
/// matches and cannot address whatever is allocated into the slot next.
struct Cell<T> {
    slot: Slot<T>,
    generation: u32,
}

/// Arena with stable `Ref`-based access: alloc/free without invalidating the
/// `Ref`s to other elements.
///
/// Freed slots are reclaimed. `remove` links the slot onto an internal free
/// list and the next `push` reuses it, so storage stays bounded by the peak
/// live count rather than the total number of inserts -- what a long-running
/// add/remove workload (a sliding-window map) needs. Reclamation is O(1) and
/// the free list is threaded through the vacated slots (an index, not a
/// pointer), so it costs no extra memory.
///
/// Reuse means a `Ref` to a removed element may, after a later `push`,
/// silently resolve to that slot's new occupant -- the same contract [`Vec`]
/// and [`Deque`] already have for reused indices. Construct with
/// [`no_reuse`](Self::no_reuse) to disable reclamation (slots then grow
/// forever, but a stale `Ref` stays dead and fails loudly) when hunting
/// use-after-remove bugs.
pub struct Arena<T> {
    slots: std::vec::Vec<Cell<T>>,
    free_head: Option<u32>,
    count: usize,
    reuse: bool,
    /// Generation a fresh slot starts at, distinct per arena so a ref from
    /// another arena of the same type is rejected.
    base: u32,
}

impl<T> Arena<T> {
    /// Creates an empty arena that reclaims freed slots (the default).
    pub fn new() -> Self {
        Arena { slots: std::vec::Vec::new(), free_head: None, count: 0, reuse: true,
                base: next_generation_base() }
    }

    /// Creates an empty arena that never reuses a freed slot: `push` always
    /// appends and a removed `Ref` stays permanently dead (`get` -> `None`,
    /// indexing panics) instead of aliasing a later element. Storage grows
    /// with every insert -- for making use-after-remove bugs fail loud while
    /// debugging, not for production.
    pub fn no_reuse() -> Self {
        Arena { slots: std::vec::Vec::new(), free_head: None, count: 0, reuse: false,
                base: next_generation_base() }
    }

    /// Creates an empty arena with room for at least `cap` slots.
    pub fn with_capacity(cap: usize) -> Self {
        Arena { slots: std::vec::Vec::with_capacity(cap), free_head: None, count: 0, reuse: true,
                base: next_generation_base() }
    }

    /// Builds an arena holding the given values, with `Ref` indices `0..n`.
    pub fn from_vec(v: std::vec::Vec<T>) -> Self {
        let count = v.len();
        let base = next_generation_base();
        Arena { slots: v.into_iter().map(|x| Cell { slot: Slot::Occupied(x), generation: base }).collect(),
                free_head: None, count, reuse: true, base }
    }

    /// Reserves capacity for at least `additional` more slots.
    pub fn reserve(&mut self, additional: usize) {
        self.slots.reserve(additional);
    }

    /// Inserts a value and returns a `Ref` to it, reusing a freed slot when
    /// one is available (unless the arena was created with
    /// [`no_reuse`](Self::no_reuse)).
    pub fn push(&mut self, val: T) -> Ref<T> {
        if self.reuse {
            if let Some(idx) = self.free_head {
                let next = match &self.slots[idx as usize].slot {
                    Slot::Free { next } => *next,
                    Slot::Occupied(_) => unreachable!("free list points at an occupied slot"),
                };
                self.free_head = next;
                self.slots[idx as usize].slot = Slot::Occupied(val);
                self.count += 1;
                return Ref::new_gen(idx, self.slots[idx as usize].generation);
            }
        }
        let idx = self.slots.len() as u32;
        assert!(self.slots.len() as u32 <= MAX_INDEX,
            "refs::Arena cannot hold more than {} slots", MAX_INDEX);
        self.slots.push(Cell { slot: Slot::Occupied(val), generation: self.base });
        self.count += 1;
        Ref::new_gen(idx, self.slots[idx as usize].generation)
    }

    /// Removes the element at `r` and returns it, or `None` if already
    /// removed. Every other `Ref` stays valid; the freed slot joins the free
    /// list and a later `push` reclaims it (see the type docs for the
    /// resulting aliasing contract, or [`no_reuse`](Self::no_reuse) to opt out).
    pub fn remove(&mut self, r: Ref<T>) -> Option<T> {
        let head = self.free_head;
        match self.slots.get_mut(r.index() as usize) {
            Some(cell) if matches!(cell.slot, Slot::Occupied(_)) => {
                cell.generation = (cell.generation + 1) % GENERATIONS;
                let old = std::mem::replace(&mut cell.slot, Slot::Free { next: head });
                self.free_head = Some(r.index());
                self.count -= 1;
                match old {
                    Slot::Occupied(v) => Some(v),
                    Slot::Free { .. } => unreachable!(),
                }
            }
            _ => None,
        }
    }

    /// Retains only the elements for which the predicate returns `true`,
    /// freeing the rest in place. Unlike `Vec::retain` this never compacts:
    /// surviving elements keep their slots, so every `Ref` to a retained
    /// element stays valid. Freed slots join the free list for reuse.
    pub fn retain(&mut self, mut f: impl FnMut(&T) -> bool) {
        for idx in 0..self.slots.len() {
            let drop = matches!(&self.slots[idx].slot, Slot::Occupied(v) if !f(v));
            if drop {
                self.slots[idx].generation = (self.slots[idx].generation + 1) % GENERATIONS;
                self.slots[idx].slot = Slot::Free { next: self.free_head };
                self.free_head = Some(idx as u32);
                self.count -= 1;
            }
        }
    }

    /// Returns true if `r` refers to a live (non-removed) element.
    pub fn contains_ref(&self, r: Ref<T>) -> bool {
        matches!(self.slots.get(r.index() as usize),
            Some(Cell { slot: Slot::Occupied(_), generation }) if *generation == r.generation())
    }

    /// Deprecated alias for [`contains_ref`](Self::contains_ref) (renamed to
    /// match `Deque::contains_ref`).
    #[deprecated(note = "renamed to `contains_ref`")]
    pub fn contains(&self, r: Ref<T>) -> bool {
        self.contains_ref(r)
    }

    /// Returns a reference to the element at `r`, or `None` if removed or out of bounds.
    pub fn get(&self, r: Ref<T>) -> Option<&T> {
        match self.slots.get(r.index() as usize) {
            Some(Cell { slot: Slot::Occupied(v), generation }) if *generation == r.generation() => Some(v),
            _ => None,
        }
    }

    /// Returns a mutable reference to the element at `r`, or `None` if removed or out of bounds.
    pub fn get_mut(&mut self, r: Ref<T>) -> Option<&mut T> {
        match self.slots.get_mut(r.index() as usize) {
            Some(Cell { slot: Slot::Occupied(v), generation }) if *generation == r.generation() => Some(v),
            _ => None,
        }
    }

    /// Returns the number of live (non-removed) elements.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Returns true if the arena contains no live elements.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Total number of slots, live plus free. With slot reuse this tracks the
    /// peak live count, not the number of inserts.
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// Removes all elements and deallocates storage.
    pub fn clear(&mut self) {
        self.slots.clear();
        self.free_head = None;
        self.count = 0;
    }

    /// Returns an iterator over references to live elements, skipping freed slots.
    pub fn iter(&self) -> ArenaIter<'_, T> {
        ArenaIter { inner: self.slots.iter() }
    }

    /// Returns an iterator over mutable references to live elements, skipping freed slots.
    pub fn iter_mut(&mut self) -> ArenaIterMut<'_, T> {
        ArenaIterMut { inner: self.slots.iter_mut() }
    }

    /// Returns an iterator yielding `Ref<T>` for each live element, skipping freed slots.
    pub fn refs(&self) -> ArenaRefIter<'_, T> {
        ArenaRefIter { current: 0, slots: &self.slots, _marker: PhantomData }
    }

    /// Returns an iterator yielding `(Ref<T>, &T)` for each live element,
    /// skipping freed slots -- the stable handle alongside the value.
    pub fn iter_refs(&self) -> impl Iterator<Item = (Ref<T>, &T)> + '_ {
        self.slots.iter().enumerate().filter_map(|(i, s)| match s {
            Cell { slot: Slot::Occupied(v), generation } => Some((Ref::new_gen(i as u32, *generation), v)),
            _ => None,
        })
    }

    /// Returns an iterator yielding `(Ref<T>, &mut T)` for each live element.
    pub fn iter_refs_mut(&mut self) -> impl Iterator<Item = (Ref<T>, &mut T)> + '_ {
        self.slots.iter_mut().enumerate().filter_map(|(i, s)| match s {
            Cell { slot: Slot::Occupied(v), generation } => Some((Ref::new_gen(i as u32, *generation), v)),
            _ => None,
        })
    }

}

impl<T: serde::Serialize> serde::Serialize for Arena<T> {
    fn serialize<S: serde::Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        // Wire format is a sequence of `(Option<T>, generation)` -- Some for
        // live, None for free -- so the free-list wiring never leaks out, but
        // the generations do: refs stored alongside the arena have to keep
        // resolving after a round trip.
        ser.collect_seq(self.slots.iter().map(|cell| match &cell.slot {
            Slot::Occupied(v) => (Some(v), cell.generation),
            Slot::Free { .. } => (None, cell.generation),
        }))
    }
}

impl<'de, T: serde::Deserialize<'de>> serde::Deserialize<'de> for Arena<T> {
    fn deserialize<D: serde::Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        let opts: std::vec::Vec<(Option<T>, u32)> = serde::Deserialize::deserialize(de)?;
        // Generations come back as saved, so refs serialized with the arena
        // still resolve. A fresh base would invalidate every one of them.
        let base = next_generation_base();
        let mut slots = std::vec::Vec::with_capacity(opts.len());
        let mut free_head = None;
        for (o, generation) in opts {
            match o {
                Some(v) => slots.push(Cell { slot: Slot::Occupied(v), generation }),
                None => {
                    slots.push(Cell { slot: Slot::Free { next: free_head }, generation });
                    free_head = Some((slots.len() - 1) as u32);
                }
            }
        }
        let count = slots.iter().filter(|c| matches!(c.slot, Slot::Occupied(_))).count();
        Ok(Arena { slots, free_head, count, reuse: true, base })
    }
}

impl<T> ops::Index<Ref<T>> for Arena<T> {
    type Output = T;
    fn index(&self, r: Ref<T>) -> &T {
        let cell = &self.slots[r.index() as usize];
        if cell.generation != r.generation() { r.stale(cell.generation); }
        match &cell.slot {
            Slot::Occupied(v) => v,
            Slot::Free { .. } => panic!("Arena: accessing removed slot"),
        }
    }
}

impl<T> ops::IndexMut<Ref<T>> for Arena<T> {
    fn index_mut(&mut self, r: Ref<T>) -> &mut T {
        let cell = &mut self.slots[r.index() as usize];
        if cell.generation != r.generation() { r.stale(cell.generation); }
        match &mut cell.slot {
            Slot::Occupied(v) => v,
            Slot::Free { .. } => panic!("Arena: accessing removed slot"),
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for Arena<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let live: std::vec::Vec<&T> = self.iter().collect();
        write!(f, "Arena({}/{}){:?}", self.count, self.slots.len(), live)
    }
}

impl<'a, T> IntoIterator for &'a Arena<T> {
    type Item = &'a T;
    type IntoIter = ArenaIter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut Arena<T> {
    type Item = &'a mut T;
    type IntoIter = ArenaIterMut<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<T> Default for Vec<T> {
    fn default() -> Self { Self::new() }
}

impl<T> Default for Deque<T> {
    fn default() -> Self { Self::new() }
}

impl<T> Default for Arena<T> {
    fn default() -> Self { Self::new() }
}

// ============================================================
// Arena iterators
// ============================================================

/// Iterator over references to the live elements of an [`Arena`].
pub struct ArenaIter<'a, T> {
    inner: std::slice::Iter<'a, Cell<T>>,
}

impl<'a, T> Iterator for ArenaIter<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<&'a T> {
        for cell in self.inner.by_ref() {
            if let Slot::Occupied(v) = &cell.slot {
                return Some(v);
            }
        }
        None
    }
}

/// Iterator over mutable references to the live elements of an [`Arena`].
pub struct ArenaIterMut<'a, T> {
    inner: std::slice::IterMut<'a, Cell<T>>,
}

impl<'a, T> Iterator for ArenaIterMut<'a, T> {
    type Item = &'a mut T;
    fn next(&mut self) -> Option<&'a mut T> {
        for cell in self.inner.by_ref() {
            if let Slot::Occupied(v) = &mut cell.slot {
                return Some(v);
            }
        }
        None
    }
}

/// Iterator yielding `Ref<T>` for each live (non-removed) slot in an [`Arena`].
pub struct ArenaRefIter<'a, T> {
    current: u32,
    slots: &'a [Cell<T>],
    _marker: PhantomData<T>,
}

impl<'a, T> Iterator for ArenaRefIter<'a, T> {
    type Item = Ref<T>;
    fn next(&mut self) -> Option<Ref<T>> {
        while (self.current as usize) < self.slots.len() {
            let idx = self.current;
            self.current += 1;
            if matches!(self.slots[idx as usize].slot, Slot::Occupied(_)) {
                return Some(Ref::new_gen(idx, self.slots[idx as usize].generation));
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.slots.len() - self.current as usize))
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- Ref --

    #[test]
    fn test_ref_basics() {
        let r: Ref<i32> = Ref::new_gen(42, 7);
        assert_eq!(r.index(), 42);
        assert_eq!(r.generation(), 7);
        assert_eq!(r, Ref::new_gen(42, 7));
        // Same slot, different generation: a different element, one stale.
        assert_ne!(r, Ref::new_gen(42, 8));
        assert_ne!(r, Ref::new_gen(43, 7));
        assert_eq!(format!("{:?}", r), "Ref(42g7)");
        assert_eq!(Ref::<i32>::new_gen(MAX_INDEX, 255).index(), MAX_INDEX);
    }

    // -- Vec --

    #[test]
    fn test_vec_push_and_index() {
        let mut v: Vec<&str> = Vec::new();
        let r0 = v.push("hello");
        let r1 = v.push("world");
        assert_eq!(v[r0], "hello");
        assert_eq!(v[r1], "world");
        assert_eq!(v.len(), 2);
    }

    #[test]
    fn test_vec_from_vec() {
        let v = Vec::from_vec(vec![10, 20, 30]);
        assert_eq!(v.len(), 3);
        assert_eq!(v[0], 10);
        assert_eq!(v[2], 30);
    }

    #[test]
    fn test_vec_from_slice() {
        let v = Vec::from_slice(&[1.0, 2.0, 3.0]);
        assert_eq!(v.len(), 3);
        assert_eq!(*v.first().unwrap(), 1.0);
        assert_eq!(*v.last().unwrap(), 3.0);
    }

    #[test]
    fn test_vec_from_trait() {
        let v: Vec<i32> = vec![1, 2, 3].into();
        assert_eq!(v.len(), 3);
    }

    #[test]
    fn test_vec_pop() {
        let mut v: Vec<i32> = Vec::new();
        v.push(1);
        v.push(2);
        assert_eq!(v.pop(), Some(2));
        assert_eq!(v.pop(), Some(1));
        assert_eq!(v.pop(), None);
    }

    #[test]
    fn test_vec_get() {
        let mut v: Vec<i32> = Vec::new();
        let r = v.push(42);
        assert_eq!(v.get(r), Some(&42));
        assert_eq!(v.get(Ref::new_gen(99, r.generation())), None);
        *v.get_mut(r).unwrap() = 100;
        assert_eq!(v[r], 100);
    }

    #[test]
    fn test_vec_clear_and_truncate() {
        let mut v = Vec::from_vec(vec![1, 2, 3, 4, 5]);
        v.truncate(3);
        assert_eq!(v.len(), 3);
        v.clear();
        assert!(v.is_empty());
    }

    #[test]
    fn test_vec_iter() {
        let v = Vec::from_vec(vec![10, 20, 30]);
        let sum: i32 = v.iter().sum();
        assert_eq!(sum, 60);
    }

    #[test]
    fn test_vec_iter_mut() {
        let mut v = Vec::from_vec(vec![1, 2, 3]);
        for x in v.iter_mut() { *x *= 10; }
        assert_eq!(v.as_slice(), &[10, 20, 30]);
    }

    #[test]
    fn test_vec_into_iter() {
        let v = Vec::from_vec(vec![1, 2, 3]);
        let mut sum = 0;
        for x in &v { sum += x; }
        assert_eq!(sum, 6);
    }

    // -- Deque --

    #[test]
    fn test_deque_push_back_and_index() {
        let mut d: Deque<&str> = Deque::new();
        let r0 = d.push_back("a");
        let r1 = d.push_back("b");
        let r2 = d.push_back("c");
        assert_eq!(d[r0], "a");
        assert_eq!(d[r1], "b");
        assert_eq!(d[r2], "c");
        assert_eq!(d.len(), 3);
    }

    #[test]
    fn test_deque_pop_front_preserves_refs() {
        let mut d: Deque<i32> = Deque::new();
        let _r0 = d.push_back(10);
        let r1 = d.push_back(20);
        let r2 = d.push_back(30);

        // Pop the first element — r1 and r2 should still work
        assert_eq!(d.pop_front(), Some(10));
        assert_eq!(d[r1], 20);
        assert_eq!(d[r2], 30);
        assert_eq!(d.len(), 2);
    }

    #[test]
    fn test_deque_push_front() {
        let mut d: Deque<i32> = Deque::new();
        let r0 = d.push_back(10);
        let rf = d.push_front(5);

        assert_eq!(d[rf], 5);
        assert_eq!(d[r0], 10);
        assert_eq!(*d.front().unwrap(), 5);
        assert_eq!(*d.back().unwrap(), 10);
    }

    #[test]
    fn test_deque_pop_back() {
        let mut d: Deque<i32> = Deque::new();
        let r0 = d.push_back(10);
        d.push_back(20);
        assert_eq!(d.pop_back(), Some(20));
        assert_eq!(d[r0], 10);
        assert_eq!(d.len(), 1);
    }

    #[test]
    fn test_deque_front_back_ref() {
        let mut d: Deque<i32> = Deque::new();
        assert!(d.front_ref().is_none());
        assert!(d.back_ref().is_none());

        d.push_back(10);
        d.push_back(20);
        let fr = d.front_ref().unwrap();
        let br = d.back_ref().unwrap();
        assert_eq!(d[fr], 10);
        assert_eq!(d[br], 20);
    }

    #[test]
    fn test_deque_contains_ref() {
        let mut d: Deque<i32> = Deque::new();
        let r0 = d.push_back(10);
        let r1 = d.push_back(20);
        assert!(d.contains_ref(r0));
        assert!(d.contains_ref(r1));

        d.pop_front();
        assert!(!d.contains_ref(r0));
        assert!(d.contains_ref(r1));
    }

    #[test]
    fn test_deque_get_returns_none_for_popped() {
        let mut d: Deque<i32> = Deque::new();
        let r0 = d.push_back(10);
        d.push_back(20);
        d.pop_front();
        assert_eq!(d.get(r0), None);
    }

    #[test]
    fn test_deque_from_vec() {
        let d = Deque::from_vec(vec![1, 2, 3]);
        assert_eq!(d.len(), 3);
        assert_eq!(d[0], 1);
        assert_eq!(d[2], 3);
    }

    #[test]
    fn test_deque_from_slice() {
        let d = Deque::from_slice(&[10, 20]);
        assert_eq!(d.len(), 2);
        assert_eq!(*d.front().unwrap(), 10);
    }

    #[test]
    fn test_deque_clear_resets_index() {
        let mut d: Deque<i32> = Deque::new();
        d.push_back(1);
        d.push_back(2);
        d.pop_front();
        d.clear();
        // After clear, first_index resets — new pushes start from 0
        let r = d.push_back(99);
        assert_eq!(r.index(), 0);
        assert_eq!(d[r], 99);
    }

    #[test]
    fn test_deque_iter() {
        let d = Deque::from_vec(vec![1, 2, 3]);
        let sum: i32 = d.iter().sum();
        assert_eq!(sum, 6);
    }

    #[test]
    fn test_deque_iter_mut() {
        let mut d = Deque::from_vec(vec![1, 2, 3]);
        for x in d.iter_mut() { *x *= 10; }
        assert_eq!(d[0], 10);
        assert_eq!(d[2], 30);
    }

    #[test]
    fn test_deque_sliding_window() {
        // Simulate a sliding window: push_back, pop_front, refs stay valid
        let mut d: Deque<i32> = Deque::new();
        let mut refs = std::vec::Vec::new();
        for i in 0..10 {
            refs.push(d.push_back(i));
        }
        // Pop first 5
        for _ in 0..5 {
            d.pop_front();
        }
        // Remaining refs (5..10) should still work
        for i in 5..10 {
            assert_eq!(d[refs[i as usize]], i);
        }
        // Popped refs should be invalid
        for i in 0..5 {
            assert!(!d.contains_ref(refs[i]));
        }
    }

    #[test]
    fn test_deque_push_front_and_back_interleaved() {
        let mut d: Deque<i32> = Deque::new();
        let r1 = d.push_back(10);
        let r2 = d.push_front(5);
        let r3 = d.push_back(15);
        let r4 = d.push_front(1);

        assert_eq!(d[r4], 1);
        assert_eq!(d[r2], 5);
        assert_eq!(d[r1], 10);
        assert_eq!(d[r3], 15);

        // Iteration order: front to back
        let vals: std::vec::Vec<&i32> = d.iter().collect();
        assert_eq!(vals, vec![&1, &5, &10, &15]);
    }

    #[test]
    fn test_deque_truncate() {
        let mut d = Deque::from_vec(vec![1, 2, 3, 4, 5]);
        d.truncate(3);
        assert_eq!(d.len(), 3);
        assert_eq!(*d.back().unwrap(), 3);
    }

    // -- Arena --

    #[test]
    fn test_arena_push_and_index() {
        let mut a: Arena<&str> = Arena::new();
        let r0 = a.push("hello");
        let r1 = a.push("world");
        assert_eq!(a[r0], "hello");
        assert_eq!(a[r1], "world");
        assert_eq!(a.len(), 2);
    }

    #[test]
    fn test_arena_remove_preserves_other_refs() {
        let mut a: Arena<i32> = Arena::new();
        let r0 = a.push(10);
        let r1 = a.push(20);
        let r2 = a.push(30);

        assert_eq!(a.remove(r1), Some(20));
        assert_eq!(a.len(), 2);

        // r0 and r2 still valid
        assert_eq!(a[r0], 10);
        assert_eq!(a[r2], 30);

        // r1 is gone
        assert!(!a.contains_ref(r1));
        assert_eq!(a.get(r1), None);
    }

    #[test]
    fn test_arena_refs_skips_removed() {
        let mut a: Arena<i32> = Arena::new();
        let _r0 = a.push(10);
        let r1 = a.push(20);
        let _r2 = a.push(30);

        a.remove(r1);

        let refs: std::vec::Vec<Ref<i32>> = a.refs().collect();
        assert_eq!(refs.len(), 2);
        assert_eq!(a[refs[0]], 10);
        assert_eq!(a[refs[1]], 30);
    }

    #[test]
    fn test_arena_iter_skips_removed() {
        let mut a: Arena<i32> = Arena::new();
        a.push(1);
        let r1 = a.push(2);
        a.push(3);

        a.remove(r1);

        let vals: std::vec::Vec<&i32> = a.iter().collect();
        assert_eq!(vals, vec![&1, &3]);
    }

    #[test]
    fn test_arena_double_remove() {
        let mut a: Arena<i32> = Arena::new();
        let r0 = a.push(42);
        assert_eq!(a.remove(r0), Some(42));
        assert_eq!(a.remove(r0), None);
        assert_eq!(a.len(), 0);
    }

    #[test]
    fn test_arena_push_after_remove_reuses_slot() {
        let mut a: Arena<i32> = Arena::new();
        let r0 = a.push(10);
        let r1 = a.push(20);
        a.remove(r0);

        // The freed slot is reclaimed: the next push reuses r0's index and
        // does not grow storage. The generation moved on, so the two refs
        // are not equal and the old one no longer resolves.
        let r2 = a.push(30);
        assert_eq!(r0.index(), r2.index(), "push reuses the freed slot");
        assert_ne!(r0, r2, "the reused slot is a different element");
        assert_eq!(a.get(r0), None, "the ref to the removed element is dead");
        assert_eq!(a.slot_count(), 2, "no new slot allocated");
        assert_eq!(a[r1], 20);
        assert_eq!(a[r2], 30);
        assert_eq!(a.len(), 2);
    }

    #[test]
    fn test_arena_no_reuse_mode() {
        let mut a: Arena<i32> = Arena::no_reuse();
        let r0 = a.push(10);
        a.push(20);
        a.remove(r0);

        // No reclamation: the next push appends and r0 stays permanently dead.
        let r2 = a.push(30);
        assert_ne!(r0, r2, "no_reuse: push appends a fresh slot");
        assert_eq!(a.slot_count(), 3, "no_reuse: slots grow with every insert");
        assert!(!a.contains_ref(r0));
        assert_eq!(a.get(r0), None);
        assert_eq!(a.len(), 2);
    }

    #[test]
    fn test_arena_free_list_bounds_storage() {
        // Add/remove churn (the sliding-window case) must not grow storage:
        // the single freed slot is reclaimed every iteration.
        let mut a: Arena<i32> = Arena::new();
        let mut r = a.push(0);
        for i in 1..1000 {
            a.remove(r);
            r = a.push(i);
        }
        assert_eq!(a.len(), 1);
        assert_eq!(a.slot_count(), 1, "reused the freed slot every time");
        assert_eq!(a[r], 999);
    }

    #[test]
    fn arena_serde_roundtrip_rebuilds_free_list() {
        let mut a: Arena<i32> = Arena::new();
        let r0 = a.push(10);
        let r1 = a.push(20);
        a.push(30);
        a.remove(r1); // leave a hole
        let json = serde_json::to_string(&a).unwrap();
        let mut b: Arena<i32> = serde_json::from_str(&json).unwrap();
        assert_eq!(b.len(), 2);
        assert_eq!(b[r0], 10);
        assert!(!b.contains_ref(r1));
        let live: std::vec::Vec<i32> = b.iter().copied().collect();
        assert_eq!(live, vec![10, 30]);
        // The free list survives the roundtrip: a push reclaims the hole,
        // at the next generation, so the pre-removal ref stays dead.
        let r_new = b.push(99);
        assert_eq!(r_new.index(), r1.index(), "deserialized arena reuses the freed slot");
        assert_ne!(r_new, r1);
        assert_eq!(b.get(r1), None);
        assert_eq!(b.slot_count(), 3);
    }

    #[test]
    fn test_arena_clear() {
        let mut a: Arena<i32> = Arena::new();
        a.push(1);
        a.push(2);
        a.clear();
        assert!(a.is_empty());
        assert_eq!(a.len(), 0);
    }

    #[test]
    #[should_panic(expected = "stale Ref")]
    fn test_arena_index_removed_panics() {
        let mut a: Arena<i32> = Arena::new();
        let r = a.push(42);
        a.remove(r);
        let _ = a[r]; // should panic
    }

    // ============================================================
    // Deque border cases: index arithmetic near the u32 wrap point.
    // push_front decrements first_index with wrapping_sub, so the very
    // first push_front on a fresh deque already places the live index
    // range at u32::MAX. Every accessor must be wrap-safe.
    // ============================================================

    #[test]
    fn deque_refs_after_push_front_from_empty() {
        // Regression: refs() compared `current < len` without wrap
        // handling, so a deque built with push_front yielded ZERO refs.
        let mut d: Deque<i32> = Deque::new();
        let r = d.push_front(10);
        // The position is u32::MAX; the index field holds its low 24 bits
        // and the lap rides in the generation.
        assert_eq!(r.index(), MAX_INDEX);
        let collected: std::vec::Vec<Ref<i32>> = d.refs().collect();
        assert_eq!(collected.len(), 1, "refs() must yield the single element");
        assert_eq!(collected[0], r);
        assert_eq!(d[collected[0]], 10);
    }

    #[test]
    fn deque_refs_across_index_wrap() {
        // Live range straddles the wrap point: MAX-1, MAX, 0, 1.
        let mut d: Deque<i32> = Deque::new();
        let r1 = d.push_front(1); // MAX
        let r2 = d.push_front(2); // MAX - 1
        let r3 = d.push_back(3);  // 0 (wrapped)
        let r4 = d.push_back(4);  // 1
        assert_eq!(r1.index(), MAX_INDEX);
        assert_eq!(r2.index(), MAX_INDEX - 1);
        assert_eq!(r3.index(), 0);
        assert_eq!(r4.index(), 1);

        let collected: std::vec::Vec<Ref<i32>> = d.refs().collect();
        assert_eq!(collected, vec![r2, r1, r3, r4], "front-to-back order");
        // Every yielded ref must resolve to the right element, and the
        // ref-order must agree with iter() order.
        let by_ref: std::vec::Vec<i32> = d.refs().map(|r| d[r]).collect();
        let by_iter: std::vec::Vec<i32> = d.iter().copied().collect();
        assert_eq!(by_ref, by_iter);
        assert_eq!(by_ref, vec![2, 1, 3, 4]);
    }

    #[test]
    fn deque_refs_exact_size_at_wrap() {
        let mut d: Deque<i32> = Deque::new();
        d.push_front(1);
        d.push_front(2);
        d.push_back(3);
        let it = d.refs();
        assert_eq!(it.size_hint(), (3, Some(3)));
        assert_eq!(it.len(), 3);
        let empty: Deque<i32> = Deque::new();
        assert_eq!(empty.refs().size_hint(), (0, Some(0)));
        assert_eq!(empty.refs().count(), 0);
    }

    #[test]
    fn deque_front_back_get_at_wrap() {
        let mut d: Deque<i32> = Deque::new();
        let rf = d.push_front(1);   // MAX
        let rb = d.push_back(2);    // 0
        assert_eq!(d.front_ref(), Some(rf));
        assert_eq!(d.back_ref(), Some(rb));
        assert_eq!(*d.front().unwrap(), 1);
        assert_eq!(*d.back().unwrap(), 2);
        assert!(d.contains_ref(rf) && d.contains_ref(rb));
        assert_eq!(d.get(rf), Some(&1));
        assert_eq!(d.get(rb), Some(&2));
        // A never-issued ref adjacent to the live range is not contained.
        assert!(!d.contains_ref(d.ref_for(u32::MAX - 1)));
        assert!(!d.contains_ref(d.ref_for(1)));
    }

    #[test]
    fn deque_sliding_window_across_wrap() {
        // Slide a 3-element window through the wrap point and verify
        // refs()/get() stay consistent the whole way.
        let mut d: Deque<u32> = Deque::new();
        for i in 0..3 {
            d.push_front(i); // indices MAX, MAX-1, MAX-2 (values 0,1,2)
        }
        for step in 0..6u32 {
            d.push_back(100 + step);
            let evicted = d.pop_front().unwrap();
            let _ = evicted;
            assert_eq!(d.len(), 3);
            let collected: std::vec::Vec<u32> = d.refs().map(|r| d[r]).collect();
            let by_iter: std::vec::Vec<u32> = d.iter().copied().collect();
            assert_eq!(collected, by_iter, "step {}", step);
            assert_eq!(d.front_ref().map(|r| d[r]), d.front().copied());
            assert_eq!(d.back_ref().map(|r| d[r]), d.back().copied());
        }
    }

    #[test]
    fn deque_pop_front_invalidates_then_push_front_reuses_index() {
        // Index reuse is inherent to the stable-index design: a ref to a
        // popped front element becomes valid again if push_front reuses
        // that index. Pin the behavior so a change is deliberate.
        let mut d: Deque<i32> = Deque::new();
        let r0 = d.push_back(1);
        d.push_back(2);
        assert_eq!(d.pop_front(), Some(1));
        assert!(!d.contains_ref(r0), "popped ref is stale");
        let r_new = d.push_front(99);
        assert_eq!(r_new, r0, "push_front reuses the vacated index");
        assert!(d.contains_ref(r0));
        assert_eq!(d[r0], 99);
    }

    #[test]
    fn deque_truncate_invalidates_back_refs() {
        let mut d: Deque<i32> = Deque::new();
        let r0 = d.push_back(1);
        let r1 = d.push_back(2);
        let r2 = d.push_back(3);
        d.truncate(1);
        assert!(d.contains_ref(r0));
        assert!(!d.contains_ref(r1) && !d.contains_ref(r2));
        assert_eq!(d.get(r1), None);
        assert_eq!(d.refs().count(), 1);
    }

    // -- iter_refs / iter_refs_mut / ref_at (handle-aware iteration + indexing) --

    #[test]
    fn vec_iter_refs_and_ref_at() {
        let mut v: Vec<i32> = Vec::new();
        let r0 = v.push(10);
        let r1 = v.push(20);
        let r2 = v.push(30);
        let collected: std::vec::Vec<(Ref<i32>, i32)> =
            v.iter_refs().map(|(r, &x)| (r, x)).collect();
        assert_eq!(collected, vec![(r0, 10), (r1, 20), (r2, 30)]);
        assert_eq!(v.ref_at(0), r0);
        assert_eq!(v.ref_at(2), r2);
        assert_eq!(v[v.ref_at(1)], 20);
    }

    #[test]
    fn vec_iter_refs_mut() {
        let mut v = Vec::from_vec(vec![1, 2, 3]);
        for (r, x) in v.iter_refs_mut() {
            *x += r.index() as i32; // 1+0, 2+1, 3+2
        }
        assert_eq!(v[0], 1);
        assert_eq!(v[1], 3);
        assert_eq!(v[2], 5);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn vec_ref_at_out_of_range_panics() {
        let v = Vec::from_vec(vec![1, 2]);
        let _ = v.ref_at(2);
    }

    #[test]
    fn deque_iter_refs_and_ref_at_respect_front_offset() {
        let mut d: Deque<i32> = Deque::new();
        d.push_back(1);
        let r1 = d.push_back(2);
        let r2 = d.push_back(3);
        d.pop_front(); // drop the first; the front offset is now 1
        let collected: std::vec::Vec<(Ref<i32>, i32)> =
            d.iter_refs().map(|(r, &x)| (r, x)).collect();
        assert_eq!(collected, vec![(r1, 2), (r2, 3)]);
        // ref_at counts from the current front, not from original push order.
        assert_eq!(d.ref_at(0), r1);
        assert_eq!(d.ref_at(1), r2);
        assert_eq!(d[d.ref_at(0)], 2);
    }

    #[test]
    fn deque_usize_index_is_positional_ref_index_is_stable() {
        let mut d: Deque<i32> = Deque::new();
        d.push_back(10);
        let r1 = d.push_back(20);
        d.push_back(30);
        d.pop_front(); // front advances: position 0 is now the element `20`
        // usize indexes by position from the front...
        assert_eq!(d[0], 20);
        assert_eq!(d[1], 30);
        // ...while the Ref still resolves to the same element it always did.
        assert_eq!(d[r1], 20);
    }

    #[test]
    fn arena_retain_keeps_surviving_refs_valid() {
        let mut a: Arena<i32> = Arena::new();
        let r0 = a.push(10);
        let r1 = a.push(20);
        let r2 = a.push(30);
        a.retain(|&v| v != 20); // drop the middle element in place
        assert_eq!(a.len(), 2);
        // Survivors keep their slots and handles -- no compaction.
        assert_eq!(a[r0], 10);
        assert_eq!(a[r2], 30);
        assert!(a.contains_ref(r0) && a.contains_ref(r2));
        assert!(!a.contains_ref(r1));
        assert_eq!(a.get(r1), None);
        // `for x in &arena` iterates live elements.
        let seen: std::vec::Vec<i32> = (&a).into_iter().copied().collect();
        assert_eq!(seen, vec![10, 30]);
    }

    #[test]
    fn deque_serde_roundtrip_preserves_front_offset() {
        let mut d: Deque<i32> = Deque::new();
        d.push_back(1);
        let r1 = d.push_back(2);
        d.push_back(3);
        d.pop_front(); // advance the front so first_index != 0
        let json = serde_json::to_string(&d).unwrap();
        let back: Deque<i32> = serde_json::from_str(&json).unwrap();
        assert_eq!(back.len(), 2);
        // The front offset survives, so the original Ref still resolves.
        assert_eq!(back[r1], 2);
        assert_eq!(back.ref_at(0), r1);
        let vals: std::vec::Vec<i32> = back.iter().copied().collect();
        assert_eq!(vals, vec![2, 3]);
    }

    #[test]
    fn containers_default_is_empty() {
        assert!(Vec::<i32>::default().is_empty());
        assert!(Deque::<i32>::default().is_empty());
        assert!(Arena::<i32>::default().is_empty());
    }

    #[test]
    fn arena_iter_refs_and_ref_at_skip_removed() {
        let mut a: Arena<i32> = Arena::new();
        let r0 = a.push(10);
        let r1 = a.push(20);
        let r2 = a.push(30);
        a.remove(r1); // leave a hole in the middle
        let collected: std::vec::Vec<(Ref<i32>, i32)> =
            a.iter_refs().map(|(r, &x)| (r, x)).collect();
        assert_eq!(collected, vec![(r0, 10), (r2, 30)]);
        // Iteration skips the removed slot and yields the live refs in order.
        let live: std::vec::Vec<Ref<i32>> = a.refs().collect();
        assert_eq!(live, vec![r0, r2]);
        for (_, x) in a.iter_refs_mut() {
            *x += 1;
        }
        assert_eq!(a[r0], 11);
        assert_eq!(a[r2], 31);
    }
}
