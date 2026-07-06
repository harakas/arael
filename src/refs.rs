//! Type-safe indexed collections with stable Ref-based access.

use std::marker::PhantomData;
use std::fmt;
use std::ops;

// ============================================================
// Ref<T> -- typed index reference
// ============================================================

/// Typed index into a collection. Copy, lightweight (u32), avoids lifetime issues.
///
/// The phantom type parameter `T` ensures refs into different collections are not
/// accidentally mixed up at compile time.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Ref<T>(u32, PhantomData<T>);

impl<T> Clone for Ref<T> {
    fn clone(&self) -> Self { *self }
}

impl<T> Copy for Ref<T> {}

impl<T> PartialEq for Ref<T> {
    fn eq(&self, other: &Self) -> bool { self.0 == other.0 }
}

impl<T> Eq for Ref<T> {}

impl<T> std::hash::Hash for Ref<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) { self.0.hash(state); }
}

impl<T> Ref<T> {
    /// Creates a new ref from a raw u32 index.
    pub fn new(index: u32) -> Self {
        Ref(index, PhantomData)
    }

    /// Returns the raw u32 index.
    pub fn index(&self) -> u32 {
        self.0
    }
}

impl<T> fmt::Debug for Ref<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ref({})", self.0)
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
    _marker: PhantomData<T>,
}

impl<T> Iterator for RefIter<T> {
    type Item = Ref<T>;
    fn next(&mut self) -> Option<Ref<T>> {
        if self.remaining > 0 {
            let r = Ref::new(self.current);
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

// ============================================================
// Vec<T> -- indexed vector with Ref-based access
// ============================================================

/// Indexed vector with Ref-based access. Like `std::vec::Vec` but indexed by `Ref<T>`.
///
/// Also supports plain `usize` indexing for convenience. Push returns a `Ref<T>` that
/// remains valid for the lifetime of the element.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Vec<T> {
    inner: std::vec::Vec<T>,
}

impl<T> Vec<T> {
    /// Creates an empty `Vec`.
    pub fn new() -> Self {
        Vec { inner: std::vec::Vec::new() }
    }

    /// Creates an empty `Vec` with the given pre-allocated capacity.
    pub fn with_capacity(cap: usize) -> Self {
        Vec { inner: std::vec::Vec::with_capacity(cap) }
    }

    /// Wraps an existing `std::vec::Vec` into a Ref-indexed `Vec`.
    pub fn from_vec(v: std::vec::Vec<T>) -> Self {
        Vec { inner: v }
    }

    /// Appends a value and returns a `Ref` to it.
    pub fn push(&mut self, val: T) -> Ref<T> {
        let idx = self.inner.len() as u32;
        self.inner.push(val);
        Ref::new(idx)
    }

    /// Removes and returns the last element, or `None` if empty.
    pub fn pop(&mut self) -> Option<T> {
        self.inner.pop()
    }

    /// Removes all elements.
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Truncates to at most `len` elements.
    ///
    /// Ref-invalidation warning: existing `Ref`s to removed elements are
    /// NOT invalidated -- resolving one panics (index out of bounds), and
    /// a later `push` REUSES the index, silently aliasing the new
    /// element. Prefer [`Arena`] where deletion is part of the workflow.
    pub fn truncate(&mut self, len: usize) {
        self.inner.truncate(len);
    }

    /// Reserves capacity for at least `additional` more elements.
    pub fn reserve(&mut self, additional: usize) {
        self.inner.reserve(additional);
    }

    /// Retains only elements for which the predicate returns true.
    ///
    /// Ref-invalidation warning: `retain` COMPACTS storage, so every
    /// element after the first removal shifts down and existing `Ref`s
    /// silently point at DIFFERENT elements -- no panic, wrong data.
    /// This is exactly the failure class the typed-Ref design exists to
    /// prevent; use [`Arena`] (stable indices, explicit free list) for
    /// models that delete, or rebuild all `Ref`s after calling this.
    pub fn retain(&mut self, f: impl FnMut(&T) -> bool) {
        self.inner.retain(f);
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

    /// Returns a reference to the element at `r`, or `None` if out of bounds.
    pub fn get(&self, r: Ref<T>) -> Option<&T> {
        self.inner.get(r.0 as usize)
    }

    /// Returns a mutable reference to the element at `r`, or `None` if out of bounds.
    pub fn get_mut(&mut self, r: Ref<T>) -> Option<&mut T> {
        self.inner.get_mut(r.0 as usize)
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
        RefIter { current: 0, remaining: self.inner.len() as u32, _marker: PhantomData }
    }
}

impl<T: Clone> Vec<T> {
    /// Creates a `Vec` by cloning elements from a slice.
    pub fn from_slice(s: &[T]) -> Self {
        Vec { inner: s.to_vec() }
    }
}

impl<T> From<std::vec::Vec<T>> for Vec<T> {
    fn from(v: std::vec::Vec<T>) -> Self {
        Vec { inner: v }
    }
}

impl<T> ops::Index<Ref<T>> for Vec<T> {
    type Output = T;
    fn index(&self, r: Ref<T>) -> &T {
        &self.inner[r.0 as usize]
    }
}

impl<T> ops::IndexMut<Ref<T>> for Vec<T> {
    fn index_mut(&mut self, r: Ref<T>) -> &mut T {
        &mut self.inner[r.0 as usize]
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
/// Refs remain valid after push_front/push_back/pop_front/pop_back operations, as
/// long as the referenced element has not been removed. Useful for sliding-window
/// patterns where old elements are popped from the front while new ones are pushed
/// to the back.
pub struct Deque<T> {
    first_index: u32,
    inner: std::collections::VecDeque<T>,
}

impl<T> Deque<T> {
    /// Creates an empty `Deque`.
    pub fn new() -> Self {
        Deque { first_index: 0, inner: std::collections::VecDeque::new() }
    }

    /// Creates an empty `Deque` with the given pre-allocated capacity.
    pub fn with_capacity(cap: usize) -> Self {
        Deque { first_index: 0, inner: std::collections::VecDeque::with_capacity(cap) }
    }

    /// Creates a `Deque` from a `std::vec::Vec`, with indices starting at 0.
    pub fn from_vec(v: std::vec::Vec<T>) -> Self {
        Deque { first_index: 0, inner: std::collections::VecDeque::from(v) }
    }

    /// Appends a value to the back and returns a `Ref` to it.
    pub fn push_back(&mut self, val: T) -> Ref<T> {
        let idx = self.first_index.wrapping_add(self.inner.len() as u32);
        self.inner.push_back(val);
        Ref::new(idx)
    }

    /// Prepends a value to the front and returns a `Ref` to it.
    pub fn push_front(&mut self, val: T) -> Ref<T> {
        self.first_index = self.first_index.wrapping_sub(1);
        self.inner.push_front(val);
        Ref::new(self.first_index)
    }

    /// Removes and returns the back element, or `None` if empty.
    pub fn pop_back(&mut self) -> Option<T> {
        self.inner.pop_back()
    }

    /// Removes and returns the front element, or `None` if empty.
    /// Existing refs to other elements remain valid.
    ///
    /// Ref-invalidation warning: `Ref`s to the popped element are NOT
    /// invalidated; the ring slot is REUSED after enough pushes, at
    /// which point a stale `Ref` silently aliases a new element.
    /// Sliding-window models must prune constraints referencing evicted
    /// entries as part of the eviction step.
    pub fn pop_front(&mut self) -> Option<T> {
        let val = self.inner.pop_front();
        if val.is_some() {
            self.first_index = self.first_index.wrapping_add(1);
        }
        val
    }

    /// Removes all elements and resets the index counter to 0.
    pub fn clear(&mut self) {
        self.inner.clear();
        self.first_index = 0;
    }

    /// Reserves capacity for at least `additional` more elements.
    pub fn reserve(&mut self, additional: usize) {
        self.inner.reserve(additional);
    }

    /// Truncates to at most `len` elements, removing from the back.
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
        else { Some(Ref::new(self.first_index)) }
    }

    /// Returns the `Ref` of the back element, or `None` if empty.
    pub fn back_ref(&self) -> Option<Ref<T>> {
        if self.inner.is_empty() { None }
        else { Some(Ref::new(self.first_index.wrapping_add(self.inner.len() as u32 - 1))) }
    }

    /// Returns true if `r` refers to an element currently in the deque.
    pub fn contains_ref(&self, r: Ref<T>) -> bool {
        let offset = r.0.wrapping_sub(self.first_index);
        (offset as usize) < self.inner.len()
    }

    /// Returns a reference to the element at `r`, or `None` if out of range.
    pub fn get(&self, r: Ref<T>) -> Option<&T> {
        let offset = r.0.wrapping_sub(self.first_index) as usize;
        self.inner.get(offset)
    }

    /// Returns a mutable reference to the element at `r`, or `None` if out of range.
    pub fn get_mut(&mut self, r: Ref<T>) -> Option<&mut T> {
        let offset = r.0.wrapping_sub(self.first_index) as usize;
        self.inner.get_mut(offset)
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
    pub fn refs(&self) -> RefIter<T> {
        RefIter { current: self.first_index, remaining: self.inner.len() as u32, _marker: PhantomData }
    }
}

impl<T: Clone> Deque<T> {
    /// Creates a `Deque` by cloning elements from a slice.
    pub fn from_slice(s: &[T]) -> Self {
        Deque { first_index: 0, inner: std::collections::VecDeque::from(s.to_vec()) }
    }
}

impl<T> ops::Index<Ref<T>> for Deque<T> {
    type Output = T;
    fn index(&self, r: Ref<T>) -> &T {
        let offset = r.0.wrapping_sub(self.first_index) as usize;
        &self.inner[offset]
    }
}

impl<T> ops::IndexMut<Ref<T>> for Deque<T> {
    fn index_mut(&mut self, r: Ref<T>) -> &mut T {
        let offset = r.0.wrapping_sub(self.first_index) as usize;
        &mut self.inner[offset]
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

/// Arena with stable Ref-based access. Supports alloc/free without invalidating
/// other refs.
///
/// Removed slots become `None` internally; new pushes always append rather than
/// reusing freed slots. Iteration and `refs()` skip removed entries.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Arena<T> {
    slots: std::vec::Vec<Option<T>>,
    count: usize,
}

impl<T> Arena<T> {
    /// Creates an empty arena.
    pub fn new() -> Self {
        Arena { slots: std::vec::Vec::new(), count: 0 }
    }

    /// Inserts a value and returns a `Ref` to it. Always appends a new slot.
    pub fn push(&mut self, val: T) -> Ref<T> {
        let idx = self.slots.len() as u32;
        self.slots.push(Some(val));
        self.count += 1;
        Ref::new(idx)
    }

    /// Removes the element at `r` and returns it, or `None` if already removed.
    /// Other refs remain valid.
    pub fn remove(&mut self, r: Ref<T>) -> Option<T> {
        let val = self.slots.get_mut(r.0 as usize)?.take();
        if val.is_some() { self.count -= 1; }
        val
    }

    /// Returns true if `r` refers to a live (non-removed) element.
    pub fn contains(&self, r: Ref<T>) -> bool {
        self.slots.get(r.0 as usize).and_then(|s| s.as_ref()).is_some()
    }

    /// Returns a reference to the element at `r`, or `None` if removed or out of bounds.
    pub fn get(&self, r: Ref<T>) -> Option<&T> {
        self.slots.get(r.0 as usize)?.as_ref()
    }

    /// Returns a mutable reference to the element at `r`, or `None` if removed or out of bounds.
    pub fn get_mut(&mut self, r: Ref<T>) -> Option<&mut T> {
        self.slots.get_mut(r.0 as usize)?.as_mut()
    }

    /// Returns the number of live (non-removed) elements.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Returns true if the arena contains no live elements.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Total number of slots (including removed). The next push will get index slot_count().
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// Removes all elements and deallocates storage.
    pub fn clear(&mut self) {
        self.slots.clear();
        self.count = 0;
    }

    /// Returns an iterator over references to live elements, skipping removed slots.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.slots.iter().filter_map(|s| s.as_ref())
    }

    /// Returns an iterator over mutable references to live elements, skipping removed slots.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.slots.iter_mut().filter_map(|s| s.as_mut())
    }

    /// Returns an iterator yielding `Ref<T>` for each live element, skipping removed slots.
    pub fn refs(&self) -> ArenaRefIter<'_, T> {
        ArenaRefIter { current: 0, slots: &self.slots, _marker: PhantomData }
    }
}

impl<T> ops::Index<Ref<T>> for Arena<T> {
    type Output = T;
    fn index(&self, r: Ref<T>) -> &T {
        self.slots[r.0 as usize].as_ref().expect("Arena: accessing removed slot")
    }
}

impl<T> ops::IndexMut<Ref<T>> for Arena<T> {
    fn index_mut(&mut self, r: Ref<T>) -> &mut T {
        self.slots[r.0 as usize].as_mut().expect("Arena: accessing removed slot")
    }
}

impl<T: fmt::Debug> fmt::Debug for Arena<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let live: std::vec::Vec<&T> = self.iter().collect();
        write!(f, "Arena({}/{}){:?}", self.count, self.slots.len(), live)
    }
}

// ============================================================
// ArenaRefIter -- iterator that yields Ref<T> for live slots
// ============================================================

/// Iterator yielding `Ref<T>` for each live (non-removed) slot in an [`Arena`].
pub struct ArenaRefIter<'a, T> {
    current: u32,
    slots: &'a [Option<T>],
    _marker: PhantomData<T>,
}

impl<'a, T> Iterator for ArenaRefIter<'a, T> {
    type Item = Ref<T>;
    fn next(&mut self) -> Option<Ref<T>> {
        while (self.current as usize) < self.slots.len() {
            let idx = self.current;
            self.current += 1;
            if self.slots[idx as usize].is_some() {
                return Some(Ref::new(idx));
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
        let r: Ref<i32> = Ref::new(42);
        assert_eq!(r.index(), 42);
        assert_eq!(r, Ref::new(42));
        assert_ne!(r, Ref::new(43));
        assert_eq!(format!("{:?}", r), "Ref(42)");
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
        assert_eq!(v[Ref::new(2)], 30);
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
        assert_eq!(v.get(Ref::new(99)), None);
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
    fn test_vec_retain() {
        let mut v = Vec::from_vec(vec![1, 2, 3, 4, 5]);
        v.retain(|x| x % 2 == 1);
        assert_eq!(v.as_slice(), &[1, 3, 5]);
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
        assert_eq!(d[Ref::new(0)], 1);
        assert_eq!(d[Ref::new(2)], 3);
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
        assert_eq!(d[Ref::new(0)], 10);
        assert_eq!(d[Ref::new(2)], 30);
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
        assert!(!a.contains(r1));
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
    fn test_arena_push_after_remove() {
        let mut a: Arena<i32> = Arena::new();
        let r0 = a.push(10);
        let r1 = a.push(20);
        a.remove(r0);

        // New push gets a new index (does not reuse r0)
        let r2 = a.push(30);
        assert_ne!(r0, r2);
        assert_eq!(a[r1], 20);
        assert_eq!(a[r2], 30);
        assert_eq!(a.len(), 2);
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
    #[should_panic(expected = "Arena: accessing removed slot")]
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
        assert_eq!(r.index(), u32::MAX);
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
        assert_eq!(r1.index(), u32::MAX);
        assert_eq!(r2.index(), u32::MAX - 1);
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
        assert!(!d.contains_ref(Ref::new(u32::MAX - 1)));
        assert!(!d.contains_ref(Ref::new(1)));
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
}
