//! A `Ref` carries the generation it was issued at, so it stops resolving
//! once it can no longer mean what it meant: the element was removed and
//! its place reused, or the ref belongs to a different collection.

use arael::refs::{self, Ref};

// ------------------------------------------------------------------ Arena
// The arena keeps a generation per slot, so a removal invalidates exactly
// the refs to that element and nothing else.

#[test]
fn arena_reuse_does_not_resurrect_the_old_ref() {
    let mut a: refs::Arena<i32> = refs::Arena::new();
    let r0 = a.push(10);
    let r1 = a.push(20);
    a.remove(r0);
    let r2 = a.push(30); // reclaims r0's slot

    assert_eq!(r0.index(), r2.index(), "the slot really was reused");
    assert_eq!(a.get(r0), None, "the ref to the removed element is dead");
    assert_eq!(a.get(r2), Some(&30));
    assert_eq!(a.get(r1), Some(&20), "an untouched element keeps its ref");
}

#[test]
#[should_panic(expected = "stale Ref")]
fn arena_indexing_a_reused_slot_panics() {
    let mut a: refs::Arena<i32> = refs::Arena::new();
    let r0 = a.push(10);
    a.remove(r0);
    a.push(30);
    let _ = a[r0];
}

#[test]
fn arena_removal_is_precise() {
    let mut a: refs::Arena<i32> = refs::Arena::new();
    let rs: Vec<_> = (0..8).map(|i| a.push(i)).collect();
    a.remove(rs[3]);
    a.push(99); // takes slot 3 back
    for (i, r) in rs.iter().enumerate() {
        if i == 3 {
            assert_eq!(a.get(*r), None, "only the removed element is dead");
        } else {
            assert_eq!(a.get(*r), Some(&(i as i32)), "element {i} still resolves");
        }
    }
}

#[test]
fn arena_retain_keeps_survivors_addressable() {
    let mut a: refs::Arena<i32> = refs::Arena::new();
    let rs: Vec<_> = (0..5).map(|i| a.push(i)).collect();
    a.retain(|v| v % 2 == 0);
    assert_eq!(a.get(rs[0]), Some(&0));
    assert_eq!(a.get(rs[1]), None);
    assert_eq!(a.get(rs[2]), Some(&2));
}

// -------------------------------------------------------------------- Vec
// A vector's generation identifies the vector, not the element: it does not
// track position reuse. Removal leaves what remains addressable, and a ref
// held across a refill resolves to whatever now occupies the position --
// the documented contract, with Arena as the answer for handles that must
// outlive their elements.

#[test]
fn vec_pop_keeps_surviving_refs() {
    let mut v: refs::Vec<i32> = refs::Vec::new();
    let a = v.push(1);
    let b = v.push(2);
    let c = v.push(3);
    v.pop(); // drops 3

    assert_eq!(v.get(a), Some(&1), "nothing moved, so nothing is invalidated");
    assert_eq!(v.get(b), Some(&2));
    assert_eq!(v.get(c), None, "the popped element's ref is out of range");
}

#[test]
fn vec_reuse_aliases_by_contract() {
    let mut v: refs::Vec<i32> = refs::Vec::new();
    let c = v.push(3);
    v.pop();
    let d = v.push(4); // takes the position back

    assert_eq!(c.index(), d.index());
    // Pinned deliberately: a vector does not track reuse, so the old ref
    // now names the new element. Use an Arena to have this caught.
    assert_eq!(v.get(c), Some(&4));
}

// ------------------------------------------------------------------ Deque
// The deque's generation carries the lap of the index space, so its refs
// survive the sliding window and its full push lifetime is unchanged.

#[test]
fn deque_pop_front_preserves_refs_to_survivors() {
    let mut d: refs::Deque<i32> = refs::Deque::new();
    let a = d.push_back(1);
    let b = d.push_back(2);
    d.pop_front(); // drops 1

    assert_eq!(d.get(a), None, "the evicted element's ref is gone");
    assert_eq!(d.get(b), Some(&2), "the sliding window keeps survivors");
}

#[test]
fn deque_keeps_working_across_index_space_wrap() {
    let mut d: refs::Deque<i32> = refs::Deque::new();
    for i in 0..4 { d.push_back(i); }
    // Slide past the end of the 24-bit position space several times: the
    // window arithmetic wraps, so the deque keeps handing out refs that
    // resolve. Refs are taken after the slide -- one held across it could
    // name a live element again, which is the documented limit.
    for i in 0..3 * (1u32 << 24) {
        d.push_back(i as i32);
        d.pop_front();
    }
    let live: Vec<Ref<i32>> = d.refs().collect();
    assert_eq!(live.len(), 4);
    for (n, r) in live.iter().enumerate() {
        assert!(d.get(*r).is_some(), "ref {n} must resolve after the wrap");
    }
}

#[test]
fn a_deque_ref_does_not_cross_deques() {
    let mut a: refs::Deque<i32> = refs::Deque::new();
    let mut b: refs::Deque<i32> = refs::Deque::new();
    let ra = a.push_back(1);
    b.push_back(1);
    assert_eq!(b.get(ra), None, "a ref belongs to the deque that issued it");
}

// ------------------------------------------------------ across collections

#[test]
fn a_ref_from_another_collection_is_rejected() {
    let mut a: refs::Vec<i32> = refs::Vec::new();
    let mut b: refs::Vec<i32> = refs::Vec::new();
    let ra = a.push(1);
    b.push(1);
    assert_eq!(b.get(ra), None, "a ref belongs to the collection that issued it");
}

#[test]
fn arena_refs_do_not_cross_arenas() {
    let mut a: refs::Arena<i32> = refs::Arena::new();
    let mut b: refs::Arena<i32> = refs::Arena::new();
    let ra = a.push(1);
    b.push(1);
    assert_eq!(b.get(ra), None);
}

#[test]
fn a_clone_shares_its_origin_refs() {
    let mut v: refs::Vec<i32> = refs::Vec::new();
    let r = v.push(7);
    let copy = v.clone();
    assert_eq!(copy.get(r), Some(&7), "a clone is the same logical data");
}

#[test]
fn serde_roundtrip_keeps_refs_resolving() {
    let mut a: refs::Arena<i32> = refs::Arena::new();
    let r0 = a.push(10);
    let r1 = a.push(20);
    a.remove(r0);
    let json = serde_json::to_string(&a).unwrap();
    let back: refs::Arena<i32> = serde_json::from_str(&json).unwrap();
    // Refs stored alongside the arena must survive the trip, and the dead
    // one must stay dead.
    assert_eq!(back.get(r1), Some(&20));
    assert_eq!(back.get(r0), None);
}
