//! A `Ref` carries the generation it was issued at, so it stops resolving
//! once it can no longer mean what it meant: the element was removed and
//! its place reused, or the ref belongs to a different collection.

use arael::model::{CrossBlock, Param, SelfBlock};
use arael::refs::{self, Ref};
use arael::vect::vect3d;

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

// ------------------------------------------------------------------ blocks
// The arena stores cells in fixed-size blocks, so growth allocates a block
// instead of copying everything into a larger run.

#[test]
fn arena_grows_past_many_blocks() {
    // Enough elements to cross several blocks whatever the block size.
    const N: usize = 200_000;
    let mut a: refs::Arena<u64> = refs::Arena::new();
    let rs: Vec<_> = (0..N as u64).map(|i| a.push(i)).collect();
    assert_eq!(a.len(), N);
    assert_eq!(a.slot_count(), N, "a block is capacity, not occupancy");
    for (i, r) in rs.iter().enumerate() {
        assert_eq!(a[*r], i as u64, "element {i} survived the growth");
    }
}

#[test]
fn arena_elements_do_not_move_when_it_grows() {
    // A ref taken before the arena grew still names the same element after,
    // across a block boundary.
    let mut a: refs::Arena<u64> = refs::Arena::new();
    let first = a.push(7);
    for i in 0..100_000u64 { a.push(i); }
    assert_eq!(a[first], 7);
}

#[test]
fn arena_reuse_and_removal_still_work_across_blocks() {
    const N: usize = 50_000;
    let mut a: refs::Arena<u64> = refs::Arena::new();
    let rs: Vec<_> = (0..N as u64).map(|i| a.push(i)).collect();
    // Remove a spread of elements, including some deep in later blocks.
    let removed: Vec<_> = (0..N).step_by(7_777).collect();
    for &i in &removed { a.remove(rs[i]); }
    assert_eq!(a.len(), N - removed.len());
    for &i in &removed {
        assert_eq!(a.get(rs[i]), None, "removed element {i} is gone");
    }
    // Refill: the freed slots come back, at a new generation.
    for _ in &removed { a.push(999); }
    assert_eq!(a.len(), N);
    assert_eq!(a.slot_count(), N, "refill reused slots rather than growing");
    for &i in &removed {
        assert_eq!(a.get(rs[i]), None, "the old ref stays dead after refill");
    }
}

#[test]
fn arena_clear_then_refill_across_blocks() {
    let mut a: refs::Arena<u64> = refs::Arena::new();
    for i in 0..30_000u64 { a.push(i); }
    a.clear();
    assert_eq!(a.len(), 0);
    let r = a.push(1);
    assert_eq!(a[r], 1);
    assert_eq!(a.len(), 1);
}

#[test]
fn block_size_is_settable_at_construction() {
    let mut a: refs::Arena<u64> = refs::Arena::with_block_size(64);
    assert_eq!(a.block_size(), 64);
    // Not a power of two: rounded up, since a block is addressed by shift
    // and mask.
    let b: refs::Arena<u64> = refs::Arena::with_block_size(100);
    assert_eq!(b.block_size(), 128);

    // A small block size just means more blocks; nothing else changes.
    let rs: Vec<_> = (0..1000u64).map(|i| a.push(i)).collect();
    assert_eq!(a.len(), 1000);
    assert_eq!(a.slot_count(), 1000);
    for (i, r) in rs.iter().enumerate() {
        assert_eq!(a[*r], i as u64);
    }
    a.remove(rs[500]);
    assert_eq!(a.get(rs[500]), None);
    assert_eq!(a.get(rs[499]), Some(&499));
}

#[test]
fn the_default_block_size_scales_with_the_element() {
    // A block targets 64 KB, so a fat element gets fewer per block than a
    // small one -- the point of sizing in bytes rather than a fixed count.
    let small: refs::Arena<u8> = refs::Arena::new();
    let fat: refs::Arena<[u64; 64]> = refs::Arena::new();
    assert!(small.block_size() > fat.block_size(),
        "small {} vs fat {}", small.block_size(), fat.block_size());
    assert!(fat.block_size() >= 8, "a block never drops below the floor");
}

#[test]
fn an_arena_clone_shares_its_origin_refs() {
    let mut a: refs::Arena<u64> = refs::Arena::new();
    let r0 = a.push(10);
    let r1 = a.push(20);
    a.remove(r0);
    let copy = a.clone();
    // Same logical data: live refs resolve against either, and the removed
    // one stays dead in both.
    assert_eq!(copy.get(r1), Some(&20));
    assert_eq!(copy.get(r0), None);
    assert_eq!(copy.len(), a.len());
}

// --------------------------------------------------------------- round trip
// A model persisted and reloaded must still resolve its refs. The
// generations travel with the collections, so a Ref stored in a constraint
// keeps naming the same element.

#[arael::model]
#[arael(constraint(hb, { [(rtpose.pos - rtpose.prior).x * 10.0] }))]
#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct RtPose {
    pos: Param<vect3d>,
    prior: vect3d,
    #[serde(skip)]
    hb: SelfBlock<RtPose>,
}

#[arael::model]
#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct RtLandmark {
    pos: Param<vect3d>,
    frines: std::vec::Vec<RtFrine>,
    #[serde(skip)]
    hb: SelfBlock<RtLandmark>,
}

#[arael::model]
#[arael(constraint(hb, parent = lm, {
    let d = lm.pos - rtpose.pos;
    [(d.x - rtfrine.dx) * 5.0, (d.y - rtfrine.dy) * 5.0, (d.z - rtfrine.dz) * 5.0]
}))]
#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct RtFrine {
    #[arael(ref = root.poses)]
    rtpose: Ref<RtPose>,
    dx: f64, dy: f64, dz: f64,
    #[serde(skip)]
    hb: CrossBlock<RtLandmark, RtPose>,
}

/// The slam demo's shape: poses in a Deque, landmarks in an Arena, and
/// observations holding refs into both.
#[arael::model]
#[arael(root)]
#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct RtPath {
    poses: refs::Deque<RtPose>,
    landmarks: refs::Arena<RtLandmark>,
}

fn build_rt_path() -> RtPath {
    let mut p = RtPath { poses: refs::Deque::new(), landmarks: refs::Arena::new() };
    let prs: Vec<_> = (0..6).map(|i| p.poses.push_back(RtPose {
        pos: Param::new(vect3d::new(i as f64, 0.2 * i as f64, 0.0)),
        prior: vect3d::new(i as f64, 0.0, 0.0),
        hb: SelfBlock::new(),
    })).collect();
    for j in 0..3 {
        let frines = (0..6).map(|i| RtFrine {
            rtpose: prs[i], dx: 1.0 + j as f64, dy: 2.0, dz: 0.5, hb: CrossBlock::new(),
        }).collect();
        p.landmarks.push(RtLandmark {
            pos: Param::new(vect3d::new(j as f64, 1.0, 0.5)),
            frines,
            hb: SelfBlock::new(),
        });
    }
    p
}

#[test]
fn a_model_solves_the_same_after_a_round_trip() {
    use arael::model::Model;
    use arael::simple_lm::{LmConfig, LmProblem};

    let mut a = build_rt_path();
    let json = serde_json::to_string(&a).unwrap();
    let mut b: RtPath = serde_json::from_str(&json).unwrap();

    let ra = a.solve_sparse(&LmConfig::default()).unwrap();
    let rb = b.solve_sparse(&LmConfig::default()).unwrap();
    assert!(ra.iterations > 1, "the model must actually solve");
    assert_eq!(ra.iterations, rb.iterations);
    assert!((ra.start_cost - rb.start_cost).abs() < 1e-12,
        "start {} vs {}", ra.start_cost, rb.start_cost);
    assert!((ra.end_cost - rb.end_cost).abs() < 1e-12,
        "end {} vs {}", ra.end_cost, rb.end_cost);
}

#[test]
fn a_removed_element_stays_removed_after_a_round_trip() {
    let mut p = build_rt_path();
    let lm: Vec<_> = p.landmarks.refs().collect();
    p.landmarks.remove(lm[1]);

    let json = serde_json::to_string(&p).unwrap();
    let back: RtPath = serde_json::from_str(&json).unwrap();

    assert_eq!(back.landmarks.len(), 2);
    assert!(back.landmarks.get(lm[1]).is_none(), "the hole survives the trip");
    assert!(back.landmarks.get(lm[0]).is_some(), "the survivors do not");
    assert!(back.landmarks.get(lm[2]).is_some());
}
