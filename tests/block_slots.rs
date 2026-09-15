// Every Hessian block carries its position in its container's array of
// the solve's block store. One walk numbers them -- the same instance can
// be reached by several constraints, and only the container walk decides
// its place -- so the numbering must come out dense and in container
// order for every shape a block can live in.
//
// The blocks still hold their own storage here; the slot is what the
// store will address them by.

use arael::model::{CrossBlock, Param, SelfBlock};
use arael::refs::{self, Ref};
use arael::simple_lm::RootProblem;

#[arael::model]
#[arael(constraint(hb, { [(node.x - node.t) * 1.0] }))]
struct Node {
    x: Param<f64>,
    t: f64,
    hb: SelfBlock<Node>,
}

// A cross constraint at root level: its own tile, plus both ends' self
// blocks reached through references.
#[arael::model]
#[arael(constraint(hb, { [(b.x - a.x - tie.d) * 1.0] }))]
struct Tie {
    #[arael(ref = root.nodes)] a: Ref<Node>,
    #[arael(ref = root.nodes)] b: Ref<Node>,
    d: f64,
    hb: CrossBlock<Node, Node>,
}

// An entity two hops down, under a grouping sub-model.
#[arael::model]
#[arael(constraint(hb, { [(leaf.x - leaf.t) * 2.0] }))]
struct Leaf {
    x: Param<f64>,
    t: f64,
    hb: SelfBlock<Leaf>,
}

#[arael::model]
struct Bag {
    leaves: std::vec::Vec<Leaf>,
}

// An entity in an arena, which can hold freed slots.
#[arael::model]
#[arael(constraint(hb, { [(cell.x - cell.t) * 0.5] }))]
struct Cell {
    x: Param<f64>,
    t: f64,
    hb: SelfBlock<Cell>,
}

#[arael::model]
#[arael(root)]
#[arael(constraint(hb, { [(w.k - 1.0) * 1.0] }))]
struct W {
    k: Param<f64>,
    nodes: refs::Vec<Node>,
    ties: std::vec::Vec<Tie>,
    bags: std::vec::Vec<Bag>,
    cells: refs::Arena<Cell>,
    hb: SelfBlock<W>,
}

fn build() -> W {
    let mut w = W {
        k: Param::new(0.0),
        nodes: refs::Vec::new(),
        ties: std::vec::Vec::new(),
        bags: std::vec::Vec::new(),
        cells: refs::Arena::new(),
        hb: SelfBlock::new(),
    };
    for i in 0..5 {
        w.nodes.push(Node { x: Param::new(i as f64), t: i as f64, hb: SelfBlock::new() });
    }
    for i in 1..5 {
        let (a, b) = (w.nodes.ref_at(i - 1), w.nodes.ref_at(i));
        w.ties.push(Tie { a, b, d: 1.0, hb: CrossBlock::new() });
    }
    // Two bags of different sizes: the numbering has to run across both.
    for n in [3usize, 2] {
        let mut leaves = std::vec::Vec::new();
        for j in 0..n {
            leaves.push(Leaf { x: Param::new(j as f64), t: j as f64, hb: SelfBlock::new() });
        }
        w.bags.push(Bag { leaves });
    }
    for i in 0..4 {
        w.cells.push(Cell { x: Param::new(i as f64), t: i as f64, hb: SelfBlock::new() });
    }
    w
}

/// The second arena entry's reference, for the hole test.
fn second_cell(w: &W) -> Ref<Cell> {
    w.cells.iter_refs().nth(1).map(|(r, _)| r).expect("four cells")
}

/// Every block in a root collection is numbered by its position there.
#[test]
fn root_collections_number_in_order() {
    let mut w = build();
    let mut params = Vec::new();
    RootProblem::<f64>::serialize(&mut w, &mut params);

    let got: Vec<u32> = w.nodes.iter().map(|n| n.hb.slot()).collect();
    assert_eq!(got, vec![0, 1, 2, 3, 4]);
    let got: Vec<u32> = w.ties.iter().map(|t| t.hb.slot()).collect();
    assert_eq!(got, vec![0, 1, 2, 3]);
}

/// A container below a grouping sub-model numbers across its parents:
/// the array is one per path, not one per parent instance.
#[test]
fn a_nested_container_numbers_across_its_parents() {
    let mut w = build();
    let mut params = Vec::new();
    RootProblem::<f64>::serialize(&mut w, &mut params);

    let got: Vec<u32> = w.bags.iter()
        .flat_map(|b| b.leaves.iter().map(|l| l.hb.slot()))
        .collect();
    assert_eq!(got, vec![0, 1, 2, 3, 4], "5 leaves over 2 bags, one numbering");
}

/// The root is a single instance, so its own block is slot 0.
#[test]
fn the_root_block_is_slot_zero() {
    let mut w = build();
    let mut params = Vec::new();
    RootProblem::<f64>::serialize(&mut w, &mut params);
    assert_eq!(w.hb.slot(), 0);
}

/// An arena numbers its live entries, not its slots: a freed slot takes
/// no room in the array.
#[test]
fn an_arena_numbers_live_entries() {
    let mut w = build();
    let second = second_cell(&w);
    w.cells.remove(second);
    let mut params = Vec::new();
    RootProblem::<f64>::serialize(&mut w, &mut params);

    let got: Vec<u32> = w.cells.iter().map(|c| c.hb.slot()).collect();
    assert_eq!(got, vec![0, 1, 2], "3 live cells of 4 slots");
}

/// Re-numbering is idempotent and follows the container: pushing an
/// instance and wiring again renumbers everything from its new place.
#[test]
fn a_rebuild_renumbers_from_the_container() {
    let mut w = build();
    let mut params = Vec::new();
    RootProblem::<f64>::serialize(&mut w, &mut params);
    assert_eq!(w.nodes.iter().map(|n| n.hb.slot()).collect::<Vec<_>>(), vec![0, 1, 2, 3, 4]);

    w.nodes.push(Node { x: Param::new(9.0), t: 9.0, hb: SelfBlock::new() });
    let mut params = Vec::new();
    RootProblem::<f64>::serialize(&mut w, &mut params);
    assert_eq!(w.nodes.iter().map(|n| n.hb.slot()).collect::<Vec<_>>(), vec![0, 1, 2, 3, 4, 5]);
}

/// No block is left unwired: the walk reaches every container that holds
/// one, which is what a missed path would show up as.
#[test]
fn no_block_is_left_unwired() {
    let mut w = build();
    let mut params = Vec::new();
    RootProblem::<f64>::serialize(&mut w, &mut params);

    let mut unwired = Vec::new();
    if w.hb.slot() == u32::MAX { unwired.push("W.hb".to_string()); }
    for (i, n) in w.nodes.iter().enumerate() {
        if n.hb.slot() == u32::MAX { unwired.push(format!("nodes[{i}].hb")); }
    }
    for (i, t) in w.ties.iter().enumerate() {
        if t.hb.slot() == u32::MAX { unwired.push(format!("ties[{i}].hb")); }
    }
    for (i, b) in w.bags.iter().enumerate() {
        for (j, l) in b.leaves.iter().enumerate() {
            if l.hb.slot() == u32::MAX { unwired.push(format!("bags[{i}].leaves[{j}].hb")); }
        }
    }
    for (i, c) in w.cells.iter().enumerate() {
        if c.hb.slot() == u32::MAX { unwired.push(format!("cells[{i}].hb")); }
    }
    assert!(unwired.is_empty(), "unwired blocks: {:?}", unwired);
}

// One entity type in two collections: the instances are distinct, so
// they share one array and one numbering that runs straight through.

#[arael::model]
#[arael(constraint(hb, { [(mark.x - mark.t) * 1.0] }))]
struct Mark {
    x: Param<f64>,
    t: f64,
    hb: SelfBlock<Mark>,
}

#[arael::model]
#[arael(root)]
struct TwoBags {
    left: refs::Vec<Mark>,
    right: refs::Vec<Mark>,
}

#[test]
fn one_type_in_two_collections_numbers_straight_through() {
    let mut w = TwoBags { left: refs::Vec::new(), right: refs::Vec::new() };
    for i in 0..3 {
        w.left.push(Mark { x: Param::new(i as f64), t: i as f64, hb: SelfBlock::new() });
    }
    for i in 0..2 {
        w.right.push(Mark { x: Param::new(i as f64), t: i as f64, hb: SelfBlock::new() });
    }
    let mut params = Vec::new();
    RootProblem::<f64>::serialize(&mut w, &mut params);

    let left: Vec<u32> = w.left.iter().map(|m| m.hb.slot()).collect();
    let right: Vec<u32> = w.right.iter().map(|m| m.hb.slot()).collect();
    assert_eq!(left, vec![0, 1, 2]);
    assert_eq!(right, vec![3, 4], "the second collection continues the first");
}
