// A model field named like the lowercased ROOT type, and one named like the
// constraint struct's own lowercase. Both used to be renamed mid-path at
// emission (`__item.self.x`, `__item.__item.x`) and failed to compile.
use arael::model::{Param, SelfBlock};
use arael::simple_lm::{LmConfig, LmProblem};
use arael::vect::vect3d;

#[arael::model]
#[arael(constraint(hb, {
    // `m2` is the root type's lowercase, `item` the struct's own: both are
    // plain data fields here and must stay field reads.
    [(item.p.x - item.m2.x) * 1.0,
     (item.p.y - item.m2.y) * 1.0,
     (item.p.z - item.m2.z) * 1.0,
     (item.p.x - item.item.x) * 0.5,
     (item.p.y - item.item.y) * 0.5,
     (item.p.z - item.item.z) * 0.5]
}))]
struct Item {
    p: Param<vect3d>,
    m2: vect3d,
    item: vect3d,
    hb: SelfBlock<Item>,
}

#[arael::model]
#[arael(root)]
struct M2 {
    items: arael::refs::Vec<Item>,
}

/// The two measurements disagree, so the optimum is the weighted mean
/// (weights 1 and 0.5): p* = (4 m2 + item) / 5.
#[test]
fn fields_named_like_the_root_and_struct_resolve() {
    let m2 = vect3d::new(1.0, 2.0, 3.0);
    let it = vect3d::new(-1.0, 0.5, 4.0);
    let mut items = arael::refs::Vec::new();
    items.push(Item {
        p: Param::new(vect3d::new(0.0, 0.0, 0.0)),
        m2,
        item: it,
        hb: SelfBlock::new(),
    });
    let mut m = M2 { items };
    let r = m.solve_dense(&LmConfig::conservative());
    assert!(r.status.is_success(), "{:?}", r.status);
    let expect = (m2 * 4.0 + it) * (1.0 / 5.0);
    let p = m.items[0].p.value;
    assert!((p - expect).norm() < 1e-6, "{:?} vs {:?}", p, expect);
}
