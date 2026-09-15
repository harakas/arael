// An entity's parameter span -- the `(offset, width)` pair
// `Model::collect_param_blocks` reports and the covariance queries slice
// by -- comes from the entity's own parameter slots through
// `ParamSlot`/`Model::fold_param_span`, not from a Hessian block.
//
// The span must be the smallest live index and the count of live
// components. A fixed slot takes no room in the flat parameter vector,
// so an entity's live components stay contiguous there wherever the
// fixed ones sit; these tests pin that at the front, in the middle and
// at the back. A block reports no span of its own.

use arael::model::{Component, Model, Param, SelfBlock, SimpleEulerAngleParam};
use arael::simple_lm::RootProblem;
use arael::vect::vect3d;

// Three slots of width 1, 3 and 1: any of them may be fixed.
#[arael::model]
#[arael(constraint(hb, {
    [(node.a - node.t) * 1.0,
     (node.v.x - node.t) * 1.0,
     (node.v.y - node.t) * 1.0,
     (node.v.z - node.t) * 1.0,
     (node.b - node.t) * 1.0]
}))]
struct Node {
    a: Param<f64>,
    v: Param<vect3d>,
    b: Param<f64>,
    t: f64,
    hb: SelfBlock<Node>,
}

#[arael::model]
#[arael(root)]
struct Graph {
    nodes: arael::refs::Vec<Node>,
}

fn node() -> Node {
    Node {
        a: Param::new(0.0),
        v: Param::new(vect3d::new(0.0, 0.0, 0.0)),
        b: Param::new(0.0),
        t: 1.0,
        hb: SelfBlock::new(),
    }
}

fn span_of<M: Model>(m: &M) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    Model::collect_param_blocks(m, &mut out);
    out
}

/// One entity per fixing pattern: nothing fixed, the first slot, the
/// middle (multi-component) slot, the last. Each span must start at the
/// entity's first live index and cover exactly its live components.
#[test]
fn fixed_slots_do_not_break_the_span() {
    let mut g = Graph { nodes: arael::refs::Vec::new() };
    g.nodes.push(node());
    let mut n = node();
    n.a.optimize = false;
    g.nodes.push(n);
    let mut n = node();
    n.v.optimize = false;
    g.nodes.push(n);
    let mut n = node();
    n.b.optimize = false;
    g.nodes.push(n);

    let mut params = Vec::new();
    RootProblem::<f64>::serialize(&mut g, &mut params);

    // Widths in push order: 5 live, then 4, 2 and 4. The offsets follow
    // each other because a fixed slot takes no room in the vector.
    let want = [(0u32, 5u32), (5, 4), (9, 2), (11, 4)];
    for (i, &w) in want.iter().enumerate() {
        assert_eq!(span_of(&g.nodes[i]), vec![w],
            "node {} span", i);
    }
    assert_eq!(params.len(), 15);
}

/// A block reports no span: it holds Hessian values, and the span comes
/// from the parameters. Every fixing pattern still gives the span the
/// parameter vector actually has.
#[test]
fn a_block_reports_no_span() {
    let mut g = Graph { nodes: arael::refs::Vec::new() };
    for fix in 0..4 {
        let mut n = node();
        match fix {
            1 => n.a.optimize = false,
            2 => n.v.optimize = false,
            3 => n.b.optimize = false,
            _ => {}
        }
        g.nodes.push(n);
    }
    let mut params = Vec::new();
    RootProblem::<f64>::serialize(&mut g, &mut params);

    let mut at = 0u32;
    for (i, n) in g.nodes.iter().enumerate() {
        let live = 5 - [0, 1, 3, 1][i % 4];
        assert_eq!(span_of(n), vec![(at, live)], "node {} span", i);
        assert_eq!(span_of(&n.hb), vec![], "a block carries no span");
        at += live;
    }
}

/// Every slot fixed: no live parameter, so no span at all.
#[test]
fn an_all_fixed_entity_reports_nothing() {
    let mut g = Graph { nodes: arael::refs::Vec::new() };
    let mut n = node();
    n.a.optimize = false;
    n.v.optimize = false;
    n.b.optimize = false;
    g.nodes.push(n);
    g.nodes.push(node());
    let mut params = Vec::new();
    RootProblem::<f64>::serialize(&mut g, &mut params);

    assert_eq!(span_of(&g.nodes[0]), vec![]);
    assert_eq!(span_of(&g.nodes[1]), vec![(0, 5)]);
}

// --- a component's slots fold into the owning entity's span ---

#[arael::model]
#[arael(component)]
struct Shift {
    ref_c: f64,
    d: Param<f64>,
    #[arael(symbolic = ref_c + d)]
    c: f64,
}

impl Component for Shift {
    fn start(&mut self) { self.ref_c = self.c; self.d.value = 0.0; }
    fn update(&mut self) { self.ref_c += self.d.value; self.d.value = 0.0; }
    fn finish(&mut self) { self.c = self.ref_c + self.d.value; }
}

#[arael::model]
#[arael(constraint(hb, {
    [(cell.w - cell.s.c) * 1.0]
}))]
struct Cell {
    w: Param<f64>,
    s: Shift,
    hb: SelfBlock<Cell>,
}

#[arael::model]
#[arael(root)]
struct Sheet {
    cells: arael::refs::Vec<Cell>,
}

fn cell() -> Cell {
    Cell { w: Param::new(0.0), s: Shift { ref_c: 0.0, d: Param::new(0.0), c: 0.5 },
           hb: SelfBlock::new() }
}

/// A `#[arael(component)]` field has no span of its own; its slots
/// belong to the entity that holds it, which reports one span of two.
#[test]
fn component_slots_fold_into_the_owner() {
    let mut s = Sheet { cells: arael::refs::Vec::new() };
    s.cells.push(cell());
    s.cells.push(cell());
    let mut params = Vec::new();
    RootProblem::<f64>::serialize(&mut s, &mut params);

    assert_eq!(params.len(), 4);
    assert_eq!(span_of(&s.cells[0]), vec![(0, 2)]);
    assert_eq!(span_of(&s.cells[1]), vec![(2, 2)]);
    // The component itself is not an entity: it reports no span.
    assert_eq!(span_of(&s.cells[0].s), vec![]);
}

// --- components inside components, and euler slots inside components ---

#[arael::model]
#[arael(component)]
struct Inner {
    d: Param<f64>,
    r: SimpleEulerAngleParam<f64>,
}
impl Component for Inner {}

#[arael::model]
#[arael(component)]
struct Outer {
    a: Param<vect3d>,
    inner: Inner,
}
impl Component for Outer {}

#[arael::model]
struct Rig {
    w: Param<f64>,
    o: Outer,
    hb: SelfBlock<Rig>,
}

#[arael::model]
#[arael(root)]
struct Frame {
    rigs: arael::refs::Vec<Rig>,
}

fn rig() -> Rig {
    Rig {
        w: Param::new(0.0),
        o: Outer {
            a: Param::new(vect3d::new(0.0, 0.0, 0.0)),
            inner: Inner {
                d: Param::new(0.0),
                r: SimpleEulerAngleParam::new(vect3d::new(0.0, 0.0, 0.0)),
            },
        },
        hb: SelfBlock::new(),
    }
}

/// A component's slots fold into the entity that holds it, at any
/// nesting depth, and an euler-angle slot counts its three components
/// like any other. The entity here owns 1 + 3 + 1 + 3.
#[test]
fn nested_component_slots_fold_into_the_entity() {
    let mut f = Frame { rigs: arael::refs::Vec::new() };
    f.rigs.push(rig());
    f.rigs.push(rig());
    let mut params = Vec::new();
    RootProblem::<f64>::serialize(&mut f, &mut params);

    assert_eq!(params.len(), 16);
    assert_eq!(<Rig as Model>::PARAM_COUNT, 8);
    assert_eq!(span_of(&f.rigs[0]), vec![(0, 8)]);
    assert_eq!(span_of(&f.rigs[1]), vec![(8, 8)]);
    // Neither component is an entity, at either depth.
    assert_eq!(span_of(&f.rigs[0].o), vec![]);
    assert_eq!(span_of(&f.rigs[0].o.inner), vec![]);
    assert_eq!(span_of(&f.rigs[0].hb), vec![], "a block carries no span");
}

/// Fixing a slot inside a nested component drops exactly its components
/// from the owning entity's span, wherever it sits.
#[test]
fn fixing_a_component_slot_shortens_the_entity_span() {
    let mut f = Frame { rigs: arael::refs::Vec::new() };
    // The euler slot in the inner component: 3 fewer, same offset.
    let mut r = rig();
    r.o.inner.r.optimize = false;
    f.rigs.push(r);
    // The entity's own first slot: 1 fewer, and the span now starts at
    // the outer component's vector.
    let mut r = rig();
    r.w.optimize = false;
    f.rigs.push(r);
    // Every slot of the outer component, the inner one with it.
    let mut r = rig();
    r.o.a.optimize = false;
    r.o.inner.d.optimize = false;
    r.o.inner.r.optimize = false;
    f.rigs.push(r);
    let mut params = Vec::new();
    RootProblem::<f64>::serialize(&mut f, &mut params);

    assert_eq!(params.len(), 5 + 7 + 1);
    assert_eq!(span_of(&f.rigs[0]), vec![(0, 5)]);
    assert_eq!(span_of(&f.rigs[1]), vec![(5, 7)]);
    assert_eq!(span_of(&f.rigs[2]), vec![(12, 1)]);
    for r in f.rigs.iter() {
        assert_eq!(span_of(&r.hb), vec![], "a block carries no span");
    }
}

// --- an entity no constraint touches ---

#[arael::model]
struct Loose {
    x: Param<f64>,
    hb: SelfBlock<Loose>,
}

#[arael::model]
#[arael(root)]
struct Mixed {
    nodes: arael::refs::Vec<Node>,
    loose: arael::refs::Vec<Loose>,
}

/// An entity in no constraint still has parameters in the vector and a
/// well defined span, so a covariance query on it is answerable. In a
/// root collection its block is wired anyway (the passive-entity pass),
/// and the two agree.
#[test]
fn an_untouched_root_entity_reports_its_span() {
    let mut m = Mixed { nodes: arael::refs::Vec::new(), loose: arael::refs::Vec::new() };
    m.nodes.push(node());
    m.loose.push(Loose { x: Param::new(0.0), hb: SelfBlock::new() });
    m.loose.push(Loose { x: Param::new(0.0), hb: SelfBlock::new() });
    let mut params = Vec::new();
    RootProblem::<f64>::serialize(&mut m, &mut params);

    assert_eq!(params.len(), 7);
    assert_eq!(span_of(&m.nodes[0]), vec![(0, 5)]);
    assert_eq!(span_of(&m.loose[0]), vec![(5, 1)]);
    assert_eq!(span_of(&m.loose[1]), vec![(6, 1)]);
    assert_eq!(span_of(&m.loose[0].hb), vec![], "a block carries no span");

    // The root collects every entity's span, in serialize order.
    let spans = RootProblem::<f64>::param_block_spans(&m);
    assert_eq!(spans, vec![(0, 5), (5, 1), (6, 1)]);
}

#[arael::model]
struct Bag {
    loose: arael::refs::Vec<Loose>,
}

#[arael::model]
#[arael(root)]
struct Deep {
    nodes: arael::refs::Vec<Node>,
    bags: arael::refs::Vec<Bag>,
}

/// The same entity below a grouping sub-model. The passive-entity pass
/// reaches here too, so again the span and the block agree; the span is
/// now read off the parameters either way.
#[test]
fn an_untouched_nested_entity_reports_its_span() {
    let mut d = Deep { nodes: arael::refs::Vec::new(), bags: arael::refs::Vec::new() };
    d.nodes.push(node());
    let mut bag = Bag { loose: arael::refs::Vec::new() };
    bag.loose.push(Loose { x: Param::new(0.0), hb: SelfBlock::new() });
    bag.loose.push(Loose { x: Param::new(0.0), hb: SelfBlock::new() });
    d.bags.push(bag);
    let mut params = Vec::new();
    RootProblem::<f64>::serialize(&mut d, &mut params);

    assert_eq!(params.len(), 7);
    assert_eq!(span_of(&d.bags[0].loose[0]), vec![(5, 1)]);
    assert_eq!(span_of(&d.bags[0].loose[1]), vec![(6, 1)]);
    assert_eq!(span_of(&d.bags[0].loose[0].hb), vec![], "a block carries no span");
    assert_eq!(RootProblem::<f64>::param_block_spans(&d),
        vec![(0, 5), (5, 1), (6, 1)]);
}

// --- a grouping sub-model owns no span ---

#[arael::model]
struct Group {
    nodes: arael::refs::Vec<Node>,
}

#[arael::model]
#[arael(root)]
struct Nested {
    groups: arael::refs::Vec<Group>,
}

/// A sub-model with no self block is not an entity: it reports the spans
/// of the entities below it and none of its own.
#[test]
fn a_grouping_sub_model_reports_only_its_entities() {
    let mut n = Nested { groups: arael::refs::Vec::new() };
    let mut g0 = Group { nodes: arael::refs::Vec::new() };
    g0.nodes.push(node());
    g0.nodes.push(node());
    n.groups.push(g0);
    let mut g1 = Group { nodes: arael::refs::Vec::new() };
    g1.nodes.push(node());
    n.groups.push(g1);
    let mut params = Vec::new();
    RootProblem::<f64>::serialize(&mut n, &mut params);

    assert_eq!(params.len(), 15);
    assert_eq!(span_of(&n.groups[0]), vec![(0, 5), (5, 5)]);
    assert_eq!(span_of(&n.groups[1]), vec![(10, 5)]);
}

// --- every parameter type folds into the entity that holds it ---

#[arael::model]
struct Every {
    p: Param<f64>,
    a: arael::angle::AngleParam<f64>,
    u: arael::unitvec::UnitVecParam<f64>,
    se: SimpleEulerAngleParam<f64>,
    e: arael::model::EulerAngleParam<f64>,
    q: arael::model::QuaternionParam<f64>,
    t: arael::transform::TransformParam<f64>,
    st: arael::transform::ScaledTransformParam<f64>,
    hb: SelfBlock<Every>,
}

#[arael::model]
#[arael(root)]
struct Kit {
    items: arael::refs::Vec<Every>,
}

fn every() -> Every {
    use arael::quatern::quaternd;
    use arael::vect::vect3d;
    let v = vect3d::new(0.0, 0.0, 0.0);
    Every {
        p: Param::new(0.0),
        a: arael::angle::AngleParam::new(0.0),
        u: arael::unitvec::UnitVecParam::new(vect3d::new(1.0, 0.0, 0.0)),
        se: SimpleEulerAngleParam::new(v),
        e: arael::model::EulerAngleParam::new(v),
        q: arael::model::QuaternionParam::new(quaternd::identity()),
        t: arael::transform::TransformParam::new(v, quaternd::identity()),
        st: arael::transform::ScaledTransformParam::new(v, quaternd::identity(), 1.0),
        hb: SelfBlock::new(),
    }
}

/// Every parameter type carries its slots into the span of the entity
/// holding it. One that does not leaves a hole, and the entity reads as
/// two blocks instead of one -- which a block-partitioned solve rejects
/// as a parameter no constraint touches.
#[test]
fn every_param_type_folds_into_the_entity_span() {
    let mut k = Kit { items: arael::refs::Vec::new() };
    k.items.push(every());
    k.items.push(every());
    let mut params = Vec::new();
    RootProblem::<f64>::serialize(&mut k, &mut params);

    // 1 + 1 + 2 + 3 + 3 + 3 + 6 + 7.
    assert_eq!(<Every as Model>::PARAM_COUNT, 26);
    assert_eq!(params.len(), 52);
    assert_eq!(span_of(&k.items[0]), vec![(0, 26)]);
    assert_eq!(span_of(&k.items[1]), vec![(26, 26)]);
}

/// Fixing one slot drops exactly that type's components from the span,
/// whichever type it is, and the entities after it keep following on.
#[test]
fn fixing_a_slot_of_any_param_type_shortens_the_span() {
    let mut k = Kit { items: arael::refs::Vec::new() };
    // The angle: one component fewer.
    let mut e = every();
    e.a.angle.optimize = false;
    k.items.push(e);
    // The direction's tangent delta: two fewer.
    let mut e = every();
    e.u.d.optimize = false;
    k.items.push(e);
    // The transform's rotation half: three fewer.
    let mut e = every();
    e.t.optimize_rotation = false;
    k.items.push(e);
    // The scaled transform whole: seven fewer.
    let mut e = every();
    e.st.optimize_translation = false;
    e.st.optimize_rotation = false;
    e.st.optimize_scale = false;
    k.items.push(e);
    let mut params = Vec::new();
    RootProblem::<f64>::serialize(&mut k, &mut params);

    assert_eq!(params.len(), 25 + 24 + 23 + 19);
    assert_eq!(span_of(&k.items[0]), vec![(0, 25)]);
    assert_eq!(span_of(&k.items[1]), vec![(25, 24)]);
    assert_eq!(span_of(&k.items[2]), vec![(49, 23)]);
    assert_eq!(span_of(&k.items[3]), vec![(72, 19)]);
}
