// A MixedEulerAngleParam whose wide scalar is not the root's: the scope
// would read its rotation in another type, so the macro reports it.
use arael::model::{MixedEulerAngleParam, SelfBlock};
#[allow(unused_imports)]
use arael::refs;
use arael::vect::vect3;

#[arael::model]
#[arael(constraint(hb, {
    let q = node.ea.rotation_matrix() * node.p;
    [with_wide(q.x - node.d)]
}))]
struct Node {
    ea: MixedEulerAngleParam<f32, f32>,
    p: vect3<f32>,
    d: f32,
    hb: SelfBlock<Node, f32>,
}

#[arael::model]
#[arael(root, f32)]
struct Root {
    nodes: refs::Vec<Node>,
}

fn main() {}
