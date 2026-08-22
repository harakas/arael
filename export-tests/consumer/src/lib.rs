// The importing crate: registers the model crate's layouts with one
// invocation, then defines its own entity, a cross-crate constraint, and
// f64 + f32 roots mixing local and imported types.

use arael::model::{CrossBlock, Param, SelfBlock};
use arael::refs::{self, Ref};
use arael::utils::Float;
use export_models::{Beacon, Cal, Mark, Spring};

export_models::arael_import!();
// A second import of the same bundle must be harmless (diamond imports:
// two dependencies both re-exporting the same model crate).
export_models::arael_import!();

/// A local entity: one shared bias, weakly pulled to zero.
#[arael::model]
#[arael(constraint(hb, {
    [bias.v * 0.1]
}))]
pub struct Bias<T: Float> {
    pub v: Param<T>,
    pub hb: SelfBlock<Bias<T>, T>,
}

/// A LOCAL constraint over an IMPORTED entity and a local one -- the
/// CrossBlock rewrite resolves `Beacon_PARAM_COUNT` from the import.
#[arael::model]
#[arael(constraint(hb, {
    [(bk.pos.x + bl.v - biaslink.m) * 2.0]
}))]
pub struct BiasLink<T: Float> {
    #[arael(ref = root.beacons)]
    pub bk: Ref<Beacon<T>>,
    #[arael(ref = root.biases)]
    pub bl: Ref<Bias<T>>,
    pub m: T,
    pub hb: CrossBlock<Beacon<T>, Bias<T>, T>,
}

/// A local constraint reading an IMPORTED param-less record through a
/// data ref: `mk` joins no block, its fields are pure reads.
#[arael::model]
#[arael(constraint(hb, {
    [(bk.pos.y + bl.v - mk.anchor) * mk.w]
}))]
pub struct MarkLink<T: Float> {
    #[arael(ref = root.beacons)]
    pub bk: Ref<Beacon<T>>,
    #[arael(ref = root.biases)]
    pub bl: Ref<Bias<T>>,
    #[arael(ref = root.marks)]
    pub mk: Ref<Mark<T>>,
    pub hb: CrossBlock<Beacon<T>, Bias<T>, T>,
}

#[arael::model]
#[arael(root)]
pub struct World64 {
    pub beacons: refs::Vec<Beacon<f64>>,
    pub biases: refs::Vec<Bias<f64>>,
    pub springs: std::vec::Vec<Spring<f64>>,
    pub links: std::vec::Vec<BiasLink<f64>>,
    pub cals: refs::Vec<Cal<f64>>,
    pub marks: refs::Vec<Mark<f64>>,
    pub mark_links: std::vec::Vec<MarkLink<f64>>,
}

#[arael::model]
#[arael(root, f32)]
pub struct World32 {
    pub beacons: refs::Vec<Beacon<f32>>,
    pub biases: refs::Vec<Bias<f32>>,
    pub springs: std::vec::Vec<Spring<f32>>,
    pub links: std::vec::Vec<BiasLink<f32>>,
    pub cals: refs::Vec<Cal<f32>>,
    pub marks: refs::Vec<Mark<f32>>,
    pub mark_links: std::vec::Vec<MarkLink<f32>>,
}
