// A remote-block body reading its containing parent's param: the remote
// block holds only the target's params, so the read is rejected.
use arael::model::{Param, SelfBlock};
use arael::refs::Ref;
use arael::vect::vect2d;

#[arael::model]
struct Beacon {
    pos: Param<vect2d>,
    hb: SelfBlock<Beacon>,
}

#[arael::model]
#[arael(constraint(b.hb, parent = st, { [b.pos.x - sobs.m.x - st.bias, b.pos.y - sobs.m.y] }))]
struct SObs {
    #[arael(ref = root.beacons)]
    b: Ref<Beacon>,
    m: vect2d,
}

#[arael::model]
#[arael(constraint(hb, { [station.bias] }))]
struct Station {
    bias: Param<f64>,
    obs: std::vec::Vec<SObs>,
    hb: SelfBlock<Station>,
}

#[arael::model]
#[arael(root)]
struct Field {
    beacons: arael::refs::Vec<Beacon>,
    stations: std::vec::Vec<Station>,
}

fn main() {}
