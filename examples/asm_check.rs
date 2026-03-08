use arael::model::{Param, Model};

#[arael::model]
struct DataEntry {
    x: f32,
    y: f32,
}

#[arael::model]
pub struct LinearModel {
    a: Param<f32>,
    b: Param<f32>,
    data: Vec<DataEntry>,
}

#[unsafe(no_mangle)]
pub fn test_update(model: &mut LinearModel, data: &[f32]) {
    model.update32(data);
}

#[unsafe(no_mangle)]
pub fn test_serialize(model: &mut LinearModel, data: &mut Vec<f32>) {
    model.serialize_params32(data);
}

fn main() {}
