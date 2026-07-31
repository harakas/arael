use arael::model::{Param, Model};
use arael::vect::vect3f;
use arael::matrix::matrix3f;
use arael::info;

// A simple data point — no optimizable parameters
#[arael::model]
struct DataEntry {
    x: f32,
    y: f32,
}

// Linear model y = a*x + b with two optimizable parameters
#[arael::model]
#[arael(skip_self_block)]
struct LinearModel {
    a: Param<f32>,
    b: Param<f32>,
    data: Vec<DataEntry>,
}

fn main() {
    // ===== LinearModel demo =====
    info!("=== LinearModel: serialize / update / deserialize ===");
    info!("");

    let mut model = LinearModel {
        a: Param::new(1.0),
        b: Param::new(0.0),
        data: vec![
            DataEntry { x: 1.0, y: 2.1 },
            DataEntry { x: 2.0, y: 3.9 },
            DataEntry { x: 3.0, y: 6.1 },
        ],
    };

    // Step 1: serialize — collects all optimizable parameter values
    let mut params = Vec::<f32>::new();
    model.serialize_params(&mut params);
    info!("Serialized params: {:?}", params);  // [1.0, 0.0]
    info!("  a index={}, b index={}", model.a.value, model.b.value);

    // Step 2: update_self — initialize work values from current values
    model.update_self();
    info!("");
    info!("After update_self:");
    info!("  a.work={}, b.work={}", model.a.work(), model.b.work());

    // Step 3: simulate optimizer modifying the parameter vector
    params[0] = 2.0;   // a = 2.0
    params[1] = 0.1;   // b = 0.1
    info!("");
    info!("Optimizer sets params to: {:?}", params);

    // Step 4: update — copies new values from data into work
    model.update_params(&params);
    info!("After update:");
    info!("  a.work={}, b.work={}", model.a.work(), model.b.work());
    info!("  a.value={}, b.value={} (unchanged)", model.a.value, model.b.value);

    // Evaluate cost with work values
    let cost: f32 = model.data.iter().map(|d| {
        let predicted = model.a.work() * d.x + model.b.work();
        let residual = d.y - predicted;
        residual * residual
    }).sum();
    info!("  Cost = {:.4}", cost);

    // Step 5: deserialize — commit optimized values back
    model.deserialize_params(&params);
    info!("");
    info!("After deserialize (commit):");
    info!("  a.value={}, b.value={}", model.a.value, model.b.value);

    // ===== Fixed parameters demo =====
    info!("");
    info!("=== Fixed parameter demo ===");
    info!("");

    let mut model2 = LinearModel {
        a: Param::fixed(2.0),  // fixed: will not be optimized
        b: Param::new(0.0),    // optimizable
        data: vec![DataEntry { x: 1.0, y: 2.0 }],
    };

    let mut params2 = Vec::<f32>::new();
    model2.serialize_params(&mut params2);
    info!("With a=fixed(2.0), b=new(0.0):");
    info!("  Serialized params: {:?} (only b)", params2);

    params2[0] = 0.5;
    model2.update_params(&params2);
    info!("  After update: a.work={}, b.work={}", model2.a.work(), model2.b.work());

    // ===== Computed field demo =====
    info!("");
    info!("=== Computed field (rotation from euler angles) ===");
    info!("");

    #[arael::model]
    #[arael(skip_self_block)]
    struct Pose {
        pos: Param<vect3f>,
        ea: Param<vect3f>,
        #[arael(compute = ea.rotation_matrix())]
        mr2w: matrix3f,
    }

    let mut pose = Pose {
        pos: Param::new(vect3f::new(1.0, 2.0, 3.0)),
        ea: Param::new(vect3f::new(0.0, 0.0, std::f32::consts::FRAC_PI_2)),
        mr2w: matrix3f::identity(),
    };

    let mut params3 = Vec::<f32>::new();
    pose.serialize_params(&mut params3);
    info!("Pose params: {:?}", params3);  // [1, 2, 3, 0, 0, pi/2]
    info!("  {} params total (pos=3 + ea=3)", params3.len());

    // update_self: computes mr2w from ea.work()
    pose.update_self();
    info!("");
    info!("After update_self, mr2w computed from euler angles:");
    info!("  mr2w = {:?}", pose.mr2w);

    // Simulate optimizer changing euler angles
    params3[5] = 0.0;  // set ea.z = 0 (no rotation)
    pose.update_params(&params3);
    info!("");
    info!("After update with ea.z=0:");
    info!("  ea.work = {:?}", pose.ea.work());
    info!("  mr2w = {:?} (should be ~identity)", pose.mr2w);
}
