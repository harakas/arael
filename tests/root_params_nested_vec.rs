// Regression test for the parent_name binding in is_remote_block +
// is_multi_cross constraints. Residual body references a VECTOR field
// on the parent struct (e.g. `lm.pos: vect2d`) whose type tag matters
// for sym-val inference. If `parent_name` isn't registered in var_infos
// before body interpretation, `lm.pos` becomes an unknown-scalar symbol
// and subsequent vector arithmetic (`lm.pos - pose.pos`) hits a
// "type mismatch in subtraction" error.
//
// This is the exact bug that silently dropped PointFrine from
// loc_global_demo's cost function, leaving zero gradient on the root's
// global_delta/global_rot params.

#[allow(unused_imports)]
use arael::simple_lm::RootProblem;
use arael::model::{Param, SelfBlock, CrossBlock, Model};
use arael::simple_lm::LmProblem;
use arael::vect::vect2d;

#[arael::model]
struct Pose {
    pos: Param<vect2d>,
    hb_pose: SelfBlock<Pose>,
}

// Residual does a vector subtraction between the parent's non-Param
// `anchor` field and the ref'd pose's position, plus a scalar root
// param. The vector subtraction is what requires `lm` to be a
// registered binding (type vect2d) in the constraint context.
#[arael::model]
#[arael(constraint([pose.hb_pose, hb_root], parent=lm, {
    let d = lm.anchor - pose.pos;
    [(d.x + testmodel.offset) * testmodel.isigma,
     (d.y + testmodel.offset) * testmodel.isigma]
}))]
struct Frine {
    #[arael(ref = root.poses)]
    pose: arael::refs::Ref<Pose>,
    hb_root: CrossBlock<Pose, TestModel>,
}

#[arael::model]
struct Landmark {
    anchor: vect2d,
    frines: std::vec::Vec<Frine>,
}

#[arael::model]
#[arael(root)]
struct TestModel {
    poses: arael::refs::Vec<Pose>,
    landmarks: arael::refs::Vec<Landmark>,
    offset: Param<f64>,
    isigma: f64,
    hb: SelfBlock<TestModel>,
}

fn build_model() -> (TestModel, Vec<f64>) {
    let mut poses = arael::refs::Vec::new();
    poses.push(Pose { pos: Param::new(vect2d::new(1.1, 0.3)), hb_pose: SelfBlock::new() });
    poses.push(Pose { pos: Param::new(vect2d::new(-0.4, 1.7)), hb_pose: SelfBlock::new() });
    let mut landmarks = arael::refs::Vec::new();
    landmarks.push(Landmark {
        anchor: vect2d::new(2.0, 1.0),
        frines: vec![
            Frine { pose: poses.ref_at(0), hb_root: CrossBlock::new() },
            Frine { pose: poses.ref_at(1), hb_root: CrossBlock::new() },
        ],
    });
    let mut model = TestModel {
        poses, landmarks,
        offset: Param::new(0.2),
        isigma: 3.0,
        hb: SelfBlock::new(),
    };
    let mut params = Vec::new();
    model.serialize(&mut params);
    (model, params)
}

fn numerical_grad_hessian(model: &mut TestModel, params: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = params.len();
    let eps_g = 1e-6;
    let eps_h = 1e-4;
    let mut grad = vec![0.0; n];
    for i in 0..n {
        let mut pp = params.to_vec();
        pp[i] += eps_g;
        let cp = model.calc_cost(&pp);
        pp[i] -= 2.0 * eps_g;
        let cm = model.calc_cost(&pp);
        grad[i] = (cp - cm) / (2.0 * eps_g);
    }
    let mut hess = vec![0.0; n * n];
    for i in 0..n {
        for j in i..n {
            let h: f64 = if i == j {
                let c0 = model.calc_cost(params);
                let mut pp = params.to_vec();
                pp[i] += eps_h;
                let cp = model.calc_cost(&pp);
                pp[i] -= 2.0 * eps_h;
                let cm = model.calc_cost(&pp);
                (cp - 2.0 * c0 + cm) / (eps_h * eps_h)
            } else {
                let mut pp = params.to_vec();
                pp[i] += eps_h; pp[j] += eps_h;
                let cpp = model.calc_cost(&pp);
                pp[j] -= 2.0 * eps_h;
                let cpm = model.calc_cost(&pp);
                pp[i] -= 2.0 * eps_h;
                let cmm = model.calc_cost(&pp);
                pp[j] += 2.0 * eps_h;
                let cmp = model.calc_cost(&pp);
                (cpp - cpm - cmp + cmm) / (4.0 * eps_h * eps_h)
            };
            hess[i * n + j] = h;
            hess[j * n + i] = h;
        }
    }
    (grad, hess)
}

#[test]
fn test_nested_remote_plus_root_cross_with_parent_vec() {
    let (mut model, params) = build_model();
    let n = params.len();
    assert!(n > 0);

    // Check cost is nonzero -- it would be silently zero if the
    // constraint was dropped (as happened in loc_global_demo before
    // the fix).
    let cost = model.calc_cost(&params);
    assert!(cost > 0.0, "cost should be nonzero; constraint may have been silently dropped");

    let mut ag = vec![0.0; n];
    let mut ah = vec![0.0; n * n];
    model.calc_grad_hessian_dense(&params, &mut ag, &mut ah);

    let (ng, nh) = numerical_grad_hessian(&mut model, &params);

    // Sanity: the root offset param MUST have a nonzero gradient here
    // (the residual depends on it). If it's zero, the constraint isn't
    // contributing -- the exact bug this test guards against.
    let offset_idx = model.offset.index() as usize;
    assert!(ag[offset_idx].abs() > 1e-6,
        "root offset param must have nonzero analytic gradient; got {}", ag[offset_idx]);
    assert!(ng[offset_idx].abs() > 1e-6,
        "root offset param must have nonzero numerical gradient; got {}", ng[offset_idx]);

    for i in 0..n {
        assert!((ag[i] - ng[i]).abs() < 5e-4,
            "grad[{}] analytic={} numerical={}", i, ag[i], ng[i]);
    }
    for i in 0..n {
        for j in 0..n {
            let idx = i * n + j;
            let d = (ah[idx] - nh[idx]).abs();
            assert!(d < 5e-3,
                "hess[{},{}] analytic={} numerical={} diff={}", i, j, ah[idx], nh[idx], d);
        }
    }
}
