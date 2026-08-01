// Numerical grad/Hessian check for a constraint with the same shape as
// loc_global_demo's PointFrine:
//   - constraint struct nested inside a parent struct
//   - primary block is a dotted-path remote reference (like pose.hb_pose)
//   - a second local CrossBlock<Entity, Root> holds the entity-root
//     cross Hessian pairs
//   - residual body references both the Ref entity's params AND
//     root-level Params
//
// Linear residuals so Gauss-Newton Hessian = true Hessian (no 2*r*d^2r
// correction from the numerical side).

#[allow(unused_imports)]
use arael::simple_lm::RootProblem;
use arael::model::{Param, SelfBlock, CrossBlock};
use arael::simple_lm::LmProblem;

#[arael::model]
struct Pose {
    pos: Param<f64>,           // scalar pos for simplicity
    info_tag: f64,             // dummy non-param data
    hb_pose: SelfBlock<Pose>,
}

// Nested constraint inside Landmark, referencing its parent's pose
// (remote block target) plus root's offset (implicit root entity).
//
// Residual: (pose.pos - root.offset - landmark.offset_tag) * isigma
// Linear in pose.pos and root.offset; the landmark's offset_tag is a
// non-param field used as a constant shift.
#[arael::model]
#[arael(constraint([pose.hb_pose, hb_root], parent=lm, {
    [(pose.pos - testmodel.offset - lm.offset_tag) * testmodel.isigma]
}))]
struct Frine {
    #[arael(ref = root.poses)]
    pose: arael::refs::Ref<Pose>,
    hb_root: CrossBlock<Pose, TestModel>,
}

#[arael::model]
struct Landmark {
    offset_tag: f64,
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
    poses.push(Pose { pos: Param::new(1.4), info_tag: 0.0, hb_pose: SelfBlock::new() });
    poses.push(Pose { pos: Param::new(-0.7), info_tag: 0.0, hb_pose: SelfBlock::new() });
    poses.push(Pose { pos: Param::new(2.3), info_tag: 0.0, hb_pose: SelfBlock::new() });
    let mut landmarks = arael::refs::Vec::new();
    // Each landmark owns frines that couple its pose to the root offset.
    landmarks.push(Landmark {
        offset_tag: 0.1,
        frines: vec![
            Frine { pose: poses.ref_at(0), hb_root: CrossBlock::new() },
            Frine { pose: poses.ref_at(1), hb_root: CrossBlock::new() },
        ],
    });
    landmarks.push(Landmark {
        offset_tag: -0.2,
        frines: vec![
            Frine { pose: poses.ref_at(1), hb_root: CrossBlock::new() },
            Frine { pose: poses.ref_at(2), hb_root: CrossBlock::new() },
        ],
    });
    let mut model = TestModel {
        poses, landmarks,
        offset: Param::new(0.3),
        isigma: 2.0,
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
fn test_nested_remote_plus_root_cross_grad_hessian() {
    let (mut model, params) = build_model();
    let n = params.len();
    assert!(n > 0);

    let mut ag = vec![0.0; n];
    let mut ah = vec![0.0; n * n];
    model.calc_grad_hessian_dense(&params, &mut ag, &mut ah);

    let (ng, nh) = numerical_grad_hessian(&mut model, &params);

    // Print full grad + Hessian for diagnostics if something is off.
    let failed = (0..n).any(|i| (ag[i] - ng[i]).abs() >= 5e-4)
        || (0..n*n).any(|idx| (ah[idx] - nh[idx]).abs() >= 5e-3);
    if failed {
        eprintln!("params = {:?}", params);
        eprintln!("analytic grad = {:?}", ag);
        eprintln!("numerical grad = {:?}", ng);
        for i in 0..n {
            for j in 0..n {
                let idx = i * n + j;
                if (ah[idx] - nh[idx]).abs() >= 5e-3 {
                    eprintln!("hess[{},{}] analytic={:+.6} numerical={:+.6} diff={:+.6}",
                        i, j, ah[idx], nh[idx], ah[idx] - nh[idx]);
                }
            }
        }
    }

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
