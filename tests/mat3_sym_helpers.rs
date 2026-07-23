// matrix3sym helpers in constraint bodies: .col(i) and
// .get_rotation_vector_small() against the hand-written
// from_components spelling of the same residual. All three must
// produce identical gradients and Hessians -- they are the same
// expressions, so the generated code is bit-identical.

use arael::matrix::matrix3d;
use arael::model::{Param, SelfBlock};
use arael::refs;
use arael::simple_lm::LmProblem;
use arael::vect::vect3d;

// vee((meas_t * R(ea) - transpose)/2), spelled with from_components.
#[arael::model]
#[arael(constraint(hb, {
    let r = veehand.meas_t * veehand.ea.rotation_matrix();
    let c1 = r * vect3sym::from_components(1.0, 0.0, 0.0);
    let c2 = r * vect3sym::from_components(0.0, 1.0, 0.0);
    let c3 = r * vect3sym::from_components(0.0, 0.0, 1.0);
    [(c2.z - c3.y) * 0.5, (c3.x - c1.z) * 0.5, (c1.y - c2.x) * 0.5]
}))]
struct VeeHand {
    ea: Param<vect3d>,
    meas_t: matrix3d,
    hb: SelfBlock<VeeHand>,
}

// The same residual with the columns extracted by .col(i).
#[arael::model]
#[arael(constraint(hb, {
    let r = veecol.meas_t * veecol.ea.rotation_matrix();
    let c1 = r.col(0);
    let c2 = r.col(1);
    let c3 = r.col(2);
    [(c2.z - c3.y) * 0.5, (c3.x - c1.z) * 0.5, (c1.y - c2.x) * 0.5]
}))]
struct VeeCol {
    ea: Param<vect3d>,
    meas_t: matrix3d,
    hb: SelfBlock<VeeCol>,
}

// The same residual as one call.
#[arael::model]
#[arael(constraint(hb, {
    let r = veeapi.meas_t * veeapi.ea.rotation_matrix();
    let rv = r.get_rotation_vector_small();
    [rv.x, rv.y, rv.z]
}))]
struct VeeApi {
    ea: Param<vect3d>,
    meas_t: matrix3d,
    hb: SelfBlock<VeeApi>,
}

// Gravity-style direction residual: the third row of the rotation
// against a fixed measured direction, spelled with .row(2)...
#[arael::model]
#[arael(constraint(hb, {
    let d = rowa.ea.rotation_matrix().row(2) - rowa.g;
    [d.x, d.y, d.z]
}))]
struct RowA {
    ea: Param<vect3d>,
    g: vect3d,
    hb: SelfBlock<RowA>,
}

// ...and as .transpose().col(2), which must be the same expressions.
#[arael::model]
#[arael(constraint(hb, {
    let d = rowb.ea.rotation_matrix().transpose().col(2) - rowb.g;
    [d.x, d.y, d.z]
}))]
struct RowB {
    ea: Param<vect3d>,
    g: vect3d,
    hb: SelfBlock<RowB>,
}

#[arael::model]
#[arael(root)]
struct W {
    hand: refs::Vec<VeeHand>,
    col: refs::Vec<VeeCol>,
    api: refs::Vec<VeeApi>,
    rowa: refs::Vec<RowA>,
    rowb: refs::Vec<RowB>,
}

fn build() -> (W, Vec<f64>) {
    // Initial guess differs from the measurement so the residual is live.
    let ea = vect3d::new(0.02, -0.01, 0.03);
    let meas_t = matrix3d::rotation_from_euler_angles(
        vect3d::new(0.01, 0.005, -0.02)).transpose();
    let g = vect3d::new(0.01, -0.02, 0.9997).unit();
    let mut w = W {
        hand: refs::Vec::new(),
        col: refs::Vec::new(),
        api: refs::Vec::new(),
        rowa: refs::Vec::new(),
        rowb: refs::Vec::new(),
    };
    w.hand.push(VeeHand { ea: Param::new(ea), meas_t, hb: SelfBlock::new() });
    w.col.push(VeeCol { ea: Param::new(ea), meas_t, hb: SelfBlock::new() });
    w.api.push(VeeApi { ea: Param::new(ea), meas_t, hb: SelfBlock::new() });
    w.rowa.push(RowA { ea: Param::new(ea), g, hb: SelfBlock::new() });
    w.rowb.push(RowB { ea: Param::new(ea), g, hb: SelfBlock::new() });
    let mut params = Vec::new();
    w.serialize64(&mut params);
    (w, params)
}

// Param layout: hand(0..3), col(3..6), api(6..9), rowa(9..12), rowb(12..15).
#[test]
fn col_and_vee_match_hand_written() {
    let (mut w, params) = build();
    assert_eq!(params.len(), 15);
    let n = params.len();
    let mut grad = vec![0.0_f64; n];
    let mut hess = vec![0.0_f64; n * n];
    w.calc_grad_hessian_dense(&params, &mut grad, &mut hess);

    // The residual must be live at this point.
    assert!(grad[..3].iter().any(|g| g.abs() > 1e-6), "dead residual");

    for i in 0..3 {
        assert_eq!(grad[3 + i], grad[i], "col grad {i}");
        assert_eq!(grad[6 + i], grad[i], "api grad {i}");
        for j in 0..3 {
            assert_eq!(hess[(3 + i) * n + 3 + j], hess[i * n + j], "col hess {i},{j}");
            assert_eq!(hess[(6 + i) * n + 6 + j], hess[i * n + j], "api hess {i},{j}");
        }
    }

    // row(2) and transpose().col(2) are the same expressions.
    assert!(grad[9..12].iter().any(|g| g.abs() > 1e-6), "dead row residual");
    for i in 0..3 {
        assert_eq!(grad[12 + i], grad[9 + i], "row grad {i}");
        for j in 0..3 {
            assert_eq!(hess[(12 + i) * n + 12 + j], hess[(9 + i) * n + 9 + j],
                "row hess {i},{j}");
        }
    }
}
