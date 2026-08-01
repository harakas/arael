// Regression tests for the "single-struct model+root with direct-composed
// sub-model" shape. Covers:
//   - SelfBlock<Self> constraints attached directly to the root struct.
//   - A plain (non-refs::Vec) Sub field on the root, with its own SelfBlock<Sub>
//     and its own constraints.
// Both paths go through the single-instance emission branch in
// arael-macros/src/constraint.rs (EntityLocation::RootSelf /
// EntityLocation::DirectField).

use arael::simple_lm::RootProblem;
use arael::model::{JacobianModel, Model, Param, SelfBlock};
use arael::simple_lm::{self, LmConfig, LmProblem};

#[arael::model]
#[arael(constraint(hb, name = "fix_z", {
    [(sub.z - 5.0) * singleroot.isigma]
}))]
struct Sub {
    z: Param<f64>,
    #[arael(constraint_index)]
    ci: u32,
    hb: SelfBlock<Sub>,
}

#[arael::model]
#[arael(root, jacobian)]
#[arael(constraint(hb, name = "fix_x", {
    [(singleroot.x - 3.0) * singleroot.isigma]
}))]
#[arael(constraint(hb, name = "fix_y", {
    [(singleroot.y - 4.0) * singleroot.isigma]
}))]
struct SingleRoot {
    x: Param<f64>,
    y: Param<f64>,
    isigma: f64,
    sub: Sub,
    #[arael(constraint_index)]
    ci: u32,
    hb: SelfBlock<SingleRoot>,
}

fn make_model() -> (SingleRoot, Vec<f64>) {
    let mut m = SingleRoot {
        x: Param::new(0.0),
        y: Param::new(0.0),
        isigma: 10.0,
        sub: Sub { z: Param::new(0.0), ci: 0, hb: SelfBlock::new() },
        ci: 0,
        hb: SelfBlock::new(),
    };
    let mut params = Vec::new();
    m.serialize(&mut params);
    (m, params)
}

#[test]
fn param_counts_are_entity_local() {
    // PARAM_COUNT is entity-local (it sizes SelfBlock<Self>::grad), not
    // transitive through composed sub-models.
    assert_eq!(SingleRoot::PARAM_COUNT, 2);
    assert_eq!(Sub::PARAM_COUNT, 1);
}

#[test]
fn serialize_walks_composed_sub_model() {
    let (_, params) = make_model();
    // Root owns x, y; Sub (direct-composed) owns z. Three params total.
    assert_eq!(params.len(), 3);
    assert_eq!(params, vec![0.0, 0.0, 0.0]);
}

#[test]
fn calc_cost_includes_root_and_sub_constraints() {
    let (mut m, params) = make_model();
    // (0-3)^2 * 10^2 + (0-4)^2 * 10^2 + (0-5)^2 * 10^2 = 900 + 1600 + 2500 = 5000
    let cost = m.calc_cost(&params);
    assert!((cost - 5000.0).abs() < 1e-12, "expected 5000.0, got {}", cost);
}

#[test]
fn calc_jacobian_emits_all_three_residuals() {
    let (mut m, params) = make_model();
    let j = m.calc_jacobian(&params);
    assert_eq!(j.num_residuals(), 3, "expected 3 residuals, got {}", j.num_residuals());
    assert_eq!(j.num_params, 3, "expected 3 params, got {}", j.num_params);

    let cost_from_j: f64 = j.rows.iter().map(|r| r.residual * r.residual).sum();
    let cost_from_c = m.calc_cost(&params);
    assert!((cost_from_j - cost_from_c).abs() < 1e-12);

    // Every row must touch at least one param.
    for row in &j.rows {
        assert!(!row.entries.is_empty(), "row {:?} has no param entries", row.label);
    }
}

#[test]
fn lm_solve_converges_to_fixed_values() {
    let (mut m, params) = make_model();
    let config = LmConfig::<f64> { verbose: false, ..Default::default() };
    let result = simple_lm::solve(&params, &mut m, &config).unwrap();
    m.deserialize(&result.x);
    assert!((m.x.value - 3.0).abs() < 1e-10, "x.value = {}", m.x.value);
    assert!((m.y.value - 4.0).abs() < 1e-10, "y.value = {}", m.y.value);
    assert!((m.sub.z.value - 5.0).abs() < 1e-10, "sub.z.value = {}", m.sub.z.value);
    assert!(result.end_cost < 1e-20, "end_cost = {}", result.end_cost);
}

#[test]
fn constraint_indices_are_assigned() {
    // Both the root and the direct-composed Sub have #[arael(constraint_index)]
    // fields. After serialize (which runs __set_block_indices), they must
    // have been filled in with distinct, sequential IDs.
    let (m, _) = make_model();
    // Three constraints total; IDs assigned in traversal order. The exact
    // order between root-self and direct-composed-sub is implementation
    // defined by the HashMap iteration of single_instance_groups, so assert
    // only that all three IDs are distinct and in [0, 3).
    let ids = [m.ci, m.sub.ci];
    // All in range.
    for &id in &ids { assert!(id < 3, "constraint id {} out of range", id); }
    // Root and sub have different ids (distinct single-instance groups).
    assert_ne!(m.ci, m.sub.ci);
}
