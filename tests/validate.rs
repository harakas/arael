// validate() is the model linter: it reports every formulation problem
// it finds -- non-finite parameters, stale refs, unconstrained
// parameters, gradient mismatches -- instead of the solve failing on
// the first one. check_gradients()/numeric_gradient() are the
// standalone gradient pieces.

use arael::simple_lm::RootProblem;
use arael::model::{Param, SelfBlock, CrossBlock};
use arael::refs::{self, Ref};
use arael::simple_lm::LmProblem;
use arael::validate::Issue;

// --- a healthy little pose chain with refs ---

#[arael::model]
#[arael(constraint(hb, {
    [(node.x - node.ax) * 0.5]
}))]
struct Node {
    x: Param<f64>,
    ax: f64,
    hb: SelfBlock<Node>,
}

#[arael::model]
#[arael(constraint(hb, {
    [b.x - a.x - tie.d]
}))]
struct Tie {
    #[arael(ref = root.nodes)]
    a: Ref<Node>,
    #[arael(ref = root.nodes)]
    b: Ref<Node>,
    d: f64,
    hb: CrossBlock<Node, Node>,
}

#[arael::model]
#[arael(root)]
struct Chain {
    nodes: refs::Arena<Node>,
    ties: std::vec::Vec<Tie>,
}

fn chain(n: usize) -> Chain {
    let mut c = Chain { nodes: refs::Arena::new(), ties: std::vec::Vec::new() };
    let refs: Vec<Ref<Node>> = (0..n)
        .map(|i| c.nodes.push(Node { x: Param::new(0.1 * i as f64), ax: i as f64, hb: SelfBlock::new() }))
        .collect();
    for w in refs.windows(2) {
        c.ties.push(Tie { a: w[0], b: w[1], d: 1.0, hb: CrossBlock::new() });
    }
    c
}

#[test]
fn a_healthy_model_is_clean() {
    let mut c = chain(4);
    let d = c.validate();
    assert!(d.is_clean(), "unexpected issues:\n{}", d);
    assert_eq!(d.to_string(), "model is clean");
}

#[test]
fn non_finite_params_are_reported_and_stop_the_pass() {
    let mut c = chain(3);
    c.nodes.iter_mut().nth(1).unwrap().x.value = f64::NAN;
    let d = c.validate();
    assert_eq!(d.issues.len(), 1, "{}", d);
    assert!(matches!(d.issues[0], Issue::NonFiniteParam { param: 1, .. }), "{}", d);
    assert!(d.to_string().contains("non-finite"), "{}", d);
}

#[test]
fn a_stale_ref_is_reported_with_its_path() {
    let mut c = chain(3);
    // Remove the middle node and refill the slot: the tie's refs now
    // carry a dead generation.
    let stale = c.ties[1].a;
    c.nodes.remove(stale).expect("the node is live");
    c.nodes.push(Node { x: Param::new(0.0), ax: 1.0, hb: SelfBlock::new() });
    let d = c.validate();
    assert!(!d.is_clean());
    // ties[0].b and ties[1].a both pointed at the removed node.
    assert_eq!(d.issues.len(), 2, "{}", d);
    assert!(d.issues.contains(&Issue::StaleRef { path: "ties[0].b".into() }), "{}", d);
    assert!(d.issues.contains(&Issue::StaleRef { path: "ties[1].a".into() }), "{}", d);
    assert!(d.to_string().contains("stale Ref"), "{}", d);
}

// --- an unconstrained parameter: nothing ever touches `loose` ---

#[arael::model]
#[arael(constraint(hb, {
    [(loose.x - 2.0) * 10.0]
}))]
struct Loose {
    x: Param<f64>,
    loose: Param<f64>,
    hb: SelfBlock<Loose>,
}

#[arael::model]
#[arael(root)]
struct LooseRoot {
    items: std::vec::Vec<Loose>,
}

#[test]
fn an_unconstrained_parameter_is_reported() {
    let mut w = LooseRoot {
        items: vec![Loose { x: Param::new(0.0), loose: Param::new(0.0), hb: SelfBlock::new() }],
    };
    let d = w.validate();
    assert_eq!(d.issues, vec![Issue::UnconstrainedParam { param: 1 }], "{}", d);
}

// --- a wrong hand-declared derivative is caught by the gradient check ---

// The declared derivative is off by a factor of two (x instead of 2x):
// cost and gradient disagree, which only the finite-difference
// comparison can see.
#[arael::function(wd, derivs = [x])]
fn wd_eval(x: f64) -> f64 {
    x * x
}

#[arael::model]
#[arael(constraint(hb, {
    [wd(wrong.x - 3.0)]
}))]
struct Wrong {
    x: Param<f64>,
    hb: SelfBlock<Wrong>,
}

#[arael::model]
#[arael(root)]
struct WrongRoot {
    items: std::vec::Vec<Wrong>,
}

#[test]
fn a_wrong_declared_derivative_is_a_gradient_mismatch() {
    let mut w = WrongRoot {
        items: vec![Wrong { x: Param::new(1.0), hb: SelfBlock::new() }],
    };
    let d = w.validate();
    assert!(
        d.issues.iter().any(|i| matches!(i, Issue::GradientMismatch { param: 0, .. })),
        "expected a gradient mismatch:\n{}", d
    );

    // The standalone checker sees the same thing over a raw vector.
    let mut params = Vec::new();
    w.serialize(&mut params);
    let d = w.check_gradients(&params);
    assert!(!d.is_clean());
}

// --- the numeric gradient itself, against a healthy analytic one ---

#[test]
fn numeric_gradient_matches_a_healthy_assembly() {
    let mut c = chain(4);
    let mut params = Vec::new();
    c.serialize(&mut params);

    let n = params.len();
    let mut grad = vec![0.0; n];
    let mut coo = arael::simple_lm::CooMatrix::new(n);
    c.calc_grad_hessian_sparse(&params, &mut grad, &mut coo);

    let fd = c.numeric_gradient(&params);
    for i in 0..n {
        assert!((fd[i] - grad[i]).abs() < 1e-6 * (1.0 + grad[i].abs()),
            "grad[{}]: analytic={} fd={}", i, grad[i], fd[i]);
    }
    assert!(grad.iter().any(|g| g.abs() > 0.1), "gradient suspiciously small");

    // And the packaged comparison agrees.
    assert!(c.check_gradients(&params).is_clean());
}
