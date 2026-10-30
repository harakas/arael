// Covariance queries on a single parameter. A query gathers its indices
// from `Model::collect_param_blocks`, which the parameter types did not
// implement, so `marginal_cov(&line.k)` covered nothing and returned an
// empty 0x0 matrix. A parameter now reports its own span, and a query that
// covers no free parameter at all is `CovError::Empty`.
//
// Sigma = (J^T J)^-1, so the values below follow from the observations.

use arael::covariance::{CovError, CovMode, Covariance};
use arael::model::{Param, SelfBlock};

// The parameter on the root, data-only entities (a curve fit).
#[arael::model]
#[arael(constraint(root.hb, { [ob.y - line.k * ob.x] }))]
struct Ob { x: f64, y: f64 }

#[arael::model]
#[arael(root)]
struct Line {
    k: Param<f64>,
    obs: std::vec::Vec<Ob>,
    hb: SelfBlock<Line>,
}

fn line() -> Line {
    Line {
        k: Param::new(1.0),
        obs: vec![Ob { x: 1.0, y: 2.1 }, Ob { x: 2.0, y: 3.9 }, Ob { x: 3.0, y: 6.2 }],
        hb: SelfBlock::new(),
    }
}

// A parameter on the root and parameterized entities.
#[arael::model]
#[arael(constraint(root.hb, { [o2.y - mixed.g * o2.x] }))]
struct O2 { x: f64, y: f64 }

#[arael::model]
#[arael(constraint(hb, { [(pt.v - pt.t) * 2.0] }))]
struct Pt { v: Param<f64>, t: f64, hb: SelfBlock<Pt> }

#[arael::model]
#[arael(root)]
struct Mixed {
    g: Param<f64>,
    obs: std::vec::Vec<O2>,
    pts: std::vec::Vec<Pt>,
    hb: SelfBlock<Mixed>,
}

fn mixed() -> Mixed {
    Mixed {
        g: Param::new(1.0),
        obs: vec![O2 { x: 1.0, y: 1.0 }, O2 { x: 2.0, y: 2.0 }],
        pts: (0..3).map(|i| Pt { v: Param::new(0.0), t: i as f64, hb: SelfBlock::new() }).collect(),
        hb: SelfBlock::new(),
    }
}

fn close(a: f64, b: f64) -> bool { (a - b).abs() < 1e-12 }

/// J^T J of k is 1 + 4 + 9 = 14.
#[test]
fn a_root_parameter_alone() {
    for mode in [CovMode::PerQuery, CovMode::AllMarginals] {
        let mut l = line();
        let cov = l.assemble_covariance(mode).unwrap();
        let k = cov.marginal_cov(&l.k).unwrap();
        assert_eq!((k.nrows(), k.ncols()), (1, 1), "{mode:?}");
        assert!(close(k[(0, 0)], 1.0 / 14.0), "{mode:?}: {}", k[(0, 0)]);
        assert_eq!(k, cov.marginal_cov(&l).unwrap(), "{mode:?}: the root holds only k");
        assert!(close(cov.conditional_cov(&l.k).unwrap()[(0, 0)], 1.0 / 14.0), "{mode:?}");
        assert!(close(cov.std_dev(&l.k).unwrap()[0], (1.0f64 / 14.0).sqrt()), "{mode:?}");
    }
}

/// g: J^T J = 1 + 4 = 5; each v: 2^2 = 4; g and v do not couple.
#[test]
fn a_root_parameter_beside_entities() {
    for mode in [CovMode::PerQuery, CovMode::AllMarginals] {
        let mut m = mixed();
        let cov = m.assemble_covariance(mode).unwrap();
        let g = cov.marginal_cov(&m.g).unwrap();
        assert_eq!((g.nrows(), g.ncols()), (1, 1), "{mode:?}");
        assert!(close(g[(0, 0)], 0.2), "{mode:?}: {}", g[(0, 0)]);
        assert!(close(cov.std_dev(&m.g).unwrap()[0], 0.2f64.sqrt()), "{mode:?}");
        assert!(close(cov.conditional_cov(&m.g).unwrap()[(0, 0)], 0.2), "{mode:?}");
        let x = cov.cross_cov(&m.g, &m.pts[0]).unwrap();
        assert_eq!((x.nrows(), x.ncols()), (1, 1), "{mode:?}");
        assert!(close(x[(0, 0)], 0.0), "{mode:?}");
        // The joint over the root still covers g and the three v.
        let all = cov.marginal_cov(&m).unwrap();
        assert_eq!(all.nrows(), 4, "{mode:?}");
        assert!(close(cov.marginal_cov(&m.pts[1].v).unwrap()[(0, 0)], 0.25), "{mode:?}");
    }
}

/// A query that covers no free parameter is an error, not an empty
/// matrix: a fixed parameter, a data-only entity, either side of a cross.
#[test]
fn a_query_without_free_parameters_is_an_error() {
    let mut m = mixed();
    m.pts[2].v.optimize = false;
    let cov = m.assemble_covariance(CovMode::PerQuery).unwrap();
    assert_eq!(cov.marginal_cov(&m.pts[2].v).err(), Some(CovError::Empty));
    assert_eq!(cov.marginal_cov(&m.pts[2]).err(), Some(CovError::Empty));
    assert_eq!(cov.std_dev(&m.obs[0]).err(), Some(CovError::Empty));
    assert_eq!(cov.conditional_cov(&m.obs[0]).err(), Some(CovError::Empty));
    assert_eq!(cov.cross_cov(&m.g, &m.pts[2]).err(), Some(CovError::Empty));
    assert_eq!(cov.cross_cov(&m.obs[1], &m.g).err(), Some(CovError::Empty));
}
