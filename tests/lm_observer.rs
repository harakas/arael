// The iteration observer: LmConfig carries an LmObserver that the solve
// loop calls once per damped attempt (accepted, rejected, and
// factorization-failed alike). Break stops the solve with
// ObserverTerminated, keeping the current best state. Observers are
// cloned per solve like the lambda driver, so shared state rides in
// Rc/Arc handles.

use arael::simple_lm::RootProblem;
use std::cell::RefCell;
use std::ops::ControlFlow;
use std::rc::Rc;

use arael::model::{Param, SelfBlock};
use arael::simple_lm::{LmConfig, LmIter, LmObserver, LmProblem, LmStatus};

// Overdetermined on purpose: the optimum has NONZERO cost, so the
// noise-floor convergence test can fire (a cost of exactly zero would
// walk the lambda ladder to its ceiling instead).
#[arael::model]
#[arael(constraint(hb, {
    [(m.x - 3.0) * 2.0, m.x - 3.2, (m.y - m.x) * 0.7]
}))]
struct M {
    x: Param<f64>,
    y: Param<f64>,
    hb: SelfBlock<M>,
}

#[arael::model]
#[arael(root)]
struct W {
    items: std::vec::Vec<M>,
}

fn build() -> W {
    W { items: vec![M { x: Param::new(0.0), y: Param::new(10.0), hb: SelfBlock::new() }] }
}

/// Records every callback through a shared handle (the observer itself
/// is cloned per solve, so the log must live outside it).
#[derive(Clone)]
struct Spy {
    log: Rc<RefCell<Vec<(usize, bool, Vec<f64>)>>>,
    stop_after: Option<usize>,
}

impl LmObserver<f64> for Spy {
    fn on_iteration(&mut self, it: &LmIter<'_, f64>) -> ControlFlow<()> {
        self.log.borrow_mut().push((it.iter, it.accepted, it.params.to_vec()));
        match self.stop_after {
            Some(n) if it.iter >= n => ControlFlow::Break(()),
            _ => ControlFlow::Continue(()),
        }
    }
}

#[test]
fn the_observer_sees_every_attempt_and_the_final_state() {
    let log = Rc::new(RefCell::new(Vec::new()));
    let cfg = LmConfig::<f64>::conservative()
        .with_observer(Spy { log: log.clone(), stop_after: None });

    let mut w = build();
    let r = w.solve_dense(&cfg).unwrap();

    let log = log.borrow();
    // One callback per counted iteration, in order.
    assert_eq!(log.len(), r.iterations, "one callback per attempt");
    for (k, (iter, ..)) in log.iter().enumerate() {
        assert_eq!(*iter, k + 1, "attempts arrive in order");
    }
    let n_accepted = log.iter().filter(|(_, a, _)| *a).count();
    assert_eq!(n_accepted, r.accepted_iterations);
    // The last accepted callback's params are the solution.
    let last = log.iter().rev().find(|(_, a, _)| *a).expect("an accepted step");
    assert_eq!(last.2, r.x, "observer params snapshot matches the result");
}

#[test]
fn break_stops_the_solve_and_keeps_the_best_state() {
    let log = Rc::new(RefCell::new(Vec::new()));
    let cfg = LmConfig::<f64>::conservative()
        .with_observer(Spy { log: log.clone(), stop_after: Some(2) });

    let mut w = build();
    let r = w.solve_dense(&cfg).unwrap();

    assert_eq!(r.status, LmStatus::ObserverTerminated);
    assert!(r.status.is_success(), "a deliberate stop is not a failure");
    assert_eq!(r.iterations, 2, "stopped where the observer said");
    assert_eq!(log.borrow().len(), 2);
    assert!(r.end_cost < r.start_cost, "the accepted work is kept");
    // The model holds the observer-stopped state, not the optimum.
    let mut back = Vec::new();
    w.serialize(&mut back);
    assert_eq!(back, r.x);
}

#[test]
fn a_closure_is_an_observer() {
    let count = Rc::new(RefCell::new(0usize));
    let c = count.clone();
    let cfg = LmConfig::<f64>::conservative()
        .with_observer(move |_it: &LmIter<'_, f64>| {
            *c.borrow_mut() += 1;
            ControlFlow::Continue(())
        });

    let mut w = build();
    let r = w.solve_dense(&cfg).unwrap();
    assert!(r.status.is_success());
    assert_eq!(*count.borrow(), r.iterations);
}

#[test]
fn a_good_enough_check_stops_on_cost() {
    let cfg = LmConfig::<f64>::conservative()
        .with_observer(|it: &LmIter<'_, f64>| {
            if it.accepted && it.new_cost < 1.0 {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        });

    let mut w = build();
    let r = w.solve_dense(&cfg).unwrap();
    assert_eq!(r.status, LmStatus::ObserverTerminated);
    assert!(r.end_cost < 1.0, "stopped once good enough: {}", r.end_cost);
}

#[test]
fn a_reused_config_starts_each_solve_with_a_fresh_clone() {
    // A per-solve counter INSIDE the observer: if the second solve saw
    // the first solve's mutated copy, it would break immediately.
    #[derive(Clone)]
    struct CountDown(usize);
    impl LmObserver<f64> for CountDown {
        fn on_iteration(&mut self, _: &LmIter<'_, f64>) -> ControlFlow<()> {
            if self.0 == 0 {
                return ControlFlow::Break(());
            }
            self.0 -= 1;
            ControlFlow::Continue(())
        }
    }

    let cfg = LmConfig::<f64>::conservative().with_observer(CountDown(3));
    let mut w1 = build();
    let r1 = w1.solve_dense(&cfg).unwrap();
    let mut w2 = build();
    let r2 = w2.solve_dense(&cfg).unwrap();
    assert_eq!(r1.iterations, 4, "3 continues + the breaking attempt");
    assert_eq!(r2.iterations, 4, "the second solve starts from a fresh clone");
}
