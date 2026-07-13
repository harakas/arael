// Attempt counting for factrs, shared by every benchmark that runs it.
//
// factrs's LM keeps its damping retry loop INSIDE step(): a rejected step
// multiplies lambda and re-solves the damped system -- another full
// factorization -- without ever returning. Its iteration count and its
// on_step observers therefore see accepted steps only, and the retries are
// invisible while still being paid for.
//
// Counting the linear solves recovers them: factrs issues exactly one solve
// per attempt, so solves = accepted + retries. The work is still factrs's own
// CholeskySolver -- this only wraps it to count. Gauss-Newton has no retry
// loop, so there its attempts always equal its steps.

use std::sync::atomic::{AtomicUsize, Ordering};

use factrs::dtype;
use factrs::containers::Values;
use factrs::linear::{CholeskySolver, LinearSolver};
use factrs::optimizers::OptObserver;
use faer23::sparse::SparseColMatRef;
use faer23::{Mat, MatRef};

static STEPS: AtomicUsize = AtomicUsize::new(0);
static SOLVES: AtomicUsize = AtomicUsize::new(0);

/// Accepted steps: factrs notifies observers once per accepted step().
pub struct StepCounter;

impl OptObserver for StepCounter {
    fn on_step(&self, _values: &Values, _time: i64) {
        STEPS.fetch_add(1, Ordering::Relaxed);
    }
}

/// factrs's own Cholesky solver, wrapped to count the systems it is handed.
#[derive(Default)]
pub struct CountingSolver(CholeskySolver);

impl LinearSolver for CountingSolver {
    fn solve_symmetric(
        &mut self,
        a: SparseColMatRef<usize, dtype>,
        b: MatRef<dtype>,
    ) -> Mat<dtype> {
        SOLVES.fetch_add(1, Ordering::Relaxed);
        self.0.solve_symmetric(a, b)
    }

    fn solve_lst_sq(
        &mut self,
        a: SparseColMatRef<usize, dtype>,
        b: MatRef<dtype>,
    ) -> Mat<dtype> {
        SOLVES.fetch_add(1, Ordering::Relaxed);
        self.0.solve_lst_sq(a, b)
    }
}

/// Steps and solves counted so far, for differencing across one solve.
pub fn counts() -> (usize, usize) {
    (STEPS.load(Ordering::Relaxed), SOLVES.load(Ordering::Relaxed))
}

/// Accepted steps and total attempts between two [`counts`] readings.
pub fn since(before: (usize, usize)) -> (usize, usize) {
    let (steps, solves) = counts();
    (steps - before.0, solves - before.1)
}
