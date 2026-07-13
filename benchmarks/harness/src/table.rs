// The results table: one cell per system, minimized over interleaved rounds,
// validated against the common optimum, printed.
//
// A benchmark differs in what a solution IS (2D poses, SE3 poses, poses plus
// landmarks, cameras plus points) and in how to score one. It does not differ in
// what a results table is. Keeping a copy per benchmark is how the Ceres
// iteration-0 discount and the "-" rule for an unclean first iteration each
// landed in one benchmark and not the others.

use std::collections::BTreeSet;

use crate::probe::fmt1;

/// What one system's run reports.
pub struct Row<S> {
    pub solve_ms: f64,
    /// One iteration plus the setup the later ones do not pay again. NaN when
    /// that iteration was not one clean accepted step -- see
    /// [`crate::probe::first_iter_ms`].
    pub first_iter_ms: f64,
    /// Attempts: accepted steps plus damping retries, each of which costs a
    /// factorization.
    pub iterations: usize,
    /// Accepted steps, where the system reports them apart from attempts.
    pub accepted: Option<usize>,
    /// t(2 iterations). The table prints its difference against t(1); `None`
    /// when the pair cannot be differenced.
    pub full_ms: Option<f64>,
    /// Peak resident set, measured in a process of its own -- VmHWM is a
    /// process-wide high-water mark, so solvers sharing a process contaminate
    /// each other's peak.
    pub peak_mb: Option<f64>,
    pub solution: S,
}

impl<S> Row<S> {
    /// A system that reports only its outer iteration count.
    pub fn new(solve_ms: f64, first_iter_ms: f64, iterations: usize, solution: S) -> Self {
        Row {
            solve_ms,
            first_iter_ms,
            iterations,
            accepted: None,
            full_ms: None,
            peak_mb: None,
            solution,
        }
    }

    pub fn accepted(mut self, accepted: usize) -> Self {
        self.accepted = Some(accepted);
        self
    }

    pub fn full_ms(mut self, full_ms: Option<f64>) -> Self {
        self.full_ms = full_ms;
        self
    }

    pub fn peak_mb(mut self, peak_mb: Option<f64>) -> Self {
        self.peak_mb = peak_mb;
        self
    }
}

/// The geometry of one benchmark: how to score a solution, and how far apart two
/// solutions are.
pub trait Geometry {
    type Solution: Clone;
    /// The one reference cost function every system's solution is scored by, so
    /// the costs in the table are directly comparable.
    fn cost(&self, solution: &Self::Solution) -> f64;
    /// Distance between two solutions, in metres. The cost surface has near-flat
    /// directions where a plateau under 1% above the optimum can sit metres away,
    /// so cost alone cannot validate a row.
    fn distance(a: &Self::Solution, b: &Self::Solution) -> f64;
}

pub struct Table<'a, G: Geometry> {
    geometry: &'a G,
    cells: Vec<(String, Row<G::Solution>, f64)>, // label, row, cost
    /// Reported under the table: a system that crashed, a note about a row.
    pub notes: BTreeSet<String>,
    /// A problem where arael's f32 is KNOWN to stop short of the f64 solution --
    /// a measured single-precision floor, not a solver defect. Only there may an
    /// f32 row miss the gates, and only with this note against it. arael's f64
    /// must converge everywhere, always.
    f32_floor_note: Option<String>,
}

impl<'a, G: Geometry> Table<'a, G> {
    pub fn new(geometry: &'a G) -> Self {
        Table { geometry, cells: Vec::new(), notes: BTreeSet::new(), f32_floor_note: None }
    }

    pub fn with_f32_floor(geometry: &'a G, note: Option<&str>) -> Self {
        let mut t = Table::new(geometry);
        t.f32_floor_note = note.map(str::to_string);
        t
    }

    /// Record one round of one system. Times are the minimum over rounds --
    /// contention only ever slows a run down.
    pub fn record(&mut self, label: &str, row: Row<G::Solution>) {
        let cost = self.geometry.cost(&row.solution);
        if let Some((_, prev, _)) = self.cells.iter_mut().find(|(l, _, _)| l == label) {
            prev.solve_ms = prev.solve_ms.min(row.solve_ms);
            prev.first_iter_ms = prev.first_iter_ms.min(row.first_iter_ms);
            // t(2) must be minimized over rounds like t(1) is: full-iter is their
            // difference, and mixing a single sample with a minimum biases it
            // (badly enough to go negative).
            prev.full_ms = match (prev.full_ms, row.full_ms) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (a, b) => a.or(b),
            };
            prev.peak_mb = prev.peak_mb.or(row.peak_mb);
        } else {
            self.cells.push((label.to_string(), row, cost));
        }
    }

    /// Attach a peak-memory figure to a row already recorded. Memory is measured
    /// in a pass of its own, after the timing rounds.
    pub fn set_peak_mb(&mut self, label: &str, peak_mb: f64) {
        if let Some((_, row, _)) = self.cells.iter_mut().find(|(l, _, _)| l == label) {
            row.peak_mb = Some(peak_mb);
        }
    }

    /// Print the table, then validate it. A row counts as converged only when
    /// BOTH its cost is within 1% of the best AND its solution is within 5 cm of
    /// the best.
    ///
    /// Hard asserts: arael's f64 must converge, and at least one external system
    /// must agree on the optimum -- an independent-implementation anchor.
    pub fn print(&self) {
        let best_i = (0..self.cells.len())
            .min_by(|&i, &j| self.cells[i].2.partial_cmp(&self.cells[j].2).unwrap())
            .expect("no rows recorded");
        let best = self.cells[best_i].2;
        let best_solution = self.cells[best_i].1.solution.clone();
        let converged = |row: &Row<G::Solution>, cost: f64| {
            (cost - best) / best < 1e-2 && G::distance(&best_solution, &row.solution) < 0.05
        };

        let any_mem = self.cells.iter().any(|(_, r, _)| r.peak_mb.is_some());
        let mem_head = if any_mem { format!("{:>9}", "peak MB") } else { String::new() };
        // ms/iter divides the whole solve by every ATTEMPT, so it carries the
        // one-time setup amortized over however many iterations the solver took.
        // full-iter is what one COMPLETE iteration costs: min t(2 iters) - min
        // t(1 iter), the minima taken before the subtraction so a noisy pair
        // cannot difference into nonsense. It is blank where t(1) was not one
        // clean accepted iteration.
        println!("\n{:<18} {:>10} {:>9} {:>10} {:>10} {:>12}{} {:>14}",
            "system", "total ms", "iters", "ms/iter", "full-iter", "1st-iter ms",
            mem_head, "final cost");
        for (label, row, cost) in &self.cells {
            let iters = match row.accepted {
                Some(a) => format!("{}({})", a, row.iterations),
                None => format!("{}", row.iterations),
            };
            let full = match row.full_ms {
                Some(two) if two > row.first_iter_ms => {
                    format!("{:.2}", two - row.first_iter_ms)
                }
                _ => "-".to_string(),
            };
            let mem = if any_mem {
                match row.peak_mb {
                    Some(mb) => format!("{:>9.1}", mb),
                    None => format!("{:>9}", "-"),
                }
            } else {
                String::new()
            };
            let miss = if converged(row, *cost) {
                String::new()
            } else {
                format!("  <- did not reach the common optimum (distance {:.4} m)",
                    G::distance(&best_solution, &row.solution))
            };
            println!("{:<18} {:>10.1} {:>9} {:>10.2} {:>10} {:>12}{} {:>14.4}{}",
                label, row.solve_ms, iters,
                row.solve_ms / row.iterations.max(1) as f64,
                full, fmt1(row.first_iter_ms), mem, cost, miss);
        }

        let mut notes = self.notes.clone();
        for (label, row, cost) in &self.cells {
            if converged(row, *cost) || !label.starts_with("arael") {
                continue;
            }
            match (label.ends_with("f32"), &self.f32_floor_note) {
                (true, Some(n)) => { notes.insert(format!("{}: {}", label, n)); }
                _ => panic!("{} failed to converge: {} vs best {} (distance {:.4} m)",
                    label, cost, best, G::distance(&best_solution, &row.solution)),
            }
        }
        let external_agree = self.cells.iter()
            .filter(|(l, r, c)| !l.starts_with("arael") && converged(r, *c))
            .count();
        assert!(external_agree >= 1,
            "no external system confirms the best cost {} -- cannot validate", best);

        for note in &notes {
            println!("{:<18} {}", "", note);
        }
        let conv = self.cells.iter().filter(|(_, r, c)| converged(r, *c)).count();
        println!("validation: {}/{} systems at the common optimum ({:.4}: cost within \
                  1%, distance to best < 5 cm), anchored by {} external system(s)",
            conv, self.cells.len(), best, external_agree);
    }
}
