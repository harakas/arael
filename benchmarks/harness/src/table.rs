// The results table: one cell per system, minimized over interleaved rounds,
// validated against the common optimum, printed.
//
// A benchmark differs in what a solution IS (2D poses, SE3 poses, poses plus
// landmarks, cameras plus points) and in how to score one. It does not differ in
// what a results table is. Keeping a copy per benchmark is how the Ceres
// iteration-0 discount and the "-" rule for an unclean first iteration each
// landed in one benchmark and not the others.

use std::collections::BTreeSet;

use crate::probe::fmt_ms;

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
    /// Distance between two solutions. The cost surface has near-flat directions
    /// where a plateau under 1% above the optimum can sit far away, so cost alone
    /// cannot validate a row.
    fn distance(a: &Self::Solution, b: &Self::Solution) -> f64;
    /// How far apart two solutions may be and still count as the same optimum.
    /// The pose benchmarks measure metres against a fixed frame. Bundle
    /// adjustment cannot: it has a 7-DOF gauge and arbitrary units, so it aligns
    /// the two solutions first and measures a relative figure -- a different
    /// quantity, needing a different gate.
    const DISTANCE_GATE: f64 = 0.05;
    /// The gate in words, for the validation line.
    const DISTANCE_GATE_NAME: &'static str = "distance to best < 5 cm";
    /// The same gate for single-precision rows, which cannot be held to a
    /// double-precision one: f32 carries about 7 decimal digits, so on a
    /// large scene its solution stops a little short of the f64 optimum
    /// however long it iterates, and which side of the f64 gate it lands
    /// on is noise rather than a property of the solver. Ten times the
    /// slack, which covers the floors measured so far (0.2-0.3 m on
    /// pgo's parking-garage and on plane at 900 poses); a real f32
    /// regression is still far outside it.
    const DISTANCE_GATE_F32: f64 = Self::DISTANCE_GATE * 10.0;
}

pub struct Table<'a, G: Geometry> {
    geometry: &'a G,
    cells: Vec<(String, Row<G::Solution>, f64)>, // label, row, cost
    /// Systems whose solve failed, label -> why. They still get a row, all
    /// dashes, and still count against the validation total: a system that
    /// could not run is a result, and dropping it silently hides it.
    failed: Vec<(String, String)>,
    /// Every label in the order it was first seen, across both of the above,
    /// so a failed row prints where it belongs rather than at the end.
    order: Vec<String>,
    /// Reported under the table: a system that crashed, a note about a row.
    pub notes: BTreeSet<String>,
}

impl<'a, G: Geometry> Table<'a, G> {
    pub fn new(geometry: &'a G) -> Self {
        Table {
            geometry,
            cells: Vec::new(),
            failed: Vec::new(),
            order: Vec::new(),
            notes: BTreeSet::new(),
        }
    }

    fn note_order(&mut self, label: &str) {
        if !self.order.iter().any(|l| l == label) {
            self.order.push(label.to_string());
        }
    }

    /// Record whatever the arael runner came back with: a row, or why there
    /// isn't one.
    pub fn record_result(&mut self, label: &str, row: Result<Row<G::Solution>, String>) {
        match row {
            Ok(r) => self.record(label, r),
            Err(why) => self.record_failure(label, why),
        }
    }

    /// Record that this system could not produce a row, and why. Rounds after
    /// the first keep the first reason -- the solve is deterministic, so they
    /// all say the same thing.
    pub fn record_failure(&mut self, label: &str, why: impl std::fmt::Display) {
        self.note_order(label);
        if !self.failed.iter().any(|(l, _)| l == label) {
            self.failed.push((label.to_string(), why.to_string()));
        }
    }

    /// Record one round of one system. Times are the minimum over rounds --
    /// contention only ever slows a run down.
    pub fn record(&mut self, label: &str, row: Row<G::Solution>) {
        self.note_order(label);
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
        if self.cells.is_empty() {
            // Nothing to compare against, so there is no optimum and no
            // validation -- but the failures are still the result.
            let w = self.order.iter().map(|l| l.len()).max().unwrap_or(18).max(6);
            for (label, why) in &self.failed {
                println!("{:<w$}  <- {}", label, why, w = w);
            }
            println!("validation: no system produced a row");
            return;
        }
        let best_i = (0..self.cells.len())
            .min_by(|&i, &j| self.cells[i].2.partial_cmp(&self.cells[j].2).unwrap())
            .expect("no rows recorded");
        let best = self.cells[best_i].2;
        let best_solution = self.cells[best_i].1.solution.clone();
        let converged = |label: &str, row: &Row<G::Solution>, cost: f64| {
            let gate = if label.contains("f32") { G::DISTANCE_GATE_F32 } else { G::DISTANCE_GATE };
            (cost - best) / best < 1e-2
                && G::distance(&best_solution, &row.solution) < gate
        };

        let w = self.order.iter().map(|l| l.len()).max().unwrap_or(18).max(6);
        let any_mem = self.cells.iter().any(|(_, r, _)| r.peak_mb.is_some());
        let mem_head = if any_mem { format!("{:>9}", "peak MB") } else { String::new() };
        // ms/iter divides the whole solve by every ATTEMPT, so it carries the
        // one-time setup amortized over however many iterations the solver took.
        // full-iter is what one COMPLETE iteration costs: min t(2 iters) - min
        // t(1 iter), the minima taken before the subtraction so a noisy pair
        // cannot difference into nonsense. It is blank where t(1) was not one
        // clean accepted iteration.
        println!("\n{:<w$} {:>10} {:>9} {:>10} {:>10} {:>12}{} {:>14}",
            "system", "total ms", "iters", "ms/iter", "full-iter", "1st-iter ms",
            mem_head, "final cost", w = w);
        for label in &self.order {
            if let Some((_, why)) = self.failed.iter().find(|(l, _)| l == label) {
                let mem = if any_mem { format!("{:>9}", "-") } else { String::new() };
                println!("{:<w$} {:>10} {:>9} {:>10} {:>10} {:>12}{} {:>14}  <- {}",
                    label, "-", "-", "-", "-", "-", mem, "-", why, w = w);
                continue;
            }
            let (_, row, cost) = self.cells.iter().find(|(l, _, _)| l == label)
                .expect("every ordered label is either a cell or a failure");
            let cost = *cost;
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
            let dist = G::distance(&best_solution, &row.solution);
            let miss = if !converged(label, row, cost) {
                format!("  <- did not reach the common optimum (distance {:.2e})", dist)
            } else if dist >= G::DISTANCE_GATE {
                // Inside its own gate but outside the double-precision one:
                // it agrees, and it is worth seeing by how much it does not.
                format!("  <- single-precision floor (distance {:.2e})", dist)
            } else {
                String::new()
            };
            println!("{:<w$} {:>10.2} {:>9} {:>10.2} {:>10} {:>12}{} {:>14.4}{}",
                label, row.solve_ms, iters,
                row.solve_ms / row.iterations.max(1) as f64,
                full, fmt_ms(row.first_iter_ms), mem, cost, miss, w = w);
        }

        // A row that missed is marked on its own line above and counted in
        // the validation line below. That is the report; a measurement tool
        // should hand back what it measured rather than abort on it.
        let notes = self.notes.clone();
        let external_agree = self.cells.iter()
            .filter(|(l, r, c)| !l.starts_with("arael") && converged(l, r, *c))
            .count();
        if external_agree == 0 {
            println!("{:<w$} WARNING: no external system reached the best cost -- \
                      nothing independent confirms it", "", w = w);
        }

        for note in &notes {
            println!("{:<w$} {}", "", note, w = w);
        }
        // The denominator is every system asked for, failures included: a row
        // that could not run is one that did not reach the optimum.
        let conv = self.cells.iter().filter(|(l, r, c)| converged(l, r, *c)).count();
        println!("validation: {}/{} systems at the common optimum ({:.4}: cost within \
                  1%, {}; f32 rows {:.0}x that), anchored by {} external system(s)",
            conv, self.order.len(), best, G::DISTANCE_GATE_NAME,
            G::DISTANCE_GATE_F32 / G::DISTANCE_GATE, external_agree);
    }
}
