// The results table, one implementation for both dimensions.
//
// 2D and 3D differ in which systems they run and in the geometry (SE2 vs SE3
// poses, a different reference cost and alignment). They do not differ in what a
// results table IS: one cell per system, minimized over interleaved rounds,
// validated against the common optimum, printed. Keeping two copies of that is
// how the Ceres iteration-0 discount and the "-" rule for an unclean first
// iteration each landed on one dimension and not the other.
//
// A dimension supplies its geometry through `Geometry`; everything below is
// written once.

use std::collections::BTreeSet;

/// What one system's run reports. The subprocess and in-process runners both
/// hand back this shape.
pub type Row<P> = (f64, f64, usize, Option<usize>, Option<f64>, Vec<P>);

pub struct Cell<P> {
    solve_ms: f64,
    first_iter_ms: f64,
    /// Attempts: accepted steps plus damping retries.
    iterations: usize,
    /// Accepted steps, where the system reports them separately.
    accepted: Option<usize>,
    /// t(2 iterations); `None` when it cannot be differenced against a clean
    /// t(1). The table prints the difference.
    full_ms: Option<f64>,
    poses: Vec<P>,
    cost: f64,
}

/// The geometry of one dimension: how to score a solution and how to compare
/// two of them.
pub trait Geometry {
    type Pose: Clone;
    fn cost(&self, poses: &[Self::Pose]) -> f64;
    fn aligned_rmse(a: &[Self::Pose], b: &[Self::Pose]) -> f64;
}

pub struct Table<'a, G: Geometry> {
    geometry: &'a G,
    cells: Vec<(String, Cell<G::Pose>)>,
    /// Reported under the table: a system that crashed, a note about a row.
    pub notes: BTreeSet<String>,
    /// A dataset where arael's f32 is KNOWN to stop short of the f64 solution --
    /// a measured single-precision floor, not a solver defect. Only there is an
    /// f32 row allowed to miss the gates, and only with this note against it.
    /// arael's f64 must converge everywhere, always.
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
    pub fn record(&mut self, label: &str, row: Row<G::Pose>) {
        let (solve_ms, first_iter_ms, iterations, accepted, full_ms, poses) = row;
        let cost = self.geometry.cost(&poses);
        if let Some((_, prev)) = self.cells.iter_mut().find(|(l, _)| l == label) {
            prev.solve_ms = prev.solve_ms.min(solve_ms);
            prev.first_iter_ms = prev.first_iter_ms.min(first_iter_ms);
            // t(2) must be minimized over rounds like t(1) is: full-iter is their
            // difference, and mixing a single sample with a minimum biases it
            // (badly enough to go negative).
            prev.full_ms = match (prev.full_ms, full_ms) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (a, b) => a.or(b),
            };
        } else {
            self.cells.push((
                label.to_string(),
                Cell { solve_ms, first_iter_ms, iterations, accepted, full_ms, poses, cost },
            ));
        }
    }

    /// Print the table, then validate it. A row counts as converged only when
    /// BOTH its cost is within 1% of the best AND its solution is within 5 cm
    /// rigid-aligned RMSE of the best: the cost surface has near-flat directions
    /// where a plateau under 1% above the optimum can sit meters away.
    ///
    /// Hard asserts: arael rows must converge, and at least one external system
    /// must agree on the optimum (an independent-implementation anchor).
    pub fn print(&self) {
        let best_idx = (0..self.cells.len())
            .min_by(|&i, &j| {
                self.cells[i].1.cost.partial_cmp(&self.cells[j].1.cost).unwrap()
            })
            .expect("no rows recorded");
        let best = self.cells[best_idx].1.cost;
        let best_poses = self.cells[best_idx].1.poses.clone();
        let converged = |c: &Cell<G::Pose>| {
            (c.cost - best) / best < 1e-2
                && G::aligned_rmse(&best_poses, &c.poses) < 0.05
        };

        // ms/iter divides the whole solve by every ATTEMPT, so it carries the
        // one-time setup amortized over however many iterations the solver took.
        // full-iter is what one COMPLETE iteration costs: min t(2 iters) - min
        // t(1 iter), taking the minima before the subtraction so a noisy pair
        // cannot difference into nonsense. It is blank where t(1) was not one
        // clean accepted iteration -- see probe::first_iter_ms.
        println!("\n{:<18} {:>10} {:>9} {:>10} {:>10} {:>12} {:>14}",
            "system", "total ms", "iters", "ms/iter", "full-iter", "1st-iter ms",
            "final cost");
        for (label, c) in &self.cells {
            let iters = match c.accepted {
                Some(a) => format!("{}({})", a, c.iterations),
                None => format!("{}", c.iterations),
            };
            let full = match c.full_ms {
                Some(two) if two > c.first_iter_ms => {
                    format!("{:.2}", two - c.first_iter_ms)
                }
                _ => "-".to_string(),
            };
            let miss = if converged(c) {
                String::new()
            } else {
                format!("  <- did not reach the common optimum (aligned RMSE {:.4} m)",
                    G::aligned_rmse(&best_poses, &c.poses))
            };
            println!("{:<18} {:>10.1} {:>9} {:>10.2} {:>10} {:>12} {:>14.4}{}",
                label, c.solve_ms, iters,
                c.solve_ms / c.iterations.max(1) as f64,
                full,
                crate::fmt1(c.first_iter_ms), c.cost, miss);
        }

        let mut notes = self.notes.clone();
        for (label, c) in &self.cells {
            if converged(c) || !label.starts_with("arael") {
                continue;
            }
            // arael's f64 must reach the optimum on every dataset. Its f32 may
            // fall short only where the dataset declares a measured
            // single-precision floor, and the note is then reported.
            match (label.ends_with("f32"), &self.f32_floor_note) {
                (true, Some(n)) => { notes.insert(format!("{}: {}", label, n)); }
                _ => panic!("{} failed to converge: {} vs best {} (aligned RMSE {:.4} m)",
                    label, c.cost, best, G::aligned_rmse(&best_poses, &c.poses)),
            }
        }
        let external_agree = self.cells.iter()
            .filter(|(l, c)| !l.starts_with("arael") && converged(c))
            .count();
        assert!(external_agree >= 1,
            "no external system confirms the best cost {} -- cannot validate", best);

        for note in &notes {
            println!("{:<18} {}", "", note);
        }
        let conv = self.cells.iter().filter(|(_, c)| converged(c)).count();
        println!("validation: {}/{} systems at the common optimum ({:.4}: cost within \
                  1%, aligned RMSE to best < 5 cm), anchored by {} external system(s)",
            conv, self.cells.len(), best, external_agree);
    }
}
