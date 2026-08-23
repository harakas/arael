//! Blocker analysis for DOF-rejected constraints.
//!
//! When a candidate constraint is rejected because its Jacobian rows
//! already lie in the row-span of existing constraints, identify the
//! minimum-size set of existing constraints whose removal would
//! unblock the candidate.
//!
//! Operates at constraint-block granularity: each constraint is a
//! block of one or more Jacobian rows (a DistancePP is 1 row, a
//! CoincidentPP is 2 rows, a DistanceConcentric is 3 rows, etc.),
//! and the candidate may itself be multi-row. A subset S of existing
//! constraints is a blocker iff, with all rows from S removed, at
//! least one candidate row falls out of the remaining row-span.
//!
//! The search is exhaustive at constraint granularity for sizes
//! k = 1, 2, 3, with cutoffs on existing-constraint count to keep
//! cost bounded. For typical sketches (1-50 constraints) the search
//! is instantaneous; for larger sketches k=3 may be skipped and the
//! report flagged as truncated.

use std::collections::{HashMap, HashSet};
use arael::model::Jacobian;
use nalgebra::DMatrix;

/// Result of a blocker analysis.
pub struct BlockerReport {
    /// Size of the minimum blocker set. 0 means no blocker was found
    /// within the enumeration cutoff.
    pub minimum_size: usize,
    /// Enumerated minimum-size blocker sets (capped at MAX_SETS).
    /// Each inner Vec is a set of existing-constraint CIDs whose
    /// removal unblocks the candidate. Empty if nothing found within
    /// the cutoff.
    pub sets: Vec<Vec<u32>>,
    /// Total number of existing constraints considered.
    pub existing_count: usize,
    /// True if enumeration was cut off before reaching k=3 due to
    /// existing-constraint count exceeding the per-k limit.
    pub truncated: bool,
    /// True if the existing-constraint Jacobian itself has internal
    /// rank deficiency (linear dependencies among existing
    /// constraints that dedup did not collapse). Hints that blocker
    /// sets may be approximate.
    pub existing_redundant: bool,
    /// Per-step timing and counter breakdown. Always populated;
    /// callers decide whether to print it.
    pub stats: BlockerStats,
}

/// Instrumentation data for a single `analyze` call.
#[derive(Default)]
pub struct BlockerStats {
    /// Wall time for the whole `analyze` call.
    pub total_ms: f64,
    /// Time spent building the union-find and filtering existing
    /// constraints to the candidate's connected component.
    pub component_prune_ms: f64,
    /// Existing constraint count before component pruning.
    pub existing_before_prune: usize,
    /// Existing constraint count after component pruning (input to
    /// the brute-force search).
    pub existing_after_prune: usize,
    /// Time spent verifying the rejection condition (every candidate
    /// row projected onto rowspan(A)).
    pub rejection_check_ms: f64,
    /// Number of candidate rows analysed (after zero-row filtering).
    pub candidate_rows: usize,
    /// Number of existing constraint rows in the pruned problem.
    pub existing_rows: usize,
    /// Per-`k` search stats in order (k=1, k=2, k=3).
    pub per_k: Vec<BlockerKStats>,
}

/// Breakdown for one `k` level of the brute-force search.
pub struct BlockerKStats {
    pub k: usize,
    /// True if this level was skipped due to the pool-size cutoff.
    pub skipped: bool,
    /// Number of subsets enumerated (subset_check calls).
    pub subsets_tested: usize,
    /// Number of subsets accepted as a minimum blocker set.
    pub blockers_found: usize,
    /// Wall time for this level.
    pub time_ms: f64,
}

// Relative tolerance for treating a projection residual as zero.
const ZERO_TOL_REL: f64 = 1e-8;
// Maximum number of equivalent minimum-size blocker sets to report.
const MAX_SETS: usize = 5;
// Skip k=2 search if more existing constraints than this.
const K2_LIMIT: usize = 200;
// Skip k=3 search if more existing constraints than this.
const K3_LIMIT: usize = 40;
/// Dense-path budget: flops of the rowspan SVD (m * n * min(m, n)).
const SVD_FLOP_LIMIT: usize = 50_000_000;
/// Sparse path: verify at most this many certificate-named candidates.
const CERT_VERIFY_CAP: usize = 16;
/// Sparse path: relative certificate weight that names a constraint.
const CERT_SUPPORT_TOL: f64 = 1e-6;

/// Run blocker analysis on a post-apply Jacobian.
///
/// `candidate_cids` names the CIDs of the just-added constraint block.
/// All other CIDs in `jac.rows` are treated as existing constraints.
/// Returns `None` if the rejection condition is not actually present
/// (e.g. the candidate's rows do add rank, so there is nothing to
/// explain).
pub fn analyze(
    jac: &Jacobian<f64>,
    candidate_cids: &HashSet<u32>,
) -> Option<BlockerReport> {
    analyze_with_limit(jac, candidate_cids, SVD_FLOP_LIMIT)
}

/// [`analyze`] with an explicit dense-path work budget; above it the
/// sparse certificate path runs instead. Tests force one path or the
/// other with 0 / usize::MAX.
pub fn analyze_with_limit(
    jac: &Jacobian<f64>,
    candidate_cids: &HashSet<u32>,
    dense_limit: usize,
) -> Option<BlockerReport> {
    let t_total = web_time::Instant::now();
    let mut stats = BlockerStats::default();

    let n = jac.num_params;
    if n == 0 { return None; }

    // Union-find over params: two params are in the same component
    // if some (non-zero) row touches both. A constraint outside the
    // candidate's component(s) cannot affect whether candidate rows
    // lie in rowspan -- its rows have zero entries on all candidate
    // params. Pruning those constraints up front drops the per-k
    // brute-force cost proportionally to the disjoint sketch size.
    let t_prune = web_time::Instant::now();
    let mut uf = UnionFind::new(n);
    for row in &jac.rows {
        let mut first: Option<usize> = None;
        for &(j, v) in &row.entries {
            if v == 0.0 { continue; }
            let j = j as usize;
            match first {
                None => first = Some(j),
                Some(a) => uf.union(a, j),
            }
        }
    }
    // Collect components touched by candidate rows.
    let mut cand_components: HashSet<usize> = HashSet::new();
    for row in &jac.rows {
        if !candidate_cids.contains(&row.constraint) { continue; }
        for &(j, v) in &row.entries {
            if v != 0.0 { cand_components.insert(uf.find(j as usize)); }
        }
    }

    let in_candidate_component = |row: &arael::model::JacobianRow<f64>,
                                   uf: &mut UnionFind| -> bool {
        row.entries.iter().any(|&(j, v)|
            v != 0.0 && cand_components.contains(&uf.find(j as usize)))
    };

    // Partition rows into candidate vs existing, grouped by CID.
    // Preserve first-seen order for existing CIDs so enumeration is
    // deterministic across runs. Skip all-zero rows (drift-suppressed
    // regularisers, guarded constraints that didn't activate): they
    // don't contribute to row-span and would only inflate m_a.
    // Existing rows outside the candidate's component are also
    // skipped -- they can't possibly be blockers.
    let mut existing_cids_ordered: Vec<u32> = Vec::new();
    let mut existing_rows_by_cid: HashMap<u32, Vec<Vec<(u32, f64)>>> = HashMap::new();
    let mut candidate_sparse: Vec<Vec<(u32, f64)>> = Vec::new();
    let mut total_existing_cids: HashSet<u32> = HashSet::new();
    for row in &jac.rows {
        let nonzero = row.entries.iter().any(|&(_, v)| v != 0.0);
        if !nonzero { continue; }
        if candidate_cids.contains(&row.constraint) {
            candidate_sparse.push(row.entries.clone());
            continue;
        }
        total_existing_cids.insert(row.constraint);
        if !in_candidate_component(row, &mut uf) { continue; }
        if !existing_rows_by_cid.contains_key(&row.constraint) {
            existing_cids_ordered.push(row.constraint);
        }
        existing_rows_by_cid.entry(row.constraint).or_default().push(row.entries.clone());
    }
    stats.existing_before_prune = total_existing_cids.len();
    stats.existing_after_prune = existing_cids_ordered.len();
    stats.component_prune_ms = t_prune.elapsed().as_secs_f64() * 1000.0;
    if candidate_sparse.is_empty() || existing_cids_ordered.is_empty() {
        return None;
    }

    // Flatten existing into parallel (row, cid) vectors.
    let mut a_sparse: Vec<Vec<(u32, f64)>> = Vec::new();
    let mut a_row_cid: Vec<u32> = Vec::new();
    for &cid in &existing_cids_ordered {
        for row in &existing_rows_by_cid[&cid] {
            a_sparse.push(row.clone());
            a_row_cid.push(cid);
        }
    }
    let m_a = a_sparse.len();
    stats.candidate_rows = candidate_sparse.len();
    stats.existing_rows = m_a;

    // The exhaustive path below runs a dense O(m * n * min(m, n))
    // rowspan SVD on the frame thread; above the budget the sparse
    // certificate path answers k = 1 instead.
    if m_a.saturating_mul(n).saturating_mul(m_a.min(n)) > dense_limit {
        return analyze_sparse(
            n, &candidate_sparse, &existing_cids_ordered, &a_sparse, &a_row_cid,
            stats, t_total,
        );
    }

    // Densify for the exhaustive path.
    let to_dense = |entries: &Vec<(u32, f64)>| -> Vec<f64> {
        let mut dense = vec![0.0f64; n];
        for &(j, v) in entries { dense[j as usize] = v; }
        dense
    };
    let candidate_rows: Vec<Vec<f64>> = candidate_sparse.iter().map(to_dense).collect();
    let a_rows: Vec<Vec<f64>> = a_sparse.iter().map(to_dense).collect();

    // Compute SVD of A once. V_r (n x r) is an orthonormal basis for
    // rowspan(A), where r = rank(A). Project everything into that
    // basis: B = A V_r (m_a x r) holds the coordinates of each
    // existing row, y_i = V_r^T c_i (length r) the coordinates of
    // each candidate row. All subsequent subset rank checks run in
    // R^r instead of R^n, which is the optimisation win: r is
    // bounded by min(m_a, n) and is usually much smaller than n
    // (one row per constraint, params often outnumber residuals).
    //
    // Correctness: c_i is in rowspan(A - S) iff y_i is in rowspan of
    // the corresponding subset of B, since rowspan(A - S) lives
    // entirely in the V_r subspace and V_r is injective restricted
    // to that subspace.
    let t_rej = web_time::Instant::now();
    let a_mat = rows_to_matrix(&a_rows, n);
    let svd_a = a_mat.clone().svd(false, true);
    let vt_a = svd_a.v_t.as_ref().expect("V^T computed");
    let svs_a = &svd_a.singular_values;
    let max_sv_a = svs_a.iter().copied().fold(0.0f64, f64::max);
    let rank_tol_a = 1e-12 * max_sv_a.max(1.0);
    let a_rank = svs_a.iter().filter(|&&s| s > rank_tol_a).count();
    // Build V_r (n x r) row-major by taking the first r rows of V^T
    // and transposing conceptually. In the loop below we just index
    // vt_a.row(k) for k < a_rank (each row is V_r column k).
    // Project existing rows: B[j, k] = <V_r col k, a_rows[j]> = <vt_a row k, a_rows[j]>.
    let mut b_rows: Vec<Vec<f64>> = Vec::with_capacity(m_a);
    for j in 0..m_a {
        let aj = &a_rows[j];
        let mut bj = vec![0.0f64; a_rank];
        for k in 0..a_rank {
            let row_k = vt_a.row(k);
            let mut s = 0.0f64;
            for i in 0..n { s += row_k[i] * aj[i]; }
            bj[k] = s;
        }
        b_rows.push(bj);
    }
    // Project candidate rows and verify the rejection condition:
    // every c_i must lie in rowspan(A) (else projection residual is
    // non-zero and there's no blocker to find).
    let scale = max_sv_a.max(1e-30);
    let rej_tol = ZERO_TOL_REL * scale;
    let mut y_rows: Vec<Vec<f64>> = Vec::with_capacity(candidate_rows.len());
    for c in &candidate_rows {
        // y = V_r^T c
        let mut y = vec![0.0f64; a_rank];
        // Reconstruction for residual check: c_proj = V_r y.
        let mut c_proj = vec![0.0f64; n];
        for k in 0..a_rank {
            let row_k = vt_a.row(k);
            let mut s = 0.0f64;
            for i in 0..n { s += row_k[i] * c[i]; }
            y[k] = s;
            for i in 0..n { c_proj[i] += s * row_k[i]; }
        }
        let mut resid_sq = 0.0f64;
        for i in 0..n { let d = c[i] - c_proj[i]; resid_sq += d * d; }
        if resid_sq.sqrt() > rej_tol { return None; }
        y_rows.push(y);
    }
    stats.rejection_check_ms = t_rej.elapsed().as_secs_f64() * 1000.0;

    let existing_redundant = a_rank < m_a;
    let n_ex = existing_cids_ordered.len();

    // Subset check: given a set of CIDs to remove, return true iff at
    // least one candidate row is no longer in rowspan(B - S). Builds
    // B_rest (small, m' x r), runs one SVD per subset, then shares
    // V^T across all candidate projections.
    let check_blocker = |to_remove: &HashSet<u32>| -> bool {
        let mut rest: Vec<&Vec<f64>> = Vec::with_capacity(m_a);
        for (row, cid) in b_rows.iter().zip(a_row_cid.iter()) {
            if !to_remove.contains(cid) {
                rest.push(row);
            }
        }
        if rest.is_empty() {
            // No existing rows left; any non-zero candidate is now
            // out of the (empty) rowspan.
            return true;
        }
        let b_rest = rows_refs_to_matrix(&rest, a_rank);
        let svd = b_rest.svd(false, true);
        let vt = match svd.v_t.as_ref() { Some(v) => v, None => return true };
        let svs = &svd.singular_values;
        let max_sv = svs.iter().copied().fold(0.0f64, f64::max);
        let rank_tol = 1e-12 * max_sv.max(1.0);
        let local_tol = ZERO_TOL_REL * max_sv.max(1e-30);
        for y in &y_rows {
            // Residual of projecting y onto rowspan(b_rest).
            let mut y_proj = vec![0.0f64; a_rank];
            for i in 0..svs.len() {
                if svs[i] <= rank_tol { continue; }
                let row_i = vt.row(i);
                let mut coeff = 0.0f64;
                for j in 0..a_rank { coeff += row_i[j] * y[j]; }
                for j in 0..a_rank { y_proj[j] += coeff * row_i[j]; }
            }
            let mut resid_sq = 0.0f64;
            for j in 0..a_rank { let d = y[j] - y_proj[j]; resid_sq += d * d; }
            if resid_sq.sqrt() > local_tol { return true; }
        }
        false
    };

    let mut truncated = false;
    let mut minimum: Option<(usize, Vec<Vec<u32>>)> = None;
    for k in 1..=3 {
        let t_k = web_time::Instant::now();
        let mut kstat = BlockerKStats {
            k, skipped: false, subsets_tested: 0, blockers_found: 0, time_ms: 0.0,
        };
        if (k == 2 && n_ex > K2_LIMIT) || (k == 3 && n_ex > K3_LIMIT) {
            kstat.skipped = true;
            kstat.time_ms = t_k.elapsed().as_secs_f64() * 1000.0;
            stats.per_k.push(kstat);
            truncated = true;
            break;
        }
        let mut found: Vec<Vec<u32>> = Vec::new();
        each_combination(&existing_cids_ordered, k, |combo| {
            kstat.subsets_tested += 1;
            let set: HashSet<u32> = combo.iter().copied().collect();
            if check_blocker(&set) {
                kstat.blockers_found += 1;
                found.push(combo.to_vec());
            }
            found.len() < MAX_SETS
        });
        kstat.time_ms = t_k.elapsed().as_secs_f64() * 1000.0;
        stats.per_k.push(kstat);
        if !found.is_empty() {
            minimum = Some((k, found));
            break;
        }
    }
    stats.total_ms = t_total.elapsed().as_secs_f64() * 1000.0;

    Some(match minimum {
        Some((k, sets)) => BlockerReport {
            minimum_size: k, sets,
            existing_count: n_ex,
            truncated: false,
            existing_redundant,
            stats,
        },
        None => BlockerReport {
            minimum_size: 0,
            sets: Vec::new(),
            existing_count: n_ex,
            truncated,
            existing_redundant,
            stats,
        },
    })
}

/// The certificate path for sketches too big for the dense SVD: the
/// candidate row is a linear combination of existing rows; the
/// regularised minimum-norm coefficients name the rows carrying the
/// dependency, and each named constraint is verified as a k = 1
/// blocker with one sparse rank test. k >= 2 is not explored here
/// (the dense path skips it beyond K2_LIMIT anyway).
#[allow(clippy::too_many_arguments)]
fn analyze_sparse(
    n: usize,
    candidate_rows: &[Vec<(u32, f64)>],
    existing_cids_ordered: &[u32],
    a_rows: &[Vec<(u32, f64)>],
    a_row_cid: &[u32],
    mut stats: BlockerStats,
    t_total: web_time::Instant,
) -> Option<BlockerReport> {
    use arael::model::JacobianRow;
    let t_rej = web_time::Instant::now();
    let mk_jac = |skip_cid: Option<u32>| -> Jacobian<f64> {
        Jacobian {
            num_params: n,
            rows: a_rows.iter().zip(a_row_cid.iter())
                .filter(|&(_, &cid)| Some(cid) != skip_cid)
                .map(|(r, &cid)| JacobianRow {
                    constraint: cid,
                    label: "",
                    residual: 0.0,
                    entries: r.clone(),
                })
                .collect(),
        }
    };
    let ex_jac = mk_jac(None);
    let opts = arael::rank::RankOptions::default();
    let rr = ex_jac.numeric_rank(&opts).ok()?;
    let existing_redundant = rr.rank < a_rows.len();
    // The rejection condition: every candidate row lies in the
    // existing rowspan; otherwise there is no blocker to find.
    for c in candidate_rows {
        if rr.reduces_rank(c) {
            return None;
        }
    }

    // Certificate support: constraints carrying the dependency.
    let mut strength: HashMap<u32, f64> = HashMap::new();
    for c in candidate_rows {
        let lam = ex_jac.rowspan_certificate(c, 1e-10).ok()?;
        let max = lam.iter().fold(0.0f64, |a, l| a.max(l.abs())).max(1e-30);
        for (l, &cid) in lam.iter().zip(a_row_cid.iter()) {
            let rel = l.abs() / max;
            if rel > CERT_SUPPORT_TOL {
                let e = strength.entry(cid).or_insert(0.0);
                if rel > *e {
                    *e = rel;
                }
            }
        }
    }
    stats.rejection_check_ms = t_rej.elapsed().as_secs_f64() * 1000.0;

    let mut support: Vec<(u32, f64)> = strength.into_iter().collect();
    support.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let mut truncated = support.len() > CERT_VERIFY_CAP;
    support.truncate(CERT_VERIFY_CAP);

    let t_k = web_time::Instant::now();
    let mut kstat = BlockerKStats {
        k: 1, skipped: false, subsets_tested: 0, blockers_found: 0, time_ms: 0.0,
    };
    let mut found: Vec<Vec<u32>> = Vec::new();
    for &(cid, _) in &support {
        if found.len() >= MAX_SETS {
            truncated = true;
            break;
        }
        kstat.subsets_tested += 1;
        let sub = mk_jac(Some(cid));
        let Ok(sr) = sub.numeric_rank(&opts) else { continue };
        if candidate_rows.iter().any(|c| sr.reduces_rank(c)) {
            kstat.blockers_found += 1;
            found.push(vec![cid]);
        }
    }
    // Report in first-seen constraint order, like the dense path.
    found.sort_by_key(|set| {
        existing_cids_ordered.iter().position(|&c| c == set[0]).unwrap_or(usize::MAX)
    });
    kstat.time_ms = t_k.elapsed().as_secs_f64() * 1000.0;
    stats.per_k.push(kstat);
    stats.total_ms = t_total.elapsed().as_secs_f64() * 1000.0;

    let minimum_size = if found.is_empty() { 0 } else { 1 };
    let truncated = truncated || found.is_empty();
    Some(BlockerReport {
        minimum_size,
        sets: found,
        existing_count: existing_cids_ordered.len(),
        truncated,
        existing_redundant,
        stats,
    })
}

fn rows_to_matrix(rows: &[Vec<f64>], n: usize) -> DMatrix<f64> {
    let m = rows.len();
    if m == 0 { return DMatrix::zeros(0, n); }
    let mut data = Vec::with_capacity(m * n);
    for r in rows { data.extend_from_slice(r); }
    DMatrix::from_row_slice(m, n, &data)
}

fn rows_refs_to_matrix(rows: &[&Vec<f64>], n: usize) -> DMatrix<f64> {
    let m = rows.len();
    if m == 0 { return DMatrix::zeros(0, n); }
    let mut data = Vec::with_capacity(m * n);
    for r in rows { data.extend_from_slice(r); }
    DMatrix::from_row_slice(m, n, &data)
}


/// Path-compressing union-find over integer indices [0, n).
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self { parent: (0..n).collect(), rank: vec![0; n] }
    }
    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }
    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb { return; }
        let (lo, hi) = if self.rank[ra] < self.rank[rb] { (ra, rb) } else { (rb, ra) };
        self.parent[lo] = hi;
        if self.rank[lo] == self.rank[hi] { self.rank[hi] += 1; }
    }
}

/// Call `f` with every size-`k` combination of `items` in lexicographic
/// order. Stops early when `f` returns false.
fn each_combination<F: FnMut(&[u32]) -> bool>(items: &[u32], k: usize, mut f: F) {
    if k == 0 { f(&[]); return; }
    if k > items.len() { return; }
    let mut current: Vec<u32> = Vec::with_capacity(k);
    fn recur<F: FnMut(&[u32]) -> bool>(
        items: &[u32], k: usize, start: usize,
        current: &mut Vec<u32>, f: &mut F,
    ) -> bool {
        if current.len() == k {
            return f(current);
        }
        let need = k - current.len();
        let max_start = items.len() + 1 - need;
        for i in start..max_start {
            current.push(items[i]);
            let cont = recur(items, k, i + 1, current, f);
            current.pop();
            if !cont { return false; }
        }
        true
    }
    recur(items, k, 0, &mut current, &mut f);
}

#[cfg(test)]
mod tests {
    use super::*;
    use arael::model::JacobianRow;

    fn row(cid: u32, _n: usize, entries: &[(u32, f64)], residual: f64) -> JacobianRow<f64> {
        JacobianRow {
            constraint: cid,
            label: "test",
            residual,
            entries: entries.to_vec(),
        }
    }

    fn jac(n: usize, rows: Vec<JacobianRow<f64>>) -> Jacobian<f64> {
        Jacobian { num_params: n, rows }
    }

    #[test]
    fn single_row_duplicate() {
        // Two single-row constraints, second copies the first's row.
        // The candidate is CID=20, existing CIDs 10 (original). Add
        // one more existing (CID=11) that's different. Candidate
        // duplicates CID=10's row. Expect minimum size 1, set {10}.
        let j = jac(3, vec![
            row(10, 3, &[(0, 1.0), (1, -1.0)], 0.0), // existing: x0 - x1
            row(11, 3, &[(2, 1.0)], 0.0),             // existing: x2
            row(20, 3, &[(0, 2.0), (1, -2.0)], 0.0), // candidate: 2*(x0 - x1), colinear with CID 10
        ]);
        let candidate: HashSet<u32> = [20u32].into_iter().collect();
        let r = analyze(&j, &candidate).expect("analysis runs");
        assert_eq!(r.minimum_size, 1);
        assert_eq!(r.sets.len(), 1);
        assert_eq!(r.sets[0], vec![10u32]);
    }

    #[test]
    fn multi_row_candidate_requires_size_two() {
        // Candidate has 2 rows (1,0) and (0,1). Three existing
        // single-row constraints (1,0), (0,1), (1,1) jointly span
        // the 2D row-space but any single removal still leaves 2D
        // spanning rows, so candidate stays implied. Minimum blocker
        // set is size 2 with three equivalent alternatives.
        let j = jac(2, vec![
            row(10, 2, &[(0, 1.0)], 0.0),
            row(11, 2, &[(1, 1.0)], 0.0),
            row(12, 2, &[(0, 1.0), (1, 1.0)], 0.0),
            row(20, 2, &[(0, 1.0)], 0.0), // candidate row 1
            row(20, 2, &[(1, 1.0)], 0.0), // candidate row 2
        ]);
        let candidate: HashSet<u32> = [20u32].into_iter().collect();
        let r = analyze(&j, &candidate).expect("analysis runs");
        assert_eq!(r.minimum_size, 2);
        use std::collections::BTreeSet;
        let got: BTreeSet<BTreeSet<u32>> = r.sets.iter()
            .map(|s| s.iter().copied().collect::<BTreeSet<u32>>())
            .collect();
        let expected: BTreeSet<BTreeSet<u32>> = [
            [10u32, 11].into_iter().collect(),
            [10u32, 12].into_iter().collect(),
            [11u32, 12].into_iter().collect(),
        ].into_iter().collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn size_one_with_alternatives() {
        // Candidate row (1, 1). Existing (1,0) and (0,1) jointly span
        // the plane, so candidate is implied. Remove either single
        // existing row and the remaining 1D span no longer contains
        // (1, 1). Minimum size 1 with two alternatives.
        let j = jac(2, vec![
            row(10, 2, &[(0, 1.0)], 0.0),
            row(11, 2, &[(1, 1.0)], 0.0),
            row(20, 2, &[(0, 1.0), (1, 1.0)], 0.0), // candidate
        ]);
        let candidate: HashSet<u32> = [20u32].into_iter().collect();
        let r = analyze(&j, &candidate).expect("analysis runs");
        assert_eq!(r.minimum_size, 1);
        use std::collections::BTreeSet;
        let sets: BTreeSet<Vec<u32>> = r.sets.into_iter().collect();
        assert_eq!(sets, [vec![10u32], vec![11u32]].into_iter().collect());
    }

    #[test]
    fn not_a_rejection() {
        // Candidate adds a fresh independent direction. analyze should
        // return None (no blocker to find).
        let j = jac(3, vec![
            row(10, 3, &[(0, 1.0)], 0.0),
            row(20, 3, &[(1, 1.0)], 0.0), // candidate, independent
        ]);
        let candidate: HashSet<u32> = [20u32].into_iter().collect();
        assert!(analyze(&j, &candidate).is_none());
    }
}
