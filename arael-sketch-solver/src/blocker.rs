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
    /// Certificate path: building the shared rowspan factor.
    pub factor_ms: f64,
    /// Certificate path: certificate factor + solves.
    pub certificate_ms: f64,
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
/// Sparse path: verify at most this many certificate-named candidates.
const CERT_VERIFY_CAP: usize = 16;
/// Sparse path: relative certificate weight that names a constraint.
const CERT_SUPPORT_TOL: f64 = 1e-6;
/// Regulariser of the shared rowspan factor.
const CERT_EPS: f64 = 1e-10;
/// Relative residual above which a row counts as outside a rowspan:
/// in-span residuals are ~sqrt(eps), out-of-span ones are the
/// out-of-span fraction of the row's norm.
const CERT_OUT_TOL: f64 = 1e-4;
/// Backstop on factor solves across the whole expansion.
const CERT_SOLVE_BUDGET: usize = 256;

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
    // The certificate path wins at every measured size (the dense
    // k = 1 sweep runs one SVD per constraint); the dense path stays
    // as the exhaustive oracle behind [`analyze_with_limit`].
    analyze_with_limit(jac, candidate_cids, 0)
}

/// [`analyze`] with an explicit dense-path work budget; above it the
/// sparse certificate path runs instead. Production uses 0 (always
/// certificate); tests force the dense oracle with usize::MAX.
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

/// The certificate path: every query runs against one shared
/// regularised factor of the existing rows (arael::rank::
/// RowspanAnalyzer). The candidate's rejection residual, the
/// certificate naming the constraints that carry the dependency, and
/// every subset test are closed-form in that factor: removing a
/// subset's rows is a low-rank downdate, so the reduced-system
/// residual is the base solution plus a tiny Woodbury system over the
/// subset's rows -- no per-subset factorization or rank computation.
/// k = 1..3 search by support expansion (a minimal set has a member
/// in the base certificate's support; the rest of the set lies in the
/// reduced system's support, recursively), under the same K2 / K3
/// limits as the dense path. `existing_redundant` is not computed on
/// this path (always false): it has no cheap equivalent and only
/// softens the hint's wording.
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
    use arael::rank::RowspanAnalyzer;
    use std::collections::BTreeSet;

    let t_rej = web_time::Instant::now();
    let ex_jac = Jacobian {
        num_params: n,
        rows: a_rows.iter().zip(a_row_cid.iter())
            .map(|(r, &cid)| JacobianRow {
                constraint: cid,
                label: "",
                residual: 0.0,
                entries: r.clone(),
            })
            .collect(),
    };
    let t_factor = web_time::Instant::now();
    let ana = RowspanAnalyzer::new(&ex_jac, CERT_EPS).ok()?;
    stats.factor_ms = t_factor.elapsed().as_secs_f64() * 1000.0;

    // Per candidate row: the factor solution z, the certificate
    // lambda = A_n z, and the in-span residual. The regularised
    // least-squares objective against a row set B is
    // eps * c^T (B^T B + eps I)^{-1} c: ~eps for a row in span(B),
    // ~|c_out|^2 for a row with an out-of-span component.
    struct Cand {
        lam: Vec<f64>,
        czc: f64,
        norm: f64,
    }
    let t_cert = web_time::Instant::now();
    let mut cands: Vec<Cand> = Vec::new();
    for c in candidate_rows {
        let z = ana.solve_row(c);
        let czc = ana.row_dot(c, &z).max(0.0);
        let norm = ana.row_norm(c).max(1e-30);
        if (CERT_EPS * czc).sqrt() > CERT_OUT_TOL * norm {
            // Not in the existing rowspan: the rejection was not a
            // row-dependency, nothing to explain.
            return None;
        }
        let lam = ana.matvec(&z);
        cands.push(Cand { lam, czc, norm });
    }
    stats.certificate_ms = t_cert.elapsed().as_secs_f64() * 1000.0;
    stats.rejection_check_ms = t_rej.elapsed().as_secs_f64() * 1000.0;

    // Rows of each constraint, and the support aggregation: the cids
    // whose rows carry weight in a certificate, strongest first.
    let mut rows_of: HashMap<u32, Vec<usize>> = HashMap::new();
    for (i, &cid) in a_row_cid.iter().enumerate() {
        rows_of.entry(cid).or_default().push(i);
    }
    let mut truncated = false;
    let mut support_from = |lams: &[Vec<f64>], excluded: &BTreeSet<u32>| -> Vec<u32> {
        let mut strength: HashMap<u32, f64> = HashMap::new();
        for lam in lams {
            let max = lam.iter().enumerate()
                .filter(|(i, _)| !excluded.contains(&a_row_cid[*i]))
                .fold(0.0f64, |a, (_, l)| a.max(l.abs()))
                .max(1e-30);
            for (l, &cid) in lam.iter().zip(a_row_cid.iter()) {
                if excluded.contains(&cid) {
                    continue;
                }
                let rel = l.abs() / max;
                if rel > CERT_SUPPORT_TOL {
                    let e = strength.entry(cid).or_insert(0.0);
                    if rel > *e {
                        *e = rel;
                    }
                }
            }
        }
        let mut support: Vec<(u32, f64)> = strength.into_iter().collect();
        support.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        if support.len() > CERT_VERIFY_CAP {
            truncated = true;
            support.truncate(CERT_VERIFY_CAP);
        }
        support.into_iter().map(|(c, _)| c).collect()
    };

    // Lazy per-row Woodbury pieces: w_r = M a_r and u_r = A_n w_r.
    let mut w_cache: HashMap<usize, (Vec<f64>, Vec<f64>)> = HashMap::new();
    let mut solves = 0usize;
    let piece = |r: usize, w_cache: &mut HashMap<usize, (Vec<f64>, Vec<f64>)>,
                     solves: &mut usize| {
        if !w_cache.contains_key(&r) {
            *solves += 1;
            let w = ana.solve_row(&a_rows[r]);
            let u = ana.matvec(&w);
            w_cache.insert(r, (w, u));
        }
    };

    // Woodbury test of a cid set S with rows R: with G[i][j] =
    // a_i^T M a_j and t = lambda*[R], the reduced objective is
    // eps * (c^T M c + t^T (I - G)^{-1} t). Blocked when some
    // candidate's reduced residual exceeds the span tolerance. Also
    // returns the reduced certificates for the expansion:
    // lambda' = lambda* + sum_r beta_r u_r.
    let verify = |set: &BTreeSet<u32>,
                      w_cache: &mut HashMap<usize, (Vec<f64>, Vec<f64>)>,
                      solves: &mut usize| -> (bool, Vec<Vec<f64>>) {
        let rows: Vec<usize> = set.iter()
            .flat_map(|cid| rows_of.get(cid).cloned().unwrap_or_default())
            .collect();
        for &r in &rows {
            piece(r, w_cache, solves);
        }
        let k = rows.len();
        // G is symmetric with eigenvalues strictly below 1 (eps > 0).
        let mut g = vec![0.0f64; k * k];
        for (i, &ri) in rows.iter().enumerate() {
            let wi = &w_cache[&ri].0;
            for (j, &rj) in rows.iter().enumerate() {
                g[i * k + j] = ana.row_dot(&a_rows[rj], wi);
            }
        }
        let mut blocked = false;
        let mut reduced: Vec<Vec<f64>> = Vec::with_capacity(cands.len());
        for cand in &cands {
            let t: Vec<f64> = rows.iter().map(|&r| cand.lam[r]).collect();
            // Solve (I - G) beta = t, small dense with partial pivoting.
            let mut m = vec![0.0f64; k * k];
            for i in 0..k {
                for j in 0..k {
                    m[i * k + j] = (if i == j { 1.0 } else { 0.0 }) - g[i * k + j];
                }
            }
            let mut beta = t.clone();
            solve_small(&mut m, &mut beta, k);
            let tb: f64 = t.iter().zip(&beta).map(|(a, b)| a * b).sum();
            let obj = (CERT_EPS * (cand.czc + tb.max(0.0))).max(0.0);
            if obj.sqrt() > CERT_OUT_TOL * cand.norm {
                blocked = true;
            }
            let mut lam = cand.lam.clone();
            for (bi, &r) in beta.iter().zip(rows.iter()) {
                let u = &w_cache[&r].1;
                for (li, ui) in lam.iter_mut().zip(u.iter()) {
                    *li += bi * ui;
                }
            }
            reduced.push(lam);
        }
        (blocked, reduced)
    };

    let base_lams: Vec<Vec<f64>> = cands.iter().map(|c| c.lam.clone()).collect();
    let n_ex = existing_cids_ordered.len();
    let pos = |cid: u32| existing_cids_ordered.iter().position(|&c| c == cid).unwrap_or(usize::MAX);

    // k = 1..3 by support expansion; each level only runs when the
    // previous found nothing, mirroring the dense path's levels.
    let mut minimum: Option<(usize, Vec<Vec<u32>>)> = None;
    let none: BTreeSet<u32> = BTreeSet::new();
    // Frontier: non-blocking subsets of size k-1 with the reduced
    // certificates of the system they leave behind.
    let mut frontier: Vec<(BTreeSet<u32>, Vec<Vec<f64>>)> = vec![(none, base_lams)];
    'levels: for k in 1..=3 {
        let t_k = web_time::Instant::now();
        let mut kstat = BlockerKStats {
            k, skipped: false, subsets_tested: 0, blockers_found: 0, time_ms: 0.0,
        };
        if (k == 2 && n_ex > K2_LIMIT) || (k == 3 && n_ex > K3_LIMIT) || solves > CERT_SOLVE_BUDGET {
            kstat.skipped = true;
            kstat.time_ms = t_k.elapsed().as_secs_f64() * 1000.0;
            stats.per_k.push(kstat);
            truncated = true;
            break 'levels;
        }
        let mut found: Vec<Vec<u32>> = Vec::new();
        let mut next_frontier: Vec<(BTreeSet<u32>, Vec<Vec<f64>>)> = Vec::new();
        let mut tested: BTreeSet<BTreeSet<u32>> = BTreeSet::new();
        for (base, lams) in &frontier {
            for cid in support_from(lams, base) {
                let mut set = base.clone();
                set.insert(cid);
                if set.len() != k || !tested.insert(set.clone()) {
                    continue;
                }
                kstat.subsets_tested += 1;
                let (blocked, reduced) = verify(&set, &mut w_cache, &mut solves);
                if blocked {
                    kstat.blockers_found += 1;
                    let mut sorted: Vec<u32> = set.iter().copied().collect();
                    sorted.sort_by_key(|&c| pos(c));
                    found.push(sorted);
                } else {
                    next_frontier.push((set, reduced));
                }
            }
        }
        kstat.time_ms = t_k.elapsed().as_secs_f64() * 1000.0;
        stats.per_k.push(kstat);
        if !found.is_empty() {
            // First MAX_SETS in the dense path's enumeration order.
            found.sort_by_key(|set| set.iter().map(|&c| pos(c)).collect::<Vec<_>>());
            found.truncate(MAX_SETS);
            minimum = Some((k, found));
            break 'levels;
        }
        frontier = next_frontier;
    }
    stats.total_ms = t_total.elapsed().as_secs_f64() * 1000.0;

    Some(match minimum {
        Some((k, sets)) => BlockerReport {
            minimum_size: k,
            sets,
            existing_count: existing_cids_ordered.len(),
            truncated: false,
            existing_redundant: false,
            stats,
        },
        None => BlockerReport {
            minimum_size: 0,
            sets: Vec::new(),
            existing_count: existing_cids_ordered.len(),
            truncated,
            existing_redundant: false,
            stats,
        },
    })
}

/// Gaussian elimination with partial pivoting for the tiny Woodbury
/// systems (a handful of rows). `m` is k x k row-major, `b` the rhs,
/// solved in place.
fn solve_small(m: &mut [f64], b: &mut [f64], k: usize) {
    for col in 0..k {
        let mut p = col;
        for r in col + 1..k {
            if m[r * k + col].abs() > m[p * k + col].abs() {
                p = r;
            }
        }
        if p != col {
            for j in 0..k {
                m.swap(col * k + j, p * k + j);
            }
            b.swap(col, p);
        }
        let d = m[col * k + col];
        if d.abs() < 1e-300 {
            continue;
        }
        for r in col + 1..k {
            let f = m[r * k + col] / d;
            if f == 0.0 {
                continue;
            }
            for j in col..k {
                m[r * k + j] -= f * m[col * k + j];
            }
            b[r] -= f * b[col];
        }
    }
    for col in (0..k).rev() {
        let d = m[col * k + col];
        if d.abs() < 1e-300 {
            b[col] = 0.0;
            continue;
        }
        let mut s = b[col];
        for j in col + 1..k {
            s -= m[col * k + j] * b[j];
        }
        b[col] = s / d;
    }
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
