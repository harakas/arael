// Numeric rank and null-space basis (arael::rank).

use arael::model::{Jacobian, JacobianRow};
use arael::rank::{RankError, RankMethod, RankOptions};

fn row(entries: &[(u32, f64)]) -> JacobianRow<f64> {
    JacobianRow {
        constraint: 0,
        label: "test",
        residual: 0.0,
        entries: entries.to_vec(),
    }
}

fn jac(num_params: usize, rows: Vec<JacobianRow<f64>>) -> Jacobian<f64> {
    Jacobian { num_params, rows }
}

fn dense_opts() -> RankOptions {
    RankOptions { dense_cutoff: usize::MAX, ..Default::default() }
}

fn iter_opts() -> RankOptions {
    RankOptions { dense_cutoff: 0, ..Default::default() }
}

#[test]
fn dense_and_iterative_agree_on_dependent_rows() {
    // Three rows in six params; the third row is the sum of the first
    // two, so rank 2, nullity 4.
    let j = jac(6, vec![
        row(&[(0, 1.0)]),
        row(&[(1, 1.0)]),
        row(&[(0, 1.0), (1, 1.0)]),
    ]);
    let d = j.numeric_rank(&dense_opts()).unwrap();
    let it = j.numeric_rank(&iter_opts()).unwrap();
    assert_eq!(d.rank, 2);
    assert_eq!(d.nullity, 4);
    assert_eq!(it.rank, 2);
    assert_eq!(it.nullity, 4);
    // Params 2..5 are free singletons, so both paths split: one
    // 2-param component plus four free ones.
    assert_eq!(d.method, RankMethod::Components { count: 5, largest_n: 2 });
    assert_eq!(it.method, RankMethod::Components { count: 5, largest_n: 2 });
}

#[test]
fn reduces_rank_answers_the_probe_question() {
    let j = jac(6, vec![
        row(&[(0, 1.0)]),
        row(&[(1, 1.0)]),
    ]);
    for opts in [dense_opts(), iter_opts()] {
        let r = j.numeric_rank(&opts).unwrap();
        assert_eq!(r.nullity, 4);
        // A row inside the existing span does not reduce rank.
        assert!(!r.reduces_rank(&[(0, 2.0), (1, -3.0)]));
        // A row touching a free parameter does.
        assert!(r.reduces_rank(&[(2, 1.0)]));
        // A mixed row still constrains a free direction.
        assert!(r.reduces_rank(&[(0, 1.0), (3, 0.5)]));
    }
}

#[test]
fn non_finite_entry_is_an_error_not_a_panic() {
    let j = jac(3, vec![
        row(&[(0, 1.0)]),
        JacobianRow {
            constraint: 1,
            label: "bad_row",
            residual: 0.0,
            entries: vec![(1, f64::NAN)],
        },
    ]);
    match j.numeric_rank(&RankOptions::default()) {
        Err(RankError::NonFinite { label }) => assert_eq!(label, "bad_row"),
        other => panic!("expected NonFinite, got {:?}", other.map(|r| r.nullity)),
    }
}

#[test]
fn no_rows_means_everything_is_free() {
    let j = jac(5, vec![]);
    let r = j.numeric_rank(&RankOptions::default()).unwrap();
    assert_eq!(r.rank, 0);
    assert_eq!(r.nullity, 5);
    assert!(r.reduces_rank(&[(4, 1.0)]));
}

#[test]
fn thin_svd_null_basis_is_completed_when_m_less_than_n() {
    // 3 independent rows in 8 params: rank 3, nullity 5, but a thin
    // SVD only carries k = 3 right vectors -- the basis must be
    // completed to all 5 free directions.
    let j = jac(8, vec![
        row(&[(0, 1.0), (1, 0.5)]),
        row(&[(2, 1.0)]),
        row(&[(3, 1.0), (0, -0.25)]),
    ]);
    let r = j.numeric_rank(&dense_opts()).unwrap();
    assert_eq!(r.rank, 3);
    assert_eq!(r.nullity, 5);
    let (basis, k) = r.null_basis();
    assert_eq!(k, 5);
    let n = 8;
    // Orthonormality.
    for a in 0..k {
        for b in a..k {
            let dot: f64 = (0..n).map(|i| basis[a * n + i] * basis[b * n + i]).sum();
            let want = if a == b { 1.0 } else { 0.0 };
            assert!((dot - want).abs() < 1e-9, "gram[{},{}] = {}", a, b, dot);
        }
    }
    // The span test still answers correctly across the completion.
    assert!(!r.reduces_rank(&[(2, 5.0)]));
    assert!(r.reduces_rank(&[(4, 1.0)]));
    assert!(r.reduces_rank(&[(7, 1.0)]));
}

#[test]
fn random_sparse_problem_agrees_across_paths() {
    // 300 rows over the first 150 of 200 params; params 150..200 are
    // untouched, so nullity is at least 50. Deterministic LCG values.
    let mut state = 0x12345678u64;
    let mut next = move || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((state >> 33) as f64 / (1u64 << 31) as f64) - 0.5
    };
    let mut rows = Vec::new();
    for r in 0..300u32 {
        let a = (r * 7 % 150) as u32;
        let b = (r * 13 % 150) as u32;
        let entries = if a == b {
            vec![(a, 1.0 + next())]
        } else {
            vec![(a, 1.0 + next()), (b, next())]
        };
        rows.push(row(&entries));
    }
    let j = jac(200, rows);
    let d = j.numeric_rank(&dense_opts()).unwrap();
    let it = j.numeric_rank(&iter_opts()).unwrap();
    assert_eq!(d.nullity, it.nullity, "dense={} iter={}", d.nullity, it.nullity);
    assert!(d.nullity >= 50);
}

#[test]
fn warm_start_matches_cold() {
    let mut rows = Vec::new();
    for i in 0..40u32 {
        rows.push(row(&[(i, 1.0), (i + 1, -1.0)]));
    }
    let j = jac(80, rows);
    let opts = iter_opts();
    let cold = j.numeric_rank(&opts).unwrap();

    // Value-only change: scale every entry, same structure.
    let mut rows2 = Vec::new();
    for i in 0..40u32 {
        rows2.push(row(&[(i, 2.0), (i + 1, -2.0)]));
    }
    let j2 = jac(80, rows2);
    let warm_opts = RankOptions { sweeps: 1, ..iter_opts() };
    let warm = j2.numeric_rank_warm(&warm_opts, &cold).unwrap();
    assert_eq!(cold.nullity, warm.nullity);
}

#[test]
fn components_add_ranks_and_bases_exactly() {
    // Three islands with known ranks: a 4-chain (rank 3), a triangle
    // with one dependent row (rank 2), and two free params.
    let mut rows = Vec::new();
    for i in 0..3u32 {
        rows.push(row(&[(i, 1.0), (i + 1, -1.0)]));
    }
    rows.push(row(&[(4, 1.0), (5, 0.5)]));
    rows.push(row(&[(5, 1.0), (6, -0.5)]));
    rows.push(row(&[(4, 1.0), (5, 1.0), (6, -0.25)])); // dependent combo
    let j = jac(9, rows);
    for opts in [dense_opts(), iter_opts()] {
        let r = j.numeric_rank(&opts).unwrap();
        assert_eq!(r.rank, 5, "3 + 2");
        assert_eq!(r.nullity, 4, "1 + 1 + 2 free");
        assert_eq!(r.method, RankMethod::Components { count: 4, largest_n: 4 });
        // Basis orthonormality across the component embedding.
        let (basis, k) = r.null_basis();
        let n = 9;
        for a in 0..k {
            for b in a..k {
                let dot: f64 = (0..n).map(|i| basis[a * n + i] * basis[b * n + i]).sum();
                let want = if a == b { 1.0 } else { 0.0 };
                assert!((dot - want).abs() < 1e-9, "gram[{},{}] = {}", a, b, dot);
            }
        }
        // Every existing row lies in the row space.
        for rr in &j.rows {
            assert!(!r.reduces_rank(&rr.entries), "existing row must not reduce rank");
        }
        // Free directions are seen across components.
        assert!(r.reduces_rank(&[(7, 1.0)]));
        assert!(r.reduces_rank(&[(0, 1.0)]));
        // A cross-component row constrains a new relative direction.
        assert!(r.reduces_rank(&[(0, 1.0), (4, -1.0)]));
    }
}

#[test]
fn components_survive_extreme_scale_differences() {
    // A mm-scale island next to a km-scale island: per-component
    // spectra keep both decisions exact.
    let j = jac(4, vec![
        row(&[(0, 1e-6), (1, -1e-6)]),
        row(&[(2, 1e6), (3, 0.5e6)]),
    ]);
    for opts in [dense_opts(), iter_opts()] {
        let r = j.numeric_rank(&opts).unwrap();
        assert_eq!(r.rank, 2);
        assert_eq!(r.nullity, 2);
        assert!(!r.reduces_rank(&[(0, 2e-6), (1, -2e-6)]));
        assert!(r.reduces_rank(&[(0, 1.0), (1, 1.0)]));
    }
}

#[test]
fn warm_start_matches_cold_across_components() {
    // Two chains and some free params; warm re-rank after a value-only
    // change slices the previous basis per component.
    let mut rows = Vec::new();
    for i in 0..30u32 {
        rows.push(row(&[(i, 1.0), (i + 1, -1.0)]));
    }
    for i in 40..60u32 {
        rows.push(row(&[(i, 1.0), (i + 1, 2.0)]));
    }
    let rows2: Vec<_> = rows.iter().map(|r| {
        let mut r2 = row(&r.entries);
        for e in &mut r2.entries { e.1 *= 3.0; }
        r2
    }).collect();
    let j = jac(70, rows);
    let opts = iter_opts();
    let cold = j.numeric_rank(&opts).unwrap();
    assert!(matches!(cold.method, RankMethod::Components { .. }));

    let j2 = jac(70, rows2);
    let warm_opts = RankOptions { sweeps: 1, ..iter_opts() };
    let warm = j2.numeric_rank_warm(&warm_opts, &cold).unwrap();
    assert_eq!(cold.rank, warm.rank);
    assert_eq!(cold.nullity, warm.nullity);
}

#[test]
fn loose_chain_starts_at_the_nullity_floor() {
    // One connected component, 101 params, 50 rows: nullity >= 51, a
    // large fraction of n. The n - m block floor must open the block
    // at the null-space scale -- no growth steps.
    let mut rows = Vec::new();
    for i in 0..50u32 {
        rows.push(row(&[(2 * i, 1.0), (2 * i + 1, -1.0), (2 * i + 2, 0.5)]));
    }
    let j = jac(101, rows);
    let r = j.numeric_rank(&iter_opts()).unwrap();
    assert_eq!(r.rank, 50);
    assert_eq!(r.nullity, 51);
    // floor(51) + margin: the block opens above the nullity at once.
    assert!(matches!(r.method, RankMethod::Iterative { block, grew: 0 } if block > 51),
        "{:?}", r.method);
}
