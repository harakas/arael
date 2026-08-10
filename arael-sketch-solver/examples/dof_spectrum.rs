// Dump the column-normalised singular-value spectrum of a sketch's
// Jacobian exactly as compute_dof sees it: the rank cut, the gap it
// keys on, and the cluster boundaries. The numbers here choose the
// shift and margins of the iterative rank method (SKETCH.md phase 1a).
//
// Usage: cargo run -r -p arael-sketch-solver --example dof_spectrum -- <sketch.json>...

use arael::model::JacobianModel;
use arael::simple_lm::RootProblem;
use arael_sketch_solver::Sketch;

// Mirror of compute_dof's rank_from_svs: ratio-gap search below 1% of
// the max, values floored at max*1e-20, fallback to counting < 1e-15
// when no gap beats 1e3. Returns (cut, best_gap).
fn rank_cut(sorted: &[f64]) -> (usize, f64) {
    let max_sv = sorted.last().copied().unwrap_or(0.0);
    let upper_bound = max_sv * 0.01;
    let floor = max_sv * 1e-20;
    let mut best_gap = 0.0f64;
    let mut best_cut = 0;
    for i in 0..sorted.len().saturating_sub(1) {
        let lo = sorted[i].max(floor);
        let hi = sorted[i + 1].max(floor);
        if lo > upper_bound { break; }
        let gap = hi / lo;
        if gap > best_gap {
            best_gap = gap;
            best_cut = i + 1;
        }
    }
    if best_gap < 1e3 {
        best_cut = sorted.iter().filter(|&&v| v < 1e-15).count();
    }
    (best_cut, best_gap)
}

fn main() {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    if paths.is_empty() {
        eprintln!("usage: dof_spectrum <sketch.json>...");
        std::process::exit(2);
    }
    for path in paths {
        let json = std::fs::read_to_string(&path).expect("read sketch file");
        let mut sketch: Sketch = serde_json::from_str(&json).expect("parse sketch json");

        sketch.prepare_expr_constraints();
        sketch.update_tangent_flags();
        sketch.update_perpendicular_flags();
        sketch.update_line_dir_flags();
        let saved_drift = sketch.drift_isigma;
        sketch.drift_isigma = 0.0;
        let mut params = Vec::new();
        sketch.serialize(&mut params);
        let n = params.len();

        let t = std::time::Instant::now();
        let mut jacobian = sketch.calc_jacobian(&params);
        sketch.drift_isigma = saved_drift;
        jacobian.rows.retain(|r| r.label != "range");
        let m = jacobian.num_residuals();
        let t_jac = t.elapsed();

        let t = std::time::Instant::now();
        let mut svs: Vec<f64> = jacobian.singular_values_column_normalised();
        let t_svd = t.elapsed();
        svs.sort_by(|a: &f64, b| a.partial_cmp(b).unwrap());

        let (cut, best_gap) = rank_cut(&svs);
        let rank = svs.len() - cut;
        let dof = n.saturating_sub(rank);
        let max_sv = svs.last().copied().unwrap_or(0.0);

        println!("== {}", path);
        println!("m={} n={} jacobian={:?} svd={:?}", m, n, t_jac, t_svd);
        println!("dof={} rank={} cut={} best_gap={:.3e} max_sv={:.6e}", dof, rank, cut, best_gap, max_sv);
        if cut > 0 && cut < svs.len() {
            let last_zero = svs[cut - 1];
            let first_real = svs[cut];
            println!("cluster: largest-zero={:.3e} ({:.1e} of max)  smallest-real={:.3e} ({:.1e} of max)  separation={:.3e}",
                last_zero, last_zero / max_sv, first_real, first_real / max_sv, first_real / last_zero.max(max_sv * 1e-20));
        }

        // The spectrum around the cut, ascending.
        let lo = cut.saturating_sub(8);
        let hi = (cut + 8).min(svs.len());
        for i in lo..hi {
            let side = if i < cut { "zero" } else { "real" };
            println!("  sv[{:4}] = {:.6e}  rel={:.3e}  [{}]", i, svs[i], svs[i] / max_sv, side);
        }

        // Decade histogram of the whole spectrum, relative to max.
        let mut buckets = std::collections::BTreeMap::new();
        for &s in &svs {
            let rel = s / max_sv;
            let dec = if rel <= 0.0 { i32::MIN } else { rel.log10().floor() as i32 };
            *buckets.entry(dec).or_insert(0usize) += 1;
        }
        print!("decades:");
        for (dec, count) in buckets.iter().rev() {
            if *dec == i32::MIN {
                print!("  [=0]x{}", count);
            } else {
                print!("  [1e{}]x{}", dec, count);
            }
        }
        println!();
    }
}
