//! Isolated f32 accuracy of a single Schur reduction, no solve loop.
//!
//! Builds a SLAM-shaped H -- large pose self-blocks, many small pose-landmark
//! couplings -- reduces it once in f32 and once in f64, and reports how far the
//! f32 reduced system S drifts from the f64 one. This measures the reduction's
//! own rounding, without the LM trajectory that muddies a converged-cost
//! comparison.

use arael_faer::bsc::{SparseBlockColMat, SymbolicSparseBlockColMat};
use arael_faer::schur::{schur_reduce, schur_symbolic, SchurContext};

struct Lcg(u64);
impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f64 / (1u64 << 31) as f64) - 1.0
    }
}

/// P poses (6-wide) in an odometry chain, then P landmarks (3-wide); landmark l
/// is seen by poses [l, l+W). Returns the block partition and the upper-triangle
/// cell list (block row <= block col).
fn structure(p: usize, w: usize) -> (Vec<usize>, Vec<(usize, usize)>) {
    let mut part = vec![0usize];
    for _ in 0..p {
        part.push(part.last().unwrap() + 6);
    }
    for _ in 0..p {
        part.push(part.last().unwrap() + 3);
    }
    let mut cells = Vec::new();
    for i in 0..p {
        cells.push((i, i));
        if i + 1 < p {
            cells.push((i, i + 1));
        }
    }
    for l in 0..p {
        cells.push((p + l, p + l));
        for i in l..(l + w).min(p) {
            cells.push((i, p + l));
        }
    }
    (part, cells)
}

/// Reduce a SLAM-shaped H (pose self-blocks ~big, landmark self-blocks ~med,
/// couplings ~coup) at scalar type $t, eliminating the landmarks, and return
/// the reduced S (dense, upper) and rhs cast up to f64.
macro_rules! reduce_at {
    ($name:ident, $t:ty) => {
        fn $name(
            part: &[usize],
            cells: &[(usize, usize)],
            p: usize,
            big: f64,
            med: f64,
            coup: f64,
            seed: u64,
        ) -> (Vec<f64>, Vec<f64>, usize) {
            let anchors: Vec<(usize, usize)> =
                cells.iter().map(|&(r, c)| (part[r], part[c])).collect();
            let (sym, _) = SymbolicSparseBlockColMat::from_scalar_coords(
                part.to_vec(),
                part.to_vec(),
                anchors.len(),
                |k| anchors[k],
            );
            let mut m = SparseBlockColMat::<usize, $t>::zeroed(sym);
            let sm = m.symbolic().clone();
            let mut rng = Lcg(seed);
            for &(r, c) in cells {
                let b = sm.col_range(c).find(|&b| sm.blk_row(b) == r).unwrap();
                let (rows, cols) = sm.block_dims(b);
                let range = sm.val_range(b);
                let tile = &mut m.vals_mut()[range];
                if r == c {
                    // SPD self-block: scale*I + small symmetric noise, upper-only
                    let scale = if r < p { big } else { med };
                    for j in 0..cols {
                        for i in 0..=j {
                            let v = if i == j { scale + rng.unit().abs() } else { rng.unit() };
                            tile[i + j * rows] = v as $t;
                        }
                    }
                } else {
                    for v in tile.iter_mut() {
                        *v = (coup * rng.unit()) as $t;
                    }
                }
            }
            let n = *part.last().unwrap();
            let mut rr = Lcg(seed ^ 0x9e3779b9);
            let rhs: Vec<$t> = (0..n).map(|_| rr.unit() as $t).collect();

            let elim: Vec<usize> = (p..2 * p).collect();
            let ssym = schur_symbolic(m.symbolic(), &elim).unwrap();
            let mut s = ssym.alloc_s::<$t>();
            let mut ctx = SchurContext::new();
            let mut rk = vec![0 as $t; s.symbolic().nrows()];
            schur_reduce(&ssym, &m, &rhs, &mut ctx, &mut s, &mut rk).unwrap();

            let nk = s.symbolic().nrows();
            let sd = s.to_dense();
            let mut sflat = vec![0.0f64; nk * nk];
            for j in 0..nk {
                for i in 0..nk {
                    sflat[i + j * nk] = sd[(i, j)] as f64;
                }
            }
            let rkf: Vec<f64> = rk.iter().map(|&v| v as f64).collect();
            (sflat, rkf, nk)
        }
    };
}

reduce_at!(reduce_f64, f64);
reduce_at!(reduce_f32, f32);

#[test]
fn f32_reduction_tracks_f64() {
    // The regime where accumulation order matters: pose self-blocks 1e4,
    // couplings ~10, each pose accumulating many small landmark contributions
    // on its diagonal.
    let (p, w) = (40, 8);
    let (big, med, coup) = (1e4, 1e3, 10.0);
    let (mut worst_s, mut worst_r) = (0.0f64, 0.0f64);
    for seed in [1u64, 7, 42, 1234, 99999] {
        let (part, cells) = structure(p, w);
        let (s64, r64, nk) = reduce_f64(&part, &cells, p, big, med, coup, seed);
        let (s32, r32, _) = reduce_f32(&part, &cells, p, big, med, coup, seed);
        let (mut ns, mut ds) = (0.0f64, 0.0f64);
        for j in 0..nk {
            for i in 0..=j {
                ns += (s32[i + j * nk] - s64[i + j * nk]).powi(2);
                ds += s64[i + j * nk].powi(2);
            }
        }
        let (mut nr, mut dr) = (0.0f64, 0.0f64);
        for i in 0..nk {
            nr += (r32[i] - r64[i]).powi(2);
            dr += r64[i].powi(2);
        }
        let rel_s = (ns / ds).sqrt();
        let rel_r = (nr / dr).sqrt();
        eprintln!("seed {seed:>6}: ||dS||/||S|| = {rel_s:.3e}   ||db||/||b|| = {rel_r:.3e}");
        worst_s = worst_s.max(rel_s);
        worst_r = worst_r.max(rel_r);
    }
    eprintln!("worst: dS {worst_s:.3e}  db {worst_r:.3e}");
    // The pot-then-diagonal accumulation (zero S, sum the small coupling terms,
    // fold Hkk in once) holds this near 0.3 f32-ULP. Seeding S with Hkk and
    // rounding every contribution against it lands at ~1.1 ULP, which this trips.
    assert!(worst_s < 7e-8, "S drift {worst_s:.3e} too large");
    assert!(worst_r < 5e-8, "rhs drift {worst_r:.3e} too large");
}
