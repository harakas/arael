//! The pieces of a device assembly that every model shares: the kernels
//! a constraint kernel reads from (the entities' parameter values and
//! rotation caches) and the `#[cube]` functions it writes with (a
//! residual row's rank-one update of an entity's tile and of a cross
//! tile of the Hessian, its gradient terms, and the cost). The constraint
//! kernels themselves are generated per model by arael's macros, which
//! print the constraint body and call these.
//!
//! H and the gradient are f64 whatever the model's scalar `F` (the
//! reduction reads H in f64), and every write is an atomic add into
//! buffers zeroed first. The tile layout is the CPU scatter's: an
//! entity's tile holds its upper block triangle, entry (i, j) of the
//! entity's live parameters at `base + (g_j - start) * stride + (g_i -
//! start)`, `g` the global parameter indices and `start` the first live
//! one; a cross tile holds the lower-numbered entity's span on its rows.

// Cube code, as kernels.rs says.
#![allow(clippy::unnecessary_cast, clippy::manual_div_ceil, clippy::assign_op_pattern, clippy::too_many_arguments)]

use cubecl::prelude::*;

/// A fixed parameter's global index.
pub const FIXED: u32 = u32::MAX;

/// An entity's parameter values from `x` (f64), in the model's scalar `F`,
/// one unit per value: `ev[i] = x[idx[i]]`, or `fixed[i]` for a fixed
/// parameter.
#[cube(launch_unchecked)]
pub fn k_gather_params<F: Float>(x: &[f64], idx: &[u32], fixed: &[F], ev: &mut [F], n: u32) {
    let i = ABSOLUTE_POS;
    if i < n as usize {
        let g = idx[i];
        if g == FIXED {
            ev[i] = fixed[i];
        } else {
            ev[i] = F::cast_from(x[g as usize]);
        }
    }
}

/// One Euler-angle rotation per unit: from the entity's reference rotation
/// `refr` (row-major 3x3, f64) and its step `delta` (three values of `ev`
/// at `at[e]`, a fixed rotation's being zero), the composed rotation
/// `R_ref R(delta)` in f64 (`rot_wide`) and in `F` (`rot`), and its three
/// derivatives `R_ref dR(delta)/d delta_k` in `F` (`drot`, k-major), as
/// `EulerAngleParam::__precompute` composes them (roll, pitch, yaw; R =
/// rot(z) rot(y) rot(x)).
#[cube(launch_unchecked)]
pub fn k_euler_rotations<F: Float>(refr: &[f64], ev: &[F], at: &[u32], rot_wide: &mut [f64], rot: &mut [F], drot: &mut [F], n: u32) {
    let e = ABSOLUTE_POS;
    if e < n as usize {
        let a = at[e] as usize;
        let dx = f64::cast_from(ev[a]);
        let dy = f64::cast_from(ev[a + 1usize]);
        let dz = f64::cast_from(ev[a + 2usize]);
        let sx = f64::sin(dx);
        let cx = f64::cos(dx);
        let sy = f64::sin(dy);
        let cy = f64::cos(dy);
        let sz = f64::sin(dz);
        let cz = f64::cos(dz);
        // R(delta), row-major
        let mut d = Array::<f64>::new(9usize);
        d[0] = cy * cz;
        d[1] = sx * sy * cz - cx * sz;
        d[2] = cx * sy * cz + sx * sz;
        d[3] = cy * sz;
        d[4] = cx * cz + sx * sy * sz;
        d[5] = cx * sy * sz - sx * cz;
        d[6] = 0.0f64 - sy;
        d[7] = cy * sx;
        d[8] = cx * cy;
        // dR/d roll, pitch, yaw
        let mut dd = Array::<f64>::new(27usize);
        dd[0] = 0.0f64;
        dd[1] = sx * sz + cz * cx * sy;
        dd[2] = cx * sz - sx * cz * sy;
        dd[3] = 0.0f64;
        dd[4] = cx * sy * sz - sx * cz;
        dd[5] = 0.0f64 - sx * sy * sz - cz * cx;
        dd[6] = 0.0f64;
        dd[7] = cy * cx;
        dd[8] = 0.0f64 - sx * cy;
        dd[9] = 0.0f64 - sy * cz;
        dd[10] = cz * sx * cy;
        dd[11] = cx * cz * cy;
        dd[12] = 0.0f64 - sy * sz;
        dd[13] = sx * cy * sz;
        dd[14] = cx * cy * sz;
        dd[15] = 0.0f64 - cy;
        dd[16] = 0.0f64 - sy * sx;
        dd[17] = 0.0f64 - cx * sy;
        dd[18] = 0.0f64 - cy * sz;
        dd[19] = 0.0f64 - cx * cz - sz * sx * sy;
        dd[20] = sx * cz - cx * sz * sy;
        dd[21] = cy * cz;
        dd[22] = sx * sy * cz - cx * sz;
        dd[23] = cx * sy * cz + sz * sx;
        dd[24] = 0.0f64;
        dd[25] = 0.0f64;
        dd[26] = 0.0f64;
        let r0 = e * 9usize;
        #[unroll]
        for i in 0..3usize {
            #[unroll]
            for j in 0..3usize {
                let mut v = 0.0f64;
                #[unroll]
                for k in 0..3usize {
                    v += refr[r0 + i * 3usize + k] * d[k * 3usize + j];
                }
                rot_wide[r0 + i * 3usize + j] = v;
                rot[r0 + i * 3usize + j] = F::cast_from(v);
            }
        }
        let q0 = e * 27usize;
        #[unroll]
        for m in 0..3usize {
            #[unroll]
            for i in 0..3usize {
                #[unroll]
                for j in 0..3usize {
                    let mut v = 0.0f64;
                    #[unroll]
                    for k in 0..3usize {
                        v += refr[r0 + i * 3usize + k] * dd[m * 9usize + k * 3usize + j];
                    }
                    drot[q0 + m * 9usize + i * 3usize + j] = F::cast_from(v);
                }
            }
        }
    }
}

/// The reference rotations moved by their steps, one unit per entity:
/// `refr = refr R(delta)`, the step read from `ev` at `at[e]`, as
/// `EulerAngleParam::advance` does after an accepted step.
#[cube(launch_unchecked)]
pub fn k_euler_advance<F: Float>(refr: &mut [f64], ev: &[F], at: &[u32], n: u32) {
    let e = ABSOLUTE_POS;
    if e < n as usize {
        let a = at[e] as usize;
        let dx = f64::cast_from(ev[a]);
        let dy = f64::cast_from(ev[a + 1usize]);
        let dz = f64::cast_from(ev[a + 2usize]);
        let sx = f64::sin(dx);
        let cx = f64::cos(dx);
        let sy = f64::sin(dy);
        let cy = f64::cos(dy);
        let sz = f64::sin(dz);
        let cz = f64::cos(dz);
        let mut d = Array::<f64>::new(9usize);
        d[0] = cy * cz;
        d[1] = sx * sy * cz - cx * sz;
        d[2] = cx * sy * cz + sx * sz;
        d[3] = cy * sz;
        d[4] = cx * cz + sx * sy * sz;
        d[5] = cx * sy * sz - sx * cz;
        d[6] = 0.0f64 - sy;
        d[7] = cy * sx;
        d[8] = cx * cy;
        let r0 = e * 9usize;
        let mut m = Array::<f64>::new(9usize);
        #[unroll]
        for i in 0..3usize {
            #[unroll]
            for j in 0..3usize {
                let mut v = 0.0f64;
                #[unroll]
                for k in 0..3usize {
                    v += refr[r0 + i * 3usize + k] * d[k * 3usize + j];
                }
                m[i * 3usize + j] = v;
            }
        }
        #[unroll]
        for i in 0..9usize {
            refr[r0 + i] = m[i];
        }
    }
}

/// H's diagonal gathered: `out[i] = h[pos[i]]`, one unit per scalar.
#[cube(launch_unchecked)]
pub fn k_gather_diag(h: &[f64], pos: &[u32], out: &mut [f64], n: u32) {
    let i = ABSOLUTE_POS;
    if i < n as usize {
        out[i] = h[pos[i] as usize];
    }
}

/// One residual row's update of an entity's tile and of the gradient:
/// `H += s dr dr^T` over the upper triangle of the entity's live
/// parameters and `g[g_i] += s r dr_i`, `s` being 2 (2w under a loss).
/// `idx[at..at + n]` are the entity's global parameter indices; its tile
/// is at `base` with columns `stride` apart, from parameter `start`.
#[cube]
pub fn self_rank1<F: Float>(
    h: &mut [Atomic<f64>],
    g: &mut [Atomic<f64>],
    idx: &[u32],
    at: usize,
    base: usize,
    stride: usize,
    start: u32,
    s: f64,
    r: F,
    dr: &Array<F>,
    #[comptime] n: usize,
) {
    #[unroll]
    for i in 0..n {
        let gi = idx[at + i];
        if gi != FIXED {
            let di = f64::cast_from(dr[i]);
            g[gi as usize].fetch_add(s * f64::cast_from(r) * di);
            let ri = (gi - start) as usize;
            #[unroll]
            for j in i..n {
                let gj = idx[at + j];
                if gj != FIXED {
                    let cj = (gj - start) as usize;
                    h[base + cj * stride + ri].fetch_add(s * di * f64::cast_from(dr[j]));
                }
            }
        }
    }
}

/// One residual row's update of a cross tile: `H += s dra drb^T` between
/// the live parameters of entities a (`idx_a[at_a..]`, first live
/// `start_a`) and b. The tile holds the lower-numbered entity on its rows:
/// `flip` 0 when that is a, 1 when it is b.
#[cube]
pub fn cross_rank1<F: Float>(
    h: &mut [Atomic<f64>],
    idx_a: &[u32],
    at_a: usize,
    start_a: u32,
    idx_b: &[u32],
    at_b: usize,
    start_b: u32,
    base: usize,
    stride: usize,
    flip: u32,
    s: f64,
    dra: &Array<F>,
    drb: &Array<F>,
    #[comptime] na: usize,
    #[comptime] nb: usize,
) {
    #[unroll]
    for i in 0..na {
        let gi = idx_a[at_a + i];
        if gi != FIXED {
            let di = s * f64::cast_from(dra[i]);
            let ra = (gi - start_a) as usize;
            #[unroll]
            for j in 0..nb {
                let gj = idx_b[at_b + j];
                if gj != FIXED {
                    let cb = (gj - start_b) as usize;
                    let v = di * f64::cast_from(drb[j]);
                    if flip == 0u32 {
                        h[base + cb * stride + ra].fetch_add(v);
                    } else {
                        h[base + ra * stride + cb].fetch_add(v);
                    }
                }
            }
        }
    }
}

/// The cube's sum of its units' `v` added to `cost[0]`: a shared-memory
/// tree over a cube of 256 units, then one atomic add per cube.
#[cube]
pub fn cost_add(cost: &mut [Atomic<f64>], v: f64) {
    let t = UNIT_POS as usize;
    let mut sh = Shared::<[f64]>::new_slice(256usize);
    sh[t] = v;
    sync_cube();
    let mut half = 128usize;
    while half > 0usize {
        if t < half {
            sh[t] += sh[t + half];
        }
        sync_cube();
        half = half / 2usize;
    }
    if t == 0usize {
        cost[0].fetch_add(sh[0]);
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::{GpuBackend, GpuContext};

    /// A test constraint between an entity of 3 parameters (a) and one of 2
    /// (b), with one datum c: r0 = a0 a1 - c b0, r1 = a2 + b0 b1, written as
    /// a generated kernel would: one unit per instance.
    #[cube(launch_unchecked)]
    fn k_test_constraint(
        ev_a: &[f64],
        ev_b: &[f64],
        idx_a: &[u32],
        idx_b: &[u32],
        tile_a: &[u32],
        tile_b: &[u32],
        inst: &[u32],
        data: &[f64],
        h: &mut [Atomic<f64>],
        g: &mut [Atomic<f64>],
        cost: &mut [Atomic<f64>],
        n: u32,
    ) {
        let o = ABSOLUTE_POS;
        let mut c = 0.0f64;
        if o < n as usize {
            // instance: a slot, b slot, cross base, cross stride, flip
            let sa = inst[o * 5] as usize;
            let sb = inst[o * 5 + 1] as usize;
            let a0 = ev_a[sa * 3];
            let a1 = ev_a[sa * 3 + 1];
            let a2 = ev_a[sa * 3 + 2];
            let b0 = ev_b[sb * 2];
            let b1 = ev_b[sb * 2 + 1];
            let cc = data[o];
            let r0 = a0 * a1 - cc * b0;
            let r1 = a2 + b0 * b1;
            c = r0 * r0 + r1 * r1;
            let mut dra = Array::<f64>::new(3usize);
            let mut drb = Array::<f64>::new(2usize);
            // row 0
            dra[0] = a1;
            dra[1] = a0;
            dra[2] = 0.0f64;
            drb[0] = 0.0f64 - cc;
            drb[1] = 0.0f64;
            self_rank1::<f64>(h, g, idx_a, sa * 3, tile_a[sa * 3] as usize, tile_a[sa * 3 + 1] as usize, tile_a[sa * 3 + 2], 2.0f64, r0, &dra, 3usize);
            self_rank1::<f64>(h, g, idx_b, sb * 2, tile_b[sb * 3] as usize, tile_b[sb * 3 + 1] as usize, tile_b[sb * 3 + 2], 2.0f64, r0, &drb, 2usize);
            cross_rank1::<f64>(h, idx_a, sa * 3, tile_a[sa * 3 + 2], idx_b, sb * 2, tile_b[sb * 3 + 2], inst[o * 5 + 2] as usize, inst[o * 5 + 3] as usize, inst[o * 5 + 4], 2.0f64, &dra, &drb, 3usize, 2usize);
            // row 1
            dra[0] = 0.0f64;
            dra[1] = 0.0f64;
            dra[2] = 1.0f64;
            drb[0] = b1;
            drb[1] = b0;
            self_rank1::<f64>(h, g, idx_a, sa * 3, tile_a[sa * 3] as usize, tile_a[sa * 3 + 1] as usize, tile_a[sa * 3 + 2], 2.0f64, r1, &dra, 3usize);
            self_rank1::<f64>(h, g, idx_b, sb * 2, tile_b[sb * 3] as usize, tile_b[sb * 3 + 1] as usize, tile_b[sb * 3 + 2], 2.0f64, r1, &drb, 2usize);
            cross_rank1::<f64>(h, idx_a, sa * 3, tile_a[sa * 3 + 2], idx_b, sb * 2, tile_b[sb * 3 + 2], inst[o * 5 + 2] as usize, inst[o * 5 + 3] as usize, inst[o * 5 + 4], 2.0f64, &dra, &drb, 3usize, 2usize);
        }
        cost_add(cost, c);
    }

    /// The composed rotation is `R_ref` at a zero step and orthonormal at
    /// any, its derivatives are the finite differences of it, and the `F`
    /// copy is the f64 one narrowed.
    #[test]
    fn euler_rotations_and_their_derivatives() {
        let ctx = GpuContext::new(GpuBackend::Cuda, 0).unwrap();
        let c = ctx.client();
        let arr = |hd: &cubecl::server::Handle, len: usize| unsafe { BufferArg::from_raw_parts(hd.clone(), len.max(1)) };
        // a reference rotation: about z by 0.7, then about x by -0.3
        let (s1, c1, s2, c2) = (0.7f64.sin(), 0.7f64.cos(), (-0.3f64).sin(), (-0.3f64).cos());
        let rz = [c1, -s1, 0.0, s1, c1, 0.0, 0.0, 0.0, 1.0];
        let rx = [1.0, 0.0, 0.0, 0.0, c2, -s2, 0.0, s2, c2];
        let mut refr = [0.0f64; 9];
        for i in 0..3 {
            for j in 0..3 {
                refr[i * 3 + j] = (0..3).map(|k| rz[i * 3 + k] * rx[k * 3 + j]).sum();
            }
        }
        let eps = 1e-6;
        let base = [0.11f64, -0.23, 0.31];
        // entity 0 at the zero step, 1 at `base`, then 2 + 2k +- eps along k
        let mut deltas = vec![[0.0f64; 3], base];
        for k in 0..3 {
            for sgn in [1.0, -1.0] {
                let mut d = base;
                d[k] += sgn * eps;
                deltas.push(d);
            }
        }
        let n = deltas.len();
        let ev: Vec<f32> = deltas.iter().flatten().map(|&v| v as f32).collect();
        let ev64: Vec<f64> = ev.iter().map(|&v| v as f64).collect();
        let at: Vec<u32> = (0..n as u32).map(|e| e * 3).collect();
        let refs: Vec<f64> = (0..n).flat_map(|_| refr).collect();
        let h_ref = c.create_from_slice(f64::as_bytes(&refs));
        let h_ev = c.create_from_slice(f32::as_bytes(&ev));
        let h_at = c.create_from_slice(u32::as_bytes(&at));
        let (h_rw, h_r, h_dr) = (c.empty(n * 9 * 8), c.empty(n * 9 * 4), c.empty(n * 27 * 4));
        unsafe {
            k_euler_rotations::launch_unchecked::<f32>(c, CubeCount::Static(1, 1, 1), CubeDim::new_1d(32), arr(&h_ref, n * 9), arr(&h_ev, n * 3), arr(&h_at, n), arr(&h_rw, n * 9), arr(&h_r, n * 9), arr(&h_dr, n * 27), n as u32)
        };
        let rw = f64::from_bytes(&crate::read_one(c, h_rw).unwrap()).to_vec();
        let r32 = f32::from_bytes(&crate::read_one(c, h_r).unwrap()).to_vec();
        let dr = f32::from_bytes(&crate::read_one(c, h_dr).unwrap()).to_vec();
        for i in 0..9 {
            assert!((rw[i] - refr[i]).abs() < 1e-15, "zero step: entry {i}");
        }
        for e in 0..n {
            let m = &rw[e * 9..e * 9 + 9];
            for i in 0..3 {
                for j in 0..3 {
                    let dot: f64 = (0..3).map(|k| m[i * 3 + k] * m[j * 3 + k]).sum();
                    assert!((dot - if i == j { 1.0 } else { 0.0 }).abs() < 1e-14, "entity {e} not orthonormal");
                }
            }
            for i in 0..9 {
                assert_eq!(r32[e * 9 + i], m[i] as f32);
            }
        }
        for k in 0..3 {
            let (p, q) = (2 + 2 * k, 3 + 2 * k);
            // the steps actually applied are the f32 ones
            let h = ev64[p * 3 + k] - ev64[q * 3 + k];
            for i in 0..9 {
                let fd = (rw[p * 9 + i] - rw[q * 9 + i]) / h;
                // entity 1 sits at `base`; its derivatives are stored in f32
                let an = dr[27 + k * 9 + i] as f64;
                assert!((fd - an).abs() < 1e-6, "d/d delta_{k} entry {i}: {fd} against {an}");
            }
        }
    }

    /// The device writes against a dense reference: entities of type a and
    /// b with interleaved global indices (so cross tiles come in both
    /// orientations), one parameter of an a fixed, every instance's rows
    /// added into its tiles; H read back through the same tile layout.
    #[test]
    fn rank_one_writes_land_in_the_tiles_of_the_cpu_layout() {
        let (na, nb) = (4usize, 5usize);
        // global indices: a0 b0 a1 b1 ... interleaved by entity, a's second
        // parameter of entity 2 fixed
        let mut idx_a = vec![0u32; na * 3];
        let mut idx_b = vec![0u32; nb * 2];
        let mut next = 0u32;
        for e in 0..nb {
            if e < na {
                for k in 0..3usize {
                    idx_a[e * 3 + k] = if e == 2 && k == 1 { FIXED } else { let v = next; next += 1; v };
                }
            }
            for k in 0..2 {
                idx_b[e * 2 + k] = next;
                next += 1;
            }
        }
        let n = next as usize;
        // a dense column-major n x n "tile store": one tile per entity and
        // per instance's cross pair, each its own region of the buffer
        let live = |ix: &[u32]| -> (u32, usize) {
            let l: Vec<u32> = ix.iter().copied().filter(|&g| g != FIXED).collect();
            (l[0], l.len())
        };
        let mut hlen = 0usize;
        let mut tile_a = vec![0u32; na * 3];
        let mut tile_b = vec![0u32; nb * 3];
        for e in 0..na {
            let (st, w) = live(&idx_a[e * 3..e * 3 + 3]);
            tile_a[e * 3..e * 3 + 3].copy_from_slice(&[hlen as u32, w as u32, st]);
            hlen += w * w;
        }
        for e in 0..nb {
            let (st, w) = live(&idx_b[e * 2..e * 2 + 2]);
            tile_b[e * 3..e * 3 + 3].copy_from_slice(&[hlen as u32, w as u32, st]);
            hlen += w * w;
        }
        let pairs: Vec<(usize, usize, f64)> = (0..17).map(|o| (o % na, (o * 3) % nb, 0.3 + 0.1 * o as f64)).collect();
        let mut inst = Vec::new();
        let mut cross_tiles = Vec::new();
        for &(sa, sb, _) in &pairs {
            let (st_a, wa) = live(&idx_a[sa * 3..sa * 3 + 3]);
            let (st_b, wb) = live(&idx_b[sb * 2..sb * 2 + 2]);
            let flip = (st_b < st_a) as u32;
            let rows = if flip == 0 { wa } else { wb };
            let cols = if flip == 0 { wb } else { wa };
            inst.extend_from_slice(&[sa as u32, sb as u32, hlen as u32, rows as u32, flip]);
            cross_tiles.push((hlen, rows, cols, flip, sa, sb));
            hlen += rows * cols;
        }
        let ev_a: Vec<f64> = (0..na * 3).map(|i| 0.5 + 0.25 * i as f64).collect();
        let ev_b: Vec<f64> = (0..nb * 2).map(|i| -0.4 + 0.15 * i as f64).collect();
        let data: Vec<f64> = pairs.iter().map(|p| p.2).collect();
        // the dense reference: J^T J and J^T r over the global indices
        let mut hd = vec![0.0f64; n * n];
        let mut gd = vec![0.0f64; n];
        let mut cd = 0.0f64;
        for &(sa, sb, cc) in &pairs {
            let (a0, a1, a2) = (ev_a[sa * 3], ev_a[sa * 3 + 1], ev_a[sa * 3 + 2]);
            let (b0, b1) = (ev_b[sb * 2], ev_b[sb * 2 + 1]);
            let rows = [(a0 * a1 - cc * b0, [a1, a0, 0.0], [-cc, 0.0]), (a2 + b0 * b1, [0.0, 0.0, 1.0], [b1, b0])];
            for (r, da, db) in rows {
                cd += r * r;
                let mut jac = vec![(0u32, 0.0f64); 0];
                for k in 0..3usize {
                    if idx_a[sa * 3 + k] != FIXED {
                        jac.push((idx_a[sa * 3 + k], da[k]));
                    }
                }
                for k in 0..2 {
                    jac.push((idx_b[sb * 2 + k], db[k]));
                }
                for &(gi, di) in &jac {
                    gd[gi as usize] += 2.0 * r * di;
                    for &(gj, dj) in &jac {
                        hd[gi as usize + gj as usize * n] += 2.0 * di * dj;
                    }
                }
            }
        }
        let ctx = GpuContext::new(GpuBackend::Cuda, 0).unwrap();
        let c = ctx.client();
        let up = |v: &[u32]| c.create_from_slice(u32::as_bytes(v));
        let upf = |v: &[f64]| c.create_from_slice(f64::as_bytes(v));
        let (h, g, cost) = (upf(&vec![0.0; hlen]), upf(&vec![0.0; n]), upf(&[0.0]));
        let arr = |hd: &cubecl::server::Handle, len: usize| unsafe { BufferArg::from_raw_parts(hd.clone(), len.max(1)) };
        let (hea, heb, hia, hib, hta, htb, hin, hda) = (upf(&ev_a), upf(&ev_b), up(&idx_a), up(&idx_b), up(&tile_a), up(&tile_b), up(&inst), upf(&data));
        let ni = pairs.len() as u32;
        unsafe {
            k_test_constraint::launch_unchecked(
                c,
                CubeCount::Static(ni.div_ceil(256), 1, 1),
                CubeDim::new_1d(256),
                arr(&hea, ev_a.len()),
                arr(&heb, ev_b.len()),
                arr(&hia, idx_a.len()),
                arr(&hib, idx_b.len()),
                arr(&hta, tile_a.len()),
                arr(&htb, tile_b.len()),
                arr(&hin, inst.len()),
                arr(&hda, data.len()),
                arr(&h, hlen),
                arr(&g, n),
                arr(&cost, 1),
                ni,
            )
        };
        let hv = f64::from_bytes(&crate::read_one(c, h).unwrap()).to_vec();
        let gv = f64::from_bytes(&crate::read_one(c, g).unwrap()).to_vec();
        let cv = f64::from_bytes(&crate::read_one(c, cost).unwrap())[0];
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-12 * (1.0 + b.abs());
        assert!(close(cv, cd), "cost {cv} against {cd}");
        for i in 0..n {
            assert!(close(gv[i], gd[i]), "gradient {i}: {} against {}", gv[i], gd[i]);
        }
        // every entity tile's upper triangle, then every cross tile, summed
        // where tiles of one pair repeat
        for (e, (ix, tiles, w)) in [(&idx_a, &tile_a, 3usize), (&idx_b, &tile_b, 2usize)].into_iter().enumerate() {
            for s in 0..tiles.len() / 3 {
                let (base, stride, st) = (tiles[s * 3] as usize, tiles[s * 3 + 1] as usize, tiles[s * 3 + 2]);
                for i in 0..w {
                    for j in i..w {
                        let (gi, gj) = (ix[s * w + i], ix[s * w + j]);
                        if gi == FIXED || gj == FIXED {
                            continue;
                        }
                        let v = hv[base + (gj - st) as usize * stride + (gi - st) as usize];
                        assert!(close(v, hd[gi as usize + gj as usize * n]), "entity {e} slot {s} ({i}, {j}): {v}");
                    }
                }
            }
        }
        let mut seen = std::collections::HashMap::<(usize, usize), f64>::new();
        for &(base, rows, cols, flip, sa, sb) in &cross_tiles {
            for i in 0..3usize {
                for j in 0..2 {
                    let (gi, gj) = (idx_a[sa * 3 + i], idx_b[sb * 2 + j]);
                    if gi == FIXED {
                        continue;
                    }
                    let (st_a, st_b) = (live(&idx_a[sa * 3..sa * 3 + 3]).0, live(&idx_b[sb * 2..sb * 2 + 2]).0);
                    let (r, cc) = if flip == 0 { ((gi - st_a) as usize, (gj - st_b) as usize) } else { ((gj - st_b) as usize, (gi - st_a) as usize) };
                    assert!(r < rows && cc < cols);
                    *seen.entry((gi as usize, gj as usize)).or_insert(0.0) += hv[base + cc * rows + r];
                }
            }
        }
        for ((gi, gj), v) in seen {
            assert!(close(v, hd[gi + gj * n]), "cross ({gi}, {gj}): {v} against {}", hd[gi + gj * n]);
        }
    }
}
