//! The Levenberg-Marquardt loop's vector work on the device, for a solve
//! whose parameters, step and linearization stay there: what the loop
//! decides on comes back as a few scalars, read with the attempt's result.
//!
//! Per linearization, [`k_lin_pass`] gathers H's diagonal, writes the
//! damping scale, and reduces the gradient's largest entry, the Jacobi-
//! scaled one and the first faulty diagonal entry; per attempt,
//! [`k_attempt_pass`] forms the trial point `x - delta` and reduces the
//! step's and the trial point's squared norms and the predicted
//! reduction. Both leave partials per cube, which [`k_combine_lin`] and
//! [`k_combine_attempt`] fold into the scalars buffer, at the slots below.
//! [`LmVectors`] holds the buffers and launches the passes.

// Cube code, as kernels.rs says.
#![allow(clippy::unnecessary_cast, clippy::manual_div_ceil, clippy::assign_op_pattern, clippy::too_many_arguments, clippy::neg_cmp_op_on_partial_ord)]

use cubecl::bytes::Bytes;
use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::context::{GpuContext, Lease};

/// The scalars buffer's slots.
pub const S_COST_X: usize = 0;
pub const S_COST_TRY: usize = 1;
pub const S_GRAD_MAX: usize = 2;
pub const S_GRAD_SCALED: usize = 3;
/// The first faulty diagonal entry as `4 i + kind` (1 NaN, 2 negative, 3
/// zero), [`NO_FAULT`] when there is none.
pub const S_FAULT: usize = 4;
pub const S_STEP2: usize = 5;
pub const S_TRY2: usize = 6;
pub const S_PREDICTED: usize = 7;
/// The scalars a solve reads.
pub const N_SCALARS: usize = 8;
/// No faulty diagonal entry.
pub const NO_FAULT: f64 = 1e300;

/// Partials per cube of each pass.
pub const LIN_PARTIALS: usize = 3;
pub const ATTEMPT_PARTIALS: usize = 3;

/// How [`cube_reduce`] combines: a sum, a maximum, a minimum.
pub const OP_SUM: u32 = 0;
pub const OP_MAX: u32 = 1;
pub const OP_MIN: u32 = 2;

/// `v` combined over a cube of 256 units by `op`, the result in every unit.
#[cube]
fn cube_reduce(v: f64, #[comptime] op: u32) -> f64 {
    let t = UNIT_POS as usize;
    let mut sh = Shared::<[f64]>::new_slice(256usize);
    // a previous call's readers are done with the shared words first
    sync_cube();
    sh[t] = v;
    sync_cube();
    let mut half = 128usize;
    while half > 0usize {
        if t < half {
            let a = sh[t];
            let b = sh[t + half];
            if comptime!(op == OP_SUM) {
                sh[t] = a + b;
            } else if comptime!(op == OP_MAX) {
                sh[t] = select(b > a, b, a);
            } else {
                sh[t] = select(b < a, b, a);
            }
        }
        sync_cube();
        half = half / 2usize;
    }
    sh[0]
}

/// One unit per parameter: H's diagonal `diag[i] = h[pos[i]]`, the damping
/// scale (`max(diag, floor)` when `has_floor`, else the diagonal), and per
/// cube the gradient's largest magnitude, the largest `|g| / sqrt(d)` over
/// `d > 0`, and the first faulty entry (with a floor a NaN or negative
/// diagonal, without one also a zero).
#[cube(launch_unchecked)]
pub fn k_lin_pass(g: &[f64], h: &[f64], pos: &[u32], diag: &mut [f64], damp: &mut [f64], floor: f64, has_floor: u32, partials: &mut [f64], n: u32) {
    let i = ABSOLUTE_POS;
    let mut gm = 0.0f64;
    let mut gs = 0.0f64;
    let mut fault = 1e300f64;
    if i < n as usize {
        let d = h[pos[i] as usize];
        diag[i] = d;
        let gi = f64::abs(g[i]);
        gm = gi;
        if d > 0.0f64 {
            gs = gi / f64::sqrt(d);
        }
        let mut bad = !(d > 0.0f64);
        if has_floor != 0u32 {
            bad = !(d >= 0.0f64);
            damp[i] = select(d > floor, d, floor);
        } else {
            damp[i] = d;
        }
        if bad {
            let mut kind = 3.0f64;
            if d != d {
                kind = 1.0f64;
            } else if d < 0.0f64 {
                kind = 2.0f64;
            }
            fault = 4.0f64 * f64::cast_from(i as u32) + kind;
        }
    }
    let gm = cube_reduce(gm, OP_MAX);
    let gs = cube_reduce(gs, OP_MAX);
    let fault = cube_reduce(fault, OP_MIN);
    if UNIT_POS == 0u32 {
        let c = CUBE_POS as usize * LIN_PARTIALS;
        partials[c] = gm;
        partials[c + 1usize] = gs;
        partials[c + 2usize] = fault;
    }
}

/// The damped diagonal written into H's values: `h[pos[i]] = diag[i] +
/// lambda[0] damp[i]`, one unit per scalar.
#[cube(launch_unchecked)]
pub fn k_damp(h: &mut [f64], pos: &[u32], diag: &[f64], damp: &[f64], lambda: &[f64], n: u32) {
    let i = ABSOLUTE_POS;
    if i < n as usize {
        h[pos[i] as usize] = diag[i] + lambda[0] * damp[i];
    }
}

/// One unit per parameter: the trial point `x_try = x - delta`, and per
/// cube the sums of `delta^2`, `x_try^2` and `delta (g + lambda[0] d
/// delta)` (twice the predicted reduction).
#[cube(launch_unchecked)]
pub fn k_attempt_pass(x: &[f64], delta: &[f64], g: &[f64], diag: &[f64], lambda: &[f64], x_try: &mut [f64], partials: &mut [f64], n: u32) {
    let i = ABSOLUTE_POS;
    let mut s2 = 0.0f64;
    let mut t2 = 0.0f64;
    let mut pr = 0.0f64;
    if i < n as usize {
        let d = delta[i];
        let t = x[i] - d;
        x_try[i] = t;
        s2 = d * d;
        t2 = t * t;
        pr = d * (g[i] + lambda[0] * diag[i] * d);
    }
    let s2 = cube_reduce(s2, OP_SUM);
    let t2 = cube_reduce(t2, OP_SUM);
    let pr = cube_reduce(pr, OP_SUM);
    if UNIT_POS == 0u32 {
        let c = CUBE_POS as usize * ATTEMPT_PARTIALS;
        partials[c] = s2;
        partials[c + 1usize] = t2;
        partials[c + 2usize] = pr;
    }
}

/// One partial of `cubes` cubes (`per` a cube, entry `k`) folded by `op`,
/// over the units of one cube of 256: the result in every unit.
#[cube]
fn fold_partials(partials: &[f64], cubes: u32, #[comptime] per: usize, #[comptime] k: usize, #[comptime] op: u32) -> f64 {
    let mut acc = 0.0f64;
    if comptime!(op == OP_MIN) {
        acc = 1e300f64;
    }
    let mut c = UNIT_POS as usize;
    while c < cubes as usize {
        let v = partials[c * per + k];
        if comptime!(op == OP_SUM) {
            acc += v;
        } else if comptime!(op == OP_MAX) {
            acc = select(v > acc, v, acc);
        } else {
            acc = select(v < acc, v, acc);
        }
        c += 256usize;
    }
    cube_reduce(acc, op)
}

/// [`k_lin_pass`]'s partials folded into the scalars, one cube of 256.
#[cube(launch_unchecked)]
pub fn k_combine_lin(partials: &[f64], cubes: u32, out: &mut [f64]) {
    let gm = fold_partials(partials, cubes, LIN_PARTIALS, 0usize, OP_MAX);
    let gs = fold_partials(partials, cubes, LIN_PARTIALS, 1usize, OP_MAX);
    let fault = fold_partials(partials, cubes, LIN_PARTIALS, 2usize, OP_MIN);
    if UNIT_POS == 0u32 {
        out[S_GRAD_MAX] = gm;
        out[S_GRAD_SCALED] = gs;
        out[S_FAULT] = fault;
    }
}

/// [`k_attempt_pass`]'s partials folded into the scalars, one cube of 256;
/// the predicted reduction is half its sum.
#[cube(launch_unchecked)]
pub fn k_combine_attempt(partials: &[f64], cubes: u32, out: &mut [f64]) {
    let s2 = fold_partials(partials, cubes, ATTEMPT_PARTIALS, 0usize, OP_SUM);
    let t2 = fold_partials(partials, cubes, ATTEMPT_PARTIALS, 1usize, OP_SUM);
    let pr = fold_partials(partials, cubes, ATTEMPT_PARTIALS, 2usize, OP_SUM);
    if UNIT_POS == 0u32 {
        out[S_STEP2] = s2;
        out[S_TRY2] = t2;
        out[S_PREDICTED] = 0.5f64 * pr;
    }
}

/// `dst = src`, `n` values.
#[cube(launch_unchecked)]
pub fn k_copy(dst: &mut [f64], src: &[f64], n: u32) {
    let i = ABSOLUTE_POS;
    if i < n as usize {
        dst[i] = src[i];
    }
}

/// The rotation steps of `x` zeroed, one unit per entity: the three global
/// indices at `idx[at[e]..]`, a fixed one left alone; what re-centring
/// does to the parameters after an accepted step.
#[cube(launch_unchecked)]
pub fn k_zero_steps(x: &mut [f64], idx: &[u32], at: &[u32], n: u32) {
    let e = ABSOLUTE_POS;
    if e < n as usize {
        let a = at[e] as usize;
        #[unroll]
        for k in 0..3usize {
            let g = idx[a + k];
            if g != crate::assembly::FIXED {
                x[g as usize] = 0.0f64;
            }
        }
    }
}

/// A faulty diagonal entry of H, as [`k_lin_pass`] finds it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiagFault {
    Nan,
    Negative,
    Zero,
}

/// The scalars buffer decoded: what the loop decides on.
#[derive(Clone, Copy, Debug)]
pub struct Scalars {
    /// The cost at the linearization point and at the trial point.
    pub cost_x: f64,
    pub cost_try: f64,
    /// The gradient's largest magnitude and the largest `|g_i| / sqrt(d_i)`
    /// over `d_i > 0`.
    pub grad_max: f64,
    pub grad_scaled: f64,
    /// The first faulty diagonal entry: its parameter and the fault.
    pub fault: Option<(usize, DiagFault)>,
    /// The squared norms of the step and of the trial point.
    pub step2: f64,
    pub try2: f64,
    /// The cost reduction the damped model predicts for the step,
    /// `(delta . g + lambda delta . D delta) / 2` with D the diagonal.
    pub predicted: f64,
}

impl Scalars {
    /// Decode the [`N_SCALARS`] f64 at the start of `bytes`.
    pub fn decode(bytes: &[u8]) -> Scalars {
        let s = |i: usize| bytemuck::pod_read_unaligned::<f64>(&bytes[8 * i..8 * i + 8]);
        let code = s(S_FAULT);
        let fault = (code < NO_FAULT).then(|| {
            let code = code as u64;
            let kind = match code % 4 {
                1 => DiagFault::Nan,
                2 => DiagFault::Negative,
                _ => DiagFault::Zero,
            };
            ((code / 4) as usize, kind)
        });
        Scalars {
            cost_x: s(S_COST_X),
            cost_try: s(S_COST_TRY),
            grad_max: s(S_GRAD_MAX),
            grad_scaled: s(S_GRAD_SCALED),
            fault,
            step2: s(S_STEP2),
            try2: s(S_TRY2),
            predicted: s(S_PREDICTED),
        }
    }
}

fn cubes(n: usize) -> u32 {
    (n as u32).div_ceil(256).max(1)
}

fn arr(h: &Handle, len: usize) -> BufferArg {
    unsafe { BufferArg::from_raw_parts(h.clone(), len.max(1)) }
}

/// The Levenberg-Marquardt loop's vectors on a device, all f64: the
/// parameters `x` and the trial point, H's undamped diagonal and the
/// damping scale, and the [`Scalars`] the loop decides on. H and the
/// gradient are the caller's (a [`crate::GpuSchur`]'s targets).
pub struct LmVectors {
    ctx: GpuContext,
    n: usize,
    /// H's scalar diagonal: its positions in H's values (u32).
    pos: Lease,
    x: Lease,
    x_try: Lease,
    diag: Lease,
    damp: Lease,
    /// The damping of the next attempt, one f64.
    lambda: Lease,
    lin: Lease,
    att: Lease,
    scalars: Lease,
}

impl LmVectors {
    /// The vectors of `pos.len()` parameters, H's diagonal at `pos` in its
    /// values.
    pub fn new(ctx: &GpuContext, pos: &[u32]) -> Self {
        let n = pos.len();
        let vec = || ctx.lease(n.max(1) * 8);
        let pos_lease = ctx.lease(n.max(1) * 4);
        if n > 0 {
            ctx.client().write(pos_lease.handle(), Bytes::from_elems(pos.to_vec()));
        }
        let parts = cubes(n) as usize;
        LmVectors {
            ctx: ctx.clone(),
            n,
            pos: pos_lease,
            x: vec(),
            x_try: vec(),
            diag: vec(),
            damp: vec(),
            lambda: ctx.lease(8),
            lin: ctx.lease(parts * LIN_PARTIALS * 8),
            att: ctx.lease(parts * ATTEMPT_PARTIALS * 8),
            scalars: ctx.lease(N_SCALARS * 8),
        }
    }

    /// The parameter count.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether there are no parameters.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Write the parameters (no wait).
    pub fn write_x(&self, x: Vec<f64>) {
        assert_eq!(x.len(), self.n);
        if self.n > 0 {
            self.ctx.client().write(self.x.handle(), Bytes::from_elems(x));
        }
    }

    /// The parameters.
    pub fn x(&self) -> Handle {
        self.x.view(self.n.max(1) * 8)
    }

    /// The trial point of the last [`step`](Self::step).
    pub fn x_try(&self) -> Handle {
        self.x_try.view(self.n.max(1) * 8)
    }

    /// H's undamped diagonal, as the last [`linearized`](Self::linearized)
    /// or [`gather_diag`](Self::gather_diag) left it.
    pub fn diag(&self) -> Handle {
        self.diag.view(self.n.max(1) * 8)
    }

    /// The scalars buffer, [`N_SCALARS`] f64, to read and
    /// [`decode`](Scalars::decode).
    pub fn scalars(&self) -> Handle {
        self.scalars.view(N_SCALARS * 8)
    }

    /// The scalar at `slot` (an `S_*` constant), a target for a cost.
    pub fn slot(&self, slot: usize) -> Handle {
        assert!(slot < N_SCALARS);
        let end = self.scalars.capacity() - 8 * (slot + 1);
        self.scalars.handle().clone().offset_start(8 * slot as u64).offset_end(end as u64)
    }

    /// The diagonal of H (`nh` values) gathered.
    pub fn gather_diag(&self, h: &Handle, nh: usize) {
        let n = self.n;
        unsafe {
            crate::assembly::k_gather_diag::launch_unchecked(self.ctx.client(), CubeCount::Static(cubes(n), 1, 1), CubeDim::new_1d(256), arr(h, nh), arr(self.pos.handle(), n), arr(self.diag.handle(), n), n as u32)
        };
    }

    /// After an assembly into `g` and `h` (`nh` values): the diagonal, the
    /// damping scale (`max(diag, floor)` with a floor) and the
    /// linearization's scalars.
    pub fn linearized(&self, g: &Handle, h: &Handle, nh: usize, floor: Option<f64>) {
        let (n, c) = (self.n, self.ctx.client());
        let parts = cubes(n);
        unsafe {
            k_lin_pass::launch_unchecked(
                c,
                CubeCount::Static(parts, 1, 1),
                CubeDim::new_1d(256),
                arr(g, n),
                arr(h, nh),
                arr(self.pos.handle(), n),
                arr(self.diag.handle(), n),
                arr(self.damp.handle(), n),
                floor.unwrap_or(0.0),
                floor.is_some() as u32,
                arr(self.lin.handle(), parts as usize * LIN_PARTIALS),
                n as u32,
            );
            k_combine_lin::launch_unchecked(c, CubeCount::Static(1, 1, 1), CubeDim::new_1d(256), arr(self.lin.handle(), parts as usize * LIN_PARTIALS), parts, arr(self.scalars.handle(), N_SCALARS));
        }
    }

    /// The damping of the next [`damp`](Self::damp) and
    /// [`step`](Self::step), written to the device (no wait).
    pub fn set_lambda(&self, lambda: f64) {
        self.ctx.client().write(self.lambda.handle(), Bytes::from_elems(vec![lambda]));
    }

    /// H's diagonal damped, `diag + lambda damp`, in its `nh` values.
    pub fn damp(&self, h: &Handle, nh: usize) {
        let n = self.n;
        unsafe {
            k_damp::launch_unchecked(
                self.ctx.client(),
                CubeCount::Static(cubes(n), 1, 1),
                CubeDim::new_1d(256),
                arr(h, nh),
                arr(self.pos.handle(), n),
                arr(self.diag.handle(), n),
                arr(self.damp.handle(), n),
                arr(self.lambda.handle(), 1),
                n as u32,
            )
        };
    }

    /// The trial point `x - step` and the attempt's scalars, for the step
    /// solved at the set lambda against the gradient `g`.
    pub fn step(&self, step: &Handle, g: &Handle) {
        let (n, c) = (self.n, self.ctx.client());
        let parts = cubes(n);
        unsafe {
            k_attempt_pass::launch_unchecked(
                c,
                CubeCount::Static(parts, 1, 1),
                CubeDim::new_1d(256),
                arr(self.x.handle(), n),
                arr(step, n),
                arr(g, n),
                arr(self.diag.handle(), n),
                arr(self.lambda.handle(), 1),
                arr(self.x_try.handle(), n),
                arr(self.att.handle(), parts as usize * ATTEMPT_PARTIALS),
                n as u32,
            );
            k_combine_attempt::launch_unchecked(c, CubeCount::Static(1, 1, 1), CubeDim::new_1d(256), arr(self.att.handle(), parts as usize * ATTEMPT_PARTIALS), parts, arr(self.scalars.handle(), N_SCALARS));
        }
    }

    /// The trial point becomes the parameters: copied, so the buffers keep
    /// their roles (and a captured graph its addresses).
    pub fn accept(&self) {
        let n = self.n;
        unsafe { k_copy::launch_unchecked(self.ctx.client(), CubeCount::Static(cubes(n), 1, 1), CubeDim::new_1d(256), arr(self.x.handle(), n), arr(self.x_try.handle(), n), n as u32) };
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::{GpuBackend, GpuContext};

    fn read(ctx: &GpuContext, h: Handle) -> Vec<f64> {
        f64::from_bytes(&crate::read_one(ctx.client(), h).unwrap()).to_vec()
    }

    #[test]
    fn the_passes_match_the_host() {
        let ctx = GpuContext::new(GpuBackend::Cuda, 0).unwrap();
        // past one cube, so the partials fold
        let n = 1000usize;
        let nh = 3 * n;
        let pos: Vec<u32> = (0..n as u32).map(|i| 3 * i + 1).collect();
        let mut h = vec![0.0f64; nh];
        for i in 0..n {
            h[3 * i + 1] = 0.5 + (i % 7) as f64;
        }
        let g: Vec<f64> = (0..n).map(|i| ((i * 37) % 11) as f64 - 5.0).collect();
        let x: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let step: Vec<f64> = (0..n).map(|i| 1e-2 * (i as f64).cos()).collect();
        let lambda = 0.3;
        let up = |v: &[f64]| {
            let l = ctx.lease(v.len() * 8);
            ctx.client().write(l.handle(), Bytes::from_elems(v.to_vec()));
            l
        };
        let (hd, gd, sd) = (up(&h), up(&g), up(&step));
        let v = LmVectors::new(&ctx, &pos);
        v.write_x(x.clone());
        v.linearized(gd.handle(), hd.handle(), nh, None);
        v.set_lambda(lambda);
        v.damp(hd.handle(), nh);
        v.step(&sd.view(n * 8), gd.handle());
        let s = Scalars::decode(&crate::read_one(ctx.client(), v.scalars()).unwrap());
        let diag: Vec<f64> = pos.iter().map(|&p| h[p as usize]).collect();
        let gmax = g.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let gs = g.iter().zip(&diag).fold(0.0f64, |m, (g, d)| m.max(g.abs() / d.sqrt()));
        let step2: f64 = step.iter().map(|d| d * d).sum();
        let try2: f64 = x.iter().zip(&step).map(|(x, d)| (x - d) * (x - d)).sum();
        let pr: f64 = 0.5 * (0..n).map(|i| step[i] * (g[i] + lambda * diag[i] * step[i])).sum::<f64>();
        let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1e-300);
        assert_eq!(s.grad_max, gmax);
        assert_eq!(s.grad_scaled, gs);
        assert_eq!(s.fault, None);
        assert!(rel(s.step2, step2) < 1e-13, "{} {}", s.step2, step2);
        assert!(rel(s.try2, try2) < 1e-13, "{} {}", s.try2, try2);
        assert!(rel(s.predicted, pr) < 1e-13, "{} {}", s.predicted, pr);
        let damped = read(&ctx, hd.view(nh * 8));
        for (i, &p) in pos.iter().enumerate() {
            assert_eq!(damped[p as usize], diag[i] + lambda * diag[i]);
        }
        let xt = read(&ctx, v.x_try());
        for i in 0..n {
            assert_eq!(xt[i], x[i] - step[i]);
        }
        v.accept();
        assert_eq!(read(&ctx, v.x()), xt);
    }

    #[test]
    fn the_first_faulty_diagonal_entry_is_found() {
        let ctx = GpuContext::new(GpuBackend::Cuda, 0).unwrap();
        let n = 700usize;
        let pos: Vec<u32> = (0..n as u32).collect();
        let g = vec![1.0f64; n];
        let up = |v: &[f64]| {
            let l = ctx.lease(v.len() * 8);
            ctx.client().write(l.handle(), Bytes::from_elems(v.to_vec()));
            l
        };
        let gd = up(&g);
        let v = LmVectors::new(&ctx, &pos);
        let fault = |h: &[f64], floor: Option<f64>| {
            let hd = up(h);
            v.linearized(gd.handle(), hd.handle(), n, floor);
            Scalars::decode(&crate::read_one(ctx.client(), v.scalars()).unwrap()).fault
        };
        let mut h = vec![1.0f64; n];
        assert_eq!(fault(&h, None), None);
        // in the third cube, then an earlier one in the second
        h[600] = 0.0;
        assert_eq!(fault(&h, None), Some((600, DiagFault::Zero)));
        // a floor lets a zero through
        assert_eq!(fault(&h, Some(1e-6)), None);
        h[300] = -1.0;
        assert_eq!(fault(&h, Some(1e-6)), Some((300, DiagFault::Negative)));
        h[299] = f64::NAN;
        assert_eq!(fault(&h, Some(1e-6)), Some((299, DiagFault::Nan)));
        assert_eq!(fault(&h, None), Some((299, DiagFault::Nan)));
    }
}
