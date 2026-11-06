//! The Levenberg-Marquardt loop's vector work on a [`Device`], for a solve
//! whose parameters, step and linearization stay there: what the loop
//! decides on comes back as a few scalars, read with the attempt's result.
//! The kernels are `kernels/lm.cpp`.
//!
//! Per linearization, `k_lin_pass` gathers H's diagonal, writes the
//! damping scale, and reduces the gradient's largest entry, the Jacobi-
//! scaled one and the first faulty diagonal entry; per attempt,
//! `k_attempt_pass` forms the trial point `x - delta` and reduces the
//! step's and the trial point's squared norms and the predicted
//! reduction. Both leave partials per block, which `k_combine_lin` and
//! `k_combine_attempt` fold into the scalars, at the slots below
//! ([`Scalars`]). [`LmVectors`] holds the blocks and launches the passes.

use std::sync::Arc;

use crate::{Arg, Device, GpuError, Launch, Layout, WIDE, Wide, define_u32s};

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
/// Fault codes are below this, and every route's no-fault value above it
/// ([`NO_FAULT`] in f64, 3e38 in df32).
pub const FAULT_BELOW: f64 = 1e30;

/// Partials per block of each pass.
pub const LIN_PARTIALS: usize = 3;
pub const ATTEMPT_PARTIALS: usize = 3;

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
        let v: Vec<f64> = (0..N_SCALARS).map(|i| bytemuck::pod_read_unaligned::<f64>(&bytes[8 * i..8 * i + 8])).collect();
        Self::from_values(&v)
    }

    /// The scalars from their [`N_SCALARS`] values, read back in any wide
    /// scalar.
    pub fn from_values(s: &[f64]) -> Scalars {
        let code = s[S_FAULT];
        let fault = (code < FAULT_BELOW).then(|| {
            let code = code as u64;
            let kind = match code % 4 {
                1 => DiagFault::Nan,
                2 => DiagFault::Negative,
                _ => DiagFault::Zero,
            };
            ((code / 4) as usize, kind)
        });
        Scalars {
            cost_x: s[S_COST_X],
            cost_try: s[S_COST_TRY],
            grad_max: s[S_GRAD_MAX],
            grad_scaled: s[S_GRAD_SCALED],
            fault,
            step2: s[S_STEP2],
            try2: s[S_TRY2],
            predicted: s[S_PREDICTED],
        }
    }
}

/// The kernels.
const SOURCE: &str = include_str!("../kernels/lm.cpp");

/// Threads per block of every kernel.
const THREADS: u32 = 256;

struct Kernels<K> {
    gather_diag: K,
    lin_pass: K,
    damp: K,
    attempt_pass: K,
    combine_lin: K,
    combine_attempt: K,
    copy: K,
    zero_steps: K,
}

/// Byte offsets of the block's sections.
struct Sections {
    /// H's scalar diagonal: its positions in H's values (u32).
    pos: usize,
    x: usize,
    x_try: usize,
    diag: usize,
    damp: usize,
    /// The damping of the next attempt, one f64.
    lambda: usize,
    lin: usize,
    att: usize,
    scalars: usize,
}

/// The Levenberg-Marquardt loop's vectors on a [`Device`], in the route's
/// [`Wide`] scalar, in one block: the parameters `x` and the trial point, H's undamped
/// diagonal and the damping scale, and the [`Scalars`](crate::lm::Scalars)
/// the loop decides on. H and the gradient are the caller's (a
/// [`GpuSchur`](crate::GpuSchur)'s targets).
pub struct LmVectors<D: Device> {
    dev: Arc<D>,
    wide: Wide,
    k: Kernels<D::Kernel>,
    n: usize,
    block: D::Block,
    sec: Sections,
}

/// Blocks of a pass over `n` entries, at least one (a pass over nothing
/// still leaves its partials).
fn blocks(n: usize) -> u32 {
    (n as u32).div_ceil(THREADS).max(1)
}

impl<D: Device> LmVectors<D> {
    /// The vectors of `pos.len()` parameters in `wide`, H's diagonal at
    /// `pos` in its values.
    pub fn new(dev: &Arc<D>, pos: &[u32], wide: Wide) -> Result<Self, GpuError> {
        let n = pos.len();
        let parts = blocks(n) as usize;
        let mut l = Layout::new();
        let sec = Sections {
            pos: l.take(n * 4),
            x: l.take(n * 8),
            x_try: l.take(n * 8),
            diag: l.take(n * 8),
            damp: l.take(n * 8),
            lambda: l.take(8),
            lin: l.take(parts * LIN_PARTIALS * 8),
            att: l.take(parts * ATTEMPT_PARTIALS * 8),
            scalars: l.take(N_SCALARS * 8),
        };
        let mut block = dev.alloc(l.bytes())?;
        if n > 0 {
            dev.write(&mut block, sec.pos, bytemuck::cast_slice(pos))?;
        }
        let mut defines = wide.defines().to_string();
        define_u32s(
            &mut defines,
            &[
                ("THREADS", THREADS as usize),
                ("LIN_PARTIALS", LIN_PARTIALS),
                ("ATTEMPT_PARTIALS", ATTEMPT_PARTIALS),
                ("S_GRAD_MAX", S_GRAD_MAX),
                ("S_GRAD_SCALED", S_GRAD_SCALED),
                ("S_FAULT", S_FAULT),
                ("S_STEP2", S_STEP2),
                ("S_TRY2", S_TRY2),
                ("S_PREDICTED", S_PREDICTED),
                ("FIXED", crate::FIXED as usize),
            ],
        );
        // above FAULT_BELOW in either scalar
        defines.push_str("#define NO_FAULT ARAEL_WIDE(1e300, 3.0e38f, 0.0f)\n");
        let source = format!("{WIDE}\n{SOURCE}");
        let entries = ["k_gather_diag", "k_lin_pass", "k_damp", "k_attempt_pass", "k_combine_lin", "k_combine_attempt", "k_copy", "k_zero_steps"];
        let [gather_diag, lin_pass, damp, attempt_pass, combine_lin, combine_attempt, copy, zero_steps]: [D::Kernel; 8] =
            dev.compile_many(&source, &defines, &entries)?.try_into().ok().expect("a kernel per entry point");
        let k = Kernels { gather_diag, lin_pass, damp, attempt_pass, combine_lin, combine_attempt, copy, zero_steps };
        Ok(LmVectors { dev: dev.clone(), wide, k, n, block, sec })
    }

    /// The parameter count.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether there are no parameters.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// The scalar the vectors are kept in.
    pub fn wide(&self) -> Wide {
        self.wide
    }

    /// Write the parameters (no wait).
    pub fn write_x(&mut self, x: &[f64]) -> Result<(), GpuError> {
        assert_eq!(x.len(), self.n);
        if self.n > 0 {
            self.dev.write(&mut self.block, self.sec.x, &self.wide.encode(x))?;
        }
        Ok(())
    }

    /// The parameters on the device (block, byte offset), `len` values.
    pub fn x(&self) -> (&D::Block, usize) {
        (&self.block, self.sec.x)
    }

    /// The trial point of the last [`step`](Self::step).
    pub fn x_try(&self) -> (&D::Block, usize) {
        (&self.block, self.sec.x_try)
    }

    /// H's undamped diagonal, as the last [`linearized`](Self::linearized)
    /// or [`gather_diag`](Self::gather_diag) left it.
    pub fn diag(&self) -> (&D::Block, usize) {
        (&self.block, self.sec.diag)
    }

    /// The scalars, [`N_SCALARS`] values, to read and decode with
    /// [`Wide::decode`] and [`Scalars::from_values`](crate::lm::Scalars::from_values).
    pub fn scalars(&self) -> (&D::Block, usize) {
        (&self.block, self.sec.scalars)
    }

    /// The scalar at `slot` (an `S_*` constant of [`crate::lm`]), a target for a cost.
    pub fn slot(&self, slot: usize) -> (&D::Block, usize) {
        assert!(slot < N_SCALARS);
        (&self.block, self.sec.scalars + 8 * slot)
    }

    fn run(&self, kernel: &D::Kernel, blocks: u32, args: &[Arg<'_, D::Block>]) -> Result<(), GpuError> {
        self.dev.launch(kernel, Launch { grid: [blocks, 1, 1], block: [THREADS, 1, 1], shared_bytes: 0 }, args)
    }

    fn at(&self, offset: usize) -> Arg<'_, D::Block> {
        Arg::Block(&self.block, offset)
    }

    /// The diagonal of H (whose values start at `h`) gathered.
    pub fn gather_diag(&self, h: (&D::Block, usize)) -> Result<(), GpuError> {
        let s = &self.sec;
        self.run(&self.k.gather_diag, blocks(self.n), &[Arg::Block(h.0, h.1), self.at(s.pos), self.at(s.diag), Arg::U32(self.n as u32)])
    }

    /// After an assembly into `g` and `h`: the diagonal, the damping scale
    /// (`max(diag, floor)` with a floor) and the linearization's scalars.
    pub fn linearized(&self, g: (&D::Block, usize), h: (&D::Block, usize), floor: Option<f64>) -> Result<(), GpuError> {
        let (s, parts) = (&self.sec, blocks(self.n));
        let args = [
            Arg::Block(g.0, g.1),
            Arg::Block(h.0, h.1),
            self.at(s.pos),
            self.at(s.diag),
            self.at(s.damp),
            self.wide.arg(floor.unwrap_or(0.0)),
            Arg::U32(floor.is_some() as u32),
            self.at(s.lin),
            Arg::U32(self.n as u32),
        ];
        self.run(&self.k.lin_pass, parts, &args)?;
        self.run(&self.k.combine_lin, 1, &[self.at(s.lin), Arg::U32(parts), self.at(s.scalars)])
    }

    /// The damping of the next [`damp`](Self::damp) and
    /// [`step`](Self::step), written to the device (no wait).
    pub fn set_lambda(&mut self, lambda: f64) -> Result<(), GpuError> {
        self.dev.write(&mut self.block, self.sec.lambda, &self.wide.bits(lambda).to_le_bytes())
    }

    /// H's diagonal damped, `diag + lambda damp`, in its values at `h`.
    pub fn damp(&self, h: (&D::Block, usize)) -> Result<(), GpuError> {
        let s = &self.sec;
        let args = [Arg::Block(h.0, h.1), self.at(s.pos), self.at(s.diag), self.at(s.damp), self.at(s.lambda), Arg::U32(self.n as u32)];
        self.run(&self.k.damp, blocks(self.n), &args)
    }

    /// The trial point `x - step` and the attempt's scalars, for the step
    /// solved at the set lambda against the gradient `g`.
    pub fn step(&self, step: (&D::Block, usize), g: (&D::Block, usize)) -> Result<(), GpuError> {
        let (s, parts) = (&self.sec, blocks(self.n));
        let args = [
            self.at(s.x),
            Arg::Block(step.0, step.1),
            Arg::Block(g.0, g.1),
            self.at(s.diag),
            self.at(s.lambda),
            self.at(s.x_try),
            self.at(s.att),
            Arg::U32(self.n as u32),
        ];
        self.run(&self.k.attempt_pass, parts, &args)?;
        self.run(&self.k.combine_attempt, 1, &[self.at(s.att), Arg::U32(parts), self.at(s.scalars)])
    }

    /// The trial point becomes the parameters: copied, so the sections keep
    /// their roles (and a captured graph its addresses).
    pub fn accept(&self) -> Result<(), GpuError> {
        let s = &self.sec;
        self.run(&self.k.copy, blocks(self.n), &[self.at(s.x), self.at(s.x_try), Arg::U32(self.n as u32)])
    }

    /// The rotation steps of the parameters zeroed, for `entities`
    /// entities: each one's three global indices at `idx[at[e]..]` (u32), a
    /// fixed one ([`FIXED`](crate::FIXED)) left alone; what
    /// re-centring does to the parameters after an accepted step.
    pub fn zero_steps(&self, idx: (&D::Block, usize), at: (&D::Block, usize), entities: usize) -> Result<(), GpuError> {
        if entities == 0 {
            return Ok(());
        }
        let args = [self.at(self.sec.x), Arg::Block(idx.0, idx.1), Arg::Block(at.0, at.1), Arg::U32(entities as u32)];
        self.run(&self.k.zero_steps, (entities as u32).div_ceil(THREADS), &args)
    }
}

#[cfg(all(test, any(feature = "cuda", feature = "hip")))]
mod tests {
    use super::*;
    use crate::lm::{DiagFault, Scalars};
    use crate::{TestBlock, TestDevice};

    fn device() -> Arc<TestDevice> {
        Arc::new(TestDevice::new(0).unwrap())
    }

    fn up<T: bytemuck::Pod>(dev: &TestDevice, v: &[T]) -> TestBlock {
        let mut b = dev.alloc(core::mem::size_of_val(v)).unwrap();
        dev.write(&mut b, 0, bytemuck::cast_slice(v)).unwrap();
        b
    }

    fn read(dev: &TestDevice, at: (&TestBlock, usize), n: usize) -> Vec<f64> {
        let mut v = vec![0.0f64; n];
        dev.read(at.0, at.1, bytemuck::cast_slice_mut(&mut v)).unwrap();
        v
    }

    fn scalars(dev: &TestDevice, v: &LmVectors<TestDevice>) -> Scalars {
        Scalars::from_values(&readw(dev, v.scalars(), N_SCALARS, v.wide()))
    }

    /// `v` uploaded in `wide`.
    fn upw(dev: &TestDevice, v: &[f64], wide: Wide) -> TestBlock {
        let bytes = wide.encode(v);
        let mut b = dev.alloc(bytes.len()).unwrap();
        dev.write(&mut b, 0, &bytes).unwrap();
        b
    }

    /// `n` values of `wide` read back.
    fn readw(dev: &TestDevice, at: (&TestBlock, usize), n: usize, wide: Wide) -> Vec<f64> {
        let mut bytes = vec![0u8; 8 * n];
        dev.read(at.0, at.1, &mut bytes).unwrap();
        let mut v = vec![0.0f64; n];
        wide.decode(&bytes, &mut v);
        v
    }

    /// The passes in df32 against the host's f64: to df32's precision, the
    /// maxima and the copies exact.
    #[test]
    fn the_passes_match_the_host_in_df32() {
        let wide = Wide::Df32;
        let dev = device();
        let Fixture { n, pos, h, g, x, step } = fixture();
        // the inputs as df32 holds them
        let held = |v: &[f64]| v.iter().map(|&a| wide.value(wide.bits(a))).collect::<Vec<f64>>();
        let (h, g, x, step) = (held(&h), held(&g), held(&x), held(&step));
        let nh = h.len();
        let lambda = 0.3;
        let (hd, gd, sd) = (upw(&dev, &h, wide), upw(&dev, &g, wide), upw(&dev, &step, wide));
        let mut v = LmVectors::new(&dev, &pos, wide).unwrap();
        v.write_x(&x).unwrap();
        v.linearized((&gd, 0), (&hd, 0), None).unwrap();
        v.set_lambda(lambda).unwrap();
        v.damp((&hd, 0)).unwrap();
        v.step((&sd, 0), (&gd, 0)).unwrap();
        let s = scalars(&dev, &v);
        let diag: Vec<f64> = pos.iter().map(|&p| h[p as usize]).collect();
        let gmax = g.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let gs = g.iter().zip(&diag).fold(0.0f64, |m, (g, d)| m.max(g.abs() / d.sqrt()));
        let step2: f64 = step.iter().map(|d| d * d).sum();
        let try2: f64 = x.iter().zip(&step).map(|(x, d)| (x - d) * (x - d)).sum();
        let pr: f64 = 0.5 * (0..n).map(|i| step[i] * (g[i] + lambda * diag[i] * step[i])).sum::<f64>();
        let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1e-300);
        assert_eq!(s.grad_max, gmax);
        assert!(rel(s.grad_scaled, gs) < 1e-13, "{} {}", s.grad_scaled, gs);
        assert_eq!(s.fault, None);
        assert!(rel(s.step2, step2) < 1e-12, "{} {}", s.step2, step2);
        assert!(rel(s.try2, try2) < 1e-12, "{} {}", s.try2, try2);
        assert!(rel(s.predicted, pr) < 1e-12, "{} {}", s.predicted, pr);
        assert_eq!(readw(&dev, v.diag(), n, wide), diag);
        let damped = readw(&dev, (&hd, 0), nh, wide);
        for (i, &p) in pos.iter().enumerate() {
            assert!(rel(damped[p as usize], diag[i] + lambda * diag[i]) < 1e-13, "entry {i}");
        }
        let xt = readw(&dev, v.x_try(), n, wide);
        for i in 0..n {
            assert!((xt[i] - (x[i] - step[i])).abs() <= 1e-13 * x[i].abs().max(1.0), "entry {i}");
        }
        v.accept().unwrap();
        assert_eq!(readw(&dev, v.x(), n, wide), xt);
    }

    /// The fault's code is exact in df32, and its no-fault value reads as none.
    #[test]
    fn the_first_faulty_diagonal_entry_is_found_in_df32() {
        let wide = Wide::Df32;
        let dev = device();
        let n = 700usize;
        let pos: Vec<u32> = (0..n as u32).collect();
        let gd = upw(&dev, &vec![1.0f64; n], wide);
        let v = LmVectors::new(&dev, &pos, wide).unwrap();
        let fault = |h: &[f64]| {
            let hd = upw(&dev, h, wide);
            v.linearized((&gd, 0), (&hd, 0), None).unwrap();
            scalars(&dev, &v).fault
        };
        let mut h = vec![1.0f64; n];
        assert_eq!(fault(&h), None);
        h[600] = 0.0;
        assert_eq!(fault(&h), Some((600, DiagFault::Zero)));
        h[299] = f64::NAN;
        assert_eq!(fault(&h), Some((299, DiagFault::Nan)));
    }

    /// The passes' inputs: past one block, so the partials fold.
    struct Fixture {
        n: usize,
        pos: Vec<u32>,
        h: Vec<f64>,
        g: Vec<f64>,
        x: Vec<f64>,
        step: Vec<f64>,
    }

    fn fixture() -> Fixture {
        let n = 1000usize;
        let pos: Vec<u32> = (0..n as u32).map(|i| 3 * i + 1).collect();
        let mut h = vec![0.0f64; 3 * n];
        for i in 0..n {
            h[3 * i + 1] = 0.5 + (i % 7) as f64;
        }
        let g: Vec<f64> = (0..n).map(|i| ((i * 37) % 11) as f64 - 5.0).collect();
        let x: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let step: Vec<f64> = (0..n).map(|i| 1e-2 * (i as f64).cos()).collect();
        Fixture { n, pos, h, g, x, step }
    }

    #[test]
    fn the_passes_match_the_host() {
        let dev = device();
        let Fixture { n, pos, h, g, x, step } = fixture();
        let nh = h.len();
        let lambda = 0.3;
        let (hd, gd, sd) = (up(&dev, &h), up(&dev, &g), up(&dev, &step));
        let mut v = LmVectors::new(&dev, &pos, Wide::F64).unwrap();
        v.write_x(&x).unwrap();
        v.linearized((&gd, 0), (&hd, 0), None).unwrap();
        v.set_lambda(lambda).unwrap();
        v.damp((&hd, 0)).unwrap();
        v.step((&sd, 0), (&gd, 0)).unwrap();
        let s = scalars(&dev, &v);
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
        assert_eq!(read(&dev, v.diag(), n), diag);
        let damped = read(&dev, (&hd, 0), nh);
        for (i, &p) in pos.iter().enumerate() {
            assert_eq!(damped[p as usize], diag[i] + lambda * diag[i]);
        }
        let xt = read(&dev, v.x_try(), n);
        for i in 0..n {
            assert_eq!(xt[i], x[i] - step[i]);
        }
        v.accept().unwrap();
        assert_eq!(read(&dev, v.x(), n), xt);
        // the gather alone, from the damped values
        v.gather_diag((&hd, 0)).unwrap();
        assert_eq!(read(&dev, v.diag(), n), pos.iter().map(|&p| damped[p as usize]).collect::<Vec<_>>());
    }

    #[test]
    fn the_first_faulty_diagonal_entry_is_found() {
        let dev = device();
        let n = 700usize;
        let pos: Vec<u32> = (0..n as u32).collect();
        let gd = up(&dev, &vec![1.0f64; n]);
        let v = LmVectors::new(&dev, &pos, Wide::F64).unwrap();
        let fault = |h: &[f64], floor: Option<f64>| {
            let hd = up(&dev, h);
            v.linearized((&gd, 0), (&hd, 0), floor).unwrap();
            scalars(&dev, &v).fault
        };
        let mut h = vec![1.0f64; n];
        assert_eq!(fault(&h, None), None);
        // in the third block, then an earlier one in the second
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

    /// The damping scale is the diagonal floored when a floor is given.
    #[test]
    fn a_floor_raises_the_damping_scale() {
        let dev = device();
        let n = 300usize;
        let pos: Vec<u32> = (0..n as u32).collect();
        let h: Vec<f64> = (0..n).map(|i| if i % 3 == 0 { 1e-9 } else { 2.0 }).collect();
        let (hd, gd) = (up(&dev, &h), up(&dev, &vec![1.0f64; n]));
        let mut v = LmVectors::new(&dev, &pos, Wide::F64).unwrap();
        v.linearized((&gd, 0), (&hd, 0), Some(1e-3)).unwrap();
        v.set_lambda(2.0).unwrap();
        v.damp((&hd, 0)).unwrap();
        let damped = read(&dev, (&hd, 0), n);
        for i in 0..n {
            assert_eq!(damped[i], h[i] + 2.0 * h[i].max(1e-3), "entry {i}");
        }
    }

    /// Each entity's three rotation steps at idx[at[e]..] are zeroed, a
    /// fixed index skipped, everything else left alone.
    #[test]
    fn zero_steps_zero_the_rotation_steps() {
        let dev = device();
        let n = 40usize;
        let pos: Vec<u32> = (0..n as u32).collect();
        let mut v = LmVectors::new(&dev, &pos, Wide::F64).unwrap();
        let x: Vec<f64> = (0..n).map(|i| 1.0 + i as f64).collect();
        v.write_x(&x).unwrap();
        // two entities of six parameters (rotation first), the second's middle angle fixed
        let fixed = crate::FIXED;
        let idx: Vec<u32> = vec![10, 11, 12, 13, 14, 15, 30, fixed, 32, 33, 34, 35];
        let at: Vec<u32> = vec![0, 6];
        let (id, ad) = (up(&dev, &idx), up(&dev, &at));
        v.zero_steps((&id, 0), (&ad, 0), 2).unwrap();
        let got = read(&dev, v.x(), n);
        for i in 0..n {
            let zeroed = [10, 11, 12, 30, 32].contains(&i);
            assert_eq!(got[i], if zeroed { 0.0 } else { x[i] }, "entry {i}");
        }
    }

    /// An LM attempt resident on the device, composed from the reduction's
    /// targets and views and these passes: the undamped H in place, its
    /// diagonal and the linearization's scalars, the damping, the Schur
    /// solve, the trial point; captured once as a graph and replayed at
    /// another lambda, each against the damped system solved on the host.
    #[test]
    fn an_attempt_on_the_device_solves_the_damped_system() {
        for wide in [Wide::F64, Wide::Df32] {
            attempt_in(wide);
        }
    }

    fn attempt_in(wide: Wide) {
        use crate::GpuPrecision;
        use crate::{GpuFactor, GpuSchur};
        use crate::test_util::{ba_like, rel_residual};
        use arael_faer::schur::schur_symbolic;
        use arael_faer::supernodal::{SupernodalParams, SupernodalSymbolic, nd_block_order};
        let (h, dense, rhs, elim) = ba_like(30, 250, 7);
        let hs = h.symbolic().clone();
        let n = hs.ncols();
        let sym = schur_symbolic(&hs, &elim).unwrap();
        let order = nd_block_order(&sym.s);
        let sn = SupernodalSymbolic::new(&sym.s, Some(&order), &SupernodalParams::default()).unwrap();
        let mut pos = vec![0u32; n];
        for j in 0..hs.nblk_cols() {
            let t = hs.col_range(j).find(|&b| hs.blk_row(b) == j).unwrap();
            let (w, _) = hs.block_dims(t);
            for i in 0..w {
                pos[hs.col_span(j).start + i] = (hs.val_range(t).start + i * (w + 1)) as u32;
            }
        }
        let dev = device();
        let mut g = GpuSchur::new(&dev, &sym, &hs, wide).unwrap();
        let f = GpuFactor::new(&dev, GpuPrecision::F32, &sn, &sym.s, &Default::default(), wide).unwrap();
        let mut v = LmVectors::new(&dev, &pos, wide).unwrap();
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).cos()).collect();
        v.write_x(&x).unwrap();
        g.upload(h.vals(), &rhs).unwrap();
        v.linearized(g.rhs_target(), g.h_target(), None).unwrap();
        let refine = Some((4usize, 1e-10f64));
        let attempt = |g: &GpuSchur<TestDevice>, v: &LmVectors<TestDevice>, f: &GpuFactor<TestDevice>| {
            v.damp(g.h_target())?;
            g.launch(f, refine)?;
            v.step(g.step_view(), g.rhs_target())
        };
        let check = |v: &LmVectors<TestDevice>, lambda: f64| {
            let s = scalars(&dev, v);
            let step = readw(&dev, g.step_view(), n, wide);
            let mut damped = dense.clone();
            for i in 0..n {
                damped[i + i * n] *= 1.0 + lambda;
            }
            let res = rel_residual(&damped, n, &step, &rhs);
            assert!(res < 1e-8, "{wide:?}, lambda {lambda}: residual {res}");
            let step2: f64 = step.iter().map(|d| d * d).sum();
            assert!((s.step2 - step2).abs() <= 1e-12 * step2, "{} {}", s.step2, step2);
            // the parameters and the gradient as the wide scalar holds them
            let held = |a: f64| wide.value(wide.bits(a));
            let xt = readw(&dev, v.x_try(), n, wide);
            assert!((0..n).all(|i| (xt[i] - (held(x[i]) - step[i])).abs() <= 1e-14 * xt[i].abs().max(1.0)));
            let gmax = rhs.iter().fold(0.0f64, |m, &v| m.max(held(v).abs()));
            assert_eq!(s.grad_max, gmax);
        };
        v.set_lambda(0.5).unwrap();
        attempt(&g, &v, &f).unwrap();
        check(&v, 0.5);
        let graph = dev.capture(|| attempt(&g, &v, &f)).unwrap();
        for lambda in [2.0, 1e-3] {
            v.set_lambda(lambda).unwrap();
            dev.replay(&graph).unwrap();
            check(&v, lambda);
        }
    }

}
