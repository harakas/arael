//! The device assembly on arael's own GPU runtime (feature
//! `gpu-native-cuda`): what [`DeviceAssembly`](crate::device::DeviceAssembly)
//! does, with the root's generated C++ kernels
//! ([`DeviceRoot::device_cpp`]) and the model's data laid out as columns in
//! one block of device memory.
//!
//! The block starts with a layout table: the byte offset of every column,
//! one `u64` each, in the order [`columns`] gives them. The kernels read a
//! column through its slot of the table (`ARAEL_COL_<name>`), so a root's
//! kernels compile once whatever the problem's size, and only the offsets
//! change with it.

use std::collections::HashMap;
use std::sync::Arc;

use arael_gpu::native::{Arg, Device, Launch, NativeError};

use crate::device::{DevicePack, DeviceRoot, DeviceShape};

/// A fixed parameter's global index, as the tables hold it.
const FIXED: u32 = u32::MAX;

/// Threads per block of every assembly kernel (`kernels/assembly.cpp`).
const THREADS: u32 = 256;

/// One column of the model block: its name in the kernels, the size of an
/// element in bytes, and its element count.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Column {
    name: String,
    elem: usize,
    len: usize,
}

/// The columns of a root, in the layout table's order: per entity array its
/// global indices, tiles, fixed values and gathered values; per rotation
/// field its step offsets, reference rotations and composed rotations; per
/// cross array its tiles; per constraint its slots and data leaves. The
/// order depends on the shape alone.
fn columns(shape: &DeviceShape, slots: &[usize], cross_slots: &[usize], counts: &[usize], ts: usize) -> Vec<Column> {
    let col = |name: String, elem: usize, len: usize| Column { name, elem, len };
    let mut v = Vec::new();
    for (e, &n) in shape.entities.iter().enumerate() {
        let s = slots[e];
        v.push(col(format!("E{e}_IDX"), 4, n * s));
        v.push(col(format!("E{e}_TILE"), 4, 3 * s));
        v.push(col(format!("E{e}_FIXED"), ts, n * s));
        v.push(col(format!("E{e}_EV"), ts, n * s));
    }
    for (r, &(e, _)) in shape.rotations.iter().enumerate() {
        let s = slots[e];
        v.push(col(format!("R{r}_AT"), 4, s));
        v.push(col(format!("R{r}_REFR"), 8, 9 * s));
        v.push(col(format!("R{r}_ROT"), ts, 9 * s));
        v.push(col(format!("R{r}_ROTW"), 8, 9 * s));
        v.push(col(format!("R{r}_DROT"), ts, 27 * s));
    }
    for (x, &s) in cross_slots.iter().enumerate() {
        v.push(col(format!("X{x}_TILE"), 4, 3 * s));
    }
    for (c, &(_, nd)) in shape.constraints.iter().enumerate() {
        let n = counts[c];
        for w in ["SA", "SB", "SX"] {
            v.push(col(format!("C{c}_{w}"), 4, n));
        }
        for j in 0..nd {
            v.push(col(format!("C{c}_D{j}"), ts, n));
        }
    }
    v
}

/// Where every column sits in the block: the table first, then each column
/// on a 256-byte boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Layout {
    columns: Vec<Column>,
    offsets: Vec<usize>,
    bytes: usize,
}

impl Layout {
    fn new(columns: Vec<Column>) -> Self {
        let align = |b: usize| b.div_ceil(256) * 256;
        let mut at = align(columns.len() * 8);
        let mut offsets = Vec::with_capacity(columns.len());
        for c in &columns {
            offsets.push(at);
            at += align((c.elem * c.len).max(1));
        }
        Layout { columns, offsets, bytes: at }
    }

    /// The `#define`s of the table's slots, for the kernels.
    fn defines(&self) -> String {
        self.columns.iter().enumerate().map(|(k, c)| format!("#define ARAEL_COL_{} {k}\n", c.name)).collect()
    }

    fn index(&self) -> HashMap<String, usize> {
        self.columns.iter().zip(&self.offsets).map(|(c, &o)| (c.name.clone(), o)).collect()
    }
}

/// The kernels of a root, compiled once per device and layout of slots.
struct Kernels<K> {
    gather: K,
    euler_values: K,
    euler_rotations: K,
    euler_advance: K,
    zero: K,
    zero_steps: K,
    gh: Vec<K>,
    cost: Vec<K>,
}

/// A buffer the assembly writes or reads that is not its own: a block and a
/// byte offset into it.
pub type Target<'a, B> = (&'a B, usize);

/// A root laid out on a [`Device`] in one block, its tables, instances and
/// entity data uploaded once per structure and refreshed per solve; the
/// parameter vector, the targets and the cost are the caller's buffers.
pub struct NativeDeviceAssembly<T, D: Device> {
    dev: Arc<D>,
    shape: DeviceShape,
    slots: Vec<usize>,
    cross_slots: Vec<usize>,
    counts: Vec<usize>,
    /// The structure's tables, kept for the block image of each refresh.
    idx: Vec<Vec<u32>>,
    tile: Vec<Vec<u32>>,
    cross: Vec<Vec<u32>>,
    at: Vec<Vec<u32>>,
    /// Per rotation field, per slot: the global index of the rotation's
    /// first parameter, [`FIXED`] when fixed.
    rot_index: Vec<Vec<u32>>,
    layout: Layout,
    offset: HashMap<String, usize>,
    block: D::Block,
    kernels: Kernels<D::Kernel>,
    generation: u64,
    _t: core::marker::PhantomData<fn() -> T>,
}

fn launch_n(n: usize) -> Launch {
    Launch::linear((n as u32).max(1), THREADS)
}

impl<T: crate::utils::Float + 'static, D: Device> NativeDeviceAssembly<T, D> {
    /// Lay `root` out on `dev` against the tiled pattern `resolve`
    /// describes, and compile its kernels. The model's working copies must
    /// be current (its fixed values and reference rotations are read).
    pub fn new(
        dev: &Arc<D>,
        root: &dyn DeviceRoot<T>,
        resolve: &mut dyn FnMut(u32, u32) -> (usize, usize, usize),
    ) -> Result<Self, NativeError> {
        let ts = core::mem::size_of::<T>();
        assert!(ts == 4 || ts == 8, "the device assembly takes an f32 or an f64 model");
        let shape = root.device_shape();
        let tables = root.device_tables(resolve);
        let slots: Vec<usize> = shape.entities.iter().zip(&tables.idx).map(|(&n, ix)| ix.len() / n.max(1)).collect();
        let cross_slots: Vec<usize> = tables.cross.iter().map(|c| c.len() / 3).collect();
        let at: Vec<Vec<u32>> =
            shape.rotations.iter().map(|&(e, off)| (0..slots[e]).map(|s| (s * shape.entities[e] + off) as u32).collect()).collect();
        let rot_index =
            shape.rotations.iter().map(|&(e, off)| (0..slots[e]).map(|s| tables.idx[e][s * shape.entities[e] + off]).collect()).collect();
        let mut pack = DevicePack::<T>::new(&shape, &slots);
        root.device_pack(&mut pack);
        let counts = instance_counts(&shape, &pack);
        let layout = Layout::new(columns(&shape, &slots, &cross_slots, &counts, ts));
        let scalar = if ts == 8 { "double" } else { "float" };
        let defines = format!("#define SCALAR {scalar}\n{}", layout.defines());
        let source = format!("{}\n{}", arael_gpu::native::ASSEMBLY, root.device_cpp());
        let k = |entry: &str| dev.compile(&source, &defines, entry);
        let kernels = Kernels {
            gather: k("k_gather_params")?,
            euler_values: k("k_euler_values")?,
            euler_rotations: k("k_euler_rotations")?,
            euler_advance: k("k_euler_advance")?,
            zero: k("k_zero_f64")?,
            zero_steps: k("k_zero_steps")?,
            gh: (0..shape.constraints.len()).map(|c| k(&format!("__dk{c}_gh"))).collect::<Result<_, _>>()?,
            cost: (0..shape.constraints.len()).map(|c| k(&format!("__dk{c}_cost"))).collect::<Result<_, _>>()?,
        };
        let block = dev.alloc(layout.bytes)?;
        let offset = layout.index();
        let mut asm = NativeDeviceAssembly {
            dev: dev.clone(),
            shape,
            slots,
            cross_slots,
            counts,
            idx: tables.idx,
            tile: tables.tile,
            cross: tables.cross,
            at,
            rot_index,
            layout,
            offset,
            block,
            kernels,
            generation: 0,
            _t: core::marker::PhantomData,
        };
        asm.write(&pack)?;
        Ok(asm)
    }

    /// Read the model's data again: the constraint instances and their
    /// data, the entities' fixed values and the reference rotations, which
    /// a solve may have been handed changed (the structure is the same).
    /// The block keeps its place while the layout is the same, so graphs
    /// recorded against it stay valid; otherwise a new block is laid out
    /// and the [`generation`](Self::generation) moves on. The model's
    /// working copies must be current.
    pub fn refresh(&mut self, root: &dyn DeviceRoot<T>) -> Result<(), NativeError> {
        let mut pack = DevicePack::<T>::new(&self.shape, &self.slots);
        root.device_pack(&mut pack);
        let counts = instance_counts(&self.shape, &pack);
        if counts != self.counts {
            // The slots of the table are the same, so the kernels are too;
            // only the offsets move.
            let ts = core::mem::size_of::<T>();
            let layout = Layout::new(columns(&self.shape, &self.slots, &self.cross_slots, &counts, ts));
            if layout.bytes > self.layout.bytes {
                self.block = self.dev.alloc(layout.bytes)?;
            }
            self.offset = layout.index();
            self.layout = layout;
            self.counts = counts;
            self.generation += 1;
        }
        self.write(&pack)
    }

    /// The block's image on the host, written in one copy: the table, the
    /// structure's tables and the model's data. The columns the kernels
    /// write (the gathered values and the rotations) are left zero.
    fn write(&mut self, pack: &DevicePack<T>) -> Result<(), NativeError> {
        let mut image = vec![0u8; self.layout.bytes];
        for (k, &o) in self.layout.offsets.iter().enumerate() {
            image[8 * k..8 * k + 8].copy_from_slice(&(o as u64).to_le_bytes());
        }
        let put = |image: &mut Vec<u8>, name: &str, bytes: &[u8]| {
            let o = self.offset[name];
            image[o..o + bytes.len()].copy_from_slice(bytes);
        };
        for e in 0..self.shape.entities.len() {
            put(&mut image, &format!("E{e}_IDX"), u32_bytes(&self.idx[e]));
            put(&mut image, &format!("E{e}_TILE"), u32_bytes(&self.tile[e]));
            put(&mut image, &format!("E{e}_FIXED"), scalar_bytes(&pack.fixed[e]));
        }
        for r in 0..self.shape.rotations.len() {
            put(&mut image, &format!("R{r}_AT"), u32_bytes(&self.at[r]));
            put(&mut image, &format!("R{r}_REFR"), f64_bytes(&pack.refr[r]));
        }
        for x in 0..self.cross.len() {
            put(&mut image, &format!("X{x}_TILE"), u32_bytes(&self.cross[x]));
        }
        // the walk's instances and data, one record per instance, into columns
        let ts = core::mem::size_of::<T>();
        for (c, &(w, nd)) in self.shape.constraints.iter().enumerate() {
            let n = self.counts[c];
            assert_eq!(w, 3, "an instance is two entity slots and a cross slot");
            for (k, name) in ["SA", "SB", "SX"].iter().enumerate() {
                let o = self.offset[&format!("C{c}_{name}")];
                for i in 0..n {
                    image[o + 4 * i..o + 4 * i + 4].copy_from_slice(&pack.inst[c][3 * i + k].to_le_bytes());
                }
            }
            let data = scalar_bytes(&pack.data[c]);
            for j in 0..nd {
                let o = self.offset[&format!("C{c}_D{j}")];
                for i in 0..n {
                    let src = (i * nd + j) * ts;
                    image[o + ts * i..o + ts * i + ts].copy_from_slice(&data[src..src + ts]);
                }
            }
        }
        self.dev.write(&mut self.block, 0, &image)
    }

    /// Changes whenever the block or its offsets change: a graph of the
    /// launches replays only under the generation it was captured in.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// A column of the block as a kernel argument.
    fn col(&self, name: &str) -> Arg<'_, D::Block> {
        Arg::Block(&self.block, self.offset[name])
    }

    fn run(&self, kernel: &D::Kernel, n: usize, args: &[Arg<'_, D::Block>]) -> Result<(), NativeError> {
        if n == 0 {
            return Ok(());
        }
        self.dev.launch(kernel, launch_n(n), args)
    }

    fn zero(&self, t: Target<'_, D::Block>, n: usize) -> Result<(), NativeError> {
        self.run(&self.kernels.zero, n, &[Arg::Block(t.0, t.1), Arg::U32(n as u32)])
    }

    /// The entities' parameter values and the rotations at the `nx` f64
    /// parameters `x`; with `derivatives`, the rotations' derivatives too.
    fn gather(&self, x: Target<'_, D::Block>, derivatives: bool) -> Result<(), NativeError> {
        for (e, &n) in self.shape.entities.iter().enumerate() {
            let len = n * self.slots[e];
            let args = [Arg::Block(x.0, x.1), self.col(&format!("E{e}_IDX")), self.col(&format!("E{e}_FIXED")), self.col(&format!("E{e}_EV")), Arg::U32(len as u32)];
            self.run(&self.kernels.gather, len, &args)?;
        }
        for (r, &(e, _)) in self.shape.rotations.iter().enumerate() {
            let s = self.slots[e];
            let (refr, ev, at, rotw, rot) =
                (self.col(&format!("R{r}_REFR")), self.col(&format!("E{e}_EV")), self.col(&format!("R{r}_AT")), self.col(&format!("R{r}_ROTW")), self.col(&format!("R{r}_ROT")));
            if derivatives {
                let args = [refr, ev, at, rotw, rot, self.col(&format!("R{r}_DROT")), Arg::U32(s as u32)];
                self.run(&self.kernels.euler_rotations, s, &args)?;
            } else {
                self.run(&self.kernels.euler_values, s, &[refr, ev, at, rotw, rot, Arg::U32(s as u32)])?;
            }
        }
        Ok(())
    }

    /// The gradient (`ng` f64 at `g`), the pattern's tile values (`nh` f64
    /// at `h`) and the cost (one f64 at `cost`) at the f64 parameters `x`,
    /// each zeroed first; nothing read back.
    pub fn assemble_at(
        &self,
        x: Target<'_, D::Block>,
        g: Target<'_, D::Block>,
        ng: usize,
        h: Target<'_, D::Block>,
        nh: usize,
        cost: Target<'_, D::Block>,
    ) -> Result<(), NativeError> {
        self.zero(g, ng)?;
        self.zero(h, nh)?;
        self.zero(cost, 1)?;
        self.gather(x, true)?;
        for (c, &n) in self.counts.iter().enumerate() {
            let args = [Arg::Block(&self.block, 0), Arg::Block(h.0, h.1), Arg::Block(g.0, g.1), Arg::Block(cost.0, cost.1), Arg::U32(n as u32)];
            self.run(&self.kernels.gh[c], n, &args)?;
        }
        Ok(())
    }

    /// The cost (one f64 at `cost`) at the f64 parameters `x`, zeroed
    /// first; the rotations' derivatives are not computed. Nothing read back.
    pub fn cost_at(&self, x: Target<'_, D::Block>, cost: Target<'_, D::Block>) -> Result<(), NativeError> {
        self.zero(cost, 1)?;
        self.gather(x, false)?;
        for (c, &n) in self.counts.iter().enumerate() {
            let args = [Arg::Block(&self.block, 0), Arg::Block(cost.0, cost.1), Arg::U32(n as u32)];
            self.run(&self.kernels.cost[c], n, &args)?;
        }
        Ok(())
    }

    /// The rotation steps of the f64 parameters `x` zeroed, as re-centring
    /// after [`advance`](Self::advance) leaves them.
    pub fn zero_steps(&self, x: Target<'_, D::Block>) -> Result<(), NativeError> {
        for (r, &(e, _)) in self.shape.rotations.iter().enumerate() {
            let s = self.slots[e];
            let args = [Arg::Block(x.0, x.1), self.col(&format!("E{e}_IDX")), self.col(&format!("R{r}_AT")), Arg::U32(s as u32)];
            self.run(&self.kernels.zero_steps, s, &args)?;
        }
        Ok(())
    }

    /// The reference rotations moved by the steps of the last gathered `x`
    /// (an accepted step's), as the model's `advance_params` moves its own.
    pub fn advance(&self) -> Result<(), NativeError> {
        for (r, &(e, _)) in self.shape.rotations.iter().enumerate() {
            let s = self.slots[e];
            let args = [self.col(&format!("R{r}_REFR")), self.col(&format!("E{e}_EV")), self.col(&format!("R{r}_AT")), Arg::U32(s as u32)];
            self.run(&self.kernels.euler_advance, s, &args)?;
        }
        Ok(())
    }

    /// Write the device's reference rotations into the model's optimized
    /// rotations ([`DeviceRoot::device_restore`]), one read per rotation
    /// field.
    pub fn restore(&self, root: &mut dyn DeviceRoot<T>) -> Result<(), NativeError> {
        let mut refr: Vec<Vec<f64>> = Vec::new();
        for (r, &(e, _)) in self.shape.rotations.iter().enumerate() {
            let mut v = vec![0f64; 9 * self.slots[e]];
            self.dev.read(&self.block, self.offset[&format!("R{r}_REFR")], f64_bytes_mut(&mut v))?;
            refr.push(v);
        }
        let mut at = HashMap::new();
        for (r, ix) in self.rot_index.iter().enumerate() {
            for (s, &g) in ix.iter().enumerate() {
                if g != FIXED {
                    at.insert(g, (r, s));
                }
            }
        }
        let mut found = std::collections::HashSet::new();
        root.device_restore(&mut |g| {
            let &(r, s) = at.get(&g)?;
            found.insert(g);
            Some(refr[r][9 * s..9 * s + 9].try_into().unwrap())
        });
        assert_eq!(found.len(), at.len(), "the model's walk reaches every rotation the device re-centred");
        Ok(())
    }

    /// The device the assembly runs on.
    pub fn device(&self) -> &Arc<D> {
        &self.dev
    }
}

/// Assemble `problem` at `x` on the CPU and on `dev` through the native
/// assembly against one tiled pattern (the reduced route's), and compare
/// (for tests), as [`device_check`](crate::device::device_check) does for
/// the CubeCL route.
pub fn native_device_check<T, D>(problem: &mut dyn crate::simple_lm::LmProblemInternals<T>, x: &[T], dev: &Arc<D>) -> crate::device::DeviceCheck
where
    T: crate::utils::Float + 'static,
    D: Device,
{
    use crate::store::{HessianBinder, PositionStream};
    let mut ctx = crate::threads::Context::new();
    problem.begin_with_context(&mut ctx);
    let (mut cells, mut spans) = (Vec::new(), Vec::new());
    problem.collect_hessian_cells(&mut cells, &mut ctx);
    problem.collect_param_block_spans(&mut spans, &mut ctx);
    let n = x.len();
    let partition = crate::store::block_partition_from_spans(&spans, n);
    let part = crate::simple_lm::partition_idx(&partition);
    let hsym = arael_faer::bsc::SymbolicSparseBlockColMat::covering(part.clone(), part, cells.len(), |k| (cells[k].0 as usize, cells[k].1 as usize));
    let mut resolver = arael_faer::bsc::PositionResolver::new(&hsym);
    let mut positions = PositionStream::new();
    problem.bind_hessian_positions(&mut HessianBinder::Tiled(&mut |i, j| resolver.resolve_tile(i as usize, j as usize)), &mut positions, &mut ctx);
    let nv = hsym.val_count();
    let (mut grad, mut vals) = (vec![T::zero(); n], vec![T::zero(); nv]);
    let cpu_cost = problem.calc_grad_hessian_sparse_indexed(x, &mut grad, &mut vals, &positions, &mut ctx).to_f64().unwrap();
    let root = problem.device_root().expect("the root has a device assembly");
    let mut resolver = arael_faer::bsc::PositionResolver::new(&hsym);
    let asm = NativeDeviceAssembly::<T, D>::new(dev, root, &mut |i, j| resolver.resolve_tile(i as usize, j as usize)).expect("lay the root out on the device");
    // x, the gradient, the values and the cost in one block of the test's own
    let (at_g, at_h) = (n * 8, n * 16);
    let at_cost = at_h + nv * 8;
    let mut work = dev.alloc(at_cost + 8).expect("allocate the targets");
    let x64: Vec<f64> = x.iter().map(|v| v.to_f64().unwrap()).collect();
    dev.write(&mut work, 0, f64_bytes(&x64)).expect("upload x");
    asm.assemble_at((&work, 0), (&work, at_g), n, (&work, at_h), nv, (&work, at_cost)).expect("assemble on the device");
    let (mut gd, mut hd, mut cost) = (vec![0f64; n], vec![0f64; nv], [0f64]);
    dev.read(&work, at_g, f64_bytes_mut(&mut gd)).expect("read the gradient");
    dev.read(&work, at_h, f64_bytes_mut(&mut hd)).expect("read the Hessian");
    dev.read(&work, at_cost, f64_bytes_mut(&mut cost)).expect("read the cost");
    let err = |dev: &[f64], cpu: &[T]| -> f64 {
        let max = cpu.iter().fold(0.0f64, |m, v| m.max(v.to_f64().unwrap().abs()));
        let diff = dev.iter().zip(cpu).fold(0.0f64, |m, (d, c)| m.max((d - c.to_f64().unwrap()).abs()));
        if max > 0.0 { diff / max } else { diff }
    };
    crate::device::DeviceCheck { cpu_cost, device_cost: cost[0], grad_err: err(&gd, &grad), hess_err: err(&hd, &vals), values: nv }
}

/// Instances per constraint, from the pack walk's records.
fn instance_counts<T>(shape: &DeviceShape, pack: &DevicePack<T>) -> Vec<usize> {
    shape.constraints.iter().zip(&pack.inst).map(|(&(w, _), inst)| inst.len() / w.max(1)).collect()
}

fn u32_bytes(v: &[u32]) -> &[u8] {
    arael_gpu::bytemuck::cast_slice(v)
}

fn f64_bytes(v: &[f64]) -> &[u8] {
    arael_gpu::bytemuck::cast_slice(v)
}

fn f64_bytes_mut(v: &mut [f64]) -> &mut [u8] {
    arael_gpu::bytemuck::cast_slice_mut(v)
}

/// The bytes of a model scalar slice (`f32` or `f64`).
fn scalar_bytes<T: crate::utils::Float>(v: &[T]) -> &[u8] {
    // SAFETY: the model's scalar is f32 or f64 (asserted in `new`), plain
    // values without padding.
    unsafe { core::slice::from_raw_parts(v.as_ptr() as *const u8, core::mem::size_of_val(v)) }
}
