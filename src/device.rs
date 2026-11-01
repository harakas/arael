//! The device assembly (feature `gpu`): the gradient, the Hessian's tile
//! values and the cost computed on a GPU from the model's constraints.
//!
//! A root the device head covers gets an implementation of [`DeviceRoot`]
//! from `#[arael(root)]`: its layout ([`DeviceShape`]), its block tables
//! against the solver's pattern, a pack walk that reads the constraint
//! instances and the entities' fixed values and reference rotations from
//! the model, and the launches of its generated constraint kernels.
//! [`DeviceAssembly`] lays a root out on a [`GpuContext`] once per
//! structure and then assembles at any parameter vector into device
//! buffers, which the device solve reads in place.

use arael_gpu::assembly::{k_euler_advance, k_euler_rotations, k_gather_params};
use arael_gpu::cubecl::bytes::Bytes;
use arael_gpu::cubecl::prelude::*;
use arael_gpu::cubecl::server::Handle;
use arael_gpu::host::Upload;
use arael_gpu::kernels::k_zero;
use arael_gpu::{GpuContext, Lease};

/// A root's device layout, as its generated code describes it.
#[derive(Clone, Debug)]
pub struct DeviceShape {
    /// Per entity type with a self block, in the store's order: the
    /// parameters of one entity.
    pub entities: Vec<usize>,
    /// Per Euler-angle rotation field a constraint reads: its entity
    /// array and the offset of its step among that entity's parameters.
    pub rotations: Vec<(usize, usize)>,
    /// How many cross arrays the store holds.
    pub crosses: usize,
    /// Per constraint: the words of one instance (its entities' slots and
    /// its cross slot) and its data leaves.
    pub constraints: Vec<(usize, usize)>,
}

/// A root's block store against a tiled pattern, slot by slot: per entity
/// array the global parameter indices and the tile `(base, stride,
/// start)`, per cross array the tile `(base, stride, flip)`.
#[derive(Clone, Debug, Default)]
pub struct DeviceTables {
    pub idx: Vec<Vec<u32>>,
    pub tile: Vec<Vec<u32>>,
    pub cross: Vec<Vec<u32>>,
}

/// What a root's pack walk reads from the model: per entity array the
/// values of its fixed parameters, per rotation field the reference
/// rotations (row-major, f64), per constraint its instances and their
/// data leaves.
#[derive(Clone, Debug)]
pub struct DevicePack<T> {
    pub fixed: Vec<Vec<T>>,
    pub refr: Vec<Vec<f64>>,
    pub inst: Vec<Vec<u32>>,
    pub data: Vec<Vec<T>>,
}

impl<T: Copy + Default> DevicePack<T> {
    /// Empty, for `shape` with `slots[e]` entities in entity array `e`:
    /// fixed values zero and reference rotations the identity until the
    /// walk writes them.
    pub fn new(shape: &DeviceShape, slots: &[usize]) -> Self {
        const IDENTITY: [f64; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        DevicePack {
            fixed: shape.entities.iter().zip(slots).map(|(&n, &s)| vec![T::default(); n * s]).collect(),
            refr: shape.rotations.iter().map(|&(e, _)| IDENTITY.repeat(slots[e])).collect(),
            inst: shape.constraints.iter().map(|_| Vec::new()).collect(),
            data: shape.constraints.iter().map(|_| Vec::new()).collect(),
        }
    }
}

/// The device buffers a root's generated launches read, by the indices of
/// its [`DeviceShape`].
pub struct DeviceBuffers {
    pub idx: Vec<Lease>,
    pub tile: Vec<Lease>,
    pub fixed: Vec<Lease>,
    /// Per entity array: its entities' parameter values at the last `x`.
    pub ev: Vec<Lease>,
    pub refr: Vec<Lease>,
    pub at: Vec<Lease>,
    /// Per rotation field: the composed rotations in the model's scalar
    /// and in f64, and their derivatives.
    pub rot: Vec<Lease>,
    pub rotw: Vec<Lease>,
    pub drot: Vec<Lease>,
    pub cross: Vec<Lease>,
    pub inst: Vec<Lease>,
    pub data: Vec<Lease>,
    /// Instances per constraint.
    pub count: Vec<u32>,
    /// Entities per entity array.
    pub slots: Vec<usize>,
}

/// The targets of one assembly: the gradient (`n` f64), the pattern's
/// tile values (f64) and the cost (one f64), all zeroed by it.
pub struct DeviceTargets<'a> {
    pub g: &'a Handle,
    pub ng: usize,
    pub h: &'a Handle,
    pub nh: usize,
    pub cost: &'a Handle,
}

/// What `#[arael(root)]` generates for a root the device assembly covers.
pub trait DeviceRoot<T> {
    /// The root's layout.
    fn device_shape(&self) -> DeviceShape;
    /// The root's block store, built for the model as it is, against the
    /// tiled pattern `resolve` describes (as `bind_hessian_positions`).
    fn device_tables(&self, resolve: &mut dyn FnMut(u32, u32) -> (usize, usize, usize)) -> DeviceTables;
    /// Read the instances and the entities' fixed values and reference
    /// rotations from the model, whose working copies are current.
    fn device_pack(&self, pack: &mut DevicePack<T>);
    /// Launch the constraint kernels: the assembly's into the targets, or
    /// with `assemble` false the cost's alone.
    fn device_launch(&self, client: &Client, bufs: &DeviceBuffers, targets: &DeviceTargets, assemble: bool);
}

/// A root laid out on a device: its tables, instances and entity data
/// uploaded once per structure, the parameter vector per call. The
/// model's scalar `T` is `f32` or `f64`, and the device code is
/// instantiated for whichever it is.
pub struct DeviceAssembly<T> {
    ctx: GpuContext,
    bufs: DeviceBuffers,
    shape: DeviceShape,
    /// The parameter vector, in f64 whatever `T`.
    x: Upload<f64>,
    cost: Lease,
    _t: core::marker::PhantomData<T>,
}

/// Lease a buffer for `v` and write it, handing CubeCL the vector (no wait).
fn upload<E: arael_gpu::bytemuck::Pod + CubeElement>(ctx: &GpuContext, v: Vec<E>) -> Lease {
    let lease = ctx.lease(v.len().max(1) * core::mem::size_of::<E>());
    if !v.is_empty() {
        ctx.client().write(lease.handle(), Bytes::from_elems(v));
    }
    lease
}

/// [`upload`] of model scalars, in the device scalar of the same width.
fn upload_scalars<T: crate::utils::Float>(ctx: &GpuContext, v: Vec<T>) -> Lease {
    if wide::<T>() {
        upload(ctx, v.into_iter().map(|x| x.to_f64().unwrap()).collect::<Vec<f64>>())
    } else {
        upload(ctx, v.into_iter().map(|x| x.to_f32().unwrap()).collect::<Vec<f32>>())
    }
}

/// Whether the model's scalar is the 8-byte one.
fn wide<T>() -> bool {
    core::mem::size_of::<T>() == 8
}

fn arr(h: &Handle, len: usize) -> BufferArg {
    unsafe { BufferArg::from_raw_parts(h.clone(), len.max(1)) }
}

/// A kernel argument over a lease's whole capacity, in elements of `E`.
pub fn lease_arg<E>(l: &Lease) -> BufferArg {
    arr(l.handle(), l.capacity() / core::mem::size_of::<E>())
}

/// A kernel argument over `len` elements of a buffer.
pub fn handle_arg(h: &Handle, len: usize) -> BufferArg {
    arr(h, len)
}

fn cubes(n: usize) -> CubeCount {
    CubeCount::Static((n as u32).div_ceil(256).max(1), 1, 1)
}

impl<T: crate::utils::Float + 'static> DeviceAssembly<T> {
    /// Lay `root` out on `ctx` against the tiled pattern `resolve`
    /// describes. The model's working copies must be current (its fixed
    /// values and reference rotations are read).
    pub fn new(ctx: &GpuContext, root: &dyn DeviceRoot<T>, resolve: &mut dyn FnMut(u32, u32) -> (usize, usize, usize)) -> Self {
        assert!(core::mem::size_of::<T>() == 4 || wide::<T>(), "the device assembly takes an f32 or an f64 model");
        let shape = root.device_shape();
        let tables = root.device_tables(resolve);
        let slots: Vec<usize> = shape.entities.iter().zip(&tables.idx).map(|(&n, ix)| ix.len() / n.max(1)).collect();
        let mut pack = DevicePack::<T>::new(&shape, &slots);
        root.device_pack(&mut pack);
        let count: Vec<u32> = shape.constraints.iter().zip(&pack.inst).map(|(&(w, _), inst)| (inst.len() / w.max(1)) as u32).collect();
        let ts = core::mem::size_of::<T>();
        let at: Vec<Lease> = shape.rotations.iter().map(|&(e, off)| upload(ctx, (0..slots[e]).map(|s| (s * shape.entities[e] + off) as u32).collect::<Vec<u32>>())).collect();
        let rot_lease = |per: usize, size: usize| -> Vec<Lease> { shape.rotations.iter().map(|&(e, _)| ctx.lease((slots[e] * per).max(1) * size)).collect() };
        let bufs = DeviceBuffers {
            idx: tables.idx.into_iter().map(|v| upload(ctx, v)).collect(),
            tile: tables.tile.into_iter().map(|v| upload(ctx, v)).collect(),
            fixed: pack.fixed.into_iter().map(|v| upload_scalars(ctx, v)).collect(),
            ev: shape.entities.iter().zip(&slots).map(|(&n, &s)| ctx.lease((n * s).max(1) * ts)).collect(),
            refr: pack.refr.into_iter().map(|v| upload(ctx, v)).collect(),
            at,
            rot: rot_lease(9, ts),
            rotw: rot_lease(9, 8),
            drot: rot_lease(27, ts),
            cross: tables.cross.into_iter().map(|v| upload(ctx, v)).collect(),
            inst: pack.inst.into_iter().map(|v| upload(ctx, v)).collect(),
            data: pack.data.into_iter().map(|v| upload_scalars(ctx, v)).collect(),
            count,
            slots,
        };
        DeviceAssembly { ctx: ctx.clone(), bufs, shape, x: Upload::default(), cost: ctx.lease(8), _t: core::marker::PhantomData }
    }

    /// The parameter values of every entity and the rotations at `x`.
    fn prepare(&mut self, x: &[T]) {
        let c = self.ctx.client();
        // SAFETY: the conversion to f64 lands in the upload's own buffer,
        // which the caller's read of the cost (every public entry makes
        // one before returning) waits past.
        unsafe { self.x.write(&self.ctx, x, |v| v.to_f64().unwrap()) };
        let d256 = CubeDim::new_1d(256);
        for (e, &n) in self.shape.entities.iter().enumerate() {
            let len = n * self.bufs.slots[e];
            if len == 0 {
                continue;
            }
            let b = &self.bufs;
            let args = (arr(self.x.handle(), x.len()), arr(b.idx[e].handle(), len), arr(b.fixed[e].handle(), len), arr(b.ev[e].handle(), len));
            unsafe {
                if wide::<T>() {
                    k_gather_params::launch_unchecked::<f64>(c, cubes(len), d256, args.0, args.1, args.2, args.3, len as u32)
                } else {
                    k_gather_params::launch_unchecked::<f32>(c, cubes(len), d256, args.0, args.1, args.2, args.3, len as u32)
                }
            };
        }
        for (r, &(e, _)) in self.shape.rotations.iter().enumerate() {
            let s = self.bufs.slots[e];
            if s == 0 {
                continue;
            }
            let b = &self.bufs;
            let n = self.shape.entities[e] * s;
            let args = (
                arr(b.refr[r].handle(), 9 * s),
                arr(b.ev[e].handle(), n),
                arr(b.at[r].handle(), s),
                arr(b.rotw[r].handle(), 9 * s),
                arr(b.rot[r].handle(), 9 * s),
                arr(b.drot[r].handle(), 27 * s),
            );
            unsafe {
                if wide::<T>() {
                    k_euler_rotations::launch_unchecked::<f64>(c, cubes(s), d256, args.0, args.1, args.2, args.3, args.4, args.5, s as u32)
                } else {
                    k_euler_rotations::launch_unchecked::<f32>(c, cubes(s), d256, args.0, args.1, args.2, args.3, args.4, args.5, s as u32)
                }
            };
        }
    }

    fn zero(&self, h: &Handle, n: usize) {
        unsafe { k_zero::launch_unchecked::<f64>(self.ctx.client(), cubes(n), CubeDim::new_1d(256), arr(h, n), n as u32) };
    }

    fn read_cost(&self) -> f64 {
        let bytes = arael_gpu::read_one(self.ctx.client(), self.cost.view(8)).expect("read the cost");
        f64::from_bytes(&bytes)[0]
    }

    /// The gradient and the pattern's tile values at `x`, into `g` (`ng`
    /// f64) and `h` (`nh` f64) on the device; returns the cost, the one
    /// value read back. Both targets are on the assembly's context.
    pub fn assemble(&mut self, root: &dyn DeviceRoot<T>, x: &[T], g: &Handle, ng: usize, h: &Handle, nh: usize) -> f64 {
        self.prepare(x);
        self.zero(g, ng);
        self.zero(h, nh);
        self.zero(self.cost.handle(), 1);
        let targets = DeviceTargets { g, ng, h, nh, cost: self.cost.handle() };
        root.device_launch(self.ctx.client(), &self.bufs, &targets, true);
        self.read_cost()
    }

    /// The cost at `x`.
    pub fn cost(&mut self, root: &dyn DeviceRoot<T>, x: &[T]) -> f64 {
        self.prepare(x);
        self.zero(self.cost.handle(), 1);
        let none = self.cost.handle();
        let targets = DeviceTargets { g: none, ng: 0, h: none, nh: 0, cost: self.cost.handle() };
        root.device_launch(self.ctx.client(), &self.bufs, &targets, false);
        self.read_cost()
    }

    /// The reference rotations moved by the steps of the last `x` (an
    /// accepted step's), as the model's `advance_params` moves its own.
    pub fn advance(&mut self) {
        let c = self.ctx.client();
        for (r, &(e, _)) in self.shape.rotations.iter().enumerate() {
            let s = self.bufs.slots[e];
            if s == 0 {
                continue;
            }
            let b = &self.bufs;
            let n = self.shape.entities[e] * s;
            let args = (arr(b.refr[r].handle(), 9 * s), arr(b.ev[e].handle(), n), arr(b.at[r].handle(), s));
            unsafe {
                if wide::<T>() {
                    k_euler_advance::launch_unchecked::<f64>(c, cubes(s), CubeDim::new_1d(256), args.0, args.1, args.2, s as u32)
                } else {
                    k_euler_advance::launch_unchecked::<f32>(c, cubes(s), CubeDim::new_1d(256), args.0, args.1, args.2, s as u32)
                }
            };
        }
    }

    /// The context the assembly runs on.
    pub fn context(&self) -> &GpuContext {
        &self.ctx
    }
}

/// What [`device_check`] found: the CPU's and the device's cost, and the
/// largest differences of the gradient and of the Hessian's values, each
/// relative to the largest CPU entry.
#[derive(Clone, Copy, Debug)]
pub struct DeviceCheck {
    pub cpu_cost: f64,
    pub device_cost: f64,
    pub grad_err: f64,
    pub hess_err: f64,
    /// The pattern's values.
    pub values: usize,
}

/// Assemble `problem` at `x` on the CPU and on the device against one
/// tiled pattern (the reduced route's), and compare (for tests).
pub fn device_check<T>(problem: &mut dyn crate::simple_lm::LmProblemInternals<T>, x: &[T], gpu: &GpuContext) -> DeviceCheck
where
    T: crate::utils::Float + 'static,
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
    let mut dev = DeviceAssembly::<T>::new(gpu, root, &mut |i, j| resolver.resolve_tile(i as usize, j as usize));
    let (g, h) = (gpu.lease(n * 8), gpu.lease(nv.max(1) * 8));
    let device_cost = dev.assemble(root, x, g.handle(), n, h.handle(), nv);
    let gd = f64::from_bytes(&arael_gpu::read_one(gpu.client(), g.view(n * 8)).expect("read the gradient")).to_vec();
    let hd = f64::from_bytes(&arael_gpu::read_one(gpu.client(), h.view(nv * 8)).expect("read the Hessian")).to_vec();
    let err = |dev: &[f64], cpu: &[T]| -> f64 {
        let max = cpu.iter().fold(0.0f64, |m, v| m.max(v.to_f64().unwrap().abs()));
        let diff = dev.iter().zip(cpu).fold(0.0f64, |m, (d, c)| m.max((d - c.to_f64().unwrap()).abs()));
        if max > 0.0 { diff / max } else { diff }
    };
    DeviceCheck { cpu_cost, device_cost, grad_err: err(&gd, &grad), hess_err: err(&hd, &vals), values: nv }
}
