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
use arael_gpu::lm::k_zero_steps;
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
    pub constraints: Vec<DeviceConstraint>,
}

/// One constraint of a [`DeviceShape`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceConstraint {
    /// The words of one instance: its entities' slots and its cross slot.
    pub words: usize,
    /// Its data leaves.
    pub data: usize,
    /// Its residual rows.
    pub rows: usize,
    /// The entity arrays of its two sides, and its cross array.
    pub a: usize,
    pub b: usize,
    pub cross: usize,
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

/// A root's constraint kernels launched: the assembly's into the targets,
/// or with `assemble` false the cost's alone. A plain function, so the
/// launches can be recorded into a graph on the device's thread.
pub type DeviceLauncher = fn(&Client, &DeviceBuffers, &DeviceTargets, bool);

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
    /// The launches of the constraint kernels ([`DeviceLauncher`]).
    fn device_launcher(&self) -> DeviceLauncher;
    /// The constraint kernels in C++ for arael's own GPU runtime: per
    /// constraint `c`, `__dk<c>_gh` and `__dk<c>_cost`, compiled after
    /// `arael-gpu/kernels/assembly.cpp` with the layout table's slots
    /// (`ARAEL_COL_*`) defined in front.
    fn device_cpp(&self) -> &'static str;
    /// Set the reference rotations of the model's optimized rotations from
    /// `refs`, keyed by a rotation's first parameter index (row-major), and
    /// zero their steps: a device solve's re-centring handed back.
    fn device_restore(&mut self, refs: &mut dyn FnMut(u32) -> Option<[f64; 9]>);
}

/// A root laid out on a device: its tables, instances and entity data
/// uploaded once per structure, the parameter vector per call. The
/// model's scalar `T` is `f32` or `f64`, and the device code is
/// instantiated for whichever it is.
pub struct DeviceAssembly<T> {
    ctx: GpuContext,
    bufs: DeviceBuffers,
    shape: DeviceShape,
    /// Per rotation field, per slot: the global index of the rotation's
    /// first parameter, [`FIXED`](arael_gpu::assembly::FIXED) when fixed.
    rot_index: Vec<Vec<u32>>,
    launch: DeviceLauncher,
    generation: u64,
    /// The parameter vector, in f64 whatever `T`.
    x: Upload<f64>,
    cost: Lease,
    _t: core::marker::PhantomData<fn() -> T>,
}

/// Lease a buffer for `v` and write it, handing CubeCL the vector (no wait).
fn upload<E: arael_gpu::bytemuck::Pod + CubeElement>(ctx: &GpuContext, v: Vec<E>) -> Lease {
    let lease = ctx.lease(v.len().max(1) * core::mem::size_of::<E>());
    if !v.is_empty() {
        ctx.client().write(lease.handle(), Bytes::from_elems(v));
    }
    lease
}

/// Write each of `vs` into its lease in `leases`: in place where the lease
/// holds it, so the device memory (which a captured graph records) stays
/// the same, into a new lease otherwise. Whether any lease is new.
fn rewrite<E: arael_gpu::bytemuck::Pod + CubeElement>(ctx: &GpuContext, leases: &mut Vec<Lease>, vs: Vec<Vec<E>>) -> bool {
    let mut moved = leases.len() != vs.len();
    if moved {
        leases.clear();
    }
    for (k, v) in vs.into_iter().enumerate() {
        if k < leases.len() && leases[k].capacity() >= v.len().max(1) * core::mem::size_of::<E>() {
            if !v.is_empty() {
                ctx.client().write(leases[k].handle(), Bytes::from_elems(v));
            }
        } else {
            let lease = upload(ctx, v);
            if k < leases.len() {
                leases[k] = lease;
            } else {
                leases.push(lease);
            }
            moved = true;
        }
    }
    moved
}

/// [`rewrite`] of model scalars, in the device scalar of the same width.
fn rewrite_scalars<T: crate::utils::Float>(ctx: &GpuContext, leases: &mut Vec<Lease>, vs: Vec<Vec<T>>) -> bool {
    if wide::<T>() {
        rewrite(ctx, leases, vs.into_iter().map(|v| v.into_iter().map(|x| x.to_f64().unwrap()).collect::<Vec<f64>>()).collect())
    } else {
        rewrite(ctx, leases, vs.into_iter().map(|v| v.into_iter().map(|x| x.to_f32().unwrap()).collect::<Vec<f32>>()).collect())
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
        let ts = core::mem::size_of::<T>();
        let at: Vec<Lease> = shape.rotations.iter().map(|&(e, off)| upload(ctx, (0..slots[e]).map(|s| (s * shape.entities[e] + off) as u32).collect::<Vec<u32>>())).collect();
        let rot_index = shape.rotations.iter().map(|&(e, off)| (0..slots[e]).map(|s| tables.idx[e][s * shape.entities[e] + off]).collect()).collect();
        let rot_lease = |per: usize, size: usize| -> Vec<Lease> { shape.rotations.iter().map(|&(e, _)| ctx.lease((slots[e] * per).max(1) * size)).collect() };
        let bufs = DeviceBuffers {
            idx: tables.idx.into_iter().map(|v| upload(ctx, v)).collect(),
            tile: tables.tile.into_iter().map(|v| upload(ctx, v)).collect(),
            fixed: Vec::new(),
            ev: shape.entities.iter().zip(&slots).map(|(&n, &s)| ctx.lease((n * s).max(1) * ts)).collect(),
            refr: Vec::new(),
            at,
            rot: rot_lease(9, ts),
            rotw: rot_lease(9, 8),
            drot: rot_lease(27, ts),
            cross: tables.cross.into_iter().map(|v| upload(ctx, v)).collect(),
            inst: Vec::new(),
            data: Vec::new(),
            count: Vec::new(),
            slots,
        };
        let launch = root.device_launcher();
        let mut asm = DeviceAssembly { ctx: ctx.clone(), bufs, shape, rot_index, launch, generation: 0, x: Upload::default(), cost: ctx.lease(8), _t: core::marker::PhantomData };
        asm.refresh(root);
        asm
    }

    /// Read the model's data again: the constraint instances and their
    /// data, the entities' fixed values and the reference rotations, which
    /// a solve may have been handed changed (the structure is the same).
    /// The data lands in the buffers it was in, so graphs recorded against
    /// them stay valid; where it does not fit, the
    /// [`generation`](Self::generation) moves on. The model's working
    /// copies must be current.
    pub fn refresh(&mut self, root: &dyn DeviceRoot<T>) {
        let ctx = &self.ctx;
        let mut pack = DevicePack::<T>::new(&self.shape, &self.bufs.slots);
        root.device_pack(&mut pack);
        let b = &mut self.bufs;
        b.count = self.shape.constraints.iter().zip(&pack.inst).map(|(c, inst)| (inst.len() / c.words.max(1)) as u32).collect();
        let mut moved = rewrite_scalars(ctx, &mut b.fixed, pack.fixed);
        moved |= rewrite(ctx, &mut b.refr, pack.refr);
        moved |= rewrite(ctx, &mut b.inst, pack.inst);
        moved |= rewrite_scalars(ctx, &mut b.data, pack.data);
        if moved {
            self.generation += 1;
        }
    }

    /// Changes whenever a buffer the launches read is replaced by another:
    /// a graph of them replays only under the generation it was captured in.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// The parameter values of every entity and the rotations at `x`.
    fn prepare(&mut self, x: &[T]) {
        // SAFETY: the conversion to f64 lands in the upload's own buffer,
        // which the caller's read of the cost (every public entry makes
        // one before returning) waits past.
        unsafe { self.x.write(&self.ctx, x, |v| v.to_f64().unwrap()) };
        let xh = self.x.handle().clone();
        self.gather(&xh, x.len());
    }

    /// [`prepare`](Self::prepare) from `n` f64 parameters on the device.
    fn gather(&self, x: &Handle, nx: usize) {
        let c = self.ctx.client();
        let d256 = CubeDim::new_1d(256);
        for (e, &n) in self.shape.entities.iter().enumerate() {
            let len = n * self.bufs.slots[e];
            if len == 0 {
                continue;
            }
            let b = &self.bufs;
            let args = (arr(x, nx), arr(b.idx[e].handle(), len), arr(b.fixed[e].handle(), len), arr(b.ev[e].handle(), len));
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
    pub fn assemble(&mut self, x: &[T], g: &Handle, ng: usize, h: &Handle, nh: usize) -> f64 {
        self.prepare(x);
        let cost = self.cost.handle().clone();
        self.launch(&DeviceTargets { g, ng, h, nh, cost: &cost }, true);
        self.read_cost()
    }

    /// [`assemble`](Self::assemble) at the `nx` f64 parameters `x` on the
    /// device, the cost into `targets.cost`, nothing read back.
    pub fn assemble_at(&self, x: &Handle, nx: usize, targets: &DeviceTargets) {
        self.gather(x, nx);
        self.launch(targets, true);
    }

    /// The cost at the `nx` f64 parameters `x` on the device, into `cost`
    /// (one f64), nothing read back.
    pub fn cost_at(&self, x: &Handle, nx: usize, cost: &Handle) {
        self.gather(x, nx);
        self.launch(&DeviceTargets { g: cost, ng: 0, h: cost, nh: 0, cost }, false);
    }

    /// The constraint kernels into `targets`, zeroed first.
    fn launch(&self, targets: &DeviceTargets, assemble: bool) {
        if assemble {
            self.zero(targets.g, targets.ng);
            self.zero(targets.h, targets.nh);
        }
        self.zero(targets.cost, 1);
        (self.launch)(self.ctx.client(), &self.bufs, targets, assemble);
    }

    /// The rotation steps of the `nx` f64 parameters `x` on the device
    /// zeroed, as re-centring after [`advance`](Self::advance) leaves them.
    pub fn zero_steps(&self, x: &Handle, nx: usize) {
        let c = self.ctx.client();
        for (r, &(e, _)) in self.shape.rotations.iter().enumerate() {
            let s = self.bufs.slots[e];
            if s == 0 {
                continue;
            }
            let b = &self.bufs;
            let n = self.shape.entities[e] * s;
            unsafe { k_zero_steps::launch_unchecked(c, cubes(s), CubeDim::new_1d(256), arr(x, nx), arr(b.idx[e].handle(), n), arr(b.at[r].handle(), s), s as u32) };
        }
    }

    /// Write the device's reference rotations into the model's optimized
    /// rotations ([`DeviceRoot::device_restore`]), one read.
    pub fn restore(&self, root: &mut dyn DeviceRoot<T>) {
        let views: Vec<Handle> = self.shape.rotations.iter().zip(&self.bufs.refr).map(|(&(e, _), l)| l.view(9 * self.bufs.slots[e].max(1) * 8)).collect();
        let back = arael_gpu::read(self.ctx.client(), views).expect("read the reference rotations");
        let refr: Vec<&[f64]> = back.iter().map(|b| f64::from_bytes(b)).collect();
        let mut at = std::collections::HashMap::new();
        for (r, ix) in self.rot_index.iter().enumerate() {
            for (s, &g) in ix.iter().enumerate() {
                if g != arael_gpu::assembly::FIXED {
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
    }

    /// The reference rotations moved by the steps of the last `x` (an
    /// accepted step's), as the model's `advance_params` moves its own.
    pub fn advance(&self) {
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
    let device_cost = dev.assemble(x, g.handle(), n, h.handle(), nv);
    let gd = f64::from_bytes(&arael_gpu::read_one(gpu.client(), g.view(n * 8)).expect("read the gradient")).to_vec();
    let hd = f64::from_bytes(&arael_gpu::read_one(gpu.client(), h.view(nv * 8)).expect("read the Hessian")).to_vec();
    let err = |dev: &[f64], cpu: &[T]| -> f64 {
        let max = cpu.iter().fold(0.0f64, |m, v| m.max(v.to_f64().unwrap().abs()));
        let diff = dev.iter().zip(cpu).fold(0.0f64, |m, (d, c)| m.max((d - c.to_f64().unwrap()).abs()));
        if max > 0.0 { diff / max } else { diff }
    };
    DeviceCheck { cpu_cost, device_cost, grad_err: err(&gd, &grad), hess_err: err(&hd, &vals), values: nv }
}
