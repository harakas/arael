//! The device assembly (feature `gpu`): the gradient, the Hessian's tile
//! values and the cost computed on a GPU from the model's constraints.
//!
//! A root the device head covers gets an implementation of [`DeviceRoot`]
//! from `#[arael(root)]`: its layout ([`DeviceShape`]), its block tables
//! against the solver's pattern, a pack walk that reads the constraint
//! instances and the entities' fixed values and references from
//! the model, and its constraint kernels in C++ ([`DeviceRoot::device_cpp`]).
//! [`DeviceAssembly`] lays a root out on a [`Device`] once per structure and
//! then assembles at any parameter vector into device blocks, which the
//! device solve reads in place: the model's data as columns in one block,
//! the sums in the route's [`Wide`] scalar and no atomic writes
//! (`arael-gpu/kernels/assembly.cpp`).
//!
//! The block starts with a layout table: the byte offset of every column,
//! one `u64` each, in the order [`columns`] gives them. The kernels read a
//! column through its slot of the table (`ARAEL_COL_<name>`), so a root's
//! kernels compile once whatever the problem's size, and only the offsets
//! change with it.

use std::collections::HashMap;
use std::sync::Arc;

use arael_gpu::{Arg, Device, FIXED, GpuError, Launch, WIDE, Wide};

/// A root's device layout, as its generated code describes it.
#[derive(Clone, Debug)]
pub struct DeviceShape {
    /// Per entity type with a self block, in the store's order: the
    /// parameters of one entity.
    pub entities: Vec<usize>,
    /// Per referenced parameter field a constraint reads.
    pub references: Vec<DeviceReference>,
    /// How many cross arrays the store holds.
    pub crosses: usize,
    pub constraints: Vec<DeviceConstraint>,
}

/// One referenced parameter field of a [`DeviceShape`]: a parameter whose
/// step is measured from a reference that moves to it after every accepted
/// step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceReference {
    /// Its entity array.
    pub entity: usize,
    /// The offset of its step among the entity's parameters.
    pub step: usize,
    pub kind: ReferenceKind,
}

/// The kinds of referenced parameters, each with kernels that compose its
/// reference with its step and move the reference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReferenceKind {
    /// `EulerAngleParam`: a rotation matrix (row-major) and Euler angles.
    Euler,
    /// `QuaternionParam`: a unit quaternion (w, x, y, z) and a rotation
    /// vector.
    Quaternion,
    /// `TransformParam`: a unit quaternion and a translation, and a twist
    /// (rotation vector, then translation rate).
    Transform,
    /// `UnitVecParam`: a unit quaternion and a 2D tangent step.
    UnitVec,
}

impl ReferenceKind {
    /// The values of one reference.
    pub fn reference_len(self) -> usize {
        match self {
            ReferenceKind::Euler => 9,
            ReferenceKind::Quaternion | ReferenceKind::UnitVec => 4,
            ReferenceKind::Transform => 7,
        }
    }

    /// The parameters of one step.
    pub fn step_len(self) -> usize {
        match self {
            ReferenceKind::Euler | ReferenceKind::Quaternion => 3,
            ReferenceKind::Transform => 6,
            ReferenceKind::UnitVec => 2,
        }
    }

    /// The values a constraint reads of one entity: a rotation matrix (9),
    /// a rotation matrix and a translation (12), a direction (3).
    pub fn values(self) -> usize {
        match self {
            ReferenceKind::Euler | ReferenceKind::Quaternion => 9,
            ReferenceKind::Transform => 12,
            ReferenceKind::UnitVec => 3,
        }
    }

    /// The derivatives of the values by the step a linearization reads.
    pub fn derivatives(self) -> usize {
        match self {
            ReferenceKind::Euler | ReferenceKind::Quaternion => 27,
            ReferenceKind::Transform => 45,
            ReferenceKind::UnitVec => 6,
        }
    }

    /// The identity reference.
    fn identity(self) -> &'static [f64] {
        match self {
            ReferenceKind::Euler => &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            ReferenceKind::Quaternion | ReferenceKind::UnitVec => &[1.0, 0.0, 0.0, 0.0],
            ReferenceKind::Transform => &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    }

    /// Its kernels' name in `arael-gpu/kernels/assembly.cpp`
    /// (`k_<name>_values`, `_derivatives`, `_advance`).
    fn kernel_name(self) -> &'static str {
        match self {
            ReferenceKind::Euler => "euler",
            ReferenceKind::Quaternion => "quat",
            ReferenceKind::Transform => "transform",
            ReferenceKind::UnitVec => "unitvec",
        }
    }

    /// The reference as the model takes it back.
    fn reference(self, v: &[f64]) -> crate::model::Reference {
        use crate::model::Reference;
        match self {
            ReferenceKind::Euler => Reference::Matrix(v.try_into().unwrap()),
            ReferenceKind::Quaternion | ReferenceKind::UnitVec => Reference::Quaternion(v.try_into().unwrap()),
            ReferenceKind::Transform => Reference::Transform(v.try_into().unwrap()),
        }
    }

    const ALL: [ReferenceKind; 4] = [ReferenceKind::Euler, ReferenceKind::Quaternion, ReferenceKind::Transform, ReferenceKind::UnitVec];
}

/// One constraint of a [`DeviceShape`]: of one entity, which it writes into
/// its own tile and gradient, or of two and their cross tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceConstraint {
    /// The words of one instance: its entities' slots and its cross slot
    /// (3), or its one entity's slot (1).
    pub words: usize,
    /// Its data leaves.
    pub data: usize,
    /// Its residual rows.
    pub rows: usize,
    /// The entity array of its first side.
    pub a: usize,
    /// Of a constraint of two entities: the entity array of its second
    /// side and its cross array.
    pub pair: Option<(usize, usize)>,
}

impl DeviceConstraint {
    /// The parameters of its entities, the first side's then the second's.
    fn params(&self, shape: &DeviceShape) -> usize {
        shape.entities[self.a] + self.pair.map_or(0, |(b, _)| shape.entities[b])
    }
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

/// A model's leaf value (`f32`, `f64` or `df32`) as `f64`, for the pack
/// walk the device head generates: a mixed parameter keeps its step and
/// its reference rotation in the root's wide scalar.
pub fn leaf_f64<V: num::ToPrimitive>(v: V) -> f64 {
    v.to_f64().expect("a model's scalars convert to f64")
}

/// What a root's pack walk reads from the model: per entity array the
/// values of its fixed parameters, per referenced field the references
/// ([`ReferenceKind::reference_len`] values each, f64) and which slots the
/// walk wrote them for, per constraint its instances and their data leaves.
#[derive(Clone, Debug)]
pub struct DevicePack<T> {
    pub fixed: Vec<Vec<T>>,
    pub refr: Vec<Vec<f64>>,
    /// The slots of a referenced field an instance packed reads, whose
    /// references are the model's; the others keep the identity, and the
    /// device leaves the model's alone.
    pub packed: Vec<Vec<bool>>,
    pub inst: Vec<Vec<u32>>,
    pub data: Vec<Vec<T>>,
}

impl<T: Copy + Default> DevicePack<T> {
    /// Empty, for `shape` with `slots[e]` entities in entity array `e`:
    /// fixed values zero and references the identity until the walk
    /// writes them.
    pub fn new(shape: &DeviceShape, slots: &[usize]) -> Self {
        DevicePack {
            fixed: shape.entities.iter().zip(slots).map(|(&n, &s)| vec![T::default(); n * s]).collect(),
            refr: shape.references.iter().map(|r| r.kind.identity().repeat(slots[r.entity])).collect(),
            packed: shape.references.iter().map(|r| vec![false; slots[r.entity]]).collect(),
            inst: shape.constraints.iter().map(|_| Vec::new()).collect(),
            data: shape.constraints.iter().map(|_| Vec::new()).collect(),
        }
    }
}

/// What `#[arael(root)]` generates for a root the device assembly covers.
pub trait DeviceRoot<T> {
    /// The root's layout.
    fn device_shape(&self) -> DeviceShape;
    /// The root's block store, built for the model as it is, against the
    /// tiled pattern `resolve` describes (as `bind_hessian_positions`).
    fn device_tables(&self, resolve: &mut dyn FnMut(u32, u32) -> (usize, usize, usize)) -> DeviceTables;
    /// Read the instances and the entities' fixed values and reference
    /// references from the model, whose working copies are current.
    fn device_pack(&self, pack: &mut DevicePack<T>);
    /// The constraint kernels in C++: per constraint `c`, `__dk<c>_gh` and `__dk<c>_cost`, compiled after
    /// `arael-gpu/kernels/assembly.cpp` with the layout table's slots
    /// (`ARAEL_COL_*`) defined in front.
    fn device_cpp(&self) -> &'static str;
    /// Set the references of the model's optimized referenced parameters
    /// from `refs`, keyed by the first live index of the step, and zero
    /// their steps: a device solve's re-centring handed back.
    fn device_restore(&mut self, refs: &mut dyn FnMut(u32) -> Option<crate::model::Reference>);
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

/// Blocks of [`THREADS`] over `n` instances: the cost's parts.
fn parts(n: usize) -> usize {
    n.div_ceil(THREADS as usize).max(1)
}

/// The values of one instance's record: per row the residual and the two
/// entities' derivatives.
fn record_width(shape: &DeviceShape, c: usize) -> usize {
    let k = shape.constraints[c];
    k.rows * (1 + k.params(shape))
}

/// The columns of a root, in the layout table's order, and how many of
/// them the host writes: first those, per entity array its global indices,
/// tiles and fixed values; per referenced field its step offsets and
/// references; per cross array its tiles; per constraint its
/// slots, data leaves and its instances by entity and by cross slot. Then
/// the columns the kernels write: per entity array its gathered values, per
/// referenced field its values (in F and W) and derivatives, per constraint its records and
/// cost parts. The order depends on the shape alone. W values are 8 bytes.
fn columns(shape: &DeviceShape, slots: &[usize], cross_slots: &[usize], counts: &[usize], ts: usize) -> (Vec<Column>, usize) {
    let col = |name: String, elem: usize, len: usize| Column { name, elem, len };
    let mut v = Vec::new();
    for (e, &n) in shape.entities.iter().enumerate() {
        let s = slots[e];
        v.push(col(format!("E{e}_IDX"), 4, n * s));
        v.push(col(format!("E{e}_TILE"), 4, 3 * s));
        v.push(col(format!("E{e}_FIXED"), ts, n * s));
    }
    for (r, rf) in shape.references.iter().enumerate() {
        let s = slots[rf.entity];
        v.push(col(format!("R{r}_AT"), 4, s));
        v.push(col(format!("R{r}_REFR"), 8, rf.kind.reference_len() * s));
    }
    for (x, &s) in cross_slots.iter().enumerate() {
        v.push(col(format!("X{x}_TILE"), 4, 3 * s));
    }
    for (c, k) in shape.constraints.iter().enumerate() {
        let n = counts[c];
        let words: &[&str] = if k.pair.is_some() { &["SA", "SB", "SX"] } else { &["SA"] };
        for w in words {
            v.push(col(format!("C{c}_{w}"), 4, n));
        }
        for j in 0..k.data {
            v.push(col(format!("C{c}_D{j}"), ts, n));
        }
        v.push(col(format!("C{c}_AOFF"), 4, slots[k.a] + 1));
        v.push(col(format!("C{c}_AINST"), 4, n));
        if let Some((b, x)) = k.pair {
            v.push(col(format!("C{c}_BOFF"), 4, slots[b] + 1));
            v.push(col(format!("C{c}_BINST"), 4, n));
            v.push(col(format!("C{c}_XOFF"), 4, cross_slots[x] + 1));
            v.push(col(format!("C{c}_XINST"), 4, n));
        }
    }
    let host = v.len();
    for (e, &n) in shape.entities.iter().enumerate() {
        v.push(col(format!("E{e}_EV"), ts, n * slots[e]));
    }
    for (r, rf) in shape.references.iter().enumerate() {
        let s = slots[rf.entity];
        v.push(col(format!("R{r}_VAL"), ts, rf.kind.values() * s));
        v.push(col(format!("R{r}_VALW"), 8, rf.kind.values() * s));
        v.push(col(format!("R{r}_DER"), ts, rf.kind.derivatives() * s));
    }
    for (c, &n) in counts.iter().enumerate() {
        v.push(col(format!("C{c}_REC"), ts, n * record_width(shape, c)));
        v.push(col(format!("C{c}_PART"), 8, parts(n)));
    }
    (v, host)
}

/// Where every column sits in the block: the table first, then each column
/// on a 256-byte boundary; the host's columns end at `host_bytes`.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Layout {
    columns: Vec<Column>,
    offsets: Vec<usize>,
    host_bytes: usize,
    bytes: usize,
}

impl Layout {
    fn new((columns, host): (Vec<Column>, usize)) -> Self {
        let align = |b: usize| b.div_ceil(256) * 256;
        let mut at = align(columns.len() * 8);
        let mut offsets = Vec::with_capacity(columns.len());
        let mut host_bytes = at;
        for (k, c) in columns.iter().enumerate() {
            offsets.push(at);
            at += align((c.elem * c.len).max(1));
            if k + 1 == host {
                host_bytes = at;
            }
        }
        Layout { columns, offsets, host_bytes, bytes: at }
    }

    /// The `#define`s of the table's slots, for the kernels.
    fn defines(&self) -> String {
        self.columns.iter().enumerate().map(|(k, c)| format!("#define ARAEL_COL_{} {k}\n", c.name)).collect()
    }

    fn index(&self) -> HashMap<String, usize> {
        self.columns.iter().zip(&self.offsets).map(|(c, &o)| (c.name.clone(), o)).collect()
    }
}

/// A constraint's kernels: the instance kernel (records and cost parts),
/// the sums of its records per entity side and (of two entities) per cross
/// tile, the sum of its cost parts, and the cost alone.
struct ConstraintKernels<K> {
    gh: K,
    ra: K,
    /// The second side's and the cross tiles' sums.
    pair: Option<(K, K)>,
    csum: K,
    cost: K,
}

/// The kernels of one [`ReferenceKind`]: the values, the values with
/// their derivatives, and the references moved by their steps.
struct ReferenceKernels<K> {
    values: K,
    derivatives: K,
    advance: K,
}

/// The kernels of a root, compiled once per device, wide scalar and layout
/// of slots.
struct Kernels<K> {
    gather: K,
    /// Per [`ReferenceKind`], in [`ReferenceKind::ALL`]'s order.
    references: Vec<ReferenceKernels<K>>,
    zero: K,
    zero_steps: K,
    constraints: Vec<ConstraintKernels<K>>,
}

impl<K> Kernels<K> {
    fn reference(&self, kind: ReferenceKind) -> &ReferenceKernels<K> {
        &self.references[ReferenceKind::ALL.iter().position(|&k| k == kind).expect("every kind has kernels")]
    }
}

/// A buffer the assembly writes or reads that is not its own: a block and a
/// byte offset into it.
pub type Target<'a, B> = (&'a B, usize);

/// A root laid out on a [`Device`] in one block, its tables, instances and
/// entity data uploaded once per structure and refreshed per solve; the
/// parameter vector, the targets and the cost are the caller's buffers, in
/// the assembly's [`Wide`] scalar.
pub struct DeviceAssembly<T, D: Device> {
    dev: Arc<D>,
    wide: Wide,
    shape: DeviceShape,
    slots: Vec<usize>,
    cross_slots: Vec<usize>,
    counts: Vec<usize>,
    /// The structure's tables, kept for the block image of each refresh.
    idx: Vec<Vec<u32>>,
    tile: Vec<Vec<u32>>,
    cross: Vec<Vec<u32>>,
    at: Vec<Vec<u32>>,
    /// Per referenced field, per slot: the global index of the step's first
    /// live parameter, [`FIXED`] when the step is fixed.
    ref_index: Vec<Vec<u32>>,
    /// The last pack's [`DevicePack::packed`].
    packed: Vec<Vec<bool>>,
    /// Per cross array, per slot: the first slot on the same tile of H
    /// (two instances of one pair of entities, each with a cross block of
    /// its own, share a tile).
    tile_owner: Vec<Vec<u32>>,
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

/// Entry `j` of every record of `nd` values in `src` into `dst`, as `E`
/// (the model scalar's bits).
fn transpose<E: arael_gpu::bytemuck::Pod>(dst: &mut [u8], src: &[u8], j: usize, nd: usize) {
    let dst: &mut [E] = arael_gpu::bytemuck::cast_slice_mut(dst);
    let src: &[E] = arael_gpu::bytemuck::cast_slice(src);
    for (d, rec) in dst.iter_mut().zip(src.chunks_exact(nd)) {
        *d = rec[j];
    }
}

/// Per cross slot of the tables `tiles` (`(base, stride, flip)` each), the
/// first slot whose tile starts at the same value of H.
fn tile_owners(tiles: &[u32]) -> Vec<u32> {
    let n = tiles.len() / 3;
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_unstable_by_key(|&s| (tiles[3 * s as usize], s));
    let mut owner = vec![0u32; n];
    let mut first = 0u32;
    for (k, &s) in order.iter().enumerate() {
        if k == 0 || tiles[3 * s as usize] != tiles[3 * order[k - 1] as usize] {
            first = s;
        }
        owner[s as usize] = first;
    }
    owner
}

/// The instances, numbered in order, grouped by the slot each refers to:
/// `(ptr, inst)` with slot `s`'s instances at `inst[ptr[s]..ptr[s + 1]]`,
/// ascending.
fn by_slot(of: impl Iterator<Item = u32> + Clone, slots: usize) -> (Vec<u32>, Vec<u32>) {
    let mut ptr = vec![0u32; slots + 1];
    for s in of.clone() {
        ptr[s as usize + 1] += 1;
    }
    for s in 0..slots {
        ptr[s + 1] += ptr[s];
    }
    let mut fill = ptr.clone();
    let mut inst = vec![0u32; ptr[slots] as usize];
    for (o, s) in of.enumerate() {
        inst[fill[s as usize] as usize] = o as u32;
        fill[s as usize] += 1;
    }
    (ptr, inst)
}

impl<T: crate::utils::Float + 'static, D: Device> DeviceAssembly<T, D> {
    /// Lay `root` out on `dev` against the tiled pattern `resolve`
    /// describes, its sums in `wide`, and compile its kernels. The model's
    /// working copies must be current (its fixed values and references are
    /// read).
    pub fn new(
        dev: &Arc<D>,
        root: &dyn DeviceRoot<T>,
        resolve: &mut dyn FnMut(u32, u32) -> (usize, usize, usize),
        wide: Wide,
    ) -> Result<Self, GpuError> {
        let ts = core::mem::size_of::<T>();
        assert!(ts == 4 || ts == 8, "the device assembly takes an f32 or an f64 model");
        let shape = root.device_shape();
        let tables = root.device_tables(resolve);
        let slots: Vec<usize> = shape.entities.iter().zip(&tables.idx).map(|(&n, ix)| ix.len() / n.max(1)).collect();
        let cross_slots: Vec<usize> = tables.cross.iter().map(|c| c.len() / 3).collect();
        let at: Vec<Vec<u32>> = shape
            .references
            .iter()
            .map(|r| (0..slots[r.entity]).map(|s| (s * shape.entities[r.entity] + r.step) as u32).collect())
            .collect();
        // a step's key: its first live parameter (a TransformParam may hold
        // one half of its step fixed)
        let ref_index = shape
            .references
            .iter()
            .map(|r| {
                (0..slots[r.entity])
                    .map(|s| {
                        let at = s * shape.entities[r.entity] + r.step;
                        tables.idx[r.entity][at..at + r.kind.step_len()].iter().copied().find(|&g| g != FIXED).unwrap_or(FIXED)
                    })
                    .collect()
            })
            .collect();
        let tile_owner = tables.cross.iter().map(|t| tile_owners(t)).collect();
        let mut pack = DevicePack::<T>::new(&shape, &slots);
        root.device_pack(&mut pack);
        let counts = instance_counts(&shape, &pack);
        let layout = Layout::new(columns(&shape, &slots, &cross_slots, &counts, ts));
        let scalar = if ts == 8 { "double" } else { "float" };
        let defines = format!("{}#define SCALAR {scalar}\n{}", wide.defines(), layout.defines());
        let source = format!("{WIDE}\n{}\n{}", arael_gpu::ASSEMBLY, root.device_cpp());
        // every entry point in one lookup of the source, in the order the
        // kernels take them below
        let mut entries: Vec<String> = ["k_gather_params", "k_zero_wide", "k_zero_steps"].map(String::from).to_vec();
        for kind in ReferenceKind::ALL {
            entries.extend(["values", "derivatives", "advance"].map(|e| format!("k_{}_{e}", kind.kernel_name())));
        }
        for (c, k) in shape.constraints.iter().enumerate() {
            entries.extend(["gh", "ra", "csum", "cost"].map(|e| format!("__dk{c}_{e}")));
            if k.pair.is_some() {
                entries.extend(["rb", "rx"].map(|e| format!("__dk{c}_{e}")));
            }
        }
        let entries: Vec<&str> = entries.iter().map(String::as_str).collect();
        let mut compiled = dev.compile_many(&source, &defines, &entries)?.into_iter();
        let mut k = || compiled.next().expect("a kernel per entry point");
        let kernels = Kernels {
            gather: k(),
            zero: k(),
            zero_steps: k(),
            references: ReferenceKind::ALL.iter().map(|_| ReferenceKernels { values: k(), derivatives: k(), advance: k() }).collect(),
            constraints: shape
                .constraints
                .iter()
                .map(|c| ConstraintKernels { gh: k(), ra: k(), csum: k(), cost: k(), pair: c.pair.map(|_| (k(), k())) })
                .collect(),
        };
        let block = dev.alloc(layout.bytes)?;
        let offset = layout.index();
        let mut asm = DeviceAssembly {
            dev: dev.clone(),
            wide,
            shape,
            slots,
            cross_slots,
            counts,
            idx: tables.idx,
            tile: tables.tile,
            cross: tables.cross,
            at,
            ref_index,
            packed: Vec::new(),
            tile_owner,
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
    /// data, the entities' fixed values and the references, which
    /// a solve may have been handed changed (the structure is the same).
    /// The block keeps its place while the layout is the same, so graphs
    /// recorded against it stay valid; otherwise a new block is laid out
    /// and the [`generation`](Self::generation) moves on. The model's
    /// working copies must be current.
    pub fn refresh(&mut self, root: &dyn DeviceRoot<T>) -> Result<(), GpuError> {
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

    /// The host's part of the block, written in one copy: the table, the
    /// structure's tables, the model's data and the instance lists; the
    /// columns the kernels write after it are not.
    fn write(&mut self, pack: &DevicePack<T>) -> Result<(), GpuError> {
        use arael_gpu::bytemuck::{cast_slice, cast_slice_mut};
        self.packed.clone_from(&pack.packed);
        // in words, so every column (on a 256-byte boundary) is a typed slice
        let mut words = vec![0u64; self.layout.host_bytes / 8];
        for (k, &o) in self.layout.offsets.iter().enumerate() {
            words[k] = o as u64;
        }
        let image: &mut [u8] = cast_slice_mut(&mut words);
        let offset = &self.offset;
        // the byte range of `len` bytes of column `name`
        let col = |name: &str, len: usize| -> std::ops::Range<usize> {
            let o = offset[name];
            o..o + len
        };
        let put = |image: &mut [u8], name: &str, bytes: &[u8]| image[col(name, bytes.len())].copy_from_slice(bytes);
        for e in 0..self.shape.entities.len() {
            put(image, &format!("E{e}_IDX"), u32_bytes(&self.idx[e]));
            put(image, &format!("E{e}_TILE"), u32_bytes(&self.tile[e]));
            put(image, &format!("E{e}_FIXED"), scalar_bytes(&pack.fixed[e]));
        }
        for r in 0..self.shape.references.len() {
            put(image, &format!("R{r}_AT"), u32_bytes(&self.at[r]));
            put(image, &format!("R{r}_REFR"), &self.wide.encode(&pack.refr[r]));
        }
        for x in 0..self.cross.len() {
            put(image, &format!("X{x}_TILE"), u32_bytes(&self.cross[x]));
        }
        // the walk's instances and data, one record per instance, into
        // columns; the instances grouped by entity and by cross slot
        let ts = core::mem::size_of::<T>();
        for (c, k) in self.shape.constraints.iter().enumerate() {
            let n = self.counts[c];
            let words: &[&str] = if k.pair.is_some() { &["SA", "SB", "SX"] } else { &["SA"] };
            assert_eq!(k.words, words.len(), "an instance is two entity slots and a cross slot, or one entity slot");
            let inst = &pack.inst[c];
            for (w, name) in words.iter().enumerate() {
                let r = col(&format!("C{c}_{name}"), 4 * n);
                let dst: &mut [u32] = cast_slice_mut(&mut image[r]);
                for (d, rec) in dst.iter_mut().zip(inst.chunks_exact(k.words)) {
                    *d = rec[w];
                }
            }
            let data = scalar_bytes(&pack.data[c]);
            for j in 0..k.data {
                let r = col(&format!("C{c}_D{j}"), ts * n);
                if ts == 8 {
                    transpose::<u64>(&mut image[r], data, j, k.data);
                } else {
                    transpose::<u32>(&mut image[r], data, j, k.data);
                }
            }
            let mut sides = vec![("A", 0, self.slots[k.a])];
            if let Some((b, _)) = k.pair {
                sides.push(("B", 1, self.slots[b]));
            }
            for (side, w, slots) in sides {
                let (ptr, list) = by_slot(inst.chunks_exact(k.words).map(|rec| rec[w]), slots);
                put(image, &format!("C{c}_{side}OFF"), cast_slice(&ptr));
                put(image, &format!("C{c}_{side}INST"), cast_slice(&list));
            }
            if let Some((_, x)) = k.pair {
                // every instance in the list of its tile's first slot, so a
                // tile has one summing thread per entry, the other slots none
                let owner = &self.tile_owner[x];
                let (ptr, list) = by_slot(inst.as_chunks::<3>().0.iter().map(|rec| owner[rec[2] as usize]), self.cross_slots[x]);
                put(image, &format!("C{c}_XOFF"), cast_slice(&ptr));
                put(image, &format!("C{c}_XINST"), cast_slice(&list));
            }
        }
        self.dev.write(&mut self.block, 0, cast_slice(&words))
    }

    /// Changes whenever the block or its offsets change: a graph of the
    /// launches replays only under the generation it was captured in.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// The scalar of the sums, the parameters, the targets and the references.
    pub fn wide(&self) -> Wide {
        self.wide
    }

    /// A column of the block as a kernel argument.
    fn col(&self, name: &str) -> Arg<'_, D::Block> {
        Arg::Block(&self.block, self.offset[name])
    }

    fn run(&self, kernel: &D::Kernel, n: usize, args: &[Arg<'_, D::Block>]) -> Result<(), GpuError> {
        if n == 0 {
            return Ok(());
        }
        self.dev.launch(kernel, launch_n(n), args)
    }

    fn zero(&self, t: Target<'_, D::Block>, n: usize) -> Result<(), GpuError> {
        self.run(&self.kernels.zero, n, &[Arg::Block(t.0, t.1), Arg::U32(n as u32)])
    }

    /// The entities' parameter values and the referenced fields' values at
    /// the parameters `x`; with `derivatives`, their derivatives too.
    fn gather(&self, x: Target<'_, D::Block>, derivatives: bool) -> Result<(), GpuError> {
        for (e, &n) in self.shape.entities.iter().enumerate() {
            let len = n * self.slots[e];
            let args = [Arg::Block(x.0, x.1), self.col(&format!("E{e}_IDX")), self.col(&format!("E{e}_FIXED")), self.col(&format!("E{e}_EV")), Arg::U32(len as u32)];
            self.run(&self.kernels.gather, len, &args)?;
        }
        for (r, rf) in self.shape.references.iter().enumerate() {
            let (e, s, k) = (rf.entity, self.slots[rf.entity], self.kernels.reference(rf.kind));
            let (refr, ev, at, valw, val) =
                (self.col(&format!("R{r}_REFR")), self.col(&format!("E{e}_EV")), self.col(&format!("R{r}_AT")), self.col(&format!("R{r}_VALW")), self.col(&format!("R{r}_VAL")));
            if derivatives {
                let args = [refr, ev, at, valw, val, self.col(&format!("R{r}_DER")), Arg::U32(s as u32)];
                self.run(&k.derivatives, s, &args)?;
            } else {
                self.run(&k.values, s, &[refr, ev, at, valw, val, Arg::U32(s as u32)])?;
            }
        }
        Ok(())
    }

    /// The gradient (`ng` values at `g`), the pattern's tile values (`nh`
    /// at `h`) and the cost (one value at `cost`) at the parameters `x`,
    /// each zeroed first, all in [`wide`](Self::wide); nothing read back.
    pub fn assemble_at(
        &self,
        x: Target<'_, D::Block>,
        g: Target<'_, D::Block>,
        ng: usize,
        h: Target<'_, D::Block>,
        nh: usize,
        cost: Target<'_, D::Block>,
    ) -> Result<(), GpuError> {
        self.zero(g, ng)?;
        self.zero(h, nh)?;
        self.zero(cost, 1)?;
        self.gather(x, true)?;
        let (b, h, g, cost) = (Arg::Block(&self.block, 0), Arg::Block(h.0, h.1), Arg::Block(g.0, g.1), Arg::Block(cost.0, cost.1));
        for (c, &n) in self.counts.iter().enumerate() {
            if n == 0 {
                continue;
            }
            let (k, shape) = (&self.kernels.constraints[c], self.shape.constraints[c]);
            let (na, sa) = (self.shape.entities[shape.a], self.slots[shape.a]);
            self.run(&k.gh, n, &[b, Arg::U32(n as u32)])?;
            self.dev.launch(&k.csum, Launch::linear(THREADS, THREADS), &[b, cost, Arg::U32(parts(n) as u32)])?;
            self.run(&k.ra, sa * (na * (na + 1) / 2 + na), &[b, h, g, Arg::U32(sa as u32)])?;
            if let (Some((rb, rx)), Some((eb, x))) = (&k.pair, shape.pair) {
                let (nb, sb, sx) = (self.shape.entities[eb], self.slots[eb], self.cross_slots[x]);
                self.run(rb, sb * (nb * (nb + 1) / 2 + nb), &[b, h, g, Arg::U32(sb as u32)])?;
                self.run(rx, sx * na * nb, &[b, h, Arg::U32(sx as u32)])?;
            }
        }
        Ok(())
    }

    /// The cost (one value at `cost`) at the parameters `x`, zeroed first;
    /// the referenced fields' derivatives are not computed. Nothing read back.
    pub fn cost_at(&self, x: Target<'_, D::Block>, cost: Target<'_, D::Block>) -> Result<(), GpuError> {
        self.zero(cost, 1)?;
        self.gather(x, false)?;
        let (b, cost) = (Arg::Block(&self.block, 0), Arg::Block(cost.0, cost.1));
        for (c, &n) in self.counts.iter().enumerate() {
            if n == 0 {
                continue;
            }
            let k = &self.kernels.constraints[c];
            self.run(&k.cost, n, &[b, Arg::U32(n as u32)])?;
            self.dev.launch(&k.csum, Launch::linear(THREADS, THREADS), &[b, cost, Arg::U32(parts(n) as u32)])?;
        }
        Ok(())
    }

    /// The referenced fields' steps of the parameters `x` zeroed, as
    /// re-centring after [`advance`](Self::advance) leaves them.
    pub fn zero_steps(&self, x: Target<'_, D::Block>) -> Result<(), GpuError> {
        for (r, rf) in self.shape.references.iter().enumerate() {
            let (e, s) = (rf.entity, self.slots[rf.entity]);
            let args = [
                Arg::Block(x.0, x.1),
                self.col(&format!("E{e}_IDX")),
                self.col(&format!("R{r}_AT")),
                Arg::U32(rf.kind.step_len() as u32),
                Arg::U32(s as u32),
            ];
            self.run(&self.kernels.zero_steps, s, &args)?;
        }
        Ok(())
    }

    /// The references moved by the steps of the last gathered `x` (an
    /// accepted step's), as the model's `advance_params` moves its own.
    pub fn advance(&self) -> Result<(), GpuError> {
        for (r, rf) in self.shape.references.iter().enumerate() {
            let (e, s) = (rf.entity, self.slots[rf.entity]);
            let args = [self.col(&format!("R{r}_REFR")), self.col(&format!("E{e}_EV")), self.col(&format!("R{r}_AT")), Arg::U32(s as u32)];
            self.run(&self.kernels.reference(rf.kind).advance, s, &args)?;
        }
        Ok(())
    }

    /// Write the device's references into the model's optimized referenced
    /// fields ([`DeviceRoot::device_restore`]), one read per field.
    pub fn restore(&self, root: &mut dyn DeviceRoot<T>) -> Result<(), GpuError> {
        let mut refr: Vec<Vec<f64>> = Vec::new();
        for (r, rf) in self.shape.references.iter().enumerate() {
            let n = rf.kind.reference_len() * self.slots[rf.entity];
            let mut bytes = vec![0u8; 8 * n];
            self.dev.read(&self.block, self.offset[&format!("R{r}_REFR")], &mut bytes)?;
            let mut v = vec![0f64; n];
            self.wide.decode(&bytes, &mut v);
            refr.push(v);
        }
        let mut at = HashMap::new();
        for (r, ix) in self.ref_index.iter().enumerate() {
            for (s, &g) in ix.iter().enumerate() {
                if g != FIXED && self.packed[r][s] {
                    at.insert(g, (r, s));
                }
            }
        }
        let mut found = std::collections::HashSet::new();
        root.device_restore(&mut |g| {
            let &(r, s) = at.get(&g)?;
            found.insert(g);
            let kind = self.shape.references[r].kind;
            let n = kind.reference_len();
            Some(kind.reference(&refr[r][n * s..n * s + n]))
        });
        assert_eq!(found.len(), at.len(), "the model's walk reaches every reference the device re-centred");
        Ok(())
    }

    /// The device the assembly runs on.
    pub fn device(&self) -> &Arc<D> {
        &self.dev
    }
}

/// Assemble `problem` at `x` on the CPU and on `dev` in `wide` against one
/// tiled pattern (the reduced route's), and compare (for tests). The device
/// is laid out from the model at `x0`, as a solve lays it out at its start
/// and assembles at the points it moves to: what it computes at `x` must
/// not lean on the model's caches at `x0`.
pub fn device_check<T, D>(problem: &mut dyn crate::simple_lm::LmProblemInternals<T>, x0: &[T], x: &[T], dev: &Arc<D>, wide: Wide) -> DeviceCheck
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
    // the device laid out from the model at x0
    problem.update_params(x0);
    let root = problem.device_root().expect("the root has a device assembly");
    let mut resolver = arael_faer::bsc::PositionResolver::new(&hsym);
    let asm = DeviceAssembly::<T, D>::new(dev, root, &mut |i, j| resolver.resolve_tile(i as usize, j as usize), wide)
        .expect("lay the root out on the device");
    // x, the gradient, the values and the cost in one block of the test's own
    let (at_g, at_h) = (n * 8, n * 16);
    let at_cost = at_h + nv * 8;
    let mut work = dev.alloc(at_cost + 8).expect("allocate the targets");
    let x64: Vec<f64> = x.iter().map(|v| v.to_f64().unwrap()).collect();
    dev.write(&mut work, 0, &wide.encode(&x64)).expect("upload x");
    asm.assemble_at((&work, 0), (&work, at_g), n, (&work, at_h), nv, (&work, at_cost)).expect("assemble on the device");
    let read = |at: usize, len: usize| -> Vec<f64> {
        let mut bytes = vec![0u8; 8 * len];
        dev.read(&work, at, &mut bytes).expect("read a target");
        let mut v = vec![0f64; len];
        wide.decode(&bytes, &mut v);
        v
    };
    let (gd, hd, cost) = (read(at_g, n), read(at_h, nv), read(at_cost, 1));
    let err = |dev: &[f64], cpu: &[T]| -> f64 {
        let max = cpu.iter().fold(0.0f64, |m, v| m.max(v.to_f64().unwrap().abs()));
        let diff = dev.iter().zip(cpu).fold(0.0f64, |m, (d, c)| m.max((d - c.to_f64().unwrap()).abs()));
        if max > 0.0 { diff / max } else { diff }
    };
    DeviceCheck { cpu_cost, device_cost: cost[0], grad_err: err(&gd, &grad), hess_err: err(&hd, &vals), values: nv }
}

/// Instances per constraint, from the pack walk's records.
fn instance_counts<T>(shape: &DeviceShape, pack: &DevicePack<T>) -> Vec<usize> {
    shape.constraints.iter().zip(&pack.inst).map(|(c, inst)| inst.len() / c.words.max(1)).collect()
}

fn u32_bytes(v: &[u32]) -> &[u8] {
    arael_gpu::bytemuck::cast_slice(v)
}

/// The bytes of a model scalar slice (`f32` or `f64`).
fn scalar_bytes<T: crate::utils::Float>(v: &[T]) -> &[u8] {
    // SAFETY: the model's scalar is f32 or f64 (asserted in `new`), plain
    // values without padding.
    unsafe { core::slice::from_raw_parts(v.as_ptr() as *const u8, core::mem::size_of_val(v)) }
}
