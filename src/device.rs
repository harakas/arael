//! The device assembly (feature `gpu`): the gradient, the Hessian's tile
//! values and the cost computed on a GPU from the model's constraints.
//!
//! A root the device head covers gets an implementation of [`DeviceRoot`]
//! from `#[arael(root)]`: its layout ([`DeviceShape`]), a structure walk
//! (the entities' indices, the guards, the kept instances' slots), a values
//! walk (the entities' fixed values, references and data, the root's data,
//! the instances' data), the data records' Rust and C++ mirrors and its
//! constraint kernels in C++ ([`DeviceRoot::device_cpp`]).
//! [`DeviceAssembly`] lays a root out on a [`Device`] once per structure and
//! then assembles at any parameter vector into device blocks, which the
//! device solve reads in place: the model's data as columns and records in
//! one block, each value stored once in the table it belongs to (the root,
//! an entity table, an instance table), the sums in the route's [`Wide`]
//! scalar and no atomic writes (`arael-gpu/kernels/assembly.cpp`). The
//! structure is uploaded once; a refresh writes and uploads the values only
//! ([`Staging`]).
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
    /// Per entity table: the bytes of one entity's data record, a multiple
    /// of 16, zero for none.
    pub entity_data: Vec<usize>,
    /// The bytes of the root's data record, a multiple of 16, zero for
    /// none.
    pub root_data: usize,
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
    /// `SimpleEulerAngleParam`: no reference, the angles the parameters
    /// themselves; computed, never re-centred.
    SimpleEuler,
}

impl ReferenceKind {
    /// The values of one reference.
    pub fn reference_len(self) -> usize {
        match self {
            ReferenceKind::Euler => 9,
            ReferenceKind::Quaternion | ReferenceKind::UnitVec => 4,
            ReferenceKind::Transform => 7,
            ReferenceKind::SimpleEuler => 0,
        }
    }

    /// Whether an accepted step moves the reference and zeroes the step.
    pub fn recentres(self) -> bool {
        self != ReferenceKind::SimpleEuler
    }

    /// The parameters of one step.
    pub fn step_len(self) -> usize {
        match self {
            ReferenceKind::Euler | ReferenceKind::Quaternion | ReferenceKind::SimpleEuler => 3,
            ReferenceKind::Transform => 6,
            ReferenceKind::UnitVec => 2,
        }
    }

    /// The values a constraint reads of one entity: a rotation matrix (9),
    /// a rotation matrix and a translation (12), a direction (3).
    pub fn values(self) -> usize {
        match self {
            ReferenceKind::Euler | ReferenceKind::Quaternion | ReferenceKind::SimpleEuler => 9,
            ReferenceKind::Transform => 12,
            ReferenceKind::UnitVec => 3,
        }
    }

    /// The derivatives of the values by the step a linearization reads.
    pub fn derivatives(self) -> usize {
        match self {
            ReferenceKind::Euler | ReferenceKind::Quaternion | ReferenceKind::SimpleEuler => 27,
            ReferenceKind::Transform => 45,
            ReferenceKind::UnitVec => 6,
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
            ReferenceKind::SimpleEuler => "simple_euler",
        }
    }

    /// The reference as the model takes it back.
    fn reference(self, v: &[f64]) -> crate::model::Reference {
        use crate::model::Reference;
        match self {
            ReferenceKind::Euler => Reference::Matrix(v.try_into().unwrap()),
            ReferenceKind::Quaternion | ReferenceKind::UnitVec => Reference::Quaternion(v.try_into().unwrap()),
            ReferenceKind::Transform => Reference::Transform(v.try_into().unwrap()),
            ReferenceKind::SimpleEuler => unreachable!("a SimpleEulerAngleParam has no reference"),
        }
    }

    const ALL: [ReferenceKind; 5] =
        [ReferenceKind::Euler, ReferenceKind::Quaternion, ReferenceKind::Transform, ReferenceKind::UnitVec, ReferenceKind::SimpleEuler];
}

/// One constraint of a [`DeviceShape`]: its entities (its sides), each of
/// which it writes into its own tile and gradient, and the cross tiles
/// between pairs of them.
///
/// ```ignore
/// // three lines, a cross tile per pair: an instance is the words
/// // [s0, s1, s2, x01, x02, x12]
/// DeviceConstraint { data: 0, rows: 3, sides: vec![0, 0, 0], pairs: vec![(0, 1, 0), (0, 2, 1), (1, 2, 2)] }
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceConstraint {
    /// Its instance data columns.
    pub data: usize,
    /// Its residual rows.
    pub rows: usize,
    /// Per side, its entity array, in the order of an instance's slots and
    /// of a row's partials.
    pub sides: Vec<usize>,
    /// Per cross tile of an instance: its two sides and its cross array.
    pub pairs: Vec<(usize, usize, usize)>,
}

impl DeviceConstraint {
    /// The words of one instance: its sides' slots, then its pairs' cross
    /// slots.
    pub fn words(&self) -> usize {
        self.sides.len() + self.pairs.len()
    }

    /// The parameters of its sides, in order.
    fn params(&self, shape: &DeviceShape) -> usize {
        self.sides.iter().map(|&e| shape.entities[e]).sum()
    }
}

/// What a root's structure walk reads from the model, once per structure:
/// per entity table the parameters' global indices slot by slot (`FIXED`
/// for a fixed one), per constraint the instances its guard keeps (their
/// words, [`DeviceConstraint::words`] each: each side's slot, then each
/// pair's cross slot) and, in walk order, which instances it keeps.
#[derive(Clone, Debug, Default)]
pub struct DeviceStructure {
    pub idx: Vec<Vec<u32>>,
    pub rows: Vec<Vec<u32>>,
    pub keep: Vec<Vec<bool>>,
}

impl DeviceStructure {
    /// Empty, for `shape`.
    pub fn new(shape: &DeviceShape) -> Self {
        DeviceStructure {
            idx: shape.entities.iter().map(|_| Vec::new()).collect(),
            rows: shape.constraints.iter().map(|_| Vec::new()).collect(),
            keep: shape.constraints.iter().map(|_| Vec::new()).collect(),
        }
    }
}

/// Word `q` of every instance of `rows`, `w` words each.
fn word(rows: &[u32], w: usize, q: usize) -> impl Iterator<Item = u32> + Clone + '_ {
    rows.iter().skip(q).step_by(w).copied()
}

/// A data leaf of a model as `f64`, for the values walk to write into a
/// column or record of the root's scalar: a float (`f32`, `f64`, `df32`),
/// or an integer or `bool` field that picks a select's arm (exact as long
/// as it is below 2^24 in `f32`, 2^53 in `f64`; a larger one stays outside
/// any arm).
pub trait DeviceLeaf {
    fn device_leaf(self) -> f64;
}

macro_rules! device_leaf_as {
    ($($t:ty),*) => { $(
        impl DeviceLeaf for $t {
            #[inline]
            fn device_leaf(self) -> f64 { self as f64 }
        }
    )* };
}
device_leaf_as!(f32, f64, i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);

impl DeviceLeaf for bool {
    #[inline]
    fn device_leaf(self) -> f64 {
        u8::from(self) as f64
    }
}

impl DeviceLeaf for crate::df32 {
    #[inline]
    fn device_leaf(self) -> f64 {
        leaf_f64(self)
    }
}

/// A model's leaf value (`f32`, `f64` or `df32`) as `f64`, for the walks
/// the device head generates: a mixed parameter keeps its step and its
/// reference rotation in the root's wide scalar.
pub fn leaf_f64<V: num::ToPrimitive>(v: V) -> f64 {
    v.to_f64().expect("a model's scalars convert to f64")
}

/// Where the values walk writes, in the image's staging buffer: the values
/// columns of a [`DeviceShape`], each at its offset. A write takes a whole
/// row, checked against its column's extent once.
pub struct DeviceWriter<'a, T> {
    bytes: &'a mut [u8],
    wide: Wide,
    /// Per entity table: the fixed values' offset, the parameters, the slots.
    fixed: Vec<(usize, usize, usize)>,
    /// Per entity table: the data records' offset and record bytes.
    data: Vec<(usize, usize)>,
    /// Per referenced field: the references' offset and values per slot.
    refr: Vec<(usize, usize, usize)>,
    /// The root's record: offset and bytes.
    root: (usize, usize),
    /// Per constraint: its instance columns' offsets and its rows.
    inst: Vec<(Vec<usize>, usize)>,
    _t: core::marker::PhantomData<fn(T)>,
}

impl<T: crate::utils::Float> DeviceWriter<'_, T> {
    /// The values `v` from byte `at` on, back to back.
    #[inline]
    fn run<V: Copy>(&mut self, at: usize, v: &[V]) {
        let n = core::mem::size_of_val(v);
        assert!(at + n <= self.bytes.len(), "a write past the staging buffer's {} bytes", self.bytes.len());
        // SAFETY: within the buffer (checked above); `V` is a plain value
        // (a model scalar, a `u64`, a `Pod` record), copied as bytes.
        unsafe { core::ptr::copy_nonoverlapping(v.as_ptr() as *const u8, self.bytes.as_mut_ptr().add(at), n) }
    }

    /// The parameters of the entity at `slot` of entity table `e`: their
    /// values where they are fixed (a live one is read from the parameters).
    #[inline]
    pub fn fixed<const N: usize>(&mut self, e: usize, slot: usize, v: [T; N]) {
        let (at, n, slots) = self.fixed[e];
        assert!(slot < slots && n == N, "entity table {e}: {N} parameters at slot {slot}, {n} each over {slots} slots");
        self.run(at + slot * N * core::mem::size_of::<T>(), &v);
    }

    /// The reference of referenced field `r` of the entity at `slot`, in
    /// the route's wide scalar.
    #[inline]
    pub fn reference<const K: usize>(&mut self, r: usize, slot: usize, v: [f64; K]) {
        let (at, len, slots) = self.refr[r];
        assert!(slot < slots && len == K, "referenced field {r}: {K} values at slot {slot}, {len} each over {slots} slots");
        let wide = self.wide;
        self.run(at + slot * K * 8, &v.map(|x| wide.bits(x)));
    }

    /// The data record of the entity at `slot` of entity table `e`.
    #[inline]
    pub fn entity_data<R: arael_gpu::bytemuck::Pod>(&mut self, e: usize, slot: usize, rec: R) {
        let (at, bytes) = self.data[e];
        let (_, _, slots) = self.fixed[e];
        assert!(slot < slots && core::mem::size_of::<R>() == bytes, "entity table {e}: a record of {} bytes at slot {slot}, {bytes} over {slots} slots", core::mem::size_of::<R>());
        self.run(at + slot * bytes, core::slice::from_ref(&rec));
    }

    /// The root's data record.
    pub fn root_data<R: arael_gpu::bytemuck::Pod>(&mut self, rec: R) {
        let (at, bytes) = self.root;
        assert_eq!(core::mem::size_of::<R>(), bytes, "the root's record");
        self.run(at, core::slice::from_ref(&rec));
    }

    /// The instance data of constraint `c` at `row`, one value per column.
    #[inline]
    pub fn instance<const N: usize>(&mut self, c: usize, row: usize, v: [T; N]) {
        let (cols, rows) = &self.inst[c];
        assert!(row < *rows && cols.len() == N, "constraint {c}: {N} columns at row {row}, {} over {rows} rows", cols.len());
        let (base, size, len) = (self.bytes.as_mut_ptr(), core::mem::size_of::<T>(), self.bytes.len());
        for (&at, &x) in cols.iter().zip(&v) {
            debug_assert!(at + (row + 1) * size <= len);
            // SAFETY: column `j` holds `rows` values of `T` within the
            // buffer (the layout's), and `row < rows`.
            unsafe { (base.add(at + row * size) as *mut T).write_unaligned(x) }
            let _ = len;
        }
    }
}

/// What `#[arael(root)]` generates for a root the device assembly covers.
pub trait DeviceRoot<T> {
    /// The root's layout.
    fn device_shape(&self) -> DeviceShape;
    /// The structure: every entity table's indices in slot order (the slots
    /// the block markers hold, which the walk checks), every constraint's
    /// guards and its kept instances' slots. The model's working copies
    /// must be current (a guard may read them).
    fn device_structure(&self, s: &mut DeviceStructure);
    /// The values: every entity's fixed values, references and data, the
    /// root's data, every kept instance's data (`keep` from
    /// [`device_structure`](Self::device_structure), whose walk order this
    /// walk follows). Panics when the model has other instances than at
    /// the structure walk.
    fn device_values(&self, keep: &[Vec<bool>], w: &mut DeviceWriter<'_, T>);
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

/// The values of one instance's record: per row the residual and the
/// sides' derivatives.
fn record_width(shape: &DeviceShape, c: usize) -> usize {
    let k = &shape.constraints[c];
    k.rows * (1 + k.params(shape))
}

/// The columns of an instance's words, in order: `S<i>` per side, `X<p>`
/// per pair.
fn word_names(k: &DeviceConstraint) -> impl Iterator<Item = String> {
    (0..k.sides.len()).map(|i| format!("S{i}")).chain((0..k.pairs.len()).map(|p| format!("X{p}")))
}

/// The columns of a root, in the layout table's order, and how many of them
/// each of its first two ranges ends at. The structure: per entity table
/// its indices and tiles, per referenced field its step offsets, per cross
/// array its tiles, per constraint its slots and its instances grouped by
/// each side's entity and by each pair's tile. The values: per entity table its fixed values and
/// data records, per referenced field its references, the root's record,
/// per constraint its instance data. Then the columns the kernels write: per
/// entity table its gathered values, per referenced field its values (in F
/// and W) and derivatives, per constraint its records and cost parts. The
/// order depends on the shape alone. W values are 8 bytes.
fn columns(shape: &DeviceShape, slots: &[usize], cross_slots: &[usize], counts: &[usize], ts: usize) -> (Vec<Column>, usize, usize) {
    let col = |name: String, elem: usize, len: usize| Column { name, elem, len };
    let mut v = Vec::new();
    for (e, &n) in shape.entities.iter().enumerate() {
        let s = slots[e];
        v.push(col(format!("E{e}_IDX"), 4, n * s));
        v.push(col(format!("E{e}_TILE"), 4, 3 * s));
    }
    for (r, rf) in shape.references.iter().enumerate() {
        v.push(col(format!("R{r}_AT"), 4, slots[rf.entity]));
    }
    for (x, &s) in cross_slots.iter().enumerate() {
        v.push(col(format!("X{x}_TILE"), 4, 3 * s));
    }
    for (c, k) in shape.constraints.iter().enumerate() {
        let n = counts[c];
        for name in word_names(k) {
            v.push(col(format!("C{c}_{name}"), 4, n));
        }
        for (i, &e) in k.sides.iter().enumerate() {
            v.push(col(format!("C{c}_OFF{i}"), 4, slots[e] + 1));
            v.push(col(format!("C{c}_INST{i}"), 4, n));
        }
        for (p, &(_, _, x)) in k.pairs.iter().enumerate() {
            v.push(col(format!("C{c}_XOFF{p}"), 4, cross_slots[x] + 1));
            v.push(col(format!("C{c}_XINST{p}"), 4, n));
        }
    }
    let structure = v.len();
    for (e, &n) in shape.entities.iter().enumerate() {
        let s = slots[e];
        v.push(col(format!("E{e}_FIXED"), ts, n * s));
        v.push(col(format!("E{e}_DATA"), shape.entity_data[e], s));
    }
    for (r, rf) in shape.references.iter().enumerate() {
        v.push(col(format!("R{r}_REFR"), 8, rf.kind.reference_len() * slots[rf.entity]));
    }
    v.push(col("ROOT".to_string(), shape.root_data, 1));
    for (c, k) in shape.constraints.iter().enumerate() {
        for j in 0..k.data {
            v.push(col(format!("C{c}_D{j}"), ts, counts[c]));
        }
    }
    let values = v.len();
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
    (v, structure, values)
}

/// Where every column sits in the block: the table first, then each column
/// on a 256-byte boundary. The table and the structure columns end at
/// `structure_bytes`, the values columns at `host_bytes`: what the host
/// writes. The kernels write the rest.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Layout {
    columns: Vec<Column>,
    offsets: Vec<usize>,
    structure_bytes: usize,
    host_bytes: usize,
    bytes: usize,
}

impl Layout {
    fn new((columns, structure, values): (Vec<Column>, usize, usize)) -> Self {
        let align = |b: usize| b.div_ceil(256) * 256;
        let mut at = align(columns.len() * 8);
        let mut offsets = Vec::with_capacity(columns.len());
        let (mut structure_bytes, mut host_bytes) = (at, at);
        for (k, c) in columns.iter().enumerate() {
            offsets.push(at);
            at += align((c.elem * c.len).max(1));
            if k < structure {
                structure_bytes = at;
            }
            if k < values {
                host_bytes = at;
            }
        }
        Layout { columns, offsets, structure_bytes, host_bytes, bytes: at }
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
/// the sum of its cost parts, the cost alone, and the sums of its records
/// per side and per pair's cross tiles.
struct ConstraintKernels<K> {
    gh: K,
    csum: K,
    cost: K,
    sides: Vec<K>,
    pairs: Vec<K>,
}

/// The kernels of one [`ReferenceKind`]: the values, the values with
/// their derivatives, and, of a kind that re-centres, the references moved
/// by their steps.
struct ReferenceKernels<K> {
    values: K,
    derivatives: K,
    advance: Option<K>,
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

/// Where the host writes the device image before it is copied over.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Staging {
    /// Pinned (page-locked) memory: a copy from it is asynchronous, so the
    /// host goes on while it moves.
    #[default]
    Pinned,
    /// Ordinary memory: the runtime stages each copy itself, and the call
    /// returns once it holds the bytes. For platforms where copies from
    /// pinned memory are slow.
    Pageable,
}

/// The image's host memory of a [`Staging`] kind.
enum HostBuf<H> {
    Pinned(H),
    Pageable(Vec<u8>),
}

impl<H> HostBuf<H> {
    /// `bytes` of `staging`'s kind on `dev`.
    fn new<D: Device<Host = H>>(dev: &D, staging: Staging, bytes: usize) -> Result<Self, GpuError> {
        Ok(match staging {
            Staging::Pinned => HostBuf::Pinned(dev.alloc_host(bytes)?),
            Staging::Pageable => HostBuf::Pageable(vec![0u8; bytes]),
        })
    }

    /// Its bytes, to write, once the copies from it have completed.
    fn bytes_mut<'a, D: Device<Host = H>>(&'a mut self, dev: &D) -> Result<&'a mut [u8], GpuError> {
        match self {
            HostBuf::Pinned(h) => dev.host_mut(h),
            HostBuf::Pageable(v) => Ok(v.as_mut_slice()),
        }
    }

    /// Its first `len` bytes into `dst` from byte `offset` on.
    fn upload<D: Device<Host = H>>(&self, dev: &D, dst: &mut D::Block, offset: usize, len: usize) -> Result<(), GpuError> {
        match self {
            HostBuf::Pinned(h) => dev.write_from_host(dst, offset, h, 0..len),
            HostBuf::Pageable(v) => dev.write(dst, offset, &v[..len]),
        }
    }
}

/// A buffer the assembly writes or reads that is not its own: a block and a
/// byte offset into it.
pub type Target<'a, B> = (&'a B, usize);

/// A root laid out on a [`Device`] in one block: its structure uploaded
/// once, its values written into pinned staging and uploaded per solve; the
/// parameter vector, the targets and the cost are the caller's buffers, in
/// the assembly's [`Wide`] scalar.
pub struct DeviceAssembly<T, D: Device> {
    dev: Arc<D>,
    wide: Wide,
    shape: DeviceShape,
    slots: Vec<usize>,
    cross_slots: Vec<usize>,
    counts: Vec<usize>,
    /// Per referenced field, per slot: the global index of the step's first
    /// live parameter, [`FIXED`] when the step is fixed.
    ref_index: Vec<Vec<u32>>,
    /// Per constraint, in walk order: which instances its guard keeps.
    keep: Vec<Vec<bool>>,
    layout: Layout,
    offset: HashMap<String, usize>,
    block: D::Block,
    /// The values range of the block on the host: what a refresh rewrites.
    values: HostBuf<D::Host>,
    kernels: Kernels<D::Kernel>,
    generation: u64,
    _t: core::marker::PhantomData<fn() -> T>,
}

fn launch_n(n: usize) -> Launch {
    Launch::linear((n as u32).max(1), THREADS)
}

/// Per cross slot of the tables `tiles` (`(base, stride, flip)` each), the
/// first slot whose tile starts at the same value of H.
fn tile_owners(tiles: &[u32]) -> Vec<u32> {
    let n = tiles.len() / 3;
    // (base, slot) in one word, so the sort compares plain integers
    let mut keys: Vec<u64> = (0..n).map(|s| (tiles[3 * s] as u64) << 32 | s as u64).collect();
    keys.sort_unstable();
    let mut owner = vec![0u32; n];
    let mut first = 0u32;
    for (k, &key) in keys.iter().enumerate() {
        let s = key as u32;
        if k == 0 || key >> 32 != keys[k - 1] >> 32 {
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

/// The positions in H, the resolver of the solver's pattern.
type Resolver<'a> = arael_faer::bsc::PositionResolver<'a, arael_faer::SparseIndex>;

/// Lookups a thread takes at least before the work is split over more.
const PAR_MIN: usize = 32 * 1024;

/// `f` over `0..n` cut into at most `threads` ranges of at least `min`,
/// on arael's worker pool (`arael_faer::pool`, the sweeps' threads), each
/// range with its own clone of `r`, the results in range order; on the
/// calling thread alone when there is one range.
fn par_ranges<R: Send>(n: usize, threads: usize, min: usize, r: &Resolver<'_>, f: impl Fn(std::ops::Range<usize>, &mut Resolver<'_>) -> R + Sync) -> Vec<R> {
    let parts = threads.min(n / min.max(1)).max(1);
    if parts == 1 {
        return vec![f(0..n, &mut r.clone())];
    }
    let step = n.div_ceil(parts);
    let mut out: Vec<(Resolver<'_>, Option<R>)> = (0..parts).map(|_| (r.clone(), None)).collect();
    arael_faer::pool::run_over(&mut out, |k, (r, res)| *res = Some(f(k * step..((k + 1) * step).min(n), r)));
    out.into_iter().map(|(_, res)| res.expect("every range ran")).collect()
}

/// The tiles of the structure `st` in the pattern `r` resolves, on up to
/// `threads` threads of at least `min` lookups each: per entity table
/// `(base, stride, start)` per slot,
/// per cross array `(base, stride, flip)` per slot its instances name,
/// `flip` 1 when the second side's span is the lower-numbered one and lies
/// on the tile's rows. Zeros where a side is fixed throughout.
fn tiles(
    shape: &DeviceShape,
    st: &DeviceStructure,
    slots: &[usize],
    cross_slots: &[usize],
    r: &Resolver<'_>,
    (threads, min): (usize, usize),
) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
    use crate::store::{device_word, tile_extent};
    let extent = |e: usize, s: usize| {
        let n = shape.entities[e];
        tile_extent(&st.idx[e][s * n..(s + 1) * n])
    };
    let entity: Vec<Vec<u32>> = (0..shape.entities.len())
        .map(|e| {
            par_ranges(slots[e], threads, min, r, |range, r| {
                range
                    .flat_map(|s| {
                        let (start, live) = extent(e, s);
                        if live == 0 {
                            return [0, 0, 0];
                        }
                        let (base, stride, _) = r.resolve_tile(start as usize, start as usize);
                        [device_word(base), device_word(stride), start]
                    })
                    .collect::<Vec<u32>>()
            })
            .concat()
        })
        .collect();
    let mut cross: Vec<Vec<u32>> = cross_slots.iter().map(|&n| vec![0u32; 3 * n]).collect();
    let mut done: Vec<Vec<bool>> = cross_slots.iter().map(|&n| vec![false; n]).collect();
    for (c, k) in shape.constraints.iter().enumerate() {
        let (w, rows) = (k.words(), &st.rows[c]);
        for (p, &(i, j, x)) in k.pairs.iter().enumerate() {
            let (ea, eb) = (k.sides[i], k.sides[j]);
            // per row its tile, then the first row of each cross slot keeps it
            let row_tiles = par_ranges(rows.len() / w, threads, min, r, |range, r| {
                rows[range.start * w..range.end * w]
                    .chunks_exact(w)
                    .map(|rec| (rec[i], rec[j]))
                    .map(|(a, b)| {
                        // the sides' first live indices from their entity
                        // tiles, a zero stride where a side is fixed throughout
                        let (ta, tb) = (&entity[ea][3 * a as usize..], &entity[eb][3 * b as usize..]);
                        if ta[1] == 0 || tb[1] == 0 {
                            return [0, 0, 0];
                        }
                        let (ra, rb) = (ta[2], tb[2]);
                        let flip = rb < ra;
                        let (base, stride, _) = if flip { r.resolve_tile(rb as usize, ra as usize) } else { r.resolve_tile(ra as usize, rb as usize) };
                        [device_word(base), device_word(stride), flip as u32]
                    })
                    .collect::<Vec<[u32; 3]>>()
            });
            for (s, tile) in word(rows, w, k.sides.len() + p).zip(row_tiles.iter().flatten()) {
                let s = s as usize;
                if !done[x][s] {
                    done[x][s] = true;
                    cross[x][3 * s..3 * s + 3].copy_from_slice(tile);
                }
            }
        }
    }
    (entity, cross)
}

/// The block's table and structure columns into `image` (the structure
/// range): the entity tables' indices and tiles, the referenced fields'
/// step offsets, the cross tiles, the instances' slots and their groupings
/// by entity and by tile. Returns per referenced field, per slot, the
/// step's first live global index (the key of its reference).
#[allow(clippy::too_many_arguments)]
fn write_structure(
    image: &mut [u8],
    layout: &Layout,
    offset: &HashMap<String, usize>,
    shape: &DeviceShape,
    st: &DeviceStructure,
    slots: &[usize],
    cross_slots: &[usize],
    entity_tiles: &[Vec<u32>],
    cross_tiles: &[Vec<u32>],
) -> Vec<Vec<u32>> {
    use arael_gpu::bytemuck::cast_slice;
    let put = |image: &mut [u8], name: &str, bytes: &[u8]| {
        let o = offset[name];
        image[o..o + bytes.len()].copy_from_slice(bytes);
    };
    for (k, &o) in layout.offsets.iter().enumerate() {
        image[8 * k..8 * k + 8].copy_from_slice(&(o as u64).to_le_bytes());
    }
    for (e, (idx, tile)) in st.idx.iter().zip(entity_tiles).enumerate() {
        put(image, &format!("E{e}_IDX"), cast_slice(idx));
        put(image, &format!("E{e}_TILE"), cast_slice(tile));
    }
    let mut ref_index = Vec::with_capacity(shape.references.len());
    for (r, rf) in shape.references.iter().enumerate() {
        let n = shape.entities[rf.entity];
        let at: Vec<u32> = (0..slots[rf.entity]).map(|s| (s * n + rf.step) as u32).collect();
        put(image, &format!("R{r}_AT"), cast_slice(&at));
        // a step's key: its first live parameter (a TransformParam may hold
        // one half of its step fixed)
        let idx = &st.idx[rf.entity];
        ref_index.push(
            at.iter()
                .map(|&a| idx[a as usize..a as usize + rf.kind.step_len()].iter().copied().find(|&g| g != FIXED).unwrap_or(FIXED))
                .collect(),
        );
    }
    let owners: Vec<Vec<u32>> = cross_tiles.iter().map(|t| tile_owners(t)).collect();
    for (x, t) in cross_tiles.iter().enumerate() {
        put(image, &format!("X{x}_TILE"), cast_slice(t));
    }
    for (c, k) in shape.constraints.iter().enumerate() {
        let (w, rows) = (k.words(), &st.rows[c]);
        for (q, name) in word_names(k).enumerate() {
            let o = offset[&format!("C{c}_{name}")];
            for (b, v) in image[o..].as_chunks_mut::<4>().0.iter_mut().zip(word(rows, w, q)) {
                *b = v.to_le_bytes();
            }
        }
        for (i, &e) in k.sides.iter().enumerate() {
            let (ptr, list) = by_slot(word(rows, w, i), slots[e]);
            put(image, &format!("C{c}_OFF{i}"), cast_slice(&ptr));
            put(image, &format!("C{c}_INST{i}"), cast_slice(&list));
        }
        for (p, &(_, _, x)) in k.pairs.iter().enumerate() {
            // every instance in the list of its tile's first slot, so a
            // tile has one summing thread per entry, the other slots none
            let owner = &owners[x];
            let (ptr, list) = by_slot(word(rows, w, k.sides.len() + p).map(|s| owner[s as usize]), cross_slots[x]);
            put(image, &format!("C{c}_XOFF{p}"), cast_slice(&ptr));
            put(image, &format!("C{c}_XINST{p}"), cast_slice(&list));
        }
    }
    ref_index
}

impl<T: crate::utils::Float + 'static, D: Device> DeviceAssembly<T, D> {
    /// Lay `root` out on `dev` against the tiled pattern `pattern` (its
    /// tiles looked up on up to `threads` threads), its sums in `wide`,
    /// compile its kernels, and upload its structure and values from
    /// `staging`. The model's working copies must be current (its guards,
    /// fixed values and references are read).
    pub fn new(
        dev: &Arc<D>,
        root: &dyn DeviceRoot<T>,
        pattern: &arael_faer::bsc::SymbolicSparseBlockColMat<arael_faer::SparseIndex>,
        threads: usize,
        wide: Wide,
        staging: Staging,
    ) -> Result<Self, GpuError> {
        let ts = core::mem::size_of::<T>();
        assert!(ts == 4 || ts == 8, "the device assembly takes an f32 or an f64 model");
        let shape = root.device_shape();
        let mut st = DeviceStructure::new(&shape);
        root.device_structure(&mut st);
        let slots: Vec<usize> = shape.entities.iter().zip(&st.idx).map(|(&n, ix)| ix.len() / n.max(1)).collect();
        let counts: Vec<usize> = shape.constraints.iter().zip(&st.rows).map(|(k, r)| r.len() / k.words()).collect();
        let mut cross_slots = vec![0usize; shape.crosses];
        for (c, k) in shape.constraints.iter().enumerate() {
            for (p, &(_, _, x)) in k.pairs.iter().enumerate() {
                if let Some(m) = word(&st.rows[c], k.words(), k.sides.len() + p).max() {
                    cross_slots[x] = cross_slots[x].max(m as usize + 1);
                }
            }
        }
        let (entity_tiles, cross_tiles) = tiles(&shape, &st, &slots, &cross_slots, &Resolver::new(pattern), (threads, PAR_MIN));
        let layout = Layout::new(columns(&shape, &slots, &cross_slots, &counts, ts));
        let scalar = if ts == 8 { "double" } else { "float" };
        let defines = format!("{}#define SCALAR {scalar}\n{}", wide.defines(), layout.defines());
        let source = format!("{WIDE}\n{}\n{}", arael_gpu::ASSEMBLY, root.device_cpp());
        // every entry point in one lookup of the source, in the order the
        // kernels take them below
        let mut entries: Vec<String> = ["k_gather_params", "k_zero_wide", "k_zero_steps"].map(String::from).to_vec();
        for kind in ReferenceKind::ALL {
            entries.extend(["values", "derivatives"].map(|e| format!("k_{}_{e}", kind.kernel_name())));
            if kind.recentres() {
                entries.push(format!("k_{}_advance", kind.kernel_name()));
            }
        }
        for (c, k) in shape.constraints.iter().enumerate() {
            entries.extend(["gh", "csum", "cost"].map(|e| format!("__dk{c}_{e}")));
            entries.extend((0..k.sides.len()).map(|i| format!("__dk{c}_r{i}")));
            entries.extend((0..k.pairs.len()).map(|p| format!("__dk{c}_x{p}")));
        }
        let entries: Vec<&str> = entries.iter().map(String::as_str).collect();
        let mut compiled = dev.compile_many(&source, &defines, &entries)?.into_iter();
        let mut k = || compiled.next().expect("a kernel per entry point");
        let kernels = Kernels {
            gather: k(),
            zero: k(),
            zero_steps: k(),
            references: ReferenceKind::ALL
                .iter()
                .map(|kind| ReferenceKernels { values: k(), derivatives: k(), advance: kind.recentres().then(&mut k) })
                .collect(),
            constraints: shape
                .constraints
                .iter()
                .map(|c| ConstraintKernels {
                    gh: k(),
                    csum: k(),
                    cost: k(),
                    sides: c.sides.iter().map(|_| k()).collect(),
                    pairs: c.pairs.iter().map(|_| k()).collect(),
                })
                .collect(),
        };
        let mut block = dev.alloc(layout.bytes)?;
        let offset = layout.index();
        // the structure: the table and the structure columns, copied once
        let mut host = HostBuf::new(&**dev, staging, layout.structure_bytes)?;
        let ref_index = write_structure(host.bytes_mut(&**dev)?, &layout, &offset, &shape, &st, &slots, &cross_slots, &entity_tiles, &cross_tiles);
        host.upload(&**dev, &mut block, 0, layout.structure_bytes)?;
        drop(host);
        let values = HostBuf::new(&**dev, staging, layout.host_bytes - layout.structure_bytes)?;
        let mut asm = DeviceAssembly {
            dev: dev.clone(),
            wide,
            shape,
            slots,
            cross_slots,
            counts,
            ref_index,
            keep: st.keep,
            layout,
            offset,
            block,
            values,
            kernels,
            generation: 0,
            _t: core::marker::PhantomData,
        };
        asm.refresh(root)?;
        Ok(asm)
    }

    /// Read the model's values again (the entities' fixed values,
    /// references and data, the root's data, the instances' data), which a
    /// solve may have been handed changed, and upload them. The structure,
    /// the block and its layout stay, so graphs recorded against it stay
    /// valid. The model's working copies must be current; a model of
    /// another structure panics (the session's structure is rebuilt by
    /// `LmSession::invalidate`).
    pub fn refresh(&mut self, root: &dyn DeviceRoot<T>) -> Result<(), GpuError> {
        let base = self.layout.structure_bytes;
        let at = |name: &str| self.offset[name] - base;
        let n_ent = self.shape.entities.len();
        let mut w = DeviceWriter {
            bytes: self.values.bytes_mut(&*self.dev)?,
            wide: self.wide,
            fixed: (0..n_ent).map(|e| (at(&format!("E{e}_FIXED")), self.shape.entities[e], self.slots[e])).collect(),
            data: (0..n_ent).map(|e| (at(&format!("E{e}_DATA")), self.shape.entity_data[e])).collect(),
            refr: self
                .shape
                .references
                .iter()
                .enumerate()
                .map(|(r, rf)| (at(&format!("R{r}_REFR")), rf.kind.reference_len(), self.slots[rf.entity]))
                .collect(),
            root: (at("ROOT"), self.shape.root_data),
            inst: self
                .shape
                .constraints
                .iter()
                .enumerate()
                .map(|(c, k)| ((0..k.data).map(|j| at(&format!("C{c}_D{j}"))).collect(), self.counts[c]))
                .collect(),
            _t: core::marker::PhantomData,
        };
        root.device_values(&self.keep, &mut w);
        let len = self.layout.host_bytes - base;
        self.values.upload(&*self.dev, &mut self.block, base, len)
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
            let (k, shape) = (&self.kernels.constraints[c], &self.shape.constraints[c]);
            self.run(&k.gh, n, &[b, Arg::U32(n as u32)])?;
            self.dev.launch(&k.csum, Launch::linear(THREADS, THREADS), &[b, cost, Arg::U32(parts(n) as u32)])?;
            // per side its entities' tiles and gradients, per pair its cross
            // tiles, a thread per entry
            for (&e, kr) in shape.sides.iter().zip(&k.sides) {
                let (ne, se) = (self.shape.entities[e], self.slots[e]);
                self.run(kr, se * (ne * (ne + 1) / 2 + ne), &[b, h, g, Arg::U32(se as u32)])?;
            }
            for (&(i, j, x), kx) in shape.pairs.iter().zip(&k.pairs) {
                let (ni, nj, sx) = (self.shape.entities[shape.sides[i]], self.shape.entities[shape.sides[j]], self.cross_slots[x]);
                self.run(kx, sx * ni * nj, &[b, h, Arg::U32(sx as u32)])?;
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
        for (r, rf) in self.shape.references.iter().enumerate().filter(|(_, rf)| rf.kind.recentres()) {
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
            let Some(advance) = &self.kernels.reference(rf.kind).advance else { continue };
            let (e, s) = (rf.entity, self.slots[rf.entity]);
            let args = [self.col(&format!("R{r}_REFR")), self.col(&format!("E{e}_EV")), self.col(&format!("R{r}_AT")), Arg::U32(s as u32)];
            self.run(advance, s, &args)?;
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
        for (r, ix) in self.ref_index.iter().enumerate().filter(|(r, _)| self.shape.references[*r].kind.recentres()) {
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
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(8);
    let asm = DeviceAssembly::<T, D>::new(dev, root, &hsym, threads, wide, Staging::Pinned)
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The tiles looked up on several threads equal the ones looked up on
    /// one: two entity tables (three and two parameters a slot, some fixed
    /// throughout), a cross constraint between them, every row a lookup.
    #[test]
    fn tiles_on_several_threads_are_the_tiles_on_one() {
        let (s0, s1, nrows) = (50usize, 70usize, 300usize);
        let mut idx0: Vec<u32> = (0..3 * s0 as u32).collect();
        let mut idx1: Vec<u32> = (0..2 * s1 as u32).map(|i| 3 * s0 as u32 + i).collect();
        // entity 7 of the first table and entity 11 of the second fixed
        idx0[21..24].fill(FIXED);
        idx1[22..24].fill(FIXED);
        let cols: Vec<Vec<u32>> = vec![
            (0..nrows as u32).map(|k| k * 7 % s0 as u32).collect(),
            (0..nrows as u32).map(|k| k * 11 % s1 as u32).collect(),
            (0..nrows as u32).collect(),
        ];
        let shape = DeviceShape {
            entities: vec![3, 2],
            references: Vec::new(),
            crosses: 1,
            constraints: vec![DeviceConstraint { data: 0, rows: 1, sides: vec![0, 1], pairs: vec![(0, 1, 0)] }],
            entity_data: vec![0, 0],
            root_data: 0,
        };
        let rows: Vec<u32> = (0..nrows).flat_map(|k| [cols[0][k], cols[1][k], cols[2][k]]).collect();
        let st = DeviceStructure { idx: vec![idx0, idx1], rows: vec![rows], keep: vec![vec![true; nrows]] };
        // one block per entity, the diagonal blocks and the coupled pairs
        let mut part = vec![0usize];
        for k in 0..s0 + s1 {
            part.push(part[k] + if k < s0 { 3 } else { 2 });
        }
        let mut cells: Vec<(usize, usize)> = (0..s0 + s1).map(|b| (b, b)).collect();
        for (&a, &b) in cols[0].iter().zip(&cols[1]) {
            cells.push((a as usize, s0 + b as usize));
        }
        cells.sort_unstable();
        cells.dedup();
        // one scalar coordinate per block cell, its first
        let p = crate::simple_lm::partition_idx(&part);
        let pattern = arael_faer::bsc::SymbolicSparseBlockColMat::covering(p.clone(), p, cells.len(), |k| (part[cells[k].0], part[cells[k].1]));
        let r = Resolver::new(&pattern);
        let one = tiles(&shape, &st, &[s0, s1], &[nrows], &r, (1, usize::MAX));
        let many = tiles(&shape, &st, &[s0, s1], &[nrows], &r, (4, 1));
        assert_eq!(one, many);
        // the fixed entities have no tile, the others one each
        assert_eq!(one.0[0][21..24], [0, 0, 0]);
        assert!(one.0[0][..21].chunks(3).all(|t| t[1] > 0));
    }
}
