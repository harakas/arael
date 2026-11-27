//! The device assembly (feature `gpu`): the gradient, the Hessian's tile
//! values and the cost computed on a GPU from the model's constraints.
//!
//! A root whose constraints all have a device form gets an implementation
//! of [`DeviceRoot`] from `#[arael(root)]`: its layout ([`DeviceShape`]), a
//! walk of its structure (the entities' indices, the guards, the kept
//! instances' slots), a walk of its values (the entities' fixed values,
//! references and data, the root's data, the instances' data), the data
//! records' Rust and C++ mirrors and its constraint kernels in C++
//! ([`DeviceRoot::device_cpp`]).
//! [`DeviceAssembly`] lays a root out on a [`Device`] once per structure and
//! then assembles at any parameter vector into device blocks, which the
//! device solve reads in place: the model's data as columns and records in
//! one block, each value stored once in the table it belongs to (the root,
//! an entity table, an instance table), the sums in the route's [`Wide`]
//! scalar and no atomic writes (`arael-gpu/kernels/assembly.cpp`). The
//! structure is uploaded once; a refresh writes and uploads the values only
//! ([`Staging`]).
//!
//! The layout table holds the byte offset of every column in the block, one
//! `u64` each, in the order [`columns`] gives them, and the kernels take it
//! as a launch parameter (`Arg::Bytes`, `TABLE_PARAM(ArgLayout) L`, read
//! from the launch's parameters, not from device memory). A kernel reads a
//! column through its slot of the table (`ARAEL_COL_<name>`), so a root's
//! kernels compile once whatever the problem's size, and only the offsets
//! change with it.

use std::collections::HashMap;
use std::sync::Arc;

use arael_gpu::{Arg, Device, FIXED, GpuError, Launch, Scope, Section, TILE_STORE, Values, WIDE, Wide};

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
    /// The root's constraint types, in the order of their kernels.
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
    /// The parameter's type: how its reference composes with its step.
    pub kind: ReferenceKind,
}

/// The kinds of referenced parameters, each with kernels that compose its
/// reference with its step and move the reference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReferenceKind {
    /// `EulerAngleParam`: a rotation matrix (row-major) and Euler angles.
    Euler,
    /// `MixedEulerAngleParam`: [`Euler`](Self::Euler)'s, its reference
    /// kept in the wide scalar.
    MixedEuler,
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
            ReferenceKind::Euler | ReferenceKind::MixedEuler => 9,
            ReferenceKind::Quaternion | ReferenceKind::UnitVec => 4,
            ReferenceKind::Transform => 7,
            ReferenceKind::SimpleEuler => 0,
        }
    }

    /// Whether an accepted step moves the reference and zeroes the step.
    pub fn recentres(self) -> bool {
        self != ReferenceKind::SimpleEuler
    }

    /// The scalar its reference is kept and composed in on a route of
    /// `values`: the root's, as the host's parameter keeps it, the wide one
    /// for a [`MixedEuler`](Self::MixedEuler).
    ///
    /// ```
    /// use arael::__gpu::{ReferenceKind, Values, Wide};
    /// let f32s = Values::f32(Wide::Df32);
    /// assert_eq!(ReferenceKind::Euler.scalar(f32s), f32s);
    /// assert_eq!(ReferenceKind::MixedEuler.scalar(f32s), Values::from(Wide::Df32));
    /// ```
    pub fn scalar(self, values: Values) -> Values {
        if self == ReferenceKind::MixedEuler { values.wide().into() } else { values }
    }

    /// The parameters of one step.
    pub fn step_len(self) -> usize {
        match self {
            ReferenceKind::Euler | ReferenceKind::MixedEuler | ReferenceKind::Quaternion | ReferenceKind::SimpleEuler => 3,
            ReferenceKind::Transform => 6,
            ReferenceKind::UnitVec => 2,
        }
    }

    /// The values a constraint reads of one entity: a rotation matrix (9),
    /// a rotation matrix and a translation (12), a direction (3).
    pub fn values(self) -> usize {
        match self {
            ReferenceKind::Euler | ReferenceKind::MixedEuler | ReferenceKind::Quaternion | ReferenceKind::SimpleEuler => 9,
            ReferenceKind::Transform => 12,
            ReferenceKind::UnitVec => 3,
        }
    }

    /// The derivatives of the values by the step a linearization reads.
    pub fn derivatives(self) -> usize {
        match self {
            ReferenceKind::Euler | ReferenceKind::MixedEuler | ReferenceKind::Quaternion | ReferenceKind::SimpleEuler => 27,
            ReferenceKind::Transform => 45,
            ReferenceKind::UnitVec => 6,
        }
    }

    /// Its kernels' name in `arael-gpu/kernels/assembly.cpp`
    /// (`k_<name>_values`, `_derivatives`, `_advance`).
    fn kernel_name(self) -> &'static str {
        match self {
            ReferenceKind::Euler => "euler",
            ReferenceKind::MixedEuler => "mixed_euler",
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
            ReferenceKind::Euler | ReferenceKind::MixedEuler => Reference::Matrix(v.try_into().unwrap()),
            ReferenceKind::Quaternion | ReferenceKind::UnitVec => Reference::Quaternion(v.try_into().unwrap()),
            ReferenceKind::Transform => Reference::Transform(v.try_into().unwrap()),
            ReferenceKind::SimpleEuler => unreachable!("a SimpleEulerAngleParam has no reference"),
        }
    }

    const ALL: [ReferenceKind; 6] = [
        ReferenceKind::Euler,
        ReferenceKind::MixedEuler,
        ReferenceKind::Quaternion,
        ReferenceKind::Transform,
        ReferenceKind::UnitVec,
        ReferenceKind::SimpleEuler,
    ];
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

/// The structure of `root` ([`DeviceRoot::device_structure`]). The model's
/// working copies must be current: the walk reads the guards.
pub(crate) fn structure_of<T>(root: &dyn DeviceRoot<T>) -> DeviceStructure {
    let shape = root.device_shape();
    let mut st = DeviceStructure::new(&shape);
    root.device_structure(&mut st);
    st
}

/// H's cells of the structure `st` of `root`, as the root's block stores
/// list them, one scalar coordinate per block, the upper one: per entity
/// table, per slot with a live parameter, its first live index twice;
/// per constraint, per kept instance and pair of sides, the two sides'
/// first live indices, the lower one first. A pair with a side fixed
/// throughout lists its live side's diagonal again (or the first entity's,
/// with neither live), a cell already in the list, so that every pair of
/// every instance takes one place and the list is written on up to
/// `threads` threads of the worker pool, by ranges of at least `min`
/// instances. The blocks of an instance a guard drops, and a cross block
/// no constraint writes, are not among the cells: the device writes none
/// of them.
pub(crate) fn cells_of(shape: &DeviceShape, st: &DeviceStructure, out: &mut Vec<(u32, u32)>, (threads, min): (usize, usize)) {
    use crate::store::tile_extent;
    // per entity table, per slot: its first live index, u32::MAX without one
    let first: Vec<Vec<u32>> = shape
        .entities
        .iter()
        .zip(&st.idx)
        .map(|(&n, idx)| if n == 0 { Vec::new() } else { idx.chunks_exact(n).map(|ix| tile_extent(ix).0).collect() })
        .collect();
    let base = out.len();
    for f in &first {
        out.extend(f.iter().filter(|&&s| s != u32::MAX).map(|&s| (s, s)));
    }
    let Some(&fill) = out.get(base) else {
        // no live entity, no cell
        return;
    };
    // ranges of instances of one constraint: (constraint, rows)
    let mut runs: Vec<(usize, core::ops::Range<usize>)> = Vec::new();
    let mut total = 0usize;
    for (c, k) in shape.constraints.iter().enumerate() {
        if k.pairs.is_empty() {
            continue;
        }
        let n = st.rows[c].len() / k.words();
        let parts = threads.min(n / min.max(1)).max(1);
        let step = n.div_ceil(parts).max(1);
        for r in 0..parts {
            let rows = r * step..((r + 1) * step).min(n);
            if !rows.is_empty() {
                total += rows.len() * k.pairs.len();
                runs.push((c, rows));
            }
        }
    }
    let at = out.len();
    out.resize(at + total, fill);
    let mut tasks: Vec<((usize, core::ops::Range<usize>), &mut [(u32, u32)])> = Vec::with_capacity(runs.len());
    let mut rest = &mut out[at..];
    for run in runs {
        let (head, tail) = core::mem::take(&mut rest).split_at_mut(run.1.len() * shape.constraints[run.0].pairs.len());
        tasks.push((run, head));
        rest = tail;
    }
    // the range's cells pair by pair, so the instances that share a cross
    // block (a parent's, say) list it in a run, which the pattern's run
    // compression shrinks to one key
    let write = |_: usize, ((c, rows), cells): &mut ((usize, core::ops::Range<usize>), &mut [(u32, u32)])| {
        let k = &shape.constraints[*c];
        let w = k.words();
        let recs = &st.rows[*c][rows.start * w..rows.end * w];
        for (pc, &(i, j, _)) in cells.chunks_mut(rows.len()).zip(&k.pairs) {
            let (fa, fb) = (&first[k.sides[i]], &first[k.sides[j]]);
            for (cell, rec) in pc.iter_mut().zip(recs.chunks_exact(w)) {
                let (a, b) = (fa[rec[i] as usize], fb[rec[j] as usize]);
                *cell = match (a != u32::MAX, b != u32::MAX) {
                    (true, true) => (a.min(b), a.max(b)),
                    (true, false) => (a, a),
                    (false, true) => (b, b),
                    (false, false) => fill,
                };
            }
        }
    };
    if tasks.len() <= 1 || threads <= 1 {
        tasks.iter_mut().for_each(|t| write(0, t));
    } else {
        arael_faer::pool::run_over(&mut tasks, write);
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
    /// The significand bits the type carries: 24 for `f32`, 48 for `df32`,
    /// 53 for `f64`; an integer or `bool` counts as 53, exact as data.
    const BITS: u32;
    /// The value as the device's data stores it.
    fn device_leaf(self) -> f64;
}

macro_rules! device_leaf_as {
    ($bits:literal: $($t:ty),*) => { $(
        impl DeviceLeaf for $t {
            const BITS: u32 = $bits;
            #[inline]
            fn device_leaf(self) -> f64 { self as f64 }
        }
    )* };
}
device_leaf_as!(24: f32);
device_leaf_as!(53: f64, i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);

impl DeviceLeaf for bool {
    const BITS: u32 = 53;
    #[inline]
    fn device_leaf(self) -> f64 {
        u8::from(self) as f64
    }
}

impl DeviceLeaf for crate::df32 {
    const BITS: u32 = 48;
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
    /// Per entity table: the fixed values' offset, the parameters, the slots.
    fixed: Vec<(usize, usize, usize)>,
    /// Per entity table: the data records' offset and record bytes.
    data: Vec<(usize, usize)>,
    /// Per referenced field: the references' offset, values per slot, slots
    /// and scalar ([`ReferenceKind::scalar`]).
    refr: Vec<(usize, usize, usize, Values)>,
    /// The root's record: offset and bytes.
    root: (usize, usize),
    /// Per constraint: its instance columns' offsets and its rows.
    inst: Vec<(Vec<usize>, usize)>,
    /// The fewest significand bits of the leaves read ([`DeviceLeaf::BITS`]).
    bits: u32,
    _t: core::marker::PhantomData<fn(T)>,
}

impl<T: crate::utils::Float> DeviceWriter<'_, T> {
    /// The leaf `l` as `f64`, its type's bits counted.
    #[inline]
    pub fn leaf<L: DeviceLeaf>(&mut self, l: L) -> f64 {
        self.bits = self.bits.min(L::BITS);
        l.device_leaf()
    }

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
    /// its kind's scalar ([`ReferenceKind::scalar`]).
    #[inline]
    pub fn reference<const K: usize>(&mut self, r: usize, slot: usize, v: [f64; K]) {
        let (at, len, slots, scalar) = self.refr[r];
        assert!(slot < slots && len == K, "referenced field {r}: {K} values at slot {slot}, {len} each over {slots} slots");
        if scalar.is_f32() {
            self.run(at + slot * K * 4, &v.map(|x| x as f32));
        } else {
            let wide = scalar.wide();
            self.run(at + slot * K * 8, &v.map(|x| wide.bits(x)));
        }
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
    /// The root's type name: the name of its kernel dump.
    fn device_name(&self) -> &'static str;
    /// Set the references of the model's optimized referenced parameters
    /// from `refs`, keyed by the first live index of the step, and zero
    /// their steps: a device solve's re-centring handed back.
    fn device_restore(&mut self, refs: &mut dyn FnMut(u32) -> Option<crate::model::Reference>);
}

/// What [`device_check`] found: the host's and the device's cost, and the
/// largest differences of the gradient and of the Hessian's values, each
/// relative to the largest host entry.
#[derive(Clone, Copy, Debug)]
pub struct DeviceCheck {
    /// The cost the host's assembly computed.
    pub host_cost: f64,
    /// The cost the device's assembly computed.
    pub device_cost: f64,
    /// The largest difference of a gradient entry, relative to the largest
    /// host entry.
    pub grad_err: f64,
    /// The largest difference of a Hessian value, relative to the largest
    /// host value.
    pub hess_err: f64,
    /// The pattern's values.
    pub values: usize,
    /// The fewest significand bits of the model's leaves
    /// ([`DeviceAssembly::leaf_bits`]).
    pub leaf_bits: u32,
    /// The most instances of one constraint: the longest chain of adds in
    /// the host's cost sum ([`check_cost_tolerance`]).
    pub chain: usize,
    /// H's blocks from the block stores' cells, and from the device
    /// structure's ([`cells_of`]); the device's are among the stores'
    /// (checked), fewer by the blocks the device never writes.
    pub store_blocks: usize,
    pub device_blocks: usize,
}

/// Threads per block of every assembly kernel (`kernels/assembly.cpp`).
const THREADS: u32 = 256;

/// The environment variable naming the directory a device assembly writes
/// the translation unit it compiles to, as `<root>.<target>.<wide>.cu`
/// (`SPath.sm_89.f64.cu`): the architecture's and the layout's defines,
/// the kernel headers, the shared kernels and the root's.
///
/// ```sh
/// ARAEL_KERNEL_DUMP=target/kernels cargo test --features gpu-cuda --test device_assembly
/// ```
pub const KERNEL_DUMP_VAR: &str = "ARAEL_KERNEL_DUMP";

/// Writes `unit` to `<dir>/<root>.<target>.<wide>.cu`, creating `dir`.
fn dump_unit(dir: &std::path::Path, root: &str, target: &str, wide: Wide, unit: &str) -> Result<std::path::PathBuf, GpuError> {
    let err = |e: std::io::Error| GpuError::Dump(format!("{}: {e}", dir.display()));
    std::fs::create_dir_all(dir).map_err(err)?;
    let path = dir.join(format!("{root}.{target}.{}.cu", wide.name()));
    std::fs::write(&path, unit).map_err(err)?;
    Ok(path)
}

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

/// The image's columns of an instance's words: `S<i>` per side, the first
/// words of its row. Its cross slots, a word per pair after them, stay on
/// the host: no kernel reads them, the instances are grouped by them
/// (`XOFF`, `XINST`).
fn word_names(k: &DeviceConstraint) -> impl Iterator<Item = String> {
    (0..k.sides.len()).map(|i| format!("S{i}"))
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
/// order depends on the shape alone. W values are 8 bytes, the model's
/// `ts`; a reference is in its kind's scalar on a route of `route`.
/// Words of a tile's record in the image's TILE columns: its base in H's
/// values, its stride (with [`TILE_STORE`]), and an entity's start or a
/// cross block's flip ([`tiles`]).
const TILE_WORDS: usize = 3;


fn columns(shape: &DeviceShape, slots: &[usize], cross_slots: &[usize], counts: &[usize], ts: usize, route: Values) -> (Vec<Column>, usize, usize) {
    let col = |name: String, elem: usize, len: usize| Column { name, elem, len };
    let mut v = Vec::new();
    for (e, &n) in shape.entities.iter().enumerate() {
        let s = slots[e];
        v.push(col(format!("E{e}_IDX"), 4, n * s));
        v.push(col(format!("E{e}_TILE"), 4, TILE_WORDS * s));
    }
    for (r, rf) in shape.references.iter().enumerate() {
        v.push(col(format!("R{r}_AT"), 4, slots[rf.entity]));
    }
    for (x, &s) in cross_slots.iter().enumerate() {
        v.push(col(format!("X{x}_TILE"), 4, TILE_WORDS * s));
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
        v.push(col(format!("R{r}_REFR"), rf.kind.scalar(route).bytes(), rf.kind.reference_len() * slots[rf.entity]));
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
        // the cost's partials in the root's CACC: W or narrower
        v.push(col(format!("C{c}_PART"), 8, parts(n)));
    }
    (v, structure, values)
}

/// Where every column sits in the block: each column on a 256-byte
/// boundary. The structure columns end at `structure_bytes`, the values
/// columns at `host_bytes`: what the host writes. The kernels write the
/// rest. `table`: the offsets as the kernels take them, the layout table.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Layout {
    columns: Vec<Column>,
    offsets: Vec<usize>,
    table: Vec<u64>,
    structure_bytes: usize,
    host_bytes: usize,
    bytes: usize,
}

impl Layout {
    fn new((columns, structure, values): (Vec<Column>, usize, usize)) -> Self {
        let align = |b: usize| b.div_ceil(256) * 256;
        let mut at = 0;
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
        let table = offsets.iter().map(|&o| o as u64).collect();
        Layout { columns, offsets, table, structure_bytes, host_bytes, bytes: at }
    }

    /// The `#define`s of the table's size and slots, for the kernels.
    fn defines(&self) -> String {
        let slots: String = self.columns.iter().enumerate().map(|(k, c)| format!("#define ARAEL_COL_{} {k}\n", c.name)).collect();
        format!("#define ARAEL_COLS {}\n{slots}", self.columns.len())
    }

    /// The layout table as a launch's argument.
    fn arg<B>(&self) -> Arg<'_, B> {
        Arg::Bytes(arael_gpu::bytemuck::cast_slice(&self.table))
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
    /// Zeroes the cost (W), and H and the gradient (V, the root's scalar).
    zero: K,
    zero_values: K,
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
/// once, its values written into the staging ([`Staging`]) and uploaded per
/// solve; the
/// parameter vector, the targets and the cost are the caller's buffers:
/// the parameters, H and the gradient in the assembly's [`Values`], the
/// cost in its [`Wide`] scalar.
pub struct DeviceAssembly<T, D: Device> {
    dev: Arc<D>,
    wide: Wide,
    /// The root's [`Values`] on the route of `wide`.
    route: Values,
    shape: DeviceShape,
    slots: Vec<usize>,
    cross_slots: Vec<usize>,
    counts: Vec<usize>,
    /// H's values in the pattern laid out against.
    nh: usize,
    /// Whether an assembly zeroes H first: not when every block of H is
    /// one stored tile ([`stores`]).
    zero_h: bool,
    /// Per referenced field, per slot: the global index of the step's first
    /// live parameter, [`FIXED`] when the step is fixed.
    ref_index: Vec<Vec<u32>>,
    /// Per constraint, in walk order: which instances its guard keeps.
    keep: Vec<Vec<bool>>,
    layout: Layout,
    offset: HashMap<String, usize>,
    block: D::Block,
    /// The values range of the block on the host: what a refresh rewrites.
    staged: HostBuf<D::Host>,
    kernels: Kernels<D::Kernel>,
    generation: u64,
    /// The fewest significand bits of the model's leaves at the last
    /// refresh ([`DeviceLeaf::BITS`]).
    leaf_bits: u32,
    _t: core::marker::PhantomData<fn() -> T>,
}

fn launch_n(n: usize) -> Launch {
    Launch::linear((n as u32).max(1), THREADS)
}

/// Per cross slot of one cross array, given the block of H each slot's
/// tile lies in ([`Tiles::cross_block`], [`NO_BLOCK`] without a tile), the
/// first slot whose tile is in the same block: the array's slots between
/// the same two entities share their tile, the only one of the array in
/// that block. A slot without a tile (a side fixed throughout, or no
/// instance names it) is its own. `first` holds [`NO_BLOCK`] per block of
/// H, and holds it again on return. On up to `threads` threads of the
/// worker pool, each taking at least `min` slots: each block's first slot
/// as an atomic minimum over ranges of slots, then each slot's owner read
/// by ranges.
fn tile_owners(blocks: &[u32], first: &mut [u32], (threads, min): (usize, usize)) -> Vec<u32> {
    use core::sync::atomic::{AtomicU32, Ordering::Relaxed};
    // An AtomicU32 has the layout of a u32, and `first` is borrowed
    // through this alone until it is reset.
    let at: &[AtomicU32] = unsafe { &*(first as *mut [u32] as *const [AtomicU32]) };
    let parts = threads.min(blocks.len() / min.max(1)).max(1);
    let step = blocks.len().div_ceil(parts).max(1);
    let over = |f: &(dyn Fn(core::ops::Range<usize>) + Sync)| {
        if parts == 1 {
            f(0..blocks.len());
        } else {
            arael_faer::pool::run(parts, &|p| f(p * step..((p + 1) * step).min(blocks.len())));
        }
    };
    over(&|range| {
        for s in range {
            let b = blocks[s];
            if b != NO_BLOCK {
                at[b as usize].fetch_min(s as u32, Relaxed);
            }
        }
    });
    let mut owner = vec![0u32; blocks.len()];
    {
        let out = arael_faer::pool::SharedMut(owner.as_mut_ptr());
        over(&|range| {
            for s in range {
                let b = blocks[s];
                let o = if b == NO_BLOCK { s as u32 } else { at[b as usize].load(Relaxed) };
                // each range writes its own slots only
                unsafe { *out.ptr().add(s) = o };
            }
        });
    }
    // the blocks this array touched back to NO_BLOCK, for the next array
    over(&|range| {
        for s in range {
            let b = blocks[s];
            if b != NO_BLOCK {
                at[b as usize].store(NO_BLOCK, Relaxed);
            }
        }
    });
    owner
}

/// [`tile_owners`] of every cross array of `t`, against H's `nblocks`.
fn all_tile_owners(t: &Tiles, nblocks: usize, par: (usize, usize)) -> Vec<Vec<u32>> {
    let mut first = vec![NO_BLOCK; nblocks];
    t.cross_block.iter().map(|b| tile_owners(b, &mut first, par)).collect()
}

/// [`Tiles::cross_block`] of a slot without a tile.
const NO_BLOCK: u32 = u32::MAX;

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
pub(crate) const PAR_MIN: usize = 8 * 1024;

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

/// `n` copies of `value`, written by ranges of at least `min` on up to
/// `threads` threads of the worker pool: a large array's pages are first
/// touched by the threads that fill it, not by the caller alone.
fn filled(n: usize, value: u32, (threads, min): (usize, usize)) -> Vec<u32> {
    let parts = threads.min(n / min.max(1)).max(1);
    if parts == 1 {
        return vec![value; n];
    }
    let mut v: Vec<u32> = Vec::with_capacity(n);
    let mut chunks: Vec<&mut [core::mem::MaybeUninit<u32>]> = v.spare_capacity_mut()[..n].chunks_mut(n.div_ceil(parts)).collect();
    arael_faer::pool::run_over(&mut chunks, |_, c| {
        for x in c.iter_mut() {
            x.write(value);
        }
    });
    drop(chunks);
    // every element of the first `n` was written above
    unsafe { v.set_len(n) };
    v
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
) -> Tiles {
    use crate::store::{device_word, tile_extent};
    let sym = r.symbolic();
    // a diagonal block square, a cross tile between whole tiles whole
    assert!(sym.parts().0 == sym.parts().1, "H's rows and columns are cut alike");
    let extent = |e: usize, s: usize| {
        let n = shape.entities[e];
        tile_extent(&st.idx[e][s * n..(s + 1) * n])
    };
    let mut entity: Vec<Vec<u32>> = Vec::with_capacity(slots.len());
    // per entity table, per slot: its block row of H and its stride word
    // (`NO_BLOCK` and 0 where it is fixed throughout), what a cross tile
    // between whole tiles is found from
    let mut ekey: Vec<Vec<u32>> = Vec::with_capacity(slots.len());
    let mut whole = Vec::with_capacity(slots.len());
    for (e, &ns) in slots.iter().enumerate() {
        // per range its records and keys, and whether each of its tiles is
        // its block
        let parts: Vec<(Vec<u32>, Vec<u32>, bool)> = par_ranges(ns, threads, min, r, |range, r| {
            let mut all = true;
            let mut recs = Vec::with_capacity(3 * range.len());
            let mut keys = Vec::with_capacity(2 * range.len());
            for s in range {
                let (start, live) = extent(e, s);
                if live == 0 {
                    recs.extend_from_slice(&[0, 0, 0]);
                    keys.extend_from_slice(&[NO_BLOCK, 0]);
                    continue;
                }
                let (base, stride, b) = r.resolve_tile_block(start as usize, start as usize);
                all &= stride == live && base == sym.val_range(b).start;
                recs.extend_from_slice(&[device_word(base), device_word(stride), start]);
                keys.extend_from_slice(&[sym.blk_row(b) as u32, device_word(stride)]);
            }
            (recs, keys, all)
        });
        whole.push(parts.iter().all(|p| p.2));
        entity.push(parts.iter().map(|p| &p.0[..]).collect::<Vec<_>>().concat());
        ekey.push(parts.iter().map(|p| &p.1[..]).collect::<Vec<_>>().concat());
    }
    let mut cross: Vec<Vec<u32>> = cross_slots.iter().map(|&n| vec![0u32; 3 * n]).collect();
    let mut cross_block: Vec<Vec<u32>> = cross_slots.iter().map(|&n| filled(n, NO_BLOCK, (threads, min))).collect();
    // per stored block of H its block row
    let blk_rows: &[arael_faer::SparseIndex] = sym.parts().3;
    let mut done: Vec<Vec<bool>> = cross_slots.iter().map(|&n| vec![false; n]).collect();
    for (c, k) in shape.constraints.iter().enumerate() {
        let (w, rows) = (k.words(), &st.rows[c]);
        let nrows = rows.len() / w;
        // The rows of one slot name the same two entities, so a slot's
        // tile is its first row's. Rows that name a pair's slots in
        // non-decreasing order (the order a walk of the model gives them)
        // have each slot's rows in one run: its first row starts the run,
        // and a range of rows starts the runs of a range of slots, which
        // it writes alone. Per pair, whether its slots come so.
        let ns_sides = k.sides.len();
        let sorted: Vec<bool> = par_ranges(nrows, threads, min, r, |range, _| {
            let mut ok = vec![true; k.pairs.len()];
            for row in range.start.max(1)..range.end {
                let (prev, this) = (&rows[(row - 1) * w + ns_sides..row * w], &rows[row * w + ns_sides..(row + 1) * w]);
                for (o, (a, b)) in ok.iter_mut().zip(prev.iter().zip(this)) {
                    *o &= a <= b;
                }
            }
            ok
        })
        .into_iter()
        .fold(vec![true; k.pairs.len()], |all, ok| all.iter().zip(&ok).map(|(a, b)| *a && *b).collect());
        for (p, &(i, j, x)) in k.pairs.iter().enumerate() {
            let (ea, eb) = (k.sides[i], k.sides[j]);
            let q = ns_sides + p;
            let ns = cross_slots[x];
            // The tile between the slots `a` and `b` of the pair's sides,
            // its record and its block of H; `None` where a side is fixed
            // throughout. Between whole tiles the tile is its block, found
            // from the two block rows alone; else through the resolver,
            // from the sides' first live indices.
            let direct = whole[ea] && whole[eb];
            let tile = |a: usize, b: usize, r: &mut Resolver<'_>| -> Option<([u32; 3], u32)> {
                if direct {
                    let (ka, kb) = (&ekey[ea][2 * a..2 * a + 2], &ekey[eb][2 * b..2 * b + 2]);
                    if ka[0] == NO_BLOCK || kb[0] == NO_BLOCK {
                        return None;
                    }
                    let flip = kb[0] < ka[0];
                    let (row, col) = if flip { (kb, ka) } else { (ka, kb) };
                    // the block of row `row[0]` among the column's, which
                    // are in row order
                    let range = sym.col_range(col[0] as usize);
                    let bl = range.start + blk_rows[range].binary_search(&row[0]).expect("a pair's block is outside H's structure");
                    Some(([device_word(sym.val_range(bl).start), row[1], flip as u32], bl as u32))
                } else {
                    let (ta, tb) = (&entity[ea][3 * a..3 * a + 3], &entity[eb][3 * b..3 * b + 3]);
                    if ta[1] == 0 || tb[1] == 0 {
                        return None;
                    }
                    let (ra, rb) = (ta[2], tb[2]);
                    let flip = rb < ra;
                    let (base, stride, bl) = if flip { r.resolve_tile_block(rb as usize, ra as usize) } else { r.resolve_tile_block(ra as usize, rb as usize) };
                    Some(([device_word(base), device_word(stride), flip as u32], bl as u32))
                }
            };
            let slot = |row: usize| rows[row * w + q] as usize;
            if sorted[p] {
                let parts = threads.min(nrows / min.max(1)).max(1);
                let step = nrows.div_ceil(parts).max(1);
                let row_cuts: Vec<usize> = (0..=parts).map(|t| (t * step).min(nrows)).collect();
                // a range starts the runs of the slots above the one its
                // preceding row names
                let mut slot_cuts: Vec<usize> = row_cuts[..parts].iter().map(|&r0| if r0 == 0 { 0 } else { slot(r0 - 1) + 1 }).collect();
                slot_cuts.push(ns);
                let rec_cuts: Vec<usize> = slot_cuts.iter().map(|&s| 3 * s).collect();
                type Task<'t, 'r> = (core::ops::Range<usize>, usize, &'t mut [u32], &'t mut [u32], &'t mut [bool], Resolver<'r>);
                let mut tasks: Vec<Task<'_, '_>> = arael_faer::cut_slices(&mut cross[x], &rec_cuts)
                    .into_iter()
                    .zip(arael_faer::cut_slices(&mut cross_block[x], &slot_cuts))
                    .zip(arael_faer::cut_slices(&mut done[x], &slot_cuts))
                    .enumerate()
                    .map(|(t, ((rec, blk), dn))| (row_cuts[t]..row_cuts[t + 1], slot_cuts[t], rec, blk, dn, r.clone()))
                    .collect();
                let lookup = |_: usize, (range, s0, rec, blk, dn, r): &mut Task<'_, '_>| {
                    for row in range.clone() {
                        let s = slot(row);
                        if row > 0 && slot(row - 1) == s {
                            continue;
                        }
                        let ls = s - *s0;
                        if dn[ls] {
                            continue;
                        }
                        dn[ls] = true;
                        let at = row * w;
                        if let Some((t, bl)) = tile(rows[at + i] as usize, rows[at + j] as usize, r) {
                            rec[3 * ls..3 * ls + 3].copy_from_slice(&t);
                            blk[ls] = bl;
                        }
                    }
                };
                if let [one] = &mut tasks[..] {
                    lookup(0, one);
                } else {
                    arael_faer::pool::run_over(&mut tasks, lookup);
                }
            } else {
                // per cross slot not yet done, the first row naming it
                let first: Vec<core::sync::atomic::AtomicU32> = (0..ns).map(|_| core::sync::atomic::AtomicU32::new(u32::MAX)).collect();
                {
                    let done = &done[x];
                    par_ranges(nrows, threads, min, r, |range, _| {
                        for row in range {
                            let s = slot(row);
                            if !done[s] {
                                first[s].fetch_min(row as u32, core::sync::atomic::Ordering::Relaxed);
                            }
                        }
                    });
                }
                // per range of slots, its slices of the records, blocks and marks
                let parts = threads.min(ns / min.max(1)).max(1);
                let step = ns.div_ceil(parts).max(1);
                let mut tasks: Vec<(usize, &mut [u32], &mut [u32], &mut [bool], Resolver<'_>)> = cross[x]
                    .chunks_mut(3 * step)
                    .zip(cross_block[x].chunks_mut(step))
                    .zip(done[x].chunks_mut(step))
                    .enumerate()
                    .map(|(t, ((rec, blk), dn))| (t * step, rec, blk, dn, r.clone()))
                    .collect();
                let lookup = |_: usize, (s0, rec, blk, dn, r): &mut (usize, &mut [u32], &mut [u32], &mut [bool], Resolver<'_>)| {
                    for ls in 0..blk.len() {
                        let row = first[*s0 + ls].load(core::sync::atomic::Ordering::Relaxed);
                        if row == u32::MAX {
                            continue;
                        }
                        dn[ls] = true;
                        let at = row as usize * w;
                        if let Some((t, bl)) = tile(rows[at + i] as usize, rows[at + j] as usize, r) {
                            rec[3 * ls..3 * ls + 3].copy_from_slice(&t);
                            blk[ls] = bl;
                        }
                    }
                };
                if let [one] = &mut tasks[..] {
                    lookup(0, one);
                } else {
                    arael_faer::pool::run_over(&mut tasks, lookup);
                }
            }
        }
    }
    Tiles { entity, cross, cross_block, whole }
}

/// The tiles of a structure in H ([`tiles`]): per entity table and per
/// cross array the slots' records ([`TILE_WORDS`] each), and per entity
/// table whether each of its tiles is all of its block of H: then a cross
/// tile between two such tables is all of its block too, H's rows and
/// columns being cut alike.
#[derive(Debug, PartialEq, Eq)]
struct Tiles {
    entity: Vec<Vec<u32>>,
    cross: Vec<Vec<u32>>,
    /// Per cross array, per slot, the block of H its tile lies in
    /// ([`NO_BLOCK`] without a tile).
    cross_block: Vec<Vec<u32>>,
    whole: Vec<bool>,
}

/// A constraint's instances grouped ([`by_slot`], `(ptr, inst)` each): per
/// side by its entity slot, per pair by the owner of its cross slot's tile
/// ([`tile_owners`]), so a tile has one summing thread per entry.
#[derive(Debug, PartialEq)]
struct Groups {
    sides: Vec<(Vec<u32>, Vec<u32>)>,
    pairs: Vec<(Vec<u32>, Vec<u32>)>,
}

/// Every constraint's [`Groups`], its [`by_slot`] passes on up to
/// `threads` threads of arael's worker pool, each thread taking at least
/// `min` instances.
fn groups(shape: &DeviceShape, st: &DeviceStructure, slots: &[usize], cross_slots: &[usize], owners: &[Vec<u32>], (threads, min): (usize, usize)) -> Vec<Groups> {
    // a pass per (constraint, side), then per (constraint, pair): word q
    // of the constraint's instances
    let passes: Vec<(usize, usize)> = shape.constraints.iter().enumerate().flat_map(|(c, k)| (0..k.sides.len() + k.pairs.len()).map(move |q| (c, q))).collect();
    let pass = |(c, q): (usize, usize)| {
        let (k, rows) = (&shape.constraints[c], &st.rows[c]);
        let of = word(rows, k.words(), q);
        match q.checked_sub(k.sides.len()) {
            None => by_slot(of, slots[k.sides[q]]),
            Some(p) => {
                let x = k.pairs[p].2;
                by_slot(of.map(|s| owners[x][s as usize]), cross_slots[x])
            }
        }
    };
    let instances: usize = st.rows.iter().zip(&shape.constraints).map(|(r, k)| r.len() / k.words()).sum();
    let tasks = threads.min(instances / min.max(1)).clamp(1, passes.len().max(1));
    // task t takes the passes t, t + tasks, t + 2 tasks, ...
    let mut out: Vec<Vec<(Vec<u32>, Vec<u32>)>> = vec![Vec::new(); tasks];
    let run = |t: usize, out: &mut Vec<(Vec<u32>, Vec<u32>)>| *out = passes.iter().skip(t).step_by(tasks).map(|&j| pass(j)).collect();
    if let [one] = &mut out[..] {
        run(0, one);
    } else {
        arael_faer::pool::run_over(&mut out, run);
    }
    let mut done = (0..passes.len()).map(|j| core::mem::take(&mut out[j % tasks][j / tasks]));
    shape
        .constraints
        .iter()
        .map(|k| Groups { sides: done.by_ref().take(k.sides.len()).collect(), pairs: done.by_ref().take(k.pairs.len()).collect() })
        .collect()
}

/// Which tiles of H their kernels store, and whether the assembly zeroes H
/// first. An entity table one constraint side alone reads has that side as
/// the only writer of its entities' diagonal tiles; a cross array one
/// constraint pair alone uses, between two tables no other array joins,
/// has that pair as the only writer of its tiles. Those kernels store
/// their sums: [`TILE_STORE`] in the stride word of their tables' records
/// (a slot without instances writes nothing, so the bit on its record is
/// never read). H needs no zeroing when such tiles, each all of its block
/// ([`Tiles::whole`]) and with instances (`groups`), are every one of the
/// pattern's `nblocks` blocks. Returns whether H is zeroed first.
///
/// ```ignore
/// // bundle adjustment: one side reads the cameras, one the points, one
/// // pair joins them: every tile stored, H not zeroed;
/// // a pose graph: both sides of an edge read the poses: their tiles
/// // added to, H zeroed
/// ```
fn stores(shape: &DeviceShape, groups: &[Groups], t: &mut Tiles, nblocks: usize) -> bool {
    let (ne, nx) = (t.entity.len(), t.cross.len());
    // per entity table its (constraint, side) readers, per cross array its
    // (constraint, pair) users and the tables it joins
    let mut readers: Vec<Vec<(usize, usize)>> = vec![Vec::new(); ne];
    let mut users: Vec<Vec<(usize, usize)>> = vec![Vec::new(); nx];
    let mut joins: Vec<Option<(usize, usize)>> = vec![None; nx];
    for (c, k) in shape.constraints.iter().enumerate() {
        for (i, &e) in k.sides.iter().enumerate() {
            readers[e].push((c, i));
        }
        for (p, &(i, j, x)) in k.pairs.iter().enumerate() {
            users[x].push((c, p));
            let (ea, eb) = (k.sides[i], k.sides[j]);
            joins[x] = Some((ea.min(eb), ea.max(eb)));
        }
    }
    let has = |ptr: &[u32], s: usize| ptr[s + 1] > ptr[s];
    let mut covered = 0usize;
    for e in 0..ne {
        if let [(c, i)] = readers[e][..] {
            let ptr = &groups[c].sides[i].0;
            for (s, rec) in t.entity[e].chunks_exact_mut(TILE_WORDS).enumerate().filter(|(_, rec)| rec[1] != 0) {
                rec[1] |= TILE_STORE;
                covered += (t.whole[e] && has(ptr, s)) as usize;
            }
        }
    }
    for x in 0..nx {
        let (Some((ea, eb)), [(c, p)]) = (joins[x], &users[x][..]) else { continue };
        if ea == eb || joins.iter().filter(|&&j| j == joins[x]).count() > 1 {
            continue;
        }
        let (whole, ptr) = (t.whole[ea] && t.whole[eb], &groups[*c].pairs[*p].0);
        for (s, rec) in t.cross[x].chunks_exact_mut(TILE_WORDS).enumerate().filter(|(_, rec)| rec[1] != 0) {
            rec[1] |= TILE_STORE;
            covered += (whole && has(ptr, s)) as usize;
        }
    }
    covered != nblocks
}

/// The block's structure columns into `image` (the structure range): the
/// entity tables' indices and tiles, the referenced fields' step offsets,
/// the cross tiles, the instances' slots and their [`Groups`], each column
/// (or range of at least `min` instances of an instance column) a job on
/// up to `threads` threads of the worker pool. Returns per referenced
/// field, per slot, the step's first live global index (the key of its
/// reference).
#[allow(clippy::too_many_arguments)]
fn write_structure(
    image: &mut [u8],
    offset: &HashMap<String, usize>,
    shape: &DeviceShape,
    st: &DeviceStructure,
    slots: &[usize],
    t: &Tiles,
    groups: &[Groups],
    (threads, min): (usize, usize),
) -> Vec<Vec<u32>> {
    use arael_gpu::bytemuck::cast_slice;
    let mut ref_index = Vec::with_capacity(shape.references.len());
    let mut ats: Vec<Vec<u32>> = Vec::with_capacity(shape.references.len());
    for rf in &shape.references {
        let n = shape.entities[rf.entity];
        let at: Vec<u32> = (0..slots[rf.entity]).map(|s| (s * n + rf.step) as u32).collect();
        // a step's key: its first live parameter (a TransformParam may hold
        // one half of its step fixed)
        let idx = &st.idx[rf.entity];
        ref_index.push(
            at.iter()
                .map(|&a| idx[a as usize..a as usize + rf.kind.step_len()].iter().copied().find(|&g| g != FIXED).unwrap_or(FIXED))
                .collect(),
        );
        ats.push(at);
    }
    // every column a job writing its own bytes of the image: a copy, or a
    // range of one word of a constraint's instances
    enum Fill<'a> {
        Copy(&'a [u8]),
        Word { rows: &'a [u32], w: usize, q: usize, from: usize },
    }
    let mut jobs: Vec<(usize, usize, Fill<'_>)> = Vec::new();
    for (e, (idx, tile)) in st.idx.iter().zip(&t.entity).enumerate() {
        jobs.push((offset[&format!("E{e}_IDX")], idx.len() * 4, Fill::Copy(cast_slice(idx))));
        jobs.push((offset[&format!("E{e}_TILE")], tile.len() * 4, Fill::Copy(cast_slice(tile))));
    }
    for (r, at) in ats.iter().enumerate() {
        jobs.push((offset[&format!("R{r}_AT")], at.len() * 4, Fill::Copy(cast_slice(at))));
    }
    for (x, tile) in t.cross.iter().enumerate() {
        jobs.push((offset[&format!("X{x}_TILE")], tile.len() * 4, Fill::Copy(cast_slice(tile))));
    }
    for ((c, k), g) in shape.constraints.iter().enumerate().zip(groups) {
        let (w, rows) = (k.words(), &st.rows[c]);
        let n = rows.len() / w;
        let parts = threads.min(n / min.max(1)).max(1);
        let step = n.div_ceil(parts).max(1);
        for (q, name) in word_names(k).enumerate() {
            let o = offset[&format!("C{c}_{name}")];
            for from in (0..n).step_by(step) {
                let len = step.min(n - from);
                jobs.push((o + 4 * from, 4 * len, Fill::Word { rows, w, q, from }));
            }
        }
        for (i, (ptr, list)) in g.sides.iter().enumerate() {
            jobs.push((offset[&format!("C{c}_OFF{i}")], ptr.len() * 4, Fill::Copy(cast_slice(ptr))));
            jobs.push((offset[&format!("C{c}_INST{i}")], list.len() * 4, Fill::Copy(cast_slice(list))));
        }
        for (p, (ptr, list)) in g.pairs.iter().enumerate() {
            jobs.push((offset[&format!("C{c}_XOFF{p}")], ptr.len() * 4, Fill::Copy(cast_slice(ptr))));
            jobs.push((offset[&format!("C{c}_XINST{p}")], list.len() * 4, Fill::Copy(cast_slice(list))));
        }
    }
    // each job its own slice of the image, cut in offset order
    jobs.retain(|j| j.1 > 0);
    jobs.sort_by_key(|j| j.0);
    let mut tasks: Vec<(&mut [u8], Fill<'_>)> = Vec::with_capacity(jobs.len());
    let (mut rest, mut at) = (&mut image[..], 0usize);
    for (o, len, fill) in jobs {
        let (_, tail) = core::mem::take(&mut rest).split_at_mut(o - at);
        let (dst, tail) = tail.split_at_mut(len);
        tasks.push((dst, fill));
        rest = tail;
        at = o + len;
    }
    let write = |_: usize, (dst, fill): &mut (&mut [u8], Fill<'_>)| match fill {
        Fill::Copy(bytes) => dst.copy_from_slice(bytes),
        Fill::Word { rows, w, q, from } => {
            for (b, rec) in dst.as_chunks_mut::<4>().0.iter_mut().zip(rows[*from * *w..].chunks_exact(*w)) {
                *b = rec[*q].to_le_bytes();
            }
        }
    };
    if threads <= 1 || tasks.len() <= 1 {
        tasks.iter_mut().for_each(|t| write(0, t));
    } else {
        arael_faer::pool::run_over(&mut tasks, write);
    }
    ref_index
}

impl<T: crate::utils::Float + 'static, D: Device> DeviceAssembly<T, D> {
    /// Lay `root` out on `dev` against the tiled pattern `pattern` (its
    /// tiles looked up on up to `threads` threads), its sums in `wide`,
    /// compile its kernels, and upload its structure and values from
    /// `staging`. `structure` is the root's structure when the caller has
    /// walked it already ([`structure_of`]), else it is walked here. The
    /// model's working copies must be current (its guards, fixed values and
    /// references are read).
    pub fn new(
        dev: &Arc<D>,
        root: &dyn DeviceRoot<T>,
        pattern: &arael_faer::bsc::SymbolicSparseBlockColMat<arael_faer::SparseIndex>,
        threads: usize,
        wide: Wide,
        staging: Staging,
        structure: Option<DeviceStructure>,
    ) -> Result<Self, GpuError> {
        let ts = core::mem::size_of::<T>();
        assert!(ts == 4 || ts == 8, "the device assembly takes an f32 or an f64 model");
        let shape = root.device_shape();
        let st = structure.unwrap_or_else(|| structure_of(root));
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
        let mut t = tiles(&shape, &st, &slots, &cross_slots, &Resolver::new(pattern), (threads, PAR_MIN));
        let owners = all_tile_owners(&t, pattern.nblocks(), (threads, PAR_MIN));
        let groups = groups(&shape, &st, &slots, &cross_slots, &owners, (threads, PAR_MIN));
        let zero_h = stores(&shape, &groups, &mut t, pattern.nblocks());
        let route = Values::of(ts, wide);
        let layout = Layout::new(columns(&shape, &slots, &cross_slots, &counts, ts, route));
        let scalar = if ts == 8 { "double" } else { "float" };
        let defines = format!("{}#define SCALAR {scalar}\n{}", route.defines(), layout.defines());
        let source = format!("{WIDE}\n{}\n{}", arael_gpu::ASSEMBLY, root.device_cpp());
        if let Some(dir) = std::env::var_os(KERNEL_DUMP_VAR) {
            dump_unit(dir.as_ref(), root.device_name(), &dev.arch().target, wide, &dev.unit(&source, &defines))?;
        }
        // every entry point in one lookup of the source, in the order the
        // kernels take them below
        let mut entries: Vec<String> = ["k_gather_params", "k_zero_wide", "k_zero_values", "k_zero_steps"].map(String::from).to_vec();
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
            zero_values: k(),
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
        // the structure: the structure columns, copied once
        let mut host = HostBuf::new(&**dev, staging, layout.structure_bytes)?;
        let ref_index = write_structure(host.bytes_mut(&**dev)?, &offset, &shape, &st, &slots, &t, &groups, (threads, PAR_MIN));
        host.upload(&**dev, &mut block, 0, layout.structure_bytes)?;
        drop(host);
        let staged = HostBuf::new(&**dev, staging, layout.host_bytes - layout.structure_bytes)?;
        let mut asm = DeviceAssembly {
            dev: dev.clone(),
            wide,
            route,
            shape,
            slots,
            cross_slots,
            counts,
            nh: pattern.val_count(),
            zero_h,
            ref_index,
            keep: st.keep,
            layout,
            offset,
            block,
            staged,
            kernels,
            generation: 0,
            leaf_bits: 53,
            _t: core::marker::PhantomData,
        };
        asm.dev.describe(&asm.block, "image", &|| asm.sections());
        // the structure's working copies, released off this thread
        arael_faer::pool::drop_later((st.idx, st.rows, t, groups, owners));
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
        let _scope = Scope::new(&*self.dev, "image");
        let base = self.layout.structure_bytes;
        let at = |name: &str| self.offset[name] - base;
        let n_ent = self.shape.entities.len();
        let mut w = DeviceWriter {
            bytes: self.staged.bytes_mut(&*self.dev)?,
            fixed: (0..n_ent).map(|e| (at(&format!("E{e}_FIXED")), self.shape.entities[e], self.slots[e])).collect(),
            data: (0..n_ent).map(|e| (at(&format!("E{e}_DATA")), self.shape.entity_data[e])).collect(),
            refr: self
                .shape
                .references
                .iter()
                .enumerate()
                .map(|(r, rf)| (at(&format!("R{r}_REFR")), rf.kind.reference_len(), self.slots[rf.entity], rf.kind.scalar(self.route)))
                .collect(),
            root: (at("ROOT"), self.shape.root_data),
            inst: self
                .shape
                .constraints
                .iter()
                .enumerate()
                .map(|(c, k)| ((0..k.data).map(|j| at(&format!("C{c}_D{j}"))).collect(), self.counts[c]))
                .collect(),
            bits: 53,
            _t: core::marker::PhantomData,
        };
        root.device_values(&self.keep, &mut w);
        self.leaf_bits = w.bits;
        let len = self.layout.host_bytes - base;
        self.staged.upload(&*self.dev, &mut self.block, base, len)
    }

    /// The block's sections, for the tools that inspect a device's work
    /// ([`Device::describe`]): every column with its group (`structure`:
    /// written once; `values`: rewritten per solve; `computed`: written by
    /// the kernels), stored type and meaning.
    fn sections(&self) -> Vec<Section> {
        let l = &self.layout;
        let (f, w) = (if core::mem::size_of::<T>() == 4 { "f32" } else { "f64" }, self.wide.name());
        let mut v = Vec::with_capacity(l.columns.len());
        for (c, &at) in l.columns.iter().zip(&l.offsets) {
            let group = if at < l.structure_bytes {
                "structure"
            } else if at < l.host_bytes {
                "values"
            } else {
                "computed"
            };
            // E3_IDX: the table letter, its number, the column's kind
            let kind = c.name.split_once('_').map_or(c.name.as_str(), |(_, k)| k);
            let kind = kind.trim_end_matches(|ch: char| ch.is_ascii_digit());
            let number = |name: &str| name[1..].split('_').next().and_then(|n| n.parse::<usize>().ok());
            let (ty, role, doc): (&str, &str, &str) = match (c.name.as_bytes()[0], kind) {
                (b'E', "IDX") => ("u32", "", "per entity parameter: its global index in x, FIXED when not optimized"),
                (b'E', "TILE") => ("u32", "", "per entity: its diagonal tile in H's values (base, stride with TILE_STORE when its kernel stores it, first live index)"),
                (b'E', "FIXED") => (f, "F", "per entity parameter: its value where fixed"),
                (b'E', "DATA") => (f, "F", "per entity: the record of its data the constraints read (F values)"),
                (b'E', "EV") => (f, "F", "per entity parameter: its value at the current x, gathered"),
                (b'R', "AT") => ("u32", "", "per entity: where the referenced field's step starts among its parameters"),
                (b'R', "REFR") => (number(&c.name).map_or(w, |r| self.shape.references[r].kind.scalar(self.route).ty()), "", "per entity: the referenced field's reference (rotation or quaternion)"),
                (b'R', "VAL") => (f, "F", "per entity: the referenced field's composed values"),
                (b'R', "VALW") => (w, "W", "per entity: the referenced field's composed values in the wide scalar"),
                (b'R', "DER") => (f, "F", "per entity: the composed values' derivatives by the step"),
                (b'X', "TILE") => ("u32", "", "per cross block: its tile in H's values (base, stride with TILE_STORE when its kernel stores it, flip: 1 when the second side's span lies on the tile's rows)"),
                (b'C', "S") => ("u32", "", "per instance: this side's entity slot"),
                (b'C', "OFF") => ("u32", "", "per entity slot of this side: where its instances start in INST"),
                (b'C', "INST") => ("u32", "", "the instances grouped by this side's entity"),
                (b'C', "XOFF") => ("u32", "", "per cross tile slot: where its instances start in XINST"),
                (b'C', "XINST") => ("u32", "", "the instances grouped by this pair's cross tile"),
                (b'C', "D") => (f, "F", "per instance: one datum of the constraint"),
                (b'C', "REC") => (f, "F", "per instance: its record, the residual rows and their partials"),
                (b'C', "PART") => ("bytes", "CACC", "per block of 256 instances: its part of the cost, in the root's CACC (V, or W with cost_f64), an 8-byte slot each"),
                (b'R', "ROOT") => (f, "F", "the root's record of data the constraints read (F values)"),
                _ if c.elem == 4 && group == "structure" => ("u32", "", "structure words"),
                _ => ("bytes", "", ""),
            };
            // a record of F values counts by its values
            let ts = core::mem::size_of::<T>();
            let elem = if role == "F" && c.elem > ts { ts } else { c.elem };
            let record = if kind == "TILE" { TILE_WORDS } else { 1 };
            v.push(Section::new(c.name.clone(), at, c.elem * c.len, elem, ty).role(role).group(group).doc(doc).record(record));
        }
        v
    }

    /// A counter of the block's layout: it changes whenever the block or its
    /// offsets change. A captured graph of the launches replays only under
    /// the generation it was captured in.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// The fewest significand bits of the model's leaves at the last
    /// refresh ([`DeviceLeaf::BITS`]): below the root scalar's, a leaf of a
    /// narrower type, whose rows the host computes in that type while the
    /// device computes every row in the root's.
    pub fn leaf_bits(&self) -> u32 {
        self.leaf_bits
    }

    /// The scalar of the cost and the references.
    pub fn wide(&self) -> Wide {
        self.wide
    }

    /// The scalar of the parameters, H and the gradient: the root's, as the
    /// host keeps them (f32 for an f32 root, else [`wide`](Self::wide)).
    pub fn values(&self) -> Values {
        self.route
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

    /// `n` values of H or the gradient zeroed, in [`values`](Self::values).
    fn zero_values(&self, t: Target<'_, D::Block>, n: usize) -> Result<(), GpuError> {
        self.run(&self.kernels.zero_values, n.div_ceil(4), &[Arg::Block(t.0, t.1), Arg::U32(n as u32)])
    }

    /// H's values (`nh` of them, the pattern's) zeroed, unless every block
    /// of H is a tile its kernel stores ([`stores`]).
    fn zero_h(&self, h: Target<'_, D::Block>, nh: usize) -> Result<(), GpuError> {
        assert_eq!(nh, self.nh, "an H of {nh} values, the assembly laid out against {}", self.nh);
        if self.zero_h { self.zero_values(h, nh) } else { Ok(()) }
    }

    /// The entities' parameter values and the referenced fields' values at
    /// the parameters `x`; with `derivatives`, their derivatives too.
    fn gather(&self, x: Target<'_, D::Block>, derivatives: bool) -> Result<(), GpuError> {
        let _scope = Scope::new(&*self.dev, "gather");
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
    /// at `h`, every one written: zeroed first unless every block is a tile
    /// its kernel stores) and the cost (one value at `cost`) at the
    /// parameters `x`, the gradient and the cost zeroed first: the
    /// parameters, the gradient and H in
    /// [`values`](Self::values), the cost in [`wide`](Self::wide); nothing
    /// read back.
    pub fn assemble_at(
        &self,
        x: Target<'_, D::Block>,
        g: Target<'_, D::Block>,
        ng: usize,
        h: Target<'_, D::Block>,
        nh: usize,
        cost: Target<'_, D::Block>,
    ) -> Result<(), GpuError> {
        let _scope = Scope::new(&*self.dev, "assemble");
        {
            let _scope = Scope::new(&*self.dev, "zero");
            self.zero_values(g, ng)?;
            self.zero_h(h, nh)?;
            self.zero(cost, 1)?;
        }
        self.gather(x, true)?;
        let _constraints = Scope::new(&*self.dev, "constraints");
        let (b, l) = (Arg::Block(&self.block, 0), self.layout.arg());
        let (h, g, cost) = (Arg::Block(h.0, h.1), Arg::Block(g.0, g.1), Arg::Block(cost.0, cost.1));
        for (c, &n) in self.counts.iter().enumerate() {
            if n == 0 {
                continue;
            }
            let (k, shape) = (&self.kernels.constraints[c], &self.shape.constraints[c]);
            self.run(&k.gh, n, &[b, l, Arg::U32(n as u32)])?;
            self.dev.launch(&k.csum, Launch::linear(THREADS, THREADS), &[b, l, cost, Arg::U32(parts(n) as u32)])?;
            // per side its entities' tiles and gradients, per pair its cross
            // tiles, a thread per entry
            for (&e, kr) in shape.sides.iter().zip(&k.sides) {
                let (ne, se) = (self.shape.entities[e], self.slots[e]);
                self.run(kr, se * (ne * (ne + 1) / 2 + ne), &[b, l, h, g, Arg::U32(se as u32)])?;
            }
            for (&(i, j, x), kx) in shape.pairs.iter().zip(&k.pairs) {
                let (ni, nj, sx) = (self.shape.entities[shape.sides[i]], self.shape.entities[shape.sides[j]], self.cross_slots[x]);
                self.run(kx, sx * ni * nj, &[b, l, h, Arg::U32(sx as u32)])?;
            }
        }
        Ok(())
    }

    /// The cost (one value at `cost`) at the parameters `x`, zeroed first;
    /// the referenced fields' derivatives are not computed. Nothing read back.
    pub fn cost_at(&self, x: Target<'_, D::Block>, cost: Target<'_, D::Block>) -> Result<(), GpuError> {
        let _scope = Scope::new(&*self.dev, "cost");
        self.zero(cost, 1)?;
        self.gather(x, false)?;
        let _constraints = Scope::new(&*self.dev, "constraints");
        let (b, l, cost) = (Arg::Block(&self.block, 0), self.layout.arg(), Arg::Block(cost.0, cost.1));
        for (c, &n) in self.counts.iter().enumerate() {
            if n == 0 {
                continue;
            }
            let k = &self.kernels.constraints[c];
            self.run(&k.cost, n, &[b, l, Arg::U32(n as u32)])?;
            self.dev.launch(&k.csum, Launch::linear(THREADS, THREADS), &[b, l, cost, Arg::U32(parts(n) as u32)])?;
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
            let scalar = rf.kind.scalar(self.route);
            let mut bytes = vec![0u8; scalar.bytes() * n];
            self.dev.read(&self.block, self.offset[&format!("R{r}_REFR")], &mut bytes)?;
            let mut v = vec![0f64; n];
            scalar.decode(&bytes, &mut v);
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

/// Assemble `problem` at `x` on the host and on `dev` in `wide` against one
/// tiled pattern (the reduced route's), and compare (for tests). The device
/// is laid out from the model at `x0`, as a solve lays it out at its start
/// and assembles at the points it moves to: what it computes at `x` must
/// not lean on the model's caches at `x0`.
pub fn device_check<T, D>(problem: &mut dyn crate::simple_lm::LmProblemInternals<T>, x0: &[T], x: &[T], dev: &Arc<D>, wide: Wide) -> DeviceCheck
where
    T: crate::utils::Float + 'static,
    D: Device,
{
    try_device_check(problem, x0, x, dev, wide).unwrap_or_else(|panic| std::panic::resume_unwind(panic))
}

/// [`device_check`], a panic of the host's assembly at `x` (a model that
/// panics on its data) caught and returned, the model put back at `x0`.
fn try_device_check<T, D>(
    problem: &mut dyn crate::simple_lm::LmProblemInternals<T>,
    x0: &[T],
    x: &[T],
    dev: &Arc<D>,
    wide: Wide,
) -> Result<DeviceCheck, Box<dyn std::any::Any + Send>>
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
    let host_cost = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        problem.calc_grad_hessian_sparse_indexed(x, &mut grad, &mut vals, &positions, &mut ctx)
    }));
    // the device laid out from the model at x0
    problem.update_params(x0);
    let host_cost = host_cost?.to_f64().unwrap();
    let root = problem.device_root().expect("the root has a device assembly");
    // H's cells from the device structure: every one a block store's
    let (store_blocks, device_blocks) = {
        let mut stored: Vec<(u32, u32)> = cells.iter().map(|&(i, j)| (i as u32, j as u32)).collect();
        stored.sort_unstable();
        stored.dedup();
        let mut device = Vec::new();
        cells_of(&root.device_shape(), &structure_of(root), &mut device, (1, usize::MAX));
        device.sort_unstable();
        device.dedup();
        let foreign: Vec<&(u32, u32)> = device.iter().filter(|c| stored.binary_search(c).is_err()).collect();
        assert!(foreign.is_empty(), "the device structure's cells {foreign:?} are no block store's");
        (stored.len(), device.len())
    };
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(8);
    let asm = DeviceAssembly::<T, D>::new(dev, root, &hsym, threads, wide, Staging::Pinned, None)
        .expect("lay the root out on the device");
    // the cost (wide), x, the gradient and the values (the root's values)
    // in one block of the test's own
    let values = asm.values();
    let (at_cost, at_x) = (0, 8);
    let at_g = at_x + n * values.bytes();
    let at_h = at_g + n * values.bytes();
    let bytes = at_h + nv * values.bytes();
    let mut work = dev.alloc(bytes).expect("allocate the targets");
    // every target NaN first: the assembly writes all of them (H's values
    // stored or zeroed, the gradient and the cost zeroed)
    dev.write(&mut work, 0, &vec![0xffu8; bytes]).expect("fill the targets");
    let x64: Vec<f64> = x.iter().map(|v| v.to_f64().unwrap()).collect();
    dev.write(&mut work, at_x, &values.encode(&x64)).expect("upload x");
    asm.assemble_at((&work, at_x), (&work, at_g), n, (&work, at_h), nv, (&work, at_cost)).expect("assemble on the device");
    let read = |at: usize, len: usize, values: Values| -> Vec<f64> {
        let mut bytes = vec![0u8; values.bytes() * len];
        dev.read(&work, at, &mut bytes).expect("read a target");
        let mut v = vec![0f64; len];
        values.decode(&bytes, &mut v);
        v
    };
    let (gd, hd, cost) = (read(at_g, n, values), read(at_h, nv, values), read(at_cost, 1, wide.into()));
    let err = |dev: &[f64], cpu: &[T]| -> f64 {
        let max = cpu.iter().fold(0.0f64, |m, v| m.max(v.to_f64().unwrap().abs()));
        let diff = dev.iter().zip(cpu).fold(0.0f64, |m, (d, c)| m.max((d - c.to_f64().unwrap()).abs()));
        if max > 0.0 { diff / max } else { diff }
    };
    let chain = asm.counts.iter().copied().max().unwrap_or(0);
    Ok(DeviceCheck { host_cost, device_cost: cost[0], grad_err: err(&gd, &grad), hess_err: err(&hd, &vals), values: nv, leaf_bits: asm.leaf_bits(), chain, store_blocks, device_blocks })
}

/// The environment variable that turns on the device check: set (to
/// anything but `0`), every solve checks the device assembly against the
/// host's. The first solve of each root type with a device assembly also
/// assembles that root on the device, near its starting point (each
/// parameter moved by up to three thousandths of its size, or absolutely
/// below 1), in both wide scalars ([`device_check`]), and panics where the
/// device's cost, gradient or Hessian values differ from the host's beyond
/// [`check_tolerance`] of the model's narrowest scalar (the root's, or a
/// leaf's: an `f32` entity in an `f64` root has its rows computed in `f32`
/// on the host). A test suite run with it set checks the device assembly
/// of every model it solves:
///
/// ```sh
/// ARAEL_DEVICE_CHECK=1 cargo test --features gpu-cuda
/// ```
pub const DEVICE_CHECK_VAR: &str = "ARAEL_DEVICE_CHECK";

/// The environment variable naming a file the device check appends a line
/// to per root type it meets: checked, with its largest differences, or
/// without a device assembly, and why.
pub const DEVICE_CHECK_LOG_VAR: &str = "ARAEL_DEVICE_CHECK_LOG";

/// The device check's tolerance for a model whose narrowest scalar (the
/// root's, or a leaf's, [`DeviceCheck::leaf_bits`]) carries `bits`
/// significand bits: relative to the host's cost, and to the largest host
/// entry of the gradient and of the Hessian. Where it is `f32` the host
/// computes and sums in it, the device in its own scalars.
pub fn check_tolerance(bits: u32) -> f64 {
    match bits {
        53.. => 1e-10,
        48.. => 1e-8,
        _ => 1e-4,
    }
}

/// The device check's tolerance for the cost, relative to the host's:
/// [`check_tolerance`] of the narrowest scalar's `bits`, plus the rounding
/// of the host's cost sum, a chain of up to `chain` adds in the root's
/// scalar of `root_bits` significand bits (`chain * 2^-root_bits`, the
/// bound for positive terms; the device sums in trees):
///
/// ```ignore
/// check_cost_tolerance(24, 24, 204_472) // 1e-4 + 1.2e-2
/// check_cost_tolerance(53, 53, 204_472) // 1e-10 + 2.3e-11
/// ```
pub fn check_cost_tolerance(bits: u32, root_bits: u32, chain: usize) -> f64 {
    check_tolerance(bits) + chain as f64 * (-(root_bits as f64)).exp2()
}

/// The device the check runs on: CUDA when it is compiled in, else HIP,
/// else Metal, else the CPU device.
#[cfg(feature = "gpu-cuda")]
type CheckDevice = arael_gpu::cuda::CudaDevice;
#[cfg(all(feature = "gpu-hip", not(feature = "gpu-cuda")))]
type CheckDevice = arael_gpu::hip::HipDevice;
#[cfg(all(gpu_metal, not(any(feature = "gpu-cuda", feature = "gpu-hip"))))]
type CheckDevice = arael_gpu::metal::MetalDevice;
#[cfg(all(feature = "gpu-cpu", not(any(feature = "gpu-cuda", feature = "gpu-hip", gpu_metal))))]
type CheckDevice = arael_gpu::cpu::CpuDevice;

/// The check's device, made at its first use.
#[cfg(any(feature = "gpu-cuda", feature = "gpu-hip", feature = "gpu-cpu", gpu_metal))]
fn check_device() -> &'static Arc<CheckDevice> {
    static DEV: std::sync::OnceLock<Arc<CheckDevice>> = std::sync::OnceLock::new();
    DEV.get_or_init(|| Arc::new(CheckDevice::new(0).unwrap_or_else(|e| panic!("{DEVICE_CHECK_VAR}: no device: {e}"))))
}

/// Whether the check's device computes in f64 (Metal does not).
#[cfg(any(feature = "gpu-cuda", feature = "gpu-hip", feature = "gpu-cpu", gpu_metal))]
fn check_device_has_f64() -> bool {
    check_device().arch().has_f64
}

#[cfg(not(any(feature = "gpu-cuda", feature = "gpu-hip", feature = "gpu-cpu", gpu_metal)))]
fn check_device_has_f64() -> bool {
    true
}

/// The device check ([`DEVICE_CHECK_VAR`]) at a solve's start and at a
/// root's assembly entry points: `problem`, of the root type `P`, at `x0`.
/// Reads the variable once per process.
#[doc(hidden)]
pub fn check_solve<T, P>(problem: &mut P, x0: &[T])
where
    T: crate::utils::Float + 'static,
    P: crate::simple_lm::LmProblemInternals<T>,
{
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os(DEVICE_CHECK_VAR).is_some_and(|v| v != "0")) {
        return;
    }
    // the root types checked, one check at a time (tests solve on several
    // threads); a root is marked when its check ends, so a check a panic
    // cuts short runs again at the next call, and so does one whose CPU
    // assembly panics on the model's data (a test's deliberately bad
    // model): there is nothing to check against, and the solve goes on to
    // meet the panic itself, or not on the device's route
    static SEEN: std::sync::Mutex<Vec<&'static str>> = std::sync::Mutex::new(Vec::new());
    let name = std::any::type_name::<P>();
    let mut seen = SEEN.lock().unwrap_or_else(|p| p.into_inner());
    if seen.contains(&name) {
        return;
    }
    let line = match problem.device_unsupported() {
        Some(why) => {
            seen.push(name);
            format!("{name}: no device assembly: {why}")
        }
        // an f64 root assembles on the CPU where the device has no f64
        None if core::mem::size_of::<T>() == 8 && !check_device_has_f64() => {
            seen.push(name);
            format!("{name} (f64): no device assembly here: the device has no f64")
        }
        None => {
            let precision = if core::mem::size_of::<T>() == 8 { "f64" } else { "f32" };
            match check_root(problem, x0, name) {
                Ok(found) => {
                    seen.push(name);
                    format!("{name} ({precision}): checked: {found}")
                }
                Err(why) => format!("{name} ({precision}): not checked here: the CPU assembly panics: {why}"),
            }
        }
    };
    drop(seen);
    if let Some(path) = std::env::var_os(DEVICE_CHECK_LOG_VAR) {
        use std::io::Write;
        // one write per line: test processes append to one log at once
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .and_then(|mut f| f.write_all(format!("{line}\n").as_bytes()))
            .unwrap_or_else(|e| panic!("{DEVICE_CHECK_LOG_VAR}: cannot append to {}: {e}", std::path::Path::new(&path).display()));
    }
}

/// [`check_solve`]'s check of the root `name`: its largest differences,
/// or a panic where one is beyond [`check_tolerance`]; `Err` with the
/// message of a panic of the host's assembly.
#[cfg(any(feature = "gpu-cuda", feature = "gpu-hip", feature = "gpu-cpu", gpu_metal))]
fn check_root<T, P>(problem: &mut P, x0: &[T], name: &str) -> Result<String, String>
where
    T: crate::utils::Float + 'static,
    P: crate::simple_lm::LmProblemInternals<T>,
{
    let dev = check_device();
    // moved off the start by up to three thousandths of each parameter's
    // size, or absolutely below 1: the residuals then stand well above the
    // rounding of the terms they are made of, and the re-centred
    // parameters' steps leave zero
    let x1: Vec<T> = x0
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let v = v.to_f64().unwrap();
            T::from(v + 1e-3 * ((i % 7) as f64 - 3.0) * v.abs().max(1.0)).unwrap()
        })
        .collect();
    let root_bits = if core::mem::size_of::<T>() == 8 { 53 } else { 24 };
    let mut worst = [0.0f64; 3];
    let mut bits = root_bits;
    let mut cost_tol = 0.0f64;
    // both wide scalars, df32 alone on a device without f64
    let wides: &[Wide] = if dev.arch().has_f64 { &[Wide::F64, Wide::Df32] } else { &[Wide::Df32] };
    for &wide in wides {
        let chk = try_device_check(problem, x0, &x1, dev, wide).map_err(|panic| {
            panic.downcast_ref::<&str>().map(|m| m.to_string()).or_else(|| panic.downcast_ref::<String>().cloned()).unwrap_or_else(|| "a panic without a message".into())
        })?;
        bits = root_bits.min(chk.leaf_bits);
        let tol = check_tolerance(bits);
        cost_tol = check_cost_tolerance(bits, root_bits, chk.chain);
        let cost = (chk.device_cost - chk.host_cost).abs() / chk.host_cost.abs().max(f64::MIN_POSITIVE);
        if !(cost <= cost_tol && chk.grad_err <= tol && chk.hess_err <= tol) {
            panic!(
                "{DEVICE_CHECK_VAR}: `{name}` assembles on the device ({wide:?} sums) other than on the host: \
                 cost {} against {} ({cost:.1e}, tolerance {cost_tol:.1e}), gradient {:.1e}, Hessian {:.1e}, tolerance {tol:.0e}",
                chk.device_cost, chk.host_cost, chk.grad_err, chk.hess_err
            );
        }
        for (w, e) in worst.iter_mut().zip([cost, chk.grad_err, chk.hess_err]) {
            *w = w.max(e);
        }
    }
    let narrow = if bits < root_bits { format!(" (leaves of {bits} significand bits, tolerance {:.0e})", check_tolerance(bits)) } else { String::new() };
    Ok(format!("cost {:.1e} (tolerance {cost_tol:.1e}), gradient {:.1e}, Hessian {:.1e}{narrow}", worst[0], worst[1], worst[2]))
}

#[cfg(not(any(feature = "gpu-cuda", feature = "gpu-hip", feature = "gpu-cpu", gpu_metal)))]
fn check_root<T, P>(_problem: &mut P, _x0: &[T], name: &str) -> Result<String, String>
where
    T: crate::utils::Float + 'static,
    P: crate::simple_lm::LmProblemInternals<T>,
{
    panic!("{DEVICE_CHECK_VAR}: `{name}`: the check needs a device runtime (feature gpu-cuda, gpu-hip, gpu-metal or gpu-cpu)")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The block holds the columns alone, from byte 0 on 256-byte
    /// boundaries; the layout table, each column's offset, is the kernels'
    /// launch parameter, its size and slots their defines.
    #[test]
    fn the_layout_table_is_a_parameter_not_in_the_block() {
        let col = |name: &str, elem, len| Column { name: name.into(), elem, len };
        let layout = Layout::new((vec![col("E0_IDX", 4, 10), col("E0_VAL", 8, 100), col("C0_REC", 4, 1)], 1, 2));
        assert_eq!(layout.offsets, [0, 256, 1280]);
        assert_eq!(layout.table, [0, 256, 1280]);
        assert_eq!((layout.structure_bytes, layout.host_bytes, layout.bytes), (256, 1280, 1536));
        assert_eq!(layout.defines(), "#define ARAEL_COLS 3\n#define ARAEL_COL_E0_IDX 0\n#define ARAEL_COL_E0_VAL 1\n#define ARAEL_COL_C0_REC 2\n");
        match layout.arg::<()>() {
            Arg::Bytes(b) => assert_eq!(b, arael_gpu::bytemuck::cast_slice::<u64, u8>(&[0, 256, 1280])),
            _ => panic!("the table is a bytes argument"),
        }
    }

    /// A kernel dump lands at `<dir>/<root>.<target>.<wide>.cu`, the
    /// directory made, and a dump that cannot be written is an error.
    #[test]
    fn a_kernel_dump_is_named_by_root_target_and_wide_scalar() {
        let dir = std::env::temp_dir().join(format!("arael-kernel-dump-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dump_unit(&dir.join("sub"), "Scene", "sm_89", Wide::Df32, "// unit\n").unwrap();
        assert_eq!(path, dir.join("sub").join("Scene.sm_89.df32.cu"));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "// unit\n");
        // a file where the directory would be
        let blocked = dir.join("file");
        std::fs::write(&blocked, "").unwrap();
        assert!(matches!(dump_unit(&blocked, "Scene", "sm_89", Wide::F64, ""), Err(GpuError::Dump(_))));
    }

    /// A cross slot's owner is the first slot whose tile is in the same
    /// block; a slot without a tile is its own.
    #[test]
    fn a_slot_is_owned_by_the_first_slot_of_its_tile() {
        // the block per slot: 4, 2, 4, none, 2, 0, 4, none
        let blocks = [4, 2, 4, NO_BLOCK, 2, 0, 4, NO_BLOCK];
        let mut first = vec![NO_BLOCK; 5];
        // on one thread, and on threads in ranges of one slot and of three;
        // `first` back to NO_BLOCK after each
        for par in [(1, usize::MAX), (8, 1), (3, 3)] {
            assert_eq!(tile_owners(&blocks, &mut first, par), [0, 1, 0, 3, 1, 5, 0, 7], "{par:?}");
            assert!(first.iter().all(|&f| f == NO_BLOCK), "{par:?}");
        }
    }

    /// The tiles looked up, the instances grouped, H's cells listed and the
    /// structure written on several threads equal the ones on one: two
    /// entity tables (three and two parameters a slot, some fixed
    /// throughout), a cross constraint between them, every row a lookup,
    /// and a one-sided constraint on the second table.
    #[test]
    fn the_layout_on_several_threads_is_the_layout_on_one() {
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
            constraints: vec![
                DeviceConstraint { data: 0, rows: 1, sides: vec![0, 1], pairs: vec![(0, 1, 0)] },
                DeviceConstraint { data: 0, rows: 1, sides: vec![1], pairs: Vec::new() },
            ],
            entity_data: vec![0, 0],
            root_data: 0,
        };
        let rows: Vec<u32> = (0..nrows).flat_map(|k| [cols[0][k], cols[1][k], cols[2][k]]).collect();
        let priors: Vec<u32> = (0..40).map(|k| k * 3 % s1 as u32).collect();
        let st = DeviceStructure { idx: vec![idx0, idx1], rows: vec![rows, priors], keep: vec![vec![true; nrows], vec![true; 40]] };
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
        // every slot's tile is the one the resolver gives its two entities
        // from their first live indices
        let agrees = |t: &Tiles, r: &Resolver<'_>| {
            let mut resolve = r.clone();
            for k in 0..nrows {
                let (a, b) = (cols[0][k] as usize, cols[1][k] as usize);
                let rec = &t.cross[0][3 * k..3 * k + 3];
                if a == 7 || b == 11 {
                    assert_eq!((rec, t.cross_block[0][k]), (&[0u32, 0, 0][..], NO_BLOCK), "slot {k}: a side fixed throughout");
                    continue;
                }
                let (base, stride, bl) = resolve.resolve_tile_block(3 * a, 3 * s0 + 2 * b);
                assert_eq!((rec, t.cross_block[0][k]), (&[base as u32, stride as u32, 0][..], bl as u32), "slot {k}");
            }
        };
        agrees(&one, &r);
        // the first two entities in one block of H: the first table's
        // tiles are not whole blocks, and its pairs' tiles are parts of
        // theirs
        let mut part2 = vec![0usize, 6];
        for k in 2..s0 + s1 {
            part2.push(part2[k - 1] + if k < s0 { 3 } else { 2 });
        }
        let coords: Vec<(usize, usize)> = (0..s0)
            .map(|a| (3 * a, 3 * a))
            .chain((0..s1).map(|b| (3 * s0 + 2 * b, 3 * s0 + 2 * b)))
            .chain((0..nrows).map(|k| (3 * cols[0][k] as usize, 3 * s0 + 2 * cols[1][k] as usize)))
            .collect();
        let p2 = crate::simple_lm::partition_idx(&part2);
        let pattern2 = arael_faer::bsc::SymbolicSparseBlockColMat::covering(p2.clone(), p2, coords.len(), |k| coords[k]);
        let r2 = Resolver::new(&pattern2);
        let loose = tiles(&shape, &st, &[s0, s1], &[nrows], &r2, (1, usize::MAX));
        assert_eq!(loose.whole, [false, true]);
        assert_eq!(tiles(&shape, &st, &[s0, s1], &[nrows], &r2, (4, 1)), loose);
        agrees(&loose, &r2);
        // the rows in another order name their slots out of order: the same tiles
        let reversed: Vec<u32> = (0..nrows).rev().flat_map(|k| [cols[0][k], cols[1][k], cols[2][k]]).collect();
        let st_rev = DeviceStructure { idx: st.idx.clone(), rows: vec![reversed, st.rows[1].clone()], keep: st.keep.clone() };
        for par in [(1, usize::MAX), (4, 1)] {
            assert_eq!(tiles(&shape, &st_rev, &[s0, s1], &[nrows], &r, par), one, "{par:?}");
        }
        // three rows a slot, in runs that straddle the threads' ranges (43
        // rows each), and the same rows reversed
        let run_rows = |order: &mut dyn Iterator<Item = usize>| -> Vec<u32> { order.flat_map(|k| [cols[0][k / 3], cols[1][k / 3], (k / 3) as u32]).collect() };
        let st_runs = DeviceStructure { idx: st.idx.clone(), rows: vec![run_rows(&mut (0..nrows)), st.rows[1].clone()], keep: st.keep.clone() };
        let st_runs_rev = DeviceStructure { idx: st.idx.clone(), rows: vec![run_rows(&mut (0..nrows).rev()), st.rows[1].clone()], keep: st.keep.clone() };
        let runs = tiles(&shape, &st_runs, &[s0, s1], &[nrows / 3], &r, (1, usize::MAX));
        assert_eq!(runs.cross[0][..], one.cross[0][..nrows], "slot k of the runs is slot k of the rows");
        assert_eq!(tiles(&shape, &st_runs, &[s0, s1], &[nrows / 3], &r, (7, 1)), runs);
        for par in [(1, usize::MAX), (7, 1)] {
            assert_eq!(tiles(&shape, &st_runs_rev, &[s0, s1], &[nrows / 3], &r, par), runs, "{par:?}");
        }
        // the fixed entities have no tile, the others one each, all of its block
        assert_eq!(one.entity[0][21..24], [0, 0, 0]);
        assert!(one.entity[0][..21].chunks(3).all(|t| t[1] > 0));
        assert_eq!(one.whole, [true, true]);
        // four passes: on two threads two each, on eight one each
        // a cross slot's tile is in its block, and the slots of a block share it
        for (t, b) in one.cross.iter().zip(&one.cross_block) {
            for (s, &blk) in b.iter().enumerate() {
                if blk == NO_BLOCK {
                    assert_eq!(t[3 * s + 1], 0, "slot {s} has no tile");
                } else {
                    assert!(pattern.val_range(blk as usize).contains(&(t[3 * s] as usize)), "slot {s}: its tile in its block");
                }
            }
        }
        let owners = all_tile_owners(&one, pattern.nblocks(), (1, usize::MAX));
        let grouped = |threads| groups(&shape, &st, &[s0, s1], &[nrows], &owners, (threads, 1));
        let want = grouped(1);
        assert_eq!(want[1].sides[0].1.len(), 40);
        assert_eq!(grouped(2), want);
        assert_eq!(grouped(8), want);
        // H's cells from the structure, every one a block of the pattern
        let cells_with = |par| {
            let mut out = Vec::new();
            cells_of(&shape, &st, &mut out, par);
            out
        };
        let cells_one = cells_with((1, usize::MAX));
        assert_eq!(cells_with((4, 7)), cells_one);
        assert_eq!(cells_one.len(), s0 + s1 - 2 + nrows);
        assert!(cells_one.iter().all(|&(i, j)| pattern.block_of_scalar(i as usize, j as usize).is_some()));
        // the structure columns, a job per column and in ranges of 7 instances
        let mut t = tiles(&shape, &st, &[s0, s1], &[nrows], &r, (1, usize::MAX));
        stores(&shape, &want, &mut t, pattern.nblocks());
        let layout = Layout::new(columns(&shape, &[s0, s1], &[nrows], &[nrows, 40], 8, Values::of(8, Wide::F64)));
        let offset = layout.index();
        let write = |par| {
            let mut image = vec![0u8; layout.structure_bytes];
            let refs = write_structure(&mut image, &offset, &shape, &st, &[s0, s1], &t, &want, par);
            (image, refs)
        };
        let image_one = write((1, usize::MAX));
        assert_eq!(write((4, 7)), image_one);
        let slots0 = &image_one.0[offset["C0_S0"]..offset["C0_S0"] + 4 * nrows];
        assert!(slots0.as_chunks::<4>().0.iter().map(|b| u32::from_le_bytes(*b)).eq(cols[0].iter().copied()));
    }

    /// Which kernels store their tiles, and whether H is zeroed first. A
    /// camera-point-like scene (two entity tables of three and two
    /// parameters a slot, one constraint between them, its pairs repeating
    /// in cross slots of their own) stores every tile and zeroes nothing.
    /// A prior on the second table's entities takes that table's stores
    /// back; a second constraint joining the two tables through an array
    /// of its own takes back both tables' (a second reader each) and the
    /// first array's (its tiles may be the second's); a partly fixed entity
    /// (its tile part of its block) or an entity in no instance (its block
    /// written by none) keep the stores and zero H.
    #[test]
    fn kernels_store_the_tiles_they_alone_write() {
        // per table and the first cross array whether each slot is stored,
        // and whether H is zeroed
        let case = |prior: bool, second: bool, partial: bool, unused: bool| {
            let (s0, s1, n) = (20usize, 30usize, 120usize);
            let mut idx0: Vec<u32> = (0..3 * s0 as u32).collect();
            if partial {
                idx0[30] = FIXED;
            }
            let idx1: Vec<u32> = (0..2 * s1 as u32).map(|i| 3 * s0 as u32 + i).collect();
            let named = if unused { 19 } else { 20 };
            let pairs: Vec<(u32, u32)> = (0..n as u32).map(|k| (k * 7 % named, k * 11 % s1 as u32)).collect();
            let rows: Vec<u32> = pairs.iter().enumerate().flat_map(|(k, &(a, b))| [a, b, k as u32]).collect();
            let mut constraints = vec![DeviceConstraint { data: 0, rows: 1, sides: vec![0, 1], pairs: vec![(0, 1, 0)] }];
            let mut all_rows = vec![rows];
            if prior {
                constraints.push(DeviceConstraint { data: 0, rows: 1, sides: vec![1], pairs: vec![] });
                all_rows.push((0..10).collect());
            }
            if second {
                constraints.push(DeviceConstraint { data: 0, rows: 1, sides: vec![0, 1], pairs: vec![(0, 1, 1)] });
                all_rows.push(pairs[..5].iter().enumerate().flat_map(|(k, &(a, b))| [a, b, k as u32]).collect());
            }
            let keep = all_rows.iter().zip(&constraints).map(|(r, k)| vec![true; r.len() / k.words()]).collect();
            let shape = DeviceShape { entities: vec![3, 2], references: Vec::new(), crosses: 2, constraints, entity_data: vec![0, 0], root_data: 0 };
            let st = DeviceStructure { idx: vec![idx0, idx1], rows: all_rows, keep };
            let mut part = vec![0usize];
            for k in 0..s0 + s1 {
                part.push(part[k] + if k < s0 { 3 } else { 2 });
            }
            let mut cells: Vec<(usize, usize)> = (0..s0 + s1).map(|b| (b, b)).collect();
            cells.extend(pairs.iter().map(|&(a, b)| (a as usize, s0 + b as usize)));
            cells.sort_unstable();
            cells.dedup();
            let p = crate::simple_lm::partition_idx(&part);
            let pattern = arael_faer::bsc::SymbolicSparseBlockColMat::covering(p.clone(), p, cells.len(), |k| (part[cells[k].0], part[cells[k].1]));
            let (slots, cross_slots) = ([s0, s1], [n, if second { 5 } else { 0 }]);
            let mut t = tiles(&shape, &st, &slots, &cross_slots, &Resolver::new(&pattern), (1, usize::MAX));
            let owners = all_tile_owners(&t, pattern.nblocks(), (1, usize::MAX));
            let zero = stores(&shape, &groups(&shape, &st, &slots, &cross_slots, &owners, (1, usize::MAX)), &mut t, pattern.nblocks());
            let stored = |recs: &[u32]| recs.chunks(3).map(|r| r[1] & TILE_STORE != 0).collect::<Vec<bool>>();
            (stored(&t.entity[0]), stored(&t.entity[1]), stored(&t.cross[0]), zero)
        };
        let all = |v: &[bool], want: bool| v.iter().all(|&b| b == want);
        let (e0, e1, x0, zero) = case(false, false, false, false);
        assert!(all(&e0, true) && all(&e1, true) && all(&x0, true) && !zero, "bundle adjustment");
        let (e0, e1, x0, zero) = case(true, false, false, false);
        assert!(all(&e0, true) && all(&e1, false) && all(&x0, true) && zero, "a prior");
        let (e0, e1, x0, zero) = case(false, true, false, false);
        assert!(all(&e0, false) && all(&e1, false) && all(&x0, false) && zero, "a second array");
        let (e0, e1, x0, zero) = case(false, false, true, false);
        assert!(all(&e0, true) && all(&e1, true) && all(&x0, true) && zero, "a partly fixed entity");
        let (e0, e1, x0, zero) = case(false, false, false, true);
        assert!(all(&e0, true) && all(&e1, true) && all(&x0, true) && zero, "an entity in no instance");
    }
}
