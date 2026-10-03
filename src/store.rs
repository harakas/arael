//! The block store: what the generated sweeps and the solver share.
//!
//! Inner API. Everything here is public so that the code `#[arael(root)]`
//! generates can reach it and so that it can be read, and none of it is
//! stable: it changes with the macro. A model declares its blocks with
//! the markers in [`crate::model`](mod@crate::model) (`SelfBlock`,
//! `CrossBlock`) and never
//! names anything in this module; a solver reaches it through
//! [`Context`].
//!
//! Per root the macro generates a `<Root>Blocks` struct holding one
//! [`SelfBlockArray`] per entity container with a self block and one
//! [`CrossBlockArray`] per cross-block field, plus the COO list. A
//! [`Context`] owns one such store per thread,
//! type-erased behind [`BlockStore`], and a [`Cut`] says which slot range
//! of each walk a store covers. A sweep takes its store, walks the ranges
//! [`Leaves`] gives it, and writes through the arrays; the join scatters
//! the stores into the Hessian with the positions a [`HessianBinder`]
//! bound.

use std::any::Any;
use std::time::Duration;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

use arael_faer::{value_index, ValueIndex};
use crate::threads::{Context, ParTiming};

// ---------------------------------------------------------------------------
// Walking part of a container
// ---------------------------------------------------------------------------

/// A container a sweep can walk part of: the five shapes a constraint
/// or entity collection takes, each able to yield a slot range.
pub trait Leaves {
    type Item;
    /// Number of index slots (an arena counts its free slots too).
    fn count(&self) -> usize;
    /// Every live item with its slot index, in container order. The cut
    /// is built over this: a weight per unit, and the slot it falls on.
    fn each(&self, f: impl FnMut(u32, &Self::Item));
    /// The live items of the slot range `[lo, hi)`, in container order.
    ///
    /// Slots, not positions, so a range means the same thing whether or
    /// not the container has holes, and `hi` clamps to [`count`](Self::count).
    /// This is how a sweep walks part of a container: `0 .. count()` is the
    /// whole of it and has to cost what the plain iterator costs.
    fn range_iter(&self, lo: u32, hi: u32) -> impl Iterator<Item = &Self::Item>;
}

impl<T> Leaves for std::vec::Vec<T> {
    type Item = T;
    fn count(&self) -> usize { self.len() }
    fn each(&self, mut f: impl FnMut(u32, &T)) {
        for (i, x) in self.iter().enumerate() { f(i as u32, x); }
    }
    #[inline(always)]
    fn range_iter(&self, lo: u32, hi: u32) -> impl Iterator<Item = &T> {
        let hi = (hi as usize).min(self.len());
        let lo = (lo as usize).min(hi);
        self[lo..hi].iter()
    }
}

impl<T> Leaves for crate::refs::Vec<T> {
    type Item = T;
    fn count(&self) -> usize { self.len() }
    fn each(&self, mut f: impl FnMut(u32, &T)) {
        for (i, x) in self.iter().enumerate() { f(i as u32, x); }
    }
    #[inline(always)]
    fn range_iter(&self, lo: u32, hi: u32) -> impl Iterator<Item = &T> {
        self.range(lo, hi)
    }
}

impl<T> Leaves for crate::refs::Deque<T> {
    type Item = T;
    fn count(&self) -> usize { self.len() }
    fn each(&self, mut f: impl FnMut(u32, &T)) {
        for (i, x) in self.iter().enumerate() { f(i as u32, x); }
    }
    #[inline(always)]
    fn range_iter(&self, lo: u32, hi: u32) -> impl Iterator<Item = &T> {
        self.range(lo, hi)
    }
}

impl<T> Leaves for crate::refs::Arena<T> {
    type Item = T;
    fn count(&self) -> usize { self.slot_count() }
    fn each(&self, mut f: impl FnMut(u32, &T)) {
        for (r, x) in self.iter_refs() { f(r.index(), x); }
    }
    #[inline(always)]
    fn range_iter(&self, lo: u32, hi: u32) -> impl Iterator<Item = &T> {
        self.iter_range(lo, hi)
    }
}

impl<T> Leaves for Option<T> {
    type Item = T;
    fn count(&self) -> usize { 1 }
    fn each(&self, mut f: impl FnMut(u32, &T)) {
        if let Some(x) = self { f(0, x); }
    }
    #[inline(always)]
    fn range_iter(&self, lo: u32, hi: u32) -> impl Iterator<Item = &T> {
        // One slot, taken or not.
        let live = lo == 0 && hi > 0;
        self.as_ref().filter(|_| live).into_iter()
    }
}


// ---------------------------------------------------------------------------
// The clock
// ---------------------------------------------------------------------------

/// A clock the timing turns on: `start` reads the time only then, so a
/// solve without timing makes no clock calls at all.
#[derive(Clone, Copy, Debug)]
pub struct Clock {
    on: bool,
}

impl Clock {
    /// A clock that reads the time only when `on`.
    #[inline]
    pub fn new(on: bool) -> Self { Clock { on } }

    #[inline]
    pub fn start(self) -> Option<Instant> {
        if self.on { Some(Instant::now()) } else { None }
    }

    #[inline]
    pub fn stop(self, started: Option<Instant>) -> Duration {
        started.map_or(Duration::ZERO, |t| t.elapsed())
    }
}


// ---------------------------------------------------------------------------
// The stores a context holds
// ---------------------------------------------------------------------------

/// A root's generated block store, held type-erased so the context does
/// not name it. `Send + Sync` so a root may keep a context in a field
/// and stay `Sync` for the dispatch.
pub(crate) trait AnyStore: Any + Send + Sync {
    fn store_any(&self) -> &dyn Any;
    fn store_any_mut(&mut self) -> &mut dyn Any;
    fn store_clone(&self) -> Box<dyn AnyStore>;
    /// The footprints of the first `n` stores held.
    fn store_footprints(&self, n: usize) -> std::vec::Vec<StoreFootprint>;
}

/// What a store holds: its block counts and the bytes of its arrays.
/// Summed over a solve's stores and set against the whole model, it
/// says how much the split stores duplicate: a self block every store
/// whose range touches its entity claims, a cross block exactly one
/// store holds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StoreFootprint {
    /// Self blocks held (entities with a tile and a gradient stash).
    pub self_blocks: usize,
    /// Cross blocks held.
    pub cross_blocks: usize,
    /// COO entries held after the last sweep.
    pub coo_entries: usize,
    /// Bytes of the arrays: tiles, stashes, indices, maps and the COO list.
    pub bytes: usize,
}

impl StoreFootprint {
    /// Add another store's footprint to this one.
    pub fn add(&mut self, other: StoreFootprint) {
        self.self_blocks += other.self_blocks;
        self.cross_blocks += other.cross_blocks;
        self.coo_entries += other.coo_entries;
        self.bytes += other.bytes;
    }

    /// The sum over stores.
    pub fn sum<'a>(stores: impl IntoIterator<Item = &'a StoreFootprint>) -> StoreFootprint {
        let mut total = StoreFootprint::default();
        for s in stores { total.add(*s); }
        total
    }
}

/// What a generated block store is: the macro implements it on the type
/// it emits per root.
pub trait BlockStore: Clone + Default + Send + Sync + 'static {
    /// The store's block counts and bytes.
    fn footprint(&self) -> StoreFootprint;
}

impl<S: BlockStore> AnyStore for S {
    fn store_any(&self) -> &dyn Any { self }
    fn store_any_mut(&mut self) -> &mut dyn Any { self }
    fn store_clone(&self) -> Box<dyn AnyStore> { Box::new(self.clone()) }
    fn store_footprints(&self, _n: usize) -> std::vec::Vec<StoreFootprint> { vec![self.footprint()] }
}

// The context's payload is the whole list, not one store. Spelled out
// rather than blanket: a blanket over `Any + Send + Sync + Clone` also
// covers `&Box<dyn AnyStore>`, and method calls stop derefing to the
// trait object.
impl<S: BlockStore> AnyStore for std::vec::Vec<S> {
    fn store_any(&self) -> &dyn Any { self }
    fn store_any_mut(&mut self) -> &mut dyn Any { self }
    fn store_clone(&self) -> Box<dyn AnyStore> { Box::new(self.clone()) }
    fn store_footprints(&self, n: usize) -> std::vec::Vec<StoreFootprint> {
        self[..n.min(self.len())].iter().map(|s| s.footprint()).collect()
    }
}


// ---------------------------------------------------------------------------
// The context's stores
// ---------------------------------------------------------------------------

/// The store accessors the generated code uses: a root sizes and fills
/// its stores in `begin_with_context` and reads them back per sweep. A
/// user of [`Context`] needs none of these.
impl Context {
    /// This root's Hessian block stores, if the context holds them: one
    /// per thread. A solve that threads nothing has exactly one, and
    /// every walk over the list reads the same whatever its length.
    pub fn blocks_list<S: BlockStore>(&self) -> Option<&[S]> {
        self.blocks.as_ref()
            .and_then(|b| b.store_any().downcast_ref::<std::vec::Vec<S>>())
            .map(|v| &v[..self.blocks_len.min(v.len())])
    }

    /// The stores of this solve, `n` of them, with the cut that divides
    /// the walks among them. Empty stores when the context held none or
    /// another root's; the generated build fills them.
    ///
    /// The list is never shortened, only re-lengthened: a solve with fewer
    /// stores than the last one keeps the spare allocations and uses the
    /// front of the list. The two come together because a build needs
    /// both and taking them one at a time would borrow the context twice.
    pub fn stores_mut<S: BlockStore>(&mut self, n: usize) -> (&mut [S], &Cut) {
        let n = n.max(1);
        let Context { blocks, blocks_len, cut, .. } = self;
        let v = ensure::<S>(blocks, blocks_len, n);
        (&mut v[..n], &*cut)
    }

    /// What a sweep needs from the context at once: the stores this solve
    /// is already using (without changing how many), the cut, and
    /// somewhere to record where the time went. Three disjoint fields,
    /// so one call rather than three borrows.
    pub fn sweep_parts_mut<S: BlockStore>(&mut self)
        -> (&mut [S], &Cut, &mut ParTiming)
    {
        let n = self.blocks_len.max(1);
        let Context { blocks, blocks_len, cut, sweeps, .. } = self;
        let v = ensure::<S>(blocks, blocks_len, n);
        (&mut v[..n], &*cut, sweeps)
    }

    /// How this solve's walks are divided among its stores.
    pub fn cut(&self) -> &Cut { &self.cut }

    /// The cut, to build.
    pub fn cut_mut(&mut self) -> &mut Cut { &mut self.cut }

    /// The shape the stores were built for: the model's count per block
    /// array, in array order. Empty until a build records one.
    pub fn shape(&self) -> &[u64] { &self.shape }

    /// What each of this solve's stores holds, in store order; empty
    /// when the context holds no stores.
    pub fn footprints(&self) -> std::vec::Vec<StoreFootprint> {
        self.blocks.as_ref().map_or_else(std::vec::Vec::new, |b| b.store_footprints(self.blocks_len.max(1)))
    }

    /// What one whole store of the model holds: the model's block
    /// counts and the bytes they take with nothing duplicated, as the
    /// build records it. Set against [`footprints`](Self::footprints)
    /// summed, the difference is what the split stores duplicate.
    pub fn whole(&self) -> StoreFootprint { self.whole }

    /// Record the whole model's footprint a build was made for.
    pub fn set_whole(&mut self, whole: StoreFootprint) { self.whole = whole; }

    /// Record the shape a build was made for.
    pub fn set_shape(&mut self, shape: &[u64]) {
        self.shape.clear();
        self.shape.extend_from_slice(shape);
    }
}

/// The store list for `S` in `blocks`, `n` long at least, replacing
/// another root's list if that is what was held; `blocks_len` records
/// how many of them the solve uses.
fn ensure<'a, S: BlockStore>(
    blocks: &'a mut Option<Box<dyn AnyStore>>,
    blocks_len: &mut usize,
    n: usize,
) -> &'a mut std::vec::Vec<S> {
    let holds = blocks.as_ref()
        .is_some_and(|b| b.store_any().is::<std::vec::Vec<S>>());
    if !holds {
        *blocks = Some(Box::new(std::vec::Vec::<S>::new()));
    }
    let v = blocks.as_mut().unwrap().store_any_mut()
        .downcast_mut::<std::vec::Vec<S>>().unwrap();
    if v.len() < n { v.resize_with(n, S::default); }
    *blocks_len = n;
    v
}


// ---------------------------------------------------------------------------
// The dispatch
// ---------------------------------------------------------------------------

/// What the threaded sweeps need of a root's model: every store's task
/// reads it at once, so it must be `Sync`. Every `Sync` type is one. A
/// generated root asserts it, so a root holding an `Rc` or a `Cell` is
/// told to opt out with `#[arael(root, seq)]`; the `Sync` error beside
/// it names the field.
#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot be swept on several threads: it is not `Sync`",
    label = "this root's model is not `Sync`",
    note = "opt the root out of the threaded sweeps with `#[arael(root, seq)]`, \
            or replace the field that is not `Sync` (`Rc`, `Cell`, `RefCell`)"
)]
pub trait SweepsInParallel {}
#[diagnostic::do_not_recommend]
impl<T: Sync + ?Sized> SweepsInParallel for T {}

/// Run `f` over every store with the model, told which store it is so it
/// can take its own row of the cut: one task per store on the sweep
/// workers ([`crate::pool`]) when threaded, else in order on the calling
/// thread. The two forms do the same arithmetic in the same order per
/// store -- the region is what changes, not the work.
pub fn run_indexed<Mo: Sync + ?Sized, M: Send>(
    model: &Mo, par: bool, stores: &mut [M], f: impl Fn(&Mo, usize, &mut M) + Sync,
) {
    #[cfg(feature = "threads")]
    if par {
        crate::pool::run_over(stores, |i, m| f(model, i, m));
        return;
    }
    let _ = par;
    for (i, m) in stores.iter_mut().enumerate() {
        f(model, i, m);
    }
}
// ---------------------------------------------------------------------------
// The partition
// ---------------------------------------------------------------------------

/// The start of store `t`'s range when `n` slots are divided into `p`
/// contiguous ranges of about equal length; `t == p` gives `n`.
#[inline]
pub fn cut(n: usize, p: usize, t: usize) -> usize {
    (n * t).div_ceil(p)
}

/// How a solve's walks are divided among its stores.
///
/// One row per store, one entry per walk: the slot range that store
/// covers of that walk. An empty cut is the whole model in one store,
/// which is what a sequential solve uses and what every solve uses until
/// a cut is built for it -- every row then reads as the empty table
/// [`walk_range`] treats as "all of it".
#[derive(Clone, Debug, Default)]
pub struct Cut {
    /// `stores * walks` entries, one row per store.
    ranges: std::vec::Vec<(u32, u32)>,
    walks: usize,
    stores: usize,
}

impl Cut {
    /// A cut dividing nothing: the whole model, one store.
    pub fn new() -> Self { Cut { ranges: std::vec::Vec::new(), walks: 0, stores: 0 } }

    /// True while no cut has been built.
    pub fn is_empty(&self) -> bool { self.stores == 0 }

    /// The stores this cut divides the walks among.
    pub fn stores(&self) -> usize { self.stores }

    /// The walks it has a range for.
    pub fn walks(&self) -> usize { self.walks }

    /// Forget the cut: back to the whole model in one store. This is the
    /// single answer to "is a store a slice or all of it" -- an empty cut
    /// means whole, everywhere that asks.
    pub fn clear(&mut self) {
        self.ranges.clear();
        self.stores = 0;
        self.walks = 0;
    }

    /// Start a cut of `walks` walks over `stores` stores, every range
    /// covering nothing until it is set. Keeps its allocation.
    pub fn reset(&mut self, stores: usize, walks: usize) {
        self.ranges.clear();
        self.ranges.resize(stores * walks, (0, 0));
        self.stores = stores;
        self.walks = walks;
    }

    /// Give store `s` the slot range `[lo, hi)` of walk `w`.
    pub fn set(&mut self, s: usize, w: usize, lo: u32, hi: u32) {
        self.ranges[s * self.walks + w] = (lo, hi);
    }

    /// The ranges store `s` covers, one per walk. Empty while no cut has
    /// been built, which every walk reads as the whole of its container.
    #[inline]
    pub fn store(&self, s: usize) -> &[(u32, u32)] {
        if self.is_empty() { return &[]; }
        &self.ranges[s * self.walks..(s + 1) * self.walks]
    }
}

/// The slot range walk `k` of a sweep should cover.
///
/// A sweep is handed one range per top-level walk, in the order the walks
/// are emitted. An empty table means the whole model, which is what the
/// sequential path passes and what every walk gets until a cut exists to
/// fill the table in: [`Leaves::range_iter`] clamps `u32::MAX` to the
/// container's length, so the full range is the plain walk.
#[inline(always)]
pub fn walk_range(ranges: &[(u32, u32)], k: usize) -> (u32, u32) {
    if ranges.is_empty() { (0, u32::MAX) } else { ranges[k] }
}


// ---------------------------------------------------------------------------
// Hessian tiles and their positions
// ---------------------------------------------------------------------------

/// Upper triangle index: element (i,j) with i<=j in an NxN symmetric matrix.
#[inline]
fn tri_idx(n: usize, i: usize, j: usize) -> usize {
    i * (2 * n - i - 1) / 2 + j
}


/// One tiled block in the position stream: which block it is and where
/// its tile lies in the assembled value buffer. The tile is the dense
/// `rows` by `cols` rectangle the pattern stores for the block's cell,
/// column-major from `base` with the columns `stride` apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TileEntry {
    base: ValueIndex,
    stride: ValueIndex,
    /// The block's index in its array.
    index: u32,
    /// The store the block is in, and the array of that store.
    store: u8,
    array: u8,
    rows: u8,
    cols: u8,
}

/// The scatter targets of a bound pattern: what every assembly of a
/// solve replays. Built by binding the root's block stores against the
/// pattern (`LmProblemInternals::bind_hessian_positions`) and closed
/// with [`finish`](Self::finish). The blocks keep nothing, so the stream
/// is the whole binding and a fresh set of blocks of the same shape
/// scatters through it as is.
///
/// Against a tile-expanded pattern ([`HessianBinder::Tiled`]) every
/// block binds to its tile. The tiles are sorted by where they lie in
/// the value buffer, so the blocks that land on one tile are neighbours
/// in the stream: the scatter writes the first of them over the tile
/// and adds the rest to it, and needs no zeroed buffer to add into. The
/// sorted stream is cut into
/// one range of the buffer per store, for the scatter to run on every
/// thread ([`scatter_tiles`]). The COO list has no tiles and scatters
/// through one position per entry after them. Against a pattern without
/// tiles ([`HessianBinder::Scalar`]) every block and the COO list bind
/// to one position per entry, and the stores replay them in walk order.
#[derive(Clone, Debug, Default)]
pub struct PositionStream {
    /// One position per entry for the blocks without a tile and the COO
    /// list, in walk order.
    positions: std::vec::Vec<ValueIndex>,
    /// The tiled blocks: in binding order until `finish`, by tile after.
    tiles: std::vec::Vec<TileEntry>,
    /// Per tiled block the first value index of its panel, the block
    /// column its tile lies in. Emptied by `finish`, which cuts on it.
    panels: std::vec::Vec<ValueIndex>,
    /// Chunk `c` is the tiles `chunks[c].0..chunks[c + 1].0`, writing
    /// the value buffer from `chunks[c].1` up to `chunks[c + 1].1`; the
    /// last pair closes the table with `ValueIndex::MAX` for the
    /// buffer's end.
    chunks: std::vec::Vec<(u32, ValueIndex)>,
    /// How many cells of the value buffer the tiles write.
    covered: usize,
    /// How many stores the tiles were bound from.
    stores: usize,
    /// Whether the blocks bound to tiles.
    tiled: bool,
}

impl PositionStream {
    pub fn new() -> Self { Self::default() }

    /// The stream of a pattern given as one position per emitted entry,
    /// a [`CooMatrix::to_csc_with_map`](crate::simple_lm::CooMatrix::to_csc_with_map)
    /// map: the blocks and the COO list replay it in the order they
    /// emitted.
    pub fn from_map(map: &[ValueIndex]) -> Self {
        PositionStream { positions: map.to_vec(), ..Default::default() }
    }

    /// The per-entry positions.
    pub fn positions(&self) -> &[ValueIndex] { &self.positions }

    /// How many blocks bound to tiles.
    pub fn tiles(&self) -> usize { self.tiles.len() }

    /// Whether the blocks bound to tiles rather than to per-entry
    /// positions.
    pub fn tiled(&self) -> bool { self.tiled }

    /// Whether the tiles write every cell of a value buffer `len` long,
    /// so that the scatter needs no zeroing before it.
    pub fn covers(&self, len: usize) -> bool { self.tiled && self.covered == len }

    /// How many chunks the tiles are cut into.
    pub fn chunks(&self) -> usize { self.chunks.len().saturating_sub(1) }

    /// The bytes the stream holds.
    pub fn bytes(&self) -> usize {
        self.positions.len() * std::mem::size_of::<ValueIndex>()
            + self.tiles.len() * std::mem::size_of::<TileEntry>()
            + self.chunks.len() * std::mem::size_of::<(u32, ValueIndex)>()
    }

    pub(crate) fn push_position(&mut self, position: usize) {
        self.positions.push(value_index(position));
    }

    /// Room for `n` more tiles.
    pub(crate) fn reserve_tiles(&mut self, n: usize) {
        self.tiles.reserve(n);
        self.panels.reserve(n);
    }

    #[allow(clippy::too_many_arguments)]
    fn push_tile(
        &mut self, base: usize, stride: usize, panel: usize,
        store: usize, array: usize, index: usize, rows: usize, cols: usize,
    ) {
        assert!(store <= u8::MAX as usize && array <= u8::MAX as usize,
            "the position stream holds up to {} stores of {} arrays", u8::MAX, u8::MAX);
        assert!(rows <= u8::MAX as usize && cols <= u8::MAX as usize,
            "a tile of {} by {}: an entity's block spans up to {} parameters", rows, cols, u8::MAX);
        self.tiles.push(TileEntry {
            base: value_index(base), stride: value_index(stride), index: index as u32,
            store: store as u8, array: array as u8, rows: rows as u8, cols: cols as u8,
        });
        self.panels.push(value_index(panel));
    }

    /// Close a binding from `stores` stores: sort the tiles by where
    /// they lie, count the cells they cover and cut them into `stores`
    /// chunks of about equal length. A chunk starts where a panel does,
    /// so no tile straddles two, and two chunks never write one cell.
    pub fn finish(&mut self, stores: usize) {
        self.stores = stores;
        let n = self.tiles.len();
        // By tile, and in binding order within one, so the sum over a
        // tile's blocks is in one fixed order: the tile's base over the
        // entry's place, as one key.
        let mut keys: std::vec::Vec<u64> = self.tiles.iter().enumerate()
            .map(|(k, t)| ((t.base as u64) << 32) | k as u64)
            .collect();
        arael_faer::sort_keys(&mut keys, 32..64);
        let at = |key: u64| (key & 0xffff_ffff) as usize;
        let tiles: std::vec::Vec<TileEntry> = keys.iter().map(|&key| self.tiles[at(key)]).collect();
        let panels: std::vec::Vec<ValueIndex> = keys.iter().map(|&key| self.panels[at(key)]).collect();
        self.tiles = tiles;
        self.panels = std::vec::Vec::new();
        // Every block on a tile writes the whole of it, so all agree on
        // its shape, and the tile counts once.
        self.covered = 0;
        for k in 0..n {
            let t = &self.tiles[k];
            if k > 0 && self.tiles[k - 1].base == t.base {
                let p = &self.tiles[k - 1];
                assert!((p.stride, p.rows, p.cols) == (t.stride, t.rows, t.cols),
                    "two blocks bound to one tile with different shapes");
            } else {
                self.covered += t.rows as usize * t.cols as usize;
            }
        }
        let k = stores.max(1);
        self.chunks.clear();
        let mut prev = 0usize;
        for c in 0..k {
            let mut at = cut(n, k, c).max(prev);
            while at > 0 && at < n && panels[at] == panels[at - 1] {
                at += 1;
            }
            self.chunks.push((at as u32, if at < n { panels[at] } else { ValueIndex::MAX }));
            prev = at;
        }
        self.chunks.push((n as u32, ValueIndex::MAX));
    }
}

/// The per-entry positions by entry: what a hand-written problem reads
/// its map through.
impl std::ops::Index<usize> for PositionStream {
    type Output = ValueIndex;
    fn index(&self, k: usize) -> &ValueIndex { &self.positions[k] }
}

/// A root's block store as the tiled scatter reads it: any block of any
/// of its arrays, written into its tile. The tile lies in `vals`
/// column-major from `base`, its columns `stride` apart. Implemented by
/// the generated store.
pub trait TileSource: Sync {
    /// Write block `index` of array `array` over its tile: every cell
    /// of the tile is assigned, the ones the block does not reach as
    /// zeros.
    fn assign_block<F: crate::utils::Float>(&self, array: u8, index: u32, vals: &mut [F], base: usize, stride: usize);
    /// Add block `index` of array `array` to what its tile holds.
    fn add_block<F: crate::utils::Float>(&self, array: u8, index: u32, vals: &mut [F], base: usize, stride: usize);
}

/// One store's share of the tiled scatter: a range of the value buffer
/// and the sorted tiles that lie in it.
struct TileChunk<'a, F> {
    vals: &'a mut [F],
    /// The value index `vals` starts at.
    start: usize,
    tiles: &'a [TileEntry],
}

/// Write every tiled block of `stream` into `vals`: the first block of
/// a tile over the whole of it, the rest of its blocks added, so a cell
/// of a tile owes nothing to what the buffer held. One chunk of the
/// buffer per store the stream was bound from, each on a thread of its
/// own when there are several ([`crate::pool`]).
pub fn scatter_tiles<S: TileSource, F: crate::utils::Float + Send>(
    stores: &[S], stream: &PositionStream, vals: &mut [F],
) {
    if stream.tiles.is_empty() {
        return;
    }
    assert!(stream.stores == stores.len(),
        "the position stream was bound from {} stores and is scattered from {}: \
         a bound pattern serves one store count", stream.stores, stores.len());
    let chunks = &stream.chunks;
    let n = chunks.len() - 1;
    let len = vals.len();
    let at = |c: usize| {
        let v = chunks[c].1;
        if v == ValueIndex::MAX { len } else { v as usize }
    };
    let mut parts: std::vec::Vec<TileChunk<F>> = std::vec::Vec::with_capacity(n);
    let mut rest: &mut [F] = vals;
    let mut taken = 0usize;
    for c in 0..n {
        let (lo, hi) = (at(c), at(c + 1));
        let (_, r) = std::mem::take(&mut rest).split_at_mut(lo - taken);
        let (mine, r) = r.split_at_mut(hi - lo);
        rest = r;
        taken = hi;
        parts.push(TileChunk {
            vals: mine,
            start: lo,
            tiles: &stream.tiles[chunks[c].0 as usize..chunks[c + 1].0 as usize],
        });
    }
    run_indexed(stores, n > 1, &mut parts, |stores, _, chunk| scatter_chunk(stores, chunk));
}

/// The scatter of one chunk, straight into its range of the buffer: the
/// first entry of a tile assigns the tile, the entries after it on the
/// same tile add. A chunk opens on a panel, so a tile's entries are all
/// in one chunk and its first is the first the chunk sees.
fn scatter_chunk<S: TileSource, F: crate::utils::Float>(stores: &[S], chunk: &mut TileChunk<F>) {
    let vals = &mut *chunk.vals;
    let mut tile = usize::MAX;
    for e in chunk.tiles {
        let store = &stores[e.store as usize];
        let (base, stride) = (e.base as usize - chunk.start, e.stride as usize);
        if e.base as usize != tile {
            tile = e.base as usize;
            store.assign_block(e.array, e.index, vals, base, stride);
        } else {
            store.add_block(e.array, e.index, vals, base, stride);
        }
    }
}

/// The next per-entry position of the stream, and the caller's error
/// explained when there is none.
#[inline]
fn take_position(positions: &[ValueIndex], cursor: &mut usize) -> usize {
    let Some(&p) = positions.get(*cursor) else {
        panic!("Hessian scatter ran past the bound pattern: the stores were never bound \
                to it (LmProblemInternals::bind_hessian_positions), their emission order \
                changed within the solve, or the pattern was bound at another store count");
    };
    *cursor += 1;
    p as usize
}

/// Smallest live index in `indices`, or `u32::MAX` if every slot is fixed.
///
/// Parameters serialize in declaration order and the index arrays are filled
/// in that same order, so live indices ascend with the slot and the first one
/// found is the smallest. `bind_tile` asserts this before relying on it.
#[inline]
fn tile_start(indices: &[u32]) -> u32 {
    for &i in indices {
        if i != u32::MAX {
            return i;
        }
    }
    u32::MAX
}

/// How a backend hands out scatter targets when a root binds its block
/// stores to an assembled pattern (`LmProblemInternals::bind_hessian_positions`).
pub enum HessianBinder<'a> {
    /// Tile-expanded pattern: every stored cell holds a full dense tile.
    /// Given a tile's first scalar row and column: the tile's first value
    /// index, the distance between its columns, and the first value index
    /// of its panel, the block column it lies in, whose storage is one
    /// range of the buffer that no other panel's interleaves with.
    Tiled(&'a mut dyn FnMut(u32, u32) -> (usize, usize, usize)),
    /// Pattern built from a COO pass: a cell's entries are not contiguous in
    /// the value buffer, so every scalar needs its own position and the
    /// blocks fall back to the per-scalar map.
    Scalar(&'a mut dyn FnMut(u32, u32) -> usize),
}

/// The first live index in `indices` and how many follow it: the block's
/// span along one side of its tile. `u32::MAX` and 0 when every slot is
/// fixed. An entity's live parameters serialize as one run, which the
/// tile arithmetic relies on, so a gap is an error here.
fn tile_extent(indices: &[u32]) -> (u32, usize) {
    let (mut first, mut last, mut live, mut ascends) = (u32::MAX, 0u32, 0usize, true);
    for &i in indices {
        if i == u32::MAX { continue; }
        if live == 0 {
            first = i;
        } else {
            ascends &= i > last;
        }
        last = i;
        live += 1;
    }
    if live == 0 {
        return (u32::MAX, 0);
    }
    assert!(ascends && (last - first) as usize + 1 == live,
        "live parameter indices must ascend with slot order and form one run");
    (first, live)
}

/// Bind one block to its tile: the block's spans from its indices, the
/// tile from the binder. A block with a side entirely fixed has no tile
/// and scatters nothing.
#[inline]
fn bind_tile(
    bind: &mut dyn FnMut(u32, u32) -> (usize, usize, usize),
    row: &[u32], col: &[u32],
    out: &mut PositionStream, store: usize, array: usize, index: usize,
) {
    out.tiled = true;
    let (r, rows) = tile_extent(row);
    // A self block binds its indices on both sides.
    let (c, cols) = if std::ptr::eq(row, col) { (r, rows) } else { tile_extent(col) };
    if rows == 0 || cols == 0 {
        return;
    }
    // The tile holds the upper block triangle, so the lower-numbered span
    // walks the rows.
    let (r, rows, c, cols) = if r <= c { (r, rows, c, cols) } else { (c, cols, r, rows) };
    let (base, stride, panel) = bind(r, c);
    out.push_tile(base, stride, panel, store, array, index, rows, cols);
}



// ---------------------------------------------------------------------------
// Self blocks
// ---------------------------------------------------------------------------

// The self-block arithmetic, shared by [`SelfBlock`] and the
// [`SelfBlockArray`] slabs: one block is its N parameter indices and the
// M entries of its upper triangle.

/// The entity's parameter span as `(offset, width)`; nothing when every
/// slot is fixed.
#[inline]
fn self_param_block<const N: usize>(indices: &[u32; N], out: &mut std::vec::Vec<(u32, u32)>) {
    let mut min = u32::MAX;
    let mut count = 0u32;
    for &i in indices {
        if i != u32::MAX {
            if i < min { min = i; }
            count += 1;
        }
    }
    if count > 0 {
        out.push((min, count));
    }
}

/// One representative coordinate of the block's cell: its anchor on the
/// diagonal of the entity partition.
#[inline]
fn self_cells<const N: usize>(indices: &[u32; N], out: &mut std::vec::Vec<(u32, u32)>) {
    let min = tile_start(indices);
    if min != u32::MAX {
        out.push((min, min));
    }
}

/// The block's scatter target: against a tiled pattern its tile, in the
/// stream's tile list; otherwise one position per entry.
#[inline]
fn self_bind<const N: usize>(
    indices: &[u32; N], binder: &mut HessianBinder, out: &mut PositionStream,
    store: usize, array: usize, index: usize,
) {
    let resolve = match binder {
        HessianBinder::Tiled(bind) => return bind_tile(*bind, indices, indices, out, store, array, index),
        HessianBinder::Scalar(resolve) => resolve,
    };
    for i in 0..N {
        let gi = indices[i];
        if gi == u32::MAX { continue; }
        for j in i..N {
            let gj = indices[j];
            if gj == u32::MAX { continue; }
            let (lo, hi) = if gi <= gj { (gi, gj) } else { (gj, gi) };
            out.push_position(resolve(lo, hi));
        }
    }
}

/// Write one block's triangle into its tile in the value buffer, the
/// cell `(r, c)` of the tile at `base + c * stride + r`. With `ASSIGN`
/// the whole tile is written, its lower triangle as zeros; without, the
/// triangle is added to what the tile holds.
#[inline]
fn self_tile<const ASSIGN: bool, const N: usize, const M: usize, T: crate::utils::Float, F: crate::utils::Float>(
    indices: &[u32; N], hessian: &[T; M], vals: &mut [F], base: usize, stride: usize,
) {
    let start = tile_start(indices) as usize;
    for i in 0..N {
        let gi = indices[i];
        if gi == u32::MAX { continue; }
        let r = gi as usize - start;
        let tri = i * (2 * N - i - 1) / 2;
        for j in i..N {
            let gj = indices[j];
            if gj == u32::MAX { continue; }
            let c = gj as usize - start;
            let val = F::from(hessian[tri + j]).unwrap();
            if ASSIGN {
                vals[base + c * stride + r] = val;
                if r != c {
                    vals[base + r * stride + c] = F::zero();
                }
            } else {
                vals[base + c * stride + r] += val;
            }
        }
    }
}

/// The dense symmetric scatter of one block.
#[inline]
fn self_dense<const N: usize, const M: usize, T: crate::utils::Float, F: crate::utils::Float>(
    indices: &[u32; N], hessian: &[T; M], out: &mut [F],
) {
    let n_total = (out.len() as f64).sqrt() as usize;
    for i in 0..N {
        let gi = indices[i];
        if gi == u32::MAX { continue; }
        let gi = gi as usize;
        for j in i..N {
            let gj = indices[j];
            if gj == u32::MAX { continue; }
            let gj = gj as usize;
            let val = F::from(hessian[tri_idx(N, i, j)]).unwrap();
            out[gi * n_total + gj] += val;
            if gi != gj {
                out[gj * n_total + gi] += val;
            }
        }
    }
}

/// The band scatter of one block (column-major, (kd+1)*n).
#[inline]
fn self_band<const N: usize, const M: usize, T: crate::utils::Float, F: crate::utils::Float>(
    indices: &[u32; N], hessian: &[T; M], band: &mut [F], kd: usize,
) -> Result<(), crate::simple_lm::BandOverflow> {
    let ldab = kd + 1;
    for i in 0..N {
        let gi = indices[i];
        if gi == u32::MAX { continue; }
        let gi = gi as usize;
        for j in i..N {
            let gj = indices[j];
            if gj == u32::MAX { continue; }
            let gj = gj as usize;
            let (lo, hi) = if gi <= gj { (gi, gj) } else { (gj, gi) };
            if hi - lo > kd {
                return Err(crate::simple_lm::BandOverflow { row: lo, col: hi, kd });
            }
            band[(kd + lo - hi) + hi * ldab] += F::from(hessian[tri_idx(N, i, j)]).unwrap();
        }
    }
    Ok(())
}

/// The COO scatter of one block, upper triangle only.
#[inline]
fn self_coo<const N: usize, const M: usize, T: crate::utils::Float, F: crate::utils::Float>(
    indices: &[u32; N], hessian: &[T; M], coo: &mut crate::simple_lm::CooMatrix<F>,
) {
    for i in 0..N {
        let gi = indices[i];
        if gi == u32::MAX { continue; }
        for j in i..N {
            let gj = indices[j];
            if gj == u32::MAX { continue; }
            let (lo, hi) = if gi <= gj { (gi, gj) } else { (gj, gi) };
            coo.push(lo, hi, F::from(hessian[tri_idx(N, i, j)]).unwrap());
        }
    }
}

/// The direct CSC scatter of one block, by position lookup.
#[inline]
fn self_direct<const N: usize, const M: usize, T: crate::utils::Float, F: crate::utils::Float>(
    indices: &[u32; N], hessian: &[T; M], csc: &mut crate::simple_lm::CscMatrix<F>,
) {
    for i in 0..N {
        let gi = indices[i];
        if gi == u32::MAX { continue; }
        for j in i..N {
            let gj = indices[j];
            if gj == u32::MAX { continue; }
            let (lo, hi) = if gi <= gj { (gi, gj) } else { (gj, gi) };
            let val = F::from(hessian[tri_idx(N, i, j)]).unwrap();
            if let Some(pos) = csc.find_pos(lo as usize, hi as usize) {
                csc.vals[pos] += val;
            }
        }
    }
}

/// The scatter of one block through the per-entry positions, `cursor`
/// advancing in lockstep with the block traversal. An all-fixed block
/// emits nothing and so consumes none.
#[inline]
fn self_mapped<const N: usize, const M: usize, T: crate::utils::Float, F: crate::utils::Float>(
    indices: &[u32; N], hessian: &[T; M],
    vals: &mut [F], positions: &[ValueIndex], cursor: &mut usize,
) {
    for i in 0..N {
        let gi = indices[i];
        if gi == u32::MAX { continue; }
        for j in i..N {
            if indices[j] == u32::MAX { continue; }
            vals[take_position(positions, cursor)] += F::from(hessian[tri_idx(N, i, j)]).unwrap();
        }
    }
}

/// A slab of self blocks with the indices apart from the values: per
/// entity its container slot and its N parameter indices in one array,
/// the M entries of its upper triangle and its N gradient entries in one
/// flat array. A root's generated block store holds one per entity
/// container, for the entities its range writes (see [`Context`]).
///
/// The build pushes the indices only and then calls
/// [`finish`](Self::finish), which sizes the value array in one zeroed
/// allocation and keeps it when the length has not changed. The sweep
/// zeroes the whole slab before it writes.
///
/// One entity's triangle and its gradient stash are adjacent in `data`,
/// at stride `M + N`: a residual writes both, so they share the region
/// the write pulls in rather than sitting in two arrays a thread has to
/// stream separately.
#[derive(Clone)]
pub struct SelfBlockArray<const N: usize, const M: usize, T: crate::utils::Float = f64> {
    entity: std::vec::Vec<u32>,
    indices: std::vec::Vec<[u32; N]>,
    /// Per entity: `M` triangle entries, then `N` gradient entries.
    data: std::vec::Vec<T>,
    /// Global slot to this slab's position, for a slab holding only some
    /// of the entities. Empty on a slab that holds them all, where the
    /// marker's slot addresses the data directly.
    map: std::vec::Vec<u32>,
}

impl<const N: usize, const M: usize, T: crate::utils::Float> Default for SelfBlockArray<N, M, T> {
    fn default() -> Self { Self::new() }
}

impl<const N: usize, const M: usize, T: crate::utils::Float> SelfBlockArray<N, M, T> {
    const CHECK_M: () = assert!(M == N * (N + 1) / 2, "SelfBlockArray: M must equal N*(N+1)/2");
    /// One entity's span of [`data`](Self::data): the triangle then the stash.
    const STRIDE: usize = M + N;

    /// An empty slab.
    pub fn new() -> Self {
        let () = Self::CHECK_M;
        SelfBlockArray {
            entity: std::vec::Vec::new(),
            indices: std::vec::Vec::new(),
            data: std::vec::Vec::new(),
            map: std::vec::Vec::new(),
        }
    }

    /// The number of entities.
    pub fn len(&self) -> usize { self.entity.len() }

    /// Bytes one block takes in a whole store: its triangle and stash,
    /// its indices and its entity slot.
    pub const BLOCK_BYTES: usize = Self::STRIDE * std::mem::size_of::<T>() + N * 4 + 4;

    /// The blocks held and the bytes of the arrays, the map included.
    pub fn footprint(&self) -> StoreFootprint {
        StoreFootprint {
            self_blocks: self.entity.len(),
            cross_blocks: 0,
            coo_entries: 0,
            bytes: self.data.len() * std::mem::size_of::<T>()
                + self.indices.len() * N * 4 + self.entity.len() * 4 + self.map.len() * 4,
        }
    }

    /// True if the slab holds no entity.
    pub fn is_empty(&self) -> bool { self.entity.is_empty() }

    /// Drop every entity's indices; the value storage stays for `finish`.
    ///
    /// The map goes too (its allocation stays): a whole build never
    /// touches it and must find it empty, and a split build reads it to
    /// ask whether it has already given a slot a place, where a stale
    /// entry from the last solve would answer yes.
    pub fn clear(&mut self) {
        self.entity.clear();
        self.indices.clear();
        self.map.clear();
    }

    /// True if `slot` already has a place in this slab.
    #[inline]
    pub fn holds(&self, slot: usize) -> bool {
        self.map.get(slot).is_some_and(|&k| k != u32::MAX)
    }

    /// Room for `n` more entities' indices.
    pub fn reserve(&mut self, n: usize) {
        self.entity.reserve(n);
        self.indices.reserve(n);
    }

    /// Append the entity at container slot `entity` with its parameter
    /// indices, and return the slab slot it took. A store holding a whole
    /// container is built this way, in container order, so the marker's
    /// slot is the slab slot and no map is kept.
    #[inline]
    pub fn push(&mut self, entity: u32, indices: &[u32; N]) -> u32 {
        let k = self.entity.len() as u32;
        self.entity.push(entity);
        self.indices.push(*indices);
        k
    }

    /// Record that global `slot` sits at slab slot `k`. The build sizes
    /// the map first ([`map_resize`](Self::map_resize)) to what the
    /// numbering can reach, so growing here means that count was short;
    /// a debug build says so, a release build grows and carries on.
    #[inline]
    fn map_to(&mut self, slot: u32, k: u32) {
        if self.map.len() <= slot as usize {
            debug_assert!(false,
                "slot {} lies past the map's {} entries: the build presized it short",
                slot, self.map.len());
            self.map.resize(slot as usize + 1, u32::MAX);
        }
        self.map[slot as usize] = k;
    }

    /// Room for `n` global slots in the map.
    ///
    /// The entries are not cleared, and do not need to be: a slot this
    /// slab never claims is never read, because a sweep walks the very
    /// instances the build walked and the build claims a place for every
    /// one of them. Only the length has to be right.
    pub fn map_resize(&mut self, n: usize) {
        if self.map.len() < n { self.map.resize(n, u32::MAX); }
    }

    /// Take global `slot` into this slab, returning where it landed. A
    /// store holding part of a container is built this way, and the map
    /// it fills is what makes the slab addressable by the marker's slot.
    pub fn push_at(&mut self, slot: u32, entity: u32, indices: &[u32; N]) -> u32 {
        let k = self.push(entity, indices);
        self.map_to(slot, k);
        k
    }

    /// Where global `slot` sits in this slab: the slot itself in a slab
    /// holding the whole container, else through the map. `split` says
    /// which; the sweep decides it once from its range table and hands
    /// it to every write, so the test is on a register and nothing is
    /// reloaded from the slab it is writing.
    ///
    /// A slot a split build never claimed lands here as `u32::MAX` and
    /// the write would run off the end. That means the build's idea of
    /// which entities a store touches disagrees with what its sweep
    /// writes, so it says which slot rather than panicking on an index
    /// far away.
    #[inline(always)]
    pub fn slab_of(&self, split: bool, slot: usize) -> usize {
        if !split {
            return slot;
        }
        let k = self.map[slot];
        debug_assert!(k != u32::MAX,
            "slot {} is written by this store's sweep but the build gave it no place \
             in the slab", slot);
        k as usize
    }

    /// Size the value array to the entities pushed: a fresh zeroed
    /// allocation when the length changed, else the existing storage,
    /// whose stale values [`zero`](Self::zero) clears.
    pub fn finish(&mut self) {
        let n = self.entity.len() * Self::STRIDE;
        if self.data.len() != n {
            self.data = vec![T::zero(); n];
        }
    }

    /// Reset every triangle and every gradient stash.
    pub fn zero(&mut self) {
        self.data.fill(T::zero());
    }

    /// The container slot of entity `k`.
    pub fn entity(&self, k: usize) -> u32 { self.entity[k] }

    /// The parameter indices of entity `k` (`u32::MAX` for a fixed one).
    pub fn indices(&self, k: usize) -> &[u32; N] { &self.indices[k] }

    #[inline]
    fn block(&self, k: usize) -> &[T; M] {
        let base = k * Self::STRIDE;
        (&self.data[base..base + M]).try_into().unwrap()
    }

    /// Add one residual's contribution to the entity at global `slot`
    /// (the marker's; see [`slab_of`](Self::slab_of)): `2 r dr` into its
    /// gradient stash and `2 dr dr^T` into its triangle, every parameter,
    /// fixed or not. A fixed parameter's row is accumulated like any
    /// other and dropped by [`scatter_grad`](Self::scatter_grad) and the
    /// Hessian walks, which read the indices.
    #[inline]
    pub fn add_residual(&mut self, split: bool, slot: usize, r: T, dr: &[T; N]) {
        let k = self.slab_of(split, slot);
        self.add_scaled(k, T::two(), r, dr);
    }

    /// [`add_residual`](Self::add_residual) scaled by the loss weight `w`.
    #[inline]
    pub fn add_residual_with_loss(&mut self, split: bool, slot: usize, w: T, r: T, dr: &[T; N]) {
        let k = self.slab_of(split, slot);
        self.add_scaled(k, T::two() * w, r, dr);
    }

    #[inline]
    fn add_scaled(&mut self, k: usize, scale: T, r: T, dr: &[T; N]) {
        let base = k * Self::STRIDE;
        let (h, g) = self.data[base..base + Self::STRIDE].split_at_mut(M);
        let h: &mut [T; M] = h.try_into().unwrap();
        let g: &mut [T; N] = g.try_into().unwrap();
        let sr = scale * r;
        for i in 0..N {
            g[i] += sr * dr[i];
            let tdi = scale * dr[i];
            for j in i..N {
                h[tri_idx(N, i, j)] += tdi * dr[j];
            }
        }
    }

    /// Add every entity's stash into the global gradient at its live
    /// indices; a fixed slot has none and is dropped here.
    pub fn scatter_grad<F: crate::utils::Float>(&self, grad: &mut [F]) {
        for k in 0..self.entity.len() {
            let idx = &self.indices[k];
            let base = k * Self::STRIDE + M;
            let g = &self.data[base..base + N];
            for i in 0..N {
                let gi = idx[i];
                if gi == u32::MAX { continue; }
                grad[gi as usize] += F::from(g[i]).unwrap();
            }
        }
    }

    /// Append entity `k`'s `(offset, width)` span of the flat parameter
    /// vector, or nothing when all of its params are fixed.
    pub fn collect_param_block(&self, k: usize, out: &mut std::vec::Vec<(u32, u32)>) {
        self_param_block(&self.indices[k], out);
    }

    /// Append one representative scalar coordinate of each entity's
    /// triangle, over every entity. Structure only, no values.
    pub fn collect_hessian_cells(&self, out: &mut std::vec::Vec<(u32, u32)>) {
        for idx in &self.indices {
            self_cells(idx, out);
        }
    }

    /// Record where each entity's triangle scatters in the assembled value
    /// buffer, over every entity, as array `array` of store `store`. See
    /// [`LmProblemInternals::bind_hessian_positions`](crate::simple_lm::LmProblemInternals::bind_hessian_positions).
    pub fn bind_hessian_positions(&self, binder: &mut HessianBinder, out: &mut PositionStream, store: usize, array: usize) {
        out.reserve_tiles(self.indices.len());
        for (k, idx) in self.indices.iter().enumerate() {
            self_bind(idx, binder, out, store, array, k);
        }
    }

    /// Write entity `k`'s triangle over its tile, which lies in `vals`
    /// from `base` with its columns `stride` apart (see [`TileSource`]).
    #[inline]
    pub fn assign_block<F: crate::utils::Float>(&self, k: usize, vals: &mut [F], base: usize, stride: usize) {
        self_tile::<true, N, M, _, _>(&self.indices[k], self.block(k), vals, base, stride);
    }

    /// Add entity `k`'s triangle to what its tile holds.
    #[inline]
    pub fn add_block<F: crate::utils::Float>(&self, k: usize, vals: &mut [F], base: usize, stride: usize) {
        self_tile::<false, N, M, _, _>(&self.indices[k], self.block(k), vals, base, stride);
    }

    /// Add every entity's triangle into the dense symmetric Hessian,
    /// both sides of the diagonal.
    pub fn accumulate_hessian<F: crate::utils::Float>(&self, hessian: &mut [F]) {
        for k in 0..self.entity.len() {
            self_dense(&self.indices[k], self.block(k), hessian);
        }
    }

    /// Add every entity's triangle into an upper-band Hessian of
    /// half-bandwidth `kd` (LAPACK layout). Errors when an entry falls
    /// outside the band.
    pub fn accumulate_hessian_band<F: crate::utils::Float>(&self, band: &mut [F], kd: usize)
        -> Result<(), crate::simple_lm::BandOverflow>
    {
        for k in 0..self.entity.len() {
            self_band(&self.indices[k], self.block(k), band, kd)?;
        }
        Ok(())
    }

    /// Push every entity's triangle into a COO Hessian as `(row, col,
    /// value)` triplets.
    pub fn accumulate_hessian_sparse<F: crate::utils::Float>(&self, coo: &mut crate::simple_lm::CooMatrix<F>) {
        for k in 0..self.entity.len() {
            self_coo(&self.indices[k], self.block(k), coo);
        }
    }

    /// Add every entity's triangle straight into an already-patterned CSC
    /// Hessian, looking each entry up by column.
    pub fn accumulate_hessian_sparse_direct<F: crate::utils::Float>(&self, csc: &mut crate::simple_lm::CscMatrix<F>) {
        for k in 0..self.entity.len() {
            self_direct(&self.indices[k], self.block(k), csc);
        }
    }

    /// Add every entity's triangle into the assembled value buffer through
    /// the per-entry positions [`bind_hessian_positions`](Self::bind_hessian_positions)
    /// recorded against a pattern without tiles, reading them in the same
    /// order.
    pub fn accumulate_hessian_sparse_indexed<F: crate::utils::Float>(&self, vals: &mut [F], positions: &[ValueIndex], cursor: &mut usize) {
        for k in 0..self.entity.len() {
            self_mapped(&self.indices[k], self.block(k), vals, positions, cursor);
        }
    }
}


// ---------------------------------------------------------------------------
// Cross blocks
// ---------------------------------------------------------------------------

// The cross-block arithmetic, shared by [`CrossBlock`] and the
// [`CrossBlockArray`] lists: one block is its A and B indices, its tile
// position and its NA x NB row-major values.

/// `values += scale * dr_a * dr_b^T`, skipping the rows whose `dr_a` is zero.
#[inline]
fn cross_add<const NA: usize, const NB: usize, const P: usize, T: crate::utils::Float>(
    values: &mut [T; P], scale: T, dr_a: &[T; NA], dr_b: &[T; NB],
) {
    for i in 0..NA {
        let dai = dr_a[i];
        if dai == T::zero() { continue; }
        let row = i * NB;
        for j in 0..NB {
            values[row + j] += scale * dai * dr_b[j];
        }
    }
}

/// The dense symmetric scatter of one block.
#[inline]
fn cross_dense<const NA: usize, const NB: usize, const P: usize, T: crate::utils::Float, F: crate::utils::Float>(
    a: &[u32; NA], b: &[u32; NB], values: &[T; P], hessian: &mut [F],
) {
    let n_total = (hessian.len() as f64).sqrt() as usize;
    for i in 0..NA {
        let gi = a[i];
        if gi == u32::MAX { continue; }
        let gi = gi as usize;
        let row = i * NB;
        for j in 0..NB {
            let gj = b[j];
            if gj == u32::MAX { continue; }
            let gj = gj as usize;
            let val = F::from(values[row + j]).unwrap();
            hessian[gi * n_total + gj] += val;
            hessian[gj * n_total + gi] += val;
        }
    }
}

/// The band scatter of one block (column-major, (kd+1)*n).
#[inline]
fn cross_band<const NA: usize, const NB: usize, const P: usize, T: crate::utils::Float, F: crate::utils::Float>(
    a: &[u32; NA], b: &[u32; NB], values: &[T; P], band: &mut [F], kd: usize,
) -> Result<(), crate::simple_lm::BandOverflow> {
    let ldab = kd + 1;
    for i in 0..NA {
        let gi = a[i];
        if gi == u32::MAX { continue; }
        let gi = gi as usize;
        let row = i * NB;
        for j in 0..NB {
            let gj = b[j];
            if gj == u32::MAX { continue; }
            let gj = gj as usize;
            let (lo, hi) = if gi <= gj { (gi, gj) } else { (gj, gi) };
            if hi - lo > kd {
                return Err(crate::simple_lm::BandOverflow { row: lo, col: hi, kd });
            }
            let val = F::from(values[row + j]).unwrap();
            // Aliased slots (same entity in both refs): the triangle
            // stores each symmetric pair once, so the diagonal needs
            // both of the 2*dr_a*dr_b contributions explicitly.
            let val = if gi == gj { val + val } else { val };
            band[(kd + lo - hi) + hi * ldab] += val;
        }
    }
    Ok(())
}

/// One representative coordinate of a block's cell: the two entities are
/// contiguous spans, so every pair lands in one cell of the entity
/// partition (the diagonal cell when aliased).
#[inline]
fn cross_cells<const NA: usize, const NB: usize>(a: &[u32; NA], b: &[u32; NB], out: &mut std::vec::Vec<(u32, u32)>) {
    let min_live = |idx: &[u32]| {
        let mut min = u32::MAX;
        for &i in idx {
            if i != u32::MAX && i < min { min = i; }
        }
        min
    };
    let (ma, mb) = (min_live(a), min_live(b));
    if ma != u32::MAX && mb != u32::MAX {
        out.push((ma.min(mb), ma.max(mb)));
    }
}

/// The block's scatter target: against a tiled pattern its tile, in the
/// stream's tile list; otherwise one position per entry.
#[inline]
fn cross_bind<const NA: usize, const NB: usize>(
    a: &[u32; NA], b: &[u32; NB], binder: &mut HessianBinder, out: &mut PositionStream,
    store: usize, array: usize, index: usize,
) {
    let resolve = match binder {
        HessianBinder::Tiled(bind) => return bind_tile(*bind, a, b, out, store, array, index),
        HessianBinder::Scalar(resolve) => resolve,
    };
    for i in 0..NA {
        let gi = a[i];
        if gi == u32::MAX { continue; }
        for j in 0..NB {
            let gj = b[j];
            if gj == u32::MAX { continue; }
            let (lo, hi) = if gi <= gj { (gi, gj) } else { (gj, gi) };
            out.push_position(resolve(lo, hi));
        }
    }
}

/// Write one block into its tile in the value buffer, the cell `(r, c)`
/// of the tile at `base + c * stride + r`. The tile holds the upper
/// block triangle, so the lower-numbered entity walks the rows. With
/// `ASSIGN` the whole tile is written; without, the block is added to
/// what the tile holds.
#[inline]
fn cross_tile<const ASSIGN: bool, const NA: usize, const NB: usize, const P: usize, T: crate::utils::Float, F: crate::utils::Float>(
    a: &[u32; NA], b: &[u32; NB], values: &[T; P], vals: &mut [F], base: usize, stride: usize,
) {
    let (sa, sb) = (tile_start(a) as usize, tile_start(b) as usize);
    if sa == sb {
        // Aliased: both slots index one entity, so the pairs land on a
        // diagonal tile and which side is the row flips per element.
        // Two pairs share a cell, so an assigning block clears the tile
        // and adds.
        if ASSIGN {
            let n = a.iter().filter(|&&g| g != u32::MAX).count();
            for c in 0..n {
                vals[base + c * stride..base + c * stride + n].fill(F::zero());
            }
        }
        for i in 0..NA {
            let gi = a[i];
            if gi == u32::MAX { continue; }
            let row = i * NB;
            for j in 0..NB {
                let gj = b[j];
                if gj == u32::MAX { continue; }
                let (lo, hi) = if gi <= gj { (gi, gj) } else { (gj, gi) };
                let val = F::from(values[row + j]).unwrap();
                // The triangle stores each symmetric pair once, so a pair
                // that lands on the diagonal needs both contributions.
                let val = if gi == gj { val + val } else { val };
                vals[base + (hi as usize - sa) * stride + (lo as usize - sa)] += val;
            }
        }
        return;
    }
    let (step_a, step_b) = if sa < sb { (1, stride) } else { (stride, 1) };
    for i in 0..NA {
        let gi = a[i];
        if gi == u32::MAX { continue; }
        let pa = base + (gi as usize - sa) * step_a;
        let row = i * NB;
        for j in 0..NB {
            let gj = b[j];
            if gj == u32::MAX { continue; }
            let val = F::from(values[row + j]).unwrap();
            let at = pa + (gj as usize - sb) * step_b;
            if ASSIGN {
                vals[at] = val;
            } else {
                vals[at] += val;
            }
        }
    }
}

/// The COO scatter of one block, upper triangle only.
#[inline]
fn cross_coo<const NA: usize, const NB: usize, const P: usize, T: crate::utils::Float, F: crate::utils::Float>(
    a: &[u32; NA], b: &[u32; NB], values: &[T; P], coo: &mut crate::simple_lm::CooMatrix<F>,
) {
    for i in 0..NA {
        let gi = a[i];
        if gi == u32::MAX { continue; }
        let row = i * NB;
        for j in 0..NB {
            let gj = b[j];
            if gj == u32::MAX { continue; }
            let (lo, hi) = if gi <= gj { (gi, gj) } else { (gj, gi) };
            let val = F::from(values[row + j]).unwrap();
            // Aliased diagonal: see cross_band.
            let val = if gi == gj { val + val } else { val };
            coo.push(lo, hi, val);
        }
    }
}

/// The direct CSC scatter of one block, by position lookup.
#[inline]
fn cross_direct<const NA: usize, const NB: usize, const P: usize, T: crate::utils::Float, F: crate::utils::Float>(
    a: &[u32; NA], b: &[u32; NB], values: &[T; P], csc: &mut crate::simple_lm::CscMatrix<F>,
) {
    for i in 0..NA {
        let gi = a[i];
        if gi == u32::MAX { continue; }
        let row = i * NB;
        for j in 0..NB {
            let gj = b[j];
            if gj == u32::MAX { continue; }
            let (lo, hi) = if gi <= gj { (gi, gj) } else { (gj, gi) };
            let val = F::from(values[row + j]).unwrap();
            // Aliased diagonal: see cross_band.
            let val = if gi == gj { val + val } else { val };
            if let Some(pos) = csc.find_pos(lo as usize, hi as usize) {
                csc.vals[pos] += val;
            }
        }
    }
}

/// The scatter of one block through the per-entry positions.
#[inline]
fn cross_mapped<const NA: usize, const NB: usize, const P: usize, T: crate::utils::Float, F: crate::utils::Float>(
    a: &[u32; NA], b: &[u32; NB], values: &[T; P], vals: &mut [F], positions: &[ValueIndex], cursor: &mut usize,
) {
    for i in 0..NA {
        let gi = a[i];
        if gi == u32::MAX { continue; }
        let row = i * NB;
        for j in 0..NB {
            let gj = b[j];
            if gj == u32::MAX { continue; }
            let val = F::from(values[row + j]).unwrap();
            // Aliased diagonal: see cross_band.
            let val = if gi == gj { val + val } else { val };
            vals[take_position(positions, cursor)] += val;
        }
    }
}

/// A list of cross blocks with the indices apart from the values: per
/// block the A and B indices and the tile position in one array, the
/// `NA x NB` values in one flat array. A block store holds its cross
/// blocks this way. The build pushes the indices only and
/// then calls [`finish`](Self::finish), which sizes the value array in
/// one zeroed allocation (never written by the build; the sweep zeroes
/// each block before it writes it) and keeps it when the length has not
/// changed.
#[derive(Clone)]
pub struct CrossBlockArray<const NA: usize, const NB: usize, const P: usize, T: crate::utils::Float = f64> {
    a: std::vec::Vec<[u32; NA]>,
    b: std::vec::Vec<[u32; NB]>,
    values: std::vec::Vec<T>,
    /// The global slot this array's first block holds. A range of the walk
    /// is a contiguous run of the numbering, so an array covering one runs
    /// from here; zero on an array covering all of them.
    base: u32,
}

impl<const NA: usize, const NB: usize, const P: usize, T: crate::utils::Float> Default for CrossBlockArray<NA, NB, P, T> {
    fn default() -> Self { Self::new() }
}

/// One block of a [`CrossBlockArray`] list, borrowed for the sweep's writes.
pub struct CrossBlockMut<'a, const NA: usize, const NB: usize, const P: usize, T: crate::utils::Float> {
    values: &'a mut [T; P],
}

impl<const NA: usize, const NB: usize, const P: usize, T: crate::utils::Float> CrossBlockMut<'_, NA, NB, P, T> {
    /// Reset the values to zero.
    #[inline]
    pub fn zero(&mut self) {
        self.values.fill(T::zero());
    }

    /// Add one residual's cross pairs `2 dr_a dr_b^T` into the tile. The
    /// residual itself belongs to the two entities' gradients, not here.
    #[inline]
    pub fn add_residual_cross(&mut self, dr_a: &[T; NA], dr_b: &[T; NB]) {
        cross_add(self.values, T::two(), dr_a, dr_b);
    }

    /// [`add_residual_cross`](Self::add_residual_cross) with the pairs
    /// scaled by the loss weight `w`. `w = 1` is the plain form.
    #[inline]
    pub fn add_residual_cross_with_loss(&mut self, w: T, dr_a: &[T; NA], dr_b: &[T; NB]) {
        cross_add(self.values, T::two() * w, dr_a, dr_b);
    }
}

impl<const NA: usize, const NB: usize, const P: usize, T: crate::utils::Float> CrossBlockArray<NA, NB, P, T> {
    const CHECK_P: () = assert!(P == NA * NB, "CrossBlockArray: P must equal NA*NB");

    /// An empty list.
    pub fn new() -> Self {
        let () = Self::CHECK_P;
        CrossBlockArray {
            a: std::vec::Vec::new(),
            b: std::vec::Vec::new(),
            values: std::vec::Vec::new(),
            base: 0,
        }
    }

    /// The number of blocks.
    pub fn len(&self) -> usize { self.a.len() }

    /// Bytes one block takes: its tile and its two index rows.
    pub const BLOCK_BYTES: usize = P * std::mem::size_of::<T>() + (NA + NB) * 4;

    /// The blocks held and the bytes of the arrays.
    pub fn footprint(&self) -> StoreFootprint {
        StoreFootprint {
            self_blocks: 0,
            cross_blocks: self.a.len(),
            coo_entries: 0,
            bytes: self.values.len() * std::mem::size_of::<T>() + self.a.len() * NA * 4 + self.b.len() * NB * 4,
        }
    }

    /// True if the list holds no block.
    pub fn is_empty(&self) -> bool { self.a.is_empty() }

    /// Drop every block's indices; the value storage stays for `finish`.
    pub fn clear(&mut self) {
        self.a.clear();
        self.b.clear();
    }

    /// Room for `n` more blocks' indices.
    pub fn reserve(&mut self, n: usize) {
        self.a.reserve(n);
        self.b.reserve(n);
    }

    /// Append a block with its parameter indices, unbound.
    #[inline]
    pub fn push(&mut self, a: &[u32; NA], b: &[u32; NB]) {
        self.a.push(*a);
        self.b.push(*b);
    }

    /// Size the value array to the blocks pushed: a fresh zeroed
    /// allocation when the length changed, else the existing storage,
    /// whose stale values the sweep overwrites.
    pub fn finish(&mut self) {
        let n = self.a.len() * P;
        if self.values.len() != n {
            self.values = vec![T::zero(); n];
        }
    }

    /// Reset every tile.
    pub fn zero(&mut self) {
        self.values.fill(T::zero());
    }

    /// The block at the marker's global `slot`, for writing. An array
    /// covering one run of the walk (`split`) opens on its
    /// [`base`](Self::base); a whole one is addressed by the slot as it
    /// stands. The sweep decides `split` once from its range table, as
    /// for [`SelfBlockArray::slab_of`].
    #[inline]
    pub fn block_mut(&mut self, split: bool, slot: usize) -> CrossBlockMut<'_, NA, NB, P, T> {
        let k = if split { slot - self.base as usize } else { slot };
        let values: &mut [T; P] = (&mut self.values[k * P..(k + 1) * P]).try_into().unwrap();
        CrossBlockMut { values }
    }

    /// The global slot this array opens on.
    pub fn base(&self) -> u32 { self.base }

    /// Set the global slot this array opens on.
    pub fn set_base(&mut self, base: u32) { self.base = base; }

    /// The indices of block `k`.
    pub fn indices(&self, k: usize) -> (&[u32; NA], &[u32; NB]) {
        (&self.a[k], &self.b[k])
    }

    /// The values of block `k`, row-major `NA x NB`.
    pub fn values(&self, k: usize) -> &[T] {
        &self.values[k * P..(k + 1) * P]
    }

    fn block(&self, k: usize) -> &[T; P] {
        (&self.values[k * P..(k + 1) * P]).try_into().unwrap()
    }

    /// Append one representative scalar coordinate of each tile, over
    /// every block. Structure only, no values.
    pub fn collect_hessian_cells(&self, out: &mut std::vec::Vec<(u32, u32)>) {
        for k in 0..self.a.len() {
            cross_cells(&self.a[k], &self.b[k], out);
        }
    }

    /// Record where each tile scatters in the assembled value buffer, over
    /// every block, as array `array` of store `store`. See
    /// [`LmProblemInternals::bind_hessian_positions`](crate::simple_lm::LmProblemInternals::bind_hessian_positions).
    pub fn bind_hessian_positions(&self, binder: &mut HessianBinder, out: &mut PositionStream, store: usize, array: usize) {
        out.reserve_tiles(self.a.len());
        for k in 0..self.a.len() {
            cross_bind(&self.a[k], &self.b[k], binder, out, store, array, k);
        }
    }

    /// Write block `k` over its tile, which lies in `vals` from `base`
    /// with its columns `stride` apart (see [`TileSource`]).
    #[inline]
    pub fn assign_block<F: crate::utils::Float>(&self, k: usize, vals: &mut [F], base: usize, stride: usize) {
        cross_tile::<true, NA, NB, P, _, _>(&self.a[k], &self.b[k], self.block(k), vals, base, stride);
    }

    /// Add block `k` to what its tile holds.
    #[inline]
    pub fn add_block<F: crate::utils::Float>(&self, k: usize, vals: &mut [F], base: usize, stride: usize) {
        cross_tile::<false, NA, NB, P, _, _>(&self.a[k], &self.b[k], self.block(k), vals, base, stride);
    }

    /// Add every tile into the dense symmetric Hessian, at `(A, B)` and
    /// its transpose.
    pub fn accumulate_hessian<F: crate::utils::Float>(&self, hessian: &mut [F]) {
        for k in 0..self.a.len() {
            cross_dense(&self.a[k], &self.b[k], self.block(k), hessian);
        }
    }

    /// Add every tile into an upper-band Hessian of half-bandwidth `kd`
    /// (LAPACK layout). Errors when an entry falls outside the band.
    pub fn accumulate_hessian_band<F: crate::utils::Float>(&self, band: &mut [F], kd: usize)
        -> Result<(), crate::simple_lm::BandOverflow>
    {
        for k in 0..self.a.len() {
            cross_band(&self.a[k], &self.b[k], self.block(k), band, kd)?;
        }
        Ok(())
    }

    /// Push every tile into a COO Hessian as `(row, col, value)` triplets.
    pub fn accumulate_hessian_sparse<F: crate::utils::Float>(&self, coo: &mut crate::simple_lm::CooMatrix<F>) {
        for k in 0..self.a.len() {
            cross_coo(&self.a[k], &self.b[k], self.block(k), coo);
        }
    }

    /// Add every tile straight into an already-patterned CSC Hessian,
    /// looking each entry up by column.
    pub fn accumulate_hessian_sparse_direct<F: crate::utils::Float>(&self, csc: &mut crate::simple_lm::CscMatrix<F>) {
        for k in 0..self.a.len() {
            cross_direct(&self.a[k], &self.b[k], self.block(k), csc);
        }
    }

    /// Add every block into the assembled value buffer through the
    /// per-entry positions [`bind_hessian_positions`](Self::bind_hessian_positions)
    /// recorded against a pattern without tiles, reading them in the same
    /// order.
    pub fn accumulate_hessian_sparse_indexed<F: crate::utils::Float>(&self, vals: &mut [F], positions: &[ValueIndex], cursor: &mut usize) {
        for k in 0..self.a.len() {
            cross_mapped(&self.a[k], &self.b[k], self.block(k), vals, positions, cursor);
        }
    }
}



/// Expand an upper-band matrix (LAPACK layout, `ldab = kd + 1`) into a
/// full dense symmetric `n * n` matrix. For the tests of the band walks.
#[cfg(test)]
pub(crate) fn densify_band(band: &[f64], n: usize, kd: usize) -> Vec<f64> {
    let ldab = kd + 1;
    let mut full = vec![0.0; n * n];
    for j in 0..n {
        for i in j.saturating_sub(kd)..=j {
            let v = band[(kd + i - j) + j * ldab];
            full[i * n + j] = v;
            full[j * n + i] = v;
        }
    }
    full
}


#[cfg(test)]
mod tests {
    use super::*;

    // The property every later stage rests on: a set of contiguous slot
    // ranges covering the container yields the full walk, in order, once
    // each -- holes and container kind notwithstanding.
    fn ranges_partition_the_walk<C: Leaves>(c: &C, whole: &[i32])
    where C::Item: PartialEq<i32> + std::fmt::Debug + Copy + Into<i32> {
        let n = c.count() as u32;
        let got: std::vec::Vec<i32> = c.range_iter(0, n).map(|x| (*x).into()).collect();
        assert_eq!(got, whole, "0..count is the whole walk");
        for cuts in [vec![], vec![n / 2], vec![n / 3, 2 * n / 3], (0..=n).collect()] {
            let mut bounds = vec![0u32];
            bounds.extend(cuts);
            bounds.push(n);
            let mut seen: std::vec::Vec<i32> = std::vec::Vec::new();
            for w in bounds.windows(2) {
                seen.extend(c.range_iter(w[0], w[1]).map(|x| (*x).into()));
            }
            assert_eq!(seen, whole, "ranges {:?} must rebuild the walk", bounds);
        }
    }

    #[test]
    fn an_unbuilt_cut_is_the_whole_model() {
        let c = Cut::new();
        assert!(c.is_empty());
        // Every store reads the empty table, which every walk takes as all
        // of its container.
        assert!(c.store(0).is_empty());
        assert_eq!(walk_range(c.store(0), 0), (0, u32::MAX));
    }

    #[test]
    fn a_cut_hands_each_store_its_own_row() {
        let mut c = Cut::new();
        c.reset(3, 2);
        assert!(!c.is_empty());
        assert_eq!((c.stores(), c.walks()), (3, 2));
        // Two walks of 100 slots, cut three ways.
        for (s, (lo, hi)) in [(0usize, (0u32, 34u32)), (1, (34, 67)), (2, (67, 100))] {
            c.set(s, 0, lo, hi);
            c.set(s, 1, lo, hi);
        }
        assert_eq!(c.store(0), &[(0, 34), (0, 34)]);
        assert_eq!(c.store(2), &[(67, 100), (67, 100)]);
        assert_eq!(walk_range(c.store(1), 1), (34, 67));
        // The rows together cover each walk once, which is what keeps a
        // contribution from being counted twice or dropped.
        let mut covered: std::vec::Vec<(u32, u32)> =
            (0..3).map(|s| walk_range(c.store(s), 0)).collect();
        covered.sort();
        assert_eq!(covered, vec![(0, 34), (34, 67), (67, 100)]);
    }

    #[derive(Clone, Default, Debug, PartialEq)]
    struct TestStore { tag: u32 }
    impl BlockStore for TestStore {
        fn footprint(&self) -> StoreFootprint { StoreFootprint::default() }
    }

    #[test]
    fn the_store_list_keeps_its_stores_and_uses_the_front() {
        let mut ctx = Context::new();
        for (i, s) in ctx.stores_mut::<TestStore>(4).0.iter_mut().enumerate() {
            s.tag = i as u32 + 1;
        }
        assert_eq!(ctx.blocks_list::<TestStore>().unwrap().len(), 4);

        // A solve wanting fewer stores uses the front of the list.
        assert_eq!(ctx.stores_mut::<TestStore>(1).0.len(), 1);
        assert_eq!(ctx.blocks_list::<TestStore>().unwrap().len(), 1);
        assert_eq!(ctx.blocks_list::<TestStore>().unwrap()[0].tag, 1);

        // The spares were kept, not dropped and rebuilt: growing back finds
        // the same stores, which is what keeps their allocations.
        let back: std::vec::Vec<u32> =
            ctx.stores_mut::<TestStore>(4).0.iter().map(|s| s.tag).collect();
        assert_eq!(back, vec![1, 2, 3, 4]);
    }

    #[test]
    fn the_sweep_parts_keep_the_count_the_solve_began_with() {
        let mut ctx = Context::new();
        ctx.stores_mut::<TestStore>(3).0[2].tag = 7;
        // A sweep reads the active list; it must not narrow it.
        let (stores, cut, _timing) = ctx.sweep_parts_mut::<TestStore>();
        assert_eq!(stores.len(), 3);
        assert_eq!(stores[2].tag, 7);
        assert!(cut.is_empty(), "no cut was built");
        assert_eq!(ctx.blocks_list::<TestStore>().unwrap().len(), 3);
        // On a context that holds nothing, the parts are one empty store.
        let mut fresh = Context::new();
        assert_eq!(fresh.sweep_parts_mut::<TestStore>().0.len(), 1);
    }

    #[test]
    fn another_roots_store_starts_over() {
        #[derive(Clone, Default)]
        struct OtherStore;
        impl BlockStore for OtherStore {
            fn footprint(&self) -> StoreFootprint { StoreFootprint::default() }
        }

        let mut ctx = Context::new();
        ctx.stores_mut::<TestStore>(2).0[1].tag = 5;
        assert!(ctx.blocks_list::<OtherStore>().is_none(), "a different root holds none");
        assert_eq!(ctx.stores_mut::<OtherStore>(2).0.len(), 2);
        // Taking it for one root drops the other's, as a context serves one.
        assert!(ctx.blocks_list::<TestStore>().is_none());
    }

    #[test]
    fn range_iter_partitions_every_container() {
        let v: std::vec::Vec<i32> = vec![1, 2, 3, 4, 5];
        ranges_partition_the_walk(&v, &[1, 2, 3, 4, 5]);

        let rv = crate::refs::Vec::from_vec(vec![1, 2, 3, 4, 5]);
        ranges_partition_the_walk(&rv, &[1, 2, 3, 4, 5]);

        let mut dq: crate::refs::Deque<i32> = crate::refs::Deque::new();
        for x in [2, 3, 4, 5] { dq.push_back(x); }
        dq.push_front(1);
        ranges_partition_the_walk(&dq, &[1, 2, 3, 4, 5]);

        // An arena with holes: the slot count spans them, the walk does not.
        let mut ar: crate::refs::Arena<i32> = crate::refs::Arena::with_block_size(2);
        let r: std::vec::Vec<_> = (1..=7).map(|x| ar.push(x)).collect();
        ar.remove(r[1]);
        ar.remove(r[5]);
        ranges_partition_the_walk(&ar, &[1, 3, 4, 5, 7]);

        let some: Option<i32> = Some(9);
        ranges_partition_the_walk(&some, &[9]);
        let none: Option<i32> = None;
        ranges_partition_the_walk(&none, &[]);
    }

    #[test]
    fn cut_covers_the_leaves_in_order() {
        for &(n, p) in &[(0usize, 1usize), (1, 4), (3, 4), (4, 4), (10, 3), (204_472, 4)] {
            assert_eq!(cut(n, p, 0), 0);
            assert_eq!(cut(n, p, p), n);
            let lens: std::vec::Vec<usize> = (0..p).map(|t| cut(n, p, t + 1) - cut(n, p, t)).collect();
            let (lo, hi) = (lens.iter().min().unwrap(), lens.iter().max().unwrap());
            assert!(hi - lo <= 1, "n {} p {}: {:?}", n, p, lens);
        }
    }

    #[test]
    fn a_clock_off_reads_nothing() {
        let off = Clock { on: false };
        assert!(off.start().is_none());
        assert_eq!(off.stop(None), Duration::ZERO);
        let on = Clock { on: true };
        let t = on.start();
        assert!(t.is_some());
        let _ = on.stop(t);
        let mut ctx = Context::new();
        assert!(!ctx.timing());
        ctx.set_timing(true);
        assert!(ctx.clone().timing());
    }

    // A slab holding only some entities reaches them through the map, and
    // must accumulate exactly what the whole slab does for those it holds.
    #[test]
    fn a_mapped_slab_matches_the_whole_one() {
        // Whole: four entities, the marker's slot addressing the data.
        let mut whole: SelfBlockArray<3, 6, f64> = SelfBlockArray::new();
        for e in 0..4u32 { whole.push(e, &[3 * e, 3 * e + 1, 3 * e + 2]); }
        whole.finish();

        // Partial: entities 1 and 3 only, in the order first met.
        let mut part: SelfBlockArray<3, 6, f64> = SelfBlockArray::new();
        part.map_resize(4);
        part.push_at(3, 3, &[9, 10, 11]);
        part.push_at(1, 1, &[3, 4, 5]);
        part.finish();
        assert_eq!(part.slab_of(true, 3), 0);
        assert_eq!(part.slab_of(true, 1), 1);

        let rows = [(1usize, 0.5, [1.0, -0.25, 0.75]), (3, -0.4, [0.2, 1.5, -0.6]),
                    (1, 0.9, [-1.0, 0.3, 0.1])];
        for (slot, r, dr) in rows {
            whole.add_residual(false, slot, r, &dr);
            part.add_residual(true, slot, r, &dr);
        }
        for (slot, r, dr) in rows {
            whole.add_residual_with_loss(false, slot, 0.25, r, &dr);
            part.add_residual_with_loss(true, slot, 0.25, r, &dr);
        }

        let n = 12;
        let (mut a, mut b) = (vec![0.0; n * n], vec![0.0; n * n]);
        whole.accumulate_hessian(&mut a);
        part.accumulate_hessian(&mut b);
        assert_eq!(a, b, "the mapped slab's Hessian must be the whole one's");
        let (mut ga, mut gb) = (vec![0.0; n], vec![0.0; n]);
        whole.scatter_grad(&mut ga);
        part.scatter_grad(&mut gb);
        assert_eq!(ga, gb, "and its gradient too");
    }

    #[test]
    fn map_resize_keeps_what_was_claimed() {
        let mut arr: SelfBlockArray<3, 6, f64> = SelfBlockArray::new();
        arr.map_resize(4);
        arr.push_at(2, 2, &[6, 7, 8]);
        // Sizing again must not forget a claim: the build sizes once per
        // solve and the entries outlive it.
        arr.map_resize(4);
        arr.map_resize(2);
        assert_eq!(arr.slab_of(true, 2), 0);
    }

    // A cross array covering one run of the walk is addressed by the same
    // global slot the marker carries.
    #[test]
    fn a_based_cross_array_matches_the_whole_one() {
        let mut whole: CrossBlockArray<2, 2, 4, f64> = CrossBlockArray::new();
        for k in 0..4u32 { whole.push(&[2 * k, 2 * k + 1], &[8 + 2 * k, 9 + 2 * k]); }
        whole.finish();
        assert_eq!(whole.base(), 0);

        // The run [2, 4), which is what a thread owning that range holds.
        let mut run: CrossBlockArray<2, 2, 4, f64> = CrossBlockArray::new();
        for k in 2..4u32 { run.push(&[2 * k, 2 * k + 1], &[8 + 2 * k, 9 + 2 * k]); }
        run.finish();
        run.set_base(2);

        for slot in 2..4usize {
            let dr_a = [1.0 + slot as f64, -0.5];
            let dr_b = [0.25, 2.0 - slot as f64];
            whole.block_mut(false, slot).add_residual_cross(&dr_a, &dr_b);
            run.block_mut(true, slot).add_residual_cross(&dr_a, &dr_b);
        }
        assert_eq!(whole.values(2), run.values(0), "slot 2 is the run's first block");
        assert_eq!(whole.values(3), run.values(1));
    }

    // The property a cut has to satisfy: sweeping disjoint ranges into
    // separate stores and summing them gives what one store sweeping the
    // whole walk gives. This drives the arrays the way the generated sweep
    // does -- the entity's marker slot for a self write, the instance's for
    // a cross one -- so a base or a map that is wrong shows up here rather
    // than as a wrong Hessian in a threaded solve.
    //
    // Instance i couples two of six entities; entities 0, 2, 3 and 4 are
    // reached from both halves, which is the case where two slabs land on
    // the same Hessian tile.
    const PAIRS: [(usize, usize); 8] =
        [(0, 1), (1, 2), (2, 3), (3, 4), (3, 4), (4, 5), (5, 0), (0, 2)];

    fn residual(i: usize) -> (f64, [f64; 2], [f64; 2]) {
        (0.1 * (i as f64 + 1.0), [1.0 + i as f64, -0.5], [0.25, 2.0 - i as f64])
    }

    /// Sweep instances `[lo, hi)` into one store. `split` says whether the
    /// store holds a slice of the walk (a thread's) or all of it; the
    /// writes are the same either way.
    fn sweep(
        split: bool,
        lo: usize, hi: usize,
        hess: &mut [f64], grad: &mut [f64],
    ) {
        let mut selfs: SelfBlockArray<2, 3, f64> = SelfBlockArray::new();
        let mut cross: CrossBlockArray<2, 2, 4, f64> = CrossBlockArray::new();
        if split {
            // The build claims a slab place the first time this range meets
            // an entity, and the cross array opens on the range's first slot.
            selfs.map_resize(6);
            let mut seen = [false; 6];
            for i in lo..hi {
                for e in [PAIRS[i].0, PAIRS[i].1] {
                    if !seen[e] {
                        seen[e] = true;
                        selfs.push_at(e as u32, e as u32, &[2 * e as u32, 2 * e as u32 + 1]);
                    }
                }
            }
            for i in lo..hi {
                let (a, b) = PAIRS[i];
                cross.push(&[2 * a as u32, 2 * a as u32 + 1], &[2 * b as u32, 2 * b as u32 + 1]);
            }
            cross.set_base(lo as u32);
        } else {
            for e in 0..6usize {
                selfs.push(e as u32, &[2 * e as u32, 2 * e as u32 + 1]);
            }
            for &(a, b) in PAIRS.iter() {
                cross.push(&[2 * a as u32, 2 * a as u32 + 1], &[2 * b as u32, 2 * b as u32 + 1]);
            }
        }
        selfs.finish();
        cross.finish();

        for i in lo..hi {
            let (a, b) = PAIRS[i];
            let (r, dr_a, dr_b) = residual(i);
            // The slot is the marker's either way; the array resolves it.
            selfs.add_residual(split, a, r, &dr_a);
            selfs.add_residual(split, b, r, &dr_b);
            cross.block_mut(split, i).add_residual_cross(&dr_a, &dr_b);
        }
        selfs.accumulate_hessian(hess);
        cross.accumulate_hessian(hess);
        selfs.scatter_grad(grad);
    }

    #[test]
    fn disjoint_ranges_sum_to_the_whole_walk() {
        let n = 12;
        let (mut hw, mut gw) = (vec![0.0; n * n], vec![0.0; n]);
        sweep(false, 0, 8, &mut hw, &mut gw);

        // Every way of cutting the walk in two must rebuild it.
        for cut in 1..8usize {
            let (mut h, mut g) = (vec![0.0; n * n], vec![0.0; n]);
            sweep(true,0, cut, &mut h, &mut g);
            sweep(true,cut, 8, &mut h, &mut g);
            for k in 0..n * n {
                assert!((h[k] - hw[k]).abs() < 1e-12,
                    "cut at {}: hessian[{}] {} vs whole {}", cut, k, h[k], hw[k]);
            }
            for k in 0..n {
                assert!((g[k] - gw[k]).abs() < 1e-12,
                    "cut at {}: grad[{}] {} vs whole {}", cut, k, g[k], gw[k]);
            }
        }
    }

    #[test]
    fn three_ranges_sum_to_the_whole_walk() {
        let n = 12;
        let (mut hw, mut gw) = (vec![0.0; n * n], vec![0.0; n]);
        sweep(false, 0, 8, &mut hw, &mut gw);
        let (mut h, mut g) = (vec![0.0; n * n], vec![0.0; n]);
        for (lo, hi) in [(0, 3), (3, 3), (3, 6), (6, 8)] {   // one empty range
            sweep(true,lo, hi, &mut h, &mut g);
        }
        for k in 0..n * n {
            assert!((h[k] - hw[k]).abs() < 1e-12, "hessian[{}] {} vs {}", k, h[k], hw[k]);
        }
        for k in 0..n {
            assert!((g[k] - gw[k]).abs() < 1e-12, "grad[{}] {} vs {}", k, g[k], gw[k]);
        }
    }

    #[test]
    fn self_blocks_band_matches_dense() {
        let n = 4;
        let kd = 2;
        let mut arr: SelfBlockArray<3, 6, f64> = SelfBlockArray::new();
        arr.push(0, &[0, 1, 2]);
        arr.finish();
        arr.add_residual(false, 0, 0.3, &[1.0, 0.5, -0.25]);
        arr.add_residual(false, 0, -0.7, &[0.2, -1.5, 0.75]);

        let mut dense = vec![0.0; n * n];
        arr.accumulate_hessian(&mut dense);
        let mut band = vec![0.0; (kd + 1) * n];
        arr.accumulate_hessian_band(&mut band, kd).unwrap();
        assert_eq!(densify_band(&band, n, kd), dense);
    }

    #[test]
    fn cross_blocks_band_matches_dense() {
        let n = 5;
        let kd = 3;
        let mut arr: CrossBlockArray<2, 2, 4, f64> = CrossBlockArray::new();
        arr.push(&[0, 1], &[2, 3]);
        arr.finish();
        arr.block_mut(false, 0).add_residual_cross(&[1.0, -0.5], &[0.25, 2.0]);
        arr.block_mut(false, 0).add_residual_cross(&[0.3, 0.7], &[-0.6, 0.1]);

        let mut dense = vec![0.0; n * n];
        arr.accumulate_hessian(&mut dense);
        let mut band = vec![0.0; (kd + 1) * n];
        arr.accumulate_hessian_band(&mut band, kd).unwrap();
        assert_eq!(densify_band(&band, n, kd), dense);
    }

    // A hand-written store for the tiled scatter: one self array of
    // 2-wide entities and one cross array over pairs of them.
    #[derive(Default)]
    struct TwoArrays {
        selfs: SelfBlockArray<2, 3, f64>,
        cross: CrossBlockArray<2, 2, 4, f64>,
    }

    impl TileSource for TwoArrays {
        fn assign_block<F: crate::utils::Float>(&self, array: u8, index: u32, vals: &mut [F], base: usize, stride: usize) {
            match array {
                0 => self.selfs.assign_block(index as usize, vals, base, stride),
                1 => self.cross.assign_block(index as usize, vals, base, stride),
                _ => unreachable!(),
            }
        }
        fn add_block<F: crate::utils::Float>(&self, array: u8, index: u32, vals: &mut [F], base: usize, stride: usize) {
            match array {
                0 => self.selfs.add_block(index as usize, vals, base, stride),
                1 => self.cross.add_block(index as usize, vals, base, stride),
                _ => unreachable!(),
            }
        }
    }

    /// Three 2-wide entities over a dense 6 by 6 column-major buffer:
    /// tile `(bi, bj)` at `12 bj + 2 bi`, its columns 6 apart, in the
    /// panel of block column `bj` at `12 bj`.
    const N6: usize = 6;
    fn dense_tiles(i: u32, j: u32) -> (usize, usize, usize) {
        ((j as usize) * N6 + i as usize, N6, (j as usize / 2) * 2 * N6)
    }

    /// Two stores that share entity 1: self blocks for entities 0 and 1
    /// in the first, 1 and 2 in the second; cross blocks on the pair
    /// (0, 1) in both, one aliased on entity 1 in the first, one with the
    /// higher entity on its A side in the second.
    fn two_stores() -> std::vec::Vec<TwoArrays> {
        let mut s0 = TwoArrays::default();
        s0.selfs.push(0, &[0, 1]);
        s0.selfs.push(1, &[2, 3]);
        s0.selfs.finish();
        s0.selfs.add_residual(false, 0, 0.5, &[1.0, -0.25]);
        s0.selfs.add_residual(false, 1, -0.4, &[0.2, 1.5]);
        s0.cross.push(&[0, 1], &[2, 3]);
        s0.cross.push(&[2, 3], &[2, 3]);
        s0.cross.finish();
        s0.cross.block_mut(false, 0).add_residual_cross(&[1.0, -0.5], &[0.25, 2.0]);
        s0.cross.block_mut(false, 1).add_residual_cross(&[0.3, 0.7], &[-0.6, 0.1]);

        let mut s1 = TwoArrays::default();
        s1.selfs.push(1, &[2, 3]);
        s1.selfs.push(2, &[4, 5]);
        s1.selfs.finish();
        s1.selfs.add_residual(false, 0, 0.9, &[-1.0, 0.3]);
        s1.selfs.add_residual(false, 1, 0.1, &[0.4, -0.8]);
        s1.cross.push(&[0, 1], &[2, 3]);
        s1.cross.push(&[4, 5], &[2, 3]);
        s1.cross.finish();
        s1.cross.block_mut(false, 0).add_residual_cross(&[-0.2, 0.9], &[1.1, 0.4]);
        s1.cross.block_mut(false, 1).add_residual_cross(&[0.6, -0.3], &[0.8, -1.2]);
        vec![s0, s1]
    }

    fn bind_two_stores(stores: &[TwoArrays], chunks: usize) -> PositionStream {
        let mut stream = PositionStream::new();
        for (s, store) in stores.iter().enumerate() {
            store.selfs.bind_hessian_positions(&mut HessianBinder::Tiled(&mut dense_tiles), &mut stream, s, 0);
            store.cross.bind_hessian_positions(&mut HessianBinder::Tiled(&mut dense_tiles), &mut stream, s, 1);
        }
        stream.finish(chunks);
        stream
    }

    /// What the tiled scatter must leave in the buffer: the upper
    /// triangle of the dense sum on every cell of a tile some block
    /// binds to, zero on the lower triangle of a diagonal tile, and
    /// `garbage` where no tile lies.
    fn expected_tiles(stores: &[TwoArrays], garbage: f64) -> std::vec::Vec<f64> {
        let mut dense = vec![0.0; N6 * N6];
        for s in stores {
            s.selfs.accumulate_hessian(&mut dense);
            s.cross.accumulate_hessian(&mut dense);
        }
        let mut out = vec![garbage; N6 * N6];
        for (bi, bj) in [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2)] {
            for i in 2 * bi..2 * bi + 2 {
                for j in 2 * bj..2 * bj + 2 {
                    out[j * N6 + i] = if i <= j { dense[i * N6 + j] } else { 0.0 };
                }
            }
        }
        out
    }

    /// The buffer against the expectation: a cell no tile writes and the
    /// lower triangle of a diagonal tile exactly, a summed cell to
    /// rounding (the dense reference adds an aliased pair and its
    /// transpose one at a time, the tile adds them doubled).
    fn assert_tiles(vals: &[f64], expected: &[f64]) {
        assert_eq!(vals.len(), expected.len());
        for (k, (v, e)) in std::iter::zip(vals, expected).enumerate() {
            assert!((v - e).abs() <= 1e-14 * (1.0 + e.abs()), "cell {}: {} vs {}", k, v, e);
        }
    }

    #[test]
    fn the_tiled_scatter_sums_each_tile_once_from_every_store() {
        let stores = two_stores();
        let stream = bind_two_stores(&stores, 2);
        assert!(stream.tiled());
        assert_eq!(stream.tiles(), 8, "every block binds");
        assert_eq!(stream.chunks(), 2);
        assert!(!stream.covers(N6 * N6), "five tiles of four cells do not cover the buffer");
        assert!(stream.positions().is_empty());
        let mut vals = vec![9.0; N6 * N6];
        scatter_tiles(&stores, &stream, &mut vals);
        assert_tiles(&vals, &expected_tiles(&stores, 9.0));
    }

    #[test]
    fn the_chunks_cut_at_panels_and_may_be_empty() {
        let mut stores = two_stores();
        // A third store with nothing in it. Eight entries over five tiles
        // in three panels: the first cut, wanted after entry 3, moves to
        // the panel boundary after entry 6, and the second, wanted after
        // entry 6, is already there, so the middle chunk is empty.
        stores.push(TwoArrays::default());
        let stream = bind_two_stores(&stores, 3);
        assert_eq!(stream.chunks(), 3);
        let panel2 = 2 * N6 as ValueIndex * 2;
        assert_eq!(stream.chunks[0], (0, 0));
        assert_eq!(stream.chunks[1], (6, panel2), "the cut after the tiles of block column 1");
        assert_eq!(stream.chunks[2], (6, panel2), "empty");
        assert_eq!(stream.chunks[3], (8, ValueIndex::MAX));
        let mut vals = vec![-1.0; N6 * N6];
        scatter_tiles(&stores, &stream, &mut vals);
        assert_tiles(&vals, &expected_tiles(&stores, -1.0));
    }

    /// An aliased cross block as the first writer of a diagonal tile:
    /// two of its pairs share a cell, so it clears the tile and adds, and
    /// the tile's lower triangle comes out zero whatever the buffer held.
    #[test]
    fn an_aliased_block_assigns_its_whole_tile() {
        let mut s = TwoArrays::default();
        s.selfs.finish();
        s.cross.push(&[2, 3], &[2, 3]);
        s.cross.finish();
        s.cross.block_mut(false, 0).add_residual_cross(&[0.3, 0.7], &[-0.6, 0.1]);
        let stores = vec![s];
        let stream = bind_two_stores(&stores, 1);
        assert_eq!(stream.tiles(), 1);
        let mut vals = vec![5.0; N6 * N6];
        scatter_tiles(&stores, &stream, &mut vals);
        // values = 2 dr_a dr_b^T, row-major over (a, b).
        let v = [2.0 * 0.3 * -0.6, 2.0 * 0.3 * 0.1, 2.0 * 0.7 * -0.6, 2.0 * 0.7 * 0.1];
        let mut expected = vec![5.0; N6 * N6];
        expected[2 * N6 + 2] = v[0] + v[0];
        expected[3 * N6 + 2] = v[1] + v[2];
        expected[2 * N6 + 3] = 0.0;
        expected[3 * N6 + 3] = v[3] + v[3];
        assert_tiles(&vals, &expected);
    }

    #[test]
    #[should_panic(expected = "one store count")]
    fn a_stream_scatters_from_the_store_count_it_was_bound_at() {
        let stores = two_stores();
        let stream = bind_two_stores(&stores, 2);
        let mut vals = vec![0.0; N6 * N6];
        scatter_tiles(&stores[..1], &stream, &mut vals);
    }

    #[test]
    fn a_pattern_without_tiles_binds_one_position_per_entry() {
        let stores = two_stores();
        let mut stream = PositionStream::new();
        let mut scalar = |i: u32, j: u32| dense_tiles(i, j).0;
        for (s, store) in stores.iter().enumerate() {
            store.selfs.bind_hessian_positions(&mut HessianBinder::Scalar(&mut scalar), &mut stream, s, 0);
            store.cross.bind_hessian_positions(&mut HessianBinder::Scalar(&mut scalar), &mut stream, s, 1);
        }
        stream.finish(2);
        assert!(!stream.tiled());
        assert_eq!(stream.tiles(), 0);
        assert_eq!(stream.positions().len(), 4 * 3 + 4 * 4, "three per triangle, four per cross block");
        assert!(!stream.covers(N6 * N6));
        assert!(!stream.covers(0), "a pattern without tiles writes nothing on its own");
        let map = PositionStream::from_map(&[3, 1, 2]);
        assert_eq!(map[1], 1);
        assert_eq!(map.positions(), &[3, 1, 2]);
    }

    #[test]
    #[should_panic(expected = "form one run")]
    fn a_gap_in_an_entitys_live_indices_is_refused() {
        tile_extent(&[0, u32::MAX, 2]);
    }

    #[test]
    fn tile_extent_spans_the_live_run() {
        assert_eq!(tile_extent(&[u32::MAX, 4, 5, 6]), (4, 3));
        assert_eq!(tile_extent(&[7, u32::MAX, u32::MAX]), (7, 1));
        assert_eq!(tile_extent(&[u32::MAX, u32::MAX]), (u32::MAX, 0));
    }
}
