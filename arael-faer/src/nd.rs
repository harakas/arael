//! Nested dissection: a fill-reducing ordering for matrices that have no band
//! and no small degrees to chew on.
//!
//! Cholesky fill spreads along paths in the matrix's graph. Nested dissection
//! cuts those paths: find a set of vertices whose removal splits the graph in
//! two, order each half first and the separator last, and no fill can ever
//! reach from one half to the other. Recurse.
//!
//! ```text
//! order(V) = order(A) ++ order(B) ++ S      S separates A from B
//! ```
//!
//! This is what bundle adjustment needs and minimum degree cannot give it: a
//! 3D point seen by k cameras makes a k-clique among them, and AMD drowns in
//! cliques. On the 1723-camera Ladybug problem AMD leaves 83.1M values in the
//! factor and faer takes 4.7 s over it; this ordering leaves 46.9M and takes
//! 2.3 s.
//!
//! It is NOT a general win, and the caller must know which matrix it has:
//!
//! * **banded** (a SLAM trajectory's reduced system) -- the natural order is
//!   already at the fill limit, and dissecting it is 3.4x SLOWER.
//! * **very sparse graphs** (a pose graph) -- AMD wins outright.
//! * **cliquey, no band** (bundle adjustment) -- this is the one.
//!
//! # Ordering the block graph, not the matrix
//!
//! [`NestedDissection::of_blocks`] dissects the graph of BLOCKS -- one node per
//! camera, not per parameter. Two reasons, both measured:
//!
//! * a block's parameters stay contiguous in the permutation, so the factor
//!   keeps its supernodes, and a separator is priced in the parameters it
//!   actually costs;
//! * the graph is 9x smaller, and the ordering runs in 50 ms instead of 916.
//!
//! # What to minimise
//!
//! Not fill. Cholesky flops go as the SUM OF SQUARES of the column heights, so
//! a smaller separator sitting in a worse place can cost MORE arithmetic --
//! measured: moving the cut to shrink the separator cut fill by 3% and raised
//! flops by 2%, and the factorization got slower. What pays is shrinking the
//! separator WITHOUT moving the cut, which is what the minimum vertex cover
//! does: it drops vertices the cut never needed.

use crate::bsc::SymbolicSparseBlockColMat;
use faer::Index;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::perm::PermRef;
use faer::sparse::SymbolicSparseColMat;

/// How hard to work at it.
#[derive(Clone, Copy, Debug)]
pub struct NdParams {
    /// Stop dissecting below this many nodes and order the rest with AMD.
    /// Barely matters: 16 / 64 / 256 measured 2398 / 2419 / 2333 ms on
    /// Ladybug-1723.
    pub leaf: usize,
    /// Passes of Fiduccia-Mattheyses refinement over each separator. 0 leaves
    /// the raw cut, and costs a lot: on Ladybug-1723 the factor grows from
    /// 39.2M values to 48.7M.
    pub fm_passes: usize,
    /// How lopsided a bisection may get, as a fraction of the subgraph's
    /// weight on the heavier side. Refinement will not push past it.
    pub max_imbalance: f64,
}

impl Default for NdParams {
    fn default() -> Self {
        NdParams { leaf: 128, fm_passes: 4, max_imbalance: 0.6 }
    }
}

/// An undirected graph in CSR, no self-loops, both directions stored.
#[derive(Clone, Debug)]
pub struct Graph {
    xadj: Vec<usize>,
    adj: Vec<usize>,
    /// what eliminating this node costs -- its scalar width. A separator's
    /// price is paid in parameters, not in nodes, so a 9-wide camera is worth
    /// nine times a 1-wide one.
    vwgt: Vec<usize>,
}

impl Graph {
    /// Build from an edge list. Duplicate and reversed edges are fine; self
    /// loops are dropped.
    pub fn from_edges(n: usize, edges: impl IntoIterator<Item = (usize, usize)>) -> Self {
        let mut e: Vec<(usize, usize)> = Vec::new();
        for (a, b) in edges {
            if a != b {
                e.push((a, b));
                e.push((b, a));
            }
        }
        e.sort_unstable();
        e.dedup();
        let mut xadj = vec![0usize; n + 1];
        for &(a, _) in &e {
            xadj[a + 1] += 1;
        }
        for i in 0..n {
            xadj[i + 1] += xadj[i];
        }
        let mut adj = vec![0usize; e.len()];
        let mut cur = xadj.clone();
        for &(a, b) in &e {
            adj[cur[a]] = b;
            cur[a] += 1;
        }
        Graph { xadj, adj, vwgt: vec![1; n] }
    }

    /// Give the nodes weights (their scalar widths). Length must be `nodes()`.
    pub fn with_weights(mut self, vwgt: Vec<usize>) -> Self {
        assert_eq!(vwgt.len(), self.nodes());
        self.vwgt = vwgt;
        self
    }

    fn w(&self, u: usize) -> usize {
        self.vwgt[u]
    }

    /// The graph of a symmetric block matrix: one node per block column, an
    /// edge wherever an off-diagonal tile couples two of them.
    pub fn of_blocks<I: Index>(sym: &SymbolicSparseBlockColMat<I>) -> Self {
        let n = sym.nblk_cols();
        let mut edges = Vec::new();
        for j in 0..n {
            for b in sym.col_range(j) {
                let i = sym.blk_row(b);
                if i != j {
                    edges.push((i, j));
                }
            }
        }
        let widths = (0..n).map(|j| sym.col_span(j).len()).collect();
        Graph::from_edges(n, edges).with_weights(widths)
    }

    pub fn nodes(&self) -> usize {
        self.xadj.len() - 1
    }

    pub fn neighbours(&self, u: usize) -> &[usize] {
        &self.adj[self.xadj[u]..self.xadj[u + 1]]
    }
}

/// Scratch reused across the whole recursion, so a dissection allocates once.
struct Scratch {
    /// which nodes belong to the subgraph being worked on
    inset: Vec<bool>,
    /// BFS bookkeeping
    seen: Vec<u32>,
    stamp: u32,
    queue: Vec<usize>,
    /// which half of the current bisection a node landed in (1 or 2)
    side: Vec<u8>,
    in_sep: Vec<bool>,
    /// FM refinement: where each node sits (A / B / SEP) and whether it has
    /// already been moved in this pass
    part: Vec<u8>,
    locked: Vec<bool>,
}

impl Scratch {
    fn new(n: usize) -> Self {
        Scratch {
            inset: vec![false; n],
            seen: vec![0; n],
            stamp: 0,
            queue: Vec::with_capacity(n),
            side: vec![0; n],
            in_sep: vec![false; n],
            part: vec![0; n],
            locked: vec![false; n],
        }
    }
}

const A: u8 = 0;
const B: u8 = 1;
const SEP: u8 = 2;

/// One vertex moved out of the separator, and the neighbours it dragged in.
struct Move {
    v: usize,
    to: u8,
    dragged: Vec<usize>,
}

/// Fiduccia-Mattheyses refinement of a vertex separator.
///
/// The move: take `v` out of the separator into side `T`. Every neighbour of
/// `v` that sat on the *other* side must then enter the separator, or `v`
/// would touch it directly and the separator would not separate. So
///
/// ```text
/// gain(v -> T) = w(v) - sum of w(u) for u in N(v) on the far side
/// ```
///
/// and a positive gain shrinks the separator. The point of FM -- and the
/// reason a greedy hill-climber is not FM -- is that it takes the best move
/// available even when the gain is NEGATIVE, keeps going, and afterwards rolls
/// back to the best separator it saw along the way. That is how it climbs out
/// of a local minimum: a chain of bad moves can open a much better cut.
///
/// Balance is a hard constraint, not a term in the objective: a perfectly thin
/// separator that leaves 99% of the graph on one side has dissected nothing.
fn refine(
    g: &Graph,
    nodes: &[usize],
    a: &mut Vec<usize>,
    b: &mut Vec<usize>,
    sep: &mut Vec<usize>,
    s: &mut Scratch,
    p: NdParams,
) {
    if sep.is_empty() || a.is_empty() || b.is_empty() {
        return;
    }
    for &u in a.iter() {
        s.part[u] = A;
    }
    for &u in b.iter() {
        s.part[u] = B;
    }
    for &u in sep.iter() {
        s.part[u] = SEP;
    }

    let total: usize = nodes.iter().map(|&u| g.w(u)).sum();
    let max_side = (total as f64 * p.max_imbalance) as usize;
    let mut wa: usize = a.iter().map(|&u| g.w(u)).sum();
    let mut wb: usize = b.iter().map(|&u| g.w(u)).sum();
    let mut wsep: usize = sep.iter().map(|&u| g.w(u)).sum();

    for _pass in 0..p.fm_passes {
        for &u in nodes {
            s.locked[u] = false;
        }
        let mut moves: Vec<Move> = Vec::new();
        let mut best_sep = wsep;
        let mut best_step = 0usize;
        let (start_wa, start_wb, start_wsep) = (wa, wb, wsep);

        loop {
            // Best move over the whole separator. The separator is small (it
            // is the point), so scanning it beats maintaining gain buckets.
            let mut best: Option<(isize, usize, u8)> = None;
            for k in 0..sep.len() {
                let v = sep[k];
                if s.part[v] != SEP || s.locked[v] {
                    continue;
                }
                for to in [A, B] {
                    let far = if to == A { B } else { A };
                    let drag: usize = g
                        .neighbours(v)
                        .iter()
                        .filter(|&&u| s.part[u] == far)
                        .map(|&u| g.w(u))
                        .sum();
                    // v joins `to`; the dragged neighbours leave `far` for SEP
                    let new_to = if to == A { wa } else { wb } + g.w(v);
                    if new_to > max_side {
                        continue;
                    }
                    let gain = g.w(v) as isize - drag as isize;
                    if best.is_none_or(|(gb, _, _)| gain > gb) {
                        best = Some((gain, v, to));
                    }
                }
            }
            let Some((gain, v, to)) = best else { break };

            let far = if to == A { B } else { A };
            let dragged: Vec<usize> = g
                .neighbours(v)
                .iter()
                .copied()
                .filter(|&u| s.part[u] == far)
                .collect();
            let dragw: usize = dragged.iter().map(|&u| g.w(u)).sum();

            s.part[v] = to;
            for &u in &dragged {
                s.part[u] = SEP;
                sep.push(u);
            }
            s.locked[v] = true;
            if to == A {
                wa += g.w(v);
                wb -= dragw;
            } else {
                wb += g.w(v);
                wa -= dragw;
            }
            wsep = (wsep as isize - gain) as usize;
            moves.push(Move { v, to, dragged });

            if wsep < best_sep {
                best_sep = wsep;
                best_step = moves.len();
            }
            // A long tail of bad moves that never pays off is just wasted
            // work; FM's escape is usually a few moves deep.
            if moves.len() > best_step + 50 {
                break;
            }
        }

        // Roll back everything after the best separator we saw.
        for m in moves.drain(best_step..).rev() {
            let far = if m.to == A { B } else { A };
            s.part[m.v] = SEP;
            for &u in &m.dragged {
                s.part[u] = far;
            }
        }
        // Rebuild the three sets from `part`. This is NOT optional even when
        // the pass improved nothing: the moves pushed dragged nodes into
        // `sep`, and although the rollback restored `part`, those pushes are
        // still there -- a node would be handed to the recursion twice.
        // It has to come from `nodes`, the subgraph, each node exactly once.
        a.clear();
        b.clear();
        sep.clear();
        wa = 0;
        wb = 0;
        wsep = 0;
        for &u in nodes {
            match s.part[u] {
                A => {
                    a.push(u);
                    wa += g.w(u);
                }
                B => {
                    b.push(u);
                    wb += g.w(u);
                }
                _ => {
                    sep.push(u);
                    wsep += g.w(u);
                }
            }
        }
        debug_assert_eq!(a.len() + b.len() + sep.len(), nodes.len());
        if best_step == 0 {
            // Nothing improved, and another pass would start from exactly
            // here: the rollback put every node back where it was.
            debug_assert_eq!((wa, wb, wsep), (start_wa, start_wb, start_wsep));
            break;
        }
    }
}

/// Breadth-first search inside the current subgraph. Fills `queue` with the
/// nodes in visit order and returns how many it reached -- fewer than the
/// subgraph holds means the subgraph is disconnected.
fn bfs(g: &Graph, start: usize, s: &mut Scratch) -> usize {
    s.stamp += 1;
    s.queue.clear();
    s.queue.push(start);
    s.seen[start] = s.stamp;
    let mut head = 0;
    while head < s.queue.len() {
        let u = s.queue[head];
        head += 1;
        for &v in g.neighbours(u) {
            if s.inset[v] && s.seen[v] != s.stamp {
                s.seen[v] = s.stamp;
                s.queue.push(v);
            }
        }
    }
    s.queue.len()
}

/// A node far from the graph's middle. BFS from such a node gives deep, thin
/// level sets, and a thin level set is a small separator. Two rounds of "go to
/// the last node reached, start again" is the standard cheap heuristic.
fn pseudo_peripheral(g: &Graph, nodes: &[usize], s: &mut Scratch) -> usize {
    let mut start = nodes[0];
    for _ in 0..2 {
        bfs(g, start, s);
        start = *s.queue.last().unwrap();
    }
    start
}

/// The smallest set of vertices that covers every cut edge -- which is exactly
/// the smallest vertex separator for a given edge cut.
///
/// The cut edges form a BIPARTITE graph between A's boundary and B's boundary,
/// and a vertex separator is precisely a vertex cover of it: leave a cut edge
/// uncovered and the two sides still touch. Taking one side's whole boundary
/// (the obvious construction) is only an upper bound on the smallest such
/// cover, and usually a loose one.
///
/// Minimum-WEIGHT vertex cover in a bipartite graph is a min cut: source ->
/// left with capacity w(u), left -> right with infinite capacity on each cut
/// edge, right -> sink with capacity w(v). The min cut has to sever a
/// source-side or a sink-side edge for every cut edge -- it cannot cut the
/// infinite middle -- so it names a cover, and being minimum it names the
/// smallest one. Weight matters here: a 9-wide camera in the separator costs
/// nine times a 1-wide landmark, and Koenig's unweighted matching would not
/// know that.
///
/// Returns `None` when the cut is degenerate (no cut edges at all).
fn min_vertex_cover(
    g: &Graph,
    s: &Scratch,
    bnd_a: &[usize],
    bnd_b: &[usize],
) -> Option<Vec<usize>> {
    if bnd_a.is_empty() || bnd_b.is_empty() {
        return None;
    }
    // Dinic on a tiny network: source, |bnd_a| + |bnd_b| nodes, sink.
    let (nl, nr) = (bnd_a.len(), bnd_b.len());
    let (src, snk) = (0usize, 1 + nl + nr);
    let n = snk + 1;
    let inf = usize::MAX / 4;

    let mut head: Vec<usize> = Vec::new(); // to
    let mut cap: Vec<usize> = Vec::new();
    let mut next: Vec<Vec<usize>> = vec![Vec::new(); n]; // edge ids out of each node
    let add = |u: usize,
                   v: usize,
                   c: usize,
                   head: &mut Vec<usize>,
                   cap: &mut Vec<usize>,
                   next: &mut Vec<Vec<usize>>| {
        next[u].push(head.len());
        head.push(v);
        cap.push(c);
        next[v].push(head.len());
        head.push(u);
        cap.push(0); // the reverse edge
    };

    let mut idx_a = std::collections::HashMap::with_capacity(nl);
    for (i, &u) in bnd_a.iter().enumerate() {
        idx_a.insert(u, 1 + i);
        add(src, 1 + i, g.w(u), &mut head, &mut cap, &mut next);
    }
    let mut idx_b = std::collections::HashMap::with_capacity(nr);
    for (j, &v) in bnd_b.iter().enumerate() {
        idx_b.insert(v, 1 + nl + j);
        add(1 + nl + j, snk, g.w(v), &mut head, &mut cap, &mut next);
    }
    let mut any_edge = false;
    for &u in bnd_a {
        for &v in g.neighbours(u) {
            if s.inset[v] && s.side[v] == 2 && let Some(&jv) = idx_b.get(&v) {
                add(idx_a[&u], jv, inf, &mut head, &mut cap, &mut next);
                any_edge = true;
            }
        }
    }
    if !any_edge {
        return None;
    }

    // Dinic: level graph by BFS, then blocking flow by DFS.
    let mut level = vec![usize::MAX; n];
    let mut iter = vec![0usize; n];
    loop {
        level.fill(usize::MAX);
        level[src] = 0;
        let mut q = std::collections::VecDeque::from([src]);
        while let Some(u) = q.pop_front() {
            for &e in &next[u] {
                if cap[e] > 0 && level[head[e]] == usize::MAX {
                    level[head[e]] = level[u] + 1;
                    q.push_back(head[e]);
                }
            }
        }
        if level[snk] == usize::MAX {
            break; // no augmenting path: the flow is maximal
        }
        iter.fill(0);
        // iterative DFS, so a deep network cannot blow the stack
        loop {
            let mut path: Vec<usize> = Vec::new();
            let mut u = src;
            let pushed = loop {
                if u == snk {
                    break path.iter().map(|&e| cap[e]).min().unwrap_or(0);
                }
                let mut advanced = false;
                while iter[u] < next[u].len() {
                    let e = next[u][iter[u]];
                    let v = head[e];
                    if cap[e] > 0 && level[v] == level[u] + 1 {
                        path.push(e);
                        u = v;
                        advanced = true;
                        break;
                    }
                    iter[u] += 1;
                }
                if !advanced {
                    level[u] = usize::MAX; // dead end; never come back
                    match path.pop() {
                        Some(e) => u = head[e ^ 1],
                        None => break 0,
                    }
                }
            };
            if pushed == 0 {
                break;
            }
            for &e in &path {
                cap[e] -= pushed;
                cap[e ^ 1] += pushed;
            }
        }
    }

    // The min cut, read off the residual graph: a left node whose source edge
    // is saturated (unreachable from the source) is in the cover, and so is a
    // right node still reachable from it.
    let mut seen = vec![false; n];
    let mut q = std::collections::VecDeque::from([src]);
    seen[src] = true;
    while let Some(u) = q.pop_front() {
        for &e in &next[u] {
            if cap[e] > 0 && !seen[head[e]] {
                seen[head[e]] = true;
                q.push_back(head[e]);
            }
        }
    }
    let mut cover = Vec::new();
    for (i, &u) in bnd_a.iter().enumerate() {
        if !seen[1 + i] {
            cover.push(u);
        }
    }
    for (j, &v) in bnd_b.iter().enumerate() {
        if seen[1 + nl + j] {
            cover.push(v);
        }
    }
    if cover.is_empty() { None } else { Some(cover) }
}

/// Split a subgraph into two halves and the vertex separator between them.
///
/// Grow a region from a peripheral node by BFS until it holds about half the
/// nodes -- a BFS frontier is a natural cut -- then turn that edge cut into the
/// SMALLEST vertex separator it admits, by minimum vertex cover.
fn bisect(
    g: &Graph,
    nodes: &[usize],
    s: &mut Scratch,
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let start = pseudo_peripheral(g, nodes, s);
    let reached = bfs(g, start, s);

    // Disconnected: peel the reached component off. The two parts are already
    // independent, so there is nothing to separate.
    if reached < nodes.len() {
        let a: Vec<usize> = s.queue.clone();
        let stamp = s.stamp;
        let b: Vec<usize> = nodes.iter().copied().filter(|&u| s.seen[u] != stamp).collect();
        return (a, b, Vec::new());
    }

    let half = nodes.len() / 2;
    for (i, &u) in s.queue.iter().enumerate() {
        s.side[u] = if i < half { 1 } else { 2 };
    }

    let mut bnd_a = Vec::new();
    let mut bnd_b = Vec::new();
    for &u in nodes {
        let cuts = g.neighbours(u).iter().any(|&v| s.inset[v] && s.side[v] != s.side[u]);
        if cuts {
            if s.side[u] == 1 {
                bnd_a.push(u);
            } else {
                bnd_b.push(u);
            }
        }
    }
    if let Some(cover) = min_vertex_cover(g, s, &bnd_a, &bnd_b) {
        let mut in_sep = vec![false; g.nodes()];
        for &u in &cover {
            in_sep[u] = true;
        }
        let (mut a, mut b) = (Vec::new(), Vec::new());
        for &u in nodes {
            if in_sep[u] {
                continue;
            }
            if s.side[u] == 1 {
                a.push(u);
            } else {
                b.push(u);
            }
        }
        return (a, b, cover);
    }
    let sep = if bnd_a.len() <= bnd_b.len() { bnd_a } else { bnd_b };

    for &u in &sep {
        s.in_sep[u] = true;
    }
    let mut a = Vec::new();
    let mut b = Vec::new();
    for &u in nodes {
        if s.in_sep[u] {
            continue;
        }
        if s.side[u] == 1 {
            a.push(u);
        } else {
            b.push(u);
        }
    }
    for &u in &sep {
        s.in_sep[u] = false;
    }
    (a, b, sep)
}

/// Order a small subgraph with faer's AMD -- the same minimum-degree code the
/// Cholesky uses, so the leaves cost nothing to get right.
fn amd_leaf(g: &Graph, nodes: &[usize], s: &mut Scratch) -> Vec<usize> {
    let n = nodes.len();
    if n <= 2 {
        return nodes.to_vec();
    }
    // node -> local index, through the seen array (stamped, so no clearing)
    s.stamp += 1;
    let stamp = s.stamp;
    let mut local = vec![0usize; g.nodes()];
    for (i, &u) in nodes.iter().enumerate() {
        s.seen[u] = stamp;
        local[u] = i;
    }
    let mut col_ptr = vec![0usize; n + 1];
    let mut row_idx: Vec<usize> = Vec::new();
    let mut col = Vec::new();
    for (i, &u) in nodes.iter().enumerate() {
        col.clear();
        col.push(i); // the diagonal
        for &v in g.neighbours(u) {
            if s.seen[v] == stamp && local[v] < i {
                col.push(local[v]);
            }
        }
        col.sort_unstable();
        col.dedup();
        row_idx.extend_from_slice(&col);
        col_ptr[i + 1] = row_idx.len();
    }
    let sym = SymbolicSparseColMat::<usize>::new_checked(n, n, col_ptr, None, row_idx);
    let mut perm = vec![0usize; n];
    let mut perm_inv = vec![0usize; n];
    let mut mem = MemBuffer::new(faer::sparse::linalg::amd::order_scratch::<usize>(
        n,
        sym.compute_nnz(),
    ));
    faer::sparse::linalg::amd::order(
        &mut perm,
        &mut perm_inv,
        sym.as_ref(),
        Default::default(),
        MemStack::new(&mut mem),
    )
    .expect("amd on a leaf");
    perm.iter().map(|&i| nodes[i]).collect()
}

// An explicit work stack rather than recursion: the bisection makes no
// balance promise, and on a bundle-adjustment whole-graph (tens of
// thousands of point leaves peeled a few at a time) the recursion depth
// grew past the thread stack. The LIFO order reproduces the recursive
// traversal exactly: dissect A fully, then B, then emit the separator
// (eliminated last: it pays for both halves).
enum DissectTask {
    Dissect(Vec<usize>),
    EmitSep(Vec<usize>),
}

fn dissect(g: &Graph, nodes: Vec<usize>, p: NdParams, s: &mut Scratch, out: &mut Vec<usize>) {
    let mut work = vec![DissectTask::Dissect(nodes)];
    while let Some(task) = work.pop() {
        let nodes = match task {
            DissectTask::EmitSep(sep) => {
                out.extend(sep);
                continue;
            }
            DissectTask::Dissect(nodes) => nodes,
        };
        if nodes.len() <= p.leaf {
            let leaf = amd_leaf(g, &nodes, s);
            out.extend(leaf);
            continue;
        }
        for &u in &nodes {
            s.inset[u] = true;
        }
        let (mut a, mut b, mut sep) = bisect(g, &nodes, s);
        if p.fm_passes > 0 {
            refine(g, &nodes, &mut a, &mut b, &mut sep, s, p);
        }
        for &u in &nodes {
            s.inset[u] = false;
        }

        // A cut that separates nothing would loop forever.
        if a.is_empty() || b.is_empty() {
            let leaf = amd_leaf(g, &nodes, s);
            out.extend(leaf);
            continue;
        }
        work.push(DissectTask::EmitSep(sep));
        work.push(DissectTask::Dissect(b));
        work.push(DissectTask::Dissect(a));
    }
}

/// Order a graph's nodes by nested dissection. The result is the elimination
/// order: `order[k]` is the node eliminated k-th.
pub fn order_graph(g: &Graph, p: NdParams) -> Vec<usize> {
    let n = g.nodes();
    let mut s = Scratch::new(n);
    let mut out = Vec::with_capacity(n);
    dissect(g, (0..n).collect(), p, &mut s, &mut out);
    debug_assert_eq!(out.len(), n);
    out
}

/// A nested-dissection permutation of a block matrix, ready for faer.
///
/// Hand it to a symbolic Cholesky as
/// `SymmetricOrdering::Custom(nd.perm())` -- no change to faer is needed, the
/// custom ordering is part of its public API.
pub struct NestedDissection {
    fwd: Vec<crate::SparseIndex>,
    inv: Vec<crate::SparseIndex>,
    block_order: Vec<usize>,
}

impl NestedDissection {
    /// Dissect the block graph and expand the result to scalar coordinates.
    /// Every block's parameters stay contiguous and in their original relative
    /// order -- which is what keeps the factor's supernodes intact.
    pub fn of_blocks<I: Index>(sym: &SymbolicSparseBlockColMat<I>, p: NdParams) -> Self {
        let g = Graph::of_blocks(sym);
        let block_order = order_graph(&g, p);

        let n = sym.ncols();
        let mut fwd: Vec<crate::SparseIndex> = Vec::with_capacity(n);
        for &b in &block_order {
            fwd.extend(sym.col_span(b).map(|i| i as crate::SparseIndex));
        }
        let mut inv = vec![0 as crate::SparseIndex; n];
        for (new, &old) in fwd.iter().enumerate() {
            inv[old as usize] = new as crate::SparseIndex;
        }
        NestedDissection { fwd, inv, block_order }
    }

    /// The permutation, as faer wants it.
    pub fn perm(&self) -> PermRef<'_, crate::SparseIndex> {
        PermRef::new_checked(&self.fwd, &self.inv, self.fwd.len())
    }

    /// `fwd[k]` is the scalar index eliminated k-th.
    pub fn forward(&self) -> &[crate::SparseIndex] {
        &self.fwd
    }

    /// The order in BLOCK units the scalar permutation was expanded from:
    /// `block_order()[k]` is the block eliminated k-th. What a block-level
    /// factorization ([`crate::supernodal`]) consumes.
    pub fn block_order(&self) -> &[usize] {
        &self.block_order
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A bundle-adjustment-shaped graph -- a chain of camera hubs, each
    /// with a crowd of point leaves seen by neighboring hubs too -- makes
    /// the bisection peel small pieces, which used to grow the dissection
    /// recursion past the thread stack (a 47k-block whole Hessian
    /// overflowed 8 MB). Run in a deliberately small-stack thread so any
    /// return to deep recursion fails here first.
    #[test]
    fn a_ba_shaped_graph_dissects_on_a_small_stack() {
        std::thread::Builder::new()
            .stack_size(256 * 1024)
            .spawn(|| {
                let hubs = 300usize;
                let per_hub = 60usize;
                let n = hubs + hubs * per_hub;
                let mut edges: Vec<(usize, usize)> = Vec::new();
                for h in 1..hubs {
                    edges.push((h - 1, h));
                }
                for h in 0..hubs {
                    for l in 0..per_hub {
                        let leaf = hubs + h * per_hub + l;
                        edges.push((h, leaf));
                        if h + 1 < hubs {
                            edges.push((h + 1, leaf));
                        }
                    }
                }
                let g = Graph::from_edges(n, edges);
                let order = order_graph(&g, NdParams::default());
                assert_eq!(order.len(), n);
                let mut seen = vec![false; n];
                for &u in &order {
                    assert!(!seen[u], "node {} ordered twice", u);
                    seen[u] = true;
                }
            })
            .expect("spawn")
            .join()
            .expect("the dissection must not overflow a small stack");
    }

    /// A grid: the classic nested-dissection case. A separator down the middle
    /// splits it, and every recursion does the same. The ordering must be a
    /// permutation, and it must put the top-level separator LAST -- that is the
    /// whole mechanism.
    #[test]
    fn a_grid_is_dissected_and_the_separator_goes_last() {
        let (w, h) = (16usize, 16usize);
        let id = |x: usize, y: usize| y * w + x;
        let mut edges = Vec::new();
        for y in 0..h {
            for x in 0..w {
                if x + 1 < w {
                    edges.push((id(x, y), id(x + 1, y)));
                }
                if y + 1 < h {
                    edges.push((id(x, y), id(x, y + 1)));
                }
            }
        }
        let g = Graph::from_edges(w * h, edges);
        let order = order_graph(&g, NdParams { leaf: 16, ..Default::default() });

        assert_eq!(order.len(), w * h);
        let mut seen = vec![false; w * h];
        for &u in &order {
            assert!(!seen[u], "node {} ordered twice", u);
            seen[u] = true;
        }

        // The last nodes eliminated must separate the graph: remove them, and
        // what is left must fall into at least two pieces.
        let sep: std::collections::HashSet<usize> =
            order[order.len() - w..].iter().copied().collect();
        let mut comp = vec![usize::MAX; w * h];
        let mut ncomp = 0;
        for start in 0..w * h {
            if sep.contains(&start) || comp[start] != usize::MAX {
                continue;
            }
            let mut q = std::vec![start];
            comp[start] = ncomp;
            while let Some(u) = q.pop() {
                for &v in g.neighbours(u) {
                    if !sep.contains(&v) && comp[v] == usize::MAX {
                        comp[v] = ncomp;
                        q.push(v);
                    }
                }
            }
            ncomp += 1;
        }
        assert!(
            ncomp >= 2,
            "the last {} nodes should separate the grid, but it stayed in one piece",
            w
        );
    }

    /// The separator must be the SMALLEST vertex set covering the cut, not
    /// just one side's boundary. Two hubs, one per side, cover every cut edge
    /// here -- while either boundary alone is five nodes. Taking a boundary
    /// would be a valid separator and a bad one, and nothing else in the suite
    /// could tell the difference: a worse separator still gives a correct
    /// ordering, just a slower factorization.
    #[test]
    fn the_separator_is_a_minimum_cover_of_the_cut() {
        // a0..a4 on one side, b0..b4 on the other. a0 sees every b; b0 sees
        // every a. So {a0, b0} covers all nine cut edges.
        let mut edges = Vec::new();
        for i in 0..5 {
            for j in 0..5 {
                if i > 0 && j > 0 {
                    continue; // only the hubs reach across
                }
                edges.push((i, 5 + j));
            }
        }
        // keep each side connected so the bisection sees two halves
        for i in 1..5 {
            edges.push((0, i));
            edges.push((5, 5 + i));
        }
        let g = Graph::from_edges(10, edges);
        let mut s = Scratch::new(10);
        for u in 0..10 {
            s.inset[u] = true;
            s.side[u] = if u < 5 { 1 } else { 2 };
        }
        let bnd_a: Vec<usize> = (0..5)
            .filter(|&u| g.neighbours(u).iter().any(|&v| v >= 5))
            .collect();
        let bnd_b: Vec<usize> = (5..10)
            .filter(|&u| g.neighbours(u).iter().any(|&v| v < 5))
            .collect();
        assert_eq!(bnd_a.len(), 5, "every a touches across");
        assert_eq!(bnd_b.len(), 5, "every b touches across");

        let cover = min_vertex_cover(&g, &s, &bnd_a, &bnd_b).expect("a cover exists");
        assert_eq!(
            cover.len(),
            2,
            "the two hubs cover every cut edge; got {:?}",
            cover
        );

        // and it really is a cover: no cut edge survives it
        let in_sep: std::collections::HashSet<usize> = cover.iter().copied().collect();
        for u in 0..5 {
            for &v in g.neighbours(u) {
                if v >= 5 {
                    assert!(
                        in_sep.contains(&u) || in_sep.contains(&v),
                        "cut edge {}-{} left uncovered",
                        u,
                        v
                    );
                }
            }
        }
    }

    /// A disconnected graph has nothing to separate: each component is ordered
    /// on its own, and no node is lost.
    #[test]
    fn disconnected_components_are_each_ordered() {
        let mut edges = Vec::new();
        for k in 0..4 {
            let base = k * 50;
            for i in 0..49 {
                edges.push((base + i, base + i + 1));
            }
        }
        let g = Graph::from_edges(200, edges);
        let order = order_graph(&g, NdParams { leaf: 8, ..Default::default() });
        assert_eq!(order.len(), 200);
        let mut seen = vec![false; 200];
        for &u in &order {
            assert!(!seen[u]);
            seen[u] = true;
        }
    }

    /// Isolated nodes, empty graphs, and a graph smaller than the leaf size all
    /// have to come out as valid permutations.
    #[test]
    fn degenerate_graphs() {
        for n in [0usize, 1, 2, 5] {
            let g = Graph::from_edges(n, []);
            let order = order_graph(&g, NdParams::default());
            assert_eq!(order.len(), n);
            let mut seen = vec![false; n];
            for &u in &order {
                assert!(!seen[u]);
                seen[u] = true;
            }
        }
    }

    /// The block entry point: the permutation must be a valid scalar
    /// permutation, and every block's parameters must stay contiguous -- that
    /// contiguity is the whole reason we dissect the block graph.
    #[test]
    fn a_block_permutation_keeps_blocks_together() {
        // three 3-wide blocks in a path: 0 - 1 - 2
        let part = std::vec![0usize, 3, 6, 9];
        let (sym, _) = SymbolicSparseBlockColMat::<usize>::from_scalar_coords(
            part.clone(),
            part,
            4,
            |k| [(0, 0), (0, 1), (1, 1), (1, 2)][k],
        );
        let nd = NestedDissection::of_blocks(&sym, NdParams { leaf: 1, ..Default::default() });
        let fwd = nd.forward();
        assert_eq!(fwd.len(), 9);

        let mut seen = vec![false; 9];
        for &i in fwd {
            let i = i as usize;
            assert!(!seen[i], "scalar {} appears twice", i);
            seen[i] = true;
        }
        // each block of 3 must appear as a contiguous, ascending run
        for chunk in fwd.chunks(3) {
            assert_eq!(chunk[1], chunk[0] + 1);
            assert_eq!(chunk[2], chunk[0] + 2);
            assert_eq!(chunk[0] % 3, 0, "a block was split across the permutation");
        }
    }
}
