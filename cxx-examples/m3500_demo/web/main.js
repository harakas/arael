// The M3500 pose graph solved in the page: the generated wasm interface
// builds the graph from the g2o file, one LmSession re-solves it warm,
// and the canvas shows the poses. A pose can be dragged (a lock, a
// soft prior, follows the pointer and stays where it is released),
// locked in place, fixed (its parameters taken out of the solve) and
// cleared, one or many at a time, and every such step undone.
import init, { Graph, LmConfig, LmSession, araelVersion } from "../model/wasm/pkg/m3500_demo_wasm.js";
import { Dataset2 } from "../model/wasm/js/arael/g2o.js";

// `datasets` links to the vendored g2o files under benchmarks/pgo, so
// the example directory is all a server has to hold.
const DATASET = "datasets/input_M3500_g2o.g2o";
const LOCK_WEIGHT = 10.0;
const CENTER_WEIGHT = 0.01;
const DOUBLE_CLICK_MS = 400;
const PICK_RADIUS = 8;
const DRAG_START_PX = 3;
const UNDO_DEPTH = 50;
// Solver time per frame while dragging.
const QUICK_BUDGET_S = 0.03;
// Link color: gray at no cost, bright red at and above the cost of the
// worst percent of edges, in this many steps.
const LINK_BUCKETS = 16;
const LINK_RED_QUANTILE = 0.99;
const LOCKED = 1;
const FIXED = 2;

const canvas = document.getElementById("view");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const versionEl = document.getElementById("version");
const weightedEl = document.getElementById("weighted");
const liveEl = document.getElementById("live");
const tipEl = document.getElementById("tip");

// The two toggle buttons in the tool column.
function pressed(el) {
  return el.getAttribute("aria-pressed") === "true";
}

function toggle(el) {
  el.setAttribute("aria-pressed", pressed(el) ? "false" : "true");
}

let ds = null;
let graph = null;
let session = null;
let n = 0;
let edgesAB = new Int32Array(0);
// Each edge's measurement and whitening rows, as given to the model.
let edgeMeas = null;
let xy = new Float64Array(0);
let th = new Float64Array(0);
let loaded = { xy: new Float64Array(0), th: new Float64Array(0) };
let flags = new Uint8Array(0);
let selected = new Set();
let hover = -1;
let view = { scale: 1, ox: 0, oy: 0 };
let drag = null;
let dirty = false;
let text = "";
let undo = [];
let redo = [];

// The full solve, and the short one a drag step runs: a time budget per
// frame, and its damping carried from the previous frame, since a graph
// bent between locks restarted at Gauss-Newton every frame overshoots
// and spends its iterations on rejected steps. Both made once the
// module is initialized, at the bottom.
let full = null;
let quick = null;
let quickLambda0 = 0;

// ------------------------------------------------------------ the model

function build() {
  graph = new Graph();
  n = ds.poses.length;
  const poses = graph.poses();
  poses.pushN(n);
  const p = new Float64Array(2 * n);
  const t = new Float64Array(n);
  ds.poses.forEach((q, i) => {
    p[2 * i] = q.t.x;
    p[2 * i + 1] = q.t.y;
    t[i] = q.th;
  });
  poses.setPosN(0, p);
  poses.setRotAngleN(0, t);
  loaded = { xy: p.slice(), th: t.slice() };

  const refs = poses.getRefsN(0, n);
  const m = ds.deltas.length;
  const edges = graph.edges();
  edges.pushN(m);
  const a = new Uint32Array(m);
  const b = new Uint32Array(m);
  const delta = new Float64Array(2 * m);
  const dth = new Float64Array(m);
  const s0 = new Float64Array(3 * m);
  const s1 = new Float64Array(3 * m);
  const s2 = new Float64Array(3 * m);
  edgesAB = new Int32Array(2 * m);
  const weighted = pressed(weightedEl);
  ds.deltas.forEach((d, k) => {
    a[k] = refs[d.a];
    b[k] = refs[d.b];
    edgesAB[2 * k] = d.a;
    edgesAB[2 * k + 1] = d.b;
    delta[2 * k] = d.dt.x;
    delta[2 * k + 1] = d.dt.y;
    dth[k] = d.dth;
    // Exact whitening for any symmetric information matrix: rows of
    // diag(w) * R^T from its eigendecomposition; identity otherwise.
    let c0 = [1, 0, 0], c1 = [0, 1, 0], c2 = [0, 0, 1];
    if (weighted) {
      const { r, w } = d.eigenSqrtInfo();
      c0 = [r[0][0] * w.x, r[1][0] * w.x, r[2][0] * w.x];
      c1 = [r[0][1] * w.y, r[1][1] * w.y, r[2][1] * w.y];
      c2 = [r[0][2] * w.z, r[1][2] * w.z, r[2][2] * w.z];
    }
    s0.set(c0, 3 * k);
    s1.set(c1, 3 * k);
    s2.set(c2, 3 * k);
  });
  edgeMeas = { delta, dth, s0, s1, s2 };
  edges.setAN(0, a);
  edges.setBN(0, b);
  edges.setDeltaN(0, delta);
  edges.setDthN(0, dth);
  edges.setS0N(0, s0);
  edges.setS1N(0, s1);
  edges.setS2N(0, s2);

  // One lock per pose, off: switching one changes no structure.
  const locks = graph.locks();
  locks.pushN(n);
  locks.setPN(0, refs);
  locks.setPosN(0, p);
  locks.setThN(0, t);
  locks.setWN(0, new Float64Array(n).fill(LOCK_WEIGHT));
  locks.setOnN(0, new Uint8Array(n));
  // The gauge: one more lock, on pose 0, very weak and always on. It
  // pins the frame and little else, so every pose moves under a drag.
  const center = locks.push();
  center.p = refs[0];
  center.pos = { x: p[0], y: p[1] };
  center.th = t[0];
  center.w = CENTER_WEIGHT;
  center.on = true;

  flags = new Uint8Array(n);
  selected = new Set();
  hover = -1;
  undo = [];
  redo = [];
  session = new LmSession();
  refresh();
}

function refresh() {
  const poses = graph.poses();
  xy = poses.getPosN(0, n);
  th = poses.getRotAngleN(0, n);
}

function solve(cfg) {
  const t0 = performance.now();
  let r;
  try {
    r = session.solve(graph, cfg);
  } catch (e) {
    setStatus(`solve failed: ${e.message}`);
    return null;
  }
  const ms = performance.now() - t0;
  refresh();
  setStatus(`${n} poses, ${ds.deltas.length} edges | ${r.iterations} iterations, ` +
    `cost ${r.startCost.toFixed(3)} -> ${r.endCost.toFixed(3)}, ${r.statusName} | ` +
    `solve ${ms.toFixed(1)} ms`);
  return r;
}

function setLock(i, on, x, y, heading) {
  const lock = graph.locks().at(i);
  lock.on = on;
  if (on) {
    lock.pos = { x, y };
    lock.th = heading;
    flags[i] |= LOCKED;
  } else {
    flags[i] &= ~LOCKED;
  }
}

function setFixed(i, fixed) {
  const pose = graph.poses().at(i);
  pose.posOptimize = !fixed;
  pose.rotAngleOptimize = !fixed;
  if (fixed) flags[i] |= FIXED; else flags[i] &= ~FIXED;
}

function lockSelection() {
  if (selected.size === 0) return;
  remember();
  for (const i of selected) setLock(i, true, xy[2 * i], xy[2 * i + 1], th[i]);
  solve(full);
}

function fixSelection() {
  if (selected.size === 0) return;
  remember();
  for (const i of selected) setFixed(i, true);
  // Parameters left the solve: the session's structure is stale.
  session.invalidate();
  solve(full);
}

function clearSelection() {
  if (selected.size === 0) return;
  remember();
  let unfixed = false;
  for (const i of selected) {
    if (flags[i] & LOCKED) setLock(i, false, 0, 0, 0);
    if (flags[i] & FIXED) {
      setFixed(i, false);
      unfixed = true;
    }
  }
  if (unfixed) session.invalidate();
  solve(full);
}

function reset() {
  if (n === 0) return;
  remember();
  const poses = graph.poses();
  poses.setPosN(0, loaded.xy);
  poses.setRotAngleN(0, loaded.th);
  let unfixed = false;
  for (let i = 0; i < n; i++) {
    if (flags[i] & FIXED) unfixed = true;
  }
  poses.setPosOptimizeN(0, new Uint8Array(n).fill(1));
  poses.setRotAngleOptimizeN(0, new Uint8Array(n).fill(1));
  graph.locks().setOnN(0, new Uint8Array(n));
  flags.fill(0);
  selected.clear();
  if (unfixed) session.invalidate();
  solve(full);
  fit();
}

// ------------------------------------------------------------------ undo

// What an action changes: the pose values, the locks and the flags. A
// snapshot is taken before each action and put back whole, so an undo
// needs no solve.
function snapshot() {
  const locks = graph.locks();
  return {
    xy: xy.slice(),
    th: th.slice(),
    flags: flags.slice(),
    lockPos: locks.getPosN(0, n),
    lockTh: locks.getThN(0, n),
  };
}

function remember() {
  undo.push(snapshot());
  if (undo.length > UNDO_DEPTH) undo.shift();
  redo.length = 0;
}

function restore(s) {
  const poses = graph.poses();
  const locks = graph.locks();
  const opt = new Uint8Array(n);
  const on = new Uint8Array(n);
  let fixedChanged = false;
  for (let i = 0; i < n; i++) {
    opt[i] = (s.flags[i] & FIXED) ? 0 : 1;
    on[i] = (s.flags[i] & LOCKED) ? 1 : 0;
    if ((s.flags[i] ^ flags[i]) & FIXED) fixedChanged = true;
  }
  poses.setPosN(0, s.xy);
  poses.setRotAngleN(0, s.th);
  poses.setPosOptimizeN(0, opt);
  poses.setRotAngleOptimizeN(0, opt);
  locks.setPosN(0, s.lockPos);
  locks.setThN(0, s.lockTh);
  locks.setOnN(0, on);
  flags = s.flags.slice();
  // Parameters entered or left the solve: the session's structure is stale.
  if (fixedChanged) session.invalidate();
  refresh();
  dirty = true;
}

function undoStep() {
  if (undo.length === 0) return;
  redo.push(snapshot());
  restore(undo.pop());
  setStatus(`undo: ${undo.length} more`);
}

function redoStep() {
  if (redo.length === 0) return;
  undo.push(snapshot());
  restore(redo.pop());
  setStatus(`redo: ${redo.length} more`);
}

// ------------------------------------------------------------- the view

function resize() {
  const r = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(r.width * dpr);
  canvas.height = Math.round(r.height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  dirty = true;
}

function fit() {
  if (n === 0) return;
  let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
  for (let i = 0; i < n; i++) {
    const x = xy[2 * i], y = xy[2 * i + 1];
    if (x < xmin) xmin = x;
    if (x > xmax) xmax = x;
    if (y < ymin) ymin = y;
    if (y > ymax) ymax = y;
  }
  const r = canvas.getBoundingClientRect();
  const pad = 30;
  view.scale = Math.min((r.width - 2 * pad) / Math.max(xmax - xmin, 1e-9),
                        (r.height - 2 * pad) / Math.max(ymax - ymin, 1e-9));
  view.ox = r.width / 2 - view.scale * (xmin + xmax) / 2;
  view.oy = r.height / 2 + view.scale * (ymin + ymax) / 2;
  dirty = true;
}

function toScreen(x, y) {
  return [view.ox + x * view.scale, view.oy - y * view.scale];
}

function toWorld(sx, sy) {
  return [(sx - view.ox) / view.scale, (view.oy - sy) / view.scale];
}

function wrapAngle(d) {
  return d - 2 * Math.PI * Math.floor((d + Math.PI) / (2 * Math.PI));
}

// Each edge's cost on the current poses: the model's edge residual, the
// whitened error of pose b seen from pose a against the measurement,
// squared and summed.
function edgeCosts() {
  const m = edgesAB.length / 2;
  const c = new Float64Array(m);
  const { delta, dth, s0, s1, s2 } = edgeMeas;
  for (let k = 0; k < m; k++) {
    const a = edgesAB[2 * k], b = edgesAB[2 * k + 1];
    const dx = xy[2 * b] - xy[2 * a];
    const dy = xy[2 * b + 1] - xy[2 * a + 1];
    const ca = Math.cos(th[a]), sa = Math.sin(th[a]);
    const lx = ca * dx + sa * dy - delta[2 * k];
    const ly = -sa * dx + ca * dy - delta[2 * k + 1];
    const rr = wrapAngle(th[b] - (th[a] + dth[k]));
    const r0 = s0[3 * k] * lx + s0[3 * k + 1] * ly + s0[3 * k + 2] * rr;
    const r1 = s1[3 * k] * lx + s1[3 * k + 1] * ly + s1[3 * k + 2] * rr;
    const r2 = s2[3 * k] * lx + s2[3 * k + 1] * ly + s2[3 * k + 2] * rr;
    c[k] = r0 * r0 + r1 * r1 + r2 * r2;
  }
  return c;
}

function linkColor(t) {
  const r = Math.round(128 + 127 * t);
  const gb = Math.round(128 * (1 - t));
  return `rgba(${r},${gb},${gb},${(0.35 + 0.65 * t).toFixed(2)})`;
}

function draw() {
  const r = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, r.width, r.height);
  if (n === 0) return;
  // The links, bucketed by cost and drawn gray to red, the reddest last.
  const costs = edgeCosts();
  const sorted = Float64Array.from(costs).sort();
  const red = Math.max(sorted[Math.floor(LINK_RED_QUANTILE * (sorted.length - 1))], 1e-12);
  const paths = Array.from({ length: LINK_BUCKETS }, () => new Path2D());
  for (let k = 0; k < costs.length; k++) {
    const a = edgesAB[2 * k], b = edgesAB[2 * k + 1];
    const [ax, ay] = toScreen(xy[2 * a], xy[2 * a + 1]);
    const [bx, by] = toScreen(xy[2 * b], xy[2 * b + 1]);
    const t = Math.min(1, costs[k] / red);
    const path = paths[Math.min(LINK_BUCKETS - 1, Math.floor(t * LINK_BUCKETS))];
    path.moveTo(ax, ay);
    path.lineTo(bx, by);
  }
  ctx.lineWidth = 1;
  paths.forEach((path, i) => {
    ctx.strokeStyle = linkColor(i / (LINK_BUCKETS - 1));
    ctx.stroke(path);
  });
  const colors = ["#3a7bd5", "#e0a000", "#d33", "#d33"];
  for (let c = 0; c < 4; c++) {
    ctx.fillStyle = colors[c];
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      if ((flags[i] & 3) !== c) continue;
      const [sx, sy] = toScreen(xy[2 * i], xy[2 * i + 1]);
      ctx.moveTo(sx + 2.5, sy);
      ctx.arc(sx, sy, 2.5, 0, 2 * Math.PI);
    }
    ctx.fill();
  }
  ctx.strokeStyle = "#2c2";
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (const i of selected) {
    const [sx, sy] = toScreen(xy[2 * i], xy[2 * i + 1]);
    ctx.moveTo(sx + 6, sy);
    ctx.arc(sx, sy, 6, 0, 2 * Math.PI);
  }
  ctx.stroke();
  if (hover >= 0) {
    const [sx, sy] = toScreen(xy[2 * hover], xy[2 * hover + 1]);
    ctx.strokeStyle = "#888";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(sx, sy, 8, 0, 2 * Math.PI);
    ctx.stroke();
  }
  if (drag && drag.rect) {
    const [x0, y0, x1, y1] = drag.rect;
    ctx.strokeStyle = "#2c2";
    ctx.setLineDash([4, 3]);
    ctx.strokeRect(Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0));
    ctx.setLineDash([]);
  }
}

// The pose under the pointer: a locked or fixed pose within the pick
// radius wins over the plain poses crowding it, so it can be picked out
// to clear it; otherwise the nearest.
function pick(sx, sy) {
  let best = -1, bestD = PICK_RADIUS * PICK_RADIUS;
  let held = -1, heldD = bestD;
  for (let i = 0; i < n; i++) {
    const [px, py] = toScreen(xy[2 * i], xy[2 * i + 1]);
    const d = (px - sx) * (px - sx) + (py - sy) * (py - sy);
    if (flags[i]) {
      if (d < heldD) {
        heldD = d;
        held = i;
      }
    } else if (d < bestD) {
      bestD = d;
      best = i;
    }
  }
  return held >= 0 ? held : best;
}

function setStatus(s) {
  text = s;
  statusEl.textContent = s;
}

function frame() {
  if (drag && drag.pose >= 0 && drag.pending && pressed(liveEl)) {
    drag.pending = false;
    quick.initialLambda = drag.lambda ?? quickLambda0;
    const r = solve(quick);
    if (r) drag.lambda = r.finalLambda;
  }
  if (dirty) {
    dirty = false;
    draw();
  }
  requestAnimationFrame(frame);
}

// ------------------------------------------------------------ the mouse

function pointer(e) {
  const r = canvas.getBoundingClientRect();
  return [e.clientX - r.left, e.clientY - r.top];
}

// The middle button's own action, autoscroll, would take the canvas.
canvas.addEventListener("mousedown", (e) => { if (e.button === 1) e.preventDefault(); });

let lastMiddle = 0;

canvas.addEventListener("pointerdown", (e) => {
  if (n === 0) return;
  const [sx, sy] = pointer(e);
  if (e.button === 1) {
    // The middle button pans; a middle double-click fits the view to
    // everything.
    const now = performance.now();
    if (now - lastMiddle < DOUBLE_CLICK_MS) {
      lastMiddle = 0;
      fit();
      return;
    }
    lastMiddle = now;
    canvas.setPointerCapture(e.pointerId);
    drag = { pose: -1, pan: [sx, sy, view.ox, view.oy] };
    return;
  }
  if (e.button !== 0) return;
  const i = pick(sx, sy);
  canvas.setPointerCapture(e.pointerId);
  if (e.shiftKey) {
    if (i >= 0) {
      if (selected.has(i)) selected.delete(i); else selected.add(i);
      dirty = true;
      drag = { pose: -1 };
    } else {
      drag = { pose: -1, rect: [sx, sy, sx, sy] };
    }
    return;
  }
  if (i >= 0) {
    if (!selected.has(i)) {
      selected = new Set([i]);
    }
    if (flags[i] & FIXED) {
      drag = { pose: -1 };
      dirty = true;
      return;
    }
    // A press selects; the lock comes with the first movement, so a
    // click leaves the pose as it is.
    drag = { pose: i, pending: false, moved: false, from: [sx, sy] };
    dirty = true;
    return;
  }
  selected.clear();
  drag = { pose: -1, pan: [sx, sy, view.ox, view.oy] };
  dirty = true;
});

canvas.addEventListener("pointermove", (e) => {
  if (n === 0) return;
  const [sx, sy] = pointer(e);
  if (!drag) {
    const h = pick(sx, sy);
    if (h !== hover) {
      hover = h;
      dirty = true;
    }
    return;
  }
  if (drag.pose >= 0) {
    const i = drag.pose;
    if (!drag.moved) {
      const [fx, fy] = drag.from;
      if ((sx - fx) * (sx - fx) + (sy - fy) * (sy - fy) < DRAG_START_PX * DRAG_START_PX) return;
      drag.moved = true;
      remember();
      setLock(i, true, xy[2 * i], xy[2 * i + 1], th[i]);
    }
    const [x, y] = toWorld(sx, sy);
    const lock = graph.locks().at(i);
    lock.pos = { x, y };
    if (!pressed(liveEl)) {
      const pose = graph.poses().at(i);
      pose.pos = { x, y };
      xy[2 * i] = x;
      xy[2 * i + 1] = y;
    }
    drag.pending = true;
    dirty = true;
  } else if (drag.pan) {
    const [px, py, ox, oy] = drag.pan;
    view.ox = ox + (sx - px);
    view.oy = oy + (sy - py);
    dirty = true;
  } else if (drag.rect) {
    drag.rect[2] = sx;
    drag.rect[3] = sy;
    dirty = true;
  }
});

canvas.addEventListener("pointerup", (e) => {
  if (!drag) return;
  const d = drag;
  drag = null;
  if (d.pose >= 0) {
    // After a drag the lock stays, at the place the pointer let go; a
    // click changed nothing.
    if (d.moved) solve(full);
  } else if (d.rect) {
    const [x0, y0, x1, y1] = d.rect;
    const [lx, hx] = [Math.min(x0, x1), Math.max(x0, x1)];
    const [ly, hy] = [Math.min(y0, y1), Math.max(y0, y1)];
    for (let i = 0; i < n; i++) {
      const [sx, sy] = toScreen(xy[2 * i], xy[2 * i + 1]);
      if (sx >= lx && sx <= hx && sy >= ly && sy <= hy) selected.add(i);
    }
  }
  dirty = true;
});

canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  const [sx, sy] = pointer(e);
  const f = Math.exp(-e.deltaY * 0.0015);
  view.ox = sx - (sx - view.ox) * f;
  view.oy = sy - (sy - view.oy) * f;
  view.scale *= f;
  dirty = true;
}, { passive: false });

window.addEventListener("keydown", (e) => {
  if (n === 0 || e.target.tagName === "INPUT") return;
  switch (e.key) {
    case "l": case "L": lockSelection(); break;
    case "f": case "F": fixSelection(); break;
    case "c": case "C": clearSelection(); break;
    case "r": case "R": reset(); break;
    case "Escape": selected.clear(); break;
    case "z": case "Z":
      if (!(e.ctrlKey || e.metaKey)) return;
      e.preventDefault();
      if (e.shiftKey) redoStep(); else undoStep();
      break;
    case "y": case "Y":
      if (!(e.ctrlKey || e.metaKey)) return;
      e.preventDefault();
      redoStep();
      break;
    default: return;
  }
  dirty = true;
});

// ------------------------------------------------------------- the tools

const actions = {
  solve: () => solve(full),
  reset,
  undo: undoStep,
  redo: redoStep,
  lock: lockSelection,
  fix: fixSelection,
  clear: clearSelection,
  recenter: fit,
  load: () => document.getElementById("file").click(),
  weighted: () => { toggle(weightedEl); build(); solve(full); },
  live: () => toggle(liveEl),
};
for (const [id, act] of Object.entries(actions)) {
  document.getElementById(id).addEventListener("click", () => {
    if (n === 0 && id !== "load") return;
    act();
    dirty = true;
  });
}

// The tooltip: one panel, placed beside the hovered or focused button.
function showTip(el) {
  const text = el.dataset.tip;
  if (!text) return;
  tipEl.textContent = text;
  tipEl.hidden = false;
  const r = el.getBoundingClientRect();
  const h = tipEl.offsetHeight;
  const top = Math.max(4, Math.min(window.innerHeight - h - 4, r.top + r.height / 2 - h / 2));
  tipEl.style.left = `${r.right + 8}px`;
  tipEl.style.top = `${top}px`;
}

function hideTip() {
  tipEl.hidden = true;
}

for (const el of document.querySelectorAll("#tools button")) {
  el.addEventListener("pointerenter", () => showTip(el));
  el.addEventListener("pointerleave", hideTip);
  el.addEventListener("focus", () => showTip(el));
  el.addEventListener("blur", hideTip);
}

document.getElementById("file").addEventListener("change", async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  try {
    ds = Dataset2.parse(await f.text());
  } catch (err) {
    setStatus(`${f.name}: ${err.message}`);
    return;
  }
  build();
  solve(full);
  fit();
  dirty = true;
});

window.addEventListener("resize", resize);

// -------------------------------------------------------------- start

await init();
const ver = araelVersion();
versionEl.textContent = `arael ${ver.major}.${ver.minor}.${ver.patch}${ver.pre ? "-" + ver.pre : ""}`;
full = LmConfig.wellConditioned();
quick = LmConfig.wellConditioned();
quick.maxIters = 50;
quick.minIters = 1;
quick.patience = 3;
quick.timeLimitSeconds = QUICK_BUDGET_S;
quickLambda0 = quick.initialLambda;
resize();
requestAnimationFrame(frame);
setStatus("loading the dataset...");
try {
  ds = await Dataset2.load(DATASET);
} catch (e) {
  setStatus(`${DATASET}: ${e.message} (serve the example directory: python3 -m http.server in cxx-examples/m3500_demo)`);
  throw e;
}
build();
solve(full);
fit();
dirty = true;
