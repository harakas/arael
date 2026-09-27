// The M3500 pose graph solved in the page: the generated wasm interface
// builds the graph from the g2o file, one LmSession re-solves it warm,
// and the canvas shows the poses. A pose can be dragged (a lock, a
// soft prior, follows the pointer and stays where it is released),
// locked in place, fixed (its parameters taken out of the solve) and
// cleared, one or many at a time.
import init, { Graph, LmConfig, LmSession } from "../model/wasm/pkg/m3500_demo_wasm.js";
import { Dataset2 } from "../model/wasm/js/arael/g2o.js";

// `datasets` links to the vendored g2o files under benchmarks/pgo, so
// the example directory is all a server has to hold.
const DATASET = "datasets/input_M3500_g2o.g2o";
const LOCK_WEIGHT = 10.0;
const CENTER_WEIGHT = 0.01;
const DOUBLE_CLICK_MS = 400;
const PICK_RADIUS = 8;
const LOCKED = 1;
const FIXED = 2;

const canvas = document.getElementById("view");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const weightedEl = document.getElementById("weighted");
const liveEl = document.getElementById("live");

let ds = null;
let graph = null;
let session = null;
let n = 0;
let edgesAB = new Int32Array(0);
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

// The full solve, and the short one a drag step runs; both made once
// the module is initialized, at the bottom.
let full = null;
let quick = null;

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
  const weighted = weightedEl.checked;
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
  for (const i of selected) setLock(i, true, xy[2 * i], xy[2 * i + 1], th[i]);
  solve(full);
}

function fixSelection() {
  if (selected.size === 0) return;
  for (const i of selected) setFixed(i, true);
  // Parameters left the solve: the session's structure is stale.
  session.invalidate();
  solve(full);
}

function clearSelection() {
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

function draw() {
  const r = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, r.width, r.height);
  if (n === 0) return;
  ctx.strokeStyle = "rgba(128,128,128,0.35)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let k = 0; k < edgesAB.length; k += 2) {
    const a = edgesAB[k], b = edgesAB[k + 1];
    const [ax, ay] = toScreen(xy[2 * a], xy[2 * a + 1]);
    const [bx, by] = toScreen(xy[2 * b], xy[2 * b + 1]);
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
  }
  ctx.stroke();
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

function pick(sx, sy) {
  let best = -1, bestD = PICK_RADIUS * PICK_RADIUS;
  for (let i = 0; i < n; i++) {
    const [px, py] = toScreen(xy[2 * i], xy[2 * i + 1]);
    const d = (px - sx) * (px - sx) + (py - sy) * (py - sy);
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  }
  return best;
}

function setStatus(s) {
  text = s;
  statusEl.textContent = s;
}

function frame() {
  if (drag && drag.pose >= 0 && drag.pending && liveEl.checked) {
    drag.pending = false;
    solve(quick);
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
    setLock(i, true, xy[2 * i], xy[2 * i + 1], th[i]);
    drag = { pose: i, pending: false, heading: th[i] };
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
    const [x, y] = toWorld(sx, sy);
    const i = drag.pose;
    const lock = graph.locks().at(i);
    lock.pos = { x, y };
    if (!liveEl.checked) {
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
    // The lock stays, at the place the pointer let go.
    solve(full);
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
    default: return;
  }
  dirty = true;
});

document.getElementById("solve").addEventListener("click", () => { solve(full); dirty = true; });
document.getElementById("reset").addEventListener("click", () => { reset(); dirty = true; });
document.getElementById("recenter").addEventListener("click", () => { fit(); dirty = true; });
weightedEl.addEventListener("change", () => { build(); solve(full); dirty = true; });
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
  fit();
  solve(full);
  dirty = true;
});

window.addEventListener("resize", resize);

// -------------------------------------------------------------- start

await init();
full = LmConfig.wellConditioned();
quick = LmConfig.wellConditioned();
quick.maxIters = 3;
quick.minIters = 1;
quick.patience = 1;
resize();
requestAnimationFrame(frame);
setStatus("loading the dataset…");
try {
  ds = await Dataset2.load(DATASET);
} catch (e) {
  setStatus(`${DATASET}: ${e.message} (serve the example directory: python3 -m http.server in cxx-examples/m3500_demo)`);
  throw e;
}
build();
fit();
solve(full);
dirty = true;
