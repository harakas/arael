// The fixture problem through the generated JavaScript interface under
// node: composed as parity_verify::fill composes it in Rust, solved
// dense, the result printed as `name value` lines for wasm.rs to
// compare against the Rust solve exactly.
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const pkg = process.argv[2];
const w = require(pkg + "/cxx_fit_wasm.js");

const fit = new w.Fit();
const obs = fit.obs();
for (let i = 0; i < 6; i++) {
  const o = obs.push();
  o.x = i;
  o.y = 2.0 * i + 1.0 + (i % 2 === 0 ? 0.05 : -0.05);
}
const t = [1.5, -0.3, 0.7];
const wt = [1.0, 2.0, 0.5];
const items = fit.items();
for (let i = 0; i < 3; i++) {
  const n = items.push();
  n.t = t[i];
  n.w = wt[i];
}
const vn = fit.vns().push();
vn.v = [0.4, -0.1, 0.9, 0.0];
vn.t = [0.1, 0.2, 0.5, -0.3];
vn.h = [[1.0, 0.5, 0.0, -0.2], [0.0, 1.0, 0.3, 0.4]];
vn.wp = 0.7;
vn.w = 1.3;

const p = (name, v) => console.log(`${name} ${v}`);
const ver = w.araelVersion();
p("arael_version_major", ver.major);
p("arael_version_minor", ver.minor);
p("arael_version_patch", ver.patch);
p("arael_version_pre_len", ver.pre.length);
p("clean", fit.validate() === "" ? 1 : 0);
p("n_obs", obs.length);
p("n_items", items.length);
p("obs3_y", obs.at(3).y);
p("vn_h11", vn.h[1][1]);
p("vn_v2", vn.v[2]);
p("initial_cost", fit.cost());

const cfg = new w.LmConfig();
cfg.maxIters = 50;
const r = fit.solveDense(cfg);
p("status", r.status);
p("status_named", r.status === w.LmStatus.Converged ? 1 : 0);
p("status_text_len", r.statusText.length);
const th = r.threads();
p("th_sweeps_asked", th.sweepsAsked);
p("th_linear", th.linear);
p("th_fell_back", th.fellBack ? 1 : 0);
p("th_has_sweeps", th.sweeps === undefined ? 0 : 1);
p("enum_schur_force", w.SchurPolicy.Force);
p("iterations", r.iterations);
p("start_cost", r.startCost);
p("end_cost", r.endCost);
p("m", fit.m);
p("c", fit.c);
for (let i = 0; i < 3; i++) p(`item${i}_v`, items.at(i).v);
const v = fit.vns().at(0).v;
for (let i = 0; i < 4; i++) p(`vn_v${i}_after`, v[i]);
