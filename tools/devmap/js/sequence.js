// devmap view: Sequence. The operations of a phase (kernel launches and
// host copies) one by one in execution order, as one occurrence of the
// phase issues them: the Pipeline merges a kernel's launches into one step,
// this lists each where it runs. A row: its place, the kernel (with its
// number among the step's launches) or the copy, its stage, its geometry,
// the arguments that differ between the step's launches, and on a timed
// case its time and the device time elapsed in the phase after it.
"use strict";
(function () {
  var DM = window.DM, h = DM.h;

  var cur = null;       // { rows: [{ tr, T }] }

  function th(text, tip, cls) { return h("th", { tip: tip, cls: cls }, text); }

  // "npanels 2, k0 32": a launch's values of its step's differing arguments.
  function differs(c) {
    if (!c.r) return null;
    var out = [];
    c.T.runs.vary.forEach(function (G, j) {
      var x = c.r.v[j];
      if (!x) return;
      if (out.length) out.push(h("span", { cls: "dim" }, ", "));
      out.push(h("span", { cls: "dim" }, (G ? G.param : "?") + " "));
      out.push(x.S ? [x.S.label, x.off ? h("span", { cls: "dim" }, "+" + DM.fmtInt(x.off)) : null]
        : x.value !== null ? x.value : x.at !== null ? "+" + DM.fmtInt(x.at) : "-");
    });
    return out;
  }

  function phaseTable(M, P) {
    var calls = DM.callsOf(P);
    var box = h("div", { cls: "seq-phase" });
    var launches = 0, total = 0, timed = false;
    (calls || []).forEach(function (c) {
      if (c.T.kind === "launch") launches += 1;
      var t = c.r ? c.r.time : c.T.time;
      if (t !== null) { timed = true; total += t; }
    });
    box.appendChild(h("h3", { cls: "seq-title" }, DM.link.scope(M, P.id, null, P.title),
      h("span", { cls: "dim" }, "  x" + DM.fmtInt(P.times) + (calls ? "   " + calls.length + " operations an occurrence: " + launches + " launches, " +
        (calls.length - launches) + " copies" + (timed ? "; " + DM.fmtTime(total) + " on the device" : "") : ""))));
    if (!calls) {
      box.appendChild(h("div", { cls: "empty" }, "This case does not record the order of this phase's operations (a case written before the Sequence view: regenerate it)."));
      return box;
    }
    var head = h("tr", null, th("#", "its place among the phase's operations, from 1"), th("operation", "the kernel launched, with its number among its step's launches, or the host copy"),
      th("stage", "the scope below the phase"), th("grid x threads", "a launch's blocks and threads a block; a copy's bytes"),
      th("differs", "of a step of several launches: this launch's values of the arguments that differ between them"));
    if (timed) {
      head.appendChild(th("time", "the launch's mean device time over the phase's occurrences", "numc"));
      head.appendChild(th("elapsed", "the device time of the phase's launches up to and with this one", "numc"));
    }
    var body = h("tbody", null), elapsed = 0;
    calls.forEach(function (c) {
      var T = c.T, launch = T.kind === "launch", t = c.r ? c.r.time : T.time;
      if (t !== null) elapsed += t;
      var tr = h("tr", { cls: "seq-row" + (launch ? "" : " seq-copy"), onclick: function () { DM.select(DM.sel.step(T), { reveal: false }); } },
        h("td", { cls: "numc dim" }, String(c.n + 1)),
        h("td", { cls: "mono" }, T.label, c.r ? h("span", { cls: "dim" }, "  " + (c.i + 1) + "/" + T.launches) : null, h("span", { cls: "sid" }, "  " + T.id)),
        h("td", { cls: "dim" }, T.scope.slice(1).join(" / ")),
        h("td", { cls: "numc" }, launch ? DM.geomText(c.r || T) : DM.fmtBytes(T.bytes)),
        h("td", { cls: "seq-diff" }, differs(c)));
      if (timed) {
        tr.appendChild(h("td", { cls: "numc" }, t !== null ? DM.fmtTime(t) : "-"));
        tr.appendChild(h("td", { cls: "numc dim" }, t !== null ? DM.fmtTime(elapsed) : ""));
      }
      cur.rows.push({ tr: tr, T: T });
      body.appendChild(tr);
    });
    box.appendChild(h("table", { cls: "grid seq" }, h("thead", null, head), body));
    return box;
  }

  function render(ctx) {
    var M = ctx.M, phases = M.phasesFor(DM.state.phase);
    cur = { rows: [] };
    ctx.tools.appendChild(DM.phasePicker(M));
    ctx.tools.appendChild(h("span", { cls: "legend" }, h("span", { cls: "lg dim" },
      "one occurrence of a phase, each launch and copy in the order issued; click a row for its step: all the step's launches are marked")));
    var wrap = h("div", { cls: "seq-wrap" });
    phases.forEach(function (P) { wrap.appendChild(phaseTable(M, P)); });
    if (!phases.length) wrap.appendChild(h("div", { cls: "empty" }, "This case has no phases."));
    ctx.body.appendChild(wrap);
  }

  // The rows of the selected step, or of every step of the selected kernel.
  function select(res, reveal) {
    if (!cur) return;
    var first = null;
    cur.rows.forEach(function (x) {
      var on = !!res && (((res.kind === "step" || res.kind === "access") && x.T === res.T) || (res.kind === "kernel" && x.T.kernel === res.K));
      x.tr.classList.toggle("sel", on);
      if (on && !first) first = x.tr;
    });
    if (reveal && first && first.scrollIntoView) first.scrollIntoView({ block: "center" });
  }

  DM.registerView({ id: "sequence", title: "Sequence", phased: true, render: render, select: select });
})();
