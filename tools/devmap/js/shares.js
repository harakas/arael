// devmap view: Shares. What each kernel takes of an iteration (or of one
// phase, or of the whole solve), by area: a treemap of phase > stage >
// kernel, and the same kernels ranked as bars beside it. The measure is
// the device time of the launches where the case has it (a timed trace),
// or the bytes they move (a probed case).
"use strict";
(function () {
  var DM = window.DM, h = DM.h;

  var ui = { metric: "", span: "iter" };
  var cur = null;     // { leafEls: { kernel name: [el] }, rowEls: { kernel name: [tr] } }

  var HEAD_P = 19, HEAD_S = 15, PAD = 2;   // header heights and the padding inside a group, px
  var LIST_W = 500;                        // the ranked list's width, px
  var LIST_MAX = 60;

  function isIndex(S) { return S.ty === "u32" || S.ty === "u64"; }

  // A step's amount of each measure, or null where the case does not have it.
  var METRICS = [
    { id: "time", title: "device time", fmt: DM.fmtTime,
      doc: "the launches' time on the device, each timed alone",
      of: function (T) { return T.time !== null ? T.time * T.launches : null; } },
    { id: "read", title: "bytes read", fmt: DM.fmtBytes,
      doc: "bytes the kernels read (measured)",
      of: function (T) { return T.rN && T.rUnk === T.rN ? null : T.rBytes; } },
    { id: "index", title: "index bytes read", fmt: DM.fmtBytes,
      doc: "bytes read from index sections (u32, u64): plans, tables, index columns",
      of: function (T) {
        var n = 0, any = false;
        T.acc.forEach(function (A) {
          if (A.r && A.r.bytes !== null) {
            any = true;
            if (isIndex(A.S)) n += A.r.bytes;
          }
        });
        return any ? n : null;
      } },
    { id: "write", title: "bytes written", fmt: DM.fmtBytes,
      doc: "bytes the kernels wrote (measured)",
      of: function (T) { return T.wN && T.wUnk === T.wN ? null : T.wBytes; } }
  ];

  function metricById(id) {
    for (var i = 0; i < METRICS.length; i++) if (METRICS[i].id === id) return METRICS[i];
    return null;
  }

  // Whether the case has a measure: some launch with an amount above zero.
  function has(M, m) {
    return M.steps.some(function (T) {
      if (T.kind !== "launch") return false;
      var v = m.of(T);
      return v !== null && v > 0;
    });
  }

  // The phases a span covers and how often each counts.
  //   iter:  one linearization and the attempts that go with it
  //   solve: every phase as often as it ran
  //   <id>:  one occurrence of that phase
  function spanOf(M, span) {
    var by = M.phaseById;
    if (span === "iter") {
      var lin = by.linearize || by.compute, att = by.attempt;
      if (lin && att) {
        var nLin = (by.linearize ? by.linearize.times : 0) + (by.compute ? by.compute.times : 0);
        return { title: "one iteration", parts: [{ P: lin, w: 1 }, { P: att, w: att.times / Math.max(nLin, 1) }] };
      }
      span = "solve";
    }
    if (span !== "solve" && by[span]) return { title: "one " + by[span].title.toLowerCase(), parts: [{ P: by[span], w: 1 }] };
    return { title: "the whole solve", parts: M.phases.map(function (P) { return { P: P, w: P.times }; }) };
  }

  // The hue of a phase: linearizations, attempts, the rest.
  function hueOf(P) {
    return P.id === "linearize" || P.id === "compute" ? "lin" : P.id === "attempt" ? "att" : "oth";
  }

  // phase > stage > kernel, each with its amount v; sorted largest first.
  function build(M, m, span) {
    var root = { v: 0, kids: [] };
    span.parts.forEach(function (part) {
      var P = part.P, g = { P: P, times: part.w, title: P.title, v: 0, kids: [], by: Object.create(null) };
      P.steps.forEach(function (T) {
        if (T.kind !== "launch") return;
        var v = m.of(T);
        if (v === null || !(v > 0)) return;
        v *= part.w;
        var key = T.scope.length > 1 ? T.scope[1] : "";
        var st = g.by[key];
        if (!st) {
          st = g.by[key] = { key: key, title: key ? M.scopeInfo(P.id + "/" + key).title : "(in " + P.title + ")", v: 0, kids: [], by: Object.create(null) };
          g.kids.push(st);
        }
        var leaf = st.by[T.kname];
        if (!leaf) {
          leaf = st.by[T.kname] = { name: T.kname, T: T, P: P, stage: st, v: 0, launches: 0, steps: 0 };
          st.kids.push(leaf);
        }
        leaf.v += v;
        leaf.launches += T.launches * part.w;
        leaf.steps++;
        st.v += v;
        g.v += v;
      });
      if (g.v > 0) {
        root.kids.push(g);
        root.v += g.v;
      }
    });
    var desc = function (a, b) { return b.v - a.v; };
    root.kids.sort(desc);
    root.kids.forEach(function (g) {
      g.kids.sort(desc);
      g.kids.forEach(function (st) { st.kids.sort(desc); });
    });
    return root;
  }

  // Squarified treemap (Bruls, Huizing, van Wijk): items, largest first,
  // each with an amount v, laid into the rectangle; sets x, y, w, h.
  function squarify(items, x, y, w, hh) {
    var total = 0;
    items.forEach(function (it) { total += it.v; });
    if (!(total > 0) || w <= 0 || hh <= 0) {
      items.forEach(function (it) { it.x = x; it.y = y; it.w = 0; it.h = 0; });
      return;
    }
    var scale = w * hh / total;
    var rx = x, ry = y, rw = w, rh = hh;
    // the worst aspect ratio of a row of areas along a side
    function worst(areas, sum, side) {
      var max = 0, min = Infinity;
      areas.forEach(function (a) { if (a > max) max = a; if (a < min) min = a; });
      var s2 = side * side, r2 = sum * sum;
      return Math.max(s2 * max / r2, r2 / (s2 * min));
    }
    function place(row, areas, sum) {
      var side = Math.min(rw, rh), thick = sum / side, at;
      if (rw >= rh) {
        at = ry;
        row.forEach(function (it, k) { var len = areas[k] / thick; it.x = rx; it.y = at; it.w = thick; it.h = len; at += len; });
        rx += thick;
        rw -= thick;
      } else {
        at = rx;
        row.forEach(function (it, k) { var len = areas[k] / thick; it.x = at; it.y = ry; it.w = len; it.h = thick; at += len; });
        ry += thick;
        rh -= thick;
      }
    }
    var row = [], areas = [], sum = 0;
    items.forEach(function (it) {
      var a = it.v * scale, side = Math.min(rw, rh);
      if (row.length && worst(areas.concat([a]), sum + a, side) > worst(areas, sum, side)) {
        place(row, areas, sum);
        row = [];
        areas = [];
        sum = 0;
      }
      row.push(it);
      areas.push(a);
      sum += a;
    });
    if (row.length) place(row, areas, sum);
  }

  function pct(v, total) { return DM.fmtPct(total > 0 ? v / total : 0); }

  // How often a phase counts in the span: "1", "1.14", "19".
  function times(g) { return g.times % 1 ? g.times.toFixed(2) : String(g.times); }

  function leafTip(leaf, m, root) {
    var per = m.id === "time" && leaf.launches > 0 ? "\n" + DM.fmtTime(leaf.v / leaf.launches) + " a launch" : "";
    return leaf.name + "\n" + leaf.P.title + " / " + leaf.stage.title + "\n" +
      m.title + ": " + m.fmt(leaf.v) + "  (" + pct(leaf.v, root.v) + " of the whole, " + pct(leaf.v, leaf.stage.v) + " of its stage)\n" +
      DM.fmtNum(leaf.launches, leaf.launches % 1 ? 1 : 0) + " launches" + per + "\nclick: the step's details";
  }

  function rect(el, x, y, w, hh) {
    el.style.left = x + "px";
    el.style.top = y + "px";
    el.style.width = Math.max(w, 0) + "px";
    el.style.height = Math.max(hh, 0) + "px";
  }

  // The treemap: phases, in them stages, in them kernels. A group has a
  // header where it is large enough; a kernel its name, and its share and
  // amount where they fit.
  function treemap(M, m, root, W, H) {
    var box = h("div", { cls: "tm", style: "width:" + W + "px;height:" + H + "px", role: "img",
      "aria-label": "treemap of " + m.title + " by phase, stage and kernel; the ranked list beside it has the same figures" });
    squarify(root.kids, 0, 0, W, H);
    root.kids.forEach(function (g) {
      var hue = hueOf(g.P);
      var ge = h("div", { cls: "tm-phase hue-" + hue });
      rect(ge, g.x + 1, g.y + 1, g.w - 2, g.h - 2);
      var headP = g.h > 2 * HEAD_P && g.w > 90;
      if (headP) {
        ge.appendChild(h("div", { cls: "tm-head", tip: g.title + ", counted " + times(g) + " times: " + m.fmt(g.v) + ", " + pct(g.v, root.v) },
          h("b", null, g.title), h("span", { cls: "tm-x" }, "x" + times(g)),
          h("span", { cls: "tm-share" }, pct(g.v, root.v)), h("span", { cls: "tm-amt" }, m.fmt(g.v))));
      } else {
        ge.setAttribute("data-tip", g.title + ": " + m.fmt(g.v) + ", " + pct(g.v, root.v));
      }
      var ix = PAD, iy = (headP ? HEAD_P : PAD), iw = g.w - 2 - 2 * PAD, ih = g.h - 2 - iy - PAD;
      squarify(g.kids, ix, iy, iw, ih);
      g.kids.forEach(function (st) {
        var se = h("div", { cls: "tm-stage" });
        rect(se, st.x + 1, st.y + 1, st.w - 2, st.h - 2);
        var headS = st.h > 2.6 * HEAD_S && st.w > 70;
        if (headS) {
          se.appendChild(h("div", { cls: "tm-shead", tip: g.title + " / " + st.title + ": " + m.fmt(st.v) + ", " + pct(st.v, root.v) + " of the whole" },
            h("span", { cls: "tm-sname" }, st.title), h("span", { cls: "tm-share" }, pct(st.v, root.v))));
        }
        var sx = 0, sy = headS ? HEAD_S : 0, sw = st.w - 2, sh = st.h - 2 - sy;
        squarify(st.kids, sx, sy, sw, sh);
        st.kids.forEach(function (leaf) {
          var le = h("div", { cls: "tm-leaf", tip: leafTip(leaf, m, root), "data-k": leaf.name,
            onclick: function () { DM.select(DM.sel.step(leaf.T)); } });
          // a 2px gap between neighbours
          rect(le, leaf.x + 1, leaf.y + 1, leaf.w - 2, leaf.h - 2);
          var lw = leaf.w - 2, lh = leaf.h - 2;
          if (lw >= 34 && lh >= 14) {
            le.appendChild(h("div", { cls: "tm-name" }, DM.fit(leaf.name, lw - 6, 10.5)));
            if (lh >= 28 && lw >= 44) le.appendChild(h("div", { cls: "tm-fig" }, DM.fit(pct(leaf.v, root.v) + "  " + m.fmt(leaf.v), lw - 6, 10)));
          }
          (cur.leafEls[leaf.name] = cur.leafEls[leaf.name] || []).push(le);
          se.appendChild(le);
        });
        ge.appendChild(se);
      });
      box.appendChild(ge);
    });
    return box;
  }

  // The same kernels ranked, largest first: the figures to compare exactly.
  function ranked(M, m, root) {
    var leaves = [];
    root.kids.forEach(function (g) { g.kids.forEach(function (st) { st.kids.forEach(function (l) { leaves.push(l); }); }); });
    leaves.sort(function (a, b) { return b.v - a.v; });
    var top = leaves.length ? leaves[0].v : 1;
    var t = h("table", { cls: "grid sh-list" }, h("tr", null,
      h("th", null, "#"), h("th", null, "kernel"),
      h("th", { tip: "share of the whole, and a bar of it against the largest; the bar's colour is the phase" }, "share"),
      h("th", { cls: "numc" }, m.id === "time" ? "time" : "bytes"),
      m.id === "time" ? h("th", { cls: "numc", tip: "its mean time a launch; the launches are in the row's tooltip" }, "a launch") : null,
      h("th", { tip: "the stage of its phase" }, "stage")));
    var cum = 0;
    leaves.slice(0, LIST_MAX).forEach(function (l, i) {
      cum += l.v;
      var tr = h("tr", { "data-k": l.name, tip: leafTip(l, m, root), onclick: function () { DM.select(DM.sel.step(l.T)); } },
        h("td", { cls: "numc dim" }, String(i + 1)),
        h("td", { cls: "mono" }, l.name),
        h("td", { cls: "sh-bar" }, h("span", { cls: "sh-pct" }, pct(l.v, root.v)),
          h("span", { cls: "sh-track" }, h("i", { cls: "hue-" + hueOf(l.P), style: "width:" + (100 * l.v / top).toFixed(1) + "%" }))),
        h("td", { cls: "numc" }, m.fmt(l.v)),
        m.id === "time" ? h("td", { cls: "numc" }, l.launches > 0 ? DM.fmtTime(l.v / l.launches) : "-") : null,
        h("td", { cls: "dim sh-stage" }, l.stage.key ? l.stage.title : l.P.title));
      (cur.rowEls[l.name] = cur.rowEls[l.name] || []).push(tr);
      t.appendChild(tr);
    });
    var wrap = h("div", { cls: "sh-side" }, h("div", { cls: "sh-side-title" }, "Ranked", h("span", { cls: "dim" }, "  " + leaves.length + " kernels by " + m.title)), t);
    if (leaves.length > LIST_MAX) {
      wrap.appendChild(h("div", { cls: "dim sh-more" }, (leaves.length - LIST_MAX) + " more, " + pct(root.v - cum, root.v) + " together"));
    }
    return wrap;
  }

  function render(ctx) {
    var M = ctx.M;
    cur = { leafEls: Object.create(null), rowEls: Object.create(null) };
    var avail = METRICS.filter(function (m) { return has(M, m); });
    if (!avail.length) {
      ctx.body.appendChild(h("div", { cls: "empty" },
        "This case has neither launch times (a timed trace: ARAEL_GPU_TRACE_TIME=1) nor measured accesses (a probed case)."));
      return;
    }
    var m = metricById(ui.metric);
    if (!m || avail.indexOf(m) < 0) m = avail[0];
    var spans = [["iter", "one iteration"], ["solve", "the whole solve"]].concat(M.phases.map(function (P) { return [P.id, "one " + P.title.toLowerCase()]; }));
    var span = spanOf(M, ui.span);

    ctx.tools.appendChild(DM.picker("area", avail.map(function (x) { return [x.id, x.title]; }), m.id, function (v) {
      ui.metric = v;
      DM.renderView(false);
    }));
    ctx.tools.appendChild(DM.picker("of", spans, ui.span, function (v) {
      ui.span = v;
      DM.renderView(false);
    }));

    var root = build(M, m, span);
    var parts = root.kids.map(function (g) {
      return g.title + (g.times === 1 ? "" : " x" + times(g)) + " " + m.fmt(g.v);
    });
    ctx.tools.appendChild(h("span", { cls: "tool-title", tip: m.doc }, span.title + ": " + m.fmt(root.v) + " of " + m.title));
    ctx.tools.appendChild(h("span", { cls: "lg dim" }, parts.join("  +  ")));
    var legend = h("span", { cls: "legend" });
    var seen = Object.create(null);
    root.kids.forEach(function (g) {
      var hue = hueOf(g.P);
      if (seen[hue]) return;
      seen[hue] = true;
      legend.appendChild(h("span", { cls: "lg" }, h("i", { cls: "sw hue-" + hue }), hue === "lin" ? "linearization" : hue === "att" ? "attempt" : "other phases"));
    });
    legend.appendChild(h("span", { cls: "lg dim" }, m.id === "time" ? "area: time on the device, each launch timed alone" : "area: " + m.doc));
    ctx.tools.appendChild(legend);

    if (!(root.v > 0)) {
      ctx.body.appendChild(h("div", { cls: "empty" }, "Nothing of " + m.title + " in " + span.title + "."));
      return;
    }
    var W = Math.max(520, ctx.body.clientWidth - LIST_W - 36), H = Math.max(380, ctx.body.clientHeight - 24);
    var wrap = h("div", { cls: "sh" });
    wrap.appendChild(treemap(M, m, root, W, H));
    wrap.appendChild(ranked(M, m, root));
    ctx.body.appendChild(wrap);
  }

  function select(res) {
    if (!cur) return;
    var mark = function (map, on) {
      Object.keys(map).forEach(function (k) { map[k].forEach(function (el) { el.classList.toggle("sel", on && k === on); }); });
    };
    var name = null;
    if (res && res.kind === "kernel") name = res.K.name;
    else if (res && (res.kind === "step" || res.kind === "access") && res.T.kernel) name = res.T.kernel.name;
    mark(cur.leafEls, name);
    mark(cur.rowEls, name);
  }

  // the treemap is laid out in pixels: again when the window changes
  var timer = null;
  window.addEventListener("resize", function () {
    if (!DM.state || DM.state.view !== "shares") return;
    if (timer) clearTimeout(timer);
    timer = setTimeout(DM.guard(function () { DM.renderView(false); }), 150);
  });

  DM.registerView({ id: "shares", title: "Shares", render: render, select: select });
})();
