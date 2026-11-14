// devmap view 1: Overview. The solve as stages in flow order, in two lanes
// (host above, device below). Loops of the flow are boxes around their
// stages. Inside a stage time runs left to right: sub-scopes are boxes in
// the device lane, uploads and downloads are arrows crossing the lanes.
//
// Layout is HTML (flex), so text wraps and clips by itself. The lanes line
// up because every stage starts at the same height: a stage nested in fewer
// loops than the deepest one gets the missing loop headers as a margin.
"use strict";
(function () {
  var DM = window.DM, h = DM.h, s = DM.s;

  var LOOP_HEAD = 24;   // a loop box's header, border included
  var LOOP_FOOT = 10;   // a loop box's bottom padding, border included
  var HEAD_H = 94;      // a stage's header
  var HOST_H = 78;      // the host lane
  var PAD = 14;
  var MAX_K = 9;        // kernels listed per sub-scope box
  var SCOPE_W = 200;    // a sub-scope item, with its kernel list
  var SCOPE_NARROW = 132;   // and without it
  var XFER_W = 104;     // an upload or download item
  var STAGE_MIN = 200;  // a stage is at least this wide

  var ui = { kernels: true };
  var stageEls = Object.create(null);

  function scopeW() {
    return ui.kernels ? SCOPE_W : SCOPE_NARROW;
  }

  function depth(list) {
    var d = 0;
    list.forEach(function (n) {
      if (n.body) d = Math.max(d, 1 + depth(n.body));
    });
    return d;
  }

  function stat(label, value, tip) {
    return h("span", { cls: "stat", tip: tip }, h("i", null, label), value);
  }

  function toolbar(tools, M) {
    var t = M.tot;
    tools.appendChild(h("span", { cls: "tool-title" }, "the solve"));
    tools.appendChild(stat("phases", String(M.phases.length)));
    tools.appendChild(stat("launches", DM.fmtInt(t.launches), "kernel launches, each phase times its occurrences"));
    tools.appendChild(stat("device read", DM.amountEl(t.r, t.rUnk, "rd"), "bytes the kernels read (sum of access)"));
    tools.appendChild(stat("device written", DM.amountEl(t.w, t.wUnk, "wr"), "bytes the kernels wrote (sum of access)"));
    tools.appendChild(stat("uploaded", DM.bytesEl(t.up), "host to device copies"));
    tools.appendChild(stat("downloaded", DM.bytesEl(t.down), "device to host copies"));
    tools.appendChild(stat("host waits", DM.fmtInt(t.waits), "downloads: each one is a point where the host waits for the device"));
    if (t.hasTime) tools.appendChild(stat("kernel time", DM.fmtTime(t.time)));
    tools.appendChild(DM.checkbox("kernel lists", ui.kernels, function (on) {
      ui.kernels = on;
      DM.renderView(false);
    }, "list the kernels inside each sub-scope box (off: a narrower diagram)"));
    tools.appendChild(h("span", { cls: "legend" },
      h("span", { cls: "lg" }, h("b", { cls: "wr" }, DM.CH.down), " upload"),
      h("span", { cls: "lg" }, h("b", { cls: "rd" }, DM.CH.up), " download: the host waits"),
      h("span", { cls: "lg dim" }, "numbers in a stage are per occurrence; click a stage or a box to open it in the pipeline")));
  }

  // The steps of a phase as a time-ordered row: runs of launches in one
  // sub-scope, and runs of uploads or downloads.
  function stageItems(P) {
    var items = [], cur = null;
    P.steps.forEach(function (T) {
      var kind, key, path = "";
      if (T.kind === "upload" || T.kind === "download") {
        kind = T.kind;
        key = T.kind;
      } else {
        kind = "scope";
        path = T.scope.slice(0, 2).join("/");
        key = "scope:" + path;
      }
      if (!cur || cur.key !== key) {
        cur = { kind: kind, key: key, path: path, steps: [] };
        items.push(cur);
      }
      cur.steps.push(T);
    });
    return items;
  }

  function arrow(up) {
    return s("svg", { cls: "ov-arrow " + (up ? "up" : "down"), width: 14, height: 30, viewBox: "0 0 14 30", style: "top:" + (HOST_H - 15) + "px" },
      s("path", { d: up ? "M7 0 V28 M2 21 L7 29 L12 21" : "M7 30 V2 M2 9 L7 1 L12 9" }));
  }

  function xferItem(it) {
    var up = it.kind === "upload", bytes = 0, n = it.steps.length, first = it.steps[0];
    it.steps.forEach(function (T) { bytes += T.bytes; });
    var names = it.steps.map(function (T) { return T.target; });
    var tip = (up ? "upload: host -> device" : "download: device -> host, the host waits for the device here") + "\n" +
      it.steps.map(function (T) { return "  " + T.target + "  " + DM.fmtBytes(T.bytes) + (T.via ? "  " + T.via : ""); }).join("\n");
    var targets = names.slice(0, 3).map(function (nm) { return h("div", { cls: "ov-x-name" }, nm); });
    if (names.length > 3) targets.push(h("div", { cls: "ov-x-name dim" }, "+" + (names.length - 3) + " more"));
    return h("div", { cls: "ov-item ov-xfer " + (up ? "up" : "down"), style: "width:" + XFER_W + "px", tip: tip,
      onclick: function () { DM.select(DM.sel.step(first), { reveal: false }); } },
      h("div", { cls: "ov-host", style: "height:" + HOST_H + "px" },
        up ? null : h("div", { cls: "ov-wait" }, "host waits"),
        h("div", { cls: "ov-x-label" }, (up ? "upload" : "download") + (n > 1 ? " x" + n : "")),
        h("div", { cls: "ov-x-bytes" }, DM.fmtBytes(bytes))),
      arrow(up),
      h("div", { cls: "ov-dev" }, h("div", { cls: "ov-x-target" }, targets)));
  }

  function scopeItem(it, P, M) {
    var info = M.scopeInfo(it.path), tot = DM.sumSteps(it.steps), top = it.path.indexOf("/") < 0;
    var ks = [], by = Object.create(null);
    it.steps.forEach(function (T) {
      var name = T.kname || T.kind, k = by[name];
      if (!k) {
        k = by[name] = { name: name, launches: 0 };
        ks.push(k);
      }
      k.launches += T.launches;
    });
    var tip = (top ? P.title + " (steps directly in the phase)" : info.title + "   " + it.path) + (info.doc && !top ? "\n" + info.doc : "") + "\n" +
      ks.map(function (k) { return "  " + k.name + "  x" + k.launches; }).join("\n") +
      "\nread " + DM.fmtAmount(tot.r, tot.rUnk) + ", written " + DM.fmtAmount(tot.w, tot.wUnk) + " per occurrence" +
      (tot.hasTime ? "\ntime " + DM.fmtTime(tot.time) : "") + "\nclick: open in the pipeline";
    var list = ks.slice(0, MAX_K).map(function (k) {
      return h("div", { cls: "ov-k" }, h("span", { cls: "ov-k-name" }, k.name), h("span", { cls: "ov-k-n" }, "x" + DM.fmtInt(k.launches)));
    });
    if (ks.length > MAX_K) {
      var rest = 0;
      ks.slice(MAX_K).forEach(function (k) { rest += k.launches; });
      list.push(h("div", { cls: "ov-k dim" }, h("span", { cls: "ov-k-name" }, "+" + (ks.length - MAX_K) + " more kernels"), h("span", { cls: "ov-k-n" }, "x" + DM.fmtInt(rest))));
    }
    var box = h("div", { cls: "ov-box" + (top ? " top" : ""), tip: tip, onclick: function () { DM.select(DM.sel.scope(it.path), { view: "pipeline", phase: P.id }); } },
      h("div", { cls: "ov-box-title" }, top ? "(in " + P.title + ")" : info.title),
      info.doc && !top && ui.kernels ? h("div", { cls: "ov-box-doc" }, info.doc) : null,
      ui.kernels ? h("div", { cls: "ov-klist" }, list) : h("div", { cls: "ov-k dim" }, ks.length + (ks.length === 1 ? " kernel" : " kernels")),
      h("div", { cls: "ov-box-tot" }, h("span", null, DM.fmtInt(tot.launches) + " launches"),
        DM.uncounted(tot.r, tot.rUnk, tot.w, tot.wUnk) ? null :
          [h("span", { cls: "rd" }, "R " + DM.fmtAmount(tot.r, tot.rUnk, true)), h("span", { cls: "wr" }, "W " + DM.fmtAmount(tot.w, tot.wUnk, true))]));
    return h("div", { cls: "ov-item ov-scope", style: "width:" + scopeW() + "px" },
      h("div", { cls: "ov-host", style: "height:" + HOST_H + "px" }), h("div", { cls: "ov-dev" }, box));
  }

  function stageEl(node, level, D, M) {
    // The width is set here, from the items, so that the header's text never widens a stage.
    var P = node.P, items = P ? stageItems(P) : [], width = 0;
    items.forEach(function (it) { width += it.kind === "scope" ? scopeW() : XFER_W; });
    var style = "margin-top:" + ((D - level) * LOOP_HEAD) + "px;margin-bottom:" + ((D - level) * LOOP_FOOT) + "px;width:" + Math.max(STAGE_MIN, width) + "px";
    if (!P) {
      return h("div", { cls: "ov-stage missing", style: style },
        h("div", { cls: "ov-head", style: "height:" + HEAD_H + "px" }, h("div", { cls: "ov-title" }, h("span", { cls: "ov-name" }, node.id)),
          h("div", { cls: "ov-doc" }, "named by flow, but not in phases")));
    }
    var t = P.tot;
    var el = h("div", { cls: "ov-stage" + (node.loose ? " loose" : ""), style: style });
    stageEls[P.id] = el;
    var tip = P.title + "   x" + DM.fmtInt(P.times) + (P.doc ? "\n" + P.doc : "") +
      "\nper occurrence: " + t.steps + " steps, " + t.launches + " launches, read " + DM.fmtAmount(t.r, t.rUnk) + ", written " + DM.fmtAmount(t.w, t.wUnk) +
      "\nuploads " + t.nUp + " (" + DM.fmtBytes(t.up) + "), downloads " + t.nDown + " (" + DM.fmtBytes(t.down) + ")" +
      (t.hasTime ? "\nkernel time " + DM.fmtTime(t.time) : "") +
      (P.variants > 1 ? "\n" + P.variants + " distinct step sequences; the most frequent one is shown" : "") +
      "\nclick: open in the pipeline";
    el.appendChild(h("div", { cls: "ov-head", style: "height:" + HEAD_H + "px", tip: tip, onclick: function () { DM.select(DM.sel.scope(P.id), { view: "pipeline", phase: P.id }); } },
      h("div", { cls: "ov-title" }, h("span", { cls: "ov-name" }, P.title), h("span", { cls: "ov-times" }, "x" + DM.fmtInt(P.times)),
        P.variants > 1 ? h("span", { cls: "ov-var" }, P.variants + " variants") : null,
        node.loose ? h("span", { cls: "ov-var" }, "not in flow") : null),
      h("div", { cls: "ov-doc" }, P.doc),
      h("div", { cls: "ov-tot" }, h("span", null, DM.fmtInt(t.launches) + " launches"),
        DM.uncounted(t.r, t.rUnk, t.w, t.wUnk) ? null :
          [h("span", { cls: "rd" }, "R " + DM.fmtAmount(t.r, t.rUnk)), h("span", { cls: "wr" }, "W " + DM.fmtAmount(t.w, t.wUnk))]),
      h("div", { cls: "ov-tot" }, h("span", { cls: t.nUp ? null : "dim" }, "up " + (t.nUp ? DM.fmtBytes(t.up) : "-")),
        h("span", { cls: t.nDown ? null : "dim" }, "down " + (t.nDown ? DM.fmtBytes(t.down) : "-")),
        t.nDown ? h("span", { cls: "ov-waits" }, t.nDown + (t.nDown > 1 ? " waits" : " wait")) : null,
        t.hasTime ? h("span", { cls: "dim" }, DM.fmtTime(t.time)) : null)));
    var row = h("div", { cls: "ov-items" });
    if (!items.length) {
      row.appendChild(h("div", { cls: "ov-item", style: "width:" + STAGE_MIN + "px" }, h("div", { cls: "ov-host", style: "height:" + HOST_H + "px" }),
        h("div", { cls: "ov-dev" }, h("div", { cls: "ov-none" }, "no steps"))));
    }
    items.forEach(function (it) {
      row.appendChild(it.kind === "scope" ? scopeItem(it, P, M) : xferItem(it));
    });
    el.appendChild(row);
    return el;
  }

  function loopEl(node, level, D, M) {
    var body = h("div", { cls: "ov-loop-body" });
    addNodes(body, node.body, level + 1, D, M);
    return h("div", { cls: "ov-loop", style: "padding-bottom:" + (LOOP_FOOT - 1) + "px" },
      h("div", { cls: "ov-loop-head", style: "height:" + (LOOP_HEAD - 1) + "px", tip: "loop: " + node.loop }, h("b", null, DM.CH.loop), " " + node.loop),
      body);
  }

  function addNodes(parent, list, level, D, M) {
    list.forEach(function (node, i) {
      if (i > 0) parent.appendChild(h("div", { cls: "ov-next", style: "margin-top:" + ((D - level) * LOOP_HEAD + 8) + "px" }, DM.CH.right));
      parent.appendChild(node.body ? loopEl(node, level, D, M) : stageEl(node, level, D, M));
    });
  }

  function render(ctx) {
    var M = ctx.M;
    stageEls = Object.create(null);
    toolbar(ctx.tools, M);
    if (!M.flow.length) {
      ctx.body.appendChild(h("div", { cls: "empty" }, "This case has no phases."));
      return;
    }
    var D = depth(M.flow), hostTop = PAD + D * LOOP_HEAD + HEAD_H;
    var wrap = h("div", { cls: "ov-wrap" });
    wrap.appendChild(h("div", { cls: "ov-lane ov-lane-host", style: "top:" + hostTop + "px;height:" + HOST_H + "px" },
      h("span", { cls: "ov-lane-label" }, "host")));
    wrap.appendChild(h("div", { cls: "ov-lane ov-lane-dev", style: "top:" + (hostTop + HOST_H) + "px;bottom:" + (PAD + D * LOOP_FOOT) + "px" },
      h("span", { cls: "ov-lane-label" }, "device")));
    var flow = h("div", { cls: "ov-flow", style: "padding:" + PAD + "px " + PAD + "px " + PAD + "px 70px" });
    addNodes(flow, M.flow, 0, D, M);
    wrap.appendChild(flow);
    ctx.body.appendChild(wrap);
  }

  function select(res, reveal) {
    Object.keys(stageEls).forEach(function (k) { stageEls[k].classList.remove("sel"); });
    if (!res) return;
    var pid = res.kind === "scope" ? res.path.split("/")[0] : (res.kind === "step" || res.kind === "access") ? res.T.phase.id : null;
    var el = pid !== null ? stageEls[pid] : null;
    if (!el) return;
    el.classList.add("sel");
    if (reveal && el.scrollIntoView) el.scrollIntoView({ block: "nearest", inline: "nearest" });
  }

  DM.registerView({ id: "overview", title: "Overview", render: render, select: select });
})();
