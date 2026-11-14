// devmap view 4: Memory. Each block is a bar cut into its sections in
// memory order; widths follow the bytes (linear or log) with a minimum so
// that small sections stay clickable. Gaps between sections are drawn as
// gaps. Colour is the stored type, or (heat) the bytes read or written.
"use strict";
(function () {
  var DM = window.DM, h = DM.h;

  var MIN_S = 18, MIN_G = 5;      // minimum width of a section and of a gap, px
  var ui = { scale: "linear", color: "type", table: true };
  var cur = null;                 // { segEls{key}, rowEls{key}, blockEls{id} }

  // A block as sections and gaps, in memory order.
  function segments(B) {
    var out = [], cursor = 0;
    B.order.forEach(function (S) {
      if (S.at > cursor) out.push({ gap: true, tail: false, at: cursor, bytes: S.at - cursor, S: null });
      out.push({ gap: false, tail: false, at: S.at, bytes: S.bytes, S: S, overlap: S.at < cursor });
      cursor = Math.max(cursor, S.at + S.bytes);
    });
    if (B.bytes > cursor) out.push({ gap: true, tail: true, at: cursor, bytes: B.bytes - cursor, S: null });
    return out;
  }

  // Bytes read (or written) per section: one occurrence of a phase, or the
  // whole solve when the phase is "all".
  function heatOf(M, phaseId, which) {
    var all = phaseId === "all" || !M.phaseById[phaseId];
    var heat = { val: Object.create(null), unk: Object.create(null), lo: Infinity, hi: 0,
                 cssVar: which === "read" ? "rd" : "wr",
                 label: (which === "read" ? "read" : "written") + (all ? " in the solve" : " per occurrence of " + M.phaseById[phaseId].title) };
    M.sections.forEach(function (S) {
      var sum = 0, any = false;
      S.touches.forEach(function (A) {
        var P = A.T.phase;
        if (!all && P.id !== phaseId) return;
        var d = which === "read" ? A.r : A.w;
        if (!d) return;
        any = true;
        if (d.bytes === null) {
          heat.unk[S.key] = true;
          return;
        }
        sum += d.bytes * (all ? P.times : 1);
      });
      if (!any) return;
      heat.val[S.key] = sum;
      if (sum > 0) {
        if (sum < heat.lo) heat.lo = sum;
        if (sum > heat.hi) heat.hi = sum;
      }
    });
    return heat;
  }

  // The group labels above a bar: runs of sections with one group.
  function bandRow(segs) {
    var bands = [], last = null;
    segs.forEach(function (g, i) {
      var grp;
      if (g.gap) {
        // a gap continues the band when the sections on both sides share it
        var a = segs[i - 1], b = segs[i + 1];
        grp = a && b && !a.gap && !b.gap && a.S.group && a.S.group === b.S.group ? a.S.group : "";
      } else {
        grp = g.S.group;
      }
      if (last && last.group === grp) last.px += g.px;
      else {
        last = { group: grp, px: g.px };
        bands.push(last);
      }
    });
    var row = h("div", { cls: "mem-bands" });
    bands.forEach(function (b) {
      row.appendChild(h("div", { cls: b.group ? "band" : "band none", style: "width:" + (Math.round(b.px * 10) / 10) + "px", tip: b.group ? "group: " + b.group : "" }, b.group));
    });
    return row;
  }

  function th(text, tip) { return h("th", { tip: tip }, text); }

  function blockEl(M, B, avail, heat) {
    var segs = segments(B), sumW = 0, sumMin = 0, hasGroup = false;
    segs.forEach(function (g) {
      g.wt = ui.scale === "log" ? Math.log(1 + Math.max(0, g.bytes)) / Math.LN2 : Math.max(0, g.bytes);
      sumW += g.wt;
      sumMin += g.gap ? MIN_G : MIN_S;
      if (!g.gap && g.S.group) hasGroup = true;
    });
    var free = Math.max(0, avail - sumMin);
    segs.forEach(function (g) {
      g.px = Math.round(((g.gap ? MIN_G : MIN_S) + (sumW > 0 ? g.wt / sumW * free : 0)) * 10) / 10;
    });

    var gapBytes = Math.max(0, B.bytes - B.used);
    var el = h("div", { cls: "mem-block" });
    cur.blockEls[B.id] = el;
    el.appendChild(h("div", { cls: "mem-head" },
      h("span", { cls: "mem-name lnk", tip: "block " + B.id + ": click for its details", onclick: function () { DM.select(DM.sel.block(B), { reveal: false }); } }, B.name),
      B.owner ? h("span", { cls: "mem-owner" }, "owner ", h("b", null, B.owner)) : null,
      DM.bytesEl(B.bytes, "mem-bytes"),
      h("span", { cls: "dim" }, B.sections.length + " sections, " + DM.fmtBytes(gapBytes) + " in gaps (" + DM.fmtPct(B.bytes > 0 ? gapBytes / B.bytes : 0) + ")"),
      B.missing ? h("span", { cls: "bad" }, "not listed in blocks") : null,
      B.doc ? h("span", { cls: "mem-doc" }, B.doc) : null));

    var scroll = h("div", { cls: "mem-scroll" });
    if (hasGroup) scroll.appendChild(bandRow(segs));
    var bar = h("div", { cls: "mem-bar" });
    segs.forEach(function (g) {
      if (g.gap) {
        bar.appendChild(h("div", { cls: "gap" + (g.tail ? " tail" : ""), style: "width:" + g.px + "px",
          tip: (g.tail ? "unused tail: " : "gap: ") + DM.fmtBytesFull(g.bytes) + " at " + DM.fmtInt(g.at) }));
        return;
      }
      var S = g.S, cls = "seg " + DM.tyClass(S.ty), style = "width:" + g.px + "px", lv = DM.worstLevel(S.flags), tip = DM.secTip(S);
      if (heat) {
        var v = heat.val[S.key];
        if (v === undefined) {
          cls += " cold";
        } else if (v > 0) {
          var a = DM.heatAlpha(DM.logScale(v, heat.lo, heat.hi));
          cls += " hot" + (a > 0.55 ? " hi" : "");
          style += ";background-color:rgba(var(--" + heat.cssVar + ")," + a + ")";
        } else {
          cls += " hot unk";
        }
        tip += "\n\n" + heat.label + ": " + (v === undefined ? "nothing" : DM.fmtBytesFull(v)) + (heat.unk[S.key] ? " (+ uncounted)" : "");
      }
      if (lv === "bug" || lv === "warn") cls += " flag-" + lv;
      if (g.overlap) cls += " overlap";
      var seg = h("div", { cls: cls, style: style, "data-key": S.key, tip: tip },
        h("span", { cls: "seg-name" }, S.name), h("span", { cls: "seg-bytes" }, DM.fmtBytes(S.bytes)));
      cur.segEls[S.key] = seg;
      bar.appendChild(seg);
    });
    scroll.appendChild(bar);
    el.appendChild(scroll);

    if (ui.table) {
      var t = h("table", { cls: "grid mem-table" }, h("tr", null, th("i"), th("section"), th("type", "stored type, with the kernel-side role"), th("group"),
        th("at", "byte offset in the block"), th("bytes"), th("elements", "count x element size"),
        th("computable", "index sections: what computes the words from their position (i: the word's index, r: the record's index, ?: a field nothing computes), " +
          "and other sections that hold the same words or a multiple of them; no: neither"),
        th("read", "bytes read in the whole solve"), th("written", "bytes written in the whole solve"),
        th("R / W", "steps that read / write it (per occurrence of their phases)"), th("checks"), th("doc")));
      B.order.forEach(function (S) {
        var tr = h("tr", { "data-key": S.key },
          h("td", { cls: "numc dim" }, String(S.i)),
          h("td", { cls: "mono" }, S.name),
          h("td", null, DM.tyEl(S.ty, S.role)),
          h("td", { cls: "dim" }, S.group),
          h("td", { cls: "numc" }, DM.fmtInt(S.at)),
          h("td", { cls: "numc" }, DM.bytesEl(S.bytes)),
          h("td", { cls: "numc" }, DM.fmtInt(S.count) + " x " + S.elem),
          h("td", { cls: "mem-td-cmp" }, DM.computableEl(S, true)),
          h("td", { cls: "numc rd" }, S.tot.readers ? DM.amountEl(S.tot.r, S.tot.rUnk) : "-"),
          h("td", { cls: "numc wr" }, S.tot.writers ? DM.amountEl(S.tot.w, S.tot.wUnk) : "-"),
          h("td", { cls: "numc" }, S.tot.readers + " / " + S.tot.writers),
          h("td", null, S.flags.map(function (f) { return h("span", { cls: "fl lv-" + f.level, tip: f.title + "\n" + f.text }, f.code); })),
          h("td", { cls: "mem-td-doc" }, S.doc));
        cur.rowEls[S.key] = tr;
        t.appendChild(tr);
      });
      el.appendChild(h("details", { cls: "fold", open: true }, h("summary", null, "sections of " + B.name), h("div", { cls: "tscroll" }, t)));
    }
    return el;
  }

  function legend(M, heat) {
    var box = h("span", { cls: "legend" }), seen = [];
    M.sections.forEach(function (S) {
      var ty = S.ty || "?";
      if (seen.indexOf(ty) < 0) seen.push(ty);
    });
    DM.TYPES.concat(seen).forEach(function (ty, i, all) {
      if (all.indexOf(ty) !== i || seen.indexOf(ty) < 0) return;
      box.appendChild(h("span", { cls: "lg" }, DM.tyEl(ty)));
    });
    box.appendChild(h("span", { cls: "lg" }, h("i", { cls: "sw gapsw" }), "gap"));
    box.appendChild(h("span", { cls: "lg" }, h("i", { cls: "sw flag" }), "has checks"));
    if (heat) box.appendChild(h("span", { cls: "lg" }, h("i", { cls: "sw " + heat.cssVar }), heat.label + " (log scale; the top strip stays the type)"));
    return box;
  }

  function render(ctx) {
    var M = ctx.M, st = ctx.state;
    ctx.tools.appendChild(DM.picker("scale", [["linear", "linear"], ["log", "log"]], ui.scale, function (v) {
      ui.scale = v;
      DM.renderView(false);
    }));
    ctx.tools.appendChild(DM.picker("colour", [["type", "stored type"], ["read", "heat: bytes read"], ["write", "heat: bytes written"]], ui.color, function (v) {
      ui.color = v;
      DM.renderView(false);
    }));
    if (ui.color !== "type") ctx.tools.appendChild(DM.phasePicker(M, "heat of"));
    ctx.tools.appendChild(DM.checkbox("tables", ui.table, function (on) {
      ui.table = on;
      DM.renderView(false);
    }, "the sections of each block as a table"));
    var heat = ui.color === "type" ? null : heatOf(M, st.phase, ui.color);
    ctx.tools.appendChild(legend(M, heat));
    if (M.index.words) {
      var ix = M.index;
      ctx.tools.appendChild(h("span", { cls: "lg", tip: DM.fmtInt(ix.words) + " index words (u32, u64) in " + ix.sections + " sections\n" +
        DM.fmtInt(Math.round(ix.fromPos)) + " computable from their position\n" +
        DM.fmtInt(ix.repeated) + " stored more than once (" + ix.groups + " groups of sections with the same words)" },
        "index words: ", h("b", null, DM.fmtPct(ix.fromPos / ix.words)), " computable, ", h("b", null, DM.fmtPct(ix.repeated / ix.words)), " stored twice"));
    }

    cur = { segEls: Object.create(null), rowEls: Object.create(null), blockEls: Object.create(null) };
    var wrap = h("div", { cls: "mem " + (heat ? "byheat" : "bytype") });
    ctx.body.appendChild(wrap);
    if (!M.blocks.length) {
      wrap.appendChild(h("div", { cls: "empty" }, "This case lists no blocks."));
      return;
    }
    var avail = Math.max(360, (ctx.body.clientWidth || 1000) - 44);
    M.blocks.forEach(function (B) {
      wrap.appendChild(blockEl(M, B, avail, heat));
    });

    DM.on(wrap, "click", function (e) {
      var t = e.target;
      if (!t || !t.closest) return;
      var el = t.closest("[data-key]");
      if (el) DM.select("section:" + el.getAttribute("data-key"), { reveal: false });
    });
  }

  function bring(el, opts) {
    if (!el || !el.scrollIntoView) return;
    try {
      el.scrollIntoView(opts);
    } catch (e) {
      el.scrollIntoView();
    }
  }

  function select(res, reveal) {
    if (!cur) return;
    Object.keys(cur.segEls).forEach(function (k) { cur.segEls[k].classList.remove("sel", "touch-r", "touch-w"); });
    Object.keys(cur.rowEls).forEach(function (k) { cur.rowEls[k].classList.remove("sel"); });
    Object.keys(cur.blockEls).forEach(function (k) { cur.blockEls[k].classList.remove("sel"); });
    if (!res) return;
    if (res.kind === "section" || res.kind === "access") {
      var seg = cur.segEls[res.S.key], row = cur.rowEls[res.S.key];
      if (seg) seg.classList.add("sel");
      if (row) row.classList.add("sel");
      if (reveal && seg) bring(seg, { block: "center", inline: "center" });
    } else if (res.kind === "block") {
      var el = cur.blockEls[res.B.id];
      if (el) {
        el.classList.add("sel");
        if (reveal) bring(el, { block: "start", inline: "nearest" });
      }
    }
    // a step or a kernel: the sections it touches
    var steps = res.kind === "step" || res.kind === "access" ? [res.T] : res.kind === "kernel" ? res.K.steps : [];
    var firstEl = null;
    steps.forEach(function (T) {
      T.acc.forEach(function (A) {
        var x = cur.segEls[A.S.key];
        if (!x) return;
        if (A.r) x.classList.add("touch-r");
        if (A.w) x.classList.add("touch-w");
        if (!firstEl) firstEl = x;
      });
    });
    if (reveal && firstEl && res.kind !== "access") bring(firstEl, { block: "center", inline: "center" });
  }

  DM.registerView({ id: "memory", title: "Memory", render: render, select: select });
})();
