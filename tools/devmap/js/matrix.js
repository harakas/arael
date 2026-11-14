// devmap view 3: Access matrix. Rows are steps in execution order, columns
// are sections grouped by block; a cell says whether the step reads (left
// half, read colour) or writes (right half, write colour) the section, how
// much (intensity, log scale of bytes) and in which pattern (the letter).
//
// The table is built as one HTML string (it can have tens of thousands of
// cells); events are delegated from the scrolling wrapper.
"use strict";
(function () {
  var DM = window.DM, h = DM.h, esc = DM.esc;

  var CELL_W = 24, HEAD_W = 300;
  var ui = { all: false };
  var cur = null;     // { M, rows[T], cols[S], colOf{key: c}, rowOf{step id: r}, groups, colEls, secThs, rowEls, hoverC }

  function dirLine(name, d, S) {
    if (!d) return name + "  -";
    if (d.bytes === null && d.n === null) {
      return name + "  yes, not measured" + (d.unseen ? "; not seen when probed" : d.borrowed ? ";  " + d.pattern + " in the probed case" : "");
    }
    var cov = DM.coverage(S, d), parts = [DM.fmtBytesFull(d.bytes)];
    if (d.n !== null) parts.push(DM.fmtInt(d.n) + " accesses");
    if (d.unique !== null) parts.push(DM.fmtInt(d.unique) + " distinct" + (cov !== null ? " (" + DM.fmtPct(cov) + ")" : ""));
    if (d.lo !== null && d.hi !== null) parts.push("range " + DM.fmtInt(d.lo) + ".." + DM.fmtInt(d.hi));
    if (d.threads !== null) parts.push(DM.fmtInt(d.threads) + " threads");
    if (d.sizes.length) parts.push("width " + d.sizes.join(","));
    parts.push(d.pattern);
    return name + "  " + parts.join(",  ");
  }

  function cellTip(T, A) {
    var S = A.S;
    return T.label + (T.launches > 1 ? " x" + T.launches : "") + "  [" + T.id + "]  " + T.scope.join(" / ") + "\n" +
      S.qname + "  " + (S.ty || "?") + (S.role ? " (role " + S.role + ")" : "") + "  " + DM.fmtBytes(S.bytes) + ", " + S.elem + "-byte elements\n" +
      dirLine("read ", A.r, S) + "\n" + dirLine("write", A.w, S) + "\n" + A.src;
  }

  // One half (or the whole) of a cell.
  function markHtml(d, which, half, lo, hi) {
    var cls = which + (half ? " hf" : ""), letter = DM.patLetter(d.pattern);
    if (d.bytes === null) return "<i class=\"" + cls + " unk" + (d.unseen ? " uns" : "") + "\">" + letter + "</i>";
    var a = DM.heatAlpha(DM.logScale(d.bytes, lo, hi));
    return "<i class=\"" + cls + (a > 0.55 ? " hi" : "") + "\" style=\"background:rgba(var(--" + which + ")," + a + ")\">" + letter + "</i>";
  }

  function legend(M) {
    var box = h("span", { cls: "legend" });
    box.appendChild(h("span", { cls: "lg" }, h("i", { cls: "sw rd" }), "read"));
    box.appendChild(h("span", { cls: "lg" }, h("i", { cls: "sw wr" }), "write"));
    if (M.hasMeasured) box.appendChild(h("span", { cls: "lg dim" }, "intensity: bytes, log scale"));
    if (M.hasUnknown) box.appendChild(h("span", { cls: "lg", tip: DM.NM_TIP }, h("i", { cls: "sw unk" }), "outlined: not measured"));
    if (M.hasBorrowed) box.appendChild(h("span", { cls: "lg", tip: "the pattern measured for the kernel and section in the probed case" }, h("i", { cls: "bor" }, "italic"), ": borrowed pattern"));
    DM.patLegend(DM.PAT_ORDER.filter(function (p) { return p !== "unknown"; })).forEach(function (el) { box.appendChild(el); });
    return box;
  }

  function render(ctx) {
    var M = ctx.M, st = ctx.state;
    ctx.tools.appendChild(DM.phasePicker(M));
    ctx.tools.appendChild(DM.checkbox("all sections", ui.all, function (on) {
      ui.all = on;
      DM.renderView(false);
    }, "also show the sections no shown step touches"));
    ctx.tools.appendChild(legend(M));

    var rows = [];
    M.phasesFor(st.phase).forEach(function (P) {
      P.steps.forEach(function (T) { rows.push(T); });
    });
    var touched = Object.create(null);
    rows.forEach(function (T) {
      T.acc.forEach(function (A) { touched[A.S.key] = true; });
    });
    var cols = [], groups = [], startOf = [];
    M.blocks.forEach(function (B) {
      var list = B.order.filter(function (S) { return ui.all || touched[S.key]; });
      if (!list.length) return;
      groups.push({ B: B, n: list.length });
      list.forEach(function (S, i) {
        startOf.push(i === 0);
        cols.push(S);
      });
    });
    cur = { M: M, rows: rows, cols: cols, groups: groups, colOf: Object.create(null), rowOf: Object.create(null), colEls: [], secThs: [], rowEls: [], hoverC: -1 };
    cols.forEach(function (S, c) { cur.colOf[S.key] = c; });
    rows.forEach(function (T, r) { cur.rowOf[T.id] = r; });

    if (!rows.length || !cols.length) {
      ctx.body.appendChild(h("div", { cls: "empty" }, !rows.length ? "No steps to show." :
        "No step shown here has access data or transfers, so there are no columns. Tick \"all sections\" to see the sections anyway."));
      return;
    }

    // the intensity scale: all byte counts shown
    var lo = Infinity, hi = 0;
    rows.forEach(function (T) {
      T.acc.forEach(function (A) {
        [A.r, A.w].forEach(function (d) {
          if (d && d.bytes !== null && d.bytes > 0) {
            if (d.bytes < lo) lo = d.bytes;
            if (d.bytes > hi) hi = d.bytes;
          }
        });
      });
    });

    var out = [];
    out.push("<table class=\"mx\" style=\"width:" + (HEAD_W + cols.length * CELL_W) + "px\"><colgroup><col style=\"width:" + HEAD_W + "px\">");
    cols.forEach(function () { out.push("<col style=\"width:" + CELL_W + "px\">"); });
    out.push("</colgroup><thead><tr class=\"mx-blocks\"><th class=\"mx-corner\" rowspan=\"2\"><div>" + rows.length + " steps x " + cols.length +
      " sections</div><div class=\"dim\">row: step, R and W bytes per occurrence</div><div class=\"dim\">column: section, by block, in memory order</div></th>");
    groups.forEach(function (g, gi) {
      out.push("<th class=\"mx-block\" colspan=\"" + g.n + "\" data-b=\"" + gi + "\" data-tip=\"" +
        esc(g.B.name + "  " + DM.fmtBytes(g.B.bytes) + (g.B.owner ? "  owner " + g.B.owner : "") + (g.B.doc ? "\n" + g.B.doc : "")) + "\"><span>" + esc(g.B.name) + "</span></th>");
    });
    out.push("</tr><tr class=\"mx-secs\">");
    cols.forEach(function (S, c) {
      out.push("<th class=\"mx-sec" + (startOf[c] ? " bs" : "") + (S.flags.length ? " lv-" + DM.worstLevel(S.flags) : "") + "\" data-c=\"" + c + "\" data-tip=\"" + esc(DM.secTip(S)) +
        "\"><div class=\"mx-sec-name\">" + esc(S.name) + "</div><i class=\"mx-ty " + DM.tyClass(S.ty) + "\"></i></th>");
    });
    out.push("</tr></thead><tbody>");

    var lastGroup = null, lastPhase = null;
    rows.forEach(function (T, r) {
      var gk = T.phase.id + "|" + T.scopeKey;
      if (gk !== lastGroup) {
        var newPhase = T.phase !== lastPhase;
        lastGroup = gk;
        lastPhase = T.phase;
        out.push("<tr class=\"mx-group" + (newPhase ? " ph" : "") + "\"><th class=\"mx-gh\" data-scope=\"" + esc(T.scopeKey) + "\" colspan=\"" + (cols.length + 1) + "\"><span>" +
          "<b>" + esc(T.phase.title) + "</b> x" + DM.fmtInt(T.phase.times) +
          (T.scope.length > 1 ? " <i>/ " + esc(T.scope.slice(1).join(" / ")) + "</i>" : "") + "</span></th></tr>");
      }
      var rw = DM.rwShort(T), tot = "";
      if (T.kind === "upload" || T.kind === "download") tot = "<b class=\"" + (T.kind === "upload" ? "wr" : "rd") + "\">" + DM.fmtBytesShort(T.bytes) + "</b>";
      else if (rw && !DM.uncounted(T.rBytes, T.rUnk, T.wBytes, T.wUnk)) {
        tot = "<b class=\"rd\">" + (T.rN ? (T.rUnk === T.rN ? DM.NM : DM.fmtBytesShort(T.rBytes)) : "-") + "</b> <b class=\"wr\">" +
          (T.wN ? (T.wUnk === T.wN ? DM.NM : DM.fmtBytesShort(T.wBytes)) : "-") + "</b>";
      }
      out.push("<tr data-r=\"" + r + "\"><th class=\"mx-step k-" + (T.kind === "upload" ? "up" : T.kind === "download" ? "down" : "launch") +
        (T.flags.length ? " lv-" + DM.worstLevel(T.flags) : "") + "\" data-tip=\"" + esc(DM.stepTip(T)) + "\"><div class=\"mx-sl\"><span class=\"sid\">" + esc(T.id) +
        "</span><span class=\"mx-sn\">" + esc(T.label) + (T.launches > 1 ? " x" + T.launches : "") + "</span><span class=\"mx-sb\">" + tot + "</span></div></th>");
      var byCol = Object.create(null);
      T.acc.forEach(function (A) {
        var c = cur.colOf[A.S.key];
        if (c !== undefined) byCol[c] = A;
      });
      for (var c = 0; c < cols.length; c++) {
        var A = byCol[c];
        if (!A) {
          out.push(startOf[c] ? "<td class=\"bs\"></td>" : "<td></td>");
          continue;
        }
        var both = !!A.r && !!A.w;
        out.push("<td class=\"x" + (startOf[c] ? " bs" : "") + (A.src !== "measured" && A.src !== "host" ? " st" : "") + "\" data-tip=\"" + esc(cellTip(T, A)) + "\">" +
          (A.r ? markHtml(A.r, "rd", both, lo, hi) : "") + (A.w ? markHtml(A.w, "wr", both, lo, hi) : "") + "</td>");
      }
      out.push("</tr>");
    });
    out.push("</tbody></table>");

    var wrap = h("div", { cls: "mx-wrap" });
    wrap.innerHTML = out.join("");
    ctx.body.appendChild(wrap);

    var table = wrap.firstChild;
    cur.colEls = Array.prototype.slice.call(table.querySelectorAll("colgroup col"), 1);
    cur.secThs = Array.prototype.slice.call(table.querySelectorAll("tr.mx-secs th"));
    cur.rowEls = Array.prototype.slice.call(table.querySelectorAll("tbody tr[data-r]"));

    DM.on(wrap, "click", function (e) {
      var t = e.target;
      if (!t || !t.closest) return;
      var td = t.closest("td"), th = t.closest("th"), r, c;
      if (td) {
        r = Number(td.parentNode.getAttribute("data-r"));
        c = td.cellIndex - 1;
        var T = cur.rows[r], S = cur.cols[c];
        if (!T || !S) return;
        if (td.classList.contains("x")) DM.select("access:" + T.id + ":" + S.key, { reveal: false });
        else DM.select(DM.sel.step(T), { reveal: false });
      } else if (th) {
        if (th.classList.contains("mx-step")) {
          r = Number(th.parentNode.getAttribute("data-r"));
          if (cur.rows[r]) DM.select(DM.sel.step(cur.rows[r]), { reveal: false });
        } else if (th.classList.contains("mx-sec")) {
          c = Number(th.getAttribute("data-c"));
          if (cur.cols[c]) DM.select(DM.sel.section(cur.cols[c]), { reveal: false });
        } else if (th.classList.contains("mx-block")) {
          var g = cur.groups[Number(th.getAttribute("data-b"))];
          if (g) DM.select(DM.sel.block(g.B), { reveal: false });
        } else if (th.classList.contains("mx-gh")) {
          DM.select(DM.sel.scope(th.getAttribute("data-scope")), { reveal: false });
        }
      }
    });

    function hoverCol(c) {
      if (c === cur.hoverC) return;
      if (cur.hoverC >= 0 && cur.colEls[cur.hoverC]) {
        cur.colEls[cur.hoverC].classList.remove("hl");
        cur.secThs[cur.hoverC].classList.remove("hl");
      }
      cur.hoverC = c;
      if (c >= 0 && cur.colEls[c]) {
        cur.colEls[c].classList.add("hl");
        cur.secThs[c].classList.add("hl");
      }
    }
    DM.on(wrap, "mouseover", function (e) {
      var t = e.target, cell = t && t.closest ? t.closest("td, th.mx-sec") : null, c = -1;
      if (cell) c = cell.tagName === "TD" ? cell.cellIndex - 1 : Number(cell.getAttribute("data-c"));
      hoverCol(c);
    });
    DM.on(wrap, "mouseleave", function () { hoverCol(-1); });
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
    if (!cur || !cur.rowEls.length) return;
    cur.rowEls.forEach(function (tr) { tr.classList.remove("sel"); });
    cur.colEls.forEach(function (col, c) {
      col.classList.remove("sel");
      cur.secThs[c].classList.remove("sel");
    });
    if (!res) return;
    var rowsHit = [], c = -1;
    if (res.kind === "step" || res.kind === "access") {
      if (cur.rowOf[res.T.id] !== undefined) rowsHit.push(cur.rowOf[res.T.id]);
    } else if (res.kind === "kernel") {
      res.K.steps.forEach(function (T) { if (cur.rowOf[T.id] !== undefined) rowsHit.push(cur.rowOf[T.id]); });
    } else if (res.kind === "scope") {
      cur.rows.forEach(function (T, r) {
        if (T.scopeKey === res.path || T.scopeKey.indexOf(res.path + "/") === 0) rowsHit.push(r);
      });
    }
    if (res.kind === "section" || res.kind === "access") {
      if (cur.colOf[res.S.key] !== undefined) c = cur.colOf[res.S.key];
    }
    rowsHit.forEach(function (r) { if (cur.rowEls[r]) cur.rowEls[r].classList.add("sel"); });
    if (c >= 0 && cur.colEls[c]) {
      cur.colEls[c].classList.add("sel");
      cur.secThs[c].classList.add("sel");
    }
    if (!reveal) return;
    if (rowsHit.length && cur.rowEls[rowsHit[0]]) bring(cur.rowEls[rowsHit[0]], { block: "center", inline: "nearest" });
    if (c >= 0 && cur.secThs[c]) bring(cur.secThs[c], { block: "nearest", inline: "center" });
  }

  DM.registerView({ id: "matrix", title: "Access matrix", phased: true, render: render, select: select });
})();
