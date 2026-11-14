// devmap view 5: Kernels. One sortable table of every kernel of the case:
// its unit and the unit's resolved types, where it is launched, how much
// it reads and writes over the solve, and the access patterns seen.
"use strict";
(function () {
  var DM = window.DM, h = DM.h;

  var ui = { sort: "order", dir: 1 };
  var cur = null;       // { rowEls{kernel name} }

  function typesText(K) {
    if (!K.unit) return "";
    return Object.keys(K.unit.types).map(function (k) { return k + "=" + K.unit.types[k]; }).join(" ");
  }

  function patCell(list, cls) {
    return list.filter(function (p) { return p !== "unknown"; }).map(function (p) {
      return h("b", { cls: "pl " + cls, tip: p + ": " + DM.patDoc(p) }, DM.patLetter(p));
    });
  }

  // id, title, tip, num (sort descending first), key(K) for sorting, cell(K, M)
  function columns(M) {
    var cols = [
      { id: "order", title: "#", tip: "order of first launch", num: false,
        key: function (K) { return K.order < 0 ? 1e15 : K.order; },
        cell: function (K) { return h("td", { cls: "numc dim" }, K.order < 0 ? "-" : String(K.order + 1)); } },
      { id: "name", title: "kernel", num: false,
        key: function (K) { return K.name.toLowerCase(); },
        cell: function (K) {
          var lv = DM.worstLevel(K.flags);
          return h("td", { cls: "mono kname" }, K.name, lv ? [" ", h("span", { cls: "fl lv-" + lv, tip: K.flags.map(function (f) { return f.title + ": " + f.text; }).join("\n") }, K.flags.length + (K.flags.length > 1 ? " checks" : " check"))] : null,
            K.defined ? null : h("span", { cls: "dim", tip: "the case has no entry for it in kernels: no doc, parameters or source" }, "  (no entry)"));
        } },
      { id: "unit", title: "unit", num: false,
        key: function (K) { return K.unit ? K.unit.name.toLowerCase() : "~"; },
        cell: function (K) { return h("td", null, K.unit ? DM.link.unit(K.unit) : h("span", { cls: "dim" }, "-")); } },
      { id: "types", title: "unit types", tip: "the unit's typedefs, resolved", num: false,
        key: typesText,
        cell: function (K) { return h("td", { cls: "types" }, K.unit ? DM.unitTypes(K.unit) : null); } },
      { id: "phases", title: "launched in", tip: "phases, with launches per occurrence of the phase", num: true,
        key: function (K) { return K.tot.occ; },
        cell: function (K) {
          if (!K.phases.length) return h("td", { cls: "dim" }, "not launched");
          return h("td", null, K.phases.map(function (e, i) {
            return [i ? ", " : null, DM.link.scope(M, e.P.id, null, e.P.title), h("span", { cls: "dim" }, " " + DM.fmtInt(e.launches))];
          }));
        } },
      { id: "launches", title: "launches", tip: "launches in the whole solve (per occurrence x occurrences)", num: true,
        key: function (K) { return K.tot.launches; },
        cell: function (K) { return h("td", { cls: "numc" }, K.steps.length ? DM.fmtInt(K.tot.launches) : "-"); } },
      { id: "r", title: "read", tip: "bytes read in the whole solve", num: true,
        key: function (K) { return K.tot.r; },
        cell: function (K) { return h("td", { cls: "numc rd" }, K.steps.length ? DM.amountEl(K.tot.r, K.tot.rUnk) : "-"); } },
      { id: "w", title: "written", tip: "bytes written in the whole solve", num: true,
        key: function (K) { return K.tot.w; },
        cell: function (K) { return h("td", { cls: "numc wr" }, K.steps.length ? DM.amountEl(K.tot.w, K.tot.wUnk) : "-"); } },
      { id: "pat", title: "patterns", tip: "access patterns seen: reads, then writes", num: false,
        key: function (K) { return K.patR.concat(["/"], K.patW).join(" "); },
        cell: function (K) { return h("td", { cls: "pats" }, patCell(K.patR, "rd"), K.patR.length && K.patW.length ? h("span", { cls: "dim" }, " / ") : null, patCell(K.patW, "wr")); } },
      { id: "params", title: "params", tip: "pointer + scalar parameters", num: true,
        key: function (K) { return K.params.length; },
        cell: function (K) {
          var ptr = K.params.filter(function (Q) { return Q.ptr; }).length;
          return h("td", { cls: "numc" }, K.defined ? ptr + " + " + (K.params.length - ptr) : "-");
        } }
    ];
    if (M.hasTime) {
      cols.push({ id: "time", title: "time", tip: "profiled time in the whole solve", num: true,
        key: function (K) { return K.tot.time; },
        cell: function (K) { return h("td", { cls: "numc" }, K.tot.hasTime ? DM.fmtTime(K.tot.time) : "-"); } });
    }
    cols.push({ id: "doc", title: "doc", num: false,
      key: function (K) { return K.doc.toLowerCase(); },
      cell: function (K) { return h("td", { cls: "kdoc", tip: K.doc }, K.doc); } });
    return cols;
  }

  function render(ctx) {
    var M = ctx.M, cols = columns(M);
    cur = { rowEls: Object.create(null) };
    var sortCol = cols[0];
    cols.forEach(function (c) { if (c.id === ui.sort) sortCol = c; });
    var list = M.kernelList.slice();
    list.forEach(function (K, i) { K.sortIdx = i; });
    list.sort(function (a, b) {
      var x = sortCol.key(a), y = sortCol.key(b);
      var d = x < y ? -1 : x > y ? 1 : 0;
      return d * ui.dir || a.sortIdx - b.sortIdx;
    });

    var launched = list.filter(function (K) { return K.steps.length > 0; }).length;
    ctx.tools.appendChild(h("span", { cls: "tool-title" }, list.length + " kernels, " + launched + " launched"));
    ctx.tools.appendChild(h("span", { cls: "legend" }, h("span", { cls: "lg dim" }, "click a column to sort; totals are for the whole solve"),
      DM.patLegend(DM.PAT_ORDER.slice(0, 7))));

    var head = h("tr", null);
    cols.forEach(function (c) {
      head.appendChild(h("th", { cls: "sortable" + (c.id === ui.sort ? " sorted" : ""), tip: c.tip, onclick: function () {
        if (ui.sort === c.id) ui.dir = -ui.dir;
        else {
          ui.sort = c.id;
          ui.dir = c.num ? -1 : 1;
        }
        DM.renderView(false);
      } }, c.title, c.id === ui.sort ? h("span", { cls: "arrow" }, ui.dir > 0 ? " " + DM.CH.up : " " + DM.CH.down) : null));
    });
    var table = h("table", { cls: "grid ktable" }, h("thead", null, head));
    var body = h("tbody", null);
    list.forEach(function (K) {
      var tr = h("tr", { cls: K.steps.length ? null : "unused", onclick: function () { DM.select(DM.sel.kernel(K), { reveal: false }); } });
      cols.forEach(function (c) { tr.appendChild(c.cell(K, M)); });
      cur.rowEls[K.name] = tr;
      body.appendChild(tr);
    });
    table.appendChild(body);
    ctx.body.appendChild(h("div", { cls: "kwrap" }, table));
    if (!list.length) ctx.body.appendChild(h("div", { cls: "empty" }, "This case lists no kernels."));
  }

  function select(res, reveal) {
    if (!cur) return;
    Object.keys(cur.rowEls).forEach(function (k) { cur.rowEls[k].classList.remove("sel"); });
    if (!res) return;
    var names = [];
    if (res.kind === "kernel") names.push(res.K.name);
    else if ((res.kind === "step" || res.kind === "access") && res.T.kernel) names.push(res.T.kernel.name);
    else if (res.kind === "unit") res.U.kernels.forEach(function (K) { names.push(K.name); });
    names.forEach(function (n, i) {
      var tr = cur.rowEls[n];
      if (!tr) return;
      tr.classList.add("sel");
      if (reveal && i === 0 && tr.scrollIntoView) tr.scrollIntoView({ block: "nearest" });
    });
  }

  DM.registerView({ id: "kernels", title: "Kernels", render: render, select: select });
})();
