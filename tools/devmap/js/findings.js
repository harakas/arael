// devmap view 6: Findings. The case's own findings, then the viewer's
// computed checks grouped by kind. Every reference is a link that selects
// the thing in the view that shows it best.
"use strict";
(function () {
  var DM = window.DM, h = DM.h;

  function refLinks(M, refs) {
    if (!refs.length) return null;
    var box = h("div", { cls: "fx-refs" });
    refs.forEach(function (sel) {
      var kind = sel.slice(0, sel.indexOf(":"));
      box.appendChild(h("span", { cls: "fx-ref" }, h("i", null, kind), DM.link.sel(M, sel, { view: DM.viewFor[kind] })));
    });
    return box;
  }

  function item(M, f, withTitle) {
    return h("div", { cls: "fx-item lv-" + f.level }, DM.levelBadge(f.level),
      h("div", { cls: "fx-main" },
        withTitle && f.title ? h("div", { cls: "fx-title" }, f.title) : null,
        f.text ? h("div", { cls: "fx-text" }, f.text) : null,
        refLinks(M, f.refs)));
  }

  function counts(list) {
    var n = { bug: 0, warn: 0, info: 0 };
    list.forEach(function (f) { n[f.level]++; });
    var parts = [];
    ["bug", "warn", "info"].forEach(function (lv) {
      if (n[lv]) parts.push(h("span", { cls: "fx-count" }, DM.levelBadge(lv), " " + n[lv]));
    });
    return parts;
  }

  function render(ctx) {
    var M = ctx.M;
    var wrap = h("div", { cls: "fx" });
    ctx.body.appendChild(wrap);

    wrap.appendChild(h("h3", null, "Findings", h("span", { cls: "h-extra" }, "recorded in the case (" + M.findings.length + ")"), counts(M.findings)));
    if (!M.findings.length) wrap.appendChild(h("div", { cls: "fx-none" }, "The case records no findings."));
    M.findings.forEach(function (f) { wrap.appendChild(item(M, f, true)); });

    wrap.appendChild(h("h3", null, "Computed checks", h("span", { cls: "h-extra" }, "by the viewer, from the case's data (" + M.checks.length + ")"), counts(M.checks)));
    if (!M.checks.length) wrap.appendChild(h("div", { cls: "fx-none" }, "No check fired."));
    // one collapsible group per kind of check, worst level first
    var groups = [], by = Object.create(null);
    M.checks.forEach(function (f) {
      var k = f.code + "|" + f.level, g = by[k];
      if (!g) {
        g = by[k] = { code: f.code, level: f.level, title: f.title, items: [] };
        groups.push(g);
      }
      g.items.push(f);
    });
    groups.forEach(function (g) {
      var box = h("details", { cls: "fx-group lv-" + g.level, open: g.level !== "info" || g.items.length <= 12 },
        h("summary", null, DM.levelBadge(g.level), h("span", { cls: "fx-gtitle" }, g.title), h("span", { cls: "fx-gn" }, String(g.items.length))));
      g.items.forEach(function (f) { box.appendChild(item(M, f, false)); });
      wrap.appendChild(box);
    });
  }

  function badge(M) {
    return M.findings.length + M.checks.filter(function (f) { return f.level !== "info"; }).length;
  }

  DM.registerView({ id: "findings", title: "Findings", render: render, badge: badge });
})();
