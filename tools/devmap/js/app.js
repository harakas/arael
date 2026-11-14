// devmap: the shell. State and its URL hash, case loading, tabs, search,
// the selection with its history, and the details panel's frame.
//
// A view is { id, title, phased, render(ctx), select(res, reveal), badge(M) }:
//   render(ctx)   builds ctx.tools (toolbar) and ctx.body from ctx.M
//   select(res)   marks the selection (res from M.resolve, or null) in what
//                 render drew; reveal asks to scroll it into view
//   phased        the view shows the phase in state.phase
"use strict";
(function () {
  var DM = window.DM = window.DM || {};
  var h = DM.h;

  DM.state = { caseId: null, view: "overview", phase: "all", sel: null };
  DM.M = null;                 // the model of the current case
  DM.loadLog = [];             // { file, ok, note } per data file

  var caseList = [];           // { key, c }
  var models = Object.create(null);
  var hist = [];               // earlier selections, for the back button
  var els = {};

  // ------------------------------------------------------------------
  // URL hash: #case=..&view=..&phase=..&sel=..
  // ------------------------------------------------------------------

  function parseHash() {
    var out = {}, s = String(window.location.hash || "");
    if (s.charAt(0) === "#") s = s.slice(1);
    s.split("&").forEach(function (kv) {
      var i = kv.indexOf("=");
      if (i <= 0) return;
      try {
        out[kv.slice(0, i)] = decodeURIComponent(kv.slice(i + 1));
      } catch (e) {
        // a malformed escape: the key is dropped
      }
    });
    return out;
  }

  function writeHash() {
    var st = DM.state, parts = [];
    if (st.caseId !== null) parts.push("case=" + encodeURIComponent(st.caseId));
    parts.push("view=" + encodeURIComponent(st.view));
    parts.push("phase=" + encodeURIComponent(st.phase));
    if (st.sel) parts.push("sel=" + encodeURIComponent(st.sel));
    var hash = "#" + parts.join("&");
    if (hash === window.location.hash) return;
    try {
      window.history.replaceState(null, "", hash);
    } catch (e) {
      window.location.hash = hash;
    }
  }

  function onHashChange() {
    var q = parseHash(), st = DM.state;
    var same = (!("case" in q) || q["case"] === st.caseId) && (!q.view || q.view === st.view) &&
               (!q.phase || q.phase === st.phase) && ((q.sel || null) === st.sel);
    if (same || !caseList.length) return;
    applyHash(q);
    writeHash();        // back to the canonical form (an unknown case or view was replaced)
  }

  // ------------------------------------------------------------------
  // Cases
  // ------------------------------------------------------------------

  function hasCase(key) {
    return caseList.some(function (x) { return x.key === key; });
  }

  function getModel(key) {
    if (models[key]) return models[key];
    var entry = null;
    caseList.forEach(function (x) { if (x.key === key) entry = x; });
    var M;
    try {
      M = DM.buildModel(entry ? entry.c : {});
    } catch (e) {
      DM.showError(e, "indexing case " + key);
      M = DM.buildModel({ id: key, title: key + " (could not be indexed: see the error above)" });
    }
    models[key] = M;
    return M;
  }

  function applyHash(q) {
    var st = DM.state;
    var key = "case" in q && hasCase(q["case"]) ? q["case"] : (st.caseId !== null && hasCase(st.caseId) ? st.caseId : caseList[0].key);
    if (key !== st.caseId) {
      st.caseId = key;
      hist = [];
    }
    DM.M = getModel(key);
    if (q.view && DM.views[q.view]) st.view = q.view;
    if (!DM.views[st.view]) st.view = DM.viewOrder[0];
    st.phase = q.phase && DM.M.phaseById[q.phase] ? q.phase : "all";
    st.sel = q.sel && DM.M.resolve(q.sel) ? q.sel : null;
    renderAll();
  }

  DM.setCase = function (key) {
    var st = DM.state;
    if (!hasCase(key) || key === st.caseId) return;
    st.caseId = key;
    st.sel = null;
    st.phase = "all";
    hist = [];
    DM.M = getModel(key);
    renderAll();
    writeHash();
  };

  // ------------------------------------------------------------------
  // Rendering the shell
  // ------------------------------------------------------------------

  function renderTabs() {
    DM.clear(els.tabs);
    DM.viewOrder.forEach(function (id) {
      var v = DM.views[id];
      var b = h("button", { type: "button", cls: "tab" + (id === DM.state.view ? " on" : ""), onclick: function () { DM.setView(id); } }, v.title);
      if (v.badge && DM.M) {
        var n = 0;
        try {
          n = v.badge(DM.M);
        } catch (e) {
          DM.showError(e, "tab " + id);
        }
        if (n) b.appendChild(h("span", { cls: "tab-n" }, String(n)));
      }
      els.tabs.appendChild(b);
    });
  }

  function chip(label, value, tip) {
    return h("span", { cls: "chip", tip: tip }, h("i", null, label), value);
  }

  function renderMeta() {
    var M = DM.M, bar = els.meta;
    DM.clear(bar);
    if (!M) return;
    var m = M.meta;
    bar.appendChild(h("span", { cls: "meta-title", tip: M.title }, M.title));
    if (m.problem !== undefined) bar.appendChild(chip("problem", String(m.problem)));
    if (m.model !== undefined) bar.appendChild(chip("model", DM.tyEl(String(m.model)), "the model's scalar"));
    if (m.wide !== undefined) bar.appendChild(chip("wide", DM.tyEl(String(m.wide)), "the route's wide scalar"));
    if (m.runtime !== undefined || m.device !== undefined) {
      bar.appendChild(chip("device", String(m.runtime !== undefined ? m.runtime : "?") + " / " + String(m.device !== undefined ? m.device : "?"), "runtime / device"));
    }
    bar.appendChild(chip("access", M.hasMeasured ? "measured" : M.hasAccess ? "not measured" : "none",
      "measured: every read and write counted by the CPU device's probe\nnot measured: read or written only, from the kernels' signatures" +
      (M.nDerived ? " (for " + M.nDerived + " launches derived by the viewer from their arguments)" : "")));
    if (m.commit !== undefined) bar.appendChild(chip("commit", String(m.commit)));
    if (m.generated !== undefined) bar.appendChild(chip("generated", String(m.generated)));
    if (M.schema !== undefined && M.schema !== 1) bar.appendChild(chip("schema", String(M.schema), "the viewer was written for schema 1"));
    var notes = Array.isArray(m.notes) ? m.notes : [];
    if (notes.length) bar.appendChild(h("span", { cls: "meta-notes", tip: notes.join("\n") }, notes.join("; ")));
    renderNotice(M);
  }

  // A case without measured accesses says so once, here, and points at the
  // probed case that has the volumes; the views then leave the amounts out.
  function renderNotice(M) {
    var box = els.notice;
    DM.clear(box);
    box.hidden = M.hasMeasured || !M.hasAccess;
    if (box.hidden) return;
    var from = typeof M.meta.patterns_from === "string" ? M.meta.patterns_from : "";
    box.appendChild(h("b", null, "Accesses not measured."));
    box.appendChild(h("span", null, " A GPU does not count its kernels' reads and writes: this case shows the launch sequence and the memory at full size, " +
      "and which sections a kernel reads or writes, from its signature. Amounts read "));
    box.appendChild(DM.nmEl());
    box.appendChild(h("span", null, "."));
    if (M.hasBorrowed && from) {
      box.appendChild(h("span", null, " Patterns in italics are borrowed from the probed case "));
      box.appendChild(caseLink(from));
      box.appendChild(h("span", null, ", which has the volumes."));
    } else {
      box.appendChild(h("span", null, " Volumes and patterns are in the probed (-cpu) cases."));
    }
  }

  function caseLink(key) {
    if (!hasCase(key)) return h("span", { cls: "mono" }, key);
    return h("button", { type: "button", cls: "case-link", tip: "open the case " + key, onclick: function () { DM.setCase(key); } }, key);
  }

  function emptyState() {
    var box = h("div", { cls: "empty" }, h("p", null, "No case is loaded."),
      h("p", null, "data/cases.js lists the case files (window.DEVMAP_FILES = [\"name.js\", ...]); each file calls DEVMAP.add({...})."));
    if (DM.loadLog.length) {
      var ul = h("ul", null);
      DM.loadLog.forEach(function (x) {
        ul.appendChild(h("li", null, x.file + ": " + (x.ok ? "loaded" : (x.note || "not loaded"))));
      });
      box.appendChild(ul);
    }
    return box;
  }

  function applySel(reveal) {
    var st = DM.state, v = DM.views[st.view];
    if (!v || !v.select || !DM.M) return;
    try {
      v.select(st.sel ? DM.M.resolve(st.sel) : null, !!reveal);
    } catch (e) {
      DM.showError(e, "select in " + st.view);
    }
  }

  // Draw the current view from scratch.
  DM.renderView = function (reveal) {
    var st = DM.state, v = DM.views[st.view], M = DM.M;
    DM.hideTip();
    DM.clear(els.tools);
    DM.clear(els.body);
    els.body.className = "view-" + st.view;
    els.body.scrollTop = 0;
    els.body.scrollLeft = 0;
    if (!M || !v) {
      els.body.appendChild(emptyState());
      return;
    }
    try {
      v.render({ M: M, tools: els.tools, body: els.body, state: st });
    } catch (e) {
      DM.showError(e, "render " + st.view);
      els.body.appendChild(h("div", { cls: "empty" }, "This view failed to draw; the error is in the strip at the top of the page."));
    }
    applySel(reveal);
  };

  DM.renderDetails = function () {
    var st = DM.state, M = DM.M;
    DM.clear(els.detBody);
    els.detBody.scrollTop = 0;
    els.detBack.disabled = hist.length === 0;
    els.detClear.disabled = !st.sel;
    els.detTitle.textContent = "";
    if (!M) return;
    try {
      var res = st.sel ? M.resolve(st.sel) : null;
      els.detTitle.textContent = res ? res.kind : "case";
      DM.details.render(els.detBody, M, res);
    } catch (e) {
      DM.showError(e, "details");
    }
  };

  function renderAll() {
    var st = DM.state;
    els.caseSelect.value = st.caseId;
    renderTabs();
    renderMeta();
    DM.renderDetails();
    DM.renderView(true);
  }

  // ------------------------------------------------------------------
  // Navigation
  // ------------------------------------------------------------------

  DM.setView = function (id) {
    if (!DM.views[id]) return;
    DM.state.view = id;
    renderTabs();
    DM.renderView(true);
    writeHash();
  };

  DM.setPhase = function (id) {
    var st = DM.state;
    st.phase = id !== "all" && DM.M && DM.M.phaseById[id] ? id : "all";
    DM.renderView(false);
    writeHash();
  };

  // Select something: sel is a selection string ("step:s9") or null.
  // opts.view   switch to this view
  // opts.phase  show this phase ("all" or an id)
  // opts.reveal false when the click was on the thing itself (no scrolling)
  // opts.push   false to leave the back history alone
  DM.select = function (sel, opts) {
    opts = opts || {};
    var st = DM.state, M = DM.M;
    if (!M) return;
    sel = sel || null;
    var res = sel ? M.resolve(sel) : null;
    if (sel && !res) sel = null;
    if (sel !== st.sel && st.sel && opts.push !== false) {
      hist.push(st.sel);
      if (hist.length > 60) hist.shift();
    }
    st.sel = sel;
    var reveal = opts.reveal !== false, redraw = false;
    if (opts.view && DM.views[opts.view] && opts.view !== st.view) {
      st.view = opts.view;
      redraw = true;
    }
    var v = DM.views[st.view];
    if (opts.phase) {
      var ph = opts.phase !== "all" && M.phaseById[opts.phase] ? opts.phase : "all";
      if (ph !== st.phase) {
        st.phase = ph;
        redraw = true;
      }
    } else if (res && reveal && v && v.phased && st.phase !== "all") {
      // a step of another phase: the phase follows the selection
      var want = res.kind === "step" || res.kind === "access" ? res.T.phase.id : res.kind === "scope" ? res.path.split("/")[0] : null;
      if (want && want !== st.phase && M.phaseById[want]) {
        st.phase = want;
        redraw = true;
      }
    }
    DM.renderDetails();
    if (redraw) {
      renderTabs();
      DM.renderView(reveal);
    } else {
      applySel(reveal);
    }
    writeHash();
  };

  DM.back = function () {
    if (!hist.length) return;
    DM.select(hist.pop(), { push: false });
  };

  // A phase picker bound to state.phase.
  DM.phasePicker = function (M, label) {
    var sel = h("select", { cls: "phase-pick", onchange: function () { DM.setPhase(sel.value); } });
    sel.appendChild(h("option", { value: "all" }, "all phases"));
    M.phases.forEach(function (P) {
      sel.appendChild(h("option", { value: P.id }, P.title + (P.title !== P.id ? " (" + P.id + ")" : "") + "  x" + DM.fmtInt(P.times)));
    });
    sel.value = DM.state.phase;
    return h("label", { cls: "tool" }, label || "phase", sel);
  };

  // A labelled checkbox for a toolbar.
  DM.checkbox = function (label, checked, onchange, tip) {
    var box = h("input", { type: "checkbox", onchange: function () { onchange(box.checked); } });
    box.checked = !!checked;
    return h("label", { cls: "tool check", tip: tip }, box, label);
  };

  // A labelled select for a toolbar: options are [value, text] pairs.
  DM.picker = function (label, options, value, onchange) {
    var sel = h("select", { onchange: function () { onchange(sel.value); } });
    options.forEach(function (o) { sel.appendChild(h("option", { value: o[0] }, o[1])); });
    sel.value = value;
    return h("label", { cls: "tool" }, label, sel);
  };

  // ------------------------------------------------------------------
  // Search
  // ------------------------------------------------------------------

  function initSearch() {
    var input = els.search, box = els.results, items = [], rows = [], active = 0;

    function close() {
      box.hidden = true;
      DM.clear(box);
      items = [];
      rows = [];
    }
    function pick(i) {
      var it = items[i];
      if (!it) return;
      close();
      input.blur();
      DM.select(it.sel);
    }
    function setActive(i) {
      if (!rows.length) return;
      if (rows[active]) rows[active].classList.remove("on");
      active = (i + rows.length) % rows.length;
      rows[active].classList.add("on");
      if (rows[active].scrollIntoView) rows[active].scrollIntoView({ block: "nearest" });
    }
    function update() {
      var M = DM.M, q = input.value.trim().toLowerCase();
      DM.clear(box);
      items = [];
      rows = [];
      active = 0;
      if (!M || !q) {
        box.hidden = true;
        return;
      }
      var toks = q.split(" ").filter(function (t) { return t !== ""; });
      var scored = [];
      M.search.forEach(function (it) {
        var score = it.rank;
        for (var i = 0; i < toks.length; i++) {
          var p = it.text.indexOf(toks[i]);
          if (p < 0) return;
          if (p > 0) {
            var before = it.text.charAt(p - 1);
            score += before === "." || before === "_" || before === " " || before === "/" ? 1 : 2;
          }
        }
        scored.push({ it: it, score: score });
      });
      scored.sort(function (a, b) { return a.score - b.score || a.it.text.length - b.it.text.length; });
      items = scored.slice(0, 40).map(function (x) { return x.it; });
      if (!items.length) box.appendChild(h("div", { cls: "sr-none" }, "no match"));
      items.forEach(function (it, i) {
        var row = h("div", { cls: "sr" + (i === 0 ? " on" : ""), onmousedown: function (e) { e.preventDefault(); pick(i); } },
          h("span", { cls: "sr-kind" }, it.kind), h("span", { cls: "sr-label" }, it.label), h("span", { cls: "sr-sub" }, it.sub));
        rows.push(row);
        box.appendChild(row);
      });
      if (scored.length > items.length) box.appendChild(h("div", { cls: "sr-none" }, (scored.length - items.length) + " more: type more to narrow"));
      box.hidden = false;
    }

    DM.on(input, "input", update);
    DM.on(input, "focus", update);
    DM.on(input, "blur", close);
    DM.on(input, "keydown", function (e) {
      if (e.key === "ArrowDown") { e.preventDefault(); setActive(active + 1); }
      else if (e.key === "ArrowUp") { e.preventDefault(); setActive(active - 1); }
      else if (e.key === "Enter") { e.preventDefault(); pick(active); }
      else if (e.key === "Escape") { close(); input.blur(); }
    });
    DM.on(document, "keydown", function (e) {
      if (e.key !== "/" || e.ctrlKey || e.metaKey || e.altKey) return;
      var t = e.target, tag = t && t.tagName ? t.tagName : "";
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      e.preventDefault();
      input.focus();
      input.select();
    });
  }

  // ------------------------------------------------------------------
  // The details panel's width
  // ------------------------------------------------------------------

  function initResize() {
    var det = els.details, drag = null;
    try {
      var saved = Number(window.localStorage.getItem("devmap.detw"));
      if (saved >= 280 && saved <= 1400) det.style.width = saved + "px";
    } catch (e) {
      // storage is not available from this origin: the default width stays
    }
    DM.on(els.resize, "mousedown", function (e) {
      drag = { x: e.clientX, w: det.getBoundingClientRect().width };
      document.body.classList.add("resizing");
      e.preventDefault();
    });
    DM.on(window, "mousemove", function (e) {
      if (!drag) return;
      var w = Math.max(280, Math.min(1400, drag.w + (drag.x - e.clientX)));
      det.style.width = w + "px";
    });
    DM.on(window, "mouseup", function () {
      if (!drag) return;
      drag = null;
      document.body.classList.remove("resizing");
      try {
        window.localStorage.setItem("devmap.detw", String(Math.round(det.getBoundingClientRect().width)));
      } catch (e) {
        // not saved
      }
    });
  }

  // ------------------------------------------------------------------
  // Boot: load the case files, then draw.
  // ------------------------------------------------------------------

  function loadCases(done) {
    var files = window.DEVMAP_FILES;
    if (files === undefined) {
      // no generated cases yet (gen/run.sh writes them and data/cases.js): the hand-made sample
      DM.loadLog.push({ file: "data/cases.js", ok: false, note: "not there: no generated cases, showing data/sample.js (gen/run.sh writes them)" });
      files = ["sample.js"];
    } else if (!Array.isArray(files)) {
      DM.loadLog.push({ file: "data/cases.js", ok: false, note: "did not set window.DEVMAP_FILES to an array" });
      window.devmapReport("data/cases.js did not set window.DEVMAP_FILES to an array of case file names");
      files = [];
    }
    var left = files.length, seen = window.DEVMAP.cases.length;
    if (!left) {
      done();
      return;
    }
    files.forEach(function (f) {
      var entry = { file: "data/" + f, ok: false, note: "" };
      DM.loadLog.push(entry);
      var tag = document.createElement("script");
      tag.async = false;        // run in list order
      tag.onload = DM.guard(function () {
        var n = window.DEVMAP.cases.length;
        if (n > seen) {
          entry.ok = true;
        } else {
          entry.note = "loaded, but DEVMAP.add was not reached (a syntax error in the file?)";
          window.devmapReport(entry.file + ": " + entry.note);
        }
        seen = n;
        left--;
        if (left === 0) done();
      }, "load " + f);
      tag.onerror = DM.guard(function () {
        entry.note = "failed to load";    // the page-level handler reports it in the strip
        left--;
        if (left === 0) done();
      }, "load " + f);
      tag.src = "data/" + f;
      document.head.appendChild(tag);
    });
  }

  function start() {
    caseList = [];
    var seenIds = Object.create(null);
    window.DEVMAP.cases.forEach(function (c, i) {
      if (!c || typeof c !== "object") return;
      var key = String(c.id !== undefined && c.id !== null ? c.id : "case" + i);
      if (seenIds[key]) key = key + "~" + i;
      seenIds[key] = true;
      caseList.push({ key: key, c: c });
    });
    DM.clear(els.caseSelect);
    caseList.forEach(function (x) {
      els.caseSelect.appendChild(h("option", { value: x.key }, x.c.title ? String(x.c.title) : x.key));
    });
    if (!caseList.length) {
      renderTabs();
      DM.clear(els.body);
      els.body.appendChild(emptyState());
      return;
    }
    applyHash(parseHash());
    writeHash();
  }

  DM.boot = DM.guard(function () {
    els = {
      caseSelect: document.getElementById("case-select"),
      tabs: document.getElementById("tabs"),
      search: document.getElementById("search"),
      results: document.getElementById("search-results"),
      meta: document.getElementById("meta-bar"),
      notice: document.getElementById("notice"),
      tools: document.getElementById("view-tools"),
      body: document.getElementById("view-body"),
      details: document.getElementById("details"),
      resize: document.getElementById("det-resize"),
      detBack: document.getElementById("det-back"),
      detClear: document.getElementById("det-clear"),
      detTitle: document.getElementById("det-title"),
      detBody: document.getElementById("det-body")
    };
    DM.initTips();
    initSearch();
    initResize();
    DM.on(els.caseSelect, "change", function () { DM.setCase(els.caseSelect.value); });
    DM.on(els.detBack, "click", function () { DM.back(); });
    DM.on(els.detClear, "click", function () { DM.select(null); });
    DM.on(window, "hashchange", onHashChange);
    loadCases(DM.guard(start, "start"));
  }, "boot");
})();
