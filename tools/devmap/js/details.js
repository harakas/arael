// devmap: cross-reference links and the details panel's content, one
// function per kind of selection (step, access, kernel, section, block,
// unit, scope) plus the case summary shown when nothing is selected.
"use strict";
(function () {
  var DM = window.DM = window.DM || {};
  var h = DM.h;

  // ------------------------------------------------------------------
  // Selection strings and links
  // ------------------------------------------------------------------

  DM.sel = {
    step: function (T) { return "step:" + T.id; },
    kernel: function (K) { return "kernel:" + K.name; },
    section: function (S) { return "section:" + S.key; },
    block: function (B) { return "block:" + B.id; },
    unit: function (U) { return "unit:" + U.id; },
    scope: function (path) { return "scope:" + path; },
    access: function (A) { return "access:" + A.T.id + ":" + A.S.key; }
  };

  function clip(text, n) {
    text = String(text);
    return text.length > n ? text.slice(0, n - 2) + ".." : text;
  }

  // Tooltip text of a step.
  DM.stepTip = function (T) {
    var lines = [T.label + (T.launches > 1 ? "  x" + T.launches : "") + "   [" + T.id + "]", T.phase.title + ": " + T.scope.join(" / ")];
    if (T.kind === "upload" || T.kind === "download") {
      lines.push((T.kind === "upload" ? "host -> device, " : "device -> host (the host waits), ") + DM.fmtBytesFull(T.bytes) + (T.via ? ", " + T.via : ""));
      if (T.acc.length > 1) lines.push("covers " + T.acc.length + " sections");
    } else {
      var geo = DM.geomText(T), n = DM.threadCount(T), rw = DM.rwShort(T);
      if (geo) lines.push("grid x threads: " + geo + (n !== null ? " = " + DM.fmtInt(n) + " threads" : "") + (T.shared ? ", shared " + DM.fmtInt(T.shared) + " B" : ""));
      lines.push(rw ? "read " + rw.r + ", written " + rw.w + " per occurrence" : "no access data");
      if (T.interleaved) lines.push("interleaved with other kernels' launches");
    }
    if (T.time !== null) lines.push("time " + DM.fmtTime(T.time) + " per launch");
    if (T.flags.length) lines.push(T.flags.length + (T.flags.length > 1 ? " checks: " : " check: ") + T.flags[0].title);
    if (T.kernel && T.kernel.doc) lines.push("", clip(T.kernel.doc, 240));
    return lines.join("\n");
  };

  // Tooltip text of a section.
  DM.secTip = function (S) {
    var lines = [S.qname + "   " + (S.ty || "?") + (S.role ? " (role " + S.role + ")" : ""),
      DM.fmtBytesFull(S.bytes) + " at " + DM.fmtInt(S.at) + ",  " + DM.fmtInt(S.count) + " x " + S.elem + " bytes"];
    if (S.group) lines.push("group: " + S.group);
    if (S.tot.readers || S.tot.writers) {
      lines.push("solve: read " + DM.fmtAmount(S.tot.r, S.tot.rUnk) + " by " + S.tot.readers + ", written " + DM.fmtAmount(S.tot.w, S.tot.wUnk) + " by " + S.tot.writers);
    }
    if (S.flags.length) lines.push(S.flags.length + (S.flags.length > 1 ? " checks: " : " check: ") + S.flags.map(function (f) { return f.title; }).slice(0, 3).join("; "));
    if (S.doc) lines.push("", clip(S.doc, 240));
    if (S.about) lines.push(clip(S.about, 300));
    return lines.join("\n");
  };

  // A clickable reference; opts are DM.select's.
  function lnk(cls, sel, opts, tip, kids) {
    return h("span", { cls: "lnk " + cls, tip: tip, onclick: function (e) { e.stopPropagation(); DM.select(sel, opts); } }, kids);
  }

  DM.link = {
    step: function (T, opts) {
      return lnk("lnk-step", DM.sel.step(T), opts, DM.stepTip(T),
        [h("span", { cls: "sid" }, T.id), T.label + (T.launches > 1 ? " x" + T.launches : "")]);
    },
    kernel: function (K, opts) {
      return lnk("lnk-kernel", DM.sel.kernel(K), opts, K.doc ? clip(K.doc, 300) : "", K.name);
    },
    section: function (S, opts) {
      return lnk("lnk-sec" + (S.missing ? " dead" : ""), DM.sel.section(S), opts, DM.secTip(S),
        [h("i", { cls: "dot " + DM.tyClass(S.ty) }), S.qname]);
    },
    block: function (B, opts) {
      return lnk("lnk-block", DM.sel.block(B), opts, B.doc, B.name);
    },
    unit: function (U, opts) {
      return lnk("lnk-unit", DM.sel.unit(U), opts, U.file, U.name);
    },
    scope: function (M, path, opts, label) {
      var info = M.scopeInfo(path);
      return lnk("lnk-scope", DM.sel.scope(path), opts, path + (info.doc ? "\n" + info.doc : ""), label || info.title);
    },
    // A link from a selection string; text that names nothing is shown dead.
    sel: function (M, sel, opts) {
      var r = M.resolve(sel);
      if (!r) return h("span", { cls: "dead", tip: "not in this case" }, sel);
      if (r.kind === "step") return DM.link.step(r.T, opts);
      if (r.kind === "kernel") return DM.link.kernel(r.K, opts);
      if (r.kind === "section") return DM.link.section(r.S, opts);
      if (r.kind === "block") return DM.link.block(r.B, opts);
      if (r.kind === "unit") return DM.link.unit(r.U, opts);
      if (r.kind === "scope") return DM.link.scope(M, r.path, opts);
      return lnk("lnk-step", sel, opts, "", sel);
    }
  };

  // The typedefs of a unit as chips: V = f32, W = f64.
  DM.unitTypes = function (U) {
    var out = [];
    Object.keys(U.types).forEach(function (k) {
      out.push(DM.tyEl(String(U.types[k]), k));
    });
    return out;
  };

  // "R 4.46 MB" with the exact count on hover; letter is "R" or "W".
  DM.rwSpan = function (letter, n, unk, bytes) {
    var v = n === 0 ? h("span", { cls: "dim" }, "-") : unk === n ? DM.nmEl() : DM.bytesEl(bytes);
    return h("span", { cls: "rw " + (letter === "R" ? "rd" : "wr") }, h("b", null, letter), " ", v, unk && unk !== n ? DM.nmEl(" +" + DM.NM) : null);
  };

  // A level badge: bug, warn, info.
  DM.levelBadge = function (level) {
    return h("span", { cls: "badge lv-" + level }, level);
  };

  // ------------------------------------------------------------------
  // Builders
  // ------------------------------------------------------------------

  function th(text, tip) { return h("th", { tip: tip }, text); }

  function kv(rows) {
    var t = h("table", { cls: "kv" });
    rows.forEach(function (r) {
      if (!r || r[1] === null || r[1] === undefined || r[1] === "") return;
      t.appendChild(h("tr", null, h("th", null, r[0]), h("td", null, r[1])));
    });
    return t;
  }

  function sect(root, title, extra) {
    var d = h("div", { cls: "dsec" }, h("h4", null, title, extra ? h("span", { cls: "h-extra" }, extra) : null));
    root.appendChild(d);
    return d;
  }

  function head(root, kind, name, sub) {
    root.appendChild(h("div", { cls: "dhead" }, h("span", { cls: "badge kind kind-" + kind }, kind),
      h("span", { cls: "dname" }, name), sub ? h("span", { cls: "dsub" }, sub) : null));
  }

  function go(sel, view) {
    return function () { DM.select(sel, { view: view }); };
  }

  // A row of "show in" buttons: pairs of [label, handler].
  function showIn(root, pairs) {
    var row = h("div", { cls: "showin" }, "show in");
    pairs.forEach(function (p) {
      if (p) row.appendChild(h("button", { type: "button", cls: "mini", onclick: p[1] }, p[0]));
    });
    root.appendChild(row);
  }

  function doc(root, text) {
    if (text) root.appendChild(h("p", { cls: "doc" }, text));
  }

  function flagList(root, flags) {
    if (!flags.length) return;
    var s = sect(root, "Checks", String(flags.length));
    flags.forEach(function (f) {
      s.appendChild(h("div", { cls: "flag lv-" + f.level }, DM.levelBadge(f.level), h("span", { cls: "flag-title" }, f.title),
        h("div", { cls: "flag-text" }, f.text)));
    });
  }

  function fold(title, node, open) {
    var d = h("details", { cls: "fold" }, h("summary", null, title), node);
    if (open) d.setAttribute("open", "");
    return d;
  }

  function scopePath(M, scope) {
    var out = [];
    for (var i = 0; i < scope.length; i++) {
      if (i) out.push(h("span", { cls: "dim" }, " / "));
      out.push(DM.link.scope(M, scope.slice(0, i + 1).join("/")));
    }
    return out;
  }

  function accOf(T, S) {
    for (var i = 0; i < T.acc.length; i++) if (T.acc[i].S === S) return T.acc[i];
    return null;
  }

  function dirBrief(letter, d) {
    if (!d) return null;
    return h("span", { cls: "rw " + (letter === "R" ? "rd" : "wr") }, h("b", null, letter), " ", d.bytes !== null ? DM.bytesEl(d.bytes) : DM.nmEl(), " ");
  }

  var SRC_DOC = {
    measured: "measured: counted by the CPU device's probe",
    "static": "static: from the kernel's signature (const-ness and the column macros); no counts",
    args: "from arguments: derived by the viewer (const pointer = read, other pointer = write)",
    host: "host copy"
  };

  // One table row for one direction of an access. who goes in the second
  // column (the section in a step's table, the step in a section's table).
  function dirRow(A, d, dir, who, sub) {
    var S = A.S, cov = DM.coverage(S, d);
    var widthBad = S.elem > 0 && d.sizes.some(function (z) { return z !== S.elem; });
    var per = d.n !== null && d.unique ? d.n / d.unique : null;
    var extra = [];
    if (d.sizes.length) extra.push("width " + d.sizes.join(",") + (widthBad ? " (elem " + S.elem + ")" : ""));
    if (d.seq !== null) extra.push("seq " + DM.fmtNum(d.seq));
    if (d.same !== null) extra.push("same " + DM.fmtNum(d.same));
    return h("tr", { cls: "acc-row" },
      h("td", { cls: "dir " + (dir === "r" ? "rd" : "wr"), tip: dir === "r" ? "read" : "write" }, dir === "r" ? "R" : "W"),
      h("td", null, who, h("div", { cls: "sub", tip: SRC_DOC[A.src] || "" }, sub)),
      h("td", { cls: "numc" }, d.bytes !== null ? DM.bytesEl(d.bytes) : DM.nmEl(),
        h("div", { cls: "sub" }, d.n !== null ? DM.fmtInt(d.n) + " acc" + (per !== null && per >= 1.5 ? " (x" + per.toFixed(per >= 10 ? 0 : 1) + ")" : "") : "")),
      h("td", { cls: "numc" }, d.unique !== null ? DM.fmtInt(d.unique) + " el" : h("span", { cls: "dim" }, "-"),
        h("div", { cls: "sub" }, cov !== null ? DM.fmtPct(cov) + " of section" : "")),
      h("td", { cls: "numc" }, d.lo !== null && d.hi !== null ? DM.fmtInt(d.lo) + ".." + DM.fmtInt(d.hi) : h("span", { cls: "dim" }, "-"),
        h("div", { cls: "sub" }, d.threads !== null ? DM.fmtInt(d.threads) + " thr" : "")),
      h("td", null, DM.dirPatEl(d),
        h("div", { cls: "sub" + (widthBad ? " bad" : ""), tip: "width: access widths in bytes\nseq: fraction of a thread's accesses adjacent to its previous one\nsame: fraction equal to its previous one" }, extra.join(", "))));
  }

  function accHead(second) {
    return h("tr", null, th(""), th(second), th("bytes", "bytes moved / accesses (x = accesses per distinct element)"),
      th("distinct", "distinct elements / share of the section's elements"), th("range", "byte range touched within the section / device threads"),
      th("pattern", "access pattern / access widths, seq and same fractions"));
  }

  // ------------------------------------------------------------------
  // Step
  // ------------------------------------------------------------------

  function edgeRows(root, T) {
    if (!T.edgesIn.length && !T.edgesOut.length) return;
    var s = sect(root, "Dataflow", (T.edgesIn.length + T.edgesOut.length) + " edges");
    T.edgesIn.forEach(function (E) {
      s.appendChild(h("div", { cls: "edge-row" }, h("span", { cls: "tag rd" }, "in"), DM.link.section(E.S), h("span", { cls: "dim" }, " from "),
        E.from ? DM.link.step(E.from) : h("span", { cls: "hostref" }, "host")));
    });
    T.edgesOut.forEach(function (E) {
      s.appendChild(h("div", { cls: "edge-row" }, h("span", { cls: "tag wr" }, "out"), DM.link.section(E.S), h("span", { cls: "dim" }, " to "),
        E.to ? DM.link.step(E.to) : h("span", { cls: "hostref" }, "host")));
    });
  }

  function mismatchText(G) {
    var Q = G.def, S = G.S;
    if (G.mismatch === "type") {
      return "type mismatch: the parameter is " + Q.type + (Q.role ? " (role " + Q.role + ")" : "") + ", the section stores " + S.ty +
        (S.role ? " (role " + S.role + ")" : "");
    }
    return "roles differ: parameter " + Q.role + ", section " + S.role + " (both " + S.ty + " in this build)";
  }

  function stepDetails(root, M, T) {
    var K = T.kernel, P = T.phase, me = DM.sel.step(T);
    head(root, T.kind, K ? DM.link.kernel(K) : T.label, T.id);
    root.appendChild(h("div", { cls: "dpath" }, scopePath(M, T.scope), h("span", { cls: "dim" }, "   phase runs x" + DM.fmtInt(P.times))));
    showIn(root, [["pipeline", go(me, "pipeline")], ["matrix", go(me, "matrix")], K ? ["kernels", go(DM.sel.kernel(K), "kernels")] : null]);

    if (T.kind === "upload" || T.kind === "download") {
      doc(root, T.kind === "upload" ? "Host to device copy." : "Device to host copy: the host waits for the device here.");
      root.appendChild(kv([
        ["section", T.S ? [DM.link.section(T.S), " ", DM.tyEl(T.S.ty, T.S.role), T.acc.length > 1 ? h("span", { cls: "dim" }, "  the first of " + T.acc.length) : null] : "?"],
        ["offset in block", T.at !== null ? DM.fmtInt(T.at) + " (" + DM.hex(T.at) + ")" : null],
        ["bytes", DM.fmtBytesFull(T.bytes)],
        ["via", T.via],
        ["in the solve", "x" + DM.fmtInt(P.times) + " = " + DM.fmtBytesFull(T.bytes * P.times)],
        ["time", T.time !== null ? DM.fmtTime(T.time) : null]
      ]));
      flagList(root, T.flags);
      if (T.acc.length > 1 || (T.acc.length === 1 && T.acc[0].S !== T.S)) {
        var sc = sect(root, "Sections covered", String(T.acc.length));
        var tc = h("table", { cls: "grid" }, h("tr", null, th("section"), th("type"), th("bytes"), th("range in section")));
        T.acc.forEach(function (A) {
          var d = A.r || A.w;
          tc.appendChild(h("tr", null, h("td", null, DM.link.section(A.S)), h("td", null, DM.tyEl(A.S.ty, A.S.role)),
            h("td", { cls: "numc" }, DM.bytesEl(d.bytes)),
            h("td", { cls: "numc" }, d.lo !== null ? DM.fmtInt(d.lo) + ".." + DM.fmtInt(d.hi) : "-")));
        });
        sc.appendChild(h("div", { cls: "tscroll" }, tc));
      }
      edgeRows(root, T);
      return;
    }

    doc(root, K ? K.doc : "");
    var geo = DM.geomText(T), n = DM.threadCount(T), rw = DM.rwShort(T);
    root.appendChild(kv([
      ["unit", K && K.unit ? [DM.link.unit(K.unit), " ", DM.unitTypes(K.unit)] : null],
      ["geometry", geo ? geo + (n !== null ? " = " + DM.fmtInt(n) + " threads" : "") + "  (grid x threads)" : null],
      ["shared", T.shared !== null ? DM.fmtInt(T.shared) + " bytes" : null],
      ["launches", DM.fmtInt(T.launches) + " per occurrence, x" + DM.fmtInt(P.times) + " = " + DM.fmtInt(T.launches * P.times) + " in the solve" +
        (T.geoms > 1 ? "; " + T.geoms + " distinct geometries (the first is shown" + (T.runs ? ", each launch is under Launches)" : ")") : "")],
      ["lane", (T.lane || "-") + (T.graph ? ", inside a captured graph" : "")],
      ["interleaved", T.interleaved ? "its launches alternate with other kernels'; they are merged into one step per kernel, in first-launch order" : null],
      ["time", T.time !== null ? DM.fmtTime(T.time) + " per launch, " + DM.fmtTime(T.time * T.launches * P.times) + " in the solve" : null],
      ["device bytes", rw ? [DM.rwSpan("R", T.rN, T.rUnk, T.rBytes), "  ", DM.rwSpan("W", T.wN, T.wUnk, T.wBytes), h("span", { cls: "dim" }, "  per occurrence")]
        : h("span", { cls: "dim" }, "no access data")]
    ]));
    flagList(root, T.flags);

    if (T.args.length) {
      var sa = sect(root, "Arguments", String(T.args.length));
      var ta = h("table", { cls: "grid args" }, h("tr", null, th("param"), th("C type"), th("resolved", "the parameter's type with the unit's typedefs resolved"),
        th("bound to", "the section the pointer is bound to (+ offset into it), or the scalar's value"), th("this step", "what this step's access data says about that section")));
      T.args.forEach(function (G) {
        var Q = G.def, target, touched = null;
        if (G.S) {
          target = [DM.link.section(G.S), " ", DM.tyEl(G.S.ty, G.S.role), G.off ? h("span", { cls: "dim" }, " +" + DM.fmtInt(G.off)) : null];
          var A = accOf(T, G.S);
          touched = A ? [dirBrief("R", A.r), dirBrief("W", A.w)] : (T.acc.length ? h("span", { cls: "dim" }, "not touched") : null);
        } else if (G.value !== null) {
          target = h("code", null, G.value);
        } else {
          target = h("span", { cls: "dim" }, "?");
        }
        ta.appendChild(h("tr", { cls: G.mismatch === "type" ? "mm" : G.mismatch === "role" ? "mm-soft" : null },
          h("td", { cls: "mono" }, G.param),
          h("td", { cls: "mono" }, Q ? Q.ctype : h("span", { cls: "dim" }, "?")),
          h("td", null, Q ? DM.tyEl(Q.type, Q.role) : null),
          h("td", null, target),
          h("td", null, touched)));
        if (G.mismatch === "type" || G.mismatch === "role") {
          ta.appendChild(h("tr", { cls: "mm-note" + (G.mismatch === "role" ? " soft" : "") }, h("td", { colspan: "5" }, mismatchText(G))));
        }
      });
      sa.appendChild(h("div", { cls: "tscroll" }, ta));
    }

    if (T.runs) launchRows(root, T);

    if (T.acc.length) {
      var srcs = [];
      T.acc.forEach(function (A) { if (srcs.indexOf(A.src) < 0) srcs.push(A.src); });
      var sx = sect(root, "Access", T.acc.length + " sections, " + srcs.join(" + "));
      var tx = h("table", { cls: "grid acc" }, accHead("section"));
      T.acc.forEach(function (A) {
        var who = function () { return [DM.link.section(A.S), " ", DM.tyEl(A.S.ty, A.S.role)]; };
        if (A.r) tx.appendChild(dirRow(A, A.r, "r", who(), A.src));
        if (A.w) tx.appendChild(dirRow(A, A.w, "w", who(), A.src));
      });
      sx.appendChild(h("div", { cls: "tscroll" }, tx));
    }

    edgeRows(root, T);
    if (K && K.source) {
      root.appendChild(fold("Kernel source" + (K.file ? "  " + K.file + (K.line ? ":" + K.line : "") : ""), h("pre", { cls: "src" }, K.source), false));
    }
  }

  // The launches of a step of several, in execution order: each one's place
  // among the phase's launches, its geometry, the arguments that differ
  // between them and, on a timed case, its time and share of the step's.
  function launchRows(root, T) {
    var R = T.runs, total = 0, top = 0, timed = false, threads = {};
    R.list.forEach(function (r) {
      if (r.time !== null) { timed = true; total += r.time; top = Math.max(top, r.time); }
      threads[r.threads.join(",")] = true;
    });
    var manyThreads = Object.keys(threads).length > 1;
    var names = R.vary.map(function (G) { return G ? G.param : "?"; });
    var s = sect(root, "Launches", R.list.length + (names.length ? ", differing in " + names.join(", ") : ", the same arguments"));
    var head = h("tr", null, th("#"), th("at", "its place among the phase's operations (launches and copies), in execution order, from 1: the Sequence view lists them"),
      th("grid", "blocks"), manyThreads ? th("threads", "threads a block") : null);
    R.vary.forEach(function (G, j) { head.appendChild(th(names[j], "this launch's value of the argument")); });
    if (timed) {
      head.appendChild(th("time", "the launch's mean device time over the phase's occurrences"));
      head.appendChild(th("of the step", "its share of the step's time, and a bar of it against the longest launch"));
    }
    var t = h("table", { cls: "grid runs" }, head);
    R.list.forEach(function (r, i) {
      var tr = h("tr", null, h("td", { cls: "numc dim" }, String(i + 1)), h("td", { cls: "numc" }, r.n !== null ? String(r.n + 1) : "-"),
        h("td", { cls: "numc" }, r.grid.join(" x ")), manyThreads ? h("td", { cls: "numc" }, r.threads.join(" x ")) : null);
      r.v.forEach(function (x) {
        tr.appendChild(x.S ? h("td", null, DM.link.section(x.S), x.off ? h("span", { cls: "dim" }, " +" + DM.fmtInt(x.off)) : null)
          : h("td", { cls: "numc" }, x.value !== null ? x.value : x.at !== null ? "+" + DM.fmtInt(x.at) : "-"));
      });
      if (timed) {
        tr.appendChild(h("td", { cls: "numc" }, r.time !== null ? DM.fmtTime(r.time) : "-"));
        tr.appendChild(h("td", { cls: "sh-bar" }, h("span", { cls: "sh-pct" }, r.time !== null && total > 0 ? DM.fmtPct(r.time / total) : "-"),
          h("span", { cls: "sh-track" }, h("i", { cls: "run-bar", style: "width:" + (r.time !== null && top > 0 ? 100 * r.time / top : 0).toFixed(1) + "%" }))));
      }
      t.appendChild(tr);
    });
    if (timed) {
      var cells = 3 + (manyThreads ? 1 : 0) + R.vary.length;
      t.appendChild(h("tr", { cls: "total" }, h("td", { colspan: String(cells) }, "all " + R.list.length), h("td", { cls: "numc" }, DM.fmtTime(total)), h("td", null, "")));
    }
    s.appendChild(h("div", { cls: "tscroll runs-scroll" }, t));
  }

  // ------------------------------------------------------------------
  // Access (one step on one section)
  // ------------------------------------------------------------------

  function dirKV(S, d) {
    var cov = DM.coverage(S, d), span = d.lo !== null && d.hi !== null ? d.hi - d.lo + 1 : null;
    var widthBad = S.elem > 0 && d.sizes.some(function (z) { return z !== S.elem; });
    return kv([
      ["bytes", d.bytes !== null ? DM.fmtBytesFull(d.bytes) : DM.nmEl("not measured")],
      ["accesses", d.n !== null ? DM.fmtInt(d.n) : null],
      ["distinct elements", d.unique !== null ? DM.fmtInt(d.unique) + (cov !== null ? "  (" + DM.fmtPct(cov) + " of the section's " + DM.fmtInt(S.count) + ")" : "") : null],
      ["accesses per element", d.n !== null && d.unique ? DM.fmtNum(d.n / d.unique, 2) : null],
      ["byte range", span !== null ? DM.fmtInt(d.lo) + " .. " + DM.fmtInt(d.hi) + (S.bytes > 0 ? "  (" + DM.fmtPct(span / S.bytes) + " of the section)" : "") : null],
      ["threads", d.threads !== null ? DM.fmtInt(d.threads) : null],
      ["access widths", d.sizes.length ? h("span", { cls: widthBad ? "bad" : null }, d.sizes.join(", ") + " bytes" + (widthBad ? "  (the element is " + S.elem + ")" : "")) : null],
      ["pattern", [DM.dirPatEl(d), d.pattern !== "unknown" ? h("span", { cls: "dim" }, "  " + DM.patDoc(d.pattern) + (d.borrowed ? " (borrowed from the probed case)" : "")) : null]],
      ["seq", d.seq !== null ? DM.fmtNum(d.seq) + "  of a thread's accesses are adjacent to its previous one" : null],
      ["same", d.same !== null ? DM.fmtNum(d.same) + "  are equal to its previous one" : null]
    ]);
  }

  function accessDetails(root, M, res) {
    var T = res.T, S = res.S, A = res.A;
    head(root, "access", [T.label, h("span", { cls: "dim" }, " on "), S.name], T.id);
    showIn(root, [["matrix", go(res.sel, "matrix")], ["pipeline", go(DM.sel.step(T), "pipeline")], ["memory", go(DM.sel.section(S), "memory")]]);
    var binds = [];
    T.args.forEach(function (G) {
      if (G.S !== S) return;
      binds.push(h("div", { cls: G.mismatch === "type" ? "bad" : null }, h("span", { cls: "mono" }, G.param + (G.def ? ": " + G.def.ctype : "")), " ",
        G.def ? DM.tyEl(G.def.type, G.def.role) : null, G.off ? h("span", { cls: "dim" }, " +" + DM.fmtInt(G.off)) : null,
        G.mismatch === "type" || G.mismatch === "role" ? h("div", { cls: "sub" }, mismatchText(G)) : null));
    });
    root.appendChild(kv([
      ["step", DM.link.step(T)],
      ["section", DM.link.section(S)],
      ["stored", [DM.tyEl(S.ty, S.role), "  " + DM.fmtInt(S.count) + " x " + S.elem + " bytes = " + DM.fmtBytes(S.bytes)]],
      ["through", binds.length ? binds : h("span", { cls: "dim" }, "no argument of the step is bound to this section")],
      ["source", SRC_DOC[A.src] || A.src]
    ]));
    if (A.r) sect(root, "Read").appendChild(dirKV(S, A.r));
    if (A.w) sect(root, "Write").appendChild(dirKV(S, A.w));
    flagList(root, S.flags);
  }

  // ------------------------------------------------------------------
  // Kernel
  // ------------------------------------------------------------------

  function kernelDetails(root, M, K) {
    head(root, "kernel", K.name, K.defined ? "" : "no entry in kernels");
    var first = K.steps[0];
    showIn(root, [["kernels", go(DM.sel.kernel(K), "kernels")],
      first ? ["pipeline", function () { DM.select(DM.sel.kernel(K), { view: "pipeline", phase: K.phases.length === 1 ? K.phases[0].P.id : "all" }); }] : null]);
    doc(root, K.doc);
    var t = K.tot;
    root.appendChild(kv([
      ["unit", K.unit ? [DM.link.unit(K.unit), " ", DM.unitTypes(K.unit)] : null],
      ["file", K.file ? K.file + (K.line ? ":" + K.line : "") : null],
      ["launched in", K.phases.length ? K.phases.map(function (e) {
        return h("div", null, DM.link.scope(M, e.P.id, null, e.P.title), " " + DM.fmtInt(e.launches) + " per occurrence, x" + DM.fmtInt(e.P.times));
      }) : h("span", { cls: "dim" }, "never launched in this case")],
      ["in the solve", K.steps.length ? [DM.fmtInt(t.launches) + " launches,  ", h("span", { cls: "rw rd" }, h("b", null, "R"), " ", DM.amountEl(t.r, t.rUnk)), "  ",
        h("span", { cls: "rw wr" }, h("b", null, "W"), " ", DM.amountEl(t.w, t.wUnk))] : null],
      ["time", t.hasTime ? DM.fmtTime(t.time) + " in the solve" : null],
      ["read patterns", K.patR.length ? K.patR.map(DM.patEl) : null],
      ["write patterns", K.patW.length ? K.patW.map(DM.patEl) : null]
    ]));
    flagList(root, K.flags);

    // parameters, with what the launches bound them to
    var bound = Object.create(null), names = K.params.map(function (Q) { return Q.name; });
    K.steps.forEach(function (T) {
      T.args.forEach(function (G) {
        var b = bound[G.param];
        if (!b) {
          b = bound[G.param] = { secs: [], vals: [], mm: "" };
          if (names.indexOf(G.param) < 0) names.push(G.param);
        }
        if (G.S) {
          if (b.secs.indexOf(G.S) < 0) b.secs.push(G.S);
          if (G.mismatch === "type") b.mm = "type";
          else if (G.mismatch === "role" && !b.mm) b.mm = "role";
        } else if (G.value !== null && b.vals.indexOf(G.value) < 0) {
          b.vals.push(G.value);
        }
      });
    });
    if (names.length) {
      var sp = sect(root, "Parameters", String(names.length));
      var tp = h("table", { cls: "grid args" }, h("tr", null, th("param"), th("C type"), th("resolved"), th("bound to", "sections and values the launches of this case pass")));
      names.forEach(function (name) {
        var Q = K.paramByName[name] || null, b = bound[name] || { secs: [], vals: [], mm: "" }, cell = [];
        b.secs.forEach(function (S) { cell.push(h("div", null, DM.link.section(S), " ", DM.tyEl(S.ty, S.role))); });
        if (b.vals.length) cell.push(h("code", null, b.vals.slice(0, 4).join(", ") + (b.vals.length > 4 ? ", .. (" + b.vals.length + " values)" : "")));
        tp.appendChild(h("tr", { cls: b.mm === "type" ? "mm" : b.mm === "role" ? "mm-soft" : null },
          h("td", { cls: "mono" }, name),
          h("td", { cls: "mono" }, Q ? Q.ctype + (Q.ptr ? "" : Q.table ? "  (launch parameter)" : "  (scalar)") : h("span", { cls: "dim" }, "?")),
          h("td", null, Q ? DM.tyEl(Q.type, Q.role) : null),
          h("td", null, cell)));
      });
      sp.appendChild(h("div", { cls: "tscroll" }, tp));
    }

    if (K.steps.length) {
      var ss = sect(root, "Launched by", K.steps.length + " steps");
      var ts = h("table", { cls: "grid" }, h("tr", null, th("step"), th("scope"), th("geometry", "grid x threads"), th("R"), th("W")));
      K.steps.slice(0, 200).forEach(function (T) {
        ts.appendChild(h("tr", null, h("td", null, DM.link.step(T)), h("td", null, scopePath(M, T.scope)),
          h("td", { cls: "mono" }, DM.geomText(T)),
          h("td", { cls: "numc rd" }, T.rN ? (T.rUnk === T.rN ? DM.nmEl() : DM.bytesEl(T.rBytes)) : "-"),
          h("td", { cls: "numc wr" }, T.wN ? (T.wUnk === T.wN ? DM.nmEl() : DM.bytesEl(T.wBytes)) : "-")));
      });
      ss.appendChild(h("div", { cls: "tscroll" }, ts));
      if (K.steps.length > 200) ss.appendChild(h("div", { cls: "dim" }, (K.steps.length - 200) + " more steps"));
    }
    if (K.source) root.appendChild(fold("Source", h("pre", { cls: "src" }, K.source), true));
    if (K.unit && K.unit.defines) root.appendChild(fold("Unit defines (" + K.unit.name + ")", h("pre", { cls: "src" }, K.unit.defines), false));
  }

  // ------------------------------------------------------------------
  // Section
  // ------------------------------------------------------------------

  function accessList(root, title, S, dir) {
    var list = S.touches.filter(function (A) { return dir === "r" ? !!A.r : !!A.w; });
    var s = sect(root, title, String(list.length));
    if (!list.length) {
      s.appendChild(h("div", { cls: "none-warn" }, dir === "r" ? "nothing reads it" : "nothing writes it"));
      return;
    }
    var t = h("table", { cls: "grid acc" }, accHead("step"));
    list.slice(0, 150).forEach(function (A) {
      t.appendChild(dirRow(A, dir === "r" ? A.r : A.w, dir, DM.link.step(A.T), A.T.phase.id + ", " + A.src));
    });
    s.appendChild(h("div", { cls: "tscroll" }, t));
    if (list.length > 150) s.appendChild(h("div", { cls: "dim" }, (list.length - 150) + " more"));
  }

  // ------------------------------------------------------------------
  // Computable index words
  // ------------------------------------------------------------------

  // "= 9 x schur.plan.SO field 3": what a sequence's words are of another's.
  function copyEl(c, o) {
    return h("span", { cls: "cmp-copy", tip: DM.fmtInt(c.n) + " words equal to " + (o.times ? o.times + "those" : "those") + " of " + o.S.qname + (o.field >= 0 ? ", field " + o.field + " of its records" : "") },
      (c.field >= 0 ? "field " + c.field + " " : "") + "= " + o.times, DM.link.section(o.S), o.field >= 0 ? " field " + o.field : null);
  }

  // How an index section's words could be had without storing them: from
  // their position (a formula in i, the word's index, or r, the record's),
  // or from another section that holds the same words or a multiple of
  // them. brief: one line for a table cell. null for other sections.
  DM.computableEl = function (S, brief) {
    if (!S.content) return null;
    var out = h("span", { cls: "cmp" });
    var per = S.content.kind === "records" ? Number(S.content.period) || 0 : 0, fits = Math.round(S.computable * per);
    if (S.computable >= 1) {
      out.appendChild(h("span", { cls: "cmp-yes", tip: "computable from the position: " + (per ? "r is the record's index" : "i is the word's index") }, S.formula || "yes"));
    } else if (S.computable > 0) {
      out.appendChild(h("span", { cls: "cmp-part", tip: fits + " of " + per + " fields of a record are computable from the record's index r; ? marks the others" },
        h("b", null, fits + "/" + per), " ", S.formula));
    }
    var shown = 0, more = 0;
    S.copies.forEach(function (c) {
      c.others.forEach(function (o) {
        if (brief && shown >= 1) { more++; return; }
        if (out.firstChild) out.appendChild(h("span", { cls: "dim" }, brief ? "  " : "; "));
        out.appendChild(copyEl(c, o));
        shown++;
      });
    });
    if (more) out.appendChild(h("span", { cls: "dim", tip: more + " more sections hold the same words" }, " +" + more));
    if (!out.firstChild) {
      out.appendChild(h("span", { cls: "dim", tip: "not computable from the position, and no other index section holds the same words" }, "no"));
      // what it holds instead: its first words
      if (brief && S.sample) {
        out.appendChild(h("span", { cls: "cmp-peek", tip: "its first words" },
          "  " + S.sample.head.slice(0, 6).join(", ") + (S.count > 6 ? ", ..." : "")));
      }
    }
    return out;
  };

  // ------------------------------------------------------------------
  // An index section's words
  // ------------------------------------------------------------------

  function signed(d) { return d > 0 ? "+" + d : String(d); }

  // Rows of `cols` words, the row's label the index of its first word.
  // parts: [{ words, first }] with first the index of words[0]; a gap row
  // stands between two parts. With `steps` a cell holds the word's
  // difference to the one `lag` words before it (1: its neighbour; a
  // record's width: the same field of the record before), a zero dimmed.
  function wordGrid(parts, cols, steps, lag, heads) {
    lag = lag || 1;
    var t = h("table", { cls: "vals" + (steps ? " steps" : "") });
    if (heads && heads.length) {
      // a record's fields: what computes each from the record's index r, "?" where nothing does
      t.appendChild(h("tr", { cls: "vals-head" }, h("th", null, "r"), heads.map(function (f, j) {
        return h("td", { cls: f === "?" ? "no" : "yes", tip: "field " + j + (f === "?" ? ": not computable from the record's index" : ": " + f + ", r the record's index") }, f);
      })));
    }
    parts.forEach(function (part, pi) {
      var words = part.words;
      if (pi > 0) t.appendChild(h("tr", { cls: "vals-skip" }, h("th", null, ""), h("td", { colspan: cols }, ". . .")));
      for (var r = 0; r < words.length; r += cols) {
        var tr = h("tr", null, h("th", null, DM.fmtInt(part.first + r)));
        for (var c = r; c < Math.min(r + cols, words.length); c++) {
          var d = c >= lag ? words[c] - words[c - lag] : null, step = d === null ? "" : signed(d);
          tr.appendChild(h("td", { cls: steps && d === 0 ? "z" : null,
            tip: "word " + DM.fmtInt(part.first + c) + ": " + words[c] + (step ? "\n" + step + (lag > 1 ? " from the same field of the record before it" : " from the word before it") : "") },
            steps ? step : String(words[c])));
        }
        t.appendChild(tr);
      }
    });
    return t;
  }

  // The words the trace kept of an index section: the first ones and the
  // last ones, as they were on the device when the block was described.
  // Records are a record a row, each field's formula above its column.
  function valuesBlock(root, S) {
    var sm = S.sample;
    if (!sm) return;
    var C = S.content || {}, kind = C.kind || "";
    // a row: a record where the section's words are records (declared, or found from the data), else 8 words
    var rec = kind === "records" && C.period >= 2 && C.period <= 16 ? C.period : S.record >= 2 && S.record <= 16 ? S.record : 0;
    var cols = rec || 8;
    var n = S.count || C.n || sm.head.length;
    var constant = kind === "constant", whole = !sm.tail.length;
    // whole rows only where more follow, so that every column keeps its meaning
    var head = constant ? sm.head.slice(0, cols) : whole ? sm.head : sm.head.slice(0, sm.head.length - sm.head.length % cols);
    var parts = [{ words: head, first: 0 }];
    if (!whole && !constant) {
      var first = n - sm.tail.length, skip = (cols - first % cols) % cols;
      if (skip < sm.tail.length) parts.push({ words: sm.tail.slice(skip), first: first + skip });
    }
    var shown = parts.length > 1 ? "the first " + head.length + " and the last " + parts[1].words.length + " of " + DM.fmtInt(n) + " words" :
      "the first " + head.length + " of " + DM.fmtInt(n) + " words";
    var s = sect(root, "Values", constant ? "every one of " + DM.fmtInt(n) + " words is " + sm.head[0] : whole ? "all " + DM.fmtInt(head.length) + " words" : shown);
    s.appendChild(h("div", { cls: "vals-note dim" }, "As on the device when the block was described" +
      (rec ? "; a record of " + rec + " words a row" : "") + ". Left: the index of the row's first word."));
    s.appendChild(h("div", { cls: "tscroll" }, wordGrid(parts, cols, false, rec || 1, rec && S.fields.length === rec ? S.fields : null)));
    if (kind === "monotone" || kind === "other" || (kind === "records" && S.computable < 1)) {
      s.appendChild(h("div", { cls: "vals-note dim" }, rec ? "Steps: each word minus the same field of the record before it." : "Steps: each word minus the one before it."));
      s.appendChild(h("div", { cls: "tscroll" }, wordGrid([parts[0]], cols, true, rec || 1)));
    }
  }

  function sectionDetails(root, M, S) {
    var B = S.block, me = DM.sel.section(S);
    head(root, "section", S.name, S.missing ? "not in blocks" : "");
    showIn(root, [["memory", go(me, "memory")], ["matrix", go(me, "matrix")], ["pipeline", go(me, "pipeline")]]);
    doc(root, S.doc);
    if (S.about) root.appendChild(h("p", { cls: "doc about" }, S.about));
    var natural = Object.prototype.hasOwnProperty.call(DM.TYPE_SIZE, S.ty) ? DM.TYPE_SIZE[S.ty] : 0;
    root.appendChild(kv([
      ["block", [DM.link.block(B), h("span", { cls: "dim" }, "  section " + S.i)]],
      ["offset", DM.fmtInt(S.at) + " (" + DM.hex(S.at) + ")  ..  " + DM.fmtInt(S.at + S.bytes)],
      ["bytes", DM.fmtBytesFull(S.bytes)],
      ["elements", [DM.fmtInt(S.count) + " x " + S.elem + " bytes", S.record > 1 ? h("span", { cls: "dim" }, "  in records of " + S.record) : null,
        natural && S.elem && natural !== S.elem ? h("span", { cls: "bad" }, "  (" + S.ty + " is " + natural + " bytes)") : null]],
      ["stored type", [DM.tyEl(S.ty), S.role ? [h("span", { cls: "dim" }, "  kernel-side role "), h("b", { cls: "mono" }, S.role)] : h("span", { cls: "dim" }, "  no role")]],
      ["group", S.group],
      ["computable", DM.computableEl(S, false)]
    ]));
    flagList(root, S.flags);
    valuesBlock(root, S);

    // totals per phase
    var per = Object.create(null), order = [];
    S.touches.forEach(function (A) {
      var P = A.T.phase, e = per[P.id];
      if (!e) {
        e = per[P.id] = { P: P, r: 0, w: 0, nr: 0, nw: 0, ru: false, wu: false };
        order.push(e);
      }
      if (A.r) {
        e.nr++;
        if (A.r.bytes !== null) e.r += A.r.bytes; else e.ru = true;
      }
      if (A.w) {
        e.nw++;
        if (A.w.bytes !== null) e.w += A.w.bytes; else e.wu = true;
      }
    });
    if (order.length) {
      var sp = sect(root, "Per phase", "one occurrence, and the solve");
      var tp = h("table", { cls: "grid" }, h("tr", null, th("phase"), th("x"), th("R / occ"), th("W / occ"), th("R solve"), th("W solve")));
      order.forEach(function (e) {
        tp.appendChild(h("tr", null, h("td", null, DM.link.scope(M, e.P.id, null, e.P.title)), h("td", { cls: "numc" }, DM.fmtInt(e.P.times)),
          h("td", { cls: "numc rd" }, e.nr ? DM.amountEl(e.r, e.ru) : "-"), h("td", { cls: "numc wr" }, e.nw ? DM.amountEl(e.w, e.wu) : "-"),
          h("td", { cls: "numc rd" }, e.nr ? DM.amountEl(e.r * e.P.times, e.ru) : "-"), h("td", { cls: "numc wr" }, e.nw ? DM.amountEl(e.w * e.P.times, e.wu) : "-")));
      });
      tp.appendChild(h("tr", { cls: "total" }, h("td", null, "solve"), h("td", null, ""), h("td", null, ""), h("td", null, ""),
        h("td", { cls: "numc rd" }, DM.amountEl(S.tot.r, S.tot.rUnk)), h("td", { cls: "numc wr" }, DM.amountEl(S.tot.w, S.tot.wUnk))));
      sp.appendChild(h("div", { cls: "tscroll" }, tp));
    }

    accessList(root, "Written by", S, "w");
    accessList(root, "Read by", S, "r");

    // the kernel parameters the section is bound to
    var uses = Object.create(null), uorder = [];
    S.uses.forEach(function (u) {
      var key = u.T.kname + "." + u.G.param, e = uses[key];
      if (!e) {
        e = uses[key] = { T: u.T, G: u.G, n: 0 };
        uorder.push(e);
      }
      e.n++;
    });
    if (uorder.length) {
      var su = sect(root, "Bound to parameters", String(uorder.length));
      var tu = h("table", { cls: "grid args" }, h("tr", null, th("kernel.param"), th("C type"), th("resolved"), th("steps")));
      uorder.forEach(function (e) {
        var Q = e.G.def, K = e.T.kernel, mm = e.G.mismatch;
        tu.appendChild(h("tr", { cls: mm === "type" ? "mm" : mm === "role" ? "mm-soft" : null },
          h("td", null, K ? DM.link.kernel(K) : e.T.label, h("span", { cls: "mono" }, "." + e.G.param)),
          h("td", { cls: "mono" }, Q ? Q.ctype : h("span", { cls: "dim" }, "?")),
          h("td", null, Q ? DM.tyEl(Q.type, Q.role) : null),
          h("td", { cls: "numc" }, DM.link.step(e.T), e.n > 1 ? h("span", { cls: "dim" }, " +" + (e.n - 1)) : null)));
        if (mm === "type" || mm === "role") {
          tu.appendChild(h("tr", { cls: "mm-note" + (mm === "role" ? " soft" : "") }, h("td", { colspan: "4" }, mismatchText(e.G))));
        }
      });
      su.appendChild(h("div", { cls: "tscroll" }, tu));
    }

    if (S.edges.length) {
      var se = sect(root, "Dataflow", S.edges.length + " edges");
      S.edges.slice(0, 150).forEach(function (E) {
        se.appendChild(h("div", { cls: "edge-row" }, E.from ? DM.link.step(E.from) : h("span", { cls: "hostref" }, "host"),
          h("span", { cls: "dim" }, "  ->  "), E.to ? DM.link.step(E.to) : h("span", { cls: "hostref" }, "host")));
      });
      if (S.edges.length > 150) se.appendChild(h("div", { cls: "dim" }, (S.edges.length - 150) + " more"));
    }
  }

  // ------------------------------------------------------------------
  // Block, unit, scope, case
  // ------------------------------------------------------------------

  function blockDetails(root, M, B) {
    head(root, "block", B.name, B.missing ? "not in blocks" : "id " + B.id);
    showIn(root, [["memory", go(DM.sel.block(B), "memory")]]);
    doc(root, B.doc);
    var gaps = B.bytes - B.used, types = Object.create(null);
    B.sections.forEach(function (S) { types[S.ty || "?"] = (types[S.ty || "?"] || 0) + S.bytes; });
    root.appendChild(kv([
      ["owner", B.owner],
      ["bytes", DM.fmtBytesFull(B.bytes)],
      ["sections", DM.fmtInt(B.sections.length) + ", " + DM.fmtBytes(B.used) + " used, " + DM.fmtBytes(gaps) + " in gaps (" + DM.fmtPct(B.bytes > 0 ? gaps / B.bytes : 0) + ")"],
      ["by type", Object.keys(types).map(function (ty) { return h("span", { cls: "ty-sum" }, DM.tyEl(ty), " " + DM.fmtBytes(types[ty]) + "  "); })]
    ]));
    var s = sect(root, "Sections", "in memory order");
    var t = h("table", { cls: "grid" }, h("tr", null, th("section"), th("type"), th("at"), th("bytes"), th("checks")));
    B.order.forEach(function (S) {
      var lv = DM.worstLevel(S.flags);
      t.appendChild(h("tr", null, h("td", null, DM.link.section(S)), h("td", null, DM.tyEl(S.ty, S.role)),
        h("td", { cls: "numc" }, DM.fmtInt(S.at)), h("td", { cls: "numc" }, DM.bytesEl(S.bytes)),
        h("td", null, lv ? [DM.levelBadge(lv), " " + S.flags.length] : null)));
    });
    s.appendChild(h("div", { cls: "tscroll" }, t));
  }

  function unitDetails(root, M, U) {
    head(root, "unit", U.name, "id " + U.id);
    root.appendChild(kv([
      ["file", U.file],
      ["types", DM.unitTypes(U)],
      ["kernels", String(U.kernels.length)]
    ]));
    if (U.defines) root.appendChild(fold("Defines", h("pre", { cls: "src" }, U.defines), true));
    var s = sect(root, "Kernels", String(U.kernels.length));
    U.kernels.forEach(function (K) {
      s.appendChild(h("div", { cls: "edge-row" }, DM.link.kernel(K),
        h("span", { cls: "dim" }, K.steps.length ? "  " + DM.fmtInt(K.tot.launches) + " launches" : "  not launched")));
    });
  }

  function scopeDetails(root, M, path) {
    var info = M.scopeInfo(path), steps = M.scopeSteps(path), parts = path.split("/");
    var P = M.phaseById[parts[0]] || steps[0].phase, isPhase = parts.length === 1 && !!M.phaseById[path];
    var me = DM.sel.scope(path), tot = DM.sumSteps(steps);
    head(root, isPhase ? "phase" : "scope", isPhase ? P.title : info.title, path);
    showIn(root, [["pipeline", function () { DM.select(me, { view: "pipeline", phase: P.id }); }],
      ["matrix", function () { DM.select(me, { view: "matrix", phase: P.id }); }], ["overview", go(me, "overview")]]);
    doc(root, isPhase ? P.doc : info.doc);
    root.appendChild(kv([
      ["phase", isPhase ? null : DM.link.scope(M, P.id, null, P.title)],
      ["runs", "x" + DM.fmtInt(P.times) + " in the solve"],
      ["variants", isPhase && P.variants > 1 ? h("span", { cls: "bad" }, P.variants + " distinct step sequences; the steps shown are the most frequent one's") : null],
      ["steps", DM.fmtInt(tot.steps) + ", " + DM.fmtInt(tot.launches) + " launches per occurrence"],
      ["device bytes", [h("span", { cls: "rw rd" }, h("b", null, "R"), " ", DM.amountEl(tot.r, tot.rUnk)), "  ",
        h("span", { cls: "rw wr" }, h("b", null, "W"), " ", DM.amountEl(tot.w, tot.wUnk)), h("span", { cls: "dim" }, "  per occurrence")]],
      ["uploads", tot.nUp ? tot.nUp + ", " + DM.fmtBytesFull(tot.up) : null],
      ["downloads", tot.nDown ? tot.nDown + ", " + DM.fmtBytesFull(tot.down) + "  (the host waits)" : null],
      ["time", tot.hasTime ? DM.fmtTime(tot.time) + " per occurrence, " + DM.fmtTime(tot.time * P.times) + " in the solve" : null]
    ]));
    // sub-scopes, one level down
    var kids = [], seen = Object.create(null);
    steps.forEach(function (T) {
      if (T.scope.length <= parts.length) return;
      var k = T.scope.slice(0, parts.length + 1).join("/");
      if (!seen[k]) {
        seen[k] = true;
        kids.push(k);
      }
    });
    if (kids.length) {
      var sk = sect(root, "Sub-scopes", String(kids.length));
      kids.forEach(function (k) {
        var ki = M.scopeInfo(k);
        sk.appendChild(h("div", { cls: "edge-row" }, DM.link.scope(M, k), ki.doc ? h("span", { cls: "dim" }, "  " + ki.doc) : null));
      });
    }
    var ss = sect(root, "Steps", String(steps.length));
    steps.slice(0, 150).forEach(function (T) {
      ss.appendChild(h("div", { cls: "edge-row" }, DM.link.step(T)));
    });
    if (steps.length > 150) ss.appendChild(h("div", { cls: "dim" }, (steps.length - 150) + " more"));
  }

  function caseSummary(root, M) {
    var m = M.meta, t = M.tot;
    head(root, "case", M.title, M.id);
    root.appendChild(h("p", { cls: "doc" }, "Click a kernel, step, section or block in any view to see it here. The search box finds them by name."));
    var rows = [];
    var known = ["problem", "model", "wide", "runtime", "device", "probe", "commit", "generated"];
    Object.keys(m).forEach(function (k) {
      if (known.indexOf(k) < 0 && k !== "notes") known.push(k);
    });
    known.forEach(function (k) {
      var v = m[k];
      if (v !== undefined && v !== null && typeof v !== "object") rows.push([k, String(v)]);
    });
    root.appendChild(kv(rows));
    if (Array.isArray(m.notes) && m.notes.length) {
      var sn = sect(root, "Notes");
      m.notes.forEach(function (n) { sn.appendChild(h("div", { cls: "note" }, String(n))); });
    }
    sect(root, "The solve").appendChild(kv([
      ["launches", DM.fmtInt(t.launches)],
      ["device bytes", [h("span", { cls: "rw rd" }, h("b", null, "R"), " ", DM.amountEl(t.r, t.rUnk)), "  ", h("span", { cls: "rw wr" }, h("b", null, "W"), " ", DM.amountEl(t.w, t.wUnk))]],
      ["uploaded", DM.fmtBytesFull(t.up)],
      ["downloaded", DM.fmtBytesFull(t.down) + ", " + DM.fmtInt(t.waits) + " host waits"],
      ["time", t.hasTime ? DM.fmtTime(t.time) : null],
      ["findings", M.findings.length + " recorded, " + M.checks.length + " computed checks"]
    ]));
    var sp = sect(root, "Phases", String(M.phases.length));
    M.phases.forEach(function (P) {
      sp.appendChild(h("div", { cls: "edge-row" }, DM.link.scope(M, P.id, null, P.title),
        h("span", { cls: "dim" }, "  x" + DM.fmtInt(P.times) + ", " + P.tot.steps + " steps, " + P.tot.launches + " launches")));
    });
    var sb = sect(root, "Blocks", String(M.blocks.length));
    M.blocks.forEach(function (B) {
      sb.appendChild(h("div", { cls: "edge-row" }, DM.link.block(B), h("span", { cls: "dim" }, "  " + DM.fmtBytes(B.bytes) + ", " + B.sections.length + " sections")));
    });
    var su = sect(root, "Units", String(M.units.length));
    M.units.forEach(function (U) {
      su.appendChild(h("div", { cls: "edge-row" }, DM.link.unit(U), " ", DM.unitTypes(U)));
    });
    if (DM.loadLog.length) {
      var sl = sect(root, "Data files");
      DM.loadLog.forEach(function (x) {
        sl.appendChild(h("div", { cls: x.ok ? "note" : "note bad" }, x.file + ": " + (x.ok ? "loaded" : (x.note || "not loaded"))));
      });
    }
  }

  DM.details = {
    render: function (root, M, res) {
      if (!res) caseSummary(root, M);
      else if (res.kind === "step") stepDetails(root, M, res.T);
      else if (res.kind === "access") accessDetails(root, M, res);
      else if (res.kind === "kernel") kernelDetails(root, M, res.K);
      else if (res.kind === "section") sectionDetails(root, M, res.S);
      else if (res.kind === "block") blockDetails(root, M, res.B);
      else if (res.kind === "unit") unitDetails(root, M, res.U);
      else if (res.kind === "scope") scopeDetails(root, M, res.path);
      else caseSummary(root, M);
    }
  };
})();
