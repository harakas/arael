// devmap: the data-indexing layer. DM.buildModel(case) wraps a case
// (SCHEMA.md section 1) in objects with the lookups the views need, and
// runs the viewer's own checks. Nothing here touches the DOM.
//
// Objects (the fields the views use):
//   M  model    units, unitById, kernels (by name), kernelList, blocks, blockById,
//               sections, secByKey, phases, phaseById, steps, stepById, edges,
//               flow, findings, checks, search, tot, resolve(sel), phasesFor(id),
//               scopeInfo(path), scopeSteps(path)
//   U  unit     id, name, file, defines, types, kernels[K]
//   K  kernel   name, unit, file, line, doc, params[Q], paramByName, source, defined,
//               steps[T], phases[{P, launches, steps, r, w}], tot, patR, patW, flags
//   Q  param    name, ctype, type, ptr, isConst, role
//   B  block    id, name, owner, bytes, doc, sections[S], order[S] (by offset), used
//   S  section  block, i, key ("block:i"), name, qname, label, at, bytes, elem, count,
//               ty, role, group, doc, touches[A], uses[{T, G}], edges[E], tot, flags
//   P  phase    id, title, doc, times, variants, steps[T], tot
//   T  step     id, kind, phase, idx, scope[], scopeKey, label, acc[A], flags,
//               edgesIn[E], edgesOut[E], time, n (its place among the phase's
//               operations, or null; DM.callsOf lists them in order);
//               launch: kernel, kname, launches, grid, threads, shared, geoms, lane,
//                       graph, interleaved, args[G], rBytes, wBytes, rN, wN, rUnk, wUnk,
//                       runs (null, or of a step of several launches: vary[G], the
//                       arguments that differ, and list[{n, grid, threads, time, v[]}],
//                       v a value per vary: {value} or {S, at, off})
//               upload / download: S (the first section), target (text), bytes, at, via
//   G  argument param, def (Q), S, at, off, value, mismatch ("", "type", "role", "untyped")
//   A  access   T, S, src (measured, static, args, host), r, w, host, derived, maybeR
//      r / w    n, bytes, unique, lo, hi, threads, sizes[], seq, same, pattern
//   E  edge     from (T or null = host), to (T or null = host), S, fromHost, toHost
"use strict";
(function () {
  var DM = window.DM = window.DM || {};

  function arr(x) { return Array.isArray(x) ? x : []; }
  function num(x, d) { return typeof x === "number" && isFinite(x) ? x : d; }
  function str(x, d) { return x === null || x === undefined ? d : String(x); }
  function has(o, k) { return Object.prototype.hasOwnProperty.call(o, k); }
  function isObj(x) { return !!x && typeof x === "object" && !Array.isArray(x); }
  function given(x) { return x !== null && x !== undefined; }

  // ------------------------------------------------------------------
  // Helpers the views share
  // ------------------------------------------------------------------

  // Totals of a list of steps, for one occurrence of their phase.
  DM.sumSteps = function (steps) {
    var t = { steps: steps.length, launches: 0, r: 0, w: 0, up: 0, down: 0, nUp: 0, nDown: 0, time: 0, hasTime: false,
              rUnk: false, wUnk: false, unk: false };
    steps.forEach(function (T) {
      if (T.kind === "upload") { t.up += T.bytes; t.nUp++; }
      else if (T.kind === "download") { t.down += T.bytes; t.nDown++; }
      else {
        if (T.kind === "launch") t.launches += T.launches;
        t.r += T.rBytes;
        t.w += T.wBytes;
        if (T.rUnk) t.rUnk = true;
        if (T.wUnk) t.wUnk = true;
        if (T.rUnk || T.wUnk) t.unk = true;
      }
      if (T.time !== null) { t.time += T.time * T.launches; t.hasTime = true; }
    });
    return t;
  };

  // "125 x 256": grid x threads of the step's first launch.
  DM.geomText = function (T) {
    if (!T.grid.length && !T.threads.length) return "";
    var g = !T.grid.length ? "?" : T.grid.length === 1 ? String(T.grid[0]) : "(" + T.grid.join(",") + ")";
    var t = !T.threads.length ? "?" : T.threads.length === 1 ? String(T.threads[0]) : "(" + T.threads.join(",") + ")";
    return g + " x " + t;
  };

  // The operations (launches and copies) of one occurrence of the phase P
  // in execution order: { T, r, i, n } each, r one launch of a step of
  // several (i its index among them) or null, n its place from 0. null when
  // the case does not say the places.
  DM.callsOf = function (P) {
    var out = [], known = true;
    P.steps.forEach(function (T) {
      if (T.runs) {
        T.runs.list.forEach(function (r, i) {
          if (r.n === null) known = false;
          out.push({ T: T, r: r, i: i, n: r.n });
        });
      } else {
        if (T.n === null || T.launches > 1) known = false;
        out.push({ T: T, r: null, i: 0, n: T.n });
      }
    });
    if (!known) return null;
    out.sort(function (a, b) { return a.n - b.n; });
    return out;
  };

  // grid x threads multiplied out, or null.
  DM.threadCount = function (T) {
    if (!T.grid.length || !T.threads.length) return null;
    var n = 1;
    T.grid.concat(T.threads).forEach(function (v) { n *= num(v, 1); });
    return n;
  };

  function dirText(n, unk, bytes) {
    return n === 0 ? "-" : unk === n ? DM.NM : DM.fmtBytes(bytes) + (unk ? " +" + DM.NM : "");
  }
  // { r: "4.46 MB", w: "-" } for a step, or null when it has no access data
  // ("n/m": the step reads or writes, uncounted).
  DM.rwShort = function (T) {
    if (!T.acc.length) return null;
    return { r: dirText(T.rN, T.rUnk, T.rBytes), w: dirText(T.wN, T.wUnk, T.wBytes) };
  };

  // Elements a direction touched / the section's elements, or null.
  DM.coverage = function (S, d) {
    if (!d || d.unique === null) return null;
    if (S.count > 0) return d.unique / S.count;
    return null;
  };

  // Which view shows a kind of selection best.
  DM.viewFor = { step: "pipeline", scope: "pipeline", kernel: "kernels", unit: "kernels", section: "memory", block: "memory", access: "matrix" };

  // The viewer's checks: level and title per code, in display order.
  DM.CHECKS = {
    "missing":           { level: "bug",  title: "Reference to a block or section that is not listed" },
    "overlap":           { level: "bug",  title: "Sections overlap" },
    "out-of-block":      { level: "bug",  title: "Section past the end of its block" },
    "elem-too-small":    { level: "bug",  title: "Element smaller than the stored type" },
    "type-mismatch":     { level: "warn", title: "Parameter type differs from the section's stored type" },
    "width-mismatch":    { level: "warn", title: "Access width differs from the element size" },
    "narrow-in-wide":    { level: "warn", title: "Element wider than the stored type" },
    "read-before-write": { level: "warn", title: "Read before any write" },
    "never-written":     { level: "warn", title: "Read but never written" },
    "never-read":        { level: "warn", title: "Written but never read" },
    "alias":             { level: "info", title: "One section passed through several parameters" },
    "role-mismatch":     { level: "info", title: "Parameter role differs from the section's role (same type in this build)" },
    "size":              { level: "info", title: "Section bytes differ from elem x count" },
    "untouched":         { level: "info", title: "Never accessed" },
    "variants":          { level: "info", title: "Phase with several step sequences" },
    "bad-edge":          { level: "info", title: "Edges that name unknown steps" },
    "no-access":         { level: "info", title: "Access data incomplete" }
  };
  DM.CHECK_ORDER = Object.keys(DM.CHECKS);
  DM.LEVEL_RANK = { bug: 0, warn: 1, info: 2 };

  // The worst level in a list of findings or checks: "bug", "warn", "info" or "".
  DM.worstLevel = function (flags) {
    var best = "";
    flags.forEach(function (f) {
      if (best === "" || DM.LEVEL_RANK[f.level] < DM.LEVEL_RANK[best]) best = f.level;
    });
    return best;
  };

  // ------------------------------------------------------------------
  // buildModel
  // ------------------------------------------------------------------

  DM.buildModel = function (c) {
    c = isObj(c) ? c : {};
    var M = { c: c, id: str(c.id, "case"), title: str(c.title, str(c.id, "case")), meta: isObj(c.meta) ? c.meta : {}, schema: c.schema };
    var scopes = isObj(c.scopes) ? c.scopes : {};

    // Title and doc of a scope path ("attempt/reduce").
    M.scopeInfo = function (path) {
      var sc = has(scopes, path) && isObj(scopes[path]) ? scopes[path] : null;
      var parts = String(path).split("/");
      return { path: path, title: sc && sc.title ? String(sc.title) : parts[parts.length - 1], doc: sc && sc.doc ? String(sc.doc) : "" };
    };

    indexUnits(M);
    indexKernels(M);
    indexBlocks(M);
    indexCopies(M);
    indexPhases(M);
    indexEdges(M);
    indexFlow(M);
    aggregate(M);
    indexFindings(M);
    runChecks(M);
    buildSearch(M);

    // The steps in a scope and below it.
    M.scopeSteps = function (path) {
      var pre = path + "/";
      return M.steps.filter(function (T) { return T.scopeKey === path || T.scopeKey.slice(0, pre.length) === pre; });
    };

    // "all" or a phase id -> the phases shown.
    M.phasesFor = function (id) {
      return id !== "all" && M.phaseById[id] ? [M.phaseById[id]] : M.phases;
    };

    // A selection string ("step:s9", "section:3:0", ...) -> what it names, or null.
    M.resolve = function (sel) {
      if (typeof sel !== "string") return null;
      var i = sel.indexOf(":");
      if (i < 0) return null;
      var kind = sel.slice(0, i), rest = sel.slice(i + 1);
      if (kind === "step") return M.stepById[rest] ? { kind: kind, sel: sel, T: M.stepById[rest] } : null;
      if (kind === "kernel") return M.kernels[rest] ? { kind: kind, sel: sel, K: M.kernels[rest] } : null;
      if (kind === "section") return M.secByKey[rest] ? { kind: kind, sel: sel, S: M.secByKey[rest] } : null;
      if (kind === "block") return M.blockById[rest] ? { kind: kind, sel: sel, B: M.blockById[rest] } : null;
      if (kind === "unit") return M.unitById[rest] ? { kind: kind, sel: sel, U: M.unitById[rest] } : null;
      if (kind === "scope") return M.scopeSteps(rest).length ? { kind: kind, sel: sel, path: rest } : null;
      if (kind === "access") {
        // access:<step id>:<block>:<section>
        var p = rest.split(":");
        if (p.length < 3) return null;
        var S = M.secByKey[p[p.length - 2] + ":" + p[p.length - 1]];
        var T = M.stepById[p.slice(0, p.length - 2).join(":")];
        if (!S || !T) return null;
        for (var j = 0; j < T.acc.length; j++) {
          if (T.acc[j].S === S) return { kind: kind, sel: sel, T: T, S: S, A: T.acc[j] };
        }
        return null;
      }
      return null;
    };

    return M;
  };

  // ------------------------------------------------------------------
  // Units and kernels
  // ------------------------------------------------------------------

  function indexUnits(M) {
    M.units = [];
    M.unitById = Object.create(null);
    arr(M.c.units).forEach(function (u, i) {
      if (!isObj(u)) return;
      var U = { id: given(u.id) ? u.id : i, name: str(u.name, "unit " + i), file: str(u.file, ""), defines: str(u.defines, ""),
                types: isObj(u.types) ? u.types : {}, kernelNames: arr(u.kernels).map(String), kernels: [], raw: u };
      M.units.push(U);
      M.unitById[U.id] = U;
    });
  }

  function makeKernel(M, name) {
    var K = { name: name, unit: null, file: "", line: 0, doc: "", params: [], paramByName: Object.create(null), source: "",
              defined: false, steps: [], phases: [], patR: [], patW: [], flags: [], order: -1,
              tot: { occ: 0, launches: 0, r: 0, w: 0, time: 0, hasTime: false, rUnk: false, wUnk: false, unk: false } };
    M.kernels[name] = K;
    M.kernelList.push(K);
    return K;
  }

  // The type name in a C parameter type: "const V*" -> "V".
  function baseType(ctype) {
    var toks = String(ctype).split("*").join(" ").split("&").join(" ").split(" "), out = "";
    for (var i = 0; i < toks.length; i++) {
      var t = toks[i];
      if (!t || t === "const" || t === "volatile" || t === "restrict" || t === "RESTRICT" || t === "GLOBAL") continue;
      out = t;
    }
    return out;
  }

  function indexKernels(M) {
    M.kernels = Object.create(null);
    M.kernelList = [];
    var raw = isObj(M.c.kernels) ? M.c.kernels : {};
    Object.keys(raw).forEach(function (name) {
      var k = isObj(raw[name]) ? raw[name] : {};
      var K = makeKernel(M, name);
      K.defined = true;
      K.file = str(k.file, "");
      K.line = num(k.line, 0);
      K.doc = str(k.doc, "");
      K.source = str(k.source, "");
      if (given(k.unit) && M.unitById[k.unit]) K.unit = M.unitById[k.unit];
      arr(k.params).forEach(function (p) {
        if (!isObj(p)) return;
        var Q = { name: str(p.name, "?"), ctype: str(p.ctype, ""), type: str(p.type, ""), ptr: !!p.ptr, isConst: !!p["const"], role: "" };
        K.params.push(Q);
        K.paramByName[Q.name] = Q;
      });
    });
    // kernels a unit lists (they may have no entry in kernels)
    M.units.forEach(function (U) {
      U.kernelNames.forEach(function (name) {
        var K = M.kernels[name] || makeKernel(M, name);
        if (!K.unit) K.unit = U;
        if (U.kernels.indexOf(K) < 0) U.kernels.push(K);
      });
    });
    M.kernelList.forEach(function (K) {
      if (K.unit && K.unit.kernels.indexOf(K) < 0) K.unit.kernels.push(K);
      K.params.forEach(function (Q) {
        var b = baseType(Q.ctype);
        Q.role = K.unit && has(K.unit.types, b) ? b : "";
      });
    });
  }

  // ------------------------------------------------------------------
  // Blocks and sections
  // ------------------------------------------------------------------

  function makeBlock(M, b, idx) {
    var id = given(b.id) ? b.id : idx;
    var B = { id: id, name: str(b.name, "block#" + id), owner: str(b.owner, ""), bytes: num(b.bytes, 0), doc: str(b.doc, ""),
              sections: [], secByI: Object.create(null), order: [], used: 0, missing: false, raw: b };
    M.blocks.push(B);
    M.blockById[B.id] = B;
    return B;
  }

  // The words a trace kept of an index section: { head: [...], tail: [...] }
  // (tail empty when head holds them all), or null.
  function sampleOf(x) {
    if (!isObj(x)) return null;
    var nums = function (v) {
      return arr(v).filter(function (n) { return typeof n === "number" && isFinite(n); });
    };
    var head = nums(x.head);
    return head.length ? { head: head, tail: nums(x.tail) } : null;
  }

  function makeSection(M, B, s, idx) {
    var i = given(s.i) ? s.i : idx;
    var S = { block: B, i: i, key: B.id + ":" + i, name: str(s.name, "sec#" + i), at: num(s.at, null), bytes: num(s.bytes, 0),
              elem: num(s.elem, 0), count: num(s.count, 0), ty: str(s.ty, ""), role: str(s.role, ""), group: str(s.group, ""),
              doc: str(s.doc, ""), about: str(s.about, ""), missing: false, touches: [], uses: [], edges: [], flags: [],
              record: num(s.record, 1),                        // elements per record (1: plain elements)
              content: isObj(s.content) ? s.content : null,   // an index section: what kind of sequence its words are
              computable: isObj(s.content) ? num(s.computable, 0) : null,   // the share of its words computable from the position
              formula: str(s.formula, ""),                     // what computes them: "27 i", "(?, 3, 3348 + 3 r)"
              fields: arr(s.fields).map(String),               // a record section: its fields' formulas ("?": none)
              copies: [],                                      // { field, n, others: [{ S, field, times }] }: the same words elsewhere
              sample: sampleOf(s.sample),                      // and the first and last of them, or null
              tot: { r: 0, w: 0, readers: 0, writers: 0, rUnk: false, wUnk: false, unk: false }, raw: s };
    S.qname = B.name + "." + S.name;
    S.label = S.qname;
    B.sections.push(S);
    B.secByI[i] = S;
    M.secByKey[S.key] = S;
    M.sections.push(S);
    return S;
  }

  function finishBlock(B) {
    B.order = B.sections.slice().sort(function (a, b) { return a.at - b.at; });
    var used = 0, end = 0;
    B.order.forEach(function (S) {
      used += S.bytes;
      end = Math.max(end, S.at + S.bytes);
    });
    B.used = used;
    if (!B.bytes) B.bytes = end;
  }

  function indexBlocks(M) {
    M.blocks = [];
    M.blockById = Object.create(null);
    M.sections = [];
    M.secByKey = Object.create(null);
    arr(M.c.blocks).forEach(function (b, bi) {
      if (!isObj(b)) return;
      var B = makeBlock(M, b, bi), end = 0;
      arr(b.sections).forEach(function (s, si) {
        if (!isObj(s)) return;
        var S = makeSection(M, B, s, si);
        if (S.at === null) S.at = end;      // no offset given: packed after the previous one
        end = Math.max(end, S.at + S.bytes);
      });
      finishBlock(B);
    });
    // the short label is the bare name where that is unique in the case
    var n = Object.create(null);
    M.sections.forEach(function (S) { n[S.name] = (n[S.name] || 0) + 1; });
    M.sections.forEach(function (S) { S.label = n[S.name] > 1 ? S.qname : S.name; });
  }

  // The section a step names; one that is not in blocks becomes a marked stub,
  // so the reference stays visible instead of breaking the views.
  // The case's copy groups: index sequences (a section's words, or one
  // field of its records) holding the same words or a multiple of them.
  // Each member section gets { field, n, others: [{ S, field, times }] },
  // times saying what its words are of the other's ("", "9 x", "1/9 x").
  // M.index sums the index words: all, computable from the position, and
  // those a copy group stores more than once.
  function indexCopies(M) {
    M.index = { sections: 0, words: 0, fromPos: 0, repeated: 0, groups: 0 };
    M.sections.forEach(function (S) {
      if (!S.content) return;
      M.index.sections++;
      M.index.words += num(S.content.n, S.count);
      M.index.fromPos += num(S.content.n, S.count) * S.computable;
    });
    arr(M.c.copies).forEach(function (g) {
      if (!isObj(g)) return;
      var members = arr(g.members).map(function (m) {
        var B = isObj(m) ? M.blockById[m.block] : null, S = B && has(B.secByI, m.section) ? B.secByI[m.section] : null;
        return S ? { S: S, field: num(m.field, -1), scale: num(m.scale, 1) } : null;
      }).filter(function (m) { return m; });
      if (members.length < 2) return;
      M.index.groups++;
      M.index.repeated += num(g.n, 0) * (members.length - 1);
      members.forEach(function (m) {
        m.S.copies.push({ field: m.field, n: num(g.n, 0), others: members.filter(function (o) { return o !== m; }).map(function (o) {
          var times = m.scale === o.scale ? "" : m.scale % o.scale === 0 ? (m.scale / o.scale) + " x " :
            o.scale % m.scale === 0 ? "1/" + (o.scale / m.scale) + " x " : m.scale + "/" + o.scale + " x ";
          return { S: o.S, field: o.field, times: times };
        }) });
      });
    });
  }

  function secOrStub(M, blockId, secI) {
    var B = M.blockById[blockId];
    if (!B) {
      B = makeBlock(M, { id: blockId }, M.blocks.length);
      B.missing = true;
    }
    var S = B.secByI[secI];
    if (!S) {
      S = makeSection(M, B, { i: secI, at: 0, bytes: 0 }, B.sections.length);
      S.missing = true;
      B.order.push(S);
    }
    return S;
  }

  // The section of a block that holds byte offset at, or null.
  function sectionAt(M, blockId, at) {
    var B = M.blockById[blockId];
    if (!B || at === null) return null;
    for (var i = 0; i < B.order.length; i++) {
      var S = B.order[i];
      if (at >= S.at && at < S.at + S.bytes) return S;
    }
    return null;
  }

  // ------------------------------------------------------------------
  // Phases and steps
  // ------------------------------------------------------------------

  function dirOf(d) {
    if (!isObj(d)) return null;
    return { n: num(d.n, null), bytes: num(d.bytes, null), unique: num(d.unique, null), lo: num(d.lo, null), hi: num(d.hi, null),
             threads: num(d.threads, null), sizes: arr(d.sizes), seq: num(d.seq, null), same: num(d.same, null),
             pattern: str(d.pattern, "unknown"),
             borrowed: !!d.borrowed,   // the pattern is the probed case's (meta.patterns_from), not this case's
             unseen: !!d.unseen };     // the probed case's launches of this kernel never did this
  }

  function unknownDir() {
    return { n: null, bytes: null, unique: null, lo: null, hi: null, threads: null, sizes: [], seq: null, same: null, pattern: "unknown",
             borrowed: false, unseen: false };
  }

  function untyped(ty) { return !ty || ty === "bytes" || ty === "u8" || ty === "void"; }

  // How a parameter and the section it is bound to disagree.
  function argMismatch(Q, S) {
    if (!Q || S.missing) return "";
    if (untyped(Q.type)) return untyped(S.ty) ? "" : "untyped";
    if (untyped(S.ty)) return "";
    if (Q.type !== S.ty) return "type";
    if (Q.role && S.role && Q.role !== S.role) return "role";
    return "";
  }

  function addAccess(M, T, A) {
    for (var i = 0; i < T.acc.length; i++) {
      if (T.acc[i].S === A.S) {            // one entry per section: merge a repeat
        var old = T.acc[i];
        if (!old.r && A.r) { old.r = A.r; countDir(T, A.r, "r"); }
        if (!old.w && A.w) { old.w = A.w; countDir(T, A.w, "w"); }
        old.maybeR = old.maybeR || A.maybeR;
        return;
      }
    }
    T.acc.push(A);
    A.S.touches.push(A);
    if (A.r) countDir(T, A.r, "r");
    if (A.w) countDir(T, A.w, "w");
  }

  function countDir(T, d, which) {
    if (which === "r") {
      T.rN++;
      if (d.bytes === null) T.rUnk++; else T.rBytes += d.bytes;
    } else {
      T.wN++;
      if (d.bytes === null) T.wUnk++; else T.wBytes += d.bytes;
    }
  }

  // One launch's value of the argument G: a scalar's text, or for a pointer
  // (a byte offset in G's block, or [block, offset]) its section S and the
  // offset `off` in it; `at` the offset in the block.
  function runValue(M, G, x) {
    var out = { value: null, S: null, at: null, off: null };
    if (!given(x)) return out;
    if (G && G.S || Array.isArray(x)) {
      var block = Array.isArray(x) ? x[0] : G.S.block.id;
      out.at = num(Array.isArray(x) ? x[1] : x, null);
      out.S = sectionAt(M, block, out.at);
      if (out.S) out.off = out.at - out.S.at;
    } else {
      out.value = String(x);
    }
    return out;
  }

  function initLaunch(M, T, t) {
    T.kname = str(t.kernel, "");
    if (T.kname) {
      T.kernel = M.kernels[T.kname] || makeKernel(M, T.kname);
      T.kernel.steps.push(T);
      if (T.kernel.order < 0) T.kernel.order = T.idx;
    }
    T.launches = num(t.launches, 1);
    T.grid = arr(t.grid);
    T.threads = arr(t.threads);
    T.shared = num(t.shared, null);
    T.geoms = num(t.geoms, 1);
    T.lane = str(t.lane, "");
    T.graph = t.graph === true;
    T.interleaved = t.interleaved === true;
    T.label = T.kname || T.kind;

    arr(t.args).forEach(function (a) {
      if (!isObj(a)) return;
      var Q = T.kernel && given(a.param) ? (T.kernel.paramByName[a.param] || null) : null;
      var G = { param: str(a.param, "?"), def: Q, S: null, at: num(a.at, null), off: null,
                value: given(a.value) ? String(a.value) : null, mismatch: "" };
      if (given(a.block)) {
        G.S = given(a.section) ? secOrStub(M, a.block, a.section) : sectionAt(M, a.block, G.at);
        if (G.S) {
          if (G.at !== null) G.off = G.at - G.S.at;
          G.mismatch = argMismatch(Q, G.S);
          G.S.uses.push({ T: T, G: G });
        }
      }
      T.args.push(G);
    });

    // A step of several launches: each one. vary: the arguments (G) that
    // differ between them; a launch's v: their values, a pointer's as the
    // section it lands in and its offset there.
    if (isObj(t.runs) && Array.isArray(t.runs.list) && t.runs.list.length) {
      var vary = arr(t.runs.vary).map(function (i) { return T.args[i] || null; });
      T.runs = { vary: vary, list: t.runs.list.filter(isObj).map(function (r) {
        return { n: num(r.n, null), grid: arr(r.grid), threads: arr(r.threads), time: num(r.time_us, null),
                 v: vary.map(function (G, j) { return runValue(M, G, arr(r.v)[j]); }) };
      }) };
    }

    if (Array.isArray(t.access)) {
      t.access.forEach(function (a) {
        if (!isObj(a) || !given(a.block)) return;
        var A = { T: T, S: secOrStub(M, a.block, given(a.section) ? a.section : "?"), src: str(a.src, "measured"),
                  r: dirOf(a.r), w: dirOf(a.w), host: false, derived: false, maybeR: false };
        if (!A.r && !A.w) return;
        if (A.src === "measured") M.hasMeasured = true;
        [A.r, A.w].forEach(function (d) {
          if (d && d.pattern === "unknown") M.hasUnknown = true;
          if (d && d.borrowed) M.hasBorrowed = true;
        });
        addAccess(M, T, A);
      });
    } else if (T.kind === "launch") {
      // No access data: what the arguments say. A const pointer is read, another
      // pointer is written (and may be read), an undeclared one is both.
      var any = false;
      T.args.forEach(function (G) {
        if (!G.S) return;
        if (G.def && !G.def.ptr) return;
        var ro = !!G.def && G.def.isConst, known = !!G.def;
        addAccess(M, T, { T: T, S: G.S, src: "args", r: ro || !known ? unknownDir() : null, w: ro ? null : unknownDir(),
                          host: false, derived: true, maybeR: !ro });
        any = true;
      });
      if (any) {
        M.nDerived++;
        M.hasUnknown = true;
      }
    }
    if (T.acc.length) M.hasAccess = true;
  }

  function initTransfer(M, T, t) {
    var up = T.kind === "upload";
    T.bytes = num(t.bytes, 0);
    T.at = num(t.at, null);
    T.via = str(t.via, "");
    if (given(t.block)) {
      T.S = given(t.section) ? secOrStub(M, t.block, t.section) : (sectionAt(M, t.block, T.at) || secOrStub(M, t.block, "?"));
      // the copy is a byte range of the block: every section it overlaps is touched
      var list = [], B = T.S.block;
      if (T.at !== null && T.bytes > 0 && !T.S.missing) {
        var lo = T.at, hi = T.at + T.bytes;
        B.order.forEach(function (S) {
          var a = Math.max(lo, S.at), b = Math.min(hi, S.at + S.bytes);
          if (b > a) list.push({ S: S, bytes: b - a, lo: a - S.at, hi: b - 1 - S.at });
        });
      }
      // sections the case lists for the copy that the byte range did not give
      arr(t.sections).forEach(function (si) {
        var S = B.secByI[si];
        if (!S || list.some(function (x) { return x.S === S; })) return;
        list.push({ S: S, bytes: list.length ? null : T.bytes, lo: null, hi: null });
      });
      if (!list.length) list.push({ S: T.S, bytes: T.bytes, lo: null, hi: null });
      list.forEach(function (x) {
        var d = unknownDir();
        d.bytes = x.bytes;
        d.lo = x.lo;
        d.hi = x.hi;
        d.pattern = up ? "upload" : "download";
        var A = { T: T, S: x.S, src: "host", r: up ? null : d, w: up ? d : null, host: true, derived: false, maybeR: false };
        T.acc.push(A);
        x.S.touches.push(A);
      });
    }
    // what the copy covers, in words: one section, or a run of them
    var nsec = T.acc.length;
    T.target = !T.S ? "?" : nsec > 1 ? T.S.block.name + " (" + nsec + " sections)" : T.S.qname;
    T.label = T.kind + " " + (!T.S ? "?" : nsec > 1 ? T.target : T.S.label);
  }

  function makeStep(M, P, t, ti) {
    var id = str(t.id, P.id + "#" + ti);
    if (M.stepById[id]) id = id + "~" + M.steps.length;       // a repeated id stays distinct
    var scope = arr(t.scope).map(String);
    if (!scope.length) scope = [P.id];
    var T = { id: id, kind: str(t.kind, "launch"), raw: t, phase: P, idx: M.steps.length, pidx: ti,
              scope: scope, scopeKey: scope.join("/"), label: "", flags: [], edgesIn: [], edgesOut: [], acc: [],
              time: num(t.time_us, null), n: num(t.n, null),
              kernel: null, kname: "", launches: 1, grid: [], threads: [], shared: null, geoms: 1, lane: "", graph: false,
              interleaved: false, args: [], runs: null,
              rBytes: 0, wBytes: 0, rN: 0, wN: 0, rUnk: 0, wUnk: 0,
              S: null, bytes: 0, at: null, via: "", target: "" };
    if (T.time !== null) M.hasTime = true;
    M.steps.push(T);
    M.stepById[id] = T;
    if (T.kind === "upload" || T.kind === "download") initTransfer(M, T, t);
    else initLaunch(M, T, t);
    return T;
  }

  function indexPhases(M) {
    M.phases = [];
    M.phaseById = Object.create(null);
    M.steps = [];
    M.stepById = Object.create(null);
    M.hasMeasured = false;
    M.hasAccess = false;
    M.hasUnknown = false;     // some access has no pattern (not measured)
    M.hasBorrowed = false;    // some access has the probed case's pattern (meta.patterns_from)
    M.hasTime = false;
    M.nDerived = 0;
    arr(M.c.phases).forEach(function (p, pi) {
      if (!isObj(p)) return;
      var id = str(p.id, "phase" + pi);
      if (M.phaseById[id]) id = id + "~" + pi;
      var info = M.scopeInfo(id);
      var P = { id: id, title: p.title ? String(p.title) : info.title, doc: p.doc ? String(p.doc) : info.doc,
                times: num(p.times, 1), variants: num(p.variants, 1), steps: [], index: M.phases.length, tot: null, raw: p };
      M.phases.push(P);
      M.phaseById[id] = P;
      arr(p.steps).forEach(function (t, ti) {
        if (!isObj(t)) return;
        P.steps.push(makeStep(M, P, t, ti));
      });
      P.tot = DM.sumSteps(P.steps);
    });
  }

  // ------------------------------------------------------------------
  // Edges, flow
  // ------------------------------------------------------------------

  function indexEdges(M) {
    M.edges = [];
    M.badEdges = [];
    arr(M.c.edges).forEach(function (e) {
      if (!isObj(e)) return;
      var fromHost = e.from === "host", toHost = e.to === "host";
      var a = fromHost || !given(e.from) ? null : (M.stepById[e.from] || null);
      var b = toHost || !given(e.to) ? null : (M.stepById[e.to] || null);
      if ((!fromHost && !a) || (!toHost && !b) || (fromHost && toHost) || !given(e.block)) {
        M.badEdges.push(e);
        return;
      }
      var S = secOrStub(M, e.block, given(e.section) ? e.section : "?");
      var E = { from: a, to: b, S: S, fromHost: fromHost, toHost: toHost };
      M.edges.push(E);
      S.edges.push(E);
      if (a) a.edgesOut.push(E);
      if (b) b.edgesIn.push(E);
    });
  }

  // flow as a tree of { P, id } and { loop, body }; phases flow does not name
  // are appended (loose) so that nothing is hidden.
  function indexFlow(M) {
    var used = Object.create(null);
    function conv(list) {
      var out = [];
      arr(list).forEach(function (f) {
        if (!isObj(f)) return;
        if (given(f.phase)) {
          used[f.phase] = true;
          out.push({ P: M.phaseById[f.phase] || null, id: String(f.phase), loose: false });
        } else if (given(f.loop) || Array.isArray(f.body)) {
          out.push({ loop: str(f.loop, "loop"), body: conv(f.body) });
        }
      });
      return out;
    }
    var flow = conv(M.c.flow), given_ = flow.length > 0;
    M.phases.forEach(function (P) {
      if (!used[P.id]) flow.push({ P: P, id: P.id, loose: given_ });
    });
    M.flow = flow;
  }

  // ------------------------------------------------------------------
  // Totals
  // ------------------------------------------------------------------

  function patternList(set) {
    var out = DM.PAT_ORDER.filter(function (p) { return set[p]; });
    Object.keys(set).forEach(function (p) { if (out.indexOf(p) < 0) out.push(p); });
    return out;
  }

  function aggregate(M) {
    M.kernelList.forEach(function (K) {
      var byPhase = Object.create(null), pr = Object.create(null), pw = Object.create(null);
      K.steps.forEach(function (T) {
        var P = T.phase, e = byPhase[P.id];
        if (!e) {
          e = byPhase[P.id] = { P: P, launches: 0, steps: 0, r: 0, w: 0 };
          K.phases.push(e);
        }
        e.launches += T.launches;
        e.steps++;
        e.r += T.rBytes;
        e.w += T.wBytes;
        K.tot.occ += T.launches;
        K.tot.launches += T.launches * P.times;
        K.tot.r += T.rBytes * P.times;
        K.tot.w += T.wBytes * P.times;
        if (T.rUnk) K.tot.rUnk = true;
        if (T.wUnk) K.tot.wUnk = true;
        if (T.rUnk || T.wUnk) K.tot.unk = true;
        if (T.time !== null) {
          K.tot.time += T.time * T.launches * P.times;
          K.tot.hasTime = true;
        }
        T.acc.forEach(function (A) {
          if (A.r) pr[A.r.pattern] = true;
          if (A.w) pw[A.w.pattern] = true;
        });
      });
      K.patR = patternList(pr);
      K.patW = patternList(pw);
    });

    M.sections.forEach(function (S) {
      var t = S.tot;
      S.touches.forEach(function (A) {
        var times = A.T.phase.times;
        if (A.r) {
          t.readers++;
          if (A.r.bytes !== null) t.r += A.r.bytes * times; else t.rUnk = t.unk = true;
        }
        if (A.w) {
          t.writers++;
          if (A.w.bytes !== null) t.w += A.w.bytes * times; else t.wUnk = t.unk = true;
        }
      });
    });

    // the whole solve: each phase's occurrence times how often it ran
    var tot = M.tot = { steps: M.steps.length, launches: 0, r: 0, w: 0, up: 0, down: 0, waits: 0, time: 0, hasTime: false,
                        rUnk: false, wUnk: false, unk: false };
    M.phases.forEach(function (P) {
      var t = P.tot;
      tot.launches += t.launches * P.times;
      tot.r += t.r * P.times;
      tot.w += t.w * P.times;
      tot.up += t.up * P.times;
      tot.down += t.down * P.times;
      tot.waits += t.nDown * P.times;
      if (t.rUnk) tot.rUnk = true;
      if (t.wUnk) tot.wUnk = true;
      if (t.unk) tot.unk = true;
      if (t.hasTime) {
        tot.time += t.time * P.times;
        tot.hasTime = true;
      }
    });
  }

  // ------------------------------------------------------------------
  // Findings (from the case) and checks (computed here)
  // ------------------------------------------------------------------

  function refSel(r) {
    if (!isObj(r)) return null;
    if (r.kind === "section") return "section:" + r.block + ":" + r.section;
    if (r.kind === "kernel") return "kernel:" + r.name;
    if (r.kind === "step") return "step:" + r.id;
    if (r.kind === "block") return "block:" + (given(r.id) ? r.id : r.block);
    if (r.kind === "unit") return "unit:" + r.id;
    if (r.kind === "scope" || r.kind === "phase") return "scope:" + (given(r.path) ? r.path : r.id);
    return null;
  }

  function indexFindings(M) {
    M.findings = [];
    arr(M.c.findings).forEach(function (f) {
      if (!isObj(f)) return;
      var refs = [];
      arr(f.refs).forEach(function (r) {
        var s = refSel(r);
        if (s) refs.push(s);
      });
      M.findings.push({ level: has(DM.LEVEL_RANK, f.level) ? f.level : "info", title: str(f.title, ""), text: str(f.text, ""), refs: refs });
    });
  }

  function stepName(T) { return T.label + " (" + T.id + ", " + T.phase.id + ")"; }

  function runChecks(M) {
    var list = M.checks = [];
    function add(code, text, refs, owners, level) {
      var item = { code: code, level: level || DM.CHECKS[code].level, title: DM.CHECKS[code].title, text: text, refs: refs };
      list.push(item);
      owners.forEach(function (o) {
        if (o && o.flags.indexOf(item) < 0) o.flags.push(item);
      });
    }
    function sS(S) { return "section:" + S.key; }
    function sT(T) { return "step:" + T.id; }
    function sK(K) { return "kernel:" + K.name; }

    M.sections.forEach(function (S) {
      if (S.missing) {
        add("missing", S.qname + " is named by a step, an argument or an edge but is not in blocks", [sS(S)], [S]);
        return;
      }
      var B = S.block;

      // layout
      var size = has(DM.TYPE_SIZE, S.ty) ? DM.TYPE_SIZE[S.ty] : 0;
      if (size && S.elem > size) {
        add("narrow-in-wide", S.qname + ": " + S.elem + "-byte elements hold " + S.ty + " (" + size + " bytes)" +
          (S.role ? ", role " + S.role : ""), [sS(S)], [S]);
      }
      if (size && S.elem > 0 && S.elem < size) {
        add("elem-too-small", S.qname + ": " + S.elem + "-byte elements cannot hold " + S.ty + " (" + size + " bytes)", [sS(S)], [S]);
      }
      if (S.elem > 0 && S.count > 0 && S.elem * S.count !== S.bytes) {
        add("size", S.qname + ": " + DM.fmtInt(S.bytes) + " bytes, but " + DM.fmtInt(S.count) + " x " + S.elem + " = " +
          DM.fmtInt(S.elem * S.count) + " (" + DM.fmtInt(S.bytes - S.elem * S.count) + " bytes apart)", [sS(S)], [S]);
      }
      if (!B.missing && B.bytes > 0 && S.at + S.bytes > B.bytes) {
        add("out-of-block", S.qname + " ends at " + DM.fmtInt(S.at + S.bytes) + ", the block has " + DM.fmtInt(B.bytes) + " bytes", [sS(S)], [S]);
      }

      // access widths
      var odd = [], oddSizes = [];
      S.touches.forEach(function (A) {
        if (A.host || !S.elem) return;
        [A.r, A.w].forEach(function (d) {
          if (!d) return;
          var bad = d.sizes.filter(function (z) { return z !== S.elem; });
          if (!bad.length) return;
          if (odd.indexOf(A.T) < 0) odd.push(A.T);
          bad.forEach(function (z) { if (oddSizes.indexOf(z) < 0) oddSizes.push(z); });
        });
      });
      if (odd.length) {
        add("width-mismatch", S.qname + " has " + S.elem + "-byte elements (" + (S.ty || "?") + "); accessed in " + oddSizes.join(", ") +
          "-byte units by " + odd.slice(0, 3).map(stepName).join(", ") + (odd.length > 3 ? " and " + (odd.length - 3) + " more" : ""),
          [sS(S)].concat(odd.slice(0, 4).map(sT)), [S]);
      }

      // parameters bound to the section
      var mm = Object.create(null), mmOrder = [];
      S.uses.forEach(function (u) {
        if (u.G.mismatch !== "type" && u.G.mismatch !== "role") return;
        var key = u.G.mismatch + "|" + u.T.kname + "|" + u.G.param, e = mm[key];
        if (!e) {
          e = mm[key] = { kind: u.G.mismatch, G: u.G, steps: [] };
          mmOrder.push(e);
        }
        e.steps.push(u.T);
      });
      mmOrder.forEach(function (e) {
        var Q = e.G.def, T0 = e.steps[0], K = T0.kernel;
        var who = T0.kname + "." + Q.name + " is " + Q.ctype;
        var refs = [sS(S)].concat(K ? [sK(K)] : []).concat([sT(T0)]);
        if (e.kind === "type") {
          add("type-mismatch", who + " (" + Q.type + "); " + S.qname + " stores " + S.ty + (S.role ? " (role " + S.role + ")" : "") +
            (e.steps.length > 1 ? "; " + e.steps.length + " steps" : ""), refs, [S].concat(K ? [K] : []).concat(e.steps));
        } else {
          add("role-mismatch", who + " (role " + Q.role + "); " + S.qname + " has role " + S.role + "; both are " + S.ty, refs, [S]);
        }
      });

      // reads and writes over the solve, in execution order. Only with measured
      // data: static directions do not see what is read through a base pointer.
      if (!M.hasMeasured) return;
      if (!S.touches.length) {
        add("untouched", S.qname + (S.uses.length ? ": passed to " + S.uses.length + " launch arguments, but no access is recorded"
          : ": no step touches it"), [sS(S)], [S]);
        return;
      }
      var hasR = false, firstWrite = null, firstRead = null;
      S.touches.forEach(function (A) {
        if (A.r || A.maybeR) hasR = true;
        if (A.r && !A.w && !firstWrite && !firstRead) firstRead = A;
        if (A.w && !firstWrite) firstWrite = A;
      });
      if (!hasR && firstWrite) {
        // work a kernel does for nothing is a warning; an upload nobody reads is a note
        var byKernel = S.touches.some(function (A) { return !!A.w && !A.host; });
        add("never-read", S.qname + ": written by " + stepName(firstWrite.T) + (S.tot.writers > 1 ? " and " + (S.tot.writers - 1) + " more" : "") +
          ", never read or downloaded", [sS(S), sT(firstWrite.T)], [S], byKernel ? "warn" : "info");
      }
      if (!firstWrite) {
        var rd = firstRead || S.touches[0];
        add("never-written", S.qname + ": read by " + stepName(rd.T) + (S.tot.readers > 1 ? " and " + (S.tot.readers - 1) + " more" : "") +
          ", never written or uploaded", [sS(S), sT(rd.T)], [S]);
      } else if (firstRead) {
        add("read-before-write", S.qname + ": " + stepName(firstRead.T) + " reads it before the first write, by " + stepName(firstWrite.T),
          [sS(S), sT(firstRead.T), sT(firstWrite.T)], [S]);
      }
    });

    // sections that overlap the next one in their block
    M.blocks.forEach(function (B) {
      for (var i = 0; i + 1 < B.order.length; i++) {
        var a = B.order[i], b = B.order[i + 1];
        if (a.missing || b.missing || a.bytes <= 0) continue;
        if (b.at < a.at + a.bytes) {
          add("overlap", a.qname + " (" + DM.fmtInt(a.at) + " + " + DM.fmtInt(a.bytes) + ") runs into " + b.qname + " at " + DM.fmtInt(b.at),
            [sS(a), sS(b)], [a, b]);
        }
      }
    });

    // one section through several parameters of a launch
    M.steps.forEach(function (T) {
      if (T.args.length < 2) return;
      var by = Object.create(null), order = [];
      T.args.forEach(function (G) {
        if (!G.S) return;
        var e = by[G.S.key];
        if (!e) {
          e = by[G.S.key] = [];
          order.push(e);
        }
        e.push(G);
      });
      order.forEach(function (gs) {
        if (gs.length < 2) return;
        var S = gs[0].S;
        var writable = gs.some(function (G) { return !!G.def && G.def.ptr && !G.def.isConst; });
        var names = gs.map(function (G) { return G.param + (G.off ? "+" + G.off : ""); });
        add("alias", stepName(T) + ": " + names.join(", ") + " point at " + S.qname + (writable ? "; one of them is writable" : ""),
          [sT(T), sS(S)].concat(T.kernel ? [sK(T.kernel)] : []), [T], writable ? "warn" : "info");
      });
    });

    M.phases.forEach(function (P) {
      if (P.variants > 1) {
        add("variants", P.title + ": " + P.variants + " distinct step sequences in " + DM.fmtInt(P.times) +
          " occurrences; the steps shown are the most frequent one's", ["scope:" + P.id], []);
      }
    });
    if (M.badEdges.length) {
      add("bad-edge", M.badEdges.length + " edges name a step or a section that is not in this case", [], []);
    }
    var nLaunch = M.steps.filter(function (T) { return T.kind === "launch"; }).length;
    if (nLaunch && !M.hasMeasured) {
      add("no-access", (M.hasAccess ? "No measured access data (directions only)" : "No access data") +
        ": the never-read, never-written, read-before-write and access-width checks were skipped", [], []);
    }
    if (M.nDerived) {
      add("no-access", M.nDerived + " launches have no access data; their reads and writes are taken from their arguments " +
        "(const pointer = read, other pointer = write)", [], []);
    }

    list.forEach(function (item, i) { item.seq = i; });
    list.sort(function (a, b) {
      return DM.LEVEL_RANK[a.level] - DM.LEVEL_RANK[b.level] ||
        DM.CHECK_ORDER.indexOf(a.code) - DM.CHECK_ORDER.indexOf(b.code) || a.seq - b.seq;
    });
  }

  // ------------------------------------------------------------------
  // Search index: kernels, sections, blocks, then units, scopes, steps.
  // ------------------------------------------------------------------

  function buildSearch(M) {
    var out = M.search = [];
    function add(kind, label, sub, sel, rank) {
      out.push({ kind: kind, label: label, sub: sub, sel: sel, text: label.toLowerCase(), rank: rank });
    }
    M.kernelList.forEach(function (K) { add("kernel", K.name, K.unit ? K.unit.name : "", "kernel:" + K.name, 0); });
    M.blocks.forEach(function (B) { add("block", B.name, B.owner, "block:" + B.id, 0); });
    M.sections.forEach(function (S) {
      add("section", S.qname, (S.ty || "?") + (S.role ? " " + S.role : "") + ", " + DM.fmtBytes(S.bytes), "section:" + S.key, 1);
    });
    M.units.forEach(function (U) { add("unit", U.name, U.file, "unit:" + U.id, 4); });
    var seen = Object.create(null);
    M.steps.forEach(function (T) {
      for (var i = 1; i <= T.scope.length; i++) {
        var path = T.scope.slice(0, i).join("/");
        if (seen[path]) continue;
        seen[path] = true;
        add("scope", path, M.scopeInfo(path).title, "scope:" + path, 4);
      }
    });
    M.steps.forEach(function (T) { add("step", T.id + " " + T.label, T.scopeKey, "step:" + T.id, 6); });
  }
})();
