// devmap view 2: Pipeline. The steps of a phase (or of all phases) as nodes
// in one column, top to bottom in execution order, inside collapsible boxes
// that follow the scope path. Dataflow edges run in lanes to the right of
// the column; host transfers come from the host rail on the left.
//
// Layout (all x positions are fixed by nesting depth, y grows downwards):
//
//   | host | host edge labels |  scope boxes around the node column  | edge labels | lanes ...
//   0   HOST_W            X0 = HOST_W + HOST_ZONE              portX   labelX      laneX0
//
// A step, or a collapsed scope, is an "item". Edges are grouped into nets:
// one net per (producing item, section) with one tap per consuming item.
// A net gets one lane; an edge whose other end is not shown becomes a stub.
"use strict";
(function () {
  var DM = window.DM, h = DM.h, s = DM.s;

  var C = {
    HOST_W: 46,       // the host rail
    HOST_ZONE: 150,   // between the rail and the scope boxes: host edge labels
    INDENT: 12,       // a scope box's inset per nesting level
    NODE_W: 250,
    NODE_H: 38,
    PORT_H: 13,       // vertical room of one edge port
    NODE_GAP: 9,
    GROUP_HEAD: 22,
    GROUP_PAD: 8,
    LABEL_W: 180,     // right of the boxes: the edge labels
    LANE_W: 11,
    TOP: 30
  };

  // What survives a redraw: collapsed scopes (per case), pan and zoom.
  var ui = { caseId: null, key: "", collapsed: Object.create(null), k: 1, tx: 16, ty: 8, edges: true, fresh: true };
  var cur = null;                 // the drawing: { M, wrap, tree, L, svg, vp, hover }
  var drag = null, dragMoved = false, winHooked = false;

  // ------------------------------------------------------------------
  // The scope tree
  // ------------------------------------------------------------------

  // Steps grouped by scope path, in execution order. A scope that is left
  // and entered again is a second group.
  function buildTree(M, phases) {
    var root = { kind: "root", children: [], groups: [], stepNode: Object.create(null), nSteps: 0 };
    var occ = Object.create(null);
    phases.forEach(function (P) {
      var stack = [root], path = [];
      P.steps.forEach(function (T) {
        var p = T.scope, common = 0;
        while (common < path.length && common < p.length && path[common] === p[common]) common++;
        stack.length = common + 1;
        path.length = common;
        for (var i = common; i < p.length; i++) {
          var key = p.slice(0, i + 1).join("/");
          var n = occ[key] || 0;
          occ[key] = n + 1;
          var info = M.scopeInfo(key);
          var g = { kind: "group", gid: key + "#" + n, path: key, depth: i, title: info.title, doc: info.doc, times: null,
                    children: [], steps: [], parent: stack[stack.length - 1], item: null, el: null, x: 0, y: 0, w: 0, h: 0 };
          if (i === 0 && p[0] === P.id) {
            g.title = P.title;
            g.doc = P.doc;
            g.times = P.times;
          }
          g.parent.children.push(g);
          root.groups.push(g);
          stack.push(g);
          path.push(p[i]);
        }
        var top = stack[stack.length - 1];
        var node = { kind: "step", T: T, parent: top, item: null };
        top.children.push(node);
        root.stepNode[T.id] = node;
        for (var a = top; a && a.kind === "group"; a = a.parent) a.steps.push(T);
        root.nSteps++;
      });
    });
    return root;
  }

  // ------------------------------------------------------------------
  // Layout
  // ------------------------------------------------------------------

  function newItem(L, kind, node) {
    var it = { uid: L.items.length, kind: kind, node: node, T: null, g: null, rin: [], rout: [], lin: [], lout: [], right: [], left: [],
               keys: Object.create(null), x: 0, y: 0, w: C.NODE_W, h: C.NODE_H, el: null };
    L.items.push(it);
    return it;
  }

  function portRank(p) {
    return p.kind === "xin" ? -1 : p.net.from.uid;
  }

  function layout(M, tree) {
    var L = { items: [], groups: [], itemOf: Object.create(null), nets: [], edgeEls: [], maxD: -1,
              nodeX: 0, portX: 0, labelX: 0, laneX0: 0, nLanes: 0, width: 0, height: 0 };

    // 1. what is shown: steps and collapsed scopes are items, open scopes are boxes
    (function walk(node) {
      node.children.forEach(function (ch) {
        var it;
        if (ch.kind === "step") {
          it = newItem(L, "step", ch);
          it.T = ch.T;
          ch.item = it;
          L.itemOf[ch.T.id] = it;
        } else if (ui.collapsed[ch.gid]) {
          it = newItem(L, "cgroup", ch);
          it.g = ch;
          ch.item = it;
          ch.steps.forEach(function (T) { L.itemOf[T.id] = it; });
        } else {
          ch.item = null;
          L.groups.push(ch);
          if (ch.depth > L.maxD) L.maxD = ch.depth;
          walk(ch);
        }
      });
    })(tree);

    // 2. edges -> nets and ports
    var netBy = Object.create(null);
    function netOf(a, S) {
      var k = a.uid + "|" + S.key, n = netBy[k];
      if (!n) {
        n = netBy[k] = { from: a, S: S, taps: [], ext: [], port: null, lane: 0, lo: 0, hi: 0 };
        n.port = { kind: "out", S: S, net: n, item: a, y: 0 };
        a.rout.push(n.port);
        L.nets.push(n);
      }
      return n;
    }
    M.edges.forEach(function (E) {
      var a = E.from ? L.itemOf[E.from.id] : null, b = E.to ? L.itemOf[E.to.id] : null, k, p, n;
      if (E.fromHost) {
        if (!b) return;
        k = "hi|" + E.S.key;
        if (!b.keys[k]) {
          b.keys[k] = true;
          b.lin.push({ kind: "hostin", S: E.S, item: b, y: 0 });
        }
      } else if (E.toHost) {
        if (!a) return;
        k = "ho|" + E.S.key;
        if (!a.keys[k]) {
          a.keys[k] = true;
          a.lout.push({ kind: "hostout", S: E.S, item: a, y: 0 });
        }
      } else if (a && b) {
        if (a === b) return;                    // inside one collapsed scope
        n = netOf(a, E.S);
        k = "in|" + a.uid + "|" + E.S.key;
        if (!b.keys[k]) {
          b.keys[k] = true;
          p = { kind: "in", S: E.S, net: n, item: b, y: 0 };
          n.taps.push(p);
          b.rin.push(p);
        }
      } else if (a) {                           // the consumer is not shown
        n = netOf(a, E.S);
        if (n.ext.indexOf(E.to) < 0) n.ext.push(E.to);
      } else if (b) {                           // the producer is not shown
        k = "xin|" + E.S.key;
        p = b.keys[k];
        if (!p) {
          p = b.keys[k] = { kind: "xin", S: E.S, froms: [], item: b, y: 0 };
          b.rin.push(p);
        }
        if (p.froms.indexOf(E.from) < 0) p.froms.push(E.from);
      }
    });

    // 3. item heights: one port slot per edge end, inputs above outputs
    L.items.forEach(function (it) {
      it.rin.sort(function (p, q) { return portRank(p) - portRank(q); });
      it.right = it.rin.concat(it.rout);
      it.left = it.lin.concat(it.lout);
      it.h = Math.max(C.NODE_H, 10 + C.PORT_H * Math.max(it.right.length, it.left.length));
    });

    // 4. positions
    var X0 = C.HOST_W + C.HOST_ZONE, nodeX = X0 + (L.maxD + 1) * C.INDENT, y = C.TOP;
    (function place(node) {
      node.children.forEach(function (ch) {
        var it = ch.item;
        if (it) {
          it.x = nodeX;
          it.y = y;
          y += it.h + C.NODE_GAP;
        } else {
          ch.x = X0 + ch.depth * C.INDENT;
          ch.w = C.NODE_W + 2 * (L.maxD + 1 - ch.depth) * C.INDENT;
          ch.y = y;
          y += C.GROUP_HEAD;
          place(ch);
          y += C.GROUP_PAD - C.NODE_GAP;
          ch.h = y - ch.y;
          y += C.NODE_GAP;
        }
      });
    })(tree);
    L.items.forEach(function (it) {
      [it.right, it.left].forEach(function (list) {
        var y0 = it.y + (it.h - list.length * C.PORT_H) / 2 + C.PORT_H / 2;
        list.forEach(function (p, i) { p.y = Math.round(y0 + i * C.PORT_H); });
      });
    });
    L.nodeX = nodeX;
    L.portX = nodeX + C.NODE_W;
    L.labelX = X0 + C.NODE_W + 2 * (L.maxD + 1) * C.INDENT + 8;
    L.laneX0 = L.labelX + C.LABEL_W;

    // 5. lanes: short nets first, each in the innermost lane that is free
    var lanes = [], wired = L.nets.filter(function (n) { return n.taps.length > 0; });
    wired.forEach(function (n) {
      n.lo = n.hi = n.port.y;
      n.taps.forEach(function (t) {
        if (t.y < n.lo) n.lo = t.y;
        if (t.y > n.hi) n.hi = t.y;
      });
    });
    wired.sort(function (a, b) { return (a.hi - a.lo) - (b.hi - b.lo) || a.lo - b.lo; });
    wired.forEach(function (n) {
      for (var li = 0; ; li++) {
        var iv = lanes[li];
        if (!iv) iv = lanes[li] = [];
        var free = true;
        for (var j = 0; j < iv.length; j++) {
          if (n.hi + 5 >= iv[j][0] && n.lo - 5 <= iv[j][1]) {
            free = false;
            break;
          }
        }
        if (free) {
          iv.push([n.lo, n.hi]);
          n.lane = li;
          break;
        }
      }
    });
    L.nLanes = lanes.length;
    L.width = L.laneX0 + Math.max(1, L.nLanes) * C.LANE_W + 30;
    L.height = y + 24;
    return L;
  }

  // ------------------------------------------------------------------
  // Painting
  // ------------------------------------------------------------------

  function groupTip(g, tot) {
    var lines = [g.title + "   " + g.path + (g.times !== null ? "   x" + DM.fmtInt(g.times) : "")];
    if (g.doc) lines.push(g.doc);
    lines.push(tot.steps + " steps, " + tot.launches + " launches per occurrence");
    lines.push("read " + DM.fmtAmount(tot.r, tot.rUnk) + ", written " + DM.fmtAmount(tot.w, tot.wUnk));
    if (tot.nUp || tot.nDown) lines.push("uploads " + tot.nUp + " (" + DM.fmtBytes(tot.up) + "), downloads " + tot.nDown + " (" + DM.fmtBytes(tot.down) + ")");
    if (tot.hasTime) lines.push("kernel time " + DM.fmtTime(tot.time));
    return lines.join("\n");
  }

  // An open scope: its box and its header (click: collapse).
  function groupBox(g) {
    var tot = DM.sumSteps(g.steps);
    var el = s("g", { cls: "pgroup d" + Math.min(g.depth, 3), transform: "translate(" + g.x + "," + g.y + ")" });
    el.appendChild(s("rect", { cls: "gbox", width: g.w, height: g.h, rx: 5 }));
    var head = s("g", { cls: "ghead", "data-act": "toggle", "data-gid": g.gid, "data-path": g.path, tip: groupTip(g, tot) + "\nclick: collapse" });
    head.appendChild(s("rect", { cls: "ghit", width: g.w, height: C.GROUP_HEAD, rx: 5 }));
    head.appendChild(s("path", { cls: "tri", d: "M8 8 H16 L12 15 Z" }));
    var title = g.title + (g.times !== null ? "  x" + DM.fmtInt(g.times) : "");
    var sum = tot.steps + (tot.steps === 1 ? " step" : " steps") + (tot.launches ? ", " + tot.launches + " launches" : "");
    var room = g.w - 22 - 10, sumW = DM.textW(sum, 10) + 12;
    if (room - sumW < DM.textW(title, 11.5) && room - sumW < 90) {
      sum = "";
      sumW = 0;
    }
    head.appendChild(s("text", { cls: "gtitle", x: 22, y: 15 }, DM.fit(title, room - sumW, 11.5)));
    if (sum) head.appendChild(s("text", { cls: "gsum", x: g.w - 10, y: 15, "text-anchor": "end" }, sum));
    el.appendChild(head);
    g.el = el;
    return el;
  }

  // A collapsed scope, drawn as a node (click: expand).
  function groupNode(it) {
    var g = it.g, w = it.w, tot = DM.sumSteps(g.steps);
    var el = s("g", { cls: "pnode cg", transform: "translate(" + it.x + "," + it.y + ")", "data-act": "toggle", "data-gid": g.gid,
      "data-path": g.path, "data-uid": it.uid, tip: groupTip(g, tot) + "\nclick: expand" });
    el.appendChild(s("rect", { cls: "box", width: w, height: it.h, rx: 4 }));
    el.appendChild(s("path", { cls: "tri", d: "M10 9 L17 13 L10 17 Z" }));
    var mult = g.times !== null ? "x" + DM.fmtInt(g.times) : "";
    var multW = mult ? DM.textW(mult, 12) + 10 : 0;
    el.appendChild(s("text", { cls: "name", x: 23, y: 16 }, DM.fit(g.title, w - 35 - multW, 12)));
    if (mult) el.appendChild(s("text", { cls: "mult", x: w - 12, y: 16, "text-anchor": "end" }, mult));
    var rwW = 0;
    if (!DM.uncounted(tot.r, tot.rUnk, tot.w, tot.wUnk)) {
      var rt = "R " + DM.fmtAmount(tot.r, tot.rUnk), wt = "W " + DM.fmtAmount(tot.w, tot.wUnk);
      rwW = DM.textW(rt + wt, 10) + 14;
      el.appendChild(s("text", { cls: "sub", x: w - 12, y: 30, "text-anchor": "end" }, s("tspan", { cls: "rd" }, rt), s("tspan", { cls: "wr", dx: 8 }, wt)));
    }
    el.appendChild(s("text", { cls: "sub", x: 23, y: 30 }, DM.fit(tot.steps + " steps, " + tot.launches + " launches", w - 35 - rwW, 10)));
    return el;
  }

  // A step: a launch is a box, an upload or download a flag shape.
  function stepNode(it) {
    var T = it.T, w = it.w, hh = it.h;
    var kind = T.kind === "upload" ? "up" : T.kind === "download" ? "down" : "launch";
    var lv = DM.worstLevel(T.flags);
    var el = s("g", { cls: "pnode " + kind + (T.lane === "side" ? " side" : ""), transform: "translate(" + it.x + "," + it.y + ")",
      "data-act": "step", "data-id": T.id, "data-uid": it.uid, tip: DM.stepTip(T) });
    if (kind === "launch") {
      el.appendChild(s("rect", { cls: "box", width: w, height: hh, rx: 4 }));
      var mult = (T.launches > 1 ? "x" + DM.fmtInt(T.launches) : "") + (T.interleaved ? "~" : "");
      var multW = mult ? DM.textW(mult, 12) + 10 : 0;
      el.appendChild(s("text", { cls: "name", x: 10, y: 16 }, DM.fit(T.label, w - 22 - multW, 12)));
      if (mult) el.appendChild(s("text", { cls: "mult", x: w - 12, y: 16, "text-anchor": "end" }, mult));
      var rw = DM.rwShort(T), rwW = 0;
      if (rw && !DM.uncounted(T.rBytes, T.rUnk, T.wBytes, T.wUnk)) {
        var rt = "R " + rw.r, wt = "W " + rw.w;
        rwW = DM.textW(rt + wt, 10) + 14;
        el.appendChild(s("text", { cls: "sub", x: w - 12, y: 30, "text-anchor": "end" }, s("tspan", { cls: "rd" }, rt), s("tspan", { cls: "wr", dx: 8 }, wt)));
      }
      var geo = DM.geomText(T);
      if (geo) el.appendChild(s("text", { cls: "sub", x: 10, y: 30 }, DM.fit(geo, w - 22 - rwW, 10)));
    } else {
      var up = kind === "up", my = hh / 2;
      el.appendChild(s("path", { cls: "box", d: up ? "M0 0 H" + w + " V" + hh + " H0 L9 " + my + " Z" : "M9 0 H" + w + " V" + hh + " H9 L0 " + my + " Z" }));
      el.appendChild(s("text", { cls: "name", x: 17, y: 16 }, DM.fit(T.label, w - 29, 12)));
      el.appendChild(s("text", { cls: "sub", x: 17, y: 30 },
        DM.fit(DM.fmtBytes(T.bytes) + (T.via ? "  " + T.via : "") + (up ? "  from the host" : "  to the host, which waits"), w - 29, 10)));
    }
    if (lv) el.appendChild(s("rect", { cls: "flagmark lv-" + lv, x: w - 5, y: 4, width: 3, height: hh - 8, rx: 1.5 }));
    return el;
  }

  // The thick arrow between the host rail and an upload or download.
  function connector(it) {
    var up = it.T.kind === "upload", y = it.y + it.h / 2, x0 = C.HOST_W, x1 = it.x;
    var g = s("g", { cls: "xfer " + (up ? "up" : "down") });
    g.appendChild(s("path", { cls: "xline", d: "M" + x0 + " " + y + " H" + x1 }));
    g.appendChild(s("path", { cls: "xhead", d: up ? "M" + (x1 + 8) + " " + y + " l-10 -5 v10 Z" : "M" + x0 + " " + y + " l10 -5 v10 Z" }));
    g.appendChild(s("text", { cls: "xlabel", x: x0 + 16, y: y - 5 }, DM.fit(DM.fmtBytes(it.T.bytes) + (up ? "" : "  host waits"), C.HOST_ZONE - 26, 10)));
    return g;
  }

  // producer port (px, y0) -> lane lx -> consumer port (px, y1)
  function wirePath(px, y0, lx, y1) {
    var dy = y1 - y0, sg = dy >= 0 ? 1 : -1, r = Math.min(5, Math.abs(dy) / 2);
    return "M" + (px + 3) + " " + y0 + " H" + (lx - r) + " Q" + lx + " " + y0 + " " + lx + " " + (y0 + sg * r) +
      " V" + (y1 - sg * r) + " Q" + lx + " " + y1 + " " + (lx - r) + " " + y1 + " H" + (px + 8);
  }

  function otherEnd(T) {
    return T.phase.id + ":" + T.label;
  }

  function paintEdges(L, layer) {
    var px = L.portX, fs = 10.5, room = C.LABEL_W - 12;
    function reg(el, from, to, key) {
      el.setAttribute("data-ei", String(L.edgeEls.length));
      L.edgeEls.push({ el: el, from: from, to: to, key: key });
      layer.appendChild(el);
    }

    L.nets.forEach(function (n) {
      var S = n.S, y0 = n.port.y, label = S.label;
      if (n.ext.length === 1) label += " > " + otherEnd(n.ext[0]);
      else if (n.ext.length > 1) label += " > " + n.ext.length + " steps elsewhere";
      var tip = S.qname + "  " + (S.ty || "?") + (S.role ? " (" + S.role + ")" : "") + "\nwritten by " + (n.from.T ? n.from.T.label : n.from.g.title) +
        (n.taps.length ? "\nread by " + n.taps.map(function (t) { return t.item.T ? t.item.T.label : t.item.g.title; }).join(", ") : "") +
        (n.ext.length ? "\nread in other phases by " + n.ext.map(otherEnd).join(", ") : "") + "\nclick: the section's details";
      // one wire per consumer, from the producer's port through the net's lane
      n.taps.forEach(function (t) {
        var lx = L.laneX0 + n.lane * C.LANE_W + C.LANE_W / 2, d = wirePath(px, y0, lx, t.y);
        var e = s("g", { cls: "edge tap", "data-act": "sec", "data-key": S.key, tip: tip });
        e.appendChild(s("path", { cls: "wire", d: d }));
        e.appendChild(s("path", { cls: "hit", d: d }));
        e.appendChild(s("path", { cls: "ahead", d: "M" + (px + 1) + " " + t.y + " l7 -3.5 v7 Z" }));
        e.appendChild(s("text", { cls: "elabel", x: L.labelX, y: t.y + 3.5 }, DM.fit(S.label, room, fs)));
        reg(e, n.from.uid, t.item.uid, S.key);
      });
      // the producer's port: drawn after the wires, so that its label lies over them
      var g = s("g", { cls: "edge port-out" + (n.taps.length ? "" : " stub"), "data-act": "sec", "data-key": S.key, tip: tip });
      if (!n.taps.length) g.appendChild(s("path", { cls: "wire", d: "M" + (px + 3) + " " + y0 + " H" + (L.labelX - 2) }));
      g.appendChild(s("rect", { cls: "pdot", x: px + 1, y: y0 - 2.5, width: 5, height: 5 }));
      g.appendChild(s("text", { cls: "elabel" + (n.ext.length ? " ext" : ""), x: L.labelX, y: y0 + 3.5 }, DM.fit(label, room, fs)));
      reg(g, n.from.uid, -1, S.key);
    });

    L.items.forEach(function (it) {
      // producers that are not shown
      it.right.forEach(function (p) {
        if (p.kind !== "xin") return;
        var who = p.froms.length === 1 ? otherEnd(p.froms[0]) : p.froms.length + " steps elsewhere";
        var e = s("g", { cls: "edge tap stub", "data-act": "sec", "data-key": p.S.key,
          tip: p.S.qname + "  " + (p.S.ty || "?") + "\nwritten in other phases by " + p.froms.map(otherEnd).join(", ") + "\nclick: the section's details" });
        e.appendChild(s("path", { cls: "wire", d: "M" + (px + 8) + " " + p.y + " H" + (L.labelX - 2) }));
        e.appendChild(s("path", { cls: "ahead", d: "M" + (px + 1) + " " + p.y + " l7 -3.5 v7 Z" }));
        e.appendChild(s("text", { cls: "elabel ext", x: L.labelX, y: p.y + 3.5 }, DM.fit(p.S.label + " < " + who, room, fs)));
        reg(e, -1, it.uid, p.S.key);
      });
      // host edges, on the left
      it.left.forEach(function (p) {
        var isIn = p.kind === "hostin", x0 = C.HOST_W, x1 = it.x;
        var e = s("g", { cls: "edge host " + (isIn ? "tap" : "port-out"), "data-act": "sec", "data-key": p.S.key,
          tip: p.S.qname + "  " + (p.S.ty || "?") + (isIn ? "\nfrom the host (an upload)" : "\nto the host (a download)") + "\nclick: the section's details" });
        e.appendChild(s("path", { cls: "wire", d: "M" + x0 + " " + p.y + " H" + (isIn ? x1 - 8 : x1) }));
        e.appendChild(s("path", { cls: "ahead", d: isIn ? "M" + (x1 - 1) + " " + p.y + " l-7 -3.5 v7 Z" : "M" + (x0 + 1) + " " + p.y + " l7 -3.5 v7 Z" }));
        e.appendChild(s("text", { cls: "elabel", x: x0 + 14, y: p.y + 3.5 }, DM.fit(p.S.label, C.HOST_ZONE - 26, fs)));
        reg(e, isIn ? -1 : it.uid, isIn ? it.uid : -1, p.S.key);
      });
    });
  }

  function paint(L) {
    var svg = s("svg", { cls: "pipe" + (ui.edges ? "" : " no-edges") });
    var vp = s("g", { cls: "vp" });
    svg.appendChild(vp);
    vp.appendChild(s("rect", { cls: "rail", x: 0, y: 0, width: C.HOST_W, height: L.height, rx: 4 }));
    vp.appendChild(s("text", { cls: "col-label", x: C.HOST_W / 2, y: 18, "text-anchor": "middle" }, "host"));
    vp.appendChild(s("text", { cls: "col-label", x: C.HOST_W + C.HOST_ZONE, y: 18 }, "device"));
    vp.appendChild(s("text", { cls: "col-label", x: L.labelX, y: 18 }, "dataflow"));
    var gG = s("g", { cls: "groups" }), gX = s("g", { cls: "xfers" }), gE = s("g", { cls: "edges" }), gN = s("g", { cls: "nodes" });
    L.groups.forEach(function (g) { gG.appendChild(groupBox(g)); });
    paintEdges(L, gE);
    L.items.forEach(function (it) {
      it.el = it.kind === "cgroup" ? groupNode(it) : stepNode(it);
      gN.appendChild(it.el);
      if (it.T && (it.T.kind === "upload" || it.T.kind === "download")) gX.appendChild(connector(it));
    });
    vp.appendChild(gG);
    vp.appendChild(gX);
    vp.appendChild(gE);
    vp.appendChild(gN);
    cur.vp = vp;
    return svg;
  }

  // ------------------------------------------------------------------
  // Pan and zoom
  // ------------------------------------------------------------------

  function applyTransform() {
    if (cur && cur.vp) cur.vp.setAttribute("transform", "translate(" + ui.tx + "," + ui.ty + ") scale(" + ui.k + ")");
  }

  function viewRect() {
    return cur && cur.wrap ? cur.wrap.getBoundingClientRect() : { width: 0, height: 0, left: 0, top: 0 };
  }

  // Actual size from the top left; a pipeline wider than the view is shrunk to fit its width.
  function resetView() {
    if (!cur || !cur.L) return;
    var r = viewRect();
    ui.k = 1;
    if (r.width > 100 && cur.L.width > r.width - 32) ui.k = Math.max(0.5, (r.width - 32) / cur.L.width);
    ui.tx = 16;
    ui.ty = 8;
    applyTransform();
  }

  function fitView() {
    if (!cur || !cur.L) return;
    var r = viewRect();
    if (!r.width || !r.height) return;
    var k = Math.min((r.width - 32) / cur.L.width, (r.height - 24) / cur.L.height);
    ui.k = Math.max(0.02, Math.min(2, k));
    ui.tx = Math.max(8, (r.width - cur.L.width * ui.k) / 2);
    ui.ty = Math.max(8, (r.height - cur.L.height * ui.k) / 2);
    applyTransform();
  }

  // Bring a rectangle of the drawing into view if it is not.
  function focusRect(x, y, w, hh, toTop) {
    var r = viewRect();
    if (!r.width || !r.height) return;
    var top = y * ui.k + ui.ty, bottom = (y + hh) * ui.k + ui.ty;
    if (toTop) ui.ty = 40 - y * ui.k;
    else if (top < 30 || bottom > r.height - 30) ui.ty = r.height / 2 - (y + hh / 2) * ui.k;
    var left = x * ui.k + ui.tx, right = (x + w) * ui.k + ui.tx;
    if (left < 0 || right > r.width) ui.tx = 60 - x * ui.k;
    applyTransform();
  }

  function hookWindow() {
    if (winHooked) return;
    winHooked = true;
    DM.on(window, "mousemove", function (e) {
      if (!drag) return;
      if (e.buttons === 0) {
        endDrag();
        return;
      }
      var dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      if (!dragMoved && Math.abs(dx) + Math.abs(dy) < 4) return;
      dragMoved = true;
      ui.tx = drag.tx + dx;
      ui.ty = drag.ty + dy;
      applyTransform();
      if (cur && cur.svg) cur.svg.classList.add("panning");
    });
    DM.on(window, "mouseup", endDrag);
  }

  function endDrag() {
    drag = null;
    if (cur && cur.svg) cur.svg.classList.remove("panning");
  }

  // ------------------------------------------------------------------
  // Hover and selection
  // ------------------------------------------------------------------

  // tag: "i<item uid>", "e<edge index>" or null
  function setHover(tag) {
    if (!cur || !cur.L || !cur.svg || tag === cur.hover) return;
    var L = cur.L, svg = cur.svg;
    if (cur.hover !== null) {
      L.edgeEls.forEach(function (r) { r.el.classList.remove("on-in", "on-out"); });
      L.items.forEach(function (it) { it.el.classList.remove("rel"); });
      svg.classList.remove("hover");
    }
    cur.hover = tag;
    if (tag === null) return;
    var idx = Number(tag.slice(1));
    function rel(uid) {
      if (uid >= 0 && L.items[uid]) L.items[uid].el.classList.add("rel");
    }
    if (tag.charAt(0) === "i") {
      if (!L.items[idx]) return;
      svg.classList.add("hover");
      rel(idx);
      L.edgeEls.forEach(function (r) {
        if (r.to === idx) r.el.classList.add("on-in");
        else if (r.from === idx) r.el.classList.add("on-out");
        else return;
        rel(r.from);
        rel(r.to);
      });
    } else {
      var me = L.edgeEls[idx];
      if (!me) return;
      svg.classList.add("hover");
      L.edgeEls.forEach(function (r) {
        var same = r === me || (me.from >= 0 && r.from === me.from && r.key === me.key);
        if (!same) return;
        r.el.classList.add(r.to >= 0 ? "on-in" : "on-out");
        rel(r.from);
        rel(r.to);
      });
    }
  }

  function mark(res) {
    if (!cur || !cur.L) return;
    var L = cur.L;
    L.items.forEach(function (it) { it.el.classList.remove("sel", "touch"); });
    L.groups.forEach(function (g) { if (g.el) g.el.classList.remove("sel"); });
    L.edgeEls.forEach(function (r) { r.el.classList.remove("sel"); });
    if (!res) return;
    var it;
    if (res.kind === "step" || res.kind === "access") {
      it = L.itemOf[res.T.id];
      if (it) it.el.classList.add("sel");
    }
    if (res.kind === "kernel") {
      res.K.steps.forEach(function (T) {
        var x = L.itemOf[T.id];
        if (x) x.el.classList.add("sel");
      });
    }
    if (res.kind === "section" || res.kind === "access") {
      L.edgeEls.forEach(function (r) { if (r.key === res.S.key) r.el.classList.add("sel"); });
    }
    if (res.kind === "section") {
      res.S.touches.forEach(function (A) {
        var x = L.itemOf[A.T.id];
        if (x) x.el.classList.add("touch");
      });
    }
    if (res.kind === "scope") {
      L.groups.forEach(function (g) { if (g.path === res.path && g.el) g.el.classList.add("sel"); });
      L.items.forEach(function (x) { if (x.g && x.g.path === res.path) x.el.classList.add("sel"); });
    }
  }

  function firstShown(steps) {
    for (var i = 0; i < steps.length; i++) {
      if (cur.tree.stepNode[steps[i].id]) return steps[i];
    }
    return null;
  }

  function select(res, reveal) {
    if (!cur || !cur.L) return;
    var target = null;                  // the step to bring into view
    if (res && reveal) {
      if (res.kind === "step" || res.kind === "access") target = firstShown([res.T]);
      else if (res.kind === "kernel") target = firstShown(res.K.steps);
      else if (res.kind === "section") target = firstShown(res.S.touches.map(function (A) { return A.T; }));
    }
    if (target) {
      // open the scopes that hide it
      var opened = false;
      for (var g = cur.tree.stepNode[target.id].parent; g && g.kind === "group"; g = g.parent) {
        if (ui.collapsed[g.gid]) {
          delete ui.collapsed[g.gid];
          opened = true;
        }
      }
      if (opened) draw();
    }
    mark(res);
    if (!res || !reveal || !cur.L) return;
    if (target) {
      var it = cur.L.itemOf[target.id];
      if (it) focusRect(it.x, it.y, it.w, it.h, false);
    } else if (res.kind === "scope") {
      var done = false;
      cur.L.groups.forEach(function (gr) {
        if (!done && gr.path === res.path) {
          done = true;
          focusRect(gr.x, gr.y, gr.w, gr.h, true);
        }
      });
      cur.L.items.forEach(function (x) {
        if (!done && x.g && x.g.path === res.path) {
          done = true;
          focusRect(x.x, x.y, x.w, x.h, false);
        }
      });
    }
  }

  // ------------------------------------------------------------------
  // Events of one drawing
  // ------------------------------------------------------------------

  function closest(target, selector) {
    var t = target;
    if (t && t.nodeType !== 1) t = t.parentNode;
    return t && t.closest ? t.closest(selector) : null;
  }

  function hookSvg(svg) {
    svg.addEventListener("wheel", DM.guard(function (e) {
      e.preventDefault();
      var dy = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaMode === 2 ? e.deltaY * 400 : e.deltaY;
      var dx = e.deltaMode === 1 ? e.deltaX * 16 : e.deltaMode === 2 ? e.deltaX * 400 : e.deltaX;
      if (e.shiftKey) {                 // shift + wheel scrolls
        ui.ty -= dy + dx;
        applyTransform();
        return;
      }
      var r = viewRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
      var k = Math.max(0.02, Math.min(4, ui.k * Math.exp(-dy * 0.0015))), f = k / ui.k;
      ui.tx = mx - (mx - ui.tx) * f;
      ui.ty = my - (my - ui.ty) * f;
      ui.k = k;
      applyTransform();
    }, "pipeline wheel"), { passive: false });

    DM.on(svg, "mousedown", function (e) {
      if (e.button !== 0) return;
      drag = { x: e.clientX, y: e.clientY, tx: ui.tx, ty: ui.ty };
      dragMoved = false;
    });

    DM.on(svg, "click", function (e) {
      if (dragMoved) {                  // the end of a pan, not a click
        dragMoved = false;
        return;
      }
      var t = closest(e.target, "[data-act]");
      if (!t) return;
      var act = t.getAttribute("data-act");
      if (act === "step") {
        DM.select("step:" + t.getAttribute("data-id"), { reveal: false });
      } else if (act === "sec") {
        DM.select("section:" + t.getAttribute("data-key"), { reveal: false });
      } else if (act === "toggle") {
        var gid = t.getAttribute("data-gid");
        if (ui.collapsed[gid]) delete ui.collapsed[gid];
        else ui.collapsed[gid] = true;
        draw();
        DM.select("scope:" + t.getAttribute("data-path"), { reveal: false });
      }
    });

    DM.on(svg, "mouseover", function (e) {
      var n = closest(e.target, ".pnode");
      if (n) {
        setHover("i" + n.getAttribute("data-uid"));
        return;
      }
      var ed = closest(e.target, ".edge");
      setHover(ed ? "e" + ed.getAttribute("data-ei") : null);
    });
    DM.on(svg, "mouseleave", function () { setHover(null); });
  }

  // ------------------------------------------------------------------
  // Drawing and the view
  // ------------------------------------------------------------------

  function draw() {
    if (!cur) return;
    var M = cur.M, wrap = cur.wrap;
    DM.clear(wrap);
    cur.svg = null;
    cur.vp = null;
    cur.L = null;
    cur.hover = null;
    var tree = cur.tree = buildTree(M, M.phasesFor(DM.state.phase));
    if (!tree.nSteps) {
      wrap.appendChild(h("div", { cls: "empty" }, "No steps to show."));
      return;
    }
    if (ui.fresh && tree.nSteps > 400) {
      // a big pipeline starts with its sub-scopes closed
      tree.groups.forEach(function (g) { if (g.depth >= 1) ui.collapsed[g.gid] = true; });
    }
    cur.L = layout(M, tree);
    cur.svg = paint(cur.L);
    wrap.appendChild(cur.svg);
    hookSvg(cur.svg);
    if (ui.fresh) {
      ui.fresh = false;
      resetView();
    } else {
      applyTransform();
    }
    mark(DM.state.sel ? M.resolve(DM.state.sel) : null);
  }

  function btn(label, fn, tip) {
    return h("button", { type: "button", cls: "mini", tip: tip, onclick: fn }, label);
  }

  function render(ctx) {
    var M = ctx.M, st = ctx.state, key = st.caseId + "|" + st.phase;
    if (ui.caseId !== st.caseId) {
      ui.caseId = st.caseId;
      ui.collapsed = Object.create(null);
    }
    if (ui.key !== key) {
      ui.key = key;
      ui.fresh = true;
    }
    hookWindow();

    var tools = ctx.tools;
    tools.appendChild(DM.phasePicker(M));
    tools.appendChild(btn("fit", fitView, "the whole pipeline in the view"));
    tools.appendChild(btn("1:1", resetView, "actual size, from the top"));
    tools.appendChild(btn("expand all", function () {
      ui.collapsed = Object.create(null);
      draw();
    }));
    tools.appendChild(btn("collapse scopes", function () {
      if (!cur || !cur.tree) return;
      cur.tree.groups.forEach(function (g) { if (g.depth >= 1) ui.collapsed[g.gid] = true; });
      draw();
    }, "close every sub-scope box"));
    tools.appendChild(DM.checkbox("edges", ui.edges, function (on) {
      ui.edges = on;
      if (cur && cur.svg) cur.svg.classList.toggle("no-edges", !on);
    }, "show the dataflow edges"));
    tools.appendChild(h("span", { cls: "legend" },
      h("span", { cls: "lg" }, h("i", { cls: "sw rd" }), "read / consumes"),
      h("span", { cls: "lg" }, h("i", { cls: "sw wr" }), "write / produces"),
      h("span", { cls: "lg" }, h("i", { cls: "sw flag" }), "has checks"),
      h("span", { cls: "lg", tip: "xN: N launches merged into the step; ~: they alternate with other kernels' launches; a dashed box: the side lane" }, h("b", { cls: "pl" }, "xN~"), "merged launches"),
      h("span", { cls: "lg dim" }, "wheel: zoom, drag: pan, shift+wheel: scroll, click a scope header: collapse")));

    var wrap = h("div", { cls: "pipe-wrap" });
    ctx.body.appendChild(wrap);
    cur = { M: M, wrap: wrap, tree: null, L: null, svg: null, vp: null, hover: null };
    draw();
  }

  DM.registerView({ id: "pipeline", title: "Pipeline", phased: true, render: render, select: select });
})();
