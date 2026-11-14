// devmap: shared helpers. Errors, DOM building, number formatting, the
// pattern and type vocabularies, and the tooltip.
"use strict";
(function () {
  var DM = window.DM = window.DM || {};
  var SVG_NS = "http://www.w3.org/2000/svg";

  // ------------------------------------------------------------------
  // Errors: everything lands in the strip at the top of the page.
  // ------------------------------------------------------------------

  DM.showError = function (err, where) {
    var text = (where ? "[" + where + "] " : "") + (err && err.message ? err.message : String(err));
    if (err && err.stack) text += "\n" + err.stack;
    if (typeof window.devmapReport === "function") window.devmapReport(text);
    if (window.console && window.console.error) window.console.error(err);
  };

  // fn wrapped so that an exception is shown instead of lost.
  DM.guard = function (fn, where) {
    return function () {
      try {
        return fn.apply(this, arguments);
      } catch (e) {
        DM.showError(e, where);
      }
      return undefined;
    };
  };

  // addEventListener with a guarded handler.
  DM.on = function (el, type, fn, opts) {
    el.addEventListener(type, DM.guard(fn, "on " + type), opts === undefined ? false : opts);
  };

  // ------------------------------------------------------------------
  // The view registry (the view files register themselves at load).
  // ------------------------------------------------------------------

  DM.views = Object.create(null);
  DM.viewOrder = [];
  DM.registerView = function (def) {
    DM.views[def.id] = def;
    DM.viewOrder.push(def.id);
  };

  // ------------------------------------------------------------------
  // DOM building
  // ------------------------------------------------------------------

  function append(el, kids, from) {
    for (var i = from; i < kids.length; i++) {
      var k = kids[i];
      if (k === null || k === undefined || k === false) continue;
      if (Array.isArray(k)) append(el, k, 0);
      else if (k.nodeType) el.appendChild(k);
      else el.appendChild(document.createTextNode(String(k)));
    }
  }

  // attrs: cls -> class, tip -> data-tip, onX: function -> listener,
  // null / undefined / false are skipped, true is an empty attribute.
  function setAttrs(el, attrs) {
    if (!attrs) return;
    for (var k in attrs) {
      if (!Object.prototype.hasOwnProperty.call(attrs, k)) continue;
      var v = attrs[k];
      if (v === null || v === undefined || v === false) continue;
      if (k.length > 2 && k.charAt(0) === "o" && k.charAt(1) === "n" && typeof v === "function") DM.on(el, k.slice(2), v);
      else if (k === "cls") el.setAttribute("class", String(v));
      else if (k === "tip") { if (String(v) !== "") el.setAttribute("data-tip", String(v)); }
      else el.setAttribute(k, v === true ? "" : String(v));
    }
  }

  // An HTML element: h("div", { cls: "x" }, child, "text", [more]).
  DM.h = function (tag, attrs) {
    var el = document.createElement(tag);
    setAttrs(el, attrs);
    append(el, arguments, 2);
    return el;
  };

  // An SVG element, same calling convention.
  DM.s = function (tag, attrs) {
    var el = document.createElementNS(SVG_NS, tag);
    setAttrs(el, attrs);
    append(el, arguments, 2);
    return el;
  };

  // Symbols for UI text, built from code points so that the source stays ASCII.
  DM.CH = {
    right: String.fromCharCode(8594),
    up: String.fromCharCode(8593),
    down: String.fromCharCode(8595),
    loop: String.fromCharCode(8635)
  };

  DM.clear = function (el) {
    while (el.firstChild) el.removeChild(el.firstChild);
  };

  // Text escaped for an innerHTML string (element content and attributes).
  DM.esc = function (x) {
    return String(x).split("&").join("&amp;").split("<").join("&lt;").split(">").join("&gt;")
      .split("\"").join("&quot;").split("'").join("&#39;");
  };

  // ------------------------------------------------------------------
  // Numbers
  // ------------------------------------------------------------------

  function isNum(n) { return typeof n === "number" && isFinite(n); }
  DM.isNum = isNum;

  // 1234567 -> "1,234,567"
  DM.fmtInt = function (n) {
    if (!isNum(n)) return "-";
    var neg = n < 0, s = String(Math.round(Math.abs(n))), out = "";
    while (s.length > 3) {
      out = "," + s.slice(-3) + out;
      s = s.slice(0, -3);
    }
    return (neg ? "-" : "") + s + out;
  };

  function scaled(n, units, sep) {
    var a = Math.abs(n);
    if (a < 1024) return Math.round(n) + sep + units[0];
    var v = a / 1024, i = 1;
    while (v >= 1024 && i < units.length - 1) {
      v /= 1024;
      i++;
    }
    var t = v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2);
    return (n < 0 ? "-" : "") + t + sep + units[i];
  }

  // 1536 -> "1.50 KB" (powers of 1024)
  DM.fmtBytes = function (n) {
    if (!isNum(n)) return "-";
    return scaled(n, ["B", "KB", "MB", "GB", "TB"], " ");
  };

  // 1536 -> "1.50K", for tight places
  DM.fmtBytesShort = function (n) {
    if (!isNum(n)) return "-";
    return scaled(n, ["B", "K", "M", "G", "T"], "");
  };

  // "1.50 KB (1,536 bytes)"
  DM.fmtBytesFull = function (n) {
    if (!isNum(n)) return "-";
    return n < 1024 ? DM.fmtInt(n) + " bytes" : DM.fmtBytes(n) + " (" + DM.fmtInt(n) + " bytes)";
  };

  // A span with the short form shown and the exact count on hover.
  DM.bytesEl = function (n, cls) {
    if (!isNum(n)) return DM.h("span", { cls: "num dim" + (cls ? " " + cls : "") }, "-");
    return DM.h("span", { cls: "num" + (cls ? " " + cls : ""), tip: DM.fmtInt(n) + " bytes" }, DM.fmtBytes(n));
  };

  // What stands for an amount nobody counted. A GPU's case has no measured
  // accesses: it knows that a kernel reads or writes a section, not how much.
  DM.NM = "n/m";
  DM.NM_TIP = "not measured: this case has no access counts, only whether a kernel reads or writes a section";
  DM.nmEl = function (text) {
    return DM.h("span", { cls: "nm", tip: DM.NM_TIP }, text || DM.NM);
  };

  // A byte total of which a part may be uncounted: "n/m" when nothing was
  // counted, "1.20 MB +n/m" when some was (the host's copies are always counted).
  DM.fmtAmount = function (bytes, unk, short) {
    if (unk && !bytes) return DM.NM;
    return (short ? DM.fmtBytesShort(bytes) : DM.fmtBytes(bytes)) + (unk ? " +" + DM.NM : "");
  };
  DM.amountEl = function (bytes, unk, cls) {
    if (unk && !bytes) return DM.nmEl();
    var el = DM.bytesEl(bytes, cls);
    if (unk) el.appendChild(DM.nmEl(" +" + DM.NM));
    return el;
  };
  // Whether a total holds no counted byte and an uncounted part: a compact
  // label leaves such a total out.
  DM.uncounted = function (r, rUnk, w, wUnk) {
    return !r && !w && !!(rUnk || wUnk);
  };

  // 0.8312 -> "83.1%"
  DM.fmtPct = function (x) {
    if (!isNum(x)) return "-";
    var p = x * 100;
    if (p === 0) return "0%";
    if (p < 0.1) return "<0.1%";
    if (p < 10) return p.toFixed(1) + "%";
    return p.toFixed(0) + "%";
  };

  DM.fmtNum = function (x, digits) {
    if (!isNum(x)) return "-";
    return x.toFixed(digits === undefined ? 2 : digits);
  };

  // microseconds -> "12.5 us", "3.20 ms", "1.20 s"
  DM.fmtTime = function (us) {
    if (!isNum(us)) return "-";
    if (us < 1000) return (us < 10 ? us.toFixed(2) : us < 100 ? us.toFixed(1) : us.toFixed(0)) + " us";
    if (us < 1e6) return (us / 1000).toFixed(us < 1e4 ? 2 : 1) + " ms";
    return (us / 1e6).toFixed(2) + " s";
  };

  DM.hex = function (n) {
    if (!isNum(n)) return "-";
    return "0x" + Math.round(n).toString(16);
  };

  // ------------------------------------------------------------------
  // Text that must fit (monospace: the width is estimated per character)
  // ------------------------------------------------------------------

  DM.CHAR_W = 0.62;   // advance of the monospace face, in em, rounded up

  DM.textW = function (text, fontPx) {
    return String(text).length * fontPx * DM.CHAR_W;
  };

  // text cut to fit in px pixels at fontPx, ".." marking the cut.
  DM.fit = function (text, px, fontPx) {
    text = String(text);
    var max = Math.floor(px / (fontPx * DM.CHAR_W));
    if (text.length <= max) return text;
    if (max <= 2) return text.slice(0, Math.max(0, max));
    return text.slice(0, max - 2) + "..";
  };

  // ------------------------------------------------------------------
  // Scales
  // ------------------------------------------------------------------

  // v in lo..hi on a log scale -> 0..1
  DM.logScale = function (v, lo, hi) {
    if (!isNum(v) || v <= 0) return 0;
    if (!(hi > lo) || lo <= 0) return 1;
    var t = (Math.log(v) - Math.log(lo)) / (Math.log(hi) - Math.log(lo));
    return t < 0 ? 0 : t > 1 ? 1 : t;
  };

  // 0..1 -> the alpha of a heat fill
  DM.heatAlpha = function (t) {
    return Math.round((0.14 + 0.78 * t) * 100) / 100;
  };

  // ------------------------------------------------------------------
  // Access patterns
  // ------------------------------------------------------------------

  DM.PAT = {
    coalesced: { letter: "C", doc: "adjacent threads touch adjacent elements, each element about once" },
    serial:    { letter: "S", doc: "each thread walks a contiguous run" },
    strided:   { letter: "T", doc: "regular, but not adjacent" },
    gather:    { letter: "G", doc: "index-driven reads, irregular" },
    scatter:   { letter: "X", doc: "index-driven writes, irregular" },
    broadcast: { letter: "B", doc: "every thread reads the same few elements" },
    mixed:     { letter: "M", doc: "no single pattern dominates" },
    unknown:   { letter: "", name: "not measured", doc: "only the direction (read or written) is known" },
    upload:    { letter: "U", doc: "host to device copy" },
    download:  { letter: "D", doc: "device to host copy: the host waits for the device" }
  };
  DM.PAT_ORDER = ["coalesced", "serial", "strided", "gather", "scatter", "broadcast", "mixed", "unknown", "upload", "download"];

  DM.patLetter = function (p) {
    return Object.prototype.hasOwnProperty.call(DM.PAT, p) ? DM.PAT[p].letter : "?";
  };
  DM.patDoc = function (p) {
    return Object.prototype.hasOwnProperty.call(DM.PAT, p) ? DM.PAT[p].doc : "";
  };
  // The name shown for a pattern ("unknown" reads "not measured").
  DM.patName = function (p) {
    return Object.prototype.hasOwnProperty.call(DM.PAT, p) && DM.PAT[p].name ? DM.PAT[p].name : p;
  };
  // <span class="pat"><b>C</b>coalesced</span>
  DM.patEl = function (p) {
    p = p || "unknown";
    if (p === "unknown") return DM.nmEl(DM.patName(p));
    return DM.h("span", { cls: "pat", tip: p + ": " + DM.patDoc(p) }, DM.h("b", null, DM.patLetter(p)), p);
  };
  // The pattern of one direction of an access. A borrowed one (measured in
  // the probed case `from` names, not in this case) is in italics.
  DM.dirPatEl = function (d) {
    var from = DM.M && typeof DM.M.meta.patterns_from === "string" ? DM.M.meta.patterns_from : "";
    if (d.unseen) return DM.h("span", { cls: "nm", tip: "the kernel's signature allows this access; its launches in the probed case" + (from ? " " + from : "") + " did not do it" }, "not seen when probed");
    var el = DM.patEl(d.pattern);
    if (d.borrowed) {
      el.setAttribute("class", "pat bor");
      el.setAttribute("data-tip", d.pattern + ": " + DM.patDoc(d.pattern) + "\nborrowed: measured for this kernel and section in the probed case" + (from ? " " + from : "") + ", not in this case");
    }
    return el;
  };
  // A legend entry per pattern; `unknown` only where the case has such accesses.
  DM.patLegend = function (pats, cls) {
    return pats.map(function (p) {
      return DM.h("span", { cls: "lg", tip: DM.patName(p) + ": " + DM.patDoc(p) }, DM.h("b", { cls: "pl" + (cls ? " " + cls : "") }, DM.patLetter(p)), DM.patName(p));
    });
  };

  // ------------------------------------------------------------------
  // Stored types
  // ------------------------------------------------------------------

  DM.TYPES = ["u32", "u64", "f32", "f64", "df32", "bytes"];
  DM.TYPE_SIZE = { u8: 1, i8: 1, u16: 2, i16: 2, u32: 4, i32: 4, f32: 4, u64: 8, i64: 8, f64: 8, df32: 8 };

  // The CSS class that carries a type's colour.
  DM.tyClass = function (ty) {
    return DM.TYPES.indexOf(ty) >= 0 ? "ty-" + ty : "ty-other";
  };

  // A type chip; role is the kernel-side typedef ("V = f32").
  DM.tyEl = function (ty, role) {
    var text = ty || "?";
    return DM.h("span", { cls: "ty " + DM.tyClass(ty), tip: role ? "role " + role + ", resolved type " + text : "type " + text },
      role ? DM.h("i", null, role + " = ") : null, text);
  };

  // ------------------------------------------------------------------
  // Tooltip: any element with data-tip, HTML or SVG.
  // ------------------------------------------------------------------

  var tipEl = null, tipFor = null;

  function tipHide() {
    if (tipEl) tipEl.style.display = "none";
    tipFor = null;
  }
  DM.hideTip = tipHide;

  DM.initTips = function () {
    tipEl = DM.h("div", { id: "tip" });
    tipEl.style.display = "none";
    document.body.appendChild(tipEl);
    DM.on(document, "mousemove", function (e) {
      var t = e.target;
      if (t && t.nodeType !== 1) t = t.parentNode;
      var el = t && t.closest ? t.closest("[data-tip]") : null;
      if (!el) {
        if (tipFor) tipHide();
        return;
      }
      if (el !== tipFor) {
        tipFor = el;
        tipEl.textContent = el.getAttribute("data-tip");
        tipEl.style.display = "block";
      }
      var w = tipEl.offsetWidth, hh = tipEl.offsetHeight;
      var x = e.clientX + 14, y = e.clientY + 18;
      if (x + w > window.innerWidth - 6) x = Math.max(6, e.clientX - w - 12);
      if (y + hh > window.innerHeight - 6) y = Math.max(6, e.clientY - hh - 12);
      tipEl.style.left = x + "px";
      tipEl.style.top = y + "px";
    });
    DM.on(document, "mousedown", tipHide);
    DM.on(document, "keydown", tipHide);
    DM.on(document, "wheel", tipHide, { passive: true, capture: true });
  };
})();
