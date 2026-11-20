#!/usr/bin/env python3
"""devmap.py: a device trace (ARAEL_GPU_TRACE) into a case for the viewer.

    devmap.py TRACE --id bal-49-f32-cpu --title "..." [--problem ..] [--model f32]
              [--solve N] [--out ../data] [--repo ../../..] [--patterns-from CASE.js]

A trace holds every solve of the process; one is turned into the case: the
one with the most attempts, or the N-th (--solve). A trace without measured
accesses (a GPU's) takes its access patterns from a probed case of the same
model (--patterns-from). See ../SCHEMA.md.
"""
import argparse
import datetime
import glob
import json
import os
import re
import subprocess
import sys

# ---------------------------------------------------------------- scopes

SCOPE_DOCS = {
    "setup": ("Setup", "Once per sparsity structure: the factor's and the reduction's plans, the model's image and the kernels go to the device."),
    "setup/factor": ("Factor plan", "The supernodal factor's plan (index arrays) and its work block."),
    "setup/reduction": ("Reduction plan", "The Schur reduction's plan and its work block (H, the gradient, S, the step)."),
    "setup/route": ("Model image", "The model laid out in columns (structure, then values), and the LM loop's vectors."),
    "setup/route/image": ("Values upload", "The model's values (fixed parameters, data, references) written and uploaded."),
    "compute": ("First linearization", "The gradient, H and the cost at the starting point, assembled on the device; the gradient, H's diagonal and the cost are read back."),
    "start": ("Start", "The parameters go to the device; the diagonal pass over the linearization already there."),
    "linearize": ("Linearize", "After an accepted step: the trial point becomes the parameters, then the gradient, H and the cost are assembled there. One captured graph."),
    "attempt": ("Attempt", "One damped solve and its trial point: damp H, reduce, factorize, solve, back-substitute, form x - step, evaluate the cost there. One captured graph; the host waits once, for the status and the scalars."),
    "finish": ("Finish", "The parameters and the re-centred references come back to the host."),
    "factorize": ("Factorize (host values)", "A factorization of values uploaded by the host."),
    "solve": ("Solve (host vector)", "A solve of a vector uploaded by the host."),
    "solve_reduced": ("Reduced solve (host H)", "H and the gradient uploaded by the host, the damped solve on the device."),
    "params": ("Parameters", "The parameters read back."),
    "accept": ("Accept", "The trial point becomes the parameters."),
}

SUB_DOCS = {
    "assemble": ("Assembly", "The gradient, H's tiles and the cost from the model's constraints."),
    "zero": ("Zero the targets", "The gradient, H and the cost slot zeroed before the sums add into them."),
    "gather": ("Gather", "Each entity's parameter values from x, then the referenced fields' composed values (rotations) and derivatives."),
    "constraints": ("Constraints", "Per constraint: a record per instance (residual rows and partials), then the sums of the records into each entity's tile and gradient, the cross tiles, and the cost."),
    "pass": ("Diagonal pass", "H's diagonal, the damping scale, the gradient's maxima and the first faulty diagonal entry."),
    "accept": ("Accept", "References advance by their steps, the trial point is copied over the parameters, the rotation steps are zeroed."),
    "damp": ("Damp", "H's diagonal gets diag + lambda * scale, in place."),
    "reduce": ("Schur reduction", "Eliminated blocks' factors and z, W = C D^-1 per observer, S per tile, the reduced right-hand side."),
    "factorize": ("Factorize S", "The supernodal Cholesky of S, level by level."),
    "solve": ("Solve", "The reduced system solved: permute in, forward and backward level solves, permute out."),
    "refine": ("Refine", "Iterative refinement of the reduced solve against S."),
    "backsub": ("Back-substitution", "The kept solution scattered into the step, then the eliminated blocks' part."),
    "step": ("Trial point", "x - step, the step's and the trial point's squared norms, the predicted reduction."),
    "cost": ("Trial cost", "The cost at the trial point: gather, then each constraint's cost alone."),
    "image": ("Values upload", "The model's values written into the staging buffer and uploaded."),
    "factor": ("Factor plan", "The supernodal factor's plan and work block."),
    "reduction": ("Reduction plan", "The Schur reduction's plan and work block."),
    "route": ("Model image", "The model's image and the LM loop's vectors."),
}

OWNERS = {"image": "DeviceAssembly (src/device.rs)", "lm": "LmVectors (arael-gpu/src/lm.rs)",
          "schur.plan": "GpuSchur (arael-gpu/src/schur.rs)", "schur.work": "GpuSchur (arael-gpu/src/schur.rs)",
          "factor.plan": "GpuSupernodal (arael-gpu/src/supernodal.rs)", "factor.work": "GpuSupernodal (arael-gpu/src/supernodal.rs)"}

BLOCK_DOCS = {"image": "The model's data in columns: structure (written once), values (rewritten per solve), then what the kernels compute.",
              "lm": "The LM loop's vectors and the scalars the loop decides on.",
              "schur.plan": "The Schur reduction's index arrays.",
              "schur.work": "The reduction's values: H, the gradient, the eliminated blocks' factors, W, S, the reduced vectors, the step and the status.",
              "factor.plan": "The supernodal factor's index arrays.",
              "factor.work": "The factor's values, the solve's working vector, the inverted diagonal tiles."}

FLOW_ORDER = ["setup", "compute", "start", "attempt", "linearize", "finish"]

# ---------------------------------------------------------------- kernels

SCALARS = {"float": "f32", "double": "f64", "u32": "u32", "i32": "i32", "u64": "u64", "i64": "i64", "u8": "bytes", "df32": "df32", "unsigned": "u32", "int": "i32"}


def unit_types(defines, source):
    """The unit's typedefs resolved: F, W, V, R, CACC."""
    d = dict(re.findall(r"#define\s+(\w+)\s+(\S+)", defines))
    t = {}
    t["W"] = "df32" if "ARAEL_WIDE_DF32" in d else "f64"
    if "SCALAR" in d:
        t["F"] = SCALARS.get(d["SCALAR"], d["SCALAR"])
    if "VALUES_F32" in d or re.search(r"\bV\b", source):
        t["V"] = "f32" if d.get("VALUES_F32") == "1" else t["W"]
    if "SCHUR_F32" in d:
        t["R"] = "f32" if d["SCHUR_F32"] == "1" else t["W"]
    m = re.search(r"typedef\s+(\w+)\s+CACC\s*;", source)
    if m:
        t["CACC"] = t.get(m.group(1), SCALARS.get(m.group(1), m.group(1)))
    return t


def split_params(text):
    out, depth, cur = [], 0, ""
    for ch in text:
        if ch in "(<[":
            depth += 1
        elif ch in ")>]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return out


def parse_kernel(source, name, types):
    """The entry point `name` of `source`: its doc comment, parameters and text."""
    m = re.search(r"^[ \t]*[A-Z_]+(?:\([^)\n]*\))?[ \t]+void[ \t]+" + re.escape(name) + r"[ \t]*\(", source, re.M)
    if not m:
        return None
    # the parameter list: to the matching parenthesis
    i, depth = m.end(), 1
    while depth and i < len(source):
        depth += {"(": 1, ")": -1}.get(source[i], 0)
        i += 1
    params_text = source[m.end():i - 1]
    # the body: to the matching brace
    j = source.index("{", i)
    k, depth = j + 1, 1
    while depth and k < len(source):
        depth += {"{": 1, "}": -1}.get(source[k], 0)
        k += 1
    # the comment block right above (preprocessor lines between are skipped)
    lines = source[:m.start()].split("\n")
    doc, crossed = [], False
    for ln in reversed(lines[:-1] if lines and lines[-1].strip() == "" else lines):
        s = ln.strip()
        if s.startswith("//"):
            doc.append(s[2:].strip())
        elif s.startswith("#"):
            crossed = True
        elif s == "":
            # a comment inside a preprocessor block is not the whole doc: one blank line more
            if doc and not crossed:
                break
            crossed = False
        else:
            break
    if not doc:
        # an entry point that only calls a helper: the helper's comment; else the comment block before
        body = source[j:k]
        for callee in re.findall(r"\b([a-z_][a-z0-9_]*)\s*(?:<[^;(]*>)?\(", body):
            hm = re.search(r"^[ \t]*(?:template\s*<[^>\n]*>\s*)?DEVICE(?:_INLINE)?[ \t]+\w+[ \t]+" + re.escape(callee) + r"[ \t]*\(", source, re.M)
            if hm:
                hl = source[:hm.start()].rstrip("\n").split("\n")
                while hl and hl[-1].strip().startswith("//"):
                    doc.append(hl.pop().strip()[2:].strip())
                if doc:
                    break
        if not doc:
            # one comment over several entry points: the entry point right above shares it
            prev = [pm for pm in re.finditer(r"^[ \t]*[A-Z_]+(?:\([^)\n]*\))?[ \t]+void[ \t]+(\w+)[ \t]*\(", source[:m.start()], re.M)]
            if prev and "KERNEL" in prev[-1].group(0) and source[prev[-1].start():m.start()].count("\n") < 12:
                shared = parse_kernel(source, prev[-1].group(1), types)
                if shared and shared["doc"]:
                    doc = [shared["doc"]]
    params = []
    for p in split_params(params_text):
        p = " ".join(p.replace("GLOBAL", "").replace("RESTRICT", "").split())
        # a struct passed with the launch (Arg::Bytes): read from the launch's parameters, not from device memory
        tm = re.match(r"TABLE_PARAM\(\s*(\w+)\s*\)\s+(\w+)$", p)
        if tm:
            params.append({"name": tm.group(2), "ctype": "TABLE_PARAM(%s)" % tm.group(1), "type": tm.group(1), "ptr": False, "const": True, "table": True})
            continue
        pm = re.match(r"(.*?)(\w+)$", p)
        if not pm:
            continue
        ctype, pname = pm.group(1).strip(), pm.group(2)
        ptr = "*" in ctype
        base = ctype.replace("const", "").replace("*", "").strip()
        params.append({"name": pname, "ctype": ctype, "type": types.get(base, SCALARS.get(base, base)), "ptr": ptr,
                       "const": "const" in ctype or not ptr})
    return {"doc": " ".join(reversed(doc)), "params": params, "source": source[m.start():k], "line_in_unit": source[:m.start()].count("\n") + 1}


def find_in_repo(repo, name):
    for f in sorted(glob.glob(os.path.join(repo, "arael-gpu/kernels/*.cpp"))):
        for n, line in enumerate(open(f), 1):
            if re.search(r"\bvoid\s+" + re.escape(name) + r"\s*\(", line):
                return os.path.relpath(f, repo), n
    return "generated", 0


def col_accesses(kernel_source, unit_source):
    """The columns a generated kernel names through COL (read) and COL_MUT (written)."""
    alias = dict(re.findall(r"#define\s+ARAEL_COL_(\w+)\s+ARAEL_COL_(\w+)", unit_source))
    reads = {alias.get(n, n) for n in re.findall(r"\bCOL\(\s*\w+\s*,[^,]+,\s*(\w+)\s*\)", kernel_source)}
    writes = {alias.get(n, n) for n in re.findall(r"\bCOL_MUT\(\s*\w+\s*,[^,]+,\s*(\w+)\s*\)", kernel_source)}
    return reads, writes


# ---------------------------------------------------------------- trace

def load(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def split_solves(events):
    """Indices where a solve starts: a `setup` scope whose first child is `factor`."""
    starts = []
    for i, e in enumerate(events):
        if e["e"] == "scope" and e["name"] == "setup" and i + 1 < len(events) and events[i + 1].get("name") == "factor":
            starts.append(i)
    return starts


def touch_sum(a, b):
    if a is None:
        return dict(b) if b else None
    if b is None:
        return a
    out = dict(a)
    for k in ("n", "bytes", "unique", "threads", "seq", "same", "adjacent", "shared"):
        out[k] = a[k] + b[k]
    out["lo"], out["hi"], out["sizes"] = min(a["lo"], b["lo"]), max(a["hi"], b["hi"]), a["sizes"] | b["sizes"]
    return out


class Builder:
    def __init__(self, events, repo):
        self.events = events
        self.repo = repo
        self.units = {}
        self.blocks = {}
        self.graphs = {}
        self.device = {}
        self.probe = False

    def scan_globals(self):
        for e in self.events:
            t = e["e"]
            if t == "device":
                self.device = e
            elif t == "unit":
                u = self.units.setdefault(e["unit"], {"entries": [], "defines": e["defines"], "source": e["source"]})
                u["entries"] = sorted(set(u["entries"]) | set(e["entries"]))
            elif t == "capture":
                self.graphs[e["graph"]] = e["events"]

    def flatten(self, lo, hi):
        """The solve's operations in execution order, each with its scope path."""
        items, stack = [], []
        blocks = self.blocks

        def launch(e, acc, graph, us=None):
            a = []
            for x in acc or []:
                b = x["b"] if "b" in x else e["args"][x["arg"]].get("b", 0)
                a.append({"block": b, "s": x["s"], "r": x["r"], "w": x["w"]})
            if acc is not None:
                self.probe = True
            items.append({"kind": "launch", "k": e["k"], "unit": e["unit"], "grid": e["grid"], "threads": e["threads"], "shared": e["shared"],
                          "lane": e["lane"], "args": e["args"], "acc": a if acc is not None else None, "scope": list(stack), "graph": graph, "us": us})

        for e in self.events[lo:hi]:
            t = e["e"]
            if t == "alloc":
                blocks[e["block"]] = {"id": e["block"], "bytes": e["bytes"], "name": "block%d" % e["block"], "sections": None}
            elif t == "describe":
                b = blocks.setdefault(e["block"], {"id": e["block"], "bytes": 0})
                b["name"], b["sections"] = e["name"], e["sections"]
            elif t == "scope":
                stack.append(e["name"])
            elif t == "end":
                if stack:
                    stack.pop()
            elif t == "launch":
                launch(e, e.get("acc"), False, e.get("us"))
            elif t == "replay":
                acc, us = e.get("acc"), e.get("us")
                n = 0
                for ge in self.graphs.get(e["graph"], []):
                    if ge["e"] == "scope":
                        stack.append(ge["name"])
                    elif ge["e"] == "end":
                        stack.pop()
                    elif ge["e"] == "launch":
                        launch(ge, acc[n] if acc is not None and n < len(acc) else None, True, us[n] if us is not None and n < len(us) else None)
                        n += 1
            elif t == "write":
                items.append({"kind": "upload", "block": e["block"], "at": e["at"], "bytes": e["bytes"], "via": e["via"], "scope": list(stack)})
            elif t == "read":
                items.append({"kind": "download", "block": e["block"], "at": e["at"], "bytes": e["bytes"], "group": e.get("group"), "scope": list(stack)})
        return items


def section_at(block, at):
    """The index of the section of `block` holding byte `at`, or -1."""
    secs = block.get("sections") or []
    best = -1
    for i, s in enumerate(secs):
        if s["at"] <= at < s["at"] + max(s["bytes"], 1):
            return i
        if s["at"] <= at and (best < 0 or s["at"] > secs[best]["at"]):
            best = i
    return best if secs and at == 0 else -1 if not secs else best


# the kinds of sequence a kernel could compute from the position instead of reading
REGULAR = ("constant", "arithmetic", "pieces")


def linear(base, stride, var):
    """base + stride * var, written short."""
    if stride == 0:
        return str(base)
    term = var if abs(stride) == 1 else "%d %s" % (abs(stride), var)
    if base == 0:
        return term if stride > 0 else "-" + term
    if stride < 0:
        return "%d - %s" % (base, term)
    return "%d + %s" % (base, term) if base > 0 else "%s - %d" % (term, -base)


def seq_formula(c, var):
    """A sequence's formula in `var` (i: the word's index, r: the record's), "?" where it has none."""
    k = c["kind"]
    if k == "constant":
        return str(c["value"])
    if k == "arithmetic":
        return linear(c["base"], c["stride"], var)
    if k == "pieces":
        # a piece (at, base, stride) is base + (var - at) * stride from at on
        parts = [linear(base - at * stride, stride, var) for at, base, stride in c["pieces"]]
        return "; ".join(parts[:1] + ["from %d: %s" % (piece[0], text) for piece, text in zip(c["pieces"][1:], parts[1:])])
    if k == "records":
        return "(" + ", ".join(seq_formula(f, "r") for f in c["fields"]) + ")"
    return "?"


def computable_of(c):
    """How much of an index section a kernel could compute from the position:
    (the share of its words, the formula). A record counts by its fields."""
    k = c["kind"]
    if k in REGULAR:
        return 1.0, seq_formula(c, "i")
    if k == "records":
        fits = sum(1 for f in c["fields"] if f["kind"] in REGULAR)
        return fits / c["period"], seq_formula(c, "r") if fits else ""
    return 0.0, ""


def irregular_text(c, n):
    """An irregular sequence of n words in words."""
    # a stretch: consecutive words with one step between neighbours (0, 3, 6, 9, 13, 17 is two);
    # one stretch would be base + i * stride
    return "%s, %d distinct values in %d words; the step between neighbours is constant over %d stretches, %.1f words each on average" % (
        "non-decreasing" if c["kind"] == "monotone" else "irregular", c["distinct"], n, c["runs"], n / max(c["runs"], 1))


def content_text(c):
    """An index section's content classification in words."""
    if not c:
        return ""
    k = c["kind"]
    if k == "constant":
        return "Content: every word is %d." % c["value"]
    if k == "arithmetic":
        return "Content: word i is %s." % seq_formula(c, "i")
    if k == "pieces":
        return "Content: %d stretches of base + i * stride: %s." % (len(c["pieces"]), seq_formula(c, "i"))
    if k == "records":
        m, fits = c["n"] // c["period"], sum(1 for f in c["fields"] if f["kind"] in REGULAR)
        rest = "; ".join("field %d: %s" % (j, irregular_text(f, m)) for j, f in enumerate(c["fields"]) if f["kind"] not in REGULAR)
        return "Content: %d records of %d words%s, record r is %s: %d of %d fields computable from r.%s" % (
            m, c["period"], "" if c.get("declared") else " (width found from the data)", seq_formula(c, "r"), fits, c["period"], " Of the others, " + rest + "." if rest else "")
    return "Content: %s (a single stretch would be computable)." % irregular_text(c, c["n"])


def copy_groups(blocks):
    """The irregular index sequences (a section's words, or one field of its
    records) that hold the same words or a multiple of them: equal count and
    equal hash of the words divided by their greatest common divisor.
    [{n, members: [{block, section, field (-1: the section's own words), scale}]}],
    a member's words being scale / scale' times another's."""
    groups = {}
    for blk in blocks:
        for sec in blk["sections"]:
            c = sec.get("content")
            if not c:
                continue
            if c["kind"] in ("monotone", "other"):
                seqs = [(-1, c, c["n"])]
            elif c["kind"] == "records":
                seqs = [(j, f, c["n"] // c["period"]) for j, f in enumerate(c["fields"]) if f["kind"] in ("monotone", "other")]
            else:
                continue
            for j, f, n in seqs:
                groups.setdefault((n, f["hash"]), []).append({"block": blk["id"], "section": sec["i"], "field": j, "scale": f["gcd"]})
    return [{"n": n, "members": members} for (n, _), members in groups.items() if len(members) > 1]


def sections_in(block, at, nbytes):
    """The indices of the sections of `block` that bytes at..at+nbytes overlap."""
    return [i for i, s in enumerate(block.get("sections") or []) if s["bytes"] > 0 and s["at"] < at + nbytes and at < s["at"] + s["bytes"]]


def pattern(t, write, elem):
    """A touch's pattern, from its counts."""
    n, threads, unique = t["n"], t["threads"], t["unique"]
    pairs = max(threads - 1, 1)
    adj, bro, seq = t["adjacent"] / pairs, t["shared"] / pairs, t["seq"] / max(n, 1)
    if threads > 4 and (bro > 0.8 or unique <= 4):
        return "broadcast"
    if threads > 1 and adj > 0.8:
        return "coalesced"
    if seq > 0.6:
        return "serial"
    dense = unique * elem >= 0.95 * (t["hi"] - t["lo"] + 1)
    if dense and n <= 1.2 * unique and threads > 1 and adj > 0.4:
        return "strided"
    if threads <= 1 and n <= 4:
        return "serial"
    return "scatter" if write else "gather"


def load_case(path):
    """A case file's object: the argument of its DEVMAP.add call."""
    text = open(path).read()
    head = "DEVMAP.add("
    return json.loads(text[text.index(head) + len(head):text.rindex(");")])


def write_case(case, path):
    with open(path, "w") as f:
        f.write("// Generated by tools/devmap/gen/devmap.py from a device trace; see ../SCHEMA.md.\n")
        f.write("DEVMAP.add(")
        json.dump(case, f, separators=(",", ":"))
        f.write(");\n")


def borrow_patterns(case, donor):
    """Gives the unmeasured accesses of `case` the patterns measured in
    `donor`, a probed case of the same model: the pattern of the same kernel
    on the same section in the same scope, else the kernel's pattern on that
    section over the donor's solve (the one that moved the most bytes).
    Such a direction is marked `borrowed`. One the donor's launches of the
    kernel never did (the signature allows it: a failure flag, an unused
    column) is marked `unseen`. Returns (borrowed, unseen, left unknown: the
    kernel is not in the donor)."""
    def names(c):
        return {(blk["id"], s["i"]): (blk["name"], s["name"]) for blk in c["blocks"] for s in blk["sections"]}

    def launches(c):
        nm = names(c)
        for ph in c["phases"]:
            for s in ph["steps"]:
                if s["kind"] == "launch":
                    for a in s["access"]:
                        if (a["block"], a["section"]) in nm:
                            yield s, a, nm[(a["block"], a["section"])]

    in_scope, in_solve, probed = {}, {}, set()
    for s, a, nm in launches(donor):
        if a["src"] != "measured":
            continue
        probed.add(s["kernel"])
        for rw in ("r", "w"):
            if a[rw]:
                for table, key in ((in_scope, (tuple(s["scope"]), s["kernel"], nm, rw)), (in_solve, (s["kernel"], nm, rw))):
                    seen = table.setdefault(key, {})
                    seen[a[rw]["pattern"]] = seen.get(a[rw]["pattern"], 0) + a[rw]["bytes"]
    borrowed = unseen = unknown = 0
    missing = set()
    for s, a, nm in launches(case):
        for rw in ("r", "w"):
            d = a[rw]
            if not d or d.get("pattern", "unknown") != "unknown":
                continue
            seen = in_scope.get((tuple(s["scope"]), s["kernel"], nm, rw)) or in_solve.get((s["kernel"], nm, rw))
            if seen:
                d["pattern"], d["borrowed"] = max(seen, key=seen.get), True
                borrowed += 1
            elif s["kernel"] in probed:
                d["unseen"] = True
                unseen += 1
            else:
                missing.add(s["kernel"])
                unknown += 1
    case["meta"]["patterns_from"] = donor["id"]
    case["meta"]["notes"].append("access patterns borrowed from %s: %d of %d accesses" % (donor["id"], borrowed, borrowed + unseen + unknown))
    if missing:
        case["findings"].append({"level": "info", "title": "%d kernels of this case are not in the probed case %s" % (len(missing), donor["id"]),
                                 "text": "Nothing is measured for them: the probed problem does not launch them. " + ", ".join(sorted(missing)) + ".",
                                 "refs": [{"kind": "kernel", "name": k} for k in sorted(missing)]})
    return borrowed, unseen, unknown


def build_case(args):
    events = load(args.trace)
    b = Builder(events, args.repo)
    b.scan_globals()
    starts = split_solves(events)
    if not starts:
        sys.exit("no solve in the trace (no setup/factor scope)")
    bounds = [(s, starts[i + 1] if i + 1 < len(starts) else len(events)) for i, s in enumerate(starts)]
    # blocks allocated before a solve's start (shared devices) are seen by scanning from 0
    def attempts(lo, hi):
        return sum(1 for e in events[lo:hi] if e["e"] == "scope" and e["name"] == "attempt")
    pick = args.solve if args.solve is not None else max(range(len(bounds)), key=lambda i: attempts(*bounds[i]))
    lo, hi = bounds[pick]
    items = b.flatten(lo, hi)
    blocks = {i: blk for i, blk in b.blocks.items()}

    # ------------------------------------------------ units and kernels
    used_units = sorted({it["unit"] for it in items if it["kind"] == "launch"})
    units, kernels = [], {}
    unit_index = {}
    for uid in used_units:
        u = b.units[uid]
        types = unit_types(u["defines"], u["source"])
        first = u["entries"][0] if u["entries"] else ""
        root = re.search(r"Device assembly of the root `(\w+)`", u["source"])
        if root:
            name, file = "assembly:" + root.group(1), "generated + arael-gpu/kernels/assembly.cpp"
        elif any(e.startswith("k_schur") for e in u["entries"]):
            name, file = "schur", "arael-gpu/kernels/schur.cpp"
        elif any(e.startswith("k_cg") for e in u["entries"]):
            name, file = "cg", "arael-gpu/kernels/cg.cpp"
        elif "k_lin_pass" in u["entries"]:
            name, file = "lm", "arael-gpu/kernels/lm.cpp"
        elif "k_potf2" in u["entries"] or "k_seed_f32" in u["entries"]:
            name, file = "factor", "arael-gpu/kernels/supernodal.cpp"
        else:
            name, file = first, ""
        unit_index[uid] = len(units)
        units.append({"id": len(units), "name": name, "file": file, "defines": u["defines"], "types": types, "kernels": []})
    for it in items:
        if it["kind"] != "launch" or it["k"] in kernels:
            continue
        u = b.units[it["unit"]]
        ui = unit_index[it["unit"]]
        parsed = parse_kernel(u["source"], it["k"], units[ui]["types"]) or {"doc": "", "params": [], "source": ""}
        file, line = find_in_repo(args.repo, it["k"])
        kernels[it["k"]] = {"unit": ui, "file": file, "line": line, "doc": parsed["doc"], "params": parsed["params"], "source": parsed["source"]}
        units[ui]["kernels"].append(it["k"])
    unit_sources = {unit_index[uid]: b.units[uid]["source"] for uid in used_units}

    # ------------------------------------------------ blocks of the solve
    touched = set()
    for it in items:
        if it["kind"] == "launch":
            touched |= {a["b"] for a in it["args"] if "b" in a}
        else:
            touched.add(it["block"])
    out_blocks = []
    for bid in sorted(touched):
        blk = blocks.get(bid, {"id": bid, "bytes": 0, "name": "block%d" % bid, "sections": None})
        secs = blk.get("sections")
        if not secs:
            secs = [{"name": "(all)", "at": 0, "bytes": blk["bytes"], "elem": 4, "ty": "bytes", "role": "", "group": "", "doc": "not described"}]
            blk["sections"] = secs
        out_blocks.append({"id": bid, "name": blk["name"], "owner": OWNERS.get(blk["name"], ""), "bytes": blk["bytes"], "doc": BLOCK_DOCS.get(blk["name"], ""),
                           "sections": [{"i": i, "name": s["name"], "at": s["at"], "bytes": s["bytes"], "elem": s["elem"], "count": s["bytes"] // max(s["elem"], 1),
                                         "ty": s["ty"], "role": s["role"], "group": s["group"],
                                         "doc": s["doc"],
                                         **({"about": content_text(s["content"])} if s.get("content") else {}),
                                         **({"record": s["record"]} if s.get("record", 1) > 1 else {}),
                                         **({"content": s["content"]} if s.get("content") else {}),
                                         **(dict(zip(("computable", "formula"), computable_of(s["content"]))) if s.get("content") else {}),
                                         **({"fields": [seq_formula(f, "r") for f in s["content"]["fields"]]} if (s.get("content") or {}).get("kind") == "records" else {}),
                                         **({"sample": s["sample"]} if s.get("sample") else {})} for i, s in enumerate(secs)]})
    by_id = {blk["id"]: blk for blk in out_blocks}
    # the cost's parts are stored in the root's CACC: its scalar from the assembly unit
    cacc = next((u["types"]["CACC"] for u in units if "CACC" in u["types"]), None)
    for blk in out_blocks:
        for sec in blk["sections"]:
            if sec["role"] == "CACC" and cacc:
                sec["ty"] = cacc
    other = {}  # block id -> index of its "(other)" pseudo section

    def other_section(bid):
        if bid not in other:
            blk = by_id[bid]
            blk["sections"].append({"i": len(blk["sections"]), "name": "(outside sections)", "at": 0, "bytes": 0, "elem": 4, "count": 0, "ty": "bytes", "role": "",
                                    "group": "", "doc": "accesses outside every described section"})
            other[bid] = len(blk["sections"]) - 1
        return other[bid]

    # ------------------------------------------------ steps
    def raw_step(it):
        if it["kind"] != "launch":
            blk = blocks[it["block"]]
            return {"kind": it["kind"], "scope": it["scope"], "block": it["block"], "section": section_at(blk, it["at"]), "at": it["at"], "bytes": it["bytes"],
                    "sections": sections_in(blk, it["at"], it["bytes"]), **({"via": it["via"]} if it["kind"] == "upload" else {})}
        k = kernels[it["k"]]
        sargs = []
        for i, a in enumerate(it["args"]):
            pname = k["params"][i]["name"] if i < len(k["params"]) else "arg%d" % i
            if "b" in a:
                sargs.append({"param": pname, "block": a["b"], "at": a["at"], "section": section_at(blocks[a["b"]], a["at"])})
            else:
                sargs.append({"param": pname, "value": str(next(iter(a.values())))})
        access = {}
        if it["acc"] is not None:
            for a in it["acc"]:
                s = a["s"] if a["s"] >= 0 else other_section(a["block"])
                access[(a["block"], s)] = {"block": a["block"], "section": s, "src": "measured", "r": a["r"], "w": a["w"]}
        else:
            # static: pointer arguments by their const-ness, a block's columns by COL / COL_MUT
            for i, a in enumerate(it["args"]):
                if "b" not in a or i >= len(k["params"]):
                    continue
                p = k["params"][i]
                blk = by_id[a["b"]]
                if p["type"] == "bytes":
                    reads, writes = col_accesses(k["source"], unit_sources[k["unit"]])
                    names = {s["name"]: s["i"] for s in blk["sections"]}
                    for nm in reads | writes:
                        if nm in names:
                            e = access.setdefault((a["b"], names[nm]), {"block": a["b"], "section": names[nm], "src": "static", "r": None, "w": None})
                            if nm in reads:
                                e["r"] = {"pattern": "unknown"}
                            if nm in writes:
                                e["w"] = {"pattern": "unknown"}
                else:
                    s = section_at(blocks[a["b"]], a["at"])
                    if s < 0:
                        continue
                    e = access.setdefault((a["b"], s), {"block": a["b"], "section": s, "src": "static", "r": None, "w": None})
                    e["r" if p["const"] else "w"] = {"pattern": "unknown"}
        return {"kind": "launch", "kernel": it["k"], "scope": it["scope"], "launches": 1, "grid": [g for g in it["grid"] if g > 1] or [1],
                "threads": [t for t in it["threads"] if t > 1] or [1], "shared": it["shared"], "geoms": {(tuple(it["grid"]), tuple(it["threads"]))},
                "lane": it["lane"], "graph": it["graph"], "args": sargs, "access": access,
                # microseconds of the step's launches together, when the trace timed them
                "time": it.get("us"),
                # the step's launches one by one (n: the launch's place among its occurrence's, set by steps_of)
                "runs": [{"n": 0, "grid": [g for g in it["grid"] if g > 1] or [1], "threads": [t for t in it["threads"] if t > 1] or [1],
                          "args": it["args"], "time": it.get("us")}]}

    def merge(a, c):
        a["time"] = a["time"] + c["time"] if a["time"] is not None and c["time"] is not None else None
        a["launches"] += c["launches"]
        a["runs"] += c["runs"]
        a["geoms"] |= c["geoms"]
        a["graph"] = a["graph"] or c["graph"]
        for key, e in c["access"].items():
            if key in a["access"] and a["access"][key]["src"] == "measured":
                t = a["access"][key]
                t["r"], t["w"] = touch_sum(t["r"], e["r"]), touch_sum(t["w"], e["w"])
            elif key not in a["access"]:
                a["access"][key] = e

    def steps_of(occ):
        # n: an operation's place (a launch or a copy) among its occurrence's; a merged step keeps its first's
        raw = []
        for n, it in enumerate(occ):
            s = raw_step(it)
            s["n"] = n
            if s["kind"] == "launch":
                s["runs"][0]["n"] = n
            raw.append(s)
        merged = []
        for s in raw:
            if merged and s["kind"] == "launch" and merged[-1]["kind"] == "launch" and merged[-1]["kernel"] == s["kernel"] and merged[-1]["scope"] == s["scope"]:
                merge(merged[-1], s)
            else:
                merged.append(s)
        # a scope whose kernels interleave over many launches (the factor's levels): one step per kernel
        out, i = [], 0
        while i < len(merged):
            j = i
            while j < len(merged) and merged[j]["scope"] == merged[i]["scope"] and merged[j]["kind"] == "launch":
                j += 1
            run = merged[i:j]
            names = [s["kernel"] for s in run]
            if len(run) > 12 and len(set(names)) < len(names):
                first = {}
                for s in run:
                    if s["kernel"] in first:
                        merge(first[s["kernel"]], s)
                        first[s["kernel"]]["interleaved"] = True
                    else:
                        first[s["kernel"]] = s
                out.extend(first.values())
            else:
                out.extend(run)
            if j == i:
                out.append(merged[i])
                j = i + 1
            i = j
        return out

    def runs_of(runs):
        """A merged step's launches for the case: the arguments that differ between them (`vary`,
        indices into the step's args), and per launch its place among the occurrence's operations,
        its geometry, those arguments' values (a pointer: its byte offset in the block, or
        [block, offset] in another block than the first launch's) and its time."""
        first = runs[0]["args"]
        vary = [i for i in range(len(first)) if any(i >= len(r["args"]) or r["args"][i] != first[i] for r in runs)]

        def value(a, base):
            if "b" in a:
                return a["at"] if a["b"] == base.get("b") else [a["b"], a["at"]]
            return str(next(iter(a.values())))

        return {"vary": vary,
                "list": [{"n": r["n"], "grid": r["grid"], "threads": r["threads"],
                          "v": [value(r["args"][i], first[i]) if i < len(r["args"]) else None for i in vary],
                          **({"time_us": round(r["time"], 3)} if r["time"] is not None else {})} for r in runs]}

    def signature(steps):
        return tuple((s["kind"], s.get("kernel"), tuple(s["scope"]), s.get("block"), s.get("section"), s.get("launches")) for s in steps)

    # occurrences of the top-level scopes
    occs, order = {}, []
    for it in items:
        top = it["scope"][0] if it["scope"] else "(outside)"
        if top not in occs:
            occs[top] = []
            order.append(top)
    cur_top, cur = None, None
    # an occurrence ends when the top scope closes: recover the boundaries from the events
    depth, top = 0, None
    marks = []
    for e in events[lo:hi]:
        if e["e"] == "scope":
            if depth == 0:
                top = e["name"]
                marks.append(top)
            depth += 1
        elif e["e"] == "end":
            depth -= 1
    # walk the items again, cutting at each top-level scope opening
    idx = 0
    seq = []
    depth = 0
    pending = []
    flat = iter(items)
    # simpler: re-flatten per top-level scope span
    spans, depth, start = [], 0, None
    for i, e in enumerate(events[lo:hi]):
        if e["e"] == "scope":
            if depth == 0:
                start = (i, e["name"])
            depth += 1
        elif e["e"] == "end":
            depth -= 1
            if depth == 0 and start is not None:
                spans.append((start[1], lo + start[0], lo + i + 1))
                start = None
    phase_occ = {}
    phase_order = []
    for name, s, e in spans:
        b2 = Builder(events, args.repo)
        b2.graphs, b2.blocks = b.graphs, dict(b.blocks)
        its = b2.flatten(s, e)
        if name not in phase_occ:
            phase_occ[name] = []
            phase_order.append(name)
        phase_occ[name].append(steps_of(its))

    phases, sid = [], 0
    for name in sorted(phase_order, key=lambda n: (FLOW_ORDER.index(n) if n in FLOW_ORDER else len(FLOW_ORDER), phase_order.index(n))):
        variants = {}
        for steps in phase_occ[name]:
            variants.setdefault(signature(steps), []).append(steps)
        if name == "setup":
            # the setup's parts are separate occurrences: one phase of all of them,
            # its operations numbered through
            steps, at = [], 0
            for part in phase_occ[name]:
                for s in part:
                    s["n"] += at
                    for r in s.get("runs", []):
                        r["n"] += at
                    steps.append(s)
                at += sum(len(s.get("runs", [0])) for s in part)
            times, nvar = 1, 1
        else:
            best = max(variants.values(), key=len)
            steps, times, nvar = best[-1], len(phase_occ[name]), len(variants)
            # a step's time, and each of its launches': the mean over the occurrences of this sequence
            for i, s in enumerate(steps):
                took = [occ[i]["time"] for occ in best if occ[i].get("time") is not None]
                if took:
                    s["time"] = sum(took) / len(took)
                for j, r in enumerate(s.get("runs", [])):
                    took = [occ[i]["runs"][j]["time"] for occ in best if occ[i]["runs"][j]["time"] is not None]
                    if took:
                        r = s["runs"][j] = dict(r)
                        r["time"] = sum(took) / len(took)
        out_steps = []
        for s in steps:
            s = dict(s)
            s["id"] = "s%d" % sid
            sid += 1
            if s["kind"] == "launch":
                s["geoms"] = len(s["geoms"])
                # per launch, as the viewer multiplies by the launches
                took = s.pop("time", None)
                if took is not None:
                    s["time_us"] = round(took / s["launches"], 3)
                runs = s.pop("runs")
                if len(runs) > 1:
                    s["runs"] = runs_of(runs)
                acc = []
                for (bid, sec), e in s["access"].items():
                    elem = by_id[bid]["sections"][sec]["elem"] if sec < len(by_id[bid]["sections"]) else 4
                    ent = {"block": bid, "section": sec, "src": e["src"], "r": None, "w": None}
                    for rw in ("r", "w"):
                        t = e[rw]
                        if t is None:
                            continue
                        if e["src"] != "measured":
                            ent[rw] = t
                            continue
                        n, threads = max(t["n"], 1), max(t["threads"] - 1, 1)
                        count = by_id[bid]["sections"][sec]["count"] or t["unique"]
                        ent[rw] = {"n": t["n"], "bytes": t["bytes"], "unique": min(t["unique"], count) if s["launches"] > 1 else t["unique"], "lo": t["lo"], "hi": t["hi"],
                                   "threads": t["threads"], "sizes": [w for w in (1, 2, 4, 8, 16) if t["sizes"] & w],
                                   "seq": round(t["seq"] / n, 3), "same": round(t["same"] / n, 3), "adj": round(t["adjacent"] / threads, 3),
                                   "bro": round(t["shared"] / threads, 3), "pattern": pattern(t, rw == "w", max(elem, 1))}
                    acc.append(ent)
                s["access"] = acc
            out_steps.append(s)
        title, doc = SCOPE_DOCS.get(name, (name.capitalize(), ""))
        phases.append({"id": name, "title": title, "doc": doc, "times": times, "variants": nvar, "steps": out_steps})

    # ------------------------------------------------ scopes, flow, edges
    scopes = {}
    for ph in phases:
        scopes[ph["id"]] = dict(zip(("title", "doc"), SCOPE_DOCS.get(ph["id"], (ph["id"].capitalize(), ""))))
        for s in ph["steps"]:
            for d in range(2, len(s["scope"]) + 1):
                path = "/".join(s["scope"][:d])
                if path not in scopes:
                    t = SCOPE_DOCS.get(path) or SUB_DOCS.get(s["scope"][d - 1]) or (s["scope"][d - 1], "")
                    scopes[path] = {"title": t[0], "doc": t[1]}
    names = [p["id"] for p in phases]
    flow = [{"phase": n} for n in ("setup", "compute", "start") if n in names]
    body = []
    if "attempt" in names:
        body.append({"loop": "attempts until a step is accepted", "body": [{"phase": "attempt"}]})
    if "linearize" in names:
        body.append({"phase": "linearize"})
    if body:
        flow.append({"loop": "LM iterations", "body": body})
    flow += [{"phase": n} for n in names if n not in ("setup", "compute", "start", "attempt", "linearize", "finish")]
    if "finish" in names:
        flow.append({"phase": "finish"})

    # dataflow: the last writer of each section, walking the phases in flow order, the loop twice
    by_phase = {p["id"]: p for p in phases}
    walk = [n for n in ("setup", "compute", "start") if n in names] + 2 * [n for n in ("attempt", "linearize") if n in names] + \
           [n for n in names if n not in ("setup", "compute", "start", "attempt", "linearize")]
    writer, edges, seen = {}, [], set()

    def edge(frm, to, bid, sec):
        key = (frm, to, bid, sec)
        if frm != to and key not in seen:
            seen.add(key)
            edges.append({"from": frm, "to": to, "block": bid, "section": sec})

    for n in walk:
        for s in by_phase[n]["steps"]:
            if s["kind"] == "upload":
                for sec in s["sections"]:
                    writer[(s["block"], sec)] = s["id"]
            elif s["kind"] == "download":
                for sec in s["sections"]:
                    if (s["block"], sec) in writer:
                        edge(writer[(s["block"], sec)], s["id"], s["block"], sec)
            else:
                for a in s["access"]:
                    key = (a["block"], a["section"])
                    if a["r"] is not None and key in writer:
                        edge(writer[key], s["id"], a["block"], a["section"])
                for a in s["access"]:
                    if a["w"] is not None:
                        writer[(a["block"], a["section"])] = s["id"]
    for e in edges:
        if e["from"] == "host":
            continue

    # ------------------------------------------------ findings
    findings = []
    for bid, sec in other.items():
        findings.append({"level": "warn", "title": "%s: accesses outside its described sections" % by_id[bid]["name"],
                         "text": "Some kernels touch bytes of this block that no section describes.", "refs": [{"kind": "section", "block": bid, "section": sec}]})
    def human(n):
        for unit in ("B", "KB", "MB", "GB"):
            if n < 1024 or unit == "GB":
                return ("%d %s" if unit == "B" else "%.1f %s") % (n, unit)
            n /= 1024.0

    # what each section saw over the solve's phases
    seen_r, seen_w, seen_up, seen_down = set(), set(), set(), set()
    for ph in phases:
        for s in ph["steps"]:
            if s["kind"] == "upload":
                seen_up.update((s["block"], sec) for sec in s["sections"])
            elif s["kind"] == "download":
                seen_down.update((s["block"], sec) for sec in s["sections"])
            else:
                for a in s["access"]:
                    if a["r"] is not None:
                        seen_r.add((a["block"], a["section"]))
                    if a["w"] is not None:
                        seen_w.add((a["block"], a["section"]))
    if b.probe:
        for blk in out_blocks:
            # an upload covers the sections from its start on: the block's first upload reaches them all
            for sec in blk["sections"]:
                key = (blk["id"], sec["i"])
                uploaded = key in seen_up
                if sec["bytes"] == 0 or key in seen_r or key in seen_down:
                    continue
                ref = [{"kind": "section", "block": blk["id"], "section": sec["i"]}]
                where = "%s.%s (%s)" % (blk["name"], sec["name"], human(sec["bytes"]))
                if key in seen_w:
                    findings.append({"level": "warn", "title": "%s is written by kernels and never read" % where,
                                     "text": "No kernel reads it and the host does not read it back in this solve.", "refs": ref})
                elif uploaded:
                    findings.append({"level": "info", "title": "%s is uploaded and never read by a kernel" % where,
                                     "text": "The host uploads it and no kernel of this solve reads it.", "refs": ref})
                else:
                    findings.append({"level": "info", "title": "%s is never touched" % where,
                                     "text": "Allocated; no upload, no kernel access and no read-back in this solve.", "refs": ref})
        # index bytes against value bytes, per step
        index_types = ("u32", "u64")
        for ph in phases:
            for s in ph["steps"]:
                if s["kind"] != "launch":
                    continue
                idx = val = 0
                for a in s["access"]:
                    sec = by_id[a["block"]]["sections"][a["section"]]
                    moved = (a["r"]["bytes"] if a["r"] else 0) + (a["w"]["bytes"] if a["w"] else 0)
                    if sec["ty"] in index_types:
                        idx += moved
                    else:
                        val += moved
                if idx > 2 * val and idx > 256 * 1024:
                    findings.append({"level": "info", "title": "%s (%s) reads %s of index words for %s of values" % (s["kernel"], ph["id"], human(idx), human(val)),
                                     "text": "Per launch%s. Index and plan words read per thread: candidates for kernel arguments, per-block hoisting or a denser layout."
                                     % (" (sum of %d launches)" % s["launches"] if s["launches"] > 1 else ""), "refs": [{"kind": "step", "id": s["id"]}, {"kind": "kernel", "name": s["kernel"]}]})
        # access widths that are not the section's element
        flagged = set()
        for ph in phases:
            for s in ph["steps"]:
                if s["kind"] != "launch":
                    continue
                for a in s["access"]:
                    sec = by_id[a["block"]]["sections"][a["section"]]
                    for rw in ("r", "w"):
                        t = a[rw]
                        if t and t.get("sizes") and sec["bytes"] and any(w != sec["elem"] for w in t["sizes"]) and (a["block"], a["section"]) not in flagged:
                            flagged.add((a["block"], a["section"]))
                            findings.append({"level": "warn", "title": "%s.%s: %d-byte elements accessed %s bytes wide" % (by_id[a["block"]]["name"], sec["name"], sec["elem"], "/".join(map(str, t["sizes"]))),
                                             "text": "Stored type %s, role %s; first seen in %s." % (sec["ty"], sec["role"] or "-", s["kernel"]),
                                             "refs": [{"kind": "section", "block": a["block"], "section": a["section"]}, {"kind": "kernel", "name": s["kernel"]}]})
    # index words a kernel could compute from the position, and those stored more than once
    indexed = [(blk, sec) for blk in out_blocks for sec in blk["sections"] if sec.get("content")]
    copies = copy_groups(out_blocks)
    if indexed:
        words = sum(sec["content"]["n"] for _, sec in indexed)
        from_pos = sum(sec["content"]["n"] * sec["computable"] for _, sec in indexed)
        whole = [(blk, sec) for blk, sec in indexed if sec["computable"] == 1]
        part = [(blk, sec) for blk, sec in indexed if 0 < sec["computable"] < 1]
        text = ("Of %s index words in %d sections, %s (%d%%) are computable from their position: one value, base + i * stride, a few stretches of that, "
                "or such fields of records. %d sections wholly, %d in part.") % (
            "{:,}".format(words), len(indexed), "{:,}".format(round(from_pos)), round(100.0 * from_pos / words), len(whole), len(part))
        read_by = {}
        for ph in phases:
            if ph["id"] not in ("attempt", "linearize"):
                continue
            for st in ph["steps"]:
                for a in st.get("access", []):
                    if a["r"] and a["r"].get("bytes"):
                        read_by[(a["block"], a["section"])] = read_by.get((a["block"], a["section"]), 0) + a["r"]["bytes"]
        total = sum(read_by.get((blk["id"], sec["i"]), 0) for blk, sec in indexed)
        if b.probe and total:
            # a record's fields are taken as read alike
            comp = sum(read_by.get((blk["id"], sec["i"]), 0) * sec["computable"] for blk, sec in indexed)
            text += " In one attempt and one linearization the kernels read %s of index words, %s of it (%d%%) computable from the position." % (
                human(total), human(comp), round(100.0 * comp / total))
        findings.append({"level": "info", "title": "%d%% of the index words are computable from the position" % round(100.0 * from_pos / words), "text": text,
                         "refs": [{"kind": "section", "block": blk["id"], "section": sec["i"]} for blk, sec in whole + part]})
    if copies:
        def member(m):
            sec = by_id[m["block"]]["sections"][m["section"]]
            return by_id[m["block"]]["name"] + "." + sec["name"] + (" field %d" % m["field"] if m["field"] >= 0 else "")

        def group(g):
            # the member with the smallest divisor first; the others are its multiples
            members = sorted(g["members"], key=lambda m: m["scale"])
            base = members[0]["scale"]
            names = [member(m) + ("" if m["scale"] == base else " (x %d)" % (m["scale"] // base) if m["scale"] % base == 0 else " (x %d/%d)" % (m["scale"], base))
                     for m in members]
            return "%s: %s words" % (" = ".join(names), "{:,}".format(g["n"]))
        repeated = sum(g["n"] * (len(g["members"]) - 1) for g in copies)
        findings.append({"level": "info", "title": "%d groups of index sections hold the same words" % len(copies),
                         "text": "Irregular index sequences that are equal, or a multiple of one another (an offset that is 9 times an index): %s. Storing each once saves %s words (%s)." % (
                             "; ".join(group(g) for g in sorted(copies, key=lambda g: -g["n"] * (len(g["members"]) - 1))), "{:,}".format(repeated), human(4 * repeated)),
                         "refs": [{"kind": "section", "block": m["block"], "section": m["section"]} for g in copies for m in g["members"]]})
    for ph in phases:
        if ph["variants"] > 1:
            findings.append({"level": "info", "title": "phase %s ran %d different step sequences" % (ph["id"], ph["variants"]),
                             "text": "The steps shown are the most frequent sequence's (%d occurrences in all)." % ph["times"], "refs": []})

    wide = "f64"
    for u in units:
        wide = u["types"].get("W", wide)
    commit = ""
    try:
        commit = subprocess.run(["git", "-C", args.repo, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        pass
    dev = b.device
    notes = ["solve %d of %d in the trace" % (pick + 1, len(bounds))]
    if dev.get("timed"):
        notes.append("launches timed on the device, one at a time: a kernel's time is its own; the solve's wall time, with its overlaps and its host work, is not their sum")
    if b.probe:
        notes.append("accesses measured on the CPU device (clang, no vectorization): counts are of the host compiler's code")
    else:
        notes.append("no measured accesses: reads and writes are from the kernels' parameter types and COL macros")
    return {"schema": 1, "id": args.id, "title": args.title or args.id,
            "meta": {"problem": args.problem or "", "model": args.model or "", "runtime": dev.get("runtime", ""), "device": dev.get("name", ""),
                     "target": dev.get("target", ""), "warp": dev.get("warp", 0), "wide": wide, "probe": b.probe, "timed": bool(dev.get("timed")), "commit": commit,
                     "generated": datetime.date.today().isoformat(), "notes": notes},
            "units": units, "kernels": kernels, "blocks": out_blocks, "copies": copies, "scopes": scopes, "phases": phases, "flow": flow, "edges": edges, "findings": findings}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("--id", required=True)
    ap.add_argument("--title")
    ap.add_argument("--problem")
    ap.add_argument("--model")
    ap.add_argument("--solve", type=int, help="which solve of the trace, from 0 (default: the one with the most attempts)")
    ap.add_argument("--out", default=os.path.join(here, "..", "data"))
    ap.add_argument("--repo", default=os.path.join(here, "..", "..", ".."))
    ap.add_argument("--patterns-from", metavar="CASE.js", help="a probed case of the same model: its measured patterns label this case's unmeasured accesses")
    args = ap.parse_args()
    case = build_case(args)
    if args.patterns_from and not case["meta"]["probe"]:
        borrowed, unseen, unknown = borrow_patterns(case, load_case(args.patterns_from))
        print("patterns from %s: %d borrowed, %d not seen when probed, %d of kernels it does not launch" % (os.path.basename(args.patterns_from), borrowed, unseen, unknown))
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, args.id + ".js")
    write_case(case, path)
    # the menu: the probed cases first (they hold the most), then by bench, problem size and model; the sample last
    def menu_key(n):
        parts = n[:-3].split("-")
        size = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        return ("-cpu" not in n, parts[0], size, n)
    files = sorted((os.path.basename(p) for p in glob.glob(os.path.join(args.out, "*.js")) if os.path.basename(p) not in ("cases.js", "sample.js")), key=menu_key)
    with open(os.path.join(args.out, "cases.js"), "w") as f:
        f.write("// The case files the viewer loads, in menu order (gen/devmap.py rewrites this).\n")
        f.write("window.DEVMAP_FILES = %s;\n" % json.dumps(files + ["sample.js"]))
    n_steps = sum(len(p["steps"]) for p in case["phases"])
    print("%s: %d phases, %d steps, %d blocks, %d kernels, %d edges, %d bytes" % (path, len(case["phases"]), n_steps, len(case["blocks"]), len(case["kernels"]),
                                                                        len(case["edges"]), os.path.getsize(path)))


if __name__ == "__main__":
    main()
