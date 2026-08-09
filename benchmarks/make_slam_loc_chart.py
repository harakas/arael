# Generates the SLAM + localization bar chart committed as
# chart-slam-loc-light.svg / chart-slam-loc-dark.svg and embedded in the
# top-level README.md and src/lib.rs.
#
# Two panels side by side -- landmark SLAM at 1200 poses on an Apple M4 Pro
# (benchmarks/slam README, 1200-pose figure-8 table) and localization at 60
# poses on a Raspberry Pi 5 (benchmarks/loc README, Pi 5 table) -- over two
# rows: ms per solver iteration, then peak memory for those same runs. One
# bar per system showing its best validated configuration, all arael rows
# shown. Each row is ordered by its own metric, so a system does not sit at
# the same height in both. Update the data from the results tables after
# re-running the benchmarks, then run:
#
#   python3 make_slam_loc_chart.py
#
# Pure stdlib, no dependencies.

# Per panel: (title, value decimals, [(label, full_iter_ms, first_iter_ms, kind)])
# full-iter is one complete iteration (t(2 iters) - t(1 iter), setup cancelled).
# first-iter is that same iteration plus the setup paid once. Their difference is
# the setup, which the second chart draws.
# kind: "arael" solid blue bar, "other" neutral bar.
PANELS = [
    # 2026-08-04, min of 8 rounds (benchmarks/slam README, 1200-pose figure-8
    # table). Best validated configuration per system: Ceres is sparse_cholesky
    # (its sparse_schur is slower on this scene, and iterative_schur is inexact
    # and misses the gate), SymForce is f64. The arael CG rows are inexact too,
    # and have no full-iter to plot.
    ("Landmark SLAM -- 1200 poses, 21.6k params (Apple M4 Pro)", 1, [
        ("arael (f32)", 203.54, 574.21, "arael"),
        ("arael (f64)", 312.93, 725.00, "arael"),
        ("g2o (LM)", 1075.42, 1516.79, "other"),
        ("Ceres (LM)", 1162.71, 1430.46, "other"),
        ("GTSAM (LM)", 1322.54, 2544.32, "other"),
        ("factrs (LM)", 1401.12, 1764.26, "other"),
        ("SymForce (f64)", 1695.11, 2384.76, "other"),
    ]),
    # 2026-07-26, min of 32 rounds (benchmarks/loc README, Pi 5 table). Best
    # validated configuration per system: Ceres is sparse_cholesky (a fixed
    # landmark map leaves nothing to marginalize, and iterative_schur is inexact),
    # SymForce is f64.
    ("Localization -- 60 poses, 360 params (Raspberry Pi 5)", 2, [
        ("arael (f32)", 1.02, 1.0, "arael"),
        ("arael (f64)", 1.06, 1.1, "arael"),
        ("SymForce (f64)", 1.34, 16.4, "other"),
        ("g2o (LM)", 4.04, 7.8, "other"),
        ("Ceres (LM)", 5.34, 11.4, "other"),
        ("factrs (LM)", 13.06, 20.5, "other"),
        ("GTSAM (LM)", 13.74, 16.3, "other"),
    ]),
]

# The second row: peak MB from the same two tables, the same systems in the
# same configurations, ordered by memory rather than by time.
# Per panel: (title, value decimals, [(label, peak_mb, kind)])
MEM_PANELS = [
    ("Landmark SLAM -- peak process memory", 1, [
        ("arael (f32)", 216.4, "arael"),
        ("arael (f64)", 298.2, "arael"),
        ("Ceres (LM)", 461.0, "other"),
        ("g2o (LM)", 729.8, "other"),
        ("SymForce (f64)", 838.7, "other"),
        ("factrs (LM)", 865.0, "other"),
        ("GTSAM (LM)", 4691.3, "other"),
    ]),
    ("Localization -- peak process memory", 1, [
        ("arael (f32)", 4.3, "arael"),
        ("arael (f64)", 4.9, "arael"),
        ("g2o (LM)", 10.1, "other"),
        ("Ceres (LM)", 11.5, "other"),
        ("factrs (LM)", 12.1, "other"),
        ("GTSAM (LM)", 15.4, "other"),
        ("SymForce (f64)", 20.5, "other"),
    ]),
]

class Axis:
    """Value to a fraction of the plot width, plus the ticks to draw.

    Linear by default. With `break_at` set, values above it are compressed
    logarithmically into the last `tail` of the width. That is for a panel
    with one outlier: GTSAM's SLAM peak is 3x the next system and 15x arael's,
    and on a linear axis it flattens every other bar into the left margin.
    """

    def __init__(self, x_max, tick, break_at=None, tail=0.28, break_ticks=()):
        self.x_max, self.tick = x_max, tick
        self.break_at, self.tail, self.break_ticks = break_at, tail, break_ticks

    def pos(self, v):
        """Fraction of the plot width, 0 to 1."""
        if self.break_at is None:
            return v / self.x_max
        head = 1.0 - self.tail
        if v <= self.break_at:
            return v / self.break_at * head
        import math
        span = math.log(self.x_max / self.break_at)
        return head + math.log(v / self.break_at) / span * self.tail

    def ticks(self):
        """(value, fraction, is_break) up the axis, in drawing order."""
        out, t = [], 0.0
        top = self.break_at if self.break_at is not None else self.x_max
        while t <= top + 1e-9:
            out.append((t, self.pos(t), self.break_at is not None and t == top))
            t += self.tick
        for t in self.break_ticks:
            out.append((t, self.pos(t), False))
        return out


# Axis per panel, per chart: the two charts plot different quantities, so they do
# not share a scale. In PANELS order.
AXES = {
    "iter":  [Axis(1800.0, 600.0), Axis(16.0, 4.0)],
    "setup": [Axis(2700.0, 900.0), Axis(25.0, 5.0)],
    # SLAM memory is linear to 1 GB and logarithmic above, so the field is
    # readable against arael without dropping GTSAM's bar off the panel.
    "mem":   [Axis(4800.0, 250.0, break_at=1000.0, break_ticks=(2000.0, 4800.0)),
              Axis(24.0, 6.0)],
}

# The two charts. "iter" is the front-page one: one bar, the durable cross-system
# number. "setup" decomposes the first iteration into that same iteration plus the
# one-time setup, which is a busier read -- it lives in benchmarks/loc/README.md,
# not on the front page.
CHARTS = {
    "iter": {
        "file": "slam-loc",
        "value_w": 34,
        "mem_row": True,
        "title": "Landmark SLAM and localization: time per iteration and peak memory",
        "subtitle": ("Landmark SLAM on a desktop core, fixed-map localization on "
                     "an edge board; single thread, best validated configuration "
                     "per system. Lower is better."),
        "foot": [
            ("Time excludes setup -- assembly, ordering and symbolic "
             "factorization -- which every system pays once, during its first "
             "iteration."),
            ("Peak memory is the process high-water mark (VmHWM), each solver "
             "measured in a process of its own. Each row is ordered by its own "
             "metric."),
            ("The SLAM memory axis is linear to 1 GB and logarithmic past the "
             "dashed line, so one outlier does not flatten the rest."),
        ],
    },
    "setup": {
        "file": "slam-loc-setup",
        "value_w": 76,
        "title": ("Landmark SLAM and localization: per-iteration cost and "
                  "one-time setup"),
        "subtitle": ("Solid: one complete iteration. Faded: the setup, paid once. "
                     "Together they are what the first iteration costs."),
        "foot": [
            ("Setup is assembly, ordering and symbolic factorization: done once, "
             "reused by every later iteration. arael's band solver has almost "
             "none to do."),
        ],
    },
}

# Appended to both charts.
FOOT = [
    ("Every bar reaches its problem's common optimum, cross-validated "
     "against all systems."),
    ("arael factorizes the whole SLAM system under nested dissection and "
     "solves the localization panel with its band solver."),
]

THEMES = {
    "light": {
        "surface": "#fcfcfb", "border": "#e1e0d9", "grid": "#e1e0d9",
        "ink": "#0b0b0b", "secondary": "#5f5e58", "muted": "#85847c",
        "arael": "#2a78d6", "other": "#8f8e86",
    },
    "dark": {
        "surface": "#1a1a19", "border": "#2c2c2a", "grid": "#2c2c2a",
        "ink": "#ffffff", "secondary": "#c0bfb8", "muted": "#908f88",
        "arael": "#3987e5", "other": "#77766f",
    },
}

FONT = ("ui-sans-serif, system-ui, -apple-system, 'Segoe UI', "
        "Helvetica, Arial, sans-serif")

W = 880
MARGIN = 18
# The levers on the whitespace. The panels split whatever the canvas leaves, so
# COL_GAP alone cannot tighten anything -- the panels just absorb it. What the eye
# reads as the gap between the panels is COL_GAP plus the slack in the columns on
# either side of it: LABEL_W beyond its longest row label, VALUE_W beyond its
# longest value. The plot's right edge is likewise the canvas less MARGIN and
# VALUE_W. So the columns are cut to what their text actually needs, and the rest
# goes to the bars.
COL_GAP = 8
PANEL_W = (W - 2 * MARGIN - COL_GAP) // 2   # 418
LABEL_W = 92    # row labels, right-aligned; the longest ("SymForce (f64)") is 81
# VALUE_W (the room after a bar for its value) is per chart: "160.4" needs less
# than "135.1 + 98.5". PLOT_W follows from it.
BAR_H = 12
PITCH = 19
PANEL_TITLE_H = 20
AXIS_H = 20
ROWS = max(len(rows) for _, _, rows in PANELS + MEM_PANELS)
PANEL_H = PANEL_TITLE_H + ROWS * PITCH + AXIS_H
HEADER_H = 58
ROW_GAP = 22    # between the time band and the memory band


def bar_path(x0, y, w, h, r):
    """Bar with the data end rounded, baseline end flat."""
    return (f"M{x0},{y} L{x0 + w - r},{y} Q{x0 + w},{y} {x0 + w},{y + r} "
            f"L{x0 + w},{y + h - r} Q{x0 + w},{y + h} {x0 + w - r},{y + h} "
            f"L{x0},{y + h} Z")


def render_panel(s, c, px, py, title, axis, decimals, rows,
                 with_setup, plot_w, unit):
    plot_x = px + LABEL_W
    plot_top = py + PANEL_TITLE_H
    plot_h = ROWS * PITCH  # common height so the two panels' axes align
    s.append(f'<text x="{px}" y="{py + 12}" font-size="12.5" '
             f'font-weight="600" fill="{c["ink"]}">{title}</text>')
    # gridlines + ticks
    ticks = axis.ticks()
    for t, frac, is_break in ticks:
        x = plot_x + frac * plot_w
        # The break is where the scale changes, so it is drawn as a broken
        # line rather than another gridline.
        dash = ' stroke-dasharray="2 2"' if is_break else ""
        s.append(f'<line x1="{x:.1f}" y1="{plot_top}" x2="{x:.1f}" '
                 f'y2="{plot_top + plot_h + 3}" stroke="{c["grid"]}" '
                 f'stroke-width="1"{dash}/>')
        label = f"{t:.0f} {unit}" if t == ticks[-1][0] else f"{t:.0f}"
        s.append(f'<text x="{x:.1f}" y="{plot_top + plot_h + 15}" '
                 f'font-size="10" text-anchor="middle" '
                 f'fill="{c["muted"]}">{label}</text>')
    for i, row in enumerate(rows):
        # Time rows carry (label, full, first, kind); memory rows have no
        # second segment and drop the middle field.
        label, full, kind = row[0], row[1], row[-1]
        first = row[2] if len(row) == 4 else full
        is_arael = kind.startswith("arael")
        y = plot_top + i * PITCH + (PITCH - BAR_H) / 2
        ty = y + BAR_H / 2 + 3.5
        weight = ' font-weight="600"' if is_arael else ""
        name_ink = c["ink"] if is_arael else c["secondary"]
        s.append(f'<text x="{plot_x - 8}" y="{ty:.1f}" font-size="11.5" '
                 f'text-anchor="end"{weight} fill="{name_ink}">{label}</text>')
        fill = c["arael"] if is_arael else c["other"]
        w = axis.pos(full) * plot_w
        s.append(f'<path d="{bar_path(plot_x, y, w, BAR_H, 3)}" fill="{fill}"/>')
        end, value = plot_x + w, f"{full:.{decimals}f}"
        if with_setup:
            # Setup is a measured difference, so on a system that has none it can
            # land marginally below zero (arael's band solver on the Pi 5).
            setup = max(0.0, first - full)
            w2 = (axis.pos(first) - axis.pos(full)) * plot_w
            if w2 > 0.5:
                x2 = plot_x + w + 2   # 2px surface gap between the segments
                s.append(f'<path d="{bar_path(x2, y, w2, BAR_H, 3)}" '
                         f'fill="{fill}" fill-opacity="0.38"/>')
            end = plot_x + w + 2 + w2
            value = f"{full:.{decimals}f} + {setup:.{decimals}f}"
        s.append(f'<text x="{end + 6:.1f}" y="{ty:.1f}" '
                 f'font-size="10.5"{weight} fill="{c["ink"]}">{value}</text>')



def arael_version():
    """Read the workspace version, so the stamp cannot drift from the code.

    A chart travels: it gets copied into slides, issues and blog posts, far
    away from the README that explains it. Without a version on the image
    itself, a reader has no way to know which arael produced the numbers.
    """
    import os, re
    here = os.path.dirname(os.path.abspath(__file__))
    root = here
    for _ in range(4):
        manifest = os.path.join(root, "Cargo.toml")
        if os.path.exists(manifest):
            with open(manifest) as f:
                text = f.read()
            if '[package]\nname = "arael"' in text or 'name = "arael"' in text:
                m = re.search(r'^version = "([^"]+)"', text, re.M)
                if m:
                    return m.group(1)
        root = os.path.dirname(root)
    raise SystemExit("cannot find the arael version in any parent Cargo.toml")


def render(theme, chart):
    c = THEMES[theme]
    cfg = CHARTS[chart]
    foot = cfg["foot"] + FOOT
    plot_w = PANEL_W - LABEL_W - cfg["value_w"]
    # Bands drawn top to bottom: (panels, axis key, unit, draws a setup segment)
    bands = [(PANELS, chart, "ms", chart == "setup")]
    if cfg.get("mem_row"):
        bands.append((MEM_PANELS, "mem", "MB", False))
    band_y = [HEADER_H + i * (PANEL_H + ROW_GAP) for i in range(len(bands))]
    foot_y = band_y[-1] + PANEL_H + 18
    height = foot_y + len(foot) * 14 + 10

    s = []
    s.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" '
             f'height="{height}" viewBox="0 0 {W} {height}" '
             f'font-family="{FONT}">')
    s.append(f'<rect x="0.5" y="0.5" width="{W - 1}" height="{height - 1}" '
             f'rx="8" fill="{c["surface"]}" stroke="{c["border"]}"/>')
    s.append(f'<text x="{MARGIN}" y="30" font-size="15" font-weight="600" '
             f'fill="{c["ink"]}">{cfg["title"]}</text>')
    # The version that produced these numbers, on the image itself: a chart
    # gets copied out of the README and has to carry its own provenance.
    s.append(f'<text x="{W - MARGIN}" y="30" font-size="11.5" text-anchor="end" '
             f'fill="{c["muted"]}">arael {arael_version()}</text>')
    s.append(f'<text x="{MARGIN}" y="48" font-size="11.5" '
             f'fill="{c["secondary"]}">{cfg["subtitle"]}</text>')

    for bi, (panels, axis_key, unit, with_setup) in enumerate(bands):
        for k, (title, decimals, rows) in enumerate(panels):
            px = MARGIN + k * (PANEL_W + COL_GAP)
            render_panel(s, c, px, band_y[bi], title, AXES[axis_key][k],
                         decimals, rows, with_setup, plot_w, unit)

    for i, line in enumerate(foot):
        s.append(f'<text x="{MARGIN}" y="{foot_y + i * 14}" font-size="10.5" '
                 f'fill="{c["muted"]}">{line}</text>')
    s.append("</svg>")
    return "\n".join(s) + "\n"


def main():
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "charts", f"v{arael_version()}")
    os.makedirs(out, exist_ok=True)
    for chart, cfg in CHARTS.items():
        for theme in THEMES:
            path = os.path.join(out, f"{cfg['file']}-{theme}.svg")
            with open(path, "w") as f:
                f.write(render(theme, chart))
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
# Charts are written to benchmarks/charts/v<version>/ -- a new directory each
# release. The path carries the version, so a chart URL never changes meaning:
# crates.io rewrites the README's relative paths against the default branch, and
# a versioned path still resolves to the numbers that version shipped with.
#
# The four charts at the old unversioned paths (benchmarks/chart-slam-loc-*.svg,
# benchmarks/pgo/chart-*.svg) are FROZEN. Crates published up to 0.7.0 reference
# them and their READMEs can never be changed. Do not regenerate them.

