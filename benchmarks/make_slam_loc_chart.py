# Generates the SLAM + localization bar chart committed as
# chart-slam-loc-light.svg / chart-slam-loc-dark.svg and embedded in the
# top-level README.md and src/lib.rs.
#
# Two panels side by side, both on ms per solver iteration: landmark
# SLAM at 300 poses on an Apple M4 Pro (benchmarks/slam README, 300-pose
# table) and fixed-map localization at 60 poses on a Raspberry Pi 5
# (benchmarks/loc README, Pi 5 table). One bar per system showing its
# best validated configuration, all arael rows shown. Update the data
# from the results tables after re-running the benchmarks, then run:
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
    # 2026-07-26, min of 16 rounds (benchmarks/slam README, 300-pose table). Best
    # validated configuration per system: Ceres is sparse_schur (iterative_schur
    # is inexact and misses the gate), SymForce is f64.
    ("Landmark SLAM -- 300 poses, 5.4k params (Apple M4 Pro)", 1, [
        ("arael (f32)", 31.04, 39.29, "arael"),
        ("arael (f64)", 44.16, 53.55, "arael"),
        ("g2o (LM)", 60.29, 111.83, "other"),
        ("Ceres (LM)", 81.87, 157.95, "other"),
        ("factrs (LM)", 120.73, 173.36, "other"),
        ("SymForce (f64)", 132.51, 224.34, "other"),
        ("GTSAM (LM)", 153.17, 167.10, "other"),
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

# Axis per panel, per chart: the two charts plot different quantities, so they do
# not share a scale. (x_max, tick), in PANELS order.
AXES = {
    "iter":  [(170.0, 50.0), (16.0, 4.0)],
    "setup": [(240.0, 60.0), (25.0, 5.0)],
}

# The two charts. "iter" is the front-page one: one bar, the durable cross-system
# number. "setup" decomposes the first iteration into that same iteration plus the
# one-time setup, which is a busier read -- it lives in benchmarks/loc/README.md,
# not on the front page.
CHARTS = {
    "iter": {
        "file": "slam-loc",
        "value_w": 34,
        "title": "Landmark SLAM and localization: time per solver iteration",
        "subtitle": ("Landmark SLAM on a desktop core, fixed-map localization on "
                     "an edge board; single thread, best validated configuration "
                     "per system. Lower is better."),
        "foot": [
            ("Excludes setup -- assembly, ordering and symbolic factorization -- "
             "which every system pays once, during its first iteration."),
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
    ("arael solves the SLAM panel with its Schur solver, the localization "
     "panel with its band solver."),
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
ROWS = max(len(rows) for _, _, rows in PANELS)
PANEL_H = PANEL_TITLE_H + ROWS * PITCH + AXIS_H
HEADER_H = 58


def bar_path(x0, y, w, h, r):
    """Bar with the data end rounded, baseline end flat."""
    return (f"M{x0},{y} L{x0 + w - r},{y} Q{x0 + w},{y} {x0 + w},{y + r} "
            f"L{x0 + w},{y + h - r} Q{x0 + w},{y + h} {x0 + w - r},{y + h} "
            f"L{x0},{y + h} Z")


def render_panel(s, c, px, py, title, x_max, tick, decimals, rows,
                 with_setup, plot_w):
    plot_x = px + LABEL_W
    plot_top = py + PANEL_TITLE_H
    plot_h = ROWS * PITCH  # common height so the two panels' axes align
    s.append(f'<text x="{px}" y="{py + 12}" font-size="12.5" '
             f'font-weight="600" fill="{c["ink"]}">{title}</text>')
    # gridlines + ticks
    t = 0.0
    while t <= x_max + 1e-9:
        x = plot_x + t / x_max * plot_w
        s.append(f'<line x1="{x:.1f}" y1="{plot_top}" x2="{x:.1f}" '
                 f'y2="{plot_top + plot_h + 3}" stroke="{c["grid"]}" '
                 f'stroke-width="1"/>')
        label = f"{t:.0f} ms" if t + tick > x_max + 1e-9 else f"{t:.0f}"
        s.append(f'<text x="{x:.1f}" y="{plot_top + plot_h + 15}" '
                 f'font-size="10" text-anchor="middle" '
                 f'fill="{c["muted"]}">{label}</text>')
        t += tick
    for i, (label, full, first, kind) in enumerate(rows):
        is_arael = kind.startswith("arael")
        y = plot_top + i * PITCH + (PITCH - BAR_H) / 2
        ty = y + BAR_H / 2 + 3.5
        weight = ' font-weight="600"' if is_arael else ""
        name_ink = c["ink"] if is_arael else c["secondary"]
        s.append(f'<text x="{plot_x - 8}" y="{ty:.1f}" font-size="11.5" '
                 f'text-anchor="end"{weight} fill="{name_ink}">{label}</text>')
        fill = c["arael"] if is_arael else c["other"]
        w = full / x_max * plot_w
        s.append(f'<path d="{bar_path(plot_x, y, w, BAR_H, 3)}" fill="{fill}"/>')
        end, value = plot_x + w, f"{full:.{decimals}f}"
        if with_setup:
            # Setup is a measured difference, so on a system that has none it can
            # land marginally below zero (arael's band solver on the Pi 5).
            setup = max(0.0, first - full)
            w2 = setup / x_max * plot_w
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
    foot_y = HEADER_H + PANEL_H + 18
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

    for k, (title, decimals, rows) in enumerate(PANELS):
        px = MARGIN + k * (PANEL_W + COL_GAP)
        x_max, tick = AXES[chart][k]
        render_panel(s, c, px, HEADER_H, title, x_max, tick, decimals, rows,
                     chart == "setup", plot_w)

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

