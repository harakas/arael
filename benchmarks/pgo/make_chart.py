# Generates the benchmark bar chart committed as chart-light.svg /
# chart-dark.svg and embedded in the top-level README.md and src/lib.rs.
#
# A 2x2 small-multiple assembly: one panel per dataset, all on the same
# metric (ms per solver step, the durable cross-system number -- see
# README), one bar per system showing its best validated configuration
# (best total time among configurations that pass both validation
# gates; the ms/step shown is that configuration's). tiny-solver is
# omitted from the bars for scale (6-9x arael) and reported in the
# footnote instead. Update the data from the results tables after
# re-running the benchmark, then run:
#
#   python3 make_chart.py
#
# Pure stdlib, no dependencies.

# Per panel: (title, [ (label, full_iter_ms, first_iter_ms, kind) ]).
# full-iter is one complete iteration (t(2 iters) - t(1 iter), setup cancelled).
# first-iter is that same iteration plus the setup paid once. Their difference
# is the setup, which the second chart draws. 2026-08-01, min of 32 rounds; one
# row per system, its best validated configuration by total time.
# kind: "arael" solid blue, "other" neutral, "arael*" adds a star to the value.
# full_iter None -> italic text row (did not converge).
PANELS = [
    ("M3500 (2D, 10.5k params)", [
        ("arael (f32)", 1.55, 3.68, "arael"),
        ("arael (f64)", 1.87, 4.02, "arael"),
        ("SymForce (f32)", 3.32, 18.15, "other"),
        ("g2o (GN)", 3.48, 7.42, "other"),
        ("Ceres (LM)", 4.72, 12.35, "other"),
        ("factrs (GN)", 6.19, 12.12, "other"),
        ("GTSAM (GN)", 13.63, 14.08, "other"),
    ]),
    ("city10000 (2D, 30k params)", [
        ("arael (f32)", 7.51, 15.81, "arael"),
        ("arael (f64)", 9.58, 18.04, "arael"),
        ("g2o (GN)", 17.31, 32.67, "other"),
        ("SymForce (f64)", 22.23, 92.28, "other"),
        ("Ceres (LM)", 22.30, 48.00, "other"),
        ("factrs (GN)", 22.97, 45.47, "other"),
        ("GTSAM", None, None, "other"),
    ]),
    ("sphere2500 (3D, 15k params)", [
        ("arael (f32)", 11.12, 15.65, "arael"),
        ("arael (f64)", 16.34, 20.97, "arael"),
        ("g2o (LM)", 19.27, 23.82, "other"),
        ("Ceres (LM)", 23.52, 35.85, "other"),
        ("factrs (GN)", 25.99, 44.68, "other"),
        ("GTSAM (GN)", 28.53, 28.29, "other"),
        ("SymForce (f32)", 73.74, 96.50, "other"),
    ]),
    ("parking-garage (3D, 10k params)", [
        ("arael (f32)", 3.65, 7.03, "arael*"),
        ("arael (f64)", 4.36, 7.93, "arael"),
        ("g2o (GN)", 6.27, 12.12, "other"),
        ("SymForce (f32)", 9.25, 28.13, "other"),
        ("Ceres (LM)", 12.98, 26.49, "other"),
        ("GTSAM (GN)", 13.24, 13.69, "other"),
        ("factrs (GN)", 19.61, 44.42, "other"),
    ]),
]

# Axis per panel, per chart: the two charts plot different quantities, so they
# do not share a scale. (x_max, tick), in PANELS order.
AXES = {
    "iter":  [(15.0, 5.0), (26.0, 10.0), (75.0, 20.0), (22.0, 5.0)],
    "setup": [(20.0, 5.0), (100.0, 30.0), (100.0, 25.0), (60.0, 20.0)],
}

# The two charts. "iter" is the front-page one: one bar, the durable
# cross-system number. "setup" decomposes the first iteration into that same
# iteration plus the one-time setup, which is a busier read -- it lives in
# benchmarks/pgo/README.md, not on the front page.
CHARTS = {
    "iter": {
        "file": "pgo",
        "value_w": 18,
        "title": "Pose-graph optimization: time per iteration",
        "subtitle": ("Four datasets, seven systems, single thread (Apple M4 Pro); "
                     "best validated configuration per system, both arael "
                     "precisions. Lower is better."),
        "foot": [
            ("Excludes setup -- assembly, ordering and symbolic factorization -- "
             "which every system pays once, during its first iteration."),
        ],
    },
    "setup": {
        "file": "pgo-setup",
        "value_w": 60,
        "title": "Pose-graph optimization: per-iteration cost and one-time setup",
        "subtitle": ("Solid: one complete iteration. Faded: the setup, paid once. "
                     "Together they are what the first iteration costs. Single "
                     "thread (Apple M4 Pro)."),
        "foot": [
            ("Setup is assembly, ordering and symbolic factorization: done once, "
             "reused by every later iteration. GTSAM redoes it each time, so shows "
             "none."),
        ],
    },
}

# Appended to both charts.
FOOT = [
    ("Every bar is validated against its dataset's common optimum "
     "(cost within 1%, rigid-aligned RMSE under 5 cm)."),
    ("* arael f32 on the parking garage passes the cost gate (0.02% "
     "above the optimum) but not the geometric one -- a "
     "single-precision floor shared by SymForce's f32."),
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

W = 984
MARGIN = 24
COL_GAP = 6
PANEL_W = (W - 2 * MARGIN - COL_GAP) // 2
# The gap between the columns is COL_GAP plus whatever LABEL_W and VALUE_W
# reserve and do not use -- those two reserves dominate it, so they are cut to
# what the widest label actually needs, and the plots take the width back.
LABEL_W = 92                                 # row labels ('SymForce (f32)')
# VALUE_W (space after the bar for its value label) is per chart: "12.0"
# needs less room than "12.0 + 4.6". PLOT_W follows from it.
BAR_H = 12
PITCH = 19
PANEL_TITLE_H = 20
AXIS_H = 20
ROWS = 7
PANEL_H = PANEL_TITLE_H + ROWS * PITCH + AXIS_H
HEADER_H = 58
ROW_GAP = 18


def bar_path(x0, y, w, h, r):
    """Bar with the data end rounded, baseline end flat."""
    return (f"M{x0},{y} L{x0 + w - r},{y} Q{x0 + w},{y} {x0 + w},{y + r} "
            f"L{x0 + w},{y + h - r} Q{x0 + w},{y + h} {x0 + w - r},{y + h} "
            f"L{x0},{y + h} Z")


def render_panel(s, c, px, py, title, x_max, tick, rows, with_setup, plot_w):
    plot_x = px + LABEL_W
    plot_top = py + PANEL_TITLE_H
    plot_h = ROWS * PITCH
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
        if full is None:
            s.append(f'<text x="{plot_x + 4}" y="{ty:.1f}" font-size="10.5" '
                     f'font-style="italic" fill="{c["muted"]}">did not '
                     f'converge</text>')
            continue
        fill = c["arael"] if is_arael else c["other"]
        star = "*" if kind.endswith("*") else ""
        w = full / x_max * plot_w
        s.append(f'<path d="{bar_path(plot_x, y, w, BAR_H, 3)}" fill="{fill}"/>')
        end, value = plot_x + w, f"{full:.1f}{star}"
        if with_setup:
            # Setup is a measured difference, so on a system that has none it can
            # land marginally below zero (GTSAM rebuilds it every iteration).
            setup = max(0.0, first - full)
            w2 = setup / x_max * plot_w
            if w2 > 0.5:
                x2 = plot_x + w + 2   # 2px surface gap between the segments
                s.append(f'<path d="{bar_path(x2, y, w2, BAR_H, 3)}" '
                         f'fill="{fill}" fill-opacity="0.38"/>')
            end = plot_x + w + 2 + w2
            value = f"{full:.1f}{star} + {setup:.1f}"
        s.append(f'<text x="{end + 6:.1f}" y="{ty:.1f}" font-size="10.5"'
                 f'{weight} fill="{c["ink"]}">{value}</text>')


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
    foot_y = HEADER_H + 2 * PANEL_H + ROW_GAP + 18
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

    for k, (title, rows) in enumerate(PANELS):
        px = MARGIN + (k % 2) * (PANEL_W + COL_GAP)
        py = HEADER_H + (k // 2) * (PANEL_H + ROW_GAP)
        x_max, tick = AXES[chart][k]
        render_panel(s, c, px, py, title, x_max, tick, rows,
                     chart == "setup", PANEL_W - LABEL_W - cfg["value_w"])

    for i, line in enumerate(foot):
        s.append(f'<text x="{MARGIN}" y="{foot_y + i * 14}" font-size="10.5" '
                 f'fill="{c["muted"]}">{line}</text>')
    s.append("</svg>")
    return "\n".join(s) + "\n"


def main():
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, os.pardir, "charts", f"v{arael_version()}")
    os.makedirs(out, exist_ok=True)
    for chart in CHARTS:
        for theme in THEMES:
            path = os.path.join(out, f"{CHARTS[chart]['file']}-{theme}.svg")
            with open(path, "w") as f:
                f.write(render(theme, chart))
            print(f"wrote {os.path.normpath(path)}")


if __name__ == "__main__":
    main()
