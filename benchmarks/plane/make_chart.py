# Generates the plane-SLAM bar chart committed as plane-setup-light.svg /
# plane-setup-dark.svg and embedded in benchmarks/plane/README.md.
#
# A 2x2 small-multiple assembly: one panel per scene size, all on the same
# metric (ms per solver iteration, the durable cross-system number -- see
# README), one bar per system. Each bar is split into the iteration itself
# and the setup paid once, which together are the first iteration. Update
# the data from the results tables after re-running the benchmark, then:
#
#   python3 make_chart.py
#
# Pure stdlib, no dependencies.

# Per panel: (title, [ (label, full_iter_ms, first_iter_ms, kind) ]).
# full-iter is one complete iteration (t(2 iters) - t(1 iter), setup cancelled).
# first-iter is that same iteration plus the setup paid once. Their difference
# is the setup, drawn faded. 2026-08-02, min of 128 rounds (64 at 900 poses).
# kind: "arael" solid blue, "other" neutral, "arael*" adds a star to the value.
# full_iter None -> italic text row (no clean first iteration to measure).
PANELS = [
    ("60 poses, 24 planes (492 params)", [
        ("arael (f32)", 0.09, 0.30, "arael"),
        ("arael (f64)", 0.10, 0.32, "arael"),
        ("SymForce (f32)", 0.17, 0.91, "other"),
        ("SymForce (f64)", 0.19, 0.92, "other"),
        ("Ceres", 0.39, 0.90, "other"),
        ("GTSAM", 0.56, 0.65, "other"),
        ("factrs", 0.64, 1.14, "other"),
        ("g2o", 1.06, 1.13, "other"),
    ]),
    ("120 poses, 45 planes (975 params)", [
        ("arael (f32)", 0.19, 0.84, "arael"),
        ("arael (f64)", 0.21, 0.87, "arael"),
        ("SymForce (f32)", 0.39, 1.78, "other"),
        ("SymForce (f64)", 0.42, 1.80, "other"),
        ("Ceres", 0.84, 1.73, "other"),
        ("GTSAM", 1.15, 1.30, "other"),
        ("factrs", 1.26, 2.51, "other"),
        ("g2o", 2.02, 2.18, "other"),
    ]),
    ("300 poses, 114 planes (2442 params)", [
        ("arael (f32)", 0.54, 2.13, "arael"),
        ("arael (f64)", 0.64, 2.24, "arael"),
        ("SymForce (f32)", 1.00, 4.74, "other"),
        ("SymForce (f64)", 1.13, 4.83, "other"),
        ("Ceres", 2.13, 4.57, "other"),
        ("GTSAM", 3.13, 3.34, "other"),
        ("factrs", 3.45, 6.71, "other"),
        ("g2o", 5.30, 5.60, "other"),
    ]),
    ("900 poses, 339 planes (7317 params)", [
        ("arael (f32)", 1.56, 6.64, "arael*"),
        ("arael (f64)", 1.73, 6.96, "arael"),
        ("SymForce (f32)", 3.33, 15.62, "other"),
        ("SymForce (f64)", 3.80, 15.69, "other"),
        ("Ceres", 6.67, 14.24, "other"),
        ("GTSAM", 9.49, 56.98, "other"),
        ("g2o", 16.35, 17.10, "other"),
        ("factrs", None, None, "other"),
    ]),
]

# (x_max, tick) per panel, in PANELS order. The panels do not share a scale:
# the scene grows 15x across them, so one scale would flatten the small ones.
AXES = [(1.5, 0.5), (3.0, 1.0), (7.5, 2.5), (60.0, 20.0)]

# What a row with no full-iter says instead of a bar. The harness reports
# full-iter only when the first iteration was a single accepted step.
NO_ITER = "rejected step in iteration 1"

TITLE = "Plane SLAM: per-iteration cost and one-time setup"
SUBTITLE = ("Solid: one complete iteration. Faded: the setup, paid once. "
            "Together they are what the first iteration costs. Single "
            "thread (Apple M4 Pro). Lower is better.")
FOOT = [
    ("Setup is assembly, ordering and symbolic factorization: done once, "
     "reused by every later iteration."),
    ("Every bar is validated against its scene's common optimum (cost within "
     "1%, distance to the best solution under 5 cm; f32 rows 10x that)."),
    ("* arael f32 at 900 poses passes the cost gate but sits 0.32 m from the "
     "f64 solution -- the single-precision floor on this scene."),
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
LABEL_W = 92                                 # row labels ('SymForce (f32)')
VALUE_W = 66                                 # after the bar: '16.4 + 0.8'
PLOT_W = PANEL_W - LABEL_W - VALUE_W
BAR_H = 12
PITCH = 19
PANEL_TITLE_H = 20
AXIS_H = 20
ROWS = 8
PANEL_H = PANEL_TITLE_H + ROWS * PITCH + AXIS_H
HEADER_H = 58
ROW_GAP = 18


def bar_path(x0, y, w, h, r):
    """Bar with the data end rounded, baseline end flat."""
    return (f"M{x0},{y} L{x0 + w - r},{y} Q{x0 + w},{y} {x0 + w},{y + r} "
            f"L{x0 + w},{y + h - r} Q{x0 + w},{y + h} {x0 + w - r},{y + h} "
            f"L{x0},{y + h} Z")


def panel_decimals(rows):
    """Decimals for a panel's value labels.

    One decimal reads best, but it is not enough twice over: a panel with
    sub-millisecond bars throws away a real digit ("0.5" for 0.54), and a
    close pair can round onto the same label. Two decimals in either case.
    """
    fulls = [full for _, full, _, _ in rows if full is not None]
    if min(fulls) < 1.0:
        return 2
    at_one = [f"{f:.1f}" for f in fulls]
    return 2 if len(set(at_one)) < len(at_one) else 1


def render_panel(s, c, px, py, title, x_max, tick, rows):
    plot_x = px + LABEL_W
    plot_top = py + PANEL_TITLE_H
    plot_h = ROWS * PITCH
    s.append(f'<text x="{px}" y="{py + 12}" font-size="12.5" '
             f'font-weight="600" fill="{c["ink"]}">{title}</text>')
    # gridlines + ticks
    t = 0.0
    while t <= x_max + 1e-9:
        x = plot_x + t / x_max * PLOT_W
        s.append(f'<line x1="{x:.1f}" y1="{plot_top}" x2="{x:.1f}" '
                 f'y2="{plot_top + plot_h + 3}" stroke="{c["grid"]}" '
                 f'stroke-width="1"/>')
        label = f"{t:g} ms" if t + tick > x_max + 1e-9 else f"{t:g}"
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
                     f'font-style="italic" fill="{c["muted"]}">{NO_ITER}</text>')
            continue
        fill = c["arael"] if is_arael else c["other"]
        star = "*" if kind.endswith("*") else ""
        w = full / x_max * PLOT_W
        s.append(f'<path d="{bar_path(plot_x, y, w, BAR_H, 3)}" fill="{fill}"/>')
        setup = max(0.0, first - full)
        w2 = setup / x_max * PLOT_W
        if w2 > 0.5:
            x2 = plot_x + w + 2   # 2px surface gap between the segments
            s.append(f'<path d="{bar_path(x2, y, w2, BAR_H, 3)}" '
                     f'fill="{fill}" fill-opacity="0.38"/>')
        end = plot_x + w + 2 + w2
        d = panel_decimals(rows)
        s.append(f'<text x="{end + 6:.1f}" y="{ty:.1f}" font-size="10.5"'
                 f'{weight} fill="{c["ink"]}">{full:.{d}f}{star} + '
                 f'{setup:.{d}f}</text>')


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
            if 'name = "arael"' in text:
                m = re.search(r'^version = "([^"]+)"', text, re.M)
                if m:
                    return m.group(1)
        root = os.path.dirname(root)
    raise SystemExit("cannot find the arael version in any parent Cargo.toml")


def render(theme):
    c = THEMES[theme]
    foot_y = HEADER_H + 2 * PANEL_H + ROW_GAP + 18
    height = foot_y + len(FOOT) * 14 + 10

    s = []
    s.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" '
             f'height="{height}" viewBox="0 0 {W} {height}" '
             f'font-family="{FONT}">')
    s.append(f'<rect x="0.5" y="0.5" width="{W - 1}" height="{height - 1}" '
             f'rx="8" fill="{c["surface"]}" stroke="{c["border"]}"/>')
    s.append(f'<text x="{MARGIN}" y="30" font-size="15" font-weight="600" '
             f'fill="{c["ink"]}">{TITLE}</text>')
    s.append(f'<text x="{W - MARGIN}" y="30" font-size="11.5" text-anchor="end" '
             f'fill="{c["muted"]}">arael {arael_version()}</text>')
    s.append(f'<text x="{MARGIN}" y="48" font-size="11.5" '
             f'fill="{c["secondary"]}">{SUBTITLE}</text>')

    for k, (title, rows) in enumerate(PANELS):
        px = MARGIN + (k % 2) * (PANEL_W + COL_GAP)
        py = HEADER_H + (k // 2) * (PANEL_H + ROW_GAP)
        x_max, tick = AXES[k]
        render_panel(s, c, px, py, title, x_max, tick, rows)

    for i, line in enumerate(FOOT):
        s.append(f'<text x="{MARGIN}" y="{foot_y + i * 14}" font-size="10.5" '
                 f'fill="{c["muted"]}">{line}</text>')
    s.append("</svg>")
    return "\n".join(s) + "\n"


def main():
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, os.pardir, "charts", f"v{arael_version()}")
    os.makedirs(out, exist_ok=True)
    for theme in THEMES:
        path = os.path.join(out, f"plane-setup-{theme}.svg")
        with open(path, "w") as f:
            f.write(render(theme))
        print(f"wrote {os.path.normpath(path)}")


if __name__ == "__main__":
    main()
