# Generates the SLAM 2x2 chart committed as slam-light.svg /
# slam-dark.svg and embedded in benchmarks/slam/README.md.
#
# One cell per desktop/edge results table, each bar decomposed into one
# complete solver iteration (solid) plus the one-time setup paid during
# the first iteration (faded). One bar per system, its best validated
# configuration (Ceres sparse_schur, SymForce f64; the inexact
# iterative_schur is omitted). Update the data from the slam README
# tables after re-running, then:
#
#   python3 make_slam_chart.py
#
# Pure stdlib, no dependencies.

# Per panel: (title, value decimals, (x_max, tick),
#             [(label, full_iter_ms, first_iter_ms, kind)])
# full-iter is one complete iteration (t(2 iters) - t(1 iter), setup
# cancelled); first-iter is that same iteration plus the setup paid
# once. Rows: arael first, then the rest by full-iter ascending.
PANELS = [
    ("60 poses, 1,080 params -- Apple M4 Pro", 1, (18.0, 6.0), [
        ("arael (f32)", 1.57, 2.28, "arael"),
        ("arael (f64)", 1.92, 2.78, "arael"),
        ("g2o (LM)", 3.69, 7.26, "other"),
        ("Ceres (LM)", 5.29, 11.23, "other"),
        ("SymForce (f64)", 6.04, 15.45, "other"),
        ("factrs (LM)", 9.28, 15.68, "other"),
        ("GTSAM (LM)", 10.02, 12.09, "other"),
    ]),
    ("300 poses, 5,400 params -- Apple M4 Pro", 1, (250.0, 50.0), [
        ("arael (f32)", 27.82, 37.66, "arael"),
        ("arael (f64)", 40.76, 59.56, "arael"),
        ("g2o (LM)", 60.57, 111.16, "other"),
        ("Ceres (LM)", 82.12, 157.01, "other"),
        ("factrs (LM)", 122.27, 173.27, "other"),
        ("SymForce (f64)", 128.78, 224.71, "other"),
        ("GTSAM (LM)", 153.02, 164.91, "other"),
    ]),
    ("60 poses, 1,080 params -- Raspberry Pi 5", 1, (64.0, 16.0), [
        ("arael (f32)", 6.83, 10.11, "arael"),
        ("arael (f64)", 9.04, 12.90, "arael"),
        ("g2o (LM)", 16.67, 29.20, "other"),
        ("SymForce (f64)", 17.11, 52.25, "other"),
        ("Ceres (LM)", 24.70, 43.56, "other"),
        ("factrs (LM)", 35.41, 54.81, "other"),
        ("GTSAM (LM)", 43.50, 49.67, "other"),
    ]),
    ("60 poses, 1,080 params -- Raspberry Pi Zero W (ARMv6)", 1, (2000.0, 500.0), [
        ("arael (f32)", 221.28, 292.21, "arael"),
        ("arael (f64)", 346.00, 412.88, "arael"),
        ("factrs (LM)", 1265.49, 1574.91, "other"),
    ]),
]

TITLE = "Monocular SLAM: per-iteration cost and one-time setup"
SUBTITLE = ("Solid: one complete iteration. Faded: the setup, paid once. "
            "Single thread, best validated configuration per system. "
            "Lower is better.")
FOOT = [
    ("Setup is assembly, ordering and symbolic factorization: done once, "
     "reused by every later iteration."),
    ("Ceres is sparse_schur, SymForce is f64; the inexact iterative_schur "
     "is omitted. Every bar reaches the common optimum."),
    ("The Pi Zero W (ARMv6) has no C++ cross-build, so only arael and "
     "factrs run there."),
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

# Columns cut to what their text needs; the canvas is derived from the
# panels so no empty band is left between the columns or to their right.
MARGIN = 18
LABEL_W = 84    # row labels, right-aligned; longest ("SymForce (f64)") is 81
PLOT_W = 248    # the bars
VALUE_W = 54    # room after a bar for its "full + setup" value
COL_GAP = 16    # gutter between the two panel columns
ROW_GAP = 20
PANEL_W = LABEL_W + PLOT_W + VALUE_W
W = 2 * MARGIN + 2 * PANEL_W + COL_GAP
BAR_H = 12
PITCH = 19
PANEL_TITLE_H = 20
AXIS_H = 20
GRID_ROWS = 7   # tallest panel; sets each grid row's height so axes align
GRID_ROW_H = PANEL_TITLE_H + GRID_ROWS * PITCH + AXIS_H
HEADER_H = 58


def bar_path(x0, y, w, h, r):
    """Bar with the data end rounded, baseline end flat."""
    return (f"M{x0},{y} L{x0 + w - r},{y} Q{x0 + w},{y} {x0 + w},{y + r} "
            f"L{x0 + w},{y + h - r} Q{x0 + w},{y + h} {x0 + w - r},{y + h} "
            f"L{x0},{y + h} Z")


def render_panel(s, c, px, py, title, x_max, tick, decimals, rows):
    plot_x = px + LABEL_W
    plot_top = py + PANEL_TITLE_H
    plot_h = len(rows) * PITCH   # axis sits directly under this panel's bars
    s.append(f'<text x="{px}" y="{py + 12}" font-size="12.5" '
             f'font-weight="600" fill="{c["ink"]}">{title}</text>')
    t = 0.0
    while t <= x_max + 1e-9:
        x = plot_x + t / x_max * PLOT_W
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
        w = full / x_max * PLOT_W
        s.append(f'<path d="{bar_path(plot_x, y, w, BAR_H, 3)}" fill="{fill}"/>')
        setup = max(0.0, first - full)
        w2 = setup / x_max * PLOT_W
        if w2 > 0.5:
            x2 = plot_x + w + 2   # 2px surface gap between the segments
            s.append(f'<path d="{bar_path(x2, y, w2, BAR_H, 3)}" '
                     f'fill="{fill}" fill-opacity="0.38"/>')
        end = plot_x + w + 2 + w2
        value = f"{full:.{decimals}f} + {setup:.{decimals}f}"
        s.append(f'<text x="{end + 6:.1f}" y="{ty:.1f}" '
                 f'font-size="10.5"{weight} fill="{c["ink"]}">{value}</text>')


def arael_version():
    """Read the workspace version, so the stamp cannot drift from the code."""
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
    foot_y = HEADER_H + 2 * GRID_ROW_H + ROW_GAP + 18
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

    for k, (title, decimals, (x_max, tick), rows) in enumerate(PANELS):
        px = MARGIN + (k % 2) * (PANEL_W + COL_GAP)
        py = HEADER_H + (k // 2) * (GRID_ROW_H + ROW_GAP)
        render_panel(s, c, px, py, title, x_max, tick, decimals, rows)

    for i, line in enumerate(FOOT):
        s.append(f'<text x="{MARGIN}" y="{foot_y + i * 14}" font-size="10.5" '
                 f'fill="{c["muted"]}">{line}</text>')
    s.append("</svg>")
    return "\n".join(s) + "\n"


def main():
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "charts", f"v{arael_version()}")
    os.makedirs(out, exist_ok=True)
    for theme in THEMES:
        path = os.path.join(out, f"slam-{theme}.svg")
        with open(path, "w") as f:
            f.write(render(theme))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
