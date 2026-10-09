# Generates the threads bar chart committed as charts/v<version>/threads-light.svg
# / threads-dark.svg and embedded in the top-level README.md and src/lib.rs.
#
# Two panels side by side -- landmark SLAM at 1200 poses and bundle adjustment on
# Ladybug-372, both on an AMD Ryzen 9 7950X3D -- each a group of four bars per
# system: one complete iteration at 1, 2, 4 and 8 threads, light to dark. The
# number after the darkest bar is its speedup over one thread. Update the data
# from the results tables after re-running the benchmarks, then run:
#
#   python3 make_threads_chart.py
#
# Pure stdlib, no dependencies.

THREADS = [1, 2, 4, 8]

# Per panel: (title, [(label, [full_iter_ms at each of THREADS], kind)])
# full-iter is one complete iteration (t(2 iters) - t(1 iter), setup cancelled).
# kind: "arael" the blue ramp, "other" the neutral ramp.
PANELS = [
    # 2026-10-09, 4 rounds, unpinned, on the 7950X3D (benchmarks/slam README,
    # 1200-pose figure-8 threads table). Ceres is sparse_cholesky, its best
    # validated configuration on this scene.
    ("Landmark SLAM -- 1200 poses, 21.6k params", [
        ("arael (f64)", [266.87, 159.01, 97.50, 74.85], "arael"),
        ("arael (f32)", [186.32, 112.59, 71.05, 49.59], "arael"),
        ("Ceres (LM)", [933.19, 651.99, 488.55, 427.24], "other"),
    ]),
    # 2026-10-09, 4 rounds, unpinned, on the 7950X3D (benchmarks/bal README,
    # Ladybug-372 threads table). arael is its Schur route, Ceres sparse_schur;
    # the iterative rows are inexact and have no full iteration to plot.
    ("Bundle adjustment -- Ladybug-372, 146k params", [
        ("arael (f64)", [172.40, 103.39, 67.71, 42.85], "arael"),
        ("arael (f32)", [120.36, 75.83, 52.58, 30.30], "arael"),
        ("Ceres (LM)", [284.15, 169.25, 105.50, 78.05], "other"),
    ]),
]

# Axis per panel, in PANELS order: (x_max, tick).
AXES = [(1000.0, 250.0), (300.0, 100.0)]

TITLE = "Threads: time per iteration at 1, 2, 4 and 8 threads"
SUBTITLE = ("Landmark SLAM and bundle adjustment on an AMD Ryzen 9 7950X3D, "
            "unpinned; one complete iteration, lower is better.")
FOOT = [
    ("Bars from light to dark: 1, 2, 4, 8 threads. The number after the "
     "darkest bar is its speedup over one thread."),
    ("Time excludes setup -- assembly, ordering and symbolic factorization -- "
     "which every system pays once, during its first iteration."),
    ("Every bar reaches its problem's common optimum, cross-validated "
     "against Ceres."),
]

# The ramp: the system's fill blended toward the surface, one step per thread
# count, lightest for one thread. Blended to explicit colours rather than drawn
# with opacity, so every renderer shows the same ramp.
ALPHA = [0.35, 0.55, 0.75, 1.0]


def blend(fill, surface, alpha):
    """`fill` over `surface` at `alpha`, as a hex colour."""
    f = [int(fill[i:i + 2], 16) for i in (1, 3, 5)]
    s = [int(surface[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(a * alpha + b * (1 - alpha)):02x}"
                         for a, b in zip(f, s))

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
COL_GAP = 8
PANEL_W = (W - 2 * MARGIN - COL_GAP) // 2   # 418
LABEL_W = 84    # row labels, right-aligned; the longest ("arael (f64)") is 62
VALUE_W = 72    # room after a bar for "74.9 ms  3.6x"
PLOT_W = PANEL_W - LABEL_W - VALUE_W
BAR_H = 9
BAR_PITCH = 11
GROUP_H = len(THREADS) * BAR_PITCH
GROUP_GAP = 12
GROUPS = max(len(rows) for _, rows in PANELS)
PANEL_TITLE_H = 20
AXIS_H = 20
PLOT_H = GROUPS * GROUP_H + (GROUPS - 1) * GROUP_GAP
PANEL_H = PANEL_TITLE_H + PLOT_H + AXIS_H
HEADER_H = 58


def bar_path(x0, y, w, h, r):
    """Bar with the data end rounded, baseline end flat."""
    return (f"M{x0},{y} L{x0 + w - r},{y} Q{x0 + w},{y} {x0 + w},{y + r} "
            f"L{x0 + w},{y + h - r} Q{x0 + w},{y + h} {x0 + w - r},{y + h} "
            f"L{x0},{y + h} Z")


def render_panel(s, c, px, py, title, axis, rows):
    x_max, tick = axis
    plot_x = px + LABEL_W
    plot_top = py + PANEL_TITLE_H
    s.append(f'<text x="{px}" y="{py + 12}" font-size="12.5" '
             f'font-weight="600" fill="{c["ink"]}">{title}</text>')
    # gridlines + ticks
    t, ticks = 0.0, []
    while t <= x_max + 1e-9:
        ticks.append(t)
        t += tick
    for t in ticks:
        x = plot_x + t / x_max * PLOT_W
        s.append(f'<line x1="{x:.1f}" y1="{plot_top}" x2="{x:.1f}" '
                 f'y2="{plot_top + PLOT_H + 3}" stroke="{c["grid"]}" '
                 f'stroke-width="1"/>')
        label = f"{t:.0f} ms" if t == ticks[-1] else f"{t:.0f}"
        s.append(f'<text x="{x:.1f}" y="{plot_top + PLOT_H + 15}" '
                 f'font-size="10" text-anchor="middle" '
                 f'fill="{c["muted"]}">{label}</text>')
    for g, (label, values, kind) in enumerate(rows):
        is_arael = kind == "arael"
        gy = plot_top + g * (GROUP_H + GROUP_GAP)
        ty = gy + GROUP_H / 2 + 4
        weight = ' font-weight="600"' if is_arael else ""
        name_ink = c["ink"] if is_arael else c["secondary"]
        s.append(f'<text x="{plot_x - 8}" y="{ty:.1f}" font-size="11.5" '
                 f'text-anchor="end"{weight} fill="{name_ink}">{label}</text>')
        fill = c["arael"] if is_arael else c["other"]
        for i, (v, a) in enumerate(zip(values, ALPHA)):
            y = gy + i * BAR_PITCH + (BAR_PITCH - BAR_H) / 2
            w = v / x_max * PLOT_W
            s.append(f'<path d="{bar_path(plot_x, y, w, BAR_H, 3)}" '
                     f'fill="{blend(fill, c["surface"], a)}"/>')
            value = f"{v:.0f}"
            if i == len(values) - 1:
                value += f", {values[0] / v:.1f}x"
            vy = y + BAR_H / 2 + 3.2
            s.append(f'<text x="{plot_x + w + 6:.1f}" y="{vy:.1f}" '
                     f'font-size="9.5"{weight} fill="{c["ink"]}">{value}</text>')


def legend(s, c):
    """The ramp, top right: four swatches labelled with their thread count."""
    x = W - MARGIN
    for n, a in reversed(list(zip(THREADS, ALPHA))):
        s.append(f'<text x="{x}" y="48" font-size="10.5" text-anchor="end" '
                 f'fill="{c["secondary"]}">{n}</text>')
        x -= 14
        s.append(f'<rect x="{x - 10}" y="40" width="10" height="9" rx="2" '
                 f'fill="{blend(c["arael"], c["surface"], a)}"/>')
        x -= 16
    s.append(f'<text x="{x}" y="48" font-size="10.5" text-anchor="end" '
             f'fill="{c["secondary"]}">threads:</text>')


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
                    return m.group(1).removesuffix("-dev")
        root = os.path.dirname(root)
    raise SystemExit("cannot find the arael version in any parent Cargo.toml")


def render(theme):
    c = THEMES[theme]
    foot_y = HEADER_H + PANEL_H + 18
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
    legend(s, c)

    for k, (title, rows) in enumerate(PANELS):
        px = MARGIN + k * (PANEL_W + COL_GAP)
        render_panel(s, c, px, HEADER_H, title, AXES[k], rows)

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
        path = os.path.join(out, f"threads-{theme}.svg")
        with open(path, "w") as f:
            f.write(render(theme))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
