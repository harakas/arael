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

# Per panel: (title, x_max, tick step, [(label, ms_per_step, kind)])
# kind: "arael" solid blue bar, "other" neutral bar; "arael*" adds a
# star to the value (arael f32 where it does not pass the geometric
# validation gate -- see footnote). None ms -> italic text row.
PANELS = [
    # full-iter: one complete iteration, t(2 iters) - t(1 iter), so the one-time
    # setup cancels (2026-07-13, min of 16 rounds). Each bar is the system's best
    # validated configuration by total time.
    ("M3500 (2D, 10.5k params)", 15.0, 5.0, [
        ("arael (f32)", 1.86, "arael"),
        ("arael (f64)", 2.02, "arael"),
        ("g2o (GN)", 3.36, "other"),
        ("SymForce (f32)", 3.50, "other"),
        ("Ceres (LM)", 4.48, "other"),
        ("factrs (GN)", 6.37, "other"),
        ("GTSAM (GN)", 13.29, "other"),
    ]),
    ("city10000 (2D, 30k params)", 26.0, 10.0, [
        ("arael (f32)", 8.26, "arael"),
        ("arael (f64)", 10.36, "arael"),
        ("g2o (GN)", 16.18, "other"),
        ("SymForce (f64)", 20.64, "other"),
        ("Ceres (LM)", 22.05, "other"),
        ("factrs (GN)", 24.88, "other"),
        ("GTSAM", None, "other"),  # did not converge; text row
    ]),
    ("sphere2500 (3D, 15k params)", 75.0, 20.0, [
        ("arael (f32)", 11.98, "arael"),
        ("arael (f64)", 16.66, "arael"),
        ("g2o (LM)", 18.80, "other"),
        ("Ceres (LM)", 22.68, "other"),
        ("GTSAM (GN)", 28.03, "other"),
        ("factrs (GN)", 35.77, "other"),
        ("SymForce (f32)", 71.45, "other"),
    ]),
    ("parking-garage (3D, 10k params)", 34.0, 10.0, [
        ("arael (f32)", 4.07, "arael*"),
        ("arael (f64)", 4.62, "arael"),
        ("g2o (GN)", 6.49, "other"),
        ("SymForce (f64)", 8.63, "other"),
        ("Ceres (LM)", 12.20, "other"),
        ("GTSAM (GN)", 13.58, "other"),
        ("factrs (GN)", 32.40, "other"),
    ]),
]

TITLE = "Pose-graph optimization: time per iteration"
SUBTITLE = ("Four datasets, seven systems, single thread (Apple M4 Pro); "
            "best validated configuration per system, both arael "
            "precisions. Lower is better.")
FOOT = [
    ("Excludes setup -- assembly, ordering and symbolic factorization -- "
     "which every system pays once, during its first iteration."),
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

W = 880
MARGIN = 24
COL_GAP = 34
PANEL_W = (W - 2 * MARGIN - COL_GAP) // 2   # 399
LABEL_W = 104                                # row labels, right-aligned
VALUE_W = 42                                 # value labels after bars
PLOT_W = PANEL_W - LABEL_W - VALUE_W
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
        label = f"{t:.0f} ms" if t + tick > x_max + 1e-9 else f"{t:.0f}"
        s.append(f'<text x="{x:.1f}" y="{plot_top + plot_h + 15}" '
                 f'font-size="10" text-anchor="middle" '
                 f'fill="{c["muted"]}">{label}</text>')
        t += tick
    for i, (label, ms, kind) in enumerate(rows):
        is_arael = kind.startswith("arael")
        y = plot_top + i * PITCH + (PITCH - BAR_H) / 2
        ty = y + BAR_H / 2 + 3.5
        weight = ' font-weight="600"' if is_arael else ""
        name_ink = c["ink"] if is_arael else c["secondary"]
        s.append(f'<text x="{plot_x - 8}" y="{ty:.1f}" font-size="11.5" '
                 f'text-anchor="end"{weight} fill="{name_ink}">{label}</text>')
        if ms is None:
            s.append(f'<text x="{plot_x + 4}" y="{ty:.1f}" font-size="10.5" '
                     f'font-style="italic" fill="{c["muted"]}">did not '
                     f'converge</text>')
            continue
        w = ms / x_max * PLOT_W
        fill = c["arael"] if is_arael else c["other"]
        s.append(f'<path d="{bar_path(plot_x, y, w, BAR_H, 3)}" fill="{fill}"/>')
        star = "*" if kind.endswith("*") else ""
        s.append(f'<text x="{plot_x + w + 6:.1f}" y="{ty:.1f}" '
                 f'font-size="10.5"{weight} fill="{c["ink"]}">{ms:.1f}{star}</text>')



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
    # The version that produced these numbers, on the image itself: a chart
    # gets copied out of the README and has to carry its own provenance.
    s.append(f'<text x="{W - MARGIN}" y="30" font-size="11.5" text-anchor="end" '
             f'fill="{c["muted"]}">arael {arael_version()}</text>')
    s.append(f'<text x="{MARGIN}" y="48" font-size="11.5" '
             f'fill="{c["secondary"]}">{SUBTITLE}</text>')

    for k, (title, x_max, tick, rows) in enumerate(PANELS):
        px = MARGIN + (k % 2) * (PANEL_W + COL_GAP)
        py = HEADER_H + (k // 2) * (PANEL_H + ROW_GAP)
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
        path = os.path.join(out, f"pgo-{theme}.svg")
        with open(path, "w") as f:
            f.write(render(theme))
        print(f"wrote {os.path.normpath(path)}")


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

