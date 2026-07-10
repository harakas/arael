# Generates the SLAM + localization bar chart committed as
# chart-slam-loc-light.svg / chart-slam-loc-dark.svg and embedded in the
# top-level README.md and src/lib.rs.
#
# Two panels side by side, both on ms per solver iteration: landmark
# SLAM at 300 poses on an Apple M4 Pro (benchmarks/slam README, 300-pose
# table) and fixed-map localization at 60 poses on a Raspberry Pi 5
# (benchmarks/loc README, Pi 5 table). One bar per system showing its
# best validated configuration, all arael rows shown. tiny-solver is
# omitted from the bars for scale and reported in the footnote. Update
# the data from the results tables after re-running the benchmarks,
# then run:
#
#   python3 make_slam_loc_chart.py
#
# Pure stdlib, no dependencies.

# Per panel: (title, x_max, tick step, value decimals,
#             [(label, ms_per_iter, kind)])
# kind: "arael" solid blue bar, "other" neutral bar.
PANELS = [
    ("Landmark SLAM -- 300 poses, 5.4k params (Apple M4 Pro)", 200.0, 50.0, 1, [
        ("arael (f64 cholmod-gpl)", 59.34, "arael"),
        ("arael (f32)", 77.70, "arael"),
        ("g2o (LM)", 77.94, "other"),
        ("arael (f64)", 101.00, "arael"),
        ("Ceres (LM)", 101.61, "other"),
        ("SymForce (f64)", 120.83, "other"),
        ("factrs (LM)", 156.30, "other"),
        ("GTSAM (LM)", 167.35, "other"),
    ]),
    ("Localization -- 60 poses, 360 params (Raspberry Pi 5)", 20.0, 5.0, 2, [
        ("arael (f32 band)", 1.03, "arael"),
        ("arael (f64 band)", 1.09, "arael"),
        ("SymForce (f32)", 4.82, "other"),
        ("Ceres (LM)", 5.04, "other"),
        ("g2o (LM)", 5.58, "other"),
        ("GTSAM (LM)", 15.48, "other"),
        ("factrs (LM)", 18.22, "other"),
    ]),
]

TITLE = "Landmark SLAM and localization: time per solver iteration"
SUBTITLE = ("Landmark SLAM on a desktop core, fixed-map localization on an "
            "edge board; single thread, best validated configuration per "
            "system. Lower is better.")
FOOT = [
    ("Every bar reaches its problem's common optimum, cross-validated "
     "against all systems."),
    ("arael cholmod-gpl is the optional CHOLMOD supernodal backend "
     "(GPL-licensed); the localization rows use arael's band-Cholesky "
     "solver on the block-tridiagonal Hessian."),
    ("tiny-solver omitted for scale: 317.8 ms/iter on the SLAM panel, "
     "88.5 ms/iter on the Pi 5."),
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
LABEL_W = 140                                # row labels, right-aligned
VALUE_W = 44                                 # value labels after bars
PLOT_W = PANEL_W - LABEL_W - VALUE_W
BAR_H = 12
PITCH = 19
PANEL_TITLE_H = 20
AXIS_H = 20
ROWS = max(len(rows) for _, _, _, _, rows in PANELS)
PANEL_H = PANEL_TITLE_H + ROWS * PITCH + AXIS_H
HEADER_H = 58


def bar_path(x0, y, w, h, r):
    """Bar with the data end rounded, baseline end flat."""
    return (f"M{x0},{y} L{x0 + w - r},{y} Q{x0 + w},{y} {x0 + w},{y + r} "
            f"L{x0 + w},{y + h - r} Q{x0 + w},{y + h} {x0 + w - r},{y + h} "
            f"L{x0},{y + h} Z")


def render_panel(s, c, px, py, title, x_max, tick, decimals, rows):
    plot_x = px + LABEL_W
    plot_top = py + PANEL_TITLE_H
    plot_h = ROWS * PITCH  # common height so the two panels' axes align
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
        w = ms / x_max * PLOT_W
        fill = c["arael"] if is_arael else c["other"]
        s.append(f'<path d="{bar_path(plot_x, y, w, BAR_H, 3)}" fill="{fill}"/>')
        s.append(f'<text x="{plot_x + w + 6:.1f}" y="{ty:.1f}" '
                 f'font-size="10.5"{weight} fill="{c["ink"]}">'
                 f'{ms:.{decimals}f}</text>')


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
    s.append(f'<text x="{MARGIN}" y="48" font-size="11.5" '
             f'fill="{c["secondary"]}">{SUBTITLE}</text>')

    for k, (title, x_max, tick, decimals, rows) in enumerate(PANELS):
        px = MARGIN + k * (PANEL_W + COL_GAP)
        render_panel(s, c, px, HEADER_H, title, x_max, tick, decimals, rows)

    for i, line in enumerate(FOOT):
        s.append(f'<text x="{MARGIN}" y="{foot_y + i * 14}" font-size="10.5" '
                 f'fill="{c["muted"]}">{line}</text>')
    s.append("</svg>")
    return "\n".join(s) + "\n"


def main():
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    for theme in THEMES:
        path = os.path.join(here, f"chart-slam-loc-{theme}.svg")
        with open(path, "w") as f:
            f.write(render(theme))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
