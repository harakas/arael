# Generates the covariance-recovery 2x2 chart committed as
# cov-light.svg / cov-dark.svg and embedded in benchmarks/slam/README.md.
#
# One cell per (entity, query-count): top row poses, bottom row
# landmarks; left column a single marginal, right column all of them.
# One bar per system, time to recover. Each cell has its own linear
# time axis (scaled to its slowest bar), since the four cells span 0.1 s
# to 37 s; some systems do not finish. Lower is better.
# Update the data from the slam README covariance tables after
# re-running, then:
#
#   python3 make_cov_chart.py
#
# Pure stdlib, no dependencies.

DNF = None   # a cell a system does not finish within the 120 s cap

# Per panel: (title, [(label, ms_or_DNF, kind)]). Rows sorted fastest
# first; arael's two methods (per-query and the bulk AllMarginals pass)
# are both highlighted. g2o marginalizes landmarks away, so it recovers
# poses only -- absent from the landmark rows.
PANELS = [
    ("Pose marginals -- single query", 5000.0, 1000.0, [
        ("arael PerQuery", 124.6, "arael"),
        ("GTSAM", 195.9, "other"),
        ("g2o", 2644.1, "other"),
        ("Ceres", 4035.0, "other"),
    ]),
    ("Pose marginals -- all 300 poses", 8000.0, 2000.0, [
        ("arael AllMarg", 476.3, "arael"),
        ("arael PerQuery", 1236.8, "arael"),
        ("g2o", 4066.1, "other"),
        ("Ceres", 7031.4, "other"),
        ("GTSAM", 37411.5, "other"),
    ]),
    ("Landmark marginals -- single query", 5000.0, 1000.0, [
        ("arael PerQuery", 124.4, "arael"),
        ("GTSAM", 465.9, "other"),
        ("Ceres", 4054.7, "other"),
    ]),
    ("Landmark marginals -- all 1200 landmarks", 3000.0, 1000.0, [
        ("arael AllMarg", 476.3, "arael"),
        ("arael PerQuery", 2812.9, "arael"),
        ("Ceres", DNF, "other"),
        ("GTSAM", DNF, "other"),
    ]),
]

TITLE = "Covariance recovery: time to read marginals from the solution"
SUBTITLE = ("300 poses, 1200 landmarks. One bar per system, each cell on its "
            "own linear axis; lower is better.")
FOOT = [
    ("arael factors once and reads marginals off it; Ceres and GTSAM rebuild "
     "cold; g2o reuses its solve factor (poses only)."),
    ("\"> 120 s\": did not finish within the per-cell cap. At 1200 landmarks "
     "only arael's bulk pass stays sub-second."),
    ("The all-poses axis is capped at 8 s; GTSAM's bar is clipped (true time "
     "labelled)."),
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

MARGIN = 18
LABEL_W = 88    # row labels, right-aligned; longest ("arael PerQuery") is ~84
PLOT_W = 244    # the bars
VALUE_W = 46    # room after a bar for its value
COL_GAP = 18
ROW_GAP = 20
PANEL_W = LABEL_W + PLOT_W + VALUE_W
W = 2 * MARGIN + 2 * PANEL_W + COL_GAP
BAR_H = 12
PITCH = 19
PANEL_TITLE_H = 20
AXIS_H = 20
GRID_ROWS = 5   # tallest panel; sets each grid row's height so axes align
GRID_ROW_H = PANEL_TITLE_H + GRID_ROWS * PITCH + AXIS_H
HEADER_H = 58

def fmt(ms):
    return f"{ms:.0f}"


def tick_label(tk, last):
    if tk == 0:
        return "0"
    lab = f"{tk // 1000}k" if tk >= 1000 else f"{tk:.0f}"
    return lab + " ms" if last else lab


def bar_path(x0, y, w, h, r):
    """Bar with the data end rounded, baseline end flat."""
    return (f"M{x0},{y} L{x0 + w - r},{y} Q{x0 + w},{y} {x0 + w},{y + r} "
            f"L{x0 + w},{y + h - r} Q{x0 + w},{y + h} {x0 + w - r},{y + h} "
            f"L{x0},{y + h} Z")


def render_panel(s, c, px, py, title, x_max, tick, rows):
    plot_x = px + LABEL_W
    plot_top = py + PANEL_TITLE_H
    plot_h = len(rows) * PITCH
    s.append(f'<text x="{px}" y="{py + 12}" font-size="12.5" '
             f'font-weight="600" fill="{c["ink"]}">{title}</text>')
    t = 0.0
    while t <= x_max + 1e-9:
        x = plot_x + t / x_max * PLOT_W
        s.append(f'<line x1="{x:.1f}" y1="{plot_top}" x2="{x:.1f}" '
                 f'y2="{plot_top + plot_h + 3}" stroke="{c["grid"]}" '
                 f'stroke-width="1"/>')
        s.append(f'<text x="{x:.1f}" y="{plot_top + plot_h + 15}" '
                 f'font-size="10" text-anchor="middle" '
                 f'fill="{c["muted"]}">{tick_label(t, t + tick > x_max + 1e-9)}</text>')
        t += tick
    for i, (label, ms, kind) in enumerate(rows):
        is_arael = kind == "arael"
        y = plot_top + i * PITCH + (PITCH - BAR_H) / 2
        ty = y + BAR_H / 2 + 3.5
        weight = ' font-weight="600"' if is_arael else ""
        name_ink = c["ink"] if is_arael else c["secondary"]
        s.append(f'<text x="{plot_x - 8}" y="{ty:.1f}" font-size="11.5" '
                 f'text-anchor="end"{weight} fill="{name_ink}">{label}</text>')
        if ms is DNF:
            s.append(f'<text x="{plot_x + 2:.1f}" y="{ty:.1f}" font-size="10.5" '
                     f'font-style="italic" fill="{c["muted"]}">'
                     f'&gt; 120 s (did not finish)</text>')
            continue
        fill = c["arael"] if is_arael else c["other"]
        clipped = ms > x_max
        w = PLOT_W if clipped else ms / x_max * PLOT_W
        w = max(w, 1.5)
        r = min(3.0, w / 2)
        s.append(f'<path d="{bar_path(plot_x, y, w, BAR_H, r)}" fill="{fill}"/>')
        if clipped:
            # a surface-coloured chevron notched into the bar end reads as
            # "off the axis, continues".
            s.append(f'<text x="{plot_x + PLOT_W - 9:.1f}" y="{ty:.1f}" '
                     f'font-size="11" font-weight="700" text-anchor="middle" '
                     f'fill="{c["surface"]}">&gt;</text>')
        s.append(f'<text x="{plot_x + w + 6:.1f}" y="{ty:.1f}" '
                 f'font-size="10.5"{weight} fill="{c["ink"]}">{fmt(ms)}</text>')


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

    for k, (title, x_max, tick, rows) in enumerate(PANELS):
        px = MARGIN + (k % 2) * (PANEL_W + COL_GAP)
        py = HEADER_H + (k // 2) * (GRID_ROW_H + ROW_GAP)
        render_panel(s, c, px, py, title, x_max, tick, rows)

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
        path = os.path.join(out, f"cov-{theme}.svg")
        with open(path, "w") as f:
            f.write(render(theme))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
