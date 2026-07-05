# Generates the benchmark bar chart committed as chart-light.svg /
# chart-dark.svg and embedded in the top-level README.md and src/lib.rs.
#
# Plots the per-step cost (the durable cross-system number, see README)
# on city10000, best configuration per system -- the same rows and
# values as the top-level README table. Update DATA from the results
# table after re-running the benchmark, then run:
#
#   python3 make_chart.py
#
# Pure stdlib, no dependencies.

# (label, ms per accepted step, vs-best ratio, is_arael)
DATA = [
    ("arael (LM, f32)", 10.1, "1.00x", True),
    ("arael (LM, f64)", 13.1, "1.29x", True),
    ("g2o (GN)", 19.8, "1.95x", False),
    ("g2o (LM)", 21.5, "2.12x", False),
    ("Ceres (LM)", 25.8, "2.55x", False),
    ("SymForce (LM)", 28.3, "2.79x", False),
    ("factrs (GN)", 30.0, "2.96x", False),
    ("tiny-solver (GN)", 84.8, "8.37x", False),
]

TITLE = "Pose-graph optimization: time per solver step"
SUBTITLE = ("city10000 (10,000 poses, 20,687 constraints), "
            "single thread, Apple M4 Pro. Lower is better.")
FOOT1 = ("Best configuration per system; every row verified to reach the "
         "same minimum. Ratios are relative to the fastest.")
FOOT2 = ("GTSAM (batch) did not converge on this dataset; its incremental "
         "ISAM2 solves it in 10.4 s.")

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

W = 760
PLOT_X = 192       # left edge of the bars; row labels right-align to it
PLOT_W = 460
BAR_H = 20
PITCH = 30
PLOT_TOP = 76
X_MAX = 90.0       # ms; axis ticks every 20
PX_PER_MS = PLOT_W / X_MAX


def bar_path(x0, y, w, h, r):
    """Bar with the data end rounded (4px), baseline end flat."""
    return (f"M{x0},{y} L{x0 + w - r},{y} Q{x0 + w},{y} {x0 + w},{y + r} "
            f"L{x0 + w},{y + h - r} Q{x0 + w},{y + h} {x0 + w - r},{y + h} "
            f"L{x0},{y + h} Z")


def render(theme):
    c = THEMES[theme]
    plot_h = len(DATA) * PITCH
    axis_y = PLOT_TOP + plot_h + 18
    foot_y = axis_y + 24
    height = foot_y + 34

    s = []
    s.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" '
             f'height="{height}" viewBox="0 0 {W} {height}" '
             f'font-family="{FONT}">')
    s.append(f'<rect x="0.5" y="0.5" width="{W - 1}" height="{height - 1}" '
             f'rx="8" fill="{c["surface"]}" stroke="{c["border"]}"/>')
    s.append(f'<text x="24" y="32" font-size="15" font-weight="600" '
             f'fill="{c["ink"]}">{TITLE}</text>')
    s.append(f'<text x="24" y="52" font-size="12" '
             f'fill="{c["secondary"]}">{SUBTITLE}</text>')

    # gridlines + tick labels
    for ms in range(0, int(X_MAX) + 1, 20):
        x = PLOT_X + ms * PX_PER_MS
        s.append(f'<line x1="{x:.1f}" y1="{PLOT_TOP - 6}" x2="{x:.1f}" '
                 f'y2="{PLOT_TOP + plot_h + 4}" stroke="{c["grid"]}" '
                 f'stroke-width="1"/>')
        label = f"{ms} ms" if ms == 80 else str(ms)
        s.append(f'<text x="{x:.1f}" y="{axis_y}" font-size="11" '
                 f'text-anchor="middle" fill="{c["muted"]}">{label}</text>')

    for i, (label, ms, ratio, is_arael) in enumerate(DATA):
        y = PLOT_TOP + i * PITCH + (PITCH - BAR_H) / 2
        w = ms * PX_PER_MS
        fill = c["arael"] if is_arael else c["other"]
        weight = ' font-weight="600"' if is_arael else ""
        name_ink = c["ink"] if is_arael else c["secondary"]
        s.append(f'<text x="{PLOT_X - 10}" y="{y + BAR_H / 2 + 4:.1f}" '
                 f'font-size="12.5" text-anchor="end"{weight} '
                 f'fill="{name_ink}">{label}</text>')
        s.append(f'<path d="{bar_path(PLOT_X, y, w, BAR_H, 4)}" '
                 f'fill="{fill}"/>')
        vx = PLOT_X + w + 8
        ty = y + BAR_H / 2 + 4
        s.append(f'<text x="{vx:.1f}" y="{ty:.1f}" font-size="12"{weight} '
                 f'fill="{c["ink"]}">{ms} ms</text>')
        s.append(f'<text x="{vx + 48:.1f}" y="{ty:.1f}" font-size="12" '
                 f'fill="{c["muted"]}">{ratio}</text>')

    s.append(f'<text x="24" y="{foot_y}" font-size="11" '
             f'fill="{c["muted"]}">{FOOT1}</text>')
    s.append(f'<text x="24" y="{foot_y + 15}" font-size="11" '
             f'fill="{c["muted"]}">{FOOT2}</text>')
    s.append("</svg>")
    return "\n".join(s) + "\n"


def main():
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    for theme in THEMES:
        path = os.path.join(here, f"chart-{theme}.svg")
        with open(path, "w") as f:
            f.write(render(theme))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
