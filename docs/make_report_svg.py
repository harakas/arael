#!/usr/bin/env python3
"""Render an ANSI-coloured terminal capture as an SVG, light and dark.

README.md is also the crates.io page, and crates.io renders neither ANSI escapes
nor GitHub's ```ansi blocks -- an image is the only way a coloured sample shows
up in both places. Same trick the benchmark charts use.

Regenerate (from the repo root):

    cargo run -r -q --example slam2d_simple_demo 2>/dev/null \
        | sed -n '/^LM /,/half-bandwidth/p' \
        | python3 docs/make_report_svg.py docs/report

writes docs/report-light.svg and docs/report-dark.svg.
"""

import re
import sys

# ANSI SGR -> (light, dark). Only the codes LmResult::pretty_report emits.
INK = {
    "31": ("#cf222e", "#ff7b72"),  # red    -- factorization failed
    "32": ("#1a7f37", "#3fb950"),  # green  -- accepted
    "33": ("#9a6700", "#d29922"),  # yellow -- rejected
}
THEME = {
    #        background   default ink  dim ink
    "light": ("#ffffff", "#24292f", "#6e7781"),
    "dark": ("#0d1117", "#c9d1d9", "#8b949e"),
}

FONT = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'DejaVu Sans Mono', monospace"
SIZE = 13          # px
LINE = 19          # px between baselines
CHAR = 7.82        # advance of one monospace char at SIZE, measured
PAD_X, PAD_Y = 16, 14

SGR = re.compile(r"\x1b\[([0-9;]*)m")


def runs(line):
    """Split one ANSI line into (text, colour_code|None, bold, dim) runs."""
    out, pos = [], 0
    colour, bold, dim = None, False, False
    for m in SGR.finditer(line):
        if m.start() > pos:
            out.append((line[pos : m.start()], colour, bold, dim))
        for code in (m.group(1) or "0").split(";"):
            if code in ("", "0"):
                colour, bold, dim = None, False, False
            elif code == "1":
                bold = True
            elif code == "2":
                dim = True
            elif code in INK:
                colour = code
        pos = m.end()
    if pos < len(line):
        out.append((line[pos:], colour, bold, dim))
    return out


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render(lines, mode):
    bg, fg, dim_ink = THEME[mode]
    cols = max((len(SGR.sub("", ln)) for ln in lines), default=0)
    # +1 col of slack: a font whose advance runs a little over CHAR would
    # otherwise push the last glyph against the right edge.
    w = round(PAD_X * 2 + (cols + 1) * CHAR)
    h = PAD_Y * 2 + len(lines) * LINE

    body = []
    for i, line in enumerate(lines):
        y = PAD_Y + SIZE + i * LINE
        col = 0
        for text, colour, bold, dim in runs(line):
            if not text:
                continue
            # One <text> per run, one explicit x per CHARACTER. Implicit tspan
            # flow is not honoured the same way by every SVG renderer (it stacked
            # every run at the same x here), and a font whose advance is not
            # exactly CHAR would drift within a long run. An explicit grid depends
            # on neither.
            #
            # Leading spaces are never emitted: XML collapses them out of a text
            # node, after which the x-list shifts every remaining character left
            # by however many were dropped. So the indent is carried in the
            # starting column, not in the string.
            lead = len(text) - len(text.lstrip(" "))
            core = text.strip()
            if core:
                start = col + lead
                xs = " ".join(
                    f"{PAD_X + (start + k) * CHAR:.2f}" for k in range(len(core))
                )
                style = []
                if colour:
                    style.append(f'fill="{INK[colour][0 if mode == "light" else 1]}"')
                elif dim:
                    style.append(f'fill="{dim_ink}"')
                if bold:
                    style.append('font-weight="600"')
                attrs = (" " + " ".join(style)) if style else ""
                body.append(f'  <text x="{xs}" y="{y}"{attrs}>{esc(core)}</text>')
            col += len(text)

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" \
viewBox="0 0 {w} {h}" role="img" aria-label="arael LmResult report">
  <rect width="{w}" height="{h}" rx="6" fill="{bg}"/>
  <g font-family="{FONT}" font-size="{SIZE}" fill="{fg}" xml:space="preserve">
{chr(10).join(body)}
  </g>
</svg>
"""


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: <ansi capture> | make_report_svg.py <out-prefix>")
    stem = sys.argv[1]
    lines = [ln.rstrip("\n") for ln in sys.stdin.read().rstrip("\n").split("\n")]
    for mode in ("light", "dark"):
        path = f"{stem}-{mode}.svg"
        with open(path, "w") as fh:
            fh.write(render(lines, mode))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
